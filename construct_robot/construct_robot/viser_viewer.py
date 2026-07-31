import argparse
import math
import threading
import time
from pathlib import Path

from geometry_msgs.msg import PoseArray
from moveit_msgs.msg import DisplayTrajectory
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState

from construct_robot.viser_utils import (
    default_urdf_path,
    merge_joint_positions,
    resolve_ros_resource,
    trajectory_time,
    xyzw_to_wxyz,
)
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient

LEFT_TCP = "left_manipulator_ee_point"
RIGHT_TCP = "right_manipulator_ee_point"
VIEW_MODES = ("Live joint states", "Planned trajectory", "Manual joints")


class RosViserBridge(Node):
    """Bridge KIRO ROS visualization topics into a browser-based Viser scene."""

    def __init__(
        self,
        server,
        robot,
        robot_root,
        live_robot,
        live_robot_root,
        fk_model,
        joint_names,
        fixed_frame,
        joint_state_topic,
        weld_pose_topic,
        trajectory_topic,
    ):
        super().__init__("kiro_viser_debug_viewer")
        self._server = server
        self._robot = robot
        self._robot_root = robot_root
        self._live_robot = live_robot
        self._live_robot_root = live_robot_root
        self._fk_model = fk_model
        self._joint_names = tuple(joint_names)
        self._joint_index = {
            name: index for index, name in enumerate(self._joint_names)
        }
        self._fixed_frame = fixed_frame
        self._lock = threading.Lock()
        self._live_configuration = np.zeros(len(self._joint_names))
        self._manual_configuration = self._live_configuration.copy()
        self._plan_configurations = np.empty((0, len(self._joint_names)))
        self._plan_times = np.empty(0)
        self._pending_weld_poses = None
        self._pending_trajectory = None
        self._last_joint_state = None
        self._last_weld_poses = None
        self._last_trajectory = None
        self._last_status_update = 0.0
        self._play_started_at = None
        self._manual_sliders = []
        self._weld_handles = {}
        self._weld_label_handles = []
        self._trajectory_handles = {}

        self._move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action"
        )
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        trajectory_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            JointState,
            joint_state_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseArray,
            weld_pose_topic,
            self._on_weld_poses,
            latched_qos,
        )
        self.create_subscription(
            DisplayTrajectory,
            trajectory_topic,
            self._on_trajectory,
            trajectory_qos,
        )

        self._grid = server.scene.add_grid(
            "/debug/grid",
            width=8.0,
            height=8.0,
            cell_size=0.25,
            section_size=1.0,
            plane="xy",
        )
        self._build_gui(
            joint_state_topic,
            weld_pose_topic,
            trajectory_topic,
        )
        self.get_logger().info(
            f"Viser bridge ready: fixed_frame={fixed_frame}, "
            f"joints={len(self._joint_names)}"
        )

    def _build_gui(self, joint_topic, weld_topic, trajectory_topic):
        self._server.gui.add_markdown(
            "# KIRO ROS Debug Viewer\n"
            "Live robot state, weld 6D poses, and MoveIt trajectories."
        )
        with self._server.gui.add_folder("ROS topic status"):
            self._joint_status = self._server.gui.add_text(
                joint_topic, "waiting", disabled=True
            )
            self._weld_status = self._server.gui.add_text(
                weld_topic, "waiting", disabled=True
            )
            self._trajectory_status = self._server.gui.add_text(
                trajectory_topic, "waiting", disabled=True
            )
            self._frame_status = self._server.gui.add_text(
                "Fixed frame", self._fixed_frame, disabled=True
            )

        with self._server.gui.add_folder("Layers"):
            self._show_robot = self._server.gui.add_checkbox(
                "Selected/planned robot", initial_value=True
            )
            self._show_live_robot = self._server.gui.add_checkbox(
                "Transparent live robot state", initial_value=True
            )
            self._show_grid = self._server.gui.add_checkbox(
                "Ground grid", initial_value=True
            )
            self._show_weld_points = self._server.gui.add_checkbox(
                "Weld points + labels", initial_value=True
            )
            self._show_weld_axes = self._server.gui.add_checkbox(
                "Weld 6D axes", initial_value=True
            )
            self._show_weld_seam = self._server.gui.add_checkbox(
                "Weld seam", initial_value=True
            )
            self._show_tcp_paths = self._server.gui.add_checkbox(
                "Planned TCP paths", initial_value=True
            )
        with self._server.gui.add_folder(
            "Robot commands",
            expand_by_default=False,
        ):
            enable_commands = self._server.gui.add_checkbox(
                "Enable real robot commands",
                initial_value=False,
            )
            go_initial = self._server.gui.add_button(
                "Move RIGHT arm to initial pose",
            )
            command_status = self._server.gui.add_text(
                "Command status",
                "disarmed",
                disabled=True
            )
            @go_initial.on_click
            def _go_initial(_event):
                if not enable_commands.value:
                    command_status.value = "REJECTED: enable commands first"
                    return
                command_status.value = "sending collision-checked MoveIt goal"
                self.move_right_to_initial(command_status)

 
        with self._server.gui.add_folder("Trajectory playback"):
            self._mode = self._server.gui.add_dropdown(
                "Robot source",
                options=VIEW_MODES,
                initial_value=VIEW_MODES[0],
            )
            self._play = self._server.gui.add_checkbox(
                "Play", initial_value=False
            )
            self._auto_preview = self._server.gui.add_checkbox(
                "Auto-preview new RViz/MoveIt plans",
                initial_value=True,
            )
            self._loop = self._server.gui.add_checkbox(
                "Loop", initial_value=True
            )
            self._speed = self._server.gui.add_slider(
                "Speed",
                min=0.1,
                max=3.0,
                step=0.1,
                initial_value=1.0,
            )
            self._progress = self._server.gui.add_slider(
                "Plan progress",
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=0.0,
            )
            self._plan_status = self._server.gui.add_text(
                "Plan", "waiting", disabled=True
            )

        with self._server.gui.add_folder(
            "Manual joint debugging",
            expand_by_default=False,
        ):
            copy_live = self._server.gui.add_button("Copy live joint state")
            for index, name in enumerate(self._joint_names):
                slider = self._server.gui.add_slider(
                    name,
                    min=-math.pi,
                    max=math.pi,
                    step=0.001,
                    initial_value=0.0,
                )

                @slider.on_update
                def _update_manual(_event, joint_index=index, handle=slider):
                    self._manual_configuration[joint_index] = handle.value

                self._manual_sliders.append(slider)

            @copy_live.on_click
            def _copy_live(_event):
                with self._lock:
                    self._manual_configuration = (
                        self._live_configuration.copy()
                    )
                for index, slider in enumerate(self._manual_sliders):
                    slider.value = self._manual_configuration[index]
                self._mode.value = VIEW_MODES[2]

        @self._show_robot.on_update
        def _toggle_robot(_event):
            self._robot_root.visible = self._show_robot.value

        @self._show_live_robot.on_update
        def _toggle_live_robot(_event):
            self._live_robot_root.visible = self._show_live_robot.value

        @self._show_grid.on_update
        def _toggle_grid(_event):
            self._grid.visible = self._show_grid.value

        for checkbox in (
            self._show_weld_points,
            self._show_weld_axes,
            self._show_weld_seam,
            self._show_tcp_paths,
        ):

            @checkbox.on_update
            def _toggle_layers(_event):
                self._apply_layer_visibility()

        @self._mode.on_update
        def _change_mode(_event):
            self._play_started_at = None

        @self._play.on_update
        def _change_play_state(_event):
            self._play_started_at = None

        @self._progress.on_update
        def _seek_plan(_event):
            if not self._play.value:
                self._play_started_at = None

    def _on_joint_state(self, message):
        now = time.monotonic()
        with self._lock:
            self._live_configuration = merge_joint_positions(
                self._live_configuration,
                self._joint_index,
                message.name,
                message.position,
            )
            self._last_joint_state = now

    def _on_weld_poses(self, message):
        with self._lock:
            self._pending_weld_poses = message
            self._last_weld_poses = time.monotonic()

    def _on_trajectory(self, message):
        with self._lock:
            self._pending_trajectory = message
            self._last_trajectory = time.monotonic()

    @staticmethod
    def _remove_handles(handles):
        for handle in handles:
            handle.remove()

    def _render_weld_poses(self, message):
        if message.header.frame_id != self._fixed_frame:
            self._frame_status.value = (
                f"{self._fixed_frame}; rejected weld frame "
                f"{message.header.frame_id}"
            )
            return
        self._frame_status.value = self._fixed_frame
        positions = np.array(
            [
                [pose.position.x, pose.position.y, pose.position.z]
                for pose in message.poses
            ],
            dtype=np.float32,
        )
        orientations = np.array(
            [xyzw_to_wxyz(pose.orientation) for pose in message.poses],
            dtype=np.float32,
        )
        for handle in self._weld_handles.values():
            handle.remove()
        self._remove_handles(self._weld_label_handles)
        self._weld_handles.clear()
        self._weld_label_handles.clear()
        if len(positions) == 0:
            return

        self._weld_handles["points"] = self._server.scene.add_point_cloud(
            "/debug/weld/points",
            points=positions,
            colors=(255, 170, 0),
            point_size=0.025,
            point_shape="circle",
        )
        self._weld_handles["axes"] = self._server.scene.add_batched_axes(
            "/debug/weld/axes",
            batched_wxyzs=orientations,
            batched_positions=positions,
            axes_length=0.08,
            axes_radius=0.004,
        )
        if len(positions) > 1:
            segments = np.stack([positions[:-1], positions[1:]], axis=1)
            self._weld_handles["seam"] = (
                self._server.scene.add_line_segments(
                    "/debug/weld/seam",
                    points=segments,
                    colors=(255, 35, 20),
                    line_width=3.0,
                )
            )
        for index, position in enumerate(positions, start=1):
            self._weld_label_handles.append(
                self._server.scene.add_label(
                    f"/debug/weld/labels/{index}",
                    text=f"W{index}",
                    position=position + np.array([0.0, 0.0, 0.06]),
                )
            )
        self._apply_layer_visibility()
        self.get_logger().info(
            f"Rendered {len(positions)} weld 6D poses in Viser"
        )

    def _extract_plan(self, message):
        if not message.trajectory:
            self._plan_configurations = np.empty(
                (0, len(self._joint_names))
            )
            self._plan_times = np.empty(0)
            self._plan_status.value = "empty trajectory"
            return
        joint_trajectory = message.trajectory[0].joint_trajectory
        base = self._live_configuration.copy()
        configurations = []
        times = []
        for point in joint_trajectory.points:
            base = merge_joint_positions(
                base,
                self._joint_index,
                joint_trajectory.joint_names,
                point.positions,
            )
            configurations.append(base)
            times.append(trajectory_time(point))
        if not configurations:
            self._plan_status.value = "trajectory has no points"
            return
        self._plan_configurations = np.asarray(configurations)
        self._plan_times = np.asarray(times)
        if self._plan_times[-1] <= 0.0:
            self._plan_times = np.arange(len(times), dtype=float) * 0.1
        self._progress.value = 0.0
        self._play.value = False
        self._play_started_at = None
        self._plan_status.value = (
            f"{len(configurations)} points / "
            f"{self._plan_times[-1]:.2f} s"
        )
        self._render_tcp_paths()
        if self._auto_preview.value:
            self._mode.value = VIEW_MODES[1]
            self._play.value = True
        self.get_logger().info(
            f"Loaded {len(configurations)} trajectory points in Viser"
        )

    def _render_tcp_paths(self):
        for handle in self._trajectory_handles.values():
            handle.remove()
        self._trajectory_handles.clear()
        if len(self._plan_configurations) < 2:
            return
        paths = {LEFT_TCP: [], RIGHT_TCP: []}
        for configuration in self._plan_configurations:
            self._fk_model.update_cfg(configuration)
            for link_name in paths:
                transform = self._fk_model.get_transform(
                    link_name,
                    self._fixed_frame,
                )
                paths[link_name].append(transform[:3, 3])
        colors = {
            LEFT_TCP: (40, 200, 255),
            RIGHT_TCP: (255, 70, 210),
        }
        for link_name, points in paths.items():
            points_array = np.asarray(points, dtype=np.float32)
            segments = np.stack(
                [points_array[:-1], points_array[1:]],
                axis=1,
            )
            self._trajectory_handles[link_name] = (
                self._server.scene.add_line_segments(
                    f"/debug/trajectory/{link_name}",
                    points=segments,
                    colors=colors[link_name],
                    line_width=5.0,
                )
            )
        self._apply_layer_visibility()

    def _apply_layer_visibility(self):
        if "points" in self._weld_handles:
            self._weld_handles["points"].visible = (
                self._show_weld_points.value
            )
        for handle in self._weld_label_handles:
            handle.visible = self._show_weld_points.value
        if "axes" in self._weld_handles:
            self._weld_handles["axes"].visible = self._show_weld_axes.value
        if "seam" in self._weld_handles:
            self._weld_handles["seam"].visible = self._show_weld_seam.value
        for handle in self._trajectory_handles.values():
            handle.visible = self._show_tcp_paths.value

    def _planned_configuration(self, now):
        if len(self._plan_configurations) == 0:
            return self._live_configuration
        duration = self._plan_times[-1]
        if duration <= 0.0:
            return self._plan_configurations[-1]
        if self._play.value:
            if self._play_started_at is None:
                elapsed = self._progress.value * duration
                self._play_started_at = now - elapsed / self._speed.value
            elapsed = (now - self._play_started_at) * self._speed.value
            if elapsed >= duration:
                if self._loop.value:
                    elapsed %= duration
                    self._play_started_at = now - elapsed / self._speed.value
                else:
                    elapsed = duration
                    self._play.value = False
            self._progress.value = elapsed / duration
        else:
            self._play_started_at = None
            elapsed = self._progress.value * duration
        return np.array(
            [
                np.interp(elapsed, self._plan_times, column)
                for column in self._plan_configurations.T
            ]
        )

    @staticmethod
    def _status_text(last_received, now, count_description):
        if last_received is None:
            return "waiting"
        age = now - last_received
        state = "live" if age < 1.0 else "stale"
        return f"{state} · {count_description} · {age:.1f}s ago"

    def tick(self):
        now = time.monotonic()
        with self._lock:
            pending_weld = self._pending_weld_poses
            pending_trajectory = self._pending_trajectory
            self._pending_weld_poses = None
            self._pending_trajectory = None
            live_configuration = self._live_configuration.copy()
            joint_time = self._last_joint_state
            weld_time = self._last_weld_poses
            trajectory_time_received = self._last_trajectory

        if pending_weld is not None:
            self._render_weld_poses(pending_weld)
        if pending_trajectory is not None:
            self._extract_plan(pending_trajectory)

        if self._mode.value == VIEW_MODES[0]:
            configuration = live_configuration
        elif self._mode.value == VIEW_MODES[1]:
            configuration = self._planned_configuration(now)
        else:
            configuration = self._manual_configuration
        self._robot.update_cfg(configuration)
        self._live_robot.update_cfg(live_configuration)

        if now - self._last_status_update > 0.2:
            weld_count = (
                len(pending_weld.poses)
                if pending_weld is not None
                else len(self._weld_label_handles)
            )
            plan_count = len(self._plan_configurations)
            self._joint_status.value = self._status_text(
                joint_time,
                now,
                f"{len(self._joint_names)} configured joints",
            )
            self._weld_status.value = self._status_text(
                weld_time,
                now,
                f"{weld_count} poses",
            )
            self._trajectory_status.value = self._status_text(
                trajectory_time_received,
                now,
                f"{plan_count} points",
            )
            self._last_status_update = now

    def move_right_to_intial(self, status_handle):
        joint_names = (
            "right_manipulator_joint1",
            "right_manipulator_joint2",
            "right_manipulator_joint3",
            "right_manipulator_joint4",
            "right_manipulator_joint5",
            "right_manipulator_joint6",
        )
        target = (
            0,
            0,
            0,
            0,
            0,
            0
        )
        if not self._move_group_client.wait_for_server(timeout_sec=2.0):
            status_handle.value = "ERROR: /move_ation unavailable"
            return
        constraints = Constraints()
        for name, position in zip(joint_names, target):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = position
            joint.tolerance_above = 0.005
            joint.tolerance_below = 0.005
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)

        goal = MoveGroup.Goal()
        goal.request.group_name = "right_manipulator"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.05
        goal.request.max_acceleration_scaling_factor = 0.05
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        futurue = self._move_group_client.send_goal_async(goal)

        def goal_response_callback(done):
            handle = done.result()
            if not handle.accepted:
                status_handle.value = "REJECTED by MoveIt"
                return
            status_handle.value = "accepted; planning/executing"
            result_future = handle.get_result_async()

            def finished(result_done):
                error_code = result_done.result().error_code.val
                status_handle.value = f"finished; MoveIt error code {error_code}"
                result_future.add_done_callback(finished)
            result_future.add_done_callback(finished)
        futurue.add_done_callback(goal_response_callback)

def main(args=None):
    parser = argparse.ArgumentParser(
        description="View live KIRO ROS state and debug overlays in Viser"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--fixed-frame", default="World")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--weld-pose-topic", default="/weld_6d_poses")
    parser.add_argument(
        "--trajectory-topic",
        default="/display_planned_path",
    )
    parsed, ros_args = parser.parse_known_args(args=args)

    try:
        import viser
        from viser.extras import ViserUrdf
        import yourdfpy
    except ImportError as error:
        parser.error(
            f"Missing optional Viser dependency: {error}. "
            "Install viser and yourdfpy first."
        )

    urdf_path = parsed.urdf or default_urdf_path()
    if not urdf_path.is_file():
        parser.error(f"URDF does not exist: {urdf_path}")
    model = yourdfpy.URDF.load(
        str(urdf_path),
        filename_handler=resolve_ros_resource,
    )
    live_model = yourdfpy.URDF.load(
        str(urdf_path),
        filename_handler=resolve_ros_resource,
    )
    fk_model = yourdfpy.URDF.load(
        str(urdf_path),
        filename_handler=resolve_ros_resource,
        load_meshes=False,
    )
    server = viser.ViserServer(port=parsed.port, label="KIRO ROS Debug")
    robot_root = server.scene.add_frame(
        "/robot",
        show_axes=True,
        axes_length=0.2,
    )
    robot = ViserUrdf(
        server,
        urdf_or_path=model,
        root_node_name="/robot",
    )
    live_robot_root = server.scene.add_frame(
        "/live_robot",
        show_axes=False,
    )
    live_robot = ViserUrdf(
        server,
        urdf_or_path=live_model,
        root_node_name="/live_robot",
        mesh_color_override=(0.15, 0.85, 0.95, 0.25),
    )
    joint_names = robot.get_actuated_joint_names()

    rclpy.init(args=ros_args)
    bridge = RosViserBridge(
        server=server,
        robot=robot,
        robot_root=robot_root,
        live_robot=live_robot,
        live_robot_root=live_robot_root,
        fk_model=fk_model,
        joint_names=joint_names,
        fixed_frame=parsed.fixed_frame,
        joint_state_topic=parsed.joint_state_topic,
        weld_pose_topic=parsed.weld_pose_topic,
        trajectory_topic=parsed.trajectory_topic,
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(bridge)

    def spin_executor():
        try:
            executor.spin()
        except Exception as error:
            if rclpy.ok():
                bridge.get_logger().error(f"ROS executor stopped: {error}")

    executor_thread = threading.Thread(target=spin_executor, daemon=True)
    executor_thread.start()
    try:
        while rclpy.ok():
            bridge.tick()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=1.0)
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        server.stop()
    return 0
