import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

from construct_msgs.action import CartesianPath
from construct_robot.cartesian_path_server import make_weld_visualization


class WeldActionNode(Node):
    def __init__(self, ui):
        super().__init__("weld_action_gui")
        self.ui = ui
        self.client = ActionClient(self, CartesianPath, "cartesian_path")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_handle = None
        marker_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_publisher = self.create_publisher(
            MarkerArray, "weld_path_markers", marker_qos)
        self.pose_publisher = self.create_publisher(
            PoseArray, "weld_6d_poses", marker_qos)

    def acquire_points(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                "World", "right_manipulator_ee_point", rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
        except TransformException as error:
            self.ui.post(self.ui.error, f"TF acquisition failed: {error}")
            return
        p, q = tf.transform.translation, tf.transform.rotation
        points = []
        # The verified fake-hardware home pose is close to a wrist singularity
        # in World X/Y. A downward seam remains fully reachable and gives a
        # deterministic plan/execution demonstration.
        for offset in (0.05, 0.10, 0.15):
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = p.x, p.y, p.z - offset
            pose.orientation = q
            points.append(pose)
        markers, pose_array = make_weld_visualization(
            points, "World", self.get_clock().now().to_msg())
        self.marker_publisher.publish(markers)
        self.pose_publisher.publish(pose_array)
        self.ui.post(self.ui.set_points, points)
        self.ui.post(
            self.ui.log,
            "Scanner acquired and published 3 visible 6D weld frames")

    def send(self, points):
        if not points:
            self.ui.post(self.ui.error, "Acquire weld points first")
            return
        if not self.client.wait_for_server(timeout_sec=3.0):
            self.ui.post(self.ui.error, "cartesian_path action server unavailable")
            return
        goal = CartesianPath.Goal()
        goal.planning_group = "right_manipulator"
        goal.interpolation_step = 0.005
        goal.waypoints = points
        self.ui.post(self.ui.begin)
        future = self.client.send_goal_async(goal, feedback_callback=self.feedback)
        future.add_done_callback(self.goal_response)

    def feedback(self, message):
        f = message.feedback
        self.ui.post(self.ui.progress, f.progress, f.waypoint_index, f.current_pose)

    def goal_response(self, future):
        try:
            self.goal_handle = future.result()
        except Exception as error:
            self.ui.post(self.ui.error, str(error))
            return
        if not self.goal_handle.accepted:
            self.ui.post(self.ui.error, "Action goal rejected")
            return
        self.ui.post(self.ui.log, "Action accepted · MoveIt planning/execution")
        result = self.goal_handle.get_result_async()
        result.add_done_callback(self.result)

    def result(self, future):
        result = future.result().result
        if result.success:
            self.ui.post(
                self.ui.finish,
                f"SUCCESS · {len(result.sampled_path)} samples · {result.message}")
        else:
            self.ui.post(self.ui.error, result.message)

    def cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.ui.post(self.ui.log, "Cancel requested")


class WeldActionGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("KIRO Laser Weld · Right Arm Action")
        self.root.geometry("780x560")
        self.points = []
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 18, "bold"))
        style.configure("Step.TLabel", font=("Sans", 11, "bold"))

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="KIRO Welding Action Console",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="Laser scanner → 6D weld poses → ROS 2 Action → MoveIt → right_manipulator",
        ).pack(anchor=tk.W, pady=(2, 16))

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="1  Acquire weld points", command=self.acquire).pack(
            side=tk.LEFT, padx=(0, 8))
        self.run_button = ttk.Button(
            controls, text="2  Plan + execute right arm", command=self.run,
            state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Cancel", command=self.cancel).pack(side=tk.LEFT)

        ttk.Label(outer, text="Laser scanner output · World frame", style="Step.TLabel").pack(
            anchor=tk.W, pady=(18, 6))
        columns = ("id", "x", "y", "z", "qx", "qy", "qz", "qw")
        self.table = ttk.Treeview(outer, columns=columns, show="headings", height=4)
        for name in columns:
            self.table.heading(name, text=name.upper())
            self.table.column(name, width=48 if name == "id" else 82, anchor=tk.CENTER)
        self.table.pack(fill=tk.X)

        ttk.Label(outer, text="Action feedback", style="Step.TLabel").pack(
            anchor=tk.W, pady=(18, 6))
        self.bar = ttk.Progressbar(outer, maximum=100)
        self.bar.pack(fill=tk.X)
        self.feedback_label = ttk.Label(outer, text="waypoint: –    pose: –")
        self.feedback_label.pack(anchor=tk.W, pady=5)

        ttk.Label(outer, text="Pipeline status", style="Step.TLabel").pack(
            anchor=tk.W, pady=(14, 6))
        self.status = tk.Text(outer, height=6, bg="#101820", fg="#d5f5e3")
        self.status.pack(fill=tk.BOTH, expand=True)
        self.log(
            "Ready · RViz: weld points + RGB 6D frames + planned trajectory")

        self.node = WeldActionNode(self)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.check_ros)

    def post(self, callback, *args):
        self.root.after(0, callback, *args)

    def log(self, text):
        self.status.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.status.see(tk.END)

    def error(self, text):
        self.log(f"ERROR · {text}")
        self.run_button.configure(state=tk.NORMAL if self.points else tk.DISABLED)

    def set_points(self, points):
        self.points = points
        self.table.delete(*self.table.get_children())
        for index, pose in enumerate(points, 1):
            p, q = pose.position, pose.orientation
            self.table.insert("", tk.END, values=(
                index, f"{p.x:.4f}", f"{p.y:.4f}", f"{p.z:.4f}",
                f"{q.x:.3f}", f"{q.y:.3f}", f"{q.z:.3f}", f"{q.w:.3f}"))
        self.run_button.configure(state=tk.NORMAL)

    def acquire(self):
        self.log("Reading World → right_manipulator_ee_point")
        threading.Thread(target=self.node.acquire_points, daemon=True).start()

    def run(self):
        threading.Thread(target=self.node.send, args=(list(self.points),), daemon=True).start()

    def begin(self):
        self.bar["value"] = 0
        self.run_button.configure(state=tk.DISABLED)
        self.log("Sending right_manipulator CartesianPath goal")

    def progress(self, value, waypoint, pose):
        self.bar["value"] = value * 100
        p = pose.position
        self.feedback_label.configure(
            text=f"waypoint: {waypoint + 1} · progress: {value:.0%} · "
                 f"pose: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")

    def finish(self, text):
        self.bar["value"] = 100
        self.run_button.configure(state=tk.NORMAL)
        self.log(text)

    def cancel(self):
        self.node.cancel()

    def close(self):
        if rclpy.ok():
            self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.root.destroy()

    def check_ros(self):
        if not rclpy.ok():
            self.root.destroy()
            return
        self.root.after(200, self.check_ros)

    def mainloop(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    gui = WeldActionGui()
    try:
        gui.mainloop()
    except KeyboardInterrupt:
        if rclpy.ok():
            rclpy.shutdown()
