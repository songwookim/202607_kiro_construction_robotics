"""PrecisionSensing client and passive RViz inspection on the existing GUI node."""

import threading
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosidl_runtime_py.convert import message_to_ordereddict
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from construct_robot.core.precision_sensing_inspection import finite_values
from construct_robot.io.precision_sensing_results import inspect_document
from construct_robot.nodes.action_call import ActionCall, wait_for_future


class PrecisionSensingTestBridge:
    def __init__(self, node):
        self.node = node
        self._lock = threading.RLock()
        self._busy = False
        self._cancelled = threading.Event()
        self._call = None
        self._transforms = ()
        self._clients = {}
        self.tf = TransformBroadcaster(node)
        self.markers = node.create_publisher(
            MarkerArray, "/hongik_test/path_markers",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.timer = node.create_timer(0.1, self._broadcast)

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def cancel(self):
        with self._lock:
            self._cancelled.set()
            if self._call is not None:
                self._call.cancel()

    def server_status(self, status_service):
        """Query the Hongik server status. Returns (success, message, fields)."""
        client = self.node.create_client(Trigger, status_service)
        try:
            if not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f"Status service unavailable: {status_service}")
            status = wait_for_future(client.call_async(Trigger.Request()), 5.0, "Hongik status")
        finally:
            self.node.destroy_client(client)
        fields = dict(word.split("=", 1) for word in status.message.split() if "=" in word)
        return status.success, status.message, {k: v.rstrip(",;") for k, v in fields.items()}

    def _require_ready(self, status_service, frame=None):
        success, message, fields = self.server_status(status_service)
        # The actions contain no frame_id. Never silently reinterpret input points.
        if (not success or (frame is not None and fields.get("base_frame") != frame)
                or fields.get("ready") != "True" or fields.get("running") != "False"):
            suffix = f"; input frame={frame}" if frame is not None else ""
            raise RuntimeError(f"Server frame/readiness check failed: {message}{suffix}")

    def _session(self, work, on_error, on_finished):
        """Run one Hongik goal workflow; only one at a time, never beside scanner live view."""
        with self._lock:
            if self._busy:
                raise RuntimeError("A Hongik scan is already active")
            live = getattr(self.node, "scanner_live", None)
            if live is not None and live.active:
                raise RuntimeError("Stop scanner live view first: the sensor accepts one client")
            self._busy = True
            self._cancelled.clear()
        try:
            return work()
        except Exception as error:
            if on_error is None:
                raise
            self.node.ui.post(on_error, str(error))
        finally:
            with self._lock:
                self._call = None
                self._busy = False
            if on_finished is not None:
                self.node.ui.post(on_finished)

    def _send(self, description, action_type, action_name, goal, log_topic, on_progress, feedback_text,
              cancel_requested=None):
        """Submit an explicitly requested motion goal and wait for its result (worker thread)."""
        subscription = None
        call = None
        try:
            client = self._clients.get(action_name)
            if client is None:
                client = ActionClient(self.node, action_type, action_name)
                self._clients[action_name] = client
            if not client.wait_for_server(timeout_sec=3.0):
                raise RuntimeError(f"Action server unavailable: {action_name}")
            subscription = self.node.create_subscription(
                String, log_topic,
                lambda msg: self.node.ui.post(on_progress, msg.data), 50,
            )
            def accepted(handle):
                self.node.active_motion_goal = handle
                self.node.ui.post(on_progress, f"{description} accepted · robot moving")
            call = ActionCall(description, on_accepted=accepted)
            with self._lock:
                self._call = call
                if self._cancelled.is_set():
                    call.cancel()
            with self.node._motion_action_lock:
                self.node._motion_action_calls.add(call)
            # Register before checking the sequence STOP flag: a concurrent
            # STOP can then cancel either this registered call or a late goal
            # before submission, including while server preflight was waiting.
            if cancel_requested is not None and cancel_requested():
                call.cancel()
            call.send(client, goal, feedback_text)
            return call.wait(5.0, 900.0)
        finally:
            if call is not None:
                with self.node._motion_action_lock:
                    self.node._motion_action_calls.discard(call)
                if self.node.active_motion_goal is call.handle:
                    self.node.active_motion_goal = None
            if subscription is not None:
                self.node.destroy_subscription(subscription)

    def scan(self, *, a, b, frame, action_name, status_service, log_topic,
             on_progress, on_result, on_error, on_finished, cancel_requested=None):
        """Called by a worker. Only this explicit operation submits a motion goal."""
        def work():
            try:
                from welding_interfaces.action import PrecisionSensing
            except ImportError as error:
                raise RuntimeError("welding_interfaces is unavailable. Build/source KIRO's "
                                   "interface workspace and restart the GUI.") from error
            start, goal_point = finite_values(a, 3), finite_values(b, 3)
            if start == goal_point:
                raise ValueError("A and B must be distinct rough seam points")
            self._require_ready(status_service, frame)
            goal = PrecisionSensing.Goal()
            goal.welding_position = 1
            for pose, values in ((goal.pose_a, start), (goal.pose_b, goal_point)):
                pose.position.x, pose.position.y, pose.position.z = values
                pose.orientation.w = 1.0
            result = self._send(
                "Hongik PrecisionSensing", PrecisionSensing, action_name, goal, log_topic, on_progress,
                lambda msg: self.node.ui.post_latest(
                    "hongik_test_elapsed", on_progress,
                    f"Scanning · elapsed {float(msg.feedback.elapsed_time):.1f} s",
                ),
                cancel_requested=cancel_requested,
            )
            document = dict(message_to_ordereddict(result))
            data_file = str(document.get("data_file_path", ""))
            run_dir = Path(data_file).parent if data_file and Path(data_file).is_file() else None
            inspection = inspect_document(document, frame, run_dir=run_dir)
            if on_result is not None:
                self.node.ui.post(on_result, inspection)
            return inspection
        return self._session(work, on_error, on_finished)

    def post_weld_scan(self, *, run_dir, after_pass, measure_speed_mps, confirm_mode, dry_run,
                       action_name, status_service, log_topic,
                       on_progress, on_confirm, on_result, on_error, on_finished, direction="", tool_orientation="pass",
                       cancel_requested=None):
        """Called by a worker. Raw-data scan of a finished run; moves the robot unless dry_run."""
        def work():
            try:
                from hongik_scan_msgs.action import PostWeldScan
            except ImportError as error:
                raise RuntimeError("hongik_scan_msgs is unavailable. Build/source Hongik's "
                                   "hongik_scan_msgs package and restart the GUI.") from error
            speed = finite_values((measure_speed_mps,), 1)[0]
            if speed < 0.0 or int(after_pass) < 0:
                raise ValueError("Speed and pass must be non-negative")
            if confirm_mode not in ("each", "auto"):
                raise ValueError("Confirm mode must be each or auto")
            if direction not in ("", "S->E", "E->S"):
                raise ValueError("Direction must be Auto, S->E or E->S")
            orientation = str(tool_orientation or "pass").strip().replace("-", "_")
            if orientation not in ("pass", "least_rotation"):
                raise ValueError("Tool orientation must be pass or least_rotation")
            goal = PostWeldScan.Goal()
            if not all(hasattr(goal, field) for field in ("direction", "tool_orientation")):
                raise RuntimeError("New PostWeldScan fields unavailable. Source the rebuilt Hongik overlay and restart the GUI.")
            self._require_ready(status_service)
            goal.direction = direction
            goal.tool_orientation = orientation
            goal.run_dir = str(run_dir).strip()
            goal.after_pass = int(after_pass)
            goal.measure_speed_mps = float(speed)
            goal.confirm_mode = confirm_mode
            goal.dry_run = bool(dry_run)
            def feedback(msg):
                fb = msg.feedback
                self.node.ui.post_latest("hongik_post_weld", on_progress, f"{fb.phase} · {fb.progress_text}")
                self.node.ui.post_latest("hongik_post_weld_confirm", on_confirm, bool(fb.awaiting_confirmation))
            result = self._send("Hongik post-weld scan", PostWeldScan, action_name, goal,
                                log_topic, on_progress, feedback, cancel_requested=cancel_requested)
            self.node.ui.post(on_confirm, False)
            document = dict(message_to_ordereddict(result))
            if on_result is not None:
                self.node.ui.post(on_result, document)
            return document
        return self._session(work, on_error, on_finished)

    def confirm_move(self, confirm_service, accept):
        """Answer the server's CONFIRM prompt (~/confirm_move, std_srvs/SetBool)."""
        client = self.node.create_client(SetBool, confirm_service)
        try:
            if not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f"Confirm service unavailable: {confirm_service}")
            reply = wait_for_future(client.call_async(SetBool.Request(data=bool(accept))), 5.0, "Hongik confirm")
        finally:
            self.node.destroy_client(client)
        return reply.success, reply.message

    def current_tcp(self, frame):
        transform = self.node.tf_buffer.lookup_transform(
            frame, "right_manipulator_ee_point", rclpy.time.Time(),
            timeout=Duration(seconds=1.0),
        )
        now = self.node.get_clock().now()
        stamp = rclpy.time.Time.from_msg(transform.header.stamp, clock_type=now.clock_type)
        age = (now - stamp).nanoseconds * 1e-9
        max_age = float(self.node.get_parameter("teaching_tf_max_age_ms").value) / 1000.0
        if age < -0.1 or age > max_age:
            raise RuntimeError(f"Right TCP TF is stale: age={age:.3f} s")
        t = transform.transform.translation
        return (t.x, t.y, t.z)

    @staticmethod
    def _transform(pose, child):
        msg = TransformStamped()
        msg.header.frame_id = pose.frame
        msg.child_frame_id = child
        t, q = msg.transform.translation, msg.transform.rotation
        t.x, t.y, t.z = pose.position
        q.x, q.y, q.z, q.w = pose.quaternion
        return msg

    def visualize(self, inspection, selected=None):
        """The TFs represent returned poses, never commands or live robot links."""
        transforms = []
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for k in range(1, len(inspection.document.get("paths", [])) + 1):
            rows = [p for p in inspection.poses if p.pass_number == k and p.kind == "wire"]
            if not rows:
                continue
            for label, pose in (("start", rows[0]), ("goal", rows[-1])):
                transforms.append(self._transform(pose, f"hongik_test/pass_{k}_wire_{label}"))
            marker = Marker()
            marker.header.frame_id = inspection.frame
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns, marker.id = "hongik_test", k
            marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.002
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
                ((1.0, 0.3, 0.2), (0.2, 1.0, 0.3), (0.3, 0.5, 1.0), (1.0, 0.8, 0.1))[(k - 1) % 4] + (1.0,)
            )
            marker.points = [Point(x=p.position[0], y=p.position[1], z=p.position[2]) for p in rows]
            markers.markers.append(marker)
        for pose in inspection.poses:
            if pose.kind == "tcp":
                transforms.append(self._transform(pose, f"hongik_test/pass_{pose.pass_number}_tcp_{pose.point.lower()}"))
        if selected is not None:
            transforms.append(self._transform(selected, f"hongik_test/selected_{selected.kind}"))
        with self._lock:
            self._transforms = tuple(transforms)
        self.markers.publish(markers)
        self._broadcast()

    def clear(self):
        with self._lock:
            self._transforms = ()
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.markers.publish(MarkerArray(markers=[marker]))

    def _broadcast(self):
        with self._lock:
            transforms = self._transforms
            stamp = self.node.get_clock().now().to_msg()
            for transform in transforms:
                transform.header.stamp = stamp
            if transforms:
                self.tf.sendTransform(list(transforms))
