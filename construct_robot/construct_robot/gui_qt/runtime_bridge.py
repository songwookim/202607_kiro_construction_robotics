"""Optional boundary to an existing, externally supplied execution runtime.

This shell never constructs ROS nodes, Tk widgets, or hardware clients.  A
runtime port must already own the production safety checks and worker threads.
"""

from PySide6.QtCore import QObject, Signal, Slot


class SequenceRuntimeBridge(QObject):
    status_received = Signal(object)
    io_status_received = Signal(object)
    error_received = Signal(str)
    runtime_event = Signal()

    def __init__(self, sequence_state, runtime=None, parent=None,
                 teaching_state=None, multipass_state=None):
        super().__init__(parent)
        self.sequence_state = sequence_state
        self.teaching_state = teaching_state
        self.multipass_state = multipass_state
        self.runtime = runtime
        self.runtime_event.connect(self.refresh)
        subscribe = getattr(runtime, "subscribe_status", None)
        self._unsubscribe = subscribe(self.runtime_event.emit) if callable(subscribe) else None

    @property
    def available(self):
        # Never enable hardware buttons for a runtime backed by a different
        # Builder.  The previewed/edited rows must be the rows it executes.
        return self.runtime is not None and getattr(self.runtime, "sequence_state", None) is self.sequence_state and all(
            callable(getattr(self.runtime, name, None))
            for name in ("plan_sequence", "execute_sequence", "stop_sequence")
        )

    def request(self, action):
        if not self.available:
            self.error_received.emit("A runtime sharing this SequenceModel is not attached; use the existing Tkinter GUI for hardware actions")
            return False
        try:
            {"plan": self.runtime.plan_sequence,
             "execute": self.runtime.execute_sequence,
             "stop": self.runtime.stop_sequence}[action]()
        except Exception as error:
            self.error_received.emit(str(error))
            return False
        return True

    def supports(self, method, model=None):
        if self.runtime is None or not callable(getattr(self.runtime, method, None)):
            return False
        if model == "sequence":
            return getattr(self.runtime, "sequence_state", None) is self.sequence_state
        if model == "teaching":
            return self.teaching_state is not None and getattr(self.runtime, "teaching_state", None) is self.teaching_state
        if model == "multipass":
            return self.multipass_state is not None and getattr(self.runtime, "multipass_state", None) is self.multipass_state
        return True

    def invoke(self, method, *args, model=None):
        allowed = {
            "capture_teaching_pose", "load_teaching_pose",
            "begin_multi_pass_registration",
            "capture_multi_pass_start", "capture_multi_pass_goal",
            "load_multi_pass_references",
            "load_selected_pass", "stop_multi_pass_registration",
            "build_weld_scenario", "apply_weld_configuration",
            "plan_cleaner", "execute_cleaner",
        }
        if method not in allowed or not self.supports(method, model):
            self.error_received.emit(f"Production runtime action unavailable: {method}")
            return False
        try:
            getattr(self.runtime, method)(*args)
        except Exception as error:
            self.error_received.emit(str(error))
            return False
        self.refresh()
        return True

    @Slot()
    def refresh(self):
        robots = {}
        if self.runtime is not None and callable(getattr(self.runtime, "robot_status", None)):
            try:
                robots = dict(self.runtime.robot_status())
            except Exception as error:
                self.error_received.emit(str(error))
        extra = {}
        if self.runtime is not None and callable(getattr(self.runtime, "runtime_status", None)):
            try:
                extra = dict(self.runtime.runtime_status())
            except Exception as error:
                self.error_received.emit(str(error))
        self.status_received.emit({
            "robots": robots,
            "running": self.sequence_state.running,
            "selected_index": self.sequence_state.selected_index,
            "current_indices": self.sequence_state.current_step_indices,
            "progress": self.sequence_state.group_progress,
            "status": self.sequence_state.status,
            **extra,
        })
        io = {}
        if self.runtime is not None and callable(getattr(self.runtime, "io_status", None)):
            try:
                io = dict(self.runtime.io_status())
            except Exception as error:
                self.error_received.emit(str(error))
        self.io_status_received.emit(io)

    def close(self):
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
