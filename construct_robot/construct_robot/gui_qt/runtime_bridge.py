"""Optional boundary to an existing, externally supplied execution runtime.

This shell never constructs ROS nodes, Tk widgets, or hardware clients.  A
runtime port must already own the production safety checks and worker threads.
"""

from PySide6.QtCore import QObject, Signal, Slot


class SequenceRuntimeBridge(QObject):
    status_received = Signal(object)
    io_status_received = Signal(object)
    error_received = Signal(str)

    def __init__(self, sequence_state, runtime=None, parent=None):
        super().__init__(parent)
        self.sequence_state = sequence_state
        self.runtime = runtime

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

    @Slot()
    def refresh(self):
        robots = {}
        if self.runtime is not None and callable(getattr(self.runtime, "robot_status", None)):
            try:
                robots = dict(self.runtime.robot_status())
            except Exception as error:
                self.error_received.emit(str(error))
        self.status_received.emit({
            "robots": robots,
            "running": self.sequence_state.running,
            "selected_index": self.sequence_state.selected_index,
            "current_indices": self.sequence_state.current_step_indices,
            "progress": self.sequence_state.group_progress,
            "status": self.sequence_state.status,
        })
        io = {}
        if self.runtime is not None and callable(getattr(self.runtime, "io_status", None)):
            try:
                io = dict(self.runtime.io_status())
            except Exception as error:
                self.error_received.emit(str(error))
        self.io_status_received.emit(io)
