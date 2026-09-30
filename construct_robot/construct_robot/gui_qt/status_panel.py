"""Persistent execution bar projecting the existing sequence/runtime state."""

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout


class StatusPanel(QFrame):
    def __init__(self, sequence_state=None, parent=None):
        super().__init__(parent)
        self.sequence_state = sequence_state
        self.setObjectName("ExecutionBar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(4)
        main_row = QHBoxLayout()
        self.labels = {key: QLabel("—", self) for key in (
            "left", "right", "sequence", "selected", "current", "progress",
            "active_motion", "welding", "runtime_error", "error")}
        # Compatibility/read-only projections not shown in the compact bar.
        # Keeping them hidden avoids un-managed child labels painting over it.
        for key in ("left", "right", "selected", "active_motion", "welding", "runtime_error"):
            self.labels[key].hide()
        main_row.addWidget(self.labels["sequence"])
        main_row.addWidget(QLabel("STEP"))
        main_row.addWidget(self.labels["current"])
        self.total_label = QLabel("/ —")
        main_row.addWidget(self.total_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(145)
        self.progress_bar.setTextVisible(False)
        main_row.addWidget(self.progress_bar)
        main_row.addWidget(self.labels["progress"])
        main_row.addStretch(1)
        self.controls_host = QHBoxLayout()
        main_row.addLayout(self.controls_host)
        layout.addLayout(main_row)
        self.labels["error"].setWordWrap(True)
        layout.addWidget(self.labels["error"])
        self.set_snapshot({})
        self.set_error("")

    def set_execution_controls(self, controls):
        self.controls_host.addLayout(controls)

    def set_snapshot(self, snapshot):
        snapshot = snapshot or {}
        robots = snapshot.get("robots", {})
        for arm in ("left", "right"):
            state = robots.get(arm)
            self.labels[arm].setText(
                "Unknown / runtime not attached" if state is None else
                ("Connected" if state else "Disconnected")
            )
        self.labels["sequence"].setText(
            ("Running" if snapshot.get("running") else "Idle") +
            f" · {snapshot.get('status', 'idle')}"
        )
        selected = snapshot.get("selected_index")
        self.labels["selected"].setText("—" if selected is None else str(selected + 1))
        current = snapshot.get("current_indices", ())
        self.labels["current"].setText(", ".join(str(index + 1) for index in current) or "—")
        self.total_label.setText(
            f"/ {len(self.sequence_state.steps)}" if self.sequence_state is not None else "/ —")
        group, total = snapshot.get("progress", (0, 0))
        self.labels["progress"].setText(f"{group}/{total}" if total else "—")
        self.progress_bar.setRange(0, max(1, int(total)))
        self.progress_bar.setValue(min(max(0, int(group)), max(1, int(total))))
        active = snapshot.get("active_motion")
        self.labels["active_motion"].setText(
            "Unknown" if active is None else "Active" if active else "Idle")
        self.labels["welding"].setText(str(snapshot.get("welding") or "Unknown"))
        self.labels["runtime_error"].setText(str(snapshot.get("error") or "—"))
        if snapshot.get("error"):
            self.set_error(str(snapshot["error"]))

    def set_error(self, message):
        self.labels["error"].setText(message or "—")
        self.labels["error"].setVisible(bool(message))
