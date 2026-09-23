"""Read-only Qt status projection of existing sequence/runtime state."""

from PySide6.QtWidgets import QGridLayout, QLabel, QWidget


class StatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        self.labels = {}
        for row, (key, title) in enumerate((
            ("left", "LEFT robot"), ("right", "RIGHT robot"),
            ("sequence", "Sequence"), ("selected", "Selected step"),
            ("current", "Current step"), ("progress", "Progress"),
            ("error", "Validation / error"),
        )):
            layout.addWidget(QLabel(title), row, 0)
            label = QLabel("—")
            layout.addWidget(label, row, 1)
            self.labels[key] = label
        self.set_snapshot({})

    def set_snapshot(self, snapshot):
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
        group, total = snapshot.get("progress", (0, 0))
        self.labels["progress"].setText(f"{group}/{total}" if total else "—")

    def set_error(self, message):
        self.labels["error"].setText(message or "—")
