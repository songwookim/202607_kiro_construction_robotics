"""Non-modal property view; all writes go through SequenceModel validation."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QScrollArea,
    QSpinBox, QVBoxLayout, QWidget,
)


class StepPropertyPanel(QWidget):
    EDITABLE = {"seconds", "duration", "parallel_slot", "pose_label"}

    def __init__(self, table_model, parent=None):
        super().__init__(parent)
        self.table_model = table_model
        layout = QVBoxLayout(self)
        self.heading = QLabel("Select a sequence step")
        layout.addWidget(self.heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        scroll.setWidget(self.form_host)
        layout.addWidget(scroll)
        self.table_model.selection_changed.connect(self.render)
        self.table_model.state_event.connect(self._on_state_event)
        self.render(None)

    def _on_state_event(self, event):
        if event == "steps":
            self.render(self.table_model.state.selected_index)

    def render(self, index):
        while self.form.rowCount():
            self.form.removeRow(0)
        state = self.table_model.state
        if index is None or not 0 <= index < len(state.steps):
            self.heading.setText("Select a sequence step")
            return
        step = state.steps[index]
        self.heading.setText(f"Step {index + 1} · {step.get('type', 'unknown')}")
        for key, value in step.items():
            if key in self.EDITABLE and isinstance(value, (str, int, float)):
                widget = self._editor(key, value)
            else:
                widget = QLabel(self._display(value))
                widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.form.addRow(key.replace("_", " "), widget)
        if step.get("type") != "sleep":
            self.form.addRow(QLabel("Only basic fields are editable in this first Qt slice; welding/motion settings remain in Tkinter."))

    @staticmethod
    def _display(value):
        if isinstance(value, (dict, list, tuple)):
            return f"{type(value).__name__} ({len(value)} items; read-only)"
        return str(value)

    def _editor(self, key, value):
        if key == "parallel_slot":
            widget = QSpinBox()
            widget.setRange(1, 999)
            widget.setValue(int(value))
            widget.editingFinished.connect(lambda w=widget, k=key: self._commit(k, w.value()))
        elif key in ("seconds", "duration"):
            widget = QDoubleSpinBox()
            widget.setRange(0.0, 3600.0)
            widget.setDecimals(3)
            widget.setValue(float(value))
            widget.editingFinished.connect(lambda w=widget, k=key: self._commit(k, w.value()))
        else:
            widget = QLineEdit(str(value))
            widget.editingFinished.connect(lambda w=widget, k=key: self._commit(k, w.text()))
        return widget

    def _commit(self, key, value):
        try:
            self.table_model.update_selected_field(key, value)
        except (IndexError, TypeError, ValueError):
            self.render(self.table_model.state.selected_index)
