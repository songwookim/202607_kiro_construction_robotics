"""Qt view adapter for the existing, UI-independent SequenceModel."""

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal, Slot

from construct_robot.sequence_model import SequenceModel


class SequenceTableModel(QAbstractTableModel):
    state_event = Signal(str)
    selection_changed = Signal(object)
    validation_error = Signal(str)

    HEADERS = ("#", "Type", "Step", "Parallel slot", "Execution")

    def __init__(self, state: SequenceModel, parent=None):
        super().__init__(parent)
        self.state = state
        self.state_event.connect(self._apply_state_event, Qt.QueuedConnection)
        self._unsubscribe = state.subscribe(self.state_event.emit)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.state.steps)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.state.steps):
            return None
        if role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        step = self.state.steps[index.row()]
        if role == Qt.ToolTipRole:
            return f"{step.get('type', 'unknown')} · row {index.row() + 1}"
        column = index.column()
        if column == 0:
            return index.row() + 1
        if column == 1:
            return str(step.get("type", "unknown"))
        if column == 2:
            return str(step.get("pose_label") or step.get("path_kind") or
                       step.get("weld_scenario_stage") or step.get("type", ""))
        if column == 3:
            return "—" if step.get("type") == "sleep" else step.get("parallel_slot", index.row() + 1)
        if column == 4:
            return "CURRENT" if self.state.running and index.row() in self.state.current_step_indices else ""
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    @Slot(str)
    def _apply_state_event(self, event):
        if event == "steps":
            self.beginResetModel()
            self.endResetModel()
            try:
                self.state.validate(require_complete=True)
            except (TypeError, ValueError) as error:
                self.validation_error.emit(str(error))
            else:
                self.validation_error.emit("")
        elif event in ("progress", "execution") and self.rowCount():
            self.dataChanged.emit(self.index(0, 4), self.index(self.rowCount() - 1, 4))
        if event in ("steps", "selection"):
            self.selection_changed.emit(self.state.selected_index)

    def select_row(self, row):
        return self.state.select(row)

    def _can_edit(self):
        if self.state.running:
            self.validation_error.emit("Finish the running sequence before editing Builder rows")
            return False
        return True

    def add_wait(self):
        if not self._can_edit():
            return None
        index = self.state.add({"type": "sleep", "seconds": 1.0,
                                "parallel_slot": len(self.state.steps) + 1})
        self.state.select(index)
        return index

    def delete_selected(self):
        if not self._can_edit():
            return None
        index = self.state.selected_index
        if index is not None:
            self.state.delete(index)

    def delete_all(self):
        if not self._can_edit():
            return None
        self.state.clear()

    def duplicate_selected(self):
        if not self._can_edit():
            return None
        index = self.state.selected_index
        if index is not None:
            self.state.select(self.state.duplicate(index))

    def move_selected(self, offset):
        if not self._can_edit():
            return None
        index = self.state.selected_index
        if index is not None:
            return self.state.move(index, offset)
        return None

    def update_selected_field(self, key, value):
        if not self._can_edit():
            raise ValueError("Finish the running sequence before editing Builder rows")
        index = self.state.selected_index
        if index is None:
            raise ValueError("Select a sequence step")
        try:
            self.state.update_fields(index, {key: value})
        except (IndexError, TypeError, ValueError) as error:
            self.validation_error.emit(str(error))
            raise
        self.validation_error.emit("")
        self.dataChanged.emit(self.index(index, 0), self.index(index, self.columnCount() - 1))

    def close(self):
        self._unsubscribe()
