"""Offline Qt projections of existing operator models; no equipment ownership."""

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from construct_robot.sequence_model import SequenceModel
from construct_robot.hicomm_welder import (
    DIAMETER_CODES, GAS_CODES, MATERIAL_CODES, MODE_CODES,
)
from construct_robot.task_teaching_model import (
    TASK_GROUPS, TEACHING_POSES, TaskOrderState, atomic_yaml,
    build_task_path_steps, encode, load_task_file, safe_task_name,
    task_base_path, validate_task_group, validated_task_speed,
)
from construct_robot.teaching_paths import teaching_config_dir
from construct_robot.teaching_yaml import load_initial_state_yaml
from construct_robot.torch_cleaner_teaching import (
    CleanerTeachingState, build_cleaner_sequence_steps, load_cleaner_order,
)


class TeachingPanel(QWidget):
    error = Signal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Named teaching snapshots · selection is offline; capture/motion requires production runtime"))
        self.table = QTableWidget(len(TEACHING_POSES), 4)
        self.table.setHorizontalHeaderLabels(("Pose", "Joints", "TCP", "Source"))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)
        self.table.itemSelectionChanged.connect(self._select)
        self.load_button = QPushButton("Load selected pose YAML · no motion")
        self.load_button.clicked.connect(self._choose_file)
        layout.addWidget(self.load_button)
        self.capture_button = QPushButton("Capture current pose · runtime required")
        self.capture_button.setEnabled(False)
        layout.addWidget(self.capture_button)
        self.refresh()

    def _select(self):
        row = self.table.currentRow()
        if 0 <= row < len(TEACHING_POSES):
            try:
                self.state.select(tuple(TEACHING_POSES)[row])
            except ValueError as error:
                self.error.emit(str(error))

    def load_selected_file(self, path):
        name = self.state.selected_name
        if name not in TEACHING_POSES:
            raise ValueError("Select a named teaching pose")
        snapshot = load_initial_state_yaml(path)
        self.state.store(name, snapshot, {"source": str(Path(path))})
        self.refresh()
        return snapshot

    def _choose_file(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load selected teaching pose", str(teaching_config_dir()),
            "Teaching YAML (*.yaml *.yml)")
        if path:
            try:
                self.load_selected_file(path)
            except (OSError, TypeError, ValueError) as error:
                self.error.emit(str(error))

    @Slot()
    def refresh(self):
        with QSignalBlocker(self.table):
            for row, (name, label) in enumerate(TEACHING_POSES.items()):
                snapshot = self.state.poses.get(name)
                joints = isinstance(snapshot, (tuple, list)) and len(snapshot) >= 3 and bool(snapshot[1])
                tcp = isinstance(snapshot, (tuple, list)) and len(snapshot) >= 4 and snapshot[3] is not None
                provenance = self.state.provenance.get(name)
                source = (str(provenance.get("source", provenance)) if isinstance(provenance, dict)
                          else str(provenance) if provenance is not None else "—")
                for col, value in enumerate((label, "Available" if joints else "—",
                                             "Available" if tcp else "—", source)):
                    self.table.setItem(row, col, QTableWidgetItem(value))
            selected = list(TEACHING_POSES).index(self.state.selected_name) if self.state.selected_name in TEACHING_POSES else -1
            if selected >= 0:
                self.table.selectRow(selected)


class MultiPassPanel(QWidget):
    error = Signal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        selector = QHBoxLayout()
        selector.addWidget(QLabel("Selected pass"))
        self.pass_combo = QComboBox()
        self.pass_combo.addItems([f"Pass {number}" for number in range(1, 5)])
        self.pass_combo.currentIndexChanged.connect(self._select)
        selector.addWidget(self.pass_combo)
        selector.addStretch()
        layout.addLayout(selector)
        self.folder_label = QLabel()
        layout.addWidget(self.folder_label)
        self.table = QTableWidget(4, 4)
        self.table.setHorizontalHeaderLabels(("Pass", "Source", "Corrected START", "Corrected GOAL"))
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.registration_label = QLabel()
        self.status_label = QLabel()
        self.history_label = QLabel()
        for label in (self.registration_label, self.status_label, self.history_label):
            label.setWordWrap(True)
            layout.addWidget(label)
        self.correct_button = QPushButton("Physical correction · production runtime required")
        self.correct_button.setEnabled(False)
        layout.addWidget(self.correct_button)
        self.refresh()

    def _select(self, index):
        if index >= 0:
            try:
                self.state.select(index + 1)
            except ValueError as error:
                self.error.emit(str(error))
            self.refresh()

    @Slot()
    def refresh(self):
        with QSignalBlocker(self.pass_combo):
            self.pass_combo.setCurrentIndex(self.state.selected_pass - 1)
        self.folder_label.setText(f"Reference folder: {self.state.loaded_folder or '—'} · Corrected folder: {self.state.output_folder or '—'}")
        for row in range(4):
            number = row + 1
            source = self.state.references.get(number)
            corrected = self.state.corrected.get(number)
            values = (f"Pass {number}", "Available" if source is not None else "—",
                      self._pose_text(corrected.get("start") if isinstance(corrected, dict) else None),
                      self._pose_text(corrected.get("goal") if isinstance(corrected, dict) else None))
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
        registration = self.state.registration or {}
        self.registration_label.setText(
            f"Registration: {registration.get('status', registration.get('phase', '—'))} · "
            f"START {'measured' if registration.get('measured_start') is not None else '—'} · "
            f"GOAL {'measured' if registration.get('measured_goal') is not None else '—'}"
        )
        available = bool(self.state.corrected.get(self.state.selected_pass))
        self.status_label.setText(
            f"Status: {self.state.status} · Selected corrected pass: {'available' if available else '—'}"
            " · Apply/load to robot requires production runtime")
        latest = str(self.state.history[-1])[:180] if self.state.history else "—"
        self.history_label.setText(f"Correction history: {len(self.state.history)} event(s) · Latest: {latest}")

    @staticmethod
    def _pose_text(pose):
        if pose is None:
            return "—"
        if hasattr(pose, "position"):
            p = pose.position
            return f"({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) m"
        if isinstance(pose, dict):
            position = pose.get("position_m", pose)
            if all(axis in position for axis in ("x", "y", "z")):
                return "({x:.4f}, {y:.4f}, {z:.4f}) m".format(
                    **{axis: float(position[axis]) for axis in ("x", "y", "z")})
        return "Available"


class WeldingPanel(QWidget):
    error = Signal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.editors = {}
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body = QWidget()
        form = QFormLayout(body)
        self._number(form, "Current [A]", "current_a", 30, 400, 1)
        self._number(form, "Voltage [V]", "voltage_tenths", 3, 80, 0.1, scale=10)
        self._choice(form, "Wire material", "material", tuple(MATERIAL_CODES))
        self._choice(form, "Wire diameter [mm]", "diameter_mm", tuple(map(str, DIAMETER_CODES)))
        self._choice(form, "Mode", "mode", tuple(MODE_CODES))
        self._choice(form, "Shielding gas", "gas", tuple(GAS_CODES))
        self._boolean(form, "Synergic", "synergic")
        self._number(form, "Synergic correction", "correction", -100, 100, 1)
        self._number(form, "Travel TCP [mm/s]", "weld_tcp_speed_mm_s", 0.1, 100, 0.1, motion=True)
        self._boolean(form, "Weave enabled", "weld_weave_enabled", motion=True)
        self._choice(form, "Weave pattern", "weld_weave_pattern", ("sine", "crescent", "circle"), motion=True)
        self._choice(form, "Weave axis", "weld_weave_axis", ("tool_x", "tool_y"), motion=True)
        self._number(form, "Amplitude ±A [mm]", "weld_weave_amplitude_mm", 0.1, 50, 0.1, motion=True)
        self._number(form, "Pitch [mm/cycle]", "weld_weave_pitch_mm", 0.1, 100, 0.1, motion=True)
        self._number(form, "Left dwell [s]", "weld_weave_left_dwell_s", 0, 10, 0.01, motion=True)
        self._number(form, "Right dwell [s]", "weld_weave_right_dwell_s", 0, 10, 0.01, motion=True)
        self._boolean(form, "Native Hot Start enabled", "hot_start_enabled")
        self._number(form, "Native Hot Start boost [%]", "hot_start_percent", 0, 100, 1)
        self._number(form, "Native hold adjustment", "hot_start_hold_adjustment", -15, 15, 1)
        self._boolean(form, "Custom Hot Start enabled", "custom_hot_start_enabled")
        self._number(form, "Custom Hot Start boost [%]", "custom_hot_start_percent", 0, 100, 1)
        self._number(form, "Custom Hot Start hold [s]", "custom_hot_start_hold_s", 0.01, 5, 0.01)
        self._boolean(form, "Software crater enabled", "software_crater_enabled")
        self._number(form, "Software crater ratio [%]", "software_crater_ratio_percent", 20, 40, 1)
        self._number(form, "Software crater voltage [V]", "software_crater_voltage_v", 10, 40, 0.1)
        self._number(form, "Software crater hold [s]", "software_crater_hold_s", 0.01, 5, 0.01)
        self._boolean(form, "Native crater expected (observation)", "expect_native_crater")
        self._number(form, "Panel crater current ref [A]", "crater_panel_current_ref_a", 0, 600, 1)
        self._number(form, "Panel crater voltage ref [V]", "crater_panel_voltage_ref_v", 3, 80, 0.1)
        self._number(form, "Panel crater time ref [s]", "crater_panel_time_ref_s", 0, 30, 0.1)
        self._number(form, "Wire consumable allowance [mm]", "wire_consumable_alpha_mm", -1000, 1000, 1)
        form.addRow(QLabel("Offline draft only · no Hi-COMM setpoint/ARC command is sent"))
        scroll.setWidget(body)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        self.refresh()

    def _number(self, form, label, key, minimum, maximum, step, motion=False, scale=1):
        editor = QDoubleSpinBox()
        editor.setRange(minimum, maximum)
        editor.setSingleStep(step)
        editor.setDecimals(2)
        self.editors[key] = editor
        editor.editingFinished.connect(lambda k=key, m=motion, s=scale, e=editor: self._commit(k, e.value() * s, m))
        form.addRow(label, editor)

    def _boolean(self, form, label, key, motion=False):
        editor = QCheckBox()
        self.editors[key] = editor
        editor.toggled.connect(lambda value, k=key, m=motion: self._commit(k, value, m))
        form.addRow(label, editor)

    def _choice(self, form, label, key, choices, motion=False):
        editor = QComboBox()
        editor.addItems(choices)
        self.editors[key] = editor
        editor.currentTextChanged.connect(lambda value, k=key, m=motion: self._commit(k, value, m))
        form.addRow(label, editor)

    def _commit(self, key, value, motion):
        try:
            if motion:
                self.state.set_motion(key, value)
            else:
                if key == "diameter_mm":
                    value = float(value)
                self.state.set_recipe(key, value)
        except (TypeError, ValueError) as error:
            self.error.emit(str(error))
        self.refresh()

    @Slot()
    def refresh(self):
        for key, editor in self.editors.items():
            value = self.state.motion[key] if key in self.state.motion else self.state.recipe[key]
            with QSignalBlocker(editor):
                if isinstance(editor, QCheckBox):
                    editor.setChecked(bool(value))
                elif isinstance(editor, QComboBox):
                    editor.setCurrentText(str(value))
                else:
                    editor.setValue(float(value) / (10 if key == "voltage_tenths" else 1))


class TorchCleanerPanel(QWidget):
    error = Signal(str)

    def __init__(self, state: CleanerTeachingState, sequence_state: SequenceModel,
                 load_pose=None, parent=None):
        super().__init__(parent)
        self.state, self.sequence_state, self.load_pose = state, sequence_state, load_pose
        layout = QVBoxLayout(self)
        self.folder_edit = QLineEdit(str(state.folder))
        self.folder_edit.editingFinished.connect(self._folder_changed)
        layout.addWidget(self.folder_edit)
        self.positions = QListWidget()
        layout.addWidget(QLabel("Available cleaner teaching positions"))
        layout.addWidget(self.positions)
        self.order = QListWidget()
        layout.addWidget(QLabel("Cleaner order (one shared Builder is generated from this order)"))
        layout.addWidget(self.order)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(1, 100)
        self.speed.setValue(20)
        layout.addWidget(self.speed)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.preview_label = QLabel("Preview: validate the order before building")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        preview_button = QPushButton("Validate / Preview")
        preview_button.clicked.connect(self._preview_clicked)
        layout.addWidget(preview_button)
        self.build_button = QPushButton("Build → Sequence Builder")
        self.build_button.clicked.connect(self._build_clicked)
        layout.addWidget(self.build_button)
        layout.addWidget(QLabel("DO5/6/7 and motion execution are not connected in this Qt migration slice"))
        self.refresh()

    def _folder_changed(self):
        self.state.folder = Path(self.folder_edit.text()).expanduser().resolve()
        self.refresh()

    def refresh(self):
        folder = self.state.folder
        self.positions.clear()
        self.positions.addItems(sorted(path.stem for path in folder.glob("*.yaml") if path.stem != "sequence"))
        try:
            if (folder / "sequence.yaml").is_file():
                self.state.set_order(load_cleaner_order(folder))
            self.order.clear()
            self.order.addItems(self.state.tokens)
            self.status.setText(f"{len(self.state.tokens)} order item(s) · {folder}")
        except (OSError, TypeError, ValueError) as error:
            self.status.setText(str(error))
            self.error.emit(str(error))

    def build(self):
        if self.sequence_state.running:
            raise ValueError("Finish the active sequence first")
        steps = self.preview()
        replacement = self.sequence_state.with_replaced_cleaner(steps)
        SequenceModel(replacement).validate(require_complete=True)
        self.sequence_state.replace(replacement)
        self.status.setText(f"Built {len(steps)} cleaner rows in shared Sequence Builder")
        return steps

    def preview(self):
        def unavailable(_path):
            raise ValueError("TCP teaching YAML loader requires a production-independent pose loader")
        steps = build_cleaner_sequence_steps(self.state.folder, self.state.tokens,
                                             self.speed.value(), self.load_pose or unavailable)
        replacement = self.sequence_state.with_replaced_cleaner(steps)
        SequenceModel(replacement).validate(require_complete=True)
        outputs = [f"DO{step['port']} ({step['duration']:g}s)" for step in steps
                   if step["type"] == "digital_output"]
        self.preview_label.setText(
            f"Preview: {len(steps)} rows · {len(steps) - len(outputs)} taught positions · "
            f"{', '.join(outputs) or 'no outputs'} · no commands sent")
        return steps

    def _preview_clicked(self):
        try:
            self.preview()
        except (OSError, TypeError, ValueError) as error:
            self.preview_label.setText(f"Preview invalid: {error}")
            self.error.emit(str(error))

    def _build_clicked(self):
        try:
            self.build()
        except (OSError, TypeError, ValueError) as error:
            self.status.setText(str(error))
            self.error.emit(str(error))


class TaskLibraryPanel(QWidget):
    error = Signal(str)

    def __init__(self, sequence_state: SequenceModel, order_state=None,
                 folder=None, load_pose=None, parent=None):
        super().__init__(parent)
        self.sequence_state = sequence_state
        self.order_state = order_state if order_state is not None else TaskOrderState()
        self.load_pose = load_pose
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.folder_edit = QLineEdit(str(folder or teaching_config_dir() / "robot_tasks"))
        self.folder_edit.editingFinished.connect(self.refresh)
        form.addRow("Library folder", self.folder_edit)
        self.category = QComboBox()
        self.category.addItems(TASK_GROUPS)
        self.category.setCurrentText("Left · Spray path")
        self.category.currentTextChanged.connect(self._category_changed)
        form.addRow("Category", self.category)
        self.task_name = QLineEdit("task_1")
        form.addRow("Task name", self.task_name)
        self.saved_tasks = QComboBox()
        self.saved_tasks.currentTextChanged.connect(self._select_saved_task)
        form.addRow("Saved tasks", self.saved_tasks)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.01, 50)
        self.speed.setValue(5)
        form.addRow("TCP speed [mm/s]", self.speed)
        layout.addLayout(form)
        lists = QHBoxLayout()
        self.poses = QListWidget()
        self.order = QListWidget()
        lists.addWidget(self.poses)
        lists.addWidget(self.order)
        layout.addWidget(QLabel("Stored poses                                              Visit order"))
        layout.addLayout(lists)
        buttons = QHBoxLayout()
        for label, callback in (("Add →", self._add_clicked), ("Remove", self.remove_selected),
                                ("↑", lambda: self.move_selected(-1)), ("↓", lambda: self.move_selected(1))):
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        actions = QHBoxLayout()
        self.build_button = QPushButton("Build taught path")
        self.build_button.setEnabled(load_pose is not None)
        self.build_button.clicked.connect(self._build_clicked)
        actions.addWidget(self.build_button)
        self.load_button = QPushButton("Load YAML → Builder")
        self.load_button.clicked.connect(self._load_clicked)
        actions.addWidget(self.load_button)
        self.save_button = QPushButton("Save Builder → YAML")
        self.save_button.clicked.connect(self._save_clicked)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        self.status = QLabel("Loading/saving never moves a robot")
        layout.addWidget(self.status)
        self.refresh()

    def base(self):
        return task_base_path(self.folder_edit.text(), self.category.currentText())

    def task_path(self):
        return self.base() / (safe_task_name(self.task_name.text().strip()) + ".yaml")

    @Slot()
    def refresh(self):
        base = self.base()
        tasks = sorted(path.stem for path in base.glob("*.yaml"))
        with QSignalBlocker(self.saved_tasks):
            self.saved_tasks.clear()
            self.saved_tasks.addItems(tasks)
            self.saved_tasks.setCurrentText(self.task_name.text().strip())
        self.poses.clear()
        self.poses.addItems(sorted(path.stem for path in (base / "poses").glob("*.yaml")))
        self._refresh_order()
        self.status.setText(f"{TASK_GROUPS[self.category.currentText()]} · {base}")

    def _select_saved_task(self, name):
        if name:
            self.task_name.setText(name)

    def _category_changed(self, _name):
        self.order_state.replace(())
        self.refresh()

    def _add_clicked(self):
        try:
            self.add_selected_pose()
        except (OSError, TypeError, ValueError) as error:
            self.error.emit(str(error))

    def _refresh_order(self, selection=None):
        self.order.clear()
        self.order.addItems(self.order_state.names)
        if selection is not None and 0 <= selection < self.order.count():
            self.order.setCurrentRow(selection)

    def add_selected_pose(self):
        item = self.poses.currentItem()
        if item is None:
            return
        name = safe_task_name(item.text())
        if self.load_pose is not None:
            group, *_rest = self.load_pose(self.base() / "poses" / (name + ".yaml"))
            if group != TASK_GROUPS[self.category.currentText()]:
                self.error.emit("Pose belongs to another arm")
                return
        self.order_state.add(name)
        self._refresh_order(len(self.order_state.names) - 1)

    def remove_selected(self):
        row = self.order.currentRow()
        if row >= 0:
            self.order_state.remove(row)
            self._refresh_order(min(row, len(self.order_state.names) - 1))

    def move_selected(self, direction):
        row = self.order.currentRow()
        if row >= 0:
            moved = self.order_state.move(row, direction)
            if moved is not None:
                self._refresh_order(moved)

    def build_path(self):
        if self.load_pose is None:
            raise ValueError("A production-independent teaching pose loader is not attached")
        if self.sequence_state.running:
            raise ValueError("Finish the active sequence first")
        group = TASK_GROUPS[self.category.currentText()]
        names = list(self.order_state.names)
        stored = [self.load_pose(self.base() / "poses" / (safe_task_name(name) + ".yaml")) for name in names]
        steps = build_task_path_steps(names, stored, group, validated_task_speed(self.speed.value()))
        validate_task_group(steps, group)
        SequenceModel(steps).validate(require_complete=True)
        self.sequence_state.replace(steps)
        return steps

    def load_task(self, path=None):
        if self.sequence_state.running:
            raise ValueError("Finish the active sequence first")
        document, steps = load_task_file(path or self.task_path(), self.category.currentText())
        SequenceModel(steps).validate(require_complete=True)
        names = [safe_task_name(name) for name in document.get("visit_order", [])]
        self.sequence_state.replace(steps)
        self.order_state.replace(names)
        self._refresh_order()
        self.status.setText(f"Loaded {len(steps)} Builder rows · no motion sent")
        return steps

    def save_task(self, path=None, overwrite=False):
        if self.sequence_state.running:
            raise ValueError("Finish the active sequence first")
        target = Path(path) if path is not None else self.task_path()
        if target.exists() and not overwrite:
            raise FileExistsError(f"Task exists: {target}")
        self.sequence_state.validate(require_complete=True)
        validate_task_group(self.sequence_state.steps, TASK_GROUPS[self.category.currentText()])
        document = {"schema": "robot_task_v1", "category": self.category.currentText(),
                    "visit_order": list(self.order_state.names),
                    "steps": encode(self.sequence_state.steps)}
        atomic_yaml(target, document)
        self.status.setText(f"Saved {target}")
        return target

    def _build_clicked(self):
        try:
            self.build_path()
        except (OSError, TypeError, ValueError) as error:
            self.error.emit(str(error))

    def _load_clicked(self):
        try:
            self.load_task()
        except (KeyError, OSError, TypeError, ValueError) as error:
            self.error.emit(str(error))

    def _save_clicked(self):
        try:
            target = self.task_path()
            overwrite = not target.exists() or QMessageBox.question(
                self, "Replace task", f"Overwrite {target}?"
            ) == QMessageBox.Yes
            if overwrite:
                self.save_task(overwrite=True)
        except (KeyError, OSError, TypeError, ValueError) as error:
            self.error.emit(str(error))


class TouchIoStatusPanel(QWidget):
    """Read-only runtime projection; absent fields remain explicitly unknown."""

    FIELDS = (("fastech_connected", "Fastech connection"),
              ("touch_input", "Touch input (Fastech DI4)"),
              ("touch_enabled", "Touch sensing (DO0)"),
              ("hicomm_connected", "Hi-COMM connection"),
              ("welder_output_state", "Welder output state"),
              ("control_box_io", "Control-box IO"))

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        self.labels = {}
        for row, (key, title) in enumerate(self.FIELDS):
            layout.addWidget(QLabel(title), row, 0)
            label = QLabel("Unknown / runtime not attached")
            layout.addWidget(label, row, 1)
            self.labels[key] = label
        layout.addWidget(QLabel("Read-only · no touch enable, digital output, or ARC controls"), len(self.FIELDS), 0, 1, 2)

    @Slot(object)
    def set_snapshot(self, snapshot):
        snapshot = snapshot or {}
        for key, _title in self.FIELDS:
            value = snapshot.get(key)
            self.labels[key].setText("Unknown / runtime not attached" if value is None else str(value))
