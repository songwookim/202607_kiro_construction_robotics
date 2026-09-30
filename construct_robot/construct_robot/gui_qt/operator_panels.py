"""Offline Qt projections of existing operator models; no equipment ownership."""

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QFrame,
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
from .shell_widgets import CollapsibleSection


class TeachingPanel(QWidget):
    error = Signal(str)
    capture_requested = Signal(str, str)
    load_requested = Signal(str, str, str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.use_runtime_loader = False
        layout = QVBoxLayout(self)
        heading = QLabel("Teaching")
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Select to inspect · Load and Capture are explicit actions · no motion on selection"))
        self.table = QTableWidget(len(TEACHING_POSES), 4)
        self.table.setHorizontalHeaderLabels(("Pose", "Joints", "TCP", "Source"))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)
        self.table.itemSelectionChanged.connect(self._select)
        self.detail = QLabel("Select a named pose to inspect its source and TCP snapshot")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Capture arm"))
        self.planning_group = QComboBox()
        self.planning_group.addItems(("right_manipulator", "left_manipulator"))
        group_row.addWidget(self.planning_group)
        group_row.addStretch()
        layout.addLayout(group_row)
        self.load_button = QPushButton("Load selected pose YAML · no motion")
        self.load_button.clicked.connect(self._choose_file)
        layout.addWidget(self.load_button)
        self.capture_button = QPushButton("Capture current pose · runtime required")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(lambda: self.capture_requested.emit(
            self.state.selected_name or "", self.planning_group.currentText()))
        layout.addWidget(self.capture_button)
        self.refresh()

    def _select(self):
        row = self.table.currentRow()
        if 0 <= row < len(TEACHING_POSES):
            try:
                self.state.select(tuple(TEACHING_POSES)[row])
            except ValueError as error:
                self.error.emit(str(error))
            self._refresh_detail()

    def _refresh_detail(self):
        name = self.state.selected_name
        if name not in TEACHING_POSES:
            self.detail.setText("Select a named pose to inspect its source and TCP snapshot")
            return
        snapshot = self.state.poses.get(name)
        if not isinstance(snapshot, (tuple, list)):
            self.detail.setText(f"{TEACHING_POSES[name]} · Not captured")
            return
        group = snapshot[0] if snapshot else "—"
        joints = len(snapshot[1]) if len(snapshot) > 1 and snapshot[1] else 0
        tcp = snapshot[3] if len(snapshot) > 3 else None
        source = self.state.provenance.get(name)
        if isinstance(source, dict):
            source = source.get("source", source)
        self.detail.setText(
            f"{TEACHING_POSES[name]} · {group} · {joints} joint(s) · "
            f"TCP {'available' if tcp is not None else 'empty'} · Source: {source or '—'}")

    def load_selected_file(self, path):
        name = self.state.selected_name
        if name not in TEACHING_POSES:
            raise ValueError("Select a named teaching pose")
        if self.use_runtime_loader:
            self.load_requested.emit(name, self.planning_group.currentText(), str(path))
            return None
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
                for col, value in enumerate((("● " if joints or tcp else "○ ") + label, "Available" if joints else "—",
                                             "Available" if tcp else "—", source)):
                    self.table.setItem(row, col, QTableWidgetItem(value))
            selected = list(TEACHING_POSES).index(self.state.selected_name) if self.state.selected_name in TEACHING_POSES else -1
            if selected >= 0:
                self.table.selectRow(selected)
        self._refresh_detail()


class MultiPassPanel(QWidget):
    error = Signal(str)
    references_requested = Signal(str)
    begin_requested = Signal(int)
    start_capture_requested = Signal()
    goal_capture_requested = Signal()
    load_requested = Signal(int)
    stop_requested = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.start_capture_available = False
        self.goal_capture_available = False
        self.load_available = False
        self.stop_available = False
        self.references_available = False
        layout = QVBoxLayout(self)
        heading = QLabel("Multi-pass correction")
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        layout.addWidget(QLabel("1G pass registration · select a pass, then use the existing guarded runtime workflow"))
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        folder_row.addWidget(self.folder_edit)
        self.references_button = QPushButton("Load 4 references")
        self.references_button.setEnabled(False)
        self.references_button.clicked.connect(
            lambda: self.references_requested.emit(self.folder_edit.text().strip()))
        folder_row.addWidget(self.references_button)
        layout.addLayout(folder_row)
        selector = QHBoxLayout()
        selector.addWidget(QLabel("SELECT PASS"))
        self.pass_combo = QComboBox(self)
        self.pass_combo.addItems([f"Pass {number}" for number in range(1, 5)])
        self.pass_combo.currentIndexChanged.connect(self._select)
        self.pass_combo.hide()  # Kept as the existing selection adapter for callers.
        self.pass_buttons = {}
        for number in range(1, 5):
            button = QPushButton(str(number))
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, n=number: self.pass_combo.setCurrentIndex(n - 1))
            selector.addWidget(button)
            self.pass_buttons[number] = button
        selector.addStretch()
        layout.addLayout(selector)
        self.folder_label = QLabel()
        layout.addWidget(self.folder_label)
        self.table = QTableWidget(4, 4)
        self.table.setHorizontalHeaderLabels(("Pass", "Source", "Corrected START", "Corrected GOAL"))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setColumnWidth(0, 90)
        self.table.setColumnWidth(1, 130)
        self.table.setColumnWidth(2, 220)
        layout.addWidget(self.table)
        layout.addWidget(QLabel("REGISTRATION  ·  current selected pass"))
        self.registration_label = QLabel()
        self.status_label = QLabel()
        self.history_label = QLabel()
        for label in (self.registration_label, self.status_label, self.history_label):
            label.setWordWrap(True)
            layout.addWidget(label)
        actions = QHBoxLayout()
        self.correct_button = QPushButton("Begin physical correction · Tk confirmation")
        self.correct_button.setEnabled(False)
        self.correct_button.clicked.connect(lambda: self.begin_requested.emit(self.state.selected_pass))
        actions.addWidget(self.correct_button)
        self.load_button = QPushButton("Load selected corrected pass · runtime required")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(lambda: self.load_requested.emit(self.state.selected_pass))
        actions.addWidget(self.load_button)
        layout.addLayout(actions)
        capture_row = QHBoxLayout()
        self.start_capture_button = QPushButton("Capture START (I path)")
        self.goal_capture_button = QPushButton("Capture GOAL (J path)")
        for button, signal in ((self.start_capture_button, self.start_capture_requested),
                               (self.goal_capture_button, self.goal_capture_requested)):
            button.setEnabled(False)
            button.clicked.connect(signal.emit)
            capture_row.addWidget(button)
        layout.addLayout(capture_row)
        self.stop_button = QPushButton("STOP multi-pass correction")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        layout.addWidget(self.stop_button)
        layout.addWidget(QLabel("I/J capture delegates to the existing Tk keyboard-teaching guards; the Tk host remains visible"))
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
        self.references_button.setEnabled(self.references_available and not bool(self.state.registration))
        if self.state.loaded_folder is not None and not self.folder_edit.text():
            self.folder_edit.setText(str(self.state.loaded_folder))
        with QSignalBlocker(self.pass_combo):
            self.pass_combo.setCurrentIndex(self.state.selected_pass - 1)
        for number, button in self.pass_buttons.items():
            with QSignalBlocker(button):
                button.setChecked(number == self.state.selected_pass)
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
        phase = registration.get("phase")
        self.start_capture_button.setEnabled(
            self.start_capture_available and phase == "waiting_start_capture")
        self.goal_capture_button.setEnabled(
            self.goal_capture_available and phase == "waiting_goal_capture")
        self.stop_button.setEnabled(self.stop_available and bool(registration))
        self.load_button.setEnabled(
            self.load_available and bool(self.state.corrected.get(self.state.selected_pass)))
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
    apply_requested = Signal(object)
    reload_requested = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.editors = {}
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body = QWidget()
        sections = QVBoxLayout(body)
        sections.setSpacing(8)

        def section(title, expanded=False):
            panel = CollapsibleSection(title, expanded)
            form = QFormLayout()
            form.setSpacing(6)
            panel.content_layout.addLayout(form)
            sections.addWidget(panel)
            return form

        recipe = section("Weld recipe", True)
        self._number(recipe, "Current [A]", "current_a", 30, 400, 1)
        self._number(recipe, "Voltage [V]", "voltage_tenths", 3, 80, 0.1, scale=10)
        self._choice(recipe, "Wire material", "material", tuple(MATERIAL_CODES))
        self._choice(recipe, "Wire diameter [mm]", "diameter_mm", tuple(map(str, DIAMETER_CODES)))
        self._choice(recipe, "Mode", "mode", tuple(MODE_CODES))
        self._choice(recipe, "Shielding gas", "gas", tuple(GAS_CODES))
        self._boolean(recipe, "Synergic", "synergic")
        self._number(recipe, "Synergic correction", "correction", -100, 100, 1)

        motion = section("Motion", True)
        self._number(motion, "Travel TCP [mm/s]", "weld_tcp_speed_mm_s", 0.1, 100, 0.1, motion=True)
        weave = section("Weave")
        self._boolean(weave, "Weave enabled", "weld_weave_enabled", motion=True)
        self._choice(weave, "Weave pattern", "weld_weave_pattern", ("sine", "crescent", "circle"), motion=True)
        self._choice(weave, "Weave axis", "weld_weave_axis", ("tool_x", "tool_y"), motion=True)
        self._number(weave, "Amplitude ±A [mm]", "weld_weave_amplitude_mm", 0.1, 50, 0.1, motion=True)
        self._number(weave, "Pitch [mm/cycle]", "weld_weave_pitch_mm", 0.1, 100, 0.1, motion=True)
        self._number(weave, "Left dwell [s]", "weld_weave_left_dwell_s", 0, 10, 0.01, motion=True)
        self._number(weave, "Right dwell [s]", "weld_weave_right_dwell_s", 0, 10, 0.01, motion=True)

        start = section("Start · Hot Start")
        self._boolean(start, "Native Hot Start enabled", "hot_start_enabled")
        self._number(start, "Native Hot Start boost [%]", "hot_start_percent", 0, 100, 1)
        self._number(start, "Native hold adjustment", "hot_start_hold_adjustment", -15, 15, 1)
        self._boolean(start, "Custom Hot Start enabled", "custom_hot_start_enabled")
        self._number(start, "Custom Hot Start boost [%]", "custom_hot_start_percent", 0, 100, 1)
        self._number(start, "Custom Hot Start hold [s]", "custom_hot_start_hold_s", 0.01, 5, 0.01)

        end = section("End · Crater")
        self._boolean(end, "Software crater enabled", "software_crater_enabled")
        self._number(end, "Software crater ratio [%]", "software_crater_ratio_percent", 20, 40, 1)
        self._number(end, "Software crater voltage [V]", "software_crater_voltage_v", 10, 40, 0.1)
        self._number(end, "Software crater hold [s]", "software_crater_hold_s", 0.01, 5, 0.01)
        self._boolean(end, "Native crater expected (observation)", "expect_native_crater")
        self._number(end, "Panel crater current ref [A]", "crater_panel_current_ref_a", 0, 600, 1)
        self._number(end, "Panel crater voltage ref [V]", "crater_panel_voltage_ref_v", 3, 80, 0.1)
        self._number(end, "Panel crater time ref [s]", "crater_panel_time_ref_s", 0, 30, 0.1)
        logging = section("Logging")
        self._number(logging, "Wire consumable allowance [mm]", "wire_consumable_alpha_mm", -1000, 1000, 1)
        sections.addStretch()
        scroll.setWidget(body)
        layout = QVBoxLayout(self)
        heading = QLabel("Welding configuration")
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Edit Builder defaults only · no Hi-COMM command is sent"))
        layout.addWidget(scroll)
        self.apply_button = QPushButton("Apply to future production Builder defaults")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.state))
        actions = QHBoxLayout()
        actions.addWidget(self.apply_button)
        self.reload_button = QPushButton("Reload current production Builder defaults")
        self.reload_button.setEnabled(False)
        self.reload_button.clicked.connect(self.reload_requested.emit)
        actions.addWidget(self.reload_button)
        layout.addLayout(actions)
        layout.addWidget(QLabel("Existing Sequence Builder rows keep their snapshotted settings; rebuild to apply changes"))
        self.refresh()

    def _number(self, form, label, key, minimum, maximum, step, motion=False, scale=1):
        editor = QDoubleSpinBox()
        editor.setMaximumWidth(220)
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
        editor.setMaximumWidth(220)
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
    plan_requested = Signal()
    execute_requested = Signal()

    def __init__(self, state: CleanerTeachingState, sequence_state: SequenceModel,
                 load_pose=None, parent=None):
        super().__init__(parent)
        self.state, self.sequence_state, self.load_pose = state, sequence_state, load_pose
        layout = QVBoxLayout(self)
        heading = QLabel("Torch Cleaner")
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Taught positions → ordered moves / DO pulses → shared Sequence Builder"))
        self.folder_edit = QLineEdit(str(state.folder))
        self.folder_edit.editingFinished.connect(self._folder_changed)
        layout.addWidget(self.folder_edit)
        self.positions = QListWidget()
        self.order = QListWidget()
        columns = QHBoxLayout()
        positions_column = QVBoxLayout()
        positions_column.addWidget(QLabel("TAUGHT POSITIONS"))
        positions_column.addWidget(self.positions)
        order_column = QVBoxLayout()
        order_column.addWidget(QLabel("ORDERED MOVES + OUTPUTS"))
        order_column.addWidget(self.order)
        columns.addLayout(positions_column, 1)
        columns.addLayout(order_column, 2)
        layout.addLayout(columns, 1)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Travel speed %"))
        self.speed = QDoubleSpinBox()
        self.speed.setRange(1, 100)
        self.speed.setValue(20)
        speed_row.addWidget(self.speed)
        speed_row.addStretch()
        layout.addLayout(speed_row)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.preview_label = QLabel("Preview: validate the order before building")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        preview_button = QPushButton("Validate / Preview")
        preview_button.clicked.connect(self._preview_clicked)
        self.build_button = QPushButton("Build → Sequence Builder")
        self.build_button.clicked.connect(self._build_clicked)
        build_row = QHBoxLayout()
        build_row.addWidget(preview_button)
        build_row.addWidget(self.build_button)
        layout.addLayout(build_row)
        actions = QHBoxLayout()
        self.plan_button = QPushButton("Plan cleaner rows")
        self.execute_button = QPushButton("Execute cleaner rows")
        for button, signal in ((self.plan_button, self.plan_requested),
                               (self.execute_button, self.execute_requested)):
            button.setEnabled(False)
            button.clicked.connect(signal.emit)
            actions.addWidget(button)
        layout.addLayout(actions)
        layout.addWidget(QLabel("DO5/6/7 execute only through the existing production Sequence runner; no direct output controls"))
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
    plan_requested = Signal()
    execute_requested = Signal()

    def __init__(self, sequence_state: SequenceModel, order_state=None,
                 folder=None, load_pose=None, parent=None):
        super().__init__(parent)
        self.sequence_state = sequence_state
        self.order_state = order_state if order_state is not None else TaskOrderState()
        self.load_pose = load_pose
        layout = QVBoxLayout(self)
        heading = QLabel("Task Library")
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Saved tasks · teaching poses · visit order · Builder generation"))
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
        pose_column = QVBoxLayout()
        pose_column.addWidget(QLabel("STORED TEACHING POSES"))
        pose_column.addWidget(self.poses)
        order_column = QVBoxLayout()
        order_column.addWidget(QLabel("VISIT ORDER"))
        order_column.addWidget(self.order)
        lists.addLayout(pose_column)
        lists.addLayout(order_column)
        layout.addLayout(lists, 1)
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
        execution = QHBoxLayout()
        self.plan_button = QPushButton("Plan Builder")
        self.execute_button = QPushButton("Execute Builder")
        for button, signal in ((self.plan_button, self.plan_requested),
                               (self.execute_button, self.execute_requested)):
            button.setEnabled(False)
            button.clicked.connect(signal.emit)
            execution.addWidget(button)
        layout.addLayout(execution)
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
        layout = QVBoxLayout(self)
        heading = QLabel("Touch / IO")
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Read-only equipment state · unknown values are not inferred"))
        primary = QFrame()
        primary.setObjectName("PanelCard")
        grid = QGridLayout(primary)
        self.labels = {}
        for row, (key, title) in enumerate(self.FIELDS[:-1]):
            grid.addWidget(QLabel(title), row, 0)
            label = QLabel("Unknown / runtime not attached")
            grid.addWidget(label, row, 1)
            self.labels[key] = label
        layout.addWidget(primary)
        secondary = CollapsibleSection("Control-box IO · details")
        detail = QLabel("Unknown / runtime not attached")
        secondary.content_layout.addWidget(detail)
        self.labels["control_box_io"] = detail
        layout.addWidget(secondary)
        layout.addWidget(QLabel("No touch enable, digital output, or ARC controls on this page"))
        layout.addStretch()

    @Slot(object)
    def set_snapshot(self, snapshot):
        snapshot = snapshot or {}
        for key, _title in self.FIELDS:
            value = snapshot.get(key)
            self.labels[key].setText("Unknown / runtime not attached" if value is None else str(value))
