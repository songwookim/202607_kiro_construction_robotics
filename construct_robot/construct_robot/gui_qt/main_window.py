"""PySide6 offline operator shell backed by canonical workflow models."""

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QStackedWidget, QTableView,
    QVBoxLayout, QWidget, QFrame,
)

from construct_robot.sequence_model import SequenceModel
from construct_robot.task_teaching_model import (
    TEACHING_POSES, TaskOrderState, TeachingState, load_task_file, safe_task_name,
)
from construct_robot.multipass import MultiPassState
from construct_robot.teaching_paths import teaching_config_dir
from construct_robot.teaching_yaml import load_initial_state_yaml
from construct_robot.torch_cleaner_teaching import CleanerTeachingState
from construct_robot.weld_config import WeldConfigurationState
from .operator_panels import (
    TeachingPanel, MultiPassPanel, WeldingPanel, TorchCleanerPanel,
    TaskLibraryPanel, TouchIoStatusPanel,
)
from .property_panel import StepPropertyPanel
from .runtime_bridge import SequenceRuntimeBridge
from .sequence_model_qt import SequenceTableModel
from .status_panel import StatusPanel
from .shell_widgets import APP_STYLESHEET, TopStatusBar


class SequenceMainWindow(QMainWindow):
    def __init__(self, sequence_state=None, runtime=None, parent=None, *,
                 teaching_state=None, multipass_state=None, weld_state=None,
                 cleaner_state=None, task_order_state=None, load_pose=None):
        super().__init__(parent)
        self.setWindowTitle("Welding Workflow · Qt Production" if runtime is not None
                            else "Welding Workflow · Qt Preview")
        self.resize(1250, 780)
        self.setStyleSheet(APP_STYLESHEET)
        self.sequence_state = sequence_state if sequence_state is not None else SequenceModel()
        load_pose = load_pose or load_initial_state_yaml
        self.teaching_state = teaching_state if teaching_state is not None else TeachingState(TEACHING_POSES)
        self.multipass_state = multipass_state if multipass_state is not None else MultiPassState()
        if weld_state is None and runtime is not None and callable(getattr(runtime, "weld_configuration", None)):
            recipe, motion = runtime.weld_configuration()
            weld_state = WeldConfigurationState(recipe, motion)
        self.weld_state = weld_state if weld_state is not None else WeldConfigurationState()
        self.cleaner_state = cleaner_state if cleaner_state is not None else CleanerTeachingState(
            teaching_config_dir() / "torch_cleaner_teaching")
        self.task_order_state = task_order_state if task_order_state is not None else TaskOrderState()
        self.table_model = SequenceTableModel(self.sequence_state, self)
        self.runtime_bridge = SequenceRuntimeBridge(
            self.sequence_state, runtime, self,
            teaching_state=self.teaching_state, multipass_state=self.multipass_state)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.top_status = TopStatusBar(production=runtime is not None)
        layout.addWidget(self.top_status)
        workspace = QHBoxLayout()
        workspace.setSpacing(8)
        layout.addLayout(workspace, 1)
        navigation = QFrame()
        navigation.setObjectName("Navigation")
        navigation.setFixedWidth(166)
        nav_layout = QVBoxLayout(navigation)
        nav_layout.setContentsMargins(5, 10, 5, 10)
        nav_layout.setSpacing(2)
        self.pages = QStackedWidget()
        self.nav_buttons = {}
        workspace.addWidget(navigation)
        workspace.addWidget(self.pages, 1)
        sequence_page = QWidget()
        self._add_page(nav_layout, "WORKFLOW", "Sequence", sequence_page, first=True)
        self._build_sequence_page(sequence_page)
        self.teaching_panel = TeachingPanel(self.teaching_state)
        if runtime is not None and callable(getattr(runtime, "selected_planning_group", None)):
            self.teaching_panel.planning_group.setCurrentText(runtime.selected_planning_group())
        self.multipass_panel = MultiPassPanel(self.multipass_state)
        if runtime is not None and callable(getattr(runtime, "multi_pass_folder", None)):
            self.multipass_panel.folder_edit.setText(str(runtime.multi_pass_folder()))
        self.welding_panel = WeldingPanel(self.weld_state)
        self.cleaner_panel = TorchCleanerPanel(self.cleaner_state, self.sequence_state, load_pose)
        self.task_panel = TaskLibraryPanel(self.sequence_state, self.task_order_state,
                                           load_pose=load_pose)
        self.io_panel = TouchIoStatusPanel()
        for group, title, panel in ((None, "Teaching", self.teaching_panel),
                                    (None, "Multi-pass", self.multipass_panel),
                                    (None, "Welding", self.welding_panel),
                                    ("TASKS", "Torch Cleaner", self.cleaner_panel),
                                    (None, "Task Library", self.task_panel),
                                    ("SYSTEM", "Touch / IO", self.io_panel)):
            self._add_page(nav_layout, group, title, panel)
        nav_layout.addStretch()

        self.status_panel = StatusPanel(self.sequence_state)
        self.status_panel.set_execution_controls(self._build_execution_controls())
        layout.addWidget(self.status_panel)
        self.setCentralWidget(root)
        self.table_model.validation_error.connect(self.status_panel.set_error)
        self.runtime_bridge.status_received.connect(self.status_panel.set_snapshot)
        self.runtime_bridge.status_received.connect(self.top_status.set_snapshot)
        self.runtime_bridge.io_status_received.connect(self.io_panel.set_snapshot)
        self.runtime_bridge.io_status_received.connect(self.top_status.set_io_snapshot)
        self.runtime_bridge.error_received.connect(self.status_panel.set_error)
        self.table_model.selection_changed.connect(self._select_from_state)
        for panel in (self.teaching_panel, self.multipass_panel, self.welding_panel,
                      self.cleaner_panel, self.task_panel):
            panel.error.connect(self.status_panel.set_error)
        self._connect_operator_actions()

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(200)
        self.status_timer.timeout.connect(self.runtime_bridge.refresh)
        self.status_timer.timeout.connect(self.teaching_panel.refresh)
        self.status_timer.timeout.connect(self.multipass_panel.refresh)
        self.status_timer.start()
        self.runtime_bridge.refresh()

    def _add_page(self, nav_layout, group, title, page, first=False):
        if group:
            heading = QLabel(group)
            heading.setObjectName("Muted")
            nav_layout.addWidget(heading)
        index = self.pages.addWidget(page)
        button = QPushButton(title)
        button.setObjectName("NavButton")
        button.setProperty("pageIndex", index)
        button.clicked.connect(lambda _checked=False, i=index: self.select_page(i))
        nav_layout.addWidget(button)
        self.nav_buttons[title] = button
        if first:
            self.select_page(index)

    def select_page(self, index):
        self.pages.setCurrentIndex(index)
        for button in self.nav_buttons.values():
            selected = button.property("pageIndex") == index
            button.setProperty("selected", "true" if selected else "false")
            button.style().unpolish(button)
            button.style().polish(button)

    def _connect_operator_actions(self):
        bridge = self.runtime_bridge
        self.teaching_panel.capture_button.setEnabled(
            bridge.supports("capture_teaching_pose", "teaching"))
        self.teaching_panel.use_runtime_loader = bridge.supports("load_teaching_pose", "teaching")
        self.teaching_panel.load_requested.connect(
            lambda name, group, path: bridge.invoke("load_teaching_pose", name, group, path,
                                                    model="teaching"))
        if self.teaching_panel.capture_button.isEnabled():
            self.teaching_panel.capture_button.setText("Capture measured pose · existing production path")
        self.teaching_panel.capture_requested.connect(
            lambda name, group: bridge.invoke("capture_teaching_pose", name, group, model="teaching"))
        self.multipass_panel.correct_button.setEnabled(
            bridge.supports("begin_multi_pass_registration", "multipass"))
        if self.multipass_panel.correct_button.isEnabled():
            self.multipass_panel.correct_button.setText("Begin physical correction · Tk confirmation")
        self.multipass_panel.load_available = bridge.supports("load_selected_pass", "multipass")
        self.multipass_panel.references_available = bridge.supports("load_multi_pass_references", "multipass")
        self.multipass_panel.start_capture_available = bridge.supports("capture_multi_pass_start", "multipass")
        self.multipass_panel.goal_capture_available = bridge.supports("capture_multi_pass_goal", "multipass")
        self.multipass_panel.stop_available = bridge.supports("stop_multi_pass_registration", "multipass")
        self.multipass_panel.refresh()
        self.multipass_panel.begin_requested.connect(
            lambda number: bridge.invoke("begin_multi_pass_registration", number, model="multipass"))
        self.multipass_panel.references_requested.connect(
            lambda folder: bridge.invoke("load_multi_pass_references", folder, model="multipass"))
        self.multipass_panel.start_capture_requested.connect(
            lambda: bridge.invoke("capture_multi_pass_start", model="multipass"))
        self.multipass_panel.goal_capture_requested.connect(
            lambda: bridge.invoke("capture_multi_pass_goal", model="multipass"))
        self.multipass_panel.load_requested.connect(
            lambda number: bridge.invoke("load_selected_pass", number, model="multipass"))
        self.multipass_panel.stop_requested.connect(
            lambda: bridge.invoke("stop_multi_pass_registration", model="multipass"))
        self.welding_panel.apply_button.setEnabled(
            bridge.supports("apply_weld_configuration", "sequence"))
        self.welding_panel.reload_button.setEnabled(
            self.runtime_bridge.runtime is not None
            and callable(getattr(self.runtime_bridge.runtime, "weld_configuration", None)))
        self.welding_panel.apply_requested.connect(
            lambda state: bridge.invoke("apply_weld_configuration", state, model="sequence"))
        self.welding_panel.reload_requested.connect(self._reload_weld_configuration)
        for panel in (self.cleaner_panel, self.task_panel):
            if panel is self.cleaner_panel:
                panel.plan_button.setEnabled(bridge.supports("plan_cleaner", "sequence"))
                panel.execute_button.setEnabled(bridge.supports("execute_cleaner", "sequence"))
                panel.plan_requested.connect(lambda: bridge.invoke("plan_cleaner", model="sequence"))
                panel.execute_requested.connect(lambda: bridge.invoke("execute_cleaner", model="sequence"))
            else:
                panel.plan_button.setEnabled(bridge.available)
                panel.execute_button.setEnabled(bridge.available)
                panel.plan_requested.connect(lambda: bridge.request("plan"))
                panel.execute_requested.connect(lambda: bridge.request("execute"))

    def _reload_weld_configuration(self):
        try:
            recipe, motion = self.runtime_bridge.runtime.weld_configuration()
            self.weld_state.replace(recipe, motion)
            self.welding_panel.refresh()
        except Exception as error:
            self.status_panel.set_error(str(error))

    def _build_sequence_page(self, page):
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("Sequence Builder")
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split, 1)

        palette = QWidget()
        palette_layout = QVBoxLayout(palette)
        palette_title = QLabel("STEP PALETTE")
        palette_title.setObjectName("SectionTitle")
        palette_layout.addWidget(palette_title)
        self.palette = QListWidget()
        self.palette.addItem("Wait / sleep")
        for title, explanation in (
            ("Robot motion", "Motion rows require the existing production teaching workflow"),
            ("Weld scenario", "Use Build Weld Scenario with the existing production runtime"),
        ):
            item = QListWidgetItem(title)
            item.setToolTip(explanation)
            item.setFlags(Qt.NoItemFlags)
            self.palette.addItem(item)
        palette_layout.addWidget(self.palette)
        add_wait = QPushButton("Add Wait")
        add_wait.clicked.connect(self.table_model.add_wait)
        palette_layout.addWidget(add_wait)
        self.build_weld_button = QPushButton("Build Weld Scenario")
        self.build_weld_button.setEnabled(
            self.runtime_bridge.supports("build_weld_scenario", "sequence"))
        self.build_weld_button.clicked.connect(
            lambda: self.runtime_bridge.invoke("build_weld_scenario", self.weld_state,
                                               model="sequence"))
        palette_layout.addWidget(self.build_weld_button)
        split.addWidget(palette)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        sequence_title = QLabel("ORDERED SEQUENCE")
        sequence_title.setObjectName("SectionTitle")
        center_layout.addWidget(sequence_title)
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 105)
        self.table.setColumnWidth(3, 95)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self._select_from_view)
        center_layout.addWidget(self.table)
        load_task = QPushButton("Open saved Task YAML")
        load_task.clicked.connect(self._choose_task_file)
        center_layout.addWidget(load_task)
        row = QHBoxLayout()
        for text, method in (("Delete", self.table_model.delete_selected),
                             ("Delete All", self.table_model.delete_all),
                             ("Duplicate", self.table_model.duplicate_selected),
                             ("↑", lambda: self.table_model.move_selected(-1)),
                             ("↓", lambda: self.table_model.move_selected(1))):
            button = QPushButton(text)
            button.clicked.connect(method)
            row.addWidget(button)
        center_layout.addLayout(row)
        split.addWidget(center)

        self.property_panel = StepPropertyPanel(self.table_model)
        split.addWidget(self.property_panel)
        split.setSizes((190, 670, 320))

    def _build_execution_controls(self):
        execution = QHBoxLayout()
        self.plan_button = QPushButton("Plan")
        self.execute_button = QPushButton("Execute")
        self.stop_button = QPushButton("STOP")
        self.execute_button.setObjectName("PrimaryButton")
        self.stop_button.setObjectName("StopButton")
        for button, action in ((self.plan_button, "plan"),
                               (self.execute_button, "execute"),
                               (self.stop_button, "stop")):
            button.setEnabled(self.runtime_bridge.available)
            button.clicked.connect(lambda _checked=False, name=action: self.runtime_bridge.request(name))
            execution.addWidget(button)
        return execution

    def _select_from_view(self, *_args):
        selected = self.table.selectionModel().selectedRows()
        self.table_model.select_row(selected[0].row() if selected else None)

    def load_task_path(self, path):
        """Import existing task-library rows without executing equipment."""
        if self.sequence_state.running:
            raise ValueError("Finish the active sequence first")
        document, steps = load_task_file(path)
        SequenceModel(steps).validate(require_complete=True)
        visit_order = [safe_task_name(name) for name in document.get("visit_order", ())]
        self.sequence_state.replace(steps)
        with QSignalBlocker(self.task_panel.category):
            self.task_panel.category.setCurrentText(document["category"])
        self.task_panel.task_name.setText(Path(path).stem)
        self.task_order_state.replace(visit_order)
        self.task_panel.refresh()

    def _choose_task_file(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Open saved Task YAML", "", "Task YAML (*.yaml *.yml)"
        )
        if path:
            try:
                self.load_task_path(path)
            except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
                self.status_panel.set_error(str(error))

    def _select_from_state(self, row):
        selection = self.table.selectionModel()
        with QSignalBlocker(selection):
            if row is None or not 0 <= row < self.table_model.rowCount():
                self.table.clearSelection()
            else:
                self.table.selectRow(row)

    def closeEvent(self, event):
        if self.runtime_bridge.runtime is not None:
            active = self.sequence_state.running
            status = getattr(self.runtime_bridge.runtime, "runtime_status", None)
            if callable(status):
                try:
                    active = active or bool(status().get("active_motion"))
                except Exception:
                    active = True
            if active:
                QMessageBox.warning(self, "Motion active", "STOP and wait for all motion to finish before closing Qt.")
                event.ignore()
                return
        self.runtime_bridge.close()
        self.table_model.close()
        super().closeEvent(event)


def launch(sequence_state=None, runtime=None):
    app = QApplication.instance() or QApplication([])
    window = SequenceMainWindow(sequence_state, runtime)
    window.show()
    return app.exec()
