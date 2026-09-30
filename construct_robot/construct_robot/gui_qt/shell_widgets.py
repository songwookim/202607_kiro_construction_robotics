"""Presentation-only widgets for the Qt operator shell."""

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


APP_STYLESHEET = """
QMainWindow, QWidget { background: #171d24; color: #e6ebf0; font-size: 12px; }
QLabel { background: transparent; }
QFrame#TopBar, QFrame#ExecutionBar, QFrame#Navigation, QFrame#PanelCard {
    background: #202832; border: 1px solid #34404d; border-radius: 5px;
}
QLabel#PageTitle { font-size: 18px; font-weight: 700; color: #f2f5f8; }
QLabel#SectionTitle { font-size: 13px; font-weight: 700; color: #cbd8e4; }
QLabel#Muted { color: #96a7b8; }
QLabel#StatusChip { background: #29323d; border: 1px solid #465363;
    border-radius: 4px; padding: 5px 9px; font-weight: 600; }
QLabel#StatusChip[state="ready"] { color: #9ee1b5; border-color: #3c7858; }
QLabel#StatusChip[state="warning"] { color: #f1cb84; border-color: #8a6833; }
QLabel#StatusChip[state="error"] { color: #f4a6a6; border-color: #9b5050; }
QPushButton { background: #303b48; border: 1px solid #526070; border-radius: 4px;
    padding: 6px 10px; min-height: 20px; }
QPushButton:hover { background: #3b4a58; }
QPushButton:checked { background: #315b74; border-color: #6db9e7; color: #f0f8ff; }
QPushButton:disabled { color: #758391; background: #252e37; border-color: #384450; }
QPushButton#NavButton { text-align: left; border: 0; background: transparent; padding: 9px 12px; }
QPushButton#NavButton[selected="true"] { background: #2b4456; color: #e3f4ff;
    border-left: 3px solid #6db9e7; font-weight: 700; }
QPushButton#StopButton { background: #8e3535; border-color: #b95b5b; color: white; font-weight: 700; }
QPushButton#StopButton:hover { background: #a34444; }
QPushButton#PrimaryButton { background: #315b74; border-color: #4984a7; font-weight: 700; }
QPushButton#SectionToggle { text-align: left; font-weight: 700; background: #26313d; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #26313c;
    border: 1px solid #526070; border-radius: 3px; padding: 4px; min-height: 20px; }
QTableView, QTableWidget, QListWidget { background: #202934; alternate-background-color: #26313b;
    border: 1px solid #3d4b59; gridline-color: #374554; selection-background-color: #335675; }
QHeaderView::section { background: #2b3642; color: #d5e0ea; border: 0;
    border-right: 1px solid #43515f; padding: 6px; }
QScrollArea { border: 0; }
QProgressBar { background: #293540; border: 1px solid #455563; border-radius: 4px;
    text-align: center; min-height: 12px; }
QProgressBar::chunk { background: #4a9fbd; border-radius: 3px; }
"""


class StatusChip(QLabel):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.setObjectName("StatusChip")
        self.set_state(None)

    def set_state(self, value, detail=None):
        if value is None:
            state, label = "unknown", "UNKNOWN"
        elif isinstance(value, bool):
            state, label = ("ready", "CONNECTED") if value else ("unknown", "DISCONNECTED")
        else:
            label = str(value)
            state = "unknown"
        rendered = f"●  {self.title}  {detail or label}"
        if self.text() == rendered and self.property("state") == state:
            return
        self.setText(rendered)
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)


class TopStatusBar(QFrame):
    def __init__(self, production=False, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 7, 12, 7)
        title = QLabel("WELD WORKSTATION")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        layout.addStretch()
        self.chips = {name: StatusChip(name.upper()) for name in ("left", "right", "welder", "touch")}
        for chip in self.chips.values():
            layout.addWidget(chip)
        self.mode = StatusChip("MODE")
        self.mode.set_state("PRODUCTION" if production else "OFFLINE")
        layout.addWidget(self.mode)

    def set_snapshot(self, snapshot):
        robots = (snapshot or {}).get("robots", {})
        for arm in ("left", "right"):
            self.chips[arm].set_state(robots.get(arm))

    def set_io_snapshot(self, snapshot):
        snapshot = snapshot or {}
        self.chips["welder"].set_state(snapshot.get("hicomm_connected"))
        self.chips["touch"].set_state(snapshot.get("fastech_connected"))


class CollapsibleSection(QWidget):
    """Expands presentation only; it has no model or runtime connection."""

    def __init__(self, title, expanded=False, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.toggle = QPushButton(title)
        self.toggle.setObjectName("SectionToggle")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        layout.addWidget(self.toggle)
        self.body = QWidget()
        self.content_layout = QVBoxLayout(self.body)
        self.content_layout.setContentsMargins(8, 5, 8, 8)
        self.content_layout.setSpacing(5)
        layout.addWidget(self.body)
        self.body.setVisible(expanded)
        self.toggle.toggled.connect(self.body.setVisible)
