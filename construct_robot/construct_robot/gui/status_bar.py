"""Always-visible operator status strip and event log for the weld GUI.

Both views mirror state owned elsewhere (connection indicators, keyboard
teaching status, pipeline status); they keep no workflow state of their own.
The strip carries the software STOP for every tab, and errors stay flagged
until the operator acknowledges them instead of being overwritten by the next
status line.
"""

import collections
import time
import tkinter as tk
from tkinter import ttk


BAR_BG = "#1f2937"
CHIP_IDLE = ("#374151", "#e5e7eb")
CHIP_GOOD = ("#16a34a", "#ffffff")
CHIP_BUSY = ("#f59e0b", "#111827")
CHIP_BAD = ("#dc2626", "#ffffff")
PIPELINE_COLORS = {
    "WAITING": ("#374151", "#f3f4f6"),
    "RESULT": ("#14532d", "#dcfce7"),
    "ERROR": ("#7f1d1d", "#fee2e2"),
}
LOG_COLORS = {"WAITING": "#d1d5db", "RESULT": "#86efac", "ERROR": "#fca5a5"}


def _state_from_label(text):
    """'LEFT (192.168.1.11): O' -> 'O'."""
    return str(text).rsplit(":", 1)[-1].strip() or "-"


def _chip_colors(state):
    if state in ("O", "LIVE"):
        return CHIP_GOOD
    if state in ("X", "!"):
        return CHIP_BAD
    return CHIP_IDLE


def _jog_colors(status):
    text = str(status).upper()
    if text.startswith("MOVING") or text.startswith("WIRE"):
        return CHIP_BUSY
    if "FAILED" in text or "FALLBACK" in text:
        return CHIP_BAD
    if text.startswith("READY"):
        return CHIP_GOOD
    return CHIP_IDLE


class OperatorStatusBar:
    """Connection chips, keyboard teaching, pipeline status and STOP."""

    REFRESH_MS = 250
    CHIPS = (
        ("left", "LEFT"), ("right", "RIGHT"), ("head", "HEAD"),
        ("welder", "WELDER"), ("scanner", "SCANNER"),
    )

    def __init__(self, gui, parent):
        self.gui = gui
        self.unacknowledged_errors = 0
        self.frame = tk.Frame(parent, bg=BAR_BG)
        self.frame.pack(side=tk.TOP, fill=tk.X)

        # STOP spans both rows on the right; it is never pushed off-screen.
        self.stop_button = tk.Button(
            self.frame, text="STOP", command=gui.emergency_stop_all,
            bg="#dc2626", fg="white", activebackground="#991b1b",
            activeforeground="white", font=("Sans", 18, "bold"),
            relief=tk.RAISED, borderwidth=3, padx=26, pady=0,
            takefocus=0,
        )
        self.stop_button.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 6), pady=4)

        rows = tk.Frame(self.frame, bg=BAR_BG)
        rows.pack(side=tk.LEFT, fill=tk.X, expand=True)
        top = tk.Frame(rows, bg=BAR_BG)
        top.pack(fill=tk.X, padx=(6, 0), pady=(4, 2))
        self.chips = {}
        for key, title in self.CHIPS:
            chip = tk.Label(
                top, text=f"{title} -", font=("Sans", 10, "bold"),
                padx=7, pady=2, bg=CHIP_IDLE[0], fg=CHIP_IDLE[1],
            )
            chip.pack(side=tk.LEFT, padx=2)
            self.chips[key] = (chip, title)
        self.jog_chip = tk.Label(
            top, textvariable=gui.keyboard_jog_status, anchor=tk.W,
            font=("Sans", 10, "bold"), padx=7, pady=2,
            bg=CHIP_IDLE[0], fg=CHIP_IDLE[1],
        )
        self.jog_chip.pack(side=tk.LEFT, padx=(10, 2))
        self.error_chip = tk.Label(
            top, text="", font=("Sans", 10, "bold"), padx=7, pady=2,
            bg=BAR_BG, fg=BAR_BG, cursor="hand2",
        )
        self.error_chip.pack(side=tk.RIGHT, padx=2)
        self.error_chip.bind("<Button-1>", self.acknowledge_errors)

        self.pipeline_label = tk.Label(
            rows, text="WAITING · starting", anchor=tk.W,
            font=("Sans", 10), padx=8, pady=2,
            bg=PIPELINE_COLORS["WAITING"][0], fg=PIPELINE_COLORS["WAITING"][1],
        )
        self.pipeline_label.pack(fill=tk.X, padx=(8, 2), pady=(0, 4))
        gui.root.after(self.REFRESH_MS, self.refresh)

    def _sources(self):
        gui = self.gui
        labels = dict(getattr(gui, "robot_connection_labels", {}) or {})
        labels["welder"] = getattr(gui, "welder_connection_label", None)
        labels["scanner"] = getattr(gui, "scanner_connection_label", None)
        return labels

    def refresh(self):
        if getattr(self.gui, "_closing", False):
            return
        try:
            sources = self._sources()
            for key, (chip, title) in self.chips.items():
                source = sources.get(key)
                if source is None:
                    continue
                state = _state_from_label(source.cget("text"))
                background, foreground = _chip_colors(state)
                chip.configure(text=f"{title} {state}", bg=background, fg=foreground)
            background, foreground = _jog_colors(self.gui.keyboard_jog_status.get())
            self.jog_chip.configure(bg=background, fg=foreground)
        except tk.TclError:
            pass
        finally:
            if not getattr(self.gui, "_closing", False):
                self.gui.root.after(self.REFRESH_MS, self.refresh)

    def pipeline(self, state, message):
        background, foreground = PIPELINE_COLORS.get(state, PIPELINE_COLORS["WAITING"])
        self.pipeline_label.configure(
            text=f"{state} · {message}", bg=background, fg=foreground,
        )
        if state == "ERROR":
            self.unacknowledged_errors += 1
            self.error_chip.configure(
                text=f"ERROR × {self.unacknowledged_errors} · {time.strftime('%H:%M:%S')}  (click = OK)",
                bg=CHIP_BAD[0], fg=CHIP_BAD[1],
            )

    def acknowledge_errors(self, _event=None):
        self.unacknowledged_errors = 0
        self.error_chip.configure(text="", bg=BAR_BG, fg=BAR_BG)


class EventLogPanel:
    """Timestamped history of pipeline status lines (newest at the bottom)."""

    MAX_LINES = 400
    FLUSH_MS = 150

    def __init__(self, gui, parent):
        self.gui = gui
        self.pending = collections.deque(maxlen=self.MAX_LINES)
        self.flush_scheduled = False
        self.expanded = True
        self.frame = ttk.Frame(parent)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X)
        self.toggle_button = ttk.Button(
            header, text="▼ Event log", command=self.toggle, takefocus=0,
        )
        self.toggle_button.pack(side=tk.LEFT, padx=4, pady=(2, 0))
        ttk.Button(header, text="Clear", command=self.clear, takefocus=0).pack(
            side=tk.LEFT, padx=2, pady=(2, 0))
        self.body = ttk.Frame(self.frame)
        self.body.pack(fill=tk.X)
        self.text = tk.Text(
            self.body, height=6, wrap=tk.NONE, state=tk.DISABLED,
            font=("DejaVu Sans Mono", 9), bg="#111827", fg="#e5e7eb",
            insertbackground="#111827", relief=tk.FLAT, takefocus=0,
        )
        scroll = ttk.Scrollbar(self.body, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.text.tag_configure("time", foreground="#9ca3af")
        for state, color in LOG_COLORS.items():
            self.text.tag_configure(state, foreground=color)
        # Viewing the log must not take keyboard focus: a focused Text widget
        # swallows the teaching arrows (input-field focus rule).
        self.text.bind("<FocusIn>", lambda _event: gui.root.focus_set())

    def owns(self, widget):
        return widget is self.text

    def append(self, state, message):
        self.pending.append((time.strftime("%H:%M:%S"), state, str(message)))
        if not self.flush_scheduled:
            self.flush_scheduled = True
            self.gui.root.after(self.FLUSH_MS, self.flush)

    def flush(self):
        self.flush_scheduled = False
        if not self.pending:
            return
        try:
            at_bottom = self.text.yview()[1] >= 0.999
            self.text.configure(state=tk.NORMAL)
            while self.pending:
                stamp, state, message = self.pending.popleft()
                self.text.insert(tk.END, f"{stamp} ", ("time",))
                self.text.insert(tk.END, f"{state:<7} {message}\n", (state,))
            lines = int(self.text.index("end-1c").split(".")[0])
            if lines > self.MAX_LINES:
                self.text.delete("1.0", f"{lines - self.MAX_LINES}.0")
            self.text.configure(state=tk.DISABLED)
            if at_bottom:
                self.text.see(tk.END)
        except tk.TclError:
            pass

    def clear(self):
        self.pending.clear()
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.body.pack(fill=tk.X)
            self.toggle_button.configure(text="▼ Event log")
        else:
            self.body.pack_forget()
            self.toggle_button.configure(text="▶ Event log")
