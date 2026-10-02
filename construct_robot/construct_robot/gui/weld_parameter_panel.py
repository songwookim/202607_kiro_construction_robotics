"""Tk table: current weld settings vs the last weld's settings and results."""

import tkinter as tk
from tkinter import ttk

from construct_robot.core.weld_parameter_table import build_rows
from construct_robot.io.weld_logging import read_weld_log_sections


# log key -> (GUI variable attribute, scale applied to the GUI value)
SETTING_VARIABLES = {
    "current_a": ("weld_current_raw", None),
    "voltage": ("weld_voltage_raw", 0.1),
    "material": ("weld_material", None),
    "diameter_mm": ("weld_diameter_mm", None),
    "mode": ("weld_mode", None),
    "gas": ("weld_gas", None),
    "synergic": ("weld_synergic", None),
    "correction": ("weld_correction", None),
    "hot_start_enabled": ("weld_hot_start_enabled", None),
    "hot_start_percent": ("weld_hot_start_percent", None),
    "hot_start_hold_adjustment": ("weld_hot_start_hold_adjustment", None),
    "custom_hot_start_enabled": ("weld_custom_hot_start_enabled", None),
    "custom_hot_start_hold_s": ("weld_custom_hot_start_hold_s", None),
    "custom_hot_start_percent": ("weld_custom_hot_start_percent", None),
    "expect_native_crater": ("weld_expect_native_crater", None),
    "software_crater_enabled": ("weld_software_crater_enabled", None),
    "software_crater_ratio_percent": ("weld_software_crater_ratio_percent", None),
    "software_crater_voltage_v": ("weld_software_crater_voltage_v", None),
    "software_crater_hold_s": ("weld_software_crater_hold_s", None),
    "weld_tcp_speed_mm_s": ("weld_tcp_speed_mm_s", None),
    "weld_lead_in_mm": ("weld_lead_in_mm", None),
    "weld_lead_out_mm": ("weld_lead_out_mm", None),
    "weld_arc_off_delay_ms": ("weld_arc_off_delay_ms", None),
    "weld_weave_enabled": ("weld_weave_enabled", None),
    "weld_weave_pattern": ("weave_pattern", None),
    "weld_weave_reference": ("weld_weave_reference", None),
    "weld_weave_axis": ("weave_axis", None),
    "weld_weave_amplitude_mm": ("weave_amplitude_mm", None),
    "weld_weave_pitch_mm": ("weave_pitch_mm", None),
    "weld_weave_left_dwell_s": ("weave_left_dwell_s", None),
    "weld_weave_right_dwell_s": ("weave_right_dwell_s", None),
}


def current_settings(host):
    """Flat {log key: value} of the GUI's settings; unreadable fields are None."""
    settings = {}
    for key, (attribute, scale) in SETTING_VARIABLES.items():
        variable = getattr(host, attribute, None)
        try:
            value = variable.get() if variable is not None else None
            if scale is not None and value is not None:
                value = float(value) * scale
        except (tk.TclError, TypeError, ValueError):
            value = None
        settings[key] = value
    return settings


class WeldParameterPanel:
    COLUMNS = (
        ("unit", "단위", 80),
        ("now", "현재 설정", 170),
        ("last", "직전 용접 설정", 170),
        ("measured", "직전 용접 실측", 230),
    )

    def __init__(self, parent, host, latest_log_path):
        self.host = host
        self.latest_log_path = latest_log_path
        self._refresh_pending = None
        header = ttk.Frame(parent)
        header.pack(fill=tk.X)
        self.source_label = ttk.Label(header, text="")
        self.source_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(header, text="새로고침", command=self.refresh).pack(side=tk.RIGHT, padx=4)
        ttk.Label(
            header, text="노란색 = 직전 용접과 다른 설정", foreground="#a16207",
        ).pack(side=tk.RIGHT, padx=8)
        style = ttk.Style()
        style.configure("WeldCheck.Treeview", rowheight=26, font=("Sans", 10))
        style.configure("WeldCheck.Treeview.Heading", font=("Sans", 10, "bold"),
                        background="#1f2937", foreground="#ffffff", padding=(4, 5))
        style.map("WeldCheck.Treeview.Heading", background=[("active", "#374151")])
        self.tree = ttk.Treeview(
            parent, columns=[name for name, _, _ in self.COLUMNS],
            height=36, selectmode="none", style="WeldCheck.Treeview",
        )
        self.tree.heading("#0", text="항목")
        self.tree.column("#0", width=260, stretch=False)
        for name, title, width in self.COLUMNS:
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, anchor=tk.CENTER)
        self.tree.tag_configure("group", background="#dbeafe",
                                foreground="#1e3a8a", font=("Sans", 10, "bold"))
        self.tree.tag_configure("even", background="#ffffff")
        self.tree.tag_configure("odd", background="#f3f4f6")
        self.tree.tag_configure("changed", background="#fde68a")
        self.tree.pack(fill=tk.X, pady=(2, 4))
        for attribute, _scale in SETTING_VARIABLES.values():
            variable = getattr(host, attribute, None)
            if variable is not None:
                variable.trace_add("write", lambda *_args: self.schedule_refresh())
        self.refresh()

    def schedule_refresh(self, delay_ms=300):
        """Coalesce bursts of variable writes (typing, spinbox repeat)."""
        if self._refresh_pending is not None:
            return
        self._refresh_pending = self.tree.after(delay_ms, self.refresh)

    def refresh(self):
        self._refresh_pending = None
        last = read_weld_log_sections(self.latest_log_path)
        ended = last.get("header", {}).get("ended")
        self.source_label.configure(
            text=f"직전 용접: {ended} · {self.latest_log_path.name}" if ended
            else "직전 용접 로그 없음"
        )
        self.tree.delete(*self.tree.get_children())
        groups = {}
        for group, label, unit, now, previous, measured, changed in build_rows(
            current_settings(self.host), last
        ):
            if group not in groups:
                groups[group] = self.tree.insert(
                    "", tk.END, text=f"  {group}", open=True, tags=("group",)
                )
            stripe = "odd" if len(self.tree.get_children(groups[group])) % 2 else "even"
            self.tree.insert(
                groups[group], tk.END, text=f"   {label}",
                values=(unit, now, previous, measured),
                # A changed row gets only the highlight: Tk would let the
                # stripe background win otherwise.
                tags=("changed",) if changed else (stripe,),
            )
        self.tree.configure(height=sum(
            1 + len(self.tree.get_children(group)) for group in groups.values()
        ))
