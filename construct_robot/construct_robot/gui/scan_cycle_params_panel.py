"""4-PASS PARAMS tab: edit / save the Scan + 4 Passes parameters and build the cycle with them (no motion)."""

import tkinter as tk
from tkinter import ttk

from construct_robot.application import scan_cycle_params as scp

COLUMNS = (("current_a", "Current A", 7), ("voltage_v", "Voltage V", 7), ("speed_mm_s", "Speed mm/s", 7),
           ("weave", "Weave", 0), ("weave_pattern", "Pattern", 9), ("weave_amplitude_mm", "Amp ± mm", 6),
           ("weave_pitch_mm", "Pitch mm", 6), ("weave_left_dwell_s", "Dwell L s", 6),
           ("weave_right_dwell_s", "Dwell R s", 6), ("start_touch", "Traverse", 0))
ARC_COLUMNS = (("hot_start_enabled", "Hot start", 0), ("hot_start_percent", "Boost %", 5),
               ("hot_start_hold_adjustment", "Hold adj", 4), ("custom_hot_start_enabled", "Custom hot", 0),
               ("custom_hot_start_percent", "Boost %", 5), ("custom_hot_start_hold_s", "Hold s", 5),
               ("software_crater_enabled", "SW crater", 0), ("software_crater_ratio_percent", "Current %", 5),
               ("software_crater_voltage_v", "Voltage V", 5), ("software_crater_hold_s", "Hold s", 5))
TOUCH_FIELDS = (("speed_mm_s", "Descent speed below the reference TCP mm/s (≤ 0.5)"),
                ("max_descent_mm", "Max descent below the reference TCP mm (≤ 10)"))


class ScanCycleParamsPanel:
    def __init__(self, host, page):
        self.host = host
        self.vars = {k: {} for k in scp.PASSES}
        self.touch_vars = {}
        self.arc_vars = {k: {} for k in scp.PASSES}
        self.arc_use_table = tk.BooleanVar(value=False)
        self.cleaner = tk.StringVar(value="cut_only")
        # not saved with the parameters: every GUI start begins at P1 again
        self.start_pass = tk.StringVar(value="1")
        self.status = tk.StringVar(value="")
        ttk.Label(page, text="Scan + 4 Passes · parameters", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(page, foreground="#5f6368", wraplength=900, justify=tk.LEFT, text=(
            "The Sequence Builder's Scan + 4 Passes buttons keep the fixed values. The buttons below build the same "
            "cycle (scan → P1 → clean → P2 → clean → P3 → clean → P4) with this table. Build only freezes the "
            "sequence; nothing moves until Execute. Weave runs across the seam in the scanned plate plane; "
            "Traverse = START touch: move to the pass's reference TCP (its START), then descend World −Z slowly "
            "until the touch input, and shift that pass by the touched height.")).pack(anchor=tk.W, pady=(2, 8))

        table = ttk.LabelFrame(page, text="Per pass", padding=6)
        table.pack(fill=tk.X)
        ttk.Label(table, text="Pass").grid(row=0, column=0, padx=4, sticky=tk.W)
        for c, (_key, title, _w) in enumerate(COLUMNS, 1):
            ttk.Label(table, text=title).grid(row=0, column=c, padx=4)
        for r, k in enumerate(scp.PASSES, 1):
            ttk.Label(table, text=f"P{k}").grid(row=r, column=0, padx=4, sticky=tk.W)
            for c, (key, _title, width) in enumerate(COLUMNS, 1):
                if key in ("weave", "start_touch"):
                    var = tk.BooleanVar()
                    ttk.Checkbutton(table, variable=var).grid(row=r, column=c, padx=4)
                elif key == "weave_pattern":
                    var = tk.StringVar()
                    ttk.Combobox(table, textvariable=var, values=scp.WEAVE_PATTERNS, state="readonly",
                                 width=width).grid(row=r, column=c, padx=4)
                else:
                    var = tk.StringVar()
                    ttk.Entry(table, textvariable=var, width=width).grid(row=r, column=c, padx=4, pady=2)
                self.vars[k][key] = var

        arc = ttk.LabelFrame(page, text="Hot start / crater per pass", padding=6)
        arc.pack(fill=tk.X, pady=(8, 0))
        head = ttk.Frame(arc)
        head.grid(row=0, column=0, columnspan=len(ARC_COLUMNS) + 1, sticky=tk.W)
        ttk.Checkbutton(head, text="Use this table (off: every pass takes the Welding tab recipe at Build)",
                        variable=self.arc_use_table).pack(side=tk.LEFT)
        ttk.Button(head, text="Copy from Welding tab", command=self.copy_arc_from_recipe).pack(side=tk.LEFT, padx=8)
        ttk.Label(arc, text="Pass").grid(row=1, column=0, padx=4, sticky=tk.W)
        for c, (_key, title, _w) in enumerate(ARC_COLUMNS, 1):
            ttk.Label(arc, text=title).grid(row=1, column=c, padx=4)
        for r, k in enumerate(scp.PASSES, 2):
            ttk.Label(arc, text=f"P{k}").grid(row=r, column=0, padx=4, sticky=tk.W)
            for c, (key, _title, width) in enumerate(ARC_COLUMNS, 1):
                if width == 0:
                    var = tk.BooleanVar()
                    ttk.Checkbutton(arc, variable=var).grid(row=r, column=c, padx=4)
                else:
                    var = tk.StringVar()
                    ttk.Entry(arc, textvariable=var, width=width).grid(row=r, column=c, padx=4, pady=2)
                self.arc_vars[k][key] = var
        ttk.Label(arc, foreground="#5f6368", text=(
            "Hot start = the welder's native boost (Hi-COMM); Custom hot = PC-timed boost; SW crater = PC crater at "
            "the end (current % of that pass's current).")).grid(row=6, column=0, columnspan=len(ARC_COLUMNS) + 1,
                                                                   sticky=tk.W, pady=(4, 0))

        other = ttk.LabelFrame(page, text="Traverse (START touch) and cleaner", padding=6)
        other.pack(fill=tk.X, pady=(8, 0))
        for c, (key, title) in enumerate(TOUCH_FIELDS):
            ttk.Label(other, text=title).grid(row=0, column=2 * c, padx=(4, 2), sticky=tk.E)
            var = tk.StringVar()
            ttk.Entry(other, textvariable=var, width=6).grid(row=0, column=2 * c + 1, padx=(2, 10))
            self.touch_vars[key] = var
        ttk.Label(other, text="Cleaner between passes").grid(row=1, column=0, padx=(4, 2), pady=(6, 0), sticky=tk.E)
        ttk.Combobox(other, textvariable=self.cleaner, values=scp.CLEANER_MODES, state="readonly",
                     width=9).grid(row=1, column=1, padx=(2, 10), pady=(6, 0), sticky=tk.W)

        buttons = ttk.Frame(page)
        buttons.pack(fill=tk.X, pady=(10, 0))
        for text, command in (("Load saved", self.load_saved), ("Defaults", self.load_defaults),
                              ("Save", self.save)):
            ttk.Button(buttons, text=text, command=command).pack(side=tk.LEFT, padx=2)
        ttk.Separator(buttons, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        for text, kwargs in (("Build Scan + 4 Passes", {}),
                             ("Build Scan + 4 Passes (1 Scan)", {"rescan_after_passes": False})):
            ttk.Button(buttons, text=text, command=lambda kw=kwargs: self.build(**kw)).pack(side=tk.LEFT, padx=2)
        no_scan = ttk.LabelFrame(page, text="No Scan (saved targets of the latest scan)", padding=6)
        no_scan.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(no_scan, text="Start at pass").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Combobox(no_scan, textvariable=self.start_pass, values=("1", "2", "3", "4"), state="readonly",
                     width=3).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(no_scan, text="Build No Scan + Passes",
                   command=lambda: self.build(rescan_after_passes=False, reuse_latest=True,
                                              start_pass=int(self.start_pass.get()))).pack(side=tk.LEFT, padx=2)
        ttk.Label(no_scan, foreground="#5f6368", text=(
            "Start at 2 / 3 / 4: the earlier passes are taken as already welded; the cycle runs that pass → clean → "
            "... → P4 with the same saved targets")).pack(side=tk.LEFT, padx=8)
        ttk.Label(page, textvariable=self.status, foreground="#5f6368", wraplength=900,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))
        try:
            self.set_params(scp.load_params())
        except (OSError, ValueError) as error:
            self.set_params(scp.default_params())
            self.status.set(f"saved parameters unreadable, defaults shown: {error}")

    def set_params(self, params):
        params = scp.validate_params(params)
        for k in scp.PASSES:
            for key, var in self.vars[k].items():
                value = params["passes"][k][key]
                var.set(value if isinstance(value, (bool, str)) else f"{value:g}")
        for key, var in self.touch_vars.items():
            var.set(f"{params['start_touch'][key]:g}")
        self.cleaner.set(params["cleaner"])
        self.arc_use_table.set(params["arc"]["use_table"])
        for k in scp.PASSES:
            for key, var in self.arc_vars[k].items():
                value = params["arc"]["passes"][k][key]
                var.set(value if isinstance(value, bool) else f"{value:g}")
        self.status.set(scp.summary(params))

    def params(self):
        """Validated params from the table (ValueError names the field)."""
        raw = {"passes": {k: {key: var.get() for key, var in self.vars[k].items()} for k in scp.PASSES},
               "start_touch": {key: var.get() for key, var in self.touch_vars.items()},
               "cleaner": self.cleaner.get(),
               "arc": {"use_table": self.arc_use_table.get(),
                       "passes": {k: {key: var.get() for key, var in self.arc_vars[k].items()} for k in scp.PASSES}}}
        return scp.validate_params(raw)

    def copy_arc_from_recipe(self):
        """Fill every pass row with the Welding tab's current hot start / crater values (then edit per pass)."""
        try:
            recipe = self.host._digital_weld_settings()
        except ValueError as error:
            self.host.error(f"4-PASS PARAMS: {error}")
            return
        for k in scp.PASSES:
            for key, var in self.arc_vars[k].items():
                value = recipe[key]
                var.set(bool(value) if scp.ARC_LIMITS[key] is None else f"{float(value):g}")
        self.status.set("hot start / crater copied from the Welding tab (switch 'Use this table' on to apply)")

    def load_saved(self):
        try:
            self.set_params(scp.load_params())
        except (OSError, ValueError) as error:
            self.host.error(f"4-PASS PARAMS: {error}")

    def load_defaults(self):
        self.set_params(scp.default_params())
        self.status.set("defaults (not saved) · " + self.status.get())

    def save(self):
        try:
            params = self.params()
            path = scp.save_params(params)
        except (OSError, ValueError) as error:
            self.host.error(f"4-PASS PARAMS: {error}")
            return
        self.status.set(f"saved {path} · {scp.summary(params)}")
        self.host.log(f"4-PASS PARAMS saved · {scp.summary(params)}")

    def build(self, **kwargs):
        try:
            params = self.params()
        except ValueError as error:
            self.host.error(f"4-PASS PARAMS: {error}")
            return False
        ok = self.host.build_scan_weld_cycle(params=params, **kwargs)
        if ok:
            first = kwargs.get("start_pass", 1)
            start = f"starts at P{first} · " if first != 1 else ""
            self.status.set(f"built {start}with: {scp.summary(params)} · check the Sequence table, then Execute")
        return ok
