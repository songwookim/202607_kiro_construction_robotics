"""Readable parameter-input tables for the Welding page.

Each group is a small grid: 항목 | 값 | 단위 | 설명, one parameter per row.
Inputs bind directly to the GUI's existing Tk variables, so the welding
workflow keeps reading exactly the same values.
"""

import tkinter as tk
from tkinter import ttk

HINT_COLOR = "#5b6472"
GROUP_COLUMNS = 2


def spin(variable, low, high, increment, width=8):
    return lambda parent: ttk.Spinbox(
        parent, from_=low, to=high, increment=increment,
        textvariable=variable, width=width,
    )


def combo(variable, values, width=14):
    return lambda parent: ttk.Combobox(
        parent, textvariable=variable, values=tuple(values),
        state="readonly", width=width,
    )


def check(variable, text="사용"):
    return lambda parent: ttk.Checkbutton(parent, text=text, variable=variable)


def entry(variable, width=9):
    return lambda parent: ttk.Entry(parent, textvariable=variable, width=width)


def parameter_group(parent, title, rows):
    """One titled grid; rows are (label, widget factory, unit, hint).

    ``hint`` may be a plain string or a tk.StringVar for live values.
    """
    frame = ttk.LabelFrame(parent, text=title, padding=(8, 4))
    for column, heading in enumerate(("항목", "값", "단위", "설명")):
        ttk.Label(frame, text=heading, font=("Sans", 9, "bold"),
                  foreground=HINT_COLOR).grid(
            row=0, column=column, sticky=tk.W, padx=(0, 10), pady=(0, 2))
    ttk.Separator(frame).grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(0, 3))
    for index, (label, factory, unit, hint) in enumerate(rows, start=2):
        ttk.Label(frame, text=label).grid(row=index, column=0, sticky=tk.W,
                                          padx=(0, 10), pady=2)
        factory(frame).grid(row=index, column=1, sticky=tk.W, padx=(0, 10), pady=2)
        ttk.Label(frame, text=unit, foreground=HINT_COLOR).grid(
            row=index, column=2, sticky=tk.W, padx=(0, 10))
        hint_options = {"textvariable": hint} if isinstance(hint, tk.StringVar) else {"text": hint}
        ttk.Label(frame, foreground=HINT_COLOR, wraplength=260, **hint_options).grid(
            row=index, column=3, sticky=tk.W)
    frame.columnconfigure(3, weight=1)
    return frame


def parameter_tables(parent, groups):
    """Lay out [(title, rows)] as group grids, GROUP_COLUMNS per line."""
    container = ttk.Frame(parent)
    container.pack(fill=tk.X, pady=(2, 4))
    for index, (title, rows) in enumerate(groups):
        frame = parameter_group(container, title, rows)
        frame.grid(row=index // GROUP_COLUMNS, column=index % GROUP_COLUMNS,
                   sticky=tk.NSEW, padx=4, pady=4)
    for column in range(GROUP_COLUMNS):
        container.columnconfigure(column, weight=1, uniform="parameter_groups")
    return container
