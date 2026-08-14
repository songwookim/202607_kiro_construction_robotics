"""Interactive 3D diagnostic plot for saved DI8 seam touch geometry."""

import argparse
from pathlib import Path
import sys

# Ubuntu's system matplotlib registers its mpl_toolkits namespace from a .pth
# file before the workspace venv is searched.  Keep mplot3d paired with the
# NumPy-2-compatible matplotlib installed in this venv.
import mpl_toolkits

_venv_toolkits = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "mpl_toolkits"
)
if _venv_toolkits.is_dir():
    mpl_toolkits.__path__.insert(0, str(_venv_toolkits))

import matplotlib.pyplot as plt
import yaml


COLORS = {
    "wall": "tab:red",
    "floor": "tab:blue",
    "seam": "limegreen",
    "midpoint": "gold",
    "wait": "tab:purple",
}


def _position(record, field):
    pose = record.get(field)
    if not pose:
        return None
    value = pose.get("position_m")
    if not isinstance(value, dict):
        return None
    return tuple(float(value[key]) for key in ("x", "y", "z"))


def _scatter(ax, point, color, marker, label, size=70, alpha=1.0):
    ax.scatter(*point, c=color, marker=marker, s=size, alpha=alpha, label=label)
    ax.text(
        point[0],
        point[1],
        point[2],
        f"  {label}\n  ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})",
        fontsize=8,
    )


def _mean_point(*points):
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    return tuple(
        sum(point[index] for point in valid) / len(valid)
        for index in range(3)
    )


def plot_touch_yaml(path, wall_offset_mm=0.0, floor_offset_mm=0.0, endpoint=None):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    records = document.get("touches", {})
    endpoints = (endpoint,) if endpoint else ("start", "goal")
    complete = []
    for name in endpoints:
        wall_record = records.get(f"{name}_wall", {})
        floor_record = records.get(f"{name}_floor", {})
        wall = _position(wall_record, "contact_tcp")
        floor = _position(floor_record, "contact_tcp")
        if wall is not None and floor is not None:
            complete.append((name, wall_record, floor_record, wall, floor))
    if not complete:
        raise ValueError("No complete wall/floor touch pair exists in the YAML")

    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(111, projection="3d")
    all_points = []
    table_lines = []
    for name, wall_record, floor_record, wall, floor in complete:
        midpoint = tuple((wall[i] + floor[i]) * 0.5 for i in range(3))
        seam = (
            midpoint[0],
            wall[1] + wall_offset_mm * 0.001,
            floor[2] + floor_offset_mm * 0.001,
        )
        # Both probes should start at the same physical wait position.  Their
        # mean suppresses the small TF/joint-feedback difference between the
        # two captures and gives one unambiguous WAIT point in the plot.
        wait = _mean_point(
            _position(wall_record, "probe_start_tcp"),
            _position(floor_record, "probe_start_tcp"),
        )
        prefix = name.upper()
        if wait is not None:
            _scatter(
                axis,
                wait,
                COLORS["wait"],
                "s",
                f"{prefix} WELD WAIT / PROBE START",
                75,
            )
        _scatter(axis, wall, COLORS["wall"], "o", f"{prefix} WALL")
        _scatter(axis, floor, COLORS["floor"], "o", f"{prefix} FLOOR")
        _scatter(axis, seam, COLORS["seam"], "*", f"{prefix} SEAM", 150)
        _scatter(
            axis, midpoint, COLORS["midpoint"], "D", f"{prefix} MIDPOINT", 65
        )
        axis.plot(
            (wall[0], midpoint[0], floor[0]),
            (wall[1], midpoint[1], floor[1]),
            (wall[2], midpoint[2], floor[2]),
            color="0.65",
            linewidth=1.5,
        )
        # These right-angle guides make it obvious that seam Y comes only
        # from WALL and seam Z comes only from FLOOR.
        axis.plot(
            (wall[0], seam[0]),
            (wall[1], seam[1]),
            (wall[2], seam[2]),
            color=COLORS["wall"],
            linestyle="--",
            alpha=0.7,
        )
        axis.plot(
            (floor[0], seam[0]),
            (floor[1], seam[1]),
            (floor[2], seam[2]),
            color=COLORS["floor"],
            linestyle="--",
            alpha=0.7,
        )
        all_points.extend((wall, floor, midpoint, seam))
        if wait is not None:
            all_points.append(wait)
        table_lines.extend((
            f"{prefix} WAIT     {wait}",
            f"{prefix} WALL     {wall}",
            f"{prefix} FLOOR    {floor}",
            f"{prefix} MIDPOINT {midpoint}",
            f"{prefix} COMPUTED {seam}",
        ))

    axis.set_xlabel("World X [m]")
    axis.set_ylabel("World Y [m]")
    axis.set_zlabel("World Z [m]")
    axis.set_title(
        "DI8 seam touch geometry: wait, two contacts, midpoint, computed point\n"
        "SEAM = (mean touch X, WALL Y, FLOOR Z)"
    )
    ranges = [
        max(point[index] for point in all_points)
        - min(point[index] for point in all_points)
        for index in range(3)
    ]
    maximum_range = max(max(ranges), 0.010)
    axis.set_box_aspect(tuple(max(value, maximum_range * 0.25) for value in ranges))
    axis.legend(loc="upper left", fontsize=8)
    figure.text(
        0.01,
        0.01,
        "\n".join(table_lines),
        family="monospace",
        fontsize=8,
        va="bottom",
    )
    figure.tight_layout(rect=(0.0, 0.15, 1.0, 1.0))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path")
    parser.add_argument("--wall-offset-mm", type=float, default=0.0)
    parser.add_argument("--floor-offset-mm", type=float, default=0.0)
    parser.add_argument("--endpoint", choices=("start", "goal"))
    arguments = parser.parse_args()
    plot_touch_yaml(
        arguments.yaml_path,
        arguments.wall_offset_mm,
        arguments.floor_offset_mm,
        arguments.endpoint,
    )


if __name__ == "__main__":
    main()
