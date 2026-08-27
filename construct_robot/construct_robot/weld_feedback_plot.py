"""Plot electrical feedback and complete 3D trajectories from a weld log."""

import argparse
import math
from pathlib import Path
import re
import sys

import yaml


SAMPLE_COLUMNS = (
    "elapsed_s",
    "raw0",
    "state",
    "arc",
    "gas",
    "fwd",
    "wcr",
    "current_a",
    "voltage_v",
    "wire_feed_m_min",
    "set_current_a",
    "set_voltage_v",
    "error",
    "db",
    "collision",
)

TCP_TRAJECTORY_COLUMNS = (
    "elapsed_s",
    "x_m",
    "y_m",
    "z_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "speed_m_s",
    "tf_stamp_s",
    "along_mm",
    "remaining_mm",
    "cross_track_mm",
    "progress",
    "waypoint_index",
    "phase",
)

_STEP_POSITION = re.compile(
    r"^steps\[(\d+)\]\.(.+)\.position_m\.(x|y|z)$"
)


def _prepare_mplot3d():
    """Prefer the active venv's mpl_toolkits over Ubuntu's older namespace."""
    import mpl_toolkits

    candidate = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "mpl_toolkits"
    )
    if (candidate / "mplot3d" / "__init__.py").is_file():
        path = str(candidate)
        if path not in mpl_toolkits.__path__:
            mpl_toolkits.__path__.insert(0, path)
    # Importing this explicitly registers the projection before pyplot creates
    # a figure.  It also turns a mixed Matplotlib install into a clear error.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def parse_weld_feedback_log(path):
    """Return scalar sections and time-series samples from one feedback log."""
    path = Path(path).expanduser()
    sections = {}
    samples = []
    section = "header"
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            sections.setdefault(section, {})
            continue
        if section == "samples":
            if line.startswith("elapsed_s "):
                continue
            values = line.split()
            if len(values) != len(SAMPLE_COLUMNS):
                continue
            sample = dict(zip(SAMPLE_COLUMNS, values))
            try:
                for key in (
                    "elapsed_s",
                    "current_a",
                    "voltage_v",
                    "wire_feed_m_min",
                    "set_current_a",
                    "set_voltage_v",
                ):
                    sample[key] = float(sample[key])
                for key in ("arc", "gas", "fwd", "wcr", "error", "db", "collision"):
                    sample[key] = int(sample[key])
            except ValueError:
                continue
            samples.append(sample)
        elif "=" in line:
            key, value = line.split("=", 1)
            sections.setdefault(section, {})[key] = value
    if not samples:
        raise ValueError(f"no feedback samples found in {path}")
    return sections, samples


def _section_text(lines, section_name):
    marker = f"[{section_name}]"
    collecting = False
    result = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if collecting:
                break
            collecting = stripped == marker
            continue
        if collecting:
            result.append(raw_line)
    return "\n".join(result).strip()


def _finite_position(values):
    try:
        position = tuple(float(values[axis]) for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError):
        return None
    return position if all(math.isfinite(value) for value in position) else None


def _yaml_positions(value, prefix=""):
    """Yield every position_m/tcp_pose_world position in a YAML snapshot."""
    if isinstance(value, dict):
        if "position_m" in value and isinstance(value["position_m"], dict):
            position = _finite_position(value["position_m"])
            if position is not None:
                yield prefix or "pose", position
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _yaml_positions(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _yaml_positions(child, f"{prefix}[{index}]")


def parse_weld_trajectory_log(path):
    """Parse measured TCP and all embedded commanded/teaching/touch poses."""
    path = Path(path).expanduser().resolve()
    lines = path.read_text(encoding="utf-8").splitlines()
    section = "header"
    execution = {}
    actual = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "execution_conditions" and "=" in line:
            key, value = line.split("=", 1)
            execution[key] = value
        elif section == "tcp_trajectory":
            if line.startswith("elapsed_s "):
                continue
            values = line.split(maxsplit=len(TCP_TRAJECTORY_COLUMNS) - 1)
            if len(values) != len(TCP_TRAJECTORY_COLUMNS):
                continue
            sample = dict(zip(TCP_TRAJECTORY_COLUMNS, values))
            try:
                for key in TCP_TRAJECTORY_COLUMNS[:-2]:
                    sample[key] = float(sample[key])
                sample["waypoint_index"] = int(sample["waypoint_index"])
            except ValueError:
                continue
            if all(
                math.isfinite(sample[key]) for key in ("x_m", "y_m", "z_m")
            ):
                actual.append(sample)

    steps = {}
    for key, value in execution.items():
        match = _STEP_POSITION.match(key)
        if match:
            step_index = int(match.group(1))
            entity = match.group(2)
            axis = match.group(3)
            steps.setdefault(step_index, {}).setdefault(
                "positions", {}
            ).setdefault(entity, {})[axis] = value
            continue
        metadata = re.match(r"^steps\[(\d+)\]\.(.+)$", key)
        if metadata:
            steps.setdefault(int(metadata.group(1)), {}).setdefault(
                "metadata", {}
            )[metadata.group(2)] = value

    for step in steps.values():
        step["positions"] = {
            name: position
            for name, axes in step.get("positions", {}).items()
            if (position := _finite_position(axes)) is not None
        }

    snapshots = {}
    for section_name in ("teaching_snapshot_yaml", "touch_snapshot_yaml"):
        source = _section_text(lines, section_name)
        try:
            document = yaml.safe_load(source) if source else {}
        except yaml.YAMLError:
            document = {}
        snapshots[section_name] = dict(_yaml_positions(document or {}))

    return {
        "path": path,
        "actual": actual,
        "steps": steps,
        "teaching": snapshots["teaching_snapshot_yaml"],
        "touch": snapshots["touch_snapshot_yaml"],
    }


def _waypoint_number(name):
    match = re.fullmatch(r"waypoints\[(\d+)\]", name)
    return int(match.group(1)) if match else None


def _set_equal_3d(axis, points):
    if not points:
        return
    ranges = [
        (min(point[index] for point in points), max(point[index] for point in points))
        for index in range(3)
    ]
    centers = [(low + high) * 0.5 for low, high in ranges]
    radius = max(max(high - low for low, high in ranges) * 0.55, 1.0)
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _decorate_3d(axis, title):
    axis.set_title(title)
    axis.set_xlabel("World X (mm)")
    axis.set_ylabel("World Y (mm)")
    axis.set_zlabel("World Z (mm)")
    axis.grid(True, alpha=0.25)


def plot_weld_trajectory_3d(path, output=None, show=True):
    """Save a two-view 3D plot of every trajectory/pose embedded in a log."""
    data = parse_weld_trajectory_log(path)
    actual_mm = [
        (sample["x_m"] * 1000.0, sample["y_m"] * 1000.0, sample["z_m"] * 1000.0)
        for sample in data["actual"]
    ]
    step_positions = {
        index: {
            name: tuple(value * 1000.0 for value in position)
            for name, position in step.get("positions", {}).items()
        }
        for index, step in data["steps"].items()
    }
    teaching_mm = {
        name: tuple(value * 1000.0 for value in position)
        for name, position in data["teaching"].items()
    }
    touch_mm = {
        name: tuple(value * 1000.0 for value in position)
        for name, position in data["touch"].items()
    }
    all_commanded = [
        point for positions in step_positions.values() for point in positions.values()
    ]
    if not actual_mm and not all_commanded and not teaching_mm and not touch_mm:
        raise ValueError(f"no 3D trajectory or pose data found in {data['path']}")

    try:
        _prepare_mplot3d()
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is unavailable; run with ~/ros2_ws/.venv/bin/python"
        ) from error

    figure = plt.figure(figsize=(17, 8), constrained_layout=True)
    full_axis = figure.add_subplot(1, 2, 1, projection="3d")
    detail_axis = figure.add_subplot(1, 2, 2, projection="3d")
    result = "unknown"
    try:
        sections, _samples = parse_weld_feedback_log(data["path"])
        result = sections.get("header", {}).get("result", "unknown")
    except ValueError:
        pass
    figure.suptitle(
        f"Weld trajectories · {data['path'].name} · result={result}"
    )

    full_points = []
    detail_points = []
    command_colors = ("#2563eb", "#0891b2", "#7c3aed", "#ea580c", "#16a34a")
    for order, index in enumerate(sorted(step_positions)):
        positions = step_positions[index]
        metadata = data["steps"][index].get("metadata", {})
        stage = metadata.get("weld_scenario_stage", f"step {index + 1}")
        waypoints = sorted(
            (
                (_waypoint_number(name), point)
                for name, point in positions.items()
                if _waypoint_number(name) is not None
            ),
            key=lambda item: item[0],
        )
        color = command_colors[order % len(command_colors)]
        if waypoints:
            points = [point for _number, point in waypoints]
            xs, ys, zs = zip(*points)
            for axis in (full_axis, detail_axis):
                axis.plot(
                    xs, ys, zs, marker="o", markersize=3, linestyle="--",
                    linewidth=1.2, color=color, alpha=0.75,
                    label=f"command {index + 1}: {stage}" if axis is full_axis else None,
                )
            full_points.extend(points)
            detail_points.extend(points)
        target = positions.get("target_tcp")
        if target is not None:
            full_axis.scatter(*target, marker="s", s=28, color=color)
            full_axis.text(*target, f" {index + 1}:{stage}", fontsize=7)
            full_points.append(target)

        for name in (
            "safe_approach",
            "approach_lead",
            "lead_start",
            "usable_seam_start",
            "usable_seam_goal",
            "lead_end",
        ):
            point = positions.get(name)
            if point is None:
                continue
            detail_axis.scatter(*point, marker="D", s=35, color=color)
            detail_axis.text(*point, f" {index + 1}:{name}", fontsize=7)
            detail_points.append(point)

    if actual_mm:
        xs, ys, zs = zip(*actual_mm)
        elapsed = [sample["elapsed_s"] for sample in data["actual"]]
        for axis in (full_axis, detail_axis):
            axis.plot(xs, ys, zs, color="#111827", linewidth=2.2, label="actual TCP")
        colored = detail_axis.scatter(
            xs, ys, zs, c=elapsed, cmap="turbo", s=10, alpha=0.85,
            label="actual TCP samples",
        )
        figure.colorbar(colored, ax=detail_axis, shrink=0.68, label="Elapsed (s)")
        full_points.extend(actual_mm)
        detail_points.extend(actual_mm)

    if teaching_mm:
        points = list(teaching_mm.values())
        xs, ys, zs = zip(*points)
        full_axis.scatter(
            xs, ys, zs, marker="^", s=34, facecolors="none",
            edgecolors="#6b7280", label="teaching poses",
        )
        for name, point in teaching_mm.items():
            full_axis.text(*point, f" {name.split('.')[0]}", fontsize=7, color="#4b5563")
        full_points.extend(points)

    if touch_mm:
        points = list(touch_mm.values())
        xs, ys, zs = zip(*points)
        for axis in (full_axis, detail_axis):
            axis.scatter(xs, ys, zs, marker="x", s=40, color="#db2777", label="touch poses")
        for name, point in touch_mm.items():
            detail_axis.text(*point, f" {name}", fontsize=7, color="#9d174d")
        full_points.extend(points)
        detail_points.extend(points)

    _decorate_3d(full_axis, "Complete workflow: commands + teaching + touch + actual")
    _decorate_3d(detail_axis, "Weld detail: measured TCP + seam/lead/safe geometry")
    _set_equal_3d(full_axis, full_points)
    _set_equal_3d(detail_axis, detail_points or full_points)
    for axis in (full_axis, detail_axis):
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), loc="best", fontsize=7)

    path = data["path"]
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else path.with_name(path.stem + ".trajectory_3d.png")
    )
    figure.savefig(output_path, dpi=170)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def plot_all_weld_trajectories_3d(paths, output=None, show=True):
    """Overlay every log in World coordinates and start-aligned coordinates."""
    paths = [Path(path).expanduser().resolve() for path in paths]
    parsed = []
    for path in sorted(paths, key=lambda item: item.name):
        data = parse_weld_trajectory_log(path)
        actual = [
            (
                sample["x_m"] * 1000.0,
                sample["y_m"] * 1000.0,
                sample["z_m"] * 1000.0,
            )
            for sample in data["actual"]
        ]
        commanded = []
        for index in sorted(data["steps"]):
            positions = data["steps"][index].get("positions", {})
            waypoints = sorted(
                (
                    (_waypoint_number(name), position)
                    for name, position in positions.items()
                    if _waypoint_number(name) is not None
                ),
                key=lambda item: item[0],
            )
            commanded.extend(
                tuple(value * 1000.0 for value in position)
                for _number, position in waypoints
            )
        trajectory = actual or commanded
        if trajectory:
            parsed.append((path, trajectory, bool(actual)))
    if not parsed:
        raise ValueError("no 3D trajectories found in the selected logs")

    try:
        _prepare_mplot3d()
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is unavailable; run with ~/ros2_ws/.venv/bin/python"
        ) from error

    figure = plt.figure(figsize=(19, 10), constrained_layout=True)
    world_axis = figure.add_subplot(1, 2, 1, projection="3d")
    aligned_axis = figure.add_subplot(1, 2, 2, projection="3d")
    figure.suptitle(
        f"All weld trajectories · {len(parsed)}/{len(paths)} logs"
    )
    color_map = plt.get_cmap("turbo")
    world_points = []
    aligned_points = []
    handles = []
    labels = []
    denominator = max(1, len(parsed) - 1)
    for index, (path, trajectory, measured) in enumerate(parsed):
        color = color_map(index / denominator)
        xs, ys, zs = zip(*trajectory)
        style = "-" if measured else "--"
        width = 1.8 if measured else 1.2
        handle, = world_axis.plot(
            xs, ys, zs, linestyle=style, linewidth=width,
            color=color, alpha=0.82,
        )
        world_axis.scatter(
            xs[0], ys[0], zs[0], marker="o", s=14, color=color
        )
        world_axis.scatter(
            xs[-1], ys[-1], zs[-1], marker="x", s=22, color=color
        )
        origin = trajectory[0]
        aligned = [
            (
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            )
            for point in trajectory
        ]
        axs, ays, azs = zip(*aligned)
        aligned_axis.plot(
            axs, ays, azs, linestyle=style, linewidth=width,
            color=color, alpha=0.82,
        )
        aligned_axis.scatter(
            axs[-1], ays[-1], azs[-1], marker="x", s=22, color=color
        )
        world_points.extend(trajectory)
        aligned_points.extend(aligned)
        handles.append(handle)
        labels.append(
            path.stem + ("" if measured else " [command only]")
        )

    _decorate_3d(world_axis, "World-frame overlay · ○ start / × end")
    _decorate_3d(
        aligned_axis,
        "Start-aligned comparison · each first TCP = (0, 0, 0)",
    )
    aligned_axis.set_xlabel("ΔX (mm)")
    aligned_axis.set_ylabel("ΔY (mm)")
    aligned_axis.set_zlabel("ΔZ (mm)")
    _set_equal_3d(world_axis, world_points)
    _set_equal_3d(aligned_axis, aligned_points)
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        fontsize=6.5,
        frameon=True,
    )

    if output is None:
        common_parent = paths[0].parent
        output_path = common_parent / "all_weld_trajectories_3d.png"
    else:
        output_path = Path(output).expanduser().resolve()
    figure.savefig(output_path, dpi=170)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def plot_weld_feedback(path, output=None, show=True):
    """Create a two-panel current/voltage plot and return its PNG path."""
    path = Path(path).expanduser().resolve()
    sections, samples = parse_weld_feedback_log(path)
    try:
        _prepare_mplot3d()
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is unavailable; run with ~/ros2_ws/.venv/bin/python"
        ) from error

    times = [sample["elapsed_s"] for sample in samples]
    currents = [sample["current_a"] for sample in samples]
    voltages = [sample["voltage_v"] for sample in samples]
    wcr = [sample["wcr"] for sample in samples]
    set_current = [sample["set_current_a"] for sample in samples]
    set_voltage = [sample["set_voltage_v"] for sample in samples]

    commanded = sections.get("commanded", {})
    result = sections.get("header", {}).get("result", "unknown")
    title = (
        f"Weld feedback · result={result} · "
        f"requested={commanded.get('current_a', '?')} A / "
        f"{commanded.get('voltage', '?')} V"
    )
    figure, (current_axis, voltage_axis) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
    )
    figure.suptitle(title)
    current_axis.plot(times, currents, color="#c62828", label="Current feedback")
    current_axis.plot(
        times, set_current, color="#7f1d1d", linestyle="--", alpha=0.65,
        label="RX current setting echo",
    )
    current_axis.set_ylabel("Current (A)")
    current_axis.grid(True, alpha=0.25)
    current_axis.legend(loc="upper right")

    voltage_axis.plot(times, voltages, color="#1565c0", label="Voltage feedback")
    voltage_axis.plot(
        times, set_voltage, color="#1e3a8a", linestyle="--", alpha=0.65,
        label="RX voltage setting echo",
    )
    voltage_axis.fill_between(
        times,
        0,
        1,
        where=[bool(value) for value in wcr],
        transform=voltage_axis.get_xaxis_transform(),
        color="#22c55e",
        alpha=0.10,
        label="WCR detected",
    )
    voltage_axis.set_xlabel("Elapsed time (s)")
    voltage_axis.set_ylabel("Voltage (V)")
    voltage_axis.grid(True, alpha=0.25)
    voltage_axis.legend(loc="upper right")

    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else path.with_suffix(".png")
    )
    figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="*",
    )
    parser.add_argument("--output")
    parser.add_argument("--trajectory-output")
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args(argv)
    logs = arguments.logs or [
        str(Path.home() / "ros2_ws/weld_feedback/latest_weld_feedback.log")
    ]
    if arguments.output and len(logs) != 1:
        parser.error("--output can only be used with one log")
    if arguments.trajectory_output and len(logs) != 1:
        parser.error("--trajectory-output can only be used with one log")
    for log in logs:
        generated = []
        errors = []
        try:
            generated.append(plot_weld_feedback(
                log, arguments.output, show=not arguments.no_show
            ))
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"feedback plot: {error}")
        try:
            generated.append(plot_weld_trajectory_3d(
                log,
                arguments.trajectory_output,
                show=not arguments.no_show,
            ))
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"3D trajectory plot: {error}")
        for output_path in generated:
            print(output_path)
        for error in errors:
            print(f"SKIPPED {Path(log).name} · {error}")
        if not generated:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
