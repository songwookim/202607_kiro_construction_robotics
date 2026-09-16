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
    """Plot every recorded actual-TCP sample as one World-frame 3D point."""
    data = parse_weld_trajectory_log(path)
    actual_mm = [
        (sample["x_m"] * 1000.0, sample["y_m"] * 1000.0, sample["z_m"] * 1000.0)
        for sample in data["actual"]
    ]
    if not actual_mm:
        raise ValueError(f"no recorded actual TCP samples found in {data['path']}")

    try:
        _prepare_mplot3d()
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is unavailable; run with ~/ros2_ws/.venv/bin/python"
        ) from error

    figure = plt.figure(figsize=(11, 9), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    result = "unknown"
    try:
        sections, _samples = parse_weld_feedback_log(data["path"])
        result = sections.get("header", {}).get("result", "unknown")
    except ValueError:
        pass
    figure.suptitle(
        f"Actual TCP samples · {data['path'].name} · result={result}"
    )
    xs, ys, zs = zip(*actual_mm)
    elapsed = [sample["elapsed_s"] for sample in data["actual"]]
    colored = axis.scatter(
        xs, ys, zs, c=elapsed, cmap="turbo", marker="o", s=10,
        alpha=0.8, edgecolors="none",
    )
    figure.colorbar(colored, ax=axis, shrink=0.72, label="Elapsed (s)")
    _decorate_3d(axis, f"Recorded TCP point density · {len(actual_mm)} samples")
    _set_equal_3d(axis, actual_mm)

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
    """Overlay every recorded actual-TCP sample from all logs as 3D points."""
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
        if actual:
            parsed.append((path, actual))
    if not parsed:
        raise ValueError("no recorded actual TCP samples found in the selected logs")

    try:
        _prepare_mplot3d()
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is unavailable; run with ~/ros2_ws/.venv/bin/python"
        ) from error

    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    world_axis = figure.add_subplot(111, projection="3d")
    figure.suptitle(
        f"All weld trajectories · {len(parsed)}/{len(paths)} logs"
    )
    color_map = plt.get_cmap("turbo")
    world_points = []
    handles = []
    labels = []
    denominator = max(1, len(parsed) - 1)
    for index, (path, trajectory) in enumerate(parsed):
        color = color_map(index / denominator)
        xs, ys, zs = zip(*trajectory)
        handle = world_axis.scatter(
            xs, ys, zs, marker="o", s=8, color=color,
            alpha=0.70, edgecolors="none",
        )
        world_points.extend(trajectory)
        handles.append(handle)
        labels.append(f"{path.stem} · {len(trajectory)} samples")

    _decorate_3d(
        world_axis,
        f"Recorded TCP point density · {sum(len(item[1]) for item in parsed)} samples",
    )
    _set_equal_3d(world_axis, world_points)
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
