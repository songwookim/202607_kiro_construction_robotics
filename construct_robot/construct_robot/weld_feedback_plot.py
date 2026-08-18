"""Plot current and voltage feedback from a weld_action_gui text log."""

import argparse
from pathlib import Path


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


def plot_weld_feedback(path, output=None, show=True):
    """Create a two-panel current/voltage plot and return its PNG path."""
    path = Path(path).expanduser().resolve()
    sections, samples = parse_weld_feedback_log(path)
    try:
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
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args(argv)
    logs = arguments.logs or [
        str(Path.home() / "ros2_ws/weld_feedback/latest_weld_feedback.log")
    ]
    if arguments.output and len(logs) != 1:
        parser.error("--output can only be used with one log")
    for log in logs:
        output = plot_weld_feedback(
            log, arguments.output, show=not arguments.no_show
        )
        print(output)


if __name__ == "__main__":
    main()
