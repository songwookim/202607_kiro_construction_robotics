"""Named welding/cleaner teaching state and validation, independent of widgets."""
from dataclasses import dataclass, field
import math
from pathlib import Path


TEACHING_POSES = {
    "robot_start": "1 · Initial pose",
    "weld_wait": "2 · Weld wait pose",
    "weld_start_wait": "3 · Weld start wait pose",
    "weld_start": "4 · Reference TCP 1 / Weld start",
    "weld_goal_wait": "5 · Weld goal wait pose",
    "weld_end": "6 · Reference TCP 2 / Weld goal",
    "weld_finish": "7 · Weld end pose",
}

# Every Named TCP Teaching execution is contact guarded.  Planning remains
# unguarded because it does not command physical motion.
TOUCH_GUARDED_TEACHING_POSES = frozenset(TEACHING_POSES)

# Corrected seam teaching poses combine sensed/corrected XYZ with the
# orientation originally captured for that individual named pose.
TCP_POSE_TEACHING_POSES = frozenset((
    "weld_start",
    "weld_end",
))

JOINT_RECALL_TEACHING_POSES = frozenset(TEACHING_POSES) - TCP_POSE_TEACHING_POSES

SEAM_REFERENCE_TEACHING_POSES = frozenset((
    "weld_start_wait", "weld_start", "weld_goal_wait", "weld_end", "weld_finish",
))


class TeachingState:
    """Named TCP/joint snapshots and provenance, independent of widgets."""

    def __init__(self, names):
        self.poses = {name: None for name in names}
        self.provenance = {}
        self.selected_name = next(iter(self.poses), None)

    def select(self, name):
        if name not in self.poses:
            raise ValueError(f"Unknown teaching pose: {name}")
        self.selected_name = name
        return name

    def store(self, name, snapshot, provenance=None):
        if name not in self.poses:
            raise ValueError(f"Unknown teaching pose: {name}")
        self.poses[name] = snapshot
        if provenance is None:
            self.provenance.pop(name, None)
        else:
            self.provenance[name] = provenance


@dataclass
class CleanerTeachingState:
    """Cleaner teaching folder, ordered tokens, and selected pose."""

    folder: Path
    tokens: list = field(default_factory=list)
    selected: str = "start"

    def set_order(self, tokens):
        self.tokens = list(tokens)


def cleaner_output_step(name):
    channel, value = name.split(":", 1)
    if channel not in ("DO5", "DO6", "DO7"):
        raise ValueError("Cleaner output must be DO5/6/7")
    if value not in ("ON", "OFF"):
        duration = float(value)
        if not math.isfinite(duration) or not 0 < duration <= 30:
            raise ValueError("Pulse duration must be 0..30 seconds")
    return int(channel[2:]), value
