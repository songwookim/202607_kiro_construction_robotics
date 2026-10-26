"""Platform-independent motion types shared by the ports (scan_l0.ports), the run (scan_l0.autoteach / teach_1g) and the MoveIt
adapter (hongik_scan_ros.moveit_adapter): the motion settings of one session, the result of one motion, the motion error and the
measured TCP speed of a motion. Nothing here talks to a robot controller.

  MotionConfig      v_eff_max_mps / a_eff_max_mps2 = the platform's declared TCP speed / acceleration limits (moveit_jtc.limits);
                    the commanded TCP speed is the effective speed (MoveIt adapter). default_timeout_s = the timeout floor of a motion
                    (the adapter derives a longer one from the planned trajectory). progress_s / progress_cb = operator feedback while
                    a motion runs (a slow move looks stationary).
  MotionResult      what one move_linear / move_joint did (times on the host monotonic clock, the measured pose samples, events).
  SpeedMeasurement  the TCP speed measured from the pose samples of one motion (cruise = median of the interval speeds >= 60 % of
                    the maximum).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = ["MotionConfig", "MotionResult", "MotionError", "PlatformUnavailable", "SpeedMeasurement"]


class MotionError(RuntimeError):
    """A motion was refused or failed (the message is the reason)."""


class PlatformUnavailable(MotionError):
    """The platform's planning / checking service did not answer (an infrastructure failure, not a property of the candidate being
    checked): a search over candidates stops at once instead of rejecting every candidate in turn (KIRO field 2026-10-14)."""


@dataclass
class SpeedMeasurement:
    v_measured_mps: Optional[float] = None   # cruise estimate (median of interval speeds >= 60 % of max)
    v_mean_mps: Optional[float] = None       # path length / duration (includes accel/decel)
    v_max_mps: Optional[float] = None
    path_length_m: float = 0.0
    duration_s: float = 0.0
    n_samples: int = 0
    source: str = "none"                      # pose source of the samples (MoveIt adapter: 'ros_tf')

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class MotionConfig:
    default_timeout_s: float = 120.0         # timeout floor of one motion [s]
    v_eff_max_mps: float = 0.05              # platform TCP speed limit (commanded = effective)
    a_eff_max_mps2: Optional[float] = None   # platform TCP acceleration limit; None = not declared
    progress_s: float = 0.0                  # > 0: call progress_cb every progress_s during a motion
    progress_cb: Optional[Callable[[str], None]] = None


@dataclass
class MotionResult:
    script: str
    started: bool = False
    finished: bool = False
    aborted: bool = False
    reason: str = ""
    elapsed_s: float = 0.0
    t_send_ns: int = 0
    t_ack_ns: int = 0
    t_start_ns: int = 0   # host time at which the motion started (goal accepted)
    t_stop_ns: int = 0    # host time at which the motion ended
    v_cmd: float = 0.0    # commanded cruise speed [mm/s]
    a_cmd: float = 0.0
    timeout_s: float = 0.0   # timeout actually applied
    responses: list = field(default_factory=list)
    default_speed: float = 0.0            # speed scale of the platform (MoveIt adapter: 1.0 - commanded = effective)
    policy: str = ""                      # motion backend kind that executed it
    v_eff_expected_mps: float = 0.0       # planned effective TCP speed
    samples: list = field(default_factory=list)      # measured pose samples during the motion
    monitor_events: list = field(default_factory=list)
    v_measured_mps: Optional[float] = None
    speed_measurement: Optional[SpeedMeasurement] = None

    @property
    def ok(self) -> bool:
        return self.finished and not self.aborted

    @property
    def motion_duration_s(self) -> float:
        return (self.t_stop_ns - self.t_start_ns) * 1e-9 if self.started and self.finished else 0.0

    def as_dict(self) -> dict:
        return {
            "script": self.script, "started": self.started, "finished": self.finished, "aborted": self.aborted,
            "reason": self.reason, "elapsed_s": self.elapsed_s, "t_send_ns": self.t_send_ns,
            "t_ack_ns": self.t_ack_ns, "t_start_ns": self.t_start_ns, "t_stop_ns": self.t_stop_ns,
            "v_cmd": self.v_cmd, "a_cmd": self.a_cmd, "timeout_s": self.timeout_s,
            "default_speed": self.default_speed, "policy": self.policy,
            "v_eff_expected_mps": self.v_eff_expected_mps,
            "n_samples": len(self.samples), "monitor_events": list(self.monitor_events),
            "v_measured_mps": self.v_measured_mps,
            "speed_measurement": self.speed_measurement.as_dict() if self.speed_measurement else None,
        }
