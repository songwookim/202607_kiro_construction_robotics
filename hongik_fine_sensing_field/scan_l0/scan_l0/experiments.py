"""1G core 실행 뒤 실험 단계 (작업 파일 `experiments:` 블록이 있을 때만; DESIGN_ENDS_r1 §3.2 E3, §3.4 E5, 2026-10-12 아침 실기).

실행이 추정·세계 (끝을 안다) 를 낸 뒤, 같은 목표 (액션) 안에서 같은 이동 경로 (D3 포락선 → D4 MoveIt → 확인 ~/confirm_move) 로 두
실험을 한다. 실행의 상태·추정·출력 (sections·weld_plan) 은 바꾸지 않는다 — 결과는 <out_dir>/experiments/ 에만 (원자료 =
verification/legs 와 같은 npz 규약, 요약 JSON). 취소 = 실험 정지 (실행 결과는 그대로), 운전자 거절도 같다.

실험 3 (standoff) — 절대 진행 방향 / 레이저 면 기울기 w_x:
  같은 끝을 판 거리 D 여러 개에서 전체 패스 (끝 너머 → 끝 너머, 두 방향) 로 읽는다. 피치·툴 자세·속도·오버스캔은 실행의 패스와 같다
  (Teach1G._pass 그대로 — 판 거리만 다르다). D 가 센서 거리 창 밖이면 (모형: 판 상면 ≥ 가까운 한계 + 도착 허용, 루트 ≤ 먼 한계 − 도착
  허용) 그 D 는 이유를 남기고 건너뛴다. 끝 위치가 D 에 선형이면 기울기 = tan w_x (오프라인 exp3_analyze).

실험 4 (end_views) — 정지 회전 끝 뷰:
  끝마다 툴을 판 법선 둘레로 ψ 돌려 (손목2 부호 유지 — 토치 뒤집기 아님) 정지 프레임을 잡는다. 판 끝이 프로파일 한 장 안의 단차
  (판 상면 → 아래 면) 로 보여 시각·출렁임과 무관하다. 자세 (모형·설계·캘리브에서):
    - 레이저 선 가운데 = 그 판의 끝 (실행의 끝 모형 판독, 없으면 추정 끝) 의 판 상면 위 점. 옆 위치 = 패스 프로파일이 그 판에서 보는
      상면 띠 (설계 상부 코너 … 패스 거리 시야 반폭) 의 가운데 — 패스의 끝 판독이 쓰는 같은 띠 (곧은 선 하나는 같은 직선 위의 두 판 끝을
      한꺼번에 가로지를 수 없다: 판마다 한 자세);
    - 선 = 패스 자세를 판 법선 둘레로 ψ (그 점을 지나는 축) — 판 거리·피치 (판에 대한 기울기) 는 패스와 같다;
    - 회전 부호 = MoveIt IK (플랫폼 reach_candidates: 물러난 자세 → 뷰 직선 추종) 로 손목2 부호가 지금과 같은 해 중 지금 관절에서
      관절 이동이 가장 적은 쪽.
  반복마다: 물러난 자세 (판 법선을 따라 retreat_m 위) → 곧게 내려가기 → 안정 대기 (서보 안정 모형 scan_core.settling: 실행 구간들의
  잰 안정 시간 + k·σ, 제어기 지연 사전) → N 프레임 grab → 자세·프레임 기록. 닿지 않거나 충돌하는 자세 (IK·D3·D4 거부:
  보낸 것 없음) 는 이유를 남기고 건너뛴다 (오류 아님). 확인 모드는 실행의 것과 블록의 confirm (있으면) 중 더 엄격한 쪽 — 새 모션의 첫 사용은
  each.

순서 (이동 최소): 블록의 키 순서; standoff 는 D 마다 두 방향을 앞 패스가 끝난 끝에서 시작하게; end_views 는 지금 자세에 가까운 끝부터,
끝마다 각도 → 판 (각도마다 판 순서를 뒤집어 옆 이동을 줄인다) → 반복.
"""
from __future__ import annotations

import json
import math
import pickle
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from scan_core.frames import unit

__all__ = ["ExperimentsPlan", "StandoffPlan", "EndViewPlan", "parse_experiments", "plate_window", "end_view_pose", "view_target",
           "CONFIRM_STRICTNESS", "EXPERIMENT_KEYS"]

EXPERIMENT_KEYS = ("standoff", "end_views")
STANDOFF_KEYS = ("plate_distances_m", "directions")
END_VIEW_KEYS = ("angles_deg", "repeats", "frames", "retreat_m", "ends", "plates", "confirm")
CONFIRM_STRICTNESS = ("none", "auto", "scan", "large", "each")       # 뒤로 갈수록 더 많이 묻는다 (autoteach.confirm_needed)
_DIST_RE = re.compile(r"^\s*(pass|transit)\s*(?:([+-])\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?))?\s*$")


@dataclass
class StandoffPlan:
    plate_distances_m: list                 # 요청 (해석 뒤 절대값, m) — 실행의 패스 판 거리 / 트랜싯 판 거리에서 유도한 것 포함
    labels: list                            # 요청 문구 ('pass', 'pass+0.060', 0.4 …)
    directions: list                        # +1 = S->E, −1 = E->S
    derived: bool = False                   # 기본값 (판 거리를 주지 않음): 패스 · (패스+트랜싯)/2 · 트랜싯

    def to_dict(self) -> dict:
        return {"plate_distances_m": [float(v) for v in self.plate_distances_m], "labels": [str(v) for v in self.labels],
                "directions": [int(v) for v in self.directions], "derived": bool(self.derived)}


@dataclass
class EndViewPlan:
    angles_deg: list
    repeats: int
    frames: int
    retreat_m: float
    ends: list
    plates: list                            # 설계 판 이름 'left' | 'right' (설계 bevel_side 와 같은 규약)
    confirm: Optional[str] = None           # 이 실험의 확인 모드 (실행의 것보다 엄격할 때만 바뀐다)
    derived: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"angles_deg": [float(a) for a in self.angles_deg], "repeats": int(self.repeats), "frames": int(self.frames),
                "retreat_m": float(self.retreat_m), "ends": list(self.ends), "plates": list(self.plates), "confirm": self.confirm,
                "derived": dict(self.derived)}


@dataclass
class ExperimentsPlan:
    order: list                             # 블록의 키 순서 ('standoff', 'end_views')
    standoff: Optional[StandoffPlan] = None
    end_views: Optional[EndViewPlan] = None

    def to_dict(self) -> dict:
        return {"order": list(self.order), "standoff": self.standoff.to_dict() if self.standoff else None,
                "end_views": self.end_views.to_dict() if self.end_views else None}

    def summary(self) -> str:
        parts = []
        for k in self.order:
            if k == "standoff" and self.standoff is not None:
                p = self.standoff
                parts.append(f"standoff: plate distances {', '.join(f'{d * 1e3:.1f}' for d in p.plate_distances_m)} mm "
                             f"({', '.join(str(x) for x in p.labels)}{'; derived: pass, mid, transit' if p.derived else ''}), "
                             f"directions {', '.join('S->E' if d > 0 else 'E->S' for d in p.directions)} -> "
                             f"{len(p.plate_distances_m) * len(p.directions)} full passes")
            if k == "end_views" and self.end_views is not None:
                v = self.end_views
                n = len(v.ends) * len(v.angles_deg) * len(v.plates) * v.repeats
                parts.append(f"end_views: ends {'/'.join(v.ends)}, angles {', '.join(f'{a:g}' for a in v.angles_deg)} deg, plates "
                             f"{'/'.join(v.plates)}, {v.repeats} repeats x {v.frames} frames, retreat {v.retreat_m * 1e3:.0f} mm"
                             + (f", confirm {v.confirm}" if v.confirm else "") + f" -> {n} static views"
                             + (f" ({', '.join(f'{k_} {x}' for k_, x in v.derived.items())})" if v.derived else ""))
        return "; ".join(parts) or "nothing"


def _direction(v) -> int:
    if isinstance(v, str):
        t = v.strip().upper().replace(" ", "")
        if t in ("S->E", "SE", "+1", "+"):
            return 1
        if t in ("E->S", "ES", "-1", "-"):
            return -1
        raise ValueError(f"direction {v!r}: use +1 (S->E) or -1 (E->S)")
    iv = int(v)
    if iv not in (1, -1) or float(v) != iv:
        raise ValueError(f"direction {v!r}: use +1 (S->E) or -1 (E->S)")
    return iv


def _distance(v, pass_m: float, transit_m: float) -> float:
    """판 거리 하나: 수 [m] 또는 'pass' / 'transit' (+/− 수 [m]) — 실행의 패스 판 거리·트랜싯 (거친) 판 거리에 상대."""
    if isinstance(v, str):
        m = _DIST_RE.match(v)
        if not m:
            raise ValueError(f"plate distance {v!r}: a number [m] or 'pass' / 'transit' with an optional +/- offset [m] "
                             "(e.g. 'pass+0.06')")
        base = pass_m if m.group(1) == "pass" else transit_m
        off = 0.0 if m.group(2) is None else (1.0 if m.group(2) == "+" else -1.0) * float(m.group(3))
        return float(base) + off
    d = float(v)
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError(f"plate distance {v!r} must be > 0 m")
    return d


def parse_experiments(block, *, pass_plate_m: float, transit_plate_m: float, dwell_frames: int, retreat_default_m: float) -> \
        Optional[ExperimentsPlan]:
    """작업 파일 `experiments:` 블록 → ExperimentsPlan (없거나 빈 블록이면 None). 모르는 키·잘못된 값은 ValueError (이동 전 거부).
    기본값 (유도): standoff 판 거리 = 실행의 패스 판 거리 · 그와 트랜싯 판 거리의 가운데 · 트랜싯 판 거리 (두 정합 거리 + 선형성 점),
    방향 = 두 방향; end_views frames = 실행의 드웰 grab 프레임 수, retreat_m = 획득의 물러남 (job safe_clearance_m), ends = S, E,
    plates = 설계의 두 판. 각도·반복 수는 실험 설계 (입력) — 없으면 거부."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"experiments: must be a mapping with {list(EXPERIMENT_KEYS)}, got {type(block).__name__}")
    if not block:
        return None
    bad = sorted(set(block) - set(EXPERIMENT_KEYS))
    if bad:
        raise ValueError(f"experiments: unknown key(s) {bad} (allowed: {list(EXPERIMENT_KEYS)})")
    order = [k for k in block if k in EXPERIMENT_KEYS and block[k] is not None]
    plan = ExperimentsPlan(order=order)
    if block.get("standoff") is not None:
        b = block["standoff"]
        if not isinstance(b, dict):
            raise ValueError("experiments.standoff must be a mapping")
        bad = sorted(set(b) - set(STANDOFF_KEYS))
        if bad:
            raise ValueError(f"experiments.standoff: unknown key(s) {bad} (allowed: {list(STANDOFF_KEYS)})")
        raw = b.get("plate_distances_m")
        derived = raw is None
        if derived:
            raw = ["pass", f"pass+{0.5 * (transit_plate_m - pass_plate_m):.6f}", "transit"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("experiments.standoff.plate_distances_m must be a non-empty list")
        dists = [_distance(v, pass_plate_m, transit_plate_m) for v in raw]
        dirs = [_direction(v) for v in (b.get("directions") if b.get("directions") is not None else [1, -1])]
        if not dirs or len(set(dirs)) != len(dirs):
            raise ValueError(f"experiments.standoff.directions {b.get('directions')!r}: one or both of +1, -1, each once")
        plan.standoff = StandoffPlan(plate_distances_m=dists, labels=list(raw), directions=dirs, derived=derived)
    if block.get("end_views") is not None:
        b = block["end_views"]
        if not isinstance(b, dict):
            raise ValueError("experiments.end_views must be a mapping")
        bad = sorted(set(b) - set(END_VIEW_KEYS))
        if bad:
            raise ValueError(f"experiments.end_views: unknown key(s) {bad} (allowed: {list(END_VIEW_KEYS)})")
        for req in ("angles_deg", "repeats"):
            if b.get(req) is None:
                raise ValueError(f"experiments.end_views.{req} is required (the experiment's design)")
        angles = [float(a) for a in b["angles_deg"]]
        if not angles or any(not (0.0 < abs(a) <= 90.0) for a in angles):
            raise ValueError(f"experiments.end_views.angles_deg {b['angles_deg']!r}: each 0 < |psi| <= 90 deg (the rotation sign is "
                             "chosen by the arm configuration; 90 deg = the line along the seam)")
        angles = [abs(a) for a in angles]
        reps = int(b["repeats"])
        if reps < 1 or float(b["repeats"]) != reps:
            raise ValueError(f"experiments.end_views.repeats {b['repeats']!r} must be an integer >= 1")
        derived = {}
        frames = b.get("frames")
        if frames is None:
            frames, derived["frames"] = int(dwell_frames), "the run's dwell grab"
        frames = int(frames)
        if frames < 1:
            raise ValueError(f"experiments.end_views.frames {frames} must be >= 1")
        retreat = b.get("retreat_m")
        if retreat is None:
            retreat, derived["retreat_m"] = float(retreat_default_m), "the acquisition retreat (job safe_clearance_m)"
        retreat = float(retreat)
        if not (math.isfinite(retreat) and retreat > 0.0):
            raise ValueError(f"experiments.end_views.retreat_m {retreat!r} must be > 0 m")
        ends = [str(e).strip().upper() for e in (b.get("ends") or ["S", "E"])]
        if not ends or any(e not in ("S", "E") for e in ends) or len(set(ends)) != len(ends):
            raise ValueError(f"experiments.end_views.ends {b.get('ends')!r}: S and/or E, each once")
        plates = [str(p).strip().lower() for p in (b.get("plates") or ["left", "right"])]
        if not plates or any(p not in ("left", "right") for p in plates) or len(set(plates)) != len(plates):
            raise ValueError(f"experiments.end_views.plates {b.get('plates')!r}: left and/or right (the design's plates), each once")
        conf = b.get("confirm")
        if conf is not None and str(conf) not in CONFIRM_STRICTNESS:
            raise ValueError(f"experiments.end_views.confirm {conf!r} must be one of {list(CONFIRM_STRICTNESS)}")
        plan.end_views = EndViewPlan(angles_deg=angles, repeats=reps, frames=frames, retreat_m=retreat, ends=ends, plates=plates,
                                     confirm=None if conf is None else str(conf), derived=derived)
    return plan if (plan.standoff is not None or plan.end_views is not None) else None


def stricter_confirm(a: str, b: Optional[str]) -> str:
    """두 확인 모드 중 더 많이 묻는 쪽."""
    if b is None:
        return a
    return a if CONFIRM_STRICTNESS.index(a) >= CONFIRM_STRICTNESS.index(b) else b


def plate_window(plate_m: float, depth_m: float, pitch_deg: float, *, near_m: float, far_m: float, pose_tol_m: float) -> Optional[str]:
    """센서 거리 창 모형 (scan_l0.sensor_view 와 같은 양): 패스 프로파일의 판 상면 (광축 거리 plate_m) 이 가까운 한계 + 도착 허용 안쪽,
    그루브 바닥 (plate_m + 깊이 / cos 피치) 이 먼 한계 − 도착 허용 안쪽이어야 한다. 밖이면 사유 (영어), 안이면 None."""
    d = max(0.0, float(depth_m)) / max(math.cos(math.radians(float(pitch_deg))), 1e-9)
    e = float(pose_tol_m)
    lo, hi = float(near_m) + e, float(far_m) - e - d
    if plate_m < lo:
        return (f"plate distance {plate_m * 1e3:.1f} mm is closer than the sensor's near limit {near_m * 1e3:.1f} mm + arrival "
                f"tolerance {e * 1e3:.1f} mm")
    if plate_m > hi:
        return (f"plate distance {plate_m * 1e3:.1f} mm puts the groove bottom ({(plate_m + d) * 1e3:.1f} mm) beyond the sensor's far "
                f"limit {far_m * 1e3:.1f} mm - arrival tolerance {e * 1e3:.1f} mm")
    return None


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    a = unit(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def end_view_pose(T_pass_opt: np.ndarray, n_hat: np.ndarray, P_target: np.ndarray, psi_rad: float) -> np.ndarray:
    """패스 광학 자세 T_pass_opt 를 판 법선 n̂ 둘레로 ψ 돌려, 그 광축이 판 상면 평면 (P_target 을 지나고 n̂ 에 수직) 과 만나는 점을 P_target
    에 둔 광학 자세. 판 거리 (광축을 따라 원점 → 상면) 와 판에 대한 기울기 (피치) 는 패스와 같다."""
    T0 = np.asarray(T_pass_opt, dtype=float)
    n = unit(np.asarray(n_hat, dtype=float))
    P = np.asarray(P_target, dtype=float).reshape(3)
    o, Z = T0[:3, 3], T0[:3, 2]
    den = float(Z @ n)
    if abs(den) < 1e-9:
        raise ValueError("the pass's optical axis is parallel to the plate top")
    t = float((P - o) @ n) / den
    Q0 = o + t * Z
    R = _rot(n, float(psi_rad))
    T = np.eye(4)
    T[:3, :3] = R @ T0[:3, :3]
    T[:3, 3] = P + R @ (o - Q0)
    return T


def view_target(r_end: np.ndarray, u_hat: np.ndarray, n_hat: np.ndarray, side_w: np.ndarray, corner_m: float, half_fov_m: float,
                top_h_m: float) -> tuple:
    """레이저 선 가운데 목표 (판 상면 위, 판 끝): 루트 선 점 r_end 에서 그 판 쪽 ŵ 로 패스 프로파일이 그 판에서 보는 상면 띠 (설계 상부 코너
    |corner| … 시야 반폭) 의 가운데, 판 상면 높이 top_h (루트 선에서 n̂ 방향). → (점, 옆 거리 m)."""
    w = 0.5 * (abs(float(corner_m)) + float(half_fov_m))
    return (np.asarray(r_end, dtype=float) + unit(side_w) * w + unit(n_hat) * float(top_h_m), w)


# ====================================================================================================================
# 실행: Teach1G 의 원시 (이동·기록·작업 프로세스) 를 그대로 쓴다
# ====================================================================================================================
class PostRunExperiments:
    """실행 뒤 실험 단계 하나 (모듈 docstring). t = 끝난 Teach1G (추정·세계·센서·작업 프로세스가 살아 있다)."""

    def __init__(self, t, plan: ExperimentsPlan, est):
        self.t = t
        self.plan = plan
        self.est = est
        self.dir = Path(t.out) / "experiments"
        self.n_leg = 0
        self.records: dict = {"standoff": [], "end_views": []}
        self.stopped: Optional[dict] = None
        self.t_start = None

    # ---- 기록 ------------------------------------------------------------------------------------------------------
    def _say(self, msg: str, **data) -> None:
        self.t._decide("experiment", msg, **data)

    def _leg_dir(self, name: str) -> Path:
        self.n_leg += 1
        return self.dir / "legs" / f"{self.n_leg:02d}_{name}"

    def _frame(self) -> dict:
        """실행의 심 틀 (최종 추정): S0 (s = 0 의 루트 선 점), û, n̂ (잰 판 법선), ŵ = n̂ × û, 끝, 판별 끝 판독."""
        est, t = self.est, self.t
        u = unit(np.asarray(est.u, dtype=float))
        n = np.asarray(t._n_hat, dtype=float)
        n = unit(n - float(n @ u) * u)
        w = np.cross(n, u)
        ef = (est.plate or {}).get("end_faces") or {}
        return {"S0_m": np.asarray(est.r(0.0), dtype=float).tolist(), "u": u.tolist(), "n": n.tolist(), "w": w.tolist(),
                "w0": np.asarray(t._f0()["w0"], dtype=float).tolist(),
                "ends": {X: {"s_m": float(e.s_m), "sigma_m": float(e.sigma_m), "point_m": np.asarray(e.point_m, dtype=float).tolist(),
                             "two_sided": bool(e.two_sided)} for X, e in (est.ends or {}).items()},
                "plates": ef.get("plates") or {}, "bevel_sign_w": int(est.bevel_sign_w),
                "left_plate_side_w0": t._left_plate_side(est)}

    def _write_summary(self) -> Path:
        t = self.t
        doc = {"schema": "g1core.experiments/1", "plan": self.plan.to_dict(),
               "run": {"out_dir": str(t.out), "status": t.res.status, "reason": t.res.reason,
                       "elapsed_s_before_experiments": self.t_start, "estimates": len(t.est_history),
                       "estimate_file": str(t.vdir / "estimates" / f"est_{len(t.est_history)}.json"),
                       "run_legs": [str(p) for p in t.legs], "calib": t.calib.source, "k": float(t.k),
                       "speed_mps": float(t._scan_speed()), "profile_fps": float(t.job0.profile_fps),
                       "pass_plate_m": float(t.pass_plate_m), "transit_plate_m": float(t._coarse_plate_m())},
               "frame": self._frame(), "standoff": self.records["standoff"], "end_views": self.records["end_views"],
               "stopped": self.stopped, "elapsed_s": round(t._clock.monotonic() - t.t0 - float(self.t_start or 0.0), 3)}
        from . import g1_worker as _gw
        from scan_core.g1_types import jsonable
        p = self.dir / "summary.json"
        _gw.atomic_write_text(p, json.dumps(jsonable(doc), indent=1, ensure_ascii=False, default=str))
        return p

    # ---- 실행 ------------------------------------------------------------------------------------------------------
    def run(self) -> dict:
        """실험 단계 전체. 예외를 밖으로 내지 않는다 (실행 결과는 이미 정해졌다) — 정지·오류는 요약과 로그에."""
        from .autoteach import _Abort
        t = self.t
        self.t_start = round(t._clock.monotonic() - t.t0, 3)
        self.dir.mkdir(parents=True, exist_ok=True)
        t._set_phase("experiment")
        self._say(f"experiments after the run ({t.res.status}, {self.t_start:.1f} s; the run's status, estimate and outputs are not "
                  f"changed): {self.plan.summary()} -> {self.dir}", plan=self.plan.to_dict())
        try:
            with open(self.dir / "prior.pkl", "wb") as f:                # 오프라인 재추정 (exp3_analyze) 의 사전 — 실행과 같은 것
                pickle.dump(t.prior, f)
        except Exception as e:  # noqa: BLE001 - offline convenience only
            t.res.notes.append(f"experiments: prior not pickled ({type(e).__name__}: {e})")
        try:
            t._apply_world("experiments: the run's final world")       # 출력 단계가 장면을 비웠다 (도달성 검사) — 다시 건다
            for kind in self.plan.order:
                if kind == "standoff" and self.plan.standoff is not None:
                    self._standoff(self.plan.standoff)
                elif kind == "end_views" and self.plan.end_views is not None:
                    self._end_views(self.plan.end_views)
        except _Abort as e:
            self.stopped = {"status": e.status, "reason": e.reason, "category": e.category}
            t._p(f"EXPERIMENTS STOPPED ({e.status}): {e.reason} - the run's result is unchanged ({t.res.status})")
            t.res.notes.append(f"experiments stopped ({e.status}): {e.reason}")
            self._safe_stop(e.reason, bool(e.safety))
        except Exception as e:  # noqa: BLE001 - a fault of the experiment phase never changes the run's result
            self.stopped = {"status": "ERROR", "reason": f"{type(e).__name__}: {e}", "category": "internal"}
            t._p(f"EXPERIMENTS STOPPED (ERROR): {type(e).__name__}: {e} - the run's result is unchanged ({t.res.status})")
            t.res.notes.append(f"experiments stopped (internal error): {traceback.format_exc(limit=8)}")
            self._safe_stop(f"experiments internal error: {type(e).__name__}: {e}", True)
        finally:
            try:
                p = self._write_summary()
            except Exception as e:  # noqa: BLE001
                t.res.notes.append(f"experiments summary not written: {type(e).__name__}: {e}")
                p = None
        n_s = sum(1 for r in self.records["standoff"] if r.get("status") == "done")
        n_v = sum(1 for r in self.records["end_views"] if r.get("status") == "done")
        sk = [r for r in self.records["standoff"] + self.records["end_views"] if r.get("status") == "skipped"]
        self._say(f"experiments {'stopped' if self.stopped else 'done'}: {n_s} plate-distance pass(es), {n_v} end view(s) recorded, "
                  f"{len(sk)} skipped" + (f" ({'; '.join(sorted({str(r.get('why'))[:120] for r in sk})[:3])})" if sk else "")
                  + f", {t._clock.monotonic() - t.t0 - self.t_start:.1f} s -> {p}")
        return {"stopped": self.stopped, "summary": str(p) if p else None}

    def _safe_stop(self, reason: str, safety: bool) -> None:
        """실험 정지의 안전 부분 = 실행 중단 경로 (AutoTeach._abort_path) 의 것 그대로, 상태만 바꾸지 않는다: 센서 정지, 정지 기록. 정지
        명령은 보내지 않는다 — 이 패키지의 정지는 액션 취소뿐이고 (사용자 2026-10-11), 실패한 이동은 JTC 안에서 이미 끝났다."""
        t = self.t
        sess = t.sess
        if sess is not None and sess.sensor is not None and sess.sensor.client is not None:
            try:
                if sess.sensor.recording:
                    sess.sensor.stop_recording(settle_s=0.1)
            except Exception as e:  # noqa: BLE001
                t.res.notes.append(f"experiments: sensor stop failed: {e}")
        t.res.notes.append(f"experiments stopped ({reason}): no stop command (the only stop of this package is the action cancel)")
        t.res.estop_record = t._latch_record_of_this_run()

    # ---- 실험 3: 판 거리 -------------------------------------------------------------------------------------------
    def _standoff(self, sp: StandoffPlan) -> None:
        from .autoteach import STATUS_REFUSED, _Abort
        t, est = self.t, self.est
        qd = t._plan_quantities(est)
        depth = float(qd["depth_m"])
        sv = t._sv()
        near = float(sv.near_m if sv.near_m is not None else sv.z_near_m)
        pitch = float(t.job0.pitch_deg)
        sig = {"S": qd["sigma_plan_S_m"], "E": qd["sigma_plan_E_m"]}
        # 패스마다 지금 툴이 있는 끝에서 시작하는 방향부터 (앞 패스가 끝난 끝에서 다음 패스가 시작 — 재배치 최소)
        o_now = (t._T_now() @ t.calib.T_flange_opt)[:3, 3]
        at_end = "S" if np.linalg.norm(o_now - est.r(qd["e_S_m"])) <= np.linalg.norm(o_now - est.r(qd["e_E_m"])) else "E"
        for D, lab in zip(sp.plate_distances_m, sp.labels):
            order = sorted(sp.directions, key=lambda d_: 0 if (d_ > 0) == (at_end == "S") else 1)
            for d in order:
                direction = "S->E" if d > 0 else "E->S"
                name = f"x3_D{D * 1e3:.0f}_{'SE' if d > 0 else 'ES'}"
                rec = {"name": name, "plate_distance_requested_m": float(D), "label": str(lab), "direction": direction}
                why = plate_window(float(D), depth, pitch, near_m=near, far_m=float(sv.z_far_m), pose_tol_m=float(sv.pose_tol_m))
                if why:
                    rec.update(status="skipped", why=f"outside the sensor window: {why}")
                    self.records["standoff"].append(rec)
                    self._say(f"{name}: skipped - {rec['why']}")
                    continue
                t._check_cancel()
                leg_dir = self._leg_dir(name)
                self._say(f"{name}: full pass {direction} at plate distance {D * 1e3:.1f} mm ({lab}), same pitch / tool orientation / "
                          f"speed {t._scan_speed() * 1e3:.1f} mm/s and planner overscan as the run's passes -> {leg_dir}")
                try:
                    out = t._pass(est, len(self.records["standoff"]) + 1, direction, qd["e_S_m"], qd["e_E_m"], sig,
                                  "estimated ends r(e_S) .. r(e_E) + k sigma_end (experiment 3: the same ends at another plate "
                                  "distance)", plate_m=float(D), name=name, leg_dir=leg_dir)
                except _Abort as e:
                    if e.status == STATUS_REFUSED and not t._cancelled:
                        rec.update(status="skipped", why=f"refused before motion: {e.reason}")
                        self.records["standoff"].append(rec)
                        self._say(f"{name}: skipped - {rec['why']}")
                        continue
                    rec.update(status="stopped", why=f"{e.status}: {e.reason}", dir=str(leg_dir))
                    self.records["standoff"].append(rec)
                    raise
                summ = out.get("summary") or {}
                li = getattr(t, "_last_pass_plan", None) or {}
                rec.update(status="done", dir=str(leg_dir), leg_path=out.get("leg_path"), n_frames=summ.get("n_frames"),
                           n_profiles=summ.get("n_profiles"), groove_profiles=summ.get("groove_profiles"), **li)
                self.records["standoff"].append(rec)
                at_end = "E" if d > 0 else "S"                          # the pass ended there
                self._say(f"{name}: recorded {summ.get('n_frames')} frames -> {summ.get('n_profiles')} profiles "
                          f"({summ.get('groove_profiles')} with groove evidence); plate distance "
                          f"{float(li.get('plate_distance_m', float('nan'))) * 1e3:.1f} mm, standoff "
                          f"{float(li.get('standoff_m', float('nan'))):.4f} m")

    # ---- 실험 4: 정지 회전 끝 뷰 ------------------------------------------------------------------------------------
    def _pass_pose(self) -> tuple:
        """패스 자세 기준: 최종 추정의 패스 작업 (실행의 패스와 같은 틀·피치·진행 부호 — 지금 툴 자세에 가까운 쪽) 의 광학 자세, 그 판 거리,
        그 작업."""
        from scan_core.path_gen import sensor_frames
        from scan_core.seam_estimate import _section_basis
        from .autoteach import PassPlan, SeamLine
        t, est = self.t, self.est
        qd = t._plan_quantities(est)
        D = float(qd["depth_m"])
        P_S, P_E = est.r(qd["e_S_m"]), est.r(qd["e_E_m"])
        u_l, n_l, _ = _section_basis(P_E - P_S, t._n_hat)
        line = SeamLine(c=P_S, u=u_l, n=n_l, s_S=0.0, s_E=float(np.linalg.norm(P_E - P_S)), k=0, direction="S->E")
        plan0 = PassPlan(direction="S->E", s_S=0.0, s_E=line.s_E, ov_S=0.0, ov_E=0.0, rule="g1")
        job = t._pass_job(line, plan0, 0, D, quiet=True)
        return sensor_frames(job, t.calib).T_base_opt(0.0), float(job.standoff_m) - D, job, D

    def _plates(self) -> dict:
        """설계 판 'left' / 'right' → (ŵ 쪽 부호 (w0 기준), 설계 상부 코너 |x| [m], 끝 모형 판 이름 'L'/'R' (외면개선: L = 베벨 판; 그 밖 None))."""
        t, est = self.t, self.est
        sp = t.spec
        bevel = sp.resolved_bevel_side() if str(sp.kind) == "single_bevel" else None
        sL = t._left_plate_side(est)
        if sL is None and bevel is not None:
            # 외면개선: 베벨 판의 쪽은 추정이 정한다 (bevel_sign_w, w0) — 설계의 'left' 는 resolved_bevel_side 가 left 면 베벨 판 (끝 모형 'L')
            sL = int(est.bevel_sign_w) if bevel == "left" else -int(est.bevel_sign_w)
        if sL is None:
            sL = 1
            self._say("plate sides: a symmetric groove (single V / square) - 'left' is taken on +w0 (left looking S->E); the views are "
                      "recorded with their w side")
        cL, cR = (abs(float(c)) * 1e-3 for c in sp.top_corners_mm())
        lab = {"left": ("L" if bevel == "left" else "R") if bevel else None, "right": ("L" if bevel == "right" else "R") if bevel else None}
        return {"left": (int(sL), cL, lab["left"]), "right": (-int(sL), cR, lab["right"])}

    def _wrist2_req(self) -> str:
        """이 실행의 손목 형태 (손목2 부호) — 설정이 'any' 면 지금 팔의 부호 (형태 유지)."""
        t = self.t
        req = t._wrist2_required()
        if req in ("neg", "pos"):
            return req
        js = t._joints_now()
        if js is None or "wrist2" not in js[0]:
            return "any"
        w2 = (math.degrees(float(js[1][js[0].index("wrist2")])) + 180.0) % 360.0 - 180.0
        return "pos" if w2 > 0 else "neg"

    def _choose_sign(self, T_ret_by_sign: dict, T_view_by_sign: dict) -> tuple:
        """회전 부호: 부호마다 물러난 자세의 IK 해 (뷰 자세까지 직선 추종) 중 손목2 부호가 같은 해의 지금 관절에서의 관절 이동 Σ|Δq| 최소 →
        (부호 또는 None, 부호별 기록)."""
        from .autoteach import wrist2_branch_problem
        t = self.t
        req = self._wrist2_req()
        q_now = t._q_now()
        rows, best = {}, None
        for sg, T_ret in T_ret_by_sign.items():
            try:
                cands = t.rs.reach_candidates(T_ret, follow=[T_view_by_sign[sg]], q_now=q_now, sigma_radius_m=t._lever_m())
            except Exception as e:  # noqa: BLE001 - this sign is simply not reachable as far as we know
                rows[sg] = {"ok": 0, "n_ik": 0, "why": f"{type(e).__name__}: {e}"}
                continue
            ok = []
            fails = set()
            for c in cands:
                if not c.get("ok"):
                    fails.add(str(c.get("reason"))[:100])
                    continue
                why = wrist2_branch_problem(c.get("names") or [], np.asarray(c["q"]), req) if req != "any" else None
                if why:
                    fails.add(f"wrist2: {why[:80]}")
                    continue
                ok.append(c)
            dmin = min((float(c["dist_rad"]) for c in ok if c.get("dist_rad") is not None), default=None)
            if ok and dmin is None:
                dmin = 0.0
            rows[sg] = {"ok": len(ok), "n_ik": len(cands), "joint_motion_deg": None if dmin is None else math.degrees(dmin),
                        "why": "; ".join(sorted(fails)[:3])}
            if dmin is not None and (best is None or dmin < best[1]):
                best = (sg, dmin)
        return (None if best is None else best[0]), rows

    def _end_views(self, vp: EndViewPlan) -> None:
        from .autoteach import STATUS_REFUSED, _Abort
        t, est = self.t, self.est
        mode0 = t.confirm_mode
        mode = stricter_confirm(mode0, vp.confirm)
        if mode != mode0:
            self._say(f"end views: confirm mode {mode} (the experiments block; the run used {mode0}) - a new motion is confirmed move by "
                      "move on its first use")
        t.confirm_mode = mode
        try:
            T0, r_p, job_p, depth = self._pass_pose()
            n_hat = unit(np.asarray(t._n_hat, dtype=float))
            u_hat = unit(np.asarray(est.u, dtype=float))
            w0 = np.asarray(t._f0()["w0"], dtype=float)
            w_hat = unit(w0 - float(w0 @ u_hat) * u_hat - float(w0 @ n_hat) * n_hat)
            half_fov = 0.5 * float(t._sv().fov_m(r_p))
            plates = self._plates()
            ef_pl = ((est.plate or {}).get("end_faces") or {}).get("plates") or {}
            Tfo_inv = np.linalg.inv(np.asarray(t.calib.T_flange_opt, dtype=float))
            st = t._settle()
            wait_s = float(st["t_settle_s"]) + float(t.k) * float(st.get("sigma_s") or 0.0)
            self._say(f"end views: pass pose at plate distance {r_p * 1e3:.1f} mm (pitch {float(job_p.pitch_deg):+.1f} deg), sensor "
                      f"FOV {2 * half_fov * 1e3:.0f} mm there; the line centre on each plate's top at its end, "
                      + ", ".join(f"{nm} plate {abs(0.5 * (c + half_fov)) * 1e3:.1f} mm from the root line on the "
                                  f"{'+' if sw > 0 else '-'}w0 side (mid of the top band the passes see: design corner {c * 1e3:.2f} .. "
                                  f"FOV/2)" for nm, (sw, c, _l) in plates.items() if nm in vp.plates)
                      + f"; settle wait {wait_s * 1e3:.0f} ms (servo settling {float(st['t_settle_s']) * 1e3:.0f} ms + k sigma, "
                      f"{st.get('source')}); wrist2 kept {self._wrist2_req()}", settling=st)
            # 끝 순서: 지금 광학 원점에 가까운 끝부터
            o_now = (t._T_now() @ t.calib.T_flange_opt)[:3, 3]
            ends = sorted(vp.ends, key=lambda X: float(np.linalg.norm(o_now - est.r(float(est.ends[X].s_m)))))
            k_view = 0
            for X in ends:
                pl_order = [p for p in ("left", "right") if p in vp.plates]
                for ia, psi_deg in enumerate(vp.angles_deg):
                    for p in (pl_order if ia % 2 == 0 else list(reversed(pl_order))):
                        sw, corner, lab = plates[p]
                        rd = ef_pl.get(f"{X}:{lab}") if lab else None
                        s_end = float(rd["s_m"]) if (rd and rd.get("s_m") is not None and math.isfinite(float(rd["s_m"]))) \
                            else float(est.ends[X].s_m)
                        h_top = float(rd["top_h_m"]) if (rd and rd.get("top_h_m") is not None and math.isfinite(float(rd["top_h_m"]))) \
                            else depth
                        P_c, w_off = view_target(est.r(s_end), u_hat, n_hat, w_hat * sw, corner, half_fov, h_top)
                        T_view, T_ret = {}, {}
                        for sg in (1, -1):
                            To = end_view_pose(T0, n_hat, P_c, sg * math.radians(psi_deg))
                            T_view[sg] = To @ Tfo_inv
                            T_ret[sg] = np.array(T_view[sg], copy=True)
                            T_ret[sg][:3, 3] += n_hat * float(vp.retreat_m)
                        base = f"x4_{X}_{p}_{psi_deg:g}"
                        geo = {"end": X, "plate": p, "end_model_plate": lab, "side_w0": int(sw), "psi_deg": float(psi_deg),
                               "target_m": P_c.tolist(), "s_end_m": s_end, "s_end_source": f"end model {X}:{lab}" if rd else
                               f"estimate end {X}", "lateral_m": float(w_off), "top_h_m": h_top, "plate_distance_m": r_p}
                        t._check_cancel()
                        sg, rows = self._choose_sign(T_ret, T_view)
                        if sg is None:
                            why = "no arm configuration with the run's wrist2 sign reaches it: " + "; ".join(
                                f"{'+' if s_ > 0 else '-'}{psi_deg:g} deg: {r_.get('ok')}/{r_.get('n_ik')} ({r_.get('why')})"
                                for s_, r_ in rows.items())
                            for rep in range(1, vp.repeats + 1):
                                self.records["end_views"].append({**geo, "name": f"{base}_r{rep}", "repeat": rep, "status": "skipped",
                                                                  "why": why, "ik": rows})
                            self._say(f"{base}: skipped - {why}")
                            continue
                        psi_s = sg * psi_deg
                        self._say(f"{base}: end {X}, {p} plate, psi {psi_s:+g} deg about the plate normal (sign by the least joint motion "
                                  f"with wrist2 kept: " + ", ".join(f"{'+' if s_ > 0 else '-'}: "
                                                                    + (f"{r_['joint_motion_deg']:.0f} deg" if r_.get('joint_motion_deg')
                                                                       is not None else f"none ({r_.get('why')})")
                                                                    for s_, r_ in rows.items())
                                  + f"); line centre at s {s_end * 1e3:.2f} mm ({geo['s_end_source']}), {w_off * 1e3:.1f} mm from "
                                  f"the root line, plate top +{h_top * 1e3:.2f} mm", ik=rows, **geo)
                        skip_rest = None
                        for rep in range(1, vp.repeats + 1):
                            k_view += 1
                            nm = f"{base}_r{rep}"
                            rec = {**geo, "name": nm, "repeat": rep, "psi_signed_deg": float(psi_s), "sign": int(sg),
                                   "T_view_flange": T_view[sg].tolist(), "T_retreat_flange": T_ret[sg].tolist()}
                            if skip_rest:
                                rec.update(status="skipped", why=skip_rest)
                                self.records["end_views"].append(rec)
                                continue
                            try:
                                rec.update(self._one_view(nm, T_ret[sg], T_view[sg], vp, job_p, wait_s, rec))
                            except _Abort as e:
                                if e.status == STATUS_REFUSED and not t._cancelled:
                                    skip_rest = f"refused before motion: {e.reason}"
                                    rec.update(status="skipped", why=skip_rest)
                                    self.records["end_views"].append(rec)
                                    self._say(f"{nm}: skipped (and the remaining repeats of this view) - {skip_rest}")
                                    continue
                                rec.update(status="stopped", why=f"{e.status}: {e.reason}")
                                self.records["end_views"].append(rec)
                                raise
                            self.records["end_views"].append(rec)
        finally:
            t.confirm_mode = mode0

    def _one_view(self, nm: str, T_ret: np.ndarray, T_view: np.ndarray, vp: EndViewPlan, job_p, wait_s: float, geo: dict) -> dict:
        """반복 하나: 물러난 자세로 (지금 자세에서 직선; 구간 태그는 물려받은 규칙) → 판 법선을 따라 곧게 내려가기 → 안정 대기 → grab → 원자료."""
        t = self.t
        Tfo = t.calib.T_flange_opt
        T_now = t._T_now()
        dvec = (T_ret @ Tfo)[:3, 3] - (T_now @ Tfo)[:3, 3]
        n = unit(np.asarray(t._n_hat, dtype=float))
        dn = float(dvec @ n)
        from .autoteach import _rot_angle_deg
        rot = _rot_angle_deg(T_now[:3, :3], np.asarray(T_ret)[:3, :3])
        ph = t._leg_phase(dn, float(np.linalg.norm(dvec - n * dn)), roll_deg=rot, orientation=t.orientation, tol_m=t._sv().pose_tol_m)
        from .executor import start_pose_mismatch
        if start_pose_mismatch(T_ret, T_now, t.policy.goal_tol_m, t.policy.goal_tol_deg) is not None:
            t._leg(T_ret, ph, f"{nm}_retreat", extra=f"(end view {nm}: above the view, {vp.retreat_m * 1e3:.0f} mm along the plate normal)")
        t._leg(T_view, t._leg_phase(-float(vp.retreat_m), 0.0, orientation=t.orientation, tol_m=t._sv().pose_tol_m), f"{nm}_approach",
               extra=f"(end view {nm}: straight down onto the view along the plate normal)")
        t._clock.sleep(max(0.0, float(wait_s)))                       # 서보 안정 (모형) — 그 뒤의 정지 프레임
        leg_dir = self._leg_dir(nm)
        out = self._grab(nm, int(vp.frames), leg_dir, job_p, {**geo, "settle_wait_s": float(wait_s)})
        w2 = out.get("wrist2_deg")
        self._say(f"{nm}: {out['n_frames']} static frames (flange moved {out['flange_motion_mm']:.3f} mm during the grab, wrist2 "
                  + ("n/a" if w2 is None else f"{float(w2):+.1f} deg") + f") -> {leg_dir}")
        return {"status": "done", "dir": str(leg_dir), "settle_wait_s": float(wait_s), **out}

    def _grab(self, nm: str, n: int, leg_dir: Path, job_p, meta: dict) -> dict:
        """정지 grab → raw.npz (verification/legs 와 같은 규약 + still_T_base_flange + view_json)."""
        from . import g1_worker as _gw
        from .executor import STATUS_OK, ScanRunResult, raw_payload
        from scan_core.types import ScanJob
        from .config import job_to_dict
        t = self.t
        t._check_cancel()
        t._check_estop(f"before end view '{nm}'")
        fps = float(t.job0.profile_fps) if float(t.job0.profile_fps) > 0 else 1.0
        timeout = float(t.at.first_frame_timeout_s) + n / fps + float(t.settings.sensor.get("data_wait_ms", 1000) or 0) * 1e-3
        t0_ns = t._clock.monotonic_ns()
        recs = t.ss.grab(n, timeout_s=timeout)
        t._check_estop(f"after end view '{nm}'")
        snap = t.rs.pose.snapshot()
        T_bf = np.asarray(snap.T, dtype=float)
        samples = t.rs.pose.history(t0_ns)
        P = np.array([np.asarray(s_.T_base_flange, dtype=float)[:3, 3] for s_ in samples]) if samples else np.zeros((0, 3))
        motion = float(np.max(np.linalg.norm(P - P.mean(axis=0), axis=1))) * 1e3 if len(P) else float("nan")
        js = t._joints_now()
        w2 = None
        if js is not None and "wrist2" in js[0]:
            w2 = (math.degrees(float(js[1][js[0].index("wrist2")])) + 180.0) % 360.0 - 180.0
        job_rec = ScanJob.from_dict({**job_to_dict(job_p), "torch_lateral_m": tuple(job_p.torch_lateral_m),
                                     "job_id": f"{t.job0.job_id or 'job'}_{nm}"})
        run = ScanRunResult(status=STATUS_OK, reason=f"end view {nm}", out_dir=str(leg_dir), records=recs, robot_samples=samples,
                            sensor_info=t.ss.info.as_dict(), safety_init=t.rs.backend.safety_init_result)
        run.t_sensor_start_ns = t.ss.t_start_ns
        run.t_sensor_stop_ns = t.ss.t_stop_ns
        arrays, jsd = raw_payload(run, job_rec, t.calib, t.settings, t.cfg, t.rs.pose)
        arrays["still_T_base_flange"] = T_bf
        tau = None
        fn = getattr(t.rs, "tracking_residual", None)
        if callable(fn):
            try:
                snap_ = (fn() or (None,))[0]
                tau = None if snap_ is None else float(snap_.tau_s)
            except Exception:  # noqa: BLE001 - a record only
                tau = None
        view = {**meta, "name": nm, "n_frames": len(recs), "T_base_flange": T_bf.tolist(),
                "T_base_opt": (T_bf @ t.calib.T_flange_opt).tolist(), "flange_motion_mm": motion, "wrist2_deg": w2,
                "controller_lag_s": tau, "t_grab_ns": int(t0_ns)}
        jsd["view_json"] = json.dumps(_plain(view), default=str)
        leg_dir.mkdir(parents=True, exist_ok=True)
        _gw.atomic_savez(leg_dir / "raw.npz", **arrays, **{k: np.array(v) for k, v in jsd.items()})
        return {"n_frames": len(recs), "T_base_flange": T_bf.tolist(), "flange_motion_mm": motion, "wrist2_deg": w2,
                "controller_lag_s": tau}


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o
