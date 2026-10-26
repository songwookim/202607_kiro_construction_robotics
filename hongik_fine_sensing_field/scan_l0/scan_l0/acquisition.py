"""범용 시작 (2026-10-07, 사용자): 입력은 착점 두 개의 좌표·자세 (1G/3G)·그루브 규격뿐 — 교시 자세·템플릿 법선·테이블 높이는 쓰지 않는다.
로봇은 아무 자세 (운송용 접힌 자세) 에서 시작하고 그때 센서는 시편을 못 볼 수 있다. 여기서는 순수 기하만 (로봇·센서 상수 없이 — 센서는
거리 창·열 간격 모형, 툴은 캘리브의 키포인트·몸체, 시편은 규격에서):

* initial_frame: 중력 (up) · 착점 → 심 방향 u0 와 판 법선 추정 n0. 1G: n0 = up (판은 수평으로 가정, 드웰이 잰다). 3G: n0 = 착점에서
  로봇 (base 원점) 을 향한 수평 방향을 u0 에 직교화 — 로봇과 시편 사이에는 장애물이 없고 판 면은 로봇이 볼 수 있는 쪽 (첫 드웰이 잰다).
  base 축 (x/y/z) 은 쓰지 않는다: 시편이 로봇의 어느 쪽에 있든, 로봇이 옆으로 달려 중력이 base y 여도 같은 규칙.
* conservative_floor_height: 측정 전 툴이 내려가면 안 되는 높이 (up 방향) — 3G: 아래 착점 높이 (착점은 시편 위의 점이고 시편은 테이블
  위에 서 있으니 테이블 윗면은 그보다 낮다), 1G: 착점 − 판 두께 − 뒷댐재 (착점이 판 상면이든 그루브 안이든 판 밑면은 그보다 높다).
* cell_boxes: MoveIt 충돌 장면용 상자 (시편 부피, 바닥 판 — 크기는 규격·센서 standoff 에서; 로봇 쪽은 호출자가 지금 자세와 겹치지 않게 깎는다).
* plan_support_probe / measure_support (3G): 아래 착점 옆 판 면을 비스듬히 내려다보며 레이저 면을 세로 (판 법선 추정·up 평면) 로 두면 한
  프로파일에 판 (세로 선) 과 그 앞 지지물 (테이블·지그) 이 담긴다 — 판 법선을 몰라도 (세운 판 ∩ 세로 면 = 세로 선). 자세는 고정 각도가
  아니라, 툴 몸체 전체가 측정 전 바닥 위에 있게 둔 채 센서 거리 창 안에서 착점 아래 가장 깊은 곳까지 보이는 것 (계산). 지지면 = 판 앞
  표면의 윗면 (테이블이면 그 면, 지그면 그 꼭대기) — 시야보다 낮으면 보인 판의 가장 낮은 점이 보수적 상한.
* rank_reach_candidates: the supplied order of the acquisition IK solutions that follow the coarse line (least joint motion from the
  current joints, the singularity distance only breaks ties) - the move itself is chosen by planned duration (scan_l0.joint_selection).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from scan_core.measure import HALF_TURN_DEG
from scan_core.sensor_model import VC_NANO_3DZ_LARGE
from scan_core.frames import make_transform, unit

__all__ = ["InitialFrame", "initial_frame", "rough_normal", "conservative_floor_height", "cell_boxes", "plan_support_probe", "measure_support",
           "SupportReading", "rank_reach_candidates", "visit_order"]

SEAM_POSITION_TOL_DEG = HALF_TURN_DEG  # 자세의 정의 (원칙 2 중간점): 1G 심은 수직보다 수평에, 3G 심은 수평보다 수직에 가깝다 (넘으면 모순 — 거부)


@dataclass
class InitialFrame:
    position: str                     # '1G' | '3G'
    up: np.ndarray                    # −중력 (base)
    u0: np.ndarray                    # S0 → E0 (1G: 수평 투영; 3G: 판 면 안)
    n0: np.ndarray                    # 판 법선 추정 (센서 쪽)
    S0: np.ndarray
    E0: np.ndarray
    length_m: float
    notes: list = field(default_factory=list)

    def height(self, p) -> float:
        return float(np.asarray(p, dtype=float) @ self.up)

    @property
    def lower(self) -> np.ndarray:
        return self.S0 if self.height(self.S0) <= self.height(self.E0) else self.E0


def initial_frame(S0, E0, position: str, gravity_axis_base, *, base_origin=(0.0, 0.0, 0.0)) -> InitialFrame:
    """착점 두 개 + 자세 + 중력 → (u0, n0). 자세와 맞지 않는 착점 (1G 인데 심이 수직에 더 가깝거나 3G 인데 수평에 더 가까움) 은 ValueError."""
    S0 = np.asarray(S0, dtype=float).reshape(3)
    E0 = np.asarray(E0, dtype=float).reshape(3)
    up = -unit(np.asarray(gravity_axis_base, dtype=float))
    d = E0 - S0
    L = float(np.linalg.norm(d))
    if L < 1e-3:
        raise ValueError(f"the two hit points are only {L * 1e3:.1f} mm apart")
    u = d / L
    vert = abs(float(u @ up))
    pos = str(position).upper()
    notes: list = []
    drop = abs(float(d @ up))
    apart = f"the two points are {math.sqrt(max(0.0, L * L - drop * drop)) * 1e3:.1f} mm apart horizontally, {drop * 1e3:.1f} mm vertically"
    if pos == "1G":
        if vert > math.sin(math.radians(SEAM_POSITION_TOL_DEG)):
            raise ValueError(f"1G: the seam S0->E0 is {math.degrees(math.asin(min(1.0, vert))):.1f} deg from horizontal "
                             f"(> {SEAM_POSITION_TOL_DEG:g}) - position and hit points disagree ({apart}; if pose_a and pose_b have "
                             "the same XY, one point was taught twice at two heights: check the taught end point)")
        n0 = up.copy()
        u0 = unit(u - n0 * float(u @ n0))
    elif pos == "3G":
        if vert < math.cos(math.radians(SEAM_POSITION_TOL_DEG)):
            raise ValueError(f"3G: the seam S0->E0 is {math.degrees(math.acos(min(1.0, vert))):.1f} deg from vertical "
                             f"(> {SEAM_POSITION_TOL_DEG:g}) - position and hit points disagree ({apart})")
        M = 0.5 * (S0 + E0)
        h = np.asarray(base_origin, dtype=float).reshape(3) - M
        h = h - up * float(h @ up)                          # 수평
        h = h - u * float(h @ u)                            # 심에 직교 (3G 심은 거의 수직이라 거의 그대로)
        if float(np.linalg.norm(h)) < 1e-6:
            raise ValueError("3G: the seam is on the robot's vertical axis - no plate side can be guessed")
        n0 = unit(h)
        u0 = unit(u - n0 * float(u @ n0))
        notes.append("3G plate normal guessed as the horizontal direction toward the robot (measured at the first dwell)")
    else:
        raise ValueError(f"position must be 1G or 3G, got {position!r}")
    return InitialFrame(position=pos, up=up, u0=u0, n0=n0, S0=S0, E0=E0, length_m=L, notes=notes)


def conservative_floor_height(fr: InitialFrame, *, thickness_m: float, backing_m: float = 0.0, face_offset_m: float = 0.0) -> float:
    """측정 전 바닥 (up 높이): 3G = 아래 착점 높이, 1G = 낮은 착점 + 판 면까지 (face_offset_m: 착점을 루트 깊이에 교시했을 때의 판 면,
    4d) − 판 두께 − 뒷댐재. 지지면 (테이블) 은 이보다 낮다."""
    lo = fr.height(fr.lower)
    if fr.position == "3G":
        return lo
    return lo + float(face_offset_m) - float(thickness_m) - float(backing_m)


def _box(center, axes, size, bid: str) -> dict:
    R = np.column_stack([unit(a) for a in axes])
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return {"id": bid, "size": [float(v) for v in size], "T": make_transform(R, np.asarray(center, dtype=float))}


def cell_boxes(fr: InitialFrame, floor_h: float, *, length_m: float, plate_width_m: float, thickness_m: float, reach_m: float,
               slab_m: float, near_clip_m: Optional[float] = None, base_origin=(0.0, 0.0, 0.0), pad_m: float = 0.0,
               face_offset_m: float = 0.0) -> list:
    """MoveIt 충돌 장면 상자 (관절 공간 이동이 피한다):
      * 'specimen': 착점 중점에 심 방향 L_hits + 2·max(0, length_m − L_hits) (착점이 한쪽 끝 바로 안쪽이어도 반대 끝을 덮는다 — 4d P10)
        × 판 폭, 판 면 = 착점 + n·face_offset_m (착점을 루트 깊이에 교시했어도 판 상면을 덮는다) 에서 뒤로 face_offset_m + thickness_m
      * 'floor_estimate': floor_h 아래 두께 slab_m (호출자: 안전 정책 여유 clearance_m) 판 — 시편 둘레 (판 폭/2 + reach_m; 1G 는 + 길이/2) 안 (reach_m = 툴이 시편 앞에서
        쓰는 거리: standoff + 접근 여유), 로봇 쪽은 시편에서 near_clip_m 까지만 (None = 대칭; 호출자가 지금 자세와 겹치면 줄인다).
    pad_m: 모든 상자를 사방으로 이만큼 키운다 (안전 한계 L — 관절 이동의 팔·툴도 모든 면에서 L 을 지킨다)."""
    up, u, n = fr.up, fr.u0, fr.n0
    w = unit(np.cross(n, u))
    M = 0.5 * (fr.S0 + fr.E0)
    L = float(fr.length_m) + 2.0 * max(0.0, float(length_m) - float(fr.length_m))
    T_box = float(face_offset_m) + float(thickness_m)
    out = [_box(M + n * (float(face_offset_m) - T_box / 2.0), (u, w, n), (L, plate_width_m, T_box), "specimen")]
    base = np.asarray(base_origin, dtype=float).reshape(3)
    a = M - base
    a = a - up * float(a @ up)
    if float(np.linalg.norm(a)) < 1e-6:
        a = w - up * float(w @ up)
    a = unit(a)                                             # 수평: 로봇 → 시편
    b = unit(np.cross(up, a))
    R = 0.5 * float(plate_width_m) + float(reach_m) + (0.5 * L if fr.position == "1G" else 0.0)
    near = R if near_clip_m is None else max(0.0, min(R, float(near_clip_m)))
    if near + R > 1e-3:
        c = M + a * (0.5 * (R - near))
        c = c + up * (float(floor_h) - slab_m / 2.0 - float(c @ up))
        out.append(_box(c, (a, b, up), (near + R, 2.0 * R, slab_m), "floor_estimate"))
    pad = max(0.0, float(pad_m))
    for bx in out:
        bx["size"] = [float(v) + 2.0 * pad for v in bx["size"]]
    return out


# ---- 3G 지지면 probe -------------------------------------------------------------------------------------------
def _probe_rotation(fr: InitialFrame, look_down_deg: float, flip: bool) -> np.ndarray:
    """광학 프레임 (base): Z = 판 쪽으로 look_down 만큼 내려다봄, Y = 판 면 안 수평 (레이저 면 법선), X = 레이저 선 (n·up 평면)."""
    up, n = fr.up, fr.n0
    b = math.radians(float(look_down_deg))
    Z = unit(-n * math.cos(b) - up * math.sin(b))
    Y = unit(np.cross(up, n))
    X = unit(np.cross(Y, Z))
    if flip:
        X, Y = -X, -Y
    return np.column_stack([X, Y, Z])


def rough_normal(S, E, position: str, gravity_axis_base) -> np.ndarray:
    """착점 두 개 + 자세 + 중력 → 대략 판 법선 (scan_l0.acquisition.initial_frame 와 같은 추정: 1G up, 3G 로봇 쪽 수평), 착점 방향에
    직교. 자세와 착점이 맞지 않으면 착점 방향에 직교하는 아무 방향 — auto-teach 의 범용 시작이 그 이유로 거부한다."""
    S = np.asarray(S, dtype=float)
    E = np.asarray(E, dtype=float)
    u = (E - S) / max(float(np.linalg.norm(E - S)), 1e-12)
    try:
        n = np.asarray(initial_frame(S, E, position, gravity_axis_base).n0, dtype=float)
    except ValueError:
        g = np.asarray(gravity_axis_base, dtype=float)
        n = -g / float(np.linalg.norm(g))
        if abs(float(n @ u)) > 0.99:
            n = np.cross(u, [1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.cross(u, [0.0, 1.0, 0.0])
    n = n - u * float(n @ u)
    return n / float(np.linalg.norm(n))


def plan_support_probe(fr: InitialFrame, lowest_below_origin, *, floor_h: float, margin_m: float, z_valid_mm: tuple, dx_coeffs: tuple,
                       lateral_m: float = 0.0, n_columns: int = VC_NANO_3DZ_LARGE.n_columns, look_down_deg=None, front_m=None,
                       depth_step_m: float = 0.005,
                       max_depth_m: Optional[float] = None, min_fraction: float = 0.03, reach_toward_plate=None,
                       face_offset_m: float = 0.0) -> list:
    """3G 지지면 probe 자세 후보 (좋은 순서): 고정 각도 없이 내려다보는 각 β (5–85°) × 판 앞 거리 d × 광축 둘레 뒤집기마다
    광학 원점 높이를 '툴 몸체 전체가 floor_h + margin 위' 의 가장 낮은 값으로 두고 (lowest_below_origin(T_opt) = 원점 아래로 가장 낮은 툴
    점까지 [m]), 센서 모형 (열 k 의 광선 x = k·(a + b·z), 거리 창 z_valid) 으로 착점 아래 깊이 D 의 수평면이 판에 가리지 않고 열의
    min_fraction 넘게 보이는 D 를 센다. 점수 = 아래 착점부터 이어져 보이는 깊이 (클수록 낮은 테이블까지), 같으면 판이 보이는 열 수.
    레이저 면은 세로 (판 법선 추정·up) 로 아래 착점에서 lateral_m 옆 (그루브 개구 밖 판 면).
    reach_toward_plate(T_opt) = 광학 원점에서 판 쪽 (−n0) 으로 가장 멀리 나온 툴 점까지 [m]: 주면 판 면과도 같은 여유 (판 앞 거리 −
    그 값 ≥ margin_m) 를 지키는 후보만 (모든 툴 점 ↔ 모든 면, 안전 정책).
    반환 [{look_down_deg, front_m, flip, T_opt, depth_m, plate_cols, lowest_below_m}]."""
    up, n = fr.up, fr.n0
    w = unit(np.cross(up, n))
    P = fr.lower + w * float(lateral_m) + n * float(face_offset_m)     # 판 면 (착점을 루트 깊이에 교시했어도; 4d P10)
    hP = fr.height(fr.lower + w * float(lateral_m))
    a_, b_ = (float(dx_coeffs[0]), float(dx_coeffs[1]))
    zlo, zhi = float(z_valid_mm[0]), float(z_valid_mm[1])
    k = (np.arange(n_columns, dtype=float) - (n_columns - 1) / 2.0)[:: max(1, n_columns // 176)]
    need = max(3, int(math.ceil(min_fraction * k.size)))
    betas = np.arange(5.0, 90.0, 5.0) if look_down_deg is None else np.asarray(look_down_deg, dtype=float)
    fronts = np.arange(0.025, zhi * 1e-3, 0.025) if front_m is None else np.asarray(front_m, dtype=float)
    D = np.arange(0.0, (zhi * 1e-3 if max_depth_m is None else float(max_depth_m)) + 1e-9, float(depth_step_m))   # 기본: 센서 먼 거리
    out = []
    for beta in betas:
        for flip in (False, True):
            R = _probe_rotation(fr, beta, flip)
            X, Z = R[:, 0], R[:, 2]
            Xu, Zu, Xn, Zn = float(X @ up), float(Z @ up), float(X @ n), float(Z @ n)
            du = (Xu * k * b_ + Zu) * 1e-3                    # 열 광선의 높이 변화 / mm
            dn = (Xn * k * b_ + Zn) * 1e-3                    # 판 앞 거리 변화 / mm
            for d in fronts:
                T0 = make_transform(R, P + n * float(d))
                if reach_toward_plate is not None and float(d) - float(reach_toward_plate(T0)) < float(margin_m):
                    continue                                                    # 툴이 판 면에 여유보다 가깝다
                lb = float(lowest_below_origin(T0))
                hO = float(floor_h) + float(margin_m) + lb
                with np.errstate(divide="ignore", invalid="ignore"):
                    zp = (-float(d) - Xn * k * a_ * 1e-3) / dn                                   # 판 면 (앞 거리 0) 을 지나는 z
                    zt = ((hP - D[:, None]) - hO - Xu * k[None, :] * a_ * 1e-3) / du[None, :]     # (깊이, 열)
                front_t = float(d) + Xn * k[None, :] * a_ * 1e-3 + dn[None, :] * zt
                ok = np.isfinite(zt) & (zt > zlo) & (zt < zhi) & (front_t > 0.0)
                blocked = np.isfinite(zp)[None, :] & (zp[None, :] > 0.0) & (zp[None, :] < zt)    # 판에 먼저 닿는다
                vis = (ok & ~blocked).sum(axis=1) >= need
                cov = 0.0
                if vis[0]:
                    j = int(np.argmin(vis)) if not vis.all() else vis.size
                    cov = float(D[j - 1])
                hp = hO + Xu * k * a_ * 1e-3 + du * zp
                plate_cols = int(np.sum(np.isfinite(zp) & (zp > zlo) & (zp < zhi) & (hp >= hP - cov - 1e-9)))
                T = np.array(T0, copy=True)
                T[:3, 3] += up * (hO - hP)
                out.append({"look_down_deg": float(beta), "front_m": float(d), "flip": bool(flip), "T_opt": T, "depth_m": cov,
                            "plate_cols": plate_cols, "lowest_below_m": lb, "need_cols": need})
    out.sort(key=lambda r: (-r["depth_m"], -min(r["plate_cols"], r["need_cols"]), r["look_down_deg"]))
    return out


@dataclass
class SupportReading:
    height: Optional[float]           # 지지면 윗면 up 높이 (테이블 면 / 지그 꼭대기; 못 찾으면 None)
    n_support: int                    # 지지물로 분류된 점 수 (판 앞)
    n_plate: int                      # 판 (세로 선) 점 수
    flat: Optional[bool]              # 지지물이 수평면 하나인가 (테이블) — 아니면 지그·물체 (높이는 그 윗면)
    foot_height: Optional[float]      # 판 점 중 가장 낮은 높이 (지지물을 못 보면 지지면은 이보다 낮다: 보수적 상한)
    note: str = ""
    tol_m: float = 0.0                # 높이 판정 허용 (3 × resolution) — 이 안의 점은 같은 지지면


def measure_support(points_base: np.ndarray, up, toward_plate, *, resolution_m: float = 0.001, min_fraction: float = 0.03) -> SupportReading:
    """지지면 probe 프로파일 (base 점 Nx3; 레이저 면 = toward_plate·up 을 담는 세로 면) → 지지면 높이. 판 법선을 몰라도 된다: 세운 판과
    세로 레이저 면의 교선은 늘 세로 선 (수평 거리 r = P·toward 일정).
      * 판 = r 이 몰린 곳 (폭 3 × resolution 칸의 최빈) 에서 그 폭 안, 높이로 10 × resolution 넘게 펼쳐진 점들
      * 지지물 = 판 앞 (센서 쪽) 점 — 지지면 높이 = 그 높이의 상위 2 % 분위수 (테이블이면 그 면, 지그면 그 윗면)
      * 판이 안 보이면 프로파일 전체가 지지물
    점 수 기준 = 유효점의 min_fraction. 못 찾으면 height None + foot_height (보인 판의 가장 낮은 점) + 이유."""
    P = np.asarray(points_base, dtype=float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    upv = unit(np.asarray(up, dtype=float))
    a = np.asarray(toward_plate, dtype=float)
    a = unit(a - upv * float(a @ upv))
    need = max(3, int(math.ceil(min_fraction * max(1, P.shape[0]))))
    if P.shape[0] < 3:
        return SupportReading(None, 0, 0, None, None, f"only {P.shape[0]} valid points in the probe profile", 3.0 * resolution_m)
    h = P @ upv
    r = P @ a
    tol = 1.5 * float(resolution_m)
    edges = np.arange(float(r.min()), float(r.max()) + 2.0 * tol + 1e-12, 2.0 * tol)
    if edges.size >= 2:
        cnt, edges = np.histogram(r, bins=edges)
        kk = int(np.argmax(cnt))
        sel = (r >= edges[kk] - tol) & (r <= edges[kk + 1] + tol)
    else:
        sel = np.ones(r.size, dtype=bool)
    r_p = float(np.median(r[sel]))
    on_plate = np.abs(r - r_p) <= tol
    plate_seen = bool(on_plate.sum() >= need and float(np.ptp(h[on_plate])) > 10.0 * resolution_m)
    foot = float(h[on_plate].min()) if plate_seen else None
    sup = (r < r_p - tol) if plate_seen else np.ones(r.size, dtype=bool)
    if int(sup.sum()) < need:
        return SupportReading(None, int(sup.sum()), int(on_plate.sum()), None, foot,
                              "no surface in front of the plate in view (the support is lower than the view, or out of range)",
                              3.0 * resolution_m)
    hs = h[sup]
    top = float(np.percentile(hs, 98.0))
    flat = bool(float(np.percentile(hs, 98.0) - np.percentile(hs, 2.0)) <= 3.0 * resolution_m)
    note = "" if plate_seen else "only the support in view (the plate is outside the laser line)"
    if not flat:
        note = (note + "; " if note else "") + "the support is not one horizontal surface (jig / fixture) - its top is used"
    return SupportReading(top, int(sup.sum()), int(on_plate.sum()), flat, foot, note, 3.0 * resolution_m)


def visit_order(S0, E0, position: str, gravity_axis_base, *, base_origin=(0.0, 0.0, 0.0)) -> dict:
    """거친 단계에서 먼저 찾아갈 착점 (내부 S) — 교시 순서 (시·종점) 와 무관 (사용자 2026-10-09). → {swap, rule, d_start_m, d_end_m,
    h_start_m, h_end_m, why}. swap = True 면 교시한 종점 (end) 을 먼저.

    1G: 로봇 (베이스 원점) 에 가까운 착점 — 실기 2026-10-08: 교시한 S 가 늘 먼 끝이었고 시작 자세에서 그 끝까지 관절이 크게 돌았다 (센서
        뒤집기 163–180° 26/31 실행, 손목1 127.5° ~30 s). 같으면 교시 순서.
    3G: 위 점 → 아래 점 (사용자 2026-10-05 바닥 규칙: 패스 1 = 보정된 E → S 가 아래 끝에서 시작해 시편 아래 끝을 재고 그 끝이 바닥이
        된다) — 가까운 점 규칙과 다르면 (세운 판에서는 보통 아래 점이 베이스에 가깝다) 바닥 규칙을 지킨다 (제약). 높이가 같으면 교시 순서.
    결과물 (job_refined, weld_plan, specimen_report, 액션 결과) 은 교시 S→E 그대로 (호출자가 되돌린다)."""
    S0 = np.asarray(S0, dtype=float).reshape(3)
    E0 = np.asarray(E0, dtype=float).reshape(3)
    o = np.asarray(base_origin, dtype=float).reshape(3)
    up = -unit(np.asarray(gravity_axis_base, dtype=float))
    dS, dE = float(np.linalg.norm(S0 - o)), float(np.linalg.norm(E0 - o))
    hS, hE = float(up @ S0), float(up @ E0)
    out = {"d_start_m": dS, "d_end_m": dE, "h_start_m": hS, "h_end_m": hE}
    if str(position).upper() == "3G":
        swap = hS < hE
        out.update(swap=swap, rule="3G floor rule: upper point first",
                   why=(f"3G: the coarse stage goes upper -> lower (first pass from the lower end); taught "
                        f"{'start' if not swap else 'end'} is the upper point ({max(hS, hE):.3f} m vs {min(hS, hE):.3f} m)"
                        + (f"; the nearer-to-the-robot rule would pick the {'start' if dS <= dE else 'end'} - the floor rule is a constraint"
                           if (dS <= dE) == swap and abs(dS - dE) > 0.0 else "")))
        return out
    swap = dE < dS
    out.update(swap=swap, rule="1G: the hit point nearer to the robot base first",
               why=(f"1G: taught {'end' if swap else 'start'} is nearer to the robot base ({min(dS, dE):.3f} m vs {max(dS, dE):.3f} m) - "
                    "visited first regardless of the taught order (user 2026-10-09)"))
    return out


def rank_reach_candidates(cands: list) -> list:
    """The reach candidates that follow (platform reach_candidates: {q, ok, min_sigma, dist_rad, ...}) in the supplied order of the
    joint-space search: least joint motion from the current joints (dist_rad = sum |dq|), the singularity distance (larger min_sigma)
    only breaks ties. No relative-condition groups (KIRO field 2026-10-14, run 134838: the old 'condition >= half the best' group of
    singularity distance and joint-limit margin held only far solutions, 910-1489 deg, ahead of 404 / 580 deg ones and a 352 deg
    flange turn was executed): joint limits along the coarse line are already enforced by the follow check, and the move itself is
    chosen by planned duration (scan_l0.joint_selection). The wrist configuration is chosen once from this pool (user 2026-10-07 Q3:
    no wrist2 sign preference) and the straight moves continue from it."""
    ok = [c for c in cands if c.get("ok")]

    def dist(c):
        return float(c["dist_rad"]) if c.get("dist_rad") is not None else 0.0
    return sorted(ok, key=lambda c: (dist(c), -float(c.get("min_sigma", 0.0))))
