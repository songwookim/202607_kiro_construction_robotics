"""심 추적 (P2, 2026-10-09): 이동 중 스트림 프레임의 그루브 특징 → 심 직선 (방향 u, 판 법선 n, 일관 단면 범위). ROS 없음 (순수 numpy).

용도 (scan_l0.autoteach): S 첫 드웰 뒤 S→E 방향 안쪽으로 짧게 스트리밍하며 이동 (S_track, 기선 b) 해서 심 방향을 3D 로 재고 (기울기 포함),
툴을 판 법선 둘레로 한 번 돌려 레이저 선을 심에 수직으로 맞춘다 (S_align). 그 뒤 S 사이트·트랜싯·E 사이트는 같은 툴 자세 — 트랜싯 스트림이
E 드웰 위치와 E 쪽 심 방향을 준다. 레이저 선 방향 (툴 yaw) 을 바로잡지 않으면 선 끝이 심 방향으로 ±FOV/2·sin ψ 퍼져 시편 끝을 넘고, 중심
맞춤이 기운 선을 따라 끝 쪽으로 미끄러진다 (2026-10-08 1G 배치: 중단 10 건 중 9 건, 보고서 §2).

규칙 (설정 상수 없음 — scan_core.measure 원칙):
  * 프레임마다 (패스와 같은 특징 추출) 후보 궤적 점 세 개: 왼쪽·오른쪽 상부 코너, 상부 개구 중심. 설계 일치 (matches_design) 프레임의 점만.
  * 궤적마다 3D 직선 (ransac_line_3d: LMedS, 인라이어 k·σ, 하한 센서 양자화) → 잔차로 두 번 다시 (|r| ≤ k·σ_LMedS). 덮임 = 직선 위 이동 중
    프레임 / 그 직선 위 첫 이동 중 프레임부터의 이동 중 프레임 (그루브가 이동 중에 나타났으면 그 앞 프레임은 직선의 반증이 아니다).
    덮임 ≥ ½ (원칙 2 증거 중간점) 인 궤적 중 방향 불확도 (잔차 중앙값 / 직선 위 단면 범위) 가 가장 작은 것 (원칙 6 상대 비교; 같은
    프레임을 덮으면 잔차 최소와 같다) — 홑베벨이면 사각 쪽 코너, 대칭 V 면 중심이 자연히 골라진다 (실측: 사각 코너 b 50 mm σ 0.11°,
    중심 0.25°).
  * 판 법선 n = 직선 위 프레임들의 판 점 평면 (프레임마다 두 판의 중앙 점, k·σ 반복 절사), 센서 쪽. u 는 그 평면에 다시 투영 (심 기울기
    보존: 판 기울기는 n 에).
  * 결과 없음: 'none' (설계 일치 프레임 없음 / 흩어짐), 'late' (그루브가 이동 중에 나타나 그 뒤로 이어짐 — 덮임 부족), 'groove_mismatch' (그루브로
    분류되고 직선 위에 있는데 설계와 맞지 않는 프레임이 이동의 ½ 이상 — 시편 위의 증거).
  * 정밀도는 자기 보고 σ (√n 평균) 를 쓰지 않는다 (실기 스트림 σ 는 보정되지 않았다: |z| > 3 이 26 % — 상관 잡음) — 여유 전파는 두 끝
    단면 모형 σψ_model = √2·σ_단면/기선. σ_단면 = 설계 상한 (4a 잡음 규칙상 통과 가능한 단면 잡음 tol_root/k; 작업에 root_position_mm
    허용이 있을 때) 과 이 이동에서 잰 궤적 잔차 σ (LMedS, 하한 센서 양자화) 중 큰 값 — 설계 허용이 없으면 잰 값. 자기 보고 σ 는 기록만.
  * S 추적 기선 (track_length_m): 필요한 기선 b_need(σ) = √2·k·σ/(ψ_tol − θ_goal) — k·σψ ≤ ψ_tol − θ_goal. 기하 상한 cap = L_r − L_min − m_E.
    측정 바닥 (센서 양자화) 의 기선조차 cap 을 넘으면 그 대략점으로는 심 방향을 잴 수 없다 (호출자가 이동 전에 거부).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .frames import unit
from .localize import ransac_line_3d
from .measure import DESIGN_HALF, K_DEFAULT, MAD_TO_SIGMA, lmeds_sigma
from .path_gen import laser_plane_line_param

__all__ = ["TRACKS", "TrackRows", "SeamEstimate", "rows_from_features", "concat_rows", "moving_mask", "fit_track", "predict_x_at",
           "consistent_upto", "yaw_tolerance_rad", "baseline_m", "track_length_m", "footprint_m", "yaw_to_seam_rad", "seam_frame",
           "laser_footprint_s"]

TRACKS = ("left", "right", "centre")


# --------------------------------------------------------------------------------------
# 유도량 (1 절)
# --------------------------------------------------------------------------------------
def yaw_tolerance_rad(fov_m: float, tol_lat_m: float) -> float:
    """ψ_tol = asin(tol_lat / (FOV/2)): 레이저 선 끝의 심 방향 발자국 (FOV/2·sin ψ) 이 도착 허용 (드웰 수렴의 측면 허용과 같은 값) 안인 yaw —
    한 번 더 돌려도 착점 도착 오차보다 좋아지지 않는 각 (드웰 수렴 _converged 와 같은 원리)."""
    h = 0.5 * float(fov_m)
    if not (h > 0.0 and math.isfinite(h)):
        return float("nan")
    return math.asin(min(1.0, max(0.0, float(tol_lat_m)) / h))


def baseline_m(sigma_sec_m: float, k: float, psi_tol_rad: float, theta_goal_rad: float) -> float:
    """필요한 S 추적 기선 b_need = √2·k·σ / (ψ_tol − θ_goal): 단면 잡음 σ 인 두 끝 단면 모형의 방향 σψ = √2·σ/b 가 k·σψ ≤ ψ_tol − θ_goal
    (회전 뒤 남는 yaw 가 도착 허용 + 플랫폼 자세 허용 안) 이 되는 가장 짧은 기선. 여유가 없으면 inf."""
    room = float(psi_tol_rad) - float(theta_goal_rad)
    return math.sqrt(2.0) * float(k) * float(sigma_sec_m) / room if room > 0.0 else float("inf")


def track_length_m(sigma_sec_m: float, k: float, psi_tol_rad: float, theta_goal_rad: float, *, L_r_m: float, L_min_m: float,
                   m_E_m: float = 0.0) -> tuple:
    """S 추적 기선 b = min(b_need(σ), cap), cap = L_r − L_min − m_E (S 사이트가 E 에서 최소 거리 + E 여유를 남기는 기하 상한).
    σ = 단면 잡음 상한 (설계: tol_root/k — b_need = b* = √2·tol_root/(ψ_tol − θ_goal); 설계 허용이 없으면 측정 바닥). → (b, b_need, cap)."""
    b_need = baseline_m(sigma_sec_m, k, psi_tol_rad, theta_goal_rad)
    cap = float(L_r_m) - float(L_min_m) - max(0.0, float(m_E_m))
    return min(b_need, cap), b_need, cap


def footprint_m(fov_m: float, psi_rad: float, x_m: float, e_hit_m: float) -> float:
    """f(ψ, x) = (FOV/2 + |x|)·sin|ψ| + e_hit: 그루브 교차점에서 레이저 선 끝까지의 심 방향 반폭 (그루브가 선 위 x 에 있으면 먼 쪽 팔이
    FOV/2 + |x|) + 착점 도착 허용. x 를 모르면 (NaN: 그 드웰이 그루브를 읽지 못했다) 0 으로 바꾸지 않고 불확도를 전파한다 — 그루브는 시야
    어디든 있을 수 있어 |x| ≤ FOV/2, 먼 쪽 팔은 선 전체 (FOV)."""
    h = 0.5 * float(fov_m)
    x = abs(float(x_m)) if math.isfinite(float(x_m)) else h
    return (h + x) * math.sin(min(abs(float(psi_rad)), 0.5 * math.pi)) + max(0.0, float(e_hit_m))


def seam_frame(n, u) -> tuple:
    """(n, u, w): n 단위, u 를 n 평면에 투영한 단위, w = n × u."""
    n = unit(n)
    u = unit(np.asarray(u, dtype=float) - n * float(np.asarray(u, dtype=float) @ n))
    return n, u, unit(np.cross(n, u))


def yaw_to_seam_rad(R_opt, n, u) -> float:
    """δ (4.2 절): 판 법선 n 둘레로 레이저 선 (X_opt 의 판면 투영 e) 을 심에 수직인 축 w = ±n×u (e·w ≥ 0 인 쪽) 에 얹는 부호 있는 각.
    |δ| < 90° — 180° 가 될 수 없어 진행 부호·토치 쪽·손목 형태가 유지된다."""
    n, u, w = seam_frame(n, u)
    X = np.asarray(R_opt, dtype=float)[:3, 0]
    e = X - n * float(X @ n)
    if float(np.linalg.norm(e)) < 1e-9:
        return float("nan")
    e = unit(e)
    if float(e @ w) < 0.0:
        w = -w
    return math.atan2(float(np.cross(e, w) @ n), float(e @ w))


def laser_footprint_s(T_base_opt, c, u, n, fov_m: float) -> tuple:
    """레이저 선 양 끝 (광학 원점에서 ±FOV/2, 광축이 판 평면 (c, n) 을 찌르는 점 둘레) 의 심 진행 좌표 (u, c 기준) → (s_min, s_max)."""
    T = np.asarray(T_base_opt, dtype=float)
    o, X, Z = T[:3, 3], T[:3, 0], T[:3, 2]
    n = unit(n)
    den = float(Z @ n)
    if abs(den) < 1e-9:
        return float("nan"), float("nan")
    t = float((np.asarray(c, dtype=float) - o) @ n) / den
    hit = o + Z * t
    e = X - n * float(X @ n)
    h = 0.5 * float(fov_m)
    s = [float((hit + e * sg * h - np.asarray(c, dtype=float)) @ unit(u)) for sg in (-1.0, 1.0)]
    return min(s), max(s)


# --------------------------------------------------------------------------------------
# 프레임 행
# --------------------------------------------------------------------------------------
@dataclass
class TrackRows:
    """스트림 프레임 행 (재구성 시각 순서; 센서 프레임 값 — 캘리브가 바뀌면 (광학 분기 뒤집기) 기반 좌표만 다시 계산한다)."""
    T_bf: np.ndarray                       # (N,4,4) 녹화 시각 보간 플랜지 자세 (reconstruct 와 같은 시계 동기·dt_s)
    cand_xz: np.ndarray                    # (N,3,2) 센서 (x, z) [mm]: 왼쪽 코너, 오른쪽 코너, 상부 개구 중심 (없으면 NaN)
    design: np.ndarray                     # (N,) 설계 일치 (matches_design)
    groove: np.ndarray                     # (N,) 그루브로 분류 (groove_found)
    moving: np.ndarray                     # (N,) 이동 중 프레임
    plate_xz: list = field(default_factory=list)   # 행마다 판 인라이어 점 (M_i, 2) float32 [mm]
    x_c: np.ndarray = field(default_factory=lambda: np.zeros(0))     # (N,) 상부 개구 중심 센서 x [mm]
    part: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    counter: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def n(self) -> int:
        return int(np.asarray(self.design).shape[0])


def rows_from_features(feats: list, T_bf, X, Z, moving, spec, *, part: int = 0, counter=None) -> TrackRows:
    """프레임별 CrossSectionFeatures (None = 계산 안 함) + 그 행의 플랜지 자세·유효 점 (무효 NaN) → TrackRows."""
    from .seam_features import matches_design
    N = len(feats)
    cand = np.full((N, 3, 2), np.nan)
    design = np.zeros(N, dtype=bool)
    groove = np.zeros(N, dtype=bool)
    x_c = np.full(N, np.nan)
    plates = []
    for i, f in enumerate(feats):
        if f is None:
            plates.append(np.zeros((0, 2), dtype=np.float32))
            continue
        for j, xz in enumerate((f.top_corner_left_xz, f.top_corner_right_xz)):
            if xz is not None and len(xz) == 2:
                cand[i, j] = [float(xz[0]), float(xz[1])]
        if f.groove_found and math.isfinite(float(f.groove_centre_x_mm)) and math.isfinite(float(f.groove_centre_z_mm)):
            cand[i, 2] = [float(f.groove_centre_x_mm), float(f.groove_centre_z_mm)]
            x_c[i] = float(f.groove_centre_x_mm)
        groove[i] = bool(f.groove_found)
        design[i] = bool(f.groove_found and f.ok and matches_design(f, spec))
        idx = f.plate_inlier_idx()
        xi = np.asarray(X[i], dtype=float).reshape(-1)
        zi = np.asarray(Z[i], dtype=float).reshape(-1)
        if idx.size:
            pts = np.column_stack([xi[idx], zi[idx]])
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            plates.append(pts.astype(np.float32))
        else:
            plates.append(np.zeros((0, 2), dtype=np.float32))
    return TrackRows(T_bf=np.asarray(T_bf, dtype=float).reshape(-1, 4, 4), cand_xz=cand, design=design, groove=groove,
                     moving=np.asarray(moving, dtype=bool).reshape(-1), plate_xz=plates, x_c=x_c,
                     part=np.full(N, int(part), dtype=np.int64),
                     counter=(np.asarray(counter, dtype=np.int64).reshape(-1) if counter is not None else np.arange(N, dtype=np.int64)))


def concat_rows(parts: list) -> TrackRows:
    """여러 이동 (S_track, S_track2) 의 행을 이어 붙인다 (시각 순서)."""
    parts = [p for p in parts if p is not None and p.n > 0]
    if not parts:
        raise ValueError("concat_rows: no rows")
    if len(parts) == 1:
        return parts[0]
    return TrackRows(T_bf=np.concatenate([p.T_bf for p in parts]), cand_xz=np.concatenate([p.cand_xz for p in parts]),
                     design=np.concatenate([p.design for p in parts]), groove=np.concatenate([p.groove for p in parts]),
                     moving=np.concatenate([p.moving for p in parts]), plate_xz=sum((list(p.plate_xz) for p in parts), []),
                     x_c=np.concatenate([p.x_c for p in parts]), part=np.concatenate([p.part for p in parts]),
                     counter=np.concatenate([p.counter for p in parts]))


def moving_mask(origins, still_tol_m: float) -> tuple:
    """이동 하나의 광학 원점 이력 (N,3) → (이동 중 마스크, 진행 비율 0..1): 시작·끝 자세에서 도착 허용 (still_tol) 안의 프레임은 정지."""
    o = np.asarray(origins, dtype=float).reshape(-1, 3)
    N = o.shape[0]
    if N == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    d = o[-1] - o[0]
    D = float(np.linalg.norm(d))
    if not (D > 2.0 * float(still_tol_m)):
        return np.zeros(N, dtype=bool), np.zeros(N)
    p = (o - o[0]) @ (d / D)
    return (p > float(still_tol_m)) & (p < D - float(still_tol_m)), np.clip(p / D, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# 결과
# --------------------------------------------------------------------------------------
@dataclass
class SeamEstimate:
    c: np.ndarray                 # 직선 위 점 (base, m) — 마지막 직선 위 프레임의 궤적 점을 직선에 투영
    u: np.ndarray                 # 심 방향 (판 평면 안, 진행 쪽: u·t̂0 > 0; 기울기 포함 3D)
    n: np.ndarray                 # 판 법선 (추적 판 점 평면 적합, 센서 쪽)
    track: str                    # 'left' | 'right' | 'centre' (잔차 최소 후보)
    n_frames: int
    n_inliers: int
    coverage: float
    mad_m: float                  # 고른 궤적의 직선 잔차 (중앙값)
    s_first: float                # 직선 위·설계 일치 프레임 (레이저 평면 ∩ 직선) 의 진행 좌표 범위 (c 기준, u 방향)
    s_last: float
    sections: np.ndarray          # (N,2) 프레임별 (s, 일관 0/1)
    length_m: float               # 기선 = 일관 단면 범위 s_last − s_first
    sigma_psi_model: float        # √2·σ_단면/length [rad] — 여유 전파용 (σ_단면 = max(설계 tol_root/k, 잰 궤적 잔차 σ))
    source: str                   # 'S_track' | 'transit'
    w: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))      # n × u
    centre_offset_m: float = 0.0  # 상부 개구 중심 − 궤적 점 (w 방향 중앙값) — 예측 x 를 드웰 x_c 와 같은 점으로
    sigma_psi_self: float = float("nan")    # 자기 보고 방향 σ [rad] (기록만)
    move_m: float = float("nan")            # 이동 기선 (첫·끝 광학 원점)
    arrival_consistent: Optional[bool] = None   # 이동 끝 정지 프레임이 일관 단면인가 (없으면 마지막 이동 프레임)
    s_arrival: float = float("nan")         # 마지막 프레임의 레이저 평면 ∩ 직선 (c 기준)
    tracks: dict = field(default_factory=dict)  # 후보 궤적별 {coverage, mad_m, n}
    groove_mismatch: bool = False
    sigma_sec_m: float = float("nan")       # σψ_model 에 쓴 단면 잡음 = max(설계 상한, 잰 궤적 잔차)
    sigma_sec_meas_m: float = float("nan")  # 고른 궤적의 잰 잔차 σ (LMedS, 하한 센서 양자화)
    sigma_sec_source: str = ""              # 'design' | 'measured'
    first_on_line: int = -1                 # 직선 위 첫 이동 중 프레임 (덮임의 시작)

    def point_at(self, s: float) -> np.ndarray:
        return np.asarray(self.c, dtype=float) + np.asarray(self.u, dtype=float) * float(s)

    def s_of(self, P) -> float:
        return float((np.asarray(P, dtype=float).reshape(3) - np.asarray(self.c, dtype=float)) @ np.asarray(self.u, dtype=float))

    def to_dict(self) -> dict:
        return {"c": np.asarray(self.c).tolist(), "u": np.asarray(self.u).tolist(), "n": np.asarray(self.n).tolist(),
                "w": np.asarray(self.w).tolist(), "track": self.track, "n_frames": int(self.n_frames), "n_inliers": int(self.n_inliers),
                "coverage": float(self.coverage), "mad_mm": float(self.mad_m) * 1e3, "s_first_mm": float(self.s_first) * 1e3,
                "s_last_mm": float(self.s_last) * 1e3, "length_mm": float(self.length_m) * 1e3,
                "sigma_psi_model_deg": math.degrees(float(self.sigma_psi_model)), "sigma_psi_self_deg": math.degrees(self.sigma_psi_self)
                if math.isfinite(self.sigma_psi_self) else None, "source": self.source, "centre_offset_mm": float(self.centre_offset_m) * 1e3,
                "move_mm": float(self.move_m) * 1e3, "arrival_consistent": self.arrival_consistent,
                "s_arrival_mm": float(self.s_arrival) * 1e3, "tracks": self.tracks, "groove_mismatch": bool(self.groove_mismatch),
                "sigma_sec_mm": float(self.sigma_sec_m) * 1e3, "sigma_sec_meas_mm": float(self.sigma_sec_meas_m) * 1e3,
                "sigma_sec_source": self.sigma_sec_source}


# --------------------------------------------------------------------------------------
# 맞춤 (4.1 절)
# --------------------------------------------------------------------------------------
def _points_base(T_bo: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """(N,2) 센서 (x, z) [mm] + 자세 (N,4,4) → (N,3) base [m] (NaN 유지)."""
    x = xz[:, 0] * 1e-3
    z = xz[:, 1] * 1e-3
    return T_bo[:, :3, 3] + x[:, None] * T_bo[:, :3, 0] + z[:, None] * T_bo[:, :3, 2]


def _line_fit(P: np.ndarray, *, k: float, floor_m: float) -> Optional[tuple]:
    """3D 직선 (ransac_line_3d LMedS) → 잔차 k·σ 로 두 번 다시 맞춘다 → (c, u, 인라이어 마스크, 거리, σ) 또는 None."""
    if P.shape[0] < 5:
        return None
    c, u, inl, _rms = ransac_line_3d(P, k=float(k), floor_m=float(floor_m), rng=np.random.default_rng(0))
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(u))):
        return None
    dist = np.zeros(P.shape[0])
    sig = float(floor_m)
    for _ in range(2):
        rel = P - c
        dist = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
        sig = max(float(lmeds_sigma(dist, 2)), float(floor_m))
        inl = dist <= float(k) * sig
        if int(inl.sum()) < 2:
            return None
        Q = P[inl]
        c = Q.mean(axis=0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        u = unit(Vt[0])
    rel = P - c
    dist = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    inl = dist <= float(k) * sig
    return c, u, inl, dist, sig


def _plane_normal(P: np.ndarray, *, k: float, floor_m: float) -> Optional[np.ndarray]:
    """판 점 평면 (최소제곱 → 부호 있는 거리의 k·1.4826·MAD 절사, 두 번) → 법선 (부호 미정) 또는 None."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if P.shape[0] < 3:
        return None
    sel = np.ones(P.shape[0], dtype=bool)
    n = None
    for _ in range(3):
        Q = P[sel]
        if Q.shape[0] < 3:
            break
        c = Q.mean(axis=0)
        _, sv, Vt = np.linalg.svd(Q - c, full_matrices=False)
        if sv.size < 2 or not (sv[1] > 0.0):
            break
        n = unit(Vt[-1])
        r = (P - c) @ n
        med = float(np.median(r[sel]))
        sig = max(MAD_TO_SIGMA * float(np.median(np.abs(r[sel] - med))), float(floor_m))
        sel = np.abs(r - med) <= float(k) * sig
    return n


def _fit_one(rows: TrackRows, P: np.ndarray, use: np.ndarray, *, k: float, floor_m: float) -> Optional[dict]:
    """궤적 하나: use 행의 점으로 직선 → {c, u, inlier rows (N,), coverage, mad, sigma, span, first}. 덮임 = 직선 위 이동 중 프레임 / 그
    직선 위 첫 이동 중 프레임부터의 이동 중 프레임 (그루브가 나타나기 전 프레임은 세지 않는다)."""
    sel = use & np.all(np.isfinite(P), axis=1)
    idx = np.flatnonzero(sel)
    fit = _line_fit(P[idx], k=k, floor_m=floor_m)
    if fit is None:
        return None
    c, u, inl, dist, sig = fit
    rows_inl = np.zeros(rows.n, dtype=bool)
    rows_inl[idx[inl]] = True
    on = rows_inl & rows.moving
    first = int(np.flatnonzero(on)[0]) if on.any() else -1
    n_mov = int(rows.moving[first:].sum()) if first >= 0 else 0
    cov = float(on.sum()) / n_mov if n_mov else 0.0
    t = (P[idx[inl]] - c) @ u
    span = float(t.max() - t.min()) if t.size else 0.0
    return {"c": c, "u": u, "inl": rows_inl, "coverage": cov, "mad_m": float(np.median(dist[inl])) if inl.any() else float("inf"),
            "sigma_m": float(sig), "n": int(inl.sum()), "span_m": span, "first": first}


def _best_track(rows: TrackRows, T_bo: np.ndarray, use: np.ndarray, *, k: float, floor_m: float) -> tuple:
    """후보 궤적마다 맞춤 → (가장 좋은 (덮임 ≥ ½ 중 방향 불확도 = 잔차 중앙값 / 직선 위 범위 최소) 이름 또는 None, {이름: 맞춤})."""
    fits = {}
    for j, name in enumerate(TRACKS):
        P = _points_base(T_bo, rows.cand_xz[:, j, :])
        f = _fit_one(rows, P, use, k=k, floor_m=floor_m)
        if f is not None:
            f["P"] = P
            fits[name] = f
    ok = [nm for nm, f in fits.items() if f["coverage"] >= DESIGN_HALF and f["span_m"] > 0.0]
    best = min(ok, key=lambda nm: (fits[nm]["mad_m"] / fits[nm]["span_m"], TRACKS.index(nm))) if ok else None
    return best, fits


def fit_track(rows: TrackRows, calib, *, k: float = K_DEFAULT, tol_root_m: Optional[float], t_hat0, floor_m: float,
              source: str = "S_track") -> tuple:
    """TrackRows + 캘리브 → (SeamEstimate 또는 None, 결과 종류 'ok'|'late'|'none'|'groove_mismatch', 기록 dict). 모듈 docstring 규칙.
    tol_root_m: 설계 루트 위치 허용 (없으면 None — σψ_model 은 잰 궤적 잔차로)."""
    T_bo = np.asarray(rows.T_bf, dtype=float) @ np.asarray(calib.T_flange_opt, dtype=float)
    N = rows.n
    info = {"n_frames": N, "n_moving": int(rows.moving.sum()), "n_design": int(rows.design.sum()), "n_groove": int(rows.groove.sum()),
            "source": source}
    if N == 0:
        return None, "none", {**info, "why": "no frames"}
    best, fits = _best_track(rows, T_bo, rows.design, k=k, floor_m=floor_m)
    info["tracks"] = {nm: {"coverage": round(f["coverage"], 4), "mad_mm": round(f["mad_m"] * 1e3, 4), "n": f["n"],
                           "span_mm": round(f["span_m"] * 1e3, 3)} for nm, f in fits.items()}
    if best is None:
        # 그루브로 분류되고 직선 위인데 설계와 맞지 않는 프레임이 이동의 절반 이상 → 선은 시편 위에서 설계와 다른 그루브를 본다
        mis_best, mis = _best_track(rows, T_bo, rows.groove & ~rows.design, k=k, floor_m=floor_m)
        info["mismatch_tracks"] = {nm: {"coverage": round(f["coverage"], 4), "n": f["n"]} for nm, f in mis.items()}
        if mis_best is not None:
            return None, "groove_mismatch", {**info, "why": f"a non-design groove on a line over {mis[mis_best]['coverage'] * 100:.0f} % "
                                                            "of the move"}
        if not rows.design.any():
            return None, "none", {**info, "why": "no design-consistent frame"}
        # 직선 위 덮임은 이미 직선 위 첫 프레임부터 센다 — 맞춘 직선이 없거나 (나타난 뒤 프레임이 직선 맞춤에 모자라다) 덮임이 모자라면:
        # 늦게 나타남 = 설계 일치 프레임이 처음 나타난 곳 (첫 이동 프레임 뒤 — 이동 끝 정지 프레임 포함) 부터 끝까지의 프레임 중 절반 이상이
        # 설계 일치 (나타난 뒤 이어졌다 — 같은 방향으로 더 가면 기선이 생긴다), 아니면 흩어짐 ('none')
        mv = np.flatnonzero(rows.moving)
        start = int(mv[0]) if mv.size else 0
        dz = np.flatnonzero(rows.design[start:])
        cov_after, first_p = 0.0, None
        if dz.size:
            i0 = start + int(dz[0])
            cov_after = float(rows.design[i0:].sum()) / float(N - i0)
            first_p = float(np.searchsorted(mv, i0)) / float(max(1, mv.size))
        info.update(coverage_after_first=round(cov_after, 4), first_seen_fraction=first_p)
        if cov_after >= DESIGN_HALF:
            return None, "late", {**info, "why": f"the groove appeared at {first_p * 100:.0f} % of the move and stayed (too few frames "
                                                 "for a line)"}
        return None, "none", {**info, "why": "design-consistent frames are scattered (no line over half the frames since the first)"}
    f = fits[best]
    c, u, inl = f["c"], f["u"], f["inl"]
    # 판 법선: 직선 위 프레임들의 판 점 평면 — 프레임마다 그루브 양쪽 판의 중앙 점 (판 인라이어의 x·z 중앙값; 한 판 위 점들은 레이저 선 위
    # 직선이라 중앙값이 그 위의 점) 두 개씩: 심 방향 (이동) × 심 가로 (두 판) 로 평면이 정해지고, 프레임당 ~1000 점을 다 넣는 것과 같은
    # 평면을 이동 뒤 지연 없이 (실기 트랜싯 1059 프레임 × 1300 점 = 1.4 M 점 SVD 대신 ~2000 점)
    Q = []
    for i in np.flatnonzero(inl):
        xz = np.asarray(rows.plate_xz[i], dtype=float).reshape(-1, 2)
        if not xz.shape[0]:
            continue
        xc = float(rows.x_c[i]) if rows.x_c.size == rows.n and math.isfinite(float(rows.x_c[i])) else float("nan")
        sides = [xz[xz[:, 0] < xc], xz[xz[:, 0] > xc]] if math.isfinite(xc) else [xz]
        Ti = T_bo[i]
        for sd in sides:
            if sd.shape[0]:
                m = np.median(sd, axis=0)
                Q.append(Ti[:3, 3] + (m[0] * 1e-3) * Ti[:3, 0] + (m[1] * 1e-3) * Ti[:3, 2])
    n = _plane_normal(np.asarray(Q, dtype=float).reshape(-1, 3), k=k, floor_m=floor_m)
    o_mean = T_bo[inl][:, :3, 3].mean(axis=0)
    if n is None:
        # 판 점이 없으면 광축 반대 방향을 u 에 직교화 (기록)
        Zm = unit(T_bo[inl][:, :3, 2].mean(axis=0))
        n = unit(-Zm - u * float(-Zm @ u))
        info["normal_source"] = "optical_axis (no plate points)"
    else:
        info["normal_source"] = "plate points"
    if float(n @ (o_mean - c)) < 0.0:
        n = -n
    n, u, w = seam_frame(n, u)
    if float(u @ unit(np.asarray(t_hat0, dtype=float))) < 0.0:
        u = -u
        w = -w
    last = int(np.flatnonzero(inl)[-1])
    P_last = f["P"][last]
    c = c + u * float((P_last - c) @ u)
    s_all = np.asarray(laser_plane_line_param(T_bo, c, u), dtype=float).reshape(-1)
    s_in = s_all[inl & np.isfinite(s_all)]
    s_first, s_last = (float(s_in.min()), float(s_in.max())) if s_in.size else (float("nan"), float("nan"))
    span = s_last - s_first if s_in.size else 0.0
    sig_design = float(tol_root_m) / float(k) if (tol_root_m is not None and math.isfinite(float(tol_root_m))
                                                   and float(tol_root_m) > 0.0) else 0.0
    # 잰 단면 잡음 = 직선 위 점의 측면 (w, 방향 오차를 만드는 성분) 잔차의 강건 σ (1.4826·MAD, 하한 센서 양자화) — 3D 거리의 LMedS σ (dof 2)
    # 는 측면 한 성분 잡음을 0.57 배로 작게 읽는다
    r_w = (f["P"][inl] - c) @ w
    r_w = r_w[np.isfinite(r_w)]
    sig_meas = max(MAD_TO_SIGMA * float(np.median(np.abs(r_w - np.median(r_w)))), float(floor_m)) if r_w.size else float(f["sigma_m"])
    sig_sec = max(sig_design, sig_meas)
    sig_model = math.sqrt(2.0) * sig_sec / span if span > 0.0 else float("inf")
    n_in = int(inl.sum())
    sig_self = (f["sigma_m"] * math.sqrt(12.0 / n_in) / span) if (span > 0.0 and n_in > 0) else float("nan")
    # 상부 개구 중심 − 궤적 점 (w 방향) — 예측 x 를 드웰이 읽는 점 (상부 중심) 으로
    Pc = _points_base(T_bo, rows.cand_xz[:, 2, :])
    dc = (Pc[inl] - f["P"][inl]) @ w
    dc = dc[np.isfinite(dc)]
    off = float(np.median(dc)) if dc.size else 0.0
    # 이동 끝 정지 프레임 (마지막 이동 프레임 뒤) 이 일관 단면인가
    mv = np.flatnonzero(rows.moving)
    tail = np.arange(int(mv[-1]) + 1, N) if mv.size else np.arange(N)
    if tail.size == 0 and mv.size:
        tail = np.array([int(mv[-1])])
    arr_ok = bool(inl[tail].mean() >= DESIGN_HALF) if tail.size else None
    o_all = T_bo[:, :3, 3]
    est = SeamEstimate(c=c, u=u, n=n, w=w, track=best, n_frames=N, n_inliers=n_in, coverage=float(f["coverage"]),
                       mad_m=float(f["mad_m"]), s_first=s_first, s_last=s_last,
                       sections=np.column_stack([s_all, inl.astype(float)]), length_m=float(span), sigma_psi_model=float(sig_model),
                       source=source, centre_offset_m=off, sigma_psi_self=float(sig_self),
                       move_m=float(np.linalg.norm(o_all[-1] - o_all[0])), arrival_consistent=arr_ok,
                       s_arrival=float(s_all[-1]) if np.isfinite(s_all[-1]) else float("nan"),
                       tracks=info["tracks"], sigma_sec_m=float(sig_sec), sigma_sec_meas_m=sig_meas,
                       sigma_sec_source="design" if sig_design >= sig_meas else "measured", first_on_line=int(f["first"]))
    return est, "ok", info


def predict_x_at(T_base_opt, est: SeamEstimate) -> float:
    """그 광학 자세의 레이저 평면과 (상부 중심 쪽으로 옮긴) 심 직선의 교점 → 센서 x [mm] (드웰 x_c 와 같은 점)."""
    T = np.asarray(T_base_opt, dtype=float)
    q0 = np.asarray(est.c, dtype=float) + np.asarray(est.w, dtype=float) * float(est.centre_offset_m)
    t = float(laser_plane_line_param(T, q0, est.u))
    if not math.isfinite(t):
        return float("nan")
    P = q0 + unit(est.u) * t
    return float((P - T[:3, 3]) @ T[:3, 0]) * 1e3


def consistent_upto(est: SeamEstimate, s: float) -> bool:
    """s 가 스트림이 일관되게 읽은 단면 범위 [s_first, s_last] 안인가."""
    return bool(math.isfinite(float(s)) and est.s_first <= float(s) <= est.s_last)
