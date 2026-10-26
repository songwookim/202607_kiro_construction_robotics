"""부분 충전 그루브 특징 추출 (2026-10-03, 다층 멀티패스 2단계 — 공정: 패스 1–3 충전 + 패스 4 위빙 캡).

패스 k 뒤 프로파일 1 개 → 다음 패스 계획에 필요한 양:
  - 남은 단면적 [mm²] = 판 상면선과 측정 표면 사이 (두 상부 모서리 사이) 적분 — 비드 모양 모델에 기대지 않는다.
  - 토우 (충전 표면 ∩ 벽) 양쪽 위치·높이, 토우 폭, 토우 높이 차 (비드 쏠림).
  - 충전 표면 깊이 (판 상면 아래: 가운데·최소·최대), 크라운 (토우를 잇는 선 대비 + 볼록 / − 오목) 과 꼭대기 위치.
  - 캡 상태 (표면이 판 상면 위): 보강 높이·폭·단면적, 토우 미충전 (판 상면 아래로 파인 곳) 단면적.
빈 그루브도 같은 규칙 (표면 = 뒷댐재 바닥 → 남은 단면적 = 그루브 단면적, 토우 폭 = 루트갭).

절차 (프로파일 1 개; 프레임·판 검출은 seam_features 와 같다):
  1. 유효점 → h = −z. 양 끝에서 판 상면 영역 성장 (_grow_plate) → 판 직선 → de-tilt (판 정렬 프레임 x', h').
  2. 판 상면선 top(x') (왼 판 직선 → 오른 판 직선, 그루브 폭에서 선형 혼합 — hi-lo 를 담는다); 내부점 깊이 d = top − h.
  3. 상태: 내부점이 판 위 (d < −cap_min) 가 더 많으면 'cap', 아래가 많으면 'groove'.
  4. groove: 각 점의 국소 기울기 (열 순서 이웃 PCA) → 완만한 점 (< surface_max_slope) 으로 RANSAC 2차식 = 충전 표면 (인라이어);
     표면 아래로 멀리 떨어진 점 = 반사 (제외). 판 끝에 이어진 가파른 점 = 벽 → 직선 (RANSAC; 수직에서 12° 안이면 '수직벽').
     벽 점이 모자라면 bevel_side_hint 쪽은 bevel_angle_hint 로, 그 외는 판 끝의 수직벽으로 둔다 (표면이 판 상면 가까이 차서 보이는
     벽이 짧을 때 — 그만큼 빠지는 삼각형도 작다). 상부 모서리 = 판 직선 ∩ 벽, 토우 = 벽 ∩ 표면 2차식 (모서리에서 내려가며 첫 교점).
     높이 H(x') = 벽 (모서리~토우) / 측정 표면 (0.1 mm 구간 중앙값, 빈 구간은 2차식) / 벽 → 남은 단면적·깊이·크라운.
  5. cap: 토우 = 판 영역 끝 (판에서 표면이 올라오는 곳), 구간 중앙값 프로파일 → 보강 높이·단면적·폭, 판 상면 아래 미충전 단면적.

출력 점 (토우·표면 가운데·가장 깊은 점) 은 센서 프레임 (x, z) [mm] 로도 돌려 base 3D 점으로 바꿀 수 있다 (along-scan).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .seam_features import (Line2D, _attached, _grow_plate, _plate_info, _rot, _unrot, _xz, fit_line_pca, ransac_line)
from .validity import dx_linear_coeffs

__all__ = ["FillProfileFeatures", "extract_fill_features", "FillScanFeatures", "extract_fill_features_along_scan",
           "ransac_quadratic"]


# --------------------------------------------------------------------------------------
# 결과
# --------------------------------------------------------------------------------------
@dataclass
class FillProfileFeatures:
    ok: bool = False
    state: str = "unknown"                     # groove | cap | no_groove | unknown
    flags: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    n_valid: int = 0
    tilt_deg: float = float("nan")             # 판 상면 기울기 (센서 프레임)
    detilt_phi_deg: float = float("nan")
    hi_lo_mm: float = float("nan")             # 오른 판 − 왼 판 (그루브 가운데)
    corner_left: Optional[list] = None         # 상부 모서리 (x', h') [mm] (cap 이면 None)
    corner_right: Optional[list] = None
    opening_mm: float = float("nan")
    toe_left: Optional[list] = None            # 토우 (x', h') [mm]
    toe_right: Optional[list] = None
    toe_width_mm: float = float("nan")
    toe_tilt_mm: float = float("nan")          # 오른 토우 − 왼 토우 높이
    toe_depth_left_mm: float = float("nan")    # 판 상면선 아래 토우 깊이 (+ 아래)
    toe_depth_right_mm: float = float("nan")
    remaining_area_mm2: float = float("nan")   # 판 상면선 아래 빈 단면적
    reinforcement_area_mm2: float = float("nan")  # 판 상면선 위 단면적 (캡)
    depth_center_mm: float = float("nan")      # 토우 가운데에서 표면 깊이 (+ 판 상면 아래)
    depth_min_mm: float = float("nan")         # 표면에서 가장 얕은 깊이 (볼록 꼭대기)
    depth_max_mm: float = float("nan")         # 가장 깊은 깊이
    crown_mm: float = float("nan")             # 토우를 잇는 선 대비 (+ 볼록 / − 오목)
    crown_peak_frac: float = float("nan")      # 크라운 극값 위치 (왼 토우 0 → 오른 토우 1)
    cap_height_mm: float = float("nan")        # 판 상면선 위 최대 높이 (cap)
    cap_width_mm: float = float("nan")         # 캡 토우 사이 폭 (cap)
    surface_coverage: float = float("nan")     # 토우 사이 폭 중 측정점이 있는 비율 (0.1 mm 구간)
    surface_rms_mm: float = float("nan")       # 표면 인라이어의 2차식 잔차 RMS
    n_surface: int = 0
    n_reflections: int = 0                     # 표면 아래로 떨어진 점 (반사, 제외)
    n_stray: int = 0                           # 표면·벽 어디에도 속하지 않는 점 (제외)
    wall_left: dict = field(default_factory=dict)   # {kind: bevel|vertical|hint|edge, angle_from_vertical_deg, n, visible_mm}
    wall_right: dict = field(default_factory=dict)
    toe_left_xz: Optional[list] = None         # 센서 프레임 (x, z) [mm]
    toe_right_xz: Optional[list] = None
    centre_xz: Optional[list] = None           # 토우 가운데의 표면 점
    low_xz: Optional[list] = None              # 가장 깊은 표면 점

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def report(self) -> str:
        if self.state == "cap":
            return (f"cap: height {self.cap_height_mm:.2f} width {self.cap_width_mm:.2f} reinforcement {self.reinforcement_area_mm2:.2f} "
                    f"mm2 underfill {self.remaining_area_mm2:.2f} mm2 flags={self.flags}")
        return (f"{self.state}: remaining {self.remaining_area_mm2:.2f} mm2, depth centre {self.depth_center_mm:.2f} "
                f"[{self.depth_min_mm:.2f}..{self.depth_max_mm:.2f}] crown {self.crown_mm:+.2f} @ {self.crown_peak_frac:.2f}, toes "
                f"{self.toe_width_mm:.2f} apart (tilt {self.toe_tilt_mm:+.2f}), opening {self.opening_mm:.2f}, coverage "
                f"{self.surface_coverage:.2f} flags={self.flags}")


# --------------------------------------------------------------------------------------
# 도우미
# --------------------------------------------------------------------------------------
def ransac_quadratic(x, h, thr_mm: float = 0.3, iters: int = 200, rng: Optional[np.random.Generator] = None,
                     min_pts: int = 5, min_span_mm: float = 0.5) -> tuple:
    """RANSAC 2차식 h = c0 + c1·x + c2·x² → (계수 (3,), 인라이어 마스크). 후보를 한 번에 계산 (3 점 표본, x 간격 ≥ min_span/2)."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    M = x.size
    if M < max(3, min_pts) or float(np.ptp(x)) < min_span_mm:
        return None, np.zeros(M, dtype=bool)
    rng = rng if rng is not None else np.random.default_rng(0)
    I = rng.integers(0, M, (int(iters), 3))
    X = x[I]
    Xs = np.sort(X, axis=1)
    good = (np.min(np.diff(Xs, axis=1), axis=1) >= 0.25 * min_span_mm)
    if not good.any():
        return None, np.zeros(M, dtype=bool)
    X, Hs = X[good], h[I[good]]
    A = np.stack([np.ones_like(X), X, X * X], axis=2)
    coef = np.linalg.solve(A, Hs[..., None])[..., 0]                  # (n, 3)
    pred = coef[:, :1] + coef[:, 1:2] * x[None, :] + coef[:, 2:3] * (x * x)[None, :]
    cnt = (np.abs(pred - h[None, :]) < thr_mm).sum(axis=1)
    k = int(np.argmax(cnt))
    if int(cnt[k]) < min_pts:
        return None, np.zeros(M, dtype=bool)
    inl = np.abs(pred[k] - h) < thr_mm
    c = coef[k]
    for _ in range(3):                                                 # 최소제곱 정련
        V = np.column_stack([np.ones(int(inl.sum())), x[inl], x[inl] ** 2])
        c = np.linalg.lstsq(V, h[inl], rcond=None)[0]
        inl_new = np.abs(c[0] + c[1] * x + c[2] * x * x - h) < thr_mm
        if int(inl_new.sum()) < min_pts or np.array_equal(inl_new, inl):
            break
        inl = inl_new
    return np.asarray(c, dtype=float), inl


def _quad(c, x):
    x = np.asarray(x, dtype=float)
    return c[0] + c[1] * x + c[2] * x * x


def _x_at_h(ln: Line2D, hh: float) -> float:
    if abs(ln.d[1]) < 1e-12:
        return float("nan")
    return float(ln.p[0] + (hh - ln.p[1]) * ln.d[0] / ln.d[1])


def _line_quad_hits(ln: Line2D, c) -> list:
    """직선 p + t·d 와 2차식의 교점 [(x, h)]."""
    px, ph = ln.p
    dx, dh = ln.d
    A = c[2] * dx * dx
    B = 2 * c[2] * px * dx + c[1] * dx - dh
    C = c[2] * px * px + c[1] * px + c[0] - ph
    ts = []
    if abs(A) < 1e-12:
        if abs(B) > 1e-12:
            ts = [-C / B]
    else:
        disc = B * B - 4 * A * C
        if disc >= 0:
            r = math.sqrt(disc)
            ts = [(-B - r) / (2 * A), (-B + r) / (2 * A)]
    return [(float(px + t * dx), float(ph + t * dh)) for t in ts]


def _local_slope_deg(xp: np.ndarray, hp: np.ndarray, idx: np.ndarray, k: int, rmax: float) -> np.ndarray:
    """열 순서 이웃 (±k, 거리 ≤ rmax) PCA 방향의 수평 기준 각 [deg] (0 수평 … 90 수직)."""
    N = xp.size
    out = np.full(idx.size, np.nan)
    for j, i in enumerate(idx):
        a, b = max(0, i - k), min(N, i + k + 1)
        xs, hs = xp[a:b], hp[a:b]
        near = np.hypot(xs - xp[i], hs - hp[i]) <= rmax
        if int(near.sum()) < 3:
            continue
        ln = fit_line_pca(xs[near], hs[near])
        out[j] = math.degrees(math.atan2(abs(ln.d[1]), abs(ln.d[0])))
    return out


def _ray_dx_per_depth(xp_c: float, hp_c: float, phi: float) -> float:
    """판 정렬 프레임 점 (x', h') 을 지나는 센서 광선이 아래로 1 mm 내려갈 때의 x' 변화 (광선 부채꼴: x = k·(a + b·z), 가상 원점
    z0 = −a/b — synth 와 같은 모델)."""
    a, b = dx_linear_coeffs()
    z0 = -a / b
    xs, hs = _unrot(float(xp_c), float(hp_c), phi)
    zs = -hs
    dxp, dhp = _rot(xs / (zs - z0), -1.0, phi)
    return float(dxp / -dhp) if abs(dhp) > 1e-9 else 0.0


def _binned(x: np.ndarray, h: np.ndarray, x0: float, x1: float, step: float) -> tuple:
    """[x0, x1] 를 step 구간으로 → (구간 가운데 x, 중앙값 h (없으면 NaN))."""
    n = max(1, int(math.ceil((x1 - x0) / step)))
    edges = np.linspace(x0, x1, n + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    hb = np.full(n, np.nan)
    if x.size:
        b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)
        inside = (x >= x0) & (x <= x1)
        for i in np.unique(b[inside]):
            hb[i] = float(np.median(h[inside & (b == i)]))
    return xc, hb


# --------------------------------------------------------------------------------------
# 단면 1 개
# --------------------------------------------------------------------------------------
def extract_fill_features(x_mm, z_mm, *, bevel_side_hint: Optional[str] = None, bevel_angle_hint_deg: Optional[float] = None,
                          inlier_mm: float = 0.25, merge_gap_mm: float = 1.0, min_plate_len_mm: float = 5.0,
                          edge_tol_mm: float = 3.0, ransac_iters: int = 200, min_points: int = 20,
                          cap_min_mm: float = 0.3, surface_max_slope_deg: float = 40.0, surface_thr_mm: float = 0.3,
                          reflection_mm: float = 0.6, wall_min_points: int = 4, wall_min_span_mm: float = 0.8,
                          vertical_max_lean_deg: float = 12.0, slope_k: int = 4, slope_r_mm: float = 0.6,
                          bin_mm: float = 0.1, toe_local_mm: float = 1.5, wall_min_slope_deg: float = 30.0,
                          droop_mm: float = 1.0, seed: int = 0) -> FillProfileFeatures:
    """프로파일 1 개 (센서 프레임 x, z [mm]) → FillProfileFeatures. 모듈 docstring 의 절차.
    bevel_side_hint: 'left'|'right' (센서 x 기준 — 빈 그루브 스캔의 CrossSectionFeatures.bevel_side), bevel_angle_hint_deg: 그 베벨각."""
    f = FillProfileFeatures()
    x = np.asarray(x_mm, dtype=float).reshape(-1)
    z = np.asarray(z_mm, dtype=float).reshape(-1)
    with np.errstate(invalid="ignore"):
        v = np.isfinite(x) & np.isfinite(z) & (z > 0.0)
    f.n_valid = int(v.sum())
    if f.n_valid < min_points:
        f.flags.append("insufficient_points")
        return f
    order = np.argsort(x[v], kind="stable")
    x = x[v][order]
    h = -z[v][order]
    dxs = np.diff(x)
    dxs = dxs[dxs > 0]
    dx = float(np.median(dxs)) if dxs.size else 0.13
    rng = np.random.default_rng(seed)
    grow_kw = dict(thr=inlier_mm, merge_gap=merge_gap_mm, min_len=min_plate_len_mm, edge_tol=max(edge_tol_mm, 3.0 * dx),
                   iters=ransac_iters, rng=rng, min_pts=max(3, min_points // 4))
    mL = _grow_plate(x, h, True, **grow_kw)
    mR = _grow_plate(x, h, False, **grow_kw)
    if mL is None or mR is None:
        f.flags.append("insufficient_points")
        f.notes.append("plate top not found on both sides")
        return f
    if np.array_equal(mL, mR):
        f.state = "no_groove"
        f.flags.append("no_groove")
        return f
    lnL0, _ = _plate_info(x, h, mL)
    lnR0, _ = _plate_info(x, h, mR)
    phi = math.radians(0.5 * (lnL0.angle_deg + lnR0.angle_deg))
    f.detilt_phi_deg = math.degrees(phi)
    f.tilt_deg = -math.degrees(phi)
    xp, hp = _rot(x, h, phi)
    xL_end, xR_start = float(xp[mL].max()), float(xp[mR].min())
    if xR_start - xL_end <= 3 * dx:
        f.state = "no_groove"
        f.flags.append("no_groove")
        return f

    def _local_line(m, toward_groove_x, left: bool):
        loc = m & ((xp >= toward_groove_x - 10.0) if left else (xp <= toward_groove_x + 10.0))
        return fit_line_pca(xp[loc], hp[loc]) if int(loc.sum()) >= 3 and float(np.ptp(xp[loc])) > 1.0 else fit_line_pca(xp[m], hp[m])

    pl_L = _local_line(mL, xL_end, True)
    pl_R = _local_line(mR, xR_start, False)
    span = xR_start - xL_end

    def top(xq):
        xq = np.asarray(xq, dtype=float)
        w = np.clip((xq - xL_end) / span, 0.0, 1.0)
        return (1.0 - w) * pl_L.h_at(xq) + w * pl_R.h_at(xq)

    xc0 = 0.5 * (xL_end + xR_start)
    f.hi_lo_mm = float(pl_R.h_at(xc0) - pl_L.h_at(xc0))
    inter = np.nonzero((~mL) & (~mR) & (xp > xL_end) & (xp < xR_start))[0]
    if inter.size < 5:
        f.flags.append("insufficient_points")
        f.notes.append(f"only {inter.size} points between the plates")
        return f
    d = top(xp[inter]) - hp[inter]
    n_above, n_below = int(np.sum(d < -cap_min_mm)), int(np.sum(d > cap_min_mm))
    if n_above > n_below and n_above >= 5:
        return _cap(f, xp, hp, inter, d, xL_end, xR_start, top, pl_L, pl_R, phi, bin_mm, reflection_mm)
    if n_below < 5:
        f.state = "no_groove"
        f.flags.append("flush")
        f.notes.append("interior within the plate band: flush with the plate tops")
        return f
    f.state = "groove"

    # ---- 표면 (완만한 점의 RANSAC 2차식) ----
    slope = _local_slope_deg(xp, hp, inter, int(slope_k), float(slope_r_mm))
    cand = (slope < surface_max_slope_deg) & (d > -cap_min_mm)
    coef, inl_c = ransac_quadratic(xp[inter][cand], hp[inter][cand], surface_thr_mm, ransac_iters, rng, min_pts=5)
    if coef is None:
        f.flags.append("no_surface")
        f.notes.append("no fill surface (too few gentle points)")
        return f
    r_all = hp[inter] - _quad(coef, xp[inter])
    surf = np.zeros(inter.size, dtype=bool)
    surf[np.flatnonzero(cand)[inl_c]] = True
    refl = r_all < -reflection_mm
    f.n_surface, f.n_reflections = int(surf.sum()), int(refl.sum())
    f.surface_rms_mm = float(np.sqrt(np.mean(r_all[surf] ** 2)))
    if f.n_reflections:
        f.flags.append("reflections")

    # ---- 벽 ----
    # 벽 후보: 가파른 점 또는 판 상면 바로 아래 (droop_mm 안 — 판 끝 둥근 처짐). 벽 높이 중간에 떠 있는 완만한 점 무리 (상호 반사
    # 유령 — 실기 at07 수직벽 쪽 판 아래 ~5 mm) 는 벽이 아니다 (stray)
    steep = np.nan_to_num(slope, nan=0.0) >= wall_min_slope_deg
    wall_cand = (~surf) & (~refl) & (r_all > -surface_thr_mm) & (d > -cap_min_mm) & (steep | (d < droop_mm))
    walls = {}
    for side, x_edge, plus, pl in (("left", xL_end, True, pl_L), ("right", xR_start, False, pl_R)):
        mw = _attached(xp[inter], wall_cand & ((xp[inter] < xc0) if plus else (xp[inter] >= xc0)), x_edge, plus, 0.6)
        info = {"n": int(mw.sum())}
        ln = None
        if int(mw.sum()) >= wall_min_points and float(np.ptp(hp[inter][mw])) >= wall_min_span_mm:
            ln, _inl = ransac_line(xp[inter][mw], hp[inter][mw], 0.2, ransac_iters, rng, min_pts=3)
        xs_side = xp[inter][surf & ((xp[inter] < xc0) if plus else (xp[inter] >= xc0))]
        if ln is not None:
            info["kind"] = "vertical" if ln.angle_from_vertical_deg < vertical_max_lean_deg else "bevel"
            info["angle_from_vertical_deg"] = float(ln.angle_from_vertical_deg)
            if info["kind"] == "vertical":
                # 벽 면 x: 안쪽 분위 (판 끝 둥근 처짐 점은 벽보다 바깥이므로 중앙값이 바깥으로 끌린다)
                xw = float(np.percentile(xp[inter][mw], 80 if plus else 20))
                ln = Line2D([xw, float(np.median(hp[inter][mw]))], [0.0, 1.0], int(mw.sum()))
        else:
            # 직선을 맞출 만큼 벽 점이 없다 (벽이 안 보임 / 판 상면 가까이 차서 짧음). 판 끝에 이어진 점 (모서리 둥근 처짐 등) 의 가장
            # 안쪽 + 반 점 간격 = 벽 (2026-10-03 실기 at07: 수직벽 쪽 판 끝이 0.3–0.6 mm 처져 판 영역이 벽보다 일찍 끝난다) — 단 충전
            # 표면 점보다 바깥 (벽이 표면 안쪽일 수는 없다)
            mdr = mw & (d < droop_mm)
            x_run = (float(xp[inter][mdr].max()) if plus else float(xp[inter][mdr].min())) if mdr.any() else x_edge
            xe = x_run + (0.5 * dx if plus else -0.5 * dx)
            if xs_side.size:
                x_floor = float(xs_side.min()) if plus else float(xs_side.max())       # 벽 쪽 첫 표면점
                # 그늘: 모서리를 스친 광선이 그루브 안쪽으로 기울면 벽 밑동 옆 바닥이 (깊이 × 기울기) 만큼 가려진다
                h_floor = float(np.median(hp[inter][surf][np.abs(xp[inter][surf] - x_floor) <= 0.5]))
                k_ray = _ray_dx_per_depth(x_run, float(pl.h_at(x_run)), phi)
                shadow = max(0.0, (k_ray if plus else -k_ray) * (float(pl.h_at(x_run)) - h_floor))
                info["shadow_mm"] = shadow
                if shadow > 0.1:
                    xe = x_floor - shadow if plus else x_floor + shadow
                elif abs(x_floor - x_run) <= 1.0:                                     # 그늘 없음: 처짐 끝과 첫 표면점 사이 가운데
                    xe = 0.5 * (x_run + x_floor)
                # 벽은 처짐 끝 (판 쪽 마지막 점) 과 첫 표면점 사이에 있다 (반 점 간격 제한은 두 점 간격이 점 간격보다 좁을 때 서로 부딪힌다)
                xe = min(max(xe, x_run), x_floor) if plus else max(min(xe, x_run), x_floor)
            if bevel_side_hint == side and bevel_angle_hint_deg is not None:
                b = math.radians(float(bevel_angle_hint_deg))
                p0 = ([float(np.mean(xp[inter][mw])), float(np.mean(hp[inter][mw]))] if int(mw.sum()) >= 2
                      else [x_edge + (0.5 * dx if plus else -0.5 * dx), float(pl.h_at(x_edge))])
                ln = Line2D(p0, [math.sin(b) if plus else -math.sin(b), -math.cos(b)])
                info.update(kind="hint", angle_from_vertical_deg=float(bevel_angle_hint_deg))
            else:
                ln = Line2D([xe, float(pl.h_at(xe))], [0.0, 1.0])
                info.update(kind="edge", angle_from_vertical_deg=0.0)
        walls[side] = (ln, info)
    f.n_stray = int(np.sum(wall_cand)) - sum(w[1]["n"] for w in walls.values())

    # ---- 모서리·토우 ----
    pts = {}
    for side, pl in (("left", pl_L), ("right", pl_R)):
        ln, info = walls[side]
        if abs(ln.d[0]) < 1e-9:                                         # 수직
            cx = float(ln.p[0])
            corner = (cx, float(pl.h_at(cx)))
        else:
            c_ = ln.intersect(pl)
            corner = (float(c_[0]), float(c_[1])) if c_ is not None else (float(ln.p[0]), float(pl.h_at(ln.p[0])))
        hits = [p for p in _line_quad_hits(ln, coef) if p[1] <= corner[1] + 0.3 and abs(p[0] - corner[0]) <= 0.6 * span + 1.0]
        if not hits:
            f.flags.append(f"toe_{side}_unresolved")
            return f
        toe = max(hits, key=lambda p: p[1])                            # 모서리에서 내려가며 첫 교점
        # 국소 정련: 토우 안쪽 toe_local_mm 의 표면점만으로 2차식 (적으면 직선) → 벽과 다시 교점 (전체 2차식은 비대칭 비드에서 토우를 놓친다)
        xs_, hs_ = xp[inter][surf], hp[inter][surf]
        near = ((xs_ >= toe[0]) & (xs_ <= toe[0] + toe_local_mm)) if side == "left" else \
            ((xs_ <= toe[0]) & (xs_ >= toe[0] - toe_local_mm))
        if int(near.sum()) >= 6 and float(np.ptp(xs_[near])) >= 0.6:
            deg = 2 if (int(near.sum()) >= 10 and float(np.ptp(xs_[near])) >= 1.0) else 1
            pc = np.polyfit(xs_[near], hs_[near], deg)[::-1]
            cl = np.zeros(3)
            cl[:pc.size] = pc
            h2 = [p for p in _line_quad_hits(ln, cl) if abs(p[0] - toe[0]) <= 0.5 and p[1] <= corner[1] + 0.3]
            if h2:
                toe = min(h2, key=lambda p: abs(p[0] - toe[0]))
                info["toe_local_fit"] = deg
        info["visible_mm"] = float(corner[1] - toe[1])
        pts[side] = (corner, toe)
    (cL, tL), (cR, tR) = pts["left"], pts["right"]
    if not (tR[0] - tL[0] > 0.2 and cR[0] > cL[0]):
        f.flags.append("toes_crossed")
        return f
    f.wall_left, f.wall_right = walls["left"][1], walls["right"][1]
    f.corner_left, f.corner_right = [cL[0], cL[1]], [cR[0], cR[1]]
    f.toe_left, f.toe_right = [tL[0], tL[1]], [tR[0], tR[1]]
    f.opening_mm = float(cR[0] - cL[0])
    f.toe_width_mm = float(tR[0] - tL[0])
    f.toe_tilt_mm = float(tR[1] - tL[1])
    f.toe_depth_left_mm = float(top(tL[0]) - tL[1])
    f.toe_depth_right_mm = float(top(tR[0]) - tR[1])

    # ---- 높이 프로파일 H(x') 와 적분 ----
    xs_s, hs_s = xp[inter][surf], hp[inter][surf]
    xc_b, hb = _binned(xs_s, hs_s, tL[0], tR[0], bin_mm)
    f.surface_coverage = float(np.mean(np.isfinite(hb)))
    hb = np.where(np.isfinite(hb), hb, _quad(coef, xc_b))
    # 적분 경로: 모서리 → (베벨면) → 토우 → 표면 구간 가운데들 → 토우 → (벽) → 모서리. 토우·모서리 점을 늘 넣는다 (수직벽은 같은 x 의
    # 두 점 — 넓이 0; 빠뜨리면 마지막 구간 가운데에서 끝나 bin/2 × 깊이 만큼 잃는다)
    lnL, lnR = walls["left"][0], walls["right"][0]

    def _wall_pts(ln, xa, xb):
        if xb - xa <= 1e-6:
            return np.zeros(0), np.zeros(0)
        xw = np.linspace(xa, xb, max(2, int(math.ceil((xb - xa) / bin_mm)) + 1))[1:-1]
        return xw, ln.h_at(xw)

    xwL, hwL = _wall_pts(lnL, cL[0], tL[0])
    xwR, hwR = _wall_pts(lnR, tR[0], cR[0])
    X = np.concatenate([[cL[0]], xwL, [tL[0]], xc_b, [tR[0]], xwR, [cR[0]]])
    H = np.concatenate([[cL[1]], hwL, [tL[1]], hb, [tR[1]], hwR, [cR[1]]])
    dep = top(X) - H
    f.remaining_area_mm2 = float(np.trapz(np.clip(dep, 0.0, None), X))
    f.reinforcement_area_mm2 = float(np.trapz(np.clip(-dep, 0.0, None), X))
    ds = top(xc_b) - hb
    xm = 0.5 * (tL[0] + tR[0])
    f.depth_center_mm = float(np.interp(xm, xc_b, ds))
    f.depth_min_mm, f.depth_max_mm = float(np.min(ds)), float(np.max(ds))
    chord = tL[1] + (xc_b - tL[0]) * (tR[1] - tL[1]) / (tR[0] - tL[0])
    dev = hb - chord
    k = int(np.argmax(np.abs(dev)))
    f.crown_mm = float(dev[k])
    f.crown_peak_frac = float((xc_b[k] - tL[0]) / (tR[0] - tL[0]))
    f.toe_left_xz, f.toe_right_xz = _xz(tL[0], tL[1], phi), _xz(tR[0], tR[1], phi)
    f.centre_xz = _xz(xm, float(np.interp(xm, xc_b, hb)), phi)
    kl = int(np.argmax(ds))
    f.low_xz = _xz(float(xc_b[kl]), float(hb[kl]), phi)
    f.ok = True
    return f


def _cap(f: FillProfileFeatures, xp, hp, inter, d, xL_end, xR_start, top, pl_L, pl_R, phi, bin_mm, reflection_mm):
    """캡 상태: 토우 = 판 영역 끝, 구간 중앙값 프로파일로 보강·미충전."""
    f.state = "cap"
    keep = d < reflection_mm * 2.5                                    # 판 상면선보다 한참 아래 = 반사 (캡 아래 그루브는 안 보인다)
    coef, _inl = ransac_quadratic(xp[inter][keep], hp[inter][keep], 0.3, 200, np.random.default_rng(0), min_pts=5)
    if coef is not None:                                              # 캡 면 (2차식) 에서 reflection_mm 넘게 벗어난 점 = 반사·유령
        r = hp[inter] - _quad(coef, xp[inter])
        keep &= np.abs(r) <= reflection_mm
        f.surface_rms_mm = float(np.sqrt(np.mean(r[keep] ** 2))) if keep.any() else float("nan")
    f.n_reflections = int(np.sum(~keep))
    if f.n_reflections:
        f.flags.append("reflections")
    xc_b, hb = _binned(xp[inter][keep], hp[inter][keep], xL_end, xR_start, bin_mm)
    good = np.isfinite(hb)
    f.surface_coverage = float(np.mean(good))
    if int(good.sum()) < 5:
        f.flags.append("insufficient_points")
        return f
    hb = np.interp(xc_b, xc_b[good], hb[good])
    X = np.concatenate([[xL_end], xc_b, [xR_start]])
    H = np.concatenate([[float(pl_L.h_at(xL_end))], hb, [float(pl_R.h_at(xR_start))]])
    dep = top(X) - H
    f.reinforcement_area_mm2 = float(np.trapz(np.clip(-dep, 0.0, None), X))
    f.remaining_area_mm2 = float(np.trapz(np.clip(dep, 0.0, None), X))
    up = -(top(xc_b) - hb)
    k = int(np.argmax(up))
    f.cap_height_mm = float(up[k])
    f.cap_width_mm = float(xR_start - xL_end)
    f.crown_peak_frac = float((xc_b[k] - xL_end) / (xR_start - xL_end))
    f.toe_left = [float(xL_end), float(pl_L.h_at(xL_end))]
    f.toe_right = [float(xR_start), float(pl_R.h_at(xR_start))]
    f.toe_width_mm = f.cap_width_mm
    f.toe_tilt_mm = float(f.toe_right[1] - f.toe_left[1])
    f.toe_left_xz, f.toe_right_xz = _xz(*f.toe_left, phi), _xz(*f.toe_right, phi)
    xm = 0.5 * (xL_end + xR_start)
    f.centre_xz = _xz(xm, float(np.interp(xm, xc_b, hb)), phi)
    f.ok = True
    return f


# --------------------------------------------------------------------------------------
# 패스 전체
# --------------------------------------------------------------------------------------
FILL_KEYS = ("remaining_area_mm2", "reinforcement_area_mm2", "depth_center_mm", "depth_min_mm", "depth_max_mm", "crown_mm",
             "crown_peak_frac", "toe_width_mm", "toe_tilt_mm", "toe_depth_left_mm", "toe_depth_right_mm", "opening_mm",
             "cap_height_mm", "cap_width_mm", "surface_coverage", "hi_lo_mm", "tilt_deg")


@dataclass
class FillScanFeatures:
    n_profiles: int = 0
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    s_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ok: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    state: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    values: dict = field(default_factory=dict)                    # FILL_KEYS → (M,)
    rejected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))   # 심 방향 일관성으로 뺀 이유 ('' = 아님)
    obliquity_area: np.ndarray = field(default_factory=lambda: np.zeros(0))           # 단면적 보정 계수 |Y_opt·u| (seam_u 없으면 1)
    wall_shift_mm: dict = field(default_factory=dict)                                  # 수직벽 평활화 이동 [mm] {'left','right'} → (M,)
    toe_left_base_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    toe_right_base_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    centre_base_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    low_base_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    features: Optional[list] = None

    def series(self, key: str, s=None, states=None) -> tuple:
        """ok (∧ state ∈ states) 프로파일의 (좌표, 값) — 좌표 오름차순 (기본 s_m)."""
        v = np.asarray(self.values[key], dtype=float)
        s_ = np.asarray(self.s_m if s is None else s, dtype=float)
        m = np.asarray(self.ok, dtype=bool) & np.isfinite(v) & np.isfinite(s_)
        if states is not None:
            m &= np.isin(self.state, list(states))
        o = np.argsort(s_[m], kind="stable")
        return s_[m][o], v[m][o]

    def medians(self) -> dict:
        out = {"n_profiles": int(self.n_profiles), "n_ok": int(np.sum(self.ok)),
               "states": {str(k): int(np.sum(self.state == k)) for k in np.unique(self.state)}}
        for k in FILL_KEYS:
            _, v = self.series(k)
            out[k] = float(np.median(v)) if v.size else float("nan")
        return out

    def along_seam_bins(self, bin_m: float = 0.020, keys=("remaining_area_mm2", "reinforcement_area_mm2", "depth_center_mm",
                                                           "crown_mm", "toe_width_mm", "cap_height_mm"), s=None,
                        origin: Optional[float] = None, min_n: int = 5) -> list:
        """심 방향 구간별 중앙값 [{s_from_m, s_to_m, n, state (최빈), <key>: …}] (프로파일 < min_n 인 구간은 뺀다)."""
        s_all = np.asarray(self.s_m if s is None else s, dtype=float)
        m_ok = np.asarray(self.ok, dtype=bool) & np.isfinite(s_all)
        if not m_ok.any():
            return []
        lo0 = float(s_all[m_ok].min())
        if origin is not None and np.isfinite(origin):
            lo0 = float(origin) + math.floor((lo0 - float(origin)) / bin_m + 1e-9) * bin_m
        edges = np.arange(lo0, float(s_all[m_ok].max()) + bin_m, bin_m)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            mb = m_ok & (s_all >= a) & (s_all < b)
            if int(mb.sum()) < min_n:
                continue
            st, cnt = np.unique(self.state[mb], return_counts=True)
            row = {"s_from_m": float(a), "s_to_m": float(b), "n": int(mb.sum()), "state": str(st[int(np.argmax(cnt))])}
            for k in keys:
                vk = np.asarray(self.values[k], dtype=float)[mb]
                vk = vk[np.isfinite(vk)]
                row[k] = float(np.median(vk)) if vk.size >= min_n else float("nan")
            out.append(row)
        return out


# 심 방향 일관성 (반사가 판 끝을 가린 프로파일 등): 상태별 키 → 이동 중앙값에서 허용 편차
CONSISTENCY_TOL = {"groove": {"opening_mm": 0.4, "toe_width_mm": 0.5, "remaining_area_mm2": 2.0},
                   "cap": {"cap_width_mm": 0.6, "cap_height_mm": 0.3, "reinforcement_area_mm2": 2.0}}


def _running_median(s: np.ndarray, v: np.ndarray, window: float) -> np.ndarray:
    return _running_median_mad(s, v, window)[0]


def _running_median_mad(s: np.ndarray, v: np.ndarray, window: float) -> tuple:
    """(s 오름차순) 창 window 이동 중앙값과 강건 σ (1.4826·MAD)."""
    lo = np.searchsorted(s, s - 0.5 * window, side="left")
    hi = np.searchsorted(s, s + 0.5 * window, side="right")
    med = np.array([np.median(v[a:b]) for a, b in zip(lo, hi)])
    sig = np.array([1.4826 * np.median(np.abs(v[a:b] - m)) for (a, b), m in zip(zip(lo, hi), med)])
    return med, sig


AREA_KEYS = ("remaining_area_mm2", "reinforcement_area_mm2")
DEPTH_KEYS = ("depth_center_mm", "depth_min_mm", "depth_max_mm", "crown_mm", "toe_tilt_mm", "toe_depth_left_mm", "toe_depth_right_mm",
              "cap_height_mm", "hi_lo_mm")
WIDTH_KEYS = ("toe_width_mm", "opening_mm", "cap_width_mm")


def extract_fill_features_along_scan(result, *, stride: int = 1, keep_features: bool = False, consistency_window_m: float = 0.010,
                                     consistency_tol: Optional[dict] = None, consistency_k_sigma: float = 4.0, s_along=None,
                                     seam_u=None, plate_n=None, wall_smooth_window_m: float = 0.020, **kw) -> FillScanFeatures:
    """ScanResult (x_mm, z_mm, valid, T_base_opt, s_m) → 프로파일별 충전 특징 + 토우·표면 점의 base 3D 좌표 [m].

    심 방향 일관성: 같은 상태의 ok 프로파일마다 consistency_tol 의 키가 consistency_window_m 이동 중앙값에서 허용치보다 벗어나면 ok
    False (rejected 에 이유) — 상부 개구·토우 폭·단면적은 심 방향으로 천천히 변하므로, 한 프로파일만 튀면 그 프로파일의 측정 문제다
    (예: 반사 띠가 판 끝을 덮어 모서리가 밀림 — 프로파일 하나만으로는 실제 모서리와 구별할 수 없다). 좌표는 s_along (프로파일별 심
    방향 위치 [m]; 기본 s_m).

    비스듬한 단면 보정 (seam_u = 심 방향 단위벡터 [base] 를 주면): 레이저 평면 (법선 Y_opt) 이 심에 수직이 아니면 (피치·요) 단면이
    비스듬히 잘려 넓이가 1/|Y·u| 배로 커진다 (일정 단면 프리즘의 사잇단면) → 단면적에 |Y·u| 를 곱한다. plate_n 도 주면 깊이류에
    √(1 − (Y·n)²), 폭류에 √(1 − (Y·w)²) (w = n × u) 를 곱한다 (순수 피치 α: 넓이·깊이 cos α, 폭 1). 보정 계수는 obliquity_area.

    수직벽 평활화 (seam_u·plate_n 이 있고 wall_smooth_window_m > 0): 보이지 않는 수직벽 ('edge') 과 보이는 수직벽의 위치는 프로파일
    하나로는 점 간격·판 끝 처짐만큼 흔들린다 (실기 at07: 토우 폭 σ ≈ 0.28 mm → 단면적 σ ≈ 2.8 mm²). 벽은 심을 따라 곧은 면이므로 벽
    밑동 (토우) 의 base 횡위치 ((P·w)) 를 심 방향 이동 중앙값으로 바꾸고, 그 차 δ 만큼 폭 (토우 폭·개구) 과 단면적 (δ × 벽 높이) 을
    산술로 고친다 (다시 추출하지 않는다). 일관성 허용치 = max(consistency_tol, consistency_k_sigma × 창 안 강건 σ)."""
    N = int(result.x_mm.shape[0])
    idx = np.arange(0, N, max(1, int(stride)), dtype=np.int64)
    M = idx.size
    out = FillScanFeatures(n_profiles=M, indices=idx)
    vals = {k: np.full(M, np.nan) for k in FILL_KEYS}
    ok = np.zeros(M, dtype=bool)
    state = np.empty(M, dtype=object)
    P = {k: np.full((M, 3), np.nan) for k in ("toe_left", "toe_right", "centre", "low")}
    s_all = np.asarray(getattr(result, "s_m", np.full(N, np.nan)), dtype=float)
    valid = np.asarray(result.valid, dtype=bool) if getattr(result, "valid", None) is not None else None
    feats = [] if keep_features else None
    u_ = np.asarray(seam_u, dtype=float) / np.linalg.norm(seam_u) if seam_u is not None else None
    n_ = np.asarray(plate_n, dtype=float) / np.linalg.norm(plate_n) if plate_n is not None else None
    w_ = None
    if u_ is not None and n_ is not None:
        w_ = np.cross(n_, u_)
        w_ = w_ / np.linalg.norm(w_)
    k_area = np.ones(M)
    k_w = np.ones(M)
    xdotw = np.full(M, np.nan)
    wall_kind = {"left": np.empty(M, dtype=object), "right": np.empty(M, dtype=object)}
    wall_h = {"left": np.full(M, np.nan), "right": np.full(M, np.nan)}
    for j, i in enumerate(idx):
        xi = np.asarray(result.x_mm[i], dtype=float)
        zi = np.asarray(result.z_mm[i], dtype=float)
        if valid is not None:
            xi = np.where(valid[i], xi, np.nan)
            zi = np.where(valid[i], zi, np.nan)
        f = extract_fill_features(xi, zi, **kw)
        Yo = np.asarray(result.T_base_opt[i], dtype=float)[:3, 1]
        if u_ is not None:
            k_area[j] = abs(float(Yo @ u_))
            for k in AREA_KEYS:
                setattr(f, k, getattr(f, k) * k_area[j])
            if n_ is not None:
                kh, kw_ = math.sqrt(max(0.0, 1.0 - float(Yo @ n_) ** 2)), math.sqrt(max(0.0, 1.0 - float(Yo @ w_) ** 2))
                for k in DEPTH_KEYS:
                    setattr(f, k, getattr(f, k) * kh)
                for k in WIDTH_KEYS:
                    setattr(f, k, getattr(f, k) * kw_)
                k_w[j] = kw_
                xdotw[j] = float(np.asarray(result.T_base_opt[i], dtype=float)[:3, 0] @ w_)
        for side, info in (("left", f.wall_left), ("right", f.wall_right)):
            wall_kind[side][j] = info.get("kind")
            wall_h[side][j] = float(info.get("visible_mm", float("nan")))
        if feats is not None:
            feats.append(f)
        state[j] = f.state
        ok[j] = bool(f.ok)
        for k in FILL_KEYS:
            vals[k][j] = getattr(f, k)
        T = np.asarray(result.T_base_opt[i], dtype=float)
        for k in P:
            xz = getattr(f, f"{k}_xz")
            if f.ok and xz is not None:
                P[k][j] = (T @ np.array([xz[0] * 1e-3, 0.0, xz[1] * 1e-3, 1.0]))[:3]
    out.s_m = np.array([s_all[i] if i < s_all.size else np.nan for i in idx], dtype=float)
    wall_shift = {"left": np.zeros(M), "right": np.zeros(M)}
    if w_ is not None and wall_smooth_window_m > 0:
        for side, key in (("left", "toe_left"), ("right", "toe_right")):
            m = (ok & (state == "groove") & np.isin(wall_kind[side], ["edge", "vertical"]) & np.all(np.isfinite(P[key]), axis=1)
                 & np.isfinite(xdotw) & (np.abs(xdotw) > 0.5) & np.isfinite(wall_h[side]))
            if int(m.sum()) < 5:
                continue
            ii = np.flatnonzero(m)
            sa, lat = P[key][ii] @ u_, P[key][ii] @ w_
            o = np.argsort(sa, kind="stable")
            ii, sa, lat = ii[o], sa[o], lat[o]
            d_lat = _running_median(sa, lat, float(wall_smooth_window_m)) - lat          # [m]
            dx = d_lat * 1e3 / xdotw[ii]                                                    # 프로파일 x [mm]
            dwid = -dx if side == "left" else dx                                            # 왼 벽이 +x 로 가면 폭이 준다
            for key_w in ("toe_width_mm", "opening_mm"):
                vals[key_w][ii] += dwid * k_w[ii]
            vals["remaining_area_mm2"][ii] += dwid * wall_h[side][ii] * k_area[ii]
            P[key][ii] += np.outer(d_lat, w_)
            wall_shift[side][ii] = dx
    out.wall_shift_mm = wall_shift
    rejected = np.full(M, "", dtype=object)
    s_c = out.s_m if s_along is None else np.asarray(s_along, dtype=float)
    tol_all = CONSISTENCY_TOL if consistency_tol is None else consistency_tol
    for st, tols in tol_all.items():
        for key, tol in tols.items():
            m = ok & (state == st) & np.isfinite(vals[key]) & np.isfinite(s_c)
            if int(m.sum()) < 5:
                continue
            ii = np.flatnonzero(m)
            o = np.argsort(s_c[ii], kind="stable")
            ii = ii[o]
            med, sig = _running_median_mad(s_c[ii], vals[key][ii], float(consistency_window_m))
            bad = ii[np.abs(vals[key][ii] - med) > np.maximum(float(tol), float(consistency_k_sigma) * sig)]
            for b in bad:
                if ok[b]:
                    rejected[b] = f"{key} {vals[key][b]:.2f} vs running median {med[int(np.flatnonzero(ii == b)[0])]:.2f}"
            ok[bad] = False
    out.rejected = rejected
    out.obliquity_area = k_area
    out.ok, out.state, out.values, out.features = ok, state, vals, feats
    out.toe_left_base_m, out.toe_right_base_m, out.centre_base_m, out.low_base_m = P["toe_left"], P["toe_right"], P["centre"], P["low"]
    return out
