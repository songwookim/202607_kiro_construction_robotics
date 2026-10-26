"""심(seam) 단면 특징 추출 — **스캔 검증용** (용접 경로 생성은 범위 밖, 설계 §5.9 의 전단계).

입력: 프로파일 1개 (x_mm, z_mm; 센서 z = range, 멀수록 큼; 무효 NaN 또는 z=0) 또는 시간 중앙값 프로파일.
좌표: 내부에서는 (x, h = −z) 로 다룬다 (h 는 센서 쪽이 +). 판 상면 기울기 φ 를 추정해 회전(de-tilt)한 프레임 (x', h')
에서 그루브 치수를 계산하고, 코너·중심점은 다시 센서 프레임 (x, z) 로 돌려 보고한다.

절차
  1. 유효점 x 정렬 → RANSAC 직선 L1 (지배 판 상면). L1 인라이어를 x 연속 구간(run)으로 묶는다 (merge_gap 이하의 틈은 결합).
  2. 프로파일 양끝(x_min/x_max, edge_tol 이내)에 닿는 run = 왼/오른 판. 한쪽이 L1 에 없으면 그쪽 점만으로 RANSAC 재시도
     (hi-lo 가 큰 경우 두 판이 다른 직선).  두 run 이 같으면 그루브 없음.
  3. 두 판 각도의 평균으로 de-tilt. 각 판을 그루브 쪽으로 **연속성**으로 연장한다 (점마다 직전 plate_extend_k 점의 h' 중앙값에서
     plate_extend_tol 이내, 두 점 연속 벗어나면 멈춤): 끝에서 맞춘 직선이 판의 약한 굴곡·원거리 왜곡 때문에 그루브 직전 몇~20 mm
     에서 0.3 mm 를 벗어나 그 판 점이 그루브 내부로 섞이던 문제 (실기 2026-10-02, 495 mm). 판 레벨·코너는 연장된 판의 그루브 쪽
     plate_local_mm 구간 직선으로. 판 레벨(h')·hi-lo(= right − left, + 는 오른 판이 센서에 더 가까움).
  4. 두 run 사이 점 = 그루브 내부. 바닥 = 판보다 depth_min 이상 낮은 **가장 넓은 수평 대역** (폭 floor_band, x 연속 구간의
     점 수 최대) → 바닥 직선/레벨/폭. 최저 대역이 아니다: 실기(2026-10-02) 루트 모서리에는 폭 0.5–1.8 mm 의 깊은 반사 스파이크
     (바닥 아래 6–13 mm) 와 베벨 발치 아래 폭 ~2 mm 의 얕은 반사 (바닥 아래 0.5–1.6 mm) 가 생기는데, 실제 바닥(뒷댐재)은 루트갭
     전체 폭이라 가장 넓다. 바닥선보다 floor_band 이상 낮은 점 = 반사 (n_below_floor, 이후 모든 단계에서 제외).
     바닥 위 점을 바닥 중심 기준 좌/우로 나누되 **그쪽 판 끝에 이어진 점만** 쓴다 (판 끝에서 안쪽으로 merge_gap 이하 간격으로
     이어진 무리; 그루브 가운데 떠 있는 반사점은 n_stray_points 로 세고 제외). x' 퍼짐이 큰 쪽(> wall_spread_max) 이고 RANSAC
     직선이 수직에서 bevel_min_angle 이상 누운 쪽 = 베벨면 (판 법선 기준 각), 그 밖 = 직각 벽 (x' 중앙값; 요·피치로 몇 도 기울어
     보이는 벽 포함 — 실기 2026-10-02 라인 요 −8.8° 에서 벽이 ~7° 로 보여 'both' 로 오판하던 것). 양쪽 모두 베벨이면 'both'(V),
     모두 벽이면 'none'.
  5. 상부 코너: 베벨 쪽 = 베벨선∩판선, 직각 쪽 = 벽 x' (벽 점이 없으면 판 run 끝과 첫 내부점의 중간).
     루트 모서리: 베벨 쪽 = 베벨선∩바닥선, 직각 쪽 = 벽 x'.  상부 개구 = 코너 간 거리, 루트갭 = 루트 모서리 간 거리
     (root_gap_edges; 바닥 폭 root_gap_floor 는 가림 때문에 하한), 깊이 = 베벨 판 상면 − 바닥.
  6. 플래그: insufficient_points, no_groove, no_floor, occluded_floor, bevel_unresolved, hi_lo_ambiguous(|hi-lo| > 2 mm:
     판 단차(H1) / 라인이 그루브를 비스듬히 지남·롤(H2) / 낮은 면이 뒷댐재·다른 물체(H3) 구분 불가 → 보고서에 명시).
  4b/5b. 정렬 진단 (localize.fidelity_gate 용): wall_shear_k = 직각 벽(베벨 반대쪽; both/unknown 이면 수직에 가까운 쪽 → 'wall_ambiguous')
     인라이어 직선의 de-tilt 기울기 dx'/dh' — 평면 벽은 요·피치가 함께 있을 때만 기울고 k = tan(yaw)·sin(pitch) (분석 문서 §5 H2);
     정렬·중심 자세에서는 벽에 점이 없어 NaN(미평가). floor_visible = 바닥 직선 인라이어 ≥ floor_min_points 이고 x' 범위 ≥ 60 % × 기대 루트갭.

상세 형상(볼트·bevel 수치)은 관측이 본 추정이고 GrooveSpec 은 위치·스케일 비교(spec_deltas)에만 쓴다.

판정 문턱 (inlier·merge_gap·바닥 대역·벽/베벨 경계·판 최소 길이 …) 은 설정 상수가 아니다 (사용자 2026-10-07 4c): extractor_params 가 설계
(개구·갭·깊이·베벨각), 작업 허용치 (qc_targets 깊이·갭·hi-lo) 와 측정 열 간격에서 유도한다 — 인자로 주면 그 값.
"""
from __future__ import annotations

import atexit
import bisect
import math
import multiprocessing
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .measure import DESIGN_HALF, HALF_TURN_DEG, K_DEFAULT, MAD_TO_SIGMA, design_match, evidence_count, quant_sigma_mm, robust_sigma
from . import proc_pool as _proc_pool
from .proc_pool import OwnedPool
from .validity import dX_mm
from .specimen import DEFAULT_QC_TARGETS, GrooveSpec

__all__ = ["Line2D", "fit_line_pca", "ransac_line", "CrossSectionFeatures", "extract_cross_section_features", "points_above_plates",
           "matches_design", "notch_candidates", "extract_on_design_window",
           "SeamScanFeatures", "extract_seam_features_along_scan", "resolve_workers", "worker_pool", "shutdown_feature_pool", "HI_LO_AMBIGUOUS_MM",
           "extractor_params", "plate_min_len_mm", "detrended_noise", "PlateSeed", "cross_section_plate_seed", "plate_seeds_chunk",
           "FACE_KINDS", "FACE_COLS"]

HI_LO_AMBIGUOUS_MM = 2.0
# 단면의 면 (사용자 2026-10-08: 단면 상세 법선): 추출기가 이미 고른 면 점 (판 상면 그루브 쪽 국소 구간, 베벨면 RANSAC 인라이어, 직각 벽
# 인라이어, 바닥 = 뒷댐재 상면 인라이어) 에 센서 프레임 (x, z) 직선을 맞춘 것. left/right 는 센서 x 의 쪽 (심 좌표의 쪽은
# scan_core.cross_sections 가 base 3D 에서 정한다). 행 = FACE_KINDS, 열 = FACE_COLS (없는 면은 NaN 행).
FACE_KINDS = ("plate_left", "plate_right", "bevel_left", "bevel_right", "wall_left", "wall_right", "floor")
FACE_COLS = ("x_mm", "z_mm", "dx", "dz", "rms_mm", "n")      # 면 위 점 (직선 중심), 단위 방향 (센서 x, z), 직선 잔차 RMS, 점 수
_FACE_MIN_POINTS = 3          # 수치 최소: 직선 = 두 점 + 잔차를 잴 한 점 (원칙 7 — 판정 기준이 아니다; 면 선택은 추출기의 증거 규칙)
_FACE_TRIM_ITERS = 3          # 면 직선 k·σ 정련 반복 (원칙 7 — 수치 반복 횟수)
def _station_window_m() -> float:
    """용접 경로 단면 창 [m] (scan_core.weld_plan.WeldProcess 기본) — 측정 잡음의 이동 중앙값 길이: 용접 경로가 한 단면에서 쓰는 길이보다
    긴 심 방향 변화는 실제 형상 (시공 오차·휨), 그 안의 흔들림은 측정 잡음."""
    from .weld_plan import WeldProcess
    return float(WeldProcess().station_window_mm) * 1e-3


NOISE_WINDOW_M = _station_window_m()


# --------------------------------------------------------------------------------------
# 2-D 직선
# --------------------------------------------------------------------------------------
@dataclass
class Line2D:
    """p + t·d (d 단위, d[0] ≥ 0). (x, h) 프레임."""
    p: np.ndarray
    d: np.ndarray
    n: int = 0
    rms_mm: float = float("nan")
    x_min: float = float("nan")
    x_max: float = float("nan")

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=float).reshape(2)
        d = np.asarray(self.d, dtype=float).reshape(2)
        nrm = float(np.linalg.norm(d))
        d = d / nrm if nrm > 0 else np.array([1.0, 0.0])
        if d[0] < 0 or (d[0] == 0 and d[1] < 0):
            d = -d
        self.d = d

    @property
    def angle_deg(self) -> float:
        """x 축 기준 각 [deg] (h 가 x 와 함께 증가하면 +)."""
        return math.degrees(math.atan2(self.d[1], self.d[0]))

    @property
    def angle_from_vertical_deg(self) -> float:
        return math.degrees(math.atan2(abs(self.d[0]), abs(self.d[1])))

    def h_at(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if abs(self.d[0]) < 1e-12:
            return np.full(x.shape, np.nan)
        return self.p[1] + (x - self.p[0]) * self.d[1] / self.d[0]

    def dist(self, x, h) -> np.ndarray:
        """부호 있는 수직 거리 (법선 = (−d_h, d_x))."""
        nx, nh = -self.d[1], self.d[0]
        return (np.asarray(x, dtype=float) - self.p[0]) * nx + (np.asarray(h, dtype=float) - self.p[1]) * nh

    def intersect(self, other: "Line2D") -> Optional[np.ndarray]:
        A = np.column_stack([self.d, -other.d])
        det = float(np.linalg.det(A))
        if abs(det) < 1e-9:
            return None
        t = np.linalg.solve(A, other.p - self.p)
        return self.p + t[0] * self.d

    def rotated(self, phi: float) -> "Line2D":
        c, s = math.cos(phi), math.sin(phi)
        R = np.array([[c, s], [-s, c]])          # (x', h') = R·(x, h)  — 각 −phi 회전
        return Line2D(R @ self.p, R @ self.d, self.n, self.rms_mm)

    def to_dict(self) -> dict:
        return {"p": self.p.tolist(), "d": self.d.tolist(), "n": int(self.n), "rms_mm": float(self.rms_mm),
                "angle_deg": self.angle_deg, "x_min": float(self.x_min), "x_max": float(self.x_max)}


def fit_line_pca(x, h) -> Line2D:
    """직교 회귀(PCA) 직선."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.size < 2:
        c = np.array([float(x.mean()) if x.size else np.nan, float(h.mean()) if h.size else np.nan])
        return Line2D(c, [1.0, 0.0], int(x.size), 0.0, float(x.min()) if x.size else np.nan,
                      float(x.max()) if x.size else np.nan)
    c = np.array([float(x.mean()), float(h.mean())])
    # 2×2 공분산의 주축 (닫힌 식; SVD 첫 오른쪽 특이벡터와 같다 — 2026-10-02 속도)
    dx, dh = x - c[0], h - c[1]
    th = 0.5 * math.atan2(2.0 * float(dx @ dh), float(dx @ dx) - float(dh @ dh))
    ln = Line2D(c, [math.cos(th), math.sin(th)], int(x.size))
    r = ln.dist(x, h)
    ln.rms_mm = float(np.sqrt(np.mean(r ** 2)))
    ln.x_min, ln.x_max = float(x.min()), float(x.max())
    return ln


def ransac_line(x, h, thr_mm: float = 0.3, iters: int = 200, rng: Optional[np.random.Generator] = None,
                min_pts: int = 3) -> tuple:
    """RANSAC 직선 → (Line2D(인라이어 PCA 정련), inlier mask). 점이 부족하면 (None, zeros)."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    M = x.size
    if M < max(2, min_pts):
        return None, np.zeros(M, dtype=bool)
    rng = rng if rng is not None else np.random.default_rng(0)
    P = np.column_stack([x, h])
    # 후보 직선 iters 개를 한 번에 (2026-10-02: 표본마다 파이썬 루프 → 프로파일당 ~20 ms, 실기 패스 2900 프로파일에 ~75 s 의 대부분)
    n_it = int(iters)
    I = rng.integers(0, M, n_it)
    J = rng.integers(0, M - 1, n_it)
    J = J + (J >= I)                                         # 서로 다른 두 점
    D = P[J] - P[I]
    nrm = np.hypot(D[:, 0], D[:, 1])
    good = nrm >= 1e-9
    if not good.any():
        return None, np.zeros(M, dtype=bool)
    I, D, nrm = I[good], D[good], nrm[good]
    D = D / nrm[:, None]
    R = (x[None, :] - P[I, 0][:, None]) * (-D[:, 1][:, None]) + (h[None, :] - P[I, 1][:, None]) * D[:, 0][:, None]
    cnt = (np.abs(R) < thr_mm).sum(axis=1)
    k = int(np.argmax(cnt))                                  # 첫 최대 (순차 루프의 strict '>' 와 같은 규칙)
    best_cnt = int(cnt[k])
    if best_cnt < min_pts:
        return None, np.zeros(M, dtype=bool)
    ln = Line2D(P[I[k]].copy(), D[k].copy())
    inl = np.abs(ln.dist(x, h)) < thr_mm
    for _ in range(2):                                   # 정련 2회
        ln = fit_line_pca(x[inl], h[inl])
        inl_new = np.abs(ln.dist(x, h)) < thr_mm
        if inl_new.sum() < min_pts:
            break
        inl = inl_new
    ln = fit_line_pca(x[inl], h[inl])
    return ln, inl


# --------------------------------------------------------------------------------------
# 결과
# --------------------------------------------------------------------------------------
@dataclass
class CrossSectionFeatures:
    ok: bool = False
    n_valid: int = 0
    dx_mm: float = float("nan")                  # 열 간격 추정 (중앙값 |Δx|)
    flags: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    # 판
    tilt_deg: float = float("nan")               # 판 상면 기울기 atan(dz/dx) [deg], + = x 증가 시 z(range) 증가 (양 판 평균)
    plate_left: Optional[dict] = None            # {tilt_deg, rms_mm, n, x_min, x_max, level_mm}
    plate_right: Optional[dict] = None
    plate_angle_diff_deg: float = float("nan")
    hi_lo_mm: float = float("nan")               # right − left (de-tilt h', 센서 쪽 +) — 그루브 중심 x 에서
    # 그루브
    groove_found: bool = False
    left_edge_x_mm: float = float("nan")         # 상부 코너 x' (de-tilt)
    right_edge_x_mm: float = float("nan")
    top_opening_mm: float = float("nan")
    root_gap_mm: float = float("nan")            # = root_gap_edges (없으면 floor)
    root_gap_edges_mm: float = float("nan")
    root_gap_floor_mm: float = float("nan")
    bevel_side: str = "unknown"                  # left | right | both | none | unknown
    bevel_angle_deg: float = float("nan")        # 판 법선 기준
    bevel_angle_left_deg: float = float("nan")
    bevel_angle_right_deg: float = float("nan")
    depth_mm: float = float("nan")               # 베벨 판 상면 − 바닥 (none/both 는 평균)
    depth_left_mm: float = float("nan")
    depth_right_mm: float = float("nan")
    floor_level_mm: float = float("nan")         # de-tilt h'
    floor_width_mm: float = float("nan")
    n_floor_points: int = 0
    n_interior_points: int = 0
    n_below_floor: int = 0                       # 바닥선보다 floor_band 이상 낮은 내부점 (반사; 제외됨)
    n_stray_points: int = 0                      # 바닥 위이나 어느 판 끝에도 이어지지 않은 내부점 (반사; 제외됨)
    occluded_floor: bool = False
    insufficient_points: bool = False
    min_points: int = 0                          # 이 프로파일에 쓴 최소 유효 점 수 (extractor_params; 드웰의 '보이는 면 없음' 판정)
    # 센서 프레임 점 (x, z) [mm]
    top_corner_left_xz: Optional[list] = None
    top_corner_right_xz: Optional[list] = None
    root_edge_left_xz: Optional[list] = None
    root_edge_right_xz: Optional[list] = None
    groove_centre_x_mm: float = float("nan")     # 상부 개구 중심 (센서 x, z)
    groove_centre_z_mm: float = float("nan")
    root_centre_x_mm: float = float("nan")       # 루트갭 중심
    root_centre_z_mm: float = float("nan")
    detilt_phi_deg: float = float("nan")         # (x,h) → (x',h') 회전각
    # 정렬 진단 (fidelity_gate 용) — 직각 벽 전단 · 바닥(뒷댐재) 가시성
    wall_shear_k: float = float("nan")           # 직각(비베벨) 벽의 de-tilt 프레임 기울기 dx'/dh' (0 = 수직 = 레이저 평면이 심에 정렬;
                                                 # 평면 벽은 요·피치가 **동시에** 있을 때만 기운다: k = tan(yaw)·sin(pitch)). NaN = 미평가
    wall_points_n: int = 0                       # 벽 직선 인라이어 수 (0 = 벽 점 없음: 정렬·중심 상태에서는 수직 벽에 광선이 닿지 않음)
    wall_span_mm: float = float("nan")           # 벽 인라이어의 h' 범위
    wall_side: str = "none"                      # 평가한 벽: left | right | none
    floor_visible: Optional[bool] = None         # 바닥 인라이어 ≥ floor_min_points 이고 x' 범위 ≥ floor_visible_frac·기대 루트갭.
                                                 # None = 바닥 대역을 탐색하지 않음(그루브 없음/내부점 부족) 또는 기대 갭 미지
    floor_span_mm: float = float("nan")          # 바닥 직선 인라이어의 x' 범위
    floor_expected_mm: float = float("nan")      # 기대 루트갭 (spec.root_gap_mm, 없으면 root_gap_edges_mm)
    plate_slope_left: float = float("nan")       # 판 상면 dz/dx [mm/mm] (센서 프레임, = tan(tilt)) — 롤 진단용
    plate_slope_right: float = float("nan")
    # 판 인라이어 열 인덱스 (입력 x_mm/z_mm 배열 기준; localize 가 base 3D 점으로 옮긴다. to_dict 에는 넣지 않음)
    plate_inlier_idx_left: Optional[np.ndarray] = None
    plate_inlier_idx_right: Optional[np.ndarray] = None
    # 면 직선 (센서 프레임; FACE_KINDS × FACE_COLS, 없는 면 NaN) — 단면 법선 (scan_core.cross_sections). to_dict 에는 넣지 않음
    faces: Optional[np.ndarray] = None
    # 공칭 비교
    spec_name: Optional[str] = None
    spec_deltas: Optional[dict] = None
    qc: Optional[dict] = None

    hi_lo_warn_mm: float = HI_LO_AMBIGUOUS_MM    # 이 넘는 |hi-lo| = 장면 해석 불확실 (작업 qc_targets hi_lo_mm.warn_above)

    @property
    def hi_lo_ambiguous(self) -> bool:
        return bool(np.isfinite(self.hi_lo_mm) and abs(self.hi_lo_mm) > self.hi_lo_warn_mm)

    def plate_inlier_idx(self) -> np.ndarray:
        """양 판 인라이어 열 인덱스 합집합 (정렬, 중복 제거; 없으면 빈 배열)."""
        parts = [np.asarray(a, dtype=np.int64).reshape(-1) for a in (self.plate_inlier_idx_left, self.plate_inlier_idx_right)
                 if a is not None]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("plate_inlier_idx_left", None)
        d.pop("plate_inlier_idx_right", None)
        d.pop("faces", None)
        d["hi_lo_ambiguous"] = self.hi_lo_ambiguous
        return _plain(d)

    def report(self) -> str:
        L = []
        L.append(f"cross-section: ok={self.ok} n_valid={self.n_valid} dx~{self.dx_mm:.3f} mm flags={self.flags}")
        for name, p in (("left", self.plate_left), ("right", self.plate_right)):
            if p:
                L.append(f"  plate {name:<5s}: tilt {p['tilt_deg']:+.2f} deg rms {p['rms_mm']:.3f} mm n={p['n']} "
                         f"x in [{p['x_min']:.1f}, {p['x_max']:.1f}] mm")
            else:
                L.append(f"  plate {name:<5s}: not found")
        L.append(f"  tilt(avg) {self.tilt_deg:+.2f} deg, plate angle diff {self.plate_angle_diff_deg:.2f} deg, "
                 f"hi-lo (right-left) {self.hi_lo_mm:+.2f} mm")
        if self.hi_lo_ambiguous:
            L.append(f"  AMBIGUOUS: |hi-lo| {abs(self.hi_lo_mm):.1f} mm > {self.hi_lo_warn_mm:.0f} mm - cannot distinguish "
                     "(H1) misaligned plate tops, (H2) laser line crossing the groove obliquely / sensor roll, "
                     "(H3) the lower surface being the backing strip or another object; re-scan with a perpendicular, "
                     "centred pose before trusting gap/bevel values")
        if self.groove_found:
            L.append(f"  groove: edges x'=[{self.left_edge_x_mm:.2f}, {self.right_edge_x_mm:.2f}] mm top opening {self.top_opening_mm:.2f} mm "
                     f"root gap {self.root_gap_mm:.2f} mm (edges {self.root_gap_edges_mm:.2f} / floor {self.root_gap_floor_mm:.2f}) "
                     f"depth {self.depth_mm:.2f} mm (L {self.depth_left_mm:.2f} / R {self.depth_right_mm:.2f}) "
                     f"bevel {self.bevel_side} {self.bevel_angle_deg:.1f} deg floor pts {self.n_floor_points} occluded={self.occluded_floor}")
            L.append(f"  centre (sensor x,z): top ({self.groove_centre_x_mm:.2f}, {self.groove_centre_z_mm:.2f}) "
                     f"root ({self.root_centre_x_mm:.2f}, {self.root_centre_z_mm:.2f}) mm")
            k_txt = f"{self.wall_shear_k:+.4f}" if np.isfinite(self.wall_shear_k) else "n/a"
            fv_txt = "n/a" if self.floor_visible is None else str(bool(self.floor_visible))
            L.append(f"  align: wall({self.wall_side}) shear k={k_txt} n={self.wall_points_n} span {self.wall_span_mm:.1f} mm"
                     f"{' (wall-ambiguous)' if 'wall_ambiguous' in self.flags else ''}; floor visible={fv_txt} "
                     f"span {self.floor_span_mm:.2f} / expected {self.floor_expected_mm:.2f} mm")
        else:
            L.append("  groove: not found")
        if self.spec_deltas is not None:
            L.append(f"  vs spec {self.spec_name}: " + ", ".join(f"{k} {v:+.2f}" for k, v in self.spec_deltas.items()
                                                                if isinstance(v, float) and np.isfinite(v)))
            if self.qc:
                bad = [k for k, v in self.qc.items() if v.get("pass") is False]
                L.append("  QC: " + ("PASS" if not bad else "FAIL " + ", ".join(bad)) +
                         ("" if not self.hi_lo_ambiguous else " (hi-lo ambiguity: verdict provisional)"))
        for n_ in self.notes:
            L.append(f"  note: {n_}")
        return "\n".join(L)


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


@dataclass
class PlateSeed:
    """판 성장 (extract_cross_section_features 1~2 단계: 양 끝에서 판 상면 영역 성장) 의 결과 — 프로파일 점 (x, z) 과 설계·작업 허용치·
    추출 인자만의 함수다 (로봇 자세·시각 모형·depth_scale 와 무관: depth_scale 은 3 단계 de-tilt 에서 처음 쓴다). 패스 중 녹화되는 대로
    미리 계산해 (scan_l0.post_scan 스트림) 이동이 끝난 뒤에는 나머지 단계만 한다 (2026-10-08: 특징 추출 시간의 ~70 % 가 판 성장).
    n = 유효 점 수 (확인용), mL/mR = 정렬된 유효점의 판 인라이어 마스크 (없으면 None), rng_state = 두 판 성장 뒤 RANSAC 난수 상태
    (다음 단계의 RANSAC 이 같은 난수열을 이어 쓴다 — 결과가 한 번에 계산한 것과 같다)."""
    n: int
    mL: Optional[np.ndarray]
    mR: Optional[np.ndarray]
    rng_state: dict


# --------------------------------------------------------------------------------------
# 도우미
# --------------------------------------------------------------------------------------
def _runs(mask: np.ndarray, x: np.ndarray, merge_gap_mm: float) -> list:
    """mask True 인 인덱스를 (i0, i1) 포함 구간으로. 인접 run 사이 x 틈 < merge_gap 이면 결합."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    # 벡터화 (2026-10-02 속도): 이웃 인덱스가 연속이거나 x 틈 < merge_gap 이면 잇는다 — 이전 순차 루프와 같은 규칙
    cont = (np.diff(idx) == 1) | ((x[idx[1:]] - x[idx[:-1]]) < merge_gap_mm)
    br = np.flatnonzero(~cont)
    starts = np.concatenate(([0], br + 1))
    ends = np.concatenate((br, [idx.size - 1]))
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)]


def _run_mask(n: int, run, inl: np.ndarray) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    a, b = run
    m[a:b + 1] = inl[a:b + 1]
    return m


def _grow_plate(x, h, from_left: bool, *, thr, merge_gap, min_len, edge_tol, iters, rng, min_pts,
                init_frac: float = 0.10, init_min_mm: float = 10.0, max_iter: int = 20,
                init_scales: tuple = (1.0, 2.0, 3.0)) -> Optional[np.ndarray]:
    """프로파일 한쪽 끝에서 판 상면을 영역 성장으로 찾는다 → 인라이어 마스크 (없으면 None).

    끝쪽 초기 창(≥ init_min_mm, 폭의 init_frac)에서 RANSAC 직선 → 전체 점에 대해 인라이어 재계산 → 끝에 닿는 run 채택 →
    그 점들로 직선 재피팅 … 수렴까지 반복. 지배 직선(양 판 절충선)에 의존하지 않으므로 hi-lo 가 크거나 판이 약간 휘어도
    각 판을 따로 따라가며, 그루브(단차 > thr, 폭 > merge_gap)에서 멈춘다. 그루브는 프로파일 끝에서 초기 창 이상 떨어져야 한다.
    초기 창을 init_scales 배로 키워 가며 성장시키고 가장 긴 (인라이어가 많은) 결과를 쓴다 (2026-10-06 g3_23 S_corr1: 판이 0.2 mm 휜
    3G 시편에서 20 mm 초기 창의 직선이 조금 틀어져 왼쪽 판이 61 점 (~8 mm) 에서 멈추고 roll 이 −1.18° 로 읽혀 정렬이 발산; 같은 자세의
    g3_22 는 648 점, −0.11°)."""
    N = x.size
    span = float(x[-1] - x[0])
    best = None
    for k_ in init_scales:
        init = max(init_min_mm, init_frac * span) * float(k_)
        if k_ != init_scales[0] and init > 0.4 * span:              # 키운 창이 반대쪽 판까지 덮지 않게
            break
        sel = (x <= x[0] + init) if from_left else (x >= x[-1] - init)
        idx = np.nonzero(sel)[0]
        if idx.size < min_pts:
            continue
        # 첫 창은 호출자의 rng (예전과 같은 난수열 — 다른 판의 RANSAC 도 그대로), 키운 창은 따로 고정 시드
        r_ = rng if k_ == init_scales[0] else np.random.default_rng(int(1000 * float(k_)))
        ln, inl = ransac_line(x[idx], h[idx], thr, iters, r_, min_pts=min_pts)
        if ln is None:
            continue
        m = np.zeros(N, dtype=bool)
        m[idx[inl]] = True
        for _ in range(max_iter):
            ln = fit_line_pca(x[m], h[m])
            inl_all = np.abs(ln.dist(x, h)) < thr
            runs = _runs(inl_all, x, merge_gap)
            if from_left:
                r = next((r for r in runs if x[r[0]] <= x[0] + edge_tol), None)
            else:
                r = next((r for r in reversed(runs) if x[r[1]] >= x[-1] - edge_tol), None)
            if r is None:
                break
            m_new = _run_mask(N, r, inl_all)
            if np.array_equal(m_new, m):
                break
            m = m_new
        if m.sum() < min_pts or (x[m].max() - x[m].min()) < min_len:
            continue
        if k_ != init_scales[0] and not (x[m].min() <= x[0] + edge_tol if from_left else x[m].max() >= x[-1] - edge_tol):
            continue                                                  # 키운 창의 결과는 자기 쪽 끝에 닿아야 한다 (다른 판 직선 금지)
        if best is None or int(m.sum()) > int(best.sum()):
            best = m
    return best


def _plate_info(x, h, m: np.ndarray) -> tuple:
    """마스크 m 의 점으로 직선 (원 프레임) + 요약 dict. (run 은 x 정렬 인덱스 구간이지만 원근 때문에 바닥 점이
    판 점 사이에 끼어들 수 있으므로 반드시 인라이어 마스크로 골라 피팅한다.)"""
    ln = fit_line_pca(x[m], h[m])
    info = {"tilt_deg": -ln.angle_deg, "rms_mm": ln.rms_mm, "n": int(m.sum()), "x_min": float(x[m].min()),
            "x_max": float(x[m].max())}
    return ln, info


def _rot(x, h, phi):
    c, s = math.cos(phi), math.sin(phi)
    return x * c + h * s, -x * s + h * c


def _unrot(xp, hp, phi):
    c, s = math.cos(phi), math.sin(phi)
    return xp * c - hp * s, xp * s + hp * c


def _xz(xp, hp, phi) -> Optional[list]:
    if xp is None or hp is None or not (np.isfinite(xp) and np.isfinite(hp)):
        return None
    x, h = _unrot(float(xp), float(hp), phi)
    return [float(x), float(-h)]


def _dominant_floor(x: np.ndarray, h: np.ndarray, *, h_max: float, half_band: float, merge_gap: float,
                    min_pts: int = 3, step: float = 0.1, max_angle_deg: float = 15.0) -> tuple:
    """그루브 내부점 (x' 오름차순) → (바닥 인라이어 마스크, 바닥 직선) 또는 (None, None).

    후보 높이 c (h_max 아래, step 간격) 마다 대역 |h − c| ≤ half_band 의 점을 x 연속 구간(run, merge_gap 결합)으로 나눠 가장 큰
    run 의 점 수를 센다 → 최대인 c 의 run = 바닥 (같으면 더 낮은 c). 그 점으로 직선 (|기울기| > max_angle_deg 면 수평) 을 맞추고
    run 의 x 범위 안에서 |거리| ≤ half_band 인 점을 인라이어로 다시 고른다."""
    cand = h < h_max
    if int(cand.sum()) < min_pts:
        return None, None
    hs = np.sort(h[cand])
    levels = np.arange(float(hs[0]), float(h_max) + 1e-9, step)
    band_n = np.searchsorted(hs, levels + half_band, side="right") - np.searchsorted(hs, levels - half_band, side="left")
    # 대역 점 수가 많은 후보부터 (가장 큰 run ≤ 대역 점 수 이므로 그보다 작은 후보는 볼 필요 없다 — 같은 결과, 2026-10-02 속도);
    # 같으면 더 낮은 높이 (이전 오름차순 순회의 규칙)
    best = None
    for li in np.argsort(-band_n, kind="stable"):
        if band_n[li] < min_pts or (best is not None and band_n[li] < best[0]):
            break
        c = float(levels[li])
        m = cand & (np.abs(h - c) <= half_band)
        for a, b in _runs(m, x, merge_gap):
            n_ = int(m[a:b + 1].sum())
            if best is None or n_ > best[0] or (n_ == best[0] and c < best[4]):
                best = (n_, a, b, m.copy(), c)
    if best is None or best[0] < min_pts:
        return None, None
    _n, a, b, m, _c = best
    run = np.zeros(x.size, dtype=bool)
    run[a:b + 1] = m[a:b + 1]
    ln = fit_line_pca(x[run], h[run]) if int(run.sum()) >= 2 and float(np.ptp(x[run])) > 0.5 else None
    if ln is None or abs(ln.angle_deg) > max_angle_deg:
        ln = Line2D([float(np.median(x[run])), float(np.median(h[run]))], [1.0, 0.0], int(run.sum()))
    inl = (x >= x[a] - 1e-9) & (x <= x[b] + 1e-9) & (np.abs(ln.dist(x, h)) <= half_band)
    if int(inl.sum()) < min_pts:
        inl = run
    return inl, ln


def _extend_plate(x: np.ndarray, h: np.ndarray, m: np.ndarray, toward_plus: bool, *, tol: float, k: int,
                  gap: float) -> np.ndarray:
    """판 마스크 m 을 그루브 쪽 (toward_plus: x 증가 방향) 으로 연속성으로 연장. 다음 점 (x 순) 이 직전 k 개 판 점의 h 중앙값에서
    tol 이내면 채택, 한 점 벗어남은 건너뛰고 두 점 연속이면 멈춘다. x 간격 > gap 이어도 멈춘다."""
    out = m.copy()
    idx_m = np.nonzero(m)[0]
    if idx_m.size == 0:
        return out
    order = np.argsort(x, kind="stable")
    pos = np.empty(x.size, dtype=int)
    pos[order] = np.arange(x.size)
    if toward_plus:
        last = idx_m[np.argmax(x[idx_m])]
        seq = order[pos[last] + 1:]
    else:
        last = idx_m[np.argmin(x[idx_m])]
        seq = order[:pos[last]][::-1]
    acc = list(idx_m[np.argsort(x[idx_m])][-k:] if toward_plus else idx_m[np.argsort(x[idx_m])][:k])
    x_last = float(x[last])
    misses = 0
    for i in seq:
        if abs(float(x[i]) - x_last) > gap:
            break
        ref = float(np.median(h[np.asarray(acc[-k:])]))
        if abs(float(h[i]) - ref) <= tol:
            out[i] = True
            acc.append(i)
            x_last = float(x[i])
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    return out


def _attached(x: np.ndarray, m: np.ndarray, x_edge: float, toward_plus: bool, gap: float,
              bridge: Optional[np.ndarray] = None) -> np.ndarray:
    """m 의 점 중 판 끝 x_edge 에서 그루브 안쪽으로 간격 ≤ gap 으로 이어진 점 (첫 점도 판 끝에서 gap 이내).
    bridge: 이어짐 판단에만 쓰는 점 (그룹에는 넣지 않음) — 반사로 버린 점 (바닥 아래) 의 x 는 면이 있던 자리라 간격을 메운다
    (2026-10-02 at06: 베벨 윗모서리의 깊은 반사 띠 ~1.5 mm 를 버리자 판 끝과 베벨면 사이가 ~2 mm 벌어져 베벨 전체가 떠 있는
    점으로 빠지고 갭이 개구 폭 11 mm 로 읽혔다)."""
    out = np.zeros(x.size, dtype=bool)
    idx = np.nonzero(m)[0]
    if idx.size == 0:
        return out
    use = m if bridge is None else (m | bridge)
    iu = np.nonzero(use)[0]
    d = (x[iu] - x_edge) if toward_plus else (x_edge - x[iu])
    order = np.argsort(d)
    prev = 0.0
    for k in order:
        if d[k] - prev > gap:
            break
        if m[iu[k]]:
            out[iu[k]] = True
        prev = max(prev, float(d[k]))
    return out


def _bridge_deep_strips(xp: np.ndarray, hp: np.ndarray, mL: np.ndarray, mR: np.ndarray, lines: dict, *, tol: float,
                        merge_gap: float, min_run: float, depth_min: float, depth_cap: float, min_deep: int = 3) -> tuple:
    """두 판 사이 (판 정렬 프레임) 에 판 높이의 연속 구간 (≥ min_run) 이 있어 내부가 여러 조각이면, 판 뒤로 아주 깊은 점
    (> depth_cap) 만 있는 조각은 판을 끊은 반사·투과 띠로 보고 판을 그 너머로 잇는다 — 그루브 깊이 범위 (depth_min ~ depth_cap)
    의 점이 있는 조각이 정확히 하나일 때만. 2026-10-05 g3_04 E (3G, 센서가 테이블 높이까지 내려감): 그루브 왼쪽 25 mm 에 판 뒤
    85–112 mm 의 점 띠 (폭 ~5 mm) 가 왼쪽 판을 끊어, 그 띠와 진짜 그루브 사이의 판 21 mm 가 그루브로 들어가 개구 22.4 mm (설계
    10.8) 로 읽혔다. → (mL, mR, note 또는 None). 빈 조각 (점 없음) 과 애매한 경우는 그대로 둔다 (열린 I형 그루브를 판으로 덮지 않게)."""
    xL_end, xR_start = float(xp[mL].max()), float(xp[mR].min())
    inside = (~mL) & (~mR) & (xp > xL_end) & (xp < xR_start)
    idx = np.nonzero(inside)[0]
    if idx.size == 0 or "left" not in lines or "right" not in lines:
        return mL, mR, None
    idx = idx[np.argsort(xp[idx], kind="stable")]
    xi, hi = xp[idx], hp[idx]
    hL = np.asarray(lines["left"].h_at(xi), dtype=float)
    hR = np.asarray(lines["right"].h_at(xi), dtype=float)
    dev = np.where(np.abs(hi - hL) <= np.abs(hi - hR), hi - hL, hi - hR)      # 가까운 판 직선 기준 (+ 위, − 깊음)
    lvl = np.abs(dev) <= tol
    pi = np.nonzero(lvl)[0]
    runs = []
    if pi.size:
        st = pr = int(pi[0])
        for q in pi[1:]:
            if xi[q] - xi[pr] > merge_gap:
                runs.append((st, pr))
                st = int(q)
            pr = int(q)
        runs.append((st, pr))
    runs = [(a, b) for a, b in runs if xi[b] - xi[a] >= min_run]
    if not runs:
        return mL, mR, None
    cuts = [xL_end] + [v for a, b in runs for v in (float(xi[a]), float(xi[b]))] + [xR_start]
    segs = []
    for j in range(0, len(cuts), 2):
        lo, hi_ = cuts[j], cuts[j + 1]
        m = (xi > lo) & (xi < hi_) & ~lvl
        dep = -dev[m]
        segs.append({"lo": lo, "hi": hi_, "n_mod": int(((dep >= depth_min) & (dep <= depth_cap)).sum()),
                     "n_deep": int((dep > depth_cap).sum()), "deep_med": float(np.median(dep[dep > depth_cap])) if (dep > depth_cap).any()
                     else float("nan")})
    groove = [g for g in segs if g["n_mod"] >= 3]
    others = [g for g in segs if g not in groove]
    if len(groove) != 1 or not others or any(g["n_mod"] > 1 or g["n_deep"] < min_deep for g in others):
        return mL, mR, None
    g = groove[0]
    lvl_full = np.zeros(xp.size, dtype=bool)
    lvl_full[idx[lvl]] = True
    mL2 = mL | (lvl_full & (xp <= g["lo"]))
    mR2 = mR | (lvl_full & (xp >= g["hi"]))
    note = "; ".join(f"plate interrupted by {o['n_deep']} points {o['deep_med']:.0f} mm behind it at x' [{o['lo']:.1f}, {o['hi']:.1f}] mm "
                     "outside the groove (reflection / see-through strip): bridged" for o in others)
    return mL2, mR2, note


def _spread(v: np.ndarray) -> float:
    if v.size == 0:
        return 0.0
    if v.size >= 4:
        return float(np.percentile(v, 90) - np.percentile(v, 10))
    return float(v.max() - v.min())


# --------------------------------------------------------------------------------------
# 단면 특징 추출
# --------------------------------------------------------------------------------------
def points_above_plates(x_mm, z_mm, f: "CrossSectionFeatures", min_height_mm: float = 5.0) -> dict:
    """판 상면보다 min_height_mm 넘게 높은 점 (센서에 더 가까움) — 용접 전 빈 그루브에서는 그런 표면이 없다 (판 상면이 가장 높다; 단차·가접
    비드는 수 mm). 있으면 레이저 선 위의 이물 (늘어진 케이블·공구 등; 2026-10-05 at21: 손목이 반대로 감겨 센서 케이블이 센서 앞에 늘어져 판 위
    ~60 mm 에 걸렸고, 추출기가 그 그림자 사이를 개구 53 mm 그루브로 오인했다). 판 인라이어 (f.plate_inlier_idx_*, 입력 배열 인덱스) 의
    기울기 제거 높이 중 높은 쪽을 판 상면으로 본다. → {n, max_height_mm, x_min_mm, x_max_mm} (판을 못 찾았으면 n = 0)."""
    x = np.asarray(x_mm, dtype=float).reshape(-1)
    z = np.asarray(z_mm, dtype=float).reshape(-1)
    out = {"n": 0, "max_height_mm": float("nan"), "x_min_mm": float("nan"), "x_max_mm": float("nan")}
    sides = [np.asarray(i, dtype=np.int64).reshape(-1) for i in (f.plate_inlier_idx_left, f.plate_inlier_idx_right) if i is not None]
    sides = [i for i in sides if i.size >= 3]
    if not sides or not np.isfinite(f.detilt_phi_deg):
        return out
    phi = math.radians(float(f.detilt_phi_deg))
    with np.errstate(invalid="ignore"):
        v = np.isfinite(x) & np.isfinite(z) & (z > 0.0)
    _xp, hp = _rot(x, -z, phi)
    level = max(float(np.median(hp[i])) for i in sides)
    with np.errstate(invalid="ignore"):
        above = v & (hp > level + float(min_height_mm))
    if above.any():
        out.update(n=int(above.sum()), max_height_mm=float(np.max(hp[above]) - level), x_min_mm=float(np.min(x[above])),
                   x_max_mm=float(np.max(x[above])))
    return out


def _qc_tol(qc_targets: Optional[dict], key: str, field_: str = "tol") -> float:
    """작업 qc_targets (없으면 DEFAULT_QC_TARGETS) 의 설계 허용치."""
    tg = dict(DEFAULT_QC_TARGETS)
    for k_, v_ in (qc_targets or {}).items():
        tg[k_] = {**tg.get(k_, {}), **(v_ or {})}
    v = (tg.get(key) or {}).get(field_)
    return float("nan") if v is None else float(v)


def extractor_params(spec: Optional[GrooveSpec] = None, qc_targets: Optional[dict] = None, dx_mm: Optional[float] = None) -> dict:
    """단면 특징 추출기의 판정 문턱 — 설정 상수 없이 (scan_core.measure 원칙, 사용자 2026-10-07 4c). 원천:
      * 설계 (spec — 없으면 기본 시편 GrooveSpec()): 개구 O, 갭 G, 깊이 D, 베벨각 β (베벨 없는 종류는 0).
      * 작업 허용치 (qc_targets — 없으면 DEFAULT_QC_TARGETS): 깊이 tol_D, 갭 tol_G, hi-lo 목표·경고.
      * 측정: 열 간격 dx (프로파일의 x 간격 중앙값).
    문턱:
      * 면 소속 (원칙 2: 작업 정밀도로 같은 면과 구별되지 않으면 같은 면): 판 직선 인라이어 = tol_D/2 (전체 판에 맞춘 한 직선 — 판의 휨에
        나머지 반), 판 연장 연속성·바닥 대역 반폭 = tol_D (이웃 점·국소 대역 기준), 바닥은 판보다 tol_D 넘게 깊어야, 판이 들렸다 =
        판 바닥이 바닥보다 tol_D 넘게 높음, 바닥이 루트 모서리에 닿음 = tol_G + dx 안.
      * 길이 (설계): 결측 이음 = G/2 (점이 빠진 틈 ↔ 설계상 가장 좁은 틈인 루트 갭의 중간; 갭 없는 설계는 판 최소 길이의 절반),
        판 최소 길이 = O/2 (그루브 개구의 절반 — 루트면·띠와 판의 중간), 그루브 쪽 국소 판 직선 = O, 판이 프로파일
        끝에 닿음 = 판 최소 길이 안, 깊은 띠 = 바닥보다 D/2 넘게 깊음 (깊이 D + D/2).
      * 각도 (설계 중간점): 벽 ↔ 베벨 경계 = β/2 (베벨 없는 종류는 45°), 베벨 ↔ 수평 경계 = (β + 90)/2, 바닥 직선 기울기 한계 =
        (90 − β)/2 (바닥 ↔ 베벨의 중간), 두 판 평행 = 두 판 직선이 개구를 가로질러 만드는 높이 차가 hi-lo 목표 이하
        (atan(target / O)).
      * 증거 (원칙 2: 기대 점 수의 절반): 프로파일 최소 점 = 판 최소 길이 둘의 열 수 절반, 판 연장 기준 점 = 판 최소 길이의 열 수 절반,
        바닥 보임 = 갭 절반 폭 이상 + 그 폭의 열 수 절반. 벽 전단 (진단 전용) 직선은 수치 최소 4 점 (2 점 가설 + 확인 2 점 — 수직 벽은
        정렬 자세에서 열이 거의 닿지 않아 기대 점 수가 없다).
      * 벽/베벨 직선을 맞출 만큼 퍼졌나 = x 퍼짐이 면 소속 띠 폭 (바닥 대역과 같은 2·tol_D) 을 넘음 — 그 안이면 한 x 에 선 벽; 베벨/벽
        결정은 각도 경계가 한다.
      * hi-lo 장면 경고 = 작업 hi_lo_mm.warn_above."""
    sp = spec if spec is not None else GrooveSpec()
    dx = float(dx_mm) if (dx_mm is not None and np.isfinite(float(dx_mm)) and float(dx_mm) > 0.0) else 0.1
    O = float(sp.top_opening_mm)
    G = float(sp.root_gap_mm)
    D = float(sp.groove_depth_mm)
    beta = float(sp.bevel_angle_deg) if sp.n_bevels > 0 else 0.0
    tol_D = _qc_tol(qc_targets, "groove_depth_mm")
    tol_G = _qc_tol(qc_targets, "root_gap_mm")
    hl_t = _qc_tol(qc_targets, "hi_lo_mm", "target_max")
    hl_w = _qc_tol(qc_targets, "hi_lo_mm", "warn_above")
    split = 0.5 * beta if beta > 0.0 else HALF_TURN_DEG
    plate_min = DESIGN_HALF * O
    return {
        "inlier_mm": DESIGN_HALF * tol_D,
        "plate_extend_tol_mm": tol_D,
        "floor_band_mm": 2.0 * tol_D,
        "depth_min_mm": tol_D,
        "lift_tol_mm": tol_D,
        "floor_reach_mm": (tol_G if np.isfinite(tol_G) else tol_D) + dx,
        "merge_gap_mm": DESIGN_HALF * G if G > 0.0 else DESIGN_HALF * plate_min,
        "min_plate_len_mm": plate_min,
        "plate_local_mm": O,
        "edge_tol_mm": max(plate_min, 3.0 * dx),
        "depth_cap_mm": D + DESIGN_HALF * D,
        "bevel_min_angle_deg": split,
        "wall_max_lean_deg": split,
        "bevel_max_angle_deg": 0.5 * (beta + 90.0) if beta > 0.0 else 90.0,
        "floor_max_angle_deg": 0.5 * (90.0 - beta) if beta > 0.0 else HALF_TURN_DEG,
        "max_plate_angle_diff_deg": math.degrees(math.atan2(hl_t, O)) if np.isfinite(hl_t) else HALF_TURN_DEG,
        "min_points": evidence_count(2.0 * plate_min / dx),
        "plate_extend_k": evidence_count(plate_min / dx),
        "floor_min_points": evidence_count(DESIGN_HALF * G / dx) if G > 0.0 else 1,
        "floor_visible_frac": DESIGN_HALF,
        "wall_min_points": 4,
        "wall_spread_max_mm": 2.0 * tol_D,
        "hi_lo_warn_mm": hl_w if np.isfinite(hl_w) else HI_LO_AMBIGUOUS_MM,
    }


def plate_min_len_mm(spec: Optional[GrooveSpec] = None) -> float:
    """특징 추출기가 그루브 옆 판 하나로 인정하는 최소 길이 [mm] = 설계 개구의 절반 (extractor_params; 게이트의 데이터 유효성 기준과
    끝 판정의 판 최소 면 길이도 같은 원천)."""
    return float(extractor_params(spec)["min_plate_len_mm"])


def extract_cross_section_features(x_mm, z_mm, spec: Optional[GrooveSpec] = None, sensor_tilt_hint: Optional[float] = None, *,
                                   inlier_mm: Optional[float] = None, merge_gap_mm: Optional[float] = None,
                                   min_plate_len_mm: Optional[float] = None, edge_tol_mm: Optional[float] = None,
                                   floor_band_mm: Optional[float] = None, wall_spread_max_mm: Optional[float] = None,
                                   depth_min_mm: Optional[float] = None, ransac_iters: int = 200, min_points: Optional[int] = None,
                                   max_plate_angle_diff_deg: Optional[float] = None, qc_targets: Optional[dict] = None,
                                   seed: int = 0, wall_min_points: Optional[int] = None, wall_max_lean_deg: Optional[float] = None,
                                   floor_min_points: Optional[int] = None, floor_visible_frac: Optional[float] = None,
                                   plate_extend_tol_mm: Optional[float] = None, plate_extend_k: Optional[int] = None,
                                   plate_local_mm: Optional[float] = None, bevel_min_angle_deg: Optional[float] = None,
                                   depth_scale: float = 1.0, plate_seed: Optional[PlateSeed] = None,
                                   _plates_only: bool = False) -> CrossSectionFeatures:
    """프로파일 1개 (또는 시간 중앙값) → CrossSectionFeatures. 모듈 docstring 의 절차.

    depth_scale = cos(레이저 평면과 판 법선 사이 각) = √(1 − (n·Y_opt)²) (2026-10-06 g3_07): 레이저 평면이 판 법선에서 t 만큼 (입사 기울기
    + 피치) 기울면 평면 안에서 판 법선 방향 깊이가 1/cos t 로 늘어 보인다 (35°+5°: 깊이 10.07 → 13.11, 베벨 31.7° → 22.9°, 그리고 '판 상면 −
    설계 두께' 비교가 판이 들린 것으로 오인해 갭 4.95 → 7.31). 판 정렬 프레임의 h' 에 depth_scale 을 곱해 판 법선 단위로 추출하고, 센서 좌표
    출력 (코너·루트 모서리·중심) 은 되돌린다 — 3D 점은 그대로, 깊이·베벨·hi-lo·갭 판정만 바뀐다. 1 = 보정 없음 (예전).
    정렬 진단 (4b/5b): wall_shear_k = 직각 벽 인라이어(≥ wall_min_points, 수직에서 wall_max_lean_deg 이내) 직선의 dx'/dh';
    floor_visible = 바닥 직선 인라이어 ≥ floor_min_points 이고 x' 범위 ≥ floor_visible_frac × 기대 루트갭.
    plate_seed: 같은 (x, z)·같은 인자로 미리 계산한 판 성장 (cross_section_plate_seed) — 주면 1~2 단계를 건너뛰고 그 결과·난수 상태를
    이어받는다 (결과는 한 번에 계산한 것과 같다; 입력이 같은지는 호출자가 확인한다 — 유효 점 수가 다르면 쓰지 않는다).
    """
    f = CrossSectionFeatures()
    x = np.asarray(x_mm, dtype=float).reshape(-1)
    z = np.asarray(z_mm, dtype=float).reshape(-1)
    with np.errstate(invalid="ignore"):
        v = np.isfinite(x) & np.isfinite(z) & (z > 0.0)
    f.n_valid = int(v.sum())
    if f.n_valid < 3:
        f.flags.append("insufficient_points")
        f.insufficient_points = True
        f.notes.append(f"only {f.n_valid} valid points")
        return f
    order = np.argsort(x[v], kind="stable")
    orig_idx = np.nonzero(v)[0][order]                 # 정렬된 유효점 → 입력 배열 열 인덱스
    x = x[v][order]
    h = -z[v][order]
    dxs = np.diff(x)
    dxs = dxs[np.isfinite(dxs) & (dxs > 0)]
    dx = float(np.median(dxs)) if dxs.size else float(dX_mm(float(np.median(-h))))   # 열 간격 (측정; 없으면 센서 모형)
    f.dx_mm = dx
    # 판정 문턱은 설계·작업 허용치·측정 열 간격에서 (extractor_params; 명시한 인자가 우선)
    P_ = extractor_params(spec, qc_targets, dx)
    _given = dict(inlier_mm=inlier_mm, merge_gap_mm=merge_gap_mm, min_plate_len_mm=min_plate_len_mm, edge_tol_mm=edge_tol_mm,
                  floor_band_mm=floor_band_mm, wall_spread_max_mm=wall_spread_max_mm, depth_min_mm=depth_min_mm,
                  min_points=min_points, max_plate_angle_diff_deg=max_plate_angle_diff_deg, wall_min_points=wall_min_points,
                  wall_max_lean_deg=wall_max_lean_deg, floor_min_points=floor_min_points, floor_visible_frac=floor_visible_frac,
                  plate_extend_tol_mm=plate_extend_tol_mm, plate_extend_k=plate_extend_k, plate_local_mm=plate_local_mm,
                  bevel_min_angle_deg=bevel_min_angle_deg)
    P_.update({k_: v_ for k_, v_ in _given.items() if v_ is not None})
    inlier_mm, merge_gap_mm, min_plate_len_mm = P_["inlier_mm"], P_["merge_gap_mm"], P_["min_plate_len_mm"]
    floor_band_mm, wall_spread_max_mm, depth_min_mm = P_["floor_band_mm"], P_["wall_spread_max_mm"], P_["depth_min_mm"]
    min_points, max_plate_angle_diff_deg = int(P_["min_points"]), P_["max_plate_angle_diff_deg"]
    wall_min_points, wall_max_lean_deg = int(P_["wall_min_points"]), P_["wall_max_lean_deg"]
    floor_min_points, floor_visible_frac = int(P_["floor_min_points"]), P_["floor_visible_frac"]
    plate_extend_tol_mm, plate_extend_k, plate_local_mm = P_["plate_extend_tol_mm"], int(P_["plate_extend_k"]), P_["plate_local_mm"]
    bevel_min_angle_deg = P_["bevel_min_angle_deg"]
    f.hi_lo_warn_mm = float(P_["hi_lo_warn_mm"])
    f.min_points = int(min_points)
    if f.n_valid < min_points:
        f.flags.append("insufficient_points")
        f.insufficient_points = True
        f.notes.append(f"only {f.n_valid} valid points (< {min_points})")
        return f
    rng = np.random.default_rng(seed)
    edge_tol = P_["edge_tol_mm"] if edge_tol_mm is None else max(edge_tol_mm, 3.0 * dx)

    # ---- 1~2. 판 (양 끝에서 영역 성장, 인라이어 마스크) ----
    grow_kw = dict(thr=inlier_mm, merge_gap=merge_gap_mm, min_len=min_plate_len_mm, edge_tol=edge_tol, iters=ransac_iters,
                   rng=rng, min_pts=max(3, min_points // 4), init_min_mm=plate_local_mm)
    if plate_seed is not None and int(plate_seed.n) == int(x.size):
        # 패스 중 미리 계산한 판 성장 (같은 점·같은 인자): 마스크와 그 뒤 난수 상태를 이어받는다 — 아래 단계는 한 번에 계산한 것과 같다
        mL = None if plate_seed.mL is None else np.array(plate_seed.mL, dtype=bool, copy=True)
        mR = None if plate_seed.mR is None else np.array(plate_seed.mR, dtype=bool, copy=True)
        rng.bit_generator.state = plate_seed.rng_state
    else:
        mL = _grow_plate(x, h, True, **grow_kw)
        mR = _grow_plate(x, h, False, **grow_kw)
    if _plates_only:
        return PlateSeed(n=int(x.size), mL=mL, mR=mR, rng_state=rng.bit_generator.state)

    if mL is None and mR is None:
        f.flags.append("insufficient_points")
        f.insufficient_points = True
        f.notes.append("no plate-top line touching either end of the profile")
        return f
    f.plate_inlier_idx_left = orig_idx[mL] if mL is not None else None
    f.plate_inlier_idx_right = orig_idx[mR] if mR is not None else None

    def _hint_note():
        if sensor_tilt_hint is not None and abs(abs(f.tilt_deg) - abs(float(sensor_tilt_hint))) > max_plate_angle_diff_deg:
            f.notes.append(f"plate tilt {f.tilt_deg:+.1f} deg differs from sensor_tilt_hint {float(sensor_tilt_hint):+.1f} deg by > "
                           f"{max_plate_angle_diff_deg:.1f} deg")

    if mL is not None and mR is not None and np.array_equal(mL, mR):
        # 하나의 연속 판: 그루브 없음
        ln, info = _plate_info(x, h, mL)
        f.plate_left = f.plate_right = dict(info)
        f.tilt_deg = info["tilt_deg"]
        f.detilt_phi_deg = ln.angle_deg
        f.hi_lo_mm = 0.0
        f.plate_angle_diff_deg = 0.0
        f.flags.append("no_groove")
        _hint_note()
        f.ok = True
        return f

    # ---- 3. de-tilt ----
    lines0 = {}
    if mL is not None:
        lines0["left"], f.plate_left = _plate_info(x, h, mL)
    if mR is not None:
        lines0["right"], f.plate_right = _plate_info(x, h, mR)
    # 판 상면 기울기 dz/dx (센서 프레임) — h = −z 이므로 dz/dx = −dh/dx = tan(tilt_deg)
    for side_, attr_ in (("left", "plate_slope_left"), ("right", "plate_slope_right")):
        if side_ in lines0:
            dl = lines0[side_].d
            setattr(f, attr_, float(-dl[1] / dl[0]) if abs(dl[0]) > 1e-12 else float("nan"))
    angs = [ln.angle_deg for ln in lines0.values()]
    phi = math.radians(float(np.mean(angs)))
    f.detilt_phi_deg = math.degrees(phi)
    f.tilt_deg = -math.degrees(phi)
    if len(angs) == 2:
        f.plate_angle_diff_deg = abs(angs[0] - angs[1])
        if f.plate_angle_diff_deg > max_plate_angle_diff_deg:
            # 한쪽 '판'이 베벨/벽/다른 물체일 가능성 (그루브가 프로파일 끝 초기 창 안에 들어온 경우 등) → 결과 신뢰 불가
            f.flags.append("plates_not_parallel")
            f.insufficient_points = True
            f.notes.append(f"plate-top lines differ by {f.plate_angle_diff_deg:.1f} deg > {max_plate_angle_diff_deg} deg: "
                           "one side is probably not a plate top (groove too close to the profile end?)")
    _hint_note()
    xp, hp = _rot(x, h, phi)
    ds = float(depth_scale) if (depth_scale is not None and np.isfinite(depth_scale) and 0.0 < float(depth_scale) <= 1.0) else 1.0
    if ds != 1.0:
        hp = hp * ds                                              # 판 법선 단위 깊이 (센서 좌표 출력은 /ds 로 되돌림)
        f.notes.append(f"depth axis scaled by {ds:.4f} (laser plane {math.degrees(math.acos(ds)):.1f} deg off the plate normal)")
    lines = {}
    if mL is not None:
        lines["left"] = fit_line_pca(xp[mL], hp[mL])
    if mR is not None:
        lines["right"] = fit_line_pca(xp[mR], hp[mR])

    if mL is None or mR is None:
        f.flags.append("insufficient_points")
        f.insufficient_points = True
        f.notes.append("only one plate found; groove edges undetermined")
        side, m = ("left", mL) if mL is not None else ("right", mR)
        lvl = float(lines[side].h_at(xp[m].max() if side == "left" else xp[m].min()))
        (f.plate_left if side == "left" else f.plate_right)["level_mm"] = lvl / ds
        return f
    # 판을 그루브 쪽으로 연속성 연장 (mL/mR 는 엄격한 인라이어 그대로 = plate_inlier_idx; 그루브 기하는 연장 마스크)
    ext_kw = dict(tol=plate_extend_tol_mm, k=max(3, int(plate_extend_k)), gap=merge_gap_mm)
    mL = _extend_plate(xp, hp, mL, True, **ext_kw)
    mR = _extend_plate(xp, hp, mR, False, **ext_kw)
    depth_cap = P_["depth_cap_mm"]
    mL, mR, bridged = _bridge_deep_strips(xp, hp, mL, mR, lines, tol=plate_extend_tol_mm, merge_gap=merge_gap_mm,
                                          min_run=min_plate_len_mm, depth_min=depth_min_mm, depth_cap=depth_cap,
                                          min_deep=evidence_count(merge_gap_mm / dx))
    if bridged:
        f.flags.append("plate_strip_bridged")
        f.notes.append(bridged)
    face_pts: dict = {}                                             # 면 이름 → 정렬된 유효점 인덱스 (센서 프레임 면 직선, FACE_KINDS)
    for side_, m_ in (("left", mL), ("right", mR)):
        xe = float(xp[m_].max()) if side_ == "left" else float(xp[m_].min())
        # 그루브 쪽 국소 판 직선 (레벨·코너용). 한계 (2026-10-04): 연장 점 (허용 plate_extend_tol 안) 이 개선면 시작을 포함해 국소 기울기를
        # 틀 수 있고, 레벨을 그루브 중심까지 외삽하므로 넓은 그루브일수록 커진다 (합성 25T/45°: 깊이 −0.23, 갭 +0.2 mm). 엄격한 인라이어만
        # 쓰면 그 합성은 낫지만 실측 17 자세 중 2 개에서 hi-lo 가 +0.5/+0.9 mm 로 나빠졌다 (판 모서리 근처 엄격 인라이어가 적다) — 그대로 둔다.
        loc = m_ & ((xp >= xe - plate_local_mm) if side_ == "left" else (xp <= xe + plate_local_mm))
        face_pts[f"plate_{side_}"] = np.flatnonzero(m_)
        if int(loc.sum()) >= 3 and float(np.ptp(xp[loc])) > 1.0:
            lines[side_] = fit_line_pca(xp[loc], hp[loc])           # 그루브 쪽 국소 판 직선 (레벨·코너용)
            face_pts[f"plate_{side_}"] = np.flatnonzero(loc)        # 판 면 = 같은 국소 구간 (용접이 닿는 그루브 쪽)
    # 그루브 x' 범위 (판 정렬 프레임): 왼 판의 최대 x' ~ 오른 판의 최소 x'
    xL_end, xR_start = float(xp[mL].max()), float(xp[mR].min())
    if xR_start - xL_end <= dx:
        f.flags.append("no_groove")
        f.hi_lo_mm = float(lines["right"].h_at(xR_start) - lines["left"].h_at(xL_end))
        f.ok = "plates_not_parallel" not in f.flags
        return f
    xc0 = 0.5 * (xL_end + xR_start)
    lvl_L = float(lines["left"].h_at(xc0))
    lvl_R = float(lines["right"].h_at(xc0))
    f.plate_left["level_mm"] = lvl_L / ds
    f.plate_right["level_mm"] = lvl_R / ds
    f.hi_lo_mm = lvl_R - lvl_L
    if f.hi_lo_ambiguous:
        f.flags.append("hi_lo_ambiguous")
    f.groove_found = True

    # ---- 4. 내부 (판 정렬 프레임에서 두 판 사이, 판 인라이어 제외) ----
    mI = (~mL) & (~mR) & (xp > xL_end) & (xp < xR_start)
    inter = np.nonzero(mI)[0]
    f.n_interior_points = int(inter.size)
    o_i = np.argsort(xp[inter], kind="stable")                  # 내부점을 x' 순으로 (run 판정용)
    inter = inter[o_i]
    xi, hi = xp[inter], hp[inter]
    lvl_low = min(lvl_L, lvl_R)
    floor_line = None
    floor_level = float("nan")
    floor_x = None
    keep = np.ones(inter.size, dtype=bool)                        # 반사로 판정되지 않은 내부점
    below = np.zeros(inter.size, dtype=bool)                      # 바닥 아래 반사 (제외하되 판 끝 이어짐은 메운다)
    if inter.size >= 3:
        fl, floor_line = _dominant_floor(xi, hi, h_max=lvl_low - depth_min_mm, half_band=0.5 * floor_band_mm,
                                         merge_gap=merge_gap_mm, min_pts=3, max_angle_deg=P_["floor_max_angle_deg"], step=dx)
        if floor_line is None:
            f.flags.append("no_floor")
            f.notes.append(f"interior not deeper than {depth_min_mm} mm below the lower plate top (shallow/none)")
        else:
            fx, fh = xi[fl], hi[fl]
            face_pts["floor"] = inter[fl]                           # 바닥 면 (뒷댐재 상면) 인라이어
            f.n_floor_points = int(fl.sum())
            floor_x = (float(fx.min()), float(fx.max()))
            floor_level = float(floor_line.h_at(0.5 * (floor_x[0] + floor_x[1])))
            f.floor_level_mm = floor_level / ds
            f.floor_width_mm = floor_x[1] - floor_x[0]
            f.root_gap_floor_mm = f.floor_width_mm
            below = hi < floor_line.h_at(xi) - floor_band_mm
            f.n_below_floor = int(below.sum())
            keep &= ~below
            if f.n_below_floor:
                f.flags.append("below_floor_reflections")
                f.notes.append(f"{f.n_below_floor} interior points more than {floor_band_mm} mm below the floor "
                               f"(down to {floor_level - float(hi[below].min()):.1f} mm): reflections, ignored")
    else:
        f.flags.append("insufficient_points")
        f.insufficient_points = True
        f.notes.append(f"only {inter.size} interior points between the plate runs")

    # 바닥 위 내부점 → 좌/우 그룹
    if floor_line is not None:
        above = keep & (hi > (floor_line.h_at(xi) + floor_band_mm))
        xc_f = 0.5 * (floor_x[0] + floor_x[1])
    else:
        above = keep.copy()
        xc_f = 0.5 * (xL_end + xR_start)
    groups = {"left": _attached(xi, above & (xi < xc_f), xL_end, True, merge_gap_mm, bridge=below & (xi < xc_f)),
              "right": _attached(xi, above & (xi >= xc_f), xR_start, False, merge_gap_mm, bridge=below & (xi >= xc_f))}
    stray = above & ~groups["left"] & ~groups["right"]
    f.n_stray_points = int(stray.sum())
    if f.n_stray_points:
        f.notes.append(f"{f.n_stray_points} points above the floor not attached to either plate edge (reflections, ignored)")
    bevel_lines = {}
    wall_x = {}
    for side, m in groups.items():
        n_ = int(m.sum())
        if n_ == 0:
            continue
        sp = _spread(xi[m])
        if n_ >= 4 and sp > wall_spread_max_mm:
            ln, inl = ransac_line(xi[m], hi[m], inlier_mm, ransac_iters, rng, min_pts=3)
            if ln is None:
                ln = fit_line_pca(xi[m], hi[m])
                inl = np.ones(n_, dtype=bool)
            if float(bevel_min_angle_deg) <= ln.angle_from_vertical_deg < P_["bevel_max_angle_deg"]:   # 거의 수평·거의 수직(기운 벽)은 베벨이 아님
                bevel_lines[side] = ln
                face_pts[f"bevel_{side}"] = inter[np.flatnonzero(m)[inl]]     # 베벨면 = 이 직선의 인라이어
            else:
                wall_x[side] = float(np.median(xi[m]))
        else:
            wall_x[side] = float(np.median(xi[m]))
    if len(bevel_lines) == 2:
        f.bevel_side = "both"
    elif len(bevel_lines) == 1:
        f.bevel_side = next(iter(bevel_lines))
    elif inter.size >= 3:
        f.bevel_side = "none"
    else:
        f.bevel_side = "unknown"
        f.flags.append("bevel_unresolved")
    if "left" in bevel_lines:
        f.bevel_angle_left_deg = bevel_lines["left"].angle_from_vertical_deg
    if "right" in bevel_lines:
        f.bevel_angle_right_deg = bevel_lines["right"].angle_from_vertical_deg
    if bevel_lines:
        f.bevel_angle_deg = float(np.mean([ln.angle_from_vertical_deg for ln in bevel_lines.values()]))

    # ---- 5. 코너 · 루트 모서리 ----
    def _root_line(side: str) -> Optional[Line2D]:
        """베벨이 끝나는 수평선 (루트 모서리 = 베벨 ∩ 이 선). 단면 모델: 벽 = 루트면 (수직, 높이 rf, 설계값) + 베벨면.
        밑면 = 바닥선 (뒷댐재 상면); spec 이 있고 그 판의 바닥(상면 − t)이 바닥보다 높으면 (판이 들림 = hi-lo) 판 바닥 높이 — 베벨을
        뒷댐재까지 외삽하면 갭이 t_lift·tan(bevel) 만큼 작아지기 때문. 루트면이 있으면 (rf > 0) 밑면 + rf — 베벨을 밑면까지 외삽하면 갭이
        rf·tan(bevel) 만큼 (V 는 양쪽) 작아진다 (2026-10-04: 10T rf 2 mm 갭 3 → 1.89, 12T V rf 2 갭 2 → −0.32 mm 로 나왔다)."""
        if floor_line is None:
            return None
        if spec is None:
            return floor_line
        lvl = lvl_L if side == "left" else lvl_R
        root_h = lvl - spec.thickness_mm
        rf = float(spec.root_face_mm)
        if root_h > floor_level + P_["lift_tol_mm"]:
            f.notes.append(f"{side} bevel root extrapolated to plate bottom (top - t = {root_h - floor_level:+.2f} mm above the floor)")
            return Line2D([xc0, root_h + rf], [1.0, 0.0])
        if rf > 0.0:
            return Line2D([xc0, float(floor_line.h_at(xc0)) + rf], [1.0, 0.0])
        return floor_line

    def _corner(side: str):
        """(x'_top, h'_top, x'_root)"""
        pl = lines[side]
        if side in bevel_lines:
            pt = bevel_lines[side].intersect(pl)
            xt = float(pt[0]) if pt is not None else float("nan")
            xr = float("nan")
            rl = _root_line(side)
            if rl is not None:
                pr = bevel_lines[side].intersect(rl)
                xr = float(pr[0]) if pr is not None else float("nan")
        elif side in wall_x:
            xt = xr = wall_x[side]
        else:
            # 벽 점 없음(그늘): 판 상면은 모서리 직전 열까지 보이므로 모서리 = 마지막 판 점 + dx/2
            # (다음 내부점이 dx 안에 있으면 그 중간)
            if side == "left":
                nxt = xi[xi > xL_end]
                xt = 0.5 * (xL_end + float(nxt.min())) if nxt.size and float(nxt.min()) - xL_end < dx else xL_end + 0.5 * dx
            else:
                prv = xi[xi < xR_start]
                xt = 0.5 * (xR_start + float(prv.max())) if prv.size and xR_start - float(prv.max()) < dx else xR_start - 0.5 * dx
            xr = xt
        ht = float(pl.h_at(xt)) if np.isfinite(xt) else float("nan")
        return xt, ht, xr

    xtL, htL, xrL = _corner("left")
    xtR, htR, xrR = _corner("right")
    f.left_edge_x_mm, f.right_edge_x_mm = xtL, xtR
    f.top_opening_mm = xtR - xtL
    f.root_gap_edges_mm = xrR - xrL
    f.root_gap_mm = f.root_gap_edges_mm if np.isfinite(f.root_gap_edges_mm) else f.root_gap_floor_mm
    if spec is not None and spec.kind != "square" and float(spec.bevel_angle_deg) > 0.0 and not bevel_lines:
        # 설계에 베벨이 있는데 이 프로파일에서 베벨면을 못 찾았다: 루트 모서리가 판 모서리로 대체돼 갭 = 개구 — 갭 판독 불가 (2026-10-06
        # g3_11: 입사 37° 에서 프로파일 절반이 그랬고 그 '갭 11 mm' 가 끝 판정의 설계 범위와 갭 중앙값을 망쳤다)
        f.root_gap_mm = float("nan")
        f.flags.append("gap_unresolved_no_bevel")
    if np.isfinite(floor_level):
        f.depth_left_mm = lvl_L - floor_level
        f.depth_right_mm = lvl_R - floor_level
        if f.bevel_side in ("left", "right"):
            f.depth_mm = f.depth_left_mm if f.bevel_side == "left" else f.depth_right_mm
        else:
            f.depth_mm = 0.5 * (f.depth_left_mm + f.depth_right_mm)
    # 가림: 바닥 점이 양쪽 루트 모서리에 못 미침
    if floor_x is None:
        f.occluded_floor = True
        if "no_floor" not in f.flags:
            f.flags.append("no_floor")
    else:
        reach = P_["floor_reach_mm"]
        if (np.isfinite(xrL) and floor_x[0] - xrL > reach) or (np.isfinite(xrR) and xrR - floor_x[1] > reach):
            f.occluded_floor = True
            f.flags.append("occluded_floor")
            f.notes.append("floor points do not reach both root edges (near-vertical wall shadow); root_gap_floor is a lower bound")

    # ---- 4b. 직각 벽 전단 k = dx'/dh' (측 무관: 검출된 베벨의 반대쪽 벽; both/unknown/none 은 양쪽 평가 → 점 많은 쪽) ----
    # 평면 벽 ∩ 레이저 평면은 레이저 평면이 판 법선 또는 판 y(가로) 축을 포함하는 한 판 상면에 수직이다. 즉 순수 요(법선 회전)나
    # 순수 피치(가로축 회전)로는 기울지 않고 둘이 함께 있을 때 k = tan(yaw)·sin(pitch) 만큼 기운다 (GROOVE_PROFILE_ANALYSIS §5 H2).
    # 벽 후보점: 판 인라이어가 아니고 x' 가 두 판 사이(판 끝 열 ± dx/2 포함 — 벽은 판 끝 열과 같은 x 에서 아래로 이어지므로
    # 내부 마스크 mI 의 개구간(xL_end, xR_start) 에서는 부동소수 순서에 따라 빠질 수 있다)이며 바닥 대역 위에 있는 점.
    tol_w = 0.5 * dx
    mW = (~mL) & (~mR) & (xp >= xL_end - tol_w) & (xp <= xR_start + tol_w)
    xw_all, hw_all = xp[mW], hp[mW]
    above_w = (hw_all > (floor_line.h_at(xw_all) + floor_band_mm)) if floor_line is not None else np.ones(xw_all.size, dtype=bool)
    groups_w = {"left": above_w & (xw_all < xc_f), "right": above_w & (xw_all >= xc_f)}
    _wall_cands = []
    for side, m in groups_w.items():
        if f.bevel_side in ("left", "right") and side == f.bevel_side:
            continue                                        # 단일 베벨: 베벨면은 벽이 아니다
        n_ = int(m.sum())
        if n_ < max(2, int(wall_min_points)):
            continue
        wx, wh = xw_all[m], hw_all[m]
        ln, inl = (ransac_line(wx, wh, inlier_mm, ransac_iters, rng, min_pts=max(2, int(wall_min_points)))
                   if n_ >= 4 else (None, None))
        if ln is None:
            ln, inl = fit_line_pca(wx, wh), np.ones(n_, dtype=bool)
        n_in = int(inl.sum())
        if n_in < int(wall_min_points) or ln.angle_from_vertical_deg > float(wall_max_lean_deg):
            continue                                        # 점 부족이거나 벽이라기엔 너무 누움(베벨/바닥 잔여)
        d_ = ln.d if ln.d[1] > 0 else -ln.d                 # h' 증가(센서 쪽) 방향으로 정규화 → k 부호 = x' 증가 방향
        _wall_cands.append((n_in, side, float(d_[0] / d_[1]), float(wh[inl].max() - wh[inl].min())))
        if side not in bevel_lines:
            face_pts[f"wall_{side}"] = np.flatnonzero(mW)[np.flatnonzero(m)[inl]]   # 직각 벽 면 (베벨이 아닌 쪽)
    if _wall_cands:
        _wall_cands.sort(key=lambda c: -c[0])
        n_in, side, k_, span_ = _wall_cands[0]
        f.wall_shear_k, f.wall_points_n, f.wall_span_mm, f.wall_side = k_ * ds, n_in, span_ / ds, side   # 센서 평면 단위 (진단 공식)
        if f.bevel_side in ("both", "unknown"):
            f.flags.append("wall_ambiguous")
            f.notes.append(f"wall shear evaluated on the {side} face (bevel side {f.bevel_side}: square wall not identified)")

    # ---- 5b. 바닥(뒷댐재) 가시성: 바닥 직선 인라이어 수·x' 범위 vs 기대 루트갭 ----
    if spec is not None:
        f.floor_expected_mm = float(spec.root_gap_mm)
    elif np.isfinite(f.root_gap_edges_mm):
        f.floor_expected_mm = float(f.root_gap_edges_mm)
    if floor_line is not None:
        fin = np.abs(floor_line.dist(fx, fh)) < inlier_mm
        if fin.any():
            f.floor_span_mm = float(fx[fin].max() - fx[fin].min())
        n_fin = int(fin.sum())
        if np.isfinite(f.floor_expected_mm) and f.floor_expected_mm > 0:
            f.floor_visible = bool(n_fin >= int(floor_min_points) and np.isfinite(f.floor_span_mm)
                                   and f.floor_span_mm >= float(floor_visible_frac) * f.floor_expected_mm)
        else:
            f.notes.append("floor visibility not judged: expected root gap unknown (no spec, no root edges)")
    elif inter.size >= 3:
        f.floor_visible = False                             # 대역은 탐색했으나 바닥 없음 (얕음 = 뒷댐재가 보이지 않음)
    # 센서 프레임 점 (h' 를 센서 단위로 되돌린다)
    def _xzs(xp_, hp_):
        return _xz(xp_, hp_ / ds if hp_ is not None else None, phi)
    f.top_corner_left_xz = _xzs(xtL, htL)
    f.top_corner_right_xz = _xzs(xtR, htR)
    f.root_edge_left_xz = _xzs(xrL, floor_level)
    f.root_edge_right_xz = _xzs(xrR, floor_level)
    if f.top_corner_left_xz and f.top_corner_right_xz:
        c = _xzs(0.5 * (xtL + xtR), 0.5 * (htL + htR))
        f.groove_centre_x_mm, f.groove_centre_z_mm = c
    if f.root_edge_left_xz and f.root_edge_right_xz:
        c = _xzs(0.5 * (xrL + xrR), floor_level)
        f.root_centre_x_mm, f.root_centre_z_mm = c
    elif f.top_corner_left_xz and f.top_corner_right_xz:
        f.root_centre_x_mm, f.root_centre_z_mm = f.groove_centre_x_mm, f.groove_centre_z_mm
    f.ok = not f.insufficient_points and "plates_not_parallel" not in f.flags
    f.faces = _face_rows(x, h, face_pts)

    # ---- 6. 공칭 비교 ----
    if spec is not None:
        _compare_spec(f, spec, qc_targets)
    return f


def _pca_line(x: np.ndarray, h: np.ndarray) -> tuple:
    """fit_line_pca 와 같은 직교 회귀 (닫힌 식) 를 객체 없이 → (중심 x, 중심 h, 방향 cos, 방향 sin (cos ≥ 0), 수직 잔차) — 면 직선은 프로파일당
    여러 번 맞추므로 (패스 처리 시간) 가볍게."""
    n = x.size
    cx, ch = float(x.sum()) / n, float(h.sum()) / n
    dx, dh = x - cx, h - ch
    th = 0.5 * math.atan2(2.0 * float(dx @ dh), float(dx @ dx) - float(dh @ dh))
    c, s = math.cos(th), math.sin(th)
    return cx, ch, c, s, dh * c - dx * s


def _face_rows(x: np.ndarray, h: np.ndarray, face_pts: dict) -> np.ndarray:
    """면 이름 → 정렬된 유효점 인덱스 (센서 프레임 x, h = −z) → (len(FACE_KINDS), len(FACE_COLS)) float32: 면마다 직교 회귀 (PCA) 직선의
    중심 (x, z), 단위 방향 (dx, dz), 잔차 RMS, 점 수. 점이 수치 최소보다 적거나 한 점에 몰린 면은 NaN 행. 직선은 센서 프레임 그대로
    맞춘다 (추출기는 판 정렬·깊이 축 보정 프레임에서 면을 골랐지만 그 변환은 선형이라 면 소속은 같다 — 방향은 원래 단위로)."""
    out = np.full((len(FACE_KINDS), len(FACE_COLS)), np.nan, dtype=np.float32)
    for i, name in enumerate(FACE_KINDS):
        ii = face_pts.get(name)
        if ii is None or np.asarray(ii).size < _FACE_MIN_POINTS:
            continue
        xs, hs = x[ii], h[ii]
        if float(xs.max() - xs.min()) + float(hs.max() - hs.min()) <= 0.0:
            continue
        cx, ch, c, s, r = _pca_line(xs, hs)
        # 면 끝의 이웃 면 점 (바닥 대역 반폭 안의 베벨 발치, 판 연장에 섞인 개선면 시작) 을 k·σ 로 뺀다 (원칙 1: σ = 이 직선 잔차의 강건
        # 표준편차, 하한 센서 양자화 dX(z)/√12) — 합성 10T: 바닥 대역 안 베벨 발치 두 점이 5 mm 바닥 직선을 1.9° 기울였다
        q_k = K_DEFAULT * quant_sigma_mm(-ch)
        for _ in range(_FACE_TRIM_ITERS):
            a = np.abs(r)
            if float(a.max()) <= q_k:
                break                                               # 모든 점이 하한 문턱 안: σ 를 잴 필요 없다 (같은 결과, 패스 처리 시간)
            sig = robust_sigma(r)
            keep = a <= max(K_DEFAULT * sig if np.isfinite(sig) else 0.0, q_k)
            if keep.all() or int(keep.sum()) < _FACE_MIN_POINTS:
                break
            xs, hs = xs[keep], hs[keep]
            cx, ch, c, s, r = _pca_line(xs, hs)
        out[i] = (cx, -ch, c, -s, math.sqrt(float(r @ r) / r.size), xs.size)
    return out


def _compare_spec(f: CrossSectionFeatures, spec: GrooveSpec, qc_targets: Optional[dict]) -> None:
    f.spec_name = spec.name
    f.spec_deltas = {
        "root_gap_mm": f.root_gap_mm - spec.root_gap_mm,
        "bevel_angle_deg": f.bevel_angle_deg - spec.bevel_angle_deg,
        "groove_depth_mm": f.depth_mm - spec.groove_depth_mm,
        "top_opening_mm": f.top_opening_mm - spec.top_opening_mm,
        "hi_lo_mm": f.hi_lo_mm - spec.hi_lo_mm,
        # 측 무관 spec('unknown'/'auto') 이면 비교하지 않음 (None); 왼/오른쪽 spec 은 기존 동작
        "bevel_side_match": (None if spec.side_agnostic or f.bevel_side in ("unknown",)
                             else bool(f.bevel_side == spec.bevel_side)),
    }
    tg = dict(DEFAULT_QC_TARGETS)
    if qc_targets:
        for k, v in qc_targets.items():
            tg[k] = {**tg.get(k, {}), **(v or {})}
    qc = {}
    for key, meas, nom in (("root_gap_mm", f.root_gap_mm, spec.root_gap_mm),
                           ("bevel_angle_deg", f.bevel_angle_deg, spec.bevel_angle_deg),
                           ("groove_depth_mm", f.depth_mm, spec.groove_depth_mm)):
        tol_v = tg.get(key, {}).get("tol")
        tol = float("nan") if tol_v is None else float(tol_v)         # null = 보고만
        ok = bool(abs(meas - nom) <= tol) if (np.isfinite(meas) and np.isfinite(tol)) else None
        qc[key] = {"value": float(meas), "nominal": float(nom), "tol": tol, "pass": ok}
    hl = tg.get("hi_lo_mm", {})
    tmax = float(hl.get("target_max", 1.0))
    warn = float(hl.get("warn_above", HI_LO_AMBIGUOUS_MM))
    if np.isfinite(f.hi_lo_mm):
        qc["hi_lo_mm"] = {"value": float(f.hi_lo_mm), "target_max": tmax, "warn_above": warn,
                          "pass": bool(abs(f.hi_lo_mm) <= tmax), "flag": bool(abs(f.hi_lo_mm) > warn)}
    f.qc = qc


# --------------------------------------------------------------------------------------
# 스캔 전체
# --------------------------------------------------------------------------------------
@dataclass
class SeamScanFeatures:
    """프로파일별 특징 배열 + 심 폴리라인 (base 프레임, m)."""
    n_profiles: int = 0
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    s_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ok: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    groove_centre_x_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    groove_centre_z_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    root_centre_x_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    root_centre_z_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    top_opening_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    root_gap_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bevel_angle_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    depth_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hi_lo_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tilt_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # 정렬 진단 (프로파일별; NaN = 미평가)
    wall_shear_k: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wall_points_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    floor_visible: np.ndarray = field(default_factory=lambda: np.zeros(0))          # 1.0 / 0.0 / NaN(미평가)
    floor_span_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    root_edges_resolved: np.ndarray = field(default_factory=lambda: np.zeros(0))    # 1.0 = 양쪽 루트 모서리 (x, z) 모두 결정 / 0.0
    in_cruise: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # result.in_cruise[idx] (없으면 전부 True)
    polyline_base_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # ok 프로파일의 심 중심 (base, m)
    polyline_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # 면 직선의 base 3D (프로파일별 실측 자세 T_base_opt; 단면 법선 scan_core.cross_sections): 면 위 점 (M, F, 3) [m], 레이저 평면 안 면
    # 방향 (M, F, 3) 단위, 맞춤 (M, F, 2) = (잔차 RMS mm, 점 수), 광학 원점 (M, 3) [m] (법선을 센서 쪽으로). F = len(FACE_KINDS), 없는 면 NaN
    face_point_base: np.ndarray = field(default_factory=lambda: np.zeros((0, len(FACE_KINDS), 3)))
    face_dir_base: np.ndarray = field(default_factory=lambda: np.zeros((0, len(FACE_KINDS), 3)))
    face_fit: np.ndarray = field(default_factory=lambda: np.zeros((0, len(FACE_KINDS), 2)))
    opt_origin_base: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    noise_window_m: float = NOISE_WINDOW_M      # 측정 잡음의 이동 중앙값 길이 = 작업의 용접 경로 단면 창 (fit_pass 가 job weld_plan 으로 설정)
    centre: str = "root"
    straightness_rms_mm: float = float("nan")
    polyline_dir: Optional[np.ndarray] = None
    features: Optional[list] = None
    notes: list = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return int(self.ok.sum())

    def _align_selection(self) -> np.ndarray:
        """정렬 진단 중앙값에 쓰는 프로파일: ok ∧ in_cruise (크루즈 프로파일이 없으면 ok 전부)."""
        ok = np.asarray(self.ok, dtype=bool)
        cr = np.asarray(self.in_cruise, dtype=bool)
        if cr.shape == ok.shape and (ok & cr).any():
            return ok & cr
        return ok

    def medians(self) -> dict:
        out = {}
        for k in ("top_opening_mm", "root_gap_mm", "bevel_angle_deg", "depth_mm", "hi_lo_mm", "tilt_deg"):
            v = getattr(self, k)
            v = v[np.isfinite(v)]
            out[k] = float(np.median(v)) if v.size else float("nan")
        # 정렬 진단: ok ∧ cruise 프로파일의 nanmedian / 비율 (NaN = 어느 프로파일에서도 평가되지 않음)
        sel = self._align_selection()
        ks = np.asarray(self.wall_shear_k, dtype=float)
        ks = ks[sel] if ks.shape == sel.shape else ks
        ks = ks[np.isfinite(ks)]
        out["wall_shear_k"] = float(np.median(ks)) if ks.size else float("nan")
        fv = np.asarray(self.floor_visible, dtype=float)
        fv = fv[sel] if fv.shape == sel.shape else fv
        fv = fv[np.isfinite(fv)]
        out["floor_visible_fraction"] = float(np.mean(fv)) if fv.size else float("nan")
        wn = np.asarray(self.wall_points_n, dtype=float)
        wn = wn[sel] if wn.shape == sel.shape else wn
        out["wall_points_median"] = float(np.median(wn)) if wn.size else 0.0
        # 충실도 게이트 (형상 식별): 양쪽 루트 모서리 결정 비율, 심 방향 일관성 (갭·깊이의 강건 표준편차 = 1.4826·MAD) — ok ∧ cruise
        re_ = np.asarray(self.root_edges_resolved, dtype=float)
        re_ = re_[sel] if re_.shape == sel.shape else re_
        re_ = re_[np.isfinite(re_)]
        out["root_edges_fraction"] = float(np.mean(re_)) if re_.size else float("nan")
        for k in ("root_gap_mm", "depth_mm"):
            v = np.asarray(getattr(self, k), dtype=float)
            v = v[sel] if v.shape == sel.shape else v
            v = v[np.isfinite(v)]
            out[f"{k[:-3]}_robust_std_mm"] = (float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size >= 3
                                              else float("nan"))
        # 측정 잡음 (2026-10-02): 심 방향 실제 변화 (시공 오차 — 갭이 끝으로 벌어짐 등) 를 NOISE_WINDOW_M 이동 중앙값으로 빼고 남은
        # 프로파일 간 흔들림의 강건 표준편차. 시편의 변화는 along_seam_bins 로 따로 보고한다.
        for k in ("root_gap_mm", "depth_mm", "top_opening_mm", "hi_lo_mm"):
            out[f"{k[:-3]}_noise_mm"] = self._noise(k)
        return out

    def _ok_series(self, key: str, s=None) -> tuple:
        """ok 프로파일의 (s, 값) — s 오름차순 (s = 패스 진행 좌표 s_m, 또는 주어진 프로파일별 좌표)."""
        v = np.asarray(getattr(self, key), dtype=float)
        s_ = np.asarray(self.s_m if s is None else s, dtype=float)
        ok = np.asarray(self.ok, dtype=bool)
        if v.shape != ok.shape or s_.shape != ok.shape:
            return np.zeros(0), np.zeros(0)
        m = ok & np.isfinite(v) & np.isfinite(s_)
        o = np.argsort(s_[m], kind="stable")
        return s_[m][o], v[m][o]

    def _noise(self, key: str, window_m: Optional[float] = None) -> float:
        s_, v = self._ok_series(key)
        w = getattr(self, "noise_window_m", NOISE_WINDOW_M) if window_m is None else float(window_m)
        return detrended_noise(s_, v, w)

    def along_seam_bins(self, bin_m: float = 0.020, keys=("root_gap_mm", "depth_mm", "top_opening_mm", "hi_lo_mm",
                                                           "bevel_angle_deg"), s=None, origin: Optional[float] = None) -> list:
        """심 방향 구간 (좌표 s, bin_m 간격) 별 ok 프로파일의 중앙값 — 시편의 실제 치수 변화 (시공 오차) 보고용. s 는 프로파일별
        좌표 (M,) [m] (기본: 패스 진행 좌표 s_m; auto-teach 는 정련 심 직선 좌표 (P_root − c)·u 를 준다). origin 을 주면 구간 경계를
        origin + j·bin_m 에 맞춘다 (auto-teach: 식별된 판 끝).
        반환 [{s_from_m, s_to_m, n, <key>: 중앙값 …}] (s 오름차순; 프로파일이 용접 경로 단면의 최소 수 (WeldProcess.min_profiles) 보다
        적은 구간은 뺀다)."""
        from .weld_plan import WeldProcess
        n_min = int(WeldProcess().min_profiles)
        s_all, _ = self._ok_series(keys[0], s)
        if s_all.size == 0:
            return []
        lo0 = float(s_all.min())
        if origin is not None and np.isfinite(origin):
            lo0 = float(origin) + np.floor((lo0 - float(origin)) / bin_m + 1e-9) * bin_m
        edges = np.arange(lo0, float(s_all.max()) + bin_m, bin_m)
        out = []
        series = {k: self._ok_series(k, s) for k in keys}
        for a, b in zip(edges[:-1], edges[1:]):
            row = {"s_from_m": float(a), "s_to_m": float(b)}
            n0 = 0
            for k, (sk, vk) in series.items():
                m = (sk >= a) & (sk < b)
                row[k] = float(np.median(vk[m])) if m.sum() >= n_min else float("nan")
                n0 = max(n0, int(m.sum()))
            row["n"] = n0
            if n0 >= n_min:
                out.append(row)
        return out

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("n_profiles", "indices", "s_m", "ok", "groove_centre_x_mm", "groove_centre_z_mm",
                                           "root_centre_x_mm", "root_centre_z_mm", "top_opening_mm", "root_gap_mm",
                                           "bevel_angle_deg", "depth_mm", "hi_lo_mm", "tilt_deg", "wall_shear_k", "wall_points_n",
                                           "floor_visible", "floor_span_mm", "root_edges_resolved", "in_cruise", "polyline_base_m",
                                           "polyline_indices", "centre", "straightness_rms_mm", "polyline_dir", "notes")}
        d["n_ok"] = self.n_ok
        d["medians"] = self.medians()
        return _plain(d)

    def summary(self) -> str:
        m = self.medians()
        k_txt = f"{m['wall_shear_k']:+.4f}" if np.isfinite(m["wall_shear_k"]) else "n/a"
        fv_txt = f"{m['floor_visible_fraction']:.2f}" if np.isfinite(m["floor_visible_fraction"]) else "n/a"
        return (f"seam features: {self.n_ok}/{self.n_profiles} profiles ok; median top opening {m['top_opening_mm']:.2f} mm, "
                f"root gap {m['root_gap_mm']:.2f} mm, bevel {m['bevel_angle_deg']:.1f} deg, depth {m['depth_mm']:.2f} mm, "
                f"hi-lo {m['hi_lo_mm']:+.2f} mm, tilt {m['tilt_deg']:+.2f} deg; polyline ({self.centre}) "
                f"{self.polyline_base_m.shape[0]} pts straightness rms {self.straightness_rms_mm:.3f} mm; "
                f"align: wall shear k {k_txt} (median wall pts {m['wall_points_median']:.0f}), floor visible {fv_txt}")


def detrended_noise(s, v, window_m: float) -> float:
    """심 방향 값 v (좌표 s, 오름차순) 의 프로파일 간 잡음: window_m 이동 중앙값 (심 방향 실제 변화 — 시공 오차, 휨) 을 뺀 나머지의
    강건 표준편차 (1.4826·MAD). 점이 10 개 미만이면 NaN."""
    s_ = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.isfinite(s_) & np.isfinite(v)
    s_, v = s_[m], v[m]
    if v.size < 10:
        return float("nan")
    o = np.argsort(s_, kind="stable")
    s_, v = s_[o], v[o]
    w = float(window_m)
    lo = np.searchsorted(s_, s_ - 0.5 * w, side="left")
    hi = np.searchsorted(s_, s_ + 0.5 * w, side="right")
    trend = _window_medians(v, lo, hi)
    r = v - trend
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _window_medians(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """창 v[lo_i:hi_i] 마다의 중앙값 — np.median 과 같은 값 (홀수 = 가운데 값, 짝수 = 가운데 두 값의 평균 (a + b) / 2). 창이 앞으로만
    움직이면 (lo·hi 비감소, 빈 창 없음 — 정렬된 s 의 ± 반창) 정렬 목록에 넣고 빼며 한 번에 (2026-10-08: 창마다 np.median 을 부르던 것이
    패스당 ~0.2 s); 아니면 창마다 np.median."""
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    if lo.size == 0 or not (np.all(hi > lo) and np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0)):
        return np.array([np.median(v[a:b]) for a, b in zip(lo, hi)])
    vals = np.asarray(v, dtype=float).tolist()
    win: list = []
    a = b = 0
    out = np.empty(lo.size)
    for i, (l_, h_) in enumerate(zip(lo.tolist(), hi.tolist())):
        while b < h_:
            bisect.insort(win, vals[b])
            b += 1
        while a < l_:
            del win[bisect.bisect_left(win, vals[a])]
            a += 1
        n_ = len(win)
        # np.median 은 가운데 값(들)의 np.mean (합은 0.0 에서 시작) — 같은 식이라 −0.0 까지 같다
        out[i] = (0.0 + win[n_ // 2]) if n_ % 2 else (0.0 + win[n_ // 2 - 1] + win[n_ // 2]) / 2
    return out


# ---- 프로파일별 특징의 프로세스 병렬 (2026-10-03) ------------------------------------------------------------------------
# 실기 at11: 패스 후 특징 추출이 ROS 서버 안에서 ~100 s (같은 계산을 따로 돌리면 9 s) — 서버 프로세스의 ROS 콜백 스레드들과 GIL 을 나눠서.
# 프로파일마다 독립 (RANSAC 시드도 프로파일마다 0 부터) 이라 결과는 순차와 똑같다. 'spawn' 프로세스 (ROS·스레드 상태를 물려받지 않음) 를
# 처음 쓸 때 만들고 계속 쓴다. 풀은 scan_core.proc_pool.OwnedPool (주인은 자식만 쓰는 파이프 끝을 들지 않고 자식의 죽음을 프로세스 상태로 안다):
# 자식이 끝난 풀은 다음에 쓸 때 새로 만든다 (_feature_pool) — 그 풀을 쓴 모든 사용처 (특징·용접 경로·도달성) 가 같은 규칙.
_POOL: Optional[OwnedPool] = None
_POOL_N = 0
_POOL_LOCK = threading.Lock()
# 마지막 특징 추출이 어떻게 돌았나 (패스 처리 시간 기록용, 2026-10-05: 실기 서버의 패스 처리가 오프라인보다 ~3 배 느려 풀이 조용히 순차로
# 물러나는지 확인) — mode 'pool' | 'executor' | 'sequential' (설정·프로파일 수) | 'fallback' (풀이 돌려주지 못한 덩어리를 이 프로세스에서),
# error = 그 오류 (+ 이 프로세스에서 계산한 덩어리 수)
FEATURE_RUN: dict = {"mode": None, "workers": 0, "error": None, "note": None}
PARALLEL_MIN_PROFILES = 64


def _feature_pool(n: int) -> OwnedPool:
    global _POOL, _POOL_N
    with _POOL_LOCK:
        if _POOL is None or _POOL_N != n or _POOL.child_died():     # 자식이 끝난 풀 (프로세스 상태) = 깨진 풀
            if _POOL is not None:
                _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = OwnedPool(max_workers=n, mp_context=multiprocessing.get_context("spawn"))
            _POOL_N = n
        return _POOL


def worker_pool(workers="auto") -> OwnedPool:
    """같은 spawn 프로세스 풀 (특징 추출과 공유) — ROS 서버 안의 무거운 순수 계산을 GIL 경합 없이 (용접 경로 도달성 확인 등)."""
    return _feature_pool(resolve_workers(workers))


def shutdown_feature_pool(pool: Optional[OwnedPool] = None, *, kill: bool = False) -> None:
    """풀을 내린다 (pool: 그 풀일 때만 — 기본은 지금 것). 다음에 쓸 때 새로 만든다. kill: 살았지만 한계 안에 답이 없는 풀 (멈춤 —
    proc_pool.wait_within) 의 자식을 끝낸다 (그래야 정리가 그 자식을 기다리지 않는다)."""
    global _POOL, _POOL_N
    with _POOL_LOCK:
        if _POOL is not None and (pool is None or _POOL is pool):
            if kill:
                _POOL.kill_children()
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL, _POOL_N = None, 0


atexit.register(shutdown_feature_pool)


def _features_chunk(X, Z, spec, kw, DS=None, seeds=None) -> list:
    if seeds is not None:
        out = []
        for j in range(X.shape[0]):
            kw_j = kw if DS is None else {**kw, "depth_scale": float(DS[j])}
            out.append(extract_cross_section_features(X[j], Z[j], spec, **kw_j) if seeds[j] is None
                       else extract_cross_section_features(X[j], Z[j], spec, plate_seed=seeds[j], **kw_j))
        return out
    if DS is None:
        return [extract_cross_section_features(X[j], Z[j], spec, **kw) for j in range(X.shape[0])]
    return [extract_cross_section_features(X[j], Z[j], spec, **{**kw, "depth_scale": float(DS[j])}) for j in range(X.shape[0])]


def cross_section_plate_seed(x_mm, z_mm, spec: Optional[GrooveSpec] = None, **kw) -> Optional[PlateSeed]:
    """프로파일 1개 → 판 성장 결과 (PlateSeed; extract_cross_section_features 와 같은 인자, depth_scale 은 쓰지 않는다). 판 성장 전에 끝나는
    프로파일 (점 부족) 은 None — 그 프로파일은 나중에 처음부터 계산해도 싸다."""
    kw.pop("depth_scale", None)
    r = extract_cross_section_features(x_mm, z_mm, spec, _plates_only=True, **kw)
    return r if isinstance(r, PlateSeed) else None


def plate_seeds_chunk(X, Z, spec, kw) -> list:
    """프로파일 묶음 (행 = 프로파일 — 2 차원 배열 또는 1 차원 배열 목록, 무효 점 NaN) → [PlateSeed 또는 None] (작업 프로세스 풀에서;
    scan_l0.post_scan 스트림)."""
    return [cross_section_plate_seed(X[j], Z[j], spec, **kw) for j in range(len(X))]


DESIGN_BAND_FRAC = DESIGN_HALF  # 정지 국소화의 설계 일치 띠 = 공칭 ±50 % (scan_core.measure 원칙 2 — 판 끝 식별·게이트와 같은 원천)


def matches_design(f: "CrossSectionFeatures", spec: Optional[GrooveSpec], band: float = DESIGN_BAND_FRAC) -> bool:
    """그루브를 찾았고 개구·깊이가 설계 ± band 안인가 (spec 없으면 True)."""
    if spec is None:
        return True
    if not f.groove_found:
        return False
    for val, nom in ((f.top_opening_mm, spec.top_opening_mm), (f.depth_mm, spec.groove_depth_mm)):
        if nom > 0.0 and not (np.isfinite(val) and abs(val - nom) <= band * nom):
            return False
    return True


def notch_candidates(x_mm, z_mm, spec: GrooveSpec, *, iters: int = 4, qc_targets: Optional[dict] = None) -> list:
    """프로파일 → 판 아래로 파인 자리 후보 [(중심 x, 폭, 최대 깊이)] (깊이 큰 순). 판 직선 = 반복 절사 최소제곱 (판이 점의 대부분;
    남기는 점 = 직선에서 max(판 소속 띠, k·강건 σ) 안), 파임 = 그 직선보다 설계 깊이의 절반 넘게 깊은 (센서 z 가 큰) 연속 구간 (판 상면 ↔
    그루브 바닥의 중간, 원칙 2; 결측 이음 간격까지 끊김 허용). 문턱은 extractor_params 와 같은 원천 (설계·작업 허용치·측정 열 간격)."""
    x = np.asarray(x_mm, dtype=float)
    z = np.asarray(z_mm, dtype=float)
    m = np.isfinite(x) & np.isfinite(z)
    xs0 = np.sort(x[m])
    dxs = np.diff(xs0)
    dxs = dxs[dxs > 0]
    P_ = extractor_params(spec, qc_targets, float(np.median(dxs)) if dxs.size else None)
    if m.sum() < P_["min_points"]:
        return []
    xs, zs = x[m], z[m]
    keep = np.ones(xs.size, dtype=bool)
    a = b = 0.0
    for _ in range(iters):
        if keep.sum() < 3:
            return []
        a, b = np.polyfit(xs[keep], zs[keep], 1)
        r = zs - (a * xs + b)
        thr = max(P_["inlier_mm"], K_DEFAULT * MAD_TO_SIGMA * float(np.median(np.abs(r[keep]))))
        keep = np.abs(r) <= thr
    r = zs - (a * xs + b)
    deep = r > DESIGN_HALF * float(spec.groove_depth_mm)
    order = np.argsort(xs)
    xs, r, deep = xs[order], r[order], deep[order]
    out, run = [], []
    for xi, ri, di in zip(xs, r, deep):
        if di and run and xi - run[-1][0] > P_["merge_gap_mm"]:
            out.append(run)
            run = []
        if di:
            run.append((xi, ri))
    if run:
        out.append(run)
    cands = []
    for rn in out:
        xr = np.array([q[0] for q in rn])
        cands.append((float(0.5 * (xr.min() + xr.max())), float(xr.max() - xr.min()), float(max(q[1] for q in rn))))
    return sorted(cands, key=lambda c: -c[2])


def _full_design_match(f: "CrossSectionFeatures", spec: GrooveSpec, band: float) -> bool:
    """창 재추출의 채택 조건 (첫 추출보다 엄격): 개구·깊이 ± band 에 더해 루트갭을 읽었고 설계 ±50 % 안 (설계 갭이 0 이면 판정 안 함),
    설계에 베벨이 있으면 베벨면을 찾았고 설계 ±50 % 안 (원칙 2, measure.design_match) — 갭·베벨 없이 개구만 맞는 그림자·이물 사이
    (at21 늘어진 케이블, at17 E 탐색) 는 받지 않는다."""
    if not matches_design(f, spec, band):
        return False
    g0 = float(getattr(spec, "root_gap_mm", 0.0) or 0.0)
    if not (np.isfinite(f.root_gap_mm) and design_match(f.root_gap_mm, g0)):
        return False
    b0 = float(getattr(spec, "bevel_angle_deg", 0.0) or 0.0)
    if getattr(spec, "kind", "square") != "square" and b0 > 0.0:
        if not (np.isfinite(f.bevel_angle_deg) and design_match(f.bevel_angle_deg, b0)):
            return False
    return True


def extract_on_design_window(x_mm, z_mm, spec: GrooveSpec, *, max_candidates: int = 3, band: float = DESIGN_BAND_FRAC, **kw):
    """첫 추출이 설계와 맞지 않을 때 (정지 국소화): 파인 자리 후보마다 그 둘레 창 (± 2.5 × 설계 개구 = 개구 반 + 그루브 쪽 국소 판 직선
    (개구) + 판 최소 길이 (개구 반) + 판 끝 허용 (개구 반) — extractor_params 의 길이들) 만으로 다시 추출해
    설계 (개구·깊이 ± band, 갭·베벨: _full_design_match) 에 맞는 후보가 **하나뿐일 때만** 그것 → (CrossSectionFeatures 또는 None, 창 [x0, x1]
    또는 None). 둘 이상이면 모호 → None (예전처럼 설계 불일치로 멈춘다).
    2026-10-06 g3_14 E: 같은 자세·같은 프로파일이 g3_13 에서는 맞게 읽혔는데, 판 표면의 ~0.5 mm 물결이 왼쪽 판을 x −88 mm 에서 둘로
    나눠 그 사이 전체 (개구 118.5 mm) 를 그루브로 읽었다 — 실제 그루브 (x +19..+31) 둘레 창에서는 개구 11.4 mm 로 맞게 읽힌다."""
    x = np.asarray(x_mm, dtype=float)
    z = np.asarray(z_mm, dtype=float)
    W = 2.5 * float(spec.top_opening_mm)
    ok = []
    for xc, _w, _d in notch_candidates(x, z, spec, qc_targets=kw.get("qc_targets"))[:max_candidates]:
        win = (x >= xc - W) & (x <= xc + W)
        f = extract_cross_section_features(np.where(win, x, np.nan), np.where(win, z, np.nan), spec, **kw)
        if not _full_design_match(f, spec, band):
            continue
        if any(abs(f.root_centre_x_mm - q[0].root_centre_x_mm) < 0.5 * float(spec.top_opening_mm) for q in ok):
            continue                                              # 겹친 창이 같은 그루브를 다시 읽음
        ok.append((f, (float(xc - W), float(xc + W))))
    return ok[0] if len(ok) == 1 else (None, None)


def depth_scales(T_base_opt: np.ndarray, plate_normal_base) -> Optional[np.ndarray]:
    """프로파일 자세 (N,4,4) + 판 법선 (base) → depth_scale (N,) = √(1 − (n·Y_opt)²) (extract_cross_section_features). 법선 없으면 None."""
    if plate_normal_base is None:
        return None
    n = np.asarray(plate_normal_base, dtype=float).reshape(3)
    nn = float(np.linalg.norm(n))
    if not (nn > 0 and np.isfinite(nn)):
        return None
    Y = np.asarray(T_base_opt, dtype=float)[:, :3, 1]
    c = np.clip(np.abs(Y @ (n / nn)), 0.0, 1.0)
    return np.sqrt(1.0 - c * c)


def resolve_workers(workers) -> int:
    """설정값 → 작업 프로세스 수: None/'auto' = min(8, CPU − 2), 0·1 = 같은 프로세스에서 순차."""
    if workers is None or workers == "auto":
        return max(1, min(8, (os.cpu_count() or 2) - 2))
    return max(1, int(workers))


def _features_for(X, Z, spec, kw, workers: int, executor=None, DS=None, seeds=None) -> list:
    M = X.shape[0]

    def _ds(a, b):
        return None if DS is None else DS[a:b]

    def _sd(a, b):
        return None if seeds is None else seeds[a:b]
    n = resolve_workers(workers)
    if M < PARALLEL_MIN_PROFILES or (executor is None and n <= 1):
        FEATURE_RUN.update(mode="sequential", workers=1, error=None, note=None)
        return _features_chunk(X, Z, spec, kw, DS, seeds)
    # 프로세스 병렬: 호출자가 준 실행기 (패스 뒤 처리 작업 프로세스 안의 fork 풀, scan_l0.post_scan — 그 주인이 닫는다) 또는 이 모듈의 spawn 풀.
    # 덩어리마다 독립이라 (같은 계산·같은 순서) 풀이 돌려주지 못한 덩어리 — 죽은 자식 (OwnedPool: 그 풀의 Future 는 바로 BrokenProcessPool),
    # 그 밖의 풀 오류, 한계 (proc_pool.RESULT_TIMEOUT_S, 전체에 한 번) 안에 답이 없는 멈춘 풀 (그 자식은 끝낸다) — 는 이 프로세스에서 같은
    # 계산으로: 결과는 순차와 같고 패스를 잃지 않는다
    bounds = np.linspace(0, M, min(M, 4 * n) + 1).astype(int)
    spans = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    pool, err, futs = executor, None, []
    try:
        if pool is None:
            pool = _feature_pool(n)
        for a, b in spans:
            futs.append(pool.submit(_features_chunk, X[a:b], Z[a:b], spec, kw, _ds(a, b), _sd(a, b)))
    except Exception as e:  # noqa: BLE001 - the chunks not submitted are computed here
        err = e
    done, stalled = _proc_pool.wait_within(futs, _proc_pool.RESULT_TIMEOUT_S)
    if stalled:                                         # 살았지만 답이 없는 풀 — 그 덩어리는 여기서 (어떤 실행기든 그 까닭을 기록한다)
        err = err if err is not None else TimeoutError(f"no answer from the process pool within {_proc_pool.RESULT_TIMEOUT_S:.0f} s "
                                                       f"(alive but stalled)")
        if isinstance(pool, OwnedPool):
            pool.kill_children()                        # 주인의 정리가 그 자식을 기다리지 않는다
    out: list = []
    n_here = 0
    for i, (a, b) in enumerate(spans):
        if i < len(futs) and futs[i] in done:
            e = futs[i].exception()
            if e is None:
                out.extend(futs[i].result())
                continue
            err = err if err is not None else e
        out.extend(_features_chunk(X[a:b], Z[a:b], spec, kw, _ds(a, b), _sd(a, b)))
        n_here += 1
    if executor is None and n_here and isinstance(pool, OwnedPool) and (stalled or pool.child_died()):
        shutdown_feature_pool(pool)                     # 이 모듈이 주인인 깨진·멈춘 풀 — 나머지 자식을 끝내고 다음에 새로
    note = pool.rule_note if isinstance(pool, OwnedPool) else None   # 규칙 없이 도는 풀 (표준 내부가 다름 — proc_pool) 이면 그 까닭
    if n_here:
        FEATURE_RUN.update(mode="fallback", workers=n, note=note,
                           error=f"{type(err).__name__}: {err}"[:240] + f" ({n_here}/{len(spans)} chunks computed in this process)")
    else:
        FEATURE_RUN.update(mode="executor" if executor is not None else "pool", workers=n, error=None, note=note)
    return out


def _set_face_geometry(out: SeamScanFeatures, flist: list, T: np.ndarray) -> None:
    """프로파일별 면 직선 (센서 프레임, CrossSectionFeatures.faces) → base 3D: 점 P = T·[x, 0, z], 방향 d = R·[dx, 0, dz] (각 프로파일의
    실측 자세 T_base_opt — 3G·기울인 입사·피치에서도 그 자세의 레이저 평면 그대로). 벡터 연산 (패스당 수 ms)."""
    M, F = T.shape[0], len(FACE_KINDS)
    A = np.full((M, F, len(FACE_COLS)), np.nan)
    for j, f in enumerate(flist):
        if f.faces is not None:
            A[j] = f.faces
    R, o = T[:, :3, :3], T[:, :3, 3]
    X, Z = R[:, None, :, 0], R[:, None, :, 2]                               # (M, 1, 3) 광학 x·z 축
    out.face_point_base = o[:, None, :] + (A[..., 0:1] * X + A[..., 1:2] * Z) * 1e-3
    out.face_dir_base = A[..., 2:3] * X + A[..., 3:4] * Z
    out.face_fit = A[..., 4:6].copy()
    out.opt_origin_base = o.copy()


def extract_seam_features_along_scan(result, spec: Optional[GrooveSpec] = None, *, stride: int = 1, only_cruise: bool = False,
                                     centre: str = "root", keep_features: bool = False, workers=1, executor=None,
                                     plate_normal_base=None, plate_seeds=None, **kw) -> SeamScanFeatures:
    """ScanResult (x_mm, z_mm, valid, T_base_opt, s_m, in_cruise) → 프로파일별 특징 + 심 폴리라인.

    폴리라인 점 = T_base_opt[i]·[xc·1e-3, 0, zc·1e-3, 1] (xc, zc = 루트갭 중심 ('root', 기본) 또는 상부 개구 중심 ('top')).
    직진도 = 폴리라인 점의 최적 직선(PCA) 대비 수직 거리 RMS [mm].
    workers: 프로파일별 특징을 나눠 계산할 프로세스 수 (1 = 순차, 'auto'/None = min(8, CPU − 2)); 결과는 같다.
    plate_normal_base: 판 법선 사전값 → 프로파일마다 depth_scale (레이저 평면 기울기 보정, depth_scales). None = 보정 없음.
    plate_seeds: result 행마다 미리 계산한 판 성장 (PlateSeed 또는 None, 길이 N; scan_l0.post_scan 스트림) — 그 행의 (x, z)·spec·kw 로
    계산한 것만 넣는다 (호출자가 확인). 결과는 없이 계산한 것과 같다.
    """
    if centre not in ("root", "top"):
        raise ValueError("centre must be 'root' or 'top'")
    N = int(result.x_mm.shape[0])
    idx = np.arange(0, N, max(1, int(stride)), dtype=np.int64)
    if only_cruise and hasattr(result, "in_cruise"):
        idx = idx[np.asarray(result.in_cruise, dtype=bool)[idx]]
    out = SeamScanFeatures(n_profiles=int(idx.size), indices=idx, centre=centre)
    M = idx.size
    arrs = {k: np.full(M, np.nan) for k in ("s_m", "groove_centre_x_mm", "groove_centre_z_mm", "root_centre_x_mm",
                                             "root_centre_z_mm", "top_opening_mm", "root_gap_mm", "bevel_angle_deg",
                                             "depth_mm", "hi_lo_mm", "tilt_deg", "wall_shear_k", "floor_visible", "floor_span_mm",
                                             "root_edges_resolved")}
    wall_n = np.zeros(M, dtype=np.int64)
    cr_all = getattr(result, "in_cruise", None)
    out.in_cruise = (np.asarray(cr_all, dtype=bool)[idx] if cr_all is not None and np.asarray(cr_all).shape[0] >= N
                     else np.ones(M, dtype=bool))
    ok = np.zeros(M, dtype=bool)
    feats = [] if keep_features else None
    s_all = np.asarray(getattr(result, "s_m", np.full(N, np.nan)), dtype=float)
    valid = np.asarray(result.valid, dtype=bool) if getattr(result, "valid", None) is not None else None
    pts = []
    pidx = []
    X = np.asarray(result.x_mm, dtype=float)[idx]
    Z = np.asarray(result.z_mm, dtype=float)[idx]
    if valid is not None:
        X = np.where(valid[idx], X, np.nan)
        Z = np.where(valid[idx], Z, np.nan)
    T_all = getattr(result, "T_base_opt", None)
    DS = depth_scales(np.asarray(T_all, dtype=float)[idx], plate_normal_base) if (T_all is not None and plate_normal_base is not None) else None
    SD = None
    if plate_seeds is not None and len(plate_seeds) == N and any(sd is not None for sd in plate_seeds):
        SD = [plate_seeds[int(i)] for i in idx]
    flist = _features_for(X, Z, spec, kw, workers, executor, DS, SD)
    for j, i in enumerate(idx):
        f = flist[j]
        if feats is not None:
            feats.append(f)
        arrs["s_m"][j] = s_all[i] if i < s_all.size else np.nan
        for k in ("groove_centre_x_mm", "groove_centre_z_mm", "root_centre_x_mm", "root_centre_z_mm", "top_opening_mm",
                  "root_gap_mm", "bevel_angle_deg", "depth_mm", "hi_lo_mm", "tilt_deg", "wall_shear_k", "floor_span_mm"):
            arrs[k][j] = getattr(f, k)
        arrs["floor_visible"][j] = np.nan if f.floor_visible is None else float(bool(f.floor_visible))
        arrs["root_edges_resolved"][j] = float(bool(f.root_edge_left_xz and f.root_edge_right_xz)) if f.groove_found else np.nan
        wall_n[j] = int(f.wall_points_n)
        ok[j] = bool(f.ok and f.groove_found)
        xc, zc = (f.root_centre_x_mm, f.root_centre_z_mm) if centre == "root" else (f.groove_centre_x_mm, f.groove_centre_z_mm)
        if ok[j] and np.isfinite(xc) and np.isfinite(zc):
            T = np.asarray(result.T_base_opt[i], dtype=float)
            p = T @ np.array([xc * 1e-3, 0.0, zc * 1e-3, 1.0])
            pts.append(p[:3])
            pidx.append(int(i))
    for k, v in arrs.items():
        setattr(out, k, v)
    out.wall_points_n = wall_n
    out.ok = ok
    out.features = feats
    if T_all is not None and M:
        _set_face_geometry(out, flist, np.asarray(T_all, dtype=float)[idx])
    if pts:
        P = np.stack(pts)
        out.polyline_base_m = P
        out.polyline_indices = np.asarray(pidx, dtype=np.int64)
        if P.shape[0] >= 2:
            c = P.mean(axis=0)
            _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
            d = Vt[0]
            resid = (P - c) - np.outer((P - c) @ d, d)
            out.straightness_rms_mm = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))) * 1e3
            out.polyline_dir = d
    else:
        out.notes.append("no profile yielded a groove centre")
    return out
