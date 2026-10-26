"""1G 프로파일 모형 (G1_CORE_SPEC §3, WP1): 설계 단면 템플릿의 강건 맞춤 + 그루브/비그루브 증거.

한 프로파일 (센서 레이저 평면의 x, z [mm]) 에 대해:

* **광선 (RayFan, §3.1):** 열 간격 dX(z) = a + b·z (`validity.dx_linear_coeffs`, 센서 사양 knot 0.082@250 / 0.155@525) →
  모든 열 광선은 가상 발광점 o = (0, z0), z0 = −a/b 를 지난다. 측정점 p_j 의 광선 방향 d_j = unit(p_j − o), 거리 ρ_j = ‖p_j − o‖,
  보정 z 창을 광선이 지나는 길이 W_j (균일 이상치 밀도 1/W_j, [L§1 Torr & Zisserman]).
* **템플릿 (Template, §3.2):** 설계 단면 (single_bevel / single_v / square, 뒷댐재) 을 레이저 평면이 자른 겉보기 다각선.
  θ = (x_R, z_R, α, G, β, rf, h_L, h_R, γ_L, γ_R) — `g1_types.THETA_NAMES`. L = 센서 x 쪽 `bevel_x_sign` 의 판 (외면개선의 베벨 판),
  R = 다른 판. 정규 틀 (ξ, η): ξ 는 L 판 → R 판, η 는 뒷댐재 위 높이 (센서 쪽). 센서 점 = R + m·ξ·e1 + η·e2, m = −bevel_x_sign,
  e1 = (cos α, sin α), e2 = (sin α, −cos α).
* **보임 (가림):** 광선 j 의 예측 = o 에서 나간 광선이 다각선과 처음 만나는 면 (벡터화 광선 투사). 그 면 뒤에 가린 면은 그 광선에
  보이지 않는다 (삼각측량 기하의 자기 가림). 만나는 면이 없으면 직접 반환 예측 없음 → 그 열은 이상치 성분만.
* **우도 (§3.3):** 열 j 의 잔차 = 측정점에서 맞은 면 직선까지의 수직 거리 r_j (Ahn 2004 직교 거리). 광선 거리 ρ 단위의 밀도
  p(ρ_j | θ) = (1 − ε)·N(r_j; 0, σ_r²)·|n·d_j| / … (수직 거리 r = (ρ − t̂)·|n·d_j| 의 야코비안) + ε / W_j.
  유효 반환이 없는 열은 우도에 들지 않는다 (§3.3 phase 1). 프로파일마다 σ_r, ε 를 EM 으로 추정.
* **추정기:** (1) 시작 = 설계 템플릿 1-D 정합 필터 (판 수평 높이 곡선, 열 격자 = dX, Huber 손실) [L§1 Kim 2008] 과 (있으면) 심 추정
  예측 — 둘 다 맞추고 사후 확률이 큰 쪽; (2) 결정적 담금질 혼합 EM (시작 척도 = 설계 허용치 척도, 절반씩 EM σ 까지);
  (3) 열 발자국 혼합 EM: E 단계 책임도, M 단계 = 사전 분포를 더한 Gauss–Newton (가중 γ·Huber/σ_r², 면별 모멘트로 정확한 정규
  방정식, 경계 활성 집합, 직선 탐색), σ_r² = Σγr²/Σγ, ε = mean(1 − γ). 공분산 (Laplace / GUM 1차): Σ_θ = (Jᵀ diag(γ/σ_r²) J + 발자국
  비율 항 + Λ_prior)⁻¹. 열 발자국: 픽셀 열은 광선 각 δ = b·cos²ϑ 를 덮는다 — 꼭짓점이 든 열은 보이는 면들의 혼합 (비율 π(θ)), 가려진
  벽·루트페이스 옆 꼭짓점의 측면 정보는 여기서 온다. 위상 대안 (BIC): 루트페이스 = 설계값 대 자유; 판 끝 너머 직선 (대안과 같은 선택지).
  판 틀: 베벨·루트페이스·직각 벽은 자기 판의 법선 기준 (β = 판 법선 대비). 기지 기하: c, φ, 전단 κ (`cut_geometry`).
* **대안 가설 (§3.5):** 그루브 없음 = 직선 1개 또는 자유 단차로 이어진 직선 2개 (판; 판 + 판 끝/모서리 단차 → 테이블/뒷댐재),
  같은 혼합 우도, K 는 BIC. λ_i = ln L̂_g − ln L̂_alt − ½(p_g − p_alt)·ln n_valid (BIC 근사 log Bayes factor). 임계값으로 쓰지 않는다.
* 모서리 점은 특별 취급하지 않는다: 가중 감소는 광선 가림 (면 배정) · Huber · 혼합 우도에서만 나온다.

판정 형식은 명세의 것만: 모형 선택은 증거 (BIC, Bayes factor) 로, 임계값 없음. [R] 수치 상수 (반복 상한·수렴 ε·σ 바닥·먼 끝 길이)
만 이 모듈에 있다. 통계 상수: Huber 1.345, MAD 1.4826, BIC ln n.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .g1_types import LegObs, THETA_NAMES, Tolerances
from .specimen import GrooveSpec
from .validity import Z_VALID_MM_DEFAULT, dx_linear_coeffs

__all__ = ["RayFan", "Rays", "Template", "Prior", "ProfileFit", "profile_prior", "fit_profile", "fit_alternative", "leg_context", "fit_rows",
           "rows_phase_a", "rows_phase_b", "row_starts", "row_counts", "lookback", "lookbacks", "assemble_leg", "ROW_KEYS",
           "rows_select", "merge_rows", "finish_candidates", "finish_select",
           "template_search_start", "fit_leg", "cut_geometry", "root_face_design", "choose_bevel_side", "root_points_base", "face_lines_base", "vertices_base",
           "bevel_x_sign_for", "fit_dwell", "DwellFit", "parse_prediction", "VERTEX_NAMES", "FACE_IDS", "N_THETA"]

N_THETA = len(THETA_NAMES)
I_XR, I_ZR, I_AL, I_G, I_BE, I_RF, I_HL, I_HR, I_GL, I_GR = range(N_THETA)

# 통계 상수 (조정값 아님)
HUBER_C = 1.345                 # Huber 95 % 효율
MAD_K = 1.4826                  # MAD → σ (정규)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# [R] 수치 상수만 (판정에 쓰이지 않는다)
_IT_HUBER = 40                  # 1단계 (담금질) 반복 상한
_IT_EM = 60                     # EM 반복 상한
_STEP_TOL = 0.1                 # 수렴: |Δθ_i| ≤ _STEP_TOL·σ_θi (추정 오차 예산보다 훨씬 작다)
_SIG_TOL = 1e-2                 # 수렴: |Δσ|/σ
_SIG_FLOOR_MM = 1e-4            # σ 수치 바닥
_EPS_CLIP = 1e-9                # ε ∈ [_EPS_CLIP, 1 − _EPS_CLIP]
_FAR_MM = 1e4                   # 판 상면 다각선의 먼 끝 (시야보다 충분히 멀리)
_COS_FLOOR = 1e-12              # |n·d| 수치 바닥
_RIDGE = 1e-9                   # 정보 없는 파라미터의 수치 능선 (최대 대각 대비)
_IT_SPLIT = 6                   # 2-직선 대안의 분할 재배정 반복 상한
_HALVE_MAX = 3                  # 직선 탐색 절반 횟수 상한 (걸음 1/8 아래는 이득이 없다)
_Q_TOL = 1e-2                   # 수렴: 반복당 관측 로그 사후 이득 [nat] (무시할 증거)

VERTEX_NAMES = {
    "single_bevel": ("top_corner_L", "root_face_top_L", "root_edge_L", "root_edge_R", "top_corner_R"),
    "single_v": ("top_corner_L", "root_face_top_L", "root_edge_L", "root_edge_R", "root_face_top_R", "top_corner_R"),
    "square": ("top_corner_L", "root_edge_L", "root_edge_R", "top_corner_R"),
}
FACE_IDS = {
    "single_bevel": ("top_L", "bevel_L", "root_face_L", "gap_bottom", "wall_R", "top_R"),
    "single_v": ("top_L", "bevel_L", "root_face_L", "gap_bottom", "root_face_R", "bevel_R", "top_R"),
    "square": ("top_L", "wall_L", "gap_bottom", "wall_R", "top_R"),
}


# ======================================================================================================================
# 광선
# ======================================================================================================================
@dataclass(frozen=True)
class RayFan:
    """열 광선 법칙 dX(z) = a + b·z (센서 사양 knot) → 가상 발광점 o = (0, z0), z0 = −a/b. FOV(z) = n_columns·dX(z)."""
    a: float
    b: float
    z_window_mm: tuple = Z_VALID_MM_DEFAULT
    n_columns: int = 1408            # 센서 열 수 (dX 법칙이 정의된 열 격자)

    @classmethod
    def from_validity(cls, z_valid_mm=None) -> "RayFan":
        a, b = dx_linear_coeffs()
        return cls(a=float(a), b=float(b), z_window_mm=tuple(float(v) for v in (z_valid_mm or Z_VALID_MM_DEFAULT)))

    @property
    def z0(self) -> float:
        return -self.a / self.b

    @property
    def origin(self) -> np.ndarray:
        return np.array([0.0, self.z0])

    def dx_mm(self, z_mm):
        return self.a + self.b * np.asarray(z_mm, dtype=float)

    def fov_mm(self, z_mm, n_columns: int = 1408):
        return float(n_columns) * self.dx_mm(z_mm)

    def rays(self, x_mm, z_mm, n_points: Optional[int] = None) -> "Rays":
        """유효 측정점 (n,) → 광선 묶음. 열 발자국 (픽셀 열이 덮는 광선 각) δ = b·cos²ϑ·(n_columns / n_points):
        tan ϑ = x/(z − z0) = k·b (열 번호 k) 이므로 이웃 열의 각 간격 (n_points < n_columns 이면 열 솎음)."""
        P = np.stack([np.asarray(x_mm, dtype=float), np.asarray(z_mm, dtype=float)], axis=-1)
        v = P - self.origin
        rho = np.hypot(v[:, 0], v[:, 1])
        D = v / rho[:, None]
        zlo, zhi = self.z_window_mm
        W = (zhi - zlo) / np.maximum(D[:, 1], _COS_FLOOR)       # 광선이 보정 z 창 안을 지나는 길이 (D_z > 0)
        ang = np.arctan2(D[:, 0], D[:, 1])
        step = float(self.n_columns) / float(n_points) if n_points else 1.0
        delta = self.b * np.square(np.cos(ang)) * step
        return Rays(P=P, D=D, rho=rho, W=W, o=self.origin.copy(), ang=ang, delta=delta, lo=ang - 0.5 * delta,
                    hi=ang + 0.5 * delta)


@dataclass
class Rays:
    P: np.ndarray        # (n,2) 측정점 (x, z) [mm]
    D: np.ndarray        # (n,2) 단위 광선 방향 (o → p)
    rho: np.ndarray      # (n,) 광선 거리 [mm]
    W: np.ndarray        # (n,) 보정 z 창의 광선 길이 [mm]
    o: np.ndarray        # (2,) 가상 발광점
    ang: np.ndarray = None      # (n,) 광선 각 (z 축에서 x 쪽, rad)
    delta: np.ndarray = None    # (n,) 열 발자국 각폭
    lo: np.ndarray = None       # (n,) 발자국 왼쪽 끝 각
    hi: np.ndarray = None       # (n,) 발자국 오른쪽 끝 각

    @property
    def n(self) -> int:
        return int(self.P.shape[0])


# ======================================================================================================================
# 템플릿
# ======================================================================================================================
class Template:
    """설계 단면 템플릿 (레이저 평면, 겉보기 기하). 정규 틀에서 L 판이 −ξ, R 판이 +ξ; 센서로는 m = −bevel_x_sign 로 거울."""

    def __init__(self, spec: GrooveSpec, bevel_x_sign: int, shear: float = 0.0) -> None:
        """shear κ: 판 법선의 겉보기 상 = e2 + κ·e1 (레이저 평면이 심에 비스듬하고 (ψ) 동시에 기울 때 (φ) 의 기지 기하 상수,
        `cut_geometry` 로 계산; 거친 레그 (φ = 0)·정밀 레그 (ψ ≈ 0) 에서는 0)."""
        if int(bevel_x_sign) not in (1, -1):
            raise ValueError(f"bevel_x_sign must be +1 | -1, got {bevel_x_sign!r}")
        self.spec = spec
        self.kind = spec.kind
        self.backing = bool(spec.backing)
        self.bevel_x_sign = int(bevel_x_sign)
        self.m = -float(bevel_x_sign)
        self.shear = float(shear)
        names = FACE_IDS[self.kind]
        self.face_names = tuple(n for n in names if self.backing or n != "gap_bottom")
        self.vertex_names = VERTEX_NAMES[self.kind]

    # ---- 정규 틀 꼭짓점 (ξ, η) 과 θ 미분 -----------------------------------------------------------------------------
    def _canon(self, th):
        """→ (verts [(ξ, η, dξ{i:v}, dη{i:v})], faces [(name, a, b, anchor, φ, {i: dφ/dθ_i})]) — 꼭짓점 인덱스 a/b 는
        확장 목록 [far_L] + verts + [far_R] 기준 (anchor 는 유한 꼭짓점).

        각 판은 자기 틀: 상면 방향 t = (cos γ, sin γ), 법선 n = (−sin γ, cos γ) (뒷댐재 e1 기준 γ). 베벨·루트페이스·직각 벽은 그 판의
        법선 기준 (베벨각 β 는 판 법선 대비 — 판과 뒷댐재가 평행하지 않아도 β 는 가공된 판의 값). 루트 모서리 R_L, R_R 는 뒷댐재 선
        (η = 0) 위, 상부 코너 높이 h 는 뒷댐재 선 위의 높이."""
        G, be, rf, hL, hR, gL, gR = (float(th[I_G]), float(th[I_BE]), float(th[I_RF]), float(th[I_HL]), float(th[I_HR]),
                                     float(th[I_GL]), float(th[I_GR]))
        g2 = 0.5 * G
        k = self.kind
        cL, sL = math.cos(gL), math.sin(gL)
        cR, sR = math.cos(gR), math.sin(gR)
        hpi = 0.5 * math.pi
        RL = (-g2, 0.0, {I_G: -0.5}, {})
        RR = (g2, 0.0, {I_G: 0.5}, {})

        def bevel_L():
            B = be + gL
            tB = math.tan(B)
            sec2 = 1.0 + tB * tB
            a_ = hL - rf * cL
            FL = (-g2 - rf * sL, rf * cL, {I_G: -0.5, I_RF: -sL, I_GL: -rf * cL}, {I_RF: cL, I_GL: -rf * sL})
            CL = (-g2 - rf * sL - a_ * tB, hL,
                  {I_G: -0.5, I_BE: -a_ * sec2, I_RF: -sL + cL * tB, I_HL: -tB, I_GL: -rf * cL - rf * sL * tB - a_ * sec2},
                  {I_HL: 1.0})
            return CL, FL

        if k == "single_bevel":
            CL, FL = bevel_L()
            CR = (g2 - hR * sR / cR, hR, {I_G: 0.5, I_HR: -sR / cR, I_GR: -hR / (cR * cR)}, {I_HR: 1.0})
            verts = [CL, FL, RL, RR, CR]
            # 확장 인덱스: 0 far_L, 1 CL, 2 FL, 3 RL, 4 RR, 5 CR, 6 far_R
            faces = [("top_L", 0, 1, 1, gL, {I_GL: 1.0}),
                     ("bevel_L", 1, 2, 1, be + gL - hpi, {I_BE: 1.0, I_GL: 1.0}),
                     ("root_face_L", 2, 3, 3, gL - hpi, {I_GL: 1.0}),
                     ("gap_bottom", 3, 4, 3, 0.0, {}),
                     ("wall_R", 4, 5, 4, hpi + gR, {I_GR: 1.0}),
                     ("top_R", 5, 6, 5, gR, {I_GR: 1.0})]
        elif k == "single_v":
            CL, FL = bevel_L()
            Bp = be - gR
            tBp = math.tan(Bp)
            sec2p = 1.0 + tBp * tBp
            a_r = hR - rf * cR
            FR = (g2 - rf * sR, rf * cR, {I_G: 0.5, I_RF: -sR, I_GR: -rf * cR}, {I_RF: cR, I_GR: -rf * sR})
            CR = (g2 - rf * sR + a_r * tBp, hR,
                  {I_G: 0.5, I_BE: a_r * sec2p, I_RF: -sR - cR * tBp, I_HR: tBp, I_GR: -rf * cR + rf * sR * tBp - a_r * sec2p},
                  {I_HR: 1.0})
            verts = [CL, FL, RL, RR, FR, CR]
            faces = [("top_L", 0, 1, 1, gL, {I_GL: 1.0}),
                     ("bevel_L", 1, 2, 1, be + gL - hpi, {I_BE: 1.0, I_GL: 1.0}),
                     ("root_face_L", 2, 3, 3, gL - hpi, {I_GL: 1.0}),
                     ("gap_bottom", 3, 4, 3, 0.0, {}),
                     ("root_face_R", 4, 5, 4, hpi + gR, {I_GR: 1.0}),
                     ("bevel_R", 5, 6, 6, hpi - be + gR, {I_BE: -1.0, I_GR: 1.0}),
                     ("top_R", 6, 7, 6, gR, {I_GR: 1.0})]
        else:  # square
            CL = (-g2 - hL * sL / cL, hL, {I_G: -0.5, I_HL: -sL / cL, I_GL: -hL / (cL * cL)}, {I_HL: 1.0})
            CR = (g2 - hR * sR / cR, hR, {I_G: 0.5, I_HR: -sR / cR, I_GR: -hR / (cR * cR)}, {I_HR: 1.0})
            verts = [CL, RL, RR, CR]
            faces = [("top_L", 0, 1, 1, gL, {I_GL: 1.0}),
                     ("wall_L", 1, 2, 1, gL - hpi, {I_GL: 1.0}),
                     ("gap_bottom", 2, 3, 2, 0.0, {}),
                     ("wall_R", 3, 4, 3, hpi + gR, {I_GR: 1.0}),
                     ("top_R", 4, 5, 4, gR, {I_GR: 1.0})]
        if not self.backing:
            faces = [f for f in faces if f[0] != "gap_bottom"]
        return verts, faces

    # ---- 센서 평면 기하 ---------------------------------------------------------------------------------------------
    def geometry(self, th) -> dict:
        """센서 평면 (x, z) 의 꼭짓점·면 직선과 θ 미분. 사상: (ξ, η) → R + (m·ξ + κ·η)·e1 + η·e2.

        반환: V (nv,2) 유한 꼭짓점, dV (nv,2,10); seg0/seg1 (nf,2) 선분 끝 (먼 끝 포함); A (nf,2) 면 기준점, dA (nf,2,10);
        u (nf,2) 면 방향, n (nf,2) 공기 쪽 법선, dn/du (nf,2,10); names (nf,)."""
        th = np.asarray(th, dtype=float)
        xR, zR, al = float(th[I_XR]), float(th[I_ZR]), float(th[I_AL])
        m = self.m
        kap = self.shear
        ca, sa = math.cos(al), math.sin(al)
        e1x, e1z, e2x, e2z = ca, sa, sa, -ca
        verts, faces = self._canon(th)
        nv = len(verts)
        V = np.empty((nv, 2))
        dV = np.zeros((nv, 2, N_THETA))
        for i, (xi, eta, dxi, deta) in enumerate(verts):
            a1 = m * xi + kap * eta
            V[i, 0] = xR + a1 * e1x + eta * e2x
            V[i, 1] = zR + a1 * e1z + eta * e2z
            dV[i, 0, I_XR] = 1.0
            dV[i, 1, I_ZR] = 1.0
            # de1/dα = J e1 = (−e1z, e1x), de2/dα = J e2 = e1
            dV[i, 0, I_AL] = -a1 * e1z + eta * e1x
            dV[i, 1, I_AL] = a1 * e1x + eta * e1z
            for q, v in dxi.items():
                dV[i, 0, q] += m * v * e1x
                dV[i, 1, q] += m * v * e1z
            for q, v in deta.items():
                dV[i, 0, q] += kap * v * e1x + v * e2x
                dV[i, 1, q] += kap * v * e1z + v * e2z
        nf = len(faces)
        u = np.empty((nf, 2))
        n = np.empty((nf, 2))
        dn = np.zeros((nf, 2, N_THETA))
        du = np.zeros((nf, 2, N_THETA))
        A = np.empty((nf, 2))
        dA = np.empty((nf, 2, N_THETA))
        seg0 = np.empty((nf, 2))
        seg1 = np.empty((nf, 2))
        names = []
        for f, (name, a, b, anc, phi, dphi) in enumerate(faces):
            cp, sp = math.cos(phi), math.sin(phi)
            b1 = m * cp + kap * sp
            ux, uz = b1 * e1x + sp * e2x, b1 * e1z + sp * e2z
            L2 = ux * ux + uz * uz
            L = math.sqrt(L2)
            ux, uz = ux / L, uz / L
            u[f] = (ux, uz)
            # n = −m·J·u (공기 쪽), J = 90° 회전 (x, z) → (−z, x)
            nx, nz = m * uz, -m * ux
            n[f] = (nx, nz)
            du[f, 0, I_AL], du[f, 1, I_AL] = -uz, ux                 # du/dα = J u
            dn[f, 0, I_AL], dn[f, 1, I_AL] = -nz, nx                 # dn/dα = J n
            for q, v in dphi.items():
                rate = -m / L2 * v                                   # dψ_xz/dθ_q
                du[f, 0, q] += rate * -uz
                du[f, 1, q] += rate * ux
                dn[f, 0, q] += -(v / L2) * ux                        # dn/dφ = −u/|u'|²
                dn[f, 1, q] += -(v / L2) * uz
            A[f] = V[anc - 1]
            dA[f] = dV[anc - 1]
            names.append(name)
        far_l = V[0] - _FAR_MM * u[0]
        far_r = V[-1] + _FAR_MM * u[-1]
        ext = np.vstack([far_l[None], V, far_r[None]])
        for f, (name, a, b, anc, phi, dphi) in enumerate(faces):
            seg0[f] = ext[a]
            seg1[f] = ext[b]
        return {"V": V, "dV": dV, "seg0": seg0, "seg1": seg1, "A": A, "dA": dA, "u": u, "n": n, "dn": dn,
                "du": du, "names": tuple(names)}

    def polyline(self, th) -> tuple:
        """꼭짓점 (먼 끝 포함, (nv+2, 2)) 과 면 이름 (nf,)."""
        g = self.geometry(th)
        return np.vstack([g["seg0"][:1], g["V"], g["seg1"][-1:]]), g["names"]

    def cast(self, th, rays: Rays, geom: Optional[dict] = None) -> tuple:
        """광선 투사: 각 광선이 처음 만나는 면 → (t̂ (n,) 광선 거리 [mm] (없으면 inf), face (n,) 면 인덱스 (없으면 −1))."""
        g = self.geometry(th) if geom is None else geom
        return _cast(g["seg0"], g["seg1"], rays.o, rays.D)


def _cast(seg0, seg1, o, D):
    E = seg1 - seg0                                    # (K,2)
    w = seg0 - o                                       # (K,2)
    dx = D[:, 0:1]
    dz = D[:, 1:2]
    den = dx * E[None, :, 1] - dz * E[None, :, 0]      # (n,K)
    tn = w[:, 0] * E[:, 1] - w[:, 1] * E[:, 0]         # (K,)
    sn = w[None, :, 0] * dz - w[None, :, 1] * dx       # (n,K)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = tn[None, :] / den
        s = sn / den
    ok = (den != 0.0) & (t > 0.0) & (s >= 0.0) & (s <= 1.0)
    t = np.where(ok, t, np.inf)
    k = np.argmin(t, axis=1)
    th = t[np.arange(t.shape[0]), k]
    k = np.where(np.isfinite(th), k, -1)
    return th, k


# ======================================================================================================================
# 사전 분포
# ======================================================================================================================
@dataclass
class Prior:
    """θ 사전: 가우시안 (sd = inf → 없음) + 영역 경계 (lo == hi → 고정). start = 설계 시작값 (정합 필터·초기화)."""
    mu: np.ndarray
    sd: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    start: np.ndarray
    c: float = 1.0
    sigma_c: float = 0.0
    phi_rad: float = 0.0
    shear: float = 0.0

    def __post_init__(self):
        sd = np.asarray(self.sd, dtype=float)
        ok = np.isfinite(sd) & (sd > 0)
        self._prec = np.where(ok, 1.0 / np.where(ok, sd, 1.0) ** 2, 0.0)
        self._free = np.asarray(self.hi) > np.asarray(self.lo)

    @property
    def prec(self) -> np.ndarray:
        return self._prec

    @property
    def free(self) -> np.ndarray:
        return self._free

    def with_fixed(self, i: int, value: float) -> "Prior":
        """θ_i 를 value 에 고정한 사본 (위상 대안의 모형 선택용)."""
        lo, hi, start, mu = self.lo.copy(), self.hi.copy(), self.start.copy(), self.mu.copy()
        lo[i] = hi[i] = start[i] = mu[i] = float(value)
        return Prior(mu=mu, sd=self.sd.copy(), lo=lo, hi=hi, start=start, c=self.c, sigma_c=self.sigma_c,
                     phi_rad=self.phi_rad, shear=self.shear)

    def clip(self, th) -> np.ndarray:
        th = np.minimum(np.maximum(np.asarray(th, dtype=float), self.lo), self.hi)
        if self.hi[I_RF] > self.lo[I_RF]:
            th[I_RF] = min(max(th[I_RF], self.lo[I_RF]), max(min(th[I_HL], th[I_HR]), self.lo[I_RF]))   # rf ∈ [0, h]
        return th

    def log_prior(self, th) -> float:
        d = np.asarray(th, dtype=float) - self.mu
        return float(-0.5 * np.dot(self._prec, np.where(self._prec > 0, d * d, 0.0)))

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": [float(v) if math.isfinite(v) else None for v in self.sd],
                "c": self.c, "sigma_c": self.sigma_c, "phi_rad": self.phi_rad, "shear": self.shear}


def root_face_design(qc_targets: Optional[dict], spec: Optional[GrooveSpec] = None) -> tuple:
    """job qc_targets 의 루트면 설계·허용치 `root_face_mm: {design, tol}` → (design_mm, tol_mm | None). design 이 없으면 spec 값."""
    rf = (qc_targets or {}).get("root_face_mm") if isinstance(qc_targets, dict) else None
    d = float(spec.root_face_mm) if spec is not None else 0.0
    if not isinstance(rf, dict):
        return d, None
    if rf.get("design") is not None:
        d = float(rf["design"])
    t = rf.get("tol")
    return d, (float(t) if t is not None else None)


def profile_prior(spec: GrooveSpec, tol: Optional[Tolerances], *, c: float = 1.0, sigma_c: float = 0.0,
                  phi_rad: float = 0.0, shear: float = 0.0, qc_targets: Optional[dict] = None) -> Prior:
    """설계 D 와 qc 허용치 (§3.2 표): G ~ N(c·G_d, (c·tol_G)² + (G_d·σ_c)²), tan β_app = c·cos φ·tan β_d (sd tol_β),
    h_L ~ N(D_d/cos φ, (tol_D/cos φ)²), rf ~ N(rf_d/cos φ, (tol_rf/cos φ)²) on [0, h_L] (job `root_face_mm: {design, tol}`, 사용자
    2026-10-09 ±0.2 mm; 허용치가 없으면 평탄 + BIC 위상 선택), h_R > 0 평탄; x_R, z_R, α, γ 없음. 허용치 None → 그 사전 없음.
    루트면 허용치는 `tol.root_face_mm` (g1_types 에 생기면) 또는 `qc_targets`."""
    c = float(c)
    cphi = math.cos(float(phi_rad))
    inf = math.inf
    G_d = float(spec.root_gap_mm)
    D_d = float(spec.thickness_mm)
    beta_d = math.radians(float(spec.bevel_angle_deg))
    mu = np.zeros(N_THETA)
    sd = np.full(N_THETA, inf)
    lo = np.full(N_THETA, -inf)
    hi = np.full(N_THETA, inf)
    lo[I_AL], hi[I_AL] = -0.5 * math.pi, 0.5 * math.pi
    lo[I_GL], hi[I_GL] = -0.5 * math.pi, 0.5 * math.pi
    lo[I_GR], hi[I_GR] = -0.5 * math.pi, 0.5 * math.pi
    lo[I_G] = 0.0
    lo[I_HL] = lo[I_HR] = _SIG_FLOOR_MM
    lo[I_RF] = 0.0
    mu[I_G] = c * G_d
    t_g = getattr(tol, "gap_mm", None) if tol is not None else None
    if t_g is not None:
        sd[I_G] = math.hypot(c * float(t_g), G_d * float(sigma_c))
    if spec.kind == "square":
        mu[I_BE] = 0.0
        lo[I_BE] = hi[I_BE] = 0.0
        lo[I_RF] = hi[I_RF] = 0.0
    else:
        mu[I_BE] = math.atan(c * cphi * math.tan(beta_d))
        lo[I_BE], hi[I_BE] = 0.0, 0.5 * math.pi - 1e-6
        t_b = getattr(tol, "bevel_deg", None) if tol is not None else None
        if t_b is not None:
            sd[I_BE] = math.radians(float(t_b))
    mu[I_HL] = D_d / cphi
    t_d = getattr(tol, "depth_mm", None) if tol is not None else None
    if t_d is not None:
        sd[I_HL] = float(t_d) / cphi
    start = mu.copy()
    rf_d, rf_t = root_face_design(qc_targets, spec)
    rf_t_obj = getattr(tol, "root_face_mm", None) if tol is not None else None
    if rf_t_obj is not None:
        rf_t = float(rf_t_obj)
    start[I_RF] = rf_d / cphi if spec.kind != "square" else 0.0
    if spec.kind != "square" and rf_t is not None:
        sd[I_RF] = rf_t / cphi
    start[I_HR] = (D_d + float(spec.hi_lo_mm)) / cphi
    mu[I_HR] = start[I_HR]                 # sd inf: 평탄 (§3.2; hi-lo = h_R − h_L 은 출력)
    mu[I_RF] = start[I_RF]
    return Prior(mu=mu, sd=sd, lo=lo, hi=hi, start=start, c=c, sigma_c=float(sigma_c), phi_rad=float(phi_rad),
                 shear=float(shear))


def cut_geometry(T_base_opt, u_base, n_base) -> dict:
    """레이저 평면이 심 (단위 방향 u, 판 법선 n — 현재 추정) 을 자르는 겉보기 기하 (§3.2): 단면은 u 를 따라 평면에 평행 투영된다.
    → {"c": 측면 늘음 (= 1/|cos ψ|), "phi_rad": 높이 늘음 1/cos φ 의 φ, "shear": 판 법선 상의 e1 성분 κ, "psi_rad"}.
    T_base_opt (4,4) 또는 (N,4,4) (평균 자세)."""
    T = np.asarray(T_base_opt, dtype=float)
    if T.ndim == 3:
        X, Y, Z = (np.mean(T[:, :3, i], axis=0) for i in range(3))
    else:
        X, Y, Z = T[:3, 0], T[:3, 1], T[:3, 2]
    u = np.asarray(u_base, dtype=float)
    u = u / np.linalg.norm(u)
    n = np.asarray(n_base, dtype=float)
    n = n - (n @ u) * u
    n = n / np.linalg.norm(n)
    w = np.cross(n, u)
    yu = float(Y @ u)
    if abs(yu) < 1e-12:
        raise ValueError("cut_geometry: laser plane contains the seam direction")
    wp = w - (float(Y @ w) / yu) * u
    npj = n - (float(Y @ n) / yu) * u
    w2 = np.array([X @ wp, Z @ wp])
    n2 = np.array([X @ npj, Z @ npj])
    c = float(np.linalg.norm(w2))
    e1 = w2 / c
    if e1[0] < 0:
        e1 = -e1
    e2 = np.array([e1[1], -e1[0]])                        # (sin α, −cos α): 센서 쪽
    a_n = float(n2 @ e2)
    if a_n < 0:                                           # n 은 센서 쪽 법선
        n2, a_n = -n2, -a_n
    shear = float(n2 @ e1) / a_n
    phi = math.acos(min(1.0, 1.0 / a_n)) if a_n >= 1.0 else 0.0
    psi = math.acos(min(1.0, abs(yu)))
    return {"c": c, "phi_rad": phi, "shear": shear, "psi_rad": psi}


# ======================================================================================================================
# 손실 · 우도 조각
# ======================================================================================================================
def _huber_w(t):
    a = np.abs(t)
    return np.where(a <= HUBER_C, 1.0, HUBER_C / np.maximum(a, 1e-300))


def _huber_rho(t):
    a = np.abs(t)
    return np.where(a <= HUBER_C, 0.5 * a * a, HUBER_C * a - 0.5 * HUBER_C * HUBER_C)


def _mad_scale(r) -> float:
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return 1.0
    return max(MAD_K * float(np.median(np.abs(r))), _SIG_FLOOR_MM)


def _log_in(r, cosv, sig, eps):
    """수직 잔차 r 의 인라이어 밀도를 광선 거리 단위로 (ln)."""
    return (math.log1p(-eps) + np.log(np.maximum(cosv, _COS_FLOOR)) - math.log(sig) - _LOG_SQRT_2PI
            - 0.5 * np.square(r / sig))


# ======================================================================================================================
# 결과
# ======================================================================================================================
@dataclass
class ProfileFit:
    theta: np.ndarray
    cov: np.ndarray
    log_bf: float
    sigma_r: float
    eps: float
    n_valid: int
    converged: bool
    face_inliers: np.ndarray              # (nf,) Σγ (기대 인라이어 수) — 템플릿 면 순서 (Template.face_names)
    lnL_g: float = float("nan")
    lnL_alt: float = float("nan")
    p_alt: int = 0
    log_post: float = float("nan")
    start: str = ""
    n_iter: int = 0
    starts: dict = field(default_factory=dict)
    p_g: int = N_THETA
    log_bf_free: float = float("nan")    # BIC λ without the seam prior (= log_bf when no prediction covariance was given)
    seam_term: float = float("nan")      # ln N₂(r̂ − r_pred; 0, Σ_pred + Σ_r̂) when the caller gave (θ0, Σ_R)
    plate_inliers: np.ndarray = field(default_factory=lambda: np.zeros(2))   # Σγ over the faces of plate L, R
    plate_columns: np.ndarray = field(default_factory=lambda: np.zeros(2))   # predicted visible columns of plate L, R (Σπ)
    beyond_plate: int = -1               # −1 none; 0 / 1 = the beyond-the-end line replaces plate L / R's side

    def to_dict(self) -> dict:
        return {"theta": self.theta.tolist(), "sigma_theta": np.sqrt(np.maximum(np.diag(self.cov), 0)).tolist(),
                "log_bf": self.log_bf, "sigma_r": self.sigma_r, "eps": self.eps, "n_valid": self.n_valid,
                "converged": self.converged, "face_inliers": self.face_inliers.tolist(), "lnL_g": self.lnL_g,
                "lnL_alt": self.lnL_alt, "p_alt": self.p_alt, "log_post": self.log_post, "start": self.start,
                "n_iter": self.n_iter, "starts": self.starts, "log_bf_free": self.log_bf_free, "seam_term": self.seam_term,
                "plate_inliers": np.asarray(self.plate_inliers).tolist(), "plate_columns": np.asarray(self.plate_columns).tolist(),
                "beyond_plate": self.beyond_plate}


# ======================================================================================================================
# 대안 가설: 직선 1개 / 자유 단차 직선 2개
# ======================================================================================================================
def _tls(P, w):
    """가중 TLS 직선: (중심 c (2,), 센서 쪽 단위 법선 n (2,))."""
    sw = float(np.sum(w))
    if not (sw > 0):
        w = np.ones(P.shape[0])
        sw = float(P.shape[0])
    c = (w @ P) / sw
    Q = P - c
    Cm = (Q * w[:, None]).T @ Q
    ang = 0.5 * math.atan2(2.0 * Cm[0, 1], Cm[0, 0] - Cm[1, 1])                # 주방향
    nvec = np.array([math.sin(ang), -math.cos(ang)])
    if nvec[1] > 0:
        nvec = -nvec
    return c, nvec


def _line_huber(P, it_max=_IT_HUBER):
    """Huber IRLS 직선 (단조 M 추정). → (c, n, s)."""
    c, nv = _tls(P, np.ones(P.shape[0]))
    s = _mad_scale((P - c) @ nv)
    half = 0.5 * float(np.ptp(P @ np.array([-nv[1], nv[0]]))) if P.shape[0] > 1 else 0.0
    for _ in range(it_max):
        r = (P - c) @ nv
        s_new = _mad_scale(r)
        c2, n2 = _tls(P, _huber_w(r / s_new))
        # 수렴: 자료 범위 끝에서 직선이 움직인 거리 ≤ _STEP_TOL·s (다른 반복과 같은 형식)
        moved = abs(float((c2 - c) @ n2)) + abs(float(n2[0] * nv[1] - n2[1] * nv[0])) * half
        c, nv, s = c2, n2, s_new
        if moved <= _STEP_TOL * s_new:
            break
    return c, nv, s


def _em_lines(P, D, W, segs, lines, it_max=_IT_EM, sig0: Optional[float] = None, eps0: Optional[float] = None):
    """구간별 직선 (segs = [(i0, i1)], 공유 σ, ε) 의 혼합 EM (Huber M 단계). sig0/eps0: 앞 맞춤에서 이어 받는 시작 (없으면 MAD, ½).
    → (lines, σ, ε, lnL, ll (n,))."""
    n = P.shape[0]
    r = np.empty(n)
    cosv = np.empty(n)

    def resid(kk):
        i0, i1 = segs[kk]
        c, nv = lines[kk]
        r[i0:i1] = (P[i0:i1] - c) @ nv
        cosv[i0:i1] = np.abs(D[i0:i1] @ nv)
    for kk in range(len(segs)):
        resid(kk)
    sig = _mad_scale(r) if sig0 is None else float(sig0)
    eps = 0.5 if eps0 is None else float(eps0)
    lnW = np.log(W)
    for _ in range(it_max):
        li = _log_in(r, cosv, sig, eps)
        lo = math.log(eps) - lnW
        gam = 1.0 / (1.0 + np.exp(np.clip(lo - li, -700, 700)))
        sg = float(np.sum(gam))
        sig_new = max(math.sqrt(float(gam @ (r * r)) / sg), _SIG_FLOOR_MM) if sg > 0 else sig
        eps_new = min(max(1.0 - sg / n, _EPS_CLIP), 1.0 - _EPS_CLIP)
        w = gam * _huber_w(r / sig_new)
        moved = 0.0
        for kk, (i0, i1) in enumerate(segs):
            if i1 - i0 < 2:
                continue
            c, nv = _tls(P[i0:i1], w[i0:i1])
            half = 0.5 * float(np.ptp(P[i0:i1] @ np.array([-nv[1], nv[0]])))
            moved = max(moved, (abs(float((c - lines[kk][0]) @ nv))
                                + abs(float(nv[0] * lines[kk][1][1] - nv[1] * lines[kk][1][0])) * half) / sig_new)
            lines[kk] = (c, nv)
            resid(kk)
        conv = abs(sig_new - sig) <= _SIG_TOL * sig and abs(eps_new - eps) <= _SIG_TOL * max(eps, 1e-3) and moved <= _STEP_TOL
        sig, eps = sig_new, eps_new
        if conv:
            break
    ll = np.logaddexp(_log_in(r, cosv, sig, eps), math.log(eps) - lnW)
    return lines, sig, eps, float(np.sum(ll)), ll


def _ll_line(P, D, W, line, sig, eps):
    r = (P - line[0]) @ line[1]
    return np.logaddexp(_log_in(r, np.abs(D @ line[1]), sig, eps), math.log(eps) - np.log(W))


def _split_init(x, z) -> int:
    """수직 최소제곱 두 직선의 SSE 최소 분할 (누적합, O(n)) → 왼쪽 개수 b (2 ≤ b ≤ n−2)."""
    n = x.size
    xc = x - x.mean()
    zc = z - z.mean()

    def sse(xa, za):
        k = np.arange(1, xa.size + 1, dtype=float)
        Sx = np.cumsum(xa)
        Sz = np.cumsum(za)
        Sxx = np.cumsum(xa * xa) - Sx * Sx / k
        Szz = np.cumsum(za * za) - Sz * Sz / k
        Sxz = np.cumsum(xa * za) - Sx * Sz / k
        with np.errstate(divide="ignore", invalid="ignore"):
            out = Szz - np.where(Sxx > 0, Sxz * Sxz / np.where(Sxx > 0, Sxx, 1.0), 0.0)
        return out
    left = sse(xc, zc)                      # left[b−1] = 앞 b 개
    right = sse(xc[::-1], zc[::-1])[::-1]   # right[b] = b.. 끝
    b = np.arange(2, n - 1)
    tot = left[b - 1] + right[b]
    return int(b[int(np.argmin(tot))])


def _alt_fit(rays: Rays) -> dict:
    """그루브 없음 가설: K=2 (자유 단차, p=5; 누적합 분할 → 강건 재배정) 를 먼저, K=1 (p=2) 은 두 구간 중 열이 많은 쪽 직선에서
    시작 (Huber 는 지렛점에 약하다 — 큰 단차가 시야에 있으면 전체 TLS 시작은 타협 직선으로 간다). BIC 로 K 를 고른다.
    정합 필터의 수평 기준 = 고른 모형의 지배 직선 ("line1": K=1 이면 그 직선, K=2 이면 열이 많은 구간의 직선)."""
    P, D, W = rays.P, rays.D, rays.W
    n = P.shape[0]
    lnn = math.log(n)
    out = {}
    start1 = None
    if n >= 6:
        order = np.argsort(P[:, 0], kind="stable")
        Ps, Ds, Ws = P[order], D[order], W[order]
        b = _split_init(Ps[:, 0], Ps[:, 1])
        lines = [_line_huber(Ps[:b])[:2], _line_huber(Ps[b:])[:2]]
        lnL2 = -math.inf
        bb = np.arange(2, n - 1)
        sig2 = eps2 = None
        for _ in range(_IT_SPLIT):
            lines, sig2, eps2, lnL2, _ = _em_lines(Ps, Ds, Ws, [(0, b), (b, n)], lines, sig0=sig2, eps0=eps2)
            l0 = _ll_line(Ps, Ds, Ws, lines[0], sig2, eps2)
            l1 = _ll_line(Ps, Ds, Ws, lines[1], sig2, eps2)
            cum0 = np.concatenate([[0.0], np.cumsum(l0)])
            cum1 = np.concatenate([[0.0], np.cumsum(l1[::-1])])[::-1]
            b_new = int(bb[int(np.argmax(cum0[bb] + cum1[bb]))])
            if b_new == b:
                break
            b = b_new
        major = 0 if b >= n - b else 1
        start1 = lines[major]
        out.update({"lines2": lines, "split_x": float(Ps[b, 0]), "lnL2": lnL2, "sigma2": sig2, "eps2": eps2,
                    "major2": major})
    c1, n1, _ = _line_huber(P)
    lines1, sig1, eps1, lnL1, _ = _em_lines(P, D, W, [(0, n)], [(c1, n1)])
    if start1 is not None:                                   # 두 시작 (전체 Huber TLS, K=2 의 큰 구간) 중 우도가 큰 쪽
        la, sa, ea, lna, _ = _em_lines(P, D, W, [(0, n)], [start1])
        if lna > lnL1:
            lines1, sig1, eps1, lnL1 = la, sa, ea, lna
    out.update({"lnL1": lnL1, "sigma1_line": lines1[0], "sigma1": sig1})
    best = {"K": 1, "lnL": lnL1, "p": 2, "bic": lnL1 - 0.5 * 2 * lnn}
    if "lnL2" in out and out["lnL2"] - 0.5 * 5 * lnn > best["bic"]:
        best = {"K": 2, "lnL": out["lnL2"], "p": 5, "bic": out["lnL2"] - 0.5 * 5 * lnn}
    if best["K"] == 1:
        out["line1"] = lines1[0]
    else:
        out["line1"] = out["lines2"][out["major2"]]
        out["sigma1"] = out["sigma2"]
    out.update(best)
    return out


def fit_alternative(x_mm, z_mm, valid, fan: Optional[RayFan] = None) -> tuple:
    """H_alt (§3.5): 직선 1개 또는 자유 단차 직선 2개, 같은 혼합 우도, K 는 BIC. → (ln L̂_alt, p_alt)."""
    fan = fan or RayFan.from_validity()
    rays = _rays_of(x_mm, z_mm, valid, fan)
    if rays.n < 2:
        return 0.0, 0
    a = _alt_fit(rays)
    return float(a["lnL"]), int(a["p"])


def _rays_of(x_mm, z_mm, valid, fan: RayFan) -> Rays:
    x = np.asarray(x_mm, dtype=float)
    z = np.asarray(z_mm, dtype=float)
    v = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(z)
    return fan.rays(x[v], z[v], n_points=x.size)


# ======================================================================================================================
# 시작: 설계 템플릿 1-D 정합 필터
# ======================================================================================================================
def _design_levelled(tpl: Template, prior: Prior, xi):
    """설계 (사전 시작값) 템플릿의 판 수평 높이 곡선 g(ξ) (L 판 상면 = 0, 위쪽 +; 정규 틀, 위에서 본 첫 면). 뒷댐재가 없으면 갭은 NaN."""
    th = prior.start
    G, be, rf, hL, hR = th[I_G], th[I_BE], th[I_RF], th[I_HL], th[I_HR]
    g2 = 0.5 * G
    tb = math.tan(be)
    xi = np.asarray(xi, dtype=float)
    floor = 0.0 if tpl.backing else np.nan
    if tpl.kind == "square":
        h = np.where(xi < -g2, hL, np.where(xi > g2, hR, floor))
    else:
        xcL = -g2 - (hL - rf) * tb
        hb = np.where(xi <= xcL, hL, rf + (hL - rf) * (-g2 - xi) / max((hL - rf) * tb, 1e-12))
        h = np.where(xi < -g2, hb, floor)
        if tpl.kind == "single_v":
            xcR = g2 + (hR - rf) * tb
            hr = np.where(xi >= xcR, hR, rf + (hR - rf) * (xi - g2) / max((hR - rf) * tb, 1e-12))
            h = np.where(xi > g2, hr, h)
        else:
            h = np.where(xi > g2, hR, h)
    return h - hL


def template_search_start(x_mm, z_mm, valid, tpl: Template, prior: Prior, *, fan: Optional[RayFan] = None,
                          _rays: Optional[Rays] = None, _alt: Optional[dict] = None) -> np.ndarray:
    """1-D 설계 템플릿 정합 필터 (§3.4 (1), [L§1 Kim 2008]): 지배 직선 (K=1 대안의 강건 직선) 으로 판 수평 높이 곡선을 만들어 dX 격자
    (열 간격의 중앙값) 에 칸 나눔 (빈 칸 = 자료 없음, 기여 0), 설계 높이 곡선을 측면 오프셋 c 로 밀며 Huber 손실 합 (척도 = 직선 σ)
    이 최소인 c → θ0 (나머지 = 설계 시작값). Huber 는 큰 잔차에서 선형이라 설계와 실물의 차이 (허용치 수준) 에도 정보가 남는다."""
    fan = fan or RayFan.from_validity()
    rays = _rays if _rays is not None else _rays_of(x_mm, z_mm, valid, fan)
    alt = _alt if _alt is not None else _alt_fit(rays)
    c0, nl = alt["line1"]
    s_l = float(alt["sigma1"])
    ul = np.array([-nl[1], nl[0]])                       # 직선 방향 (x 증가 쪽)
    if ul[0] < 0:
        ul = -ul
    P = rays.P
    s = (P - c0) @ ul                                    # 직선을 따른 좌표
    h = (P - c0) @ nl                                    # 판 수평 높이 (센서 쪽 +)
    m = tpl.m
    th = prior.start
    G, be, hL, hR = th[I_G], th[I_BE], th[I_HL], th[I_HR]
    ds = float(np.median(np.abs(np.diff(np.sort(s))))) if s.size > 1 else 1.0
    ds = ds if ds > 0 else 1.0
    s0 = float(np.min(s))
    cell = np.floor((s - s0) / ds + 0.5).astype(int)
    ncell = int(cell.max()) + 1
    cnt = np.bincount(cell, minlength=ncell)
    hg = np.bincount(cell, weights=h, minlength=ncell) / np.maximum(cnt, 1)
    has = cnt > 0
    half = 0.5 * G + max(hL, hR) * math.tan(be)          # 설계 템플릿이 0 이 아닌 반폭
    kw = int(math.ceil(half / ds))
    xi_w = m * ds * np.arange(-kw, kw + 1)               # 창 안의 정규 ξ (센서 s 순서)
    gw = _design_levelled(tpl, prior, xi_w)
    g_lo = float(_design_levelled(tpl, prior, np.array([-1e9]))[0])
    g_hi = float(_design_levelled(tpl, prior, np.array([1e9]))[0])
    g_left, g_right = (g_lo, g_hi) if m > 0 else (g_hi, g_lo)
    rl = np.where(has, _huber_rho((hg - g_left) / s_l), 0.0)
    rr = np.where(has, _huber_rho((hg - g_right) / s_l), 0.0)
    cl = np.concatenate([[0.0], np.cumsum(rl)])
    cr = np.concatenate([[0.0], np.cumsum(rr)])
    if ncell < 2 * kw + 1:
        k = int(np.argmax(cnt))
        cbest = s0 + k * ds
    else:
        win = np.lib.stride_tricks.sliding_window_view(hg, 2 * kw + 1)                 # (ncell − 2kw, 2kw+1)
        hw_ = np.lib.stride_tricks.sliding_window_view(has, 2 * kw + 1)
        rw = np.where(hw_ & np.isfinite(gw)[None, :], _huber_rho((win - np.where(np.isfinite(gw), gw, 0.0)[None, :]) / s_l), 0.0)
        centres = np.arange(kw, ncell - kw)
        score = cl[centres - kw] + rw.sum(axis=1) + (cr[ncell] - cr[centres + kw + 1])
        k = int(centres[int(np.argmin(score))])
        cbest = s0 + k * ds
    alpha = math.atan2(ul[1], ul[0])
    e2 = np.array([math.sin(alpha), -math.cos(alpha)])
    top = c0 + cbest * ul                                 # L 판 상면 위의 점 (정규 ξ = 0 위)
    R = top - hL * e2
    th0 = prior.start.copy()
    th0[I_XR], th0[I_ZR], th0[I_AL] = R[0], R[1], alpha
    th0[I_GL] = th0[I_GR] = 0.0
    return prior.clip(th0)


# ======================================================================================================================
# 템플릿 맞춤
# ======================================================================================================================
def _outer_faces(seg0, seg1, o) -> tuple:
    """꼭짓점 각 범위 밖의 광선이 보는 바깥 판 상면 (왼쪽 = 광선 각이 작은 쪽, 오른쪽): 먼 끝의 광선 각으로 정한다 (거울 부호와 무관).
    → (왼쪽 면 색인, 오른쪽 면 색인)."""
    a0 = math.atan2(seg0[0, 0] - o[0], seg0[0, 1] - o[1])
    a1 = math.atan2(seg1[-1, 0] - o[0], seg1[-1, 1] - o[1])
    k = seg0.shape[0] - 1
    return (0, k) if a0 < a1 else (k, 0)


def _vertex_angles(g: dict, o):
    """유한 꼭짓점의 광선 각 ϑ_v (nv,) 와 dϑ_v/dθ (nv,10)."""
    Vp = g["V"] - o
    r2 = Vp[:, 0] ** 2 + Vp[:, 1] ** 2
    av = np.arctan2(Vp[:, 0], Vp[:, 1])
    dV = g["dV"]
    dav = (Vp[:, 1:2] * dV[:, 0, :] - Vp[:, 0:1] * dV[:, 1, :]) / r2[:, None]
    return av, dav


def _components(tpl: Template, th, rays: Rays, want_jac: bool = True) -> dict:
    """열 발자국 혼합 (§3.3 의 광선 투사를 픽셀 열의 각폭으로): 열 j 의 발자국 [ϑ_j − δ_j/2, ϑ_j + δ_j/2] 를 그 안의 꼭짓점 광선 각으로
    나눈 조각마다 첫 면 k 와 비율 π_jk(θ) (꼭짓점이 움직이면 연속으로 변한다). 첫 면은 꼭짓점 광선 각에서만 바뀌므로 발자국 안에
    꼭짓점이 없는 열은 중심 광선 하나로 충분하다. 성분 = (열, 면, π, dπ/dθ, 잔차, |n·d|, 야코비안); 모호한 열의 성분은 열별로 연속.
    가림·스치는 면 (벽·루트페이스) 과 모서리 열은 여기서 자연히 처리된다 — 특별 취급 없음."""
    g = tpl.geometry(th)
    seg0, seg1, o = g["seg0"], g["seg1"], rays.o
    av, dav = _vertex_angles(g, o)
    nv = av.size
    nfc = seg0.shape[0]
    # 꼭짓점 각 범위 밖의 열은 바깥 판 상면만 본다 (첫 면은 꼭짓점 각에서만 바뀐다) → 투사·발자국은 범위 안의 열만
    a_min, a_max = float(np.min(av)), float(np.max(av))
    left = rays.hi <= a_min
    right = rays.lo >= a_max
    mid = np.flatnonzero(~(left | right))
    f_left, f_right = _outer_faces(seg0, seg1, o)
    kC = np.empty(rays.n, dtype=int)
    kC[left] = f_left
    kC[right] = f_right
    if mid.size:
        _, kC[mid] = _cast(seg0, seg1, o, rays.D[mid])
    amb_mask = np.zeros(rays.n, dtype=bool)
    if mid.size:
        amb_mask[mid] = np.any((rays.lo[mid, None] < av[None, :]) & (av[None, :] < rays.hi[mid, None]), axis=1)
    amb = np.flatnonzero(amb_mask)
    one = np.flatnonzero(~amb_mask & (kC >= 0))
    n_one = one.size
    col = [one]
    face = [kC[one]]
    pi = [np.ones(n_one)]
    dpi_amb = np.zeros((0, N_THETA))
    starts = np.zeros(0, dtype=int)
    amb_cols = np.zeros(0, dtype=int)
    if amb.size:
        lo = rays.lo[amb]
        hi = rays.hi[amb]
        inside = (av[None, :] > lo[:, None]) & (av[None, :] < hi[:, None])             # (m, nv)
        Bv = np.where(inside, av[None, :], hi[:, None])
        ids = np.where(inside, np.arange(nv)[None, :], -1)
        Bfull = np.concatenate([lo[:, None], Bv, hi[:, None]], axis=1)                   # (m, nv+2)
        Ifull = np.concatenate([np.full((amb.size, 1), -1), ids, np.full((amb.size, 1), -1)], axis=1)
        srt = np.argsort(Bfull, axis=1, kind="stable")
        Bs = np.take_along_axis(Bfull, srt, axis=1)
        Is = np.take_along_axis(Ifull, srt, axis=1)
        length = Bs[:, 1:] - Bs[:, :-1]                                                  # (m, nv+1)
        mids = 0.5 * (Bs[:, 1:] + Bs[:, :-1])
        r0, c0 = np.nonzero(length > 0)
        mm = mids[r0, c0]
        Dm = np.stack([np.sin(mm), np.cos(mm)], axis=-1)
        _, km = _cast(seg0, seg1, o, Dm)
        ok = km >= 0
        rr, cc, km = r0[ok], c0[ok], km[ok]
        dl = rays.delta[amb]
        col.append(amb[rr])
        face.append(km)
        pi.append(length[rr, cc] / dl[rr])
        dav0 = np.vstack([dav, np.zeros((1, N_THETA))])                                  # 인덱스 −1 → 0 행
        dpi_amb = (dav0[Is[rr, cc + 1]] - dav0[Is[rr, cc]]) / dl[rr][:, None]
        if rr.size:
            newg = np.ones(rr.size, dtype=bool)
            newg[1:] = rr[1:] != rr[:-1]
            starts = np.flatnonzero(newg)
            amb_cols = amb[rr[starts]]
    col = np.concatenate(col)
    face = np.concatenate(face).astype(int)
    pi = np.concatenate(pi)
    nf_ = g["n"]
    A = g["A"]
    nk = nf_[face]
    Pc = rays.P[col]
    r = nk[:, 0] * Pc[:, 0] + nk[:, 1] * Pc[:, 1] - np.einsum("kc,kc->k", nf_, A)[face]
    D = rays.D[col]
    cosv = np.abs(nk[:, 0] * D[:, 0] + nk[:, 1] * D[:, 1])
    # 성분 야코비안은 점에 선형: J_c = x_c·dn_f[0] + z_c·dn_f[1] + c_f, c_f = −A_f·dn_f − n_f·dA_f → 면별 (3,10) 계수 Dm
    dn = g["dn"]
    cf = -np.einsum("kc,kcp->kp", A, dn) - np.einsum("kc,kcp->kp", nf_, g["dA"])
    Dm = np.stack([dn[:, 0, :], dn[:, 1, :], cf], axis=1)                                # (K,3,10)
    return {"g": g, "col": col, "face": face, "pi": pi, "dpi_amb": dpi_amb, "n_one": n_one, "one": one,
            "starts": starts, "amb_cols": amb_cols, "r": r, "cos": cosv, "Dm": Dm, "x": Pc[:, 0], "z": Pc[:, 1],
            "amb": amb.size}


def _jac(C: dict) -> np.ndarray:
    """성분 야코비안 행렬 (M,10) (검사·진단용; 맞춤은 면별 모멘트를 쓴다)."""
    Dm = C["Dm"][C["face"]]
    return C["x"][:, None] * Dm[:, 0, :] + C["z"][:, None] * Dm[:, 1, :] + Dm[:, 2, :]


def _col_logsumexp(C: dict, vals, n):
    """열별 log Σ exp(vals) (성분 → 열). 성분이 없는 열은 −inf."""
    out = np.full(n, -np.inf)
    k = C["n_one"]
    out[C["one"]] = vals[:k]
    if C["starts"].size:
        out[C["amb_cols"]] = np.logaddexp.reduceat(vals[k:], C["starts"])
    return out


def _estep(C: dict, rays: Rays, sig, eps):
    """→ (열별 ln p(ρ_j|θ) (n,), 성분 책임도 γ_c (M,))."""
    lc = (math.log1p(-eps) + np.log(np.maximum(C["pi"], 1e-300)) + np.log(np.maximum(C["cos"], _COS_FLOOR))
          - math.log(sig) - _LOG_SQRT_2PI - 0.5 * np.square(C["r"] / sig))
    lin = _col_logsumexp(C, lc, rays.n)
    lo = math.log(eps) - np.log(rays.W)
    ltot = np.logaddexp(lin, lo)
    gam = np.exp(lc - ltot[C["col"]])
    return ltot, gam


def _hard_min(C: dict, n: int) -> np.ndarray:
    """1단계 (Huber) 용: 열마다 |r| 이 가장 작은 성분 하나 (발자국 안의 보이는 면 중) → 성분 가중 0/1."""
    k = C["n_one"]
    w = np.zeros(C["col"].size)
    w[:k] = 1.0
    st = C["starts"]
    if st.size:
        a = np.abs(C["r"][k:])
        grp = np.zeros(a.size, dtype=int)
        grp[st[1:]] = 1
        grp = np.cumsum(grp)
        order = np.lexsort((a, grp))
        first = np.ones(order.size, dtype=bool)
        first[1:] = grp[order][1:] != grp[order][:-1]
        sel = np.zeros(a.size)
        sel[order[first]] = 1.0
        w[k:] = sel
    return w


def _solve(H, g, free):
    idx = np.flatnonzero(free)
    Hs = H[idx][:, idx]
    rid = _RIDGE * max(float(np.max(np.abs(np.diagonal(Hs)))), 1.0)
    Hs[np.diag_indices(idx.size)] += rid
    try:
        Ci = np.linalg.inv(Hs)
    except np.linalg.LinAlgError:
        Ci = np.linalg.pinv(Hs)
    delta = np.zeros(H.shape[0])
    delta[idx] = -(Ci @ g[idx])
    C = np.zeros_like(H)
    C[idx[:, None], idx[None, :]] = Ci
    return delta, C


def _step(H, grad, free, th, prior: "Prior"):
    """경계 활성 집합 GN 걸음: 경계에 있고 걸음이 바깥을 향하는 파라미터는 이번 풀이에서 묶는다. → (Δθ, 공분산 (활성 집합 제외 전체))."""
    act = free.copy()
    delta, C = _solve(H, grad, act)
    for _ in range(N_THETA):
        lo_b = th <= prior.lo + 1e-12
        hi_b = th >= prior.hi - 1e-12
        if th[I_RF] >= min(th[I_HL], th[I_HR]) - 1e-12:
            hi_b = hi_b.copy()
            hi_b[I_RF] = True
        out = act & ((lo_b & (delta < 0)) | (hi_b & (delta > 0)))
        if not np.any(out):
            break
        act = act & ~out
        delta, _ = _solve(H, grad, act)
    return delta, C


def _normal_eq(C: dict, wls, wpi, prior: "Prior", th):
    """GN 정규 방정식: ½Σ wls r² − Σ wpi ln π + 사전 의 (H, grad). 최소제곱 부분은 면별 가중 모멘트 Σ w·ỹỹᵀ (ỹ = (x, z, 1))
    로 정확히: H = Σ_f Dm_fᵀ M_f Dm_f, grad = Σ_f Dm_fᵀ Σ w r ỹ."""
    f = C["face"]
    K = C["Dm"].shape[0]
    x, z, r = C["x"], C["z"], C["r"]
    wx, wz = wls * x, wls * z
    S = np.stack([np.bincount(f, wx * x, K), np.bincount(f, wx * z, K), np.bincount(f, wx, K),
                  np.bincount(f, wz * z, K), np.bincount(f, wz, K), np.bincount(f, wls, K)], axis=1)
    M = np.empty((K, 3, 3))
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = S[:, 0], S[:, 1], S[:, 2]
    M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = S[:, 1], S[:, 3], S[:, 4]
    M[:, 2, 0], M[:, 2, 1], M[:, 2, 2] = S[:, 2], S[:, 4], S[:, 5]
    wr = wls * r
    b = np.stack([np.bincount(f, wr * x, K), np.bincount(f, wr * z, K), np.bincount(f, wr, K)], axis=1)   # (K,3)
    Dm = C["Dm"]
    H = np.sum(np.transpose(Dm, (0, 2, 1)) @ (M @ Dm), axis=0) + np.diag(prior.prec)
    grad = np.einsum("kai,ka->i", Dm, b) + prior.prec * np.where(prior.prec > 0, th - prior.mu, 0.0)
    if wpi is not None and C["dpi_amb"].shape[0]:
        k = C["n_one"]
        Dp = C["dpi_amb"]
        pi = np.maximum(C["pi"][k:], 1e-300)
        a = wpi[k:] / pi
        H = H + (Dp * (a / pi)[:, None]).T @ Dp
        grad = grad - Dp.T @ a
    return H, grad


def _huber_obj(C: dict, hw, s, rays: Rays, prior: Prior, th) -> float:
    """1단계 목적 (최대화): −Σ ρ_H(r/s) (열마다 가장 가까운 보이는 면) − 반환 예측 없는 열의 ρ_H(W/s) + ln 사전."""
    t = C["r"] / s
    n_hit = np.zeros(rays.n, dtype=bool)
    n_hit[C["col"][hw > 0]] = True
    return float(-np.sum(hw * _huber_rho(t)) - np.sum(_huber_rho(rays.W[~n_hit] / s))) + prior.log_prior(th)


def _fit_template(tpl: Template, prior: Prior, rays: Rays, th0, sigma0: Optional[float] = None) -> dict:
    """Huber IRLS (열마다 발자국 안의 가장 가까운 보이는 면) → 열 발자국 혼합 EM (γ, σ_r, ε) + 사전 GN.
    걸음마다 같은 척도에서 목적이 나아질 때까지 절반 (직선 탐색), 그 다음 척도 갱신. 반환: θ, Σ, σ, ε, lnL, log_post, 면별 기대
    인라이어 수, 반복, 수렴."""
    free = prior.free
    n = rays.n
    th = prior.clip(np.asarray(th0, dtype=float))
    n_it = 0
    n_eval = 1
    # ---- 1단계: 결정적 담금질 혼합 EM — 시작 척도 = 설계 시작의 기대 어긋남 (사전 σ 중 길이 [mm] 의 최대 = 허용치 척도),
    #      걸음마다 절반으로 줄여 EM 추정 σ 에 닿으면 끝. 큰 이상치 (판 끝 너머 테이블 등) 는 처음부터 γ≈0, 허용치 수준의 어긋남은 잡힌다.
    C = _components(tpl, th, rays)
    len_sd = prior.sd[[I_G, I_RF, I_HL, I_HR]]
    len_sd = len_sd[np.isfinite(len_sd)]
    hw = _hard_min(C, n)
    s_mad = _mad_scale(C["r"][hw > 0]) if np.any(hw > 0) else 1.0
    if sigma0 is None:                                       # 설계 시작: 기대 어긋남 = 허용치 척도
        sig = float(np.max(len_sd)) if len_sd.size else s_mad
        sig = max(sig, s_mad)
    else:                                                    # 예측 시작: 그 시작의 자기 잔차 척도
        sig = max(float(sigma0), s_mad)
    eps = 0.5
    conv1 = False
    for it in range(_IT_HUBER):
        n_it += 1
        _, gam = _estep(C, rays, sig, eps)
        sg = float(np.sum(gam))
        sig_em = max(math.sqrt(float(np.sum(gam * C["r"] ** 2)) / sg), _SIG_FLOOR_MM) if sg > 0 else sig
        eps = min(max(1.0 - sg / n, _EPS_CLIP), 1.0 - _EPS_CLIP)
        ltot, gam = _estep(C, rays, sig, eps)
        Q0 = float(np.sum(ltot)) + prior.log_prior(th)
        H, grad = _normal_eq(C, gam * _huber_w(C["r"] / sig) / (sig * sig), gam, prior, th)
        delta, Cv = _step(H, grad, free, th, prior)
        sd = np.sqrt(np.maximum(np.diag(Cv), 1e-300))
        lam = 1.0
        moved = False
        Qn = Q0
        n_try = 0 if -0.5 * float(grad @ delta) <= _Q_TOL else _HALVE_MAX     # GN 예측 이득 (뉴턴 감소량) 이 무시할 만하면 탐색 생략
        for _ in range(n_try):
            thn = prior.clip(th + lam * delta)
            Cn = _components(tpl, thn, rays)
            n_eval += 1
            ltn, _ = _estep(Cn, rays, sig, eps)
            Qn = float(np.sum(ltn)) + prior.log_prior(thn)
            if Qn >= Q0:
                moved = bool(np.any(np.abs(thn - th)[free] > _STEP_TOL * sd[free])) and (Qn - Q0) > _Q_TOL
                th, C = thn, Cn
                break
            lam *= 0.5
        if moved:
            continue                                         # 이 척도에서 아직 움직인다 — 같은 척도로 한 번 더
        if sig <= sig_em * (1.0 + _SIG_TOL):                 # EM 척도에 닿음 (σ_em 은 매번 다시 추정 — 수치 허용 안에서)
            conv1 = True
            break
        sig = max(0.5 * sig, sig_em)                         # 이 척도에서 수렴 → 척도 절반
    s = sig
    # ---- 2단계: 열 발자국 혼합 EM (ECM: σ_r, ε 갱신 → 같은 σ_r, ε 에서 θ 걸음) -------------------------------------
    sig = max(s, _SIG_FLOOR_MM)
    eps = 0.5
    conv2 = False
    for it in range(_IT_EM):
        n_it += 1
        _, gam = _estep(C, rays, sig, eps)
        sg = float(np.sum(gam))
        sig_new = max(math.sqrt(float(np.sum(gam * C["r"] ** 2)) / sg), _SIG_FLOOR_MM) if sg > 0 else sig
        eps_new = min(max(1.0 - sg / n, _EPS_CLIP), 1.0 - _EPS_CLIP)
        s_ok = abs(sig_new - sig) <= _SIG_TOL * sig and abs(eps_new - eps) * n <= 1.0 + 1e-9     # ε 해상도 = 열 하나
        sig, eps = sig_new, eps_new
        ltot, gam = _estep(C, rays, sig, eps)
        Q0 = float(np.sum(ltot)) + prior.log_prior(th)
        H, grad = _normal_eq(C, gam * _huber_w(C["r"] / sig) / (sig * sig), gam, prior, th)
        delta, Cv = _step(H, grad, free, th, prior)
        sd = np.sqrt(np.maximum(np.diag(Cv), 1e-300))
        lam = 1.0
        accepted = False
        n_try = 0 if -0.5 * float(grad @ delta) <= _Q_TOL else _HALVE_MAX
        for _ in range(n_try):
            thn = prior.clip(th + lam * delta)
            Cn = _components(tpl, thn, rays)
            n_eval += 1
            ltn, _ = _estep(Cn, rays, sig, eps)
            Qn = float(np.sum(ltn)) + prior.log_prior(thn)
            if Qn >= Q0:
                accepted = True
                break
            lam *= 0.5
        if accepted:
            step = thn - th
            th, C = thn, Cn
        else:
            step = np.zeros(N_THETA)
            Qn = Q0
        small = bool(np.all(np.abs(step[free]) <= _STEP_TOL * sd[free])) or (Qn - Q0) <= _Q_TOL
        if small and s_ok:
            conv2 = True
            break
    # ---- 최종: E 단계, 우도, Laplace 공분산 ---------------------------------------------------------------------------
    ltot, gam = _estep(C, rays, sig, eps)
    lnL = float(np.sum(ltot))
    H, _ = _normal_eq(C, gam / (sig * sig), gam, prior, th)
    _, Cov = _solve(H, np.zeros(N_THETA), free)
    nf = len(C["g"]["names"])
    face_in = np.bincount(C["face"], weights=gam, minlength=nf)[:nf]
    return {"theta": th, "cov": Cov, "sigma": sig, "eps": eps, "lnL": lnL, "log_post": lnL + prior.log_prior(th),
            "gamma": gam, "comp": C, "face_inliers": face_in, "n_iter": n_it, "n_eval": n_eval, "converged": bool(conv2),
            "huber_converged": conv1}


def _empty_fit(prior: Prior, n: int, nf: int) -> ProfileFit:
    """자료 없음 → 증거 없음 (λ = 0), θ = 사전."""
    C = np.diag(np.where(np.isfinite(prior.sd), np.square(prior.sd), np.inf))
    return ProfileFit(theta=prior.start.copy(), cov=C, log_bf=0.0, sigma_r=float("nan"), eps=float("nan"), n_valid=n,
                      converged=False, face_inliers=np.zeros(nf), start="none")


def parse_prediction(pred) -> tuple:
    """예측 시작 → (θ0 (10,) | None, Σ_R (2,2) mm² | None). 받는 형식: None, θ0, (θ0, Σ), {"theta": θ0, "cov": Σ}; Σ 는 (x_R, z_R) 의
    2×2 또는 θ 의 10×10 (그 앞 2×2 를 쓴다). Σ_R = 융합된 심 상태에서 이 프로파일의 루트 예측 공분산 (선 공분산 + 프로파일 흔들림 + 자세
    부류 오프셋)."""
    if pred is None:
        return None, None
    cov = None
    if isinstance(pred, dict):
        th, cov = pred.get("theta"), pred.get("cov")
    elif isinstance(pred, (tuple, list)) and len(pred) == 2 and np.ndim(pred[0]) == 1:
        th, cov = pred
    else:
        th = pred
    th = np.asarray(th, dtype=float) if th is not None else None
    if th is None or th.shape != (N_THETA,) or not np.all(np.isfinite(th)):
        return None, None
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        cov = cov[:2, :2] if cov.shape[0] >= 2 else None
        if cov is None or not np.all(np.isfinite(cov)):
            cov = None
    return th, cov


def _window_area_mm2(rays: Rays) -> float:
    """레이저 평면에서 보정 z 창의 넓이 [mm²] (열 부채꼴: 너비 = n_columns·dX(z), z 창 사다리꼴) — '그루브가 창 어디엔가' 의 균일 사전."""
    fan = RayFan.from_validity()
    zlo, zhi = fan.z_window_mm
    return float((zhi - zlo) * 0.5 * (fan.fov_mm(zlo) + fan.fov_mm(zhi)))


def _seam_term(f: dict, th_pred, cov_pred) -> float:
    """ln N₂(r̂ − r_pred; 0, Σ_pred + Σ_r̂) — 이 심의 예측 위치에 있는 그루브인가 (위치 사전의 주변 우도)."""
    e = np.asarray(f["theta"], dtype=float)[:2] - th_pred[:2]
    S = cov_pred + np.asarray(f["cov"], dtype=float)[:2, :2]
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return -math.inf
    y = np.linalg.solve(L, e)
    return float(-0.5 * (y @ y) - np.sum(np.log(np.diag(L))) - math.log(2.0 * math.pi))


def fit_profile(x_mm, z_mm, valid, tpl: Template, prior: Prior, *, start_pred=None, fan: Optional[RayFan] = None) -> ProfileFit:
    """한 프로파일: 시작 두 개 (정합 필터, 예측) 를 모두 맞추고 사후가 큰 쪽 + 대안 가설 → λ (§3.3–3.6).
    start_pred: θ0 | (θ0, Σ_R) | {"theta", "cov"} (`parse_prediction`). Σ_R 가 있으면 시작 선택과 λ 가 이 심의 예측 위치를 쓴다
    (λ_seam = λ_free + ln N₂(r̂ − r_pred; 0, Σ_R + Σ_r̂) + ln n: 위치 두 파라미터의 BIC 몫을 실제 사전으로 바꾼 것); 그 밖에는 BIC λ."""
    fan = fan or RayFan.from_validity()
    if tpl.shear != prior.shear:
        tpl = Template(tpl.spec, tpl.bevel_x_sign, shear=prior.shear)
    rays = _rays_of(x_mm, z_mm, valid, fan)
    n = rays.n
    if n < 2:
        return _empty_fit(prior, n, len(tpl.face_names))
    alt = _alt_fit(rays)
    starts = {"search": template_search_start(None, None, None, tpl, prior, fan=fan, _rays=rays, _alt=alt)}
    th_p, cov_p = parse_prediction(start_pred)
    if th_p is not None:
        starts["pred"] = th_p
    fits = {name: _fit_template(tpl, prior, rays, th0, sigma0=(_SIG_FLOOR_MM if name == "pred" else None))
            for name, th0 in starts.items()}
    return _finish_profile(tpl, prior, rays, alt, fits, pred=(th_p, cov_p) if cov_p is not None else None)


def _finish_profile(tpl: Template, prior: Prior, rays: Rays, alt: dict, fits: dict, pred: Optional[tuple] = None) -> ProfileFit:
    """시작들의 맞춤 중 사후가 큰 쪽 → (허용치 없는 rf 의 위상 BIC) → 판 끝 너머 선택지 (BIC) → λ, 판별 존재. fits: 이름 →
    _fit_template 결과 (일괄 맞춤 결과도 같은 키; 'comp' 가 없으면 여기서 계산). pred = (θ_pred, Σ_R) 이면 선택·λ 에 심 위치 사전.
    = _finish_choose → _finish_candidate (고른 맞춤만) → _finish_fit. 심 위치 사전과 무관한 부분 (_finish_candidate) 은 시작마다 따로
    계산해 두었다가 사전을 나중에 적용해도 같은 값이다 (`finish_candidates` / `finish_select`, 스트림)."""
    name, info = _finish_choose(fits, pred)
    C = _finish_candidate(tpl, prior, rays, alt, fits[name])
    return _finish_fit(prior, alt, rays.n, name, info, C, pred, _window_area_mm2(rays))


def _finish_choose(fits: dict, pred: Optional[tuple]) -> tuple:
    """시작들의 맞춤 중 사후가 큰 쪽 (pred = (θ_pred, Σ_R) 이면 + 심 위치 항). → (이름, info)."""
    info = {name: {"log_post": float(f["log_post"]), "converged": bool(f["converged"])} for name, f in fits.items()}
    if pred is not None:
        for nm_, f_ in fits.items():
            info[nm_]["seam_term"] = _seam_term(f_, *pred)
        name = max(fits, key=lambda k: fits[k]["log_post"] + info[k]["seam_term"])
    else:
        name = max(fits, key=lambda k: fits[k]["log_post"])
    return name, info


_FIN_KEYS = ("theta", "cov", "sigma", "eps", "lnL", "log_post", "face_inliers", "n_iter", "converged")


def _finish_candidate(tpl: Template, prior: Prior, rays: Rays, alt: dict, f: dict) -> dict:
    """심 위치 사전과 무관한 마무리 (한 시작의 맞춤): 루트면 위상 (BIC, 허용치가 없을 때), 판 끝 너머 선택지 (BIC), 판별 존재."""
    n = rays.n
    lnn = math.log(n)
    p_g = int(np.count_nonzero(prior.free))
    extra = {}
    # 위상 대안 (D10, BIC): 루트페이스 = 설계값 (rf 고정) 대 rf 자유 — 작업에 루트면 허용치가 없을 때만 (있으면 설계 사전).
    rf_model = "prior" if np.isfinite(prior.sd[I_RF]) else "free"
    if rf_model == "free" and prior.free[I_RF] and abs(float(f["theta"][I_RF]) - float(prior.start[I_RF])) > 0.0:
        pr0 = prior.with_fixed(I_RF, float(prior.start[I_RF]))
        f0 = _fit_template(tpl, pr0, rays, f["theta"])
        if f0["lnL"] - 0.5 * (p_g - 1) * lnn >= f["lnL"] - 0.5 * p_g * lnn:
            f, p_g, rf_model = f0, p_g - 1, "design"
        extra["rf_design"] = {"lnL": f0["lnL"], "converged": f0["converged"]}
    # 판 끝 너머 (§3.5 의 대안과 같은 선택지): 템플릿은 단차의 한쪽, 너머는 자유 직선 (테이블·뒷댐재) — p + 3 (직선 2, 단차 1), BIC.
    beyond = None
    if "lines2" in alt:
        if "comp" not in f:
            f = dict(f, comp=_components(tpl, f["theta"], rays))
        ext = _beyond_extension(tpl, prior, rays, f, alt)
        if ext is not None and ext["lnL"] - 0.5 * (p_g + 3) * lnn > f["lnL"] - 0.5 * p_g * lnn:
            f = ext["fit"]
            beyond = {"side": ext["side"], "n_out": ext["n_out"], "lnL": ext["lnL"]}
            p_g += 3
            f = dict(f, lnL=ext["lnL"])
    extra["beyond"] = beyond
    # 판별 존재: 템플릿 면을 판 L / R 로 묶어 기대 인라이어 수 Σγ 와 예측 보임 열 수 Σπ (모든 유효 열, 맞춘 θ)
    Cf = _components(tpl, f["theta"], rays)
    _, gam_f = _estep(Cf, rays, float(f["sigma"]), float(f["eps"]))
    pl = np.array([0 if nm.endswith("_L") else (1 if nm.endswith("_R") else 2) for nm in Cf["g"]["names"]])[Cf["face"]]
    plate_in = np.bincount(pl, gam_f, 3)[:2]
    plate_cols = np.bincount(pl, Cf["pi"], 3)[:2]
    bplate = -1
    if beyond is not None:                               # 센서 −x 쪽 = 판 L 이면 m = +1 (bevel_x_sign = −1)
        left_is_L = tpl.m > 0
        bplate = 0 if ((beyond["side"] == "left") == left_is_L) else 1
    return {"f": {k_: f[k_] for k_ in _FIN_KEYS}, "p_g": p_g, "rf_model": rf_model, "extra": extra, "plate_in": plate_in,
            "plate_cols": plate_cols, "bplate": bplate}


def _finish_fit(prior: Prior, alt: dict, n: int, name: str, info: dict, C: dict, pred: Optional[tuple], area_mm2: float) -> ProfileFit:
    """고른 시작의 마무리된 맞춤 (C) → λ (pred 가 있으면 심 위치 사전 λ) → ProfileFit."""
    f, p_g = C["f"], C["p_g"]
    lnn = math.log(n)
    lam_free = f["lnL"] - alt["lnL"] - 0.5 * (p_g - alt["p"]) * lnn
    lam, seam = lam_free, float("nan")
    if pred is not None:
        # H1 = 이 심의 그루브 (융합 상태의 위치 사전 N(r_pred, Σ_R)), H0 = 이 심이 아님 = 대안 ∪ 창 어디엔가 있는 그루브 모양 (위치 균일,
        # 창 넓이 A_w = 센서 법칙의 사다리꼴). 루트 위치는 Laplace (가우스 우도 × 사전의 적분), 나머지 파라미터는 BIC.
        seam = _seam_term(f, *pred)
        Cr = np.asarray(f["cov"], dtype=float)[:2, :2]
        det = float(np.linalg.det(Cr))
        occ = math.log(2.0 * math.pi * math.sqrt(det)) if det > 0 else -math.inf
        base = f["lnL"] - 0.5 * (p_g - 2) * lnn + occ
        lnZ_alt = alt["lnL"] - 0.5 * alt["p"] * lnn
        lam = (base + seam) - float(np.logaddexp(lnZ_alt, base - math.log(area_mm2)))
    return ProfileFit(theta=np.asarray(f["theta"], dtype=float), cov=np.asarray(f["cov"], dtype=float), log_bf=float(lam),
                      sigma_r=float(f["sigma"]), eps=float(f["eps"]), n_valid=n, converged=bool(f["converged"]),
                      face_inliers=np.asarray(f["face_inliers"], dtype=float), lnL_g=float(f["lnL"]),
                      lnL_alt=float(alt["lnL"]), p_alt=int(alt["p"]), log_post=float(f["log_post"]), start=name,
                      n_iter=int(f["n_iter"]), starts=dict(info, **C["extra"], rf_model=C["rf_model"]), p_g=p_g,
                      log_bf_free=float(lam_free), seam_term=seam, plate_inliers=C["plate_in"], plate_columns=C["plate_cols"],
                      beyond_plate=C["bplate"])


def finish_candidates(tpl: Template, prior: Prior, rays: Rays, alt: dict, fits: dict) -> dict:
    """심 위치 사전을 나중에 (이동 뒤, 재구성 자세의 예측으로) 적용할 행의 마무리: 시작마다 _finish_candidate. → 기록 (작다; 피클)."""
    return {"n": int(rays.n), "alt": {"lnL": float(alt["lnL"]), "p": int(alt["p"])}, "area": _window_area_mm2(rays),
            "raw": {nm: {"theta": np.asarray(f["theta"], dtype=float), "cov": np.asarray(f["cov"], dtype=float),
                         "log_post": float(f["log_post"]), "converged": bool(f["converged"])} for nm, f in fits.items()},
            "cand": {nm: _finish_candidate(tpl, prior, rays, alt, f) for nm, f in fits.items()}}


def finish_select(prior: Prior, rec: dict, pred: Optional[tuple]) -> ProfileFit:
    """`finish_candidates` 의 기록 + 심 위치 사전 (θ_pred, Σ_R) | None → `_finish_profile` 과 같은 ProfileFit (같은 선택, 같은 λ)."""
    name, info = _finish_choose(rec["raw"], pred)
    return _finish_fit(prior, rec["alt"], rec["n"], name, info, rec["cand"][name], pred, rec["area"])


def _beyond_extension(tpl: Template, prior: Prior, rays: Rays, f: dict, alt: dict) -> Optional[dict]:
    """템플릿 + 판 끝 너머 직선: 열 순서 (x) 로 단차 b 를 골라 한쪽은 템플릿, 다른 쪽은 대안의 2-직선 중 그쪽 직선 (같은 혼합 우도).
    BIC 이득이 (다시 맞추기 전에도) 있으면 안쪽 열만으로 템플릿을 다시 맞춘다 (다시 맞추기는 우도를 줄이지 않는다). → None 이면 이득 없음."""
    C = f["comp"]
    lt, _ = _estep(C, rays, f["sigma"], f["eps"])                       # 열별 템플릿 ln p
    order = np.argsort(rays.P[:, 0], kind="stable")
    n = rays.n
    best = None
    for side, line in (("left", alt["lines2"][0]), ("right", alt["lines2"][1])):
        ll = _ll_line(rays.P, rays.D, rays.W, line, alt.get("sigma2", f["sigma"]), alt.get("eps2", f["eps"]))
        a = ll[order]
        g = lt[order]
        if side == "left":                                               # 앞 b 열 = 너머
            cand = np.concatenate([[0.0], np.cumsum(a)]) + (np.sum(g) - np.concatenate([[0.0], np.cumsum(g)]))
        else:                                                            # 뒤 n − b 열 = 너머
            cand = np.concatenate([[0.0], np.cumsum(g)]) + (np.sum(a) - np.concatenate([[0.0], np.cumsum(a)]))
        b = int(np.argmax(cand))
        if best is None or cand[b] > best[0]:
            best = (float(cand[b]), side, b)
    tot, side, b = best
    if tot - float(np.sum(lt)) <= 1.5 * math.log(n):                    # BIC 이득 없음 (3 파라미터)
        return None
    out_idx = order[:b] if side == "left" else order[b:]
    in_mask = np.ones(n, dtype=bool)
    in_mask[out_idx] = False
    if np.count_nonzero(in_mask) < N_THETA or out_idx.size < 2:
        return None
    sub = Rays(P=rays.P[in_mask], D=rays.D[in_mask], rho=rays.rho[in_mask], W=rays.W[in_mask], o=rays.o,
               ang=rays.ang[in_mask], delta=rays.delta[in_mask], lo=rays.lo[in_mask], hi=rays.hi[in_mask])
    fi = _fit_template(tpl, prior, sub, f["theta"])
    Po, Do, Wo = rays.P[out_idx], rays.D[out_idx], rays.W[out_idx]
    c, nv, _ = _line_huber(Po)
    # 대안 (K=2) 과 같이 σ, ε 하나를 두 부분이 공유: 짧은 결합 EM (템플릿 θ 고정, 너머 직선 재맞춤)
    Cin = fi["comp"]
    sig, eps = fi["sigma"], fi["eps"]
    lnWo = np.log(Wo)
    ntot = sub.n + Po.shape[0]
    lnL = -math.inf
    for _ in range(_IT_EM):
        lt_in, g_in = _estep(Cin, sub, sig, eps)
        r_o = (Po - c) @ nv
        li_o = _log_in(r_o, np.abs(Do @ nv), sig, eps)
        lo_o = math.log(eps) - lnWo
        lnL_new = float(np.sum(lt_in)) + float(np.sum(np.logaddexp(li_o, lo_o)))
        g_o = 1.0 / (1.0 + np.exp(np.clip(lo_o - li_o, -700, 700)))
        sg = float(np.sum(g_in)) + float(np.sum(g_o))
        sig_new = max(math.sqrt((float(np.sum(g_in * Cin["r"] ** 2)) + float(np.sum(g_o * r_o * r_o))) / sg), _SIG_FLOOR_MM)
        eps_new = min(max(1.0 - sg / ntot, _EPS_CLIP), 1.0 - _EPS_CLIP)
        c, nv = _tls(Po, g_o * _huber_w(r_o / sig_new))
        done = abs(lnL_new - lnL) <= _SIG_TOL
        lnL = lnL_new
        sig, eps = sig_new, eps_new
        if done:
            break
    return {"fit": fi, "lnL": lnL, "side": side, "n_out": int(out_idx.size)}


# ======================================================================================================================
# 한 레그
# ======================================================================================================================
def _fit_range(ctx: dict, rng_: tuple) -> dict:
    """프로파일 구간 [i0, i1) 맞춤 (fork 자식이 부를 수 있는 평범한 함수). ctx: X, Z, V, tpls {sign: Template}, prior, predict, fan."""
    i0, i1 = int(rng_[0]), int(rng_[1])
    X, Z, Vd = ctx["X"], ctx["Z"], ctx["V"]
    prior, predict, fan = ctx["prior"], ctx["predict"], ctx["fan"]
    out = {}
    for sgn, tpl in ctx["tpls"].items():
        m = i1 - i0
        nf = len(tpl.face_names)
        res = {"theta": np.full((m, N_THETA), np.nan), "cov": np.full((m, N_THETA, N_THETA), np.nan),
               "log_bf": np.zeros(m), "sigma_r_mm": np.full(m, np.nan), "eps": np.full(m, np.nan),
               "n_valid": np.zeros(m, dtype=np.int64), "converged": np.zeros(m, dtype=bool),
               "face_inliers": np.zeros((m, nf)), "lnL_g": np.full(m, np.nan), "lnL_alt": np.full(m, np.nan),
               "p_alt": np.zeros(m, dtype=np.int64), "start_pred": np.zeros(m, dtype=bool), "n_iter": np.zeros(m, dtype=np.int64),
               "t_s": np.zeros(m), "p_g": np.zeros(m, dtype=np.int64), "rf_design": np.zeros(m, dtype=bool),
               "log_bf_free": np.zeros(m), "seam_used": np.zeros(m, dtype=bool), "plate_inliers": np.zeros((m, 2)),
               "plate_columns": np.zeros((m, 2)), "beyond_plate": np.full(m, -1, dtype=np.int64)}
        prev = None
        for j, i in enumerate(range(i0, i1)):
            t0 = time.perf_counter()
            sp = predict(i) if predict is not None else (prev if ctx.get("warm", True) else None)   # 레그를 따라 이전 θ (warm start)
            # predict(i) 는 θ0 또는 (θ0, Σ_R) — Σ_R 가 있으면 심 위치 사전 λ (fit_profile)
            f = fit_profile(X[i], Z[i], Vd[i], tpl, prior, start_pred=sp, fan=fan)
            res["theta"][j] = f.theta
            prev = f.theta if np.all(np.isfinite(f.theta)) else prev
            res["cov"][j] = f.cov
            res["log_bf"][j] = f.log_bf
            res["sigma_r_mm"][j] = f.sigma_r
            res["eps"][j] = f.eps
            res["n_valid"][j] = f.n_valid
            res["converged"][j] = f.converged
            res["face_inliers"][j] = f.face_inliers
            res["lnL_g"][j] = f.lnL_g
            res["lnL_alt"][j] = f.lnL_alt
            res["p_alt"][j] = f.p_alt
            res["start_pred"][j] = f.start == "pred"
            res["n_iter"][j] = f.n_iter
            res["p_g"][j] = f.p_g
            res["rf_design"][j] = f.starts.get("rf_model") == "design"
            res["log_bf_free"][j] = f.log_bf_free
            res["seam_used"][j] = math.isfinite(f.seam_term)
            res["plate_inliers"][j] = f.plate_inliers
            res["plate_columns"][j] = f.plate_columns
            res["beyond_plate"][j] = f.beyond_plate
            res["t_s"][j] = time.perf_counter() - t0
        out[sgn] = res
    return out


# ---- 프로파일 행 커널 (일괄, 행마다 순수 함수 — 2026-10-10 스트리밍) ------------------------------------------------------------
# 한 행 (프로파일) 의 맞춤 결과는 그 행의 자료 + 레그 문맥 (사전, 템플릿, 고정 너비) + 그 행의 두 번째 시작 (예측 θ0 [, Σ_R], 또는 이전
# 비어 있지 않은 행의 1차 맞춤 θ — warm start) 만의 함수다: 일괄 GN 의 묶음은 고정 너비 (프로파일의 열 수) 로 채워 어떤 행들과 한 묶음이
# 되든 바이트까지 같다 (groove_batch.rays_batch(width)). 그래서 레그를 어떻게 나누어 (자식 수·덩어리·녹화 중 스트림·이동 뒤) 계산하든
# 결과가 같다. 1단계 (A) = 자료만: 광선, 대안, 정합 필터 시작, 1차 일괄 맞춤 (탐색 시작). 2단계 (B) = 두 번째 시작의 일괄 맞춤 + 마무리.
ROW_KEYS = ("theta", "cov", "log_bf", "sigma_r_mm", "eps", "n_valid", "converged", "face_inliers", "lnL_g", "lnL_alt", "p_alt",
            "start_pred", "n_iter", "t_s", "p_g", "rf_design", "log_bf_free", "seam_used", "plate_inliers", "plate_columns", "beyond_plate")
_F1_KEYS = ("theta", "cov", "sigma", "eps", "lnL", "log_post", "face_inliers", "n_iter", "converged")


def _row_arrays(m: int, nf: int) -> dict:
    return {"theta": np.full((m, N_THETA), np.nan), "cov": np.full((m, N_THETA, N_THETA), np.nan),
            "log_bf": np.zeros(m), "sigma_r_mm": np.full(m, np.nan), "eps": np.full(m, np.nan),
            "n_valid": np.zeros(m, dtype=np.int64), "converged": np.zeros(m, dtype=bool),
            "face_inliers": np.zeros((m, nf)), "lnL_g": np.full(m, np.nan), "lnL_alt": np.full(m, np.nan),
            "p_alt": np.zeros(m, dtype=np.int64), "start_pred": np.zeros(m, dtype=bool), "n_iter": np.zeros(m, dtype=np.int64),
            "t_s": np.zeros(m), "p_g": np.zeros(m, dtype=np.int64), "rf_design": np.zeros(m, dtype=bool),
            "log_bf_free": np.zeros(m), "seam_used": np.zeros(m, dtype=bool), "plate_inliers": np.zeros((m, 2)),
            "plate_columns": np.zeros((m, 2)), "beyond_plate": np.full(m, -1, dtype=np.int64)}


def leg_context(spec: GrooveSpec, tol: Optional[Tolerances], *, width: int, prior: Optional[Prior] = None,
                bevel_x_sign: Optional[int] = None, fan: Optional[RayFan] = None, warm: bool = True, defer_seam: bool = False) -> dict:
    """레그 하나의 맞춤 문맥: 사전, 거울 템플릿 (bevel_x_sign None → 둘 다), 광선 법칙, 고정 너비 (프로파일 열 수), warm start 여부.
    defer_seam: 심 위치 사전 (재구성 자세의 예측) 을 행 맞춤 뒤에 적용한다 — 2단계는 시작마다 마무리 기록 (`finish_candidates`) 을
    남기고, `rows_select` 가 사전으로 고르고 λ 를 계산한다 (사전을 행 맞춤 안에서 적용한 것과 같은 값)."""
    fan = fan or RayFan.from_validity()
    prior = prior or profile_prior(spec, tol)
    signs = (int(bevel_x_sign),) if bevel_x_sign is not None else (1, -1)
    return {"tpls": {s: Template(spec, s, shear=prior.shear) for s in signs}, "prior": prior, "fan": fan, "width": int(width),
            "warm": bool(warm), "signs": signs, "defer_seam": bool(defer_seam)}


def row_counts(X, Z, V) -> np.ndarray:
    """행마다 광선 수 (유효하고 유한한 열 — `_rays_of` 의 n): n < 2 인 행은 맞춤이 없다 (빈 행)."""
    X, Z = np.asarray(X), np.asarray(Z)
    return np.count_nonzero(np.asarray(V, dtype=bool) & np.isfinite(X) & np.isfinite(Z), axis=-1).astype(np.int64)


def lookback(n_rows: np.ndarray, i: int) -> Optional[int]:
    """warm start 의 근원: 행 i 앞의 가장 가까운 비어 있지 않은 행 (n ≥ 2), 없으면 None."""
    prev = np.flatnonzero(np.asarray(n_rows[:int(i)]) >= 2)
    return int(prev[-1]) if prev.size else None


def lookbacks(n_rows: np.ndarray) -> np.ndarray:
    """모든 행의 `lookback` 한 번에: (N,) — 앞의 가장 가까운 비어 있지 않은 행, 없으면 −1."""
    n = np.asarray(n_rows)
    idx = np.where(n >= 2, np.arange(n.size), -1)
    run = np.maximum.accumulate(idx) if n.size else idx
    return np.concatenate([[-1], run[:-1]]).astype(np.int64) if n.size else np.zeros(0, dtype=np.int64)


def rows_phase_a(ctx: dict, X, Z, V, idx) -> dict:
    """1단계 (자료만): 행 idx (임의 순서·묶음) 마다 광선 수, 대안 (두 거울에 같다), 거울마다 정합 필터 시작과 1차 일괄 맞춤 (설계 척도
    시작). → {"idx", "n", "alt": [dict|None], "F1": {sign: {key: (m, …)}}} (빈 행 = NaN)."""
    from .groove_batch import fit_template_batch, rays_batch
    idx = [int(i) for i in idx]
    fan, prior = ctx["fan"], ctx["prior"]
    m = len(idx)
    n = np.zeros(m, dtype=np.int64)
    alts: list = [None] * m
    rays, keep = [], []
    for q, i in enumerate(idx):
        r = _rays_of(X[i], Z[i], V[i], fan)
        n[q] = r.n
        if r.n < 2:
            continue
        alts[q] = _alt_fit(r)
        rays.append(r)
        keep.append(q)
    F1 = {}
    for s, tpl in ctx["tpls"].items():
        out = {}
        if keep:
            ths = np.array([template_search_start(None, None, None, tpl, prior, fan=fan, _rays=r, _alt=alts[q])
                            for r, q in zip(rays, keep)])
            F = fit_template_batch(tpl, prior, rays_batch(rays, width=ctx["width"]), ths)
            for k_ in _F1_KEYS:
                v = np.asarray(F[k_])
                full = np.full((m,) + v.shape[1:], np.nan if v.dtype.kind == "f" else 0, dtype=v.dtype)
                full[keep] = v
                out[k_] = full
        F1[s] = out
    return {"idx": idx, "n": n, "alt": alts, "F1": F1}


def rows_phase_b(ctx: dict, X, Z, V, A: dict, starts: dict, *, t_a: Optional[dict] = None) -> dict:
    """2단계: 행 A['idx'] 마다 두 번째 시작 (starts[sign] = {q: θ0 | (θ0, Σ_R) | {"theta", "cov"}}, 없는 q = 시작 없음; Σ_R 가 있으면 심
    위치 사전 λ)
    의 일괄 맞춤 (σ 바닥 시작) → 사후가 큰 쪽 + 마무리 (`_finish_profile`). → {sign: ROW_KEYS 배열 (m, …)}. t_a: 행마다 1단계 시간."""
    from .groove_batch import fit_template_batch, rays_batch
    idx = A["idx"]
    fan, prior = ctx["fan"], ctx["prior"]
    m = len(idx)
    out = {}
    for s, tpl in ctx["tpls"].items():
        nf = len(tpl.face_names)
        res = _row_arrays(m, nf)
        t_all = time.perf_counter()
        rays, keep = [], []
        for q, i in enumerate(idx):
            if A["n"][q] < 2:
                f = _empty_fit(prior, int(A["n"][q]), nf)
                res["theta"][q], res["cov"][q], res["n_valid"][q] = f.theta, f.cov, int(A["n"][q])
                continue
            rays.append(_rays_of(X[i], Z[i], V[i], fan))
            keep.append(q)
        if keep:
            th_p = np.full((len(keep), N_THETA), np.nan)
            cov_p = {}
            st = starts.get(s) or {}
            defer = bool(ctx.get("defer_seam"))
            for p_, q in enumerate(keep):
                t_, c_ = parse_prediction(st.get(q))
                if t_ is not None:
                    th_p[p_] = t_
                    if c_ is not None and not defer:          # defer: 시작만 — 심 위치 사전은 rows_select 가
                        cov_p[p_] = (t_, c_)
            has_p = np.flatnonzero(np.all(np.isfinite(th_p), axis=1))
            F2 = None
            if has_p.size:
                F2 = fit_template_batch(tpl, prior, rays_batch([rays[p_] for p_ in has_p], width=ctx["width"]), th_p[has_p],
                                        sigma0=np.full(has_p.size, _SIG_FLOOR_MM))
            pos2 = {int(p_): j for j, p_ in enumerate(has_p)}
            t_batch = time.perf_counter() - t_all
            F1 = A["F1"][s]
            if defer:
                res["_cand"] = [None] * m
            for p_, q in enumerate(keep):
                t0 = time.perf_counter()
                fits = {"search": {k_: F1[k_][q] for k_ in _F1_KEYS}}
                if p_ in pos2:
                    fits["pred"] = {k_: F2[k_][pos2[p_]] for k_ in _F1_KEYS}
                if defer:
                    res["_cand"][q] = finish_candidates(tpl, prior, rays[p_], A["alt"][q], fits)
                    res["n_valid"][q] = int(rays[p_].n)
                    res["t_s"][q] = time.perf_counter() - t0 + t_batch / len(keep) + float((t_a or {}).get(q, 0.0))
                    continue
                f = _finish_profile(tpl, prior, rays[p_], A["alt"][q], fits, pred=cov_p.get(p_))
                _fill_row(res, q, f)
                res["t_s"][q] = time.perf_counter() - t0 + t_batch / len(keep) + float((t_a or {}).get(q, 0.0))
        out[s] = res
    return out


def _fill_row(res: dict, q: int, f: ProfileFit) -> None:
    """한 행의 ProfileFit → 행 배열 (ROW_KEYS, t_s 제외)."""
    res["theta"][q] = f.theta
    res["cov"][q] = f.cov
    res["log_bf"][q] = f.log_bf
    res["sigma_r_mm"][q] = f.sigma_r
    res["eps"][q] = f.eps
    res["n_valid"][q] = f.n_valid
    res["converged"][q] = f.converged
    res["face_inliers"][q] = f.face_inliers
    res["lnL_g"][q] = f.lnL_g
    res["lnL_alt"][q] = f.lnL_alt
    res["p_alt"][q] = f.p_alt
    res["start_pred"][q] = f.start == "pred"
    res["n_iter"][q] = f.n_iter
    res["p_g"][q] = f.p_g
    res["rf_design"][q] = f.starts.get("rf_model") == "design"
    res["log_bf_free"][q] = f.log_bf_free
    res["seam_used"][q] = math.isfinite(f.seam_term)
    res["plate_inliers"][q] = f.plate_inliers
    res["plate_columns"][q] = f.plate_columns
    res["beyond_plate"][q] = f.beyond_plate


def rows_select(rows: dict, idx, seam_pred: Optional[Callable]) -> dict:
    """defer_seam 행의 마무리: 기록 (`finish_candidates`) + 심 위치 사전 seam_pred(i) = (θ_pred, Σ_R) | θ | None → 고른 시작, λ →
    행 배열을 채운다 (제자리; 기록은 비운다). rows: {sign: ROW_KEYS 배열 + '_cand'} (행 순서 = idx)."""
    idx = [int(i) for i in idx]
    for s, res in rows.items():
        cands = res.get("_cand")
        if not cands:
            continue
        for q, rec in enumerate(cands):
            if rec is None:
                continue
            t0 = time.perf_counter()
            th, cv = parse_prediction(seam_pred(idx[q])) if seam_pred is not None else (None, None)
            f = finish_select(None, rec, (th, cv) if cv is not None else None)
            _fill_row(res, q, f)
            res["t_s"][q] = float(res["t_s"][q]) + time.perf_counter() - t0
        res["_cand"] = [None] * len(cands)
    return rows


def take_phase_a(A: dict, rows) -> dict:
    """1단계 결과 A 에서 행 rows (레그 행 번호, 그 순서) 만."""
    pos = {int(i): q for q, i in enumerate(A["idx"])}
    sel = [pos[int(i)] for i in rows]
    return {"idx": [int(i) for i in rows], "n": np.asarray(A["n"])[sel] if sel else np.zeros(0, dtype=np.int64),
            "alt": [A["alt"][q] for q in sel],
            "F1": {s: {k_: np.asarray(v)[sel] for k_, v in F.items()} for s, F in A["F1"].items()}}


def merge_phase_a(parts: list) -> dict:
    """1단계 결과 여러 개 → 하나 (행 번호가 겹치면 앞의 것; 행 결과는 묶음과 무관하므로 같은 값이다)."""
    parts = [P for P in parts if P is not None and len(P["idx"])]
    if not parts:
        return {"idx": [], "n": np.zeros(0, dtype=np.int64), "alt": [], "F1": {}}
    seen, rows, src = set(), [], []
    for k, P in enumerate(parts):
        for q, i in enumerate(P["idx"]):
            if int(i) not in seen:
                seen.add(int(i))
                rows.append(int(i))
                src.append((k, q))
    signs = parts[0]["F1"].keys()
    F1 = {}
    for s in signs:
        F = {}
        filled = [P for P in parts if P["F1"].get(s)]
        if filled:
            keys = filled[0]["F1"][s].keys()
            for k_ in keys:
                ref = np.asarray(filled[0]["F1"][s][k_])
                arr = np.full((len(rows),) + ref.shape[1:], np.nan if ref.dtype.kind == "f" else 0, dtype=ref.dtype)
                for j, (k, q) in enumerate(src):
                    if parts[k]["F1"].get(s):
                        arr[j] = np.asarray(parts[k]["F1"][s][k_])[q]
                F[k_] = arr
        F1[s] = F
    return {"idx": rows, "n": np.array([parts[k]["n"][q] for k, q in src], dtype=np.int64),
            "alt": [parts[k]["alt"][q] for k, q in src], "F1": F1}


def row_starts(ctx: dict, A: dict, rows, *, n_rows: Optional[np.ndarray] = None, predict: Optional[Callable] = None) -> dict:
    """행마다 두 번째 시작 → {sign: {q: start}} (q = rows 안의 위치): predict(i) 가 있으면 그것만 (심 추정의 예측), 아니면 (warm) 레그
    순서에서 앞의 비어 있지 않은 행 (`lookback`) 의 1차 맞춤 θ (A 에 있어야 한다)."""
    starts: dict = {s: {} for s in ctx["tpls"]}
    if predict is not None:
        for q, i in enumerate(rows):
            sp = predict(int(i))
            for s in starts:
                starts[s][q] = sp
        return starts
    if not ctx["warm"]:
        return starts
    pos = {int(i): q for q, i in enumerate(A["idx"])}
    for q, i in enumerate(rows):
        j = lookback(n_rows, int(i))
        if j is None:
            continue
        for s in starts:
            F = A["F1"].get(s)
            if F:
                th = np.asarray(F["theta"])[pos[j]]
                if np.all(np.isfinite(th)):
                    starts[s][q] = th
    return starts


def needs_lookback(ctx: dict, rows, n_rows: np.ndarray, predict: Optional[Callable] = None) -> list:
    """warm start 에 필요한데 rows 밖에 있는 앞 행들 (그 1단계도 계산해야 한다)."""
    if predict is not None or not ctx["warm"]:
        return []
    rs = {int(i) for i in rows}
    return sorted({j for j in (lookback(n_rows, int(i)) for i in rows) if j is not None} - rs)


def fit_rows(ctx: dict, X, Z, V, idx, *, n_rows: Optional[np.ndarray] = None, predict: Optional[Callable] = None,
             A: Optional[dict] = None) -> dict:
    """행 idx 의 맞춤 (1단계 + 2단계) → {sign: ROW_KEYS 배열}. 두 번째 시작은 `row_starts` (predict, 아니면 warm: 앞 비어 있지 않은 행의
    1차 맞춤 — 그 행이 idx 밖이면 여기서 그 행의 1단계도 계산한다; 행 결과는 묶음과 무관하므로 그 행이 자기 덩어리에서 얻는 것과
    같다). n_rows = 레그 전체 행의 광선 수 (`row_counts`). A: 이미 계산한 1단계 (스트림) — 모자란 행만 여기서."""
    idx = [int(i) for i in idx]
    if n_rows is None:
        n_rows = row_counts(X, Z, V)
    want = idx + needs_lookback(ctx, idx, n_rows, predict)
    have = set() if A is None else {int(i) for i in A["idx"]}
    todo = [i for i in want if i not in have]
    t0 = time.perf_counter()
    An = rows_phase_a(ctx, X, Z, V, todo) if todo else None
    t_a = (time.perf_counter() - t0) / max(len(todo), 1)
    A_all = merge_phase_a([A, An])
    starts = row_starts(ctx, A_all, idx, n_rows=n_rows, predict=predict)
    new = {i for i in todo}
    return rows_phase_b(ctx, X, Z, V, take_phase_a(A_all, idx), starts, t_a={q: (t_a if i in new else 0.0) for q, i in enumerate(idx)})


def _local_map(fn, chunks, n):
    return [fn(c) for c in chunks]


def _velocity(t, T) -> np.ndarray:
    p = np.asarray(T, dtype=float)[:, :3, 3]
    t = np.asarray(t, dtype=float)
    if p.shape[0] < 2 or not np.all(np.isfinite(t)):
        return np.zeros_like(p)
    tt = t.copy()
    if np.any(np.diff(tt) <= 0):                            # 같은 시각 (정지 grab 등) → 순번
        tt = np.arange(tt.size, dtype=float)
    return np.gradient(p, tt, axis=0)


def _side_bf(lnZg_p, lnZg_m, lnZalt) -> float:
    """런 수준 ln BF(+1 vs −1): 프로파일마다 '그루브 (그 쪽) 또는 그루브 없음' 의 증거 합 (같은 사전 확률)."""
    a = np.logaddexp(lnZg_p, lnZalt)
    b = np.logaddexp(lnZg_m, lnZalt)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sum(a[ok] - b[ok]))


def fit_leg(result, spec: GrooveSpec, tol: Optional[Tolerances], *, cls: str, direction: int, bevel_x_sign: Optional[int],
            predict: Optional[Callable] = None, n_children: int = 1, name: str = "leg", prior: Optional[Prior] = None,
            map_fn: Optional[Callable] = None, fan: Optional[RayFan] = None, warm: bool = True, batched: bool = True,
            seam_pred: Optional[Callable] = None) -> LegObs:
    """ScanResult 한 레그의 모든 프로파일 맞춤 → LegObs (§3.6, §8.3).

    predict: callable(i) → θ0 | (θ0, Σ_R) | None (심 추정 예측 시작). n_children > 1 이면 map_fn(fn, chunks, n) (작업 프로세스의 fork_map)
    로 구간을 나눠 맞춘다 (map_fn 없으면 이 프로세스에서 순서대로). bevel_x_sign None → 두 거울 템플릿을 모두 맞추고 런 수준 Bayes factor
    (D10) 로 고른다 (meta['bevel_side_ln_bf']). batched (기본): 행 커널 (`fit_rows`) — 결과가 행마다의 순수 함수라 덩어리·자식 수와
    무관하다 (warm start 는 레그 순서의 앞 비어 있지 않은 행, 덩어리 경계를 넘어서도).
    seam_pred: callable(i) → (θ_pred, Σ_R) — 주면 predict 는 두 번째 시작만 (θ0), 시작의 선택과 심 위치 λ 는 seam_pred (재구성 자세의
    예측) 로 행 맞춤 뒤에 (`rows_select`; batched 만). 빈 프로파일의 결측 증거도 seam_pred."""
    t_start = time.perf_counter()
    fan = fan or RayFan.from_validity()
    prior = prior or profile_prior(spec, tol)
    X = np.asarray(result.x_mm)
    Z = np.asarray(result.z_mm)
    Vd = np.asarray(result.valid, dtype=bool)
    N = int(X.shape[0])
    n_ch = max(1, int(n_children))
    bounds = np.linspace(0, N, min(n_ch, max(N, 1)) + 1).astype(int)
    chunks = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if batched:
        ctx = leg_context(spec, tol, width=int(X.shape[1]) if X.ndim == 2 else 0, prior=prior, bevel_x_sign=bevel_x_sign, fan=fan,
                          warm=warm, defer_seam=seam_pred is not None)
        n_rows = row_counts(X, Z, Vd)

        def fn(ch):
            return fit_rows(ctx, X, Z, Vd, range(ch[0], ch[1]), n_rows=n_rows, predict=predict)
    else:
        signs = (int(bevel_x_sign),) if bevel_x_sign is not None else (1, -1)
        ctx = {"X": X, "Z": Z, "V": Vd, "tpls": {s: Template(spec, s, shear=prior.shear) for s in signs}, "prior": prior,
               "predict": predict, "warm": bool(warm), "fan": fan, "signs": signs}

        def fn(ch):
            return _fit_range(ctx, ch)
    parts = (map_fn if (map_fn is not None and n_ch > 1) else _local_map)(fn, chunks, n_ch)
    merged = merge_rows(parts, ctx)
    if seam_pred is not None:
        rows_select(merged, range(N), seam_pred)
    return assemble_leg(result, spec, prior, ctx, merged, cls=cls, direction=direction, name=name,
                        predict=seam_pred if seam_pred is not None else predict, n_children=n_ch, fan=fan, t_start=t_start)


def merge_rows(parts: list, ctx: dict) -> dict:
    """행 결과 조각들 (레그 순서) → 하나 (ROW_KEYS 배열 이어 붙임, '_cand' 목록 이어 붙임)."""
    merged = {}
    for s in ctx["tpls"]:
        if not parts:
            merged[s] = {}
            continue
        merged[s] = {k: np.concatenate([p[s][k] for p in parts], axis=0) for k in parts[0][s] if k != "_cand"}
        if any("_cand" in p[s] for p in parts):
            merged[s]["_cand"] = [c for p in parts for c in (p[s].get("_cand") or [None] * len(p[s]["theta"]))]
    return merged


def assemble_leg(result, spec: GrooveSpec, prior: Prior, ctx: dict, merged: dict, *, cls: str, direction: int, name: str,
                 predict: Optional[Callable], n_children: int, fan: RayFan, t_start: float, meta_extra: Optional[dict] = None) -> LegObs:
    """레그 수준 마무리: 행 결과 (거울마다 레그 순서의 ROW_KEYS 배열) → 베벨 쪽 (D10, 두 거울이면 런 수준 BF), 빈 프로파일의 결측
    증거, meta, LegObs. 행을 어떻게 계산했든 (한 번에·덩어리·스트림) 같은 행 결과면 같은 LegObs (시간 meta 제외)."""
    X = np.asarray(result.x_mm)
    Vd = np.asarray(result.valid, dtype=bool)
    N = int(X.shape[0])
    signs = tuple(ctx["tpls"])
    ln_bf_side = None
    if len(signs) == 2 and N:
        lnn = np.log(np.maximum(merged[1]["n_valid"], 2))
        Zp = merged[1]["lnL_g"] - 0.5 * merged[1]["p_g"] * lnn
        Zm = merged[-1]["lnL_g"] - 0.5 * merged[-1]["p_g"] * lnn
        Za = merged[1]["lnL_alt"] - 0.5 * merged[1]["p_alt"] * lnn
        ln_bf_side = _side_bf(Zp, Zm, Za)
        chosen = 1 if ln_bf_side >= 0 else -1
    else:
        chosen = signs[0]
    r = merged.get(chosen, {})
    r, miss_meta = missing_return_evidence(r, ctx["tpls"][chosen], Vd, fan, predict) if N else (r, {})
    T = np.asarray(result.T_base_opt, dtype=float)
    t_acq = np.asarray(result.t_acq_s, dtype=float)
    tt = np.asarray(r.get("t_s", np.zeros(0)))
    meta = {"spec": spec.name, "kind": spec.kind, "prior": prior.to_dict(), "n_children": int(n_children), "n_profiles": N,
            "face_names": list(ctx["tpls"][chosen].face_names),
            "face_inliers": np.round(r["face_inliers"], 1).tolist() if N else [],
            "start_pred_frac": float(np.mean(r["start_pred"])) if N else 0.0,
            "rf_design_frac": float(np.mean(r["rf_design"])) if N else 0.0,
            # 판별 존재 (엇갈린 판 끝: 심 끝 = 두 판이 다 있는 곳; WP2 가 판마다 변화점): Σγ, 예측 보임 열 Σπ, 너머 직선이 가린 판
            "plate_inliers": np.round(r["plate_inliers"], 1).tolist() if N else [],
            "plate_columns": np.round(r["plate_columns"], 1).tolist() if N else [],
            "beyond_plate": r["beyond_plate"].astype(int).tolist() if N else [],
            "log_bf_kind": ("seam" if (N and bool(np.all(r["seam_used"]))) else ("mixed" if (N and bool(np.any(r["seam_used"])))
                                                                               else "free")),
            "log_bf_seam_used": r["seam_used"].astype(bool).tolist() if (N and bool(np.any(r["seam_used"]))) else [],
            "log_bf_free": np.round(r["log_bf_free"], 3).tolist() if (N and bool(np.any(r["seam_used"]))) else [],
            "n_iter_median": float(np.median(r["n_iter"])) if N else 0.0,
            "timing": {"wall_s": time.perf_counter() - t_start, "profile_ms_median": float(np.median(tt) * 1e3) if tt.size else 0.0,
                       "profile_ms_p95": float(np.percentile(tt, 95) * 1e3) if tt.size else 0.0,
                       "cpu_s": float(np.sum(tt))},
            "z_window_mm": list(fan.z_window_mm), "missing_returns": miss_meta}
    if ln_bf_side is not None:
        meta["bevel_side_ln_bf"] = ln_bf_side
    if getattr(result, "meta", None):
        for k_ in ("speed_mps", "standoff_m", "pitch_deg", "fps", "spacing_mm"):
            if k_ in result.meta:
                meta[k_] = result.meta[k_]
    if meta_extra:
        meta.update(meta_extra)
    return LegObs(name=name, cls=cls, direction=direction, t_acq_s=t_acq, T_base_opt=T, v_opt_mps=_velocity(t_acq, T),
                  theta=r.get("theta", np.zeros((0, N_THETA))), cov=r.get("cov", np.zeros((0, N_THETA, N_THETA))),
                  log_bf=r.get("log_bf", np.zeros(0)), sigma_r_mm=r.get("sigma_r_mm", np.zeros(0)), eps=r.get("eps", np.zeros(0)),
                  n_valid=r.get("n_valid", np.zeros(0)), converged=r.get("converged", np.zeros(0, dtype=bool)),
                  bevel_x_sign=chosen, meta=meta)


def column_rays(fan: RayFan, n_points: int) -> np.ndarray:
    """센서 열 광선의 단위 방향 (n_points, 2): 열 k (가운데 0, n_points < n_columns 이면 솎은 격자) 의 점은 x = k·dX(z) → o 에서 (k·b, 1)."""
    k = (np.arange(int(n_points)) - (int(n_points) - 1) / 2.0) * (float(fan.n_columns) / float(n_points))
    D = np.stack([k * fan.b, np.ones(int(n_points))], axis=-1)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def predicted_hits(tpl: Template, th, fan: RayFan, D: np.ndarray) -> np.ndarray:
    """θ 의 템플릿이 열 광선 D 에서 보정 z 창 안에 표면을 예측하는 열 (bool, (n,))."""
    g = tpl.geometry(np.asarray(th, dtype=float))
    t, face = _cast(g["seg0"], g["seg1"], fan.origin, D)
    with np.errstate(invalid="ignore"):
        z = fan.z0 + t * D[:, 1]
    zlo, zhi = fan.z_window_mm
    return np.isfinite(t) & (face >= 0) & (z > zlo) & (z < zhi)


def missing_return_evidence(r: dict, tpl: Template, V: np.ndarray, fan: RayFan, predict: Optional[Callable] = None) -> tuple:
    """빈 프로파일 (유효 열 없음) 의 증거 (spec r2 §3.3 / §12-5 'missing class'): 열 하나가 비는 확률 μ (그루브가 예측된 열이 빈 비율,
    이 레그의 모든 프로파일을 그루브 사후 확률 π_i = logistic(λ_i) 로 가중 — 문턱 없음), 프로파일 전체가 비는 확률 ν (시편 구간 =
    단일 구간 변화점 argmax Σλ 안의 빈 프로파일 비율; 둘 다 승계 규칙 (k + 1)/(n + 2) — 균등 Beta 사전). H_g (이 프로파일에 그루브 = 예측 θ (predict(i), 없으면 레그에서 가장 가까운 그루브
    맞춤) 의 템플릿이 창 안 n_pred 열에 표면) 에서 모두 빔: ν + (1 − ν)·μ^n_pred; H_alt 에는 '창 안에 아무 표면 없음' (열린 판 끝 너머) 이
    들어 있어 모두 빔 = 1. → λ_i = ln(ν + (1 − ν)·μ^n_pred) (≤ 0). 예전에는 0 (무증거) 이라 열린 판 끝 너머의 변화점이 평평했다
    (가상 팔 10-09: E 끝 읽기 σ 0.4–0.7 mm, S 0.2 mm). 그루브 프로파일·데이터가 있는 프로파일은 그대로. → (λ 고친 r, meta)."""
    lb = np.asarray(r["log_bf"], dtype=float).copy()
    nv = np.asarray(r["n_valid"])
    th = np.asarray(r["theta"], dtype=float)
    N = lb.shape[0]
    if N == 0:
        return r, {}
    D = column_rays(fan, V.shape[1])
    fin = np.all(np.isfinite(th), axis=1) & (nv > 0)
    empty = np.flatnonzero(nv == 0)
    # 그루브 사후 확률 π_i = logistic(λ_i) (λ 는 BIC/Laplace log BF, 같은 사전 확률) — 문턱 없음, 모든 프로파일의 가중
    w = np.where(fin, 0.5 * (1.0 + np.tanh(0.5 * np.clip(lb, -700.0, 700.0))), 0.0)
    meta = {"n_empty": int(empty.size), "w_groove": float(np.sum(w))}
    if empty.size == 0 or not np.any(fin):
        return r, meta
    n_pred = m_miss = 0.0
    for i in np.flatnonzero(w > 0.0):
        h = predicted_hits(tpl, th[i], fan, D)
        n_pred += w[i] * float(h.sum())
        m_miss += w[i] * float(np.sum(h & ~np.asarray(V[i], dtype=bool)))
    mu = (m_miss + 1.0) / (n_pred + 2.0)               # 계승 규칙 (균등 Beta 사전), 가중 수
    # 시편 구간 = 단일 구간 변화점 모형의 정확한 해 argmax_{a<b} Σ_[a,b) λ (§4.3 형식) — 빈 프로파일은 아직 λ = 0
    cs = np.concatenate([[0.0], np.cumsum(np.where(nv > 0, lb, 0.0))])
    run_min = np.minimum.accumulate(cs)
    arg_min = np.zeros(N + 1, dtype=int)
    for j in range(1, N + 1):
        arg_min[j] = arg_min[j - 1] if cs[arg_min[j - 1]] <= cs[j] else j
    gain = cs[1:] - run_min[:-1]
    hi = int(np.argmax(gain)) + 1
    lo = int(arg_min[hi - 1])
    n_in = hi - lo
    nu = (float(np.sum((empty >= lo) & (empty < hi))) + 1.0) / (n_in + 2.0)
    cand = np.flatnonzero(fin)
    inside = cand[(cand >= lo) & (cand < hi)]
    pool = inside if inside.size else cand
    scored = 0
    for i in empty:
        th_i = parse_prediction(predict(int(i)))[0] if predict is not None else None
        if th_i is None:
            th_i = th[pool[int(np.argmin(np.abs(pool - i)))]]
        k = int(predicted_hits(tpl, th_i, fan, D).sum())
        lb[i] = float(np.logaddexp(math.log(nu), math.log1p(-nu) + k * math.log(mu)))
        scored += 1
    meta.update(mu=mu, nu=nu, n_scored=scored, span=[lo, hi])
    out = dict(r)
    out["log_bf"] = lb
    return out, meta


def choose_bevel_side(result, spec: GrooveSpec, tol: Optional[Tolerances], *, stride: int = 1, prior: Optional[Prior] = None,
                      fan: Optional[RayFan] = None, map_fn: Optional[Callable] = None, n_children: int = 1) -> tuple:
    """베벨 쪽 (D10): 프로파일 (stride 간격) 마다 두 거울 템플릿과 대안을 맞추고 런 수준 ln BF 의 부호 → (bevel_x_sign, ln BF)."""
    fan = fan or RayFan.from_validity()
    prior = prior or profile_prior(spec, tol)
    idx = np.arange(0, int(np.asarray(result.x_mm).shape[0]), max(1, int(stride)))

    class _Sub:
        pass
    sub = _Sub()
    sub.x_mm = np.asarray(result.x_mm)[idx]
    sub.z_mm = np.asarray(result.z_mm)[idx]
    sub.valid = np.asarray(result.valid)[idx]
    sub.T_base_opt = np.asarray(result.T_base_opt)[idx]
    sub.t_acq_s = np.asarray(result.t_acq_s)[idx]
    sub.meta = {}
    leg = fit_leg(sub, spec, tol, cls="coarse", direction=1, bevel_x_sign=None, prior=prior, fan=fan, map_fn=map_fn,
                  n_children=n_children)
    return int(leg.bevel_x_sign), float(leg.meta.get("bevel_side_ln_bf", 0.0))


def bevel_x_sign_for(T_base_opt, w_base, bevel_sign_w: int) -> int:
    """심 틀의 베벨 쪽 (bevel_sign_w: +1 = +w) → 이 레그의 센서 x 쪽 (센서 X 축과 w 의 부호; 레그의 첫 자세 기준)."""
    T = np.asarray(T_base_opt, dtype=float)
    X = T[..., :3, 0].reshape(-1, 3).mean(axis=0)
    s = float(X @ np.asarray(w_base, dtype=float))
    return int(bevel_sign_w) * (1 if s >= 0 else -1)


# ======================================================================================================================
# 드웰 (정지 자세의 여러 프레임)
# ======================================================================================================================
@dataclass
class DwellFit:
    """정지 자세 드웰의 프로파일 맞춤 + 보정에 필요한 양 (모두 센서 레이저 평면, mm / rad; base 는 T_base_opt 가 있을 때)."""
    fit: ProfileFit
    bevel_x_sign: int
    side_ln_bf: float
    n_frames: int
    x_mm: np.ndarray                 # 맞춤에 쓴 열별 중앙값 프로파일
    z_mm: np.ndarray
    valid: np.ndarray
    temporal_sigma_mm: float         # 열별 프레임 간 산포 (MAD) 의 중앙값 — 정보 (σ_r 은 중앙값 프로파일의 잔차에서)
    root_x_mm: float
    root_z_mm: float
    root_cov_mm2: np.ndarray         # (2,2)
    standoff_mm: float               # 루트의 x 에서 두 판 상면 선의 range (z) 평균
    standoff_sigma_mm: float
    roll_rad: float                  # 두 판 상면 선의 센서 x 대비 기울기 평균 (z 가 x 와 함께 늘면 +)
    roll_sigma_rad: float
    top_angles_rad: tuple            # (L, R) 판 상면 선 기울기
    root_base_m: Optional[np.ndarray] = None
    root_base_cov_m2: Optional[np.ndarray] = None

    @property
    def log_bf(self) -> float:
        return self.fit.log_bf

    def to_dict(self) -> dict:
        d = {"bevel_x_sign": self.bevel_x_sign, "side_ln_bf": self.side_ln_bf, "n_frames": self.n_frames,
             "temporal_sigma_mm": self.temporal_sigma_mm, "root_x_mm": self.root_x_mm, "root_z_mm": self.root_z_mm,
             "root_sigma_mm": np.sqrt(np.maximum(np.diag(self.root_cov_mm2), 0)).tolist(), "standoff_mm": self.standoff_mm,
             "standoff_sigma_mm": self.standoff_sigma_mm, "roll_rad": self.roll_rad, "roll_sigma_rad": self.roll_sigma_rad,
             "top_angles_rad": list(self.top_angles_rad), "fit": self.fit.to_dict()}
        if self.root_base_m is not None:
            d["root_base_m"] = self.root_base_m.tolist()
            d["root_base_sigma_m"] = np.sqrt(np.maximum(np.diag(self.root_base_cov_m2), 0)).tolist()
        return d


def _median_profile(x, z, valid) -> tuple:
    """프레임 (F,N) → 열별 중앙값 (유효 프레임만) 과 열별 프레임 간 산포 (MAD). 한 프레임 (N,) 이면 그대로.

    열의 반환 여부도 중앙값 (센서 유령점 버스트 모형): 열은 grab 의 프레임 **과반** 에서 돌아왔을 때만 유효 (2·n_valid > F). 유령점은 한
    번에 많아야 g 프레임 연속 (sensor.ghost_burst_frames) 이고 grab 은 F ≥ 2g + 1 프레임 (호출자) 이라, 돌아오지 않는 열 (판 끝 너머·그루브
    틈) 에 생긴 버스트 하나는 과반이 될 수 없다 (예: 48/50 프레임에서 빠진 열의 2 프레임 유령점). 새 상수 없음."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    v = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(z)
    if x.ndim == 1:
        return x, z, v, 1, float("nan")
    xm = np.where(v, x, np.nan)
    zm = np.where(v, z, np.nan)
    any_v = 2 * v.sum(axis=0) > v.shape[0]                  # 열별 과반 (이름은 그대로: 아래 산포·반환이 쓰는 유효 열)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xmed = np.nanmedian(xm, axis=0)
            zmed = np.nanmedian(zm, axis=0)
            mad = MAD_K * np.nanmedian(np.abs(zm - zmed[None, :]), axis=0)
    ts = float(np.nanmedian(mad[any_v & (v.sum(axis=0) > 1)])) if np.any(any_v & (v.sum(axis=0) > 1)) else float("nan")
    return xmed, zmed, any_v, int(x.shape[0]), ts


def _fd_jac(fn, th, h=1e-6) -> np.ndarray:
    """스칼라 함수의 θ 유한차분 기울기 [R]."""
    th = np.asarray(th, dtype=float)
    g = np.zeros(N_THETA)
    for i in range(N_THETA):
        tp, tm = th.copy(), th.copy()
        tp[i] += h
        tm[i] -= h
        g[i] = (fn(tp) - fn(tm)) / (2 * h)
    return g


def fit_dwell(x_mm, z_mm, valid, spec: GrooveSpec, tol: Optional[Tolerances], *, bevel_x_sign: Optional[int] = None,
              prior: Optional[Prior] = None, T_base_opt=None, start_pred=None, fan: Optional[RayFan] = None) -> DwellFit:
    """드웰 (정지 자세 F 프레임, (F,N) 또는 이미 중앙값인 (N,)) 의 맞춤.

    잡음 모형 근거: 정지 장면의 스펙클은 프레임 사이에 상관 (Dorsch 1994) — 프레임 평균은 시간 잡음만 줄이고 지배 성분 (스펙클·모형
    어긋남) 은 남는다. 그래서 F 프레임을 독립으로 결합해 맞추면 정밀도를 √F 만큼 과대평가한다. 대신 열별 중앙값 프로파일 (간헐 반사·
    고스트 프레임에 강건) 하나를 fit_profile 로 맞추고, σ_r 은 그 프로파일의 잔차에서 EM 으로 추정한다 (남은 잡음 그대로).
    bevel_x_sign None → 두 거울 템플릿을 맞추고 증거 ln BF (D10 형식) 의 부호. 보정에 필요한 양: 루트 (x, z) 와 공분산, 스탠드오프
    (루트의 x 에서 판 상면 range), 판 기울기 (레이저 평면 안), base 의 루트 점 (T_base_opt 가 있으면)."""
    fan = fan or RayFan.from_validity()
    prior = prior or profile_prior(spec, tol)
    xm, zm, vm, nfr, ts = _median_profile(x_mm, z_mm, valid)
    signs = (int(bevel_x_sign),) if bevel_x_sign is not None else (1, -1)
    fits = {sg: fit_profile(xm, zm, vm, Template(spec, sg, shear=prior.shear), prior, start_pred=start_pred, fan=fan)
            for sg in signs}
    if len(signs) == 2:
        lnn = math.log(max(fits[1].n_valid, 2))
        za = fits[1].lnL_alt - 0.5 * fits[1].p_alt * lnn
        zp = fits[1].lnL_g - 0.5 * fits[1].p_g * lnn
        zn = fits[-1].lnL_g - 0.5 * fits[-1].p_g * lnn
        side_bf = float(np.logaddexp(zp, za) - np.logaddexp(zn, za)) if all(np.isfinite([zp, zn, za])) else 0.0
        sgn = 1 if side_bf >= 0 else -1
    else:
        sgn, side_bf = signs[0], float("nan")
    f = fits[sgn]
    tpl = Template(spec, sgn, shear=prior.shear)
    th, C = f.theta, f.cov
    names = tpl.face_names

    def top_angle(t, side):
        g = tpl.geometry(t)
        u = g["u"][names.index("top_" + side)]
        a = math.atan2(u[1], u[0])
        return a if abs(a) <= 0.5 * math.pi else a - math.copysign(math.pi, a)

    def top_z_at_root(t, side):
        g = tpl.geometry(t)
        k = names.index("top_" + side)
        A, u = g["A"][k], g["u"][k]
        return float(A[1] + (t[I_XR] - A[0]) * u[1] / u[0])

    def standoff(t):
        return 0.5 * (top_z_at_root(t, "L") + top_z_at_root(t, "R"))

    def roll(t):
        return 0.5 * (top_angle(t, "L") + top_angle(t, "R"))

    js, jr = _fd_jac(standoff, th), _fd_jac(roll, th)
    out = DwellFit(fit=f, bevel_x_sign=sgn, side_ln_bf=side_bf, n_frames=nfr, x_mm=xm, z_mm=zm, valid=vm,
                   temporal_sigma_mm=ts, root_x_mm=float(th[I_XR]), root_z_mm=float(th[I_ZR]), root_cov_mm2=C[:2, :2].copy(),
                   standoff_mm=standoff(th), standoff_sigma_mm=float(math.sqrt(max(js @ C @ js, 0.0))),
                   roll_rad=roll(th), roll_sigma_rad=float(math.sqrt(max(jr @ C @ jr, 0.0))),
                   top_angles_rad=(top_angle(th, "L"), top_angle(th, "R")))
    if T_base_opt is not None:
        T = np.asarray(T_base_opt, dtype=float)
        X, Zax, o = T[:3, 0], T[:3, 2], T[:3, 3]
        out.root_base_m = o + 1e-3 * (th[I_XR] * X + th[I_ZR] * Zax)
        B = 1e-3 * np.stack([X, Zax], axis=-1)
        out.root_base_cov_m2 = B @ C[:2, :2] @ B.T
    return out


# ======================================================================================================================
# base 프레임 산출
# ======================================================================================================================
def _leg_template(leg: LegObs) -> Template:
    """레그의 템플릿 (종류·베벨 쪽·전단 κ = 맞춤에 쓴 사전의 값; 꼭짓점 정의는 뒷댐재 유무와 무관)."""
    kind = leg.meta.get("kind", "single_bevel")
    shear = float((leg.meta.get("prior") or {}).get("shear", 0.0) or 0.0)
    nv = len(VERTEX_NAMES[kind])
    tpl = Template(GrooveSpec(kind=kind), leg.bevel_x_sign, shear=shear)
    assert len(tpl.vertex_names) == nv
    return tpl


def _plane_basis(leg: LegObs) -> tuple:
    T = np.asarray(leg.T_base_opt, dtype=float)
    return T[:, :3, 0], T[:, :3, 2], T[:, :3, 3]          # X, Z (센서 축, base), 원점 [m]


def root_points_base(leg: LegObs) -> tuple:
    """루트 점 R (gap 바닥 중심) 을 base 로 → ((N,3) [m], (N,3,3) [m²], 레이저 평면 안 계수 2)."""
    X, Zax, o = _plane_basis(leg)
    th = leg.theta
    P = o + 1e-3 * (th[:, I_XR, None] * X + th[:, I_ZR, None] * Zax)
    B = 1e-3 * np.stack([X, Zax], axis=-1)                 # (N,3,2)
    C2 = leg.cov[:, :2, :2]
    return P, np.einsum("nij,njk,nlk->nil", B, C2, B)


def vertices_base(leg: LegObs) -> tuple:
    """템플릿 유한 꼭짓점의 base 좌표 (N, nv, 3) [m] 와 θ 야코비안 (N, nv, 3, 10) [m / θ 단위]; 이름 = VERTEX_NAMES[kind]."""
    tpl = _leg_template(leg)
    X, Zax, o = _plane_basis(leg)
    N = leg.n
    nv = len(tpl.vertex_names)
    Vb = np.full((N, nv, 3), np.nan)
    Jb = np.full((N, nv, 3, N_THETA), np.nan)
    for i in range(N):
        if not np.all(np.isfinite(leg.theta[i])):
            continue
        g = tpl.geometry(leg.theta[i])
        V, dV = g["V"], g["dV"]
        Vb[i] = o[i] + 1e-3 * (V[:, 0:1] * X[i] + V[:, 1:2] * Zax[i])
        Jb[i] = 1e-3 * (dV[:, 0:1, :] * X[i][None, :, None] + dV[:, 1:2, :] * Zax[i][None, :, None])
    return Vb, Jb


def face_lines_base(leg: LegObs, u_base=None) -> dict:
    """템플릿 면 직선 → base: {FACE_NAMES 이름: (P (N,3) 면 위 점 [m], D (N,3) 단위 방향, cov (N,2,2) (법선 오프셋 [m], 각 [rad]))}.

    좌/우 (+w = n × u, n = 센서 쪽 판 법선 ≈ −Z_opt) 는 u_base (교시 S→E) 로 정한다. 없으면 레그의 이동 방향 × leg.direction."""
    tpl = _leg_template(leg)
    X, Zax, o = _plane_basis(leg)
    N = leg.n
    if u_base is None:
        v = np.nanmean(leg.v_opt_mps, axis=0)
        if not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
            raise ValueError("face_lines_base: leg has no travel; pass u_base")
        u_base = leg.direction * v / np.linalg.norm(v)
    u_base = np.asarray(u_base, dtype=float)
    nrm = -Zax.mean(axis=0)
    w = np.cross(nrm, u_base)
    x_is_left = float(X.mean(axis=0) @ w) > 0             # 센서 +x 가 심 왼쪽 (+w)?
    m = tpl.m
    out = {}
    for f, fname in enumerate(tpl.face_names):
        if fname.startswith("root_face"):
            continue                                      # 루트페이스는 치수 블록에서 (§4.6)
        if fname == "gap_bottom":
            name = "gap_bottom"
        else:
            base, side = fname.rsplit("_", 1)             # side L: 센서 x 쪽 = bevel_x_sign = −m
            sx = -m if side == "L" else m
            left = (sx > 0) == x_is_left
            name = {"top": "plate_top", "bevel": "bevel", "wall": "wall"}[base] + ("_left" if left else "_right")
        P = np.full((N, 3), np.nan)
        Dd = np.full((N, 3), np.nan)
        Cv = np.full((N, 2, 2), np.nan)
        for i in range(N):
            if not np.all(np.isfinite(leg.theta[i])):
                continue
            g = tpl.geometry(leg.theta[i])
            A, u, n_ = g["A"][f], g["u"][f], g["n"][f]
            P[i] = o[i] + 1e-3 * (A[0] * X[i] + A[1] * Zax[i])
            Dd[i] = u[0] * X[i] + u[1] * Zax[i]
            # (법선 오프셋, 각) 의 θ 야코비안: 오프셋 = n·A, 각 = atan2(u_z, u_x)
            # (법선 오프셋 n·A, 각 ψ = atan2(u_z, u_x)) 의 θ 야코비안 (n 의 회전은 오프셋 기준점 A 에서 0 이 되도록 A 기준)
            Joff = 1e-3 * (n_ @ g["dA"][f])
            du_ = g["du"][f]
            Jang = u[0] * du_[1] - u[1] * du_[0]
            Jm = np.vstack([Joff, Jang])
            Cv[i] = Jm @ leg.cov[i] @ Jm.T
        out[name] = (P, Dd, Cv)
    return out
