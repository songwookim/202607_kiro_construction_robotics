"""1G 프로파일 모형의 일괄 Gauss–Newton (G1_CORE_SPEC §12-9, WP1): `groove_model._fit_template` 과 같은 알고리즘 (담금질 혼합 EM →
열 발자국 혼합 EM, 같은 척도의 직선 탐색, 경계 활성 집합, 뉴턴 감소량 정지) 을 프로파일 축으로 벡터화한다. 프로파일마다 상태 (θ, σ_r,
ε, 단계, 반복 수) 를 따로 가지며, 한 반복에서 아직 끝나지 않은 프로파일 모두를 한 번에 계산한다 (파이썬 호출 오버헤드를 나눠 낸다).

성분 (열 × 보이는 면) 은 프로파일 색인을 가진 평평한 배열, 프로파일별 합은 bincount, 정규 방정식은 (프로파일, 면) 별 3×3 모멘트,
풀이는 (B, 10, 10) 일괄. 결과는 프로파일별 `_fit_template` 과 같은 딕셔너리 키 (comp 제외)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import groove_model as gm
from .groove_model import (I_AL, I_BE, I_G, I_GL, I_GR, I_HL, I_HR, I_RF, I_XR, I_ZR, MAD_K, N_THETA, Prior, Rays, Template,
                           _COS_FLOOR, _EPS_CLIP, _FAR_MM, _HALVE_MAX, _IT_EM, _IT_HUBER, _LOG_SQRT_2PI, _Q_TOL, _RIDGE,
                           _SIG_FLOOR_MM, _SIG_TOL, _STEP_TOL, _huber_w)

__all__ = ["RaysB", "rays_batch", "geometry_b", "fit_template_batch"]


# ======================================================================================================================
# 묶음 광선
# ======================================================================================================================
@dataclass
class RaysB:
    P: np.ndarray        # (B,N,2)
    D: np.ndarray        # (B,N,2)
    W: np.ndarray        # (B,N)
    ang: np.ndarray
    delta: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    valid: np.ndarray    # (B,N) bool (채움 열 = False)
    n: np.ndarray        # (B,) 유효 열 수
    o: np.ndarray        # (2,)
    lnW: np.ndarray = None   # (B,N) ln W

    def __post_init__(self):
        if self.lnW is None:
            self.lnW = np.log(self.W)

    def take(self, idx) -> "RaysB":
        """프로파일 부분 집합."""
        return RaysB(P=self.P[idx], D=self.D[idx], W=self.W[idx], ang=self.ang[idx], delta=self.delta[idx], lo=self.lo[idx],
                     hi=self.hi[idx], valid=self.valid[idx], n=self.n[idx], o=self.o, lnW=self.lnW[idx])

    @property
    def B(self) -> int:
        return int(self.P.shape[0])

    @property
    def N(self) -> int:
        return int(self.P.shape[1])


def rays_batch(rays_list: list, width: Optional[int] = None) -> RaysB:
    """광선 묶음 (B, N): 채움 열은 무효. width = 고정 너비 N (프로파일의 열 수) — 주면 한 프로파일의 맞춤 결과가 어떤 프로파일들과 한
    묶음이 되든 바이트까지 같다 (행 합의 쌍별 합산 나무가 너비에 달려 있다: 너비가 '묶음 안 최대 광선 수' 면 묶음 구성에 따라 반올림이
    달라진다). None = 묶음 안 최대 광선 수 (예전)."""
    B = len(rays_list)
    N = max(1, max(r.n for r in rays_list))
    if width is not None:
        if int(width) < N:
            raise ValueError(f"rays_batch: width {int(width)} < the largest ray count {N}")
        N = int(width)
    P = np.zeros((B, N, 2))
    D = np.zeros((B, N, 2))
    D[..., 1] = 1.0
    W = np.ones((B, N))
    ang = np.zeros((B, N))
    delta = np.zeros((B, N))
    valid = np.zeros((B, N), dtype=bool)
    for b, r in enumerate(rays_list):
        k = r.n
        P[b, :k], D[b, :k], W[b, :k], ang[b, :k], delta[b, :k] = r.P, r.D, r.W, r.ang, r.delta
        valid[b, :k] = True
    lo = np.where(valid, ang - 0.5 * delta, np.inf)            # 채움 열은 어떤 각 구간에도 들지 않는다
    hi = np.where(valid, ang + 0.5 * delta, -np.inf)
    return RaysB(P=P, D=D, W=W, ang=ang, delta=delta, lo=lo, hi=hi, valid=valid,
                 n=np.array([r.n for r in rays_list], dtype=int), o=rays_list[0].o.copy())


# ======================================================================================================================
# 묶음 템플릿 기하 (Template._canon / geometry 의 배열판)
# ======================================================================================================================
def _canon_b(tpl: Template, TH: np.ndarray):
    G, be, rf, hL, hR, gL, gR = (TH[:, I_G], TH[:, I_BE], TH[:, I_RF], TH[:, I_HL], TH[:, I_HR], TH[:, I_GL], TH[:, I_GR])
    g2 = 0.5 * G
    z = np.zeros_like(G)
    cL, sL = np.cos(gL), np.sin(gL)
    cR, sR = np.cos(gR), np.sin(gR)
    hpi = 0.5 * math.pi
    RL = (-g2, z, {I_G: -0.5}, {})
    RR = (g2, z, {I_G: 0.5}, {})

    def bevel_L():
        Bb = be + gL
        tB = np.tan(Bb)
        sec2 = 1.0 + tB * tB
        a_ = hL - rf * cL
        FL = (-g2 - rf * sL, rf * cL, {I_G: -0.5, I_RF: -sL, I_GL: -rf * cL}, {I_RF: cL, I_GL: -rf * sL})
        CL = (-g2 - rf * sL - a_ * tB, hL,
              {I_G: -0.5, I_BE: -a_ * sec2, I_RF: -sL + cL * tB, I_HL: -tB, I_GL: -rf * cL - rf * sL * tB - a_ * sec2},
              {I_HL: 1.0})
        return CL, FL

    k = tpl.kind
    if k == "single_bevel":
        CL, FL = bevel_L()
        CR = (g2 - hR * sR / cR, hR, {I_G: 0.5, I_HR: -sR / cR, I_GR: -hR / (cR * cR)}, {I_HR: 1.0})
        verts = [CL, FL, RL, RR, CR]
        faces = [("top_L", 0, 1, 1, gL, {I_GL: 1.0}),
                 ("bevel_L", 1, 2, 1, be + gL - hpi, {I_BE: 1.0, I_GL: 1.0}),
                 ("root_face_L", 2, 3, 3, gL - hpi, {I_GL: 1.0}),
                 ("gap_bottom", 3, 4, 3, z, {}),
                 ("wall_R", 4, 5, 4, hpi + gR, {I_GR: 1.0}),
                 ("top_R", 5, 6, 5, gR, {I_GR: 1.0})]
    elif k == "single_v":
        CL, FL = bevel_L()
        Bp = be - gR
        tBp = np.tan(Bp)
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
                 ("gap_bottom", 3, 4, 3, z, {}),
                 ("root_face_R", 4, 5, 4, hpi + gR, {I_GR: 1.0}),
                 ("bevel_R", 5, 6, 6, hpi - be + gR, {I_BE: -1.0, I_GR: 1.0}),
                 ("top_R", 6, 7, 6, gR, {I_GR: 1.0})]
    else:
        CL = (-g2 - hL * sL / cL, hL, {I_G: -0.5, I_HL: -sL / cL, I_GL: -hL / (cL * cL)}, {I_HL: 1.0})
        CR = (g2 - hR * sR / cR, hR, {I_G: 0.5, I_HR: -sR / cR, I_GR: -hR / (cR * cR)}, {I_HR: 1.0})
        verts = [CL, RL, RR, CR]
        faces = [("top_L", 0, 1, 1, gL, {I_GL: 1.0}),
                 ("wall_L", 1, 2, 1, gL - hpi, {I_GL: 1.0}),
                 ("gap_bottom", 2, 3, 2, z, {}),
                 ("wall_R", 3, 4, 3, hpi + gR, {I_GR: 1.0}),
                 ("top_R", 4, 5, 4, gR, {I_GR: 1.0})]
    if not tpl.backing:
        faces = [f for f in faces if f[0] != "gap_bottom"]
    return verts, faces


def geometry_b(tpl: Template, TH: np.ndarray) -> dict:
    """`Template.geometry` 의 묶음판: 모든 배열의 첫 축 = 프로파일."""
    TH = np.asarray(TH, dtype=float)
    B = TH.shape[0]
    xR, zR, al = TH[:, I_XR], TH[:, I_ZR], TH[:, I_AL]
    m = tpl.m
    kap = tpl.shear
    ca, sa = np.cos(al), np.sin(al)
    e1x, e1z, e2x, e2z = ca, sa, sa, -ca
    verts, faces = _canon_b(tpl, TH)
    nv = len(verts)
    V = np.empty((B, nv, 2))
    dV = np.zeros((B, nv, 2, N_THETA))
    for i, (xi, eta, dxi, deta) in enumerate(verts):
        a1 = m * xi + kap * eta
        V[:, i, 0] = xR + a1 * e1x + eta * e2x
        V[:, i, 1] = zR + a1 * e1z + eta * e2z
        dV[:, i, 0, I_XR] = 1.0
        dV[:, i, 1, I_ZR] = 1.0
        dV[:, i, 0, I_AL] = -a1 * e1z + eta * e1x
        dV[:, i, 1, I_AL] = a1 * e1x + eta * e1z
        for q, v in dxi.items():
            dV[:, i, 0, q] += m * v * e1x
            dV[:, i, 1, q] += m * v * e1z
        for q, v in deta.items():
            dV[:, i, 0, q] += kap * v * e1x + v * e2x
            dV[:, i, 1, q] += kap * v * e1z + v * e2z
    nf = len(faces)
    u = np.empty((B, nf, 2))
    n = np.empty((B, nf, 2))
    dn = np.zeros((B, nf, 2, N_THETA))
    du = np.zeros((B, nf, 2, N_THETA))
    A = np.empty((B, nf, 2))
    dA = np.empty((B, nf, 2, N_THETA))
    for f, (name, a, b, anc, phi, dphi) in enumerate(faces):
        cp, sp = np.cos(phi), np.sin(phi)
        b1 = m * cp + kap * sp
        ux, uz = b1 * e1x + sp * e2x, b1 * e1z + sp * e2z
        L2 = ux * ux + uz * uz
        L = np.sqrt(L2)
        ux, uz = ux / L, uz / L
        u[:, f, 0], u[:, f, 1] = ux, uz
        nx, nz = m * uz, -m * ux
        n[:, f, 0], n[:, f, 1] = nx, nz
        du[:, f, 0, I_AL], du[:, f, 1, I_AL] = -uz, ux
        dn[:, f, 0, I_AL], dn[:, f, 1, I_AL] = -nz, nx
        for q, v in dphi.items():
            rate = -m / L2 * v
            du[:, f, 0, q] += rate * -uz
            du[:, f, 1, q] += rate * ux
            dn[:, f, 0, q] += -(v / L2) * ux
            dn[:, f, 1, q] += -(v / L2) * uz
        A[:, f] = V[:, anc - 1]
        dA[:, f] = dV[:, anc - 1]
    far_l = V[:, 0] - _FAR_MM * u[:, 0]
    far_r = V[:, -1] + _FAR_MM * u[:, -1]
    ext = np.concatenate([far_l[:, None], V, far_r[:, None]], axis=1)
    ia = np.array([f[1] for f in faces])
    ib = np.array([f[2] for f in faces])
    return {"V": V, "dV": dV, "seg0": ext[:, ia], "seg1": ext[:, ib], "A": A, "dA": dA, "u": u, "n": n, "dn": dn, "du": du,
            "names": tuple(f[0] for f in faces)}


def _cast_b(seg0, seg1, o, D):
    """광선마다 자기 프로파일의 선분들 (M,K,2) 과의 첫 교점 → (t̂ (M,), 면 (M,), −1 = 없음)."""
    E = seg1 - seg0
    w = seg0 - o
    dx = D[:, 0:1]
    dz = D[:, 1:2]
    den = dx * E[..., 1] - dz * E[..., 0]
    tn = w[..., 0] * E[..., 1] - w[..., 1] * E[..., 0]
    sn = w[..., 0] * dz - w[..., 1] * dx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = tn / den
        s = sn / den
    ok = (den != 0.0) & (t > 0.0) & (s >= 0.0) & (s <= 1.0)
    t = np.where(ok, t, np.inf)
    k = np.argmin(t, axis=1)
    th = t[np.arange(t.shape[0]), k]
    return th, np.where(np.isfinite(th), k, -1)


# ======================================================================================================================
# 묶음 성분 · E 단계 · 정규 방정식 · 걸음
# ======================================================================================================================
def _components_b(tpl: Template, TH: np.ndarray, RB: RaysB, bi: np.ndarray) -> dict:
    """`groove_model._components` 의 묶음판 (프로파일 bi 의 θ = TH). 면이 하나뿐인 열은 (프로파일 × 열) 격자에 그대로 (판 상면은 방송,
    그루브 창의 열만 모음), 발자국에 꼭짓점이 든 열의 조각만 평평한 배열 (열별로 연속)."""
    g = geometry_b(tpl, TH)
    o = RB.o
    Bs = TH.shape[0]
    N = RB.N
    nfc = g["seg0"].shape[1]
    Vp = g["V"] - o
    r2 = Vp[..., 0] ** 2 + Vp[..., 1] ** 2
    av = np.arctan2(Vp[..., 0], Vp[..., 1])                                    # (Bs,nv)
    dV = g["dV"]
    dav = (Vp[..., 1:2] * dV[..., 0, :] - Vp[..., 0:1] * dV[..., 1, :]) / r2[..., None]
    nv = av.shape[1]
    lo = RB.lo[bi]
    hi = RB.hi[bi]
    valid = RB.valid[bi]
    P = RB.P[bi]
    Dd = RB.D[bi]
    amin = av.min(axis=1)[:, None]
    amax = av.max(axis=1)[:, None]
    left = valid & (hi <= amin)
    right = valid & (lo >= amax)
    mid = valid & ~left & ~right
    # 바깥 판 상면: 먼 끝의 광선 각이 작은 쪽이 왼쪽 (거울 부호와 무관 — `groove_model._outer_faces`)
    a0 = np.arctan2(g["seg0"][:, 0, 0] - o[0], g["seg0"][:, 0, 1] - o[1])
    a1 = np.arctan2(g["seg1"][:, -1, 0] - o[0], g["seg1"][:, -1, 1] - o[1])
    fl = np.where(a0 < a1, 0, nfc - 1)
    fr = (nfc - 1) - fl
    kC = np.where(left, fl[:, None], np.where(right, fr[:, None], -1))
    nf_ = g["n"]
    A = g["A"]
    nA = np.einsum("bkc,bkc->bk", nf_, A)
    # 판 상면 (첫·끝 면) 의 잔차·입사 — 격자 전체에 방송
    Px, Pz = P[..., 0], P[..., 1]
    Dx, Dz = Dd[..., 0], Dd[..., 1]
    bb = np.arange(Bs)
    nL, nR = nf_[bb, fl], nf_[bb, fr]
    aL, aR = nA[bb, fl][:, None], nA[bb, fr][:, None]
    r1 = np.where(left, nL[:, 0:1] * Px + nL[:, 1:2] * Pz - aL, nR[:, 0:1] * Px + nR[:, 1:2] * Pz - aR)
    c1 = np.abs(np.where(left, nL[:, 0:1] * Dx + nL[:, 1:2] * Dz, nR[:, 0:1] * Dx + nR[:, 1:2] * Dz))
    mb, mj = np.nonzero(mid)
    amb_m = np.zeros((Bs, N), dtype=bool)
    if mb.size:
        # 첫 면은 꼭짓점 광선 각에서만 바뀐다 → 정렬된 꼭짓점 각 사이 구간마다 광선 하나 (구간 중점) 로 면을 정하고 열은 구간의 면
        avs = np.sort(av, axis=1)                                                # (Bs,nv)
        midang = 0.5 * (avs[:, 1:] + avs[:, :-1])                               # (Bs,nv−1)
        Dmid = np.stack([np.sin(midang), np.cos(midang)], axis=-1).reshape(-1, 2)
        rep = np.repeat(np.arange(Bs), nv - 1)
        _, kint = _cast_b(g["seg0"][rep], g["seg1"][rep], o, Dmid)
        kint = kint.reshape(Bs, nv - 1)
        angm = RB.ang[bi][mb, mj]
        iv = np.sum(angm[:, None] > avs[mb], axis=1) - 1                       # 0 .. nv−2 (mid 열은 꼭짓점 각 범위 안)
        km = kint[mb, np.clip(iv, 0, nv - 2)]
        kC[mb, mj] = km
        amb_m[mb, mj] = np.any((lo[mb, mj, None] < av[mb]) & (av[mb] < hi[mb, mj, None]), axis=1)
        kk = np.maximum(km, 0)
        nm = nf_[mb, kk]
        r1[mb, mj] = nm[:, 0] * Px[mb, mj] + nm[:, 1] * Pz[mb, mj] - nA[mb, kk]
        c1[mb, mj] = np.abs(nm[:, 0] * Dx[mb, mj] + nm[:, 1] * Dz[mb, mj])
    one = valid & ~amb_m & (kC >= 0)
    kC = np.where(one, kC, -1)
    ab, aj = np.nonzero(amb_m)
    pba = np.zeros(0, dtype=int)
    ja = np.zeros(0, dtype=int)
    fa = np.zeros(0, dtype=int)
    pia = np.zeros(0)
    dpi_amb = np.zeros((0, N_THETA))
    starts = np.zeros(0, dtype=int)
    if ab.size:
        loa = lo[ab, aj]
        hia = hi[ab, aj]
        ava = av[ab]
        inside = (ava > loa[:, None]) & (ava < hia[:, None])
        Bv = np.where(inside, ava, hia[:, None])
        ids = np.where(inside, np.arange(nv)[None, :], -1)
        Bfull = np.concatenate([loa[:, None], Bv, hia[:, None]], axis=1)
        Ifull = np.concatenate([np.full((ab.size, 1), -1), ids, np.full((ab.size, 1), -1)], axis=1)
        srt = np.argsort(Bfull, axis=1, kind="stable")
        Bs_ = np.take_along_axis(Bfull, srt, axis=1)
        Is = np.take_along_axis(Ifull, srt, axis=1)
        length = Bs_[:, 1:] - Bs_[:, :-1]
        mids = 0.5 * (Bs_[:, 1:] + Bs_[:, :-1])
        r0, c0 = np.nonzero(length > 0)
        mm = mids[r0, c0]
        Dm_ = np.stack([np.sin(mm), np.cos(mm)], axis=-1)
        _, kmm = _cast_b(g["seg0"][ab[r0]], g["seg1"][ab[r0]], o, Dm_)
        ok = kmm >= 0
        rr, cc, fa = r0[ok], c0[ok], kmm[ok]
        dl = RB.delta[bi][ab, aj]
        pba = ab[rr]
        ja = aj[rr]
        pia = length[rr, cc] / dl[rr]
        dav0 = np.concatenate([dav, np.zeros((Bs, 1, N_THETA))], axis=1)       # 색인 −1 → nv (0 행)
        i1 = np.where(Is[rr, cc + 1] < 0, nv, Is[rr, cc + 1])
        i0 = np.where(Is[rr, cc] < 0, nv, Is[rr, cc])
        dpi_amb = (dav0[pba, i1] - dav0[pba, i0]) / dl[rr][:, None]
        if rr.size:
            newg = np.ones(rr.size, dtype=bool)
            newg[1:] = rr[1:] != rr[:-1]
            starts = np.flatnonzero(newg)
    nka = nf_[pba, fa]
    xa, za = Px[pba, ja], Pz[pba, ja]
    ra = nka[:, 0] * xa + nka[:, 1] * za - nA[pba, fa]
    ca = np.abs(nka[:, 0] * Dx[pba, ja] + nka[:, 1] * Dz[pba, ja])
    dn = g["dn"]
    cf = -np.einsum("bkc,bkcp->bkp", A, dn) - np.einsum("bkc,bkcp->bkp", nf_, g["dA"])
    Dm = np.stack([dn[..., 0, :], dn[..., 1, :], cf], axis=2)                    # (Bs,K,3,10)
    return {"g": g, "kC": kC, "one": one, "r1": r1, "c1": c1, "x": Px, "z": Pz, "valid": valid, "lnW": RB.lnW[bi],
            "pba": pba, "ja": ja, "fa": fa, "pia": pia, "ra": ra, "ca": ca, "xa": xa, "za": za, "dpi_amb": dpi_amb,
            "starts": starts, "Dm": Dm, "Bs": Bs, "N": N, "K": nfc, "bi": bi}


def _estep_b(C: dict, RB: RaysB, sig: np.ndarray, eps: np.ndarray):
    """→ (열별 ln p (Bs,N), 하나뿐인 열의 γ (Bs,N), 조각 γ (Ma,), 프로파일별 Σ ln p (Bs,)) — 채움 열은 0."""
    one = C["one"]
    base = np.log1p(-eps) - np.log(sig) - _LOG_SQRT_2PI
    lc1 = base[:, None] + np.log(np.maximum(C["c1"], _COS_FLOOR)) - 0.5 * np.square(C["r1"] / sig[:, None])
    lin = np.where(one, lc1, -np.inf)
    pba = C["pba"]
    lca = None
    if pba.size:
        lca = (base[pba] + np.log(np.maximum(C["pia"], 1e-300)) + np.log(np.maximum(C["ca"], _COS_FLOOR))
               - 0.5 * np.square(C["ra"] / sig[pba]))
        st = C["starts"]
        lin[pba[st], C["ja"][st]] = np.logaddexp.reduceat(lca, st)
    lo = np.log(eps)[:, None] - C["lnW"]
    ltot = np.where(C["valid"], np.logaddexp(lin, lo), 0.0)
    gam1 = np.where(one, np.exp(lin - ltot), 0.0)
    gama = np.exp(lca - ltot[pba, C["ja"]]) if lca is not None else np.zeros(0)
    return ltot, gam1, gama, ltot.sum(axis=1)


def _hard_min_b(C: dict) -> np.ndarray:
    """조각 성분의 0/1 (열마다 |r| 이 가장 작은 조각); 하나뿐인 열은 늘 1."""
    st = C["starts"]
    a = np.abs(C["ra"])
    sel = np.zeros(a.size)
    if st.size:
        grp = np.zeros(a.size, dtype=int)
        grp[st[1:]] = 1
        grp = np.cumsum(grp)
        order = np.lexsort((a, grp))
        first = np.ones(order.size, dtype=bool)
        first[1:] = grp[order][1:] != grp[order][:-1]
        sel[order[first]] = 1.0
    return sel


def _group_median(vals: np.ndarray, grp: np.ndarray, G: int) -> np.ndarray:
    """그룹별 중앙값 (np.median 과 같은 짝수 규약); 빈 그룹은 NaN."""
    out = np.full(G, np.nan)
    if vals.size == 0:
        return out
    order = np.lexsort((vals, grp))
    v = vals[order]
    gsorted = grp[order]
    cnt = np.bincount(gsorted, minlength=G)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    has = cnt > 0
    lo_i = start + (cnt - 1) // 2
    hi_i = start + cnt // 2
    out[has] = 0.5 * (v[lo_i[has]] + v[hi_i[has]])
    return out


def _normal_eq_b(C: dict, w1, wa, wpi, prior: Prior, TH: np.ndarray):
    """½Σ w r² − Σ wpi ln π + 사전. w1: 하나뿐인 열 (Bs,N), wa: 조각 (Ma,), wpi: 조각 γ (Ma,) 또는 None."""
    Bs, K = C["Bs"], C["K"]
    one = C["one"]
    key1 = (np.arange(Bs)[:, None] * K + np.where(one, C["kC"], 0)).ravel()
    w1 = np.where(one, w1, 0.0).ravel()
    x1, z1, r1 = C["x"].ravel(), C["z"].ravel(), C["r1"].ravel()
    keya = C["pba"] * K + C["fa"]
    xa, za, ra = C["xa"], C["za"], C["ra"]
    Lk = Bs * K

    def mom(wt1, wta):
        return np.bincount(key1, wt1, Lk) + (np.bincount(keya, wta, Lk) if keya.size else 0.0)
    wx1, wz1, wxa, wza = w1 * x1, w1 * z1, wa * xa, wa * za
    S = [mom(wx1 * x1, wxa * xa), mom(wx1 * z1, wxa * za), mom(wx1, wxa), mom(wz1 * z1, wza * za), mom(wz1, wza), mom(w1, wa)]
    S = [v.reshape(Bs, K) for v in S]
    M = np.empty((Bs, K, 3, 3))
    M[..., 0, 0], M[..., 0, 1], M[..., 0, 2] = S[0], S[1], S[2]
    M[..., 1, 0], M[..., 1, 1], M[..., 1, 2] = S[1], S[3], S[4]
    M[..., 2, 0], M[..., 2, 1], M[..., 2, 2] = S[2], S[4], S[5]
    wr1, wra = w1 * r1, wa * ra
    bvec = np.stack([mom(wr1 * x1, wra * xa), mom(wr1 * z1, wra * za), mom(wr1, wra)], axis=1).reshape(Bs, K, 3)
    Dm = C["Dm"]
    H = np.sum(np.transpose(Dm, (0, 1, 3, 2)) @ (M @ Dm), axis=1) + np.diag(prior.prec)[None]
    grad = np.einsum("bkai,bka->bi", Dm, bvec) + prior.prec * np.where(prior.prec > 0, TH - prior.mu, 0.0)
    if wpi is not None and C["dpi_amb"].shape[0]:
        Dp = C["dpi_amb"]
        pi = np.maximum(C["pia"], 1e-300)
        a = wpi / pi
        np.add.at(H, C["pba"], Dp[:, :, None] * Dp[:, None, :] * (a / pi)[:, None, None])
        np.add.at(grad, C["pba"], -Dp * a[:, None])
    return H, grad


def _solve_b(H: np.ndarray, g: np.ndarray, act: np.ndarray):
    """프로파일마다 활성 파라미터만으로 풀이 (나머지는 단위 행렬·기울기 0) → (Δθ, 공분산 (비활성 0))."""
    Bs = H.shape[0]
    Hs = H.copy()
    inact = ~act
    Hs[np.broadcast_to(inact[:, :, None], Hs.shape) | np.broadcast_to(inact[:, None, :], Hs.shape)] = 0.0
    dg = np.abs(np.diagonal(Hs, axis1=1, axis2=2))
    rid = _RIDGE * np.maximum(np.max(np.where(act, dg, 0.0), axis=1), 1.0)
    idx = np.arange(N_THETA)
    Hs[:, idx, idx] += np.where(act, rid[:, None], 1.0)
    try:
        Ci = np.linalg.inv(Hs)
    except np.linalg.LinAlgError:
        Ci = np.empty_like(Hs)
        for b in range(Bs):
            try:
                Ci[b] = np.linalg.inv(Hs[b])
            except np.linalg.LinAlgError:
                Ci[b] = np.linalg.pinv(Hs[b])
    Ci = Ci * (act[:, :, None] & act[:, None, :])
    delta = -np.einsum("bij,bj->bi", Ci, np.where(act, g, 0.0))
    return delta, Ci


def _clip_b(prior: Prior, TH: np.ndarray) -> np.ndarray:
    th = np.minimum(np.maximum(np.asarray(TH, dtype=float), prior.lo), prior.hi)
    if prior.hi[I_RF] > prior.lo[I_RF]:
        top = np.maximum(np.minimum(th[:, I_HL], th[:, I_HR]), prior.lo[I_RF])
        th[:, I_RF] = np.minimum(np.maximum(th[:, I_RF], prior.lo[I_RF]), top)
    return th


def _log_prior_b(prior: Prior, TH: np.ndarray) -> np.ndarray:
    d = TH - prior.mu
    p = prior.prec
    return -0.5 * np.sum(p * np.where(p > 0, d * d, 0.0), axis=1)


def _step_b(H, grad, prior: Prior, TH):
    act = np.broadcast_to(prior.free, TH.shape).copy()
    delta, Cv = _solve_b(H, grad, act)
    for _ in range(N_THETA):
        lo_b = TH <= prior.lo + 1e-12
        hi_b = TH >= prior.hi - 1e-12
        hi_b[:, I_RF] |= TH[:, I_RF] >= np.minimum(TH[:, I_HL], TH[:, I_HR]) - 1e-12
        out = act & ((lo_b & (delta < 0)) | (hi_b & (delta > 0)))
        if not np.any(out):
            break
        act &= ~out
        rows = np.flatnonzero(out.any(axis=1))
        delta[rows], _ = _solve_b(H[rows], grad[rows], act[rows])
    return delta, Cv


# ======================================================================================================================
# 일괄 맞춤
# ======================================================================================================================
def fit_template_batch(tpl: Template, prior: Prior, RB: RaysB, TH0: np.ndarray, sigma0: Optional[np.ndarray] = None) -> dict:
    """프로파일 B 개를 `_fit_template` 과 같은 알고리즘으로 한꺼번에. sigma0 (B,) NaN = 설계 시작 (허용치 척도), 수 = 예측 시작
    (그 시작의 잔차 척도). → 배열 딕셔너리: theta (B,10), cov (B,10,10), sigma, eps, lnL, log_post, face_inliers (B,nf), n_iter,
    n_eval, converged, huber_converged."""
    B = RB.B
    free = prior.free
    th = _clip_b(prior, TH0)
    all_b = np.arange(B)
    n_it = np.zeros(B, dtype=int)
    n_eval = np.ones(B, dtype=int)
    it1 = np.zeros(B, dtype=int)
    it2 = np.zeros(B, dtype=int)
    conv1 = np.zeros(B, dtype=bool)
    conv2 = np.zeros(B, dtype=bool)
    phase = np.ones(B, dtype=int)
    nvalid = RB.n.astype(float)
    # 시작 척도
    C = _components_b(tpl, th, RB, all_b)
    sel_a = _hard_min_b(C) > 0
    vals = np.concatenate([np.abs(C["r1"][C["one"]]), np.abs(C["ra"][sel_a])])
    grp = np.concatenate([np.nonzero(C["one"])[0], C["pba"][sel_a]])
    med = _group_median(vals, grp, B)
    s_mad = np.where(np.isfinite(med), np.maximum(MAD_K * med, _SIG_FLOOR_MM), 1.0)
    len_sd = prior.sd[[I_G, I_RF, I_HL, I_HR]]
    len_sd = len_sd[np.isfinite(len_sd)]
    sig_design = np.maximum(float(np.max(len_sd)), s_mad) if len_sd.size else s_mad
    s0 = np.full(B, np.nan) if sigma0 is None else np.asarray(sigma0, dtype=float)
    sig = np.where(np.isfinite(s0), np.maximum(np.where(np.isfinite(s0), s0, 0.0), s_mad), sig_design)
    eps = np.full(B, 0.5)
    while True:
        A = np.flatnonzero(phase < 3)
        if A.size == 0:
            break
        thA = th[A]
        CA = _components_b(tpl, thA, RB, A)
        pba = CA["pba"]
        n_it[A] += 1
        p1 = phase[A] == 1
        sA, eA = sig[A].copy(), eps[A].copy()
        _, g1, ga, _ = _estep_b(CA, RB, sA, eA)
        sg = g1.sum(axis=1) + (np.bincount(pba, ga, A.size) if pba.size else 0.0)
        sr = (g1 * CA["r1"] ** 2).sum(axis=1) + (np.bincount(pba, ga * CA["ra"] ** 2, A.size) if pba.size else 0.0)
        sig_new = np.where(sg > 0, np.maximum(np.sqrt(sr / np.where(sg > 0, sg, 1.0)), _SIG_FLOOR_MM), sA)
        eps_new = np.clip(1.0 - sg / nvalid[A], _EPS_CLIP, 1.0 - _EPS_CLIP)
        s_ok = (np.abs(sig_new - sA) <= _SIG_TOL * sA) & (np.abs(eps_new - eA) * nvalid[A] <= 1.0 + 1e-9)
        sig_em = sig_new
        sA = np.where(p1, sA, sig_new)
        eA = eps_new
        _, g1, ga, Qs = _estep_b(CA, RB, sA, eA)
        Q0 = Qs + _log_prior_b(prior, thA)
        w1 = g1 * _huber_w(CA["r1"] / sA[:, None]) / (sA[:, None] ** 2)
        wa = ga * _huber_w(CA["ra"] / sA[pba]) / (sA[pba] ** 2) if pba.size else np.zeros(0)
        H, grad = _normal_eq_b(CA, w1, wa, ga, prior, thA)
        delta, Cv = _step_b(H, grad, prior, thA)
        sd = np.sqrt(np.maximum(np.diagonal(Cv, axis1=1, axis2=2), 1e-300))
        dec = -0.5 * np.einsum("bi,bi->b", grad, delta)
        lam = np.ones(A.size)
        accepted = np.zeros(A.size, dtype=bool)
        Qn = Q0.copy()
        thn_acc = thA.copy()
        pending = np.flatnonzero(dec > _Q_TOL)
        for _h in range(_HALVE_MAX):
            if pending.size == 0:
                break
            thn = _clip_b(prior, thA[pending] + lam[pending, None] * delta[pending])
            Cn = _components_b(tpl, thn, RB, A[pending])
            n_eval[A[pending]] += 1
            _, _, _, Qt = _estep_b(Cn, RB, sA[pending], eA[pending])
            Qt = Qt + _log_prior_b(prior, thn)
            Qn[pending] = Qt
            ok = Qt >= Q0[pending]
            acc = pending[ok]
            accepted[acc] = True
            thn_acc[acc] = thn[ok]
            lam[pending[~ok]] *= 0.5
            pending = pending[~ok]
        step = np.where(accepted[:, None], thn_acc - thA, 0.0)
        th[A] = np.where(accepted[:, None], thn_acc, thA)
        big = np.any((np.abs(step) > _STEP_TOL * sd) & free[None, :], axis=1)
        # ---- 1단계 (담금질) ----
        i1 = np.flatnonzero(p1)
        if i1.size:
            gi = A[i1]
            eps[gi] = eA[i1]
            it1[gi] += 1
            moved = accepted[i1] & big[i1] & ((Qn[i1] - Q0[i1]) > _Q_TOL)
            reached = (~moved) & (sA[i1] <= sig_em[i1] * (1.0 + _SIG_TOL))
            conv1[gi[reached]] = True
            halve = (~moved) & ~reached
            sig[gi[halve]] = np.maximum(0.5 * sA[i1][halve], sig_em[i1][halve])
            to2 = reached | (it1[gi] >= _IT_HUBER)
            g2 = gi[to2]
            if g2.size:                                         # 2단계로: σ = 지금 척도, ε = ½
                sig[g2] = np.maximum(sig[g2], _SIG_FLOOR_MM)
                eps[g2] = 0.5
                phase[g2] = 2
        # ---- 2단계 (EM) ----
        i2 = np.flatnonzero(~p1)
        if i2.size:
            gi = A[i2]
            sig[gi] = sA[i2]
            eps[gi] = eA[i2]
            it2[gi] += 1
            Qn2 = np.where(accepted[i2], Qn[i2], Q0[i2])
            small = (~big[i2]) | ((Qn2 - Q0[i2]) <= _Q_TOL)
            done = small & s_ok[i2]
            conv2[gi[done]] = True
            phase[gi[done | (it2[gi] >= _IT_EM)]] = 3
    # ---- 최종 ----
    C = _components_b(tpl, th, RB, all_b)
    _, g1, ga, Qs = _estep_b(C, RB, sig, eps)
    pba = C["pba"]
    H, _ = _normal_eq_b(C, g1 / (sig[:, None] ** 2), ga / (sig[pba] ** 2) if pba.size else np.zeros(0), ga, prior, th)
    _, Cov = _solve_b(H, np.zeros((B, N_THETA)), np.broadcast_to(prior.free, th.shape).copy())
    nf = C["K"]
    key1 = (np.arange(B)[:, None] * nf + np.where(C["one"], C["kC"], 0)).ravel()
    face_in = (np.bincount(key1, np.where(C["one"], g1, 0.0).ravel(), B * nf)
               + (np.bincount(pba * nf + C["fa"], ga, B * nf) if pba.size else 0.0)).reshape(B, nf)
    return {"theta": th, "cov": Cov, "sigma": sig, "eps": eps, "lnL": Qs, "log_post": Qs + _log_prior_b(prior, th),
            "face_inliers": face_in, "n_iter": n_it, "n_eval": n_eval, "converged": conv2, "huber_converged": conv1}
