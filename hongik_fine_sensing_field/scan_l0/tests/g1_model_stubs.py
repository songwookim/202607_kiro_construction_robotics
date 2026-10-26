"""시험 전용 대역 (WP3): WP1 groove_model / WP2 seam_estimate / WP4 g1_outputs 가 오기 전까지 Teach1G 흐름을 가상 팔에서 끝까지 돌리는
같은 서명의 간단한 모형 (G1_CORE_SPEC §8.3). 정식 모형이 아니다 — 문턱이 있는 단순 판독이며 실기·출력 품질 판단에 쓰지 않는다.
제품 코드에는 고르는 장치가 없다: 시험이 서버 쪽 ``g1_worker.models`` 와 작업 프로세스의 시작 코드 ``G1Worker.CHILD_CODE`` 를
monkeypatch 한다 (``use(monkeypatch, names)`` — 자식은 ``install(names)`` 를 먼저 부르고 같은 _child_main 을 돈다).

G1_STUB_END_SCATTER_MM: 정밀 구간마다 끝 읽기에 ±값 (구간 번호의 홀짝) 을 더한다 — §10.1 (k) '구간별 끝 산포' 의 대역.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from scan_core.g1_types import (EndEst, LegObs, SeamEstimate, StationEst, THETA_NAMES, atomic_write_bytes, jsonable,
                                taught_frame)

__all__ = ["fit_leg", "profile_prior", "estimate", "plan_quantities", "done_test", "sections_doc", "write_sections",
           "weld_plan_from_estimate", "write_outputs", "save_weld_plan", "retract_weld_plan", "STUB_FUNCS", "use", "install"]

_LBF = 10.0
# g1_worker.MODEL_FUNCS 중 이 모듈이 대신하는 것 (site_state 는 늘 정식 WP2)
STUB_FUNCS = ("fit_leg", "fit_dwell", "profile_prior", "profile_predictor", "estimate", "plan_quantities", "write_outputs",
              "save_weld_plan", "retract_weld_plan")


def _models_with(names, real_models):
    sel = tuple(names) if names else STUB_FUNCS
    bad = [n for n in sel if n not in STUB_FUNCS]
    if bad:
        raise ValueError(f"no stand-in for {bad} (stand-ins: {STUB_FUNCS})")

    def models():
        m = real_models()
        for fn in sel:
            setattr(m, fn, globals()[fn])
        return m
    return models


def install(names=None) -> None:
    """작업 프로세스 안 (CHILD_CODE 가 부른다): g1_worker.models 를 대역 섞인 것으로."""
    import scan_l0.g1_worker as gw
    gw.models = _models_with(names, gw.models)


def use(monkeypatch, names=None) -> None:
    """시험에서: 서버 쪽 models 와 작업 프로세스 시작 코드를 monkeypatch (names None = 모든 대역, 아니면 그 함수만)."""
    import scan_l0.g1_worker as gw
    sel = tuple(names) if names else None
    monkeypatch.setattr(gw, "models", _models_with(sel, gw.models))
    monkeypatch.setattr(gw.G1Worker, "CHILD_CODE", f"import sys; import g1_model_stubs as s; s.install({sel!r}); "
                                                   "from scan_l0.g1_worker import _child_main; _child_main(int(sys.argv[1]), int(sys.argv[2]))")  # no stream channel: stand-ins fit post hoc


# --------------------------------------------------------------------------------------------------------------------------
# WP1 대역: 프로파일마다 판 직선 + 깊은 점 (그루브 바닥)
# --------------------------------------------------------------------------------------------------------------------------
def _profile(x, z, D_mm, O_mm=20.0):
    """→ (found, x_R, z_R, alpha, depth, sigma_r) — 판 직선 (반복 최소제곱), 판 아래 D/2 넘게 깊은 점 = 그루브, 그 가운데 D·0.75 넘게
    깊은 점의 중앙값 = 루트. 양옆에 판 점이 있어야 그루브."""
    if x.size < 20:
        return False, np.nan, np.nan, np.nan, np.nan, np.nan
    keep = np.ones(x.size, dtype=bool)
    a = b = 0.0
    for _ in range(3):
        if keep.sum() < 10:
            return False, np.nan, np.nan, np.nan, np.nan, np.nan
        A = np.column_stack([np.ones(int(keep.sum())), x[keep]])
        a, b = np.linalg.lstsq(A, z[keep], rcond=None)[0]
        res = z - (a + b * x)
        keep = np.abs(res) < 1.5
    res = z - (a + b * x)
    sig = float(np.std(res[keep])) if keep.sum() > 2 else float("nan")
    bot = (res > 0.75 * D_mm) & (res < 1.25 * D_mm)                       # 그루브 바닥 (뒷댐재 상면) 깊이의 점
    if bot.sum() < 3:
        return False, np.nan, np.nan, float(math.atan(b)), np.nan, sig
    xR = float(np.median(x[bot]))
    g = (res > 0.5 * D_mm) & (np.abs(x - xR) < O_mm)
    left, right = keep & (x < x[g].min()), keep & (x > x[g].max())
    if max(left.sum(), right.sum()) < 10:                                  # 판이 적어도 한쪽
        return False, np.nan, np.nan, float(math.atan(b)), np.nan, sig
    zR = float(np.median(z[bot & (np.abs(x - xR) < O_mm)]))
    return True, xR, zR, float(math.atan(b)), float(zR - (a + b * xR)), sig


def profile_prior(spec, tol, *, c: float = 1.0, sigma_c: float = 0.0, phi_rad: float = 0.0, **_kw):
    return {"c": c, "sigma_c": sigma_c, "phi_rad": phi_rad}


def profile_predictor(est, T_base_opt, bevel_x_sign=None, spec=None, cls: str = "pass"):     # seam_estimate.profile_predictor 의 서명
    return lambda i: None


def fit_dwell(x_mm, z_mm, valid, spec, tol, *, bevel_x_sign=None, prior=None, T_base_opt=None, start_pred=None):
    """열별 중앙값 프로파일 하나의 판독 (WP1 fit_dwell 의 대역; 중앙값 프로파일은 정식과 같은 groove_model._median_profile — 열별 과반)."""
    from types import SimpleNamespace
    from scan_core.groove_model import _median_profile
    xm, zm, vm, _nf, _ts = _median_profile(x_mm, z_mm, valid)
    m = vm & np.isfinite(xm) & np.isfinite(zm)
    ok, xR, zR, al, dep, sg = _profile(xm[m], zm[m], float(spec.groove_depth_mm), float(spec.top_opening_mm))
    th = np.full(len(THETA_NAMES), np.nan)
    cv = np.zeros((len(THETA_NAMES), len(THETA_NAMES)))
    if ok:
        th[:] = [xR, zR, al, spec.root_gap_mm, math.radians(spec.bevel_angle_deg), spec.root_face_mm, dep, dep, 0.0, 0.0]
        cv[:] = np.diag([0.15 ** 2, 0.05 ** 2, 1e-6, 0.05 ** 2, 1e-4, 0.01, 0.05 ** 2, 0.05 ** 2, 1e-6, 1e-6])
    fit = SimpleNamespace(theta=th, cov=cv, log_bf=_LBF * 10.0 if ok else -_LBF, sigma_r=sg, eps=0.01, n_valid=int(m.sum()), converged=ok)
    return SimpleNamespace(fit=fit, bevel_x_sign=int(bevel_x_sign) if bevel_x_sign is not None else 1, side_ln_bf=float("nan"))


def fit_leg(result, spec, tol, *, cls: str, direction: int, bevel_x_sign=None, predict=None, n_children: int = 1, prior=None,
            map_fn=None, seam_pred=None) -> LegObs:
    X, Z, V = np.asarray(result.x_mm, float), np.asarray(result.z_mm, float), np.asarray(result.valid, bool)
    N = X.shape[0]
    D = float(spec.groove_depth_mm)
    theta = np.full((N, len(THETA_NAMES)), np.nan)
    cov = np.zeros((N, len(THETA_NAMES), len(THETA_NAMES)))
    lbf = np.full(N, -_LBF)
    sr = np.full(N, np.nan)
    nv = V.sum(axis=1).astype(np.int64)
    conv = np.zeros(N, dtype=bool)
    for i in range(N):
        m = V[i] & np.isfinite(X[i]) & np.isfinite(Z[i])
        ok, xR, zR, al, dep, s = _profile(X[i, m], Z[i, m], D, float(spec.top_opening_mm))
        sr[i] = s
        if ok:
            theta[i] = [xR, zR, al, spec.root_gap_mm, math.radians(spec.bevel_angle_deg), spec.root_face_mm, dep, dep, 0.0, 0.0]
            cov[i] = np.diag([0.15 ** 2, 0.05 ** 2, 1e-6, 0.05 ** 2, 1e-4, 0.01, 0.05 ** 2, 0.05 ** 2, 1e-6, 1e-6])
            lbf[i] = _LBF
            conv[i] = True
    T = np.asarray(result.T_base_opt, float)
    t = np.asarray(result.t_acq_s, float)
    v = np.gradient(T[:, :3, 3], t, axis=0) if N > 1 else np.zeros((N, 3))
    return LegObs(name="", cls=cls, direction=int(direction), t_acq_s=t, T_base_opt=T, v_opt_mps=v, theta=theta, cov=cov, log_bf=lbf,
                  sigma_r_mm=sr, eps=np.full(N, 0.01), n_valid=nv, converged=conv,
                  bevel_x_sign=int(bevel_x_sign) if bevel_x_sign is not None else 1, meta={"model": "g1_model_stubs"})


# --------------------------------------------------------------------------------------------------------------------------
# WP2 대역: 직선 최소제곱, 구간별 최대 부분합 끝, 단순 결합
# --------------------------------------------------------------------------------------------------------------------------
def _roots(leg: LegObs):
    ok = (leg.log_bf > 0) & np.all(np.isfinite(leg.theta[:, :2]), axis=1)
    P = np.full((leg.n, 3), np.nan)
    if ok.any():
        loc = np.column_stack([leg.theta[ok, 0], np.zeros(int(ok.sum())), leg.theta[ok, 1], np.full(int(ok.sum()), 1e3)]) * 1e-3
        P[ok] = np.einsum("nij,nj->ni", leg.T_base_opt[ok], loc)[:, :3]
    return ok, P


def _max_segment(vals):
    best, cur, a0, a, b = -np.inf, 0.0, 0, 0, -1
    for i, v in enumerate(vals):
        if cur <= 0.0:
            cur, a0 = v, i
        else:
            cur += v
        if cur > best:
            best, a, b = cur, a0, i
    return float(best), a, b


def estimate(legs: list, prior, **_kw) -> SeamEstimate:            # (map_fn / n_children of the worker: not used)
    f = taught_frame(prior.S0, prior.E0, prior.up)
    S0, u0, w0, n0 = f["S0"], f["u0"], f["w0"], f["n0"]
    k = float(prior.k)
    fine = [lg for lg in legs if lg.cls == "pass"]
    use = fine if any((lg.log_bf > 0).any() for lg in fine) else legs
    Ps = []
    for lg in use:
        ok_, P_ = _roots(lg)
        Ps.append(P_[ok_])
    P = np.vstack([p for p in Ps if p.size]) if any(p.size for p in Ps) else np.zeros((0, 3))
    if P.shape[0] < 3:
        return _empty(prior, f, legs)
    rel = P - S0
    s, a, b = rel @ u0, rel @ w0, rel @ n0
    A = np.column_stack([np.ones_like(s), s])
    pa, ra = np.linalg.lstsq(A, a, rcond=None)[:2]
    pb, rb = np.linalg.lstsq(A, b, rcond=None)[:2]
    res_a, res_b = a - A @ pa, b - A @ pb
    AtA = np.linalg.inv(A.T @ A)
    va, vb = max(float(np.var(res_a)), 1e-10), max(float(np.var(res_b)), 1e-10)
    cov = np.zeros((4, 4))
    cov[:2, :2], cov[2:, 2:] = AtA * va, AtA * vb
    d = u0 + pa[1] * w0 + pb[1] * n0
    u_hat = d / np.linalg.norm(d)
    p0 = S0 + pa[0] * w0 + pb[0] * n0
    # 판 법선: 판 직선 방향 (센서 평면) × û
    ns = []
    for lg in use:
        ok = (lg.log_bf > 0) & np.isfinite(lg.theta[:, 2])
        if ok.any():
            al = lg.theta[ok, 2]
            dd = np.einsum("nij,nj->ni", lg.T_base_opt[ok, :3, :3], np.column_stack([np.cos(al), np.zeros_like(al), np.sin(al)]))
            nn = np.cross(dd, u_hat)
            nn *= np.sign(nn @ n0)[:, None]
            ns.append(nn)
    n_pl = np.median(np.vstack(ns), axis=0) if ns else n0.copy()
    n_pl = n_pl - (n_pl @ u_hat) * u_hat
    n_pl /= np.linalg.norm(n_pl)
    # 끝: 구간마다 최대 부분합
    scatter = float(os.environ.get("G1_STUB_END_SCATTER_MM", "0") or 0.0) * 1e-3
    readings = {"S": [], "E": []}
    lbf_run = -np.inf
    fine_i = 0
    for lg in legs:
        if (lg.meta or {}).get("kind") == "dwell":
            continue                                                       # 드웰: 한 자리 — 끝 읽기 없음
        Y = lg.T_base_opt[:, :3, 1]
        o = lg.T_base_opt[:, :3, 3]
        den = Y @ d
        with np.errstate(divide="ignore", invalid="ignore"):
            sst = np.einsum("ni,ni->n", Y, o - p0) / den
        order = np.argsort(sst)
        ss, lb = sst[order], lg.log_bf[order]
        best, ia, ib = _max_segment(lb)
        if lg.cls != "pass":
            lbf_run = max(lbf_run, best)
        off = 0.0
        if lg.cls == "pass":
            fine_i += 1
            off = scatter * (1.0 if fine_i % 2 else -1.0)
        if ib < ia:
            continue
        if ia > 0:
            sp = abs(ss[ia] - ss[ia - 1])
            readings["S"].append({"leg": lg.name, "cls": lg.cls, "s": 0.5 * (ss[ia] + ss[ia - 1]) + off, "sigma": sp / math.sqrt(12)})
        if ib < len(ss) - 1:
            sp = abs(ss[ib + 1] - ss[ib])
            readings["E"].append({"leg": lg.name, "cls": lg.cls, "s": 0.5 * (ss[ib] + ss[ib + 1]) + off, "sigma": sp / math.sqrt(12)})
    he = float(prior.he_sigma_m)
    L_d = float(prior.spec.weld_length_mm) * 1e-3
    ends = {}
    for X in ("S", "E"):
        rd = [r_ for r_ in readings[X] if r_["cls"] == "pass"] or readings[X]
        if rd:
            vals = np.array([r_["s"] for r_ in rd])
            sc = np.array([r_["sigma"] for r_ in rd])
            n = len(rd)
            tau2 = float(np.var(vals, ddof=1)) if n >= 2 else he ** 2
            if rd[0]["cls"] != "pass":
                tau2 = max(tau2, he ** 2)
            sig = math.sqrt(float(np.mean(sc ** 2)) / n + tau2 / n)
            ends[X] = [float(np.mean(vals)), sig, True, n, rd]
        else:
            ends[X] = None
    for X, Y_ in (("S", "E"), ("E", "S")):
        if ends[X] is None and ends[Y_] is not None:
            ends[X] = [ends[Y_][0] + (-L_d if X == "S" else L_d), ends[Y_][1] + 1e-3, False, 0, []]
    if ends["S"] is None:                                                  # 끝을 본 구간이 없다: 본 루트 구간을 설계 길이로 (σ = 남은 길이)
        s_obs = ((P - S0) @ u0)
        lo, hi = float(np.min(s_obs)), float(np.max(s_obs))
        R = max(0.0, L_d - (hi - lo))
        sR = max(R, 1e-3) / math.sqrt(12.0)                                 # 끝 ≈ 본 구간 너머 R 위의 균등 (spec r2 §4.4)
        ends = {"S": [lo - 0.5 * R, sR, False, 0, []], "E": [hi + 0.5 * R, sR, False, 0, []]}
    est_ends = {X: EndEst(s_m=v[0], sigma_m=v[1], point_m=_r(S0, u0, w0, n0, pa, pb, v[0]), two_sided=v[2], n_legs=v[3], readings=v[4])
                for X, v in ends.items()}
    # 단면
    stations = []
    step, win = float(prior.station_step_m), float(prior.station_window_m)
    eS, eE = est_ends["S"].s_m, est_ends["E"].s_m
    deps = []
    for lg in use:
        ok, Pl = _roots(lg)
        deps.append((((Pl[ok] - S0) @ u0), lg.theta[ok, 6]))
    s_all = np.concatenate([q[0] for q in deps]) if deps else np.zeros(0)
    d_all = np.concatenate([q[1] for q in deps]) if deps else np.zeros(0)
    tops = []
    s_list = list(np.arange(eS, eE, step)) + [eE]
    for sj in s_list:
        sj = float(sj)
        msk = np.abs(s_all - sj) <= 0.5 * win
        nprof = int(msk.sum())
        g = np.array([1.0, sj])
        sl = math.sqrt(max(float(g @ cov[:2, :2] @ g), 0.0))
        sn = math.sqrt(max(float(g @ cov[2:, 2:] @ g), 0.0))
        dep = float(np.median(d_all[msk])) if nprof else float("nan")
        dm = d_all[msk]
        sdep = (1.4826 * float(np.median(np.abs(dm - np.median(dm)))) / math.sqrt(nprof) + 0.01) if nprof > 1 else \
            (0.05 if nprof else float("inf"))
        dims = {"gap_mm": {"value": float(prior.spec.root_gap_mm), "sigma": 0.05 if nprof else float("inf"), "sigma_sys": 0.0, "n": nprof},
                "depth_mm": {"value": dep, "sigma": sdep, "sigma_sys": 0.0, "n": nprof},
                "bevel_angle_deg": {"value": float(prior.spec.bevel_angle_deg), "sigma": 0.2 if nprof else float("inf"), "sigma_sys": 0.0,
                                    "n": nprof},
                "top_opening_mm": {"value": float(prior.spec.top_opening_mm), "sigma": 0.05 if nprof else float("inf"),
                                   "sigma_sys": 0.0, "n": nprof},
                "hi_lo_mm": {"value": 0.0, "sigma": 0.05, "sigma_sys": 0.0, "n": nprof},
                "root_face_mm": {"value": float(prior.spec.root_face_mm), "sigma": 0.05, "sigma_sys": 0.0, "n": nprof}}
        root = _r(S0, u0, w0, n0, pa, pb, sj)
        stations.append(StationEst(s_m=float(sj), root_m=root, root_sigma_m=(sl, sn), u=u_hat, n=n_pl, w=np.cross(n_pl, u_hat),
                                   dims=dims, faces={}, n_profiles=nprof))
        if math.isfinite(dep):
            tops.append(root + n_pl * dep * 1e-3)
    top = max(tops, key=lambda p: float(p @ n0)) if tops else S0
    est = SeamEstimate(groove_found=bool(2.0 * lbf_run > k * k), log_bf_run=float(lbf_run), line_model="line",
                       line_params=np.array([pa, pb]), line_cov=cov, u=u_hat, n_plate=n_pl,
                       plate={"n_left": n_pl, "n_right": n_pl, "roll_deg": math.degrees(math.atan2(float(n_pl @ w0), float(n_pl @ n0))),
                              "misalignment_deg": 0.0, "top_point_m": top, "sigma_top_m": 5e-5},
                       ends=est_ends, dt_s=float(prior.dt_s), dt_sigma_s=float(prior.dt_sigma_s),
                       class_offset_m=({"transit": [0.0, 0.0, 0.0], "pass": [0.0, 0.0, 0.0]} if fine else {"transit": [0.0, 0.0, 0.0]}),
                       leg_eta_m={}, sigma_zeta_m={"pass": math.sqrt(va)}, tau_end_m=float("nan"), bevel_sign_w=1, stations=stations,
                       ratios={}, done=False, binding=[], unresolved=[], frame0=f)
    est.done, est.ratios, est.unresolved = done_test(est, prior.tol, k)
    est.binding = sorted(est.ratios, key=lambda q: -est.ratios[q])[:3]
    return est


def _r(S0, u0, w0, n0, pa, pb, s):
    return S0 + s * u0 + (pa[0] + pa[1] * s) * w0 + (pb[0] + pb[1] * s) * n0


def _empty(prior, f, legs):
    S0 = f["S0"]
    e = {X: EndEst(s_m=0.0, sigma_m=float("inf"), point_m=S0, two_sided=False, n_legs=0) for X in ("S", "E")}
    lbf = max((_max_segment(lg.log_bf)[0] for lg in legs if lg.cls != "pass"), default=-np.inf)
    return SeamEstimate(groove_found=bool(2.0 * lbf > prior.k ** 2), log_bf_run=float(lbf), line_model="line",
                        line_params=np.zeros((2, 2)), line_cov=np.eye(4), u=f["u0"], n_plate=f["n0"], plate={}, ends=e, dt_s=0.0,
                        dt_sigma_s=0.0, class_offset_m={}, leg_eta_m={}, sigma_zeta_m={}, tau_end_m=float("nan"), bevel_sign_w=1,
                        stations=[], ratios={}, done=False, binding=[], unresolved=["ends"], frame0=f)


def done_test(est: SeamEstimate, tol, k: float) -> tuple:
    r = {}
    for X in ("S", "E"):
        r[f"end_{X}"] = float(k) * float(est.ends[X].sigma_m) / (float(tol.end_position_mm) * 1e-3)
    lat = [float(k) * st.root_sigma_m[0] / (tol.root_position_mm * 1e-3) for st in est.stations]
    nor = [float(k) * st.root_sigma_m[1] / (tol.root_position_mm * 1e-3) for st in est.stations]
    r["root_lateral"] = max(lat) if lat else float("inf")
    r["root_normal"] = max(nor) if nor else float("inf")
    for nm in ("gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm"):
        t = tol.for_dim(nm)
        if t is None:
            continue
        vals = [float(k) * float(st.dims[nm]["sigma"]) / t for st in est.stations]
        r[nm] = max(vals) if vals else float("inf")
    unresolved = [q for q, v in r.items() if not v <= 1.0]
    return (not unresolved), r, unresolved


def plan_quantities(est: SeamEstimate, prior) -> dict:
    """WP2 plan_quantities 와 같은 키 (e_X_m, point_X_m, sigma_e_X_m, sigma_plan_X_m, psi_res_rad, depth_m / D_hat_mm)."""
    measured = "pass" in (est.class_offset_m or {})
    he = 0.0 if measured else float(prior.he_sigma_m)
    deps = [st.dims["depth_mm"]["value"] for st in est.stations if math.isfinite(st.dims["depth_mm"]["value"])]
    D = float(np.median(deps)) * 1e-3 if deps else float(prior.spec.groove_depth_mm) * 1e-3
    s_dir = math.sqrt(max(float(est.line_cov[1, 1]), 0.0) + max(float(est.line_cov[3, 3]), 0.0))
    out = {"u": est.u, "n": est.n_plate, "pass_class_measured": measured, "psi_res_rad": float(prior.k) * s_dir, "D_hat_mm": D * 1e3,
           "depth_m": D}
    for X in ("S", "E"):
        e = est.ends[X]
        out[f"e_{X}_m"], out[f"point_{X}_m"], out[f"sigma_e_{X}_m"] = float(e.s_m), e.point_m, float(e.sigma_m)
        out[f"sigma_plan_{X}_m"] = math.hypot(float(e.sigma_m), he)
    return out


# --------------------------------------------------------------------------------------------------------------------------
# WP4 대역: 최소 sections + weld_plan (scan_core.weld_plan.plan_weld + check_plan_contract)
# --------------------------------------------------------------------------------------------------------------------------
def sections_doc(est: SeamEstimate, prior, run: dict) -> dict:
    return jsonable({"schema": "g1core.sections/1", "base_frame": "link0", "units": {"position": "m", "dims": "mm"}, "k": prior.k,
                     "status": run.get("status"), "done": {"resolved": est.done, "binding": est.binding, "unresolved": est.unresolved},
                     "ends": {X: {"point_m": e.point_m, "s_m": e.s_m, "sigma_m": e.sigma_m, "two_sided": e.two_sided}
                              for X, e in est.ends.items()},
                     "stations": [st.to_dict() for st in est.stations], "notes": ["g1_model_stubs (test stand-in)"]})


def write_sections(doc: dict, out_dir) -> list:
    out = Path(out_dir)
    atomic_write_bytes(out / "sections.json", json.dumps(doc, indent=1).encode())
    import io
    buf = io.BytesIO()
    st = doc["stations"]
    np.savez(buf, schema_version=np.array(101), s_m=np.array([x["s_m"] for x in st]),
             root_point_m=np.array([x["root_m"] for x in st]).reshape(-1, 3))
    atomic_write_bytes(out / "sections.npz", buf.getvalue())
    return [str(out / "sections.json"), str(out / "sections.npz")]


def weld_plan_from_estimate(est: SeamEstimate, spec, process, *, base_frame: str, gravity_axis_base, k: float, design=None, torch=None):
    from scan_core.seam_frame import seam_frame
    from scan_core.weld_plan import GrooveStation, PlanContractError, check_plan_contract, plan_weld
    S, E = est.ends["S"], est.ends["E"]
    frame = seam_frame(S.point_m, E.point_m, est.n_plate, sigma_S_m=S.sigma_m, sigma_E_m=E.sigma_m)
    n = np.asarray(est.n_plate, float)
    u = frame.u
    w = np.cross(n, u)
    sts = [GrooveStation(s_mm=(st.s_m - S.s_m) * 1e3, origin=np.asarray(st.root_m, float), u=u, n=n, w=w, spec=spec,
                         n_profiles=int(st.n_profiles)) for st in est.stations]
    plan = plan_weld(sts, process, spec_design=spec, base_frame=base_frame, position="1G", gravity_axis_base=gravity_axis_base,
                     design=design, torch=torch)
    viol, checked = check_plan_contract(plan, frame, k=k)
    if viol:
        raise PlanContractError("; ".join(viol))
    plan.ends = {**frame.to_dict(), "checked": checked}
    return plan


def write_outputs(est, prior, process, out_dir, *, base_frame="link0", gravity_axis_base=None, k=None, run=None, design=None,
                  torch=None) -> dict:
    """g1_outputs.write_outputs 와 같은 계약 (단면 + 계획 → out_dir; 계약 거부면 단면만 쓰고 다시 던진다)."""
    files = write_sections(sections_doc(est, prior, dict(run or {})), out_dir)
    plan = weld_plan_from_estimate(est, prior.spec, process, base_frame=base_frame, gravity_axis_base=gravity_axis_base,
                                   k=float(prior.k if k is None else k), design=design, torch=torch)
    files += save_weld_plan(plan, out_dir)
    return {"files": files, "plan": plan, "plan_summary": plan.summary_lines(), "contract": plan.ends.get("checked")}


def save_weld_plan(plan, out_dir) -> list:
    import shutil
    import tempfile
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".weld_plan.", dir=str(out))
    try:
        dst = []
        for f in plan.save(tmp):
            os.replace(f, out / Path(f).name)
            dst.append(str(out / Path(f).name))
        return dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def retract_weld_plan(out_dir) -> list:
    out = Path(out_dir)
    moved = []
    for nm in ("weld_plan.json", "weld_plan.npz"):
        if (out / nm).exists():
            (out / "verification").mkdir(parents=True, exist_ok=True)
            os.replace(out / nm, out / "verification" / nm)
            moved.append(str(out / "verification" / nm))
    return moved
