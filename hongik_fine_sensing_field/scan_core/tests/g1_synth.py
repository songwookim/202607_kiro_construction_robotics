"""합성 SeamEstimate (g1_outputs 시험용): 곧은 심, 알려진 치수 (설계 10T/30°/갭 5, 개구 10.77), 교시 틀 대비 측면 기울기·법선 기울기,
판 롤. 스테이션은 끝 e_S 에서 e_E 까지 계약 좌표 5 mm 간격 (마지막 = e_E)."""
import math

import numpy as np

from scan_core.g1_types import EndEst, SeamEstimate, SeamPrior, StationEst, Precision, Tolerances
from scan_core.specimen import GrooveSpec

QC = {"root_gap_mm": {"tol": 0.5}, "bevel_angle_deg": {"tol": 2.0}, "groove_depth_mm": {"tol": 0.5},
      "hi_lo_mm": {"target_max": 1.0, "warn_above": 2.0}, "weld_length_mm": {"tol": 1.0, "ambiguous": 5.0},
      "top_opening_mm": {"tol": 0.5}, "root_position_mm": {"tol": 1.0}}


def synth_prior(spec=None, k=3.0) -> SeamPrior:
    spec = spec or GrooveSpec()
    return SeamPrior(S0=[0.90, -0.20, 0.05], E0=[0.92, 0.12, 0.05], up=[0, 0, 1], spec=spec, tol=Tolerances.from_qc(QC), k=k,
                     dt_s=0.0037, dt_sigma_s=0.001, rep_sigma_m=5e-5, he_sigma_m=0.002, sigma_eps=0.0,
                     station_step_m=0.005, station_window_m=0.010,
                     precision=Precision(end_position_mm=0.25, root_position_mm=0.2, length_mm=0.1))


def synth_estimate(prior: SeamPrior, *, length_m=0.380, a=(0.003, 0.02), b=(-0.010, 0.001), e_S=0.02, sig_end=2e-4,
                   roll_deg=2.0, bevel_sign_w=1, dims=None, sig_root=(1e-4, 1.5e-4), missing=()) -> SeamEstimate:
    f = prior.frame0()
    params = np.array([list(a), list(b)], dtype=float)
    est = SeamEstimate(groove_found=True, log_bf_run=5000.0, line_model="line", line_params=params, line_cov=np.diag([1e-9] * 4),
                       u=f["u0"], n_plate=f["n0"], plate={}, ends={}, dt_s=prior.dt_s, dt_sigma_s=prior.dt_sigma_s,
                       class_offset_m={"coarse": np.zeros(3), "fine": np.zeros(3)}, leg_eta_m={}, sigma_zeta_m={"fine": 5e-5},
                       tau_end_m=1e-4, bevel_sign_w=bevel_sign_w, stations=[], ratios={}, done=True, binding=[], unresolved=[],
                       frame0=f)
    uh = est.tangent(0.0)
    est.u = uh
    nr = f["n0"] - (f["n0"] @ uh) * uh
    nr /= np.linalg.norm(nr)
    wr = np.cross(nr, uh)
    th = math.radians(roll_deg)
    n_pl = math.cos(th) * nr + math.sin(th) * wr
    est.n_plate = n_pl
    est.plate = {"n_left": n_pl, "n_right": n_pl, "roll_deg": roll_deg, "misalignment_deg": 0.0, "sigma_deg": 0.05}
    speed = float(np.linalg.norm(f["u0"] + a[1] * f["w0"] + b[1] * f["n0"]))       # |dr/ds|
    e_E = e_S + length_m / speed
    est.ends = {"S": EndEst(e_S, sig_end, est.r(e_S), True, 2, []), "E": EndEst(e_E, sig_end, est.r(e_E), True, 2, [])}
    spec = prior.spec
    dims = dims or {"gap_mm": spec.root_gap_mm, "depth_mm": spec.groove_depth_mm, "bevel_angle_deg": spec.bevel_angle_deg,
                    "top_opening_mm": spec.top_opening_mm, "hi_lo_mm": 0.0, "root_face_mm": 0.0}
    step = prior.station_step_m / speed
    s = np.arange(e_S, e_E - 1e-12, step)
    s = np.append(s, e_E) if e_E - s[-1] > 1e-9 else s
    sts = []
    for j, sj in enumerate(s):
        empty = j in missing
        d = {q: {"value": float("nan") if empty else float(v), "sigma": float("inf") if empty else 0.05, "sigma_sys": 0.0,
                 "n": 0 if empty else 30} for q, v in dims.items()}
        faces = {} if empty else {"plate_top_left": {"normal": n_pl, "point_m": est.r(sj) + 0.01 * n_pl + 0.01 * wr,
                                                     "line_wn_mm": [10.0, 10.0, 1.0, 0.0], "sigma_deg": 0.1, "n": 30},
                                  "gap_bottom": {"normal": n_pl, "point_m": est.r(sj), "line_wn_mm": [0.0, 0.0, 1.0, 0.0],
                                                 "sigma_deg": 0.2, "n": 30}}
        sts.append(StationEst(s_m=float(sj), root_m=est.r(sj), root_sigma_m=sig_root, u=uh, n=nr, w=wr, dims=d, faces=faces,
                              n_profiles=0 if empty else 30))
    est.stations = sts
    return est
