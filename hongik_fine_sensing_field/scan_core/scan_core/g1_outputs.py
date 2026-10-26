"""1G core 출력 (G1_CORE_SPEC §5.4, §8.3, §9; WP4): 융합 추정 (SeamEstimate) → out_dir 의 두 결과물만.

  1. sections.json + sections.npz — 심 방향 단면 형상 (스테이션마다 루트 점·면·치수, 각각 σ 와 resolved), 끝, 판 법선, dt.
     npz 는 viz.load_outputs / cross_section_markers 가 읽는 cross_sections.npz 키의 상위 집합 (face_names, station_window_m, normal,
     point_m, root_point_m, n …) — RViz 표시는 그대로 동작한다.
  2. weld_plan.json + weld_plan.npz — 기존 WeldPlan.save 형식 (weld_plan.plan_weld + 비드 모델 + check_plan_contract 를 그대로 재사용)
     + `source` + 패스별 `aim_sigma_m` (스테이션 루트 σ, 보고).
  3. directions/S_to_E/ and directions/E_to_S/ (user 2026-10-14: every run delivers BOTH weld directions, KIRO welds either way
     without sensing again): each a complete weld plan of every pass in that ONE direction (weld_plan.json / .npz; kiro_plan adds
     kiro/) - the same plan_weld on the same measured sections with weld_plan.direction = that direction, nothing else differs (the
     torch orientation is built for each pass's own travel: travel angle relative to it, work angle toward the bevel, TCP roll by the
     NECK_RULE - a pass file's start / goal are never swapped). The job's direction stays the primary plan (item 2, unchanged); a set
     whose every pass already has the primary plan's direction IS the primary plan object (no second plan, no second D14). A set that
     cannot be delivered (its plan not built here, or D14 on the robot) lives in verification/directions/<set>/ with its reason.

좌표 (§9): base frame, m; 치수 mm; 각 deg. S/E 는 사전 (SeamPrior) 의 S0/E0 쪽 끝 — 1G core 에서 S = 센싱이 먼저 간 끝 (사용자 2026-10-10).
계약 좌표 (seam_frame): S = r(e_S), E = r(e_E), u = unit(E − S) ⟂ n̂, n̂ = 판 법선 (센서 쪽),
w = n̂ × u (S→E 로 볼 때 왼쪽), 스테이션 s = (P − S)·u (S = 0, E = L). 스테이션 단면 틀: u = 루트 선 접선, n = n̂ 을 u 에 직교화, w = n × u.
판정은 §2 의 형태만: 수량별 k·σ ≤ tol (보고; 종료 판정 D12 는 추정기), D14 = check_plan_contract (S→E). 새 문턱 없음.
쓰기는 원자적 (임시 파일 + os.replace); sections.npz 는 압축 없음.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np

from .cross_sections import FACE_NAMES
from .g1_types import DIM_NAMES, SeamEstimate, SeamPrior, atomic_write_bytes, jsonable, _savez_bytes
from .seam_frame import SeamFrame, seam_frame
from .specimen import GrooveSpec
from .weld_plan import DIRECTIONS as WELD_DIRECTIONS
from .weld_plan import (GrooveStation, PlanContractError, WeldDesign, WeldPlan, WeldProcess, check_plan_contract, plan_weld)

__all__ = ["SECTIONS_SCHEMA", "SECTIONS_NPZ_VERSION", "SECTIONS_SOURCE", "Q_ROOT", "G1WeldPlan", "contract_frame", "station_frames",
           "pass_summary", "sections_doc", "write_sections", "weld_stations", "weld_plan_from_estimate", "save_weld_plan",
           "write_outputs", "retract_weld_plan", "sections_npz_arrays", "DIRECTIONS_DIR", "DIRECTION_SETS", "DIRECTION_SETS_RULE",
           "DirectionSet", "direction_sets", "direction_plan", "save_direction_set", "direction_sets_summary"]

SECTIONS_SCHEMA = "g1core.sections/1"
SECTIONS_NPZ_VERSION = 101
SECTIONS_SOURCE = "g1core sections/1"
ACCURACY = "relative (repeatability); absolute needs calibration covariance"
Q_ROOT = ("root_lateral", "root_normal")                 # 루트 점의 판정 수량 (root_position_mm)
_SPEC_KEYS = {"gap_mm": "root_gap_mm", "depth_mm": "depth_mm", "top_opening_mm": "top_opening_mm", "hi_lo_mm": "hi_lo_mm",
              "bevel_angle_deg": "bevel_angle_deg", "root_face_mm": "root_face_mm"}
# both weld directions (user 2026-10-14): out_dir/directions/<set>/, set -> the direction of every pass in it
DIRECTIONS_DIR = "directions"
DIRECTION_SETS = {"S_to_E": "S->E", "E_to_S": "E->S"}
DIRECTION_SETS_RULE = ("every run delivers the weld plan in BOTH directions (user 2026-10-14): directions/S_to_E/ = every pass S->E (from "
                       "the S end - the end the sensing run visited first - to the E end), directions/E_to_S/ = every pass E->S; each "
                       "set = the same plan_weld on the same measured sections, only weld_plan.direction differs (the torch orientation "
                       "is built for each pass's own travel; never swap a pass file's start and goal - the orientation is fixed in "
                       "space). Weld in the direction of the set you take; per pass either set may be used (the same layers at the same "
                       "aim points). The top-level weld_plan.* / kiro/ / result.paths = the job's direction (weld_plan.direction)")


# ---------------------------------------------------------------- frames

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _plate_normal(est: SeamEstimate) -> np.ndarray:
    """n̂ 을 교시 틀 n0 (센서 쪽) 쪽으로 (부호 규약)."""
    n = _unit(est.n_plate)
    n0 = est.frame0["n0"] if est.frame0 is not None else n
    return n if float(n @ n0) >= 0.0 else -n


def contract_frame(est: SeamEstimate) -> SeamFrame:
    """S→E 계약 좌표: S = r(e_S), E = r(e_E), n = n̂, σ = 끝 σ (seam_frame)."""
    eS, eE = est.ends["S"], est.ends["E"]
    return seam_frame(est.r(float(eS.s_m)), est.r(float(eE.s_m)), _plate_normal(est),
                      sigma_S_m=float(eS.sigma_m), sigma_E_m=float(eE.sigma_m))


def station_frames(est: SeamEstimate, s_m, roots=None) -> tuple:
    """사전 틀 좌표 s (F0: 착점 S0 에서 S0→E0) → (origin, u = 접선, n = n̂ ⟂ u, w = n × u), 각 (K, 3). origin = roots (스테이션의 잰 루트
    — 직선 + 국소 편차의 BLUP, 사용자 2026-10-11) 가 주어지면 그것, 아니면 직선 r(s)."""
    s = np.asarray(s_m, dtype=float).reshape(-1)
    o = est.r(s).reshape(-1, 3) if roots is None else np.asarray(roots, dtype=float).reshape(-1, 3)
    u = est.tangent(s).reshape(-1, 3)
    nh = _plate_normal(est)
    n = _unit(nh[None, :] - (u @ nh)[:, None] * u)
    w = _unit(np.cross(n, u))
    return o, u, n, w


# ---------------------------------------------------------------- sections doc

def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _ratio(sigma, sigma_sys, target) -> tuple:
    """(ratio = √(σ² + σ_sys²)/precision, resolved = ratio ≤ 1) — the done test's form (1-σ precision target of the job, user
    2026-10-11); no target → (None, None) = report only. σ missing → ∞."""
    if target is None:
        return None, None
    s, ss = _f(sigma), _f(sigma_sys)
    s = math.inf if not math.isfinite(s) else s
    ss = 0.0 if not math.isfinite(ss) else ss
    r = math.hypot(s, ss) / float(target)
    return r, bool(r <= 1.0)


def pass_summary(leg) -> dict:
    """LegObs (또는 이미 요약된 dict) → sections.json passes[] 행. groove_profiles = Σ π_i (π_i = logistic(λ_i), 기대 개수 — 문턱 없음)."""
    if isinstance(leg, dict):
        return dict(leg)
    lam = np.asarray(leg.log_bf, dtype=float)
    fin = np.isfinite(lam)
    pi = np.zeros_like(lam)
    pi[fin] = 0.5 * (1.0 + np.tanh(0.5 * lam[fin]))         # logistic, overflow 없이
    sr = np.asarray(leg.sigma_r_mm, dtype=float)
    ep = np.asarray(leg.eps, dtype=float)
    return {"name": leg.name, "class": leg.cls, "direction": int(leg.direction), "n_profiles": int(leg.n),
            "sigma_r_mm_median": float(np.nanmedian(sr)) if np.isfinite(sr).any() else None,
            "outlier_frac_median": float(np.nanmedian(ep)) if np.isfinite(ep).any() else None,
            "groove_profiles": float(pi.sum())}


def _bevel_words(spec: GrooveSpec, bevel_sign_w: int) -> str:
    """S→E 로 볼 때 베벨 판 쪽: bevel_sign_w = +1 → +w = 왼쪽 (§4.1)."""
    if spec.kind == "single_v":
        return "both"
    if spec.kind == "square":
        return "none"
    return "left" if int(bevel_sign_w) > 0 else "right"


def sections_doc(est: SeamEstimate, prior: SeamPrior, run: Optional[dict] = None) -> dict:
    """추정 + 사전 정보 (+ 실행 정보) → sections.json 문서 (§9).

    run (선택): base_frame (기본 'link0'), status (기본 done/unresolved), passes ([LegObs | dict]), notes ([str]),
    bevel_side_ln_bf (WP1 choose_bevel_side 의 ln BF; 추정에 없으므로 실행이 넘긴다)."""
    run = dict(run or {})
    k = float(prior.k)
    tol = prior.tol                                    # design tolerances: measured-vs-design warnings only (user 2026-10-11)
    pr = prior.precision                               # 1-σ precision targets: the done test

    def target(q):
        if pr is None:
            return None
        if q == "root":
            return float(pr.root_position_mm)
        if q == "end":
            return float(pr.end_position_mm)
        return pr.for_dim(q, prior.spec)
    fr = contract_frame(est)
    L = fr.length_m
    notes = list(run.get("notes") or [])
    st_list = sorted(est.stations, key=lambda s: float(s.s_m))
    s_f0 = np.array([float(s.s_m) for s in st_list])
    o, u, n, w = station_frames(est, s_f0, roots=[st.root_m for st in st_list]) if st_list else (np.zeros((0, 3)),) * 4
    stations = []
    for i, st in enumerate(st_list):
        lat, nor = (_f(v) for v in st.root_sigma_m)
        rr = [_ratio(x * 1e3, 0.0, target("root")) for x in (lat, nor)]                    # σ m → mm
        row = {"s_m": float(fr.s(o[i])), "s_rough_m": float(st.s_m), "root_point_m": o[i], "root_sigma_m": {"lateral": lat, "normal": nor},
               "root_ratio": max(r[0] for r in rr), "root_resolved": all(r[1] for r in rr),
               "u": u[i], "n": n[i], "w": w[i], "n_profiles": int(st.n_profiles)}
        for q in DIM_NAMES:
            d = (st.dims or {}).get(q) or {}
            r_, ok = _ratio(d.get("sigma"), d.get("sigma_sys"), target(q))
            row[q] = {"value": _f(d.get("value")), "sigma": _f(d.get("sigma")), "sigma_sys": _f(d.get("sigma_sys")),
                      "n": int(d.get("n") or 0), "tol": tol.for_dim(q), "precision": target(q), "ratio": r_, "resolved": ok}
        row["faces"] = {nm: {"normal": fc.get("normal"), "point_m": fc.get("point_m"), "line_wn_mm": fc.get("line_wn_mm"),
                             "sigma_deg": _f(fc.get("sigma_deg")), "n_profiles": int(fc.get("n") or 0)}
                        for nm, fc in (st.faces or {}).items() if nm in FACE_NAMES}
        stations.append(row)
    ends = {}
    ef = (est.plate or {}).get("end_faces") or {}
    for X in ("S", "E"):
        e = est.ends[X]
        r_, _ = _ratio(_f(e.sigma_m) * 1e3, 0.0, target("end"))
        ends[X] = {"point_m": est.r(float(e.s_m)), "s_m": 0.0 if X == "S" else L, "s_rough_m": float(e.s_m),
                   "sigma_m": _f(e.sigma_m), "two_sided": bool(e.two_sided), "n_legs": int(e.n_legs), "ratio": r_}
        # the end at the top edge (cap pass; square end faces → the same end as at the root) and the plate ends
        top = ((ef.get("ends") or {}).get(X) or {}).get("top") or {}
        st_ = _f(top.get("s_m"))
        rt, _ = _ratio(_f(top.get("sigma_m")) * 1e3, 0.0, target("end"))
        ends[X]["top_edge"] = {"point_m": est.r(st_) if math.isfinite(st_) else None,
                               "s_m": float(fr.s(est.r(st_))) if math.isfinite(st_) else None, "s_taught_m": st_,
                               "sigma_m": _f(top.get("sigma_m")), "h_m": _f(top.get("h_m")), "inner_plate": top.get("inner_plate"),
                               "ratio": rt}
        ends[X]["plates"] = {key.split(":")[1]: {"s_taught_m": _f(q.get("s_m")), "sigma_m": _f(q.get("sigma_m"))}
                             for key, q in (ef.get("plates") or {}).items() if key.startswith(X + ":")}
    # 길이 σ: 끝 공분산이 추정에 없으므로 독립으로 합친다 (공유 오프셋이 있으면 과대 — 보수적, 보고만); 끝 σ (교시 축 s) 를 심 길이로
    # 옮긴다 (seam_estimate.delivered_length — 설계 대비 경고와 같은 값)
    from .seam_estimate import delivered_length
    sig_L = delivered_length(est)[1]
    L_d = float(prior.spec.weld_length_mm)
    ratios = [v for v in (est.ratios or {}).values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
    pl = dict(est.plate or {})
    # seam-level section dims pooled over the stations — these gate "done" (per-station dims are reported, never gate)
    seam_dims = {}
    for q, d in (pl.get("seam_dims") or {}).items():
        r_, ok = _ratio(d.get("sigma"), d.get("sigma_sys"), target(q))
        gates = q in ("gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm") and target(q) is not None
        seam_dims[q] = {"value": _f(d.get("value")), "sigma": _f(d.get("sigma")), "sigma_sys": _f(d.get("sigma_sys")),
                        "along_seam_sd": _f(d.get("along_seam_sd")), "n_stations": int(d.get("n_stations") or 0),
                        "tol": tol.for_dim(q), "precision": target(q), "ratio": r_, "resolved": ok, "gates_done": gates}
    doc = {
        "schema": SECTIONS_SCHEMA, "base_frame": str(run.get("base_frame") or "link0"),
        "units": {"position": "m", "dims": "mm", "angles": "deg"}, "k": k, "accuracy": ACCURACY,
        "rough_points": {"S_m": prior.S0, "E_m": prior.E0, "S_taught_point": run.get("S_taught_point"),
                         "s_rough": "s_rough_m = distance from the rough S point along the rough S->E axis"},
        "frame": {"S_m": fr.S, "E_m": fr.E, "u": fr.u, "n": fr.n, "w": _unit(np.cross(fr.n, fr.u)), "length_m": L,
                  "s": "s = (P - S).u from the measured S end (S->E contract)",
                  "S": "the end the sensing run visited first (1G core); S_taught_point says which taught point that was"},
        "tolerances": tol.to_dict(),                                  # design: the measured-vs-design warnings only
        "precision": ({**pr.to_dict(), "bevel_angle_deg": pr.for_dim("bevel_angle_deg", prior.spec),
                       "ratio": "sigma / precision (1-sigma targets); done = every ratio <= 1"} if pr is not None else None),
        "status": str(run.get("status") or ("done" if est.done else "unresolved")),
        "done": {"resolved": bool(est.done), "max_ratio": max(ratios) if ratios else None, "binding": list(est.binding or []),
                 "unresolved": list(est.unresolved or [])},
        "ends": ends,
        "seam_dims": seam_dims,
        "length_mm": {"value": L * 1e3, "sigma": sig_L * 1e3, "design": L_d,
                      "z_vs_design": (L * 1e3 - L_d) / (sig_L * 1e3) if sig_L > 0.0 and math.isfinite(sig_L) else None},
        "bevel_side": _bevel_words(prior.spec, est.bevel_sign_w),
        "bevel_side_ln_bf": run.get("bevel_side_ln_bf", pl.get("bevel_side_ln_bf")),
        "plate": {"normal_left": pl.get("n_left"), "normal_right": pl.get("n_right"), "normal": _plate_normal(est),
                  "roll_deg": pl.get("roll_deg"), "misalignment_deg": pl.get("misalignment_deg"),
                  "sigma_deg": pl.get("sigma_deg", pl.get("sigmas"))},
        "time_offset_ms": {"value": _f(est.dt_s) * 1e3, "sigma": _f(est.dt_sigma_s) * 1e3},
        "laser_pitch_w_x_deg": {"value": _f((ef.get("w_x") or {}).get("deg")), "sigma": _f((ef.get("w_x") or {}).get("sigma_deg")),
                                "source": "calibration (laser_pitch_sigma_rad); one tool orientation does not identify it"},
        "sensor_edge_response": {"sigma_L_mm": _f((ef.get("sensor") or {}).get("sigma_L_m")) * 1e3,
                                 "theta_tri_deg": _f((ef.get("sensor") or {}).get("theta_tri_deg")),
                                 "lit_ends": sorted({r.get("end") for r in (ef.get("readings") or []) if r.get("lit")})},
        "station_window_m": float(prior.station_window_m), "station_step_m": float(prior.station_step_m),
        "stations": stations,
        "passes": [pass_summary(p) for p in (run.get("passes") or [])],
        # measured joint vs design (user rule 2026-10-11): a report, never a judgement — the values above are measurements
        "design_deviations": pl.get("design_deviations"),
        "warnings": list(pl.get("warnings") or []),
        "model_checks": {k_: (pl.get("model_checks") or {}).get(k_) for k_ in ("summary", "hypotheses")} if pl.get("model_checks")
        else None,
        "notes": notes,
    }
    return jsonable(doc)


def _q_names(doc: dict) -> list:
    pr = doc.get("precision") or {}
    have = {"gap_mm": pr.get("length_mm"), "depth_mm": pr.get("length_mm"), "bevel_angle_deg": pr.get("bevel_angle_deg"),
            "top_opening_mm": pr.get("length_mm")}
    return list(Q_ROOT) + [q for q in DIM_NAMES if have.get(q) is not None]


def _arr(v, shape) -> np.ndarray:
    """JSON 값 (None = NaN) → 모양 shape 의 float 배열; 없거나 모양이 다르면 NaN."""
    if v is None:
        return np.full(shape, np.nan)
    a = np.array([np.nan if x is None else float(x) for x in np.ravel(np.asarray(v, dtype=object))], dtype=float)
    return a.reshape(shape) if a.size == int(np.prod(shape)) else np.full(shape, np.nan)


def sections_npz_arrays(doc: dict) -> dict:
    """sections.json 문서 → sections.npz 배열 (§9; viz 호환 cross_sections.npz 상위 집합)."""
    st = doc.get("stations") or []
    K, F = len(st), len(FACE_NAMES)
    win = float(doc.get("station_window_m") or np.nan)
    fr = doc.get("frame") or {}
    a = {"schema_version": np.array(SECTIONS_NPZ_VERSION), "base_frame": np.array(str(doc.get("base_frame", ""))),
         "k": np.array(float(doc.get("k"))), "face_names": np.array(FACE_NAMES),
         "s_m": np.array([_f(r["s_m"]) for r in st], dtype=float).reshape(K),
         "root_point_m": np.array([_arr(r["root_point_m"], (3,)) for r in st], dtype=float).reshape(K, 3),
         "root_sigma_m": np.array([[_f(r["root_sigma_m"]["lateral"]), _f(r["root_sigma_m"]["normal"])] for r in st],
                                  dtype=float).reshape(K, 2),
         "u": _arr(fr.get("u"), (3,)), "n": _arr(fr.get("n"), (3,)), "w": _arr(fr.get("w"), (3,)),
         "station_window_m": np.array(win),
         "normal": np.full((K, F, 3), np.nan), "point_m": np.full((K, F, 3), np.nan), "sigma_deg": np.full((K, F), np.nan),
         "n_profiles": np.zeros((K, F), np.int64)}
    a["s_range_m"] = np.stack([a["s_m"] - 0.5 * win, a["s_m"] + 0.5 * win], axis=1).reshape(K, 2)
    for i, r in enumerate(st):
        for j, nm in enumerate(FACE_NAMES):
            fc = (r.get("faces") or {}).get(nm)
            if not fc:
                continue
            a["normal"][i, j] = _arr(fc.get("normal"), (3,))
            a["point_m"][i, j] = _arr(fc.get("point_m"), (3,))
            a["sigma_deg"][i, j] = _f(fc.get("sigma_deg"))
            a["n_profiles"][i, j] = int(fc.get("n_profiles") or 0)
    for q in DIM_NAMES:
        for suf, key in (("", "value"), ("_sigma", "sigma"), ("_sigma_sys", "sigma_sys")):
            a[q + suf] = np.array([_f((r.get(q) or {}).get(key)) for r in st], dtype=float).reshape(K)
    ends = doc.get("ends") or {}
    a["ends_point_m"] = np.array([_arr((ends.get(X) or {}).get("point_m"), (3,)) for X in ("S", "E")], dtype=float)
    a["ends_sigma_m"] = np.array([_f((ends.get(X) or {}).get("sigma_m")) for X in ("S", "E")], dtype=float)
    qn = _q_names(doc)
    a["q_names"] = np.array(qn)
    res = np.zeros((K, len(qn)), bool)
    for i, r in enumerate(st):
        for j, q in enumerate(qn):
            if q in Q_ROOT:
                res[i, j] = bool(r.get("root_resolved"))
            else:
                res[i, j] = bool((r.get(q) or {}).get("resolved"))
    a["resolved"] = res
    return a


def write_sections(doc: dict, out_dir) -> list:
    """sections.json + sections.npz (원자적, npz 압축 없음) → [json 경로, npz 경로]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, npz = out / "sections.json", out / "sections.npz"
    atomic_write_bytes(jp, json.dumps(jsonable(doc), indent=1, ensure_ascii=False).encode("utf-8"))
    atomic_write_bytes(npz, _savez_bytes(**sections_npz_arrays(doc)))
    return [str(jp), str(npz)]


# ---------------------------------------------------------------- weld plan

def _fill_missing(s: np.ndarray, v: np.ndarray) -> tuple:
    """유한하지 않은 값은 이웃 스테이션에서 선형 보간 (stations_from_features 와 같은 처리). 반환 (값, 보간 표시). 값이 하나도 없으면 NaN."""
    have = np.isfinite(v)
    if not have.any():
        return v, ~have
    return np.interp(s, s[have], v[have]), ~have


def _station_spec(spec: GrooveSpec, side: str, G: float, D: float, H: float, beta: float, rf: float, L_mm: float,
                  name: str) -> GrooveSpec:
    """실측 단면 (갭 G, 베벨 쪽 깊이 D, hi-lo H = 직각 쪽 − 베벨 쪽 상면 (§4.6), 베벨각 β, 루트면 rf) → GrooveSpec (단면 좌표: x<0 'left',
    hi_lo = right − left). 물리 정의역으로만 자른다 (G ≥ 0, 0 ≤ rf ≤ t, 0 ≤ β < 90)."""
    kind = spec.kind
    if kind == "single_bevel":
        h_rl = H if side == "left" else -H              # right − left
        t = D if side == "left" else D - h_rl          # left 판 상면 높이
    else:
        h_rl = H
        t = D
    t = max(float(t), 0.0)
    rf_ = min(max(float(rf), 0.0), t)
    b = 0.0 if kind == "square" else min(max(float(beta), 0.0), 89.999)
    return GrooveSpec(kind=kind, thickness_mm=t, bevel_angle_deg=b, bevel_side=side if kind == "single_bevel" else spec.bevel_side,
                      root_gap_mm=max(float(G), 0.0), root_face_mm=rf_, backing=spec.backing,
                      backing_thickness_mm=spec.backing_thickness_mm, weld_length_mm=L_mm, plate_width_mm=spec.plate_width_mm,
                      hi_lo_mm=float(h_rl), name=name)


def weld_stations(est: SeamEstimate, spec: GrooveSpec) -> list:
    """추정 스테이션 → GrooveStation 목록 (s = 계약 좌표, 원점 = r(s_j) = 루트갭 중심·뒷댐재 상면, u = 접선, n = n̂ ⟂ u, w = n × u,
    spec = 그 스테이션의 실측 G/D/H/β/rf). 프로파일이 없는 스테이션의 치수는 이웃에서 보간 (interpolated=True); 어느 스테이션에도 없는
    치수는 설계값 (사후 = 사전)."""
    st_list = sorted(est.stations, key=lambda s: float(s.s_m))
    if len(st_list) < 2:
        raise ValueError("weld plan needs at least two stations")
    fr = contract_frame(est)
    s_f0 = np.array([float(s.s_m) for s in st_list])
    o, u, n, w = station_frames(est, s_f0, roots=[st.root_m for st in st_list])          # the measured station roots (BLUP)
    s_c = np.asarray(fr.s(o), dtype=float)
    design = {"gap_mm": spec.root_gap_mm, "depth_mm": spec.groove_depth_mm, "top_opening_mm": spec.top_opening_mm,
              "hi_lo_mm": spec.hi_lo_mm, "bevel_angle_deg": spec.bevel_angle_deg, "root_face_mm": spec.root_face_mm}
    vals, interp = {}, np.zeros(len(st_list), bool)
    for q in DIM_NAMES:
        v = np.array([_f(((st.dims or {}).get(q) or {}).get("value")) for st in st_list])
        if not np.isfinite(v).any():
            v = np.full(v.shape, float(design[q]))
        v, miss = _fill_missing(s_c, v)
        vals[q] = v
        if q in ("gap_mm", "depth_mm", "top_opening_mm"):
            interp |= miss
    side = ("right" if int(est.bevel_sign_w) > 0 else "left") if spec.kind == "single_bevel" else spec.bevel_side
    L_mm = fr.length_m * 1e3
    out = []
    for i, st in enumerate(st_list):
        sp = _station_spec(spec, side, vals["gap_mm"][i], vals["depth_mm"][i], vals["hi_lo_mm"][i], vals["bevel_angle_deg"][i],
                           vals["root_face_mm"][i], L_mm, f"section_{s_c[i] * 1e3:.0f}mm")
        meas = {_SPEC_KEYS[q]: float(vals[q][i]) for q in DIM_NAMES}
        meas["sigma"] = {_SPEC_KEYS[q]: _f(((st.dims or {}).get(q) or {}).get("sigma")) for q in DIM_NAMES}
        meas["root_sigma_m"] = [_f(v) for v in st.root_sigma_m]
        meas["s_rough_m"] = float(st.s_m)
        out.append(GrooveStation(s_mm=float(s_c[i] * 1e3), origin=o[i].copy(), u=u[i].copy(), n=n[i].copy(), w=w[i].copy(), spec=sp,
                                 n_profiles=int(st.n_profiles), interpolated=bool(interp[i]), measured=meas))
    return out


@dataclass
class G1WeldPlan(WeldPlan):
    """WeldPlan + 출처 + 패스별 겨냥점 σ (보고). save / to_dict 형식은 WeldPlan 그대로에 키만 더한다."""
    source: str = SECTIONS_SOURCE
    aim_sigma_m: dict = field(default_factory=dict)       # k -> (N, 2) (lateral, normal) at the pass's sections, pass order
    layer_ends: dict = field(default_factory=dict)        # k -> the seam ends at the pass's layer height (end-face model)
    # the direction sets (DirectionSet by set name; write_outputs): not part of this plan's own files (to_dict / save unchanged)
    directions: dict = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["source"] = self.source
        d["weld_direction"] = {"setting": self.process.direction, "options": list(WELD_DIRECTIONS),
                               "meaning": "S->E = every pass welds from the S end (the end the sensing run visited first) to the E end; "
                                          "E->S the reverse; alternate = odd passes S->E; a plan setting (weld_plan.direction), "
                                          "independent of the teaching and sensing order"}
        from .weld_line import weld_line
        by_k = {int(wp.k): wp for wp in self.passes}
        for p in d.get("passes") or []:
            le = self.layer_ends.get(int(p["k"]))
            if le is not None:
                p["ends_at_layer"] = jsonable(le)
            wp = by_k.get(int(p["k"]))
            if wp is not None and len(self.stations) >= 2:
                # the pass's straight weld line (= kiro/pass_<k>.yaml start / goal, result.paths): the measured ends at the layer height
                # moved inside by the weld design's end inset (weld_design.end_inset_mm)
                ln = weld_line(self, wp, self.torch, self.torch_roll_deg)     # THE delivered line (D14 checks it; kiro, paths)
                e = ln.ends
                p["weld_line"] = jsonable({
                    "start_s_mm": e["start"]["s_mm"], "goal_s_mm": e["goal"]["s_mm"], "start_tip_base_m": e["start"]["tip_base_m"],
                    "goal_tip_base_m": e["goal"]["tip_base_m"],
                    "measured_ends": {X: {"end": e["measured"][X]["end"], "s_mm": e["measured"][X]["s_mm"],
                                          "sigma_mm": e["measured"][X]["sigma_mm"]} for X in ("start", "goal")},
                    "end_inset_mm": e["inset_mm"], "ends": e["source"],
                    "tcp_base_m": None if ln.tcp_start is None else {"start": ln.tcp_start, "goal": ln.tcp_goal},
                    "R_base_tcp": ln.R_tcp, "torch_axis_base": ln.axis, "roll_deg": ln.roll_deg,
                    "weave": None if ln.weave is None else {k_: v_ for k_, v_ in ln.weave.items()},
                    "line": "the delivered line (scan_core.weld_line): the straight wire-tip (bead aim point) line of the action "
                            "result's paths (base frame) at ONE torch orientation (R_base_tcp: the rotation mean of the pass's station "
                            "torch frames, the TCP roll); kiro/pass_<k>.yaml gives the weld torch's TCP (tcp_base_m: the torch tip end "
                            "centre = these points moved back along -z_TCP by the torch's nozzle end + stick_out_mm); D14 checked "
                            "exactly these poses (torch_orientation, line_step_mm); the dense paths of weld_plan.npz are information"})
            a = self.aim_sigma_m.get(int(p["k"]))
            if a is not None:
                a = np.asarray(a, dtype=float)
                p["aim_sigma_m"] = jsonable({"lateral": a[:, 0], "normal": a[:, 1],
                                             "max": float(np.nanmax(a)) if np.isfinite(a).any() else None})
        return d


def weld_plan_from_estimate(est: SeamEstimate, spec: GrooveSpec, process: WeldProcess, *, base_frame: str = "link0",
                            gravity_axis_base=None, k: float, bead_model=None, design: Optional[WeldDesign] = None,
                            torch=None) -> G1WeldPlan:
    """추정 → 4패스 (WPS·비드 모델에 따라) 작업 계획: weld_stations → plan_weld (그대로) → check_plan_contract (S→E, k·σ_끝).
    계약 위반이면 PlanContractError (예외의 .plan = 거부된 계획 — verification/ 에 남기도록)."""
    stations = weld_stations(est, spec)
    notes = []
    n_int = sum(1 for s in stations if s.interpolated)
    if n_int:
        notes.append(f"{n_int} station(s) without profiles: dimensions interpolated from neighbours")
    notes += [f"WARNING (measured vs design, report only): {w}" for w in ((est.plate or {}).get("warnings") or [])]
    plan = plan_weld(stations, process, spec_design=spec, base_frame=base_frame, position="1G", gravity_axis_base=gravity_axis_base,
                     notes=notes, bead_model=bead_model, design=design, torch=torch)
    fr = contract_frame(est)
    by_s = {float(st.s_mm): np.asarray(st.measured["root_sigma_m"], dtype=float) for st in stations}
    aim = {int(p.k): np.array([by_s.get(float(s), np.full(2, np.nan)) for s in p.s_mm]) for p in plan.passes}
    # a pass uses the seam end at its own layer height (user 2026-10-10): root pass → root end, cap → top-edge end
    from .seam_estimate import end_at
    layer_ends = {}
    for p in plan.passes:
        hz = np.concatenate([np.asarray(p.layer_from_mm, float)[[0, -1]], np.asarray(p.layer_to_mm, float)[[0, -1]]])
        h_m = float(np.nanmean(hz)) * 1e-3 if np.isfinite(hz).any() else 0.0
        le = {"layer_mid_h_m": h_m}
        for X in ("S", "E"):
            s_, sg = end_at(est, X, h_m)
            le[X] = {"s_taught_m": s_, "sigma_m": sg, "point_m": est.r(s_), "s_m": float(fr.s(est.r(s_)))}
        layer_ends[int(p.k)] = le
    g1 = G1WeldPlan(**{f.name: getattr(plan, f.name) for f in fields(WeldPlan)}, aim_sigma_m=aim, layer_ends=layer_ends)
    viol, checked = check_plan_contract(g1, fr, k=float(k))
    g1.ends = {**fr.to_dict(), "contract": "u = unit(E - S); s from S; first section at S, last at E within k sigma",
               "checked": checked, "violations": list(viol)}
    if viol:
        err = PlanContractError("weld plan refused (S->E contract: sections must run from the estimated S end to the estimated E end): "
                                + "; ".join(viol))
        err.plan = g1
        raise err
    return g1


def save_weld_plan(plan: WeldPlan, out_dir) -> list:
    """WeldPlan.save 형식 그대로, 파일마다 원자적 (임시 폴더에 저장 → os.replace). → [json, npz]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".weld_plan.", dir=str(out))
    try:
        files = plan.save(tmp)
        dst = []
        for f in files:
            d = out / Path(f).name
            os.replace(f, d)
            dst.append(str(d))
        return dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _move_tree(src: Path, dst: Path) -> list:
    """Every file under src → the same relative path under dst (os.replace; an existing file there is replaced), then the emptied
    folders of src removed. → the moved files (their new paths)."""
    moved = []
    if not src.is_dir():
        return moved
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        d = dst / f.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        os.replace(f, d)
        moved.append(str(d))
    for d in sorted((p for p in src.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)) + [src]:
        try:
            d.rmdir()
        except OSError:
            pass
    return moved


def retract_weld_plan(out_dir) -> list:
    """out_dir 의 weld_plan.* 를 verification/ 으로 옮긴다 (D14 도달성 실패: 계획은 남기되 결과물이 아님). The direction sets go with it
    (out_dir/directions/ → verification/directions/): no set is delivered without the primary plan."""
    out = Path(out_dir)
    ver = out / "verification"
    moved = []
    for nm in ("weld_plan.json", "weld_plan.npz"):
        p = out / nm
        if p.exists():
            ver.mkdir(parents=True, exist_ok=True)
            os.replace(p, ver / nm)
            moved.append(str(ver / nm))
    return moved + _move_tree(out / DIRECTIONS_DIR, ver / DIRECTIONS_DIR)


# ---------------------------------------------------------------- both weld directions (user 2026-10-14)

@dataclass
class DirectionSet:
    """One direction set (DIRECTION_SETS_RULE): the weld plan of every pass in ONE direction. plan = its G1WeldPlan; None when it is
    the primary plan itself (same_as_primary: every pass of the job's plan already welds this way - direction_plan gives the primary
    object: no second plan, no second D14) or when it could not be built (reason). delivered False = not delivered: its files (if any)
    in verification/directions/<name>/, reason = why. files = where its weld_plan.* are now (kiro_plan lists its kiro/ files)."""
    name: str                                   # 'S_to_E' | 'E_to_S' (DIRECTION_SETS)
    direction: str                              # 'S->E' | 'E->S'
    plan: Optional["G1WeldPlan"] = field(default=None, repr=False, compare=False)
    same_as_primary: bool = False
    delivered: bool = True
    reason: str = ""
    files: list = field(default_factory=list)

    @property
    def folder(self) -> str:
        """The set's folder relative to out_dir (delivered: directions/<name>; not delivered: verification/directions/<name>)."""
        return f"{DIRECTIONS_DIR}/{self.name}" if self.delivered else f"verification/{DIRECTIONS_DIR}/{self.name}"

    def to_dict(self) -> dict:
        return {"name": self.name, "direction": self.direction, "same_as_primary": bool(self.same_as_primary),
                "delivered": bool(self.delivered), "reason": self.reason or None, "folder": self.folder, "files": list(self.files)}


def direction_plan(primary, ds: DirectionSet):
    """The plan of a direction set: the primary plan object when the set is the primary (same_as_primary), else its own (or None)."""
    return primary if ds.same_as_primary else ds.plan


def direction_sets(est: SeamEstimate, spec: GrooveSpec, process: WeldProcess, primary: G1WeldPlan, *, base_frame: str = "link0",
                   gravity_axis_base=None, k: float, bead_model=None, design: Optional[WeldDesign] = None, torch=None) -> dict:
    """The two direction sets (DIRECTION_SETS_RULE) beside the primary plan (the job's weld_plan.direction): a set whose direction
    every primary pass already has = the primary plan object (same_as_primary - no second computation); every other set =
    weld_plan_from_estimate on the same estimate with only process.direction replaced (the same sections, S->E contract check, bead
    model, design and torch). A set whose plan cannot be built (PlanContractError - its refused plan kept in .plan - or any other
    error) is not delivered, with the reason; the primary plan is never affected. → {name: DirectionSet} in DIRECTION_SETS order."""
    dirs = {str(wp.direction) for wp in primary.passes}
    out = {}
    for name, d in DIRECTION_SETS.items():
        if dirs == {d}:
            out[name] = DirectionSet(name=name, direction=d, same_as_primary=True)
            continue
        try:
            p = weld_plan_from_estimate(est, spec, dataclasses.replace(process, direction=d), base_frame=base_frame,
                                        gravity_axis_base=gravity_axis_base, k=k, bead_model=bead_model, design=design, torch=torch)
            out[name] = DirectionSet(name=name, direction=d, plan=p)
        except Exception as e:  # noqa: BLE001 - this set is not delivered (reason kept); the primary plan and the run stand
            out[name] = DirectionSet(name=name, direction=d, plan=getattr(e, "plan", None), delivered=False,
                                     reason=f"weld plan not built ({type(e).__name__}: {e})")
    return out


_SET_FILES = re.compile(r"^(weld_plan\.(json|npz)|kiro/(README\.md|pass_\d+(_info)?\.yaml))$")   # what a delivered set holds


def save_direction_set(primary, ds: DirectionSet, out_dir) -> list:
    """The set's weld_plan.* (save_weld_plan, atomic per file): delivered → out_dir/directions/<name>/, not delivered →
    out_dir/verification/directions/<name>/ (and an earlier write's delivered files of this set in out_dir/directions/<name>/ are
    removed - an offline estimator writes into the same folder again); a set without a plan writes nothing. → its files (also
    ds.files)."""
    out = Path(out_dir)
    if not ds.delivered:
        old = out / DIRECTIONS_DIR / ds.name
        for f in (sorted(old.rglob("*")) if old.is_dir() else []):
            if f.is_file() and _SET_FILES.match(f.relative_to(old).as_posix()):
                f.unlink()
        for d in (old / "kiro", old, out / DIRECTIONS_DIR):  # the emptied folders (anything else there stays)
            try:
                d.rmdir()
            except OSError:
                pass
    p = direction_plan(primary, ds)
    ds.files = [] if p is None else save_weld_plan(p, out / ds.folder)
    return list(ds.files)


def direction_sets_summary(plan) -> Optional[dict]:
    """The plan's direction sets as {rule, sets: {name: DirectionSet.to_dict()}}, None for a plan without them."""
    sets = getattr(plan, "directions", None) or {}
    if not sets:
        return None
    return jsonable({"rule": DIRECTION_SETS_RULE, "primary_setting": str(plan.process.direction),
                     "sets": {nm: ds.to_dict() for nm, ds in sets.items()}})


def write_outputs(est: SeamEstimate, prior: SeamPrior, process: WeldProcess, out_dir, *, base_frame: str = "link0",
                  gravity_axis_base=None, k: Optional[float] = None, run: Optional[dict] = None, bead_model=None,
                  design: Optional[WeldDesign] = None, torch=None) -> dict:
    """worker 'outputs' 작업 본체: sections.* + weld_plan.* 를 out_dir 에, then the two direction sets (DIRECTION_SETS_RULE:
    direction_sets, saved to out_dir/directions/<set>/ - a set not built: verification/directions/<set>/; plan.directions carries
    them for D14 and kiro_plan). 계약 위반이면 단면은 쓰고, 거부된 계획은 out_dir/verification/ 에 남기고 PlanContractError 를 다시
    던진다 (D14 → DATA_FAIL; no direction set then).
    → {files (sections.* + the primary weld_plan.*), plan (G1WeldPlan), plan_summary (줄 목록), contract (검사 값), directions
    (direction_sets_summary: per set its folder and files, or the reason it is not delivered)}."""
    k = float(prior.k if k is None else k)
    run = dict(run or {})
    run.setdefault("base_frame", base_frame)
    files = write_sections(sections_doc(est, prior, run), out_dir)
    try:
        plan = weld_plan_from_estimate(est, prior.spec, process, base_frame=base_frame, gravity_axis_base=gravity_axis_base, k=k,
                                       bead_model=bead_model, design=design, torch=torch)
    except PlanContractError as e:
        if getattr(e, "plan", None) is not None:
            save_weld_plan(e.plan, Path(out_dir) / "verification")
        raise
    files += save_weld_plan(plan, out_dir)
    plan.directions = direction_sets(est, prior.spec, process, plan, base_frame=base_frame, gravity_axis_base=gravity_axis_base, k=k,
                                     bead_model=bead_model, design=design, torch=torch)
    for ds in plan.directions.values():
        save_direction_set(plan, ds, out_dir)
    return {"files": files, "plan": plan, "plan_summary": plan.summary_lines(), "contract": plan.ends.get("checked"),
            "directions": direction_sets_summary(plan)}
