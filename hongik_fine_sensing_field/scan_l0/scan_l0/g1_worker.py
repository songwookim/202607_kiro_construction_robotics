"""1G core 계산 작업 프로세스 (G1_CORE_SPEC §8.3/§8.5, S3/S4) — WP3 의 최소판 (WP5 가 g1-core 에서 정식판을 만든다; 통합에서 맞춘다,
INTERFACE_NOTES 2026-10-09 WP3 5번).

서버 (ROS 액션 서버 = I/O + 오케스트레이션) 는 numpy 맞춤을 하지 않는다: 실행마다 작업 프로세스 하나 (G1Worker) 를 띄우고 일감
'leg' / 'estimate' / 'outputs' 를 소켓 한 쌍 (길이 머리 + pickle) 으로 보낸다.

* 띄우기: ``subprocess`` (fork+exec — 스레드가 많은 서버에서 fork 만 하는 위험이 없다). 자식은 시작하자마자
  PR_SET_PDEATHSIG(SIGKILL) 을 걸고, 그 전에 부모가 이미 죽었으면 바로 끝난다 (S3). 리눅스의 '부모' 는 자식을 띄운 **스레드** 라 작업
  프로세스는 그 실행 (run) 의 스레드에서 띄우고 그 실행 안에서 닫는다 (실행 스레드가 작업 프로세스보다 오래 산다).
* 기다림 (S4): ``call`` 은 select({결과 소켓, 취소 fd}) 로만 기다린다 — 시간 한계 없음. 취소 fd 가 읽히면 바로 Cancelled.
  보내기도 같은 select (자식이 멈춰 받지 않아도 취소가 들어온다).
* 죽음 (D8): 결과 소켓의 끝 (EOF) = 자식이 죽었다 → 새로 띄우고 같은 일감을 한 번 다시 보낸다; 두 번째 죽음은 WorkerDied.
* 일감 안의 예외는 WorkerError (원격 traceback 포함) — 서버는 ERROR 로 보고한다.
* 녹화 중 맞춤 (스트림, 2026-10-10): 작업 프로세스마다 두 번째 소켓 (AF_UNIX SOCK_SEQPACKET — 메시지 하나가 통째로 가거나 안 간다)
  이 있다. 서버는 스트림 구간 (트랜싯·패스) 의 녹화기 탭 (StreamTap — 녹화 스레드에서 막지 않는 send 만) 으로 프레임 (+ 패스면 그
  프레임에 서버가 짝지은 자세 = 힌트) 을 보내고, 작업 프로세스는 그 레그의 문맥 ('stream_open') 으로 받은 행들을 덩어리마다 fork 자식에서
  맞춘다 (가장 낮은 우선순위). 이동 뒤 'leg' 일감이 오면 재구성 행과 바이트까지 같은 행 (그리고 패스면 힌트의 격자 번호가 재구성
  자세의 것과 같은 행) 만 스트림 결과를 쓰고 나머지는 그때 맞춘다 — 행 결과는 행마다의 순수 함수라 (groove_model 행 커널) 스트림이
  있든 없든·늦든·죽든 LegObs 가 같다. 스트림은 판정을 하지 않고 시간 한계도 없다; 취소는 작업 프로세스를 죽여 자식까지 끝낸다.

모형 함수 (WP1 groove_model, WP2 seam_estimate, WP4 g1_outputs) 는 ``models()`` — 정식 모듈뿐 (대역·대체 함수 없음; 가져오기 실패는
그대로 오류). 시험은 ``models`` 와 작업 프로세스의 시작 코드 ``G1Worker.CHILD_CODE`` 를 monkeypatch 한다 (scan_l0/tests/g1_model_stubs.py).
작업 프로세스는 OS 의 보통 시분할에서 가장 낮은 우선순위로 돈다 (post_scan._lowest_priority, legacy 스트림 자식과 같은 원시 — 서버의
ROS 실행기·가드를 굶기지 않는다); fork_map 의 맞춤 자식은 그것을 물려받고 스스로도 다시 건다.
"""
from __future__ import annotations

import importlib
import json
import os
import pickle
import select
import signal
import socket
import struct
import subprocess
import sys
import time
import threading
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np

__all__ = ["G1Worker", "Cancelled", "WorkerDied", "WorkerError", "atomic_savez", "models", "MODEL_FUNCS", "run_job", "fork_map",
           "StreamTap", "stream_key", "STREAM_CHUNK_ROWS"]

_HDR = struct.Struct("!Q")
_CHUNK = 1 << 20
PRIORITY: dict = {}                     # 이 프로세스에 건 CPU 우선순위 (post_scan._lowest_priority 의 답, 예: 'nice 19')
MODEL_FUNCS = {"fit_leg": "scan_core.groove_model", "fit_dwell": "scan_core.groove_model", "profile_prior": "scan_core.groove_model",
               "profile_predictor": "scan_core.seam_estimate", "estimate": "scan_core.seam_estimate",
               "site_state": "scan_core.seam_estimate", "plan_quantities": "scan_core.seam_estimate", "write_outputs": "scan_core.g1_outputs",
               "save_weld_plan": "scan_core.g1_outputs", "retract_weld_plan": "scan_core.g1_outputs",
               "delivered_length": "scan_core.seam_estimate",
               # both weld directions (user 2026-10-14): the server's D14 per direction set saves / withdraws each set
               "direction_plan": "scan_core.g1_outputs", "save_direction_set": "scan_core.g1_outputs",
               "direction_sets_summary": "scan_core.g1_outputs"}


class Cancelled(Exception):
    """취소 fd 가 읽혔다 (운전자 취소, S4)."""


class WorkerDied(Exception):
    """작업 프로세스가 같은 일감에서 두 번 죽었다 (D8)."""


class WorkerError(Exception):
    """일감 함수가 예외를 냈다 (작업 프로세스는 살아 있다). remote_tb = 원격 traceback."""

    def __init__(self, msg: str, remote_tb: str = ""):
        super().__init__(msg)
        self.remote_tb = remote_tb


# --------------------------------------------------------------------------------------
# 원자적 쓰기
# --------------------------------------------------------------------------------------
def atomic_savez(path, **arrays) -> None:
    """압축 없는 npz 를 같은 디렉터리의 임시 파일에 쓰고 os.replace (§8.5: 원자료가 먼저, 압축 없음)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:                    # 임시 파일을 아직 만들지 못했다
            pass
        raise


def atomic_write_text(path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


# --------------------------------------------------------------------------------------
# 모형 함수 고르기
# --------------------------------------------------------------------------------------
def models() -> SimpleNamespace:
    """모형 함수 묶음 = 정식 모듈 (MODEL_FUNCS). 없는 모듈·함수는 그대로 ImportError / AttributeError (대체 없음)."""
    return SimpleNamespace(**{fn: getattr(importlib.import_module(mod), fn) for fn, mod in MODEL_FUNCS.items()})


# --------------------------------------------------------------------------------------
# 일감 (작업 프로세스 안)
# --------------------------------------------------------------------------------------
def still_result(frames: list, T_base_flange, calib, z_valid_mm):
    """정지 자세 (드웰) 의 프레임들 → ScanResult: 모든 프로파일이 한 자세 (그 grab 의 실측 플랜지 자세). 움직임이 없으므로 시각 맞춤·보간
    재구성 대신 그대로 (열 수가 다르면 NaN 으로 채운다)."""
    from scan_core.types import ScanResult
    N = len(frames)
    W = max((len(np.asarray(f.x_mm)) for f in frames), default=0)
    X = np.full((N, W), np.nan, dtype=np.float32)
    Z = np.full((N, W), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        x, z = np.asarray(f.x_mm, dtype=np.float32), np.asarray(f.z_mm, dtype=np.float32)
        X[i, :x.size], Z[i, :z.size] = x, z
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(X) & np.isfinite(Z) & (Z > float(z_valid_mm[0])) & (Z < float(z_valid_mm[1]))
    T_bf = np.repeat(np.asarray(T_base_flange, dtype=float)[None], N, axis=0)
    t = np.array([float(f.t_rx_s) for f in frames]) - float(calib.dt_s)
    return ScanResult(x_mm=X, z_mm=Z, valid=valid, result_counter=np.array([int(f.result_counter) for f in frames], dtype=np.int64),
                      ts_ms=np.array([int(f.ts_ms) for f in frames], dtype=np.int64), t_rx_s=t + float(calib.dt_s), t_acq_s=t,
                      T_base_flange=T_bf, T_base_opt=T_bf @ calib.T_flange_opt, in_cruise=np.zeros(N, dtype=bool),
                      unsynced=np.zeros(N, dtype=bool), ts_flag=np.zeros(N, dtype=bool), speed_mps=np.zeros(N), s_m=np.zeros(N))


def _dwell_leg(m, res, p: dict, kw: dict):
    """드웰 grab → WP1 fit_dwell (열별 중앙값 프로파일 하나 — 정지 장면의 스펙클은 프레임 사이에 상관이라 프레임을 독립으로 결합하지 않는다)
    → LegObs 한 줄 (cls 'dwell', 방향 0 — g1_types 가 받으면; meta: site, grab, n_frames)."""
    from scan_core.g1_types import LegObs
    T = np.asarray(res.T_base_opt, dtype=float)
    df = m.fit_dwell(np.asarray(res.x_mm), np.asarray(res.z_mm), np.asarray(res.valid), p["spec"], p["tol"],
                     bevel_x_sign=p.get("bevel_x_sign"), prior=kw["prior"], T_base_opt=T[0])
    f = df.fit
    t_mid = float(np.median(np.asarray(res.t_acq_s))) if len(res.t_acq_s) else 0.0
    meta = {"kind": "dwell", "site": p.get("site"), "grab": p.get("grab"), "n_frames": int(len(res.t_acq_s)),
            "bevel_side_ln_bf": float(getattr(df, "side_ln_bf", float("nan")))}
    return LegObs(name=str(p["name"]), cls=p["cls"], direction=int(p["direction"]), t_acq_s=np.array([t_mid]), T_base_opt=T[:1],
                  v_opt_mps=np.zeros((1, 3)), theta=np.asarray(f.theta, dtype=float)[None], cov=np.asarray(f.cov, dtype=float)[None],
                  log_bf=np.array([float(f.log_bf)]), sigma_r_mm=np.array([float(f.sigma_r)]), eps=np.array([float(f.eps)]),
                  n_valid=np.array([int(f.n_valid)]), converged=np.array([bool(f.converged)]), bevel_x_sign=int(df.bevel_x_sign),
                  meta=meta)


def _job_leg(p: dict) -> dict:
    """원자료 → (압축 없이, 원자적) raw.npz 먼저 → 재구성 (드웰: 정지 자세 그대로) → recon.npz → fit_leg → profiles.npz (+ meta.json,
    시각화 점구름; 드웰이면 그 자리 상태 — WP2 site_state (spec §4.8, §8.3 'dwell': 원자료 먼저, 맞춤, 그다음 site_state))."""
    from scan_core.reconstruct import reconstruct_scan
    t: dict = {}
    out = Path(p["out_leg_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    atomic_savez(out / "raw.npz", **p["raw_arrays"], **{k: np.array(v) for k, v in p["raw_json"].items()})
    t["save_raw_s"] = time.perf_counter() - t0
    job = p["job"]
    frames, robot = p["frames"], p["robot"]
    still = p.get("kind") == "dwell"                    # 프레임 수는 서버가 정한다 (드웰: 2g + 1 이상, D9); 재구성의 수학 조건은 재구성이 본다
    t0 = time.perf_counter()
    if still:
        res = still_result(frames, p["T_base_flange"], p["calib"], p["z_valid_mm"])
    else:
        res = reconstruct_scan(frames, robot, p["calib"], job, fps_nominal=job.profile_fps, z_valid_mm=p["z_valid_mm"])
    t["reconstruct_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    atomic_savez(out / "recon.npz", **{k: np.asarray(v) for k, v in res.array_fields().items()})
    t["save_recon_s"] = time.perf_counter() - t0
    m = models()
    # §3.2 사전: c = 1/|cos ψ| (추정 전 1), σ_c (추정 전 1/cos ψ_b − 1), φ = 패스 피치 (거친 구간 0) — WP1 profile_prior (서버가 늘 보낸다)
    kw = {"prior": m.profile_prior(p["spec"], p["tol"], **p["profile_prior"])}
    t0 = time.perf_counter()
    pred_src = None
    if still:
        leg = _dwell_leg(m, res, p, kw)
    else:
        # §3.4 시작: 추정이 있으면 (패스) 그 추정이 계획한 직선 레그 위에서 예측하는 θ0, Σ_R (g1_stream.LegPredictor: 레그 위 격자, 행의
        # 격자 번호 = 재구성 자세의 레그 위치); 추정 전 (트랜싯) 은 예측이 없다 — 설계 템플릿 시작 + 앞 행의 warm start. 어느 쪽인지 구간
        # meta 'predictor' 에 남긴다; 예측기의 오류는 그대로 오류
        lp = _leg_predictor(m, p)
        if lp is not None:
            # 두 번째 시작 = 계획한 레그 위 격자의 θ0 (녹화 중에도 정해진다); 시작의 선택과 심 위치 λ = 재구성 자세의 예측 (예전과 같은
            # seam_estimate.profile_predictor) — 행 맞춤 뒤에 (값이 싸다), 스트림이 있든 없든 같은 입력으로
            kw["predict"] = lp.for_rows(np.asarray(res.T_base_flange))
            kw["seam_pred"] = m.profile_predictor(p["est"], np.asarray(res.T_base_opt), p.get("bevel_x_sign"), p["spec"])
            pred_src = (f"estimate: second start on the planned leg (g1_stream.LegPredictor, {lp.J + 1}-point grid, step "
                        f"{lp.leg.length_m / lp.J * 1e3:.2f} mm); start choice and seam-position evidence from the reconstructed poses "
                        "(seam_estimate.profile_predictor)")
        else:
            kw["predict"] = None
            pred_src = "none: no estimate yet (design-template start)"
        st = _STREAM
        use = (st is not None and p.get("stream_token") is not None and int(p["stream_token"]) == st.token
               and str(p.get("stream_key")) == st.key)
        leg = None
        if use:
            gm = st.gm
            ctx_ref = gm.leg_context(p["spec"], p["tol"], width=int(np.asarray(res.x_mm).shape[1]), prior=kw["prior"],
                                     bevel_x_sign=p.get("bevel_x_sign"), defer_seam=kw.get("seam_pred") is not None)
            try:
                leg = st.finalize(p, res, ctx_ref, kw["predict"], n_children=int(p.get("n_children", 1)), t_start=t0,
                                  seam_pred=kw.get("seam_pred"))
            except Exception:  # noqa: BLE001 - the stream is only a speed-up: fit every row after the leg (same result)
                sys.stderr.write(f"g1_worker stream finalize failed - every row fitted after the leg:\n{traceback.format_exc(limit=8)}")
                _close_stream()
                leg = None
        if leg is None:
            leg = m.fit_leg(res, p["spec"], p["tol"], cls=p["cls"], direction=int(p["direction"]), bevel_x_sign=p.get("bevel_x_sign"),
                            n_children=int(p.get("n_children", 1)), map_fn=fork_map, **kw)
        _close_stream()
    t["fit_leg_s"] = time.perf_counter() - t0
    leg.name = str(p["name"])
    # 자세 표본 주기 (로봇 상태 시각 간격의 중앙값): 시간 오프셋 사전의 폭 (seam_estimate.timing_prior) — raw.npz 를 다시 읽지 않게
    rs_ = np.sort(np.asarray((p.get("raw_arrays") or {}).get("robot_stamp_ns", np.zeros(0)), dtype=float)) * 1e-9
    d_ = np.diff(rs_)
    d_ = d_[d_ > 0]
    pose_period = {"pose_period_s": float(np.median(d_))} if d_.size else {}
    # the joint-state source's evidence of this leg (scan_core.js_source, user 2026-10-12: measured encoder angles or the control
    # box's reference echo — from the recorded /joint_states values and our commanded motions only); seam_estimate.timing_prior
    # pools the legs' evidence into the time-offset prior — stored with the leg so the estimates do not reread raw.npz
    t0 = time.perf_counter()
    js_ev = {}
    try:
        from scan_core.js_source import raw_evidence
        ra = p.get("raw_arrays") or {}
        if "robot_joint_positions_rad" in ra:
            js_ev = {"joint_state_evidence": raw_evidence({**ra, "motions_json": (p.get("raw_json") or {}).get("motions_json")})}
    except Exception as e:  # noqa: BLE001 - evidence only: the estimate then reads the leg's raw.npz itself
        t["joint_state_evidence_error"] = f"{type(e).__name__}: {e}"[:300]
    t["joint_state_evidence_s"] = time.perf_counter() - t0
    leg.meta = {**dict(leg.meta or {}), **dict(p.get("meta") or {}), **pose_period, **js_ev, "timing": dict(t),
                "cpu_priority": PRIORITY.get("applied", "unchanged (in-process)"),
                **({} if pred_src is None else {"predictor": pred_src})}
    leg.save(out / "profiles.npz")
    cloud = None
    if p.get("cloud_step_m"):
        try:
            from scan_core.cross_sections import decimated_cloud
            cloud = decimated_cloud(res, float(p["cloud_step_m"]))
        except Exception as e:  # noqa: BLE001 - visualisation only
            t["cloud_error"] = f"{type(e).__name__}: {e}"
    lb = np.asarray(leg.log_bf, dtype=float)
    summary = {"name": leg.name, "cls": leg.cls, "direction": leg.direction, "n_frames": len(frames), "n_profiles": leg.n,
               "groove_profiles": int(np.sum(lb > 0.0)), "converged": int(np.sum(np.asarray(leg.converged, dtype=bool))),
               "sigma_r_mm_median": float(np.nanmedian(leg.sigma_r_mm)) if leg.n else float("nan"),
               "outlier_frac_median": float(np.nanmedian(leg.eps)) if leg.n else float("nan"),
               "bevel_x_sign": int(leg.bevel_x_sign),
               "bevel_ln_bf": next((leg.meta[k_] for k_ in ("bevel_side_ln_bf", "bevel_ln_bf") if k_ in (leg.meta or {})), None),
               "stream": (leg.meta or {}).get("stream")}
    atomic_write_text(out / "meta.json", json.dumps({"summary": summary, "timing": t, "meta": _plainish(leg.meta)}, indent=1,
                                                     default=str))
    res_out = {"leg_path": str(out / "profiles.npz"), "summary": summary, "cloud": cloud, "timing": t,
               "bevel_x_sign": int(leg.bevel_x_sign)}
    if still:
        # §4.8 (D15 의 입력): 이 grab (N = 1) 과 지금 추정 (두 번째 자리부터) → 루트·자세 σ, 목표 자세 T*, 보정 필요 여부
        t0 = time.perf_counter()
        res_out["site_state"] = m.site_state([leg], p.get("est"), p["prior"], str(p.get("site")), T_now=p["T_base_flange"],
                                             calib=p["calib"], view=p["view"], goal_tol_m=float(p["goal_tol_m"]))
        t["site_state_s"] = time.perf_counter() - t0
        res_out["log_bf"] = float(lb[0]) if leg.n else float("nan")
        res_out["converged"] = bool(np.asarray(leg.converged, dtype=bool)[0]) if leg.n else False
    return res_out


def _job_estimate(p: dict):
    from scan_core.g1_types import LegObs
    m = models()
    legs = [LegObs.load(pth) for pth in p["leg_paths"]]
    t0 = time.perf_counter()
    est = m.estimate(legs, p["prior"], map_fn=fork_map, n_children=int(p.get("n_children", 1)))   # plate-end readings in fork children
    dt = time.perf_counter() - t0
    if p.get("est_path"):
        d = est.to_dict()
        d["timing_s"] = dt
        atomic_write_text(p["est_path"], json.dumps(d, indent=1, default=str))
    return est


def _job_outputs(p: dict) -> dict:
    """g1_outputs.write_outputs (WP4): sections.* + weld_plan.* → out_dir. 계약 (S→E) 거부면 단면은 쓰고 거부된 계획은 verification/ —
    여기서는 plan None + plan_error (서버가 D14 DATA_FAIL). 서버는 도달성 (check_weld_plan) 뒤 save_weld_plan / retract_weld_plan.
    The plan carries both weld directions (plan.directions, written to out_dir/directions/<set>/ here; 'directions' = their
    summary): the server runs D14 per set that is not the primary plan and saves or withdraws each (save_direction_set)."""
    m = models()
    t0 = time.perf_counter()
    try:
        out = m.write_outputs(p["est"], p["prior"], p["process"], p["out_dir"], base_frame=p["base_frame"],
                              gravity_axis_base=p["gravity_axis_base"], k=float(p["k"]), run=p.get("run"), design=p.get("design"),
                              torch=p.get("torch"))
    except Exception as e:  # noqa: BLE001 - PlanContractError (D14): sections are written, the plan is in verification/
        if type(e).__name__ != "PlanContractError":
            raise
        files = [str(Path(p["out_dir"]) / nm) for nm in ("sections.json", "sections.npz") if (Path(p["out_dir"]) / nm).exists()]
        return {"files": files, "plan": None, "plan_error": f"{type(e).__name__}: {e}", "plan_summary": None,
                "timing": {"outputs_s": time.perf_counter() - t0}}
    return {"files": list(out.get("files") or []), "plan": out.get("plan"), "plan_error": None, "plan_summary": out.get("plan_summary"),
            "contract": out.get("contract"), "directions": out.get("directions"), "timing": {"outputs_s": time.perf_counter() - t0}}


def fork_map(fn, chunks: list, n: int) -> list:
    """덩어리마다 fork 한 자식 (PDEATHSIG + 부모 재확인, 소켓 한 쌍) 에서 fn(덩어리) → 결과 목록 (덩어리 순서). 자식이 죽거나 실패하면 그
    덩어리는 이 프로세스에서 다시 계산한다 (§8.5). 작업 프로세스 안 (스레드 하나) 에서만 부른다 — fn 은 피클하지 않는다 (fork)."""
    me = os.getpid()
    out: list = [None] * len(chunks)
    todo = list(enumerate(chunks))
    width = max(1, int(n))
    while todo:
        batch, todo = todo[:width], todo[width:]
        kids = []
        for i, ch in batch:
            a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            pid = os.fork()
            if pid == 0:                                                    # 자식
                code = 0
                try:
                    a.close()
                    _set_pdeathsig()
                    if os.getppid() != me:
                        os._exit(1)
                    _lowest()                                               # 맞춤 자식: 서버·가드를 굶기지 않는다
                    try:
                        res = ("ok", fn(ch))
                    except BaseException as e:  # noqa: BLE001 - goes back to the parent, which recomputes and reports
                        res = ("failed", f"{type(e).__name__}: {e}")
                    _send_blocking(b, res)
                except BaseException:  # noqa: BLE001
                    code = 1
                finally:
                    os._exit(code)
            b.close()
            kids.append((i, ch, pid, a))
        for i, ch, pid, a in kids:
            try:
                res = _recv_blocking(a)
            except (EOFError, OSError) as e:
                res = ("died", f"{type(e).__name__}: {e}")
            finally:
                a.close()
                os.waitpid(pid, 0)                                       # 이 자식은 여기서만 거둔다
            if res[0] == "ok":
                out[i] = res[1]
            else:                                                        # §8.5: 그 덩어리는 이 프로세스에서 — 알리고 다시 계산
                sys.stderr.write(f"g1_worker fork_map: chunk {i} child {res[0]} ({res[1]}) - recomputed in the worker\n")
                out[i] = fn(ch)
    return out


# --------------------------------------------------------------------------------------
# 녹화 중 맞춤 (스트림): 서버 탭 → 작업 프로세스 → 덩어리 자식 (행 커널)
# --------------------------------------------------------------------------------------
STREAM_CHUNK_ROWS = 16              # [R] 스트림 자식 하나가 맡는 행 수 (일괄 GN 의 묶음 효율 ↔ 이동 끝에 남는 꼬리) — 결과와 무관 (행 커널)
HINT_EVERY_FRAMES = 10              # [R] 패스 힌트 (서버가 짝지은 자세) 를 붙이는 프레임 간격 — 사이 프레임은 작업 프로세스가 레그 위치를
#                                     보간한다 (격자 번호 추정; 녹화 스레드의 pose_at 비용 ↓) — 결과와 무관 (틀린 격자 번호의 행은 이동 뒤 다시)
_S_HDR = struct.Struct("<cQqqqI")   # 종류 (b'F' 프레임 | b'H' 힌트 | b'N' 힌트 없음), 토큰, result_counter, t_rx_ns, ts_ms, 점 수
_S_POSE = struct.Struct("<16d")     # 힌트: 서버가 그 프레임에 짝지은 T_base←flange (행 우선)
_S_BUF = 1 << 18                    # 받기 버퍼 [바이트] (프레임 하나 = 머리 + 점 수 × 8; 센서 1408 점 ≈ 11 KB)
_SSOCK: Optional[socket.socket] = None   # 작업 프로세스 안: 스트림 소켓 (없으면 스트림 없음)
_STREAM = None                            # 작업 프로세스 안: 지금 레그의 스트림 (_LegStream)


def stream_key(ctx: dict) -> str:
    """스트림 ('stream_open') 과 'leg' 일감이 같은 레그 문맥인가 — 서버가 같은 객체에서 한 번 만들어 둘 다에 넣는다 (행 결과의 입력:
    설계·허용치·사전·거울·부류·z 창·패스면 추정과 계획)."""
    import hashlib
    keys = ("spec", "tol", "profile_prior", "bevel_x_sign", "cls", "z_valid_mm", "est", "leg_plan")
    return hashlib.blake2b(pickle.dumps(tuple(ctx.get(k) for k in keys), protocol=4), digest_size=16).hexdigest()


class StreamTap:
    """서버 쪽 녹화기 탭 (SensorSession.add_tap): 녹화 스레드가 프레임마다 부른다. 프레임 (그리고 pose_at 이 있으면 그 프레임의 수신 시각
    − dt 에 플랫폼 자세 이력이 주는 자세 = 힌트; 아직 없으면 다음 프레임 때 다시, 이력 밖이면 '힌트 없음') 을 작업 프로세스의 스트림
    소켓에 **막지 않고** 보낸다 (MSG_DONTWAIT; SEQPACKET 은 통째로 가거나 EAGAIN). 소켓이 차면 (작업 프로세스가 밀리면) 보내지 못한
    것을 순서대로 들고 있다가 다음 프레임 때 잇는다 — 녹화가 끝날 때 남은 것은 버린다 (그 행은 이동 뒤 'leg' 일감이 맞춘다). 어떤 실패도
    녹화로 올라가지 않는다 (dead 에 남기고 멈춘다). 힌트는 속도에만 쓰인다 (결과는 재구성 자세로 정해진다)."""

    def __init__(self, sock: socket.socket, token: int, *, pose_at: Optional[Callable] = None, dt_s: float = 0.0,
                 hint_every: int = HINT_EVERY_FRAMES):
        self.sock, self.token, self.pose_at, self.dt_s = sock, int(token), pose_at, float(dt_s)
        self.hint_every = max(1, int(hint_every))
        self._last = None                 # 마지막 프레임 (close 때 힌트가 없으면 붙인다 — 끝 행들의 보간)
        self._q: deque = deque()          # 보낼 메시지 (조각 목록) 순서대로
        self._wait: deque = deque()       # 힌트를 기다리는 프레임
        self._lock = threading.Lock()     # 녹화 스레드 (탭) ↔ 서버 스레드 (close)
        self.frames = self.hints = self.no_hints = self.sent = self.blocked = self.calls = 0
        self.dead: Optional[str] = None
        self.closed = False
        self.cpu_s = 0.0

    def __call__(self, rec) -> None:
        t0 = time.thread_time()
        with self._lock:
            if self.closed or self.dead:
                return
            x = np.ascontiguousarray(rec.x, dtype=np.float32).reshape(-1)
            z = np.ascontiguousarray(rec.z, dtype=np.float32).reshape(-1)
            self._q.append([_S_HDR.pack(b"F", self.token, int(rec.result_counter), int(rec.t_rx_ns), int(rec.ts_ms), x.size),
                            memoryview(x).cast("B"), memoryview(z).cast("B")])
            self.frames += 1
            if self.pose_at is not None:
                if (self.frames - 1) % self.hint_every == 0:
                    self._wait.append(rec)
                    self._last = None
                else:
                    self._last = rec
                self._resolve()
            self._flush()
            self.calls += 1
        self.cpu_s += time.thread_time() - t0

    def _resolve(self) -> None:
        while self._wait:
            rec = self._wait[0]
            try:
                T = self.pose_at(rec.t_rx_ns * 1e-9 - self.dt_s)
            except Exception:  # noqa: BLE001 - LookupError (outside the history) or any platform failure: no hint for this frame
                T = False
            if T is None:                                     # 그 시각의 자세가 아직 오지 않았다 — 뒤 프레임도 그렇다
                return
            self._wait.popleft()
            if T is False:
                self._q.append([_S_HDR.pack(b"N", self.token, int(rec.result_counter), 0, 0, 0)])
                self.no_hints += 1
            else:
                self._q.append([_S_HDR.pack(b"H", self.token, int(rec.result_counter), 0, 0, 0),
                                _S_POSE.pack(*np.asarray(T, dtype=float).reshape(16))])
                self.hints += 1

    def _flush(self) -> None:
        while self._q:
            try:
                self.sock.sendmsg(self._q[0], [], socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                self.blocked += 1
                return
            except OSError as e:                               # 작업 프로세스가 죽었다 등 — 스트림만 멈춘다
                self.dead = f"{type(e).__name__}: {e}"
                self._q.clear()
                self._wait.clear()
                return
            self._q.popleft()
            self.sent += 1

    def close(self) -> dict:
        """녹화가 끝난 뒤 (서버 스레드): 힌트를 한 번 더 풀고 들어가는 만큼 보낸 뒤 (막지 않는다) 멈춘다. → 기록."""
        t0 = time.thread_time()
        with self._lock:
            if not self.closed and not self.dead:
                if self.pose_at is not None:
                    if self._last is not None:
                        self._wait.append(self._last)
                        self._last = None
                    self._resolve()
                self._flush()
            left = len(self._q) + len(self._wait)
            self._q.clear()
            self._wait.clear()
            self.closed = True
        self.cpu_s += time.thread_time() - t0
        return {"frames": self.frames, "messages_sent": self.sent, "hints": self.hints, "no_hints": self.no_hints,
                "send_blocked": self.blocked, "left_unsent": left, "tap_cpu_s": self.cpu_s, "dead": self.dead}


def _leg_predictor(m, p: dict):
    """패스의 두 번째 시작·심 위치 사전 (g1_stream.LegPredictor): 계획한 직선 레그 (p['leg_plan']) + 추정 (p['est']). 추정이 없으면 None
    (트랜싯: 설계 템플릿 시작 + warm start)."""
    if p.get("est") is None:
        return None
    from scan_core.g1_stream import LegPredictor, PlannedLeg
    lp = p.get("leg_plan")
    if lp is None:
        raise ValueError("a leg with an estimate needs its leg_plan (T_from, T_to, T_flange_opt, step_m) for the prediction")
    return LegPredictor(p["est"], PlannedLeg.from_dict(lp), float(lp["step_m"]), p.get("bevel_x_sign"), p["spec"], cls=str(p["cls"]),
                        profile_predictor=m.profile_predictor)


def _fork_task(fn, arg) -> tuple:
    """fn(arg) 을 fork 한 자식 하나에서 (PDEATHSIG + 부모 재확인, 가장 낮은 우선순위, 소켓 한 쌍) → (pid, 부모 쪽 소켓 (논블로킹))."""
    me = os.getpid()
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    pid = os.fork()
    if pid == 0:                                                          # 자식
        code = 0
        try:
            a.close()
            _set_pdeathsig()
            if os.getppid() != me:
                os._exit(1)
            _lowest()
            try:
                res = ("ok", fn(arg))
            except BaseException as e:  # noqa: BLE001 - goes back to the worker, which computes these rows itself
                res = ("failed", f"{type(e).__name__}: {e}", traceback.format_exc(limit=6))
            _send_blocking(b, res)
        except BaseException:  # noqa: BLE001
            code = 1
        finally:
            os._exit(code)
    b.close()
    a.setblocking(False)
    return pid, a


class _LegStream:
    """작업 프로세스 안: 한 레그의 스트림 — 받은 행 (프레임) 과 힌트, 덩어리 자식, 끝난 행 결과. 'leg' 일감의 finalize 가 재구성 행과
    맞춘다 (모듈 설명)."""

    def __init__(self, p: dict, ssock: socket.socket):
        from scan_core import groove_model as gm
        self.gm = gm
        self.token, self.key = int(p["stream_token"]), str(p["stream_key"])
        self.ssock = ssock
        self.spec, self.tol, self.cls = p["spec"], p["tol"], str(p["cls"])
        self.prior = models().profile_prior(p["spec"], p["tol"], **p["profile_prior"])
        self.bevel_x_sign = p.get("bevel_x_sign")
        self.z_valid = tuple(float(v) for v in p["z_valid_mm"])
        self.n_children = max(1, int(p.get("n_children", 1)))
        self.chunk = max(1, int(p.get("chunk_rows", STREAM_CHUNK_ROWS)))
        self.pred = _leg_predictor(models(), p)
        self.ctx: Optional[dict] = None
        self.W: Optional[int] = None
        self.counter: list = []
        self.X = self.Z = self.V = self.nrow = None
        self.n = 0
        self.k_of: dict = {}                    # result_counter → 행 (중복이면 -1)
        self.hint_rows: list = []               # 힌트가 온 행 (오름차순; 힌트는 행 순서로 온다)
        self.hint_pos: list = []                # 그 행의 플랜지 위치 (3,) | None (힌트 없음)
        self.n_hint = 0                         # 레그 위치를 정할 수 있는 앞쪽 행 수 (= 마지막 유효 힌트 행 + 1)
        self.next = 0                           # 다음에 덩어리로 낼 행
        self.kids: dict = {}                    # fd → {pid, sock, rows, buf, kind}
        self.done: dict = {}                    # 행 → {sign: {key: 값}}
        self.jhat: dict = {}                    # 행 → 힌트의 격자 번호 (패스)
        self.buf = bytearray(_S_BUF)
        self.stats = {"frames": 0, "hints": 0, "no_hints": 0, "stale": 0, "truncated": 0, "bad_width": 0, "chunks": 0,
                      "chunk_failures": [], "worker_pid": os.getpid()}

    # ---- 받기 ----
    def _grow(self, need: int) -> None:
        cap = 0 if self.X is None else self.X.shape[0]
        if need <= cap:
            return
        new = max(need, 2 * cap, 256)
        for nm, dt, fill in (("X", np.float32, np.nan), ("Z", np.float32, np.nan), ("V", bool, False)):
            a = np.full((new, self.W), fill, dtype=dt)
            if cap:
                a[:cap] = getattr(self, nm)
            setattr(self, nm, a)
        nr = np.zeros(new, dtype=np.int64)
        if cap:
            nr[:cap] = self.nrow
        self.nrow = nr

    def _take(self, mv: memoryview) -> None:
        kind, tok, cnt, _t_rx, _ts, n = _S_HDR.unpack_from(mv, 0)
        if tok != self.token:
            self.stats["stale"] += 1
            return
        if kind == b"F":
            if self.W is None:
                self.W = int(n)
                self.ctx = self.gm.leg_context(self.spec, self.tol, width=self.W, prior=self.prior, bevel_x_sign=self.bevel_x_sign,
                                               defer_seam=self.pred is not None)
            if int(n) != self.W:                       # 점 수가 바뀐 프레임: 재구성은 그 행을 비운다 — 스트림에 넣지 않는다 (이동 뒤 맞춤)
                self.stats["bad_width"] += 1
                self.k_of[int(cnt)] = -1
                return
            k = self.n
            self._grow(k + 1)
            o = _S_HDR.size
            self.X[k] = np.frombuffer(mv, dtype=np.float32, count=n, offset=o)
            self.Z[k] = np.frombuffer(mv, dtype=np.float32, count=n, offset=o + 4 * n)
            from scan_core.validity import valid_mask
            self.V[k] = valid_mask(self.X[k:k + 1], self.Z[k:k + 1], self.z_valid)[0]
            self.nrow[k] = self.gm.row_counts(self.X[k:k + 1], self.Z[k:k + 1], self.V[k:k + 1])[0]
            self.counter.append(int(cnt))
            self.k_of[int(cnt)] = -1 if int(cnt) in self.k_of else k
            self.n = k + 1
            self.stats["frames"] += 1
        elif kind in (b"H", b"N"):
            k = self.k_of.get(int(cnt), -1)
            if k >= 0 and (not self.hint_rows or k > self.hint_rows[-1]):
                pos = (np.array(_S_POSE.unpack_from(mv, _S_HDR.size), dtype=float).reshape(4, 4)[:3, 3].copy() if kind == b"H" else None)
                self.hint_rows.append(k)
                self.hint_pos.append(pos)
                self.stats["hints" if kind == b"H" else "no_hints"] += 1
                if pos is not None:
                    self.n_hint = k + 1

    def _hint_position(self, k: int):
        """행 k 의 플랜지 위치 힌트: 힌트 행이면 그것, 아니면 앞뒤 유효 힌트 행 사이의 선형 보간 (프레임은 같은 간격) — 앞이나 뒤가
        없으면 None (그 행은 이동 뒤에 맞춘다)."""
        import bisect
        j = bisect.bisect_left(self.hint_rows, k)
        if j < len(self.hint_rows) and self.hint_rows[j] == k and self.hint_pos[j] is not None:
            return self.hint_pos[j]
        lo = next((i for i in range(j - 1, -1, -1) if self.hint_pos[i] is not None), None)
        hi = next((i for i in range(j, len(self.hint_rows)) if self.hint_pos[i] is not None), None)
        if lo is None or hi is None:
            return None
        a, b = self.hint_rows[lo], self.hint_rows[hi]
        w = (k - a) / float(b - a)
        return (1.0 - w) * self.hint_pos[lo] + w * self.hint_pos[hi]

    def drain(self) -> None:
        """스트림 소켓에 와 있는 메시지를 모두 (기다리지 않는다)."""
        while True:
            try:
                nb, _anc, flags, _addr = self.ssock.recvmsg_into([self.buf], 0, socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                return
            if nb == 0:
                return
            if flags & socket.MSG_TRUNC:
                self.stats["truncated"] += 1
                continue
            self._take(memoryview(self.buf)[:nb])

    # ---- 덩어리 자식 ----
    def _ready(self) -> int:
        return self.n if self.pred is None else self.n_hint

    def _stream_rows(self, rows: list) -> dict:
        """(자식 안) 스트림 행 rows 의 맞춤: 패스는 힌트가 있는 행만 (격자 번호 = 힌트 자세의 레그 위치), 트랜싯은 모두 (warm start)."""
        n = self.n
        X, Z, V = self.X[:n], self.Z[:n], self.V[:n]
        jj, predict = {}, None
        if self.pred is not None:
            pos = {k: self._hint_position(k) for k in rows}
            rows = [k for k in rows if pos[k] is not None]
            jj = {k: int(self.pred.index_of_positions(pos[k][None])[0]) for k in rows}
            predict = (lambda i: self.pred(jj[int(i)]))
        if not rows:
            return {"rows": [], "res": None, "j": {}}
        return {"rows": rows, "res": self.gm.fit_rows(self.ctx, X, Z, V, rows, n_rows=self.nrow[:n], predict=predict), "j": jj}

    def _spawn(self, fn, rows: list, kind: str) -> None:
        pid, sk = _fork_task(fn, rows)
        self.kids[sk.fileno()] = {"pid": pid, "sock": sk, "rows": list(rows), "buf": bytearray(), "kind": kind}
        if kind == "stream":
            self.stats["chunks"] += 1

    def dispatch(self) -> None:
        while len(self.kids) < self.n_children and self.ctx is not None and self.next + self.chunk <= self._ready():
            rows = list(range(self.next, self.next + self.chunk))
            self.next += self.chunk
            self._spawn(self._stream_rows, rows, "stream")

    def _kid_readable(self, fd: int) -> Optional[tuple]:
        """자식 소켓이 읽힌다: 있는 만큼 모은다 → (자식, 결과 | None = 죽음) 이면 끝난 자식, 아직이면 None."""
        kid = self.kids[fd]
        dead = False
        while True:
            try:
                chunk = kid["sock"].recv(_CHUNK)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                dead = True
                break
            if not chunk:
                dead = True
                break
            kid["buf"] += chunk
        b = kid["buf"]
        if len(b) >= _HDR.size:
            (need,) = _HDR.unpack_from(b, 0)
            if len(b) >= _HDR.size + need:
                del self.kids[fd]
                kid["sock"].close()
                os.waitpid(kid["pid"], 0)
                return kid, pickle.loads(bytes(b[_HDR.size:_HDR.size + need]))
        if dead:
            del self.kids[fd]
            kid["sock"].close()
            os.waitpid(kid["pid"], 0)
            return kid, None
        return None

    def _store_stream(self, kid: dict, out) -> None:
        if out is None or out[0] != "ok":
            self.stats["chunk_failures"].append({"rows": [kid["rows"][0], kid["rows"][-1]],
                                                 "why": "child died" if out is None else out[1]})
            return
        r = out[1]
        if r["res"] is None:
            return
        for q, k in enumerate(r["rows"]):
            self.done[k] = {s: {key: v[q] for key, v in R.items()} for s, R in r["res"].items()}
            if k in r["j"]:
                self.jhat[k] = r["j"][k]

    def pump(self, job_sock: socket.socket) -> None:
        """녹화 동안 (서버는 일감을 보내지 않는다): 스트림 받기 → 덩어리 내기 → 끝난 자식 거두기, 일감 소켓이 읽힐 때까지."""
        while True:
            self.dispatch()
            rl = [job_sock, self.ssock] + [k["sock"] for k in self.kids.values()]
            r, _, _ = select.select(rl, [], [])
            for sk in r:
                if sk is job_sock or sk is self.ssock:
                    continue
                got = self._kid_readable(sk.fileno())
                if got is not None:
                    self._store_stream(*got)
            if self.ssock in r:
                self.drain()
            if job_sock in r:
                return

    def close(self) -> None:
        """남은 자식을 끝낸다 (SIGKILL + 거두기)."""
        for kid in list(self.kids.values()):
            try:
                os.kill(kid["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            kid["sock"].close()
            os.waitpid(kid["pid"], 0)
        self.kids.clear()

    # ---- 이동 뒤: 재구성 행과 맞추기 ----
    def finalize(self, p: dict, res, ctx_ref: dict, predict_ref, *, n_children: int, t_start: float, seam_pred=None):
        """'leg' 일감: 스트림 결과 중 재구성 행 i 와 같은 입력으로 계산된 것만 쓰고 (같은 result_counter, x·z·유효 바이트 같음; 패스면
        힌트 격자 번호 = 재구성 자세의 격자 번호; 트랜싯이면 warm start 의 앞 행도 같은 행), 덩어리 자식이 계산 중인 행은 기다리고,
        나머지는 지금 맞춘다 (자식 n_children 개까지 나란히). → LegObs (assemble_leg — 사후 fit_leg 와 같은 함수)."""
        gm = self.gm
        t_fin = time.perf_counter()
        self.drain()
        Xp, Zp = np.asarray(res.x_mm), np.asarray(res.z_mm)
        Vp = np.asarray(res.valid, dtype=bool)
        N = int(Xp.shape[0])
        nrow_p = gm.row_counts(Xp, Zp, Vp)
        cnt = np.asarray(res.result_counter, dtype=np.int64)
        j_ref = self.pred.index(res.T_base_flange) if self.pred is not None else None
        same_w = self.W is not None and Xp.ndim == 2 and int(Xp.shape[1]) == int(self.W)
        mk = np.full(N, -1, dtype=np.int64)
        if same_w:
            for i in range(N):
                k = self.k_of.get(int(cnt[i]), -1)
                if k >= 0 and Xp[i].tobytes() == self.X[k].tobytes() and Zp[i].tobytes() == self.Z[k].tobytes() \
                        and Vp[i].tobytes() == self.V[k].tobytes():
                    mk[i] = k
        warm = self.pred is None and bool(ctx_ref["warm"])
        lb_p = gm.lookbacks(nrow_p)                                   # warm start 의 앞 행 (재구성 순서)
        lb_s = gm.lookbacks(self.nrow[:self.n]) if self.n else np.zeros(0, dtype=np.int64)   # (스트림 순서)

        def usable(i: int) -> bool:
            k = int(mk[i])
            if k < 0 or k not in self.done:
                return False
            if self.pred is not None:
                return self.jhat.get(k) == int(j_ref[i])
            if warm:
                li, lk = int(lb_p[i]), int(lb_s[k])
                return (li < 0 and lk < 0) or (li >= 0 and lk >= 0 and int(mk[li]) == lk)
            return True

        rows: dict = {}
        for i in range(N):
            if usable(i):
                rows[i] = self.done[int(mk[i])]
        n_stream = len(rows)
        inflight = {}                                   # 재구성 행 → 그 행을 계산 중인 스트림 자식 fd
        k2i = {int(k): i for i, k in enumerate(mk) if k >= 0}
        for fd, kid in self.kids.items():
            for k in kid["rows"]:
                i = k2i.get(int(k))
                if i is not None and i not in rows:
                    inflight[i] = fd
        todo = [i for i in range(N) if i not in rows and i not in inflight]

        def post_rows(rr: list) -> dict:
            return {"rows": rr, "res": gm.fit_rows(ctx_ref, Xp, Zp, Vp, rr, n_rows=nrow_p, predict=predict_ref)}

        n_waited = len(inflight)
        queue: deque = deque()                          # 이동 뒤 맞출 행 조각 (자식 하나에 하나) — 자식 n_children 개가 고르게 나눠 갖는 크기

        def enqueue(rr: list) -> None:
            """조각 크기 = min(덩어리 행 수, 행 수 / 자식 수) — 자식이 고르게 바쁘고, 묶음이 너무 작지 않게 (결과와 무관)."""
            if not rr:
                return
            piece = max(1, min(self.chunk, -(-len(rr) // self.n_children)))
            for part in np.array_split(np.asarray(sorted(rr), dtype=np.int64), -(-len(rr) // piece)):
                queue.append([int(v) for v in part])
        enqueue(todo)
        todo = []
        while len(rows) < N:
            while queue and len(self.kids) < self.n_children:
                self._spawn(post_rows, queue.popleft(), "post")
            if not self.kids:
                break
            r, _, _ = select.select([k["sock"] for k in self.kids.values()], [], [])
            for sk in r:
                got = self._kid_readable(sk.fileno())
                if got is None:
                    continue
                kid, out = got
                if kid["kind"] == "stream":
                    self._store_stream(kid, out)
                    for k in kid["rows"]:
                        i = k2i.get(int(k))
                        if i is None or i in rows:
                            continue
                        inflight.pop(i, None)
                        if usable(i):
                            rows[i] = self.done[int(mk[i])]
                        else:
                            todo.append(i)
                    enqueue(todo)
                    todo = []
                else:
                    if out is None or out[0] != "ok":   # 자식이 죽었다·실패했다 (§8.5): 그 행은 이 프로세스에서 — 알리고 다시
                        why = "died" if out is None else out[1]
                        sys.stderr.write(f"g1_worker stream finalize: rows {kid['rows'][0]}..{kid['rows'][-1]} child {why} - "
                                         "recomputed in the worker\n")
                        out = ("ok", post_rows(kid["rows"]))
                    rr = out[1]
                    for q, i in enumerate(rr["rows"]):
                        rows[int(i)] = {s: {key: v[q] for key, v in R.items()} for s, R in rr["res"].items()}
        if len(rows) < N:                               # 자식 없이 남은 행 (자식 수 0 등): 이 프로세스에서
            rest = [i for i in range(N) if i not in rows]
            rr = post_rows(rest)
            for q, i in enumerate(rr["rows"]):
                rows[int(i)] = {s: {key: v[q] for key, v in R.items()} for s, R in rr["res"].items()}
        merged = {}
        for s in ctx_ref["tpls"]:
            merged[s] = {key: np.stack([rows[i][s][key] for i in range(N)]) for key in gm.ROW_KEYS} if N else {}
            if N and any("_cand" in rows[i][s] for i in range(N)):
                merged[s]["_cand"] = [rows[i][s].get("_cand") for i in range(N)]
        if seam_pred is not None:                          # 시작의 선택·심 위치 λ: 재구성 자세의 예측 (사후 맞춤과 같은 함수·입력)
            gm.rows_select(merged, range(N), seam_pred)
        st = dict(self.stats)
        from_stream = int(sum(1 for i in range(N) if int(mk[i]) in self.done and rows[i] is self.done[int(mk[i])]))
        st.update(rows=N, rows_matched=int(np.count_nonzero(mk >= 0)), rows_from_stream=from_stream,
                  rows_ready_at_leg_job=n_stream, rows_in_flight_at_leg_job=n_waited, rows_after_leg=N - from_stream,
                  hint_mismatch=(int(sum(1 for i in range(N) if mk[i] >= 0 and int(mk[i]) in self.jhat
                                         and self.jhat[int(mk[i])] != int(j_ref[i]))) if self.pred is not None else 0),
                  finalize_s=time.perf_counter() - t_fin)
        return gm.assemble_leg(res, p["spec"], ctx_ref["prior"], ctx_ref, merged, cls=p["cls"], direction=int(p["direction"]),
                               name=str(p.get("name", "leg")), predict=seam_pred if seam_pred is not None else predict_ref,
                               n_children=n_children, fan=ctx_ref["fan"],
                               t_start=t_start, meta_extra={"stream": st})


def _close_stream() -> None:
    global _STREAM
    if _STREAM is not None:
        try:
            _STREAM.close()
        finally:
            _STREAM = None


def _job_stream_open(p: dict) -> dict:
    """녹화 전 (서버): 이 레그의 문맥으로 스트림을 연다 (앞 스트림은 닫는다). 스트림 소켓이 없으면 (시험 대역 등) 스트림 없음."""
    global _STREAM
    _close_stream()
    if _SSOCK is None:
        return {"stream": None, "why": "this worker has no stream channel"}
    while True:                                       # 앞 레그의 남은 메시지는 버린다 (토큰이 다르다)
        try:
            _SSOCK.recv(_S_BUF, socket.MSG_DONTWAIT)
        except (BlockingIOError, InterruptedError):
            break
    _STREAM = _LegStream(p, _SSOCK)
    return {"stream": int(p["stream_token"]), "pid": os.getpid(), "chunk_rows": _STREAM.chunk}


_ORIENT: dict = {}                       # the torch-orientation search of this run (weld_orient): plan, tool, rows, searches per (pass, roll)
_MAIN_SOCK: Optional[socket.socket] = None   # the worker's job socket to the server (closed in the search children)


def _job_orient_open(p: dict) -> dict:
    """The weld-pose clearance search (scan_core.weld_orient.search_pass) for this run's plan: plan, tool body (the platform's URDF
    collision geometry), weld torch, required clearance [mm], the delivered lines' sample step [m] (step_m: the platform's Cartesian
    step - the same step choose_orientation records). The numpy work runs here, not in the server (GIL): each (pass, roll) search in
    its own forked child of this process (orient_ask / orient_take - the passes in parallel; _OrientSearch)."""
    from scan_core.weld_clearance import warm_caches
    from scan_core.weld_line import line_step_m
    from scan_core.weld_orient import line_rows
    _orient_close()
    plan = p["plan"]
    st_m = line_step_m(plan, p.get("step_m"))[0]
    _ORIENT.update(plan=plan, body=p["body"], torch=p["torch"], required_mm=float(p["required_mm"]), searches={}, reach=p.get("reach"),
                   rows={wp.k: line_rows(plan, wp, st_m) for wp in plan.passes}, step_m=st_m)
    warm_caches(p["body"])                     # once here: the searches' children inherit them
    return {"passes": sorted(_ORIENT["rows"]), "pid": os.getpid(), "step_m": st_m}


def _orient_gen(k: int, roll: float, bound):
    """search_pass of pass k at TCP roll 'roll' on this run's plan (orient_open), cut at max_deviation_deg = bound."""
    from scan_core.weld_orient import search_pass
    wp = {w.k: w for w in _ORIENT["plan"].passes}[int(k)]
    return search_pass(_ORIENT["plan"], wp, _ORIENT["body"], _ORIENT["torch"], float(roll), _ORIENT["required_mm"],
                       rows=_ORIENT["rows"][int(k)], reach=_ORIENT.get("reach"), max_deviation_deg=bound)


def _orient_step(gen) -> tuple:
    """The generator's next candidate as ('ok', dict) - the grid's end as 'exhausted' -, or ('err', text, traceback)."""
    try:
        try:
            return ("ok", next(gen))
        except StopIteration:
            return ("ok", {"exhausted": True, "n_evaluated": None, "best_seen_mm": float("nan"), "best_seen_at": None})
    except BaseException as e:  # noqa: BLE001 - goes back to the server as the job's failure
        return ("err", f"{type(e).__name__}: {e}", traceback.format_exc(limit=12))


class _OrientSearch:
    """One (pass, roll) candidate search of this run in a forked child of the worker (the plan, the tool and the rows inherited, no
    pickling): requests (id, bound) answered in order with the generator's next candidate - the searches of the passes run in parallel
    while the server waits for one of them or for the robot check. The answers are the generator's, whoever computes them: a child
    that dies is replaced here by the same generator replayed past the answers already received (deterministic), so the candidates and
    their order never depend on the children (fork_map's rule: a dead child's work is redone in the worker)."""

    def __init__(self, key: tuple, bound):
        self.key, self.bound = key, bound
        self.next_id = 0
        self.pending: list = []           # request ids sent to the child, not yet answered
        self.got: dict = {}               # id -> ('ok', candidate) | ('err', ...)
        self.n_recv = 0                   # answers received from the child (the replay length if it dies)
        self.gen = None                   # the generator here (after a child's death, or when no child could be forked)
        self.pid, self.sock = None, None
        me = os.getpid()
        try:
            a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        except OSError:
            return
        pid = os.fork()
        if pid == 0:                                                        # the child
            code = 0
            try:
                a.close()
                for s_ in (_MAIN_SOCK, _SSOCK):                             # never the server's sockets
                    if s_ is not None:
                        s_.close()
                _set_pdeathsig()
                if os.getppid() != me:
                    os._exit(1)
                _lowest()                                                   # like fork_map's children: never starve the server
                gen = None
                while True:
                    try:
                        rid, bnd = _recv_blocking(b)
                    except EOFError:
                        break
                    if gen is None:
                        gen = _orient_gen(key[0], key[1], bnd)
                    _send_blocking(b, (rid, _orient_step(gen)))
            except BaseException:  # noqa: BLE001
                code = 1
            finally:
                os._exit(code)
        b.close()
        self.pid, self.sock = pid, a

    def ask(self) -> int:
        """Request the next candidate (the child starts on it; no wait) → its id."""
        rid = self.next_id
        self.next_id += 1
        if self.sock is not None:
            try:
                _send_blocking(self.sock, (rid, self.bound))
                self.pending.append(rid)
                return rid
            except OSError as e:
                self._lost(f"{type(e).__name__}: {e}")
        self.got[rid] = self._here()
        return rid

    def take(self, rid: int) -> tuple:
        """The answer of request rid (waits for the child)."""
        while rid not in self.got:
            if rid not in self.pending:
                raise ValueError(f"orientation search {self.key}: request {rid} is not outstanding")
            try:
                r_id, out = _recv_blocking(self.sock)
            except (EOFError, OSError) as e:
                self._lost(f"{type(e).__name__}: {e}")
                continue
            self.pending.remove(r_id)
            self.got[r_id] = out
            self.n_recv += 1
        return self.got.pop(rid)

    def _here(self) -> tuple:
        """The next answer computed in the worker: the same generator, replayed past the answers the child gave."""
        if self.gen is None:
            self.gen = _orient_gen(self.key[0], self.key[1], self.bound)
            for _ in range(self.n_recv):
                _orient_step(self.gen)
        return _orient_step(self.gen)

    def _lost(self, why: str) -> None:
        """The child died (or its socket failed): this search continues in the worker (class docstring)."""
        sys.stderr.write(f"g1_worker orientation search child {self.key} lost ({why}) - continued in the worker (replayed)\n")
        self.close()
        for r_ in self.pending:
            self.got[r_] = self._here()
        self.pending = []

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGKILL)                           # a child still computing an unwanted candidate
            except ProcessLookupError:
                pass
            try:
                os.waitpid(self.pid, 0)                                     # reaped here only
            except ChildProcessError:
                pass
            self.pid = None


def _orient_search(p: dict) -> _OrientSearch:
    key = (int(p["k"]), float(p["roll"]))
    srch = _ORIENT["searches"].get(key)
    if srch is None:                                                        # the bound of its first request (choose_orientation: one
        srch = _OrientSearch(key, p.get("max_deviation_deg"))              # bound per roll)
        _ORIENT["searches"][key] = srch
    return srch


def _orient_close() -> None:
    for srch in list((_ORIENT.get("searches") or {}).values()):
        srch.close()
    if _ORIENT.get("searches"):
        _ORIENT["searches"] = {}


def _orient_out(out: tuple):
    if out[0] == "ok":
        return out[1]
    raise RuntimeError(f"orientation search failed: {out[1]}\n{out[2] if len(out) > 2 else ''}")


def _job_orient_ask(p: dict) -> dict:
    """Start the next candidate of pass k at TCP roll 'roll' (its search's child computes it; no wait) → {'id'} for orient_take."""
    return {"id": _orient_search(p).ask()}


def _job_orient_take(p: dict) -> dict:
    """The candidate of request 'id' (orient_ask) of pass k at roll 'roll' - waits for it."""
    return _orient_out(_orient_search(p).take(int(p["id"])))


def _job_orient_next(p: dict) -> dict:
    """The next candidate of pass k at TCP roll 'roll' (closest feasible first; 'exhausted' at the end of the grid)."""
    srch = _orient_search(p)
    return _orient_out(srch.take(srch.ask()))


def _job_orient_close(p: dict) -> dict:
    """The orientation search's children ended (the candidates still being computed are not wanted)."""
    n = len(_ORIENT.get("searches") or {})
    _orient_close()
    return {"closed": n}


JOBS: dict = {"leg": _job_leg, "estimate": _job_estimate, "outputs": _job_outputs, "stream_open": _job_stream_open,
              "orient_open": _job_orient_open, "orient_next": _job_orient_next, "orient_ask": _job_orient_ask,
              "orient_take": _job_orient_take, "orient_close": _job_orient_close,
              "ping": lambda p: {"pid": os.getpid(), "cpu_priority": PRIORITY.get("applied"), "nice": os.nice(0),
                                 "stream": None if _STREAM is None else {"token": _STREAM.token, "frames": _STREAM.n,
                                                                         "rows_done": len(_STREAM.done), "children": len(_STREAM.kids),
                                                                         "child_pids": [k["pid"] for k in _STREAM.kids.values()]}}}


def run_job(job: str, payload):
    """같은 일감을 이 프로세스에서 (시험·오프라인)."""
    return JOBS[job](payload)


def _plainish(o):
    try:
        json.dumps(o, default=str)
        return o
    except Exception:  # noqa: BLE001
        return str(o)


# --------------------------------------------------------------------------------------
# 틀 (길이 머리 + pickle)
# --------------------------------------------------------------------------------------
def _send_blocking(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HDR.pack(len(data)))
    sock.sendall(data)


def _recv_exact_blocking(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], min(n - got, _CHUNK))
        if k == 0:
            raise EOFError("peer closed")
        got += k
    return bytes(buf)


def _recv_blocking(sock: socket.socket):
    (n,) = _HDR.unpack(_recv_exact_blocking(sock, _HDR.size))
    return pickle.loads(_recv_exact_blocking(sock, n))


_PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """PR_SET_PDEATHSIG(SIGKILL) (S3). 실패하면 OSError — 부르는 자식은 끝난다 (작업 프로세스: D8, fork_map 자식: 그 덩어리는 부모에서)."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_PDEATHSIG): {os.strerror(err)}")


def _lowest() -> str:
    """보통 시분할의 가장 낮은 우선순위 (legacy 스트림 자식과 같은 원시 post_scan._lowest_priority — 서버의 ROS 실행기·가드·녹화 (nice 0)
    아래)."""
    from .post_scan import _lowest_priority
    PRIORITY["applied"] = _lowest_priority()
    return PRIORITY["applied"]


NUMERIC_ENV = "HONGIK_WORKER_NUMERIC"      # the server's numeric packages {name: [version, file]} - the worker must load the same


def numeric_packages() -> dict:
    """{name: [version, file]} of the numeric packages loaded in this process (numpy, and scipy when imported)."""
    out = {}
    for name in ("numpy", "scipy"):
        m = sys.modules.get(name)
        if m is not None and getattr(m, "__file__", None):
            out[name] = [str(getattr(m, "__version__", "?")), os.path.abspath(m.__file__)]
    return out


def worker_pythonpath(path=None, numeric: Optional[dict] = None) -> str:
    """The worker's PYTHONPATH: the site directories the server's numeric packages came from first, then the server's sys.path.
    The worker is a fresh interpreter (python -c), so it resolves 'numpy' by path order: on a host with another NumPy earlier on
    the path (KIRO 2026-10-14: NumPy 2 in /usr/local and the user site, the apt NumPy 1.21 / SciPy 1.8 pair loaded by the server)
    it would otherwise load a different NumPy than the server - arrays pickled by one cannot be read by the other."""
    roots = []
    for _name, (_v, f) in (numeric_packages() if numeric is None else numeric).items():
        d = os.path.dirname(os.path.dirname(f))
        if d not in roots:
            roots.append(d)
    rest = [p for p in (sys.path if path is None else path) if p and p not in roots]
    return os.pathsep.join(roots + rest)


def _check_numeric() -> None:
    """In the worker: the numeric packages must be the server's (NUMERIC_ENV) - else exit with the reason (the server records the
    worker's death; no computation with a different NumPy)."""
    want = json.loads(os.environ.get(NUMERIC_ENV) or "{}")
    if "scipy" in want:
        import scipy  # noqa: F401,PLC0415 - compare the same set the server had
    have = numeric_packages()
    bad = [f"{n}: worker {have.get(n)} != server {w}" for n, w in want.items() if have.get(n) != w]
    if bad:
        sys.stderr.write("hongik G1 worker: numeric packages differ from the server's - " + "; ".join(bad) + "\n")
        sys.stderr.flush()
        os._exit(3)


def _child_main(fd: int, parent: int, stream_fd: int = -1) -> None:
    """작업 프로세스 본체: PDEATHSIG + 부모 재확인 (S3) → 가장 낮은 CPU 우선순위 → 일감 고리 (EOF = 서버가 닫았다 → 끝). 스트림이 열려
    있으면 일감을 기다리는 동안 스트림을 받아 덩어리 자식을 낸다 (_LegStream.pump); 'leg'·'ping' 밖의 일감은 스트림을 닫는다."""
    global _SSOCK, _MAIN_SOCK
    _set_pdeathsig()
    if os.getppid() != int(parent):
        os._exit(1)
    _check_numeric()
    _lowest()
    sock = socket.socket(fileno=int(fd))
    sock.setblocking(True)
    _MAIN_SOCK = sock
    if int(stream_fd) >= 0:
        _SSOCK = socket.socket(fileno=int(stream_fd))
        _SSOCK.setblocking(False)
    while True:
        if _STREAM is not None:
            try:
                _STREAM.pump(sock)
            except Exception:  # noqa: BLE001 - the stream is only a speed-up: drop it, the 'leg' job fits every row itself
                sys.stderr.write(f"g1_worker stream dropped:\n{traceback.format_exc(limit=6)}")
                _close_stream()
        try:
            job, payload = _recv_blocking(sock)
        except EOFError:
            os._exit(0)
        if job == "exit":
            os._exit(0)
        if job not in ("leg", "ping", "stream_open"):
            _close_stream()
        try:
            out = ("ok", JOBS[job](payload))
        except BaseException as e:  # noqa: BLE001 - every failure goes back to the server
            out = ("err", f"{type(e).__name__}: {e}", traceback.format_exc(limit=12))
        try:
            _send_blocking(sock, out)
        except (BrokenPipeError, ConnectionResetError, OSError):
            os._exit(0)


# --------------------------------------------------------------------------------------
# 서버 쪽
# --------------------------------------------------------------------------------------
class G1Worker:
    """실행 하나의 작업 프로세스 (§8.5). cancel_fd: 읽히면 모든 기다림이 Cancelled 로 끝나는 fd (Teach1G 의 취소 파이프 읽는 끝).
    CHILD_CODE: 자식의 시작 코드 (python -c; argv = 소켓 fd, 부모 pid)."""

    CHILD_CODE = "import sys; from scan_l0.g1_worker import _child_main; _child_main(*map(int, sys.argv[1:4]))"

    def __init__(self, cancel_fd: int, *, log: Optional[Callable[[str], None]] = None):
        self.cancel_fd = int(cancel_fd)
        self.log = log or (lambda s: None)
        self.proc: Optional[subprocess.Popen] = None
        self.sock: Optional[socket.socket] = None
        self.respawns = 0
        self.deaths: list = []
        self.stream_sock: Optional[socket.socket] = None     # 서버 쪽 스트림 소켓 (SEQPACKET, 논블로킹) — 작업 프로세스마다 새로

    @property
    def pid(self) -> Optional[int]:
        return None if self.proc is None else self.proc.pid

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.alive():
            return
        self._reap()
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        sa, sb = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)       # 스트림 (녹화기 탭 → 작업 프로세스)
        try:
            sa.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)        # OS 한계 안에서 크게 (작업 프로세스가 밀릴 때의 여유)
        except OSError:
            pass
        env = dict(os.environ)
        num = numeric_packages()
        env["PYTHONPATH"] = worker_pythonpath(numeric=num)          # the server's numpy / scipy first (worker_pythonpath)
        env[NUMERIC_ENV] = json.dumps(num)
        for k_ in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):       # fork_map 자식: BLAS 스레드 풀 없이 (fork 안전)
            env.setdefault(k_, "1")
        try:
            self.proc = subprocess.Popen([sys.executable, "-c", self.CHILD_CODE, str(b.fileno()), str(os.getpid()), str(sb.fileno())],
                                         pass_fds=(b.fileno(), sb.fileno()), env=env, stdin=subprocess.DEVNULL, close_fds=True)
        except BaseException:
            sa.close()
            raise
        finally:
            b.close()
            sb.close()
        a.setblocking(False)
        sa.setblocking(False)
        self.sock = a
        self.stream_sock = sa

    # ---- 기다림: select({소켓, 취소 fd}) ----
    def _wait(self, want_write: bool) -> None:
        r, w = ([self.cancel_fd], [self.sock]) if want_write else ([self.cancel_fd, self.sock], [])
        while True:
            rr, ww, _ = select.select(r, w, [])
            if self.cancel_fd in rr:
                raise Cancelled("cancel requested")
            if rr or ww:
                return

    def _send(self, job: str, payload) -> None:
        data = pickle.dumps((job, payload), protocol=pickle.HIGHEST_PROTOCOL)
        view = memoryview(_HDR.pack(len(data)) + data) if len(data) < _CHUNK else None
        parts = [memoryview(_HDR.pack(len(data))), memoryview(data)] if view is None else [view]
        for mv in parts:
            off = 0
            while off < len(mv):
                self._wait(want_write=True)
                try:
                    off += self.sock.send(mv[off:off + _CHUNK])
                except BlockingIOError:
                    continue
                except (BrokenPipeError, ConnectionResetError) as e:
                    raise EOFError(f"worker gone while sending: {e}") from e

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            self._wait(want_write=False)
            try:
                k = self.sock.recv_into(view[got:], min(n - got, _CHUNK))
            except BlockingIOError:
                continue
            except ConnectionResetError as e:
                raise EOFError(str(e)) from e
            if k == 0:
                raise EOFError("worker closed the result socket")
            got += k
        return bytes(buf)

    def _recv(self):
        (n,) = _HDR.unpack(self._recv_exact(_HDR.size))
        return pickle.loads(self._recv_exact(n))

    def call(self, job: str, payload):
        """일감 → 결과. 취소 → Cancelled (바로); 작업 프로세스 죽음 → 새로 띄워 같은 일감 한 번 더 (D8), 두 번째 → WorkerDied;
        일감 예외 → WorkerError."""
        for attempt in (1, 2):
            if not self.alive():
                self.start()
            try:
                self._send(job, payload)
                out = self._recv()
            except EOFError as e:
                code = self._reap()
                self.deaths.append({"job": job, "attempt": attempt, "exit": code, "why": str(e)})
                if attempt == 2:
                    raise WorkerDied(f"the compute worker died twice during '{job}' (exit codes "
                                     f"{[d['exit'] for d in self.deaths[-2:]]})") from e
                self.respawns += 1
                self.log(f"compute worker died during '{job}' (exit {code}) - respawned, '{job}' resubmitted once (D8)")
                continue
            if out[0] == "ok":
                return out[1]
            raise WorkerError(f"{job}: {out[1]}", out[2] if len(out) > 2 else "")
        raise WorkerDied(job)                                        # not reached

    def _reap(self) -> Optional[int]:
        """죽은 (또는 버릴) 작업 프로세스 정리: 소켓 닫기, 살아 있으면 SIGKILL (멈춘 프로세스도 끝난다), 거두기."""
        code = None
        if self.sock is not None:
            self.sock.close()                                            # socket.close 는 두 번 닫아도 오류가 없다
            self.sock = None
        if self.stream_sock is not None:
            self.stream_sock.close()
            self.stream_sock = None
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    self.proc.kill()
                except ProcessLookupError:                               # poll 과 kill 사이에 끝났다
                    pass
            code = self.proc.wait()
            self.proc = None
        return code

    def close(self) -> None:
        """끝: 살아 있으면 'exit' 를 짧게 시도하지 않고 바로 SIGKILL (계산 중·멈춤 상태라도 남기지 않는다 — zero leftovers)."""
        self._reap()
