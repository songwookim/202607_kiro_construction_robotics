"""1G core: profiles fitted WHILE a leg is recorded (2026-10-10) — the result must not depend on it.

Synthetic straight legs (scan_core.synth heightmap renderer, a plate end inside the leg) go through the REAL worker path: G1Worker
subprocess, 'stream_open', the recorder tap (StreamTap) and forked chunk children, then the 'leg' job. Checked:
* streamed == post hoc, byte for byte (LegObs arrays + meta without timing), for a transit (both mirrors, warm starts) and a pass
  (seam prediction on the planned leg, hints from a pose source) — also when every hint is wrong, when frames are missing / reordered;
* the per-row kernel does not depend on the chunking (fit_leg with 1 or 5 chunks);
* worker killed mid-leg: respawn + resubmit (D8) fits after the leg — same LegObs; its chunk children are gone;
* cancel while chunk children run: Cancelled at once, worker and children gone;
* the tap never blocks the recorder (worker stopped, socket full: frames are kept, nothing waits).
"""
from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scan_core import groove_model as gm
from scan_core.g1_types import LegObs, SeamPrior, Tolerances
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_profile
from scan_core.types import ProfileFrame, RobotSample, ScanJob
from scan_core.validity import valid_mask
from scan_l0 import config as C
from scan_l0 import g1_worker as GW

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scan_core", "tests"))
from g1_synth import synth_estimate  # noqa: E402

CALIB = os.path.join(os.path.dirname(os.path.dirname(HERE)), "eoat_description", "calib", "eoat_lab_01.yaml")
TOL = Tolerances(root_position_mm=1.0, end_position_mm=1.0, gap_mm=0.5, depth_mm=0.5, bevel_deg=2.0, opening_mm=0.5, length_mm=1.0)
SPEC = GrooveSpec(plate_width_mm=600.0)
Z_VALID = (233.1, 544.0)
FPS = 100.0
ARR = ("theta", "cov", "log_bf", "sigma_r_mm", "eps", "n_valid", "converged", "t_acq_s", "T_base_opt", "v_opt_mps")
META_SKIP = ("timing", "stream", "predictor")


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _T_opt(x_m, standoff_mm, phi_deg, y_m=0.0007):
    """laser line along base y, travel along base x, optical axis down, pitch phi about the line."""
    X, Z = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    Z = _rot(X, math.radians(phi_deg)) @ Z
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = X, np.cross(Z, X), Z
    T[:3, 3] = [x_m, y_m, standoff_mm * 1e-3]
    return T


def synth_leg(kind: str, n: int = 36, x_end: float = 0.0045):
    """A straight leg along +x over a plate end at x_end (beyond it: no return). Frames at FPS with their capture time t_k − dt and
    robot samples at FPS (other phase) of the same constant-velocity motion; poses from the robot samples (exact interpolation)."""
    cal = C.load_calib(CALIB)
    T_fo = np.asarray(cal.T_flange_opt, float)
    standoff, phi = (444.0, 0.0) if kind == "transit" else (342.0, -5.0)
    v = 0.004 * FPS / 10.0                                       # 4 mm per 10 frames
    Zp = _T_opt(0.0, standoff, phi)[:3, 2]
    hit = standoff * 1e-3 * float(Zp[0]) / abs(float(Zp[2]))      # laser hit ahead of the optical origin along x (pitch)
    x_a = x_end - hit - 0.7 * v * n / FPS                        # the laser line crosses the plate end at ~70 % of the leg
    t0, dt = 100.0, float(cal.dt_s)
    spec = SPEC
    scene = GrooveScene(spec=spec, y_range=(-0.3, 0.3), x_range=(-1.0, x_end))
    cfg = SynthConfig(sigma_z_mm=0.03, seed=0, z_grid_mm=(standoff - 30.0, standoff + 70.0, 0.05), bisect_iters=20)
    rng = np.random.default_rng(7)

    def T_bf_at(t):
        return _T_opt(x_a + v * (t - t0), standoff, phi) @ np.linalg.inv(T_fo)
    frames = []
    for k in range(n):
        t_rx = t0 + k / FPS + dt
        T_bo = T_bf_at(t_rx - dt) @ T_fo
        x, z = simulate_profile(T_bo, scene, np.eye(4), cfg, rng)
        frames.append(ProfileFrame(x_mm=np.asarray(x, np.float32), z_mm=np.asarray(z, np.float32), ts_ms=10 * k, result_counter=1000 + k,
                                   t_rx_s=t_rx))
    robot = [RobotSample(t_rx_s=t, T_base_flange=T_bf_at(t)) for t in t0 - 0.05 + np.arange(n + 12) / FPS + 0.0037]
    T_from, T_to = T_bf_at(t0 - 0.05), T_bf_at(t0 + (n + 6) / FPS)
    job = ScanJob(start=[x_a, 0.0, 0.0], end=[x_a + 0.1, 0.0, 0.0], surface_normal=[0, 0, 1], table_z_m=-0.05,
                  specimen_top_z_m=0.0, job_id="synth", standoff_m=standoff * 1e-3, speed_mps=v, profile_fps=FPS, pitch_deg=phi)
    prior = SeamPrior(S0=np.array([-0.30, 0.0, 0.0]), E0=np.array([x_end, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]), spec=spec, tol=TOL,
                      k=3.0, dt_s=dt, dt_sigma_s=1e-3, rep_sigma_m=float("nan"), he_sigma_m=2e-3, sigma_eps=float("nan"),
                      station_step_m=0.005, station_window_m=0.010)
    est = None
    if kind == "pass":
        est = synth_estimate(prior, a=(0.0, 0.0), b=(-0.010, 0.0), e_S=0.0, length_m=0.30 + x_end, bevel_sign_w=-1)
    R_opt = (T_bf_at(t0) @ T_fo)[:3, :3]
    if est is None:
        pp = {"c": 1.0, "sigma_c": 1.0 / math.cos(0.2) - 1.0, "phi_rad": 0.0}
        bxs = None
    else:
        pp = {"c": 1.0 / abs(float(R_opt[:, 1] @ est.u)), "sigma_c": 1.0 / math.cos(0.01) - 1.0, "phi_rad": math.radians(phi)}
        bxs = int(est.bevel_sign_w) * (1 if float(R_opt[:, 0] @ prior.frame0()["w0"]) >= 0 else -1)
    plan = {"T_from": T_from, "T_to": T_to, "T_flange_opt": T_fo, "step_m": prior.station_step_m}
    fit = {"spec": spec, "tol": TOL, "profile_prior": pp, "bevel_x_sign": bxs, "cls": kind, "z_valid_mm": Z_VALID, "est": est,
           "leg_plan": plan}
    return SimpleNamespace(kind=kind, frames=frames, robot=robot, job=job, calib=cal, fit=fit, est=est, dt=dt, T_bf_at=T_bf_at)


def payload(L, out_dir, n_children=3, **kw):
    return dict(name=f"{L.kind}_synth", kind=L.kind, out_leg_dir=str(out_dir), raw_arrays={"n": np.array(len(L.frames))}, raw_json={},
                frames=L.frames, robot=L.robot, T_base_flange=None, calib=L.calib, job=L.job, direction=1, predict=None,
                k=3.0, cloud_step_m=None, n_children=n_children, meta={"kind": L.kind}, **L.fit, **kw)


def records(L, order=None):
    idx = range(len(L.frames)) if order is None else order
    return [SimpleNamespace(x=L.frames[i].x_mm, z=L.frames[i].z_mm, result_counter=L.frames[i].result_counter,
                            t_rx_ns=int(round(L.frames[i].t_rx_s * 1e9)), ts_ms=L.frames[i].ts_ms) for i in idx]


def same_leg(a: LegObs, b: LegObs, skip=()) -> list:
    bad = [f for f in ARR if not np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)]
    ma = {k: v for k, v in a.meta.items() if k not in META_SKIP + tuple(skip)}
    mb = {k: v for k, v in b.meta.items() if k not in META_SKIP + tuple(skip)}
    if repr(ma) != repr(mb):
        bad.append("meta")
    if a.bevel_x_sign != b.bevel_x_sign:
        bad.append("bevel_x_sign")
    return bad


class Worker:
    def __init__(self):
        self.r, self.w = os.pipe()
        self.W = GW.G1Worker(self.r)
        self.W.start()

    def cancel(self):
        os.write(self.w, b"c")

    def close(self):
        self.W.close()
        os.close(self.r)
        os.close(self.w)


@pytest.fixture
def worker():
    w = Worker()
    yield w
    w.close()


def stream_leg(W, L, out_dir, *, token=11, pose_at=None, order=None, wait_rows=None, n_children=3, feed=None):
    key = GW.stream_key(L.fit)
    so = W.call("stream_open", {**L.fit, "stream_token": token, "stream_key": key, "name": L.kind, "direction": 1,
                                "n_children": n_children, "chunk_rows": 4})
    assert so["stream"] == token, so
    tap = GW.StreamTap(W.stream_sock, token, pose_at=pose_at, dt_s=L.dt)
    for r in records(L, order):
        tap(r)
        if feed is not None:
            feed(r)
    if wait_rows is not None:                      # let chunk children finish some rows before the leg job (test only)
        t_end = time.monotonic() + 120.0
        while time.monotonic() < t_end:
            st = W.call("ping", None)["stream"]
            if st and st["rows_done"] >= wait_rows:
                break
            time.sleep(0.05)
    stats = tap.close()
    out = W.call("leg", payload(L, out_dir, n_children=n_children, stream_token=token, stream_key=key))
    return LegObs.load(out["leg_path"]), stats


def post_hoc(W, L, out_dir):
    out = W.call("leg", payload(L, out_dir))
    return LegObs.load(out["leg_path"])


def exact_pose(L):
    return lambda t: L.T_bf_at(t)


# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["transit", "pass"])
def test_streamed_leg_equals_post_hoc_byte_for_byte(worker, tmp_path, kind):
    L = synth_leg(kind)
    A, st = stream_leg(worker.W, L, tmp_path / "s", pose_at=exact_pose(L) if kind == "pass" else None, wait_rows=8)
    B = post_hoc(worker.W, L, tmp_path / "p")
    assert same_leg(A, B) == [], same_leg(A, B)
    s = A.meta["stream"]
    assert s["rows_from_stream"] >= 8 and s["rows"] == len(L.frames) and st["dead"] is None, (s, st)
    assert np.any(A.log_bf > 0) and np.any(A.n_valid == 0)          # groove profiles and empty profiles beyond the plate end
    if kind == "pass":
        # a hint every HINT_EVERY_FRAMES frames + the last one; the worker interpolates the leg position in between (exact here)
        assert s["hints"] == -(-len(L.frames) // GW.HINT_EVERY_FRAMES) + 1 and s["hint_mismatch"] == 0, s
        assert A.meta["predictor"].startswith("estimate: second start on the planned leg") and A.meta["log_bf_kind"] in ("seam", "mixed")


def test_wrong_hints_missing_and_reordered_frames_change_nothing(worker, tmp_path):
    """hints 3 mm off along the leg (grid index differs for most rows), one frame never streamed, two swapped: those rows are fitted
    after the leg — same LegObs."""
    L = synth_leg("pass")
    shift = np.eye(4)
    shift[0, 3] = 0.003
    order = list(range(len(L.frames)))
    order.remove(9)
    order[3], order[4] = order[4], order[3]
    A, _st = stream_leg(worker.W, L, tmp_path / "s", pose_at=lambda t: shift @ L.T_bf_at(t), order=order, wait_rows=8)
    B = post_hoc(worker.W, L, tmp_path / "p")
    assert same_leg(A, B) == []
    s = A.meta["stream"]
    assert s["hint_mismatch"] > 0 and s["rows_after_leg"] > s["hint_mismatch"] - 1, s


def test_row_kernel_does_not_depend_on_the_chunking():
    L = synth_leg("transit", n=18)
    from scan_core.reconstruct import reconstruct_scan
    res = reconstruct_scan(L.frames, L.robot, L.calib, L.job, fps_nominal=FPS, z_valid_mm=Z_VALID)
    prior = gm.profile_prior(SPEC, TOL, **L.fit["profile_prior"])
    a = gm.fit_leg(res, SPEC, TOL, cls="transit", direction=1, bevel_x_sign=None, prior=prior, n_children=1)
    b = gm.fit_leg(res, SPEC, TOL, cls="transit", direction=1, bevel_x_sign=None, prior=prior, n_children=5, map_fn=gm._local_map)
    assert same_leg(a, b, skip=("n_children",)) == []                # meta n_children records the setting


def test_seam_prior_applied_after_the_row_fits_equals_applying_it_inside():
    """defer_seam: the start choice and the seam-position λ computed after the row fits (finish_candidates + rows_select, from the
    reconstructed poses) give exactly what applying the same prediction inside the fit gives (the old single-pass path)."""
    L = synth_leg("pass", n=14)
    from scan_core import seam_estimate as se
    from scan_core.reconstruct import reconstruct_scan
    res = reconstruct_scan(L.frames, L.robot, L.calib, L.job, fps_nominal=FPS, z_valid_mm=Z_VALID)
    prior = gm.profile_prior(SPEC, TOL, **L.fit["profile_prior"])
    pm = se.profile_predictor(L.est, np.asarray(res.T_base_opt), L.fit["bevel_x_sign"], SPEC)
    kw = dict(cls="pass", direction=1, bevel_x_sign=L.fit["bevel_x_sign"], prior=prior)
    inline = gm.fit_leg(res, SPEC, TOL, predict=pm, **kw)
    deferred = gm.fit_leg(res, SPEC, TOL, predict=lambda i: None if pm(i) is None else pm(i)[0], seam_pred=pm, n_children=3,
                          map_fn=gm._local_map, **kw)
    assert same_leg(inline, deferred, skip=("n_children",)) == []
    assert inline.meta["log_bf_kind"] in ("seam", "mixed") and np.any(inline.log_bf > 0)


def _children_of(pid: int) -> list:
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    if int(f.read().rsplit(")", 1)[1].split()[1]) == pid:
                        out.append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    return out


def _gone(pids, within_s=2.0) -> bool:
    t_end = time.monotonic() + within_s
    while time.monotonic() < t_end:
        alive = []
        for p in pids:
            try:
                os.kill(p, 0)
                with open(f"/proc/{p}/stat") as f:
                    if f.read().rsplit(")", 1)[1].split()[0] != "Z":
                        alive.append(p)
            except (ProcessLookupError, FileNotFoundError):
                pass
        if not alive:
            return True
        time.sleep(0.02)
    return False


def test_worker_killed_mid_leg_respawn_fits_after_the_leg_same_result(worker, tmp_path):
    L = synth_leg("transit")
    n_half = len(L.frames) // 2
    seen = {}

    def feed(r):
        if r.result_counter == 1000 + n_half and not seen:
            time.sleep(0.3)                                          # chunk children running
            seen["kids"] = _children_of(worker.W.pid)
            seen["pid"] = worker.W.pid
            os.kill(worker.W.pid, signal.SIGKILL)
    A, st = stream_leg(worker.W, L, tmp_path / "s", feed=feed)
    assert st["frames"] == len(L.frames), st                         # every recorder call returned (no exception into the recorder)
    assert worker.W.respawns == 1 and _gone([seen["pid"]] + seen["kids"])
    assert "stream" not in A.meta                                    # the new worker had no stream: every row after the leg
    B = post_hoc(worker.W, L, tmp_path / "p")
    assert same_leg(A, B) == []


def test_cancel_while_chunk_children_run_ends_everything_at_once(worker, tmp_path):
    L = synth_leg("transit")
    W = worker.W
    key = GW.stream_key(L.fit)
    W.call("stream_open", {**L.fit, "stream_token": 5, "stream_key": key, "name": "t", "direction": 1, "n_children": 3, "chunk_rows": 4})
    tap = GW.StreamTap(W.stream_sock, 5)
    for r in records(L):
        tap(r)
    tap.close()
    t_end = time.monotonic() + 30.0
    st = None
    while time.monotonic() < t_end:
        st = W.call("ping", None)["stream"]
        if st["children"] > 0:
            break
        time.sleep(0.02)
    pids = [W.pid] + list(st["child_pids"])
    assert st["children"] > 0, st
    threading.Timer(0.2, worker.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(GW.Cancelled):
        W.call("leg", payload(L, tmp_path / "c", stream_token=5, stream_key=key))
    assert time.monotonic() - t0 < 0.5
    W.close()                                                        # the run's finally (Teach1G): SIGKILL, children by PDEATHSIG
    assert _gone(pids)


def test_the_tap_never_blocks_the_recorder(worker, tmp_path):
    """worker stopped (SIGSTOP): the socket fills, every tap call returns at once, unsent frames are kept in order and dropped at close."""
    L = synth_leg("transit", n=60)
    W = worker.W
    W.call("stream_open", {**L.fit, "stream_token": 9, "stream_key": GW.stream_key(L.fit), "name": "t", "direction": 1, "n_children": 2})
    os.kill(W.pid, signal.SIGSTOP)
    try:
        tap = GW.StreamTap(W.stream_sock, 9)
        worst = 0.0
        recs = records(L) * 8                                          # ~5 MB: more than any socket buffer
        for r in recs:
            t0 = time.perf_counter()
            tap(r)
            worst = max(worst, time.perf_counter() - t0)
        st = tap.close()
    finally:
        os.kill(W.pid, signal.SIGCONT)
    assert st["send_blocked"] > 0 and st["left_unsent"] > 0 and st["dead"] is None, st
    assert worst < 0.05, worst
