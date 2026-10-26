"""패스 중 판 성장 스트림 (scan_l0.post_scan, 2026-10-08): 녹화되는 대로 판 성장을 미리 계산하고 이동 뒤에는 나머지만 — 결과는 스트림 없이
계산한 것과 바이트까지 같아야 한다 (특징·맞춤·끝·중앙값·잔차, 그리고 그 위의 판정·용접 경로). 스트림이 어떻게 실패해도 (다른 입력·프레임
값이 다름·빠진 프레임·망가진 프레임·계산 오류·중단) 같은 결과로 돌아가고 느려질 뿐이다. 프로세스가 죽거나 멈춰도 (2026-10-09: 작업
프로세스·스트림 계산 프로세스 SIGKILL, 계산 프로세스 SIGSTOP; 특징 풀 자식 SIGKILL → 작업 프로세스 안에서 바로 같은 계산) 풀의 주인은
시간 초과가 아니라 프로세스 상태로 바로 알고 같은 결과 — 남는 프로세스·스트림 디렉터리·실행기 스레드 없음. 서버는 '작업 프로세스가 죽었다'
를 예외 종류가 아니라 작업 프로세스의 상태로 가른다. 서버 쪽 비용은 파이프 용량으로 묶인다 (작업자 수·프레임 수와 따로).

실기 재생 (g1_b, 1G 4 패스, 패스당 ~1350 프로파일) 은 자료가 있을 때만 (~/scan_data/hongik_auto_teach/g1_b)."""
import copy
import json
import math
import os
import pickle
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scan_core.path_gen import sensor_frames
from scan_core.reconstruct import reconstruct_scan
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_scan
from scan_core.types import Calib, ScanJob
from scan_l0 import autoteach as A
from scan_l0 import config as C
from scan_l0 import post_scan as PS
from scan_l0.autoteach import pass_feature_inputs
from sim_harness import exact_diff

STACK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAB_RUN = os.environ.get("HONGIK_REPLAY_G1B", os.path.expanduser("~/scan_data/hongik_auto_teach/g1_b"))
Z_VALID = (180.0, 520.0)
_HDR_BYTES = PS._HDR.size


@pytest.fixture(scope="module")
def synthetic_pass():
    """합성 1G 패스 (피치 −5° → 프로파일마다 depth_scale ≠ 1, 판 끝 너머까지): 작업 프로세스 입력 dict (스트림 없음)."""
    calib = Calib.lab_initial(eoat_yaw_rad=-np.pi / 2, opt_yaw_branch=-1)
    spec = GrooveSpec(bevel_side="left")
    job = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, -0.08, 0.05], surface_normal=[0, 0, 1], table_z_m=-0.05,
                  specimen_top_z_m=0.05, lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.04, accel_mps2=0.5,
                  profile_fps=20.0, pitch_deg=-5.0, job_id="stream_syn")
    scene = GrooveScene.from_spec(spec, yc=0.002)
    scene.x_range = (0.01, 0.11)                                        # 판 끝 둘 다 패스 안
    syn = simulate_scan(job, calib, scene, SynthConfig(n_points=704, seed=11, sigma_z_mm=0.02, dropout_prob=0.01,
                                                       robot_poll_rate_hz=0.0))
    fr = sensor_frames(job, calib)
    fk = dict(spec=spec, u_prior=job.unit_dir, n_prior=job.normal_unit, stage="fine", u_travel=fr.u_travel,
              target_standoff_m=float(job.standoff_m), gravity_axis_base=job.gravity_axis_base, end_params=None, process=None,
              feature_workers=1, qc_targets=None)
    p = dict(arrays={}, js={}, frames=syn.frames, robot=syn.robot, job=job, calib=calib, z_valid_mm=Z_VALID, fit_kwargs=fk,
             defer_save=True)
    return p


def _writer():
    """시험이 서버 역할: 지금 스트림의 FIFO 를 쓰기로 연다 (블로킹 — 이 프로세스의 받기 스레드가 읽는다)."""
    return os.open(PS._STREAM["fifo"], os.O_WRONLY)


def _send(fd, frames, mutate=None):
    """프레임을 서버와 같은 바이트 (PS._frame_iov) 로 쓴다."""
    for f in frames:
        iov = PS._frame_iov(SimpleNamespace(result_counter=f.result_counter, x=f.x_mm if mutate is None else mutate(f), z=f.z_mm))
        while iov:
            iov = PS._advance(iov, os.writev(fd, iov))


def _feed(frames, mutate=None):
    fd = _writer()
    try:
        _send(fd, frames, mutate)
    finally:
        os.close(fd)


def _wait_stream(cond):
    """시험만: 스트림 상태가 cond 를 만족할 때까지 (계산은 가장 낮은 우선순위 프로세스에서 비동기)."""
    st = PS._STREAM
    deadline = time.monotonic() + PS.RESULT_TIMEOUT_S
    while not cond(st) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cond(st), {k: st[k] for k in ("received", "frames", "error")}


def _children(pid) -> list:
    """pid 의 자식 프로세스 (리눅스 /proc/<pid>/task/*/children)."""
    out = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for t in tids:
        try:
            with open(f"/proc/{pid}/task/{t}/children", encoding="ascii") as fh:
                out += [int(s) for s in fh.read().split()]
        except OSError:
            pass
    return out


def _running(pid) -> bool:
    """pid 가 아직 도는가 (끝났거나 거두기만 남은 좀비면 False)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _fd_targets(pid) -> set:
    """pid 가 연 fd 중 프로세스 사이 연결 (pipe:[inode]·socket:[inode] — inode 가 곧 그 연결) 의 대상 — 표준 입출력 (0–2) 은 빼고."""
    out = set()
    try:
        names = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return out
    for s in names:
        if int(s) > 2:
            try:
                t = os.readlink(f"/proc/{pid}/fd/{s}")
            except OSError:
                continue
            if t.startswith(("pipe:", "socket:")):
                out.add(t)
    return out


def _until(cond, limit_s: float) -> bool:
    """시험만: cond 가 참이 될 때까지 (한계 안에서) — 결과를 돌려준다."""
    t_end = time.monotonic() + limit_s
    while not cond() and time.monotonic() < t_end:
        time.sleep(0.005)
    return bool(cond())


def _rec(f):
    return SimpleNamespace(result_counter=f.result_counter, x=f.x_mm, z=f.z_mm)


def _stream_dirs(d) -> list:
    return sorted(n for n in os.listdir(d) if n.startswith("fsv1-plate-stream-"))


def _fresh_worker(n: int) -> tuple:
    """새 작업 프로세스 (spawn, 데움) → (풀, pid, 그 작업 프로세스가 서버에서 받은 fd 의 대상 — 데운 뒤 스트림 전이라 열린 것이 전부 그것)."""
    PS.shutdown_post_scan_pool()
    PS.warm_post_scan_worker(n).result(PS.RESULT_TIMEOUT_S)
    pool = PS.post_scan_pool()
    wpid = pool.submit(os.getpid).result(PS.RESULT_TIMEOUT_S)
    return pool, wpid, _fd_targets(wpid)


def _worker_stream_state():
    """(작업 프로세스 안에서 — 실행기로 부른다) 지금 스트림의 (받은 프레임, 계산한 프레임, 오류) — 스트림이 없으면 None."""
    st = PS._STREAM
    return None if st is None else (int(st["received"]), int(st["frames"]), st["error"])


def _baseline(p, tmp_path):
    out = PS.compute_pass(dict(p, out_dir=str(tmp_path / "base")))
    PS._PENDING.clear()
    assert out["fit"] is not None and out["recon_ok"]
    return out


def test_stream_gives_identical_pass_outputs_and_falls_back_row_by_row(synthetic_pass, tmp_path):
    p = synthetic_pass
    base = _baseline(p, tmp_path)
    frames = p["frames"]
    bad = frames[5].result_counter
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    PS.stream_open("t1", spec, kw, Z_VALID, 2)                          # 계산 프로세스 2 (가장 낮은 우선순위)
    # 마지막 3 프레임은 스트림에 없다 (이동 끝 뒤), 프레임 하나는 스트림에서 값이 다르다 (그 행은 처음부터)
    _feed(frames[:-3], mutate=lambda f: f.x_mm + np.float32(0.5) if f.result_counter == bad else f.x_mm)
    _wait_stream(lambda st: st["frames"] == len(frames) - 3)
    assert PS._STREAM["received"] == len(frames) - 3 and PS._STREAM["error"] is None
    # 쓸 수 있는 행 = 스트림에 있고 값이 같고 판 성장까지 간 프레임 (판 끝 너머의 빈 프레임은 씨앗 없음 — 나중에 처음부터, 싸다)
    usable = sum(1 for c, e in PS._STREAM["rows"].items() if e[3] is not None and c != bad)
    out = PS.compute_pass(dict(p, out_dir=str(tmp_path / "s1"), stream_token="t1", in_worker=True))
    PS._PENDING.clear()
    assert PS._STREAM is None                                           # 패스 뒤 처리가 스트림을 닫았다
    t = out["timing"]
    n = len(frames)
    assert t["stream_rows"] == n and t["stream_seeded"] == usable >= 0.5 * n, (t, usable)
    assert not exact_diff(base["fit"], out["fit"]), exact_diff(base["fit"], out["fit"])


def test_stream_with_other_inputs_or_errors_is_not_used(synthetic_pass, tmp_path):
    p = synthetic_pass
    base = _baseline(p, tmp_path)
    frames = p["frames"]
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    # 다른 작업 허용치로 연 스트림 → 쓰지 않는다
    PS.stream_open("t2", spec, {"qc_targets": {"groove_depth_mm": {"tol": 0.9}}}, Z_VALID, 1)
    _feed(frames)
    out = PS.compute_pass(dict(p, out_dir=str(tmp_path / "s2"), stream_token="t2"))
    assert "differ" in out["timing"]["stream_note"] and "stream_seeded" not in out["timing"]
    assert not exact_diff(base["fit"], out["fit"])
    # 다른 패스의 스트림 (토큰) → 쓰지 않는다 (남은 스트림은 닫는다)
    PS.stream_open("other", spec, kw, Z_VALID, 1)
    _feed(frames)
    out = PS.compute_pass(dict(p, out_dir=str(tmp_path / "s3"), stream_token="t3"))
    assert out["timing"]["stream_note"] == "no stream for this pass" and PS._STREAM is None
    assert not exact_diff(base["fit"], out["fit"])
    # 망가진 프레임 (머리의 x/z 점 수가 다르다) → 스트림이 멈추고 그 앞 프레임만 쓴다 (같은 결과)
    PS.stream_open("t4", spec, kw, Z_VALID, 1)
    m = 2 * len(frames) // 3
    fd = _writer()
    _send(fd, frames[:m])
    _wait_stream(lambda st: st["frames"] == m)
    os.write(fd, PS._HDR.pack(1, 2, 3))
    os.close(fd)
    _wait_stream(lambda st: st["error"] is not None)
    assert "lengths differ" in PS._STREAM["error"] and PS._STREAM["frames"] == m
    out = PS.compute_pass(dict(p, out_dir=str(tmp_path / "s4"), stream_token="t4"))
    PS._PENDING.clear()
    assert out["timing"]["stream_error"] and 0 < out["timing"]["stream_seeded"] <= m
    assert not exact_diff(base["fit"], out["fit"])
    # 스트림 결과를 맞추다 예외 (망가진 스트림 상태) → 그 패스는 스트림 없이 (같은 결과, 오류는 시간 기록에)
    PS.stream_open("t5", spec, kw, Z_VALID, 1)
    _feed(frames[:10])
    _wait_stream(lambda st: st["frames"] == 10)
    PS._STREAM["rows"] = None
    out = PS.compute_pass(dict(p, out_dir=str(tmp_path / "s5"), stream_token="t5"))
    PS._PENDING.clear()
    assert "AttributeError" in out["timing"]["stream_error"] and PS._STREAM is None and not exact_diff(base["fit"], out["fit"])


def test_frames_larger_than_the_socket_buffer_reach_the_compute_process(synthetic_pass, tmp_path, monkeypatch):
    """계산 프로세스로의 보내기는 막히지 않고 이어진다: 소켓 보내기 버퍼를 OS 최소로 줄여 프레임마다 여러 번에 나눠 가야 해도 모든 프레임이
    계산되고 결과는 같다."""
    import socket
    p = synthetic_pass
    base = _baseline(p, tmp_path)
    frames = p["frames"]
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    real = socket.socketpair
    small: list = []

    def tiny_pair(*a, **k):
        x, y = real(*a, **k)
        for s_ in (x, y):
            s_.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1)       # OS 가 허용하는 최소로
        small.append(x.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
        return x, y
    monkeypatch.setattr(PS.socket, "socketpair", tiny_pair)
    PS.stream_open("tiny", spec, kw, Z_VALID, 2)
    _feed(frames)
    _wait_stream(lambda st: st["frames"] == len(frames) or st["error"] is not None)
    frame_bytes = _HDR_BYTES + 8 * frames[0].x_mm.size
    assert PS._STREAM["error"] is None and small and max(small) < frame_bytes, (PS._STREAM["error"], small, frame_bytes)
    out = PS.compute_pass(dict(p, out_dir=str(tmp_path / "tiny"), stream_token="tiny", in_worker=True))
    PS._PENDING.clear()
    assert out["timing"]["stream_seeded"] > 0 and not exact_diff(base["fit"], out["fit"])


def test_stream_computation_runs_at_the_lowest_priority_for_any_worker_count(synthetic_pass):
    """판 성장 계산은 작업자 수와 무관하게 (1 도 — 작은 제어 PC 의 'auto') 별도 프로세스에서 보통 시분할의 가장 낮은 우선순위로 돈다 (nice 를
    OS 가 허용하는 끝까지 — 리눅스 19; 정책은 그대로 SCHED_OTHER) — 계산 프로세스가 알린 것과 OS 가 보는 것이 같다; 이 프로세스 (작업
    프로세스 역할) 의 우선순위는 그대로다."""
    spec, kw = pass_feature_inputs(synthetic_pass["fit_kwargs"])
    frames = synthetic_pass["frames"]
    mine = (os.sched_getscheduler(0), os.nice(0))
    for n in (1, 2):
        info = PS.stream_open(f"prio{n}", spec, kw, Z_VALID, n)
        _feed(frames[:2 * n])
        _wait_stream(lambda st: st["frames"] == 2 * n and all(q["prio"] for q in st["procs"]))
        st = PS._STREAM
        pids = [q["pid"] for q in st["procs"]]
        told = [q["prio"] for q in st["procs"]]
        seen = [(os.sched_getscheduler(pid), os.getpriority(os.PRIO_PROCESS, pid)) for pid in pids]
        PS.stream_close()
        assert info["procs"] == pids and len(pids) == n and os.getpid() not in pids, (info, pids)
        assert all(t_ == f"nice {nice}" and pol == mine[0] and nice >= mine[1] for t_, (pol, nice) in zip(told, seen)), (told, seen)
        if sys.platform.startswith("linux"):
            assert all(nice == 19 for _, nice in seen), seen                # 리눅스 nice 의 끝
    assert (os.sched_getscheduler(0), os.nice(0)) == mine


def test_post_scan_never_waits_for_the_lowest_priority_frames_in_flight(synthetic_pass, tmp_path):
    """이동 뒤 처리 (보통 우선순위) 는 가장 낮은 우선순위 계산 프로세스가 계산 중인 프레임을 기다리지 않는다 (우선순위 역전 없음): 그 계산은
    버리고 (그 프로세스는 끝낸다) 그 행은 처음부터 — 같은 결과. 극단: 계산 프로세스가 CPU 를 전혀 못 받는다 (SIGSTOP)."""
    p = synthetic_pass
    t0 = time.monotonic()
    base = _baseline(p, tmp_path)
    t_base = time.monotonic() - t0
    frames = p["frames"]
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    PS.stream_open("stall", spec, kw, Z_VALID, 2)
    m = len(frames) // 2 - 10                                          # 판이 있는 가운데 20 프레임 먼저
    _feed(frames[m:m + 20])
    _wait_stream(lambda st: st["frames"] == 20)
    pids = [q["pid"] for q in PS._STREAM["procs"]]
    for pid in pids:
        os.kill(pid, signal.SIGSTOP)
    _feed(frames[:m] + frames[m + 20:])
    _wait_stream(lambda st: st["received"] == len(frames) and all(q["busy"] is not None for q in st["procs"]))
    box: dict = {}
    th = threading.Thread(target=lambda: box.update(out=PS.compute_pass(dict(p, out_dir=str(tmp_path / "stall"), stream_token="stall",
                                                                             in_worker=True))))
    t0 = time.monotonic()
    th.start()
    th.join(10.0 * t_base + 10.0)                                       # 예전: 멈춘 프로세스를 끝없이 기다렸다
    assert not th.is_alive(), "compute_pass waited for the stopped lowest-priority computation"
    PS._PENDING.clear()
    t = box["out"]["timing"]
    assert t["stream_abandoned"] == 2 and t["stream_frames"] == 20 and t["stream_unprocessed"] == len(frames) - 22, t
    assert 0 < t["stream_seeded"] <= 20 and not exact_diff(base["fit"], box["out"]["fit"])
    assert _until(lambda: not any(_running(pid) for pid in pids), 30.0), [pid for pid in pids if _running(pid)]

    def reaped():
        PS.stream_close()                                               # 끝난 계산 프로세스를 거둔다 (기다리지 않고)
        return not any(os.path.exists(f"/proc/{pid}") for pid in pids)
    assert _until(reaped, 30.0)


def test_seed_feeder_streams_the_recording_to_the_worker(synthetic_pass, tmp_path, monkeypatch):
    """서버 쪽 SeedFeeder → 실제 패스 뒤 처리 작업 프로세스 (spawn): 녹화 스레드가 목록에 붙이는 동안 읽기만 해서 FIFO 로 넘기고, 패스 뒤
    처리는 그 결과를 써서 같은 출력. 서버 쪽 비용의 구조: 작업 프로세스 실행기 왕복은 패스마다 한 번 (stream_open) — 프레임 수와 무관;
    보고하는 서버 CPU 는 잰 것의 합 (피더 스레드 + 시작·멈춤 호출 + 실행기 스레드) 이고, 실행기 스레드를 못 재면 합은 NaN (잰 피더만 남는다
    — 재지 못한 것을 0 으로 치지 않는다). 중단 (cancel) 하면 작업 프로세스의 스트림도 닫힌다. 프레임률을 모르면 시작하지 않는다. 서버가
    만든 스트림 디렉터리는 어느 경로로 끝나도 남지 않는다."""
    p = dict(synthetic_pass, fit_kwargs=dict(synthetic_pass["fit_kwargs"], feature_workers=2))
    base = _baseline(p, tmp_path)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))                  # 이 프로세스 (서버 역할) 가 만드는 스트림 디렉터리
    frames = p["frames"]
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    PS.warm_post_scan_worker(2).result(PS.RESULT_TIMEOUT_S)
    pool = PS.post_scan_pool()
    submitted: list = []
    real_submit = pool.submit
    monkeypatch.setattr(pool, "submit", lambda fn, *a, **k: (submitted.append(fn.__name__), real_submit(fn, *a, **k))[1])
    records: list = []
    feeder = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps)
    assert feeder.start(records)

    def recorder():                                                     # 녹화 스레드 흉내 (프레임 간격 = 1/fps 의 1/10 로 빠르게)
        for f in frames:
            records.append(SimpleNamespace(result_counter=f.result_counter, x=f.x_mm, z=f.z_mm))
            time.sleep(0.1 / float(p["job"].profile_fps))
    th = threading.Thread(target=recorder)
    th.start()
    th.join()
    deadline = time.monotonic() + 30.0
    while feeder.frames < len(frames) and time.monotonic() < deadline:   # 시험만: 마지막 프레임까지 쓰게 기다린다
        time.sleep(0.05)
    st = feeder.stop()
    assert st["stream_sent_frames"] == len(frames) and 1 <= st["stream_sent_writes"] and "stream_feeder_error" not in st, st
    assert submitted == ["stream_open"] and st["stream_round_trips"] == 1, (submitted, st)
    assert st["stream_feeder_cpu_s"] > 0.0 and math.isfinite(st["stream_executor_cpu_s"]) and st["stream_calls_cpu_s"] > 0.0, st
    assert st["stream_server_cpu_s"] == st["stream_feeder_cpu_s"] + st["stream_calls_cpu_s"] + st["stream_executor_cpu_s"], st
    assert _stream_dirs(tmp) == []                                      # 양쪽이 연 뒤 바로 지웠다
    # 시험만: 가장 낮은 우선순위 계산이 받은 프레임을 다 계산할 때까지 — 패스 뒤 처리는 설계상 그것을 기다리지 않으므로 (계산 중·못 준 프레임은
    # 처음부터) 부하 아래에서는 스트림 결과를 쓴 행이 0 일 수 있다 (2026-10-09 검증: 부하 ~30 에서 0–8 행). 여기서는 쓰이는지를 본다
    assert _until(lambda: (lambda s: s is not None and (s[2] is not None or s[1] == s[0] == len(frames)))(
        pool.submit(_worker_stream_state).result(PS.RESULT_TIMEOUT_S)), PS.RESULT_TIMEOUT_S)
    out = PS.run_post_scan(dict(p, out_dir=str(tmp_path / "w1"), stream_token=feeder.token))
    t = out["timing"]
    assert out["where"] == "worker" and t["stream_received"] == len(frames) and t["stream_seeded"] > 0 and "stream_error" not in t, t
    assert not exact_diff(base["fit"], out["fit"])
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    # 중단된 패스: cancel → 작업 프로세스 스트림 닫힘 → 그 토큰의 패스 뒤 처리는 스트림 없이 (같은 결과)
    f2 = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps)
    rec2 = [SimpleNamespace(result_counter=f.result_counter, x=f.x_mm, z=f.z_mm) for f in frames[:30]]
    assert f2.start(rec2)
    f2.cancel()
    out = PS.run_post_scan(dict(p, out_dir=str(tmp_path / "w2"), stream_token=f2.token))
    assert out["timing"]["stream_note"] == "no stream for this pass" and not exact_diff(base["fit"], out["fit"])
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    assert _until(lambda: _stream_dirs(tmp) == [], 30.0), _stream_dirs(tmp)
    # 프레임률을 모르면 시작하지 않는다 (예외 없음)
    f3 = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=2, fps=float("nan"))
    assert not f3.start([]) and f3.stop()["stream_sent_frames"] == 0
    # 실행기 스레드를 못 읽는 파이썬 (내부 속성 없음): 서버 CPU 합은 NaN — 잰 피더 스레드는 그대로
    monkeypatch.setattr(PS, "_executor_threads", lambda pool: None)
    f4 = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps)
    assert f4.start([_rec(f) for f in frames[:10]])
    assert _until(lambda: f4.frames == 10, 60.0)
    st4 = f4.stop()
    f4.cancel()
    assert math.isnan(st4["stream_executor_cpu_s"]) and math.isnan(st4["stream_server_cpu_s"]), st4
    assert st4["stream_feeder_cpu_s"] > 0.0 and st4["stream_sent_frames"] == 10, st4
    PS.post_scan_pool().submit(int).result(PS.RESULT_TIMEOUT_S)        # cancel 의 stream_close 뒤
    assert _until(lambda: _stream_dirs(tmp) == [], 30.0), _stream_dirs(tmp)


@pytest.mark.parametrize("workers", [1, 4])
def test_seed_feeder_server_cost_is_bounded_by_the_pipe_not_by_frames_or_workers(synthetic_pass, workers):
    """서버 쪽 비용 (피더가 깨는 횟수·writev) 은 받는 쪽 파이프 용량으로 묶인다: 쓰는 주기 = 파이프 한 통의 프레임 수 / 프레임률, 깨면 쌓인
    것을 한 번에 — 깨는 횟수 ≲ 2 × (프레임 / 파이프 한 통) (주기 + 파이프가 찼을 때 이어 쓰기) + 시작·멈춤, 작업자 수와 무관 (첫 판의 주기
    작업자 수 / 프레임률은 작업자 1 이면 프레임마다 깼다)."""
    frames = synthetic_pass["frames"]
    spec, kw = pass_feature_inputs(synthetic_pass["fit_kwargs"])
    fps = 200.0                                                         # 녹화 흉내를 이 프레임률로 (시험 시간)
    PS.warm_post_scan_worker(workers).result(PS.RESULT_TIMEOUT_S)
    records: list = []
    feeder = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=workers, fps=fps)
    assert feeder.start(records)
    assert _until(lambda: feeder._open.done(), PS.RESULT_TIMEOUT_S)     # 녹화 전에 스트림이 열렸다 (따라잡기 없이 재려고)
    t0 = time.monotonic()
    for i, f in enumerate(frames):
        time.sleep(max(0.0, t0 + i / fps - time.monotonic()))
        records.append(_rec(f))
    assert _until(lambda: feeder.frames == len(frames), 30.0)
    st = feeder.stop()
    PS.post_scan_pool().submit(PS.stream_close).result(PS.RESULT_TIMEOUT_S)
    pf, n = st["stream_pipe_frames"], len(frames)
    frame_bytes = _HDR_BYTES + 8 * frames[0].x_mm.size
    assert pf >= workers and st["stream_write_period_s"] == pf / fps, st
    assert pf * frame_bytes <= feeder.pipe_bytes < (pf + 1) * frame_bytes, (st, feeder.pipe_bytes)
    bound = 2 * math.ceil(n / pf) + 3
    assert st["stream_sent_frames"] == n and st["stream_feeder_wakeups"] <= bound and st["stream_sent_writes"] <= bound, (st, bound)
    assert bound < n / 2                                                # 프레임 수보다 훨씬 적다


def test_worker_death_during_the_pass_is_detected_by_process_state(synthetic_pass, tmp_path, monkeypatch):
    """녹화 중 작업 프로세스가 죽는다 (SIGKILL): 스트림 계산 프로세스는 서버와의 연결 fd 를 붙들지 않았고 함께 죽는다 → 서버 실행기는 작업
    프로세스의 센티널 (프로세스 상태) 로 바로 안다 (예전: 계산 프로세스가 센티널을 붙들어 RESULT_TIMEOUT_S 300 s 를 기다리고 고아로 남아 서버
    종료를 막았다) → 피더는 멈추고, 패스 뒤 처리는 다음 쓰임에 새로 만든 작업 프로세스에서 (post_scan_pool — 작업 프로세스가 끝난 실행기는
    다음에 쓸 때 새로; 죽은 작업 프로세스의 스트림 없이 처음부터) 같은 결과, 남는 프로세스·스트림 디렉터리·실행기 스레드 없음."""
    from concurrent.futures.process import BrokenProcessPool
    p = dict(synthetic_pass, fit_kwargs=dict(synthetic_pass["fit_kwargs"], feature_workers=2))
    base = _baseline(p, tmp_path)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    frames = p["frames"]
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    pool, wpid, server_fds = _fresh_worker(2)
    records = [_rec(f) for f in frames[:len(frames) // 2]]
    feeder = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps)
    assert feeder.start(records)
    assert _until(lambda: feeder._open.done() and feeder.frames == len(records), 60.0)
    kids = feeder._open.result()["procs"]
    assert len(kids) == 2 and set(kids) <= set(_children(wpid)), (kids, _children(wpid))
    for k in kids:                                                      # 서버와의 연결 (실행기 파이프·센티널) 을 붙들지 않는다
        assert _until(lambda: not (_fd_targets(k) & server_fds), 10.0), (k, _fd_targets(k) & server_fds)
    t_kill = time.monotonic()
    os.kill(wpid, signal.SIGKILL)
    records.extend(_rec(f) for f in frames[len(frames) // 2:])         # 녹화는 계속된다
    assert _until(lambda: not any(_running(k) for k in kids), 30.0), [k for k in kids if _running(k)]
    try:                                                                # 서버 실행기가 아는 데 걸린 시간 (센티널)
        exc = pool.submit(os.getpid).exception(PS.RESULT_TIMEOUT_S)
    except BrokenProcessPool as e:
        exc = e
    t_detect = time.monotonic() - t_kill
    assert isinstance(exc, BrokenProcessPool) and t_detect < 30.0, (exc, t_detect)
    mt = pool._executor_manager_thread                                  # 서버 종료를 막는 실행기 스레드가 남지 않는다 (아래)
    st = feeder.stop()
    assert feeder.closed_by_worker or "stream_feeder_error" in st, st
    out = PS.run_post_scan(dict(p, out_dir=str(tmp_path / "dead"), stream_token=feeder.token))
    assert out["where"] == "worker" and "worker_error" not in out, (out["where"], out.get("worker_error"))
    assert out["timing"].get("stream_note") == "no stream for this pass" and PS.post_scan_pool() is not pool, out["timing"]
    assert not exact_diff(base["fit"], out["fit"])
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    assert _stream_dirs(tmp) == []
    assert mt is None or (mt.join(30.0) or not mt.is_alive())
    PS.shutdown_post_scan_pool()


@pytest.fixture(scope="module")
def long_pass(synthetic_pass):
    """같은 장면을 느리게 (프로파일 4 배) — 특징 풀이 살아 있는 동안 작업 프로세스를 죽이려고."""
    calib, job0 = synthetic_pass["calib"], synthetic_pass["job"]
    job = ScanJob(start=job0.start, end=job0.end, surface_normal=[0, 0, 1], table_z_m=-0.05, specimen_top_z_m=0.05, lead_in_m=0.02,
                  lead_out_m=0.02, speed_mps=0.01, accel_mps2=0.5, profile_fps=20.0, pitch_deg=-5.0, job_id="stream_syn_long")
    scene = GrooveScene.from_spec(synthetic_pass["fit_kwargs"]["spec"], yc=0.002)
    scene.x_range = (0.01, 0.11)
    syn = simulate_scan(job, calib, scene, SynthConfig(n_points=704, seed=11, sigma_z_mm=0.02, dropout_prob=0.01,
                                                       robot_poll_rate_hz=0.0))
    fr = sensor_frames(job, calib)
    fk = dict(synthetic_pass["fit_kwargs"], u_prior=job.unit_dir, n_prior=job.normal_unit, u_travel=fr.u_travel,
              target_standoff_m=float(job.standoff_m), gravity_axis_base=job.gravity_axis_base, feature_workers=2)
    return dict(synthetic_pass, frames=syn.frames, robot=syn.robot, job=job, fit_kwargs=fk)


def test_worker_death_inside_the_feature_pool_is_detected_by_process_state(long_pass, tmp_path):
    """패스 뒤 처리의 특징 풀 (fork, 보통 우선순위) 이 도는 동안 작업 프로세스가 죽는다: 풀 프로세스도 서버와의 연결 fd 를 놓았고 함께 죽어
    서버는 시간 초과가 아니라 센티널로 바로 알고 서버 안에서 같은 결과 (2026-10-08 전부터 작업자 ≥ 2 에서 있던 정지)."""
    p = long_pass
    t0 = time.monotonic()
    base = _baseline(p, tmp_path)
    t_base = time.monotonic() - t0
    pool, wpid, server_fds = _fresh_worker(2)
    box: dict = {}
    th = threading.Thread(target=lambda: box.update(out=PS.run_post_scan(dict(p, out_dir=str(tmp_path / "dead")))))
    th.start()
    assert _until(lambda: len(_children(wpid)) >= 2, 120.0), "the feature pool did not start"
    kids = _children(wpid)
    for k in kids:
        assert _until(lambda: not (_fd_targets(k) & server_fds), 10.0), (k, _fd_targets(k) & server_fds)
    t_kill = time.monotonic()
    os.kill(wpid, signal.SIGKILL)
    th.join(PS.RESULT_TIMEOUT_S + 60.0)
    t_done = time.monotonic() - t_kill
    assert not th.is_alive()
    out = box["out"]
    assert out["where"] == "in_process" and "BrokenProcessPool" in out["worker_error"], out.get("worker_error")
    assert t_done < 5.0 * t_base + 30.0, (t_done, t_base)               # 감지 + 서버 안 계산 (시간 초과 300 s 가 아니다)
    assert not exact_diff(base["fit"], out["fit"])
    assert _until(lambda: not any(_running(k) for k in kids), 30.0), [k for k in kids if _running(k)]
    PS.shutdown_post_scan_pool()


def test_a_dead_feature_pool_child_is_recomputed_in_the_worker_at_once(long_pass, tmp_path, monkeypatch):
    """특징 풀 (작업 프로세스 안 fork) 의 자식 하나가 패스 도중 죽는다 (SIGKILL), 작업 프로세스는 산다: 풀의 주인 (작업 프로세스) 은 그 죽음을
    센티널 (프로세스 상태) 로 바로 알고 풀이 돌려주지 못한 덩어리만 자기 안에서 같은 계산으로 — 패스는 작업 프로세스에서 끝나고 (서버는
    RESULT_TIMEOUT_S 를 기다리지 않고 작업 프로세스를 버리지 않는다), 결과는 같고, 풀 자식은 모두 끝나 거둬졌고, 스트림 디렉터리는 남지 않는다.
    예전: 주인이 호출 파이프의 읽는 끝을 들고 있어 자식이 모두 끝나도 보내기 스레드가 가득 찬 파이프에서 EPIPE 를 받지 못하고 영원히 막혔다 →
    풀 정리와 그것을 기다리는 with 의 끝이 멈춰 서버가 300 s 뒤 작업 프로세스를 죽이고 서버 안에서 다시 계산했다."""
    p = long_pass
    t0 = time.monotonic()
    base = _baseline(p, tmp_path)
    t_base = time.monotonic() - t0
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    frames = p["frames"]
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    pool, wpid, _ = _fresh_worker(2)
    records = [_rec(f) for f in frames[:len(frames) // 2]]               # 스트림은 앞 절반만 — 나머지 행은 패스 뒤 특징 풀이 처음부터
    feeder = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps)
    assert feeder.start(records)
    assert _until(lambda: feeder._open.done() and feeder.frames == len(records), 120.0)
    streamers = set(feeder._open.result()["procs"])
    feeder.stop()
    box: dict = {}
    th = threading.Thread(target=lambda: box.update(out=PS.run_post_scan(dict(p, out_dir=str(tmp_path / "kid"), stream_token=feeder.token))))
    th.start()

    def pool_kids():
        return [k for k in _children(wpid) if k not in streamers and _running(k)]
    assert _until(lambda: len(pool_kids()) >= 2, 120.0), "the feature pool did not start"
    kids = pool_kids()
    t_kill = time.monotonic()
    os.kill(kids[0], signal.SIGKILL)
    assert _until(lambda: not any(_running(k) for k in kids), 30.0), [k for k in kids if _running(k)]
    t_detect = time.monotonic() - t_kill                                # 주인이 센티널로 알고 나머지 자식을 끝냈다
    th.join(PS.RESULT_TIMEOUT_S + 60.0)
    t_done = time.monotonic() - t_kill
    assert not th.is_alive()
    out = box["out"]
    t = out["timing"]
    assert out["where"] == "worker" and "worker_error" not in out, (out["where"], out.get("worker_error"))
    assert t.get("fit_features_mode") == "fallback" and "BrokenProcessPool" in t.get("fit_features_pool_error", ""), t
    assert t_done < 5.0 * t_base + 30.0 and t_done < PS.RESULT_TIMEOUT_S / 4, (t_done, t_base)   # 감지 + 못 받은 덩어리 계산뿐
    assert not exact_diff(base["fit"], out["fit"])
    assert not any(_running(k) for k in kids) and not set(kids) & set(_children(wpid)), (kids, _children(wpid))   # 끝났고 거둬졌다
    assert pool.submit(os.getpid).result(PS.RESULT_TIMEOUT_S) == wpid   # 같은 작업 프로세스가 그대로 일한다
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    assert _stream_dirs(tmp) == []
    print(f"\nfeature-pool child killed: other pool child ended {t_detect:.2f} s, pass done in the worker {t_done:.2f} s after the kill "
          f"(baseline pass {t_base:.2f} s, limit {PS.RESULT_TIMEOUT_S:.0f} s); {t['fit_features_pool_error']}")
    PS.shutdown_post_scan_pool()


_LIVE_PROBE = '''
import os
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool

ERRORS = {"BrokenProcessPool": BrokenProcessPool, "TimeoutError": FutureTimeout}


class OutDir:
    """out_dir: 만든 프로세스 (서버 역할) 에서는 그 경로, 다른 프로세스 (작업 프로세스) 에서는 그 예외 — 산 작업 프로세스 안에서 생긴
    BrokenProcessPool·TimeoutError 가 피클로 서버에 돌아오는 경우."""

    def __init__(self, path, error="BrokenProcessPool"):
        self.path, self.owner, self.error = path, os.getpid(), error

    def __fspath__(self):
        if os.getpid() != self.owner:
            raise ERRORS[self.error]("raised inside a live worker")
        return self.path
'''


@pytest.mark.parametrize("error", ["BrokenProcessPool", "TimeoutError"])
def test_an_exception_from_a_live_worker_is_not_taken_for_its_death(synthetic_pass, tmp_path, monkeypatch, error):
    """살아 있는 작업 프로세스의 계산이 BrokenProcessPool·TimeoutError 를 낸다 (피클로 서버에 돌아온다): 서버는 예외 종류가 아니라 작업 프로세스 자신의 상태
    (센티널) 로 가른다 → 'alive' — 끝내지 않고 보통으로 내린다 (스스로 끝난다: 종료 코드 0), 패스는 서버 안에서 같은 결과. 예전: 모든
    BrokenProcessPool 을 '작업 프로세스가 죽었다' 로 보고 산 작업 프로세스를 버렸다."""
    p = synthetic_pass
    base = _baseline(p, tmp_path)
    (tmp_path / "live_probe.py").write_text(_LIVE_PROBE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))                          # 새 작업 프로세스 (spawn) 도 이 경로로 연다
    import live_probe
    pool, wpid, _ = _fresh_worker(1)
    (kid,) = pool.children()
    logs: list = []
    out = PS.run_post_scan(dict(p, out_dir=live_probe.OutDir(str(tmp_path / "live"), error)), log=logs.append)
    assert out["where"] == "in_process" and out["worker_state"] == "alive", (out["where"], out.get("worker_state"))
    assert error in out["worker_error"] and "worker alive" in logs[0], (out["worker_error"], logs)   # 예외 종류가 아니라 상태로
    assert not exact_diff(base["fit"], out["fit"])
    assert _until(lambda: kid.exitcode is not None, 30.0) and kid.exitcode == 0, kid.exitcode   # 끝내지 않았다 (SIGKILL 이면 −9)


def test_worker_death_while_the_pass_data_is_being_sent_does_not_block_the_server(synthetic_pass, tmp_path):
    """작업 프로세스가 패스 원자료 (파이프 용량보다 큼) 를 다 받기 전에 죽는다: 서버는 프로세스 상태로 바로 알고 서버 안에서 같은 결과, 그리고
    서버 실행기의 관리 스레드가 끝난다 (서버 종료가 멈추지 않는다 — 예전: 아무도 읽지 않는 호출 파이프에 쓰던 스레드가 영원히 막혀 정리와 종료가
    그것을 기다렸다)."""
    p = synthetic_pass
    base = _baseline(p, tmp_path)
    pool, wpid, _ = _fresh_worker(1)
    cap = PS._pipe_capacity(pool._call_queue._writer.fileno())
    assert len(pickle.dumps(dict(p, out_dir="x", defer_save=True, in_worker=True), protocol=pickle.HIGHEST_PROTOCOL)) > 2 * cap > 0
    os.kill(wpid, signal.SIGSTOP)                                       # 받기 전에 멈춤 → 보내기가 파이프 용량에서 막힌다
    box: dict = {}
    th = threading.Thread(target=lambda: box.update(out=PS.run_post_scan(dict(p, out_dir=str(tmp_path / "mid")))))
    th.start()
    assert _until(lambda: pool._call_queue._buffer or pool._pending_work_items, 30.0)
    time.sleep(0.5)                                                     # 보내기 스레드가 파이프에 쓰는 중 (가득)
    mt = pool._executor_manager_thread
    os.kill(wpid, signal.SIGKILL)
    th.join(PS.RESULT_TIMEOUT_S)
    assert not th.is_alive()
    out = box["out"]
    assert out["where"] == "in_process" and "BrokenProcessPool" in out["worker_error"], out.get("worker_error")
    assert not exact_diff(base["fit"], out["fit"])
    mt.join(30.0)
    assert not mt.is_alive(), "the executor manager thread is still blocked (server exit would hang)"


def test_a_stalled_worker_is_abandoned_at_the_limit(synthetic_pass, tmp_path, monkeypatch):
    """살았지만 응답이 없는 작업 프로세스 (SIGSTOP): RESULT_TIMEOUT_S 뒤 서버는 그 프로세스를 끝내고 (그래야 실행기가 정리되고 서버 종료가 멈추지
    않는다) 서버 안에서 같은 결과."""
    p = synthetic_pass
    base = _baseline(p, tmp_path)
    pool, wpid, _ = _fresh_worker(1)
    mt = pool._executor_manager_thread                                  # (shutdown 이 참조를 놓기 전에)
    os.kill(wpid, signal.SIGSTOP)
    monkeypatch.setattr(PS, "RESULT_TIMEOUT_S", 2.0)                    # 시험 시간만 (한계의 값이 아니라 한계 뒤의 정리를 본다)
    t0 = time.monotonic()
    out = PS.run_post_scan(dict(p, out_dir=str(tmp_path / "stalled")))
    assert out["where"] == "in_process" and "TimeoutError" in out["worker_error"], out.get("worker_error")
    assert not exact_diff(base["fit"], out["fit"])
    assert _until(lambda: not _running(wpid), 30.0), "the stalled worker was left running"
    mt.join(30.0)
    assert not mt.is_alive() and time.monotonic() - t0 < 60.0


def test_wait_result_ends_on_the_result_the_worker_death_or_the_cancel():
    """S4 기다림 규칙 (wait_result): 결과 (예외 포함) · 취소 Future · 한계 중 먼저 — 결과가 와 있으면 취소보다 먼저 (받은 것은 잃지 않는다)."""
    from concurrent.futures import Future
    from concurrent.futures import TimeoutError as FutureTimeout
    done, cancel = Future(), Future()
    done.set_result(7)
    assert PS.wait_result(done, cancel) == 7
    cancel.set_result("operator cancel")
    assert PS.wait_result(done, cancel) == 7                            # 둘 다 끝났으면 결과
    with pytest.raises(PS.Cancelled, match="operator cancel"):
        PS.wait_result(Future(), cancel)                                # 끝나지 않을 일 + 끝난 취소: 바로
    failed = Future()
    failed.set_exception(BrokenPipeError("worker died"))
    with pytest.raises(BrokenPipeError):
        PS.wait_result(failed, Future())                                # 일의 예외 (작업 프로세스의 죽음 = BrokenProcessPool) 는 그대로
    with pytest.raises(FutureTimeout):
        PS.wait_result(Future(), Future(), limit_s=0.05)                # 살았지만 멈춘 작업 프로세스의 한계 (취소 없음)


def test_a_cancel_ends_the_wait_on_a_stopped_worker_at_once(synthetic_pass, tmp_path):
    """S4 (2026-10-09 심층 분석): 작업 프로세스가 멈춘 (SIGSTOP) 채 패스 뒤 처리를 기다리는 중에 실행이 취소된다 (서버의 액션 취소가 cancel Future
    를 끝낸다) → 기다림은 한계 (RESULT_TIMEOUT_S 300 s) 가 아니라 취소에서 바로 끝나 Cancelled — 서버 안 다시 계산 없음. 끝나지 않은 일을 쥔
    작업 프로세스는 한계에서와 같은 규칙으로 끝낸다 (고아·다음 실행을 붙드는 멈춘 작업 프로세스 없음), 실행기의 관리 스레드도 끝난다."""
    from concurrent.futures import Future
    p = synthetic_pass
    pool, wpid, _ = _fresh_worker(1)
    mt = pool._executor_manager_thread
    os.kill(wpid, signal.SIGSTOP)
    cancel = Future()
    box: dict = {}
    logs: list = []

    def run():
        try:
            box["out"] = PS.run_post_scan(dict(p, out_dir=str(tmp_path / "cancelled")), cancel=cancel, log=logs.append)
        except PS.Cancelled as e:
            box["cancelled"] = str(e)
    th = threading.Thread(target=run)
    th.start()
    assert _until(lambda: pool._pending_work_items or not th.is_alive(), 30.0)   # 맡겼다: 실행은 기다리는 중
    assert th.is_alive()
    t0 = time.monotonic()
    cancel.set_result("auto_teach action cancel requested")
    th.join(PS.RESULT_TIMEOUT_S)
    dt = time.monotonic() - t0
    assert not th.is_alive() and box == {"cancelled": "auto_teach action cancel requested"}, box
    assert dt < 10.0, f"the cancelled wait took {dt:.1f} s"
    assert any("cancelled while waiting for the worker" in m for m in logs) and not any("computing in the server" in m for m in logs), logs
    assert _until(lambda: not _running(wpid), 30.0), "the stopped worker was left (orphan / blocks the next run)"
    mt.join(30.0)
    assert not mt.is_alive(), "the executor manager thread is still blocked (server exit would hang)"
    assert PS.post_scan_pool() is not pool                               # 다음 사용은 새 작업 프로세스
    PS.shutdown_post_scan_pool()
    print(f"\ncancel -> wait returned in {dt * 1e3:.0f} ms (limit {PS.RESULT_TIMEOUT_S:.0f} s)")


def test_a_dead_stream_compute_process_only_stops_the_stream(synthetic_pass, tmp_path, monkeypatch):
    """스트림 계산 프로세스 하나가 죽는다 (SIGKILL): 작업 프로세스는 그 소켓의 끝으로 바로 알고 스트림만 멈춘다 — 패스 뒤 처리는 작업
    프로세스에서, 받은 결과만 쓰고 나머지는 처음부터 (같은 결과), 다른 계산 프로세스도 끝나고 디렉터리는 남지 않는다."""
    p = dict(synthetic_pass, fit_kwargs=dict(synthetic_pass["fit_kwargs"], feature_workers=2))
    base = _baseline(p, tmp_path)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    frames = p["frames"]
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    pool, wpid, _ = _fresh_worker(2)
    records = [_rec(f) for f in frames[:len(frames) // 2]]
    feeder = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps)
    assert feeder.start(records)
    assert _until(lambda: feeder._open.done() and feeder.frames == len(records), 60.0)
    kids = feeder._open.result()["procs"]
    os.kill(kids[0], signal.SIGKILL)
    records.extend(_rec(f) for f in frames[len(frames) // 2:])
    st = feeder.stop()
    assert "stream_feeder_error" not in st, st
    out = PS.run_post_scan(dict(p, out_dir=str(tmp_path / "kid"), stream_token=feeder.token))
    t = out["timing"]
    assert out["where"] == "worker" and "ended" in t.get("stream_error", ""), t
    assert not exact_diff(base["fit"], out["fit"])
    assert _until(lambda: not any(_running(k) for k in kids), 30.0), [k for k in kids if _running(k)]
    assert _stream_dirs(tmp) == []
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    PS.shutdown_post_scan_pool()


def test_a_worker_that_died_between_passes_is_replaced_at_the_next_use(synthetic_pass, tmp_path, monkeypatch):
    """작업 프로세스가 패스 사이 (쉬는 동안 — 예: 미룬 파일 쓰기 중 메모리 부족) 에 죽는다 (SIGKILL): 다음에 쓸 때 (접근 중 데우기·녹화 중 스트림
    시작·패스 뒤 처리) 새 작업 프로세스 — 그 패스는 스트림도 작업 프로세스 경로도 잃지 않고 같은 결과 (특징 풀 seam_features._feature_pool 과
    같은 규칙: 자식이 끝난 풀은 다음에 쓸 때 새로). 예전: 깨진 실행기를 계속 내줘 그 패스의 데우기·스트림이 BrokenProcessPool 로 실패하고 패스
    뒤 처리가 서버 안에서 돌았다."""
    p = dict(synthetic_pass, fit_kwargs=dict(synthetic_pass["fit_kwargs"], feature_workers=2))
    base = _baseline(p, tmp_path)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    frames = p["frames"]
    spec, kw = pass_feature_inputs(p["fit_kwargs"])
    old, wpid, _ = _fresh_worker(2)
    os.kill(wpid, signal.SIGKILL)
    assert _until(old.child_died, 30.0)
    PS.warm_post_scan_worker(2).result(PS.RESULT_TIMEOUT_S)            # 다음 패스: 접근 중 데우기
    pool = PS.post_scan_pool()
    assert pool is not old and pool.submit(os.getpid).result(PS.RESULT_TIMEOUT_S) != wpid
    feeder = PS.SeedFeeder(spec, kw, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps)
    assert feeder.start([_rec(f) for f in frames])
    assert _until(lambda: feeder._open.done() and feeder.frames == len(frames), 60.0)
    st = feeder.stop()
    assert "stream_feeder_error" not in st and st["stream_sent_frames"] == len(frames), st
    out = PS.run_post_scan(dict(p, out_dir=str(tmp_path / "next"), stream_token=feeder.token))
    t = out["timing"]
    assert out["where"] == "worker" and "worker_error" not in out, (out["where"], out.get("worker_error"))
    assert t["stream_received"] == len(frames) and "stream_error" not in t and "stream_note" not in t, t
    assert not exact_diff(base["fit"], out["fit"])
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    assert _until(lambda: _stream_dirs(tmp) == [], 30.0), _stream_dirs(tmp)
    assert all(k.exitcode is not None for k in old.children())
    PS.shutdown_post_scan_pool()


# 서버 역할 프로세스 (시험이 kill -9 한다): 첫 사용은 짧은 스레드에서 (패스 피더처럼) — 그 스레드가 끝나도 작업 프로세스는 산다; 작업 프로세스는
# 바쁘다 (FIFO 읽기에서 막힘 — 쓰는 끝은 시험이 든다: 계산·쓰기 중에는 호출 파이프의 끝 (EOF) 을 읽지 않는다, 예전에 고아로 남던 경우)
_SERVER_HARNESS = r"""
import os, pathlib, sys, threading, time
from scan_l0 import post_scan as PS
box = {}
th = threading.Thread(target=lambda: box.update(pool=PS.post_scan_pool()))
th.start()
th.join()
while os.path.exists(f"/proc/self/task/{th.native_id}"):     # 그 OS 스레드가 정말 끝날 때까지 (끝나는 순간 PDEATHSIG 가 갈 수 있다)
    time.sleep(0.005)
pool = box["pool"]
wpid = pool.submit(os.getpid).result(60)
pool.submit(pathlib.Path(sys.argv[1]).read_bytes)
print(wpid, flush=True)
sys.stdin.read()
"""


def test_the_worker_dies_with_the_server(tmp_path):
    """S3 (2026-10-09 심층 분석): 작업 프로세스 (spawn) 는 계산 프로세스 (fork) 와 같은 원시로 서버와 함께 죽는다 — PR_SET_PDEATHSIG(SIGKILL) +
    부모 재확인 (_worker_init). 서버 역할 프로세스를 kill -9 하면 바쁜 작업 프로세스도 바로 끝난다 (예전: 계산·쓰기 중이거나 멈춘 작업 프로세스는
    고아로 남아 메모리·CPU·스트림 디렉터리를 쥐었다). 리눅스의 '부모' 는 띄운 스레드라 띄우기는 서버 수명 동안 사는 스레드에서 한다: 첫 사용을 한
    짧은 스레드가 끝나도 작업 프로세스는 살아 답한다 (하네스 안에서 확인)."""
    import select
    import subprocess
    fifo = str(tmp_path / "busy")
    os.mkfifo(fifo)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    proc = subprocess.Popen([sys.executable, "-c", _SERVER_HARNESS, fifo], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                            env=env)
    wpid, wfd = None, None
    try:
        ready, _, _ = select.select([proc.stdout], [], [], 120.0)
        assert ready, "the harness did not report its worker"
        wpid = int(proc.stdout.readline())
        assert _running(wpid) and wpid not in (proc.pid, os.getpid())

        def _open_writer():                                              # 작업 프로세스가 FIFO 를 읽기로 열면 열린다 (= 그 일을 시작했다)
            nonlocal wfd
            try:
                wfd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                return False
            return True
        assert _until(_open_writer, 60.0), "the worker did not start its job"
        pfd = os.pidfd_open(wpid)                                        # 작업 프로세스의 끝을 사건으로 (pidfd 가 읽기 가능 = 끝남)
        try:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()
            gone, _, _ = select.select([pfd], [], [], 30.0)
        finally:
            os.close(pfd)
        assert gone and not _running(wpid), "the busy post-scan worker outlived its server (orphan)"
    finally:
        if wfd is not None:
            os.close(wfd)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if wpid is not None and _running(wpid):
            os.kill(wpid, signal.SIGKILL)                                # 실패해도 남기지 않는다 (이 시험이 만든 작업 프로세스)


def _batches(frames, size):
    for a in range(0, len(frames), size):
        yield frames[a:a + size]


def _track_payload(p, out_dir, **kw):
    """같은 합성 패스를 S 추적·트랜싯처럼 (P2): 원자료 + 재구성 + 행마다 특징 + 심 맞춤 (scan_core.seam_track.fit_track)."""
    job = p["job"]
    d = dict(out_dir=str(out_dir), arrays={}, js={}, frames=p["frames"], robot=p["robot"], calib=p["calib"], z_valid_mm=Z_VALID,
             fps=float(job.profile_fps), spec=p["fit_kwargs"]["spec"], kw={"qc_targets": None}, depth_scale=0.99619, still_tol_m=0.5e-3,
             part=0, fit=dict(k=3.0, tol_root_m=1.0e-3, t_hat0=[float(v) for v in job.unit_dir], floor_m=2e-5, source="transit"),
             prior=None, stride=1, defer_save=True)
    d.update(kw)
    return d


def test_track_stream_gives_the_same_rows_and_seam_line(synthetic_pass, tmp_path):
    """S 추적·트랜싯 특징 스트림 (FeatureFeeder → feature_stream_*): 이동 중에 계산한 프레임 특징을 쓴 행·맞춤 == 스트림 없이 이동 뒤에
    모두 계산한 것 (바이트까지). 스트림에 없는 프레임 (이동 끝 뒤) 과 값이 다른 프레임은 이동 뒤에 계산한다."""
    p = synthetic_pass
    base = PS.compute_track(_track_payload(p, tmp_path / "t0"))
    PS._PENDING.clear()
    assert base["est"] is not None and base["outcome"] == "ok", (base["outcome"], base["info"])
    assert abs(float(np.asarray(base["est"].u) @ p["job"].unit_dir)) > 0.9999
    frames = p["frames"]
    bad = frames[len(frames) // 2].result_counter                       # 판 위 프레임 (값이 유한 — 바꾸면 다르다)
    PS.feature_stream_open("k1", p["fit_kwargs"]["spec"], {"qc_targets": None}, Z_VALID, 2, 0.99619)
    for fr in _batches(frames[:-3], 5):
        PS.feature_stream_batch("k1", [f.result_counter for f in fr],
                                [f.x_mm + np.float32(0.5) if f.result_counter == bad else f.x_mm for f in fr], [f.z_mm for f in fr])
    assert PS._FSTREAM["frames"] == len(frames) - 3 and PS._FSTREAM["error"] is None
    out = PS.compute_track(_track_payload(p, tmp_path / "t1", stream_token="k1"))
    PS._PENDING.clear()
    assert PS._FSTREAM is None                                          # 맞춤 뒤 스트림을 닫았다
    t = out["timing"]
    assert t["track_from_stream"] == len(frames) - 4 and t["track_computed_after"] == 4, t
    assert not exact_diff(base["rows"], out["rows"]), exact_diff(base["rows"], out["rows"])
    assert not exact_diff(base["est"], out["est"]), exact_diff(base["est"], out["est"])
    # 피더 보폭 (부하에 따라 달라진다) 은 이동 중 미리 계산한 행만 정한다 — 이동 뒤 나머지 행을 모두 계산하므로 행·맞춤은 보폭과 무관
    # (바이트까지; 예전에는 남은 행을 같은 보폭으로 골라 부하가 맞춤 행 집합을 바꿨다). 작업자 2 — 이 프로세스가 주인인 fork OwnedPool 에
    # 패스의 특징 단계와 같은 함수로 나눠도 (seam_features._features_for) 같다
    half = PS.compute_track(_track_payload(p, tmp_path / "t2", stride=2, workers=2))
    PS._PENDING.clear()
    assert half["timing"]["track_computed_after"] == len(frames) and half["timing"]["track_feeder_stride"] == 2
    assert half["timing"]["track_features_workers"] == 2 and half["timing"]["track_features_mode"] == "executor", half["timing"]
    assert not exact_diff(base["rows"], half["rows"]), exact_diff(base["rows"], half["rows"])
    assert not exact_diff(base["est"], half["est"]), exact_diff(base["est"], half["est"])


def test_feature_feeder_streams_the_recording_to_the_worker(synthetic_pass, tmp_path):
    """서버 쪽 FeatureFeeder → 패스 뒤 처리 작업 프로세스 (spawn): 녹화 목록을 읽기만 해서 넘기고, run_track 은 그 특징을 써서 같은 맞춤."""
    p = synthetic_pass
    base = PS.compute_track(_track_payload(p, tmp_path / "f0"))
    PS._PENDING.clear()
    PS.warm_post_scan_worker(2).result(PS.RESULT_TIMEOUT_S)
    records: list = []
    feeder = PS.FeatureFeeder(p["fit_kwargs"]["spec"], {"qc_targets": None}, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps,
                              depth_scale=0.99619)
    assert feeder.start(records)
    for f in p["frames"]:
        records.append(SimpleNamespace(result_counter=f.result_counter, x=f.x_mm, z=f.z_mm))
        time.sleep(0.1 / float(p["job"].profile_fps))
    # 시험만: 피더가 녹화를 다 보낼 때까지 (보폭만큼 — 또는 피더가 멈출 때까지), 그리고 묶음 하나는 답을 받을 때까지
    assert _until(lambda: feeder.frames >= len(p["frames"]) // max(1, feeder.stride) and feeder.answered_frames >= 1, PS.RESULT_TIMEOUT_S), \
        (feeder.frames, feeder.answered_frames, feeder.error)
    st = feeder.stop()
    assert st["track_sent_frames"] >= 1 and "track_feeder_error" not in st, st
    # 이동 뒤 처리는 스트림이 못 준 행을 모두 (보폭 1) 계산한다 (피더 보폭은 부하에 따라 달라진다 — 맞춤은 그와 무관): 결과 = 스트림 없이
    # 모든 행을 계산한 것 (바이트까지), 그중 스트림 행이 쓰였다. 피더가 실제로 쓴 보폭을 그대로 넘긴다 (생산과 같은 호출)
    assert st["track_stride"] >= 1
    out = PS.run_track(_track_payload(p, tmp_path / "f1", stream_token=feeder.token, stride=st["track_stride"], workers=2))
    t = out["timing"]
    # 작업 프로세스 경로 (예전 1g-integ 808d81d: fork 풀 이름이 병합에서 사라져 스트림 묶음·이동 뒤 특징이 NameError 로 조용히 실패 → 서버 안)
    assert out["where"] == "worker" and "track_stream_error" not in t, t
    assert t["track_from_stream"] >= st["track_answered_frames"] >= 1, (t, st)     # 돌려받은 묶음의 행은 모두 쓰였다
    assert t["track_from_stream"] + t["track_computed_after"] == t["track_rows"] == len(p["frames"]), t
    assert out["est"] is not None and not exact_diff(base["rows"], out["rows"]) and not exact_diff(base["est"], out["est"])
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    f3 = PS.FeatureFeeder(p["fit_kwargs"]["spec"], {}, z_valid_mm=Z_VALID, workers=1, fps=float("nan"), depth_scale=1.0)
    assert not f3.start([]) and f3.stop()["track_sent_frames"] == 0


def _track_records(p) -> list:
    return [SimpleNamespace(result_counter=f.result_counter, x=f.x_mm, z=f.z_mm) for f in p["frames"]]


def _track_stream_kids() -> list:
    """(작업 프로세스 안에서 — 실행기로 부른다) 지금 추적 특징 스트림 계산 풀의 자식 pid (풀이 아직 없으면 빈 목록)."""
    st = PS._FSTREAM
    return [] if st is None or st["ex"] is None else [k.pid for k in st["ex"].children()]


def test_track_stream_batch_wait_is_bounded_and_a_stalled_stream_pool_is_ended(synthetic_pass, monkeypatch):
    """추적 특징 스트림 묶음의 기다림은 다른 모든 풀 기다림과 같은 한계 (wait_within, RESULT_TIMEOUT_S): 계산 풀의 자식이 멈추면 (SIGSTOP)
    묶음은 한계에서 돌아오고 (작업 프로세스를 붙잡지 않는다) 그 자식을 끝낸다 — 스트림은 멈추고 (오류 기록) 그 행은 이동 뒤에. 스트림 계산
    풀은 이 프로세스가 주인인 OwnedPool (fork). 닫기는 막히지 않는다."""
    from scan_core import proc_pool as PP
    p = synthetic_pass
    fr = p["frames"][10:14]
    PS.feature_stream_open("stall", p["fit_kwargs"]["spec"], {"qc_targets": None}, Z_VALID, 2, 0.99619)
    assert PS.feature_stream_batch("stall", [f.result_counter for f in fr[:2]], [f.x_mm for f in fr[:2]], [f.z_mm for f in fr[:2]])[0] == 2
    kids = _track_stream_kids()
    assert len(kids) == 2 and isinstance(PS._FSTREAM["ex"], PP.OwnedPool)
    monkeypatch.setattr(PP, "RESULT_TIMEOUT_S", 1.0)                   # 시험 시간만 (한계의 값이 아니라 한계 뒤를 본다)
    for k in kids:
        os.kill(k, signal.SIGSTOP)
    try:
        t0 = time.monotonic()
        r = PS.feature_stream_batch("stall", [f.result_counter for f in fr[2:]], [f.x_mm for f in fr[2:]], [f.z_mm for f in fr[2:]])
        dt = time.monotonic() - t0
        assert r == (0, 0.0) and "TimeoutError" in (PS._FSTREAM["error"] or "") and 0.9 < dt < 30.0, (r, PS._FSTREAM["error"], dt)
        assert _until(lambda: not any(_running(k) for k in kids), 30.0), [k for k in kids if _running(k)]
        assert PS.feature_stream_batch("stall", [fr[0].result_counter], [fr[0].x_mm], [fr[0].z_mm]) == (0, 0.0)   # 멈춘 스트림
        t0 = time.monotonic()
        PS.feature_stream_close()
        assert time.monotonic() - t0 < 30.0 and PS._FSTREAM is None
    finally:
        for k in kids:
            try:
                os.kill(k, signal.SIGKILL)
            except OSError:
                pass


def test_a_dead_track_stream_child_stops_only_the_stream(synthetic_pass, tmp_path):
    """추적 특징 스트림의 계산 프로세스 하나가 죽는다 (SIGKILL): 그 풀의 주인 (작업 프로세스) 은 프로세스 상태로 바로 알고 (OwnedPool — 나머지
    자식도 끝낸다) 스트림만 멈춘다 — 추적은 작업 프로세스에서 모든 행을 계산하고 같은 행·같은 심 직선."""
    p = synthetic_pass
    base = PS.compute_track(_track_payload(p, tmp_path / "b0"))
    PS._PENDING.clear()
    pool, _wpid, _ = _fresh_worker(2)
    recs = _track_records(p)
    records = recs[:len(recs) // 2]
    feeder = PS.FeatureFeeder(p["fit_kwargs"]["spec"], {"qc_targets": None}, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps,
                              depth_scale=0.99619)
    assert feeder.start(records)
    assert _until(lambda: feeder.answered_frames >= 1, PS.RESULT_TIMEOUT_S), feeder.error
    kids = pool.submit(_track_stream_kids).result(PS.RESULT_TIMEOUT_S)
    assert len(kids) == 2
    os.kill(kids[0], signal.SIGKILL)
    records.extend(recs[len(recs) // 2:])
    assert _until(lambda: feeder.frames >= len(recs) // max(1, feeder.stride), PS.RESULT_TIMEOUT_S), (feeder.frames, feeder.error)
    st = feeder.stop()
    out = PS.run_track(_track_payload(p, tmp_path / "b1", stream_token=feeder.token, stride=st["track_stride"], workers=2))
    t = out["timing"]
    assert out["where"] == "worker" and "BrokenProcessPool" in t.get("track_stream_error", ""), t
    assert t["track_from_stream"] + t["track_computed_after"] == t["track_rows"] == len(recs), t
    assert not exact_diff(base["rows"], out["rows"]) and not exact_diff(base["est"], out["est"])
    assert _until(lambda: not any(_running(k) for k in kids), 30.0), [k for k in kids if _running(k)]
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    PS.shutdown_post_scan_pool()


def test_worker_death_during_a_track_stream_ends_its_children_and_the_track_survives(synthetic_pass, tmp_path):
    """작업 프로세스가 추적 특징 스트림 중에 죽는다 (SIGKILL): 스트림 계산 프로세스는 서버와의 연결을 붙들지 않고 (예전 P2: 표준 fork 풀이라
    서버 fd 를 물려받았다 — 작업 프로세스의 죽음이 서버에 보이지 않을 수 있었다) 부모와 함께 죽고, 다음에 쓸 때 새 작업 프로세스가 모든
    행으로 같은 결과 (서버는 시간 한계가 아니라 프로세스 상태로 안다)."""
    p = synthetic_pass
    base = PS.compute_track(_track_payload(p, tmp_path / "d0"))
    PS._PENDING.clear()
    old, wpid, server_ends = _fresh_worker(2)
    feeder = PS.FeatureFeeder(p["fit_kwargs"]["spec"], {"qc_targets": None}, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps,
                              depth_scale=0.99619)
    assert feeder.start(_track_records(p)[:20])
    assert _until(lambda: feeder.answered_frames >= 1, PS.RESULT_TIMEOUT_S), feeder.error
    kids = old.submit(_track_stream_kids).result(PS.RESULT_TIMEOUT_S)
    assert kids and _until(lambda: all(not (_fd_targets(k) & server_ends) for k in kids), 30.0), "a stream child holds a server connection"
    t0 = time.monotonic()
    os.kill(wpid, signal.SIGKILL)
    assert _until(lambda: not any(_running(k) for k in kids), 30.0), [k for k in kids if _running(k)]
    assert _until(old.child_died, 30.0)
    st = feeder.stop()
    out = PS.run_track(_track_payload(p, tmp_path / "d1", stream_token=feeder.token, stride=st["track_stride"], workers=2))
    assert out["where"] == "worker" and out["timing"]["track_computed_after"] == len(p["frames"]), out["timing"]
    assert not exact_diff(base["rows"], out["rows"]) and not exact_diff(base["est"], out["est"])
    assert time.monotonic() - t0 < PS.RESULT_TIMEOUT_S / 4
    PS.submit_flush().result(PS.RESULT_TIMEOUT_S)
    PS.shutdown_post_scan_pool()


def test_a_stalled_worker_is_abandoned_by_the_track_wait(synthetic_pass, tmp_path, monkeypatch):
    """살았지만 응답이 없는 작업 프로세스 (SIGSTOP) 에 추적: 피더는 이동 끝 (stop) 에 기다림을 놓고, 서버의 추적 기다림 (run_track — 다른 작업
    프로세스 계산과 같은 규칙 _run_on_worker) 은 RESULT_TIMEOUT_S 에서 그 작업 프로세스를 멈춘 것 (상태) 으로 끝내고 서버 안에서 같은 행·같은
    심 직선 (예전 P2 run_track: 어떤 실패든 작업 프로세스를 끝내지 않고 내리기만 했다)."""
    p = synthetic_pass
    base = PS.compute_track(_track_payload(p, tmp_path / "s0"))
    PS._PENDING.clear()
    pool, wpid, _ = _fresh_worker(2)
    os.kill(wpid, signal.SIGSTOP)
    monkeypatch.setattr(PS, "RESULT_TIMEOUT_S", 2.0)                    # 시험 시간만 (한계의 값이 아니라 한계 뒤의 정리를 본다)
    try:
        t0 = time.monotonic()
        feeder = PS.FeatureFeeder(p["fit_kwargs"]["spec"], {"qc_targets": None}, z_valid_mm=Z_VALID, workers=2, fps=p["job"].profile_fps,
                                  depth_scale=0.99619)
        assert feeder.start(_track_records(p))
        assert _until(lambda: feeder.batches >= 1, 30.0)
        st = feeder.stop()                                              # 이동 끝
        assert _until(lambda: not feeder._thr.is_alive(), 30.0), "the feeder kept waiting after stop"
        logs: list = []
        out = PS.run_track(_track_payload(p, tmp_path / "s1", stream_token=feeder.token, stride=st["track_stride"], workers=2),
                           log=logs.append)
        assert out["where"] == "in_process" and out["worker_state"] == "stalled" and "TimeoutError" in out["worker_error"], out.get("worker_error")
        assert "track worker stalled" in logs[0], logs
        assert not exact_diff(base["rows"], out["rows"]) and not exact_diff(base["est"], out["est"])
        assert _until(lambda: not _running(wpid), 30.0), "the stalled worker was left running"
        assert time.monotonic() - t0 < 60.0
    finally:
        try:
            os.kill(wpid, signal.SIGKILL)
        except OSError:
            pass
    PS.shutdown_post_scan_pool()


def test_run_refit_follows_the_worker_rule(synthetic_pass, tmp_path):
    """다시 맞춤 (run_refit) 도 같은 규칙 (_run_on_worker): 작업 프로세스에서 같은 값, 작업 프로세스가 죽어 있으면 (다음에 쓸 때 새로)
    여전히 같은 값 — meta 에 where·timing."""
    p = synthetic_pass
    base = PS.compute_track(_track_payload(p, tmp_path / "r0"))
    PS._PENDING.clear()
    fit = _track_payload(p, tmp_path / "r0")["fit"]
    ref = PS.refit_track([base["rows"]], p["calib"], fit)
    _fresh_worker(1)
    meta: dict = {}
    got = PS.run_refit([base["rows"]], p["calib"], fit, meta=meta)
    assert meta["where"] == "worker" and "track_refit_wall_s" in meta["timing"], meta
    assert not exact_diff(ref, got)
    PS.shutdown_post_scan_pool()


@pytest.mark.skipif(not os.path.isdir(LAB_RUN), reason=f"lab replay data not found: {LAB_RUN}")
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_g1_b_lab_pass_identical_with_the_stream(k, tmp_path, monkeypatch):
    """실기 g1_b 패스 k: 오프라인 정련 (run_refine_offline — fit_pass → 끝·판정·정련 작업·용접 경로) 을 스트림 없이 한 결과 == 패스 뒤 처리
    작업 프로세스 계산 (in_worker fork 풀) 에 패스 중 스트림 (녹화 순서로 FIFO) 을 쓴 fit 출력과 그 위의 판정·정련 작업·용접 경로 (바이트까지)."""
    raw = os.path.join(LAB_RUN, f"pass_{k}", "raw.npz")
    settings = C.load_settings(os.path.join(STACK, "scan_l0", "tests", "data", "hongik_rb20_fake_moveit.yaml"))   # 플랫폼 무관 값만 쓴다
    with open(os.path.join(LAB_RUN, "autoteach_log.json"), encoding="utf-8") as fh:
        log = json.load(fh)
    _job, doc = C.load_job(str(C.JOBS_DIR / "groove_10T_1G.yaml"))
    doc = dict(doc, job=dict(log["job_rough"]))
    cap: dict = {}
    orig = A.fit_pass

    def fit_capture(result, spec, **kw):
        out = orig(result, spec, **kw)
        cap["kw"], cap["out"] = dict(kw, spec=spec), out
        return out
    monkeypatch.setattr(A, "fit_pass", fit_capture)
    ref = A.run_refine_offline(raw, doc, settings)
    monkeypatch.setattr(A, "fit_pass", orig)
    fk = cap["kw"]
    frames, robot, job_k, calib = A._load_raw_for_refine(raw)
    spec, kw = pass_feature_inputs(fk)
    PS.stream_open("g1b", spec, kw, settings.z_valid_mm, fk["feature_workers"])
    _feed(frames)
    _wait_stream(lambda st: st["frames"] == len(frames))
    out = PS.compute_pass(dict(out_dir=str(tmp_path / "w"), arrays={}, js={}, frames=frames, robot=robot, job=job_k, calib=calib,
                               z_valid_mm=settings.z_valid_mm, fit_kwargs=fk, defer_save=True, in_worker=True, stream_token="g1b"))
    PS._PENDING.clear()
    assert out["timing"]["stream_seeded"] == out["timing"]["stream_rows"] > 1000, out["timing"]
    assert not exact_diff(cap["out"], out["fit"]), exact_diff(cap["out"], out["fit"])
    seeded = out["fit"]
    monkeypatch.setattr(A, "fit_pass", lambda *a, **kw_: copy.deepcopy(seeded))
    got = A.run_refine_offline(raw, doc, settings)
    assert ref[2].ok and ref[3].status
    assert not exact_diff(ref, got), exact_diff(ref, got)               # 정련 작업·정보 (끝·시편 보고·용접 경로)·맞춤·판정·잔차
