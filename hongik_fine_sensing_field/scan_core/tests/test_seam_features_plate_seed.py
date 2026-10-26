"""패스 뒤 처리 시간 줄이기 (2026-10-08) — 결과는 바이트까지 같아야 한다:
  * 판 성장 씨앗 (PlateSeed): 패스 중 미리 계산한 판 성장 + 나머지 단계 == 한 번에 계산 — 그루브 종류·판 두께·롤·depth_scale (자세
    1G/3G·피치·입사가 바꾸는 유일한 추출 입력)·작업 허용치를 바꿔도. 판 성장은 (x, z)·설계·허용치만의 함수라 자세와 무관하다.
  * 이동 중앙값 (_window_medians): 창마다 np.median 과 같은 값 (−0.0·같은 값·빈 창 없는 모든 창 폭).
  * profile_points_base: np.outer 를 브로드캐스트 곱으로 — 같은 값."""
import dataclasses
import math
import os

import numpy as np
import pytest

from scan_core.frames import make_transform, rot_x
from scan_core.localize import profile_points_base
from scan_core.path_gen import plate_frame
from scan_core.reconstruct import reconstruct_scan
from scan_core import seam_features as SF
from scan_core.seam_features import (cross_section_plate_seed, extract_cross_section_features, extract_seam_features_along_scan,
                                     plate_seeds_chunk)
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_profile, simulate_scan
from scan_core.types import Calib, ScanJob

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REAL_NPZ = os.path.join(DATA, "real_groove_median_sh100_gain300.npz")
T_PLATE = plate_frame([0, 0, 0], [1, 0, 0], [0, 0, 1])
R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)


def _diff(a, b, path="", out=None) -> list:
    """엄격 비교: 배열 dtype·shape·바이트, float 비트 (NaN·−0 포함), dict 키 순서, dataclass 속성."""
    out = [] if out is None else out
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or a.dtype != b.dtype or a.shape != b.shape:
            out.append(f"{path}: array type/shape")
        elif a.dtype == object:
            for i, (x, y) in enumerate(zip(a.ravel(), b.ravel())):
                _diff(x, y, f"{path}[{i}]", out)
        elif a.tobytes() != b.tobytes():
            out.append(f"{path}: array bytes")
    elif type(a) is not type(b):
        out.append(f"{path}: {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, (float, np.floating)):
        if np.float64(a).tobytes() != np.float64(b).tobytes():
            out.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, dict):
        if list(a) != list(b):
            out.append(f"{path}: keys")
        else:
            for k in a:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: len")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diff(x, y, f"{path}[{i}]", out)
    elif dataclasses.is_dataclass(a):
        _diff(vars(a), vars(b), path, out)
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def _profile(spec, roll_deg, y_off_m=0.006, z_m=0.43, noise=0.02, dropout=0.02, seed=1, angle_deg=0.0):
    sc = GrooveScene.from_spec(spec, angle_deg=angle_deg)
    T_bo = make_transform(rot_x(math.radians(roll_deg)) @ R0, (0.0, y_off_m, z_m))
    return simulate_profile(T_bo, sc, T_PLATE, SynthConfig(n_points=1408, seed=seed, sigma_z_mm=noise, dropout_prob=dropout))


def _cases():
    """(이름, x, z, spec) — 그루브 종류 전부 (한쪽 베벨 좌/우·V·I형 (square)·두꺼운 판·루트면·갭 — test_groove_families 의 FAMILIES),
    연구실 시편 (베벨 좌/우, 롤 0/±8°, hi-lo, 비스듬), 실기 중앙값 프로파일, 그루브 없음, 판 각도 불일치, 점 부족, 전부 NaN, 한쪽 판만."""
    from test_groove_families import FAMILIES
    out = []
    fam = dict(FAMILIES, square_10T_g5=GrooveSpec(kind="square", thickness_mm=10, root_gap_mm=5, root_face_mm=0))
    for i, (name, spec) in enumerate(fam.items()):
        for roll in (0.0, 6.0):
            x, z = _profile(spec, roll, y_off_m=0.004, seed=20 + i)
            out.append((f"{name}_{roll}", x, z, spec))
    for side in ("left", "right"):
        for roll in (0.0, 8.0, -8.0):
            spec = GrooveSpec(bevel_side=side)
            x, z = _profile(spec, roll, seed=int(10 + roll))
            out.append((f"{side}_{roll}", x, z, spec))
    spec = GrooveSpec(bevel_side="right", hi_lo_mm=1.5)
    out.append(("hi_lo", *_profile(spec, 6.0), spec))
    spec = GrooveSpec(bevel_side="left")
    out.append(("oblique", *_profile(spec, 8.0, y_off_m=-0.03, angle_deg=5.0), spec))
    d = np.load(REAL_NPZ)
    out.append(("real_median", d["x_mm"].astype(float), d["z_mm"].astype(float), GrooveSpec(bevel_side="unknown")))
    xf = np.linspace(-80, 80, 1200)
    out.append(("flat", xf, 430.0 + 0.14 * xf, None))
    out.append(("not_parallel", xf, np.where(xf < 0, 430.0, 430.0 + 0.3 * xf), None))
    out.append(("few", xf[:10], (430.0 + 0.14 * xf)[:10], None))
    out.append(("nan", np.full(100, np.nan), np.full(100, np.nan), None))
    x, z = _profile(GrooveSpec(bevel_side="left"), 0.0)
    z = np.array(z, dtype=float, copy=True)
    z[np.asarray(x) > 2.0] = np.nan                                     # 오른 판 없음 (한쪽 판만)
    out.append(("one_plate", x, z, GrooveSpec(bevel_side="left")))
    return out


@pytest.mark.parametrize("ds", [1.0, 0.9962, 0.82])
def test_plate_seed_then_the_rest_equals_one_shot(ds):
    """판 성장 (1~2 단계) 은 depth_scale·자세와 무관 — 미리 계산한 씨앗 + 나머지 단계 == 한 번에 (바이트까지), 작업 허용치를 줘도."""
    qcs = (None, {"groove_depth_mm": {"tol": 0.5}, "root_gap_mm": {"tol": 0.3}})
    n_seeded = 0
    for name, x, z, spec in _cases():
        for qc in qcs:
            kw = {"qc_targets": qc}
            one = extract_cross_section_features(x, z, spec, depth_scale=ds, **kw)
            seed = cross_section_plate_seed(x, z, spec, **kw)
            two = extract_cross_section_features(x, z, spec, depth_scale=ds, plate_seed=seed, **kw)
            assert not _diff(one, two), (name, qc, _diff(one, two)[:5])
            if name in ("few", "nan"):
                assert seed is None                                 # 판 성장 전에 끝나는 프로파일은 씨앗 없음 (나중 계산이 싸다)
            else:
                n_seeded += int(seed is not None)
                # 다른 프로파일의 씨앗 (유효 점 수가 다름) 은 쓰지 않는다
                wrong = SF.PlateSeed(n=seed.n + 1, mL=None, mR=None, rng_state=seed.rng_state)
                assert not _diff(one, extract_cross_section_features(x, z, spec, depth_scale=ds, plate_seed=wrong, **kw))
    assert n_seeded >= 40


def test_along_scan_with_plate_seeds_equals_without():
    """스캔 전체 (판 법선 사전값 → 프로파일마다 depth_scale ≠ 1, 피치 −5°): 행마다 씨앗을 준 특징 == 씨앗 없이 (순차·프로세스 병렬 모두)."""
    calib = Calib.lab_initial(eoat_yaw_rad=-np.pi / 2, opt_yaw_branch=-1)
    spec = GrooveSpec(bevel_side="left")
    job = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, -0.12, 0.05], surface_normal=[0, 0, 1], table_z_m=-0.05,
                  specimen_top_z_m=0.05, lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5,
                  profile_fps=20.0, pitch_deg=-5.0, job_id="seed_syn")
    syn = simulate_scan(job, calib, GrooveScene.from_spec(spec, yc=0.002),
                        SynthConfig(n_points=704, seed=7, sigma_z_mm=0.02, dropout_prob=0.01, robot_poll_rate_hz=0.0))
    res = reconstruct_scan(syn.frames, syn.robot, calib, job, fps_nominal=20.0)
    n = np.asarray(job.normal_unit, dtype=float)
    ds = SF.depth_scales(res.T_base_opt, n)
    assert ds is not None and float(np.max(ds)) < 1.0                   # 실제로 depth_scale 이 걸린다
    kw = {"qc_targets": None}
    X = np.where(res.valid, np.asarray(res.x_mm, dtype=float), np.nan)
    Z = np.where(res.valid, np.asarray(res.z_mm, dtype=float), np.nan)
    seeds = plate_seeds_chunk(X, Z, spec, kw)
    seeds[3] = None                                                     # 씨앗이 없는 행은 처음부터
    assert sum(s is not None for s in seeds) >= 0.8 * len(seeds)
    a = extract_seam_features_along_scan(res, spec, keep_features=True, plate_normal_base=n, **kw)
    b = extract_seam_features_along_scan(res, spec, keep_features=True, plate_normal_base=n, plate_seeds=seeds, **kw)
    assert a.n_ok >= 0.8 * a.n_profiles and not _diff(a, b), _diff(a, b)[:5]
    old_min = SF.PARALLEL_MIN_PROFILES
    SF.PARALLEL_MIN_PROFILES = 8
    try:
        c = extract_seam_features_along_scan(res, spec, keep_features=True, plate_normal_base=n, plate_seeds=seeds, workers=2, **kw)
    finally:
        SF.PARALLEL_MIN_PROFILES = old_min
        SF.shutdown_feature_pool()
    assert SF.FEATURE_RUN["mode"] == "pool" and not _diff(a, c), _diff(a, c)[:5]


def test_a_stalled_feature_pool_is_bounded_and_its_chunks_are_computed_here(monkeypatch):
    """특징 풀의 자식들이 살아 있지만 멈춘다 (SIGSTOP — 프로세스 상태로는 알 수 없다): 결과 대기는 한계 (proc_pool.RESULT_TIMEOUT_S, 전체에 한 번)
    뒤 그 풀의 자식을 끝내고 돌려받지 못한 덩어리는 이 프로세스에서 같은 계산 — 결과는 순차와 같고, 멈춘 자식은 남지 않고, 다음에 쓸 때 새 풀.
    예전: 한계 없는 Future.result() 라 서버 안 특징 추출 (작업 프로세스 대체 경로) 이 영원히 기다렸다."""
    import os
    import signal
    import time
    from scan_core import proc_pool as PP
    calib = Calib.lab_initial(eoat_yaw_rad=-np.pi / 2, opt_yaw_branch=-1)
    spec = GrooveSpec(bevel_side="left")
    job = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, -0.12, 0.05], surface_normal=[0, 0, 1], table_z_m=-0.05,
                  specimen_top_z_m=0.05, lead_in_m=0.02, lead_out_m=0.02, speed_mps=0.05, accel_mps2=0.5,
                  profile_fps=20.0, pitch_deg=-5.0, job_id="stall_syn")
    syn = simulate_scan(job, calib, GrooveScene.from_spec(spec, yc=0.002),
                        SynthConfig(n_points=704, seed=7, sigma_z_mm=0.02, dropout_prob=0.01, robot_poll_rate_hz=0.0))
    res = reconstruct_scan(syn.frames, syn.robot, calib, job, fps_nominal=20.0)
    n = np.asarray(job.normal_unit, dtype=float)
    kw = {"qc_targets": None}
    a = extract_seam_features_along_scan(res, spec, keep_features=True, plate_normal_base=n, **kw)
    monkeypatch.setattr(SF, "PARALLEL_MIN_PROFILES", 8)
    monkeypatch.setattr(PP, "RESULT_TIMEOUT_S", 3.0)                     # 시험 시간만 (한계의 값이 아니라 한계 뒤를 본다)
    SF.shutdown_feature_pool()
    pool = SF.worker_pool(2)
    assert [f.result(60.0) for f in [pool.submit(int) for _ in range(2)]] == [0, 0]
    kids = pool.children()
    for k in kids:                                                      # 풀 전체가 살아 있지만 답이 없다
        os.kill(k.pid, signal.SIGSTOP)
    try:
        t0 = time.monotonic()
        c = extract_seam_features_along_scan(res, spec, keep_features=True, plate_normal_base=n, workers=2, **kw)
        dt = time.monotonic() - t0
        assert SF.FEATURE_RUN["mode"] == "fallback" and "TimeoutError" in SF.FEATURE_RUN["error"], SF.FEATURE_RUN
        assert not _diff(a, c), _diff(a, c)[:5]
        assert SF._POOL is not pool and dt < PP.RESULT_TIMEOUT_S + 60.0, dt
        t_end = time.monotonic() + 30.0
        while any(k.exitcode is None and k.is_alive() for k in kids) and time.monotonic() < t_end:
            time.sleep(0.01)
        assert not any(k.is_alive() for k in kids), [k.pid for k in kids if k.is_alive()]
    finally:
        for k in kids:                                                  # (시험이 실패해도 이 시험이 멈춘 자식을 남기지 않는다)
            if k.is_alive():
                os.kill(k.pid, signal.SIGKILL)
        SF.shutdown_feature_pool()


def test_window_medians_equal_np_median_bit_for_bit():
    """이동 중앙값: 창마다 np.median 과 비트까지 같다 (같은 s 여럿 = 멈춘 로봇, 같은 값, ±0.0, 창 폭 0 ~ 전체)."""
    rng = np.random.default_rng(1)
    for trial in range(600):
        n = int(rng.integers(10, 300))
        s = np.sort(rng.uniform(0, 0.4, n))
        if trial % 3 == 0:
            s = np.round(s, 3)
        v = rng.normal(0, 1, n)
        if trial % 4 == 0:
            v = np.round(v, 1)
        if trial % 7 == 0:
            v = np.where(rng.random(n) < 0.3, -0.0, np.where(rng.random(n) < 0.3, 0.0, v))
        w = float(rng.choice([0.0, 0.002, 0.01, 0.02, 1.0]))
        lo = np.searchsorted(s, s - 0.5 * w, side="left")
        hi = np.searchsorted(s, s + 0.5 * w, side="right")
        ref = np.array([np.median(v[a:b]) for a, b in zip(lo, hi)])
        assert ref.tobytes() == SF._window_medians(v, lo, hi).tobytes(), trial
    # 앞으로만 움직이지 않는 창 (호출 규칙 밖) 도 같은 값 (창마다 np.median)
    v = rng.normal(0, 1, 20)
    lo, hi = np.array([5, 0, 3]), np.array([9, 4, 10])
    assert SF._window_medians(v, lo, hi).tobytes() == np.array([np.median(v[a:b]) for a, b in zip(lo, hi)]).tobytes()
    # detrended_noise = 예전 정의 (창마다 np.median 추세를 뺀 강건 σ)
    s = np.sort(rng.uniform(0, 0.38, 900))
    v = 5.0 + 0.002 * np.sin(s * 40) + rng.normal(0, 0.05, 900)
    lo = np.searchsorted(s, s - 0.005, side="left")
    hi = np.searchsorted(s, s + 0.005, side="right")
    r = v - np.array([np.median(v[a:b]) for a, b in zip(lo, hi)])
    assert SF.detrended_noise(s, v, 0.01) == float(1.4826 * np.median(np.abs(r - np.median(r))))


def test_profile_points_base_equals_the_outer_product_form():
    rng = np.random.default_rng(2)
    for n in (0, 1, 7, 1408):
        T = np.eye(4)
        T[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        T[:3, 3] = rng.normal(size=3)
        x = rng.normal(0, 50, n).astype(np.float32)
        z = rng.uniform(200, 400, n).astype(np.float32)
        xs, zs = np.asarray(x, dtype=float) * 1e-3, np.asarray(z, dtype=float) * 1e-3
        ref = T[:3, 3] + np.outer(xs, T[:3, 0]) + np.outer(zs, T[:3, 2])
        got = profile_points_base(T, x, z)
        assert got.shape == ref.shape and got.tobytes() == ref.tobytes()


def test_a_stall_on_a_caller_executor_is_recorded_as_a_timeout(monkeypatch):
    """호출자가 준 실행기 (OwnedPool 이 아닌 것) 가 한계 안에 답하지 않아도 그 덩어리는 여기서 같은 계산이고, 기록된 까닭은 TimeoutError 다
    (예전: 'NoneType: None')."""
    from concurrent.futures import Future
    from scan_core import proc_pool as PP

    class NeverAnswers:                                                 # 받기만 하고 답하지 않는 실행기 (살았지만 멈춘 풀)
        def submit(self, fn, *a, **k):
            return Future()
    rng = np.random.default_rng(3)
    M = SF.PARALLEL_MIN_PROFILES
    X = np.tile(np.linspace(-20.0, 20.0, 200), (M, 1))
    Z = 300.0 + 0.01 * rng.standard_normal((M, 200))
    spec = GrooveSpec(bevel_side="left")
    a = SF._features_chunk(X, Z, spec, {"qc_targets": None})
    monkeypatch.setattr(PP, "RESULT_TIMEOUT_S", 0.2)                     # 시험 시간만
    b = SF._features_for(X, Z, spec, {"qc_targets": None}, 2, executor=NeverAnswers())
    assert SF.FEATURE_RUN["mode"] == "fallback" and SF.FEATURE_RUN["error"].startswith("TimeoutError"), SF.FEATURE_RUN
    assert len(b) == len(a) and not _diff(a, b), _diff(a, b)[:5]
