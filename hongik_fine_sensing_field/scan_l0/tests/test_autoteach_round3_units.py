"""auto-teach 리뷰 라운드 3 단위 테스트 (로봇·센서 없음 또는 가짜 센서만):
S1 (센서 첫 프레임/RESET 지연, SensorSession.grab 의 콜백), S2 (coarse psi 독립 추정), S3 (누적 캡 재기준),
S6 (센서 파라미터 원본은 세션당 한 번, 미복원 경고), S9 (3G 판 평면 p95), F-C9 (명시 --motion-timeout 부족 판정)."""
import math
import os
import time

import numpy as np
import pytest
from conftest import CALIB, CONFIG, JOBS

from scan_core.localize import StaticLocalization
from scan_core.seam_features import CrossSectionFeatures
from scan_core.types import ScanJob
from scan_l0 import config as C
from scan_l0.session import ScanSession
from vc3dz_driver import params as VP
from vc3dz_driver.fake_sensor import FakeSensor, synthetic_frames


# --------------------------------------------------------------------------------------
# S1: sensor start / RESET latency (the first frame is waited for before a leg is sent)
# --------------------------------------------------------------------------------------
def test_s1_fake_sensor_start_and_reset_delays_and_grab_callbacks():
    """FakeSensor(start_delay_s, reset_delay_s) 가 실기 타이밍을 재현하고, SensorSession.grab 의 콜백이 첫 프레임 뒤와 RESET 전에
    한 번씩 불린다 (S1 a)."""
    from scan_l0.session import SensorSession
    with FakeSensor(frames=synthetic_frames(20, 200, z0=320.0), params={"AutoTriggerFPS": 50.0},
                    start_delay_s=0.8, reset_delay_s=0.8) as fs:
        with SensorSession("127.0.0.1", fs.port) as ss:
            events = []
            t0 = time.monotonic()
            recs = ss.grab(10, timeout_s=5.0, on_first_frame=lambda: events.append(("first", time.monotonic() - t0)),
                           pre_stop=lambda: events.append(("pre_stop", time.monotonic() - t0)))
            t_end = time.monotonic() - t0
            assert len(recs) == 10
            assert events[0][0] == "first" and events[0][1] >= 0.75                 # 첫 프레임은 시작 0.8 s 뒤
            assert events[1][0] == "pre_stop" and t_end - events[1][1] >= 0.75      # RESET 응답 0.8 s 대기
            # start_recording + wait_first_frame: 0.8 s 뒤 첫 프레임
            ss.start_recording()
            assert not ss.wait_first_frame(0.3) and ss.wait_first_frame(2.0)
            ss.stop_recording(settle_s=0.0)


def test_s1_start_sensor_recording_refuses_when_no_first_frame():
    from scan_l0.executor import SensorNotStarted, start_sensor_recording
    from scan_l0.session import ScanSession, SensorSession
    with FakeSensor(frames=synthetic_frames(5, 100), start_delay_s=30.0) as fs:
        with SensorSession("127.0.0.1", fs.port) as ss:
            sess = ScanSession(robot=None, sensor=ss)                                 # type: ignore[arg-type]
            t0 = time.monotonic()
            with pytest.raises(SensorNotStarted, match="sensor did not start"):
                start_sensor_recording(sess, first_frame_timeout_s=0.4)
            assert time.monotonic() - t0 < 3.0
            assert not ss.recording and fs.params["NbrLines"] == 0


# --------------------------------------------------------------------------------------
# S6: original parameters captured once per session; warning when the backup already equals the INIT values
# --------------------------------------------------------------------------------------
def test_s6_params_original_captured_once_and_restored():
    from scan_l0.session import SensorSession
    with FakeSensor(frames=synthetic_frames(5, 100)) as fs:
        orig = dict(fs.params)
        with SensorSession("127.0.0.1", fs.port) as ss:
            ss.init(None)
            assert ss.params_original["GainVal"] == orig["GainVal"] and fs.params["GainVal"] == VP.INIT_STANDARD["GainVal"]
            ss.init(dict(VP.INIT_STANDARD, AutoTriggerFPS=20.0))                  # 두 번째 init: 원본을 덮어쓰지 않는다
            assert ss.params_original["GainVal"] == orig["GainVal"] and ss.params_original["DataMode"] == orig["DataMode"]
            assert ss.warnings == []
            rr = ss.restore_backup()
            assert rr["restored"] is True and rr["n"] > 5, rr
        for k in ("DataMode", "GainVal", "ShutterTime", "AutoTriggerFPS", "TransferAllPos_MM"):
            assert fs.params[k] == orig[k], k


def test_s6_warns_when_previous_session_did_not_restore():
    from scan_l0.session import SENSOR_FILE_BACKUP_HINT, SensorSession
    with FakeSensor(frames=synthetic_frames(5, 100), params=dict(VP.INIT_STANDARD)) as fs:   # 직전 세션이 INIT 상태로 남겼다
        with SensorSession("127.0.0.1", fs.port, restore_on_close=True) as ss:
            ss.init(None)
            assert len(ss.warnings) == 1
            w = ss.warnings[0]
            assert "previous session did not restore" in w and "restore --file" in w and SENSOR_FILE_BACKUP_HINT in w
        assert ss.restore_result is not None                                          # restore_on_close


# --------------------------------------------------------------------------------------
# S2 → 4d P4: sensor x scale judged directly (same lateral move seen by the sensor and by the kinematics)
# --------------------------------------------------------------------------------------
def _site_model(x_scale: float, lat_pre_true: float = 0.012, resid_true: float = 0.0003):
    """도착 지점의 보정 전 참 오프셋 lat_pre_true, 센서 x 스케일 x_scale: 센서 = x_scale × 참값, 운동학 = 보정이 옮긴 양 (참값 − 잔차)
    + 측정 잔차 (x_scale × 잔차) — 보정 루프는 '측정 0' 으로 수렴한다."""
    return x_scale * lat_pre_true, (lat_pre_true - resid_true) + x_scale * resid_true


def test_scale_reading_normal_scale_and_flip():
    from scan_l0.autoteach import scale_reading
    eps_max = 0.5 / 10.77                                                     # 개구 tol / 공칭 (4.64 %)
    s, k_ = _site_model(1.0)
    r = scale_reading(s, k_, 443.0, k=3.0, eps_max=eps_max)
    assert r["testable"] and not r["exceeds"] and abs(r["eps"]) < 1e-9
    s, k_ = _site_model(1.03)                                                 # 3 %: 허용 안
    r = scale_reading(s, k_, 443.0, k=3.0, eps_max=eps_max)
    assert r["testable"] and not r["exceeds"] and r["eps"] == pytest.approx(0.03, abs=0.002)
    for scale in (1.10, -1.0):                                                # 10 % 와 X 방향 뒤집힘 (−200 %) 은 넘는다
        s, k_ = _site_model(scale)
        r = scale_reading(s, k_, 443.0, k=3.0, eps_max=eps_max)
        assert r["testable"] and r["exceeds"], (scale, r)
    # 오프셋이 작으면 (1 mm) 그 스케일 오차를 볼 수 없다 → 판정에 쓰지 않는다
    s, k_ = _site_model(1.10, lat_pre_true=0.001)
    r = scale_reading(s, k_, 443.0, k=3.0, eps_max=eps_max)
    assert not r["testable"] and not r["exceeds"]
    r = scale_reading(float("nan"), 0.01, 443.0, k=3.0, eps_max=eps_max)
    assert math.isnan(r["eps"]) and not r["testable"]


def test_s2_site_pre_laterals_sensor_and_kinematic():
    from scan_l0.autoteach import site_pre_laterals

    def loc(axis_hit, lateral, e=(1.0, 0.0, 0.0)):
        f = CrossSectionFeatures(ok=True, n_valid=1000, groove_found=True)
        lo = StaticLocalization(features=f, T_base_opt=np.eye(4), opt_origin_base=np.asarray(axis_hit) + [0, 0, 0.4],
                                groove_found=True)
        lo.axis_hit_base = np.asarray(axis_hit, dtype=float)
        lo.lateral_axis_base = np.asarray(e, dtype=float)
        lo.lateral_offset_m = lateral
        return lo

    A0 = np.array([0.6, 0.19, 0.05])
    arr = loc(A0, 0.0120)
    fin = loc(A0 + [0.0115, 0.0, 0.0], 0.0003)                            # 보정이 광축∩판 점을 e_lat 로 11.5 mm 옮겼고 잔차 0.3 mm
    s, k = site_pre_laterals(arr, arr, fin)
    assert s == pytest.approx(0.0120) and k == pytest.approx(0.0118)
    # 탐색 스텝(+75 mm) 뒤에 그루브를 찾은 경우: 센서 값 = 찾은 국소화의 lateral + 탐색 스텝
    pre = loc(A0 + [0.075, 0.0, 0.0], -0.004)
    fin2 = loc(A0 + [0.0712, 0.0, 0.0], 0.0002)
    s, k = site_pre_laterals(arr, pre, fin2)
    assert s == pytest.approx(0.071) and k == pytest.approx(0.0714)


# --------------------------------------------------------------------------------------
# S3: cumulative cap re-based after a successful search step / retreat
# --------------------------------------------------------------------------------------
def _bare_autoteach(tmp_path, job_name="groove_10T_1G.yaml"):
    from scan_l0.autoteach import AutoTeach
    settings = C.load_settings(CONFIG)
    calib = C.load_calib(CALIB)
    job0, doc = C.load_job(os.path.join(JOBS, job_name))
    return AutoTeach(settings, job0, doc, calib, out_dir=str(tmp_path), log=lambda s: None,
                     session=ScanSession(robot=None, sensor=None, owned=False)), calib


def test_corrections_have_no_step_or_cumulative_caps(tmp_path):
    """2026-10-07: 보정 상한 (스텝 40 mm/30 mm/7°, 누적 120 mm/20°) 은 없다 — 보정은 잰 만큼, 안전은 가드·포락선, 수렴은 반복 예산."""
    at, _calib = _bare_autoteach(tmp_path)
    for name in ("_caps", "_cumulative_check", "_arrive", "_rebase_site", "_site_ref"):
        assert not hasattr(at, name)
    assert not hasattr(at.at, "caps") and at.at.max_corrections >= 1


# --------------------------------------------------------------------------------------
# S9 → 4d P5: 3G plate plane level = the highest extractor plate inlier (along n0 for the envelope plane, along the measured
# normal for the measured plane; here they coincide)
# --------------------------------------------------------------------------------------
def test_s9_3g_plane_uses_the_highest_plate_inlier_level(tmp_path):
    from scan_l0.autoteach import make_coarse_job
    at, calib = _bare_autoteach(tmp_path, "groove_10T_3G.yaml")
    job0 = at.job0
    at.orientation = job0.resolved_orientation()
    at.up = job0.up_unit
    at.job_c = make_coarse_job(job0, at._sv(), depth_m=at._depth_spec_m())
    at.u0, at.n0 = at.job_c.unit_dir, at.job_c.normal_unit
    at.job_ref = ScanJob.from_dict(C.job_to_dict(at.job_c))
    n0, start0 = at.n0, at.job_ref.start.copy()
    lat = np.cross(n0, at.u0)
    d = np.linspace(0.0, 0.010, 41)                                        # 판 점이 n0 방향으로 0..10 mm 퍼짐 (거친 판·기울기)
    pts = np.array([start0 + n0 * di + lat * (0.002 * i - 0.04) for i, di in enumerate(d)])
    f = CrossSectionFeatures(ok=True, n_valid=1000, groove_found=True, plate_left={"tilt_deg": 0.0}, plate_right={"tilt_deg": 0.0})
    loc = StaticLocalization(features=f, T_base_opt=np.eye(4), opt_origin_base=start0 + n0 * 0.40, groove_found=True)
    loc.plate_points_base, loc.plate_normal_base, loc.plate_dir_base = pts, n0.copy(), lat
    loc.standoff_m = 0.40
    loc.diag = {"n_valid": 1000}
    at._raise_plate_top(loc)
    top, med = float(np.max(d)), float(np.median(d))
    assert top > med + 0.004
    assert float(n0 @ (at.job_ref.start - start0)) == pytest.approx(top, abs=1e-9)             # 이동량 = 가장 높은 판 인라이어 (중앙값 아님)
    pp, pn = at.job_ref.plate_plane()
    assert float(n0 @ (pp - start0)) == pytest.approx(top, abs=1e-9) and float(pn @ n0) == pytest.approx(1.0)


def test_plate_edge_outliers_only_at_the_ends():
    """판 끝 모서리에 걸친 프로파일의 루트 점 (관측 범위 양끝 20 mm 안, 직선에서 문턱보다 멀다) 만 납품에서 뺀다 — 가운데의 벗어난 점은
    그대로 (직선이 아닌 실제 형상을 숨기지 않는다). 고정 문턱 2 mm (thr_m, 라운드 3 규칙) 와 적응 문턱 (내부 잔차가 0 → k × 하한 (센서 양자화;
    여기서는 0.1 mm 를 준다) = 0.3 mm: 1.5 mm 끝 점도 아웃라이어)."""
    from scan_core.localize import PassFit
    from scan_l0.autoteach import plate_edge_outliers
    s = np.linspace(0.0, 0.3, 61)
    P = np.column_stack([np.full_like(s, 0.6), s, np.full_like(s, 0.04)])
    P[0, 0] += 0.008                                                       # 첫 프로파일: 판 끝, 8 mm
    P[30, 0] += 0.004                                                      # 가운데: 4 mm (판 끝 아님)
    P[-1, 0] += 0.0015                                                     # 끝이지만 1.5 mm (< 2 mm)
    P[5] = np.nan
    fit = PassFit(c=np.array([0.6, 0.0, 0.04]), u=np.array([0.0, 1.0, 0.0]), n=np.array([0.0, 0.0, 1.0]), s_a=0.0, s_b=0.3)
    m = plate_edge_outliers(P, fit, thr_m=0.002, edge_m=0.020)
    assert list(np.flatnonzero(m)) == [0]
    info = {}
    m = plate_edge_outliers(P, fit, edge_m=0.020, floor_m=1.0e-4, diag=info)
    assert list(np.flatnonzero(m)) == [0, 60] and info["rule"] == "adaptive" and info["thr_m"] == pytest.approx(3.0e-4)   # k · 하한
    assert not plate_edge_outliers(P, PassFit(c=np.full(3, np.nan), u=np.full(3, np.nan), n=np.full(3, np.nan))).any()


# --------------------------------------------------------------------------------------
# F-C9: explicit --motion-timeout shorter than 2*dist/v_eff + 10 s
# --------------------------------------------------------------------------------------
def test_fc9_motion_timeout_shortfall():
    from scan_l0.executor import motion_timeout_shortfall
    assert motion_timeout_shortfall(None, 0.46, 0.002) is None                          # 유도 타임아웃: 검사 없음
    why = motion_timeout_shortfall(120.0, 0.46, 0.002, "lead_out")                      # 2·460/2 + 10 = 470 s > 120 s
    assert why is not None and "--motion-timeout" in why and "lead_out" in why and "470.0 s" in why
    assert motion_timeout_shortfall(480.0, 0.46, 0.002) is None
