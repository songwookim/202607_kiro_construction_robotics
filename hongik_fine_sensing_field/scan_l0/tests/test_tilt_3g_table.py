"""3G + 테이블 높이 (임시 입력, 2026-10-05): 툴 몸체 부착과 입사 각도 자동 선택 (g3_04 실기 배치)."""

import numpy as np
import pytest

from scan_core.types import ScanJob
from scan_core.localize import EndParams
from scan_l0 import config as C
from scan_l0.autoteach import choose_3g_end_postures, make_coarse_job

from conftest import CALIB, CONFIG

# 2026-10-05 다시 교시한 3G 시편 (base, m): 위 점 S, 아래 점 E, 판 법선 (실측 평면)
S_UP = [1.2500, 0.0879, 0.1666]
E_LO = [1.2542, 0.0630, -0.1553]
N = [-0.9226, -0.3854, 0.0179]


@pytest.fixture
def settings():
    return C.load_settings(CONFIG)


@pytest.fixture
def calib_body(settings):
    return C.attach_tool_body(C.load_calib(CALIB), settings)


def _job(table, tilt=20.0):
    job, doc = C.load_job(C.JOBS_DIR / "groove_10T_3G.yaml")
    d = C.job_to_dict(job)
    d.update(start=S_UP, end=E_LO, surface_normal=N, table_z_m=table, incidence_tilt_deg=tilt, specimen_top_z_m=0.267,
             torch_lateral_m=tuple(d["torch_lateral_m"]), vertical_direction="down")
    return ScanJob.from_dict(d), C.specimen_from_doc(doc)


def test_attach_tool_body_from_settings(settings, tmp_path):
    c = C.attach_tool_body(C.load_calib(CALIB), settings)
    assert c.body_points_eoat is not None and c.body_points_eoat.shape[0] >= 50 and "hull" in c.body_source
    assert C.attach_tool_body(c, settings) is c                                       # 이미 붙어 있으면 그대로
    assert c.to_dict()["body_points_n"] == c.body_points_eoat.shape[0]
    bare = C.Settings({**settings.params, "safety": {k: v for k, v in settings.safety.items() if k != "tool_body_mesh"}})
    assert C.attach_tool_body(C.load_calib(CALIB), bare).body_points_eoat is None
    bad = C.Settings({**settings.params, "safety": {**settings.safety, "tool_body_mesh": str(tmp_path / "missing.stl")}})
    with pytest.raises(ValueError, match="tool_body_mesh"):
        C.attach_tool_body(C.load_calib(CALIB), bad)


def _qf(pred=1.0):
    return lambda jp: {"pred": pred, "pred_why": "stub", "torch_above": None, "phi_deg": 0.0, "incidence_deg": 0.0}


def _view(z_valid_mm=(233.1, 544.0)):
    from scan_l0.sensor_view import sensor_view
    return sensor_view(z_valid_mm, pose_tol_m=0.002)


def _end(job, calib_body, at, spec, view=None, **kw):
    from scan_core.types import INCIDENCE_TILT_MAX_DEG
    return choose_3g_end_postures(job, calib_body, at, view or _view(), spec, quality_fn=kw.pop("quality_fn", _qf()),
                                  t_max_deg=min(INCIDENCE_TILT_MAX_DEG, job.max_laser_tilt_deg),
                                  ghost_burst_frames=kw.pop("ghost_burst_frames", C.load_settings(CONFIG).ghost_burst_frames), **kw)


def test_end_postures_keep_the_whole_tool_above_the_measured_support(settings, calib_body):
    """잰 지지면 −0.17 m (아래 착점 −0.155 보다 15 mm 아래; 연구실 3G 배치): 모든 후보는 E 드웰에서 툴 몸체 전체가 지지면 + 여유 위이고 측정 전
    3G 바닥 (광학 원점·토치 팁 ≥ 아래 착점) 도 지킨다 — 작은 내려다봄은 몸체가 지지면 아래라 빠진다. 아래 목표는 지지면 − overscan 까지."""
    at = settings.auto_teach_config()
    job, spec = _job(-0.17)
    ch = _end(job, calib_body, at, spec)
    assert ch["feasible"] and ch["ranked"]
    assert all(r["margin_E_m"] >= 0.0 and r["floor_margin_E_m"] >= 0.0 for r in ch["ranked"])
    rows = [r for r in ch["rows"] if "skip" not in r]
    assert any(r["tilt_deg"] <= 20.0 and not r["ok_E"] for r in rows)                 # 작은 내려다봄: 몸체가 지지면 아래
    from scan_l0.autoteach import overscan_need_m
    ov = overscan_need_m(job, EndParams.ghost_burst_frames)[0]
    assert ch["target_lower_hit"][2] == pytest.approx(-0.17 - ov, abs=1e-9)                  # 아래 목표 착점 = 지지면 − 오버스캔
    sc = [r["score"] for r in ch["ranked"]]
    assert sc == sorted(sc, reverse=True) or all(abs(a - b) < 0.005 for a, b in zip(sc, sorted(sc, reverse=True)))


def test_end_postures_rank_by_predicted_quality_times_reach(settings, calib_body):
    """점수 = 예상 품질 × 아래 끝 도달 비율: 품질 모형이 어떤 자세를 더 좋게 보면 그 자세가 1 위 (고정 각도 규칙 없음)."""
    at = settings.auto_teach_config()
    job, spec = _job(-0.32)
    base = _end(job, calib_body, at, spec)
    assert base["ranked"] and base["ranked"][0]["coverage"] == pytest.approx(1.0)
    assert all(r["plate_margin_pass1_m"] >= 0.0 and r["plate_margin_E_m"] >= 0.0 for r in base["ranked"])  # 팁·몸체 ↔ 판면 ≥ 한계
    target = [r for r in base["ranked"] if r["coverage"] == pytest.approx(1.0)][-1]   # 끝까지 닿는 가장 낮던 후보를 품질 모형이 선호하면
    pick = (target["tilt_deg"], target["sign"])

    def qf(jp):
        good = abs(float(jp.incidence_tilt_deg) - pick[0]) < 1e-9 and int(jp.progression_sign) * 0 == 0
        return {"pred": 1.0 if good else 0.5, "pred_why": "stub", "torch_above": None, "phi_deg": 0.0, "incidence_deg": 0.0}
    ch = _end(job, calib_body, at, spec, quality_fn=qf)
    assert ch["ranked"][0]["tilt_deg"] == pytest.approx(pick[0])


def test_end_postures_none_when_the_support_is_at_the_lower_hit(settings, calib_body):
    at = settings.auto_teach_config()
    job, spec = _job(0.0)                                                              # 지지면이 아래 착점 (−0.155) 보다 위
    assert not _end(job, calib_body, at, spec)["feasible"]
    with pytest.raises(ValueError, match="table_z_m"):
        _end(_job(None)[0], calib_body, at, spec)


def test_end_postures_limited_by_max_laser_tilt(settings, calib_body):
    at = settings.auto_teach_config()
    job, spec = _job(-0.12)
    j30 = ScanJob.from_dict({**C.job_to_dict(job), "torch_lateral_m": tuple(job.torch_lateral_m), "max_laser_tilt_deg": 30.0})
    ch = _end(j30, calib_body, at, spec)
    p0 = abs(float(j30.pitch_deg))
    assert all("max_laser_tilt_deg" in r["skip"] for r in ch["rows"] if r["tilt_deg"] > 30.0 - p0 + 1e-9)


def test_3g_coarse_arc_radius_is_the_derived_coarse_range_for_every_tilt(settings):
    """거친 작업의 호 반지름 (착점까지 광축 거리) = 센서 시야 모형의 거친 거리 (기울기마다; 대략 착점 오차를 모든 방향으로 가장 크게
    견딘다). 드웰 보정 목표 = 그 × cos(입사) 판 법선 거리 (같은 작업에서 나온다 — g3_07 의 보정 하강 없음)."""
    import math
    v = _view()
    for tilt in (0.0, 10.0, 35.0):
        job, spec = _job(None, tilt=tilt)
        d, o = spec.groove_depth_mm * 1e-3, spec.top_opening_mm * 1e-3
        jc = make_coarse_job(job, v, depth_m=d, opening_m=o)
        assert jc.standoff_m == pytest.approx(v.coarse_range_m(d, tilt, o))
        assert jc.standoff_m + d / math.cos(math.radians(tilt)) < 0.544 - 0.002              # 루트가 창 안
        assert v.coarse_plate_m(d, tilt, o) == pytest.approx(jc.standoff_m * math.cos(math.radians(tilt)))
    j1 = ScanJob(start=[0.6, -0.2, 0.05], end=[0.6, 0.3, 0.05], surface_normal=[0, 0, 1], specimen_top_z_m=0.05)
    assert make_coarse_job(j1, v, depth_m=0.01, opening_m=0.0108).standoff_m == pytest.approx(v.coarse_range_m(0.01, 0.0, 0.0108))


def test_cap_lower_target_at_the_table():
    """g3_10: 패스 1 아래 목표가 지지면보다 ~5 cm 아래 → 레이저가 판 앞 테이블만 비춘다. 목표 착점은 지지면 − overscan (20 mm) 까지."""
    import numpy as np
    from scan_l0.autoteach import PassPlan, SeamLine, cap_lower_target
    up = np.array([0.0, 0.0, 1.0])
    line = SeamLine(c=np.array([1.25, 0.09, 0.20]), u=np.array([0.0, 0.0, -1.0]), n=np.array([-1.0, 0.0, 0.0]), s_S=0.03, s_E=0.355)
    plan = PassPlan(direction="E->S", s_S=0.0, s_E=0.355 + 0.057, ov_S=0.02, ov_E=0.02)     # 착점 최저 0.20 − 0.432 = −0.232
    msg = cap_lower_target(plan, line, up, -0.170, below_m=0.020)
    assert msg and "capped" in msg
    assert 0.20 - (plan.s_E + plan.ov_E) == pytest.approx(-0.190, abs=1e-9)
    plan2 = PassPlan(direction="E->S", s_S=0.0, s_E=0.355 + 0.010, ov_S=0.02, ov_E=0.005)   # 이미 위: 그대로
    assert cap_lower_target(plan2, line, up, -0.170, below_m=0.020) is None and plan2.s_E == pytest.approx(0.365)
    plan3 = PassPlan(direction="E->S", s_S=0.0, s_E=0.412, ov_S=0.02, ov_E=0.02)
    msg3 = cap_lower_target(plan3, line, up, -0.120, below_m=0.020, measured=False)          # 상한뿐인 지지면 (보인 판 최저점): 자르지 않음
    assert "not capped" in msg3 and plan3.s_E == pytest.approx(0.412) and plan3.ov_E == pytest.approx(0.02)
    assert cap_lower_target(plan3, line, up, None, below_m=0.020) is None
    # 잰 지지면은 언제나 자른다 — 예전 '대략 점 + 10 mm' 예외는 3.3 mm 오버스캔에서 g3_17 같은 배치 (지지면 바로 위 아래 점) 를 남겼다
    plan4 = PassPlan(direction="E->S", s_S=0.0, s_E=0.355 + 0.057, ov_S=0.0033, ov_E=0.0033)
    msg4 = cap_lower_target(plan4, line, up, 0.20 - 0.360, below_m=0.0033)                    # 지지면이 아래 점 5 mm 아래
    assert msg4 and "capped" in msg4 and 0.20 - (plan4.s_E + plan4.ov_E) == pytest.approx(-0.160 - 0.0033, abs=1e-9)


def test_end_pass_pitch_is_trimmed_when_the_reference_normal_puts_the_laser_tilt_just_over_the_limit(settings, calib_body):
    """g3_26 (2026-10-06): 끝 패스 입사 35° + 피치 5° = 40° 가 포락선 기준 법선 (job_ref) 으로 40.23° → 패스 시작 거부. 이제 넘친 만큼 피치를
    줄인다 (4d P10: 산포 상한 2° 없음 — 줄인 작업이 다시 검사를 지날 때만 바꾸고, 못 지나면 그대로 두어 포락선이 거부)."""
    import math
    from types import SimpleNamespace
    from scan_core.frames import unit
    from scan_core.path_gen import sensor_frames
    from scan_l0.autoteach import AutoTeach
    from scipy.spatial.transform import Rotation
    job, _spec = _job(-0.30, tilt=35.0)
    job = ScanJob.from_dict({**C.job_to_dict(job), "torch_lateral_m": tuple(job.torch_lateral_m), "pitch_deg": 5.0, "max_laser_tilt_deg": 40.0})
    fr = sensor_frames(job, calib_body)

    def tilt_vs(j, n):
        Z = sensor_frames(j, calib_body).T_base_opt(0.0)[:3, 2]
        return math.degrees(math.acos(float(Z @ (-unit(n)))))
    base = tilt_vs(job, fr.n)
    # the laser tilt is the calibrated optical axis's: the commanded 35 + 5 deg on the tool plus the calibration's own rotation about the
    # laser line (w_x -1.43 deg, eoat_lab_01 2026-10-12; job.validate bounds the commanded sum) — in this posture 38.6 deg, so the 0.3 deg
    # reference normals stay inside and the 3 deg ones go over
    assert abs(base - 40.0) < 2.0, base
    lim = float(job.max_laser_tilt_deg)
    trimmed = 0
    for ang in (0.3, -0.3, 3.0, -3.0):
        for axis in (fr.u, np.cross(fr.n, fr.u)):
            n2 = Rotation.from_rotvec(unit(axis) * math.radians(ang)).as_matrix() @ fr.n
            ref = ScanJob.from_dict({**C.job_to_dict(job), "torch_lateral_m": tuple(job.torch_lateral_m), "surface_normal": n2.tolist()})
            stub = SimpleNamespace(job_ref=ref, calib=calib_body, _decide=lambda *a, **k: None)
            out = AutoTeach._fit_laser_tilt(stub, job, 1)
            t_in = tilt_vs(job, sensor_frames(ref, calib_body).n)
            if t_in <= lim:
                assert out is job, (ang, t_in)                              # 한계 안: 그대로
            elif out is not job:
                trimmed += 1
                assert out.pitch_deg < job.pitch_deg and out.incidence_tilt_deg == job.incidence_tilt_deg
                assert tilt_vs(out, sensor_frames(ref, calib_body).n) <= lim, (ang, out.pitch_deg)
            else:
                assert t_in - lim > abs(job.pitch_deg) - 0.05, (ang, t_in)   # 피치를 다 줄여도 못 맞추면 그대로 (포락선이 거부)
    assert trimmed >= 1
