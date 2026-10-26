"""범용 시작의 순수 기하 (scan_l0.acquisition): 착점 두 개 + 자세 + 중력만으로 — 시편이 로봇의 어느 쪽에 있든 (0/90/180/270°), 로봇이
옆으로 달려 중력이 base y 여도 같은 결과 (2026-10-07 사용자: 범용, 교시 자세·템플릿 법선·테이블 높이 미사용, 하드 파라미터 없이)."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scan_l0.acquisition import (cell_boxes, conservative_floor_height, initial_frame, measure_support, plan_support_probe,
                                 rank_reach_candidates)

S3, E3, N3 = np.array([0.50, 0.70, 0.30]), np.array([0.50, 0.70, 0.68]), np.array([0.0, -1.0, 0.0])     # 연구실 3G (판 y 0.70)
S1, E1 = np.array([0.60, -0.19, 0.05]), np.array([0.60, 0.19, 0.05])                                    # 연구실 1G
G = np.array([0.0, 0.0, -1.0])


def _rot(deg, axis=(0.0, 0.0, 1.0)):
    return Rotation.from_rotvec(np.asarray(axis) * math.radians(deg)).as_matrix()


@pytest.mark.parametrize("az", [0.0, 90.0, 180.0, 270.0, 37.0])
def test_3g_frame_follows_the_specimen_around_the_robot(az):
    """판 법선 추정 = 착점에서 로봇을 향한 수평 방향 (첫 드웰이 진짜 법선을 잰다): 수평·심에 직교·로봇 쪽. 판이 로봇을 정면으로 보면 같고,
    연구실 3G 처럼 비껴 있으면 그만큼 (~35°) 다르다 — 그래서 첫 드웰 뒤 실측 법선으로 획득 자세를 다시 계획한다."""
    R = _rot(az)
    fr = initial_frame(R @ S3, R @ E3, "3G", G)
    assert abs(float(fr.n0 @ fr.up)) < 1e-12 and abs(float(fr.n0 @ fr.u0)) < 1e-12
    M = 0.5 * (R @ S3 + R @ E3)
    assert float(fr.n0 @ -M) > 0.0                             # 로봇 쪽
    assert math.degrees(math.acos(float(fr.n0 @ (R @ N3)))) < 45.0
    fr_sq = initial_frame(R @ np.array([0.0, 0.70, 0.30]), R @ np.array([0.0, 0.70, 0.68]), "3G", G)   # 정면
    assert np.allclose(fr_sq.n0, R @ N3, atol=1e-9)
    assert np.allclose(fr.u0, [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(fr.lower, R @ S3)
    assert conservative_floor_height(fr, thickness_m=0.01) == pytest.approx(0.30)


@pytest.mark.parametrize("az", [0.0, 90.0, 180.0, 270.0])
def test_frames_on_a_sideways_robot_gravity_along_base_y(az):
    """KIRO 처럼 로봇이 옆으로 달려 중력이 base 축이 아니어도: 세계 전체를 base 기준으로 돌린 것과 같다."""
    Rm = _rot(90.0, (1.0, 0.0, 0.0))
    g = Rm @ G
    R = Rm @ _rot(az)
    fr3 = initial_frame(R @ np.array([0.0, 0.70, 0.30]), R @ np.array([0.0, 0.70, 0.68]), "3G", g)
    assert np.allclose(fr3.n0, R @ N3, atol=1e-9) and np.allclose(fr3.up, -g)
    fr1 = initial_frame(R @ S1, R @ E1, "1G", g)
    assert np.allclose(fr1.n0, -g) and abs(float(fr1.u0 @ fr1.up)) < 1e-12


def test_position_and_hit_points_must_agree():
    with pytest.raises(ValueError, match="1G"):
        initial_frame(S3, E3, "1G", G)                         # 수직 심을 1G 로
    with pytest.raises(ValueError, match="3G"):
        initial_frame(S1, E1, "3G", G)                         # 수평 심을 3G 로
    with pytest.raises(ValueError, match="apart"):
        initial_frame(S1, S1 + 0.0002, "1G", G)


def test_1g_floor_is_the_plate_bottom_below_the_hit():
    fr = initial_frame(S1, E1, "1G", G)
    assert conservative_floor_height(fr, thickness_m=0.010, backing_m=0.006) == pytest.approx(0.05 - 0.010 - 0.006)


def _corners(b):
    T, h = np.asarray(b["T"]), 0.5 * np.asarray(b["size"])
    return np.array([T[:3, :3] @ (h * (np.array(s) * 2 - 1)) + T[:3, 3] for s in np.ndindex(2, 2, 2)])


@pytest.mark.parametrize("az", [0.0, 90.0, 180.0, 270.0])
def test_cell_boxes_cover_the_specimen_and_the_floor_and_cut_on_the_robot_side(az):
    R = _rot(az)
    fr = initial_frame(R @ S3, R @ E3, "3G", G)
    kw = dict(length_m=0.38, plate_width_m=0.30, thickness_m=0.02, reach_m=0.50, slab_m=0.05)
    boxes = {b["id"]: b for b in cell_boxes(fr, 0.285, **kw)}
    assert set(boxes) == {"specimen", "floor_estimate"}
    f = boxes["floor_estimate"]
    assert float(np.asarray(f["T"])[2, 3] + 0.5 * f["size"][2]) == pytest.approx(0.285)         # 윗면 = 바닥 추정
    c = np.asarray(boxes["specimen"]["T"])[:3, 3]
    assert float((c - R @ S3) @ (R @ N3)) < 0                                                    # 판 면 뒤쪽
    M = 0.5 * (R @ S3 + R @ E3)
    a = M[:2] / np.linalg.norm(M[:2])                                                           # 로봇 → 시편 (수평)
    near = lambda b: float(np.min(_corners(b)[:, :2] @ a))
    full = near(f)
    cut = near({b["id"]: b for b in cell_boxes(fr, 0.285, near_clip_m=0.10, **kw)}["floor_estimate"])
    assert cut == pytest.approx(float(M[:2] @ a) - 0.10, abs=1e-9) and cut > full                # 로봇 쪽만 깎인다
    # 4d P10: 시편 상자 길이 = 착점 거리 + 2 × 모자란 설계 길이 (착점이 한쪽 끝 바로 안쪽이어도 반대 끝을 덮는다), 판 면 = 착점 + n·face
    sb = {b["id"]: b for b in cell_boxes(fr, 0.285, face_offset_m=0.010, **kw)}["specimen"]
    L_hits = float(np.linalg.norm(R @ E3 - R @ S3))
    assert sb["size"][0] == pytest.approx(L_hits + 2.0 * max(0.0, 0.38 - L_hits)) and sb["size"][2] == pytest.approx(0.030)
    face = float(((np.asarray(sb["T"])[:3, 3] - M) @ fr.n0)) + 0.5 * sb["size"][2]
    assert face == pytest.approx(0.010, abs=1e-9)


def _probe_points(world, T):
    x, z = world.render(T)
    ok = np.isfinite(z)
    return (T[:3, :3] @ np.vstack([x[ok], 0.0 * x[ok], z[ok]]) * 1e-3 + T[:3, 3:4]).T


def _probe_plan(fr, margin=0.02):
    from groove_world import JOB_YAML_3G
    from conftest import CALIB
    from scan_core.path_gen import lowest_below_origin
    from scan_core.validity import Z_VALID_MM_DEFAULT, dx_linear_coeffs
    from scan_l0 import config as C
    calib = C.load_calib(CALIB)
    job = C.load_job(JOB_YAML_3G)[0]
    inv = np.linalg.inv(calib.T_flange_opt)
    lb = lambda T: lowest_below_origin(T @ inv, calib, job)
    return calib, lb, plan_support_probe(fr, lb, floor_h=fr.height(fr.lower), margin_m=margin, z_valid_mm=Z_VALID_MM_DEFAULT,
                                         dx_coeffs=dx_linear_coeffs(), lateral_m=0.0108)


def test_support_probe_pose_is_computed_from_the_tool_and_sensor_models():
    """고정 각도 없이: 툴 (키포인트) 의 가장 낮은 점이 아래 착점 + 여유 위, 레이저 면은 세로 (판 법선 추정·up), 내려다본다 — 그중
    착점 아래 가장 깊이까지 보이는 것. 여유가 크면 원점도 그만큼 높다."""
    fr = initial_frame(S3, E3, "3G", G)
    _calib, lb, c = _probe_plan(fr)
    best = c[0]
    T = best["T_opt"]
    assert best["depth_m"] >= 0.25
    assert fr.height(T[:3, 3]) - lb(T) == pytest.approx(0.30 + 0.02, abs=1e-6)                 # 툴 가장 낮은 점 = 아래 착점 + 여유
    assert float(T[:3, 2] @ fr.up) < -0.2                                                       # 내려다봄
    assert abs(float(T[:3, 1] @ fr.up)) < 1e-9 and abs(float(T[:3, 1] @ fr.n0)) < 1e-9           # 레이저 면 = n0·up 세로 면
    _c2, _lb2, c2 = _probe_plan(fr, margin=0.05)
    hi = next(r for r in c2 if r["look_down_deg"] == best["look_down_deg"] and r["flip"] == best["flip"]
              and r["front_m"] == best["front_m"])
    assert fr.height(hi["T_opt"][:3, 3]) == pytest.approx(fr.height(T[:3, 3]) + 0.03, abs=1e-9)


@pytest.mark.parametrize("drop,yaw,jig", [(0.015, 0.0, False), (0.05, 0.0, False), (0.15, 0.0, False), (0.30, 0.0, False),
                                          (0.015, 35.0, False), (0.20, -35.0, False), (0.10, 0.0, True)])
def test_support_probe_measures_the_table_or_jig_without_knowing_the_plate_normal(drop, yaw, jig):
    """가짜 세계의 3G 지지면 (아래 착점에서 drop 아래), 판이 추정 법선 (로봇 쪽 수평) 에서 yaw 만큼 돌아 있어도 계산된 probe 한 번으로
    지지면 높이를 0.5 mm 안에 잰다. jig: 판 앞 테이블 위 40 mm 높이 블록 → 그 윗면이 지지면 (평면 아님)."""
    from groove_world import make_world_3g
    fr = initial_frame(S3 + [0.003, 0.0, -0.004], E3 + [0.0, 0.002, 0.003], "3G", G)
    calib, _lb, c = _probe_plan(fr)
    T = c[0]["T_opt"]
    w = make_world_3g(calib, pose_fn=lambda: np.eye(4), support_height_m=0.30 - drop,
                      n_true=tuple(_rot(yaw) @ N3))
    P = _probe_points(w, T)
    if jig:
        r_ = P @ unit_h(T[:3, 2])
        r_p = float((S3 @ unit_h(T[:3, 2])))
        blk = (r_ < r_p - 0.03) & (r_ > r_p - 0.08)                    # 판 앞 30–80 mm 에 지그 블록 (윗면 = 테이블 + 40 mm)
        P = P.copy()
        P[blk, 2] = 0.30 - drop + 0.040
    r = measure_support(P, fr.up, T[:3, 2])
    want = 0.30 - drop + (0.040 if jig else 0.0)
    assert r.height == pytest.approx(want, abs=5e-4), r
    assert r.flat is (not jig)


def unit_h(v):
    v = np.asarray(v, dtype=float).copy()
    v[2] = 0.0
    return v / np.linalg.norm(v)


def test_support_reading_bound_and_refusals():
    fr = initial_frame(S3, E3, "3G", G)
    toward = -N3
    d = np.linspace(0.01, 0.12, 200)
    plate_only = np.column_stack([np.full_like(d, 0.5), np.full_like(d, 0.70), 0.30 - d])
    r = measure_support(plate_only, fr.up, toward)
    assert r.height is None and r.n_plate == d.size and "no surface in front" in r.note
    assert r.foot_height == pytest.approx(0.30 - 0.12, abs=1e-9)                                 # 지지면은 그보다 아래 (상한)
    table_only = np.column_stack([np.full_like(d, 0.5), 0.70 - d, np.full_like(d, 0.21)])
    r = measure_support(table_only, fr.up, toward)
    assert r.height == pytest.approx(0.21, abs=1e-9) and "only the support" in r.note and r.flat


def test_reach_candidates_supplied_order_is_least_joint_motion_with_singularity_distance_only_as_a_tie_break():
    """B2 (KIRO field 2026-10-14): no relative-condition groups - the supplied order is sum |dq| from the current joints, min_sigma only
    breaks ties; candidates that do not follow the coarse line are out. The move itself is chosen by duration (joint_selection)."""
    c = [{"q": [3.0] * 6, "ok": True, "limit_margin_rad": 1.0, "min_sigma": 0.05, "dist_rad": 18.0},
         {"q": [0.1] * 6, "ok": True, "limit_margin_rad": 1.0, "min_sigma": 0.004, "dist_rad": 0.6},
         {"q": [0.2] * 6, "ok": True, "limit_margin_rad": 0.1, "min_sigma": 0.05, "dist_rad": 1.2},
         {"q": [1.0] * 6, "ok": True, "limit_margin_rad": 0.7, "min_sigma": 0.04, "dist_rad": 6.0},
         {"q": [0.5] * 6, "ok": True, "limit_margin_rad": 0.7, "min_sigma": 0.08, "dist_rad": 6.0},    # tie on dist: larger sigma first
         {"q": [0.0] * 6, "ok": False, "limit_margin_rad": 2.0, "min_sigma": 0.9, "dist_rad": 0.0}]    # does not follow the coarse line
    r = rank_reach_candidates(c)
    assert [cc["q"][0] for cc in r] == [0.1, 0.2, 0.5, 1.0, 3.0]
    assert all("condition_rel" not in cc for cc in r)
    c10 = [{**cc, "min_sigma": cc["min_sigma"] * 10.0, "limit_margin_rad": cc["limit_margin_rad"] * 0.1} for cc in c]
    assert [cc["q"][0] for cc in rank_reach_candidates(c10)] == [0.1, 0.2, 0.5, 1.0, 3.0]


def test_field_run_134838_near_solutions_are_no_longer_pushed_behind_far_ones():
    """KIRO 20261014-134838 (the 352 deg acquisition move): the recorded top candidates of the chosen order (progression -1, the
    visited end first). The old relative-condition gate put the 910 deg solution first (condition 0.72 > 0.5) and the 580 / 772 deg
    ones (condition 0.41) behind it; without the gate the nearest comes first."""
    rows = [([-72.5, 145.4, -88.4, 180.0, -123.0, 107.1], 0.3766, 180.0, 910.4),
            ([-72.5, 145.4, -88.4, -180.0, -123.0, -252.9], 0.3766, 107.0, 1462.5),
            ([-72.5, 57.0, 88.4, 180.0, -34.6, -252.9], 0.1552, 107.0, 771.7),
            ([-72.5, 57.0, 88.4, 180.0, -34.6, 107.1], 0.1552, 180.0, 579.6),
            ([-72.5, -214.6, 271.6, 180.0, -123.0, -252.9], 0.3766, 87.1, 1314.9)]
    c = [{"q": np.radians(q), "ok": True, "min_sigma": s_, "limit_margin_rad": math.radians(m_), "dist_rad": math.radians(d_)}
         for q, s_, m_, d_ in rows]
    assert [round(math.degrees(cc["dist_rad"]), 1) for cc in rank_reach_candidates(c)] == [579.6, 771.7, 910.4, 1314.9, 1462.5]


def test_seam_refusal_names_the_horizontal_distance_and_the_vertical_drop():
    """B6: KIRO 134028 sent pose_a / pose_b with the same XY 101.7 mm apart vertically (one point taught twice) - the 1G refusal says
    so (message only, no new rule)."""
    with pytest.raises(ValueError) as e:
        initial_frame([0.061566, -1.715875, 0.670799], [0.061564, -1.715897, 0.569050], "1G", G)
    msg = str(e.value)
    assert "90.0 deg from horizontal" in msg and "position and hit points disagree" in msg
    assert "0.0 mm apart horizontally, 101.7 mm vertically" in msg and "pose_a and pose_b have the same XY" in msg
    assert "check the taught end point" in msg
