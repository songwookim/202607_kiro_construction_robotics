"""툴 몸체 (EOAT 메시 껍질) 와 테이블 규칙, 패스 끝의 테이블 제한, 판을 끊는 깊은 점 띠 (2026-10-05 g3_04)."""
import math
import os
import struct

import numpy as np
import pytest

from scan_core.path_gen import (EnvelopeError, build_transit_plan, check_envelope, lowest_below_origin, pass_overscan,
                                sensor_frames, tool_keypoints)
from scan_core.seam_features import extract_cross_section_features
from scan_core.specimen import GrooveSpec
from scan_core.tool_body import body_points_base, body_points_from_mesh, hull_points, load_stl_vertices
from scan_core.types import ScanJob

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(HERE, "..", "..", "eoat_description", "meshes", "eoat_base_link_dustplate.STL")   # the CAD torch EOAT mesh (a fixture; the lab model has the torch turned in its clamp: _torch180)


def _with(job, **kw):
    return ScanJob.from_dict({**job.to_dict(), **kw})


def _write_box_stl(path, lo, hi):
    """축 정렬 상자의 binary STL (삼각형 12 개)."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=np.float32)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
             (1, 5, 7), (1, 7, 3)]
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(faces)))
        for a, b, d in faces:
            f.write(struct.pack("<3f", 0, 0, 0) + c[a].tobytes() + c[b].tobytes() + c[d].tobytes() + b"\0\0")


def test_box_stl_hull_and_transform(tmp_path):
    p = tmp_path / "box.stl"
    _write_box_stl(p, (-0.01, -0.02, 0.0), (0.2, 0.02, 0.15))
    V = load_stl_vertices(p)
    assert V.shape == (36, 3)
    H = body_points_from_mesh(p)
    assert H.shape == (8, 3)
    np.testing.assert_allclose(H.min(axis=0), [-0.01, -0.02, 0.0], atol=1e-6)
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(body_points_base(T, H).min(axis=0), [0.99, 1.98, 3.0], atol=1e-6)
    assert body_points_base(np.stack([T, T]), H).shape == (2, 8, 3)
    assert hull_points(np.vstack([V, V.mean(axis=0, keepdims=True)])).shape == (8, 3)       # 안쪽 점은 껍질이 아니다


@pytest.mark.skipif(not os.path.exists(MESH), reason="EOAT mesh not in this checkout")
def test_eoat_mesh_hull_contains_the_sensor_camera_end():
    H = body_points_from_mesh(MESH)
    assert 50 <= H.shape[0] <= 2000
    # 센서 하우징이 eoat +x 로 ~0.20 m (카메라 끝), 토치 쪽 −x 0.167 m, 플랜지 면 z −0.0225 ~ 토치 0.315 m
    assert H[:, 0].max() == pytest.approx(0.200, abs=0.002) and H[:, 0].min() == pytest.approx(-0.167, abs=0.002)
    assert H[:, 2].min() == pytest.approx(-0.0225, abs=0.002) and H[:, 2].max() == pytest.approx(0.315, abs=0.002)


def _calib_with_body(calib_lab):
    calib_lab.body_points_eoat = body_points_from_mesh(MESH)
    calib_lab.body_source = "test mesh"
    return calib_lab


def _torch_up(job, calib):
    def tip_up(j):
        kp = tool_keypoints(sensor_frames(j, calib).T_base_flange(0.0), calib, j)
        return float(j.height(kp["tip"]) - j.height(kp["opt"]))
    return max((_with(job, progression_sign=s) for s in (1, -1)), key=tip_up)


@pytest.mark.skipif(not os.path.exists(MESH), reason="EOAT mesh not in this checkout")
def test_table_rule_sees_the_tool_body_not_only_keypoints(job_3g, calib_lab):
    """3G 토치 위 자세: 카메라 끝이 원점보다 아래 — 키포인트만으로는 테이블 위, 몸체를 넣으면 테이블 아래 (g3_04 E 의 상황)."""
    calib = _calib_with_body(calib_lab)
    job = _torch_up(_with(job_3g, incidence_tilt_deg=20.0, vertical_direction="up"), calib)
    T = sensor_frames(job, calib).T_base_flange(0.0)
    below = lowest_below_origin(T, calib, job)
    below_kp = lowest_below_origin(T, calib, job, body=False)
    assert below > 0.15 and below_kp <= 1e-9                    # 키포인트 중 가장 낮은 것 = 원점 (토치 팁은 위)
    h_opt = float(job.height(tool_keypoints(T, calib, job)["opt"]))
    table = h_opt - below + 0.01                                 # 몸체 최저점 10 mm 위 (마진 0 → 몸체만 걸린다)
    job_t = _with(job, table_z_m=table)
    plan = build_transit_plan(job_t, calib, check=False)
    calib_nobody = type(calib)(**{**calib.__dict__, "body_points_eoat": None})
    check_envelope(plan, job_t, calib_nobody)                                     # 키포인트만: 통과 (원점이 테이블 + 한계 위)
    with pytest.raises(EnvelopeError, match="rule=never_below_table.*tool body"):
        check_envelope(plan, job_t, calib)
    stats = check_envelope(build_transit_plan(_with(job, table_z_m=table - 0.2), calib, check=False), _with(job, table_z_m=table - 0.2),
                           calib)
    assert stats["min_body_height"] < stats["min_opt_height"] - 0.15


@pytest.mark.skipif(not os.path.exists(MESH), reason="EOAT mesh not in this checkout")
def test_pass_overscan_stops_the_lower_end_at_the_table(job_3g, calib_lab):
    """3G 위로 가는 패스 (start = 아래 끝): 테이블이 끝을 정하면 'table' 출처, 끝 자세의 몸체 최저점 = 테이블 + 여유."""
    calib = _calib_with_body(calib_lab)
    job = _torch_up(_with(job_3g, incidence_tilt_deg=20.0, vertical_direction="up", lead_in_m=0.001, lead_out_m=0.001), calib)
    free = pass_overscan(job, calib, 0.02)
    assert not free["floor_limited"]
    T0 = free["T_from"]
    low0 = float(job.height(tool_keypoints(T0, calib, job)["opt"])) - lowest_below_origin(T0, calib, job)
    table = low0 + 0.03 - 0.02                                   # 여유 0.02 포함 바닥이 지금 최저점보다 30 mm 위
    ov = pass_overscan(_with(job, table_z_m=table, safe_clearance_m=0.02, pose_tol_m=0.002), calib, 0.02)
    assert ov["floor_limited"] == ["first"] and ov["floor_source"] == {"first": "table"}
    assert ov["overscan_first_m"] == pytest.approx(0.02 - 0.032 / abs(float(job.up_unit @ job.unit_dir)), abs=1e-3)   # + 2 mm 계획 여유
    T1 = ov["T_from"]
    low1 = float(job.height(tool_keypoints(T1, calib, job)["opt"])) - lowest_below_origin(T1, calib, job)
    assert low1 == pytest.approx(table + 0.02 + 0.002, abs=1e-6)                 # 테이블 + 한계 + 도착 허용
    assert "first" in ov["floor_blocked"]                       # 계획 오버스캔 20 mm 의 절반 미만 (−10 mm)
    # 3G 바닥이 더 높으면 그쪽이 정한다
    ov2 = pass_overscan(_with(job, table_z_m=table - 1.0), calib, 0.02, floor_h=float(job.height(job.start)) + 0.12)
    assert ov2["floor_source"] == {"first": "3G floor"}


def _bevel_profile(x, plate_z=430.0, open_l=-5.8, open_r=5.0, gap_l=-0.8, depth=10.7):
    """단면 V (왼쪽 30° 베벨) + 바닥 — 센서 z (mm, 클수록 멂)."""
    z = np.full_like(x, plate_z)
    bev = (x > open_l) & (x <= gap_l)
    z[bev] = plate_z + (x[bev] - open_l) / (gap_l - open_l) * depth
    fl = (x > gap_l) & (x < open_r)
    z[fl] = plate_z + depth
    return z


def test_deep_strip_beside_the_groove_does_not_split_the_plate():
    spec = GrooveSpec(kind="single_bevel", thickness_mm=10.0, bevel_angle_deg=30.0, bevel_side="left", root_gap_mm=5.0, backing=True,
                      backing_thickness_mm=10.0, weld_length_mm=380.0)
    x = np.arange(-88.0, 93.0, 0.13)
    z = _bevel_profile(x, open_l=18.0, open_r=29.0, gap_l=23.8)
    ref = extract_cross_section_features(x, z, spec)
    assert ref.groove_found and ref.top_opening_mm == pytest.approx(11.0, abs=0.6) and "plate_strip_bridged" not in ref.flags
    zs = z.copy()
    strip = (x > -6.0) & (x < -1.5)                              # 그루브 왼쪽 ~22 mm, 판 뒤 86 mm 띠 (g3_04 E)
    zs[strip] += 86.0
    f = extract_cross_section_features(x, zs, spec)
    assert f.groove_found and "plate_strip_bridged" in f.flags
    assert f.top_opening_mm == pytest.approx(ref.top_opening_mm, abs=0.3)
    assert f.left_edge_x_mm == pytest.approx(ref.left_edge_x_mm, abs=0.3)
    assert any("86 mm behind" in n for n in f.notes)


def test_two_deep_only_gaps_stay_ambiguous():
    """그루브 깊이 범위의 점이 있는 조각이 없으면 (열린 I형 + 띠) 판을 잇지 않는다."""
    x = np.arange(-88.0, 93.0, 0.13)
    z = np.full_like(x, 430.0)
    for lo, hi in ((-6.0, -1.5), (20.0, 25.0)):
        m = (x > lo) & (x < hi)
        z[m] += 90.0
    f = extract_cross_section_features(x, z, None)
    assert "plate_strip_bridged" not in f.flags


@pytest.mark.skipif(not os.path.exists(MESH), reason="EOAT mesh not in this checkout")
def test_measured_start_a_hair_below_the_floor_is_tolerated_but_not_later(job_3g, calib_lab):
    """g3_06: 패스 시작 실측 자세가 계획 (바닥) 보다 <0.1 mm 아래 → 패스 거부. 지금 자세·첫 구간은 도착 허용 (job.pose_tol_m) 까지,
    뒤 구간은 그대로 엄격."""
    from scan_core.path_gen import TransitPlan, Waypoint
    calib = _calib_with_body(calib_lab)
    job = _torch_up(_with(job_3g, incidence_tilt_deg=20.0, vertical_direction="up", lead_in_m=0.001, lead_out_m=0.001), calib)
    fr = sensor_frames(job, calib)
    T0 = fr.T_base_flange(0.0)
    low0 = float(job.height(tool_keypoints(T0, calib, job)["opt"])) - lowest_below_origin(T0, calib, job)
    T_up = np.array(T0, copy=True)
    T_up[:3, 3] += job.up_unit * 0.05
    T_dn = np.array(T0, copy=True)
    T_dn[:3, 3] -= job.up_unit * 0.003

    def plan(*Ts):
        wps = [Waypoint("now", "start", Ts[0], 0.0, 0.0)] + [Waypoint(f"w{i}", "scan", T, 0.02, 0.2) for i, T in enumerate(Ts[1:])]
        return TransitPlan(waypoints=wps, frames=fr, orientation="vertical_3G", z_safe_m=1.0, z_safe_opt_m=1.0, specimen_top_z_m=0.7,
                           rules=["never_below_table"], calib=calib)
    L = 0.02
    jt = _with(job, table_z_m=low0 + 0.0005 - L, safe_clearance_m=L, pose_tol_m=0.002)   # 지금 자세가 바닥 (테이블 + 한계) 보다 0.5 mm 아래
    check_envelope(plan(T0, T_up), jt, calib)                                         # 지금 자세 → 위로: 허용
    with pytest.raises(EnvelopeError, match="never_below_table"):
        check_envelope(plan(T0, T_up, T_dn), jt, calib)                               # 둘째 구간이 바닥 아래로: 거부
    with pytest.raises(EnvelopeError, match="never_below_table"):                    # 도착 허용 없이는 지금 자세부터 거부
        check_envelope(plan(T0, T_up), _with(jt, pose_tol_m=0.0), calib)


def test_depth_scale_undoes_the_out_of_plane_stretch():
    """g3_07 (레이저 평면이 판 법선에서 40°): 단면 깊이가 1/cos40 으로 늘어 보인다. depth_scale = cos40 이면 깊이·베벨·갭이 설계로,
    센서 좌표 출력 (코너·루트) 은 보정 없이 뽑은 것과 같은 자리 (3D 점 불변)."""
    from scan_core.seam_features import depth_scales
    spec = GrooveSpec(kind="single_bevel", thickness_mm=10.0, bevel_angle_deg=30.0, bevel_side="left", root_gap_mm=5.0, backing=True,
                      backing_thickness_mm=10.0, weld_length_mm=380.0)
    c = math.cos(math.radians(40.0))
    x = np.arange(-88.0, 93.0, 0.13)
    run = 10.0 * math.tan(math.radians(30.0))
    z = _bevel_profile(x, open_l=-run - 2.5, open_r=2.5, gap_l=-2.5, depth=10.0 / c)        # 깊이만 1/cos 로 늘어남
    raw = extract_cross_section_features(x, z, spec)
    fix = extract_cross_section_features(x, z, spec, depth_scale=c)
    assert raw.depth_mm == pytest.approx(13.05, abs=0.1)
    assert fix.depth_mm == pytest.approx(10.0, abs=0.1)
    assert fix.bevel_angle_deg == pytest.approx(30.0, abs=1.0) and raw.bevel_angle_deg < 25.0
    assert fix.root_gap_mm == pytest.approx(5.0, abs=0.3)
    np.testing.assert_allclose(fix.top_corner_left_xz, raw.top_corner_left_xz, atol=0.3)
    assert fix.root_centre_z_mm == pytest.approx(430.0 + 10.0 / c, abs=0.2)                 # 센서 z 는 늘어난 그대로 (실제 점)
    T = np.tile(np.eye(4), (3, 1, 1))
    T[1, :3, :3] = np.array([[1, 0, 0], [0, c, -math.sin(math.radians(40.0))], [0, math.sin(math.radians(40.0)), c]])   # X 축 둘레 40°
    np.testing.assert_allclose(depth_scales(T, [0, 0, 1])[:2], [1.0, c], atol=1e-9)
    assert depth_scales(T, None) is None

