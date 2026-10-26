import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scan_core.calib.sphere import fit_circle, solve_hand_eye, sphere_center_along_track, sphere_center_from_arc
from scan_core.frames import inv_transform, make_transform, rotation_error, transform_points
from scan_core.path_gen import plate_frame
from scan_core.synth import SphereScene, SynthConfig, static_profile
from scan_core.types import Calib


def _poses(rng, n_trans=10, n_rot=10, center=(0.6, 0.0, 0.05), standoff=0.32, T_true=None):
    """광학 원점이 구 중심 위 ~standoff 에 오는 병진 10 + 회전 10 자세의 **플랜지** 포즈.
    광학 Z 는 아래(-z_base), X_opt = +x_base. T_true 미지정 시 연구실 초기값."""
    T_true = Calib.lab_initial().T_flange_opt if T_true is None else T_true
    C = np.asarray(center, dtype=float)
    R_opt0 = np.diag([1.0, -1.0, -1.0])
    Ts = []
    for _ in range(n_trans):
        off = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.005, 0.005), rng.uniform(-0.08, 0.08)])
        T_opt = make_transform(R_opt0, C + np.array([0, 0, standoff]) + off)
        Ts.append(T_opt @ inv_transform(T_true))
    for _ in range(n_rot):
        rv = np.radians([rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-30, 30)])
        R = Rotation.from_rotvec(rv).as_matrix() @ R_opt0
        # 회전 후에도 구 중심이 광학축 근방(x ±20 mm, y ±5 mm)에 오도록 원점을 잡는다
        c_opt = np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.005, 0.005), standoff])
        T_opt = make_transform(R, C - R @ c_opt)
        Ts.append(T_opt @ inv_transform(T_true))
    return Ts


def test_fit_circle_exact_and_noisy():
    th = np.linspace(0.3, 2.6, 80)
    x = 3.0 + 20.0 * np.cos(th)
    z = 320.0 - 20.0 * np.sin(th)
    xc, zc, r, rms = fit_circle(x, z)
    assert abs(xc - 3.0) < 1e-9 and abs(zc - 320.0) < 1e-9 and abs(r - 20.0) < 1e-9
    rng = np.random.default_rng(0)
    xc, zc, r, rms = fit_circle(x + rng.normal(0, 0.01, 80), z + rng.normal(0, 0.01, 80))
    assert abs(r - 20.0) < 0.01 and rms < 0.02


def test_sphere_center_from_arc_and_along_track():
    c, info = sphere_center_from_arc(3.0 + 15.0 * np.cos(np.linspace(0.2, 2.9, 50)),
                                     320.0 - 15.0 * np.sin(np.linspace(0.2, 2.9, 50)), radius_mm=20.0, y_sign=-1)
    assert np.allclose(c, [3.0, -np.sqrt(400 - 225), 320.0], atol=1e-6)
    s = np.linspace(-0.015, 0.015, 31)
    r = np.sqrt(20.0 ** 2 - ((s - 0.004) * 1e3) ** 2)
    sc, R = sphere_center_along_track(s, r, 20.0)
    assert abs(sc - 0.004) < 1e-9 and abs(R - 20.0) < 1e-9


def test_hand_eye_recovers_injected_transform_noise_free():
    rng = np.random.default_rng(2)
    T_true = Calib.lab_initial().T_flange_opt @ make_transform(Rotation.from_rotvec([0.01, -0.02, 0.03]).as_matrix(),
                                                              (0.002, -0.001, 0.003))
    C = np.array([0.6, 0.0, 0.07])
    Ts = _poses(rng, center=(0.6, 0.0, 0.07), T_true=T_true)
    cs = [transform_points(inv_transform(T @ T_true), C) for T in Ts]
    r = solve_hand_eye(Ts, cs, refine=True)
    assert np.linalg.norm(r.T_flange_opt[:3, 3] - T_true[:3, 3]) < 0.05e-3
    assert rotation_error(r.T_flange_opt[:3, :3], T_true[:3, :3]) < 0.1e-3
    assert np.linalg.norm(r.C_base - C) < 0.05e-3
    assert r.rms_m < 1e-9
    # 선형 초기값만으로도 근접
    r0 = solve_hand_eye(Ts, cs, refine=False)
    assert np.linalg.norm(r0.T_flange_opt[:3, 3] - T_true[:3, 3]) < 1e-6


def test_hand_eye_with_noise_reasonable():
    rng = np.random.default_rng(3)
    T_true = Calib.lab_initial().T_flange_opt
    C = np.array([0.6, 0.0, 0.07])
    Ts = _poses(rng, center=(0.6, 0.0, 0.07))
    cs = [transform_points(inv_transform(T @ T_true), C) + rng.normal(0, 0.02e-3, 3) for T in Ts]
    r = solve_hand_eye(Ts, cs)
    assert np.linalg.norm(r.T_flange_opt[:3, 3] - T_true[:3, 3]) < 0.2e-3
    assert rotation_error(r.T_flange_opt[:3, :3], T_true[:3, :3]) < 1e-3


def test_hand_eye_from_raytraced_arcs():
    """합성 프로파일(정지 자세 20개) → 원 피팅 → 구 중심 → 핸드아이. 부호 ±y 는 진실값으로 선택."""
    rng = np.random.default_rng(4)
    calib_true = Calib.lab_initial()
    T_true = calib_true.T_flange_opt
    R_mm = 20.0
    C = np.array([0.6, 0.0, 0.05 + R_mm * 1e-3])
    T_plate = plate_frame([0.6, 0.0, 0.05], [0, 1, 0], [0, 0, 1])
    scene = SphereScene(xc=0.0, yc=0.0, radius=R_mm * 1e-3, h0=0.0)
    cfg = SynthConfig(n_points=400, sigma_z_mm=0.0, z_grid_mm=(200.0, 450.0, 1.0), bisect_iters=40)
    Ts, cs = [], []
    for T in _poses(rng, n_trans=10, n_rot=10, center=tuple(C)):
        T_bo = T @ T_true
        c_true = transform_points(inv_transform(T_bo), C)
        x, z = static_profile(T, calib_true, scene, T_plate, cfg, rng)
        h = transform_points(inv_transform(T_plate), transform_points(T_bo, np.stack([x, 0 * x, z], -1) * 1e-3))[:, 2]
        on_sphere = np.isfinite(x) & (h > R_mm * 1e-3 + 1e-3)   # 적도 위 (높이맵의 가상 벽 제외)
        assert on_sphere.sum() > 30
        c_est, info = sphere_center_from_arc(x[on_sphere], z[on_sphere], R_mm, y_sign=int(np.sign(c_true[1]) or 1))
        assert abs(info["r"] - np.sqrt(R_mm ** 2 - (c_true[1] * 1e3) ** 2)) < 0.02
        assert np.linalg.norm(c_est * 1e-3 - c_true) < 0.03e-3
        Ts.append(T)
        cs.append(c_est * 1e-3)
    r = solve_hand_eye(Ts, cs)
    assert np.linalg.norm(r.T_flange_opt[:3, 3] - T_true[:3, 3]) < 0.05e-3
    assert rotation_error(r.T_flange_opt[:3, :3], T_true[:3, :3]) < 0.2e-3
    assert np.linalg.norm(r.C_base - C) < 0.05e-3
