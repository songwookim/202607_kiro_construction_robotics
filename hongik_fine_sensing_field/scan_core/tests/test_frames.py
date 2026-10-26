import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scan_core import frames as F


@pytest.mark.parametrize("rpy", [(0.3, -0.4, 1.2), (0, 0, 0), (np.pi, 0, -np.pi / 2), (-1.0, 1.2, 3.0)])
def test_rpy_matches_scipy_extrinsic_xyz(rpy):
    R = F.rpy_to_matrix(*rpy)
    assert np.allclose(R, Rotation.from_euler("xyz", rpy).as_matrix())
    assert np.allclose(R, F.rot_z(rpy[2]) @ F.rot_y(rpy[1]) @ F.rot_x(rpy[0]))


@pytest.mark.parametrize("rpy", [(0.3, -0.4, 1.2), (-2.5, 0.9, -3.0), (0.1, 1.4, 0.2)])
def test_rpy_roundtrip(rpy):
    R = F.rpy_to_matrix(*rpy)
    r2 = F.matrix_to_rpy(R)
    assert np.allclose(F.rpy_to_matrix(*r2), R, atol=1e-12)


def test_transform_inverse_compose_points():
    T = F.transform_from_xyz_rpy((0.1, -0.2, 0.3), (0.2, 0.3, -0.4))
    assert F.is_transform(T)
    assert np.allclose(F.inv_transform(T) @ T, np.eye(4), atol=1e-12)
    P = np.random.default_rng(0).normal(size=(10, 3))
    Q = F.transform_points(T, P)
    assert np.allclose(F.transform_points(F.inv_transform(T), Q), P)
    xyz, rpy = F.xyz_rpy_from_transform(T)
    assert np.allclose(F.transform_from_xyz_rpy(xyz, rpy), T)


def test_laser_joint_rotation_is_diag():
    R = F.rpy_to_matrix(np.pi, 0, np.pi)
    assert np.allclose(R, np.diag([-1, 1, -1]), atol=1e-12)


def test_interpolate_transforms_endpoints():
    T0 = np.eye(4)
    T1 = F.transform_from_xyz_rpy((1, 0, 0), (0, 0, np.pi / 2))
    Ts = F.interpolate_transforms(T0, T1, 5)
    assert Ts.shape == (5, 4, 4)
    assert np.allclose(Ts[0], T0) and np.allclose(Ts[-1], T1)
    assert abs(F.rotation_error(Ts[2][:3, :3], F.rot_z(np.pi / 4))) < 1e-9
