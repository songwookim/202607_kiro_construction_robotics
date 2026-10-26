"""Numpy forward kinematics from a URDF (or rbpodo_description joint.yaml) for one serial chain.

용도: 5001 ``jnt_ang`` (엔코더) → 플랜지(=URDF ``tcp``, link6 + 0.13 m) 포즈. ``tcp_pos`` 는 reference 복사본이라
측정 포즈는 이 FK 로만 얻는다.

joint.yaml 의 shoulder 오프셋 (검증된 회귀)
------------------------------------------
upstream ``rbpodo_description/robots/rb20_1900es_u/joint.yaml`` 은 shoulder origin 이 ``xyz: 0 0 0`` 이지만
정적 URDF ``rb20_1900es_u.urdf`` 와 제품 치수표(254.5 mm)는 ``0 -0.2545 0`` 이다. ``chain_from_joint_yaml`` 은
기본적으로 이를 교정한다(``fix_shoulder=True``). 0 자세 플랜지 = (0, -0.067, 2.141) m:
  y = -0.2545 + 0.1875 - 0.1292 + 0.1292 = -0.067,  z = 0.241 + 0.885 + 0.885 + 0.13 = 2.141.

규약: URDF rpy = Rz(yaw)·Ry(pitch)·Rx(roll) (extrinsic xyz). 관절 회전은 axis 기준 우수좌표.
"""
from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

RB20_1900ES_U_SHOULDER_Y = -0.2545
RB20_1900ES_U_TCP_XYZ = (0.0, 0.0, 0.13)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF/ROS convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def make_tf(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


@dataclass(frozen=True)
class Joint:
    name: str
    type: str  # 'revolute' | 'continuous' | 'prismatic' | 'fixed'
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float = -math.inf
    upper: float = math.inf

    @property
    def movable(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")

    def origin(self) -> np.ndarray:
        return make_tf(self.xyz, self.rpy)

    def motion(self, q: float) -> np.ndarray:
        T = np.eye(4)
        if self.type in ("revolute", "continuous"):
            T[:3, :3] = axis_angle(self.axis, q)
        elif self.type == "prismatic":
            T[:3, 3] = np.asarray(self.axis) * q
        return T


class KinematicChain:
    """Serial chain base_frame -> tip_frame. ``fk(q)`` takes movable joint values in chain order."""

    def __init__(self, joints: Sequence[Joint], base_frame: str, tip_frame: str):
        self.joints = list(joints)
        self.base_frame = base_frame
        self.tip_frame = tip_frame
        self.movable_joints = [j for j in self.joints if j.movable]
        self.joint_names = [j.name for j in self.movable_joints]
        self.dof = len(self.movable_joints)
        # 고정 원점 변환은 한 번만 계산 (fk 마다 rpy → 행렬을 다시 만들면 표본 수천 개에서 수백 ms — 2026-10-02 실기: 342 mm 이동 뒤
        # 속도 측정 FK 2614 회가 0.5 s 동안 GIL 을 잡아 센서 가드 평가가 밀리고 G3 거짓 정지)
        self._origins = [j.origin() for j in self.joints]

    def limit_problems(self, q) -> list:
        """Movable joints outside their URDF limits: q in chain order, or a dict {joint name: value}. Returns one text per
        offending (or missing) joint, e.g. 'elbow -152.07 deg outside [-150, 150] deg (2.07 deg beyond)'; [] when all fit."""
        if isinstance(q, dict):
            vals = [q.get(n) for n in self.joint_names]
        else:
            qa = np.asarray(q, dtype=np.float64).ravel()
            vals = list(qa) if qa.size == self.dof else [None] * self.dof
        out = []
        for j, v in zip(self.movable_joints, vals):
            if v is None or not math.isfinite(float(v)):
                out.append(f"{j.name} missing")
                continue
            v = float(v)
            if v < j.lower - 1e-9 or v > j.upper + 1e-9:
                rot = j.type != "prismatic"
                f = math.degrees if rot else (lambda x: x)
                unit = "deg" if rot else "m"
                over = (j.lower - v) if v < j.lower else (v - j.upper)
                out.append(f"{j.name} {f(v):.2f} {unit} outside [{f(j.lower):.0f}, {f(j.upper):.0f}] {unit} ({f(over):.2f} {unit} beyond)")
        return out

    def limit_margins(self, q) -> dict:
        """{joint name: distance to the nearer URDF limit} (rad / m; negative = outside) for q in chain order or a dict."""
        vals = [q.get(n) for n in self.joint_names] if isinstance(q, dict) else list(np.asarray(q, dtype=np.float64).ravel())
        return {j.name: (float(min(float(v) - j.lower, j.upper - float(v))) if v is not None else float("nan"))
                for j, v in zip(self.movable_joints, vals)}

    def reach_bound_m(self) -> float:
        """An upper bound on the distance from the base origin to the tip for ANY joint values: the sum of the joints' origin offsets
        (triangle inequality; prismatic strokes added at their larger limit)."""
        r = sum(float(np.linalg.norm(j.xyz)) for j in self.joints)
        r += sum(max(abs(j.lower), abs(j.upper)) for j in self.movable_joints if j.type == "prismatic"
                 and math.isfinite(j.lower) and math.isfinite(j.upper))
        return float(r)

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """4x4 T_base_tip [m] for joint values q [rad] (revolute) / [m] (prismatic)."""
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.size != self.dof:
            raise ValueError(f"expected {self.dof} joint values, got {q.size}")
        T = np.eye(4)
        k = 0
        for j, O in zip(self.joints, self._origins):
            T = T @ O
            if j.movable:
                T = T @ j.motion(float(q[k]))
                k += 1
        return T

    def fk_deg(self, q_deg: Sequence[float]) -> np.ndarray:
        return self.fk(np.deg2rad(np.asarray(q_deg, dtype=np.float64)))

    def fk_all(self, q: Sequence[float]) -> dict[str, np.ndarray]:
        """Transform of every link frame in the chain (keyed by child link name)."""
        q = np.asarray(q, dtype=np.float64).ravel()
        T = np.eye(4)
        out = {self.base_frame: T.copy()}
        k = 0
        for j in self.joints:
            T = T @ j.origin()
            if j.movable:
                T = T @ j.motion(float(q[k]))
                k += 1
            out[j.child] = T.copy()
        return out

    def fk_batch(self, Q: np.ndarray) -> np.ndarray:
        """(N, dof) 관절값 [rad / m] → (N, 4, 4) T_base_tip — 벡터화 (관절마다 N 개 행렬을 한 번에: Rodrigues
        R = I + sinθ·K + (1−cosθ)·K²). fk() 와 같은 값 (부동소수 반올림 차이만)."""
        Q = np.asarray(Q, dtype=np.float64).reshape(-1, self.dof)
        N = Q.shape[0]
        T = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
        k = 0
        for j, O in zip(self.joints, self._origins):
            T = T @ O
            if j.movable:
                q = Q[:, k]
                M = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
                if j.type in ("revolute", "continuous"):
                    a = np.asarray(j.axis, dtype=np.float64)
                    n = float(np.linalg.norm(a))
                    if n > 0.0:
                        a = a / n
                        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
                        M[:, :3, :3] = (np.eye(3)[None] + np.sin(q)[:, None, None] * K[None]
                                        + (1.0 - np.cos(q))[:, None, None] * (K @ K)[None])
                elif j.type == "prismatic":
                    M[:, :3, 3] = np.asarray(j.axis, dtype=np.float64)[None, :] * q[:, None]
                T = T @ M
                k += 1
        return T

    def fk_deg_batch(self, Q_deg: np.ndarray) -> np.ndarray:
        return self.fk_batch(np.deg2rad(np.asarray(Q_deg, dtype=np.float64)))

    def check_limits(self, q: Sequence[float]) -> list[str]:
        out = []
        for j, v in zip(self.movable_joints, q):
            if not (j.lower <= v <= j.upper):
                out.append(f"{j.name}={v:.4f} outside [{j.lower:.4f}, {j.upper:.4f}]")
        return out

    def __repr__(self) -> str:
        return f"KinematicChain({self.base_frame}->{self.tip_frame}, joints={[j.name for j in self.joints]})"


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------
def _floats(s: Optional[str], default: tuple[float, ...]) -> tuple[float, ...]:
    if s is None:
        return default
    v = tuple(float(x) for x in s.replace(",", " ").split())
    return v if v else default


def parse_urdf_joints(urdf_path: str) -> dict[str, Joint]:
    """Parse ``<joint>`` elements that are direct children of ``<robot>`` (skips <ros2_control> joints)."""
    return parse_urdf_joints_root(ET.parse(urdf_path).getroot())


def parse_urdf_joints_root(root) -> dict[str, Joint]:
    """parse_urdf_joints on an already parsed ``<robot>`` element."""
    joints: dict[str, Joint] = {}
    for el in root.findall("joint"):
        name = el.get("name")
        jtype = el.get("type", "fixed")
        parent = el.find("parent")
        child = el.find("child")
        if name is None or parent is None or child is None:
            continue
        origin = el.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _floats(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        axis_el = el.find("axis")
        axis = _floats(axis_el.get("xyz") if axis_el is not None else None, (1.0, 0.0, 0.0))
        lim = el.find("limit")
        lower = float(lim.get("lower", "-inf")) if lim is not None else -math.inf
        upper = float(lim.get("upper", "inf")) if lim is not None else math.inf
        joints[name] = Joint(name, jtype, parent.get("link"), child.get("link"), xyz, rpy, axis, lower, upper)
    return joints


def chain_from_joints(joints: dict[str, Joint], base_frame: str, tip_frame: str) -> KinematicChain:
    by_child = {j.child: j for j in joints.values()}
    path: list[Joint] = []
    link = tip_frame
    seen = set()
    while link != base_frame:
        if link in seen:
            raise ValueError(f"cycle while walking from {tip_frame} to {base_frame}")
        seen.add(link)
        j = by_child.get(link)
        if j is None:
            raise ValueError(f"frame '{link}' has no parent joint; cannot reach '{base_frame}' from '{tip_frame}'")
        path.append(j)
        link = j.parent
    path.reverse()
    return KinematicChain(path, base_frame, tip_frame)


def chain_from_urdf(urdf_path: str, base_frame: str = "link0", tip_frame: str = "tcp") -> KinematicChain:
    return chain_from_joints(parse_urdf_joints(urdf_path), base_frame, tip_frame)


# ---------------------------------------------------------------------------
# rbpodo_description joint.yaml loader (fallback when no static URDF is available)
# ---------------------------------------------------------------------------
def chain_from_joint_yaml(
    yaml_path: str,
    base_frame: str = "link0",
    tcp_xyz: Sequence[float] = RB20_1900ES_U_TCP_XYZ,
    tcp_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    tcp_frame: str = "tcp",
    fix_shoulder: bool = True,
    shoulder_y: float = RB20_1900ES_U_SHOULDER_Y,
) -> KinematicChain:
    """Build the chain from ``joint.yaml`` + a fixed tcp joint (link6 -> tcp).

    ``fix_shoulder``: upstream joint.yaml has the shoulder origin at 0 0 0 (regression). When the parsed
    shoulder xyz is all-zero and fix_shoulder is True, it is replaced by (0, shoulder_y, 0) and a warning
    is issued. The static URDF in the same package already carries -0.2545.
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    joints: dict[str, Joint] = {}
    for name, spec in data.items():
        org = spec.get("origin", {})
        xyz = _floats(str(org.get("xyz", "0 0 0")), (0.0, 0.0, 0.0))
        rpy = _floats(str(org.get("rpy", "0 0 0")), (0.0, 0.0, 0.0))
        axis = _floats(str(spec.get("axis", {}).get("xyz", "0 0 1")), (0.0, 0.0, 1.0))
        lim = spec.get("limit", {})
        lower = float(lim.get("lower", -math.inf))
        upper = float(lim.get("upper", math.inf))
        jtype = spec.get("type", "revolute")
        if name == "shoulder" and fix_shoulder and all(abs(v) < 1e-12 for v in xyz):
            warnings.warn(
                f"{yaml_path}: shoulder origin is 0 0 0 (upstream regression); using 0 {shoulder_y} 0",
                stacklevel=2,
            )
            xyz = (0.0, shoulder_y, 0.0)
        joints[name] = Joint(name, jtype, spec["parent"], spec["child"], tuple(xyz), tuple(rpy), tuple(axis), lower, upper)
    # tcp joint (flange face) as in rb20_1900es_u.urdf.xacro tcp_xyz="0 0 0.13"
    last_child = "link6" if "link6" in {j.child for j in joints.values()} else max(joints.values(), key=lambda j: j.child).child
    joints["tcp_joint"] = Joint("tcp_joint", "fixed", last_child, tcp_frame, tuple(float(v) for v in tcp_xyz),
                                tuple(float(v) for v in tcp_rpy))
    return chain_from_joints(joints, base_frame, tcp_frame)


def rb20_1900es_u_chain() -> KinematicChain:
    """Hard-coded RB20-1900ES-U chain (manual dimension table: 241, 885, 885, 130, 254.5, 187.5, 129.2 mm)."""
    lim = (-math.pi, math.pi)
    js = [
        Joint("base", "revolute", "link0", "link1", (0, 0, 0.241), (0, 0, 0), (0, 0, 1), *lim),
        Joint("shoulder", "revolute", "link1", "link2", (0, RB20_1900ES_U_SHOULDER_Y, 0), (0, 0, 0), (0, 1, 0), *lim),
        Joint("elbow", "revolute", "link2", "link3", (0, 0, 0.885), (0, 0, 0), (0, 1, 0), *lim),
        Joint("wrist1", "revolute", "link3", "link4", (0, 0.1875, 0), (0, 0, 0), (0, 0, 1), *lim),
        Joint("wrist2", "revolute", "link4", "link5", (0, -0.1292, 0.885), (0, 0, 0), (0, 1, 0), *lim),
        Joint("wrist3", "revolute", "link5", "link6", (0, 0.1292, 0), (0, 0, 0), (0, 0, 1), *lim),
        Joint("tcp_joint", "fixed", "link6", "tcp", RB20_1900ES_U_TCP_XYZ, (0, 0, 0)),
    ]
    return KinematicChain(js, "link0", "tcp")


def load_chain(urdf: Optional[str] = None, joint_yaml: Optional[str] = None,
               base_frame: str = "link0", tip_frame: str = "tcp") -> KinematicChain:
    """Best available chain: URDF > joint.yaml (+fix) > hard-coded RB20 dims."""
    if urdf:
        return chain_from_urdf(urdf, base_frame, tip_frame)
    if joint_yaml:
        return chain_from_joint_yaml(joint_yaml, base_frame=base_frame, tcp_frame=tip_frame)
    return rb20_1900es_u_chain()


# ---------------------------------------------------------------------------
# The robot's fixed body: collision geometry of the links rigidly attached to the base (no joint moves them)
# ---------------------------------------------------------------------------
def _stl_bounds(path: str) -> Optional[tuple]:
    """Axis-aligned bounds (lo, hi) of an STL file (binary or ASCII) in its own frame, or None when it has no vertex."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n and n > 0:
            rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
            P = rec["v"].reshape(-1, 3).astype(np.float64)
            return P.min(axis=0), P.max(axis=0)
    pts = [[float(x) for x in ln.split()[1:4]] for ln in data.decode("ascii", errors="replace").splitlines()
           if ln.strip().startswith("vertex")]
    if not pts:
        return None
    P = np.asarray(pts, dtype=np.float64)
    return P.min(axis=0), P.max(axis=0)


def default_mesh_path(filename: str) -> Optional[str]:
    """URDF mesh file name → local path: package://<pkg>/<rel> through the ament index, file://<path> and plain paths as they are.
    None when the package cannot be found here."""
    if filename.startswith("package://"):
        pkg, _, rel = filename[len("package://"):].partition("/")
        try:
            from ament_index_python.packages import get_package_share_directory
            import os
            return os.path.join(get_package_share_directory(pkg), rel)
        except Exception:  # noqa: BLE001 - not installed / no ament index here
            return None
    if filename.startswith("file://"):
        return filename[len("file://"):]
    return filename


def fixed_link_boxes(urdf_source, base_frame: str, *, mesh_path=default_mesh_path) -> list:
    """The robot's fixed body: for every link rigidly attached to ``base_frame`` (connected to it through fixed joints only, in
    either direction — the base itself included), the axis-aligned box in the base frame that holds that link's collision geometry
    (box / cylinder / sphere / STL mesh, with the collision origin and the mesh scale). urdf_source: a path or a file object.
    → [{"link", "lo": (3,), "hi": (3,)}] — a link whose geometry cannot be read here (mesh not found, not STL) is listed with
    lo = hi = None and "why"; links without collision geometry are left out."""
    root = ET.parse(urdf_source).getroot()
    joints = [j for j in parse_urdf_joints_root(root).values() if j.type == "fixed"]
    T_of = {base_frame: np.eye(4)}
    todo = [base_frame]
    while todo:
        cur = todo.pop()
        for j in joints:
            if j.parent == cur and j.child not in T_of:
                T_of[j.child] = T_of[cur] @ j.origin()
                todo.append(j.child)
            elif j.child == cur and j.parent not in T_of:
                T_of[j.parent] = T_of[cur] @ np.linalg.inv(j.origin())
                todo.append(j.parent)
    out = []
    for el in root.findall("link"):
        name = el.get("name")
        if name not in T_of:
            continue
        cols = el.findall("collision")
        if not cols:
            continue
        lo_all, hi_all, why = [], [], None
        for c in cols:
            org = c.find("origin")
            T = T_of[name] @ make_tf(_floats(org.get("xyz") if org is not None else None, (0.0, 0.0, 0.0)),
                                     _floats(org.get("rpy") if org is not None else None, (0.0, 0.0, 0.0)))
            g = c.find("geometry")
            g = list(g)[0] if g is not None and len(list(g)) else None
            b = None
            if g is None:
                why = "collision element without geometry"
            elif g.tag == "box":
                h = 0.5 * np.asarray(_floats(g.get("size"), (0.0, 0.0, 0.0)), dtype=np.float64)
                b = (-h, h)
            elif g.tag == "cylinder":
                r, ln = float(g.get("radius", "0")), float(g.get("length", "0"))
                b = (np.array([-r, -r, -0.5 * ln]), np.array([r, r, 0.5 * ln]))
            elif g.tag == "sphere":
                r = float(g.get("radius", "0"))
                b = (np.full(3, -r), np.full(3, r))
            elif g.tag == "mesh":
                fn = g.get("filename", "")
                p = mesh_path(fn) if mesh_path is not None else None
                if p is None or not fn.lower().endswith(".stl"):
                    why = f"mesh {fn} not readable here" + ("" if fn.lower().endswith(".stl") else " (not STL)")
                else:
                    try:
                        bb = _stl_bounds(p)
                    except OSError as e:
                        bb, why = None, f"mesh {fn}: {e}"
                    if bb is not None:
                        s = np.asarray(_floats(g.get("scale"), (1.0, 1.0, 1.0)), dtype=np.float64)
                        lo_s, hi_s = bb[0] * s, bb[1] * s
                        b = (np.minimum(lo_s, hi_s), np.maximum(lo_s, hi_s))
            else:
                why = f"geometry <{g.tag}> not supported"
            if b is None:
                continue
            corners = np.array([[x, y, z] for x in (b[0][0], b[1][0]) for y in (b[0][1], b[1][1]) for z in (b[0][2], b[1][2])])
            P = corners @ T[:3, :3].T + T[:3, 3]
            lo_all.append(P.min(axis=0))
            hi_all.append(P.max(axis=0))
        if why is not None or not lo_all:
            out.append({"link": name, "lo": None, "hi": None, "why": why or "no readable collision geometry"})
        else:
            out.append({"link": name, "lo": np.min(lo_all, axis=0), "hi": np.max(hi_all, axis=0)})
    return out


# ---------------------------------------------------------------------------
# What is mounted on a link: the collision geometry of the fixed-joint subtree below it (the tool on the flange)
# ---------------------------------------------------------------------------
def mounted_collision_points(urdf_source, mount_frame: str, *, mesh_path=default_mesh_path) -> dict:
    """The tool mounted on ``mount_frame`` as the robot model describes it: the collision geometry of every link below it through
    fixed joints only (children direction - the parts carried by that frame, not the arm it is attached to), as convex hull points
    in the ``mount_frame`` frame [m]. STL meshes contribute their hull vertices (with the collision origin and the mesh scale);
    box / cylinder / sphere contribute the corners of their box. urdf_source: a path or a file object.
    → {"points": (K, 3) or None, "links": [links with geometry], "why": [geometry that could not be read here]}. A convex object's
    lowest point in any direction is one of its hull vertices, so a height / plane distance check over these points equals the
    check over the full geometry."""
    from .tool_body import hull_points, load_stl_vertices
    root = ET.parse(urdf_source).getroot()
    joints = [j for j in parse_urdf_joints_root(root).values() if j.type == "fixed"]
    T_of = {mount_frame: np.eye(4)}
    todo = [mount_frame]
    while todo:
        cur = todo.pop()
        for j in joints:
            if j.parent == cur and j.child not in T_of:
                T_of[j.child] = T_of[cur] @ j.origin()
                todo.append(j.child)
    pts, links, why = [], [], []
    for el in root.findall("link"):
        name = el.get("name")
        if name not in T_of:
            continue
        got = False
        for c in el.findall("collision"):
            org = c.find("origin")
            T = T_of[name] @ make_tf(_floats(org.get("xyz") if org is not None else None, (0.0, 0.0, 0.0)),
                                     _floats(org.get("rpy") if org is not None else None, (0.0, 0.0, 0.0)))
            g = c.find("geometry")
            g = list(g)[0] if g is not None and len(list(g)) else None
            P = None
            if g is None:
                why.append(f"{name}: collision element without geometry")
            elif g.tag == "mesh":
                fn = g.get("filename", "")
                p = mesh_path(fn) if mesh_path is not None else None
                if p is None or not fn.lower().endswith(".stl"):
                    why.append(f"{name}: mesh {fn} not readable here" + ("" if fn.lower().endswith(".stl") else " (not STL)"))
                else:
                    try:
                        s = np.asarray(_floats(g.get("scale"), (1.0, 1.0, 1.0)), dtype=np.float64)
                        P = hull_points(load_stl_vertices(p) * s[None, :])
                    except (OSError, ValueError) as e:
                        why.append(f"{name}: mesh {fn}: {e}")
            elif g.tag in ("box", "cylinder", "sphere"):
                if g.tag == "box":
                    h = 0.5 * np.asarray(_floats(g.get("size"), (0.0, 0.0, 0.0)), dtype=np.float64)
                elif g.tag == "cylinder":
                    r, ln = float(g.get("radius", "0")), float(g.get("length", "0"))
                    h = np.array([r, r, 0.5 * ln])
                else:
                    h = np.full(3, float(g.get("radius", "0")))
                P = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])
            else:
                why.append(f"{name}: geometry <{g.tag}> not supported")
            if P is not None:
                pts.append(P @ T[:3, :3].T + T[:3, 3])
                got = True
        if got:
            links.append(name)
    return {"points": np.vstack(pts) if pts else None, "links": links, "why": why}
