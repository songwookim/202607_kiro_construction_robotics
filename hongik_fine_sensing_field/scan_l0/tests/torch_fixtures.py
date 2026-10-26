"""Torch fixtures of the tests (2026-10-13). The lab robot model is the weld torch turned 180 deg in its clamp (eoat.xacro; the virtual
arm's URDF scan_l0/tests/data): every test of the tool itself takes it from the model (tcp_text, the configs). A test whose subject is
a rule or a planner designed / tuned with the CAD torch geometry - not the tool - takes the CAD torch explicitly from here (like
kiro-1g's groove_world.LAB_TORCH_KEYPOINT): its keypoint (test_torch_model_consistency checks it is the CAD torch's wire end) and its
tool body mesh (the turn generator's input)."""
from __future__ import annotations

import io
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.dirname(os.path.dirname(HERE))
URDF = os.path.join(HERE, "data", "rb20_1900es_u_eoat.urdf")
CAD_TORCH_KEYPOINT = {"torch_lateral_m": [-0.0366, -0.0005], "torch_protrusion_m": 0.1924}
CAD_TOOL_BODY_MESH = "package://eoat_description/meshes/eoat_base_link_dustplate.STL"

__all__ = ["CAD_TORCH_KEYPOINT", "CAD_TOOL_BODY_MESH", "with_cad_tool_body", "tcp_mm", "tcp_text"]


def with_cad_tool_body(settings):
    """The settings with the CAD torch's tool body mesh (safety.tool_body_mesh) - the explicit CAD fixture."""
    settings.params["safety"] = dict(settings.params.get("safety") or {}, tool_body_mesh=CAD_TOOL_BODY_MESH)
    return settings


def tcp_mm(parent: str = "tcp", child: str = "torch_tcp") -> np.ndarray:
    """parent <- child position [mm] in the virtual arm's URDF (the lab robot model)."""
    from scan_core.fk_urdf import chain_from_joints, parse_urdf_joints
    T = np.eye(4)
    for j in chain_from_joints(parse_urdf_joints(io.StringIO(open(URDF, encoding="utf-8").read())), parent, child).joints:
        T = T @ j.origin()
    return T[:3, 3] * 1e3


def tcp_text() -> str:
    """'(x, y, z) mm' of flange <- torch_tcp as the D14 tool description prints it."""
    t = tcp_mm()
    return f"({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) mm"
