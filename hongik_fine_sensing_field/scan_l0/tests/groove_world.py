"""합성 그루브 세계 (오프라인 테스트 전용: 실로봇·실센서 없음).

로봇의 **현재 실측 플랜지 자세** (pose_fn() → T_base←flange: 가상 팔 sim_arm, MoveIt 가짜 하드웨어의 TF, 또는 시험의 PoseBox) 를
읽어, 그 자세에서 VC nano 3D-Z 가 보았을 그루브 시편 프로파일을 scan_core.synth 광선 추적으로 렌더링한다.
FakeSensor(frame_fn=world.frame) 에 꽂으면 센서 스트림이 로봇 자세를 따라간다.

기하 (모두 base 프레임, m):
  * 시편 = GrooveSpec 그루브 판 두 장. 루트갭 중심선이 S_true → E_true (판 상면 위 점), 판 법선 n_true.
    판 프레임 T_base←plate = plate_frame(S, u, n) (원점 S, x = u = unit(E−S), z = n, y = n×u). 판은 유한:
    x ∈ [0, |E−S|] (용접선 길이), y ∈ ±plate_len_m/2 (가로 폭 — 기본은 그 spec 의 plate_width_mm). 판 밖은 기본적으로 표면 없음(NaN) — 판 끝이 '무반환'으로
    관측된다. table_drop_m 을 주면 판 상면보다 그만큼 아래의 무한 테이블 평면을 함께 렌더링한다 (판 끝 너머 오버스캔에서 센서가
    아래 테이블을 보는 실기 상황; 2026-10-01 가드 거짓 정지 시험). extras = 판 프레임 높이맵 프리미티브 목록 (BoxPrim, PlanePrim,
    클램프·펜스·기운 평면·끝에 붙은 동일 평면 판·지그 — 판 끝 식별 시험). +x_spec(= 시편 단면 좌표, bevel_side 'left' 는
    −쪽) = 판 +y = n×u.
  * 센서 = T_base←opt = T_base←flange(pose_fn) · calib_true.T_flange←opt. 레이저 평면 = 광학 XZ 평면.
  * 진실값: true_root_polyline(s) = S + u·s − n·t (그루브 바닥 = 뒷댐재 상면 = 판 상면 − 두께),
    true_plate_top_height() = up·S.

세운 판 (3G): make_world_3g — 같은 렌더러, 판 법선이 수평(로봇 쪽) 이고 용접선이 중력 방향 (S 아래, E 위).

기울기 인식 (I2): 프로파일 z 는 광학축 거리일 뿐 높이가 아니다. 렌더링은 완전한 3D 광선 추적이므로 센서가 중력/판에
대해 얼마나 기울었든 자세에 맞는 프로파일이 나온다. 높이·거리 검증에는 point_base(x, z) (P_base = T_base←opt·[x,0,z,1])
와 up·P 를 쓴다 — 테스트에서도 raw z 를 높이로 쓰지 말 것.

자세 캐시: 플랜지 위치·회전 행렬을 1e-5 (0.01 mm) 로 반올림한 키로 무잡음 렌더를 캐시하고, 노이즈·드롭아웃은 프레임마다
(seed, i) 로 결정적으로 새로 뽑는다. (예전 가짜 제어박스의 tcp_ref / jnt_ang FK 자세 모드는 2026-10-07 제어박스 어댑터와 함께 제거.)

Capture instant (pose_at_fn, 2026-10-13): with FakeSensor(timed_frame_fn=True) frame(i, t_capture_ns) renders the pose AT the frame's
capture instant — pose_at_fn(t_capture_ns) (e.g. ros_harness.TfPoseAt: the stamped /joint_states stream interpolated at that instant,
a servo lag earlier) — instead of pose_fn() at the call: on a real clock the call comes a render, a pose sample period and a delivery
later, and the content then trails its stamp by a load-dependent amount (fake-arm e2e 2026-10-13: 20–40 ms, ±10 ms per frame).
"""
from __future__ import annotations

import dataclasses
import math
import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import yaml

from scan_core.frames import inv_transform, make_transform, rot_x, rot_y, rot_z, transform_points, unit
from scan_core.path_gen import plate_frame
from scan_core.specimen import GrooveSpec
from scan_core.synth import CompositeScene, GrooveScene, PlaneScene, SynthConfig, simulate_profile
from scan_core.types import Calib
from scan_core.validity import Z_VALID_MM_DEFAULT

__all__ = ["GrooveWorld", "make_world", "FAKE_SENSOR_EDGE_RESPONSE", "make_world_3g", "lab_spec", "LAB_SPECIMEN", "LAB_TORCH_KEYPOINT", "lab_job_doc", "true_root_polyline", "LAB_S_TRUE", "LAB_E_TRUE", "LAB_N_TRUE",
           "BoxPrim", "SwitchablePrim", "PlanePrim", "EndSagScene", "lateral_edge_clamps", "end_fixture", "slot_fixture", "GhostBand", "synthetic_pass",
           "LAB3G_S_TRUE", "LAB3G_E_TRUE", "LAB3G_N_TRUE", "JOB_YAML", "JOB_YAML_3G", "PoseBox"]

_HERE = os.path.dirname(os.path.abspath(__file__))
JOB_YAML = os.path.join(os.path.dirname(_HERE), "scan_l0", "jobs", "groove_10T_1G.yaml")

# the fake sensor's edge response (the 1G end model's sensor constants, calib g1: edge_response): this world is an exact ray cast —
# no laser-sheet spread, so its σ_L is below anything a scan resolves (the lower end of the end model's support). A host that
# reconstructs this world's profiles needs it in its calibration (sim_harness.calib_for_sensor, ros_harness.fake_sensor_calib_file):
# the real sensor's calibrated σ_L 0.47 mm read the exact steps 0.3–0.6 mm outward (fake-arm e2e / KIRO stand-in 2026-10-13)
FAKE_SENSOR_EDGE_RESPONSE = {"sigma_l_m": 3e-5, "sigma_l_log_sd": 0.5, "theta_tri_deg": 26.0, "theta_tri_sigma_deg": 2.0,
                             "source": "fake sensor: exact ray cast (no sheet spread)"}

# 연구실 기본 진실값 (the lab cell: seam 0.38 m along base +y, plate top z = 0.05 - the lab line's template placeholders)
LAB_S_TRUE = (0.60, -0.19, 0.05)
LAB_E_TRUE = (0.60, 0.19, 0.05)
# The lab specimen of this world - an explicit fixture of the virtual-arm sims (kiro-1g, 2026-10-13): the shipped 1G template
# (groove_10T_1G.yaml) carries KIRO's site specimen (200 mm x plates 150 mm); the lab cell's world above is the Hongik lab specimen
# (weld length 380 mm, plates 300 mm each - user 2026-10-09), and its design must match that world. Everything else (groove, process,
# precision, tolerances) comes from the template.
LAB_SPECIMEN = {"weld_length_mm": 380.0, "plate_widths_mm": [300.0, 300.0]}


def _lab_torch_keypoint() -> dict:
    """The lab arm's torch keypoint (the envelope's torch tip of the lab cell's jobs) = the lab line's template values, read from the
    lab 1G example job (example_1G.yaml: the virtual arm's URDF torch_tcp - the torch turned in its clamp since 2026-10-13 - + the CAD
    wire stub 20 mm; test_torch_model_consistency checks them). The shipped 1G template carries KIRO's wire end (URDF v2 r4 at the
    template's stick-out from the nozzle end, kiro-1g 2026-10-13)."""
    with open(os.path.join(os.path.dirname(_HERE), "scan_l0", "jobs", "example_1G.yaml"), "r", encoding="utf-8") as f:
        job = yaml.safe_load(f)["job"]
    return {"torch_lateral_m": [float(v) for v in job["torch_lateral_m"]], "torch_protrusion_m": float(job["torch_protrusion_m"])}


LAB_TORCH_KEYPOINT = _lab_torch_keypoint()
LAB_N_TRUE = (0.0, 0.0, 1.0)

# 3G (세운 판, vertical_3G): 용접선이 중력 방향 (base +z), 판 법선은 수평으로 로봇 베이스 쪽 (−y; 판은 y = 0.70 에 서 있다).
# S = 아래 점, E = 위 점 (vertical_direction='up' 규약: 아래 점 → 위 점, 리드인 없음, 리드아웃은 위 점 위쪽; 3G_floor = 아래 점 높이).
# groove_10T_3G.yaml 자리표시자와 같은 위치 (L = 0.38 m, 판 상단 모서리 높이 0.68 m = specimen_top_z_m).
JOB_YAML_3G = os.path.join(os.path.dirname(_HERE), "scan_l0", "jobs", "groove_10T_3G.yaml")
LAB3G_S_TRUE = (0.50, 0.70, 0.30)
LAB3G_E_TRUE = (0.50, 0.70, 0.68)
LAB3G_N_TRUE = (0.0, -1.0, 0.0)



# --------------------------------------------------------------------------------------
# 판 끝 너머의 물체 (판 프레임 높이맵: x = 심 방향 (S 에서 0, E 에서 L), y = n×u 측면, h = 판 상면 위 높이 [m]; 밖은 NaN)
# --------------------------------------------------------------------------------------
@dataclass
class BoxPrim:
    """직육면체 (평면 위 회전 yaw_deg, 중심 (xc, yc), 크기 sx × sy, 윗면 높이 h_top). 클램프 블록·펜스 판 (얇고 높은 상자)."""
    xc: float
    yc: float
    sx: float
    sy: float
    h_top: float
    yaw_deg: float = 0.0

    def height(self, xp, yp):
        a = math.radians(self.yaw_deg)
        dx = np.asarray(xp, dtype=float) - self.xc
        dy = np.asarray(yp, dtype=float) - self.yc
        lx = dx * math.cos(a) + dy * math.sin(a)
        ly = -dx * math.sin(a) + dy * math.cos(a)
        inside = (np.abs(lx) <= self.sx / 2.0) & (np.abs(ly) <= self.sy / 2.0)
        return np.where(inside, self.h_top, np.nan)


@dataclass
class SwitchablePrim:
    """A primitive present only while `on` - e.g. the weld bead that appears after a test's pre-weld run (post-weld scan). After
    switching, clear the world's render cache (GrooveWorld._cache); the render prefetcher's world fingerprint changes with `on`."""
    prim: object
    on: bool = False

    def height(self, xp, yp):
        if not self.on:
            return np.full(np.broadcast(np.asarray(xp, dtype=float), np.asarray(yp, dtype=float)).shape, np.nan)
        return self.prim.height(xp, yp)


@dataclass
class PlanePrim:
    """유한 평면 h = h0 + slope_x·(x − x0) + slope_y·y (x ∈ [x0, x1], y ∈ [y0, y1]): 테이블 조각·기운 판·끝에 붙은 동일 평면 판 (h0 = 0)."""
    x0: float
    x1: float
    y0: float
    y1: float
    h0: float = 0.0
    slope_x: float = 0.0
    slope_y: float = 0.0

    def height(self, xp, yp):
        xp = np.asarray(xp, dtype=float)
        yp = np.asarray(yp, dtype=float)
        inside = (xp >= self.x0) & (xp <= self.x1) & (yp >= self.y0) & (yp <= self.y1)
        return np.where(inside, self.h0 + self.slope_x * (xp - self.x0) + self.slope_y * yp, np.nan)


@dataclass
class EndSagScene:
    """판 끝 처짐 (2026-10-08 1G 배치 E 실패 4 건: −l 판의 E 끝 마지막 ~5 mm 가 0.6–1.3 mm 낮고 7–11° 기울었다 — 그 단면의 판 맞춤이
    두 판 각도 차로 plates_not_parallel 가 됐다). base 장면 (보통 GrooveScene) 의 높이에서 구역 [x0, x1] × [y0, y1] (판 프레임) 안만
    깎는다: sag(x) = max(0, drop_m − (x1 − x)·tan(slope_deg)) — 끝 x1 에서 drop_m 낮고 slope_deg 로 기운 면 (구역 시작에서는 남은 계단).
    CompositeScene 은 장면들의 최댓값이라 처짐 (낮아짐) 을 더할 수 없다 — 그래서 base 를 감싸는 장면으로 쓴다 (GrooveWorld(scene_wrap=...))."""
    base: object
    x0: float
    x1: float
    y0: float
    y1: float
    drop_m: float
    slope_deg: float
    at_x0: bool = False                     # True: 처진 끝이 x0 (S 끝 쪽 구역) — sag(x) = max(0, drop − (x − x0)·tan)

    def height(self, xp, yp):
        h = np.asarray(self.base.height(xp, yp), dtype=float)
        xp = np.asarray(xp, dtype=float)
        yp = np.asarray(yp, dtype=float)
        inside = (xp >= self.x0) & (xp <= self.x1) & (yp >= self.y0) & (yp <= self.y1)
        dist = (xp - float(self.x0)) if self.at_x0 else (float(self.x1) - xp)
        sag = np.maximum(0.0, float(self.drop_m) - dist * math.tan(math.radians(float(self.slope_deg))))
        return np.where(inside, h - sag, h)


def lateral_edge_clamps(L: float, half_width_m: float, h: float = 0.025, *, inset_m: float = 0.030) -> list:
    """연구실 1G 시편 클램프 (사용자 2026-10-09): 시편을 그 **바깥 옆 가장자리** (경로에 평행한 판 가장자리 — 심에서 시편 폭의 반,
    half_width_m; 연구실 시편 전체 폭 600 mm → ~300 mm) 에서 누른다. 그루브 경로 위·경로 양끝 근처에는 아무것도 없다 (실기 자료의
    E 쪽 '둥근 물체' ~140 mm 옆은 클램프가 아니다; 예전 ±130–150 mm 는 추정이었고 틀렸다). 판 프레임 높이맵 BoxPrim 넷: 옆
    [half_width − 20 mm, half_width], 끝 모서리에서 inset 안쪽 (x = inset, L − inset; 40 mm 길이), 판 상면 위 h."""
    sy = 0.020
    yc = float(half_width_m) - 0.5 * sy
    return [BoxPrim(xc=xc, yc=sg * yc, sx=0.040, sy=sy, h_top=float(h)) for xc in (float(inset_m), float(L) - float(inset_m))
            for sg in (-1.0, 1.0)]


def end_fixture(L: float, at: str = "both", *, lateral_m: float = 0.067, beyond_m: float = 0.006, h: float = 0.018) -> list:
    """일반 '모르는 고정물' 시나리오 (클램프가 아니다): 경로 끝 바로 너머의 구조물. 실기 근거 (2026-10-08 보고서): 연구실 S 끝 너머 5–7 mm,
    옆 ~64–70 mm, 판 위 최대 ~17–20 mm 의 정체 모를 구조 (사용자: 클램프 아님). at = 'S' | 'E' | 'both' (끝마다 옆 ±lateral 두 개, 10 × 12 mm,
    판 상면 위 h) — 첫 드웰처럼 레이저 선이 끝을 넘는 읽기에서만 보여야 하고 (기록만), 확인된 드웰 (S_trk 뒤) 의 선은 닿지 않아야 한다."""
    sx, sy = 0.010, 0.012
    xs = {"S": [-(float(beyond_m) + 0.5 * sx)], "E": [float(L) + float(beyond_m) + 0.5 * sx]}
    xs["both"] = xs["S"] + xs["E"]
    return [BoxPrim(xc=xc, yc=sg * float(lateral_m), sx=sx, sy=sy, h_top=float(h)) for xc in xs[at] for sg in (-1.0, 1.0)]


def slot_fixture(L: float, *, gap_m: float = 0.003, length_m: float = 0.040, slot_m: float = 0.005, depth_m: float = 0.010,
                 half_width_m: float = 0.15) -> list:
    """A fixture at plate height beyond the E end (판 프레임 높이맵): two coplanar blocks (top = plate top) whose slot of width slot_m and
    depth depth_m lies on the seam line — a run-off-tab-like jig butted against the plate end with a gap gap_m, length length_m. Seen by
    a pass it is a groove-like profile at the fused line (the template reads the root on the line); only the gap and the plate-top
    presence tell it from the seam (the seam ends where BOTH plates are present, user 2026-10-09)."""
    x0 = float(L) + float(gap_m)
    xc = x0 + 0.5 * float(length_m)
    yb = 0.5 * float(slot_m) + 0.5 * float(half_width_m)
    return [BoxPrim(xc=xc, yc=yb, sx=float(length_m), sy=float(half_width_m), h_top=0.0),
            BoxPrim(xc=xc, yc=-yb, sx=float(length_m), sy=float(half_width_m), h_top=0.0),
            BoxPrim(xc=xc, yc=0.0, sx=float(length_m), sy=float(slot_m), h_top=-float(depth_m))]


@dataclass
class GhostBand:
    """A deterministic ghost in a zone of the seam (lab at07 / at06: an interreflection band under the lower bevel on a shiny stretch):
    columns whose true return lies in the lowest band_m of the groove (backing in the gap + the foot of the bevel and wall) report a point
    depth_m further along their ray instead — the profile shows the groove bottom depth_m too deep. s0, s1 = zone along the seam from S
    [m]. Same for every pose and frame (geometry, not noise)."""
    s0: float
    s1: float
    depth_m: float = 0.0025
    band_m: float = 0.003

    def apply(self, world: "GrooveWorld", x, z, T):
        from scan_core.groove_model import RayFan
        xd = np.asarray(x, dtype=float)
        zd = np.asarray(z, dtype=float)
        Pb = world.point_base(xd, zd, T)
        s = (Pb - world.S) @ world.u
        h = (Pb - world.S) @ world.n
        t = world.spec.thickness_mm * 1e-3
        with np.errstate(invalid="ignore"):
            on = (np.isfinite(xd) & (s >= self.s0) & (s <= self.s1) & (h < -t + self.band_m) & (h > -t - 1e-3))
        z0 = RayFan.from_validity().z0
        d = np.stack([xd, zd - z0], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
        x2, z2 = xd.copy(), zd.copy()
        x2[on] = xd[on] + 1e3 * self.depth_m * d[on, 0]
        z2[on] = zd[on] + 1e3 * self.depth_m * d[on, 1]
        return x2.astype(np.float32), z2.astype(np.float32)


class PoseBox:
    """시험용 플랜지 자세 보관함 — pose_fn 으로 넘긴다 (box.T 를 바꾸면 다음 프레임이 그 자세에서 렌더된다)."""

    def __init__(self, T=None):
        self.T = np.eye(4) if T is None else np.asarray(T, dtype=float).reshape(4, 4)

    def __call__(self) -> np.ndarray:
        return self.T


def lab_job_doc(three_g: bool = False) -> dict:
    """The job template document of the lab cell's world: groove_10T_1G.yaml with the lab specimen (LAB_SPECIMEN: the 1G world
    LAB_S_TRUE..LAB_E_TRUE is 380 mm long) and the lab arm's torch keypoint (LAB_TORCH_KEYPOINT), or groove_10T_3G.yaml as it is
    (380 x 300, the lab torch)."""
    with open(JOB_YAML_3G if three_g else JOB_YAML, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not three_g:
        doc["specimen"].update(LAB_SPECIMEN)
        doc["job"].update(LAB_TORCH_KEYPOINT)
    return doc


def lab_spec(bevel_side: str = "left", *, three_g: bool = False) -> GrooveSpec:
    """The lab world's specimen (lab_job_doc: the template's groove 10T, 30 deg, gap 5, backing 10T with the lab specimen 380 x 300 +
    300 - user 2026-10-09; three_g: groove_10T_3G.yaml, 380 x 300) with bevel_side changed. The test world and the rough job the test
    uses (sim_harness.write_rough_job - the same document) have the same design."""
    spec = GrooveSpec.from_dict(lab_job_doc(three_g)["specimen"])
    return dataclasses.replace(spec, bevel_side=bevel_side)


def true_root_polyline(S_true, E_true, n_true, spec: GrooveSpec, s_values) -> np.ndarray:
    """진행좌표 s [m] (S 에서 0) → 그루브 바닥(뒷댐재 상면) 중심선 base 점 (N,3): S + u·s − n·thickness."""
    S = np.asarray(S_true, dtype=float).reshape(3)
    E = np.asarray(E_true, dtype=float).reshape(3)
    n = unit(n_true)
    u = unit(E - S)
    s = np.asarray(s_values, dtype=float).reshape(-1)
    return S[None, :] + s[:, None] * u[None, :] - (spec.thickness_mm * 1e-3) * n[None, :]


class GrooveWorld:
    """실측 플랜지 자세 (pose_fn) → 그루브 시편 프로파일 렌더러 (모듈 docstring). pose_fn 이 None 이면 render() / synthetic_pass 만."""

    def __init__(self, spec: GrooveSpec, S_true, E_true, n_true, calib_true: Calib, *,
                 plate_len_m: Optional[float] = None, sigma_z_mm: float = 0.02, dropout: float = 0.01, seed: int = 0,
                 n_points: int = 1408, z_valid_mm: tuple = Z_VALID_MM_DEFAULT, up=(0.0, 0.0, 1.0), bisect_iters: int = 16,
                 table_drop_m: Optional[float] = None, extras: Optional[list] = None, pose_fn=None,
                 support_height_m: Optional[float] = None, scene_wrap=None, ghost: Optional[GhostBand] = None,
                 pose_at_fn=None):
        self.pose_fn = pose_fn
        self.pose_at_fn = pose_at_fn         # T_base<-flange at a capture instant (frame(i, t_capture_ns); module doc)
        self.ghost = ghost
        self.spec = spec
        self.S = np.asarray(S_true, dtype=float).reshape(3).copy()
        self.E = np.asarray(E_true, dtype=float).reshape(3).copy()
        self.n = unit(n_true)
        d = self.E - self.S
        self.L = float(np.linalg.norm(d))
        if self.L <= 0.0:
            raise ValueError("E_true must differ from S_true")
        self.u = d / self.L
        if abs(float(self.u @ self.n)) > math.sin(math.radians(5.0)):
            raise ValueError("E_true - S_true is not perpendicular to n_true")
        self.calib = calib_true
        self.plate_len_m = float(spec.plate_width_mm) * 1e-3 if plate_len_m is None else float(plate_len_m)   # 기본: 설계 폭
        self.sigma_z_mm = float(sigma_z_mm)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.up = unit(up)
        self.T_base_plate = plate_frame(self.S, self.u, self.n)
        W = self.plate_len_m
        self.scene = GrooveScene(spec=spec, yc=0.0, angle_deg=0.0, h_top=0.0, x_range=(0.0, self.L),
                                 y_range=(-W / 2.0, W / 2.0), outside=math.nan)
        self.table_drop_m = None if table_drop_m is None else float(table_drop_m)
        # 3G 시편이 서 있는 수평 테이블 (base up 높이; 2026-10-07 범용 시작의 지지면 측정): 판에 수직인 무한 수평면 — 광선마다 교점을 풀어
        # 시편보다 가까우면 그것을 쓴다 (시편이 앞을 가리면 시편)
        self.support_height_m = None if support_height_m is None else float(support_height_m)
        self.extras = list(extras or [])
        if scene_wrap is not None:
            # 시편 장면을 감싸는 장면 (EndSagScene 등: 판 끝 처짐처럼 시편 높이를 낮추는 것 — 최댓값 합성으로는 못 한다)
            self.scene = scene_wrap(self.scene)
        layers = [self.scene] + self.extras
        if self.table_drop_m is not None:
            # 판 상면 아래 table_drop_m 의 무한 테이블 (판 프레임 z = −drop). 판·뒷댐재 (두께 t + 뒷댐재) 가 그 위에 놓여 있다.
            layers.append(PlaneScene(h0=-self.table_drop_m))
        if len(layers) > 1:
            self.scene = CompositeScene(scenes=layers)
        # 무잡음 렌더: 노이즈·드롭아웃은 frame() 에서 프레임별로 넣는다 (캐시는 자세만의 함수)
        # bisect 16 회 → 격자 2 mm / 2^16 ≈ 3e-5 mm (노이즈 0.02 mm 보다 훨씬 작다), 렌더 ~25 ms/자세
        self.cfg = SynthConfig(n_points=int(n_points), sigma_z_mm=0.0, dropout_prob=0.0, z_valid_mm=tuple(z_valid_mm),
                               seed=self.seed, bisect_iters=int(bisect_iters))
        self._cache: dict = {}
        self._lock = threading.Lock()
        self.n_frames = 0            # frame() 호출 수
        self.n_rendered = 0          # 캐시 미스(실제 광선 추적) 수
        self.last_T_base_flange: Optional[np.ndarray] = None
        self._prefetch = None        # 렌더 선계산 (render_prefetch.RenderPrefetcher; sim_cell 이 가상 시계에서만 붙인다)

    # ------------------------------------------------------------------ 진실값
    @property
    def lateral(self) -> np.ndarray:
        """판 +y = n×u (= +x_spec 방향). 센서 +X 와의 관계는 캘리브 가지에 따른다: 이전 초기값 (−π/2) 에서는 공칭 자세의 센서 +X,
        확정값 (+π/2, 2026-10-02) 에서는 −X — 센서 프레임 베벨 라벨을 기대할 때는 X_opt·lateral 로 계산할 것."""
        return np.cross(self.n, self.u)

    def true_root_polyline(self, s_values) -> np.ndarray:
        return true_root_polyline(self.S, self.E, self.n, self.spec, s_values)

    def true_plate_top_height(self) -> float:
        """판 상면 높이 = up·S (S, E 는 판 상면 위 점). 1G 전용 — 3G 에서는 높이가 아니라 plate_distance 가 바닥이다."""
        return float(self.up @ self.S)

    @property
    def vertical(self) -> bool:
        """세운 판(3G): 용접선이 중력축과 나란하다 (|u·up| ≥ cos 5°)."""
        return abs(float(self.u @ self.up)) >= math.cos(math.radians(5.0))

    def lower_point_height(self) -> float:
        """3G 바닥 기준: 진실 용접선 양끝 중 낮은 점의 높이 up·P."""
        return float(min(self.up @ self.S, self.up @ self.E))

    def seam_point(self, s: float) -> np.ndarray:
        return self.S + self.u * float(s)

    # ------------------------------------------------------------------ 자세 읽기
    def read_flange_pose(self, t_capture_ns: Optional[int] = None) -> Tuple[np.ndarray, tuple]:
        """pose_fn() → (T_base←flange, 캐시 키 — 위치·회전 1e-5 반올림). With a capture instant and pose_at_fn: the pose at that
        instant (module doc)."""
        if t_capture_ns is not None and self.pose_at_fn is not None:
            T = np.asarray(self.pose_at_fn(int(t_capture_ns)), dtype=float).reshape(4, 4)
        elif self.pose_fn is None:
            raise ValueError("this world has no pose_fn (render() / synthetic_pass only)")
        else:
            T = np.asarray(self.pose_fn(), dtype=float).reshape(4, 4)
        key = tuple(np.round(T[:3, 3], 5).tolist()) + tuple(np.round(T[:3, :3].ravel(), 5).tolist())
        return T, key

    def T_base_flange(self) -> np.ndarray:
        return self.read_flange_pose()[0]

    def T_base_opt(self) -> np.ndarray:
        return self.T_base_flange() @ self.calib.T_flange_opt

    # ------------------------------------------------------------------ 렌더링
    def render(self, T_base_opt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """광학 자세 1개 → 무잡음 (x_mm, z_mm) float32. 표면 없음/교정범위 밖 = NaN (x, z 모두).

        synth 높이맵의 판은 바닥이 없는 기둥이라 판 가장자리(끝·측면)를 비스듬히 지나는 광선이 임의 깊이의 '측벽'에 맞는다.
        실물 측면은 두께(판 t + 뒷댐재)만큼만 있으므로, 맞은 점의 판 프레임 z 가 −(t + 뒷댐재) 보다 낮으면 무반환(NaN) 처리
        (테이블은 렌더링하지 않는다). 판 프레임 z 는 base 3D 점에서 계산한다 (raw z 아님)."""
        T = np.asarray(T_base_opt, dtype=float)
        x, z = simulate_profile(T, self.scene, self.T_base_plate, self.cfg)
        x = np.asarray(x, dtype=np.float32).copy()
        z = np.asarray(z, dtype=np.float32).copy()
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(z) | (z <= 0.0)      # synth 의 z=0 센티널(교정범위 밖)도 NaN 으로
        x[bad] = np.nan
        z[bad] = np.nan
        solid_mm = self.spec.thickness_mm + (self.spec.backing_thickness_mm if self.spec.backing else 0.0)
        if self.table_drop_m is not None:
            solid_mm = max(solid_mm, self.table_drop_m * 1e3)      # 테이블 아래는 없다 (테이블 자체는 남긴다)
        Pb = self.point_base(x, z, T)
        depth = self.plate_distance(Pb)                               # 판 상면 기준 법선 거리 (음수 = 판 속)
        q = (Pb - self.S) @ np.column_stack([self.u, self.lateral])   # 판 프레임 (x, y)
        W = self.plate_len_m
        with np.errstate(invalid="ignore"):
            # 시편 기둥의 옆면 (판 끝·측면) 만 두께로 자른다 — 시편 밖 물체 (extras) 는 그대로
            on_wall = (q[:, 0] >= -1e-4) & (q[:, 0] <= self.L + 1e-4) & (np.abs(q[:, 1]) <= W / 2.0 + 1e-4)
            under = np.isfinite(depth) & (depth < -(solid_mm * 1e-3 + 1e-6)) & on_wall
        x[under] = np.nan
        z[under] = np.nan
        if self.ghost is not None:
            x, z = self.ghost.apply(self, x, z, T)
        if self.support_height_m is not None:
            x, z = self._with_support(T, x, z)
        return x, z

    def _with_support(self, T: np.ndarray, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """수평 테이블 (up·p = support_height) 과 열 광선 (광학 프레임 x = k(a + b z)) 의 교점 — 더 가까우면 그 점."""
        from scan_core.validity import dx_linear_coeffs
        a_, b_ = dx_linear_coeffs()
        N = x.size
        k = (np.arange(N, dtype=float) - (N - 1) / 2.0) * (1408.0 / N)
        o, X, Z = T[:3, 3], T[:3, 0], T[:3, 2]
        uo, uX, uZ = float(self.up @ o), float(self.up @ X), float(self.up @ Z)
        den = (uX * k * b_ + uZ) * 1e-3
        with np.errstate(divide="ignore", invalid="ignore"):
            zt = (self.support_height_m - uo - uX * k * a_ * 1e-3) / den
        zlo, zhi = self.cfg.z_valid_mm
        ok = np.isfinite(zt) & (zt > zlo) & (zt < zhi)
        nearer = ok & (~np.isfinite(z) | (zt < z))
        x = x.copy()
        z = z.copy()
        x[nearer] = (k[nearer] * (a_ + b_ * zt[nearer])).astype(np.float32)
        z[nearer] = zt[nearer].astype(np.float32)
        return x, z

    def clean_frame(self, t_capture_ns: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """현재 CB 자세의 무잡음 프로파일 (캐시). 미스면 선계산 (있으면: 정확히 이 자세를 미리 렌더한 것) 또는 동기 렌더 — 같은 내용.
        t_capture_ns: the pose at that capture instant (pose_at_fn)."""
        T_bf, key = self.read_flange_pose(t_capture_ns)
        with self._lock:
            hit = self._cache.get(key)
            if hit is None:
                pf = self._prefetch
                hit = pf.take(T_bf) if pf is not None else None
                if hit is None:
                    hit = self.render(T_bf @ self.calib.T_flange_opt)
                self._cache[key] = hit
                self.n_rendered += 1
            self.last_T_base_flange = T_bf
        return hit

    def frame(self, i: int, t_capture_ns: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """FakeSensor.frame_fn: 프레임 i → 현재 CB 자세의 프로파일 + 노이즈 N(0, sigma_z) + 드롭아웃(NaN). 결정적 (seed, i).
        FakeSensor(timed_frame_fn=True): the profile at the capture instant t_capture_ns (pose_at_fn; module doc)."""
        x0, z0 = self.clean_frame(t_capture_ns)
        x = x0.copy()
        z = z0.copy()
        rng = np.random.default_rng([self.seed, int(i)])
        if self.sigma_z_mm > 0.0:
            z = z + rng.normal(0.0, self.sigma_z_mm, z.size).astype(np.float32)
        if self.dropout > 0.0:
            drop = rng.random(z.size) < self.dropout
            x[drop] = np.nan
            z[drop] = np.nan
        with self._lock:
            self.n_frames += 1
        return x.astype(np.float32), z.astype(np.float32)

    # ------------------------------------------------------------------ 기울기 인식 도우미 (I2)
    def point_base(self, x_mm, z_mm, T_base_opt: Optional[np.ndarray] = None) -> np.ndarray:
        """프로파일 점 → base 3D 점 (N,3) [m]: P = T_base←opt·[x, 0, z, 1]·1e-3. 무효 점은 NaN 행."""
        T = self.T_base_opt() if T_base_opt is None else np.asarray(T_base_opt, dtype=float)
        x = np.asarray(x_mm, dtype=float).reshape(-1)
        z = np.asarray(z_mm, dtype=float).reshape(-1)
        P_opt = np.column_stack([x, np.zeros_like(x), z]) * 1e-3
        return transform_points(T, P_opt)

    def height(self, P_base) -> np.ndarray:
        """base 점 → 높이 up·P (raw z 가 아니라 3D 점으로 잰다)."""
        return np.asarray(P_base, dtype=float) @ self.up

    def plate_distance(self, P_base) -> np.ndarray:
        """base 점 → 판 상면까지 법선 거리 n·(P − S) (3G 용)."""
        return (np.asarray(P_base, dtype=float) - self.S) @ self.n

    # ------------------------------------------------------------------ 자세 만들기 (테스트 편의)
    def opt_pose(self, s: float, standoff_m: float = 0.34, *, lateral_m: float = 0.0, yaw_deg: float = 0.0,
                 pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
        """공칭 광학 자세 T_base←opt: R = [n×u, u, −n] · Rz(yaw) · Rx(pitch) · Ry(roll), 광학 Z 축이 점
        S + u·s + (n×u)·lateral 을 standoff 거리에서 향한다 (원점 = 그 점 − standoff·Z_opt: 기울여도 심 점이 시야 중앙에
        남도록 원점이 standoff 호 위를 움직인다, path_gen 입사 기울기와 같은 규약).
        yaw: 광학 Z(= −n) 둘레 (레이저 선이 판 안에서 돌아 심을 비스듬히 가로지름), pitch: 광학 X(라인축) 둘레 (광학축이
        진행방향으로 기움), roll: 광학 Y(= u) 둘레 (레이저 선이 판에 대해 기움 → 프로파일에 기울기)."""
        R = np.column_stack([self.lateral, self.u, -self.n]) @ rot_z(math.radians(yaw_deg)) \
            @ rot_x(math.radians(pitch_deg)) @ rot_y(math.radians(roll_deg))
        target = self.seam_point(s) + self.lateral * float(lateral_m)
        o = target - float(standoff_m) * R[:, 2]
        return make_transform(R, o)

    def flange_for_opt(self, T_base_opt: np.ndarray) -> np.ndarray:
        """T_base←opt → 그 광학 자세의 플랜지 자세 T_base←flange (진짜 캘리브 기준) — PoseBox 에 넣는다."""
        return np.asarray(T_base_opt, dtype=float) @ inv_transform(self.calib.T_flange_opt)


def synthetic_pass(world: "GrooveWorld", job, calib: Calib, s_values, *, seed: int = 0, z_valid_mm: tuple = Z_VALID_MM_DEFAULT):
    """가짜 CB 없이 패스 하나를 직접 렌더링 → ScanResult (판 끝 식별 단위 시험용). 광학 자세 = sensor_frames(job, calib).T_base_opt(s)
    (job 매개변수 s = 광학 원점의 심 점), 노이즈 N(0, sigma_z) + 드롭아웃 (world 설정, 프로파일마다 결정적), valid = 유한·z 창 안."""
    from scan_core.path_gen import sensor_frames
    from scan_core.types import ScanResult
    fr = sensor_frames(job, calib)
    s_values = np.asarray(s_values, dtype=float).reshape(-1)
    N = s_values.size
    X = np.full((N, world.cfg.n_points), np.nan, dtype=np.float32)
    Z = np.full((N, world.cfg.n_points), np.nan, dtype=np.float32)
    T_bf = np.stack([fr.T_base_flange(s) for s in s_values])
    T_bo = T_bf @ calib.T_flange_opt
    for i in range(N):
        x, z = world.render(T_bf[i] @ world.calib.T_flange_opt)        # 세계의 진짜 광학 자세
        x = x.copy()
        z = z.copy()
        rng = np.random.default_rng([seed, i])
        if world.sigma_z_mm > 0.0:
            z = z + rng.normal(0.0, world.sigma_z_mm, z.size).astype(np.float32)
        if world.dropout > 0.0:
            drop = rng.random(z.size) < world.dropout
            x[drop] = np.nan
            z[drop] = np.nan
        X[i], Z[i] = x, z
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(X) & np.isfinite(Z) & (Z > z_valid_mm[0]) & (Z < z_valid_mm[1])
    t = np.arange(N, dtype=float) * 0.01
    return ScanResult(x_mm=X, z_mm=Z, valid=valid, result_counter=np.arange(N, dtype=np.int64), ts_ms=(t * 1e3).astype(np.int64),
                      t_rx_s=t, t_acq_s=t, T_base_flange=T_bf, T_base_opt=T_bo, in_cruise=np.ones(N, dtype=bool),
                      unsynced=np.zeros(N, dtype=bool), ts_flag=np.zeros(N, dtype=bool), speed_mps=np.full(N, 0.03),
                      s_m=s_values.copy(), T_base_plate=fr.T_base_plate)


def make_world_3g(calib_true: Calib, *, bevel_side: str = "left", **overrides) -> GrooveWorld:
    """세운 판(3G) 세계: S/E/n = LAB3G_*_TRUE (S 아래, E 위, n 수평·로봇 쪽), spec = lab_spec(bevel_side, three_g=True). 렌더는 1G 와 같은 완전한
    3D 광선 추적이다 (판 프레임 x = u = 위쪽, y = n×u, z = n) — 판 위/아래 끝 밖은 무반환(NaN), 판 거리 = plate_distance (n 방향)."""
    kw = dict(spec=lab_spec(bevel_side, three_g=True), S_true=LAB3G_S_TRUE, E_true=LAB3G_E_TRUE, n_true=LAB3G_N_TRUE)
    kw.update(overrides)
    w = make_world(calib_true, bevel_side=bevel_side, **kw)
    if abs(float(w.n @ w.up)) > math.sin(math.radians(5.0)) or abs(float(w.u @ w.up)) < math.cos(math.radians(5.0)):
        raise ValueError("make_world_3g: the plate normal must be horizontal and the seam vertical")
    return w


def make_world(calib_true: Calib, *, bevel_side: str = "left", **overrides) -> GrooveWorld:
    """연구실 기본값 세계: S/E/n = LAB_*_TRUE, spec = lab_spec(bevel_side). overrides 로 GrooveWorld 인자 덮어쓰기."""
    kw = dict(spec=lab_spec(bevel_side), S_true=LAB_S_TRUE, E_true=LAB_E_TRUE, n_true=LAB_N_TRUE)
    kw.update(overrides)
    spec = kw.pop("spec")
    S, E, n = kw.pop("S_true"), kw.pop("E_true"), kw.pop("n_true")
    return GrooveWorld(spec, S, E, n, calib_true, **kw)
