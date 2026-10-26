"""용접 계획의 토치 모형 — 플랫폼 설정 `weld_torch:` 섹션. 용접 토치 = 우리 1차 브라켓 토치 (사용자·KIRO 합의 2026-10-12: KIRO 에 실제로
달린 브라켓도 우리 1차 브라켓 — 연구실 EOAT 와 같은 브라켓·토치·VC nano 3D-Z, 같은 RB20 팔. KIRO URDF 의 2차 브라켓·치즐링 건은 만든 적이
없는 부품이라 빠지고 우리 브라켓·토치·센서 모형이 들어간다).

용접 계획의 끝 틀 (와이어 끝 = 비드 겨냥점; [x 진행, y = z × x, z 토치 축 (공작물 쪽)], WPS 진행각·작업각) 은 토치와 무관하고, 이 모형이
그 틀을 그 플랫폼의 TCP 와 플랜지에 붙인다:

  TCP ← 끝 틀       회전 = 설정 tip_in_tcp.axes (우리 토치: 항등 — TCP z = 노즐 축 (공작물 쪽), x = 굽힘 면 안),
                    끝 위치 = 토치 축을 따라 (노즐 끝 + stick_out): stick_out = 가스 노즐 끝 → 와이어 끝 (용접 관행의 돌출; 사용자
                    2026-10-13) — 공정 값 job weld_plan.stick_out_mm (WeldProcess), 노즐 끝 = TCP 에서 토치 축으로 잰 가스 노즐 끝
                    (nozzle_end_mm: 이 팔의 토치면 그 로봇 URDF 충돌 형상에서 유도 — weld_clearance.nozzle_end_mm, 우리 토치 CAD 1.2 mm:
                    컨택트 팁 끝이 노즐 안으로 1.2 mm 들어가 있다; 이 팔에 없는 토치면 설정 nozzle_end_mm). 사용자 2026-10-12 밤: KIRO
                    실행기는 와이어 돌출을 적용하지 않는다 → 우리 TCP 자세가 겨냥점에서 −z_TCP 로 (stick_out + 노즐 끝) 만큼 물러난다
  플랜지 ← TCP      flange_from_tcp  (CAD: 연구실 플랜지 (rbpodo tcp = link6 + 0.13 m) 기준 (−0.45, +46.55, +297.17) mm, rpy (0, −30, +90)°,
                                       KIRO (identified 2026-10-13, URDF v2 r4): the same mounting with the torch turned 180 deg in
                                       its clamp -> (-0.45, +237.86, +297.17) mm, rpy (0, -30, -90) deg; eoat.xacro torch_tcp)
  토치 롤           TCP 를 토치 축 둘레로 0° 또는 180° (TCP y 가 심을 가로지르는 두 롤 — KIRO 실행기는 TCP y 로 위빙한다): 계획마다 하나 (모든
                    패스 같은 롤), 토치 자세 고르기와 함께 고른다 (scan_core.weld_orient.choose_orientation: 모든 패스가 되는 롤 가운데 WPS 에서
                    벗어난 각의 합이 작은 것, 같으면 관절 한계 여유가 큰 것) — WeldPlan.torch_roll_deg.

TCP = 토치 팁 끝 중심 = CAD 컨택트 팁 끝 (KIRO 답 2026-10-12 밤: KIRO 의 TCP 는 토치 팁 끝, 와이어 끝이 아니다) — 공칭값. CAD 의 모형 와이어
토막 (컨택트 팁 끝에서 20.0 mm) 은 포락선 키포인트의 공칭 형상일 뿐 계획 값이 아니다 (계획은 노즐 끝 + 공정의 stick_out_mm).
연구실 팔에도 이 토치가 달려 있다 (on_this_arm true: TCP = URDF torch_tcp). 이 팔의 토치 형상은 그 밖에 충돌 형상 (URDF EOAT 메시) 과
포락선 키포인트 (작업 torch_lateral_m·safety.torch_protrusion_m = 토치의 가장 먼 점 = CAD 와이어 끝) 에도 쓰인다.

D14 (용접 계획의 로봇 확인, 모션 없음; 사용자 결정 2026-10-12): 용접 토치가 이 팔에 있을 때만 (on_this_arm true: 연구실·KIRO) 돈다 — 도구 =
reach_tool: 그 로봇 자신의 TF 의 플랜지 ← TCP (그 로봇의 URDF 에 토치 TCP 가 있다) · TCP ← 끝 (stick_out·롤 포함). 용접 토치가 이 팔에
없다고 한 설정이면 (on_this_arm false) D14 를 하지 않고 계획을 납품하며, 계획 요약·실행 기록·weld_plan.json 의 weld_torch 블록·kiro/
info·README 에 D14_NOT_ON_THIS_ARM 을 적는다 (d14_on_this_arm; 다른 토치로 대신하지 않는다; 롤은 기본 0°). 설정의 flange_from_tcp 는 그
토치가 플랜지 (flange_frame) 어디에 붙는지의 기록 — 정보 파일·README·weld_plan.json, 가상 팔 (URDF 에 그 TCP 가 없을 때).
(2026-10-12 앞의 'KIRO 토치' 모형 — right_manipulator_ee_point 를 KIRO 로그의 FK 맞춤 (−6.46, +145.20, +329.17) mm 로, 토치 축 −x_ee —
은 폐기: 그 ee_point 는 만든 적이 없는 치즐링 건의 것이고, KIRO 로그는 이 토치로 기록되지 않았다.)
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .g1_types import jsonable
from .rigid import RigidTransform, transform_from_axes, transform_from_quat

__all__ = ["SECTION", "WeldTorch", "weld_torch_config", "reach_tool", "d14_on_this_arm", "D14_NOT_ON_THIS_ARM", "D14_NO_WELD_TORCH",
           "ROLLS_DEG", "roll_matrix", "plan_weld_torch", "torch_with_nozzle_end"]

SECTION = "weld_torch"
# D14 을 하지 않을 때의 기록 문구 (설정이 용접 토치가 이 팔에 없다고 할 때 — 다른 토치로 대신하지 않는다)
D14_NOT_ON_THIS_ARM = ("weld-plan reachability (D14) not checked: the platform config says the weld torch is not mounted on this arm "
                       "(weld_torch.on_this_arm false); it is checked on the arm that carries it")
D14_NO_WELD_TORCH = ("weld-plan reachability (D14) not checked: the platform config has no weld torch model (weld_torch); it is checked "
                     "on the welding robot")


# TCP 의 토치 축 둘레 롤 후보 [deg]: TCP y 가 심을 가로지르는 두 롤 (x = +진행 / −진행, 굽힘 면 안) — KIRO 실행기는 TCP y 로 위빙한다
# (사용자 2026-10-12 밤). 계획마다 하나 (scan_core.weld_orient.choose_orientation).
ROLLS_DEG = (0.0, 180.0)


def roll_matrix(roll_deg: float) -> np.ndarray:
    """끝 틀 z (토치 축) 둘레 회전 Rz(roll) (3×3)."""
    c, s_ = math.cos(math.radians(float(roll_deg))), math.sin(math.radians(float(roll_deg)))
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _flag(v, where: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ValueError(f"{where} must be true or false, got {v!r}")


@dataclass(frozen=True, eq=False)
class WeldTorch:
    """용접 계획의 토치 모형 (모듈 docstring). tcp_from_tip = TCP ← 계획의 끝 틀 (회전 = 설정 축, 위치 = 토치 축 위 노즐 끝 +
    stick_out), flange_from_tcp = flange_frame ← TCP. 설정에서 읽은 모형은 끝 위치 0 — 노즐 끝은 with_nozzle_end (이 팔의 토치:
    torch_with_nozzle_end 가 로봇 URDF 충돌 형상에서), 계획이 공정 값으로 with_stick_out 한다 (plan_weld)."""
    tcp_frame: str
    tcp_from_tip: RigidTransform
    flange_frame: str
    flange_from_tcp: RigidTransform
    on_this_arm: bool
    source: str = ""
    status: str = ""
    stick_out_mm: float = 0.0                # 가스 노즐 끝 → 와이어 끝 (겨냥점), 토치 축을 따라 [mm] (공정 값; with_stick_out)
    stick_out_source: str = ""
    nozzle_end_mm: Optional[float] = None    # TCP → 가스 노즐 끝, 토치 축 (+z_TCP) 을 따라 [mm] (토치 형상; None = 아직 모름)
    nozzle_end_source: str = ""

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "WeldTorch":
        d = dict(d or {})
        known = {"tcp_frame", "tip_in_tcp", "flange_frame", "flange_from_tcp", "on_this_arm", "source", "status", "nozzle_end_mm",
                 "nozzle_end_source"}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"{SECTION}: unknown keys {unknown}")
        for key in ("tcp_frame", "flange_frame"):
            if not str(d.get(key) or "").strip():
                raise ValueError(f"{SECTION}.{key} is required")
        for key in ("tip_in_tcp", "flange_from_tcp", "on_this_arm"):
            if d.get(key) is None:
                raise ValueError(f"{SECTION}.{key} is required")
        tip = dict(d["tip_in_tcp"] or {})
        if "xyz_m" in tip:
            raise ValueError(f"{SECTION}.tip_in_tcp.xyz_m: removed (2026-10-12) - the tip offset is the process stick-out along the torch "
                             "axis (job weld_plan.stick_out_mm); tip_in_tcp gives the axes only")
        on = _flag(d["on_this_arm"], f"{SECTION}.on_this_arm")
        noz = d.get("nozzle_end_mm")
        if noz is not None:
            if on:
                raise ValueError(f"{SECTION}.nozzle_end_mm: the weld torch is on this arm - its nozzle end is derived from the robot's "
                                 "URDF collision geometry (scan_core.weld_clearance.nozzle_end_mm); give it only for a torch not on "
                                 "this arm")
            noz = float(noz)
            if not (math.isfinite(noz) and noz >= 0.0):
                raise ValueError(f"{SECTION}.nozzle_end_mm must be >= 0, got {d.get('nozzle_end_mm')!r}")
        return cls(tcp_frame=str(d["tcp_frame"]).strip(),
                   tcp_from_tip=transform_from_axes(dict(tip, xyz_m=[0.0, 0.0, 0.0]), f"{SECTION}.tip_in_tcp", "axes"),
                   flange_frame=str(d["flange_frame"]).strip(),
                   flange_from_tcp=transform_from_quat(d["flange_from_tcp"], f"{SECTION}.flange_from_tcp"),
                   on_this_arm=on, source=str(d.get("source") or ""), status=str(d.get("status") or ""),
                   nozzle_end_mm=noz, nozzle_end_source=(str(d.get("nozzle_end_source") or f"platform config {SECTION}.nozzle_end_mm")
                                                         if noz is not None else ""))

    def with_nozzle_end(self, nozzle_end_mm: float, source: str = "") -> "WeldTorch":
        """같은 토치, 가스 노즐 끝 = TCP 에서 토치 축으로 nozzle_end_mm (토치 형상 — 와이어 돌출의 기준). 끝 위치를 이미 놓았으면 (with_stick_out)
        새 노즐 끝으로 다시 놓는다."""
        n_mm = float(nozzle_end_mm)
        if not (math.isfinite(n_mm) and n_mm >= 0.0):
            raise ValueError(f"nozzle_end_mm must be >= 0, got {nozzle_end_mm!r}")
        t = dataclasses.replace(self, nozzle_end_mm=n_mm, nozzle_end_source=str(source or ""))
        placed = bool(np.any(self.tcp_from_tip.t != 0.0))                    # with_stick_out needs the nozzle end: placed after it
        return t.with_stick_out(self.stick_out_mm, self.stick_out_source) if placed else t

    @property
    def tip_offset_mm(self) -> float:
        """TCP → 계획의 끝 (와이어 끝 = 겨냥점), 토치 축을 따라 [mm] = 노즐 끝 + stick_out (노즐 끝을 모르면 nan)."""
        return float("nan") if self.nozzle_end_mm is None else float(self.nozzle_end_mm) + float(self.stick_out_mm)

    def with_stick_out(self, stick_out_mm: float, source: str = "") -> "WeldTorch":
        """같은 토치, 끝 (겨냥점) = TCP 에서 토치 축 (끝 틀 z = TCP 안의 tcp_from_tip.R[:, 2]) 을 따라 노즐 끝 + stick_out_mm (stick_out =
        가스 노즐 끝 → 와이어 끝). 노즐 끝을 모르면 ValueError (컨택트 팁에서 재지 않는다)."""
        s_mm = float(stick_out_mm)
        if not (math.isfinite(s_mm) and s_mm >= 0.0):
            raise ValueError(f"stick_out_mm must be >= 0, got {stick_out_mm!r}")
        if self.nozzle_end_mm is None:
            raise ValueError(f"the weld torch {self.tcp_frame}: its nozzle end is not known - the wire stick-out is measured from the gas "
                             "nozzle end (on this arm: derived from the robot's URDF collision geometry; a torch not on this arm: "
                             f"{SECTION}.nozzle_end_mm)")
        R = self.tcp_from_tip.R
        t = R[:, 2] * (float(self.nozzle_end_mm) + s_mm) * 1e-3
        tip = dataclasses.replace(self.tcp_from_tip, t=t)
        return dataclasses.replace(self, tcp_from_tip=tip, stick_out_mm=s_mm, stick_out_source=str(source or ""))

    def tcp_from_tip_rolled(self, roll_deg: float = 0.0) -> np.ndarray:
        """4×4 TCP ← 계획의 끝 틀, TCP 를 토치 축 둘레로 roll_deg 돌린 경우: (TCP ← 끝) · Rz(roll)⁻¹ (끝 틀 기준 회전)."""
        Rz = np.eye(4)
        Rz[:3, :3] = roll_matrix(roll_deg)
        return self.tcp_from_tip.matrix() @ np.linalg.inv(Rz)

    def flange_from_tip(self, roll_deg: float = 0.0) -> np.ndarray:
        """4×4 flange_frame ← 계획의 끝 틀 (TCP 롤 roll_deg)."""
        return self.flange_from_tcp.matrix() @ self.tcp_from_tip_rolled(roll_deg)

    def tcp_poses(self, P_tip, R_tip, roll_deg: float = 0.0) -> tuple:
        """계획의 끝 자세 (base: 위치 (…, 3), 회전 (…, 3, 3) [x 진행, y, z 토치 축]) → 이 토치 TCP 의 base 자세 (위치, 회전), TCP 롤
        roll_deg: R_tcp = R_tip · Rz(roll) · (TCP ← 끝)ᵀ, P_tcp = P_tip − R_tcp · (TCP 안의 끝 위치) — 우리 토치: P_tcp = 겨냥점 − (노즐 끝 +
        stick_out) · z_TCP."""
        R_tip = np.asarray(R_tip, dtype=float) @ roll_matrix(roll_deg)
        R_tcp = R_tip @ self.tcp_from_tip.R.T
        P_tcp = np.asarray(P_tip, dtype=float) - R_tcp @ self.tcp_from_tip.t
        return P_tcp, R_tcp

    def rpy_deg(self) -> list:
        from scipy.spatial.transform import Rotation
        return [float(v) for v in Rotation.from_matrix(self.flange_from_tcp.R).as_euler("xyz", degrees=True)]

    def describe(self) -> str:
        t = self.flange_from_tcp.t * 1e3
        r = self.rpy_deg()
        noz = "" if self.nozzle_end_mm is None else f"; nozzle end {self.nozzle_end_mm:.2f} mm along +z_TCP"
        return (f"{self.tcp_frame} on {self.flange_frame}: ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) mm, rpy ({r[0]:+.3f}, {r[1]:+.3f}, "
                f"{r[2]:+.3f}) deg{noz}; {'on this arm' if self.on_this_arm else 'not on this arm'}")

    def to_dict(self) -> dict:
        return jsonable({"tcp_frame": self.tcp_frame, "flange_frame": self.flange_frame, "on_this_arm": self.on_this_arm,
                         "tcp_from_tip": self.tcp_from_tip.to_dict(), "flange_from_tcp": self.flange_from_tcp.to_dict(),
                         "flange_from_tcp_rpy_deg": self.rpy_deg(), "source": self.source, "status": self.status,
                         "stick_out_mm": self.stick_out_mm, "stick_out_source": self.stick_out_source,
                         "stick_out_definition": "gas nozzle end -> wire end along the torch axis (welding practice)",
                         "nozzle_end_mm": self.nozzle_end_mm, "nozzle_end_source": self.nozzle_end_source,
                         "tip_offset_mm": self.tip_offset_mm,
                         "reachability_d14": ("checked on this arm (D14: flange <- TCP from the robot's TF)" if self.on_this_arm
                                              else D14_NOT_ON_THIS_ARM),
                         "tip_frame": "the plan's tip frame: origin = wire tip (bead aim point), x = travel, y = z x x, z = torch axis "
                                      "toward the work; the TCP (torch tip end centre = the contact-tip end) = aim point - "
                                      "(nozzle_end_mm + stick_out_mm) x z_TCP"})


def plan_weld_torch(plan, torch: Optional[WeldTorch]) -> Optional[WeldTorch]:
    """계획의 용접 토치: plan.torch (plan_weld 가 공정 stick_out 을 넣은 것), 없으면 (토치 없이 만든 계획) 주어진 토치에 계획 공정의
    stick_out_mm. 둘 다 없으면 None."""
    t = getattr(plan, "torch", None)
    if t is not None or torch is None:
        return t
    pr = getattr(plan, "process", None)
    if pr is None:                                                  # a bare pose list (no process): the TCP is the tip
        return torch
    return torch.with_stick_out(pr.stick_out_mm, f"job weld_plan.stick_out_mm {pr.stick_out_mm:g} mm "
                                                 f"({pr.provenance.get('stick_out_mm', 'no provenance')})")


def torch_with_nozzle_end(torch: Optional[WeldTorch], body_fn: Optional[Callable] = None) -> Optional[WeldTorch]:
    """The weld torch with its nozzle end (the reference of the wire stick-out): kept when known (a torch not on this arm: the platform
    config's nozzle_end_mm), else - the torch on this arm - derived from body_fn() (this robot's URDF collision geometry in the TCP
    frame, the D14 tool; scan_core.weld_clearance.nozzle_end_mm). ValueError when it cannot be known (never measured from the contact
    tip instead)."""
    if torch is None or torch.nozzle_end_mm is not None:
        return torch
    if not torch.on_this_arm:
        raise ValueError(f"{SECTION}.nozzle_end_mm: the weld torch is not on this arm - its nozzle end (the reference of the wire "
                         "stick-out) must be given in the platform config")
    if not callable(body_fn):
        raise ValueError("the platform gives no tool collision geometry - the weld torch's nozzle end (the reference of the wire "
                         "stick-out) cannot be derived")
    from .weld_clearance import nozzle_end_mm                       # noqa: PLC0415
    body = body_fn()
    if body is None:
        raise ValueError("the platform gives no tool collision geometry - the weld torch's nozzle end (the reference of the wire "
                         "stick-out) cannot be derived")
    return torch.with_nozzle_end(nozzle_end_mm(body), "derived from the robot's URDF collision geometry (the tool parts the torch axis "
                                                      "passes through, farthest along +z_TCP = the gas nozzle rim; "
                                                      f"{body.describe()})")


def weld_torch_config(params: Optional[dict]) -> Optional[WeldTorch]:
    """플랫폼 설정 (ros__parameters) → WeldTorch, 섹션이 없으면 None. 잘못된 섹션은 ValueError."""
    sec = (params or {}).get(SECTION)
    if not sec:
        return None
    return WeldTorch.from_dict(sec)


def d14_on_this_arm(torch: Optional[WeldTorch]) -> tuple:
    """D14 를 이 팔에서 하는가: (True, '') = 용접 토치가 이 팔에 있다 (on_this_arm: 연구실·KIRO); (False, 기록 문구) = 없다고 한 설정 또는
    토치 모형이 없다 — 그때는 D14 를 하지 않고 계획을 납품하며 그 문구를 남긴다 (다른 토치로 대신하지 않는다)."""
    if torch is None:
        return False, D14_NO_WELD_TORCH
    if not torch.on_this_arm:
        return False, D14_NOT_ON_THIS_ARM
    return True, ""


def reach_tool(torch: Optional[WeldTorch], lookup: Callable[[str, str], np.ndarray], flange_frame: str, roll_deg: float = 0.0) -> tuple:
    """D14 의 도구 (용접 토치가 이 팔에 있을 때만): 이 팔의 flange_frame ← 계획의 끝 틀 (4×4) = lookup(flange_frame, torch.tcp_frame) ·
    TCP ← 끝 (stick_out, TCP 롤 roll_deg) — 플랜지 ← TCP 는 그 로봇 자신의 모형 (TF), 과 그 설명 (TF 에서 읽은 TCP 위치 포함 — 실행 기록에서 어느 토치 모형으로
    확인했는지 보인다). 용접 토치가 이 팔에 없다고 한 설정이면 ValueError (D14 는 하지 않는다: d14_on_this_arm).
    lookup(target, source) → 4×4 (MoveIt 어댑터: TF, 가상 팔: 그 URDF, 없으면 설정)."""
    on, why = d14_on_this_arm(torch)
    if not on:
        raise ValueError(why)
    T_ft = np.asarray(lookup(flange_frame, torch.tcp_frame), dtype=float)
    p = T_ft[:3, 3] * 1e3
    return T_ft @ torch.tcp_from_tip_rolled(roll_deg), (f"the weld torch {torch.tcp_frame} (weld_torch, on this arm: {flange_frame} <- "
                                                 f"{torch.tcp_frame} from the robot's TF: ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) mm)")
