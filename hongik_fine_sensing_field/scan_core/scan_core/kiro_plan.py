"""KIRO 형식의 용접 계획 출력 (사용자 승인 2026-10-12 (A); 같은 날 지시: KIRO 데이터셋의 형식·내용 그대로) — out_dir/kiro/ 에:

  pass_<k>.yaml          KIRO 의 `xy_endpoint_review_v1` 그대로 (키·순서·형이 KIRO 표본 pass_1–4.yaml 과 같다): schema, pass, frame_id,
                         requires_ik, current_start, current_goal = {position_m {x, y, z}, orientation_xyzw {x, y, z, w}} — 그 층의 직선
                         (시작 → 끝, 일정 자세; KIRO 실행기가 그 직선을 일정 TCP 속도로 가고 위빙·리드인/아웃·아크 끄기는 스스로 만든다).
                         시작·끝 = 그 층의 쓸 수 있는 심 시작·끝 (우리 잰 끝 — 층 높이의 끝 (G1WeldPlan.layer_ends), 작업 inset — 과 층
                         계획의 겨냥점). 자세 = 우리 계획 토치 자세 (WPS 진행각·작업각) 를 용접 토치 모형 (설정 weld_torch — 우리 1차
                         브라켓 토치, 사용자·KIRO 합의 2026-10-12) 의 TCP ← 끝 변환으로 그 TCP (KIRO: right_manipulator_ee_point = 토치
                         팁 끝 = 컨택트 팁 끝, 공칭 — 겨냥점에서 토치 축을 따라 와이어 돌출 20 mm 뒤) 에, KIRO World 프레임에서. 촘촘한
                         경로·위빙 점은 없다 (weld_plan.npz 는 그대로).
  pass_<k>_info.yaml     공유용 정보 (명령 아님 — KIRO 의 설정은 KIRO 것 그대로): 우리 모형이 가정한 KIRO 설정 (KIRO 로그 [commanded] /
                         [execution_conditions] 의 키 이름) + 그 설정에서 유도한 층 정보 (층 단면적·높이·겨냥 오프셋, 심 방향 변화, 잰 심이
                         직선에서 벗어난 정도, 쓴 자세의 각, 프레임 출처).
  README.md              파일·프레임·확인 대기 항목 (ee ← 끝 변환) 설명.

프레임 사슬 (설정 kiro_output — 코드에 수가 없다):
  World ← base   base = 실행 때 MoveIt 에서 읽은 계획 프레임 (팔 그룹의 base — 설정의 프레임 이름이 아니다, 2026-10-12 KIRO 답).
                 그것이 World 면 항등. 아니면 설정 world_from_base 의 고정값 (KIRO 의 몸통·베이스는 움직이지 않는다: KIRO 로그 FK 맞춤
                 (−249.83, −499.71, 1330.42) mm + 맞춤 회전 0.016°, 출처와 함께). TF 는 선택 덮어쓰기 (world_from_tf true) 뿐.
  ee ← 와이어 끝  계획의 용접 토치 모형 (plan.torch = 설정 weld_torch + 토치의 노즐 끝 + 공정 weld_plan.stick_out_mm; scan_core.weld_torch):
                 ee 원점 = TCP = 토치 팁 끝 중심 (컨택트 팁 끝, 공칭; 가스 노즐 안으로 nozzle_end_mm), 와이어 끝 (비드 겨냥점) = ee 에서
                 +z_ee 로 nozzle_end_mm + stick_out_mm (돌출 = 노즐 끝 → 와이어 끝, 사용자 2026-10-13; KIRO 실행기는 돌출을 적용하지 않는다
                 — 그래서 쓰는 자세는 겨냥점에서 −z_ee 로 그만큼 물러난 TCP), +z_ee = 노즐 축 (공작물 쪽), x_ee = ±진행
                 (TCP 롤 0 / 180°, plan.torch_roll_deg — D14 가 고른다). 플랜지 ← ee (weld_torch.flange_from_tcp, CAD) 는 정보 파일·README
                 에 적는다. 연구실은 World := link0 이고 토치 TCP = URDF torch_tcp 라 연구실 파일 = 우리 base 의 우리 토치 TCP 자세.
자세: 우리 계획 토치 자세 (WPS 진행각·작업각; 패스 안 일정 = 단면 토치 자세들의 회전 평균) · TCP 롤 → ee. ('reference' 정책 — KIRO 가 준
  pass_<k>.yaml 의 자세 — 은 2026-10-12 사용자 결정으로 없앴다: 그 자세는 폐기된 KIRO 로그 프레임의 것.)
직선·자세는 scan_core.weld_line.weld_line 하나가 만든다 (2026-10-13: D14 가 확인하는 자세 = 여기 쓰는 자세 = result.paths — 같은 함수).

Both weld directions (user 2026-10-14): a plan with direction sets (g1_outputs.write_outputs: plan.directions) also gets each
delivered set in the same format under directions/<S_to_E | E_to_S>/kiro/ (pass files exactly as for that set's own plan; the
primary README names the sets and how to choose, a set's README says which set it is).

모든 문구·파일은 영어 (사용자 2026-10-10). 오프라인 재생성: python3 -m scan_core.kiro_plan <실행 폴더> --platform-config <yaml> (--help).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .g1_types import atomic_write_bytes, jsonable
from .rigid import RigidTransform, _nearest_rotation, _quat_to_R, _R_to_quat, _vec, compose, transform_from_quat
from .weld_line import LINE_RULE, line_rotation, pass_endpoints, weld_line
from .weld_torch import D14_NOT_ON_THIS_ARM, WeldTorch, plan_weld_torch, weld_torch_config

__all__ = ["SCHEMA", "INFO_SCHEMA", "KIRO_DIR", "RigidTransform", "KiroOutputConfig", "kiro_output_config",
           "read_pass_yaml", "plan_torch", "format_pass_yaml", "pass_doc", "pass_endpoints", "planned_rotation",
           "resolve_world_from_base", "compose", "write_kiro_outputs", "KIRO_CAP_VS_PASS3_MM", "main"]

SCHEMA = "xy_endpoint_review_v1"              # KIRO 의 끝점 파일 형식 (pass_1–4.yaml)
INFO_SCHEMA = "hongik.kiro_pass_info/1"
KIRO_DIR = "kiro"
REMOVED_KEYS = ("orientation_policy", "reference_dir")        # 2026-10-12 사용자: 'reference' 정책은 폐기된 KIRO 로그 프레임에 기댄다
_PASS_RE = re.compile(r"^pass_(\d+)\.yaml$")
# KIRO 데이터셋의 믿을 만한 패스 간 오프셋 (캡 − 패스 3, ee_point 의 측면·위, mm; interpass_offsets_summary.csv: n 23, sd 0.09 / 0.18) —
# KIRO 로그의 ee_point 는 옛 URDF 의 치즐링 건 프레임 (만든 적 없는 부품; 2026-10-12) 이라 우리 토치의 와이어 끝이 아니다: 비교 메모로만
KIRO_CAP_VS_PASS3_MM = {"lateral": 6.42, "up": 3.89, "source": "KIRO dataset 2026-09-15/16 (fsv1_work/kiro_dataset/"
                                                              "interpass_offsets_summary.csv: n 23, sd 0.09 / 0.18 mm)"}


# --------------------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------------------

def _aim_rule(plan, wp) -> dict:
    """The aim rule of this pass (weld_plan.AIM_RULE + the plan's per-pass record): vertical = the current surface (the earlier passes'
    convex beads); lateral = KIRO's practice from the measured section (root / fill 1 / fill 2 / cap), or the geometric rule for a
    layer it does not cover; the 09-22 verification row of this pass."""
    from .weld_plan import AIM_KIRO, AIM_RULE              # noqa: PLC0415
    rec = dict(getattr(plan, "aim", None) or {})
    row = next((r_ for r_ in rec.get("passes") or [] if int(r_.get("k", -1)) == int(wp.k)), None)
    out = {"vertical": AIM_RULE["vertical"]}
    if row is None:                                        # a plan without the per-pass record
        out.update(lateral=AIM_RULE["lateral"], uncovered=AIM_RULE["uncovered"])
        return out
    covered = row.get("lateral") == AIM_KIRO
    ck = next((c for c in (rec.get("check_0922") or {}).get("passes") or [] if int(c.get("k", -1)) == int(wp.k)), None)
    out.update({"lateral": AIM_RULE["lateral"] if covered else AIM_RULE["uncovered"], "this_pass": row.get("lateral"),
                "why": row.get("why"), "toward_bevel": rec.get("toward_bevel"),
                "verification_0922": ck if ck is not None else (rec.get("check_0922") or {}).get("why")})
    return out


def _world_from_base(d: dict) -> Optional[RigidTransform]:
    if not d:
        return None
    return transform_from_quat(d, "kiro_output.world_from_base",
                               " (the base is the runtime MoveIt planning frame - no base frame name here)")


@dataclass(frozen=True)
class KiroOutputConfig:
    """플랫폼 설정의 kiro_output 섹션 + 용접 계획의 토치 모형 (weld_torch 섹션 — 쓰는 자세가 그 토치 TCP 의 것)."""
    world_frame: str
    torch: WeldTorch
    world_from_base: Optional[RigidTransform] = None
    enabled: bool = True
    world_from_tf: bool = False      # true = 실행 때 TF 에 World ← base 가 있으면 그것으로 덮어쓴다 (선택; 기본은 설정 고정값)

    @property
    def tcp_frame(self) -> str:
        """쓰는 자세의 TCP 프레임 (용접 토치의 TCP)."""
        return self.torch.tcp_frame

    @property
    def ee_from_tip(self) -> RigidTransform:
        """TCP (ee) ← 계획의 끝 틀 (용접 토치 모형)."""
        return self.torch.tcp_from_tip

    @classmethod
    def from_dict(cls, d: Optional[dict], torch: Optional[WeldTorch]) -> "KiroOutputConfig":
        d = dict(d or {})
        moved = sorted(set(d) & {"tcp_frame", "ee_from_tip"})
        if moved:
            raise ValueError(f"kiro_output.{', '.join(moved)}: moved to the weld_torch section (the weld plan's torch model, "
                             "2026-10-12): weld_torch.tcp_frame / weld_torch.tip_in_tcp")
        removed = sorted(set(d) & set(REMOVED_KEYS))
        if removed:
            raise ValueError(f"kiro_output.{', '.join(removed)}: removed (user 2026-10-12) - the 'reference' orientation (KIRO's logged "
                             "pass_<k>.yaml orientations) rests on the void KIRO-log frame; the written orientation is always our planned "
                             "torch orientation")
        known = {"enabled", "world_frame", "world_from_base", "world_from_tf"}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"kiro_output: unknown keys {unknown}")
        if torch is None:
            raise ValueError("kiro_output needs the weld_torch section (the weld plan's torch model: the TCP the poses are written for)")
        wf = str(d.get("world_frame") or "").strip()
        if not wf:
            raise ValueError("kiro_output.world_frame is required (KIRO's world frame name, their logs: World)")
        def flag(v, default):
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)
        return cls(world_frame=wf, torch=torch, world_from_base=_world_from_base(d.get("world_from_base")),
                   enabled=flag(d.get("enabled"), True), world_from_tf=flag(d.get("world_from_tf"), False))

    def to_dict(self) -> dict:
        return jsonable({"world_frame": self.world_frame, "weld_torch": self.torch.to_dict(),
                         "world_from_base": self.world_from_base.to_dict() if self.world_from_base is not None else None,
                         "enabled": self.enabled,
                         "world_from_tf": self.world_from_tf})


def kiro_output_config(params: Optional[dict]) -> Optional[KiroOutputConfig]:
    """플랫폼 설정 (ros__parameters) → KiroOutputConfig (kiro_output + weld_torch 섹션), kiro_output 이 없거나 enabled false 면 None.
    잘못된 섹션 (weld_torch 없음 포함) 은 ValueError."""
    sec = (params or {}).get("kiro_output")
    if not sec:
        return None
    cfg = KiroOutputConfig.from_dict(sec, weld_torch_config(params))
    return cfg if cfg.enabled else None


def resolve_world_from_base(cfg: KiroOutputConfig, base_frame: str,
                            tf_lookup: Optional[Callable[[str, str], np.ndarray]] = None) -> tuple:
    """World ← base (base = 실행 때의 계획 프레임 이름): base 가 World 면 항등, 아니면 설정 world_from_base 의 고정값. 설정 world_from_tf
    가 true 이고 tf_lookup(target, source) → 4×4 가 답하면 그것으로 덮어쓴다 (선택; 실패면 고정값). → (RigidTransform | None, 메모 목록)."""
    notes = []
    if str(base_frame) == cfg.world_frame:
        return RigidTransform(R=np.eye(3), t=np.zeros(3), source=f"the runtime planning frame is {cfg.world_frame} (identity)",
                              status="runtime planning frame", base_frame=str(base_frame)), notes
    if cfg.world_from_tf and tf_lookup is not None:
        try:
            T = np.asarray(tf_lookup(cfg.world_frame, base_frame), dtype=float).reshape(4, 4)
            return RigidTransform(R=_nearest_rotation(T[:3, :3], "TF rotation"), t=T[:3, 3].copy(),
                                  source=f"TF {cfg.world_frame} <- {base_frame} at output time", status="runtime TF",
                                  base_frame=base_frame), notes
        except Exception as e:  # noqa: BLE001 - no such frame in TF: the configured fallback decides
            notes.append(f"TF has no {cfg.world_frame} <- {base_frame} ({type(e).__name__}: {str(e)[:160]})")
    fb = cfg.world_from_base
    if fb is not None:
        return dataclasses.replace(fb, base_frame=str(base_frame)), notes
    notes.append(f"no {cfg.world_frame} <- {base_frame} transform: no kiro_output.world_from_base in the platform config")
    return None, notes


# --------------------------------------------------------------------------------------
# KIRO pass_<k>.yaml (xy_endpoint_review_v1)
# --------------------------------------------------------------------------------------
def _num(v) -> str:
    """KIRO 표본과 같은 수 표기 (파이썬 repr: 17 유효 자리, 정수는 정수)."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"not a finite number: {v!r}")
    return repr(f)


def pass_doc(k: int, frame_id: str, start_p, start_q, goal_p, goal_q) -> dict:
    """KIRO 끝점 문서 (키 순서 = KIRO 표본)."""
    def pose(p, q):
        p = _vec(p, 3, "position_m")
        q = _vec(q, 4, "orientation_xyzw")
        return {"position_m": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                "orientation_xyzw": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}}
    return {"schema": SCHEMA, "pass": int(k), "frame_id": str(frame_id), "requires_ik": True,
            "current_start": pose(start_p, start_q), "current_goal": pose(goal_p, goal_q)}


def format_pass_yaml(doc: dict) -> str:
    """끝점 문서 → KIRO 표본과 글자까지 같은 YAML (2 칸 들여쓰기, 블록 형식, repr 수, 끝 줄바꿈)."""
    lines = [f"schema: {doc['schema']}", f"pass: {_num(int(doc['pass']))}", f"frame_id: {doc['frame_id']}",
             f"requires_ik: {_num(bool(doc['requires_ik']))}"]
    for key in ("current_start", "current_goal"):
        lines.append(f"{key}:")
        pose = doc[key]
        lines.append("  position_m:")
        lines += [f"    {c}: {_num(float(pose['position_m'][c]))}" for c in "xyz"]
        lines.append("  orientation_xyzw:")
        lines += [f"    {c}: {_num(float(pose['orientation_xyzw'][c]))}" for c in "xyzw"]
    return "\n".join(lines) + "\n"


def read_pass_yaml(path) -> dict:
    """KIRO pass_<k>.yaml → {'pass', 'frame_id', 'start': (p, q), 'goal': (p, q), 'path'} (형식 검사)."""
    import yaml
    p = Path(path)
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {d.get('schema')!r} is not {SCHEMA}")
    out = {"pass": int(d["pass"]), "frame_id": str(d.get("frame_id")), "path": str(p)}
    for key, nm in (("current_start", "start"), ("current_goal", "goal")):
        pose = d.get(key) or {}
        pp = _vec([(pose.get("position_m") or {}).get(c) for c in "xyz"], 3, f"{p.name} {key}.position_m")
        qq = _vec([(pose.get("orientation_xyzw") or {}).get(c) for c in "xyzw"], 4, f"{p.name} {key}.orientation_xyzw")
        _quat_to_R(qq, f"{p.name} {key}.orientation_xyzw")
        out[nm] = (pp, qq)
    return out


def plan_torch(plan, cfg: "KiroOutputConfig") -> WeldTorch:
    """계획의 용접 토치 (plan.torch: 설정 weld_torch + 공정 stick_out_mm) — 계획이 토치 없이 만들어졌으면 설정의 토치에 계획 공정의 stick_out_mm."""
    return plan_weld_torch(plan, cfg.torch)


# --------------------------------------------------------------------------------------
# 계획 → 패스 끝점 (base, 와이어 끝): scan_core.weld_line (D14 와 같은 출처)
# --------------------------------------------------------------------------------------
planned_rotation = line_rotation          # 패스 안 일정 토치 자세 (base ← 끝 틀): 단면 토치 자세들의 회전 평균 (weld_line.line_rotation)


def _angles(a, d, n, bev) -> dict:
    """토치 축 a (공작물 쪽) 의 진행각 (+ 미는 각) 과 작업각 (+ 베벨면 쪽) — AWS A3.0 정의 (weld_plan._tcp_R 과 같다): 진행각 = a 와
    진행 방향 d 에 수직인 면 사이 각, 작업각 = 그 면에서 −n 과 a 의 사영 사이 각. d = 용접선 방향 (기준), n = 판 법선 (d 에 직교화),
    bev = 베벨 쪽."""
    d = d / np.linalg.norm(d)
    n = n - (n @ d) * d
    n = n / np.linalg.norm(n)
    b = bev - (bev @ n) * n - (bev @ d) * d
    b = b / np.linalg.norm(b)
    a = a / np.linalg.norm(a)
    return {"travel_angle_deg": math.degrees(math.asin(float(np.clip(a @ d, -1.0, 1.0)))),
            "work_angle_deg": math.degrees(math.atan2(float(a @ b), float(-a @ n)))}


# --------------------------------------------------------------------------------------
# 쓰기
# --------------------------------------------------------------------------------------
def _rng(v) -> dict:
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())} if a.size else {}


def _setting_num(v: float):
    """설정값: 정수면 정수 (KIRO [commanded] current_a=190), 아니면 float."""
    f = float(v)
    return int(f) if f.is_integer() else f


def _line_deviation(plan, wp, ends: dict) -> dict:
    """계획 겨냥점 (단면마다, 잰 심을 따른다 — 선 모형이 2차면 휜다) 이 KIRO 의 직선 (시작 → 끝) 에서 벗어난 정도 [mm] — 정보만."""
    P0, P1 = np.asarray(ends["start"]["tip_base_m"], dtype=float), np.asarray(ends["goal"]["tip_base_m"], dtype=float)
    e = (P1 - P0) / np.linalg.norm(P1 - P0)
    st = {float(x.s_mm): x for x in plan.stations}
    lo_s, hi_s = sorted((ends["start"]["s_mm"], ends["goal"]["s_mm"]))
    rows = []
    for s_, P in zip(np.asarray(wp.s_mm, dtype=float), np.asarray(wp.aim_base_m, dtype=float)):
        if not (lo_s - 1e-9 <= s_ <= hi_s + 1e-9):
            continue
        dv = (P - P0) - float((P - P0) @ e) * e
        x = st[float(s_)]
        rows.append((s_, float(dv @ x.w) * 1e3, float(dv @ x.n) * 1e3))
    if not rows:
        return {}
    a = np.array(rows)
    i_l, i_v = int(np.argmax(np.abs(a[:, 1]))), int(np.argmax(np.abs(a[:, 2])))
    return {"max_lateral_mm": float(a[i_l, 1]), "at_s_lateral_mm": float(a[i_l, 0]), "max_vertical_mm": float(a[i_v, 2]),
            "at_s_vertical_mm": float(a[i_v, 0]), "rms_mm": float(np.sqrt(np.mean(a[:, 1] ** 2 + a[:, 2] ** 2))), "n_sections": int(len(a)),
            "note": "our layer-plan aim points along the measured seam minus KIRO's straight line start -> goal (lateral along w, "
                    "vertical along the plate normal); information only - the executor runs the straight line"}


def _info_doc(plan, wp, ends: dict, pose: dict, wt: RigidTransform, cfg: KiroOutputConfig, prev, torch: WeldTorch,
              location: Optional[str] = None) -> dict:
    """pass_<k>_info.yaml: 우리 모형이 가정한 KIRO 설정 (KIRO 로그 키 이름) + 그 설정에서 유도한 층 정보 — 공유용, 명령 아님.
    location = where the pass file is (a direction set's, _set_location); None = the primary kiro/ of the PrecisionSensing action."""
    pr = plan.process
    role = wp.role
    bm = pr.bead_model()
    prm = pr.params(role)
    L_mm = float(np.linalg.norm(ends["goal"]["tip_base_m"] - ends["start"]["tip_base_m"])) * 1e3
    weave = wp.kind == "cap"
    a_req = np.asarray(wp.area_required_mm2, dtype=float)
    a_mean = float(wp.area_mean_mm2)
    # 설정선 위 등가 전류 (정보): 공칭 속도에서 필요 단면적의 심 평균을 내는 WFS → I
    wfs_eq = a_mean * float(prm.speed_mps) * 1e3 / (bm.deposition_efficiency * bm.wire_area_mm2) * 60.0 / 1000.0
    ax = np.asarray(wp.aim_xz_mm, dtype=float)
    derived = {
        "role": role, "layer": int(wp.layer), "bead": int(wp.bead), "n_beads": int(wp.n_beads), "direction": wp.direction,
        "wps_source": str(pr.source),
        "straight_line": {"length_mm": L_mm, "s_start_mm": ends["start"]["s_mm"], "s_goal_mm": ends["goal"]["s_mm"],
                          "ends": ends["source"], "s": "distance from the measured S end along the seam axis",
                          "tip_base_m": {"frame": str(plan.base_frame), "start": ends["start"]["tip_base_m"],
                                         "goal": ends["goal"]["tip_base_m"]},
                          "measured_ends": {X: {"end": ends["measured"][X]["end"], "s_mm": ends["measured"][X]["s_mm"],
                                                "sigma_mm": ends["measured"][X]["sigma_mm"],
                                                "tip_base_m": ends["measured"][X]["tip_base_m"]} for X in ("start", "goal")},
                          "end_inset_mm": {**ends["inset_mm"],
                                           "meaning": "weld design input (job weld_design.end_inset_mm): start / goal moved this far "
                                                      "inside the measured ends along the seam; sensing and the measured ends unchanged"},
                          "measured_seam_deviation": _line_deviation(plan, wp, ends)},
        "area_mm2": {"deposited_along_seam": _rng(wp.area_mm2), "required_along_seam": _rng(a_req), "required_seam_mean": a_mean,
                     "layer_deposited_seam_mean": float(np.mean(wp.area_mm2)) * (int(wp.n_beads) if wp.kind == "fill" else 1),
                     "model": "deposited = eta x wire area x WFS / v at the assumed settings; required = the area to the planned layer "
                              "top (equal groove-area fraction at every section)"},
        "layer_mm": {"bottom": _rng(wp.layer_from_mm), "top": _rng(wp.layer_to_mm),
                     **({"toe_left_z": _rng(wp.bead_xz("toe_l")[:, 1]), "toe_right_z": _rng(wp.bead_xz("toe_r")[:, 1]),
                         "toe_left_x": _rng(wp.bead_xz("toe_l")[:, 0]), "toe_right_x": _rng(wp.bead_xz("toe_r")[:, 0]),
                         "crown_x": _rng(wp.bead_xz("crown")[:, 0]), "crown_ratio": _rng([b.kappa for b in wp.beads])}
                        if getattr(wp, "beads", None) else {}),
                     "z": "above the backing-bar top along the plate normal; the pass's convex bead (bead_geometry): bottom = the "
                          "lowest point of the surface under it, top = its crown (the highest point of its free surface), toes = where "
                          "its free surface meets the walls / earlier beads (x along w from the root-gap centre); heights follow the "
                          "deposits at the assumed settings"},
        "aim_offset_mm": {"frame": "x along w = n x u from the root-gap centre (+x = left looking S->E), z along the plate normal "
                                   "from the backing-bar top",
                          "rule": _aim_rule(plan, wp),
                          "start": {"x": ends["start"]["aim_xz_mm"][0], "z": ends["start"]["aim_xz_mm"][1]},
                          "goal": {"x": ends["goal"]["aim_xz_mm"][0], "z": ends["goal"]["aim_xz_mm"][1]},
                          "x_along_seam": _rng(ax[:, 0]), "z_along_seam": _rng(ax[:, 1])},
        "speed_mm_s": {"assumed_setting": float(prm.speed_mps) * 1e3,
                       "required_along_seam": _rng(np.asarray(wp.speed_required_mps, dtype=float) * 1e3),
                       "constant_for_required_mean": float(wp.speed_const_mps) * 1e3,
                       "current_for_required_mean_at_the_setting_speed_a": pr.current_on_line(wfs_eq),
                       "note": "information only: the speed (or, on the settings line WFS(I), the current) that would deposit the "
                               "seam-mean required area; the assumed settings are not changed"},
        "orientation": {k_: v for k_, v in pose.items() if not k_.startswith("_")},
        "frames": {"world_frame": cfg.world_frame, "base_frame": str(plan.base_frame), "world_from_base": wt.source,
                   "world_from_base_status": wt.status, "tcp_frame": torch.tcp_frame,
                   "tcp_definition": "the TCP is the torch tip end centre (the contact-tip end, recessed nozzle_end_mm inside the gas "
                                     "nozzle); each written pose = the bead aim point (the wire end) moved back along -z_TCP by "
                                     "nozzle_end_mm + stick_out_mm (the stick-out is measured from the gas nozzle end); z_TCP = the torch "
                                     "axis toward the work",
                   "stick_out_mm": torch.stick_out_mm, "stick_out_source": torch.stick_out_source,
                   "stick_out_definition": "gas nozzle end -> wire end along the torch axis",
                   "nozzle_end_mm": torch.nozzle_end_mm, "nozzle_end_source": torch.nozzle_end_source,
                   "tcp_to_aim_point_mm": torch.tip_offset_mm,
                   "aim_point_World_m": {X: wt.apply(ends[X]["tip_base_m"]) for X in ("start", "goal")},
                   "ee_from_tip": cfg.ee_from_tip.source, "ee_from_tip_xyz_mm": [float(v) * 1e3 for v in torch.tcp_from_tip.t],
                   "ee_from_tip_status": cfg.ee_from_tip.status,
                   "weld_torch": {"model": torch.source, "status": torch.status,
                                  "flange_frame": torch.flange_frame,
                                  "flange_from_tcp_xyz_mm": [float(v) * 1e3 for v in torch.flange_from_tcp.t],
                                  "flange_from_tcp_rpy_deg": torch.rpy_deg(), "flange_from_tcp_source": torch.flange_from_tcp.source,
                                  "flange_from_tcp_status": torch.flange_from_tcp.status,
                                  "on_this_arm": torch.on_this_arm,
                                  "reachability_d14": torch.to_dict()["reachability_d14"]}},
    }
    if weave:
        y_ee = np.asarray(pose["_R_world_ee"])[:, 1]
        w_world = wt.R @ np.asarray(ends["start"]["w"])
        amp = float(pr.weave_amplitude_mm) if pr.weave_amplitude_mm is not None else float("nan")
        derived["reinforcement_mm"] = _rng(wp.reinforcement_mm) if wp.reinforcement_mm is not None else {}
        derived["weave"] = {"lateral_amplitude_mm": amp * abs(float(y_ee @ w_world)), "band_width_mm": _rng(wp.weave_width_mm),
                            "pitch_with_the_cycles_mm": L_mm / pr.weave_cycles if pr.weave_cycles > 0 else None,
                            "cycles_at_the_set_pitch": int(math.ceil(L_mm / float(pr.weave_pitch_mm) - 1e-9)),
                            "note": f"the assumed amplitude along {pr.weave_axis} projected on the section's lateral axis for this "
                                    f"orientation; {pr.weave_cycles} cycles over this {L_mm:.1f} mm line (KIRO's pass 4: 16 cycles over "
                                    "119.6 mm = 7.48 mm pitch, set pitch 7.8 mm)"}
        if prev is not None:
            pa = np.asarray(prev.aim_xz_mm, dtype=float)
            derived["comparison_kiro_cap_vs_pass3"] = {
                "kiro_mm": {"lateral": KIRO_CAP_VS_PASS3_MM["lateral"], "up": KIRO_CAP_VS_PASS3_MM["up"]},
                "ours_mm": {"lateral": float(np.median(ax[:, 0]) - np.median(pa[:, 0])), "up": float(np.median(ax[:, 1]) - np.median(pa[:, 1])),
                            "previous_pass": int(prev.k)},
                "note": "comparison only: KIRO's offsets are of the right_manipulator_ee_point of KIRO's old URDF (the chiseling-gun "
                        "frame of a part that was never manufactured - not our torch's wire tip; the logs were not recorded with this "
                        "frame's tool) and their lateral sign is not tied to our section frame; source " + KIRO_CAP_VS_PASS3_MM["source"]}
    return {
        "schema": INFO_SCHEMA, "pass": int(wp.k), "pass_file": f"pass_{int(wp.k)}.yaml",
        "pass_file_location": location.format(k=int(wp.k)) if location is not None else (
            f"<dirname(data_file_path)>/kiro/pass_{int(wp.k)}.yaml for the PrecisionSensing action (data_file_path = "
            f"the run's precision_sensing.npz, which embeds the same text as kiro_pass_{int(wp.k)}_yaml)"),
        "purpose": "information for sharing: the KIRO settings our layer model assumed and the values derived from them - not "
                   "commands (KIRO's executor keeps its own settings)",
        "assumed_kiro_settings": {
            "commanded": {"current_a": _setting_num(pr.role_value("current_a", role)), "voltage": float(pr.role_value("voltage_v", role)),
                          "material": str(pr.wire_material), "diameter_mm": float(pr.wire_diameter_mm), "mode": str(pr.welding_mode),
                          "gas": str(pr.shielding_gas)},
            "execution_conditions": {
                "weld_tcp_speed_mm_s": float(prm.speed_mps) * 1e3, "weld_weave_enabled": bool(weave),
                "weld_weave_pattern": str(pr.weave_pattern), "weld_weave_pitch_mm": float(pr.weave_pitch_mm),
                "weld_weave_left_dwell_s": float(pr.weave_dwell_s) if weave else 0.0,
                "weld_weave_right_dwell_s": float(pr.weave_dwell_s) if weave else 0.0,
                "weld_weave_amplitude_mm": (float(pr.weave_amplitude_mm) if pr.weave_amplitude_mm is not None else None) if weave else 0.0,
                "weld_weave_cycles": int(pr.weave_cycles) if weave else 0,
                "weld_weave_samples_per_cycle": int(pr.weave_samples_per_cycle), "weld_weave_axis": str(pr.weave_axis),
                "weld_lead_in_mm": float(pr.role_value("lead_in_mm", role)), "weld_lead_out_mm": float(pr.role_value("lead_out_mm", role)),
                "weld_arc_off_delay_ms": float(pr.role_value("arc_off_delay_ms", role))}},
        "derived": derived,
    }


README = """# KIRO-format weld plan (hongik fine sensing)

Written by `scan_core.kiro_plan` from this run's weld plan (`../weld_plan.json`), {n} pass(es). The files under kiro/ give each pass
in the form of KIRO's dataset and list the values we assumed and derived **for sharing - they are not commands**: KIRO's process
settings stay as they are (KIRO's executor keeps its own settings).

## Files
- `pass_<k>.yaml`: KIRO's endpoint format `{schema}` (keys, order and types as KIRO's pass_1.yaml ... pass_4.yaml): the straight
  line of that pass's layer, start pose -> goal pose (position + orientation) - KIRO's executor runs it at constant orientation and
  constant TCP speed and generates the weave, lead-in / lead-out and the arc-off pre-distance itself (no dense path, no weave point
  here). `current_start` / `current_goal` = the poses of the TCP `{tcp}` = the torch tip end centre for the usable seam start /
  end of that layer: each pose = the bead aim point of our layer plan (the wire end, on the surface before the pass: the earlier
  passes' convex beads; laterally KIRO's practice from the measured section - {aim_words}) at our measured ends
  ({ends_src}){inset},
  moved back along -z_TCP (the torch axis) by {tipoff:.2f} mm = the TCP's recess inside the gas nozzle {noz:.2f} mm ({noz_src})
  + the wire stick-out {stick:g} mm measured from the gas nozzle end ({stick_src}) - KIRO's executor does not apply a
  stick-out. The action result's WeldingPath paths are the aim-point (wire-tip) lines themselves (KIRO's WeldingPath definition).
- `pass_<k>_info.yaml`: information only. `assumed_kiro_settings` = the KIRO settings our layer model assumed (KIRO's own key names
  from its weld logs: `commanded` current_a, voltage, material, diameter_mm, mode, gas; `execution_conditions` weld_tcp_speed_mm_s,
  weld_weave_*, weld_lead_in_mm, weld_lead_out_mm, weld_arc_off_delay_ms). `derived` = what our model derives with them: deposited
  and required areas, layer heights and aim offsets (with their along-seam variation), the cap reinforcement, how far the measured
  seam deviates from the straight line, the travel / work angles of the written orientation and the frames used.

## Frames
- `frame_id: {world}` (KIRO's schema). {world} <- {base}: {wsrc} ({wstat}).{world_def}
- Poses are of the TCP frame `{tcp}` (the torch tip end centre) of the weld torch model ({torch_src}; {torch_stat}). Our plan's
  tip frame (origin = the wire end = the bead aim point, z = torch axis toward the work) in that frame (ee <- tip): {tip_txt} -
  the nozzle end + the stick-out along z_TCP ({esrc}; {estat}). On the flange: `{flange}` <- `{tcp}` = {flange_txt} ({flange_src};
  {flange_stat}).{torch_note}
- A quaternion q and -q are the same orientation (written with w >= 0).

## Orientation
- Our torch orientation per layer: the process travel and work angles ({wps}; AWS A3.0 definitions: travel = angle out of
  the plane normal to the weld axis, + push / - drag, work = angle from the plate normal in that plane, + toward the bevel face)
  are the target; the plan takes per pass
  the collision-free and reachable orientation closest to them (no bound on the deviation - the deviation is information, listed
  per pass below and in `pass_<k>_info.yaml` derived.orientation), constant over the pass (rotation mean of our section torch
  frames), for KIRO's TCP frame through the ee <- tip transform. The weld-plan check (D14: clearance by tool part role, the robot's
  IK / limits / collisions) ran on exactly these straight lines and orientations (scan_core.weld_line), not on the per-station
  layer plan; `pass_<k>_info.yaml` derived.orientation deviation_why says why a pass left the WPS angles.{orient}
- TCP roll about the torch axis: {roll} - KIRO's executor weaves along the TCP y axis, so the TCP y lies across the seam
  (roll 0 deg: TCP x = travel; 180 deg: TCP x = -travel); the same roll for every pass of this plan.

{delivery}{directions}{extra}"""
README_DELIVERY = """## Delivery
- These files are written to `kiro/` next to the run's outputs: `<dirname(data_file_path)>/kiro/pass_<k>.yaml` (+ `_info.yaml`,
  README.md) for the PrecisionSensing action (data_file_path = the run's precision_sensing.npz, which embeds the same yaml text as
  kiro_pass_<k>_yaml). The PrecisionSensing action result carries the same straight lines in result.paths, in
  WeldingPath's documented form (robot base frame, <= 5 mm waypoints, every position = the welding wire-tip TCP = our bead aim
  point, +Z = plate normal; first / last waypoint = the aim points of these start / goal poses: TCP position + stick-out x z_TCP,
  in the base frame). WeldingPath has no frame_id and no torch orientation, so the World poses with the torch orientation travel
  in the result's npz (data_file_path: kiro_pass_<k>_start_pose / _goal_pose and the yaml text); see its meta
  `kiro_outputs.action_mapping`.
"""
# a direction set's own README (user 2026-10-14: every run delivers both weld directions; scan_core.g1_outputs.DIRECTION_SETS_RULE)
README_DELIVERY_SET = """## Delivery
- This is the `{name}` set of the run (`<dirname(data_file_path)>/{folder}/kiro/`): every pass welds {direction}
  ({dir_words}), built by the same plan from the same measured sections as the run's primary set - only the weld
  direction differs. The primary set (the job's weld_plan.direction {setting}) is `../../../kiro/`{same}; the PrecisionSensing
  action's result.paths and precision_sensing.npz carry the primary set only (meta kiro_outputs.directions lists this set).
- Weld every pass of this set {direction} and take its pass_<k>.yaml as it is: start / goal and the torch orientation are built for
  this travel (travel angle relative to it, work angle toward the bevel, TCP roll by the torch neck rule) - never swap start and goal
  of a file (the orientation is fixed in space). Per pass either set may be used: pass k of every set lays the same layer at the
  same aim points.
"""
README_DIRECTIONS = """
## Both weld directions
- This run delivers the weld plan in both directions (user decision 2026-10-14); take the set of the direction you weld in:{rows}
  These files (`kiro/`, the action's result.paths and precision_sensing.npz) are the job's direction (weld_plan.direction
  {setting}){same}.
- Weld every pass of a set in that set's direction and take its pass_<k>.yaml as it is: start / goal and the torch orientation are
  built for that travel (travel angle relative to it, work angle toward the bevel, TCP roll by the torch neck rule) - never swap
  start and goal of a file (the orientation is fixed in space). Per pass either set may be used: pass k of every set lays the same
  layer at the same aim points.
"""
_DIR_WORDS = {"S->E": "from the S end (the end the sensing run visited first) to the E end",
              "E->S": "from the E end to the S end (the end the sensing run visited first)"}


def _aim_words(plan) -> str:
    """README words for the lateral aim: per pass the rule (KIRO's practice or the geometric rule) and where it puts the wire end."""
    from .weld_plan import AIM_KIRO, _role_label            # noqa: PLC0415
    rec = dict(getattr(plan, "aim", None) or {})
    rows = rec.get("passes") or []
    if not rows:
        return "the per-pass aim record is missing"
    return "; ".join(f"pass {r_['k']} {_role_label(r_)}: " + str(r_.get("why", "")).split(": ", 1)[-1]
                     + ("" if r_.get("lateral") == AIM_KIRO else " (geometric rule)") for r_ in rows)


def _orient_words(plan) -> str:
    """README lines: per pass the WPS nominal and the chosen torch angles, the deviation and the clearance (weld_orient)."""
    o = getattr(plan, "orientation", None) or {}
    rows = o.get("passes") or []
    if not rows:
        return "\n  (no weld-pose clearance search for this plan: the WPS angles as planned)"
    req = (o.get("required") or {})
    out = [f"\n  Required clearance tool <-> specimen {float(req.get('required_mm', float('nan'))):.2f} mm"
           + (f" (k {req.get('k')} x {float(req.get('sigma_mm', float('nan'))):.2f} mm; not documented: "
              f"{', '.join(req.get('missing') or [])} - a lower bound)" if req.get("missing") else
              f" (k {req.get('k')} x {float(req.get('sigma_mm', float('nan'))):.2f} mm)" if req.get("k") is not None else "")
           + (f"; the gas nozzle and contact tip (designed to approach the work) "
              f"{float(req['by_role']['process']['required_mm']):.2f} mm - no interpenetration (KIRO practice)"
              if (req.get("by_role") or {}).get("process") else "") + ":"]
    for r in rows:
        nm, ch = r.get("nominal") or {}, r.get("chosen")
        if ch is None:
            out.append(f"\n  - pass {r['k']}: WPS travel {nm.get('travel_deg', float('nan')):+.0f} / work {nm.get('work_deg', float('nan')):+.0f} "
                       f"deg - no feasible orientation ({r.get('why')})")
        else:
            out.append(f"\n  - pass {r['k']}: WPS travel {nm['travel_deg']:+.0f} / work {nm['work_deg']:+.0f} deg -> travel "
                       f"{ch['travel_deg']:+.0f} / work {ch['work_deg']:+.0f} deg (deviation {float(r['deviation_deg']):.1f} deg), "
                       f"clearance {float(r['clearance_mm']):.2f} mm"
                       + "".join(f", gas nozzle + contact tip {float(v_['clearance_mm']):.2f} mm" for k_, v_ in
                                 (r.get("clearance_by_role") or {}).items() if k_ == "process")
                       + (f", limiting: {r['limiting_part']['role']} parts" if r.get("limiting_part") else ""))
    return "".join(out)


def _tip_words(et: RigidTransform) -> str:
    t = et.t * 1e3
    return f"({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) mm" + (", axes identity" if np.allclose(et.R, np.eye(3), atol=1e-12) else ", axes as given")


def _flange_words(torch: WeldTorch) -> str:
    t = torch.flange_from_tcp.t * 1e3
    r = torch.rpy_deg()
    return f"({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) mm, rpy ({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f}) deg"


def _set_location(name: str, folder: str, direction: str) -> str:
    """pass_file_location of a direction set's info files (a template with {k})."""
    return (f"<dirname(data_file_path)>/{folder}/kiro/pass_{{k}}.yaml: the {name} set (every pass {direction}) of the run whose "
            "PrecisionSensing data_file_path is precision_sensing.npz (it embeds the primary set's files only; its meta "
            "kiro_outputs.directions lists this set)")


def _directions_words(plan, sets: dict) -> str:
    """The primary README's section on the direction sets (README_DIRECTIONS): per set where it is, whether it is these files, or
    why it is not delivered."""
    from .g1_outputs import DIRECTIONS_DIR                       # noqa: PLC0415
    rows, same = [], ""
    for nm, ds in sets.items():
        where = f"`../{DIRECTIONS_DIR}/{nm}/kiro/`"
        what = f"every pass {ds.direction}, {_DIR_WORDS.get(ds.direction, ds.direction)}"
        if not ds.delivered:
            rows.append(f"\n  - `{nm}` ({what}): NOT delivered - {ds.reason}; its plan (if built) is in "
                        f"`../verification/{DIRECTIONS_DIR}/{nm}/`")
            continue
        rows.append(f"\n  - {where}: {what}" + (" (the same plan as these files)" if ds.same_as_primary else ""))
        if ds.same_as_primary:
            same = f" = the `{nm}` set"
    if not same:
        same = " - not one of the sets (the passes do not all weld one way)"
    return README_DIRECTIONS.format(rows="".join(rows), setting=str(plan.process.direction), same=same)


def write_kiro_outputs(plan, out_dir, cfg: KiroOutputConfig, *, world_from_base: RigidTransform, notes: Optional[list] = None) -> dict:
    """계획 → out_dir/kiro/ (pass_<k>.yaml, pass_<k>_info.yaml, README.md; 원자적). 패스 끝점·자세는 모듈 docstring: TCP (토치 팁 끝 중심) =
    겨냥점 − stick_out · z_TCP (계획의 토치 plan_torch), 자세 = 계획 토치 자세 · TCP 롤 (plan.torch_roll_deg).
    A plan that carries direction sets (g1_outputs.write_outputs: plan.directions, user 2026-10-14 - every run delivers both weld
    directions) also gets each DELIVERED set written in the same format to out_dir/directions/<set>/kiro/ (its own plan, or the
    primary plan object for the set that is the primary), and the primary README names the sets and how to choose; the files of the
    primary kiro/ are otherwise exactly those of a plan without sets.
    → {dir, files, passes: [{k, start/goal World 위치·쿼터니언, 겨냥점, 자세 각, 롤}], world_from_base, ee_from_tip, stick_out, notes} of
    the primary (+ directions: {name: {direction, delivered, same_as_primary, folder, dir, files, weld_plan_files, passes} | {...,
    delivered False, reason}} when the plan carries sets)."""
    if world_from_base is None:
        raise ValueError("no World <- base transform (resolve_world_from_base)")
    sets = dict(getattr(plan, "directions", None) or {})
    ko = _write_kiro_set(plan, out_dir, cfg, world_from_base=world_from_base, notes=notes,
                         directions=_directions_words(plan, sets) if sets else "")
    if not sets:
        return ko
    from .g1_outputs import DIRECTIONS_DIR, direction_plan    # noqa: PLC0415
    ko["directions"] = {}
    for nm, ds in sets.items():
        rec = {"direction": ds.direction, "delivered": bool(ds.delivered), "same_as_primary": bool(ds.same_as_primary),
               "folder": ds.folder, "weld_plan_files": list(ds.files)}
        p = direction_plan(plan, ds)
        if not ds.delivered or p is None:
            ko["directions"][nm] = jsonable({**rec, "delivered": False, "reason": ds.reason or "no plan for this set"})
            continue
        try:
            sub = _write_kiro_set(p, Path(out_dir) / DIRECTIONS_DIR / nm, cfg, world_from_base=world_from_base, notes=notes,
                                  location=_set_location(nm, ds.folder, ds.direction),
                                  delivery=README_DELIVERY_SET.format(
                                      name=nm, folder=ds.folder, direction=ds.direction, dir_words=_DIR_WORDS.get(ds.direction, ""),
                                      setting=str(plan.process.direction),
                                      same=(" - the same plan as this set" if ds.same_as_primary else "")))
        except Exception as e:  # noqa: BLE001 - one set's KIRO files never break the primary's (reason recorded)
            ko["directions"][nm] = jsonable({**rec, "kiro_written": False, "kiro_reason": f"{type(e).__name__}: {e}"})
            continue
        ko["directions"][nm] = jsonable({**rec, "kiro_written": True, "dir": sub["dir"], "files": sub["files"],
                                         "torch_roll": {k_: (sub["torch_roll"] or {}).get(k_) for k_ in ("deg", "reason")},
                                         "passes": [{k_: r_.get(k_) for k_ in _SET_PASS_KEYS} for r_ in sub["passes"]]})
    return ko


# the per-pass summary of a direction set in write_kiro_outputs' result (the primary keeps its full rows)
_SET_PASS_KEYS = ("k", "role", "start_World_m", "goal_World_m", "quat_xyzw", "tip_start_World_m", "tip_goal_World_m", "length_mm",
                  "torch_roll_deg", "wps_nominal_travel_angle_deg", "wps_nominal_work_angle_deg", "written_travel_angle_deg",
                  "written_work_angle_deg", "deviation_from_wps_deg", "clearance_mm", "required_clearance_mm")


def _write_kiro_set(plan, out_dir, cfg: KiroOutputConfig, *, world_from_base: RigidTransform, notes: Optional[list] = None,
                    location: Optional[str] = None, delivery: str = README_DELIVERY, directions: str = "") -> dict:
    """One set of KIRO-format files of one plan → out_dir/kiro/ (write_kiro_outputs; location = the info files' pass_file_location
    template, delivery / directions = the README's delivery section and its direction-sets section)."""
    notes = list(notes or [])
    torch = plan_torch(plan, cfg)
    roll = float(getattr(plan, "torch_roll_deg", 0.0) or 0.0)
    roll_rec = dict(getattr(plan, "torch_roll", None) or {})
    d = Path(out_dir) / KIRO_DIR
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():                                            # 같은 폴더의 앞선 출력 (패스 수가 바뀌었을 수 있다)
        if _PASS_RE.match(f.name) or re.match(r"^pass_\d+_(info|process)\.yaml$", f.name):
            f.unlink()
    wt, et = world_from_base, torch.tcp_from_tip
    passes = sorted(plan.passes, key=lambda p: int(p.k))
    bev_sign = 1.0 if str(getattr(plan, "bevel_side", "left")) == "right" else -1.0     # plan_weld 와 같은 기울임 쪽
    files, rows = [], []
    prev = None
    import yaml
    for wp in passes:
        k = int(wp.k)
        ln = weld_line(plan, wp, torch, roll)                            # THE delivered line (D14 checked these poses)
        ends = ln.ends
        tipW = {"start": wt.apply(ln.tip_start), "goal": wt.apply(ln.tip_goal)}
        R_W_ee = wt.R @ ln.R_tcp
        q = _R_to_quat(R_W_ee)
        Rq = {X: (R_W_ee, q) for X in ("start", "goal")}
        eeW = {"start": wt.apply(ln.tcp_start), "goal": wt.apply(ln.tcp_goal)}     # ee 원점 = TCP = 끝 − R·(ee 안의 끝 위치) (weld_line)
        doc = pass_doc(k, cfg.world_frame, eeW["start"], Rq["start"][1], eeW["goal"], Rq["goal"][1])
        p_yaml = d / f"pass_{k}.yaml"
        atomic_write_bytes(p_yaml, format_pass_yaml(doc).encode("utf-8"))
        # 쓴 자세의 토치 축 (ee ← 끝 의 z 축) 이 이 패스 심에 대해 이루는 각 (정책과 무관하게 같은 정의로)
        dW = tipW["goal"] - tipW["start"]
        nW = wt.R @ ends["start"]["n"]
        bevW = wt.R @ (bev_sign * ends["start"]["w"])
        a_start = Rq["start"][0] @ et.R[:, 2]
        meas = _angles(a_start, dW, nW, bevW)
        pose = {"plan_travel_angle_deg": float(wp.travel_angle_deg), "plan_work_angle_deg": float(wp.work_angle_deg),
                "written_travel_angle_deg": meas["travel_angle_deg"], "written_work_angle_deg": meas["work_angle_deg"],
                "angles": "AWS A3.0 travel / work angles of the torch axis (the ee <- tip z axis) to this pass's seam: travel + = "
                          "push, work + = toward the bevel face (our convention)",
                "torch_roll_deg": roll, "torch_roll_reason": str(roll_rec.get("reason") or "default roll 0 deg (no D14 roll choice)"),
                "_R_world_ee": Rq["start"][0]}
        o = dict(getattr(wp, "orientation", None) or {})
        t_nom, w_nom = float(getattr(wp, "travel_nominal_deg", float("nan"))), float(getattr(wp, "work_nominal_deg", float("nan")))
        pose.update({"wps_nominal_travel_angle_deg": t_nom if math.isfinite(t_nom) else float(wp.travel_angle_deg),
                     "wps_nominal_work_angle_deg": w_nom if math.isfinite(w_nom) else float(wp.work_angle_deg),
                     "deviation_from_wps_deg": o.get("deviation_deg"), "clearance_mm": o.get("clearance_mm"),
                     "required_clearance_mm": o.get("required_mm"),
                     "clearance_by_role": {k_: {"clearance_mm": v_.get("clearance_mm"), "required_mm": v_.get("required_mm"),
                                                "link": v_.get("link"), "closest_solid": v_.get("solid")}
                                           for k_, v_ in (o.get("clearance_by_role") or {}).items()},
                     "clearance_roles": ("process = the gas nozzle + contact tip (required 0 mm: no interpenetration, KIRO practice), "
                                         "other = every other tool part (required k x the combined sigma)"),
                     "limiting_part": o.get("limiting_part"),
                     "orientation_choice": o.get("why") or "not chosen by a weld-pose clearance search (the WPS angles as planned)",
                     "deviation_why": (o.get("deviation_why") or {}).get("text"),
                     "checked_on": (f"D14 checked these poses: the line's samples every <= {float(plan.line_step_mm):g} mm (the same "
                                    f"poses as this file and result.paths - {LINE_RULE})"
                                    if math.isfinite(float(getattr(plan, "line_step_mm", float("nan")))) else
                                    "no D14 on this line (no weld-pose clearance search for this plan)")})
        info = _info_doc(plan, wp, ends, pose, wt, cfg, prev, torch, location)
        p_info = d / f"pass_{k}_info.yaml"
        head = (f"# Information for sharing (hongik fine sensing), KIRO pass {k}: the KIRO settings our layer model assumed and the values\n"
                f"# derived from them - not commands. The pass itself (straight line start -> goal, orientation): pass_{k}.yaml.\n")
        atomic_write_bytes(p_info, (head + yaml.safe_dump(jsonable(info), sort_keys=False, allow_unicode=False)).encode("utf-8"))
        files += [str(p_yaml), str(p_info)]
        rows.append({"k": k, "role": wp.role, "start_World_m": eeW["start"], "goal_World_m": eeW["goal"],
                     "quat_xyzw": Rq["start"][1], "quat_goal_xyzw": Rq["goal"][1], "tip_start_World_m": tipW["start"],
                     "tip_goal_World_m": tipW["goal"],
                     "assumed_speed_mm_s": float(plan.process.params(wp.role).speed_mps) * 1e3,
                     "length_mm": float(np.linalg.norm(dW)) * 1e3,
                     "line_deviation_mm": info["derived"]["straight_line"]["measured_seam_deviation"],
                     **{k_: v for k_, v in pose.items() if not k_.startswith("_")}})
        prev = wp
    extra = "".join(f"\n- note: {n}" for n in notes)
    ident = bool(np.allclose(wt.R, np.eye(3), atol=1e-12) and np.allclose(wt.t, 0.0, atol=1e-12))
    world_def = (f"\n  Here {cfg.world_frame} is defined as the robot base frame `{plan.base_frame}` (identity: platform definition) - the poses "
                 "are in our base frame." if ident else "")
    from .weld_plan import end_insets_mm
    i0, i1 = end_insets_mm(plan)
    inset = ("" if not (i0 > 0.0 or i1 > 0.0) else
             f", moved inside the measured ends along the seam by the weld design's end inset (job weld_design.end_inset_mm): start "
             f"{i0:g} mm, goal {i1:g} mm (the info files list the measured ends)")
    readme = README.format(n=len(passes), schema=SCHEMA, aim_words=_aim_words(plan), ends_src=(rows and pass_endpoints(plan, passes[0])["source"]) or "-", inset=inset,
                           world_def=world_def,
                           world=cfg.world_frame, base=str(plan.base_frame), wsrc=wt.source or "-", wstat=wt.status or "-",
                           tcp=torch.tcp_frame or "-", esrc=torch.stick_out_source or "-", estat=cfg.ee_from_tip.status or "-",
                           stick=torch.stick_out_mm, stick_src=torch.stick_out_source or "-", tipoff=torch.tip_offset_mm,
                           noz=float("nan") if torch.nozzle_end_mm is None else torch.nozzle_end_mm, noz_src=torch.nozzle_end_source or "-",
                           roll=(f"{roll:g} deg - " + str(roll_rec.get("reason") or "the default (no D14 roll choice for this plan)")),
                           orient=_orient_words(plan),
                           torch_src=torch.source or "-", flange=torch.flange_frame, flange_src=torch.flange_from_tcp.source or "-",
                           flange_txt=_flange_words(torch),
                           torch_stat=torch.status or "-", tip_txt=_tip_words(et),
                           flange_stat=torch.flange_from_tcp.status or "-",
                           torch_note=(f"\n- The weld torch is on the arm that made this run ({plan.base_frame} is its base): weld-plan "
                                       "reachability (D14) is checked on that arm with this torch before a plan is delivered (the result: "
                                       "../weld_plan.json reachability / feasibility; an offline regeneration does not check it)."
                                       if torch.on_this_arm else
                                       "\n- The platform config says the weld torch is not on the arm that made this run - the poses here "
                                       f"are the weld torch's. {D14_NOT_ON_THIS_ARM}."),
                           wps=str(plan.process.source), extra=("\n## Notes" + extra + "\n") if extra else "", delivery=delivery,
                           directions=directions)
    p_readme = d / "README.md"
    atomic_write_bytes(p_readme, readme.encode("utf-8"))
    files.append(str(p_readme))
    return jsonable({"dir": str(d), "files": files, "schema": SCHEMA, "frame_id": cfg.world_frame, "passes": rows,
                     "world_from_base": wt.to_dict(), "ee_from_tip": et.to_dict(), "tcp_frame": torch.tcp_frame,
                     "stick_out_mm": torch.stick_out_mm, "stick_out_source": torch.stick_out_source,
                     "nozzle_end_mm": torch.nozzle_end_mm, "tcp_to_aim_point_mm": torch.tip_offset_mm,
                     "tcp_definition": "TCP = the torch tip end centre (contact-tip end, inside the gas nozzle); each pose = the bead aim "
                                       "point (wire tip) moved back along -z_TCP by nozzle_end_mm + stick_out_mm (stick-out from the "
                                       "nozzle end); result.paths = the aim-point (wire-tip) lines",
                     "torch_roll": roll_rec or {"deg": roll}, "weld_torch": torch.to_dict(), "notes": notes,
                     "info_files": "information for sharing (assumed KIRO settings + derived values), not commands"})


# --------------------------------------------------------------------------------------
# 오프라인 재생성 (실행 폴더의 마지막 추정 + 사전 → weld_plan.* + kiro/)
# --------------------------------------------------------------------------------------
def _latest_estimate(run_dir: Path) -> Path:
    ests = sorted((run_dir / "verification" / "estimates").glob("est_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not ests:
        raise FileNotFoundError(f"{run_dir}/verification/estimates/est_*.json: no estimate")
    return ests[-1]


def _offline_nozzle_end(torch: WeldTorch, run: Path, urdf: Optional[str]) -> WeldTorch:
    """The offline torch's nozzle end: from the given URDF's collision geometry (as the run derives it on its arm), else the run's own
    record (weld_plan.json, or verification/weld_plan.json, weld_torch.nozzle_end_mm). ValueError when neither gives it."""
    from .weld_clearance import nozzle_end_mm, tool_body_from_urdf      # noqa: PLC0415
    if urdf:
        text = str(urdf) if str(urdf).lstrip().startswith("<") else Path(urdf).read_text(encoding="utf-8")
        if f'name="{torch.tcp_frame}"' in text:
            body = tool_body_from_urdf(text, torch.tcp_frame)
        else:                                                      # a TCP the URDF does not carry: the tool on the flange (config)
            body = tool_body_from_urdf(text, torch.tcp_frame, root_frame=torch.flange_frame, T_root_tcp=torch.flange_from_tcp.matrix())
        return torch.with_nozzle_end(nozzle_end_mm(body), "derived offline from the URDF's collision geometry (the tool parts the "
                                                          f"torch axis passes through, farthest along +z_TCP = the gas nozzle rim; "
                                                          f"{body.describe()})")
    for p in (run / "weld_plan.json", run / "verification" / "weld_plan.json"):
        if p.exists():
            rec = (json.loads(p.read_text(encoding="utf-8")).get("weld_torch") or {})
            if rec.get("nozzle_end_mm") is not None:
                return torch.with_nozzle_end(float(rec["nozzle_end_mm"]), f"the run's record ({p.name}): "
                                                                          f"{rec.get('nozzle_end_source') or '-'}")
    raise ValueError("not known: the run's weld_plan.json has no weld_torch.nozzle_end_mm (a run before 2026-10-13) - pass --urdf "
                     "with the welding robot's URDF")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m scan_core.kiro_plan",
        description="Offline: rebuild a 1G run's weld plan (weld_plan.json/.npz) from its final estimate and prior (verification/"
                    "estimates/est_<n>.json, verification/run_log.json) with a weld process, and write the KIRO-format files "
                    "(kiro/pass_<k>.yaml, pass_<k>_info.yaml, README.md) - the job's direction, plus both weld directions as "
                    "complete sets in directions/S_to_E/ and directions/E_to_S/ (weld_plan.* + kiro/; no D14 offline). "
                    "World <- base comes from the platform config "
                    "(kiro_output.world_from_base; no TF offline); the TCP poses are the platform config's weld torch model "
                    "(weld_torch).")
    ap.add_argument("run_dir", help="a 1G core run folder (sections.json, verification/)")
    ap.add_argument("--platform-config", required=True, help="platform YAML with kiro_output and weld_torch sections")
    ap.add_argument("--out", help="output folder (default <run_dir>/kiro_regen); the run folder itself is never written")
    ap.add_argument("--job", help="job YAML whose weld_plan / weld_design blocks are the process and the weld design (default: the "
                                  "WeldProcess / WeldDesign defaults = KIRO WPS and the KIRO-derived cap reinforcement)")
    ap.add_argument("--process-from-plan", help="weld_plan.json whose 'process' block is the process (e.g. the run's own plan)")
    ap.add_argument("--urdf", help="the welding robot's URDF (path or XML; meshes by package:// or file://): the weld torch's nozzle end "
                                   "(the reference of the wire stick-out) is derived from its collision geometry (default: the run's "
                                   "weld_plan.json weld_torch.nozzle_end_mm, as that run derived it)")
    ap.add_argument("--end-inset-mm", type=float, nargs=2, metavar=("START", "GOAL"),
                    help="the weld design's end inset [mm]: every pass starts START and ends GOAL inside the measured seam ends "
                         "(overrides the job's weld_design.end_inset_mm; default: the job's, else 0)")
    a = ap.parse_args(argv)
    import yaml
    from .g1_outputs import write_outputs
    from .g1_types import SeamEstimate, SeamPrior
    from .weld_plan import WeldProcess, weld_design_from_doc
    run = Path(a.run_dir).expanduser().resolve()
    out = Path(a.out).expanduser().resolve() if a.out else run / "kiro_regen"
    if out == run:
        print("--out must not be the run folder (its outputs stay as recorded)", file=sys.stderr)
        return 2
    with open(a.platform_config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    params = raw["/**"]["ros__parameters"] if "/**" in raw else raw
    sec = dict(params.get("kiro_output") or {})
    cfg = KiroOutputConfig.from_dict(sec, weld_torch_config(params))
    if cfg.torch is not None and cfg.torch.nozzle_end_mm is None:     # the stick-out's reference: the torch's gas nozzle end
        try:
            cfg = dataclasses.replace(cfg, torch=_offline_nozzle_end(cfg.torch, run, a.urdf))
        except (OSError, ValueError, KeyError) as e:
            print(f"weld torch nozzle end: {e}", file=sys.stderr)
            return 2
    log = json.loads((run / "verification" / "run_log.json").read_text(encoding="utf-8"))
    est_path = _latest_estimate(run)
    est = SeamEstimate.from_dict(json.loads(est_path.read_text(encoding="utf-8")))
    prior = SeamPrior.from_dict(log["prior"])
    jr = log.get("job_rough") or {}
    base_frame = str(jr.get("base_frame") or "link0")
    jdoc = {}
    if a.process_from_plan:
        process = WeldProcess.from_dict(json.loads(Path(a.process_from_plan).read_text(encoding="utf-8"))["process"])
    elif a.job:
        with open(a.job, "r", encoding="utf-8") as f:
            jdoc = yaml.safe_load(f) or {}
        process = WeldProcess.from_dict(jdoc.get("weld_plan"))
    else:
        process = WeldProcess()
    design = weld_design_from_doc(jdoc, process)
    if a.end_inset_mm is not None:
        if min(a.end_inset_mm) < 0.0:
            print("--end-inset-mm: the insets must be >= 0", file=sys.stderr)
            return 2
        design = dataclasses.replace(design, end_inset_start_mm=float(a.end_inset_mm[0]), end_inset_goal_mm=float(a.end_inset_mm[1]),
                                     end_inset_source="offline --end-inset-mm")
    res = write_outputs(est, prior, process, out, base_frame=base_frame, gravity_axis_base=np.asarray(jr.get("gravity_axis_base")
                                                                                                      or (0.0, 0.0, -1.0), dtype=float),
                        k=float(prior.k), run={"notes": [f"offline regeneration from {est_path} (scan_core.kiro_plan)"]}, design=design,
                        torch=cfg.torch)
    plan = res["plan"]
    from .g1_outputs import DIRECTIONS_DIR                            # noqa: PLC0415

    def recorded_roll(p_, rp):                                        # the run's TCP roll (chosen by its D14), if it recorded one
        tr = (json.loads(rp.read_text(encoding="utf-8")).get("torch_roll") or {}) if rp.exists() else {}
        if tr.get("deg") is not None:
            p_.torch_roll_deg = float(tr["deg"])
            p_.torch_roll = dict(tr, reason=f"{tr.get('reason') or 'recorded'} (the run's choice, {rp.relative_to(run)})")
    recorded_roll(plan, run / "weld_plan.json")
    for nm, ds in (getattr(plan, "directions", None) or {}).items():  # each direction set: its own record in the run, if any
        if ds.plan is not None:
            recorded_roll(ds.plan, run / DIRECTIONS_DIR / nm / "weld_plan.json")
    wt, wnotes = resolve_world_from_base(cfg, base_frame)
    print(f"estimate {est_path.name}, prior {run.name}/verification/run_log.json, process: {process.source}")
    for line in plan.summary_lines():
        print(line)
    if wt is None:
        print("KIRO outputs not written: " + "; ".join(wnotes), file=sys.stderr)
        return 1
    ko = write_kiro_outputs(plan, out, cfg, world_from_base=wt, notes=[f"offline regeneration from {run} ({est_path.name})"] + wnotes)
    for f in res["files"] + ko["files"]:
        print(f"wrote {f}")
    for nm, d in (ko.get("directions") or {}).items():               # both weld directions (user 2026-10-14)
        if d.get("delivered"):
            for f in list(d.get("weld_plan_files") or []) + list(d.get("files") or []):
                print(f"wrote {f}")
        else:
            print(f"direction set {nm} ({d.get('direction')}) not delivered: {d.get('reason')}"
                  + "".join(f"; kept {f}" for f in d.get("weld_plan_files") or []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
