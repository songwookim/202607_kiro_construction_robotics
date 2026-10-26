"""용접 전 스캔 (pre-weld scan) → 다층·다비드 용접 경로 (사용자 2026-10-03: 용접 경로 생성까지가 용접 전 스캔의 역할; 2026-10-04: 로직을
먼저 세우고 데이터로 검증한다 — 우리 시편만이 아니라 V형·루트면·두꺼운 판·넓은 그루브에 같은 로직).

사전 정보 (job 문서): 자세 (1G/3G), 그루브 형상 (GrooveSpec: kind·두께·개선각·루트면·갭·뒷댐재), 용접 절차 (WeldProcess = WPS: 역할
root/fill/cap 별 WFS·진행 속도·토치 각도, 와이어). 스캔이 주는 것: 심 방향 단면마다 실측 갭·깊이·개구·hi-lo 와 국소 좌표.

모델
  단면 (bead_fill 과 같은 정의): 벽 = 루트면 (수직, 높이 rf) + 개선면 (각 β, 개선된 판 수 n_b). 높이 z 의 폭 W(z) = g + n_b·tanβ·r,
    그 아래 단면적 A(z) = g·z + n_b·tanβ·r²/2, r = max(z − rf, 0).
  비드 (bead_model.BeadModel — 기본은 질량 보존 a = η·(πd²/4)·WFS/v + 원호 단면 폭; KIRO 데이터 기반 모델로 갈아 끼운다):
    역할별 공칭 비드 단면적 a_r, 폭 w_r, 볼록 단면의 crown ratio (모양 변수 하나 = 비드 토우각 θ_toe = bead_toe_angle_deg, 높이/폭
    κ = tan(θ_toe/2)/2; 위빙이면 같은 단면적의 위빙 폭).
  Bead geometry (user + KIRO 2026-10-13: beads are convex, not flat-filled layers - scan_core.bead_geometry, the ONE geometry of the
    plan): each pass deposits its mass-balance area on the current surface (walls + earlier beads), fills the lowest region it wets
    (a level pool from its aim, spilling over ridges / flats, over the plate tops once the groove is full) and is bounded on top by a
    circular arc over its toe chord with height κ_p x chord (the free bead's shape); the layer heights, the surface the aims meet, the
    D14 weld scene (weld_scene: earlier beads as convex solids) and every output come from these beads.
  층: 층 j 의 비드 n_j 개 — 층의 수평 풀 (볼록 표면 위, 비드 단면적 합) 의 폭을 비드 폭으로 덮는 최소 개수 (n_j > 1: 풀을 등면적 세로
    띠로 나눠 비드마다 띠 하나). 캡: 판 상면 + 양쪽 cap_overlap 폭 W_c 를 cap_weave_width_max 이하 폭의 위빙 패스 n_c 개가 덮고, 보강 =
    캡 뒤 표면이 그루브 상부 모서리를 잇는 선 위로 올라온 높이 (비드 기하로).

계획 (기준 단면 = 실측 단면들의 강건 중앙값; plan_topology)
  1. λ 하나마다 층을 아래부터 쌓는다 (비드 = λ·a_r, n_j 는 폭 조건; 층의 풀이 그루브 상부 모서리를 넘으면 멈추고 나머지는 캡).
  2. 충전 비드의 공통 용착 배율 λ (WFS 고정이면 공통 속도 배율 1/λ) 와 캡 배율 μ (설계 보강을 내는 캡 단면적 — 비드 기하로) 를 고른다:
     모든 패스의 비드 단면적이 WPS 속도 창 (speed_limits) 안에서 Σ ln²(a/a_nom) 최소 — 진행 속도 로그 편차의 제곱합. KIRO 공정
     (충전 3 + 위빙 캡 1) 은 10T/30°/갭 5 단면에서 기본 WPS 의 특수해다 (test_weld_plan).
  3. 단면마다: 층 j 까지 용착 = 그 단면 단면적의 같은 비율 F_j (모든 단면에 같은 패스 — 패스는 심 방향으로 끊기지 않는다), 속도 = 비드
     모델 역산 (WFS 고정 → 같은 용착률; 속도 창으로 자르고 표시), 비드 = 그 용착 단면적을 그 단면의 지금 표면에 놓은 볼록 비드. 겨냥
     (AIM_RULE, KIRO's practice - user + KIRO 2026-10-13 night): vertical = the wire end meets the current surface at the aim
     point (the specimen before the pass); lateral, from the measured section's features (any single bevel; mirrors with the bevel
     side resolved from the scan): root = the corner of the root face and the bevel face, fill 1 = the corner where the root bead
     meets the square face (its toe on the square wall), fill 2 = the lateral centre of the groove at its current surface (between
     fill 1's toes), cap = the lateral centre of the groove top (midway between the two groove top edges). Layers outside KIRO's
     4-pass practice (a 3rd fill, multi-bead layers, several cap passes, V / square grooves) keep the geometric rule (the bead's band
     centre), labelled per pass. KIRO's 2026-09-22 per-pass offsets are kept as a verification only (KIRO_0922, plan aim record).
  이 예측 층 높이 (stations[].levels_mm: 층의 벽 토우 높이, 패스의 crown) 를 패스 간 스캔 (inter-pass scan) 이 실측과 비교한다.

입력 = 마지막 패스 (또는 패스 병합) 의 프로파일별 특징 (autoteach._features_arrays / scan_quality.merge_feature_arrays: 루트·상부 중심
base 폴리라인, 갭·깊이·개구·hi-lo·베벨 쪽, 판 법선 n) + 관측된 양끝 S/E — **base 점** (S→E 계약, scan_core.seam_frame: 축 u = unit(E − S),
심 방향 좌표 = S 에서 잰 거리; 패스 진행 방향·정련 직선의 u 와 무관 — 2026-10-09, E→S 마지막 패스의 거울상 계획). 심 방향
station_step_mm 마다 ±station_window_mm/2 창의 강건 중앙값으로 그 자리의 단면 (GrooveSpec; 루트면은 설계값) 과 국소 좌표 (원점 = 루트갭
중심·뒷댐재 상면, u = S→E, n = 판 법선 (그루브가 열린 쪽), w = n × u) 를 만든다. 계획은 계약 검사 (check_plan_contract: 첫 단면 = S,
마지막 = E (k·σ_끝 안), u·(E − S) > 0, 패스마다 표기한 방향의 끝에서 시작) 를 통과해야 나온다 — 어기면 PlanContractError (계획 거부).

단면 좌표 (bead_fill·GrooveSpec 과 같다): x [mm] 는 w 방향 (루트갭 중심 0), z [mm] 는 n 방향 (뒷댐재 상면 0). 베벨 쪽은 상부 개구 중심이
루트갭 중심에서 치우친 쪽 (센서 좌우와 무관).

TCP 경로: 계획의 TCP = 끝 = 와이어 끝 (겨냥점; 플랫폼 토치의 TCP 자세는 weld_torch 의 TCP ← 끝 으로 p<k>_tcp_* — 우리 토치: 컨택트 팁
끝 = 끝에서 토치 축으로 20 mm 뒤). 토치 축 a (토치 → 공작물) = −n 을 작업각만큼 옆으로 (진행에 수직인 면에서; 층에 비드 하나: + = 와이어 /
토치 축이 베벨면 쪽으로 기운다 — work_angle; 여럿: 벽에 붙는 비드가 그 벽 쪽으로 side_work_angle), 그리고 진행각만큼 진행 방향으로 (+ = 미는 각) — AWS A3.0 의
진행각·작업각 정의 (_tcp_R). TCP 자세 R = [x = 진행 방향 (a 에 수직),
y = a × x, z = a]. 캡 위빙은 로봇 제어기의 위빙 기능이 아니라 **촘촘한 TCP 경로** (MoveIt 이 그대로 따라감; 모든 이동은 MoveIt — 사용자
2026-10-03) 로 만든다.

공정값 (WPS) = KIRO 데이터셋 (2026-09-15/16 weld_testspecimen_logs, 사용자 승인 2026-10-12): 역할별 전류·전압·WFS·속도, 설정선
WFS(I)·U(I), 캡 위빙 (±4 mm, tool_y), 리드인·아웃, 아크 끄기 지연, 그리고 진행각·작업각 = KIRO 가 9 월에 실제로 용접한 각 (2026-09-15/16 +
09-22 로그, 식별한 토치의 노즐 축; 사용자 승인 2026-10-13) — WeldProcess 기본값 (출처 provenance) 과 1G 작업
템플릿의 weld_plan: 블록이 같은 값 (test_weld_plan 이 대조), job 문서 weld_plan: 블록으로 바꾼다. KIRO 의 설정은 그대로 (사용자 2026-10-12:
우리가 설정을 명령으로 보내지 않는다) → 속도 창 [1, 1] (속도 적응 없음): 단면마다 실제 용착 = 그 설정의 비드 단면적, 층 높이 = 그 용착을
아래부터 쌓은 높이 (용착 일관 — 창으로 자른 속도의 단면적을 쌓는다; 창 안이면 계획 비율 그대로), 캡 보강 = 고정 캡의 결과 (목표값 없음 =
cap_reinforcement_mm null). 일정 속도 실행기 (KIRO: 패스마다 직선·일정 자세·일정 TCP 속도) 의 정보로 패스마다 **일정 속도 하나**
(speed_const_mps: 필요 단면적의 심 방향 평균 → 질량 보존 속도, 창으로 자름) 와 단면마다의 필요 속도 범위를 보고한다 — 전류가 아니라 속도를
보는 까닭: KIRO 가 검증한 아크 설정 (전류·전압·WFS 세 점) 을 그대로 두고 단면적 ∝ 1/v 가 정확한 (WFS 고정) 쪽; 설정선 위의 등가 전류는
정보로만 (kiro_plan 의 pass_<k>_info.yaml).
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .bead_fill import groove_area_below_mm2, wall_x_mm
from .bead_geometry import BEAD_RULE, bare_profile, groove_edges_mm, place_bead, pool_cuts, pool_level
from .bead_geometry import _layer_span
from .bead_model import (BeadModel, MassBalanceBeadModel, PassParams, crown_ratio_for_toe_angle, crown_ratio_of,
                         toe_angle_for_crown_ratio)
from .measure import K_DEFAULT
from .seam_frame import SeamFrame, SeamFrameError, frame_from_extent, seam_frame
from .specimen import GrooveSpec
from .weld_torch import WeldTorch

__all__ = ["WeldProcess", "WeldDesign", "GrooveStation", "WeldPass", "WeldPlan", "Layer", "Topology", "stations_from_features", "plan_topology",
           "plan_weld", "weld_plan_from_features", "format_plan_summary", "weld_design_from_doc", "PLAN_SCHEMA", "ROLES", "plan_frame", "check_plan_contract",
           "PlanContractError", "end_insets_mm", "AIM_RULE", "AIM_KIRO", "AIM_GEOMETRIC", "aim_tip_offset_mm", "KIRO_0922",
           "kiro_0922_toward_bevel_mm", "SECTION_STEP_MM"]

PLAN_SCHEMA = 3                         # 3 (2026-10-13 night): convex beads (bead_geometry), KIRO's aim practice
DIRECTIONS = ("S->E", "E->S", "alternate", "uphill")
ROLES = ("root", "fill", "cap")
MAX_BEADS_PER_LAYER = 12
# the section surface's sampling for the bead geometry (bead_geometry.bare_profile) [mm] - numerical resolution: the arc between two
# points lies at most step^2 / (8 R) above the polyline (2e-4 mm for a 10 mm, kappa 0.3 bead); walls and edges are exact points
SECTION_STEP_MM = 0.1
_LEGACY_KEYS = {"fill_fractions", "fill_speed_mps", "cap_speed_mps"}
_REMOVED_KEYS = {"aim_height_frac": "removed 2026-10-13 (user): the wire end meets the current surface - the vertical aim is the top of the layer below (the bottom of the new layer; the backing top for the root), not a fraction of the new layer",
                 "aim_lateral_tcp_mm": "removed 2026-10-13 night (user + KIRO): the lateral aim follows KIRO's practice from the measured section (root: the root-face / bevel-face corner; fill 1: the root bead's toe on the square face; fill 2: the groove's lateral centre at its current surface; cap: the groove top's lateral centre) - KIRO's 09-22 per-pass offsets are a verification only (weld_plan.KIRO_0922)",
                 "aim_lateral_axis": "removed 2026-10-13 night (user + KIRO): see aim_lateral_tcp_mm - the 09-22 table is a verification only (weld_plan.KIRO_0922)"}
AIM_KIRO = "kiro_practice"              # per-pass label: the lateral aim follows KIRO's practice (root, fill 1, fill 2, cap of a single bevel)
AIM_GEOMETRIC = "geometric"             # per-pass label: the bead's band centre (layers KIRO's practice does not cover)
AIM_RULE = {"vertical": "the wire end meets the current surface at the aim point: the specimen before the pass - the plates, the "
                        "backing and the earlier passes' convex beads (bead_geometry; weld_scene shows the same solids)",
            "lateral": "KIRO's practice (user + KIRO 2026-10-13 night, their description of the September welds), from the measured "
                       "section's features - any single-bevel geometry, mirrored with the bevel side resolved from the scan: root = the "
                       "corner of the root face and the bevel face (the bevel-side wall at the root-face height); fill 1 = the corner "
                       "where the root bead meets the square face (the root bead's toe on the square wall); fill 2 = about the lateral "
                       "centre of the groove at its current surface (midway between fill 1's toes); cap = the lateral centre of the "
                       "groove top (midway between the two groove top edges at the plate surfaces - the cap band's centre)",
            "uncovered": "layers KIRO's 4-pass practice does not cover (a 3rd and later fill, multi-bead layers, several cap passes, "
                         "groove kinds without a single bevel) keep the geometric rule: a single-bead layer at the groove's lateral "
                         "centre at its current surface (the root: the root-gap centre), a bead of a multi-bead layer at the centre of "
                         "its equal-area band of the layer's pool at mid-height, a cap pass at the centre of its band of the cap width "
                         "(toe to toe + cap_overlap_mm each side)",
            "verification": "KIRO's 2026-09-22 per-pass offsets (TCP + stick-out x nozzle axis, relative to pass 1) are not a rule any "
                            "more - the plan reports the rule against them (aim record check_0922; KIRO_0922)"}
_ROLE_VALUES = ("wfs_mpm", "speed_mps", "current_a", "voltage_v", "lead_in_mm", "lead_out_mm", "arc_off_delay_ms")
_ROLE_ANGLES = ("travel_angle_deg", "work_angle_deg")      # 역할마다 값 하나, 또는 그 역할의 층 순서별 목록 (마지막 값이 이어진다)

# KIRO's 2026-09-22 session (verification only, 2026-10-13 night): per pass the wire end relative to pass 1 at L = TCP -> wire end,
# lateral = (tcp - tcp_root) + L (axis - axis_root) toward the RIGHT of KIRO's travel (the pass-1 line frame w = u x n); the bevel was
# on the LEFT of KIRO's travel (user 2026-10-13 night), so toward the bevel = -that. Rise relative to pass 1: the spread of the tool
# identification (95 % (psi, phi) region at stick-out 15 / 20 mm and its 'free point' fit - fill 1 is poorly determined).
KIRO_0922 = {
    "tcp_right_mm": {"root": 0.0, "fill 1": -5.34, "fill 2": -1.63, "cap": 0.04},
    "axis_right": {"root": -0.0356, "fill 1": 0.3097, "fill 2": -0.0358, "cap": -0.0311},
    "rise_mm": {"root": (0.0, 0.0), "fill 1": (3.0, 9.5), "fill 2": (5.2, 6.5), "cap": (13.1, 13.9)},
    "bevel": "left of KIRO's travel (user 2026-10-13 night): toward the bevel = -(toward the right of travel)",
    "source": "KIRO 2026-09-22 session (weld_20260922_four_logs_arc_on_tcp: taught weld_start / weld_end joints, FK, tool URDF v2 r4); "
              "lateral: fsv1_work/kiro_dataset2/aim0922/derive_aim0922.py (aim0922_table.json; start / end averaged); rise: "
              "fsv1_work/kiro_dataset2/toolid/toolid_run.out.txt section F (relative to pass 1)",
}


def kiro_0922_toward_bevel_mm(label: str, tip_offset_mm: float) -> Optional[float]:
    """KIRO's 09-22 wire end of pass `label` ('root' / 'fill 1' / 'fill 2' / 'cap') toward the bevel from pass 1's [mm] at L =
    tip_offset_mm (TCP -> wire end) - the bevel on the LEFT of KIRO's travel (KIRO_0922); None for a pass the session did not have."""
    t, a = KIRO_0922["tcp_right_mm"], KIRO_0922["axis_right"]
    if label not in t:
        return None
    return -((t[label] - t["root"]) + float(tip_offset_mm) * (a[label] - a["root"]))


WPS_SOURCE = "KIRO dataset 2026-09-15/16, weld_testspecimen_logs"
WPS_PROVENANCE = {
    "source": WPS_SOURCE + " (1.log-4.log, pass_1.yaml-pass_4.yaml; analysis fsv1_work/kiro_dataset, user approval 2026-10-12)",
    "process": "LSM, CO2 100%, FE-SOLID 1.2 mm, non-synergic; per role (root / fill / cap): 190 / 230 / 220 A, 24.5 / 26.5 / 26.0 V, "
               "WFS 8.4 / 10.8 / 10.2 m/min, speed 5 / 7 / 6 mm/s ([commanded] + [execution_conditions] of the logs)",
    "settings_lines": "WFS = 0.06 I - 3.0 m/min and U = 15 + 0.05 I V: exact on the three settings 190 / 220 / 230 A",
    "deposition_efficiency": "0.95: KIRO root + 2 fills (mass balance) = 85.2 mm2 = the measured 10T groove of our k20_2_a run (85.5 mm2) "
                             "-> eta 0.953 fills it flush (KIRO gave no groove drawing: same 10T / 30 deg / gap 5 design assumed)",
    "speed_limits": "[1, 1]: KIRO's settings stay as they are (user 2026-10-12: we do not send settings as commands) - no speed "
                    "adaptation; the layer heights follow the fixed deposits",
    "cap_reinforcement_mm": "not a process value any more: the weld design input weld_design.cap_reinforcement_mm (WeldDesign; "
                            "the KIRO-derived 1.87 +- 0.19 mm by default - convex beads, toe angle 30 deg); the process key is read from old jobs only",
    "geometry": "KIRO's specimen = ours (10T, 30 deg single bevel, gap 5, backing; user 2026-10-12): the plan uses the measured "
                "sections (k20_2_a reference: depth 10.5, gap 5.2, bevel 30.5 deg -> 85.5 mm2; the nominal design is 78.9 mm2)",
    "angles": "KIRO welds 2026-09-15/16 + 09-22, tool identified (mount -90 deg, torch flipped), nozzle axis; work-angle sign: the "
              "bevel on the LEFT of KIRO's travel (user 2026-10-13 night - corrects the earlier 'right'). Per role and layer order "
              "(root / fill 1 / fill 2 / cap = KIRO passes 1-4): the AWS A3.0 angles of the nozzle axis at KIRO's taught weld_start "
              "poses (FK of the logged joints; plate normal = World +z; weld axis = the session's pass-1 line), travel (+ push / - drag) "
              "09-15/16 -44.3 / -44.5 / -44.1 / -44.2, 09-22 -39.8 / -38.2 / -43.8 / -44.8 deg; work in KIRO's sense (+ = the "
              "electrode tip leans toward the right of travel) 09-15/16 -11.5 / +20.6 / -7.0 / -7.0, 09-22 -6.6 / +19.3 / -7.1 / -6.8 "
              "deg; the defaults = the mean of the two sessions rounded to 1 deg: travel -42 / -41 / -44 / -44 (drag; KIRO "
              "confirmed 2026-10-13 that they weld backhand / drag - kept as the default; a site sets another with the run-time "
              "override weld_plan.travel_angle_deg, and the TCP roll follows its sign, scan_core.weld_orient NECK_RULE); work in OUR "
              "convention (+ = the wire / torch axis toward the work tilts toward the bevel face, _tcp_R) with the bevel on the left of "
              "KIRO's travel = minus KIRO's sense: +9 / -20 / +7 / +7. KIRO's description (2026-10-13 night) agrees: root +9 = the wire "
              "toward the root-face / bevel-face corner; fill 1 -20 = the wire into the square-face corner; fill 2 and cap +7 = the "
              "torch leans slightly toward the square wall so the wire tip looks at the bevel face. (Was -9 / +20 / -7 / -7 under "
              "the earlier 'bevel on the right' assumption, 2026-10-13.) Tool = our 1st bracket bolted at mount -90 deg (RB tcp "
              "flange convention, as in the lab) with the CAD torch (30 deg bend) turned in its clamp about its straight body axis by "
              "173.9 +- 1.5 deg (the fit to the logs with the bolted mount; at the nominal 180 deg the work angles read ~4 deg further "
              "toward the right of travel) and the wire end 20 mm beyond the contact-tip end (analysis fsv1_work/kiro_dataset2/toolid, "
              "2026-10-13). Replaces the void travel +14 / work 22-55-28-28 (KIRO's logged poses read with -x of KIRO's old "
              "right_manipulator_ee_point, the chiseling-gun frame, as the torch axis)",
    "aim_rule": "KIRO's lateral aim practice (user + KIRO 2026-10-13 night: their description of the September welds) from the "
                "measured section: root = the corner of the root face and the bevel face; fill 1 = the corner where the root bead "
                "meets the square face; fill 2 = about the lateral centre of the groove at its current surface; cap = the lateral "
                "centre of the groove top (midway between the groove top edges; user refinement 2026-10-13 night). Vertical: the wire "
                "end meets the current surface (the convex beads of the passes before). Replaces the 09-22 per-pass reproduction "
                "(aim_lateral_tcp_mm / aim_lateral_axis), kept as a verification (KIRO_0922)",
    "bead_geometry": "convex beads (user + KIRO 2026-10-13 night; scan_core.bead_geometry): each pass's mass-balance area on the "
                     "current surface, filling the lowest region it wets (a level pool from the aim; multi-pass groove filling models "
                     "fill bottom-up the same way - equal-area layers with trapezoid / parallelogram beads, e.g. the equal "
                     "cross-section planning in Adv. Manuf. 5:93-104, 2017 - here with the free bead's convex top), bounded by a "
                     "circular arc over the toe chord with height kappa_p x chord - the circular arc because GMAW bead sections follow "
                     "an arc better than a parabola at wire-feed / travel speed ratios above 12.5 (Xiong, Zhang, Gao, Wu 2013, Robot. "
                     "Comput.-Integr. Manuf. 29:417-423; parabola / arc / cosine bead models also in Suryakumar et al. 2011, "
                     "Comput.-Aided Des. 43:331-344) and KIRO's WPS runs at 28 / 26 / 28. One shape parameter: the bead's toe "
                     "(flank) angle bead_toe_angle_deg (provenance['bead_toe_angle_deg']) -> kappa = tan(toe / 2) / 2 for the arc; "
                     "kappa_p = kappa without weave, the cap's free bead spreads over the woven width with the same area (flatter). "
                     "It sets the free bead's width (the bead count) too",
    "bead_toe_angle_deg": "30 deg (user 2026-10-13 night): a typical value of the literature range for as-welded GMAW weld toe / "
                          "reinforcement flank angles (~20-40 deg) - not verified against a source for this process; replaces the "
                          "kappa 0.3 of bead_aspect_ratio (a 62 deg toe: too round for groove beads, a third of the area in the dome). "
                          "To be calibrated with KIRO's post-weld scans per pass (bead_geometry.crown_ratio_from_profiles: kappa_p -> "
                          "toe = 2 atan(2 kappa_p))",
    "cap_weave": "KIRO executor: sine, amplitude 4.0 mm along tool_y, 16 cycles (on its 119.6 mm pass: pitch 7.48 mm; pitch set "
                 "7.8 mm), dwell 0.2 / 0.2 s, 12 samples per cycle (pass 4 log); our cap model weaves +-4 mm along the torch y axis "
                 "(tool_y of our torch's TCP = the tip frame's y = lateral; KIRO's tool_y was its old ee_point's)",
    "lead_and_arc_off": "lead-in / lead-out 1 / 6 mm and arc-off delay 500 ms on root and fill, 0 / 0 mm and 0 ms on the cap",
    "stick_out_mm": "the wire stick-out from the GAS NOZZLE END (the nozzle rim) to the wire end along the torch axis (welding "
                    "practice; user 2026-10-13: KIRO's 15-20 mm and our 20 mm are nozzle end -> wire end); provisional 20.0 mm (KIRO's "
                    "range 15-20 mm; not in KIRO's weld logs - checked 1.log-4.log and pass_1-4.yaml: no stick-out / CTWD, "
                    "wire_consumable_alpha_mm is a wire-consumption term); replace with KIRO's value. The TCP (the contact-tip end face "
                    "centre) is recessed inside the nozzle: TCP -> wire end = the torch's nozzle end (weld_torch.nozzle_end_mm, derived "
                    "from the robot's URDF collision geometry: our torch's CAD 1.2 mm) + stick_out_mm. KIRO's executor does not apply a "
                    "stick-out (user 2026-10-12 night): our TCP poses are the bead aim points moved back along -z_TCP by that sum",
    "assumed": "bead_toe_angle_deg (the convex bead's one shape parameter, bead_geometry), cap_overlap_mm (the cap band width "
               "that sets the number of cap passes), cap_weave_width_max_mm, side_work_angle_deg: not in the KIRO data (no bead "
               "width / height / toe angles, no CTWD)",
}


def _angle_entry(v, key: str):
    """역할 하나의 각: 수 → float, 목록 → 층 순서별 tuple (마지막 값이 이어진다; 한 개면 float)."""
    if isinstance(v, (list, tuple)):
        t = tuple(float(x) for x in v)
        if not t:
            raise ValueError(f"weld_plan.{key}: an empty per-layer list")
        return t[0] if len(t) == 1 else t
    return float(v)


def _listify(v):
    if isinstance(v, (list, tuple)):
        return [_listify(x) for x in v]
    if isinstance(v, dict):
        return {k: _listify(x) for k, x in v.items()}
    return v


@dataclass
class WeldProcess:
    """용접 절차 (WPS, 사전 정보) + 계획 설정. 역할별 값은 (root, fill, cap) 순서. 기본값 = KIRO 데이터셋의 WPS (WPS_PROVENANCE) —
    측정한 10T 그루브 (k20_2_a 기준 단면 85.5 mm²) 에서 root + 충전 2 + 위빙 캡 1, 비드 30.1 / 27.6 / 27.6 / 30.4 mm² (η 0.95).
    The lateral aim is KIRO's practice from the measured section (AIM_RULE; provenance["aim_rule"]) - no per-pass offsets here."""
    wfs_mpm: tuple = (8.4, 10.8, 10.2)               # 와이어 송급 속도 [m/min]
    speed_mps: tuple = (0.005, 0.007, 0.006)         # 공칭 진행 속도 (비드 모델이 이 속도에서 공칭 비드 단면적을 낸다)
    travel_angle_deg: tuple = (-42.0, (-41.0, -44.0), -44.0)   # + = 미는 각 (와이어가 진행 방향 앞을 향함), − = 끄는 각 (KIRO 9 월: 끄는 각)
    work_angle_deg: tuple = (9.0, (-20.0, 7.0), 7.0)    # 층에 비드 하나일 때: + = 와이어 / 토치 축이 베벨면 쪽으로 (역할의 층 순서별 목록
                                                         # 가능; 베벨은 KIRO 진행의 왼쪽 — 사용자 2026-10-13 밤, provenance["angles"])
    side_work_angle_deg: float = 10.0               # 층에 비드 여럿일 때: 벽에 붙는 비드가 그 벽 쪽으로
    wire_diameter_mm: float = 1.2
    stick_out_mm: float = 20.0                       # 와이어 돌출: 가스 노즐 끝 → 와이어 끝 (겨냥점), 토치 축을 따라 [mm] (용접 관행;
                                                     # 사용자 2026-10-13) — 우리 TCP 자세 = 겨냥점 − (토치의 노즐 끝 + 이 값) · z_TCP (TCP =
                                                     # 컨택트 팁 끝, 노즐 안 1.2 mm; KIRO 실행기는 돌출을 적용하지 않는다). 잠정 (KIRO 15–20; provenance)
    deposition_efficiency: float = 0.95              # 기본 비드 모델 (질량 보존) 의 η
    bead_toe_angle_deg: float = 30.0                 # 비드 토우 (플랭크) 각 [deg]: 토우 현과 원호 접선 사이 각 — 볼록 비드 기하의 모양 변수 하나
                                                     # (자유 비드 높이/폭 κ = tan(θ/2)/2; provenance["bead_toe_angle_deg"]). 옛 키
                                                     # bead_aspect_ratio κ 는 θ = 2 atan(2κ) 로 읽는다 (from_dict; 둘 다 주면 거부)
    cap_weave_width_max_mm: float = 16.0             # 위빙 캡 패스 하나가 덮는 최대 폭 (넘으면 캡 패스를 나란히 더)
    equipment: dict = field(default_factory=dict)    # 장비 설정값 (전압·전류·CTWD …; 값 또는 [root, fill, cap]) → 비드 모델 입력
    cap_overlap_mm: float = 1.5         # 캡 띠 폭 (상부 모서리 바깥으로 이만큼씩) — 캡 패스 수와 여러 캡 패스의 띠를 정한다 (캡 비드 자체의 폭은 비드 기하)
    cap_reinforcement_mm: Optional[float] = None   # 옛 작업의 키 (보강 목표): 이제 용접 설계 입력 (WeldDesign, job weld_design:) —
                                                   # 값이 있고 설계 블록이 없으면 그 값이 설계 공칭 (옛 작업·3G 템플릿)
    weave_edge_margin_mm: float = 1.0   # 캡 띠 모형의 위빙 끝 여유 (weave_amplitude_mm null 일 때: 진폭 = 띠 반폭 − 이 값)
    weave_pitch_mm: float = 7.8         # 위빙 한 주기 동안의 진행 거리 (KIRO 설정값; 실행기는 주기 수 = ceil(길이 / 피치))
    weave_dwell_s: float = 0.2          # 위빙 양 끝 머묾 (KIRO left / right 같은 값)
    speed_limits: tuple = (1.0, 1.0)    # 공칭 속도 대비 하한·상한 (= 비드 단면적 창; 넘으면 잘라 내고 표시) — KIRO 설정 그대로 = [1, 1]
    direction: str = "S->E"             # S->E | E->S | alternate (패스 1, 3, … = S->E) | uphill (3G: 아래 끝 → 위 끝)
    start_inset_mm: float = 0.0         # 옛 작업의 키 (경로 시작·끝 inset): 이제 용접 설계 입력 weld_design.end_inset_mm {start, goal}
    end_inset_mm: float = 0.0           # (WeldDesign) — 설계 블록이 inset 을 주지 않으면 이 값이 그 inset (둘 다 주면 ValueError)
    station_step_mm: float = 5.0
    station_window_mm: float = 10.0
    min_profiles: int = 5               # 단면 하나에 필요한 유효 프로파일 수 (모자라면 이웃 단면에서 보간)
    dense_step_mm: float = 1.0          # 충전 경로 점 간격
    weave_sample_s: float = 0.01        # 캡 위빙 경로 시간 간격
    source: str = WPS_SOURCE
    # ---- KIRO 실행기 값 (용접기 설정·실행 조건; KIRO 로그 [commanded] / [execution_conditions] 의 키로 내보낸다 — kiro_plan) ----
    current_a: tuple = (190.0, 230.0, 220.0)         # 설정 전류 [A] (역할별)
    voltage_v: tuple = (24.5, 26.5, 26.0)            # 설정 전압 [V]
    wfs_from_current: tuple = (0.06, -3.0)           # 설정선 WFS [m/min] = 기울기 · I + 절편
    voltage_from_current: tuple = (0.05, 15.0)       # 설정선 U [V] = 기울기 · I + 절편
    wire_material: str = "FE-SOLID"
    welding_mode: str = "LSM"
    shielding_gas: str = "CO2 100%"
    lead_in_mm: tuple = (1.0, 1.0, 0.0)              # 실행기 리드인 (시작 앞)
    lead_out_mm: tuple = (6.0, 6.0, 0.0)             # 실행기 리드아웃 (끝 뒤)
    arc_off_delay_ms: tuple = (500.0, 500.0, 0.0)    # 아크 끄기 지연 (실행기는 끝 앞 속도 × 지연에서 끈다)
    weave_pattern: str = "sine"                      # 캡 위빙 (실행기가 만든다): 모양·축·진폭·주기 수·주기당 표본
    weave_axis: str = "tool_y"
    weave_amplitude_mm: Optional[float] = 4.0        # 위빙 축 (토치 y) 방향 진폭 — 우리 캡 모형도 이 값 (null = 캡 띠 모형)
    weave_cycles: int = 16                           # KIRO 설정 (그 패스 길이에서; 정보)
    weave_samples_per_cycle: int = 12
    provenance: dict = field(default_factory=lambda: dict(WPS_PROVENANCE))

    def __post_init__(self):
        for k in _ROLE_VALUES:
            v = tuple(float(x) for x in getattr(self, k))
            if len(v) != len(ROLES):
                raise ValueError(f"weld_plan.{k} needs {len(ROLES)} values (root, fill, cap), got {v}")
            setattr(self, k, v)
        for k in _ROLE_ANGLES:
            v = tuple(_angle_entry(x, k) for x in getattr(self, k))
            if len(v) != len(ROLES):
                raise ValueError(f"weld_plan.{k} needs {len(ROLES)} values (root, fill, cap), got {v}")
            setattr(self, k, v)
        for k in ("wfs_from_current", "voltage_from_current"):
            v = tuple(float(x) for x in getattr(self, k))
            if len(v) != 2 or not all(math.isfinite(x) for x in v) or v[0] <= 0.0:
                raise ValueError(f"weld_plan.{k} must be [slope > 0, intercept], got {v}")
            setattr(self, k, v)
        self.speed_limits = tuple(float(v) for v in self.speed_limits)
        self.equipment = dict(self.equipment or {})
        self.provenance = dict(self.provenance or {})
        if min(self.wfs_mpm) <= 0.0 or min(self.speed_mps) <= 0.0:
            raise ValueError("weld_plan: wfs_mpm and speed_mps must be > 0 for every role")
        if min(self.current_a) <= 0.0 or min(self.voltage_v) <= 0.0:
            raise ValueError("weld_plan: current_a and voltage_v must be > 0 for every role")
        if min(self.lead_in_mm + self.lead_out_mm + self.arc_off_delay_ms) < 0.0:
            raise ValueError("weld_plan: lead_in_mm, lead_out_mm and arc_off_delay_ms must be >= 0")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"weld_plan.direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if not (0.0 < self.speed_limits[0] <= 1.0 <= self.speed_limits[1]):
            raise ValueError("weld_plan.speed_limits must be (lo <= 1 <= hi)")
        if min(self.station_step_mm, self.station_window_mm, self.dense_step_mm, self.weave_pitch_mm, self.weave_sample_s,
               self.cap_weave_width_max_mm) <= 0.0:
            raise ValueError("weld_plan: steps, window, pitch, sample time and cap weave width must be > 0")
        if not (math.isfinite(float(self.stick_out_mm)) and float(self.stick_out_mm) >= 0.0):
            raise ValueError(f"weld_plan.stick_out_mm must be >= 0, got {self.stick_out_mm!r}")
        self.stick_out_mm = float(self.stick_out_mm)
        if "stick_out_mm" not in self.provenance:       # an older process block (before stick_out_mm): the value is the WPS default's
            if self.stick_out_mm == float(WeldProcess.__dataclass_fields__["stick_out_mm"].default):
                self.provenance["stick_out_mm"] = WPS_PROVENANCE["stick_out_mm"]
            else:
                self.provenance["stick_out_mm"] = "the job's value (no provenance given)"
        if self.weave_amplitude_mm is not None and not float(self.weave_amplitude_mm) >= 0.0:
            raise ValueError("weld_plan: weave_amplitude_mm must be >= 0 (or null = the cap band model)")
        if self.cap_reinforcement_mm is not None and not float(self.cap_reinforcement_mm) >= 0.0:
            raise ValueError("weld_plan: cap_reinforcement_mm must be >= 0 (or null = the result of the cap's process)")
        if not (int(self.weave_samples_per_cycle) >= 1 and int(self.weave_cycles) >= 0):
            raise ValueError("weld_plan: weave_samples_per_cycle must be >= 1 and weave_cycles >= 0")
        self.weave_samples_per_cycle = int(self.weave_samples_per_cycle)
        self.weave_cycles = int(self.weave_cycles)
        self.bead_toe_angle_deg = float(self.bead_toe_angle_deg)
        if self.provenance.get("bead_toe_angle_deg") in (None, WPS_PROVENANCE["bead_toe_angle_deg"]):   # no own provenance given:
            self.provenance["bead_toe_angle_deg"] = (WPS_PROVENANCE["bead_toe_angle_deg"] if self.bead_toe_angle_deg ==   # say whether
                                                     float(WeldProcess.__dataclass_fields__["bead_toe_angle_deg"].default)  # it is the
                                                     else "the job's value (no provenance given)")                           # default
        MassBalanceBeadModel(self.wire_diameter_mm, self.deposition_efficiency, self.bead_toe_angle_deg)    # 값 검사

    def role_angle(self, key: str, role: str, occurrence: int = 0) -> float:
        """역할 role 의 occurrence 번째 층 (0 부터, 그 역할의 층 순서) 의 각 (travel_angle_deg | work_angle_deg)."""
        e = getattr(self, key)[ROLES.index(role)]
        if isinstance(e, tuple):
            return float(e[min(max(int(occurrence), 0), len(e) - 1)])
        return float(e)

    def role_value(self, key: str, role: str) -> float:
        """역할별 값 하나 (_ROLE_VALUES: wfs_mpm, speed_mps, current_a, voltage_v, lead_in_mm, lead_out_mm, arc_off_delay_ms)."""
        if key not in _ROLE_VALUES:
            raise KeyError(key)
        return float(getattr(self, key)[ROLES.index(role)])

    def wfs_on_line(self, current_a: float) -> float:
        a, b = self.wfs_from_current
        return a * float(current_a) + b

    def current_on_line(self, wfs_mpm: float) -> float:
        a, b = self.wfs_from_current
        return (float(wfs_mpm) - b) / a

    def voltage_on_line(self, current_a: float) -> float:
        a, b = self.voltage_from_current
        return a * float(current_a) + b

    def params(self, role: str, position: str = "1G", weave_amp_mm: float = 0.0, occurrence: int = 0) -> PassParams:
        i = ROLES.index(role)
        st = tuple(sorted((str(k), (v[i] if isinstance(v, (list, tuple)) and len(v) == len(ROLES) else v))
                          for k, v in self.equipment.items()))
        return PassParams(role=role, wfs_mpm=self.wfs_mpm[i], speed_mps=self.speed_mps[i],
                          travel_angle_deg=self.role_angle("travel_angle_deg", role, occurrence),
                          work_angle_deg=self.role_angle("work_angle_deg", role, occurrence), weave_amp_mm=float(weave_amp_mm),
                          position=str(position or "1G"), settings=st)

    def bead_model(self) -> MassBalanceBeadModel:
        return MassBalanceBeadModel(self.wire_diameter_mm, self.deposition_efficiency, self.bead_toe_angle_deg)

    @property
    def bead_aspect_ratio(self) -> float:
        """The free bead's height / width (kappa) of the toe angle - derived, read-only (the shape parameter is bead_toe_angle_deg)."""
        return crown_ratio_for_toe_angle(self.bead_toe_angle_deg)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "WeldProcess":
        d = dict(d or {})
        if "bead_aspect_ratio" in d:                    # the shape parameter before 2026-10-13 night: kappa -> the toe angle
            if "bead_toe_angle_deg" in d:
                raise ValueError("weld_plan: bead_aspect_ratio and bead_toe_angle_deg both given - the bead's one shape parameter is "
                                 "bead_toe_angle_deg (bead_aspect_ratio kappa = tan(toe / 2) / 2 is read only when it is alone)")
            k = float(d.pop("bead_aspect_ratio"))
            if not (math.isfinite(k) and k > 0.0):
                raise ValueError(f"weld_plan.bead_aspect_ratio must be > 0, got {k!r}")
            d["bead_toe_angle_deg"] = toe_angle_for_crown_ratio(k)
            prov = dict(d.get("provenance") or {})
            prov["bead_toe_angle_deg"] = (f"the job's bead_aspect_ratio {k:g} (the older key) read as the toe angle 2 atan(2 kappa) = "
                                          f"{d['bead_toe_angle_deg']:.2f} deg")
            d["provenance"] = prov
        names = {f.name for f in dataclasses.fields(cls)}
        gone = sorted(set(d) & set(_REMOVED_KEYS))
        if gone:
            raise ValueError("weld_plan: " + "; ".join(f"{k_}: {_REMOVED_KEYS[k_]}" for k_ in gone))
        unknown = sorted(set(d) - names)
        if unknown:
            hint = (" (the layer plan is derived from the WPS now: wfs_mpm / speed_mps / travel_angle_deg / work_angle_deg per role "
                    "root, fill, cap)") if set(unknown) & _LEGACY_KEYS else ""
            raise ValueError(f"weld_plan: unknown keys {unknown}{hint}")
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})

    def to_dict(self) -> dict:
        return {k: _listify(v) for k, v in dataclasses.asdict(self).items()}


def aim_tip_offset_mm(process: WeldProcess, torch: Optional[WeldTorch] = None) -> tuple:
    """L [mm] = TCP -> wire end along the torch axis (where KIRO's 09-22 offsets are evaluated for the verification) and its source:
    the plan torch's nozzle recess (derived from the tool's collision geometry, weld_torch.nozzle_end_mm) + the job's stick-out
    (gas nozzle end -> wire end); without a torch model (or a torch whose nozzle end is not known) the stick-out alone."""
    s = float(process.stick_out_mm)
    noz = getattr(torch, "nozzle_end_mm", None) if torch is not None else None
    if noz is None:
        why = "no weld torch model" if torch is None else f"the weld torch {torch.tcp_frame} has no known nozzle end"
        return s, (f"{why}: the stick-out alone (job weld_plan.stick_out_mm {s:g} mm from the gas nozzle end; the nozzle recess "
                   "is not known without the torch)")
    return float(noz) + s, (f"the torch's nozzle recess {float(noz):.2f} mm ({torch.nozzle_end_source or 'weld_torch'}) + the "
                            f"stick-out {s:g} mm from the gas nozzle end (job weld_plan.stick_out_mm)")


# 용접 설계의 기본 캡 보강 (KIRO 데이터셋에서 유도 — 1G 작업 템플릿의 weld_design: 블록이 같은 값, test_weld_plan 이 대조)
DESIGN_PROVENANCE = {
    "cap_reinforcement_mm": "nominal 1.87 mm: what KIRO's cap (30.4 mm2) gives on the measured 10T groove (k20_2_a reference section, "
                            "85.5 mm2, hi-lo -0.69 mm) after KIRO's root + 2 fills, with the convex bead geometry (bead_geometry, "
                            "2026-10-13 night: the beads laid as the plan lays them, bead toe angle 30 deg; reinforcement = the highest "
                            "point of the cap above the line between the groove top edges). Tolerance 0.19 mm: the same derivation over "
                            "the deposition efficiency band 0.90..1.00 of the KIRO analysis gives 1.68..2.05 mm (fsv1_work/bead/"
                            "derive_design.py). Was 3.2 +- 0.65 mm with the flat layers and the parabolic cap band (bead_fill.CapSurface)"
                            ", 2.42 +- 0.25 mm with the arc of kappa 0.3. Source: " + WPS_SOURCE,
}


@dataclass
class WeldDesign:
    """용접 설계 입력 (job weld_design: 블록 — 이음 설계 specimen 과 나란히): 캡 보강 (판 상면 위 높이) 의 공칭·공차. 층 수 고르기: 보강이
    공칭 ± 공차 안인 배치 가운데 가장 싼 것 (속도 창 안), 없으면 가장 가까운 것 + 표시 (판정 아님). nominal null = 목표 없음 (캡은 공칭 용착),
    tol null = 공칭에 가장 가까운 것 (옛 동작).

    끝 inset (사용자 2026-10-12, end_inset_mm {start, goal}): 모든 층의 용접선 시작·끝 (그 패스의 진행 방향으로 시작 = start, 끝 = goal)
    을 잰 끝에서 심을 따라 이만큼 안쪽으로 — 용접 계획의 모든 출력 (weld_plan.json/npz, kiro/pass_<k>.yaml·info, 액션 결과 paths) 에.
    기본 0 = 시작·끝이 정확히 잰 끝. 센싱과는 무관하다 (추정·끝은 그대로). None = 주지 않음 → 옛 공정 키 (weld_plan.start_inset_mm /
    end_inset_mm), 없으면 0 (resolve)."""
    cap_reinforcement_mm: Optional[float] = 1.87
    cap_reinforcement_tol_mm: Optional[float] = 0.19
    source: str = "KIRO-derived default (" + WPS_SOURCE + ")"
    provenance: dict = field(default_factory=lambda: dict(DESIGN_PROVENANCE))
    end_inset_start_mm: Optional[float] = None
    end_inset_goal_mm: Optional[float] = None
    end_inset_source: str = ""

    def __post_init__(self):
        for k in ("cap_reinforcement_mm", "cap_reinforcement_tol_mm", "end_inset_start_mm", "end_inset_goal_mm"):
            v = getattr(self, k)
            if v is not None:
                v = float(v)
                if not (math.isfinite(v) and v >= 0.0):
                    raise ValueError(f"weld_design.{k} must be >= 0 (or null), got {v!r}")
                setattr(self, k, v)
        self.provenance = dict(self.provenance or {})

    @property
    def end_insets_mm(self) -> tuple:
        """(start, goal) [mm] — 주지 않은 값은 0 (resolve 뒤에는 늘 값이 있다)."""
        return (float(self.end_inset_start_mm or 0.0), float(self.end_inset_goal_mm or 0.0))

    @classmethod
    def from_doc(cls, block: Optional[dict]) -> "WeldDesign":
        """job weld_design: 블록 {cap_reinforcement_mm: {nominal, tol}, end_inset_mm: {start, goal}, source} → WeldDesign (빠진 값 = 기본)."""
        b = dict(block or {})
        unknown = sorted(set(b) - {"cap_reinforcement_mm", "end_inset_mm", "source", "provenance"})
        if unknown:
            raise ValueError(f"weld_design: unknown keys {unknown}")
        kw = {}
        ei = b.get("end_inset_mm")
        if ei is not None:
            if not isinstance(ei, dict):
                raise ValueError(f"weld_design.end_inset_mm must be {{start: <mm>, goal: <mm>}}, got {ei!r}")
            bad = sorted(set(ei) - {"start", "goal"})
            if bad:
                raise ValueError(f"weld_design.end_inset_mm: unknown keys {bad} (start, goal)")
            kw["end_inset_start_mm"] = float(ei.get("start") or 0.0)
            kw["end_inset_goal_mm"] = float(ei.get("goal") or 0.0)
            kw["end_inset_source"] = "job weld_design.end_inset_mm"
        cr = b.get("cap_reinforcement_mm")
        if isinstance(cr, dict):
            bad = sorted(set(cr) - {"nominal", "tol"})
            if bad:
                raise ValueError(f"weld_design.cap_reinforcement_mm: unknown keys {bad} (nominal, tol)")
            if "nominal" in cr:
                kw["cap_reinforcement_mm"] = cr["nominal"]
            if "tol" in cr:
                kw["cap_reinforcement_tol_mm"] = cr["tol"]
        elif cr is not None:
            kw["cap_reinforcement_mm"] = cr
        if "source" in b:
            kw["source"] = str(b["source"])
        if "provenance" in b:
            kw["provenance"] = dict(b["provenance"] or {})
        return cls(**kw)

    @classmethod
    def resolve(cls, design: Optional["WeldDesign"], process: Optional["WeldProcess"] = None) -> "WeldDesign":
        """쓸 설계: 준 것 > 공정의 옛 키 cap_reinforcement_mm (옛 작업·3G 템플릿; 공차 없음) > 기본 (KIRO 유도). 끝 inset: 설계가 주면
        그것, 아니면 공정의 옛 키 start_inset_mm / end_inset_mm (없으면 0) — 둘 다 주면 ValueError (한 곳에서만)."""
        if design is None:
            if process is not None and process.cap_reinforcement_mm is not None:
                design = cls(cap_reinforcement_mm=float(process.cap_reinforcement_mm), cap_reinforcement_tol_mm=None,
                             source="job weld_plan.cap_reinforcement_mm (old process key)", provenance={})
            else:
                design = cls()
        old = (float(process.start_inset_mm), float(process.end_inset_mm)) if process is not None else (0.0, 0.0)
        if design.end_inset_start_mm is None and design.end_inset_goal_mm is None:
            src = ("job weld_plan.start_inset_mm / end_inset_mm (old process keys)" if any(old) else
                   "default: the start and goal at the measured ends")
            return dataclasses.replace(design, end_inset_start_mm=old[0], end_inset_goal_mm=old[1], end_inset_source=src)
        if any(old) and design.end_insets_mm != old:
            raise ValueError("the end inset is given twice: weld_design.end_inset_mm and the old weld_plan.start_inset_mm / end_inset_mm "
                             "- keep weld_design.end_inset_mm")
        return dataclasses.replace(design, end_inset_start_mm=float(design.end_inset_start_mm or 0.0),
                                   end_inset_goal_mm=float(design.end_inset_goal_mm or 0.0))

    def to_dict(self) -> dict:
        s, g = self.end_insets_mm
        return {"cap_reinforcement_mm": {"nominal": self.cap_reinforcement_mm, "tol": self.cap_reinforcement_tol_mm},
                "end_inset_mm": {"start": s, "goal": g, "source": self.end_inset_source,
                                 "meaning": "every pass's weld line starts / ends this far inside the measured seam ends along the "
                                            "seam (start = where the pass begins in its welding direction); sensing and the "
                                            "measured ends are unchanged"},
                "source": self.source, "provenance": dict(self.provenance)}


def weld_design_from_doc(doc: Optional[dict], process: Optional["WeldProcess"] = None) -> WeldDesign:
    """job 문서 → 쓸 설계 (weld_design: 블록이 있으면 그것, 없으면 공정의 옛 키·기본) — WeldDesign.resolve 를 거친 값 (끝 inset 확정;
    두 곳에서 주면 ValueError)."""
    block = (doc or {}).get("weld_design")
    return WeldDesign.resolve(WeldDesign.from_doc(block) if block else None, process)


def end_insets_mm(plan) -> tuple:
    """계획의 끝 inset (start, goal) [mm]: 계획의 설계 (plan_weld 가 확정) — 설계가 없는 계획이면 공정의 옛 키."""
    d = getattr(plan, "design", None)
    if d is not None:
        return d.end_insets_mm
    pr = plan.process
    return (float(pr.start_inset_mm), float(pr.end_inset_mm))


@dataclass
class GrooveStation:
    """심 방향 s [mm] (S 끝 = 0) 의 실측 단면. origin = 루트갭 중심·뒷댐재 상면 (base, m)."""
    s_mm: float
    origin: np.ndarray
    u: np.ndarray
    n: np.ndarray
    w: np.ndarray
    spec: GrooveSpec
    n_profiles: int = 0
    interpolated: bool = False
    measured: dict = field(default_factory=dict)

    def to_base(self, x_mm, z_mm) -> np.ndarray:
        x = np.asarray(x_mm, dtype=float)[..., None]
        z = np.asarray(z_mm, dtype=float)[..., None]
        return self.origin + (x * self.w + z * self.n) * 1e-3


@dataclass
class Layer:
    """기준 단면의 충전 층 하나 (convex beads, bead_geometry). frac_top = 이 층까지 용착한 단면적 / 그루브 단면적 (단면마다 이 비율로 층의
    용착을 정한다); z_top_ref_mm = the layer's toe level (where its free surface meets the walls - the old flat 'layer top' at the
    walls), z_crown_ref_mm = its crown (the highest point of its free surface), z_pool_ref_mm = the level of its pool without crown
    (the bead-count criterion, width_top_ref_mm: one bead's toe chord <= bead_width_mm, or for n_beads > 1 that pool's width <=
    n_beads x bead_width_mm)."""
    role: str
    n_beads: int
    frac_top: float
    z_top_ref_mm: float
    bead_area_ref_mm2: float
    bead_area_nom_mm2: float
    width_top_ref_mm: float
    bead_width_mm: float
    z_crown_ref_mm: float = float("nan")
    z_pool_ref_mm: float = float("nan")
    crown_ratio: float = float("nan")


@dataclass
class Topology:
    """기준 단면에서 정한 층·비드 배치 + 캡 (plan_topology)."""
    layers: list
    n_cap: int
    cap_width_ref_mm: float
    cap_area_ref_mm2: float             # 캡 패스 하나가 용착하는 단면적 (기준 단면; 속도 창으로 자른 속도에서)
    cap_area_nom_mm2: float
    groove_area_ref_mm2: float
    reinforcement_area_ref_mm2: float   # 판 상면 위 보강 단면적 (기준 단면; 목표가 있으면 목표, 없으면 캡의 결과)
    cost: float                         # Σ ln²(a/a_nom)
    feasible: bool                      # 모든 패스가 속도 창 안이고 캡이 남은 그루브를 덮는다 (보강 ≥ 0)
    deposit_scale: float = 1.0          # λ: 충전 비드 단면적 / 공칭
    cap_scale: float = 1.0              # μ: 캡 비드 단면적 / 공칭
    reference: GrooveSpec = field(default_factory=GrooveSpec)
    notes: list = field(default_factory=list)
    reinforcement_ref_mm: float = float("nan")   # 그 보강의 높이: 캡 뒤 표면이 그루브 상부 모서리를 잇는 선 위로 올라온 최대 높이 (비드 기하)
    target_met: bool = True                      # 보강이 설계 공칭 ± 공차 안 (공차 없으면: 속도 창 안에서 목표를 맞춘다)
    design: dict = field(default_factory=dict)   # 쓴 용접 설계 (WeldDesign.to_dict)
    beads_ref: list = field(default_factory=list)  # the reference section's beads (BeadSection, pass order) - figures, reports

    @property
    def n_passes(self) -> int:
        return sum(ly.n_beads for ly in self.layers) + self.n_cap

    def to_dict(self) -> dict:
        return {"layers": [dataclasses.asdict(ly) for ly in self.layers], "n_cap": self.n_cap, "cap_width_ref_mm": self.cap_width_ref_mm,
                "cap_area_ref_mm2": self.cap_area_ref_mm2, "cap_area_nom_mm2": self.cap_area_nom_mm2,
                "groove_area_ref_mm2": self.groove_area_ref_mm2, "reinforcement_area_ref_mm2": self.reinforcement_area_ref_mm2,
                "cost": self.cost, "feasible": self.feasible, "deposit_scale": self.deposit_scale, "cap_scale": self.cap_scale,
                "n_passes": self.n_passes, "reference_section": self.reference.to_dict(),
                "reinforcement_ref_mm": self.reinforcement_ref_mm, "target_met": self.target_met, "weld_design": dict(self.design),
                "beads_ref": [b.to_dict() for b in self.beads_ref], "notes": list(self.notes)}


@dataclass
class WeldPass:
    """패스 하나: 단면 (station) 마다의 계획 + 촘촘한 TCP 경로 (dense_*). layer = 층 번호 (1 부터; 캡 = 충전 층 수 + 1),
    bead = 층 안 순서 (1 부터, 단면 −x 벽 → +x 벽)."""
    k: int
    kind: str                       # fill | cap (root 패스도 kind 'fill', role 'root')
    direction: str                  # S->E | E->S
    s_mm: np.ndarray                # 진행 순서의 단면 위치
    aim_xz_mm: np.ndarray           # (N, 2) 단면 좌표 겨냥점
    aim_base_m: np.ndarray          # (N, 3)
    R_base_tcp: np.ndarray          # (N, 3, 3)
    speed_mps: np.ndarray
    area_mm2: np.ndarray            # 이 패스의 용착 단면적
    layer_from_mm: np.ndarray       # the bead's bottom: the lowest point of the surface under it (the root: the backing-bar top)
    layer_to_mm: np.ndarray         # the bead's crown: the highest point of its convex free surface
    speed_clamped: np.ndarray
    weave_amp_mm: Optional[np.ndarray] = None
    weave_width_mm: Optional[np.ndarray] = None
    role: str = "fill"
    layer: int = 1
    bead: int = 1
    n_beads: int = 1
    dense_t_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dense_s_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dense_pos_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    dense_R: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    dense_lateral_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # 일정 속도 실행기용 (KIRO: 패스마다 TCP 속도 하나): 단면적의 심 방향 평균 (s 사다리꼴) → 그 역할의 WFS 로 질량 보존 속도 (속도 창으로
    # 자르고 표시). speed_required_mps = 단면마다의 필요 속도 (자르기 전) — 심 방향 변화 보고
    area_mean_mm2: float = float("nan")
    speed_const_mps: float = float("nan")
    speed_const_clamped: bool = False
    speed_required_mps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    travel_angle_deg: float = float("nan")      # 이 패스에 쓴 진행각·작업각 (WPS 의 역할·층 순서 값; 옆 비드는 side_work_angle_deg)
    work_angle_deg: float = float("nan")
    area_required_mm2: np.ndarray = field(default_factory=lambda: np.zeros(0))   # 계획 층 (비율 F_j) 을 채우는 데 필요한 단면적
    reinforcement_mm: Optional[np.ndarray] = None   # 캡: 판 상면 위 보강 높이 (캡이 실제 용착한 단면적의 결과)
    # WPS 의 각 (목표) 과 고른 각 (사용자 2026-10-13: 패스마다 충돌 없고 닿는 가장 가까운 토치 자세 — scan_core.weld_orient). 고르기 전에는
    # travel/work_angle_deg = 공칭; orientation = 고른 기록 (공칭·고른 각·편차·여유·롤·이유)
    travel_nominal_deg: float = float("nan")
    work_nominal_deg: float = float("nan")
    orientation: dict = field(default_factory=dict)
    # lateral aim rule of this pass (AIM_RULE): AIM_KIRO (KIRO's practice) or AIM_GEOMETRIC (the band centre), and why
    aim_lateral_rule: str = ""
    aim_why: str = ""
    # the pass's convex bead in each section (bead_geometry.BeadSection, in the pass's station order - like s_mm): toes, the arc,
    # its convex solid (the weld scene of the later passes)
    beads: list = field(default_factory=list)

    def bead_xz(self, what: str) -> np.ndarray:
        """(N, 2) per station: 'toe_l' / 'toe_r' / 'crown' (x, z) [mm] of the pass's bead."""
        if what == "crown":
            return np.array([b.apex for b in self.beads], dtype=float).reshape(-1, 2)
        return np.array([getattr(b, what) for b in self.beads], dtype=float).reshape(-1, 2)

    @property
    def duration_s(self) -> float:
        return float(self.dense_t_s[-1]) if self.dense_t_s.size else float("nan")

    def summary(self) -> dict:
        def rng(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            return [float(np.median(a)), float(a.min()), float(a.max())] if a.size else [float("nan")] * 3
        if self.dense_t_s.size > 1:
            v = np.linalg.norm(np.diff(self.dense_pos_m, axis=0), axis=1) / np.maximum(np.diff(self.dense_t_s), 1e-9)
            vmax = float(np.max(v)) * 1e3
        else:
            vmax = float("nan")
        d = {"k": self.k, "kind": self.kind, "role": self.role, "layer": self.layer, "bead": self.bead, "n_beads": self.n_beads,
             "direction": self.direction, "n_stations": int(self.s_mm.size), "max_tcp_speed_mm_s": vmax,
             "layer_to_mm": rng(self.layer_to_mm), "area_mm2": rng(self.area_mm2), "speed_mm_s": rng(self.speed_mps * 1e3),
             "aim_x_mm": rng(self.aim_xz_mm[:, 0]), "aim_z_mm": rng(self.aim_xz_mm[:, 1]), "duration_s": self.duration_s,
             "n_speed_clamped": int(np.sum(self.speed_clamped)), "n_dense": int(self.dense_t_s.size),
             "area_mean_mm2": float(self.area_mean_mm2), "speed_const_mm_s": float(self.speed_const_mps) * 1e3,
             "speed_const_clamped": bool(self.speed_const_clamped), "speed_required_mm_s": rng(self.speed_required_mps * 1e3),
             "travel_angle_deg": float(self.travel_angle_deg), "work_angle_deg": float(self.work_angle_deg),
             "travel_nominal_deg": float(self.travel_nominal_deg), "work_nominal_deg": float(self.work_nominal_deg),
             "area_required_mm2": rng(self.area_required_mm2), "aim_lateral_rule": self.aim_lateral_rule, "aim_why": self.aim_why,
             "layer_from_mm": rng(self.layer_from_mm)}
        if self.beads:
            tl, tr, cr = self.bead_xz("toe_l"), self.bead_xz("toe_r"), self.bead_xz("crown")
            d["bead_geometry"] = {"toe_left_x_mm": rng(tl[:, 0]), "toe_left_z_mm": rng(tl[:, 1]), "toe_right_x_mm": rng(tr[:, 0]),
                         "toe_right_z_mm": rng(tr[:, 1]), "crown_x_mm": rng(cr[:, 0]), "crown_z_mm": rng(cr[:, 1]),
                         "chord_mm": rng([b.chord_mm for b in self.beads]), "crown_ratio": rng([b.kappa for b in self.beads]),
                         "phase": sorted({b.phase for b in self.beads})}
        if self.orientation:
            d["orientation"] = dict(self.orientation)
        if self.reinforcement_mm is not None:
            d["reinforcement_mm"] = rng(self.reinforcement_mm)
        if self.weave_amp_mm is not None:
            d["weave_amp_mm"] = rng(self.weave_amp_mm)
            d["weave_width_mm"] = rng(self.weave_width_mm)
        return d


@dataclass
class WeldPlan:
    stations: list
    passes: list
    process: WeldProcess
    spec_design: GrooveSpec
    base_frame: str = "link0"
    position: str = "1G"
    length_mm: float = float("nan")
    notes: list = field(default_factory=list)
    bevel_side: str = "unknown"
    feasibility: list = field(default_factory=list)   # 로봇 도달성 (플랫폼이 확인: MoveItPlatform.check_weld_plan; 없으면 빈 목록)
    topology: Optional[Topology] = None
    levels_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (단면, 충전 층) the layer's toe level (walls)
    bead_model: dict = field(default_factory=dict)
    ends: dict = field(default_factory=dict)   # S→E 계약: 관측 끝 S/E (base), 축, σ, 계약 검사 값 (weld_plan_from_features)
    torch: Optional[WeldTorch] = None          # 용접 계획의 토치 모형 (플랫폼 설정 weld_torch — 우리 1차 브라켓 토치, 2026-10-12): 패스의 TCP 자세
    design: Optional[WeldDesign] = None        # 쓴 용접 설계 (plan_weld 가 확정: 캡 보강, 끝 inset)
    reachability: dict = field(default_factory=dict)   # D14 를 하지 않았으면 {checked: False, note} (용접 토치가 이 팔에 없다 —
                                                       # scan_core.weld_torch.d14_on_this_arm); 했으면 feasibility 가 결과
    torch_roll_deg: float = 0.0                # TCP 의 토치 축 둘레 롤 (0 / 180): plan_weld 가 NECK_RULE 로 정하고 (weld_orient.rule_roll), D14 가
                                               # 있으면 choose_orientation 이 그 롤 안에서 확인한다 (모든 패스 같은 롤)
    torch_roll: dict = field(default_factory=dict)     # 그 선택·이유 (rule_roll / weld_orient)
    clearance: dict = field(default_factory=dict)      # 필요 여유 (불확도 모형: k × 결합 σ, 출처·빠진 출처 — weld_orient.clearance_requirement)
    orientation: dict = field(default_factory=dict)    # 토치 자세 고르기 기록 (weld_orient.choose_orientation: 규칙·패스별 공칭·고른 각·편차·여유)
    aim: dict = field(default_factory=dict)            # the aim record (plan_weld): per pass the rule and why, the 09-22 check
    # the convex beads of every pass in every section (bead_geometry.BeadSection; pass k -> the stations' beads in station order S -> E):
    # the specimen before a pass (weld_scene) whatever subset of passes a check carries
    deposits: dict = field(default_factory=dict)
    line_step_mm: float = float("nan")                 # D14's sample step on the delivered lines (scan_core.weld_line; the platform's
                                                       # Cartesian step - weld_orient.choose_orientation records it; nan = no D14 search)

    # ---- 보고 ----
    def summary_lines(self) -> list:
        return format_plan_summary(self.to_dict())

    def to_dict(self) -> dict:
        def st_d(i: int, st: GrooveStation) -> dict:
            lev = [float(v) for v in self.levels_mm[i]] if self.levels_mm.size else []
            return {"s_mm": st.s_mm, "origin_base_m": st.origin.tolist(), "u": st.u.tolist(), "n": st.n.tolist(), "w": st.w.tolist(),
                    "spec": st.spec.to_dict(), "n_profiles": st.n_profiles, "interpolated": st.interpolated, "measured": st.measured,
                    "groove_area_mm2": float(groove_area_below_mm2(st.spec, st.spec.thickness_mm)), "levels_mm": lev}
        return _plain({"schema_version": PLAN_SCHEMA, "base_frame": self.base_frame, "position": self.position,
                       "length_mm": self.length_mm, "bevel_side": self.bevel_side, "bevel_plate_looking_S_to_E": self.bevel_view,
                       "ends": dict(self.ends),
                       "n_interpolated_sections": sum(1 for st in self.stations if st.interpolated),
                       "feasibility": list(self.feasibility), "reachability": dict(self.reachability),
                       "process": self.process.to_dict(), "bead_model": dict(self.bead_model),
                       "topology": self.topology.to_dict() if self.topology is not None else None,
                       "spec_design": self.spec_design.to_dict(), "notes": list(self.notes),
                       "tcp": "wire tip at the aim point; R = [x travel, y = z x x, z = torch axis toward the work]"
                              + (f"; the weld torch's TCP {self.torch.tcp_frame} (torch tip end centre) poses of the dense path: "
                                 f"weld_plan.npz p<k>_tcp_pos_m / p<k>_tcp_R = the aim point moved back along -z_TCP by the nozzle end "
                                 f"{self.torch.nozzle_end_mm:.2f} + the stick-out (from the nozzle end) {self.torch.stick_out_mm:g} = "
                                 f"{self.torch.tip_offset_mm:.2f} mm, TCP roll {self.torch_roll_deg:g} deg (weld_torch, torch_roll)"
                                 if self.torch is not None else ""),
                       "torch_roll": dict(self.torch_roll) or {"deg": float(self.torch_roll_deg)},
                       "clearance_requirement": dict(self.clearance), "torch_orientation": dict(self.orientation),
                       "line_step_mm": self.line_step_mm,
                       "weld_torch": self.torch.to_dict() if self.torch is not None else None,
                       "weld_design": self.design.to_dict() if self.design is not None else None,
                       "section_frame": "x along w = n x u (root-gap centre 0), z along n (backing-bar top 0), mm",
                       "aim_rule": {**AIM_RULE, **self.aim},
                       "bead_geometry": {"rule": BEAD_RULE, "crown_ratio": "bead_model crown_ratio (the toe angle "
                                         f"{self.process.bead_toe_angle_deg:g} deg -> kappa {self.process.bead_aspect_ratio:.4f} "
                                         "without weave)",
                                         "source": self.process.provenance.get("bead_geometry", "no provenance"),
                                         "levels_mm": "stations[].levels_mm: each fill layer's toe level (its free surface on the walls)"},
                       "passes": [dict(p.summary(), sections=dict(
                           s_mm=p.s_mm, aim_xz_mm=p.aim_xz_mm, aim_base_m=p.aim_base_m, speed_mps=p.speed_mps, area_mm2=p.area_mm2,
                           layer_from_mm=p.layer_from_mm, layer_to_mm=p.layer_to_mm, speed_clamped=p.speed_clamped,
                           **({"bead_toe_left_xz_mm": p.bead_xz("toe_l"), "bead_toe_right_xz_mm": p.bead_xz("toe_r"),
                               "bead_crown_xz_mm": p.bead_xz("crown"), "bead_crown_ratio": [b.kappa for b in p.beads],
                               "bead_phase": [b.phase for b in p.beads]} if p.beads else {}),
                           **({"weave_amp_mm": p.weave_amp_mm, "weave_width_mm": p.weave_width_mm} if p.weave_amp_mm is not None else {})))
                                  for p in self.passes],
                       "stations": [st_d(i, st) for i, st in enumerate(self.stations)]})

    @property
    def bevel_view(self) -> str:
        """베벨 판이 S→E 로 볼 때 어느 쪽인가 (단면 +x = w = n × u = 진행 방향 왼쪽)."""
        return {"left": "right", "right": "left"}.get(self.bevel_side, self.bevel_side if self.bevel_side in ("both", "none") else "both")

    def save(self, out_dir, stem: str = "weld_plan") -> list:
        """weld_plan.json (요약·층 배치·단면·패스별 단면 계획) + weld_plan.npz (패스별 촘촘한 TCP 경로: p<k>_t_s, _s_mm, _pos_m, _R,
        _lateral_mm; 용접 토치 모형이 있으면 그 TCP 자세 p<k>_tcp_pos_m, _tcp_R 도)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        jp, npz = out / f"{stem}.json", out / f"{stem}.npz"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, ensure_ascii=False)
        arr = {"base_frame": np.array(self.base_frame), "schema_version": np.array(PLAN_SCHEMA)}
        if self.torch is not None:
            arr["tcp_frame"] = np.array(self.torch.tcp_frame)
        for p in self.passes:
            arr.update({f"p{p.k}_t_s": p.dense_t_s, f"p{p.k}_s_mm": p.dense_s_mm, f"p{p.k}_pos_m": p.dense_pos_m,
                        f"p{p.k}_R": p.dense_R, f"p{p.k}_lateral_mm": p.dense_lateral_mm, f"p{p.k}_kind": np.array(p.kind)})
            if self.torch is not None and p.dense_pos_m.shape[0]:
                P_tcp, R_tcp = self.torch.tcp_poses(p.dense_pos_m, p.dense_R, roll_deg=self.torch_roll_deg)
                arr.update({f"p{p.k}_tcp_pos_m": P_tcp, f"p{p.k}_tcp_R": R_tcp})
        np.savez_compressed(npz, **arr)
        return [str(jp), str(npz)]

    def plot_sections(self, path, n_sections: int = 4) -> Optional[str]:
        """단면 그림: 실측 그루브 (단면별 GrooveSpec) + 패스별 볼록 비드 (bead_geometry; 계획과 같은 비드) + 겨냥점 + 캡 위빙 폭."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return None
        idx = np.unique(np.linspace(0, len(self.stations) - 1, max(1, int(n_sections))).round().astype(int))
        fig, axes = plt.subplots(1, len(idx), figsize=(4.2 * len(idx), 4.2), squeeze=False)
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
        for ax, i in zip(axes[0], idx):
            st = self.stations[int(i)]
            sp = st.spec
            t = sp.thickness_mm
            zs = np.linspace(0.0, t, 60)
            xl, xr = wall_x_mm(sp, zs)
            xlt, _tl, xrt, trt = groove_edges_mm(sp)
            ax.plot(xl, zs, "k-", lw=1.5)
            ax.plot(np.asarray(wall_x_mm(sp, np.linspace(0.0, trt, 60))[1]), np.linspace(0.0, trt, 60), "k-", lw=1.5)
            ax.plot([xlt - 12, xlt], [t, t], "k-", lw=1.5)
            ax.plot([xrt, xrt + 12], [trt, trt], "k-", lw=1.5)
            ax.plot([xl[0], xr[0]], [0, 0], "k-", lw=1.5)
            for p in self.passes:
                jj = int(np.argmin(np.abs(p.s_mm - st.s_mm)))
                col = colors[(p.k - 1) % len(colors)]
                if p.beads:
                    poly = p.beads[jj].region()
                    ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.35, lw=0.6, ec=col)
                ax.plot(p.aim_xz_mm[jj, 0], p.aim_xz_mm[jj, 1], "o", color=col, mec="k", ms=5,
                        label=f"pass {p.k} {p.role} (L{p.layer}.{p.bead})")
                if p.kind == "cap":
                    xc, zc = p.aim_xz_mm[jj]
                    A = float(p.weave_amp_mm[jj])
                    ax.plot([xc - A, xc + A], [zc + 0.6, zc + 0.6], "-", color=col, lw=2)
            ax.set_title(f"s = {st.s_mm:.0f} mm  gap {sp.root_gap_mm:.2f}  open {st.measured.get('top_opening_mm', float('nan')):.2f}",
                         fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlim(xlt - 8, xrt + 8)
            ax.set_ylim(-1, t + 5)
            ax.grid(alpha=0.3)
            ax.set_xlabel("x [mm] (w)")
        axes[0][0].set_ylabel("z [mm] (n, backing top = 0)")
        axes[0][0].legend(fontsize=7, loc="upper left")
        fig.suptitle(f"pre-weld scan weld plan ({self.position}; {len(self.passes)} passes, convex beads; {_bevel_words(self.to_dict())}); "
                     f"process values: {self.process.source}", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)


# --------------------------------------------------------------------------------------
# 단면 (station)
# --------------------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / float(np.linalg.norm(v))


def _section_spec(spec_design: GrooveSpec, side: str, G: float, D: float, O: float, H: float, L: float, name: str) -> GrooveSpec:
    """실측 (갭 G, 깊이 D, 개구 O, hi-lo H) + 설계 (kind·루트면·뒷댐재) → 단면 GrooveSpec. 개선각 = atan(한쪽 개선 폭 / 개선면 높이),
    개선면 높이 = t − rf (루트면은 설계값 — 루트 모서리가 그 높이에서 정해진다, seam_features._root_line)."""
    kind = spec_design.kind
    t = D if side == "left" or kind != "single_bevel" else D - H
    t = max(t, 0.1)
    rf = min(float(spec_design.root_face_mm), 0.9 * t)
    run = max(0.0, O - G) / max(spec_design.n_bevels, 1)
    beta = math.degrees(math.atan2(run, max(t - rf, 1e-6))) if kind != "square" else 0.0
    return GrooveSpec(kind=kind, thickness_mm=t, bevel_angle_deg=min(max(beta, 0.0), 89.0),
                      bevel_side=side if kind == "single_bevel" else spec_design.bevel_side,
                      root_gap_mm=max(G, 0.0), root_face_mm=rf, backing=spec_design.backing,
                      weld_length_mm=L, hi_lo_mm=H, name=name)


class PlanContractError(ValueError):
    """용접 계획이 S→E 계약 (scan_core.seam_frame) 을 지키지 않는다 — 계획을 내지 않는다 (거부 이유를 담는다)."""


def plan_frame(feats: dict, S, E, *, sigma_S_m: float = float("nan"), sigma_E_m: float = float("nan")) -> SeamFrame:
    """용접 계획의 단면 좌표 (S→E 계약): 축 u = unit(E − S) (판 법선에 직교화), 깊이 축 n = 판 법선 (feats n_base) 을 그루브가 열린 쪽으로
    — 실측 상부 중심 − 루트 중심이 +n 쪽 (단면 z = 뒷댐재 상면에서 판 상면 쪽; 패스·센서 방향과 무관한 기하). w = n × u."""
    n = _unit(feats["n_base"])
    Pr = np.asarray(feats.get("root_polyline_base_m", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    Pt = np.asarray(feats.get("top_polyline_base_m", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    if Pr.shape == Pt.shape and Pr.size:
        fin = np.all(np.isfinite(Pr), axis=1) & np.all(np.isfinite(Pt), axis=1)
        if fin.any() and float(np.median((Pt[fin] - Pr[fin]) @ n)) < 0.0:
            n = -n
    return seam_frame(S, E, n, sigma_S_m=sigma_S_m, sigma_E_m=sigma_E_m)


def stations_from_features(feats: dict, spec_design: GrooveSpec, frame: SeamFrame, *, step_mm: float = 5.0,
                           window_mm: float = 10.0, min_profiles: int = 5, notes: Optional[list] = None) -> list:
    """프로파일별 특징 (features.npz 와 같은 키) + S→E 좌표 (plan_frame: 관측 끝 S, E) → 심 방향 단면 목록 (s = S 에서 잰 거리, 0..L).
    정련 직선 (line_c_base_m, u_base) 은 쓰지 않는다 — 그 u 는 패스 진행 방향이라 E→S 패스에서 S→E 와 반대다."""
    notes = notes if notes is not None else []
    if frame.n is None:
        raise ValueError("weld plan: the S->E frame needs the plate normal (plan_frame)")
    c, u, n = frame.S, frame.u, frame.n
    w = _unit(np.cross(n, u))
    Pr = np.asarray(feats["root_polyline_base_m"], dtype=float).reshape(-1, 3)
    Pt = np.asarray(feats["top_polyline_base_m"], dtype=float).reshape(-1, 3)
    gap = np.asarray(feats["root_gap_mm"], dtype=float)
    dep = np.asarray(feats["depth_mm"], dtype=float)
    opn = np.asarray(feats["top_opening_mm"], dtype=float)
    hil = np.asarray(feats.get("hi_lo_mm", np.zeros(gap.size)), dtype=float)
    ok = np.asarray(feats.get("ok", np.ones(gap.size)), dtype=bool)
    good = ok & np.all(np.isfinite(Pr), axis=1) & np.isfinite(gap) & np.isfinite(dep) & np.isfinite(opn)
    if good.sum() < max(3, int(min_profiles)):
        raise ValueError(f"weld plan: only {int(good.sum())} valid groove profiles")
    s_rel = np.full(gap.size, np.nan)
    s_rel[good] = ((Pr[good] - c) @ u) * 1e3                        # S 에서 S→E 로 [mm]
    off = np.full((gap.size, 3), np.nan)
    off[good] = (Pr[good] - c) - np.outer((Pr[good] - c) @ u, u)    # S–E 축에서 벗어난 양 (w·n 성분)
    # 베벨 쪽: 상부 개구 중심이 루트갭 중심에서 치우친 쪽 (센서 좌우와 무관)
    lat = (Pt - Pr) @ w * 1e3
    lat_g = lat[good & np.all(np.isfinite(Pt), axis=1)]
    shift = float(np.median(lat_g)) if lat_g.size else 0.0
    if spec_design.kind == "single_bevel" and abs(shift) > 0.5:
        side = "right" if shift > 0 else "left"
    else:
        side = spec_design.resolved_bevel_side() if spec_design.kind == "single_bevel" else "left"
        if spec_design.kind == "single_bevel":
            notes.append(f"bevel side not resolved from the scan (top-centre shift {shift:+.2f} mm): design side {side}")
    # hi-lo 부호: 특징의 hi_lo 는 센서 x 의 오른쪽 − 왼쪽. 센서가 본 베벨 쪽과 단면 좌표의 베벨 쪽이 다르면 센서 x = −w
    bs = np.asarray(feats.get("bevel_side", np.array([], dtype=str)))
    sens = [str(v) for v, g in zip(bs, good) if g and str(v) in ("left", "right")] if bs.size == gap.size else []
    hi_sign = 1.0
    if sens and spec_design.kind == "single_bevel":
        maj = max(set(sens), key=sens.count)
        hi_sign = 1.0 if maj == side else -1.0
    # 창 안에 min_profiles 개가 들어가도록 (빠른 스캔·낮은 fps 면 프로파일 간격이 창보다 성기다)
    ds = np.diff(np.sort(s_rel[good]))
    spacing = float(np.median(ds)) if ds.size else 0.0
    if spacing > 0.0 and window_mm < (int(min_profiles) + 0.5) * spacing:
        window_mm = (int(min_profiles) + 0.5) * spacing
        notes.append(f"section window widened to {window_mm:.1f} mm (profile spacing {spacing:.2f} mm)")
    L = frame.length_m * 1e3
    s_grid = np.arange(0.0, L + 1e-9, float(step_mm))
    if s_grid[-1] < L - 1e-6:
        s_grid = np.append(s_grid, L)
    rows = []
    for s in s_grid:
        m = good & (np.abs(s_rel - s) <= 0.5 * float(window_mm))
        k = int(m.sum())
        if k >= int(min_profiles):
            rows.append((s, k, float(np.median(gap[m])), float(np.median(dep[m])), float(np.median(opn[m])),
                         float(np.median(hil[m])) * hi_sign, np.median(off[m], axis=0)))
        else:
            rows.append((s, k, np.nan, np.nan, np.nan, np.nan, np.full(3, np.nan)))
    have = np.array([np.isfinite(r[2]) for r in rows])
    if have.sum() < 1:
        raise ValueError("weld plan: no section has enough valid profiles")
    S = np.array([r[0] for r in rows])

    def interp(vals):
        v = np.asarray(vals, dtype=float)
        return np.interp(S, S[have], v[have])
    G, D, O, H = (interp([r[i] for r in rows]) for i in (2, 3, 4, 5))
    OFF = np.stack([interp([r[6][j] for r in rows]) for j in range(3)], axis=1)
    # 단면 원점은 그 단면 (s 의 u 수직 면) 위: 성분별 중앙값이 남기는 u 방향 몫은 루트 산포에 비례할 뿐 단면 위치가 아니다 — 빼야 첫·마지막
    # 단면이 정확히 S·E (계약 검사 k·σ_끝 과 무관한 오차가 계획을 거부하지 않게)
    OFF = OFF - np.outer(OFF @ u, u)
    out = []
    for i, s in enumerate(S):
        sp = _section_spec(spec_design, side, G[i], D[i], O[i], H[i], L, f"section_{s:.0f}mm")
        origin = c + (s * 1e-3) * u + OFF[i]
        out.append(GrooveStation(s_mm=float(s), origin=origin, u=u.copy(), n=n.copy(), w=w.copy(), spec=sp, n_profiles=int(rows[i][1]),
                                 interpolated=not bool(have[i]),
                                 measured={"root_gap_mm": float(G[i]), "depth_mm": float(D[i]), "top_opening_mm": float(O[i]),
                                           "hi_lo_mm": float(H[i]), "bevel_angle_deg": float(sp.bevel_angle_deg)}))
    return out


def reference_section(stations: list) -> GrooveSpec:
    """층 배치를 정할 기준 단면: 실측 단면들 (보간 제외) 의 단면 치수 중앙값."""
    use = [st for st in stations if not st.interpolated] or list(stations)
    sp0 = use[0].spec

    def med(k):
        return float(np.median([getattr(st.spec, k) for st in use]))
    return GrooveSpec(kind=sp0.kind, thickness_mm=med("thickness_mm"), bevel_angle_deg=med("bevel_angle_deg"), bevel_side=sp0.bevel_side,
                      root_gap_mm=med("root_gap_mm"), root_face_mm=med("root_face_mm"), backing=sp0.backing,
                      weld_length_mm=sp0.weld_length_mm, hi_lo_mm=med("hi_lo_mm"), name="reference_section")


# --------------------------------------------------------------------------------------
# 층·비드 배치 (기준 단면)
# --------------------------------------------------------------------------------------
def _width(sp: GrooveSpec, z) -> np.ndarray:
    xl, xr = wall_x_mm(sp, z)
    return np.asarray(xr - xl, dtype=float)


def _level_for_area(sp: GrooveSpec, area) -> np.ndarray:
    """A(z) = area 인 z [mm] (A(z) = g·z + k·r²/2, r = max(z − rf, 0), k = n_b·tanβ 의 닫힌 역; [0, t] 로 자른다)."""
    a = np.maximum(np.asarray(area, dtype=float), 0.0)
    g, rf = float(sp.root_gap_mm), float(sp.root_face_mm)
    xl1, xr1 = wall_x_mm(sp, rf + 1.0)
    k = float(xr1 - xl1) - g                                   # 루트면 위 1 mm 당 폭 증가 = n_b·tanβ
    a_rf = g * rf
    with np.errstate(divide="ignore", invalid="ignore"):
        z_low = np.where(g > 0.0, a / max(g, 1e-12), rf)
        rest = a - a_rf
        if k > 1e-12:
            r = (-g + np.sqrt(g * g + 2.0 * k * np.maximum(rest, 0.0))) / k
        else:
            r = np.where(g > 0.0, np.maximum(rest, 0.0) / max(g, 1e-12), 0.0)
    z = np.where(a <= a_rf, z_low, rf + r)
    return np.clip(z, 0.0, sp.thickness_mm)


def _section_span_mm(process: WeldProcess, bm, position: str) -> float:
    """How far a section's surface reaches beyond the groove top edges [mm]: twice the widest free bead of the WPS at the slowest
    window speed (weave included) plus the cap bands' overlap - room for any bead over the plate tops (a numerical domain from the
    bead model, no constant)."""
    lo = float(process.speed_limits[0])
    amp = float(process.weave_amplitude_mm or 0.0)
    w = 0.0
    for r in ROLES:
        p = process.params(r, position, weave_amp_mm=amp if r == "cap" else 0.0)
        w = max(w, float(bm.width_mm(p.with_speed(p.speed_mps * lo))))
    return 2.0 * w + 2.0 * float(process.cap_overlap_mm)


def _weave_lateral_mm(process: WeldProcess, travel_deg: float, work_deg: float, band_width_mm: float) -> float:
    """The cap's weave amplitude across the seam at the WPS angles: the WPS amplitude along the torch y projected on the section
    (plan_weld's projection), or the cap band model (half the band - weave_edge_margin_mm) when the WPS gives none."""
    if process.weave_amplitude_mm is None:
        return max(0.0, 0.5 * float(band_width_mm) - float(process.weave_edge_margin_mm))
    n, u = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    w = np.cross(n, u)
    R = _tcp_R(n, w, u, w, travel_deg, work_deg)
    return float(process.weave_amplitude_mm) * abs(float(R[:, 1] @ w))


class _Section:
    """One section while the passes are laid on it (bead_geometry): its surface, the beads so far and the aim rule (AIM_RULE)."""

    def __init__(self, spec: GrooveSpec, span_mm: float):
        self.sp = spec
        self.prof = bare_profile(spec, span_mm, SECTION_STEP_MM)
        self.edges = groove_edges_mm(spec)
        self.single_bevel = spec.kind == "single_bevel"
        side = spec.resolved_bevel_side() if self.single_bevel else "left"
        self.bevel_x = 1.0 if side == "right" else -1.0                   # the bevelled wall's side in the section (wall_x_mm)
        self.beads: list = []                                              # [(role, occurrence, n_beads, BeadSection)]
        self.all_single = True

    def copy(self) -> "_Section":
        c = _Section.__new__(_Section)
        c.__dict__.update(self.__dict__)
        c.beads = list(self.beads)
        return c

    def _centre(self) -> float:
        """The groove's lateral centre at its current surface: midway between the walls where the surface meets them (the last bead's
        toes); the bare groove: the root-gap centre."""
        if not self.beads:
            return 0.0
        b = self.beads[-1][3]
        return 0.5 * (b.toe_l[0] + b.toe_r[0])

    def aim(self, role: str, occ: int, n: int, n_cap: int = 1) -> tuple:
        """The single-bead aim (x, z) [mm], its rule label and why (AIM_RULE); multi-bead layers and several cap passes: band centres
        in fill_layer / cap_passes."""
        g, rf = 0.5 * float(self.sp.root_gap_mm), float(self.sp.root_face_mm)
        covered = self.single_bevel and self.all_single and n == 1
        if role == "cap":
            xeL, _tL, xeR, _tR = self.edges
            x = 0.5 * (xeL + xeR)
            if covered and n_cap == 1:
                return x, float(self.prof.at(x)), AIM_KIRO, ("KIRO practice, cap: the lateral centre of the groove top (midway between "
                                                             "the groove top edges)")
            return x, float(self.prof.at(x)), AIM_GEOMETRIC, ("the geometric rule (the groove top centre): " + (
                "not a single-bevel groove" if not self.single_bevel else "multi-bead layers before it"))
        if role == "root":
            if self.single_bevel:
                return self.bevel_x * g, rf, AIM_KIRO, "KIRO practice, root: the corner of the root face and the bevel face"
            return 0.0, float(self.prof.at(0.0)), AIM_GEOMETRIC, "the geometric rule (the root-gap centre): not a single-bevel groove"
        prev = self.beads[-1][3] if self.beads else None
        if covered and occ == 0 and prev is not None and self.beads[-1][0] == "root":
            toe = prev.toe_l if self.bevel_x > 0 else prev.toe_r                 # the square wall is opposite the bevel
            return float(toe[0]), float(toe[1]), AIM_KIRO, ("KIRO practice, fill 1: the corner where the root bead meets the square "
                                                           "face (the root bead's toe on the square wall)")
        x = self._centre()
        if covered and occ == 1:
            return x, float(self.prof.at(x)), AIM_KIRO, ("KIRO practice, fill 2: the lateral centre of the groove at its current "
                                                         "surface (midway between fill 1's toes)")
        why = ("not a single-bevel groove" if not self.single_bevel else "multi-bead layers before it" if not self.all_single else
               f"fill {occ + 1} is beyond KIRO's 4-pass practice (root, fill 1, fill 2, cap)")
        return x, float(self.prof.at(x)), AIM_GEOMETRIC, f"the geometric rule (the groove's lateral centre at its current surface): {why}"

    def fill_layer(self, role: str, occ: int, n: int, areas, kappas) -> list:
        """Lay the layer's n beads → [(BeadSection, (aim x, aim z), rule, why)]."""
        out = []
        if n == 1:
            x, z, rule, why = self.aim(role, occ, 1)
            bead, self.prof = place_bead(self.prof, (x, z), float(areas[0]), float(kappas[0]))
            self.beads.append((role, occ, 1, bead))
            return [(bead, (x, z), rule, why)]
        self.all_single = False
        pl = pool_level(self.prof, float(np.sum(areas)))
        if pl is None:                                                     # the layer overflows: its bands over the groove top
            z1 = min(self.prof.edge_heights())
            sp_ = _layer_span(self.prof, z1)
            xl, xr = (self.edges[0], self.edges[2]) if sp_ is None or not math.isfinite(sp_[0]) else sp_
        else:
            z1, xl, xr = pl
        cuts = pool_cuts(self.prof, z1, xl, xr, n)
        sel = (self.prof.x >= xl) & (self.prof.x <= xr)
        zmid = 0.5 * (float(np.min(self.prof.z[sel])) + z1) if sel.any() else z1
        mid = _layer_span(self.prof, zmid)
        wl, wr = (xl, xr) if mid is None or not math.isfinite(mid[0]) else mid
        for b in range(n):
            bl = xl if b == 0 else float(cuts[b - 1])                     # the band (the layer's pool ends at its level, the cuts)
            br = xr if b == n - 1 else float(cuts[b])
            left, right = max(bl, wl), min(br, wr)                        # ... at the layer's mid-height
            x = 0.5 * (left + right) if left < right else 0.5 * (bl + br)
            z = float(self.prof.at(x))
            band = (None if b == 0 else float(cuts[b - 1]), None if b == n - 1 else float(cuts[b]))
            bead, self.prof = place_bead(self.prof, (x, z), float(areas[b]), float(kappas[b]), band=band)
            self.beads.append((role, occ, n, bead))
            out.append((bead, (x, z), AIM_GEOMETRIC, f"the geometric rule (the centre of bead {b + 1}'s equal-area band of the layer): "
                                                     f"{n} beads in the layer"))
        return out

    def cap_passes(self, n_c: int, areas, kappas, overlap_mm: float, spread_mm: float = 0.0) -> list:
        """Lay the cap pass(es) (weaving: the deposit lands over aim x +- spread_mm, the weave across the seam) → [(BeadSection,
        (aim x, aim z), rule, why, reinforcement after it)]."""
        out = []
        if n_c == 1:
            x, z, rule, why = self.aim("cap", 0, 1, n_cap=1)
            bead, self.prof = place_bead(self.prof, (x, z), float(areas[0]), float(kappas[0]), spread_mm=spread_mm)
            self.beads.append(("cap", 0, 1, bead))
            return [(bead, (x, z), rule, why, self.reinforcement())]
        xeL, _tL, xeR, _tR = self.edges
        x0 = xeL - overlap_mm
        ws = (xeR - xeL + 2.0 * overlap_mm) / n_c
        for b in range(n_c):
            x = x0 + (b + 0.5) * ws
            z = float(self.prof.at(x))
            band = (None if b == 0 else x0 + b * ws, None if b == n_c - 1 else x0 + (b + 1) * ws)
            bead, self.prof = place_bead(self.prof, (x, z), float(areas[b]), float(kappas[b]), band=band, spread_mm=spread_mm)
            self.beads.append(("cap", b, n_c, bead))
            out.append((bead, (x, z), AIM_GEOMETRIC, f"the geometric rule (the centre of cap band {b + 1}): {n_c} cap passes",
                        self.reinforcement()))
        return out

    def reinforcement(self) -> float:
        """The surface's highest point above the line between the groove top edges (the plate surfaces) over the groove top [mm] -
        the cap reinforcement once the caps are laid (negative: the groove is not full)."""
        xeL, tL, xeR, tR = self.edges
        X, Z = self.prof.x, self.prof.z
        m = (X >= xeL) & (X <= xeR)
        base = tL + (tR - tL) * (X[m] - xeL) / max(xeR - xeL, 1e-12)
        return float(np.max(Z[m] - base))

    def layer_levels(self, n: int) -> tuple:
        """(toe level, crown) of the last layer of n beads: the mean height of its outermost toes, its highest crown [mm]."""
        bs = [b[3] for b in self.beads[-n:]]
        return 0.5 * (bs[0].toe_l[1] + bs[-1].toe_r[1]), max(b.apex[1] for b in bs)


def _cap_area_for(sec: "_Section", n_c: int, h_target: float, a0: float, kappa_of, overlap_mm: float, spread_mm: float,
                  rel_tol: float = 1e-6) -> float:
    """The cap area per pass whose caps give the reinforcement h_target on this section (monotone in the area; bead geometry)."""
    from .bead_geometry import _solve                                       # noqa: PLC0415

    def h(a):
        s_ = sec.copy()
        try:
            s_.cap_passes(n_c, [a] * n_c, [kappa_of(a)] * n_c, overlap_mm, spread_mm)
        except ValueError:                       # a cap too small to form over the surface (its arc flatter than an earlier crown)
            return -math.inf
        return s_.reinforcement()
    if sec.reinforcement() >= h_target:                  # the surface already stands that high: no cap area gives less
        return 0.0
    # bracket by secant steps from the first guess (the reinforcement is close to linear in the area), then the bracketed solve
    pts = [(a0, h(a0))]
    a1 = a0 * (1.1 if pts[0][1] < h_target else 1.0 / 1.1)
    pts.append((a1, h(a1)))
    for _ in range(60):
        below = [p_ for p_ in pts if p_[1] < h_target]
        above = [p_ for p_ in pts if p_[1] >= h_target]
        if below and above:
            break
        (ap, hp), (ac, hc) = pts[-2], pts[-1]
        slope = (hc - hp) / (ac - ap) if ac != ap and math.isfinite(hc) and math.isfinite(hp) else 0.0
        a_n = ac + (h_target - hc) / slope if slope > 0.0 else (ac * 2.0 if hc < h_target else ac / 2.0)
        a_n = min(max(a_n, ac / 4.0), ac * 4.0)
        if a_n <= 1e-9 * a0:
            return ac
        pts.append((a_n, h(a_n)))
    below = [p_ for p_ in pts if p_[1] < h_target]
    above = [p_ for p_ in pts if p_[1] >= h_target]
    if not above:
        return max(p_[0] for p_ in pts)
    if not below:
        return min(p_[0] for p_ in pts)
    lo, h_lo = max(below)
    hi, h_hi = min(above)
    if lo > hi:                                                            # not monotone over the points tried: the smallest hit
        return hi
    lo, hi = _solve(h, lo, hi, float(h_target), f_lo=h_lo, f_hi=h_hi, tol=rel_tol * a0)
    return hi


def plan_topology(spec_ref: GrooveSpec, process: WeldProcess, bead_model: Optional[BeadModel] = None, position: str = "1G",
                  n_scale: int = 241, design: Optional[WeldDesign] = None) -> Topology:
    """기준 단면 → 층·비드 배치 + 캡 (모듈 docstring '계획' 1–2), with the convex beads (bead_geometry) laid on the reference section.

    충전 (root·fill) 비드 단면적 = λ·a_nom (공통 용착 배율 λ — WFS 고정이면 공통 속도 배율 1/λ). λ 하나마다 아래부터 층을 쌓는다:
    층 비드 수 n = the smallest n whose layer pool (the level pool of n beads' area on the current convex surface, no crown) is no
    wider than n bead widths (bead model at the scaled speed); the beads are laid (bead_geometry: convex, KIRO's aims); the layer whose
    pool would rise over the lower groove top edge is the last (candidates: without it, and with it laid over the groove top - the cap
    covers it). 캡 n_c 패스 (n_c = 위빙 최대 폭으로 판 상면 + overlap 을 덮는 최소 개수) 의 배율 μ: the cap area per pass whose caps,
    laid on the filled section, give the design reinforcement (no target: the nominal cap, more when the groove left needs it). 비용 =
    Σ_패스 ln²(a/a_nom) = N_fill·ln²λ + n_c·ln²μ 를 λ ∈ 속도 창 [1/hi, 1/lo] 에서 최소화 - on the grid of n_scale values, searched
    coarse-to-fine (every ceil(sqrt(n_scale))-th value, then around the best at a quarter of the step, down to every value; a fixed
    window [1, 1] is one value). 보강
    목표 = 용접 설계 (WeldDesign: 공칭 ± 공차). 후보의 순위: (1) 캡이 낼 수 있는 보강 (속도 창 안) 이 공칭 ± 공차 밖인 만큼, (2) μ 가 창
    밖인 만큼, (3) 비용 — 설정이 고정이면 (창 [1, 1]) 보강이 공차 안인 층 수, 없으면 공칭에 가장 가까운 층 수 (+ 표시). 보강의 결과 =
    reinforcement_ref_mm (the caps laid on the reference section: the highest point above the line between the groove top edges)."""
    design = WeldDesign.resolve(design, process)
    h_t, h_tol = design.cap_reinforcement_mm, design.cap_reinforcement_tol_mm
    bm = bead_model or process.bead_model()
    sp = spec_ref
    t = sp.thickness_mm
    A_t = float(groove_area_below_mm2(sp, t))
    lo, hi = process.speed_limits
    lb, ub = math.log(1.0 / hi), math.log(1.0 / lo)          # ln(a/a_nom) 창
    prm = {r: process.params(r, position) for r in ROLES}
    a_nom = {r: float(bm.area_mm2(prm[r])) for r in ("root", "fill")}
    a_cap = float(bm.area_mm2(prm["cap"]))
    W_c = float(_width(sp, t)) + 2.0 * process.cap_overlap_mm
    n_c = max(1, int(math.ceil(W_c / process.cap_weave_width_max_mm - 1e-9)))
    span = _section_span_mm(process, bm, position)
    amp_c = _weave_lateral_mm(process, process.role_angle("travel_angle_deg", "cap"), process.role_angle("work_angle_deg", "cap"),
                              W_c / n_c)
    p_cap = process.params("cap", position, weave_amp_mm=amp_c)

    def cap_kappa(a):
        return crown_ratio_of(bm, p_cap.with_speed(bm.speed_for_area(p_cap, max(float(a), 1e-9))))
    builds: dict = {}

    sec0 = _Section(sp, span)
    z_top0 = max(sec0.prof.edge_heights())
    xl0, xr0 = _layer_span(sec0.prof, z_top0)
    cap0 = sec0.prof.area_below(xl0, xr0, z_top0)                         # the groove's capacity up to its higher top edge

    def build(lam: float):
        key = float(lam)
        if key in builds:
            return builds[key]
        sec = _Section(sp, span)
        layers, cum, over = [], 0.0, None
        while len(layers) < 400:
            role = "root" if not layers else "fill"
            occ = sum(1 for L in layers if L["role"] == role)
            p0 = process.params(role, position, occurrence=occ)
            p_l = p0.with_speed(p0.speed_mps / lam)
            a, wb, kap = float(bm.area_mm2(p_l)), float(bm.width_mm(p_l)), float(crown_ratio_of(bm, p_l))
            # bead count: one bead if it spans its pool (its toe chord within its free width), else the smallest n whose layer pool
            # (level, no crown) is no wider than n bead widths; the layer whose deposit does not fit in the groove (mass balance: the
            # deposits so far + this layer over the groove's capacity up to its higher top edge) is the last (laid over the groove top)
            one = sec.copy()
            try:
                one.fill_layer(role, occ, 1, [a], [kap])
                single = one.beads[-1][3].chord_mm <= wb + 1e-9
            except ValueError:
                single = False
            fit, fits = None, True
            if single:
                fit, fits = (1, one, None), cum + a <= cap0 + 1e-9
            else:
                for n in range(2, MAX_BEADS_PER_LAYER + 1):
                    if cum + n * a > cap0 + 1e-9:
                        fits = False
                        break
                    pl = pool_level(sec.prof, n * a)
                    if pl is not None and pl[2] - pl[1] <= n * wb + 1e-9:
                        s_ = sec.copy()
                        s_.fill_layer(role, occ, n, [a] * n, [kap] * n)
                        fit = (n, s_, pl)
                        break
            if fit is None or not fits:
                # one layer more (overflow): over the groove top
                if fit is None:
                    n_o = max(1, int(math.ceil(float(_width(sp, t)) / wb - 1e-9)))
                    so = sec.copy()
                    try:
                        so.fill_layer(role, occ, n_o, [a] * n_o, [kap] * n_o)
                    except ValueError:
                        so = None
                else:
                    n_o, so, _pl = fit
                if so is not None:
                    toe, crown = so.layer_levels(n_o)
                    over = (layers + [dict(role=role, n=n_o, cum=cum + n_o * a, a=a, wb=wb, kap=kap, z_pool=float("nan"),
                                           width=float(_width(sp, t)), toe=toe, crown=crown)], so, cum + n_o * a)
                break
            n, sec, pl = fit
            cum += n * a
            toe, crown = sec.layer_levels(n)
            bs = [b[3] for b in sec.beads[-n:]]
            # the width the bead count was set by: one bead's toe chord, or the layer pool's width at its level
            width = float(bs[0].chord_mm) if n == 1 else float(pl[2] - pl[1])
            layers.append(dict(role=role, n=n, cum=cum, a=a, wb=wb, kap=kap, z_pool=float(pl[0]) if pl is not None else float("nan"),
                               width=width, toe=toe, crown=crown))
        builds[key] = ((layers, sec, cum), over)
        return builds[key]
    caps: dict = {}

    def cap_h(key, sec, mu):
        k2 = (key, float(mu))
        if k2 not in caps:
            s_ = sec.copy()
            s_.cap_passes(n_c, [mu * a_cap] * n_c, [cap_kappa(mu * a_cap)] * n_c, process.cap_overlap_mm, amp_c)
            caps[k2] = (s_.reinforcement(), s_)
        return caps[k2]
    xs = np.linspace(lb, ub, int(n_scale)) if ub > lb else np.array([lb])
    scored: dict = {}

    def score(ix: int):
        if ix in scored:
            return scored[ix]
        x = float(xs[ix])
        lam = math.exp(x)
        under, over = build(lam)
        out = []
        for which, cand in (("under", under), ("over", over)):
            if cand is None:
                continue
            lay, sec, cum = cand
            n_fill = sum(L["n"] for L in lay)
            R = A_t - cum                                                  # 넘치면 음수
            if h_t is not None:
                a0 = (A_t - cum) / n_c + 2.0 / 3.0 * float(h_t) * W_c / n_c                 # a first guess (a parabola over W_c)
                mu_need = _cap_area_for(sec, n_c, float(h_t), a0 if a0 > 0.0 else a_cap, cap_kappa, process.cap_overlap_mm,
                                        amp_c, rel_tol=1e-4) / a_cap                 # ranking precision (the plan solves it exactly)
            else:
                mu_need = max(1.0, R / n_c / a_cap)
            xc = math.log(max(mu_need, 1e-9))
            viol = n_c * (max(0.0, lb - xc) + max(0.0, xc - ub))
            cost = n_fill * x ** 2 + n_c * xc * xc
            mu = float(np.clip(max(mu_need, 1e-9), 1.0 / hi, 1.0 / lo))
            h_c = cap_h((lam, which), sec, mu)[0]
            out_tol = max(0.0, abs(h_c - float(h_t)) - float(h_tol)) if (h_t is not None and h_tol is not None) else 0.0
            key = (round(out_tol, 9), round(viol, 9), cost + 1e-6 * (n_fill + n_c))
            out.append((key, dict(lam=lam, which=which, layers=lay, sec=sec, cum=cum, R=R, mu_need=mu_need, mu=mu, viol=viol,
                                  cost=cost, h_c=h_c)))
        scored[ix] = out
        return out
    step = max(1, int(math.ceil(math.sqrt(len(xs)))))
    for ix in sorted(set(range(0, len(xs), step)) | {len(xs) - 1}):
        score(ix)
    while step > 1:                                          # around the best so far, at finer steps
        b0 = min((c[0], ix) for ix, cs in scored.items() for c in cs)[1]
        fine = max(1, step // 4)
        for ix in range(max(0, b0 - step), min(len(xs), b0 + step + 1), fine):
            score(ix)
        step = fine
    best = min((c for cs in scored.values() for c in cs), key=lambda c: c[0])[1]
    lam, lay, R, mu, viol, cost = best["lam"], best["layers"], best["R"], best["mu"], best["viol"], best["cost"]
    h_r, sec_c = cap_h((lam, best["which"]), best["sec"], mu)
    layers = [Layer(role=L["role"], n_beads=L["n"], frac_top=L["cum"] / A_t, z_top_ref_mm=L["toe"], bead_area_ref_mm2=L["a"],
                    bead_area_nom_mm2=a_nom[L["role"]], width_top_ref_mm=L["width"], bead_width_mm=L["wb"], z_crown_ref_mm=L["crown"],
                    z_pool_ref_mm=L["z_pool"], crown_ratio=L["kap"]) for L in lay]
    RA = n_c * mu * a_cap - R                                   # 판 상면 위 보강 단면적 (캡의 실제 용착에서, 질량 보존)
    notes = []
    in_tol = None if (h_t is None or h_tol is None) else bool(abs(h_r - float(h_t)) <= float(h_tol) + 1e-9)
    if in_tol is False:
        notes.append(f"layer plan: no layout gives the design cap reinforcement {float(h_t):g} +- {float(h_tol):g} mm with these "
                     f"settings (groove {A_t:.1f} mm2, {R:+.1f} mm2 left after the fill layers, beads {a_nom['root']:.1f}/"
                     f"{a_nom['fill']:.1f}, cap {a_cap:.1f} mm2 nominal, speed window {lo:g}..{hi:g}): closest layout - reinforcement "
                     f"{h_r:.2f} mm (report only)")
    elif viol > 0.0 and in_tol is None:
        what = f"the {float(h_t):g} mm reinforcement" if h_t is not None else "the groove left after the fill layers"
        notes.append(f"layer plan: the cap gives {what} only approximately inside the speed window {lo:g}..{hi:g} x nominal (groove "
                     f"{A_t:.1f} mm2, {R:+.1f} mm2 left after the fill layers, beads {a_nom['root']:.1f}/{a_nom['fill']:.1f}, cap "
                     f"{a_cap:.1f} mm2 nominal): closest layout - the cap deposits {n_c * mu * a_cap:.1f} mm2, reinforcement {h_r:.2f} mm")
    if not layers:
        notes.append("layer plan: the groove is filled by the cap pass(es) alone")
    return Topology(layers=layers, n_cap=n_c, cap_width_ref_mm=W_c, cap_area_ref_mm2=mu * a_cap, cap_area_nom_mm2=a_cap,
                    groove_area_ref_mm2=A_t, reinforcement_area_ref_mm2=RA, cost=float(cost), feasible=bool(RA >= -1e-9),
                    deposit_scale=float(lam), cap_scale=float(mu), reference=sp, notes=notes, reinforcement_ref_mm=float(h_r),
                    target_met=bool(in_tol) if in_tol is not None else bool(viol <= 0.0), design=design.to_dict(),
                    beads_ref=[b[3] for b in sec_c.beads])


# --------------------------------------------------------------------------------------
# 계획
# --------------------------------------------------------------------------------------
def _travel_dirs(direction: str, n_passes: int, stations: list, gravity_axis_base) -> list:
    if direction == "alternate":
        return ["S->E" if k % 2 == 1 else "E->S" for k in range(1, n_passes + 1)]
    if direction == "uphill":
        g = _unit(gravity_axis_base if gravity_axis_base is not None else (0.0, 0.0, -1.0))
        d = stations[-1].origin - stations[0].origin                # S → E
        return ["S->E" if float(d @ g) < 0.0 else "E->S"] * n_passes  # 중력 반대 쪽으로 (위로)
    return [direction] * n_passes


def _bead_work_angle(kind: str, n: int, b: int, work_deg: float, process: WeldProcess) -> float:
    """비드에 쓴 작업각 크기 (plan_weld 의 기울임 규칙과 같다): 캡·층에 비드 하나 = WPS 작업각, 벽에 붙는 옆 비드 = side_work_angle_deg,
    가운데 비드 = 0."""
    if kind == "cap" or n == 1:
        return float(work_deg)
    return float(process.side_work_angle_deg) if b in (0, n - 1) else 0.0


def _tcp_R(n: np.ndarray, w: np.ndarray, d: np.ndarray, tilt_dir: np.ndarray, travel_deg: float, work_deg: float) -> np.ndarray:
    """토치 자세 [x 진행, y = z × x, z = a]. 토치 축 a (공작물 쪽) 는 AWS A3.0 의 정의대로: 작업각 W = 진행 방향에 수직인 면에서 −n 과
    a 의 사영 사이 각 (tilt_dir 쪽), 진행각 T = a 와 그 수직 면 사이 각 (+ = 미는 각) → a = sin T · d + cos T · (cos W · (−n) + sin W ·
    tilt). KIRO 9 월 용접의 각 (WeldProcess 기본값: 끄는 각 42 / 41 / 44 / 44°, 작업각 −9 / +20 / −7 / −7°) 도 이 정의로 잰 값
    (W = 0 이면 예전 구성과 같다)."""
    T, W = math.radians(travel_deg), math.radians(work_deg)
    p = _unit(-n * math.cos(W) + tilt_dir * math.sin(W))
    a = _unit(d * math.sin(T) + p * math.cos(T))
    x = _unit(d - (d @ a) * a)
    y = np.cross(a, x)
    return np.column_stack([x, y, a])


def plan_weld(stations: list, process: Optional[WeldProcess] = None, *, spec_design: Optional[GrooveSpec] = None,
              base_frame: str = "link0", position: str = "1G", gravity_axis_base=None, notes: Optional[list] = None,
              bead_model: Optional[BeadModel] = None, design: Optional[WeldDesign] = None,
              torch: Optional[WeldTorch] = None) -> WeldPlan:
    """단면 목록 → 다층·다비드 계획 (충전 층들 + 캡) + 촘촘한 TCP 경로 (모듈 docstring '계획' 3). 단면마다 패스를 아래부터: 계획 층 (비율
    F_j) 까지 필요한 단면적 → 필요 속도 → 속도 창으로 자른 속도 → 그 속도의 용착 단면적의 볼록 비드 (bead_geometry) 를 그 단면의 지금
    표면에 놓는다 (용착 일관; 창 안이면 F_j 그대로, KIRO 처럼 설정이 고정이면 고정 용착의 비드). 겨냥 = AIM_RULE (KIRO's practice), 지금
    표면 위. 캡: 설계 보강을 내는 캡 단면적이 필요 단면적 (목표가 없으면 공칭, 그루브를 다 못 채우면 그만큼), 보강 = 놓인 캡의 결과."""
    process = process or WeldProcess()
    spec_design = spec_design or GrooveSpec()
    bm = bead_model or process.bead_model()
    notes = list(notes or [])
    if len(stations) < 2:
        raise ValueError("weld plan needs at least two sections")
    position = str(position or "1G")
    S = np.array([st.s_mm for st in stations])
    design = WeldDesign.resolve(design, process)
    topo = plan_topology(reference_section(stations), process, bm, position, design=design)
    notes.extend(topo.notes)
    N, nL = len(stations), len(topo.layers)
    F = np.array([ly.frac_top for ly in topo.layers])
    lo, hi = process.speed_limits
    occ_of = [sum(1 for ly2 in topo.layers[:j] if ly2.role == ly.role) for j, ly in enumerate(topo.layers)]  # 그 역할의 몇 번째 층

    def deposit(prm_, a_req):
        """필요 단면적 → (필요 속도, 창으로 자른 속도, 그 속도의 용착 단면적)."""
        v_req = bm.speed_for_area(prm_, max(float(a_req), 1e-9))
        v = prm_.speed_mps * float(np.clip(v_req / prm_.speed_mps, lo, hi))
        return v_req, v, float(bm.area_mm2(prm_.with_speed(v)))
    seq = [(ly.role, j, b, ly.n_beads) for j, ly in enumerate(topo.layers) for b in range(ly.n_beads)] \
        + [("cap", nL, b, topo.n_cap) for b in range(topo.n_cap)]
    K = len(seq)
    dirs = _travel_dirs(process.direction, K, stations, gravity_axis_base)
    single_bevel = stations[0].spec.kind == "single_bevel"
    side = stations[0].spec.resolved_bevel_side() if single_bevel else "left"
    bevel_x = 1.0 if side == "right" else -1.0
    ptorch = (torch.with_stick_out(process.stick_out_mm, "job weld_plan.stick_out_mm "
                                   f"{process.stick_out_mm:g} mm ({process.provenance.get('stick_out_mm', 'no provenance')})")
              if torch is not None else None)
    L_aim, L_src = aim_tip_offset_mm(process, ptorch)
    span = _section_span_mm(process, bm, position)
    prm_l = [process.params(ly.role, position, occurrence=occ_of[j]) for j, ly in enumerate(topo.layers)]
    # torch rotations per pass and station (they do not depend on the beads) and the cap's weave across the seam
    R = np.zeros((K, N, 3, 3))
    amp = np.zeros((K, N))
    width = np.zeros((K, N))
    ang = []
    for k, (role, j, b, n) in enumerate(seq):
        kind = "cap" if role == "cap" else "fill"
        occ = occ_of[j] if kind == "fill" else 0
        travel_deg = process.role_angle("travel_angle_deg", role, occ)
        work_deg = process.role_angle("work_angle_deg", role, occ)
        ang.append((travel_deg, work_deg))
        for i, st in enumerate(stations):
            d = st.u if dirs[k] == "S->E" else -st.u
            bev = st.w if side == "right" else -st.w
            if kind == "cap" or n == 1:
                tilt, wa = bev, work_deg
            elif b == 0:
                tilt, wa = -st.w, process.side_work_angle_deg
            elif b == n - 1:
                tilt, wa = st.w, process.side_work_angle_deg
            else:
                tilt, wa = bev, 0.0
            R[k, i] = _tcp_R(st.n, st.w, d, tilt, travel_deg, wa)
            if kind == "cap":                                       # 위빙 진폭 (단면 가로): WPS 의 토치 y 방향 진폭의 사영, 없으면 캡 띠 모형
                xeL, _tL, xeR, _tR = groove_edges_mm(st.spec)
                width[k, i] = (xeR - xeL + 2.0 * process.cap_overlap_mm) / n
                amp[k, i] = (float(process.weave_amplitude_mm) * abs(float(R[k, i][:, 1] @ st.w)) if process.weave_amplitude_mm is not None
                             else max(0.0, 0.5 * width[k, i] - process.weave_edge_margin_mm))
    # the beads, station by station (each section's surface evolves pass by pass)
    aim_xz = np.zeros((K, N, 2))
    area, a_need = np.zeros((K, N)), np.zeros((K, N))
    reinf = np.full((K, N), np.nan)
    beads = [[None] * N for _ in range(K)]
    rules = [[""] * N for _ in range(K)]
    whys = [[""] * N for _ in range(K)]
    levels = np.zeros((N, nL))
    narrow = 0
    k_cap0 = K - topo.n_cap
    prm_cap = process.params("cap", position)
    for i, st in enumerate(stations):
        sec = _Section(st.spec, span)
        A_i = float(groove_area_below_mm2(st.spec, st.spec.thickness_mm))
        cum, k = 0.0, 0
        for j, ly in enumerate(topo.layers):
            n = ly.n_beads
            a_req = max(F[j] * A_i - cum, 0.0) / n
            _vr, v, a_dep = deposit(prm_l[j], a_req)
            p_dep = prm_l[j].with_speed(v)
            kap = crown_ratio_of(bm, p_dep)
            w_dep = float(bm.width_mm(p_dep))
            if n > 1:
                pl = pool_level(sec.prof, n * a_dep)
                narrow += int(pl is not None and pl[2] - pl[1] > n * w_dep + 1e-9)
            recs = sec.fill_layer(ly.role, occ_of[j], n, [a_dep] * n, [kap] * n)
            if n == 1:
                narrow += int(recs[0][0].chord_mm > w_dep + 1e-9)
            for bead, xz, rule, why in recs:
                aim_xz[k, i], area[k, i], a_need[k, i] = xz, a_dep, a_req
                beads[k][i], rules[k][i], whys[k][i] = bead, rule, why
                k += 1
            cum += n * a_dep
            levels[i, j] = sec.layer_levels(n)[0]
        n_c = topo.n_cap
        R_left = A_i - cum                                                    # 충전 뒤 남은 그루브 (넘쳤으면 음수)

        def kappa_cap(a_, i=i):
            p_ = process.params("cap", position, weave_amp_mm=float(amp[k_cap0, i]))
            return crown_ratio_of(bm, p_.with_speed(bm.speed_for_area(p_, max(float(a_), 1e-9))))
        if design.cap_reinforcement_mm is not None:                          # 설계 보강까지 (비드 기하)
            a_c = _cap_area_for(sec, n_c, float(design.cap_reinforcement_mm), float(topo.cap_area_ref_mm2), kappa_cap,
                                process.cap_overlap_mm, float(amp[k_cap0, i]))
        else:                                                                 # 공칭 (그루브를 다 못 채우면 그만큼)
            a_c = max(float(bm.area_mm2(prm_cap)), R_left / n_c)
        _vr, _v, a_dep = deposit(prm_cap, a_c)
        for bead, xz, rule, why, h in sec.cap_passes(n_c, [a_dep] * n_c, [kappa_cap(a_dep)] * n_c, process.cap_overlap_mm,
                                                     float(amp[k_cap0, i])):
            aim_xz[k, i], area[k, i], a_need[k, i], reinf[k, i] = xz, a_dep, a_c, h
            beads[k][i], rules[k][i], whys[k][i] = bead, rule, why
            k += 1
    passes, aim_rows = [], []
    for k, (role, j, b, n) in enumerate(seq):
        kind = "cap" if role == "cap" else "fill"
        occ = occ_of[j] if kind == "fill" else 0
        prm = process.params(role, position, occurrence=occ)
        travel_deg, work_deg = ang[k]
        v_req = np.array([bm.speed_for_area(prm, max(a_, 1e-9)) for a_ in a_need[k]])
        ratio = v_req / prm.speed_mps
        clamped = (ratio < lo - 1e-9) | (ratio > hi + 1e-9)
        speed = prm.speed_mps * np.clip(ratio, lo, hi)
        # 일정 속도 하나 (KIRO 실행기): 필요 단면적의 심 방향 평균 → 질량 보존 속도 (창으로 자름) — 정보
        a_mean = float(np.trapz(a_need[k], S) / (S[-1] - S[0])) if S[-1] > S[0] else float(np.mean(a_need[k]))
        r_c = bm.speed_for_area(prm, a_mean) / prm.speed_mps
        c_clamped = bool(r_c < lo - 1e-9 or r_c > hi + 1e-9)
        v_const = prm.speed_mps * float(np.clip(r_c, lo, hi))
        order = np.arange(N) if dirs[k] == "S->E" else np.arange(N)[::-1]
        aim_base = np.array([st.to_base(aim_xz[k, i, 0], aim_xz[k, i, 1]) for i, st in enumerate(stations)])
        bottom = np.array([float(np.min(bd.under_z)) if bd.under_z.size else bd.toe_l[1] for bd in beads[k]])
        crown = np.array([bd.apex[1] for bd in beads[k]])
        rule0, why0 = rules[k][0], whys[k][0]
        wa_k = _bead_work_angle(kind, n, b, work_deg, process)
        wp = WeldPass(k=k + 1, kind=kind, direction=dirs[k], s_mm=S[order], aim_xz_mm=aim_xz[k][order], aim_base_m=aim_base[order],
                      R_base_tcp=R[k][order], speed_mps=speed[order], area_mm2=area[k][order], layer_from_mm=bottom[order],
                      layer_to_mm=crown[order], speed_clamped=clamped[order],
                      weave_amp_mm=amp[k][order] if kind == "cap" else None, weave_width_mm=width[k][order] if kind == "cap" else None,
                      role=role, layer=j + 1, bead=b + 1, n_beads=n, area_mean_mm2=a_mean, speed_const_mps=v_const,
                      speed_const_clamped=c_clamped, speed_required_mps=v_req[order], travel_angle_deg=float(travel_deg),
                      work_angle_deg=wa_k, area_required_mm2=a_need[k][order], travel_nominal_deg=float(travel_deg),
                      work_nominal_deg=wa_k, reinforcement_mm=reinf[k][order] if kind == "cap" else None,
                      aim_lateral_rule=rule0, aim_why=why0, beads=[beads[k][i] for i in order])
        _densify(wp, stations, process, design.end_insets_mm)
        passes.append(wp)
        if any(r_ != rule0 for r_ in rules[k]):
            notes.append(f"pass {k + 1}: the aim rule differs between sections ({sorted(set(rules[k]))})")
        aim_rows.append({"k": k + 1, "role": role, "occurrence": occ, "layer": j + 1, "bead": b + 1, "n_beads": n, "lateral": rule0,
                         "why": why0, "x_mm": float(np.median(aim_xz[k, :, 0])), "z_mm": float(np.median(aim_xz[k, :, 1]))})
    if narrow:
        notes.append(f"{narrow} section layer(s) wider than their beads span (a bead's toe chord over its free width, or a layer pool "
                     "over beads x bead width): the bead count was set on the reference section")
    aim = {"tip_offset_mm": float(L_aim), "tip_offset_source": L_src,
           "toward_bevel_x": bevel_x if single_bevel else None,
           "toward_bevel": (f"section {'+' if bevel_x > 0 else '-'}x (bevel plate on the {side} in the section = on the "
                            f"{'left' if side == 'right' else 'right'} looking S->E)" if single_bevel else "no single bevel"),
           "passes": aim_rows, "check_0922": _check_0922(aim_rows, aim_xz, bevel_x, L_aim, L_src, single_bevel)}
    L = float(S[-1] - S[0])
    bmd = bm.describe() if hasattr(bm, "describe") else {"model": type(bm).__name__}
    # the TCP roll is part of the torch orientation: the NECK_RULE for the WPS travel angles (D14 later searches only the rule's roll)
    from .weld_orient import rule_roll                                  # noqa: PLC0415
    roll, roll_rec = rule_roll(ptorch, [float(wp.travel_nominal_deg) for wp in passes])
    return WeldPlan(stations=stations, passes=passes, process=process, spec_design=spec_design, base_frame=base_frame,
                    position=position, length_mm=L, notes=notes,
                    bevel_side=side if stations[0].spec.kind == "single_bevel" else ("both" if stations[0].spec.kind == "single_v" else "none"),
                    topology=topo, levels_mm=levels,
                    bead_model=bmd, design=design, torch=ptorch, aim=aim, deposits={k + 1: list(beads[k]) for k in range(K)},
                    torch_roll_deg=float(roll), torch_roll=roll_rec)


def _check_0922(rows: list, aim_xz: np.ndarray, bevel_x: float, L_mm: float, L_src: str, single_bevel: bool) -> dict:
    """The aim rule against KIRO's 2026-09-22 session (verification only, KIRO_0922): per pass of KIRO's 4-pass layout (root, fill 1,
    fill 2, cap on KIRO's practice) the wire end relative to the root pass's - toward the bevel and up (median over the sections) -
    beside the 09-22 values (lateral at this plan's TCP -> wire end L, the bevel on the left of KIRO's travel; rise as identified)."""
    labels = [(_role_label(r_), r_) for r_ in rows]
    if not single_bevel or [lb for lb, _r in labels] != ["root", "fill 1", "fill 2", "cap"] or \
            any(r_["lateral"] != AIM_KIRO for _lb, r_ in labels):
        return {"applies": False, "why": "the plan is not KIRO's 4-pass layout on KIRO's practice (root, fill 1, fill 2, cap)"}
    out = {"applies": True, "tip_offset_mm": float(L_mm), "tip_offset_source": L_src, "source": KIRO_0922["source"],
           "bevel": KIRO_0922["bevel"], "passes": []}
    x0, z0 = aim_xz[0, :, 0], aim_xz[0, :, 1]
    for k, (lb, r_) in enumerate(labels):
        lat = float(np.median(bevel_x * (aim_xz[k, :, 0] - x0)))
        rise = float(np.median(aim_xz[k, :, 1] - z0))
        kl = kiro_0922_toward_bevel_mm(lb, L_mm)
        rr = KIRO_0922["rise_mm"][lb]
        out["passes"].append({"k": int(r_["k"]), "pass": lb, "rule_toward_bevel_from_root_mm": lat,
                              "kiro_0922_toward_bevel_from_root_mm": kl, "lateral_difference_mm": lat - kl,
                              "rule_rise_from_root_mm": rise, "kiro_0922_rise_from_root_mm": list(rr),
                              "rise_outside_0922_mm": max(rr[0] - rise, rise - rr[1], 0.0)})
    return out


def _densify(wp: WeldPass, stations: list, process: WeldProcess, insets_mm: tuple = (0.0, 0.0)) -> None:
    """단면 계획 → 촘촘한 TCP 경로 (진행 순서; 설계의 끝 inset (start, goal) 적용 — 시작 = 그 패스가 시작하는 끝). 충전: dense_step_mm
    간격. 캡: weave_sample_s 간격의 위빙."""
    S = np.array([st.s_mm for st in stations])
    order = np.argsort(S)
    S = S[order]
    i0, i1 = (float(v) for v in insets_mm)
    s0, s1 = S[0] + i0, S[-1] - i1
    if wp.direction == "E->S":
        s0, s1 = S[-1] - i0, S[0] + i1
    if (s1 - s0) * (1 if wp.direction == "S->E" else -1) <= 0:
        raise ValueError("weld plan: start/end insets leave no path")
    s_order = np.argsort(wp.s_mm)
    Ss = wp.s_mm[s_order]

    def at(s, arr):
        a = np.asarray(arr, dtype=float)[s_order]
        if a.ndim == 1:
            return np.interp(s, Ss, a)
        flat = a.reshape(a.shape[0], -1)
        return np.stack([np.interp(s, Ss, flat[:, j]) for j in range(flat.shape[1])], axis=-1).reshape(np.shape(s) + a.shape[1:])
    org = np.array([st.origin for st in stations])[order]
    W_ = np.array([st.w for st in stations])[order]
    N_ = np.array([st.n for st in stations])[order]

    def frame(s):
        o = np.stack([np.interp(s, S, org[:, j]) for j in range(3)], axis=-1)
        w = np.stack([np.interp(s, S, W_[:, j]) for j in range(3)], axis=-1)
        n = np.stack([np.interp(s, S, N_[:, j]) for j in range(3)], axis=-1)
        return o, w, n
    sign = 1.0 if wp.direction == "S->E" else -1.0
    if wp.kind == "fill":
        n_pts = max(2, int(math.ceil(abs(s1 - s0) / process.dense_step_mm)) + 1)
        s = np.linspace(s0, s1, n_pts)
        v = at(s, wp.speed_mps)
        ds = np.abs(np.diff(s)) * 1e-3
        t = np.concatenate([[0.0], np.cumsum(ds / (0.5 * (v[1:] + v[:-1])))])
        lat = np.zeros_like(s)
    else:
        t_list, s_list, lat_list = [0.0], [s0], [0.0]
        tt, ss, ph = 0.0, s0, 0.0
        dt = process.weave_sample_s
        while (ss - s1) * sign < 0:
            v = float(at(ss, wp.speed_mps))
            T = process.weave_pitch_mm * 1e-3 / v
            dw = min(process.weave_dwell_s / T, 0.4)
            r = (1.0 - 2.0 * dw) / 4.0
            ph = (ph + dt / T) % 1.0
            tt += dt
            ss = ss + sign * v * dt * 1e3
            if (ss - s1) * sign > 0:
                ss = s1
            A = float(at(ss, wp.weave_amp_mm))
            if ph < r:
                l = ph / r
            elif ph < r + dw:
                l = 1.0
            elif ph < 3 * r + dw:
                l = 1.0 - 2.0 * (ph - r - dw) / (2 * r)
            elif ph < 3 * r + 2 * dw:
                l = -1.0
            else:
                l = -1.0 + (ph - 3 * r - 2 * dw) / r
            t_list.append(tt)
            s_list.append(ss)
            lat_list.append(l * A)
        # 끝: 그 자리 (s1) 에서 램프 속도로 중앙에 돌아온다 (한 표본 만에 0 으로 끊으면 TCP 속도가 튄다)
        v = float(at(s1, wp.speed_mps))
        T = process.weave_pitch_mm * 1e-3 / v
        dw = min(process.weave_dwell_s / T, 0.4)
        ramp = float(at(s1, wp.weave_amp_mm)) / max((1.0 - 2.0 * dw) / 4.0 * T, 1e-6)       # mm/s
        while abs(lat_list[-1]) > 1e-9:
            step = math.copysign(min(abs(lat_list[-1]), ramp * dt), lat_list[-1])
            tt += dt
            t_list.append(tt)
            s_list.append(s1)
            lat_list.append(lat_list[-1] - step)
        t, s, lat = np.array(t_list), np.array(s_list), np.array(lat_list)
    axz = at(s, wp.aim_xz_mm)
    o, w, n = frame(s)
    pos = o + ((axz[:, 0] + lat)[:, None] * w + axz[:, 1][:, None] * n) * 1e-3
    R = at(s, wp.R_base_tcp)
    for i in range(R.shape[0]):                                       # 보간 뒤 직교화
        U, _, Vt = np.linalg.svd(R[i])
        R[i] = U @ Vt
    wp.dense_t_s, wp.dense_s_mm, wp.dense_pos_m, wp.dense_R, wp.dense_lateral_mm = t, s, pos, R, lat


def check_plan_contract(plan: WeldPlan, frame: SeamFrame, *, k: float = K_DEFAULT) -> tuple:
    """계획이 S→E 계약을 지키는가 (계획과 독립인 관측 끝 S, E 로 잰다; 허용 = k·σ_끝, 새 상수 없음):
      * 축: u·(E − S) > 0, 모든 단면의 u 도 (E − S) 쪽.
      * 첫 단면 = S, 마지막 단면 = E (축 방향 거리 k·σ 안).
      * 패스마다 표기한 방향 ('S->E' | 'E->S') 의 시작 끝에서 시작해 다른 끝에서 끝난다 — 단면 계획 (aim) 은 끝에서, 촘촘한 TCP 경로는
        설계의 끝 inset (weld_design.end_inset_mm {start, goal}; end_insets_mm) 만큼 안쪽에서.
    반환 (위반 목록 — 비면 통과, 잰 값 dict)."""
    d = frame.E - frame.S
    tol = {X: float(k) * frame.sigma_m(X) for X in ("S", "E")}
    viol: list = []
    if not all(math.isfinite(t) and t > 0.0 for t in tol.values()):
        viol.append(f"end uncertainty unknown (sigma S {frame.sigma_S_m * 1e3:.3f} / E {frame.sigma_E_m * 1e3:.3f} mm) - the ends "
                    "cannot be checked")
    if not float(frame.u @ d) > 0.0:
        viol.append(f"axis u.(E - S) = {float(frame.u @ d) * 1e3:+.3f} mm is not positive")

    def at(P, X) -> float:                                   # 축 방향으로 끝 X 에서 잰 거리 [m]
        return float((np.asarray(P, dtype=float) - (frame.S if X == "S" else frame.E)) @ frame.u)

    def off(name, val, X):
        if not abs(val) <= tol[X]:
            viol.append(f"{name} {val * 1e3:+.2f} mm from {X} along S->E (> k sigma {tol[X] * 1e3:.2f} mm)")
    st = plan.stations
    rev = [i for i, x in enumerate(st) if not float(np.asarray(x.u, dtype=float) @ d) > 0.0]
    if rev:
        viol.append(f"{len(rev)}/{len(st)} section axes point E->S (first at section {rev[0]})")
    d0, dN = at(st[0].origin, "S"), at(st[-1].origin, "E")
    off("first section", d0, "S")
    off("last section", dN, "E")
    ins = end_insets_mm(plan)
    per = []
    for p in plan.passes:
        if p.direction not in ("S->E", "E->S"):
            viol.append(f"pass {p.k}: direction {p.direction!r} is not S->E or E->S")
            continue
        a, b = ("S", "E") if p.direction == "S->E" else ("E", "S")
        sg = 1.0 if p.direction == "S->E" else -1.0
        e0, e1 = at(p.aim_base_m[0], a), at(p.aim_base_m[-1], b)
        off(f"pass {p.k} ({p.direction}) first section", e0, a)
        off(f"pass {p.k} ({p.direction}) last section", e1, b)
        row = {"k": p.k, "direction": p.direction, "first_mm": e0 * 1e3, "last_mm": e1 * 1e3}
        if p.dense_pos_m.shape[0]:
            g0 = at(p.dense_pos_m[0], a) - sg * float(ins[0]) * 1e-3
            g1 = at(p.dense_pos_m[-1], b) + sg * float(ins[1]) * 1e-3
            off(f"pass {p.k} ({p.direction}) path start (after the start inset)", g0, a)
            off(f"pass {p.k} ({p.direction}) path end (after the end inset)", g1, b)
            row.update(path_start_mm=g0 * 1e3, path_end_mm=g1 * 1e3)
        per.append(row)
    return viol, {"first_section_from_S_mm": d0 * 1e3, "last_section_from_E_mm": dN * 1e3, "tol_S_mm": tol["S"] * 1e3,
                  "tol_E_mm": tol["E"] * 1e3, "k": float(k), "passes": per}


def weld_plan_from_features(feats: dict, extent: dict, spec_design: GrooveSpec, process: Optional[WeldProcess] = None, *,
                            base_frame: str = "link0", position: str = "1G", gravity_axis_base=None,
                            bead_model: Optional[BeadModel] = None, k: float = K_DEFAULT,
                            design: Optional[WeldDesign] = None) -> WeldPlan:
    """features.npz 와 같은 사전 + 패스 extent (관측 끝 S/E 의 base 점 'point', σ 'sigma_m', 착점 간격 'spacing_mm') → WeldPlan.
    S→E 계약 (scan_core.seam_frame): 축·심 좌표는 끝 점에서만 유도한다. 끝 점이 없거나 계획이 계약 검사 (check_plan_contract, k = 판정
    배수) 를 통과하지 못하면 PlanContractError — 계획을 내지 않는다."""
    process = process or WeldProcess()
    notes: list = []
    try:
        ends = frame_from_extent(extent)
        frame = plan_frame(feats, ends.S, ends.E, sigma_S_m=ends.sigma_S_m, sigma_E_m=ends.sigma_E_m)
    except SeamFrameError as e:
        raise PlanContractError(f"weld plan refused (S->E contract): {e}") from e
    for end in ("S", "E"):
        if not extent.get(end, {}).get("observed", False):
            notes.append(f"{end} end not observed: the plan ends at the inferred/last scanned point")
    st = stations_from_features(feats, spec_design, frame, step_mm=process.station_step_mm,
                                window_mm=process.station_window_mm, min_profiles=process.min_profiles, notes=notes)
    plan = plan_weld(st, process, spec_design=spec_design, base_frame=base_frame, position=position,
                     gravity_axis_base=gravity_axis_base, notes=notes, bead_model=bead_model, design=design)
    viol, checked = check_plan_contract(plan, frame, k=k)
    if viol:
        raise PlanContractError("weld plan refused (S->E contract: sections must run from the observed S end to the observed E end): "
                                + "; ".join(viol))
    plan.ends = {**frame.to_dict(), "contract": "u = unit(E - S); s from S; first section at S, last at E within k sigma",
                 "checked": checked}
    return plan


def format_plan_summary(d: dict) -> list:
    """weld_plan.json (WeldPlan.to_dict) → 요약 표 줄 (서버 진행 줄·클라이언트·CLI 공통)."""
    pr = d.get("process", {}) or {}
    passes = d.get("passes", []) or []

    def f3(v, w=5, dd=2):
        v = [float("nan") if x is None else float(x) for x in (v or [None] * 3)]
        return f"{v[0]:{w}.{dd}f} ({v[1]:.{dd}f}..{v[2]:.{dd}f})"
    out = [f"weld plan: {len(passes)} passes over {float(d.get('length_mm') or float('nan')):.1f} mm, "
           f"{len(d.get('stations', []) or [])} sections every {pr.get('station_step_mm', 5.0):g} mm; {_bevel_words(d)}; "
           f"process values: {pr.get('source', '?')}"]
    tp = d.get("topology") or {}
    if tp:
        ref = tp.get("reference_section") or {}
        lay = " ".join(f"L{j + 1}:{ly['n_beads']}x{float(ly['bead_area_ref_mm2']):.1f}" for j, ly in enumerate(tp.get("layers") or []))
        out.append(f"  layers on the reference section (gap {float(ref.get('root_gap_mm') or 0):.2f}, t {float(ref.get('thickness_mm') or 0):.2f}, "
                   f"bevel {float(ref.get('bevel_angle_deg') or 0):.1f} deg, root face {float(ref.get('root_face_mm') or 0):.1f} mm; groove "
                   f"{float(tp.get('groove_area_ref_mm2') or 0):.1f} mm2): {lay or '-'} + cap {tp.get('n_cap')}x"
                   f"{float(tp.get('cap_area_ref_mm2') or 0):.1f} mm2 (beads x area; deposit x{float(tp.get('deposit_scale') or 1):.3f}, "
                   f"cap x{float(tp.get('cap_scale') or 1):.3f} of nominal; speed window {_win(pr)}"
                   + (f"; cap reinforcement {float(tp['reinforcement_ref_mm']):.2f} mm" if tp.get("reinforcement_ref_mm") is not None else "")
                   + _design_words(tp.get("weld_design")) + ")")
    ar = d.get("aim_rule") or {}
    if ar:
        rows = ar.get("passes") or []
        if "provisional" in str(ar.get("lateral", "")):                    # a plan written before 2026-10-13 (aim0922)
            lat = "provisional geometric rule (layer band centre), not yet from data"
        elif rows and "09-22" in str(ar.get("lateral", "")) and not str(ar.get("lateral", "")).startswith("KIRO's practice"):
            lat = "KIRO 09-22 reproduction (a plan written before 2026-10-13 night)"
        elif rows:
            lat = "; ".join(f"pass {r_['k']} {_role_label(r_)}: " + str(r_.get("why", "")).replace("KIRO practice, ", "KIRO practice - ")
                            .split(": ", 1)[-1] + (f" (x {float(r_['x_mm']):+.2f}, z {float(r_['z_mm']):.2f})" if r_.get("x_mm") is not None else "")
                            + ("" if r_.get("lateral") == AIM_KIRO else " [geometric rule]") for r_ in rows)
        else:
            lat = str(ar.get("lateral"))
        convex = bool(d.get("bead_geometry"))                              # plans before 2026-10-13 night: flat layers
        out.append("  aim: vertical = the wire end on the current surface (" + ("the earlier passes' convex beads); lateral (KIRO's "
                   "practice from the measured section) = " if convex else "the top of the layer below); lateral = ") + lat)
        ck = ar.get("check_0922") or {}
        if ck.get("applies"):
            out.append("  aim vs KIRO 09-22 (verification; toward the bevel / up from the root pass, mm): " + "; ".join(
                f"{c['pass']} {float(c['rule_toward_bevel_from_root_mm']):+.2f} vs {float(c['kiro_0922_toward_bevel_from_root_mm']):+.2f}, "
                f"up {float(c['rule_rise_from_root_mm']):.1f} vs {c['kiro_0922_rise_from_root_mm'][0]:g}..{c['kiro_0922_rise_from_root_mm'][1]:g}"
                for c in ck.get("passes") or [] if c["pass"] != "root"))
    out.append("  pass role L.b  dir    " + ("crown [mm]    " if d.get("bead_geometry") else "layer top [mm]")
               + "       area [mm2]           speed [mm/s]      aim x [mm]   time [s]  weave amp [mm]")
    for p in passes:
        wv = f3(p.get("weave_amp_mm"), 4, 2) if p.get("weave_amp_mm") else "-"
        ax = (p.get("aim_x_mm") or [float("nan")])[0]
        lb = f"{p.get('layer', '?')}.{p.get('bead', '?')}"
        out.append(f"  {p['k']:3d}  {p.get('role', p['kind']):4s} {lb:4s} {p['direction']:5s} {f3(p.get('layer_to_mm'))}  "
                   f"{f3(p.get('area_mm2'), 5, 1)}  "
                   f"{f3(p.get('speed_mm_s'), 4, 2)}  {float('nan') if ax is None else float(ax):+6.2f}      "
                   f"{float(p.get('duration_s') or float('nan')):6.1f}   {wv}"
                   + (f"  [{p['n_speed_clamped']} sections need a speed outside the window]" if p.get("n_speed_clamped") else ""))
    bd = [p for p in passes if isinstance(p.get("bead_geometry"), dict)]
    if bd:
        out.append("  convex beads (median; toes z left / right, crown z, crown ratio): " + "; ".join(
            f"{p['k']} {float(p['bead_geometry']['toe_left_z_mm'][0]):.2f}/{float(p['bead_geometry']['toe_right_z_mm'][0]):.2f}, "
            f"{float(p['bead_geometry']['crown_z_mm'][0]):.2f}, {float(p['bead_geometry']['crown_ratio'][0]):.2f}" for p in bd))
    const = [p for p in passes if p.get("speed_const_mm_s") is not None]
    if const:
        def _req(p):
            r = [float("nan") if x is None else float(x) for x in (p.get("speed_required_mm_s") or [None] * 3)]
            return f"{r[1]:.2f}..{r[2]:.2f}"
        out.append(f"  one constant speed per pass (constant-speed executor; seam-mean required area at the role's WFS, speed window "
                   f"{_win(pr)}): " + "; ".join(f"{p['k']} {p.get('role', p['kind'])} {float(p['speed_const_mm_s']):.2f} mm/s "
                                                 f"(required {float(p.get('area_mean_mm2') or float('nan')):.1f} mm2, {_req(p)} mm/s "
                                                 "along the seam)" for p in const))
    ei = ((d.get("weld_design") or {}).get("end_inset_mm") or {})
    if float(ei.get("start") or 0.0) > 0.0 or float(ei.get("goal") or 0.0) > 0.0:
        out.append(f"  weld line ends: every pass starts {float(ei.get('start') or 0.0):g} mm and ends {float(ei.get('goal') or 0.0):g} mm "
                   "inside the measured seam ends (weld_design.end_inset_mm; sensing unchanged)")
    caps = [p for p in passes if p.get("reinforcement_mm")]
    if caps:
        out.append("  cap reinforcement over the plate top [mm]: " + "; ".join(
            f"{p['k']} {f3(p['reinforcement_mm'], 4, 2)}" for p in caps))
    fz = d.get("feasibility", []) or []
    ra = d.get("reachability") or {}
    if ra.get("checked") is False and ra.get("note"):                     # D14 을 하지 않았다 (용접 토치가 이 팔에 없다)
        out.append(f"  robot check: {ra['note']}")
    if fz:
        unchecked = [f for f in fz if "reachable" in f and f["reachable"] is None]     # 플랫폼이 답을 받지 못한 확인 (하지 않음)
        bad = [f for f in fz if not f.get("reachable") and not any(f is u for u in unchecked)]
        margin = min(float(f.get("min_limit_margin_rad") if f.get("min_limit_margin_rad") is not None else float("nan")) for f in fz)
        out.append("  robot check (MoveIt, no motion): " + (f"all {len(fz)} passes reachable, joint-limit margin >= {margin:.2f} rad"
                   if not bad and not unchecked else "; ".join([f"pass {f['k']} NOT reachable ({f.get('reason')})" for f in bad]
                                                              + [f"pass {f['k']} NOT CHECKED ({f.get('reason')})" for f in unchecked])))
        for note in dict.fromkeys(f["pool_note"] for f in fz if f.get("pool_note")):   # 확인의 프로세스 풀이 수명 규칙 일부 없이 돌았다
            out.append(f"  robot check note: {note}")
        for tool in dict.fromkeys(f["tool"] for f in fz if f.get("tool")):            # 확인의 도구 (scan_core.weld_torch.reach_tool)
            out.append(f"  robot check tool: {tool}")
    tr = d.get("torch_roll") or {}
    if tr.get("reason"):                                                          # TCP 롤의 선택 (scan_core.weld_orient.choose_orientation)
        out.append(f"  torch roll: {tr['reason']}")
    cr = d.get("clearance_requirement") or {}
    if cr:
        miss = cr.get("missing") or []
        out.append(f"  required clearance tool <-> specimen: {float(cr.get('required_mm') or float('nan')):.2f} mm = k {cr.get('k')} x "
                   f"{float(cr.get('sigma_mm') or float('nan')):.2f} mm ("
                   + ", ".join(f"{s_['name']} {s_['sigma_mm']:.2f}" for s_ in cr.get("sources") or [] if s_.get("sigma_mm") is not None)
                   + " mm)" + (f"; not documented: {', '.join(miss)} - a lower bound" if miss else "")
                   + (f"; the gas nozzle and contact tip {float(cr['by_role']['process']['required_mm']):.2f} mm (no interpenetration, "
                      "KIRO practice)" if (cr.get("by_role") or {}).get("process") else ""))
    to = d.get("torch_orientation") or {}
    if to.get("passes") and to.get("line_step_mm") is not None:
        out.append(f"  torch orientation and D14 on the DELIVERED lines (one straight line per pass at one orientation = kiro/pass_<k>.yaml "
                   f"= result.paths; samples every <= {float(to['line_step_mm']):g} mm)")
    for o in to.get("passes") or []:                                              # 토치 자세 (scan_core.weld_orient): 정보, 실패 아님
        nm, ch = o.get("nominal") or {}, o.get("chosen")
        if ch is None:
            out.append(f"  torch pass {o['k']}: nominal travel {nm.get('travel_deg', float('nan')):+.0f} work "
                       f"{nm.get('work_deg', float('nan')):+.0f} deg - NO feasible orientation ({o.get('why')})")
            continue
        out.append(f"  torch pass {o['k']}: nominal travel {nm['travel_deg']:+.0f} work {nm['work_deg']:+.0f} -> chosen travel "
                   f"{ch['travel_deg']:+.0f} work {ch['work_deg']:+.0f} deg (deviation {float(o['deviation_deg']):.1f} deg), clearance "
                   f"{float(o['clearance_mm']):.2f} mm (required {float(o['required_mm']):.2f})"
                   + "".join(f", gas nozzle + contact tip {float(v_['clearance_mm']):.2f} mm (required {float(v_['required_mm']):.2f})"
                             for k_, v_ in (o.get("clearance_by_role") or {}).items() if k_ == "process")
                   + (f", limiting: {o['limiting_part']['role']} parts"
                      + (f" ({o['limiting_part']['link']})" if o["limiting_part"].get("link") else "") if o.get("limiting_part") else "")
                   + f", roll {float(o['torch_roll_deg']):g} deg"
                   + (f"; {o['why']}" if o.get("why") and float(o["deviation_deg"]) > 0 else ""))
    if d.get("n_interpolated_sections"):
        out.append(f"  {d['n_interpolated_sections']} section(s) interpolated from neighbours (too few valid profiles)")
    out.extend(f"  note: {n}" for n in d.get("notes", []) or [])
    return out


def _role_label(row: dict) -> str:
    """'root' / 'fill 1' / 'fill 2' / 'cap' of an aim record row (fill layers by their order)."""
    return f"fill {int(row.get('occurrence', 0)) + 1}" if row.get("role") == "fill" else str(row.get("role"))


def _design_words(wd) -> str:
    c = (wd or {}).get("cap_reinforcement_mm") or {}
    if c.get("nominal") is None:
        return ""
    return f" (design {float(c['nominal']):g}" + (f" +- {float(c['tol']):g}" if c.get("tol") is not None else "") + " mm)"


def _win(pr: dict) -> str:
    w = pr.get("speed_limits") or [None, None]
    return "fixed (the WPS speeds)" if w[0] == w[1] == 1.0 else f"{w[0]:g}..{w[1]:g} x nominal"


def _bevel_words(d: dict) -> str:
    v = d.get("bevel_plate_looking_S_to_E", "?")
    if v == "both":
        return "both plates bevelled (V)"
    if v == "none":
        return "no bevel (square)"
    return f"bevel plate on the {v} looking S->E"


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
