"""적응형 심 자동 티칭 오케스트레이터 (auto-teach): 거친 시점/종점 + 용접 자세(1G|3G) + 설계 정보 → 설계 형상이 식별된 그루브 스캔
+ 그루브 바닥(루트) 중심선.

설계: docs/design_panel_2026-09-30_adaptive_seam_teaching.md, docs/design_revision_2026-10-01_serpentine_passes.md (사용자 결정 1~3차 —
이 모듈의 사양), 설계서 §5.10. 목표는 시점/종점이 아니라 **스캔 충실도** (설계 형상 식별: 양 판 끝 관측 + 용접 길이 + 단면 치수);
산출물은 그루브 바닥(뒷댐재 상면) 중심선 base 폴리라인 + 프로파일별 특징. 베벨 쪽·진행 방향 무관; 첫 세션은 이동마다 확인(confirm).

입력 (범용 시작, 사용자 2026-10-07): 착점 두 개의 **좌표** S0/E0 (판 위, 끝 안쪽일 수 있음) + **용접 자세** job.position '1G' |
'3G' ('auto' = 미지정은 REFUSED 'give position: 1G|3G') + 그루브 규격 (specimen). 교시 자세·job 의 surface_normal·table_z_m 은
쓰지 않는다. 로봇은 아무 자세 (운송용 접힌 자세) 에서 시작한다. 1G 심은 수평에, 3G 심은 수직에 가까워야 한다 (자세와 착점의
일관성 검사, scan_l0.acquisition.initial_frame).

순서 (AutoTeach.run, 2026-10-01 3차 결정: S 보정 → E 보정 → 지그재그 패스)
   1. 게이트 (_gates): 자세 확인 + 범용 입력 (_universal_inputs: 착점·자세·중력 → n0 추정; 1G up, 3G 로봇 쪽 수평) + job 검증 →
      gate_and_arm (session_init: MoveIt/JTC 준비 확인). make_coarse_job:
      u0 = unit(E0−S0) 를 n0 에 직교 투영, 거친 standoff, pitch 0, **안쪽 들임 없음**.
   2. 획득 (_acquire): MoveIt 충돌 장면 (시편 상자 + 측정 전 보수적 바닥) → 3G: 지지면 probe (_support_probe: 테이블·지그 윗면) →
      끝 자세 (choose_3g_end_postures: 내려다보는 각 × 토치 쪽, 예상 품질 × 아래 끝 도달, 툴 몸체 전체가 지지면 위) → 손목 형태를 도달성
      으로 한 번 고름 (_plan_acquisition) → 관절 공간 이동 (MoveIt) → 판 쪽 직선 (포락선) → 드웰 S (3G: 잰 판 법선이
      추정과 보정 한 스텝보다 다르면 마주 보도록 다시 획득, _reacquire_if_oblique) → 그루브 없으면 레이저 라인 방향 유한 탐색 = 대비책,
      전부 NaN 이면 광축을 따라 광축 교차 거리로 다가가 다시 보고 (다가갈 수 없으면 같은 자세에서 한 번 더 — 확인 먼저) →
      2축 프로브 (_probe: 스텝·판정 띠는 착점 도착 허용과 읽기 정밀에서, 확인 먼저) → 형상 기반 상대 보정 (_correct_until_converged:
      광학 원점을 식별한 그루브 단면 중심 위, 측정 판 법선 방향 목표 standoff, 판 직선에 롤 맞춤; I3 = 현재 실측 자세 기준 델타; 수렴 =
      측면은 착점의, 거리는 광학 원점의 도착 허용 안; 발산은 두 번 연속 줄지 않을 때, 예산 max_corrections) → 지점 기록 (_site_record:
      루트·상부 중심, lat_S, 판 점 Q_S, delta_cb).
   3. 거친 E 로 이동 (_move_to_rough_end): 보정된 S 자세에서 **툴 자세 그대로** 한 번의 로컬 이동 — 광학 원점이 (E0 를 **측정한** 판
      평면에 투영한 점) + 광축 기하 (S 의 광축∩판 점 기준 평행이동) 로 간다. 포락선 검사 ('scan'), 센서 스트리밍, 프레임은
      transit_S_to_E/raw.npz 로 원자료만 저장 (재구성·판정 없음). 3G 도 같다 (중력 방향, 3G 바닥 규칙).
   4. 드웰 E: 국소화 (**보정 전** 국소화를 센서 x 스케일 판정용으로 보관) → 대비책 탐색 → 형상 기반 보정 → 지점 기록 (P_E, Q_E).
   5. 직선 L1 (_line_from_sites): u1 = unit(P_E − P_S) (두 측정 루트 중심, 핸드아이 병진 상쇄) 를 판 법선 n1 (Q_S ∪ Q_E 평면 적합, 센서
      쪽) 에 재직교화. 잔차 진단 (기록만): seam_from_pair (theta 기울기 인식, roll), psi_indep (두 정지 드웰의 센서 lateral) /
      psi_kin (로봇 운동학). 센서 x 스케일 판정 (_scale_check): 같은 측면 이동을 센서와 운동학이 잰 읽기 (E 의 보정 전 오프셋, 프로브
      X 스텝, S 의 보정 전 오프셋) 의 eps 가 작업의 측면 설계 허용 (개구·갭 tol / 공칭) 을 넘는지 — 판정 가능한 앞의 두 읽기가 모두
      넘을 때만 ABORTED.
      S·E 최소 거리는 착점 도착 허용·설계 길이·패스 시야 여유에서 (거친 점은 이동 전에, 측정 거리는 여기서).
   6. 패스 계획 (사용자 2026-10-01 추가 결정: 거친 S/E 는 프로파일이 안정적이도록 일부러 판 끝보다 조금 **안쪽**에 둔다 → S·E 사이
      패스만으로는 끝을 덮을 수 없다). L_SE = 보정된 S·E 루트 중심 간 거리 (u1), 잔여 R = max(0, L_design − L_SE).
      패스 1 (plan_first_pass) = 보정된 E (로봇이 있는 곳, 판 안쪽 → **E 오버스캔 없음**) → S_ext = 보정된 S 를 E 반대쪽으로
      R 연장 (S 끝이 있을 수 있는 가장 먼 곳) + 착점 오버스캔 — 목적: S 판 끝 관측. 패스 2 (plan_next_pass) = S 끝이 관측됐으면 지금 있는 곳 (S 끝 너머) → E_target =
      S_end + L_design (+ 착점 오버스캔): 양끝을 넘는 전체 길이 스캔, 보통 마지막 패스 (직선·판은 패스 1 정련값). 대비 경로 (패스 1 에서
      S 끝 미관측: 시편이 설계보다 길거나 끝이 가려짐): 그루브는 적어도 패스 1 경계 s_min 까지 → E 끝 ≤ s_min + L → 패스 2 = s_min →
      E_ext = s_min + L + 오버스캔 (E 끝 관측 보장, 포락선 조건), 패스 3 = 관측된 E − L − 오버스캔 까지 전체 길이. 게이트가
      정렬/데이터 사유로만 실패하면 전체 길이를 반대 방향으로 반복. 패스 거리 = 센서 모형 (그루브 깊이 가운데가 광축 교차 거리, 툴이
      판에서 한계를 못 지키면 그만큼 멀리; scan_l0.sensor_view), pitch job (−5°), 3G 입사
      job (+10°). 패스 시작까지는 현재 자세에서 짧은 로컬 이동 (_move_to_pass_start: 면내·재자세 구간 'scan' + 순수 법선 구간; 스캔
      없이 되돌아가는 이동 없음). **진행 방향이 바뀌어도 툴 자세 유지** (180° 손목 회전 없음): progression_sign 을 뒤집는다
      (_pass_job 이 현재 툴 자세에 가까운 부호를 고른다). 3G 피치 부호 (라운드 4 F1): 툴 자세 유지만으로는 피치 부호가 정해지지 않으므로
      패스 1 에서 **한 번** ±|피치| 중 광학 원점이 레이저 착점보다 더 높은 쪽 (입사 기울기에 더해지는 쪽, 0.34 m 에서 원점이 착점보다
      ≈ 90 mm 위; 반대 부호면 ≈ 29 mm) 을 골라 모든 패스와 정련 작업에 유지한다 (피치 부호는 정반사 회피와 무관).
   7. 오버스캔 (path_gen.pass_overscan): 모든 패스·1G·3G 에서 **레이저 착점** (레이저 평면 ∩ 측정 판 상면 위 심 직선) 이 첫 끝 앞
      오버스캔에서 다른 끝 뒤 오버스캔까지 지나가도록 (오버스캔 = 끝 판정이 요구하는 끝 너머 프로파일 (유령점 버스트 + 1)
      + 프로파일 한 칸 (속도/fps) + 도착 허용 — overscan_need_m) 광학 원점 시작/끝을 ScanFrames 로 계산 (피치 오프셋의 앞/뒤는
      툴 자세 유지 때문에 진행 방향에 따라 뒤집힌다). 판 끝 밖에서도 측정 판 상면 위 작업 높이 그대로 (I1); 센서는 아래 테이블이나
      아무것도 보지 않는다. 3G: 아래 끝 오버스캔은 3G 바닥 규칙 (광학 원점 ≥ 아래 주어진 점)
      으로 줄어들 수 있고, 끝 판정이 요구하는 길이 (도착 허용 제외) 미만이면 REFUSED 'lower end not reachable under the 3G floor rule'.
   8. 각 패스 뒤 (_refine_pass, fit_pass): 재구성 → 특징 (centre='root'; 추출 문턱은 설계·작업 허용치·열 간격에서 —
      seam_features.extractor_params) → refine_from_pass (루트 직선·판 평면, 인라이어 = k × 이 패스의 LMedS 척도, 하한 센서 양자화) →
      판 끝 아웃라이어 필터 (plate_edge_outliers: 끝 증거 구간 = 지연 + 단면 창 안에서 직선에서 k × 내부 잔차 강건 σ 넘게 벗어난
      루트 점 제거) → identify_seam_ends (**판 끝의 정의**, 모델 기반·기울기 인식, base 3D, 문턱은 scan_core.localize.EndParams 와
      scan_core.measure 원칙: '어느 면인가' 같은 식별은 설계 중간점 (깊이/2·갭/2·45°), 위치 추정은 국소 기준 둘레 k·σ; 끝 = 심 가운데를
      포함하는 연속 on-specimen 구간의 경계, 판별 sub-profile 끝 = 판 상면/그루브 바닥 증거의 괄호 중점; '관측' = 끝 너머 비시편
      프로파일이 유령점 버스트 + 1 개 연속; 끝 바로 너머의 판 위 물체 → 'end_occluded' (추정 끝 보고, accept_inferred_end 가 아니면
      PASS 아님); 판 끝 엇갈림 'plates_staggered' 는 보고) → support_plane_ends (끝이 테이블·받침 면에서 끝나면 그 교점) → 특징 통계·
      납품 중심선은 식별된 시편 구간 안에서만 → fidelity_gate('fidelity'): 측정 품질 (양끝 관측, 루트 위치 정밀도·형상 잡음 vs 작업
      qc_targets 허용치, 데이터 유효성) 이 판정이고 시편 치수 편차는 보고. 길이·양끝 위치·시간 오프셋 추정 dt = (s_fwd − s_rev)/(2v)
      (같은 자세의 정·역 패스가 같은 끝을 봤을 때) 보고.
   9. 이후 미관측 끝 (plan_next_pass 의 일반 규칙): 관측된 끝은 그대로 (이전 패스 관측은 새 직선에 투영), 미관측 끝 = 관측된 다른 끝
      ± 설계 길이 (spec.weld_length_mm; 그 점이 본 그루브 안쪽이면 본 끝 밖으로 오버스캔), 둘 다 미관측이면 스캔 길이 ≥ 설계 길이가
      되도록 대칭 연장 — 연장은 설계 길이가 정하는 기하이고 따로 상한을 두지 않는다. 포락선·3G 바닥 규칙이 끝 도달을 막으면 REFUSED
      'end not reachable'. 패스 수 예산은 auto_teach.max_passes, 게이트 허용치는 작업 파일 qc_targets.
  10. 정지 규칙: 게이트 PASS (패스 1 이어도) 이고 직전 패스가 있으면 일관성/수렴 (line_consistency: 두 패스가 잰 잡음의 k·σ —
      직선 위치·방향·판 법선, 두 패스가 모두 본 끝은 끝 σ) → PASS (meta.converged 'single_pass' | 'true'); 연속 (반대 방향) 패스가
      일관되지 않으면 방향을 갈라 본다 (direction_consistency, P8): 같은 방향 이전 패스와 반복되고 반대 방향 차이가 반복되는 방향 편향
      (|편향|/2 ≤ 끝 허용) 이면 수렴 — 납품 끝 = 두 방향의 편향 없는 평균, 편향은 extent.direction_combined·converged_detail 에 보고;
      두 방향 모두 반복된 편향이 허용보다 크면 더 돌려도 수렴할 수 없으므로 멈춤 (PASS + converged 'false'); 그 밖의 일관성 실패면
      예산 안에서 한 패스 더, 예산이 없으면 PASS + converged 'false'. DATA_FAIL (끝 미관측·식별 미완) → 반대 방향 다음 패스 (예산 안), SPEC_FAIL (측정
      품질 실패 = 특징 오인식) → 확인 먼저 (한 번 더 재측정, 다시 실패하면 종료); 애매한 길이는 다음 패스가 다시 재고 그것도 벗어나면
      종료; 'end_occluded' 는 확인 먼저 (첫 읽기는 기록하고 다음 패스가 그 끝을 다시 재서, 다시 가렸거나 못 보면 종료 — _occlusion_terminal).
      다음 패스는 항상 로봇이 있는 끝에서 반대 방향 (지그재그), 예산 auto_teach.max_passes.
  11. 출력 (out_dir): job_refined.yaml (meta.rough false, meta.status, meta.converged, teach.auto: 양끝·길이·시간 오프셋·지점·오버스캔),
      calib_refined.yaml, weld_plan.json/.npz (PASS 에서만: 다층 용접 경로 — 용접 전 스캔의 결과물), autoteach_log.json,
      specimen_report.json (마지막 패스의 시편 대비 설계 보고 — 판정 아님, S 끝에서 구간별; 콘솔 표도),
      transit_S_to_E/, pass_<k>/ (raw.npz, recon/, qc.json, fit.json, specimen_report.json), grab_<k>.npz. 중단 경로 = 기존 (센서 RESET,
      래치 task stop ×2, 부분 저장; KeyboardInterrupt/SystemExit 도 같은 경로 뒤 재발생). 세션을 닫을 때 센서 파라미터를 되돌린다.

불변 조건 (사용자, 2026-09-30): I1 판 상면 아래 금지 — 바닥 = **센서 실측** 판 상면(올리기만, 추출기 판 인라이어의 가장 높은 점) + clamp_height_m (3G: 판
평면 거리); I2 모든 높이·거리는 base 3D 점(FK·T_flange←opt)에서 중력축/측정 판 법선 투영 — 원시 z 는 센서 유효창 판정에만;
I3 보정은 현재 실측 플랜지 자세에 대한 상대 델타; I4 모든 이동은 포락선 검사 후 위반이면 **거부** (목표를 고치지 않는다).
게이트 (session_init, 시작 자세 게이트, 첫 프레임 대기, 이동마다 확인) 그대로.

안전 세부 (리뷰 2026-09-30 / 라운드 3 반영, 모두 유지):
  * clamp (C7): 작업 파일의 job.clamp_height_m (시편 고정구 기하) 하나가 job0 에서 모든 파생 작업 (거친 작업·job_ref·패스 작업)
    으로 간다 — 플랫폼 설정 키는 2026-10-08 은퇴 (안전 여유는 safety.clearance_m).
  * 센서 녹화 구간 (S1): 첫 프레임을 받은 뒤에 이동을 보낸다; 센서가 시작하지 않으면 구간 REFUSED.
  * 시작 자세 게이트 (C8/S4): 확인 프롬프트 뒤와 전송 직전 (센서 워밍업 뒤) 에 실측 자세를 다시 읽어 플랫폼 도착 허용 (SafetyPolicy
    goal_tol) 넘게 달라졌으면 REFUSED.
  * 3G 판 평면 (C6/S8/S9): 드웰마다 실측 판 평면 (잰 법선 방향 가장 로봇 쪽 판 인라이어 수준; 포락선 평면은 n0 방향 최대) 을 job_ref.plate_plane_measured 에 raise-only 로.
  * 구간 단계 태그 (M-2): 면내 성분 > 도착 허용 (pose_tol) 또는 회전이 있는 구간은 'scan'; 순수 법선 구간만 1G 'descend'/'ascend',
    3G 'approach_normal'/'retreat_normal' ('ascend'/'retreat_normal' 은 판에서 멀어지는 순수 이동일 때만 clamp 면제).
  * 보정 상한 없음 (2026-10-07, 예전 M-7 누적 캡·스텝 캡): 보정은 잰 만큼 — 이동은 포락선·MoveIt 이 검사하고, 잰 값은 시야·거리 창 안에서만 나오고,
    수렴하지 않으면 max_corrections (반복 예산) 뒤 멈춘다. 거리·탐색 간격은 센서 시야 모형 (scan_l0.sensor_view) 에서 유도.

포트 (KIRO 통합 M1, scan_l0.ports): 이 오케스트레이터는 ScanSession.robot (RobotPlatform) 의 포트만 쓴다 — 실측 자세 = pose
(PoseSource: latest_pose / snapshot / history), 이동 = backend.move_linear / move_joint (execute_leg 경유), 도달 후보·충돌 장면 =
platform.reach_candidates / set_scene_boxes, 정지 = latch (StopLatch), 플랫폼 게이트 = gates (모드·세션 초기화·상태).
플랫폼 = MoveIt 어댑터 (hongik_scan_ros.moveit_adapter, MoveIt + JTC; 측정 프레임 = 명령 프레임) — RB 제어박스 어댑터는 2026-10-07
사용자 결정으로 없앴다 (시험은 같은 포트의 가상 팔 scan_l0/tests/sim_arm.py). 불변 조건 I1~I4 와 확인·시작 자세·
첫 프레임·센서 복원은 이 모듈이 수행한다.

clamp_height_m 결정: 토치 팁이 plate_top + clamp 아래로 내려가는 로컬 이동은 **거부**한다 (I4). 예: clamp 0.18 m 에서 패스 거리 (판 상면
0.332 m) 는 팁 여유 0.16 m 이므로 패스 1 시작 이동이 REFUSED 되고 실행이 거친 거리에 머문 채 깨끗이 중단된다 (README 참조).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import os
import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scan_core.frames import make_transform, rot_z, unit
from scan_core.localize import (EndParams, PassFit, _profile_z_mm, StaticLocalization, apply_correction, fidelity_gate,
                                identify_seam_ends, localize_static, median_profile, plate_points_from_scan,
                                polyline_from_scan_features, refine_from_pass, relative_correction, seam_from_pair,
                                support_plane_ends, _SAMPLE_PROFILES)
from scan_core.path_gen import (EnvelopeError, body_min_plate_dist, laser_plane_line_param, lowest_below_origin, pass_overscan,
                                sensor_frames, tool_keypoints)
from scan_core.quality import ransac_plane
from scan_core.reconstruct import reconstruct_scan
from scan_core import seam_features as _sfmod
from scan_core import seam_track as _st
from scan_core.seam_track import SeamEstimate
from scan_core import cross_sections as _cs
from . import job_overrides as _jo
from . import post_scan as _post_scan
from . import posture_offsets as _po
from .acquisition import (cell_boxes, conservative_floor_height, initial_frame, measure_support, plan_support_probe,
                          rank_reach_candidates, visit_order)
from . import scan_quality as _sq
from scan_core.seam_features import SeamScanFeatures, _plain, extract_seam_features_along_scan, points_above_plates
from scan_core.specimen import GrooveSpec
from scan_core.types import (POSITION_REFUSAL, Calib, JobError, ProfileFrame, QCReport, RobotSample, ScanJob, ScanResult, _jsonable,
                             merge_plate_plane, position_from_orientation)
from scan_core.tool_body import body_points_base
from scan_core.validity import Z_VALID_MM_DEFAULT, valid_mask
from scan_core.weld_plan import PlanContractError, WeldProcess, weld_design_from_doc, weld_plan_from_features
from scan_core import measure as _ms
from scan_core import seam_frame as _seam_frame
from vc3dz_driver.clock import get_clock

from .config import (AutoTeachConfig, Settings, attach_tool_body, job_to_dict, qc_targets_effective, save_job,
                     specimen_from_doc, with_position)
from .executor import (STATUS_ABORTED, STATUS_ERROR, STATUS_OK, STATUS_REFUSED, ScanRefused, ScanRunResult,
                       SensorNotStarted, StartPoseMismatch, ensure_sensor, execute_leg, gate_and_arm,
                       motion_timeout_shortfall, reconstruct_and_qc, save_raw, start_pose_mismatch, frames_from_records, raw_payload,
                       prepare_leg, check_leg, start_sensor_recording, stop_sensor_recording)
from .planning import local_target_from_correction, make_local_move_plan, make_plan, seam_align_target
from .safety import PolicyError, SafetyPolicy, derive_policy
from .sensor_view import SensorView, sensor_view
from .ports import MotionError, PlatformNotReady, RobotPlatform
from .motion_types import PlatformUnavailable
from .joint_selection import DurationSearch, duration_lower_bound
from .session import ScanSession, SensorSession

__all__ = ["AutoTeach", "AutoTeachResult", "run_auto_teach", "run_refine_offline", "choose_3g_end_postures",
           "make_coarse_job", "make_refined_job", "probe_verdict", "probe_verdict_x", "flip_optical_branch",
           "confirm_move", "confirm_needed", "move_plan_text", "classify_localization", "search_order", "pass_residuals",
           "fit_pass", "pass_feature_inputs", "STATUSES", "CONFIRM_MODES", "joint_move_kind",
           "site_pre_laterals", "scale_reading", "SeamLine", "PassPlan", "plan_first_pass", "plan_next_pass", "line_consistency",
           "direction_consistency", "time_offset_estimate",
           "pass_extent_summary", "STATUS_PASS_UNRESOLVED", "ABORT_CATEGORIES"]

STATUS_PASS = "PASS"
STATUS_ALIGN_FAIL = "ALIGN_FAIL"
STATUS_SPEC_FAIL = "SPEC_FAIL"
STATUS_DATA_FAIL = "DATA_FAIL"
STATUS_DRY_RUN = "DRY_RUN"
# 1G core D13 (인터페이스 r3 §8-6, 사용자 2026-10-10): 패스 예산을 다 썼고 공차 밖 양이 남았다 — 출력은 PASS 와 같이 쓰고 미해결 목록을
# 함께 낸다 (converged_detail['unresolved']). 3G legacy 는 쓰지 않는다
STATUS_PASS_UNRESOLVED = "PASS_UNRESOLVED"
STATUSES = (STATUS_PASS, STATUS_PASS_UNRESOLVED, STATUS_ALIGN_FAIL, STATUS_SPEC_FAIL, STATUS_DATA_FAIL, STATUS_ABORTED,
            STATUS_REFUSED, STATUS_ERROR, STATUS_DRY_RUN)
# 실패 분류 (인터페이스 r3 §4.2 — KIRO error_code 로 옮기는 이름; 판정·동작은 바꾸지 않는다): _Abort(category=…) → AutoTeachResult.category.
# 주지 않으면 None (소비자가 상태로 정한다)
ABORT_CATEGORIES = ("input", "config", "motion", "sensor", "sensing", "waypoint", "internal")
CONFIRM_MODES = ("each", "large", "scan", "auto", "none")
# 'scan' (사용자 2026-10-08): 시편을 가로지르는 스캔 이동 (S→E 이동·패스 — 센서 녹화 구간) 과 관절 공간 이동 (큰 이동:
# 시작 자세·획득·3G 지지면 확인) 만 묻고, 정밀 보정·프로브·드웰 접근·패스 시작 위치로의 짧은 이동 (포락선) 은 자동
CONFIRM_KINDS_SCAN = ("scan", "joint", "joint_detour")
# 'auto' (사용자 2026-10-08, 반복 실기용): 모두 자동이고 관절 공간 이동이 관절 직선이 아니라 우회 경로 (MoveIt OMPL 등 — 관절 직선이
# 충돌 장면에 막힘) 일 때만 묻는다. 우회는 플랜지가 크게 돌아갈 수 있다 (3G s1_a: 1–2.7 m 우회)
CONFIRM_KINDS_AUTO = ("joint_detour",)


def jtc_tracking_record(hist: dict) -> dict:
    """The JTC's tracking during one joint move (platform controller_history: controller_state reference = the command, feedback =
    the measured joints, host receive time): per joint the largest |reference - feedback| and when (s after the first state), the
    worst joint - so the next 'path tolerance violation' abort is diagnosable (KIRO field 2026-10-14, run 134838)."""
    ref, fb = hist.get("ctrl_ref_positions_rad"), hist.get("ctrl_fb_positions_rad")
    if ref is None or fb is None or not len(ref):
        return {"n": 0, "note": "no controller_state received during the move"}
    err = np.abs(np.asarray(ref, dtype=float) - np.asarray(fb, dtype=float))
    t = np.asarray(hist.get("ctrl_t_rx_ns"), dtype=np.int64)
    names = [str(n) for n in hist.get("ctrl_joint_names", [])]
    i = np.argmax(err, axis=0)
    j = int(np.argmax(err.max(axis=0)))
    return {"source": "JTC controller_state reference - feedback (host receive time)", "n": int(err.shape[0]),
            "span_s": float((t[-1] - t[0]) * 1e-9), "joint_names": names, "max_error_rad": [float(x) for x in err.max(axis=0)],
            "max_error_deg": [float(math.degrees(x)) for x in err.max(axis=0)],
            "t_of_max_s": [float((t[k] - t[0]) * 1e-9) for k in i], "worst_joint": names[j] if j < len(names) else str(j),
            "worst_error_rad": float(err.max())}


def joint_move_kind(how: str) -> str:
    """관절 공간 계획의 방식 문구 → 확인 종류: 관절 직선이면 'joint', 아니면 (OMPL·경유점 우회) 'joint_detour'."""
    return "joint" if str(how or "").startswith("joint-space straight line") else "joint_detour"
REACH_CANDIDATE_BUDGET = 8  # 끝 자세 후보의 도달 시도 예산 (계산 예산 — 수치)
# the joint-space selection (KIRO field 2026-10-14, review B1; scan_l0.joint_selection), as recorded with every joint move
JOINT_SELECTION_RULE = ("the shortest planned duration under the platform's limits: candidates in the order of an exact lower bound of "
                        "any joint path's duration, the straight joint line first, MoveIt OMPL only when no candidate has an executable "
                        "straight line (a failed plan moves on); the search stops when no unchecked candidate can be shorter; the "
                        "selected candidate's checked trajectory is sent unchanged")
# 포락선 여유 L = safety.clearance_m + 정지 거리 (scan_l0.safety.derive_policy) 는 연구실 라인과 같은 값으로 유도한다 (사용자 2026-10-11:
# 포락선 규칙과 그 여유는 연구실 계산 그대로). 정지 거리의 반응 시간 항 1 / rate 에 연구실 센서 가드의 평가 주기 (lab scan_l0.sensor_guard.
# GuardThresholds.rate_hz) 를 그대로 쓴다 — 이 패키지에는 실시간 센서 가드가 없다.
ENVELOPE_REACTION_RATE_HZ = 100.0


# --------------------------------------------------------------------------------------
# 결과
# --------------------------------------------------------------------------------------
@dataclass
class AutoTeachResult:
    """run_auto_teach 결과. status ∈ STATUSES. 배열은 base 프레임 [m]."""
    status: str = STATUS_ERROR
    reason: str = ""
    out_dir: Optional[str] = None
    passes: list = field(default_factory=list)            # [{k, kind, direction, fit, verdict, residuals, extent, overscan, ...}]
    job_final: Optional[ScanJob] = None
    job_path: Optional[str] = None
    calib_path: Optional[str] = None
    weld_plan: Optional[object] = None                    # scan_core.weld_plan.WeldPlan — 용접 전 스캔의 결과물 (PASS 에서만; 2026-10-03)
    feature_medians: dict = field(default_factory=dict)
    residual_history: list = field(default_factory=list)
    delta_cb: list = field(default_factory=list)
    probe: dict = field(default_factory=dict)
    localizations: list = field(default_factory=list)
    moves: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    files: list = field(default_factory=list)
    estop_record: Optional[dict] = None
    optical_branch_flipped: bool = False
    n_move_l: int = 0
    elapsed_s: float = 0.0
    notes: list = field(default_factory=list)
    converged: Optional[str] = None                       # S5: 'true' | 'false' | 'single_pass' (job_refined.yaml meta.converged)
    converged_detail: dict = field(default_factory=dict)  # 마지막 line_consistency() 수치·사유
    sensor_restore: Optional[dict] = None                 # S6: 센서 파라미터 원본 복원 결과 (출력 쓰기 전에)
    position: Optional[str] = None                        # 주어진 용접 자세 '1G' | '3G'
    coarse_progression_sign: Optional[int] = None         # 거친 단계 툴 진행 부호 (3G: 끝 자세의 토치 쪽, 1G: 도달성으로 고른 쪽)
    rough_swapped: bool = False                           # 내부 순서가 교시와 반대 (먼저 찾아간 끝 = 교시 E: 1G 로봇에 가까운 점, 3G 위 점) —
                                                          # 내부 기록 (sites, moves, passes, line_history) 은 내부 순서, 결과물 (job_final, extent,
                                                          # weld_plan, specimen_report, 액션 결과) 은 교시 S→E
    visit_order: dict = field(default_factory=dict)       # 방문 순서·진행 부호 결정 기록 (acquisition.visit_order + 획득 선택지)
    s_taught_point: str = "start"                         # 결과물의 S 끝이 어느 교시 점 쪽인가 ('start' | 'end'): legacy 는 늘 'start' (결과물을
                                                          # 교시 S→E 로 되돌린다), 1G core 는 먼저 간 점 (S = 먼저 간 끝, 모든 곳에서)
    quality: list = field(default_factory=list)           # 적응형 스캔: 패스별 자세·품질 비율
    merge_choice: list = field(default_factory=list)      # 구간별로 고른 패스 (병합)
    posture_offsets: list = field(default_factory=list)   # 3G 주 패스마다 끝 패스 대비 측정 오프셋 (자세 의존 오차, 캘리브 자료)
    sites: dict = field(default_factory=dict)             # {'S': …, 'E': …} 지점 기록 (루트·상부 중심, lateral, 판 법선, delta_cb)
    line_history: list = field(default_factory=list)      # [SeamLine.to_dict()] — 0 = 두 드웰의 L1, k = 패스 k 뒤
    extent: dict = field(default_factory=dict)            # 마지막 패스: 양끝 (S/E) 관측·위치, 길이 [mm]
    time_offset: dict = field(default_factory=dict)       # dt = (s_fwd − s_rev)/(2v) (정·역 패스가 같은 끝을 봤을 때)
    specimen_report: Optional[dict] = None                # 마지막 패스의 시편 대비 설계 보고 (specimen_report; 판정 아님)
    yaw_align: dict = field(default_factory=dict)         # P2: 레이저 선 방향 정렬 요약 (S 추적·회전 판단·트랜싯·E 배치) — summary.csv 용
    cross_sections: Optional[dict] = None                 # 단면 상세 법선 요약 (cross_sections.json/.npz; PASS 에서만, 사용자 2026-10-08)
    category: Optional[str] = None                        # 실패 분류 (ABORT_CATEGORIES; 성공·취소·운전자 거절·분류 없는 중단은 None)
    kiro_outputs: dict = field(default_factory=dict)      # KIRO 형식 출력 (1G core: out_dir/kiro/ — scan_core.kiro_plan.write_kiro_outputs 요약,
                                                          # 못 썼으면 {'written': False, 'reason'})

    @property
    def ok(self) -> bool:
        return self.status == STATUS_PASS

    def to_dict(self) -> dict:
        return _jsonable({
            "status": self.status, "reason": self.reason, "out_dir": self.out_dir, "job_path": self.job_path,
            "calib_path": self.calib_path, "passes": self.passes, "residual_history": self.residual_history,
            "delta_cb": self.delta_cb, "probe": self.probe, "localizations": self.localizations, "moves": self.moves,
            "decisions": self.decisions, "files": self.files, "estop_record": self.estop_record,
            "optical_branch_flipped": self.optical_branch_flipped, "coarse_progression_sign": self.coarse_progression_sign,
            "n_move_l": self.n_move_l, "elapsed_s": self.elapsed_s, "notes": self.notes, "feature_medians": self.feature_medians,
            "converged": self.converged, "converged_detail": self.converged_detail, "sensor_restore": self.sensor_restore,
            "position": self.position, "rough_swapped": self.rough_swapped, "quality": self.quality, "merge_choice": self.merge_choice,
            "posture_offsets": self.posture_offsets, "sites": self.sites, "line_history": self.line_history,
            "extent": self.extent,
            "time_offset": self.time_offset, "specimen_report": self.specimen_report, "yaw_align": self.yaw_align,
            "visit_order": self.visit_order, "s_taught_point": self.s_taught_point,
            "job_final": job_to_dict(self.job_final) if self.job_final is not None else None,
            "weld_plan": ({"passes": [p.summary() for p in self.weld_plan.passes], "length_mm": self.weld_plan.length_mm,
                           "n_sections": len(self.weld_plan.stations), "bevel_side": self.weld_plan.bevel_side}
                          if self.weld_plan is not None else None),
            "cross_sections": self.cross_sections, "category": self.category, "kiro_outputs": self.kiro_outputs,
        })

    def summary(self) -> str:
        return f"auto-teach {self.status}: {self.reason} (passes={len(self.passes)}, move_l={self.n_move_l}, {self.elapsed_s:.1f} s)"


class _PassResultRef:
    """작업 프로세스에서 처리한 패스의 ScanResult 대신 (서버는 meta 만 쓴다 — 큰 배열은 돌려받지 않는다)."""

    def __init__(self, meta: dict):
        self.meta = dict(meta or {})


# 데이터 구간의 정지 프리롤 [s] (첫 프레임부터): 예전 워밍업 값 그대로 — 끝 증거에 필요한 정지 프레임 수는 감독 실기 비교 뒤 유도 (4d P12 보류).
PASS_PREROLL_S = 0.3


class _Abort(Exception):
    """오케스트레이터 내부 중단 (status, reason). safety: 이동 실패에서 온 중단 (기록용 표시) — 판단 정지 (측정·판정·운전자 거절) 는
    False. 중단 경로는 정지 명령을 보내지 않는다: 이 패키지의 정지는 액션 취소뿐이다 (사용자 2026-10-11)."""

    def __init__(self, status: str, reason: str, safety: bool = False, category: Optional[str] = None):
        super().__init__(reason)
        if category is not None and category not in ABORT_CATEGORIES:
            raise ValueError(f"_Abort category must be one of {ABORT_CATEGORIES} or None, got {category!r}")
        self.status = status
        self.reason = reason
        self.safety = bool(safety)
        self.category = category                  # 실패 분류 (ABORT_CATEGORIES; 이름만 — 중단 경로의 동작과 무관)


# --------------------------------------------------------------------------------------
# 순수 도우미 (단위 테스트 대상)
# --------------------------------------------------------------------------------------
def _angle_deg(a, b) -> float:
    ua, ub = unit(a), unit(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(ua @ ub)))))


def _fin(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def make_coarse_job(job0: ScanJob, view: SensorView, *, depth_m: float = 0.0, opening_m: float = 0.0, relevel: bool = True,
                    speed_mps: Optional[float] = None, range_m: Optional[float] = None) -> ScanJob:
    """거친 작업 → S/E 드웰·S→E 이동용 작업 job_c (2026-10-01 흐름: **안쪽 들임 없음** — 거친 시점 S0 자체로 접근한다).

    * 1G: n0 = up (재수평화; 판은 수평으로 가정하고 실제 법선은 드웰이 측정한다). 3G: n0 = 거친 법선.
    * u0 = unit(E0 − S0) 를 n0 에 직교 투영 (path_gen.sensor_frames 는 u ⟂ n 을 요구; 거친 점의 높이 오차가 툴 프레임을 비틀지 않게).
      start = S0, end = S0 + L0·u0 (길이 유지).
    * standoff (착점까지 광축 거리) = view.coarse_range_m(깊이, 입사 기울기, 개구) — 대략 착점 오차를 시선 방향·측면 어느 쪽으로든 가장
      크게 견디는 거리 (range_m 으로 덮어쓸 수 있다), pitch 0, incidence 0 (1G; 3G 는 job 값), 리드 0.001, speed = speed_mps (None = job0 의 패스
      속도; 호출자가 플랫폼 한계로 자른다). plate_top_measured_m 은
      그대로(None) — 첫 접근 높이만 거친 값에서 온다. 3G 바닥 규칙의 '아래 주어진 점' = start/end 중 낮은 점.
    * 3G vertical_direction = S0/E0 높이에서 (S0 가 위면 'down', 아래면 'up' — _pass_job 과 같은 규칙; 라운드 4 F4): 거친 작업의 값을
      그대로 복사하면 (템플릿 'down') S0 가 아래 점일 때 진행이 E0 → S0 가 되어 접근이 E0 로 갔다. 진행은 항상 S0 → E0.
    """
    S0 = np.asarray(job0.start, dtype=float)
    E0 = np.asarray(job0.end, dtype=float)
    orientation = job0.resolved_orientation()
    n0 = job0.up_unit if (relevel and orientation == "flat_1G") else job0.normal_unit
    d = E0 - S0
    L0 = float(np.linalg.norm(d))
    d_ip = d - n0 * float(d @ n0)
    if np.linalg.norm(d_ip) < 1e-9:
        raise JobError("rough start/end are along the plate normal")
    u0 = unit(d_ip)
    kw = job_to_dict(job0)
    tilt = 0.0 if orientation == "flat_1G" else float(job0.incidence_tilt_deg)
    # 입사 기울기에서 광학 원점은 반지름 standoff 호 위 (광축 거리). 드웰 보정은 판 법선 거리를 이 작업의 standoff·cos(tilt) 로 맞춘다
    # (AutoTeach._coarse_plate_m) — 같은 작업에서 나오므로 S 드웰에서 법선 거리 보정이 거의 없다 (2026-10-06 g3_07)
    sto = float(view.coarse_range_m(depth_m, tilt, opening_m) if range_m is None else range_m)
    kw.update(start=S0.tolist(), end=(S0 + u0 * L0).tolist(), surface_normal=n0.tolist(),
              standoff_m=sto, pitch_deg=0.0,
              incidence_tilt_deg=tilt,
              lead_in_m=0.001, lead_out_m=0.001, speed_mps=float(job0.speed_mps if speed_mps is None else speed_mps), orientation=orientation,
              job_id=(job0.job_id or "rough") + "_coarse")
    if orientation == "vertical_3G":
        kw["vertical_direction"] = "down" if float(job0.height(S0)) >= float(job0.height(S0 + u0 * L0)) else "up"
    kw["torch_lateral_m"] = tuple(kw["torch_lateral_m"])
    job = ScanJob(**kw)
    job.validate()
    return job




def wrist2_branch_problem(names, q, required: str) -> Optional[str]:
    """손목 형태 검사: required 'neg'/'pos' 와 손목2 부호가 다르거나 특이점 근처 (|손목2| < 5°) 면 사유, 맞으면 None. 같은 툴 자세의 두 손목
    해 (손목1 ±180°, 손목2 부호, 손목3 ±180°) 는 3G 자세에서 심 위치가 3–4 mm 달랐다 (2026-10-06, config wrist2_branch_3g)."""
    if required not in ("neg", "pos"):
        return None
    names = [str(n) for n in names]
    q = np.asarray(q, dtype=float)
    if "wrist2" not in names:                                       # 관절은 이름으로만 (색인 가정 없음)
        return "no wrist2 joint in the joint states"
    # 형태 = 손목2 의 sin 부호: 2π 동치 표현 (예 +290° = −70°) 은 같은 형태 — (−180, 180] 로 감아 본다
    w2 = (math.degrees(float(q[names.index("wrist2")])) + 180.0) % 360.0 - 180.0
    if abs(w2) < 5.0:
        return f"wrist2 {w2:+.1f} deg is at the wrist singularity - move the arm to a pose with wrist2 {'negative' if required == 'neg' else 'positive'}"
    if (w2 > 0.0) != (required == "pos"):
        return (f"the robot is in the other wrist configuration (wrist2 {w2:+.1f} deg; 3G runs need wrist2 "
                f"{'negative' if required == 'neg' else 'positive'}, auto_teach.wrist2_branch_3g) - at a safe height, by direct teaching "
                "bend the wrist the other way (wrist2 through 0) and turn wrist1 about half a turn, then send the goal again")
    return None


def overscan_need_m(job: ScanJob, ghost_burst_frames: int) -> tuple:
    """착점 오버스캔 (계획, 최소) [m] — 설정 상수 없이 (사용자 2026-10-07): 최소 = 끝 판정이 '관측' 으로 보는 끝 너머 프로파일
    (유령점 버스트 + 1 개, scan_core.localize.EndParams.confirm_profiles) + 프로파일 한 칸 = (버스트 + 2) × 착점 간격 (패스 속도 / fps),
    계획 = 최소 + 도착 허용 (job.pose_tol_m; 실제 착점이 그만큼 어긋나도 최소는 남는다). 3G 바닥 규칙이 끝을 최소 아래로 줄이면 그
    끝은 관측될 수 없다 (거부)."""
    need = (max(0, int(ghost_burst_frames)) + 2) * float(job.speed_mps) / float(job.profile_fps)
    return need + float(job.pose_tol_m), need


def pass_standoff_clear_of_plate(job: ScanJob, calib: Calib, depth_m: float, *, limit_m: float, range_max_m: float) -> tuple:
    """패스 작업의 standoff (루트까지 광축 거리, 호 반지름) 를 툴 (토치 팁·툴 몸체) ↔ 판 상면 거리가 limit_m (한계 + 도착 허용) 이상인
    가장 작은 값으로 (사용자 2026-10-07: 그루브는 광축 교차 거리 근처, 단 툴이 판에서 안전 한계를 지키는 범위에서). standoff 를 Δ 늘리면
    툴 전체가 R_tilt·n 방향으로 Δ 평행 이동해 모든 툴 점의 판 거리가 Δ·cos(입사) 늘어난다 — 모자란 여유 m 에 Δ = m / cos(입사) 가 정확하다.
    판 상면 = 시작 루트 점 + n·깊이를 지나는 n 평면 (직선이 판과 나란해 패스 내내 같다). 반환 (작업, 올린 양 m). 올린 루트 거리
    (standoff / cos(피치)) 가 센서 창 (range_max_m) 을 넘으면 JobError."""
    fr = sensor_frames(job, calib)
    T = fr.T_base_flange(0.0)
    n = fr.n
    P_top = fr.start + n * float(depth_m)
    d = math.inf
    if job.check_torch_tip:
        d = float((tool_keypoints(T, calib, job)["tip"] - P_top) @ n)
    db = body_min_plate_dist(np.asarray(T, dtype=float)[None], calib, P_top, n)
    if db is not None:
        d = min(d, float(db[0]))
    short = float(limit_m) - d
    if not (short > 0.0):
        return job, 0.0
    raise_m = short / math.cos(math.radians(float(job.incidence_tilt_deg)))
    s_new = float(job.standoff_m) + raise_m
    r_root = s_new / math.cos(math.radians(float(job.pitch_deg)))
    if r_root > float(range_max_m):
        raise JobError(f"keeping the tool {limit_m * 1e3:.0f} mm from the plate needs the root at {r_root * 1e3:.0f} mm, beyond the sensor "
                       f"window {range_max_m * 1e3:.0f} mm (incidence {job.incidence_tilt_deg:+.0f} deg)")
    out = ScanJob.from_dict({**job_to_dict(job), "torch_lateral_m": tuple(job.torch_lateral_m), "standoff_m": s_new})
    out.plate_top_measured_m = job.plate_top_measured_m
    out.plate_plane_measured = job.plate_plane_measured
    out.floor_z_m = job.floor_z_m
    return out, raise_m


def choose_3g_end_postures(job0: ScanJob, calib: Calib, cfg: AutoTeachConfig, view: SensorView, spec: Optional[GrooveSpec], *,
                           quality_fn, t_max_deg: float, ghost_burst_frames: int, step_deg: float = 1.0,
                           support_measured: bool = True) -> dict:
    """3G 끝 자세 (거친 단계·패스 1) 후보 (사용자 2026-10-07: 테이블·지그에 부딪히지 않게 위에서 내려다보되, 각도는 고정 규칙이 아니라
    적응적으로). job0: 3G 거친 작업 (start = 위 착점 S0, end = 아래 착점 E0), table_z_m = 잰 지지면 (필수).

    내려다보는 입사 기울기 t (0..t_max, step) × 진행 부호 (토치 위/아래) 마다 두 자세를 본다:
      * E 드웰: 거친 거리 (view.coarse_range_m: 착점 오차를 모든 방향으로 가장 크게 견디는 거리, 기울기마다 — 툴이 지지면·바닥 여유를
        못 지키면 지킬 만큼 멀리, 루트가 창 안에 남는 데까지), 피치 0, 광축이 E0 를 향함 (S→E 이동은 자세를 유지한 평행 이동)
      * 패스 1 시작: 착점 = 아래 목표 (E0 + u0·(R + 오버스캔), R = 설계 길이 − L0; 지지면 아래로는 오버스캔
        까지만 — 시편은 지지면 위에 서 있다), 패스 판 거리 (view.pass_plate_m: 그루브 깊이 가운데가 광축 교차 거리) + 그루브 깊이 —
        툴이 판면에서 한계 + 도착 허용을 못 지키는 자세면 지킬 만큼만 멀리 (pass_standoff_clear_of_plate), 피치 ±|p0| 중 원점이 착점보다
        높은 쪽
    제약 ('된다'): 툴 몸체 포함 가장 낮은 점 ≥ 지지면 + 한계 + 도착 허용 (job0.safe_clearance_m + job0.pose_tol_m), 토치 팁·툴 몸체의
    판면 거리 ≥ 같은 한계 (E 드웰·패스 1 시작; 포락선 never_cross_plate_plane), 측정 전 3G 바닥 (광학 원점·토치 팁 ≥ 아래 착점), 센서
    거리 창 (거친 거리가 기울기마다 창 안에 유도된다), 레이저 기울기 t + |p0| ≤ max_laser_tilt_deg.
    점수 = quality_fn(패스 1 시작 job)['pred'] (예상 품질) × 아래 끝 도달 비율 (여유가 모자라면 패스 1 시작을 심 (≈ 수직) 을 따라 올려야
    하는 만큼 착점이 덜 내려간다). E 드웰이 되는 후보를 점수 순 (같으면 여유 큰 순) 으로 ranked 에."""
    if job0.table_z_m is None:
        raise ValueError("choose_3g_end_postures: job.table_z_m (the measured support) is required")
    table = float(job0.table_z_m) + float(job0.safe_clearance_m) + float(job0.pose_tol_m)
    S0, E0 = np.asarray(job0.start, dtype=float), np.asarray(job0.end, dtype=float)
    n = job0.normal_unit
    up = job0.up_unit
    d = E0 - S0
    u0 = unit(d - n * float(d @ n))
    L0 = float(np.linalg.norm(d))
    depth = float(spec.groove_depth_mm) * 1e-3 if spec is not None else 0.0
    L_design = float(spec.weld_length_mm) * 1e-3 if (spec is not None and spec.weld_length_mm > 0) else None
    R = max(0.0, L_design - L0) if L_design is not None else 0.0
    ov_plan, _ov_need = overscan_need_m(job0, ghost_burst_frames)
    below = R + ov_plan
    du = float(up @ u0)
    if du < -1e-6 and support_measured:                             # 시편은 잰 지지면 위: 착점은 지지면 − 오버스캔 아래로 안 간다
        below = min(below, max(0.0, (float(E0 @ up) - (float(job0.table_z_m) - ov_plan)) / -du))   # (상한뿐인 지지면은 자르지 않는다)
    target = E0 + u0 * below
    floor3g = float(min(S0 @ up, E0 @ up))                          # 측정 전 3G 바닥 (아래 착점)

    def _floor_margin(T, j):
        kp = tool_keypoints(T, calib, j)
        names = ("opt",) + (("tip",) if j.check_torch_tip else ())
        return min(float(kp[k] @ up) for k in names) - floor3g
    lim_plate = float(job0.safe_clearance_m) + float(job0.pose_tol_m)

    def _plate_margin(T, j):
        """토치 팁·툴 몸체의 판면 (아래 목표 착점을 지나는 n 평면) 거리 − (한계 + 도착 허용): 포락선 never_cross_plate_plane 과 같은 규칙."""
        kp = tool_keypoints(T, calib, j)
        d = float((kp["tip"] - target) @ n) if j.check_torch_tip else math.inf
        db = body_min_plate_dist(np.asarray(T, dtype=float)[None], calib, target, n)
        if db is not None:
            d = min(d, float(db[0]))
        return d - lim_plate
    p0 = abs(float(job0.pitch_deg))
    pass_standoff = view.pass_plate_m(depth) + depth
    opening = float(spec.top_opening_mm) * 1e-3 if spec is not None else 0.0
    base = {**job_to_dict(job0), "torch_lateral_m": tuple(job0.torch_lateral_m)}
    rows = []
    for t in np.arange(0.0, float(t_max_deg) + 1e-9, float(step_deg)):
        t = float(t)
        row = {"tilt_deg": t}
        why = []
        if t + p0 > float(job0.max_laser_tilt_deg) + 1e-9:
            why.append(f"tilt + |pitch| {t + p0:.1f} > max_laser_tilt_deg {job0.max_laser_tilt_deg:g}")
        if why:
            rows.append({**row, "skip": "; ".join(why)})
            continue
        try:
            jc0 = make_coarse_job(ScanJob.from_dict({**base, "incidence_tilt_deg": t}), view, depth_m=depth, opening_m=opening)
        except JobError as e:
            rows.append({**row, "skip": str(e)})
            continue
        for sign in (1, -1):
            jc = ScanJob.from_dict({**job_to_dict(jc0), "torch_lateral_m": tuple(jc0.torch_lateral_m), "progression_sign": sign})
            frc = sensor_frames(jc, calib)
            T_E = frc.T_base_flange(frc.L)
            low_E = float(frc.opt_origin(frc.L) @ up) - lowest_below_origin(T_E, calib, jc)
            fm_E = _floor_margin(T_E, jc)
            short = -min(low_E - table, fm_E)
            st = math.sin(math.radians(t))
            if short > 0.0 and st > 1e-6:
                # 거친 거리를 늘려 지지면·바닥 여유를 지킨다: 호 반지름을 Δr 늘리면 툴 전체가 R_tilt·n 으로 평행 이동해 모든 툴 점이
                # Δr·sin(t) 올라간다 — 루트가 센서 창 안에 남는 데까지 (착점 오차 허용은 그만큼 줄어든다)
                r_up = float(jc.standoff_m) + short / st
                if r_up + depth / math.cos(math.radians(t)) <= view.z_far_m - view.pose_tol_m:
                    jc = ScanJob.from_dict({**job_to_dict(jc), "torch_lateral_m": tuple(jc.torch_lateral_m), "standoff_m": r_up})
                    frc = sensor_frames(jc, calib)
                    T_E = frc.T_base_flange(frc.L)
                    low_E = float(frc.opt_origin(frc.L) @ up) - lowest_below_origin(T_E, calib, jc)
                    fm_E = _floor_margin(T_E, jc)
            kpE = tool_keypoints(T_E, calib, jc)
            pb = None
            for pitch in ((p0, -p0) if p0 > 0 else (0.0,)):
                pick = None
                for ps in (1, -1):
                    kw = {**base, "start": (target - n * depth).tolist(), "end": (S0 - n * depth).tolist(), "surface_normal": n.tolist(),
                          "standoff_m": pass_standoff, "pitch_deg": float(pitch), "incidence_tilt_deg": t,
                          "progression_sign": ps, "vertical_direction": "up", "lead_in_m": 0.001, "lead_out_m": 0.001}
                    try:
                        jp, _raised = pass_standoff_clear_of_plate(ScanJob.from_dict(kw), calib, depth, limit_m=lim_plate,
                                                                   range_max_m=view.z_far_m - view.pose_tol_m)
                        frp = sensor_frames(jp, calib)
                    except JobError:
                        continue
                    rot = _rot_angle_deg(T_E[:3, :3], frp.T_base_flange(0.0)[:3, :3])
                    if pick is None or rot < pick[0]:
                        pick = (rot, jp, frp)
                if pick is None:
                    continue
                _rot, jp, frp = pick
                q0 = frp.start + frp.n * depth
                delta = float(laser_plane_line_param(frp.T_base_opt(0.0), q0, frp.u))
                if not math.isfinite(delta):
                    continue
                T_P = frp.T_base_flange(-delta)                    # 착점 = 아래 목표
                h_hit = float((frp.opt_origin(-delta) - target) @ up)
                low_P = float(frp.opt_origin(-delta) @ up) - lowest_below_origin(T_P, calib, jp)
                if pb is None or h_hit > pb[0]:
                    pb = (h_hit, float(pitch), low_P, _floor_margin(T_P, jp), jp, _plate_margin(T_P, jp))
            if pb is None:
                continue
            reach_E = _pivot_reach_m(T_E, calib, jc, E0)
            cs = {**row, "sign": sign, "coarse_range_m": float(jc.standoff_m),
                  "coarse_tol_m": view.tolerance_at_m(float(jc.standoff_m), depth, t, opening), "tip_up_m": float((kpE["tip"] - kpE["opt"]) @ up), "low_E_m": low_E, "low_pass1_m": pb[2],
                  "pitch_deg": pb[1], "origin_above_hit_m": pb[0], "table_margin_E_m": low_E - table,
                  "table_margin_pass1_m": pb[2] - table, "floor_margin_E_m": fm_E, "floor_margin_pass1_m": pb[3],
                  "margin_E_m": min(low_E - table, fm_E), "margin_pass1_m": min(pb[2] - table, pb[3]),
                  "plate_margin_E_m": _plate_margin(T_E, jc), "plate_margin_pass1_m": pb[5]}
            raise_m = max(0.0, -cs["margin_pass1_m"]) / max(abs(du), 1e-6)
            cs["coverage"] = 1.0 if below <= 1e-9 else max(0.0, (below - raise_m) / below)
            cs.update(quality_fn(pb[4]))                            # {pred, pred_why, torch_above, phi_deg, incidence_deg}
            cs["score"] = float(cs["pred"]) * cs["coverage"]
            cs["ok_E"] = cs["margin_E_m"] >= 0.0 and cs["plate_margin_E_m"] >= 0.0
            cs["ok_pass1"] = cs["margin_pass1_m"] >= 0.0
            cs["ok_plate_pass1"] = cs["plate_margin_pass1_m"] >= 0.0        # 판면 여유는 심을 따라 올려도 나아지지 않는다
            # 이 자세가 견디는 판 법선 차: 착점 둘레로 δ 돌면 툴 점이 (피벗 반경)·δ 움직인다 — E 드웰의 지지면·판면 여유를 다 쓰는 각,
            # 그리고 레이저 기울기 상한까지 남은 각 중 작은 쪽 (넘으면 첫 드웰이 잰 법선으로 자세를 다시 고른다)
            m_E = min(cs["margin_E_m"], cs["plate_margin_E_m"])
            cs["normal_tol_deg"] = max(0.0, min(math.degrees(m_E / reach_E) if reach_E > 0 else 0.0,
                                                float(job0.max_laser_tilt_deg) - t))
            rows.append(cs)
    ranked = sorted((r for r in rows if r.get("ok_E") and r.get("ok_plate_pass1")),
                    key=lambda r: (-round(r["score"], 2), -min(r["margin_E_m"], r["margin_pass1_m"]), r["tilt_deg"]))
    return {"table_floor_m": table, "target_lower_hit": target.tolist(), "target_below_E0_m": float(below), "rows": rows,
            "ranked": ranked, "feasible": bool(ranked)}


def _pivot_reach_m(T_bf: np.ndarray, calib: Calib, job: ScanJob, pivot) -> float:
    """피벗 (착점) 에서 가장 먼 툴 점 (키포인트 + 툴 몸체 껍질) 까지 [m] — 피벗 둘레 회전이 툴 점을 움직이는 반경."""
    p = np.asarray(pivot, dtype=float)
    r = max(float(np.linalg.norm(np.asarray(v, dtype=float) - p)) for v in tool_keypoints(T_bf, calib, job).values())
    B = getattr(calib, "body_points_eoat", None)
    if B is not None:
        P = body_points_base((np.asarray(T_bf, dtype=float) @ np.asarray(calib.T_flange_eoat, dtype=float))[None], B)
        r = max(r, float(np.linalg.norm(np.asarray(P, dtype=float).reshape(-1, 3) - p, axis=1).max()))
    return r


def _fit_end_points(fit: PassFit) -> tuple:
    """PassFit → ((P_first, observed_first), (P_last, observed_last)) — 프로파일(진행) 순서의 첫/마지막 끝.
    identify_seam_ends 를 거친 fit 은 diag['ends'] 를, 아니면 s_a/s_b 와 end_not_observed_* 플래그를 쓴다."""
    c, u = np.asarray(fit.c, dtype=float), unit(fit.u)
    ends = (fit.diag or {}).get("ends")
    if ends:
        return tuple((np.asarray(ends[w]["point"], dtype=float), bool(ends[w]["observed"])) for w in ("first", "last"))
    return ((c + u * float(fit.s_a), "end_not_observed_first" not in fit.flags),
            (c + u * float(fit.s_b), "end_not_observed_last" not in fit.flags))


def make_refined_job(job_base: ScanJob, fit: PassFit, *, standoff_plate_m: float, pitch_deg: float, overscan_m: float,
                     calib: Optional[Calib] = None, u_SE=None, rough_start=None, rough_end=None, plate_top_m: Optional[float] = None,
                     spec: Optional[GrooveSpec] = None, job_id: Optional[str] = None, top_polyline=None, root_polyline=None,
                     plate_plane=None, R_ref: Optional[np.ndarray] = None) -> tuple:
    """패스 정련 (PassFit) → 납품 작업 job_refined (standalone `scan` 용) + 정보 dict.

    시점/종점 = **그루브 바닥(루트) 중심선**의 S/E 끝 (u_SE 가 있으면 그 방향으로 정렬: start = S 쪽). 끝은 identify_seam_ends 의 판 모서리
    관측 끝; 관측되지 않은 끝은 거친 점의 직선 투영 ('rough_projected', 오프라인 refine 에서만 생길 수 있다 — 온라인 PASS 는 양끝 관측).
    surface_normal = n_fit (u 에 직교), standoff_m = 판 상면 기준 standoff + 그루브 깊이 (측정 중앙값, 없으면 spec 두께).
    리드 = **착점 오버스캔** 기하 (path_gen.pass_overscan, calib 필요): 착점이 각 끝을 overscan_m 지나도록 광학 원점 리드를 정한다
    (피치 오프셋 방향에 따라 양쪽이 다르다; ScanJob 리드는 > 0 이어야 하므로 최소 1 mm). 3G 는 위 끝 쪽 리드만 (R4: 아래 점에서
    시작/정지, vertical_direction 은 시점이 위면 'down', 아래면 'up'). 3G 아래 끝 (라운드 4 F1): pass_overscan(…, floor_h = 아래 점 높이)
    으로 필요한 리드를 계산하고, 아래 점에서 멈춘 광학 원점의 착점이 그 끝을 지나는 양 (구조적 오버스캔) 이 overscan_m 보다 작으면
    info['short_overscan'] 로 알린다 (조용한 1 mm 리드 없음). pitch_deg 는 패스의 값 (3G: 입사 기울기에 더해지는 부호, AutoTeach 가
    패스 1 에서 고른다). calib 이 없으면 대칭 최악값 overscan + standoff·tan(|pitch|+|tilt|).
    plate_top_measured_m 은 올리기만, plate_plane 은 job_base.plate_plane_measured 와 raise-only 결합 (S8).
    progression_sign: R_ref (패스에서 실제로 쓴 플랜지 회전, 3×3) 와 calib 이 있으면 ±1 중 그 자세에 가까운 쪽 — 패스와 **같은 물리 툴 자세**
    (그래야 패스의 pitch_deg 값이 같은 물리 피치다; 거친 단계 진행 부호를 손목 회전 최소로 고른 2026-10-02 이후 job_base 의 부호와
    다를 수 있다 — 다르면 3G 에서 피치가 입사 기울기를 상쇄해 아래 끝 착점이 모자랐다). 없으면 job_base 의 값."""
    if not fit.ok:
        raise ValueError(f"pass fit incomplete: flags={fit.flags}")
    c, u, n = np.asarray(fit.c, dtype=float), unit(fit.u), unit(fit.n)
    if u_SE is not None and float(u @ unit(u_SE)) < 0.0:
        u = -u
    (P1, o1), (P2, o2) = _fit_end_points(fit)
    ed_ = (fit.diag or {}).get("ends") or {}
    i1, i2 = (bool((ed_.get(w_) or {}).get("inferred", False)) for w_ in ("first", "last"))   # 가려진 끝의 추정 (accept_inferred_end)
    if float((P1 - c) @ u) > float((P2 - c) @ u):
        (P1, o1, i1), (P2, o2, i2) = (P2, o2, i2), (P1, o1, i1)
    s_a, s_b = float((P1 - c) @ u), float((P2 - c) @ u)
    info: dict = {"ends": {"start": ("inferred" if i1 else "observed") if o1 else "not_observed",
                           "end": ("inferred" if i2 else "observed") if o2 else "not_observed"}}
    if not o1 and rough_start is not None:
        s_a = float((np.asarray(rough_start, dtype=float) - c) @ u)
        info["ends"]["start"] = "rough_projected"
    if not o2 and rough_end is not None:
        s_b = float((np.asarray(rough_end, dtype=float) - c) @ u)
        info["ends"]["end"] = "rough_projected"
    if s_b <= s_a:
        raise ValueError(f"refined extent is empty: s_a={s_a:.4f} s_b={s_b:.4f}")
    start = c + u * s_a
    end = c + u * s_b
    fm = fit.feature_medians or {}
    if _fin(fm.get("depth_mm")):
        depth_m = float(fm["depth_mm"]) * 1e-3
    elif spec is not None:
        depth_m = float(spec.groove_depth_mm) * 1e-3
    else:
        depth_m = 0.0
    kw = job_to_dict(job_base)
    pt = job_base.plate_top_measured_m
    if plate_top_m is not None:
        pt = float(plate_top_m) if pt is None else max(float(pt), float(plate_top_m))
    kw.update(start=start.tolist(), end=end.tolist(), surface_normal=n.tolist(), standoff_m=float(standoff_plate_m) + depth_m,
              pitch_deg=float(pitch_deg), lead_in_m=0.001, lead_out_m=0.001, plate_top_measured_m=pt,
              plate_plane_measured=merge_plate_plane(job_base.plate_plane_measured, plate_plane),
              job_id=job_id or job_base.job_id)
    vertical = job_base.resolved_orientation() == "vertical_3G"
    if vertical:
        kw["vertical_direction"] = "down" if float(job_base.height(start)) >= float(job_base.height(end)) else "up"
    kw["torch_lateral_m"] = tuple(kw["torch_lateral_m"])
    if R_ref is not None and calib is not None:
        rots = {}
        for sign in (1, -1):
            try:
                cand = ScanJob(**{**kw, "progression_sign": sign})
                cand.validate()
                rots[sign] = _rot_angle_deg(np.asarray(R_ref, dtype=float)[:3, :3], sensor_frames(cand, calib).T_base_flange(0.0)[:3, :3])
            except JobError:
                rots[sign] = float("inf")
        sign = min(rots, key=lambda k_: (rots[k_], -k_))
        kw["progression_sign"] = sign
        info["progression_sign"] = sign
        info["rotation_from_pass_deg"] = rots[sign]
    job = ScanJob(**kw)
    job.validate()
    lead_info: dict = {"overscan_m": float(overscan_m)}
    short: list = []
    if calib is not None:
        floor_h = sensor_frames(job, calib).floor_h if vertical else None
        ov = pass_overscan(job, calib, overscan_m, top_offset_m=depth_m, floor_h=floor_h)
        lead_in = max(-ov["s_from"], 0.001)
        lead_out = max(ov["s_to"] - job.length_m, 0.001)
        lead_info.update(hit_offset_m=ov["hit_offset_m"], source="pass_overscan (laser hit point)")
        if vertical:
            # 3G (R4): 광학 원점은 아래 점에서 멈춘다/출발한다 (그 너머 리드는 표현할 수 없다) — 아래 끝 착점 오버스캔 = 구조적 값
            # ('down': 착점 L + δ → δ, 'up': 착점 0 + δ → −δ). 필요한 리드 (바닥 규칙 포함) 가 > 0 이면 착점이 계획만큼 넘지 못한다.
            lower_last = kw["vertical_direction"] == "down"
            delta = float(ov["hit_offset_m"])
            achieved = delta if lower_last else -delta
            need = (ov["s_to"] - job.length_m) if lower_last else -ov["s_from"]
            w_ = "last" if lower_last else "first"
            lead_info["lower_end"] = {"end": "end" if lower_last else "start", "hit_overscan_m": achieved,
                                      "planned_m": float(overscan_m), "lead_needed_m": max(0.0, float(need)),
                                      "floor_h": floor_h, "floor_limited": w_ in ov["floor_limited"]}
            if achieved < float(overscan_m) - 1e-6:
                short.append({"end": "end" if lower_last else "start", "which": "lower (3G)", "hit_overscan_m": achieved,
                              "planned_m": float(overscan_m),
                              "why": "3G scan stops at the lower point (R4); the laser hit offset does not reach beyond the lower "
                                     "end - pitch sign against the incidence tilt?"})
    else:
        lead_in = lead_out = float(overscan_m) + float(standoff_plate_m) * math.tan(
            math.radians(abs(float(pitch_deg)) + abs(float(job.incidence_tilt_deg))))
        lead_info["source"] = "symmetric worst case (no calib)"
    if vertical:
        # 3G: 리드는 위 점 바깥쪽에만 (scan_s_range); 아래 점 쪽 리드 값은 쓰이지 않는다 (ScanJob 은 > 0 을 요구: 0.001 자리표시자)
        if kw["vertical_direction"] == "down":
            lead_out = 0.001
        else:
            lead_in = 0.001
    job.lead_in_m, job.lead_out_m = float(lead_in), float(lead_out)
    job.validate()
    if short:
        info["short_overscan"] = short
    info.update({"s_a": s_a, "s_b": s_b, "length_m": s_b - s_a, "depth_m": depth_m, "standoff_plate_m": float(standoff_plate_m),
                 "lead_in_m": job.lead_in_m, "lead_out_m": job.lead_out_m, "lead": lead_info, "root_start": start,
                 "root_end": end, "u": u, "n": n})
    if top_polyline is not None and root_polyline is not None:
        Pt = np.asarray(top_polyline, dtype=float).reshape(-1, 3)
        Pr = np.asarray(root_polyline, dtype=float).reshape(-1, 3)
        ok = np.all(np.isfinite(Pt), axis=1) & np.all(np.isfinite(Pr), axis=1)
        if ok.any():
            off = np.median(Pt[ok] - Pr[ok], axis=0)
            info["top_offset"] = off
            info["top_start"] = start + off
            info["top_end"] = end + off
    return job, info


@dataclass
class SeamLine:
    """심 직선 상태 (base, m): 루트(그루브 바닥) 직선 c + u·s, u = S→E 단위, n = 판 법선(센서 쪽), 끝 s_S < s_E.

    obs_S/obs_E: 그 끝이 이 직선을 만든 패스에서 **관측**됐는가. how_S/how_E: 다음 패스 계획에 쓰는 끝의 출처
    ('dwell' | 'observed' | 'observed_prev' | 'design_length' | 'symmetric' (+ '_capped')). k: 0 = 두 드웰의 L1, k = 패스 k 뒤.
    spacing_m: 그 패스의 착점 간격, hit_range: 그 패스가 착점으로 덮은 s 범위 (S→E 좌표)."""
    c: np.ndarray
    u: np.ndarray
    n: np.ndarray
    s_S: float
    s_E: float
    obs_S: bool = False
    obs_E: bool = False
    how_S: str = "dwell"
    how_E: str = "dwell"
    k: int = 0
    spacing_m: float = float("nan")
    hit_range: tuple = (float("nan"), float("nan"))
    direction: Optional[str] = None
    line_rms_m: float = float("nan")       # 그 패스의 루트 직선 잔차 (프로파일 하나의 위치 σ) — 일관성 판정 (k·σ)
    plane_rms_m: float = float("nan")      # 그 패스의 판 평면 잔차
    sigma_S_m: float = float("nan")        # 그 패스가 잰 끝 위치 σ (identify_seam_ends; 없으면 착점 간격/√12)
    sigma_E_m: float = float("nan")
    range_mm: float = float("nan")         # 그 패스의 센서 거리 (원시 z 중앙값) — 양자화 하한 dX(z)/√12 의 z

    @property
    def S(self) -> np.ndarray:
        return np.asarray(self.c, dtype=float) + unit(self.u) * float(self.s_S)

    @property
    def E(self) -> np.ndarray:
        return np.asarray(self.c, dtype=float) + unit(self.u) * float(self.s_E)

    @property
    def length_m(self) -> float:
        return float(self.s_E - self.s_S)

    def s_of(self, P) -> float:
        return float((np.asarray(P, dtype=float) - np.asarray(self.c, dtype=float)) @ unit(self.u))

    def perp_m(self, P) -> float:
        rel = np.asarray(P, dtype=float) - np.asarray(self.c, dtype=float)
        u = unit(self.u)
        return float(np.linalg.norm(rel - u * float(rel @ u)))

    def to_dict(self) -> dict:
        return _plain({"k": self.k, "c": self.c, "u": self.u, "n": self.n, "s_S": self.s_S, "s_E": self.s_E, "S": self.S,
                       "E": self.E, "length_m": self.length_m, "obs_S": self.obs_S, "obs_E": self.obs_E, "how_S": self.how_S,
                       "how_E": self.how_E, "spacing_m": self.spacing_m, "hit_range": list(self.hit_range),
                       "direction": self.direction, "line_rms_m": self.line_rms_m, "plane_rms_m": self.plane_rms_m,
                       "sigma_S_m": self.sigma_S_m, "sigma_E_m": self.sigma_E_m, "range_mm": self.range_mm})


@dataclass
class PassPlan:
    """다음 패스의 범위 (S→E 직선 좌표). ov_S/ov_E = 그 끝에서 착점이 목표를 지나가는 거리 (0 = 목표가 판 안쪽 점이라 오버스캔 없음:
    패스 1 의 보정된 E, 대비 경로 패스 2 의 패스 1 경계). how_*: 목표의 근거 (로그·테스트)."""
    direction: str
    s_S: float
    s_E: float
    ov_S: float
    ov_E: float
    how_S: str = ""
    how_E: str = ""
    rule: str = ""
    tilt_deg: Optional[float] = None       # 적응형 재스캔: 이 패스의 입사 기울기·피치 (None = 기본 규칙)
    pitch_deg: Optional[float] = None
    torch_above: Optional[bool] = None     # 적응형 재스캔 (3G): 토치 쪽 (None = 지금 자세에 가까운 쪽)

    @property
    def first(self) -> str:
        return "E" if self.direction == "E->S" else "S"

    @property
    def last(self) -> str:
        return "S" if self.direction == "E->S" else "E"

    def to_dict(self) -> dict:
        return _plain(dataclasses.asdict(self))


def plan_first_pass(line: SeamLine, *, design_length_m: Optional[float], overscan_m: float, lower_end_first: bool = False) -> PassPlan:
    """패스 1 (사용자 2026-10-01, 거친 S/E 는 일부러 판 끝보다 조금 안쪽): L_SE = 보정된 S·E 루트 중심 간 거리 (u1 방향),
    잔여 R = max(0, L_design − L_SE). 패스 1 = 보정된 E (로봇이 있는 곳, 판 안쪽 → 오버스캔 없음) → S_ext = 보정된 S 를 E 반대쪽으로
    R 연장 (+ 착점 오버스캔 overscan_m) — 목적: S 판 끝 관측. S 끝은 보정된 S 바깥 0..R 어딘가 (두 착점이 시편 안) 이므로 R 이 그 끝이
    있을 수 있는 가장 먼 곳이다 (비율·상한 같은 설정 없음).
    lower_end_first (3G, 사용자 2026-10-05): E = 아래 점 (거친 단계가 위 → 아래), 패스 1 은 아래 → 위. E 쪽도 R 만큼 연장 + 착점
    오버스캔을 목표로 한다 — 실제로는 바닥 (대략 아래 점; 광학 원점·토치 팁 ≥) 이 허락하는 데까지 (_pass_overscan), 패스 뒤 아래 끝이
    보였으면 그 끝이 바닥이 된다 (_update_3g_floor)."""
    L_SE = float(line.s_E - line.s_S)
    L = float(design_length_m) if (design_length_m is not None and float(design_length_m) > 0.0) else None
    R = max(0.0, L - L_SE) if L is not None else 0.0
    ext = R
    how_S = f"S_ext = corrected S - R ({ext * 1e3:.1f} mm)"
    if lower_end_first:
        ext_E = R
        return PassPlan(direction="E->S", s_S=float(line.s_S) - ext, s_E=float(line.s_E) + ext_E, ov_S=float(overscan_m),
                        ov_E=float(overscan_m), how_S=how_S,
                        how_E=f"3G lower end: corrected E + R ({ext_E * 1e3:.1f} mm) + overscan, as far as the floor (lower rough point) allows",
                        rule="first_3g_bottom_up")
    return PassPlan(direction="E->S", s_S=float(line.s_S) - ext, s_E=float(line.s_E), ov_S=float(overscan_m), ov_E=0.0,
                    how_S=how_S, how_E="corrected E (inside the plate, no overscan)", rule="first")




def cap_lower_target(plan: PassPlan, line: SeamLine, up, table_z_m: Optional[float], *, below_m: float,
                     measured: bool = True) -> Optional[str]:
    """3G 패스 1 (아래 → 위) 의 아래 (E) 목표 착점 s_E + ov_E 를 지지면 높이 − below_m (= overscan: 착점이 끝을 지나가는 거리) 이 되는
    직선 좌표에서 자른다 (제자리). 시편 아래 끝은 잰 지지면보다 아래일 수 없다 (2026-10-06 g3_10: 그보다 아래를 목표로 해 레이저가 판 앞
    테이블만 비췄다). measured = 지지면 probe 가 지지면을 직접 쟀다 — 그때만 언제나 자른다 (지지면이 아래 대략 점보다 판 두께/2 넘게 위면
    probe 가 이미 멈췄다). 지지면을 못 보고 보인 판의 가장 낮은 점을 상한으로만 쓰는 경우 ('bound') 는 시편이 그 아래로 이어질 수 있어
    자르지 않는다 (툴 여유에만 쓴다). 예전 '대략 점 + 10 mm' 예외는 옛 20–30 mm 오버스캔에 맞춘 값이라 3.3 mm 오버스캔에서는 g3_17 같은
    배치를 자르지 않았다 (감사 2026-10-08). 자르면 메시지."""
    if table_z_m is None:
        return None
    if not measured:
        return (f"pass 1: support {float(table_z_m):.3f} m is only a bound (lowest plate point seen) - the lower target is not capped "
                "by it (the specimen may continue below; the bound limits the tool only)")
    upv = unit(np.asarray(up, dtype=float))
    u = unit(line.u)
    du = float(upv @ u)
    if du > -1e-6:                                         # E 쪽이 아래가 아니다
        return None
    s_lim = (float(table_z_m) - float(below_m) - float(upv @ np.asarray(line.c, dtype=float))) / du
    if float(plan.s_E) + float(plan.ov_E) <= s_lim + 1e-9:
        return None
    old = float(plan.s_E) + float(plan.ov_E)
    plan.s_E = float(min(plan.s_E, s_lim))
    plan.ov_E = float(max(0.0, s_lim - plan.s_E))
    plan.how_E = (plan.how_E + f"; capped at the measured support - {below_m * 1e3:.1f} mm (the specimen stands on it: the laser hit goes "
                  f"no lower than {float(table_z_m) - float(below_m):.4f} m)")
    return (f"pass 1: lower target capped {(old - s_lim) * 1e3:.1f} mm up - the laser hit stops at the measured support "
            f"{float(table_z_m):.3f} m - {below_m * 1e3:.1f} mm (the specimen stands on it; further down the laser would only see the "
            "support in front of the plate)")


def plan_next_pass(k_done: int, direction_done: str, line: SeamLine, known_S: Optional[float], known_E: Optional[float], *,
                   design_length_m: Optional[float], hit_range: tuple, overscan_m: float) -> PassPlan:
    """패스 k_done 뒤의 다음 패스 (로봇이 있는 끝에서 반대 방향, 지그재그). known_S/known_E: 관측된 끝 (이번 또는 이전 패스, 이 직선
    좌표; None = 미관측). hit_range: 직전 패스 착점 범위 (S→E 좌표).
      * 패스 1 뒤 S 미관측 (대비 경로): 그루브는 적어도 패스 1 경계 s_min 까지 → E 끝 ≤ s_min + L. 패스 2 = s_min (경계, 오버스캔 없음)
        → E_ext = s_min + L (+ 오버스캔) — E 끝 관측 보장 (상한·포락선 조건). 다음 패스가 관측된 E − L 까지 전체 길이.
      * 양끝 관측 → 전체 길이 [S, E], 양쪽 오버스캔 (게이트가 정렬/데이터로 실패해도 반대 방향으로 전체 길이 반복).
      * 한 끝 관측 → 다른 끝 = 관측 끝 ± L (그 점이 본 그루브 안쪽이면 본 끝 밖으로 overscan_m), 양쪽 오버스캔 — 패스 1 뒤 S 관측이면 이것이
        E_target = S_end + L 의 전체 길이 패스 (보통 마지막 패스).
      * 둘 다 미관측 → 대칭 연장 (스캔 길이 ≥ L).
    미관측 끝 목표는 설계 길이가 정하는 기하 그대로 (상한 없음 — 도달은 포락선·3G 바닥 규칙이 본다)."""
    nxt = "S->E" if direction_done == "E->S" else "E->S"
    L = float(design_length_m) if (design_length_m is not None and float(design_length_m) > 0.0) else None
    ov = float(overscan_m)
    me = ov
    h_lo, h_hi = (float(hit_range[0]), float(hit_range[1])) if all(_fin(v) for v in hit_range) else (float(line.s_S), float(line.s_E))
    if k_done == 1 and known_S is None and known_E is None and direction_done == "E->S":
        s_S = h_lo
        s_E = (h_lo + L) if L is not None else float(line.s_E) + me
        return PassPlan(direction=nxt, s_S=s_S, s_E=s_E, ov_S=0.0, ov_E=ov, how_S="pass-1 boundary s_min (inside, no overscan)",
                        how_E="fallback E_ext = s_min + L_design", rule="fallback")
    if known_S is not None and known_E is not None:
        return PassPlan(direction=nxt, s_S=float(known_S), s_E=float(known_E), ov_S=ov, ov_E=ov, how_S="observed",
                        how_E="observed", rule="full_length")
    if known_S is not None:
        tgt = (float(known_S) + L) if L is not None else float(line.s_E) + me
        if tgt <= float(line.s_E):                      # 설계 길이 끝이 본 그루브 안쪽 (시편이 설계보다 김?): 본 끝 밖으로 진전
            tgt = float(line.s_E) + me
        s_E = tgt
        return PassPlan(direction=nxt, s_S=float(known_S), s_E=s_E, ov_S=ov, ov_E=ov, how_S="observed",
                        how_E=("observed S + L_design" if L is not None else "extend"),
                        rule="design_length")
    if known_E is not None:
        tgt = (float(known_E) - L) if L is not None else float(line.s_S) - me
        if tgt >= float(line.s_S):
            tgt = float(line.s_S) - me
        s_S = tgt
        return PassPlan(direction=nxt, s_S=s_S, s_E=float(known_E), ov_S=ov, ov_E=ov,
                        how_S=("observed E - L_design" if L is not None else "extend"),
                        how_E="observed", rule="design_length")
    ext = max((L - (float(line.s_E) - float(line.s_S))) / 2.0, me) if L is not None else me
    s_S = float(line.s_S) - ext
    s_E = float(line.s_E) + ext
    return PassPlan(direction=nxt, s_S=s_S, s_E=s_E, ov_S=ov, ov_E=ov, how_S="symmetric", how_E="symmetric",
                    rule="symmetric")


def _end_sigma_m(line: SeamLine, X: str) -> float:
    """그 패스가 잰 끝 X 의 위치 σ [m] = max(identify_seam_ends 가 잰 σ, 착점 간격/√12) — 둘 다 모르면 NaN."""
    cands = [v for v in (getattr(line, f"sigma_{X}_m"), _ms.end_sigma_m(line.spacing_m)) if _fin(v)]
    return max(cands) if cands else float("nan")


def line_consistency(prev: SeamLine, cur: SeamLine, *, k: float = _ms.K_DEFAULT, aim_tol_m: Optional[float] = None) -> tuple:
    """연속 패스 일관성/수렴 (진행 방향 무관) — 기준은 두 패스가 잰 잡음에서 (scan_core.measure 원칙 1, 설정 상수 없음):
      σ_위치 = √(σ₁² + σ₂²), σᵢ = 그 패스 루트 직선 잔차 (프로파일 하나의 위치 σ, 하한 센서 양자화) — 이전 직선의 S/E 점이 현재
      직선에서 k·σ_위치 안; 직선 방향 각 ≤ atan(k·σ_위치 / 길이) (그 길이 끝에서 k·σ 만큼 비키는 각), 판 법선은 같은 식에 판 평면
      잔차; 두 패스가 **모두 관측한** 끝의 위치 차 ≤ k·√(σ_끝₁² + σ_끝₂²), σ_끝 = max(그 패스가 잰 끝 σ, 착점 간격/√12) — σ_끝을 모르면
      끝 비교는 하지 않는다. 양자화 하한은 각 패스의 센서 거리 (range_mm; 모르면 유효 창 가운데). 반환 (ok, 이유 문자열, 수치 dict).
    aim_tol_m: 용접 경로 겨냥점 허용치 (job qc_targets root_position_mm.tol, 사용자 2026-10-07: 1 mm) — 위치·방향·끝 허용의 바닥.
      두 패스가 용접에 필요한 정확도 안에서 같으면 수렴이다 (2026-10-08 1G 실기 g1_b: 패스마다 0.3–0.6 mm 안에서 같았는데 측정 3σ
      (0.36–0.6 mm) 만으로 판정해 패스 예산 4 를 다 썼다). 판 법선 각은 겨냥 위치가 아니라 그대로 측정 잡음."""
    def sig(line, v):
        q = _ms.quant_sigma_mm(line.range_mm) * 1e-3
        return max(float(v), q) if _fin(v) else q

    sig_end = _end_sigma_m
    sp_pos = math.hypot(sig(prev, prev.line_rms_m), sig(cur, cur.line_rms_m))
    sp_pl = math.hypot(sig(prev, prev.plane_rms_m), sig(cur, cur.plane_rms_m))
    span = max(abs(cur.length_m), abs(prev.length_m), 1e-6)
    floor = float(aim_tol_m) if (aim_tol_m is not None and _fin(aim_tol_m) and float(aim_tol_m) > 0.0) else 0.0
    tol_pos = max(float(k) * sp_pos, floor)
    tol_u = math.degrees(math.atan2(tol_pos, span))
    tol_n = math.degrees(math.atan2(float(k) * sp_pl, span))
    du = _angle_deg(prev.u, cur.u)
    dn = _angle_deg(prev.n, cur.n)
    dpos = max(cur.perp_m(prev.S), cur.perp_m(prev.E))
    m = {"du_deg": du, "dn_deg": dn, "dpos_mm": dpos * 1e3, "tol_pos_mm": tol_pos * 1e3, "tol_u_deg": tol_u, "tol_n_deg": tol_n,
         "k": float(k)}
    good = du <= tol_u and dn <= tol_n and dpos <= tol_pos
    parts = [f"d(u) {du:.3f} deg", f"d(n) {dn:.3f} deg", f"line offset {dpos * 1e3:.2f} mm"]
    end_txt = []
    for X in ("S", "E"):
        if getattr(prev, f"obs_{X}") and getattr(cur, f"obs_{X}"):
            end_tol = max(float(k) * math.hypot(sig_end(prev, X), sig_end(cur, X)), floor)
            d_end = abs(cur.s_of(getattr(prev, X)) - float(getattr(cur, f"s_{X}")))
            m[f"d{X}_mm"] = d_end * 1e3
            m[f"end_tol_{X}_mm"] = end_tol * 1e3
            if _fin(end_tol):
                parts.append(f"|d{X}| {d_end * 1e3:.2f} mm")
                end_txt.append(f"{X} {end_tol * 1e3:.2f} mm")
                good = good and d_end <= end_tol
            else:
                parts.append(f"|d{X}| {d_end * 1e3:.2f} mm (end sigma unknown - not compared)")
    m["aim_tol_mm"] = floor * 1e3
    why = ("consistent: " if good else "not consistent: ") + ", ".join(parts) + \
        (f" (max of k {float(k):g} x measured sigma and the weld aim tolerance {floor * 1e3:.2f} mm: line {tol_pos * 1e3:.2f} mm / "
         f"u {tol_u:.3f} deg / n {tol_n:.3f} deg" + (f", ends {' / '.join(end_txt)}" if end_txt else "") + ")")
    return bool(good), why, m


PASS_DIRECTIONS = ("S->E", "E->S")


def direction_consistency(history: list, *, k: float = _ms.K_DEFAULT, aim_tol_m: Optional[float] = None,
                          postures: Optional[list] = None) -> dict:
    """방향을 가른 수렴 (P8, 1G 실기 2026-10-08 b3 t03: 같은 방향 패스끼리는 0.24 / 0.01 mm 로 같은데 반대 방향끼리 S 끝이 1.28–1.51 mm
    달라 연속 비교 (line_consistency) 가 영원히 실패 — 패스 3·4 를 더 돌려도 수렴할 수 없었다). 연속 패스 (반대 방향) 가 일관되지 않을
    때 호출한다. 문턱은 line_consistency 와 같다 (두 패스가 잰 끝 σ 의 k 배와 용접 조준 허용치 aim_tol_m 중 큰 값) — 새 상수 없음.

    history: 비교 가능한 패스 직선 (같은 자세 사슬, 시간 순, 마지막 = 이번 패스). postures: 각 패스의 자세 키 (주면 쓰는 패스가 모두 같은
    자세여야 한다 — 자세 간 오프셋은 방향 편향이 아니다).
      1. 같은 방향 반복도: 이번 패스와 같은 방향의 가장 최근 이전 패스 j → line_consistency(L_j, 이번) (직선 위치·방향·법선·두 패스가 본
         끝). 실패 → 'not_repeatable' (실제 불일치: 한 패스 더, 예전 규칙 그대로).
      2. 반대 방향의 직선 부분: 창 (j..이번) 의 가장 최근 반대 방향 패스와 이번 패스의 직선 위치·방향·법선이 허용 안 (편향 모형은 끝의
         심 방향 위치만 다룬다). 아니면 'line_differs'.
      3. 방향 편향 추정 (창 j..이번, 끝마다 두 방향 모두 관측했을 때): 읽기 x_i = 이번 직선 좌표의 끝 위치, 방향별 평균 m_SE·m_ES,
         편향 b = m_SE − m_ES (S→E 빼기 E→S), 편향 없는 끝 = (m_SE + m_ES)/2. 창 안의 반대 방향 쌍 (j, 반대)·(반대, 이번) 이 각각 잰
         편향의 차이는 같은 방향 차이 그대로라서 1 이 곧 '편향이 일관된다' 이다.
      4. 설명됨: 편향 없는 끝이 두 방향 읽기 모두에서 끝 허용 안 — |b|/2 ≤ max(k·√(σ_SE² + σ_ES²), aim_tol) → 'converged' (방향 차이는
         반복되는 편향이라 패스를 더 돌려도 가까워지지 않는다; 납품 끝 = 편향 없는 평균, 편향은 보고).
         |b|/2 > 허용: 다른 방향도 같은 방향 쌍으로 반복됐으면 'bias_exceeds_tol' (두 방향 모두 확인 — 더 돌려도 같은 편향이 반복되므로
         멈추고 보고), 아직 한 번만 읽었으면 'bias_unconfirmed' (확인 먼저: 한 패스 더).
    반환 {state, ok (= state == 'converged'), why, metrics, ends: {X: {s_combined_m, shift_m, bias_m, ...}}} — s 는 이번 직선 좌표."""
    k = float(k)
    floor = float(aim_tol_m) if (aim_tol_m is not None and _fin(aim_tol_m) and float(aim_tol_m) > 0.0) else 0.0
    out = {"state": "no_same_direction_pass", "ok": False, "why": "", "metrics": {"k": k, "aim_tol_mm": floor * 1e3}, "ends": {}}
    if not history:
        out["why"] = "direction-separated check: no pass"
        return out
    cur = history[-1]
    d = cur.direction
    if d not in PASS_DIRECTIONS:
        out["why"] = f"direction-separated check: pass {cur.k} has no travel direction"
        return out
    j = next((i for i in range(len(history) - 2, -1, -1) if history[i].direction == d), None)
    if j is None:
        out["why"] = (f"direction-separated check: no earlier {d} pass to measure the {d} repeatability yet "
                      "(the opposite-direction difference cannot be told from noise with one pass per direction)")
        return out
    win = list(history[j:])
    opp = next((L for L in reversed(win) if L.direction in PASS_DIRECTIONS and L.direction != d), None)
    if opp is None:
        out["state"] = "no_opposite_reading"
        out["why"] = f"direction-separated check: no opposite-direction pass between pass {history[j].k} and pass {cur.k}"
        return out
    if postures is not None:
        pw = list(postures)[j:]
        if any(p_ != pw[-1] for p_ in pw):
            out["state"] = "posture_differs"
            out["why"] = (f"direction-separated check: passes {history[j].k}..{cur.k} are not in one posture (a posture offset is not a "
                          "direction bias)")
            return out
    rep_ok, rep_why, rep_m = line_consistency(history[j], cur, k=k, aim_tol_m=aim_tol_m)
    _o_ok, _o_why, opp_m = line_consistency(opp, cur, k=k, aim_tol_m=aim_tol_m)
    line_ok = opp_m["du_deg"] <= opp_m["tol_u_deg"] and opp_m["dn_deg"] <= opp_m["tol_n_deg"] and opp_m["dpos_mm"] <= opp_m["tol_pos_mm"]
    # 다른 방향의 같은 방향 쌍 (사슬 전체에서 가장 최근 두 개) — 편향이 두 방향 모두에서 반복되는지 (멈춤 확인)
    o_idx = [i for i, L in enumerate(history) if L.direction == opp.direction]
    other_rep = None
    if len(o_idx) >= 2:
        i1, i2 = o_idx[-2], o_idx[-1]
        same_post = postures is None or all(p_ == list(postures)[-1] for p_ in list(postures)[min(i1, j):])
        if same_post:
            o_ok, _o_rwhy, o_m = line_consistency(history[i1], history[i2], k=k, aim_tol_m=aim_tol_m)
            other_rep = {"passes": [history[i1].k, history[i2].k], "ok": bool(o_ok), "metrics": o_m}
    ends = {}
    u = unit(cur.u)
    n = unit(cur.n)
    for X in ("S", "E"):
        rd = [(L.direction, cur.s_of(getattr(L, X)), float((np.asarray(getattr(L, X), dtype=float) - np.asarray(cur.c, dtype=float)) @ n),
               _end_sigma_m(L, X), L.k) for L in win if getattr(L, f"obs_{X}") and L.direction in PASS_DIRECTIONS]
        se = [r_ for r_ in rd if r_[0] == "S->E"]
        es = [r_ for r_ in rd if r_[0] == "E->S"]
        e = {"passes_SE": [r_[4] for r_ in se], "passes_ES": [r_[4] for r_ in es]}
        if se and es and getattr(cur, f"obs_{X}"):
            m_se = float(np.mean([r_[1] for r_ in se]))
            m_es = float(np.mean([r_[1] for r_ in es]))
            h_se = float(np.mean([r_[2] for r_ in se]))
            h_es = float(np.mean([r_[2] for r_ in es]))
            sg_se = [r_[3] for r_ in se if _fin(r_[3])]
            sg_es = [r_[3] for r_ in es if _fin(r_[3])]
            sp = math.hypot(max(sg_se), max(sg_es)) if (sg_se and sg_es) else float("nan")
            tol = max(k * sp, floor) if _fin(sp) else (floor if floor > 0.0 else float("nan"))
            b = m_se - m_es
            comb = 0.5 * (m_se + m_es)
            e.update(s_SE_m=m_se, s_ES_m=m_es, bias_m=b, bias_normal_m=h_se - h_es, s_combined_m=comb,
                     shift_m=comb - float(getattr(cur, f"s_{X}")), tol_m=tol,
                     explained=bool(_fin(tol) and 0.5 * abs(b) <= tol))
        ends[X] = e
    est = [X for X in ("S", "E") if "bias_m" in ends[X]]
    mets = {"k": k, "aim_tol_mm": floor * 1e3, "pass": cur.k, "direction": d, "same_direction_pass": history[j].k,
            "opposite_pass": opp.k, "repeat_ok": bool(rep_ok), "repeat": rep_m,
            "opposite_line": {q: opp_m[q] for q in ("du_deg", "dn_deg", "dpos_mm", "tol_u_deg", "tol_n_deg", "tol_pos_mm")},
            "opposite_line_ok": bool(line_ok), "other_direction_repeat": other_rep,
            "ends": {X: {(q[:-2] + "_mm" if q.endswith("_m") else q): (v * 1e3 if q.endswith("_m") else v) for q, v in ends[X].items()}
                     for X in ("S", "E")}}
    out.update(metrics=mets, ends=ends)
    rep_txt = f"{d} passes {history[j].k} and {cur.k}: {rep_why}"
    if not rep_ok:
        out["state"] = "not_repeatable"
        out["why"] = f"direction-separated check: not repeatable in one direction ({rep_txt}) - a real disagreement, not a direction bias"
        return out
    if not line_ok:
        out["state"] = "line_differs"
        out["why"] = (f"direction-separated check: {d} repeatable, but the line of the opposite pass {opp.k} differs (offset "
                      f"{opp_m['dpos_mm']:.2f} mm / u {opp_m['du_deg']:.3f} deg / n {opp_m['dn_deg']:.3f} deg vs {opp_m['tol_pos_mm']:.2f} mm / "
                      f"{opp_m['tol_u_deg']:.3f} / {opp_m['tol_n_deg']:.3f} deg) - the end bias model does not cover it")
        return out
    if not est:
        out["state"] = "no_opposite_reading"
        out["why"] = f"direction-separated check: no end observed in both directions within passes {history[j].k}..{cur.k}"
        return out

    def end_txt(X):
        e = ends[X]
        return (f"{X} bias (S->E minus E->S) {e['bias_m'] * 1e3:+.2f} mm along the seam (normal {e['bias_normal_m'] * 1e3:+.2f} mm; "
                f"S->E passes {e['passes_SE']}, E->S passes {e['passes_ES']}), half {0.5 * abs(e['bias_m']) * 1e3:.2f} mm vs end tol "
                f"{e['tol_m'] * 1e3:.2f} mm, bias-free end {e['shift_m'] * 1e3:+.2f} mm from pass {cur.k}")
    txt = "; ".join(end_txt(X) for X in est)
    if all(ends[X]["explained"] for X in est):
        out.update(state="converged", ok=True)
        out["why"] = (f"consistent by direction: {rep_txt.replace('consistent: ', '')} (repeatable); the opposite-direction difference is a "
                      f"repeatable direction bias: {txt} - ends delivered as the bias-free mean of both directions")
        return out
    confirmed = bool(other_rep is not None and other_rep["ok"])
    out["state"] = "bias_exceeds_tol" if confirmed else "bias_unconfirmed"
    out["why"] = (f"direction bias larger than the end tolerance: {txt}; {d} repeatable (passes {history[j].k}, {cur.k})"
                  + (f" and {opp.direction} repeatable (passes {other_rep['passes'][0]}, {other_rep['passes'][1]}) - more passes would "
                     "repeat the same bias" if confirmed else
                     f", {opp.direction} read once - confirm with one more {opp.direction} pass first"))
    return out


def time_offset_estimate(pass_ends: list, line: SeamLine) -> dict:
    """정·역 패스가 같은 끝을 관측했을 때의 시간 오프셋 추정 dt = (s_fwd − s_rev)/(2 v) [s] (보고 전용).

    pass_ends: [{k, direction 'S->E'|'E->S', speed_mps, posture, ends: {S: {observed, point}, E: {...}}}, …] (시간 순). 끝마다 가장
    최근의 정방향(S→E) 관측과, 그와 **같은 자세** (posture = (입사 기울기, 피치, 토치 쪽) — 3G 끝 패스처럼 자세가 다르면 자세 오프셋이
    시간처럼 보인다) 의 가장 최근 역방향(E→S) 관측을 현재 직선 좌표로 옮겨 비교한다. + = 센서 프로파일이 늦게 찍힌다 (진행 방향으로
    밀려 보임: 정방향은 +v·dt, 역방향은 −v·dt — 일정한 오프셋은 한 패스 안에서 양끝을 같이 밀어 길이는 변하지 않는다)."""
    per = {}
    for X in ("S", "E"):
        fwd = next((p for p in reversed(pass_ends) if p.get("direction") == "S->E" and p["ends"][X]["observed"]), None)
        rev = None if fwd is None else next((p for p in reversed(pass_ends) if p.get("direction") == "E->S" and p["ends"][X]["observed"]
                                              and p.get("posture") == fwd.get("posture")), None)
        if fwd is None or rev is None:
            continue
        v = 0.5 * (float(fwd.get("speed_mps", float("nan"))) + float(rev.get("speed_mps", float("nan"))))
        if not (_fin(v) and v > 1e-6):
            continue
        s_f = line.s_of(fwd["ends"][X]["point"])
        s_r = line.s_of(rev["ends"][X]["point"])
        per[X] = {"s_fwd_m": s_f, "s_rev_m": s_r, "dt_s": (s_f - s_r) / (2.0 * v), "v_mps": v, "k_fwd": fwd.get("k"),
                  "k_rev": rev.get("k")}
    if not per:
        return {"dt_s": float("nan"), "per_end": {}, "n_ends": 0,
                "note": "needs a forward (S->E) and a reverse (E->S) pass in the same posture that observed the same end"}
    dt = float(np.mean([d["dt_s"] for d in per.values()]))
    return {"dt_s": dt, "per_end": per, "n_ends": len(per),
            "note": "+ = profiles stamped late (shifted along the travel direction); compare with calib dt_s"}


def pass_extent_summary(fit: PassFit, direction: str, u_SE) -> dict:
    """identify_seam_ends 의 진행 순서 끝(first/last) → 게이트·계획용 dict: {S: {observed, occluded, point, sigma_m, alt_point}, E: {...},
    first_observed, last_observed, occluded: ['first'|'last'], first_alt / last_alt (설계 길이로 본 대안 끝), first_inferred /
    last_inferred (가려진 끝의 추정 = 다른 끝 ± 설계 길이, 다른 끝이 관측됐을 때), staggered, length_mm (용접 경로 끝 사이 — 판 끝이
    엇갈리면 안쪽 판), plate_length_mm (양끝 모두 두 판 끝의 평균 사이 — 설계 길이와 비교하는 판 길이; 시편 엇갈림은 보고 대상), spacing_mm,
    hit_range_SE: [lo, hi] (fit.c 기준 S→E 직선 좌표)}. 끝은 base 점 (S→E 계약, scan_core.seam_frame) — 길이는 끝 점 사이 거리 (예전: 다른
    직선 u_SE 에 투영한 스칼라 차 — u_SE 가 대략점 방향이면 (오프라인 refine) cos ψ 만큼 짧았다; u_SE 는 S/E 이름 붙이기에만)."""
    c = np.asarray(fit.c, dtype=float)
    uSE = unit(u_SE)
    (Pf, of), (Pl, ol) = _fit_end_points(fit)
    ed = (fit.diag or {}).get("ends") or {}
    occl = [w for w in ("first", "last") if (ed.get(w) or {}).get("occluded")]
    first_is = "E" if direction == "E->S" else "S"
    last_is = "S" if first_is == "E" else "E"

    def rec(P, obs, w):
        e = ed.get(w) or {}
        return {"observed": bool(obs), "occluded": bool(e.get("occluded", False)), "point": P,
                "sigma_m": float(e.get("sigma_m", float("nan"))),     # 끝 위치 σ (S→E 계약 검사의 k·σ — scan_core.seam_frame)
                "alt_point": e.get("alt_point"), "sides": e.get("sides"), "stagger_m": e.get("stagger_m")}

    ends = {first_is: rec(Pf, of, "first"), last_is: rec(Pl, ol, "last")}
    hr = (fit.diag or {}).get("hit_range_m", [float("nan"), float("nan")])
    sgn = 1.0 if float(unit(fit.u) @ uSE) >= 0.0 else -1.0
    hr_SE = sorted([sgn * float(hr[0]), sgn * float(hr[1])]) if all(_fin(v) for v in hr) else [float("nan")] * 2
    sp = (fit.diag or {}).get("spacing_m", float("nan"))
    u_fit = unit(fit.u)

    def plates_P(w):
        """판 길이 끝 (두 판 끝의 평균, identify_seam_ends s_plates_m) 의 base 점 (없으면 용접 경로 끝)."""
        e = ed.get(w) or {}
        sp_ = e.get("s_plates_m")
        return np.asarray((c + u_fit * float(sp_)) if _fin(sp_) else (Pf if w == "first" else Pl), dtype=float)
    out = {"S": ends["S"], "E": ends["E"], "first_observed": bool(of), "last_observed": bool(ol), "occluded": occl,
           "first_alt": (ed.get("first") or {}).get("alt_point"), "last_alt": (ed.get("last") or {}).get("alt_point"),
           "first_inferred": None, "last_inferred": None, "staggered": "plates_staggered" in fit.flags,
           "length_mm": float(np.linalg.norm(np.asarray(ends["E"]["point"], dtype=float)
                                             - np.asarray(ends["S"]["point"], dtype=float))) * 1e3,
           "plate_length_mm": float(np.linalg.norm(plates_P("last") - plates_P("first"))) * 1e3,
           "spacing_mm": float(sp) * 1e3 if _fin(sp) else float("nan"),
           "hit_range_SE": hr_SE,
           "first_sigma_mm": float((ed.get("first") or {}).get("sigma_m", float("nan"))) * 1e3,
           "last_sigma_mm": float((ed.get("last") or {}).get("sigma_m", float("nan"))) * 1e3}
    for w, other_obs in (("first", ol), ("last", of)):
        if w in occl and other_obs and (ed.get(w) or {}).get("alt_point") is not None:
            out[f"{w}_inferred"] = ed[w]["alt_point"]
    out[first_is]["inferred_point"] = out["first_inferred"]
    out[last_is]["inferred_point"] = out["last_inferred"]
    return out


def _fmtf(v, fmt: str = "+.3f") -> str:
    """로그용 숫자 (NaN/None → 'nan')."""
    return format(float(v), fmt) if _fin(v) else "nan"


SPECIMEN_KEYS = ("root_gap_mm", "depth_mm", "top_opening_mm", "bevel_angle_deg", "hi_lo_mm")


def specimen_report(fit: PassFit, verdict, extent: Optional[dict], k: int, direction: str, spec: Optional[GrooveSpec]) -> dict:
    """시편 대비 설계 보고 (사용자 2026-10-02: 실제 시편의 치수 변화·시공 오차는 정확히 재서 보고할 대상 — 판정 아님). 패스의
    fit.diag['specimen_bins'] (정련 심 직선 좌표 구간) 를 **S 판 끝에서 S→E 로 잰 거리** [mm] 로 바꾸고, 판정 (verdict.specimen:
    중앙값·구간 범위 vs 설계 ± tol) 과 측정 잡음을 함께 담는다. extent (양 끝 점) 가 없으면 from_S_mm 는 None."""
    fm = dict(fit.feature_medians or {})
    bins = list((fit.diag or {}).get("specimen_bins") or [])
    frame = None
    if extent is not None and np.all(np.isfinite(fit.c)) and np.all(np.isfinite(fit.u)):
        try:
            frame = _seam_frame.frame_from_extent(extent)            # S→E 계약: 위치 = S 에서 unit(E − S) 로 잰 거리
        except _seam_frame.SeamFrameError:
            frame = None
    c_fit = np.asarray(fit.c, dtype=float)
    u_fit = unit(fit.u) if frame is not None else None
    rows = []
    for b in bins:
        row = {"n": int(b.get("n", 0)), **{k_: b.get(k_, float("nan")) for k_ in SPECIMEN_KEYS}}
        if frame is not None:                                        # 구간 경계 (패스 직선 좌표) → base 점 → S 에서 잰 거리
            a_, b_ = sorted((frame.s(c_fit + u_fit * float(b["s_from_m"])) * 1e3, frame.s(c_fit + u_fit * float(b["s_to_m"])) * 1e3))
            row.update(from_S_mm=a_, to_S_mm=b_)
        else:
            row.update(from_S_mm=None, to_S_mm=None)
        rows.append(row)
    if frame is not None:
        rows.sort(key=lambda r_: r_["from_S_mm"])
    design = None
    if spec is not None:
        design = {"root_gap_mm": spec.root_gap_mm, "depth_mm": spec.groove_depth_mm, "top_opening_mm": spec.top_opening_mm,
                  "bevel_angle_deg": spec.bevel_angle_deg, "weld_length_mm": spec.weld_length_mm}
    return _jsonable({
        "pass": int(k), "direction": direction, "scan_status": verdict.status, "scan_failed": list(verdict.failed),
        "note": "specimen vs design is a report, not a scan failure (construction deviations are measured); the scan verdict is "
                "measurement quality (ends, length, floor/root edges, profile-to-profile noise, gross deviation)",
        "design": design, "summary": dict(verdict.specimen or {}),
        "length_mm": extent.get("length_mm") if extent is not None else None,
        "plate_length_mm": extent.get("plate_length_mm") if extent is not None else None,
        "measurement_noise_mm": {k_: fm.get(f"{k_[:-3]}_noise_mm") for k_ in ("root_gap_mm", "depth_mm", "top_opening_mm", "hi_lo_mm")},
        "bin_mm": (float(bins[0]["s_to_m"] - bins[0]["s_from_m"]) * 1e3) if bins else None,
        "position": "from_S_mm / to_S_mm: distance from the observed S plate end along S->E [mm]",
        "bins": rows})


def format_specimen_table(rep: dict) -> list:
    """specimen_report → 콘솔 표 (줄 목록). 설계 ± tol 밖의 구간 값에는 '*'."""
    summ = rep.get("summary") or {}
    noise = rep.get("measurement_noise_mm") or {}
    which = f"pass {rep.get('pass')}" if rep.get("pass") else "offline refine"
    out = [f"specimen vs design ({which}, report only - not a scan failure); bins of {_fmtf(rep.get('bin_mm'), '.0f')} mm "
           f"from the S end; measurement noise gap {_fmtf(noise.get('root_gap_mm'), '.3f')} depth {_fmtf(noise.get('depth_mm'), '.3f')} "
           f"opening {_fmtf(noise.get('top_opening_mm'), '.3f')} mm"]
    cols = (("root_gap_mm", "gap"), ("depth_mm", "depth"), ("top_opening_mm", "opening"), ("bevel_angle_deg", "bevel"),
            ("hi_lo_mm", "hi-lo"))

    def cell(key, v):
        r_ = summ.get(key) or {}
        mark = ""
        if _fin(v) and r_.get("tol") is not None and _fin(r_.get("design")):
            mark = "*" if abs(float(v) - float(r_["design"])) > float(r_["tol"]) + 1e-12 else ""
        elif _fin(v) and key == "hi_lo_mm" and _fin(r_.get("target_max")):
            mark = "*" if abs(float(v)) > float(r_["target_max"]) else ""
        return f"{_fmtf(v, '.2f') + mark:>11s}"

    def mm(x) -> int:
        return int(round(float(x))) + 0                    # -0 → 0

    out.append(f"  {'from S [mm]':>13s}" + "".join(f"{h:>11s}" for _, h in cols) + f"{'n':>6s}")
    for b in rep.get("bins") or []:
        pos = (f"{mm(b['from_S_mm']):6d}-{mm(b['to_S_mm']):<6d}" if b.get("from_S_mm") is not None else f"{'?':>13s}")
        out.append(f"  {pos:>13s}" + "".join(cell(k_, b.get(k_)) for k_, _ in cols) + f"{b.get('n', 0):6d}")
    out.append(f"  {'median':>13s}" + "".join(cell(k_, (summ.get(k_) or {}).get("median")) for k_, _ in cols))
    des = []
    for k_, _ in cols:
        r_ = summ.get(k_) or {}
        if k_ == "hi_lo_mm":
            des.append(f"{('<=' + _fmtf(r_.get('target_max'), '.2f')) if _fin(r_.get('target_max')) else '-':>11s}")
        else:
            tol = r_.get("tol")
            des.append(f"{_fmtf(r_.get('design'), '.2f') + ('' if tol is None else '+-' + format(float(tol), '.1f')):>11s}")
    out.append(f"  {'design':>13s}" + "".join(des))
    dev = []
    for k_, h in cols[:3]:
        r_ = summ.get(k_) or {}
        if r_.get("within") is False or r_.get("along_within") is False:
            lo, hi = r_.get("along_min"), r_.get("along_max")
            dev.append(f"{h} {lo:.2f}..{hi:.2f}" if _fin(lo) and _fin(hi) else f"{h} median {_fmtf(r_.get('median'), '.2f')}")
    out.append("  outside design +- tol (construction deviation, reported): " + (", ".join(dev) if dev else "none"))
    return out


def probe_verdict(dx_x_mm: float, dx_y_mm: float, step_mm: float, sgn: float = 1.0) -> tuple:
    """2축 프로브 판정 (설계 §4.3; 4d P4 — 가설 중간점, 원칙 2). 반환 ('confirmed' | 'flip' | 'inconsistent', 설명).

    X_opt 로 sgn·step 이동하면 그루브 x_c 는 −sgn·step 이어야 한다 (분기 확인: design_match(−sgn·dx_X, step) — step 의 절반 안);
    +sgn·step 이면 광학 yaw 분기가 반대 ('flip'); 그 외 (90° 축 뒤바뀜/eoat_yaw 오류) 'inconsistent'. Y_opt 이동은 |dx_Y| ≤ step/2
    ('Y 가 심을 따라 0' 과 '축이 바뀌어 step' 의 중간점). step = 2·(착점 도착 허용 + k·√2·양자화) 라 절반 띠가 도착 오차와 읽기 정밀을 담는다.
    """
    s = float(step_mm)
    if not (_fin(dx_x_mm) and _fin(dx_y_mm)):
        return "inconsistent", f"probe: groove not measured after a probe move (dx_X={dx_x_mm}, dx_Y={dx_y_mm})"
    if abs(dx_y_mm) > _ms.DESIGN_HALF * s:
        return "inconsistent", (f"probe: {s:.1f} mm along Y_opt moved the groove by {dx_y_mm:+.2f} mm (> {_ms.DESIGN_HALF * s:.2f} mm): "
                                "laser line not perpendicular to Y_opt / axis swap")
    if _ms.design_match(-float(sgn) * float(dx_x_mm), s):
        return "confirmed", (f"probe: X_opt {sgn * s:+.1f} mm -> dx_c {dx_x_mm:+.2f} mm (expected {-sgn * s:+.1f} +-{_ms.DESIGN_HALF * s:.2f}): "
                             "branch confirmed")
    if _ms.design_match(float(sgn) * float(dx_x_mm), s):
        return "flip", f"probe: X_opt {sgn * s:+.1f} mm -> dx_c {dx_x_mm:+.2f} mm (= {sgn * s:+.1f}): optical yaw branch is the other one (flip)"
    return "inconsistent", (f"probe: X_opt {sgn * s:+.1f} mm -> dx_c {dx_x_mm:+.2f} mm, neither {-sgn * s:+.1f} nor {sgn * s:+.1f} "
                            f"(+-{_ms.DESIGN_HALF * s:.2f}): optical axis mapping inconsistent with calib")


def probe_verdict_x(dx_x_mm: float, step_mm: float, sgn: float = 1.0) -> tuple:
    """X 축만의 프로브 판정 (P2: 심 방향은 S 추적 스트림이 재므로 Y_opt 스텝·복귀 이동이 없다) — probe_verdict 에서 Y 조건만 뺀 것.
    반환 ('confirmed' | 'flip' | 'inconsistent', 설명)."""
    s = float(step_mm)
    if not _fin(dx_x_mm):
        return "inconsistent", f"probe: groove not measured after the X probe move (dx_X={dx_x_mm})"
    if _ms.design_match(-float(sgn) * float(dx_x_mm), s):
        return "confirmed", (f"probe: X_opt {sgn * s:+.1f} mm -> dx_c {dx_x_mm:+.2f} mm (expected {-sgn * s:+.1f} +-{_ms.DESIGN_HALF * s:.2f}): "
                             "branch confirmed")
    if _ms.design_match(float(sgn) * float(dx_x_mm), s):
        return "flip", f"probe: X_opt {sgn * s:+.1f} mm -> dx_c {dx_x_mm:+.2f} mm (= {sgn * s:+.1f}): optical yaw branch is the other one (flip)"
    return "inconsistent", (f"probe: X_opt {sgn * s:+.1f} mm -> dx_c {dx_x_mm:+.2f} mm, neither {-sgn * s:+.1f} nor {sgn * s:+.1f} "
                            f"(+-{_ms.DESIGN_HALF * s:.2f}): optical axis mapping inconsistent with calib")


def flip_optical_branch(calib: Calib) -> Calib:
    """광학 yaw 분기 뒤집기: T_flange←opt · Rz(π) (X, Y 반전; 원점·Z 불변; eoat_yaw·툴 목표 불변). T_flange←opt 와 출처만 바꾸고 나머지 필드
    ('g1:' 블록·툴 몸체 …) 는 그대로 (dataclasses.replace — 필드를 옮겨 적으면 새 필드가 빠진다; calib_tf.calib_from_tf 와 같은 이유)."""
    T = np.asarray(calib.T_flange_opt, dtype=float) @ make_transform(rot_z(math.pi), (0.0, 0.0, 0.0))
    return dataclasses.replace(calib, T_flange_opt=T, T_flange_eoat=np.array(calib.T_flange_eoat, dtype=float, copy=True),
                               source=(calib.source + "+branch_flipped(auto_teach probe)").strip("+"))


def confirm_needed(mode: str, dist_m: float, threshold_m: Optional[float] = None, kind: str = "local") -> bool:
    """confirm_move 가 이 이동을 묻는가 (판단 규칙만 — 프롬프트 없음; 묻는 이동에서만 확인 문구용 계획을 미리 하려고 따로 뺐다). mode 'each' =
    모든 이동, 'large' = dist_m > threshold_m 만, 'scan' = kind 가 'scan'·'joint'·'joint_detour' 일 때만, 'auto' = kind 가 'joint_detour'
    (관절 직선이 아닌 우회 관절 이동) 일 때만, 'none' = 없음."""
    if mode not in CONFIRM_MODES:
        raise ValueError(f"confirm mode must be one of {CONFIRM_MODES}, got {mode!r}")
    if mode == "large" and threshold_m is None:
        raise ValueError("confirm mode 'large' needs threshold_m")
    return not (mode == "none" or (mode == "large" and dist_m <= float(threshold_m)) or (mode == "scan" and kind not in CONFIRM_KINDS_SCAN)
                or (mode == "auto" and kind not in CONFIRM_KINDS_AUTO))


def move_plan_text(pl: Optional[dict], *, chord_m: float = 0.0, rot_deg: float = 0.0, refused: Optional[str] = None) -> str:
    """확인 프롬프트·결정 기록용 이동 설명 = 백엔드가 계획한 것 (2026-10-08 g1_b: 'CONFIRM start_pose: move 1893.0 mm' 는 'large' 판단
    값인 현 929.2 mm + 툴 레버 × 회전 174.2° 였다. 계획 기록의 'path 950 mm' 는 재시간 경로 변수 S — 기록한 시작 자세·목표 관절로 계획을
    다시 풀면 (226 점·S 가 기록과 같은 IK 분기) 플랜지 경로 949.6 mm·회전 176.6°·가장 많이 도는 관절 wrist1 128.4°: 모든 점에서 병진이
    회전 × 반경보다 커 S 가 병진과 같았을 뿐, S 는 일반적으로 병진 이상이다): 어떻게 계획했나 (관절 공간 직선 / MoveIt OMPL 웨이포인트 수 /
    직교 직선), 계획한 플랜지 경로 길이와 회전 (따로 — 섞지 않는다), 계획 시간, 관절 이동이면 가장 많이 도는 관절 (이름·각).
    pl = 백엔드의 계획 요약 (check_joint 의 plan, check_leg 의 plan). 직교 직선의 계획이 없으면 직선의 플랜지 경로 = 현이므로 chord_m·rot_deg
    로 쓰고 시간은 없다 — 왜 없는지는 호출자가 구조로 준다: refused 없음 = 백엔드가 보내기 전에 계획하는 기능이 없다 (RB 제어박스 어댑터:
    제어박스가 보낼 때 계획), refused = 백엔드가 지금 자세에서 계획을 거부·못 했다 (그 사유 — 보낼 때 다시 계획하고 원인이 남아 있으면 거부)."""
    pl = dict(pl or {})

    def num(v, scale: float = 1.0) -> Optional[float]:
        try:
            f = float(v) * scale
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    dur = num(pl.get("duration_s"))
    if pl.get("kind") == "joint":
        route = pl.get("route_text")
        if not route:
            n_wp = pl.get("n_waypoints")
            route = ("joint-space straight line" if pl.get("route") == "straight" else
                     f"MoveIt OMPL ({int(n_wp)} waypoints)" if pl.get("route") == "OMPL" and n_wp else str(pl.get("how") or "joint-space path"))
        tr = num(pl.get("flange_translation_m"), 1e3)
        rot = num(pl.get("flange_rotation_deg"))
        jr = num(pl.get("max_joint_rotation_deg"))
        parts = [f"flange path {tr:.1f} mm" if tr is not None else "flange path not reported by the planner",
                 f"flange rotation {rot:.1f} deg" if rot is not None else "flange rotation not reported by the planner",
                 f"duration {dur:.1f} s" if dur is not None else "duration not reported by the planner"]
        if jr is not None:
            parts.append(f"largest joint rotation {pl.get('max_joint', '?')} {jr:.1f} deg")
        return f"{route}: " + ", ".join(parts)
    if pl:
        tr = num(pl.get("length_m", pl.get("path_length_m")), 1e3)
        rot = num(pl.get("rotation_deg"))
        return (f"Cartesian straight line: flange path {(tr if tr is not None else float(chord_m) * 1e3):.1f} mm, flange rotation "
                f"{(rot if rot is not None else abs(float(rot_deg))):.1f} deg, "
                + (f"duration {dur:.1f} s" if dur is not None else "duration not reported by the planner"))
    geo = f"flange path {float(chord_m) * 1e3:.1f} mm, flange rotation {abs(float(rot_deg)):.1f} deg"
    if refused is None:
        return f"Cartesian straight line (the motion backend plans it only when sent - no duration before this prompt): {geo}"
    return (f"Cartesian straight line, NOT plannable from the current pose before this prompt ({refused or 'no reason given'}) - "
            f"planned again when sent and refused there if the cause remains: {geo}")


def confirm_move(mode: str, name: str, dist_m: float, *, source: Optional[Callable[[], Optional[str]]] = None,
                 log: Optional[Callable[[str], None]] = None, threshold_m: Optional[float] = None,
                 timeout_s: float = 300.0, extra: str = "", kind: str = "local", text: Optional[str] = None) -> bool:
    """이동 확인. mode 'each' = 모든 이동, 'large' = > threshold_m 만 (AutoTeach: 안전 정책 여유 safety.clearance_m — 그 여유를 다 쓸 수
    있는 이동), 'scan' = kind 가 'scan' (스캔 이동) 또는 'joint'/'joint_detour' (관절 공간 이동) 일 때만, 'auto' = 'joint_detour' (관절 직선이
    아닌 우회 관절 이동) 일 때만, 'none' = 없음 (confirm_needed). 'y'/'Y' 만 실행, 그 외 (타임아웃 포함) = 중단.
    kind: 'scan' | 'joint' | 'joint_detour' | 'local' (짧은 국부 이동 — 보정·프로브·드웰 접근·패스 시작).
    text: 프롬프트의 이동 설명 (AutoTeach: 계획 — move_plan_text). 없으면 'move <dist_m> mm' — dist_m 은 판단 값이다 (AutoTeach 에서는 툴
    점의 시작 → 끝 순변위 상한 = 플랜지 현 + 툴 레버 × 회전이라 경로 길이가 아니다).

    source: 답의 출처 `() -> Optional[str]` (폴링 — ROS 서버: ~/confirm_move 서비스, 시험: 주입). 없으면 stdin 한 줄."""
    if not confirm_needed(mode, dist_m, threshold_m, kind):
        return True
    p = log or (lambda s: None)
    what = text if text else f"move {dist_m * 1e3:.1f} mm"
    # 확인은 서비스 ~/confirm_move (std_srvs/SetBool) 하나뿐이다 — 액션 클라이언트는 키를 읽지 않는다 (서버가 전체 서비스 이름과 명령을 덧붙인다)
    p(f"CONFIRM {name}: {what}{(' ' + extra) if extra else ''} -> answer with the service ~/confirm_move (std_srvs/SetBool: "
      f"data true = execute, false = abort; no answer within {timeout_s:.0f} s aborts)")
    key = _read_confirm_key(source, timeout_s)
    ok = key is not None and key.strip().lower() == "y"
    p(f"  confirm {name}: {'YES' if ok else 'NO (' + repr(key) + ')'}")
    return ok


def _read_confirm_key(source, timeout_s: float) -> Optional[str]:
    """확인 답: source 를 timeout_s 동안 폴링 (없으면 None = 중단), source 가 없으면 stdin 한 줄의 첫 글자."""
    if source is not None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            k = source()
            if k:
                return str(k)
            time.sleep(0.01)
        return None
    try:
        return input().strip()[:1] or None
    except (EOFError, OSError):
        return None


def classify_localization(loc: StaticLocalization) -> str:
    """드웰 분류: 'groove' | 'plate_edge' (판 하나만, 프로파일 한쪽 끝까지 표면 없음) | 'plate_only' (연속 판, 그루브 없음) | 'no_surface'.
    'no_surface' = 유효 열이 특징 추출기의 최소 점 수 (판 최소 면 둘의 증거 개수, 이 프로파일의 열 간격 — features.min_points) 보다
    적은 시야 (4d P5, 예전 고정 20 열)."""
    f = loc.features
    if int(loc.diag.get("n_valid", f.n_valid)) < max(3, int(getattr(f, "min_points", 0) or 0)):
        return "no_surface"
    if loc.groove_found:
        return "groove"
    if f.plate_left is None or f.plate_right is None:
        return "plate_edge"
    if not loc.plate_ok:
        return "no_surface"
    return "plate_only"


def search_order(steps_mm, loc: StaticLocalization) -> list:
    """측면 탐색 스텝 순서: 판이 한쪽만 보이면 (판 끝) 보이는 판 쪽(그루브가 있을 쪽)을 먼저."""
    steps = [float(s) for s in steps_mm]
    f = loc.features
    if f.plate_left is not None and f.plate_right is None:
        sign = -1.0
    elif f.plate_right is not None and f.plate_left is None:
        sign = 1.0
    else:
        return steps
    first = [s for s in steps if s * sign > 0]
    rest = [s for s in steps if s * sign <= 0]
    return first + rest


def pass_residuals(result: ScanResult, sf: SeamScanFeatures, fit: Optional[PassFit], u_travel, target_standoff_m: float, *,
                   gravity_axis_base, process: Optional[WeldProcess] = None) -> dict:
    """패스 프로파일별 특징 → 정렬 잔차 (기울기 인식, I2):
    psi_deg = atan(d x_c / d s) (그루브 중심 x 의 진행 드리프트 — 센서 프레임 면내 양, 핸드아이 무관),
    theta_deg (판 vs 진행 피치) = **base 3D 점**에서: fit 이 있으면 atan(−d[n·(o_i − P_top_i)]/ds) (판 법선 방향 standoff 의 진행 드리프트),
        없으면 atan(d[(P_top_i − o_i)·m]/ds), m = 진행선에 수직인 연직 방향 unit(up − (u·up)u) (진행선에서 잰 상부 중심의 연직 오프셋 드리프트 —
        진행 자체의 높이 변화는 빠진다). + = 진행하며 판에 가까워진다 (= 판이 올라온다). 원시 z_c 드리프트는 theta_raw_z_deg 로 진단만
        (제어·게이트에 쓰지 않는다, M-1).
    roll_deg = median tilt, lateral_m = median groove_centre_x (상부 개구 중심; localize_static.lateral_offset_m 과 같은 기준 — 루트 중심은
    일면 개선에서 bevel_run/2 만큼 치우친다; 그 값은 lateral_root_m 으로 보고), standoff_m = median n·(o_i − P_top_i) − target (base 3D, I2;
    판 법선 fit 이 있을 때만).
    fit 없음 + 진행이 수평보다 수직에 가까움 (|u_travel·up| ≥ sin 45°, 원칙 2 중간점 — vertical_3G): 연직 방향이 진행에 거의 평행이라
    피치를 낼 수 없다 — theta NaN, theta_source 'unavailable' (F-C10).
    s = 광학 원점의 u_travel 진행좌표. 기울기를 내는 최소 = 용접 경로 단면 하나 (process.min_profiles 프로파일, 단면 창 길이 — 진단)."""
    proc = process if process is not None else WeldProcess()
    n_min, span_min = int(proc.min_profiles), float(proc.station_window_mm) * 1e-3
    ut = unit(u_travel)
    up = -unit(gravity_axis_base)
    idx = np.asarray(sf.indices, dtype=np.int64)
    ok = np.asarray(sf.ok, dtype=bool)
    o = np.asarray(result.T_base_opt[idx, :3, 3], dtype=float)
    s = (o - o[0]) @ ut
    out = {"psi_deg": float("nan"), "theta_deg": float("nan"), "roll_deg": float("nan"), "lateral_m": float("nan"),
           "standoff_m": float("nan"), "n_ok": int(ok.sum()), "n_profiles": int(idx.size), "x_c_drift_mm_per_m": float("nan"),
           "z_c_drift_mm_per_m": float("nan"), "theta_raw_z_deg": float("nan"), "theta_source": None}
    if ok.sum() >= n_min:
        xc = np.asarray(sf.groove_centre_x_mm, dtype=float)[ok]
        zc = np.asarray(sf.groove_centre_z_mm, dtype=float)[ok]
        ss = s[ok]
        fx = np.isfinite(xc)
        fz = np.isfinite(zc)
        if fx.sum() >= n_min and float(np.ptp(ss[fx])) > span_min:
            kx = _robust_slope(ss[fx], xc[fx])
            out["x_c_drift_mm_per_m"] = kx
            out["psi_deg"] = math.degrees(math.atan(kx * 1e-3))
        if fz.sum() >= n_min and float(np.ptp(ss[fz])) > span_min:
            kz = _robust_slope(ss[fz], zc[fz])
            out["z_c_drift_mm_per_m"] = kz
            out["theta_raw_z_deg"] = math.degrees(math.atan(-kz * 1e-3))     # 진단만 (원시 z 는 높이가 아니다)
        tilt = np.asarray(sf.tilt_deg, dtype=float)[ok]
        tilt = tilt[np.isfinite(tilt)]
        if tilt.size:
            out["roll_deg"] = float(np.median(tilt))
        if fx.any():
            out["lateral_m"] = float(np.median(xc[fx])) * 1e-3
        rx = np.asarray(sf.root_centre_x_mm, dtype=float)[ok]
        rx = rx[np.isfinite(rx)]
        if rx.size:
            out["lateral_root_m"] = float(np.median(rx)) * 1e-3
        # ---- 기울기 인식 theta / standoff: base 3D 상부 중심 점 ----
        P_top, _ = polyline_from_scan_features(result, sf, centre="top")
        sel = ok & np.all(np.isfinite(P_top), axis=1)
        if fit is not None and np.all(np.isfinite(fit.n)):
            n = unit(fit.n)
            d = (o[sel] - P_top[sel]) @ n
            if d.size:
                out["standoff_measured_m"] = float(np.median(d))
                out["standoff_m"] = float(np.median(d)) - float(target_standoff_m)
            if d.size >= n_min and float(np.ptp(s[sel])) > span_min:
                out["theta_deg"] = math.degrees(math.atan(-_robust_slope(s[sel], d)))   # standoff 감소 = 접근 = +
                out["theta_source"] = "plate_normal_standoff_vs_s"
        elif abs(float(ut @ up)) >= math.sin(math.radians(_ms.HALF_TURN_DEG)):
            # F-C10: 수직 쪽 진행(3G) — 판 법선(fit) 없이는 theta 를 낼 수 없다
            out["theta_source"] = "unavailable"
        elif sel.sum() >= n_min and float(np.ptp(s[sel])) > span_min:
            m = unit(up - float(ut @ up) * ut)
            dm = (P_top[sel] - o[sel]) @ m
            out["theta_deg"] = math.degrees(math.atan(_robust_slope(s[sel], dm)))   # 진행선에서 본 판(상부 중심) 상승 = 접근 = +
            out["theta_source"] = "plate_offset_vs_travel"
    return out



def site_pre_laterals(loc_arrival: StaticLocalization, loc_pre: StaticLocalization, loc_final: StaticLocalization, w=None) -> tuple:
    """도착 지점(E/S2) 의 **보정 전** 측면 오프셋 두 가지 [m] (S2):

    sensor : 보정 전 첫 그루브 국소화(loc_pre, 탐색이 있었으면 탐색 스텝 뒤) 의 lateral_offset_m + 그 광축∩판 점이 도착 광축∩판 점에서
             e_lat 방향으로 옮겨 간 양 (탐색 스텝; 탐색이 없으면 0). 센서 x 로 잰 값이다.
    kinematic : 그 지점의 보정(탐색 포함) 이 광축∩판 점을 e_lat 방향으로 옮긴 실측 양 + 보정 뒤 잔차 lateral (loc_final).
             보정 루프는 '측정 lateral = 0' 으로 수렴하므로 (센서 x 스케일·부호와 무관한 고정점) 이 값은 로봇 운동학이 잰 보정 전
             오프셋이다 — 센서 x 스케일 오차/부호 뒤집힘이 있으면 sensor 값·패스 회귀 psi 와 어긋난다.
    e_lat 은 도착 국소화의 lateral_axis_base (없으면 loc_pre 의 것). 판이 없으면 NaN.
    w (P2): 측정 심에 수직인 축 (보정이 w 로만 움직인다) — 주면 두 읽기를 모두 w 방향으로 잰다: 센서 = lat_pre·(e·w) + 탐색 이동·w, 운동학 =
    (A_final − A0)·w + lat_final·(e·w). 레이저 선이 심에 기울어 있으면 (ψ) e 방향 읽기가 cos²ψ 만큼 거짓으로 어긋난다."""
    nan = float("nan")
    e = loc_arrival.lateral_axis_base if np.all(np.isfinite(loc_arrival.lateral_axis_base)) else loc_pre.lateral_axis_base
    A0 = loc_arrival.axis_hit_base
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(A0))):
        A0 = loc_pre.axis_hit_base
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(A0))):
        return nan, nan
    a = e if w is None else unit(np.asarray(w, dtype=float))              # 이동을 재는 축

    def ew(loc):
        le = loc.lateral_axis_base if np.all(np.isfinite(loc.lateral_axis_base)) else e
        return 1.0 if w is None else float(unit(le) @ a)
    lat_sensor = nan
    if _fin(loc_pre.lateral_offset_m) and np.all(np.isfinite(loc_pre.axis_hit_base)):
        lat_sensor = float(loc_pre.lateral_offset_m) * ew(loc_pre) + float((loc_pre.axis_hit_base - A0) @ a)
    lat_kin = nan
    if _fin(loc_final.lateral_offset_m) and np.all(np.isfinite(loc_final.axis_hit_base)):
        lat_kin = float((loc_final.axis_hit_base - A0) @ a) + float(loc_final.lateral_offset_m) * ew(loc_final)
    return lat_sensor, lat_kin


def scale_reading(sensor_m: float, kin_m: float, z_mm, *, k: float, eps_max: float) -> dict:
    """센서 x 스케일 한 번 읽기 (4d P4): 같은 측면 이동을 센서 x 와 로봇 운동학 (FK) 으로 잰 두 값 → eps = (센서 − 운동학)/운동학.
    읽기 정밀 몫 a = k·√2·q(z)/|운동학| (두 위치 차의 양자화, 원칙 1). 'exceeds' = |eps| − a > eps_max (설계 허용에 닿는 스케일);
    |운동학|·eps_max < k·√2·q 이면 이 이동으로는 그 스케일 오차를 볼 수 없다 ('testable' False — 판정에 쓰지 않는다)."""
    out = {"sensor_m": float(sensor_m) if _fin(sensor_m) else float("nan"), "kin_m": float(kin_m) if _fin(kin_m) else float("nan"),
           "eps": float("nan"), "allow": float("nan"), "testable": False, "exceeds": False}
    if not (_fin(sensor_m) and _fin(kin_m) and abs(float(kin_m)) > 0.0):
        return out
    a_abs = float(k) * math.sqrt(2.0) * _ms.quant_sigma_mm(z_mm) * 1e-3
    kin = float(kin_m)
    out["eps"] = (float(sensor_m) - kin) / kin
    out["allow"] = a_abs / abs(kin)
    out["testable"] = bool(abs(kin) * float(eps_max) >= a_abs)
    out["exceeds"] = bool(out["testable"] and abs(out["eps"]) - out["allow"] > float(eps_max))
    return out


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Theil–Sen 기울기 (표본 ≤ 400 점으로 제한)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size > 400:
        sel = np.linspace(0, x.size - 1, 400).astype(int)
        x, y = x[sel], y[sel]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    m = np.triu(np.ones_like(dx, dtype=bool), 1) & (np.abs(dx) > 1e-6)
    if not m.any():
        return float("nan")
    return float(np.median(dy[m] / dx[m]))


def _data_validity(result: ScanResult, sf: SeamScanFeatures) -> dict:
    """충실도 게이트의 하드 데이터 유효성 값 (ok ∧ cruise 프로파일 중앙값): sensor_z_mm = 그루브 상부 중심의 원시 z (센서 교정 창
    판정 — 원시 z 의 허용된 용도), fov_margin_mm = min(그루브 중심 − 유효 프로파일 왼끝, 오른끝 − 그루브 중심) − 개구/2
    (그루브 양옆에 판이 충분히 보이는가)."""
    out = {"sensor_z_mm": float("nan"), "fov_margin_mm": float("nan")}
    sel = sf._align_selection()
    if sel.shape != np.asarray(sf.ok).shape or not sel.any():
        return out
    zc = np.asarray(sf.groove_centre_z_mm, dtype=float)[sel]
    zc = zc[np.isfinite(zc)]
    if zc.size:
        out["sensor_z_mm"] = float(np.median(zc))
    margins = []
    valid = getattr(result, "valid", None)
    for j in np.flatnonzero(sel):
        i = int(sf.indices[j])
        gx = float(sf.groove_centre_x_mm[j])
        op = float(sf.top_opening_mm[j])
        if not (np.isfinite(gx) and np.isfinite(op)):
            continue
        x = np.asarray(result.x_mm[i], dtype=float)
        v = np.isfinite(x) & np.isfinite(np.asarray(result.z_mm[i], dtype=float))
        if valid is not None:
            v &= np.asarray(valid[i], dtype=bool)
        if v.sum() < 2:
            continue
        margins.append(min(gx - float(x[v].min()), float(x[v].max()) - gx) - 0.5 * op)
    if margins:
        out["fov_margin_mm"] = float(np.median(margins))
    return out


def end_zone_lag_m(job: ScanJob, calib: Calib, depth_m: float, plate_normal=None) -> float:
    """판 끝에서 루트가 가려지는 기하 구간 [m] = 깊이 × tan(레이저 평면 ↔ 판 법선 각) — 레이저 면이 진행 방향으로 기울면 그루브
    바닥이 판 상면보다 그만큼 먼저 끝난다 (그 구간은 용접 경로가 판 끝 위치로 외삽한다)."""
    inc = _sq.pass_posture(job, calib, plate_normal=plate_normal)["incidence_deg"]
    return float(max(0.0, float(depth_m)) * math.tan(math.radians(min(abs(float(inc)), 89.0)))) if _fin(inc) else 0.0


def pass_coverage(feats: dict, s_a: float, s_b: float, process: WeldProcess, end_lag_m: float = 0.0) -> dict:
    """측정 충분성 (scan_core.measure 원칙 4, 사용자 2026-10-07): 용접 경로 단면 (station_step 간격, ±station_window/2 창) 마다 유효
    프로파일이 단면 계획의 min_profiles 이상인가 — 그루브 (ok), 바닥 보임, 양쪽 루트 모서리. feats = _features_arrays 출력 또는 병합
    배열 (s_line_m 필요). 평가 범위 = 판 끝에서 end_lag_m (end_zone_lag_m: 기운 레이저 면이 루트를 가리는 기하 구간) 안쪽 ∩ 크루즈
    (재구성이 잰 속도로 판정한 등속 구간, in_cruise) 프로파일이 덮는 범위 — 가속·감속 중 프로파일은 루트가 직선을 벗어나 쓰이지 않는다
    (실기 재생 2026-10-07: 패스 시작 쪽 7–22 mm 구멍 = 크루즈 전 구간). 그 밖의 끝 구간 길이는 보고 (용접 경로가 외삽).
    반환 {groove|floor|root_edges: {n_stations, min_count, median_count, short, ok, need, end_lag_m, range_m, unjudged_ends_m}}."""
    s = np.asarray(feats.get("s_line_m", []), dtype=float)
    ok = np.asarray(feats.get("ok", np.zeros(s.shape, dtype=bool)), dtype=bool)
    if s.size == 0 or ok.shape != s.shape or not (_fin(s_a) and _fin(s_b)):
        return {}

    def flag(key):
        v = np.asarray(feats.get(key, np.full(s.shape, np.nan)), dtype=float)
        return (v == 1.0) if v.shape == s.shape else np.zeros(s.shape, dtype=bool)
    ss = np.sort(s[ok & np.isfinite(s)])
    sp = float(np.median(np.diff(ss))) if ss.size >= 2 else float("nan")       # 착점 간격 (유효 프로파일의 중앙 간격)
    kw = dict(step_m=float(process.station_step_mm) * 1e-3, window_m=float(process.station_window_mm) * 1e-3,
              min_profiles=int(process.min_profiles), spacing_m=sp)
    out = {}
    a0, b0 = sorted((float(s_a), float(s_b)))
    lag = max(0.0, float(end_lag_m))
    a, b = (a0 + lag, b0 - lag) if b0 - a0 > 2.0 * lag else (a0, b0)
    cr = np.asarray(feats.get("in_cruise", np.ones(s.shape, dtype=bool)), dtype=bool)
    if cr.shape == s.shape:
        sc = s[ok & cr & np.isfinite(s)]
        if sc.size:
            a, b = max(a, float(sc.min())), min(b, float(sc.max()))
    for kind, valid in (("groove", ok), ("floor", ok & flag("floor_visible")), ("root_edges", ok & flag("root_edges_resolved"))):
        cov = _ms.station_coverage(s, valid, a, b, **kw)
        cov["end_lag_m"] = lag
        cov["range_m"] = [a, b]
        cov["unjudged_ends_m"] = [max(0.0, a - a0), max(0.0, b0 - b)]
        out[kind] = cov
    return out


def pass_feature_inputs(fit_kwargs: dict) -> tuple:
    """fit_pass 가 프로파일 특징 추출기에 주는 (설계, 추출 인자) — 패스 중 판 성장 스트림 (post_scan) 도 같은 값을 쓴다 (한 곳에서 정의)."""
    return fit_kwargs.get("spec"), {"qc_targets": fit_kwargs.get("qc_targets")}


def fit_pass(result: ScanResult, spec: Optional[GrooveSpec], *, u_prior, n_prior, stage: str, u_travel,
             target_standoff_m: float, seed: int = 0,
             gravity_axis_base, end_params: Optional[dict] = None, feature_workers=1, timing: Optional[dict] = None,
             feature_executor=None, qc_targets: Optional[dict] = None, process: Optional[WeldProcess] = None,
             plate_seeds=None) -> tuple:
    """ScanResult → (SeamScanFeatures, PassFit, 루트 폴리라인 (M,3), 상부 폴리라인 (M,3), 배치 잔차 dict). 순수 함수 (오프라인 refine 도).

    refine_from_pass (루트 중심 3D RANSAC 직선 + 판 점 평면; 인라이어 = k × 이 패스의 LMedS 척도, 하한 센서 양자화 — 2026-10-07) →
    판 끝 아웃라이어 필터 (plate_edge_outliers: 관측 범위 양끝 끝 증거 구간 (지연 + 단면 창) 안에서 직선에서 k × 내부 잔차의 강건 σ
    넘게 벗어난 루트 점은 NaN·ok False; 라운드 4 F7) → identify_seam_ends (모델 기반 판 끝: on-specimen 판정·가운데에서 키우는 연속
    구간·판별 판 모서리·관측/가림; 입력 end_params = scan_core.localize.EndParams — 판정 문턱은 측정·설계에서 유도) → 식별된 시편 구간 밖의 프로파일은
    측정 루트 점·직선 투영 (root_polyline_fit) 모두 NaN (납품 중심선은 시편 구간의 프로파일만) → 특징 중앙값 재계산
    (+ root_edges_fraction·강건 표준편차·fov_margin_mm·sensor_z_mm) → pass_residuals (센서 배치 진단: psi/theta/roll/lateral/standoff).
    plate_seeds: 행마다 패스 중 미리 계산한 판 성장 (post_scan 스트림; 결과는 없이 계산한 것과 같다)."""
    tm = timing if timing is not None else {}             # 단계별 시간 [s] (패스 처리 시간 기록, 2026-10-05)
    t_ = time.perf_counter()
    _spec, fkw = pass_feature_inputs({"spec": spec, "qc_targets": qc_targets})
    sf = extract_seam_features_along_scan(result, spec, keep_features=True, centre="root", workers=feature_workers,
                                          executor=feature_executor, plate_normal_base=n_prior, plate_seeds=plate_seeds, **fkw)
    tm["features_s"] = time.perf_counter() - t_
    tm["features_mode"], tm["features_workers"] = _sfmod.FEATURE_RUN.get("mode"), _sfmod.FEATURE_RUN.get("workers")
    if _sfmod.FEATURE_RUN.get("error"):
        tm["features_pool_error"] = _sfmod.FEATURE_RUN.get("error")
    if _sfmod.FEATURE_RUN.get("note"):                  # 프로세스 풀이 수명 규칙 없이 돌았다 (표준 라이브러리 내부가 다름 — scan_core.proc_pool)
        tm["features_pool_note"] = _sfmod.FEATURE_RUN.get("note")
    t_ = time.perf_counter()
    P_root, gf = polyline_from_scan_features(result, sf, centre="root")
    P_top, _ = polyline_from_scan_features(result, sf, centre="top")
    # 판 평면은 그루브가 잡힌 프로파일의 판 점에서만 (끝 너머 테이블·지그 면이 RANSAC 평면을 빼앗지 않도록)
    Q = plate_points_from_scan(result, [f if bool(gf[j]) else None for j, f in enumerate(sf.features)], sf.indices)
    tm["polylines_plate_points_s"] = time.perf_counter() - t_
    epar = EndParams.from_dict(end_params)
    proc = process if process is not None else WeldProcess()
    window_m = float(proc.station_window_mm) * 1e-3                       # 용접 경로 단면 창: 잡음 이동 중앙값·시편 보고 구간
    sf.noise_window_m = window_m
    rows_ = sf.indices[np.asarray(gf, dtype=bool)]
    z_pass = _profile_z_mm(result, rows_[:: max(1, int(math.ceil(rows_.size / _SAMPLE_PROFILES)))] if rows_.size else rows_)
    q_m = _ms.quant_sigma_mm(z_pass) * 1e-3                              # 센서 양자화 하한 (판정 σ 의 바닥)
    t_ = time.perf_counter()
    fit = refine_from_pass(P_root, Q, gf, sf, spec, u_prior=u_prior, n_prior=n_prior, seed=seed, k=epar.k, floor_m=q_m,
                           min_profiles=2 * int(proc.min_profiles), min_line_pts=int(proc.min_profiles))   # 단면 둘 = 방향 하나
    tm["refine_s"] = time.perf_counter() - t_
    t_ = time.perf_counter()
    edge_info: dict = {}
    edge = plate_edge_outliers(P_root, fit, edge_m=epar.look_m, floor_m=q_m, k=epar.k, diag=edge_info)
    fit.diag["edge_outlier_filter"] = edge_info
    if edge.any():
        # 판 끝 모서리에 걸친 프로파일(리드인 없는 3G 'up' 의 첫 프로파일, 판 끝을 지나는 순간의 프레임) 의 루트 점은 직선에서 수 mm
        # 벗어난다: RANSAC 은 이미 버렸지만 용접 경로 단면과 끝 판정에도 남기지 않는다 — NaN + ok False (fit.diag 에 기록)
        P_root = np.array(P_root, dtype=float, copy=True)
        P_top = np.array(P_top, dtype=float, copy=True)
        P_root[edge] = np.nan
        P_top[edge] = np.nan
        ok = np.array(sf.ok, dtype=bool, copy=True)
        ok[edge] = False
        sf.ok = ok
        fit.diag["plate_edge_outliers"] = [int(i) for i in np.flatnonzero(edge)]
    gf2 = np.asarray(sf.ok, dtype=bool) & np.all(np.isfinite(P_root), axis=1)
    if fit.ok or (np.all(np.isfinite(fit.c)) and np.all(np.isfinite(fit.u))):
        tm["edge_outliers_s"] = time.perf_counter() - t_
        t_ = time.perf_counter()
        identify_seam_ends(result, sf, fit, P_root, gf2, spec, epar)
        support_plane_ends(result, sf, fit, gravity_axis_base=gravity_axis_base, params=epar)   # 끝이 테이블·받침 면에서 끝나면 그 교점
        fit.diag.pop("extent_vs_spec_mm", None)
        if spec is not None and fit.ok and not any(f.startswith("end_not_observed") for f in fit.flags):
            fit.diag["extent_vs_spec_mm"] = (fit.s_b - fit.s_a) * 1e3 - spec.weld_length_mm
        run = fit.diag.get("run") if "ends" in fit.diag else None
        if run is not None:
            # 특징 통계·납품 중심선은 식별된 시편 (가운데를 포함하는 연속 on-specimen 구간) 안에서만 — 끝 너머의 물체 (클램프·지그)
            # 가 만든 그루브 비슷한 프로파일이 중앙값·바닥 비율·일관성에 섞이지 않게
            inside = np.zeros(P_root.shape[0], dtype=bool)
            inside[int(min(run)):int(max(run)) + 1] = True
            out = ~inside
            if out.any():
                P_root = np.array(P_root, dtype=float, copy=True)
                P_top = np.array(P_top, dtype=float, copy=True)
                P_root[out] = np.nan
                P_top[out] = np.nan
                ok = np.array(sf.ok, dtype=bool, copy=True)
                ok[out] = False
                sf.ok = ok
                for k in ("groove_centre_x_mm", "groove_centre_z_mm", "root_centre_x_mm", "root_centre_z_mm", "top_opening_mm",
                          "root_gap_mm", "bevel_angle_deg", "depth_mm", "hi_lo_mm", "tilt_deg", "wall_shear_k", "floor_visible",
                          "floor_span_mm", "root_edges_resolved"):
                    a = getattr(sf, k, None)
                    if a is not None and np.asarray(a).shape[0] == out.size:
                        a = np.array(a, dtype=float, copy=True)
                        a[out] = np.nan
                        setattr(sf, k, a)
                fit.diag["outside_specimen_profiles"] = int(out.sum())
            gf2 = np.asarray(sf.ok, dtype=bool) & np.all(np.isfinite(P_root), axis=1)
    # 납품 중심선: 직선 투영 (root_polyline_fit) 도 측정 루트 점이 남은 프로파일 (식별된 시편 구간 안, 판 끝 아웃라이어 아님) 에서만
    tm["ends_s"] = time.perf_counter() - t_
    t_ = time.perf_counter()
    rpf = np.asarray(fit.root_polyline_fit, dtype=float)
    if rpf.shape == P_root.shape:
        rpf = np.array(rpf, dtype=float, copy=True)
        rpf[~np.all(np.isfinite(P_root), axis=1)] = np.nan
        fit.root_polyline_fit = rpf
    fm = dict(sf.medians())
    fm.setdefault("wall_shear_k", float("nan"))
    # 시편의 실제 치수 변화 (시공 오차): 심 방향 구간 중앙값 — 보고용 (사용자 2026-10-02: 갭이 끝으로 벌어지는 것 등은 정확히 측정해
    # 보고할 대상이지 측정 실패가 아니다). 구간 중앙값의 최소·최대를 요약에도 넣는다.
    # 구간 좌표 = 정련 심 직선 좌표 t = (P_root − c)·u (판 끝 s_m 과 같은 좌표 — specimen_report 가 S 끝에서의 거리로 바꾼다); 경계는
    # 식별된 첫 판 끝에 맞춘다 (설계 길이가 구간 길이의 배수면 양 끝 모두 경계와 맞는다)
    t_line = None
    if np.all(np.isfinite(fit.c)) and np.all(np.isfinite(fit.u)):
        t_line = (np.asarray(P_root, dtype=float) - np.asarray(fit.c, dtype=float)) @ np.asarray(fit.u, dtype=float)
    s_first = ((fit.diag or {}).get("ends") or {}).get("first", {}).get("s_m")
    bins = (sf.along_seam_bins(window_m, s=t_line, origin=s_first if _fin(s_first) else None)
            if t_line is not None else [])
    fit.diag["specimen_bins"] = bins
    for k in ("root_gap_mm", "depth_mm", "top_opening_mm", "hi_lo_mm"):
        vals = np.array([b[k] for b in bins if np.isfinite(b.get(k, float("nan")))], dtype=float)
        fm[f"{k[:-3]}_along_min_mm"] = float(vals.min()) if vals.size else float("nan")
        fm[f"{k[:-3]}_along_max_mm"] = float(vals.max()) if vals.size else float("nan")
    # 루트 위치의 프로파일 간 잡음 (용접 경로가 쓰는 루트 중심; 사용자 2026-10-07: 루트 위치 허용치 1 mm 로 판정): 정련 직선에서 측면
    # (w = n × u) 과 판 법선 (n) 방향 거리, 심 방향 실제 굽음은 이동 중앙값으로 뺀다 (치수 잡음과 같은 규칙)
    if t_line is not None and np.all(np.isfinite(fit.n)):
        n_ax = unit(np.asarray(fit.n, dtype=float))
        w_ax = unit(np.cross(n_ax, unit(fit.u)))
        rel = np.asarray(P_root, dtype=float) - np.asarray(fit.c, dtype=float)
        fm["root_lateral_noise_mm"] = _sfmod.detrended_noise(t_line, (rel @ w_ax) * 1e3, window_m)
        fm["root_normal_noise_mm"] = _sfmod.detrended_noise(t_line, (rel @ n_ax) * 1e3, window_m)
    fm["ok_fraction"] = float(np.mean(gf2)) if gf2.size else 0.0
    fm["n_groove_profiles"] = int(gf2.sum())
    fm.update(_data_validity(result, sf))
    fit.feature_medians = fm
    tm["medians_bins_s"] = time.perf_counter() - t_
    t_ = time.perf_counter()
    res = pass_residuals(result, sf, fit if fit.ok else None, u_travel, target_standoff_m,
                         gravity_axis_base=gravity_axis_base, process=proc)
    tm["residuals_s"] = time.perf_counter() - t_
    return sf, fit, P_root, P_top, res


def plate_edge_outliers(P_root, fit: PassFit, thr_m: Optional[float] = None, edge_m: Optional[float] = None, *,
                        floor_m: Optional[float] = None, k: float = _ms.K_DEFAULT, diag: Optional[dict] = None) -> np.ndarray:
    """(M,) bool: 관측 범위 양끝 edge_m (끝 증거 구간 = 지연 + 단면 창) 안에 있고 정련 직선(c, u) 에서 문턱보다 먼 루트 점 (판 끝 모서리에
    걸친 프로파일). 가운데 (끝 구역 밖) 점은 절대 빼지 않는다 (직선이 아닌 실제 형상을 숨기지 않는다).

    문턱 (라운드 4 F7 → 2026-10-07 원칙 1): thr = k · max(σ, floor_m), σ = 내부 (끝 구역 밖, 없으면 전체) 루트 점의 직선 잔차 강건
    표준편차 — 직선에 수직인 두 성분 (판 법선 n, 측면 w = n × u) 각각의 1.4826 · MAD 중 큰 값; floor_m = 센서 양자화 dX/√12 (모르면
    유효 창 가운데 값). 잔차 크기 |r| 를 thr 와 비교한다. thr_m 을 주면 그 고정 문턱. diag (dict) 에 thr_m, sigma_m, n_interior, rule."""
    P = np.asarray(P_root, dtype=float).reshape(-1, 3)
    out = np.zeros(P.shape[0], dtype=bool)
    info = diag if diag is not None else {}
    edge_m = EndParams().look_m if edge_m is None else float(edge_m)     # 끝 증거 구간 (지연 + 단면 창; 호출자는 이 패스 값을 준다)
    fl = float(_ms.quant_sigma_mm(None) * 1e-3 if floor_m is None else floor_m)
    info.update(thr_m=float("nan"), sigma_m=float("nan"), n_interior=0, floor_m=fl, k=float(k), edge_zone_m=float(edge_m),
                rule="not_evaluated")
    if not fit.ok:
        return out
    fin = np.all(np.isfinite(P), axis=1)
    if fin.sum() < 3:
        return out
    c, u = np.asarray(fit.c, dtype=float), unit(fit.u)
    rel = P[fin] - c
    sl = rel @ u
    perp_v = rel - np.outer(sl, u)
    perp = np.linalg.norm(perp_v, axis=1)
    at_edge = (sl - sl.min() < edge_m) | (sl.max() - sl < edge_m)
    interior = ~at_edge
    info["n_interior"] = int(interior.sum())
    if thr_m is not None:
        thr = float(thr_m)
        info["rule"] = "fixed"
    else:
        base = interior if int(interior.sum()) >= 3 else np.ones(interior.shape, dtype=bool)
        n_ax = np.asarray(fit.n, dtype=float) if np.all(np.isfinite(fit.n)) else np.eye(3)[int(np.argmin(np.abs(u)))]
        n_ax = unit(n_ax - u * float(n_ax @ u))
        w_ax = unit(np.cross(n_ax, u))
        sig = max([_ms.robust_sigma(perp_v[base] @ ax) for ax in (n_ax, w_ax)])
        thr = float(k) * max(sig if np.isfinite(sig) else 0.0, fl)
        info.update(sigma_m=float(sig), rule="adaptive" if base is interior else "adaptive (all points: no interior)")
    info["thr_m"] = thr
    out[np.flatnonzero(fin)] = at_edge & (perp > thr)
    return out


def end_params_for(job: ScanJob, calib: Calib, spec: Optional[GrooveSpec], process: WeldProcess, *, k: float,
                   ghost_burst_frames: int, depth_m: Optional[float] = None, plate_normal=None) -> dict:
    """판 끝 식별 입력 (scan_core.localize.EndParams) — 판정 문턱이 아니라 다른 곳에서 정해진 값: 통계 배수 k, 용접 경로 단면 창
    (weld_plan), 끝 구역 지연 (end_zone_lag_m: 깊이 × tan 입사), 유령점 버스트 (센서), 판 최소 면 길이 (특징 추출기)."""
    d = float(depth_m) if depth_m is not None else (float(spec.groove_depth_mm) * 1e-3 if spec is not None else 0.0)
    return {"k": float(k), "window_m": float(process.station_window_mm) * 1e-3,
            "lag_m": end_zone_lag_m(job, calib, d, plate_normal), "ghost_burst_frames": int(ghost_burst_frames),
            "plate_min_len_m": float(_sfmod.plate_min_len_mm(spec)) * 1e-3}


def _features_arrays(sf: SeamScanFeatures, P_root: np.ndarray, P_top: np.ndarray, fit: Optional[PassFit],
                     frame: Optional["_seam_frame.SeamFrame"] = None) -> dict:
    """프로파일별 특징 배열 — 용접 경로 계획기 (scan_core.weld_plan) 입력, 메모리에서만 (파일로 남기지 않는다, 사용자 2026-10-03):
    indices/s_m/ok, 상부·루트 중심 (센서 프레임 x/z 와 base 3D 폴리라인), 개구·갭·베벨각·깊이·hi-lo·tilt·wall_shear_k·floor_visible·
    floor_span·in_cruise·bevel_side·바닥 점 수, 판 법선 n, 심 좌표 (c, u, s_a, s_b) 와 각 점의 좌표 s_line_m. frame (그 패스의 관측 끝
    S, E — scan_core.seam_frame) 을 주면 심 좌표는 S→E 계약 (c = S, u = unit(E − S), s = S 에서 잰 거리, 범위 0..L); 없으면 패스의 정련
    직선 (u = 진행 방향)."""
    keys = ("indices", "s_m", "ok", "groove_centre_x_mm", "groove_centre_z_mm", "root_centre_x_mm", "root_centre_z_mm",
            "top_opening_mm", "root_gap_mm", "bevel_angle_deg", "depth_mm", "hi_lo_mm", "tilt_deg", "wall_shear_k",
            "floor_visible", "floor_span_mm", "root_edges_resolved", "in_cruise")
    d = {k: np.asarray(getattr(sf, k)) for k in keys if getattr(sf, k, None) is not None}
    # 단면 면 (scan_core.cross_sections): 면 위 점·레이저 평면 안 면 방향·맞춤 (base, 프로파일 실측 자세), 광학 원점 — 구간별 병합도 이어 간다
    for k in ("face_point_base", "face_dir_base", "face_fit", "opt_origin_base"):
        v = getattr(sf, k, None)
        if v is not None and np.asarray(v).shape[:1] == np.asarray(sf.indices).shape[:1]:
            d[k] = np.asarray(v)
    d["root_polyline_base_m"] = np.asarray(P_root)
    d["top_polyline_base_m"] = np.asarray(P_top)
    if sf.features:
        d["bevel_side"] = np.array([f.bevel_side for f in sf.features])
        d["occluded_floor"] = np.array([bool(f.occluded_floor) for f in sf.features])
        d["n_floor_points"] = np.array([int(f.n_floor_points) for f in sf.features])
    if fit is not None and fit.ok:
        d["root_polyline_fit_base_m"] = np.asarray(fit.root_polyline_fit)
        n = unit(fit.n)
        if frame is not None:
            c, u, s_a, s_b = np.asarray(frame.S, dtype=float), np.asarray(frame.u, dtype=float), 0.0, frame.length_m
        else:
            c, u, s_a, s_b = np.asarray(fit.c, dtype=float), unit(fit.u), float(fit.s_a), float(fit.s_b)
        Pr = np.asarray(P_root, dtype=float).reshape(-1, 3)
        s_line = np.full(Pr.shape[0], np.nan)
        fin = np.all(np.isfinite(Pr), axis=1)
        s_line[fin] = (Pr[fin] - c) @ u
        d["s_line_m"] = s_line
        d["line_c_base_m"] = c
        d["u_base"] = u
        d["n_base"] = n
        d["s_a_m"] = np.array(float(s_a))
        d["s_b_m"] = np.array(float(s_b))
    return d


def _extent_frame(extent: Optional[dict], n=None) -> Optional["_seam_frame.SeamFrame"]:
    """패스 extent → S→E 계약 좌표 (scan_core.seam_frame; 끝 점이 없으면 None)."""
    if extent is None:
        return None
    try:
        return _seam_frame.frame_from_extent(extent, n)
    except _seam_frame.SeamFrameError:
        return None


def pass_quality_entry(k: int, job: ScanJob, calib: Calib, fit: PassFit, feats: dict, spec: Optional[GrooveSpec],
                       qc_targets: Optional[dict], process: WeldProcess, k_judge: float, full: bool = True) -> dict:
    """패스 하나의 품질 기록 (병합 입력; 온라인 _record_quality 와 오프라인 refine 이 같은 함수): 자세, 프로파일별 품질, 시편 위 비율."""
    try:
        posture = _sq.pass_posture(job, calib, plate_normal=fit.n if fit.ok else None)
    except Exception as e:  # noqa: BLE001
        posture = {"incidence_deg": float("nan"), "error": str(e)[:120]}
    q = _sq.profile_quality(feats, spec, qc_targets=qc_targets, spacing_m=_sq.spacing_of(feats), process=process,
                            z_mm=float((fit.feature_medians or {}).get("sensor_z_mm", float("nan"))), k=k_judge)
    sl = None
    if fit.ok and np.all(np.isfinite([fit.s_a, fit.s_b])):
        s_ = _sq.line_coordinate(feats, fit.c, fit.u)
        sl = np.isfinite(s_) & (s_ >= fit.s_a) & (s_ <= fit.s_b)
    sel = sl if sl is not None and sl.any() else np.ones(q["good"].size, dtype=bool)
    fr = {f"frac_{k_}": float(np.mean(v[sel])) if sel.any() else 0.0 for k_, v in q.items()}
    return {"k": k, "feats": feats, "q": q, "posture": posture, "full": bool(full), "fractions": fr}


def merged_pass_features(entries: list, ref_extent: Optional[dict], ref_fit: PassFit, process: WeldProcess) -> tuple:
    """패스 품질 기록 (entries: {k, feats, q, posture, full}) → 구간별 최선 자료 병합 (온라인 _merged_gate 와 오프라인 refine 이 같은 함수).
    좌표 = 기준 패스 (마지막 전체 패스) 의 관측 끝 S, E 의 S→E 계약 (scan_core.seam_frame) — 기준 패스의 진행 방향과 무관. 구간 = 용접
    경로 단면 창 하나, 그 증거 수 (4d P6). 반환 (병합 feats 또는 None, 구간별 선택, 구간 길이 [m])."""
    frame = _extent_frame(ref_extent, ref_fit.n if ref_fit.ok else None)
    if frame is None or not entries:
        return None, [], float("nan")
    sp_ = _sq.spacing_of(entries[-1]["feats"])
    bin_m, need = _sq.bin_for(sp_, process), _sq.bin_need(sp_, process)
    merged, choice = _sq.merge_feature_arrays(entries, frame.S, frame.E, n=frame.n, bin_m=bin_m, need=need)
    if merged is not None and ref_fit.ok:
        merged["n_base"] = unit(ref_fit.n)
    return merged, choice, bin_m


# --------------------------------------------------------------------------------------
# localize (모션 없음)
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# 오케스트레이터
# --------------------------------------------------------------------------------------
def rough_surface_offset_mm(plate_points_base, rough_points, normal, near_point) -> float:
    """정지 측정에서 본 판 표면이 가장 가까운 대략 점보다 판 법선 방향으로 얼마나 높은가 [mm] (음수 = 아래). 대략 점은 판 상면의 레이저 착점으로
    교시하므로 시편 위라면 수 mm 안이다 — 크게 다르면 레이저가 시편이 아닌 곳 (판 끝 너머의 뒷댐재·테이블 등) 을 보고 있다 (2026-10-05 R3 at24:
    대략 S 를 판 끝 밖 33 mm 에 두자 판 표면이 대략 점보다 29 mm 아래로 보였다). 판 점이 없으면 nan."""
    P = np.asarray(plate_points_base, dtype=float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    R = [np.asarray(r, dtype=float).reshape(3) for r in rough_points if r is not None]
    n = np.asarray(normal, dtype=float).reshape(3)
    if P.shape[0] == 0 or not R or not np.all(np.isfinite(n)):
        return float("nan")
    n = n / np.linalg.norm(n)
    q = np.asarray(near_point, dtype=float).reshape(3) if near_point is not None else np.median(P, axis=0)
    r = min(R, key=lambda v: float(np.linalg.norm((v - q) - ((v - q) @ n) * n)))
    return float((np.median(P, axis=0) - r) @ n) * 1e3


def weld_process_doc(doc: Optional[dict], position) -> dict:
    """job doc 의 weld_plan 블록 (공정 가정값) — 3G 는 방향을 정하지 않았으면 'uphill' (아래 → 위, 중력 반대; 2026-10-04: 기본 S->E 는
    3G 템플릿 (S = 위 점) 에서 아래로 용접하는 경로를 냈다). 1G 는 그대로."""
    wp = dict((doc or {}).get("weld_plan") or {})
    if str(position or "").strip() == "3G" and "direction" not in wp:
        wp["direction"] = "uphill"
    return wp


class AutoTeach:
    """모듈 docstring 의 시퀀스를 실행한다. run() → AutoTeachResult. 예외는 삼키고 status 로 보고."""

    def __init__(self, settings: Settings, job0: ScanJob, doc: Optional[dict], calib: Calib, *, out_dir,
                 session: ScanSession, confirm: str = "each", dry_run: bool = False,
                 max_passes: Optional[int] = None, measure_speed_mps: Optional[float] = None, confirm_source=None,
                 log: Optional[Callable[[str], None]] = None, sensor_ip: Optional[str] = None, sensor_port: Optional[int] = None,
                 sensor_init: Optional[dict] = None,
                 on_event: Optional[Callable[[str, dict], None]] = None,
                 local_speed_mps: Optional[float] = None, accel_mps2: Optional[float] = None, transit_speed_mps: Optional[float] = None,
                 descend_speed_mps: Optional[float] = None, motion_timeout_s: Optional[float] = None,
                 dwell_frames: Optional[int] = None, probe_frames: int = 30,
                 position: Optional[str] = None, release_sensor: bool = False,
                 pass_cloud_wanted: Optional[Callable[[], bool]] = None, cancel: Optional[Future] = None):
        """session: 열린 ScanSession (로봇 플랫폼 = MoveIt 어댑터 — ROS 서버가 실행 사이에 유지; 시험은 가상 팔) — 그 세션은 닫지 않는다.
        release_sensor: 이 실행이 연 센서는 출력 쓰기 전에 원본 파라미터로 복원하고 (S6) 끝에서 닫는다 (로봇 플랫폼은 그대로 둔다).
        local/descend/transit 속도·가속 None = 플랫폼 TCP 한계 (MotionConfig.v_eff_max_mps·a_eff_max_mps2 — 안전 정책의 정지 거리도 이
        속도로 잰다; 4d C-01/C-02). 명시값은 운전자 낮춤 (플랫폼 한계를 넘지 않는다).
        pass_cloud_wanted: 처리된 패스 점구름 (on_event 'pass_cloud', RViz) 을 볼 쪽이 지금 있나 — 패스마다 묻는다 (ROS 서버: ~/pass_cloud
        구독자 수). None = 만들지 않는다.
        cancel: 실행 취소 Future (액션 취소·서버 종료가 끝낸다, S4) — 작업 프로세스·풀의 모든 기다림 (패스 뒤 처리·추적·재맞춤·뒤 저장·용접 경로·
        단면) 이 결과·작업 프로세스의 죽음과 함께 이것을 기다린다 (post_scan.wait_result). None = 취소 경로 없음 (CLI·시험).
        measure_speed_mps: 이 실행의 측정 속도 [m/s] (AutoTeach 목표 measure_speed_mps · 클라이언트 --measure-speed, 사용자 2026-10-12) — 이
        실행의 작업 speed_mps 가 된다 (job0 에서 파생되는 모든 것: 간격·오버스캔·안정·구간 기록). None = 작업 파일의 speed_mps."""
        if confirm not in CONFIRM_MODES:
            raise ValueError(f"confirm must be one of {CONFIRM_MODES}")
        self.settings = settings
        self.position_override = position         # CLI --position: job 문서의 position 을 덮어쓴다 (_gates 에서 적용)
        self.job0 = job0
        # the run's measurement speed (user 2026-10-12): an override replaces the job's speed_mps for this run only (the job file is not
        # changed) - every speed-dependent quantity derives from job0 (spacing at the frame rate, the pass overscan's ramp / settling
        # terms, the legs' records); the noise priors' speed model reads the measured along-track speeds and the safety policy's
        # stopping distance the platform TCP limit, so neither depends on this value
        self.measure_speed_job_mps = float(job0.speed_mps)
        self.measure_speed_override_mps: Optional[float] = None
        self.measure_speed_check: Optional[dict] = None   # the calibrated-range check of a run-time speed (the 1G core's gates)
        self.job_effective_path: Optional[str] = None     # <out>/job_effective.yaml (legacy) | verification/job_effective.yaml (1G core)
        if measure_speed_mps is not None:
            v_ms = float(measure_speed_mps)
            if not (math.isfinite(v_ms) and v_ms > 0.0):
                raise ValueError(f"measure_speed_mps must be a finite speed > 0 m/s (None = the job's speed_mps), got {measure_speed_mps!r}")
            self.measure_speed_override_mps = v_ms
            self.job0 = dataclasses.replace(job0, speed_mps=v_ms)
        self.doc = dict(doc or {})
        self.calib0 = calib
        self.calib = calib                       # 유효 캘리브 (프로브가 광학 분기를 뒤집으면 바뀐다)
        self.out = Path(out_dir)
        self.confirm_mode = confirm
        self.dry_run = dry_run
        self.confirm_source = confirm_source
        self.log = log or (lambda s: print(s, flush=True))
        self.conn = dict(sensor_ip=sensor_ip, sensor_port=sensor_port)
        self.pose_source = str(getattr(session.robot, "pose_source", "") or "")
        self.session = session
        self.release_sensor = bool(release_sensor)
        self.sensor_init = sensor_init
        self.on_event = on_event
        self.pass_cloud_wanted = pass_cloud_wanted
        self.cancel = cancel
        self._speed_args = (local_speed_mps, descend_speed_mps, accel_mps2)   # _motion_config 에서 플랫폼 한계로 정한다
        self.local_speed = self.descend_speed = self.accel = float("nan")
        self.transit_speed = float("inf") if transit_speed_mps is None else float(transit_speed_mps)   # 실효 상한 = 플랫폼 (cfg.v_eff_max_mps)
        self.motion_timeout_s = motion_timeout_s
        at = settings.auto_teach_config()
        job_at = self.doc.get("auto_teach") or {}
        if job_at:
            # 작업 파일의 auto_teach: 덮어쓰기 (허용 키만 — 캘리브 확인 실기의 주 패스 자세 고정 등; 안전 여유·캡은 플랫폼 설정에서만)
            from .config import JOB_AUTO_TEACH_KEYS
            bad = sorted(set(job_at) - set(JOB_AUTO_TEACH_KEYS))
            if bad:
                raise ValueError(f"job file auto_teach keys {bad} are not allowed per job (allowed: {list(JOB_AUTO_TEACH_KEYS)}; "
                                 "the rest belongs in the platform config)")
            at = AutoTeachConfig.from_params({**dict(settings.auto_teach or {}), **dict(job_at)})
        self._job_at = dict(job_at)
        self.at = at
        self.max_passes = int(max_passes if max_passes is not None else at.max_passes)
        self.dwell_frames = int(dwell_frames if dwell_frames is not None else at.dwell_frames)
        self.probe_frames = int(probe_frames)
        self.spec = specimen_from_doc(self.doc)
        self.qc_targets = qc_targets_effective(self.at, self.doc, settings.z_valid_mm)   # 플랫폼 기본 ← job qc_targets
        # 상태
        self.res = AutoTeachResult(out_dir=str(self.out))
        self._motion_sent = False                     # 이동을 하나라도 보냈나 (보내기 직전에 True; 실패해도 True 로 남는다)
        self._pending_saves: list = []                # 작업 프로세스가 뒤에서 쓰는 raw.npz·recon/ (실행 끝에 기다린다)
        self._leg_cloud = None                        # 마지막 패스의 처리된 점구름 원천 (Future 또는 (재구성 결과, 간격); 볼 쪽이 있을 때만)
        self._vis_clouds: list = []                   # 만든 패스 점구름 (k, 점 수, 만든 시간 s, 어디서) — 콜백 스레드가 더한다
        self._tilt_sign: Optional[int] = None         # 3G: 끝 자세와 함께 고른 거친 진행 부호 (토치 위/아래)
        self.table_measured_h: Optional[float] = None # 레이저가 잰 시편 끝 지지면 (테이블) 높이 (_note_support)
        self.tilt_main: Optional[float] = None        # 3G 입사 분리: 패스 2 부터 쓰는 주 입사 각도 (job 값; None = 분리 안 함)
        self.tilt_end: Optional[float] = None         # 3G: 거친 단계·패스 1 의 끝 입사 각도 (choose_3g_end_postures)
        self.lower_end_obs: Optional[dict] = None     # 3G 입사 분리: 끝 패스가 관측한 아래 끝 (주 패스가 이어받는다)
        self.pass_quality: list = []                  # 적응형 스캔: 패스별 {k, feats, q, posture, full} (scan_quality)
        self.last_full: Optional[dict] = None         # 마지막 전체 패스의 fit·extent·잔차·중앙값 (재스캔 판정 기준)
        self.merge_choice: list = []                  # 구간별 선택 (병합)
        self._rescan_prepared = None                  # 다음 재스캔 패스의 (job, ov)
        self.main_posture: Optional[dict] = None      # 3G 주 패스 자세 {tilt, torch_above, phi, ...} (_main_posture; None = job 각도 규칙)
        self.bad_postures: list = []                  # 품질이 낮았거나 직선을 못 맞춘 자세 {k, phi, torch_above, why} — 다시 쓰지 않는다
        self.end_pass_ref: Optional[dict] = None      # 3G 끝 패스의 직선·위 끝 (자세 간 오프셋)
        self.posture_offsets: list = []               # 주 패스마다 끝 패스 대비 측정 오프셋 (_posture_offset) — 로그·캘리브 자료
        self.carry_refused: Optional[str] = None      # 이번 주 패스에서 아래 끝을 이어받지 못한 사유
        self._carry_refused_once = False               # 직전 주 패스도 이어받지 못했나 (확인 먼저, 4d P8)
        self._carry_refusal_final = False              # 다음 주 패스로 바뀔 수 없는 거부 (끝 패스가 위 끝을 못 봄)
        self._carry_refused_first: Optional[str] = None   # 그 첫 거부 사유 (멈출 때 함께 보고)
        self.floor_h: Optional[float] = None          # 3G 바닥 높이 (_gates 가 대략 아래 점으로 정하고, 첫 패스가 아래 끝을 재면 그 끝)
        self.floor_src = ""
        self.sess: Optional[ScanSession] = session          # 넘겨받은 세션 (MoveIt 어댑터 서버; 시험은 가상 팔) — 닫지 않는다
        self.owned_session = False
        self._latch_at_start = self._latch_last()          # 공유 세션: 앞 실행이 남긴 정지 기록 (이 실행 로그에 넣지 않는다)
        self.cfg = None
        self.policy: Optional[SafetyPolicy] = None
        self.view: Optional[SensorView] = None        # 센서 시야 모형 (거리·탐색 간격 유도; _apply_safety_policy)
        self._plate_from_sensor = False
        self.job_ref: Optional[ScanJob] = None    # 로컬 이동 포락선 규칙의 출처 (거친 작업 + 실측 판 상면/평면)
        self.job_c: Optional[ScanJob] = None      # 거친 작업 (드웰·S→E 이동)
        self.u0 = None
        self.n0 = None
        self.up = None
        self.orientation = "flat_1G"
        self.plate_top: Optional[float] = None
        self.n_grab = 0
        self.n_pass = 0
        self._last_stream = None                  # (records, samples, MotionResult) of the last streamed leg (partial save)
        self._seed_feeder = None                  # 패스 녹화 중 판 성장 스트림 (post_scan.SeedFeeder; _streamed_leg 가 준비, _local_move 가 시작)
        self.sensor_ready = False
        self._clock = get_clock()                 # 시계 이음매 (기본 REAL = 실시계 그 자체; 시험만 가상 시계)
        self.t0 = self._clock.monotonic()
        self._normal_tol_deg: Optional[float] = None    # 3G: 고른 끝 자세가 견디는 판 법선 차 (자세 여유 / 피벗 반경) — 넘으면 다시 획득
        self._coarse_range_posture: Optional[float] = None   # 3G: 고른 끝 자세의 거친 거리 (지지면 여유 때문에 늘었을 수 있다)
        self.plate_plane_measured = None               # 3G: (point, normal) 채택된 실측 판 평면 (C6)
        self.pass_ends: list = []                      # 패스별 S/E 끝 관측 (시간 오프셋 추정)
        self.end_ambiguous_seen: Optional[dict] = None  # 길이가 설계에서 어긋난 첫 읽기 {k, sign, length_mm, sigma_mm} (다음 패스로 다시 잰다)
        self._occluded_seen: dict = {}                 # 가려진 끝의 첫 읽기 {S|E: 패스} (확인 먼저: 다음 패스가 그 끝을 다시 잰다, 4d P9)
        self._prev_spec_failed: set = set()            # 직전 SPEC_FAIL 의 실패 검사 (같은 검사가 두 번이어야 확인, 4d P9)
        self._recon_failed_prev = False                # 직전 패스의 재구성 실패 (두 번 연속이면 멈춘다, 4d P9)
        self._end_repeat_pending = False               # 3G: 다음 패스는 끝 패스 반복 (아래 끝을 못 잰 자료 쪽 실패, 4d P9 F-16/F-17)
        self._end_repeat_used = False                  # 끝 패스 반복은 실행당 한 번
        self.pass_pitch_deg: Optional[float] = None    # 3G: 패스 1 에서 한 번 고른 피치 (입사 기울기에 더해지는 부호), 이후 모든 패스·정련 작업
        self.ifr = None                                # 범용 시작: acquisition.InitialFrame (착점 좌표·자세·중력만)
        self.cell_ids: list = []                       # 범용 시작: MoveIt 충돌 장면에 넣은 상자 id (획득 뒤·끝에서 지운다)
        self.support: Optional[dict] = None            # 범용 시작 3G: 지지면 probe 결과 {height, src, ...}
        self._reacquired = False                       # 범용 시작: 첫 드웰의 판 법선으로 다시 획득했나 (한 번만)
        self._first_loc: Optional[StaticLocalization] = None   # 범용 시작: 첫 드웰 (다시 획득할 때 잰 판 평면을 새 기준 작업에)
        self._n_reacq = 0                                      # 3G S 다시 획득 횟수 (장면 검사 미루기: 다시 획득이 남았을 때만)

    # ---- 기록 --------------------------------------------------------------------------------
    def _p(self, msg: str) -> None:
        self.log(msg)

    def _decide(self, step: str, msg: str, **data) -> None:
        self.res.decisions.append({"t": round(self._clock.monotonic() - self.t0, 3), "step": step, "msg": msg, **_jsonable(data)})
        self._p(f"[{step}] {msg}")

    def _event(self, evt: str, **data) -> None:
        if self.on_event is not None:
            try:
                self.on_event(evt, data)
            except Exception:  # noqa: BLE001 - a test hook must never break the run
                pass

    # ---- 세션 ------------------------------------------------------------------------------
    @property
    def rs(self) -> RobotPlatform:
        assert self.sess is not None
        return self.sess.robot

    @property
    def ss(self) -> SensorSession:
        assert self.sess is not None and self.sess.sensor is not None
        return self.sess.sensor

    def _T_now(self) -> np.ndarray:
        return self.rs.pose.latest_pose()

    def _T_opt_now(self) -> np.ndarray:
        return self._T_now() @ self.calib.T_flange_opt

    # ---- 1. 게이트 ----------------------------------------------------------------------------
    def _check_position(self) -> None:
        """용접 자세 (사용자 2026-10-01): job.position '1G'|'3G' (또는 CLI --position) 가 있어야 한다 — 'auto' 는 REFUSED. 그 뒤
        validate() 의 방향 규칙이 주어진 자세와 플랫폼 gravity_axis_base 의 일관성 검사다 (메시지에 둘 다)."""
        if self.position_override is not None:
            try:
                job = with_position(self.job0, self.position_override)
            except JobError as e:
                raise _Abort(STATUS_REFUSED, f"--position: {e}", category="input")
            if job.orientation != self.job0.orientation:
                self._decide("setup", f"welding position set by --position: {self.position_override} ({job.orientation}); "
                                      f"job file had {self.job0.position or 'none'} ({self.job0.orientation})")
            self.job0 = job
        pos = position_from_orientation(self.job0.orientation)
        if pos is None:
            raise _Abort(STATUS_REFUSED, f"rough job has no welding position (job.position missing / orientation "
                                         f"{self.job0.orientation!r}): {POSITION_REFUSAL} - set job.position in the job file or "
                                         "pass --position (auto-teach does not infer it from the rough point layout)", category="input")
        self.res.position = pos

    def _gates(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.job_effective_path = self._save_effective_job(self.out / _jo.EFFECTIVE_JOB_NAME)   # before anything else (a record)
        self._check_position()
        try:
            self.calib = attach_tool_body(self.calib, self.settings)          # 툴 몸체 (EOAT 메시 껍질): 테이블 규칙·3G 각도 선택
            self.calib0 = attach_tool_body(self.calib0, self.settings)
        except ValueError as e:
            raise _Abort(STATUS_REFUSED, f"tool body: {e} (nothing sent)")
        if self.spec is None or not (self.spec.weld_length_mm > 0.0):
            # 설계 정보는 필수 입력 (사용자 2026-09-30/10-01: 거친 시·종점 + 그루브 설계 정보로 정렬; 길이·깊이를 분명히 확인).
            # 없으면 잔여 연장 R = L_design − L_SE 와 길이 검사 (± tol) 를 할 수 없다 — 조용히 약화시키지 않고 거부 (2026-10-02,
            # 실기 job_rough.yaml 이 teach 로만 만들어져 specimen 블록이 없었다).
            raise _Abort(STATUS_REFUSED, "job has no specimen (groove design) block with weld_length_mm > 0: auto-teach needs the "
                                         "design (length, thickness, bevel, gap) — copy the 'specimen:' and 'qc_targets:' blocks "
                                         "from scan_l0/jobs/groove_10T_1G.yaml (or _3G) into the job (nothing sent)")
        self._universal_inputs()
        self._motion_config()
        job0 = self._apply_safety_policy(self.job0)
        # 거친 법선은 추정값 (템플릿 축·티칭 반올림) — 경사진 시편 (2026-10-04: 1G 용접선 8–10° 경사) 에서 거친 진행 방향과 직각이 아니라는
        # 이유로 거부하지 않고, 진행 방향에 직각인 성분으로 바꾼다 (판 법선은 S 에서 센서가 잰다). 45° 넘게 어긋나면 입력 오류 → 그대로 검증.
        try:
            n_r, u_r = job0.normal_unit, job0.unit_dir
            perp = math.degrees(math.asin(min(1.0, abs(float(n_r @ u_r)))))
            if 0.0 < perp < _ms.HALF_TURN_DEG:                       # 중간점 (원칙 2) 넘게 어긋나면 입력 모순 → 아래 검증이 거부
                n_new = unit(n_r - float(n_r @ u_r) * u_r)
                job0 = ScanJob.from_dict({**job_to_dict(job0), "surface_normal": [float(v) for v in n_new]})
                job0.plate_top_measured_m = self.job0.plate_top_measured_m
                self.job0 = job0
                self._decide("setup", f"rough surface normal made perpendicular to the rough travel direction (was {perp:.1f} deg off): "
                                      f"{np.round(n_new, 4).tolist()} - the plate normal is measured at S")
        except (JobError, ValueError):
            pass
        try:
            job0.validate()
        except JobError as e:
            raise _Abort(STATUS_REFUSED, f"rough job invalid: {e}")
        self.orientation = job0.resolved_orientation()
        self.up = job0.up_unit
        # 방문 순서 (acquisition.visit_order): 1G 로봇에 가까운 착점부터 (사용자 2026-10-09, 교시 순서와 무관), 3G 위 점 → 아래 점 (사용자
        # 2026-10-05: 첫 패스 아래 → 위, 시편 아래 끝을 재고 그 뒤 바닥으로 — 거친 단계가 위 점 → 아래 점이어야 패스 1 (보정된 E → S) 이 아래에서
        # 시작한다). 내부 S/E 만 바뀌고 결과물은 교시 S→E (_user_extent, _final_job)
        vo = visit_order(job0.start, job0.end, self.res.position, job0.gravity_axis_base)
        self.res.visit_order = _jsonable({**vo, "taught_start": job0.start.tolist(), "taught_end": job0.end.tolist(),
                                          "first_visited": "end" if vo["swap"] else "start"})
        if vo["swap"]:
            job0 = self._swap_rough(job0)
            self._decide("setup", (("3G: rough start/end swapped - the coarse stage goes upper point -> lower point so the first pass "
                                    "scans bottom -> top from the lower point and measures the lower specimen end")
                                   if self.orientation == "vertical_3G" else
                                   ("1G: rough start/end swapped - the hit point nearer to the robot base is visited first (user "
                                    "2026-10-09: regardless of the taught order)"))
                         + f" [{vo['why']}]; job_refined, weld_plan, specimen_report and the action result stay in the taught S->E frame",
                         visit_order=self.res.visit_order)
        else:
            self._decide("setup", f"visit order: taught start first [{vo['why']}]", visit_order=self.res.visit_order)
        self._derive_jobs(job0)
        L_min, lb = self._min_site_distance_m()
        L0 = float(np.linalg.norm(np.asarray(job0.end, dtype=float) - np.asarray(job0.start, dtype=float)))
        self._decide("setup", f"minimum S-E distance {L_min * 1e3:.1f} mm = 2 x arrival tolerance at the hit "
                              f"{lb['e_hit_m'] * 1e3:.2f} mm x design length {lb['L_ref_m'] * 1e3:.0f} mm / pass view room "
                              f"{lb['room_m'] * 1e3:.1f} mm; rough points {L0 * 1e3:.0f} mm apart", min_site_distance=_jsonable(lb))
        if L0 < L_min:
            raise _Abort(STATUS_REFUSED, f"rough points only {L0 * 1e3:.0f} mm apart (< {L_min * 1e3:.0f} mm: the seam direction from two "
                                         "sites that close cannot keep the far end of the pass in view) - teach them farther apart "
                                         "(nothing sent)")
        # S 추적 기선 (P2): 대략점과 설계만으로 정해진다 — 기하 상한이 측정 바닥 기선보다 짧으면 이동 전에 거부 (계획 거친 거리; 첫 드웰 뒤
        # _seek_S 가 잰 거리로 같은 규칙을 다시 본다)
        b_trk0, bb0 = self._track_length_m(float(self.job_c.standoff_m))
        why_b = self._baseline_refusal(bb0)
        if why_b:
            raise _Abort(STATUS_REFUSED, f"S track: {why_b} (nothing sent)")
        self._decide("setup", f"S track planned {b_trk0 * 1e3:.1f} mm (needed baseline {bb0['b_star_m'] * 1e3:.1f} mm from "
                              f"{bb0['sigma_sec0_source']}, geometric cap {bb0['cap_m'] * 1e3:.1f} mm, measurement-floor baseline "
                              f"{bb0['b_floor_m'] * 1e3:.1f} mm) at the coarse range {float(self.job_c.standoff_m) * 1e3:.0f} mm",
                     track_plan=_jsonable(bb0))
        if not self.doc.get("meta", {}).get("rough", False):
            self.res.notes.append("input job is not marked meta.rough=true; treated as rough anyway")
        self._decide("setup", f"rough job {job0.job_id or '-'}: position {self.res.position} ({self.orientation}, platform "
                     f"gravity_axis_base {np.round(job0.gravity_axis_base, 4).tolist()}) u0={np.round(self.u0, 4).tolist()} "
                     f"n0={np.round(self.n0, 4).tolist()} L0={job0.length_m:.4f} m coarse range {self.job_c.standoff_m:.4f} m "
                     f"(plate normal {self._coarse_plate_m():.4f} m) pass plate {self.pass_plate_m:.4f} m overscan {self.overscan_m * 1e3:.0f} mm (laser hit point) "
                     f"max_passes {self.max_passes} clamp {self.job_ref.clamp_height_m} m confirm={self.confirm_mode}; scan speed "
                     f"{float(job0.speed_mps) * 1e3:.1f} mm/s ({self.measure_speed_text()})", measure_speed=self.measure_speed_record())
        if self.orientation == "vertical_3G":
            vd = self.job_c.vertical_direction
            self._decide("setup", f"3G travel S0 -> E0: vertical_direction '{vd}' from the rough point heights (S0 is the "
                                  f"{'upper' if vd == 'down' else 'lower'} point; job file had '{job0.vertical_direction}'), approach "
                                  "to S0")
        self._latch_at_start = self._latch_last()
        self.res.decisions.append({"t": 0.0, "step": "speed", "msg": f"commanded TCP speed = effective, capped at "
                                   f"{self.cfg.v_eff_max_mps * 1e3:.1f} mm/s (platform limit)"})
        try:
            init = gate_and_arm(self.sess, self.cfg, log=self.log)
        except PlatformNotReady as e:
            raise _Abort(STATUS_REFUSED, f"{e} (nothing sent)", category="motion")
        except ScanRefused as e:
            raise _Abort(STATUS_REFUSED, str(e))
        self.default_speed = float(init.default_speed)
        self.res.decisions.append({"t": round(self._clock.monotonic() - self.t0, 3), "step": "gates", "msg": "session_init OK",
                                   "safety_init": init.as_dict(), "platform_status": self.rs.gates.status(self.rs),
                                   "platform": {"adapter": self.rs.adapter, "pose_source": self.rs.pose_source,
                                                "motion_backend": self.rs.backend.kind if self.rs.backend is not None else None}})
        if self.job_ref.plate_top_measured_m is not None:
            self.plate_top = float(self.job_ref.plate_top_measured_m)

    def _motion_config(self) -> None:
        """플랫폼 모션 설정 (속도 상한 = 플랫폼 선언 TCP 한계) — 안전 정책의 정지 거리가 쓴다."""
        if self.sess.armed:
            self.cfg = self.rs.backend.cfg
        else:
            self.cfg = self.rs.gates.motion_config(self.settings)
        self.cfg.progress_s = 2.0
        self.cfg.progress_cb = self.log
        if self.motion_timeout_s is not None:
            self.cfg.default_timeout_s = float(self.motion_timeout_s)
        # 4d C-01/C-02: 로컬·하강 속도와 가속 = 플랫폼 TCP 한계 (명시값은 그 아래로만). 가속 한계를 선언하지 않는 플랫폼은 명시값이 필요하다.
        v_max = float(self.cfg.v_eff_max_mps)
        a_max = getattr(self.cfg, "a_eff_max_mps2", None)
        v_loc, v_dsc, a_arg = self._speed_args
        self.local_speed = min(v_max, float(v_loc)) if v_loc is not None else v_max
        self.descend_speed = min(v_max, float(v_dsc)) if v_dsc is not None else v_max
        if a_arg is not None:
            self.accel = min(float(a_max), float(a_arg)) if a_max is not None else float(a_arg)
        elif a_max is not None:
            self.accel = float(a_max)
        else:
            raise _Abort(STATUS_REFUSED, f"the platform ({self.rs.adapter}) declares no TCP acceleration limit: pass accel_mps2 "
                                         "explicitly (nothing sent)", category="config")
        if not (self.local_speed > 0.0 and self.descend_speed > 0.0 and self.accel > 0.0):
            raise _Abort(STATUS_REFUSED, f"local/descend speed and acceleration must be > 0 (got {self.local_speed}, {self.descend_speed} "
                                         f"m/s, {self.accel} m/s^2; nothing sent)", category="config")

    def job_overrides(self) -> list:
        """The run-time job overrides this run's document carries (scan_l0.job_overrides records: path, old, new, source)."""
        return _jo.records_of(self.doc)

    def _job_speed_overrides(self) -> list:
        return [r_ for r_ in self.job_overrides() if r_.get("path") == "job.speed_mps"]

    def measure_speed_choice(self) -> Optional[str]:
        """How the run's measurement speed was chosen at run time: 'run override (measure_speed_mps)' (the AutoTeach goal / client
        --measure-speed), 'job override job.speed_mps (<source>)' (scan_l0.job_overrides), None = the job file's speed_mps."""
        if self.measure_speed_override_mps is not None:
            return "run override (measure_speed_mps)"
        jr = self._job_speed_overrides()
        return f"job override job.speed_mps ({jr[-1].get('source')})" if jr else None

    def measure_speed_record(self) -> dict:
        """The run's measurement speed and where it came from (run log, setup line): the job's speed_mps of this run, the job file's
        value, the run-time choice (override_mps / source: the goal's measure_speed_mps or a job.speed_mps override; None = not given)
        and, when checked, the calibrated range it was checked against (calibrated_range)."""
        ov = self.measure_speed_override_mps
        jr = self._job_speed_overrides()
        file_v = float(self.measure_speed_job_mps)
        if jr and str(jr[0].get("old_from")) == "job file" and jr[0].get("old") is not None:
            file_v = float(jr[0]["old"])                     # the document's speed is the override: the file's is the record's old
        if ov is None and jr:
            ov = float(jr[-1]["new"])
        rec = {"speed_mps": float(self.job0.speed_mps), "job_file_speed_mps": file_v, "override_mps": None if ov is None else float(ov),
               "source": self.measure_speed_choice() or "job file speed_mps"}
        if self.measure_speed_check is not None:
            rec["calibrated_range"] = dict(self.measure_speed_check)
        return rec

    def measure_speed_text(self) -> str:
        """'the job's speed_mps' or 'run override measure_speed_mps; the job file's speed_mps 15.0 mm/s' / 'job override job.speed_mps
        (goal); the job file's speed_mps 30.0 mm/s' (setup lines)."""
        how = self.measure_speed_choice()
        if how is None:
            return "the job's speed_mps"
        how = how.replace("run override (measure_speed_mps)", "run override measure_speed_mps")
        return f"{how}; the job file's speed_mps {self.measure_speed_record()['job_file_speed_mps'] * 1e3:.1f} mm/s"

    def _save_effective_job(self, path) -> Optional[str]:
        """The run's effective job document (the job file + the run-time overrides + the goal's points; meta.job_file /
        meta.job_overrides) → path; None (and a note) when it cannot be written - a record, not a gate."""
        try:
            return str(_jo.write_effective_job(path, self.doc))
        except Exception as e:  # noqa: BLE001 - a record
            self.res.notes.append(f"effective job copy not written: {type(e).__name__}: {e}")
            return None

    def _apply_safety_policy(self, job0: ScanJob) -> ScanJob:
        """안전 정책 (scan_l0.safety, 사용자 2026-10-07: 명시값은 플랫폼 safety.clearance_m 하나) 을 유도해 job0 의 안전 필드
        (한계 L·standoff_min·도착 허용, 플랫폼 max_laser_tilt_deg·torch_protrusion_m) 를 채운다 — 파생 작업은 모두 이어받는다."""
        try:
            self.policy = derive_policy(self.settings.params, self.calib, job0, v_max_mps=float(self.cfg.v_eff_max_mps),
                                        guard_rate_hz=ENVELOPE_REACTION_RATE_HZ)
        except PolicyError as e:
            raise _Abort(STATUS_REFUSED, f"{e} (nothing sent)", category="config")
        sf = self.settings.safety
        tilt, prot = sf.get("max_laser_tilt_deg"), sf.get("torch_protrusion_m")
        prot = None if prot in (None, "null") else float(prot)
        T_tcp = getattr(self.calib, "torch_tcp_eoat", None)
        if T_tcp is not None:
            # the torch keypoint from the arm's own robot model (platform tool_geometry: the weld torch's TCP in TF, its nozzle end from
            # the URDF collision geometry) instead of the job's torch_lateral_m and the configured protrusion: the wire end = TCP +
            # (nozzle end + the job's process stick-out, measured from the nozzle end) along the torch axis - the point the weld plan
            # aims and D14 checks
            try:
                stick = float(WeldProcess.from_dict(weld_process_doc(self.doc if isinstance(self.doc, dict) else None,
                                                                     job0.position)).stick_out_mm)
            except (TypeError, ValueError) as e:
                raise _Abort(STATUS_REFUSED, f"job weld_plan: {e} (nothing sent)", category="input")
            T_tcp = np.asarray(T_tcp, dtype=float)
            off_mm = float(self.calib.torch_nozzle_end_mm) + stick
            tip = T_tcp[:3, 3] + T_tcp[:3, 2] * off_mm * 1e-3
            prot = float(tip[2] - self.calib.p_opt_eoat[2])
            job0 = dataclasses.replace(job0, torch_lateral_m=(float(tip[0]), float(tip[1])))
            self._decide("safety", f"torch keypoint from the robot model: eoat ({tip[0] * 1e3:+.2f}, {tip[1] * 1e3:+.2f}, "
                                   f"{tip[2] * 1e3:+.2f}) mm = {prot * 1e3:.1f} mm beyond the optical origin: the wire end, TCP + "
                                   f"(nozzle end {float(self.calib.torch_nozzle_end_mm):.2f} + stick-out {stick:g}) mm along the torch "
                                   f"axis ({self.calib.torch_tip_source})", torch_tip_eoat_m=tip.tolist(), tcp_to_tip_mm=off_mm)
        job0 = self.policy.apply(job0, max_laser_tilt_deg=None if tilt in (None, "null") else float(tilt), torch_protrusion_m=prot)
        self.job0 = job0
        self._decide("safety", self.policy.summary(), policy=self.policy.as_dict())
        self.view = None
        dm = self._spec_dims_m()
        self._decide("view", self._sv().summary(self._depth_spec_m(), dm["opening"], dm["w"]), view=self._sv().as_dict())
        return job0

    def _sv(self) -> SensorView:
        """센서 시야 모형 (scan_l0.sensor_view): 플랫폼 유효 창 + 센서 모형 + 안전 정책 도착 허용."""
        if self.view is None:
            e = self.policy.pose_tol_m if self.policy is not None else float(self.job0.pose_tol_m)
            near = self.policy.standoff_min_m if self.policy is not None else float(self.job0.standoff_min_m)
            try:
                self.view = sensor_view(self.settings.z_valid_mm, pose_tol_m=e, near_m=near)
            except ValueError as ex:
                raise _Abort(STATUS_REFUSED, f"sensor view: {ex} (nothing sent)", category="config")
        return self.view

    def _depth_spec_m(self) -> float:
        return float(self.spec.groove_depth_mm) * 1e-3 if self.spec is not None else 0.0

    def _opening_spec_m(self) -> float:
        return float(self.spec.top_opening_mm) * 1e-3 if self.spec is not None else 0.0

    def _coarse_range_default(self) -> float:
        return self._sv().coarse_range_m(self._depth_spec_m(), 0.0, self._opening_spec_m())

    @property
    def pass_plate_m(self) -> float:
        """패스의 판 상면 거리 (광축): 설계 깊이 가운데가 광축 교차 거리 (실행 내내 같은 값 — 패스 작업 standoff = 이 값 + 잰 깊이)."""
        return self._sv().pass_plate_m(self._depth_spec_m())

    # ---- 도착 허용을 측정 점으로 (4d P3, 원칙 5) ----
    def _hit_tol_m(self, range_m: float) -> float:
        """e_hit(r): 플랫폼 도착 허용을 광축 거리 r 의 레이저 착점으로 옮긴 값 (레버 = |T_flange←opt 병진| + r) — 착점에서 재는 양
        (측면 오프셋, 프로브 이동, 판 점) 의 '한 번 더 움직여도 이보다 좋아질 수 없는' 크기."""
        lever = float(np.linalg.norm(np.asarray(self.calib.T_flange_opt, dtype=float)[:3, 3])) + max(0.0, float(range_m))
        if self.policy is None:
            return float(self.job0.pose_tol_m)
        return self.policy.tol_at_m(lever)

    def _origin_tol_m(self) -> float:
        """도착 허용을 광학 원점으로 옮긴 값 (레버 = |T_flange←opt 병진|) — standoff (원점 ↔ 판) 가 재는 양."""
        if self.policy is None:
            return float(self.job0.pose_tol_m)
        return self.policy.tol_at_m(float(np.linalg.norm(np.asarray(self.calib.T_flange_opt, dtype=float)[:3, 3])))

    @staticmethod
    def _loc_range_m(loc: StaticLocalization, fallback_m: float) -> float:
        """국소화의 광축 거리 [m] (광축 근처 판 열의 원시 z; 없으면 fallback)."""
        z = (loc.diag or {}).get("raw_z_axis_mm") if loc is not None else None
        return float(z) * 1e-3 if _fin(z) else float(fallback_m)

    def _min_site_distance_m(self) -> tuple:
        """S·E 최소 거리 (4d P3; 원칙 3·5): 두 지점이 레이저 착점에서 e_hit(r_c) 만큼씩 어긋나도 그 직선으로 계획한 패스의 끝 (설계 길이
        L 떨어진 곳) 에서 그루브가 패스 시야의 여유 (FOV(r_pass)/2 − 개구/2 − 판 최소 면: 4a 시야 여유 판정의 측면 여유) 안에 남는 거리
        L_min = 2·e_hit·L / 여유 → (L_min, 근거 dict). 여유가 없으면 inf."""
        sv = self._sv()
        r_c = float(self.job_c.standoff_m) if self.job_c is not None else self._coarse_range_default()
        e = self._hit_tol_m(r_c)
        L = float(self.spec.weld_length_mm) * 1e-3 if (self.spec is not None and self.spec.weld_length_mm > 0) else float("nan")
        room = (0.5 * sv.fov_m(self.pass_plate_m) - 0.5 * self._opening_spec_m()
                - _sfmod.plate_min_len_mm(self.spec) * 1e-3)
        basis = {"e_hit_m": e, "range_m": r_c, "L_ref_m": L, "room_m": room}
        if not (_fin(L) and room > 0.0):
            return math.inf, basis
        return 2.0 * e * L / room, basis

    def _clear_plate(self, job: ScanJob, depth_m: float) -> tuple:
        """패스 작업 standoff 를 툴 ↔ 판 상면 ≥ 한계 + 도착 허용 이 되게 (필요할 때만 광축 교차 거리보다 멀리) → (작업, 올린 양 m)."""
        try:
            return pass_standoff_clear_of_plate(job, self.calib, depth_m, limit_m=float(job.safe_clearance_m) + float(job.pose_tol_m),
                                                range_max_m=self._sv().z_far_m - self._sv().pose_tol_m)
        except JobError as e:
            raise _Abort(STATUS_REFUSED, f"{job.job_id}: {e}", category="motion")

    @property
    def overscan_m(self) -> float:
        """계획 착점 오버스캔 (overscan_need_m: 끝 판정의 확인 프로파일 + 프로파일 한 칸 + 도착 허용)."""
        ov = overscan_need_m(self.job0, self.ghost_burst_frames)[0]
        # 4d P10: 다음 패스의 계획 오버스캔은 실측 착점 간격 (마지막 전체 패스) 이 명목보다 넓으면 그것으로 (최소 오버스캔 — 3G 바닥
        # 거부 — 은 명목 그대로: CPU 부하가 기하 거부가 되지 않게)
        sp = float(((self.last_full or {}).get("extent") or {}).get("spacing_mm", float("nan"))) * 1e-3 if self.last_full else float("nan")
        nominal = float(self.job0.speed_mps) / float(self.job0.profile_fps)
        if _fin(sp) and sp > nominal:
            ov += (max(0, int(self.ghost_burst_frames)) + 2) * (sp - nominal)
        return ov

    @property
    def overscan_min_m(self) -> float:
        """끝 판정이 요구하는 최소 착점 오버스캔 (3G 바닥 규칙이 이보다 줄이면 그 끝은 관측될 수 없다)."""
        return overscan_need_m(self.job0, self.ghost_burst_frames)[1]

    def _support_measured(self) -> bool:
        """3G 지지면을 probe 가 직접 쟀나 (아니면 보인 판의 가장 낮은 점 = 상한뿐; 지지면 정보가 없으면 입력 table_z 를 잰 것으로 본다)."""
        return self.support is None or str(self.support.get("src", "")).startswith("measured")

    @property
    def ghost_burst_frames(self) -> int:
        """센서 유령점 최대 연속 프레임 (sensor.ghost_burst_frames; 안전 정책의 확인 프레임과 같은 원천)."""
        g = self.settings.ghost_burst_frames
        if g is None:
            raise ValueError("platform config: sensor.ghost_burst_frames is required (the safety policy's confirm frames and the end "
                             "detection's confirm profiles)")
        return int(g)

    def _end_params(self, job: ScanJob) -> dict:
        """판 끝 식별 입력 (scan_core.localize.EndParams): k, 용접 경로 단면 창, 끝 구역 지연, 유령점 버스트, 판 최소 면 길이."""
        depth = float(self.spec.groove_depth_mm) * 1e-3 if self.spec is not None else 0.0
        return end_params_for(job, self.calib, self.spec, self._process(), k=self.settings.judgement_k,
                              ghost_burst_frames=self.ghost_burst_frames, depth_m=depth)

    def _coarse_plate_m(self) -> float:
        """거친 단계 보정 목표 (판 법선 거리) = 거친 작업의 광축 거리 × cos(입사 기울기)."""
        jc = self.job_c
        return float(jc.standoff_m) * math.cos(math.radians(float(jc.incidence_tilt_deg)))

    def _derive_jobs(self, job0: ScanJob) -> None:
        """job0 → 거친 작업 job_c (u0, n0), 로컬 이동 규칙의 기준 job_ref, 3G 바닥. _gates 와 범용 시작 (테이블 측정·다시 획득) 이 쓴다."""
        try:
            # 4d C-01: S→E 이동 속도 = 패스 속도 (플랫폼 한계 아래) — 원자료가 패스와 같은 간격
            self.job_c = make_coarse_job(job0, self._sv(), depth_m=self._depth_spec_m(), opening_m=self._opening_spec_m(),
                                         speed_mps=min(float(job0.speed_mps), float(self.cfg.v_eff_max_mps)),
                                         range_m=self._coarse_range_posture if self.orientation == "vertical_3G" else None)
        except JobError as e:
            raise _Abort(STATUS_REFUSED, f"rough job invalid: {e}", category="input")
        self.u0 = self.job_c.unit_dir
        self.n0 = self.job_c.normal_unit
        self._plate_from_sensor = False                  # 새 기준 작업의 판 평면은 거친 착점 평면 — 다음 측정이 바꾼다
        # 로컬 이동 규칙의 기준 작업: 거친 작업의 규칙 + clamp(정규화됨) + 실측 판 상면(나중에 올림)
        self.job_ref = ScanJob.from_dict(job_to_dict(self.job_c))
        self.job_ref.plate_top_measured_m = job0.plate_top_measured_m
        if self.orientation == "vertical_3G":
            # 3G 바닥 (광학 원점·토치 팁의 높이 하한): 첫 패스 전에는 대략 아래 점, 첫 패스 (아래 → 위) 가 시편 아래 끝을 재면 그 끝
            self.floor_h = float(min(self.job_c.height(self.job_c.start), self.job_c.height(self.job_c.end)))
            self.floor_src = "the lower rough point"
            self.job_ref.floor_z_m = self.floor_h

    def _universal_inputs(self) -> None:
        """범용 시작 (사용자 2026-10-07): 입력은 착점 두 개 (job start/end) 의 좌표 + 자세 + 플랫폼 중력뿐. job 의 surface_normal·table_z_m·
        specimen_top_z_m 과 이전 실행의 측정값 (plate_top_measured_m·plate_plane_measured·floor_z_m) 은 버리고 (교시 자세에서 온 값·템플릿
        축·임시 테이블 입력·다른 배치의 측정) 판 법선 추정 n0 = acquisition.initial_frame (1G up, 3G 로봇 쪽 수평). 판 상면은 측정 전에는
        착점 높이 (포락선 기본값), 측정 뒤에는 센서 측정 (I1)."""
        j = self.job0
        try:
            fr = initial_frame(j.start, j.end, self.res.position, j.gravity_axis_base)
        except ValueError as e:
            raise _Abort(STATUS_REFUSED, f"universal start: {e} (nothing sent)", category="input")
        self.ifr = fr
        ignored = []
        for k_ in ("table_z_m", "specimen_top_z_m", "plate_top_measured_m", "floor_z_m"):
            if getattr(j, k_) is not None:
                ignored.append(f"{k_} {float(getattr(j, k_)):+.3f} m")
        if j.plate_plane_measured is not None:
            ignored.append("plate_plane_measured")
        try:
            ignored.append(f"surface_normal {np.round(j.normal_unit, 3).tolist()}")
        except (JobError, ValueError):
            pass
        top = None
        if fr.position == "3G":
            # 세운 판의 위 끝 높이 (3G 계획이 쓴다): 착점은 판 위에 있으므로 위 끝은 위 착점에서 설계 길이 중 두 착점 사이에 없는 만큼을 넘지
            # 않는다 — 입력 (착점·규격) 에서 유도한 보수적 상한 (예전 작업 파일 값·고정 여유 대신)
            up = -np.asarray(j.gravity_axis_base, dtype=float) / float(np.linalg.norm(j.gravity_axis_base))
            hs = sorted((float(up @ j.start), float(up @ j.end)))
            L_design = float(self.spec.weld_length_mm) * 1e-3 if self.spec is not None else 0.0
            top = hs[1] + max(0.0, L_design - float(np.linalg.norm(np.asarray(j.end) - np.asarray(j.start))))
        job = ScanJob.from_dict({**job_to_dict(j), "torch_lateral_m": tuple(j.torch_lateral_m), "surface_normal": fr.n0.tolist(),
                                 "table_z_m": None, "specimen_top_z_m": top, "plate_top_measured_m": None, "floor_z_m": None,
                                 "plate_plane_measured": None})
        self.job0 = job
        self._decide("setup", f"universal start: inputs = the two hit points + position {fr.position} + gravity only; plate normal guess "
                              f"n0 {np.round(fr.n0, 4).tolist()} ({'up' if fr.position == '1G' else 'horizontal toward the robot, measured at the first dwell'})"
                     + (f"; standing plate top <= {top:.4f} m (upper hit + the design length not between the hits)" if top is not None else "")
                     + (f"; ignored from the job file: {', '.join(ignored)}" if ignored else ""),
                     n0=fr.n0.tolist(), u0=fr.u0.tolist(), ignored=ignored)

    def _sensor(self) -> None:
        if self.sensor_ready:
            return
        ensure_sensor(self.sess, self.settings, sensor_ip=self.conn["sensor_ip"], sensor_port=self.conn["sensor_port"],
                      sensor_init=self.sensor_init, backup_dir=str(self.out / "sensor_backup"), log=self.log)
        self.sensor_ready = True

    # ---- 확인 · 로컬 이동 -----------------------------------------------------------------------
    def _confirm(self, name: str, dist_m: float, extra: str = "", rot_deg: float = 0.0, kind: str = "local", plan_text=None) -> None:
        """dist_m·rot_deg = 시작 → 끝 플랜지 현과 회전. plan_text = 계획 설명 (문자열, 또는 묻는 이동에서만 부르는 함수 — move_plan_text)."""
        # 4d P10: 'large' 문턱은 툴 점의 시작 → 끝 순변위 상한 (플랜지 현 + 툴 레버 × 회전 — 레버 반경 안의 점 p 는 |Δp| ≤ |Δt| +
        # |r|·θ) 과 비교 — 제자리 회전도 툴 끝은 움직인다 (판단 그대로). 경로 길이의 상한은 아니다 (경로는 순변위보다 길 수 있다).
        # 2026-10-08 g1_b: 그 값을 'move 1893.0 mm' 로 보여 이동 거리처럼 읽혔다 (계획한 플랜지 경로 949.6 mm, 현 929.2 mm) — 프롬프트는
        # 계획을 말하고, 판단 값은 'large' 모드에서만 무엇인지 밝혀 덧붙인다 (그때만 묻는 이유다)
        lever = float((self.policy.basis or {}).get("lever_m", 0.0)) if self.policy is not None else 0.0
        chord_m, rot = float(dist_m), abs(float(rot_deg or 0.0))
        dist_m = chord_m + lever * math.radians(rot)
        self._event("confirm", name=name, dist_m=dist_m)
        thr = float(self.policy.clearance_m) if self.policy is not None else 0.0
        text = None
        if plan_text is not None and confirm_needed(self.confirm_mode, dist_m, thr, kind):
            text = str(plan_text() if callable(plan_text) else plan_text)
            if self.confirm_mode == "large":
                text += (f" [asked because a tool point up to {lever * 1e3:.0f} mm from the flange can end up to {dist_m * 1e3:.1f} mm "
                         f"from where it starts (start-to-end bound = flange chord {chord_m * 1e3:.1f} mm + {lever * 1e3:.0f} mm x "
                         f"{rot:.1f} deg; not a path length) > the 'large' threshold {thr * 1e3:.1f} mm]")
        ok = confirm_move(self.confirm_mode, name, dist_m, source=self.confirm_source, log=self.log, extra=extra,
                          threshold_m=thr, kind=kind, text=text)
        if not ok:
            raise _Abort(STATUS_ABORTED, f"operator declined move '{name}'" + (f" ({text})" if text else f" ({dist_m * 1e3:.1f} mm)"))

    def _check_estop(self, where: str) -> None:
        m = self.rs.backend
        if m is not None and m.estopped:                    # 액션 취소의 정지
            raise _Abort(STATUS_ABORTED, f"stopped {where}: {m.latch.reason}", category="motion")

    def _check_stop(self, where: str) -> None:
        """The run's stop inside long planning loops (the joint-path search, KIRO field 2026-10-14: a cancel 8.5 s into run 134602 did
        not reach 106 s of planning): the run's cancel (a subclass's _check_cancel) and the latch (_check_estop)."""
        cc = getattr(self, "_check_cancel", None)
        if callable(cc):
            cc()
        self._check_estop(where)

    def _record_move(self, name: str, phase: str, T_from: np.ndarray, T_to: np.ndarray, stats: dict, speed: float,
                     res=None, refused: Optional[str] = None) -> dict:
        d = {"name": name, "phase": phase, "from_m": T_from[:3, 3].tolist(), "to_m": T_to[:3, 3].tolist(),
             "from_R": np.asarray(T_from)[:3, :3].tolist(), "to_R": np.asarray(T_to)[:3, :3].tolist(),
             "dist_mm": float(np.linalg.norm(T_to[:3, 3] - T_from[:3, 3])) * 1e3,
             "rot_deg": _rot_angle_deg(T_from[:3, :3], T_to[:3, :3]), "speed_mps": speed,
             "envelope_stats": {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in stats.items()},
             "refused": refused, "executed": res is not None and getattr(res, "ok", False),
             "motion": res.as_dict() if res is not None else None, "t": round(self._clock.monotonic() - self.t0, 3)}
        self.res.moves.append(d)                 # 같은 dict 를 보관 (실행 결과가 나중에 채워진다; to_dict 가 jsonable 로 바꾼다)
        return d

    def _move_geometry_text(self, T_from: np.ndarray, T_to: np.ndarray, stats: dict) -> str:
        """확인 프롬프트용 이동 기하 (M-11): 변위, 방향(면내/법선), 회전, 결과 키포인트의 실측 판 위 높이(포락선 통계)."""
        d = np.asarray(T_to, dtype=float)[:3, 3] - np.asarray(T_from, dtype=float)[:3, 3]
        d_n = float(d @ self.n0)
        d_ip = float(np.linalg.norm(d - self.n0 * d_n))
        parts = [f"from {np.round(T_from[:3, 3], 4).tolist()} to {np.round(np.asarray(T_to)[:3, 3], 4).tolist()} m",
                 f"in-plane {d_ip * 1e3:.1f} mm, along-normal {d_n * 1e3:+.1f} mm ({'toward' if d_n < -1e-6 else 'away from' if d_n > 1e-6 else 'level with'} the plate)",
                 f"rot {_rot_angle_deg(T_from[:3, :3], np.asarray(T_to)[:3, :3]):.2f} deg"]
        if stats:
            if self.orientation == "flat_1G":
                top = stats.get("plate_top_m")
                if top is not None and _fin(top):
                    src = "measured" if self.plate_top is not None else "rough"
                    parts.append(f"min height above the {src} plate top {float(top):.4f} m: tip "
                                 f"{(float(stats.get('min_tip_height', float('nan'))) - float(top)) * 1e3:.1f} mm, optical origin "
                                 f"{(float(stats.get('min_opt_height', float('nan'))) - float(top)) * 1e3:.1f} mm")
            else:
                src = "measured" if self.plate_plane_measured is not None else "rough"
                parts.append(f"min distance to the {src} plate plane: tip {float(stats.get('min_plate_dist_tip', float('nan'))) * 1e3:.1f} mm, "
                             f"optical origin {float(stats.get('min_plate_dist_opt', float('nan'))) * 1e3:.1f} mm")
        return "; ".join(parts)

    def _timeout_check(self, name: str, dist_m: float, v_target: float) -> None:
        """F-C9 fail-early (확인 프롬프트 전): 명시 --motion-timeout 이 이 구간의 2·거리/v_eff + 10 s 보다 짧으면 REFUSED.
        v_eff = 이 세션의 속도 정책·바(5001 default_speed) 로 계획한 실효 속도. 유도 타임아웃(None) 이면 검사 없음."""
        if self.motion_timeout_s is None or self.cfg is None:
            return
        v_eff = self.rs.gates.effective_speed(self.rs, self.cfg, float(v_target))
        why = motion_timeout_shortfall(self.motion_timeout_s, dist_m, v_eff, name)
        if why:
            raise _Abort(STATUS_REFUSED, why + " (nothing sent)", category="motion")

    def _check_start_pose(self, T_planned: np.ndarray, name: str) -> None:
        """C8: 확인 프롬프트(블로킹) 뒤 실측 자세가 포락선 검사의 시작 자세와 다르면 REFUSED (스크립트 없음)."""
        why = start_pose_mismatch(T_planned, self._T_now(), self.policy.goal_tol_m, self.policy.goal_tol_deg)
        if why:
            raise _Abort(STATUS_REFUSED, f"move '{name}': {why} (nothing sent)", category="motion")

    def _local_move(self, T_to: np.ndarray, phase: str, name: str, *, speed: Optional[float] = None, streaming: bool = False,
                    extra: str = "", kind: Optional[str] = None, feeder=None) -> tuple:
        """현재 실측 자세 → T_to: 포락선 검사(거부 = REFUSED) → 확인 → 시작 자세 재확인(C8) → move_l. 반환 (MotionResult, records, samples).

        streaming=True (패스): 녹화 프레임·로봇 샘플을 돌려주고 _last_stream 에 보관한다 (부분 저장용).
        streaming=False: 센서를 켜지 않는 이동.

        kind: 확인 종류 (confirm_needed). None = 예전 규칙 (streaming 이면 'scan', 아니면 'local') — 센서 녹화 구간 (S_track·S_track2·
        트랜싯·패스) 은 모두 'scan' ('scan' 모드 정의: 녹화 구간을 묻는다, CONFIRM_KINDS_SCAN). feeder: 녹화 중 프레임을 작업 프로세스에 넘기는 스레드 (post_scan.SeedFeeder —
        패스 판 성장, FeatureFeeder — S 추적·트랜싯 특징); None = self._seed_feeder (예전)."""
        feeder = self._seed_feeder if feeder is None else feeder
        assert self.job_ref is not None
        self._last_stream = None                 # M-3: 이전 구간의 스트림을 이번 구간의 부분 데이터로 오인하지 않는다
        self._check_estop(f"before move '{name}'")
        T_from = self._T_now()
        v = float(self.local_speed if speed is None else speed)
        try:
            plan, stats = make_local_move_plan(T_from, T_to, phase, self.job_ref, self.calib, specimen_top_z_m=self.plate_top,
                                               speed_mps=v, accel_mps2=self.accel, name=name)
        except (EnvelopeError, JobError, ValueError) as e:
            self._record_move(name, phase, T_from, np.asarray(T_to, dtype=float), {}, v, refused=str(e))
            raise _Abort(STATUS_REFUSED, f"local move '{name}' ({phase}) rejected by the envelope: {e}", category="motion")
        d = self._record_move(name, phase, T_from, np.asarray(T_to, dtype=float), stats, v)
        self._decide("move", f"{name} ({phase}): {d['dist_mm']:.1f} mm rot {d['rot_deg']:.2f} deg envelope OK "
                     f"(min_tip_h {stats.get('min_tip_height', float('nan')):.4f} m, plate_top {stats.get('plate_top_m')})")
        try:
            self._timeout_check(name, d["dist_mm"] * 1e-3, v)              # F-C9: 프롬프트 전에
        except _Abort as e:
            d["refused"] = e.reason
            raise
        geo_txt = self._move_geometry_text(T_from, T_to, stats)
        w = plan.waypoints[1]

        def plan_text() -> str:
            # 묻는 이동만, 확인 문구용으로 지금 자세에서 계획해 본다 (check_leg: 보내지도 저장하지도 않는다). 실행할 계획은 늘 아래에서
            # 프롬프트 뒤·센서를 켜기 직전의 자세로 만든다: 실행 계획의 시작 = 녹화 직전의 팔 상태여야 한다. 프롬프트 전 시점은 직전
            # 이동이 막 끝난 때라 팔이 아직 수렴 중이다 — 실기 기록 10/658 건 'prepared plan not used: the arm moved 0.10–0.20 mm' 가 모두
            # 직전 이동 바로 뒤에 한 계획이었고, 백엔드가 녹화 중에 다시 계획해 3 건은 /joint_states 처리가 0.23–0.32 s 밀렸다 (at09/at10
            # G3 거짓 정지와 같은 GIL 기제). 그 계획을 프롬프트 뒤까지 들고 가면 프롬프트 동안 수렴이 다 끝나 낡는 것이 보통이 된다
            chk = check_leg(self.sess, w, v, timeout_s=self.motion_timeout_s)
            ok = chk is not None and chk["ok"] and bool(chk.get("plan"))
            return move_plan_text(chk["plan"] if ok else None, chord_m=d["dist_mm"] * 1e-3, rot_deg=d.get("rot_deg", 0.0),
                                  refused=None if chk is None or ok else (chk.get("reason") or "the planner returned no plan"))
        self._confirm(name, d["dist_mm"] * 1e-3, extra=(extra + " " if extra else "") + f"[{geo_txt}]", rot_deg=d.get("rot_deg", 0.0),
                      kind=kind or ("scan" if streaming else "local"), plan_text=plan_text)
        self._check_start_pose(T_from, name)
        self._event("move", name=name, phase=phase, streaming=streaming)
        records: list = []
        samples: list = []
        t_rec = 0
        if streaming:
            prepare_leg(self.sess, w, v, timeout_s=self.motion_timeout_s)       # 센서를 켜기 직전의 자세에서 계획 (녹화 중 계획 계산 없음)
            t_io = float(self.at.first_frame_timeout_s)
            try:
                # S1: 센서가 시작하지 않으면 (4d C-15) RESET 뒤 한 번 더 — 두 번째도 프레임이 없을 때만 이 구간은 보내지 않고 REFUSED
                # (아무것도 움직이지 않았다)
                try:
                    t_rec = start_sensor_recording(self.sess, first_frame_timeout_s=t_io)
                except SensorNotStarted as e1:
                    self._decide("sensor", f"move '{name}': {e1} - sensor RESET done, starting once more")
                    t_rec = start_sensor_recording(self.sess, first_frame_timeout_s=t_io)
                t_first = self._clock.monotonic()                           # 첫 프레임을 받은 뒤
            except SensorNotStarted as e:
                d["refused"] = str(e)
                raise _Abort(STATUS_REFUSED, f"move '{name}' ({phase}): {e} (twice, with a sensor RESET between)", category="sensor")
            if feeder is not None:
                # 패스: 녹화되는 프레임의 판 성장 (S 추적·트랜싯: 그루브 특징) 을 작업 프로세스가 이동 중에 계산한다 (녹화 목록을 읽기만 함;
                # 실패해도 이동과 무관)
                feeder.start(self.ss.records)
            # 데이터 구간 (패스·S→E): 움직이기 전 정지 프레임도 자료다 — 첫 끝 너머의 증거 (지지면 끝 규칙, g3_07/08). 프레임 몇 개만
            # 기다리면 가상 팔 3G 패스 1 의 아래 끝 판정이 흔들렸다 (2026-10-08) — 첫 프레임부터 PASS_PREROLL_S 까지
            self._clock.sleep(max(0.0, PASS_PREROLL_S - (self._clock.monotonic() - t_first)))
            self._check_estop(f"during sensor warm-up before move '{name}'")
        sent = True
        self._motion_sent = True                      # 이 실행에서 로봇에 이동을 보냈다 (_abort_path: 정지 명령 판단)
        try:
            # C8 후속: 시작 자세 검사를 전송 바로 앞(센서 워밍업 뒤) 에서도 — 프롬프트 뒤 워밍업 창의 조그까지 잡는다
            res = execute_leg(self.sess, w, v, timeout_s=self.motion_timeout_s,
                              expected_start=T_from, start_tol=(self.policy.goal_tol_m, self.policy.goal_tol_deg))
        except StartPoseMismatch as e:
            sent = False
            if streaming:
                self._discard_stream()
            d["refused"] = str(e)
            raise _Abort(STATUS_REFUSED, f"move '{name}': {e} (nothing sent)", category="motion")
        except MotionError as e:
            if streaming:
                records = self._stop_stream()
                self._last_stream = (records, self.rs.pose.history(t_rec), None)
            d["motion"] = {"error": str(e)}
            raise _Abort(STATUS_ABORTED, f"move '{name}': {e}", safety=True, category="motion")
        finally:
            if sent:
                self.res.n_move_l += 1
        if streaming:
            records = self._stop_stream()
            samples = self.rs.pose.history(t_rec)
            self._last_stream = (records, samples, res)
        d["motion"] = res.as_dict()
        d["executed"] = bool(res.ok)
        if not res.ok:
            raise _Abort(STATUS_ABORTED, f"move '{name}': {res.reason}", safety=True, category="motion")
        self._p(f"      -> OK {res.reason} v_cmd={res.v_cmd:.1f} mm/s duration={res.motion_duration_s:.2f} s")
        return res, records, samples

    def _discard_stream(self) -> None:
        """스트리밍 창을 닫고 프레임을 버린다 (스캔 데이터가 아니다)."""
        self._stop_stream()
        try:
            self.ss.records = []
        except Exception:  # noqa: BLE001
            pass

    def _stop_stream(self) -> list:
        try:
            return stop_sensor_recording(self.sess, settle_s=0.1)
        except Exception as e:  # noqa: BLE001
            self.res.notes.append(f"sensor stop failed: {e}")
            return list(self.ss.records)

    # ---- 드웰 · 국소화 ------------------------------------------------------------------------
    def _grab(self, name: str, n_frames: Optional[int] = None) -> tuple:
        """정지 grab → (x_med, z_med, T_bf, stat, records). grab_k.npz 저장 + delta_cb 기록."""
        self._check_estop(f"before dwell '{name}'")
        n = int(self.dwell_frames if n_frames is None else n_frames)
        # 기다림 = 첫 프레임 대기 + 프레임 n 개 시간 + 센서 데이터 대기 (4d P5; 예전 max(5, 0.05 n + 3) s)
        fps = float(self.job0.profile_fps) if _fin(self.job0.profile_fps) and float(self.job0.profile_fps) > 0 else 1.0
        timeout = float(self.at.first_frame_timeout_s) + n / fps + float(self.settings.sensor.get("data_wait_ms", 1000) or 0) * 1e-3
        need = 2 * int(self.policy.confirm_frames if self.policy is not None else 1) - 1      # 유령점 버스트를 이기는 최소 (2 버스트 + 1)
        recs = []
        for attempt in (1, 2):
            recs = self.ss.grab(n, timeout_s=timeout)
            if len(recs) >= need:
                break
            if attempt == 1:
                self._decide("dwell", f"dwell '{name}': only {len(recs)} profile frames (< {need}) - grabbing again in place (confirm-first; "
                                      "the arm stands still)")
        snap = self.rs.pose.snapshot()
        T_bf = snap.T
        if len(recs) < need:
            raise _Abort(STATUS_ABORTED, f"dwell '{name}': only {len(recs)} profile frames from the sensor twice (< {need}, timeout "
                                         f"{timeout:.1f} s)")
        x_med, z_med, n_valid = median_profile(recs)
        self.n_grab += 1
        k = self.n_grab
        dcb = self.rs.gates.frame_offset_record(self.rs, snap)        # 그 자세의 플랫폼 기록 (관절)
        dcb.pop("drift_checked", None)
        dcb.update(dwell=name, k=k, pose_source=self.pose_source)
        self.res.delta_cb.append(dcb)
        path = self.out / f"grab_{k}.npz"
        try:
            np.savez_compressed(path, x_med=x_med, z_med=z_med, T_bf=T_bf, **snap.arrays, n_frames=len(recs), n_valid_cols=n_valid,
                                name=np.array(name), calib_json=np.array(json.dumps(self.calib.to_dict())))
            self.res.files.append(str(path))
        except Exception as e:  # noqa: BLE001
            self.res.notes.append(f"grab save failed: {e}")
        self._check_estop(f"after dwell '{name}'")
        return x_med, z_med, T_bf, snap, recs

    def _localize(self, x_med, z_med, T_bf, name: str, *, confirm: bool = True, scene="judge") -> StaticLocalization:
        """정지 측정 → 국소화 + 안전 바닥 갱신 + 장면 판단. scene (P2 4.7 절): 'judge' (예전 — 물체·설계 불일치는 같은 자리 다시 잡기로 확인한
        뒤 멈춘다) | 'record' (레이저 선이 시편 위에 있다는 것이 아직 측정으로 확인되지 않은 드웰: 판정은 기록만 — 다시 잡기·중단 없음, 다음
        드웰이 다시 판단한다; 확인 먼저·래치 없음) | 국소화 → 둘 중 하나를 정하는 함수 (그 드웰의 선 발자국이 측정으로 확인된 시편 구간
        안인가). 안전 바닥 (1G 판 상면, 3G 실측 판 평면 — 올리기만) 은 장면 판단과 따로 **모든** 드웰의 측정으로 갱신한다: 올리기만이라 판단하지
        않은 읽기가 더 높게 읽어도 포락선이 보수적으로 될 뿐이고, 기록만 하는 드웰 뒤의 S 추적·probe X·S 정렬도 실측 바닥으로 포락선 검사를
        받는다. 판 위 물체가 툴을 위협하는 경우는 가드 G1 이 계속 맡는다 (안전 정지, 즉시)."""
        n_hint = self.up if self.orientation == "flat_1G" else None
        loc = localize_static(T_bf, self.calib, x_med, z_med, self.spec, gravity_axis_base=self.job0.gravity_axis_base,
                              travel_dir_hint=self.u0, plate_normal_hint=n_hint, z_valid_mm=self.settings.z_valid_mm,
                              features_kw={"qc_targets": self.qc_targets})
        cls = classify_localization(loc)
        mode = scene(loc) if callable(scene) else str(scene)
        if mode not in ("judge", "record"):
            raise ValueError(f"scene mode must be 'judge' or 'record', got {mode!r}")
        loc.diag["scene_mode"] = mode
        self.res.localizations.append({"name": name, "k": self.n_grab, "class": cls, "scene_mode": mode, **loc.to_dict()})
        self._decide("localize", f"{name}: {loc.summary()} class={cls}"
                     + ("" if mode == "judge" else " [scene recorded only - the laser line is not yet measured to be on the specimen]"))
        self._raise_plate_top(loc)                       # 안전 바닥 (올리기만): 장면 판단과 따로, 모든 드웰
        if mode == "record":
            problem = self._check_dwell_scene(x_med, z_med, loc, cls, name, record_only=True)
            if problem is not None:
                loc.diag["scene_deferred"] = problem
            return loc
        problem = self._check_dwell_scene(x_med, z_med, loc, cls, name)
        if problem is not None:
            if not confirm:
                site = name[:-len("_regrab")] if name.endswith("_regrab") else name
                raise _Abort(STATUS_ABORTED, problem.replace(f"{name}:", f"{site}:", 1) + " (confirmed by a second grab in place)")
            # 확인 먼저 (4d P5): 같은 자세에서 한 번 더 잡는다 (흔들리는 케이블·지나가는 가림) — 다시 걸리면 멈추고, 깨끗하면 그 읽기만 쓴다
            self._decide("scene", f"{problem} - grabbing again in place before stopping (confirm-first)")
            x2, z2, T2, _st, _r = self._grab(f"{name}_regrab")
            return self._localize(x2, z2, T2, f"{name}_regrab", confirm=False)
        return loc

    def _check_dwell_scene(self, x_med, z_med, loc: StaticLocalization, cls: str, name: str, record_only: bool = False) -> Optional[str]:
        """정지 측정의 장면 검사 (2026-10-05 at21: 늘어진 센서 케이블이 레이저 선 위 판 위 ~60 mm 에 걸려, 추출기가 그 그림자 사이를 개구
        53 mm '그루브' 로 잡고 정렬이 엉뚱한 곳에 수렴 → 이송 3° 틀어짐 → G4 정지). 문제가 있으면 그 설명을 돌려준다 (_localize 가 같은
        자세에서 한 번 더 잡아 확인하고, 다시 걸리면 ABORTED — 4d P5):
          (a) 이물: 판 상면보다 설계 깊이의 절반 (원칙 2: '판 수준' 과 '그루브 깊이만큼 위' 의 중간점) 넘게 높은 점이 추출기가 판 하나로
              인정할 만큼 (판 최소 면의 증거 개수, 이 드웰의 열 간격) — 용접 전 빈 그루브에는 판 상면보다 높은 표면이 없다 (모델).
          (b) 설계 일관성: 그루브로 분류됐으면 상부 개구·깊이가 설계 공칭 ± 절반 (measure.design_match — 패스의 '시편 위 판정' 과 같은 밴드) 안이어야
              한다 — 밖이면 설계 그루브가 아니다 (이물·그림자·설계 정보 오류). 갭은 쓰지 않는다: 비스듬히 볼 때 직각 벽 그늘로 바닥이
              가리면 불안정하다 (실측 정지 측정 103 개 재생: at15·at18 의 정상 probe_X 갭 10.6 mm; 개구·깊이는 정상 측정에서 밴드 안)."""
        # P2: 예전 3G '비스듬한 첫 드웰 장면 판단 미룸' 은 _localize 의 scene='record' 로 일반화했다 (첫 드웰·다시 획득 드웰은 늘 기록만).
        f = loc.features
        tag = (" [recorded only - judged at a dwell whose laser line is measured to be on the specimen]" if record_only else "")
        sp = self.spec
        if f is not None and sp is not None and float(sp.groove_depth_mm) > 0.0:
            h_obj = _ms.DESIGN_HALF * float(sp.groove_depth_mm)
            dx = float(getattr(f, "dx_mm", float("nan")))
            n_obj = _ms.evidence_count(_sfmod.plate_min_len_mm(sp) / dx) if (_fin(dx) and dx > 0.0) else 1
            obj = points_above_plates(x_med, z_med, f, h_obj)
            if obj["n"] >= n_obj:
                msg = (f"{name}: object above the plate in the laser line - {obj['n']} points (>= {n_obj}) up to {obj['max_height_mm']:.0f} mm "
                       f"above the plate top (> {h_obj:.1f} mm = half the design depth) at sensor x {obj['x_min_mm']:+.0f}.."
                       f"{obj['x_max_mm']:+.0f} mm (a cable hanging in front of the sensor, a tool or a fixture?); remove it and run again - "
                       "alignment stopped before correcting on it")
                self._decide("scene", msg + tag, object=obj, deferred=bool(record_only))
                return msg
        if cls != "groove" or sp is None or f is None:
            return None
        rows = []
        for label, val, nom in (("opening", f.top_opening_mm, sp.top_opening_mm), ("depth", f.depth_mm, sp.groove_depth_mm)):
            if np.isfinite(val) and nom > 0.0 and not _ms.design_match(val, nom):
                rows.append(f"{label} {val:.1f} mm (design {nom:.1f})")
        if rows:
            off = rough_surface_offset_mm(loc.plate_points_base, (self.job0.start, self.job0.end), self.n0 if self.n0 is not None else
                                          self.job0.normal_unit, loc.axis_hit_base if np.all(np.isfinite(loc.axis_hit_base)) else None)
            why = ("a foreign object or shadow in the laser line, the rough point off the specimen, or wrong design data in the job "
                   "(specimen block)")
            # 본 면이 대략 점보다 판 두께의 절반 넘게 아래 (판 상면과 한 판 아래 면의 중간점) 또는 설계의 가장 깊은 면보다 위
            # (깊이 × (1 + ½)) 이면 시편 밖을 보고 있다는 힌트 (메시지만)
            below, above = -_ms.DESIGN_HALF * float(sp.thickness_mm), float(sp.groove_depth_mm) * (1.0 + _ms.DESIGN_HALF)
            if np.isfinite(off) and (off < below or off > above):
                why += (f"; the surface seen is {abs(off):.0f} mm {'below' if off < 0 else 'above'} the rough point - if it was taught on "
                        f"the plate top, the laser is not on the specimen here (rough point beyond a plate end? teach it a little inside)")
            msg = (f"{name}: the groove found does not match the design within +-{_ms.DESIGN_HALF * 100:.0f} %: {', '.join(rows)} - {why}; "
                   f"alignment stopped")
            self._decide("scene", msg + tag, deferred=bool(record_only))
            return msg
        return None

    def _dwell(self, name: str, n_frames: Optional[int] = None, scene="judge") -> StaticLocalization:
        x_med, z_med, T_bf, _st, _recs = self._grab(name, n_frames)
        return self._localize(x_med, z_med, T_bf, name, scene=scene)

    def _raise_plate_top(self, loc: StaticLocalization) -> None:
        """I1 바닥 갱신 (올리기만). 1G: 실측 판 상면 높이 → job_ref.plate_top_measured_m. 3G (C6/S8/S9): 실측 판 평면
        (점 = 중앙 판 점을 잰 법선 방향 가장 로봇 쪽 판 인라이어 수준으로 올린 점, 법선 = plate_normal_base) →
        job_ref.plate_plane_measured (raise-only) + job_ref.start/end 를 그 수준으로 n0 방향 이동 (로봇 쪽으로만) →
        never_cross_plate_plane / clamp(3G) / make_local_move_plan 이 실측 평면을 쓴다."""
        if not loc.plate_ok:
            return
        if self.orientation == "flat_1G":
            if not _fin(loc.plate_top_height_m):
                return
            h = float(loc.plate_top_height_m)
            if self.plate_top is None or h > self.plate_top:
                self.plate_top = h
                self.job_ref.plate_top_measured_m = h
                self._decide("floor", f"measured plate top (up.P, tilt-aware) -> floor raised to {h:.4f} m (I1)")
            return
        # ---- vertical_3G: 실측 판 평면 ----
        P = np.asarray(loc.plate_points_base, dtype=float).reshape(-1, 3)
        P = P[np.all(np.isfinite(P), axis=1)]
        n_m = np.asarray(loc.plate_normal_base, dtype=float).reshape(3)
        if P.shape[0] == 0 or not np.all(np.isfinite(n_m)):
            return
        n_m = unit(n_m)
        if float(n_m @ self.n0) < 0.0:
            n_m = -n_m                                   # 로봇 쪽 반구로
        start = np.asarray(self.job_ref.start, dtype=float)
        # S9 → 4d P5: 판 수준 = 추출기 판 인라이어 (이미 추출기 띠 안) 의 가장 로봇 쪽 점. 실측 평면은 **잰 법선** n_m 방향 최대
        # (추정 n0 가 10–35° 비껴도 판의 펼침이 수준에 섞이지 않는다), 포락선 평면 (법선 n0) 은 n0 방향 최대 (그 평면이 판 점을 모두 담게).
        dP = (P - start) @ self.n0
        lvl = float(np.max(dP))
        p_c = np.median(P, axis=0)
        p = p_c + n_m * float(np.max((P - p_c) @ n_m))
        shift = lvl                                      # 판 인라이어의 n0 방향 최대가 job_ref 평면보다 로봇 쪽(+n0) 에 있으면 > 0
        first = not self._plate_from_sensor              # 거친 착점 평면은 측정이 아니다 (I1: 거친 높이는 첫 접근에만) — 첫 측정은 그대로 채택
        if shift > 0.0 or first:
            # 포락선 평면(job_ref.start/end, 법선 n0) 을 실측 평면 위로 — 첫 측정 뒤로는 로봇 쪽으로만 (raise-only). 멀어지는 측정은 바닥을 내리지 않는다.
            kw = job_to_dict(self.job_ref)
            kw["torch_lateral_m"] = tuple(kw["torch_lateral_m"])
            kw.update(start=(start + self.n0 * shift).tolist(),
                      end=(np.asarray(self.job_ref.end, dtype=float) + self.n0 * shift).tolist())
            job_new = ScanJob(**kw)
            job_new.plate_top_measured_m = self.job_ref.plate_top_measured_m
            self.job_ref = job_new
        # S8: job_ref 에도 실측 판 평면 (raise-only) → check_envelope 판 평면 규칙이 거친 법선 평면과 실측 평면 중 가까운 쪽을 쓴다
        self.job_ref.plate_plane_measured = merge_plate_plane(None if first else self.job_ref.plate_plane_measured, (p, n_m))
        self._plate_from_sensor = True
        self.plate_plane_measured = self.job_ref.plate_plane()
        self._decide("floor", f"measured plate plane (3G): point {np.round(p, 4).tolist()} normal {np.round(n_m, 4).tolist()} -> "
                              f"job_ref plane {'shifted by %+.1f mm along n0' % (shift * 1e3) if (shift > 0.0 or first) else 'kept (measured plane %+.1f mm, not closer)' % (shift * 1e3)} "
                              f"({'first measurement replaces the rough hit-point plane' if first else 'raise-only'}, I1)")

    # ---- 2. 접근 --------------------------------------------------------------------------------
    def _tilt_limit_deg(self) -> float:
        """3G 입사 기울기 상한: 센서·모형 한계 (job.max_laser_tilt_deg, 깊이 축 모형의 INCIDENCE_TILT_MAX_DEG)."""
        from scan_core.types import INCIDENCE_TILT_MAX_DEG
        return min(float(INCIDENCE_TILT_MAX_DEG), float(self.job0.max_laser_tilt_deg))

    def _choose_3g_end_postures(self, job0: ScanJob) -> list:
        """범용 시작 3G (사용자 2026-10-07: 테이블·지그에 부딪히지 않게 위에서 내려다보되, 각도는 하드 값이 아니라 적응적으로): 잰 지지면
        (job0.table_z_m) 으로 거친 단계·패스 1 (끝 패스) 의 자세 (입사 기울기 0..센서 한계 × 토치 위/아래) 를 모두 보고, 제약 = 툴 몸체
        전체가 지지면 + 한계 + 도착 허용 위 + 측정 전 3G 바닥 (광학 원점·토치 팁 ≥ 아래 착점) + 센서 거리 창·레이저 기울기, 점수 = 예상 품질
        (scan_quality: 입사 가중 × φ 예상 좋은 비율 — 이번 실행 측정 > 데이터셋 > 사전값) × 아래 끝 도달 비율. 반환: 좋은 순서 (도달성은
        호출자가 IK 로 본다)."""
        measured = self._measured_phis()
        pos = str(self.res.position or "3G")
        ds = self.at.quality_dataset_path or None

        def qf(jp):
            post = _sq.pass_posture(jp, self.calib)
            pred, g, why = _sq.predicted_score(post, pos, ds, measured)
            return {"pred": float(pred), "pred_why": f"phi {post['phi_deg']:+.1f} incidence {post['incidence_deg']:.0f}: {why}",
                    "torch_above": bool(post.get("torch_above")), "phi_deg": float(post["phi_deg"]),
                    "incidence_deg": float(post["incidence_deg"])}
        try:
            ch = choose_3g_end_postures(job0, self.calib, self.at, self._sv(), self.spec, quality_fn=qf,
                                        t_max_deg=self._tilt_limit_deg(), ghost_burst_frames=self.ghost_burst_frames,
                                        support_measured=self._support_measured())
        except (JobError, ValueError) as e:
            raise _Abort(STATUS_REFUSED, f"3G posture choice failed: {e} (nothing sent)")
        ranked = ch.get("ranked") or []
        top = "; ".join(f"{r['tilt_deg']:.0f}deg torch {'above' if r['torch_above'] else 'below'}: pred {r['pred']:.2f} x reach "
                        f"{r['coverage'] * 100:.0f}% = {r['score']:.2f} (margins E {r['margin_E_m'] * 1e3:+.0f}/pass1 "
                        f"{r['margin_pass1_m'] * 1e3:+.0f} mm)" for r in ranked[:6])
        self._decide("posture", f"3G end posture candidates (support {float(job0.table_z_m):.4f} m + limit "
                                f"{(job0.safe_clearance_m + job0.pose_tol_m) * 1e3:.0f} mm, "
                                f"whole tool body; tilt 0..{self._tilt_limit_deg():g} deg x torch side; {len(ranked)} clear the support at "
                                f"the lower dwell): {top or 'none'}",
                     candidates=_jsonable([{k_: v for k_, v in r.items() if k_ != "pred_why"} for r in ranked[:12]]))
        if not ranked:
            raise _Abort(STATUS_REFUSED, f"3G: no posture (look-down tilt 0..{self._tilt_limit_deg():g} deg, torch above/below) keeps the "
                                         f"whole tool above the measured support {float(job0.table_z_m):.4f} m + margin at the lower hit "
                                         "point - the lower hit point is too close to the support for this tool")
        return ranked

    def _posture_normal_tol_deg(self) -> float:
        """3G 끝 자세가 견디는 판 법선 차 [deg] (choose_3g_end_postures 의 normal_tol_deg). 자세를 아직 안 골랐으면 0 (= 잰 법선이
        조금이라도 다르면 다시 고른다)."""
        return float(self._normal_tol_deg) if self._normal_tol_deg is not None else 0.0

    def _apply_3g_end_posture(self, job0: ScanJob, r: dict) -> ScanJob:
        """끝 자세 하나를 job0 에 (입사 기울기) + 거친 진행 부호. 주 패스 (2+) 는 _main_posture 가 품질로 다시 고른다 (같을 수도)."""
        t = float(r["tilt_deg"])
        self._tilt_sign = int(r["sign"])
        self._normal_tol_deg = float(r.get("normal_tol_deg", 0.0))
        self._coarse_range_posture = r.get("coarse_range_m")
        self.tilt_end = self.tilt_main = t
        self.main_posture = None
        j = ScanJob.from_dict({**job_to_dict(job0), "torch_lateral_m": tuple(job0.torch_lateral_m), "incidence_tilt_deg": t})
        j.plate_top_measured_m = job0.plate_top_measured_m
        self._decide("posture", f"3G end posture: look down {t:.0f} deg, torch {'above' if r['torch_above'] else 'below'} the laser "
                                f"(progression {int(r['sign']):+d}): predicted {r['pred']:.2f} ({r['pred_why']}), the laser reaches "
                                f"{r['coverage'] * 100:.0f} % of the way to the lower target above the support; margins E "
                                f"{r['margin_E_m'] * 1e3:+.0f} mm, pass 1 {r['margin_pass1_m'] * 1e3:+.0f} mm; tolerates a plate normal "
                                f"{r.get('normal_tol_deg', 0.0):.1f} deg off the estimate; coarse range "
                                f"{r.get('coarse_range_m', float('nan')) * 1e3:.0f} mm (tolerates a rough point "
                                f"{r.get('coarse_tol_m', float('nan')) * 1e3:.0f} mm off) - the coarse stage and pass 1 "
                                "use it, passes 2+ choose again by quality", posture=_jsonable({k_: v for k_, v in r.items()}))
        return j

    def _joints_now(self) -> Optional[tuple]:
        """실측 관절 (이름 목록, 각 [rad]): MoveIt 어댑터 스냅샷 joint_positions_rad. RB 제어박스 어댑터는 None (가짜 CB 의 jnt_ang 은 관절이
        아니다 — 이 검사는 MoveIt 연구실 설정에서만 켠다)."""
        try:
            a = self.rs.pose.snapshot().arrays or {}
        except Exception:  # noqa: BLE001
            return None
        q = a.get("joint_positions_rad")
        if q is None or not len(q):
            return None
        return [str(x) for x in a.get("joint_names", [])], np.asarray(q, dtype=float)

    # ---- 2'. 범용 시작: 아무 자세 → 획득 (2026-10-07) ----------------------------------------------------------
    def _universal_ready(self) -> None:
        plat, be = self.rs, self.rs.backend
        if not (callable(getattr(plat, "reach_candidates", None)) and callable(getattr(be, "move_checked_joint", None))
                and callable(getattr(be, "check_joint", None)) and callable(getattr(plat, "set_scene_boxes", None))):
            raise _Abort(STATUS_REFUSED, "auto-teach needs a robot platform with joint-space planning and a collision scene (the MoveIt "
                                         f"adapter); this one ({self.rs.adapter}) has none (nothing sent)", category="config")

    def _spec_dims_m(self) -> dict:
        sp = self.spec
        return {"t": float(sp.thickness_mm) * 1e-3, "b": float(sp.backing_thickness_mm) * 1e-3 if sp.backing else 0.0,
                "w": float(sp.plate_width_mm) * 1e-3, "L": float(sp.weld_length_mm) * 1e-3,
                "opening": float(sp.top_opening_mm) * 1e-3,
                "w_plates": tuple(float(x) * 1e-3 for x in sp.plate_widths_mm), "gap": float(sp.root_gap_mm) * 1e-3}

    def _hit_face_m(self) -> float:
        """착점에서 판 상면까지의 가능한 최대 [m] (4d P10): 착점은 판 상면에 교시하지만 루트 (갭이 있으면 판 두께, 없으면 설계 깊이)
        깊이에 교시했을 수도 있다 — 충돌 장면·지지면 probe 의 판 면은 그만큼 로봇 쪽에서."""
        sp = self.spec
        if sp is None:
            return 0.0
        return float(sp.thickness_mm if float(sp.root_gap_mm) > 0.0 else sp.groove_depth_mm) * 1e-3

    def _lever_m(self) -> float:
        """특이점 척도의 길이: 플랜지 → 광학 원점 + 거친 거리 (각속도 × 이 길이 = 겨눈 점의 속도)."""
        r = float(self.job_c.standoff_m) if self.job_c is not None else self._coarse_range_default()
        return float(np.linalg.norm(self.calib.T_flange_opt[:3, 3])) + r

    def _set_cell(self, floor_h: float, why: str) -> None:
        """MoveIt 충돌 장면 = 시편 상자 + 바닥 판 (floor_h 아래; 크기 = 규격 + standoff + 접근 여유), 사방으로 정책 여유 (safety.clearance_m)
        만큼 키움 — 관절 공간 이동 (획득·probe) 이 피한다. 관절 이동은 가드가 보지 않으므로 (센서로 세우지 않는다) 정지 거리는 더하지 않는다.
        직선 이동은 포락선 규칙이 본다. 지금 자세가 바닥 판과 겹치면 (로봇이 시편 가까이) 로봇 쪽을 깎는다 — 로봇 반경 같은 상수 없이."""
        dm = self._spec_dims_m()
        reach = (float(self.job_c.standoff_m) if self.job_c is not None else self._coarse_range_default()) \
            + float(self.job0.safe_clearance_m)
        R = 0.5 * dm["w"] + reach
        pad = float(self.policy.clearance_m) if self.policy is not None else float(self.job0.safe_clearance_m)
        why_clip = ""
        boxes = None
        for frac in (1.0, 0.75, 0.5, 0.25, 0.0, None):
            if frac is None:
                boxes = [b_ for b_ in cell_boxes(self.ifr, floor_h, length_m=dm["L"], plate_width_m=dm["w"], thickness_m=dm["t"] + dm["b"],
                                                 reach_m=reach, slab_m=pad, pad_m=pad, face_offset_m=self._hit_face_m())
                         if b_["id"] == "specimen"]
                why_clip = "floor slab dropped: the current arm pose overlaps it even when cut at the specimen"
            else:
                boxes = cell_boxes(self.ifr, floor_h, length_m=dm["L"], plate_width_m=dm["w"], thickness_m=dm["t"] + dm["b"], reach_m=reach,
                                   slab_m=pad, near_clip_m=frac * R, pad_m=pad, face_offset_m=self._hit_face_m())
                why_clip = "" if frac == 1.0 else f"floor slab cut on the robot side to {frac * R:.2f} m (the current arm pose overlapped it)"
            if not self.rs.set_scene_boxes(boxes):
                raise _Abort(STATUS_REFUSED, "universal start: the collision scene (specimen, floor) could not be applied to MoveIt - "
                                             "joint-space moves would not avoid the specimen (nothing sent)", category="motion")
            self.cell_ids = sorted(set(self.cell_ids) | {bx["id"] for bx in boxes})
            stale = [i for i in self.cell_ids if i not in {bx["id"] for bx in boxes}]
            if stale:
                self.rs.remove_scene_boxes(stale)
                self.cell_ids = [i for i in self.cell_ids if i not in stale]
            hit = self.rs.state_collision() if callable(getattr(self.rs, "state_collision", None)) else None
            if hit is None:
                break
        if hit is not None:
            if "specimen" in str(hit):
                # 교시를 판 가까이에서 끝냈을 때가 흔하다 (4d 검토: E 를 0.25 m 거리에서 교시) — 여유를 줄이지 않고 팔을 물리게 한다
                raise _Abort(STATUS_REFUSED, f"universal start: the arm at its current pose is inside the specimen keep-out ({hit}; the "
                                             f"specimen box padded by the clearance {pad * 1e3:.0f} mm) - move the arm back from the plate "
                                             "(e.g. after teaching the last point, before starting) and start again; if the arm is already "
                                             "far from the plate, check the hit points (nothing sent)", category="motion")
            # 시편 상자가 아닌 원인 (자기 충돌·관절 한계·관절 상태 없음): 시편 쪽으로 안내하지 않는다 (4d 검토)
            raise _Abort(STATUS_REFUSED, f"universal start: the current arm state is not valid for planning ({hit}) even without the "
                                         "floor slab - self-collision, a joint outside its limits or missing joint states; move the arm to a "
                                         "normal pose (or check /joint_states) and start again (nothing sent)", category="motion")
        self._decide("acquire", f"collision scene (each box padded by the clearance {pad * 1e3:.0f} mm): "
                                f"{', '.join(bx['id'] + ' ' + str(np.round(bx['size'], 3).tolist()) for bx in boxes)} "
                                f"(floor top {float(floor_h):.4f} m: {why}){'; ' + why_clip if why_clip else ''}",
                     boxes=_jsonable([{"id": bx["id"], "size": bx["size"], "T": bx["T"]} for bx in boxes]))

    def _clear_cell(self) -> None:
        if self.cell_ids and callable(getattr(self.rs, "remove_scene_boxes", None)):
            try:
                self.rs.remove_scene_boxes(list(self.cell_ids))
            except Exception as e:  # noqa: BLE001
                self.res.notes.append(f"collision scene clean-up failed: {e}")
        self.cell_ids = []

    @staticmethod
    def _reach_rows(ranked: list, n: int = 6) -> list:
        """결정 기록용 도달 후보 요약 (좋은 순서): 관절 [deg], 특이점 거리, 한계 여유, 지금 관절에서의 거리."""
        return [{"q_deg": np.round(np.degrees(c["q"]), 1).tolist(),
                 "min_sigma_m": round(float(c["min_sigma"]), 4), "limit_margin_deg": round(math.degrees(c["limit_margin_rad"]), 1),
                 "dist_deg": None if c.get("dist_rad") is None else round(math.degrees(c["dist_rad"]), 1)} for c in ranked[:n]]

    def _q_now(self) -> Optional[dict]:
        """지금 관절 {이름: 값} (/joint_states 순서와 무관하게 플랫폼이 체인 순서로 맞춘다)."""
        js = self._joints_now()
        return None if js is None else {n: float(v) for n, v in zip(js[0], js[1])}

    def _q_now_chain(self) -> Optional[np.ndarray]:
        """지금 관절 (플랫폼 체인 순서 배열) — 모르면 None."""
        qd = self._q_now()
        ch = self.rs.chain() if callable(getattr(self.rs, "chain", None)) else None
        if qd is None or ch is None:
            return None
        try:
            return np.array([float(qd[n]) for n in ch.joint_names])
        except KeyError:
            return None

    def _duration_bounds(self, cands: list, v: float) -> list:
        """Exact lower bound of each candidate's planned joint-space move from the current joints (joint_selection.duration_lower_bound:
        the backend's path_timing_limits, the flange FK, this move's TCP speed / acceleration). 0 for all when the platform cannot say
        (no early stop then - the search still checks in duration order)."""
        fn = getattr(self.rs.backend, "path_timing_limits", None)
        q0 = self._q_now_chain()
        fk = self.rs.fk() if callable(getattr(self.rs, "fk", None)) else None
        if not callable(fn) or q0 is None:
            return [0.0] * len(cands)
        lim = fn()
        return [duration_lower_bound(q0, np.asarray(c["q"], dtype=float), lim, fk=fk, v_mps=v, a_mps2=self.accel) for c in cands]

    def _select_joint(self, cands: list, name: str, *, prefer: Optional[Callable] = None) -> dict:
        """Joint-space path selection by planned duration (scan_l0.joint_selection.DurationSearch; KIRO field 2026-10-14, review B1):
        the candidates in the order of their exact duration lower bound, the straight joint line checked first (backend check_joint
        allow_ompl False), MoveIt OMPL for the blocked ones only when no candidate has an executable straight line; the search stops
        when no unchecked candidate can be shorter. prefer(search) -> (item, record) may pick a later item of the duration order (the
        yaw band); default the first. The run's stop is checked around every check; an unavailable platform (PlatformUnavailable:
        move_group not answering) ends the search at once - REFUSED, never a per-candidate rejection. An executable plan longer than
        an explicit motion timeout is not usable.
        → {cand, check, index, rows, bound_s, duration_s, route, n_checked, n_ompl, prefer} (cand None = no usable path)."""
        be = self.rs.backend
        v = min(float(self.transit_speed), self.cfg.v_eff_max_mps)
        mto = self.motion_timeout_s
        bounds = self._duration_bounds(cands, v)

        def check(c, allow_ompl):
            return be.check_joint(c["q"], v, self.accel, timeout_s=mto, allow_ompl=allow_ompl)

        def accept(_c, chk):
            d = float((chk.get("plan") or {}).get("duration_s") or 0.0)
            return (f"planned {d:.1f} s >= the explicit motion timeout {float(mto):.1f} s (raise it or omit it)"
                    if mto is not None and d >= float(mto) else "")

        search = DurationSearch(cands, bounds, check, stop=self._check_stop, accept=accept)
        pref: dict = {}
        try:
            if prefer is None:
                item = next(search, None)
            else:
                item, pref = prefer(search)
        except PlatformUnavailable as e:
            rows = search.finish(None)
            self._decide("joint_selection", f"{name}: move_group not answering ({e}) - the search over {len(cands)} arm configurations "
                                            f"stops here (the platform, not a candidate)", candidates=_jsonable(rows))
            raise _Abort(STATUS_REFUSED, f"{name}: move_group not answering ({e}) - nothing sent", category="motion")
        i = None if item is None else item[0]
        rows = search.finish(i)
        for row in sorted((r_ for r_ in rows if r_["checked"]), key=lambda r_: r_.get("check_order", 0)):
            self._decide("joint_candidate", f"{name}: candidate {row['rank']}/{len(rows)} ({row['checked']}): "
                                            + (f"planned {row['duration_s']:.2f} s" if row["ok"] else f"refused: {row['reason'][:160]}")
                                            + f" (lower bound {row['bound_s']:.2f} s)", candidate=_jsonable(row))
        sel = {"cand": None if item is None else item[1], "check": None if item is None else item[2], "index": i, "rows": rows,
               "n_checked": search.n_straight + search.n_ompl, "n_ompl": search.n_ompl, "prefer": pref,
               "bound_s": None if i is None else bounds[i], "duration_s": None if i is None else rows[i]["duration_s"],
               "route": None if item is None else str((item[2].get("plan") or {}).get("route") or "")}
        self._decide("joint_selection", f"{name}: " + (
            f"candidate {i + 1}/{len(cands)} - planned {sel['duration_s']:.2f} s ({sel['route']}; its lower bound {sel['bound_s']:.2f} s); "
            f"{sel['n_checked']} of {len(cands)} checked ({search.n_ompl} with OMPL) - every unchecked one's lower bound is not "
            "shorter" if item is not None else f"no usable joint-space path among {len(cands)} arm configurations")
            + (f"; {pref['text']}" if pref.get("text") else ""), candidates=_jsonable(rows), joint_selection_rule=JOINT_SELECTION_RULE)
        return sel

    def _jtc_tracking(self, t0_ns: int, t1_ns: int) -> Optional[dict]:
        """jtc_tracking_record over [t0_ns, t1_ns] (host monotonic) from the platform's controller_history; None when it has none."""
        fn = getattr(self.rs, "controller_history", None)
        if not callable(fn):
            return None
        try:
            return jtc_tracking_record(fn(int(t0_ns), int(t1_ns)) or {})
        except Exception as e:  # noqa: BLE001 - a record only
            return {"n": 0, "note": f"controller states not read: {type(e).__name__}: {e}"}

    def _joint_move(self, ranked: list, T_goal: np.ndarray, name: str, what: str, *, selection: Optional[dict] = None) -> dict:
        """관절 공간 이동 (MoveIt): 후보 중 계획 시간이 가장 짧은 경로 (_select_joint — selection 이 있으면 계획 단계에서 고른 그것) → 확인 →
        그 후보가 검사받은 궤적 그대로 (move_checked_joint: 다시 계획하지 않는다, 보내기 직전 시작·장면·다른 관절 재확인) 같은 JTC 감시 경로로
        실행. The executed candidate is the selected one by identity - nothing is sorted again after the selection (KIRO's re-sort bug
        executed another candidate). → the executed candidate. moves[]: joint_candidates (every candidate's bound / check / outcome),
        motion.tracking (the JTC's reference - feedback, jtc_tracking_record)."""
        be = self.rs.backend
        self._check_stop(f"before move '{name}'")
        T_from = self._T_now()
        v = min(float(self.transit_speed), self.cfg.v_eff_max_mps)
        sel = selection if selection is not None else self._select_joint(ranked, name)
        if sel["cand"] is None:
            fails = [r_["reason"] for r_ in sel["rows"] if r_["checked"] and not r_["ok"]]
            d = self._record_move(name, "joint", T_from, T_goal, {}, v, refused="; ".join(fails))
            d["joint_candidates"] = _jsonable(sel["rows"])
            raise _Abort(STATUS_REFUSED, f"{name}: no collision-free joint-space path from the current pose to any of the {len(sel['rows'])} "
                                         f"arm configurations for {what}: {fails[:2]} (nothing sent)", category="motion")
        c, chk = sel["cand"], sel["check"]
        d = self._record_move(name, "joint", T_from, T_goal, {}, v)
        d["joint_goal"] = [float(x) for x in c["q"]]
        d["joint_plan"] = _jsonable(chk.get("plan"))
        d["joint_candidates"] = _jsonable(sel["rows"])
        d["joint_selection"] = {"rule": JOINT_SELECTION_RULE, "candidate": sel["index"] + 1, "bound_s": sel["bound_s"],
                                "duration_s": sel["duration_s"], "n_checked": sel["n_checked"], "n_ompl": sel["n_ompl"]}
        pl = chk.get("plan") or {}
        pt = move_plan_text(pl)                     # 계획한 경로 종류·플랜지 경로와 회전 (따로)·시간·가장 많이 도는 관절 (확인과 같은 문구)
        how = "" if pl.get("route") == "straight" or not pl.get("how") else f" [{pl['how']}]"     # 우회면 왜 (막힌 곳) 도
        self._decide("move", f"{name} (joint space, {what}): {pt}{how}; arm configuration {np.round(np.degrees(c['q']), 1).tolist()} deg "
                             + (f"(limit margin {math.degrees(c['limit_margin_rad']):.0f} deg, min sigma {c['min_sigma']:.3f} m)"
                                if "limit_margin_rad" in c else "(taught configuration)"))
        kind = joint_move_kind(pl.get("how", ""))
        self._confirm(name, d["dist_mm"] * 1e-3, extra=f"(joint-space move via MoveIt to {what}; collision scene: specimen + floor"
                                                        + (f"; NOT a straight joint line ({pl.get('how', '?')}) - check the path"
                                                           if kind == "joint_detour" else "") + ")",
                      rot_deg=d.get("rot_deg", 0.0), kind=kind, plan_text=pt)
        self._check_start_pose(T_from, name)
        self._event("move", name=name, phase="joint", streaming=False)
        self._motion_sent = True
        t0_ns = time.monotonic_ns()
        try:
            res = be.move_checked_joint(chk, c["q"], v, self.accel, timeout_s=self.motion_timeout_s)
        except MotionError as e:
            d["motion"] = {"error": str(e), "tracking": self._jtc_tracking(t0_ns, time.monotonic_ns())}
            raise _Abort(STATUS_ABORTED, f"move '{name}': {e}", safety=True, category="motion")
        finally:
            self.res.n_move_l += 1
        d["motion"] = res.as_dict()
        d["motion"]["tracking"] = self._jtc_tracking(res.t_send_ns or t0_ns, res.t_stop_ns or time.monotonic_ns())
        d["executed"] = bool(res.ok)
        if not res.ok:
            raise _Abort(STATUS_ABORTED, f"move '{name}': {res.reason}", safety=True, category="motion")
        self._p(f"      -> OK {res.reason} duration={res.motion_duration_s:.2f} s")
        return c

    def _support_probe(self) -> None:
        """3G: 아래 착점 옆 판 면을 내려다보는 정지 측정 한 번 → 지지면 (테이블·지그 윗면). 자세는 계산 (acquisition.plan_support_probe:
        툴 몸체 전체가 아래 착점·판 면에서 정책 여유 + 도착 허용 떨어진 채로 센서 거리 창이 착점 아래 가장 깊이까지 닿는 내려다보는 각·거리 — 도달되는
        것 중). 지지물을 못 보면 보인 판의 가장 낮은 점이 보수적 상한. 테이블 입력 (table_z_m) 을 대신한다 (사용자 2026-10-07 Q2)."""
        from scan_core.validity import dx_linear_coeffs
        fr = self.ifr
        dm = self._spec_dims_m()
        inv = np.linalg.inv(self.calib.T_flange_opt)
        floor0 = fr.height(fr.lower)
        def reach_toward(T_opt):                                     # 광학 원점에서 판 쪽 (−n0) 으로 가장 멀리 나온 툴 점 (키포인트 + 몸체)
            T_bf = T_opt @ inv
            pts = list(tool_keypoints(T_bf, self.calib, self.job_c).values())
            B = getattr(self.calib, "body_points_eoat", None)
            if B is not None:
                pts += list(body_points_base(T_bf @ self.calib.T_flange_eoat, B).reshape(-1, 3))
            o = T_opt[:3, 3]
            return max(float(-fr.n0 @ (np.asarray(p_) - o)) for p_ in pts)
        # probe 자세는 가드 없이 (관절 이동) 가서 멈춰 재는 자세: 여유는 정책 여유 + 도착 허용 (정지 거리는 가드가 세울 때만 해당)
        probe_margin = float(self.policy.clearance_m + self.policy.pose_tol_m)
        cands = plan_support_probe(fr, lambda T: lowest_below_origin(T @ inv, self.calib, self.job_c), floor_h=floor0,
                                   margin_m=probe_margin, z_valid_mm=tuple(self.settings.z_valid_mm),
                                   dx_coeffs=dx_linear_coeffs(), lateral_m=dm["opening"], reach_toward_plate=reach_toward,
                                   n_columns=self.settings.sensor_columns, face_offset_m=self._hit_face_m())
        q_now = self._q_now()
        tried, chosen = [], None
        seen = set()
        # 보는 깊이 (probe 의 목적) 가 같으면 로봇 베이스에 가까운 자세부터 (도달성의 상대 비교 — 상수 없음); 각 (각도, 뒤집기) 에서 첫 후보
        cands = sorted(cands, key=lambda c_: (-c_["depth_m"], float(np.linalg.norm((c_["T_opt"] @ inv)[:3, 3]))))
        def reachable():                                            # 도달되는 probe 후보를 좋은 순서로 (IK 시도 예산 8 — 계산 예산)
            for c_ in cands:
                key = (c_["look_down_deg"], c_["flip"])
                if key in seen or c_["depth_m"] <= 0.0:
                    continue
                seen.add(key)
                rc_ = rank_reach_candidates(self.rs.reach_candidates(c_["T_opt"] @ inv, q_now=self._q_now(), sigma_radius_m=self._lever_m()))
                tried.append(f"{c_['look_down_deg']:.0f}deg{' flipped' if c_['flip'] else ''} {c_['front_m'] * 1e3:.0f} mm in front: sees "
                             f"{c_['depth_m'] * 1e3:.0f} mm below, {len(rc_)} arm configurations")
                if rc_:
                    yield c_, rc_
                if len(tried) >= REACH_CANDIDATE_BUDGET:
                    return
        gen = reachable()
        chosen = next(gen, None)
        self._decide("acquire", f"3G support probe (vertical laser sheet {dm['opening'] * 1e3:.0f} mm beside the lower hit point "
                                f"{np.round(fr.lower, 4).tolist()}, whole tool >= the lower hit and the plate face + "
                                f"{probe_margin * 1e3:.0f} mm): "
                                + "; ".join(tried), candidates=_jsonable([{k_: v for k_, v in c.items() if k_ != "T_opt"} for c in cands[:10]]),
                     reach=_jsonable(self._reach_rows(chosen[1]) if chosen else []), q_now=_jsonable(q_now))
        if chosen is None:
            raise _Abort(STATUS_REFUSED, "3G support probe: no reachable pose sees below the lower hit point - the specimen is out of the "
                                         "robot's reach (nothing sent)")
        if self.dry_run:
            return
        self._sensor()                                          # 센서 연결·초기화는 첫 이동 전에 (실패하면 아무것도 보내지 않았다)
        rd, k_probe = None, 0
        while chosen is not None:
            c, rc = chosen
            k_probe += 1
            nm = "support_probe" if k_probe == 1 else f"support_probe{k_probe}"
            self._joint_move(rc, c["T_opt"] @ inv, nm, f"the 3G support probe pose (look down {c['look_down_deg']:.0f} deg)")
            x_med, z_med, T_bf, _st, _recs = self._grab(nm)
            To = np.asarray(T_bf, dtype=float) @ self.calib.T_flange_opt
            ok = valid_mask(x_med, z_med, tuple(self.settings.z_valid_mm))
            P = (To[:3, :3] @ np.vstack([x_med[ok], np.zeros(int(ok.sum())), z_med[ok]]) * 1e-3 + To[:3, 3:4]).T
            rd = measure_support(P, fr.up, To[:3, 2])
            if rd.height is not None or rd.foot_height is not None:
                break
            # 계획이 가정한 판 (세로·추정 법선) 과 실제가 달라 (기운 판 등) 아무것도 못 봤다: 다음 도달 후보에서 다시 본다 (적응적, 같은 예산)
            chosen = next(gen, None)
            self._decide("acquire", f"3G support probe {k_probe}: {rd.note} - the plate is not where the probe plan assumed (leaning?)"
                                    + ("; next reachable probe pose: " + tried[-1] if chosen is not None else "; no other reachable probe pose"))
        if rd.height is not None:
            h, src = float(rd.height), "measured" + ("" if rd.flat else " (top of a jig / uneven support)")
        elif rd.foot_height is not None:
            h, src = float(rd.foot_height), "bound (lowest plate point seen; the support is lower than the view)"
        else:
            raise _Abort(STATUS_ABORTED, f"3G support probe: {rd.note} - check the hit points (the lower one should be on the specimen)")
        self.support = {"height_m": h, "src": src, "n_support": rd.n_support, "n_plate": rd.n_plate, "flat": rd.flat,
                        "foot_height_m": rd.foot_height, "note": rd.note, "below_lower_hit_m": floor0 - h, "tol_m": rd.tol_m,
                        "probe": {k_: v for k_, v in c.items() if k_ != "T_opt"}}
        self._decide("acquire", f"3G support: {h:.4f} m ({src}; {rd.n_support} support / {rd.n_plate} plate points"
                                f"{'; ' + rd.note if rd.note else ''}) = {(floor0 - h) * 1e3:.0f} mm below the lower hit point",
                     support=_jsonable(self.support))
        if h > floor0 + _ms.DESIGN_HALF * dm["t"]:
            raise _Abort(STATUS_ABORTED, f"3G support probe: a surface {(h - floor0) * 1e3:.0f} mm ABOVE the lower hit point in front of the "
                                         "plate - the lower hit point is not on the specimen, or something stands in front of the plate")
        job = ScanJob.from_dict({**job_to_dict(self.job0), "torch_lateral_m": tuple(self.job0.torch_lateral_m), "table_z_m": h})
        job.plate_top_measured_m = self.job0.plate_top_measured_m
        self.job0 = job
        self._set_cell(h, f"3G support {src}")

    def _wrist2_required(self) -> str:
        """이 실행이 허용하는 손목 형태 (손목2 부호): 캘리브를 만든 팔 형태 집합 — 3G auto_teach.wrist2_branch_3g, 1G wrist2_branch_1g
        ('any' = 제한 없음)."""
        return str(self.at.wrist2_branch_3g if self.orientation == "vertical_3G" else self.at.wrist2_branch_1g)

    def _acquisition_poses(self, sign: int) -> tuple:
        """진행 부호 sign 의 거친 작업 획득 자세 → (작업, 물러난 자세 T_pre, 드웰 T_in, 거친 직선 끝 T_out). 계획 오류는 그대로 올린다."""
        vt = min(float(self.transit_speed), self.cfg.v_eff_max_mps)
        job = ScanJob.from_dict({**job_to_dict(self.job_c), "progression_sign": sign})
        job.validate()
        plan_full, _ = make_plan(job, self.calib, None, transit_speed_mps=vt, descend_speed_mps=self.descend_speed,
                                 accel_mps2=self.accel, v_eff_max_mps=self.cfg.v_eff_max_mps, check=False)
        wn = [w.name for w in plan_full.waypoints]
        i_in = next(wn.index(nm) for nm in ("lead_in", "scan_start") if nm in wn)
        T_in = plan_full.waypoints[i_in].T_base_flange
        T_out = plan_full.waypoints[i_in + 1].T_base_flange
        T_pre = np.array(T_in, dtype=float, copy=True)
        T_pre[:3, 3] += job.normal_unit * float(self.job0.safe_clearance_m)
        return job, T_pre, T_in, T_out

    def _tcp_in_flange(self) -> tuple:
        """The robot's TCP in the flange frame: the weld torch tip (weld_torch TCP on this arm - KIRO: right_manipulator_ee_point, the
        robot model's flange <- TCP); without a torch on this arm the sensor's optical origin (the tool it carries). -> (xyz [m], name)."""
        wt = getattr(self, "weld_torch", None)
        if wt is not None and getattr(wt, "on_this_arm", False):
            return np.asarray(wt.flange_from_tcp.t, dtype=float).copy(), f"torch tip {wt.tcp_frame}"
        return np.asarray(self.calib.T_flange_opt, dtype=float)[:3, 3].copy(), "the sensor optical frame"

    def _tcp_motion(self, T_from, T_to) -> tuple:
        """TCP motion between two flange poses: (straight-line distance of the TCP [m], tool rotation [deg])."""
        tcp, _nm = self._tcp_in_flange()
        A, B = np.asarray(T_from, dtype=float), np.asarray(T_to, dtype=float)
        d = (B[:3, :3] @ tcp + B[:3, 3]) - (A[:3, :3] @ tcp + A[:3, 3])
        return float(np.linalg.norm(d)), _rot_angle_deg(A[:3, :3], B[:3, :3])

    def _acquisition_motions(self) -> dict:
        """{progression sign: (TCP distance [m], tool rotation [deg])} from the current flange pose to each sign's retreat pose of the
        current visiting order - poses only, no IK ({} without a current pose)."""
        if self._q_now() is None:
            return {}
        T_now = self._T_now()
        out = {}
        for sign in self._progression_signs():
            try:
                _job, T_pre, _T_in, _T_out = self._acquisition_poses(sign)
            except (EnvelopeError, JobError):
                continue
            out[sign] = self._tcp_motion(T_now, T_pre)
        return out

    def _tool_side(self, sign: int) -> int:
        """The physical tool orientation of a progression sign, in the taught S->E frame (the visiting order swaps the internal S/E:
        sign +1 of the swapped job = the tool yaw of sign -1 of the taught one)."""
        return int(sign) * (-1 if self.res.rough_swapped else 1)

    def _plan_acquisition(self, name: str) -> Optional[dict]:
        """거친 작업의 첫 드웰 (착점 S0) 로: 진행 부호 × IK 해마다 [물러난 자세 → 드웰 → E0 드웰] 을 직선으로 이어 따라가 보고 (플랫폼
        reach_candidates — IK 해는 지금 관절에 가장 가까운 2π 동치 표현도 후보). 캘리브를 만든 팔 형태 집합이 정해져 있으면
        (wrist2_branch_1g/3g) 그 안에서만. The tool pose (progression sign = the tool yaw about the plate normal) = the least TCP
        motion from here, TCP = the torch tip (KIRO field 2026-10-14: the least joint motion turned the torch 115 deg around to the
        far side where 75 deg kept it - and its wire - on the robot's side; the torch and wire must not swing around): the smaller
        rotation of the TCP frame first, then the torch tip's distance; the other sign only when this one has no usable path. For
        that tool pose the arm configuration =
        the shortest planned joint-space move (_select_joint: duration under the platform's limits), and in that duration order the
        first one that also follows the +-psi_b rotations S_align may need (_prefer_yaw_band, within REACH_CANDIDATE_BUDGET; else
        the shortest). 물러남 = job.safe_clearance_m. 후보가 없거나 경로가 없으면 None (self._acq_fail = why when no joint path)."""
        q_now = self._q_now()
        retreat = float(self.job0.safe_clearance_m)
        T_now = self._T_now()
        req = self._wrist2_required()
        self._acq_fail = ""
        rows, pool, ctx = [], [], {}
        for sign in self._progression_signs():
            try:
                job, T_pre, T_in, T_out = self._acquisition_poses(sign)
            except (EnvelopeError, JobError) as e:
                rows.append({"sign": sign, "error": str(e)})
                continue
            cands = self.rs.reach_candidates(T_pre, follow=[T_in, T_out], q_now=q_now, sigma_radius_m=self._lever_m())
            if req != "any":
                for c in cands:
                    why = wrist2_branch_problem(c.get("names") or [], np.asarray(c["q"]), req)
                    if why:
                        c["ok"], c["reason"] = False, f"wrist2_branch ({self.res.position}): {why}"
            for c in cands:
                c["sign"] = sign
            tcp_m, rot_deg = self._tcp_motion(T_now, T_pre)
            ctx[sign] = {"T_pre": T_pre, "T_in": T_in, "T_out": T_out, "job": job, "tool_rot_deg": rot_deg, "tcp_move_m": tcp_m}
            pool.extend(cands)
            rows.append({"sign": sign, "n_ik": len(cands), "n_ok": sum(1 for c in cands if c.get("ok")),
                         "fails": sorted({c["reason"][:80] for c in cands if not c["ok"]}), "tool_rot_deg": rot_deg,
                         "tcp_move_mm": tcp_m * 1e3})
        rejected = [c for c in pool if not c.get("ok")]
        self._decide("acquisition_filter", f"{self._site_label(name)}: {len(pool) - len(rejected)} of {len(pool)} arm configurations "
                                           "reach the acquisition pose and follow the coarse line (IK / follow / wrist2) - those go to "
                                           "the joint-space path search",
                     rejected=_jsonable([{"sign": c.get("sign"), "q_goal_deg": np.degrees(np.asarray(c["q"])).tolist(),
                                          "reason": c.get("reason")} for c in rejected]))
        ranked = rank_reach_candidates(pool)
        for r_ in rows:
            if "error" in r_:
                continue
            mine = [c for c in ranked if c["sign"] == r_["sign"]]
            r_["reach"] = self._reach_rows(mine)
            if mine:
                r_["best_dist_deg"] = None if mine[0].get("dist_rad") is None else math.degrees(float(mine[0]["dist_rad"]))
        sel, fails = None, []
        for sg in sorted(ctx, key=lambda k_: (ctx[k_]["tool_rot_deg"], ctx[k_]["tcp_move_m"])):     # least TCP motion first
            mine = [c for c in ranked if c["sign"] == sg]
            if not mine:
                continue
            sel = self._select_joint(mine, f"acquire_{self._site_label(name)}", prefer=lambda search: self._prefer_yaw_band(search, ctx))
            if sel["cand"] is not None:
                break
            fails += [r_["reason"] for r_ in sel["rows"] if r_["checked"] and not r_["ok"]]
        if ranked and (sel is None or sel["cand"] is None):
            self._acq_fail = (f"no collision-free joint-space path from the current pose to any of the {len(ranked)} arm "
                              f"configurations of the acquisition: {fails[:2]}")
        best = None
        if sel is not None and sel["cand"] is not None:
            c0 = sel["cand"]
            sg = int(c0["sign"])
            same = [c for c in ranked if c["sign"] == sg]
            pl = sel["check"].get("plan") or {}
            best = {"sign": sg, "ranked": same, "selection": sel, "T_pre": ctx[sg]["T_pre"], "T_in": ctx[sg]["T_in"], "retreat": retreat,
                    "top_band": bool(c0.get("yaw_band_ok")), "tool_rot_deg": ctx[sg]["tool_rot_deg"],
                    "tcp_move_m": ctx[sg]["tcp_move_m"],
                    "duration_s": float(sel["duration_s"]), "bound_s": sel["bound_s"], "route": sel["route"],
                    "joint_motion_deg": float(sum(pl.get("joint_rotation_deg") or [])) if pl.get("joint_rotation_deg") else None}
        opts = [{k_: r_.get(k_) for k_ in ("sign", "n_ik", "n_ok", "tcp_move_mm", "tool_rot_deg", "best_dist_deg", "error")} for r_ in rows]
        self.res.visit_order["options"] = _jsonable(opts)
        if best is not None:
            self.res.visit_order.update(progression_sign=best["sign"], tool_rotation_deg=best["tool_rot_deg"],
                                        tcp_move_mm=best["tcp_move_m"] * 1e3, tcp=self._tcp_in_flange()[1],
                                        joint_motion_deg=best["joint_motion_deg"], duration_s=best["duration_s"],
                                        duration_bound_s=best["bound_s"], route=best["route"],
                                        joint_plan=_jsonable(sel["check"].get("plan")))
        band = (sel or {}).get("prefer") or {}
        self._decide("acquire", f"{self._site_label(name)}: acquisition (dwell at the hit point, {retreat * 1e3:.0f} mm back along the plate "
                                f"normal, then "
                                f"the coarse S0->E0 line): "
                     + "; ".join(f"progression {r_['sign']:+d}: " + (r_["error"] if "error" in r_ else
                                 f"{r_['n_ok']}/{r_['n_ik']} arm configurations follow it, TCP motion from the current pose: "
                                 f"rotation {r_['tool_rot_deg']:.0f} deg, {r_['tcp_move_mm']:.0f} mm; least joint motion "
                                 f"{_fmtf(r_.get('best_dist_deg'), '.0f')} deg"
                                 + (f" ({', '.join(r_['fails'][:2])})" if r_.get("fails") else ""))
                                 for r_ in rows)
                     + (f" -> progression {best['sign']:+d}: the least TCP motion (rotation {best['tool_rot_deg']:.0f} deg, "
                        f"{self._tcp_in_flange()[1]} {best['tcp_move_m'] * 1e3:.0f} mm); its arm configuration by the shortest planned joint-space "
                        f"move, {best['duration_s']:.1f} s ({best['route']}"
                        f"{'' if req == 'any' else '; wrist2 ' + req + ' only (the calibrated configuration)'}); "
                        f"{band.get('text', '')}" if best is not None else f" -> none{(': ' + self._acq_fail) if self._acq_fail else ''}"),
                     rows=_jsonable(rows), options=_jsonable(opts), wrist2_required=req)
        return best

    def _yaw_band_rad(self) -> float:
        """ψ_b = atan((δ_c,S + δ_c,E) / L_r): 첫 시야가 받는 대략점 측면 오차 (δ_c = 거친 거리에서 첫 시야가 견디는 착점 오차, 센서 시야 모형)
        두 개가 만드는 레이저 선 yaw 의 상한 — S_align 이 돌 수 있는 각."""
        jc = self.job_c
        try:
            d_c = self._sv().coarse_tolerance_m(self._depth_spec_m(), float(jc.incidence_tilt_deg), self._opening_spec_m())
        except ValueError:
            return 0.0
        L_r = float(np.linalg.norm(np.asarray(self.job0.end, dtype=float) - np.asarray(self.job0.start, dtype=float)))
        return math.atan2(2.0 * max(0.0, d_c), L_r) if L_r > 0.0 else 0.0

    def _prefer_yaw_band(self, ordered, ctx: dict) -> tuple:
        """ordered = the candidates in nondecreasing planned duration (DurationSearch items (index, cand, check)) → (item, record): the
        first item, in that order, whose configuration also follows [T_in → R±(T_in) → R±(T_out)] (its progression sign's poses; R =
        ±ψ_b about n0 through that pose's hit point) - platform follow_from (no IK), at most REACH_CANDIDATE_BUDGET items (computation
        budget); none of them: the first (shortest) item (recorded, no new refusal). Applied after the duration order (review B1: a
        promotion before a duration sort is lost)."""
        psi_b = self._yaw_band_rad()
        first = next(ordered, None)
        ff = getattr(self.rs, "follow_from", None)
        out = {"psi_b_deg": math.degrees(psi_b), "checked": 0, "promoted": None}
        if first is None:
            out["text"] = "no arm configuration has a usable joint-space path"
            return None, out
        if not (psi_b > 0.0) or not callable(ff):
            out["text"] = (f"yaw band +-{math.degrees(psi_b):.1f} deg not checked"
                           + ("" if callable(ff) else " (platform has no follow_from)"))
            return first, out
        chains: dict = {}
        for sg, cx in ctx.items():
            n0 = cx["job"].normal_unit
            hits = (np.asarray(cx["job"].start, dtype=float), np.asarray(cx["job"].end, dtype=float))
            chains[sg] = []
            for rs in (1.0, -1.0):
                Tr_in = local_target_from_correction(cx["T_in"], np.zeros(3), roll_axis_base=n0, roll_deg=rs * math.degrees(psi_b),
                                                     pivot_base=hits[0])
                Tr_out = local_target_from_correction(cx["T_out"], np.zeros(3), roll_axis_base=n0, roll_deg=rs * math.degrees(psi_b),
                                                      pivot_base=hits[1])
                chains[sg].append([cx["T_in"], Tr_in, Tr_out])
        item = first
        while item is not None and out["checked"] < REACH_CANDIDATE_BUDGET:
            c = item[1]
            out["checked"] += 1
            try:
                ok = all(bool(ff(c["q"], ch, sigma_radius_m=self._lever_m()).get("ok")) for ch in chains[c["sign"]])
            except Exception:  # noqa: BLE001 - ranking aid only
                ok = False
            c["yaw_band_ok"] = ok
            if ok:
                out["promoted"] = out["checked"] - 1
                out["text"] = (f"duration rank {out['checked']} also follows +-{out['psi_b_deg']:.1f} deg about the plate normal"
                               + ("" if out["checked"] == 1 else " (preferred over the shorter ones)"))
                return item, out
            if out["checked"] < REACH_CANDIDATE_BUDGET:
                item = next(ordered, None)
        out["text"] = (f"none of the {out['checked']} shortest also follows +-{out['psi_b_deg']:.1f} deg about the plate normal "
                       "(the shortest kept)")
        return first, out

    def _swap_rough(self, job0: ScanJob) -> ScanJob:
        """내부 순서 뒤집기: 대략 시·종점을 바꾼 작업 (교시는 res.visit_order 에; 결과물은 교시 S→E 로 되돌린다)."""
        j = ScanJob.from_dict({**job_to_dict(job0), "start": job0.end.tolist(), "end": job0.start.tolist()})
        j.plate_top_measured_m = self.job0.plate_top_measured_m
        self.job0 = j
        self.res.rough_swapped = not self.res.rough_swapped
        return j

    # ---- 결과물은 교시 S→E (방문 순서와 무관, 사용자 2026-10-09) ----
    def _user_direction(self, direction: Optional[str]) -> Optional[str]:
        if not self.res.rough_swapped:
            return direction
        return {"S->E": "E->S", "E->S": "S->E"}.get(direction, direction)

    def _user_end(self, X: str) -> str:
        """내부 끝 이름 (S/E) → 교시 끝 이름."""
        return {"S": "E", "E": "S"}.get(X, X) if self.res.rough_swapped else X

    def _site_label(self, X: str) -> str:
        """획득 로그·이동 이름에 쓰는 끝 이름. legacy (3G·1G legacy) 는 내부 이름 그대로 (예전과 같음); 1G core 는 교시 이름 (Teach1G)."""
        return X

    def _user_extent(self, ext: Optional[dict]) -> Optional[dict]:
        """패스 extent (내부 S/E) → 교시 S/E: 끝 기록 (base 점·σ — S→E 계약, P1) 을 바꿔 단다. 축·좌표는 소비자가 끝 점에서 유도하므로
        (unit(E − S)) 끝을 바꿔 다는 것으로 충분하다. hit_range_SE (패스 맞춤 직선 위 [lo, hi], 내부 S→E 축) 는 교시 축 (반대) 으로
        [−hi, −lo]. first/last 는 패스 진행 순서 그대로."""
        if not ext or not self.res.rough_swapped:
            return ext
        out = dict(ext)
        out["S"], out["E"] = ext.get("E"), ext.get("S")
        if "direction" in out:
            out["direction"] = self._user_direction(out["direction"])
        for key in ("hit_range_SE", "hit_range_SE_m"):
            hr = ext.get(key)
            if hr is not None and len(hr) == 2:
                out[key] = [-float(hr[1]), -float(hr[0])]
        out["frame"] = "taught S->E (visited end first: the internal S is the taught E)"
        return out

    def _acquire_plan(self, name: str, phase: str) -> tuple:
        """지금 대략 순서·자세로 획득 계획 (_plan_acquisition) + 관절 이동 전에 그 뒤 판 쪽 직선의 포락선 (클램프·판면·지지면 한계 — 거부될
        직선 앞으로 팔을 옮기지 않는다; S 는 그 뒤의 S 추적 직선 (드웰에서 대략 S→E 방향으로 b, 'scan') 도, P2) → (best 또는 None, 포락선
        거부 사유 — 계획이 없으면 '')."""
        best = self._plan_acquisition(name)
        if best is None:
            return None, ""
        legs = [(best["T_in"], phase, f"acquire_{name}_in", "")]
        b_trk = 0.0
        if name == "S":
            b_trk, _bb = self._track_length_m(float(self.job_c.standoff_m))
            if b_trk > 0.0:
                T_trk = np.array(best["T_in"], dtype=float, copy=True)
                T_trk[:3, 3] += unit(self.u0) * b_trk
                legs.append((T_trk, "scan", "S_track", ""))
        env_why = self._legs_pass_envelope(best["T_pre"], legs, why=True)
        if env_why:
            return None, env_why
        if len(legs) > 1:
            self._decide("acquire", f"{name}: envelope OK for the straight leg onto the dwell and the S track ({b_trk * 1e3:.0f} mm "
                                    "inward along the rough S->E direction, 'scan' rules) - checked before the joint-space move")
        return best, ""

    def _choose_visit_order(self, name: str, phase: str) -> tuple:
        """1G 방문 순서 (D7): 두 순서 (지금 = 로봇 베이스에 가까운 착점부터, 다른 끝부터) 를 모두 이동 전에 계획해 보고 (_acquire_plan — IK·추종·
        경로 검사·yaw 띠·포락선), 되는 것 중 시작 자세에서의 획득 이동이 작은 순서 — the least TCP motion (TCP = the torch tip, KIRO field
        2026-10-14): first the physical tool orientation with the smaller rotation of the TCP frame from the start pose (the torch and
        its wire do not swing around; both orders offer the same two orientations), then the torch tip's distance (which end first),
        then the planned joint-space duration. An exact tie keeps the current order (the hit point nearer to the robot base). Pruning
        (exact): when the current order has the less-rotating orientation, the other order is not planned if its torch tip would
        travel farther with that orientation (poses only). 한 순서만 되면 그 순서 (거부 사유와 상관없이 — IK·추종·경로·포락선 모두, D6).
        → (best 또는 None, 사유). 고른 순서로 대략 작업을 남긴다."""
        opts = []
        rot_side = {self._tool_side(sg): m[1] for sg, m in self._acquisition_motions().items()}   # orientation -> TCP rotation
        rank = {sd: k_ for k_, sd in enumerate(sorted(rot_side, key=lambda sd: rot_side[sd]))}

        def side(o):
            return int(o["best"]["sign"]) * (-1 if o["swapped"] else 1)

        def key(o):
            return (rank.get(side(o), len(rank)), float(o["best"]["tcp_move_m"]), o["best"]["route"] != "straight",
                    float(o["best"]["duration_s"]))
        for i in range(2):
            if i == 1:
                self._derive_jobs(self._swap_rough(self.job0))            # 다른 끝부터 (이동 전, 계획만)
                o0 = opts[0]
                if o0["best"] is not None and rank.get(side(o0)) == 0:
                    m1 = self._acquisition_motions().get(side(o0) * (-1 if self.res.rough_swapped else 1))
                    if m1 is not None and m1[0] > float(o0["best"]["tcp_move_m"]):
                        opts.append({"swapped": bool(self.res.rough_swapped), "first_visited": "end" if self.res.rough_swapped else "start",
                                     "best": None, "lb_tcp_m": m1[0], "vo": {},
                                     "d_first_m": float(np.linalg.norm(np.asarray(self.job0.start, dtype=float))),
                                     "why": (f"not planned: with the same tool orientation its TCP would travel {m1[0] * 1e3:.0f} mm "
                                             f"(> {float(o0['best']['tcp_move_m']) * 1e3:.0f} mm, poses only)")})
                        continue
            best, env_why = self._acquire_plan(name, phase)
            vo = {k_: self.res.visit_order.get(k_) for k_ in ("options", "progression_sign", "tcp_move_mm", "tcp", "tool_rotation_deg",
                                                              "joint_motion_deg", "duration_s", "duration_bound_s", "route", "joint_plan")}
            d_first = float(np.linalg.norm(np.asarray(self.job0.start, dtype=float)))
            why = (env_why or getattr(self, "_acq_fail", "") or "no arm configuration reaches the acquisition pose and follows the coarse "
                   "line") if best is None else ""
            opts.append({"swapped": bool(self.res.rough_swapped), "first_visited": "end" if self.res.rough_swapped else "start",
                         "best": best, "why": why, "vo": vo, "d_first_m": d_first})
        feas = [k_ for k_, o in enumerate(opts) if o["best"] is not None]
        rec = [{"first_visited": o["first_visited"], "feasible": o["best"] is not None,
                "tcp_move_mm": None if o["best"] is None else float(o["best"]["tcp_move_m"]) * 1e3,
                "tool_rotation_deg": (o["best"] or {}).get("tool_rot_deg"),
                "duration_s": (o["best"] or {}).get("duration_s"), "route": (o["best"] or {}).get("route"),
                "joint_motion_deg": (o["best"] or {}).get("joint_motion_deg"),
                "progression_sign": (o["best"] or {}).get("sign"), "base_distance_m": o["d_first_m"], "why": o["why"],
                **({"pruned": True, "tcp_move_lower_bound_mm": o["lb_tcp_m"] * 1e3} if "lb_tcp_m" in o else {})}
               for o in opts]
        self.res.visit_order["order_options"] = _jsonable(rec)
        self.res.visit_order["order_rule"] = ("1G: the visiting order with the least TCP motion (TCP = the torch tip) from the start pose "
                                              "to the acquisition: the tool orientation with the smaller TCP rotation (the torch and wire "
                                              "do not swing around), then the torch tip's distance, then the planned joint-space duration "
                                              "of its arm configuration; an exact tie keeps "
                                              "the hit point nearer to the robot base; the other order is not planned when its least "
                                              "possible TCP motion is larger; an order whose acquisition is refused before motion (IK / "
                                              "follow / path / envelope) is not used")
        if not feas:
            # 둘 다 안 된다: 지금 (다른 끝) 상태 → 원래 순서로 되돌리고 거부
            self._derive_jobs(self._swap_rough(self.job0))
            self.res.visit_order.update(opts[0]["vo"])
            return None, opts[0]["why"] + f" (the other end first: {opts[1]['why']})"
        k = 1 if (len(feas) == 2 and key(opts[1]) < key(opts[0])) else feas[0]
        if k == 0:
            self._derive_jobs(self._swap_rough(self.job0))                # 원래 순서로
        o = opts[k]
        self.res.visit_order.update(o["vo"])
        self.res.visit_order["first_visited"] = o["first_visited"]
        near = opts[0]
        tie = len(feas) == 2 and key(opts[0]) == key(opts[1])
        pruned = any("lb_tcp_m" in o_ for o_ in opts)
        why = ("the least TCP motion from the start pose (the other order cannot move the TCP less)" if pruned else
               "the only order whose acquisition is not refused before motion" if len(feas) == 1 else
               ("equal moves - the hit point nearer to the robot base first" if tie else
                "the least TCP motion from the start pose"))
        self.res.visit_order["order_choice"] = why
        self.res.visit_order["agrees_with_near_end"] = bool(k == 0)
        self._decide("setup", f"visit order: taught {o['first_visited']} first - {why} ("
                     + "; ".join(f"{r_['first_visited']} first: " + (f"TCP {r_['tcp_move_mm']:.0f} mm, rotation "
                                                                   f"{r_['tool_rotation_deg']:.0f} deg, {r_['duration_s']:.1f} s "
                                                                   f"({r_['route']}), progression {r_['progression_sign']:+d}"
                                                                   if r_["feasible"] else
                                                                   (r_["why"] if r_.get("pruned") else f"refused: {r_['why']}"))
                                 + f", base distance {r_['base_distance_m']:.3f} m" for r_ in rec)
                     + ")" + ("" if k == 0 else f" - differs from the nearer-to-the-base order ({near['first_visited']} first)"),
                     visit_order=_jsonable(self.res.visit_order))
        return o["best"], ""

    def _acquire_site(self, name: str, postures: Optional[list] = None) -> None:
        """postures (3G: 끝 자세 후보, 좋은 순서): 도달되는 첫 자세로. 1G 는 자세 하나 (센서 ∥ 중력) — S 는 두 방문 순서 중 시작 자세에서의
        획득 이동이 짧은 것 (_choose_visit_order). → MoveIt 관절 이동으로 물러난 자세 (the trajectory the planning checked, sent
        unchanged) → 판 쪽 직선 (포락선) 으로 드웰. The 'kept configuration' decision names the configuration actually executed, after
        the move (KIRO field 2026-10-14: it named another one before)."""
        L = self._site_label                                    # 로그·이동 이름의 끝 이름 (legacy: 내부 이름 그대로)
        best = None
        phase = "approach_normal" if self.orientation == "vertical_3G" else "descend"
        env_why = ""
        if postures is None and self.orientation == "flat_1G" and name == "S":
            best, why = self._choose_visit_order(name, phase)
            if best is None:
                raise _Abort(STATUS_REFUSED, f"{L(name)}: {why} - the specimen is out of reach or too close to the robot for this tool, or "
                                             "the straight legs are refused by the envelope" + ("" if self._motion_sent else " (nothing sent)"),
                             category="motion")
        else:
            for i, r in enumerate(postures or [None]):
                if r is not None:
                    job = self._apply_3g_end_posture(self.job0, r)
                    self.job0 = job
                    self._derive_jobs(job)
                    if self.plate_plane_measured is not None and self._first_loc is not None:
                        self._raise_plate_top(self._first_loc)
                best, env_why = self._acquire_plan(name, phase)
                if best is not None:
                    break
                if env_why:
                    self._decide("posture", f"{L(name)}: the straight leg onto the dwell would be refused by the envelope ({env_why}) - "
                                            "next candidate" if r is not None else f"{L(name)}: {env_why}")
                    if r is None:
                        break
                    continue
                if r is not None:
                    self._decide("posture", f"{L(name)}: end posture {r['tilt_deg']:.0f} deg torch {'above' if r['torch_above'] else 'below'} is "
                                            "not reachable along the coarse line - next candidate")
                if i >= REACH_CANDIDATE_BUDGET - 1:
                    break
            if best is None:
                if env_why:
                    raise _Abort(STATUS_REFUSED, f"{L(name)}: the straight leg onto the dwell is refused by the envelope: {env_why}"
                                                 + ("" if self._motion_sent else " (nothing sent)"), category="motion")
                raise _Abort(STATUS_REFUSED, f"{L(name)}: no arm configuration reaches the acquisition pose and follows the coarse line - the "
                                             "specimen is out of reach or too close to the robot for this tool"
                                             + ("" if self._motion_sent else " (nothing sent)"), category="motion")
        self._set_progression(best["sign"])

        def kept(c) -> None:                                    # the configuration actually executed (dry run: the selected one)
            self._decide("acquire", f"{L(name)}: progression {best['sign']:+d}, arm configuration {np.round(np.degrees(c['q']), 1).tolist()} "
                                    f"deg (kept for the whole job: the following moves are continuous) - joint-space move to "
                                    f"{best['retreat'] * 1e3:.0f} mm in front of the dwell, then {phase} to the dwell", chosen=_jsonable(c))
        if self.dry_run:
            kept(best["selection"]["cand"])
            return
        self._sensor()                                          # 센서 연결·초기화는 첫 이동 전에 (실패하면 아무것도 보내지 않았다)
        kept(self._joint_move(best["ranked"], best["T_pre"], f"acquire_{L(name)}",
                              f"{best['retreat'] * 1e3:.0f} mm in front of the {L(name)} dwell", selection=best["selection"]))
        self._local_move(best["T_in"], phase, f"acquire_{L(name)}_in", speed=self.descend_speed,
                         extra=f"(onto the {L(name)} dwell, {best['retreat'] * 1e3:.0f} mm toward the plate)")

    def _progression_signs(self) -> list:
        """거친 단계의 진행 부호 후보: 3G 는 끝 자세와 함께 고른 부호 (토치 위/아래 — 지지면 여유가 그 자세로 계산됐다), 1G 는 둘 다
        (도달성으로 고른다)."""
        if self.orientation == "vertical_3G" and self._tilt_sign is not None:
            return [int(self._tilt_sign)]
        return [1, -1]

    def _set_progression(self, sign: int) -> None:
        if sign != int(self.job_c.progression_sign):
            self.job_c = ScanJob.from_dict({**job_to_dict(self.job_c), "progression_sign": sign})
            ref = ScanJob.from_dict({**job_to_dict(self.job_ref), "progression_sign": sign})
            ref.plate_top_measured_m = self.job_ref.plate_top_measured_m
            ref.plate_plane_measured = self.job_ref.plate_plane_measured
            ref.floor_z_m = self.job_ref.floor_z_m
            self.job_ref = ref
        self.res.coarse_progression_sign = sign

    def _acquire(self) -> None:
        """범용 시작 (사용자 2026-10-07): 로봇은 아무 자세 (운송용 접힌 자세 — 센서는 시편을 못 볼 수 있다) 에서 시작한다. 입력은 착점
        좌표·자세·규격뿐. 1) MoveIt 충돌 장면 (시편 + 측정 전 보수적 바닥) 2) 3G: 지지면 probe → 지지면 → 끝 자세 (내려다보는 각·토치 쪽:
        품질 × 도달, 지지물에 닿지 않게) 3) 첫 착점 획득 (손목 형태는 도달성으로 한 번). 그 뒤는 예전과 같다 (드웰 → 보정 → 패스)."""
        self._universal_ready()
        fr = self.ifr
        dm = self._spec_dims_m()
        floor0 = conservative_floor_height(fr, thickness_m=dm["t"], backing_m=dm["b"], face_offset_m=self._hit_face_m())
        why0 = "conservative before measurement: " + ("the lower hit point" if fr.position == "3G" else "hit point - plate thickness - backing")
        self._set_cell(floor0, why0)
        if self._go_start_pose():
            self._set_cell(floor0, why0)                         # 시작 자세에서 다시 (지금 자세에 맞춰 깎은 바닥 판을 되돌린다)
        postures = None
        if self.orientation == "vertical_3G":
            self._support_probe()
            if self.job0.table_z_m is None:                    # dry run: 측정 없이 보수적 바닥을 지지면으로
                j = ScanJob.from_dict({**job_to_dict(self.job0), "torch_lateral_m": tuple(self.job0.torch_lateral_m), "table_z_m": floor0})
                j.plate_top_measured_m = self.job0.plate_top_measured_m
                self.job0 = j
            postures = self._choose_3g_end_postures(self.job0)
        self._acquire_site("S", postures)

    def _start_pose_goal(self) -> Optional[np.ndarray]:
        """job 문서의 start_pose (교시 도구가 착점보다 먼저 손으로 지정한 관절각) → 플랫폼 체인 순서의 관절 목표 [rad]. 없으면 None (예전
        job: 지금 자세에서 시작). 관절 이름이 플랫폼 체인을 덮지 않거나 값이 유한하지 않으면 REFUSED."""
        sp = self.doc.get("start_pose") or {}
        if not sp:
            return None
        names = [str(n) for n in (sp.get("joint_names") or [])]
        q = [float(v) for v in (sp.get("joints_rad") or [])]
        chain = self.rs.chain() if callable(getattr(self.rs, "chain", None)) else None
        want = list(chain.joint_names) if chain is not None else None
        if want is None:
            raise _Abort(STATUS_REFUSED, f"the job has a start_pose but this platform ({self.rs.adapter}) has no joint chain to move "
                                         "to it (nothing sent)", category="config")
        m = dict(zip(names, q)) if len(names) == len(q) else {}
        missing = [n for n in want if n not in m]
        if missing:
            raise _Abort(STATUS_REFUSED, f"job start_pose joints {names} do not cover the arm joints {want} (missing {missing}; nothing "
                                         "sent) - teach the initial posture again on this robot", category="input")
        g = np.array([m[n] for n in want], dtype=float)
        if not np.all(np.isfinite(g)):
            raise _Abort(STATUS_REFUSED, f"job start_pose has non-finite joint values {g.tolist()} (nothing sent)", category="input")
        return g

    def _go_start_pose(self) -> bool:
        """시작 자세 (사용자 2026-10-08: '시작하면 수동으로 먼저 초기자세를 지정하고, 두 대략 시/종점을 지정한 다음 시작'): job 의
        start_pose 가 있으면 첫 이동으로 그 관절각에 MoveIt 관절 공간 이동 (충돌 장면 = 시편 + 보수적 바닥, 확인) — 범용 시작의 계획 (지지면
        probe·끝 자세·획득) 은 그 자세에서 한다. 이미 거기면 (플랜지가 도착 허용 안) 보내지 않는다. dry run 은 경로만 확인.
        반환: 이동했나."""
        g = self._start_pose_goal()
        if g is None:
            self._decide("acquire", "no start_pose in the job (initial posture not taught): the run starts from the current arm pose")
            return False
        fk = self.rs.fk() if callable(getattr(self.rs, "fk", None)) else None
        T_goal = np.asarray(fk(g), dtype=float) if fk is not None else self._T_now()
        if fk is not None and not start_pose_mismatch(T_goal, self._T_now(), self.policy.goal_tol_m, self.policy.goal_tol_deg):
            self._decide("acquire", f"start pose: the arm is already at the taught initial posture {np.round(np.degrees(g), 1).tolist()} deg")
            return False
        what = ("the initial posture taught by hand (it must be clear of the specimen box and the floor keep-out, each padded by the "
                "clearance)")
        if self.dry_run:
            be = self.rs.backend
            v = min(float(self.transit_speed), self.cfg.v_eff_max_mps)
            chk = be.check_joint(g, v, self.accel, timeout_s=self.motion_timeout_s)
            if not chk.get("ok"):
                raise _Abort(STATUS_REFUSED, f"start_pose: no collision-free joint-space path to {what}: {chk.get('reason')} (nothing sent)",
                             category="motion")
            pl = chk.get("plan") or {}
            self._decide("acquire", f"start pose (dry run, not moved) to {np.round(np.degrees(g), 1).tolist()} deg: {move_plan_text(pl)} - "
                                    "the plan below is from the current pose, the real run plans from the initial posture")
            return False
        self._sensor()                                          # 센서 연결·초기화는 첫 이동 전에 (실패하면 아무것도 보내지 않았다)
        self._joint_move([{"q": g}], T_goal, "start_pose", what)
        self._decide("acquire", f"start pose reached: {np.round(np.degrees(g), 1).tolist()} deg - the universal start plans from here")
        return True

    def _reacquire_if_oblique(self, loc: StaticLocalization) -> StaticLocalization:
        """3G 첫 드웰: 잰 판 법선이 지금 자세의 법선 추정과 그 자세가 견디는 각 (_posture_normal_tol_deg: 끝 자세 선택의 지지면·판면 여유
        ÷ 피벗 (착점) 에서 가장 먼 툴 점까지 반경, 그리고 레이저 기울기 상한까지 남은 각) 넘게 다르면 잰 법선을 마주 보도록
        자세를 다시 골라 (끝 자세는 법선에 따라 달라진다) 다시 획득한다 — 보정 루프로 끌고 가지 않는다. 매우 비스듬한 첫 시야 (가짜 팔
        연구실 배치: 추정이 35° 비껴 + 35° 내려다봄) 에서는 판 맞춤이 법선을 절반쯤만 잰다 — 정면에 가까워질수록 정확해지므로 상한 안에
        들 때까지 되풀이 (최대 max_corrections 번, 정렬 반복과 같은 상한). 판 평면 측정은 매번 새 기준 작업에."""
        self._first_loc = loc
        if self.orientation != "vertical_3G" or not loc.plate_ok:
            return loc
        lim = self._posture_normal_tol_deg()
        n_done = 0
        while True:
            n_m = unit(np.asarray(loc.plate_normal_base, dtype=float))
            if float(n_m @ self.n0) < 0.0:
                n_m = -n_m
            ang = _angle_deg(n_m, self.n0)
            if ang <= lim:
                self._decide("acquire", f"S: measured plate normal {ang:.1f} deg from the current one (<= {lim:.1f} deg the posture "
                                        "tolerates)"
                                        + (f" after {n_done} re-acquisition(s)" if n_done else "") + " - aligned by the dwell corrections")
                return loc
            # 반복 예산만 (4d 검토: '각이 줄지 않으면 넘김' 은 쓰지 않는다 — 각은 직전 추정 대비라 매우 비스듬한 첫 시야가 법선을 덜 재면
            # 둘째 값이 더 클 수 있다: 가짜 팔 35° 배치 15.4° → 20.8° → 정면에 가까워지며 수렴)
            if n_done >= int(self.at.max_corrections) or not loc.plate_ok:
                self._decide("acquire", f"S: measured plate normal still {ang:.1f} deg off after {n_done} re-acquisition(s) - the dwell "
                                        "corrections take over")
                return loc
            n_done += 1
            self._reacquired = True
            u = self.u0
            n_new = unit(n_m - u * float(n_m @ u))
            self._decide("acquire", f"S: measured plate normal {np.round(n_new, 4).tolist()} is {ang:.1f} deg from the current "
                                    f"{np.round(self.n0, 4).tolist()} (> {lim:.1f} deg the posture tolerates) - re-acquisition {n_done}: choosing "
                                    "the posture again and acquiring facing the measured plate", angle_deg=ang)
            T_back = np.array(self._T_now(), dtype=float, copy=True)
            T_back[:3, 3] += self.n0 * float(self.job0.safe_clearance_m)
            self._local_move(T_back, "retreat_normal", f"reacquire{n_done}_back", extra="(back off before facing the measured plate)")
            job = ScanJob.from_dict({**job_to_dict(self.job0), "torch_lateral_m": tuple(self.job0.torch_lateral_m),
                                     "surface_normal": n_new.tolist()})
            job.plate_top_measured_m = self.job0.plate_top_measured_m
            self.job0 = job
            self._derive_jobs(job)
            self._raise_plate_top(loc)                              # 잰 판 평면을 새 기준 작업에도
            self._acquire_site("S", self._choose_3g_end_postures(self.job0))
            self._n_reacq = n_done
            loc = self._dwell(f"S_reacquired{n_done}", scene="record")
            self._first_loc = loc

    # ---- 3. 드웰 S + 탐색 -----------------------------------------------------------------------------
    def _no_surface_hint(self, loc: StaticLocalization) -> tuple:
        """M-14: 전부 NaN 드웰의 진단 → (원인, 설명). 이 패키지에는 스트림 감시 통계가 없어 원인은 'ambiguous' 하나 — 드웰의 원시 z 와
        센서 유효 창을 밝힌다."""
        z_lo, z_hi = float(self.settings.z_valid_mm[0]), float(self.settings.z_valid_mm[1])
        raw = loc.diag.get("raw_z_axis_mm")
        return "ambiguous", f"raw z_c {raw}; valid window {z_lo:.0f}-{z_hi:.0f} mm"

    def _range_to_plate(self, T_bf: np.ndarray) -> float:
        """광학 원점에서 광축을 따라 기준 판 평면까지 거리 [m] (실측 판 평면 > 실측 판 상면 높이 > 거친 착점 평면; 판을 등지면 NaN)."""
        T = np.asarray(T_bf, dtype=float) @ self.calib.T_flange_opt
        o, Z = T[:3, 3], T[:3, 2]
        pp = self.job_ref.plate_plane() if self.job_ref is not None else None
        if pp is not None:
            P0, n = pp[0], unit(pp[1])
        else:
            P0, n = np.asarray(self.job_c.start, dtype=float), unit(self.n0)
            if self.plate_top is not None and self.orientation == "flat_1G":
                P0 = P0 + self.up * (float(self.plate_top) - float(P0 @ self.up))
        den = -float(Z @ n)
        return float((o - P0) @ n) / den if den > 1e-6 else float("nan")

    def _approach_blind(self, name: str, loc: StaticLocalization, scene="judge") -> StaticLocalization:
        """보이는 면 없음 (거친 거리 = 센서 창 먼 쪽): 광축을 따라 패스 거리 (광축 교차 거리 — 신호·정확도가 가장 좋은 거리) 로 다가가
        한 번 더 본다. 가드가 보는 직선 (아무것도 못 잰 동안 G3 는 광축 앞 먼 한계까지만 허용) 이고 포락선이 판·지지면 한계를 본다. 거기서
        보이면 이 실행의 거친 거리를 그 거리로 바꾼다 (보정이 다시 안 보이는 거리로 물러나지 않게). 가깝거나 가려진 증거가 있으면 다가가지
        않고 멈춘다 — 물러나는 규칙은 없다 (가까운 쪽은 가드 G0 가 막는다)."""
        why, hint = self._no_surface_hint(loc)
        self._decide("search", f"{name}: no surface in the view ({hint})")
        T0 = self._T_now()
        r_now = self._range_to_plate(T0)
        r_to = self.pass_plate_m
        step = r_now - r_to
        if why == "near" or not (_fin(step) and step > self._sv().pose_tol_m):
            # 다가가지 않는 경우 (가깝거나 가려짐 / 이미 광축 교차 거리 안): 같은 자세에서 한 번 더 보고 (확인 먼저, 4d P5) 멈춘다
            self._decide("search", f"{name}: measuring again in place before stopping (confirm-first)")
            loc2 = self._dwell(f"{name}_again", scene=scene)
            if classify_localization(loc2) != "no_surface":
                return loc2
            why, hint = self._no_surface_hint(loc2)
            if why == "near":
                raise _Abort(STATUS_ABORTED, f"{name}: no surface in the view (twice): {hint}")
            raise _Abort(STATUS_ABORTED, f"{name}: no surface in the view at {r_now * 1e3:.0f} mm from the rough plate (not farther than "
                                         f"the optical cross distance {r_to * 1e3:.0f} mm), twice: {hint}")
        Z = (T0 @ self.calib.T_flange_opt)[:3, 2]
        T_to = np.array(T0, dtype=float, copy=True)
        T_to[:3, 3] += Z * step
        d_n = float((Z * step) @ self.n0)
        d_ip = float(np.linalg.norm(Z * step - self.n0 * d_n))
        self._decide("search", f"{name}: approach {step * 1e3:.0f} mm along the optical axis to the optical cross distance "
                               f"({r_to * 1e3:.0f} mm from the rough plate) and look again")
        self._local_move(T_to, self._leg_phase(d_n, d_ip, orientation=self.orientation, tol_m=self._sv().pose_tol_m), f"{name}_approach",
                         extra=f"(along the optical axis {step * 1e3:.0f} mm)")
        loc2 = self._dwell(f"{name}_approach", scene=scene)
        if classify_localization(loc2) == "no_surface":
            _w, hint2 = self._no_surface_hint(loc2)
            raise _Abort(STATUS_ABORTED, f"{name}: no surface in the view at the coarse range nor at the optical cross distance: {hint2} "
                                         "- check the rough points (re-teach on the groove)")
        jc = self.job_c
        self.job_c = ScanJob.from_dict({**job_to_dict(jc), "torch_lateral_m": tuple(jc.torch_lateral_m), "standoff_m": float(r_to)})
        self._decide("search", f"{name}: surface seen at the optical cross distance - the coarse stage continues at "
                               f"{self.job_c.standoff_m * 1e3:.0f} mm (plate normal {self._coarse_plate_m() * 1e3:.0f} mm)")
        return loc2

    def _find_groove_at(self, name: str, loc: Optional[StaticLocalization] = None) -> StaticLocalization:
        loc = self._dwell(name) if loc is None else loc
        cls = classify_localization(loc)
        if cls == "no_surface":
            loc = self._approach_blind(name, loc)
            cls = classify_localization(loc)
        if cls == "groove":
            return loc
        if not loc.plate_ok:
            # 판이 없다 (어지러운 장면): 같은 자세에서 한 번 더 재고 (확인 먼저, 4d P5) 그래도 없으면 멈춘다
            self._decide("search", f"{name}: plate not measured ({cls}) - measuring again in place (confirm-first)")
            loc = self._dwell(f"{name}_again")
            cls = classify_localization(loc)
            if cls == "groove":
                return loc
            if not loc.plate_ok:
                raise _Abort(STATUS_ABORTED, f"{name}: plate not measured twice ({cls}, flags={loc.flags})")
        return self._lateral_search(name, loc)

    def _lateral_search(self, name: str, loc: StaticLocalization) -> StaticLocalization:
        """대비책 측면 탐색 (레이저 라인 방향, 판 거리 일정): 그루브를 찾은 드웰 또는 ABORTED. P2: S 에서는 S 추적이 설계 일치 단면을 하나도
        못 봤을 때 ('none') 만, E 에서는 트랜싯 맞춤이 없을 때만."""
        cls = classify_localization(loc)
        # 면내 탐색: 레이저 라인 방향(X_opt 의 판면 투영) 으로, 판 거리 일정. 간격·범위는 센서 시야 모형 (FOV(잰 거리) − 개구, 시편 폭)
        e_lat = loc.lateral_axis_base if np.all(np.isfinite(loc.lateral_axis_base)) else unit(
            loc.T_base_opt[:3, 0] - self.n0 * float(loc.T_base_opt[:3, 0] @ self.n0))
        T_base = self._T_now()
        r = float(loc.diag.get("raw_z_axis_mm", float("nan"))) * 1e-3
        if not _fin(r):
            r = self._range_to_plate(T_base)
        dm = self._spec_dims_m()
        steps = [round(v * 1e3, 1) for v in self._sv().search_steps_m(r, dm["opening"], dm["w"])]
        basis = (f"FOV {self._sv().fov_m(r) * 1e3:.0f} mm at {r * 1e3:.0f} mm - opening {dm['opening'] * 1e3:.1f} mm, over the specimen "
                 f"width {dm['w'] * 1e3:.0f} mm")
        if not steps:
            self._decide("search", f"{name}: {cls}: no groove in a view that already covers the specimen width ({basis}) - measuring "
                                   "again in place (confirm-first)")
            loc2 = self._dwell(f"{name}_again")
            if classify_localization(loc2) == "groove":
                return loc2
            raise _Abort(STATUS_ABORTED, f"{name}: {cls}: no groove in a view that already covers the specimen width ({basis}), twice - "
                                         "the groove does not return at this range or the rough point is off the specimen: re-teach")
        order = search_order(steps, loc)
        self._decide("search", f"{name}: {cls} (no groove) -> lateral search along the laser line, steps {order} mm ({basis})")
        for j, step in enumerate(order):
            T_to = np.array(T_base, dtype=float, copy=True)
            T_to[:3, 3] += e_lat * (step * 1e-3)
            self._local_move(T_to, "scan", f"{name}_search{j + 1}", extra=f"(lateral search {step:+.0f} mm)")
            loc_j = self._dwell(f"{name}_search{j + 1}")
            cls_j = classify_localization(loc_j)
            if cls_j == "groove":
                self._decide("search", f"{name}: groove found at lateral step {step:+.0f} mm (lateral offset {loc_j.lateral_offset_m * 1e3:+.1f} mm)")
                return loc_j
            self._decide("search", f"{name}: step {step:+.0f} mm -> {cls_j} (flags {loc_j.flags})")
        raise _Abort(STATUS_ABORTED, f"{name}: groove not found within the lateral search {order} mm ({basis}) "
                                     "(rough point off the specimen: re-teach)")

    # ---- 4. 프로브 ---------------------------------------------------------------------------------
    def _probe_move(self, T0: np.ndarray, loc0: StaticLocalization, ax: int, nm: str, sg: float, s_m: float, x0: float,
                    tag: str = "", scene="judge") -> tuple:
        """프로브 한 축: T0 에서 광학 축 ax 로 sg·s 옮겨 재고 (그루브를 못 재면 같은 자세에서 한 번 더) → (dx_c mm, FK 이동 mm, 그 드웰)."""
        R = (np.asarray(T0, dtype=float) @ self.calib.T_flange_opt)[:3, :3]
        T_to = np.array(T0, dtype=float, copy=True)
        T_to[:3, 3] += R[:, ax] * (sg * s_m)
        self._local_move(T_to, "scan", f"probe_{nm}{tag}", extra=f"({sg * s_m * 1e3:+.1f} mm along {nm}_opt)")
        lp = self._dwell(f"probe_{nm}{tag}", self.probe_frames, scene=scene)
        xc = float(lp.diag.get("x_c_mm", float("nan")))
        if not _fin(xc):
            lp = self._dwell(f"probe_{nm}{tag}_again", self.probe_frames, scene=scene)     # 확인 먼저: 같은 자세에서 한 번 더 (이동 없음)
            xc = float(lp.diag.get("x_c_mm", float("nan")))
        dx = xc - x0 if (_fin(xc) and _fin(x0)) else float("nan")
        d_meas = float((np.asarray(lp.opt_origin_base) - np.asarray(loc0.opt_origin_base)) @ R[:, ax]) * 1e3
        self._decide("probe", f"{sg * s_m * 1e3:+.1f} mm along {nm}_opt (FK {d_meas:+.2f} mm) -> groove x_c {x0:+.2f} -> {xc:+.2f} mm "
                              f"(dx {dx:+.2f})")
        return dx, d_meas, lp

    def _probe_step_m(self, loc: StaticLocalization) -> tuple:
        """프로브 스텝 s = 2·(e_hit(r) + k·√2·q(r)) — 판정 띠 (s/2) 가 착점의 도착 오차와 읽기 정밀을 담는 가장 작은 스텝 → (s, r, e_hit, q)."""
        r = self._loc_range_m(loc, float(self.job_c.standoff_m))
        k = float(self.settings.judgement_k)
        q = _ms.quant_sigma_mm(r * 1e3) * 1e-3
        e_hit = self._hit_tol_m(r)
        return 2.0 * (e_hit + k * math.sqrt(2.0) * q), r, e_hit, q

    def _probe(self, loc: StaticLocalization, axes: tuple = ("X", "Y"), scene="judge") -> StaticLocalization:
        """2축 프로브 (4d P4): 스텝 s = 2·(e_hit(r) + k·√2·q(r)) — 판정 띠 (s/2) 가 착점의 도착 오차와 읽기 정밀을 담는 가장 작은 스텝
        (r = 이 드웰의 광축 거리). X 스텝은 그루브가 시야 가운데 쪽으로 가는 부호 (sgn = sign(x_c0)). 확인 먼저: 첫 'inconsistent' 는 X/Y
        한 쌍을 한 번 되풀이해 'confirmed' 여야 하고, 첫 'flip' 은 캘리브를 뒤집은 뒤 X 한 번이 'confirmed' 여야 한다 (아니면 ABORTED).
        확인되면 프로브 전 국소화를 그대로 돌려준다 (같은 자세; 뒤집었을 때만 다시 잰다).

        axes=('X',) (P2 새 흐름): 심 방향은 S 추적 스트림이 재므로 Y_opt 스텝과 복귀 이동이 없다 — 판정은 probe_verdict_x, 확인 먼저는 지금
        자세에서 X 한 번 더, 뒤집기도 지금 자세 기준. 돌려주는 국소화 = 마지막 X 드웰 (팔이 있는 곳 — S_align 의 그루브 점 G 가 여기서)."""
        if tuple(axes) == ("X",):
            return self._probe_x(loc, scene=scene)
        T0 = self._T_now()
        x0 = float(loc.diag.get("x_c_mm", float("nan")))
        s, r, e_hit, q = self._probe_step_m(loc)
        k = float(self.settings.judgement_k)
        room = 0.5 * self._sv().fov_m(r) - 0.5 * self._opening_spec_m() - (abs(x0) * 1e-3 if _fin(x0) else 0.0)
        sgn = -1.0 if (_fin(x0) and x0 < 0.0) else 1.0
        R = (T0 @ self.calib.T_flange_opt)[:3, :3]
        # Y_opt (레이저 평면 법선 ≈ 진행 방향) 스텝은 판 **안쪽**(거친 E 쪽) 으로: 거친 S 가 판 끝 가까이 있어도 판 밖으로 나가지 않는다
        y_sign = 1.0 if (self.u0 is None or float(R[:, 1] @ unit(self.u0)) >= 0.0) else -1.0
        probe: dict = {"step_mm": s * 1e3, "x_sign": sgn, "y_step_sign": y_sign, "x_c_before_mm": x0, "range_m": r,
                       "e_hit_mm": e_hit * 1e3, "flip_room_mm": room * 1e3, "axes": list(axes)}
        self._decide("probe", f"probe step {s * 1e3:.2f} mm = 2 x (arrival tolerance at the hit {e_hit * 1e3:.2f} mm + k sqrt2 q "
                              f"{k * math.sqrt(2.0) * q * 1e3:.2f} mm) at {r * 1e3:.0f} mm; X toward the view centre ({sgn:+.0f})"
                              + ("" if room >= s else f"; note: a flipped branch would carry the groove {s * 1e3:.1f} mm outward with "
                                                      f"only {room * 1e3:.1f} mm room"))

        def pair(tag: str) -> tuple:
            dX, dmX, _lp = self._probe_move(T0, loc, 0, "X", sgn, s, x0, tag, scene=scene)
            dY, _, _lp = self._probe_move(T0, loc, 1, "Y", y_sign, s, x0, tag, scene=scene)
            self._local_move(T0, "scan", f"probe_return{tag}", extra="(back to the dwell pose)")
            v, note = probe_verdict(dX, dY, s * 1e3, sgn)
            self._decide("probe", note)
            return dX, dY, dmX, v, note

        dX, dY, dmX, verdict, note = pair("")
        probe.update(dx_c_X_mm=dX, dx_c_Y_mm=dY, d_meas_X_mm=dmX, verdict=verdict, note=note)
        if verdict == "inconsistent":
            self._decide("probe", "first probe reading inconsistent - repeating the X/Y pair once (confirm-first)")
            dX2, dY2, dmX2, v2, note2 = pair("_repeat")
            probe["repeat"] = {"dx_c_X_mm": dX2, "dx_c_Y_mm": dY2, "d_meas_X_mm": dmX2, "verdict": v2, "note": note2}
            if v2 != "confirmed":
                self.res.probe = probe
                raise _Abort(STATUS_ABORTED, f"{note}; repeated: {note2}")
            verdict = probe["verdict"] = "confirmed"
            probe["confirmed_by"] = "repeat"
            dX, dmX = dX2, dmX2
        loc_out = loc
        if verdict == "flip":
            self._flip_calib(probe)
            loc_out = self._dwell("S_after_probe", scene=scene)
            x0f = float(loc_out.diag.get("x_c_mm", float("nan")))
            sgn_f = -1.0 if (_fin(x0f) and x0f < 0.0) else 1.0
            dXf, dmXf, _lp = self._probe_move(T0, loc_out, 0, "X", sgn_f, s, x0f, "_after_flip", scene=scene)
            self._local_move(T0, "scan", "probe_return_after_flip", extra="(back to the dwell pose)")
            ok_f = _fin(dXf) and _ms.design_match(-sgn_f * dXf, s * 1e3)
            probe["after_flip"] = {"x_c_before_mm": x0f, "x_sign": sgn_f, "dx_c_X_mm": dXf, "d_meas_X_mm": dmXf, "confirmed": bool(ok_f)}
            if not ok_f:
                self.res.probe = probe
                raise _Abort(STATUS_ABORTED, f"probe: the flipped optical branch was not confirmed by a second X move ({sgn_f * s * 1e3:+.1f} "
                                             f"mm -> dx_c {dXf:+.2f} mm, expected {-sgn_f * s * 1e3:+.1f} +-{_ms.DESIGN_HALF * s * 1e3:.2f})")
            sgn, dX, dmX = sgn_f, dXf, dmXf
        probe["scale_sensor_mm"] = -sgn * dX if _fin(dX) else float("nan")      # 센서가 본 그루브 이동 (프로브 스텝 방향 +)
        probe["scale_kin_mm"] = sgn * dmX if _fin(dmX) else float("nan")         # 운동학 (FK) 이동
        self.res.probe = probe
        self._event("probe_done", verdict=verdict)
        return loc_out

    def _flip_calib(self, probe: dict) -> None:
        """광학 yaw 분기 뒤집기 (프로브 'flip'): T_flange←opt · Rz(π)."""
        self.calib = flip_optical_branch(self.calib)
        self.res.optical_branch_flipped = True
        probe["calib_after"] = self.calib.to_dict()
        self._decide("probe", "optical branch flipped: T_flange<-opt * Rz(pi); re-localizing and confirming with one X move")

    def _probe_x(self, loc: StaticLocalization, scene="judge") -> StaticLocalization:
        """X 축 프로브만 (P2, 모두 지금 자세 기준 — 복귀 이동 없음): X_opt 로 그루브가 시야 가운데 쪽으로 가는 부호 sgn·s → probe_verdict_x.
        'inconsistent' 는 지금 자세에서 X 한 번 더 (확인 먼저), 'flip' 은 캘리브를 뒤집고 지금 자세에서 다시 잰 뒤 X 한 번이 'confirmed' 여야
        한다. 반환 = 마지막 X 드웰 (팔이 있는 곳)."""
        s, r, e_hit, q = self._probe_step_m(loc)
        k = float(self.settings.judgement_k)
        x0 = float(loc.diag.get("x_c_mm", float("nan")))
        room = 0.5 * self._sv().fov_m(r) - 0.5 * self._opening_spec_m() - (abs(x0) * 1e-3 if _fin(x0) else 0.0)
        probe: dict = {"step_mm": s * 1e3, "x_c_before_mm": x0, "range_m": r, "e_hit_mm": e_hit * 1e3, "flip_room_mm": room * 1e3,
                       "axes": ["X"]}
        self._decide("probe", f"probe (X only - the seam direction comes from the S track) step {s * 1e3:.2f} mm = 2 x (arrival "
                              f"tolerance at the hit {e_hit * 1e3:.2f} mm + k sqrt2 q {k * math.sqrt(2.0) * q * 1e3:.2f} mm) at "
                              f"{r * 1e3:.0f} mm, toward the view centre, from the current pose (no return move)"
                              + ("" if room >= s else f"; note: a flipped branch would carry the groove {s * 1e3:.1f} mm outward with "
                                                      f"only {room * 1e3:.1f} mm room"))

        def x_probe(ref: StaticLocalization, tag: str) -> tuple:
            x_ref = float(ref.diag.get("x_c_mm", float("nan")))
            sg = -1.0 if (_fin(x_ref) and x_ref < 0.0) else 1.0
            dX_, dm_, lp_ = self._probe_move(self._T_now(), ref, 0, "X", sg, s, x_ref, tag, scene=scene)
            v_, note_ = probe_verdict_x(dX_, s * 1e3, sg)
            self._decide("probe", note_)
            return dX_, dm_, lp_, v_, note_, sg, x_ref

        dX, dmX, lp, verdict, note, sgn, x0 = x_probe(loc, "")
        probe.update(x_sign=sgn, dx_c_X_mm=dX, d_meas_X_mm=dmX, verdict=verdict, note=note)
        if verdict == "inconsistent":
            self._decide("probe", "first X probe reading inconsistent - one more X probe from here (confirm-first)")
            dX2, dmX2, lp2, v2, note2, sg2, _x = x_probe(lp, "_repeat")
            probe["repeat"] = {"dx_c_X_mm": dX2, "d_meas_X_mm": dmX2, "verdict": v2, "note": note2, "x_sign": sg2}
            if v2 == "inconsistent":
                self.res.probe = probe
                raise _Abort(STATUS_ABORTED, f"{note}; repeated: {note2}")
            verdict, note, dX, dmX, lp, sgn = v2, note2, dX2, dmX2, lp2, sg2
            probe["confirmed_by"] = "repeat"
            probe["verdict"] = verdict
        loc_out = lp
        if verdict == "flip":
            self._flip_calib(probe)
            here = self._dwell("S_after_probe", scene=scene)
            dXf, dmXf, lpf, vf, notef, sgf, x0f = x_probe(here, "_after_flip")
            probe["after_flip"] = {"x_c_before_mm": x0f, "x_sign": sgf, "dx_c_X_mm": dXf, "d_meas_X_mm": dmXf,
                                   "confirmed": vf == "confirmed"}
            if vf != "confirmed":
                self.res.probe = probe
                raise _Abort(STATUS_ABORTED, f"probe: the flipped optical branch was not confirmed by a second X move ({notef})")
            sgn, dX, dmX, loc_out = sgf, dXf, dmXf, lpf
        probe["scale_sensor_mm"] = -sgn * dX if _fin(dX) else float("nan")
        probe["scale_kin_mm"] = sgn * dmX if _fin(dmX) else float("nan")
        self.res.probe = probe
        self._event("probe_done", verdict=verdict)
        return loc_out

    # ---- 5. 상대 보정 --------------------------------------------------------------------------------
    def _converged(self, loc: StaticLocalization, target: float) -> tuple:
        """드웰 수렴 (4d P3; 원칙 1·5) → (ok, 설명, 정규화 잔차 ρ = max(측면/허용, 거리/허용)).
        측면 허용 = max(e_hit(r), k·q(r)) — 한 번 더 움직여도 착점에서 도착 허용보다 좋아질 수 없다 (r = 이 드웰의 광축 거리);
        거리 허용 = max(광학 원점의 도착 허용, k·q(r)). 롤은 보정은 하지만 판정하지 않는다 (지점 출력 — 루트 중심·판 법선 3D — 은 드웰
        롤에 의존하지 않고, 패스 자세는 L1/n1 에서 온다)."""
        r = self._loc_range_m(loc, target)
        kq = float(self.settings.judgement_k) * _ms.quant_sigma_mm(r * 1e3) * 1e-3
        tol_lat = max(self._hit_tol_m(r), kq)
        tol_so = max(self._origin_tol_m(), kq)
        lat = abs(float(loc.lateral_offset_m)) if _fin(loc.lateral_offset_m) else float("inf")
        so = abs(float(loc.standoff_m) - float(target)) if _fin(loc.standoff_m) else float("inf")
        roll = abs(loc.roll_signed_deg) if _fin(loc.roll_signed_deg) else float("nan")
        rho = max(lat / tol_lat, so / tol_so)
        ok = bool(loc.groove_found and lat <= tol_lat and so <= tol_so)
        return ok, (f"lateral {lat * 1e3:.2f}/{tol_lat * 1e3:.2f} mm, standoff err {so * 1e3:.2f}/{tol_so * 1e3:.2f} mm "
                    f"(arrival tolerance at the hit / origin), roll {roll:.2f} deg (corrected, not judged)"), rho

    @staticmethod
    def _leg_phase(d_along_n_m: float, d_in_plane_m: float, roll_deg: float = 0.0, orientation: str = "flat_1G",
                   tol_m: float = 0.0) -> str:
        """구간 단계 태그 (M-2): 면내 성분 > tol_m (도착 허용 — 그 안의 면내 성분은 도착 오차) 또는 회전이 있으면 'scan' (판 위
        규칙 전부). 순수 법선 이동은
        1G: +n 'ascend' (clamp 면제) / −n 'descend'; 3G: +n 'retreat_normal' (clamp 면제) / −n 'approach_normal' (판 쪽: G2·clamp 밴드)
        — 3G 의 'descend' 는 세운 판과 나란한 하강이라 판 쪽 이동이 아니다 (가드 G2 제외)."""
        if d_in_plane_m > float(tol_m) or abs(float(roll_deg)) > 1e-6:
            return "scan"
        vertical = orientation == "vertical_3G"
        if d_along_n_m < -1e-6:
            return "approach_normal" if vertical else "descend"
        if d_along_n_m > 1e-6:
            return "retreat_normal" if vertical else "ascend"
        return "scan"

    def _correct_until_converged(self, name: str, loc: StaticLocalization, target: float,
                                 max_corr: Optional[int] = None, seam: Optional[SeamEstimate] = None, scene="judge") -> tuple:
        """드웰 보정 루프 → (loc 수렴, 적용된 Correction 목록). 한 번의 보정 = 잰 만큼 그대로 (스텝·누적 상한 없음 — 사용자 2026-10-07:
        안전은 가드·포락선이 보고, 잰 값은 시야·거리 창 안에서만 나온다). 판정은 확인 먼저 (4d P3): 정규화 잔차 ρ 가 두 번 연속 줄지
        않아야 (두 번째가 확인) 발산으로 멈춘다; 예산 (max_corrections) 이 끝났는데 마지막 보정이 아직 줄이고 있었으면 한 번 더 보정하고,
        그래도 수렴하지 않으면 멈춘다. 보정량이 0 이면 같은 자세에서 한 번 더 재고, 다시 0 이면 멈춘다.
        seam (P2): 측정 심 (S 추적·트랜싯) — 면내 보정 축이 측정 심에 수직인 w (relative_correction seam_dir): 착점의 심 방향 이동 0.
        scene: 보정 드웰의 장면 판단 (_localize)."""
        max_corr = self.at.max_corrections if max_corr is None else int(max_corr)
        seam_dir = np.asarray(seam.u, dtype=float) if seam is not None else None
        applied = []
        rho_prev, n_up, extra_used, i = None, 0, False, 0
        while True:
            ok, why, rho = self._converged(loc, target)
            self._decide("correct", f"{name}: {'converged' if ok else 'not converged'} ({why})")
            if ok:
                return loc, applied
            if rho_prev is not None:
                n_up = n_up + 1 if rho >= rho_prev else 0
            if n_up >= 2:
                raise _Abort(STATUS_ABORTED, f"{name}: localization diverging: two consecutive corrections did not reduce the residual "
                                             f"({why}); hand-eye rotation or FK/CB mismatch")
            if i >= max_corr:
                if rho_prev is not None and rho < rho_prev and not extra_used:
                    extra_used = True
                    self._decide("correct", f"{name}: correction budget ({max_corr}) used but the last correction still reduced the "
                                            "residual - one more correction (confirm-first)")
                else:
                    raise _Abort(STATUS_ABORTED, f"{name}: localization not converged after {i} corrections ({why}); hand-eye "
                                                 "rotation or FK/CB mismatch")
            if n_up == 1:
                self._decide("correct", f"{name}: the last correction did not reduce the residual - one more correction to confirm "
                                        "before judging divergence")
            corr = relative_correction(loc, target_standoff_m=target, seam_dir=seam_dir)
            if corr.is_zero:
                self._decide("correct", f"{name}: no correction computed ({corr.reason}) - measuring again in place (confirm-first)")
                loc = self._dwell(f"{name}_redwell{i + 1}", scene=scene)
                ok, why, rho = self._converged(loc, target)
                if ok:
                    self._decide("correct", f"{name}: converged on the repeated reading ({why})")
                    return loc, applied
                corr = relative_correction(loc, target_standoff_m=target, seam_dir=seam_dir)
                if corr.is_zero:
                    raise _Abort(STATUS_ABORTED, f"{name}: no correction possible (twice): {corr.reason}")
            T_now = self._T_now()
            T_to = apply_correction(T_now, self.calib, corr)
            phase = self._leg_phase(corr.d_along_n_m, corr.d_in_plane_m, corr.roll_deg, orientation=self.orientation,
                                    tol_m=self._sv().pose_tol_m)
            self._decide("correct", f"{name} correction {i + 1}: {corr.reason} (I3: delta on the measured flange pose)", correction=corr.to_dict())
            self._local_move(T_to, phase, f"{name}_corr{i + 1}", extra=f"({corr.reason})")
            applied.append(corr.to_dict())
            loc = self._dwell(f"{name}_corr{i + 1}", scene=scene)
            rho_prev = rho
            i += 1

    # ---- 지점 기록 ---------------------------------------------------------------------------------------
    def _site_record(self, name: str, loc: StaticLocalization, target: float, extra: Optional[dict] = None, scene="judge") -> tuple:
        """보정이 끝난 지점(S/E) 의 측정값: 루트·상부 중심 (base), 광학 원점, 광축∩판 점, lateral, standoff, 롤, 판 법선, 판 점 Q,
        그루브 깊이 (n 방향, I2), 마지막 delta_cb → (rec, loc). 루트 중심·판 법선이 없으면 같은 자세에서 한 번 더 재서 (확인 먼저,
        4d P3) 수렴·유한한 읽기로만 지점을 만든다."""
        if not (np.all(np.isfinite(loc.root_centre_base)) and np.all(np.isfinite(loc.plate_normal_base))):
            self._decide("site", f"{name}: groove root centre / plate normal missing in the converged reading - measuring again in place")
            loc2 = self._dwell(f"{name}_site_redwell", scene=scene)
            ok, why, _rho = self._converged(loc2, target)
            if not (ok and np.all(np.isfinite(loc2.root_centre_base)) and np.all(np.isfinite(loc2.plate_normal_base))):
                raise _Abort(STATUS_ABORTED, f"site {name}: groove root centre / plate normal not measured after the corrections "
                                             f"(repeated reading: {why})")
            loc = loc2
        Q = np.asarray(loc.plate_points_base, dtype=float).reshape(-1, 3)
        rec = {"name": name, "root": np.asarray(loc.root_centre_base, dtype=float), "top": np.asarray(loc.top_centre_base, dtype=float),
               "origin": np.asarray(loc.opt_origin_base, dtype=float), "axis_hit": np.asarray(loc.axis_hit_base, dtype=float),
               "lateral_m": float(loc.lateral_offset_m), "standoff_m": float(loc.standoff_m), "roll_deg": float(loc.roll_signed_deg),
               "plate_normal": np.asarray(loc.plate_normal_base, dtype=float),
               "plate_point": np.median(Q, axis=0) if Q.shape[0] else np.full(3, np.nan), "Q": Q,
               "depth_mm": float(loc.diag.get("root_depth_along_n_mm", float("nan"))),
               "delta_cb": self.res.delta_cb[-1] if self.res.delta_cb else None, "scene_mode": (loc.diag or {}).get("scene_mode"),
               "R_opt": np.asarray(loc.T_base_opt, dtype=float)[:3, :3]}
        if extra:
            rec.update(extra)
        self.res.sites[name] = _jsonable({k: v for k, v in rec.items() if k != "Q"} | {"n_plate_points": int(Q.shape[0])})
        self._decide("site", f"{name}: root centre {np.round(rec['root'], 4).tolist()} m, top centre {np.round(rec['top'], 4).tolist()} m, "
                             f"lateral {rec['lateral_m'] * 1e3:+.2f} mm, standoff {rec['standoff_m'] * 1e3:.1f} mm, roll "
                             f"{rec['roll_deg']:+.2f} deg, depth {rec['depth_mm']:.2f} mm")
        return rec, loc

    # ---- P2: 레이저 선 방향 (툴 yaw) 정렬 — S 추적 · S 회전 · 트랜싯 · E 배치 (2026-10-09) ---------------------------------------
    # 툴 yaw 를 대략점 두 개로 한 번 정하고 끝까지 고치지 않으면, 대략 측면 오프셋이 반대 부호일 때 레이저 선이 심에 4.7–17.4° 기울어
    # 선 끝이 심 방향으로 ±FOV/2·sin ψ 퍼져 시편 끝을 넘고 (끝 너머 면을 판으로 잡아 설계 불일치·'물체'), 중심 맞춤이 기운 선을 따라 끝 쪽으로
    # 미끄러졌다 (2026-10-08 1G 배치 중단 10 건 중 9 건). 근본 수정: S 첫 드웰 뒤 안쪽으로 짧게 스트리밍하며 이동해 (S_track, 기선 b) 심 방향을
    # 3D 로 재고, 판 법선 둘레·그루브 점 피벗으로 한 번 돌려 선을 심에 수직으로 (S_align), 그 뒤 S 사이트·트랜싯·E 사이트는 같은 자세. 모든
    # 보정은 측정 심에 수직인 축 w = n×u 로만 (심 방향 미끄럼 0). E 드웰 위치는 트랜싯 스트림의 일관 단면으로. 유도량은 seam_track 모듈 참조.
    def _tol_lat_m(self, r: float) -> float:
        """측면 허용 = max(e_hit(r), k·q(r)) — 드웰 수렴 (_converged) 과 같은 값."""
        kq = float(self.settings.judgement_k) * _ms.quant_sigma_mm(r * 1e3) * 1e-3
        return max(self._hit_tol_m(r), kq)

    def _yaw_tol_rad(self, r: float) -> float:
        """ψ_tol(r) = asin(tol_lat / (FOV/2)): 레이저 선 끝 발자국이 도착 허용 안인 yaw (돌려도 더 좋아지지 않는 각)."""
        return _st.yaw_tolerance_rad(self._sv().fov_m(r), self._tol_lat_m(r))

    def _theta_goal_rad(self) -> float:
        """θ_goal = 플랫폼 도착 자세 허용 (안전 정책 goal_tol_deg)."""
        return math.radians(float(self.policy.goal_tol_deg)) if self.policy is not None else 0.0

    def _qc_tol_m(self, key: str) -> Optional[float]:
        v = ((self.qc_targets or {}).get(key) or {}).get("tol") if isinstance(self.qc_targets, dict) else None
        return float(v) * 1e-3 if _fin(v) else None

    def _tol_root_m(self) -> Optional[float]:
        """설계 루트 위치 허용 tol_root = 작업 qc_targets root_position_mm.tol (용접 경로 겨냥점 허용, 사용자 2026-10-07). 없으면 None —
        다른 설계 허용 (갭 등) 으로 바꾸지 않는다: 그때 S 추적 기선은 이 이동에서 잰 궤적 잔차가 정한다 (_track_extension)."""
        t = self._qc_tol_m("root_position_mm")
        return t if (t is not None and t > 0.0) else None

    def _sigma_sec0_m(self, r: float) -> tuple:
        """이동 전 단면 잡음 상한 σ_0 → (σ_0, 출처): 설계 허용이 있으면 tol_root/k (4a 잡음 규칙상 통과 가능한 단면 잡음의 상한), 없으면 측정
        바닥 = 센서 양자화 σ_q(r) (어떤 읽기도 이보다 작지 않다 — 실제 잡음은 S 추적의 잰 잔차로 다시 정한다)."""
        t = self._tol_root_m()
        if t is not None:
            return t / float(self.settings.judgement_k), "design root_position_mm.tol / k"
        return _ms.quant_sigma_mm(r * 1e3) * 1e-3, "sensor quantization floor (no design root tolerance in the job)"

    def _footprint_m(self, r: float, psi: float, x_m: float) -> float:
        """f(ψ, x) = (FOV(r)/2 + |x|)·sin ψ + e_hit(r): 그루브 교차점에서 레이저 선 끝까지의 심 방향 반폭."""
        return _st.footprint_m(self._sv().fov_m(r), psi, x_m, self._hit_tol_m(r))

    def _psi_res_rad(self, est: SeamEstimate, psi_rem: float = 0.0) -> float:
        """ψ_res = k·σψ_model + θ_goal + |ψ_rem|: 정렬 뒤 남는 레이저 선 yaw 의 상한 (추정 불확도 + 플랫폼 자세 허용 + 실행 못한 회전)."""
        return float(self.settings.judgement_k) * float(est.sigma_psi_model) + self._theta_goal_rad() + abs(float(psi_rem))

    def _track_length_m(self, r: float) -> tuple:
        """S 추적 기선 b = min(b_need(σ_0), cap) (seam_track.track_length_m; σ_0 = _sigma_sec0_m, cap = L_r − L_min − m_E: S 사이트가 E 에서
        최소 거리 + E 여유를 남기는 기하 상한) + 측정 바닥 기선 b_floor = b_need(σ_q) (어떤 읽기로도 이보다 짧은 기선으로는 심 방향이 허용
        안에 들 수 없다) → (b, 근거 dict)."""
        psi_tol = self._yaw_tol_rad(r)
        k = float(self.settings.judgement_k)
        m_E = self._footprint_m(r, psi_tol, 0.0)
        L_r = float(np.linalg.norm(np.asarray(self.job0.end, dtype=float) - np.asarray(self.job0.start, dtype=float)))
        L_min = float(self._min_site_distance_m()[0])
        sig0, src = self._sigma_sec0_m(r)
        b, b_need, cap = _st.track_length_m(sig0, k, psi_tol, self._theta_goal_rad(), L_r_m=L_r, L_min_m=L_min, m_E_m=m_E)
        sig_q = _ms.quant_sigma_mm(r * 1e3) * 1e-3
        b_floor = _st.baseline_m(sig_q, k, psi_tol, self._theta_goal_rad())
        return b, {"b_m": b, "b_star_m": b_need, "cap_m": cap, "b_floor_m": b_floor, "sigma_sec0_m": sig0, "sigma_sec0_source": src,
                   "sigma_q_m": sig_q, "L_r_m": L_r, "L_min_m": L_min, "m_E_m": m_E, "range_m": r,
                   "psi_tol_deg": math.degrees(psi_tol), "theta_goal_deg": math.degrees(self._theta_goal_rad()),
                   "tol_root_m": self._tol_root_m(), "fov_m": self._sv().fov_m(r)}

    @staticmethod
    def _baseline_refusal(bb: dict) -> Optional[str]:
        """대략점·설계만으로 정해지는 S 추적 거부 (이동 전): 기하 상한 cap 이 측정 바닥 기선 b_floor 보다 짧으면 (≤ 0 포함) 어떤 읽기로도 심
        방향을 허용 안으로 잴 수 없다. → 사유 문구 또는 None."""
        if bb["cap_m"] >= bb["b_floor_m"]:
            return None
        return (f"the rough points leave {bb['cap_m'] * 1e3:.1f} mm for the S track (L_r {bb['L_r_m'] * 1e3:.0f} - L_min "
                f"{bb['L_min_m'] * 1e3:.0f} - m_E {bb['m_E_m'] * 1e3:.1f} mm) < the shortest baseline any reading needs "
                f"{bb['b_floor_m'] * 1e3:.1f} mm (sqrt2 x k x sensor quantization {bb['sigma_q_m'] * 1e3:.3f} mm / (psi_tol "
                f"{bb['psi_tol_deg']:.2f} - theta_goal {bb['theta_goal_deg']:.2f} deg)) - the seam direction cannot be measured within the "
                "yaw tolerance; teach the rough points farther apart")

    def _track_fit_args(self, t_hat0, r: float, source: str) -> dict:
        """seam_track.fit_track 인자: k, 설계 루트 허용 (없으면 None — σψ 는 잰 잔차), 진행 쪽, 잔차 바닥 (센서 양자화 q(r))."""
        return {"k": float(self.settings.judgement_k), "tol_root_m": self._tol_root_m(), "t_hat0": [float(v) for v in t_hat0],
                "floor_m": _ms.quant_sigma_mm(r * 1e3) * 1e-3, "source": source}

    def _track_extension(self, est: Optional[SeamEstimate], outcome: str, info: dict, bb: dict, moved_m: float, r: float,
                         t0: np.ndarray) -> tuple:
        """S 추적 연장 (같은 방향, 한 번): 잰 궤적 잔차로 필요한 기선 b_need(σ) (σ = max(설계 상한, 잰 잔차) — est.sigma_sec_m) 보다 직선
        위 단면 범위가 짧으면 그 모자람만큼, 그루브가 늦게 나타나 직선이 없으면 ('late') 계획 기선만큼 — 기하 상한 cap 안에서. 레이저 평면이
        평행이동으로 심을 자르는 점은 이동 d 당 (d·Y)/(u·Y) 만큼 가므로 (Y = 레이저 평면 법선) 모자람 Δs 에는 이동 Δs·|u·Y|/|t̂0·Y|. 필요한
        이동이 착점 도착 허용 e_hit(r) 이하면 하지 않는다 (한 번 더 움직여도 좋아지지 않는다 — 남는 σψ 는 여유에 전파). → (이동 [m], 사유)."""
        room = float(bb["cap_m"]) - float(moved_m)
        e_hit = self._hit_tol_m(r)
        if est is None:
            if outcome != "late":
                return 0.0, ""
            e, why = float(bb["b_m"]), f"the groove appeared late ({info.get('why')})"
        else:
            k = float(self.settings.judgement_k)
            b_need = _st.baseline_m(float(est.sigma_sec_m), k, self._yaw_tol_rad(r), self._theta_goal_rad())
            short = b_need - float(est.length_m)
            if not (short > 0.0):
                return 0.0, ""
            Y = (self._T_now() @ self.calib.T_flange_opt)[:3, 1]
            tY = abs(float(unit(t0) @ Y))
            e = short * abs(float(unit(est.u) @ Y)) / tY if tY > 0.0 else short
            why = (f"consistent sections over {est.length_m * 1e3:.1f} mm < the baseline the measured section noise needs "
                   f"{b_need * 1e3:.1f} mm (sigma {est.sigma_sec_m * 1e3:.3f} mm, {est.sigma_sec_source}; groove first on the line at "
                   f"frame {est.first_on_line})")
        e = min(e, room)
        if not (e > e_hit):
            if why:
                self._decide("yaw", f"S track: {why} - no extension (room {room * 1e3:.1f} mm within the geometric cap, move "
                                    f"{max(0.0, e) * 1e3:.1f} mm <= arrival tolerance at the hit {e_hit * 1e3:.2f} mm); the remaining "
                                    "direction uncertainty is carried in the margins")
            return 0.0, ""
        return e, why

    def _depth_scale_now(self, n) -> float:
        """레이저 평면 기울기 보정 √(1 − (n·Y_opt)²) (패스와 같은 추출 인자; 평행이동 동안 일정)."""
        Y = (self._T_now() @ self.calib.T_flange_opt)[:3, 1]
        c = min(1.0, abs(float(unit(n) @ Y)))
        return math.sqrt(max(0.0, 1.0 - c * c))

    def _x_c_m(self, loc: StaticLocalization) -> float:
        """드웰의 그루브 센서 x [m]. 읽지 못했으면 NaN (발자국 계산이 불확도를 전파한다 — 0 으로 바꾸지 않는다)."""
        x = float((loc.diag or {}).get("x_c_mm", float("nan")))
        return x * 1e-3 if _fin(x) else float("nan")

    def _s_hit(self, loc: StaticLocalization, est: SeamEstimate) -> float:
        """드웰의 심 진행 좌표 (est 기준): 그루브 상부 중심, 그루브가 없으면 레이저 평면 ∩ 추정 직선."""
        G = np.asarray(loc.top_centre_base, dtype=float)
        if loc.groove_found and np.all(np.isfinite(G)):
            return est.s_of(G)
        return float(laser_plane_line_param(loc.T_base_opt, est.c, est.u))

    def _scene_trk(self, est: Optional[SeamEstimate], outcome: str):
        """S_trk (추적 끝 드웰) 의 장면 판단 (4.7 절): 선 발자국 (지금 yaw |δ|) 이 추적이 일관되게 읽은 구간 안 (s_hit − f ≥ s_first), 또는
        'groove_mismatch' 증거 (직선 위 그루브 덮임 ≥ ½ — 시편 위) 면 'judge', 아니면 'record'."""
        if outcome == "groove_mismatch":
            return "judge"
        if est is None:
            return "record"

        def mode(loc: StaticLocalization) -> str:
            r = self._loc_range_m(loc, float(self.job_c.standoff_m))
            d = _st.yaw_to_seam_rad(np.asarray(loc.T_base_opt, dtype=float)[:3, :3], est.n, est.u)
            f = self._footprint_m(r, abs(d) if _fin(d) else 0.5 * math.pi, self._x_c_m(loc))
            return "judge" if self._s_hit(loc, est) - f >= float(est.s_first) else "record"
        return mode

    def _scene_site(self, est: SeamEstimate, psi_res: float):
        """S_al / S_corr / S 사이트: s_hit − f(ψ_res, x_c) ≥ s_first 면 'judge' (회전 뒤에는 늘 참)."""
        def mode(loc: StaticLocalization) -> str:
            r = self._loc_range_m(loc, float(self.job_c.standoff_m))
            return "judge" if self._s_hit(loc, est) - self._footprint_m(r, psi_res, self._x_c_m(loc)) >= float(est.s_first) else "record"
        return mode

    def _scene_E(self, est_T: Optional[SeamEstimate]):
        """E 계열: 그 단면을 트랜싯이 일관되게 읽었으면 (s_hit ≤ s_last_T) 'judge'. 트랜싯 맞춤이 없으면 대비책 경로 = 'judge' (예전)."""
        if est_T is None:
            return "judge"

        def mode(loc: StaticLocalization) -> str:
            return "judge" if self._s_hit(loc, est_T) <= float(est_T.s_last) else "record"
        return mode

    def _first_S(self) -> StaticLocalization:
        """S2 첫 드웰 (장면은 기록만 — 아직 선이 시편 위라는 측정이 없다): 보이는 면이 없으면 광축으로 다가가 다시 (_approach_blind), 판이
        없으면 같은 자리에서 한 번 더 재고 그래도 없으면 ABORTED. 그루브가 있든 없든 S 추적으로 간다 (측면 탐색 없음)."""
        loc = self._reacquire_if_oblique(self._dwell("S", scene="record"))
        if classify_localization(loc) == "no_surface":
            loc = self._approach_blind("S", loc, scene="record")
        if not loc.plate_ok:
            self._decide("search", f"S: plate not measured ({classify_localization(loc)}) - measuring again in place (confirm-first)")
            loc = self._dwell("S_again", scene="record")
            if not loc.plate_ok:
                raise _Abort(STATUS_ABORTED, f"S: plate not measured twice ({classify_localization(loc)}, flags={loc.flags})")
        return loc

    def _track_dir(self, loc: StaticLocalization) -> np.ndarray:
        """S 추적 방향 t̂0 = 대략 S→E 방향을 첫 드웰이 잰 판 평면에 투영 (판 거리 그대로), E0 쪽."""
        n = unit(np.asarray(loc.plate_normal_base, dtype=float))
        t0 = unit(self.u0 - n * float(self.u0 @ n))
        if float(t0 @ (np.asarray(self.job0.end, dtype=float) - np.asarray(self.job0.start, dtype=float))) < 0.0:
            t0 = -t0
        return t0

    def _track_leg(self, name: str, d_vec: np.ndarray, *, part: int, prior, t_hat0, r: float, n_meas) -> dict:
        """S_track / S_track2: 현재 실측 자세 → + d_vec (순수 평행이동, 'scan' 단계 — 가드·포락선), 센서 녹화 → 프레임 특징 (작업 프로세스,
        이동 중) → 심 맞춤. 원자료는 <name>/ (패스처럼 지연 저장). 녹화 구간이라 확인 종류 'scan' ('scan' 모드: 센서 녹화 구간을 묻는다 —
        CONFIRM_KINDS_SCAN; 'auto' 모드는 묻지 않는다)."""
        T_now = self._T_now()
        T_to = np.array(T_now, dtype=float, copy=True)
        T_to[:3, 3] += np.asarray(d_vec, dtype=float)
        ds = self._depth_scale_now(n_meas)
        _run, out, _d = self._streamed_leg(
            name, T_to, self.job_c, name, speed=float(self.job_c.speed_mps), kind="scan", recon_required=False, reconstruct=False,
            extra=(f"(S track {np.linalg.norm(d_vec) * 1e3:.0f} mm inward along the rough S->E direction in the measured plate plane, "
                   "sensor on: the seam direction is measured)"),
            track={"fit": self._track_fit_args(t_hat0, r, "S_track"), "depth_scale": ds, "part": part, "prior": prior})
        return out

    def _seek_S(self, loc0: StaticLocalization) -> tuple:
        """S3–S5 (+대비책): S 추적 → 추적 끝 드웰 S_trk (장면 판단은 4.7 절) → 맞춤 결과에 따라 ('late' → 같은 방향 b 더, 'none'·
        'groove_mismatch' → 측면 탐색 뒤 S 추적 한 번 더) → 그루브가 없으면 같은 자리 다시 / 마지막 일관 단면으로 S_inward → probe X (지금
        자세 기준; 뒤집히면 추적을 새 캘리브로 다시 맞춘다). → (SeamEstimate, 지금 자세 드웰 (probe X), 행 목록, 근거 dict)."""
        r = self._loc_range_m(loc0, float(self.job_c.standoff_m))
        t0 = self._track_dir(loc0)
        n_meas = unit(np.asarray(loc0.plate_normal_base, dtype=float))
        b, bb = self._track_length_m(r)
        rec = {"track_b_mm": b * 1e3, "track_basis": {(k_[:-2] + "_mm" if k_.endswith("_m") else k_): (v_ * 1e3 if k_.endswith("_m") else v_)
                                                      for k_, v_ in bb.items()},
               "t_hat0": t0.tolist()}
        self.res.yaw_align.update(rec)
        why_no = self._baseline_refusal(bb)
        if why_no:
            # 설정 단계가 계획 거리로 이미 거부한다 (이동 전); 여기는 첫 드웰이 잰 거리로 같은 규칙 (판단 정지)
            raise _Abort(STATUS_ABORTED, f"S: {why_no}")
        self._decide("yaw", f"S track: b = min(b_need {bb['b_star_m'] * 1e3:.1f} mm = sqrt2 x k x section noise {bb['sigma_sec0_m'] * 1e3:.3f} mm "
                            f"({bb['sigma_sec0_source']}) / (psi_tol {bb['psi_tol_deg']:.2f} - theta_goal {bb['theta_goal_deg']:.2f} deg), "
                            f"cap L_r - L_min - m_E {bb['cap_m'] * 1e3:.1f} mm) = {b * 1e3:.1f} mm inward along {np.round(t0, 4).tolist()} "
                            f"(rough S->E in the measured plate plane), range {r * 1e3:.0f} mm; extended once if the measured track needs "
                            "a longer baseline", track=_jsonable(rec))
        fit = self._track_fit_args(t0, r, "S_track")
        tr = self._track_leg("S_track", t0 * b, part=0, prior=None, t_hat0=t0, r=r, n_meas=n_meas)
        rows = [tr.get("rows")]
        est, outcome, info = tr.get("est"), tr.get("outcome"), tr.get("info") or {}
        self._decide("yaw", self._track_text("S_track", est, outcome, info), track=_jsonable({"outcome": outcome, "info": info,
                                                                                             "est": est.to_dict() if est else None}))
        name_t = "S_trk"
        e_ext, why_ext = self._track_extension(est, outcome, info, bb, b, r, t0)
        # 연장이 남았으면 이 추정은 아직 기선이 모자란다 — 그 드웰의 장면은 기록만 (연장 뒤 드웰이 판단; 확인 먼저)
        loc_t = self._dwell(name_t, scene="record" if e_ext > 0.0 else self._scene_trk(est, outcome))
        if e_ext > 0.0:
            self._decide("yaw", f"S track: {why_ext} - S_track2: {e_ext * 1e3:.1f} mm further inward once (geometric cap "
                                f"{bb['cap_m'] * 1e3:.1f} mm, moved {b * 1e3:.1f} mm)")
            tr2 = self._track_leg("S_track2", t0 * e_ext, part=1, prior=tr.get("rows"), t_hat0=t0, r=r, n_meas=n_meas)
            rows.append(tr2.get("rows"))
            est, outcome, info = tr2.get("est"), tr2.get("outcome"), tr2.get("info") or {}
            self._decide("yaw", self._track_text("S_track+S_track2", est, outcome, info),
                         track=_jsonable({"outcome": outcome, "info": info, "est": est.to_dict() if est else None}))
            if est is None:
                raise _Abort(STATUS_ABORTED, f"S: {self._groove_seen_text(rows, b + e_ext)} (S track + S track 2; {outcome}: "
                                             f"{info.get('why')}) - the rough S point is off the groove or the groove does not match the "
                                             "design here")
            name_t = "S_trk2"
            loc_t = self._dwell(name_t, scene=self._scene_trk(est, outcome))
        elif outcome == "late":
            raise _Abort(STATUS_ABORTED, f"S: {self._groove_seen_text(rows, b)} (S track; {info.get('why')}) and no room to extend the S "
                                         f"track within the geometric cap {bb['cap_m'] * 1e3:.1f} mm - teach the rough S point on the groove "
                                         "a little inside the end")
        elif est is None:
            # 'none' (설계 일치 단면 없음 / 흩어짐), 'groove_mismatch' (S_trk 드웰이 판단했고 통과 — 설계와 맞는 그루브는 다른 곳), 계산 실패
            b2 = min(b, float(bb["cap_m"]) - b)                  # 두 추적의 합도 기하 상한 안 (S 사이트가 E 에서 최소 거리 + 여유)
            if not (b2 > self._hit_tol_m(r)):
                raise _Abort(STATUS_ABORTED, f"S track: {outcome} ({info.get('why')}) and no room for a second S track after a lateral "
                                             f"search (geometric cap {bb['cap_m'] * 1e3:.1f} mm, moved {b * 1e3:.1f} mm) - teach the rough "
                                             "S point on the groove")
            self._decide("yaw", f"S track: {outcome} ({info.get('why')}) - lateral search from {name_t} (fallback), then the S track once "
                                f"more ({b2 * 1e3:.1f} mm)")
            loc_f = self._find_groove_at("S", loc_t)
            tr2 = self._track_leg("S_track2", t0 * b2, part=1, prior=None, t_hat0=t0, r=r, n_meas=n_meas)
            rows = [tr2.get("rows")]
            est, outcome, info = tr2.get("est"), tr2.get("outcome"), tr2.get("info") or {}
            self._decide("yaw", self._track_text("S_track2", est, outcome, info),
                         track=_jsonable({"outcome": outcome, "info": info, "est": est.to_dict() if est else None,
                                          "found_at": loc_f.top_centre_base}))
            if est is None:
                raise _Abort(STATUS_ABORTED, f"S: the groove found by the lateral search does not continue along the S track ({outcome}: "
                                             f"{info.get('why')})")
            name_t = "S_trk2"
            loc_t = self._dwell(name_t, scene=self._scene_trk(est, outcome))
        scene_t = self._scene_trk(est, outcome)
        if classify_localization(loc_t) != "groove":
            # 추적은 그루브를 봤는데 추적 끝 드웰이 못 읽는다: 같은 자리 다시 → 추적 직선 위 마지막 일관 단면으로 (S_inward, −u) → 그래도 없으면 멈춤
            self._decide("yaw", f"{name_t}: no groove in the dwell although the S track saw it - measuring again in place (confirm-first)")
            loc_t = self._dwell(f"{name_t}_again", scene=scene_t)
            if classify_localization(loc_t) != "groove":
                s_now = float(laser_plane_line_param(loc_t.T_base_opt, est.c, est.u))
                back = s_now - float(est.s_last)
                if _fin(back) and back > 0.0:
                    T_to = self._T_now()
                    T_to[:3, 3] -= unit(est.u) * back
                    self._local_move(T_to, "scan", "S_inward", kind="local",
                                     extra=f"(back {back * 1e3:.1f} mm along the measured seam to the last consistent section of the S track)")
                    loc_t = self._dwell("S_inward", scene=scene_t)
                if classify_localization(loc_t) != "groove":
                    raise _Abort(STATUS_ABORTED, f"{name_t}: the S track saw the groove but the dwell does not read it (twice, and at the last "
                                                 "consistent section) - check the scene at S")
        flipped0 = bool(self.res.optical_branch_flipped)
        loc_p = self._probe(loc_t, axes=("X",), scene=scene_t)
        if self.res.optical_branch_flipped and not flipped0:
            # 광학 분기를 뒤집었다: 추적 특징은 센서 프레임 그대로 — 기반 좌표 변환만 새 캘리브로 다시
            meta: dict = {}
            est, outcome, info = _post_scan.run_refit([x for x in rows if x is not None], self.calib, fit,
                                                     use_worker=bool(self.at.post_scan_worker), log=self._p, meta=meta,
                                                     cancel=self.cancel)
            if meta.get("worker_error"):
                self.res.notes.append(f"S track re-fit: worker {meta.get('worker_state')} ({meta['worker_error']}) - computed in the "
                                      "server process")
            self._decide("timing", f"S track re-fit [{meta.get('where')}]", timing=_jsonable(meta))
            self._decide("yaw", self._track_text("S track (re-fitted with the flipped optical branch)", est, outcome, info),
                         track=_jsonable({"outcome": outcome, "info": info, "est": est.to_dict() if est else None}))
            if est is None:
                raise _Abort(STATUS_ABORTED, f"S track re-fitted with the flipped optical branch: {outcome} ({info.get('why')})")
        self.res.yaw_align.update({"S_track": est.to_dict(), "S_track_outcome": outcome})
        return est, loc_p, rows, rec

    @staticmethod
    def _groove_seen_text(rows: list, move_m: float) -> str:
        """맞춤 실패 문구: 이동 (move_m) 중 설계 일치 그루브가 처음 나타난 곳부터 끝까지 그 일치 프레임 비율 (그루브가 나타나기 전 프레임은
        세지 않는다)."""
        rr = [x for x in rows if x is not None and x.n > 0]
        mv = np.concatenate([x.moving for x in rr]) if rr else np.zeros(0, dtype=bool)
        dz = np.concatenate([x.design for x in rr]) if rr else np.zeros(0, dtype=bool)
        on = mv & dz
        if not on.any():
            return f"no design-consistent groove over the {move_m * 1e3:.0f} mm move"
        i0 = int(np.flatnonzero(on)[0])
        n_after = int(mv[i0:].sum())
        frac0 = float(mv[:i0].sum()) / float(max(1, mv.sum()))
        return (f"groove first consistent at {frac0 * move_m * 1e3:.0f} mm of the {move_m * 1e3:.0f} mm move, then on "
                f"{100.0 * float(on[i0:].sum()) / float(max(1, n_after)):.0f} % of the moving frames")

    @staticmethod
    def _track_text(name: str, est: Optional[SeamEstimate], outcome: str, info: dict) -> str:
        if est is None:
            return f"{name}: no seam line ({outcome}: {info.get('why')}; frames {info.get('n_frames')}, moving {info.get('n_moving')}, " \
                   f"design-consistent {info.get('n_design')}, groove {info.get('n_groove')})"
        return (f"{name}: seam line from the {est.track} track ({est.n_inliers}/{est.n_frames} frames on the line, coverage "
                f"{est.coverage * 100:.0f} %, residual {est.mad_m * 1e3:.3f} mm) u {np.round(est.u, 4).tolist()} n {np.round(est.n, 4).tolist()}, "
                f"consistent sections {est.s_first * 1e3:+.1f}..{est.s_last * 1e3:+.1f} mm (baseline {est.length_m * 1e3:.1f} mm, sigma_psi "
                f"model {math.degrees(est.sigma_psi_model):.2f} deg / self {math.degrees(est.sigma_psi_self):.2f} deg recorded), "
                f"tracks {est.tracks}")

    def _rotation_precheck(self, T_now: np.ndarray, T_align: np.ndarray, T_end: np.ndarray) -> tuple:
        """회전 사슬 사전검사 (6 절): 포락선 표본 (S_align 회전 이동 + 그 자세의 트랜싯, 'scan' 규칙 — 3G 바닥·테이블·몸체 포함) + 플랫폼
        추종 (지금 관절에서 그 사슬을 조밀 IK 로 따라가는가 — follow_from, IK 다시 풀지 않음; 없으면 reach_candidates 의 지금 형태 = dist_rad
        최소; 둘 다 없으면 포락선만). → (ok, 사유). 거부된 후보는 보내지 않는다 (판단만)."""
        why = self._legs_pass_envelope(T_now, [(T_align, "scan", "S_align", ""), (T_end, "scan", "transit_S_to_E", "")], why=True)
        if why:
            return False, f"envelope: {why}"
        q_now = self._q_now()
        ff = getattr(self.rs, "follow_from", None)
        rc = getattr(self.rs, "reach_candidates", None)
        try:
            if callable(ff) and q_now is not None:
                c = ff(q_now, [T_align, T_end], sigma_radius_m=self._lever_m())
            elif callable(rc):
                withd = [c_ for c_ in rc(T_now, follow=[T_align, T_end], q_now=q_now, sigma_radius_m=self._lever_m())
                         if c_.get("dist_rad") is not None]
                c = min(withd, key=lambda c_: float(c_["dist_rad"])) if withd else None
            else:
                c = None
        except Exception as e:  # noqa: BLE001 - a failed check is a refusal of the candidate, never a motion
            return False, f"platform follow check failed ({type(e).__name__}: {e})"
        if c is not None and not c.get("ok"):
            return False, f"platform: the current arm configuration does not follow the rotation and the transit ({c.get('reason')})"
        return True, ""

    def _transit_target(self, est: SeamEstimate, A, T_from: np.ndarray, psi_rem: float, r: float) -> tuple:
        """트랜싯 목표 (4.5 절): E0 를 S 직선에 사영 (s(E0'), A = S 그루브 점 기준), 설계 길이 상한 s_cap = s_first + L_d + tol_L (S 끝 ≤ s_first),
        E 안쪽 여유 m_E = f(ψ_res, x_pred) → s_target = min(s(E0'), s_cap) − m_E, 자세 그대로 d = u·s_target (u 에 심 기울기 — 판 거리 보존).
        → (T_to, 근거 dict)."""
        u = unit(np.asarray(est.u, dtype=float))
        A = np.asarray(A, dtype=float)
        E0 = np.asarray(self.job0.end, dtype=float)
        s_E0p = float((E0 - A) @ u)
        s_first_A = float(est.s_first) + float((np.asarray(est.c, dtype=float) - A) @ u)
        L_d = float(self.spec.weld_length_mm) * 1e-3 if (self.spec is not None and self.spec.weld_length_mm > 0) else None
        tol_L = self._qc_tol_m("weld_length_mm") or 0.0
        s_cap = (s_first_A + L_d + tol_L) if L_d is not None else math.inf
        x_pred = _st.predict_x_at(np.asarray(T_from, dtype=float) @ self.calib.T_flange_opt, est)
        psi_res = self._psi_res_rad(est, psi_rem)
        m_E = self._footprint_m(r, psi_res, x_pred * 1e-3 if _fin(x_pred) else float("nan"))
        s_target = min(s_E0p, s_cap) - m_E
        T_to = np.array(T_from, dtype=float, copy=True)
        T_to[:3, 3] += u * s_target
        return T_to, {"s_E0p_mm": s_E0p * 1e3, "s_cap_mm": s_cap * 1e3 if _fin(s_cap) else None, "cap_applied": bool(s_cap < s_E0p),
                      "m_E_mm": m_E * 1e3, "s_target_mm": s_target * 1e3, "x_pred_mm": x_pred, "psi_res_deg": math.degrees(psi_res),
                      "s_first_from_S_mm": s_first_A * 1e3, "L_d_mm": L_d * 1e3 if L_d is not None else None, "tol_L_mm": tol_L * 1e3}

    def _align_S(self, est: SeamEstimate, loc: StaticLocalization) -> tuple:
        """S6–S7: 회전 판단 (|δ| > ψ_tol 일 때만 — 한 번 더 움직여도 도착 허용보다 좋아지지 않으면 움직이지 않는다) → 사전검사 (거부되면 frac 을
        ⌈log2(|δ|/ψ_tol)⌉ 번 이분 탐색, 가장 큰 허용값; 거부 후보는 보내지 않는다) → S_align (판 법선 둘레 그루브 점 피벗 회전 + 수직축 측면·
        거리·롤, 한 이동) → S_al 드웰. → (지금 자세 국소화, 근거 dict {delta, frac, psi_rem, rotated})."""
        r = self._loc_range_m(loc, float(self.job_c.standoff_m))
        T_now = self._T_now()
        delta = _st.yaw_to_seam_rad((T_now @ self.calib.T_flange_opt)[:3, :3], est.n, est.u)
        psi_tol = self._yaw_tol_rad(r)
        target = self._coarse_plate_m()
        rec = {"delta_deg": math.degrees(delta), "psi_tol_deg": math.degrees(psi_tol), "range_m": r, "rotated": False, "frac": 0.0,
               "precheck": []}
        if not (_fin(delta) and abs(delta) > psi_tol):
            rec.update(psi_rem_deg=math.degrees(delta) if _fin(delta) else 0.0)
            rec["psi_res_deg"] = math.degrees(self._psi_res_rad(est, math.radians(rec["psi_rem_deg"])))
            self._decide("yaw", f"S: laser line {math.degrees(delta):+.2f} deg from perpendicular to the measured seam <= psi_tol "
                                f"{math.degrees(psi_tol):.2f} deg (FOV/2 x sin psi within the arrival tolerance) - no rotation; corrections "
                                "along the measured seam normal only", align=_jsonable(rec))
            return loc, rec
        A = np.asarray(loc.top_centre_base, dtype=float)

        def candidate(frac: float) -> tuple:
            T_al, ai = seam_align_target(T_now, self.calib, loc, est, target, frac)
            T_end, _ti = self._transit_target(est, A, T_al, (1.0 - frac) * delta, r)
            ok, why = self._rotation_precheck(T_now, T_al, T_end)
            rec["precheck"].append({"frac": frac, "ok": ok, "why": why})
            return ok, why, T_al, ai
        ok, why, T_al, ai = candidate(1.0)
        frac = 1.0
        if not ok:
            n_bis = max(1, int(math.ceil(math.log2(abs(delta) / psi_tol))))
            lo, hi, best = 0.0, 1.0, None
            self._decide("yaw", f"S: full rotation {math.degrees(delta):+.2f} deg refused by the pre-check ({why}) - bisecting the "
                                f"rotation fraction ({n_bis} steps, angle resolution psi_tol)")
            for _ in range(n_bis):
                mid = 0.5 * (lo + hi)
                ok_m, why_m, T_m, ai_m = candidate(mid)
                if ok_m:
                    lo, best = mid, (T_m, ai_m)
                else:
                    hi = mid
            frac = lo
            if best is not None:
                T_al, ai = best
        psi_rem = (1.0 - frac) * delta
        rec.update(frac=frac, psi_rem_deg=math.degrees(psi_rem), rotated=frac > 0.0)
        rec["psi_res_deg"] = math.degrees(self._psi_res_rad(est, psi_rem))
        if frac <= 0.0:
            self._decide("yaw", f"S: no rotation passes the pre-check (last: {why}) - corrections along the measured seam normal only; the "
                                f"remaining yaw {math.degrees(psi_rem):+.2f} deg is added to the footprint margins", align=_jsonable(rec))
            return loc, rec
        T_now2 = self._T_now()
        d = np.asarray(T_al, dtype=float)[:3, 3] - T_now2[:3, 3]
        n = unit(est.n)
        rot = _rot_angle_deg(T_now2[:3, :3], np.asarray(T_al)[:3, :3])
        phase = self._leg_phase(float(d @ n), float(np.linalg.norm(d - n * float(d @ n))), rot, orientation=self.orientation,
                                tol_m=self._sv().pose_tol_m)
        self._decide("yaw", f"S_align: yaw {math.degrees(delta):+.2f} deg x {frac:.3f} about the measured plate normal through the groove "
                            f"point {np.round(A, 4).tolist()} + {ai['correction']['reason']} (remaining yaw {math.degrees(psi_rem):+.2f} deg)",
                     align=_jsonable({**rec, "target": ai}))
        self._local_move(T_al, phase, "S_align", kind="local",
                         extra=(f"(yaw {math.degrees(delta) * frac:+.1f} deg, pivot = groove at S_trk, axis = measured plate normal; "
                                "+ lateral/standoff/roll along the measured seam normal)"))
        loc_al = self._dwell("S_al", scene=self._scene_site(est, self._psi_res_rad(est, psi_rem)))
        return loc_al, rec

    def _finish_S(self, est: SeamEstimate, loc: StaticLocalization, align: dict) -> tuple:
        """S8: 측정 심에 수직인 축 w 로만 보정 (지금 규칙: 확인 먼저·발산 판정) → 부분 회전이면 S 사이트 발자국이 일관 단면 안이 되게 안쪽으로
        (S_inward) → 지점 기록. → (site, loc)."""
        target = self._coarse_plate_m()
        psi_rem = math.radians(float(align.get("psi_rem_deg", 0.0)))
        psi_res = self._psi_res_rad(est, psi_rem)
        scene = self._scene_site(est, psi_res)
        loc_S, _ = self._correct_until_converged("S", loc, target, seam=est, scene=scene)
        if abs(psi_rem) > 0.0:
            r = self._loc_range_m(loc_S, float(self.job_c.standoff_m))
            f_S = self._footprint_m(r, psi_res, self._x_c_m(loc_S))
            room = self._s_hit(loc_S, est) - float(est.s_first)
            if f_S > room:
                step = f_S - room
                T_to = self._T_now()
                T_to[:3, 3] += unit(est.u) * step
                self._decide("yaw", f"S: footprint {f_S * 1e3:.1f} mm (remaining yaw {math.degrees(psi_rem):+.2f} deg) > {room * 1e3:.1f} mm "
                                    f"to the first consistent section - S_inward {step * 1e3:.1f} mm along the measured seam")
                self._local_move(T_to, "scan", "S_inward", kind="local", extra=f"(+{step * 1e3:.1f} mm along the measured seam)")
                loc_S = self._dwell("S_inward", scene=scene)
                loc_S, _ = self._correct_until_converged("S_in", loc_S, target, seam=est, scene=scene)
        r = self._loc_range_m(loc_S, float(self.job_c.standoff_m))
        yaw = _st.yaw_to_seam_rad(np.asarray(loc_S.T_base_opt, dtype=float)[:3, :3], est.n, est.u)
        extra = {"tool_yaw_vs_seam_deg": math.degrees(yaw), "seam_u": np.asarray(est.u, dtype=float), "psi_res_deg": math.degrees(psi_res),
                 "footprint_mm": self._footprint_m(r, psi_res, self._x_c_m(loc_S)) * 1e3, "range_m": r}
        return self._site_record("S", loc_S, target, extra=extra, scene=scene)

    def _transit_to_E(self, site_S: dict, loc_S: StaticLocalization, est: SeamEstimate, align: dict) -> dict:
        """T (4.5 절): 보정된 S 자세 그대로 측정 심 u_S 를 따라 s_target 까지 한 번의 스트리밍 이동 ('scan' — 확인 종류 'scan', 포락선·가드),
        이동 중 프레임 특징 → 트랜싯 맞춤 (E 배치·E 쪽 심 방향). → 추적 결과 dict (+ 'target')."""
        r = self._loc_range_m(loc_S, float(self.job_c.standoff_m))
        T_now = self._T_now()
        A = np.asarray(site_S["top"], dtype=float)
        T_to, ti = self._transit_target(est, A, T_now, math.radians(float(align.get("psi_rem_deg", 0.0))), r)
        d = T_to[:3, 3] - T_now[:3, 3]
        self._decide("transit", f"S -> E along the measured seam u_S {np.round(est.u, 4).tolist()} (slope included - the plate distance is "
                                f"kept): s(E0') {ti['s_E0p_mm']:.1f} mm, design cap s_first + L_d + tol_L "
                                f"{_fmtf(ti['s_cap_mm'], '.1f')} mm{' (applied)' if ti['cap_applied'] else ''}, E margin m_E {ti['m_E_mm']:.1f} mm "
                                f"(psi_res {ti['psi_res_deg']:.2f} deg, x_pred {_fmtf(ti['x_pred_mm'], '+.1f')} mm) -> s_target "
                                f"{ti['s_target_mm']:.1f} mm; translation {np.round(d * 1e3, 1).tolist()} mm with the tool orientation kept",
                     transit=_jsonable(ti))
        self.res.yaw_align["transit"] = _jsonable(ti)
        self._event("transit_started", name="transit_S_to_E")
        _run, tr, _dir = self._streamed_leg(
            "transit_S_to_E", T_to, self.job_c, "transit_S_to_E", speed=float(self.job_c.speed_mps), recon_required=False,
            reconstruct=False, extra=f"(move to E along the measured seam, {np.linalg.norm(d) * 1e3:.0f} mm, sensor on: groove track)",
            track={"fit": self._track_fit_args(est.u, r, "transit"), "depth_scale": self._depth_scale_now(est.n), "part": 0, "prior": None})
        tr = dict(tr or {})
        tr["target"] = ti
        est_T, outcome, info = tr.get("est"), tr.get("outcome"), tr.get("info") or {}
        self._decide("transit", self._track_text("transit", est_T, outcome, info),     # 피더 보폭 (부하) 은 'timing' 결정에만
                     track=_jsonable({"outcome": outcome, "info": info, "est": est_T.to_dict() if est_T else None}))
        return tr

    def _place_E(self, tr: dict, est_S: SeamEstimate, align: dict, r: float) -> tuple:
        """E1–E2 (4.6 절): 트랜싯 맞춤이 있으면 E 드웰 = min(s_도착, s_last_T − f(ψ_res, |x_pred| + u_x)) — 도착 단면이 일관이든 아니든 레이저
        선 발자국 전체 (그루브에서 먼 쪽 팔 + 도착 허용) 가 트랜싯이 일관되게 읽은 시편 (s ≤ s_last_T) 안에 들게 (u_x = 예측 x 의 불확도 e_hit +
        k·√2·q — 드웰 x_c 대 예측 비교와 같은 허용). 도착이 그보다 앞이면 E_inward (−u_T, 같은 자세) → 드웰 E (x_c 를 예측과 비교해 기록,
        판정 아님). 그 드웰이 그루브를 못 읽으면
        같은 자리 다시 → 2·f(ψ_res) 안쪽으로 다시 → ABORTED. 맞춤이 없으면 (작업 프로세스 오류·덮임 부족) 도착 자세 드웰, 그루브가 없으면 예전
        측면 탐색. E 에서는 회전하지 않는다 (S·E 같은 자세 → L1 방향이 yaw 의존 오프셋·형태 차에서 독립). → (도착 국소화, 보정 전 국소화, 그
        이름, est_T, 기록)."""
        est_T = tr.get("est") if isinstance(tr, dict) else None
        rec = {"outcome": (tr or {}).get("outcome"), "why": ((tr or {}).get("info") or {}).get("why")}
        if est_T is None:
            self._decide("place_E", f"transit track not available ({rec['outcome']}: {rec['why']}) - E dwell at the arrival pose "
                                    "(fallback: lateral search if no groove)", place=_jsonable(rec))
            loc_E_arr = self._dwell("E")
            loc_E = loc_E_arr if classify_localization(loc_E_arr) == "groove" else self._find_groove_at("E_search")
            pre_name = self.res.localizations[-1]["name"] if self.res.localizations else "E"
            self.res.yaw_align["place_E"] = _jsonable(rec)
            return loc_E_arr, loc_E, pre_name, None, rec
        e_hit = self._hit_tol_m(r)
        psi_res = self._psi_res_rad(est_S, math.radians(float(align.get("psi_rem_deg", 0.0))))
        u_T = unit(np.asarray(est_T.u, dtype=float))
        T_arr = self._T_now()
        T_bo = T_arr @ self.calib.T_flange_opt
        s_arr = float(laser_plane_line_param(T_bo, est_T.c, u_T))
        k = float(self.settings.judgement_k)
        q = _ms.quant_sigma_mm(r * 1e3) * 1e-3
        u_x = e_hit + k * math.sqrt(2.0) * q                         # 예측 x 의 불확도 (드웰 x_c 대 예측 비교 허용과 같다)
        x_arr = _st.predict_x_at(T_bo, est_T)
        f_E = self._footprint_m(r, psi_res, abs(x_arr) * 1e-3 + u_x if _fin(x_arr) else float("nan"))
        s_E = min(s_arr, float(est_T.s_last) - f_E)
        back = s_arr - s_E
        rec.update(arrival_consistent=est_T.arrival_consistent, s_last_T_mm=float(est_T.s_last) * 1e3, s_arrival_mm=s_arr * 1e3,
                   footprint_E_mm=f_E * 1e3, x_pred_arrival_mm=x_arr, x_unc_mm=u_x * 1e3, psi_res_deg=math.degrees(psi_res),
                   s_E_mm=s_E * 1e3, E_inward_mm=0.0)
        if _fin(back) and back > 0.0:
            T_to = np.array(T_arr, dtype=float, copy=True)
            T_to[:3, 3] -= u_T * back
            self._decide("place_E", f"E arrival section {'consistent' if est_T.arrival_consistent else 'NOT consistent'} in the transit "
                                    f"stream; the line footprint {f_E * 1e3:.1f} mm (psi_res {math.degrees(psi_res):.2f} deg, groove x "
                                    f"{_fmtf(x_arr, '+.1f')} +- {u_x * 1e3:.2f} mm, arrival tolerance) must end inside the last consistent "
                                    f"section s_last_T {float(est_T.s_last) * 1e3:+.1f} mm: E_inward {back * 1e3:.1f} mm along -u_T to "
                                    f"{s_E * 1e3:+.1f} mm")
            self._local_move(T_to, "scan", "E_inward", kind="local",
                             extra=f"(back {back * 1e3:.1f} mm along the transit seam: the line footprint inside its last consistent section)")
            rec["E_inward_mm"] = back * 1e3
        x_pred = _st.predict_x_at(self._T_now() @ self.calib.T_flange_opt, est_T)
        scene = self._scene_E(est_T)
        loc_E_arr = self._dwell("E", scene=scene)
        loc = loc_E_arr
        if classify_localization(loc) != "groove":
            self._decide("place_E", "E: the transit saw the groove here but the dwell does not read it - measuring again in place "
                                    "(confirm-first)")
            loc = self._dwell("E_again", scene=scene)
            if classify_localization(loc) != "groove":
                step = 2.0 * self._footprint_m(r, psi_res, float("nan"))      # 그루브를 못 읽었다: x 불확도 전파 (시야 어디든)
                T_to = self._T_now()
                T_to[:3, 3] -= u_T * step
                self._local_move(T_to, "scan", "E_inward2", kind="local", extra=f"(back {step * 1e3:.1f} mm = 2 x footprint along -u_T)")
                loc = self._dwell("E_inward2", scene=scene)
                if classify_localization(loc) != "groove":
                    raise _Abort(STATUS_ABORTED, "E: transit saw the groove here; the dwell does not read it (twice in place and "
                                                 f"{step * 1e3:.1f} mm inward) - check the scene at E")
        tol_x = u_x * 1e3
        x_c = float(loc.diag.get("x_c_mm", float("nan")))
        rec.update(x_pred_mm=x_pred, x_c_mm=x_c, x_tol_mm=tol_x, x_within=bool(_fin(x_c) and _fin(x_pred) and abs(x_c - x_pred) <= tol_x))
        self._decide("place_E", f"E: arrival {'consistent' if est_T.arrival_consistent else 'NOT consistent'} in the transit stream, "
                                f"s_last_T {float(est_T.s_last) * 1e3:+.1f} mm, E_inward {rec['E_inward_mm']:.1f} mm; groove x_c "
                                f"{_fmtf(x_c, '+.2f')} mm vs predicted {_fmtf(x_pred, '+.2f')} mm (tolerance {tol_x:.2f} mm: "
                                f"{'within' if rec['x_within'] else 'OUTSIDE - recorded, not judged'})", place=_jsonable(rec))
        self.res.yaw_align["place_E"] = _jsonable(rec)
        pre_name = self.res.localizations[-1]["name"] if self.res.localizations else "E"
        return loc_E_arr, loc, pre_name, est_T, rec

    # ---- 5. 직선 L1 ------------------------------------------------------------------------------------
    def _line_from_sites(self, site_S: dict, site_E: dict, loc_S: StaticLocalization, loc_E_arr: StaticLocalization,
                         loc_E_pre: StaticLocalization, loc_E: StaticLocalization, pre_name: str, yaw: Optional[dict] = None) -> SeamLine:
        """두 측정 루트 중심 → L1: u1 = unit(P_E − P_S) 를 판 법선 n1 (Q_S ∪ Q_E 평면 적합, 센서 쪽) 에 재직교화. 잔차 진단 (theta
        기울기 인식, roll, psi 두 드웰/운동학 — 기록만) + 센서 x 스케일 판정 (_scale_check, 확인 먼저 — 4d P4)."""
        P_S, P_E = np.asarray(site_S["root"], dtype=float), np.asarray(site_E["root"], dtype=float)
        L = float(np.linalg.norm(P_E - P_S))
        L_min, lb = self._min_site_distance_m()
        if L < L_min:
            raise _Abort(STATUS_ABORTED, f"measured S and E groove centres are only {L * 1e3:.0f} mm apart (< {L_min * 1e3:.0f} mm: two "
                                         f"arrival errors at the hit {lb['e_hit_m'] * 1e3:.2f} mm over the design length must stay inside "
                                         f"the pass view room {lb['room_m'] * 1e3:.1f} mm): check the rough points")
        Q = np.concatenate([site_S["Q"], site_E["Q"]])
        n1 = np.full(3, np.nan)
        if min(np.asarray(site_S["Q"]).shape[0], np.asarray(site_E["Q"]).shape[0]) >= 2:   # 기하 최소: 자리마다 판 점 2 개 (직선 둘 → 평면)
            # 두 로봇 자세 (S·E 드웰) 에서 잰 판 점은 프로파일 하나씩 = 3D 직선 두 개: 최소 중앙값 가설은 점이 많은 쪽 직선만 담은
            # 평면 (그 직선 둘레로 아무 방향) 을 고를 수 있다 (가상 팔 3G 기운 판: 법선 125° 틀어짐). 두 자세의 측정은 도착 허용 e 안에서
            # 맞는다 (도착 허용을 레이저 착점으로 옮긴 e_hit, 두 드웰 중 먼 거리) — 그 거리 안 점이 가장 많은 평면 (두 직선을 함께 담는 평면)
            r_hit = max(self._loc_range_m(loc_S, self._coarse_plate_m()), self._loc_range_m(loc_E, self._coarse_plate_m()))
            pl = ransac_plane(Q, inlier_mm=self._hit_tol_m(r_hit) * 1e3, rng=np.random.default_rng(0))
            n1 = np.asarray(pl["normal"], dtype=float)
        if not np.all(np.isfinite(n1)):
            n1 = unit(np.asarray(site_S["plate_normal"]) + np.asarray(site_E["plate_normal"]))
        if float(n1 @ (np.asarray(site_S["origin"]) - P_S)) < 0.0:
            n1 = -n1
        u1 = unit(P_E - P_S)
        u1 = unit(u1 - n1 * float(u1 @ n1))
        n1 = unit(n1 - u1 * float(n1 @ u1))
        line = SeamLine(c=P_S.copy(), u=u1, n=n1, s_S=0.0, s_E=float((P_E - P_S) @ u1), k=0)
        g_axis = self.job0.gravity_axis_base
        pair = seam_from_pair(loc_S, loc_E, gravity_axis_base=g_axis)          # C10: 3G 는 standoff 차
        yaw = dict(yaw or {})
        est_S, est_T = yaw.get("est_S"), yaw.get("est_T")
        w_E = (est_T.w if est_T is not None else est_S.w) if est_S is not None else None
        lat_pre, lat_pre_kin = site_pre_laterals(loc_E_arr, loc_E_pre, loc_E, w=w_E)
        lat_S = float(loc_S.lateral_offset_m)

        def psi_of(lat_to):
            return math.degrees(math.atan2(lat_to - lat_S, L)) if (_fin(lat_to) and _fin(lat_S) and L > 0.0) else float("nan")
        cons = {"psi_indep_deg": psi_of(lat_pre), "psi_kin_deg": psi_of(lat_pre_kin), "lat_S_final_m": lat_S, "lat_E_pre_m": lat_pre,
                "lat_E_pre_kin_m": lat_pre_kin, "E_pre_name": pre_name}
        scale = self._scale_check(lat_pre, lat_pre_kin, loc_E_pre)
        roll = max(abs(loc_S.roll_signed_deg), abs(loc_E.roll_signed_deg))
        rec = {"k": 0, "stage": "line_L1", "status": "-", "theta_deg": pair.get("theta_deg"),
               "theta_source": "plate heights (horizontal travel)" if pair.get("travel_horizontal") else "standoff difference",
               "psi_pair_deg": pair.get("psi_deg"), "roll_deg": roll, "psi_indep_deg": cons["psi_indep_deg"],
               "psi_kin_deg": cons["psi_kin_deg"], "psi_consistency": cons, "sensor_scale": scale,
               "lateral_m": max(abs(loc_S.lateral_offset_m), abs(loc_E.lateral_offset_m)), "L_m": L,
               "u": u1, "n": n1, "n_vs_n0_deg": _angle_deg(n1, self.n0), "u_vs_u0_deg": _angle_deg(u1, self.u0)}
        if est_S is not None:
            # P2 기록 (8 절): 측정 yaw, 실행 못한 회전, 추적 방향들 사이 각, 툴 레이저 선 vs L1 (S·E), 같은 자세 확인
            al = dict(yaw.get("align") or {})

            def tool_yaw(site):
                R = site.get("R_opt")
                return math.degrees(_st.yaw_to_seam_rad(np.asarray(R, dtype=float), n1, u1)) if R is not None else float("nan")
            yS, yE = tool_yaw(site_S), tool_yaw(site_E)
            rec.update(yaw_track_deg=al.get("delta_deg"), psi_rem_deg=al.get("psi_rem_deg"), yaw_frac=al.get("frac"),
                       u_S_vs_u_T_deg=_angle_deg(est_S.u, est_T.u) if est_T is not None else float("nan"),
                       u1_vs_u_T_deg=_angle_deg(u1, est_T.u) if est_T is not None else float("nan"),
                       u1_vs_u_S_deg=_angle_deg(u1, est_S.u), tool_yaw_vs_L1_S_deg=yS, tool_yaw_vs_L1_E_deg=yE,
                       tool_yaw_S_minus_E_deg=yS - yE, track=est_S.track, coverage=est_S.coverage,
                       track_T=est_T.track if est_T is not None else None, coverage_T=est_T.coverage if est_T is not None else None)
            self.res.yaw_align["L1"] = _jsonable({k_: rec[k_] for k_ in ("yaw_track_deg", "psi_rem_deg", "u_S_vs_u_T_deg", "u1_vs_u_T_deg",
                                                                         "u1_vs_u_S_deg", "tool_yaw_vs_L1_S_deg", "tool_yaw_vs_L1_E_deg",
                                                                         "tool_yaw_S_minus_E_deg", "track", "coverage")})
            self._decide("yaw", f"L1: tool laser line vs L1 S {yS:+.2f} / E {yE:+.2f} deg (S - E {yS - yE:+.2f}), u_S vs u_T "
                                f"{_fmtf(rec['u_S_vs_u_T_deg'], '.3f')} deg, u1 vs u_T {_fmtf(rec['u1_vs_u_T_deg'], '.3f')} deg, yaw measured "
                                f"{_fmtf(al.get('delta_deg'), '+.2f')} deg (rotated x {_fmtf(al.get('frac'), '.2f')}, remaining "
                                f"{_fmtf(al.get('psi_rem_deg'), '+.2f')} deg)")
        self.res.residual_history.append(_jsonable(rec))
        self._decide("line", f"L1 from the two measured root centres: L {L * 1e3:.1f} mm u1={np.round(u1, 4).tolist()} "
                             f"n1={np.round(n1, 4).tolist()} (u vs rough {rec['u_vs_u0_deg']:.2f} deg, n vs n0 {rec['n_vs_n0_deg']:.2f} deg); "
                             f"theta {_fmtf(pair.get('theta_deg'))} deg ({rec['theta_source']}), roll {roll:.2f} deg; psi two-dwell "
                             f"{_fmtf(cons['psi_indep_deg'])} / kinematic {_fmtf(cons['psi_kin_deg'])} deg (recorded); {scale['summary']}",
                     line=line.to_dict())
        if scale.get("note"):
            self.res.notes.append(scale["note"])
        if scale["stop"]:
            raise _Abort(STATUS_ABORTED, f"sensor x-scale inconsistent with the robot kinematics on two independent readings "
                                         f"({scale['summary']}): check the profiler calibration (x scale, optical yaw branch, "
                                         "hand-eye) before scanning")
        self.res.line_history.append(line.to_dict())
        return line

    def _scale_tol(self) -> tuple:
        """센서 x 스케일 허용 eps_max = 측면 설계 치수 (개구·갭) 의 작업 허용치 / 공칭 중 가장 작은 것 (그 스케일에서 치수 편향이
        허용치에 닿는다) → (eps_max, 근거). 허용치가 없으면 설계 중간점 (원칙 2: 설계 특징으로 볼 수 있는 한계)."""
        cand = []
        for key, nom in (("top_opening_mm", self._opening_spec_m() * 1e3),
                         ("root_gap_mm", float(self.spec.root_gap_mm) if self.spec is not None else float("nan"))):
            tol = (self.qc_targets.get(key) or {}).get("tol") if isinstance(self.qc_targets, dict) else None
            if _fin(tol) and _fin(nom) and nom > 0.0:
                cand.append((float(tol) / float(nom), f"{key} tol {float(tol):g} / {float(nom):.2f} mm"))
        if not cand:
            return float(_ms.DESIGN_HALF), "no lateral design tolerance in the job: design midpoint"
        return min(cand)

    def _scale_check(self, lat_E_sensor: float, lat_E_kin: float, loc_E_pre: StaticLocalization) -> dict:
        """센서 x 스케일 판정 (4d P4, 확인 먼저): 같은 측면 이동을 센서와 운동학이 잰 읽기 — E 지점 (보정 전 오프셋), 프로브 X 스텝,
        S 지점 (같은 규칙) 순서 — 중 판정 가능한 (testable) 앞의 두 개가 **둘 다** 허용을 넘을 때만 멈춘다. 순서는 이동이 순수 평행
        이동에 가까운 것부터: 평행 이동이면 핸드아이 오차가 두 자세에서 같아 상쇄되고 스케일만 남는다 (프로브 = 자세 그대로 X_opt 이동;
        E = 대개 면내 이동). S 보정은 3G 에서 거리 수십 mm·롤을 함께 바꿔 핸드아이 몫이 섞인다 (실기 g3_05–07 S +25–35 %, E·프로브는
        ±3 % 안) — 마지막 대안. 하나만 넘고 확인할 읽기가 없으면 노트 ('unconfirmed'), 판정 가능한 읽기가 없으면 노트 ('not testable')."""
        eps_max, basis = self._scale_tol()
        k = float(self.settings.judgement_k)
        rd = []
        z_E = (loc_E_pre.diag or {}).get("raw_z_axis_mm")
        rd.append(("E", scale_reading(lat_E_sensor, lat_E_kin, z_E, k=k, eps_max=eps_max)))
        pr = self.res.probe or {}
        if _fin(pr.get("scale_sensor_mm")) and _fin(pr.get("scale_kin_mm")):
            rd.append(("probe", scale_reading(float(pr["scale_sensor_mm"]) * 1e-3, float(pr["scale_kin_mm"]) * 1e-3,
                                              float(pr.get("range_m", float("nan"))) * 1e3, k=k, eps_max=eps_max)))
        sp = getattr(self, "_S_pair", None)
        if sp is not None:
            ls, lk = site_pre_laterals(sp[0], sp[1], sp[2], w=sp[3] if len(sp) > 3 else None)
            rd.append(("S", scale_reading(ls, lk, (sp[1].diag or {}).get("raw_z_axis_mm"), k=k, eps_max=eps_max)))
        used = [(n, r) for n, r in rd if r["testable"]][:2]
        if {n for n, _ in used} == {"probe", "S"}:
            used = used[:1]                       # 프로브와 S 는 같은 프로브 전 그루브 중심을 쓴다 — 독립된 두 읽기가 아니다 (4d 검토)
        stop = len(used) == 2 and all(r["exceeds"] for _, r in used)
        note = None
        if len(used) == 1 and used[0][1]["exceeds"]:
            note = (f"sensor x-scale reading {used[0][0]} {used[0][1]['eps'] * 100:+.2f} % exceeds {eps_max * 100:.2f} % "
                    "(unconfirmed: no second testable reading)")
        elif not used:
            note = "sensor x-scale not testable at these offsets (the readings are below the sensor resolution)"
        txt = ", ".join(f"{n} {r['eps'] * 100:+.2f} % (+-{r['allow'] * 100:.2f}{'' if r['testable'] else ', not testable'})"
                        for n, r in rd if _fin(r["eps"]))
        return {"eps_max": eps_max, "basis": basis, "readings": {n: r for n, r in rd}, "used": [n for n, _ in used], "stop": bool(stop),
                "note": note, "summary": f"sensor x-scale {txt or 'no reading'} vs {eps_max * 100:.2f} % ({basis})"}

    # ---- 6/7. 패스 작업 · 오버스캔 · 시작 이동 -------------------------------------------------------------------
    def _depth_m(self, fm: Optional[dict]) -> float:
        """패스 작업의 그루브 깊이 [m]: 패스가 잰 깊이, 없으면 지점 깊이들의 중앙값 — 설계 깊이로 볼 수 있는 읽기 (design_match, 원칙 2) 만;
        없으면 설계 깊이 (4d P5)."""
        D = float(self.spec.groove_depth_mm) if self.spec is not None else float("nan")
        if fm and _fin(fm.get("depth_mm")) and _ms.design_match(float(fm["depth_mm"]), D):
            return float(fm["depth_mm"]) * 1e-3
        ds = [float(s.get("depth_mm")) for s in self.res.sites.values() if _fin(s.get("depth_mm")) and _ms.design_match(float(s["depth_mm"]), D)]
        if ds:
            return float(np.median(ds)) * 1e-3
        return D * 1e-3 if _fin(D) else 0.0

    def _origin_above_hit_m(self, job: ScanJob, depth_m: float) -> float:
        """패스 job 의 광학 원점이 레이저 착점 (레이저 평면 ∩ 판 상면 위 심 직선) 보다 높은 양 [m] (up = −중력, base 3D, I2)."""
        fr = sensor_frames(job, self.calib)
        q0 = fr.start + fr.n * float(depth_m)
        delta = float(laser_plane_line_param(fr.T_base_opt(0.0), q0, fr.u))
        return float(fr.up @ (fr.opt_origin(0.0) - (q0 + fr.u * delta)))

    def _pass_job(self, line: SeamLine, plan: PassPlan, k: int, depth_m: float, main: bool = False, posture: Optional[dict] = None,
                  quiet: bool = False) -> ScanJob:
        """직선 + 패스 계획 → 패스 k 의 ScanJob (start = 진행 시작 목표의 루트 점, end = 진행 끝 목표; u = 진행 방향, travel_sign +1).
        standoff = fine (판 상면) + 깊이, pitch/입사 = 거친 job 값 (1G −5°/0, 3G ±|피치|/+10°), speed = 거친 job speed.
        progression_sign 은 ±1 중 현재 툴 자세에 더 가까운 쪽 — 진행이 뒤집혀도 툴 자세가 유지된다 (180° 손목 회전 없음).
        3G 피치 부호 (라운드 4 F1): 툴 자세를 유지하면 아무것도 피치의 부호를 정하지 않아, 템플릿 배치 (S 위, 'down', progression +1) 에서는
        −5° 가 +10° 내려다보는 입사의 절반을 상쇄했다 (원점이 착점보다 29 mm 위뿐 → 아래 끝 오버스캔 10.6 mm / REFUSED). 패스 1 에서
        **한 번** ±|피치| 중 광학 원점이 착점보다 더 높은 (= 입사 기울기에 더해지는) 쪽을 고르고 (설계: 피치 부호는 정반사 회피에 무관),
        그 값을 모든 패스와 정련 작업 (make_refined_job) 에 유지한다 (진행이 뒤집혀도 progression_sign 이 함께 뒤집혀 같은 물리 자세).
        posture (3G 주 패스, _main_posture): {'tilt', 'torch_above'} — 입사 각도와 토치 쪽 (그 쪽의 진행 부호만; 토치 아래 = 광축 둘레 180°).
        quiet: 결정 기록 없이 (자세 후보 평가)."""
        direction = plan.direction
        c, u = np.asarray(line.c, dtype=float), unit(line.u)
        P_S, P_E = c + u * float(plan.s_S), c + u * float(plan.s_E)
        a, b = (P_E, P_S) if direction == "E->S" else (P_S, P_E)
        kw = job_to_dict(self.job0)
        kw["torch_lateral_m"] = tuple(kw["torch_lateral_m"])
        kw.update(start=a.tolist(), end=b.tolist(), surface_normal=unit(line.n).tolist(), orientation=self.orientation,
                  standoff_m=self.pass_plate_m + float(depth_m), pitch_deg=float(self.job0.pitch_deg),
                  incidence_tilt_deg=float(posture["tilt"] if posture is not None else
                                           (self.tilt_main if main else self.job0.incidence_tilt_deg)), lead_in_m=0.001,
                  lead_out_m=0.001,
                  bidirectional=False, plate_top_measured_m=self.plate_top,
                  plate_plane_measured=self.job_ref.plate_plane_measured if self.job_ref is not None else None,
                  job_id=f"{self.job0.job_id or 'job'}_pass{k}")
        if self.orientation == "vertical_3G":
            kw["vertical_direction"] = "down" if float(self.up @ a) >= float(self.up @ b) else "up"
            kw["floor_z_m"] = self.floor_h                       # 연장된 패스 목표가 아니라 현재 바닥 (대략 아래 점 → 실측 아래 끝)
        p0 = float(self.job0.pitch_deg)
        if main:
            pitches = (0.0,)                                     # 3G 입사 분리 주 패스: 입사 = 주 각도 그대로
        elif self.pass_pitch_deg is not None:
            pitches = (float(self.pass_pitch_deg),)
        elif self.orientation == "vertical_3G" and p0 != 0.0:
            pitches = (abs(p0), -abs(p0))                        # 3G: 부호를 한 번 고른다 (아래)
        else:
            pitches = (p0,)
        R_now = self._T_now()[:3, :3]
        side = posture.get("torch_above") if (posture is not None and self.orientation == "vertical_3G") else None
        options = []
        for pitch in pitches:
            kw["pitch_deg"] = float(pitch)
            best = None
            for sign in (1, -1):
                kw["progression_sign"] = sign
                try:
                    job = ScanJob(**kw)
                    job.validate()
                except JobError as e:
                    raise _Abort(STATUS_REFUSED, f"pass {k} job invalid: {e} (nothing sent)", category="motion")
                if side is not None and self._job_torch_above(job) != bool(side):
                    continue
                rot = _rot_angle_deg(R_now, sensor_frames(job, self.calib).T_base_flange(0.0)[:3, :3])
                if best is None or rot < best[0]:
                    best = (rot, sign, job)
            if best is None:
                raise _Abort(STATUS_REFUSED, f"pass {k}: no progression sign gives the torch {'above' if side else 'below'} the laser "
                                             "(nothing sent)", category="motion")
            h = self._origin_above_hit_m(best[2], depth_m) if self.orientation == "vertical_3G" else float("nan")
            options.append((h, float(pitch), best))
        if len(options) > 1:
            options.sort(key=lambda o: -o[0])                     # 원점이 착점보다 가장 높은 쪽 (입사 기울기에 더해지는 피치)
            h, pitch = options[0][0], options[0][1]
            h_alt, pitch_alt = options[1][0], options[1][1]
            self.pass_pitch_deg = pitch
            if not quiet:
                self._decide("pass", f"pass {k}: 3G pitch sign chosen once: pitch {pitch:+.1f} deg adds to the incidence "
                                     f"{float(self.job0.incidence_tilt_deg):+.1f} deg - optical origin {h * 1e3:.1f} mm above the laser hit "
                                     f"point ({h_alt * 1e3:.1f} mm with {pitch_alt:+.1f} deg); kept for every pass and the refined job "
                                     "(the pitch sign does not matter for specular avoidance)",
                             pitch_deg=pitch, origin_above_hit_m=h, alternative={"pitch_deg": pitch_alt, "origin_above_hit_m": h_alt})
        elif self.orientation == "vertical_3G" and self.pass_pitch_deg is None and not main:
            self.pass_pitch_deg = options[0][1]
        rot, sign, job = options[0][2]
        if self.orientation == "vertical_3G":
            job = self._fit_laser_tilt(job, k, quiet=quiet)
        job, raised = self._clear_plate(job, depth_m)
        if quiet:
            return job
        self._decide("pass", f"pass {k} ({direction}, rule {plan.rule}): S target {plan.how_S} (hit overscan {plan.ov_S * 1e3:.0f} mm), "
                             f"E target {plan.how_E} (hit overscan {plan.ov_E * 1e3:.0f} mm); progression_sign {sign:+d} keeps the tool "
                             f"orientation (rotation from the current pose {rot:.2f} deg: pitch {job.pitch_deg:+.1f} deg + alignment "
                             f"refinement); standoff {job.standoff_m:.4f} m (plate {self.pass_plate_m:.4f} + depth {depth_m * 1e3:.2f} mm"
                             + (f" + {raised * 1e3:.1f} mm to keep the tool {(job.safe_clearance_m + job.pose_tol_m) * 1e3:.0f} mm "
                                "from the plate" if raised > 0.0 else "") + "), "
                             f"L {job.length_m * 1e3:.1f} mm"
                     + (f"; posture incidence {job.incidence_tilt_deg:+.0f} deg, torch {'above' if posture.get('torch_above') else 'below'} "
                        f"the laser (phi {posture.get('phi', float('nan')):+.1f} deg)" if posture is not None else ""),
                     progression_sign=sign, rotation_deg=rot, plan=plan.to_dict())
        return job

    def _fit_laser_tilt(self, job: ScanJob, k: int, *, quiet: bool = False) -> ScanJob:
        """패스 job 의 레이저 기울기 (광축 vs 포락선 기준 법선 = job_ref 의 판 법선) 가 max_laser_tilt_deg 를 조금 넘으면 넘친 만큼 (+0.05°) 피치
        크기를 줄인다 (입사·진행 부호 유지). 3G 끝 패스는 입사 35° + 피치 5° = 40° 로 한계에 딱 맞춰 계획하는데, 패스 직선의 판 법선과 포락선
        기준 법선이 측정 산포로 0.2° 남짓 달라 패스 시작이 거부됐다 (2026-10-06 g3_26: 40.23° > 40°; g3_25 는 같은 자세가 통과). 2° 넘게
        넘치면 그대로 둔다 (포락선이 거부 — 측정 산포가 아니다)."""
        ref = self.job_ref
        p = float(job.pitch_deg)
        if ref is None or abs(p) < 1e-9:
            return job
        try:
            n_ref = unit(sensor_frames(ref, self.calib).n)

            def tilt_of(j: ScanJob) -> float:
                Z = sensor_frames(j, self.calib).T_base_opt(0.0)[:3, 2]
                return math.degrees(math.acos(max(-1.0, min(1.0, float(Z @ (-n_ref))))))
            t0 = tilt_of(job)
        except Exception:  # noqa: BLE001
            return job
        excess = t0 - float(job.max_laser_tilt_deg)
        if excess <= 0.0:
            return job                                         # 4d P10: 넘친 만큼 줄이고 다시 검사 (예전 2° 산포 상한 없음)
        new_p = math.copysign(max(0.0, abs(p) - excess - 0.05), p)
        kw = job_to_dict(job)
        kw["torch_lateral_m"] = tuple(kw["torch_lateral_m"])
        kw["pitch_deg"] = new_p
        try:
            j2 = ScanJob(**kw)
            j2.validate()
            t1 = tilt_of(j2)
        except Exception:  # noqa: BLE001
            return job
        if t1 > float(job.max_laser_tilt_deg):
            return job
        if not quiet:
            self._decide("pass", f"pass {k}: laser tilt {t0:.2f} deg vs the reference plate normal > max_laser_tilt_deg "
                                 f"{job.max_laser_tilt_deg:g} (plate-normal scatter) -> pitch {p:+.2f} -> {new_p:+.2f} deg (tilt {t1:.2f} deg)",
                         pitch_from_deg=p, pitch_to_deg=new_p, tilt_from_deg=t0, tilt_to_deg=t1)
        return j2

    def _job_torch_above(self, job: ScanJob) -> bool:
        """EOAT +x (센서 카메라 끝) 가 아래 = 토치 (−x 쪽) 가 레이저 위 (scan_quality.pass_posture 와 같은 정의)."""
        return bool(float(sensor_frames(job, self.calib).R_base_eoat[:, 0] @ job.up_unit) < 0.0)

    def _lower_end_key(self, fit: PassFit) -> Optional[str]:
        """fit.diag['ends'] 중 아래 (중력 쪽) 끝의 키 ('first'/'last')."""
        ends = (fit.diag or {}).get("ends") or {}
        if "first" not in ends or "last" not in ends:
            return None
        return "first" if float(self.up @ np.asarray(ends["first"]["point"])) <= float(self.up @ np.asarray(ends["last"]["point"])) else "last"

    def _keep_lower_end(self, fit: PassFit, k: int) -> None:
        """3G 입사 분리의 끝 패스: 관측된 아래 끝 (테이블 교점이면 그 점) 을 기억한다 — 주 패스가 이어받는다. 끝 패스의 직선과 관측된 위
        끝도 기억한다 (자세 간 오프셋: _carry_lower_end)."""
        w = self._lower_end_key(fit)
        ends = (fit.diag or {}).get("ends") or {}
        e = ends.get(w) if w else None
        if e is not None and e.get("observed") and not e.get("occluded"):
            self.lower_end_obs = {"pass": k, "point": [float(v) for v in e["point"]], "support": bool(e.get("support_plane")),
                                  "which": w, "sigma_m": float(e.get("sigma_m", float("nan")))}
            up_ = ends.get("last" if w == "first" else "first") or {}
            try:
                side = bool(self._torch_above_now())
            except Exception:  # noqa: BLE001 - 자세 없음 (dry run)
                side = None
            phi_e = float(((self.pass_quality[-1] if self.pass_quality else {}).get("posture") or {}).get("phi_deg", float("nan")))
            self.end_pass_ref = {"pass": k, "c": [float(v) for v in fit.c], "u": [float(v) for v in unit(fit.u)],
                                 "posture": {"tilt": self.tilt_end, "torch_above": side, "phi": phi_e},
                                 "upper": ([float(v) for v in up_["point"]] if up_.get("observed") and not up_.get("occluded")
                                           and not up_.get("inferred") else None)}

    def _posture_offset(self, fit: PassFit, k: int) -> Optional[dict]:
        """끝 패스 ↔ 이 (주) 패스의 측정 오프셋 (이 패스 기준 = 이 패스 − 끝 패스): 단면 (판 법선·옆) 은 끝 패스의 위 끝을 이 직선에
        투영한 차이, 심 방향은 두 패스가 모두 본 위 끝의 직선 좌표 차이. 위 끝을 둘 다 못 봤으면 None."""
        ref = getattr(self, "end_pass_ref", None)
        ends = (fit.diag or {}).get("ends") or {}
        w = self._lower_end_key(fit)
        if ref is None or ref.get("upper") is None or w is None:
            return None
        up_main = ends.get("last" if w == "first" else "first") or {}
        if not up_main.get("observed") or up_main.get("occluded") or up_main.get("inferred"):
            return None
        c, u = np.asarray(fit.c, dtype=float), unit(fit.u)
        Q = np.asarray(ref["upper"], dtype=float)
        s_q = float((Q - c) @ u)
        d = (c + u * s_q) - Q                                    # 이 직선 − 끝 패스 위 끝 (단면 성분)
        n_ = unit(np.asarray(fit.n, dtype=float)) if np.all(np.isfinite(fit.n)) else unit(np.asarray(self.n0, dtype=float))
        lat = unit(np.cross(n_, u))
        ang = math.degrees(math.acos(min(1.0, abs(float(u @ unit(np.asarray(ref["u"], dtype=float)))))))
        return {"from_pass": int(ref["pass"]), "to_pass": int(k), "along_m": float(up_main["s_m"]) - s_q,
                "normal_m": float(d @ n_), "lateral_m": float(d @ lat), "perp_m": float(np.linalg.norm(d)), "angle_deg": ang}

    def _along_bound(self) -> tuple:
        """자세 간 심 방향 오프셋 한계 B_along (자료: scan_l0.posture_offsets; 이 캘리브·이 자세 쌍) → (m, 근거)."""
        endp = (getattr(self, "end_pass_ref", None) or {}).get("posture")
        return _po.along_bound(str(getattr(self.calib, "date", "") or ""), endp, self.main_posture, float(self.settings.judgement_k))

    @staticmethod
    def _same_posture(a: Optional[dict], b: Optional[dict]) -> bool:
        """같은 자세 = φ 가 자세 격자 반 칸 안이고 토치 쪽이 같다 (4d: _bad_posture·재스캔 'tried' 와 같은 규칙; φ 가 없으면 기울기 일치)."""
        a, b = a or {}, b or {}
        if a.get("torch_above") != b.get("torch_above"):
            return False
        pa, pb = a.get("phi"), b.get("phi")
        if _fin(pa) and _fin(pb):
            return abs(float(pa) - float(pb)) < 0.5 * _sq.POSTURE_GRID_DEG
        return a.get("tilt") is not None and a.get("tilt") == b.get("tilt")

    def _same_posture_as_end(self) -> bool:
        endp = (getattr(self, "end_pass_ref", None) or {}).get("posture")
        return endp is not None and self._same_posture(endp, self.main_posture)

    def _carry_lower_end(self, fit: PassFit, k: int) -> None:
        """3G 입사 분리의 주 패스: 끝 패스의 아래 끝 (테이블 교점) 을 이 패스 기준으로 옮겨 쓴다. 끝은 끝 패스가, 특징은 주 패스가 잰다:
        주 패스가 아래 끝까지 닿아 스스로 봤어도 끝 패스 값을 쓰고 차이를 기록한다.

        자세 간 오프셋 (2026-10-06 g3_13): 토치 위 40° (끝 패스) 와 토치 아래 15° (주 패스) 가 같은 심을 판 법선 4.7 mm·심 방향 13 mm
        다르게 쟀다 (손-눈/자세 의존 오차를 뒤집기가 드러냈다) → 예전 규칙 (아래 끝이 이 직선에서 tol_m 안일 때만 투영) 이 거부해 주 패스가
        아래 끝을 끝내 못 봤다. 이제 두 패스가 모두 본 위 끝으로 오프셋을 재서 (_posture_offset) 아래 끝을 이 직선에 투영하고 심 방향
        오프셋만큼 옮긴다 — 결과 끝·길이는 이 패스 (주 자세) 기준으로 일관되고, 오프셋은 기록된다 (캘리브 자료). 4d P8: 받아들이는 심 방향
        오프셋 = 자료의 B_along (|중앙값| + k·강건 σ, scan_l0.posture_offsets); 이 패스가 아래 끝을 스스로도 봤으면 두 끝이 k·σ 안이어야
        한다. 단면·각은 이어받은 끝을 움직이지 않으므로 기록·노트만. 오프셋을 못 재면: 끝 패스와 같은 자세면 오프셋 0, 같은 자세 쌍의 자료가
        3 개 이상이면 그 중앙값 (끝 σ 를 그 척도로 넓힘), 아니면 이어받지 않는다. 이어받지 못하면 self.carry_refused 에 사유 (실행 루프가
        확인 먼저: 다음 주 패스가 다시 잰다).
        fit 을 제자리에서 고친다: ends[w] observed·s_m·point, carried_from, 플래그, s_a/s_b, 대안 끝."""
        self.carry_refused = None
        self._carry_refusal_final = False
        lo = self.lower_end_obs
        ends = (fit.diag or {}).get("ends") or {}
        w = self._lower_end_key(fit)
        if lo is None or w is None or not (np.all(np.isfinite(fit.c)) and np.all(np.isfinite(fit.u))):
            return
        own = bool(ends[w].get("observed"))
        c, u = np.asarray(fit.c, dtype=float), unit(fit.u)
        P = np.asarray(lo["point"], dtype=float)
        s_ = float((P - c) @ u)
        off = float(np.linalg.norm(P - (c + u * s_)))
        po = self._posture_offset(fit, k)
        along, widen = 0.0, 0.0
        kk = float(self.settings.judgement_k)
        sig_line = float(fit.line_rms_m) if _fin(fit.line_rms_m) else 0.0
        sig_lo = float(lo.get("sigma_m", float("nan")))
        sig_end = math.hypot(sig_lo if _fin(sig_lo) else 0.0, float(ends[w].get("sigma_m") or 0.0))
        b_along, b_basis = self._along_bound()
        if po is not None:
            po["posture"] = {k_: (self.main_posture or {}).get(k_) for k_ in ("tilt", "torch_above", "phi")}
            po["end_posture"] = dict((self.end_pass_ref or {}).get("posture") or {})
            self.posture_offsets.append(po)
            same = po["perp_m"] <= kk * math.sqrt(2.0) * sig_line and abs(po["along_m"]) <= kk * math.sqrt(2.0) * sig_end
            self._decide("ends", f"pass {k}: posture offset vs end pass {po['from_pass']} (this pass - end pass, from the upper end both "
                                 f"saw): along the seam {po['along_m'] * 1e3:+.1f} mm (bound {b_along * 1e3:.1f} mm from {b_basis['n']} "
                                 f"record(s), {b_basis['key']}), plate normal {po['normal_m'] * 1e3:+.2f} mm, lateral "
                                 f"{po['lateral_m'] * 1e3:+.2f} mm, line angle {po['angle_deg']:.2f} deg"
                                 + (" (same within the measured noise)" if same else
                                    " - posture-dependent measurement offset (hand-eye / end-zone lag); the lower end is moved by it"),
                         posture_offset=_jsonable(po), along_bound=_jsonable(b_basis))
            if abs(po["along_m"]) > b_along:
                self.carry_refused = (f"the end pass {lo['pass']} and this posture disagree along the seam by {po['along_m'] * 1e3:+.1f} mm "
                                      f"(> {b_along * 1e3:.1f} mm, the recorded posture offsets {b_basis['key']} n={b_basis['n']}) - "
                                      "not a scan-quality problem: check the calibration")
            else:
                along = float(po["along_m"])
        elif self._same_posture_as_end():
            along = 0.0                                              # 같은 자세: 오프셋 없음 (구성상)
        else:
            recs = [r for r in _po.records() if _po.pair_key(r.get("end_posture"), r.get("main_posture"))
                    == _po.pair_key((self.end_pass_ref or {}).get("posture"), self.main_posture)
                    and r.get("calib_date") == str(getattr(self.calib, "date", "") or "")]
            if len(recs) >= _po.MIN_RECORDS:
                a_ = np.array([float(r["along_m"]) for r in recs])
                along = float(np.median(a_))
                widen = 1.4826 * float(np.median(np.abs(a_ - along)))
                self._decide("ends", f"pass {k}: posture offset not measurable in this pass (the upper end was not seen by both passes) - "
                                     f"using the median of {len(recs)} recorded offsets for this calibration and posture pair "
                                     f"({along * 1e3:+.1f} mm, end sigma widened by {widen * 1e3:.1f} mm)")
            else:
                self.carry_refused = (f"lower end of pass {lo['pass']}: the posture offset cannot be measured (the upper end was not seen "
                                      f"by both passes) and there are only {len(recs)} recorded offsets for this calibration and posture pair")
                if (self.end_pass_ref or {}).get("upper") is None:
                    self._carry_refusal_final = True             # 끝 패스가 위 끝을 못 봤다: 다음 주 패스도 같은 결과 (확인할 것이 없다)
        if self.carry_refused is None and own and po is not None:
            d_own = abs((s_ + along) - float(ends[w]["s_m"]))
            tol_own = kk * sig_end
            if _fin(tol_own) and tol_own > 0.0 and d_own > tol_own:
                self.carry_refused = (f"this pass saw the lower end itself {d_own * 1e3:.1f} mm from the carried end (> k sigma "
                                      f"{tol_own * 1e3:.1f} mm) - the posture offset does not hold at the lower end")
        if self.carry_refused:
            self._decide("ends", f"pass {k}: {self.carry_refused} - not carried")
            return
        s_c = s_ + along
        old = float(ends[w]["s_m"])
        sig_c = math.hypot(sig_lo, widen) if _fin(sig_lo) else float("nan")
        ends[w].update(observed=True, s_m=s_c, s_plates_m=s_c, point=(c + u * s_c).tolist(), carried_from=int(lo["pass"]),
                       carried_support=bool(lo.get("support")), carried_offset_m=off, carried_along_m=along, own_observed=own,
                       own_s_m=old, own_sigma_m=ends[w].get("sigma_m"),
                       sigma_m=sig_c if _fin(sig_c) else ends[w].get("sigma_m"))   # 이어받은 끝 = 끝 패스의 측정 (판 길이 끝도, σ 도)
        fit.flags = [f_ for f_ in fit.flags if f_ != f"end_not_observed_{w}"]
        fit.flags.append(f"end_carried_{w}")
        other = "last" if w == "first" else "first"
        if "alt_s_m" in ends[other] and self.spec is not None and self.spec.weld_length_mm > 0:
            Ls = self.spec.weld_length_mm * 1e-3
            d_ab = float(np.sign(ends["last"]["s_m"] - ends["first"]["s_m"]) or 1.0)
            ends["first"]["alt_s_m"] = ends["last"]["s_m"] - d_ab * Ls
            ends["last"]["alt_s_m"] = ends["first"]["s_m"] + d_ab * Ls
            for wh in ("first", "last"):
                ends[wh]["alt_point"] = (c + u * ends[wh]["alt_s_m"]).tolist()
        fit.s_a, fit.s_b = sorted((float(ends["first"]["s_m"]), float(ends["last"]["s_m"])))
        self._decide("ends", f"pass {k} (main posture): lower end carried from pass {lo['pass']} "
                             f"({'where the root line meets the table' if lo.get('support') else 'observed'}): {off * 1e3:.2f} mm off "
                             f"this line" + (f", moved {along * 1e3:+.1f} mm along the seam by the posture offset" if along else "")
                             + f", {(s_c - old) * 1e3:+.1f} mm "
                             + ("from this pass's own lower end observation (the end pass value is kept)" if own else
                                "beyond this pass's last groove profile"))

    # ---- 적응형 스캔 (사용자 2026-10-06): 패스별 품질 → 구간별 최선 자료 병합 → 나쁜 구간은 다른 자세로 재스캔 -----------------
    FEATURE_FAIL_KEYS = ("root_gap_mm", "depth_mm", "top_opening_mm", "bevel_angle_deg", "ok_fraction", "measurement_noise",
                         "floor_visible_fraction", "root_edges_fraction")

    def _record_quality(self, k: int, job: ScanJob, fit: PassFit, feats: dict, full: bool) -> None:
        e_ = pass_quality_entry(k, job, self.calib, fit, feats, self.spec, self.qc_targets, self._process(), self.settings.judgement_k,
                                full)
        self.pass_quality.append(e_)
        posture, fr = e_["posture"], e_["fractions"]
        self._decide("quality", f"pass {k}{'' if full else ' (rescan)'}: incidence {posture.get('incidence_deg', float('nan')):.1f} deg "
                                f"(tilt {posture.get('tilt_deg', float('nan')):+.0f}, pitch {posture.get('pitch_deg', float('nan')):+.0f}, "
                                f"standoff {posture.get('standoff_m', float('nan')) * 1e3:.0f} mm, phi {posture.get('phi_deg', float('nan')):+.1f} deg"
                                + ("" if posture.get("torch_above") is None else
                                   f", torch {'above' if posture.get('torch_above') else 'below'}")
                                + f"): good sections {fr['frac_good'] * 100:.0f} % "
                                f"(bevel {fr['frac_bevel'] * 100:.0f} %, gap readable {fr['frac_gap'] * 100:.0f} %, groove "
                                f"{fr['frac_groove'] * 100:.0f} %) on the specimen", posture=_jsonable(posture), fractions=fr)

    def _merged_gate(self, k: int, ref: dict) -> tuple:
        """기준 (마지막 전체 패스) 의 관측 양끝 S, E (S→E 계약 좌표 — 패스 진행 방향과 무관) 로 모든 패스의 프로파일을 구간별 최선으로
        병합 → 중앙값을 바꿔 fidelity_gate 다시. 반환 (verdict, 병합 feats 또는 None, fm)."""
        fit = ref["fit"]
        entries = [e for e in self.pass_quality if e["feats"] is not None]
        merged, choice, bin_m = (merged_pass_features(entries, ref["extent"], fit, self._process()) if fit.ok and entries
                                 else (None, [], float("nan")))
        if merged is None:
            return self._gate_with(ref, ref["fm"]), None, ref["fm"]
        self.merge_choice = choice
        mm = _sq.merged_medians(merged)
        fm2 = dict(ref["fm"])
        fm2.update({k_: v for k_, v in mm.items() if np.isfinite(v)})
        used = sorted({c_["pass"] for c_ in choice})
        poor = [c_ for c_ in choice if c_["score"] < _sq.GOOD_FRAC_MIN]
        v = self._gate_with(ref, fm2, merged)
        self._decide("gate", f"pass {k}: merged features (best pass per {bin_m * 1e3:.0f} mm section; passes {used}, "
                             f"{len(poor)}/{len(choice)} sections below {_sq.GOOD_FRAC_MIN:.1f}): gap {_fmtf(mm.get('root_gap_mm'))} depth "
                             f"{_fmtf(mm.get('depth_mm'))} bevel {_fmtf(mm.get('bevel_angle_deg'))} opening {_fmtf(mm.get('top_opening_mm'))} "
                             f"-> {v.summary()}", merge=_jsonable(choice))
        return v, merged, fm2

    def _gate_with(self, ref: dict, fm: dict, feats: Optional[dict] = None):
        cov = (pass_coverage(feats, float(feats["s_a_m"]), float(feats["s_b_m"]), self._process(), float(ref.get("end_lag_m", 0.0)))
               if feats is not None else ref.get("coverage"))
        return fidelity_gate(fm, ref["residuals"], self.spec, "fidelity", qc_targets=self.qc_targets, extent=ref["extent"],
                             accept_inferred_end=self.at.accept_inferred_end, k=self.settings.judgement_k, coverage=cov,
                             validity=self._validity())

    def _process(self) -> WeldProcess:
        """용접 경로 단면 계획 (job 문서 weld_plan 블록 — 공정 가정값): 측정 충분성의 단면 창·최소 프로파일 수."""
        return WeldProcess.from_dict(weld_process_doc(self.doc if isinstance(self.doc, dict) else None, self.res.position))

    def _validity(self) -> dict:
        """게이트의 데이터 유효성 기준: 센서 창 (플랫폼), 도착 허용 (안전 정책), 특징 추출기가 그루브 옆에 필요한 판 길이."""
        e = self.policy.pose_tol_m if self.policy is not None else float(self.job0.pose_tol_m)
        return {"z_valid_mm": tuple(self.settings.z_valid_mm), "pose_tol_mm": float(e) * 1e3,
                "plate_min_mm": float(_sfmod.plate_min_len_mm(self.spec))}

    def _feature_failure(self, verdict) -> bool:
        """게이트 실패가 특징 판독 (자세로 나아질 수 있는 것) 이고 끝·길이 문제가 아닐 때."""
        failed = list(getattr(verdict, "failed", []) or [])
        if any(f_.startswith("end_") or f_ in ("weld_length_mm",) for f_ in failed):
            return False
        return any(any(key in f_ for key in self.FEATURE_FAIL_KEYS) for f_ in failed)

    def _tilt_grid(self) -> list:
        """3G 입사 기울기 후보 (4d P7, 주 패스·재스캔 공통): 0..센서 한계를 자세 격자 (POSTURE_GRID_DEG) 로 + 끝 자세 각도 + 지금 주 각도."""
        ts = [float(t) for t in np.arange(0.0, self._tilt_limit_deg() + 1e-9, _sq.POSTURE_GRID_DEG)]
        if self.tilt_end is not None:
            ts.append(float(round(abs(float(self.tilt_end)))))
        if self.main_posture is not None:
            ts.append(float(abs(float(self.main_posture["tilt"]))))
        elif self.tilt_main is not None:
            ts.append(float(round(abs(float(self.tilt_main)))))
        return sorted(t for t in dict.fromkeys(ts) if 0.0 <= t <= self._tilt_limit_deg())

    def _posture_candidates(self) -> list:
        """재스캔 자세 후보 (입사 기울기, 피치, 토치 위 여부 또는 None). 3G: 공통 격자 (_tilt_grid) × 지금 토치 쪽 (+ 뒤집은 쪽,
        torch_flip_3g); 1G: 피치 반대·±격자 (토치 쪽은 지금 자세에 가까운 쪽)."""
        if self.orientation == "vertical_3G":
            side_now = self._torch_above_now()
            sides = (side_now, not side_now) if (self.at.torch_flip_3g and self.at.posture_3g_adaptive) else (side_now,)
            return [(t, 0.0, sd) for sd in sides for t in self._tilt_grid()]
        p0 = float(self.job0.pitch_deg)
        g_ = _sq.POSTURE_GRID_DEG
        alt = -p0 if p0 != 0.0 else g_
        return [(0.0, alt, None), (0.0, alt + (g_ if alt > 0 else -g_), None)]   # 정반사 (−12°) 쪽으로는 가지 않는다 (phi_prior 가 0)

    def _torch_above_now(self) -> bool:
        return bool(float((self._T_now() @ self.calib.T_flange_eoat)[:3, 0] @ self.up) < 0.0)

    # ---- 3G 주 패스 자세 (적응형, 2026-10-06 g3_12) ----------------------------------------------------------
    def _measured_phis(self) -> list:
        """이번 실행에서 잰 (φ, 좋은 단면 비율) — 데이터셋·정적 사전값보다 우선 (scan_quality.phi_model_points)."""
        out = []
        for e in self.pass_quality:
            ph = float((e.get("posture") or {}).get("phi_deg", float("nan")))
            g = float((e.get("fractions") or {}).get("frac_good", float("nan")))
            if np.isfinite(ph) and np.isfinite(g):
                out.append((ph, g))
        return out

    def _bad_posture(self, phi: float) -> Optional[dict]:
        return next((b for b in self.bad_postures if np.isfinite(phi) and abs(float(b["phi"]) - float(phi)) < 0.5 * _sq.POSTURE_GRID_DEG),
                    None)                                         # 같은 자세 = 격자 반 칸 안 (4d P7)

    def _note_posture_outcome(self, k: int, fitted: bool) -> None:
        """3G 패스 (끝 패스 제외) 뒤: 좋은 단면 < GOOD_FRAC_MIN 이거나 직선을 못 맞췄으면 그 자세 (φ) 를 다시 쓰지 않는다."""
        e = self.pass_quality[-1] if self.pass_quality and int(self.pass_quality[-1]["k"]) == int(k) else None
        if e is None:
            return
        g = float(e["fractions"].get("frac_good", 0.0))
        if fitted and g >= _sq.GOOD_FRAC_MIN:
            return
        post = e["posture"] or {}
        ph = float(post.get("phi_deg", float("nan")))
        if not np.isfinite(ph):
            return
        why = f"{g * 100:.0f} % good sections" + ("" if fitted else ", no seam line fitted")
        self.bad_postures.append({"k": int(k), "phi": ph, "torch_above": bool(post.get("torch_above")), "frac_good": g, "why": why})
        self._decide("posture", f"pass {k}: posture phi {ph:+.1f} deg (incidence {float(post.get('incidence_deg', float('nan'))):.0f} deg, "
                                f"torch {'above' if post.get('torch_above') else 'below'}) read {why} -> not used again in this run",
                     bad=_jsonable(self.bad_postures[-1]))

    def _eval_main_posture(self, line: SeamLine, plan: PassPlan, k: int, depth_m: float, tilt: float, side: bool, measured: list,
                           T_now: np.ndarray) -> dict:
        """후보 자세 하나 → {tilt, torch_above, phi, incidence, pred, coverage, score, rot, (skip)} (뒤집기 가능 여부는 고를 때 본다)."""
        posture = {"tilt": float(tilt), "torch_above": bool(side)}
        try:
            job = self._pass_job(line, plan, k, depth_m, main=True, posture=posture, quiet=True)
            ov = self._overscan_raw(job, plan, depth_m)
            post = _sq.pass_posture(job, self.calib, plate_normal=line.n)
        except (_Abort, JobError, EnvelopeError, ValueError) as e:
            return {**posture, "skip": f"invalid: {str(getattr(e, 'reason', e))[:80]}"}
        L = float(job.length_m)
        cov = max(0.0, min(float(ov["hit_to"]), L) - max(float(ov["hit_from"]), 0.0)) / L if L > 0 else 0.0
        pred, g, why = _sq.predicted_score(post, str(self.res.position or "3G"), self.at.quality_dataset_path or None, measured)
        row = {**posture, "phi": float(post["phi_deg"]), "incidence": float(post["incidence_deg"]), "pred": float(pred), "g": float(g),
               "why": why, "coverage": float(cov), "score": float(pred) * min(1.0, float(cov)),
               "rot": _rot_angle_deg(T_now[:3, :3], np.asarray(ov["T_from"])[:3, :3]), "T_from": np.asarray(ov["T_from"])}
        bad = self._bad_posture(row["phi"])
        if bad is not None:
            row["skip"] = f"pass {bad['k']} read {bad['why']} at phi {bad['phi']:+.1f}"
        elif cov < _ms.DESIGN_HALF:
            # 주 패스가 병합 중앙값을 정하려면 프로파일의 절반 넘게를 대야 한다 (중앙값의 50 % 파괴점; 4d P7, 예전 0.4)
            row["skip"] = f"the laser reaches only {cov * 100:.0f} % of the length above the table (< half)"
        return row

    def _main_posture(self, line: SeamLine, plan: PassPlan, k: int, depth_m: float) -> tuple:
        """3G 주 패스 k 의 자세 → (posture 또는 None, 멈춤 사유 또는 None). 앞 주 패스 자세가 나쁘지 않으면 그대로. 아니면 후보
        (공통 격자 _tilt_grid) × (토치 위, 토치 아래 = 광축 둘레 180°) 를 거른 뒤 줄 세운다 (4d P7): 자격 = 예상 좋은 비율 ≥ 절반 (이번 실행
        측정 > 데이터셋 > 씨앗; 모름 = 절반은 주 자세가 아직 나쁘지 않았을 때만), 나쁜 자세·정반사·닿는 비율 < 절반·뒤집기 불가 아님;
        줄 = 예상 점수 (입사 가중 × g) × 닿는 비율, 회전 작은 쪽. 자격 있는 후보가 없으면 주 자세가 이미 나빴으면 멈추고 (같은 결과를
        되풀이하지 않는다), 아니면 가장 높은 후보를 한 번 재 본다 (확인 먼저)."""
        from .config import parse_posture_3g
        forced = parse_posture_3g(self.at.posture_3g_main)
        if forced is not None and self.orientation == "vertical_3G" and self.tilt_main is not None:
            if self.main_posture is not None:
                return self.main_posture, None
            T_now = self._T_now()
            r_ = self._eval_main_posture(line, plan, k, depth_m, forced[0], forced[1], self._measured_phis(), T_now)
            what = f"forced main posture {self.at.posture_3g_main!r} (auto_teach.posture_3g_main)"
            if "phi" not in r_:
                return None, f"3G pass {k}: {what} cannot be used: {r_.get('skip')}"
            if self._is_flip(T_now, r_["T_from"]) and self._flip_mid(T_now, r_["T_from"]) is None:
                return None, f"3G pass {k}: {what} needs a torch flip and no in-place 90 deg rotation passes the envelope here"
            self.main_posture = {k_: r_[k_] for k_ in ("tilt", "torch_above", "phi", "incidence", "pred", "coverage", "why")}
            self._decide("posture", f"pass {k}: {what}: incidence {r_['tilt']:+.0f} deg, torch {'above' if r_['torch_above'] else 'below'} "
                                    f"(phi {r_['phi']:+.1f} deg), laser reaches {r_['coverage'] * 100:.0f} % of the length above the table, "
                                    f"predicted {r_['pred']:.2f}" + (f"; TORCH FLIP: {r_['rot']:.0f} deg rotation, sent as 90 deg in place "
                                                                      "about the optical axis + the rest"
                                                                      if self._is_flip(T_now, r_["T_from"]) else ""),
                         posture=_jsonable(self.main_posture))
            return self.main_posture, None
        if not (self.at.posture_3g_adaptive and self.orientation == "vertical_3G" and self.tilt_main is not None):
            return None, None
        if self.main_posture is not None and self._bad_posture(self.main_posture["phi"]) is None:
            return self.main_posture, None
        T_now = self._T_now()
        side_now = self._torch_above_now()
        sides = (side_now, not side_now) if self.at.torch_flip_3g else (side_now,)
        tilts = self._tilt_grid()
        measured = self._measured_phis()
        rows = [self._eval_main_posture(line, plan, k, depth_m, t, sd, measured, T_now) for sd in sides for t in tilts]
        # 4d P7: 거른 뒤 줄 세운다 — 자격 = 예상 좋은 비율 g ≥ 절반 (모름 = 절반은 자격), 나쁜 자세·정반사 (g 0) 아님; 줄 = 예상 점수
        # × 닿는 비율, 회전 작은 쪽. 자격 있는 후보가 없으면: 이번 실행에서 주 자세가 이미 한 번 나빴으면 멈추고, 아니면 가장 높은 후보를
        # 한 번 재 본다 (확인 먼저). 포락선·뒤집기·무효로 모두 빠지면 REFUSED (도달).
        valid = [r_ for r_ in rows if "phi" in r_]
        for r_ in valid:
            if "skip" not in r_ and self._is_flip(T_now, r_["T_from"]) and self._flip_mid(T_now, r_["T_from"]) is None:
                r_["skip"] = "torch flip: no in-place 90 deg rotation passes the envelope here"
        reachable = [r_ for r_ in valid if not str(r_.get("skip", "")).startswith("torch flip")]
        poor_main = any(b_["k"] >= 2 for b_ in self.bad_postures)

        def evidenced(r_):                                         # 측정 근거 (씨앗·데이터셋·이번 실행·이웃 보간) 가 있는 예상
            return not str(r_.get("why", "")).startswith(("no measurement", "unknown"))
        # 주 자세가 이번 실행에서 한 번 나빴으면 근거 없는 (모름 = 절반) 후보는 더 이상 자격이 없다 — 근거 있는 좋은 자세만 (같은 결과를
        # 되풀이하지도, 잰 적 없는 자세를 하나씩 다 써 보지도 않는다)
        eligible = sorted((r_ for r_ in reachable if "skip" not in r_ and r_["g"] >= _ms.DESIGN_HALF and (evidenced(r_) or not poor_main)),
                          key=lambda r_: (-round(r_["score"], 3), r_["rot"]))
        el_ids = {id(r_) for r_ in eligible}
        others = sorted((r_ for r_ in reachable if "skip" not in r_ and id(r_) not in el_ids),
                        key=lambda r_: (-round(r_["score"], 3), r_["rot"]))
        best = eligible[0] if eligible else (others[0] if (others and not poor_main) else None)

        def txt(r_):
            return (f"{r_['tilt']:.0f}deg torch {'above' if r_['torch_above'] else 'below'}"
                    + (f" (skip: {r_['skip']})" if "skip" in r_ and "phi" not in r_ else
                       f" phi {r_['phi']:+.0f}: pred {r_['pred']:.2f} x reach {r_['coverage'] * 100:.0f}% = {r_['score']:.2f}"
                       + (f" (skip: {r_['skip']})" if "skip" in r_ else "")))
        table = "; ".join(txt(r_) for r_ in sorted(rows, key=lambda r_: -r_.get("score", -1.0)))
        cand = _jsonable([{k_: v for k_, v in r_.items() if k_ != "T_from"} for r_ in rows])
        if best is None:
            tried = ", ".join(f"pass {b['k']} phi {b['phi']:+.0f} ({b['why']})" for b in self.bad_postures) or "none"
            if not reachable:
                why = (f"3G pass {k}: no main posture is reachable here (envelope, torch flip or invalid job for every candidate) - "
                       "REFUSED (reach)")
            else:
                why = (f"3G pass {k}: no untried posture is predicted to read the groove (predicted good fraction < "
                       f"{_ms.DESIGN_HALF:.1f} for every reachable candidate; poor postures in this run: {tried}) - repeating a "
                       "posture that already failed gives the same data")
            self._decide("posture", why + f"; candidates: {table}", candidates=cand)
            return None, why
        if not eligible:
            self._decide("posture", f"pass {k}: no candidate is predicted to read half the sections - trying the best-ranked one once "
                                    "(confirm-first: the run stops only after a measured poor main pass)")
        prev = self.main_posture
        self.main_posture = {k_: best[k_] for k_ in ("tilt", "torch_above", "phi", "incidence", "pred", "coverage", "why")}
        flip = self._is_flip(T_now, best["T_from"])
        self._decide("posture", f"pass {k}: main posture incidence {best['tilt']:+.0f} deg, torch {'above' if best['torch_above'] else 'below'} "
                                f"the laser (phi {best['phi']:+.1f} deg): predicted {best['g']:.2f} good ({best['why']}) x incidence weight "
                                f"-> {best['pred']:.2f}, laser reaches {best['coverage'] * 100:.0f} % of the length above the table"
                                + (f" (replaces phi {prev['phi']:+.1f} deg)" if prev is not None else "")
                                + (f"; TORCH FLIP: {best['rot']:.0f} deg rotation, sent as 90 deg in place about the optical axis + the "
                                   "rest (first real trial: watch it)" if flip else "")
                                + f"; candidates: {table}", posture=_jsonable(self.main_posture), candidates=cand)
        return self.main_posture, None

    def _is_flip(self, T_a: np.ndarray, T_b: np.ndarray) -> bool:
        """토치 뒤집기 = 두 자세의 토치 쪽 (레이저 위/아래: EOAT +x 가 아래면 위 — _job_torch_above 와 같은 정의) 이 다르다. 그 회전에는
        광축 둘레 180° 가 들어 있어 한 번에 보내면 보간 방향이 정해지지 않는다 → 광축 둘레 90° 제자리 회전을 먼저 (3G 만; 1G 는 뒤집지
        않는다). 회전 각도 문턱 없음."""
        if self.orientation != "vertical_3G" or self.up is None:
            return False

        def above(T):
            return float((np.asarray(T, dtype=float) @ self.calib.T_flange_eoat)[:3, 0] @ self.up) < 0.0
        return above(T_a) != above(T_b)

    def _goal_tol_deg(self) -> float:
        pol = getattr(self, "policy", None)
        return float(pol.goal_tol_deg) if pol is not None else 0.0

    def _flip_mid(self, T_now: np.ndarray, T_to: np.ndarray) -> Optional[np.ndarray]:
        """토치 뒤집기의 중간 자세: 지금 자세를 광학 원점을 지나는 광축 둘레로 ±90° 돌린 것 중 (지금 → 중간, 중간 → 목표) 이 포락선을
        지나고 남은 회전이 보간 방향이 정해지는 범위 (180° − 도착 자세 허용 미만) 인 쪽. 고르는 순서: 마지막 관절 (손목3) 을 0 쪽으로 돌리는 방향 (광축 ≈ 플랜지 z 라 이
        회전은 거의 손목3 하나 — 실기 g3_12 패스 1 끝 손목3 182°: 같은 방향으로 180° 더 돌면 한계 ±360° 를 넘는다), 그다음 툴의 가장
        낮은 점이 더 높은 쪽 (테이블에서 먼 쪽). 없으면 None."""
        from scipy.spatial.transform import Rotation
        T_now = np.asarray(T_now, dtype=float)
        To = T_now @ self.calib.T_flange_opt
        o, z = To[:3, 3], unit(To[:3, 2])
        q6 = self._last_joint_now()
        zf = float(z @ unit(T_now[:3, 2]))                        # 광축 · 플랜지 z (≈ 1: 광축 둘레 +θ ≈ 손목3 +θ)
        best = None
        for sgn in (1.0, -1.0):
            R = Rotation.from_rotvec(z * sgn * math.pi / 2.0).as_matrix()
            T_mid = make_transform(R, o - R @ o) @ T_now
            rest = _rot_angle_deg(T_mid[:3, :3], np.asarray(T_to)[:3, :3])
            if rest >= 180.0 - self._goal_tol_deg():
                continue
            try:
                for a_, b_ in ((T_now, T_mid), (T_mid, T_to)):
                    make_local_move_plan(a_, b_, "scan", self.job_ref, self.calib, specimen_top_z_m=self.plate_top,
                                         speed_mps=float(self.local_speed), accel_mps2=self.accel, name="flip_check")
            except (EnvelopeError, JobError, ValueError):
                continue
            h = float(self.up @ (T_mid @ self.calib.T_flange_opt)[:3, 3]) - lowest_below_origin(T_mid, self.calib, self.job_ref)
            toward0 = 1 if (q6 is not None and abs(zf) > math.cos(math.radians(_ms.HALF_TURN_DEG)) and sgn * np.sign(zf) * q6 < 0.0) else 0
            if best is None or (toward0, h, -rest) > best[0]:
                best = ((toward0, h, -rest), T_mid)
        return None if best is None else best[1]

    def _last_joint_now(self) -> Optional[float]:
        """마지막 관절 (손목3) 실측 각 [rad]: MoveIt 어댑터 스냅샷 joint_positions_rad ('wrist3', 없으면 마지막), RB 제어박스 어댑터 jnt_ang
        [deg] 의 6 번째. 없으면 None."""
        try:
            a = self.rs.pose.snapshot().arrays or {}
        except Exception:  # noqa: BLE001
            return None
        q = a.get("joint_positions_rad")
        if q is not None and len(q):
            names = [str(x) for x in a.get("joint_names", [])]
            q = np.asarray(q, dtype=float)
            return float(q[names.index("wrist3")] if "wrist3" in names else q[-1])
        j = a.get("jnt_ang")
        if j is not None and len(j) >= 6:
            return math.radians(float(np.asarray(j, dtype=float)[5]))
        return None

    def _rescan_job(self, line: SeamLine, PA, PB, tilt: float, pitch: float, direction: str, k: int,
                    torch_above: Optional[bool] = None) -> tuple:
        """구간 PA → PB (S 쪽 → E 쪽 base 점, S→E 계약 좌표에서 — _plan_rescan) 의 재스캔 job + 착점 범위 (오버스캔 0, 바닥·테이블 제한).
        진행 부호는 지금 자세에 가까운 쪽 (torch_above 가 주어지면 (3G) 그 토치 쪽의 부호만)."""
        depth_m = self._depth_m(self.last_full["fm"] if self.last_full else None)
        PA, PB = np.asarray(PA, dtype=float), np.asarray(PB, dtype=float)
        a, b = (PB, PA) if direction == "E->S" else (PA, PB)
        kw = job_to_dict(self.job0)
        kw["torch_lateral_m"] = tuple(kw["torch_lateral_m"])
        kw.update(start=a.tolist(), end=b.tolist(), surface_normal=unit(line.n).tolist(), orientation=self.orientation,
                  standoff_m=self.pass_plate_m + float(depth_m), pitch_deg=float(pitch), incidence_tilt_deg=float(tilt),
                  lead_in_m=0.001, lead_out_m=0.001, bidirectional=False, plate_top_measured_m=self.plate_top,
                  plate_plane_measured=self.job_ref.plate_plane_measured if self.job_ref is not None else None,
                  job_id=f"{self.job0.job_id or 'job'}_rescan{k}")
        if self.orientation == "vertical_3G":
            kw["vertical_direction"] = "down" if float(self.up @ a) >= float(self.up @ b) else "up"
            kw["floor_z_m"] = self.floor_h
        R_now = self._T_now()[:3, :3]
        best = None
        for sign in (1, -1):
            kw["progression_sign"] = sign
            try:
                j = ScanJob(**kw)
                j.validate()
            except JobError:
                continue
            if torch_above is not None and self.orientation == "vertical_3G" and self._job_torch_above(j) != bool(torch_above):
                continue
            rot = _rot_angle_deg(R_now, sensor_frames(j, self.calib).T_base_flange(0.0)[:3, :3])
            if best is None or rot < best[0]:
                best = (rot, j)
        if best is None:
            return None, None, 0.0
        try:
            job, _raised = self._clear_plate(best[1], depth_m)
        except _Abort:
            return None, None, 0.0
        ov = pass_overscan(job, self.calib, 0.0, top_offset_m=depth_m, floor_h=self.floor_h if self.orientation == "vertical_3G" else None,
                           overscan_first_m=0.0, overscan_last_m=0.0, min_overscan_m=0.0)
        L = float(np.linalg.norm(PB - PA))
        cov = max(0.0, min(float(ov["hit_to"]), job.length_m) - max(float(ov["hit_from"]), 0.0)) / L if L > 0 else 0.0
        try:                                                     # 패스 구간 포락선 미리 (거부되면 실행 중 중단이 아니라 후보 탈락)
            make_local_move_plan(ov["T_from"], ov["T_to"], "scan", self.job_ref, self.calib, specimen_top_z_m=self.plate_top,
                                 speed_mps=float(job.speed_mps), accel_mps2=self.accel, name="rescan_check")
        except (EnvelopeError, JobError, ValueError):
            return job, ov, -1.0
        return job, ov, float(cov)

    def _plan_rescan(self, k: int) -> Optional[PassPlan]:
        """병합 구간 중 점수 < GOOD_FRAC_MIN 인 연속 구간 (끝 구역 제외) → 다른 자세 후보 중 덮는 비율 ≥ 절반이고 그 φ 의 측정 근거가
        좋은 비율 ≥ 절반인 것 (이번 실행 측정 > 데이터셋 > 씨앗, scan_quality.predicted_score) 으로 그 구간만 다시 스캔하는 계획 (없으면
        None). 이번 실행에서 이미 쓴 φ (격자 반 칸) 는 다시 쓰지 않는다."""
        if not self.merge_choice or self.last_full is None:
            return None
        ch = self.merge_choice
        # 병합 구간 좌표 = 기준 패스 (마지막 전체 패스) 의 관측 끝 S, E 의 S→E 계약 (S 에서 잰 거리; merged_pass_features) — 같은 좌표로
        # base 점을 만든다 (정련 직선 c·u 로 읽으면 기준 패스가 E→S 일 때 거울상 구간을 다시 스캔했다)
        fit_ = self.last_full["fit"]
        frame = _extent_frame(self.last_full.get("extent"), fit_.n if fit_.ok else None)
        if frame is None:
            return None
        # 끝 구역 (끝 지연 + 단면 창 = EndParams.look_m) 에 걸친 구간은 재스캔하지 않는다 — 끝 판정이 그 구역의 단면 품질을 낮춘다
        # (4d P6; 예전 '양 끝 구간 하나씩')
        look = float(self._end_params(self.job0).get("window_m", 0.0)) + float(self.last_full.get("end_lag_m", 0.0) or 0.0)
        poor = [i for i, c_ in enumerate(ch) if c_["score"] < _sq.GOOD_FRAC_MIN
                and float(c_["s_from_m"]) >= look and float(c_["s_to_m"]) <= frame.length_m - look]
        if not poor:
            return None
        runs, st = [], poor[0]
        for a_, b_ in zip(poor, poor[1:] + [None]):
            if b_ is None or b_ != a_ + 1:
                runs.append((st, a_))
                st = b_
        i0, i1 = max(runs, key=lambda r_: r_[1] - r_[0])
        lo, hi = float(ch[i0]["s_from_m"]), float(ch[i1]["s_to_m"])
        cur_score = float(np.mean([ch[i]["score"] for i in range(i0, i1 + 1)]))
        line = self.last_full["line"]
        tried = [float(e["posture"].get("phi_deg", float("nan"))) for e in self.pass_quality]
        measured = self._measured_phis()
        T_now = self._T_now()
        o_now = (T_now @ self.calib.T_flange_opt)[:3, 3]
        rows, best = [], None
        for tilt, pitch, side in self._posture_candidates():
            for direction in ("S->E", "E->S"):
                job, ov, cov = self._rescan_job(line, frame.point(lo), frame.point(hi), tilt, pitch, direction, k + 1, torch_above=side)
                if job is None:
                    continue
                try:
                    post = _sq.pass_posture(job, self.calib, plate_normal=line.n)
                except Exception:  # noqa: BLE001
                    continue
                inc, phi = post["incidence_deg"], post["phi_deg"]
                if any(np.isfinite(t_) and abs(t_ - phi) < 0.5 * _sq.POSTURE_GRID_DEG for t_ in tried):
                    continue                                       # 이번 실행에서 이미 쓴 자세 (같은 φ 칸)
                pred, _g, why = _sq.predicted_score(post, str(self.res.position or ""), self.at.quality_dataset_path or None, measured)
                dist = float(np.linalg.norm(np.asarray(ov["origin_from"]) - o_now)) if ov is not None else math.inf
                rot = _rot_angle_deg(T_now[:3, :3], np.asarray(ov["T_from"])[:3, :3]) if ov is not None else float("nan")
                rows.append({"tilt": tilt, "pitch": pitch, "torch_above": post["torch_above"], "dir": direction, "incidence": inc,
                             "phi": phi, "coverage": cov, "pred": pred, "why": why, "dist": dist, "rot": rot})
                # 4d P7: 덮는 비율 ≥ 절반, 그 자세의 측정 근거 (격자 한 칸 안) 가 좋은 비율 ≥ 절반 — 나쁜 구간을 좋게 바꿀 것으로 보이는 것만
                # (예전 0.2 점수 여유 없음)
                evid = not why.startswith(("no measurement", "specular", "unknown"))
                # 1G 는 측정 근거가 없으면 지금처럼 피치를 거울로 뒤집은 자세를 쓴다 (D-53 대비책; 정반사 대역은 phi_prior 가 0)
                if cov >= _ms.DESIGN_HALF and _g >= _ms.DESIGN_HALF and (evid or self.orientation == "flat_1G"):
                    if self._is_flip(T_now, ov["T_from"]) and self._flip_mid(T_now, ov["T_from"]) is None:
                        rows[-1]["skip"] = "torch flip not possible here"
                        continue
                    key_ = (pred * min(1.0, cov), -dist)
                    if best is None or key_ > best[0]:
                        best = (key_, tilt, pitch, direction, job, ov, cov, inc, pred, post)
        self._decide("rescan", f"after pass {k}: sections {lo * 1e3:.0f}..{hi * 1e3:.0f} mm from S score {cur_score:.2f} < {_sq.GOOD_FRAC_MIN}; "
                               f"candidates " + ", ".join(f"{r_['tilt']:+.0f}/{r_['pitch']:+.0f} deg"
                                                         + ("" if r_["torch_above"] is None else
                                                            f" torch {'above' if r_['torch_above'] else 'below'}")
                                                         + f" {r_['dir']} phi {r_['phi']:+.0f} "
                                                         f"cov {r_['coverage'] * 100:.0f}% pred {r_['pred']:.2f}" for r_ in rows)
                     + ("" if best else " -> none better (no rescan)"), candidates=_jsonable(rows))
        if best is None:
            return None
        _k, tilt, pitch, direction, job, ov, cov, inc, pred, post = best
        self._rescan_prepared = (job, ov)
        self._decide("rescan", f"pass {k + 1} = quality rescan of {lo * 1e3:.0f}..{hi * 1e3:.0f} mm from S at incidence {inc:.0f} deg "
                               f"(tilt {tilt:+.0f}, pitch {pitch:+.0f}"
                               + ("" if post["torch_above"] is None else f", torch {'above' if post['torch_above'] else 'below'}")
                               + f", phi {post['phi_deg']:+.1f} deg, {direction}; covers {cov * 100:.0f} %, predicted score {pred:.2f} vs {cur_score:.2f})")
        s_line = [float((frame.point(v) - np.asarray(line.c, dtype=float)) @ unit(line.u)) for v in (lo, hi)]   # PassPlan = 직선 좌표
        return PassPlan(direction=direction, s_S=s_line[0], s_E=s_line[1], ov_S=0.0, ov_E=0.0, how_S="quality rescan",
                        how_E="quality rescan", rule="rescan_quality", tilt_deg=float(tilt), pitch_deg=float(pitch),
                        torch_above=post["torch_above"])

    def _write_quality_dataset(self) -> None:
        """패스별 (자세, 품질) 을 데이터셋 jsonl 에 덧붙인다 (다음 자세 선택의 사전값; 경로 설정 'off' = 쓰지 않음)."""
        if not self.pass_quality:
            return
        path = None
        for e in self.pass_quality:
            path = _sq.append_dataset({"run": self.out.name, "position": str(self.res.position or ""), "pass": int(e["k"]),
                                       "full": bool(e["full"]), "status": str(self.res.status), "posture": _jsonable(e["posture"]),
                                       "quality": e["fractions"], "spec": getattr(self.spec, "name", None)},
                                      self.at.quality_dataset_path or None)
        if path:
            self.res.notes.append(f"scan quality dataset: {len(self.pass_quality)} pass record(s) appended to {path}")
        for po in self.posture_offsets:                         # 3G 자세 간 오프셋 자료 (4d P8)
            _po.append_offset({"run": f"{self.out.name}/{po.get('to_pass')}", "calib_date": str(getattr(self.calib, "date", "") or ""),
                               "end_posture": po.get("end_posture"), "main_posture": po.get("posture"), "along_m": po.get("along_m"),
                               "normal_m": po.get("normal_m"), "lateral_m": po.get("lateral_m"), "angle_deg": po.get("angle_deg"),
                               "source": "real run"})

    def _note_support(self, fit: PassFit, k: int) -> None:
        """패스가 시편 끝의 지지면 (테이블) 을 쟀으면 (support_plane_ends) 그 높이를 기록하고 남은 실행의 테이블로 쓴다 (사용자 2026-10-06:
        테이블 높이 입력 없이 가는 방향). job 에 table_z_m 이 없으면 실측값, 있으면 둘 중 높은 쪽 (보수적) — 툴 몸체 규칙·패스 끝 제한."""
        sp = (fit.diag or {}).get("support_planes") or {}
        hs = [(float(v["height_m"]), w, v) for w, v in sp.items() if v.get("applied") and _fin(v.get("height_m"))]
        if not hs:
            return
        h, which, v = min(hs, key=lambda t_: t_[0])
        old = self.job0.table_z_m
        new = h if old is None else max(float(old), h)
        self.table_measured_h = h
        how = ("table_z_m set to it for the rest of the run (tool body >= table + margin)" if old is None
               else (f"job table_z_m {float(old):.4f} m kept (higher, more conservative)" if float(old) >= h
                     else f"table_z_m raised {float(old):.4f} -> {h:.4f} m"))
        self._decide("table", f"pass {k}: support surface (table) at the {which} specimen end measured by the laser at height {h:.4f} m "
                              f"(rms {float(v.get('rms_mm', float('nan'))):.2f} mm, {int(v.get('n_points', 0))} points, "
                              f"{float(v.get('angle_to_plate_deg', float('nan'))):.1f} deg to the plate); the end is where the root line "
                              f"meets it ({float(v.get('shift_m', float('nan'))) * 1e3:+.1f} mm from the floor-point end); {how}",
                     table_measured_h=h, support=_jsonable(v))
        if new != old:
            for j in (self.job0, self.job_c, self.job_ref):
                if j is not None:
                    j.table_z_m = float(new)

    def _update_3g_floor(self, line: SeamLine, ext: dict, ov: dict, k: int, direction: str = "") -> Optional[str]:
        """3G (사용자 2026-10-05): 패스가 시편 아래 끝을 관측했으면 그 끝 (루트 직선 위 끝 점) 의 높이가 바닥이 된다 — 이후 모든 툴 키포인트
        (광학 원점·토치 팁) 가 그 아래로 가지 않는다 (대략 아래 점은 교시마다 달라 처음 보정까지만 기준). 첫 패스 (아래 → 위)
        에서 아래 끝을 못 봤으면 바닥을 정할 수 없으니 멈출 사유를 돌려준다 (그 뒤 패스도 대략 아래 점 아래를 볼 수 없다)."""
        lower = "E" if float(self.up @ line.E) <= float(self.up @ line.S) else "S"
        e = dict(ext.get(lower) or {})
        if e.get("observed") and e.get("point") is not None:
            h = float(self.up @ np.asarray(e["point"], dtype=float))
            old, old_src = self.floor_h, self.floor_src
            sup = self.support or {}
            if str(sup.get("src", "")).startswith("measured") and _fin(sup.get("height_m")) and h < float(sup["height_m"]):
                h = float(sup["height_m"])                           # 아래 끝 추정이 잰 지지면 아래면 지지면이 바닥 (안전 쪽)
            if old_src == "the measured lower specimen end" and h < float(old):
                # 4d P8: 첫 측정 뒤로는 올리기만 (g3_18A: 이어받은 끝이 9 mm 내려가 바닥이 잰 지지면 아래로 내려갔다)
                self._decide("floor", f"pass {k}: lower end estimate {h:.4f} m is below the 3G floor {float(old):.4f} m - floor kept "
                                      "(raise-only after the first measurement)", floor_h=float(old), estimate=h)
                return None
            self.floor_h, self.floor_src = h, "the measured lower specimen end"
            self.job_ref.floor_z_m = h
            self._decide("floor", f"pass {k}: 3G floor = measured lower specimen end ({lower}) height {h:.4f} m (was {old:.4f} m, "
                                  f"{old_src}; {(h - old) * 1e3:+.1f} mm) - the optical origin and the torch tip stay above it from now on",
                         floor_h=h, previous=old, lower_end=lower)
            return None
        if self.floor_src == "the measured lower specimen end":
            return None                                          # 앞 패스가 잰 아래 끝을 계속 쓴다
        hit_low = min(float(self.up @ np.asarray(ov["hit_point_from"], dtype=float)),
                      float(self.up @ np.asarray(ov["hit_point_to"], dtype=float)))
        reach_mm = (float(self.floor_h) - hit_low) * 1e3
        why = "an object right at it hides it" if e.get("occluded") else "the plate continued to the lowest laser hit"
        return (f"3G: pass {k} ({direction or 'end pass'}) did not see the lower specimen end ({why}): the laser reached {reach_mm:.0f} mm below "
                f"the lower rough point, the floor until the end is measured - teach the lower rough point closer to the lower plate end "
                f"(within ~{max(reach_mm - self.overscan_m * 1e3, 0.0):.0f} mm above it) and run again")

    def _floor_unmeasured(self) -> bool:
        """3G 이고 아래 시편 끝을 아직 재지 않았다 (바닥 = 대략 아래 점)."""
        return self.orientation == "vertical_3G" and self.floor_src != "the measured lower specimen end"

    def _try_end_repeat(self, k: int, why: str) -> bool:
        """4d P9 F-16/F-17: 3G 끝 패스가 자료 쪽 이유 (직선 없음·재구성 실패·기대 끝을 지나고도 끝 없음) 로 아래 끝을 못 쟀으면 끝 패스를
        한 번만 반대 방향으로 반복한다 (같은 끝 자세·대략 바닥 규칙·패스 1 규칙) — 그 뒤 주 패스 하나를 남길 예산 (n_pass + 2 ≤
        max_passes) 이 있을 때만. 반복도 실패하면 멈춘다 (확인 먼저). 반환: 반복을 계획했나."""
        if not self._floor_unmeasured() or self._end_repeat_used or self.n_pass + 2 > self.max_passes:
            return False
        self._end_repeat_used = True
        self._end_repeat_pending = True
        self._decide("gate", f"pass {k}: {why} - the end pass is repeated once in the opposite direction (same end posture and floor "
                             "rule; the run stops only if the repeat also fails)")
        return True

    def _end_repeat_why_not(self) -> str:
        """끝 패스 반복이 없는 이유 (멈춤 사유 문구)."""
        if self._end_repeat_used:
            return "the repeat end pass also failed (stopped)"
        return (f"no pass budget for a repeat end pass and a main pass (max_passes {self.max_passes}, used {self.n_pass}) "
                "(stopped, unconfirmed)")

    def _lower_end_expected_passed(self, line: SeamLine, ext: dict, ov: dict) -> Optional[str]:
        """3G 아래 끝이 안 보인 패스에서 레이저 착점이 기대 아래 끝 (관측된 위 끝에서 설계 길이 아래) 을 끝 검출 최소 오버스캔
        (overscan_min) 이상 지나갔나 — 지나갔으면 그 설명 (자료 쪽 실패), 아니면 None (도달 쪽: 바닥 때문에 거기까지 못 갔다, 또는 위 끝·설계
        길이가 없어 기대 끝을 모른다)."""
        lower = "E" if float(self.up @ line.E) <= float(self.up @ line.S) else "S"
        upper = "S" if lower == "E" else "E"
        eu = ext.get(upper) or {}
        if not (eu.get("observed") and eu.get("point") is not None) or eu.get("inferred"):
            return None
        if self.spec is None or not (self.spec.weld_length_mm > 0.0):
            return None
        u_dn = unit(np.asarray(getattr(line, lower), dtype=float) - np.asarray(getattr(line, upper), dtype=float))
        P_up = np.asarray(eu["point"], dtype=float)
        L = float(self.spec.weld_length_mm) * 1e-3
        s_hit = max(float((np.asarray(ov[k_], dtype=float) - P_up) @ u_dn) for k_ in ("hit_point_from", "hit_point_to"))
        beyond = s_hit - L
        if beyond >= float(self.overscan_min_m):
            return (f"the laser hit went {beyond * 1e3:.1f} mm past the expected lower end (observed upper end - design length "
                    f"{L * 1e3:.0f} mm; end detection needs {self.overscan_min_m * 1e3:.1f} mm)")
        return None

    def _overscan_raw(self, job: ScanJob, plan: PassPlan, depth_m: float) -> dict:
        """path_gen.pass_overscan + 패스 시작 로컬 이동의 바닥 여유 (결정 기록·거부 없음 — _pass_overscan, 자세 후보 평가).
        depth_m: 그 작업을 만든 그루브 깊이 (착점 = 루트 + n·깊이 의 판 상면)."""
        floor_h = self.floor_h if self.orientation == "vertical_3G" else None
        depth = float(depth_m)
        ov_first, ov_last = (plan.ov_E, plan.ov_S) if plan.direction == "E->S" else (plan.ov_S, plan.ov_E)
        extra = 0.0
        if floor_h is not None or job.table_z_m is not None:
            # 패스 시작까지의 로컬 이동 (_move_to_pass_start) 은 지금 판 거리에서 면내로 간 뒤 판 법선으로 다가간다: 판이 로봇 쪽으로 기울면
            # (up·n < 0) 먼 판 거리의 면내 구간이 시작점보다 (판 거리 차)·|up·n| 낮다 — 그만큼 첫 끝의 바닥 (3G 바닥 또는 테이블) 을 올린다
            # (2026-10-06: 3G 에서 판 쪽으로 다가갈 때는 법선 구간이 먼저라 (_move_to_pass_start) 면내 구간이 패스 거리 — 여유 0)
            try:
                n_ = job.normal_unit
                d_now = float((self._T_opt_now()[:3, 3] - job.start) @ n_)
                d_pass = float((sensor_frames(job, self.calib).opt_origin(0.0) - job.start) @ n_)
                extra = max(0.0, -float(self.up @ n_) * (d_now - d_pass)) if d_now < d_pass else 0.0
            except Exception:  # noqa: BLE001 - 자세 없음 (dry run 등): 여유 없이 (포락선 검사는 그대로)
                extra = 0.0
        ov = pass_overscan(job, self.calib, self.overscan_m, top_offset_m=depth, floor_h=floor_h,
                           overscan_first_m=ov_first, overscan_last_m=ov_last,
                           min_overscan_m=self.overscan_min_m, floor_first_extra_m=extra)
        return ov

    def _posture_allowance_m(self) -> tuple:
        """3G 주 패스 (끝 패스와 다른 자세) 의 끝 오버스캔에 더하는 자세 간 오프셋 몫 [m] 와 '이번 실행에서 잰 값인가': 같은 자세로 이미
        잰 심 방향 오프셋이 있으면 그 최대 + k·끝 σ (자료 한계 B_along 이하), 없으면 B_along — 받아들이는 오프셋과 같은 수라 받아들인
        오프셋은 모두 계획에 들어 있다 (4d P8; 예전 25 mm 고정)."""
        mp = self.main_posture or {}
        if self._same_posture_as_end():
            return 0.0, True                                     # 끝 패스와 같은 자세: 오프셋 없음
        b_along, _b = self._along_bound()
        same = [po for po in self.posture_offsets if self._same_posture(po.get("posture"), mp)]
        if not same:
            return float(b_along), False
        sig = float((self.lower_end_obs or {}).get("sigma_m", float("nan")))
        a = max(abs(float(po["along_m"])) for po in same) + float(self.settings.judgement_k) * (sig if _fin(sig) else 0.0)
        return float(min(a, b_along)), True

    def _with_standoff(self, job: ScanJob, standoff_m: float) -> ScanJob:
        out = ScanJob.from_dict({**job_to_dict(job), "torch_lateral_m": tuple(job.torch_lateral_m), "standoff_m": float(standoff_m)})
        out.plate_top_measured_m = job.plate_top_measured_m
        out.plate_plane_measured = job.plate_plane_measured
        out.floor_z_m = job.floor_z_m
        return out

    def _raise_for_ends(self, job: ScanJob, plan: PassPlan, depth_m: float, ov: dict) -> tuple:
        """끝이 바닥·지지면에 막혀 계획 착점 오버스캔에 못 미치면 패스 거리 (호 반지름) 를 늘려 본다 — 내려다보는 자세에서는 같은 툴
        높이에서 착점이 더 내려간다 (사용자 2026-10-07: 그루브는 광축 교차 근처, 단 툴 한계·끝 관측이 허락하는 범위에서). 1 mm 간격
        (수치 해상도), 루트가 센서 창 (z_far − e) 안에 남는 데까지: 계획에 닿는 가장 작은 거리, 없으면 끝 판정 최소에 닿는 가장 작은
        거리, 그것도 없으면 그대로. → (작업, ov, 늘린 양 m)."""
        lim = list(ov.get("floor_limited") or [])
        if not lim:
            return job, ov, 0.0
        sv = self._sv()
        s_max = (sv.z_far_m - sv.pose_tol_m) * math.cos(math.radians(float(job.pitch_deg)))
        s0 = float(job.standoff_m)

        def reaches(o, need: bool) -> bool:
            return all(float(o[f"overscan_{w}_m"]) >= (min(float(o[f"planned_{w}_m"]), self.overscan_min_m) if need
                                                        else float(o[f"planned_{w}_m"])) - 1e-9 for w in lim)
        fallback = None
        for i in range(1, int((s_max - s0) / 1e-3) + 1):
            try:
                j2 = self._with_standoff(job, s0 + i * 1e-3)
                o2 = self._overscan_raw(j2, plan, depth_m)
            except (JobError, ValueError):
                break
            if reaches(o2, False):
                return j2, o2, i * 1e-3
            if fallback is None and reaches(o2, True):
                fallback = (j2, o2, i * 1e-3)
        return fallback if fallback is not None else (job, ov, 0.0)

    def _pass_overscan(self, job: ScanJob, plan: PassPlan, k: int, depth_m: float, main: bool = False) -> tuple:
        """착점 오버스캔 (path_gen.pass_overscan): 광학 원점 시작/끝 — 계획의 끝마다 (판 안쪽 목표는 0) → (작업, ov). 끝이 바닥·지지면에
        막히면 패스 거리를 늘려 끝을 보게 해 본다 (_raise_for_ends; 아래 끝을 이어받는 주 패스는 제외). 3G: 그래도 아래 끝이 끝 판정
        최소에 못 미치면 REFUSED 'lower end not reachable under the 3G floor rule' (첫 패스는 갈 수 있는 데까지 가고 패스 뒤에 본다)."""
        ov = self._overscan_raw(job, plan, depth_m)
        if ov["floor_limited"] and not (main and self.lower_end_obs is not None):
            job2, ov2, raised = self._raise_for_ends(job, plan, depth_m, ov)
            if raised > 0.0:
                self._decide("overscan", f"pass {k}: {ov['floor_limited']} end held back by the floor/support (hit overscan "
                                         f"{min(ov['overscan_first_m'], ov['overscan_last_m']) * 1e3:+.1f} mm) - pass distance "
                                         f"{job.standoff_m * 1e3:.0f} -> {job2.standoff_m * 1e3:.0f} mm so the laser hit reaches it "
                                         "(looking down, a longer arc puts the hit lower at the same tool height)")
                job, ov = job2, ov2
        srcs = sorted(set((ov.get("floor_source") or {}).values()))
        what = " and ".join(("the table" if s_ == "table" else f"the 3G floor rule ({self.floor_src})") for s_ in srcs) or "the floor"
        info = {k_: ov[k_] for k_ in ("hit_offset_m", "overscan_m", "overscan_first_m", "overscan_last_m", "floor_limited",
                                      "floor_blocked", "floor_h", "s_from", "s_to", "hit_from", "hit_to", "origin_from",
                                      "origin_to", "hit_point_from", "hit_point_to")}
        self._decide("overscan", f"pass {k}: laser hit offset {ov['hit_offset_m'] * 1e3:+.1f} mm from the optical origin's seam point "
                                 f"(pitch {job.pitch_deg:+.1f} deg, incidence {job.incidence_tilt_deg:+.1f} deg, progression "
                                 f"{job.progression_sign:+d}); hit overscan {ov['overscan_first_m'] * 1e3:.1f} mm before the first end, "
                                 f"{ov['overscan_last_m'] * 1e3:.1f} mm beyond the last end"
                     + (f"; {what} limits {ov['floor_limited']}" if ov["floor_limited"] else "")
                     + (f" (table floor {ov['table_floor_h']:.4f} m, lowest tool point {ov['table_below_origin_m'] * 1e3:.0f} mm below "
                        "the optical origin)" if "table" in srcs else ""),
                     overscan=_jsonable({**info, "floor_source": ov.get("floor_source"), "table_floor_h": ov.get("table_floor_h")}))
        lower_carried = bool(main and self.lower_end_obs is not None)
        if ov["floor_blocked"] and lower_carried:
            self._decide("overscan", f"pass {k} (main incidence {job.incidence_tilt_deg:+.0f} deg): {what} keeps the laser hit "
                                     f"{-min(ov['overscan_first_m'], ov['overscan_last_m']) * 1e3:.1f} mm inside the lower end - expected; "
                                     f"the lower end is carried from pass {self.lower_end_obs['pass']} "
                                     f"({'table support' if self.lower_end_obs.get('support') else 'observed'})")
        elif ov["floor_blocked"] and self.orientation == "vertical_3G" and self.floor_src != "the measured lower specimen end":
            # 아래 끝을 아직 재지 않았다 (패스 1): 아래 끝은 바닥 (대략 아래 점) 이 허락하는 데까지만 — 거기서 아래 끝이 보이는지는 패스
            # 뒤에 본다 (_update_3g_floor). 4d P9: 패스 번호가 아니라 그 상태로
            self._decide("overscan", f"pass {k}: {what} keeps the laser hit at the lower end "
                                     f"to {min(ov['overscan_first_m'], ov['overscan_last_m']) * 1e3:+.1f} mm beyond the planned target - "
                                     "the pass goes as low as the floor allows; the lower specimen end must show up in it")
        elif ov["floor_blocked"]:
            raise _Abort(STATUS_REFUSED, f"pass {k}: lower end not reachable under {what} (the tool may not go below it: "
                                         f"laser hit overscan only "
                                         f"{min(ov['overscan_first_m'], ov['overscan_last_m']) * 1e3:.1f} mm < "
                                         f"{self.overscan_min_m * 1e3:.1f} mm the end detection needs at the {ov['floor_blocked']} "
                                         "end) - teach the lower "
                                         "rough point closer to the lower plate end (only slightly inside it) (nothing sent)")
        return job, ov

    def _move_to_pass_start(self, T_start: np.ndarray, job: ScanJob, k: int) -> None:
        """현재 자세 → 패스 시작 (짧은 로컬 이동, S 로 되돌아가지 않는다): 면내·재자세 성분이 있으면 현재 판 거리에서 'scan' 구간 뒤
        순수 법선 구간 (1G descend/ascend, 3G approach_normal/retreat_normal) — M-2 태그 규칙. 포락선 거부는 'end not reachable'.

        3G 에서 판 쪽으로 다가갈 때 (패스 거리가 지금보다 가까움) 는 순서를 바꾼다: 지금 자세 그대로 판 법선 구간 먼저, 그다음 패스
        거리에서 면내·재자세 'scan' 구간. 광축이 내려다보므로 (입사 기울기 + 피치) 판에서 dz 만큼 멀리 있으면 레이저 착점이 dz·tan 만큼
        더 아래다 — 2026-10-06 g3_05 (35° + 5°, 판 거리 0.40 → 0.29 m): 면내 구간에서 착점이 패스 시작보다 ~95 mm 아래 (시편 아래 끝
        너머) 로 가 앞쪽 받침면을 보고 G4 'object on the plate' 로 정지했다. 바꾼 순서에서는 착점이 패스 시작 착점보다 아래로 가지 않고,
        툴 높이도 법선 구간은 지금 높이 그대로, 면내 구간은 패스 시작과 같은 판 거리라 패스 시작 높이 아래로 가지 않는다.

        토치 뒤집기 (토치 쪽이 바뀐다 — _is_flip, 3G 적응형 자세): 먼저 지금 위치에서 광축 둘레 90° 제자리 회전 (_flip_mid: 포락선을 지나고 툴의
        가장 낮은 점이 더 높은 쪽), 그다음 위 규칙으로 나머지 — 180° 를 한 번에 보내면 보간 방향이 정해지지 않는다."""
        T_now = self._T_now()
        rot0 = _rot_angle_deg(T_now[:3, :3], np.asarray(T_start)[:3, :3])
        if self._is_flip(T_now, T_start):
            T_mid = self._flip_mid(T_now, np.asarray(T_start, dtype=float))
            if T_mid is None:
                raise _Abort(STATUS_REFUSED, f"pass {k}: the pass start needs a {rot0:.0f} deg re-orientation (torch flip) and no in-place "
                                             "90 deg rotation about the optical axis passes the envelope from here (nothing sent)")
            self._local_move(T_mid, "scan", f"pass{k}_flip",
                             extra=f"(torch flip, step 1 of 2: 90 deg in place about the optical axis of the {rot0:.0f} deg re-orientation; "
                                   "the rest follows in the move to the pass start)")
            T_now = self._T_now()
        n = job.normal_unit
        o_now = (T_now @ self.calib.T_flange_opt)[:3, 3]
        o_d = (np.asarray(T_start) @ self.calib.T_flange_opt)[:3, 3]
        dz = float((o_d - o_now) @ n)
        d_ip = (o_d - o_now) - n * dz
        rot = _rot_angle_deg(T_now[:3, :3], np.asarray(T_start)[:3, :3])
        normal_phase = self._leg_phase(dz, 0.0, orientation=self.orientation)
        T_start = np.asarray(T_start, dtype=float)
        e = self._sv().pose_tol_m                                    # 도착 허용 안의 차이는 도착 오차 (따로 보내지 않는다)
        if float(np.linalg.norm(d_ip)) > e or rot > 1e-3:
            if abs(dz) > e:
                T_nf = np.array(T_now, dtype=float, copy=True)
                T_nf[:3, 3] += n * dz                               # 법선 먼저: 지금 자세·면내 위치 그대로 패스 판 거리까지
                T_if = np.array(T_start, dtype=float, copy=True)
                T_if[:3, 3] += n * (-dz)                            # 면내·재자세 먼저: 현재 판 거리에서 패스 자세로
                normal_first = [(T_nf, normal_phase, f"pass{k}_start_normal",
                                 f"(pure normal leg {dz * 1e3:+.1f} mm first, so the laser hit never goes below the pass-{k} start)"),
                                (T_start, "scan", f"pass{k}_start",
                                 f"(in-plane/re-orientation leg at the pass distance to the pass-{k} start, {rot:.1f} deg)")]
                inplane_first = [(T_if, "scan", f"pass{k}_start_inplane",
                                  f"(in-plane/re-orientation leg to above the pass-{k} start, {rot:.1f} deg)"),
                                 (T_start, normal_phase, f"pass{k}_start", f"(pure normal leg {dz * 1e3:+.1f} mm to the pass-{k} start)")]
                # 3G 에서 다가갈 때는 법선 먼저가 좋다 (착점이 패스 시작보다 아래로 가지 않는다, g3_05) — 그 중간 자세가 포락선 (판면 한계)
                # 을 못 지나면 다른 순서. 둘 다 미리 검사하고 통과하는 첫 순서로 움직인다 (둘 다 아니면 첫 순서로 시도해 거부 사유를 남긴다).
                orders = [normal_first, inplane_first] if (self.orientation == "vertical_3G" and dz < 0.0) else [inplane_first, normal_first]
                legs = next((o for o in orders if self._legs_pass_envelope(T_now, o)), None)
                if legs is None:
                    self._decide("pass", f"pass {k}: neither leg order to the start passes the envelope: "
                                         + "; ".join(self._legs_pass_envelope(T_now, o, why=True) for o in orders))
                    legs = orders[0]
                elif legs is not orders[0]:
                    self._decide("pass", f"pass {k}: the preferred leg order to the start does not pass the envelope "
                                         f"({self._legs_pass_envelope(T_now, orders[0], why=True)}) - using the other order")
            else:
                legs = [(T_start, "scan", f"pass{k}_start", f"(to the pass-{k} start)")]
        else:
            legs = [(T_start, normal_phase, f"pass{k}_start", f"(to the pass-{k} start)")]
        try:
            for T_to, ph, nm, ex in legs:
                self._local_move(T_to, ph, nm, extra=ex)
        except _Abort as e:
            if e.status == STATUS_REFUSED and "rejected by the envelope" in e.reason:
                raise _Abort(STATUS_REFUSED, f"pass {k}: start of the pass (laser hit {self.overscan_m * 1e3:.0f} mm beyond the "
                                             f"end) not reachable: {e.reason}")
            raise

    def _legs_pass_envelope(self, T_from: np.ndarray, legs: list, why: bool = False):
        """로컬 이동 구간 목록 [(T_to, phase, name, extra)] 이 지금 자세부터 차례로 포락선을 지나는가 (_local_move 와 같은 검사, 이동 없음).
        why=True 면 첫 거부 사유 문자열 (지나면 '')."""
        T_a = np.asarray(T_from, dtype=float)
        for T_to, ph, nm, _ex in legs:
            try:
                make_local_move_plan(T_a, T_to, ph, self.job_ref, self.calib, specimen_top_z_m=self.plate_top,
                                     speed_mps=float(self.local_speed), accel_mps2=self.accel, name=nm)
            except (EnvelopeError, JobError, ValueError) as e:
                return f"{nm}: {e}" if why else False
            T_a = np.asarray(T_to, dtype=float)
        return "" if why else True

    # ---- 스트리밍 구간 (S→E 이동, 패스) ------------------------------------------------------------------------
    def _streamed_leg(self, name: str, T_to: np.ndarray, job_rec: ScanJob, subdir: str, *, speed: float, extra: str = "",
                      recon_required: bool = True, fit_kwargs: Optional[dict] = None, reconstruct: bool = True,
                      kind: Optional[str] = None, track: Optional[dict] = None) -> tuple:
        """현재 자세 → T_to 한 번의 스트리밍 move_l ('scan' 구간, 포락선·확인·가드) → (ScanRunResult(저장됨), ScanResult, dir).
        job_rec: 재구성 s 좌표·순항 판정용 작업 (패스는 패스 job). 중단 시 부분 데이터를 subdir/raw.npz 로 저장.
        reconstruct False: 원자료만 저장 (ScanResult None — S→E 이동).
        kind: 확인 종류 (None = 'scan'). track (P2 S_track·트랜싯): {fit, depth_scale, part, prior} — 녹화 중 FeatureFeeder 가 프레임 특징을
        작업 프로세스에서 계산하고, 이동 뒤 compute_track (원자료 저장 + 재구성 + 행 + fit_track) → 두 번째 반환이 그 결과 dict."""
        out_dir = self.out / subdir
        pose = self.rs.pose
        feeder = None
        if track is not None:
            try:
                f_spec, f_kw = self.spec, {"qc_targets": self.qc_targets}
                feeder = _post_scan.FeatureFeeder(f_spec, f_kw, z_valid_mm=self.settings.z_valid_mm,
                                                  workers=self.at.feature_workers_effective, fps=job_rec.profile_fps,
                                                  depth_scale=float(track.get("depth_scale", 1.0)))
            except Exception as e:  # noqa: BLE001 - the stream is an optimisation only
                self.res.notes.append(f"{name}: track feature stream not prepared ({type(e).__name__}: {e})")
                feeder = None
            if not self.at.post_scan_worker:
                feeder = None
        elif self.at.post_scan_worker and fit_kwargs is not None and reconstruct:
            try:
                f_spec, f_kw = pass_feature_inputs(fit_kwargs)
                feeder = _post_scan.SeedFeeder(f_spec, f_kw, z_valid_mm=self.settings.z_valid_mm,
                                               workers=fit_kwargs.get("feature_workers", 1), fps=job_rec.profile_fps)
            except Exception as e:  # noqa: BLE001 - the stream is an optimisation only
                self.res.notes.append(f"{name}: plate-seed stream not prepared ({type(e).__name__}: {e})")
                feeder = None
        self._seed_feeder = feeder if track is None else None
        moved = False
        try:
            res, records, samples = self._local_move(T_to, "scan", name, speed=float(speed), streaming=True, extra=extra, kind=kind,
                                                     feeder=feeder)
            moved = True
        except _Abort as e:
            if feeder is not None:
                feeder.cancel()                                 # 중단된 패스: 작업 프로세스의 스트림도 닫는다 (부분 저장 전에)
            if self._last_stream is not None:
                recs_p, samp_p, res_p = self._last_stream
                runp = ScanRunResult(status=e.status, reason=e.reason, out_dir=str(out_dir), records=recs_p, robot_samples=samp_p,
                                     motions=[(name, res_p)] if res_p is not None else [], scan_motion=res_p,
                                     sensor_info=self.ss.info.as_dict(), safety_init=self.rs.backend.safety_init_result)
                try:
                    save_raw(out_dir, runp, job_rec, self.calib, self.settings, self.cfg, pose)
                    self.res.files.append(runp.raw_path)
                    self._decide("pass", f"{name}: aborted ({e.status}); partial data saved: {len(recs_p)} frames -> {runp.raw_path}")
                except Exception as ex:  # noqa: BLE001
                    self.res.notes.append(f"partial save failed: {ex}")
            if e.status == STATUS_REFUSED and "rejected by the envelope" in e.reason and subdir.startswith("pass_"):
                raise _Abort(STATUS_REFUSED, f"{name}: end not reachable: {e.reason}")
            raise
        finally:
            self._seed_feeder = None
            if feeder is not None and not moved:
                feeder.cancel()
        run = ScanRunResult(status=STATUS_OK, reason=f"streamed leg {name}", out_dir=str(out_dir), records=records,
                            robot_samples=samples, motions=[(name, res)], scan_motion=res, sensor_info=self.ss.info.as_dict(),
                            safety_init=self.rs.backend.safety_init_result)
        run.t_sensor_start_ns = self.ss.t_start_ns
        run.t_sensor_stop_ns = self.ss.t_stop_ns
        tim: dict = {}
        if feeder is not None:
            tim.update(feeder.stop())                           # 녹화 끝: 더 보내지 않는다 (남은 프레임은 패스 뒤 처리가 처음부터)
        t_stop = getattr(res, "t_stop_ns", None)
        if t_stop:
            tim["motion_stop_to_data_s"] = (self._clock.monotonic_ns() - int(t_stop)) * 1e-9   # 이동 끝 → 녹화 정지·데이터 받음
        if track is not None:
            return run, self._track_post(name, run, job_rec, out_dir, track, feeder, tim), out_dir
        self._leg_fit = None
        self._leg_cloud = None                                # 처리된 패스 점구름 (시각화, 보는 쪽이 있을 때만)
        if self.at.post_scan_worker:
            # 원자료 저장·재구성·무결성 사실·재구성 저장 (+ 패스면 fit_pass) 를 전용 작업 프로세스에서 (post_scan; 실패하면 서버 안에서
            # 같은 계산). reconstruct False (S→E 이동) 면 원자료만
            t_ = time.perf_counter()
            arrays, js = raw_payload(run, job_rec, self.calib, self.settings, self.cfg, pose)
            sinfo = run.sensor_info or {}
            payload = dict(out_dir=str(out_dir), arrays=arrays, js=js, frames=frames_from_records(run.records),
                           robot=pose.robot_samples(run.robot_samples), job=job_rec, calib=self.calib,
                           z_valid_mm=self.settings.z_valid_mm,
                           auto_trigger_error=int(sinfo.get("params_after", {}).get("AutoTriggerError", 0) or 0),
                           extra_meta={"status": run.status, "reason": run.reason}, fit_kwargs=fit_kwargs,
                           return_result=fit_kwargs is None and reconstruct, reconstruct=bool(reconstruct),
                           stream_token=feeder.token if (feeder is not None and feeder.started) else None,
                           cloud_step_m=self._vis_step_m() if fit_kwargs is not None else None)
            tim["pack_s"] = time.perf_counter() - t_
            out = _post_scan.run_post_scan(payload, use_worker=True, log=self._p, cancel=self.cancel)
            run.raw_path, run.recon_dir, run.qc_path, run.qc = out["raw_path"], out["recon_dir"], out["qc_path"], out["qc"]
            qc = out["qc"]
            tim.update(out["timing"])
            tim["post_scan_where"] = out["where"]
            for k_ in ("post_scan_pool_note", "fit_features_pool_note"):  # 프로세스 풀이 수명 규칙 없이 돌았다 (scan_core.proc_pool) — 실행 노트에 한 번
                if tim.get(k_) and tim[k_] not in self.res.notes:
                    self.res.notes.append(tim[k_])
            if out.get("worker_error"):
                tim["post_scan_worker_error"] = out["worker_error"]
                self.res.notes.append(f"{name}: post-scan worker failed ({out['worker_error']}) - computed in the server process")
            self._leg_fit = out["fit"]
            if out.get("cloud_pending"):
                # 처리된 패스 점구름: 작업 프로세스의 다음 일 (미룬 쓰기보다 먼저) — 서버의 판정·다음 이동과 나란히 (패스 경로 밖)
                self._leg_cloud = _post_scan.submit_vis_cloud()
            elif out.get("vis_src") is not None:              # 서버 안에서 계산한 경우: 판정 뒤 스레드에서 (_emit_pass_cloud)
                self._leg_cloud = (out["vis_src"], float(payload["cloud_step_m"]))
            if out.get("deferred_save"):
                fut = _post_scan.submit_flush()                # 압축 파일 쓰기는 다음 패스 스캔 동안 (실행 끝 전에 기다린다)
                if fut is not None:
                    self._pending_saves.append((name, fut))
                else:
                    self.res.notes.append(f"{name}: deferred raw/recon save could not be submitted - files may be missing")
            if out.get("result") is not None:
                result = out["result"]
            else:
                result = _PassResultRef(out["result_meta"]) if out["recon_ok"] else None
        else:
            t_ = time.perf_counter()
            save_raw(out_dir, run, job_rec, self.calib, self.settings, self.cfg, pose)
            tim["save_raw_s"] = time.perf_counter() - t_
            t_ = time.perf_counter()
            if reconstruct:
                result, qc = reconstruct_and_qc(run, job_rec, self.calib, pose, self.settings.z_valid_mm, out_dir=out_dir)
            else:
                result, qc = None, QCReport(notes=["raw stream only (not reconstructed)"])
                run.qc = qc
            tim["reconstruct_and_qc_s"] = time.perf_counter() - t_
            tim.update({f"recon_{k_}": v_ for k_, v_ in run.timing.items()})
            tim["post_scan_where"] = "in_process"
            step = self._vis_step_m() if fit_kwargs is not None else None
            if step and result is not None:
                self._leg_cloud = (result, step)                 # 처리된 패스 점구름은 판정 뒤 스레드에서 (_emit_pass_cloud)
        self._leg_timing = tim
        if fit_kwargs is None:                                  # 패스가 아닌 구간 (S→E 이동): 여기서 시간 기록
            self._decide("timing", f"{name} post-scan: " + ", ".join(f"{k_[:-2]} {v_:.2f}" for k_, v_ in tim.items()
                                                                   if k_.endswith("_s") and isinstance(v_, (int, float)))
                         + f" [{tim.get('post_scan_where')}]", timing=_jsonable(tim))
        self.res.files.extend([run.raw_path] + ([run.recon_dir] if run.recon_dir else []) + ([run.qc_path] if run.qc_path else []))
        self._decide("pass", f"{name}: {len(records)} frames, {len(samples)} robot samples", integrity=_jsonable(qc.metrics),
                     recon_failures=qc.failures)
        if result is None and recon_required:
            raise _Abort(STATUS_DATA_FAIL, f"{name}: reconstruction failed: {qc.failures}")
        return run, result, out_dir

    def _track_post(self, name: str, run, job_rec: ScanJob, out_dir: Path, track: dict, feeder, tim: dict) -> dict:
        """S_track·트랜싯 뒤: 원자료 (지연 저장) + 재구성 + 프레임 특징 (스트림 또는 같은 보폭) + 심 맞춤 (fit_track) 을 작업 프로세스에서
        (post_scan.run_track; 실패하면 서버 안). → {rows, est, outcome, info, timing, where, payload}."""
        pose = self.rs.pose
        t_ = time.perf_counter()
        arrays, js = raw_payload(run, job_rec, self.calib, self.settings, self.cfg, pose)
        payload = dict(out_dir=str(out_dir), arrays=arrays, js=js, frames=frames_from_records(run.records),
                       robot=pose.robot_samples(run.robot_samples), calib=self.calib, z_valid_mm=self.settings.z_valid_mm,
                       fps=float(job_rec.profile_fps), spec=self.spec, kw={"qc_targets": self.qc_targets},
                       depth_scale=float(track.get("depth_scale", 1.0)),
                       stream_token=feeder.token if (feeder is not None and feeder.started) else None,
                       stride=int(feeder.stride) if (feeder is not None and feeder.started) else 1,
                       workers=self.at.feature_workers_effective,
                       still_tol_m=float(self._origin_tol_m()), part=int(track.get("part", 0)), fit=track.get("fit"),
                       prior=track.get("prior"))
        tim["pack_s"] = time.perf_counter() - t_
        try:
            out = _post_scan.run_track(payload, use_worker=bool(self.at.post_scan_worker), log=self._p, cancel=self.cancel)
        except _post_scan.Cancelled:
            raise                                               # 실행 취소 (S4): 추적 실패가 아니다
        except Exception as e:  # noqa: BLE001 - a failed track is a missing estimate (fallback path), never a lost run
            self.res.notes.append(f"{name}: track computation failed ({type(e).__name__}: {e})")
            return {"rows": None, "est": None, "outcome": "failed", "info": {"why": f"{type(e).__name__}: {e}"[:300]}, "timing": tim,
                    "where": "failed"}
        tim.update(out.get("timing") or {})
        for k_ in ("track_pool_note", "track_features_pool_note", "track_stream_pool_note"):   # 풀이 수명 규칙 일부 없이 돌았다 — 노트에 한 번
            if tim.get(k_) and tim[k_] not in self.res.notes:
                self.res.notes.append(tim[k_])
        if out.get("worker_error"):
            tim["track_worker_error"], tim["track_worker_state"] = out["worker_error"], out.get("worker_state")
            self.res.notes.append(f"{name}: track worker {out.get('worker_state')} ({out['worker_error']}) - computed in the server process")
        if out.get("deferred_save"):
            fut = _post_scan.submit_flush()
            if fut is not None:
                self._pending_saves.append((name, fut))
            else:
                self.res.notes.append(f"{name}: deferred raw save could not be submitted - the file may be missing")
        self.res.files.append(str(out.get("raw_path")))
        self._decide("timing", f"{name} track: " + ", ".join(f"{k_[:-2]} {v_:.2f}" for k_, v_ in tim.items()
                                                             if k_.endswith("_s") and isinstance(v_, (int, float)) and _fin(v_))
                     + f" [{out.get('where')}]", timing=_jsonable(tim))
        self._decide("pass", f"{name}: {len(run.records)} frames, {len(run.robot_samples)} robot samples (raw saved, groove track fitted)")
        out["timing"] = tim
        return out

    def _fit_kwargs(self, job: ScanJob) -> dict:
        """fit_pass 인자 (재구성 결과 제외) — 서버 안 계산과 작업 프로세스 (post_scan) 가 같은 값을 쓴다."""
        fr = sensor_frames(job, self.calib)
        return dict(spec=self.spec, u_prior=job.unit_dir, n_prior=job.normal_unit, stage="fine", u_travel=fr.u_travel,
                    target_standoff_m=self.pass_plate_m, gravity_axis_base=self.job0.gravity_axis_base,
                    end_params=self._end_params(job), process=self._process(),
                    feature_workers=self.at.feature_workers_effective, qc_targets=self.qc_targets)

    # ---- 8. 패스 정련 · 게이트 ------------------------------------------------------------------------------
    def _refine_pass(self, result: ScanResult, job: ScanJob, k: int, direction: str, pass_dir: Path, ov: dict, line_prev: SeamLine,
                     speed_mps: float, main: bool = False) -> dict:
        """패스 k → 특징·정련 직선·착점 양끝 (S/E) → fidelity_gate('fidelity'). 반환 {sf, fit, P_root, P_top, residuals, verdict,
        extent, line (새 SeamLine, 끝은 관측값 또는 계획 규칙 전)}."""
        t_refine0 = time.perf_counter()
        timing: dict = dict(getattr(self, "_leg_timing", None) or {})
        self._leg_timing = None
        pre = getattr(self, "_leg_fit", None)
        self._leg_fit = None
        if pre is not None:                                  # 작업 프로세스가 이미 계산 (post_scan)
            sf, fit, P_root, P_top, rp = pre
        else:
            fp_t: dict = {}
            sf, fit, P_root, P_top, rp = fit_pass(result, timing=fp_t, **self._fit_kwargs(job))
            timing["fit_pass_s"] = time.perf_counter() - t_refine0
            timing.update({f"fit_{k_}": v_ for k_, v_ in fp_t.items()})
        t_gate0 = time.perf_counter()
        if main:
            self._carry_lower_end(fit, k)
        elif self.tilt_main is not None:
            self._keep_lower_end(fit, k)
        u_SE = unit(line_prev.u)
        extent = None
        if fit.ok:
            extent = pass_extent_summary(fit, direction, u_SE)
        lag = end_zone_lag_m(job, self.calib, self._depth_spec_m(), fit.n if fit.ok else None)
        cov = None
        if fit.ok:
            fa = _features_arrays(sf, P_root, P_top, fit, _extent_frame(extent, fit.n))     # 단면 위치 = S 에서 (S→E 계약)
            cov = pass_coverage(fa, float(fa["s_a_m"]), float(fa["s_b_m"]), self._process(), lag)
        verdict = fidelity_gate(fit.feature_medians, rp, self.spec, "fidelity", qc_targets=self.qc_targets, extent=extent,
                                accept_inferred_end=self.at.accept_inferred_end, k=self.settings.judgement_k, coverage=cov,
                                validity=self._validity())
        fm = fit.feature_medians
        try:
            spec_rep = specimen_report(fit, verdict, self._user_extent(extent), k, self._user_direction(direction), self.spec)
            self.res.specimen_report = spec_rep
        except Exception as e:  # noqa: BLE001
            spec_rep = None
            self.res.notes.append(f"specimen report failed: {e}")
        if extent is not None:
            self._decide("ends", f"pass {k} ({direction}): S end {'observed' if extent['S']['observed'] else 'NOT observed'} at "
                                 f"{np.round(extent['S']['point'], 4).tolist()} m, E end {'observed' if extent['E']['observed'] else 'NOT observed'} "
                                 f"at {np.round(extent['E']['point'], 4).tolist()} m"
                                 + (f" [occluded {extent['occluded']}]" if extent["occluded"] else "")
                                 + (" [plates staggered]" if extent["staggered"] else "")
                                 + f"; length {extent['length_mm']:.2f} mm (plates {_fmtf(extent.get('plate_length_mm'), '.2f')} mm, design "
                                 f"{self.spec.weld_length_mm if self.spec is not None else float('nan'):.1f} mm), laser-hit spacing "
                                 f"{extent['spacing_mm']:.2f} mm, hit range {np.round(np.asarray(extent['hit_range_SE']) * 1e3, 1).tolist()} mm",
                         extent=_jsonable(extent))
        self._decide("fit", f"pass {k} ({direction}): fit ok={fit.ok} flags={fit.flags} line rms {fit.line_rms_m * 1e3:.3f} mm "
                     f"plane rms {fit.plane_rms_m * 1e3:.3f} mm inliers {fit.n_inliers}/{fm.get('n_groove_profiles')} "
                     f"u={np.round(fit.u, 4).tolist()} n={np.round(fit.n, 4).tolist()}")
        self._decide("gate", f"pass {k}: {verdict.summary()}; depth {_fmtf(fm.get('depth_mm'))} gap {_fmtf(fm.get('root_gap_mm'))} "
                     f"bevel {_fmtf(fm.get('bevel_angle_deg'))} opening {_fmtf(fm.get('top_opening_mm'))} mm, floor "
                     f"{_fmtf(fm.get('floor_visible_fraction'))} root edges {_fmtf(fm.get('root_edges_fraction'))}; placement "
                     f"(diagnostic) psi {_fmtf(rp.get('psi_deg'))} roll {_fmtf(rp.get('roll_deg'))} deg lateral "
                     f"{_fmtf(rp.get('lateral_m', float('nan')) * 1e3)} mm standoff err {_fmtf(rp.get('standoff_m', float('nan')) * 1e3)} mm")
        entry = {"k": k, "kind": "pass", "stage": "fidelity", "direction": direction, "dir": str(pass_dir), "fit": fit.to_dict(),
                 "verdict": verdict.to_dict(), "residuals": rp, "extent": extent, "features_medians": fm, "n_ok": sf.n_ok,
                 "n_profiles": sf.n_profiles, "straightness_rms_mm": sf.straightness_rms_mm, "speed_mps": speed_mps,
                 "progression_sign": int(job.progression_sign), "specimen_report": spec_rep,
                 "overscan": {k_: ov[k_] for k_ in ("hit_offset_m", "overscan_first_m", "overscan_last_m", "floor_limited",
                                                    "hit_from", "hit_to", "origin_from", "origin_to")}}
        self.res.passes.append(_jsonable(entry))
        self.res.residual_history.append(_jsonable({"k": k, "stage": "fidelity", "status": verdict.status, **rp,
                                                    "length_mm": extent["length_mm"] if extent else None}))
        try:
            with open(pass_dir / "fit.json", "w", encoding="utf-8") as f:
                json.dump(_jsonable({"fit": fit.to_dict(), "verdict": verdict.to_dict(), "residuals": rp, "extent": extent}),
                          f, indent=1, ensure_ascii=False, default=str)
            self.res.files.append(str(pass_dir / "fit.json"))
            if spec_rep is not None:
                with open(pass_dir / "specimen_report.json", "w", encoding="utf-8") as f:
                    json.dump(spec_rep, f, indent=1, ensure_ascii=False, default=str)
                self.res.files.append(str(pass_dir / "specimen_report.json"))
        except Exception as e:  # noqa: BLE001
            self.res.notes.append(f"features save failed: {e}")
        timing["gate_report_files_s"] = time.perf_counter() - t_gate0
        self._record_pass_timing(k, timing)
        new_line = None
        if fit.ok and extent is not None:
            c = np.asarray(fit.c, dtype=float)
            u = unit(fit.u) if float(unit(fit.u) @ u_SE) >= 0.0 else -unit(fit.u)
            n = unit(fit.n)
            if float(n @ unit(line_prev.n)) < 0.0:
                n = -n
            new_line = SeamLine(c=c, u=u, n=n, s_S=float((np.asarray(extent["S"]["point"]) - c) @ u),
                                s_E=float((np.asarray(extent["E"]["point"]) - c) @ u), obs_S=bool(extent["S"]["observed"]),
                                obs_E=bool(extent["E"]["observed"]), how_S="observed" if extent["S"]["observed"] else "seen",
                                how_E="observed" if extent["E"]["observed"] else "seen", k=k,
                                spacing_m=extent["spacing_mm"] * 1e-3 if _fin(extent["spacing_mm"]) else float("nan"),
                                hit_range=tuple(extent["hit_range_SE"]), direction=direction,
                                line_rms_m=float(fit.line_rms_m), plane_rms_m=float(fit.plane_rms_m),
                                sigma_S_m=float(extent[f"{'last' if direction == 'E->S' else 'first'}_sigma_mm"]) * 1e-3,
                                sigma_E_m=float(extent[f"{'first' if direction == 'E->S' else 'last'}_sigma_mm"]) * 1e-3,
                                range_mm=float(fit.feature_medians.get("sensor_z_mm", float("nan"))))
            try:
                pp = _sq.pass_posture(job, self.calib)
                posture_key = (round(float(pp["tilt_deg"]), 6), round(float(pp["pitch_deg"]), 6), pp.get("torch_above"))
            except Exception:  # noqa: BLE001 - report only
                posture_key = None
            self.pass_ends.append({"k": k, "direction": "S->E" if direction == "S->E" else "E->S", "speed_mps": speed_mps,
                                   "posture": posture_key,
                                   "ends": {X: {"observed": bool(extent[X]["observed"]), "point": np.asarray(extent[X]["point"]),
                                                "occluded": bool(extent[X]["occluded"])} for X in ("S", "E")}})
        return {"sf": sf, "fit": fit, "P_root": P_root, "P_top": P_top, "residuals": rp, "verdict": verdict, "extent": extent,
                "line": new_line, "coverage": cov, "end_lag_m": lag}

    def _record_pass_timing(self, k: int, t: dict) -> None:
        """패스 뒤 처리 단계별 시간 [s] (2026-10-05: 실기 서버에서 패스당 12–14 s, 같은 계산이 서버 밖에서는 4–5 s — 어디서 느린지)."""
        def g(key):
            v = t.get(key)
            return float(v) if isinstance(v, (int, float)) else float("nan")
        if t.get("post_scan_where") and math.isfinite(g("post_scan_wall_s")):   # 작업 프로세스: 저장·재구성·QC·fit 의 벽시계
            parts = ("motion_stop_to_data_s", "pack_s", "post_scan_wall_s", "gate_report_files_s")
        else:
            parts = ("motion_stop_to_data_s", "save_raw_s", "reconstruct_and_qc_s", "fit_pass_s", "gate_report_files_s")
        total = sum(g(k_) for k_ in parts if math.isfinite(g(k_)))
        feat = f"features {g('fit_features_s'):.2f} ({t.get('fit_features_mode')} x{t.get('fit_features_workers')}"
        feat += f", pool error {t['fit_features_pool_error']}" if t.get("fit_features_pool_error") else ""
        feat += f", pool note {t['fit_features_pool_note']}" if t.get("fit_features_pool_note") else ""
        if "stream_seeded" in t or "stream_sent_frames" in t:      # 패스 중 판 성장 스트림 (post_scan): 쓴 행 / 재구성 행, 서버 CPU
            # 서버 CPU = 스트림이 서버 프로세스에 더한 것 (피더 스레드 + 시작·멈춤 호출 + 그동안 실행기 스레드 — 상한). 잰 것만: 한 부분이라도
            # 못 재면 합을 내지 않고 잰 피더 스레드만 보인다
            sc = g("stream_server_cpu_s")
            cpu = (f"server cpu {sc:.3f} s all stream threads" if math.isfinite(sc)
                   else f"server cpu not measured (feeder thread {g('stream_feeder_cpu_s'):.3f} s)")
            feat += (f", plates during the pass {t.get('stream_seeded', 0)}/{t.get('stream_rows', '?')}"
                     f" ({cpu}, {t.get('stream_round_trips', '?')} round trip, {t.get('stream_sent_writes', '?')} writes)")
        for k_ in ("stream_note", "stream_error", "stream_feeder_error", "stream_feeder_running"):
            feat += f", {k_[7:]} {t[k_]}" if t.get(k_) else ""
        where = (f" [{t.get('post_scan_where')}" + (f", pack {g('pack_s'):.2f}, wall {g('post_scan_wall_s'):.2f}"
                                                    if math.isfinite(g("post_scan_wall_s")) else "") + "]") if t.get("post_scan_where") else ""
        sv = lambda key: "bg" if not math.isfinite(g(key)) and t.get("post_scan_where") == "worker" else f"{g(key):.2f}"   # noqa: E731
        self._decide("timing", f"pass {k} post-scan {total:.2f} s{where}: motion stop -> data {g('motion_stop_to_data_s'):.2f}, save raw "
                               f"{sv('save_raw_s')}, reconstruct {g('recon_reconstruct_s'):.2f}, qc {g('recon_qc_s'):.2f}, save recon "
                               f"{sv('recon_save_recon_s')} | {feat}), polylines {g('fit_polylines_plate_points_s'):.2f}, refine "
                               f"{g('fit_refine_s'):.2f}, ends {g('fit_ends_s'):.2f}, medians {g('fit_medians_bins_s'):.2f}, other "
                               f"{g('fit_edge_outliers_s') + g('fit_residuals_s'):.2f} | gate+report+files {g('gate_report_files_s'):.2f}",
                     timing=_jsonable(t))
        if self.res.passes:
            self.res.passes[-1]["timing"] = _jsonable(t)

    def _ambiguity_reading(self, verdict, ext: dict, fit: PassFit, k: int) -> Optional[dict]:
        """'end_ambiguous' 한 읽기 (4d P9): 설계 길이 대비 편차 부호, 길이, 길이 σ (게이트 괄호 σ 두 끝), 두 끝을 이 패스가 스스로 쟀나
        (이어받거나 추정한 끝이 아님)."""
        if self.spec is None or not (self.spec.weld_length_mm > 0):
            return None
        L = float(ext.get("length_mm", float("nan")))
        if not _fin(L):
            return None
        s1, s2 = (float(ext.get(f"{w}_sigma_mm", float("nan"))) for w in ("first", "last"))
        sig = math.hypot(s1 if _fin(s1) else 0.0, s2 if _fin(s2) else 0.0)
        # 확인에 쓸 수 있는 읽기 = 두 끝이 모두 잰 끝 — 추정 끝 (다른 끝 ± 설계 길이) 은 안 되고, 3G 주 패스가 이어받은 아래 끝은 끝 패스가
        # 잰 끝이라 된다 (위 끝은 이 패스가 다시 잰다; 그러지 않으면 3G 에서는 확인이 불가능해 예산을 다 쓴다 — 4d 검토)
        own = not any(ext.get(f"{w}_inferred") is not None for w in ("first", "last"))
        return {"k": int(k), "sign": float(np.sign(L - float(self.spec.weld_length_mm))), "length_mm": L, "sigma_mm": sig, "own": bool(own)}

    def _occlusion_terminal(self, ext: dict, direction: str, check: str, k: int) -> bool:
        """'end_occluded_<first|last>' 이 실행을 끝내는가 (F5): 추정이 꺼져 있거나 (accept_inferred_end false), 양끝이 다 가렸거나, 다른 끝이
        이 패스에서 이미 관측됐으면 (추정할 기회가 있었다) — 다시 스캔해도 바뀌지 않는다. 그 외 (추정은 켜져 있는데 다른 끝을 아직 못 봤다:
        패스 1 은 판 안쪽의 보정된 E 에서 시작하므로 늘 그렇다) 는 계속: 다음 패스가 다른 끝을 관측하고 게이트가 가린 끝을 추정한다."""
        which = check.rsplit("_", 1)[-1]
        X = ("E" if which == "first" else "S") if direction == "E->S" else ("S" if which == "first" else "E")
        other = "E" if X == "S" else "S"
        if X not in self._occluded_seen and self.n_pass < self.max_passes:
            # 확인 먼저 (4d P9; g3_19B 는 2 % 자세에서 첫 읽기로 멈췄다): 다음 패스가 그 끝을 다시 잰다 — 다시 가리거나 못 보면 멈춘다
            self._occluded_seen[X] = k
            self._decide("ends", f"pass {k}: {X} end occluded (object on the plate) - not stopping on the first reading: the next pass "
                                 f"measures the {X} end again and the run stops only if it is again occluded or not observed")
            return False
        if not self.at.accept_inferred_end or len(ext.get("occluded") or []) >= 2 or bool(ext[other]["observed"]):
            return True
        self._decide("ends", f"pass {k}: {X} end occluded (object on the plate) and the {other} end not yet observed - inference is "
                             f"enabled (auto_teach.accept_inferred_end): next pass observes the {other} end, the {X} end is then inferred "
                             f"as the {other} end -/+ the design length")
        return False

    def _plan_next(self, line: SeamLine, k: int, direction: str) -> PassPlan:
        """패스 k 의 직선 → 다음 패스 계획 (plan_next_pass). 이번 패스에서 관측되지 않았지만 이전 패스가 관측한 끝은 그 점을 새
        직선에 투영해 쓴다 ('observed_prev'). 가려진 끝 (판 위 물체, accept_inferred_end) 은 가림 경계를 알려진 끝으로 쓴다 — 진짜 끝은
        그 너머 (물체 아래) 이므로 다른 끝 ≤ 가림 경계 + 설계 길이: 다음 패스가 다른 끝을 관측하고 게이트가 가린 끝을 추정한다 (F5)."""
        known = {}
        last = self.pass_ends[-1] if self.pass_ends and self.pass_ends[-1].get("k") == k else None
        for X in ("S", "E"):
            if getattr(line, f"obs_{X}"):
                known[X] = (float(getattr(line, f"s_{X}")), "observed")
                continue
            prev = next((p for p in reversed(self.pass_ends[:-1]) if p["ends"][X]["observed"]), None)
            if prev is not None:
                known[X] = (line.s_of(prev["ends"][X]["point"]), f"observed_prev (pass {prev['k']})")
            elif last is not None and last["ends"][X].get("occluded") and self.at.accept_inferred_end:
                known[X] = (float(getattr(line, f"s_{X}")), f"occluded edge (pass {k}; the end is inferred from the other end)")
        L_design = float(self.spec.weld_length_mm) * 1e-3 if (self.spec is not None and self.spec.weld_length_mm > 0) else None
        plan = plan_next_pass(k, direction, line, known["S"][0] if "S" in known else None, known["E"][0] if "E" in known else None,
                              design_length_m=L_design, hit_range=line.hit_range, overscan_m=self.overscan_m)
        for X in ("S", "E"):
            if X in known and known[X][1] != "observed":
                setattr(plan, f"how_{X}", known[X][1])
        self._decide("ends", f"after pass {k}: next pass {plan.direction} (rule {plan.rule}): S target {plan.how_S} "
                             f"({(plan.s_S - line.s_S) * 1e3:+.1f} mm vs this pass), E target {plan.how_E} "
                             f"({(plan.s_E - line.s_E) * 1e3:+.1f} mm); design length {(L_design or float('nan')) * 1e3:.1f} mm", plan=plan.to_dict())
        return plan

    # ---- 11. 출력 ----------------------------------------------------------------------------------------
    def _flush_pending_saves(self) -> None:
        """뒤에서 쓰던 패스 원자료·재구성 파일을 기다린다 (출력·로그 쓰기 전; 실패는 노트)."""
        rows = []
        for name, fut in self._pending_saves:
            try:
                for raw_path, t_raw, t_rec, err in _post_scan.wait_result(fut, self.cancel):
                    rows.append(f"{os.path.basename(os.path.dirname(raw_path))} raw {t_raw:.2f} recon {t_rec:.2f}" + (f" ERROR {err}" if err else ""))
                    if err:
                        self.res.notes.append(f"{name}: background save failed: {err}")
            except _post_scan.Cancelled as e:               # 실행 취소 (S4): 기다리지 않는다 (쓰기는 작업 프로세스가 이어 한다 — 끝내지 않는다)
                self.res.notes.append(f"{name}: background save not awaited (run cancelled: {e})")
            except Exception as e:  # noqa: BLE001
                self.res.notes.append(f"{name}: background save did not finish: {type(e).__name__}: {e}")
        if rows:
            self._decide("timing", "background saves (during the next scan): " + "; ".join(rows))
        self._pending_saves = []

    def _write_outputs(self, job_final: Optional[ScanJob], feats: Optional[dict], fm: dict) -> None:
        r = self.res
        r.feature_medians = dict(fm or {})
        if r.status == STATUS_PASS and feats:
            cs_job = self._cross_sections_start(feats)     # 같은 단계: 스캔이 PASS 면 (용접 경로 도달성과 무관하게) 단면 법선 — 나란히
            self._weld_plan(feats)
            self._cross_sections_finish(cs_job)
        self._flush_pending_saves()                        # 뒤에서 쓰던 패스 파일 (로그 전에 끝나 있어야 한다)
        if self._vis_clouds:
            vc = sorted(self._vis_clouds)
            self._decide("timing", "pass clouds (RViz, after each pass result - not on the pass path): "
                                   + "; ".join(f"pass {k_} {n_} points {t_ * 1e3:.1f} ms [{w_}]" for k_, n_, t_, w_ in vc),
                         timing=[{"k": k_, "points": n_, "build_s": t_, "where": w_} for k_, n_, t_, w_ in vc])
        if job_final is not None:
            r.job_final = job_final
            cal_path = self.out / "calib_refined.yaml"
            try:
                d = self.calib.to_dict()
                d["auto_teach"] = {"optical_branch_flipped": r.optical_branch_flipped, "base_calib_source": self.calib0.source,
                                   "note": "effective calib used for the final scan: T_flange_opt (+Rz(pi) if the probe flipped the "
                                           "optical yaw branch); pass to scan --calib when flipped"}
                import yaml
                with open(cal_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(_plain(d), f, sort_keys=False)
                r.calib_path = str(cal_path)
                r.files.append(str(cal_path))
            except Exception as e:  # noqa: BLE001
                r.notes.append(f"calib save failed: {e}")
            last = r.passes[-1] if r.passes else {}
            info = last.get("job_info", {})
            teach_auto = {
                "status": r.status, "reason": r.reason, "time": _dt.datetime.now().isoformat(timespec="seconds"),
                "root_start": job_final.start.tolist(), "root_end": job_final.end.tolist(),
                "top_start": info.get("top_start"), "top_end": info.get("top_end"), "u": job_final.unit_dir.tolist(),
                "n": job_final.normal_unit.tolist(), "ends": info.get("ends"), "extent_flags": last.get("fit", {}).get("flags"),
                "residual_history": r.residual_history, "probe": r.probe, "delta_cb": r.delta_cb,
                "converged": r.converged, "converged_detail": r.converged_detail,
                "optical_branch_flipped": r.optical_branch_flipped, "position": r.position,
                "extent": r.extent, "time_offset": r.time_offset, "sites": r.sites, "line_history": r.line_history,
                "visit_order": r.visit_order, "internal_order_note": ("sites / passes / line_history are in the internal (visited) order; "
                                                                      "root_start/root_end, extent and the weld plan in the taught S->E"),
                "plate_top_measured_m": self.plate_top, "feature_medians": fm, "passes": [
                    {k_: p[k_] for k_ in ("k", "kind", "stage", "direction", "dir", "verdict", "residuals", "extent", "overscan")
                     if k_ in p} for p in r.passes],
                "coarse_range_m": float(self.job_c.standoff_m) if self.job_c is not None else None,
                "pass_plate_m": self.pass_plate_m, "sensor_view": self.view.as_dict() if self.view is not None else None,
                "overscan_m": self.overscan_m, "overscan_reference": "laser hit point (laser plane x measured plate top)",
                "calib_source": self.calib0.source, "base_point_bias_note": "base-frame coordinates carry the nominal hand-eye bias "
                                                                             "(few mm) until hand-eye calibration 5.6.2",
            }
            teach = dict(self.doc.get("teach", {}) or {})
            teach["auto"] = _plain(teach_auto)
            meta = dict(self.doc.get("meta", {}) or {})
            # M-6: meta.status = auto-teach 결과. 'scan' 은 PASS 가 아닌 정련 작업을 --allow-rough 없이 거부한다.
            meta.update(rough=False, status=r.status, auto_teach_status=r.status, calib_file=str(cal_path), config=self.settings.path,
                        pose_source=self.pose_source, out_dir=str(self.out),
                        converged=str(r.converged or "false"), converged_last=_plain(dict(r.converged_detail or {})))
            jp = self.out / "job_refined.yaml"
            save_job(jp, job_final, teach, meta, specimen=self.spec, qc_targets=self.doc.get("qc_targets"))
            r.job_path = str(jp)
            r.files.append(str(jp))
        if r.specimen_report is not None:
            try:
                with open(self.out / "specimen_report.json", "w", encoding="utf-8") as f:
                    json.dump(r.specimen_report, f, indent=1, ensure_ascii=False, default=str)
                r.files.append(str(self.out / "specimen_report.json"))
            except Exception as e:  # noqa: BLE001
                r.notes.append(f"specimen report save failed: {e}")
        try:
            with open(self.out / "autoteach_log.json", "w", encoding="utf-8") as f:
                json.dump({"config": self.at.to_dict(), "job_rough": job_to_dict(self.job0),
                           "measure_speed": self.measure_speed_record(), "job_file": _jo.job_file_of(self.doc),
                           "job_overrides": self.job_overrides(), "job_effective": getattr(self, "job_effective_path", None),
                           "job_coarse": job_to_dict(self.job_c) if self.job_c is not None else None,
                           "calib_effective": self.calib.to_dict(), **r.to_dict()}, f, indent=1, ensure_ascii=False, default=str)
            r.files.append(str(self.out / "autoteach_log.json"))
        except Exception as e:  # noqa: BLE001
            r.notes.append(f"log save failed: {e}")

    # ---- 용접 경로 (용접 전 스캔의 결과물) ------------------------------------------------------------------
    def _weld_plan(self, feats: dict) -> None:
        """마지막 패스의 프로파일 특징 + 관측 양끝 → 다층 용접 경로 (weld_plan.json/.npz; 층·비드 배치는 그 단면과 WPS 에서). 만들지 못하면 DATA_FAIL —
        용접 경로 생성까지가 용접 전 스캔의 역할 (사용자 2026-10-03)."""
        r = self.res
        try:
            t0 = time.perf_counter()
            process = WeldProcess.from_dict(weld_process_doc(self.doc if isinstance(self.doc, dict) else None, r.position))
            wkw = dict(base_frame=self.job0.base_frame, position=str(r.position or "1G"), gravity_axis_base=self.job0.gravity_axis_base,
                       k=self.settings.judgement_k,                  # S→E 계약 검사의 k·σ_끝 (판정 배수)
                       design=weld_design_from_doc(self.doc if isinstance(self.doc, dict) else None, process))
            plan, where = None, "in_process"
            if self.at.post_scan_worker:
                # 순수 계산 → 작업 프로세스 (서버 안 GIL 경합 회피). 특징 추출 풀 (서버 쪽, 이때 놀고 있다) 을 쓴다 — post_scan 작업
                # 프로세스는 마지막 패스의 파일을 뒤에서 쓰는 중일 수 있다
                pool, fut = None, None
                try:
                    pool = _sfmod.worker_pool(self.at.feature_workers_effective)
                    fut = pool.submit(weld_plan_from_features, feats, r.extent, self.spec, process, **wkw)
                    plan = _post_scan.wait_result(fut, self.cancel)
                    where = "worker"
                except PlanContractError:
                    raise                                        # 계약 거부는 계산 결과 — 서버 안에서 다시 해도 같다
                except _post_scan.Cancelled:
                    if fut is not None and not fut.done():      # 끝나지 않은 일 — 한계에서와 같은 규칙으로 그 풀의 자식을 끝낸다
                        _sfmod.shutdown_feature_pool(pool, kill=True)
                    raise
                except Exception as e:  # noqa: BLE001 - 작업 프로세스 실패: 서버 안에서 같은 계산
                    if fut is not None and not fut.done():      # 상태로 가른 멈춤 (한계 안에 끝나지 않음) — 그 풀 (이 객체) 의 자식을 끝낸다
                        _sfmod.shutdown_feature_pool(pool, kill=True)
                    r.notes.append(f"weld plan worker failed ({type(e).__name__}: {e}) - computed in the server process")
            if plan is None:
                plan = weld_plan_from_features(feats, r.extent, self.spec, process, **wkw)
            t1 = time.perf_counter()
            from scan_core.weld_torch import d14_on_this_arm, weld_torch_config
            d14, d14_note = d14_on_this_arm(weld_torch_config(self.settings.params))
            check = getattr(self.rs, "check_weld_plan", None)          # MoveIt 어댑터: MoveIt 도달성 (모션 없음)
            if not d14:                 # 설정이 용접 토치가 이 팔에 없다고 한다 (사용자 2026-10-12): D14 없이 납품 + 기록 (다른 토치로 대신하지 않는다)
                plan.reachability = {"checked": False, "note": d14_note}
                r.notes.append(f"weld plan: {d14_note}")
                self._decide("weld_plan", d14_note, reachability=plan.reachability)
            elif check is not None:     # D14 + the torch orientation per pass (scan_core.weld_orient) and one TCP roll
                from scan_core import weld_orient as _wo
                he = getattr(self.calib, "he_sigma_m", None)
                req = _wo.clearance_requirement(float(self.settings.judgement_k), [
                    ("measured surface / aim position", None, "the legacy path has no estimator covariance"),
                    ("robot posture-dependent pose error", None if he is None else float(he),
                     "calibration g1.he_sigma_m: the pose-dependent hand-eye residual between postures"),
                    ("weld torch TCP calibration", None, "not documented: the TCP is the CAD contact-tip end (nominal)"),
                    ("EOAT geometry (CAD vs the built tool)", None, "not documented: CAD with the 5.0 mm dust plate, not measured")])
                plan.clearance = req.to_dict()
                body_fn = getattr(self.rs, "weld_tool_body", None)
                if callable(body_fn):
                    reach_fn = getattr(self.rs, "weld_reach_model", None)
                    from scan_core.weld_torch import torch_with_nozzle_end
                    from scan_core.weld_clearance import tool_roles
                    step_fn = getattr(self.rs, "weld_line_step_m", None)            # D14 on the delivered lines (scan_core.weld_line)
                    _wo.choose_orientation(plan, check, body=tool_roles(body_fn()), required=req, log=self._p,
                                           torch=torch_with_nozzle_end(weld_torch_config(self.settings.params), body_fn),
                                           reach=reach_fn() if callable(reach_fn) else None,
                                           step_m=step_fn() if callable(step_fn) else None)
                    self._decide("weld_plan", f"weld plan torch roll: {plan.torch_roll['reason']}", torch_roll=plan.torch_roll,
                                 torch_orientation=plan.orientation)
                else:
                    plan.feasibility = [{"k": wp.k, "kind": wp.kind, "reachable": None,
                                         "reason": "weld-pose clearance not checked: the platform gives no tool collision geometry"}
                                        for wp in plan.passes]
            t2 = time.perf_counter()
            r.files.extend(plan.save(self.out))
            t3 = time.perf_counter()
            r.weld_plan = plan
            self._decide("timing", f"weld plan {t3 - t0:.2f} s: plan {t1 - t0:.2f} [{where}], reachability check {t2 - t1:.2f}, "
                                   f"save {t3 - t2:.2f}", timing={"plan_s": t1 - t0, "reachability_s": t2 - t1, "save_s": t3 - t2,
                                                                  "plan_where": where})
            # 도달성: False = 확인했고 안 된다 (DATA_FAIL), None = 확인하지 못했다 (플랫폼이 답을 못 받음 — 확인 없는 플랫폼처럼 실행은 계속,
            # 노트로 남긴다; MoveIt 어댑터 check_weld_plan)
            unchecked = [f for f in plan.feasibility if "reachable" in f and f["reachable"] is None]
            if unchecked:                               # 노트·결정 로그 (콘솔)·결과 사유 (액션 결과·클라이언트) 모두에 보인다
                msg = "robot reachability not checked: " + "; ".join(f"pass {f['k']} ({f.get('reason')})" for f in unchecked)
                r.notes.append(f"weld plan: {msg}")
                self._decide("weld_plan", msg, unchecked=[f["k"] for f in unchecked])
                r.reason = f"{r.reason}; weld plan {msg}" if r.reason else f"weld plan {msg}"
            bad = [f for f in plan.feasibility if not f.get("reachable") and not any(f is u for u in unchecked)]
            if bad:
                raise ValueError("; ".join(f"pass {f['k']} not reachable: {f.get('reason')}" for f in bad))
        except _post_scan.Cancelled as e:                   # 실행 취소 (S4): 용접 경로 없이 끝낸다 (서버 안 다시 계산 없음)
            r.status = STATUS_ABORTED
            r.reason = f"run cancelled while the weld plan was computed ({e}; scan: {r.reason})"
            r.notes.append(r.reason)
            self._p(f"{r.status}: {r.reason}")
        except Exception as e:  # noqa: BLE001
            r.status, r.category = STATUS_DATA_FAIL, "waypoint"           # 분류 (이름만, r3 §4.2): 용접 경로 생성 실패
            r.reason = f"weld plan failed: {type(e).__name__}: {e} (scan: {r.reason})"
            r.notes.append(r.reason)
            self._p(f"{r.status}: {r.reason}")

    def _cross_sections_start(self, feats: dict) -> Optional[dict]:
        """마지막 패스 (또는 구간별 병합) 의 프로파일 면 → 단면 상세 법선 (cross_sections.json/.npz; 사용자 2026-10-08 최종 출력 = 단면
        법선 + TCP 용접선 경로). 단면 창·증거 수 = 품질 구간과 같은 유도 (scan_quality.bin_for/bin_need: 용접 경로 단면 창, 프로파일이
        성기면 넓힘), 양끝 = 납품 양끝 (r.extent — 용접 경로와 같은 S/E). 작업 프로세스 풀에 용접 경로 계획과 **나란히** 맡긴다 (끝 처리
        시간에 더하지 않는다; 풀이 없거나 실패하면 _cross_sections_finish 가 서버 안에서). 실패는 노트 (판정·정지가 아니다)."""
        r = self.res
        try:
            ext = r.extent or {}
            proc = self._process()
            sp = _sq.spacing_of(feats)
            win, need = _sq.bin_for(sp, proc), _sq.bin_need(sp, proc)
            args = (feats, ext["S"]["point"], ext["E"]["point"])
            kw = dict(window_m=win, need=need, spec=self.spec, n_plate=feats.get("n_base"), base_frame=self.job0.base_frame,
                      position=str(r.position or ""),
                      source=f"pass {ext.get('pass')}" + (" (best pass per section)" if "source_pass" in feats else ""))
        except Exception as e:  # noqa: BLE001 - 납품 보조 출력: 실패는 기록만
            r.notes.append(f"cross sections failed: {type(e).__name__}: {e}")
            self._p(f"cross sections failed: {type(e).__name__}: {e}")
            return None
        fut, pool = None, None
        if self.at.post_scan_worker:
            try:
                pool = _sfmod.worker_pool(self.at.feature_workers_effective)
                fut = pool.submit(_cs.station_cross_sections, *args, **kw)
            except Exception as e:  # noqa: BLE001 - 풀 실패: 서버 안에서
                r.notes.append(f"cross sections worker not started ({type(e).__name__}: {e}) - computed in the server process")
        return {"t0": time.perf_counter(), "args": args, "kw": kw, "future": fut, "pool": pool}

    def _cross_sections_finish(self, job: Optional[dict]) -> None:
        if job is None:
            return
        r = self.res
        try:
            cs, where = None, "in_process"
            t_wait = time.perf_counter()
            fut = job["future"]
            if fut is not None:
                try:
                    cs = _post_scan.wait_result(fut, self.cancel)
                    where = "worker"
                except _post_scan.Cancelled as e:           # 실행 취소 (S4): 단면 없이 (서버 안 다시 계산 없음)
                    if not fut.done():
                        _sfmod.shutdown_feature_pool(job["pool"], kill=True)
                    r.notes.append(f"cross sections not computed: run cancelled ({e})")
                    return
                except Exception as e:  # noqa: BLE001 - 작업 프로세스 실패: 서버 안에서 같은 계산
                    if not fut.done():                  # 상태로 가른 멈춤 (한계 안에 끝나지 않음) — 그 풀 (이 객체) 의 자식을 끝낸다 (용접 경로와 같은 규칙)
                        _sfmod.shutdown_feature_pool(job["pool"], kill=True)
                    r.notes.append(f"cross sections worker failed ({type(e).__name__}: {e}) - computed in the server process")
            if cs is None:
                cs = _cs.station_cross_sections(*job["args"], **job["kw"])
            files = _cs.save_cross_sections(cs, self.out)
            r.files.extend(files)
            t1 = time.perf_counter()
            got = {nm: v["n_stations"] for nm, v in cs["summary"].items() if v["n_stations"]}
            win = float(job["kw"]["window_m"])
            r.cross_sections = {"files": files, "n_stations": len(cs["stations"]), "window_m": win, "faces": got,
                                "bevel_side": cs["bevel_side"], "notes": cs["notes"], "where": where,
                                "timing": {"time_s": t1 - job["t0"], "added_s": t1 - t_wait}}   # 벽시계 측정은 'timing' 아래 (결정론 비교 밖)
            self._decide("cross_sections", f"cross sections: {len(cs['stations'])} stations every {win * 1e3:.1f} mm, faces "
                                           + ", ".join(f"{nm} {n_}" for nm, n_ in got.items())
                                           + (f"; {'; '.join(cs['notes'])}" if cs["notes"] else ""),
                         summary=_jsonable(cs["summary"]))
            self._decide("timing", f"cross sections {t1 - job['t0']:.3f} s [{where}, alongside the weld plan] - added after the weld "
                                   f"plan {t1 - t_wait:.3f} s (wait + save)",
                         timing={"cross_sections_s": t1 - job["t0"], "added_s": t1 - t_wait, "where": where})
        except Exception as e:  # noqa: BLE001 - 납품 보조 출력: 실패는 기록만
            r.notes.append(f"cross sections failed: {type(e).__name__}: {e}")
            self._p(f"cross sections failed: {type(e).__name__}: {e}")

    def _vis_step_m(self) -> Optional[float]:
        """처리된 패스 점구름 (시각화) 의 점 간격 = 용접 경로 촘촘한 경로 간격 (WeldProcess.dense_step_mm — 납품물이 쓰는 가장 고운 해상도).
        지금 보는 쪽이 없으면 (pass_cloud_wanted 없음·False — ROS 서버: ~/pass_cloud 구독자 없음, KIRO with_rviz:=false) None (만들지 않는다)."""
        if self.on_event is None or self.pass_cloud_wanted is None:
            return None
        try:
            if not self.pass_cloud_wanted():
                return None
        except Exception:  # noqa: BLE001 - 시각화 질의가 실행을 막지 않는다
            return None
        return float(self._process().dense_step_mm) * 1e-3

    def _emit_pass_cloud(self, k: int, name: str) -> None:
        """패스 k 의 처리된 점구름 (실측 자세 재구성 점, scan_core.cross_sections.decimated_cloud) 이 다 만들어지면 'pass_cloud' 이벤트.
        작업 프로세스 (Future: 패스 결과 뒤 맡긴 vis_cloud) 또는 서버 안 대체 경로 ((재구성 결과, 간격): 지금 — 판정 뒤 — 짧은 스레드).
        이 스레드는 기다리지 않는다 (콜백이 이벤트를 낸다; 서버는 발행 큐에 넣기만)."""
        src, self._leg_cloud = self._leg_cloud, None
        if src is None:
            return
        if isinstance(src, tuple):
            fut, where = Future(), "server thread"

            def _build(res_, step_):
                try:
                    t_ = time.perf_counter()
                    P_ = _cs.decimated_cloud(res_, step_)
                    fut.set_result((P_, time.perf_counter() - t_))
                except BaseException as e_:  # noqa: BLE001
                    fut.set_exception(e_)
            threading.Thread(target=_build, args=src, name=f"pass_cloud_{k}", daemon=True).start()
        else:
            fut, where = src, "worker"
        base_frame = self.job0.base_frame

        def _done(f) -> None:
            try:
                out = f.result()
            except BaseException as e_:  # noqa: BLE001 - 시각화: 기록만
                self._vis_clouds.append((k, 0, float("nan"), f"{where} failed: {type(e_).__name__}: {e_}"[:200]))
                return
            if out is None:
                return
            P_, t_ = out
            self._vis_clouds.append((k, int(P_.shape[0]), float(t_), where))
            self._event("pass_cloud", k=k, name=name, points=P_, base_frame=base_frame)
        fut.add_done_callback(_done)

    # ---- abort --------------------------------------------------------------------------------------------
    def _abort_path(self, status: str, reason: str, safety: bool = True, category: Optional[str] = None) -> None:
        """중단 경로: 상태·사유 기록과 센서 정지. 정지 명령은 보내지 않는다 — 이 패키지의 정지는 액션 취소뿐이고 (사용자 2026-10-11),
        실패한 이동은 JTC 안에서 이미 끝났다. 액션 취소로 정지된 실행이면 그 기록을 남긴다. 다음 실행은 플랫폼이 준비되면 시작한다."""
        r = self.res
        r.status, r.reason, r.category = status, reason, category
        self._p(f"{status}: {reason}")
        sess = self.sess
        if sess is not None and sess.sensor is not None and sess.sensor.client is not None:
            try:
                if sess.sensor.recording:
                    sess.sensor.stop_recording(settle_s=0.1)
                else:
                    sess.sensor.client.stop()
            except Exception as e:  # noqa: BLE001
                r.notes.append(f"sensor stop during abort failed: {e}")
        m = sess.robot.backend if (sess is not None and sess.robot is not None) else None
        if m is None or status in (STATUS_REFUSED, STATUS_DRY_RUN):
            return
        try:
            stopped = bool(m.estopped)                              # 이 실행이 액션 취소로 정지됐다
        except Exception:  # noqa: BLE001
            stopped = False
        what = "no motion was sent in this run" if not self._motion_sent else "the run ended"
        if stopped:
            lt = getattr(m, "latch", None)
            why_l = str(getattr(lt, "reason", "") or "") if lt is not None else ""
            r.notes.append(f"{what}: this run had been stopped ({why_l or 'action cancel'}); nothing more was sent")
            self._p(f"{what}: the run had been stopped ({why_l or 'action cancel'}); nothing more was sent")
        else:
            r.notes.append(f"{what}: no stop command (the only stop of this package is the action cancel)")
            self._p(f"{what}: no stop command sent")
        r.estop_record = self._latch_record_of_this_run()          # 이 실행의 정지 기록 (있으면)

    def _external_stop(self) -> Optional[str]:
        """What stopped this run from outside, or None: the cancel Future once it ended - executor.CANCEL_SERVER_SHUTDOWN when its
        result is that (the hosting server shuts down), else executor.CANCEL_OPERATOR (an action cancel) - or else the platform stop
        of this run ('stopped (<latch reason>)')."""
        from .executor import CANCEL_OPERATOR, CANCEL_SERVER_SHUTDOWN
        c = getattr(self, "cancel", None)
        if c is not None and c.done():
            try:
                why = None if c.cancelled() else c.result()
            except Exception:  # noqa: BLE001 - a Future ended with an exception: still a cancel
                why = None
            return CANCEL_SERVER_SHUTDOWN if why == CANCEL_SERVER_SHUTDOWN else CANCEL_OPERATOR
        m = getattr(getattr(self, "rs", None), "backend", None)
        try:
            if m is not None and m.estopped:
                return f"stopped ({m.latch.reason})"
        except Exception:  # noqa: BLE001 - no stop state readable: not stopped from outside as far as the run knows
            pass
        return None

    def _unexpected_failure(self, e: BaseException, d7: str = "") -> tuple:
        """An exception that reached run()'s generic handler → (status, reason, category). After a stop from outside (_external_stop:
        an action cancel, the server shutting down) it is that stop's consequence - a stopped service wait, the platform closing under
        the run (KIRO 2026-10-14: InvalidHandle reported as ERROR_INTERNAL) → ABORTED '<cause><d7>: <exception>', no category (as a
        cancel). A planning / checking service that did not answer (motion_types.PlatformUnavailable) → REFUSED before the first
        motion, else ABORTED, category motion. Anything else → ERROR, category internal."""
        from .motion_types import PlatformUnavailable
        text = f"{type(e).__name__}: {e}"
        cause = self._external_stop()
        if cause is not None:
            return STATUS_ABORTED, f"{cause}{d7}: {text}", None
        if isinstance(e, PlatformUnavailable):
            sent = bool(getattr(self, "_motion_sent", False))
            return (STATUS_ABORTED, text, "motion") if sent else (STATUS_REFUSED, f"{text} (nothing sent)", "motion")
        return STATUS_ERROR, text, "internal"

    def _latch_last(self):
        """플랫폼 정지 (StopLatch) 의 마지막 기록 (없으면 None)."""
        sess = getattr(self, "sess", None)
        lt = sess.robot.latch if (sess is not None and sess.robot is not None) else None
        return getattr(lt, "last", None) if lt is not None else None

    def _latch_record_of_this_run(self) -> Optional[dict]:
        """이 실행 중에 생긴 정지 기록만 (as_dict). 서버는 플랫폼을 실행 사이에 유지한다 — 실행 시작 전의 기록 (_latch_at_start) 을 그대로
        쓰면 앞 실행의 정지 사유가 PASS 실행 로그에 남는다 (2026-10-05 pos1_a 로그에 at24 의 정지)."""
        rec = self._latch_last()
        return rec.as_dict() if rec is not None and rec is not self._latch_at_start else None

    # ---- run ---------------------------------------------------------------------------------------------
    def _final_job(self, fit: PassFit, line: SeamLine, P_root, P_top, k: int, extent: Optional[dict] = None) -> Optional[ScanJob]:
        """마지막 패스 정련 → 납품 작업 (make_refined_job: S→E, 착점 오버스캔 리드). 가려진 끝은 auto_teach.accept_inferred_end 일 때만
        추정 끝 (다른 끝 ± 설계 길이) 으로 (info.ends 'inferred'). 실패하면 None (노트)."""
        if extent is not None and self.at.accept_inferred_end and extent.get("occluded"):
            ends = dict((fit.diag or {}).get("ends") or {})
            for w in extent["occluded"]:
                if extent.get(f"{w}_inferred") is not None and w in ends:
                    ends[w] = {**ends[w], "point": list(extent[f"{w}_inferred"]), "observed": True, "inferred": True}
            fit = dataclasses.replace(fit, diag={**(fit.diag or {}), "ends": ends}, flags=list(fit.flags))
        try:
            pitch = float(self.pass_pitch_deg) if self.pass_pitch_deg is not None else float(self.job0.pitch_deg)   # F1: 3G 패스 피치 유지
            # 납품 작업은 교시 S→E (방문 순서가 내부 S/E 를 바꿨으면 되돌린다): 방향·대략점을 교시대로 — 시·종점·리드·3G 방향이 그에 맞게
            sw = bool(self.res.rough_swapped)
            kw = dict(standoff_plate_m=self.pass_plate_m, calib=self.calib, overscan_m=self.overscan_m,
                      u_SE=(-np.asarray(line.u, dtype=float)) if sw else line.u,
                      rough_start=self.job0.end if sw else self.job0.start, rough_end=self.job0.start if sw else self.job0.end,
                      plate_top_m=self.plate_top, spec=self.spec,
                      job_id=f"{self.job0.job_id or 'job'}_refined", top_polyline=P_top, root_polyline=P_root,
                      plate_plane=self.job_ref.plate_plane_measured if self.job_ref is not None else None,
                      R_ref=self._T_now()[:3, :3])                                          # 마지막 패스의 실제 툴 자세
            if self.orientation == "vertical_3G" and pitch != 0.0:
                # 진행 부호는 마지막 패스 자세 (토치를 뒤집은 주 패스일 수 있다) 에서 고르므로 피치 부호도 그 자세에서 다시: 입사 기울기에
                # 더해지는 쪽 (광학 원점이 착점보다 더 높은 쪽) — 패스 1 의 규칙과 같다 (뒤집힌 자세에 패스 1 부호를 쓰면 내려다봄이 줄어
                # 아래 끝 오버스캔이 모자랐다)
                cands = [make_refined_job(self.job0, fit, pitch_deg=p_, **kw) for p_ in (abs(pitch), -abs(pitch))]
                job, info = max(cands, key=lambda ji: self._origin_above_hit_m(ji[0], float(ji[1]["depth_m"])))
            else:
                job, info = make_refined_job(self.job0, fit, pitch_deg=pitch, **kw)
        except (ValueError, JobError) as e:
            self.res.notes.append(f"refined job after pass {k} could not be built: {e}")
            return None
        self._decide("job", f"job_refined (pass {k}): root start {np.round(job.start, 4).tolist()} end {np.round(job.end, 4).tolist()} "
                     f"(ends {info['ends']}) L {job.length_m * 1e3:.2f} mm n={np.round(job.surface_normal, 4).tolist()} standoff "
                     f"{job.standoff_m:.4f} m (plate {info['standoff_plate_m']:.3f} + depth {info['depth_m'] * 1e3:.2f} mm) pitch "
                     f"{job.pitch_deg} leads {job.lead_in_m:.4f}/{job.lead_out_m:.4f} m (laser hit overscan {self.overscan_m * 1e3:.0f} mm)",
                     info=info)
        for sh in info.get("short_overscan") or []:
            msg = (f"WARNING: job_refined (pass {k}): laser hit overscan at the {sh['which']} {sh['end']} only "
                   f"{sh['hit_overscan_m'] * 1e3:+.1f} mm < planned {sh['planned_m'] * 1e3:.0f} mm ({sh['why']})")
            self.res.notes.append(msg)
            self._decide("job", msg, short_overscan=sh)
        self.res.passes[-1]["job_next"] = job_to_dict(job)
        self.res.passes[-1]["job_info"] = _jsonable(info)
        return job

    def _pass_posture_key(self, k: int):
        """패스 k 의 자세 키 (입사 기울기, 피치, 토치 쪽) — pass_ends 기록 (없으면 None)."""
        return next((p.get("posture") for p in self.pass_ends if p.get("k") == k), None)

    def _deliver_direction_combined(self, fit: PassFit, line: SeamLine, P_root, P_top, k: int, ext: Optional[dict],
                                    dirc: dict) -> Optional[ScanJob]:
        """P8 방향 편향으로 설명된 수렴: 납품 끝 = 두 방향의 편향 없는 평균 (direction_consistency ends[X].shift_m — 이번 패스 직선 위에서
        심 방향으로만 옮긴다; 직선 위치·방향·법선은 이번 패스 그대로, 반대 방향 패스와 허용 안에서 같다). r.extent (용접 경로의 sS/sE)
        와 job_refined 의 끝을 옮기고, 이번 패스의 끝·편향을 r.extent['direction_combined'] 에 남긴다 (보고 — 실제 끝 불일치를 가리지
        않게). 다시 만든 납품 작업 (실패하면 None — 그때는 이번 패스 작업 그대로, 노트)."""
        r = self.res
        u = unit(line.u)
        first_is = "E" if line.direction == "E->S" else "S"
        ends_fit = dict((fit.diag or {}).get("ends") or {})
        extent_pass = dict(r.extent or {})
        rec = {"pass": k, "rule": "bias-free mean of the S->E and E->S readings (direction_consistency)", "ends": {}}
        for X in ("S", "E"):
            # dirc·패스 직선·fit 의 끝 이름은 내부 (방문) 순서, r.extent 는 교시 S→E (P2 방문 순서: _user_extent) — 같은 끝을 옮긴다
            Xu = self._user_end(X)
            e = dirc["ends"].get(X) or {}
            if not _fin(e.get("shift_m")) or Xu not in (r.extent or {}):
                continue
            dS = float(e["shift_m"])
            P_old = np.asarray(r.extent[Xu]["point"], dtype=float)
            P_new = P_old + u * dS
            rec["ends"][Xu] = {"pass_point": P_old.tolist(), "shift_mm": dS * 1e3,
                               "bias_mm": float(e["bias_m"]) * 1e3, "bias_normal_mm": float(e["bias_normal_m"]) * 1e3,
                               "passes_SE": e.get("passes_SE"), "passes_ES": e.get("passes_ES"), "internal_end": X}
            r.extent[Xu] = {**r.extent[Xu], "point": P_new.tolist()}  # S→E 계약 (P1): 끝은 base 점만 — 축·좌표는 소비자가 S, E 에서 유도
            w = "first" if X == first_is else "last"
            if w in ends_fit:
                ends_fit[w] = {**ends_fit[w], "point": P_new.tolist()}
        if not rec["ends"]:
            return None
        rec["pass_length_mm"] = r.extent.get("length_mm")
        r.extent["length_mm"] = float(np.linalg.norm(np.asarray(r.extent["E"]["point"], dtype=float)
                                                     - np.asarray(r.extent["S"]["point"], dtype=float))) * 1e3
        r.extent["direction_combined"] = _jsonable(rec)
        self._decide("ends", f"pass {k}: ends delivered as the bias-free mean of both directions: "
                     + ", ".join(f"{X} {v['shift_mm']:+.2f} mm along the seam (direction bias {v['bias_mm']:+.2f} mm)"
                                 for X, v in rec["ends"].items())
                     + f"; length {rec['pass_length_mm']:.2f} -> {r.extent['length_mm']:.2f} mm", combined=rec)
        fit_c = dataclasses.replace(fit, diag={**(fit.diag or {}), "ends": ends_fit}, flags=list(fit.flags))
        jf = self._final_job(fit_c, line, P_root, P_top, k, ext)
        if jf is None:
            r.extent = extent_pass                       # 작업과 용접 경로가 같은 끝을 쓰게: 이번 패스 끝 그대로
            r.extent["direction_combined"] = _jsonable({**rec, "delivered": False})
            r.notes.append(f"pass {k}: the job with the bias-free ends could not be built - the pass {k} ends are delivered "
                           "(direction bias reported in extent.direction_combined)")
        return jf

    def run(self) -> AutoTeachResult:
        r = self.res
        job_final: Optional[ScanJob] = None
        P_root = P_top = None
        feats: Optional[dict] = None
        fm: dict = {}
        try:
            if self._job_at:
                self._decide("setup", f"job file auto_teach overrides (this run only): {self._job_at}", overrides=_jsonable(self._job_at))
            self._gates()
            self._acquire()
            if self.dry_run:
                r.status, r.reason = STATUS_DRY_RUN, "dry run: gates + acquisition plan (no motion, no sensor)"
                self._p(r.reason)
                return r
            if self.at.post_scan_worker:
                self._warm_future = _post_scan.warm_post_scan_worker(self.at.feature_workers_effective)   # 첫 패스 전에 데운다
            # ---- 2. S (P2): 첫 드웰 (장면 기록만) → S 추적 (심 방향) → probe X → 회전 정렬 (판 법선·그루브 점 피벗) → 수직축 보정 ----
            loc0 = self._first_S()
            self._clear_cell()                                   # 관절 공간 이동 끝: 이후 직선 이동은 포락선 규칙 (용접 경로 도달성 검사가 시편 상자에 걸리지 않게)
            est_S, loc_p, _rows_S, _trk = self._seek_S(loc0)
            loc_al, align = self._align_S(est_S, loc_p)
            self.res.yaw_align["align"] = _jsonable(align)
            site_S, loc_S = self._finish_S(est_S, loc_al, align)
            # 센서 x 스케일의 S 읽기: 보정 시작 (probe X 드웰) → 사이트 사이가 순수 평행이동일 때만 (회전했으면 제외 — 4.8 절)
            self._S_pair = None if align.get("rotated") else (loc_p, loc_p, loc_S, est_S.w)
            # ---- 3. 트랜싯: 같은 자세로 측정 심을 따라 (스트리밍, 이동 중 그루브 추적) ----
            r_S = self._loc_range_m(loc_S, float(self.job_c.standoff_m))
            tr = self._transit_to_E(site_S, loc_S, est_S, align)
            # ---- 4. E: 트랜싯 일관 단면에 드웰 (회전 없음) · 수직축 보정 ----
            loc_E_arr, loc_E_pre, pre_name, est_T, _place = self._place_E(tr, est_S, align, r_S)
            seam_E = est_T if est_T is not None else est_S
            loc_E, _ = self._correct_until_converged("E", loc_E_pre, self._coarse_plate_m(), seam=seam_E, scene=self._scene_E(est_T))
            yaw_E = _st.yaw_to_seam_rad(np.asarray(loc_E.T_base_opt, dtype=float)[:3, :3], seam_E.n, seam_E.u)
            site_E, loc_E = self._site_record("E", loc_E, self._coarse_plate_m(), scene=self._scene_E(est_T),
                                              extra={"tool_yaw_vs_seam_deg": math.degrees(yaw_E), "seam_u": np.asarray(seam_E.u, dtype=float)})
            # ---- 5. 직선 L1 + 일관성 검사 ----
            plan_line = self._line_from_sites(site_S, site_E, loc_S, loc_E_arr, loc_E_pre, loc_E, pre_name,
                                              yaw={"est_S": est_S, "est_T": est_T, "align": align})
            # ---- 6~10. 지그재그 패스: 패스 1 = 보정된 E (오버스캔 없음) → S_ext (R + 착점 오버스캔), 패스 2 = 전체 길이 ----
            L_design = float(self.spec.weld_length_mm) * 1e-3 if (self.spec is not None and self.spec.weld_length_mm > 0) else None
            plan = plan_first_pass(plan_line, design_length_m=L_design, overscan_m=self.overscan_m,
                                   lower_end_first=self.orientation == "vertical_3G")
            if self.orientation == "vertical_3G":
                cap_msg = cap_lower_target(plan, plan_line, self.up, self.job0.table_z_m, below_m=float(self.overscan_m),
                                           measured=self._support_measured())
                if cap_msg:
                    self._decide("ends", cap_msg)
            self._decide("ends", f"pass 1 plan: L_SE {plan_line.length_m * 1e3:.1f} mm, design {(L_design or float('nan')) * 1e3:.1f} mm "
                                 f"-> {plan.how_S}; E start = {plan.how_E}", plan=plan.to_dict())
            prev_line: Optional[SeamLine] = None
            conv_chain: list = []                           # 비교 가능한 패스 직선 사슬 (prev_line 의 역사; 방향을 가른 수렴 P8)
            prev_status: Optional[str] = None
            final_status = STATUS_DATA_FAIL
            depth_m = self._depth_m(None)
            cur: Optional[SeamLine] = None
            while self.n_pass < self.max_passes:
                self.n_pass += 1
                k = self.n_pass
                direction = plan.direction
                rescan = plan.rule == "rescan_quality" and self._rescan_prepared is not None
                end_repeat, self._end_repeat_pending = self._end_repeat_pending, False
                # 3G 입사 분리: 패스 2 부터 주 각도 — 끝 패스 반복 (아래 끝을 아직 못 쟀다) 은 끝 자세 그대로
                main = self.tilt_main is not None and k >= 2 and not rescan and not end_repeat
                if rescan:
                    job_k, ov = self._rescan_prepared
                    self._rescan_prepared = None
                else:
                    posture = None
                    if main:
                        posture, stop = self._main_posture(plan_line, plan, k, depth_m)
                        if stop:
                            self.n_pass -= 1                     # 이 패스는 보내지 않았다
                            final_status = STATUS_REFUSED if "REFUSED (reach)" in stop else STATUS_DATA_FAIL
                            r.reason = stop
                            break
                    plan_k = plan
                    if main and self.end_pass_ref is not None:
                        # 끝 패스와 다른 자세: 그 자세가 본 끝을 이 자세는 자세 간 오프셋만큼 어긋나게 본다 — 그 몫을 오버스캔에 더한다
                        # (이 패스의 사본에만: 반복 패스가 같은 계획을 다시 쓰면 몫이 쌓이지 않게 — 4d 검토)
                        add, measured_add = self._posture_allowance_m()
                        if add > 0.0:
                            plan_k = dataclasses.replace(plan, **{f"ov_{X}": float(getattr(plan, f"ov_{X}")) + add
                                                                   for X in ("S", "E") if float(getattr(plan, f"ov_{X}")) > 0.0})
                        if add > 0.0:
                            self._decide("overscan", f"pass {k}: main posture differs from end pass {self.end_pass_ref['pass']} - laser hit "
                                                     f"overscan + {add * 1e3:.1f} mm for the posture-dependent offset "
                                                     + ("(measured with this posture)" if measured_add else
                                                        "(not measured yet: the recorded posture offset bound)"))
                    job_k = self._pass_job(plan_line, plan_k, k, depth_m, main=main, posture=posture)
                    job_k, ov = self._pass_overscan(job_k, plan_k, k, depth_m, main=main)
                self._move_to_pass_start(ov["T_from"], job_k, k)
                name = f"pass{k}_{'E_to_S' if direction == 'E->S' else 'S_to_E'}"
                self._event("pass_started", k=k, name=name, direction=direction)
                L_leg = float(np.linalg.norm(ov["origin_to"] - ov["origin_from"]))
                _run, result, pass_dir = self._streamed_leg(
                    name, ov["T_to"], job_k, f"pass_{k}", speed=float(job_k.speed_mps),
                    fit_kwargs=self._fit_kwargs(job_k), recon_required=False,
                    extra=f"(pass {k} {direction}, {L_leg * 1e3:.0f} mm at {job_k.speed_mps * 1e3:.0f} mm/s, laser hit "
                          f"{ov['overscan_first_m'] * 1e3:.0f}/{ov['overscan_last_m'] * 1e3:.0f} mm beyond the ends)")
                if result is None:
                    # 4d P9: 재구성 실패는 실패한 읽기 (래치하는 중단이 아니다) — 반대 방향으로 한 번 더, 두 번 연속이면 멈춘다
                    qc_f = list(getattr(_run.qc, "failures", []) or []) if _run is not None and _run.qc is not None else []
                    final_status = STATUS_DATA_FAIL
                    r.reason = f"pass {k}: reconstruction failed: {qc_f}"
                    if (self._recon_failed_prev or self.n_pass >= self.max_passes
                            or (self._floor_unmeasured() and not self._try_end_repeat(k, f"reconstruction failed ({qc_f})"))):
                        r.reason += " (stopped: " + ("second consecutive failure" if self._recon_failed_prev else
                                                     "no pass budget left" if self.n_pass >= self.max_passes else
                                                     "3G lower end not measured yet; " + self._end_repeat_why_not()) + ")"
                        break
                    self._recon_failed_prev = True
                    self._decide("gate", f"pass {k}: reconstruction failed ({qc_f}) -> pass {k + 1} in the opposite direction (confirm-first)"
                                         + (" as a full pass (the failed pass was a partial quality rescan)" if rescan else ""))
                    plan = (self._plan_next(plan_line, k, direction) if rescan else
                            dataclasses.replace(plan, direction="S->E" if direction == "E->S" else "E->S", rule="repeat (reconstruction)"))
                    continue
                self._recon_failed_prev = False
                v_pass = float((result.meta or {}).get("v_ref_mps", float("nan")))
                if not (_fin(v_pass) and v_pass > 0.0):
                    v_pass = float(job_k.speed_mps)
                pr = self._refine_pass(result, job_k, k, direction, pass_dir, ov, plan_line, v_pass, main=main)
                self._emit_pass_cloud(k, name)                     # 처리 뒤 (판정 다음): 다 만들어지면 이벤트 (시각화, 경로 밖)
                fit, verdict = pr["fit"], pr["verdict"]
                fm = fit.feature_medians
                P_root, P_top = pr["P_root"], pr["P_top"]
                feats = _features_arrays(pr["sf"], P_root, P_top, fit, _extent_frame(pr["extent"], fit.n) if fit.ok else None)
                self._record_quality(k, job_k, fit, feats, full=not rescan)
                cur = pr["line"]
                nxt_dir = "S->E" if direction == "E->S" else "E->S"
                adaptive = self.orientation == "vertical_3G" and self.at.posture_3g_adaptive and self.tilt_main is not None
                if adaptive and (main or rescan):
                    self._note_posture_outcome(k, cur is not None or rescan)
                switch = bool(adaptive and main and self.main_posture is not None
                              and self._bad_posture(self.main_posture["phi"]) is not None)   # 다음 주 패스는 다른 자세
                if main and self.carry_refused:
                    # 아래 끝을 끝 패스에서 이어받지 못했고 (자세 간 오프셋이 한계 밖 / 잴 수 없음) 이 자세는 테이블 때문에 아래 끝에 못 닿는다.
                    # 확인 먼저 (4d P8): 첫 거부는 기록하고 다음 주 패스가 공유 위 끝에서 오프셋을 다시 잰다; 다시 거부되면 멈춘다 (같은 자세를
                    # 더 반복해도 같은 결과 — g3_13 은 품질 98 % 패스를 4 회 반복했다). 예산의 마지막 패스에서 처음 거부되면 '확인 안 됨'.
                    w_ = self._lower_end_key(fit) if fit.ok else None
                    if not (w_ and (((fit.diag or {}).get("ends") or {}).get(w_) or {}).get("observed")):
                        if self._carry_refused_once or self.n_pass >= self.max_passes or self._carry_refusal_final:
                            final_status = STATUS_DATA_FAIL
                            first = self._carry_refused_first if self._carry_refused_once else None
                            r.reason = (f"3G pass {k}: lower specimen end not available - {self.carry_refused}; "
                                        + (f"confirmed by a second main pass (first: {first}), so more passes would repeat the same "
                                           "result (stopped)" if self._carry_refused_once else
                                           "the end pass did not see its upper end, so another main pass cannot measure the offset "
                                           "(stopped)" if self._carry_refusal_final else
                                           "unconfirmed (no pass budget left to re-measure the posture offset)"))
                            self._decide("gate", r.reason)
                            break
                        self._carry_refused_once = True
                        self._carry_refused_first = f"pass {k}: {self.carry_refused}"
                        self._decide("gate", f"pass {k}: lower end not carried ({self.carry_refused}) - the next main pass measures the "
                                             "posture offset again before stopping (confirm-first)")
                    else:
                        self._carry_refused_once = False
                elif main:
                    self._carry_refused_once = False
                if rescan and self.last_full is not None:
                    # 적응형 재스캔 (부분 패스): 양끝·직선은 마지막 전체 패스, 특징은 구간별 최선 자료로 다시 판정
                    verdict, feats_m, fm_m = self._merged_gate(k, self.last_full)
                    if feats_m is not None:
                        feats, fm = feats_m, fm_m
                    if verdict.status == STATUS_PASS:
                        final_status = STATUS_PASS
                        r.converged = "rescan"
                        r.reason = f"pass {k} (quality rescan) PASS with the merged features (ends from pass {self.last_full['k']})"
                        break
                    nxt = self._plan_rescan(k) if self.n_pass < self.max_passes else None
                    if nxt is not None:
                        plan = nxt
                        continue
                    final_status = verdict.status if verdict.status in (STATUS_SPEC_FAIL, STATUS_DATA_FAIL) else STATUS_DATA_FAIL
                    r.reason = f"pass {k} (quality rescan) {verdict.status}: failed={verdict.failed} (merged features)"
                    break
                if cur is None:
                    final_status = STATUS_DATA_FAIL
                    r.reason = f"pass {k}: {verdict.summary()} fit flags={fit.flags}"
                    if self.n_pass >= self.max_passes:
                        break
                    if self._floor_unmeasured():
                        if not self._try_end_repeat(k, f"no seam line fitted ({fit.flags})"):
                            r.reason = (f"3G: pass {k} ({direction}) fitted no seam line ({fit.flags}) - the lower specimen end could "
                                        f"not be measured, so no floor for further passes ({self._end_repeat_why_not()})")
                            break
                        prev_status = verdict.status
                        plan = dataclasses.replace(plan, direction=nxt_dir, rule="repeat end pass (no fit)")
                        continue
                    self._decide("gate", f"pass {k}: no seam line fitted ({fit.flags}) -> pass {k + 1} in the opposite direction "
                                         "with the previous line" + (" and another posture" if switch else ""))
                    prev_status = verdict.status
                    plan = dataclasses.replace(plan, direction=nxt_dir, rule="posture switch (no fit)" if switch else "repeat (no fit)")
                    continue
                depth_m = self._depth_m(fm)
                r.line_history.append(cur.to_dict())
                ext = pr["extent"]
                r.extent = _jsonable(self._user_extent({"pass": k, "direction": direction, "S": ext["S"], "E": ext["E"], "length_mm": ext["length_mm"],
                                      "plate_length_mm": ext.get("plate_length_mm"),
                                      "spacing_mm": ext["spacing_mm"], "hit_range_SE_m": ext["hit_range_SE"],
                                      "design_length_mm": self.spec.weld_length_mm if self.spec is not None else None}))
                if self.orientation == "vertical_3G":
                    self._note_support(fit, k)
                    stop = self._update_3g_floor(cur, ext, ov, k, direction)
                    if stop:
                        # 4d P9 F-16/F-17: 원인을 나눈다 — 레이저가 기대 아래 끝 (위 끝 − 설계 길이) 을 지나갔는데도 끝이 안 보였으면 자료 쪽
                        # (끝 패스 한 번 반복), 바닥 (대략 아래 점) 때문에 거기까지 못 갔으면 도달 쪽 (바로 멈춤: 아래 점을 다시 교시)
                        passed = self._lower_end_expected_passed(cur, ext, ov)
                        if passed is not None:
                            if self.n_pass < self.max_passes and self._try_end_repeat(k, f"lower specimen end not identified although "
                                                                                          f"{passed}"):
                                prev_status = verdict.status
                                plan = dataclasses.replace(plan, direction=nxt_dir, rule="repeat end pass (lower end not identified)")
                                continue
                            stop = (f"3G: pass {k} ({direction}) did not identify the lower specimen end although {passed} - "
                                    f"{self._end_repeat_why_not()} (check the lower plate end / fixtures)")
                        final_status = STATUS_DATA_FAIL
                        r.reason = stop
                        break
                jf = self._final_job(fit, cur, P_root, P_top, k, ext)
                if jf is not None:
                    job_final = jf
                self.last_full = {"k": k, "fit": fit, "extent": ext, "residuals": pr["residuals"], "fm": dict(fm or {}), "line": cur,
                                  "coverage": pr.get("coverage"), "end_lag_m": pr.get("end_lag_m", 0.0)}
                if sum(1 for e in self.pass_quality if e["feats"] is not None) >= 2:
                    v2, feats_m, fm_m = self._merged_gate(k, self.last_full)
                    if feats_m is not None:
                        verdict, feats, fm = v2, feats_m, fm_m
                failed_ends = [c_ for c_, (_v, _l, ok_) in verdict.checks.items()
                               if ok_ is False and (c_ == "end_ambiguous" or c_.startswith("end_occluded"))]
                # 가려진 끝의 확인 (4d P9): 첫 읽기 뒤의 패스가 그 끝을 관측했으면 기록을 지우고, 다시 가렸거나 못 봤으면 멈춘다
                for X_, k0_ in list(self._occluded_seen.items()):
                    if k0_ == k:
                        continue
                    accepted = bool(self.at.accept_inferred_end and ext[X_].get("inferred_point") is not None)
                    if bool(ext[X_]["observed"]) or accepted:
                        del self._occluded_seen[X_]              # 관측됐거나 (추정을 켠 실행에서) 추정 끝이 받아들여졌다
                    elif not any(c_.startswith("end_occluded") for c_ in failed_ends):
                        failed_ends.append(f"end_occluded_then_unobserved_{X_}")
                terminal = [c_ for c_ in failed_ends if c_ == "end_ambiguous" or c_.startswith("end_occluded_then_unobserved")
                            or self._occlusion_terminal(ext, direction, c_, k)]
                confirm_ends = False
                amb = self._ambiguity_reading(verdict, ext, fit, k) if "end_ambiguous" in terminal else None
                if amb is None and "end_ambiguous" not in failed_ends and ext["S"]["observed"] and ext["E"]["observed"]:
                    self.end_ambiguous_seen = None                # 깨끗한 읽기 뒤에는 처음부터 (4d P9)
                if amb is not None:
                    seen = self.end_ambiguous_seen
                    confirmed = bool(seen is not None and amb["own"] and seen["sign"] == amb["sign"]
                                     and abs(amb["length_mm"] - seen["length_mm"]) <= float(self.settings.judgement_k) * math.hypot(
                                         seen["sigma_mm"], amb["sigma_mm"]))
                    if not confirmed:
                        terminal = [c_ for c_ in terminal if c_ != "end_ambiguous"]
                        if self.n_pass < self.max_passes:
                            self.end_ambiguous_seen = amb
                            confirm_ends = True
                            self._decide("gate", f"pass {k}: {'; '.join(n_ for n_ in verdict.notes if 'end_ambiguous' in n_)} - "
                                                 + ("not stopping on the first reading" if seen is None else
                                                    "not confirmed by this reading (sign, own ends or length differ)")
                                                 + ": the next pass measures the ends again and the run stops only if it is also off")
                        else:
                            final_status = STATUS_DATA_FAIL
                            r.reason = (f"pass {k}: end_ambiguous on the last budgeted pass - unconfirmed: "
                                        f"{'; '.join(n_ for n_ in verdict.notes if 'end_' in n_)}")
                            break
                if terminal:
                    # 확인된 읽기 (두 번째) — 다시 스캔해도 바뀌지 않는다: 길이 교차 검사 불일치 (일반 일관성 플래그) / 판 위 물체가 끝을 가림
                    final_status = STATUS_SPEC_FAIL if "end_ambiguous" in terminal else STATUS_DATA_FAIL
                    r.reason = (f"pass {k}: {', '.join(terminal)} - {'; '.join(n_ for n_ in verdict.notes if 'end_' in n_)} "
                                "(check the specimen ends / fixtures; the measured ends were not substituted)")
                    break
                end_biased = not (self.pass_quality and _sq.posture_weight(
                    float((self.pass_quality[-1].get("posture") or {}).get("incidence_deg", float("nan")))) >= 1.0)
                if (self.tilt_main is not None and not main and cur.obs_S and cur.obs_E and self.n_pass < self.max_passes
                        and self.lower_end_obs is not None and end_biased):
                    # 3G 입사 분리: 끝 패스 (가파른 각도) 는 양끝·직선·테이블만 — 특징 판정은 주 각도 패스에서
                    self._decide("gate", f"pass {k} at the end incidence {self.tilt_end:+.0f} deg: ends, line and the lower end measured "
                                         f"(verdict {verdict.status} not final - its groove features are biased at this incidence) -> "
                                         f"pass {k + 1} " + ("in the main posture (chosen next)" if adaptive else
                                                            f"at the main incidence {self.tilt_main:+.0f} deg") + f" ({nxt_dir})")
                    prev_status = None
                    prev_line = None                       # 끝 패스는 다른 자세 — 주 패스의 수렴 비교 대상이 아니다 (자세 간 오프셋, g3_13)
                    conv_chain = []
                    plan_line = cur
                    plan = self._plan_next(cur, k, direction)
                    continue
                if (self.tilt_main is not None and not main and end_biased and verdict.status == STATUS_PASS
                        and self.n_pass >= self.max_passes):
                    # 끝 패스 (가파른 입사) 의 특징은 편향 — 뒤따를 주 패스가 없으면 PASS 로 끝내지 않는다 (4d P7)
                    final_status = STATUS_DATA_FAIL
                    r.reason = (f"pass {k}: groove features only from the biased end incidence ({self.tilt_end:+.0f} deg) and no pass "
                                "budget left for a main-posture pass")
                    break
                if confirm_ends:
                    # 끝을 다시 재는 패스 — 수렴 비교 대상은 아니지만 SPEC_FAIL 기록 (끝·길이 외 검사) 은 이어 간다 (4d 검토)
                    prev_status = verdict.status
                    self._prev_spec_failed = set(verdict.failed) - {"end_ambiguous", "weld_length_mm"}
                    prev_line = None
                    conv_chain = []
                    plan_line = cur
                    plan = self._plan_next(cur, k, direction)
                    continue
                if verdict.status != STATUS_PASS and self.n_pass < self.max_passes and self._feature_failure(verdict) and not switch:
                    nxt = self._plan_rescan(k)
                    if nxt is not None:
                        prev_status = verdict.status
                        prev_line = cur
                        conv_chain.append(cur)
                        plan_line = cur
                        plan = nxt
                        continue
                if verdict.status == STATUS_PASS:
                    if prev_line is None:
                        r.converged = "single_pass"
                        r.converged_detail = {"ok": True, "why": f"pass {k} PASS: first pass, nothing to compare", "metrics": {},
                                              "pass": k}
                        final_status = STATUS_PASS
                        r.reason = f"pass {k} PASS (single pass)"
                        break
                    aim = (self.qc_targets.get("root_position_mm") or {}).get("tol") if isinstance(self.qc_targets, dict) else None
                    aim_m = float(aim) * 1e-3 if aim is not None else None
                    ok, why, metrics = line_consistency(prev_line, cur, k=self.settings.judgement_k, aim_tol_m=aim_m)
                    self._decide("converge", f"pass {k} vs pass {prev_line.k}: {why}", metrics=metrics)
                    dirc = None
                    if not ok:
                        # P8 (1G 실기 2026-10-08 b3 t03): 연속 패스 (지그재그 = 반대 방향) 불일치 — 같은 방향 반복도와 방향 편향을 갈라 본다
                        hist = conv_chain + [cur]
                        dirc = direction_consistency(hist, k=self.settings.judgement_k, aim_tol_m=aim_m,
                                                     postures=[self._pass_posture_key(L_.k) for L_ in hist])
                        self._decide("converge", f"pass {k}: {dirc['why']}", state=dirc["state"], metrics=dirc["metrics"])
                        metrics = {**metrics, "direction_state": dirc["state"], "direction": dirc["metrics"]}
                        if dirc["ok"]:
                            ok, why = True, dirc["why"]
                            jf = self._deliver_direction_combined(fit, cur, P_root, P_top, k, ext, dirc)
                            if jf is not None:
                                job_final = jf
                    self.res.passes[-1]["converged"] = {"ok": ok, "why": why, "metrics": metrics}
                    r.converged_detail = {"ok": bool(ok), "why": why, "metrics": metrics, "pass": k}
                    if ok:
                        r.converged = "true"
                        final_status = STATUS_PASS
                        r.reason = f"pass {k} PASS: {why}"
                        break
                    if dirc is not None and dirc["state"] == "bias_exceeds_tol":
                        # 두 방향 모두 같은 방향 쌍으로 반복된 편향 — 패스를 더 돌려도 같은 차이가 반복된다 (수렴 불가): 멈추고 보고
                        r.converged = "false"
                        final_status = STATUS_PASS
                        r.reason = f"pass {k} PASS (not converged: {dirc['why']}; stopped - more passes cannot converge)"
                        r.notes.append(f"direction bias confirmed in both directions and larger than the end tolerance ({dirc['why']}); "
                                       "the delivered job is the last fit - check the sensor time offset (calib dt_s) and the end "
                                       "detection at that end")
                        break
                    if self.n_pass < self.max_passes:
                        self._decide("gate", f"pass {k} PASS but not consistent with pass {prev_line.k} -> one more pass ({nxt_dir})")
                    else:
                        r.converged = "false"
                        final_status = STATUS_PASS
                        r.reason = f"pass {k} PASS (not consistent with the previous pass: {why}; pass budget exhausted)"
                        r.notes.append(f"successive passes did not agree within k x their measured sigma ({why}); "
                                       "the delivered job is the last fit")
                        break
                elif verdict.status == STATUS_SPEC_FAIL:
                    final_status = STATUS_SPEC_FAIL
                    r.reason = (f"pass {k} SPEC_FAIL: measurement quality (profile-to-profile noise, a dimension > the gross limit "
                                f"from the design = misidentified feature) or length; failed={verdict.failed}")
                    common = sorted(set(verdict.failed) & self._prev_spec_failed)
                    confirmed = bool(prev_status == STATUS_SPEC_FAIL and common)
                    if confirmed and (not switch or self.n_pass >= self.max_passes):
                        r.reason += f" (confirmed: {common} failed in two passes)"
                        break
                    if self.n_pass >= self.max_passes:
                        final_status = STATUS_DATA_FAIL           # 예산의 마지막 패스에서 처음 본 SPEC_FAIL: 확인 안 됨 (4d P9)
                        r.reason = f"pass {k} SPEC_FAIL on the last budgeted pass - unconfirmed; failed={verdict.failed}"
                        break
                    self._prev_spec_failed = set(verdict.failed)
                    self._decide("gate", f"pass {k} SPEC_FAIL ({verdict.failed}) -> "
                                         + (f"pass {k + 1} with another posture ({nxt_dir})" if switch else
                                            f"re-measure once with the refined line ({nxt_dir})"))
                else:
                    final_status = STATUS_DATA_FAIL
                    r.reason = f"pass {k} {verdict.status}: failed={verdict.failed} notes={verdict.notes}"
                    if self.n_pass >= self.max_passes:
                        if not (cur.obs_S and cur.obs_E):
                            r.reason = (f"plate end(s) {[X for X in ('S', 'E') if not getattr(cur, f'obs_{X}')]} not observed within "
                                        f"max_passes={self.max_passes}: " + r.reason)
                        break
                    self._decide("gate", f"pass {k} {verdict.status} ({verdict.failed}) -> pass {k + 1} ({nxt_dir}"
                                         + (", another posture)" if switch else ")"))
                prev_status = verdict.status
                prev_line = cur
                conv_chain.append(cur)
                plan_line = cur
                plan = self._plan_next(cur, k, direction)
            if self.pass_ends and cur is not None:
                r.time_offset = _jsonable(time_offset_estimate(self.pass_ends, cur))
                if _fin(r.time_offset.get("dt_s")):
                    self._decide("time_offset", f"time offset estimate dt = (s_fwd - s_rev)/(2v) = {r.time_offset['dt_s'] * 1e3:+.2f} ms "
                                                f"over {r.time_offset['n_ends']} end(s) (report only; calib dt_s {self.calib.dt_s * 1e3:+.2f} ms)")
            if final_status != STATUS_PASS and not r.reason:
                r.reason = "no pass executed (max_passes too small)"
            if final_status != STATUS_PASS and r.converged is not None:
                r.converged = "false"                      # PASS 가 아니면 수렴으로 보고하지 않는다
            r.status = final_status
            self._p(f"{r.status}: {r.reason}")
            return r
        except _Abort as e:
            self._abort_path(e.status, e.reason, safety=e.safety, category=e.category)
            return r
        except _post_scan.Cancelled as e:                   # S4: 취소가 작업 프로세스 기다림에 닿았다 — 그 기다림에서 바로 끝낸다
            self._abort_path(STATUS_ABORTED, f"run cancelled while waiting for the post-scan worker: {e}")
            return r
        except Exception as e:  # noqa: BLE001 - after a cancel / server shutdown: ABORTED (_unexpected_failure)
            st_, why_, cat_ = self._unexpected_failure(e)
            self._abort_path(st_, why_, category=cat_)
            r.notes.append(traceback.format_exc(limit=6))
            return r
        except BaseException as e:  # noqa: BLE001 - M-4: Ctrl-C / SystemExit 도 중단 경로(센서 RESET, 래치 정지, 부분 저장) 를 지난 뒤 재발생
            self._abort_path(STATUS_ABORTED, f"interrupted: {type(e).__name__}")
            r.notes.append(f"interrupted by {type(e).__name__}: abort path executed (task stop, sensor RESET), outputs written")
            raise
        finally:
            r.elapsed_s = self._clock.monotonic() - self.t0
            self._clear_cell()
            try:
                if r.estop_record is None:
                    r.estop_record = self._latch_record_of_this_run()
            except Exception:  # noqa: BLE001
                pass
            if self.sess is not None and (self.owned_session or self.release_sensor):
                # S6: 센서 파라미터 원본 복원을 출력(autoteach_log.json) 쓰기 **전에** — 결과를 로그에 담는다
                try:
                    r.sensor_restore = self.sess.restore_sensor()
                    if r.sensor_restore is not None:
                        self._p(f"sensor parameters restored: {r.sensor_restore.get('restored')} (n={r.sensor_restore.get('n')}, "
                                f"mismatch={r.sensor_restore.get('mismatch')}, error={r.sensor_restore.get('error')})")
                except Exception as e:  # noqa: BLE001
                    r.notes.append(f"sensor restore failed: {e}")
            try:
                r.quality = [_jsonable({"k": e["k"], "full": e["full"], "posture": e["posture"], **e["fractions"]}) for e in self.pass_quality]
                r.merge_choice = _jsonable(self.merge_choice)
                r.posture_offsets = _jsonable(self.posture_offsets)
                self._write_outputs(job_final, feats, fm)
                self._write_quality_dataset()
            except Exception as e:  # noqa: BLE001
                r.notes.append(f"output write failed: {e}")
            self._event("outputs", out_dir=str(self.out), status=r.status, base_frame=self.job0.base_frame,
                        weld_plan=str(self.out / "weld_plan.npz") if r.weld_plan is not None else None,
                        cross_sections=(r.cross_sections or {}).get("files", [None, None])[1])
            if self.sess is not None and self.owned_session:
                try:
                    self.sess.close()
                except Exception:  # noqa: BLE001
                    pass
            elif self.sess is not None and self.release_sensor and self.sess.sensor is not None:
                try:
                    self.sess.sensor.close()             # 센서만 닫는다 (로봇 플랫폼은 호출자 소유)
                except Exception:  # noqa: BLE001
                    pass
                self.sess.sensor = None
                self.sess.sensor_ready = False
            self._p(r.summary())


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.degrees(math.acos(c))


def run_auto_teach(settings: Settings, job0: ScanJob, doc: Optional[dict], calib: Calib, **kw) -> AutoTeachResult:
    """AutoTeach(...).run() 편의 함수 (인자는 AutoTeach.__init__)."""
    return AutoTeach(settings, job0, doc, calib, **kw).run()


# --------------------------------------------------------------------------------------
# 오프라인 정련 (refine CLI)
# --------------------------------------------------------------------------------------
def _load_raw_for_refine(raw_path) -> tuple:
    """pass_k/raw.npz → (frames, robot samples, job (pass), calib)."""
    from .config import job_from_dict
    with np.load(raw_path, allow_pickle=False) as z:
        X, Z = z["sensor_x_mm"], z["sensor_z_mm"]
        n = z["sensor_n_points"]
        frames = [ProfileFrame(x_mm=X[i, : int(n[i])], z_mm=Z[i, : int(n[i])], ts_ms=int(z["sensor_ts_ms"][i]),
                               result_counter=int(z["sensor_result_counter"][i]), t_rx_s=float(z["sensor_t_rx_ns"][i]) * 1e-9)
                  for i in range(X.shape[0])]
        robot = [RobotSample(t_rx_s=float(z["robot_t_rx_ns"][i]) * 1e-9, T_base_flange=z["robot_T_base_flange"][i])
                 for i in range(z["robot_T_base_flange"].shape[0])]
        job = job_from_dict(json.loads(str(z["job_json"])))
        calib = Calib.from_dict(json.loads(str(z["calib_json"])))
    return frames, robot, job, calib


def _refine_offline_pass(raw_path, doc: dict, settings: Optional[Settings], calib: Optional[Calib], p: Callable[[str], None]) -> dict:
    """저장된 패스 하나 → 온라인과 같은 함수로 fit_pass · 양끝 · 측정 충분성 · fidelity_gate (run_refine_offline 의 패스 단계)."""
    frames, robot, job_k, calib_raw = _load_raw_for_refine(raw_path)
    calib = calib or calib_raw
    spec = specimen_from_doc(doc)
    st = settings or Settings({})
    at = st.auto_teach_config()
    z_valid = settings.z_valid_mm if settings is not None else tuple(Z_VALID_MM_DEFAULT)
    qc_targets = qc_targets_effective(at, doc, z_valid)
    result = reconstruct_scan(frames, robot, calib, job_k, fps_nominal=job_k.profile_fps, z_valid_mm=z_valid)
    fr = sensor_frames(job_k, calib)
    depth_spec = float(spec.groove_depth_mm) * 1e-3 if spec is not None else 0.0
    plate_pass = sensor_view(z_valid).pass_plate_m(depth_spec)            # 온라인과 같은 유도 (센서 모형 + 설계 깊이)
    process = WeldProcess.from_dict(weld_process_doc(doc, job_k.position))
    ghost = st.ghost_burst_frames
    if ghost is None:                                                     # 플랫폼 설정 없이 재판정: 끝 판정 기본 (알린다)
        ghost = EndParams.ghost_burst_frames
        p(f"refine: platform sensor.ghost_burst_frames not given - using {ghost} (scan_core.localize.EndParams default)")
    epar = end_params_for(job_k, calib, spec, process, k=st.judgement_k, ghost_burst_frames=ghost)
    sf, fit, P_root, P_top, residuals = fit_pass(result, spec, u_prior=job_k.unit_dir, n_prior=job_k.normal_unit,
                                                 stage="fine", u_travel=fr.u_travel,
                                                 target_standoff_m=plate_pass,
                                                 gravity_axis_base=job_k.gravity_axis_base, end_params=epar,
                                                 process=process,
                                                 feature_workers=at.feature_workers_effective, qc_targets=qc_targets)
    rough = doc.get("job", {}) if isinstance(doc, dict) else {}
    u_SE = job_k.unit_dir
    try:
        d_SE = np.asarray(rough["end"], dtype=float) - np.asarray(rough["start"], dtype=float)
        if float(np.linalg.norm(d_SE)) > 1e-6:
            u_SE = unit(d_SE)
    except (KeyError, TypeError, ValueError):
        pass
    direction = "S->E" if float(job_k.unit_dir @ u_SE) >= 0.0 else "E->S"
    extent = pass_extent_summary(fit, direction, u_SE) if fit.ok else None
    lag = end_zone_lag_m(job_k, calib, depth_spec, fit.n if fit.ok else None)
    feats = _features_arrays(sf, P_root, P_top, fit, _extent_frame(extent, fit.n) if fit.ok else None)
    cov = pass_coverage(feats, float(feats["s_a_m"]), float(feats["s_b_m"]), process, lag) if fit.ok else None
    validity = {"z_valid_mm": tuple(z_valid), "pose_tol_mm": float(job_k.pose_tol_m) * 1e3,
                "plate_min_mm": float(_sfmod.plate_min_len_mm(spec))}
    verdict = fidelity_gate(fit.feature_medians, residuals, spec, "fidelity", qc_targets=qc_targets, extent=extent,
                            accept_inferred_end=at.accept_inferred_end, k=st.judgement_k, coverage=cov, validity=validity)
    return {"sf": sf, "fit": fit, "P_root": P_root, "P_top": P_top, "residuals": residuals, "job_k": job_k, "calib": calib, "spec": spec,
            "at": at, "st": st, "qc_targets": qc_targets, "process": process, "ghost": ghost, "plate_pass": plate_pass, "u_SE": u_SE,
            "direction": direction, "extent": extent, "lag": lag, "feats": feats, "validity": validity, "verdict": verdict,
            "rough": rough}


def run_refine_offline(raw_path, doc: dict, settings: Optional[Settings] = None, *,
                       calib: Optional[Calib] = None, log: Optional[Callable[[str], None]] = None, merge_with=None) -> tuple:
    """저장된 패스 raw.npz + 작업 문서(specimen/qc_targets, 거친 start/end = S→E 방향) → (job_refined, info, fit, verdict, residuals).
    run_auto_teach 와 같은 함수 (fit_pass → identify_seam_ends → fidelity_gate → make_refined_job) 로 정련하므로 같은 입력이면 같은 job 을
    재현한다 (RANSAC seed 고정). raw 안의 job_json 이 그 패스의 작업이며 기준 작업이 된다 (progression_sign 은 문서의 거친 job 값).
    판정 기준은 온라인과 같다 (k = 설정 judgement.k, 단면 충분성 = 문서 weld_plan 블록, 데이터 유효성 = 센서 창 + 그 패스의 도착 허용).

    merge_with = 같은 실행의 앞 패스 raw.npz 경로 (패스 순서; 부분 품질 재스캔은 (경로, False)) — 주면 온라인처럼 패스마다 품질을 기록하고
    raw_path (마지막 전체 패스 = 기준) 의 관측 끝 S, E 좌표로 구간별 최선 자료를 병합해 (merged_pass_features) 병합 중앙값으로 다시 판정하고
    그 병합 특징으로 용접 경로를 만든다 (온라인 결과물 재현; 패스 번호는 순서대로 1, 2, …, 기준이 마지막)."""
    p = log or (lambda s: None)
    calib_override = calib                                    # 주면 모든 패스에 (아니면 패스마다 raw 의 캘리브)
    R = _refine_offline_pass(raw_path, doc, settings, calib_override, p)
    sf, fit, P_root, P_top, residuals = R["sf"], R["fit"], R["P_root"], R["P_top"], R["residuals"]
    job_k, calib, spec, process, extent, verdict = R["job_k"], R["calib"], R["spec"], R["process"], R["extent"], R["verdict"]
    rough, u_SE, direction = R["rough"], R["u_SE"], R["direction"]
    p(f"refine: {sf.n_ok}/{sf.n_profiles} groove profiles, fit ok={fit.ok} flags={fit.flags}; {verdict.summary()}"
      + (f"; length {extent['length_mm']:.2f} mm" if extent else ""))
    spec_rep = specimen_report(fit, verdict, extent, 0, direction, spec)
    for line in format_specimen_table(spec_rep) if spec_rep is not None else []:
        p(line)
    kw = job_to_dict(job_k)
    kw["torch_lateral_m"] = tuple(kw["torch_lateral_m"])
    try:
        kw["progression_sign"] = int(rough.get("progression_sign", 1) or 1)
    except (TypeError, ValueError):
        kw["progression_sign"] = 1
    job_base = ScanJob(**kw)
    job, info = make_refined_job(job_base, fit, standoff_plate_m=R["plate_pass"], pitch_deg=float(job_k.pitch_deg), calib=calib,
                                 overscan_m=overscan_need_m(job_k, R["ghost"])[0], u_SE=u_SE, rough_start=rough.get("start"),
                                 rough_end=rough.get("end"), plate_top_m=job_k.plate_top_measured_m, spec=spec,
                                 job_id=(job_k.job_id or "job").split("_pass")[0].replace("_refined", "") + "_refined",
                                 top_polyline=P_top, root_polyline=P_root,
                                 R_ref=sensor_frames(job_k, calib).T_base_flange(0.0)[:3, :3])     # 그 패스의 툴 자세
    info["extent"] = extent
    info["specimen_report"] = spec_rep
    info["weld_plan"] = None
    feats = R["feats"]
    k_judge = R["st"].judgement_k
    if merge_with and fit.ok and extent is not None:
        entries = []
        for i, item in enumerate(list(merge_with), start=1):
            path_i, full_i = (item if isinstance(item, (tuple, list)) else (item, True))
            Ri = _refine_offline_pass(path_i, doc, settings, calib_override, lambda s: None)
            entries.append(pass_quality_entry(i, Ri["job_k"], Ri["calib"], Ri["fit"], Ri["feats"], spec, R["qc_targets"], process,
                                              k_judge, bool(full_i)))
        entries.append(pass_quality_entry(len(entries) + 1, job_k, calib, fit, feats, spec, R["qc_targets"], process, k_judge, True))
        merged, choice, bin_m = merged_pass_features(entries, extent, fit, process)
        if merged is not None:
            fm2 = dict(fit.feature_medians or {})
            fm2.update({k_: v for k_, v in _sq.merged_medians(merged).items() if np.isfinite(v)})
            cov_m = pass_coverage(merged, float(merged["s_a_m"]), float(merged["s_b_m"]), process, R["lag"])
            verdict = fidelity_gate(fm2, residuals, spec, "fidelity", qc_targets=R["qc_targets"], extent=extent,
                                    accept_inferred_end=R["at"].accept_inferred_end, k=k_judge, coverage=cov_m, validity=R["validity"])
            feats = merged
            info["merge_choice"] = choice
            p(f"refine: merged {len(entries)} passes (best pass per {bin_m * 1e3:.0f} mm section from S; passes "
              f"{sorted({c_['pass'] for c_ in choice})}) -> {verdict.summary()}")
    if extent is not None and fit.ok:
        try:
            info["weld_plan"] = weld_plan_from_features(feats, extent, spec, process,
                                                        base_frame=job_k.base_frame, position=str(job_k.position or "1G"),
                                                        gravity_axis_base=job_k.gravity_axis_base, k=k_judge,
                                                        design=weld_design_from_doc(doc, process))
            for line in info["weld_plan"].summary_lines():
                p(line)
        except Exception as e:  # noqa: BLE001
            info["weld_plan_error"] = f"{type(e).__name__}: {e}"
            p(f"weld plan failed: {info['weld_plan_error']}")
    p(f"refined: root start {np.round(job.start, 4).tolist()} end {np.round(job.end, 4).tolist()} n={np.round(job.surface_normal, 4).tolist()} "
      f"standoff {job.standoff_m:.4f} m")
    return job, info, fit, verdict, residuals
