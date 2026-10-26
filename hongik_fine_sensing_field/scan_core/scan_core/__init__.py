"""scan_core — ROS 비의존 스캔 알고리즘 (numpy/scipy/pyyaml 만 사용).

모듈:
  types        ScanJob / Calib / ProfileFrame / RobotSample / ScanResult / QCReport
  frames       4x4 변환, RPY(R = Rz·Ry·Rx) 유틸
  path_gen     sensor_frames, build_transit_plan, check_envelope (EnvelopeError)
  clock_sync   CounterClock, ts_integrity_flags, PoseBuffer, TransformBuffer
  validity     valid_mask, dX_mm, column_monotonic_mask
  reconstruct  profile_to_base, reconstruct_scan, assemble
  quality      integrity_facts, integrity_report, ransac_plane
  synth        합성 데이터 (Scene, GrooveScene, simulate_scan)
  specimen     GrooveSpec (그루브 시편 공칭 단면·치수·YAML)
  seam_features 단면 특징 추출 (판/그루브/베벨/갭/hi-lo, 스캔 검증용) + 심 폴리라인
  sensor_model VC nano 3D-Z large FOV(z)/dx(z)/스탠드오프 선택
  calib        hysteresis(dt), sphere(hand-eye)
  io_npz       save/load ScanResult
  proc_pool    OwnedPool — 프로세스 풀의 수명 규칙 (끝은 쓰는 쪽만 든다, 죽음은 프로세스 상태로)
  localize     적응 자가 티칭 핵심 수학: 정적 국소화(기울기 인지 높이/거리) · 상대 보정 · 심 방향 · 패스 정련 · 충실도 게이트
  seam_frame   S→E 좌표 계약: 끝 = base 점 S, E; 축 = unit(E − S); s = S 에서 잰 거리 (병합·용접 계획·시편 보고가 공유)
"""
from .types import ScanJob, Calib, ProfileFrame, RobotSample, ScanResult, QCReport, JobError, IncidenceTiltWarning
from .path_gen import sensor_frames, build_transit_plan, check_envelope, EnvelopeError, ScanFrames, TransitPlan, Waypoint
from .clock_sync import CounterClock, ts_integrity_flags, PoseBuffer, TransformBuffer
from .validity import valid_mask, dX_mm, column_monotonic_mask
from .reconstruct import profile_to_base, profiles_to_base, reconstruct_scan, assemble
from .quality import integrity_facts, integrity_report, ransac_plane
from .specimen import GrooveSpec
from .seam_features import (CrossSectionFeatures, SeamScanFeatures, extract_cross_section_features,
                            extract_seam_features_along_scan)
from .sensor_model import SensorModel, VC_NANO_3DZ_LARGE, choose_standoff
from .localize import (median_profile, StaticLocalization, localize_static, Correction, relative_correction, apply_correction,
                       seam_from_pair, PassFit, refine_from_pass, Verdict, fidelity_gate)

__version__ = "0.1.0"

__all__ = [
    "ScanJob", "Calib", "ProfileFrame", "RobotSample", "ScanResult", "QCReport", "JobError", "IncidenceTiltWarning",
    "sensor_frames", "build_transit_plan", "check_envelope", "EnvelopeError", "ScanFrames", "TransitPlan", "Waypoint",
    "CounterClock", "ts_integrity_flags", "PoseBuffer", "TransformBuffer",
    "valid_mask", "dX_mm", "column_monotonic_mask",
    "profile_to_base", "profiles_to_base", "reconstruct_scan", "assemble",
    "integrity_facts", "integrity_report", "ransac_plane",
    "GrooveSpec", "CrossSectionFeatures", "SeamScanFeatures", "extract_cross_section_features",
    "extract_seam_features_along_scan", "SensorModel", "VC_NANO_3DZ_LARGE", "choose_standoff",
    "median_profile", "StaticLocalization", "localize_static", "Correction", "relative_correction", "apply_correction",
    "seam_from_pair", "PassFit", "refine_from_pass", "Verdict", "fidelity_gate",
]
