"""설정·캘리브·작업 파일 입출력 (ROS 없이 플랫폼 YAML (hongik_scan_ros/config/*.yaml) 을 그대로 읽는다).

- 설정 YAML: `/**: ros__parameters: {...}` 형식 (hongik_scan_ros/config/hongik_kiro.yaml). 플랫폼 의존 값(IP, 프레임, 치수)은
  코드에 기본값을 두지 않고 이 파일에서만 온다. null / 'null' 문자열은 "미지정"으로 취급한다.
- 캘리브 YAML: eoat_description/calib/eoat_<serial>.yaml (laser_link → profiler_optical_frame xyz/rpy + eoat_yaw + dt_s + x_sign)
  또는 scan_core.Calib.to_dict() 형식(T_flange_opt) 둘 다 허용.
- 작업(job) YAML: {job: ScanJob.to_dict(), specimen: GrooveSpec.to_dict() (선택), qc_targets: {...} (선택), teach: {...},
  meta: {...}} (평면(flat) dict 도 허용). specimen/qc_targets 는 specimen_from_doc / qc_targets_effective 로 읽는다.
- **용접 자세 (welding position)**: job 블록의 `position: 1G | 3G` 가 단일 출처다 (사용자 2026-10-01 3차 결정; 설계 정보와 함께
  주는 명시 입력이며 거친 점 배치에서 추론하지 않는다). 읽을 때 ScanJob.orientation (flat_1G | vertical_3G) 로 옮겨지고,
  save_job 은 `position` 만 쓴다 (orientation 키는 쓰지 않는다). 이전 파일의 명시 orientation flat_1G/vertical_3G 도 그대로 읽힌다;
  position 과 명시 orientation 이 다르면 JobError. orientation 'auto'(= 자세 미지정) 는 auto-teach / scan 이 거부한다
  ('give position: 1G|3G'). CLI teach-start/teach-end/auto-teach --position 1G|3G 가 쓰거나 덮어쓴다.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from scan_core.specimen import GrooveSpec
from scan_core.types import (Calib, g1_block_fields, JobError, ScanJob, orientation_from_position,
                             position_from_orientation)

HERE = Path(__file__).resolve().parent
SCAN_STACK_DIR = HERE.parent.parent                                             # .../src/scan_stack
DEFAULT_CONFIG = SCAN_STACK_DIR / "hongik_scan_ros" / "config" / "hongik_kiro.yaml"   # KIRO 플랫폼 (MoveIt)
DEFAULT_CALIB = SCAN_STACK_DIR / "eoat_description" / "calib" / "eoat_lab_01.yaml"
JOBS_DIR = HERE / "jobs"

_JOB_FIELDS = {f.name for f in dataclasses.fields(ScanJob)}
_SAFETY_JOB_KEYS = ("max_laser_tilt_deg", "torch_protrusion_m")      # 안전 한계 (safe_clearance·standoff_min·pose_tol) 는 scan_l0.safety 가 유도


def unnull(v: Any) -> Any:
    """rcl 파라미터 파서가 null 을 문자열 'null' 로 읽을 수 있어 둘 다 None 으로."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("null", "none", "~", ""):
        return None
    return v


def load_params(path) -> dict:
    """설정 YAML → ros__parameters dict (없으면 최상위 dict 그대로)."""
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if "/**" in d and isinstance(d["/**"], dict) and "ros__parameters" in d["/**"]:
        return dict(d["/**"]["ros__parameters"])
    return dict(d)


@dataclass
class Settings:
    """플랫폼 설정 (hongik_scan_ros/config/*.yaml 의 ros__parameters)."""

    params: dict
    path: str = ""

    # ---- 로봇 ----
    @property
    def platform(self) -> str:
        return str(self.params.get("platform", ""))

    @property
    def adapter(self) -> str:
        """pose_source / motion_backend → 'moveit_jtc' (scan_l0.ports.select_adapter; 그 밖의 값은 PlatformConfigError)."""
        from .ports import select_adapter
        return select_adapter(self.params)

    @property
    def base_frame(self) -> str:
        return str(self.params.get("base_frame", "link0"))

    @property
    def progression_sign(self) -> int:
        return int(self.params.get("progression_sign", 1))

    # ---- 센서 ----
    @property
    def sensor(self) -> dict:
        return dict(self.params.get("sensor", {}))

    @property
    def sensor_ip(self) -> str:
        return str(unnull(self.sensor.get("ip")) or "")

    @property
    def sensor_port(self) -> int:
        return int(self.sensor.get("port", 1096))

    @property
    def z_valid_mm(self) -> tuple:
        from scan_core.validity import Z_VALID_MM_DEFAULT
        z = self.sensor.get("z_valid_mm", list(Z_VALID_MM_DEFAULT))
        return (float(z[0]), float(z[1]))

    @property
    def sensor_columns(self) -> int:
        """프로파일 열 수 = 보내는 ROI 폭 (sensor.init_params.ROI_DX_PIX / SubSample), 없으면 points_expected, 없으면 센서 사양."""
        ip = self.sensor.get("init_params") or {}
        if ip.get("ROI_DX_PIX"):
            return max(1, int(ip["ROI_DX_PIX"]) // max(1, int(ip.get("SubSample", 1) or 1)))
        if self.sensor.get("points_expected"):
            return int(self.sensor["points_expected"])
        from scan_core.sensor_model import VC_NANO_3DZ_LARGE
        return int(VC_NANO_3DZ_LARGE.n_columns)

    @property
    def judgement_k(self) -> float:
        """판정의 통계 배수 k (플랫폼 judgement.k; 사용자 2026-10-07 결정 3 — scan_core.measure.K_DEFAULT)."""
        from scan_core.measure import K_DEFAULT
        v = (self.params.get("judgement") or {}).get("k", K_DEFAULT)
        k = float(v)
        if not (np.isfinite(k) and k > 0.0):
            raise ValueError(f"judgement.k must be > 0, got {v!r}")
        return k

    @property
    def ghost_burst_frames(self) -> Optional[int]:
        """센서 유령점 최대 연속 프레임 (sensor.ghost_burst_frames — 안전 정책의 확인 프레임과 끝 판정의 확인 프로파일이 같은 원천).
        설정에 없으면 None (안전 정책은 거부, 오프라인 재판정은 scan_core.localize.EndParams 기본)."""
        v = self.sensor.get("ghost_burst_frames")
        if v is None:
            return None
        g = int(v)
        if g < 0:
            raise ValueError(f"sensor.ghost_burst_frames must be >= 0, got {v!r}")
        return g

    # ---- 섹션 ----
    @property
    def safety(self) -> dict:
        return dict(self.params.get("safety", {}))

    @property
    def scan_defaults(self) -> dict:
        return dict(self.params.get("scan_defaults", {}))

    @property
    def speed_profile(self) -> dict:
        return dict(self.params.get("speed_profile", {}))

    @property
    def specimen(self) -> dict:
        """specimen 섹션 (포인터: source=job_file → GrooveSpec 은 job YAML 의 specimen 블록에서)."""
        return dict(self.params.get("specimen", {}))

    @property
    def auto_teach(self) -> dict:
        """auto_teach 섹션 원본 dict (없으면 {})."""
        return dict(self.params.get("auto_teach", {}) or {})

    def auto_teach_config(self) -> "AutoTeachConfig":
        """auto_teach 섹션 → 타입 지정 AutoTeachConfig (null/'null' 은 기본값으로)."""
        return AutoTeachConfig.from_params(self.auto_teach)

    @property
    def time_model(self) -> dict:
        return dict(self.params.get("time_model", {}) or {})

    def as_dict(self) -> dict:
        return {"path": self.path, "params": self.params}


RETIRED_SECTIONS = {
    "quality_thresholds": "the per-leg QC verdict was retired (2026-10-08, 4d P2): it decided nothing; each threshold is replaced by a "
                          "derived check (station coverage, refined plate plane, guard G0, pass posture, clock model) and the "
                          "integrity facts are only recorded",
}


def load_settings(path=None) -> Settings:
    p = Path(path) if path else DEFAULT_CONFIG
    params = load_params(p)
    retired = sorted(k for k in params if k in RETIRED_SECTIONS)
    if retired:
        raise ValueError(f"{p}: " + "; ".join(f"{k} is no longer a setting ({RETIRED_SECTIONS[k]})" for k in retired)
                         + " - remove it from the platform config")
    return Settings(params, str(p))


# --------------------------------------------------------------------------------------
# auto_teach 섹션 (적응형 심 자동 티칭: 설계 패널 2026-09-30 §3~§5)
# --------------------------------------------------------------------------------------
def _f(d: dict, key: str, default: float) -> float:
    v = unnull(d.get(key))
    return float(default if v is None else v)


def _i(d: dict, key: str, default: int) -> int:
    v = unnull(d.get(key))
    return int(default if v is None else v)


def _b(d: dict, key: str, default: bool) -> bool:
    v = unnull(d.get(key))
    if v is None:
        return bool(default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


FIDELITY_RETIRED_KEYS = {
    **{k: "design tolerances belong to the job file qc_targets (specimen design), not the platform"
       for k in ("weld_length_tol_mm", "top_opening_tol_mm", "root_gap_tol_mm", "groove_depth_tol_mm", "bevel_angle_tol_deg",
                 "hi_lo_max_mm")},
    "weld_length_ambiguous_mm": "derived: design length tolerance + k x sqrt2 x end sigma (profile spacing / sqrt12)",
    "floor_visible_fraction_min": "derived: every weld-plan section needs its valid profiles (weld_plan station window / min_profiles)",
    "root_edges_fraction_min": "derived: every weld-plan section needs its valid profiles (weld_plan station window / min_profiles)",
    "ok_fraction_min": "derived: every weld-plan section needs its valid profiles (weld_plan station window / min_profiles)",
    "along_seam_robust_std_max_mm": "report only (the specimen's real variation)",
    "measurement_noise_max_mm": "derived: k x profile-to-profile noise must not exceed the design tolerance of that dimension",
    "specimen_gross_max_mm": "derived: a dimension outside the design +-50 % is a misidentified feature",
    "fov_margin_min_mm": "derived: the plate length the feature extractor needs beside the groove",
    "sensor_z_margin_mm": "derived: sensor window minus the pose tolerance (and the groove depth for the floor)",
}


# 작업 파일 (job YAML) 의 auto_teach: 덮어쓰기 — 실행마다 바꿔도 안전한 키만 (안전 여유·한계·캡은 플랫폼 설정에서만)
JOB_AUTO_TEACH_KEYS = ("posture_3g_main", "posture_3g_adaptive", "torch_flip_3g", "max_passes")   # max_passes: the job's pass budget

# 설정에서 뺀 auto_teach 키 (사용자 2026-10-07: 작위적 상수 대신 유도) — 남아 있으면 조용히 무시하지 않고 거부한다
AUTO_TEACH_RETIRED_KEYS = {
    "guard": "this package has no live sensor guard; the sensor start wait is auto_teach.first_frame_timeout_s",
    "coarse_standoff_m": "the coarse range is derived: far edge of the sensor window - pose tolerance - groove depth (scan_l0.sensor_view)",
    "fine_standoff_m": "the pass distance is derived: groove mid-depth at the sensor optical cross distance (scan_l0.sensor_view)",
    "search_steps_mm": "the lateral search is derived: steps of FOV(range) - top opening over the specimen width (scan_l0.sensor_view)",
    "table_margin_m": "every tool-to-surface margin is the safety policy limit (safety.clearance_m + stopping distance)",
    "overscan_m": "the laser-hit overscan is derived: the off-specimen profiles the end detection needs (sensor.ghost_burst_frames "
                  "+ 1) + one profile spacing + pose tolerance",
    "max_end_extension_m": "end extensions follow the design length geometry (no cap; the envelope and the 3G floor rule decide reach)",
    "first_extension_fraction": "pass 1 extends S by the full remaining length R (the farthest the S end can be)",
    "floor_min_overscan_fraction": "the 3G floor rule refuses an end only below the overscan the end detection needs",
    "clamp_height_m": "a clamp is fixture geometry of the specimen: declare it in the job file (job.clamp_height_m); the platform "
                      "safety margin is safety.clearance_m",
    "consistency": "derived: successive lines agree within k x the two passes' root-line residual sigma (position, angle over the "
                   "span) and k x sqrt2 x end sigma for the ends",
    "caps": "corrections move by what was measured (safety = guard + envelope; the measurement is bounded by the view); the 3G "
            "re-acquisition angle comes from the chosen posture's clearance margins",
    "end_detection": "derived (scan_core.localize.EndParams): surface identity = half the design groove depth (plate top vs groove "
                     "floor), end-bracket precision = k x the plate / floor point noise and profile-to-profile jitter measured in the "
                     "pass, root on the line = within half the design gap, the weld-plan station window, the end-zone lag, the sensor "
                     "ghost burst (sensor.ghost_burst_frames) and the extractor's minimum plate length",
    "plate_plane_ransac_thr_m": "derived: RANSAC inliers are within k x the least-median-of-squares scale of the pass (floor = sensor "
                                "quantization dX/sqrt12)",
    "fidelity": "the gate's criteria are derived (scan_core.measure) and the specimen report bins are the weld-plan station window "
                "(job weld_plan.station_window_mm)",
    "tolerances": "dwell convergence is derived (2026-10-08, 4d P3): lateral within the arrival tolerance carried to the laser hit, "
                  "standoff within the arrival tolerance at the optical origin (floor k x sensor quantization); roll is corrected, "
                  "not judged; the coarse psi abort is replaced by a sensor x-scale check against the job's lateral design tolerances",
    "psi_consistency_note_deg": "the seam yaw estimates are recorded only; the sensor x scale is judged directly (4d P4)",
    "foreign_object_height_mm": "derived (4d P5): half the design groove depth above the plate top",
    "foreign_object_min_points": "derived (4d P5): the evidence count of the extractor's minimum plate at the dwell's column spacing",
    "rough_surface_tol_mm": "derived (4d P5): half a plate thickness below / 1.5 x the design depth above the rough point",
    "min_site_distance_m": "derived (4d P3): 2 x the arrival tolerance at the laser hit x design length / pass view room beside the "
                           "groove; the rough points are checked before any motion",
}


def parse_posture_3g(text: str):
    """'35/below' → (35.0, False), '20/above' → (20.0, True), '' → None. 형식이 틀리면 ValueError."""
    t = str(text or "").strip().lower()
    if not t:
        return None
    try:
        a, b = [x.strip() for x in t.split("/")]
        tilt = float(a)
    except ValueError:
        raise ValueError(f"auto_teach.posture_3g_main must be '<incidence deg>/<above|below>', got {text!r}") from None
    if b not in ("above", "below") or not (tilt >= 0.0):                 # 상한은 작업 검증 (센서 입사 한계) 이 거부한다
        raise ValueError(f"auto_teach.posture_3g_main must be '<incidence deg >= 0>/<above|below>', got {text!r}")
    return tilt, b == "above"


@dataclass(frozen=True)
class AutoTeachConfig:
    """auto_teach 섹션의 타입 지정 뷰 (hongik_scan_ros/config/*.yaml, lab/fake/kiro 키 동일)."""

    dwell_frames: int = 50
    max_passes: int = 4                     # 스트리밍 패스 최대 수 (패스 1 = E→S, 이후 지그재그; 2026-10-01 흐름)
    max_corrections: int = 3                # 2026-10-04 2 → 3: 판 옆 기울기 15° 는 롤 스텝 상한 7° × 2 로 못 맞췄다
    first_frame_timeout_s: float = 2.0      # S1: 센서 시작 → 첫 프레임 대기 상한 (실기 수백 ms; 없으면 그 구간을 보내지 않고 REFUSED)
    accept_inferred_end: bool = False       # 판 위 물체가 끝을 가리면 ('end_occluded') 추정 끝 (다른 끝 ± 설계 길이) 을 받아들일지
    feature_workers: int = 0                # 패스 후 프로파일별 특징을 나눠 계산할 프로세스 수: 0 = 자동 (min(8, CPU − 2)), 1 = 같은
                                            # 프로세스에서 순차 (결과는 같다). 2026-10-03 at11: ROS 서버 안 순차 ~100 s (GIL 경합) → 수 초
    post_scan_worker: bool = True           # 패스 뒤 처리 (원자료 저장·재구성·QC·특징·맞춤·끝 판정) 를 전용 작업 프로세스에서 (2026-10-05: 서버
                                            # 프로세스 안에서는 ROS 콜백과의 GIL 경합으로 패스당 12–14 s, 밖에서는 4–5 s). false = 서버 안 (예전)
    quality_dataset_path: str = ""          # 적응형 스캔: 실기 패스별 (자세, 품질) jsonl ('' = 환경 변수 SCAN_QUALITY_DATASET 또는
                                            # ~/scan_data/scan_quality_dataset.jsonl)
    posture_3g_adaptive: bool = True        # 3G 입사 분리의 주 패스 자세 (입사 각도 × 토치 위/아래) 를 예상 품질 (φ 사전값·데이터셋·이번 실행 측정)
                                            # × 시편 길이 대비 닿는 비율로 고르고, 품질이 낮았거나 직선을 못 맞춘 자세는 다시 쓰지 않는다
                                            # (2026-10-06 g3_12: 20° 토치 위 이동 스캔이 반사로 붕괴했는데 같은 자세로 세 번 반복). false = job 각도 고정 (예전)
    posture_3g_main: str = ""               # 3G 주 패스 자세 고정 '<입사°>/<above|below>' (예 '35/below'; '' = 적응형 선택). 캘리브 확인 실기용
                                            # (2026-10-06 g3_13: 토치 위 40° ↔ 아래 15° 측정 오프셋 — 같은 입사로 뒤집어 원인을 가른다)
    wrist2_branch_3g: str = "any"           # 3G 실행의 손목 형태 (손목2 부호 'neg'/'pos'; 'any' = 검사 안 함). 같은 툴 자세의 두 손목 해는 3G 자세에서
                                            # 심 위치가 3–4 mm 달랐고 (2026-10-06 g3_24/25 손목2 + ↔ g3_13/17 −; 형태마다 실행 간 0.1 mm 재현), −
                                            # 에서만 끝 패스·주 패스가 판 법선 0.3 mm 안으로 일치 → 시작 자세가 다른 형태면 아무것도 보내지 않고 거부
    wrist2_branch_1g: str = "any"           # 1G 실행의 손목 형태 (손목2 부호 'neg'/'pos'; 'any' = 제한 없음) — 캘리브를 만든 팔 형태 집합 (P7: 2026-10-08
                                            # 1G 진행 +1 실행은 손목2 < 0, 1G 캘리브의 형태; ±1 형태의 심 방향 오프셋 6.4–6.6 mm). 정해지면 그 형태 안에서만
                                            # 진행 부호·손목 형태를 고른다 (P2 2026-10-09: 고르는 규칙은 시작 자세에서 관절 이동 최소)
    torch_flip_3g: bool = True              # 그 후보에 토치 아래 (광축 둘레 180°, φ 부호 반대) 를 넣는다 — 패스 사이 회전은 광축 둘레 90° 제자리 회전 +
                                            # 나머지로 나눠 보낸다 (180° 한 번은 보간 방향이 정해지지 않는다). 새 이동: 첫 실기는 지켜보며
                                            # (2026-10-06: 35°+피치 5° 는 g3_08/09 양끝·길이 정상, 32°+5° 는 g3_11 베벨 검출 49 % → 끝 판정 실패)

    @classmethod
    def from_params(cls, d: Optional[dict]) -> "AutoTeachConfig":
        d = dict(d or {})
        retired = sorted(k for k in d if k in AUTO_TEACH_RETIRED_KEYS)
        if retired:
            raise ValueError("auto_teach: " + "; ".join(f"{k} is no longer a setting ({AUTO_TEACH_RETIRED_KEYS[k]})" for k in retired)
                             + " - remove it from the platform config")
        cfg = cls(dwell_frames=_i(d, "dwell_frames", cls.dwell_frames),
                  max_passes=_i(d, "max_passes", cls.max_passes),
                  max_corrections=_i(d, "max_corrections", cls.max_corrections),
                  first_frame_timeout_s=_f(d, "first_frame_timeout_s", cls.first_frame_timeout_s),
                  accept_inferred_end=_b(d, "accept_inferred_end", cls.accept_inferred_end),
                  feature_workers=max(0, _i(d, "feature_workers", cls.feature_workers)),
                  post_scan_worker=_b(d, "post_scan_worker", cls.post_scan_worker),
                  quality_dataset_path=str(unnull(d.get("quality_dataset_path")) or ""),
                  posture_3g_adaptive=_b(d, "posture_3g_adaptive", cls.posture_3g_adaptive),
                  torch_flip_3g=_b(d, "torch_flip_3g", cls.torch_flip_3g),
                  wrist2_branch_3g=_wrist_branch(d.get("wrist2_branch_3g", cls.wrist2_branch_3g)),
                  wrist2_branch_1g=_wrist_branch(d.get("wrist2_branch_1g", cls.wrist2_branch_1g), "wrist2_branch_1g"),
                  posture_3g_main=str(unnull(d.get("posture_3g_main")) or "").strip())
        cfg.validate()
        return cfg

    def validate(self) -> "AutoTeachConfig":
        if self.dwell_frames < 1 or self.max_passes < 1 or self.max_corrections < 0 or not self.first_frame_timeout_s > 0.0:
            raise ValueError("auto_teach: dwell_frames >= 1, max_passes >= 1, max_corrections >= 0, first_frame_timeout_s > 0")
        parse_posture_3g(self.posture_3g_main)                    # 형식 검사 (ValueError)
        return self

    @property
    def feature_workers_effective(self):
        """seam_features.extract_seam_features_along_scan(workers=...) 값: 0 → 'auto'."""
        return "auto" if int(self.feature_workers) <= 0 else int(self.feature_workers)

    def to_dict(self) -> dict:
        return {"dwell_frames": self.dwell_frames,
                "max_passes": self.max_passes, "max_corrections": self.max_corrections,
                "first_frame_timeout_s": self.first_frame_timeout_s,
                "accept_inferred_end": self.accept_inferred_end,
                "feature_workers": self.feature_workers, "post_scan_worker": self.post_scan_worker,
                "quality_dataset_path": self.quality_dataset_path, "posture_3g_adaptive": self.posture_3g_adaptive,
                "torch_flip_3g": self.torch_flip_3g, "posture_3g_main": self.posture_3g_main}


# --------------------------------------------------------------------------------------
# 캘리브
# --------------------------------------------------------------------------------------
def load_calib(path=None) -> Calib:
    """eoat_description 캘리브 YAML(laser_link→profiler_optical_frame) 또는 Calib.to_dict() YAML → Calib."""
    p = Path(path) if path else DEFAULT_CALIB
    with open(p, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if "T_flange_opt" in d or "flange_opt_xyz_m" in d:
        return Calib.from_dict(d)
    if "xyz" not in d or "rpy" not in d:
        raise ValueError(f"unrecognised calib file (need xyz/rpy or T_flange_opt): {p}")
    parent = str(d.get("parent_frame", "laser_link"))
    if parent != "laser_link":
        raise ValueError(f"calib parent_frame must be laser_link (got {parent!r}); see design §6.2 #3")
    # 센서 x 부호·시간 오프셋은 캘리브의 일부: 코드 기본값으로 채우지 않는다 (없거나 0 부호면 시작에서 거부)
    missing = [k for k in ("x_sign", "dt_s") if d.get(k) is None]
    if missing:
        raise ValueError(f"calib file {p} must state {', '.join(missing)} explicitly (x_sign +1/-1 from the probe or flip scan, "
                         "dt_s in seconds - 0.0 if not yet measured)")
    x_sign = int(d["x_sign"])
    if x_sign not in (-1, 1):
        raise ValueError(f"calib file {p}: x_sign must be +1 or -1, got {d['x_sign']!r}")
    c = Calib.from_laser_link(opt_xyz=tuple(float(v) for v in d["xyz"]), opt_rpy=tuple(float(v) for v in d["rpy"]),
                              eoat_yaw_rad=float(d.get("eoat_yaw", 0.0)), dt_s=float(d["dt_s"]),
                              x_sign=x_sign, source=f"{p.name}",
                              date=str(d.get("date", "")), sensor_param_hash=str(d.get("sensor_param_hash", "") or ""),
                              **g1_block_fields(d.get("g1")))                  # 1G core 전용 블록 (선택)
    return c


def resolve_resource_path(ref: str, *, base_dir=None) -> Path:
    """'package://<pkg>/<rel>' (ament 공유 디렉터리, 없으면 이 저장소 scan_stack/<pkg>/<rel>) 또는 경로 (상대는 base_dir 기준)."""
    ref = str(ref)
    if ref.startswith("package://"):
        pkg, _, rel = ref[len("package://"):].partition("/")
        try:
            from ament_index_python.packages import get_package_share_directory
            p = Path(get_package_share_directory(pkg)) / rel
            if p.exists():
                return p
        except Exception:  # noqa: BLE001 - ament 없음 (오프라인·시험)
            pass
        return SCAN_STACK_DIR / pkg / rel
    p = Path(ref).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    return p


def attach_tool_body(calib: Calib, settings: "Settings") -> Calib:
    """설정 safety.tool_body_mesh (EOAT 메시, eoat_base_link 프레임) → calib.body_points_eoat (볼록 껍질 점). 키가 없거나 이미
    붙어 있으면 그대로. 메시를 못 읽으면 ValueError (조용히 몸체 검사를 끄지 않는다)."""
    ref = (settings.safety or {}).get("tool_body_mesh") if settings is not None else None
    if not ref or getattr(calib, "body_points_eoat", None) is not None:
        return calib
    from scan_core.tool_body import body_points_from_mesh
    p = resolve_resource_path(str(ref))
    try:
        B = body_points_from_mesh(p, scale=float((settings.safety or {}).get("tool_body_mesh_scale", 1.0)))
    except (OSError, ValueError) as e:
        raise ValueError(f"safety.tool_body_mesh {ref!r} ({p}) unreadable: {e}") from e
    boxes = (settings.safety or {}).get("tool_body_boxes") or []
    extra = tool_body_box_points(settings)
    if extra is not None:
        B = np.vstack([B, extra])
    calib.body_points_eoat = B
    calib.body_source = f"{p.name} hull" + (f" + {len(boxes)} box(es)" if boxes else "") + f" {B.shape[0]} pts"
    return calib


def tool_body_box_points(settings: "Settings") -> Optional[np.ndarray]:
    """설정 safety.tool_body_boxes (메시·로봇 모형에 없는 부품 — 예: 센서 케이블 커넥터; eoat 프레임 축 정렬 상자 [xmin, ymin, zmin,
    xmax, ymax, zmax] m) → 상자 꼭짓점 (8·n, 3), 없으면 None. 잘못된 상자는 ValueError."""
    boxes = (settings.safety or {}).get("tool_body_boxes") or [] if settings is not None else []
    extra = []
    for b in boxes:
        b = [float(v) for v in b]
        if len(b) != 6 or not all(np.isfinite(b)) or b[0] > b[3] or b[1] > b[4] or b[2] > b[5]:
            raise ValueError(f"safety.tool_body_boxes entry must be [xmin, ymin, zmin, xmax, ymax, zmax] (m), got {b!r}")
        extra += [[x, y, z] for x in (b[0], b[3]) for y in (b[1], b[4]) for z in (b[2], b[5])]
    return np.asarray(extra, dtype=float) if extra else None


# --------------------------------------------------------------------------------------
# ScanJob ↔ 설정 / 파일
# --------------------------------------------------------------------------------------
def job_from_settings(settings: Settings, start, end, surface_normal=(0.0, 0.0, 1.0), **overrides) -> ScanJob:
    """scan_defaults + safety(max_laser_tilt_deg, torch_protrusion_m) + progression_sign → ScanJob (안전 한계는 scan_l0.safety)."""
    kw: dict = {}
    for k, v in settings.scan_defaults.items():
        if k in _JOB_FIELDS:
            kw[k] = unnull(v)
    for k in _SAFETY_JOB_KEYS:
        if k in settings.safety:
            kw[k] = settings.safety[k]
    kw["progression_sign"] = settings.progression_sign
    kw["base_frame"] = settings.base_frame
    for k, v in overrides.items():
        if k not in _JOB_FIELDS:
            raise KeyError(f"unknown ScanJob field {k!r}")
        kw[k] = v
    return ScanJob(start=np.asarray(start, dtype=float), end=np.asarray(end, dtype=float),
                   surface_normal=np.asarray(surface_normal, dtype=float), **kw)


def job_to_dict(job: ScanJob) -> dict:
    d = job.to_dict()
    d["torch_lateral_m"] = [float(v) for v in job.torch_lateral_m]
    return d


def resolve_position(d: dict) -> Optional[str]:
    """job 블록 dict → orientation 값 (position 이 단일 출처). position '1G'|'3G' → flat_1G|vertical_3G; 없으면 orientation 키
    (이전 파일: flat_1G / vertical_3G / auto) 그대로; 둘 다 명시인데 다르면 JobError. 둘 다 없으면 None (= ScanJob 기본 'auto')."""
    pos = unnull(d.get("position"))
    ori = unnull(d.get("orientation"))
    if pos is None:
        return ori
    o_pos = orientation_from_position(pos)
    if ori is not None and ori != "auto" and ori != o_pos:
        raise JobError(f"job position {pos!r} ({o_pos}) conflicts with the legacy orientation {ori!r} in the same job block: "
                       "remove 'orientation' (position is the single source of truth)")
    return o_pos


def job_from_dict(d: dict) -> ScanJob:
    """알 수 없는 키 무시, null 처리 후 ScanJob. `position` (1G|3G) 은 orientation 으로 옮긴다 (resolve_position)."""
    kw = {k: unnull(v) for k, v in d.items() if k in _JOB_FIELDS}
    if "torch_lateral_m" in kw and kw["torch_lateral_m"] is not None:
        kw["torch_lateral_m"] = tuple(float(v) for v in kw["torch_lateral_m"])
    ori = resolve_position(d)
    if ori is not None:
        kw["orientation"] = ori
    else:
        kw.pop("orientation", None)
    return ScanJob(**kw)


def with_position(job: ScanJob, position: Optional[str]) -> ScanJob:
    """CLI --position 덮어쓰기: position '1G'|'3G' → 같은 job 의 orientation 을 바꾼 새 ScanJob (None 이면 그대로).
    plate_top_measured_m / plate_plane_measured 도 유지."""
    if position is None:
        return job
    d = job_to_dict(job)
    d["orientation"] = orientation_from_position(position)
    j = ScanJob.from_dict(d)
    j.plate_top_measured_m = job.plate_top_measured_m
    return j


def job_block_for_file(job: ScanJob) -> dict:
    """save_job 이 쓰는 job 블록: job_to_dict 에서 orientation 대신 position (명시 자세일 때; 'auto' 면 orientation: auto 유지)."""
    d = job_to_dict(job)
    pos = position_from_orientation(job.orientation)
    if pos is not None:
        d.pop("orientation", None)
        out = {"position": pos}
        out.update(d)
        return out
    return d


def parse_set_overrides(items) -> dict:
    """CLI `--set key=value` → {key: 파싱값} (yaml 로 값 해석: 숫자/불리언/리스트)."""
    out: dict = {}
    for it in items or ():
        if "=" not in it:
            raise ValueError(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in _JOB_FIELDS:
            raise ValueError(f"--set: unknown ScanJob field {k!r}")
        out[k] = yaml.safe_load(v)
    return out


def specimen_from_doc(doc: Optional[dict]) -> Optional[GrooveSpec]:
    """job 문서의 specimen 블록 → GrooveSpec (없으면 None)."""
    if not doc:
        return None
    block = doc.get("specimen")
    if not isinstance(block, dict) or not block:
        return None
    return GrooveSpec.from_dict(block)


def _wrist_branch(v, key: str = "wrist2_branch_3g") -> str:
    """auto_teach.wrist2_branch_3g / _1g: 'any' | 'neg' | 'pos' (그 밖은 설정 오류)."""
    v = str(v or "any").strip().lower()
    if v not in ("any", "neg", "pos"):
        raise ValueError(f"auto_teach.{key} must be any|neg|pos, got {v!r}")
    return v


def qc_targets_effective(at: "AutoTeachConfig", doc: Optional[dict], z_valid_mm=None) -> dict:
    """충실도 게이트의 설계 허용치 = 작업 문서 qc_targets 의 설계 항목만 (설계 정보; 플랫폼 기본값 없음 — 사용자 2026-10-07). 측정 기준
    (예전 바닥·루트 모서리 비율, 심 방향 표준편차 상한) 은 무시한다 (scan_core.measure 원칙으로 유도). at·z_valid_mm 은 호환용 (안 씀)."""
    block = (doc or {}).get("qc_targets") or {}
    return {k: dict(v) for k, v in block.items() if isinstance(v, dict) and k in QC_DESIGN_KEYS}


QC_DESIGN_KEYS = ("root_gap_mm", "groove_depth_mm", "top_opening_mm", "bevel_angle_deg", "hi_lo_mm", "weld_length_mm",
                  "root_position_mm")


def save_job(path, job: ScanJob, teach: Optional[dict] = None, meta: Optional[dict] = None, *,
             specimen: Optional[GrooveSpec] = None, qc_targets: Optional[dict] = None, start_pose: Optional[dict] = None,
             precision: Optional[dict] = None, weld_plan: Optional[dict] = None, weld_design: Optional[dict] = None) -> Path:
    """start_pose: 작업 시작 자세 블록 (관절각 — 교시 도구가 착점보다 먼저 손으로 지정; auto-teach 가 첫 이동으로 그 자세에 간다).
    precision: 1G core 종료 판정의 1σ 정밀도 목표 블록 (템플릿의 것 그대로). weld_plan: 용접 절차 (WPS) 블록 (템플릿의 것 그대로 —
    교시한 작업도 그 공정값·출처를 지닌다; 없으면 WeldProcess 기본값). weld_design: 용접 설계 블록 (캡 보강 공칭·공차; 템플릿 그대로)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"job": job_block_for_file(job)}
    if specimen is not None:
        doc["specimen"] = _plain(specimen.to_dict())
    if qc_targets:
        doc["qc_targets"] = _plain(dict(qc_targets))
    if precision:
        doc["precision"] = _plain(dict(precision))
    if weld_plan:
        doc["weld_plan"] = _plain(dict(weld_plan))
    if weld_design:
        doc["weld_design"] = _plain(dict(weld_design))
    if start_pose:
        doc["start_pose"] = _plain(dict(start_pose))
    doc["teach"] = _plain(teach or {})
    doc["meta"] = _plain(dict(meta or {}))
    doc["meta"].setdefault("saved", _dt.datetime.now().isoformat(timespec="seconds"))
    with open(p, "w", encoding="utf-8") as f:
        f.write("# scan_l0 job file: job = scan_core.types.ScanJob fields (base_frame, m/s/deg), teach = teaching record\n")
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return p


def load_job(path) -> tuple:
    """job YAML → (ScanJob, 전체 문서 dict). {job: ...} 또는 평면 dict."""
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    d = doc["job"] if "job" in doc and isinstance(doc["job"], dict) else doc
    return job_from_dict(d), (doc if "job" in doc else {"job": d, "teach": {}, "meta": {}})


def _plain(o):
    """yaml.safe_dump 가능한 순수 파이썬 값으로."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return os.fspath(o)
    return o
