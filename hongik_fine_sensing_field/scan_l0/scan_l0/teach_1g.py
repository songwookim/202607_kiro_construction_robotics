"""1G core 오케스트레이터 (G1_CORE_SPEC §2/§5 — 사용자 2026-10-09 14:10: '전체 프로세스의 흐름은 지금의 틀을 유지') — 지금의 틀 그대로,
각 단계 안의 논리만 새 모형 (WP1 프로파일 모형, WP2 융합 추정, k·σ 판정) 으로.

틀: 시작 자세 → 교시한 두 점 → 시작 자세에서 MoveIt 으로 먼저 갈 점 (관절 이동 최소 — 같으면 가까운 끝, 손목2 음수) →
  [S 드웰 + 정밀 보정] grab (정지 프레임) → WP1 템플릿 맞춤 (드웰 LegObs) → WP2 site_state (§4.8: 루트·σ, 롤, 판 거리, 목표 자세 T*) →
     D11 (자리마다 첫 grab 만: 2·λ > k², 아니면 DATA_FAIL) → D15: k·max(σ_lat, σ_nor)/tol_root > 1 이고 T* 에서 더 작으면 T* 로 보정
     이동 후 다시 grab (고정 횟수·띠 없음)
  → [S→E 트랜싯] 지금처럼 자세 그대로 평행이동, 이제는 스트리밍해 추정에 융합 (심 방향을 잰다) → 추정
  → [E 드웰 + 정밀 보정] 융합 방향으로 레이저 선 ⟂ 심 (yaw) 까지 같은 보정 규칙
  → L1 = 보정된 두 점 + 융합 방향 (추정) → 패스 1 (E→S): 보정된 두 점 사이 + 남은 설계 길이의 절반씩 양쪽 + k·σ 오버스캔 → 추정 →
     패스 2 부터: 추정 양끝 너머까지 시편 전체 (k·σ_plan 오버스캔, 방향 교대) → 모든 산출량이 k·σ ≤ 공차이고 모형 적합·가설 검사
     (seam_estimate §4.9: 산출량이 기대는 구간이 모두 맞고 베벨 쪽·'이 심' 가설이 2 ln 오즈 > k²) 를 통과하면 끝 (D12), 아니면 예산
     (D13) 까지 — 예산 뒤에도 산출량이 맞지 않는 구간 (misfit) 에 기대면 MODEL_MISFIT → DATA_FAIL (category 'sensing', 출력 없음),
     그 밖의 미해결 (가설 포함) 은 PASS_UNRESOLVED
  → 출력 2개 (sections + 4패스 계획; D14).
S/E 이름 (사용자 2026-10-10): S = 먼저 가는 점 (방문 순서 규칙이 이동 전에 정한 끝), E = 다른 점 — 교시 순서와 무관. 순서가 정해지는 순간
(_choose_visit_order, 이동 전) 에 이름을 붙이고 그 뒤로는 틀 하나: 거리 기준 (s 는 S 에서), 진행, 이름, 로그, run_log, sections, weld_plan,
precision_sensing (어느 교시 점이 S 인지는 res.s_taught_point 와 visit_order 에). 용접 방향은 계획의 설정 (job weld_plan.direction, 기본 S->E).
세계 모형은 하나 (cell_world.CellWorld): 두 착점 + 설계 → 추정마다 잰 시편으로; 포락선 기준과 MoveIt 물체 ('hongik/' 이름공간) 를 둘 다 거기서.
거리: 드웰·트랜싯 = 거친 거리 (센서 창·시야에서, ≈ 443 mm), 패스 = r* − 깊이/2 (피치 job). 판정 지점: MoveIt 가능성 (D4), 포락선
(D3), 이동 중 정지 (D6), 취소 (D7), 작업 프로세스 죽음 (D8), 센서 (D9), 자리마다 첫 grab 의 그루브 LR 시험 (D11), k·σ ≤ 공차 (D12), 예산 (D13), 계획 계약·도달성 (D14).

AutoTeach 에서 물려받아 그대로 쓰는 것은 기반 원시뿐: 게이트·안전 정책·센서 시야 (_universal_inputs, _motion_config, _apply_safety_policy,
_sv, _check_position), 센서 (ensure_sensor), 이동 (_local_move, _joint_move), 획득 (_universal_ready, _go_start_pose,
_acquire_site → _choose_visit_order → _plan_acquisition → _prefer_yaw_band, _acquisition_poses,
_acquisition_motions, _theta_goal_rad, _wrist2_required, _swap_rough, _set_progression, _legs_pass_envelope), 패스 작업 (_pass_job 1G
경로, _clear_plate), 구간 태그 (_leg_phase), 기록·사건·중단 (_decide, _event, _record_move, _confirm, _check_estop, _abort_path,
_clear_cell, _latch_*), 도착 허용 (_hit_tol_m, _origin_tol_m, _coarse_plate_m, _yaw_band_rad), 시각화 간격 (_vis_step_m). **쓰지 않는 것**:
AutoTeach 의 판정 층 (DECISION_LAYERS_NOT_USED: 국소화·장면 판정·프로브·보정 반복·S 추적/정렬·E 배치·탐색·충실도/병합 게이트·P8·길이 띠)
과 legacy MoveIt 장면 (_set_cell: 가정 크기 상자 + 바닥 자르기 사다리 — 3G legacy 는 그대로 쓴다).

계산은 실행마다 띄우는 작업 프로세스 (g1_worker.G1Worker) 의 일감 'leg' (드웰 grab·트랜싯·패스) / 'estimate' / 'outputs'. out_dir 에는
sections.json/.npz 와 weld_plan.json/.npz, 그리고 납품하는 계획의 KIRO 형식 kiro/ (pass_<k>.yaml + 공유용 info + README — 플랫폼 설정
kiro_output 이 있을 때; scan_core.kiro_plan) 만, 나머지는 out_dir/verification/.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
import traceback
from concurrent.futures import Future, InvalidStateError
from pathlib import Path
from typing import Optional

import numpy as np

from scan_core.frames import unit
from scan_core.g1_types import Precision, SeamEstimate, SeamPrior, Tolerances, jsonable
from scan_core.path_gen import EnvelopeError, pass_overscan, sensor_frames
from scan_core.seam_estimate import _section_basis, plan_ends
from scan_core.types import JobError, ScanJob, _jsonable, merge_plate_plane
from scan_core.weld_plan import WeldDesign, WeldProcess, weld_design_from_doc
from scan_core import kiro_plan as _kp
from scan_core import weld_line as _wl
from scan_core import weld_orient as _wo
from scan_core import weld_torch as _wt

from . import g1_worker as _gw
from . import job_overrides as _jo
from .acquisition import visit_order
from .cell_world import CellWorld
from .autoteach import (STATUS_DATA_FAIL, STATUS_DRY_RUN, STATUS_PASS, STATUS_PASS_UNRESOLVED, AutoTeach, AutoTeachResult, PassPlan,
                        SeamLine, _Abort, _rot_angle_deg, weld_process_doc)
from .config import attach_tool_body, job_to_dict
from .executor import (STATUS_ABORTED, STATUS_ERROR, STATUS_OK, STATUS_REFUSED, ScanRunResult, check_leg,
                       frames_from_records, gate_and_arm, raw_payload, start_pose_mismatch)
from .planning import make_local_move_plan
from .ports import PlatformNotReady, ScanRefused

__all__ = ["Teach1G", "select_teach_class", "G1_PIPELINES", "PHASES", "DECISION_LAYERS_NOT_USED", "MOVE_MEASURE", "MOVE_POSITION",
           "SPEED_CLASSES"]

G1_PIPELINES = ("core",)              # KIRO 1G package: the 1G core only (no legacy / 3G routing)
PHASES = ("setup", "acquire", "dwell", "correct", "transit", "estimate", "pass", "outputs", "experiment")
# AutoTeach 의 판정 층 (§8.4 'Not used') — Teach1G 는 이것을 부르지 않는다 (test_sim_teach1g 가 확인)
DECISION_LAYERS_NOT_USED = (
    "_localize", "_check_dwell_scene", "_dwell", "_grab", "_raise_plate_top", "_first_S", "_seek_S", "_track_leg", "_track_length_m",
    "_baseline_refusal", "_align_S", "_finish_S", "_transit_to_E", "_place_E", "_line_from_sites", "_scale_check", "_probe", "_probe_x",
    "_probe_move", "_correct_until_converged", "_site_record", "_lateral_search", "_find_groove_at", "_approach_blind",
    "_reacquire_if_oblique", "_refine_pass", "_merged_gate", "_gate_with", "_plan_next", "_streamed_leg", "_move_to_pass_start",
    "_pass_overscan", "_main_posture", "_rescan_job", "_plan_rescan", "_final_job", "_deliver_direction_combined", "_weld_plan",
    "_cross_sections_start", "_support_probe", "_choose_3g_end_postures", "_occlusion_terminal", "_ambiguity_reading", "_converged",
    "_set_cell")                         # _set_cell = legacy MoveIt 장면 (가정 크기 + 바닥 자르기 사다리) — 1G core 는 세계 모형 하나 (cell_world)


def select_teach_class(position, pipeline: str = "core"):
    """§8.6 라우팅 (KIRO 1G 패키지): 1G + g1_pipeline 'core' → Teach1G. 다른 자세·pipeline 은 ValueError (이 패키지는 1G core 만 돈다)."""
    pl = str(pipeline or "").strip()
    if pl not in G1_PIPELINES:
        raise ValueError(f"g1_pipeline must be one of {G1_PIPELINES}, got {pipeline!r}")
    if str(position or "").strip() != "1G":
        raise ValueError(f"welding position {position!r} is not supported by this package: 1G only (the 1G core)")
    return Teach1G


class _CancelAbort(_Abort):
    """D7 itself: the run stops at a cancel check (before a motion, a compute job or a stream). Its reason names the cancel; every
    other abort that ends a cancelled run (the motion the cancel stopped, a job it interrupted) is a consequence of the cancel and is
    reported as D7 with its own text after it (Teach1G.run)."""

    def __init__(self, where: str = "", cause: str = "cancel requested by the operator"):
        self.where = where
        super().__init__(STATUS_ABORTED, f"{cause}{(' ' + where) if where else ''} (D7)")


def _sgn(v: float) -> int:
    return 1 if float(v) >= 0.0 else -1


DWELL_DIRECTION = 0                  # 드웰 LegObs.direction (spec r2 §3.6a: 진행 없음; g1_types 는 드웰에만 0 을 받는다)

# Move speed classes (user 2026-10-12): a move's purpose decides its speed. The end precision target (0.06 mm 1-sigma in a 3-pass
# budget) rests on the measurement moves - the per-crossing end scatter grows with speed (lab: 0.076 / 0.09 / 0.17 / 0.42 mm at
# 15 / 20 / 40 / 60 mm/s) - so they run at the job's measurement speed; every other move runs at the platform's positioning limits.
#   MOVE_MEASURE  the scan passes and the precise corrections at the dwell sites: the job's speed_mps (under the platform TCP limit)
#   MOVE_POSITION everything else - the start pose, the acquisition joint move and the descent onto the S dwell (inherited moves at
#                 the platform limits), the coarse S->E transit (it still records: its readings enter the estimate with their own
#                 measured speed in the noise priors' speed model), pass repositions and approaches, experiment moves: the platform
#                 TCP limit (cfg.v_eff_max_mps = moveit_jtc.limits.max_tcp_speed_mps; the operator may only lower it, local_speed_mps)
#                 with the platform's joint velocity limit in every time parameterisation
MOVE_MEASURE = "measure"
MOVE_POSITION = "position"
SPEED_CLASSES = (MOVE_MEASURE, MOVE_POSITION)


class Teach1G(AutoTeach):
    """1G core 실행 하나 (모듈 docstring). AutoTeach 와 같은 생성자·결과 (AutoTeachResult) — ROS 액션·클라이언트는 그대로."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.vdir = self.out / "verification"
        self._phase = "setup"
        self._cancel_r, self._cancel_w = os.pipe()
        os.set_blocking(self._cancel_w, False)
        self._cancelled = False
        # 취소는 하나 (WP5 S4): 서버가 만든 실행 취소 Future (없으면 이 실행의 것). 끝나면 → 작업 프로세스 기다림 (select) 의 fd + 이동 전 검사
        if getattr(self, "cancel", None) is None:
            self.cancel = Future()
        self.worker: Optional[_gw.G1Worker] = None
        self.tol: Optional[Tolerances] = None
        self.precision: Optional[Precision] = None
        self.prior: Optional[SeamPrior] = None
        self.legs: list = []                             # profiles.npz 경로 (기록 순서)
        self.leg_info: list = []
        self.est_history: list = []
        self.joint_state_source: Optional[dict] = None   # the latest estimate's joint-state source record (run_log, warning once)
        self.sites: dict = {}                            # 'S' (먼저 간 끝) / 'E' → 보정된 드웰 판독 (루트 점, 판 법선, …)
        self.n_pass = 0
        self._n_hat: Optional[np.ndarray] = None
        self.world: Optional[CellWorld] = None           # 세계 모형 하나 (포락선 기준 + MoveIt 물체)
        self.world_history: list = []                    # 세계 판마다 to_dict (run_log)
        self.k = float(self.settings.judgement_k)
        self._settle_devs: list = []                     # 움직인 구간마다 엔코더 진행 방향 편차 (scan_core.settling) — 패스 오버스캔의 안정 시간
        # 이동 속도 부류 (사용자 2026-10-12, MOVE_MEASURE / MOVE_POSITION): 지금 보내는 직선의 부류 — 이동 기록 (speed_class) 에 남긴다;
        # 물려받은 이동 (시작 자세·획득 관절 이동·하강) 은 위치 잡기
        self._move_class: Optional[str] = None
        self.experiments = None                          # 작업 파일 experiments: 블록 (실행 뒤 실험, experiments.py; _gates 가 읽는다)
        self.experiments_result: Optional[dict] = None
        self.kiro_cfg: Optional[_kp.KiroOutputConfig] = None   # 플랫폼 설정 kiro_output (KIRO 형식 출력; 없으면 쓰지 않는다)
        self.weld_torch = None                           # 플랫폼 설정 weld_torch (용접 계획의 토치 모형 = 우리 1차 브라켓 토치; scan_core.weld_torch)
        self.cancel.add_done_callback(self._on_cancel_done)

    # ---- 기록: 단계 (피드백 phase) ------------------------------------------------------------------------------
    def _set_phase(self, ph: str) -> None:
        assert ph in PHASES, ph
        self._phase = ph
        self._event("phase", phase=ph)

    def _decide(self, step: str, msg: str, **data) -> None:
        """결정 기록 + 로그 줄 '[단계] 결정: …' — 서버 피드백 phase 는 줄 머리에서 온다 (PHASES)."""
        self.res.decisions.append({"t": round(self._clock.monotonic() - self.t0, 3), "phase": self._phase, "step": step, "msg": msg,
                                   **_jsonable(data)})
        self._p(f"[{self._phase}] {msg}" if step == self._phase else f"[{self._phase}] {step}: {msg}")

    # ---- 취소 (D7, S4: 하나의 취소 Future) ------------------------------------------------------------------------
    def request_cancel(self, why: str = "cancel requested by the operator") -> None:
        """취소 Future 를 끝낸다 (서버 _complete_cancel 과 같은 것)."""
        try:
            self.cancel.set_result(why)
        except InvalidStateError:                             # 이미 취소됐다 (같은 취소 — 할 일 없음)
            return

    def _on_cancel_done(self, _fut) -> None:
        self._cancelled = True
        if self._cancel_w < 0:                                # 실행이 끝나 파이프를 닫았다
            return
        try:
            os.write(self._cancel_w, b"c")
        except BlockingIOError:                              # 파이프에 취소 바이트가 이미 있다 (기다림은 그것으로 깬다)
            return
        except OSError as e:
            self.res.notes.append(f"cancel pipe write failed: {e} (the cancel is still seen before the next motion)")

    def _check_cancel(self) -> None:
        if self._cancelled:
            raise _CancelAbort()

    # ---- D1 ------------------------------------------------------------------------------------------------------
    def _gates(self) -> None:
        """D1: 이동 전 입력 계약 — 자세 1G (§8.6), 설계 (specimen + qc_targets root_position), 캘리브 1G 블록
        (he_sigma_m — 시간 오프셋 σ 는 캘리브가 아니라 실행의 시간 구조 (프레임·자세 표본 주기) 에서, seam_estimate.timing_prior) 과 x_sign, S0 ≠ E0, 드웰 grab 프레임 수 ≥ 2·유령점 버스트 + 1 (spec §3.6a, §7 #40). 그 밖은 물려받은 기반
        (범용 입력, 모션 설정, 안전 정책, 방문 순서 초기값, 플랫폼 게이트)."""
        self.out.mkdir(parents=True, exist_ok=True)
        self.vdir.mkdir(parents=True, exist_ok=True)
        # the effective job (the job file + the run-time overrides + the goal's points), before anything else - a record
        self.job_effective_path = self._save_effective_job(self.vdir / _jo.EFFECTIVE_JOB_NAME)
        self._check_position()
        if self.res.position != "1G":
            raise _Abort(STATUS_REFUSED, f"position {self.res.position}: the 1G core runs 1G jobs only (nothing sent)", category="input")
        try:
            self.calib = attach_tool_body(self.calib, self.settings)
            self.calib0 = attach_tool_body(self.calib0, self.settings)
        except ValueError as e:
            raise _Abort(STATUS_REFUSED, f"tool body: {e} (nothing sent)", category="config")
        if self.spec is None or not (self.spec.weld_length_mm > 0.0):
            raise _Abort(STATUS_REFUSED, "job has no specimen (groove design) block with weld_length_mm > 0 - the 1G model is undefined "
                                         "without the design (nothing sent)", category="input")
        for note in self.spec.deprecation_notes():
            self._decide("setup", f"input: {note}")
        try:
            self.tol = Tolerances.from_qc((self.doc.get("qc_targets") or {}))
        except ValueError as e:
            raise _Abort(STATUS_REFUSED, f"job qc_targets: {e} (nothing sent)", category="input")
        try:                                             # 1-σ precision targets of the done test (user 2026-10-11; design tol = warnings)
            self.precision = Precision.from_doc(self.doc.get("precision"))
        except ValueError as e:
            raise _Abort(STATUS_REFUSED, f"job precision: {e} - the done test needs the 1-sigma precision targets (nothing sent)",
                         category="input")
        self._check_measure_speed()
        miss = [k_ for k_ in ("he_sigma_m",) if getattr(self.calib, k_, None) is None]
        if int(getattr(self.calib, "x_sign", 0)) not in (-1, 1):
            miss.append("x_sign")
        if miss:
            raise _Abort(STATUS_REFUSED, f"calibration {self.calib.source or '-'} lacks {', '.join(miss)} (the 1G core needs its 'g1:' block: "
                                         "hand-eye class-offset sigma; nothing sent)", category="config")
        g1_dt = getattr(self.calib, "g1_dt_s", None)
        if g1_dt is not None:                            # 1G 전용 시간 오프셋 (캘리브 'g1:' 블록; legacy dt_s 는 3G 경로가 그대로 읽는다)
            self.calib = dataclasses.replace(self.calib, dt_s=float(g1_dt))
        S0, E0 = np.asarray(self.job0.start, dtype=float), np.asarray(self.job0.end, dtype=float)
        up = -unit(np.asarray(self.job0.gravity_axis_base, dtype=float))
        d = E0 - S0
        if not float(np.linalg.norm(d - up * float(d @ up))) > 0.0:
            raise _Abort(STATUS_REFUSED, "the two rough points coincide in the plate plane (nothing sent)", category="input")
        try:
            weld_dir = self._process_1g().direction                     # 계획 설정 (용접 방향 포함) 은 이동 전에 읽는다
            self._design_1g()                                           # 용접 설계 입력도
        except (TypeError, ValueError) as e:
            raise _Abort(STATUS_REFUSED, f"job weld_plan / weld_design: {e} (nothing sent)", category="input")
        try:                                                            # 용접 토치 모형·KIRO 형식 출력 설정 (프레임 사슬) 도 이동 전에
            self.weld_torch = _wt.weld_torch_config(self.settings.params)
            self.kiro_cfg = _kp.kiro_output_config(self.settings.params)
        except (TypeError, ValueError, OSError) as e:
            raise _Abort(STATUS_REFUSED, f"platform config weld_torch / kiro_output: {e} (nothing sent)", category="config")
        wtm = getattr(self.rs, "weld_torch_model", None)
        if callable(wtm):                                               # the torch on this arm as the running robot model has it
            try:                                                        # (flange <- TCP from TF, not the config record; the lab's
                self.weld_torch = wtm(self.weld_torch)                  # torch turned in its clamp, 2026-10-13)
            except Exception as e:  # noqa: BLE001 - the robot model has no such TCP frame: the plan's torch would be a guess
                raise _Abort(STATUS_REFUSED, f"the weld torch {getattr(self.weld_torch, 'tcp_frame', '?')} in the robot model: "
                                             f"{type(e).__name__}: {e} (nothing sent)", category="config")
        try:                                                            # 토치의 가스 노즐 끝 (와이어 돌출의 기준, 2026-10-13): 이 팔의
            self.weld_torch = _wt.torch_with_nozzle_end(                # 토치면 이 로봇 URDF 충돌 형상 (D14 의 도구) 에서 유도
                self.weld_torch, getattr(self.rs, "weld_tool_body", None))
        except Exception as e:  # noqa: BLE001 - no nozzle end: the stick-out has no reference (never measured from the contact tip)
            raise _Abort(STATUS_REFUSED, f"weld torch nozzle end (the reference of the wire stick-out): {type(e).__name__}: {e} "
                                         "(nothing sent)", category="config")
        try:
            g_b = int(self.ghost_burst_frames)
        except ValueError as e:
            raise _Abort(STATUS_REFUSED, f"{e} (nothing sent)", category="config")
        if not int(self.dwell_frames) >= 2 * g_b + 1:
            raise _Abort(STATUS_REFUSED, f"auto_teach.dwell_frames {int(self.dwell_frames)} < 2 x sensor.ghost_burst_frames {g_b} + 1: the "
                                         "column-wise median of a dwell grab could not outvote a ghost burst (spec 3.6a; nothing sent)",
                         category="config")
        self._universal_inputs()
        self._motion_config()
        job0 = self._apply_safety_policy(self.job0)
        # 입력 정리 (판정 아님): 1G 거친 법선 up 을 거친 진행 방향에 직교화해 작업 검증을 통과시킨다 (AutoTeach._gates 와 같은 연산)
        try:
            n_r, u_r = job0.normal_unit, job0.unit_dir
            if abs(float(n_r @ u_r)) > 0.0:
                n_new = unit(n_r - float(n_r @ u_r) * u_r)
                job0 = ScanJob.from_dict({**job_to_dict(job0), "torch_lateral_m": tuple(job0.torch_lateral_m),
                                          "surface_normal": [float(v) for v in n_new]})
                self.job0 = job0
            job0.validate()
        except (JobError, ValueError) as e:
            raise _Abort(STATUS_REFUSED, f"rough job invalid: {e} (nothing sent)", category="input")
        self.orientation = job0.resolved_orientation()
        self.up = job0.up_unit
        # 방문 순서의 시작 가정 (같은 비용이면 이 순서 — 로봇에 가까운 착점부터); 실제 순서와 S/E 이름은 획득 계획에서 이동 전에 정한다
        vo = visit_order(job0.start, job0.end, self.res.position, job0.gravity_axis_base)
        self.res.visit_order = _jsonable({**vo, "taught_start": S0.tolist(), "taught_end": E0.tolist(),
                                          "first_visited": "end" if vo["swap"] else "start"})
        if vo["swap"]:
            job0 = self._swap_rough(job0)
        self._derive_jobs(job0)
        self._read_experiments()
        self._decide_move_speeds()
        self._decide("setup", f"1G core: taught rough points start {np.round(S0, 4).tolist()} m, end {np.round(E0, 4).tolist()} m; S = the "
                              "point visited first (decided before any motion: least TCP (torch tip) motion from the start pose: rotation, then distance), E = the other; "
                              f"weld direction {weld_dir} (job weld_plan.direction; S->E = from the first-visited end); dwell / "
                              f"transit range {self.job_c.standoff_m * 1e3:.0f} mm, pass plate distance {self.pass_plate_m * 1e3:.0f} mm; "
                              f"max passes {self.max_passes}, k {self.k:g}, time offset {self.calib.dt_s * 1e3:.2f} ms (calibration; "
                              "the residual is estimated in every run, prior width from the run's frame and pose sample periods), "
                              f"confirm={self.confirm_mode}; done test = sigma <= precision "
                              f"(1-sigma: ends {self.precision.end_position_mm:g} mm, root {self.precision.root_position_mm:g} mm, "
                              f"length dims {self.precision.length_mm:g} mm, bevel "
                              f"{self.precision.for_dim('bevel_angle_deg', self.spec) or float('nan'):.3f} deg); design tolerances "
                              f"(qc_targets) feed the measured-vs-design warnings only", tolerances=self.tol.to_dict(),
                     precision=self.precision.to_dict(),
                     weld_direction=weld_dir)
        try:
            init = gate_and_arm(self.sess, self.cfg, log=self.log)
        except PlatformNotReady as e:
            raise _Abort(STATUS_REFUSED, f"{e} (nothing sent)", category="motion")
        except ScanRefused as e:
            raise _Abort(STATUS_REFUSED, str(e), category="config")
        self.default_speed = float(init.default_speed)
        self._latch_at_start = self._latch_last()
        self.res.decisions.append({"t": round(self._clock.monotonic() - self.t0, 3), "phase": self._phase, "step": "gates",
                                   "msg": "session_init OK", "safety_init": init.as_dict(),
                                   "platform": {"adapter": self.rs.adapter, "pose_source": self.rs.pose_source,
                                                "motion_backend": self.rs.backend.kind if self.rs.backend is not None else None}})

    def _check_measure_speed(self) -> None:
        """A measurement speed chosen at run time (the goal's measure_speed_mps / client --measure-speed, or a job.speed_mps override
        - one check for both) must lie inside the speed range the noise priors were calibrated over (calib g1 noise_priors
        speed_range_mps: the calibration runs' commanded speeds; scan_core.noise_priors.check_measure_speed) - REFUSED otherwise,
        before any motion. The job file's own speed_mps is the job's design input and is not a run-time choice."""
        how = self.measure_speed_choice()
        if how is None:
            return
        from scan_core.noise_priors import check_measure_speed
        try:
            self.measure_speed_check = check_measure_speed(float(self.job0.speed_mps), getattr(self.calib, "noise_priors", None),
                                                           what=f"measurement speed ({how})",
                                                           calib_source=str(getattr(self.calib, "source", "") or ""))
        except ValueError as e:
            raise _Abort(STATUS_REFUSED, f"{e} (nothing sent)", category="input")

    def _read_experiments(self) -> None:
        """작업 파일 experiments: 블록 (실행 뒤 실험, experiments.py) 을 이동 전에 읽는다 — 잘못된 블록은 REFUSED (보낸 것 없음). 판 거리의
        'pass' / 'transit' 은 실행의 패스 판 거리 (센서 모형) 와 거친 (트랜싯) 판 거리."""
        from .experiments import parse_experiments
        block = self.doc.get("experiments") if isinstance(self.doc, dict) else None
        try:
            plan = parse_experiments(block, pass_plate_m=float(self.pass_plate_m), transit_plate_m=float(self._coarse_plate_m()),
                                     dwell_frames=int(self.dwell_frames), retreat_default_m=float(self.job0.safe_clearance_m))
        except (TypeError, ValueError) as e:
            raise _Abort(STATUS_REFUSED, f"job experiments: {e} (nothing sent)", category="input")
        self.experiments = plan
        if plan is not None:
            self._decide("setup", f"experiments after the run (job experiments: block; the run's status, estimate and outputs are not "
                                  f"changed, results in <out_dir>/experiments/): {plan.summary()}", experiments=plan.to_dict())

    def _experiments(self, est: Optional[SeamEstimate]) -> None:
        """실행 뒤 실험 (experiments.py): 블록이 있고 실행이 두 끝을 아는 추정을 냈을 때만. 실행의 상태·추정·출력은 바꾸지 않는다."""
        if self.experiments is None:
            return
        ends = (est.ends or {}) if est is not None else {}
        missing = [X for X in ("S", "E") if not bool(getattr(ends.get(X), "two_sided", False))]
        if est is None or missing or self._n_hat is None:
            self._set_phase("experiment")
            self._decide("experiment", "experiments skipped: the run has no estimate with both ends identified "
                                       f"({'no estimate' if est is None else 'end(s) ' + ', '.join(missing) + ' not identified'})")
            self.experiments_result = {"skipped": True, "why": "ends not identified"}
            return
        from .experiments import PostRunExperiments
        self.experiments_result = PostRunExperiments(self, self.experiments, est).run()

    def _make_prior(self) -> SeamPrior:
        """SeamPrior (§8.2): S0 = 먼저 가는 착점, E0 = 다른 착점 (방문 순서가 정해진 뒤 — _name_ends), 설계·허용치·k·캘리브 σ, 센서 x 선형성
        (데이터시트에 없다 → NaN), WPS 단면. 로봇 반복정밀도는 1G 경로에서 NaN (조정 결정 2026-10-09: 구간 오프셋은 패스에서 잰다 — WP2 REML)."""
        proc = self._process_1g()
        sens = dict(self.settings.sensor or {})
        rep = None
        eps = sens.get("x_linearity_sigma")
        return SeamPrior(S0=np.asarray(self.job0.start, dtype=float).copy(), E0=np.asarray(self.job0.end, dtype=float).copy(),
                         up=-unit(np.asarray(self.job0.gravity_axis_base, dtype=float)),
                         # dt_sigma_s: not used by the estimator — the time-offset prior is the run's own timing (timing_prior)
                         spec=self.spec, tol=self.tol, k=self.k, dt_s=float(self.calib.dt_s), dt_sigma_s=float("nan"),
                         rep_sigma_m=float("nan") if rep is None else float(rep), he_sigma_m=float(self.calib.he_sigma_m),
                         sigma_eps=float("nan") if eps is None else float(eps), station_step_m=float(proc.station_step_mm) * 1e-3,
                         station_window_m=float(proc.station_window_mm) * 1e-3, bevel_sign_w=None,
                         laser_pitch_sigma_rad=float("nan") if getattr(self.calib, "laser_pitch_sigma_rad", None) is None
                         else float(self.calib.laser_pitch_sigma_rad),
                         edge_response=getattr(self.calib, "edge_response", None), precision=self.precision,
                         noise_priors=getattr(self.calib, "noise_priors", None))

    def _process_1g(self) -> WeldProcess:
        return WeldProcess.from_dict(weld_process_doc(self.doc if isinstance(self.doc, dict) else None, "1G"))

    def _design_1g(self) -> WeldDesign:
        """용접 설계 입력 (job weld_design: 블록 — 캡 보강 공칭·공차; 없으면 공정의 옛 키, 없으면 KIRO 유도 기본)."""
        return weld_design_from_doc(self.doc if isinstance(self.doc, dict) else None, self._process_1g())

    def _derive_jobs(self, job0: ScanJob) -> None:
        """거친 작업 (물려받은 _derive_jobs: 드웰·트랜싯 = 두 착점, 거친 거리, 피치 0) + 세계 모형: 처음 한 번 두 착점 + 설계의 사전 세계
        (두 점에 대칭 — 방문 순서와 무관), 새 기준 작업마다 그 포락선 기준을 다시 건다. 거친 작업의 이동 (S→E 트랜싯) 은 위치 잡기
        부류라 그 속도 (MOVE_POSITION) 를 담는다 (드웰 grab 은 정지 — 속도 무관)."""
        super()._derive_jobs(job0)
        self.job_c = self._at_speed(self.job_c, MOVE_POSITION)
        if self.world is None:
            dm = self._spec_dims_m()
            self.world = CellWorld.from_rough_points(
                job0.start, job0.end, hit_face_m=self._hit_face_m(), up=job0.up_unit, thickness_m=dm["t"], backing_m=dm["b"],
                plate_widths_m=dm["w_plates"], gap_m=dm["gap"], design_length_m=dm["L"], clearance_m=float(self.policy.clearance_m),
                reach_m=float(self.job_c.standoff_m) + float(job0.safe_clearance_m), base_frame=str(job0.base_frame))
            self.world_history.append(self.world.to_dict())
        self._world_to_envelope()

    def _set_progression(self, sign: int) -> None:
        super()._set_progression(sign)                        # 기준 작업을 다시 만든다 → 세계의 포락선 기준을 다시 건다
        self._world_to_envelope()

    # ---- 세계 모형 하나 (cell_world.CellWorld): 포락선 기준과 MoveIt 물체 ------------------------------------------------
    def _world_to_envelope(self) -> None:
        """포락선 기준 = 세계: 기준 작업의 판 평면 (plate_plane_measured — 기준 작업 자신의 착점 평면과 가까운 쪽이 묶는다), 판 상면 높이
        (plate_top_measured_m, self.plate_top), 지지면 (table_z_m: 툴 키포인트·몸체 ≥ 지지면 + L)."""
        env = self.world.envelope()
        p, n = env["plate_point_m"], env["plate_normal"]
        self.job_ref.plate_plane_measured = merge_plate_plane(None, (p, n))
        self.job_ref.plate_top_measured_m = float(env["plate_top_m"])
        self.job_ref.table_z_m = float(env["floor_m"])
        self.plate_top = float(env["plate_top_m"])

    def _arm_contacts(self) -> tuple:
        """지금 팔 상태의 접촉 (플랫폼 state_contacts: 링크·물체·깊이) → (접촉 목록, 사유 문구 — 없으면 ''). state_contacts 가 없는 플랫폼은
        state_collision 문구에서 (링크, 물체) 쌍을 읽는다."""
        fn = getattr(self.rs, "state_contacts", None)
        if callable(fn):
            r = fn() or {}
            if r.get("valid", True):
                return [], ""
            return list(r.get("contacts") or []), str(r.get("why") or "")
        fn = getattr(self.rs, "state_collision", None)
        hit = fn() if callable(fn) else None
        if hit is None:
            return [], ""
        from .cell_world import contact_pairs
        return [{"link": a, "object": b, "kind": "world"} for a, b in contact_pairs(hit)], str(hit)

    def _apply_world(self, why: str, *, inside_ok: bool = False) -> None:
        """세계 한 판을 모두에 건다: 포락선 기준, MoveIt 물체 ('hongik/' 이름공간 — 우리 것만 넣고, 이번 판에 없는 우리 것만 지운다),
        WORLD 사건·결정 한 줄. 지금 팔 상태가 세계 안이면 (어느 링크가 어느 물체 안) 이동 전에 REFUSED — 세계를 줄이지 않는다."""
        w = self.world
        if not w.fixed_bodies_known:
            fb = getattr(self.rs, "fixed_bodies", None)
            try:
                w.set_fixed_bodies(fb() if callable(fb) else [])
            except Exception as e:  # noqa: BLE001 - the floor then simply has no fixed-body limit; the arm check still refuses
                w.set_fixed_bodies([])
                self.res.notes.append(f"robot fixed body not read ({type(e).__name__}: {e}) - the floor extent does not use it")
        if getattr(w, "reach_below_m", 0.0) is None:
            rf = getattr(self.rs, "arm_reach_m", None)
            try:
                w.set_reach_below(rf() if callable(rf) else None)
            except Exception as e:  # noqa: BLE001 - no reach bound: no under solid (the floor slab still stands), said in the notes
                self.res.notes.append(f"arm reach not read ({type(e).__name__}: {e}) - no 'under the specimen' solid")
        self._world_to_envelope()
        objs = w.moveit_objects()
        if not self.rs.set_scene_boxes(objs):
            raise _Abort(STATUS_REFUSED, f"the cell world (v{w.version}) could not be applied to the MoveIt scene"
                         + ("" if self._motion_sent else " (nothing sent)"), category="motion")
        ids = {o["id"] for o in objs}
        stale = [i for i in self.cell_ids if i not in ids]
        if stale:
            self.rs.remove_scene_boxes(stale)
        self.cell_ids = sorted(ids)
        rec = w.to_dict()
        if not self.world_history or self.world_history[-1].get("version") != rec["version"]:
            self.world_history.append(rec)
        else:
            self.world_history[-1] = rec
        self._event("world", **rec)
        self._decide("world", f"{w.summary()} [{why}]", world=rec)
        contacts, reason = self._arm_contacts()
        if (contacts or reason) and self._motion_sent:
            # the start check refuses only before the first motion (2026-10-13 lab run l6P10_k30_1: after a cancel at the weld
            # pose the run's world restored around the arm 'refused' a run that had moved and stopped): once the arm has moved in this
            # run it may rest inside a world object by design (the torch in the groove) - recorded; MoveIt judges the next plan from it
            self._decide("world", f"the arm at its current pose touches the cell world v{w.version}: "
                                  f"{w.explain(contacts) or reason} - recorded, not refused (the arm has moved in this run; the "
                                  "start-of-run check applies before the first motion only)", world_contacts=_jsonable(contacts))
            return
        if contacts or reason:
            ours = [c for c in contacts if c.get("kind") != "self" and str(c.get("object", "")).startswith("hongik/")]
            where = "nothing sent"
            if inside_ok and ours and len(ours) == len(contacts):
                # a caller that starts where the work leaves the arm (the post-weld scan right after a weld: the torch at the seam)
                # lifts it out first (PostWeldScan._lift_clear) - only our grown objects are touched, nothing else
                deep = max((float(c.get("depth_m") or 0.0) for c in ours), default=0.0)
                self._start_inside = {"deep_m": deep, "contacts": _jsonable(ours)}
                self._decide("world", f"the arm at its current pose is inside the cell world v{w.version}: {w.explain(ours)} (deepest "
                                      f"{deep * 1e3:.1f} mm into the objects grown by C {w.C * 1e3:.0f} mm) - it lifts out along the "
                                      "plate normal first", world_contacts=_jsonable(ours))
                return
            if ours:
                deep = max((float(c.get("depth_m") or 0.0) for c in ours), default=0.0)
                gap = (f"{(w.C - deep) * 1e3:.1f} mm from the solid itself" if deep <= w.C else
                       f"{(deep - w.C) * 1e3:.1f} mm inside the solid itself")
                raise _Abort(STATUS_REFUSED, f"the arm at its current pose is inside the cell world v{w.version}: {w.explain(ours)} - the "
                                             f"deepest contact is {deep * 1e3:.1f} mm into the objects, which are the solids grown by C "
                                             f"{w.C * 1e3:.0f} mm (the link is {gap}): move the arm at least that far out - start from a "
                                             "pose clear of the specimen (e.g. the scan-start pose, not the pose that taught the points) "
                                             f"and start again ({where})", category="motion")
            raise _Abort(STATUS_REFUSED, f"the current arm state is not valid for planning ({w.explain(contacts) or reason}) - "
                                         "self-collision, an object that is not ours, a joint outside its limits or missing joint "
                                         f"states; move the arm to a normal pose (or check /joint_states) and start again ({where})",
                         category="motion")

    def _acquire(self) -> None:
        """범용 시작 (1G core): MoveIt 장면 = 사전 세계 (지금 팔이 그 안이면 이동 전 거부) → 시작 자세 (job start_pose) → 첫 점 획득 (방문
        순서·S/E 이름은 그 계획에서, 이동 전). 3G 지지면 probe·끝 자세는 없다."""
        self._universal_ready()
        self._apply_world("prior: before any motion")
        self._go_start_pose()
        self._acquire_site("S", None)

    def _choose_visit_order(self, name: str, phase: str) -> tuple:
        """물려받은 방문 순서 규칙 (두 순서 × 진행 부호 × IK 해, 손목2 음수; 관절 이동 Σ|Δq| 최소, θ_goal 안 같으면 베이스에 가까운 점, 이동
        전에 거부된 순서는 다른 순서) → 정해지는 순간 S/E 이름 (_name_ends)."""
        best, why = super()._choose_visit_order(name, phase)
        if best is not None:
            self._name_ends()
        return best, why

    def _name_ends(self) -> None:
        """S = 먼저 가는 착점 (지금 job0.start), E = 다른 착점 — 이 뒤로 틀 하나 (사전 SeamPrior 의 S0/E0, s 는 S 에서, 진행 부호, 이름,
        로그, 출력). 어느 교시 점이 S 인지는 res.s_taught_point ('start' | 'end') 와 visit_order 에."""
        first = str(self.res.visit_order.get("first_visited") or ("end" if self.res.rough_swapped else "start"))
        self.res.s_taught_point = first
        self.prior = self._make_prior()
        S, E = np.asarray(self.job0.start, dtype=float), np.asarray(self.job0.end, dtype=float)
        other = "end" if first == "start" else "start"
        self._decide("setup", f"S/E: S = the first-visited point = the taught {first} {np.round(S, 4).tolist()} m, E = the taught {other} "
                              f"{np.round(E, 4).tolist()} m - from here on one frame: s from S, progression, names, logs and outputs are "
                              "S->E", prior=self.prior.to_dict(), s_taught_point=first)

    def _acquire_plan(self, name: str, phase: str) -> tuple:
        """획득 계획 (물려받은 _plan_acquisition: 두 진행 부호 × IK 해, 추종 [S 드웰 → E 드웰 = 트랜싯], 손목2 형태, ψ_b 순위) + 관절 이동
        전에 이어지는 두 직선 (S 드웰로 하강, 트랜싯 'scan') 의 포락선 (D3). S 추적 구간 없음."""
        best = self._plan_acquisition(name)
        if best is None:
            return None, ""
        _job, _T_pre, _T_in, T_out = self._acquisition_poses(int(best["sign"]))
        legs = [(best["T_in"], phase, f"acquire_{name}_in", ""), (T_out, "scan", self._transit_name(), "")]
        env_why = self._legs_pass_envelope(best["T_pre"], legs, why=True)
        if env_why:
            return None, env_why
        best["T_out"] = T_out
        return best, ""

    # ---- 이동: D3 + D4 를 보내기 전에 -----------------------------------------------------------------------------
    def _leg(self, T_to: np.ndarray, phase: str, name: str, *, speed_class: str = MOVE_POSITION, streaming: bool = False,
             extra: str = "") -> tuple:
        """직선 한 구간: 취소 확인 → D3 포락선 (make_local_move_plan) → D4 MoveIt 계획 (check_leg: 보내지 않음) → 거부면 REFUSED
        (래치 없음, 보낸 것 없음) → 물려받은 _local_move (확인 D5, 시작 자세 재확인, 센서 스트리밍 D9, move_l).
        speed_class: 이 이동의 목적이 정하는 속도 부류 (MOVE_MEASURE = 패스·드웰 자리의 정밀 보정, MOVE_POSITION = 그 밖의 모든 이동 —
        기본) — 속도는 _class_speed, 이동 기록에 speed_class 로 남는다."""
        self._check_cancel()
        T_from = self._T_now()
        T_to = np.asarray(T_to, dtype=float)
        v = self._class_speed(speed_class)
        self._move_class = speed_class
        try:
            try:
                plan, _st = make_local_move_plan(T_from, T_to, phase, self.job_ref, self.calib, specimen_top_z_m=self.plate_top,
                                                 speed_mps=v, accel_mps2=self.accel, name=name)
            except (EnvelopeError, JobError, ValueError) as e:
                self._record_move(name, phase, T_from, T_to, {}, v, refused=f"D3 envelope: {e}")
                raise _Abort(STATUS_REFUSED, f"move '{name}' ({phase}) refused before motion by the envelope (D3): {e}",
                             category="motion")
            chk = check_leg(self.sess, plan.waypoints[1], v, timeout_s=self.motion_timeout_s)
            if chk is not None and not chk["ok"]:
                self._record_move(name, phase, T_from, T_to, {}, v, refused=f"D4 MoveIt: {chk['reason']}")
                raise _Abort(STATUS_REFUSED, f"move '{name}' ({phase}) refused before motion by the motion planner (D4): {chk['reason']}",
                             category="motion")
            return self._local_move(T_to, phase, name, speed=v, streaming=streaming, extra=extra, kind="scan" if streaming else None)
        finally:
            self._move_class = None

    def _record_move(self, *a, **kw) -> dict:
        """물려받은 이동 기록 + 속도 부류 (speed_class): _leg 가 보내는 직선은 그 부류, 물려받은 이동 (시작 자세·획득 관절 이동·S 드웰로
        하강 — 플랫폼 한계) 은 위치 잡기."""
        d = super()._record_move(*a, **kw)
        d["speed_class"] = self._move_class or MOVE_POSITION
        return d

    # ---- 이동 속도 부류 (사용자 2026-10-12) ---------------------------------------------------------------------------
    def _scan_speed(self) -> float:
        """The measurement speed (MOVE_MEASURE: the passes and the precise corrections at the dwell sites): the job's speed_mps under
        the platform's TCP limit. The job's speed_mps of this run is the run override when one was given (measure_speed_mps: the
        AutoTeach goal / client --measure-speed), else the job file's value."""
        return min(float(self.job0.speed_mps), float(self.cfg.v_eff_max_mps))

    def _class_speed(self, speed_class: str) -> float:
        """A move's commanded TCP speed from its speed class: MOVE_MEASURE = the job's measurement speed (_scan_speed), MOVE_POSITION =
        the positioning speed of the straight moves (local_speed: the platform TCP limit cfg.v_eff_max_mps unless the operator lowered
        it). The platform's joint velocity limit applies to both in the time parameterisation."""
        if speed_class == MOVE_MEASURE:
            return self._scan_speed()
        if speed_class == MOVE_POSITION:
            return float(self.local_speed)
        raise ValueError(f"speed class must be one of {SPEED_CLASSES}, got {speed_class!r}")

    def _at_speed(self, job: ScanJob, speed_class: str) -> ScanJob:
        """The job of a recorded leg at its class's speed: the leg's record (reconstruction ramp zones, spacing, the settling evidence,
        the pass overscan's ramp and settling terms) carries the speed the leg is moved at."""
        v = self._class_speed(speed_class)
        return job if float(job.speed_mps) == v else dataclasses.replace(job, speed_mps=v)

    def _decide_move_speeds(self) -> None:
        """Setup decision line: the two speed classes with the values of this run (job, platform config, operator lowering)."""
        lim = (dict(self.settings.params or {}).get("moveit_jtc") or {}).get("limits") or {}
        qd = lim.get("max_joint_velocity_rad_s")
        v_m, v_p = self._scan_speed(), self._class_speed(MOVE_POSITION)
        fps = float(self.job0.profile_fps)
        capped = v_m < float(self.job0.speed_mps)
        self._decide("setup", f"move speeds (each move's purpose decides its class): measurement {v_m * 1e3:.1f} mm/s "
                              f"({self.measure_speed_text()}"
                              + (f"; {float(self.job0.speed_mps) * 1e3:.1f} mm/s capped at the platform TCP limit" if capped else "")
                              + f"; spacing {v_m / fps * 1e3:.2f} mm at {fps:g} fps) for the passes and the precise corrections "
                              f"at the dwell sites; positioning {v_p * 1e3:.1f} mm/s (platform TCP limit "
                              f"{float(self.cfg.v_eff_max_mps) * 1e3:.1f} mm/s"
                              + ("" if v_p >= float(self.cfg.v_eff_max_mps) else ", lowered by the operator (local_speed_mps)")
                              + (f", joints {float(qd):g} rad/s" if qd is not None else "")
                              + f"; joint moves {min(float(self.transit_speed), float(self.cfg.v_eff_max_mps)) * 1e3:.1f} mm/s, descent "
                              f"{float(self.descend_speed) * 1e3:.1f} mm/s) for every other move: start pose, acquisition, descent, "
                              f"the coarse S->E transit (recording, spacing {v_p / fps * 1e3:.2f} mm), pass repositions / approaches",
                     speed_classes={MOVE_MEASURE: v_m, MOVE_POSITION: v_p},
                     measure_speed={**self.measure_speed_record(), "commanded_mps": v_m})

    # ---- 센서 백업은 verification/ ----------------------------------------------------------------------------------
    def _sensor(self) -> None:
        """센서 접속·초기화 (첫 이동 전). 접속·초기화가 실패하면 (D9) 이동 전이면 REFUSED (보낸 것 없음), 이동 뒤면 ERROR — 둘 다
        분류 sensor (인터페이스 r3 §8-5: 예전에는 예외 그대로 ERROR 'OSError: …')."""
        if self.sensor_ready:
            return
        from .executor import ensure_sensor
        # the frame rate the job is designed and reconstructed for (spacing = v / fps; the reconstruction's counter clock and stall
        # model use job.profile_fps) is the rate the sensor runs at: AutoTriggerFPS = job.profile_fps on top of the given / platform set
        from vc3dz_driver import params as VP
        init = dict(self.sensor_init) if self.sensor_init else dict(VP.INIT_STANDARD, **dict(
            (getattr(self.settings, "sensor", None) or {}).get("init_params") or {}))
        fps_job = float(self.job0.profile_fps)
        if fps_job > 0 and float(init.get("AutoTriggerFPS", 0.0) or 0.0) != fps_job:
            self._decide("sensor", f"sensor AutoTriggerFPS {init.get('AutoTriggerFPS')} -> {fps_job:g} (the job's profile_fps)")
            init["AutoTriggerFPS"] = fps_job
        try:
            ensure_sensor(self.sess, self.settings, sensor_ip=self.conn["sensor_ip"], sensor_port=self.conn["sensor_port"],
                          sensor_init=init, backup_dir=str(self.vdir / "sensor_backup"), log=self.log)
        except _Abort:
            raise
        except Exception as e:  # noqa: BLE001 - any connection / initialisation failure of the sensor (D9)
            where = "after motion" if self._motion_sent else "before any motion (nothing sent)"
            raise _Abort(STATUS_ERROR if self._motion_sent else STATUS_REFUSED,
                         f"sensor connection/initialisation failed {where}: {type(e).__name__}: {e} (D9; sensor "
                         f"{self.conn['sensor_ip'] or self.settings.sensor_ip}:{self.conn['sensor_port'] or self.settings.sensor_port})",
                         category="sensor")
        self.sensor_ready = True

    # ---- 작업 프로세스 ------------------------------------------------------------------------------------------------
    def _start_worker(self) -> None:
        if self.worker is None:
            self.worker = _gw.G1Worker(self._cancel_r, log=lambda s: self._decide("worker", s))
        self.worker.start()

    def _wcall(self, job: str, payload):
        if self.worker is None:
            self._start_worker()
        try:
            return self.worker.call(job, payload)
        except _gw.Cancelled:
            raise _CancelAbort(f"during '{job}'")
        except _gw.WorkerDied as e:
            raise _Abort(STATUS_ERROR, f"{e} (D8)", category="internal")
        except _gw.WorkerError as e:
            self.res.notes.append(f"worker job '{job}' failed:\n{e.remote_tb}")
            raise _Abort(STATUS_ERROR, f"compute job failed: {e}", category="internal")

    # ---- 구간 기록 (드웰 grab · 트랜싯 · 패스) --------------------------------------------------------------------
    def _f0(self) -> dict:
        return self.prior.frame0()

    def _dir_SE(self, a, b) -> int:
        """a → b 진행의 부호 (+1 = S→E 쪽)."""
        return _sgn(float((np.asarray(b, dtype=float) - np.asarray(a, dtype=float)) @ self._f0()["u0"]))

    def _s(self, P) -> float:
        """S 에서 S→E 축 (사전 틀 F0) 을 따른 거리 [m]."""
        f = self._f0()
        return float((np.asarray(P, dtype=float) - f["S0"]) @ f["u0"])

    def _profile_prior(self, R_opt: np.ndarray, *, est_dir: Optional[tuple] = None, pitch_deg: float = 0.0) -> dict:
        """§3.2 사전: c = 1/|cos ψ| (ψ = 레이저 평면 법선 ∠ 심 방향 — 추정 전 1), σ_c = 1/cos(k σψ) − 1 (추정 전 ψ_b: 거친 시야가 받는 가장 큰
        yaw), φ = 레이저 평면의 판 법선 기울기 (패스 피치; 드웰·트랜싯 0). est_dir = (û, k·σ_ψ) — 추정의 방향과 그 k·σ (WP2 psi_res_rad)."""
        if est_dir is None:
            c, psi = 1.0, float(self._yaw_band_rad())
        else:
            u_hat, psi = est_dir
            c = 1.0 / max(abs(float(np.asarray(R_opt)[:, 1] @ unit(u_hat))), 1e-9)
            psi = abs(float(psi))
        return {"c": c, "sigma_c": 1.0 / math.cos(psi) - 1.0, "phi_rad": math.radians(float(pitch_deg)),
                "qc_targets": dict(self.doc.get("qc_targets") or {})}           # 루트면 설계 사전 (job root_face_mm {design, tol})

    def _bevel_x_sign(self, est: Optional[SeamEstimate], R_opt: np.ndarray) -> Optional[int]:
        """추정이 정한 베벨 쪽 (base, w0 부호) → 이 자세의 센서 x 쪽 (추정 전 None: WP1 이 두 거울 템플릿을 비교, D10)."""
        if est is None:
            return None
        w = self._f0()["w0"]
        return int(est.bevel_sign_w) * _sgn(float(np.asarray(R_opt)[:, 0] @ w))

    def _n_children(self) -> int:
        return max(1, min((os.cpu_count() or 1) - 4, 16))                 # [R] §8.5

    def _leg_job(self, name: str, kind: str, direction: int, job_rec: ScanJob, records: list, samples: list, *, res=None,
                 T_bf=None, bevel_x_sign=None, profile_prior=None, extra: Optional[dict] = None, est=None,
                 site_inputs: Optional[dict] = None, leg_plan: Optional[dict] = None, stream: Optional[dict] = None,
                 ctrl: Optional[dict] = None, leg_dir=None, register: bool = True) -> dict:
        """작업 프로세스 'leg': 원자료 먼저 (압축 없이, 원자적) → 재구성 (드웰: 정지 자세) → WP1 프로파일 맞춤 → profiles.npz. 스트림
        구간 (트랜싯·패스) 은 계획한 직선 레그 (leg_plan: 패스 예측의 기준) 와, 녹화 중 맞춘 스트림이 있으면 그 토큰·문맥 키를 함께 보낸다
        (작업 프로세스가 같은 입력의 행만 쓴다 — 결과는 스트림과 무관). leg_dir / register False (실행 뒤 실험, experiments.py): 그 폴더에
        같은 규약으로 쓰고 실행의 구간 목록·패스·추정 입력에는 넣지 않는다 (실행의 결과를 바꾸지 않는다)."""
        idx = len(self.legs) + 1
        leg_dir = self.vdir / "legs" / f"{idx:02d}_{name}" if leg_dir is None else Path(leg_dir)
        run = ScanRunResult(status=STATUS_OK, reason=f"{kind} {name}", out_dir=str(leg_dir), records=records, robot_samples=samples,
                            motions=[(name, res)] if res is not None else [], scan_motion=res, sensor_info=self.ss.info.as_dict(),
                            safety_init=self.rs.backend.safety_init_result)
        run.t_sensor_start_ns = self.ss.t_start_ns
        run.t_sensor_stop_ns = self.ss.t_stop_ns
        arrays, js = raw_payload(run, job_rec, self.calib, self.settings, self.cfg, self.rs.pose)
        if T_bf is not None:
            arrays["still_T_base_flange"] = np.asarray(T_bf, dtype=float)
        if ctrl:                                                 # the controller's command / measured joints over the leg (ctrl_*)
            arrays.update(ctrl)
        v = float(job_rec.speed_mps) if kind != "dwell" else 0.0
        payload = dict(name=name, kind=kind, out_leg_dir=str(leg_dir), raw_arrays=arrays, raw_json=js, frames=frames_from_records(records),
                       robot=self.rs.pose.robot_samples(samples) if samples else [], T_base_flange=T_bf, calib=self.calib, job=job_rec,
                       spec=self.spec, tol=self.tol, cls=kind, direction=int(direction), bevel_x_sign=bevel_x_sign,
                       predict=None, profile_prior=profile_prior, z_valid_mm=tuple(self.settings.z_valid_mm), k=self.k,
                       cloud_step_m=self._vis_step_m() if (kind != "dwell" and register) else None,
                       n_children=self._n_children(), leg_plan=leg_plan,
                       stream_token=(stream or {}).get("token"), stream_key=(stream or {}).get("key"),
                       est=est, **(extra or {}), **(site_inputs or {}),
                       meta={"kind": kind, "speed_mps": v, "standoff_m": float(job_rec.standoff_m), "pitch_deg": float(job_rec.pitch_deg),
                             "fps": float(job_rec.profile_fps), "spacing_m": v / float(job_rec.profile_fps), **(extra or {})})
        out = self._wcall("leg", payload)                       # 맞춤은 단계를 바꾸지 않는다 (r3 §8-7: 줄 머리 = 지금 단계, '[dwell] fit: …')
        if not register:
            return out
        self.legs.append(out["leg_path"])
        info = {"k": idx, "name": name, "kind": kind, "class": kind, **(extra or {}),
                "direction": ("S->E" if direction > 0 else "E->S") if kind != "dwell" else None, "dir": str(leg_dir),
                "n_frames": len(records), "bevel_x_sign": out.get("bevel_x_sign"), **(out.get("summary") or {}),
                "timing": out.get("timing")}
        if stream is not None:
            info["stream"] = {"tap": stream.get("tap_stats"), "worker": (out.get("summary") or {}).get("stream")}
        self.leg_info.append(info)
        if kind == "pass":                                       # 결과의 패스 = 진짜 패스만 (r3 §8-2); 모든 구간은 run_log 'legs'
            self.res.passes.append(_jsonable(info))
        self._decide("fit", f"{name}: {len(records)} frames -> {info.get('n_profiles')} profiles, {info.get('groove_profiles')} with "
                                 f"groove evidence (lnBF > 0), sigma_r median {info.get('sigma_r_mm_median', float('nan')):.3f} mm "
                                 f"[{', '.join(f'{k_[:-2]} {v_:.2f} s' for k_, v_ in (out.get('timing') or {}).items() if k_.endswith('_s'))}]",
                     leg=_jsonable(info))
        cloud = out.get("cloud")
        if cloud is not None:
            self._event("pass_cloud", k=idx, name=name, points=cloud, base_frame=self.job0.base_frame)
        return out

    def _record_leg(self, name: str, T_to, job_rec: ScanJob, kind: str, direction: int, *, speed_class: str, bevel_x_sign=None,
                    profile_prior=None, est=None, leg_dir=None) -> dict:
        """스트리밍 직선 (트랜싯·패스; D3/D4 → 확인) → 'leg' 일감. 중단되면 받은 프레임까지 verification/legs/NN_name/raw.npz.
        녹화 동안 프레임은 녹화기 탭으로 작업 프로세스에 가서 맞춰진다 (_open_stream) — 이동 뒤에는 남은 행만 (결과는 같다).
        speed_class: 이 구간의 목적이 정하는 속도 부류 (패스 MOVE_MEASURE, 트랜싯 MOVE_POSITION) — 구간 기록 (job_rec) 이 그 속도를 담는다."""
        job_rec = self._at_speed(job_rec, speed_class)
        register = leg_dir is None                               # 실행 뒤 실험 (leg_dir): 실행의 구간·패스가 아니다
        if kind == "pass" and register:                          # pass_started 의 k = 패스 번호; 트랜싯은 내지 않는다 (r3 §8-2)
            self._event("pass_started", k=int(self.n_pass), name=name, direction="S->E" if direction > 0 else "E->S")
        # 계획한 직선 레그 = 지금 잰 자세 → T_to (패스 예측의 기준 — g1_stream.LegPredictor; 스트림과 이동 뒤 맞춤이 같은 값을 쓴다)
        leg_plan = {"T_from": np.array(self._T_now(), dtype=float), "T_to": np.array(T_to, dtype=float),
                    "T_flange_opt": np.array(self.calib.T_flange_opt, dtype=float), "step_m": float(self.prior.station_step_m)}
        fit = {"spec": self.spec, "tol": self.tol, "profile_prior": profile_prior, "bevel_x_sign": bevel_x_sign, "cls": kind,
               "z_valid_mm": tuple(self.settings.z_valid_mm), "est": est, "leg_plan": leg_plan}
        stream = self._open_stream(name, kind, direction, fit)
        try:
            res, records, samples = self._leg(T_to, "scan", name, speed_class=speed_class, streaming=True,
                                              extra=f"({kind} {name}, {'S->E' if direction > 0 else 'E->S'}, sensor recording)")
        except _Abort as e:
            self._save_partial(name, job_rec, e, leg_dir=leg_dir)
            raise
        finally:
            self._close_stream_tap(stream)
        self._t_leg_end = time.perf_counter()             # 기록 (R-측정): 녹화가 끝난 뒤 → 'leg' 결과·다음 추정까지의 벽시계 시간
        self._settle_devs.append(self._leg_settle(samples, res, job_rec))
        out = self._leg_job(name, kind, direction, job_rec, records, samples, res=res, bevel_x_sign=bevel_x_sign,
                            profile_prior=profile_prior, est=est, leg_plan=leg_plan, stream=stream, ctrl=self._controller_record(samples),
                            leg_dir=leg_dir, register=register)
        if register:
            self.leg_info[-1]["after_recording_wall_s"] = {"leg_job": round(time.perf_counter() - self._t_leg_end, 3)}
        return out

    def _leg_settle(self, samples: list, res, job_rec: ScanJob) -> Optional[dict]:
        """The leg's encoder along-track deviation from its commanded trapezoid during cruise (scan_core.settling.leg_deviation) —
        the run's evidence of the servo's settling after the start ramp. None when the leg cannot give it."""
        from scan_core.settling import leg_deviation
        try:
            # the leg as commanded by this run (the platforms record v_cmd / a_cmd in different units): the leg's cruise speed (its
            # record carries its speed class's speed - transit at the positioning speed, passes at the measurement speed) and the
            # run's acceleration (already the platform's cap, _motion_config)
            v, a = float(job_rec.speed_mps), float(self.accel)
            t = np.array([float(s_.t_rx_ns) * 1e-9 for s_ in samples])
            P = np.array([np.asarray(s_.T_base_flange, dtype=float)[:3, 3] for s_ in samples])
            return leg_deviation(t, P, v, a)
        except Exception as e:  # noqa: BLE001 - evidence only: the overscan then rests on the other legs / the lag prior
            self.res.notes.append(f"settling evidence of the leg not computed: {type(e).__name__}: {e}")
            return None

    def _settle(self) -> dict:
        """The servo's settling time after a pass's start ramp (scan_core.settling.settle_time): the run's legs so far with the prior
        of the controller's estimated command-to-measurement lag (platform tracking_residual τ̂)."""
        from scan_core.settling import settle_time
        tau = None
        fn = getattr(self.rs, "tracking_residual", None) if self.sess is not None else None
        if callable(fn):
            try:
                snap = (fn() or (None,))[0]
                tau = None if snap is None else float(snap.tau_s)
            except Exception:  # noqa: BLE001 - no lag estimate: the data alone
                tau = None
        return settle_time(self._settle_devs, tau)

    def _cruise_terms(self, v_mps: float) -> tuple:
        """What a recorded leg at the speed v needs beyond its span to cross the span at steady cruise (user 2026-10-11 night): the
        speed ramp v²/2a at each end (the run's acceleration, the platform's cap) and the servo settling v·t_settle after the start
        ramp - at the first end only (_settle: the run's legs with the controller-lag prior) → (ramp_m, settle_m, settling record).
        The passes (_pass) and the post-weld scan (post_weld.PostWeldScan) add these to their spans."""
        v = float(v_mps)
        ramp = v ** 2 / (2.0 * float(self.accel)) if float(self.accel) > 0 else 0.0
        st = self._settle()
        return ramp, v * float(st["t_settle_s"]), st

    def _controller_record(self, samples: list) -> dict:
        """The platform's controller states (command and measured joints, platform controller_history) over the leg's robot samples —
        recorded with the leg for the pose smoother (scan_core.pose_smoother); {} when the platform has none."""
        fn = getattr(self.rs, "controller_history", None)
        if not callable(fn) or not samples:
            return {}
        try:
            t = [int(getattr(s_, "t_rx_ns")) for s_ in samples]
            return dict(fn(min(t), max(t)) or {})
        except Exception as e:  # noqa: BLE001 - a record only: the leg is kept without it
            self.res.notes.append(f"controller states not recorded: {type(e).__name__}: {e}")
            return {}

    def _open_stream(self, name: str, kind: str, direction: int, fit: dict) -> Optional[dict]:
        """녹화 전: 작업 프로세스에 이 레그의 맞춤 문맥으로 스트림을 열고 ('stream_open'), 녹화기에 탭 (StreamTap — 녹화 스레드에서 막지
        않는 send 만) 을 단다. 패스는 탭이 프레임마다 그 수신 시각의 플랫폼 자세 (pose_at) 를 힌트로 붙인다 (속도에만 쓰인다). 스트림은
        판정이 아니다: 열지 못하면 (작업 프로세스에 스트림 통로가 없다·일감 오류) 이동 뒤에 모든 행을 맞출 뿐이다 — 기록만."""
        if self.worker is None or self.sess is None or self.sess.sensor is None:
            return None
        self._stream_seq = int(getattr(self, "_stream_seq", 0)) + 1
        token = (os.getpid() << 24) | self._stream_seq
        key = _gw.stream_key(fit)
        try:
            out = self.worker.call("stream_open", {**fit, "stream_token": token, "stream_key": key, "name": name,
                                                   "direction": int(direction), "n_children": self._n_children(),
                                                   "chunk_rows": _gw.STREAM_CHUNK_ROWS})
        except _gw.Cancelled:
            raise _CancelAbort(f"before '{name}'")
        except _gw.WorkerDied as e:
            raise _Abort(STATUS_ERROR, f"{e} (D8)", category="internal")
        except _gw.WorkerError as e:
            self.res.notes.append(f"{name}: stream not opened ({e}) - every profile is fitted after the leg")
            return None
        if not out.get("stream") or self.worker.stream_sock is None:
            self.res.notes.append(f"{name}: no stream ({out.get('why')}) - every profile is fitted after the leg")
            return None
        pose_at = getattr(self.rs.pose, "pose_at", None) if fit.get("est") is not None else None
        tap = _gw.StreamTap(self.worker.stream_sock, token, pose_at=pose_at, dt_s=float(self.calib.dt_s))
        self.ss.add_tap(tap)
        return {"token": token, "key": key, "tap": tap, "worker_pid": out.get("pid")}

    def _close_stream_tap(self, stream: Optional[dict]) -> None:
        """녹화가 끝난 뒤 (또는 중단): 탭을 떼고 남은 것을 막지 않고 한 번 보낸 뒤 멈춘다 (기록은 stream['tap_stats'])."""
        if not stream or stream.get("tap") is None:
            return
        tap = stream.pop("tap")
        try:
            self.ss.remove_tap(tap)
        finally:
            stream["tap_stats"] = tap.close()

    def _save_partial(self, name: str, job_rec: ScanJob, e: _Abort, leg_dir=None) -> None:
        if self._last_stream is None:
            return
        leg_dir = self.vdir / "legs" / f"{len(self.legs) + 1:02d}_{name}" if leg_dir is None else Path(leg_dir)
        recs_p, samp_p, _res_p = self._last_stream
        try:
            run_p = ScanRunResult(status=e.status, reason=e.reason, out_dir=str(leg_dir), records=recs_p, robot_samples=samp_p,
                                  sensor_info=self.ss.info.as_dict() if self.sess.sensor is not None else {})
            arrays, js = raw_payload(run_p, job_rec, self.calib, self.settings, self.cfg, self.rs.pose)
            _gw.atomic_savez(leg_dir / "raw.npz", **arrays, **{k_: np.array(v_) for k_, v_ in js.items()})
            self.res.files.append(str(leg_dir / "raw.npz"))
            self._decide(self._phase, f"{name}: stopped ({e.status}); partial data saved ({len(recs_p)} frames) -> {leg_dir / 'raw.npz'}")
        except Exception as ex:  # noqa: BLE001
            self.res.notes.append(f"{name}: partial save failed: {type(ex).__name__}: {ex}")

    def _grab_leg(self, name: str, *, site: str, grab: int, est: Optional[SeamEstimate] = None) -> dict:
        """정지 grab (dwell_frames 개) → 드웰 LegObs ('leg' 일감, 정지 자세) → 그 자리 상태
        (out['site_state'] = WP2 site_state, spec §4.8). 열별 중앙값이 유령점 버스트 (g = sensor.ghost_burst_frames 프레임) 를 이기려면
        grab 이 2g + 1 프레임은 있어야 한다 (spec §3.6a) — 센서가 그만큼 주지 않으면 D9 (센서 살아 있음: RESET 한 번 다시, 그래도 모자라면
        REFUSED)."""
        self._check_cancel()
        self._check_estop(f"before dwell '{name}'")
        n = int(self.dwell_frames)
        need = 2 * int(self.ghost_burst_frames) + 1                     # ≤ n (D1)
        fps = float(self.job0.profile_fps) if float(self.job0.profile_fps) > 0 else 1.0
        timeout = float(self.at.first_frame_timeout_s) + n / fps + float(self.settings.sensor.get("data_wait_ms", 1000) or 0) * 1e-3
        recs: list = []
        for attempt in (1, 2):                                          # D9: RESET 한 번 다시 (spec §2)
            t0_ns = self._clock.monotonic_ns()
            recs = self.ss.grab(n, timeout_s=timeout)
            if len(recs) >= need:
                break
            if attempt == 1:
                self._decide("dwell", f"dwell '{name}': the sensor gave {len(recs)} frames (< 2 g + 1 = {need}) - one RESET retry in "
                                      "place (D9)")
        if len(recs) < need:
            raise _Abort(STATUS_REFUSED, f"dwell '{name}': the sensor gave {len(recs)} frames twice (< 2 g + 1 = {need} for the "
                                         f"column-wise median against ghost bursts of g = {int(self.ghost_burst_frames)}; D9)",
                         category="sensor")
        self._check_estop(f"after dwell '{name}'")
        snap = self.rs.pose.snapshot()
        T_bf = np.asarray(snap.T, dtype=float)
        samples = self.rs.pose.history(t0_ns)
        R_opt = (T_bf @ self.calib.T_flange_opt)[:3, :3]
        pp = self._profile_prior(R_opt) if est is None else self._profile_prior(R_opt, est_dir=(est.u, self._psi_res(est)))
        r_t = self._coarse_plate_m()
        return self._leg_job(name, "dwell", DWELL_DIRECTION, self.job_c, recs, samples, T_bf=T_bf,
                             bevel_x_sign=self._bevel_x_sign(est, R_opt), profile_prior=pp, extra={"site": site, "grab": grab}, est=est,
                             site_inputs={"prior": self.prior, "view": r_t, "goal_tol_m": self._hit_tol_m(r_t)})

    def _psi_res(self, est: SeamEstimate) -> float:
        """ψ_res = k·σ 의 심 방향 (WP2 plan_quantities)."""
        return float(self._plan_quantities(est)["psi_res_rad"])

    # ---- 드웰 + 정밀 보정 ------------------------------------------------------------------------------------------
    def _dwell_site(self, X: str, *, est: Optional[SeamEstimate]) -> dict:
        """점 X ('S' = 먼저 간 점, 'E') 의 드웰 + 정밀 보정 (spec r2 §4.5, §4.8, §5.1): grab (WP1 fit_dwell, 열별 중앙값) → 작업 프로세스의
        WP2 site_state (루트·σ_lat/σ_nor·롤·거리·yaw 와 목표 자세 T*: 루트가 광축 위, 판 상면이 거친 판 거리, 광축 ⟂ 판, 두 번째 점은 레이저
        평면 ⟂ 융합 방향 û) → **첫 grab 만** D11 (2·λ > k², 아니면 DATA_FAIL 'groove not in view' — 탐색 없음; 뒤 grab 은 따로 시험하지
        않는다 — 융합의 연한 가중) → D15 (needs_correction: k·max(σ_lat, σ_nor)/tol_root > 1 이고 T* 에서 더 작다) 이면 T* 로 옮기고 다시
        grab, 아니면 이 자리는 끝. 고정 횟수·띠 없음. 그 자리의 판독 = 맞춤이 있는 가장 최근 grab (뒤 grab 에 맞춤이 없으면 site_state 는
        옮길 곳이 없다 (needs_correction 거짓) → 그 앞 판독으로 끝)."""
        g = 0
        e_goal = self._hit_tol_m(self._coarse_plate_m())
        U = X                                                              # 'S' = 먼저 간 끝, 'E' = 다른 끝 (모든 이름·로그에 그대로)
        last: Optional[tuple] = None                                       # (grab 번호, site_state, 'leg' 결과) — 맞춤이 있는 가장 최근
        while True:
            g += 1
            self._set_phase("dwell")
            out = self._grab_leg(f"dwell_{U}{g}", site=U, grab=g, est=est)
            st = out["site_state"]
            if g == 1:                                                                  # D11 (spec §4.5): 첫 grab 의 중앙값 프로파일
                lnbf = float(out["log_bf"])
                if not 2.0 * lnbf > self.k ** 2:
                    raise _Abort(STATUS_DATA_FAIL, f"groove not in view at the {U} rough point (D11: first grab 2 lnBF "
                                                   f"{2.0 * lnbf:.1f} <= k^2 {self.k ** 2:g}; no lateral search)", category="sensing")
                if not st.get("ok"):
                    raise _Abort(STATUS_ERROR, f"the profile model gave no fit for the first grab at {U} although it "
                                               f"passed D11 (2 lnBF {2.0 * lnbf:.1f}): {st.get('why')}", category="sensing")
            if st.get("ok"):
                last = (g, st, out)
            else:
                self._decide("dwell", f"{U} grab {g}: no fit in this grab ({st.get('why')}) - nothing to correct onto (D15); the site keeps "
                                      f"grab {last[0]}'s reading", site_state=_jsonable(st))
            g_site, st, out_site = last
            mm, deg = 1e3, 180.0 / math.pi
            q = {"lateral": (st["x_R_m"], st["sigma_lat_m"], mm, "mm"),
                 "standoff": (st["standoff_err_m"], st["sigma_standoff_m"], mm, "mm"),
                 "roll": (st["roll_rad"], st["sigma_roll_rad"], deg, "deg")}
            if st.get("yaw_rad") is not None:
                q["yaw"] = (st["yaw_rad"], st["sigma_yaw_rad"], deg, "deg")
            txt = ", ".join(f"{nm} {v * f:+.2f} {un} (k sigma {self.k * sg * f:.2f})" for nm, (v, sg, f, un) in q.items())
            rule = (f"k max(sigma_lat, sigma_nor)/tol_root {st['ratio']:.2f} (at the target {st['ratio_at_target']:.2f}; e_goal "
                    f"{e_goal * 1e3:.2f} mm)")
            rec = _jsonable({k_: v_ for k_, v_ in st.items() if k_ != "T_target"})
            if g_site != g or not st["needs_correction"]:
                self._decide("dwell", f"{U} grab {g}: site done (D15: {rule}) - {txt}" + ("" if g_site == g else f" (grab {g_site})"),
                             site_state=rec)
                # 판 상면 점의 σ (n̂ 방향) = 맞춤의 거리 σ (site_state) ⊕ 드웰 부류 오프셋 σ_he (자세와 무관, spec §4.2)
                site = {"root_m": np.asarray(st["root_m"], dtype=float), "n": np.asarray(st["n"], dtype=float),
                        "u": np.asarray(st["u"], dtype=float), "depth_m": float(st["depth_m"]),
                        "sigma_top_m": math.hypot(float(st["sigma_standoff_m"]), float(self.prior.he_sigma_m)), "grabs": g,
                        "grab_read": g_site, "leg": out_site["leg_path"], "s_m": self._s(st["root_m"])}
                self.sites[X] = site
                self.res.sites[U] = _jsonable(site)
                return site
            self._decide("dwell", f"{U} grab {g}: correct (D15: {rule}) - {txt}", site_state=rec)
            self._set_phase("correct")
            T_to = np.asarray(st["T_target"], dtype=float)
            n = np.asarray(st["n"], dtype=float)
            Tfo = self.calib.T_flange_opt
            dvec = (T_to @ Tfo)[:3, 3] - (self._T_now() @ Tfo)[:3, 3]
            dn = float(dvec @ n)
            ph = self._leg_phase(dn, float(np.linalg.norm(dvec - n * dn)),
                                 roll_deg=math.degrees(abs(float(st["roll_rad"])) + abs(float(st.get("yaw_rad") or 0.0))),
                                 orientation=self.orientation, tol_m=self._sv().pose_tol_m)
            # the precise correction at the dwell site: a measurement move (user 2026-10-12)
            self._leg(T_to, ph, f"{U}_corr{g}", speed_class=MOVE_MEASURE, extra=f"(precise correction at the {U} point)")

    def _transit_name(self) -> str:
        """트랜싯 구간 이름 (S = 먼저 간 끝 → E)."""
        return "transit_S_to_E"

    def _transit(self, site_S: dict) -> dict:
        """S→E 트랜싯 (지금처럼: 보정된 S 자세 그대로, 잰 판 평면 안 평행이동으로 E 착점 위) — 스트리밍, 추정에 융합. 위치 잡기 부류
        (사용자 2026-10-12: 플랫폼 한계) — 그 판독은 자기 속도로 추정에 들어간다 (잡음 사전의 속도 모형은 잰 진행 속도)."""
        T_to = np.array(self._T_now(), dtype=float, copy=True)
        dvec = np.asarray(self.job0.end, dtype=float) - np.asarray(site_S["root_m"], dtype=float)
        n_S = np.asarray(site_S["n"], dtype=float)
        T_to[:3, 3] += dvec - n_S * float(dvec @ n_S)
        R_opt = (T_to @ self.calib.T_flange_opt)[:3, :3]
        return self._record_leg(self._transit_name(), T_to, self.job_c, "transit", self._dir_SE(site_S["root_m"], self.job0.end),
                                speed_class=MOVE_POSITION, profile_prior=self._profile_prior(R_opt))

    # ---- 추정 ----------------------------------------------------------------------------------------------------------
    def _estimate(self) -> SeamEstimate:
        n = len(self.est_history) + 1
        ph = self._phase
        self._set_phase("estimate")
        est = self._wcall("estimate", {"leg_paths": list(self.legs), "prior": self.prior,
                                        "est_path": str(self.vdir / "estimates" / f"est_{n}.json"),
                                        "n_children": max(1, min((os.cpu_count() or 1) - 4, 16))})
        t_leg_end, self._t_leg_end = getattr(self, "_t_leg_end", None), None
        if t_leg_end is not None and self.leg_info:                # 스트림 구간 뒤 첫 추정: 녹화 끝 → 추정 준비 (벽시계, 기록만)
            self.leg_info[-1].setdefault("after_recording_wall_s", {})["estimate_ready"] = round(time.perf_counter() - t_leg_end, 3)
        eS, eE = (est.ends or {}).get("S"), (est.ends or {}).get("E")
        L = _gw.models().delivered_length(est)[0] * 1e3 if (eS is not None and eE is not None) else float("nan")   # the seam length
        mc = self._model_checks(est)
        pl_ = est.plate or {}
        do_, to_ = pl_.get("direction_offset") or {}, pl_.get("time_offset") or {}
        rec_dir = {"b_mm": [None if (v_ is None or not math.isfinite(float(v_))) else round(float(v_) * 1e3, 4) for v_ in
                            (do_.get("b_m") or [])],
                   "sigma_mm": [None if (v_ is None or not math.isfinite(float(v_))) else round(float(v_) * 1e3, 4) for v_ in
                                (do_.get("sigma_m") or [])],
                   "identified": list(do_.get("identified") or []),
                   "leg_offset_sigma_wn_mm": [round(float(v_) * 1e3, 4) for v_ in (pl_.get("leg_offset_sigma_wn_m") or [])],
                   "root_established": list(pl_.get("root_established") or []),
                   "delta_ms": round(float(to_.get("delta_s", 0.0)) * 1e3, 3) if to_ else None,
                   "delta_sigma_ms": round(float(to_.get("sigma_s", float("nan"))) * 1e3, 3) if to_ else None,
                   "delta_prior_sigma_ms": round(float(to_.get("prior_sigma_s", float("nan"))) * 1e3, 3) if to_ else None,
                   "delta_separable": bool(to_.get("separable_from_b_u", False)) if to_ else None,
                   "joint_state": self._joint_state_rec(to_.get("joint_state_source")) if to_ else None}
        rec = {"n": n, "legs": len(self.legs), "groove_found": bool(est.groove_found), "log_bf_run": float(est.log_bf_run),
               "direction_time": rec_dir,
               "done": bool(est.done), "ratios": _jsonable(est.ratios), "binding": list(est.binding or []),
               "unresolved": list(est.unresolved or []), "length_mm": L,
               "ends": {X: {"s_m": float(e.s_m), "sigma_m": float(e.sigma_m), "two_sided": bool(e.two_sided)}
                        for X, e in (est.ends or {}).items()},
               "model_checks": _jsonable({"summary": mc.get("summary"), "hypotheses": {
                   h: {k_: v_ for k_, v_ in d_.items() if k_ != "per_leg_ln_bf_w"} for h, d_ in (mc.get("hypotheses") or {}).items()},
                   "em_rounds": mc.get("em_rounds")}) if mc else None}
        self.est_history.append(rec)
        mx = max([float(v) for v in (est.ratios or {}).values() if v is not None and math.isfinite(float(v))] or [float("nan")])
        self._decide("estimate", f"estimate {n} ({len(self.legs)} legs): groove {'found' if est.groove_found else 'NOT found'} "
                                 f"(run lnBF {float(est.log_bf_run):.1f}); ends S {self._end_txt(eS)}, E {self._end_txt(eE)}; length "
                                 f"{L:.2f} mm; done {bool(est.done)} (max sigma/precision {mx:.2f}; binding {list(est.binding or [])[:4]})"
                                 + self._direction_txt(rec_dir) + self._checks_txt(mc), estimate=rec)
        self._joint_state_line(rec_dir.get("joint_state"), (to_ or {}).get("joint_state_source"))
        self._set_phase(ph)
        return est

    @staticmethod
    def _joint_state_rec(js: Optional[dict]) -> Optional[dict]:
        """The estimate record's joint-state source (seam_estimate.joint_state_report): hypothesis, posterior, evidence per feature,
        the time-offset prior used (run_log 'estimates')."""
        if not js:
            return None
        ev = (js.get("evidence") or {}).get("ln_bf_measured_vs_reference") or {}
        return {"hypothesis": js.get("hypothesis"), "p_reference_echo": js.get("p_reference_echo"),
                "ln_odds_reference": js.get("ln_odds_reference"), "ln_bf_measured_vs_reference": dict(ev),
                "per_leg": list((js.get("evidence") or {}).get("per_leg") or []), "data": js.get("data"),
                "delta_prior": js.get("delta_prior"), "warning": js.get("warning")}

    def _joint_state_line(self, rec: Optional[dict], full: Optional[dict]) -> None:
        """One concise console line per estimate whose hypothesis changed (and the first): which joint states the run's poses are,
        the evidence per feature and the time-offset prior; under a resolved reference echo the echo WARNING (once, in
        the run's notes)."""
        if not rec:
            return
        prev = (self.joint_state_source or {}).get("hypothesis")
        self.joint_state_source = full
        ev = rec.get("ln_bf_measured_vs_reference") or {}
        dp = rec.get("delta_prior") or {}
        if prev != rec.get("hypothesis"):
            p = rec.get("p_reference_echo")
            self._decide("joint_states", f"joint states: {rec.get('hypothesis')} (P(reference echo) {float(p):.3g}, ln odds "
                                         f"{float(rec.get('ln_odds_reference') or 0.0):+.1f}; ln BF measured:reference lattice "
                                         f"{float(ev.get('lattice', 0.0)):+.1f}, rest {float(ev.get('rest', 0.0)):+.1f}); time-offset "
                                         f"prior sigma_q {float(dp.get('sigma_q_s', float('nan'))) * 1e3:.2f} ms, p_ref "
                                         f"{float(dp.get('p_ref', 0.0)):.3g}" + ("; delta = tau_echo >= 0 (unknown lag) + "
                                                                                  "quantisation" if float(dp.get('p_ref', 0.0)) > 0
                                                                                  else ""), joint_state=rec)
        w = rec.get("warning")
        if w and w not in self.res.notes:
            self.res.notes.append(w)
            self._decide("joint_states", w)

    @staticmethod
    def _direction_txt(d: dict) -> str:
        """Decision-line part (English, one line): the robot's motion-direction offset b (w, n, u) and the time offset."""
        if not d or not d.get("identified"):
            return ""

        def f(v):
            return "n/a" if v is None else f"{v:+.3f}"
        b = d.get("b_mm") or [None] * 3
        sg = d.get("sigma_mm") or [None] * 3
        txt = "; direction offset b (w, n, u) " + ", ".join(f"{f(b_)} +- {('inf' if s_ is None else f'{s_:.3f}')}" for b_, s_ in
                                                             zip(b, sg)) + " mm"
        if d.get("delta_ms") is not None:
            txt += (f"; time offset residual {d['delta_ms']:+.2f} +- {d['delta_sigma_ms']:.2f} ms (prior {d['delta_prior_sigma_ms']:.2f}; "
                    f"{'separable' if d.get('delta_separable') else 'not separable'} from b_u)")
        return txt

    @staticmethod
    def _model_checks(est: SeamEstimate) -> dict:
        """§4.9 report of the estimate (seam_estimate.model_checks) — {} for a model without it."""
        return dict(((est.plate or {}).get("model_checks") or {})) if est is not None else {}

    def _checks_txt(self, mc: dict) -> str:
        """Decision-line tail: model fit verdict counts, hypotheses, the first blocking segments (English, one line)."""
        if not mc:
            return ""
        sm = mc.get("summary") or {}
        hyp = mc.get("hypotheses") or {}
        side = hyp.get("bevel_side") or {}
        txt = (f"; model fit: {sm.get('fits', 0)} segments fit, {sm.get('ambiguous', 0)} ambiguous, {sm.get('misfit', 0)} misfit "
               f"(k^2 {self.k ** 2:g} on 2 ln odds)")
        if "two_ln_odds" in side:
            txt += f"; bevel side 2 ln odds {float(side['two_ln_odds']):.1f} ({'resolved' if side.get('resolved') else 'NOT resolved'})"
        if sm.get("legs_not_this_seam"):
            txt += f"; legs that do not show this seam (down-weighted): {', '.join(sm['legs_not_this_seam'])}"
        if sm.get("windows_root_from_own_data"):
            txt += f"; {sm['windows_root_from_own_data']} station(s) off the line: root from the window's own data (local deviation)"
        blk = list(sm.get("misfit_blocking") or []) + list(sm.get("unresolved_blocking") or [])
        if blk:
            txt += f"; not established by the model checks ({len(blk)}): " + " | ".join(b[:160] for b in blk[:3])
        return txt

    @staticmethod
    def _end_txt(e) -> str:
        if e is None:
            return "-"
        return f"{float(e.s_m) * 1e3:.2f} +- {float(e.sigma_m) * 1e3:.2f} mm ({'two-sided' if e.two_sided else 'one-sided'})"

    def _plan_quantities(self, est: SeamEstimate) -> dict:
        """seam_estimate.plan_quantities (WP2, §5.2): e_X_m (S 에서의 s), sigma_plan_X_m, psi_res_rad (= k·σ 심 방향), depth_m (D̂)."""
        q = _gw.models().plan_quantities(est, self.prior)
        out = {"psi_res_rad": float(q["psi_res_rad"]), "depth_m": float(q["depth_m"]), "raw": _jsonable(q)}
        for X in ("S", "E"):
            out[f"e_{X}_m"], out[f"sigma_plan_{X}_m"] = float(q[f"e_{X}_m"]), float(q[f"sigma_plan_{X}_m"])
        return out

    # ---- §5.3 세계·안전 갱신 (S1): 추정마다 세계 모형 한 판 ---------------------------------------------------------------------
    def _top_point(self, est: SeamEstimate) -> tuple:
        """S1 의 판 상면 점과 그 σ (n̂ 방향): 융합 모형의 가장 높은 판 상면 점 (WP2 est.plate['top_point_m'/'sigma_top_m']). 모형에 아직 판
        상면 단면이 없으면 (σ_top = ∞) 보정된 드웰의 판 상면 (루트 + n̂·깊이) 중 가장 높은 것과 그 모형 σ (맞춤의 거리 σ ⊕ 드웰 부류 오프셋
        σ_he — site_state, spec §4.2) — σ 를 0 으로 두지 않는다."""
        pl = est.plate or {}
        P, sP = pl.get("top_point_m"), pl.get("sigma_top_m")
        if P is not None and sP is not None and math.isfinite(float(sP)):
            return np.asarray(P, dtype=float), float(sP)
        n_hat = self._n_hat
        cands = [(np.asarray(site["root_m"], dtype=float) + n_hat * float(site["depth_m"]), float(site["sigma_top_m"]))
                 for site in self.sites.values()]
        if not cands:
            raise _Abort(STATUS_ERROR, "no plate-top point (the model has no plate-top section and no site is read) - the S1 envelope "
                                       "cannot be updated", category="internal")
        return max(cands, key=lambda c: float(self.up @ c[0]))

    def _apply_estimate_to_world(self, est: SeamEstimate) -> None:
        """§5.3 (S1): 세계 = 잰 시편 — 판 상면 = 가장 높은 판 상면 점 + n̂·k·σ_top (n̂ = 잰 판 법선), 시편 = 추정 끝의 사후 (패스 전에는 §4.4
        의 여유 사전 — 남은 설계 길이) 그대로 r(ê_S − k·σ_e,S) … r(ê_E + k·σ_e,E), 지지면 = 그 상면의 가장 낮은 모서리 − 두께 − 뒷댐재 → 포락선
        기준·MoveIt 물체를 한 번에 (양쪽 관측 여부로 가르지 않고, 줄이는 사다리 없이). 장면은 이동 사이에 지우지 않는다."""
        n_hat = unit(np.asarray(est.n_plate, dtype=float))
        if float(n_hat @ self.up) < 0.0:
            n_hat = -n_hat
        self._n_hat = n_hat
        top, s_top = self._top_point(est)
        pe = plan_ends(est)                         # the world box is planning: the ends with the design length (user rule 2026-10-11)
        eS, eE = pe["S"], pe["E"]
        s_lo = float(eS["s_m"]) - self.k * float(eS["sigma_m"])
        s_hi = float(eE["s_m"]) + self.k * float(eE["sigma_m"])
        ends = {X: {"s_m": float(e["s_m"]), "sigma_m": float(e["sigma_m"]), "k_sigma_m": self.k * float(e["sigma_m"])}
                for X, e in (("S", eS), ("E", eE))}
        why = (f"estimate {len(self.est_history)}: ends S {float(eS['s_m']) * 1e3:.1f} - k {float(eS['sigma_m']) * 1e3:.1f}, E "
               f"{float(eE['s_m']) * 1e3:.1f} + k {float(eE['sigma_m']) * 1e3:.1f} mm (s from S; planning ends)")
        self.world.update_measured(top_point=top, sigma_top_m=s_top, normal=n_hat, k=self.k, root_lo=est.r(s_lo), root_hi=est.r(s_hi),
                                   ends=ends, why=why)
        self.world.set_plate_sides(self._left_plate_side(est))
        self._apply_world(why)

    def _left_plate_side(self, est: SeamEstimate) -> Optional[int]:
        """The side (sign of w0: +1 = left looking S->E) of the design's LEFT plate, from the estimate's bevel side once its
        hypothesis is resolved (seam_estimate §4.9); None = not known (single V / square: the sides look alike; or unresolved) —
        the world then puts the wider plate on both sides."""
        sp = self.spec
        if sp is None or str(sp.kind) != "single_bevel" or sp.side_agnostic:
            return None
        side = (((est.plate or {}).get("model_checks") or {}).get("hypotheses") or {}).get("bevel_side") or {}
        if not side.get("resolved", False):
            return None
        bw = int(est.bevel_sign_w)
        return bw if sp.resolved_bevel_side() == "left" else -bw

    # ---- 패스 ----------------------------------------------------------------------------------------------------------
    def _first_pass_targets(self, est: SeamEstimate) -> tuple:
        """패스 1 (사용자의 틀 그대로, 조정 2026-10-09): 보정된 두 지점 사이 + 남은 거리의 절반씩 양쪽 — R = max(0, L_d − |S_c − E_c|),
        각 보정 지점 너머 R/2 (추정 직선을 따라). 끝마다 오버스캔은 센서 발자국 몫만 (k·σ 항 없음: 패스 1 의 범위는 설계 길이가 정한다 —
        조정 결정 2026-10-09 'pass 1 extent literal'). → (s_S, s_E (S 에서의 s), σ_plan = None (k·σ 항 없음), 근거 문구)."""
        sA, sB = float(self.sites["S"]["s_m"]), float(self.sites["E"]["s_m"])
        lo, hi = min(sA, sB), max(sA, sB)
        PA, PB = np.asarray(self.sites["S"]["root_m"], dtype=float), np.asarray(self.sites["E"]["root_m"], dtype=float)
        L_c = float(np.linalg.norm(PB - PA))
        R = max(0.0, float(self.spec.weld_length_mm) * 1e-3 - L_c)
        g = float(np.linalg.norm(est.r(hi) - est.r(lo))) / max(hi - lo, 1e-12)        # 직선 길이 / 사전 틀 s
        ext = 0.5 * R / g
        how = (f"corrected points {L_c * 1e3:.1f} mm apart + half the remaining design length beyond each ({0.5 * R * 1e3:.1f} mm; "
               f"R = {R * 1e3:.1f} mm)")
        return lo - ext, hi + ext, None, how

    def _pass(self, est: SeamEstimate, k: int, direction: str, s_S: float, s_E: float, sigma_plan: Optional[dict], how: str, *,
              plate_m: Optional[float] = None, name: Optional[str] = None, leg_dir=None) -> dict:
        """패스 k (방향 'S->E'|'E->S'): 추정 직선 위 r(s_S) → r(s_E), 레이저 ⟂ 직선 (물려받은 _pass_job 1G: 판 r* − D̂/2, 루트 + D̂, 피치 job,
        진행 부호 = 툴 자세 유지), 끝마다 착점 오버스캔 ov_X = k·σ_plan,X + ½·FOV(r_p)·|sin ψ_res| + D̂·tan|피치| + 끝 판독의 가장자리 응답
        도달 거리 (end_faces.EDGE_REACH_M) + v²/2a (끝을 순항 속도로 지나게) + 첫 끝에만 v·t_settle (시작 가속 뒤 서보 안정 시간 — 실행의
        구간들에서 잰 것, 제어기 지연 사전; scan_core.settling) (path_gen.pass_overscan;
        sigma_plan None = 패스 1: k·σ 항 없음) → 재배치 (_reposition) → 기록.
        plate_m / name / leg_dir (실행 뒤 실험 3, experiments.py): 판 거리만 바꾼 같은 패스 (오버스캔도 그 거리에서 같은 식) — 그 이름으로
        기록하고 실행의 패스로 세지 않는다 (leg_dir 에 같은 규약). 그 프로파일 맞춤은 실행 추정의 심 위치 사전 없이 (트랜싯처럼 설계
        템플릿 시작): 실험이 시험하는 것이 바로 그 사전의 기하 (패스 거리에서 맞춘 레이저 면 — 다른 거리에서는 w_x 가 루트를 법선으로
        z·sin w_x·sin 피치 옮긴다) 라서, 그 사전으로 맞추면 그 거리의 그루브가 '이 심이 아님' 으로 읽힌다 (가상 팔 w_x 1.3°: D + 60 / + 119 mm
        에서 λ −22 / −126)."""
        qd = self._plan_quantities(est)
        D = float(qd["depth_m"])
        P_S, P_E = est.r(s_S), est.r(s_E)
        n_hat = self._n_hat
        # 패스 틀 = §9 단면 틀 (û = 추정 직선, n̂ ⟂ û): 직선과 판 법선은 따로 추정되어 정확히 직교가 아니다 (g1c_01 2026-10-10: 3e-5 rad →
        # 툴 회전이 강체가 아니어서 D3 가 패스 1 재배치를 거부). 레이저 평면은 직선에 직각 그대로, 툴 축만 판 법선에서 그만큼 기운다.
        u_l, n_l, _ = _section_basis(P_E - P_S, n_hat)
        line = SeamLine(c=P_S, u=u_l, n=n_l, s_S=0.0, s_E=float(np.linalg.norm(P_E - P_S)), k=k, direction=direction)
        plan0 = PassPlan(direction=direction, s_S=0.0, s_E=line.s_E, ov_S=0.0, ov_E=0.0, rule="g1")
        # a pass is a measurement move: its job at the measurement speed (the overscan's ramp / settling terms and the record)
        job_q = self._at_speed(self._at_plate(self._pass_job(line, plan0, k, D, quiet=True), plate_m, D, name), MOVE_MEASURE)
        r_p = float(job_q.standoff_m) - D
        psi = abs(float(qd["psi_res_rad"]))
        fov_term = 0.5 * self._sv().fov_m(r_p) * math.sin(psi)
        pitch_term = D * math.tan(abs(math.radians(float(job_q.pitch_deg))))
        # the ends are read at the pass's cruise speed: a leg that starts or stops at an end crosses it on its speed ramp (v²/2a — 18 mm
        # at 60 mm/s and 0.1 m/s²), where the profiles' along-track speed and pose-time offsets differ from the rest of the pass; the
        # FIRST end is crossed only after the servo has settled from the start ramp (v·t_settle, t_settle measured on the run's legs
        # with the controller-lag prior — scan_core.settling); and the plate-end reading uses the sheet positions up to the edge
        # response's reach beyond the end (end_faces.EDGE_REACH_M) — all at steady cruise (user 2026-10-11 night)
        from scan_core.end_faces import EDGE_REACH_M
        v_p = float(job_q.speed_mps)
        ramp, settle, st = self._cruise_terms(v_p)
        first, last = ("E", "S") if direction == "E->S" else ("S", "E")
        ov = {X: (0.0 if sigma_plan is None else self.k * float(sigma_plan[X])) + fov_term + pitch_term + EDGE_REACH_M + ramp
              + (settle if X == first else 0.0) for X in ("S", "E")}
        plan = dataclasses.replace(plan0, ov_S=ov["S"], ov_E=ov["E"], how_S=how, how_E=how)
        job_k = self._at_speed(self._at_plate(self._pass_job(line, plan, k, D, quiet=True), plate_m, D, name),   # 한 줄만 (아래) — 물려받은
                               MOVE_MEASURE)                                                      # 계획 줄은 같은 내용을 되풀이했다 (r3 §8)
        rot = _rot_angle_deg(self._T_now()[:3, :3], sensor_frames(job_k, self.calib).T_base_flange(0.0)[:3, :3])
        raised = float(job_k.standoff_m) - (float(self.pass_plate_m if plate_m is None else plate_m) + D)
        ovs = pass_overscan(job_k, self.calib, 0.0, top_offset_m=D, overscan_first_m=ov[first], overscan_last_m=ov[last])
        label = f"pass {k}" if name is None else name
        self._last_pass_plan = {"plate_distance_m": float(job_k.standoff_m) - D, "standoff_m": float(job_k.standoff_m), "depth_m": D,
                                "pitch_deg": float(job_k.pitch_deg), "speed_mps": float(job_k.speed_mps),
                                "overscan_S_m": float(ov["S"]), "overscan_E_m": float(ov["E"]), "raised_m": float(raised),
                                "progression_sign": int(job_k.progression_sign), "t_settle_s": float(st["t_settle_s"])}
        self._decide("pass" if name is None else "experiment",
                     f"{label} ({direction}): {how}; laser hit overscan S {ov['S'] * 1e3:.2f} mm, E {ov['E'] * 1e3:.2f} mm "
                             f"({'no k sigma term (pass 1)' if sigma_plan is None else 'k sigma_plan'} + FOV/2 sin psi_res "
                             f"{fov_term * 1e3:.2f} + D tan|pitch| {pitch_term * 1e3:.2f} + edge-response reach {EDGE_REACH_M * 1e3:.2f} "
                             f"+ speed ramp v^2/2a {ramp * 1e3:.2f}; + settling v t_settle {settle * 1e3:.2f} at the first end {first}: "
                             f"t_settle {float(st['t_settle_s']) * 1e3:.0f} ms from {st['n_legs']} leg(s), {st['source']}); "
                             f"leg {np.linalg.norm(ovs['origin_to'] - ovs['origin_from']) * 1e3:.0f} mm at "
                             f"{float(job_k.speed_mps) * 1e3:.0f} mm/s, plate distance {r_p * 1e3:.1f} mm; progression "
                             f"{int(job_k.progression_sign):+d} keeps the tool orientation (rotation {rot:.2f} deg, pitch "
                             f"{float(job_k.pitch_deg):+.1f} deg), standoff {float(job_k.standoff_m):.4f} m"
                             + (f" (raised {raised * 1e3:.1f} mm to clear the plate)" if raised > 1e-9 else ""),
                     overscan=_jsonable({k_: ovs[k_] for k_ in ("hit_offset_m", "overscan_first_m", "overscan_last_m", "s_from", "s_to",
                                                                  "hit_point_from", "hit_point_to")}), plan_quantities=_jsonable(qd),
                     progression_sign=int(job_k.progression_sign), rotation_deg=rot, plan=plan.to_dict(),
                     settling=_jsonable({k_: v_ for k_, v_ in st.items()}))
        self._reposition(ovs["T_from"], k, prefix=None if name is None else name)
        R_opt = sensor_frames(job_k, self.calib).T_base_opt(0.0)[:3, :3]
        pp = self._profile_prior(R_opt, est_dir=(P_E - P_S, psi), pitch_deg=float(job_k.pitch_deg))
        return self._record_leg(f"pass{k}" if name is None else name, ovs["T_to"], job_k, "pass", 1 if direction == "S->E" else -1,
                                speed_class=MOVE_MEASURE, bevel_x_sign=self._bevel_x_sign(est, R_opt), profile_prior=pp,
                                est=est if leg_dir is None else None, leg_dir=leg_dir)

    def _at_plate(self, job: ScanJob, plate_m: Optional[float], depth_m: float, name: Optional[str]) -> ScanJob:
        """실행 뒤 실험 3: 같은 패스 작업을 판 거리 plate_m 으로 (standoff = plate_m + 깊이; 툴이 판에서 한계를 지키게 필요하면 물려받은
        규칙 (_clear_plate) 이 더 올린다). None = 실행의 패스 그대로."""
        if plate_m is None:
            return job
        out = ScanJob.from_dict({**job_to_dict(job), "torch_lateral_m": tuple(job.torch_lateral_m), "standoff_m": float(plate_m) + depth_m,
                                 "job_id": f"{self.job0.job_id or 'job'}_{name or 'experiment'}"})
        out.plate_top_measured_m = job.plate_top_measured_m
        out.plate_plane_measured = job.plate_plane_measured
        out.floor_z_m = job.floor_z_m
        return self._clear_plate(out, depth_m)[0]

    def _reposition(self, T_start, k: int, prefix: Optional[str] = None) -> None:
        """재배치 (고정 구성, 다른 순서 없음): (1) 지금 판 거리에서 면내·재자세 직선 ('scan') 으로 패스 시작 자세의 광축 위로,
        (2) 광축을 따라 패스 시작까지 (레이저 평면 = 광학 XZ 평면이 그대로라 센서가 보는 선이 쓸려 가지 않는다 — 판 법선을 따라 내려가면
        피치 때문에 착점이 판 끝을 넘어가며 가드가 '다가가는 중 면이 사라짐' 으로 멈춘다). 구간 태그는 물려받은 규칙 (_leg_phase). 둘 다
        D3/D4. 플랫폼 도착 허용 (goal_tol) 안의 구간은 보내지 않는다. 둘 다 위치 잡기 부류 (사용자 2026-10-12: 플랫폼 한계)."""
        T_start = np.asarray(T_start, dtype=float)
        T_now = self._T_now()
        n = self._n_hat
        Tfo = self.calib.T_flange_opt
        O_d = T_start @ Tfo
        Z = O_d[:3, 2]
        dz = float((O_d[:3, 3] - (T_now @ Tfo)[:3, 3]) @ n)
        t = dz / float(Z @ n)
        T_if = np.array(T_start, dtype=float, copy=True)
        T_if[:3, 3] -= Z * t
        d_ip = float(np.linalg.norm((Z * t) - n * float((Z * t) @ n)))
        ph = self._leg_phase(dz, d_ip, orientation=self.orientation, tol_m=self._sv().pose_tol_m)
        T_a = T_now
        pre = f"pass{k}" if prefix is None else prefix
        for T_b, phase, nm in ((T_if, "scan", f"{pre}_reposition"), (T_start, ph, f"{pre}_approach")):
            if start_pose_mismatch(T_b, T_a, self.policy.goal_tol_m, self.policy.goal_tol_deg) is None:
                T_a = T_b
                continue
            self._leg(T_b, phase, nm, speed_class=MOVE_POSITION,
                      extra=f"(to the start of {'pass ' + str(k) if prefix is None else prefix})")
            T_a = T_b

    # ---- 출력 ----------------------------------------------------------------------------------------------------------
    def _finish(self, est: Optional[SeamEstimate]) -> None:
        """작업 프로세스 'outputs' (g1_outputs.write_outputs: sections.* + weld_plan.* → out_dir; 계약 (S→E) 거부면 단면만, 계획은
        verification/) → MoveIt 장면 지우기 (토치가 그루브에 들어가야 한다) → D14 도달성: 플랫폼의 check_weld_plan (MoveIt 어댑터: IK·
        카르테시안 경로·조밀 IK, 모션 없음) → 모든 패스가 확인되어 도달 (reachable True) 이면 도달성을 담아 out_dir 에, **아니면** (도달
        못함 False, 확인 못함 None, 확인 기능이 없는 플랫폼, 빠진 패스) weld_plan.* 를 verification/ 으로 옮기고 DATA_FAIL — 확인되지 않은
        계획은 납품하지 않는다 (spec §2 D14). 단, 설정이 용접 토치가 이 팔에 없다고 하면 (weld_torch.on_this_arm false — 사용자 2026-10-12)
        D14 를 하지 않고 계획을 납품하며 그 사실을 계획 요약·실행 기록·weld_plan.json·kiro/ 에 남긴다 (다른 토치로 대신하지 않는다).
        연구실·KIRO 는 이 토치가 그 팔에 있다 (2026-10-12: 용접 토치 = 우리 1차 브라켓 토치) → D14 를 한다.
        Both weld directions (user 2026-10-14; g1_outputs.DIRECTION_SETS_RULE): the worker's plan carries the two direction sets
        (plan.directions, already in out_dir/directions/<set>/). Once the primary plan is delivered, _weld_direction_sets runs the same
        D14 for each set that is not the primary plan (the scene as for the primary: 'hongik/under' stays) and saves or withdraws each;
        when the primary is not delivered, no set is (they go to verification/directions/ with it). The run's status is the primary's."""
        r = self.res
        self._check_cancel()
        run = {"status": r.status, "converged": r.converged, "reason": r.reason, "passes": _jsonable(self.leg_info),
               "notes": list(r.notes), "bevel_side_ln_bf": self._bevel_ln_bf(), "estimates": len(self.est_history),
               "S_taught_point": r.s_taught_point}
        out = self._wcall("outputs", {"est": est, "prior": self.prior, "run": run, "process": self._process_1g(),
                                      "design": self._design_1g(), "torch": self.weld_torch,
                                      "out_dir": str(self.out), "base_frame": self.job0.base_frame,
                                      "gravity_axis_base": np.asarray(self.job0.gravity_axis_base, dtype=float), "k": self.k})
        r.files.extend(out.get("files") or [])
        self._clear_cell()
        uo = self.world.under_object(grow_top=False) if getattr(self, "world", None) is not None and hasattr(self.world, "under_object") \
            else None
        if uo is not None and self.rs.set_scene_boxes([uo]):   # D14: nothing of the arm under the specimen (its top on the support
            self.cell_ids = [uo["id"]]                         # plane - the torch works above it, the specimen is the weld scene)
        plan = out.get("plan")
        if r.status == STATUS_DATA_FAIL and r.category == "sensing":     # an end not identified (user 2026-10-11): sections only
            m = _gw.models()
            moved = m.retract_weld_plan(self.out)
            r.files = [f for f in r.files if os.path.basename(f) not in ("weld_plan.json", "weld_plan.npz")] + list(moved)
            self._direction_sets_withdrawn(plan, moved, "no weld paths: an end is not identified by the measurements")
            r.weld_plan = None
            self._decide("outputs", f"outputs: {sorted(os.listdir(self.out))}; no weld paths delivered (an end is not identified by "
                                    f"the measurements) - the plan from the estimate is kept in verification/ only")
            return
        if plan is None:
            r.status, r.category = STATUS_DATA_FAIL, "waypoint"
            r.reason = f"weld plan refused (D14, S->E contract): {out.get('plan_error')} - plan kept in verification/ (scan: {r.reason})"
            self._decide("outputs", r.reason)
            return
        m = _gw.models()
        d14, d14_note = _wt.d14_on_this_arm(self.weld_torch)
        verdicts, not_ok = self._weld_plan_d14(est, plan, d14, d14_note)
        if not d14:                     # the config says the weld torch is not on this arm: no D14, the plan is delivered with the note
            r.notes.append(d14_note)
            self._decide("outputs", d14_note, reachability=plan.reachability)
        m.save_weld_plan(plan, self.out)                                    # 도달성을 담아 다시 (원자적)
        r.weld_plan = plan
        if not_ok:
            moved = m.retract_weld_plan(self.out)
            r.files = [f for f in r.files if os.path.basename(f) not in ("weld_plan.json", "weld_plan.npz")] + list(moved)
            r.status, r.category = STATUS_DATA_FAIL, "waypoint"
            r.reason = ("weld plan not verified (D14): "
                        + "; ".join(f"pass {k_}: {'not reachable' if f.get('reachable') is False else 'reachability not checked'} "
                                    f"({f.get('reason')})" for k_, f in not_ok)
                        + f" - plan kept in verification/ (scan: {r.reason})")
            self._direction_sets_withdrawn(plan, moved, "the primary plan (the job's weld direction) is not delivered (D14) - no "
                                                        "direction set is checked or delivered without it")
        if not not_ok:
            self._weld_direction_sets(est, plan, d14, d14_note)
            self._kiro_outputs(plan)
        self._decide("outputs", f"outputs: {sorted(os.listdir(self.out))}; weld plan {len(plan.passes)} passes "
                                + ("delivered, reachability (D14) not checked on this arm" if not d14 else
                                   f"{'reachable (checked by ' + str(self.rs.adapter) + ')' if not not_ok else 'NOT verified -> verification/'}"),
                     timing=out.get("timing"), plan_summary=plan.summary_lines(), contract=out.get("contract"),
                     reachability=_jsonable([{"k": k_, "reachable": f.get("reachable"), "reason": f.get("reason")} for k_, f in verdicts]),
                     direction_sets=m.direction_sets_summary(plan) if getattr(plan, "directions", None) else None)

    def _weld_plan_d14(self, est, plan, d14: bool, d14_note: str, label: str = "") -> tuple:
        """D14 on one weld plan (the primary, or a direction set's - label names it in the decisions; '' = the primary, its decision
        texts as they always were): with the weld torch on this arm, the platform's check_weld_plan through the torch-orientation
        search (weld_orient.choose_orientation: per pass the closest feasible orientation to the WPS on its delivered line, one TCP
        roll; the geometric search in the worker); without it (d14 False) no check - the plan carries the note. Fills plan.clearance,
        plan.feasibility, plan.reachability, plan.orientation, plan.torch_roll (in place). → (verdicts [(k, row)], not_ok [(k, row)
        whose reachable is not True])."""
        pre = f"{label}: " if label else ""
        if not d14:                     # the config says the weld torch is not on this arm: no D14, the plan is delivered with the note
            plan.feasibility = []
            plan.reachability = {"checked": False, "note": d14_note}
            plan.torch_roll = dict(plan.torch_roll or {}, note="no D14 on this arm: the plan's NECK_RULE roll, unchecked")
            return [], []
        check = getattr(self.rs, "check_weld_plan", None)
        body_fn = getattr(self.rs, "weld_tool_body", None)
        if callable(check):         # D14 + the torch orientation per pass (closest feasible to the WPS) and one TCP roll
            req = self._clearance_requirement(est, plan)
            plan.clearance = req.to_dict()
            try:
                body = body_fn() if callable(body_fn) else None
                if body is None:
                    raise ValueError(f"the platform ({self.rs.adapter}) gives no tool collision geometry")
                from scan_core.weld_clearance import tool_roles     # the parts by role: gas nozzle + contact tip / the rest
                body = tool_roles(body)
            except Exception as e:  # noqa: BLE001 - no tool geometry: the workpiece cannot be checked (never silently)
                body = None
                plan.feasibility = [{"k": wp.k, "kind": wp.kind, "reachable": None,
                                     "reason": f"weld-pose clearance not checked: {type(e).__name__}: {e}"} for wp in plan.passes]
            if body is not None:
                self._decide("outputs", f"{pre}weld plan required clearance tool <-> specimen: {req.describe()}",
                             clearance_requirement=plan.clearance)
                wtorch = _wt.plan_weld_torch(plan, self.weld_torch)
                reach_fn = getattr(self.rs, "weld_reach_model", None)
                reach = reach_fn() if callable(reach_fn) else None             # the arm's elbow shell (exact necessary condition)
                # D14 on the DELIVERED lines (scan_core.weld_line): samples at the platform's Cartesian step - the poses MoveIt
                # executes and kiro/pass_<k>.yaml carries
                step_fn = getattr(self.rs, "weld_line_step_m", None)
                step_m, step_src = _wl.line_step_m(plan, step_fn() if callable(step_fn) else None)
                # the worker's copy of the plan for the geometric search (its direction sets are not needed there)
                wplan = dataclasses.replace(plan, directions={}) if getattr(plan, "directions", None) else plan
                self._wcall("orient_open", {"plan": wplan, "body": body, "torch": wtorch, "required_mm": req.required_mm,
                                            "reach": reach, "step_m": step_m})

                def remote(wp_, roll_, bound_=None):       # the geometric search runs in the worker process (numpy, no GIL here):
                    class _It:                             # each (pass, roll) in its own child there (g1_worker orient_ask / take)
                        rid = None

                        def __iter__(it):
                            return it

                        def ask(it):                       # start the next candidate there (no wait; choose_orientation)
                            if it.rid is None:
                                it.rid = self._wcall("orient_ask", {"k": wp_.k, "roll": roll_, "max_deviation_deg": bound_})["id"]

                        def __next__(it):
                            it.ask()
                            rid, it.rid = it.rid, None
                            return self._wcall("orient_take", {"k": wp_.k, "roll": roll_, "id": rid})
                    return _It()
                rec = _wo.choose_orientation(plan, check, body=body, required=req, torch=wtorch, candidates=remote,
                                             log=lambda m_: self._p(pre + m_), reach=reach, step_m=step_m)
                self._wcall("orient_close", {})          # (the children still computing an unwanted candidate end)
                self._decide("outputs", f"{pre}weld plan torch roll: {plan.torch_roll['reason']}", torch_roll=plan.torch_roll)
                for o in rec["passes"]:
                    nm, ch = o["nominal"], o["chosen"]
                    self._decide("outputs", (f"{pre}weld pass {o['k']} torch: nominal travel {nm['travel_deg']:+.0f} work "
                                             f"{nm['work_deg']:+.0f} deg -> " + (
                                                 f"travel {ch['travel_deg']:+.0f} work {ch['work_deg']:+.0f} deg (deviation "
                                                 f"{o['deviation_deg']:.1f} deg), clearance {o['clearance_mm']:.2f} mm "
                                                 f"(required {o['required_mm']:.2f})"
                                                 + "".join(f", gas nozzle + contact tip {v_['clearance_mm']:.2f} mm (required "
                                                           f"{v_['required_mm']:.2f})" for k_, v_ in
                                                           (o.get('clearance_by_role') or {}).items() if k_ == "process")
                                                 + f", limiting {(o.get('limiting_part') or {}).get('role', '-')} parts"
                                                 + (f" ({o['limiting_part']['link']})" if (o.get('limiting_part') or {}).get('link')
                                                    else "")
                                                 + f", roll {o['torch_roll_deg']:g} deg, on the delivered line ("
                                                 f"{(o.get('worst_sample') or {}).get('n_samples', '?')} samples every <= "
                                                 f"{step_m * 1e3:g} mm); why: {(o.get('deviation_why') or {}).get('text', '-')}"
                                                 if ch is not None else f"no feasible orientation: {o['why']}")),
                                 torch_orientation=o)
        else:
            plan.feasibility = [{"k": wp.k, "kind": wp.kind, "reachable": None,
                                 "reason": f"the platform ({self.rs.adapter}) has no reachability check"} for wp in plan.passes]
        plan.reachability = {"checked": True, "weld_torch": self.weld_torch.tcp_frame,
                             "note": "weld-plan reachability (D14) checked on this arm with the weld torch"}
        by_k = {f.get("k"): f for f in plan.feasibility}
        verdicts = [(wp.k, by_k.get(wp.k, {"reachable": None, "reason": "no reachability answer for this pass"})) for wp in plan.passes]
        return verdicts, [(k_, f) for k_, f in verdicts if f.get("reachable") is not True]

    def _weld_direction_sets(self, est, plan, d14: bool, d14_note: str) -> None:
        """Both weld directions (user 2026-10-14; g1_outputs.DIRECTION_SETS_RULE) once the primary plan is delivered: the set that IS
        the primary plan is saved again from that same object (its D14 result included - no second plan, no second D14); every other
        set gets the same D14 (_weld_plan_d14 - one per set) and is saved with its result in out_dir/directions/<set>/, or - a pass
        not reachable / not checked - saved to verification/directions/<set>/ instead (not delivered) with the reason in the run's
        notes and decisions. A set the worker could not build stays in verification/ with its reason. The run's status is the
        primary plan's alone (unchanged here)."""
        sets = getattr(plan, "directions", None) or {}
        if not sets:
            return
        r = self.res
        m = _gw.models()
        for nm, ds in sets.items():
            label = f"weld plan direction set {nm} (every pass {ds.direction})"
            p = m.direction_plan(plan, ds)
            if ds.same_as_primary:
                m.save_direction_set(plan, ds, self.out)
                self._decide("outputs", f"{label}: the primary plan itself - saved from the same plan with its D14 result "
                                        f"(no second D14) in {ds.folder}/", direction_set=ds.to_dict())
            elif p is None or not ds.delivered:
                r.notes.append(f"{label} not delivered: {ds.reason} - kept in {ds.folder}/")
                self._decide("outputs", f"{label} not delivered: {ds.reason} - kept in {ds.folder}/", direction_set=ds.to_dict())
            else:
                _verdicts, bad = self._weld_plan_d14(est, p, d14, d14_note, label=label)
                if bad:
                    ds.delivered = False
                    ds.reason = ("weld plan not verified (D14): "
                                 + "; ".join(f"pass {k_}: "
                                             + ("not reachable" if f.get("reachable") is False else "reachability not checked")
                                             + f" ({f.get('reason')})" for k_, f in bad))
                m.save_direction_set(plan, ds, self.out)              # with its D14 result: delivered, or verification/ (moved)
                if bad:
                    r.notes.append(f"{label} not delivered: {ds.reason} - kept in {ds.folder}/ (the run's status is the primary "
                                   "plan's)")
                self._decide("outputs", f"{label}: " + (f"NOT delivered - {ds.reason} - kept in {ds.folder}/" if bad else
                                                        ("delivered, reachability (D14) not checked on this arm" if not d14 else
                                                         f"reachable (checked by {self.rs.adapter})") + f" in {ds.folder}/"),
                             direction_set=ds.to_dict(), plan_summary=p.summary_lines())
            r.files.extend(ds.files)

    def _direction_sets_withdrawn(self, plan, moved: list, why: str) -> None:
        """The primary plan is not delivered: neither is any direction set (retract_weld_plan moved them to verification/directions/
        with it) - each set's record says so (run notes, decisions)."""
        sets = (getattr(plan, "directions", None) or {}) if plan is not None else {}
        if not sets:
            return
        m = _gw.models()
        for nm, ds in sets.items():
            ds.delivered, ds.reason = False, str(why)
            if ds.same_as_primary:                     # the primary plan as it is now (its D14 result, if any) - as verification/
                m.save_direction_set(plan, ds, self.out)
            else:
                ds.files = [f for f in moved if f"/{ds.folder}/" in f.replace(os.sep, "/")]
            self.res.files.extend(f for f in ds.files if f not in self.res.files)
        self.res.notes.append(f"weld plan direction sets {', '.join(sets)} not delivered: {why} - kept in verification/directions/")
        self._decide("outputs", f"weld plan direction sets {', '.join(sets)} not delivered: {why} - kept in verification/directions/",
                     direction_sets=m.direction_sets_summary(plan))

    def _clearance_requirement(self, est, plan) -> "_wo.ClearanceRequirement":
        """The required tool <-> specimen clearance at the weld poses (user 2026-10-13): k (platform judgement.k) x the root-sum-square
        of the stated uncertainties - the estimate's root line (the aim: lateral and normal sigma, largest over the plan's sections),
        its plate top, the calibration's posture-dependent pose error (he_sigma_m); the weld torch TCP calibration and the EOAT
        geometry have no documented sigma - listed as missing, not invented."""
        src = []
        aim = getattr(plan, "aim_sigma_m", None) or {}
        A = [np.asarray(a, dtype=float) for a in aim.values() if a is not None and np.size(a)]
        if A and any(np.isfinite(a).all(axis=1).any() for a in A):
            v = max(float(np.nanmax(np.hypot(a[:, 0], a[:, 1]))) for a in A)
            src.append(("measured root line (the aim point)", v, "the plan's aim-point sigma (the estimate's root line at each "
                                                                 "pass's sections, lateral and normal), largest over the passes"))
        try:
            if A:
                raise StopIteration
            eS, eE = est.ends["S"], est.ends["E"]
            L = float(plan.length_mm) if plan.length_mm and math.isfinite(float(plan.length_mm)) else 0.0
            s_pl = np.array([st.s_mm for st in plan.stations], dtype=float)
            s_est = float(eS.s_m) + (float(eE.s_m) - float(eS.s_m)) * (s_pl / L if L > 0 else np.zeros_like(s_pl))
            sw, sn = est.r_sigma(s_est)
            src.append(("measured root line (the aim point)", float(np.max(np.hypot(sw, sn))),
                        "the estimate's root-line covariance (lateral and normal), largest over the plan's sections"))
        except StopIteration:
            pass
        except Exception as e:  # noqa: BLE001
            src.append(("measured root line (the aim point)", None, f"not available ({type(e).__name__}: {e})"))
        try:
            src.append(("measured plate top", float(self._top_point(est)[1]), "the estimate's plate-top sigma (the S1 world's)"))
        except Exception as e:  # noqa: BLE001
            src.append(("measured plate top", None, f"not available ({e})"))
        he = getattr(self.prior, "he_sigma_m", None) if self.prior is not None else None
        src.append(("robot posture-dependent pose error", None if he is None or not math.isfinite(float(he)) else float(he),
                    "calibration g1.he_sigma_m: the pose-dependent hand-eye residual between postures (P7 report r3) - the robot "
                    "posture at the weld differs from the scanning one"))
        src.append(("weld torch TCP calibration", None, "not documented: the TCP is the CAD contact-tip end (nominal); KIRO "
                                                        "re-calibrates on site, no sigma given"))
        src.append(("EOAT geometry (CAD vs the built tool)", None, "not documented: the collision geometry is the CAD with the "
                                                                   "5.0 mm dust plate, not measured"))
        return _wo.clearance_requirement(self.k, src)

    def _kiro_outputs(self, plan) -> None:
        """납품하는 계획 → out_dir/kiro/ (KIRO 의 pass_<k>.yaml + 공유용 pass_<k>_info.yaml + README; scan_core.kiro_plan). World ← base: 플랫폼 TF
        (frame_transform, 있으면) 가 먼저, 아니면 설정 world_from_base. 못 쓰면 이유를 메모·결정 기록에 (계획 납품은 그대로 — 액션 결과에는
        KIRO 형식 칸이 없다: precision_sensing.npz meta 'kiro_outputs' 가 가리킨다)."""
        r = self.res
        cfg = self.kiro_cfg
        if cfg is None:
            r.kiro_outputs = {"written": False, "reason": "no kiro_output section in the platform config"}
            return
        tf = getattr(self.rs, "frame_transform", None)
        base = str(self.job0.base_frame)                                 # the plan's frame
        try:
            # the robot's base for World <- base = MoveIt's planning frame read now (not a configured name; KIRO answer 2026-10-12)
            pf_fn = getattr(self.rs, "planning_frame", None)
            rt, rt_src = pf_fn() if callable(pf_fn) else (None, f"the platform ({self.rs.adapter}) has no MoveIt planning frame")
            wnotes = []
            if rt is None:
                wnotes.append(f"planning frame not read ({rt_src}): the plan's base frame {base} is used")
                rt, rt_src = base, "the plan's base frame"
            T_rp = None                                                  # runtime base <- plan base
            if rt != base:
                if not callable(tf):
                    raise ValueError(f"the planning frame {rt} is not the plan's base frame {base} and the platform has no TF")
                M = np.asarray(tf(rt, base), dtype=float)
                T_rp = _kp.RigidTransform(R=M[:3, :3], t=M[:3, 3].copy(), source=f"TF {rt} <- {base}", base_frame=base)
            wt, n2 = _kp.resolve_world_from_base(cfg, rt, (lambda a, b: tf(a, b)) if callable(tf) else None)
            wnotes += n2
            if wt is None:
                raise ValueError("; ".join(wnotes))
            wnotes.append(f"World <- base: base = {rt} ({rt_src}); {wt.source}")
            if T_rp is not None:
                wt = _kp.compose(wt, T_rp, source=f"{wt.source} x TF {rt} <- {base}")
            ko = _kp.write_kiro_outputs(plan, self.out, cfg, world_from_base=wt, notes=wnotes)
            ko["runtime_base_frame"] = {"name": rt, "source": rt_src, "plan_base_frame": base}
        except Exception as e:  # noqa: BLE001 - the KIRO-format files never break the delivered plan (reason recorded)
            r.kiro_outputs = {"written": False, "reason": f"{type(e).__name__}: {e}"}
            r.notes.append(f"KIRO-format outputs not written: {e}")
            self._decide("outputs", f"KIRO-format outputs (kiro/pass_<k>.yaml) not written: {e}", kiro_outputs=r.kiro_outputs)
            return
        r.files.extend(ko["files"])
        for d_ in (ko.get("directions") or {}).values():             # both weld directions: each delivered set's kiro/ files
            r.files.extend(d_.get("files") or [])
        r.kiro_outputs = {"written": True, **ko}
        self._decide("outputs", f"KIRO-format outputs: {len(ko['passes'])} pass file(s) in {ko['dir']} (frame {ko['frame_id']}, "
                                f"TCP {ko['tcp_frame']} = the torch tip end centre = the bead aim point - (stick-out {ko['stick_out_mm']:g} mm "
                                f"from the nozzle end + nozzle recess) x z_TCP = aim - {ko.get('tcp_to_aim_point_mm', float('nan')):g} mm "
                                f"x z_TCP; TCP roll {float(plan.torch_roll_deg):g} deg; World <- {base}: {wt.source}); info files list "
                                "assumed / derived values (not commands)"
                                + "".join(f"; direction set {nm_} ({d_['direction']}): " + (
                                    f"{len(d_.get('files') or [])} file(s) in {d_.get('dir')}" if d_.get("kiro_written") else
                                    f"KIRO files not written ({d_.get('kiro_reason')})" if d_.get("delivered") else
                                    f"not delivered ({d_.get('reason')})") for nm_, d_ in (ko.get("directions") or {}).items()),
                     kiro_outputs=_jsonable({k_: v for k_, v in ko.items() if k_ != "passes"}),
                     kiro_passes=_jsonable(ko["passes"]))

    def _bevel_ln_bf(self) -> Optional[float]:
        for li in self.leg_info:
            v = li.get("bevel_ln_bf")
            if v is not None:
                return v
        return None

    def _write_outputs(self, *a, **kw) -> None:
        """실행 기록은 verification/run_log.json 에만 (job_refined·calib_refined·specimen_report 없음)."""
        r = self.res
        try:
            doc = {"pipeline": "g1core", "config": self.at.to_dict(), "job_rough": job_to_dict(self.job0),
                   "measure_speed": {**self.measure_speed_record(),
                                     "commanded_mps": self._scan_speed() if self.cfg is not None else None},
                   # run-time job overrides (scan_l0.job_overrides): the job file, every override {path, old, new, source, old_from},
                   # the effective job document saved before the run
                   "job_file": _jo.job_file_of(self.doc), "job_overrides": self.job_overrides(),
                   "job_effective": self.job_effective_path,
                   "job_coarse": job_to_dict(self.job_c) if self.job_c is not None else None, "calib_effective": self.calib.to_dict(),
                   "prior": self.prior.to_dict() if self.prior is not None else None,
                   "estimates": self.est_history, "legs": self.leg_info, "world": self.world_history,
                   "joint_state_source": self.joint_state_source,
                   "experiments": ({"plan": self.experiments.to_dict(), "result": self.experiments_result}
                                   if self.experiments is not None else None),
                   "worker": {"respawns": getattr(self.worker, "respawns", 0), "deaths": getattr(self.worker, "deaths", [])},
                   **r.to_dict()}
            _gw.atomic_write_text(self.vdir / "run_log.json", json.dumps(jsonable(doc), indent=1, ensure_ascii=False, default=str))
            r.files.append(str(self.vdir / "run_log.json"))
        except Exception as e:  # noqa: BLE001
            r.notes.append(f"run log save failed: {type(e).__name__}: {e}")

    def _deliver(self, est: SeamEstimate) -> None:
        """결과 필드: extent (S→E 끝·σ·양쪽 관측·길이 — S = 먼저 간 끝), job_final (액션 결과 seam_start/end 용, out_dir 에 쓰지 않는다)."""
        r = self.res
        ext = {}
        for X in ("S", "E"):
            e = est.ends[X]
            ext[X] = {"point": np.asarray(e.point_m, dtype=float).tolist(), "sigma_m": float(e.sigma_m), "observed": bool(e.two_sided),
                      "n_legs": int(e.n_legs)}
        ext["length_mm"] = float(np.linalg.norm(np.asarray(ext["E"]["point"]) - np.asarray(ext["S"]["point"]))) * 1e3
        r.extent = ext
        try:
            r.job_final = ScanJob.from_dict({**job_to_dict(self.job0), "torch_lateral_m": tuple(self.job0.torch_lateral_m),
                                             "start": list(ext["S"]["point"]), "end": list(ext["E"]["point"]),
                                             "surface_normal": [float(v) for v in self._n_hat]})
        except (JobError, ValueError) as e:
            r.notes.append(f"job_final not built: {e}")

    # ---- run -----------------------------------------------------------------------------------------------------------
    def run(self) -> AutoTeachResult:
        r = self.res
        est: Optional[SeamEstimate] = None
        try:
            self._set_phase("setup")
            self._gates()
            self._set_phase("acquire")
            if not self.dry_run:
                self._start_worker()                              # 데우기 (획득 이동과 나란히)
            self._acquire()                                       # 시작 자세 → (D2) 먼저 갈 점의 드웰 자세 (관절 이동 + 하강)
            if self.dry_run:
                r.status, r.reason = STATUS_DRY_RUN, "dry run: gates + acquisition plan (no motion, no sensor, no worker)"
                self._p(r.reason)
                return r
            self._set_phase("dwell")
            site_S = self._dwell_site("S", est=None)
            self._set_phase("transit")
            self._transit(site_S)
            est = self._estimate()
            self._apply_estimate_to_world(est)
            self._set_phase("dwell")
            self._dwell_site("E", est=est)                    # D11 은 자리마다 첫 grab 에서만 (spec §4.5) — 융합 추정에는 따로 시험이 없다
            est = self._estimate()
            self._apply_estimate_to_world(est)
            # 패스 1 은 보정된 E (로봇이 있는 곳) 에서 보정된 S 쪽으로: 보정된 두 지점 사이 + 남은 거리의 절반씩 (사용자의 틀), 발자국 오버스캔만.
            # 패스 2 부터: 추정 양끝 (사후) + k·σ_끝 + 발자국 — 시편 전체, 방향 교대
            at_hi = float(self.sites["E"]["s_m"]) >= float(self.sites["S"]["s_m"])
            direction = "E->S" if at_hi else "S->E"
            while True:
                self.n_pass += 1
                self._set_phase("pass")
                if self.n_pass == 1:
                    self._pass(est, 1, direction, *self._first_pass_targets(est))
                else:
                    q = self._plan_quantities(est)
                    self._pass(est, self.n_pass, direction, q["e_S_m"], q["e_E_m"], {"S": q["sigma_plan_S_m"], "E": q["sigma_plan_E_m"]},
                               "estimated ends r(e_S) .. r(e_E) + k sigma_end (the whole specimen past both ends)")
                est = self._estimate()
                self._apply_estimate_to_world(est)
                if est.done:                                                            # D12 (sigma <= precision + model fit + hypotheses)
                    r.converged = "true"
                    r.reason = (f"done after {self.n_pass} pass(es): every sigma <= precision"
                                + (", every segment fits the fused model and every hypothesis is resolved"
                                   if self._model_checks(est) else ""))
                    status = STATUS_PASS
                    break
                if self.n_pass >= self.max_passes:                                      # D13 → PASS_UNRESOLVED (r3 §8-6; 출력은 같이 쓴다)
                    mc = self._model_checks(est)
                    blocking = list(mc.get("blocking") or [])
                    misfit = [b for b in blocking if b.get("kind") == "misfit"]
                    r.converged = "false"
                    r.converged_detail = {"unresolved": list(est.unresolved or []), "ratios": _jsonable(est.ratios),
                                          "model_checks": _jsonable({"blocking": blocking, "summary": mc.get("summary")})}
                    if misfit:                                                          # §4.9: a done-test quantity rests on a misfit
                        raise _Abort(STATUS_DATA_FAIL, f"MODEL_MISFIT after the pass budget {self.max_passes}: "
                                     + "; ".join(f"{b['segment']}: {b['why']} (affects {', '.join(b['quantities'])})"
                                                 for b in misfit[:4])
                                     + (f"; +{len(misfit) - 4} more" if len(misfit) > 4 else ""), category="sensing")
                    # identification decides (user 2026-10-11): an end that no measurement identified (no two-sided reading that the
                    # decisions keep — only one-sided bounds) has no measured position — no weld paths, SENSING_FAILED
                    unid = [X for X in ("S", "E") if not bool(getattr((est.ends or {}).get(X), "two_sided", False))]
                    if unid:
                        off = list((mc.get("summary") or {}).get("legs_not_this_seam") or [])
                        npass = sum(1 for li in self.leg_info if li.get("kind") == "pass")
                        why = (f"every pass ({', '.join(off)}) does not show this seam" if off and len(off) >= npass else
                               "no pass crossed " + ("it" if len(unid) == 1 else "them") + " (only one-sided bounds)"
                               + (f"; passes that do not show this seam: {', '.join(off)}" if off else ""))
                        r.status, r.category = STATUS_DATA_FAIL, "sensing"
                        r.reason = (f"end(s) {', '.join(unid)} not identified by the measurements after the pass budget "
                                    f"{self.max_passes}: {why} - no weld paths (sections of what was measured are written)")
                        status = STATUS_DATA_FAIL
                        break
                    mx = max([float(v) for v in (est.ratios or {}).values() if v is not None and math.isfinite(float(v))] or [float("nan")])
                    hyp = [f"{h} (2 ln odds {float(d_.get('two_ln_odds', float('nan'))):.1f} <= k^2 {self.k ** 2:g})"
                           for h, d_ in (mc.get("hypotheses") or {}).items() if not d_.get("resolved", True)]
                    unres = [b for b in blocking if b.get("kind") == "unresolved" and not b["segment"].startswith("hypothesis ")]
                    off = list((mc.get("summary") or {}).get("legs_not_this_seam") or [])
                    r.reason = (f"pass budget {self.max_passes} used; unresolved: {list(est.unresolved or [])} (max sigma/precision {mx:.2f})"
                                + (f"; unresolved hypotheses: {', '.join(hyp)}" if hyp else "")
                                + (f"; legs that do not show this seam (down-weighted, no measurement): {', '.join(off)}" if off else "")
                                + (f"; not established by the model checks: " + "; ".join(f"{b['segment']}: {b['why']}"
                                                                                         for b in unres[:3])
                                   + (f"; +{len(unres) - 3} more" if len(unres) > 3 else "") if unres else ""))
                    status = STATUS_PASS_UNRESOLVED
                    break
                direction = "S->E" if direction == "E->S" else "E->S"
            r.status = status
            if status == STATUS_DATA_FAIL:
                r.category = "sensing"
            warn = list(((est.plate or {}).get("warnings") or []))
            if warn:                                            # (b) measured vs design: reported, the status is unchanged
                r.reason += "; WARNING (measured vs design, report only): " + "; ".join(warn)
                r.notes.extend(f"WARNING (measured vs design): {w}" for w in warn)
            self._deliver(est)
            self._set_phase("outputs")
            self._finish(est)
            self._p(f"{r.status}: {r.reason}")
            self._experiments(est)                   # 작업 파일 experiments: 블록 — 같은 목표 안 (취소 = 정지), 실행의 결과는 그대로
            return r
        except _Abort as e:
            cancelled = self._cancelled and e.status in (STATUS_ABORTED, STATUS_ERROR)   # 취소 (D7): 그 이동의 끝도 취소의 결과
            status = STATUS_ABORTED if cancelled else e.status
            # the reason names the cause: a cancelled run ends with D7 - D7's own abort as it is, any other abort (the stopped motion,
            # the interrupted job) as the consequence after it. The status does not tell them apart: a motion the cancel stopped ends
            # ABORTED like D7 itself (this package has no separate stop status). The cause: an action cancel, or the hosting server
            # shutting down (_external_stop)
            cause = (self._external_stop() or "cancel requested by the operator") if cancelled else ""
            reason = (_CancelAbort(e.where, cause).reason if isinstance(e, _CancelAbort) else f"{cause} (D7): {e.reason}") \
                if cancelled else e.reason
            self._abort_path(status, reason, safety=e.safety, category=None if cancelled else e.category)
            return r
        except Exception as e:  # noqa: BLE001 - after a cancel / server shutdown: ABORTED (_unexpected_failure)
            st_, why_, cat_ = self._unexpected_failure(e, " (D7)")
            self._abort_path(st_, why_, category=cat_)
            r.notes.append(traceback.format_exc(limit=8))
            return r
        except BaseException as e:  # noqa: BLE001
            self._abort_path(STATUS_ABORTED, f"interrupted: {type(e).__name__}")
            raise
        finally:
            r.elapsed_s = self._clock.monotonic() - self.t0
            self._clear_cell()
            if self.worker is not None:
                try:
                    self.worker.close()
                except Exception as e:  # noqa: BLE001 - clean-up: keep going, but say it
                    r.notes.append(f"compute worker close failed: {type(e).__name__}: {e}")
            try:
                if r.estop_record is None:
                    r.estop_record = self._latch_record_of_this_run()
            except Exception as e:  # noqa: BLE001 - clean-up: keep going, but say it
                r.notes.append(f"stop record of this run not read: {type(e).__name__}: {e}")
            if self.sess is not None and (self.owned_session or self.release_sensor):
                try:
                    r.sensor_restore = self.sess.restore_sensor()
                except Exception as e:  # noqa: BLE001
                    r.notes.append(f"sensor restore failed: {e}")
            self._write_outputs()
            outs = {nm: str(self.out / nm) if (self.out / nm).exists() else None for nm in ("sections.npz", "weld_plan.npz")}
            self._event("outputs", out_dir=str(self.out), status=r.status, base_frame=self.job0.base_frame,
                        weld_plan=outs["weld_plan.npz"], cross_sections=outs["sections.npz"])
            if self.sess is not None and self.release_sensor and self.sess.sensor is not None:
                try:
                    self.sess.sensor.close()
                except Exception as e:  # noqa: BLE001 - clean-up: keep going, but say it
                    r.notes.append(f"sensor close failed: {type(e).__name__}: {e}")
                self.sess.sensor = None
                self.sess.sensor_ready = False
            fds, self._cancel_r, self._cancel_w = (self._cancel_r, self._cancel_w), -1, -1
            for fd in fds:
                try:
                    os.close(fd)
                except OSError as e:
                    r.notes.append(f"cancel pipe close failed: {e}")
            # 끝 줄: 상태와 사유는 바로 앞 줄 (PASS 줄 / 중단 경로의 상태 줄) 과 결과에 있다 — 되풀이하지 않고 수만 (r3 §8)
            self._p(f"auto-teach {r.status}: {len(r.passes)} pass(es), {r.n_move_l} moves, {r.elapsed_s:.1f} s"
                    + (f" (category {r.category})" if r.category else ""))
