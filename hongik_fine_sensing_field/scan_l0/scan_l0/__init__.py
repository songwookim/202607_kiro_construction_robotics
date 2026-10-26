"""scan_l0 — 순수 Python (ROS 없음) 스캔 오케스트레이션: RB20-1900ES-U + VC nano 3D-Z 그루브 자동 교시.

의존: scan_core (ScanJob/Calib/path_gen/reconstruct/quality/trajectory), vc3dz_driver (VC3DZClient, INIT_STANDARD; import 만),
      motion_types (공용 이동 결과·설정 타입).
로봇 플랫폼은 MoveIt 어댑터 (hongik_scan_ros.moveit_adapter) 를 ROS 서버가 만들어 ScanSession 으로 넘긴다 (모든 이동은 MoveIt).
시험은 같은 포트의 가상 팔 (tests/sim_arm.py).

모듈
  config        설정 YAML(hongik_scan_ros/config/*.yaml, ros__parameters) → Settings, 캘리브 YAML → Calib, 작업(job) YAML 입출력
  ports         플랫폼 포트 (PoseSource / MotionBackend / StopLatch / PlatformGates / RobotPlatform)
  motion_types  플랫폼 무관 이동 설정·결과·오류 타입
  session       SensorSession (녹화 스레드), ScanSession (로봇 플랫폼 + 센서)
  planning      build_transit_plan 래핑, 로컬 이동 계획 (포락선 검사)
  executor      구간 실행·센서·원자료 저장·재구성 도우미
  safety        포락선 여유 정책 (safety.clearance_m + 정지 거리 → 포락선 한계·도착 허용)
  cell_world    세계 모형 하나 (포락선 평면 + MoveIt 충돌 물체)
  teach_1g      1G core 오케스트레이터 (Teach1G) + g1_worker (계산 작업 프로세스)
  acquisition   범용 시작의 순수 기하 (착점 → 법선 추정·바닥·충돌 장면·3G 지지면 probe)
  autoteach     자동 교시 오케스트레이터 (아무 자세 → 획득 → 드웰·프로브·보정 → 지그재그 패스 → job_refined.yaml, 다층 용접 경로
                weld_plan.json/.npz), refine (오프라인 재현)
"""
from .config import DEFAULT_CALIB, DEFAULT_CONFIG, Settings, job_from_settings, load_calib, load_job, load_settings, save_job  # noqa: F401
from .planning import make_plan  # noqa: F401
from .executor import ScanRunResult  # noqa: F401
from .autoteach import AutoTeach, AutoTeachResult, run_auto_teach, run_refine_offline  # noqa: F401

__version__ = "0.3.0"
