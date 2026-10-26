# RELAY_2026-10-14b 적용 결과 — kiro-1g-field

이후 `RELAY_2026-10-14c_tcp_motion.md`의 증분 패치도 적용했습니다.
현재 획득 공구 자세와 방문 순서는 TCP 회전각, 토치 팁 직선거리 순서로 선택하고,
그 자세의 팔 형태는 계획한 관절 이동 시간으로 선택합니다.
TCP 변경 기록은 [HONGIK_TCP_MOTION_UPDATE.md](HONGIK_TCP_MOTION_UPDATE.md)에 있습니다.
2026-10-15c의 시작 자세 허용값 변경과 센서 원본 복원 기록은
[HONGIK_START_TOLERANCE_UPDATE.md](HONGIK_START_TOLERANCE_UPDATE.md)에 있습니다.
2026-10-15d 후속 스캔 패치와 방향 선택은 [HONGIK_POST_WELD_UPDATE.md](HONGIK_POST_WELD_UPDATE.md)에 기록했습니다.
추가 패치 `postweld_extra_vs_855838e`의 기본 센싱 자세·최소 회전 옵션과 첫 요청 수정은
[HONGIK_POST_WELD_EXTRA_UPDATE.md](HONGIK_POST_WELD_EXTRA_UPDATE.md)에 기록했습니다.

`/home/irs/ros2_ws/hongik_to_field_2026-10-14_kiro-1g-field/RELAY_2026-10-14b_kiro-1g-field.md`를 적용했습니다.
SHA256SUMS 세 파일을 확인했고, 배포 ec00145 tar.gz를 검증해 깨끗한 소스에 통합 패치 43개 파일을 적용했습니다.
기존 현장 autoteach.py 변경분은 비교 자료로 남겼습니다.

## 소스와 빌드

- 갱신 소스: `src/construct_robot_ros2/hongik_fine_sensing_field/`.
- 별도 overlay: `~/hongik_field_ws/install/` (6개 패키지 빌드 성공).
- 기존 workspace의 `build/`, `install/`, `log/`와 원래 배포 소스는 수정하지 않았습니다.
- 갱신 소스 루트의 `COLCON_IGNORE`는 일반 workspace 탐색에서 동일 이름 패키지가 중복되는 것을 방지합니다.
  재빌드 도구는 6개 패키지 디렉터리를 명시해 이 별도 overlay에 빌드합니다.

```bash
# 빌드할 때는 venv를 해제합니다.
bash /home/irs/ros2_ws/src/construct_robot_ros2/tools/build_hongik_field.sh
```

기존 KIRO 메모리 패치·SHA 고정·NumPy 모듈 로더와 그 전용 테스트/등록은 제거했습니다.
`construct_robot/launch/hongik_precision_sensing.launch.py`는 이전 파일명 호환용으로만 남아 있으며
공식 `hongik_scan_ros/hongik_moveit.launch.py`를 그대로 include합니다. 권장 실행은 공식 launch입니다.
`~/venv_hongik`의 세 Hongik 패키지 소스 연결도 갱신본으로 바꿨습니다.

## 다음 터미널과 GUI 실행

새 터미널에서 overlay를 마지막에 source합니다. GUI도 이 환경에서 재실행하면 새 패키지와 교시 화면을 사용합니다.
현재 켜져 있던 GUI/MoveIt/로봇 제어 프로세스는 종료하지 않았습니다.

```bash
source /home/irs/ros2_ws/src/construct_robot_ros2/tools/activate_hongik_field.bash
```

서버를 새로 실행할 때에는 기존 서버를 먼저 종료하고:

```bash
ros2 launch hongik_scan_ros hongik_moveit.launch.py \
  platform_config:=$HOME/hongik/hongik_kiro.yaml \
  calib_file:=/home/irs/ros2_ws/install/eoat_description/share/eoat_description/calib/eoat_kiro_01.yaml \
  job_file:=$HOME/hongik/job_kiro.yaml \
  out_root:=$HOME/hongik/runs \
  kiro_confirm_mode:=each
```

이번 적용에서는 서버의 idle 상태를 두 번 확인한 뒤 이전 Hongik launch만 닫고,
동일한 platform_config·calib_file·job_file·out_root로 공식 field launch를 기동했습니다.
이전 서버의 `kiro_confirm_mode=none`도 유지했습니다. 위 재실행 예시는 이동별 확인을 위한 `each`를 명시합니다.
서버 PID와 설정, status 확인 기록은 `docs/artifacts/hongik_field_update/runtime_verification.json`에 있습니다.
확인 결과: `ready=True`, `running=False`, `phase=idle`; NumPy 1.21.5 / SciPy 1.8.0 (apt).
실제 센싱 목표와 로봇 이동 요청은 보내지 않았습니다.

클라이언트는 venv Python으로 실행할 수 있습니다(좌표 단위 m):

```bash
~/venv_hongik/bin/python -m hongik_scan_ros.precision_sensing_client --help
```

## 적용된 동작

- 후보 계획 시간의 정확한 하한으로 최선 우선 탐색; 직선이 없을 때 OMPL 시도, 실패하면 다음 후보.
- 선택 후보의 동일한 계획 핸들을 정확히 한 번 전송; 전송 전 전체 관절 상태·충돌·정지 재검사.
- 계획 서비스 요청을 한 번만 전송하고 무응답을 즉시 구분; 취소/종료 대기와 장면 정리 보강.
- 선언된 로봇 도달 범위로 `hongik/under` 생성; 고정 링크 겹침은 ACM에서 별도 처리.
- 토치 neck rule로 계획 단계에서 롤 결정.
- 주 방향 파일은 `kiro/`; 추가 `directions/S_to_E/`와 `directions/E_to_S/` 출력.
  서버에서는 추가 방향에도 D14를 수행하고 미검증 세트는 verification으로 이동합니다.
  로컬 오프라인 추정에는 실제 팔의 D14 검사가 없습니다. 파일의 start/goal을 바꾸는 것으로 방향을 바꾸지 않습니다.
- 서버/작업 프로세스의 수치 패키지 일치와 시작 로그, JTC reference-feedback 추종 오차 기록.

기하·관절 한계·충돌 모형에 없는 실제 케이블/호스 제약을 보장하는 변경은 아닙니다.

## GUI 교시 점 분리

I/J의 센서 스캔 자세는 기존 `scan_start.yaml` / `scan_end.yaml`에 저장합니다.
O/K의 local 스캔 자세도 기존 `scan_local_start.yaml` / `scan_local_end.yaml`을 사용합니다.
이 키들은 홍익 액션 A/B 입력을 채우지 않습니다.
홍익 액션에는 와이어 끝을 이음매 위에 두고 별도의 `Save wire tip → A/B` 버튼으로
`seam_start.yaml` / `seam_end.yaml`을 저장합니다. 직접 mm 좌표를 입력하는 것도 가능합니다.
기존 스캔 자세 파일은 이음매 점으로 자동 복사하지 않았습니다.
로컬 추정의 기본값은 측정한 groove에서 rough point를 유도하며, `--rough ab`의 fallback도
이 별도 seam 파일을 읽습니다. 교시된 실제 자세를 자동으로 수정하지 않았습니다.

## 검증

빌드: 6개 패키지 성공. eoat_description의 미사용 PYTHON_EXECUTABLE CMake 경고만 있었습니다.
RELAY 필수 시험 14개 대상은 package별 별도 pytest 프로세스로 실행하여 모두 통과했습니다.
ROS 시험은 격리 domain과 가짜 플랫폼/서비스를 사용했습니다.

- scan_l0 (100개): test_joint_selection, test_acquisition, test_autoteach_units,
  test_weld_direction_sets_server, test_cell_world, test_g1_worker, test_undefined_names,
  test_sim_teach1g::test_i_dry_run_plans_only.
- hongik_scan_ros (77개): test_moveit_checked_joint, test_service_calls_and_shutdown,
  test_kiro_io, test_package_scope.
- scan_core (24개): test_weld_directions, test_kiro_plan.
- KIRO (79개): test_hongik_sensing_launch, test_scan_teaching, test_direct_scan,
  test_scan_reference, test_precision_sensing_test, test_launch_parameters, test_package_boundaries.

합계 280개 통과. pytest 6.2.5와 무관한 호스트 anyio 플러그인의 충돌을 피하기 위해
시험 프로세스에서만 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1을 설정했습니다.
서로 다른 package의 conftest를 하나의 프로세스에서 import하면 충돌하므로 package별로 나누었습니다.
실제 서버의 수치 패키지·준비 상태, venv 새 소스 import, 클라이언트 --help도 확인했습니다.
실제 팔의 궤적 실행과 센서 측정은 수행하지 않았습니다.
