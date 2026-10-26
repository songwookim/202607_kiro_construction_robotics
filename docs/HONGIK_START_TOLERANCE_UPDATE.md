# RELAY 2026-10-15c 시작 자세 허용값 적용

`/home/irs/Downloads/Re_ Re_ Re_ swkim/RELAY_2026-10-15c_start_tolerance.md` 지침에 따라
현재 TCP 이동 선택 패치가 적용된 `hongik_fine_sensing_field/`에 증분 패치를 반영했습니다.
첨부에는 별도 SHA256SUMS가 없으므로 수신 파일의 해시와 적용 결과를 artifacts에 기록했습니다.
기존 파일 네 개는 `docs/artifacts/hongik_start_tolerance_update/before_start_tolerance.tar.gz`에 보관했습니다.

저장소 루트에서 적용한 명령:

```bash
git apply -p2 --check --directory=hongik_fine_sensing_field '/home/irs/Downloads/Re_ Re_ Re_ swkim/kiro-1g-field_starttol_vs_287bee5.patch'
git apply -p2 --directory=hongik_fine_sensing_field '/home/irs/Downloads/Re_ Re_ Re_ swkim/kiro-1g-field_starttol_vs_287bee5.patch'
```

## 변경

- 확인한 관절 궤적을 보내기 직전 시작 자세의 최대 관절 차이를
  `move_group`의 `trajectory_execution.allowed_start_tolerance`로 판정합니다.
- 파라미터는 계획 서비스와 같은 이름공간의 `/move_group/get_parameters`에서 한 번 읽고 캐시합니다.
  읽기 실패 또는 0이면 첨부 지침의 기본값 0.01 rad(약 0.573°)를 사용하고 메시지에 출처를 남깁니다.
- 과거 실행 002753의 0.0542° 차이를 거절한 0.05°/0.1 mm 재사용 기준을 실행 거절에 사용하지 않습니다.
  플랜지 위치·회전 차이는 거절 메시지에서 진단 정보로 표시합니다.
- 다른 팔의 관절 차이 검사도 같은 허용값을 사용하며, 보내기 직전 전체 궤적의 현재 장면 충돌 재검사와 정지 검사는 유지합니다.
- 1G 시작 로그의 방문 순서 설명을 이미 적용된 TCP 회전·거리 선택 기준에 맞췄습니다.
- 패키지 검사에서 새 GetParameters import에 필요한 `rcl_interfaces` 선언 누락을 발견하여
  `hongik_scan_ros/package.xml`에 exec_depend 한 줄을 추가했습니다. 이 보완은 별도 패치로 artifacts에 기록했습니다.
- 단일 서버 확인 중 내부 MoveIt 어댑터까지 전역 `__node` remap으로 같은 노드 이름을 갖게 되어
  ROS 액션 조회에서 한 서버가 두 번 집계되는 기존 문제가 확인됐습니다.
  공식 런치의 서버 Node에서 중복된 name 지정을 제거하여 서버 코드의 원래 이름과 어댑터의 별도 이름을 유지합니다.
  액션·서비스 이름과 로봇 제어 동작은 변경하지 않았고, 별도 보완 패치를 artifacts에 기록했습니다.

## 센서 복원

문서의 남은 작업인 원본 설정 복원도 완료했습니다.
`~/scan_data/sensor_backup/backup_20261014-122707.txt`는 INIT 전 원본 설정이고 LaserMode=0입니다.
현재 센서의 CameraSNR=4040760, CameraVK를 백업과 대조했고, 다른 TCP 수집 접속이 없으며
홍익 서버가 실행 중이 아님을 확인했습니다.
복원 직전 설정을 `sensor_before_restore.txt`에 저장하고, 37개 쓰기 가능한 원본 설정을 복원했습니다.
센서 reply error=0, 37/37 읽기 대조 일치, 불일치 없음.
복원 결과 LaserMode=0, DataMode=0, AutoTriggerFPS=50입니다.
새 스캔 세션에서 작업별 INIT를 다시 적용하는 동작은 유지합니다.
복원 후 레이저·RViz 수집을 새로 켜지 않았습니다.

## 검증과 실행

`tools/build_hongik_field.sh`로 별도 `~/hongik_field_ws`의 6개 패키지를 재빌드했습니다.
기존 ros2_ws의 build/install/log는 수정하지 않았습니다.
시험은 venv_hongik과 ROS overlay를 사용했고, pytest 외부 플러그인 자동 로드를 시험에서만 비활성화했습니다.

- hongik_scan_ros 78개: test_moveit_checked_joint, test_service_calls_and_shutdown, test_kiro_io, test_package_scope.
  최초 패키지 검사에서 빠진 exec_depend를 발견했으며, 보완 후 해당 검사와 관절 실행 시험 15개를 다시 실행해 통과했습니다.
- KIRO 11개: test_hongik_start_tolerance(6개), test_hongik_tcp_motion_selection(5개).
  파라미터 이름공간·한 번 읽기·실패/0의 기본값 출처와 TCP 선택 회귀를 확인했습니다.
- 런치 회귀 4개: test_hongik_sensing_launch. 서버와 내부 어댑터 이름을 전역 재지정하지 않는 것도 확인했습니다.

합계 93개 통과. 가짜 서비스·스크립트된 플랫폼만 사용했으며 실제 로봇 목표나 용접 명령은 보내지 않았습니다.
실제 move_group의 파라미터 서비스는 가용하지만 해당 값은 미설정 또는 0으로 응답하여,
첨부 지침대로 기본값 0.01 rad를 선택했습니다. 읽기 전용 조회 결과는 `move_group_tolerance_readback.json`에 있습니다.

홍익 서버의 running=False, awaiting_confirm=False를 다시 확인한 뒤 홍익 런치만 SIGINT로 종료했습니다.
동일한 config/calib/job/out_root와 kiro_confirm_mode=none을 유지해 공식 런치를 재기동했습니다.
최종 확인: 서버 PID 698937, 런치 PID 698910, ready=True, running=False, phase=idle,
PrecisionSensing Action servers=1. 준비 상태와 단일 액션 서버 확인 결과는 `runtime_verification.json`에 기록했습니다.
MoveIt·로봇 제어·GUI의 설정이나 실행 상태는 이번 작업에서 변경하지 않았습니다.
