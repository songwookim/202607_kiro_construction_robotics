# 후속 스캔 추가 패치 적용 — postweld_extra_vs_855838e

`/home/irs/Downloads/kiro-1g-field_postweld_extra_vs_855838e.patch`를 현재
`hongik_fine_sensing_field/`에 적용했습니다. 첨부 패치의 7개 파일을 변경했고,
기존 pytest 6 포락선 비교 보완과 단일 서버 이름·시작 허용값 수정은 유지했습니다.
변경 전 파일, 수신 패치와 SHA256은 `docs/artifacts/hongik_post_weld_extra_update/`에 기록했습니다.

## 현재 동작

- 기본 `tool_orientation=pass`: 실행의 센싱 패스와 같은 센서·토치 자세를 사용합니다.
  Auto 방향은 그 자세에서 TCP 거리가 가까운 끝부터 시작합니다.
- `least_rotation`: 현재 자세에서 회전이 작은 토치 쪽을 고른 뒤 가까운 끝을 선택합니다.
  방향에 따른 작은 피치 차이만으로 먼 끝을 선택하지 않도록 공구의 물리 방향부터 비교합니다.
- 새 서버의 첫 요청이 후속 스캔이어도 세션의 모션 백엔드 생성 후 능력을 검사합니다.
- `directions/E_to_S/kiro` 같은 하위 출력 폴더를 --run으로 주면 상위 실행 폴더와 방향 선택 방법을 안내하며 거절합니다.
- 용접 자세에서 판 법선으로 먼저 상승하고, 제한된 상승 구간의 장면을 복원하는 이전 패치의 동작은 유지합니다.

`PostWeldScan.action.tool_orientation` 필드가 추가되어 메시지를 포함한 6개 패키지를
별도 `~/hongik_field_ws`에 재빌드했습니다. 기존 ros2_ws의 build/install/log는 수정하지 않았습니다.
용접 아크·기존 교시·실행 결과·job_kiro.yaml은 변경하지 않았습니다.

## GUI와 CLI

GUI의 Post-weld raw scan에서 Tool orientation을 선택합니다.
`Sensing pass`가 기본값이며, `Least rotation`은 최소 회전 옵션입니다.
메시지 구조가 바뀌었으므로 새 overlay를 source한 터미널에서 GUI를 재실행해야 합니다.

```bash
source /home/irs/ros2_ws/src/construct_robot_ros2/tools/activate_hongik_field.bash
ros2 run hongik_scan_ros hongik_post_weld_scan \
  --run /home/irs/hongik/runs/20261015-005527_kiro_1G \
  --after-pass 1 --direction 'E->S' --tool-orientation pass --confirm each
```

위 명령은 실제 스캔 요청 예시이며 적용 중 실행하지 않았습니다.
현재 토치 쪽을 우선하는 옵션은 `--tool-orientation least-rotation`입니다.

## 환경 검증

기존 경로가 이미 등록된 터미널에서 setup.bash만 반복 source하면 underlay가 우선되는 상황을 확인했습니다.
활성화 도구에서 새 overlay의 AMENT/CMAKE/COLCON/PYTHON/라이브러리 경로를 앞으로 옮기도록 보완했습니다.
다른 경로는 유지하며 중복을 제거합니다. 반복 source 후에도 새 share와 메시지·순수 파이썬 모듈이 선택됨을 확인했습니다.
변경된 두 공구 자세와 세 방향을 새 ROS 메시지의 직렬화·역직렬화로 확인했습니다.
최초 시험은 기존 workspace 소스를 선택하여 실패했으며, 환경 보완 후 올바른 소스에서 다시 검증했습니다.

최종 시험은 총 154개 통과했습니다.

- scan_l0 63개: test_post_weld_units, test_cell_world, test_autoteach_units, test_sim_post_weld.
  가상 팔 통합 시험은 올바른 소스 환경에서 212.79초 후 통과했습니다.
- hongik_scan_ros 26개: test_post_weld_action, test_moveit_checked_joint, test_service_calls_and_shutdown, test_package_scope.
- KIRO 65개: test_scanner_scan_task, test_precision_sensing_test, test_hongik_pass_motion,
  test_hongik_sensing_launch, test_hongik_post_weld_safety.

Hongik 런치만 재시작했습니다. 최종 확인: 서버 PID 866463, 런치 PID 866460,
ready=True, running=False, phase=idle, PostWeldScan 액션 서버 1개.
기존 config/calib/job/out_root와 kiro_confirm_mode=none은 유지했습니다.
시험·최종 단일 서버 준비 상태는 같은 artifacts 폴더의 runtime_verification.json에 기록했습니다.
실제 로봇 이동·스캔·용접 목표는 보내지 않았습니다.
