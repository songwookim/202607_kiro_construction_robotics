# RELAY 2026-10-15d 후속 스캔 적용

이후 추가 패치에서 공구 자세의 기본값이 센싱 패스 자세로 변경됐습니다.
최신 기본값과 옵션은 [HONGIK_POST_WELD_EXTRA_UPDATE.md](HONGIK_POST_WELD_EXTRA_UPDATE.md)를 참고하세요.

`/home/irs/Downloads/Re_ Re_ Re_ swkim (1)/RELAY_2026-10-15d_post_weld.md`와
`kiro-1g-field_postweld_vs_3009cdb.patch`를 현재 현장 소스에 적용했습니다.
소스는 `hongik_fine_sensing_field/`이며, 기존 2026-10-15c 수정(시작 허용값, 의존성, 단일 서버 이름)과 GUI 원본 패스 이동은 유지했습니다.
수신 파일 해시·원본 7개 파일 백업·적용 결과는 `docs/artifacts/hongik_post_weld_update/`에 있습니다.

## 변경과 범위

- 후속 스캔만 용접 직후 자세에서 시작할 수 있습니다.
  처음 접촉이 모두 `hongik/*`의 여유 영역일 때 판 법선 방향으로 상승하고 공구 자세를 유지합니다.
- 도구 최저점의 clearance와 광학 원점의 standoff_min을 만족하는 상승 끝 자세를 포락선 검사합니다.
  해당 직선에서만 Hongik 충돌 물체를 잠시 제거하며, 로봇 자체와 현장 고정 물체 검사는 유지합니다.
  상승 실행의 finally에서 Hongik 물체를 다시 넣고 실제 팔이 밖에 있는지 확인합니다.
- 다른 물체와의 접촉이나 자기 충돌, 일반 정밀 센싱의 시작 자세 거부 조건은 유지합니다.
- 후속 스캔의 방향·공구 자세는 시작 TCP 회전각, 토치 팁 거리 순으로 선택합니다.
  강제 방향은 새 `hongik_scan_msgs/action/PostWeldScan.direction` 필드로 전달합니다.
- 스캔 탭의 Post-weld raw scan에 `Auto / S->E / E->S` 선택을 추가했습니다.
  Auto는 빈 direction 문자열을 보내고, 명시된 방향은 그대로 전달합니다.
  이전 메시지를 캐시한 GUI는 실행 전에 새 overlay source 및 재시작 안내를 표시합니다.

원자료 수집 스캔이며, 용접 아크나 기존 용접 경로·교시 파일을 수정하지 않습니다.
기존 job_kiro.yaml의 용접 방향도 이번 작업에서 바꾸지 않았습니다.
CLI에서 `>`가 셸 리다이렉션으로 해석되지 않도록 방향 문자열은 따옴표로 감쌉니다.

```bash
source /home/irs/ros2_ws/src/construct_robot_ros2/tools/activate_hongik_field.bash
ros2 run hongik_scan_ros hongik_post_weld_scan \
  --run /home/irs/hongik/runs/20261015-005527_kiro_1G \
  --after-pass 1 --direction 'E->S' --confirm each
```

위 CLI는 실제 이동을 요청하는 예시입니다. 작업 중 실행하지 않았습니다.
GUI는 동일한 overlay를 source한 터미널에서 재실행해야 새 액션 타입을 사용합니다.
GUI의 Dry run은 기본 켜짐이며, 실제 후속 스캔은 이를 해제한 뒤 Start post-weld scan으로 요청합니다.

## 빌드와 검증

`tools/build_hongik_field.sh`로 메시지를 포함한 6개 패키지를 `~/hongik_field_ws`에 재빌드했습니다.
기존 ros2_ws의 build/install/log는 수정하지 않았습니다.
새 action Goal의 세 방향 값을 CLI에서 만들고 ROS 직렬화·역직렬화로 확인했습니다. 실제 목표는 보내지 않았습니다.

가상 팔 통합 시험은 용접 직후 자세의 상승, E->S 방향, 원자료 저장까지 검사합니다.
최초 실행에서는 동작 검사가 통과한 뒤 pytest 6의 중첩 dict/list 부동소수점 비교가 실패했습니다.
같은 허용오차로 각 포락선 필드를 비교하도록 시험만 보완했으며 별도 패치로 보관했습니다.
재시험 도중 디스크 공간이 부족해 임시 시험 폴더 한 개만 삭제하고 /dev/shm에서 검증을 계속했습니다.
실제 스캔 결과·용접 로그는 삭제하지 않았습니다.
최종 시험: 총 150개 통과.

- scan_l0 63개: test_post_weld_units, test_cell_world, test_autoteach_units, test_sim_post_weld.
  가상 팔 통합 시험은 메모리 임시 공간에서 200.53초 후 통과했습니다.
- hongik_scan_ros 26개: test_post_weld_action, test_moveit_checked_joint,
  test_service_calls_and_shutdown, test_package_scope.
- KIRO 58개: test_scanner_scan_task, test_precision_sensing_test,
  test_hongik_pass_motion, test_hongik_sensing_launch.
- 복원 안전 3개: test_hongik_post_weld_safety. 상승 실패 시 장면 복원,
  포락선 거부 시 무전송, 상승 후 접촉 시 추가 이동 거부를 확인했습니다.

Hongik 런치만 재시작했습니다. 최종 확인: 서버 PID 805131, 런치 PID 805082,
ready=True, running=False, phase=idle, PostWeldScan 액션 서버 1개.
기존 config/calib/job/out_root와 kiro_confirm_mode=none을 유지했습니다.
최종 시험과 서버 재시작 상태는 runtime_verification.json에 기록했습니다.
마지막 디스크 확인에서 사용 가능한 공간이 약 44 MB뿐이므로 실제 스캔 원자료 저장 전 공간 확보가 필요합니다.

실제 로봇 이동·레이저 켜기·스캔·용접 요청은 이 적용 작업에서 수행하지 않았습니다.
