# RELAY 2026-10-14c TCP 이동 선택 적용

`/home/irs/Downloads/Re_ swkim/`의 증분 패치와 전달 문서를 사용했습니다.
두 파일은 SHA256SUMS와 일치합니다. 같은 목록의 이전 전달 문서(b)는 첨부 폴더에 없으며 이번 적용에 필요하지 않습니다.
누적 패치를 다시 적용하지 않고, 지난번 갱신한 `hongik_fine_sensing_field/`에 네 파일의 증분 패치만 적용했습니다.
변경 전 파일은 `docs/artifacts/hongik_tcp_motion_update/before_tcp_motion.tar.gz`에 보관했습니다.

이 트리는 construct_robot_ros2 저장소의 하위 폴더이므로 저장소 루트에서 다음 명령을 사용했습니다.
하위 폴더에서 git apply만 호출하면 git의 경로 필터로 파일이 건너뛰어질 수 있습니다.

```bash
git apply -p2 --check --directory=hongik_fine_sensing_field '/home/irs/Downloads/Re_ swkim/kiro-1g-field_tcpmotion_vs_70cb923.patch'
git apply -p2 --directory=hongik_fine_sensing_field '/home/irs/Downloads/Re_ swkim/kiro-1g-field_tcpmotion_vs_70cb923.patch'
```

## 선택 기준

1. 시작 자세에서 TCP 틀의 회전각이 작은 공구 방향.
2. 토치 팁의 직선 이동거리가 작은 방문 순서.
3. 그 공구 자세에 대한 팔 형태는 기존 관절 경로 계획으로 선택.

선호 방향에 사용 가능한 경로가 없으면 다른 방향을 검사합니다.
TCP는 현재 설정의 `weld_torch.tcp_frame=right_manipulator_ee_point`를 사용합니다.
이름이나 20 mm TF를 이번 패치에서 변경하지 않았습니다.
TCP 값은 시작·목표 자세 사이의 회전각과 직선거리이며, 실제 경로 전체의 누적 회전이나 케이블 안전 보장은 아닙니다.
충돌·여유 영역·관절 한계 검사와 시작 자세 거부는 유지됩니다. 설정과 신규 매개변수는 없습니다.
선택 결과는 `verification/run_log.json`의 `visit_order`에 회전각, TCP 거리, 비교 후보로 기록됩니다.

## 빌드와 시험

기존 `tools/build_hongik_field.sh`로 `~/hongik_field_ws`의 6개 패키지를 재빌드했습니다.
기존 ros2_ws의 build/install/log는 수정하지 않았습니다.
PYTHONNOUSERSITE=1, PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1을 시험 프로세스에 적용했습니다.

- scan_l0 회귀 132개: test_joint_selection, test_acquisition, test_autoteach_units,
  test_autoteach_round4_units, test_autoteach_serpentine_units, test_post_weld_units,
  test_l0_config_plan, test_undefined_names.
- test_sim_teach1g::test_i_dry_run_plans_only 1개: ROS 환경을 source한 뒤 통과.
  최초 실행은 ROS 패키지 메시 경로를 찾지 못해 시작 전에 거부됐습니다.
- construct_robot/test/test_hongik_tcp_motion_selection.py 5개:
  작은 회전 우선, 선호 경로 충돌 시 대체, 같은 회전의 거리 비교,
  양쪽 충돌 시 거부, 토치 오프셋을 포함한 거리 계산.

합계 138개 통과. 가상 플랫폼 또는 스크립트된 경로 검사만 사용했습니다.
실제 팔 이동·스캔·용접을 요청하지 않았습니다.

## 재기동

Hongik status의 running=False, awaiting_confirm=False를 확인한 뒤 Hongik 런치만 SIGINT로 종료했습니다.
기존 platform_config, calib_file, job_file, out_root와 kiro_confirm_mode=none을 유지하여 공식 런치를 재기동했습니다.
MoveIt·로봇 제어·GUI는 재시작하지 않았습니다.
최종 확인: 서버 PID 611644, 런치 PID 611632, ready=True, running=False, phase=idle.
시작 로그에서는 MoveIt 장면 서비스가 아직 발견되지 않았지만, 이후 `/get_planning_scene`의 가용성을 확인했습니다.
최종 PID·준비 상태는 같은 artifacts 폴더의 runtime_verification.json에 기록했습니다.

전달 문서의 좌표 단위와 시작 자세 조건도 그대로 적용됩니다.
액션 A/B는 m 단위이며, 교시 후에는 도구 전체가 시편의 50 mm 여유 영역 밖인 자세에서 시작해야 합니다.
