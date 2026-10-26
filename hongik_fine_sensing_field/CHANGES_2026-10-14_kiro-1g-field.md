# kiro-1g-field 변경 기록 (2026-10-14, KIRO 현장 통합)

기준: 납품본 `kiro-1g` (ec00145 + 06b2bdd venv 도구). 이 기록의 변경은 모두 복사 브랜치 `kiro-1g-field` 에 있고, `kiro-1g` 자체는 그대로다.
현장 입력: KIRO 작업공간·diff (`construct_robot_ros2`), 현장 직선 스캔 (`kiro_scan/*`), 실패한 실행 결과 (`hongik_auto_teach/*`).
상세 근거: `fsv1_work/kiro_field/2026-10-14/` 의 `REVIEW_kiro_diffs.md` (KIRO diff 검토·현장 실패 원인), `NUMPY_diagnosis.md`.

## 요약

| # | 항목 | 왜 (현장 증상) | 무엇을 바꿨나 | 커밋 |
|---|---|---|---|---|
| 1 | 토치 롤 | 토치가 반대로 꺾임: 몸통이 판 위 67–75° 로 섬 (9월 KIRO 자세는 시편과 거의 평행) | 롤을 D14 안에서만 고르던 것을 계획 단계에서 끌기 규칙으로 정함 | 65fdff5 |
| 2 | 계산 작업 프로세스 numpy | KIRO PC 의 numpy 2 를 작업 프로세스가 잡음 → 첫 배열 결과에서 실패 | 작업 프로세스가 서버가 불러온 numpy·scipy 를 그대로 씀 | a7c3dda |
| 3 | 시편 아래 모션 금지 | 시편 아래로 팔이 지나갈 수 있음 (받침 아래 50 mm 판뿐, 테스트베드가 있으면 잘려 나감) | 셀 세계에 `hongik/under` 고체 추가 | 65594d8 |
| 4 | 서비스 무응답 오류 형식 | move_group 무응답을 후보마다 하나씩 거절 (수 분 낭비) | `PlatformUnavailable` (MotionError) 추가 | e986864 |
| 5 | 런치의 numpy 환경 | 서버 자체가 numpy 2 를 잡으면 apt scipy 1.8 이 깨짐 | 우리 노드에 apt numpy·scipy 경로를 붙임 (`numeric_site`) | de64492 |
| 6 | B1 관절 경로 선택 | 260°/352° 큰 회전 (실행 134838), KIRO 패치는 기록과 다른 후보를 실행하는 버그 | 실제 계획 시간 최소 후보를 정확한 하한으로 고름 | 0eda9f1, c21c8df |
| 7 | B2 후보 순위 | 가까운 해를 뒤로 미는 상대 조건 규칙 | 규칙 삭제, Σ\|Δq\| + min_sigma 동률 깨기 | c21c8df |
| 8 | B3 서비스 호출 | 5 s 계획 요청을 3 s 마다 재전송 → move_group 무응답 (144142, 144626) | 계획 요청은 한 번만, 무응답은 PlatformUnavailable, 대기는 멈춤 신호로 즉시 종료 | 2ca130c, f3363a5 |
| 9 | B4 종료·결과 분류 | 실행 중 서버 종료 → `InvalidHandle` (134602, 144142, 144626), 우리 장면 물체가 남음 | 목표가 끝난 뒤 닫기, 취소·종료는 ABORTED, 시작 시 남은 물체 정리, 시작 기록·numpy 확인 | a333003, 72dbc02 |
| 10 | B5 추종 기록 | JTC 경로 허용오차 중단 (134838) 의 원인을 기록에서 볼 수 없었음 | 관절 이동마다 기준−피드백 최대 오차를 기록 | c21c8df |
| 11 | B6 메시지 | 거부 이유가 현장에서 바로 이해되지 않음 | 세계 안 시작·90° 심 거부 메시지 보강 | c21c8df |
| 12 | 양방향 출력 | 한 번 센싱으로 두 방향 모두 용접할 수 있어야 함 | `directions/S_to_E/`, `directions/E_to_S/` 를 매 실행 출력 (D14 포함) | e1cf40f |
| 13 | 센싱 자세 선택 = TCP 최소 이동 | 관절 이동 최소로 고른 획득 자세가 토치를 115° 돌려 베이스 반대쪽으로 보냄 → 와이어가 베이스에서 반대쪽까지 돌아감 (21:44) | 공구 자세를 토치 팁 TCP 의 회전 최소 → 토치 팁 거리 최소로 고르고, 관절 이동 시간은 그 자세의 팔 형태만 정함 | f0774b6 |
| 14 | 확인한 관절 계획의 시작 허용값 | 계획 시작과 보내기 직전 측정 관절이 0.054° 달라 실행 거부 (002753) — 재사용용 0.05°/0.1 mm 기준을 쓰고 있었음 | move_group 자신의 `allowed_start_tolerance` (실행 때 읽음, 없으면 MoveIt 기본 0.01 rad) 로 판정 | 34c5e37 |
| 15 | 용접 후 재스캔: 용접 자세에서 시작·방향 선택 | 용접 직후 토치가 시편 근처라 시작 거부; 짧은 관절 경로가 막혀 플랜지 331° 회전 후보 선택; 방향·자세가 관절 최소·센싱 패스 자세로 강제 | 먼저 판 법선으로 곧게 올라감 (자세 유지), 방향·공구 자세 = TCP 최소 이동 (회전 → 토치 팁 거리), `--direction` 으로 지정 가능 | 0fa4d35 |

---

## 1. 토치 롤을 계획 단계에서 정한다 (65fdff5)

**왜**: 용접 자세는 세 자유도다. 노즐 축의 진행각·작업각 (2) 과, 노즐 축 둘레 롤 (1) 이다. 우리 토치는 30° 꺾여 있어서 롤 0° 와 180° 의
몸통 위치가 정반대다. 롤은 실기 서버의 D14 (용접 자세 도달성 검사) 안에서만 골랐다. 그래서 D14 가 없는 경로에서는 기본값 0° 가 그대로
쓰였다. KIRO 오프라인 추정기 (`write_outputs` + `write_kiro_outputs`) 가 이런 경로다. 끌기 42° 에서 롤 0° 는 목이 진행각 반대로 꺾인다.
몸통은 판 위 67–75° 로 거의 선다.

**무엇을·어떻게**:
- `scan_core/weld_orient.py`: `rule_roll(torch, travel_degs)` 를 추가했다.
  - 규칙은 NECK_RULE 이다. 곧은 몸통은 노즐보다 더 기울고, 진행각과 같은 쪽에 있다.
  - 이 규칙으로 모든 패스의 WPS 진행각에 맞는 롤을 고른다. 고정 상수는 없다.
  - 규칙이 모든 롤을 허용하면 (진행각 0, 곧은 토치) 그렇다고 기록한다.
  - 패스마다 쪽이 다르면 가장 많은 패스를 만족하는 롤을 고르고, 나머지 패스를 이름으로 남긴다.
- `scan_core/weld_plan.py` `plan_weld`: 계획을 만들 때 `torch_roll_deg`·`torch_roll` 을 위 규칙으로 채운다. D14 는 여전히 그 롤만 탐색·확인한다.
- `scan_l0/teach_1g.py`: 토치가 이 팔에 없어 D14 를 하지 않는 경우에도 0° 로 덮어쓰지 않는다. 계획의 규칙 롤을 두고, "D14 확인 없음"을 기록한다.

**확인**:
- 현장 스캔 `20261014-160018` 을 KIRO 오프라인 파이프라인 그대로 다시 계산했다.
  - 노즐 각도와 위치는 같다.
  - 몸통은 판 위 16–18° 이고, 플랜지가 진행 방향 앞에 있다. S→E 와 E→S 모두 같다.
- 작업각은 두 방향 모두 시편 좌표에서 같은 쪽 (베벨 쪽) 이다.
- 시험: `test_weld_orient`, `test_kiro_plan` (갱신), `test_weld_line`, `test_weld_plan`, `test_weld_d14_exact` 모두 통과.

## 2. 계산 작업 프로세스가 서버의 numpy·scipy 를 그대로 쓴다 (a7c3dda)

**왜**:
- KIRO PC 에는 numpy 2 가 `~/.local` 과 `/usr/local` 두 곳에 있다. 둘 다 apt numpy 1.21.5 보다 먼저 잡힌다.
- apt scipy 1.8 은 numpy 1.x 용으로 컴파일돼 있다.
- KIRO 의 서버 래퍼는 서버 프로세스에만 apt numpy 를 넣는다. 우리 G1 작업 프로세스는 `python -c` 로 새로 뜨고 경로 순서로 numpy 를 찾는다.
  그래서 numpy 2 를 잡는다.
- 증상: 첫 배열 결과에서 서버가 `numpy._core` 를 찾지 못해 실패하고, 작업 프로세스 안에서는 scipy 가 실패한다.

**무엇을·어떻게**:
- `scan_l0/g1_worker.py`: `numeric_packages()`, `worker_pythonpath()`, `_check_numeric()` 을 추가했다.
- 작업 프로세스의 PYTHONPATH 맨 앞에 서버가 실제로 불러온 numpy·scipy 의 디렉터리를 둔다.
- 그래도 다른 것을 불러오면 작업 프로세스는 이유를 stderr 에 남기고 종료한다 (`HONGIK_WORKER_NUMERIC`). 서버는 그 종료를 기록한다.

**확인**:
- KIRO 환경을 흉내 냈다 (numpy 2 가 apt 보다 앞).
  - 수정 전: 작업 프로세스가 2.2.6 을 잡고 `ModuleNotFoundError: numpy._core` 로 실패.
  - 수정 후: 서버와 같은 1.21.5 를 잡고 배열 작업이 정상.
- 시험: `scan_l0/tests/test_g1_worker.py` 통과.

## 3. 시편 아래에는 팔이 가지 않는다: `hongik/under` (65594d8)

**왜**:
- 지금까지 받침면 아래에는 C (50 mm) 두께의 판 하나만 있었다.
- KIRO 로봇 모형처럼 고정 몸체 (testbed) 가 시편 밑에 있으면, 로봇 자신과 겹치지 않게 하는 규칙 때문에 그 판이 잘려 나간다.
- 사용자 요구: 시편 아래에서는 어떤 모션도 없어야 한다.

**무엇을·어떻게**:
- `scan_l0/cell_world.py`: `UNDER_ID = "hongik/under"`, `set_reach_below()`, `under()`, `under_object(grow_top=)` 를 추가했다.
  - 모양: 시편의 바닥 면적 (C 만큼 키움) 을 받침면에서 팔이 닿을 수 있는 가장 낮은 곳까지 내린 고체다.
  - 깊이는 플랫폼이 로봇 모형에서 계산한다. 상수는 쓰지 않는다.
- `scan_core/fk_urdf.py`: `KinematicChain.reach_bound_m()` 을 추가했다. 관절 원점 거리의 합으로, 어떤 관절값에서도 도달 거리의 상한이다.
- `hongik_scan_ros/moveit_adapter.py`:
  - `arm_reach_m()` = 체인 상한 + 플랜지에서 툴 (토치 TCP·광학 프레임) 까지의 거리.
  - `set_scene_boxes`: 이 고체에는 로봇의 고정 링크만 겹쳐도 되도록 충돌 허용 목록 (ACM) 에 넣는다. 고정 링크는 움직이지 않기 때문이다.
- `scan_l0/teach_1g.py`:
  - 세계를 적용할 때 팔 도달 상한을 받는다.
  - D14 검사 동안에도 이 고체를 둔다. 이때 윗면은 받침면에 둔다. 토치는 그 위 그루브에서 일한다.
- `scan_l0/tests/sim_arm.py`: 가상 팔에도 `arm_reach_m()` 를 넣었다.

**확인**: `test_cell_world.py` 에 시험을 추가했고 통과했다.
- 시편 아래 모든 점이 막힌다.
- C 만큼 키운 범위와 바닥 한계가 맞다.
- 도달 상한을 모르면 고체를 만들지 않고, 그렇다고 기록한다.

## 4. `PlatformUnavailable` (e986864)

`scan_l0/motion_types.py`: `class PlatformUnavailable(MotionError)` 를 추가했다.
- 서비스 무응답은 후보의 성질이 아니라 기반 시설의 실패다.
- 그래서 후보 탐색이 후보를 하나씩 거절하지 않고 바로 멈추도록 구분한다. B1·B3 에서 쓴다.

## 5. 런치: 우리 노드는 apt numpy·scipy 를 쓴다 (de64492)

**왜**: 서버 프로세스 자체가 numpy 2 를 잡으면 apt scipy 가 깨진다. 지금은 KIRO 가 서버를 자기 래퍼와 메모리 패치로 띄워 피하고 있다.
그 방식은 우리 갱신마다 깨지고 (SHA 고정), 작업 프로세스는 막지 못한다.

**무엇을·어떻게**:
- `hongik_scan_ros/launch/hongik_moveit.launch.py` 에 `numeric_site` 런치 인자를 추가했다. 기본값은 `/usr/lib/python3/dist-packages` (MANIFEST 의 apt 위치) 다.
- 서버 노드 (와 선택 프로파일러 노드) 에 `additional_env` 로 두 가지를 넣는다.
  - `PYTHONNOUSERSITE=1`
  - PYTHONPATH 끝에 `numeric_site` 를 덧붙임
- 효과:
  - PYTHONPATH 항목은 `/usr/local` 보다 앞이므로 apt 쌍이 이긴다.
  - ROS·작업공간 경로 순서는 그대로다.
  - 하위 프로세스는 모두 이 환경을 물려받는다.
- `numeric_site:=''` 이면 환경을 건드리지 않는다.
- 적용 범위는 우리 노드 프로세스뿐이다. KIRO 의 다른 노드는 각자 환경 그대로다.

**확인**:
- `numeric_env()` 단위 확인 (덧붙임·중복 없음·빈 값) 을 했다.
- `test_package_scope`, `test_job_overrides_server` 통과.
- 같은 환경 설정은 흉내 낸 KIRO 환경에서 서버·작업 프로세스·spawn 자식 모두 1.21.5 임을 확인했다.

---

## 6. B1 관절 공간 경로 선택: 실제 계획 시간 최소 (0eda9f1, c21c8df)

**왜**:
- 현장 실행 134838 에서 획득 이동이 손목4 260°, 플랜지 352° 를 돌았다 (26 s). 비상정지 뒤 JTC 가 경로 허용오차로 중단됐다.
- 원인은 우리 후보 순위였다 (7절). 404°·580° 해가 있었는데도 910–1489° 해를 골랐다.
- KIRO 는 '계획 시간 (duration_s) 최소'로 고르는 패치를 보냈다. 방향은 맞다. 하지만 세 가지 문제가 있다.
  - OMPL 확정 뒤 다시 정렬해서 기록과 다른 후보를 실행한다.
  - 정렬 과정에서 손목 띠 선호가 사라진다.
  - OMPL 이 한 번 실패하면 실행 전체를 거부한다.

**무엇을·어떻게**:
- `hongik_scan_ros/moveit_adapter.py` (KIRO 패치에서 좋은 부분을 옮김):
  - 모든 상태 검사에 측정한 전체 관절 상태 (다른 팔·머리 포함) 와 `is_diff=True` 를 보낸다 (`_fill_robot_state`). 대상은 `_first_invalid`, `state_contacts`, `_contact_pairs_at`.
  - `check_joint(allow_ompl=)`:
    - 직선이 막혀도 시간만 추정한 결과 (`estimate_only`) 를 돌려준다. 이것은 실행할 수 없다.
    - 후보마다 자기 계획 핸들 (`_checked_plan`) 을 가진다.
  - `move_checked_joint`: 확인한 계획을 정확히 한 번만 보내고, 다시 계획하지 않는다.
  - 보내기 직전 검사 (`_move`):
    - 최신 관절 상태, 시작 자세, 궤적 전체의 지금 장면 충돌, 다른 관절 정지, 멈춤 신호를 차례로 본다.
    - 하나라도 어긋나면 아무것도 보내지 않는다.
- `scan_l0/joint_selection.py` (신규, KIRO 코드를 import 하지 않음):
  - 하한: 관절별 `max(0, |Δq| − h)/v_max` 와 TCP 사다리꼴 시간 가운데 큰 값.
    - h 는 조밀화 간격이다. 한계는 플랫폼이 선언한 값 (`moveit_jtc.limits`) 이다.
    - 직선이든 OMPL 이든 어떤 관절 경로의 재시간화 결과보다도 크지 않다. 증명은 docstring 에 있다.
    - 가속 항을 넣은 사다리꼴 공식은 하한이 아니다. 재시간화가 그보다 최대 0.42 s 짧게 나오는 반례가 있어 빼고, 시험으로 남겼다.
  - `DurationSearch` (최선 우선):
    - 하한 순으로 직선만 검사한다 (`allow_ompl=False`).
    - 지금까지 최선의 실제 시간이 다음 하한 이하가 되면 멈춘다. 그 결과가 후보 집합의 진짜 최솟값이다. 상수는 없다.
    - 실행 가능한 직선이 하나도 없을 때만 OMPL 을 하한 순으로 돌린다.
    - OMPL 이 실패하면 다음 후보로 넘어간다 (KIRO 패치의 전체 거부를 고침).
  - `PlatformUnavailable` 은 후보 거절이 아니다. 탐색을 바로 멈추고 "move_group not answering" 으로 거부한다.
- `scan_l0/autoteach.py`:
  - `_select_joint` → `_joint_move`: 고른 후보를 동일성으로 실행한다. 확정 뒤 다시 정렬하지 않는다 (KIRO 버그 회귀 시험).
  - `_prefer_yaw_band`: 시간 순서를 따라가며 ±ψ_b 회전도 따르는 첫 후보를 고른다 (`REACH_CANDIDATE_BUDGET` 안).
  - `_plan_acquisition`: 선택을 계획 단계에서 한다. 그래서 dry run 도 실제 계획을 보여 준다.
  - `_choose_visit_order`: Σ|Δq| 대신 같은 시간 하한으로 가지치기를 유지한다.
  - "kept configuration" 결정 기록을 실제로 실행한 후보로 이동 뒤에 남긴다.
  - 결정 기록을 추가했다: `acquisition_filter`, `joint_candidate`, `joint_selection`, `moves[].joint_candidates`.
- 멈춤 확인: `_first_invalid` 의 표본 반복 안, OMPL 시도 사이, 후보 검사 앞뒤 (`_check_stop`).

**확인**:
- `test_joint_selection.py` (신규, 15개):
  - 하한 ≤ 실제 재시간화 시간 (연구실 RB20 URDF, 무작위 경로 7,500개), 가속 항 반례.
  - 최선 우선이 진짜 최솟값을 내며 후보 6개 중 2개만 검사함.
  - OMPL 실패 시 다음 후보로 넘어감, 재정렬 버그 회귀, PlatformUnavailable 즉시 거부, 멈춤 처리, 손목 띠 순서.
- `test_moveit_checked_joint.py` (신규, 9개): 전체 상태 + `is_diff`, 확인 계획 한 번 전송, 2π 관절 변화·다른 팔 이동·충돌·멈춤이면 아무것도 보내지 않음.
- 가상 팔 1G 시험 하나 (`test_far_end_taught_first…`) 통과.
- 가짜 move_group 종단 시험: 아래 '시험' 절 참고.

## 7. B2 후보 순위 (c21c8df)

**왜**: `rank_reach_candidates` 의 상대 조건 규칙 (설계 절반 기준 두 묶음 + 한계 여유 비율) 이 지금 관절 근처의 해를 모두 뒤로 밀었다.
첫 묶음에 910–1489° 해만 남았다 (134838).

**무엇을·어떻게**: `scan_l0/acquisition.py` `rank_reach_candidates` 에서 두 묶음 분할과 한계 여유 비율을 지웠다.
- 순위는 Σ|Δq| 이고, min_sigma 는 동률 깨기에만 쓴다.
- 실제 선택은 6절의 시간 기준이 한다.
- 134838 에 기록된 후보로 회귀 시험을 만들었다 (`test_acquisition.py`).

## 8. B3 서비스 호출 (2ca130c, f3363a5)

**왜**: `call_service` 가 같은 요청을 3 s 마다 다시 보냈다. 5 s 짜리 OMPL 계획 요청이 move_group 에 최대 10번 쌓였다.
move_group 은 서비스를 하나씩 처리하므로 그 뒤의 상태 검사가 무응답이 됐다 (144142, 144626).

**무엇을·어떻게** (`hongik_scan_ros/ros_util.py`):
- `call_service(resend=)`:
  - 계획 요청 (`plan_kinematic_path`, `compute_cartesian_path` 두 곳) 은 한 번만 보내고 제한 시간까지 기다린다.
  - 값싼 반복 가능 요청 (상태 검사·IK) 만 재전송을 유지한다.
- 시간 초과 시 걸린 요청을 거두고 `PlatformUnavailable("<서비스> did not answer within N s …")` 를 낸다.
  - 이전에는 None 을 돌려줘 조용히 실패로 처리됐다.
- 멈춤 신호 (실행 래치, 액션 취소, 서버 종료) 가 있으면 대기가 한 스핀 주기 (기존 0.1 s) 안에 끝난다. 이때 요청을 거두고 `MotionError("stopped …")` 를 낸다.
- 구현: `RosContext` 에 멈춤 원천을 등록하는 방식이다 (`add_stop_source`). 장면 정리 클라이언트는 예외로 둬서, 멈춘 실행도 자기 물체를 지울 수 있다.

## 9. B4 종료·결과 분류 (a333003, 72dbc02)

**왜**:
- 진행 중인 목표가 있는데 서버가 종료되며 노드를 먼저 닫았다. 그래서 `InvalidHandle` 오류가 났다.
- 그 결과는 ERROR_INTERNAL 로 보고됐고, 우리 `hongik/*` 물체가 KIRO 장면에 남았다.
- 취소도 계획 반복까지 닿지 않아 106 s 더 계획했다.

**무엇을·어떻게** (`hongik_scan_ros/auto_teach_server.py`, `scan_l0/autoteach.py`, `teach_1g.py`, `post_weld.py`):
- `request_shutdown`: 새 목표를 거부하고, 확인 대기를 거절하고, 래치를 걸고 취소를 완료한다 ("server shutting down").
- `destroy`:
  - 모든 실행 콜백이 끝날 때까지 실행기를 돌리며 기다린다.
  - 그 뒤 남은 물체를 지우고 `platform.close()` 한다.
  - 새 시간 상수는 없다. 멈춘 뒤 서비스 대기는 한 스핀 안에, 다른 대기는 기존 제한 시간 안에 끝난다.
- rclpy 기본 SIGINT 처리가 컨텍스트 전체를 꺼서 정리가 실패했다. 그래서 신호를 기록만 하고 위 순서로 닫는다. 두 번째 신호면 기다림을 멈춘다.
- 실행 결과 분류: 취소나 종료 뒤의 실패는 ABORTED 로 보고한다 ("cancel requested by the operator (D7) …" / 종료 문구).
  - `PlatformUnavailable` 은 motion 분류다. 첫 이동 전이면 REFUSED, 이후면 ABORTED 다.
- `_clear_leftover_scene`: 서버 시작, 매 실행 전, 종료 때 장면에서 `hongik/` 로 시작하는 우리 물체를 모두 지운다. 다른 물체는 건드리지 않는다.
- 시작 기록 (`hongik_scan_ros/numeric_env.py` 신규):
  - 사이트 설정 (platform_config, job_file, calib_file, out_root, 확인 모드) 을 한 줄로 남긴다.
  - 실제로 불러온 numpy·scipy 의 버전·파일을 한 줄로 남긴다.
  - scipy 의 컴파일된 확장이 지금 numpy 와 맞지 않으면 영어 한 줄로 이유와 해결 (INSTALL.md 6절) 을 말하고 바로 종료한다.

**확인**: `test_service_calls_and_shutdown.py` (신규, 7개) 를 실제 서버 프로세스로 돌렸다. 확인한 것:
- 한 번 전송과 무응답 오류
- 멈춤이 대기를 끝냄
- destroy 가 목표 스레드를 기다림
- 남은 물체 정리
- ABORTED 분류
- 숫자 패키지 기록·즉시 종료
- SIGINT 로 깨끗이 종료

## 10. B5 JTC 추종 기록 (c21c8df)

**왜**: 134838 의 경로 허용오차 중단 (KIRO 0.15 rad) 을 기록으로 진단할 수 없었다.
KIRO Servo-J 지연 약 0.5 s × 관절 상한 0.24 rad/s ≈ 0.12 rad 로, 평상시에도 허용오차의 80 % 다.

**무엇을·어떻게**: `jtc_tracking_record` 를 추가했다.
- 관절 이동마다 controller_state 의 관절별 최대 (기준 − 피드백) 오차와 그 시각을 `moves[].motion.tracking` 에 남긴다.
- 성공·중단·전송 실패 모두 기록한다. 속도는 바꾸지 않았다.

## 11. B6 메시지 (c21c8df)

- `teach_1g._apply_world` (팔이 세계 안에서 시작) 거부 메시지에 다음을 더했다.
  - 가장 깊은 접촉
  - 고체가 C 만큼 키워져 있다는 것
  - 링크와 실제 고체 사이 거리
  - "start from a pose clear of the specimen (e.g. the scan-start pose)"
- `acquisition.py` (심이 수평에서 90°) 거부 메시지에 다음을 더했다.
  - 두 점의 수평 거리와 수직 낙차
  - "pose_a and pose_b have the same XY: check the taught end point"

## 12. 양방향 출력 (e1cf40f)

**왜**: 한 번 센싱으로 어느 방향이든 용접할 수 있어야 한다. 패스 파일의 시작·목표만 바꾸면 안 된다. 자세가 공간에 고정돼 있어 끌기가 밀기가 된다.

**무엇을·어떻게**:
- 기존 계약은 그대로다. job 의 `weld_plan.direction` 계획이 주 계획이다.
  - 최상위 `weld_plan.*`, `kiro/pass_<k>.yaml`, 액션 결과 paths, `precision_sensing.npz` 배열은 바이트 단위로 같다.
- 매 실행 다음을 더 쓴다.
  ```
  directions/S_to_E/{weld_plan.json, weld_plan.npz, kiro/...}   모든 패스 S→E
  directions/E_to_S/{weld_plan.json, weld_plan.npz, kiro/...}   모든 패스 E→S
  verification/directions/<set>/...                              D14 를 통과하지 못한 세트 (이유와 함께)
  ```
- 구현:
  - `scan_core/g1_outputs.py`: `direction_sets` 가 같은 측정 단면에서 방향만 바꿔 `plan_weld` 로 만든다.
    - 주 계획과 같은 세트는 같은 객체를 쓴다. 다시 계산하지 않는다.
  - `scan_core/kiro_plan.py`: `write_kiro_outputs` 가 주 세트 다음에 각 세트를 쓴다.
    - 주 README 에 두 세트와 고르는 법을 적는다. 고른 세트의 방향으로 용접하고, 패스마다 다른 세트를 써도 된다.
  - `scan_l0/teach_1g.py`: 주 계획이 D14 를 통과한 뒤, 주 계획과 다른 세트마다 D14 와 자세 탐색을 한 번씩 한다.
    - 통과하지 못한 세트는 verification 으로 가고, 이유를 실행 기록에 남긴다.
    - 실행 결과 판정은 주 계획만으로 한다.
    - `hongik/under` 는 모든 D14 동안 장면에 있다.
  - `hongik_scan_ros/kiro_io.py`: npz 메타에 세트별 파일·자세, 또는 미납 이유를 남긴다.
  - KIRO 오프라인 추정기 (`write_outputs` + `write_kiro_outputs`) 도 코드 변경 없이 두 세트를 얻는다.

**확인**:
- 현장 스캔 160018 (KIRO 오프라인 파이프라인):

  | 패스 | S→E 진행/작업각 | E→S 진행/작업각 | WPS |
  |---|---|---|---|
  | 1 루트 | −41.9 / +9.0 | −42.1 / +9.0 | −42 / +9 |
  | 2 충전 | −40.9 / −20.0 | −41.1 / −20.0 | −41 / −20 |
  | 3 충전 | −43.9 / +7.0 | −44.1 / +7.0 | −44 / +7 |
  | 4 캡 | −43.9 / +7.0 | −44.1 / +7.0 | −44 / +7 |

  - 롤은 두 방향 모두 180° 이고, 몸통은 각자의 진행 방향 앞으로 눕는다.
  - 와이어 끝 직선은 같고 (199.3 mm), 시작·목표는 반대다.
  - `S_to_E` 는 주 계획과 바이트 단위로 같다.
- 시험 (모두 통과):
  - `test_weld_directions.py` (6): 기하, 형식, 주 계획 불변, E->S·alternate
  - `test_weld_direction_sets_server.py` (4): 세트마다 D14 한 번, 실패 세트 이동, 판정 불변
  - `test_kiro_io_directions.py` (1)
- 비용: D14 시간이 주 계획 외 세트 수만큼 늘어난다. job 방향이 S->E 면 한 번 더, alternate 면 두 번 더다.

---

## 13. 센싱 자세 선택: 관절 최소 → TCP (토치 팁) 최소 이동 (f0774b6)

**왜**:
- 현장 21:44 실행 (214451): 획득 자세의 두 공구 방향 (진행 부호 ±1, 판 법선 둘레 yaw 180° 차이) 이 모두 가능했다.
  - 진행 −1: 관절 이동 시간 7.3 s (관절 합 156°), 토치 회전 115°, 토치 팁 이동 226 mm.
  - 진행 +1: 관절 합 288°, 토치 회전 75°, 토치 팁 이동 418 mm.
- 6절 (B1) 의 선택은 두 방향을 한 무리로 놓고 관절 이동 시간 최소를 골랐다. 그래서 −1 이 선택됐다.
  - 시작 자세에서 토치는 플랜지 축 기준으로 로봇 베이스 쪽 (+122 mm) 에 있었다.
  - −1 은 토치를 115° 돌려 베이스 반대쪽 (−238 mm) 으로 보낸다.
  - 용접기·와이어 송급장치는 로봇 베이스에 있으므로 와이어가 베이스에서 반대쪽까지 돌아간다.
- KIRO: 관절 최소 이동은 정상 동작이지만, 직관적인 이동은 TCP 최소 이동이다. 제약 (예: 케이블 쪽 금지 규칙) 을 추가하지 않고 선택 기준을 바꾼다 (사용자 결정).
- TCP 는 토치 팁이다 (사용자). 중요한 것은 토치와 와이어가 돌아가지 않는 것이다.
  - 토치 팁 거리만으로는 −1 (226 mm < 418 mm) 이 다시 뽑힌다.
  - 시작 자세의 플랜지 축이 기울어 있어서, 돌아간 토치의 팁이 오히려 가까이 온다.
  - 그래서 TCP 틀의 회전을 먼저 본다.

**무엇을·어떻게** (`scan_l0/scan_l0/autoteach.py`):
- `_tcp_in_flange`: TCP = 이 팔의 용접 토치 팁 (`weld_torch.tcp_frame`, KIRO: `right_manipulator_ee_point`, 로봇 모형의 플랜지 ← TCP). 토치가 이 팔에 없으면 센서 광학 원점.
- `_tcp_motion(T_from, T_to)` → (토치 팁 직선 거리, TCP 틀 회전각).
- `_plan_acquisition`: 공구 방향 (진행 부호) 을 다음 순서로 고른다.
  1. 지금 자세에서 TCP 틀 회전이 작은 방향
  2. 그다음 토치 팁 거리
  - 그 방향의 IK 해 가운데 팔 형태는 이전처럼 관절 이동 시간 최소 (`_select_joint`, ψ_b 띠 선호 포함) 로 정한다.
  - 그 방향에 쓸 수 있는 경로가 없을 때만 다른 방향으로 간다.
- `_choose_visit_order` (어느 끝부터): 다음 순서로 고른다.
  1. 회전이 작은 공구 방향 (두 순서 모두 같은 두 방향을 가진다)
  2. 토치 팁 거리
  3. 관절 이동 시간
  - 가지치기: 지금 순서가 회전이 작은 방향이고, 다른 순서가 같은 방향에서 토치 팁 이동이 더 길면 계획하지 않는다 (자세만, 정확).
- 이전의 관절 시간 하한 가지치기 (`_acquisition_duration_bound`) 는 `_acquisition_motions` 로 바뀌었다.
- 기록: `visit_order.options[*].tcp_move_mm`·`tool_rot_deg`, `visit_order.tcp_move_mm`·`tcp`, `order_options[*].tcp_move_mm`·`tool_rotation_deg`.
- 설정·상수 추가 없음. 이후 모든 패스는 이전과 같이 획득 때의 물리 공구 자세를 유지한다. 그래서 고른 토치 쪽이 끝까지 간다.

**현장 영향**:
- 시작 자세에 따라 토치 쪽이 정해진다.
  - 21:44 시작 자세라면 +1 (토치 75° 회전, 베이스 쪽 유지) 이 선택된다.
  - 토치를 베이스 쪽에 두고 시작하면 그 쪽이 유지된다.
- 연구실 10-14 01:49 실행 (회전 54° vs 174°) 은 이전과 같은 −1 이다.

---

## 14. 확인한 관절 계획의 시작 허용값 = move_group 의 것 (KIRO 현장 2026-10-15, 34c5e37)

**왜**: 실행 002753 에서 획득 이동이 "checked joint plan start changed: max joint 0.0542 deg, flange 0.141 mm (> 0.05 deg / 0.1 mm)" 로
거부됐다. 0.05° / 0.1 mm 는 원래 '미리 만든 계획을 다시 쓸지' 정하는 기준 (넘으면 다시 계획) 이었는데, 6절에서 KIRO 의 '확인한 계획을
그대로 한 번' 방식을 옮기면서 실행 거부 기준이 됐다. 팔이 멈춰 있는 동안의 관절 차이 (0.054°) 만으로도 거부된다.

**무엇을·어떻게** (`hongik_scan_ros/moveit_adapter.py`):
- `MoveItPlatform.allowed_start_tolerance()`: move_group 노드의 `trajectory_execution.allowed_start_tolerance` 를 한 번 읽는다
  (계획 서비스와 같은 이름공간의 `/move_group/get_parameters`). move_group 이 자기가 실행하는 모든 궤적에 쓰는 규칙이다.
  읽지 못하거나 0 이면 MoveIt 문서의 기본값 0.01 rad (0.57°) 를 쓰고 그 사실을 문구에 남긴다.
- `_validate_checked_joint_start`: 관절 최대 차이가 그 허용값 이내면 확인한 궤적을 그대로 보낸다. 플랜지 차이는 기록만 한다.
- 다시 확인하는 동안 우리 것이 아닌 관절이 움직였는지도 같은 허용값으로 본다.
- 보내기 직전 전체 궤적의 충돌 재검사 (현재 전체 상태) 는 그대로다.
- 시험: `test_moveit_checked_joint` 에 0.054° 는 보내고 1.15° 는 거부하는 경우를 더했다 (필수 시험만 돌림).

---

## 15. 용접 후 재스캔: 용접 자세에서 시작, TCP 최소 이동, 방향 지정 (KIRO 현장 2026-10-15, 0fa4d35)

**왜**:
- 재스캔은 용접 직후에 부른다. 그래서 토치가 그루브에 있는 용접 자세에서 시작한다. 이 자세는 우리 세계의 시편 상자 (여유 C 50 mm) 안이라 시작 검사에서 거부됐다.
- 다른 실행에서는 시편 근처라 짧은 관절 경로가 막혀 (eoat–hongik/specimen 접촉), 플랜지 331°·손목4 204° 를 도는 후보가 선택됐다. 사용자가 중지했다.
- 방향 (어느 끝부터) 은 옛 '관절 이동 합 최소' 규칙으로, 공구 자세는 '센싱 패스와 같은 물리 자세' 로 강제됐다. E→S 로 용접한 뒤에는 토치를 거의 180° 돌려야 했다.

**무엇을·어떻게**:
- `teach_1g._apply_world(inside_ok=)`: 시작 자세가 우리 물체 (hongik/*) 안일 때만 거부하지 않고 기록한다. 다른 물체·자기 충돌은 그대로 거부한다. 재스캔만 이 옵션을 쓴다.
- `post_weld.PostWeldScan._lift_clear`: 판 법선 방향으로 곧게 올라간다 (공구 자세 그대로).
  - 거리는 도착 자세가 포락선의 판 거리 규칙을 만족하는 최소값이다: 툴 가장 낮은 점 ≥ safe_clearance, 광학 원점 ≥ standoff_min.
  - 그루브는 위로 열려 있으므로, 이 직선에서 모든 툴 점이 판에서 멀어진다.
  - 도착 자세는 포락선 전체를 통과해야 한다.
  - MoveIt 은 이 한 직선만 우리 물체를 장면에서 빼고 검사한다 (로봇 자신·KIRO 고정 구조물은 그대로). 끝나면 다시 넣고, 팔이 물체 밖인지 다시 확인한다.
- `post_weld._plan`: (방향 2개 × 진행 부호 2개) 가운데 지금 자세에서 TCP 최소 이동 (13절과 같은 규칙) 으로 고른다.
  - 먼저 토치 쪽 (물리 공구 yaw) 가운데 회전이 작은 쪽을 고른다. 두 방향 모두 같은 두 쪽을 가지며, 방향에 따라 피치 기울기만 뒤집힌다.
  - 그 안에서 토치 팁 거리 (어느 끝부터) 로 고른다. 피치 몇 도 차이로 먼 끝부터 시작하는 일이 없다.
  - 팔 형태는 그 자세 안에서 관절 이동 시간으로 정한다.
- 공구 방향 (사용자 2026-10-15): 기본 = 센싱 패스와 같은 공구 방향 (`tool_orientation: pass`) — 센싱 패스는 진행 방향이 바뀌어도 진행 부호를 함께 뒤집어 물리 공구 자세 (플랜지 축·피치 −5°) 를 같게 두므로, 재스캔도 S→E·E→S 어느 쪽이든 패스와 같은 시점이다. 이때 방향은 가까운 끝 (토치 팁 거리). `least_rotation` = 두 토치 쪽 가운데 지금 자세에 가까운 쪽 (용접 뒤 케이블 쪽 유지가 필요할 때). 액션 필드 `tool_orientation`, 클라이언트 `--tool-orientation pass|least-rotation`.
- 방향 지정: `PostWeldScan.action` 에 `direction` ('S->E' | 'E->S' | '' = TCP 최소 이동), 클라이언트 `--direction S->E|E->S`, 서버 입력 검사.
  - 새 액션 필드라 `hongik_scan_msgs` 를 다시 빌드해야 한다 (워크스페이스 전체 빌드에 포함된다).
- 재스캔 선 자체는 그대로다: 실측 심 [S − 20, E + 20] mm, 그 실행 패스의 standoff·피치·속도, 원자료만 저장.
- 플랫폼 능력 검사 (`_universal_ready`: 관절 계획·충돌 장면) 를 세션 시작 (모션 백엔드 생성) 뒤로 옮겼다. 6절에서 이 검사가 백엔드 함수까지 보게 되었는데, 재스캔은 백엔드를 만들기 전에 검사해서, 서버를 새로 띄운 직후 첫 동작이 재스캔이면 거부됐다 (현장 2026-10-15).
- 실행 폴더 안쪽 폴더 (예: `directions/E_to_S/kiro`) 를 `--run` 으로 주면, 상위 실행 폴더와 `--direction` 사용법을 알려 주며 거부한다.
- 시험 (필수만): `test_sim_post_weld` 에 용접 직후 자세 (토치 팁이 판 위 약 30 mm) 에서 E→S 지정 재스캔 → 올라가기·접근·하강·스캔 OK 를 추가했다. `test_post_weld_units` 와 `test_post_weld_action` (메시지 새로 빌드) 도 통과했다.

---

## KIRO 쪽에서 바뀌는 것

1. **서버는 우리 런치 (`hongik_moveit.launch.py`) 로 띄운다.**
   - KIRO 의 런타임 패치 래퍼 (`construct_robot.nodes.hongik_auto_teach_server`, 메모리 패치·SHA 고정·numpy 로더) 는 더 필요 없다.
   - KIRO 패치의 좋은 부분은 6절에 이미 옮겼다. numpy 는 5절이 맡는다.
   - 런치 인자 (platform_config, calib_file, job_file, out_root, kiro_confirm_mode 등) 는 그대로다.
2. **적용 방법**: `kiro_field/2026-10-14/patches/kiro-1g-field_vs_ec00145.patch` 한 개를 쓴다.
   - KIRO 사본의 `scan_l0/scan_l0/autoteach.py` 는 현장에서 직접 고친 상태다. 먼저 납품 상태로 되돌린다:
     `git -C <hongik_fine_sensing> checkout -- scan_l0/scan_l0/autoteach.py` (현장 사본의 git 기준 d184b97)
   - 그다음 `hongik_fine_sensing/` 안에서 `git apply -p2 kiro-1g-field_vs_ec00145.patch` 를 실행한다.
   - 납품본 ec00145 와, autoteach.py 를 되돌린 현장 사본 모두에 적용됨을 확인했다.
   - 이후 다시 빌드한다 (`colcon build`).
3. **교시 점**: 직선 스캔용 센서 자세와 우리 액션의 S/E 심 점 (와이어 끝을 심 위에) 을 따로 둔다.
   - 지금 `scan_end.yaml` (16:56) 은 심을 가로지른다.
4. **방향**: job `weld_plan.direction` 이 주 방향이다. 다른 방향은 `directions/` 에서 고른다. 파일의 시작·목표를 맞바꾸지 않는다.

## 시험

- 필수 시험 (`run_tests.py`, 격리 ROS 도메인, 가짜 장치) 14개 대상이 모두 통과했다.
  - 대상: `test_joint_selection`, `test_moveit_checked_joint`, `test_acquisition`, `test_autoteach_units`, `test_service_calls_and_shutdown`,
    `test_weld_direction_sets_server`, `test_weld_directions`, `test_kiro_io`, `test_kiro_plan`, `test_cell_world`, `test_g1_worker`,
    `test_undefined_names`, `test_package_scope`, `test_sim_teach1g::test_i_dry_run_plans_only`
  - 여기에 갈래별 시험 (`test_g1_outputs`, `test_weld_orient` 등) 을 더했다.
- 가짜 move_group 종단 시험 (`dev/kiro_e2e`, localhost·가짜 하드웨어, 패키지를 깨끗이 빌드한 설치본으로 실행):
  - 첫 실행: 13개 중 9개 통과, 4개 실패.
    - 3개는 출력 폴더 기대 목록에 새 `directions/` 가 없어서 실패했다.
    - 1개는 장면 기대 목록에 `hongik/under` 가 없어서 실패했다.
    - 모두 기대 목록을 갱신했다.
  - 다시 돌린 결과:
    - `test_m` (1G 전체 실행), `test_m3` (용접 후 스캔), `test_n` (정밀 센싱 액션), `test_n2` (운영 클라이언트) 모두 통과.
    - `test_m` 은 실행 기록의 첫 세계 판에 고정 몸체가 빠져 한 번 더 실패했다. 도달 상한이 세계 판 번호를 올리던 것을 고친 뒤 (0eea328) 통과했다.
  - 이 시험에서 확인된 것:
    - 정밀 센싱 액션 전체 흐름 (가짜 팔): 획득 → 센싱 → 계획 → D14 → 양방향 출력.
    - 시간 기준 관절 선택과 확인 궤적 전송.
    - 서비스 호출과 종료 처리.
- 13절 (TCP 최소 이동) 은 필수 시험만 돌렸다 (사용자 2026-10-14: 무거운 시험 제외).
  - 통과: `test_sim_teach1g::test_i_dry_run_plans_only` (가상 팔에서 획득·방문 순서 계획), `test_joint_selection`, `test_autoteach_units`,
    `test_autoteach_round4_units`, `test_autoteach_serpentine_units`, `test_post_weld_units`, `test_l0_config_plan`.
  - `test_seam_end_identification::test_sporadic_ghost_points_beyond_a_real_end_do_not_make_it_occluded` 1개 실패 (길이 오차 1.43 mm > 1.0).
    합성 패스 맞춤 (`fit_pass`) 시험이라 획득 계획 코드를 지나지 않는다.
  - 가상 팔 종단 (`test_sim_teach1g` 전체)·가짜 move_group 종단은 돌리지 않았다.
- 이 환경에서 원래부터 실패하는 시험은 이번 변경과 무관하다.
  - `test_seam_estimate` 시간 시험 2개
  - `test_post_weld_action::test_cli_arguments_and_goal` (연구실 메시지 빌드)

