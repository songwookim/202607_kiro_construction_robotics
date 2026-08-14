# Construct dual-arm 시스템 구성 및 안전 검증 가이드

이 문서는 다음 목표를 기준으로 현재 시스템을 설명한다.

> 왼팔과 오른팔을 함께 실제 제어기에 연결하고, RViz의 `dual_arm` 그룹에서 계획한 뒤 두 실제 로봇이 함께 실행한다.

처음 보는 사람도 장애가 어느 계층에서 발생했는지 구분할 수 있도록 연결, 계획, 실행을 별개의 단계로 설명한다. 파일과 수치는 2026-08-05 현재 소스 기준이다.

## 1. 지금 확인된 범위

이번 점검에서는 사용자가 자리에 없었으므로 실제 로봇을 움직이지 않았다.

| 항목 | 결과 |
| --- | --- |
| 왼팔 REAL 연결 | `192.168.1.11`, 명령/상태 TCP 성공 |
| 오른팔 REAL 연결 | `192.168.1.12`, 명령/상태 TCP 성공 |
| 양팔 실제 feedback | 양쪽 `system_state`의 지속 수신 확인 |
| steady feedback | 5.09초 동안 왼팔 50회, 오른팔 47회 수신 |
| command socket backlog | 양팔 TCP 5000 `Recv-Q=0` |
| ros2_control 초기 상태 | 양팔 모두 measured `jnt_ang`으로 동기화 확인 |
| RViz Goal State | 양팔 READY 뒤 현재 PlanningScene으로 1회 갱신 |
| 현재 controller | `joint_state_broadcaster`, 양팔 trajectory controller, head controller 모두 active |
| `/move_action` | server는 `move_group`, client는 RViz와 `weld_action_gui`에서 확인 |
| 현재 양팔 상태의 MoveIt 유효성 | `valid=true` |
| 작은 양팔 목표 | 실측 상태 기준 왼팔 `+0.01 rad`, 오른팔 `-0.01 rad` |
| MoveIt plan-only | 성공, error code `1`, planning time `0.323836 s` |
| 생성 trajectory | arm 12 joints만 포함, head joint 없음 |
| RB safety feedback | 양팔 collision/self-collision/soft-estop/EMS/SOS 모두 false/0 |
| FAKE 양팔 Plan+Execute | 두 arm controller 모두 goal reached, error code `1`, 최종 최대 오차 `0.004908 rad` |
| 실제 양팔 Execute | **미수행·미검증** |

따라서 현재 결론은 “양팔 연결, 계획, 양팔 controller 활성화까지 정상”이다. 실제 양팔 동시 이동과 펜던트 safety 동작은 작업자가 로봇 앞에서 비상 정지와 저속 조건을 준비한 뒤 별도로 검증해야 한다.

별도 ROS domain에서 양팔을 모두 `GenericSystem`으로 둔 실행 모드도 확인했다. MoveGroup은 arm 12축 trajectory를 왼팔·오른팔 `FollowJointTrajectory` controller에 각각 전달했고 두 controller 모두 `Goal reached, success`로 끝났다. 이는 ROS/MoveIt/controller 분배 경로가 맞다는 검증이지, 실제 RB 제어기와 펜던트 safety 검증을 대신하지는 않는다.

## 2. 실행 방법

### 2.1 기본 실행 모드

사용자용 launch는 시작 즉시 양팔 REAL 연결을 시도하며 기본적으로 `execute_motion:=true`이다. 별도 Connect 단계 없이 양팔 trajectory controller까지 활성화된다.

사용자 launch 아래의 ros2_control, MoveIt, RViz와 GUI는 모두 Fast DDS
`UDPv4` transport와 `ROS_LOCALHOST_ONLY=1`을 사용한다. SHM lock 문제와
유선·Wi-Fi NIC가 동시에 선택되어 로컬 ROS graph가 갈라지는 현상을 피하기
위한 설정이다. RB 명령/상태와 Hi-COMM은 ROS DDS가 아닌 별도 TCP이므로 이
설정의 영향을 받지 않는다.

```bash
ros2 launch construct_robot weld_action_gui.launch.py
```

### 2.2 진단 목적으로 연결과 Plan만 검사할 때

```bash
cd /home/irs/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch construct_robot weld_action_gui.launch.py execute_motion:=false
```

`execute_motion:=false`는 통신 또는 planning만 분리해서 진단할 때 명시적으로 사용하는 선택 모드다.

`execute_motion:=false`이면 세 겹으로 실행을 막는다.

1. `move_group`의 `allow_trajectory_execution`이 `false`이다.
2. `joint_state_broadcaster`만 활성화하고 왼팔·오른팔 trajectory controller는 로드/활성화하지 않는다.
3. 사용자 Cartesian action server도 실행 요청을 거부한다.

따라서 실제 하드웨어에 연결해 관절 상태를 읽고 Plan을 할 수 있지만, `move_servo_j`를 스트리밍하는 arm controller는 없다.

주의할 점은 “연결” 자체가 완전한 무명령 동작은 아니라는 것이다. RBPodo 드라이버 생성 과정에서 제어기에 operation mode와 speed bar 설정 명령을 보낸다. 다만 plan-only 모드에서는 관절 궤적 명령을 보내지 않는다.

### 2.3 실제 Execute를 검사할 때

작업자가 로봇 앞에 있고 두 펜던트의 안전 상태, TCP/tool/payload, 비상 정지, 저속 설정을 확인한 경우에만 다음처럼 실행한다.

```bash
ros2 launch construct_robot weld_action_gui.launch.py execute_motion:=true
```

처음에는 현재 자세에서 매우 작은 목표, 낮은 MoveIt velocity/acceleration scaling, 한 팔씩의 dry run, 양팔 순서로 확인한다. 이번 점검에서는 이 명령으로 실제 Execute를 수행하지 않았다.

## 3. 연결, Plan, Execute는 서로 다른 성공 조건이다

“연결됨”이라는 말을 하나로 사용하면 원인 분석이 어려워진다.

| 단계 | 성공의 의미 | 대표 확인 대상 |
| --- | --- | --- |
| RB 통신 | 명령 TCP와 상태 TCP가 열리고 실제 상태를 받음 | `/left_rbpodo_hardware/system_state`, `/right_rbpodo_hardware/system_state` |
| ROS 제어 준비 | 필요한 controller가 준비됨 | `/controller_manager`, `ros2 control list_controllers` |
| MoveIt 준비 | 로봇 모델과 현재 상태를 받고 `/move_action`이 준비됨 | `move_group`, `/move_action` |
| Plan | joint limit와 collision 검사를 통과한 trajectory가 생성됨 | MoveIt error code `1` |
| Execute | trajectory가 두 arm controller를 거쳐 두 제어기에서 수용·수행됨 | 두 `FollowJointTrajectory` action과 실제 피드백 |

Plan 성공은 실제 Execute 성공을 보장하지 않는다. Plan은 MoveIt의 모델 안에서 끝날 수 있지만, Execute에는 controller 상태, 네트워크, RB 제어기 상태, 펜던트 안전 모델이 추가로 관여한다.

## 4. 전체 구조

사용자가 실행하는 진입점은 [weld_action_gui.launch.py](../construct_robot/launch/weld_action_gui.launch.py)이다.

```text
weld_action_gui.launch.py
├─ weld_action_gui               (LEFT/RIGHT O/X 표시)
├─ moveit.launch.py
│  ├─ rviz2
│  ├─ ros2_control_node / controller_manager
│  ├─ controller_spawner
│  ├─ move_group              (spawner 완료 후 시작)
│  ├─ robot_state_publisher
│  └─ static TF: World→world, World→link0
└─ cartesian_path_action_server
```

기본값은 왼팔 `.11`, 오른팔 `.12`의 실제 hardware이며 connection supervisor나
GUI Connect/Disconnect 단계는 없다. 진단 시에는 각
`use_fake_*_hardware` launch 인자로 mock hardware를 명시할 수 있고, 종료할 때
전체 launch가 함께 종료된다. 설정된 initial pose는 자동 실행하지 않는다.

ROS 2 Humble의 MoveIt RViz에 timestamp가 있는 custom RobotState를 직접
주입했을 때 다음 crash가 재현된 적이 있다.

```text
can't compare times with different time sources
```

따라서 custom RobotState를 만들지 않는다. 양팔과 MoveGroup이 준비되면 GUI가
RViz 표준 `/rviz/moveit/update_goal_state` Empty 이벤트를 한 번 보내고, RViz가
자신의 현재 PlanningScene을 주황색 Goal State로 복사한다.

관련 launch 파일은 다음과 같다.

- [weld_action_gui.launch.py](../construct_robot/launch/weld_action_gui.launch.py): GUI, Cartesian server와 MoveIt launch를 함께 시작한다.
- [moveit.launch.py](../construct_moveit_config/launch/moveit.launch.py): RViz, 로봇 모델, controller manager, controller spawner, MoveGroup을 시작한다.

## 5. 시작 및 동기화 순서

1. 왼팔 `192.168.1.11`, 오른팔 `192.168.1.12` hardware를 함께 생성한다.
2. 각 arm이 명령/상태 TCP를 열고 measured `jnt_ang`을 읽는다.
3. hardware activation이 position state와 첫 command를 측정값으로 맞춘다.
4. controller spawner가 controller manager 응답을 최대 90초 기다린다.
5. spawner 완료 뒤 MoveGroup을 시작한다.
6. GUI는 0.5초마다 양팔 feedback, MoveGroup, controller 준비를 확인해 O/X로 표시한다.
7. 양팔이 모두 O가 되고 RViz subscriber가 준비되면 Goal State 갱신 이벤트를 한 번 발행한다.

GUI의 `READY` 조건은 다음과 같다.

- 해당 arm의 6개 측정 관절이 `/joint_states`에 모두 있고 마지막 메시지가 5초보다 오래되지 않음
- MoveGroup의 `/move_action` action server가 준비됨
- `execute_motion:=true`이면 해당 arm의 `FollowJointTrajectory` action server도 준비됨

GUI의 초기 진단 한도는 90초지만 이후에도 feedback 감시는 계속된다. RB 피드백만
도착하고 MoveGroup/controller가 준비되지 않은 중간 상태는 X로 표시한다.

### `connection lost`의 정확한 의미

현재 GUI에서 `connection lost`는 해당 arm의 완전한 측정 `/joint_states`가 5초 넘게 갱신되지 않았거나 MoveGroup/controller action이 준비되지 않았다는 뜻이다. 명령 TCP 5000과 데이터 TCP 5001 중 어느 것이 끊겼는지까지 직접 증명하는 메시지는 아니다. 다음 가능성을 함께 확인해야 한다.

- 실제 데이터 TCP 단절 또는 제어기 응답 중단
- hardware read loop 정지
- controller manager 정지/재시작
- DDS topic 전달 지연 또는 process 종료

시작 실패를 볼 때는 controller manager와 각 RBPodo hardware의 최초 오류를 먼저 확인한다.

## 6. RViz Plan과 Execute 데이터 흐름

### Plan

```text
RViz MotionPlanning
  → /move_action
  → move_group
  → 현재 /joint_states + URDF/SRDF + joint limits
  → IK / collision 검사 / planning pipeline
  → planned trajectory를 RViz에 표시
```

Plan 단계에는 Hi-COMM 용접 명령도 RBPodo 관절 명령도 필요하지 않다. 실제 로봇에 연결한 경우 `/joint_states`는 실제 측정 각도에서 온다.

### Execute

```text
RViz Plan and Execute
  → move_group trajectory execution
  ├─ /left_manipulator_controller/follow_joint_trajectory
  └─ /right_manipulator_controller/follow_joint_trajectory
       → joint_trajectory_controller
       → ros2_control position command interface (100 Hz)
       → 각 RBPodoHardwareInterface
       → move_servo_j([degree...], t1, t2, gain, alpha)
       → 각 RB Control Box
```

피드백은 반대 방향으로 흐른다.

```text
RB Control Box의 상태 TCP 5001
  → RBPodo data reader
  → measured jnt_ang
  → ros2_control state interface
  → joint_state_broadcaster
  → /joint_states
  → move_group / robot_state_publisher / RViz
```

실행 모드에서 실제로 값이 변하는 동안에는 `move_servo_j`가 주기적으로 전달되므로 펜던트가 외부 명령 상태를 주황색 등으로 표시할 수 있다. 현재 driver patch는 로봇이 Idle이고 마지막 명령과 완전히 같은 hold 값뿐이면 불필요하게 `move_servo_j`를 다시 시작하지 않도록 한다. Plan-only 모드에는 arm trajectory controller가 없으므로 ros2_control이 관절 이동 명령을 스트리밍하지 않는다.

현재 hardware `write()`에는 RB의 collision, self-collision, estop 같은 safety flag를 보고 명령을 자동 차단하는 로직이 없다. 이 flag들은 `system_state`로 관측할 수 있지만 실제 명령 허용/차단의 최종 책임은 RB 제어기와 펜던트에 있다.

## 7. MoveIt planning group

정의는 [construct_robot_0528.srdf](../construct_moveit_config/config/construct_robot_0528.srdf)에 있다.

| 그룹 | 포함 관절 | 용도 |
| --- | --- | --- |
| `left_manipulator` | 왼팔 6축 | 왼팔 단독 계획 |
| `right_manipulator` | 오른팔 6축 | 오른팔 단독 계획 |
| `dual_arm` | 왼팔 6축 + 오른팔 6축 | 양팔 동시 계획, 총 12축 |
| `robot_head` | 머리 2축 | 머리 단독 계획 |
| `whole_robot` | `dual_arm` + `robot_head` | 양팔과 머리를 함께 계획, 총 14축 |

RViz의 기본 Planning Group은 `dual_arm`이다. 머리는 실제 RB arm controller와 무관한 `GenericSystem`이며, 양팔 Execute에 끼어들지 않도록 `dual_arm`에서 분리했다.

## 8. ros2_control 구성

하드웨어 정의는 [construct_robot_0528.ros2_control.xacro](../construct_description/urdf_0528/construct_robot_0528.ros2_control.xacro), controller 설정은 [ros2_controllers.yaml](../construct_moveit_config/config/ros2_controllers.yaml)에 있다.

### Hardware component

| component | plugin | namespace |
| --- | --- | --- |
| `LeftRBPodoHardwareInterface` | `rbpodo_hardware/RBPodoHardwareInterface` | `left` |
| `RightRBPodoHardwareInterface` | `rbpodo_hardware/RBPodoHardwareInterface` | `right` |
| `HeadMoveItSystem` | `mock_components/GenericSystem` | 없음 |

각 arm은 position command와 position state interface를 내보낸다. 실제 피드백은 RB의 `jnt_ang`을 degree에서 ROS 표준인 radian으로 변환한다. 실제 명령은 반대로 radian에서 degree로 변환한다.

실제 hardware 활성화 때 position command의 초기값도 측정된 `jnt_ang`으로 맞춘다. controller가 켜지는 순간 0 또는 오래된 reference 자세를 첫 명령으로 보내 로봇이 원래 자세로 되돌아가는 현상을 막기 위한 것이다.

### Controller

| controller | 역할 | plan-only | execute 모드 |
| --- | --- | --- | --- |
| `joint_state_broadcaster` | 모든 관절 상태를 `/joint_states`로 발행 | active | active |
| `left_manipulator_controller` | 왼팔 6축 FollowJointTrajectory | 로드/활성화 안 함 | active |
| `right_manipulator_controller` | 오른팔 6축 FollowJointTrajectory | 로드/활성화 안 함 | active |
| `robot_head_controller` | 가상 머리 2축 | 로드/활성화 안 함 | active |

arm controller는 `open_loop_control: false`라서 실제 state feedback을 사용한다. partial joint goal은 허용하지 않는다. trajectory 오차 허용값은 관절당 0.15 rad, 최종 goal 오차는 0.02 rad, goal time tolerance는 2초다.

MoveIt이 controller action 이름과 관절을 찾는 설정은 [moveit_controllers.yaml](../construct_moveit_config/config/moveit_controllers.yaml)에 있다.

시작 시간을 줄이기 위해 plan-only는 OMPL만 로드하고 [plan_only_controllers.yaml](../construct_moveit_config/config/plan_only_controllers.yaml)의 가상 head 항목만 MoveIt에 등록한다. 실제 arm action client는 만들지 않으며 `allow_trajectory_execution=false`다. execute 모드에서만 OMPL·CHOMP·Pilz, 양팔·head controller 설정을 모두 로드한다. 별도 FAKE 검증에서 두 모드 모두 정상 기동했다.

## 9. 주기와 timeout

| 항목 | 현재 값 | 의미 |
| --- | ---: | --- |
| RBPodo 상태 요청 thread | 500 Hz 목표 | 각 arm의 상태 TCP에서 최신 SystemState 갱신 |
| controller manager update loop | 100 Hz | read → controller update → write |
| `joint_state_broadcaster` | 100 Hz loop를 따름 | `/joint_states` 발행 |
| arm controller action monitor | 20 Hz | FollowJointTrajectory action 상태 감시 |
| arm `system_state` 발행 | 10 Hz | 100 Hz read loop의 10회마다 발행 |
| GUI readiness 검사 | 0.5초마다, 2 Hz | feedback/MoveGroup/controller 준비 확인 |
| GUI feedback stale 기준 | 5초 | 이보다 오래되면 `connection lost` |
| GUI 전체 연결 deadline | 90초 | feedback 이후 MoveIt/controller startup 포함 |
| controller service 응답 timeout | 90초 | Humble 기본 spawner의 짧은 load timeout 회피 |
| Hi-COMM cyclic I/O | 40 ms, 25 Hz | 용접기 TX55/RX71 교환 |

RBPodo joint position streaming의 기본 파라미터는 `t1`이 실제 controller period(통상 0.01초, 최소 0.002초), `t2=0.03초`, `gain=0.5`, `alpha=0.5`이다. `t2`는 upstream 기본값을 그대로 유지한다. controller manager의 100 Hz를 무리하게 올리면 명령량과 Linux scheduling jitter가 함께 증가한다. 연결 속도 문제는 제어 주기를 올리기보다 startup timeout과 readiness 순서를 분리해 해결해야 한다.

## 10. 통신 방식과 포트

| 대상 | IP/포트 | 방향과 형식 |
| --- | --- | --- |
| 왼팔 RB 명령 | `192.168.1.11:5000` | PC → RB, newline으로 끝나는 ASCII 명령 |
| 왼팔 RB 상태 | `192.168.1.11:5001` | request/response 형태의 binary `SystemState` |
| 오른팔 RB 명령 | `192.168.1.12:5000` | PC → RB, newline으로 끝나는 ASCII 명령 |
| 오른팔 RB 상태 | `192.168.1.12:5001` | request/response 형태의 binary `SystemState` |
| Hi-COMM 용접기 | 설정된 용접기 IP `:60000` | PC가 TCP client, 용접기가 server |

양팔과 Hi-COMM 통신은 독립적이다. RViz에서 양팔 Plan을 하는 데 용접기 연결은 필요하지 않다.

## 11. 주요 ROS 인터페이스

| 종류 | 이름 | 생산자/소비자 및 역할 |
| --- | --- | --- |
| topic | `/joint_states` | joint state broadcaster → MoveIt/RSP/RViz |
| topic | `/left_rbpodo_hardware/system_state` | 왼팔 실제 상태와 안전/IO 정보 |
| topic | `/right_rbpodo_hardware/system_state` | 오른팔 실제 상태와 안전/IO 정보 |
| action | `/move_action` | RViz → MoveGroup planning 요청 |
| action | `/execute_trajectory` | MoveIt trajectory 실행 요청 |
| action | `/left_manipulator_controller/follow_joint_trajectory` | MoveIt → 왼팔 controller |
| action | `/right_manipulator_controller/follow_joint_trajectory` | MoveIt → 오른팔 controller |
| action | `/cartesian_path` | weld GUI → 사용자 Cartesian path server |
| service | `/compute_cartesian_path` | 사용자 Cartesian server → MoveGroup |
| service | `/right_rbpodo_hardware/set_digital_output` | weld GUI의 오른팔 control-box DO |
| TCP | Hi-COMM `TX55/RX71` | weld GUI ↔ 디지털 용접기 |

RViz의 일반 Plan/Execute 경로는 `/cartesian_path`와 Hi-COMM을 거치지 않는다. weld GUI의 직선/원호/위빙 기능만 사용자 Cartesian server를 거친다.

## 12. Joint limit과 collision의 위치

Joint limit은 한 곳에만 있지 않다.

1. [construct_robot_0528.urdf](../construct_description/urdf_0528/construct_robot_0528.urdf)의 `<limit>`은 관절 position, velocity, effort의 기본 한계다.
2. [joint_limits.yaml](../construct_moveit_config/config/joint_limits.yaml)은 MoveIt의 velocity/acceleration 제한과 기본 scaling을 덮어쓴다.
3. RViz MotionPlanning panel의 velocity/acceleration scaling은 각 planning 요청에 적용하는 비율이다.
4. RB 펜던트/제어기의 joint, TCP, tool, payload, safety limit은 별도다.

지속적으로 바꿀 limit은 RViz 화면만 조정하지 말고 URDF 또는 `joint_limits.yaml`을 수정한 뒤 rebuild/relaunch해야 한다. 현재 기본 velocity/acceleration scaling은 각각 0.1이다.

현재 `joint_limits.yaml`은 양팔을 같은 보수적 정책으로 맞췄다. 1~3축은 acceleration `0.5 rad/s²`, jerk `2.0 rad/s³`, 4~6축은 acceleration `0.75 rad/s²`, jerk `3.0 rad/s³`이다. 이 값은 우선 저속 검증을 위한 프로젝트 설정이므로, 실제 RB 모델·payload별 사양을 확인한 뒤 더 엄격한 값이 필요하면 낮춰야 한다.

MoveIt collision은 URDF collision mesh와 SRDF Allowed Collision Matrix를 사용한다. 펜던트의 `self collision..!`은 RB 제어기 내부의 robot/tool/TCP/payload/safety 모델을 사용한다. 따라서 MoveIt에서 “valid”여도 펜던트가 거부할 수 있고, 반대도 가능하다.

SRDF에는 CAD collision mesh가 실제 home pose에서 겹쳐 보이는 문제를 피하려고 `left_manipulator_link1`과 `right_manipulator_link2`의 충돌 검사를 비활성화한 항목이 있다. 이는 해당 링크가 물리적으로 항상 안전하다는 증명이 아니므로 실제 clearance를 별도로 확인해야 한다.

## 13. `rbpodo`와 `rbpodo_ros2`를 왜 수정했는가

### `rbpodo`

감사 시점의 기준은 commit `f6ef41a`, tag `v0.16.14`다. 로컬 `HEAD`와 추적 중인 `origin/main`이 같은 commit이고, staged·unstaged·untracked 변경이 전혀 없는 **완전히 clean한 upstream 상태**다. 즉, 양팔 구성을 위해 `rbpodo` core library는 수정하지 않았다.

다만 upstream 구현에는 다음 한계가 있다.

- `Socket::send()`는 한 번의 `send()`가 전체 문자열을 전송한다고 가정한다.
- `move_servo_j()`는 `Socket::send()`의 boolean 결과를 사용하지 않는다.
- 현재 ROS driver는 streaming 중 ACK 대기를 끈다.
- binary 상태 수신도 한 번의 `recv()`가 온전한 frame을 준다는 가정이 남아 있다.

그래서 명령 전송 실패가 즉시 상위 ROS action 오류로 전파되지 않고, 뒤늦은 상태 feedback 단절로 보일 가능성이 있다. 이것은 현재 확인된 실패의 확정 원인이라는 뜻은 아니며, 재현 로그와 packet/return-value 계측이 있어야 판단할 수 있다.

### `rbpodo_ros2`

기준 commit은 `5e8294a`이며, 하나의 controller manager 안에서 RBPodo hardware 두 개를 동시에 쓰기 위해 project patch가 있다.

| 변경 | 필요한 이유 | 성격 |
| --- | --- | --- |
| arm별 `hardware_namespace`와 고유 embedded node 이름 | 두 plugin이 같은 node/topic 이름을 만들지 않게 함 | 양팔에 필수 |
| Cartesian interface resource에 arm prefix | 두 plugin의 `x/y/z/rx/ry/rz` interface 키 중복 방지 | 양팔에 필수 |
| command-mode switch 요청을 각 plugin 소유 joint로 필터링 | controller manager가 전달한 12축 전역 요청을 각 6축 plugin이 잘못 거부하지 않게 함 | 양팔에 필수 |
| 활성화 때 command/state를 measured `jnt_ang`으로 동기화 | controller 시작 직후 0/과거 자세 명령으로 튀는 현상 방지 | 실제 로봇 안전상 필요 |
| `jnt_ref` 대신 measured `jnt_ang`을 state로 사용 | MoveIt과 controller에 실제 관절 피드백 제공 | 실제 피드백에 필요 |
| arm별 `system_state`를 10 Hz로 발행 | 연결 상태와 RB safety/IO를 GUI가 확인 | 현재 UI에 필요 |
| 동일 command socket의 response를 20 ms마다 drain | ACK 대기 없는 streaming 중 응답이 쌓여 command channel이 막히지 않게 함 | 통신 안정성에 필요 |
| Idle에서 동일 hold 명령 재전송 억제 | 같은 자세로 `move_servo_j`를 계속 재시작하지 않게 함 | 프로젝트 제어 정책 |
| control-box digital output service | weld GUI의 DO 제어 | 양팔 MoveIt 자체에는 불필요 |

vendor repository를 완전히 수정하지 않는 대안은 arm마다 controller manager를 따로 띄우고 `/joint_states`를 합치며 action namespace와 lifecycle을 따로 관리하는 것이다. 변경은 줄지만 전체 launch와 동기화 복잡도가 크게 늘어난다. 현재 구조에서는 위의 “양팔에 필수” 부분을 작은 명시적 patch로 유지하거나, 장기적으로 project-local hardware plugin으로 분리하는 편이 현실적이다.

`RobotNode`의 upstream task/eval/controller-config service와 `move_j`, `move_l`, `move_jb2`, `move_pb` action은 그대로 보존했다. 양팔 문제와 관계없는 API 삭제는 되돌려 vendor diff를 줄이고 기존 사용 프로그램과의 호환성을 유지했다. 프로젝트가 추가한 것은 arm별 `system_state`, `SetDigitalOutput`, 고유 node 이름과 command-response drain이다.

upstream 구현이 arm마다 두 번째 command socket과 10 µs busy thread를 만들던 부분은 복구하지 않았다. 실제 servo 명령을 보내는 `Robot`의 기존 command socket을 mutex 아래에서 20 ms마다 nonblocking drain하고, 받은 응답은 기존 `~/response` publisher로 내보낸다. 따라서 저수준 ROS API를 유지하면서도 잘못된 socket의 응답을 읽고 실제 socket의 `Recv-Q`가 쌓이는 문제를 피한다.

또한 `system_state`에 safety flag를 담아 발행하지만, 현재 `RBPodoHardwareInterface::write()`는 이 flag를 근거로 command write를 차단하지 않는다. `t2`도 별도 조정 없이 upstream의 `0.03초`를 유지한다.

## 14. 왜 단순한 목표가 어렵게 보였는가

목표는 단순하지만 하나의 버튼이 다음 경계를 모두 통과한다.

1. 두 제어기의 총 네 TCP channel
2. 단일 controller manager 안의 두 vendor hardware plugin
3. 양팔 joint state와 command interface 소유권
4. controller load/configure/activate 순서
5. MoveGroup startup과 현재 state 수신
6. MoveIt collision/joint limit
7. 두 FollowJointTrajectory action의 실행
8. 각 RB 제어기의 독립적인 safety/self-collision 판단

특히 원래 plugin은 사실상 한 로봇 인스턴스를 전제로 한 이름과 command-mode 전환 로직을 가지고 있었다. 또 hardware 초기화가 끝나 피드백이 보이는 시점과 controller/MoveGroup이 실제 요청을 받을 수 있는 시점 사이에 수십 초 차이가 날 수 있었다. 그래서 “TCP가 연결됐다”를 너무 일찍 전체 연결 성공으로 표시하면 곧바로 Plan 실패나 connection lost처럼 보였다.

현재 구성은 양팔 namespace/interface 충돌을 분리하고, controller spawner 완료 뒤 MoveGroup을 시작하며, GUI가 전체 준비 조건을 기다리도록 역할을 나눴다.

## 15. 장애 확인 순서

새 terminal마다 먼저 workspace를 source한다.

```bash
source /opt/ros/humble/setup.bash
source /home/irs/ros2_ws/install/setup.bash
```

### 15.1 process와 controller

```bash
ros2 node list | sort
ros2 control list_hardware_components
ros2 control list_controllers
```

plan-only에서는 `joint_state_broadcaster`만 active인 것이 정상이다. execute 모드에서는 왼팔·오른팔 trajectory controller가 active여야 한다.

### 15.2 실제 feedback

```bash
ros2 topic hz /left_rbpodo_hardware/system_state
ros2 topic hz /right_rbpodo_hardware/system_state
ros2 topic hz /joint_states
```

각 `system_state`는 약 10 Hz가 기대값이다. 한 arm만 멈추면 해당 arm의 RB/data path를 먼저 보고, 둘 다 동시에 멈추면 controller manager 또는 하위 stack 전체 종료 여부를 먼저 본다.

### 15.3 action 준비

```bash
ros2 action list | sort
```

항상 `/move_action`이 보여야 한다. execute 모드에서는 다음 두 action도 보여야 한다.

```text
/left_manipulator_controller/follow_joint_trajectory
/right_manipulator_controller/follow_joint_trajectory
```

### 15.4 TCP 연결

```bash
ss -tnp | rg '192\.168\.1\.(10|11):(5000|5001)'
```

양팔 REAL 연결이면 두 IP 각각 5000과 5001 연결이 기대된다. 단, TCP가 `ESTAB`인 것만으로 ROS feedback과 MoveIt 준비까지 보장되지는 않는다.

### 15.5 증상별 첫 확인 지점

| 증상 | 먼저 볼 곳 |
| --- | --- |
| `REAL RB feedback confirmed`가 안 나옴 | IP, 5000/5001, 제어기 Idle/operation mode, RBPodo 초기화 로그 |
| feedback은 오지만 GUI가 계속 CONNECTING | controller spawner, `/move_action`, execute 모드의 두 trajectory action |
| 현재 state가 invalid | `/joint_states` 12축 완전성, NaN, URDF/SRDF collision |
| 현재 state는 valid인데 Plan 실패 | goal collision, joint limit, IK, planning group, planner 로그 |
| Plan은 성공하지만 Execute 실패 | controller active 상태, action result, path tolerance, pendant safety 상태 |
| 한 팔만 동작 | 실패한 arm의 action result와 `system_state`, joint 이름/trajectory 포함 여부 |
| 동작 뒤 원래 자세로 복귀 | controller 활성화 직전 measured state 동기화 로그와 첫 command 값 |
| 펜던트 `self collision..!` | RB 내부 TCP/tool/payload/safety 모델과 실제 자세; MoveIt 결과만 믿지 않음 |
| `move_servo_j` parse error | 전송된 실제 한 줄, command socket framing/부분 전송, 두 arm별 최초 발생 시각 |

로그를 볼 때 마지막 traceback보다 **처음 발생한 ERROR/WARN과 그 직전의 arm 이름, controller 상태, system_state 시간**을 보존하는 것이 중요하다.

## 16. 남은 검증과 알려진 제한

- 실제 양팔 동시 Execute는 아직 검증하지 않았다.
- 두 RB 제어기가 MoveIt trajectory를 동시에 수용하는지와 한쪽 fault가 다른 쪽 action에 어떻게 전파되는지 확인해야 한다.
- 펜던트 self-collision 모델과 MoveIt collision model의 일치 여부는 보장되지 않는다.
- 현재 양팔 MoveIt acceleration/jerk limit는 대칭으로 맞췄지만, 실제 RB 모델·payload별 허용값과의 대조는 남아 있다.
- GUI의 `connection lost`는 topic freshness 기반이며 raw socket별 진단은 아니다.
- `system_state` safety flag는 관측용이며 현재 hardware `write()`를 자동 차단하지 않는다.
- ACK를 끈 streaming 구조에서는 제어기의 명령 수용을 즉시 end-to-end 확인하기 어렵다.
- `rbpodo_ros2`의 upstream task/move API는 보존했지만, 이번 양팔 MoveIt 경로에서는 별도로 검증하지 않았다.
- 실제 첫 Execute 전에는 각 arm의 첫 `move_servo_j` 목표가 측정 자세와 가까운지 로그로 확인하는 절차가 필요하다.

현재 안전한 완료선은 “시작 → 양팔 O → 현재/작은 목표 valid → Plan success”다. 다음 완료선인 “두 실제 arm의 동시 Execute success”는 작업자가 현장에 있을 때만 진행한다.
