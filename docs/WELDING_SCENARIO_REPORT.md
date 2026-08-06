# 듀얼암 용접 모션 및 Welding Scenario 설계 보고서

## 1. 목적

본 시스템은 RViz에서 실제 로봇 상태를 확인하고 TCP1에서 TCP2까지의
Cartesian 직선 용접 경로를 계획한 뒤, 승인된 동일 trajectory에 대해서만
H600 ARC 시퀀스와 로봇 동작을 수행한다. 용접 종료 후에는 작업 시작 전에
저장한 6축 joint 자세로 복귀한다.

일반 경로 생성과 모션 검증은 `weld_action_gui`, 실제 용접 절차는
`welding_scenario_gui`로 분리한다. 따라서 테스트용 GUI에서 용접 출력이
실수로 노출되지 않고, 용접 GUI는 정해진 운전 순서를 따라가게 된다.

## 2. 전체 구성

```text
Rainbow robot LEFT/RIGHT
        ^          |
        |          | measured joint/system state
        |          v
  rbpodo_ros2 hardware interface
        ^          |
        |          v
joint_trajectory_controller + joint_state_broadcaster
        ^          |
        |          v
 MoveIt 2 / move_group / robot_state_monitor <----> RViz
        ^
        |
 CartesianPath action server
        ^                         H600 Modbus TCP client
        |                                ^
 welding_scenario_gui --> H600 bridge ---+
```

주요 통신은 다음과 같다.

| 구간 | ROS 인터페이스 | 역할 |
|---|---|---|
| 로봇 상태 | `/joint_states` | 각 팔의 실제 6축 각도와 RViz 현재 상태 동기화 |
| TCP 상태 | TF `World → *_ee_point` | TCP1, TCP2 및 터치 TCP 취득 |
| 터치 입력 | `/*_rbpodo_hardware/system_state` DI0 | 상승 에지에서 해당 팔 TCP 취득 |
| 경로 계획 | `/compute_cartesian_path` | 충돌을 고려한 Cartesian IK 경로 계산 |
| 계획 실행 | `/execute_trajectory` | 승인된 MoveIt trajectory 실행 |
| GUI 작업 | `cartesian_path` action | plan-only와 동일 계획 실행을 구분 |
| 용접 명령 | `/h600/set_command` | ready, gas, ARC, 전류 및 전압 명령 |
| 용접 피드백 | `/h600/status` | Modbus 연결 및 welding ON/OFF 확인 |

## 3. GUI 시나리오

### 0단계: 초기 자세 저장

선택한 팔의 현재 TCP와 6개 measured joint angle을 함께 저장한다. 복귀 목표는
TCP 역기구학 목표가 아닌 원래 joint angle이므로, 시작 자세를 재현할 때 IK
분기가 바뀌는 문제를 줄인다.

### 1단계: TCP1 및 TCP2 취득

두 TCP는 수동 버튼 또는 DI0 터치 상승 에지로 취득한다. `Use DI0 touch
sensing`이 켜져 있을 때만 터치 이벤트를 사용하며, `Save touched TCP pose`를
끄면 접촉 TCP를 마지막 측정값으로만 보관하고 endpoint는 변경하지 않는다.

TCP1과 TCP2 사이의 위치는 다음과 같이 선형 보간한다.

```text
p(s) = (1-s) p1 + s p2,  0 <= s <= 1
```

자세 quaternion은 두 quaternion의 내적이 음수이면 한 quaternion의 부호를
뒤집어 최단 회전 경로를 선택하고, 정규화 선형 보간한다.

### 2단계: 용접 조건

H600에 전달할 `current_raw`, `voltage_raw`, `V offset raw`, pre-flow 및
post-flow 시간을 설정한다. 0이 아닌 setpoint와 ARC 출력은 GUI 확인만으로
활성화되지 않는다. bridge 실행 시 다음 두 launch safety lock도 모두
활성화되어야 한다.

```text
allow_arc_output:=true
allow_nonzero_setpoints:=true
```

### 3단계: 계획 및 용접 실행

첫 버튼은 plan-only 요청이다. MoveIt은 현재 상태에서 TCP1까지의 approach와
TCP1에서 TCP2까지의 seam을 별도 trajectory로 만들고 RViz에 표시한다. 이
단계에서는 ARC를 켜지 않는다.

사용자가 RViz 결과를 확인한 뒤 실행 버튼을 누르면 경로, 속도, 보간 간격이
동일한지 signature로 검사하고 서버에 승인된 계획 재사용을 요청한다.

```text
TCP1 approach (ARC OFF)
  -> H600 connected 확인
  -> ready + gas ON, setpoint 적용
  -> pre-flow
  -> ARC ON
  -> welding feedback ON 확인(선택 가능)
  -> approved TCP1-to-TCP2 trajectory 실행
  -> ARC OFF
  -> welding feedback OFF 확인
  -> post-flow
  -> ready/gas/ARC/setpoint SAFE OFF
```

서버는 용접 동작 중 오류가 발생해도 `finally` 정리 경로에서 ARC OFF와 SAFE
OFF를 수행한다. H600 미연결, bridge safety lock 비활성, 로봇 controller
미연결, 불완전한 Cartesian 계획은 실행 전에 차단된다.

GUI의 cancel은 ROS action 취소 요청이지 비상정지가 아니다. 이미 controller가
실행 중인 trajectory의 즉시 정지가 필요하면 반드시 로봇 비상정지 장치를
사용해야 한다.

### 4단계: 초기 자세 복귀

저장한 6개 joint angle을 MoveGroup joint constraint로 계획한다. 먼저 RViz에
plan-only 결과를 표시하고, 별도 확인 이후 그때 승인된 RobotTrajectory를
`ExecuteTrajectory`로 실행한다. 계획 버튼과 이동 버튼을 분리하여 복귀
경로도 실제 동작 전에 검토한다.

## 4. 모션 생성 및 시간 파라미터화

Cartesian 계획의 `max_step`은 GUI의 interpolation step이다. 기본값 2 mm는
용접 경로의 IK 표본을 충분히 조밀하게 만들기 위한 시작값이며 0.5~20 mm
범위에서 조정할 수 있다. 작을수록 계산량은 증가하지만 joint-space 변화의
관찰과 완만한 경로 생성에 유리하다.

MoveIt trajectory에는 다음 처리가 적용된다.

```text
Cartesian waypoint interpolation
  -> collision-aware IK sampling
  -> Time-Optimal Trajectory Generation
  -> Ruckig jerk-limited smoothing
  -> GUI velocity scale
  -> FollowJointTrajectory
```

GUI velocity scale `v`를 적용하면 시간은 대략 `1/v`, 속도는 `v`, 가속도는
`v²` 비율로 조정된다. 실제 제어 명령은 joint trajectory controller가
hardware command interface에 전달하며, `rbpodo_ros2`가 Rainbow 제어기로
전송한다. 본 시나리오 구현에서는 `rbpodo`와 `rbpodo_ros2`를 수정하지 않는다.

현재 `ros2_controllers.yaml`의 controller manager update rate는 100 Hz이며,
좌우 FollowJointTrajectory controller의 action monitor rate는 20 Hz이다.
MoveIt이 만든 시간 기반 trajectory를 100 Hz ros2_control loop가 샘플링하여
실제 장치 command로 전달하고, action 상태는 20 Hz로 감시한다.

## 5. RViz 및 연결 상태

GUI의 RIGHT/LEFT O 표시는 다음 조건을 동시에 만족할 때 켜진다.

- 최근 2초 안에 해당 팔의 완전한 6축 measured `/joint_states` 수신
- `CartesianPath` action server 준비
- 실제 실행 모드이면 해당 `FollowJointTrajectory` action server 준비

H600 O 표시는 Modbus server 실행과 실제 TCP/502 client 연결을 모두
확인한다. 경로 marker와 계획 trajectory는 각각 `/weld_path_markers`,
`/weld_6d_poses`, `/display_planned_path`를 통해 RViz에 표시된다.

## 6. 실행 방법

계획 및 연결 시험만 수행할 때는 안전 lock을 끈 기본값을 사용한다.

```bash
ros2 launch construct_robot welding_scenario_gui.launch.py
```

실제 ARC와 0이 아닌 전류/전압 명령을 허용할 때만 위험 구역을 확인하고
다음과 같이 명시적으로 실행한다.

```bash
ros2 launch construct_robot welding_scenario_gui.launch.py \
  execute_motion:=true \
  allow_arc_output:=true \
  allow_nonzero_setpoints:=true
```

GUI의 실행 버튼에도 별도의 실제 로봇 확인 창이 있으며, RViz plan 승인 없이
용접 실행 또는 초기 자세 복귀 실행을 할 수 없다.

## 7. 시험 항목

- 터치 TCP를 endpoint에 저장하지 않는 모드
- 터치 순서대로 TCP1, TCP2를 자동 저장하는 모드
- TCP1/TCP2가 포함된 linear interpolation
- 경로 또는 용접 설정 변경 시 승인 signature 무효화
- 팔 변경 시 저장 자세와 endpoint 초기화
- 기존 Cartesian/H600 command sequence 회귀 테스트

실제 ARC 시험 전에는 `execute_motion:=false`로 RViz 계획, DI0 취득, H600 연결
표시와 safety-lock 차단 동작을 먼저 확인해야 한다.
