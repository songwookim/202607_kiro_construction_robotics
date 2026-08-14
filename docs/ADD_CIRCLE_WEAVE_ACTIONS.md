# Circle / Weave ROS 2 Action 추가 가이드

## 먼저 현재 구조 이해하기

현재 Circle과 Weave는 이미 실행 가능하다. GUI가 Circle/Weave 6D pose
waypoint를 먼저 만들고, Straight와 동일한
`construct_msgs/action/CartesianPath` action으로 보낸다.

```text
Generate circle / Apply weave
        ↓
geometry_msgs/Pose[]
        ↓
/cartesian_path action
        ↓
/compute_cartesian_path → /execute_trajectory
```

따라서 단순히 원/위빙 용접을 실행하려면 새 action을 만들
필요가 없다. GUI에서 경로를 만든 뒤 `Plan Preview` →
`Execute Approved Plan`을 사용하면 된다.

아래는 학습/모듈화 목적으로 Circle과 Weave를 **별도 ROS 2 action**으로
분리하는 방법이다.

## 1. Action interface 추가

### `construct_msgs/action/CirclePath.action`

```text
string planning_group
geometry_msgs/Pose center
float64 radius
uint32 point_count
bool close_path
bool face_center
float64 velocity_scale
bool execute_requested
---
bool success
string message
geometry_msgs/Pose[] generated_path
---
float32 progress
uint32 waypoint_index
geometry_msgs/Pose current_pose
```

### `construct_msgs/action/WeavePath.action`

```text
string planning_group
geometry_msgs/Pose[] source_path
float64 amplitude
uint32 cycles
uint32 samples_per_cycle
string transverse_axis
float64 velocity_scale
bool execute_requested
---
bool success
string message
geometry_msgs/Pose[] generated_path
---
float32 progress
uint32 waypoint_index
geometry_msgs/Pose current_pose
```

`construct_msgs/CMakeLists.txt`의 `rosidl_generate_interfaces()`에 다음을 추가한다.

```cmake
"action/CirclePath.action"
"action/WeavePath.action"
```

그 뒤 interface를 먼저 빌드한다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select construct_msgs --symlink-install
source install/setup.bash
```

## 2. Circle action server 추가

파일을 만든다.

```text
construct_robot/construct_robot/circle_path_server.py
```

핵심 순서는 다음과 같다.

1. `ActionServer(self, CirclePath, "circle_path", ...)` 생성
2. goal의 `planning_group` 검사
3. `cartesian_path_common.circle_waypoints()`로 waypoint 생성
4. 생성된 waypoint를 기존 `CartesianPath` action client에 전달
5. 하위 action feedback을 Circle feedback으로 변환
6. 하위 action result를 Circle result로 변환

중복된 MoveIt 서비스 코드를 새로 작성하지 말고, 새 action server를
`/cartesian_path`의 고수준 wrapper로 만드는 것이 안전하다.

## 3. Weave action server 추가

파일을 만든다.

```text
construct_robot/construct_robot/weave_path_server.py
```

핵심 순서:

1. `ActionServer(self, WeavePath, "weave_path", ...)` 생성
2. `source_path` 최소 2개 자세 검사
3. `cartesian_path_common.weaving_from_path()` 호출
4. 생성된 waypoint를 `/cartesian_path`에 전달
5. 취소 요청을 하위 Cartesian action에도 전달
6. Plan-only result와 Execute result를 구분

## 4. Entry point 등록

`construct_robot/setup.py` → `console_scripts`에 추가한다.

```python
"circle_path_server = construct_robot.circle_path_server:main",
"weave_path_server = construct_robot.weave_path_server:main",
```

## 5. Launch에 server 추가

사용자 launch가 직접 포함하는
`construct_robot/launch/weld_stack.launch.py`에 노드를 추가한다.

```python
circle_server = Node(
    package="construct_robot",
    executable="circle_path_server",
    output="screen",
)

weave_server = Node(
    package="construct_robot",
    executable="weave_path_server",
    output="screen",
)
```

그리고 `LaunchDescription` 리스트에 `circle_server`, `weave_server`를 넣는다.

## 6. GUI client를 별도 action으로 바꾸기

`weld_action_gui.py`의 `WeldGuiNode.__init__()`에 client를 추가한다.

```python
self.circle_client = ActionClient(self, CirclePath, "circle_path")
self.weave_client = ActionClient(self, WeavePath, "weave_path")
```

그런 다음:

- `generate_circle()`은 로컬 waypoint을 만드는 대신 `CirclePath.Goal`을 보낸다.
- `generate_weave()`는 `source_path`와 weave 설정을 `WeavePath.Goal`로 보낸다.
- result의 `generated_path`를 `set_new_points()`에 넘겨 테이블/RViz에 표시한다.
- GUI의 Left/Right 선택값을 `goal.planning_group`에 넣는다.

## 7. 꼭 넣을 검증

`construct_robot/test/test_cartesian_path_math.py`에 다음을 검증한다.

- Circle의 반지름과 point count
- closed path의 마지막 pose가 첫 pose와 같은지
- `face_center=True`일 때 TCP +Z가 중심을 보는지
- Weave 시작/종료 pose가 source seam과 같은지
- amplitude, cycle, sample count
- Left/Right planning group이 해당 TCP link로 변환되는지
- Plan Preview 후 경로/속도/팔을 바꾸면 승인된 plan이 무효화되는지
- Cancel이 하위 `/cartesian_path` goal까지 전달되는지

## 8. 추천 구현 순서

1. Circle action을 **Plan-only**로 먼저 구현
2. RViz marker/result path 확인
3. Circle Execute 연결
4. Weave action을 Circle server 구조를 복사해 구현
5. FAKE/Plan-only 테스트
6. REAL RB는 낮은 속도, ARC OFF, 감시 환경에서 확인

중요: Circle/Weave action server에서 ARC를 직접 제어하지 않는다. 모션은
`CartesianPath`로 처리하고 용접은 Sequence Builder의 Hi-COMM D-WELD 단계로
분리한다.
