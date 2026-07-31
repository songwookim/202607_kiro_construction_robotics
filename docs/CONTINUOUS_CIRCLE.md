# Circle 용접을 끊김 없이 만드는 수정 실습

이 문서는 `ADD_VISER_HOME_COMMAND.md`와 같은 방식으로, 어느 파일의 어느
함수 앞뒤에 무엇을 붙여 넣는지 순서대로 설명한다.

먼저 반드시 `src` 아래 파일을 수정한다. IDE에 열려 있는
`ros2_ws/build/construct_robot/...` 파일은 빌드 생성물이므로 수정하지
않는다.

수정 대상은 다음 세 파일이다.

```text
ros2_ws/src/construct_robot_ros2/
├── construct_robot/construct_robot/cartesian_path_common.py
├── construct_robot/construct_robot/weld_action_gui.py
└── construct_robot/construct_robot/cartesian_path_server.py
```

목표는 다음과 같다.

- 원 전체를 하나의 Cartesian Action으로 보낸다.
- waypoint 사이에서 멈추지 않는다.
- quaternion 부호가 원 중간이나 닫힘 지점에서 뒤집히지 않는다.
- 여러 바퀴도 하나의 trajectory로 만든다.
- 마지막 바퀴가 끝난 뒤에만 정지한다.

## 1. Circle 기본 해상도를 48점으로 변경

파일:

```text
construct_robot/construct_robot/weld_action_gui.py
```

`class WeldActionGui`의 `__init__()`에서 다음 줄을 찾는다.

```python
self.circle_count = tk.IntVar(value=16)
```

다음처럼 바꾼다.

```python
self.circle_count = tk.IntVar(value=48)
```

4점은 다이아몬드, 16점은 거친 원에 가깝다. 반지름 20 mm 기준으로
48점부터 테스트하고 IK 계산이 너무 느리면 32점으로 낮춘다.

## 2. Quaternion 부호 연속성 helper 추가

파일:

```text
construct_robot/construct_robot/cartesian_path_common.py
```

다음 함수의 끝을 찾는다.

```python
def _quaternion_from_axes(x_axis, y_axis, z_axis):
    ...
    return qx, qy, qz, qw
```

그 함수 바로 아래, `def circle_waypoints(` 바로 위에 다음 함수를
붙여 넣는다.

```python
def keep_quaternion_hemisphere(previous, current):
    """Make q/-q representations continuous across adjacent waypoints."""
    dot = (
        previous.x * current.x
        + previous.y * current.y
        + previous.z * current.z
        + previous.w * current.w
    )
    if dot < 0.0:
        current.x *= -1.0
        current.y *= -1.0
        current.z *= -1.0
        current.w *= -1.0
```

`q`와 `-q`는 같은 회전이지만, waypoint 배열에서 갑자기 부호가
바뀌면 일부 보간기가 긴 방향으로 회전하는 것처럼 처리할 수 있다.

## 3. 각 Circle waypoint를 append하기 전에 helper 호출

같은 파일의 `circle_waypoints()` 안에서 다음 부분을 찾는다.

```python
else:
    pose.orientation = copy.deepcopy(center.orientation)
points.append(pose)
```

`points.append(pose)` 바로 앞에 다음 네 줄을 추가한다.

```python
else:
    pose.orientation = copy.deepcopy(center.orientation)

if points:
    keep_quaternion_hemisphere(
        points[-1].orientation,
        pose.orientation,
    )
points.append(pose)
```

들여쓰기는 `for index in range(count):` 안쪽이어야 한다.

## 4. Circle 닫힘 지점에도 quaternion 연속성 적용

같은 함수 아래쪽에서 다음 블록을 찾는다.

```python
if closed:
    closing_pose = Pose()
    closing_pose.position = copy.deepcopy(points[0].position)
    closing_pose.orientation = copy.deepcopy(points[0].orientation)
    points.append(closing_pose)
```

`points.append(closing_pose)` 앞에 다음 호출을 추가한다.

```python
if closed:
    closing_pose = Pose()
    closing_pose.position = copy.deepcopy(points[0].position)
    closing_pose.orientation = copy.deepcopy(points[0].orientation)
    keep_quaternion_hemisphere(
        points[-1].orientation,
        closing_pose.orientation,
    )
    points.append(closing_pose)
```

첫 점과 마지막 점의 위치와 실제 회전은 같으면서 quaternion 표현만
직전 점에 연속적으로 유지된다.

## 5. Cartesian 보간 간격을 5 mm에서 2 mm로 변경

파일:

```text
construct_robot/construct_robot/weld_action_gui.py
```

`class WeldActionNode`의 `send()` 안에서 다음 줄을 찾는다.

```python
goal.interpolation_step = 0.005
```

다음처럼 바꾼다.

```python
goal.interpolation_step = 0.002
```

단위는 metre다. `0.002`는 최대 2 mm 간격이다. IK가 불안정하거나
계획 시간이 너무 길면 `0.003`으로 올린다.

## 6. 원 전체가 한 Action으로 전송되는지 확인

같은 `send()` 함수에는 다음 구조가 한 번만 있어야 한다.

```python
goal.waypoints = points
future = self.client.send_goal_async(
    goal,
    feedback_callback=self.feedback,
)
```

다음과 같은 waypoint별 전송 loop는 추가하지 않는다.

```python
# 잘못된 방식: 각 점 끝에서 별도 계획/정지가 생긴다.
for pose in points:
    goal.waypoints = [pose]
    self.client.send_goal_async(goal)
```

현재 프로젝트 코드는 이미 올바르게 원 전체를 한 goal로 보낸다.

## 7. 중간 trajectory 속도 확인 로그 추가

파일:

```text
construct_robot/construct_robot/cartesian_path_server.py
```

`plan_with_moveit()`에서 다음 줄을 찾는다.

```python
scale_trajectory_speed(response.solution, request.velocity_scale)
```

그 줄 바로 아래에 임시로 다음 코드를 추가한다.

```python
for index, point in enumerate(
    response.solution.joint_trajectory.points
):
    max_velocity = max(
        map(abs, point.velocities),
        default=0.0,
    )
    self.get_logger().info(
        f"circle trajectory[{index}] "
        f"t={point.time_from_start.sec}."
        f"{point.time_from_start.nanosec:09d} "
        f"max|v|={max_velocity:.4f}"
    )
```

확인할 내용:

- `time_from_start`가 계속 증가한다.
- 원 중간의 모든 점에서 속도가 의도적으로 0으로 고정되지 않는다.
- 최종 점에서는 정지해도 정상이다.

확인이 끝나면 이 진단 로그는 삭제해도 된다.

## 8. 여러 바퀴를 하나의 연속 trajectory로 만들기

여러 바퀴를 실행하려면 `circle_waypoints()`를 매번 Action으로 보내지
말고, waypoint를 먼저 전부 합쳐야 한다.

`cartesian_path_common.py`에는 이미 `import copy`가 있으므로 새 import는
필요 없다. `circle_waypoints()` 아래에 다음 함수를 추가한다.

```python
def multi_lap_circle_waypoints(
    center,
    radius,
    count,
    laps,
    face_center=True,
):
    if laps < 1:
        raise ValueError("Circle laps must be at least one")

    one_lap = circle_waypoints(
        center,
        radius,
        count,
        closed=False,
        face_center=face_center,
    )
    points = []
    for _lap in range(laps):
        for pose in copy.deepcopy(one_lap):
            if points:
                keep_quaternion_hemisphere(
                    points[-1].orientation,
                    pose.orientation,
                )
            points.append(pose)

    closing_pose = copy.deepcopy(points[0])
    keep_quaternion_hemisphere(
        points[-1].orientation,
        closing_pose.orientation,
    )
    points.append(closing_pose)
    return points
```

GUI의 `WeldActionNode.generate_circle()` 안에서 기존 호출:

```python
points = circle_waypoints(
    tcp,
    radius,
    count,
    closed,
    face_center,
)
```

을 여러 바퀴 테스트용으로 다음처럼 바꿀 수 있다.

```python
points = multi_lap_circle_waypoints(
    tcp,
    radius,
    count,
    laps=3,
    face_center=face_center,
)
```

이 경우 import 목록에도 함수를 추가한다.

```python
from construct_robot.cartesian_path_common import (
    circle_waypoints,
    multi_lap_circle_waypoints,
    pose_is_valid,
    weaving_from_path,
)
```

## 9. 빌드와 테스트

수정 후 workspace root에서 실행한다.

```bash
cd /home/irs/ros2_ws
source /opt/ros/humble/setup.bash

colcon build \
  --packages-select construct_msgs construct_robot \
  --symlink-install

source install/setup.bash

colcon test \
  --packages-select construct_msgs construct_robot

colcon test-result --verbose
```

## 10. 동작 확인 순서

```bash
ros2 launch construct_robot weld_action_gui.launch.py \
  use_fake_right_hardware:=true \
  execute_motion:=true
```

1. `Generate circle`
2. `1 · Plan Preview`
3. RViz와 `http://localhost:8080` Viser에서 원과 관절 궤적 확인
4. 속도 5–10% 설정
5. 다시 `Plan Preview`
6. `Execute Approved Plan`
7. 중간 정지 없이 마지막 점에서만 정지하는지 확인

## 11. 일정한 TCP 원주 속도가 꼭 필요한 경우

현재 방식은 `GetCartesianPath`와 joint-space time parameterization을
사용하므로 부드럽지만 TCP 속도가 원 전체에서 완전히 일정하다고
보장하지는 않는다.

생산용 원형 용접에서 정확한 원과 일정한 TCP 속도가 필요하면 다음
단계는 Pilz `CIRC` motion sequence다. 수정 위치는 GUI나 Viser가 아니라
`cartesian_path_server.py`의 `plan_with_moveit()`이며,
`GetCartesianPath` 요청을 circle mode에서 Pilz `CIRC` 요청으로
분기해야 한다.

실제 RB에서는 반드시 다음 순서를 지킨다.

1. Fake hardware 5–10%
2. Real RB 5%, ARC OFF, 비상정지 대기
3. 원 한 바퀴
4. 여러 바퀴
5. 마지막에만 ARC 허용
