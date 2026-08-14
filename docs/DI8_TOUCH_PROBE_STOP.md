# DI8 터치 프로브 정지 문제와 해결

## 최종 원인

DI8 입력은 정상적으로 들어왔지만, 처음 구현은 Cartesian action의 바깥 goal만
취소했다. 취소 요청이 MoveIt `/execute_trajectory`까지 전달되어도 실제
`right_manipulator_controller/follow_joint_trajectory` goal이 계속 살아 있는
경우가 있었다. 이때 로그에는 cancel이 표시되지만 로봇은 원래 목표까지 계속
움직였다.

또한 5% 저속 모션의 작은 관절 변화량을 정지로 오판했다. 따라서 실제 controller
goal이 실행 중인데도 `DI8 smooth stop`으로 처리하고 TCP를 저장했다.

초기 구현에는 다음 문제도 함께 있었다.

- `/system_state`가 controller 10 cycle마다 발행되어 DI8 확인이 약 3 Hz였다.
- GUI와 node가 같은 DI8 edge에서 stop worker를 중복 실행했다.
- 접점 채터링을 새 터치로 처리하여 복귀 경로가 다시 취소될 수 있었다.
- 정지 때 controller를 inactive/active로 전환하면서 로봇이 순간적으로 튀었다.
- 50 mm 프로브에 1 mm waypoint를 미리 생성하고 PLAN PREVIEW에서 속도에 따른
  sleep까지 수행하여 실제 출발 전 대기시간이 길었다.

## 적용한 해결 방법

현재 DI8 정지 순서는 다음과 같다.

1. 첫 DI8 rising edge를 latch하여 같은 프로브에서는 한 번만 처리한다.
2. Cartesian/MoveIt goal 취소를 요청한다.
3. arm의 `FollowJointTrajectory` cancel service에 zero UUID와 zero timestamp를
   보내 현재 controller goal을 직접 취소한다.
4. cancel 응답이 성공했는지 확인한다.
5. 더 엄격한 관절 변화량 기준으로 0.3초 이상 실제 정지를 확인한다.
6. 정지된 TCP를 저장하고 settle 시간 동안 유지한다.
7. 터치 감시를 해제한 뒤 프로브 시작점으로 한 번만 복귀한다.

직접 controller goal 취소가 실패하거나 정지가 확인되지 않을 때만 controller
비활성화와 RBPodo `move_stop`을 fallback으로 사용한다. 정상 경로에서는
controller mode switch를 하지 않으므로 접촉/복귀 시의 튀는 동작을 피한다.

정상 로그 예시는 다음과 같다.

```text
DI8 direct trajectory cancel: OK · return_code=0 · goals_canceling=1
DI8 smooth stop: action canceled · controller kept active
```

## Settle 시간

`settle`은 DI8 접촉 직후 로봇을 그 자리에서 유지하는 시간이다. 정지 직후의
미세한 기계 진동과 센서 채터링이 가라앉은 다음 복귀를 시작하도록 한다. 현재
기본값은 `0.7 s`이다. 이 값은 터치 접근속도가 아니며, seam 좌표 계산 전에
정지 상태를 안정화하기 위한 dwell이다.

## Seam Correction 기본값

| 설정 | 기본값 |
|---|---:|
| Seam axis | World X |
| Wall probe sign | `-` |
| Floor Z probe sign | `-` |
| Max travel | `25.0 mm` |
| Probe speed | `5.0%` |
| Settle | `0.7 s` |

직선 프로브는 시작점과 끝점 두 개만 전달하며, 1 mm 세부 보간은 MoveIt의
Cartesian `max_step`이 담당한다.
