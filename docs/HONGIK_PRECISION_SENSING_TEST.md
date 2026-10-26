# Hongik 스캐너 — 1. CONNECTION 스캐너 섹션 · 2. TASK › Scan 탭

`weld_action_gui`의 `2. TASK › Scan`은 홍익 정밀 센싱·용접 후 스캔을 실행하고 결과와 pose를
확인하는 화면입니다. 결과의 경로를 기존 작업 시퀀스나 멀티패스 교정 상태에 적용하지 않습니다.

## 1. CONNECTION › Hongik Scanner

센서·서버 확인만 하며 로봇을 움직이지 않습니다. 헤더의 `SCANNER` 표시는 마지막 확인 결과입니다.

- `Check sensor`: 센서 파라미터(ID 44)를 읽기만 합니다. 펌웨어, 레이저 상태, 측정 범위를 표시합니다.
- `Laser ON` / `Laser OFF`: LaserMode만 변경하고 센서에서 다시 읽어 확인합니다.
  레이저 상태는 별도로 표시하며, RViz 시각화 중에는 같은 TCP 접속으로 조작합니다.
  레이저 버튼만 누르면 프로파일 스트리밍을 시작하지 않습니다.
- `RViz 시각화 ON`: 파라미터 백업(`~/scan_data/sensor_backup`) → 홍익 표준 측정 설정
  (현재 레이저 상태 유지, mm 모드) → 스트리밍. PointCloud2 `/hongik_scanner/raw_profile`에 센서의 모든 열을
  순서대로 발행합니다. 좌표는 `hongik_profiler_optical_frame`에서 `(x, 0, z)` [m]이며,
  `raw_x_mm`, `raw_z_mm` 필드에는 변환 전 센서 값 [mm], `column_index`에는 원본 열 번호를 보존합니다.
  양의 유한 range는 교정 범위 밖이어도 raw 화면에 표시합니다. z=0 / NaN 무효점은 원본 값을 보존하고
  표시 좌표만 NaN으로 두어 센서 원점에 가짜 점을 그리지 않습니다. `sensor_valid`는 별도의 유효 범위 검사입니다.
  기존 `/hongik_scanner/profile` 토픽은 유효점만 발행합니다. `Gain`이 0이면 홍익 표준값입니다.
  상태에는 raw 표시 가능점·유효점 비율·z=0·비유한 값 개수·센서 오류·checksum을 표시합니다.
- 같은 버튼의 `RViz 시각화 OFF`: 스트리밍 정지, RViz 점 제거, 측정 파라미터 복원을 합니다.
  마지막 레이저 ON/OFF 명령은 유지하므로 레이저를 끄려면 `Laser OFF`를 누릅니다.
  GUI 종료와 EMERGENCY STOP은 시각화를 정지하고 레이저도 OFF로 설정합니다.
- `Calib YAML`이 있으면 `hongik_laser_link → hongik_profiler_optical_frame` 정적 TF를 발행합니다.
  기본 모델 `hongik_v4`에서는 로봇 툴에 붙어 보입니다. `robot_model:=kiro_legacy`이면 RViz Fixed Frame을
  `hongik_profiler_optical_frame`으로 둡니다. 기본 `moveit.rviz`에 `Hongik Scanner Raw Profile` 표시가 있습니다.
  RViz에서 점을 선택하면 원본 mm 값과 열 번호를 확인할 수 있습니다. raw/유효점이 모두 0이면
  센서가 측정 대상을 보지 못한 상태일 수 있으므로 대상 위치·거리·레이저 신호를 확인합니다.
- `Check server`: 홍익 서버 status 서비스의 ready·running·base_frame·awaiting_confirm을 표시합니다.
- 센서 TCP 접속은 한 클라이언트만 받습니다. 라이브 뷰 중에는 Scan 탭의 스캔이 막히고,
  스캔 중에는 라이브 뷰를 시작할 수 없습니다.

Raw RViz 변경 검증: `test_scanner_scan_task.py`, `test_precision_sensing_test.py`,
`test_ui_dispatch.py`의 26개 테스트 통과. 원본 값·열 순서 보존, 교정 범위 밖 점 표시,
무효점 처리, PointCloud2 직렬화, 수신 시각 사용, 센서 복원과 기존 GUI 흐름을 확인했습니다.
실행 중인 GUI에서도 프로파일 수신과 World → optical TF를 확인했으나,
확인 시점에는 유효 프로파일이 비어 있어 실제 물체의 점 표시는 대상 배치 후 확인해야 합니다.

세 버튼 분리 검증: `test_scanner_connection_panel.py`, `test_scanner_scan_task.py`,
`test_ui_dispatch.py`, `test_touch_stop_safety.py`를 실행했습니다. 레이저만 조작할 때
스트리밍하지 않는지, RViz ON에서 OFF인 레이저를 켜지 않는지, 시각화 중 레이저 조작과
복원·종료 동작이 올바른지 확인했습니다. 실제 센서 레이저 조작 시험은 수행하지 않았습니다.

## 먼저 결과 파일로 확인

1. GUI에서 `2. TASK › Scan`을 엽니다. `Parent frame`을 해당 실행의 `sections.json`에 있는
   `base_frame`으로 설정합니다. KIRO 기본값은 `World`입니다.
2. `Load precision_sensing_result.yaml`로 결과 파일을 엽니다. 같은 실행 폴더에 있는
   `sections.json`, `kiro/pass_<k>.yaml`, `kiro/pass_<k>_info.yaml`도 읽습니다.
   이 버튼은 로봇을 움직이지 않습니다.
3. 표에서 패스·포인트를 선택해 XYZ [mm], quaternion [xyzw]를 확인합니다.
   아래에는 측정 길이·끝점·단면 치수·불확도·미해결 항목과 파일 간 일치 검사가 표시됩니다.
4. 별도로 측정한 기준 pose를 비교란에 입력하고 `Compare`를 누릅니다.
   선택한 pose와 같은 프레임, 같은 WIRE/TCP 규약으로 입력해야 합니다.
   위치 차이 [mm]와 회전각 차이 [deg]를 입력한 허용치와 비교합니다.

번들의 `samples/standin_run/precision_sensing_result.yaml`은 KIRO 대역 시험 예시입니다.
`samples/lab_run/`은 연구실 로봇 좌표이므로 실제 KIRO 로봇 위치의 기준으로 사용하지 않습니다.

## 실제 센싱 요청

KIRO의 `welding_interfaces/action/PrecisionSensing`과 홍익 `hongik_scan_ros`를 빌드하고
두 워크스페이스의 `install/setup.bash`를 GUI 실행 터미널에 source해야 합니다.
`welding_interfaces`가 없으면 기존 GUI와 파일 검사는 사용할 수 있고, 센싱 요청은 오류로 끝납니다.
서버 실행 방법은 번들의 `docs/KIRO_RUN_MANUAL.md`를 따릅니다.
2026-10-14b RELAY 갱신은 [field overlay 실행 안내](HONGIK_FIELD_UPDATE.md)를 따릅니다.
`tools/activate_hongik_field.bash`를 source하고 공식
`ros2 launch hongik_scan_ros hongik_moveit.launch.py`로 서버를 실행합니다.
I/J·O/K의 센서 스캔 자세와 액션 A/B 이음매 점은 별도 파일로 교시합니다.

1. KIRO MoveIt/JTC와 로봇 모델·TF, 센서 교정 파일, 시편 job을 준비하고 홍익 서버를 실행합니다.
2. 오른팔이 연결되고 기존 GUI의 motion execution이 활성화된 상태에서 다른 작업과
   Keyboard Teaching을 종료합니다.
3. 시편 안쪽 이음부에 있는 대략적인 두 점 A/B를 **mm**로 입력합니다.
   와이어 끝을 이음매 위에 둔 뒤 `Save wire tip → A/B`로 별도 이음매 점을 저장합니다.
   I/J 또는 O/K로 교시한 센서 스캔 자세는 이 A/B 입력을 채우지 않습니다.
   액션에는 m로 변환해 전송합니다. 입력 orientation은 identity이며 서버는 위치만 사용합니다.
4. `Start full 1G scan (ROBOT MOVES)`를 누릅니다. 이 요청은 실제 로봇을 움직입니다.
   서버 상태 서비스에서 프레임·ready·running을 확인한 뒤 액션을 보냅니다.
   실행 중 다른 작업 탭과 TASK 하위 탭은 잠기며 로그와 경과 시간이 표시됩니다.
5. `Cancel scan` 또는 기존 STOP은 센싱 액션의 취소를 요청합니다.
   GUI 종료 시에도 진행 중인 센싱 요청을 취소합니다.

## 용접 후 원자료 스캔 (Post-weld raw scan)

`hongik_scan_msgs/action/PostWeldScan`(`/hongik_fine_sensing/post_weld_scan`)을 보냅니다.
끝난 1G 실행의 측정 이음매를 따라 한 번 스캔해 원자료만 저장합니다(추정·판정·용접 계획 변경 없음).

1. `Run dir`(비우면 서버의 최신 실행), `After pass`(0 = 최종), 속도 mm/s(0 = 그 실행의 측정 속도)를 입력합니다.
   `directions/E_to_S/kiro` 같은 출력 하위 폴더가 아니라 원래 실행 폴더를 지정합니다.
   `Scan direction`은 Auto(가까운 끝부터), S->E 또는 E->S이며,
   `Tool orientation`은 Sensing pass(기본: 원래 센싱 자세) 또는 Least rotation(가까운 토치 쪽)입니다.
2. `Dry run`(기본 켜짐)은 계획만 하고 로봇·센서를 쓰지 않습니다. 끄면 로봇이 움직이며,
   오른팔 연결과 motion execution이 필요합니다.
3. `Confirm`이 `each`이면 서버가 이동마다 확인을 기다립니다. 그때 `CONFIRM move` / `Reject move`가
   활성화되고 `~/confirm_move`(std_srvs/SetBool)로 응답합니다.
4. 결과 status(OK·DRY_RUN·REFUSED·ABORTED·ERROR), 메시지, 출력 폴더를 표시합니다.
   `Cancel scan`과 기존 STOP은 이 액션도 취소합니다.

## RViz와 값 규약

기본 `moveit.rviz`에는 `Hongik Test Paths`와 `Hongik Test Poses` 표시가 포함됩니다.
사용자 RViz 설정에서는 MarkerArray `/hongik_test/path_markers`와 TF 표시를 추가합니다.
Fixed Frame은 결과의 부모 프레임과 연결되어야 합니다.

- `hongik_test/pass_<k>_wire_start/goal`: 액션의 와이어 끝 경로 양 끝. +Z는 판 법선,
  +X는 진행 방향으로, 토치 orientation이 아닙니다.
- `hongik_test/pass_<k>_tcp_start/goal`: KIRO 파일의 컨택트 팁 TCP 자세. +Z는 노즐 축입니다.
- `hongik_test/selected_wire` / `selected_tcp`: 표에서 선택한 pose.

TF는 반환된 결과의 고정 pose를 10 Hz로 재전송한 것으로 실시간 로봇 pose가 아닙니다.
`Clear result / RViz`는 경로를 지우고 TF 재전송을 멈춥니다. 기존 TF는 RViz의 timeout 뒤 사라집니다.
패스가 4개보다 많으면 추가 패스의 TF를 RViz Frames 목록에서 활성화할 수 있습니다.

파일 간 검사는 해당 실행의 `tcp_to_aim_point_mm`를 토치 로컬 +Z로 적용해 액션의 와이어 끝
위치와 비교합니다. offset 메타데이터가 없거나 부모 프레임이 다르면 검사를 생략하고 표시합니다.
`PASS_UNRESOLVED`는 액션 성공 코드 0이어도 미해결 항목을 명시합니다.
수치·쿼터니언·간격·파일 일치 검사만으로 센서의 절대 정확도가 검증되지는 않습니다.

## 구현 검증

아래 테스트 파일을 ROS 2 Humble 환경에서 함께 실행해 193개가 통과했습니다.
실제 센서·로봇을 움직이는 시험과 GUI 화면 렌더링 시험은 수행하지 않았습니다.

```text
construct_robot/test/test_precision_sensing_inspection.py
construct_robot/test/test_precision_sensing_test.py
construct_robot/test/test_ui_dispatch.py
construct_robot/test/test_package_boundaries.py
construct_robot/test/test_four_pass_correction.py
construct_robot/test/test_weld_weave.py
construct_robot/test/test_application_state.py
construct_robot/test/test_touch_stop_safety.py
construct_robot/test/test_action_call.py
construct_robot/test/test_cartesian_path_math.py
```

번들 결과 파일로도 로딩을 확인했습니다. `standin_run`의 와이어 pose 164개와 TCP 끝점 8개,
`lab_run`의 와이어 pose 308개와 TCP 끝점 8개를 읽었습니다.
대역 시험의 8개 끝점은 TCP 돌출 보정 후 와이어 위치와 일치했습니다.
연구실 데이터는 액션 부모 프레임 `link0`와 TCP 파일 부모 프레임 `World`가 달라
두 값의 직접 비교를 생략합니다.

## 원본 KIRO 패스 이동 (2026-10-15)

GUI 재실행 후 `2.TASK → Scan → Original KIRO pass · motion only (no ARC)`에서 사용합니다.
기본 파일은 `~/hongik/runs/*/kiro/pass_1.yaml` 중 최신 파일이며, Browse로 다른 패스도 고를 수 있습니다.

1. `Load pass YAML`로 원본 위치·자세를 표와 RViz TF에 표시합니다.
2. `START`, `GOAL`, `START → GOAL` 중 고릅니다. 마지막 모드는 현재 TCP에서 START로 접근한 뒤 패스 직선을 따라 GOAL로 이동합니다.
3. `Plan`으로 계획을 확인하고 `Execute (ROBOT MOVES)`로 기존 시퀀스 실행 확인 화면을 거쳐 이동합니다.
   패스 직선 속도는 기본 5 mm/s이며 조정할 수 있습니다. 접근 속도는 기존 GUI 속도 비율을 사용합니다.
4. `STOP`은 기존 정지 경로로 이동을 취소하고, 준비 중인 실행 요청도 무효화합니다.

파일은 `xy_endpoint_review_v1`, `frame_id: World` 형식이어야 합니다.
현재 시스템의 `right_manipulator_ee_point` TCP에 원본 위치·쿼터니언을 사용하며,
보정·추가 와이어 오프셋·위빙·용접 명령을 적용하지 않습니다.
충돌·접촉 정지·로봇 연결·교시 및 다른 이동과의 중복 실행 검사를 유지합니다.
준비 중 Hongik 서버의 idle 상태를 조회하므로 서버의 status 서비스도 실행 중이어야 합니다.
읽은 파일의 해시를 로그에 남기며, 파일이 바뀌었으면 다시 불러와야 합니다.
원본 결과 파일과 용접 교시 데이터는 수정하지 않습니다.

START 접근은 병렬 슬롯 1, 패스 직선 이동은 슬롯 2로 순차 실행합니다.
최초 구현에서 두 행이 같은 슬롯 1을 사용해 실행 전 검사에 걸리는 오류를 수정했습니다.
스캔 탭에서도 기존 실행 전 검사의 거절 사유(예: Fastech 접촉 정지 연결)를 표시합니다.
이 이동은 ARC 설정·Hi-COMM 연결을 요구하지 않으며, ARC 이벤트를 기다리지 않습니다.

검증: test_hongik_pass_motion, test_sequence_executor, test_scan_teaching, test_direct_scan,
test_precision_sensing_test 총 130개 통과. 실제 시퀀스 실행기에 가짜 런타임을 연결하여
Hi-COMM 없이 execute=True인 두 이동이 순차 호출되고 ARC 호출이 없음을 확인했습니다.
`/home/irs/hongik/runs/20261015-005527_kiro_1G/kiro/pass_1.yaml`도 설치 패키지 경로에서 읽어 원본 끝점과 해시를 확인했습니다.
실제 팔 이동·용접 실행 및 GUI 렌더링 시험은 수행하지 않았습니다.
