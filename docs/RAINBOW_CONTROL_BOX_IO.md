# Rainbow control-box I/O 확인

## 알 수 있는 것과 알 수 없는 것

Rainbow controller state에는 control-box port 0~15의 현재 상태가 있다.

- `digital_in[0..15]`: 외부 장치에서 control box로 들어오는 DI
- `digital_out[0..15]`: control box가 외부 장치로 내보내는 DO

현재 ON/OFF와 신호가 바뀐 순간은 확인할 수 있다. 하지만 연결된 선이
현재 OFF라면 미배선 port와 값이 같으므로, 상태값만으로 물리 배선 여부를
확정할 수는 없다. 최종 배선 확인은 전원을 안전하게 차단한 상태의 도통
검사 또는 전장 도면이 필요하다.

## ROS topic

실제 오른팔 RB가 연결되면 드라이버가 10 Hz로 게시한다.

```text
/right_rbpodo_hardware/system_state
```

전체 상태 확인:

```bash
ros2 topic echo /right_rbpodo_hardware/system_state --once
```

DI 또는 DO만 확인:

```bash
ros2 topic echo /right_rbpodo_hardware/system_state --field digital_in
ros2 topic echo /right_rbpodo_hardware/system_state --field digital_out
```

Weld GUI에는 DI/DO 0~15가 항상 표시된다. ON은 녹색이며, 관찰 후보
4, 8, 9, 10, 12, 13은 파란 배경과 굵은 테두리로 표시된다. 변화가
발생하면 Pipeline status에도 예를 들어 `DI8=ON`처럼 기록된다.

현재 장비에서 touch 입력으로 지정한 raw `DI4`는 별도
`DI04 TOUCH` 표시에도 나타난다. 입력이 High이면 녹색 ON, Low이면 OFF가
되며, Low에서 High로 변한 횟수도 누적한다. 이 표시는 물리 입력을
자동으로 따라갈 뿐 어떤 DO도 자동으로 명령하지 않는다.

즉 touch 선이 정말 API index 4에 연결되어 있다면 다음 값이 바뀌는 것이
정상이다.

```text
digital_in[4]: false -> true
GUI DI row: 04 ON
GUI badge: DI04 TOUCH: ON
```

`digital_out[4]` 또는 다른 DO가 함께 ON되는 것은 아니다. DI-to-DO 연동은
별도의 제어 로직이며 배선 목적과 안전 동작이 확정된 뒤 추가해야 한다.
터치가 짧은 펄스라 GUI에서 놓칠 수 있으면 아래 명령을 함께 켜고 눌러서
raw index를 먼저 확인한다.

```bash
ros2 topic echo /right_rbpodo_hardware/system_state --field digital_in
```

DO cell을 클릭하면 현재 상태의 반대값을
`/right_rbpodo_hardware/set_digital_output` 서비스로 요청한다. 서비스는
문자열 `eval`이 아니라 rbpodo의 `set_box_dout(port, High/Low)` API만
호출한다. 후보 port는 매번 확인창을 거쳐 조작할 수 있으며, 나머지
port는 별도 unlock 확인 후에만 클릭할 수 있다.
`Candidate DO all OFF`는 확인창 후 4, 8, 9, 10, 12, 13을 모두 Low로
명령한다.

## 후보 port를 식별하는 안전한 순서

1. 로봇과 Hi-COMM 용접기는 연결하되 ARC와 모션을 모두 비활성화한다.
2. Weld GUI에서 4, 8, 9, 10, 12, 13의 DI/DO 초기값을 기록한다.
3. 용접기 또는 control-box 화면에서 입력 신호 하나만 안전하게 바꾼다.
4. GUI에서 같은 순간 변경된 DI 번호를 기록한다.
5. 출력은 임의 토글하지 않는다. 실제 배선을 확인하기 전에는 가스,
   인칭, ARC 또는 다른 actuator가 작동할 수 있다.
6. 여러 번 ON/OFF를 반복해 동일 port가 따라오는 경우에만 임시 mapping을
   만든다.
7. 전장 도면 또는 도통 검사로 최종 확정한다.

API의 port 번호는 0~15이다. control-box 단자 표기가 1~16이라면 화면
번호와 API index 사이에 1 차이가 날 수 있으므로 실제 controller UI의
표기 방식을 반드시 확인한다.
