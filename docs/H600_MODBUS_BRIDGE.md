# H600 Modbus TCP bridge 이해하기

## 한 문장 요약

`h600_modbus_bridge`는 PC의 TCP 502 포트에서 기다리는 **Modbus TCP
server**이다. H600이 client로 접속해 명령 레지스터 201~210을 읽고,
상태 레지스터 211 이후를 쓰면 bridge가 이를 ROS topic/service로
변환한다.

```text
Weld GUI / Cartesian action
        |
        |  /h600/set_command (ROS service)
        v
 h600_modbus_bridge
   command register image 201..210
        ^                         |
        | FC03 read               | FC06/FC16 write
        | TCP port 502            | TCP port 502
        +----------- H600 --------+
                         |
                         v
             /h600/status, /h600/traffic
```

여기서 server는 로봇을 지시하는 쪽이라는 뜻이 아니다. TCP 연결을
기다리고 Modbus 요청에 응답하는 쪽이라는 뜻이다. 실제 요청은 H600이
보내므로 H600이 Modbus client이다.

## 실행

다음 한 명령이 bridge와 GUI를 함께 실행한다.

```bash
source /opt/ros/humble/setup.bash
source /home/irs/ros2_ws/install/setup.bash
ros2 launch construct_robot h600_console.launch.py
```

Linux에서 502는 privileged port이므로 launch는 bridge에만 `sudo -E`를
적용한다. 같은 터미널에 sudo 암호 프롬프트가 나타난다. GUI는 일반
사용자 권한으로 실행된다. root bridge와 사용자 GUI 사이의 Fast DDS
shared-memory 권한 충돌을 피하기 위해 두 프로세스 모두 UDPv4 transport를
사용한다. 이 설정이 없으면 ROS graph에는 topic/service가 보이면서도
status, traffic 및 service 응답이 GUI에 전달되지 않을 수 있다.

이미 별도로 bridge를 실행했다면 중복 실행하지 않는다.

```bash
ros2 launch construct_robot h600_console.launch.py start_bridge:=false
```

## 레지스터 맵

주소는 코드와 H600 패킷에 들어가는 raw holding-register 주소이다.
일부 Modbus 문서의 `4xxxx` 표기와 혼동하지 않는다.

| 주소 | 방향 | 의미 |
|---:|---|---|
| 201 bit0 | PC → H600 | Robot ready |
| 202 bit7 | PC → H600 | Robot error |
| 202 bit4 | PC → H600 | Touch |
| 202 bit3 | PC → H600 | Gas |
| 202 bit2 | PC → H600 | Reverse inching (`0x0004`) |
| 202 bit1 | PC → H600 | Inching (`0x0002`) |
| 202 bit0 | PC → H600 | ARC (`0x0001`) |
| 204 | PC → H600 | Current setpoint raw |
| 205 | PC → H600 | Voltage setpoint raw |
| 206 | PC → H600 | Voltage offset raw |
| 211 bits15..8 | H600 → PC | Heartbeat |
| 211 bit7 | H600 → PC | Welder error |
| 211 bit5 | H600 → PC | Welding |
| 211 bit4 | H600 → PC | Touch detected |
| 211 bits1..0 | H600 → PC | Welder info |
| 212 | H600 → PC | Current feedback raw |
| 213 | H600 → PC | Voltage feedback raw |
| 216 | H600 → PC | Single candidate/status register |

정인칭만 켰다면 202는 `0x0002`, 역인칭만 켰다면 `0x0004`여야 한다.
GUI는 둘을 동시에 켜지 못하게 한다.

## Modbus TCP 패킷 구조

한 프레임은 `MBAP header + PDU`로 구성된다.

```text
MBAP (7 bytes)
  Transaction ID  2 bytes  요청과 응답을 짝지음
  Protocol ID     2 bytes  Modbus TCP는 0
  Length          2 bytes  뒤에 남은 byte 수
  Unit ID         1 byte   장치/게이트웨이 식별

PDU
  Function code   1 byte
  Function data   N bytes
```

bridge가 지원하는 function code는 다음 세 가지다.

- `FC03`: H600이 holding register를 읽는다.
- `FC06`: H600이 register 하나를 쓴다.
- `FC16`: H600이 여러 register를 한 번에 쓴다.

예를 들어 H600이 201부터 10개를 읽으면 요청 PDU는 개념적으로
`03 00 C9 00 0A`이다. `0x00C9`가 201이고 `0x000A`가 10이다. bridge는
201~210의 20 byte 값을 big-endian uint16 배열로 돌려준다.

## 코드가 처리하는 순서

1. `_start_server()`가 socket을 만들 별도 thread를 시작한다.
2. `_serve()`가 `bind(0.0.0.0, 502)`, `listen()`, `accept()`를 수행한다.
3. H600 연결 후 `_handle_client()`가 MBAP 7 byte와 PDU를 정확한 길이만큼
   읽는다.
4. `H600Protocol.process_pdu()`가 FC03/06/16을 해석한다.
5. 요청의 Transaction ID와 Unit ID를 그대로 넣어 응답한다.
6. 모든 RX/TX frame을 `/h600/traffic`에 HEX와 함께 게시한다.
7. 0.2초 timer가 register 211~213과 연결 상태를 `/h600/status`에
   게시한다.

`H600State`는 socket thread와 ROS callback이 동시에 접근하므로
`threading.RLock`으로 보호된다.

## ROS 인터페이스

| 이름 | 타입 | 용도 |
|---|---|---|
| `/h600/set_command` | service | 201/202/204/205/206 명령 이미지 변경 |
| `/h600/set_server` | service | TCP/502 listener 시작/정지 및 bind 주소 설정 |
| `/h600/get_registers` | service | GUI용 register snapshot |
| `/h600/status` | topic | 연결, 명령, 211~213 decode 상태 |
| `/h600/traffic` | topic | Wireshark 형태의 RX/TX frame 기록 |

GUI 체크박스는 `/h600/set_command`를 호출할 뿐 TCP 패킷을 직접 보내지
않는다. 값은 bridge 메모리에 저장되고, H600이 다음 FC03 요청을 보낼
때 응답에 포함된다. 따라서 GUI가 `202=0x0002`를 표시해도 H600이 FC03
요청을 하지 않으면 장비에는 전달되지 않는다.

## TCP1 → TCP2 용접 실행 순서

`weld_action_gui.launch.py`는 privileged H600 bridge를 상시 실행하고,
사용자 launch 시작과 함께 올라오는 MoveIt stack은 그 bridge를 공유한다. 별도의
두 번째 bridge를 실행하지 않는다.

**H600 ARC during execution**을 켜고 Plan Preview를 수행하면 다음 두
trajectory가 함께 계획되고 승인된다.

1. 현재 로봇 자세 → TCP1: ARC OFF 접근 trajectory
2. TCP1 → TCP2: 용접 seam trajectory

Execute Approved Plan의 실제 순서는 다음과 같다.

```text
ARC/ready/gas OFF
  → TCP1 접근 실행
  → H600 연결 확인
  → ready + gas + setpoint, ARC OFF (pre-flow)
  → ARC ON
  → 선택 시 register 211 welding bit 확인
  → TCP1에서 TCP2로 seam 실행
  → ARC OFF
  → 선택 시 welding bit OFF 확인
  → post-flow
  → ready/gas/setpoint 모두 OFF
```

H600 연결 또는 필수 welding feedback이 없으면 seam trajectory는 시작하지
않는다. Plan Preview에서는 실제 H600 출력이 바뀌지 않는다.

## 안전 동작

- ARC는 `allow_arc_output:=true`, H600 연결, Robot ready가 모두 필요하다.
- nonzero 전류/전압은 launch와 요청 양쪽 안전 잠금을 풀어야 한다.
- H600 연결 해제, server 정지, node 종료 시 ready/gas/ARC/setpoint를
  모두 0으로 만든다.
- GUI에서 정인칭과 역인칭은 상호 배타적이다.

## 인칭 디버깅 순서

1. GUI 상단이 `LISTENING 0.0.0.0:502`인지 확인한다.
2. H600 상태가 `CONNECTED`인지 확인한다.
3. Inching을 켜고 `202 Command raw = 0x0002`인지 확인한다.
4. traffic에서 H600의 `FC03 registers 201..210` RX와 TX를 확인한다.
5. TX raw frame의 202 위치가 `00 02`인지 확인한다.
6. 여기까지 맞는데 동작하지 않으면 H600의 Remote mode, Robot ready,
   welding/ARC 상태 및 장비측 inching interlock을 확인한다.
