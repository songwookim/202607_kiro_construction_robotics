# Scan + 4 Passes

Welding 탭의 Sequence Builder에서 **Build Scan + 4 Passes**를 누른다.
기존 Build Full Work Cycle과 별도로, 아래 한 회의 시퀀스를 만든다.

Fake ARC 옆 **Hongik scan + estimate (A/B)** 체크로 처음의 스캔 방식을 선택한다.
체크 해제(기본)는 C→D→C 로컬 녹화·Scan 탭의 추정기·저장된 정답 보정을 사용한다.
체크하면 Scan 탭의 A/B로 홍익 `PrecisionSensing`을 호출하고 그 통합 스캔·추정 결과를 사용한다.
두 경로의 추정·보정 처리는 동일하지 않으며, 홍익 모드에는 로컬 I/O/P/J 정답 보정을 추가하지 않는다.
패스 조건, 클리너 1, ARC/STOP 처리와 실행별 로그 수집은 공유한다.
방식은 Build 시점에 저장되므로 이후 체크 변경은 이미 만든 시퀀스에 영향을 주지 않는다.

**Build Scan + 4 Passes (1 Scan)**은 처음의 정밀 스캔 한 번으로 네 패스 경로를 생성하고,
P1 → 클리너 1 → P2 → 클리너 1 → P3 → 클리너 1 → P4를 수행한다.
패스 사이와 마지막의 재스캔은 없다. 기존 **Build Scan + 4 Passes**는 재스캔 네 번을 유지한다.
두 버튼 모두 위의 스캔 방식 체크를 따른다. 홍익 내부의 정밀도 확보를 위한 측정 횟수는
서버 알고리즘이 결정하며, 1 Scan은 PrecisionSensing 호출이 처음 한 번이라는 뜻이다.

빌더의 Fake ARC/Hongik 선택 옆 **Traverse (START touch · 0.5 mm/s)** 체크박스로
시작점 터치 탐색을 선택한다(기본 ON). 체크하면 각 패스는 ARC 전에 **시작점 터치로 높이를 측정**한다.
해제하면 추가 터치 탐색·실행 중 Z 등록 없이 기존 스캔/정답 보정 경로를 사용한다.
선택은 Build 시 네 패스에 저장되며, 이후 체크 변경은 이미 만든 시퀀스를 바꾸지 않는다.
시퀀스 표와 Build 로그에 ON/OFF를 표시한다. 로컬/홍익 및 재스캔/1 Scan 모두 적용한다.
현재 TCP를 필요한 만큼 위로 올려 안전 높이에서 START의 XY와 실제 용접 자세로 정렬하고,
**World −Z, 0.5mm/s**로 내려간다. 탐색 하한은 기존 보정된 START보다 **10mm 아래**다.
생산용 Fastech 터치 입력(현재 DI4, 기존 `/touch/contact`)과 DO0 감지 출력을 사용한다.
ARC OFF의 신선한 용접기 응답, Fastech 연결, 탐색 시작 전 LOW가 필요하다. 새로운 상승 에지에서
TCP를 래치하고 기존 터치 정지·정지 확인·컨트롤러 복구 절차를 사용한다. 바운스로 첫 측정값을
덮지 않는다. 입력 연결 상실·STOP은 진행 중 모션을 취소한다.

감지 뒤 DO0 OFF를 확인하고, 측정 Z와 기존 START Z의 차이를 **이번 패스 START·GOAL 및 전체
용접/크레이터/후퇴 경로에 동일하게 적용**한다. 이후 패스는 자체 터치로 측정한다. 원본 정답,
저장된 수동 XYZ 보정, 패스 상대 배치 및 자세 파일은 수정하지 않는다. 보정 경로를 ARC 전에
재계획하며, 감지 없음·초기 HIGH/미확인 입력·정지 확인 실패·출력 OFF 실패 시 ARC를 보내지 않는다.
Traverse ON에서는 Fake ARC도 실제 터치가 필요하며, Plan은 경로만 계획하고 센서 출력·이동·높이 적용을 하지 않는다.
시퀀스 표에 탐색 속도·거리와 높이 보정을 표시한다. 새 기능 적용에는 GUI 재시작과 다시 Build가 필요하다.
실행 폴더의 `summary.json` artifacts와 `events.jsonl`에 `start_touch_registration`으로 패스 번호,
기존 START·터치 TCP·적용 Z 이동량을 저장하고, 용접 피드백 조건에도 보정된 경로와 측정 정보를 남긴다.
검증: 터치/시퀀스/STOP/TCP 보정/위빙 관련 7개 테스트 파일 162개 통과. 실제 journal 저장과
저장 실패 시 ARC 차단을 추가 확인한 터치·cycle 로그 회귀 40개도 통과했다(중복 포함).
콘솔 진입점 7개 import 및 construct_robot 오버레이 빌드 완료. 실제 하강·터치·ARC는 실행하지 않았다.

로컬 방식에서는 버튼을 누르면 저장된 정답 보정을 자동 적용하고, 그 보정에 저장된 상대 배치
(현재 `measured_layout`)를 선택한다. Scan 탭의 Apply correction 체크나 배치 콤보를 따로 설정할 필요 없다.
새로 Build한 시퀀스에는 이 설정이 저장되며, 이후 체크박스를 변경해도 그 시퀀스의 보정은 유지된다.
정답 보정이 없거나 새 캡처가 아직 미반영이면 보정 계산을 마칠 때까지 Build를 중단한다.
자동 적용 변경 검증: `test_scan_teaching.py`, `test_scan_weld_cycle.py`, `test_post_weld_scan.py`,
`test_scan_truth_capture.py` 총 76개 통과 및 construct_robot 오버레이 빌드 완료.

| 순서 | 동작 | 전류 A | 전압 V | TCP 속도 mm/s | 위빙 |
|---|---|---:|---:|---:|---|
| 1 | PrecisionSensing 정밀 스캔 | | | | |
| 2 | 패스 1 용접 | 190 | 24.5 | 5 | OFF |
| 3 | 토치 클린 → PostWeldScan 원자료 스캔 | | | | |
| 4 | 패스 2 용접 | 230 | 26.5 | 7 | OFF |
| 5 | 토치 클린 → PostWeldScan 원자료 스캔 | | | | |
| 6 | 패스 3 용접 | 230 | 26.5 | 7 | OFF |
| 7 | 토치 클린 → PostWeldScan 원자료 스캔 | | | | |
| 8 | 패스 4 용접 | 220 | 26.0 | 5 | ON |
| 9 | PostWeldScan 원자료 스캔 | | | | |

Build 시점의 핫스타트와 크레이터 설정을 네 패스에 동일하게 복사한다.
전류 비율로 설정한 핫스타트/소프트웨어 크레이터는 각 패스의 본 전류를
기준으로 기존 방식대로 계산한다. 패스 4는 Build 시 **sine, 진폭 ±4mm(전체 폭 8mm),
피치 4mm, 좌우 드웰 각각 0.2초** 프리셋을 적용하며 GUI에도 같은 값을 표시한다.
재스캔 버전과 1 Scan 버전에 모두 적용한다. 로컬 Scan + 4 Passes의 위빙 방향은 새 스캔의 시편 평면 법선과
보정된 START→GOAL 방향의 외적으로 구한다. 홈 중심선을 기준으로 평면 안에서 좌우로
움직이며, GUI의 World/Tool 축 선택보다 이 측정 방향을 우선한다. Plan은 저장된 teaching의
시편 법선을, Execute는 이번 스캔의 법선을 사용한다. 원자료 스캔의 속도·방향·도구 자세·확인 모드는
Scan 탭의 Post-weld raw scan 설정을 사용한다. 이 시퀀스에서 원자료 스캔은 실제
측정이며, 별도의 Post-weld dry run 체크박스는 시퀀스의 Execute에 적용하지 않는다.

Build 전에 Scan 탭의 A/B 심 끝점과 Welding 탭의 현재 4패스 teaching을 준비한다.
각 패스에 START WAIT/START/GOAL WAIT/GOAL 및 weld_wait/weld_finish가 있어야 한다.
패스 사이의 토치 클린은 현재 teaching에서 **클리너 1만** 수행한다(패스 1·2·3 뒤 총 세 번).
클리너 1의 위치 이동·ON/OFF·inch forward 순서를 유지하고 start로 복귀한다.
Build만 누르면 장비는 움직이지 않는다.

**Plan All (teaching preview)**는 저장된 현재 teaching으로 용접 동작을 계획한다.
PrecisionSensing/PostWeldScan 목표나 장비 출력을 보내지 않는다. 첫 스캔 자체의
실제 경로를 미리 계획하는 기능은 아니며, 실행 경로는 새 측정 결과에 따라 달라진다.

**Execute**는 기존 연결 검사와 장비 실행 확인 후 첫 정밀 스캔을 수행한다.
네 패스가 모두 성공적으로 해석되고 결과 파일을 읽을 수 있을 때만 용접 경로를 만든다.
새 측정의 wire-tip 좌표와 각 패스에 저장된 토치 자세를 사용하며, 결과 메타데이터의
TCP→wire 길이를 한 번 적용해 right_manipulator_ee_point 좌표로 변환한다.
WAIT/finish는 각 패스의 현재 teaching에서 용접 끝점까지의 오프셋을 유지한다.
A/B 방향과 pass teaching 방향이 반대이면 실행을 중단한다.
이전 측정에 대한 보정 좌표는 새 wire-tip 측정으로 대체된다.

기존 Fake ARC 선택 상태를 유지한다. 실용접은 **Fake ARC OFF**에서 실행한다.
각 패스의 기존 ARC 성립 확인, 경로 사전 계획, 크레이터, ARC OFF, 안전 퇴피,
충돌 검사 및 STOP 처리를 그대로 사용한다. 원자료 스캔은 ARC OFF/퇴피와
토치 클린이 끝난 뒤 별도 순서로 실행한다. 실패하면 이후 패스를 실행하지 않는다.
확인 모드 `each`는 Scan 탭의 확인 버튼 또는 서버 confirm_move 서비스를 사용한다.

네 번의 원자료 스캔은 모두 **이번 실행의 첫 정밀 스캔 run_dir**에 연결하며,
각 결과는 `<run_dir>/post_weld_scan_<timestamp>/`에 저장된다. 용접 피드백에는
실제로 생성한 패스별 조건과 측정 TCP 경로, run_dir을 기록한다.
시퀀스 일부만 실행하거나 행을 삭제·순서 변경한 경우에는 전체 시퀀스를 다시 만든다.

GUI 모듈은 기존 워크스페이스 출력 대신 `~/hongik_field_ws` 오버레이에 빌드했다.
새 터미널에서 다음 환경을 불러온 뒤 기존 GUI 런치를 다시 실행한다.

```bash
source ~/ros2_ws/src/construct_robot_ros2/tools/activate_hongik_field.bash
ros2 launch construct_robot weld_action_gui.launch.py
```

실물 로봇/아크 실행 검증은 수행하지 않았다.

## 패스별 정답 한 점으로 보정

Scan 탭의 **New P1-P4 reference set (current scan)**으로 현재 표시된 스캔에 대한 새 정답 세트를 시작한다.
스캔과 정답 캡처는 같은 시편 위치에서 수행한다. 이전 정답·보정·패스 자세는 `scan_teaching/history/`에
백업되며, 현재 유효한 보정은 Compute 전까지 바뀌지 않는다.

오른팔 Keyboard Teaching을 켜고 **Scan 탭의 reference 모드**에서 각 패스의 START 정답을 잡는다.
키는 P1=**I**, P2=**O**, P3=**P**, P4=**J**다. 각 패스의 측정 TCP 위치·자세·관절 시드를 별도로 저장한다.
Welding 탭의 I/O/P/J는 기존 용접 티칭 키이므로 이 정답 세트에 저장되지 않는다.

네 점 저장 후 **Compute correction**을 누른다. P1 루트 보정과 패스별 오차를 따로 구하며,
각 패스 전체에 그 패스의 새 이동량을 적용한다. 이전 보정의 START–GOAL 관계를 유지한다.
네 측정점으로 루트 대비 상대 배치도 학습하고 `measured_layout`을 사용한다.
측정 시의 토치 자세는 와이어 끝 좌표 변환에만 사용하며, 기존 패스별 용접 각도와 IK 시드는 유지한다.
P2를 루트 기준으로 삼지 않는다. 미측정 GOAL은 추정값으로 표시하며, 방향은 한 점으로 새로 측정하지 않는다.
**Write 4-pass teaching → load in Welding 4-PASS**를 누르고, 기존 Scan + 4 Passes 시퀀스는 다시 Build한다.

정답을 다시 캡처하면 Compute 전까지 Write/Build가 대기한다. 원 스캔 파일 변경도 확인한다.
보정 파일에는 정답 대비 계산 오차·미측정 끝점·캡처 세트를, 새 teaching과 cycle 로그에는 사용한 세트 ID를 남긴다.
시편을 옮기고 새 정답을 잡을 때는 새 위치 스캔으로 새 세트를 시작해 이전 위치의 점과 섞지 않는다.

검증(2026-10-15): `test_scan_truth_capture.py`, `test_scan_reference.py`, `test_scan_four_pass.py`,
`test_tcp_calibration.py`, `test_scan_teaching.py`, `test_post_weld_scan.py`, `test_four_pass_correction.py`
총 120개 통과. `~/hongik_field_ws` 오버레이에 construct_robot 빌드 확인.
16:12에 이미 저장된 15:51–15:52 Welding 캡처를 I=P1/O=P2/P=P3/J=P4 정답으로 가져왔다.
새 teaching은 `scan_teaching/calibrated/20261015-161232_saved_iopj_truth/`에 저장했다.
루트 대비 측면/높이는 P2=(+4.007,+4.142), P3=(+0.835,+8.799), P4=(+0.825,+8.796)mm다.
P3/P4의 거의 같은 위치는 사용자가 확인했다. 원본 캡처·이전 보정을 백업했고 실물 이동·아크 검증은 수행하지 않았다.

2026-10-15 16:53 업데이트: 16:42:20/36/59 및 16:43:07의 GOAL P1–P4와
16:43:44의 새 P3 START를 측정 관절 FK 기록에서 확인해 반영했다. 이제 양 끝 8점 모두
측정값이며, 미측정 GOAL을 이전 경로에서 이어 쓰던 방식은 이 세트에 사용하지 않는다.
이전 P1/P2/P4 START와 패스별 용접 자세는 유지한다.
현재 teaching: `scan_teaching/calibrated/20261015-165319_goal_truths_p3_start_plate_weave/`.
캡처 원본·변경 전 보정·로그 발췌는 `scan_teaching/history/20261015-165319_before_goal_truths_p3_start/`에 보관했다.

P2/P3 **START만 World Z −2mm**를 인식 및 정답 보정 계산 **이후**에 추가한다.
`reference_set.endpoint_adjustments_world_m`에 별도로 저장하며, 정답 원본이나 상대 배치에
섞지 않는다. 다시 Compute해도 한 번만 적용되고 GOAL에는 전파되지 않는다.
새 teaching과 cycle 로그에 이 추가 조정 및 P4 평면 위빙 방향을 남긴다.

토치 장착 변경 후 사용자가 확인한 현재 추가 위치 조정(World 기준):
P1/P3 START·GOAL X −2mm, **P2 START X −0.5mm / GOAL X −1mm**,
**P4 START·GOAL X −1mm 및 Z −1.5mm**.
P2/P3 START의 Z −2mm도 유지한다. 원본 정답을 옮기지 않고
`endpoint_adjustments_world_m`에 최종 조정값을 저장한다(이전 X −2mm에 누적하지 않는다).
새 C/D Scan + 4 Passes는 저장된 보정을 자동 사용한다.
반복 계산 시 누적하지 않으며, 이 수동 위치 조정이 새 토치의 각도나 TCP→와이어팁 모델을
재측정한 것은 아니다. 이미 생성된 시퀀스와 과거 실행 자료에는 소급 적용하지 않는다.

루트 일치만으로 전체 패스가 검증되지는 않는다. P1 루트 등록, 패스별 상대 배치,
START/GOAL 측정 여부, 측정 자세에서 실제 용접 자세로의 TCP→와이어 끝 변환이 각각 필요하다.
이번 새 GOAL과 기존 계산의 차이는 캘리브레이션 기준 스캔에서 P2 높이 −5.46mm,
P3 높이 −4.31mm였다. 새 P3 START의 와이어 끝은 이전 정답보다 높이 −4.61mm 바뀌었다.
이는 기존 정답과 미측정 끝점의 차이이며, 시편 이동으로 단정하지 않는다.

와이어 끝은 `TCP + R(자세) × tip_tcp`로 비교한다. 같은 와이어 끝을 가리켜도 자세가 다르면
TCP XYZ는 달라진다. 와이어 벡터 오차 Δv가 있을 때 자세 변경 후 남는 오차는
`(R_측정 − R_실행) × Δv`이므로, 같은 자세의 루트에서는 상쇄돼도 다른 자세에서는 남을 수 있다.
이번 캡처는 P4 자세이고 P1/P2/P3 실행 자세와 약 12° 다르다. 기존 wire-tip 보정의 RMS는
약 1.60mm이며, 이번 정답 재투영 오차가 0이라는 사실은 실제 와이어 길이나 손눈 보정의
정확성을 검증하지 않는다. 실제 와이어 길이가 바뀌었다면 동일 길이에서의 실물 확인이 필요하다.

검증: 정답 8점·자세 변환·START 추가 조정·평면 위빙·기존 시퀀스 관련 9개 테스트 파일에서
178개 통과. 위빙 방향의 피드백 로그 저장과 기존 실행기 회귀 58개도 통과했다(앞 검사와 중복 포함).
새 네 파일을 실제 4-PASS 로더로 읽었으며 모션/IK 호출이 없음을 확인했다.
콘솔 진입점 7개 import와 `~/hongik_field_ws`의 construct_robot 오버레이 빌드도 완료했다.

## 실행 로그

한 실행의 자료는 `~/ros2_ws/weld_feedback/scan_four_pass_<초기스캔폴더명>/`에 자동으로 모인다.
예: `scan_four_pass_20261015-172640_scan/`.

- `npz/initial_scan.npz` 및 `.json`: 초기 스캔과 메타데이터.
- `npz/pass_1_rescan.npz` ~ `pass_4_rescan.npz` 및 각 `.json`: 패스별 재스캔.
- `npz/pass_1_rescan.ply` ~ `pass_4_rescan.ply`: World XYZ(m), binary little-endian float32 포인트클라우드.
- `pass_1_weld_feedback.log` ~ `pass_4_weld_feedback.log`: TCP·실측 관절·용접 피드백.
- `summary.json`, `events.jsonl`, `cycle.log`: 같은 실행의 전체 기록.
- `pass_1.yaml` ~ `pass_4.yaml`: 생성된 패스 teaching.
- `manifest.json`: 모은 파일의 상대 경로·원본 경로·SHA-256 및 수집되지 못한 artifact.

원본 스캔과 피드백 파일을 유지하면서 검증된 복사본을 만든다. 실행 중 생성되는 파일부터
순서대로 모으며, STOP/실패 시에도 이미 저장된 자료를 남긴다. 종료 뒤 비동기로 저장되는
피드백도 같은 폴더와 manifest에 반영한다. 수집 중 쓰기 실패는 기존 로그 오류 처리로 보고한다.
실행 로그도 이 폴더에서 한 벌만 기록한다. 스캔명이 정해지기 전에는
`scan_four_pass_<실행시각_ID>/`에 시작하고, teaching 생성 뒤 로그를 스캔명 폴더로 옮긴다.
앞의 임시 폴더는 제거하며 이후 이벤트·요약·늦게 저장된 피드백은 최종 폴더에 기록한다.
같은 파일명을 가진 다른 실행 로그가 있으면 덮어쓰지 않고 실패를 보고한다.
티칭 history에는 네 YAML만 보관하여 재생성 시 NPZ·실행 로그가 중복 복사되지 않게 한다.
변경 적용에는 GUI 재시작이 필요하다.
재스캔 PLY는 녹화 뒤에 생성한다. C/D 방식은 NPZ의 `points_world_m`을 그대로 내보내며,
홍익 방식은 `raw.npz`의 시각·실측 로봇 pose·기록된 보정을 홍익 재구성 함수로 처리한다.
PLY는 결과 폴더에 직접 저장해 별도의 큰 PLY 복사본을 만들지 않는다.
독립 Post-weld scan은 원본 폴더의 `scan.ply`에 저장한다. 변환 실패 시 원자료를 유지하고
실패를 보고하여 다음 패스를 진행하지 않는다. 1 Scan 버전은 재스캔 및 재스캔 PLY가 없다.
스캔 방식·1 Scan·PLY 변경 검증: `test_scan_weld_cycle.py`, `test_scan_cloud.py`,
`test_post_weld_scan.py`, `test_scan_cycle_log.py`, `test_sequence_executor.py`,
`test_scan_four_pass.py`, `test_weld_weave.py` 131개 통과. 추가한 홍익 호출 경로 검증을 포함해
`test_scan_weld_cycle.py` 25개도 통과했다(중복 포함). 기존 홍익 raw 834,934점을 오프라인
재구성해 저장된 recon과 최대 약 3.7e-8 m 차이를 확인했고 RAM에 PLY를 저장했다.
콘솔 진입점 7개 import 및 construct_robot 오버레이 빌드 완료. 실제 로봇 동작은 실행하지 않았다.
자동 수집 검증: `test_scan_cycle_log.py`, `test_weld_execution_controller.py`,
`test_scan_weld_cycle.py`, `test_scan_four_pass.py` 총 65개 통과.
기존 실행의 모은 파일 21개 SHA-256 확인, 콘솔 진입점 7개 import 및 construct_robot 오버레이 빌드 완료.

`scan_four_pass_cycles/`를 별도로 만들지 않는다.
`~/ros2_ws/weld_feedback/latest_scan_cycle.txt`에는 가장 최근 실행의 통합 폴더 경로가 들어 있다.
과거 22회 실행의 분리된 로그 66개도 원본 해시를 확인해 각 결과 폴더로 병합했다.
이동·중복 제거 내역은 `weld_feedback/log_consolidation_20261015_235558.json`에 보관한다.

- `summary.json`: 전체 COMPLETED/FAILED/STOPPED 상태, 요청 조건과 실제 생성된
  패스별 TCP 경로·속도·위빙·핫스타트·크레이터, 단계 결과, 미수행 단계, 결과 파일 경로.
- `events.jsonl`: 단계 시작/종료·소요시간·실패 사유·클리너 출력 OFF 정리 결과를
  실행 중 순차 기록하고 flush/fsync한다. 비정상 종료 시에도 기록된 줄은 따로 읽을 수 있다.
- `cycle.log`: 같은 실행 이벤트의 텍스트 로그.

현재 로컬 스캔 방식과 Hongik 서버 방식 모두 첫 스캔·패스별 원자료 스캔·4패스
teaching 폴더 및 패스별 `weld_feedback_*.log` 경로를 artifacts에 연결한다.
용접 피드백 로그에는 기존 RX 전류/전압, TX 프레임, 실제 TCP trajectory 기록이 남고,
각 패스에 해당하는 속도·위빙 조건과 전체 cycle 로그 폴더를 함께 기록한다.
Fake ARC도 패스별로 파일을 닫고 `fake_arc`를 표시한다.

패스별 피드백 로그의 `[joint_trajectory]`에는 피드백 세션이 활성화된 동안 수신한
실측 `/joint_states`를 저장한다. `elapsed_s`는 TCP와 같은 세션 기준 시각이며,
`joint_stamp_s`는 ROS 메시지 시각, `arm`은 left/right다. 관절 이름으로 순서를 맞춘
J1–J6 위치(rad)를 저장하며, 해당 팔의 여섯 값이 모두 유효한 메시지만 기록한다.
목표값/IK 시드와 구분하며, Fake ARC에도 동일하게 적용한다. cycle의 `weld_feedback`
artifact에도 TCP/관절 샘플 수를 남긴다. 변경 전 실행의 실측 관절 궤적은 소급 생성하지 않는다.
관절 기록 검증: `test_weld_execution_controller.py`, `test_weld_quality_metrics.py`,
`test_touch_stop_safety.py` 43개 통과; 실행기와 `test_scan_cycle_log.py` 회귀 31개 통과(중복 포함).
콘솔 진입점 7개 import 및 `~/hongik_field_ws`의 construct_robot 빌드 완료. 실물 실행은 수행하지 않았다.

Plan/Build는 실행 로그를 만들지 않는다. 종료 기록 없는 RUNNING 상태는 비정상 종료
가능성이 있으므로 events.jsonl의 마지막 이벤트를 확인한다. 로그 폴더 생성에 실패하면
첫 스캔을 시작하지 않으며, 실행 중 기록 실패 시 기존 출력 OFF/모션 취소 절차로 중단한다.
초기 로그 루트는 `KIRO_SCAN_CYCLE_LOG_ROOT` 환경 변수로 바꿀 수 있다. 이전 설정값이
`.../scan_four_pass_cycles`라면 부모 폴더로 정규화해 분리된 로그 폴더를 다시 만들지 않는다.

## C/D 스캔 후 추정 계산 시간

2026-10-15: 프로파일별 중앙값은 프레임별로 점을 묶어 계산하고, 선 피팅 SVD는 사용하지 않는
큰 U 행렬을 만들지 않도록 변경했다. 입력 점, 중앙값, RANSAC 반복 수·문턱, 품질 검사는 유지한다.
최근 `20261015-180308_scan`의 2,177,479점에서 `estimate_groove`만 동일 조건으로 비교한 결과
24.49초 → 5.62초였고 모든 출력 필드가 일치했다(최대 수치 차이 0).
스캔 모션·파일 저장·보정·티칭 생성·IK 시간은 이 측정에 포함하지 않았다. 홍익 서버의 추정 방식에는
적용되지 않는다. 원본 NPZ 해시와 전후 결과는
[`artifacts/kiro_estimator_speed_20261015.json`](artifacts/kiro_estimator_speed_20261015.json)에 보관한다.

추가 개선: RANSAC의 2차원 거리 계산에서 반복적인 N×2 임시 배열·BLAS 호출을 제거하고,
최다 inlier 개수를 저장해 같은 배열을 매번 다시 세지 않는다. RNG 추출 순서, 300회 반복,
입력 점, 피팅·품질·커버리지 기준은 유지한다. `20261015-215016_scan`의 2,378,606점으로
전체 추정 함수를 오프라인 측정한 결과 15.65초 → 7.71초였다. 생성 시각을 제외한 추정 JSON,
KIRO pass/info 8개 YAML 및 최종 teaching 4개 YAML의 수치 1,067개는 모두 동일했다(최대 차이 0).
동시 시스템 부하가 있는 RAM 출력·cProfile 측정이며 실제 사이클 전체의 단축 시간을 뜻하지 않는다.
최근 21:49 실행의 실제 경로 준비 67.5초는 스캔·저장 50.5초, 추정 10.2초, 이후 보정·준비 6.8초였다.
스캔 경로·왕복 횟수·속도는 변경하지 않았다. 비교 기록:
[`artifacts/kiro_estimator_ransac_speed_20261015.json`](artifacts/kiro_estimator_ransac_speed_20261015.json).
`test_groove_estimate.py`, `test_kiro_estimate.py`, `test_scan_four_pass.py`,
`test_scan_truth_capture.py` 37개 통과. 실제 로봇 동작은 검증 과정에서 실행하지 않았다.


### No Scan + 4 Passes

`Build No Scan + 4 Passes` loads the most recently saved four-pass teaching folder
under `weld_feedback/scan_four_pass_*`, then freezes its current START/GOAL, waits,
orientations and measured plate-plane weave direction at Build. The folder is
shown in the sequence and Build log. It performs no initial scan, estimate or
post-pass rescan, and does not apply the saved calibration a second time.
It retains the four production recipes, Cleaner 1 between passes, Fake ARC and
Traverse selections. Build/Plan do not move the robot or enable ARC; Execute uses
the existing managed workflow and safety checks. Use this only while the workpiece
and setup still match the saved scan. Missing files/plate normal or changed source
YAML hashes require rebuilding and block execution. Each execution records a new
result folder with copies of the source pass YAMLs and source hashes; previous
scan execution logs remain unchanged.
