# 이미 저장된 I/O/P/J 캡처 기반 보정

2026-10-15 15:51–15:52 Welding 탭에서 이미 저장한 네 측정값을 가져왔습니다. 다시 캡처하지 않았습니다. I=P1, O=P2, P=P3, J=P4. P3/P4가 거의 같은 위치인 것은 사용자가 확인했습니다.

같은 시편 위치의 15:35:17 KIRO 스캔을 사용하고 P1 루트 기준을 유지합니다. 실측 네 점에서 루트 대비 상대 배치(measured_layout)를 구했습니다. 미측정 GOAL은 이전 경로의 끝점 관계를 유지해 이동합니다. 측정 자세는 와이어 끝 변환에 사용하며 기존 패스별 용접 자세·IK 시드·와이어 TCP 캘리브레이션은 보존했습니다.

GUI 재시작 후 이 폴더에서 Load 4 references를 누르세요. 다음 Scan + 4 Passes는 Apply correction과 measured_layout으로 다시 Build하세요. 이동·ARC는 실행하지 않았습니다.

원본 캡처: `/home/irs/ros2_ws/src/construct_robot_ros2/construct_description/config/scan_teaching/history/20261015-160929_received_iopj_truth`

보정 전 백업: `/home/irs/ros2_ws/src/construct_robot_ros2/construct_description/config/scan_teaching/history/20261015-161231-818284_22cb4992_before_four_pass_references`
