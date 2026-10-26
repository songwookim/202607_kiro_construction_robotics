# Hongik 결과: Load 4 references용

원본: `/home/irs/hongik/runs/20261015-152045_kiro_1G/kiro`

원본 pass_1..4.yaml의 START/GOAL TCP 위치·자세와 S→E 방향을 그대로 보존했습니다. 이전 P2 위치 보정이나 추가 TCP/와이어 오프셋을 용접 끝점에 적용하지 않았습니다.

대기 포즈는 기존 Scan 4-pass 정책을 사용합니다: 패스4 와이어 끝 + World Z 10mm, P2 대기는 자신의 용접점에서 World X -3mm / Z +3mm, weld_wait·weld_finish는 대기점 +Z 60mm. 관절값은 기존 패스별 자세 캡처의 IK 시드이며, 적용 시 모든 포즈를 IK로 다시 풉니다.

Welding → 4-PASS 폴더에 이 폴더를 지정하고 Load 4 references를 누른 뒤 패스를 선택·적용하세요. 로봇 이동·ARC 실행은 수행하지 않았습니다. 원본 파일 해시는 import_report.json에 보관했습니다.
