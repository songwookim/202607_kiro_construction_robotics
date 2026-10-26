# Hongik 관절 경로 선택 — kiro-1g-field로 교체

2026-10-14의 RELAY_2026-10-14b 지시에 따라 기존 KIRO 런타임 패치를 제거하고
홍익 패키지 자체의 kiro-1g-field 갱신으로 교체했습니다.

선택은 scan_l0.joint_selection.DurationSearch의 계획 시간과 정확한 하한을 사용합니다.
확정된 후보를 다시 정렬하지 않고 그 후보의 checked plan을 실행합니다.
OMPL이 실패하면 다음 후보를 검사하며, 서비스 무응답은 후보 실패와 구분합니다.
이전의 '선택된 후보 하나의 OMPL 실패 시 전체 거부' 방식과 메모리 SHA 패치는 사용하지 않습니다.

소스 위치·overlay source 방법·실행 인자·현재 서버 상태·테스트는
[RELAY 적용 결과와 실행 안내](HONGIK_FIELD_UPDATE.md)를 따릅니다.
이전 연구실 검토용 diff는 당시 상태의 기록으로 docs/artifacts/hongik_duration_review에 보존했습니다.
