# 코드 정리 — 2026-10-02

범위: 현재 Tk 생산 코드, application/core/io/nodes, launch, 패키지 의존성,
테스트 및 호출 관계. 기존 미커밋 변경과 이전 감사 수정 위에서 작업했다.
장비·ROS 노드를 실행하거나 실제 출력/이동 명령을 보내지 않았다.

## 적용한 정리

| 발견한 문제 | 적용 내용 |
| --- | --- |
| 파일 원자 저장 구현 9곳 중복 | `io/atomic_file.py`의 `atomic_text_writer()` / `atomic_yaml()`로 통합. `contextlib.contextmanager`, `tempfile`, `Path.replace()`를 사용하며 실패 시 `unlink(missing_ok=True)`로 정리한다. |
| 일부 저장 경로의 임시파일 정리 누락 | dump/쓰기/교체 실패 시 기존 파일 보존 및 임시파일 제거를 테스트했다. 초기 교시와 선택 pass 저장의 read-back 검증은 유지했다. |
| 서비스마다 Event·결과 dict·중첩 callback 반복 | `weld_runtime_node.py`의 서비스 응답 대기 6곳을 기존 `wait_for_future()`로 통합. discovery/response timeout, 오류 문구, 컨트롤러 readback 확인은 유지했다. |
| 리팩터링 뒤 남은 미사용 import | GUI import 31개 제거. 기존 테스트·진단 코드가 사용하는 재노출 함수는 유지하고 `__all__`로 의도를 명시했다. |
| 덮어쓰는 딕셔너리 키·미사용 계산 | Hot Start 지표의 중복 `requested_current_a`, `requested_boost_percent` 키와 쓰이지 않는 dwell/signature 계산 제거. 최종 지표 값은 동일하다. |
| 반복적인 launch 파라미터 선언 | `(파라미터, launch 인자, 타입)` 표를 기존 ROS `ParameterValue` API로 변환. 18개 파라미터의 이름·타입·기본값 및 override 전달을 테스트했다. |
| 직접 import하지만 누락된 의존성 | package.xml에 `action_msgs`, `builtin_interfaces`, `rosidl_runtime_py`, `python3-matplotlib` 선언 추가. 설치 작업은 하지 않았다. |
| 잘못된 주석 | 0.01 s TCP 로그 polling을 50 Hz로 설명하던 주석을 100 Hz로 수정. 실제 주기는 변경하지 않았다. |

이번 작업 시작 시점 대비 생산 Python은 **192줄 순감소**했다
(공통 모듈 추가분 포함, launch 포함, 테스트·문서 제외).
코드 중복/관리 지점 감소 수치이며 처리속도 개선을 측정한 수치는 아니다.

## 라이브러리로 바꿀 것 / 유지할 것

- 파일 저장은 표준 라이브러리 자원 관리 API로 통합했다. `Path.write_text()`만
  사용하면 기존 파일이 중간 상태로 보이므로, 원자 교체 자체는 작은 공통 함수가 필요하다.
- YAML 읽기는 기존 `yaml.CSafeLoader`를 유지했다. 쓰기 dumper는 바꾸지 않아
  출력 형식이나 파일 기반 provenance/hash 처리를 불필요하게 흔들지 않는다.
- ROS Future는 `concurrent.futures.Future`와 동일한 timeout API가 아니다.
  이미 별도 executor가 도는 노드를 다시 spin하는 방식 대신 완료 callback과
  Event를 재사용했다. 서비스 timeout이 원격 명령 취소를 의미하지는 않는다.
- 통계는 이미 `statistics`, 키보드 운동학은 NumPy/PyKDL을 사용한다.
  짧은 벡터 연산을 지우기 위해 SciPy 같은 새 의존성을 추가하지 않았다.
- Fastech는 기존 벤더 어댑터를, Hi-COMM은 현재 검증된 프레임 인코딩을 유지한다.
  일반 네트워크 프레임워크가 장비 프로토콜의 대체재인 것은 아니다.

## 확인했지만 보류한 부분

- `task_teaching_model.py`의 옛 Task Library codec/order/path helper는 현재
  생산 호출 없이 테스트에 남아 있다. 저장된 형식 지원을 제거하는 일이므로
  단순 미사용 import와 달리 이번 정리에서 삭제하지 않았다.
- TorchCleanerPanel의 옛 수동 `prepare/next/seed` 경로는 현재 버튼에 연결되지
  않지만 STOP의 `abort/outputs_off` 경로와 상태를 공유한다. 공통 실행기로 완전히
  이관하려면 busy/capture 중 STOP 테스트를 먼저 보강해야 한다.
- GUI host에 의존하는 application 코드, 옛 직접 비동기 action 경로,
  키보드 fallback backend는 별도 동작 리팩터링 대상으로 남겼다.
- 79자 기준의 긴 행 등 기존 E/W 형식 경고는 남아 있다. 저장소 전체 자동 포맷,
  안전 상태 확인용 sleep 제거, 좌표 계산/제어 수학 교체는 수행하지 않았다.
  이번 정적 검사 통과는 **F 및 E9 기준**이며 모든 스타일 규칙 통과라는 뜻은 아니다.

## 검증

- 시작 시 전체 비-Qt 테스트: **341 passed**.
- 저장/교시/multipass/모션 경계 집중 테스트: **158 passed**.
- 서비스/품질/호환 경로 집중 테스트: **52 passed**; launch 표 검사: **1 passed**.
- 최종 전체 비-Qt 테스트: **371 passed** (6.35 s).
- `flake8 ... --select=F,E9` 및 `git diff --check`: 통과.
- 생산 console entry point 7개 import 성공. `main()`은 호출하지 않았다.
- 요청 대상 4개 패키지 colcon 빌드 성공 (19.1 s).
  `/tmp/construct-cleanup-build-VWXSGx`에 격리하여 workspace build/install/log를
  변경하지 않았다. 두 CMake 패키지의 `PYTHON_EXECUTABLE` 미사용 경고는 남아 있다.
- 실제 교시 YAML, 용접 로그, 장비 설정은 변경하지 않았다. 로봇·용접 실기 검증은 아니다.

## 후속 정리: 실행기 단일화 및 파일 수 감소

앞서 보류했던 항목을 사용자 승인 후 정리했다. 이 절의 수치는 위 작업 이후의
추가 변경이며, 기존 키보드 노드/servo 수정은 건드리지 않았다.

- 생산 호출이 없는 Task Library codec/order/path helper와 하드코딩된 클리너
  seed 자세를 삭제했다. Named Teaching과 CleanerTeachingState는 유지했다.
- 버튼에 연결되지 않은 클리너 `prepare/next/finished_motion`, seed/browse/order
  저장 경로를 삭제했다. 현재 Plan/Execute는 기존 `SequenceExecutor`를 사용한다.
  panel의 busy interlock과 capture/correction 중 abort/DO5·6·7 OFF 처리는 유지했다.
  DO0를 끄거나 진행 중인 공통 시퀀스의 상태를 panel이 해제하지 않는다.
- 클리너 단계 생성은 `application/weld_sequence_builder.py`, YAML/해시 검증은
  `io/teaching_yaml.py`, 상태/출력 token 검증은 `core/task_teaching_model.py`로 모았다.
  중복된 클리너 joint/TCP 파일 검증도 `load_cleaner_pose()`로 합쳤다.
- `core/torch_cleaner_teaching.py`, `io/teaching_paths.py`, `io/atomic_file.py`를
  삭제하고 실제 사용되는 함수는 위 기존 모듈에 통합했다. 별도 호환 shim은 남기지 않았다.
- `load_work_cycle()`의 파일 읽기는 IO, `validate_work_cycle()`은 core로 분리했다.
  `calculate_weld_production_metrics()`는 core로, `analyze_saved_weld_log()`는 IO로
  이동했다. 계산식과 로그 형식은 그대로다.
- 옛 codec 제거 후 직접 사용처가 없어진 `rosidl_runtime_py` 의존 선언을 제거했다.
- 생산 Python 파일 **39 → 36개**, **361줄 순감소**. 테스트/문서 및 이전 변경 제외.
  옛 Task Library 전용 테스트 파일도 삭제했으나, 그 파일의 살아 있는 클리너 실패
  출력 해제 테스트는 `test_sequence_executor.py`로 옮겼다.

### 왜 프레임워크를 추가하지 않았나

[`py_trees`](https://py-trees.readthedocs.io/en/release-2.2.x/composites.html)는
Sequence/Selector/Parallel을 제공한다. 다만 Parallel은 child를 순서대로 tick하며
장비 동작의 실제 병렬화/취소는 leaf 측 책임이다.
[`transitions`](https://github.com/pytransitions/transitions)는 FSM 전이/콜백용이다.
현재 실행기의 많은 부분은 ARC 이벤트 동기화, STOP, 장비별 완료 확인/출력 cleanup이다.
이 책임은 프레임워크로 바꿔도 필요하므로, 현재 범위에서는 adapter와 의존성을 추가하기보다
데이터 기반 공통 실행기를 유지하는 편이 코드 최소화에 적합하다고 판단했다.
분기/복구 정책이 크게 늘어나면 별도 전환 후보가 될 수 있다.

### 후속 검증

- 집중 테스트: **67 passed**; 전체 자동 테스트: **378 passed** (6.49 s).
  `test_hicomm_control_v4.py`(수동 Qt 도구), `rbpodo_test.py`(수동 장비 도구)는 제외.
- 교시 JSON/YAML 파일은 생성/변경하지 않았다. 테스트는 임시 폴더만 사용했다.
- 생산 소스/launch 및 이번 수정 테스트의 flake8 `F,E9`, `git diff --check` 통과.
  전체 test 디렉터리 검사에서는 두 수동 도구의 기존 unused import 경고 2개가 남는다.
- 초기 테스트는 workspace install 부재로 ROS 메시지 import에 실패했다. 필요한 메시지
  패키지와 대상 패키지를 `/tmp/construct-sequence-cleanup-uu70gy`에 격리 빌드한 후
  위 테스트를 수행했다. workspace `build/install/log`는 생성하거나 수정하지 않았다.
- 최종 생산 console entry point **7개 import 성공** (`main()` 호출 없음).
  `colcon build --packages-select construct_robot` 최종 재빌드 성공 (3.66 s).
- 모션/ARC 순서/스레드·이벤트 실행/프로토콜/제어 주기는 변경하지 않았다.
  실제 장비 검증은 수행하지 않았다.

### 계속 진행: 미사용 시퀀스 UI와 실행기 반복 제거

- 버튼/호출처가 없는 GUI 수동 단계 추가 콜백 7개와 행 이동 콜백을 제거했다.
  현재 Build/Execute/행 편집/삭제/STOP 및 편집용 Tk 변수는 유지했다.
- SequenceModel의 미사용 observer, 단일 add/edit/move/duplicate/update_fields 및
  validate wrapper를 제거했다. 현재 사용하는 snapshot/replace/extend/선택/진행/삭제는 유지한다.
- 실행기 슬롯 집계는 `collections.Counter`, 순서 유지 그룹화는 `defaultdict(list)`로
  단순화했다. 기존 최초 등장 순서, 동일 슬롯 병렬 묶음, 별도 sleep 그룹을 보존했다.
- 같은 호출 규약의 모션 5종은 명시적 허용 목록을 통해 기존 runtime 메서드로 위임한다.
  Cartesian weld의 ARC 대기와 취소, output pulse/OFF, worker thread/join은 변경하지 않았다.
- 추가 **313줄 감소**, 후속 정리 누계 **674줄/생산 파일 3개 감소**.
  전용 테스트 64개 통과 후 dispatch/그룹 순서 테스트를 더하여 최종 전체 **391 passed**
  (6.38 s), entry point 7개 import 성공. 실장비는 실행하지 않았다.
