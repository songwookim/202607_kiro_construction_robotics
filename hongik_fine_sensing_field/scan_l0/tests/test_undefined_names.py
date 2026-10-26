"""생산 모듈에 정의되지 않은 이름이 없다 (pyflakes UndefinedName·UndefinedExport) — 2026-10-09 1g-integ.

왜: 두 갈래의 병합이 글자로는 깨끗해도 한 갈래가 지운 이름 (stream-robust: post_scan 의 ProcessPoolExecutor import·RESULT_TIMEOUT_S)
을 다른 갈래 (P2 추적 경로·P8 단면 법선) 가 계속 쓰면 실행 때에만 NameError·AttributeError 가 나고, 그 경로는 일부러 실패를 먹고 대체
경로로 간다 (패스를 잃지 않게) — 결과는 같아 시험이 조용히 지나고 작업 프로세스·스트림이 통째로 꺼진 채 남는다. 정적 검사로 그 부류를
병합 때 바로 잡는다 (속성 이름 — 모듈.이름 — 은 여기서 볼 수 없다: 그 경로의 시험이 작업 프로세스 경로를 확인한다)."""
from __future__ import annotations

import os

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
from pyflakes import messages as _msgs  # noqa: E402
from pyflakes.reporter import Reporter  # noqa: E402

STACK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGES = ("vc3dz_driver/vc3dz_driver", "scan_l0/scan_l0", "scan_core/scan_core",
            "hongik_scan_ros/hongik_scan_ros")
_UNDEFINED = (_msgs.UndefinedName, _msgs.UndefinedExport, _msgs.UndefinedLocal)


class _Collect(Reporter):
    def __init__(self):
        super().__init__(None, None)
        self.found: list = []

    def flake(self, message):
        if isinstance(message, _UNDEFINED):
            self.found.append(str(message))

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.found.append(f"{filename}:{lineno}: syntax error: {msg}")

    def unexpectedError(self, filename, msg):
        self.found.append(f"{filename}: {msg}")


def _files():
    for pkg in PACKAGES:
        d = os.path.join(STACK, pkg)
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                yield os.path.join(d, name)


def test_production_modules_have_no_undefined_names():
    rep = _Collect()
    for path in _files():
        with open(path, encoding="utf-8") as fh:
            pyflakes_api.check(fh.read(), path, rep)
    assert rep.found == []


def test_meta_a_removed_import_still_used_is_caught():
    rep = _Collect()
    pyflakes_api.check("import os\n\ndef f():\n    return ProcessPoolExecutor(max_workers=os.cpu_count())\n", "probe.py", rep)
    assert len(rep.found) == 1 and "ProcessPoolExecutor" in rep.found[0]
