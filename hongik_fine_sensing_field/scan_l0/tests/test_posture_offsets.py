"""4d P8: 3G 자세 간 오프셋 자료 → 심 방향 한계 B_along (|중앙값| + k·1.4826·MAD, 가장 구체적인 열쇠에서 3 개 이상)."""
import pytest

from scan_l0 import posture_offsets as PO


def test_seed_records_and_along_bound_levels():
    recs = PO.records()
    assert len(recs) == 5 and all(r["source"].startswith("seed") for r in recs)          # 시험은 씨앗만 (POSTURE_OFFSETS=off)
    # 같은 캘리브·쌍은 2 개뿐 → 캘리브 2 개 → 쌍 (35/위 → 15/아래) 3 개: 13.13 / 10.55 / 7.13 mm
    b, info = PO.along_bound("2026-10-06", {"tilt": 35.0, "torch_above": True}, {"tilt": 15.0, "torch_above": False}, 3.0)
    assert info["key"] == "pair" and info["n"] == 3
    assert b == pytest.approx(0.01055 + 3 * 1.4826 * 0.00258, abs=2e-5)
    # 모르는 쌍·캘리브 → 전부 (5 개): 중앙값 8.98, MAD 1.85 → 17.2 mm
    b2, info2 = PO.along_bound("unknown", {"tilt": 20.0, "torch_above": True}, {"tilt": 5.0, "torch_above": True}, 3.0)
    assert info2["key"] == "all" and info2["n"] == 5 and b2 == pytest.approx(0.0172, abs=1e-4)
    # 기록된 오프셋은 모두 이 한계 안 (이어받기가 거부되지 않는다)
    assert all(abs(r["along_m"]) <= b2 for r in recs if r.get("calib_date") == "2026-10-05")


def test_append_offset_is_off_in_tests(monkeypatch, tmp_path):
    assert PO.append_offset({"run": "x", "along_m": 0.001}) is None                       # 'off': 쓰지 않는다
    p = tmp_path / "po.jsonl"
    monkeypatch.setenv("POSTURE_OFFSETS", str(p))
    assert PO.append_offset({"run": "x/2", "calib_date": "c", "along_m": -0.004}) == str(p)
    assert any(r["run"] == "x/2" for r in PO.records())
