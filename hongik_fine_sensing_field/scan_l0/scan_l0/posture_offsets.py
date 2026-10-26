"""3G 자세 간 측정 오프셋 자료 (4d P8): 끝 패스 (아래 끝을 재는 자세) 와 주 패스 (특징을 재는 자세) 가 같은 위 끝을 다르게 재는 양.

자료 = 패키지 씨앗 (data/posture_offsets_seed.jsonl, 기록된 실기) + 실기 실행이 덧붙인 jsonl (실기만; 경로 = 환경 변수
POSTURE_OFFSETS ('off' = 쓰지 않음) 또는 ~/scan_data/posture_offsets.jsonl). 심 방향 한계 B_along = |중앙값| + k·1.4826·MAD (|along|)
— 가장 구체적인 열쇠에서 기록이 3 개 이상인 곳: (캘리브, 자세 쌍) → (캘리브) → (자세 쌍) → 전부 (씨앗이 있어 비지 않는다). 예전 고정 한계
(심 방향 25 mm, 단면 12 mm, 각 1.5°) 를 대신한다: 심 방향만 이어받은 끝을 움직이고, 단면·각은 기록·노트만 (특징은 패스마다 설계로 판정).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

SEED_PATH = Path(__file__).resolve().parent / "data" / "posture_offsets_seed.jsonl"
DATA_DEFAULT = Path.home() / "scan_data" / "posture_offsets.jsonl"
MIN_RECORDS = 3                 # 강건 척도 (중앙값·MAD) 를 내는 최소 개수 (수치)

__all__ = ["records", "along_bound", "append_offset", "pair_key"]


def _data_path() -> Optional[Path]:
    env = os.environ.get("POSTURE_OFFSETS", "").strip()
    if env.lower() in ("off", "none", "0"):
        return None
    return Path(env) if env else DATA_DEFAULT


def _read(p: Optional[Path]) -> list:
    out = []
    if p is None or not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("kind") == "offset" and math.isfinite(float(r.get("along_m", float("nan")))):
            out.append(r)
    return out


def records() -> list:
    """씨앗 + 실행 기록 (같은 run 은 씨앗 우선)."""
    seed = _read(SEED_PATH)
    seen = {r.get("run") for r in seed}
    return seed + [r for r in _read(_data_path()) if r.get("run") not in seen]


def pair_key(end_posture: Optional[dict], main_posture: Optional[dict]) -> tuple:
    def k(p):
        p = p or {}
        t = p.get("tilt")
        return (round(float(t), 1) if t is not None else None, p.get("torch_above"))
    return k(end_posture) + k(main_posture)


def along_bound(calib_date: str, end_posture: Optional[dict], main_posture: Optional[dict], k: float) -> tuple:
    """B_along [m] 와 근거 (열쇠, n, 중앙값, 척도). k = 판정 배수 (judgement.k)."""
    recs = records()
    pk = pair_key(end_posture, main_posture)
    levels = (("calib+pair", lambda r: r.get("calib_date") == calib_date and pair_key(r.get("end_posture"), r.get("main_posture")) == pk),
              ("calib", lambda r: r.get("calib_date") == calib_date),
              ("pair", lambda r: pair_key(r.get("end_posture"), r.get("main_posture")) == pk),
              ("all", lambda r: True))
    for name, f in levels:
        a = np.array([abs(float(r["along_m"])) for r in recs if f(r)], dtype=float)
        if a.size >= MIN_RECORDS or (name == "all" and a.size):
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med)))
            sig = 1.4826 * mad
            return med + float(k) * sig, {"key": name, "n": int(a.size), "median_m": med, "sigma_m": sig}
    return float("inf"), {"key": None, "n": 0}


def append_offset(rec: dict) -> Optional[str]:
    """실기 실행의 자세 오프셋 한 건을 덧붙인다 (실패해도 실행을 막지 않는다)."""
    p = _data_path()
    if p is None:
        return None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "offset", "time": _dt.datetime.now().isoformat(timespec="seconds"), **rec},
                               ensure_ascii=False, default=float) + "\n")
        return str(p)
    except Exception:  # noqa: BLE001
        return None
