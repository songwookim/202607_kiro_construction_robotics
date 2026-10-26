"""시험의 시뮬레이션 시계 (2026-10-09 사용자: "실제 시간이 아닌 시뮬레이션 시계로 — 시뮬레이션에선 결코 실제 이동속도로 할 필요가 없다").

sim_cell (sim_harness) 이 가상 팔·세계·가짜 센서를 만들기 **전에** vc3dz_driver.virtual_clock.VirtualClock 을 설치한다 — 팔 재생 틱, 촬영 박자,
지연·타임아웃은 시뮬 초로 정확히 흐르고 계산은 시뮬 시간을 쓰지 않는다 (같은 알고리즘·같은 mm 당·시뮬 초당 프레임 밀도·같은 판단).

끄는 법 (오늘과 같은 실시간): 환경 SCAN_TEST_CLOCK=real (scripts/run_tests.py --clock real), 또는 시험에 @pytest.mark.clock_real (실시계
카나리 — 부하에 따른 동시성: GIL·촬영 멈춤·G3 를 계속 시험한다).

E2 (시뮬 의미 불변식) 은 sim_cell 끝에서 매 시험 자동: 스트리밍 창마다 연속 ts 차 = 1000/fps, 카운터 공백 0, 멈춤 0, 수신 − 촬영 = 지연
(정확히), 연속 수신 차 = 주기, 창 안 프레임 수 = 길이 × fps ± 1; 이동 중 자세 틱 간격 = 8 ms (정확히), 이동 elapsed ≤ 궤적 끝 + 1 틱;
ClockMisuse·SimDeadlock·감시 (미수신·교착) 기록 0, 끝날 때 살아 있는 참여자 0.
"""
from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager

import numpy as np

_STATE = threading.local()
_MODE = {"real_marker": False}


def set_test_marker(real: bool) -> None:
    """conftest 의 autouse 픽스처가 시험마다 부른다 (clock_real 표지)."""
    _MODE["real_marker"] = bool(real)


def use_virtual() -> bool:
    """SCAN_TEST_CLOCK: sim (기본; clock_real 표지만 실시계) | real (모두 실시계) | sim-all (표지까지 가상 시계 — 결정성·A/B 용)."""
    mode = os.environ.get("SCAN_TEST_CLOCK", "sim").strip().lower()
    if mode == "real":
        return False
    if mode == "sim-all":
        return True
    return not _MODE["real_marker"]


def clock_label() -> str:
    return "sim" if use_virtual() else "real"


# --------------------------------------------------------------------------------------------- 수신 관찰 (E2: t_rx)
@contextmanager
def observe_client_frames():
    """시험 전용 관찰: VC3DZClient._on_result 를 감싸 (클라이언트, result_counter, ts_ms, t_rx_ns) 를 모은다 (생산 코드 무변경)."""
    from vc3dz_driver.client import VC3DZClient
    orig = VC3DZClient._on_result
    seen: list = []

    def wrapped(self, m, h, t_rx_ns, ok):
        if m.raw_payload is None:
            seen.append((id(self), int(m.result_counter), int(m.ts_ms), int(t_rx_ns)))
        return orig(self, m, h, t_rx_ns, ok)
    VC3DZClient._on_result = wrapped
    try:
        yield seen
    finally:
        VC3DZClient._on_result = orig


# --------------------------------------------------------------------------------------------- E2
def e2_check(fs, arm, seen: list, report: dict, *, fps: float, latency_ms: float, start_delay_s: float = 0.0) -> tuple:
    """→ (오류 목록, 요약 dict)."""
    errs: list = []
    lat_ns = int(round(float(latency_ms) * 1e6))
    stream_period = {(c, n): int(round(1e9 / f)) for c, n, _a, _b, f in fs.stream_log}
    rep = dict(report or {})
    if rep.get("misuse"):
        errs.append(f"ClockMisuse: {rep['misuse'][:2]}")
    if rep.get("deadlocks"):
        errs.append(f"SimDeadlock: {str(rep['deadlocks'][:1])[:1500]}")
    bad_watch = [w for w in rep.get("watchdog", []) if w[0] != "running"]
    if bad_watch:
        errs.append(f"watchdog: {bad_watch[:2]}")
    if rep.get("alive_participants"):
        errs.append(f"participants still alive at the end: {rep['alive_participants']}")
    if fs.stats.get("stalls", 0) != 0:
        errs.append(f"fake sensor stalls {fs.stats['stalls']} (sim time has no late capture)")
    # 스트림별
    caps: dict = {}
    for conn, sno, rc, ts, t_exp, t_send in fs.capture_log:
        caps.setdefault((conn, sno), []).append((rc, ts, t_exp, t_send))
    n_streams = 0
    n_frames = 0
    for conn, sno, t_start, t_stop, fps_s in fs.stream_log:
        ts_step = 1000.0 / fps_s
        period_ns = stream_period[(conn, sno)]
        rows = caps.get((conn, sno), [])
        n_streams += 1
        n_frames += len(rows)
        if not rows:
            continue
        a = np.array(rows, dtype=np.int64)
        if len(rows) > 1:
            if not np.all(np.diff(a[:, 0]) == 1):
                errs.append(f"stream {conn}/{sno}: result_counter gap")
            d_ts = np.diff(a[:, 1]).astype(float)
            if np.max(np.abs(d_ts - ts_step)) >= (1e-9 if float(ts_step).is_integer() else 1.0):
                errs.append(f"stream {conn}/{sno}: ts_ms steps {sorted(set(d_ts.tolist()))[:5]} != {ts_step}")
            if not np.all(np.diff(a[:, 2]) == period_ns):
                errs.append(f"stream {conn}/{sno}: capture spacing {sorted(set(np.diff(a[:, 2]).tolist()))[:5]} != {period_ns} ns")
        if not np.all(a[:, 3] - a[:, 2] == lat_ns):
            errs.append(f"stream {conn}/{sno}: send - capture {sorted(set((a[:, 3] - a[:, 2]).tolist()))[:5]} != latency {lat_ns} ns")
        if t_stop is not None:
            first = int(a[0, 2])
            # the processing latency is a pipeline (fake_sensor 2026-10-11): captures within the last latency before the stop are
            # still in processing and are not sent
            expect = (int(t_stop) - lat_ns - first) // period_ns + 1
            if abs(len(rows) - expect) > 1:
                errs.append(f"stream {conn}/{sno}: {len(rows)} frames in a {(t_stop - first) * 1e-9:.3f} s window (expected {expect} +- 1)")
            if first - int(t_start) != int(round(start_delay_s * 1e9)) and start_delay_s == 0.0:
                errs.append(f"stream {conn}/{sno}: first capture {first - t_start} ns after the start")
    # 수신 (클라이언트 관찰): 연결 순서 = 클라이언트 순서
    by_client: dict = {}
    for cid, rc, ts, t_rx in seen:
        by_client.setdefault(cid, []).append((rc, ts, t_rx))
    sends = {}
    for conn, sno, rc, ts, t_exp, t_send in fs.capture_log:
        sends[(conn, rc, ts)] = (t_exp, t_send, sno)
    conns = sorted({c for c, *_ in fs.capture_log})
    n_rx = 0
    for i, (cid, rows) in enumerate(by_client.items()):
        conn = conns[i] if i < len(conns) else None
        prev = None
        for rc, ts, t_rx in rows:
            n_rx += 1
            hit = sends.get((conn, rc, ts))
            if hit is None:
                errs.append(f"client {i}: frame rc={rc} ts={ts} not in the fake sensor log")
                break
            t_exp, t_send, sno = hit
            if t_rx - t_exp != lat_ns:
                errs.append(f"client {i}: t_rx - capture = {t_rx - t_exp} ns != latency {lat_ns} ns (rc {rc})")
                break
            per = stream_period.get((conn, sno))
            if prev is not None and prev[0] == sno and prev[1] == rc - 1 and t_rx - prev[2] != per:
                errs.append(f"client {i}: t_rx spacing {t_rx - prev[2]} ns != {per} ns (rc {rc})")
                break
            prev = (sno, rc, t_rx)
    if n_rx != n_frames:
        errs.append(f"received {n_rx} frames, sent {n_frames}")
    # 팔 재생 틱
    n_ticks = 0
    for k, mv in enumerate(arm.move_log):
        tk = np.asarray(mv["ticks"], dtype=np.int64)
        n_ticks += tk.size
        per = int(round(mv["period_s"] * 1e9))
        if tk.size > 1 and not np.all(np.diff(tk) == per):
            errs.append(f"move {k}: tick spacing {sorted(set(np.diff(tk).tolist()))[:5]} != {per} ns")
        # the joint-state model's settling after the command's end (sim_arm.JointStateModel: the echo's convergence, the physical
        # arm's lag) is part of the move
        if not mv["aborted"] and mv["elapsed_s"] > mv["T_end_s"] + float(mv.get("settle_s", 0.0)) + mv["period_s"] + 1e-9:
            errs.append(f"move {k}: elapsed {mv['elapsed_s']:.6f} s > trajectory end {mv['T_end_s']:.6f} s + settling "
                        f"{float(mv.get('settle_s', 0.0)):.3f} s + 1 tick")
    summary = {"streams": n_streams, "frames": n_frames, "frames_rx": n_rx, "moves": len(arm.move_log), "ticks": n_ticks,
               "sim_s": (rep.get("t_ns", 0) - 10 ** 12) * 1e-9, "wakes": rep.get("wakes"), "advances": rep.get("advances"),
               "watchdog_running": len([w for w in rep.get("watchdog", []) if w[0] == "running"])}
    return errs, summary


# --------------------------------------------------------------------------------------------- E4 A/B 기록
_NUM = re.compile(r"[-+]?\d+(\.\d+)?(e[-+]?\d+)?")


def reason_class(status: str, reason: str) -> str:
    return f"{status}: " + _NUM.sub("#", str(reason or ""))[:200]


def _arr(x):
    try:
        return np.asarray(x, dtype=float).tolist()
    except Exception:  # noqa: BLE001
        return None


def ab_summary(r) -> dict:
    """AutoTeach 결과 → A/B 비교용 요약 (실시계 대 가상 시계)."""
    passes = []
    for p in (r.passes or []):
        ext = p.get("extent") or {}
        hr = ext.get("hit_range_SE_m")
        L = (abs(float(hr[1]) - float(hr[0])) * 1e3) if hr else None
        stalls = None
        try:
            with open(os.path.join(p.get("dir") or "", "recon", "meta.json"), encoding="utf-8") as f:
                stalls = int(((json.load(f).get("meta") or {}).get("sensor_stalls") or {}).get("n_stalls", 0))
        except (OSError, ValueError, TypeError):
            pass
        passes.append({"k": p.get("k"), "kind": p.get("kind"), "direction": p.get("direction"), "n_profiles": p.get("n_profiles"),
                       "n_ok": p.get("n_ok"), "hit_len_mm": L, "spacing_mm": ext.get("spacing_mm"), "length_mm": ext.get("length_mm"),
                       "verdict": (p.get("verdict") or {}).get("status") if isinstance(p.get("verdict"), dict) else p.get("verdict"),
                       "sensor_stalls": stalls})
    ext = r.extent or {}
    ends = {k: _arr((ext.get(k) or {}).get("point")) for k in ("S", "E")}
    wp = None
    if r.weld_plan is not None:
        try:
            wp = {"length_mm": float(r.weld_plan.length_mm),
                  "stations": [{"origin": _arr(st.origin), "n": _arr(st.n), "w": _arr(st.w)} for st in r.weld_plan.stations],
                  "passes": [{"k": p.k, "kind": p.kind, "aim": _arr(p.aim_base_m), "z": _arr(np.asarray(p.R_base_tcp)[:, :, 2])}
                             for p in r.weld_plan.passes]}
        except Exception as e:  # noqa: BLE001
            wp = {"error": str(e)}
    jf = r.job_final
    return {"status": r.status, "reason": r.reason, "reason_class": reason_class(r.status, r.reason),
            "steps": [d.get("step") for d in r.decisions if d.get("step") != "timing"],
            "moves": [(m.get("name"), bool(m.get("executed"))) for m in r.moves], "n_passes": len(r.passes or []), "passes": passes,
            "length_mm": ext.get("length_mm"), "ends": ends,
            "job_final": None if jf is None else {"start": _arr(jf.start), "end": _arr(jf.end)},
            "weld_plan": wp, "elapsed_s": r.elapsed_s}


def ab_record(nodeid: str, idx: int, summary: dict) -> None:
    d = os.environ.get("SCAN_AB_DIR")
    if not d:
        return
    out = os.path.join(d, clock_label())
    os.makedirs(out, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid)[-180:]
    with open(os.path.join(out, f"{name}__{idx}.json"), "w", encoding="utf-8") as f:
        json.dump({"nodeid": nodeid, "idx": idx, "clock": clock_label(), **summary}, f, default=str)
