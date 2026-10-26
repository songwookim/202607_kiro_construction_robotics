"""실행 뒤 실험 (scan_l0.experiments, 작업 파일 experiments: 블록) — 가상 팔 + 가짜 센서 (GrooveWorld 광선 추적, 실기 센서 시각 모형) +
시뮬레이션 시계, 가짜 팔 MoveIt 플랫폼 설정 (test_sim_teach1g 의 틀).

(x1) 20 mm/s, 패스 예산 3, 레이저 면 기울기 w_x 주입 (진짜 광학 틀 = 캘리브 · Rx(w_x)): 실행은 그대로 납품되고 (출력은 sections·weld_plan
     + verification + experiments 만), 판 거리 세 개 × 두 방향의 전체 패스와 정지 회전 끝 뷰 (45/90°, 끝 S/E, 두 판) 가 같은 npz 규약으로
     experiments/ 에. 실험 구간은 실행의 패스·추정에 들어가지 않는다. EXP34_KEEP=<폴더> 면 출력을 그 폴더로 복사 (오프라인 exp3/exp4
     분석 시연용).
(x2) 블록의 confirm: each — 실행은 confirm none 인데 끝 뷰의 이동은 하나하나 묻는다 (새 모션의 첫 사용); 거절 = 실험만 멈춤, 실행 결과는 그대로.
(x3) 끝 뷰 중 취소 (서버 _on_cancel 과 같은 순서) = 실험 정지, 실행 상태·출력은 그대로.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil

import numpy as np
import pytest

from scan_core.frames import rot_x
from scan_l0 import config as C
from scan_l0.autoteach import AutoTeach
from scan_l0.teach_1g import DECISION_LAYERS_NOT_USED, Teach1G
from sim_harness import (CALIB, LAB_E_TRUE, LAB_S_TRUE, RUN_KW, SENSOR_REALTIME, calib_for_sensor, rolled_normal, rough_points,
                         sim_cell, sim_settings, write_rough_job)
from vc3dz_driver.clock import get_clock

PRECISION = {"end_position_mm": 0.06, "root_position_mm": 0.05, "length_mm": 0.05}


@pytest.fixture
def calib():
    return C.load_calib(CALIB)


@pytest.fixture(autouse=True)
def no_decision_layers(monkeypatch):
    """AutoTeach 의 판정 층을 부르면 실패 (test_sim_teach1g 와 같다 — 실험 단계도 1G core 의 원시만 쓴다)."""
    called: list = []
    for nm in DECISION_LAYERS_NOT_USED:
        if not hasattr(AutoTeach, nm):
            continue

        def boom(self, *a, _nm=nm, **k):
            called.append(_nm)
            raise AssertionError(f"Teach1G called the AutoTeach decision layer {_nm}")
        monkeypatch.setattr(AutoTeach, nm, boom)
    yield called
    assert called == [], called


def calib_1g(calib):
    c = calib_for_sensor(calib, SENSOR_REALTIME)
    return dataclasses.replace(c, g1_dt_s=float(c.dt_s), dt_sigma_s=1.0e-3, he_sigma_m=2.0e-3)


def run_x(settings, calib, calib_true, tmp_path, experiments: dict, *, speed: float, run_kw=None, pre_run=None, on_event=None):
    """연구실 배치 (30 mm 안쪽, 측면 +15/−10 mm, 판 4° 롤, 판 + 뒷댐재 20 mm 아래 지지면) 에서 Teach1G 한 번 (+ experiments 블록)."""
    out = tmp_path / "out"
    with sim_cell(settings, calib, calib_true, sensor_kw=SENSOR_REALTIME, n_true=rolled_normal(LAB_S_TRUE, LAB_E_TRUE, 4.0),
                  S_true=LAB_S_TRUE, E_true=LAB_E_TRUE, table_drop_m=0.020) as cell:
        S0, E0 = rough_points(cell.world, 0.015, -0.010)
        job, doc = write_rough_job(tmp_path / "rough.yaml", S0, E0, length_tol_mm=1.0, speed_mps=speed)
        doc = dict(doc)
        doc["precision"] = dict(PRECISION)
        doc["experiments"] = experiments
        a = dict(RUN_KW)
        a.update(session=cell.session, release_sensor=True, sensor_ip="127.0.0.1", sensor_port=cell.fs.port)
        a.update(run_kw or {})
        if on_event is not None:
            a["on_event"] = on_event
        at = Teach1G(settings, job, doc, calib_1g(calib), out_dir=str(out), log=a.pop("log", lambda s: None), **a)
        if pre_run is not None:
            pre_run(at, cell)
        r = at.run()
        world = cell.world
    return at, r, out, world


def summary(out) -> dict:
    return json.loads((out / "experiments" / "summary.json").read_text(encoding="utf-8"))


def assert_run_outputs_untouched(at, r, out) -> None:
    """실행의 결과는 실험이 바꾸지 않는다: 상태 = 납품 (PASS / PASS_UNRESOLVED), out_dir = 출력 둘 + verification + experiments, 추정 파일 =
    실행의 추정, 결과의 패스 = 실행의 패스 (실험 구간 없음), 실험 구간은 experiments/legs 에만."""
    assert (r.status, r.converged) in {("PASS", "true"), ("PASS_UNRESOLVED", "false")}, (r.status, r.reason)
    assert "experiment" not in r.reason.lower(), r.reason
    assert set(os.listdir(out)) == {"sections.json", "sections.npz", "weld_plan.json", "weld_plan.npz", "kiro", "verification",
                                    "experiments", "directions"}          # + both weld directions (2026-10-14)
    est_files = sorted(os.listdir(out / "verification" / "estimates"))
    assert len(est_files) == len(at.est_history), est_files
    assert all(p_["name"].startswith("pass") for p_ in r.passes) and 1 <= len(r.passes) <= at.max_passes
    assert not any(n_.split("_", 1)[1].startswith("x") for n_ in os.listdir(out / "verification" / "legs"))
    log = json.loads((out / "verification" / "run_log.json").read_text(encoding="utf-8"))
    assert log["experiments"]["plan"] and log["status"] == r.status


# --------------------------------------------------------------------------------------------------------------------------
# (x1) 두 실험 전체 (20 mm/s, 예산 3, w_x 주입)
# --------------------------------------------------------------------------------------------------------------------------
WX_DEG = 1.3


def test_x1_both_experiments_after_a_run_at_20mm_s_budget_3(calib, tmp_path):
    st = sim_settings(max_passes=3)
    cal_true = dataclasses.replace(calib, T_flange_opt=np.asarray(calib.T_flange_opt, dtype=float) @ np.block(
        [[rot_x(math.radians(WX_DEG)), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    exp = {"standoff": {"plate_distances_m": ["pass", "pass+0.06", "pass+0.119"], "directions": [1, -1]},
           "end_views": {"angles_deg": [45, 90], "repeats": 2, "frames": 10, "retreat_m": 0.03, "ends": ["S", "E"]}}
    at, r, out, world = run_x(st, calib, cal_true, tmp_path, exp, speed=0.02)
    keep = os.environ.get("EXP34_KEEP")
    if keep:
        shutil.copytree(out, keep, dirs_exist_ok=True)
        with open(os.path.join(keep, "sim_truth.json"), "w", encoding="utf-8") as f:
            json.dump({"S_true": world.S.tolist(), "E_true": world.E.tolist(), "n_true": world.n.tolist(), "w_x_deg": WX_DEG,
                       "L_m": world.L, "spec": world.spec.to_dict()}, f, indent=1)
    assert_run_outputs_untouched(at, r, out)
    sm = summary(out)
    assert sm["stopped"] is None, sm["stopped"]
    sd = sm["standoff"]
    assert [x["status"] for x in sd] == ["done"] * 6, [(x["name"], x["status"], x.get("why")) for x in sd]
    D = sorted({round(x["plate_distance_m"], 4) for x in sd})
    assert len(D) == 3 and D[2] - D[0] == pytest.approx(0.119, abs=2e-3), D
    assert sorted(x["direction"] for x in sd) == ["E->S"] * 3 + ["S->E"] * 3
    for x in sd:                                                                  # the same npz conventions as verification/legs
        files = set(os.listdir(x["dir"]))
        assert {"raw.npz", "recon.npz", "profiles.npz", "meta.json"} <= files, files
        assert x["groove_profiles"] > 0.8 * x["n_profiles"], x
    ev = sm["end_views"]
    assert len(ev) == 2 * 2 * 2 * 2
    done = [v for v in ev if v["status"] == "done"]
    assert all(v["status"] in ("done", "skipped") and (v["status"] == "done" or v.get("why")) for v in ev), \
        [(v["name"], v["status"], v.get("why")) for v in ev]
    assert len(done) >= 12, [(v["name"], v["status"], v.get("why")) for v in ev]
    for v in done:
        z = np.load(os.path.join(v["dir"], "raw.npz"), allow_pickle=False)
        assert z["sensor_x_mm"].shape[0] == 10 and z["still_T_base_flange"].shape == (4, 4)
        meta = json.loads(str(z["view_json"]))
        assert meta["psi_deg"] in (45.0, 90.0) and abs(meta["psi_signed_deg"]) == meta["psi_deg"]
        assert v["flange_motion_mm"] < 0.05                                    # static (settled) frames
        # the line centre is on the plate top at the end: the profile shows the plate top AND the support below (in-profile step)
        T = np.asarray(v["T_base_flange"]) @ np.asarray(at.calib.T_flange_opt)
        Tt = np.asarray(v["T_base_flange"]) @ np.asarray(cal_true.T_flange_opt)
        x, zz = z["sensor_x_mm"][0].astype(float), z["sensor_z_mm"][0].astype(float)
        ok = np.isfinite(x) & np.isfinite(zz)
        P = world.point_base(x[ok], zz[ok], Tt)
        h = (P - world.S) @ world.n
        assert np.sum(np.abs(h) < 5e-4) > 30 and np.sum(h < -0.015) > 30, (v["name"], np.sum(np.abs(h) < 5e-4), np.sum(h < -0.015))
        assert T.shape == (4, 4)
    # the wrist configuration is kept (the run's wrist2 sign)
    w2 = {np.sign(v["wrist2_deg"]) for v in done if v.get("wrist2_deg") is not None}
    assert len(w2) <= 1, w2
    print(f"[x1] status {r.status} passes {len(r.passes)}; standoff {[(x['name'], round(x['plate_distance_m'] * 1e3, 1)) for x in sd]}; "
          f"end views done {len(done)}/{len(ev)}; experiments {sm['elapsed_s']:.0f} s (sim), run {sm['run']['elapsed_s_before_experiments']} s")


# --------------------------------------------------------------------------------------------------------------------------
# (x2) 끝 뷰의 첫 사용은 이동마다 확인 (블록 confirm: each) — 거절 = 실험만 정지
# --------------------------------------------------------------------------------------------------------------------------
def test_x2_end_views_ask_each_move_when_the_block_says_each_and_a_no_stops_only_the_experiments(calib, tmp_path):
    st = sim_settings(max_passes=2)
    asked: list = []
    answers = {"n": 0}

    def log(line):
        if str(line).startswith("CONFIRM "):                                     # a move that is asked (confirm_move prompt)
            asked.append(str(line).split()[1].rstrip(":"))

    def source():
        # the run's own moves are not asked (confirm none); the end views ask: yes to the first two, no to the third
        answers["n"] += 1
        return "y" if answers["n"] <= 2 else "n"
    exp = {"end_views": {"angles_deg": [90], "repeats": 2, "frames": 5, "retreat_m": 0.03, "ends": ["S"], "plates": ["left"],
                         "confirm": "each"}}
    at, r, out, _w = run_x(st, calib, calib, tmp_path, exp, speed=0.04, run_kw={"confirm_source": source, "log": log})
    assert_run_outputs_untouched(at, r, out)
    assert answers["n"] == 3 and len(asked) == 3 and all(a.startswith("x4_S_left_90_") for a in asked), (answers, asked)
    assert asked == ["x4_S_left_90_r1_retreat", "x4_S_left_90_r1_approach", "x4_S_left_90_r2_retreat"], asked
    sm = summary(out)
    assert sm["stopped"]["status"] == "ABORTED" and "declined" in sm["stopped"]["reason"], sm["stopped"]
    assert [v["status"] for v in sm["end_views"]] == ["done", "stopped"], sm["end_views"]
    assert at.confirm_mode == "none"                                              # restored after the end views
    assert any(n_.startswith("experiments stopped (ABORTED)") for n_ in r.notes)


# --------------------------------------------------------------------------------------------------------------------------
# (x3) 끝 뷰 중 취소 = 실험 정지 (같은 목표 안), 실행 결과 그대로
# --------------------------------------------------------------------------------------------------------------------------
def test_x3_cancel_during_the_end_views_stops_the_experiments_and_keeps_the_run_result(calib, tmp_path):
    st = sim_settings(max_passes=2)
    holder: dict = {}

    def cancel():
        holder["arm"].latch.fire("auto_teach action cancel requested", "action_cancel")      # 서버 _on_cancel 과 같은 순서
        holder["at"].request_cancel()

    def on_event(e, d):
        if e == "move" and str(d.get("name", "")).startswith("x4_") and str(d.get("name")).endswith("_approach") and "c" not in holder:
            holder["c"] = True
            get_clock().call_later(0.5, cancel)

    def pre(at, cell):
        holder.update(at=at, arm=cell.arm)
    exp = {"end_views": {"angles_deg": [45], "repeats": 3, "frames": 5, "retreat_m": 0.03, "ends": ["S", "E"]}}
    at, r, out, _w = run_x(st, calib, calib, tmp_path, exp, speed=0.04, pre_run=pre, on_event=on_event)
    assert holder.get("c"), "no end-view approach started"
    assert_run_outputs_untouched(at, r, out)
    sm = summary(out)
    assert sm["stopped"] and sm["stopped"]["status"] in ("ABORTED", "ESTOPPED"), sm["stopped"]
    assert sum(1 for v in sm["end_views"] if v["status"] == "done") < 3 * 2 * 2
