"""1G core 오케스트레이터 (scan_l0.teach_1g.Teach1G) 가상 팔 시험 — G1_CORE_SPEC §10.1 (WP3).

가상 팔 (sim_arm) + 가짜 센서 (GrooveWorld 렌더, 실기 센서 시각 모형 SENSOR_REALTIME) + 시뮬레이션 시계. 계산 모형은 정식 WP1/WP2/WP4
(G1_TEST_MODELS 로 시험 대역 g1_model_stubs 를 고를 수 있다 — 같은 서명, 시험이 g1_worker.models 와 작업 프로세스 시작 코드를 monkeypatch
한다; 제품 코드에는 고르는 장치가 없다; (k) 의 끝 산포 주입은 대역에서만). 이 시험이 보는 것은 흐름·결정·
이동 구간·출력 위치 (out_dir 에는 sections.* 와 weld_plan.* 만, 나머지는 verification/) 와 AutoTeach 판정 층을 부르지 않는다는 것.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import signal
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from scan_core.frames import unit
from scan_l0 import config as C
from scan_l0 import g1_worker as GW
from scan_l0.autoteach import AutoTeach
from scan_l0.teach_1g import DECISION_LAYERS_NOT_USED, Teach1G, select_teach_class
from sim_harness import (STICK_OUT_TEST_MM, CALIB, LAB_E_TRUE, LAB_S_TRUE, RUN_KW, SENSOR_REALTIME, SIM_CONFIG, calib_for_sensor, executed,
                         rolled_normal, rough_points, sim_cell, sim_settings, write_rough_job)
from vc3dz_driver.clock import get_clock

SLOW = 0.018                       # 착점 간격 0.9 mm @ 20 fps (끝 σ = 간격/√12 — 1 mm 끝 공차를 k = 3 으로 가를 수 있는 간격)
FAST = 0.04                        # 2 mm (흐름만 보는 시나리오)
OUT_FILES = {"sections.json", "sections.npz", "weld_plan.json", "weld_plan.npz", "kiro", "directions", "verification"}


MODELS_IN_USE = {"spec": "real"}


def use_stubs(monkeypatch, spec: str = "g1_model_stubs") -> None:
    """시험 대역 (g1_model_stubs) 을 monkeypatch 로: 'g1_model_stubs' (모든 대역) 또는 'g1_model_stubs:fit_leg,estimate,…' (일부)."""
    import g1_model_stubs
    mod, _, names = spec.partition(":")
    assert mod == "g1_model_stubs", spec
    g1_model_stubs.use(monkeypatch, [n.strip() for n in names.split(",") if n.strip()] or None)
    monkeypatch.setitem(MODELS_IN_USE, "spec", spec)


@pytest.fixture(autouse=True)
def _models(monkeypatch):
    """모형: 기본은 정식 WP1/WP2/WP4. G1_TEST_MODELS=g1_model_stubs (전부) 또는 'g1_model_stubs:fit_leg,estimate,…' (일부) 면 시험 대역."""
    monkeypatch.delenv("G1_STUB_END_SCATTER_MM", raising=False)
    spec = os.environ.get("G1_TEST_MODELS", "").strip()
    if spec:
        use_stubs(monkeypatch, spec)


@pytest.fixture
def calib():
    return C.load_calib(CALIB)


@pytest.fixture
def settings():
    return sim_settings()


@pytest.fixture(autouse=True)
def no_decision_layers(monkeypatch):
    """AutoTeach 의 판정 층을 부르면 시험이 실패한다 (§8.4 Not used) — 이 파일의 모든 시험에서."""
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


def calib_1g(calib, sensor_kw=SENSOR_REALTIME):
    """가짜 센서 고정 지연을 아는 호스트 캘리브 + 1G core 의 'g1:' 블록 (D1): 1G 시간 오프셋 = 가짜 센서 지연 (연구실 파일의 3.7 ms 가
    아니라 — 그 값을 쓰면 가상 팔에서 36 ms 가 맞지 않아 진행 방향마다 끝이 ±0.65 mm 어긋난다), σ_dt 1 ms, σ_he 2 mm."""
    c = calib_for_sensor(calib, sensor_kw)
    return dataclasses.replace(c, g1_dt_s=float(c.dt_s), dt_sigma_s=1.0e-3, he_sigma_m=2.0e-3)


@contextmanager
def scene(settings, calib, tmp_path, *, lat_S=0.015, lat_E=-0.010, roll_deg=4.0, inside_S=0.030, inside_E=0.030, speed=SLOW,
          length_tol_mm=1.0, job_fields=None, S_true=LAB_S_TRUE, E_true=LAB_E_TRUE, arm_cfg=None, stick_out_mm=STICK_OUT_TEST_MM,
          **world_kw):
    kw = dict(n_true=rolled_normal(S_true, E_true, roll_deg), S_true=S_true, E_true=E_true)
    kw.update(world_kw)
    with sim_cell(settings, calib, calib, sensor_kw=SENSOR_REALTIME, arm_cfg=arm_cfg, **kw) as cell:
        S0, E0 = rough_points(cell.world, lat_S, lat_E, inside_S_m=inside_S, inside_E_m=inside_E)
        job, doc = write_rough_job(tmp_path / "rough.yaml", S0, E0, length_tol_mm=length_tol_mm, speed_mps=speed,
                                   stick_out_mm=stick_out_mm, **(job_fields or {}))
        yield cell, job, doc


def run1g(settings, cell, job, doc, calib, out, *, pre_run=None, calib_kw=None, end_tol_mm=None, platform_calib=None, **kw):
    """Teach1G 실행 (가짜 센서 고정 지연을 아는 캘리브 + 캘리브 σ). end_tol_mm: 작업 qc_targets.end_position_mm (빠른 시나리오의 끝 공차 —
    착점 간격 2 mm 에서 1 mm 공차는 k = 3 으로 가를 수 없다). pre_run(at): 실행 전에 (취소 스레드 등). platform_calib(calib) → Calib:
    서버가 Teach1G 에 주는 플랫폼 캘리브 (auto_teach_server: platform.calib_for_platform(파일 캘리브)) — None 이면 파일 캘리브 그대로."""
    a = dict(RUN_KW)
    a.update(session=cell.session, release_sensor=True, sensor_ip="127.0.0.1", sensor_port=cell.fs.port)
    a.update(kw)
    if end_tol_mm is not None:
        doc = dict(doc)
        doc["qc_targets"] = {**dict(doc.get("qc_targets") or {}), "end_position_mm": {"tol": float(end_tol_mm)}}
    cal = calib_1g(calib) if calib_kw is None else dataclasses.replace(calib_1g(calib), **calib_kw)
    if platform_calib is not None:
        cal = platform_calib(cal)
    at = Teach1G(settings, job, doc, cal, out_dir=str(out), log=a.pop("log", lambda s: None), **a)
    if pre_run is not None:
        pre_run(at)
    t0 = time.perf_counter()
    r = at.run()
    _report(cell.world, r, time.perf_counter() - t0, out)
    return at, r


def _report(world, r, wall_s: float, out) -> None:
    """G1_SIM_REPORT=<jsonl> 이면 시나리오 요약 한 줄 (패스 수·수렴·끝 σ·진짜 끝 대비 오차·구간 길이·시뮬/벽시계 시간·출력)."""
    path = os.environ.get("G1_SIM_REPORT")
    if not path:
        return
    import sys
    name = os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]
    ends = {}
    for X, true_s in (("S", 0.0), ("E", world.L * 1e3)):
        e = (r.extent or {}).get(X)
        if e:
            ends[X] = {"sigma_mm": round(1e3 * float(e["sigma_m"]), 3), "err_mm": round(taught_s_mm(world, e["point"]) - true_s, 3),
                       "two_sided": bool(e["observed"])}
    legs = {m["name"]: round(float(m["dist_mm"]), 1) for m in r.moves if m.get("executed")
            and (m["name"] == "transit_S_to_E" or (m["name"].startswith("pass") and "_" not in m["name"]))}
    rec = {"test": name, "status": r.status, "converged": r.converged, "reason": r.reason[:160],
           "passes": len([p_ for p_ in r.passes if p_.get("kind") == "pass"]),
           "corrections": [m["name"] for m in r.moves if m.get("executed") and "_corr" in m["name"]], "ends": ends,
           "length_mm": round(float((r.extent or {}).get("length_mm", float("nan"))), 3), "legs_mm": legs,
           "sim_s": round(float(r.elapsed_s), 1), "wall_s": round(wall_s, 1),
           "outputs": sorted(os.listdir(out)) if os.path.isdir(out) else [], "models": MODELS_IN_USE["spec"]}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), file=sys.stderr)


def names(r) -> list:
    return [m["name"] for m in executed(r)]


def last_msgs(r, n=8) -> list:
    return [f"[{d.get('phase')}] {d['step']}: {d['msg'][:300]}" for d in r.decisions[-n:]]


def assert_delivered(r, at) -> None:
    """D12 / D13 (r3 §8-6): PASS = 끝남 (converged 'true'), PASS_UNRESOLVED = 예산을 다 썼고 풀리지 않은 양을 밝힌다 (출력은 같이)."""
    assert (r.status, r.converged) in {("PASS", "true"), ("PASS_UNRESOLVED", "false")}, (r.status, r.converged, r.reason, last_msgs(r))
    if r.status == "PASS_UNRESOLVED":
        assert len(r.passes) == at.max_passes and r.converged_detail["unresolved"] and "unresolved" in r.reason, r.reason
    assert r.category is None


def run_log(out) -> dict:
    return json.loads((out / "verification" / "run_log.json").read_text(encoding="utf-8"))


def assert_out_dir_only_deliverables(out, *, plan: bool = True) -> None:
    """out_dir = 결과물만: sections.*, (납품하는 계획이면) weld_plan.* + KIRO 형식 kiro/ (가짜 팔 설정에 kiro_output 이 있다), verification/."""
    got = set(os.listdir(out))
    want = {"sections.json", "sections.npz", "verification"} | ({"weld_plan.json", "weld_plan.npz", "kiro", "directions"} if plan else set())
    assert got == want, sorted(got)
    if plan:                                        # both weld directions (user 2026-10-14): each set delivered or (D14) in verification/
        assert all((out / "directions" / nm).is_dir() or (out / "verification" / "directions" / nm).is_dir()
                   for nm in ("E_to_S", "S_to_E")), sorted(os.listdir(out / "directions"))
    if plan:
        n = len(json.loads((out / "weld_plan.json").read_text(encoding="utf-8"))["passes"])
        assert sorted(os.listdir(out / "kiro")) == sorted(["README.md"] + [f"pass_{k}.yaml" for k in range(1, n + 1)]
                                                          + [f"pass_{k}_info.yaml" for k in range(1, n + 1)])


def assert_kiro_outputs(out, r) -> None:
    """KIRO 형식 출력 (scan_core.kiro_plan): 패스마다 xy_endpoint_review_v1 (World, 용접 토치의 TCP; 설정의 고정 World ← link0), 결과 files 에
    있고, 쓴 자세의 토치 축이 그 패스의 WPS 진행각·작업각 (planned) 이다. 용접 토치 = 이 팔의 우리 토치 (2026-10-12): TCP = torch_tcp
    (컨택트 팁 끝) = 겨냥점 (와이어 끝) 에서 토치 축으로 20 mm 뒤, D14 는 이 토치로 했다."""
    import yaml
    from scipy.spatial.transform import Rotation
    ko = r.kiro_outputs
    assert ko["written"] is True and ko["frame_id"] == "World" and ko["world_from_base"]["base_frame"] == "link0", ko.get("reason")
    stick = STICK_OUT_TEST_MM                                               # the flow tests' stick-out (sim_harness, a test input)
    assert ko["world_from_base"]["source"].startswith("FK fit World <- link0") and ko["stick_out_mm"] == stick
    # the stick-out is measured from the gas nozzle end (2026-10-13): the nozzle end derived from this arm's URDF collision geometry
    noz = ko["nozzle_end_mm"]
    assert noz == pytest.approx(1.2, abs=1e-3) and r.weld_plan.torch.nozzle_end_source.startswith("derived from the robot's URDF")
    off = stick + noz                                                       # TCP -> wire end along z_TCP
    assert ko["runtime_base_frame"]["name"] == "link0"                      # the sim arm has no MoveIt planning frame: the plan's base
    assert all(f in r.files for f in ko["files"]) and len(ko["passes"]) == len(r.weld_plan.passes)
    assert ko["tcp_frame"] == "torch_tcp" and ko["weld_torch"]["on_this_arm"] is True
    assert np.allclose(ko["ee_from_tip"]["xyz_m"], [0.0, 0.0, off * 1e-3]) and np.allclose(ko["ee_from_tip"]["R"], np.eye(3))
    assert_d14_checked(out, r)
    from scan_core.weld_torch import roll_matrix
    roll = float(r.weld_plan.torch_roll_deg)                                # the TCP roll D14 chose (0 / 180)
    assert roll in (0.0, 180.0) and r.weld_plan.torch_roll["deg"] == roll and ko["torch_roll"]["deg"] == roll
    with np.load(out / "weld_plan.npz") as z:                               # the weld torch's TCP poses ride along the dense path
        assert str(z["tcp_frame"]) == "torch_tcp"
        for wp in r.weld_plan.passes:
            R_tip, R_tcp = z[f"p{wp.k}_R"], z[f"p{wp.k}_tcp_R"]
            assert np.allclose(R_tcp, R_tip @ roll_matrix(roll), atol=1e-12)
            assert np.allclose(z[f"p{wp.k}_tcp_pos_m"], z[f"p{wp.k}_pos_m"] - off * 1e-3 * R_tip[:, :, 2], atol=1e-12)
    wb = ko["world_from_base"]
    R_Wb, t_Wb = np.asarray(wb["R"], dtype=float), np.asarray(wb["xyz_m"], dtype=float)
    for row, wp in zip(ko["passes"], r.weld_plan.passes):
        d = yaml.safe_load((out / "kiro" / f"pass_{wp.k}.yaml").read_text(encoding="utf-8"))
        assert list(d) == ["schema", "pass", "frame_id", "requires_ik", "current_start", "current_goal"] and d["pass"] == wp.k
        assert d["schema"] == "xy_endpoint_review_v1" and d["frame_id"] == "World" and d["requires_ik"] is True
        q = [d["current_start"]["orientation_xyzw"][c] for c in "xyzw"]
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12 and row["torch_roll_deg"] == roll
        assert abs(row["written_travel_angle_deg"] - wp.travel_angle_deg) < 0.5 and abs(row["written_work_angle_deg"] - wp.work_angle_deg) < 0.5
        info = yaml.safe_load((out / "kiro" / f"pass_{wp.k}_info.yaml").read_text(encoding="utf-8"))
        ex = info["assumed_kiro_settings"]["execution_conditions"]
        assert ex["weld_tcp_speed_mm_s"] == pytest.approx(r.weld_plan.process.params(wp.role).speed_mps * 1e3)    # KIRO 설정 그대로
        assert ex["weld_weave_enabled"] is (wp.kind == "cap") and "not commands" in info["purpose"]
        # the written TCP = the aim point (wire end) 20 mm back along the written torch axis (World)
        R = Rotation.from_quat(q).as_matrix()
        tip_W = R_Wb @ np.asarray(info["derived"]["straight_line"]["tip_base_m"]["start"], dtype=float) + t_Wb
        assert np.allclose([d["current_start"]["position_m"][c] for c in "xyz"], tip_W - off * 1e-3 * R[:, 2], atol=1e-12)
        assert info["derived"]["frames"]["ee_from_tip_xyz_mm"] == pytest.approx([0.0, 0.0, off])
        o = info["derived"]["orientation"]                                    # the chosen orientation vs the WPS (information)
        assert o["wps_nominal_travel_angle_deg"] == pytest.approx(wp.travel_nominal_deg)
        assert o["deviation_from_wps_deg"] == pytest.approx(wp.orientation["deviation_deg"])
        assert o["clearance_mm"] >= o["required_clearance_mm"] > 0.0
    readme = (out / "kiro" / "README.md").read_text(encoding="utf-8")
    assert "the TCP `torch_tcp` = the torch tip end centre" in readme and f"+ the wire stick-out {stick:g} mm measured from the gas nozzle end" in readme
    assert f"by {off:.2f} mm = the TCP's recess inside the gas nozzle {noz:.2f} mm" in readme
    assert "the plan takes per pass\n  the collision-free and reachable orientation closest to them" in readme
    assert "The weld torch is on the arm that made this run" in readme and f"TCP roll about the torch axis: {roll:g} deg - roll" in readme


from torch_fixtures import tcp_text as _tcp_text                                       # noqa: E402 - the lab model's torch (turned in its clamp)
TOOL_TORCH_TCP = f"the weld torch torch_tcp (weld_torch, on this arm: tcp <- torch_tcp from the robot's TF: {_tcp_text()})"


def assert_d14_checked(out, r) -> None:
    """The weld torch is on this arm (2026-10-12: our torch, TCP = the URDF torch_tcp): D14 ran with it - every pass checked and
    reachable, each row naming the tool (the TCP read from the arm's own model), no skip note; weld_plan.json records it."""
    from scan_core.weld_torch import D14_NOT_ON_THIS_ARM
    fz = r.weld_plan.feasibility
    assert len(fz) == len(r.weld_plan.passes) and all(f["reachable"] is True for f in fz), fz
    assert all(f["tool"] == TOOL_TORCH_TCP for f in fz), fz
    assert r.weld_plan.reachability["checked"] is True and D14_NOT_ON_THIS_ARM not in r.notes
    assert any(l_ == f"  robot check tool: {TOOL_TORCH_TCP}" for l_ in r.weld_plan.summary_lines()), r.weld_plan.summary_lines()
    to = r.weld_plan.orientation                                            # the torch orientation per pass (weld_orient)
    req = r.weld_plan.clearance
    assert req["k"] == 3.0 and req["required_mm"] > 0.0 and "weld torch TCP calibration" in req["missing"]
    assert to["passes"] and all(o["chosen"] is not None and o["clearance_mm"] >= req["required_mm"] - 1e-9 for o in to["passes"]), to
    # per part role (2026-10-13): the gas nozzle + contact tip need only clear the nominal specimen (0 mm), the other parts k x sigma
    assert all(o["clearance_by_role"]["process"]["required_mm"] == 0.0 and o["clearance_by_role"]["process"]["clearance_mm"] >= -1e-9
               and o["clearance_by_role"]["other"]["required_mm"] == req["required_mm"] for o in to["passes"]), to
    assert req["by_role"]["process"]["required_mm"] == 0.0 and "process parts = the gas nozzle" in to["tool_roles"]
    assert all(set(f.get("clearance_by_role") or {}) == {"process", "other"} for f in fz), fz
    assert all(wp.orientation["chosen"]["travel_deg"] == wp.travel_angle_deg for wp in r.weld_plan.passes)
    assert all(f.get("workpiece", "").startswith("geometric clearance per part role") for f in fz), fz
    tr = r.weld_plan.torch_roll                                             # the TCP roll chosen by D14 (both rolls checked)
    assert tr["deg"] in (0.0, 180.0) and [c_["deg"] for c_ in tr["candidates"]] == [0.0, 180.0] and tr["reason"].startswith("roll ")
    assert all(f["torch_roll_deg"] == tr["deg"] for f in r.weld_plan.feasibility) and any(l_.startswith("  torch roll: roll ") for l_ in r.weld_plan.summary_lines())
    plan = json.loads((out / "weld_plan.json").read_text(encoding="utf-8"))
    assert plan["weld_torch"]["on_this_arm"] is True and plan["weld_torch"]["reachability_d14"].startswith("checked on this arm")
    assert plan["reachability"]["checked"] is True and len(plan["feasibility"]) == len(r.weld_plan.passes)


def assert_d14_skipped(out, r) -> None:
    """The platform config says the weld torch is not on this arm (weld_torch.on_this_arm false - the generic mechanism): no D14, the
    plan is delivered, and the note is in the plan summary, the run log, weld_plan.json (reachability + the weld_torch block) and kiro/
    info + README (user 2026-10-12)."""
    import yaml
    from scan_core.weld_torch import D14_NOT_ON_THIS_ARM as NOTE
    assert NOTE.startswith("weld-plan reachability (D14) not checked: the platform config says the weld torch is not mounted on this arm")
    assert r.weld_plan is not None and r.weld_plan.feasibility == [] and r.weld_plan.reachability == {"checked": False, "note": NOTE}
    assert f"  robot check: {NOTE}" in r.weld_plan.summary_lines()
    assert NOTE in r.notes and any(d_["step"] == "outputs" and d_["msg"] == NOTE for d_ in r.decisions)
    log = run_log(out)
    assert NOTE in log["notes"] and any(d_["msg"] == NOTE for d_ in log["decisions"])
    plan = json.loads((out / "weld_plan.json").read_text(encoding="utf-8"))
    assert plan["feasibility"] == [] and plan["reachability"] == {"checked": False, "note": NOTE}
    assert plan["weld_torch"]["reachability_d14"] == NOTE and plan["weld_torch"]["on_this_arm"] is False
    for wp in r.weld_plan.passes:
        info = yaml.safe_load((out / "kiro" / f"pass_{wp.k}_info.yaml").read_text(encoding="utf-8"))
        assert info["derived"]["frames"]["weld_torch"]["reachability_d14"] == NOTE
    assert NOTE in (out / "kiro" / "README.md").read_text(encoding="utf-8")


def off_arm_settings(settings) -> C.Settings:
    """The generic mechanism: a platform config that says the weld torch is not on this arm (weld_torch.on_this_arm false)."""
    p = dict(settings.params)
    p["weld_torch"] = dict(p["weld_torch"], on_this_arm=False)
    return C.Settings(p, settings.path)


def taught_s_mm(world, P) -> float:
    return float((np.asarray(P, dtype=float) - world.S) @ world.u) * 1e3


def assert_ends_near_truth(world, r, k: float = 3.0) -> None:
    """흐름 확인: 납품 끝이 진짜 끝에서 max(3 mm, k·σ_e) 안 (끝 정확도의 합격은 모형 시험·실기 자료 검증의 몫), 둘 다 양쪽에서 봤다."""
    for X, true_s in (("S", 0.0), ("E", world.L * 1e3)):
        e = r.extent[X]
        assert e["observed"] and abs(taught_s_mm(world, e["point"]) - true_s) <= max(3.0, k * 1e3 * e["sigma_m"]), (X, e)


# --------------------------------------------------------------------------------------------------------------------------
# (a) 연구실 배치: 30 mm 안쪽, 측면 +15/−10 mm, 판 4° 롤 → PASS, 거친 1 + 정밀 ≥ 2, 끝·길이, 4 패스 계획, 출력 위치
# --------------------------------------------------------------------------------------------------------------------------
def test_a_lab_layout_dwell_transit_dwell_then_alternating_passes(calib, settings, tmp_path):
    out = tmp_path / "out"
    events: list = []
    with scene(settings, calib, tmp_path) as (cell, job, doc):
        world = cell.world
        lines: list = []
        at, r = run1g(settings, cell, job, doc, calib, out, on_event=lambda e, d: events.append((e, dict(d))), log=lines.append)
        boxes_left = dict(cell.arm.boxes)
        nbr = cell.fs.params["NbrLines"]
    # 수렴 (D12) 은 모형의 몫: 끝나지 않았으면 예산을 다 썼고 풀리지 않은 양을 밝힌다 (D13 → PASS_UNRESOLVED)
    assert_delivered(r, at)
    assert r.estop_record is None and nbr == 0 and boxes_left == {}
    # 지금의 틀: 획득 (관절 이동 → 하강) → S 드웰·보정 → 트랜싯 → E 드웰·보정 → 패스 1, 2, … (그 밖의 이동 없음)
    nm = names(r)
    assert nm[:2] == ["acquire_S", "acquire_S_in"], nm
    i_tr = nm.index("transit_S_to_E")
    assert all(x.startswith("S_corr") for x in nm[2:i_tr]), nm
    i_p1 = nm.index("pass1")
    assert all(x.startswith("E_corr") or x.startswith("pass1_") for x in nm[i_tr + 1:i_p1]), nm
    assert all(x.startswith("pass") for x in nm[i_p1:]), nm
    log = run_log(out)
    kinds = [p_["kind"] for p_ in log["legs"]]                              # 모든 구간 (드웰 grab·트랜싯·패스) 은 run_log 'legs'
    i_t = kinds.index("transit")
    assert set(kinds[:i_t]) == {"dwell"} and "dwell" in kinds[i_t + 1:] and kinds[-1] == "pass", kinds
    # 결과의 패스 = 진짜 패스만 (r3 §8-2): 패스 수 = 패스 구간 수, pass_started 의 k = 1, 2, … (트랜싯은 내지 않는다)
    passes = list(r.passes)
    assert [p_["kind"] for p_ in passes] == ["pass"] * kinds.count("pass") and [p_["name"] for p_ in passes] == \
        [f"pass{i}" for i in range(1, len(passes) + 1)], [p_["name"] for p_ in passes]
    started = [(d["k"], d["name"]) for e, d in events if e == "pass_started"]
    assert started == [(i, f"pass{i}") for i in range(1, len(passes) + 1)], started
    assert len(passes) >= 2 and all(passes[i]["direction"] != passes[i + 1]["direction"] for i in range(len(passes) - 1))
    # 맞춤은 단계를 바꾸지 않는다 (r3 §8-7): 'estimate' 단계 사건 = 융합 추정 수, 맞춤 줄은 지금 단계 머리 '[dwell] fit: dwell_S1: …'
    assert sum(1 for e, d in events if e == "phase" and d.get("phase") == "estimate") == len(log["estimates"]), events
    fits = [d_ for d_ in r.decisions if d_["step"] == "fit"]
    assert len(fits) == len(log["legs"]) and {d_["phase"] for d_ in fits} <= {"dwell", "transit", "pass"}, fits
    assert any(l_.startswith("[dwell] fit: dwell_S1: ") for l_ in lines) and any(l_.startswith("[transit] fit: transit_S_to_E: ") for l_ in lines)
    # 중복 줄 없음 (r3 §8): 패스마다 계획 줄 하나, 끝 줄은 상태·사유를 되풀이하지 않는다
    assert sum(1 for l_ in lines if l_.startswith("[pass] pass 1 (")) == 1, [l_ for l_ in lines if l_.startswith("[pass] pass 1")]
    assert lines[-2] == f"{r.status}: {r.reason}" and lines[-1].startswith(f"auto-teach {r.status}: {len(passes)} pass(es), ") \
        and r.reason not in lines[-1], lines[-3:]
    # 정밀 패스의 레이저 선은 진짜 심에 직각 (E 드웰에서 융합 방향으로 yaw 를 맞췄다)
    from sim_harness import true_yaw_deg
    mv = {m["name"]: m for m in executed(r)}
    from scan_core.frames import make_transform
    for p_ in passes:
        T = make_transform(np.asarray(mv[p_["name"]]["to_R"]), np.asarray(mv[p_["name"]]["to_m"]))
        assert abs(true_yaw_deg(world, T)) < 1.0, (p_["name"], true_yaw_deg(world, T))
    # 끝·길이 (교시 S→E, 흐름 확인 — 끝 정확도의 합격은 모형 시험·실기 자료 검증의 몫): 진짜 끝 근처, 길이 380 ± 3
    assert_ends_near_truth(world, r)
    # 정밀 보정: 두 드웰 모두 마지막 판독에서 남은 보정이 없다 (E 는 yaw 까지)
    fin = {X: [d_ for d_ in r.decisions if d_["step"] == "dwell" and d_["msg"].startswith(f"{X} grab")][-1] for X in ("S", "E")}
    assert all("site done (D15" in d_["msg"] for d_ in fin.values()) and "yaw" in fin["E"]["msg"]
    corr = [x for x in nm if "_corr" in x]
    assert set(corr) <= {"S_corr1", "E_corr1"}, corr                       # 자리마다 많아야 한 번 (σ_ε 미공개: 가운데가 아니면 한 번)
    # 출력: out_dir 에는 4 파일 + KIRO 형식 kiro/ + verification/ 만; 4 패스 계획
    assert_out_dir_only_deliverables(out)
    assert_kiro_outputs(out, r)
    plan = json.loads((out / "weld_plan.json").read_text(encoding="utf-8"))
    assert [p_["kind"] for p_ in plan["passes"]] == ["fill", "fill", "fill", "cap"]
    assert len(plan["feasibility"]) == 4 and plan["reachability"]["checked"] is True        # D14: 이 팔의 우리 토치로 (2026-10-12)
    v = out / "verification"
    assert (v / "run_log.json").exists() and (v / "estimates" / "est_1.json").exists() and (v / "sensor_backup").is_dir()
    for i, p_ in enumerate(log["legs"], start=1):
        d = v / "legs" / f"{i:02d}_{p_['name']}"
        assert all((d / f).exists() for f in ("raw.npz", "recon.npz", "profiles.npz", "meta.json")), d
    assert log["pipeline"] == "g1core" and log["status"] == r.status and set(log["sites"]) == {"S", "E"} and "sites_internal" not in log
    sec = json.loads((out / "sections.json").read_text(encoding="utf-8"))
    assert sec["status"] == r.status
    assert {d["phase"] for d in r.decisions} <= {"setup", "acquire", "dwell", "correct", "transit", "estimate", "pass", "outputs"}
    # one world (cell_world): prior (rough points + design) before any motion, then one version per estimate; every MoveIt object is
    # ours ('hongik/'), the envelope reference is the world's plate plane
    worlds = [d for e, d in events if e == "world"]
    assert worlds[0]["version"] == 1 and worlds[0]["source"].startswith("prior: the two rough points")
    assert [w_["version"] for w_ in worlds] == sorted({w_["version"] for w_ in worlds}) and len(worlds) == 1 + len(log["estimates"])
    assert all(o["id"].startswith("hongik/") for w_ in worlds for o in w_["objects"]) and \
        {o["id"] for o in worlds[-1]["objects"]} == {"hongik/specimen", "hongik/floor"}
    assert log["world"][-1]["source"].startswith("measured: plate plane + k sigma_top") and log["world"][-1] == worlds[-1]
    first_move = next(i for i, d in enumerate(r.decisions) if d["step"] == "move")
    assert any(d["step"] == "world" for d in r.decisions[:first_move])                    # the world is in MoveIt before any motion
    assert ("outputs", ) == tuple(e for e, _ in events[-1:])
    assert r.job_final is not None and abs(taught_s_mm(world, r.job_final.start)) < 3.0
    # S = the first-visited end everywhere; the weld direction is the plan's own setting (default S->E: from the first-visited end)
    assert r.s_taught_point == r.visit_order["first_visited"] and sec["rough_points"]["S_taught_point"] == r.s_taught_point
    S_first = np.asarray(r.visit_order["taught_" + r.s_taught_point])
    assert np.allclose(sec["rough_points"]["S_m"], S_first) and np.allclose(log["prior"]["S0"], S_first)
    assert plan["weld_direction"]["setting"] == "S->E" and all(p_["direction"] == "S->E" for p_ in plan["passes"])
    for p_ in plan["passes"]:
        aim = np.asarray(p_["sections"]["aim_base_m"])
        assert np.linalg.norm(aim[0] - r.extent["S"]["point"]) < np.linalg.norm(aim[0] - r.extent["E"]["point"])


# --------------------------------------------------------------------------------------------------------------------------
# (b) 측면 ±60 mm 반대 부호 (ψ ≈ 20°), yaw ~15°, (c) 비대칭 들임 60/5 mm, 먼 끝을 먼저 교시
# --------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("lat_S,lat_E,inside_S,inside_E", [(0.060, -0.060, 0.030, 0.030), (0.043, -0.043, 0.030, 0.030),
                                                           (0.015, -0.010, 0.060, 0.005)], ids=["b_lat60", "yaw15", "c_inset60_5"])
def test_bc_rough_point_variants_pass(calib, settings, tmp_path, lat_S, lat_E, inside_S, inside_E):
    out = tmp_path / "out"
    with scene(settings, calib, tmp_path, lat_S=lat_S, lat_E=lat_E, inside_S=inside_S, inside_E=inside_E, speed=FAST) as (cell, job, doc):
        world = cell.world
        at, r = run1g(settings, cell, job, doc, calib, out, end_tol_mm=3.0)
    assert_delivered(r, at)                                                 # 수렴은 모형의 몫 (D12 PASS / D13 PASS_UNRESOLVED)
    assert_ends_near_truth(world, r)
    # E 드웰의 정밀 보정이 레이저 선을 융합 방향에 맞췄다: 패스의 레이저 선은 진짜 심에 직각 (거친 직선은 그 yaw 만큼 비껴 있었다)
    from sim_harness import true_yaw_deg
    from scan_core.frames import make_transform
    mv = {m["name"]: m for m in executed(r)}
    T_tr = make_transform(np.asarray(mv["transit_S_to_E"]["to_R"]), np.asarray(mv["transit_S_to_E"]["to_m"]))
    T_p1 = make_transform(np.asarray(mv["pass1"]["to_R"]), np.asarray(mv["pass1"]["to_m"]))
    yaw = math.degrees(math.atan2(abs(lat_S - lat_E), world.L - inside_S - inside_E))
    assert abs(abs(true_yaw_deg(world, T_tr)) - yaw) < 2.0 and abs(true_yaw_deg(world, T_p1)) < 1.0, (true_yaw_deg(world, T_tr), yaw)
    assert any(d["step"] == "dwell" and d["msg"].startswith("E grab 1: correct") and "yaw" in d["msg"] for d in r.decisions)
    assert_out_dir_only_deliverables(out)


def test_far_end_taught_first_S_is_the_first_visited_end_and_the_weld_direction_is_a_plan_setting(calib, settings, tmp_path):
    """교시 시작점 = 로봇에서 먼 끝 (N-S 배치, 먼 쪽 → 가까운 쪽): 방문 순서 규칙 (관절 이동 최소, 같으면 가까운 끝) 이 가까운 끝 (교시 끝점) 을
    먼저 고른다 → 그 끝이 S (사용자 2026-10-10: S = 먼저 간 끝, 교시 순서와 무관). 이름·로그·구간·run_log·sections·weld_plan 모두 그 틀 하나.
    용접 방향은 계획의 설정: 이 job 은 weld_plan.direction 'E->S' → 모든 패스가 E (먼 끝 = 교시 시작점) 에서 S 로 (경로 순서도)."""
    taught_start_far, taught_end_near = (0.80, 0.0, 0.05), (0.42, 0.0, 0.05)
    out = tmp_path / "out"
    lines: list = []
    events: list = []
    with scene(settings, calib, tmp_path, S_true=taught_start_far, E_true=taught_end_near, speed=FAST) as (cell, job, doc):
        world = cell.world                                      # world.S = 교시 시작점 (먼 끝), world.u = 먼 끝 → 가까운 끝
        doc = dict(doc, weld_plan={**dict(doc.get("weld_plan") or {}), "direction": "E->S"})
        at, r = run1g(settings, cell, job, doc, calib, out, end_tol_mm=3.0, log=lines.append,
                      on_event=lambda e, d: events.append((e, dict(d))))
    assert_delivered(r, at)
    vo = r.visit_order
    # 가까운 끝 (교시 끝점) 을 먼저 간다 → S = 교시 끝점; 순서가 정해지는 순간 (이동 전) 한 줄
    assert vo["first_visited"] == "end" and r.s_taught_point == "end"
    i_name = next(i for i, d_ in enumerate(r.decisions) if d_["msg"].startswith("S/E: S = the first-visited point = the taught end "))
    assert not any(d_["step"] == "move" for d_ in r.decisions[:i_name])
    nm = names(r)
    assert nm[:2] == ["acquire_S", "acquire_S_in"] and "transit_S_to_E" in nm and not any(x.startswith("E_corr") for x in
                                                                                         nm[:nm.index("transit_S_to_E")]), nm
    legs = [p_["name"] for p_ in run_log(out)["legs"]]
    i_tr = legs.index("transit_S_to_E")
    assert legs[0] == "dwell_S1" and all(x.startswith("dwell_S") for x in legs[:i_tr]) and legs[i_tr + 1] == "dwell_E1", legs
    assert not any("transit_E_to_S" in l_ or "taught S->E" in l_ for l_ in lines)
    grabs = [d_["msg"] for d_ in r.decisions if d_["step"] == "dwell" and " grab " in d_["msg"]]
    assert grabs[0].startswith("S grab 1: ") and any(m_.startswith("E grab 1: ") for m_ in grabs), grabs
    assert any(l_.startswith("[acquire] S: progression ") for l_ in lines)
    # 거리 기준은 S: S 자리 s ≈ 0, E 자리 s ≈ 길이 (> 0); S 는 가까운 끝
    near, far = np.asarray(taught_end_near), np.asarray(taught_start_far)
    assert set(r.sites) == {"S", "E"} and abs(r.sites["S"]["s_m"]) < 0.06 < r.sites["E"]["s_m"]
    assert [d["k"] for e, d in events if e == "pass_started"] == list(range(1, len(r.passes) + 1))
    assert vo["rule"].startswith("1G") and vo["order_rule"].startswith("1G: the visiting order with the least TCP motion")
    # 출력은 같은 틀: extent·job_final·sections·weld_plan 의 S = 가까운 끝, s 는 S 에서
    for X, P_true in (("S", near), ("E", far)):
        e = r.extent[X]
        assert e["observed"] and np.linalg.norm(np.asarray(e["point"])[:2] - P_true[:2]) <= max(3e-3, 3.0 * e["sigma_m"]), (X, e)
    assert np.linalg.norm(np.asarray(r.job_final.start) - np.asarray(r.extent["S"]["point"])) < 1e-9
    sec = json.loads((out / "sections.json").read_text(encoding="utf-8"))
    assert sec["rough_points"]["S_taught_point"] == "end" and np.allclose(sec["rough_points"]["S_m"], vo["taught_end"])
    assert np.linalg.norm(np.asarray(sec["ends"]["S"]["point_m"])[:2] - near[:2]) < 0.01 and sec["ends"]["S"]["s_m"] == 0.0
    st = [row["s_m"] for row in sec["stations"]]
    assert st == sorted(st) and np.linalg.norm(np.asarray(sec["stations"][0]["root_point_m"])[:2] - near[:2]) < 0.01
    log = run_log(out)
    assert np.allclose(log["prior"]["S0"], vo["taught_end"]) and log["s_taught_point"] == "end"
    plan = json.loads((out / "weld_plan.json").read_text(encoding="utf-8"))
    o0 = np.asarray(plan["stations"][0]["origin_base_m"])
    assert np.linalg.norm(o0[:2] - near[:2]) < 0.01                                             # 첫 단면 = S (가까운 끝)
    # 용접 방향 = 계획 설정 'E->S': 모든 패스, 경로 첫 점은 E (먼 끝) 쪽
    assert plan["weld_direction"]["setting"] == "E->S" and all(p_["direction"] == "E->S" for p_ in plan["passes"])
    for p_ in plan["passes"]:
        aim = np.asarray(p_["sections"]["aim_base_m"])
        assert np.linalg.norm(aim[0][:2] - far[:2]) < 0.02 and np.linalg.norm(aim[-1][:2] - near[:2]) < 0.02
    assert any("weld direction E->S" in l_ for l_ in lines)


# --------------------------------------------------------------------------------------------------------------------------
# (d) 끝 너머 테이블 (판 상면 20 mm 아래) + 시편 옆 가장자리 클램프 (심에서 300 mm) → 거짓 끝 없음, 정지 없음
# --------------------------------------------------------------------------------------------------------------------------
def test_d_table_beyond_the_ends_and_edge_clamps_no_false_end_no_stop(calib, settings, tmp_path):
    from groove_world import lateral_edge_clamps
    with scene(settings, calib, tmp_path, speed=FAST, inside_S=0.010, inside_E=0.010, table_drop_m=0.020,
               extras=lateral_edge_clamps(0.38, 0.300)) as (cell, job, doc):
        world = cell.world
        at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out", end_tol_mm=3.0)
    assert_delivered(r, at)
    assert r.estop_record is None
    assert_ends_near_truth(world, r)


# --------------------------------------------------------------------------------------------------------------------------
# (f) 취소: 거친 직선 중 (래치 + request_cancel, 서버와 같은 경로) / 추정 중 (작업 프로세스 SIGSTOP) → ABORTED, 1 s 안
# --------------------------------------------------------------------------------------------------------------------------
def test_f1_cancel_during_the_transit_aborts_within_a_second(calib, settings, tmp_path):
    holder: dict = {}

    def cancel():
        holder["t"] = get_clock().monotonic()
        holder["arm"].latch.fire("auto_teach action cancel requested", "action_cancel")   # 서버 _on_cancel 과 같은 순서
        holder["at"].request_cancel()

    def on_event(e, d):
        if e == "move" and d.get("name") == "transit_S_to_E":                # 트랜싯은 pass_started 를 내지 않는다 (r3 §8-2)
            get_clock().call_later(2.0, cancel)
        if e == "outputs":
            holder["t_end"] = get_clock().monotonic()

    with scene(settings, calib, tmp_path, speed=FAST) as (cell, job, doc):
        holder["arm"] = cell.arm
        _at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out", on_event=on_event, pre_run=lambda at: holder.update(at=at))
    assert r.status == "ABORTED" and "cancel" in r.reason and r.category is None, (r.status, r.reason, r.category, last_msgs(r))
    # D7 names the cause: the transit the cancel stopped is its consequence ("cancel ... (D7): move 'transit_S_to_E': ...")
    assert r.reason.startswith("cancel requested by the operator (D7): move 'transit_S_to_E'"), r.reason
    assert holder["t_end"] - holder["t"] < 1.0, (holder["t"], holder["t_end"])
    tr = [p_ for p_ in (tmp_path / "out" / "verification" / "legs").iterdir() if p_.name.endswith("_transit_S_to_E")]
    assert len(tr) == 1 and (tr[0] / "raw.npz").exists()                                             # 받은 프레임까지
    assert os.listdir(tmp_path / "out") == ["verification"]


def test_f2_cancel_while_the_worker_is_stopped_aborts_within_a_second(calib, settings, tmp_path, monkeypatch):
    holder: dict = {}
    orig = GW.G1Worker._send

    def send(self, job, payload):
        orig(self, job, payload)
        if job == "estimate" and "pid" not in holder:
            holder["pid"] = self.pid
            os.kill(self.pid, signal.SIGSTOP)                               # 계산 중 멈춘 작업 프로세스

            def later():
                time.sleep(0.2)
                holder["t_cancel"] = time.monotonic()
                holder["at"].request_cancel()
            threading.Thread(target=later, daemon=True).start()
    monkeypatch.setattr(GW.G1Worker, "_send", send)
    with scene(settings, calib, tmp_path, speed=FAST) as (cell, job, doc):
        _at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out", pre_run=lambda at: holder.update(at=at))
        t_end = time.monotonic()
    assert r.status == "ABORTED" and "cancel" in r.reason and "'estimate'" in r.reason, (r.status, r.reason, last_msgs(r))
    assert t_end - holder["t_cancel"] < 1.0
    with pytest.raises(ProcessLookupError):
        os.kill(holder["pid"], 0)                                           # 멈춘 작업 프로세스는 남지 않는다 (거두었다)
    assert r.estop_record is None                                           # 팔은 서 있었다: 래치 없음


# --------------------------------------------------------------------------------------------------------------------------
# (g) 작업 프로세스가 추정 중 죽음: 한 번 → 다시 띄워 같은 일감 (D8), 실행은 그대로 PASS; 두 번 → ERROR
# --------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kills", [1, 2])
def test_g_worker_death_mid_estimate(calib, settings, tmp_path, monkeypatch, kills):
    n = {"killed": 0}
    orig = GW.G1Worker._send

    def send(self, job, payload):
        orig(self, job, payload)
        if job == "estimate" and n["killed"] < kills:
            n["killed"] += 1
            os.kill(self.pid, signal.SIGKILL)
    monkeypatch.setattr(GW.G1Worker, "_send", send)
    with scene(settings, calib, tmp_path, speed=FAST) as (cell, job, doc):
        world = cell.world
        at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out", end_tol_mm=3.0)
    log = json.loads((tmp_path / "out" / "verification" / "run_log.json").read_text(encoding="utf-8"))
    if kills == 1:
        assert_delivered(r, at)
        assert log["worker"]["respawns"] == 1 and any("(D8)" in d["msg"] for d in r.decisions)
        assert_ends_near_truth(world, r)
    else:
        assert r.status == "ERROR" and "died twice" in r.reason and "(D8)" in r.reason and r.category == "internal", (r.status, r.reason)
        assert [m["name"] for m in executed(r)][-1] == "transit_S_to_E" and os.listdir(tmp_path / "out") == ["verification"]


# --------------------------------------------------------------------------------------------------------------------------
# (h) 라우팅: 이 패키지는 1G core 만 (다른 자세·pipeline 은 거부); Teach1G 는 3G 를 받지 않는다
# --------------------------------------------------------------------------------------------------------------------------
def test_h_routing_and_3g_refused_by_the_core(calib, settings, tmp_path):
    assert select_teach_class("1G", "core") is Teach1G and select_teach_class(" 1G ", "core") is Teach1G
    for pos, pl in (("3G", "core"), ("1G", "legacy"), ("1G", "new")):
        with pytest.raises(ValueError):
            select_teach_class(pos, pl)
    with scene(settings, calib, tmp_path, lat_S=0.0, lat_E=0.0) as (cell, job, doc):
        _at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out", position="3G")
        n = getattr(cell.arm.backend, "n_moves", 0)
    assert r.status == "REFUSED" and "1G jobs only" in r.reason and n == 0 and r.category == "input", (r.status, r.reason)


# --------------------------------------------------------------------------------------------------------------------------
# (i) dry run: 게이트 + 획득 계획만 (이동·센서·작업 프로세스 없음)
# --------------------------------------------------------------------------------------------------------------------------
def test_i_dry_run_plans_only(calib, settings, tmp_path, monkeypatch):
    started: list = []
    monkeypatch.setattr(GW.G1Worker, "start", lambda self: started.append(1))
    from scan_core.frames import make_transform
    foreign = {"id": "kiro_cell/table", "size": [0.2, 0.2, 0.2], "T": make_transform(np.eye(3), [-1.5, 1.5, 0.0])}   # 남의 물체
    with scene(settings, calib, tmp_path) as (cell, job, doc):
        cell.arm.set_scene_boxes([foreign])
        seen: list = []
        orig = cell.arm.set_scene_boxes
        cell.arm.set_scene_boxes = lambda boxes: (seen.extend(b_["id"] for b_ in boxes), orig(boxes))[1]
        _at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out", dry_run=True)
        conn, boxes, n = cell.fs.stats["connections"], dict(cell.arm.boxes), cell.arm.backend.n_moves
    assert r.status == "DRY_RUN", (r.status, r.reason, last_msgs(r))
    assert r.moves == [] and n == 0 and conn == 0 and list(boxes) == ["kiro_cell/table"] and started == []   # 우리 것만 넣고 지웠다
    assert seen and set(seen) == {"hongik/specimen", "hongik/floor", "hongik/under"}      # under: no motion below the specimen (2026-10-14)
    msgs = [d["msg"] for d in r.decisions]
    assert any(m.startswith("1G core: taught rough points") and "dwell / transit range" in m for m in msgs)
    assert any(m.startswith("S: acquisition") for m in msgs) and any(m.startswith("S: progression") for m in msgs)
    assert os.listdir(tmp_path / "out") == ["verification"]


# --------------------------------------------------------------------------------------------------------------------------
# (j) 기운 1G 판 (8°, 심 방향) + 착점이 루트 깊이 → S1 포락선 기준 = 잰 판 평면, 실행된 이동의 툴 점은 진짜 판에서 정책 여유 C 밖
# --------------------------------------------------------------------------------------------------------------------------
def test_j_tilted_plate_rough_points_at_the_root_s1_uses_the_measured_plane(calib, settings, tmp_path):
    from scan_core.path_gen import body_min_plate_dist, tool_keypoints
    from scan_core.frames import make_transform
    th = math.radians(8.0)
    S = np.array(LAB_S_TRUE)
    u = np.array([0.0, math.cos(th), math.sin(th)])
    n = np.array([0.0, -math.sin(th), math.cos(th)])
    E = S + u * 0.38
    with sim_cell(settings, calib, calib, sensor_kw=SENSOR_REALTIME, S_true=tuple(S), E_true=tuple(E), n_true=tuple(n)) as cell:
        world = cell.world
        S0, E0 = rough_points(world, 0.015, -0.010, dz_m=-0.010)            # 루트 깊이에 교시
        job, doc = write_rough_job(tmp_path / "rough.yaml", S0, E0, length_tol_mm=1.0, speed_mps=FAST)
        at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out", end_tol_mm=3.0)
    assert_delivered(r, at)
    log = json.loads((tmp_path / "out" / "verification" / "run_log.json").read_text(encoding="utf-8"))
    ref = log["world"][-1]
    assert ref["source"].startswith("measured: plate plane") and angle(ref["envelope"]["plate_normal"], world.n) < 0.5
    assert abs(float((np.asarray(ref["envelope"]["plate_point_m"]) - world.S) @ world.n)) < 2e-3      # 판 상면 (+ k σ)
    prior = [d for d in r.decisions if d["step"] == "world"]
    assert prior[0]["msg"].startswith("world v1 (prior: the two rough points") and "measured: plate plane" in prior[1]["msg"]
    # 사전 세계의 판 평면 = 두 착점의 선 (8° 기울기) + t_hit — 수평 평면이 아니다 (아래 끝에서 판 위 53 mm 를 막지 않는다)
    assert angle(log["world"][0]["envelope"]["plate_normal"], [0.0, -math.sin(th), math.cos(th)]) < 1.0
    C_m = float(at.policy.clearance_m)
    for m in executed(r):
        if m["phase"] == "joint":
            continue
        T = make_transform(np.asarray(m["to_R"]), np.asarray(m["to_m"]))
        kp = tool_keypoints(T, at.calib, at.job0)
        d = min(float(world.plate_distance(p[None, :])[0]) for nm, p in kp.items() if nm != "opt")
        db = body_min_plate_dist(T[None], at.calib, world.S, world.n)
        if db is not None:
            d = min(d, float(db[0]))
        assert d >= C_m, (m["name"], d)


def angle(a, b) -> float:
    a, b = unit(np.asarray(a, dtype=float)), unit(np.asarray(b, dtype=float))
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


# --------------------------------------------------------------------------------------------------------------------------
# (k) 예산 2 + 구간별 끝 산포 0.6 mm → PASS converged=false, 풀리지 않은 양 = 끝
# --------------------------------------------------------------------------------------------------------------------------
def test_k_budget_used_with_end_scatter_reports_unresolved_ends(calib, settings, tmp_path, monkeypatch):
    # 끝 산포 주입은 대역 모형에만 있다 → 이 시나리오는 대역 (g1_model_stubs) 으로 돈다 (정식 모형이 예산 2 에서 끝을 푸는지는 모형의
    # σ 에 달린 일이라 — WP2 2차부터 이 배치에서 2 패스로 풀린다 — 그것에 기대지 않는다). 대역의 자연 방향 차 (가상 팔, 반대 방향 끝
    # 읽기 ~0.7 mm) 보다 크게 넣는다 — §10.1 (k) 의 0.6 mm 는 정식 모형 (REML τ_e) 의 시험
    use_stubs(monkeypatch)
    monkeypatch.setenv("G1_STUB_END_SCATTER_MM", "1.5")
    out = tmp_path / "out"
    with scene(settings, calib, tmp_path) as (cell, job, doc):
        _at, r = run1g(settings, cell, job, doc, calib, out, max_passes=2)
    # D13 → 별도 상태 PASS_UNRESOLVED (r3 §8-6): 출력은 그대로, 미해결 목록과 함께
    assert r.status == "PASS_UNRESOLVED" and r.converged == "false" and r.category is None, (r.status, r.reason, last_msgs(r))
    assert len(r.passes) == 2 and {"end_S", "end_E"} & set(r.converged_detail["unresolved"]), r.converged_detail
    assert r.reason.startswith("pass budget 2 used; unresolved: [") and "max sigma/precision" in r.reason, r.reason
    assert_out_dir_only_deliverables(out)                                   # 출력은 그대로 (풀리지 않은 양 표시)
    sec = json.loads((out / "sections.json").read_text(encoding="utf-8"))
    assert sec["status"] == "PASS_UNRESOLVED" and run_log(out)["status"] == "PASS_UNRESOLVED"
    assert r.weld_plan is not None and all(f["reachable"] is True for f in r.weld_plan.feasibility)
    from scan_core.g1_types import LegObs                                   # 작업 프로세스가 대역을 썼다 (monkeypatch 한 시작 코드)
    p1 = [d for d in (out / "verification" / "legs").iterdir() if d.name.endswith("_pass1")][0]
    assert LegObs.load(p1 / "profiles.npz").meta.get("model") == "g1_model_stubs"


# --------------------------------------------------------------------------------------------------------------------------
# (m) §4.9 model fit + hypotheses: scenarios that ended in a silent wrong PASS with the k sigma <= tol test alone. Each must end
#     MODEL_MISFIT (DATA_FAIL, sensing) or PASS_UNRESOLVED naming the segment / hypothesis — never PASS
# --------------------------------------------------------------------------------------------------------------------------
def _last_checks(out) -> dict:
    est = sorted((out / "verification" / "estimates").glob("est_*.json"), key=lambda p_: int(p_.stem.split("_")[1]))[-1]
    return json.loads(est.read_text(encoding="utf-8"))


def test_m1_ghost_band_under_the_lower_bevel_is_never_a_wrong_pass(calib, settings, tmp_path):
    """Lab at06/at07-like ghost on the shiny last 25 mm before the E end: the groove-bottom returns come back 2.5 mm too deep (an
    interreflection band under the lower bevel); a free template fit reads the root 2.5 mm low there → those profiles are not this seam
    at the fused line and the pass-1 groove change point stops 25 mm short of the end (before: the Huber weight kept 1/25 of that
    reading and the run PASSed with E 3.9 mm short at sigma 0.76 mm). Now the end region's two readings (groove change point vs both
    plates present) decisively disagree after pass 1, the reading's segment indicator takes it out (gamma -> 1), the next passes
    (fitted with the estimate's seam prediction) read the true end: the run ends with the true ends or MODEL_MISFIT — never a wrong
    PASS."""
    from groove_world import GhostBand
    out = tmp_path / "out"
    with scene(settings, calib, tmp_path, speed=FAST, ghost=GhostBand(0.355, 0.381, depth_m=0.0025)) as (cell, job, doc):
        world = cell.world
        at, r = run1g(settings, cell, job, doc, calib, out, end_tol_mm=3.0)
    if r.status == "DATA_FAIL":
        assert r.category == "sensing" and r.reason.startswith("MODEL_MISFIT after the pass budget") and "end region" in r.reason, \
            (r.status, r.reason)
    else:
        assert_delivered(r, at)
        assert_ends_near_truth(world, r)                                             # not a wrong PASS
    log = run_log(out)
    seen = [b for e in log["estimates"] for b in ((e.get("model_checks") or {}).get("summary") or {}).get("misfit_blocking") or []]
    assert any(b.startswith("end region") and "both plates present" in b for b in seen), seen
    # the ghost's groove change point (a whole-profile reading — the delivered ends are the plate-end model's, which reads the plate
    # tops the ghost band does not touch) is down-weighted (not dropped)
    ge = _last_checks(out)["plate"]["groove_ends"]
    rd = [r_["fit"] for X in ("S", "E") for r_ in ge[X]["readings"] if (r_.get("fit") or {}).get("verdict") == "misfit"]
    assert all(x["gamma_out"] > 0.99 for x in rd), rd


def test_m2_wrong_bevel_side_prior_is_unresolved_never_pass(calib, settings, tmp_path, monkeypatch):
    """A bevel side given to the estimator (SeamPrior.bevel_sign_w — the job's side) that the data contradict: today the prior overrides
    and the passes are fitted with the wrong mirror template. The side hypothesis (pooled ln BF of the mirrored templates, signed by
    the decided side) has 2 ln odds << k^2 → every done-test quantity is unresolved and the scan names bevel_side — never PASS.
    The run ends PASS_UNRESOLVED, or DATA_FAIL when no plan can be delivered: the wrong-template passes are decisively not this seam
    (2026-10-12, with the run's real timing prior — before, the virtual arm's dwell stamps made σ_δ 160 ms and blurred that), the
    ends rest on at most one motion direction, are not established (σ ∞) and the S→E plan contract refuses the plan, or the plan
    from the wrong-side sections is not reachable (D14, 'waypoint'; the scan's reason with the unresolved bevel_side is kept)."""
    holder = {}
    orig = Teach1G._make_prior

    def wrong_side(self):
        p = orig(self)
        u0 = p.frame0()["u0"]
        # world bevel 'left' = the bevel plate on -(n x u) of the world (groove_world); w0 = n0 x u0
        true_w = -1 if float(u0 @ holder["world"].u) > 0 else 1
        return dataclasses.replace(p, bevel_sign_w=-true_w)
    monkeypatch.setattr(Teach1G, "_make_prior", wrong_side)
    out = tmp_path / "out"
    with scene(settings, calib, tmp_path, speed=FAST) as (cell, job, doc):
        holder["world"] = cell.world
        at, r = run1g(settings, cell, job, doc, calib, out, end_tol_mm=3.0)
    assert r.status != "PASS", (r.status, r.reason)
    assert "unresolved hypotheses: bevel_side (2 ln odds -" in r.reason, (r.status, r.reason, last_msgs(r))
    if r.status == "DATA_FAIL":                       # D14: the plan is refused (S->E contract) or not verified - never delivered
        assert r.category == "waypoint" and r.reason.startswith("weld plan ") and "(D14" in r.reason, (r.status, r.category,
                                                                                                         r.reason)
        assert (out / "verification" / "weld_plan.json").exists() and not (out / "weld_plan.json").exists()
    else:
        assert r.status == "PASS_UNRESOLVED", (r.status, r.reason, last_msgs(r))
    hyp = run_log(out)["estimates"][-1]["model_checks"]["hypotheses"]["bevel_side"]
    assert hyp["from"] == "prior (job)" and not hyp["resolved"] and hyp["two_ln_odds"] < -at.k ** 2, hyp
    sec = json.loads((out / "sections.json").read_text(encoding="utf-8"))
    assert sec["status"] == "PASS_UNRESOLVED" and not sec["done"]["resolved"]


def test_m3_groove_like_fixture_beyond_an_end_at_plate_height_is_a_model_misfit(calib, settings, tmp_path):
    """A jig at plate height butted flush against the plate ends, its slot on the seam line (run-off-tab-like): a pass that runs onto
    it sees a groove at the fused line, so the groove change point moves to the jig's far end (before: PASS with that end, ~41 mm off).
    The plates' presence ends at the true end (the seam ends where both plates are present) — the end region's two readings decisively
    disagree → MODEL_MISFIT naming the end region (never PASS)."""
    from groove_world import slot_fixture
    out = tmp_path / "out"
    with scene(settings, calib, tmp_path, speed=FAST, inside_S=0.060, inside_E=0.005,
               extras=slot_fixture(0.38, gap_m=0.0, length_m=0.040)) as (cell, job, doc):
        world = cell.world
        at, r = run1g(settings, cell, job, doc, calib, out, end_tol_mm=3.0)
    assert r.status != "PASS", (r.status, r.reason, r.extent)
    assert r.status == "DATA_FAIL" and r.category == "sensing" and "MODEL_MISFIT" in r.reason and "end region" in r.reason, \
        (r.status, r.reason, last_msgs(r))
    reg = [v_.get("region") for v_ in _last_checks(out)["plate"]["model_checks"]["ends"].values()]
    assert any(g_ and g_["verdict"] == "misfit" and abs(g_["groove_minus_plates_mm"]) > 10.0 for g_ in reg), reg


def test_m4_ends_never_identified_deliver_no_weld_paths(calib, settings, tmp_path):
    """Identification decides (user 2026-10-11): the joint is 500 mm long (design 380), the rough points 30 mm inside its ends — pass 1
    spans the corrected points only (no remaining design length) and the budget is that one pass, so no pass crosses either end
    (one-sided bounds only). The run delivers NO weld paths: DATA_FAIL / 'sensing' (KIRO SENSING_FAILED) naming the unidentified ends;
    sections of what was measured are written, the weld plan is in verification/ only."""
    out = tmp_path / "out"
    with scene(settings, calib, tmp_path, speed=FAST, S_true=(0.60, -0.25, 0.05), E_true=(0.60, 0.25, 0.05)) as (cell, job, doc):
        at, r = run1g(settings, cell, job, doc, calib, out, end_tol_mm=3.0, max_passes=1)
    assert r.status == "DATA_FAIL" and r.category == "sensing", (r.status, r.reason, last_msgs(r))
    assert "end(s) S, E not identified by the measurements" in r.reason and "no weld paths" in r.reason, r.reason
    assert r.weld_plan is None and set(os.listdir(out)) == {"sections.json", "sections.npz", "verification"}, sorted(os.listdir(out))
    assert (out / "verification" / "weld_plan.json").exists()
    sec = json.loads((out / "sections.json").read_text(encoding="utf-8"))
    assert sec["status"] == "DATA_FAIL" and not sec["ends"]["S"]["two_sided"] and not sec["ends"]["E"]["two_sided"]
    from hongik_scan_ros.kiro_io import ERROR_SENSING_FAILED, run_outcome
    oc = run_outcome(r.status, r.reason, category=r.category)
    assert (oc.action, oc.error_code, oc.deliver_paths) == ("abort", ERROR_SENSING_FAILED, False)


# --------------------------------------------------------------------------------------------------------------------------
# 거부: 이동 전 (D1), 그루브 없음 (D11), 포락선 (D3)
# --------------------------------------------------------------------------------------------------------------------------
def test_refusals_before_motion(calib, settings, tmp_path):
    with scene(settings, calib, tmp_path, lat_S=0.0, lat_E=0.0) as (cell, job, doc):
        _a, r1 = run1g(settings, cell, job, doc, calib, tmp_path / "o1", calib_kw={"dt_sigma_s": None, "he_sigma_m": None})
        d2 = {k_: v_ for k_, v_ in doc.items() if k_ != "specimen"}
        _a, r2 = run1g(settings, cell, job, d2, calib, tmp_path / "o2")
        d3 = dict(doc, qc_targets={k_: v_ for k_, v_ in doc["qc_targets"].items() if k_ != "root_position_mm"})
        _a, r3 = run1g(settings, cell, job, d3, calib, tmp_path / "o3")
        j4 = dataclasses.replace(job, end=job.start.copy() + np.array([0.0, 0.0, 0.05]))
        _a, r4 = run1g(settings, cell, j4, doc, calib, tmp_path / "o4")
        g = int(settings.ghost_burst_frames)
        _a, r5 = run1g(settings, cell, job, doc, calib, tmp_path / "o5", dwell_frames=2 * g)          # 2g < 2g + 1
        bad = C.Settings({**settings.params, "kiro_output": {**settings.params["kiro_output"], "orientation_policy": "reference"}},
                         settings.path)                                                              # a removed key (2026-10-12)
        _a, r6 = run1g(bad, cell, job, doc, calib, tmp_path / "o6")
        n = getattr(cell.arm.backend, "n_moves", 0)
    # the time-offset σ is not a calibration input any more (the run's timing sets it); the hand-eye class σ still is
    assert r1.status == "REFUSED" and "he_sigma_m" in r1.reason and "dt_sigma_s" not in r1.reason, r1.reason
    assert r2.status == "REFUSED" and "specimen" in r2.reason, r2.reason
    assert r3.status == "REFUSED" and "root_position_mm" in r3.reason, r3.reason
    assert r4.status == "REFUSED" and "the two rough points coincide" in r4.reason, r4.reason
    assert r5.status == "REFUSED" and "ghost_burst_frames" in r5.reason, r5.reason
    assert r6.status == "REFUSED" and "kiro_output.orientation_policy: removed" in r6.reason, r6.reason
    assert n == 0 and all(r_.n_move_l == 0 for r_ in (r1, r2, r3, r4, r5, r6))
    # 실패 분류 (r3 §4.2 — KIRO error_code 이름): 캘리브·설정 = config, 작업 입력 = input
    assert [r_.category for r_ in (r1, r2, r3, r4, r5, r6)] == ["config", "input", "input", "input", "config", "config"]


def test_arm_inside_the_world_is_refused_before_motion_naming_the_link(calib, settings, tmp_path):
    """시편이 접힌 자세의 툴 바로 아래 (툴이 시편 물체 = 시편 + C 안): 세계를 줄이지 않고 (사다리 없음) 이동 전에 REFUSED — 어느 링크가 어느
    물체 안인지 이름으로; 보낸 것 없음, 장면에 남은 것 없음, 래치 없음."""
    S, E = (0.15, -0.05, 0.20), (0.55, -0.05, 0.20)
    with scene(settings, calib, tmp_path, lat_S=0.0, lat_E=0.0, S_true=S, E_true=E, speed=FAST) as (cell, job, doc):
        _at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out")
        n, boxes, latched, conn = cell.arm.backend.n_moves, dict(cell.arm.boxes), cell.arm.latch.is_set, cell.fs.stats["connections"]
    assert r.status == "REFUSED" and r.category == "motion", (r.status, r.reason, last_msgs(r))
    assert r.reason.startswith("the arm at its current pose is inside the cell world v1: ") and r.reason.endswith("(nothing sent)")
    assert "eoat_base_link is inside hongik/specimen (the specimen (" in r.reason and "grown by C 50 mm" in r.reason, r.reason
    assert n == 0 and r.moves == [] and boxes == {} and not latched and conn == 0
    worlds = [d for d in r.decisions if d["step"] == "world"]
    assert len(worlds) == 1 and worlds[0]["world"]["version"] == 1                     # one world, not shrunk / retried


def test_sensor_connection_failure_before_any_motion_is_refused_not_error(calib, settings, tmp_path):
    """r3 §8-5: 첫 이동 전에 센서에 접속하지 못하면 (닫힌 포트) REFUSED, 분류 sensor (예전: 예외 그대로 ERROR 'ConnectionRefusedError …') —
    보낸 이동 없음, 래치 없음, 장면 상자 남지 않음."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as so:              # 비어 있는 루프백 포트 (닫은 뒤 아무도 듣지 않는다)
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    lines: list = []
    with scene(settings, calib, tmp_path, speed=FAST) as (cell, job, doc):
        _at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out", sensor_port=port, log=lines.append)
        n, boxes, latched = cell.arm.backend.n_moves, dict(cell.arm.boxes), cell.arm.latch.is_set
    assert r.status == "REFUSED" and r.category == "sensor", (r.status, r.category, r.reason)
    assert r.reason.startswith("sensor connection/initialisation failed before any motion (nothing sent): ") and f":{port})" in r.reason
    assert n == 0 and r.n_move_l == 0 and r.moves == [] and boxes == {} and not latched and r.estop_record is None
    assert not any(l_.startswith("ERROR") for l_ in lines), lines[-4:]


def test_abort_category_names_only():
    """_Abort(category=…) 는 이름만 (ABORT_CATEGORIES 밖은 ValueError); 상태 STATUS_PASS_UNRESOLVED 는 STATUSES 에 있다."""
    from scan_l0.autoteach import ABORT_CATEGORIES, STATUS_PASS_UNRESOLVED, STATUSES, AutoTeachResult, _Abort
    assert ABORT_CATEGORIES == ("input", "config", "motion", "sensor", "sensing", "waypoint", "internal")
    assert STATUS_PASS_UNRESOLVED == "PASS_UNRESOLVED" and STATUS_PASS_UNRESOLVED in STATUSES
    e = _Abort("REFUSED", "x", category="sensor")
    assert e.category == "sensor" and not e.safety and _Abort("ABORTED", "y").category is None
    with pytest.raises(ValueError):
        _Abort("REFUSED", "x", category="robot")
    assert AutoTeachResult().to_dict()["category"] is None


def test_dwell_median_profile_column_needs_a_majority_of_frames():
    """열별 중앙값 (spec §3.6a): 48/50 프레임에서 빠진 열의 2 프레임 유령점은 열을 살리지 못한다 (과반 아님); 과반이 돌아온 열은
    중앙값이 유령점 값이 아니다. 유령점 버스트 g = 2, grab 2g + 1 = 5 프레임에서도 같다."""
    from scan_core.groove_model import _median_profile
    for F in (50, 5):
        x = np.tile(np.linspace(-50.0, 50.0, 8), (F, 1))
        z = np.full((F, 8), 300.0)
        v = np.ones((F, 8), dtype=bool)
        v[2:, 3] = False                                        # 열 3: 처음 2 프레임 (유령점 버스트) 만 돌아왔다
        z[:2, 3] = 250.0
        z[:2, 5] = 250.0                                        # 열 5: 모든 프레임이 돌아왔고 그중 2 프레임이 유령점
        v[: (F - 1) // 2, 6] = False                            # 열 6: 과반 (F//2 + 1) 만 돌아왔다
        _xm, zm, vm, nf, _ts = _median_profile(x, z, v)
        assert nf == F and not vm[3] and vm[5] and zm[5] == 300.0 and vm[6], (F, vm)


def test_d11_groove_not_found_along_the_rough_line_is_data_fail(calib, settings, tmp_path):
    """착점이 시편 위지만 그루브에서 200 mm 옆 (거친 시야 여유 ≈ 88 mm 밖): 첫 드웰에 그루브 증거가 없다 → DATA_FAIL (탐색 없음)."""
    with scene(settings, calib, tmp_path, lat_S=0.200, lat_E=0.200, speed=FAST) as (cell, job, doc):
        _at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out")
    assert r.status == "DATA_FAIL" and "groove not in view at the" in r.reason and "D11: first grab" in r.reason, (r.status, r.reason)
    assert [m["name"] for m in executed(r)] == ["acquire_S", "acquire_S_in"] and r.estop_record is None and r.category == "sensing"


def test_d3_fine_pass_refused_by_the_envelope_before_motion(calib, settings, tmp_path):
    """clamp 0.18 m: 거친 거리의 이동 (드웰·보정·트랜싯) 은 통과, 패스 거리로 내려가는 구간은 포락선이 이동 전에 거부 → REFUSED (래치 없음)."""
    with scene(settings, calib, tmp_path, speed=FAST, job_fields={"clamp_height_m": 0.18}) as (cell, job, doc):
        _at, r = run1g(settings, cell, job, doc, calib, tmp_path / "out")
        latched = cell.arm.latch.is_set
    assert r.status == "REFUSED" and "(D3)" in r.reason and "clamp_clearance" in r.reason and r.category == "motion", (r.status, r.reason)
    refused = [m for m in r.moves if m.get("refused")]
    assert len(refused) == 1 and refused[0]["name"].startswith("pass1") and not latched and r.estop_record is None


# --------------------------------------------------------------------------------------------------------------------------
# D14: 도달성을 확인하지 못한 계획 (답 없음 / 확인 기능 없는 플랫폼) 은 납품하지 않는다 → verification/, DATA_FAIL
# --------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["no_answer", "no_check"])
def test_d14_unverified_weld_plan_is_retracted(calib, settings, tmp_path, monkeypatch, how):
    """D14 runs where the weld torch is on the arm (the lab / fake config: our torch, weld_torch.on_this_arm true)."""
    from sim_arm import SimArmPlatform
    assert settings.params["weld_torch"]["on_this_arm"] is True
    if how == "no_answer":
        orig = SimArmPlatform.check_weld_plan

        def check(self, plan, **kw):
            out = orig(self, plan, **kw)
            out[1].update(reachable=None, reason="no answer (test)")
            return out
        monkeypatch.setattr(SimArmPlatform, "check_weld_plan", check)
    else:
        monkeypatch.delattr(SimArmPlatform, "check_weld_plan")
    out = tmp_path / "out"
    with scene(settings, calib, tmp_path, speed=FAST) as (cell, job, doc):
        _at, r = run1g(settings, cell, job, doc, calib, out, end_tol_mm=3.0)
    assert r.status == "DATA_FAIL" and "weld plan not verified (D14)" in r.reason and "reachability not checked" in r.reason, \
        (r.status, r.reason, last_msgs(r))
    assert r.category == "waypoint"
    assert_out_dir_only_deliverables(out, plan=False)
    assert (out / "verification" / "weld_plan.json").exists() and (out / "verification" / "weld_plan.npz").exists()


# --------------------------------------------------------------------------------------------------------------------------
# (v) / (w) D14 and the weld torch (user 2026-10-12): the lab config (our torch on this arm) runs D14 with the URDF torch_tcp and
#     delivers; the generic mechanism (a config that says the weld torch is not on this arm) skips D14 and delivers with the note
# --------------------------------------------------------------------------------------------------------------------------
LAB_CONFIG_PATH = os.path.join(os.path.dirname(SIM_CONFIG), "hongik_lab_rb20_values.yaml")      # the lab platform's values (test data)


def test_v_weld_torch_declared_off_this_arm_skips_d14_and_still_delivers(calib, settings, tmp_path, monkeypatch):
    """The generic on_this_arm mechanism with the lab config's kiro_output (World := link0): a weld_torch declared not on this arm ->
    D14 is not run (the platform check is never called), the plan is delivered with the note in the summary / run log /
    weld_plan.json / kiro info + README, and kiro/ is written in our base frame."""
    from sim_arm import SimArmPlatform
    lab = C.load_settings(LAB_CONFIG_PATH)
    p = dict(settings.params)
    # a torch not on this arm: its nozzle end (the stick-out's reference) comes with the config - here our torch's, as derived from the
    # lab URDF's collision geometry (on its own arm the run derives it)
    from scan_core.weld_clearance import nozzle_end_mm, tool_body_from_urdf
    noz = nozzle_end_mm(tool_body_from_urdf(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rb20_1900es_u_eoat.urdf"),
                                                 encoding="utf-8").read(), "torch_tcp"))
    p["weld_torch"] = dict(lab.params["weld_torch"], on_this_arm=False, nozzle_end_mm=noz, nozzle_end_source="the lab URDF (test)")
    p["kiro_output"] = lab.params["kiro_output"]
    st = C.Settings(p, settings.path)
    called = []
    monkeypatch.setattr(SimArmPlatform, "check_weld_plan", lambda self, plan, **kw: called.append(1) or [])
    out = tmp_path / "out"
    with scene(st, calib, tmp_path, speed=FAST) as (cell, job, doc):
        at, r = run1g(st, cell, job, doc, calib, out, end_tol_mm=3.0)
    assert_delivered(r, at)
    assert called == []                                                     # no D14 on this arm, no other torch as a stand-in
    assert_out_dir_only_deliverables(out)
    assert_d14_skipped(out, r)
    ko = r.kiro_outputs
    assert ko["written"] is True and ko["world_from_base"]["status"] == "lab definition" and ko["tcp_frame"] == "torch_tcp"
    readme = (out / "kiro" / "README.md").read_text(encoding="utf-8")
    assert "Here World is defined as the robot base frame `link0`" in readme
    assert "The platform config says the weld torch is not on the arm that made this run" in readme


def test_w_lab_config_runs_d14_with_our_torch_on_this_arm(calib, settings, tmp_path):
    """The lab config (2026-10-12: the weld torch = our torch, on this arm): D14 runs with the URDF torch_tcp (the virtual arm answers
    flange <- torch_tcp from its URDF as TF would) · TCP <- tip (+20 mm along the torch axis) - every pass checked and reachable in the
    sim cell, each row naming the tool with the TCP it read; delivered without the skip note; kiro/ in our base frame (World := link0)
    with the TCP 20 mm back from the aim point."""
    import yaml
    from scipy.spatial.transform import Rotation
    lab = C.load_settings(LAB_CONFIG_PATH)
    p = dict(settings.params)
    p["weld_torch"], p["kiro_output"] = lab.params["weld_torch"], lab.params["kiro_output"]
    st = C.Settings(p, settings.path)
    out = tmp_path / "out"
    with scene(st, calib, tmp_path, speed=FAST) as (cell, job, doc):
        at, r = run1g(st, cell, job, doc, calib, out, end_tol_mm=3.0)
    assert_delivered(r, at)
    assert_d14_checked(out, r)
    assert (out / "kiro" / "pass_1.yaml").exists() and r.kiro_outputs["world_from_base"]["status"] == "lab definition"
    for wp in r.weld_plan.passes:                                           # World := link0: the TCP pose in our base frame
        d = yaml.safe_load((out / "kiro" / f"pass_{wp.k}.yaml").read_text(encoding="utf-8"))
        info = yaml.safe_load((out / "kiro" / f"pass_{wp.k}_info.yaml").read_text(encoding="utf-8"))
        R = Rotation.from_quat([d["current_goal"]["orientation_xyzw"][c] for c in "xyzw"]).as_matrix()
        tip = np.asarray(info["derived"]["straight_line"]["tip_base_m"]["goal"], dtype=float)
        off = (STICK_OUT_TEST_MM + r.weld_plan.torch.nozzle_end_mm) * 1e-3              # nozzle end + stick-out (2026-10-13)
        assert np.allclose([d["current_goal"]["position_m"][c] for c in "xyz"], tip - off * R[:, 2], atol=1e-12)


def test_x_provisional_stick_out_from_the_nozzle_every_pass_feasible_by_part_role(calib, settings, tmp_path):
    """The lab config at the provisional stick-out 20 mm - measured from the gas nozzle end (user 2026-10-13: TCP -> wire end = the nozzle
    end 1.2 mm, derived from the URDF, + 20 mm) - with the clearance requirement by part role (user 2026-10-13, KIRO: the gas nozzle
    almost touches the plate in the root): the gas nozzle and contact tip need only not penetrate the specimen, the other parts keep
    k x sigma (~6 mm). Before (one requirement for the whole tool, stick-out from the contact tip) the root had no feasible orientation;
    now D14 finds every pass feasible - the root's nozzle comes closer to the plates than k x sigma (the role split is what allows it),
    every other part keeps it - and the plan is delivered with each role's clearance recorded."""
    lab = C.load_settings(LAB_CONFIG_PATH)
    p = dict(settings.params)
    p["weld_torch"], p["kiro_output"] = lab.params["weld_torch"], lab.params["kiro_output"]
    st = C.Settings(p, settings.path)
    out = tmp_path / "out"
    with scene(st, calib, tmp_path, speed=FAST, stick_out_mm=20.0) as (cell, job, doc):
        at, r = run1g(st, cell, job, doc, calib, out, end_tol_mm=3.0)
    assert_delivered(r, at)
    assert_d14_checked(out, r)
    wt, req = r.weld_plan.torch, r.weld_plan.clearance
    assert wt.stick_out_mm == 20.0 and wt.tip_offset_mm == pytest.approx(21.2, abs=1e-3) and req["required_mm"] > 5.9
    root = r.weld_plan.passes[0]
    br = root.orientation["clearance_by_role"]
    assert root.role == "root" and root.orientation["chosen"] is not None
    assert 0.0 <= br["process"]["clearance_mm"] < req["required_mm"] <= br["other"]["clearance_mm"], br
    assert root.orientation["limiting_part"]["role"] == "process" and root.orientation["limiting_part"]["link"] == "eoat_base_link"
    import yaml
    d = json.loads((out / "weld_plan.json").read_text(encoding="utf-8"))
    assert d["torch_orientation"]["passes"][0]["clearance_by_role"]["process"]["required_mm"] == 0.0
    info = yaml.safe_load((out / "kiro" / "pass_1_info.yaml").read_text(encoding="utf-8"))["derived"]["orientation"]
    assert info["clearance_by_role"]["process"]["required_mm"] == 0.0 and info["limiting_part"]["role"] == "process"


def test_production_worker_has_no_model_selector():
    """제품 코드의 모형은 정식 모듈뿐 (환경 변수·대체 함수 없음); 대역은 시험의 monkeypatch 로만."""
    import inspect
    src = inspect.getsource(GW)
    assert "G1_MODEL_STUBS" not in src and "NotImplementedError" not in src and "environ.get" not in src
    m = GW.models()
    assert all(getattr(m, fn).__module__ == mod for fn, mod in GW.MODEL_FUNCS.items())


def test_worker_and_fit_children_run_at_the_lowest_cpu_priority():
    """작업 프로세스와 fork_map 맞춤 자식은 보통 시분할의 가장 낮은 우선순위 (legacy 스트림 자식과 같은 post_scan._lowest_priority) —
    서버의 ROS 실행기·가드를 굶기지 않는다."""
    r_fd, w_fd = os.pipe()
    w = GW.G1Worker(r_fd)
    try:
        info = w.call("ping", None)
        pid = os.fork()                                         # 이 OS 의 가장 큰 nice (상수 없이: 오르지 않을 때까지, 자식에서)
        if pid == 0:
            prev, cur = None, os.nice(0)
            while cur != prev:
                prev, cur = cur, os.nice(1)
            os._exit(cur)
        top = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
        assert info["nice"] == top and info["cpu_priority"] == f"nice {top}", info
        kids = GW.fork_map(lambda ch: os.nice(0), [0, 1], 2)  # 이 (시험) 프로세스에서 부른 fork_map 도 자식을 낮춘다
        assert kids == [top, top], kids
    finally:
        w.close()
        os.close(r_fd)
        os.close(w_fd)


# --------------------------------------------------------------------------------------------------------------------------
# (q) 실기 후보 속도 (사용자 2026-10-11): 60 mm/s @ 150 fps (간격 0.4 mm), 패스 예산 6, 끝 정밀도 0.06 mm
# --------------------------------------------------------------------------------------------------------------------------
def test_q_lab_speed_60mm_s_at_150fps_with_six_passes(calib, tmp_path):
    """The lab candidate's scan (user 2026-10-11): 60 mm/s at 150 fps (0.4 mm spacing), pass budget 6, end precision 0.06 mm, lab
    layout. The run is delivered with the ends near the truth, the sensor runs at the job's frame rate (the 1G core sets it), nothing
    stops the run, and every frame arrives (the sim clock's E2 check in sim_cell)."""
    st = sim_settings(max_passes=6)
    out = tmp_path / "out"
    with scene(st, calib, tmp_path, speed=0.06, job_fields={"profile_fps": 150.0}) as (cell, job, doc):
        world = cell.world
        doc = dict(doc)
        doc["precision"] = {"end_position_mm": 0.06, "root_position_mm": 0.05, "length_mm": 0.05}
        at, r = run1g(st, cell, job, doc, calib, out)
        fps_streams = sorted({round(float(row[4]), 3) for row in cell.fs.stream_log})
    assert_delivered(r, at)
    assert_ends_near_truth(world, r)
    assert fps_streams == [150.0], fps_streams                                   # every stream at the job's frame rate
    assert r.estop_record is None
    log = run_log(out)
    assert 1 <= len([p_ for p_ in log["legs"] if p_["kind"] == "pass"]) <= 6
    print(f"[q] status {r.status} passes {len(r.passes)}; legs " + "; ".join(
              f"{p_['name']} {p_['n_frames']} fr fit {p_['timing'].get('fit_leg_s', float('nan')):.2f} s "
              f"stream {bool((p_.get('stream') or {}))}" for p_ in log["legs"] if p_["kind"] == "pass"))


# --------------------------------------------------------------------------------------------------------------------------
# (r) 속도 계열 (사용자 2026-10-11 밤): 60 / 30 / 15 mm/s @ 150 fps, 패스 예산 3 (전체 패스 ~3 번), 보정된 잡음 사전, 안정 오버스캔
# --------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("speed", [0.06, 0.04, 0.03, 0.02, 0.015])
def test_r_speed_series_with_a_three_pass_budget(calib, tmp_path, speed):
    """The speed series (user 2026-10-11 night): 150 fps, pass budget 3, the lab calibration's noise priors. The run is delivered
    within three passes with the ends near the truth; every pass decision carries the settling record and the overscan's
    edge-response reach and settling terms; nothing stops the run (the longer overscan stays inside the envelope and MoveIt)."""
    st = sim_settings(max_passes=3)
    out = tmp_path / "out"
    with scene(st, calib, tmp_path, speed=speed, job_fields={"profile_fps": 150.0}) as (cell, job, doc):
        world = cell.world
        doc = dict(doc)
        doc["precision"] = {"end_position_mm": 0.06, "root_position_mm": 0.05, "length_mm": 0.05}
        at, r = run1g(st, cell, job, doc, calib, out)
    assert_delivered(r, at)
    assert_ends_near_truth(world, r)
    assert 1 <= len(r.passes) <= 3
    assert r.estop_record is None
    dec = [d for d in r.decisions if d.get("phase") == "pass" and str(d.get("msg", "")).startswith("pass ")
           and "laser hit overscan" in str(d.get("msg", ""))]
    assert dec and all("edge-response reach" in d["msg"] and "settling" in d and d["settling"]["n_legs"] >= 1 for d in dec), \
        [(d["msg"][-420:], d.get("settling")) for d in dec]
    est = json.loads(sorted((out / "verification" / "estimates").glob("est_*.json"), key=lambda p: int(p.stem.split("_")[1]))[-1]
                     .read_text(encoding="utf-8"))
    vc = (est["plate"].get("end_faces") or {}).get("variance_components") or {}
    assert (vc.get("priors") or {}).get("source", "").startswith("calibration"), vc


# --------------------------------------------------------------------------------------------------------------------------
# (t) 플랫폼 캘리브 경로 (2026-10-12 아침 실기): 서버는 Teach1G 에 MoveIt 어댑터의 캘리브 (calib_tf.calib_from_tf — 파일 캘리브 + 플랫폼
#     TF) 를 준다. 그 복사가 'g1:' 블록의 noise_priors·edge_response 를 빠뜨려, 실기 11 실행 모두 루트가 확립되지 않은 채 3 패스 예산을
#     다 썼다 (가상 팔 시험은 파일 캘리브를 바로 주어 이 경로를 지나지 않았다)
# --------------------------------------------------------------------------------------------------------------------------
def test_t_three_pass_budget_through_the_platform_calib_path_ends_pass(calib, tmp_path):
    """The run as the server starts it (2026-10-12 morning lab runs): the calib reaches Teach1G through the MoveIt adapter's
    calib_from_tf (file calib + the platform TF), 3-pass budget, 20 mm/s at 150 fps, the lab precision targets. The calibrated noise
    priors and the edge response reach the estimator (the root is established, the end model uses the calibrated sensor response) and
    the run ends PASS with the ends near the truth - not PASS_UNRESOLVED on an unestablished root."""
    calib_tf = pytest.importorskip("hongik_scan_ros.calib_tf")
    st = sim_settings(max_passes=3)
    out = tmp_path / "out"
    with scene(st, calib, tmp_path, speed=0.02, job_fields={"profile_fps": 150.0}) as (cell, job, doc):
        world = cell.world
        doc = dict(doc)
        doc["precision"] = {"end_position_mm": 0.06, "root_position_mm": 0.05, "length_mm": 0.05}
        at, r = run1g(st, cell, job, doc, calib, out,
                      platform_calib=lambda c: calib_tf.calib_from_tf(c, c.T_flange_opt, c.T_flange_eoat, note="sim platform TF"))
    assert at.calib.source.endswith("+tf(sim platform TF)"), at.calib.source
    assert r.status == "PASS" and r.converged == "true", (r.status, r.reason, last_msgs(r))
    assert_ends_near_truth(world, r)
    assert 1 <= len(r.passes) <= 3
    log = run_log(out)
    g1 = log["calib_effective"]["g1"]
    assert g1["noise_priors"] == calib.noise_priors and g1["edge_response"] == calib_1g(calib).edge_response, g1
    assert log["prior"]["noise_priors"] == calib.noise_priors
    est = json.loads(sorted((out / "verification" / "estimates").glob("est_*.json"), key=lambda p: int(p.stem.split("_")[1]))[-1]
                     .read_text(encoding="utf-8"))
    pl = est["plate"]
    assert pl["root_established"] == [True, True] and est["done"] is True, (pl["root_established"], pl["leg_offset_nu"], est["ratios"])
    ef = pl["end_faces"]
    assert ef["variance_components"]["priors"]["source"].startswith("calibration"), ef["variance_components"]
    assert not str(ef["sensor"]["source"]).startswith("online"), ef["sensor"]


# --------------------------------------------------------------------------------------------------------------------------
# (u) the section dims are the passes' (user 2026-10-12, KIRO 1G line): the coarse legs (transit, dwells — another standoff,
#     lateral position, yaw, incidence) do not inform them. Lab kv_offA_a (rough points ±30 mm aside): gap 6.00 / 5.74 / 5.40 mm,
#     bevel 27.4 / 28.9 / 30.2° after 1 / 2 / 3 passes with the transit in the dims (transit alone: 6.90 mm, 23.4°)
# --------------------------------------------------------------------------------------------------------------------------
def test_u_dims_from_the_passes_teaching_offset_no_bias_no_drift(calib, tmp_path):
    """kv_offA_a-like teaching: rough points 30 mm to either side of the seam (the taught line ~10° off it), 60 mm/s at 150 fps,
    pass budget 3. The coarse legs inform no dim: before the first pass the run has none (σ ∞), so a coarse-pose systematic cannot
    bias them; after pass 1 they do not move with the pass count beyond k·σ. (The virtual sensor has no pose-class section
    systematic — its transit reads the design within µm — so the bias itself is checked at the estimator level with an injected
    coarse-class systematic, test_seam_estimate.test_dims_are_the_passes_a_biased_coarse_leg_does_not_move_them.)"""
    st = sim_settings(max_passes=3)
    out = tmp_path / "out"
    with scene(st, calib, tmp_path, lat_S=0.030, lat_E=-0.030, speed=0.06, job_fields={"profile_fps": 150.0}) as (cell, job, doc):
        doc = dict(doc)
        doc["precision"] = {"end_position_mm": 0.06, "root_position_mm": 0.05, "length_mm": 0.05}
        at, r = run1g(st, cell, job, doc, calib, out)
    assert_delivered(r, at)
    log = run_log(out)
    kinds = [lg_["kind"] for lg_ in log["legs"]]
    by_pass = {}
    for e in log["estimates"]:
        n_pass = kinds[:int(e["legs"])].count("pass")
        est = json.loads((out / "verification" / "estimates" / f"est_{int(e['n'])}.json").read_text(encoding="utf-8"))
        by_pass[n_pass] = est["plate"]["seam_dims"]
    assert 0 in by_pass and max(by_pass) >= 2, sorted(by_pass)
    k = float(at.k)
    for q in ("gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm", "hi_lo_mm"):
        d0 = by_pass[0][q]
        assert d0["sigma"] is None or not math.isfinite(float(d0["sigma"])), (q, d0)               # coarse legs only: no dims
        last = by_pass[max(by_pass)][q]
        for n_pass in sorted(by_pass)[1:]:
            dk = by_pass[n_pass][q]
            assert math.isfinite(float(dk["sigma"])), (q, n_pass, dk)
            assert abs(float(dk["value"]) - float(last["value"])) <= k * float(dk["sigma"]), (q, n_pass, dk, last)   # no drift


# --------------------------------------------------------------------------------------------------------------------------
# (s) 이동 속도 부류 (사용자 2026-10-12): 측정 (패스·드웰 자리의 정밀 보정) = 작업 속도 15 mm/s, 그 밖의 모든 이동 (시작 자세·획득·하강·
#     거친 S→E 트랜싯 (녹화)·패스 재배치·접근) = 플랫폼 한계 (연구실 설정: TCP 60 mm/s, 0.1 m/s², 관절 0.24 rad/s), 예산 3, 150 fps
# --------------------------------------------------------------------------------------------------------------------------
LAB_CONFIG = os.path.join(os.path.dirname(SIM_CONFIG), "hongik_lab_rb20_values.yaml")           # the lab platform's values (test data)


def lab_limits_settings(**auto_teach) -> C.Settings:
    """The sim settings with the lab platform's motion limits (hongik_lab_rb20_moveit.yaml moveit_jtc.limits) - the simulated arm and
    the motion config take their limits from the settings."""
    lab = C.load_settings(LAB_CONFIG)
    st = sim_settings(**auto_teach)
    p = dict(st.params)
    p["moveit_jtc"] = {**dict(p.get("moveit_jtc") or {}), "limits": dict(lab.params["moveit_jtc"]["limits"])}
    return C.Settings(p, st.path)


def _speed_class_run(calib, tmp_path, *, job_speed: float, measure_speed_mps=None, tag: str = "s"):
    """One lab-limits run (lab motion limits, pass budget 3, 150 fps, the lab precision targets) on a job with speed_mps job_speed,
    with the run's measurement-speed override measure_speed_mps (None = the job's), then the speed-class checks shared by (s) / (s2):
    every executed move's class from its purpose and its commanded speed (measurement = the run's measurement speed, positioning =
    the platform TCP limit), the recorded legs' jobs at their class's speed, the pass overscan's ramp / settling terms at the
    measurement speed, the setup line and the run log's measurement-speed record. -> (at, r, world, v_m, run log)."""
    from scan_l0.teach_1g import MOVE_MEASURE, MOVE_POSITION
    st = lab_limits_settings(max_passes=3)
    lim = st.params["moveit_jtc"]["limits"]
    v_m = float(job_speed if measure_speed_mps is None else measure_speed_mps)
    out = tmp_path / "out"
    with scene(st, calib, tmp_path, speed=job_speed, job_fields={"profile_fps": 150.0}, lat_S=0.025, lat_E=-0.02) as (cell, job, doc):
        world = cell.world
        doc = dict(doc)
        doc["precision"] = {"end_position_mm": 0.06, "root_position_mm": 0.05, "length_mm": 0.05}
        kw = {} if measure_speed_mps is None else {"measure_speed_mps": measure_speed_mps}
        at, r = run1g(st, cell, job, doc, calib, out, local_speed_mps=None, descend_speed_mps=None, transit_speed_mps=None, **kw)
    assert float(job.speed_mps) == pytest.approx(job_speed)                       # the job itself is not changed by the override
    v_p = float(lim["max_tcp_speed_mps"])
    assert float(at.cfg.v_eff_max_mps) == pytest.approx(v_p) and float(at.accel) == pytest.approx(float(lim["max_tcp_accel_mps2"]))
    mv = executed(r)
    rows = []
    for m in mv:
        mo = m.get("motion") or {}
        rows.append(f"  {m['name']:<22} {m['speed_class']:<8} {m['speed_mps'] * 1e3:5.1f} mm/s  {m['dist_mm']:6.1f} mm  "
                    f"{float(mo.get('elapsed_s') or mo.get('motion_duration_s') or float('nan')):6.2f} s  (t {m['t']:.1f} s)")
    print(f"[{tag}] moves (name, speed class, commanded speed, chord, motion time):\n" + "\n".join(rows)
          + f"\n[{tag}] {r.status}: {len(r.passes)} pass(es), sim time {r.elapsed_s:.1f} s")
    assert_delivered(r, at)
    assert_ends_near_truth(world, r)
    assert 1 <= len(r.passes) <= 3
    # every move's class from its purpose: passes and the dwell corrections measure, everything else positions
    is_meas = lambda nm: re.fullmatch(r"pass\d+", nm) is not None or re.fullmatch(r"[SE]_corr\d+", nm) is not None  # noqa: E731
    assert all(m["speed_class"] == (MOVE_MEASURE if is_meas(m["name"]) else MOVE_POSITION) for m in mv), \
        [(m["name"], m["speed_class"]) for m in mv]
    assert all(m["speed_mps"] == pytest.approx(v_m if is_meas(m["name"]) else v_p) for m in mv), \
        [(m["name"], m["speed_mps"]) for m in mv]
    nm = [m["name"] for m in mv]
    assert "transit_S_to_E" in nm and any(re.fullmatch(r"[SE]_corr\d+", n_) for n_ in nm) and any(n_.endswith("_approach") or
                                                                                                    n_.endswith("_reposition") for n_ in nm), nm
    assert any(m["phase"] == "joint" for m in mv) and any(n_.startswith("acquire_") and n_.endswith("_in") for n_ in nm), nm
    # the recorded legs carry their class's speed (reconstruction, settling evidence, spacing)
    log = run_log(out)
    legs = [li for li in log["legs"] if li["kind"] in ("transit", "pass")]
    for li in legs:
        with np.load(os.path.join(li["dir"], "raw.npz"), allow_pickle=False) as raw:
            jd = json.loads(str(raw["job_json"]))
        assert float(jd["speed_mps"]) == pytest.approx(v_p if li["kind"] == "transit" else v_m), (li["name"], jd["speed_mps"])
    assert log["job_coarse"]["speed_mps"] == pytest.approx(v_p), log["job_coarse"]["speed_mps"]
    # the pass overscan: the speed ramp v^2/2a and the settling v t_settle at the pass speed (the transit at 60 mm/s is settling evidence)
    dec = [d for d in r.decisions if d.get("phase") == "pass" and str(d.get("msg", "")).startswith("pass ")
           and "laser hit overscan" in str(d.get("msg", ""))]
    ramp = v_m ** 2 / (2.0 * float(at.accel))
    assert dec and all(f"speed ramp v^2/2a {ramp * 1e3:.2f}" in d["msg"] and f"at {v_m * 1e3:.0f} mm/s" in d["msg"]
                       and d["settling"]["n_legs"] >= 1 for d in dec), [d["msg"][-420:] for d in dec]
    assert all(f"settling v t_settle {v_m * float(d['settling']['t_settle_s']) * 1e3:.2f}" in d["msg"] for d in dec), \
        [d["msg"][-420:] for d in dec]
    assert all(d["overscan"] and d["plan"] for d in dec)
    setup = [d for d in r.decisions if d.get("step") == "setup" and str(d.get("msg", "")).startswith("move speeds")]
    fps = 150.0
    assert setup and f"measurement {v_m * 1e3:.1f} mm/s" in setup[0]["msg"] and "positioning 60.0 mm/s" in setup[0]["msg"] \
        and "joints 0.24 rad/s" in setup[0]["msg"] and f"spacing {v_m / fps * 1e3:.2f} mm at 150 fps" in setup[0]["msg"], setup
    # the run's measurement-speed record (setup decision and run log): the source and the job file's value
    rec = log["measure_speed"]
    assert rec == setup[0]["measure_speed"], (rec, setup[0]["measure_speed"])
    assert rec["speed_mps"] == pytest.approx(v_m) and rec["commanded_mps"] == pytest.approx(v_m) \
        and rec["job_file_speed_mps"] == pytest.approx(job_speed), rec
    assert log["job_rough"]["speed_mps"] == pytest.approx(v_m)
    if measure_speed_mps is None:
        assert rec["override_mps"] is None and rec["source"] == "job file speed_mps" and "(the job's speed_mps;" in setup[0]["msg"], rec
    else:
        assert rec["override_mps"] == pytest.approx(measure_speed_mps) and rec["source"] == "run override (measure_speed_mps)", rec
        assert (f"(run override measure_speed_mps; the job file's speed_mps {job_speed * 1e3:.1f} mm/s; spacing" in setup[0]["msg"]), \
            setup[0]["msg"]
    print(f"[{tag}] recorded legs: " + "; ".join(f"{li['name']} {li['n_frames']} frames" for li in legs))
    return at, r, world, v_m, log


def test_s_move_speed_classes_measure_at_the_job_speed_and_position_at_the_platform_limits(calib, tmp_path):
    """User 2026-10-12: the measurement moves (the passes, the precise corrections at the dwell sites) at the job's measurement
    speed (15 mm/s, 150 fps); every other move (start pose, acquisition joint move, descent onto the S dwell, the coarse S->E transit
    - still recording -, pass repositions and approaches) at the platform's positioning limits (the lab config: TCP 60 mm/s,
    joints 0.24 rad/s; no operator lowering). Each executed move records its speed class; each recorded leg's job carries its
    class's speed (the transit's 0.4 mm spacing, the passes' 0.1 mm); the pass overscan's ramp / settling terms use the pass speed.
    Pass budget 3, the lab precision targets: the run ends PASS with the ends near the truth."""
    at, r, _world, _v_m, _log = _speed_class_run(calib, tmp_path, job_speed=0.015)
    assert r.status == "PASS" and r.converged == "true", (r.status, r.reason, last_msgs(r))


# --------------------------------------------------------------------------------------------------------------------------
# (s2) 실행별 측정 속도 (사용자 2026-10-12: 15 mm/s 는 실제로 너무 느리다 — 실행마다 15 / 30 mm/s 를 인자로): 클라이언트 --measure-speed 30
#      → AutoTeach 목표 measure_speed_mps 0.03 → 서버 → Teach1G(measure_speed_mps=0.03) = 이 실행의 작업 속도 (작업 파일 15 mm/s 는 그대로)
# --------------------------------------------------------------------------------------------------------------------------
def test_s2_measure_speed_override_30mm_s_passes_and_corrections_at_30_positioning_at_60(calib, tmp_path):
    """User 2026-10-12: the measurement speed is chosen per run (--measure-speed 30 -> the goal's measure_speed_mps 0.03 -> the server
    -> Teach1G measure_speed_mps), here on the 15 mm/s job (job_g1_lab4_v15-like): the passes and the precise dwell corrections run at
    30 mm/s (spacing 0.20 mm at 150 fps), every other move at the platform's positioning limits (60 mm/s, 0.24 rad/s); the recorded
    passes, the overscan's speed ramp v^2/2a and settling v t_settle follow the 30 mm/s; the setup line and the run log name the
    override and the job file's 15 mm/s. Pass budget 3, the lab precision targets: delivered (PASS, or PASS_UNRESOLVED with the paths
    when the 0.06 mm end target is not met in three passes) with the ends near the truth. The noise priors' speed model takes the
    measured along-track speeds (the calibrated priors reach the end-face estimate)."""
    at, r, _world, v_m, log = _speed_class_run(calib, tmp_path, job_speed=0.015, measure_speed_mps=0.03, tag="s2")
    assert v_m == pytest.approx(0.03) and at.job0.speed_mps == pytest.approx(0.03) and at.measure_speed_job_mps == pytest.approx(0.015)
    if r.status == "PASS_UNRESOLVED":                                            # the paths are still delivered
        assert r.weld_plan is not None, r.reason
    est = json.loads(sorted((tmp_path / "out" / "verification" / "estimates").glob("est_*.json"),
                            key=lambda p: int(p.stem.split("_")[1]))[-1].read_text(encoding="utf-8"))
    vc = (est["plate"].get("end_faces") or {}).get("variance_components") or {}
    assert (vc.get("priors") or {}).get("source", "").startswith("calibration"), vc
    passes = [li for li in log["legs"] if li["kind"] == "pass"]
    print(f"[s2] {r.status}: {len(passes)} pass(es) at {v_m * 1e3:.0f} mm/s, frames per pass "
          f"{[li['n_frames'] for li in passes]}, end sigma ratio {est.get('ratios')}")


# --------------------------------------------------------------------------------------------------------------------------
# (x) the joint-state source (user 2026-10-12, scan_core.js_source): the virtual arm's /joint_states as a measured encoder stream
#     (servo lag, vibration, dither, count lattice) or as the control box's reference echo (look-ahead filter, bit-exact hold; the
#     physical arm 25 ms behind — a lag the timing prior does not know). Default run: transit at 60 mm/s, passes at 30, 3-pass budget
# --------------------------------------------------------------------------------------------------------------------------
def _x_run(calib, tmp_path, source: str) -> dict:
    """One 3-pass run with the virtual arm's joint-state model 'default' (encoder counts, no lag), 'measured' (+ 11 ms servo lag,
    vibration, dither) or 'reference' (echo, the arm 25 ms behind) → its checks' inputs."""
    from sim_arm import JointStateModel, SimArmConfig
    st = sim_settings(max_passes=3)
    js = {"default": JointStateModel(),
          "measured": JointStateModel("measured", servo_lag_s=0.011, dither_counts=0.4, vib_amp_rad=2e-5, seed=3),
          "reference": JointStateModel("reference", lookahead_s=0.03, echo_lag_s=0.025, seed=3)}[source]
    arm_cfg = dataclasses.replace(SimArmConfig.from_settings(st), joint_states=js)
    out = tmp_path / source
    (tmp_path / f"{source}_job").mkdir()
    with scene(st, calib, tmp_path / f"{source}_job", speed=0.03, job_fields={"profile_fps": 150.0}, arm_cfg=arm_cfg) as (cell, job, doc):
        world = cell.world
        doc = dict(doc)
        doc["precision"] = {"end_position_mm": 0.06, "root_position_mm": 0.05, "length_mm": 0.05}
        at, r = run1g(st, cell, job, doc, calib, out)
    assert_delivered(r, at)
    log = run_log(out)
    est = json.loads(sorted((out / "verification" / "estimates").glob("est_*.json"), key=lambda p: int(p.stem.split("_")[1]))[-1]
                     .read_text(encoding="utf-8"))
    ends = {}
    for X, true_s in (("S", 0.0), ("E", world.L * 1e3)):
        e = r.extent[X]
        assert e["observed"]
        ends[X] = (taught_s_mm(world, e["point"]) - true_s, 1e3 * float(e["sigma_m"]))
    # the planning ends after the first pass (the next pass's extents): the estimate made with the first pass leg
    first = 1 + [li["kind"] for li in log["legs"]].index("pass")
    e1 = [e_ for e_ in log["estimates"] if e_["legs"] == first][0]
    pe1 = json.loads((out / "verification" / "estimates" / f"est_{e1['n']}.json").read_text(encoding="utf-8"))["plate"]["ends_plan"]
    plan1 = {X: (taught_s_mm(world, pe1[X]["point_m"]) - true_s, 1e3 * float(pe1[X]["sigma_m"]))
             for X, true_s in (("S", 0.0), ("E", world.L * 1e3))}
    jss = log["joint_state_source"]
    to = est["plate"]["time_offset"]
    print(f"[x {source}] plan ends after pass 1: err S {plan1['S'][0]:+.3f} (sigma {plan1['S'][1]:.3f}) E {plan1['E'][0]:+.3f} "
          f"(sigma {plan1['E'][1]:.3f}) mm")
    print(f"[x {source}] {r.status}, {len(r.passes)} passes; joint states {jss['hypothesis']} (ln odds ref "
          f"{jss['ln_odds_reference']:.1f}; lattice {jss['evidence']['ln_bf_measured_vs_reference']['lattice']:.1f}, rest "
          f"{jss['evidence']['ln_bf_measured_vs_reference']['rest']:.1f}); delta {to['delta_s'] * 1e3:.2f} +- {to['sigma_s'] * 1e3:.2f} ms; "
          f"ends err S {ends['S'][0]:+.3f} (sigma {ends['S'][1]:.3f}) E {ends['E'][0]:+.3f} (sigma {ends['E'][1]:.3f}) mm")
    return {"at": at, "r": r, "log": log, "ends": ends, "to": to, "plan1": plan1}


def test_x_joint_state_source_detected_delta_prior_switched_ends_unbiased(calib, tmp_path):
    """The run tells its joint states apart from the recorded values and its own commands only: the encoder streams (the default
    arm; with servo lag, vibration and dither) are 'measured', the echo 'reference' — then the time-offset prior carries the unknown
    lag (p_ref = 1, δ the data's: 25 ms within k·σ), the run output warns that the echo cannot show a blocked arm, and the
    delivered ends stay where the arm without any lag puts them (k·σ of the difference) although the physical arm trails the reported
    poses by 25 ms (0.75 mm along the seam at 30 mm/s; both pass directions). (The virtual cell's own end systematic at 30 mm/s —
    S +0.12, E −0.15 mm, ~2σ, with no lag at all — is the scene's, the same for every source; this test compares against it.)"""
    from scan_core.seam_estimate import TRACKING_MONITOR_ECHO_WARNING as W
    runs = {src: _x_run(calib, tmp_path, src) for src in ("default", "measured", "reference")}
    for src, x in runs.items():
        want = "reference" if src == "reference" else "measured"
        jss = x["log"]["joint_state_source"]
        assert jss["hypothesis"] == want, (src, {k_: jss.get(k_) for k_ in ("hypothesis", "p_reference_echo", "evidence")})
        last = x["log"]["estimates"][-1]["direction_time"]["joint_state"]
        assert last["hypothesis"] == want and last["delta_prior"]["p_ref"] == pytest.approx(1.0 if want == "reference" else 0.0,
                                                                                           abs=1e-9), (src, last)
        lines = [d_ for d_ in x["r"].decisions if d_["step"] == "joint_states"]
        assert lines and lines[0]["msg"].startswith(f"joint states: {want} (P(reference echo)"), (src, lines)
        assert (W in x["r"].notes and any(d_["msg"] == W for d_ in lines)) == (want == "reference"), src
    k = float(runs["default"]["at"].k)
    for src in ("measured", "reference"):
        for X in ("S", "E"):
            (e0, s0), (e1, s1) = runs["default"]["ends"][X], runs[src]["ends"][X]
            assert abs(e1 - e0) <= k * math.hypot(s0, s1), (src, X, e0, s0, e1, s1)
    to = runs["reference"]["to"]
    assert abs(to["delta_s"] - 0.025) <= k * to["sigma_s"] and to["sigma_s"] < 0.025, to       # the lag is the data's, not 0
    # the next pass is planned from the ends after the first one: under every source they hold the true ends within k·σ, with a σ
    # the next pass can cover (2026-10-13, test_m on the fake MoveIt arm: an echo's flat lag took them ±90–133 mm)
    for src, x in runs.items():
        for X in ("S", "E"):
            err, sg = x["plan1"][X]
            assert math.isfinite(sg) and sg < 30.0 and abs(err) <= k * sg, (src, X, err, sg)
