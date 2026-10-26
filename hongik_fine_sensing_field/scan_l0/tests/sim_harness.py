"""가상 팔 (sim_arm) + 가짜 센서 (GrooveWorld 렌더) 로 자동 교시를 도는 시험 도우미 — 범용 시작 (접힌 자세에서, 착점 좌표·자세·규격만).

2026-10-07: RB 제어박스 어댑터 제거 뒤 빠른 합성 종단 시험의 공용 틀. 세계는 GrooveWorld(pose_fn = 가상 팔의 실측 플랜지
자세), 플랫폼 설정은 가상 팔 설정 (tests/data/hongik_rb20_fake_moveit.yaml) 을 기본으로 한다.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Optional
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import yaml

from groove_world import FAKE_SENSOR_EDGE_RESPONSE, JOB_YAML, JOB_YAML_3G, LAB_E_TRUE, LAB_S_TRUE, lab_job_doc, make_world, make_world_3g
from scan_core.frames import make_transform, unit
from scan_core.path_gen import laser_plane_line_param
from scan_core.types import Calib
from scan_l0 import config as C
from scan_l0.autoteach import AutoTeach
from scan_l0.session import ScanSession
import render_prefetch
import sim_clock
from sim_arm import FOLDED_Q, SimArmConfig, SimArmPlatform
from vc3dz_driver import params as VP
from vc3dz_driver.clock import install
from vc3dz_driver.fake_sensor import FakeSensor
from vc3dz_driver.virtual_clock import VirtualClock

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.dirname(os.path.dirname(HERE))
SIM_CONFIG = os.path.join(HERE, "data", "hongik_rb20_fake_moveit.yaml")
CALIB = os.path.join(STACK, "eoat_description", "calib", "eoat_lab_01.yaml")
FPS = 20.0
N_POINTS = 704
LENGTH_TOL_MM = 6.0                # 빠른 패스 (착점 간격 수 mm) 시나리오의 길이 허용치 (qc_targets.weld_length_mm.tol)
SENSOR_INIT = dict(VP.INIT_STANDARD, AutoTriggerFPS=FPS)
# 실기 센서 시각 모형 (test_autoteach_e2e 와 같은 이유): 늦은 촬영은 '촬영 멈춤' 으로, 고정 처리 지연 40 ms (호스트 calib dt_s 로 보정)
SENSOR_REALTIME = dict(stall_late_ms=4.0, latency_ms=40.0)


def sim_settings(path: str = SIM_CONFIG, **auto_teach) -> C.Settings:
    s = C.load_settings(path)
    if auto_teach:
        p = dict(s.params)
        at = dict(p.get("auto_teach") or {})
        at.update(auto_teach)
        p["auto_teach"] = at
        s = C.Settings(p, s.path)
    return s


# the fake sensor's edge response: groove_world.FAKE_SENSOR_EDGE_RESPONSE (an exact ray cast — no laser-sheet spread)


def calib_for_sensor(calib: Calib, sensor_kw) -> Calib:
    """가짜 센서를 아는 호스트 캘리브: 고정 처리 지연 (dt_s = latency_ms) 과 모서리 응답 (광선 추적 — 띠 퍼짐 없음)."""
    return dataclasses.replace(calib, dt_s=float((sensor_kw or {}).get("latency_ms", 0.0)) * 1e-3,
                               edge_response=dict(FAKE_SENSOR_EDGE_RESPONSE))


def rot_axis(axis, deg: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(unit(axis) * math.radians(deg)).as_matrix()


def rolled_normal(S, E, roll_deg: float) -> np.ndarray:
    return rot_axis(np.asarray(E, dtype=float) - np.asarray(S, dtype=float), roll_deg) @ np.array([0.0, 0.0, 1.0])


def perturbed_calib(calib: Calib, d_mm=(1.2, -1.2, 1.0), axis=(1.0, 1.0, 1.0), deg: float = 1.0) -> Calib:
    T = np.array(calib.T_flange_opt, dtype=float, copy=True)
    T[:3, :3] = T[:3, :3] @ rot_axis(axis, deg)
    T[:3, 3] += np.asarray(d_mm, dtype=float) * 1e-3
    return Calib(T_flange_opt=T, eoat_yaw_rad=calib.eoat_yaw_rad, T_flange_eoat=np.array(calib.T_flange_eoat, copy=True), source="perturbed")


@contextmanager
def sim_cell(settings, calib_nominal: Calib, calib_true: Calib, *, three_g: bool = False, q0=FOLDED_Q, sensor_kw=None,
             arm_cfg: SimArmConfig = None, **world_kw):
    """가상 팔 (접힌 자세) + 세계 (calib_true 로 렌더) + 가짜 센서 → SimpleNamespace(arm, world, fs, session, clock).

    시뮬레이션 시계 (sim_clock): 팔·세계·센서를 만들기 전에 가상 시계를 설치한다 (SCAN_TEST_CLOCK=real 또는 @pytest.mark.clock_real 이면
    오늘과 같은 실시간). 끝: 세션 센서 닫기 → 가짜 센서 정지 → 팔 닫기 → E2 시간 불변식 검사 (본문이 통과했을 때만 실패로) → 시계 종료."""
    vc = VirtualClock() if sim_clock.use_virtual() else None
    with (install(vc) if vc is not None else nullcontext()):
        with (sim_clock.observe_client_frames() if vc is not None else nullcontext([])) as seen:
            arm = SimArmPlatform(settings, C.attach_tool_body(calib_nominal, settings), cfg=arm_cfg, q0=q0)
            kw = dict(n_points=N_POINTS, pose_fn=arm.pose.physical_pose)     # the sensor sees the physical arm (sim_arm JointStateModel)
            kw.update(world_kw)
            world = (make_world_3g if three_g else make_world)(calib_true, **kw)
            fs = FakeSensor(frame_fn=world.frame, params={"AutoTriggerFPS": FPS}, **(sensor_kw or {}))
            pf = None
            if vc is not None and render_prefetch.n_workers() > 0:      # 2단계: 렌더 선계산 (가상 시계에서만; 내용은 같다 — E7)
                pf = render_prefetch.RenderPrefetcher(world, arm)
                world._prefetch = pf
                arm.trajectory_listeners.append(pf.on_trajectory)
                fs.stream_listeners.append(pf.on_stream)
            fs.start()
            cell = SimpleNamespace(arm=arm, world=world, fs=fs, session=ScanSession(robot=arm, sensor=None, owned=False), clock=vc,
                                   e2=None, prefetch=pf)
            body_ok = False
            try:
                yield cell
                body_ok = True
            finally:
                ss = cell.session.sensor
                if ss is not None:
                    try:
                        ss.close()
                    except Exception:  # noqa: BLE001
                        pass
                fs.stop()
                arm.close()
                if pf is not None:
                    pf.close()
                    world._prefetch = None
                if vc is not None:
                    report = vc.shutdown()
                    errs, summary = sim_clock.e2_check(fs, arm, seen, report, fps=float(fs.params.get("AutoTriggerFPS", FPS)),
                                                       latency_ms=fs.latency_ms, start_delay_s=fs.start_delay_s)
                    if pf is not None:
                        summary["prefetch"] = dict(pf.stats)
                    cell.e2 = {"errors": errs, **summary}
                    print(f"[sim clock] E2 {'OK' if not errs else 'FAIL'}: {summary}" + (f" errors {errs[:3]}" if errs else ""))
                    if body_ok and errs:
                        raise AssertionError("E2 sim-clock invariants: " + "; ".join(errs[:6]))


def rough_points(world, lat_S_m: float, lat_E_m: float, dz_m: float = 0.020, inside_S_m: float = 0.030,
                 inside_E_m: float = 0.030) -> tuple:
    """거친 S0/E0 (착점): 진짜 끝에서 안쪽으로 inside, 측면 lat, 판 법선 쪽 dz."""
    S0 = world.S + world.u * inside_S_m + world.lateral * lat_S_m + world.n * dz_m
    E0 = world.E - world.u * inside_E_m + world.lateral * lat_E_m + world.n * dz_m
    return S0, E0


STICK_OUT_TEST_MM = 25.0


def write_rough_job(path, S0, E0, *, three_g: bool = False, bevel_side: str = "auto", length_tol_mm: float = LENGTH_TOL_MM,
                    position="keep", stick_out_mm: Optional[float] = STICK_OUT_TEST_MM, **job_fields) -> tuple:
    """템플릿 (1G/3G) 에 착점 좌표만 넣은 거친 작업 (meta.rough, spec bevel auto, fps 20, 길이 허용치) → (job, doc)."""
    doc = lab_job_doc(three_g)                       # the template with the lab specimen of the sim world (groove_world.LAB_SPECIMEN)
    doc["job"].update(start=[float(v) for v in S0], end=[float(v) for v in E0], profile_fps=FPS, job_id="rough_test")
    doc["job"].update(job_fields)
    if position is None:
        doc["job"].pop("position", None)
    elif position != "keep":
        doc["job"]["position"] = position
    doc["specimen"]["bevel_side"] = bevel_side
    if not three_g and isinstance(doc.get("weld_plan"), dict) and "work_angle_deg" in doc["weld_plan"]:
        # 가상 셀은 심이 로봇 앞을 가로지르는 0.6 m 배치: 템플릿 WPS (KIRO 데이터셋) 의 작업각을 베벨면 쪽으로 기울이면 (2026-10-12 의 22 / 55,
        # 28 / 28° 에서 확인; 2026-10-13 부터 템플릿은 KIRO 9 월 용접의 −9 / +20, −7 / −7°)
        # 베벨 판이 먼 쪽인 세계에서 토치가 로봇 쪽으로 누워 팔꿈치 한계 (D14 도달 못함; 연구실 실제 배치 k20_2_a 와 MoveIt 시험 배치는 양쪽
        # 다 닿는다 — 2026-10-12 fsv1_work/kiroout/reach_layouts.py). 흐름 시험은 이 가상 배치의 도달 때문에 작업각 0 (진행각·그 밖의
        # KIRO 값은 그대로; 작업각의 부호 = 베벨이 KIRO 진행의 오른쪽, 사용자 확인 2026-10-13) — 세계의 베벨 쪽과 무관하게 닿는다
        doc["weld_plan"]["work_angle_deg"] = [0.0, 0.0, 0.0]
    if isinstance(doc.get("weld_plan"), dict) and stick_out_mm is not None:
        # the weld-pose clearance (2026-10-13): the flow tests run the delivery path at a test stick-out (STICK_OUT_TEST_MM, a test
        # input, not a system value) - chosen when one requirement held the whole tool (k 3 x the 2 mm residual = 6 mm) and the root had
        # no feasible orientation at 20 mm; with the stick-out from the gas nozzle end and the requirement by part role (the nozzle and
        # contact tip 0 mm) the root is feasible at 20 mm too - test_sim_teach1g.test_x pins that outcome
        doc["weld_plan"]["stick_out_mm"] = float(stick_out_mm)
    doc.setdefault("qc_targets", {})["weld_length_mm"] = {"tol": float(length_tol_mm)}
    doc["meta"] = {"rough": True, "note": "test rough job"}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return C.load_job(path)


RUN_KW = dict(confirm="none", transit_speed_mps=0.06, local_speed_mps=0.04, descend_speed_mps=0.04,
              sensor_init=SENSOR_INIT, dwell_frames=20, probe_frames=10)


def run_autoteach(settings, cell, job, doc, calib, out, **kw):
    a = dict(RUN_KW)
    a.update(session=cell.session, release_sensor=True, sensor_ip="127.0.0.1", sensor_port=cell.fs.port)
    a.update(kw)
    at = AutoTeach(settings, job, doc, calib, out_dir=str(out), log=a.pop("log", lambda s: None), **a)
    return at, at.run()


def executed(r) -> list:
    return [m for m in r.moves if m.get("executed")]


def angle_deg(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(unit(a) @ unit(b))))))


def rot_between_deg(Ra, Rb) -> float:
    R = np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float)
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


def perp_to_true_root(world, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    c = world.S - world.n * (world.spec.thickness_mm * 1e-3)
    rel = P - c
    perp = rel - np.outer(rel @ world.u, world.u)
    return np.linalg.norm(perp, axis=1)


def plan_origins(r) -> np.ndarray:
    assert r.weld_plan is not None, (r.status, r.reason)
    return np.array([st.origin for st in r.weld_plan.stations])


def assert_plan_runs_S_to_E(world, r, tol_mm: float = LENGTH_TOL_MM) -> None:
    """용접 계획의 S→E 계약 (2026-10-09, 실기 b1 t07·b3 t05 거울상 계획): 첫 단면 = 진짜 S 끝, 마지막 = 진짜 E 끝 (심 방향, 길이 허용 안),
    단면 축이 S→E, 패스마다 표기한 방향의 끝에서 시작 — 마지막 전체 패스가 어느 방향이었든. (수직 거리만 보는 perp_to_true_root 는
    거울상 계획도 통과시킨다.)"""
    O = plan_origins(r)
    # 이 실행의 S = 작업의 시점 (3G 배치는 세계의 E 쪽을 S 로 줄 수 있다) — 진짜 끝 중 정련 작업의 시점에 가까운 쪽
    S_w, E_w = np.asarray(world.S, dtype=float), np.asarray(world.E, dtype=float)
    if np.linalg.norm(np.asarray(r.job_final.start) - E_w) < np.linalg.norm(np.asarray(r.job_final.start) - S_w):
        S_w, E_w = E_w, S_w
    u = (E_w - S_w) / np.linalg.norm(E_w - S_w)
    a = (O - S_w) @ u * 1e3
    L = float(np.linalg.norm(E_w - S_w)) * 1e3
    assert abs(a[0]) <= tol_mm and abs(a[-1] - L) <= tol_mm, (float(a[0]), float(a[-1]), L, r.passes[-1].get("direction"))
    assert all(float(np.asarray(st.u) @ u) > 0.0 for st in r.weld_plan.stations)
    for p in r.weld_plan.passes:
        a0 = float((p.aim_base_m[0] - S_w) @ u) * 1e3
        assert abs(a0 - (0.0 if p.direction == "S->E" else L)) <= tol_mm, (p.k, p.direction, a0)


def pass_hit_range_mm(world, calib, move: dict) -> tuple:
    Ts = [make_transform(np.asarray(move[f"{k}_R"]), np.asarray(move[f"{k}_m"])) @ calib.T_flange_opt for k in ("from", "to")]
    h = sorted(float(laser_plane_line_param(T, world.S, world.u)) * 1e3 for T in Ts)
    return h[0], h[1]


def assert_weld_uphill(r, up=(0.0, 0.0, 1.0)) -> None:
    up = np.asarray(up, dtype=float)
    assert r.weld_plan is not None and len(r.weld_plan.passes) == 4, (r.status, r.reason)
    for p in r.weld_plan.passes:
        h = p.aim_base_m @ up
        assert h[-1] - h[0] > 0.3 and np.all(np.diff(h) > -1e-4), (p.k, p.direction)
        assert float(np.median(p.R_base_tcp[:, :, 2] @ up)) > 0.1, p.k


def true_yaw_deg(world, T_base_flange, calib=None) -> float:
    """레이저 선 (X_opt 의 진짜 판면 투영) 과 진짜 심에 수직인 축 w = n×u 사이 각 [deg] (|·| ≤ 90; 부호 = n 둘레)."""
    from scan_core.seam_track import yaw_to_seam_rad
    cal = calib if calib is not None else world.calib
    R = (np.asarray(T_base_flange, dtype=float) @ cal.T_flange_opt)[:3, :3]
    return math.degrees(yaw_to_seam_rad(R, world.n, world.u))


def dwell_footprints(r, world, calib, fov_fn) -> list:
    """실행의 정지 측정 (국소화) 마다 레이저 선 발자국의 진짜 심 좌표 범위 [mm]: 광축 ∩ 진짜 판 ± X_opt 판면 투영 · FOV(그 거리)/2 →
    [{name, s_min, s_max, range_m}] (진짜 S 끝 0, E 끝 L). 자세는 국소화 기록의 T_base_opt (캘리브 적용) 을 진짜 캘리브로 바꿔 쓴다."""
    from scan_core.frames import inv_transform
    from scan_core.seam_track import laser_footprint_s
    out = []
    for d in r.localizations:
        T_bo = np.asarray(d.get("T_base_opt"), dtype=float)
        if T_bo.shape != (4, 4) or not np.all(np.isfinite(T_bo)):
            continue
        T_bf = T_bo @ inv_transform(np.asarray(calib.T_flange_opt, dtype=float))
        T_true = T_bf @ world.calib.T_flange_opt
        z = d.get("diag", {}).get("raw_z_axis_mm") if isinstance(d.get("diag"), dict) else None
        rng = float(z) * 1e-3 if z is not None and np.isfinite(float(z)) else float(np.linalg.norm(T_true[:3, 3] - world.S))
        s0, s1 = laser_footprint_s(T_true, world.S, world.u, world.n, float(fov_fn(rng)))
        out.append({"name": str(d.get("name")), "s_min": s0 * 1e3, "s_max": s1 * 1e3, "range_m": rng})
    return out


def assert_dwell_lines_inside(r, world, calib, fov_fn, e_hit_m: float, skip=("S",)) -> list:
    """I1 (P2): S 추적 끝 드웰 (S_trk) 부터 모든 드웰의 레이저 선 발자국이 진짜 시편 구간 [e_hit, L − e_hit] 안 (첫 드웰 S 등 skip 은 제외 —
    아직 추정이 없다). 반환 = 발자국 목록."""
    fp = dwell_footprints(r, world, calib, fov_fn)
    names = [f["name"] for f in fp]
    i0 = next((i for i, nm in enumerate(names) if nm.startswith("S_trk")), None)
    assert i0 is not None, names
    L_mm = world.L * 1e3
    e_mm = float(e_hit_m) * 1e3
    bad = [f for f in fp[i0:] if f["name"] not in skip and not (f["s_min"] >= e_mm - 1e-6 and f["s_max"] <= L_mm - e_mm + 1e-6)]
    assert not bad, [(f["name"], round(f["s_min"], 1), round(f["s_max"], 1)) for f in bad]
    return fp


def exact_diff(a, b, path: str = "", out: list = None, limit: int = 20) -> list:
    """엄격 비교 (패스 뒤 처리 결과가 '같다' = 바이트까지): 배열은 dtype·shape·바이트, float 는 비트 (NaN·−0 포함), dict 는 키 순서까지,
    dataclass·객체는 속성. 다른 곳 목록 (최대 limit) — 빈 목록 = 같다."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or a.dtype != b.dtype or a.shape != b.shape:
            out.append(f"{path}: {getattr(a, 'dtype', type(a))}{getattr(a, 'shape', '')} vs {getattr(b, 'dtype', type(b))}"
                       f"{getattr(b, 'shape', '')}")
        elif a.dtype == object:
            for i, (x, y) in enumerate(zip(a.ravel(), b.ravel())):
                exact_diff(x, y, f"{path}[{i}]", out, limit)
        elif a.tobytes() != b.tobytes():
            out.append(f"{path}: array bytes differ")
        return out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, (float, np.floating)):
        if np.float64(a).tobytes() != np.float64(b).tobytes():
            out.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            out.append(f"{path}: keys {list(a.keys())} vs {list(b.keys())}")
        else:
            for k in a:
                exact_diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                exact_diff(x, y, f"{path}[{i}]", out, limit)
    elif dataclasses.is_dataclass(a) or hasattr(a, "__dict__"):
        exact_diff(vars(a), vars(b), path, out, limit)
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def capture_post_scan(monkeypatch) -> list:
    """패스 뒤 처리 (scan_l0.post_scan.run_post_scan — 서버 쪽, 이 시험 프로세스) 의 (입력, 출력) 을 모은다 — 받은 그대로의 복사본 (서버는 그 뒤
    맞춤을 제자리에서 고친다: 3G 아래 끝 이어받기 등)."""
    import copy
    from scan_l0 import post_scan as PS
    caps: list = []
    orig = PS.run_post_scan

    def wrap(p, **kw):
        out = orig(p, **kw)
        caps.append((copy.deepcopy(p), copy.deepcopy(out)))
        return out
    monkeypatch.setattr(PS, "run_post_scan", wrap)
    return caps


def assert_stream_identical(caps, tmp_dir) -> int:
    """패스마다: 작업 프로세스가 패스 중 판 성장 스트림을 써서 낸 fit 출력 (특징·맞춤·끝·중앙값·잔차) == 같은 입력을 스트림 없이 이 프로세스에서
    다시 계산한 것 (바이트까지). 반환 = 스트림 결과를 쓴 행 수의 합 (호출자가 > 0 을 확인 — 스트림이 실제로 돌았다)."""
    from scan_l0 import post_scan as PS
    n_seeded = 0
    n_fit = 0
    for i, (p, out) in enumerate(caps):
        if out.get("fit") is None:
            continue
        n_fit += 1
        ref = PS.compute_pass(dict(p, out_dir=str(os.path.join(str(tmp_dir), f"stream_recheck_{i}")), defer_save=True,
                                   in_worker=False, stream_token=None))
        PS._PENDING.clear()
        d = exact_diff(out["fit"], ref["fit"])
        assert not d, (i, d)
        n_seeded += int(out["timing"].get("stream_seeded", 0) or 0)
    assert n_fit > 0, "no pass post-scan captured"
    return n_seeded


def true_face_normals(world) -> dict:
    """세계 (GrooveWorld, 일면 개선) 의 진짜 면 법선 (재료 밖) — 형상 종류별: 판 상면·바닥 (뒷댐재 상면) = n, 베벨면 = cos β·(베벨 판 →
    그루브 가운데) + sin β·n, 직각 벽 = 직각 판 → 그루브 가운데. 'bevel_plate' = 그루브 가운데에서 베벨 판 쪽 단위 벡터 (쪽 이름 확인용)."""
    sp = world.spec
    n = unit(world.n)
    bevel_plate = -world.lateral if sp.resolved_bevel_side() == "left" else world.lateral       # +x_spec = 판 +y = lateral
    b = math.radians(sp.bevel_angle_deg)
    return {"plate_top": n, "gap_bottom": n, "bevel": math.cos(b) * (-bevel_plate) + math.sin(b) * n, "wall": bevel_plate,
            "bevel_plate": bevel_plate}


def cross_section_errors(world, out_dir, k: float) -> tuple:
    """cross_sections.json (실행 출력) vs 세계의 진짜 면 법선 → (cs, [(단면 s, 면 이름, 각 오차 deg, 허용 deg)]).
    허용 (유도): k × max(그 단면 그 면의 프로파일 법선 퍼짐 σ (잰 값), 센서 양자화가 그 면에 주는 각 σ) — 양자화 각 σ = dX/√12 ·
    √(12/n) / L (점 n 개가 길이 L 에 고르게 놓인 직선 기울기의 표준편차; n = 그 면의 프로파일당 점 수 (잰 값), L = 설계 면 길이: 판 국소
    구간 = 개구, 베벨 = (t − rf)/cos β, 바닥 = 갭, 벽 = t − rf), dX/√12 = scan_core.measure.quant_sigma_mm (유효 창 가운데)."""
    import json as _json
    from scan_core.measure import quant_sigma_mm
    with open(os.path.join(str(out_dir), "cross_sections.json"), "r", encoding="utf-8") as f:
        cs = _json.load(f)
    truth = true_face_normals(world)
    sp = world.spec
    b = math.radians(sp.bevel_angle_deg)
    L_face = {"plate_top": sp.top_opening_mm, "bevel": (sp.thickness_mm - sp.root_face_mm) / math.cos(b), "gap_bottom": sp.root_gap_mm,
              "wall": sp.thickness_mm - sp.root_face_mm}
    q = quant_sigma_mm(None)
    rows = []
    for st in cs["stations"]:
        for nm, fc in st["faces"].items():
            n_pp = max(2.0, float(fc["n_points"]) / max(1, int(fc["n_profiles"])))
            sq = math.degrees(q * math.sqrt(12.0 / n_pp) / L_face[fc["type"]])
            tol = float(k) * max(float(fc["sigma_deg"] or 0.0), sq)
            rows.append((float(st["s_m"]), nm, angle_deg(fc["normal"], truth[fc["type"]]), tol))
    return cs, rows


__all__ = ["SIM_CONFIG", "CALIB", "FPS", "N_POINTS", "LENGTH_TOL_MM", "SENSOR_INIT", "SENSOR_REALTIME", "sim_settings",
           "calib_for_sensor", "rot_axis", "rolled_normal", "perturbed_calib", "sim_cell", "rough_points", "write_rough_job", "RUN_KW",
           "run_autoteach", "executed", "angle_deg", "rot_between_deg", "perp_to_true_root", "plan_origins", "pass_hit_range_mm",
           "assert_weld_uphill", "LAB_S_TRUE", "LAB_E_TRUE", "FOLDED_Q", "exact_diff", "capture_post_scan", "assert_stream_identical",
           "true_yaw_deg", "dwell_footprints", "assert_dwell_lines_inside", "true_face_normals", "cross_section_errors"]
