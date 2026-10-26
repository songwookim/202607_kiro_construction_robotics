"""단면 상세 법선 (scan_core.cross_sections, 사용자 2026-10-08): 합성 그루브 (일면 개선 좌/우·V, 1G 판 위·3G 세운 판, 입사 기울기·롤·요)
에서 프로파일 면 직선 → base 면 법선이 진짜 면 법선과 유도 허용 안, 심 좌표 쪽 이름, 단면 집계·파일 형식, 처리된 패스 점구름 간격.

허용 (유도, scan_l0/tests/sim_harness.cross_section_errors 와 같은 규칙): k × 센서 양자화가 그 면에 주는 각 σ = dX/√12 · √(12/n) / L
(점 n 개가 길이 L 에 고르게 놓인 직선 기울기의 표준편차; n = 그 면의 점 수 (잰 값), L = 설계 면 길이), 단면은 잰 퍼짐 σ 와의 큰 값."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scan_core import cross_sections as CS
from scan_core import seam_features as SF
from scan_core.frames import make_transform, rot_x, rot_y, rot_z, unit
from scan_core.localize import polyline_from_scan_features
from scan_core.measure import K_DEFAULT, quant_sigma_mm
from scan_core.path_gen import plate_frame
from scan_core.specimen import GrooveSpec
from scan_core.synth import GrooveScene, SynthConfig, simulate_profile

STANDOFF_M = 0.34
GEOM = {"1G": dict(S=(0.6, -0.2, 0.05), u=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        "3G": dict(S=(0.5, 0.7, 0.3), u=(0.0, 0.0, 1.0), n=(0.0, -1.0, 0.0))}


def _world(position: str, spec: GrooveSpec):
    g = GEOM[position]
    S, u, n = np.asarray(g["S"], float), unit(g["u"]), unit(g["n"])
    return S, u, n, np.cross(n, u), plate_frame(S, u, n), GrooveScene.from_spec(spec, centered=False)


def _pose(S, u, n, s: float, *, pitch: float = 0.0, roll: float = 0.0, yaw: float = 0.0, lateral_m: float = 0.0) -> np.ndarray:
    """광학 자세: R = [n×u, u, −n]·Rz(yaw)·Rx(pitch)·Ry(roll) (pitch = 레이저 선 축 둘레 = 입사 기울기), 광학 Z 가 심 점을 향한다."""
    lat = np.cross(n, u)
    R = np.column_stack([lat, u, -n]) @ rot_z(math.radians(yaw)) @ rot_x(math.radians(pitch)) @ rot_y(math.radians(roll))
    target = S + u * s + lat * lateral_m
    return make_transform(R, target - STANDOFF_M * R[:, 2])


def _true_normals(spec: GrooveSpec, n, lat) -> dict:
    """진짜 면 법선 (재료 밖), 심 좌표 이름이 아니라 판 프레임 쪽 (x_spec − = 'left' 판 = −lat)."""
    b = math.radians(spec.bevel_angle_deg)
    out = {"plate_top": n, "gap_bottom": n}
    bev = ("left", "right") if spec.kind == "single_v" else ((spec.resolved_bevel_side(),) if spec.kind == "single_bevel" else ())
    for side in bev:
        plate = -lat if side == "left" else lat                     # 베벨 판이 그루브 가운데에서 어느 쪽인가
        out[f"bevel@{side}"] = (math.cos(b) * (-plate) + math.sin(b) * n, plate)
    return out


def _face_len_mm(spec: GrooveSpec) -> dict:
    b = math.radians(spec.bevel_angle_deg)
    return {"plate_top": spec.top_opening_mm, "bevel": (spec.thickness_mm - spec.root_face_mm) / max(math.cos(b), 1e-9),
            "gap_bottom": spec.root_gap_mm, "wall": spec.thickness_mm - spec.root_face_mm}


def _quant_deg(n_pts: float, L_mm: float) -> float:
    return math.degrees(quant_sigma_mm(None) * math.sqrt(12.0 / max(2.0, n_pts)) / L_mm)


def _pass(position: str, spec: GrooveSpec, s_values, *, seed: int = 0, **pose_kw):
    """합성 패스 (자세 고정, 심을 따라 s) → (result 이름공간, sf, P_root, 세계)."""
    S, u, n, lat, T_plate, scene = _world(position, spec)
    s_values = np.asarray(s_values, dtype=float)
    T = np.stack([_pose(S, u, n, s, **pose_kw) for s in s_values])
    X, Z = [], []
    for i, Ti in enumerate(T):
        x, z = simulate_profile(Ti, scene, T_plate, SynthConfig(n_points=1408, seed=seed + i, sigma_z_mm=0.02, dropout_prob=0.01))
        X.append(x)
        Z.append(z)
    X, Z = np.array(X), np.array(Z)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(X) & np.isfinite(Z) & (Z > 0)
    res = SimpleNamespace(x_mm=X, z_mm=Z, valid=valid, T_base_opt=T, s_m=s_values, in_cruise=np.ones(s_values.size, bool))
    sf = SF.extract_seam_features_along_scan(res, spec, keep_features=True, plate_normal_base=n)
    P_root, _gf = polyline_from_scan_features(res, sf, centre="root")
    return res, sf, P_root, (S, u, n, lat)


def _feats(sf, P_root, n) -> dict:
    return {"root_polyline_base_m": P_root, "ok": np.asarray(sf.ok, bool), "face_point_base": sf.face_point_base,
            "face_dir_base": sf.face_dir_base, "face_fit": sf.face_fit, "opt_origin_base": sf.opt_origin_base, "n_base": np.asarray(n)}


# --------------------------------------------------------------------------------------
# 프로파일 면 → base 법선
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("position", ["1G", "3G"])
@pytest.mark.parametrize("pose", [dict(), dict(pitch=20.0), dict(pitch=35.0, roll=5.0), dict(pitch=-10.0, roll=-6.0, yaw=4.0)],
                         ids=["normal", "tilt20", "tilt35_roll5", "tiltm10_roll_yaw"])
@pytest.mark.parametrize("kind_side", [("single_bevel", "left"), ("single_bevel", "right"), ("single_v", "left")])
def test_profile_face_normals_match_the_groove_geometry(position, pose, kind_side):
    """한 프로파일: 판 상면 둘·베벨면 (쪽)·바닥의 법선 = 진짜 면 법선 (k × 양자화 각 σ 안), 쪽 이름 = S→E 로 볼 때 왼쪽 +w."""
    spec = GrooveSpec(kind=kind_side[0], bevel_side=kind_side[1])
    res, sf, P_root, (S, u, n, lat) = _pass(position, spec, [0.15], lateral_m=0.003, **pose)
    assert bool(sf.ok[0]), sf.features[0].report()
    f = sf.features[0]
    assert f.faces is not None and f.faces.shape == (len(SF.FACE_KINDS), len(SF.FACE_COLS))
    N = CS.profile_face_normals(sf.face_dir_base, sf.face_point_base, sf.opt_origin_base, u)[0]
    w = np.cross(n, u)
    sg = CS.seam_sides(sf.face_point_base, P_root, w)[0]
    assert sg in (1.0, -1.0)
    truth = _true_normals(spec, n, lat)
    L = _face_len_mm(spec)
    seen = set()
    for i, kind in enumerate(SF.FACE_KINDS):
        if not np.all(np.isfinite(N[i])):
            continue
        name = CS.FACE_NAMES[CS._seam_face_index(kind, sg)]
        typ = CS.FACE_TYPE[name]
        n_pts = float(sf.face_fit[0, i, 1])
        tol = K_DEFAULT * _quant_deg(n_pts, L[typ])
        p = sf.face_point_base[0, i]
        if typ == "bevel":
            # 쪽: 이 면 점이 그루브 가운데의 어느 쪽 (+w = left)
            side = "left" if float((p - P_root[0]) @ w) > 0 else "right"
            assert name == f"bevel_{side}", (name, side)
            plate_side = "left" if float(-lat @ w) > 0 else "right"     # 판 프레임 'left' (−lat) 판이 심 좌표 어느 쪽인가
            key = f"bevel@{'left' if side == plate_side else 'right'}"
            assert key in truth, (key, list(truth))
            exp = truth[key][0]
        elif typ == "wall":
            continue                                                 # 직각 벽은 이 자세들에서 몇 점만 (그림자) — 아래 별도 시험
        else:
            exp = truth[typ]
        err = math.degrees(math.acos(min(1.0, float(N[i] @ exp))))
        assert err <= tol, (position, pose, kind_side, name, err, tol, n_pts)
        seen.add(typ if typ != "bevel" else name)
    n_bev = {"single_bevel": 1, "single_v": 2}[spec.kind]
    assert {"plate_top", "gap_bottom"} <= seen and sum(1 for t in seen if t.startswith("bevel_")) == n_bev, seen


def test_square_wall_seen_with_a_rolled_sensor_and_floor_line_drops_bevel_foot_points():
    """직각 벽: 레이저 선을 그루브 쪽으로 기울이면 (롤) 벽에 점이 닿아 벽 법선 = 직각 판 → 그루브 가운데 수평. 바닥 직선은 대역 안에 섞인 베벨
    발치 점을 k·σ 로 빼서 수평 (빼지 않으면 5 mm 바닥이 ~2° 기운다)."""
    spec = GrooveSpec(kind="single_bevel", bevel_side="left")
    res, sf, P_root, (S, u, n, lat) = _pass("1G", spec, [0.15], roll=12.0, lateral_m=0.0)   # 광선이 +lat 쪽으로: −lat 을 보는 벽
    i_wall = [i for i, k in enumerate(SF.FACE_KINDS) if k.startswith("wall_") and np.all(np.isfinite(sf.face_dir_base[0, i]))]
    assert i_wall, [f for f in sf.features[0].flags]
    N = CS.profile_face_normals(sf.face_dir_base, sf.face_point_base, sf.opt_origin_base, u)[0]
    exp = -lat                                                       # 직각 판은 'right' (+lat) — 벽 법선은 −lat 쪽 (그루브 가운데)
    i = i_wall[0]
    err = math.degrees(math.acos(min(1.0, float(N[i] @ exp))))
    tol = K_DEFAULT * _quant_deg(float(sf.face_fit[0, i, 1]), _face_len_mm(spec)["wall"])
    assert err <= tol and int(sf.face_fit[0, i, 1]) >= 4, (err, tol, sf.face_fit[0, i])
    # 바닥 직선의 k·σ 정련 (직접): 5 mm 수평 바닥 + 끝의 베벨 발치 두 점 (0.57, 0.21 mm 위)
    x = np.linspace(-6.8, -1.6, 25)
    h = np.full(x.size, -350.0) + np.random.default_rng(0).normal(0.0, 0.02, x.size)
    h[0] += 0.57
    h[1] += 0.21
    rows = SF._face_rows(x, h, {"floor": np.arange(x.size)})
    j = SF.FACE_KINDS.index("floor")
    ang = math.degrees(math.atan2(-rows[j, 3], rows[j, 2]))
    raw = SF.fit_line_pca(x, h).angle_deg
    assert abs(raw) > 1.0 and abs(ang) < 0.3 and rows[j, 5] <= 24, (raw, ang, rows[j])


# --------------------------------------------------------------------------------------
# 단면 집계 · 파일
# --------------------------------------------------------------------------------------
def _face_s_in_station(cs, S_pt, u) -> None:
    """모든 단면의 모든 면: 면 위 점의 심 좌표가 그 단면 구간 안 (면은 자기 점의 s 로 단면에 든다 — 프로파일 루트 점의 s 가 아니다)."""
    for st in cs["stations"]:
        a, b = st["s_range_m"]
        for nm, fc in st["faces"].items():
            s_ = float((np.asarray(fc["point_m"]) - S_pt) @ u)
            assert a <= s_ <= b, (st["s_range_m"], nm, s_)


@pytest.mark.parametrize("position,pose", [("1G", dict(pitch=-5.0)), ("3G", dict(pitch=35.0, roll=3.0))], ids=["1G", "3G_tilt35"])
def test_station_cross_sections_aggregate_and_round_trip(position, pose, tmp_path):
    spec = GrooveSpec(kind="single_bevel", bevel_side="right")
    # 71 프로파일, 1 mm 간격: 단면 구간 (판 위 s 0.100–0.1405, 10 mm 창 4 개) 바깥까지 — 기운 레이저 평면 (35°) 은 한 프로파일에서 판 상면을
    # 루트에서 depth·tan(입사) ≈ 7 mm 떨어져 본다; 그 면도 모든 단면에 증거가 들도록
    s_vals = np.arange(0.085, 0.1555, 0.001)
    res, sf, P_root, (S, u, n, lat) = _pass(position, spec, s_vals, **pose)
    feats = _feats(sf, P_root, n)
    root = S - n * spec.thickness_mm * 1e-3
    # 납품 양끝: 판 위 s 0.100 부터 40.5 mm (잰 루트 점은 진짜 위치 — 기운 레이저 평면은 깊은 루트를 심 방향으로 옮겨 본다)
    S_pt = root + u * 0.100
    E_pt = S_pt + u * 0.0405
    cs = CS.station_cross_sections(feats, S_pt, E_pt, window_m=0.010, need=5, spec=spec, base_frame="link0", position=position)
    assert len(cs["stations"]) == 4 and [round(st["s_m"], 4) for st in cs["stations"]] == [0.005, 0.015, 0.025, 0.035]
    w = np.asarray(cs["seam"]["w"])
    np.testing.assert_allclose(w, np.cross(n, u), atol=1e-9)
    bev = "left" if float(lat @ w) > 0 else "right"                  # 'right' 베벨 판 = +lat
    assert cs["bevel_side"] == bev and cs["faces_expected"] == ["plate_top_left", "plate_top_right", f"bevel_{bev}",
                                                                f"wall_{'right' if bev == 'left' else 'left'}", "gap_bottom"]
    b = math.radians(30.0)
    truth = {"plate_top": n, "gap_bottom": n, "bevel": math.cos(b) * (-lat) + math.sin(b) * n}
    L = _face_len_mm(spec)
    _face_s_in_station(cs, S_pt, u)
    for st in cs["stations"]:
        assert {"plate_top_left", "plate_top_right", f"bevel_{bev}", "gap_bottom"} <= set(st["faces"]), list(st["faces"])
        assert np.linalg.norm(np.asarray(st["root_point_m"]) - (S_pt + u * st["s_m"])) < 1.5e-3
        for nm, fc in st["faces"].items():
            if fc["type"] == "wall":
                continue
            assert fc["n_profiles"] >= 5 and abs(np.linalg.norm(fc["normal"]) - 1.0) < 1e-12
            tol = K_DEFAULT * max(fc["sigma_deg"], _quant_deg(fc["n_points"] / fc["n_profiles"], L[fc["type"]]))
            err = math.degrees(math.acos(min(1.0, float(np.dot(fc["normal"], truth[fc["type"]])))))
            assert err <= tol, (position, st["s_m"], nm, err, tol)
            # 면 위 점: 판 상면은 판 상면 (n 방향 0), 바닥은 두께 아래
            d_n = float((np.asarray(fc["point_m"]) - S) @ n) * 1e3
            if fc["type"] == "plate_top":
                assert abs(d_n) < 0.1
            elif fc["type"] == "gap_bottom":
                assert abs(d_n + spec.thickness_mm) < 0.1
    # 판 법선을 재료 쪽 (반대 부호) 으로 줘도 같은 결과 (센서 쪽으로 돌린다 — 왼쪽/오른쪽이 바뀌지 않는다)
    cs2 = CS.station_cross_sections({**feats, "n_base": -n}, S_pt, E_pt, window_m=0.010, need=5, spec=spec)
    assert cs2["bevel_side"] == bev and np.allclose(cs2["seam"]["w"], w)
    # 증거가 모자라면 그 면/단면은 없다
    cs3 = CS.station_cross_sections(feats, S_pt, E_pt, window_m=0.010, need=12, spec=spec)
    assert cs3["stations"] == []
    # 납품 양끝 밖의 프로파일이 없으면 (루트가 끝 밖 = 시편 밖): 기운 레이저가 끝 단면에서 루트와 떨어져 보는 면은 다른 단면의 점을 빌리지
    # 않고 빠진다 (notes) — 남은 면은 모두 제 단면 안
    s_root = (P_root - S_pt) @ u
    inside = np.all(np.isfinite(P_root), axis=1) & (s_root >= 0.0) & (s_root <= 0.0405)
    cs4 = CS.station_cross_sections({**feats, "ok": feats["ok"] & inside}, S_pt, E_pt, window_m=0.010, need=5, spec=spec)
    _face_s_in_station(cs4, S_pt, u)
    shift = {nm: float(np.nanmedian(((sf.face_point_base[:, i] - S_pt) @ u) - s_root)) for i, nm in enumerate(SF.FACE_KINDS)
             if nm.startswith("plate_")}
    for nm in ("plate_top_left", "plate_top_right"):
        lost = [st["s_m"] for st in cs4["stations"] if nm not in st["faces"]]
        # 판 상면이 루트에서 한 창의 절반 넘게 떨어져 보이면 (3G 35°: ~7 mm) 한 끝 단면에 증거가 모자란다; 1G (−5°: ~1 mm) 는 모두 있다
        if max(abs(v) for v in shift.values()) > 0.5 * 0.010:
            assert len(lost) == 1 and any(nt.startswith(f"{nm}: measured at 3/4") for nt in cs4["notes"]), (nm, lost, cs4["notes"])
        else:
            assert lost == [], (nm, lost, shift)
    # 파일: json (요약·단면) + npz (단면 × 면 배열, 없는 면 NaN)
    files = CS.save_cross_sections(cs, tmp_path)
    assert [p.rsplit("/", 1)[-1] for p in files] == ["cross_sections.json", "cross_sections.npz"]
    d = json.load(open(files[0], encoding="utf-8"))
    assert d["schema_version"] == CS.SCHEMA_VERSION and d["base_frame"] == "link0" and d["position"] == position
    assert d["face_names"] == list(CS.FACE_NAMES) and len(d["stations"]) == 4
    z = CS.load_cross_sections(files[0])
    K, F = 4, len(CS.FACE_NAMES)
    assert z["normal"].shape == (K, F, 3) and z["point_m"].shape == (K, F, 3) and z["n_profiles"].shape == (K, F)
    assert z["s_range_m"].shape == (K, 2) and z["root_point_m"].shape == (K, 3) and str(z["base_frame"]) == "link0"
    jb = CS.FACE_NAMES.index(f"bevel_{bev}")
    np.testing.assert_allclose(z["normal"][:, jb], [st["faces"][f"bevel_{bev}"]["normal"] for st in cs["stations"]])
    jo = CS.FACE_NAMES.index(f"bevel_{'right' if bev == 'left' else 'left'}")
    assert np.all(np.isnan(z["normal"][:, jo])) and np.all(z["n_profiles"][:, jo] == 0)


def test_faces_are_stationed_by_their_own_seam_position_on_a_twisting_plate():
    """판이 심을 따라 비틀리면 (면 법선이 u 둘레로 s 에 비례해 돈다) 단면의 면 법선 = 그 단면 구간 가운데의 법선. 기운 레이저 평면은 한
    프로파일에서 판 상면을 루트에서 δ 떨어져 본다 — 루트 점의 s 로 묶으면 판 상면 법선이 rate·δ 만큼 옆 단면 것이 된다 (δ 7 mm, 0.5°/mm
    → 3.5°). 허용 (유도) = rate × 프로파일 간격 (구간 안 고른 점의 중앙값은 가운데에서 한 간격 안)."""
    rate = math.radians(0.5) * 1e3                                 # rad/m
    dsp, delta, L, win = 0.0005, 0.007, 0.040, 0.010
    S = np.zeros(3)
    u, n = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    w = np.cross(n, u)                                             # 심 왼쪽 (+w)
    s_root = np.arange(-win, L + win, dsp) + 0.25 * dsp            # 구간 경계에 점이 놓이지 않게 (경계 반올림과 무관한 증거 수)
    M, F = s_root.size, len(SF.FACE_KINDS)
    FP, FD, FF = np.full((M, F, 3), np.nan), np.full((M, F, 3), np.nan), np.full((M, F, 2), np.nan)

    def nrm(s_):
        return math.cos(rate * s_) * n + math.sin(rate * s_) * w
    # 센서 면 → 그 프로파일에서 본 심 위치·옆 위치: 판 상면은 루트에서 δ 앞, 바닥은 루트 그대로 (센서 오른쪽 판 = +w = 심 왼쪽)
    where = {"plate_left": (delta, -0.008 * w), "plate_right": (delta, 0.008 * w), "floor": (0.0, -0.010 * n)}
    for kind, (ds, off) in where.items():
        j = SF.FACE_KINDS.index(kind)
        for m_, s_ in enumerate(s_root + ds):
            FP[m_, j] = S + u * s_ + off
            FD[m_, j] = np.cross(nrm(s_), u)                       # 레이저 평면 안 면 방향 (⊥ u, ⊥ 법선)
            FF[m_, j] = (0.01, 20.0)
    feats = {"root_polyline_base_m": S + np.outer(s_root, u) - 0.010 * n, "ok": np.ones(M, bool), "face_point_base": FP,
             "face_dir_base": FD, "face_fit": FF, "opt_origin_base": S + np.outer(s_root, u) + 0.3 * n, "n_base": n}
    cs = CS.station_cross_sections(feats, S, S + u * L, window_m=win, need=5)
    assert len(cs["stations"]) == 4
    _face_s_in_station(cs, S, u)
    tol = math.degrees(rate * dsp)
    for st in cs["stations"]:
        for nm in ("plate_top_left", "plate_top_right", "gap_bottom"):
            fc = st["faces"][nm]
            err = math.degrees(math.acos(min(1.0, float(np.dot(fc["normal"], nrm(st["s_m"]))))))
            assert err <= tol and fc["n_profiles"] == round(win / dsp), (st["s_m"], nm, err, tol, fc["n_profiles"])


@pytest.mark.parametrize("position", ["1G", "3G"])
def test_stations_are_in_the_taught_S_to_E_frame_whatever_the_scan_direction(position):
    """단면은 사용자가 교시한 S→E 좌표 (용접 계획과 같은 계약: 축 = unit(E − S) (판 법선에 직교), s = S 에서 잰 거리, 첫 단면 = S 끝,
    좌/우 = S→E 로 볼 때) — 패스 진행 방향과 무관하다. 같은 시편을 S→E 로 훑은 패스와 E→S 로 (센서를 심 둘레로 뒤집어: 센서 x 가 반대)
    훑은 패스가 같은 단면 (s, 면 이름, 베벨 쪽) 을 내고, 납품 양끝을 바꿔 주면 (E 에서 S 로 교시) 단면은 그 E 끝에서 시작하고 좌/우가
    바뀐다. 법선은 모두 진짜 면 법선과 유도 허용 안 (test_station_cross_sections_aggregate_and_round_trip 와 같은 규칙)."""
    spec = GrooveSpec(kind="single_bevel", bevel_side="right")
    s_vals = np.arange(0.095, 0.1455, 0.001)
    fwd = _pass(position, spec, s_vals)
    rev = _pass(position, spec, s_vals[::-1], yaw=180.0)           # E→S: 진행 방향 −u, 센서 x = −(n × u)
    S, u, n, lat = fwd[3]
    root = S - n * spec.thickness_mm * 1e-3
    S_pt, E_pt = root + u * 0.100, root + u * 0.1405
    b = math.radians(spec.bevel_angle_deg)
    truth = {"plate_top": n, "gap_bottom": n, "bevel": math.cos(b) * (-lat) + math.sin(b) * n}   # 'right' 베벨 판 = +lat
    L = _face_len_mm(spec)

    def run(res_sf, A, B):
        _res, sf, P_root, _g = res_sf
        cs = CS.station_cross_sections(_feats(sf, P_root, n), A, B, window_m=0.010, need=5, spec=spec)
        uu = (B - A) / np.linalg.norm(B - A)
        np.testing.assert_allclose(cs["seam"]["u"], uu, atol=1e-12)
        np.testing.assert_allclose(cs["seam"]["S_m"], A, atol=1e-12)
        _face_s_in_station(cs, A, uu)
        for st in cs["stations"]:
            # 단면 기준점 = 교시 S 끝에서 S→E 로 s_m (첫 단면이 S 쪽)
            assert np.linalg.norm(np.asarray(st["root_point_m"]) - (A + uu * st["s_m"])) < 1.5e-3, st["s_m"]
            for nm, fc in st["faces"].items():
                if fc["type"] == "wall":
                    continue
                tol = K_DEFAULT * max(fc["sigma_deg"], _quant_deg(fc["n_points"] / fc["n_profiles"], L[fc["type"]]))
                err = math.degrees(math.acos(min(1.0, float(np.dot(fc["normal"], truth[fc["type"]])))))
                assert err <= tol, (position, st["s_m"], nm, err, tol)
        return cs

    a, r = run(fwd, S_pt, E_pt), run(rev, S_pt, E_pt)
    bev = "left" if float(lat @ np.asarray(a["seam"]["w"])) > 0 else "right"
    assert [st["s_m"] for st in a["stations"]] == [st["s_m"] for st in r["stations"]] == pytest.approx([0.005, 0.015, 0.025, 0.035])
    assert a["bevel_side"] == r["bevel_side"] == bev
    for sa, sr in zip(a["stations"], r["stations"]):
        assert {"plate_top_left", "plate_top_right", f"bevel_{bev}", "gap_bottom"} <= set(sa["faces"]) == set(sr["faces"]), \
            (sa["s_m"], list(sa["faces"]), list(sr["faces"]))
    # 교시를 반대로 (E 에서 S): 축 −u, 첫 단면 = 그 S (원래 E) 쪽, 좌/우 이름이 바뀐다 — 어느 패스든
    for res_sf in (fwd, rev):
        x = run(res_sf, E_pt, S_pt)
        assert x["bevel_side"] == ("right" if bev == "left" else "left") and len(x["stations"]) == 4
        assert float((np.asarray(x["stations"][0]["root_point_m"]) - E_pt) @ (S_pt - E_pt)) > 0.0


def test_expected_faces_follow_the_groove_family():
    assert CS.expected_faces(GrooveSpec(kind="single_v"), "both")[0] == ["plate_top_left", "plate_top_right", "bevel_left",
                                                                          "bevel_right", "gap_bottom"]
    assert CS.expected_faces(GrooveSpec(kind="square"), "none")[0] == ["plate_top_left", "plate_top_right", "wall_left",
                                                                        "wall_right", "gap_bottom"]
    names, missing = CS.expected_faces(GrooveSpec(kind="single_bevel", root_face_mm=2.0, backing=False), "left")
    assert names == ["plate_top_left", "plate_top_right", "bevel_left", "wall_right"] and [m["name"] for m in missing] == ["root_face_left"]


def test_decimated_cloud_keeps_measured_points_at_the_requested_spacing():
    spec = GrooveSpec(kind="single_bevel", bevel_side="left")
    s_vals = 0.10 + np.arange(50) * 0.0003                          # 0.3 mm 간격 (실기 30 mm/s · 100 fps)
    res, sf, P_root, (S, u, n, lat) = _pass("1G", spec, s_vals)
    P = CS.decimated_cloud(res, 1e-3)
    assert P.dtype == np.float32 and P.ndim == 2 and P.shape[1] == 3
    n_rows = math.ceil(s_vals.size / round(1e-3 / 3e-4))
    dx = float(np.nanmedian(np.abs(np.diff(res.x_mm[s_vals.size // 2])))) * 1e-3
    n_cols = math.ceil(res.x_mm.shape[1] / max(1, round(1e-3 / dx)))
    assert 0.8 * n_rows * n_cols < P.shape[0] <= n_rows * n_cols, (P.shape, n_rows, n_cols)
    d_n = (P.astype(float) - S) @ n
    assert np.all(d_n > -(spec.thickness_mm * 1e-3 + 1e-4)) and np.all(d_n < 1e-4)
    along = np.unique(np.round((P.astype(float) - S) @ u, 5))
    assert np.allclose(np.diff(along), 0.0009, atol=1e-5)            # round(1.0/0.3) = 3 프로파일마다
    # 열 간격은 남길 프로파일 전부에서: 가운데 프레임 하나가 비어도 (드롭아웃·유령 프레임) 솎음이 꺼지지 않는다 (예전: 한 프레임 → ~9 배)
    X, Z, V = res.x_mm.copy(), res.z_mm.copy(), res.valid.copy()
    X[s_vals.size // 2], Z[s_vals.size // 2], V[s_vals.size // 2] = np.nan, np.nan, False
    P2 = CS.decimated_cloud(SimpleNamespace(x_mm=X, z_mm=Z, valid=V, T_base_opt=res.T_base_opt), 1e-3)
    assert P2.shape[0] <= P.shape[0], (P2.shape, P.shape)
    assert CS.decimated_cloud(SimpleNamespace(x_mm=np.zeros((0, 4)), z_mm=np.zeros((0, 4)), T_base_opt=np.zeros((0, 4, 4))),
                              1e-3).shape == (0, 3)
