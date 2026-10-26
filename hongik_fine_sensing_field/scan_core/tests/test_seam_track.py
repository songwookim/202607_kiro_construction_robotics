"""심 추적 (scan_core.seam_track, P2 2026-10-09): 이동 중 스트림 프레임의 후보 궤적 (좌·우 상부 코너, 상부 중심) → 심 직선.

합성 궤적 (센서 프레임 값을 자세에서 직접 계산): 백색 + 상관 잡음, 이상치 10 %, 한 궤적의 체제 전환 구간 → 잔차 최소 궤적 선택, 덮임,
결과 종류 ('late' / 'none' / 'groove_mismatch'), 심 기울기 보존, 광학 분기 뒤집기 뒤 재변환, 도착 예측, 유도량 (연구실 값)."""
import math

import numpy as np
import pytest

from scan_core.frames import make_transform, rot_z, unit
from scan_core.seam_track import (TrackRows, baseline_m, concat_rows, consistent_upto, fit_track, footprint_m, laser_footprint_s,
                                  moving_mask, predict_x_at, seam_frame, track_length_m, yaw_tolerance_rad, yaw_to_seam_rad)
from scan_core.types import Calib

K = 3.0
FLOOR = 0.02e-3                     # 센서 양자화 바닥 (m)
OPEN_HALF = 5.4e-3                  # 상부 개구 절반 (m)


def _calib(flip: bool = False) -> Calib:
    T = make_transform(rot_z(math.pi), (0.0, 0.0, 0.0)) if flip else np.eye(4)
    return Calib(T_flange_opt=T, eoat_yaw_rad=0.0, T_flange_eoat=np.eye(4), source="test")


def _seam(slope_deg: float = 0.0):
    """진짜 심: 점 c0 (판 위), 방향 u (base +y 에서 심 방향 기울기 slope_deg), 판 법선 n (u 에 수직, 위쪽)."""
    a = math.radians(slope_deg)
    u = np.array([0.0, math.cos(a), math.sin(a)])
    n = unit(np.array([0.0, -math.sin(a), math.cos(a)]))
    return np.array([0.6, -0.15, 0.05]), u, n


def _poses(c0, u, n, *, yaw_deg: float, s0: float, s1: float, N: int, r: float = 0.443, lateral0: float = 0.02,
           still: int = 3, drift_lat: float = 0.0):
    """광학 자세 (calib 단위 → 플랜지 = 광학): Z = −n, X = n×u 를 n 둘레로 yaw_deg 돌린 것, 원점 = 심 점 (s) + w·lat + n·r.
    still: 처음·끝 정지 프레임 수 (같은 자세 반복). drift_lat: 이동 동안 측면이 그만큼 바뀐다 (대략 직선이 심에 기울어 있다)."""
    w = unit(np.cross(n, u))
    c, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    X = c * w + s_ * u
    Z = -n
    Y = np.cross(Z, X)
    R = np.column_stack([X, Y, Z])
    ss = np.concatenate([np.full(still, s0), np.linspace(s0, s1, N), np.full(still, s1)])
    lat = np.concatenate([np.full(still, lateral0), np.linspace(lateral0, lateral0 + drift_lat, N), np.full(still, lateral0 + drift_lat)])
    return [make_transform(R, c0 + u * s + w * l + n * r) for s, l in zip(ss, lat)]


def _crossing(T, P0, d, n):
    """광학 자세 T 의 레이저 평면 (원점, 법선 Y) ∩ 직선 P0 + d·t → 센서 (x, z) [mm] 또는 None."""
    o, Y = T[:3, 3], T[:3, 1]
    den = float(Y @ d)
    if abs(den) < 1e-12:
        return None
    t = float(Y @ (o - P0)) / den
    P = P0 + d * t
    return (float((P - o) @ T[:3, 0]) * 1e3, float((P - o) @ T[:3, 2]) * 1e3)


def _rows(poses, c0, u, n, *, rng, sigma=0.03e-3, corr=0.0, outliers=0.0, switch_track=None, switch_mm=0.0, design=None,
          groove=None, still_tol=0.5e-3):
    """후보 궤적 세 개: 왼쪽 코너 (심 − w·a), 오른쪽 코너 (심 + w·a), 상부 중심 (심) + 판 점 (선 위 ±30–60 mm). 잡음 (센서 x, mm):
    백색 sigma + 상관 corr (AR(1) 0.9, 궤적 공통), 이상치 비율 outliers (±3 mm), switch_track 은 이동 가운데 1/3 에서 switch_mm 계단."""
    w = unit(np.cross(n, u))
    N = len(poses)
    cand = np.full((N, 3, 2), np.nan)
    plates = []
    ar = 0.0
    origins = np.array([T[:3, 3] for T in poses])
    mv, _p = moving_mask(origins, still_tol)
    mv_idx = np.flatnonzero(mv)
    lo_sw, hi_sw = (mv_idx[len(mv_idx) // 3], mv_idx[2 * len(mv_idx) // 3]) if mv_idx.size else (0, 0)
    for i, T in enumerate(poses):
        ar = 0.9 * ar + rng.normal(0.0, corr) if corr > 0 else 0.0
        for j, off in enumerate((-OPEN_HALF, OPEN_HALF, 0.0)):
            xz = _crossing(T, c0 + w * off, u, n)
            if xz is None:
                continue
            x = xz[0] + 1e3 * (rng.normal(0.0, sigma) + ar)
            if outliers and rng.random() < outliers:
                x += rng.choice([-3.0, 3.0])
            if switch_track == j and lo_sw <= i <= hi_sw:
                x += switch_mm
            cand[i, j] = (x, xz[1])
        pts = []
        for off in (-0.06, -0.045, -0.03, 0.03, 0.045, 0.06):
            xz = _crossing(T, c0 + w * off, u, n)
            if xz is not None:
                pts.append(xz)
        plates.append(np.asarray(pts, dtype=np.float32))
    design = np.ones(N, dtype=bool) if design is None else np.asarray(design, dtype=bool)
    groove = design.copy() if groove is None else np.asarray(groove, dtype=bool)
    return TrackRows(T_bf=np.stack(poses), cand_xz=cand, design=design, groove=groove, moving=mv, plate_xz=plates,
                     x_c=cand[:, 2, 0].copy(), part=np.zeros(N, dtype=np.int64), counter=np.arange(N, dtype=np.int64))


def _ang(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(unit(a) @ unit(b)))))))


def _fit(rows, t_hat0, calib=None):
    return fit_track(rows, calib or _calib(), k=K, tol_root_m=1.0e-3, t_hat0=t_hat0, floor_m=FLOOR)


# --------------------------------------------------------------------------------------
# 유도량 (연구실 값: r 443 mm, FOV 186.9 mm, 도착 허용 2.95 mm, 루트 허용 1.0 mm, θ_goal 0.2°)
# --------------------------------------------------------------------------------------
def test_derived_quantities_at_the_lab_values():
    psi = yaw_tolerance_rad(0.1869, 2.95e-3)
    assert math.degrees(psi) == pytest.approx(1.81, abs=0.01)
    b, b_star, cap = track_length_m(1.0e-3 / K, K, psi, math.radians(0.2), L_r_m=0.33, L_min_m=0.036, m_E_m=0.0059)
    assert b_star == pytest.approx(0.0504, abs=0.0002) and b == b_star and cap == pytest.approx(0.33 - 0.036 - 0.0059)
    assert baseline_m(1.0e-3 / K, K, psi, math.radians(0.2)) == b_star                       # b* = √2·tol_root / (ψ_tol − θ_goal)
    # 대략점이 가까우면 기선이 상한으로 줄어든다 (S 사이트·E 사이트가 최소 거리를 남긴다)
    b2, _b, cap2 = track_length_m(1.0e-3 / K, K, psi, math.radians(0.2), L_r_m=0.08, L_min_m=0.036, m_E_m=0.0059)
    assert b2 == pytest.approx(cap2) and b2 < b_star
    # 측정 바닥 (센서 양자화 σ_q 0.038 mm @ 443 mm) 의 기선: ~6 mm — 잰 잡음이 크면 그만큼 길어진다 (선형)
    b_q = baseline_m(0.038e-3, K, psi, math.radians(0.2))
    assert b_q == pytest.approx(0.0057, abs=0.0002) and baseline_m(0.11e-3, K, psi, math.radians(0.2)) == pytest.approx(b_q * 0.11 / 0.038)
    assert baseline_m(1e-3, K, math.radians(0.2), math.radians(0.2)) == float("inf")             # 여유 없음
    # σψ_model(b*) = √2·(tol_root/k)/b* → k·σψ = ψ_tol − θ_goal
    sig = math.sqrt(2.0) * (1.0e-3 / K) / b_star
    assert K * sig == pytest.approx(psi - math.radians(0.2), rel=1e-9)
    # 발자국: (FOV/2 + |x|)·sin ψ + e_hit
    assert footprint_m(0.1869, math.radians(17.4), 0.0, 2.95e-3) == pytest.approx(0.0934 * math.sin(math.radians(17.4)) + 2.95e-3,
                                                                                 abs=1e-4)
    assert footprint_m(0.1869, psi, 0.0, 2.95e-3) == pytest.approx(5.9e-3, abs=0.05e-3)          # m_E (연구실)
    assert footprint_m(0.1869, -0.1, 0.01, 0.0) == pytest.approx(footprint_m(0.1869, 0.1, -0.01, 0.0))
    # 그루브 x 를 모르면 (NaN) 0 으로 바꾸지 않고 불확도 전파: 시야 어디든 → 먼 쪽 팔 = 선 전체 (FOV)
    assert footprint_m(0.1869, 0.1, float("nan"), 2.95e-3) == pytest.approx(0.1869 * math.sin(0.1) + 2.95e-3)
    assert footprint_m(0.1869, 0.1, float("nan"), 0.0) > footprint_m(0.1869, 0.1, 0.0, 0.0)


def test_yaw_to_seam_is_signed_and_below_a_quarter_turn():
    _c, u, n = _seam()
    w = np.cross(n, u)
    for yaw in (-40.0, -10.0, 0.0, 7.5, 60.0, 89.0):
        for flip in (1.0, -1.0):                                   # 진행 부호 (툴이 반대로) — 선 방향 X 가 −X 여도 같은 |δ|
            X = flip * (math.cos(math.radians(yaw)) * w + math.sin(math.radians(yaw)) * u)
            R = np.column_stack([X, np.cross(-n, X), -n])
            d = yaw_to_seam_rad(R, n, u)
            assert abs(d) < 0.5 * math.pi
            assert abs(math.degrees(d)) == pytest.approx(min(abs(yaw), 180 - abs(yaw)), abs=1e-6)
            # 그 각만큼 n 둘레로 돌리면 선이 심에 수직
            c, s_ = math.cos(d), math.sin(d)
            Xr = X * c + np.cross(n, X) * s_ + n * float(n @ X) * (1 - c)
            assert abs(float(Xr @ u)) < 1e-9
    nn, uu, ww = seam_frame(n, u + n * 0.3)
    assert abs(float(uu @ nn)) < 1e-12 and float(np.cross(nn, uu) @ ww) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# 맞춤
# --------------------------------------------------------------------------------------
def test_picks_the_lowest_residual_track_with_noise_outliers_and_a_regime_switch():
    """세 궤적: 왼쪽 코너는 이동의 절반 넘게 이상치 (LMedS 가 무너져 잔차가 크다), 상부 중심은 잡음이 크고 (베벨 쪽 코너
    외삽 — 실기에서 중심이 사각 코너보다 흔들렸다), 오른쪽 코너는 이동 가운데 1/3 이 계단 (베벨/무베벨 판독 전환, 실기 b1 t06 의 코너
    1.3–3.6 mm 끌림) — 그 구간은 직선 밖으로 빠지고 나머지로 맞춘 방향이 정확하다: 덮임 ≥ ½ 인 것 중 잔차 최소 = 오른쪽."""
    rng = np.random.default_rng(3)
    c0, u, n = _seam()
    poses = _poses(c0, u, n, yaw_deg=14.8, s0=0.0, s1=0.050, N=34, drift_lat=0.006)
    rows = _rows(poses, c0, u, n, rng=rng, sigma=0.03e-3, corr=0.02e-3, outliers=0.10, switch_track=1, switch_mm=1.5)
    rows.cand_xz[:, 2, 0] += rng.normal(0.0, 0.08, rows.n)
    bad = rng.random(rows.n) < 0.6
    rows.cand_xz[bad, 0, 0] += rng.uniform(-4.0, 4.0, int(bad.sum()))
    est, outcome, info = _fit(rows, t_hat0=u)
    assert outcome == "ok" and est is not None, info
    assert 0.5 <= est.tracks["right"]["coverage"] < 0.9, est.tracks                 # 계단 구간은 직선 밖
    assert est.tracks["left"]["mad_mm"] > 10 * est.tracks["right"]["mad_mm"], est.tracks
    assert est.track == "right", est.tracks
    assert _ang(est.u, u) < 0.15 and _ang(est.n, n) < 0.05
    assert float(est.u @ u) > 0.0 and abs(float(est.u @ est.n)) < 1e-9
    assert est.length_m == pytest.approx(0.050, abs=0.003)
    assert est.sigma_psi_model == pytest.approx(math.sqrt(2) * (1e-3 / K) / est.length_m)
    assert est.arrival_consistent is True
    # 설계 유도 σ 이 자기 보고 σ 보다 보수적 (여유는 유도 σ 로만)
    assert est.sigma_psi_model > est.sigma_psi_self


def test_seam_slope_is_kept_in_u_and_n():
    rng = np.random.default_rng(4)
    c0, u, n = _seam(slope_deg=4.0)
    poses = _poses(c0, u, n, yaw_deg=-8.0, s0=0.0, s1=0.050, N=30)
    est, outcome, _ = _fit(_rows(poses, c0, u, n, rng=rng), t_hat0=u)
    assert outcome == "ok"
    assert _ang(est.u, u) < 0.1 and _ang(est.n, n) < 0.05
    assert est.u[2] == pytest.approx(math.sin(math.radians(4.0)), abs=0.002)       # 진행 방향 기울기 (트랜싯이 판 거리를 지킨다)


def test_late_none_and_groove_mismatch_outcomes():
    rng = np.random.default_rng(5)
    c0, u, n = _seam()
    poses = _poses(c0, u, n, yaw_deg=10.0, s0=0.0, s1=0.050, N=30)
    base = _rows(poses, c0, u, n, rng=rng)
    mv = np.flatnonzero(base.moving)
    # 늦게 나타남 (D3): 이동의 앞 60 % 는 설계 일치 아님 (선이 시편 끝 너머), 그 뒤로 이어짐 — 덮임은 직선 위 첫 프레임부터 세므로
    # (그루브가 나타나기 전 프레임은 직선의 반증이 아니다) 직선이 맞고, 기선은 나타난 뒤 구간만 (호출자가 모자람만큼 연장)
    d = base.design.copy()
    d[: mv[int(0.6 * mv.size)]] = False
    late = _rows(poses, c0, u, n, rng=np.random.default_rng(5), design=d)
    est, outcome, info = _fit(late, t_hat0=u)
    assert outcome == "ok" and est is not None, info
    assert est.coverage >= 0.9 and est.first_on_line >= mv[int(0.6 * mv.size)], (est.coverage, est.first_on_line)
    assert est.length_m == pytest.approx(0.4 * 0.050, abs=0.004) and _ang(est.u, u) < 0.3
    # 예전 규칙 (이동 전체 대비 덮임) 이면 0.4 — 경계 판정으로 멈췄다 (Y7 극단: '65 mm 중 102 mm')
    on = np.zeros(late.n, dtype=bool)
    on[np.flatnonzero(late.design & late.moving)] = True
    assert float(on.sum()) / float(late.moving.sum()) < 0.5
    # 직선 맞춤에 모자라게 늦게 나타남 (마지막 이동 프레임 하나 + 끝 정지 프레임 셋 = 4 < 5 점) → 'late' (같은 방향으로 더 간다)
    d2 = np.zeros(base.n, dtype=bool)
    d2[mv[-1]:] = True
    est, outcome, info = _fit(_rows(poses, c0, u, n, rng=np.random.default_rng(5), design=d2), t_hat0=u)
    assert est is None and outcome == "late", info
    assert info["coverage_after_first"] >= 0.5 and info["first_seen_fraction"] > 0.9
    # 없음: 설계 일치 프레임 0
    none = _rows(poses, c0, u, n, rng=rng, design=np.zeros(base.n, dtype=bool), groove=np.zeros(base.n, dtype=bool))
    est, outcome, info = _fit(none, t_hat0=u)
    assert est is None and outcome == "none", info
    # 흩어진 설계 일치 (이동의 1/4 만, 띄엄띄엄) → 'none' (직선 위 덮임 부족이고 늦게 나타난 것도 아님)
    sc = np.zeros(base.n, dtype=bool)
    sc[mv[::4]] = True
    est, outcome, info = _fit(_rows(poses, c0, u, n, rng=rng, design=sc), t_hat0=u)
    assert est is None and outcome == "none", info
    # 설계와 맞지 않는 그루브가 직선 위로 이동 내내 → 'groove_mismatch' (선은 시편 위에 있다)
    mis = _rows(poses, c0, u, n, rng=rng, design=np.zeros(base.n, dtype=bool), groove=np.ones(base.n, dtype=bool))
    est, outcome, info = _fit(mis, t_hat0=u)
    assert est is None and outcome == "groove_mismatch", info


def test_flipped_optical_branch_refit_recovers_the_seam():
    """센서 프레임 값은 그대로 두고 캘리브만 바꿔 다시 맞춘다 (프로브 'flip'): 진짜 캘리브가 Rz(π) 뒤집힌 분기면 공칭으로 맞춘 방향은
    거울상 (yaw 부호 반대) 이고, 뒤집은 캘리브로 다시 맞추면 진짜 심."""
    rng = np.random.default_rng(6)
    c0, u, n = _seam()
    true_cal = _calib(flip=True)
    opt = _poses(c0, u, n, yaw_deg=12.0, s0=0.0, s1=0.050, N=30, drift_lat=0.010)
    rows = _rows(opt, c0, u, n, rng=rng)
    inv = np.linalg.inv(true_cal.T_flange_opt)
    rows.T_bf = np.stack([T @ inv for T in opt])                     # 플랜지 자세 (진짜 캘리브로 그 광학 자세)
    wrong, o1, _ = _fit(rows, t_hat0=u, calib=_calib(flip=False))
    right, o2, _ = _fit(rows, t_hat0=u, calib=true_cal)
    assert o2 == "ok" and _ang(right.u, u) < 0.1
    assert o1 == "ok" and _ang(wrong.u, u) > 1.0


def test_concat_and_prediction_and_consistency_range():
    rng = np.random.default_rng(7)
    c0, u, n = _seam()
    a = _rows(_poses(c0, u, n, yaw_deg=6.0, s0=0.0, s1=0.050, N=25), c0, u, n, rng=rng)
    b = _rows(_poses(c0, u, n, yaw_deg=6.0, s0=0.050, s1=0.100, N=25), c0, u, n, rng=rng)
    both = concat_rows([a, b])
    assert both.n == a.n + b.n and len(both.plate_xz) == both.n
    est, outcome, _ = _fit(both, t_hat0=u)
    assert outcome == "ok" and est.length_m > 0.09
    # 도착 예측: 더 앞의 자세에서 레이저 평면 ∩ (상부 중심) 직선의 센서 x
    T = _poses(c0, u, n, yaw_deg=6.0, s0=0.20, s1=0.20, N=1, still=0, lateral0=0.013)[0]
    x_true = _crossing(T, c0, u, n)[0]
    assert predict_x_at(T, est) == pytest.approx(x_true, abs=0.05)
    assert consistent_upto(est, 0.5 * (est.s_first + est.s_last)) and not consistent_upto(est, est.s_last + 0.01)
    # 발자국: 선 양끝의 심 좌표 폭 = FOV·sin(yaw)
    s0, s1 = laser_footprint_s(T, c0, u, n, 0.19)
    assert (s1 - s0) == pytest.approx(0.19 * math.sin(math.radians(6.0)), abs=1e-4)


def test_section_noise_is_the_larger_of_the_design_bound_and_the_measured_residual():
    """σψ_model = √2·σ_단면/기선, σ_단면 = max(설계 상한 tol_root/k, 잰 궤적 잔차 σ): 설계 허용이 없으면 (None) 잰 잔차, 잰 잔차가 설계 상한보다
    크면 잰 잔차 (D5)."""
    rng = np.random.default_rng(8)
    c0, u, n = _seam()
    poses = _poses(c0, u, n, yaw_deg=5.0, s0=0.0, s1=0.050, N=30)
    quiet = _rows(poses, c0, u, n, rng=rng, sigma=0.03e-3)
    est_d, o_d, _ = _fit(quiet, t_hat0=u)
    assert o_d == "ok" and est_d.sigma_sec_source == "design" and est_d.sigma_sec_m == pytest.approx(1e-3 / K)
    est_m, o_m, _ = fit_track(quiet, _calib(), k=K, tol_root_m=None, t_hat0=u, floor_m=FLOOR)
    assert o_m == "ok" and est_m.sigma_sec_source == "measured" and FLOOR <= est_m.sigma_sec_m < 0.1e-3
    assert est_m.sigma_psi_model == pytest.approx(math.sqrt(2) * est_m.sigma_sec_m / est_m.length_m)
    noisy = _rows(poses, c0, u, n, rng=np.random.default_rng(9), sigma=0.6e-3)
    est_n, o_n, _ = _fit(noisy, t_hat0=u)
    assert o_n == "ok" and est_n.sigma_sec_source == "measured" and est_n.sigma_sec_m > 1e-3 / K
