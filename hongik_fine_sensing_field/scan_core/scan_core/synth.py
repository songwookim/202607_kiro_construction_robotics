"""진실값 주입 합성 데이터 생성기 (테스트·알고리즘 검증 전용; Isaac Sim 대체).

장면(Scene)은 판 프레임(원점 start, x=u, y=n×u, z=n) 위의 높이맵 h(xp, yp) [m] 로 정의한다
(NaN = 표면 없음 → 무반환). 프리미티브: 유한 평면, 구, (기울인) 단차 에지, 게이지 블록, 합성.

센서 광선 모델: 열 i 의 점은 광학 프레임에서 x_i(z) = k_i·(a + b·z) (k_i = (i − (N−1)/2)·fov_points/N,
dX(z) ≈ a + b·z; n_points < 1408 이면 열 서브샘플링으로 취급해 FOV 를 유지) → 광선은 가상 레이저 원점 (0, −a/b) 를 지나는 직선. 표면과의 첫 교점(위에서 진입)을
격자 브래킷 + 이분법으로 구한다.

시간: 프레임 k 취득시각 t_acq = t0 + T_s·k (T_s = (1/fps)(1+ppm·1e-6)),
ts_ms = floor((t_acq − t_boot)·1000 / quant)·quant, t_rx = t_acq + rx_latency(= dt 진실) + Exp(jitter).
로봇: 10 ms 마다 갱신, 폴러 수신 t_rx = t + robot_rx_latency + U(0, 1/poll_rate) (change-detection 위상).
관절각은 만들지 않는다 — 플랜지 포즈를 직접 노출한다 (scan_core 테스트에는 충분).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .frames import inv_transform, make_transform, transform_points
from .path_gen import ScanFrames, plate_frame, sensor_frames
from .specimen import GrooveSpec
from .types import Calib, ProfileFrame, RobotSample, ScanJob
from .validity import Z_VALID_MM_DEFAULT, dx_linear_coeffs

__all__ = ["Scene", "PlaneScene", "SphereScene", "StepScene", "GaugeBlockScene", "CompositeScene", "GrooveScene",
           "TrapezoidProfile", "SynthConfig", "SynthScan", "simulate_profile", "simulate_scan", "static_profile"]


# --------------------------------------------------------------------------------------
# 장면
# --------------------------------------------------------------------------------------
class Scene:
    def height(self, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class PlaneScene(Scene):
    """유한 평판: x_range/y_range 안은 h0, 밖은 outside (기본 NaN = 무반환; 테이블면 높이를 넣어도 됨)."""
    h0: float = 0.0
    x_range: tuple = (-math.inf, math.inf)
    y_range: tuple = (-math.inf, math.inf)
    outside: float = math.nan

    def height(self, xp, yp):
        inside = (xp >= self.x_range[0]) & (xp <= self.x_range[1]) & (yp >= self.y_range[0]) & (yp <= self.y_range[1])
        return np.where(inside, self.h0, self.outside)


@dataclass
class SphereScene(Scene):
    """평면 h0 위에 놓인 구(중심 (xc, yc, hc), 반지름 r). 구 밖은 h0."""
    xc: float
    yc: float
    radius: float
    hc: Optional[float] = None       # None → 구가 평면에 접함 (hc = h0 + r)
    h0: float = 0.0

    def height(self, xp, yp):
        hc = self.h0 + self.radius if self.hc is None else self.hc
        d2 = (xp - self.xc) ** 2 + (yp - self.yc) ** 2
        r2 = self.radius ** 2
        with np.errstate(invalid="ignore"):
            top = hc + np.sqrt(np.clip(r2 - d2, 0.0, None))
        return np.where(d2 < r2, top, self.h0)


@dataclass
class StepScene(Scene):
    """단차 에지: 에지 직선 xp = x0 + tan(angle)·yp (angle=0 → 진행방향에 수직). 직선의 +x 쪽이 h_high."""
    h_high: float
    x0: float = 0.0
    angle_deg: float = 0.0
    h_low: float = 0.0
    high_side: int = +1

    def height(self, xp, yp):
        side = (xp - self.x0) - math.tan(math.radians(self.angle_deg)) * yp
        return np.where(self.high_side * side > 0, self.h_high, self.h_low)

    def edge_x_at(self, yp: float) -> float:
        return self.x0 + math.tan(math.radians(self.angle_deg)) * yp


@dataclass
class GaugeBlockScene(Scene):
    """평면 h0 위 직육면체 블록 (중심 (xc, yc), 크기 (sx, sy), 높이 h)."""
    xc: float
    yc: float
    sx: float
    sy: float
    h: float
    h0: float = 0.0

    def height(self, xp, yp):
        inside = (np.abs(xp - self.xc) <= self.sx / 2) & (np.abs(yp - self.yc) <= self.sy / 2)
        return np.where(inside, self.h0 + self.h, self.h0)


@dataclass
class GrooveScene(Scene):
    """그루브 시편 (GrooveSpec) 높이맵 [m]. 판 상면 = h_top (기본 0 = 판 프레임 z=0, 즉 ScanJob start/end 가 놓인 면),
    그루브 바닥(뒷댐재 상면) = h_top − t. 그루브(용접선) 축은 판 프레임 x(진행) 축에서 angle_deg 만큼 회전하고
    루트갭 중심선이 (x, y) = (0, yc) 를 지난다 (angle_deg ≠ 0 → H2 '비스듬히 가로지른 라인' 재현).
    lateral = (yp − yc)·cos a − xp·sin a [m] → spec.nominal_cross_section(lateral·1e3) [mm]; +x_spec = +lateral(= 판 프레임 +y ≈ 센서 +X).
    판 범위 밖(x_range/y_range) 은 outside (기본 NaN = 무반환).

    근수직면(직각 모서리·루트페이스) 가림: simulate_profile 의 광선 추적(광학 원점에서 내려오는 광선의 첫 교점)이 자동으로
    처리한다 — 광선이 판 모서리를 지나 반대편 벽에 닿으면 벽 위의 점을, 벽에 가려진 바닥은 돌려주지 않는다.
    """
    spec: GrooveSpec = field(default_factory=GrooveSpec)
    yc: float = 0.0                 # 루트갭 중심의 판 프레임 y [m] (센서 중심 오정렬 재현)
    angle_deg: float = 0.0          # 그루브 축 vs 진행축 x
    h_top: float = 0.0              # 판 상면 높이 [m] (left 판; right 판은 + hi_lo)
    x_range: tuple = (-math.inf, math.inf)
    y_range: tuple = (-math.inf, math.inf)
    outside: float = math.nan

    @classmethod
    def from_spec(cls, spec: GrooveSpec, yc: float = 0.0, angle_deg: float = 0.0, h_top: float = 0.0,
                  centered: bool = True) -> "GrooveScene":
        """시편 치수대로 유한 판: x ∈ [0, weld_length] (centered=False) 또는 ±weld_length/2, y ∈ yc ± plate_width/2."""
        L = spec.weld_length_mm * 1e-3
        W = spec.plate_width_mm * 1e-3
        xr = (-L / 2, L / 2) if centered else (0.0, L)
        return cls(spec=spec, yc=yc, angle_deg=angle_deg, h_top=h_top, x_range=xr, y_range=(yc - W / 2, yc + W / 2))

    def lateral(self, xp, yp):
        a = math.radians(self.angle_deg)
        return (np.asarray(yp, dtype=float) - self.yc) * math.cos(a) - np.asarray(xp, dtype=float) * math.sin(a)

    def height(self, xp, yp):
        xp = np.asarray(xp, dtype=float)
        yp = np.asarray(yp, dtype=float)
        lat_mm = self.lateral(xp, yp) * 1e3
        h = self.h_top - self.spec.thickness_mm * 1e-3 + self.spec.nominal_cross_section(lat_mm) * 1e-3
        inside = (xp >= self.x_range[0]) & (xp <= self.x_range[1]) & (yp >= self.y_range[0]) & (yp <= self.y_range[1])
        return np.where(inside, h, self.outside)

    def root_centre_y(self, xp: float = 0.0) -> float:
        """진행 위치 xp 에서 루트갭 중심선의 판 프레임 y [m]."""
        a = math.radians(self.angle_deg)
        return self.yc + math.tan(a) * float(xp)


@dataclass
class CompositeScene(Scene):
    """여러 장면의 최댓값 (NaN 은 무시; 전부 NaN 이면 NaN)."""
    scenes: list = field(default_factory=list)

    def height(self, xp, yp):
        hs = np.stack([np.asarray(s.height(xp, yp), dtype=float) for s in self.scenes])
        allnan = np.all(np.isnan(hs), axis=0)
        with np.errstate(all="ignore"):
            out = np.nanmax(np.where(np.isnan(hs), -np.inf, hs), axis=0)
        out = np.where(allnan, np.nan, out)
        return out


# --------------------------------------------------------------------------------------
# 사다리꼴 속도 프로파일
# --------------------------------------------------------------------------------------
class TrapezoidProfile:
    """s0 → s1 을 최대속도 v, 가속도 a 로 (삼각형 프로파일 자동)."""

    def __init__(self, s0: float, s1: float, v: float, a: float):
        self.s0, self.s1 = float(s0), float(s1)
        self.D = self.s1 - self.s0
        self.dir = 1.0 if self.D >= 0 else -1.0
        D = abs(self.D)
        v, a = float(v), float(a)
        if v * v / a > D:                       # 삼각형
            v = math.sqrt(D * a)
        self.v, self.a = v, a
        self.t_acc = v / a
        self.d_acc = 0.5 * a * self.t_acc ** 2
        self.t_cruise = (D - 2 * self.d_acc) / v if v > 0 else 0.0
        self.duration = 2 * self.t_acc + self.t_cruise

    def s(self, t):
        t = np.asarray(t, dtype=float)
        ta, tc, a, v = self.t_acc, self.t_cruise, self.a, self.v
        d = np.where(t <= 0, 0.0,
            np.where(t < ta, 0.5 * a * t ** 2,
            np.where(t < ta + tc, self.d_acc + v * (t - ta),
            np.where(t < 2 * ta + tc, self.d_acc + v * tc + v * (t - ta - tc) - 0.5 * a * (t - ta - tc) ** 2,
                     abs(self.D)))))
        return self.s0 + self.dir * d

    def v_of(self, t):
        t = np.asarray(t, dtype=float)
        ta, tc, a, v = self.t_acc, self.t_cruise, self.a, self.v
        return np.where(t <= 0, 0.0, np.where(t < ta, a * t, np.where(t < ta + tc, v,
               np.where(t < 2 * ta + tc, v - a * (t - ta - tc), 0.0))))


# --------------------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------------------
@dataclass
class SynthConfig:
    n_points: int = 1408
    fov_points: int = 1408               # 열 간격 스케일 기준 (n_points < fov_points 면 열 서브샘플링: FOV 유지)
    sigma_z_mm: float = 0.0
    dropout_prob: float = 0.0            # 열 NaN 확률
    # 반사 띠 (2026-10-03, 실기 at06/at07 의 모서리 상호 반사·광택 비드 재현): 프레임마다 reflection_prob 확률로 유효 열 중 연속
    # reflection_width_cols 개의 점이 실제 표면보다 reflection_depth_mm 만큼 멀게 (아래로) 보인다. 띠의 시작은 프로파일에서 가까운
    # 10 % 점 (판 상면) 보다 reflection_min_depth_mm 이상 먼 점 (그루브 안 — 실기 반사는 오목 모서리에서 났다; 0 = 어디서나).
    # reflection_prob 0 이면 난수를 쓰지 않는다.
    reflection_prob: float = 0.0
    reflection_depth_mm: tuple = (1.0, 4.0)
    reflection_width_cols: tuple = (10, 40)
    reflection_min_depth_mm: float = 0.5
    frame_drop_prob: float = 0.0         # 프레임 누락 확률 (카운터 갭)
    drift_ppm: float = 0.0               # 센서 클럭 드리프트
    ts_quant_ms: float = 1.0             # ts_ms 양자화
    rx_latency_s: float = 0.0            # 센서→호스트 고정 지연 (= dt 진실값)
    rx_jitter_scale_s: float = 0.0       # 수신 지터 (지수분포 scale)
    # 실기 at10 (2026-10-03) 재현: 센서 촬영 멈춤 ((프레임 번호, ms) — 그 번호부터 촬영이 ms 늦다; ts_ms 도 같이) 과 호스트 수신
    # 막힘 ((from_s, to_s) — t0 기준 이 구간에 촬영된 프레임은 to_s 에 한꺼번에 수신)
    stalls: tuple = ()
    rx_stalls_s: tuple = ()
    robot_update_s: float = 0.010        # CB 상태 갱신 주기
    robot_poll_rate_hz: float = 500.0    # 폴링 → 위상 지터 U(0, 1/rate); 0 이면 지터 없음
    robot_rx_latency_s: float = 0.0
    pre_roll_s: float = 0.3              # 모션 전 스트리밍 시간
    post_roll_s: float = 0.3
    dwell_s: float = 0.5                 # 왕복 사이 정지
    z_grid_mm: tuple = (150.0, 650.0, 2.0)   # 광선 브래킷 격자 (lo, hi, step)
    bisect_iters: int = 25
    z_valid_mm: tuple = Z_VALID_MM_DEFAULT
    t0_s: float = 1000.0                 # 호스트 monotonic 기준
    boot_offset_s: float = 3600.0        # 센서 부팅 후 경과 (ts_ms 오프셋)
    seed: int = 0


@dataclass
class SynthScan:
    frames: list
    robot: list
    truth: dict
    job: ScanJob
    calib_true: Calib
    scene: Scene


# --------------------------------------------------------------------------------------
# 광선 추적
# --------------------------------------------------------------------------------------
def simulate_profile(T_base_opt: np.ndarray, scene: Scene, T_base_plate: np.ndarray, cfg: SynthConfig,
                     rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray]:
    """광학 포즈 1개 → (x_mm, z_mm) [n_points]. 노이즈·드롭아웃·센티널 적용."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    N = int(cfg.n_points)
    a, b = dx_linear_coeffs()
    k = (np.arange(N, dtype=float) - (N - 1) / 2.0) * (float(cfg.fov_points) / N)
    T = np.asarray(T_base_opt, dtype=float)
    o, X, Z = T[:3, 3], T[:3, 0], T[:3, 2]
    R_p = np.asarray(T_base_plate, dtype=float)[:3, :3]
    p0 = np.asarray(T_base_plate, dtype=float)[:3, 3]
    zlo, zhi, step = cfg.z_grid_mm
    zg = np.arange(zlo, zhi + step, step)                                   # (G,)

    def f_of(zmm: np.ndarray) -> np.ndarray:
        """zmm (N,G) or (N,) → q_z − h  (양수 = 표면 위)."""
        xm = k.reshape(-1, *([1] * (zmm.ndim - 1))) * (a + b * zmm)         # mm
        p = o + xm[..., None] * 1e-3 * X + zmm[..., None] * 1e-3 * Z         # (...,3) base
        q = (p - p0) @ R_p                                                   # plate coords
        h = np.asarray(scene.height(q[..., 0], q[..., 1]), dtype=float)
        f = q[..., 2] - h
        return np.where(np.isnan(f), np.inf, f)

    F = f_of(np.broadcast_to(zg, (N, zg.size)).copy())                       # (N,G)
    below = F <= 0
    first = np.argmax(below, axis=1)                                          # 첫 진입 인덱스
    hit = below.any(axis=1) & (first > 0)
    z_out = np.full(N, np.nan)
    x_out = np.full(N, np.nan)
    if hit.any():
        idx = np.nonzero(hit)[0]
        lo = zg[first[idx] - 1].astype(float)
        hi = zg[first[idx]].astype(float)
        for _ in range(int(cfg.bisect_iters)):
            mid = 0.5 * (lo + hi)
            fm = np.full(N, np.inf)
            zz = np.full(N, np.nan)
            zz[idx] = mid
            fm_all = f_of(np.where(np.isnan(zz), zlo, zz))
            fm = fm_all[idx]
            pos = fm > 0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        zs = 0.5 * (lo + hi)
        z_out[idx] = zs
        x_out[idx] = k[idx] * (a + b * zs)
    if cfg.sigma_z_mm > 0:
        z_out = z_out + rng.normal(0.0, cfg.sigma_z_mm, N)
    if cfg.reflection_prob > 0 and rng.random() < cfg.reflection_prob:
        valid = np.flatnonzero(np.isfinite(z_out) & (z_out > 0))
        if valid.size and cfg.reflection_min_depth_mm > 0:
            zv_ = z_out[valid]
            valid = valid[zv_ >= float(np.percentile(zv_, 10)) + float(cfg.reflection_min_depth_mm)]
        if valid.size:
            w = int(rng.integers(int(cfg.reflection_width_cols[0]), int(cfg.reflection_width_cols[1]) + 1))
            i0 = int(valid[int(rng.integers(0, valid.size))])
            sl = slice(i0, min(N, i0 + w))
            z_out[sl] = z_out[sl] + float(rng.uniform(*cfg.reflection_depth_mm))
            x_out[sl] = np.where(np.isfinite(z_out[sl]), k[sl] * (a + b * z_out[sl]), x_out[sl])   # 같은 광선 위
    # 교정범위 밖 → z=0 센티널 (x 유지)
    zv = cfg.z_valid_mm
    with np.errstate(invalid="ignore"):
        oor = np.isfinite(z_out) & ((z_out <= zv[0]) | (z_out >= zv[1]))
    z_out = np.where(oor, 0.0, z_out)
    if cfg.dropout_prob > 0:
        drop = rng.random(N) < cfg.dropout_prob
        x_out[drop] = np.nan
        z_out[drop] = np.nan
    return x_out.astype(np.float32), z_out.astype(np.float32)


def static_profile(T_base_flange: np.ndarray, calib_true: Calib, scene: Scene, T_base_plate: np.ndarray,
                   cfg: SynthConfig, rng: Optional[np.random.Generator] = None):
    """정지 자세 1개의 프로파일 (핸드아이 캘리브 합성용)."""
    return simulate_profile(np.asarray(T_base_flange) @ calib_true.T_flange_opt, scene, T_base_plate, cfg, rng)


# --------------------------------------------------------------------------------------
# 스캔 시뮬레이션
# --------------------------------------------------------------------------------------
def simulate_scan(job: ScanJob, calib_true: Calib, scene: Scene, cfg: SynthConfig,
                  frames_obj: Optional[ScanFrames] = None) -> SynthScan:
    """ScanJob 대로 로봇을 움직이며 프레임·로봇 샘플·진실값 생성."""
    rng = np.random.default_rng(cfg.seed)
    fr = frames_obj if frames_obj is not None else sensor_frames(job, calib_true)
    T_plate = fr.T_base_plate
    # 스캔 s 구간은 ScanFrames 가 정한다 (1G: −lead_in→L+lead_out, 3G: vertical_direction 에 따라 한쪽 리드만)
    s_in, s_out = fr.s_scan_from, fr.s_scan_to
    profs = [TrapezoidProfile(s_in, s_out, job.speed_mps, job.accel_mps2)]
    if job.bidirectional:
        profs.append(TrapezoidProfile(s_out, s_in, job.speed_mps, job.accel_mps2))
    t_motion0 = cfg.t0_s + cfg.pre_roll_s
    segs = []                                     # (t_start, profile)
    t = t_motion0
    for p in profs:
        segs.append((t, p))
        t += p.duration + cfg.dwell_s
    t_end = t - cfg.dwell_s + cfg.post_roll_s

    def s_at(tt: float) -> float:
        s = profs[0].s0
        for t_s, p in segs:
            if tt >= t_s:
                s = float(p.s(min(tt - t_s, p.duration)))
        return s

    def v_at(tt: float) -> float:
        for t_s, p in segs:
            if t_s <= tt <= t_s + p.duration:
                return float(p.v_of(tt - t_s))
        return 0.0

    # 센서 프레임
    T_s = (1.0 / job.profile_fps) * (1.0 + cfg.drift_ppm * 1e-6)
    n_frames = int(math.floor((t_end - cfg.t0_s) / T_s)) + 1
    frames: list[ProfileFrame] = []
    truth_T, truth_t, truth_s, truth_k, truth_v = [], [], [], [], []
    stall_list = sorted((int(a), float(b)) for a, b in cfg.stalls)
    for kk in range(n_frames):
        t_acq = cfg.t0_s + T_s * kk + sum(ms for k_st, ms in stall_list if kk >= k_st) * 1e-3
        s = s_at(t_acq)
        T_bf = fr.T_base_flange(s)
        T_bo = T_bf @ calib_true.T_flange_opt
        truth_T.append(T_bo)
        truth_t.append(t_acq)
        truth_s.append(s)
        truth_k.append(kk)
        truth_v.append(v_at(t_acq))
        if cfg.frame_drop_prob > 0 and rng.random() < cfg.frame_drop_prob:
            continue
        x, z = simulate_profile(T_bo, scene, T_plate, cfg, rng)
        ts_raw = (t_acq - cfg.t0_s + cfg.boot_offset_s) * 1000.0
        q = cfg.ts_quant_ms
        ts_ms = int(math.floor(ts_raw / q) * q) if q > 0 else int(round(ts_raw))
        jitter = rng.exponential(cfg.rx_jitter_scale_s) if cfg.rx_jitter_scale_s > 0 else 0.0
        t_rx = t_acq + cfg.rx_latency_s + jitter
        for a, b in cfg.rx_stalls_s:
            if cfg.t0_s + a <= t_acq < cfg.t0_s + b:
                t_rx = max(t_rx, cfg.t0_s + b + cfg.rx_latency_s + jitter)
        frames.append(ProfileFrame(x_mm=x, z_mm=z, ts_ms=ts_ms, result_counter=kk, t_rx_s=t_rx, global_counter=kk))
    # 로봇 샘플 (change-detection: 갱신 틱마다 1개)
    robot: list[RobotSample] = []
    n_r = int(math.floor((t_end - cfg.t0_s) / cfg.robot_update_s)) + 1
    for j in range(n_r):
        tj = cfg.t0_s + j * cfg.robot_update_s
        T_bf = fr.T_base_flange(s_at(tj))
        phase = rng.uniform(0.0, 1.0 / cfg.robot_poll_rate_hz) if cfg.robot_poll_rate_hz > 0 else 0.0
        robot.append(RobotSample(t_rx_s=tj + cfg.robot_rx_latency_s + phase, T_base_flange=T_bf))
    truth = {
        "T_flange_opt": calib_true.T_flange_opt.copy(), "dt_s": cfg.rx_latency_s, "T_base_opt": np.stack(truth_T),
        "t_acq_s": np.array(truth_t), "s_m": np.array(truth_s), "k": np.array(truth_k, dtype=np.int64),
        "v_mps": np.array(truth_v), "T_sensor_s": T_s, "t_motion0_s": t_motion0, "segments": [(ts, p.duration) for ts, p in segs],
        "T_base_plate": T_plate,
    }
    return SynthScan(frames=frames, robot=robot, truth=truth, job=job, calib_true=calib_true, scene=scene)
