"""3D 재구성 (설계 §5.5 / §10.5).

P_base = T_base←flange(t_i) · T_flange←opt · [x·1e-3, 0, z·1e-3, 1]ᵀ,   t_i = a_lo + T_est·k_i − dt
포즈 축 = 호스트 수신시각 (change-detection 샘플). 외삽 금지 → unsynced.
in_cruise = 실측 TCP 속도(플랜지 위치 차분, 창 평균)가 기준속도 ±5 % 이내 & ts 무결성 OK & synced.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .clock_sync import CounterClock, TransformBuffer, stall_model
from .types import Calib, ProfileFrame, RobotSample, ScanJob, ScanResult
from .validity import Z_VALID_MM_DEFAULT, dX_mm, valid_mask

__all__ = ["profile_to_base", "profiles_to_base", "measured_speed", "cruise_mask", "reconstruct_scan", "assemble",
           "FLAG_VALID", "FLAG_CRUISE", "FLAG_UNSYNCED", "FLAG_TSFLAG"]

FLAG_VALID = 1
FLAG_CRUISE = 2
FLAG_UNSYNCED = 4
FLAG_TSFLAG = 8


def profile_to_base(T_base_flange: np.ndarray, T_flange_opt: np.ndarray, x_mm, z_mm, valid=None) -> np.ndarray:
    """프로파일 1개 → base 점 (n,3) [m]. 무효열은 NaN."""
    T = np.asarray(T_base_flange, dtype=float) @ np.asarray(T_flange_opt, dtype=float)
    x = np.asarray(x_mm, dtype=float) * 1e-3
    z = np.asarray(z_mm, dtype=float) * 1e-3
    P = np.stack([x, np.zeros_like(x), z], axis=-1)
    out = P @ T[:3, :3].T + T[:3, 3]
    if valid is not None:
        out = out.copy()
        out[~np.asarray(valid, dtype=bool)] = np.nan
    return out


def profiles_to_base(T_base_flange: np.ndarray, T_flange_opt: np.ndarray, x_mm: np.ndarray, z_mm: np.ndarray,
                     valid: Optional[np.ndarray] = None) -> np.ndarray:
    """벡터화: T_base_flange (N,4,4), x/z (N,n) → (N,n,3) [m]."""
    T = np.asarray(T_base_flange, dtype=float) @ np.asarray(T_flange_opt, dtype=float)   # (N,4,4)
    x = np.asarray(x_mm, dtype=float) * 1e-3
    z = np.asarray(z_mm, dtype=float) * 1e-3
    R = T[:, :3, :3]
    out = (x[..., None] * R[:, None, :, 0] + z[..., None] * R[:, None, :, 2]) + T[:, None, :3, 3]
    if valid is not None:
        out[~np.asarray(valid, dtype=bool)] = np.nan
    return out


def measured_speed(t: np.ndarray, p: np.ndarray, window_s: float = 0.3) -> np.ndarray:
    """위치 샘플 (t[M], p[M,3]) → 각 샘플의 TCP 속력 [m/s]. 창 ±window/2 양끝 차분
    (change-detection 위상 지터 σ≈0.6 ms → 0.3 s 창에서 속도 오차 ≲ ±1 %)."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float).reshape(-1, 3)
    M = t.size
    v = np.full(M, np.nan)
    if M < 2:
        return v
    half = float(window_s) / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right") - 1
    lo = np.clip(lo, 0, M - 1)
    hi = np.clip(hi, 0, M - 1)
    same = hi <= lo
    lo[same] = np.clip(np.where(same, np.arange(M) - 1, lo), 0, M - 1)[same]
    hi[same] = np.clip(np.where(same, np.arange(M) + 1, hi), 0, M - 1)[same]
    dt = t[hi] - t[lo]
    ok = dt > 0
    v[ok] = np.linalg.norm(p[hi[ok]] - p[lo[ok]], axis=1) / dt[ok]
    return v


def cruise_mask(v: np.ndarray, tol_pct: float = 5.0, v_ref: Optional[float] = None) -> tuple[np.ndarray, float]:
    """|v − v_ref|/v_ref < tol. v_ref 미지정 시 = 상위 절반(> 0.5·max) 샘플의 중앙값 (순항 추정)."""
    v = np.asarray(v, dtype=float)
    fin = np.isfinite(v)
    if not np.any(fin):
        return np.zeros_like(v, dtype=bool), float("nan")
    if v_ref is None:
        vm = v[fin]
        hi = vm[vm > 0.5 * vm.max()]
        v_ref = float(np.median(hi)) if hi.size else float(np.median(vm))
    if not (v_ref > 0):
        return np.zeros_like(v, dtype=bool), float(v_ref)
    with np.errstate(invalid="ignore"):
        m = fin & (np.abs(v - v_ref) / v_ref < tol_pct / 100.0)
    return m, float(v_ref)


def _stack_frames(frames: Sequence[ProfileFrame]):
    ns = np.array([f.n_points for f in frames], dtype=int)
    n_mode = int(np.bincount(ns).argmax()) if ns.size else 0
    N = len(frames)
    x = np.full((N, n_mode), np.nan, dtype=np.float32)
    z = np.full((N, n_mode), np.nan, dtype=np.float32)
    changed = False
    for i, f in enumerate(frames):
        if f.n_points == n_mode:
            x[i] = f.x_mm
            z[i] = f.z_mm
        else:
            changed = True                      # 점 수 변화 → 해당 프레임 전열 무효(NaN) + 플래그
    return x, z, n_mode, changed


def reconstruct_scan(frames: Sequence[ProfileFrame], robot: Sequence[RobotSample], calib: Calib,
                     job: Optional[ScanJob] = None, *, fps_nominal: Optional[float] = None,
                     clock: Optional[CounterClock] = None, z_valid_mm=Z_VALID_MM_DEFAULT,
                     cruise_tol_pct: float = 5.0, speed_window_s: float = 0.3, ts_tol_ms: Optional[float] = None,
                     base_frame: str = "link0", v_ref: Optional[float] = None,
                     column_filter: bool = False, column_k: float = 5.0) -> ScanResult:
    """원시 프레임 + 로봇 샘플 → ScanResult. clock 미지정 시 전구간 CounterClock.fit(force=True)."""
    if len(frames) == 0:
        raise ValueError("no frames")
    frames = sorted(frames, key=lambda f: f.result_counter)
    fps = float(fps_nominal if fps_nominal is not None else (job.profile_fps if job is not None else 100.0))
    k = np.array([f.result_counter for f in frames], dtype=np.int64)
    t_rx = np.array([f.t_rx_s for f in frames], dtype=float)
    ts_ms = np.array([f.ts_ms for f in frames], dtype=np.int64)

    sm = stall_model(ts_ms, k, fps, tol_ms=ts_tol_ms)               # 센서 촬영 멈춤 → 보정 카운터 k_eff (clock_sync 문서)
    if clock is None:
        clock = CounterClock(fps_nominal=fps, mode="full_span")
        clock.push_many(sm.k_eff, t_rx)
        clock.fit(force=True)
    elif not clock.locked:
        clock.fit(force=True)
    t_acq = np.asarray(clock.stamp(sm.k_eff, calib.dt_s), dtype=float)
    ts_flag = sm.flags

    x, z, n_pts, n_changed = _stack_frames(frames)
    valid = valid_mask(x, z, z_valid_mm)
    refl_removed = 0.0
    if column_filter:
        from .validity import column_monotonic_mask
        cm = column_monotonic_mask(x, z, k=column_k)
        before = valid.sum()
        valid &= cm
        refl_removed = float((before - valid.sum()) / max(1, before))

    buf = TransformBuffer(change_detect=True)
    robot_sorted = sorted(robot, key=lambda r: r.t_rx_s)
    for r in robot_sorted:
        buf.push(r.t_rx_s, r.T_base_flange)
    T_bf, unsynced = buf.T_at_many(t_acq)
    T_bo = T_bf @ calib.T_flange_opt

    tb = buf.times
    pb = buf.positions
    v_samples = measured_speed(tb, pb, window_s=speed_window_s) if len(buf) >= 2 else np.zeros(0)
    if v_samples.size:
        fin = np.isfinite(v_samples)
        speed = np.interp(t_acq, tb[fin], v_samples[fin]) if fin.sum() >= 2 else np.full(t_acq.size, np.nan)
    else:
        speed = np.full(t_acq.size, np.nan)
    cm, v_used = cruise_mask(speed, tol_pct=cruise_tol_pct, v_ref=v_ref)
    in_cruise = cm & ~unsynced & ~ts_flag

    ramp_meta: dict = {}
    if job is not None:
        from .path_gen import expected_cruise_s_range, ramp_zones_m, sensor_frames
        fr = sensor_frames(job, calib)
        s_m = fr.s_of(T_bo[:, :3, 3])
        T_scan, T_plate = fr.T_base_scan, fr.T_base_plate
        dx = float(dX_mm(job.standoff_m * 1e3))
        base_frame = job.base_frame or base_frame
        # 리드가 없는 쪽(3G)의 가감속 구간은 시편 안에 있다: 해당 프레임은 valid 그대로, in_cruise 만 False (실측 속도 기준)
        # 램프 거리 = v²/(2a) + v·(속도 측정창/2): 창 평균 속도가 ±5 % 밴드를 벗어나는 구간까지 포함
        settle = float(speed_window_s) / 2.0
        zones = ramp_zones_m(job, settle_s=settle)
        in_ramp = np.zeros(t_acq.size, dtype=bool)
        for lo, hi in zones:
            in_ramp |= (s_m >= lo - 1e-9) & (s_m <= hi + 1e-9)
        ramp_meta = {
            "s_scan_from_m": float(fr.s_scan_from), "s_scan_to_m": float(fr.s_scan_to), "travel_sign": int(fr.travel_sign),
            "cruise_s_range_m": [float(v) for v in expected_cruise_s_range(job, settle_s=settle)],
            "ramp_settle_s": settle,
            "ramp_zones_m": [[float(lo), float(hi)] for lo, hi in zones],
            "n_ramp_frames": int(in_ramp.sum()),
            "n_ramp_frames_valid": int((in_ramp & (valid.sum(axis=1) > 0)).sum()),
        }
    else:
        s_m = np.full(t_acq.size, np.nan)
        T_scan = np.eye(4)
        T_plate = np.eye(4)
        dx = float("nan")
    ds_mm = float(v_used / clock.fps_measured() * 1e3) if np.isfinite(v_used) else float("nan")

    meta = {
        "clock": clock.to_dict(), "clock_residuals": clock.residual_stats(), "sensor_stalls": sm.to_dict(), "n_points": n_pts,
        "n_points_changed": bool(n_changed), "reflection_removed_ratio": refl_removed,
        "v_ref_mps": v_used, "fps_nominal": fps, "n_robot_samples": len(buf), "cruise_tol_pct": cruise_tol_pct,
        "speed_window_s": speed_window_s, "z_valid_mm": [float(z_valid_mm[0]), float(z_valid_mm[1])],
    }
    if job is not None:
        meta["job"] = job.to_dict()
        meta.update(ramp_meta)
    return ScanResult(
        x_mm=x, z_mm=z, valid=valid, result_counter=k, ts_ms=ts_ms, t_rx_s=t_rx, t_acq_s=t_acq,
        T_base_flange=T_bf, T_base_opt=T_bo, in_cruise=in_cruise, unsynced=unsynced, ts_flag=ts_flag,
        speed_mps=speed, s_m=s_m, base_frame=base_frame, T_base_scan=T_scan, T_base_plate=T_plate,
        ds_mm=ds_mm, dx_mm=dx, calib=calib, meta=meta,
    )


def assemble(result: ScanResult, only_cruise: bool = False, only_valid: bool = True, frame: str = "base"):
    """ScanResult → 점군 (M,3) + profile_idx (M,) + col_idx (M,) + flags (M,) uint8.
    frame: 'base' | 'scan' | 'plate'."""
    P = profiles_to_base(result.T_base_flange, result.calib.T_flange_opt if result.calib is not None else np.eye(4),
                         result.x_mm, result.z_mm)
    N, n = result.x_mm.shape
    sel = np.ones((N, n), dtype=bool)
    if only_valid:
        sel &= result.valid
    if only_cruise:
        sel &= result.in_cruise[:, None]
    sel &= np.all(np.isfinite(P), axis=-1)
    pi, ci = np.nonzero(sel)
    pts = P[pi, ci]
    if frame == "scan":
        from .frames import inv_transform, transform_points
        pts = transform_points(inv_transform(result.T_base_scan), pts)
    elif frame == "plate":
        from .frames import inv_transform, transform_points
        pts = transform_points(inv_transform(result.T_base_plate), pts)
    elif frame != "base":
        raise ValueError("frame must be base|scan|plate")
    flags = np.zeros(pi.size, dtype=np.uint8)
    flags |= (result.valid[pi, ci] * FLAG_VALID).astype(np.uint8)
    flags |= (result.in_cruise[pi] * FLAG_CRUISE).astype(np.uint8)
    flags |= (result.unsynced[pi] * FLAG_UNSYNCED).astype(np.uint8)
    flags |= (result.ts_flag[pi] * FLAG_TSFLAG).astype(np.uint8)
    return pts, pi.astype(np.int64), ci.astype(np.int64), flags
