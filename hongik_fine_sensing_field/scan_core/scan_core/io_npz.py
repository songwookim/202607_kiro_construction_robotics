"""ScanResult 저장/로드: <dir>/scan.npz (배열) + <dir>/meta.json (calib, qc, meta). h5py 있으면 raw.h5 도 선택."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .types import Calib, QCReport, ScanResult, _jsonable

__all__ = ["save_scan_result", "load_scan_result", "save_h5", "has_h5py", "save_cloud_ply"]

NPZ_NAME = "scan.npz"
META_NAME = "meta.json"


def has_h5py() -> bool:
    try:
        import h5py  # noqa: F401
        return True
    except Exception:
        return False


def save_scan_result(out_dir, result: ScanResult, extra_meta: Optional[dict] = None, h5: bool = False) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in result.array_fields().items()}
    np.savez_compressed(out / NPZ_NAME, **arrays)
    meta = {
        "base_frame": result.base_frame, "ds_mm": float(result.ds_mm), "dx_mm": float(result.dx_mm),
        "calib": result.calib.to_dict() if result.calib is not None else None,
        "qc": result.qc.to_dict() if result.qc is not None else None,
        "meta": _jsonable(result.meta), "n_profiles": result.n_profiles, "n_points": result.n_points,
    }
    if extra_meta:
        meta["extra"] = _jsonable(extra_meta)
    with open(out / META_NAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=_default)
    if h5 and has_h5py():
        save_h5(out / "raw.h5", result)
    return out


def load_scan_result(in_dir) -> ScanResult:
    d = Path(in_dir)
    with np.load(d / NPZ_NAME) as z:
        arrays = {k: z[k] for k in z.files}
    with open(d / META_NAME, "r", encoding="utf-8") as f:
        meta = json.load(f)
    calib = Calib.from_dict(meta["calib"]) if meta.get("calib") else None
    qc = QCReport.from_dict(meta["qc"]) if meta.get("qc") else None
    return ScanResult(
        x_mm=arrays["x_mm"], z_mm=arrays["z_mm"], valid=arrays["valid"].astype(bool),
        result_counter=arrays["result_counter"], ts_ms=arrays["ts_ms"], t_rx_s=arrays["t_rx_s"],
        t_acq_s=arrays["t_acq_s"], T_base_flange=arrays["T_base_flange"], T_base_opt=arrays["T_base_opt"],
        in_cruise=arrays["in_cruise"].astype(bool), unsynced=arrays["unsynced"].astype(bool),
        ts_flag=arrays["ts_flag"].astype(bool), speed_mps=arrays["speed_mps"], s_m=arrays["s_m"],
        base_frame=meta.get("base_frame", "link0"), T_base_scan=arrays["T_base_scan"],
        T_base_plate=arrays.get("T_base_plate", np.eye(4)), ds_mm=float(meta.get("ds_mm", float("nan"))),
        dx_mm=float(meta.get("dx_mm", float("nan"))), calib=calib, qc=qc, meta=dict(meta.get("meta", {})),
    )


def save_h5(path, result: ScanResult) -> None:
    import h5py

    with h5py.File(path, "w") as h:
        g = h.create_group("profiles")
        g.create_dataset("x", data=result.x_mm, compression="gzip")
        g.create_dataset("z", data=result.z_mm, compression="gzip")
        g.create_dataset("valid", data=result.valid, compression="gzip")
        g.create_dataset("ts_ms", data=result.ts_ms.astype(np.uint64))
        g.create_dataset("result_counter", data=result.result_counter.astype(np.uint32))
        g.create_dataset("t_rx_ns", data=(result.t_rx_s * 1e9).astype(np.int64))
        g.create_dataset("t_stamp_ns", data=(result.t_acq_s * 1e9).astype(np.int64))
        r = h.create_group("robot")
        r.create_dataset("T_base_flange", data=result.T_base_flange)
        d = h.create_group("derived")
        d.create_dataset("T_base_opt", data=result.T_base_opt)
        d.create_dataset("in_cruise", data=result.in_cruise)
        d.create_dataset("unsynced", data=result.unsynced)
        d.create_dataset("s_m", data=result.s_m)
        c = h.create_group("calib")
        if result.calib is not None:
            c.create_dataset("T_flange_opt", data=result.calib.T_flange_opt)
            c.attrs["dt_s"] = result.calib.dt_s
            c.attrs["x_sign"] = result.calib.x_sign
        h.attrs["base_frame"] = result.base_frame
        h.attrs["meta_json"] = json.dumps(_jsonable(result.meta), default=_default)


def save_cloud_ply(path, points: np.ndarray, profile_idx=None, col_idx=None, flags=None) -> None:
    """binary little-endian PLY: x y z [+ profile_idx u32, col_idx u16, quality_flag u8]."""
    P = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = P.shape[0]
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if profile_idx is not None:
        fields.append(("profile_idx", "<u4"))
    if col_idx is not None:
        fields.append(("col_idx", "<u2"))
    if flags is not None:
        fields.append(("quality_flag", "u1"))
    rec = np.zeros(n, dtype=fields)
    rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]
    if profile_idx is not None:
        rec["profile_idx"] = np.asarray(profile_idx).astype(np.uint32)
    if col_idx is not None:
        rec["col_idx"] = np.asarray(col_idx).astype(np.uint16)
    if flags is not None:
        rec["quality_flag"] = np.asarray(flags).astype(np.uint8)
    types = {"<f4": "float", "<u4": "uint", "<u2": "ushort", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {types[t]} {nm}" for nm, t in fields]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(rec.tobytes())


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
