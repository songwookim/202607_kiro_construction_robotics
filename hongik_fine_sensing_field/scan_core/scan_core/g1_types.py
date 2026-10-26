"""1G core 공유 타입 (G1_CORE_SPEC §8.2, WP0). 모든 WP 가 이 모듈만 공유한다 — 검토 뒤 동결.

단위: 위치 m (base frame `link0`), 치수 mm, 각 deg (이름에 단위). 심 좌표 s 는 착점 S0 에서 u0 를 따라 (S→E 순서 — 1G core 에서 S0 = 먼저
가는 착점, 사용자 2026-10-10).

교시 틀 F0 (§4.1): 원점 S0, n0 = up (1G), u0 = unit((E0 − S0) ⟂ n0), w0 = n0 × u0.
루트 선 (§4.1): r(s) = S0 + s·u0 + a(s)·w0 + b(s)·n0,  a(s) = Σ_k a_k s^k,  b(s) = Σ_k b_k s^k.
  `SeamEstimate.line_params` = (2, d+1) 배열: 0행 = (a0, a1[, a2]) 측면 (w0), 1행 = (b0, b1[, b2]) 법선 (n0); s 는 m.
  d = 1 ('line') 또는 2 ('quadratic', D10 BIC). `line_cov` = 그 배열을 행 우선으로 편 벡터의 공분산 (2(d+1), 2(d+1)).
"""
from __future__ import annotations

import io
import json
import math
import os
import tempfile
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from scan_core.specimen import GrooveSpec

__all__ = ["LEG_CLASSES", "Tolerances", "Precision", "LegObs", "SeamPrior", "EndEst", "StationEst", "SeamEstimate", "taught_frame", "jsonable",
           "atomic_write_bytes", "THETA_NAMES", "DIM_NAMES", "LINE_MODELS"]

# 템플릿 파라미터 순서 (§3.2) — LegObs.theta / cov 의 열 순서
THETA_NAMES = ("x_R", "z_R", "alpha", "G", "beta", "rf", "h_L", "h_R", "gamma_L", "gamma_R")
# 스테이션 치수 이름 (§4.6, §9) — StationEst.dims 의 키
DIM_NAMES = ("gap_mm", "depth_mm", "bevel_angle_deg", "top_opening_mm", "hi_lo_mm", "root_face_mm")
LINE_MODELS = {"line": 1, "quadratic": 2}


# ---------------------------------------------------------------- helpers

def jsonable(o):
    """numpy·dataclass 가 섞인 값 → JSON 직렬화 가능한 값 (NaN/inf 는 None)."""
    if hasattr(o, "to_dict") and callable(o.to_dict):
        return jsonable(o.to_dict())
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    return o


def atomic_write_bytes(path, data: bytes) -> None:
    """같은 디렉터리의 임시 파일에 쓰고 os.replace (원자적)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _savez_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)          # 압축 없음 (§8.5)
    return buf.getvalue()


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not (n > 0.0 and math.isfinite(n)):
        raise ValueError(f"zero / non-finite vector {v}")
    return v / n


def taught_frame(S0, E0, up) -> dict:
    """교시 틀 F0 (§4.1): {"S0", "u0", "w0", "n0"} (base frame, m). S0 = E0 (면내) 이면 ValueError (D1)."""
    S0 = np.asarray(S0, dtype=float).reshape(3)
    E0 = np.asarray(E0, dtype=float).reshape(3)
    n0 = _unit(up)
    d = E0 - S0
    u0 = _unit(d - float(d @ n0) * n0)
    w0 = np.cross(n0, u0)
    return {"S0": S0, "u0": u0, "w0": w0 / float(np.linalg.norm(w0)), "n0": n0}


# ---------------------------------------------------------------- tolerances

def _tol(qc: dict, key: str) -> Optional[float]:
    v = (qc.get(key) or {}) if isinstance(qc.get(key), dict) else {}
    t = v.get("tol")
    if t is None:
        return None
    t = float(t)
    if not (math.isfinite(t) and t > 0.0):
        raise ValueError(f"qc_targets.{key}.tol must be finite and > 0, got {t!r}")
    return t


@dataclass
class Tolerances:            # mm / deg; None = report only
    root_position_mm: float
    end_position_mm: float
    gap_mm: Optional[float]
    depth_mm: Optional[float]
    bevel_deg: Optional[float]
    opening_mm: Optional[float]
    length_mm: Optional[float]

    @classmethod
    def from_qc(cls, qc_targets: dict) -> "Tolerances":
        """작업 qc_targets 의 설계 허용치 (`{key: {tol: x}}`). 플랫폼 기본값 없음 (사용자 2026-10-07).
        root_position_mm 은 필수 (D1: 없으면 모형이 정의되지 않음 → ValueError). end_position_mm 은 없으면 root_position_mm (§7 #12)."""
        qc = dict(qc_targets or {})
        root = _tol(qc, "root_position_mm")
        if root is None:
            raise ValueError("qc_targets.root_position_mm.tol is required (G1 core D1)")
        end = _tol(qc, "end_position_mm")
        return cls(root_position_mm=root, end_position_mm=root if end is None else end,
                   gap_mm=_tol(qc, "root_gap_mm"), depth_mm=_tol(qc, "groove_depth_mm"),
                   bevel_deg=_tol(qc, "bevel_angle_deg"), opening_mm=_tol(qc, "top_opening_mm"),
                   length_mm=_tol(qc, "weld_length_mm"))

    def for_dim(self, name: str) -> Optional[float]:
        """스테이션 치수 이름 (DIM_NAMES) → 허용치 (없으면 None = 보고만)."""
        return {"gap_mm": self.gap_mm, "depth_mm": self.depth_mm, "bevel_angle_deg": self.bevel_deg,
                "top_opening_mm": self.opening_mm}.get(name)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Precision:            # 1-σ measurement repeatability the done test requires (job `precision` block, user 2026-10-11), mm
    end_position_mm: float
    root_position_mm: float
    length_mm: float        # length-type section dims (gap, depth, top opening, root face)

    @classmethod
    def from_doc(cls, block: Optional[dict]) -> "Precision":
        """작업의 `precision` 블록 (1σ, mm): {end_position_mm, root_position_mm, length_mm} — 모두 필수 (D1: 없으면 ValueError).
        설계 허용치 (qc_targets) 와 별개: 그것은 설계 대비 경고에만 쓴다 (사용자 2026-10-11)."""
        b = dict(block or {})
        vals = {}
        for key in ("end_position_mm", "root_position_mm", "length_mm"):
            v = b.get(key)
            if isinstance(v, dict):
                v = v.get("sigma")
            if v is None:
                raise ValueError(f"precision.{key} (1-sigma, mm) is required")
            v = float(v)
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"precision.{key} must be finite and > 0, got {v!r}")
            vals[key] = v
        return cls(**vals)

    def for_dim(self, name: str, spec: Optional[GrooveSpec] = None) -> Optional[float]:
        """스테이션 치수 이름 → 1σ 목표: 길이형 = length_mm; 각도형 (베벨) = 길이 목표를 그 면 자신의 길이로 나눈 각
        atan(length / L_face) [deg], L_face = (D − rf)/cos β (설계 기하 — 새 상수 없음). 그 밖 None."""
        if name in ("gap_mm", "depth_mm", "top_opening_mm", "root_face_mm"):
            return self.length_mm
        if name == "bevel_angle_deg":
            if spec is None or str(spec.kind) == "square":
                return None
            b = math.radians(float(spec.bevel_angle_deg))
            face = (float(spec.thickness_mm) - float(spec.root_face_mm)) / max(math.cos(b), 1e-9)
            return math.degrees(math.atan(self.length_mm / face)) if face > 0 else None
        return None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- one leg

# 관측 부류 (spec r2 §3.6a, 2026-10-09 지금의 틀): 드웰 grab · 트랜싯 · 패스 — 'coarse'/'fine' 은 예전 이름 (WP2 별칭: transit / pass)
LEG_CLASSES = ("dwell", "transit", "pass", "coarse", "fine")
_LEG_ARRAYS = ("t_acq_s", "T_base_opt", "v_opt_mps", "theta", "cov", "log_bf", "sigma_r_mm", "eps", "n_valid", "converged")


@dataclass
class LegObs:                # one recorded leg, struct of arrays (N profiles); saved as verification/legs/<name>/profiles.npz
    name: str
    cls: str                               # spec r2 §3.6a: 'dwell' | 'transit' | 'pass' (aliases 'coarse' ≡ transit, 'fine' ≡ pass)
    direction: int                         # +1 = travel along S->E (the prior's S0 -> E0), -1 opposite, 0 = dwell (no travel)
    t_acq_s: np.ndarray                    # (N,)
    T_base_opt: np.ndarray                 # (N,4,4) at t_acq with calib dt
    v_opt_mps: np.ndarray                  # (N,3) optical-origin velocity
    theta: np.ndarray                      # (N,10) in-plane template params (THETA_NAMES order)
    cov: np.ndarray                        # (N,10,10)
    log_bf: np.ndarray                     # (N,) λ_i
    sigma_r_mm: np.ndarray
    eps: np.ndarray
    n_valid: np.ndarray
    converged: np.ndarray
    bevel_x_sign: int                      # sensor-x side of the bevel plate in this leg
    meta: dict = field(default_factory=dict)   # speed, standoff, pitch, fps, spacing, files, timing
    # the leg's measured profiles (reconstruct.ScanResult arrays: x_mm, z_mm, valid, T_base_opt, t_acq_s) — the seam-end model
    # reads the points at the plate ends from it (seam_estimate end faces); set by load() when recon.npz sits next to profiles.npz,
    # not saved (the file is the leg's own record)
    recon_path: Optional[str] = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.cls not in LEG_CLASSES:
            raise ValueError(f"LegObs.cls must be one of {LEG_CLASSES}, got {self.cls!r}")
        if int(self.direction) not in ((1, -1, 0) if self.cls == "dwell" else (1, -1)):
            raise ValueError(f"LegObs.direction must be +1 | -1 (0 only for a dwell), got {self.direction!r}")
        self.direction = int(self.direction)
        self.bevel_x_sign = int(self.bevel_x_sign)
        n = len(np.asarray(self.t_acq_s))
        shapes = {"t_acq_s": (n,), "T_base_opt": (n, 4, 4), "v_opt_mps": (n, 3), "theta": (n, len(THETA_NAMES)),
                  "cov": (n, len(THETA_NAMES), len(THETA_NAMES)), "log_bf": (n,), "sigma_r_mm": (n,), "eps": (n,),
                  "n_valid": (n,), "converged": (n,)}
        for k, shp in shapes.items():
            a = np.asarray(getattr(self, k))
            if k == "converged":
                a = a.astype(bool)
            elif k == "n_valid":
                a = a.astype(np.int64)
            else:
                a = a.astype(float)
            if a.shape != shp:
                raise ValueError(f"LegObs.{k} shape {a.shape} != {shp}")
            setattr(self, k, a)
        self.meta = dict(self.meta or {})

    @property
    def n(self) -> int:
        return int(self.t_acq_s.shape[0])

    def save(self, path) -> None:
        """압축 없는 npz 한 파일 (원자적): 배열 + `meta_json` (name, cls, direction, bevel_x_sign, meta 를 담은 JSON 문자열)."""
        head = {"schema": "g1core.legobs/1", "name": self.name, "cls": self.cls, "direction": self.direction,
                "bevel_x_sign": self.bevel_x_sign, "theta_names": list(THETA_NAMES), "meta": self.meta}
        arrays = {k: getattr(self, k) for k in _LEG_ARRAYS}
        arrays["meta_json"] = np.array(json.dumps(jsonable(head), ensure_ascii=False))
        atomic_write_bytes(path, _savez_bytes(**arrays))

    @classmethod
    def load(cls, path) -> "LegObs":
        with np.load(path, allow_pickle=False) as z:
            head = json.loads(str(z["meta_json"][()]))
            arrays = {k: np.array(z[k]) for k in _LEG_ARRAYS}
        if tuple(head.get("theta_names", THETA_NAMES)) != THETA_NAMES:
            raise ValueError(f"{path}: theta order {head.get('theta_names')} != {THETA_NAMES}")
        rp = Path(path).with_name("recon.npz")
        return cls(name=head["name"], cls=head["cls"], direction=head["direction"], bevel_x_sign=head["bevel_x_sign"],
                   meta=head.get("meta") or {}, recon_path=str(rp) if rp.is_file() else None, **arrays)


# ---------------------------------------------------------------- prior

@dataclass
class SeamPrior:
    S0: np.ndarray                         # taught rough points (taught order), base frame [m]
    E0: np.ndarray
    up: np.ndarray
    spec: GrooveSpec
    tol: Tolerances
    k: float
    dt_s: float
    dt_sigma_s: float
    rep_sigma_m: float
    he_sigma_m: float
    sigma_eps: float
    station_step_m: float
    station_window_m: float
    bevel_sign_w: Optional[int] = None     # +1 = bevel plate on +w (left looking S->E), -1 right; None = decide (D10)
    # calibration σ of the laser-plane rotation about the laser-line axis (sensor pitch) [rad]: an end-face slant measured with one
    # tool orientation is confounded with it (seam_estimate end faces); NaN = the calibration does not state it
    laser_pitch_sigma_rad: float = float("nan")
    # the sensor's edge response at a depth step (laser-sheet spread σ_L, triangulation angle θ) and their σ — calibration constants
    # (calib g1: edge_response: {sigma_l_m, sigma_l_log_sd, theta_tri_deg, theta_tri_sigma_deg, source}); None = not calibrated
    # (the end model then estimates them online from the scan)
    edge_response: Optional[dict] = None
    precision: Optional[Precision] = None  # 1-σ targets of the done test (job `precision`); None = no target → nothing is done
    # calibrated priors of the estimator's variance components (calib g1: noise_priors — end-reading crossing / reading scatter, leg
    # offset scatter w / n; scan_core.noise_priors); None = not calibrated (the run's own small-sample treatment)
    noise_priors: Optional[dict] = None

    def __post_init__(self):
        self.S0 = np.asarray(self.S0, dtype=float).reshape(3)
        self.E0 = np.asarray(self.E0, dtype=float).reshape(3)
        self.up = np.asarray(self.up, dtype=float).reshape(3)
        if self.bevel_sign_w is not None and int(self.bevel_sign_w) not in (1, -1):
            raise ValueError(f"bevel_sign_w must be +1 | -1 | None, got {self.bevel_sign_w!r}")

    def frame0(self) -> dict:
        """교시 틀 F0 (§4.1)."""
        return taught_frame(self.S0, self.E0, self.up)

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["spec"] = self.spec.to_dict()
        d["tol"] = self.tol.to_dict()
        d["precision"] = self.precision.to_dict() if self.precision is not None else None
        return jsonable(d)

    @classmethod
    def from_dict(cls, d: dict) -> "SeamPrior":
        """to_dict (run_log.json 'prior') → SeamPrior (오프라인 재생성; JSON null 인 float = NaN)."""
        d = dict(d)
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kw["spec"] = GrooveSpec.from_dict(d["spec"])
        kw["tol"] = Tolerances(**{f.name: d["tol"].get(f.name) for f in fields(Tolerances)})
        kw["precision"] = Precision(**d["precision"]) if d.get("precision") else None
        for f in fields(cls):
            if f.name in kw and kw[f.name] is None and f.type in ("float", float):
                kw[f.name] = float("nan")
        return cls(**kw)


# ---------------------------------------------------------------- estimate

@dataclass
class EndEst:
    s_m: float
    sigma_m: float
    point_m: np.ndarray
    two_sided: bool
    n_legs: int
    readings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return jsonable(asdict(self))


@dataclass
class StationEst:
    s_m: float
    root_m: np.ndarray
    root_sigma_m: tuple                    # (lateral, normal)
    u: np.ndarray
    n: np.ndarray
    w: np.ndarray
    dims: dict          # DIM_NAMES -> {"value", "sigma", "sigma_sys", "n"}
    faces: dict         # cross_sections.FACE_NAMES -> {"normal", "point_m", "line_wn_mm", "sigma_deg", "n"}
    n_profiles: int

    def to_dict(self) -> dict:
        return jsonable(asdict(self))


@dataclass
class SeamEstimate:
    groove_found: bool
    log_bf_run: float
    line_model: str                        # 'line' | 'quadratic'
    line_params: np.ndarray                # (2, d+1): rows (a_k) lateral w0, (b_k) normal n0; module docstring
    line_cov: np.ndarray                   # (2(d+1), 2(d+1)) of line_params.ravel()
    u: np.ndarray                          # û (fitted line direction, unit)
    n_plate: np.ndarray
    plate: dict                            # n_left, n_right, roll_deg, misalignment_deg, sigmas
    ends: dict                             # 'S'|'E' -> EndEst
    dt_s: float
    dt_sigma_s: float
    class_offset_m: dict
    leg_eta_m: dict
    sigma_zeta_m: dict
    tau_end_m: float
    bevel_sign_w: int
    stations: list                         # [StationEst]
    ratios: dict
    done: bool
    binding: list
    unresolved: list
    frame0: Optional[dict] = None          # taught_frame(S0, E0, up) — required by r(s) (not in the §8.2 list; WP0 addition)

    def __post_init__(self):
        if self.line_model not in LINE_MODELS:
            raise ValueError(f"line_model must be one of {tuple(LINE_MODELS)}, got {self.line_model!r}")
        d = LINE_MODELS[self.line_model]
        self.line_params = np.asarray(self.line_params, dtype=float).reshape(2, d + 1)
        self.line_cov = np.asarray(self.line_cov, dtype=float).reshape(2 * (d + 1), 2 * (d + 1))
        self.u = np.asarray(self.u, dtype=float).reshape(3)
        self.n_plate = np.asarray(self.n_plate, dtype=float).reshape(3)
        if self.frame0 is not None:
            self.frame0 = {k: np.asarray(self.frame0[k], dtype=float).reshape(3) for k in ("S0", "u0", "w0", "n0")}

    def _f0(self) -> dict:
        if self.frame0 is None:
            raise ValueError("SeamEstimate.frame0 is not set (use SeamPrior.frame0())")
        return self.frame0

    def _basis(self, s_m, deriv: int = 0) -> np.ndarray:
        s = np.asarray(s_m, dtype=float)
        d = LINE_MODELS[self.line_model]
        k = np.arange(d + 1)
        if deriv == 0:
            return s[..., None] ** k
        coef = k.astype(float)
        return np.where(k >= 1, coef * s[..., None] ** np.maximum(k - 1, 0), 0.0)

    def r(self, s_m) -> np.ndarray:
        """루트 선 점 r(s) (base, m). 스칼라 → (3,), 배열 (...) → (..., 3)."""
        f = self._f0()
        g = self._basis(s_m)
        a = g @ self.line_params[0]
        b = g @ self.line_params[1]
        s = np.asarray(s_m, dtype=float)
        return f["S0"] + s[..., None] * f["u0"] + a[..., None] * f["w0"] + b[..., None] * f["n0"]

    def dr_ds(self, s_m) -> np.ndarray:
        """dr/ds (base) — 교시 축 좌표 s 에 대한 루트 선의 도함수 (정규화 전; ‖·‖ = 심 길이 / s 길이 ≥ 1)."""
        f = self._f0()
        g = self._basis(s_m, deriv=1)
        da = g @ self.line_params[0]
        db = g @ self.line_params[1]
        return f["u0"] + np.asarray(da)[..., None] * f["w0"] + np.asarray(db)[..., None] * f["n0"]

    def tangent(self, s_m) -> np.ndarray:
        """단위 접선 dr/ds / ‖·‖ (선 모형이면 어디서나 û)."""
        t = self.dr_ds(s_m)
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def r_sigma(self, s_m) -> tuple:
        """r(s) 의 (측면 w0, 법선 n0) 표준편차 [m] (line_cov 에서)."""
        g = self._basis(s_m)
        m = g.shape[-1]
        Ca = self.line_cov[:m, :m]
        Cb = self.line_cov[m:, m:]
        va = np.einsum("...i,ij,...j->...", g, Ca, g)
        vb = np.einsum("...i,ij,...j->...", g, Cb, g)
        return np.sqrt(np.maximum(va, 0.0)), np.sqrt(np.maximum(vb, 0.0))

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["schema"] = "g1core.estimate/1"
        return jsonable(d)

    @classmethod
    def from_dict(cls, d: dict) -> "SeamEstimate":
        """to_dict (verification/estimates/est_<n>.json) → SeamEstimate (오프라인 재생성; JSON null 인 수 = NaN)."""
        def fn(v):
            return float("nan") if v is None else float(v)

        def arr(v):
            return np.array([np.nan if x is None else x for x in np.ravel(np.asarray(v, dtype=object))], dtype=float).reshape(
                np.shape(np.asarray(v, dtype=object)))
        d = dict(d)
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kw["line_params"], kw["line_cov"] = arr(d["line_params"]), arr(d["line_cov"])
        kw["log_bf_run"], kw["dt_s"], kw["dt_sigma_s"], kw["tau_end_m"] = (fn(d.get(k)) for k in ("log_bf_run", "dt_s", "dt_sigma_s",
                                                                                                     "tau_end_m"))
        kw["ends"] = {X: EndEst(s_m=fn(e["s_m"]), sigma_m=fn(e["sigma_m"]), point_m=arr(e["point_m"]), two_sided=bool(e["two_sided"]),
                                n_legs=int(e["n_legs"]), readings=list(e.get("readings") or []))
                      for X, e in (d.get("ends") or {}).items()}
        kw["stations"] = [StationEst(s_m=fn(st["s_m"]), root_m=arr(st["root_m"]), root_sigma_m=tuple(fn(v) for v in st["root_sigma_m"]),
                                     u=arr(st["u"]), n=arr(st["n"]), w=arr(st["w"]), dims=dict(st.get("dims") or {}),
                                     faces=dict(st.get("faces") or {}), n_profiles=int(st["n_profiles"]))
                          for st in d.get("stations") or []]
        return cls(**kw)
