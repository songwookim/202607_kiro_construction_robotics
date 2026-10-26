"""패스별 스캔 품질 평가 · 자세 기록 · 구간별 최선 자료 병합 · 자세–품질 데이터셋 (적응형 스캔, 사용자 2026-10-06).

auto-teach 는 패스마다 '어디를' (심 직선·끝·판 평면) 고쳐 왔지만 '어떻게' (입사 각도·standoff·자세) 는 실행 처음에 정했다. g3_11 (입사 37°)
은 프로파일 절반에서 베벨면을 못 봤는데도 다음 패스가 자세를 바꾸지 않았다. 이 모듈:
  * profile_quality: 프로파일별 '좋은 단면' (그루브·개구·깊이·갭 판독, 설계에 베벨이 있으면 베벨면 검출)
  * quality_bins: 심 직선 좌표 구간별 좋은 단면 비율
  * pass_posture: 패스 job + 캘리브 → 자세 (레이저 평면과 판 법선 사이 입사 각도, standoff, 피치, 토치 위/아래)
  * posture_weight: 입사 각도별 특징 신뢰 가중 (실측: 0–20° 좋음, 35–40° 루트갭 +1.3 mm 편향, 37° 베벨 검출 49 %)
  * merge_feature_arrays: 여러 패스의 프로파일 특징을 구간마다 점수 (좋은 비율 × 자세 가중) 가 가장 높은 패스 것으로 합친다 — 좌표는
    S→E 계약 (scan_core.seam_frame: 기준 패스의 관측 끝 S, E 의 base 점; 축 = unit(E − S), s = S 에서 잰 거리)
  * append_dataset / phi_prior / predicted_score: 실행마다 (자세, 품질) 를 jsonl 로 쌓고, φ (레이저 평면 기울기, 부호 있음) 별
    예상 품질로 다음 자세를 고른다 (정적 사전값 → 데이터셋 → 이번 실행 측정 순으로 덮어씀)
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from scan_core import measure as _ms
from scan_core.frames import unit
from scan_core.seam_frame import seam_frame
from scan_core.weld_plan import WeldProcess

BIN_M = 0.020                  # 기본 구간 (호출자가 bin_for 로 유도한 값을 넘긴다 — 4d P6)
GOOD_FRAC_MIN = _ms.DESIGN_HALF   # 구간·자세가 '좋다' = 좋은 단면이 절반 이상 (원칙 2; 예전 0.6)
MIN_BIN_PROFILES = 3           # 기본 (호출자가 bin_need 로 유도한 값을 넘긴다)


def spacing_of(feats: dict) -> float:
    """프로파일 간격 [m] = 루트 점의 심 방향 간격 중앙값 (없으면 nan)."""
    P = np.asarray(feats.get("root_polyline_base_m", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if P.shape[0] < 2:
        return float("nan")
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    d = d[d > 0]
    return float(np.median(d)) if d.size else float("nan")


def bin_need(spacing_m: float, process: Optional[WeldProcess] = None) -> int:
    """구간 점수를 매기는 최소 프로파일 수 = 단면 하나의 증거 (evidence_count(min(min_profiles, 창/간격)), station_coverage 와 같은 규칙)."""
    proc = process or WeldProcess()
    w = float(proc.station_window_mm) * 1e-3
    per = w / float(spacing_m) if (np.isfinite(spacing_m) and spacing_m > 0) else float(proc.min_profiles)
    return _ms.evidence_count(min(float(proc.min_profiles), per))


def bin_for(spacing_m: float, process: Optional[WeldProcess] = None) -> float:
    """품질 구간 폭 [m] = 용접 경로 단면 창 하나, 간격이 넓어 그 창에 증거가 모자라면 그만큼 넓힘 (4d P6; 예전 20 mm)."""
    proc = process or WeldProcess()
    w = float(proc.station_window_mm) * 1e-3
    if np.isfinite(spacing_m) and spacing_m > 0:
        return max(w, bin_need(spacing_m, proc) * float(spacing_m))
    return w
DATASET_DEFAULT = Path.home() / "scan_data" / "scan_quality_dataset.jsonl"

SEED_PATH = Path(__file__).resolve().parent / "data" / "scan_quality_seed.jsonl"


def _load_seed(path=SEED_PATH) -> dict:
    """자세–품질 모형의 바닥 층 (사용자 2026-10-07: 품질 사전값은 코드가 아니라 데이터 파일 — 4d P6). jsonl: meta 한 줄 + φ 사전값
    (phi_prior) + 입사 가중 (incidence_weight) + 상수 (constant). SCAN_QUALITY_DATASET=off 여도 읽는다 (데이터셋 건수에 세지 않는다)."""
    out = {"meta": {}, "phi_prior": [], "incidence_weight": [], "constant": {}}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            k = r.get("kind")
            if k == "meta":
                out["meta"] = r
            elif k == "phi_prior":
                out["phi_prior"].append((float(r["phi_deg"]), float(r["frac_good"])))
            elif k == "incidence_weight":
                out["incidence_weight"].append((float(r["max_incidence_deg"]), float(r["weight"])))
            elif k == "constant":
                out["constant"][str(r["name"])] = float(r["value"])
    out["phi_prior"] = tuple(sorted(out["phi_prior"]))
    out["incidence_weight"] = tuple(sorted(out["incidence_weight"]))
    return out


_SEED = _load_seed()
# 입사 각도별 특징 신뢰 가중 (씨앗 파일 incidence_weight: 실측, 같은 시편 2026-10-05/06). 반사 (정반사 쪽 φ) 는 phi_prior 가 따로 본다.
_WEIGHT_TABLE = _SEED["incidence_weight"]


def posture_weight(incidence_deg: float) -> float:
    a = abs(float(incidence_deg)) if np.isfinite(incidence_deg) else 90.0
    for lim, w in _WEIGHT_TABLE:
        if a <= lim:
            return w
    return _WEIGHT_TABLE[-1][1] if _WEIGHT_TABLE else 1.0


def _running_median(v: np.ndarray, win: int, need: int) -> np.ndarray:
    out = np.full(v.size, np.nan)
    for i in range(v.size):
        w = v[max(0, i - win): i + win + 1]
        w = w[np.isfinite(w)]
        if w.size >= need:
            out[i] = float(np.median(w))
    return out


def profile_quality(feats: dict, spec=None, *, qc_targets: Optional[dict] = None, spacing_m: Optional[float] = None,
                    process: Optional[WeldProcess] = None, z_mm=None, k: Optional[float] = None) -> dict:
    """프로파일별 판독 항목 (bool 배열) 과 '좋은 단면' (all). feats = autoteach._features_arrays 출력.
    판독 (값이 있음) 만으로는 모자란다 — 2026-10-06 g3_12: 입사 20° 이동 스캔은 반사로 단면이 무너졌는데 (개구 20, 갭 15 mm) 판독
    기준으로는 74 % '좋음' 이었다. 그래서 '타당' 과 '일관' 을 더한다 (4d P6, 설정 상수 없이):
      타당 = 개구·깊이·갭 (읽혔으면) 이 설계 ± 절반 (design_match — 게이트의 식별 규칙과 같다; 예전 ± 25 %·2.5 mm)
      일관 = 앞뒤 단면 창 절반 (± station_window/2, 간격으로 프로파일 수) 의 이동 중앙값에서 개구·깊이가 작업 허용치 안 (하한 k·양자화;
             허용치가 없으면 k·σ, σ = 이동 중앙값 둘레 강건 σ) — 이동 중앙값의 최소 개수 = 단면 하나의 증거 (bin_need)."""
    ok = np.asarray(feats.get("ok", []), dtype=bool)
    M = ok.size
    kk = float(_ms.K_DEFAULT if k is None else k)
    proc = process or WeldProcess()

    def fin(k_):
        v = feats.get(k_)
        return np.isfinite(np.asarray(v, dtype=float)) if v is not None and np.asarray(v).shape[:1] == (M,) else np.zeros(M, dtype=bool)

    bevel_designed = spec is not None and getattr(spec, "kind", "square") != "square" and float(getattr(spec, "bevel_angle_deg", 0.0)) > 0
    q = {"groove": ok.copy(), "opening": ok & fin("top_opening_mm"), "depth": ok & fin("depth_mm"), "gap": ok & fin("root_gap_mm"),
         "bevel": (ok & fin("bevel_angle_deg")) if bevel_designed else ok.copy()}
    op = np.asarray(feats.get("top_opening_mm", np.full(M, np.nan)), dtype=float)
    dp = np.asarray(feats.get("depth_mm", np.full(M, np.nan)), dtype=float)
    gp = np.asarray(feats.get("root_gap_mm", np.full(M, np.nan)), dtype=float)
    if spec is not None and op.shape == (M,) and dp.shape == (M,):
        o0, d0, g0 = float(spec.top_opening_mm), float(spec.groove_depth_mm), float(spec.root_gap_mm)
        with np.errstate(invalid="ignore"):
            pl = ((np.abs(op - o0) <= _ms.design_band(o0)) & (np.abs(dp - d0) <= _ms.design_band(d0))
                  & (~np.isfinite(gp) | (np.abs(gp - g0) <= _ms.design_band(g0)) | (g0 <= 0.0)))
        q["plausible"] = ok & pl
    else:
        q["plausible"] = ok.copy()
    sp = float(spacing_m) if (spacing_m is not None and np.isfinite(spacing_m) and spacing_m > 0) else spacing_of(feats)
    win = max(1, int(round(0.5 * float(proc.station_window_mm) * 1e-3 / sp))) if (np.isfinite(sp) and sp > 0) else int(proc.min_profiles)
    need = bin_need(sp, proc)
    if op.shape == (M,) and dp.shape == (M,) and M:
        floor = kk * _ms.quant_sigma_mm(z_mm)
        cons = ok.copy()
        for v, key in ((op, "top_opening_mm"), (dp, "groove_depth_mm")):
            mv = _running_median(np.where(ok, v, np.nan), win, need)
            tol = (qc_targets or {}).get(key, {}).get("tol") if isinstance(qc_targets, dict) else None
            if tol is not None and np.isfinite(float(tol)):
                t_ = max(float(tol), floor)
            else:
                r_ = (v - mv)[ok & np.isfinite(v) & np.isfinite(mv)]
                t_ = kk * max(_ms.robust_sigma(r_) if r_.size >= 3 else float("nan"), _ms.quant_sigma_mm(z_mm))
                if not np.isfinite(t_):
                    t_ = floor
            with np.errstate(invalid="ignore"):
                cons &= np.abs(v - mv) <= t_
        q["consistent"] = cons
    else:
        q["consistent"] = ok.copy()
    q["good"] = q["groove"] & q["opening"] & q["depth"] & q["gap"] & q["bevel"] & q["plausible"] & q["consistent"]
    return q


def line_coordinate(feats: dict, c, u) -> np.ndarray:
    """프로파일 루트 점을 기준 직선 (c, u) 에 투영한 좌표 (M,) [m] — 패스마다 정련 직선이 조금 달라도 같은 좌표로 비교."""
    P = np.asarray(feats.get("root_polyline_base_m", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    s = np.full(P.shape[0], np.nan)
    fin = np.all(np.isfinite(P), axis=1)
    s[fin] = (P[fin] - np.asarray(c, dtype=float)) @ unit(u)
    return s


def quality_bins(s: np.ndarray, q: dict, s_lo: float, s_hi: float, bin_m: float = BIN_M) -> list:
    """[s_lo, s_hi] 를 bin_m 구간으로 나눠 구간별 프로파일 수·좋은 비율·항목별 비율."""
    s = np.asarray(s, dtype=float)
    out = []
    if not (np.isfinite(s_lo) and np.isfinite(s_hi)) or s_hi <= s_lo:
        return out
    edges = np.arange(s_lo, s_hi + 1e-9, bin_m)
    if edges.size < 2 or edges[-1] < s_hi - 1e-9:
        edges = np.append(edges, s_hi)
    for a, b in zip(edges[:-1], edges[1:]):
        m = np.isfinite(s) & (s >= a) & (s < b)
        n = int(m.sum())
        row = {"s_from_m": float(a), "s_to_m": float(b), "n": n}
        for k, v in q.items():
            row[f"frac_{k}"] = float(np.mean(v[m])) if n else 0.0
        out.append(row)
    return out


def pass_posture(job, calib, plate_normal=None) -> dict:
    """패스 job → 자세: 입사 = 레이저 평면과 판 법선 사이 각 (깊이 축 보정과 같은 정의), φ = 판 법선에서 광축으로 레이저 선 축 X 둘레
    부호 있는 각, job 의 입사 기울기·피치·standoff·진행 부호, 토치가 레이저 위인지 (3G; 1G 는 None)."""
    from scan_core.path_gen import sensor_frames, tool_keypoints
    fr = sensor_frames(job, calib)
    T = fr.T_base_flange(0.0)
    Y = (T @ calib.T_flange_opt)[:3, 1]
    n = unit(plate_normal if plate_normal is not None else job.normal_unit)
    inc = math.degrees(math.asin(min(1.0, abs(float(Y @ n)))))
    up = job.up_unit
    Tbe = T @ calib.T_flange_eoat
    To = T @ calib.T_flange_opt
    X, Z = To[:3, 0], To[:3, 2]
    v = -n - X * float(-n @ X)                                   # 판 법선 (센서 쪽 반대) 의 레이저 평면 성분
    phi = math.degrees(math.atan2(float(X @ np.cross(v, Z)), float(v @ Z))) if np.linalg.norm(v) > 1e-9 else float("nan")
    vertical = job.resolved_orientation() == "vertical_3G"
    return {"incidence_deg": inc, "phi_deg": phi, "tilt_deg": float(job.incidence_tilt_deg), "pitch_deg": float(job.pitch_deg),
            "standoff_m": float(job.standoff_m), "progression_sign": int(job.progression_sign),
            "torch_above": bool(float(Tbe[:3, 0] @ up) < 0.0) if vertical else None}   # 3G: EOAT +x (카메라 끝) 가 아래 = 토치 (−x) 가 위


def bin_scores(bins: list, posture: dict, need: int = MIN_BIN_PROFILES) -> list:
    w = posture_weight(posture.get("incidence_deg", float("nan")))
    return [float(b.get("frac_good", 0.0)) * w if b.get("n", 0) >= int(need) else 0.0 for b in bins]


def merge_feature_arrays(entries: list, S, E, *, n=None, bin_m: float = BIN_M, need: int = MIN_BIN_PROFILES) -> tuple:
    """entries = [{"k", "feats", "q", "posture"}] (패스 순서). S→E 계약 좌표 (기준 패스의 관측 끝 S, E — base 점; 축 u = unit(E − S),
    n 을 주면 판 법선에 직교화; s = S 에서 잰 거리, 구간은 S 에서 E 까지) 의 구간마다 점수 (좋은 비율 × 자세 가중) 가 가장 높은 패스의
    프로파일을 고른다 (같으면 나중 패스). 패스 진행 방향 (정련 직선 u 의 부호) 과 무관하다 (2026-10-09: 기준 패스가 E→S 이면 그 u 로
    병합한 계획이 거울상이 됐다). 반환 (병합 feats — 배열 키는 프로파일 축으로 이어 붙이고 s_line_m 은 S 에서 잰 거리, line_c_base_m = S,
    u_base = S→E 축, s_a_m/s_b_m = 0/L; 구간별 선택 [{s_from_m, s_to_m, pass, score, scores}] — s 는 S 에서)."""
    if not entries:
        return None, []
    fr = seam_frame(S, E, n)
    c, u, s_lo, s_hi = fr.S, fr.u, 0.0, fr.length_m
    per = []
    for e in entries:
        s = line_coordinate(e["feats"], c, u)
        b = quality_bins(s, e["q"], s_lo, s_hi, bin_m)
        per.append((e, s, b, bin_scores(b, e["posture"], need)))
    nb = len(per[0][2])
    choice = []
    take = [np.zeros(per_[1].size, dtype=bool) for per_ in per]
    for j in range(nb):
        sc = [p[3][j] if j < len(p[3]) else 0.0 for p in per]
        ns = [int(p[2][j].get("n", 0)) if j < len(p[2]) else 0 for p in per]
        if max(sc) > 0.0:
            best = max(range(len(per)), key=lambda i: (sc[i], i))
        else:                                                     # 어느 패스도 점수 없음 (끝 구간 등): 프로파일이 가장 많은 패스,
            best = max(range(len(per)), key=lambda i: (ns[i], bool(per[i][0].get("full", True)), i))   # 같으면 전체 패스 (재스캔 아님)
        a, b_ = per[0][2][j]["s_from_m"], per[0][2][j]["s_to_m"]
        choice.append({"s_from_m": a, "s_to_m": b_, "pass": int(per[best][0]["k"]), "score": float(sc[best]),
                       "scores": {int(p[0]["k"]): float(x) for p, x in zip(per, sc)}})

        def in_bin(sv):
            return np.isfinite(sv) & (sv >= a) & ((sv < b_) | ((j == nb - 1) & (sv <= b_)))
        s_b = per[best][1]
        sel = in_bin(s_b)
        take[best] |= sel
        # 고른 패스가 구간 일부만 덮으면 (패스가 구간 안에서 끝남 — 3G 주 패스가 테이블 앞에서 멈추는 곳) 나머지 부분은 다음 순위 패스로
        # 채운다: 덮인 범위 [lo, hi] 밖의 프로파일만 (구간 하나 = 패스 하나의 원칙은 덮인 부분에서 그대로)
        okb = np.asarray(per[best][0]["feats"].get("ok", np.zeros(s_b.size, bool)), dtype=bool)
        cov_s = s_b[sel & okb] if okb.shape == s_b.shape else s_b[sel]
        lo, hi = (float(cov_s.min()), float(cov_s.max())) if cov_s.size else (float("inf"), float("-inf"))
        for i in sorted((i_ for i_ in range(len(per)) if i_ != best), key=lambda i_: (-sc[i_], -ns[i_])):
            s_o = per[i][1]
            ext = in_bin(s_o) & ((s_o < lo) | (s_o > hi))
            if ext.any():
                take[i] |= ext
                oko = np.asarray(per[i][0]["feats"].get("ok", np.zeros(s_o.size, bool)), dtype=bool)
                v_ = s_o[ext & oko] if oko.shape == s_o.shape else s_o[ext]
                if v_.size:
                    lo, hi = min(lo, float(v_.min())), max(hi, float(v_.max()))
    merged: dict = {}
    order_s = []
    for (e, s, _b, _sc), t in zip(per, take):
        order_s.append(s[t])
    s_all = np.concatenate(order_s) if order_s else np.zeros(0)
    order = np.argsort(s_all, kind="stable")
    keys = set()
    for e, s, _b, _sc in per:
        M = s.size
        keys |= {k for k, v in e["feats"].items() if np.asarray(v).shape[:1] == (M,) and M > 1}
    for k in keys:
        parts = []
        for (e, s, _b, _sc), t in zip(per, take):
            v = e["feats"].get(k)
            if v is None or np.asarray(v).shape[:1] != (s.size,):
                parts = None
                break
            parts.append(np.asarray(v)[t])
        if parts is not None:
            merged[k] = np.concatenate(parts)[order]
    merged["s_line_m"] = s_all[order]
    merged["source_pass"] = np.concatenate([np.full(int(t.sum()), int(e["k"])) for (e, _s, _b, _sc), t in zip(per, take)])[order]
    last = entries[-1]["feats"]
    if "n_base" in last:
        merged["n_base"] = last["n_base"]
    merged["line_c_base_m"] = np.asarray(c, dtype=float)
    merged["u_base"] = unit(u)
    merged["s_a_m"], merged["s_b_m"] = np.array(float(s_lo)), np.array(float(s_hi))
    return merged, choice


def merged_medians(merged: dict) -> dict:
    """병합 feats 의 특징 중앙값 (ok 프로파일)."""
    ok = np.asarray(merged.get("ok", []), dtype=bool)
    out = {}
    for k in ("top_opening_mm", "root_gap_mm", "bevel_angle_deg", "depth_mm", "hi_lo_mm"):
        v = np.asarray(merged.get(k, []), dtype=float)
        if v.shape[:1] != ok.shape:
            continue
        v = v[ok & np.isfinite(v)]
        out[k] = float(np.median(v)) if v.size else float("nan")
    return out


QUALITY_VERSION = 3            # 3 (2026-10-08, 4d P6): 타당 = 설계 ± 절반, 일관 = 단면 창·작업 허용치 (2: 고정 ± 25 %·± 10 프로파일·1 mm;
                               # 1 = 판독만 — g3_12 20° 붕괴를 74 % 로 셌다). 낮은 버전 기록은 사전값에 쓰지 않는다


def dataset_path(path=None) -> Optional[Path]:
    """데이터셋 경로: 인자 → 환경 변수 SCAN_QUALITY_DATASET ('off' = 쓰지 않음; 시험 conftest 가 임시 경로로 둔다) → 기본."""
    import os
    if path:
        return Path(path)
    env = os.environ.get("SCAN_QUALITY_DATASET", "").strip()
    if env.lower() in ("off", "none", "0"):
        return None
    return Path(env) if env else DATASET_DEFAULT


def append_dataset(record: dict, path=None) -> Optional[str]:
    """(자세, 품질) 레코드를 jsonl 에 덧붙인다 (quality_version 포함). 실패해도 실행을 막지 않는다 (경로 반환 또는 None)."""
    p = dataset_path(path)
    if p is None:
        return None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "quality_version": QUALITY_VERSION, **record}
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        return str(p)
    except Exception:  # noqa: BLE001
        return None


# 센서 프레임 φ (pass_posture: 판 법선에서 광축으로, 레이저 선 축 X 둘레 부호) 별 실측 사전값 — 이동 스캔의 좋은 단면 비율 (씨앗 파일
# phi_prior; 같은 시편 2026-10-05/06). 이 정의에서 3G 토치 위는 φ = +입사 (+피치), 토치를 광축 둘레로 뒤집으면 (토치 아래) φ = −입사;
# 1G 피치 −5° 는 φ −5°. 반사 문제는 모두 + 쪽 (정반사 ≈ +12°; types.SPECULAR_TILT_DEG −12° 는 job 기울기 부호 규약). 데이터셋·이번
# 실행의 측정이 있으면 그 칸이 이 값을 대신한다.
STATIC_PHI_PRIOR = _SEED["phi_prior"]
PHI_UNKNOWN = _ms.DESIGN_HALF                          # 근거 없는 φ 의 예상값 = 중간점 (원칙 2; 4d P6, 예전 0.7·NaN 0.5 둘)
POSTURE_GRID_DEG = 5.0                                  # 자세 후보 격자 (수치 해상도)
PHI_WINDOW_DEG = 0.5 * POSTURE_GRID_DEG                 # φ 모형 칸 폭 (± 1.25°) — 같은 자세 판정은 ± 격자/2 (autoteach)


def _dataset_phi_points(position: str, path=None, window_deg: float = PHI_WINDOW_DEG) -> dict:
    """데이터셋 (quality_version ≥ 2, 같은 자세 위치) 의 φ 칸별 좋은 비율 평균 → {칸: (평균, 건수)}."""
    p = dataset_path(path)
    out: dict = {}
    if p is None:
        return out
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return out
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if str(r.get("position")) != str(position) or int(r.get("quality_version", 1)) < QUALITY_VERSION:
            continue
        ph = float((r.get("posture") or {}).get("phi_deg", float("nan")))
        g = float((r.get("quality") or {}).get("frac_good", float("nan")))
        if np.isfinite(ph) and np.isfinite(g):
            key = round(ph / window_deg) * window_deg
            s_, n_ = out.get(key, (0.0, 0))
            out[key] = (s_ + g, n_ + 1)
    return {k: (s_ / n_, n_) for k, (s_, n_) in out.items()}


def phi_model_points(position: str, path=None, measured=None, window_deg: float = PHI_WINDOW_DEG) -> list:
    """φ 예상 모형의 점 [(φ, 값, 근거)]: 정적 사전값 위에 데이터셋 칸 평균, 그 위에 이번 실행 측정 (measured = [(φ, 좋은 비율)]) —
    새 근거가 같은 칸 (칸 중심 ± window/2) 의 옛 점을 대신한다 (이웃 칸은 남는다: +40° 측정이 +37.5° 의 g3_11 42 % 를 지우지 않게)."""
    half = 0.5 * window_deg + 1e-9
    pts = [(float(ph), float(g), "static") for ph, g in STATIC_PHI_PRIOR]
    for key, (g, n_) in sorted(_dataset_phi_points(position, path, window_deg).items()):
        pts = [q for q in pts if abs(q[0] - key) > half] + [(float(key), float(g), f"dataset n={n_}")]
    run: dict = {}
    for ph, g in (measured or []):
        if np.isfinite(ph) and np.isfinite(g):
            key = round(float(ph) / window_deg) * window_deg
            run.setdefault(key, []).append(float(g))
    for key, gs in sorted(run.items()):
        pts = [q for q in pts if abs(q[0] - key) > half] + [(float(key), float(np.mean(gs)), f"this run n={len(gs)}")]
    return sorted(pts)


def phi_prior(position: str, phi_deg: float, path=None, measured=None, window_deg: float = PHI_WINDOW_DEG) -> tuple:
    """φ 의 예상 좋은 단면 비율과 근거. 같은 칸에 점이 있으면 그 값; 격자 한 칸 안으로 이웃한 두 점 사이만 선형 보간; 그 밖은 PHI_UNKNOWN.
    같은 칸 (± window/2) 에 점이 있으면 그 값 (여럿이면 가장 낮은 값)."""
    if not np.isfinite(phi_deg):
        return PHI_UNKNOWN, "unknown"
    from scan_core.types import SPECULAR_TILT_BAND_DEG, SPECULAR_TILT_DEG
    if abs(float(phi_deg) - (-float(SPECULAR_TILT_DEG))) < float(SPECULAR_TILT_BAND_DEG):
        return 0.0, "specular band (sensor model: saturated / lost points)"   # 자료와 무관 (4d P6; 예전 손으로 넣은 +12° 점)
    pts = phi_model_points(position, path, measured, window_deg)
    if not pts:
        return PHI_UNKNOWN, "default"
    near = [q for q in pts if abs(q[0] - phi_deg) <= 0.5 * window_deg + 1e-9]
    if near:
        q = min(near, key=lambda q_: q_[1])
        return q[1], f"{q[2]} at {q[0]:+.1f} deg"
    # 측정된 두 칸이 격자 한 칸 안으로 이웃할 때만 보간; 그 밖 (자료 사이의 큰 틈·자료 범위 밖) 은 모른다 (중간점) — 측정한 적 없는
    # 자세로 옮겨 가지 않게 (4d P6; 예전 선형 보간·20° 감쇠)
    lo = [q_ for q_ in pts if q_[0] < phi_deg]
    hi = [q_ for q_ in pts if q_[0] > phi_deg]
    if lo and hi:
        a, b = lo[-1], hi[0]
        if (b[0] - a[0]) <= POSTURE_GRID_DEG + 1e-9:
            f = (phi_deg - a[0]) / (b[0] - a[0])
            return a[1] + (b[1] - a[1]) * f, f"interpolated {a[0]:+.1f}..{b[0]:+.1f} deg"
    return PHI_UNKNOWN, "no measurement near this phi (design midpoint)"


def predicted_score(posture: dict, position: str, path=None, measured=None) -> tuple:
    """자세 → (예상 점수 = 입사 가중 × φ 예상 좋은 비율, φ 예상값, 근거)."""
    g, why = phi_prior(position, float(posture.get("phi_deg", float("nan"))), path, measured)
    return posture_weight(float(posture.get("incidence_deg", float("nan")))) * g, g, why
