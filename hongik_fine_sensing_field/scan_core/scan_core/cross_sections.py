"""단면 상세 법선 (용접 전 스캔의 결과물 — 사용자 2026-10-08: 최종 출력 = 단면 형상의 상세 법선 벡터들 + TCP 용접선 경로).

심 방향 단면 (station) 마다 그 그루브 형상이 가진 면 (판 상면 둘, 베벨면 — 일면 개선은 한쪽·V 는 양쪽, 직각 벽, 바닥 = 뒷댐재 상면)
의 단위 법선을 base 프레임으로 낸다. 패스 처리 (scan_core.seam_features) 가 프로파일마다 이미 고른 면 점에 센서 프레임 직선을 맞추고
(CrossSectionFeatures.faces), 그 프로파일의 실측 자세 T_base_opt 로 면 위 점과 레이저 평면 안 면 방향 d 를 base 3D 로 옮긴다
(SeamScanFeatures.face_point_base / face_dir_base). 면은 심 방향 u 와 d 를 담으므로 면 법선 = unit(d × u) — 레이저 평면이 판 법선에서
기울어도 (3G 입사·피치) 그대로 맞다. 법선은 재료 밖 (자유 공간·센서) 쪽: 그 프로파일의 광학 원점 쪽으로 향하게 부호를 정한다 (센서가
본 면은 센서 쪽이 자유 공간이다).

단면 = 심 직선 좌표 (S 끝 = 0, S→E) 의 단면 창 (용접 경로 단면 창 WeldProcess.station_window_mm; 프로파일이 성기면 그 창에 증거가
들어가도록 넓힘 — scan_l0.scan_quality.bin_for/bin_need 와 같은 규칙을 호출자가 준다) 마다 겹치지 않는 구간. 면은 **그 면 위 점의 심 좌표**
로 단면에 넣는다 (프로파일의 루트 점이 아니라): 기운 레이저 평면 (3G 입사·피치) 은 한 프로파일에서 판 상면·베벨·바닥을 심 방향으로 서로
다른 곳에서 본다 (깊이·tan(입사) — 10 mm 판·35° 입사면 판 상면이 루트에서 ~7 mm) — 단면의 면 = 그 단면 구간에 실제로 놓인 면 점. 루트
점 (단면 기준점 root_point_m, 단면 프로파일 수) 도 자기 심 좌표로. 단면의 면 법선 = 그 구간 프로파일 법선의 성분별 중앙값을 다시 단위화
(강건 집계), 퍼짐 = 프로파일 법선과 집계 법선 사이 각의 LMedS σ (두 수직 성분, scan_core.measure.lmeds_sigma dof 2). 증거가 모자란 면
(프로파일 수 < 단면 증거 수) 은 그 단면에서 뺀다 — 양끝 단면은 기운 레이저가 끝 너머에서 본 면이 없어 (그 프로파일의 루트가 시편 밖)
면 몇 개가 빠질 수 있다 (notes).

면의 쪽 (left/right) 은 심 좌표: S→E 로 볼 때 왼쪽 = +w, w = n × u (weld_plan 단면 +x 와 같다). 센서 x 의 쪽은 프로파일마다 두 판 점의
w 순서로 바꾼다.

출력 cross_sections.json (사람·KIRO 가 읽는 요약 + 단면 목록) / cross_sections.npz (배열: 단면 × 면). weld_plan 은 바꾸지 않는다.
루트 중심선 파일은 만들지 않는다 (단면의 root_point_m 은 그 단면의 기준점일 뿐 — 사용자 2026-10-03 '중심선 출력 없음' 은 그대로).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .measure import DESIGN_HALF, lmeds_sigma
from .seam_features import FACE_KINDS
from .specimen import GrooveSpec

__all__ = ["SCHEMA_VERSION", "FACE_NAMES", "FACE_TYPE", "profile_face_normals", "seam_sides", "expected_faces",
           "station_cross_sections", "save_cross_sections", "load_cross_sections", "decimated_cloud"]

SCHEMA_VERSION = 1
# 심 좌표의 면 이름 (S→E 로 볼 때 왼쪽 = +w). 순서 = npz 의 면 축
FACE_NAMES = ("plate_top_left", "plate_top_right", "bevel_left", "bevel_right", "wall_left", "wall_right", "gap_bottom")
FACE_TYPE = {"plate_top_left": "plate_top", "plate_top_right": "plate_top", "bevel_left": "bevel", "bevel_right": "bevel",
             "wall_left": "wall", "wall_right": "wall", "gap_bottom": "gap_bottom"}
_SENSOR_KIND = {k: i for i, k in enumerate(FACE_KINDS)}
_UNIT_EPS = 1e-9               # 수치 해상도: 이보다 짧은 벡터는 방향이 없다 (원칙 7)


def _unit_rows(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = V / n
    out[~(np.isfinite(n[..., 0]) & (n[..., 0] > _UNIT_EPS))] = np.nan
    return out


def profile_face_normals(face_dir_base, face_point_base, opt_origin_base, u) -> np.ndarray:
    """프로파일별 면 법선 (M, F, 3): n = unit(d × u), 광학 원점 쪽 (n·(o − p) > 0). d ∥ u 이거나 면이 없으면 NaN."""
    d = np.asarray(face_dir_base, dtype=float)
    P = np.asarray(face_point_base, dtype=float)
    o = np.asarray(opt_origin_base, dtype=float).reshape(-1, 1, 3)
    u = np.asarray(u, dtype=float).reshape(3)
    N = _unit_rows(np.cross(d, u[None, None, :]))
    s = np.sign(np.einsum("mfk,mfk->mf", N, o - P))
    s[s == 0.0] = np.nan
    return N * s[..., None]


def seam_sides(face_point_base, root_point_base, w) -> np.ndarray:
    """프로파일별 센서 x 쪽 → 심 쪽 부호 (M,): +1 = 센서 오른쪽 (+x) 판이 +w (S→E 왼쪽) 에 있다, −1 = 반대, NaN = 판 없음.
    두 판이 있으면 그 둘의 w 순서, 한 판만 있으면 그 판이 루트 점의 어느 쪽인가."""
    P = np.asarray(face_point_base, dtype=float)
    R = np.asarray(root_point_base, dtype=float).reshape(-1, 3)
    w = np.asarray(w, dtype=float).reshape(3)
    PL, PR = P[:, _SENSOR_KIND["plate_left"]], P[:, _SENSOR_KIND["plate_right"]]
    fl, fr, fo = (np.all(np.isfinite(a), axis=1) for a in (PL, PR, R))
    sg = np.full(P.shape[0], np.nan)
    both = fl & fr
    sg[both] = np.sign((PR[both] - PL[both]) @ w)
    m = fl & ~both & fo
    sg[m] = -np.sign((PL[m] - R[m]) @ w)
    m = fr & ~both & fo
    sg[m] = np.sign((PR[m] - R[m]) @ w)
    sg[sg == 0.0] = np.nan
    return sg


def _seam_face_index(kind: str, sg: float) -> int:
    """센서 면 이름 + 프로파일 쪽 부호 (seam_sides) → FACE_NAMES 인덱스."""
    if kind == "floor":
        return FACE_NAMES.index("gap_bottom")
    base, side = kind.rsplit("_", 1)
    left = (side == "right") == (sg > 0)                  # 센서 오른쪽 판이 +w 면 센서 오른쪽 = 심 왼쪽
    name = {"plate": "plate_top", "bevel": "bevel", "wall": "wall"}[base] + ("_left" if left else "_right")
    return FACE_NAMES.index(name)


def expected_faces(spec: Optional[GrooveSpec], bevel_side: str) -> tuple:
    """그루브 형상 (GrooveSpec.kind) 이 가진 면 → (측정 대상 이름 목록, 측정하지 않는 면 [{name, why}]). bevel_side = 측정한 쪽 (심 좌표:
    left | right | both | none | unknown). 루트면 (설계 root_face_mm > 0) 은 베벨 아래 짧은 수직 띠 — 추출기가 따로 맞추지 않는다."""
    kind = spec.kind if spec is not None else "single_bevel"
    names = ["plate_top_left", "plate_top_right"]
    not_measured = []
    if kind == "single_v":
        names += ["bevel_left", "bevel_right"]
        bev = ("left", "right")
    elif kind == "single_bevel":
        bev = (bevel_side,) if bevel_side in ("left", "right") else ()
        names += [f"bevel_{s}" for s in bev]
        names += [f"wall_{s}" for s in ("left", "right") if s not in bev] if bev else []
    else:
        bev = ()
        names += ["wall_left", "wall_right"]
    if spec is None or spec.backing:
        names.append("gap_bottom")
    if spec is not None and float(spec.root_face_mm) > 0.0:
        not_measured += [{"name": f"root_face_{s}", "why": f"root face {spec.root_face_mm:g} mm under the bevel: not fitted separately "
                          "(the extractor uses the design root face for the root edge)"} for s in bev]
    return names, not_measured


def station_cross_sections(feats: dict, S_point, E_point, *, window_m: float, need: int, spec: Optional[GrooveSpec] = None,
                           n_plate=None, base_frame: str = "", position: str = "", source: str = "") -> dict:
    """프로파일별 특징 (autoteach._features_arrays 또는 구간별 병합 — face_point_base·face_dir_base·face_fit·opt_origin_base·
    root_polyline_base_m·ok) + 납품 양끝 S/E (base) → 단면 목록 dict (save_cross_sections 가 json/npz 로).

    u = unit(E − S) 를 판 법선 n_plate (없으면 feats n_base) 에 직교화, w = n × u. 단면 = [0, L] 을 window_m 구간으로 (마지막은 짧을 수
    있다; 루트 점 증거 need 프로파일 이상), 면은 그 면 위 점의 심 좌표가 그 구간인 프로파일이 need 이상일 때만."""
    S = np.asarray(S_point, dtype=float).reshape(3)
    E = np.asarray(E_point, dtype=float).reshape(3)
    n = np.asarray(n_plate if n_plate is not None else feats.get("n_base"), dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    u = E - S
    u = u - (u @ n) * n
    L = float(np.linalg.norm(u))
    if not (L > 0.0 and np.isfinite(L)):
        raise ValueError("cross sections: S and E coincide along the seam")
    u = u / L
    w = np.cross(n, u)
    Pr = np.asarray(feats["root_polyline_base_m"], dtype=float).reshape(-1, 3)
    M = Pr.shape[0]
    need_keys = ("face_point_base", "face_dir_base", "face_fit", "opt_origin_base")
    if any(k not in feats or np.asarray(feats[k]).shape[:1] != (M,) for k in need_keys):
        raise ValueError(f"cross sections: profile face arrays missing ({[k for k in need_keys if k not in feats]})")
    FP = np.asarray(feats["face_point_base"], dtype=float)
    FD = np.asarray(feats["face_dir_base"], dtype=float)
    FF = np.asarray(feats["face_fit"], dtype=float)
    O = np.asarray(feats["opt_origin_base"], dtype=float)
    ok = np.asarray(feats.get("ok", np.ones(M, dtype=bool)), dtype=bool) & np.all(np.isfinite(Pr), axis=1)
    side = (O[ok] - Pr[ok]) @ n
    side = side[np.isfinite(side)]
    if side.size and float(np.median(side)) < 0.0:
        # 판 법선은 센서 (자유 공간) 쪽으로 — 그래야 w = n × u 가 S→E 의 왼쪽이다 (병합 특징의 n 은 맞춤 부호 그대로일 수 있다)
        n, w = -n, -w
    s = np.full(M, np.nan)
    s[ok] = (Pr[ok] - S) @ u
    NP = profile_face_normals(FD, FP, O, u)                  # (M, F_sensor, 3)
    sg = seam_sides(FP, Pr, w)
    # 센서 면 → 심 면 (M, F_seam, ...)
    F2 = len(FACE_NAMES)
    N2 = np.full((M, F2, 3), np.nan)
    P2 = np.full((M, F2, 3), np.nan)
    R2 = np.full((M, F2), np.nan)
    C2 = np.zeros((M, F2))
    for kind, i in _SENSOR_KIND.items():
        if kind == "floor":
            groups = [(ok, FACE_NAMES.index("gap_bottom"))]
        else:
            groups = [(ok & (sg > 0), _seam_face_index(kind, 1.0)), (ok & (sg < 0), _seam_face_index(kind, -1.0))]
        for sel, j in groups:
            N2[sel, j] = NP[sel, i]
            P2[sel, j] = FP[sel, i]
            R2[sel, j] = FF[sel, i, 0]
            C2[sel, j] = np.nan_to_num(FF[sel, i, 1])
    # 면마다 그 면 위 점의 심 좌표 (M, F_seam): 단면에는 그 구간에 놓인 면 점만 (프로파일 루트 점의 s 가 아니다 — 기운 레이저 평면)
    fin2 = np.all(np.isfinite(N2), axis=2) & np.all(np.isfinite(P2), axis=2)
    s2 = np.full((M, F2), np.nan)
    s2[fin2] = (P2[fin2] - S) @ u
    edges = np.arange(0.0, L, float(window_m))
    edges = np.append(edges, L) if edges.size == 0 or edges[-1] < L else edges
    if edges.size < 2:
        edges = np.array([0.0, L])
    stations = []
    count = {nm: 0 for nm in FACE_NAMES}
    sig_all = {nm: [] for nm in FACE_NAMES}
    def _in(sv, a_, b_, last_):
        with np.errstate(invalid="ignore"):                    # NaN (없는 점) 은 어느 구간에도 들지 않는다
            return (sv >= a_) & ((sv < b_) | (last_ & (sv <= b_)))

    for a, b in zip(edges[:-1], edges[1:]):
        last = b >= L
        m = ok & _in(s, a, b, last)                            # 루트 점 (단면 기준점) — 자기 심 좌표
        k = int(m.sum())
        if k < int(need):
            continue
        st = {"s_m": float(0.5 * (a + b)), "s_range_m": [float(a), float(b)], "root_point_m": np.median(Pr[m], axis=0).tolist(),
              "n_profiles": k, "faces": {}}
        in_face = _in(s2, a, b, last)                          # (M, F_seam): 면 위 점이 이 구간 (ok 프로파일만 값이 있다)
        for j, nm in enumerate(FACE_NAMES):
            mj = in_face[:, j]
            kj = int(mj.sum())
            if kj < int(need):
                continue
            Nj = N2[mj, j]
            nv = np.median(Nj, axis=0)
            nn = float(np.linalg.norm(nv))
            if not nn > _UNIT_EPS:
                continue
            nv = nv / nn
            ang = np.degrees(np.arccos(np.clip(Nj @ nv, -1.0, 1.0)))
            sig = float(lmeds_sigma(ang, dof=2))
            st["faces"][nm] = {"type": FACE_TYPE[nm], "side": nm.rsplit("_", 1)[-1] if nm != "gap_bottom" else "centre",
                               "normal": nv.tolist(), "point_m": np.median(P2[mj, j], axis=0).tolist(), "n_profiles": kj,
                               "n_points": int(np.sum(C2[mj, j])), "resid_mm": float(np.median(R2[mj, j])),
                               "sigma_deg": sig}
            count[nm] += 1
            sig_all[nm].append(sig)
        stations.append(st)
    bev_counts = {sd: count[f"bevel_{sd}"] for sd in ("left", "right")}
    if bev_counts["left"] and bev_counts["right"]:
        # 양쪽 다 베벨 (V) = 적은 쪽도 많은 쪽 증거의 절반 이상 (원칙 2), 아니면 많은 쪽 (드문 쪽은 오인식)
        bevel_side = ("both" if min(bev_counts.values()) >= DESIGN_HALF * max(bev_counts.values())
                      else max(bev_counts, key=bev_counts.get))
    elif bev_counts["left"] or bev_counts["right"]:
        bevel_side = "left" if bev_counts["left"] else "right"
    else:
        bevel_side = "none"
    expected, not_measured = expected_faces(spec, bevel_side)
    summary = {nm: {"n_stations": count[nm], "sigma_deg_median": float(np.median(sig_all[nm])) if sig_all[nm] else float("nan"),
                    "expected": nm in expected} for nm in FACE_NAMES}
    notes = []
    for nm in expected:
        if count[nm] < len(stations):
            notes.append(f"{nm}: measured at {count[nm]}/{len(stations)} sections"
                         + (" (vertical square face: in the laser shadow at this posture)" if FACE_TYPE[nm] == "wall" else ""))
    return {"schema_version": SCHEMA_VERSION, "base_frame": str(base_frame), "position": str(position), "source": str(source),
            "frame": "base frame [m]; normals are unit vectors pointing out of the material (free side / sensor side); "
                     "left/right looking S->E (left = +w, w = n x u)",
            "seam": {"S_m": S.tolist(), "E_m": E.tolist(), "u": u.tolist(), "n": n.tolist(), "w": w.tolist(), "length_m": L},
            "station_window_m": float(window_m), "min_profiles": int(need), "face_names": list(FACE_NAMES),
            "kind": spec.kind if spec is not None else None, "bevel_side": bevel_side, "faces_expected": expected,
            "faces_not_measured": not_measured, "summary": summary, "notes": notes, "stations": stations}


def save_cross_sections(cs: dict, out_dir, stem: str = "cross_sections") -> list:
    """cross_sections.json (요약 + 단면 목록) + cross_sections.npz (s_m (K,), s_range_m (K,2), root_point_m (K,3), face_names (F,),
    normal·point_m (K,F,3), n_profiles·n_points (K,F), resid_mm·sigma_deg (K,F); 없는 면 NaN/0; seam u·n·w, base_frame)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, npz = out / f"{stem}.json", out / f"{stem}.npz"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(_plain(cs), f, indent=1, ensure_ascii=False)
    st = cs.get("stations") or []
    K, F = len(st), len(FACE_NAMES)
    arr = {"normal": np.full((K, F, 3), np.nan), "point_m": np.full((K, F, 3), np.nan), "n_profiles": np.zeros((K, F), np.int64),
           "n_points": np.zeros((K, F), np.int64), "resid_mm": np.full((K, F), np.nan), "sigma_deg": np.full((K, F), np.nan)}
    for i, row in enumerate(st):
        for j, nm in enumerate(FACE_NAMES):
            fc = (row.get("faces") or {}).get(nm)
            if fc is None:
                continue
            arr["normal"][i, j] = fc["normal"]
            arr["point_m"][i, j] = fc["point_m"]
            for k_ in ("n_profiles", "n_points", "resid_mm", "sigma_deg"):
                arr[k_][i, j] = fc[k_]
    seam = cs.get("seam") or {}
    np.savez_compressed(npz, schema_version=np.array(SCHEMA_VERSION), base_frame=np.array(str(cs.get("base_frame", ""))),
                        face_names=np.array(FACE_NAMES), s_m=np.array([r["s_m"] for r in st], dtype=float).reshape(K),
                        s_range_m=np.array([r["s_range_m"] for r in st], dtype=float).reshape(K, 2),
                        root_point_m=np.array([r["root_point_m"] for r in st], dtype=float).reshape(K, 3),
                        u=np.asarray(seam.get("u", [np.nan] * 3), dtype=float), n=np.asarray(seam.get("n", [np.nan] * 3), dtype=float),
                        w=np.asarray(seam.get("w", [np.nan] * 3), dtype=float),
                        station_window_m=np.array(float(cs.get("station_window_m", np.nan))), **arr)
    return [str(jp), str(npz)]


def load_cross_sections(path) -> dict:
    """cross_sections.npz (또는 같은 이름의 .json 옆 npz) → 배열 dict (시각화·검사용)."""
    p = Path(path)
    if p.suffix == ".json":
        p = p.with_suffix(".npz")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def decimated_cloud(result, step_m: float) -> np.ndarray:
    """재구성된 패스 (ScanResult: x_mm, z_mm, valid, T_base_opt) → base 점 (n, 3) float32 — 시각화용으로 심 방향·레이저 선 방향 모두
    약 step_m 간격 (프로파일 간격·열 간격에서 보폭을 정한다). 점은 그 프로파일의 실측 자세 그대로 (재구성 결과)."""
    X = np.asarray(result.x_mm)
    Zm = np.asarray(result.z_mm)
    T = np.asarray(result.T_base_opt, dtype=float)
    N = int(X.shape[0]) if X.ndim == 2 else 0
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)
    o = T[:, :3, 3]
    fin_o = np.all(np.isfinite(o), axis=1)
    d = np.linalg.norm(np.diff(o[fin_o], axis=0), axis=1) if fin_o.sum() > 1 else np.zeros(0)
    d = d[d > 0]
    sp = float(np.median(d)) if d.size else float("nan")
    rs = max(1, int(round(float(step_m) / sp))) if (np.isfinite(sp) and sp > 0) else 1
    rows = np.arange(0, N, rs)
    valid = getattr(result, "valid", None)
    valid = np.asarray(valid, dtype=bool)[rows] if valid is not None and np.asarray(valid).shape == X.shape else None
    # 열 간격 = 남길 프로파일 전부의 이웃 유효 열 간격 중앙값 (한 프레임만 보면 그 프레임이 비었을 때 (드롭아웃·유령 프레임) 솎음이 꺼져
    # 점구름이 수 배로 커진다). 남길 행만 (float32: 보폭을 정하는 데는 충분, 패스 뒤 처리 작업 프로세스의 수 ms)
    Xr = np.array(X[rows], dtype=np.float32)
    if valid is not None:
        Xr[~valid] = np.nan
    with np.errstate(invalid="ignore"):
        dxs = np.abs(np.diff(Xr, axis=1)).ravel()
        dxs = dxs[dxs > 0]                                   # NaN (빈 열 옆) 은 비교에서 빠진다
    dx = float(np.median(dxs)) * 1e-3 if dxs.size else float("nan")
    cs = max(1, int(round(float(step_m) / dx))) if (np.isfinite(dx) and dx > 0) else 1
    x = np.asarray(X[rows, ::cs], dtype=float) * 1e-3
    if valid is not None:
        x[~valid[:, ::cs]] = np.nan
    z = Zm[rows, ::cs].astype(float) * 1e-3
    with np.errstate(invalid="ignore"):
        v = np.isfinite(x) & np.isfinite(z) & (z > 0)
    v &= fin_o[rows][:, None]
    R = T[rows, :3, :3]
    P = o[rows][:, None, :] + x[..., None] * R[:, None, :, 0] + z[..., None] * R[:, None, :, 2]
    return P[v].astype(np.float32)


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
