"""툴 몸체 (EOAT 메시) 의 볼록 껍질 점 — 안전 규칙의 높이 검사용 (eoat 프레임, m).

광학 원점·토치 팁·플랜지 키포인트만으로는 툴 몸체의 가장 낮은 점을 모른다: 2026-10-05 g3_04 (3G, 토치가 레이저 위) 에서 센서의
카메라 쪽 끝 (eoat +x 0.20 m) 이 광학 원점보다 175 mm 아래 (base z −0.177 m) 로 내려가 테이블보다 ~5 cm 낮았다 (사용자 관찰).
볼록 물체의 어떤 방향 최솟값도 껍질 꼭짓점에서 나오므로 껍질 점만 보면 높이·평면 거리 검사가 메시 전체와 같다.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

_CACHE: dict = {}


def load_stl_vertices(path) -> np.ndarray:
    """STL (binary 또는 ASCII) → 꼭짓점 (M, 3) [파일 단위 그대로]."""
    b = Path(path).read_bytes()
    if len(b) >= 84:
        n = struct.unpack("<I", b[80:84])[0]
        if 84 + 50 * n == len(b):
            a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            return a["v"].reshape(-1, 3).astype(float)
    pts = [[float(t) for t in ln.split()[1:4]] for ln in b.decode("ascii", errors="ignore").splitlines()
           if ln.strip().startswith("vertex")]
    if not pts:
        raise ValueError(f"no vertices in STL {path}")
    return np.asarray(pts, dtype=float)


def hull_points(V: np.ndarray, n_dirs: int = 2000) -> np.ndarray:
    """점 집합 → 볼록 껍질 꼭짓점. scipy 가 없으면 n_dirs 방향의 지지점 (근사: 껍질 꼭짓점의 부분집합)."""
    V = np.unique(np.round(np.asarray(V, dtype=float), 6), axis=0)
    try:
        from scipy.spatial import ConvexHull
        return V[ConvexHull(V).vertices]
    except Exception:  # noqa: BLE001 - scipy 없음/퇴화
        rng = np.random.default_rng(0)
        D = rng.normal(size=(int(n_dirs), 3))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        return V[np.unique(np.argmax(V @ D.T, axis=0))]


def body_points_from_mesh(path, scale: float = 1.0) -> np.ndarray:
    """메시 파일 → 껍질 점 (K, 3) [m] (eoat 프레임 = 메시 프레임; 메시 원점이 eoat_base_link 원점이어야 한다). 경로별 캐시."""
    key = (str(Path(path).resolve()), float(scale))
    if key not in _CACHE:
        _CACHE[key] = hull_points(load_stl_vertices(path) * float(scale))
    return _CACHE[key].copy()


def body_points_base(T_base_eoat: np.ndarray, body_eoat: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """eoat 포즈 (4x4 또는 (N,4,4)) → 몸체 점 base 좌표 ((K,3) 또는 (N,K,3)). 몸체가 없으면 None."""
    if body_eoat is None:
        return None
    T = np.asarray(T_base_eoat, dtype=float)
    B = np.asarray(body_eoat, dtype=float)
    if T.ndim == 2:
        return B @ T[:3, :3].T + T[:3, 3]
    return np.einsum("nij,kj->nki", T[:, :3, :3], B) + T[:, None, :3, 3]
