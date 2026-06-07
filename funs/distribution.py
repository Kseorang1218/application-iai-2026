"""분포 측정/평가 모듈.

SVDD 거리 기반 zone 점유율, 분포 거리 지표(MMD², Wasserstein)를 담당한다.
순수 계산 레이어 (I/O·시각화 없음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from models.kernels import rbf_kernel
from .utils import median_heuristic_gamma


# ── 분포 거리 지표 ──────────────────────────────────────────────────────────────

def mmd_rbf(X_s: np.ndarray, X_t: np.ndarray, gamma: float | None = None) -> float:
    """Biased MMD²(X_s, X_t) under RBF kernel."""
    X_s = np.asarray(X_s, dtype=float)
    X_t = np.asarray(X_t, dtype=float)
    if X_s.ndim != 2 or X_t.ndim != 2 or X_s.shape[1] != X_t.shape[1]:
        raise ValueError(f"X_s, X_t must share feature dim. got {X_s.shape}, {X_t.shape}")
    if gamma is None:
        gamma = median_heuristic_gamma(np.vstack([X_s, X_t]))
    K_ss = rbf_kernel(X_s, X_s, gamma)
    K_tt = rbf_kernel(X_t, X_t, gamma)
    K_st = rbf_kernel(X_s, X_t, gamma)
    return float(K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean())


def safe_mmd2(X_s: np.ndarray, X_t: np.ndarray) -> float:
    """RBF 커널 MMD². 빈 배열 또는 차원 불일치 시 nan."""
    if X_s.size == 0 or X_t.size == 0:
        return float("nan")
    if X_s.ndim < 2 or X_t.ndim < 2 or X_s.shape[1] != X_t.shape[1]:
        return float("nan")
    return mmd_rbf(X_s, X_t)


def safe_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 거리 (1D). 빈 배열 시 nan."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


# ── SVDD 거리 분포 ─────────────────────────────────────────────────────────────

def zone_summary(
    d_groups: dict[str, np.ndarray],
    r_inner: float,
    r_outer: float,
) -> pd.DataFrame:
    """그룹별 inner / adaptation / outer zone 점유율."""
    if not (0.0 <= r_inner < r_outer):
        raise ValueError(f"need 0 ≤ r_inner < r_outer, got {r_inner}, {r_outer}")
    rows = []
    for name, d in d_groups.items():
        d = np.asarray(d)
        n = d.size
        if n == 0:
            rows.append({
                "count": 0, "inner_pct": np.nan, "adaptation_pct": np.nan,
                "outer_pct": np.nan, "mean_d": np.nan, "std_d": np.nan,
            })
            continue
        inner = int((d < r_inner).sum())
        outer = int((d > r_outer).sum())
        adapt = n - inner - outer
        rows.append({
            "count": int(n),
            "inner_pct": 100.0 * inner / n,
            "adaptation_pct": 100.0 * adapt / n,
            "outer_pct": 100.0 * outer / n,
            "mean_d": float(d.mean()),
            "std_d": float(d.std()),
        })
    return pd.DataFrame(rows, index=list(d_groups.keys()))
