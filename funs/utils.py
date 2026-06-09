"""공통 유틸리티 — CLI 인자 파싱, RBF gamma 휴리스틱."""
import argparse

import numpy as np

def parse_args(description: str = "Experiment Runner") -> argparse.Namespace:
    """CLI 인자 파싱. source/target 유효성 검증은 config 로드 후 호출측에서 수행."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cwru",
        choices=["cwru", "pu"],
        help="사용할 데이터셋 (cwru, pu)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="결과물을 저장할 디렉토리 이름 (기본값: results)",
    )
    return parser.parse_args()


def median_heuristic_gamma(
    X: np.ndarray,
    n_subsample: int = 2000,
    random_state: int = 42,
) -> float:
    """RBF gamma = 1 / median(||x_i - x_j||²)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] < 2:
        return 1.0
    rng = np.random.default_rng(random_state)
    if X.shape[0] > n_subsample:
        idx = rng.choice(X.shape[0], n_subsample, replace=False)
        X = X[idx]
    XX = np.sum(X * X, axis=1)
    sq = XX[:, None] + XX[None, :] - 2.0 * (X @ X.T)
    np.maximum(sq, 0.0, out=sq)
    iu = np.triu_indices(X.shape[0], k=1)
    sigma_sq = float(np.median(sq[iu]))
    return 1.0 / sigma_sq if sigma_sq > 0 else 1.0
