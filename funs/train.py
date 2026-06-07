"""SVDD 학습 헬퍼 모듈."""
import numpy as np

from models import LinearKernel, PolyKernel, RBFKernel, SVDD

from .utils import median_heuristic_gamma


def fit_svdd(X_train: np.ndarray, kernel_name: str, config: dict, config_section: str = 'pilot') -> SVDD:
    """Source 학습 데이터로 SVDD 학습. C = 1 / (nu * N)."""
    N = X_train.shape[0]
    if N < 2:
        raise ValueError(f"need ≥ 2 source-train samples, got {N}")
    kernel = make_kernel(kernel_name, X_train)
    cfg = config[config_section]
    C = 1.0 / (float(cfg['svdd_nu']) * N)
    svdd = SVDD(
        kernel=kernel,
        C=C,
        max_iter=int(cfg['svdd_max_iter']),
        tol=float(cfg['svdd_tol']),
    )
    svdd.fit(X_train)
    return svdd


def make_kernel(name: str, X_train: np.ndarray):
    """커널 이름과 학습 데이터로 SVDD 커널 객체 생성."""
    if name == "rbf":
        return RBFKernel(gamma=float(median_heuristic_gamma(X_train)))
    if name == "linear":
        return LinearKernel(c=0.0)
    if name == "poly":
        return PolyKernel(degree=3.0, gamma=1.0, coef0=1.0)
    raise ValueError(f"unsupported kernel: {name}")
