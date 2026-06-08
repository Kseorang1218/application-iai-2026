"""SVDD 학습 헬퍼 모듈."""
import numpy as np

from models import LinearKernel, PolyKernel, RBFKernel

from .utils import median_heuristic_gamma


def make_kernel(name: str, X_train: np.ndarray):
    """커널 이름과 학습 데이터로 SVDD 커널 객체 생성."""
    if name == "rbf":
        return RBFKernel(gamma=float(median_heuristic_gamma(X_train)))
    if name == "linear":
        return LinearKernel(c=0.0)
    if name == "poly":
        return PolyKernel(degree=3.0, gamma=1.0, coef0=1.0)
    raise ValueError(f"unsupported kernel: {name}")
