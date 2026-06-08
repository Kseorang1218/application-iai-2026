"""RBF (Gaussian) kernel — CuPy / NumPy vectorized 구현.

SVDD 학습/추론에서 사용. dual form 거리 계산에 K(x, x) = 1 이라는 RBF
특수 성질이 활용되므로, 다른 커널로 확장할 때는 호출 측에서 diag()를
명시적으로 호출하도록 인터페이스를 통일한다.
"""

try:
    import cupy as xp
except ImportError:
    import numpy as xp

import numpy as np  # __main__ 및 타입 힌트용


def rbf_kernel(X, Y, gamma: float):
    """RBF 커널 행렬 K(X, Y).

    수식:
        K(x, y) = exp(-gamma * ||x - y||^2)

    제곱 거리는 다음 항등식으로 벡터화:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x . y
    """
    X = xp.asarray(X, dtype=xp.float32)
    Y = xp.asarray(Y, dtype=xp.float32)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X, Y must be 2-D, got shapes {X.shape}, {Y.shape}")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"feature dim mismatch: {X.shape[1]} vs {Y.shape[1]}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    XX = xp.sum(X * X, axis=1, keepdims=True)  # (n, 1)
    YY = xp.sum(Y * Y, axis=1, keepdims=True)  # (m, 1)
    XY = X @ Y.T                                # (n, m)

    sq_dist = XX + YY.T - 2.0 * XY
    xp.maximum(sq_dist, 0.0, out=sq_dist)       # 부동소수 음수 방지
    return xp.exp(-gamma * sq_dist)


def rbf_kernel_diag(X, gamma: float):
    """RBF 커널 K(X, X)의 대각 성분.

    RBF의 경우 항상 1 이지만, 다른 커널 확장을 위해 함수 형태로 분리.
    """
    X = xp.asarray(X, dtype=xp.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    return xp.ones(X.shape[0], dtype=xp.float32)


class RBFKernel:
    """SVDD/OnlineSVDD에서 쓰는 stateful 커널 래퍼.

    save/load 시 json-serializable dict로 직렬화 가능하도록 to_dict/from_dict 제공.
    """

    def __init__(self, gamma: float):
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.gamma: float = float(gamma)

    def __call__(self, X, Y):
        return rbf_kernel(X, Y, self.gamma)

    def diag(self, X):
        return rbf_kernel_diag(X, self.gamma)

    @property
    def name(self) -> str:
        return "rbf"

    def to_dict(self) -> dict:
        return {"name": self.name, "gamma": self.gamma}

    @classmethod
    def from_dict(cls, d: dict) -> "RBFKernel":
        if d.get("name") != "rbf":
            raise ValueError(f"unknown kernel: {d.get('name')}")
        return cls(gamma=float(d["gamma"]))


def linear_kernel(X, Y, c: float = 0.0):
    """Linear 커널 행렬 K(X, Y).

    K(x, y) = x^T y + c
    """
    X = xp.asarray(X, dtype=xp.float32)
    Y = xp.asarray(Y, dtype=xp.float32)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X, Y must be 2-D, got shapes {X.shape}, {Y.shape}")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"feature dim mismatch: {X.shape[1]} vs {Y.shape[1]}")
    return X @ Y.T + c


def linear_kernel_diag(X, c: float = 0.0):
    """Linear 커널 K(X, X)의 대각 성분."""
    X = xp.asarray(X, dtype=xp.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    return xp.sum(X * X, axis=1) + c


class LinearKernel:
    """Linear 커널 래퍼."""
    def __init__(self, c: float = 0.0):
        self.c: float = float(c)

    def __call__(self, X, Y):
        return linear_kernel(X, Y, self.c)

    def diag(self, X):
        return linear_kernel_diag(X, self.c)

    @property
    def name(self) -> str:
        return "linear"

    def to_dict(self) -> dict:
        return {"name": self.name, "c": self.c}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearKernel":
        if d.get("name") != "linear":
            raise ValueError(f"unknown kernel: {d.get('name')}")
        return cls(c=float(d.get("c", 0.0)))


def poly_kernel(X, Y, degree: float = 3.0, gamma: float = 1.0, coef0: float = 1.0):
    """Polynomial 커널 행렬 K(X, Y).

    K(x, y) = (gamma * x^T y + coef0)^degree
    """
    X = xp.asarray(X, dtype=xp.float32)
    Y = xp.asarray(Y, dtype=xp.float32)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X, Y must be 2-D, got shapes {X.shape}, {Y.shape}")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"feature dim mismatch: {X.shape[1]} vs {Y.shape[1]}")
    return (gamma * (X @ Y.T) + coef0) ** degree


def poly_kernel_diag(X, degree: float = 3.0, gamma: float = 1.0, coef0: float = 1.0):
    """Polynomial 커널 K(X, X)의 대각 성분."""
    X = xp.asarray(X, dtype=xp.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    return (gamma * xp.sum(X * X, axis=1) + coef0) ** degree


class PolyKernel:
    """Polynomial 커널 래퍼."""
    def __init__(self, degree: float = 3.0, gamma: float = 1.0, coef0: float = 1.0):
        self.degree: float = float(degree)
        self.gamma: float = float(gamma)
        self.coef0: float = float(coef0)

    def __call__(self, X, Y):
        return poly_kernel(X, Y, self.degree, self.gamma, self.coef0)

    def diag(self, X):
        return poly_kernel_diag(X, self.degree, self.gamma, self.coef0)

    @property
    def name(self) -> str:
        return "poly"

    def to_dict(self) -> dict:
        return {"name": self.name, "degree": self.degree, "gamma": self.gamma, "coef0": self.coef0}

    @classmethod
    def from_dict(cls, d: dict) -> "PolyKernel":
        if d.get("name") != "poly":
            raise ValueError(f"unknown kernel: {d.get('name')}")
        return cls(
            degree=float(d.get("degree", 3.0)),
            gamma=float(d.get("gamma", 1.0)),
            coef0=float(d.get("coef0", 1.0))
        )


def create_kernel(d: dict):
    """Factory function for recreating a kernel from a dictionary."""
    name = d.get("name")
    if name == "rbf":
        return RBFKernel.from_dict(d)
    elif name == "linear":
        return LinearKernel.from_dict(d)
    elif name == "poly":
        return PolyKernel.from_dict(d)
    else:
        raise ValueError(f"unknown kernel: {name}")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, m, d = 30, 20, 5
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((m, d))
    gamma = 0.5

    K_xy = rbf_kernel(X, Y, gamma)
    K_xx = rbf_kernel(X, X, gamma)

    print("===== RBF kernel sanity check =====")
    print(f"K(X, Y) shape: {K_xy.shape}  (expected ({n}, {m}))")
    print(f"K(X, X) shape: {K_xx.shape}  (expected ({n}, {n}))")

    # 1) 대칭성
    sym_err = float(xp.max(xp.abs(K_xx - K_xx.T)))
    print(f"\n[symmetry]  max|K(X,X) - K(X,X)^T| = {sym_err:.2e}  (expect ~ 0)")
    assert sym_err < 1e-12, "K(X, X) is not symmetric"

    # 2) 대각 성분 = 1 (float32 + cuBLAS 누적 오차 허용: < 1e-5)
    diag = xp.diag(K_xx)
    diag_err = float(xp.max(xp.abs(diag - 1.0)))
    print(f"[diag=1]    max|diag(K(X,X)) - 1| = {diag_err:.2e}  (expect ~ 0)")
    assert diag_err < 1e-5, "K(X, X) diagonal is not 1"

    # 3) rbf_kernel_diag 와 일치
    diag_helper = rbf_kernel_diag(X, gamma)
    helper_err = float(xp.max(xp.abs(diag - diag_helper)))
    print(f"[diag fn]   max|diag(K) - rbf_kernel_diag(X)| = {helper_err:.2e}")
    assert helper_err < 1e-5

    # 4) [0, 1] 범위
    K_xy_min = float(K_xy.min())
    K_xy_max = float(K_xy.max())
    print(f"[range]     K(X,Y) min/max = {K_xy_min:.4f} / {K_xy_max:.4f}  (expect [0, 1])")
    assert K_xy_min >= 0.0 and K_xy_max <= 1.0

    # 5) gamma 의존성: gamma 커지면 비대각 빠르게 0으로
    K_small = rbf_kernel(X, Y, 0.01)
    K_large = rbf_kernel(X, Y, 10.0)
    print(f"[gamma]     mean K @ gamma=0.01 → {float(K_small.mean()):.4f}, gamma=10 → {float(K_large.mean()):.4e}")
    assert float(K_small.mean()) > float(K_large.mean())

    # 6) RBFKernel wrapper / 직렬화 round-trip
    kf = RBFKernel(gamma)
    K_wrap = kf(X, Y)
    assert xp.allclose(K_wrap, K_xy)
    kf2 = RBFKernel.from_dict(kf.to_dict())
    assert kf2.gamma == kf.gamma
    print(f"[wrapper]   RBFKernel(gamma={kf.gamma}) round-trip OK")

    print("\nAll kernel sanity checks passed.")
