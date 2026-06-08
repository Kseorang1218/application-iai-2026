"""SVDD 입력 feature 추출 모듈."""
import numpy as np


def cepstrum(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """각 window의 real cepstrum 계산: `IFFT(log(|FFT(x)|))`

    실입력의 real cepstrum은 N/2 기준 대칭이므로 후반부는 제거하고
    `[0, N//2]` 범위만 반환.

    Parameters
    ----------
    X : np.ndarray
        shape `(N, window_size)`
    eps : float
        log(0) 방지용 작은 값

    Returns
    -------
    np.ndarray
        shape `(N, window_size // 2 + 1)` 의 real cepstrum
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, window_size), got shape {X.shape}")

    n = X.shape[1]
    spectrum = np.fft.fft(X, axis=1)
    log_mag = np.log(np.abs(spectrum) + eps)
    ceps = np.fft.ifft(log_mag, axis=1).real
    return ceps[:, : n // 2 + 1].astype(np.float32)
