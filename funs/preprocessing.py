"""SVDD 입력 feature 추출 및 표준화 모듈."""
import math

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.spatial import KDTree
from scipy.stats import kurtosis, skew


TDS_FEATURE_NAMES = (
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "sample_entropy",
    "permutation_entropy",
)

# CWRU/PU/UOS 3개 dataset 공통 분석에서 universally redundant/zero-variance인 feature 제외 default
#   - mean: AC sensor 특성상 항상 ≈ 0 (모든 dataset std ~ O(1e-3) 이하) → z-score 분모 0 위험
#   - rms: std와 r=1.000 (3 dataset 전부, 평균 0 가정)
#   - clearance_factor: impulse_factor와 r ≥ 0.99 (3 dataset 전부, 가우시안 norm 비례)
# 모든 13개 feature가 필요하면 `keep=TDS_FEATURE_NAMES` 전달
DEFAULT_TDS_KEEP = (
    "std",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "shape_factor",
    "impulse_factor",
    "sample_entropy",
    "permutation_entropy",
)


class ZScoreScaler:
    """Z-score 표준화 (per-feature mean/std).

    source train에서 `fit`, val/test/target stream에서는 `transform`만 호출하여
    OTTA 전제(target fit 금지)를 유지한다. sklearn.Pipeline과 호환되도록
    `fit`/`transform`/`fit_transform` 인터페이스를 제공.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "ZScoreScaler":
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, D), got shape {X.shape}")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ZScoreScaler is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).transform(X)


def time_domain_stats(
    X: np.ndarray,
    *,
    keep: tuple[str, ...] | None = None,
    eps: float = 1e-12,
    sampen_m: int = 2,
    sampen_r_factor: float = 0.2,
    permen_m: int = 3,
    permen_tau: int = 1,
) -> np.ndarray:
    """
    Window 별 time-domain statistic feature 추출

    Parameters
    ----------
    X : np.ndarray
        shape `(N, window_size)` 의 2차원 배열. 각 row가 하나의 window
    keep : tuple[str, ...] or None
        반환할 feature 이름 (`TDS_FEATURE_NAMES` 부분집합).
        None이면 `DEFAULT_TDS_KEEP` (CWRU/PU/UOS 공통 redundancy 제거된 10개) 사용.
        전체 13개가 필요하면 `keep=TDS_FEATURE_NAMES` 전달.
    eps : float
        0 나눗셈 방지용 작은 값
    sampen_m : int
        Sample entropy embedding dimension
    sampen_r_factor : float
        Sample entropy tolerance r = sampen_r_factor * std (Richman-Moorman 표준값 0.2)
    permen_m : int
        Permutation entropy order (ordinal pattern length)
    permen_tau : int
        Permutation entropy 시간 지연 (delay)

    Returns
    -------
    np.ndarray
        shape `(N, len(keep))`. 컬럼 순서는 `keep` 순서와 동일.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, window_size), got shape {X.shape}")

    keep = keep if keep is not None else DEFAULT_TDS_KEEP
    unknown = [k for k in keep if k not in TDS_FEATURE_NAMES]
    if unknown:
        raise ValueError(
            f"unknown TDS feature names: {unknown}. allowed: {TDS_FEATURE_NAMES}"
        )
    keep_set = set(keep)

    N, n = X.shape
    abs_X = np.abs(X)
    # Amplitude/Energy 
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    peak = abs_X.max(axis=1)
    peak_to_peak = X.max(axis=1) - X.min(axis=1)
    # Distribution shape
    kurt = kurtosis(X, axis=1, fisher=True, bias=False)
    skw = skew(X, axis=1, bias=False)
    # Pulse/Impact ratio
    mean_abs = abs_X.mean(axis=1)
    mean_sqrt_abs = np.sqrt(abs_X).mean(axis=1)
    crest_factor = peak / (rms + eps)
    shape_factor = rms / (mean_abs + eps)
    impulse_factor = peak / (mean_abs + eps)
    clearance_factor = peak / (mean_sqrt_abs ** 2 + eps)

    # Complexity/Entropy — expensive, 요청된 경우에만 계산
    samp_en: np.ndarray = np.empty(0)
    perm_en: np.ndarray = np.empty(0)
    if "sample_entropy" in keep_set:
        samp_en = np.empty(N)
        if n < sampen_m + 2:
            samp_en[:] = np.nan
        else:
            K_s = n - sampen_m
            for i in range(N):
                xi = X[i]
                r = sampen_r_factor * std[i]
                if r <= 0:
                    samp_en[i] = 0.0
                    continue
                xm  = np.lib.stride_tricks.sliding_window_view(xi, sampen_m)[:K_s]
                xm1 = np.lib.stride_tricks.sliding_window_view(xi, sampen_m + 1)
                Bn = KDTree(xm).count_neighbors(KDTree(xm),   r=r, p=np.inf) - K_s
                An = KDTree(xm1).count_neighbors(KDTree(xm1), r=r, p=np.inf) - K_s
                samp_en[i] = -np.log(An / Bn) if (An > 0 and Bn > 0) else np.inf
    if "permutation_entropy" in keep_set:
        K_p = n - (permen_m - 1) * permen_tau
        if K_p <= 0:
            perm_en = np.full(N, np.nan)
        else:
            idx = np.arange(K_p)[:, None] + np.arange(permen_m)[None, :] * permen_tau
            ranks = np.argsort(X[:, idx], axis=2)
            weights = permen_m ** np.arange(permen_m)
            pattern_ids = (ranks * weights).sum(axis=2)
            perm_en = np.empty(N)
            log_mfact = np.log(math.factorial(permen_m))
            for i in range(N):
                _, counts = np.unique(pattern_ids[i], return_counts=True)
                p = counts / K_p
                perm_en[i] = -np.sum(p * np.log(p)) / log_mfact

    feature_map = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
        "skewness": skw,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "clearance_factor": clearance_factor,
        "sample_entropy": samp_en,
        "permutation_entropy": perm_en,
    }
    return np.stack([feature_map[k] for k in keep], axis=1).astype(np.float32)


def fft_magnitude(
    X: np.ndarray,
    fs: float | None = None,
    normalize: bool = True,
    drop_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    각 window에 rFFT를 적용해 single-sided magnitude spectrum 계산

    Parameters
    ----------
    X : np.ndarray
        shape `(N, window_size)`
    fs : float or None
        샘플링 주파수(Hz). 주면 주파수 축을 함께 반환
    normalize : bool
        True면 window_size로 나누어 amplitude 스케일 정규화
    drop_dc : bool
        True면 0 Hz bin을 제거. 진동 신호에서 DC는 보통 무의미

    Returns
    -------
    mag : np.ndarray
        shape `(N, window_size // 2 + 1)` 또는 `drop_dc=True`면 `(N, window_size // 2)`
    freqs : np.ndarray or None
        `fs=None`이면 None 반환
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, window_size), got shape {X.shape}")

    n = X.shape[1]
    spectrum = np.fft.rfft(X, axis=1)
    mag = np.abs(spectrum)
    if normalize:
        mag = mag / n

    freqs = np.fft.rfftfreq(n, d=1.0 / fs) if fs is not None else None

    if drop_dc:
        mag = mag[:, 1:]
        if freqs is not None:
            freqs = freqs[1:]

    return mag, freqs


def _bandpass_sos(fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """SOS bandpass filter 계수 반환. 0 < low < high < fs/2 조건 필수."""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if not (0 < low_n < high_n < 1):
        raise ValueError(
            f"bandpass must satisfy 0 < low < high < fs/2 (got low={low}, high={high}, fs={fs})"
        )
    return butter(order, [low_n, high_n], btype="band", output="sos")


def envelope(
    X: np.ndarray,
    fs: float | None = None,
    bandpass: tuple[float, float] | None = None,
    detrend: bool = True,
) -> np.ndarray:
    """
    Hilbert 변환 기반 진폭 envelope 계산

    베어링 결함의 AM 성분을 복조(demodulation)하기 위한 전처리
    원하면 Hilbert 적용 전에 resonance 대역 bandpass 필터를 적용할 수 있음

    Parameters
    ----------
    X : np.ndarray
        shape `(N, window_size)`
    fs : float or None
        샘플링 주파수(Hz). `bandpass` 사용 시 필수
    bandpass : tuple[float, float] or None
        `(low_hz, high_hz)`. 지정 시 해당 대역으로 bandpass 후 Hilbert 적용
    detrend : bool
        True면 envelope에서 평균을 제거 (envelope spectrum의 DC 성분 제거)

    Returns
    -------
    np.ndarray
        shape `(N, window_size)` 의 envelope 신호
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, window_size), got shape {X.shape}")

    sig = X
    if bandpass is not None:
        if fs is None:
            raise ValueError("fs must be provided when bandpass is set.")
        sos = _bandpass_sos(fs, bandpass[0], bandpass[1])
        sig = sosfiltfilt(sos, sig, axis=1).astype(np.float32)

    env = np.abs(np.asarray(hilbert(sig, axis=1))).astype(np.float32)
    if detrend:
        env = env - env.mean(axis=1, keepdims=True)
    return env


def envelope_spectrum(
    X: np.ndarray,
    fs: float,
    bandpass: tuple[float, float] | None = None,
    drop_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Envelope의 rFFT magnitude spectrum 계산 (베어링 결함 주파수 분석용)

    Parameters
    ----------
    X : np.ndarray
        shape `(N, window_size)`
    fs : float
        샘플링 주파수(Hz)
    bandpass : tuple[float, float] or None
        envelope 계산 시 사용할 bandpass. None이면 원 신호 그대로 Hilbert 적용
    drop_dc : bool
        True면 0 Hz bin을 제거

    Returns
    -------
    mag : np.ndarray
    freqs : np.ndarray
    """
    env = envelope(X, fs=fs, bandpass=bandpass, detrend=True)
    mag, freqs = fft_magnitude(env, fs=fs, normalize=True, drop_dc=drop_dc)
    assert freqs is not None  # fs is required, so freqs is guaranteed
    return mag, freqs


def order_envelope_spectrum(
    X: np.ndarray,
    fs: float,
    rpm: float | np.ndarray,
    bandpass: tuple[float, float] | None = None,
    samples_per_rev: int = 64,
    max_order: float | None = None,
    drop_dc: bool = True,
    n_revs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Order tracking 기반 envelope spectrum 계산

    각 window의 envelope을 해당 RPM 기준 각도(angle) 영역으로 resample 한 뒤
    rFFT를 적용. x축이 order(회전당 cycles)로 표현되므로 회전 속도 변화에
    대해 결함 component가 동일 위치에 나타남 (variable-speed 시나리오에 강건)

    각 window 내에서는 RPM이 일정하다고 가정 (computational order tracking).

    Parameters
    ----------
    X : np.ndarray
        shape `(N, window_size)`
    fs : float
        샘플링 주파수(Hz)
    rpm : float or np.ndarray
        각 window의 회전 속도(RPM). scalar 또는 shape `(N,)`
    bandpass : tuple[float, float] or None
        envelope 계산 시 사용할 bandpass. None이면 원 신호 그대로 Hilbert 적용
    samples_per_rev : int
        revolution 당 angular resample 포인트 수. order Nyquist = samples_per_rev / 2
    max_order : float or None
        반환할 최대 order. None이면 Nyquist까지
    drop_dc : bool
        True면 0 order bin 제거
    n_revs : float or None
        angular resample에 사용할 회전 수. None이면 호출 내 모든 window가
        커버하는 최소 정수 회전 수로 자동 결정. RPM이 다른 도메인 간 동일
        feature 차원을 보장하려면 명시적으로 지정 (예: source/target에 동일
        값 전달 → 두 도메인 모두 같은 n_orders 반환).

    Returns
    -------
    mag : np.ndarray
        shape `(N, n_orders)`
    orders : np.ndarray
        shape `(n_orders,)`. 단위는 회전당 cycles
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, window_size), got shape {X.shape}")

    N, n = X.shape
    rpm_arr = np.atleast_1d(np.asarray(rpm, dtype=float))
    if rpm_arr.size == 1:
        rpm_arr = np.full(N, rpm_arr.item())
    if rpm_arr.shape != (N,):
        raise ValueError(
            f"rpm must be scalar or 1-D of length N={N}, got shape {rpm_arr.shape}"
        )
    if np.any(rpm_arr <= 0):
        raise ValueError("All rpm values must be > 0")

    env = envelope(X, fs=fs, bandpass=bandpass, detrend=True)

    duration = n / fs
    n_revs_per_window = duration * rpm_arr / 60.0
    n_revs_min_avail = float(n_revs_per_window.min())

    if n_revs is None:
        n_revs_common = int(np.floor(n_revs_min_avail))
    else:
        n_revs_common = int(n_revs)
        if n_revs_common > n_revs_min_avail:
            raise ValueError(
                f"requested n_revs={n_revs_common} exceeds revs covered by window "
                f"({n_revs_min_avail:.3f} at rpm={rpm_arr.min()})"
            )

    if n_revs_common < 1:
        raise ValueError(
            f"window covers <1 revolution at min rpm={rpm_arr.min()}: "
            f"need window_size/fs * rpm/60 >= 1"
        )

    L = samples_per_rev * n_revs_common
    src_idx = np.arange(n)
    target_unit = np.arange(L) / samples_per_rev
    env_ang = np.empty((N, L), dtype=np.float32)
    for k in range(N):
        n_per_rev = 60.0 * fs / rpm_arr[k]
        env_ang[k] = np.interp(target_unit * n_per_rev, src_idx, env[k])

    spectrum = np.fft.rfft(env_ang, axis=1)
    mag = (np.abs(spectrum) / L).astype(np.float32)
    orders = np.fft.rfftfreq(L, d=1.0 / samples_per_rev)

    if max_order is not None:
        keep = orders <= max_order
        mag = mag[:, keep]
        orders = orders[keep]

    if drop_dc:
        mag = mag[:, 1:]
        orders = orders[1:]

    return mag, orders


def cepstrum(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    각 window의 real cepstrum 계산: `IFFT(log(|FFT(x)|))`

    cepstrum은 quefrency(초 단위) 영역 표현으로, 고조파(harmonic) 구조를
    단일 peak로 집약하여 회전 기계 진단에 유용

    실입력의 real cepstrum은 N/2 기준 대칭이므로 후반부는 제거하고
    `[0, N//2]` 범위만 반환

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
