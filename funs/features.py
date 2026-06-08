"""특징 추출 헬퍼 — 예비실험 파이프라인.

이 모듈은 일반 윈도우 기준 특징 추출 함수들을 담은 딕셔너리를 생성하고,
타겟 스트림의 특징을 순차적으로 추출 및 변환.

P2(FFT)/P3(EnvSpec):
  magnitude 추출 → (선택적 log1p) → per-window z-score 순서 적용.
  log1p=True면 동적 범위를 압축한 뒤 z-score로 윈도우 간 스케일 통일.
P4(Cepstrum):
  raw window에 per-window z-score 적용 후 feature 추출.
P6(TDS):
  raw window에 feature 추출 후 per-window z-score 적용.
P1(Raw): 정규화 미적용 (원본 값 그대로).
모든 feature 함수는 float32를 반환한다 (RPi Zero 2W 배포 대비).
"""
from __future__ import annotations

import numpy as np

import funs


def per_window_znorm(X: np.ndarray) -> np.ndarray:
    """각 window(행)를 독립적으로 z-score 정규화. float32 반환.

    window 내 std < 1e-10 (상수 구간)이면 std=1 로 처리 → 0 벡터 반환.
    """
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-10, 1.0, std)
    return ((X - mean) / std).astype(np.float32)


def make_feature_fns(
    fs: float,
    lifter_n: int | None = None,
    log1p: bool = True,
    bandpass: list | tuple | None | str = "auto",
    rpm: float | None = None,
    order_spec_params: dict | None = None,
) -> dict:
    """
    일반 윈도우 기준 특징 추출 함수들을 담은 딕셔너리를 반환.

    fft / env_spec / order_spec: magnitude 추출 → (log1p=True면 log1p 압축) → per-window z-score.
    cepstrum: raw window에 per-window z-score 적용 후 feature 추출.
    tds: raw window에 feature 추출 후 per-window z-score 적용.
    raw: 정규화 미수행 (원본 값 그대로, float32 캐스팅만 수행).
    order_spec: rpm이 None이면 딕셔너리에 포함되지 않음.
    모든 반환값은 float32.

    Parameters
    ----------
    fs : float
        데이터의 샘플링 주파수 (Hz)
    lifter_n : int | None
        Cepstrum liftering quefrency 상한 (bins 수).
        None이면 liftering 미수행 (arr[:, :None] == arr[:, :]).
        config.yaml의 main.cepstrum_lifter_n[dataset]에서 가져와 전달할 것.
    log1p : bool
        True면 fft/env_spec/order_spec magnitude에 log1p 압축 적용 후 z-score.
        False면 magnitude 그대로 z-score만 적용.
    bandpass : list | tuple | None | str
        env_spec / order_spec 계산 시 적용할 bandpass.
        "auto" (기본값): fs 기반 자동 계산 (2 kHz ~ min(8 kHz, Nyquist-1)).
        None: bandpass 미적용 (전대역 → PU처럼 공진 대역 불명확한 경우 권장).
        [low_hz, high_hz]: 명시적 범위.
    rpm : float | None
        order_spec 추출에 사용할 회전 속도(RPM). None이면 order_spec 미포함.
        source/target 도메인별로 다른 RPM을 사용해 각각 호출해야 함.
    order_spec_params : dict | None
        order_spec 추가 파라미터. 지원 키:
          samples_per_rev (int, default 64): 회전당 angular resample 수
          max_order (float, default 20.0): 반환할 최대 order
          n_revs (int | None, default None): 고정 회전 수.
            cross-domain feature 차원 통일을 위해 config에서 명시적으로 전달 권장.

    Returns
    -------
    dict
        특징 추출 함수 매핑 딕셔너리
    """

    # Envelope bandpass 결정
    if bandpass == "auto":
        # 기존 동작: 2 kHz ~ min(8 kHz, Nyquist-1)
        _bp_high = min(8000.0, fs / 2.0 - 1.0)
        bp = (2000.0, _bp_high) if _bp_high > 2000.0 else None
    elif bandpass is None:
        bp = None
    else:
        bp = (float(bandpass[0]), float(bandpass[1]))

    def _compress(mag: np.ndarray) -> np.ndarray:
        return np.log1p(mag) if log1p else mag

    result = {
        "raw":      lambda X: X.astype(np.float32),
        "tds":      lambda X: per_window_znorm(funs.time_domain_stats(X)),
        "fft":      lambda X: per_window_znorm(_compress(funs.fft_magnitude(X, fs=fs)[0])),
        "env_spec": lambda X: per_window_znorm(_compress(funs.envelope_spectrum(X, fs=fs, bandpass=bp)[0])),
        "cepstrum": lambda X: funs.cepstrum(per_window_znorm(X))[:, :lifter_n],
    }

    if rpm is not None:
        _osp = order_spec_params or {}
        _spr = int(_osp.get("samples_per_rev", 64))
        _mo  = float(_osp.get("max_order", 20.0))
        _nr  = _osp.get("n_revs", None)
        _nr  = int(_nr) if _nr is not None else None
        _rpm = float(rpm)

        def _order_spec(
            X,
            _fs=fs, _rpm=_rpm, _bp=bp,
            _spr=_spr, _mo=_mo, _nr=_nr,
        ) -> np.ndarray:
            mag, _ = funs.order_envelope_spectrum(
                X, fs=_fs, rpm=_rpm, bandpass=_bp,
                samples_per_rev=_spr, max_order=_mo,
                n_revs=_nr, drop_dc=True,
            )
            return per_window_znorm(_compress(mag))

        result["order_spec"] = _order_spec

    return result


def extract_target_stream_features(X: np.ndarray, feature_fns: dict, scalers: dict) -> dict:
    """
    타겟 스트림의 특징을 추출하고 학습된 스케일러를 적용.
    OTTA 전제에 따라 타겟 스트림은 윈도우 1개씩 순차적으로 처리.
    실제 OTTA loop에서 SVDD inference/update가 들어갈 위치임.

    `scalers`에 없는 key(예: 'raw', 'fft' 등 per-window-norm 키)는
    scaling을 건너뛰고 원본 값을 그대로 반환.

    Parameters
    ----------
    X : np.ndarray
        타겟 도메인의 시계열 윈도우 데이터
    feature_fns : dict
        사용할 특징 추출 함수 매핑
    scalers : dict
        Source 데이터로 사전 학습된 스케일러 매핑

    Returns
    -------
    dict
        스케일링이 완료된 타겟 특징 딕셔너리
    """
    proc = {k: [] for k in feature_fns}
    for i in range(X.shape[0]):
        win = X[i : i + 1]
        for k, fn in feature_fns.items():
            feat = fn(win)
            if k in scalers:
                feat = scalers[k].transform(feat)
            proc[k].append(feat)
    return {k: np.concatenate(v, axis=0) for k, v in proc.items()}
