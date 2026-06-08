"""특징 추출 헬퍼.

P4(Cepstrum): raw window에 per-window z-score 적용 후 feature 추출.
모든 feature 함수는 float32를 반환한다.
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


def make_feature_fns(lifter_n: int | None = None) -> dict:
    """Cepstrum 특징 추출 함수 딕셔너리 반환.

    Parameters
    ----------
    lifter_n : int | None
        Cepstrum liftering quefrency 상한 (bins 수).
        None이면 liftering 미수행.
    """
    return {
        "cepstrum": lambda X: funs.cepstrum(per_window_znorm(X))[:, :lifter_n],
    }


def extract_target_stream_features(X: np.ndarray, feature_fns: dict) -> dict:
    """타겟 스트림 특징 추출 (윈도우 1개씩 순차 처리 — OTTA 전제)."""
    proc = {k: [] for k in feature_fns}
    for i in range(X.shape[0]):
        win = X[i : i + 1]
        for k, fn in feature_fns.items():
            proc[k].append(fn(win))
    return {k: np.concatenate(v, axis=0) for k, v in proc.items()}
