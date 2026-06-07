"""OTTA 시나리오용 데이터 split / windowing 모듈."""
import warnings

import numpy as np
import pandas as pd


def _temporal_chunks(df: pd.DataFrame, ratios: list[float]) -> list[pd.DataFrame]:
    """
    각 신호를 비율에 따라 연속된 시간 청크로 분할하며 메타데이터는 유지

    청크는 동일 신호의 시간적으로 겹치지 않는 구간이므로, 각 청크에 독립적으로
    windowing을 적용해도 window 중복 leakage가 발생하지 않음

    Parameters
    ----------
    df : pd.DataFrame
        `data` 컬럼에 1차원 신호가 들어 있는 DataFrame
    ratios : list[float]
        각 청크의 길이 비율. 합이 1이어야 함.

    Returns
    -------
    list[pd.DataFrame]
        입력 비율과 동일한 길이의 DataFrame 리스트. 각 요소는 원본 row 구조를
        유지한 채 `data` 컬럼만 해당 시간 구간으로 교체된 DataFrame
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
    chunk_rows = [[] for _ in ratios]
    for _, row in df.iterrows():
        signal = np.asarray(row["data"])
        # 다차원 신호인 경우 1차원 벡터로 펼침
        if signal.ndim != 1:
            signal = signal.ravel()
        # 분할 인덱스 계산
        n = len(signal)
        cuts = np.cumsum([int(round(n * r)) for r in ratios])
        cuts[-1] = n
        # 신호 분할
        start = 0
        for i, end in enumerate(cuts):
            new_row = row.copy()
            new_row["data"] = signal[start:end]
            chunk_rows[i].append(new_row)
            start = end
    return [pd.DataFrame(rows).reset_index(drop=True) for rows in chunk_rows]


def _window_signal(
    signal: np.ndarray, window_size: int, stride: int
) -> np.ndarray:
    """
    1차원 신호를 고정 크기 window로 펼쳐 (n_windows, window_size) ndarray 반환.
    신호 길이가 window_size보다 짧으면 빈 배열 반환.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        signal = signal.ravel()
    if len(signal) < window_size:
        return np.empty((0, window_size), dtype=signal.dtype)
    n_windows = (len(signal) - window_size) // stride + 1
    return np.stack(
        [signal[w * stride : w * stride + window_size] for w in range(n_windows)]
    )


def split_dataframe(
    raw_df: pd.DataFrame,
    source_domain: int,
    target_domain: int,
    train_val_test_ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    OTTA anomaly detection 실험용 source/target DataFrame 분할 

    Source: 정상/결함 모두 (train, val, test) 비율로 시간축 분할.
    train은 정상만 사용 (결함의 train 청크는 버림).
    결함 종류별 균등 배분은 _temporal_chunks가 각 row를 동일 비율로 분할하여 자동 보장.
    Target: 도메인 필터링만 수행, 분할/정렬 없이 원본 신호 형태 그대로 반환.

    Parameters
    ----------
    raw_df : pd.DataFrame
        `rpm`, `is_anomaly`, `fault_type`, `data` 컬럼을 포함하는 DataFrame
    source_domain : int
        Source 도메인의 rpm 값
    target_domain : int
        Target 도메인의 rpm 값
    train_val_test_ratio : tuple[float, float, float]
        정상 신호를 시간축으로 나눌 (train, val, test) 비율. 합=1. 기본 (0.70, 0.15, 0.15).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `(source_train_raw, source_val_raw, source_test_raw, target_raw)`.
        모든 DataFrame은 row=신호 구조. `data` 컬럼이 시간 구간 청크로 교체된 상태.
        source_train_raw는 정상만 포함.
    """
    if abs(sum(train_val_test_ratio) - 1.0) > 1e-9:
        raise ValueError(
            f"train_val_test_ratio must sum to 1, got {sum(train_val_test_ratio)}"
        )
    train_r, val_r, test_r = train_val_test_ratio

    src_df = raw_df[raw_df["rpm"] == source_domain].reset_index(drop=True)
    tgt_df = raw_df[raw_df["rpm"] == target_domain].reset_index(drop=True)
    if src_df.empty:
        raise ValueError(f"No rows for source_domain rpm={source_domain}.")
    if tgt_df.empty:
        raise ValueError(f"No rows for target_domain rpm={target_domain}.")

    src_normal = src_df[src_df["is_anomaly"] == 0]
    src_fault = src_df[src_df["is_anomaly"] == 1]

    train_normal, val_normal, test_normal = _temporal_chunks(
        src_normal, [train_r, val_r, test_r]
    )

    # 결함: 정상과 동일 비율로 시간축 분할, train 결함은 사용하지 않음
    # 각 row(베어링)가 독립적으로 분할되므로 결함 종류별 균등 배분 보장
    if not src_fault.empty:
        _, val_fault, test_fault = _temporal_chunks(
            src_fault, [train_r, val_r, test_r]
        )
    else:
        val_fault = src_fault.iloc[0:0]
        test_fault = src_fault.iloc[0:0]

    source_train_raw = train_normal.reset_index(drop=True)
    source_val_raw = pd.concat([val_normal, val_fault], ignore_index=True)
    source_test_raw = pd.concat([test_normal, test_fault], ignore_index=True)
    target_raw = tgt_df.reset_index(drop=True)

    assert (source_train_raw["is_anomaly"] == 0).all(), (
        "source_train_raw must contain only normal signals."
    )

    return source_train_raw, source_val_raw, source_test_raw, target_raw


def build_source_xy(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    shuffle: bool = False,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DataFrame의 각 row 신호에 sliding window를 적용해 (X, y) ndarray 생성
    Label은 `is_anomaly` 컬럼(이진)을 사용. 신호 길이가 window_size보다 짧으면 스킵

    Parameters
    ----------
    df : pd.DataFrame
        `data`, `is_anomaly` 컬럼을 포함하는 DataFrame (row=신호)
    window_size : int
        window 길이(샘플 수)
    stride : int
        연속된 window 사이의 간격(샘플 수)
    shuffle : bool
        True면 생성된 window 순서를 무작위로 섞음. 기본값 False
    random_state : int
        shuffle 시드. 기본값 42

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(X, y)`. X.shape=(N, window_size), y.shape=(N,), dtype(y)=int
    """
    X_chunks, y_chunks = [], []
    for _, row in df.iterrows():
        windows = _window_signal(row["data"], window_size, stride)
        if windows.shape[0] == 0:
            warnings.warn(
                f"신호 길이 < window_size {window_size} — 스킵 "
                f"(rpm={row.get('rpm')}, fault_type={row.get('fault_type')})"
            )
            continue
        X_chunks.append(windows)
        y_chunks.append(np.full(windows.shape[0], int(row["is_anomaly"])))

    if not X_chunks:
        return (
            np.empty((0, window_size), dtype=float),
            np.empty((0,), dtype=int),
        )

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)

    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X))
        X = X[idx]
        y = y[idx]

    return X, y


def build_target_stream(
    tgt_df: pd.DataFrame,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Target DataFrame을 windowing하고 `normal → sorted(fault_type) blocks` 순으로
    정렬된 stream (X, y, fault_type)을 생성.

    셔플하지 않으며, is_anomaly는 단조 비감소, fault_type 블록 연속성을 보장.

    Parameters
    ----------
    tgt_df : pd.DataFrame
        `data`, `is_anomaly`, `fault_type` 컬럼을 포함하는 DataFrame (row=신호)
    window_size : int
        window 길이(샘플 수)
    stride : int
        window 간 간격(샘플 수)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        `(X_stream, y_stream, fault_type_stream)`.
        X_stream.shape=(N, window_size), y_stream.shape=(N,), fault_type_stream.shape=(N,)
    """
    X_rows, y_rows, ft_rows = [], [], []
    for _, row in tgt_df.iterrows():
        windows = _window_signal(row["data"], window_size, stride)
        if windows.shape[0] == 0:
            warnings.warn(
                f"신호 길이 < window_size {window_size} — 스킵 "
                f"(rpm={row.get('rpm')}, fault_type={row.get('fault_type')})"
            )
            continue
        n = windows.shape[0]
        X_rows.append(windows)
        y_rows.append(np.full(n, int(row["is_anomaly"])))
        ft_rows.append(np.array([row["fault_type"]] * n, dtype=object))

    if not X_rows:
        return (
            np.empty((0, window_size), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=object),
        )

    X = np.concatenate(X_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    ft = np.concatenate(ft_rows, axis=0)

    # normal 먼저, 이어서 fault는 fault_type 알파벳 순 블록
    normal_idx = np.where(y == 0)[0]
    fault_idx = np.where(y == 1)[0]
    fault_order = fault_idx[np.argsort(ft[fault_idx], kind="stable")]
    order = np.concatenate([normal_idx, fault_order])

    X_stream = X[order]
    y_stream = y[order]
    ft_stream = ft[order]

    if y_stream.size > 0:
        assert np.all(np.diff(y_stream) >= 0), (
            "y_stream must be monotonically non-decreasing."
        )
        fault_mask = y_stream == 1
        if fault_mask.any():
            seen, prev = set(), None
            for t in ft_stream[fault_mask]:
                if t != prev:
                    assert t not in seen, (
                        f"fault_type block {t!r} is not contiguous in target stream."
                    )
                    seen.add(t)
                    prev = t

    return X_stream, y_stream, ft_stream
