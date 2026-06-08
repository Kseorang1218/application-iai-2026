"""파이프라인 실행 모듈.

- `run_otta(...)`   — main 실험 (DualBoundarySVDD pre-train + streaming OTTA).
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np

from models.dual_boundary import Decision, DualBoundarySVDD
from models.online_svdd import OnlineSVDD

from .distribution import safe_mmd2, safe_wasserstein
from .train import make_kernel


# ── OTTA streaming pipeline ──────────────────────────────────────────────────

_DEC_CODE = {Decision.NORMAL_SKIP: 0, Decision.ADAPTED: 1, Decision.ANOMALY: 2}


def run_otta(
        prep_id: str,
        X_src_train: np.ndarray,
        X_src_val: np.ndarray,
        X_tgt_stream: np.ndarray,
        y_tgt_stream: np.ndarray,
        save_dir: pathlib.Path,
        kernel_name: str,
        scenario_label: str,
        config: dict,
        otta_mode: str = "dual_boundary",
        buffer_cap: int | None = None,
    ) -> dict:
    """SVDD pre-train + streaming OTTA.

    Raw artifacts 만 저장. 평가 지표(precision/recall/F1/AUROC/detection delay)는
    `analysis.py` 가 사후에 `otta_stream.npz` / `distances.npz` 를 읽어 계산.

    Parameters
    ----------
    X_tgt_stream : (N, d)
        Target stream feature 행렬 (정렬: normal → fault blocks, by databuilder).
    y_tgt_stream : (N,) int
        0=normal, 1=anomaly.
    otta_mode : str
        "dual_boundary" (기본) — DualBoundarySVDD: inner skip / adaptation zone /
            outer anomaly 3-zone 분기. rho_inner / rho_outer 사용.
        "single_boundary" — OnlineSVDD: 초기 n_warmup 샘플 무조건 adapt 후,
            예측 기반 TTA (score ≥ 0 → adapt, score < 0 → anomaly skip).

    산출물:
      svdd_model.npz / .json    — 최종 모델 (final SV/α/R²)
      distances.npz             — pre-trained 모델 기준 source/target d
      otta_stream.npz           — per-sample decisions/scores/latencies/R_trace
      metrics.json              — R + mmd²
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 모델 인스턴스화 + pre-train ───────────────────────────────────────
    N_src = X_src_train.shape[0]
    if N_src < 2:
        raise ValueError(f"need ≥ 2 source-train samples, got {N_src}")
    C = 1.0 / (float(config['svdd_nu']) * N_src)
    kernel = make_kernel(kernel_name, X_src_train)

    # n_target_normal: warmup 길이 산출용. 정답 라벨에서 카운트하므로 leakage 이나,
    # 본 실험에서는 의식적 비채택 (server/main/0520_bugfixtodo.md §2 ② 참고).
    n_target_normal = int((y_tgt_stream == 0).sum())
    warmup_ratio = float(config.get('warmup_ratio', 0.0))

    # 버퍼 크기 결정:
    #   상한: buffer_cap 지정 시 min(N_src+N_stream, buffer_cap), 없으면 N_src+N_stream.
    #   하한: buffer_size × C ≥ 1 feasibility 보장.
    min_feasible = math.ceil(1.0 / C) + 1
    natural_buffer = N_src + int(np.asarray(X_tgt_stream).shape[0])
    unbounded_buffer = min(natural_buffer, buffer_cap) if buffer_cap is not None else natural_buffer
    unbounded_buffer = max(unbounded_buffer, min_feasible)

    max_inner_iter = int(config.get('max_inner_iter', 10))
    if otta_mode == "dual_boundary":
        model = DualBoundarySVDD(
            kernel=kernel,
            C=C,
            buffer_size=unbounded_buffer,
            rho_inner=float(config['rho_inner']),
            rho_outer=float(config['rho_outer']),
            max_iter=int(config['svdd_max_iter']),
            tol=float(config['svdd_tol']),
            n_target_normal=n_target_normal,
            warmup_ratio=warmup_ratio,
            max_inner_iter=max_inner_iter,
        )
    else:  # single_boundary
        model = OnlineSVDD(
            kernel=kernel,
            C=C,
            buffer_size=unbounded_buffer,
            max_iter=int(config['svdd_max_iter']),
            tol=float(config['svdd_tol']),
            max_inner_iter=max_inner_iter,
        )

    model.fit(X_src_train)
    R_pretrain = float(np.sqrt(max(model.R2, 0.0)))

    # single_boundary warmup 길이 — dual_boundary 는 model.n_warmup 에서 읽음
    _n_warmup_sb = int(n_target_normal * warmup_ratio) if otta_mode == "single_boundary" else 0

    # ── 2. Pre-trained 모델 기준 거리 ───────────────────────────────────────
    X_tgt_stream = np.asarray(X_tgt_stream, dtype=np.float32)
    y_true = np.asarray(y_tgt_stream, dtype=np.int32)
    X_tgt_normal = X_tgt_stream[y_true == 0]
    X_tgt_fault  = X_tgt_stream[y_true == 1]

    d_src_train   = model.distance(X_src_train)
    d_src_val     = model.distance(X_src_val) if X_src_val.shape[0] > 0 else np.empty(0)
    d_tgt_normal  = model.distance(X_tgt_normal) if X_tgt_normal.shape[0] > 0 else np.empty(0)
    d_tgt_fault   = model.distance(X_tgt_fault)  if X_tgt_fault.shape[0]  > 0 else np.empty(0)

    _rho_inner = float(model.rho_inner) if otta_mode == "dual_boundary" else None
    _rho_outer = float(model.rho_outer) if otta_mode == "dual_boundary" else None
    _dist_extra: dict = {}
    if _rho_inner is not None:
        _dist_extra["rho_inner"] = np.array(_rho_inner)
        _dist_extra["rho_outer"] = np.array(_rho_outer)

    np.savez(
        save_dir / "distances.npz",
        source_train=d_src_train,
        source_val=d_src_val,
        target_normal=d_tgt_normal,
        target_fault=d_tgt_fault,
        R=np.array(R_pretrain),
        **_dist_extra,
    )

    # ── 3. Streaming loop ────────────────────────────────────────────────────
    n_stream = X_tgt_stream.shape[0]
    if n_stream == 0:
        raise ValueError("empty target stream")
    if y_true.shape[0] != n_stream:
        raise ValueError(
            f"y_tgt_stream length mismatch: {y_true.shape[0]} vs {n_stream}"
        )

    decisions = np.empty(n_stream, dtype=np.int8)
    scores    = np.empty(n_stream, dtype=np.float32)
    latencies = np.empty(n_stream, dtype=np.float64)
    R_trace   = np.empty(n_stream, dtype=np.float32)

    if otta_mode == "dual_boundary":
        for i in range(n_stream):
            x = X_tgt_stream[i:i+1]
            t0 = time.perf_counter()
            dec = model.process(x)
            latencies[i] = time.perf_counter() - t0
            decisions[i] = _DEC_CODE[dec]
            scores[i]    = float(model.decision_function(x)[0])  # process() 후 R2 갱신됐을 수 있어 재계산
            R_trace[i]   = float(np.sqrt(max(model.R2, 0.0)))
    else:  # single_boundary: warmup → 예측 기반 TTA
        for i in range(n_stream):
            x = X_tgt_stream[i:i+1]
            t0 = time.perf_counter()
            if i < _n_warmup_sb:
                # 워밍업 구간: 거리와 무관하게 무조건 adapt
                model.partial_fit(x)
                dec_code = _DEC_CODE[Decision.ADAPTED]
            else:
                # TTA 구간: 현재 모델 예측 기준 — 정상이면 adapt, 고장이면 skip
                score_pre = float(model.decision_function(x)[0])
                if score_pre >= 0:  # 정상으로 판단 → adapt
                    model.partial_fit(x)
                    dec_code = _DEC_CODE[Decision.ADAPTED]
                else:               # 고장으로 판단 → 탐지만, 업데이트 skip
                    dec_code = _DEC_CODE[Decision.ANOMALY]
            latencies[i] = time.perf_counter() - t0
            decisions[i] = dec_code
            scores[i]    = float(model.decision_function(x)[0])
            R_trace[i]   = float(np.sqrt(max(model.R2, 0.0)))

    R_final = float(R_trace[-1])

    # ── 4. 저장 ──────────────────────────────────────────────────────────────
    model.save(save_dir / "svdd_model")

    # Post-adaptation distances — 최종 모델 기준 재계산 (polar plot 비교용)
    _d_src_val_post = model.distance(X_src_val) if X_src_val.shape[0] > 0 else np.empty(0)
    np.savez(
        save_dir / "distances_final.npz",
        source_train=model.distance(X_src_train),
        source_val=_d_src_val_post,
        target_normal=model.distance(X_tgt_normal) if X_tgt_normal.shape[0] > 0 else np.empty(0),
        target_fault=model.distance(X_tgt_fault)   if X_tgt_fault.shape[0]  > 0 else np.empty(0),
        R=np.array(R_final),
        **_dist_extra,
    )

    np.savez(
        save_dir / "otta_stream.npz",
        decisions=decisions,
        scores=scores,
        y_true=y_true,
        latencies=latencies,
        R_trace=R_trace,
        R_pretrain=np.array(R_pretrain),
        R_final=np.array(R_final),
    )

    # 분포 거리
    mmd2 = safe_mmd2(X_src_train, X_tgt_normal) if X_tgt_normal.shape[0] > 0 else float('nan')
    wass_d = safe_wasserstein(model.distance(X_src_train), model.distance(X_tgt_normal)) if X_tgt_normal.shape[0] > 0 else float('nan')

    metrics = {
        "preprocessing": prep_id,
        "scenario": scenario_label,
        "kernel": kernel_name,
        "otta_mode": otta_mode,
        "R": R_pretrain,
        "R_final": R_final,
        "mmd2": float(mmd2) if np.isfinite(mmd2) else None,
        "wasserstein_1d_distance": float(wass_d) if np.isfinite(wass_d) else None,
        "n_source_train": int(X_src_train.shape[0]),
        "n_target_normal": int(X_tgt_normal.shape[0]),
        "n_target_fault":  int(X_tgt_fault.shape[0]),
        "feature_dim": int(X_src_train.shape[1]),
        "n_warmup": int(model.n_warmup) if otta_mode == "dual_boundary" else int(_n_warmup_sb),
        "buffer_size": int(model.buffer_size),
        "rho_inner": float(model.rho_inner) if otta_mode == "dual_boundary" else None,
        "rho_outer": float(model.rho_outer) if otta_mode == "dual_boundary" else None,
    }
    with open(save_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    return metrics
