"""파이프라인 실행 모듈.

- `run_otta(...)`   — main 실험 (DualBoundarySVDD pre-train + streaming OTTA).
  파일 저장 없이 평가 메트릭 dict 반환.
"""
from __future__ import annotations

import math
import time

import numpy as np

from models.dual_boundary import Decision, DualBoundarySVDD
from models.online_svdd import OnlineSVDD

from .evaluation import AnomalyDetectionEvaluator
from .train import make_kernel


# ── OTTA streaming pipeline ──────────────────────────────────────────────────

_DEC_CODE = {Decision.NORMAL_SKIP: 0, Decision.ADAPTED: 1, Decision.ANOMALY: 2}
_DECISION_NAMES = {0: "NORMAL_SKIP", 1: "ADAPTED", 2: "ANOMALY"}


def _detection_delay(y_true: np.ndarray, y_pred: np.ndarray) -> int | None:
    anom_idx = np.where(y_true == 1)[0]
    if anom_idx.size == 0:
        return None
    first_anom = int(anom_idx[0])
    det_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    if det_idx.size == 0:
        return -1
    return int(det_idx[0]) - first_anom


def run_otta(
        X_src_train: np.ndarray,
        X_tgt_stream: np.ndarray,
        y_tgt_stream: np.ndarray,
        kernel_name: str,
        scenario_label: str,
        config: dict,
        otta_mode: str = "dual_boundary",
        buffer_cap: int | None = None,
        save_dir: str | None = None,
    ) -> dict:
    """SVDD pre-train + streaming OTTA. 평가 메트릭 dict 반환.
    save_dir 지정 시 최종 모델을 <save_dir>/svdd_model.{npz,json}으로 저장."""

    # ── 1. 모델 인스턴스화 + pre-train ───────────────────────────────────────
    N_src = X_src_train.shape[0]
    if N_src < 2:
        raise ValueError(f"need ≥ 2 source-train samples, got {N_src}")
    C = 1.0 / (float(config['svdd_nu']) * N_src)
    kernel = make_kernel(kernel_name, X_src_train)

    n_target_normal = int((y_tgt_stream == 0).sum())
    warmup_ratio = float(config.get('warmup_ratio', 0.0))

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

    _n_warmup_sb = int(n_target_normal * warmup_ratio) if otta_mode == "single_boundary" else 0

    # ── 2. Streaming loop ────────────────────────────────────────────────────
    X_tgt_stream = np.asarray(X_tgt_stream, dtype=np.float32)
    y_true = np.asarray(y_tgt_stream, dtype=np.int32)
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
            scores[i]    = float(model.decision_function(x)[0])
            R_trace[i]   = float(np.sqrt(max(model.R2, 0.0)))
    else:  # single_boundary: warmup → 예측 기반 TTA
        for i in range(n_stream):
            x = X_tgt_stream[i:i+1]
            t0 = time.perf_counter()
            if i < _n_warmup_sb:
                model.partial_fit(x)
                dec_code = _DEC_CODE[Decision.ADAPTED]
            else:
                score_pre = float(model.decision_function(x)[0])
                if score_pre >= 0:
                    model.partial_fit(x)
                    dec_code = _DEC_CODE[Decision.ADAPTED]
                else:
                    dec_code = _DEC_CODE[Decision.ANOMALY]
            latencies[i] = time.perf_counter() - t0
            decisions[i] = dec_code
            scores[i]    = float(model.decision_function(x)[0])
            R_trace[i]   = float(np.sqrt(max(model.R2, 0.0)))

    R_final = float(R_trace[-1])

    # ── 2.5. 모델 저장 ───────────────────────────────────────────────────────
    if save_dir is not None:
        import pathlib
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        model.save(pathlib.Path(save_dir) / "svdd_model")

    # ── 3. 평가 ──────────────────────────────────────────────────────────────
    evaluator = AnomalyDetectionEvaluator()
    y_pred  = (decisions == 2).astype(int)
    y_score = -scores.astype(float)

    cm = evaluator.get_confusion_matrix(y_true, y_pred)
    if int((y_true == 1).sum()) == 0 or int((y_true == 0).sum()) == 0:
        eval_metrics = {"accuracy": float("nan"), "recall": float("nan"),
                        "f1_score": float("nan"), "auc": float("nan")}
    else:
        eval_metrics = evaluator.evaluate(y_true, y_score, y_pred)

    precision = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) > 0 else 0.0

    dec_normal = {
        f"norm_{_DECISION_NAMES[c]}": int(((y_true == 0) & (decisions == c)).sum())
        for c in (0, 1, 2)
    }
    dec_anom = {
        f"anom_{_DECISION_NAMES[c]}": int(((y_true == 1) & (decisions == c)).sum())
        for c in (0, 1, 2)
    }

    R_growth_pct = (R_final - R_pretrain) / R_pretrain * 100.0 if R_pretrain > 0 else float("nan")
    latency_ms = latencies * 1000.0

    return {
        "scenario": scenario_label,
        "kernel": kernel_name,
        "AUC":      float(eval_metrics["auc"]),
        "F1":       float(eval_metrics["f1_score"]),
        "Recall":   float(eval_metrics["recall"]),
        "precision": float(precision),
        "Accuracy": float(eval_metrics["accuracy"]),
        **cm,
        "R_pretrain":   R_pretrain,
        "R_final":      R_final,
        "R_growth_pct": float(R_growth_pct),
        "detection_delay": _detection_delay(y_true, y_pred),
        "n_stream":  int(len(decisions)),
        "n_normal":  int((y_true == 0).sum()),
        "n_anomaly": int((y_true == 1).sum()),
        **dec_normal,
        **dec_anom,
        "R_trace_mean":    float(np.mean(R_trace)),
        "R_trace_std":     float(np.std(R_trace)),
        "latency_mean_ms": float(np.mean(latency_ms)),
        "latency_p50_ms":  float(np.percentile(latency_ms, 50)),
        "latency_p99_ms":  float(np.percentile(latency_ms, 99)),
        "latency_max_ms":  float(np.max(latency_ms)),
        "total_time_s":    float(np.sum(latencies)),
    }
