"""OTTA streaming 결과 사후 분석 — `analysis.py` 의 OTTA 버전.

CLI:
  python analysis_otta.py --results-root results/{date} [--out-dir DIR]

I/O 요약:
  reads : ROOT/{kernel}/{ds}/{sc}/{prep}/otta_stream.npz
  writes: ROOT/evaluation/otta_performance_all.csv
          ROOT/{kernel}/{ds}/{sc}/R_trace.png
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import funs
from funs.evaluation import AnomalyDetectionEvaluator, KERNELS, build_rpm_domain_map


_DECISION_NAMES = {0: "NORMAL_SKIP", 1: "ADAPTED", 2: "ANOMALY"}


def _detection_delay(y_true: np.ndarray, y_pred: np.ndarray) -> int | None:
    """첫 anomaly 발생 인덱스부터 첫 정확 탐지까지의 거리.

    Returns
    -------
    None : stream 에 anomaly 가 없음.
    -1   : anomaly 가 있는데 한 번도 탐지하지 못함.
    >=0  : 탐지까지 걸린 샘플 수 (0 = 즉시 탐지).
    """
    anom_idx = np.where(y_true == 1)[0]
    if anom_idx.size == 0:
        return None
    first_anom = int(anom_idx[0])
    det_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    if det_idx.size == 0:
        return -1
    return int(det_idx[0]) - first_anom


def evaluate_otta_performance(
        results_root: pathlib.Path,
        out_dir: pathlib.Path,
        rpm_to_domain: dict | None = None,
        kernel_filter: str | None = None,
    ) -> pd.DataFrame:
    """모든 `otta_stream.npz` 스캔 → OTTA 평가 dataframe.

    kernel_filter 지정 시 해당 커널만 처리.

    per-run 산출물 (otta_stream.npz 와 같은 폴더):
      confusion_matrix_otta.png   — streaming decisions 기반 CM
      confusion_matrix_otta.json  — CM + 지표 JSON

    aggregate 산출물 ({out_dir}/):
      otta_performance_all.csv    — 전체 run × OTTA 지표

    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / "otta_performance_all.csv"

    evaluator = AnomalyDetectionEvaluator()
    rows: list[dict] = []

    for stream_path in sorted(results_root.rglob("otta_stream.npz")):
        rel = stream_path.relative_to(results_root)
        if len(rel.parts) == 5:
            kernel, dataset, scenario_id, prep_id = (
                rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
            )
        elif len(rel.parts) == 4:
            kernel = "linear"
            dataset, scenario_id, prep_id = rel.parts[0], rel.parts[1], rel.parts[2]
        elif len(rel.parts) == 3:
            kernel = "linear"
            dataset, scenario_id = rel.parts[0], rel.parts[1]
            prep_id = "p4_cepstrum"
        else:
            continue
        if kernel_filter is not None and kernel != kernel_filter:
            continue

        try:
            src_rpm_str, tgt_rpm_str = scenario_id.split("_to_")
            source_rpm, target_rpm = int(src_rpm_str), int(tgt_rpm_str)
        except ValueError:
            source_rpm, target_rpm = None, None

        try:
            with np.load(stream_path) as data:
                decisions  = data["decisions"]
                scores     = data["scores"]
                y_true     = data["y_true"].astype(int)
                latencies  = data["latencies"]
                R_trace    = data["R_trace"]
                R_pretrain = float(data["R_pretrain"])
                R_final    = float(data["R_final"])
        except Exception as e:
            print(f"[otta] skip {stream_path}: {e}")
            continue

        # ── AD 지표 (streaming decision 기반) ─────────────────────────
        # anomaly score = -decision_function (높을수록 anomaly)
        y_pred  = (decisions == 2).astype(int)
        y_score = -scores.astype(float)

        cm = evaluator.get_confusion_matrix(y_true, y_pred)
        if int((y_true == 1).sum()) == 0 or int((y_true == 0).sum()) == 0:
            metrics = {"accuracy": float("nan"), "recall": float("nan"),
                       "f1_score": float("nan"), "auc": float("nan")}
        else:
            metrics = evaluator.evaluate(y_true, y_score, y_pred)

        precision = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) > 0 else 0.0

        # ── decision rate by true label ────────────────────────────────
        dec_normal = {
            f"norm_{_DECISION_NAMES[c]}":
                int(((y_true == 0) & (decisions == c)).sum())
            for c in (0, 1, 2)
        }
        dec_anom = {
            f"anom_{_DECISION_NAMES[c]}":
                int(((y_true == 1) & (decisions == c)).sum())
            for c in (0, 1, 2)
        }

        # ── R 변화 ────────────────────────────────────────────────────
        R_growth_pct = (
            (R_final - R_pretrain) / R_pretrain * 100.0
            if R_pretrain > 0 else float("nan")
        )

        # ── latency ──────────────────────────────────────────────────
        latency_ms = latencies * 1000.0

        rows.append({
            "kernel": kernel, "dataset": dataset,
            "scenario_id": scenario_id,
            "source_rpm": source_rpm, "target_rpm": target_rpm,
            "preprocessing": prep_id,
            **cm,
            "precision": float(precision),
            "Recall":    float(metrics["recall"]),
            "F1":        float(metrics["f1_score"]),
            "AUC":       float(metrics["auc"]),
            "Accuracy":  float(metrics["accuracy"]),
            "n_stream":  int(len(decisions)),
            "n_normal":  int((y_true == 0).sum()),
            "n_anomaly": int((y_true == 1).sum()),
            **dec_normal,
            **dec_anom,
            "R_pretrain":   R_pretrain,
            "R_final":      R_final,
            "R_growth_pct": float(R_growth_pct),
            "R_trace_mean": float(R_trace.mean()),
            "R_trace_std":  float(R_trace.std()),
            "latency_mean_ms": float(latency_ms.mean()),
            "latency_p50_ms":  float(np.percentile(latency_ms, 50)),
            "latency_p99_ms":  float(np.percentile(latency_ms, 99)),
            "latency_max_ms":  float(latency_ms.max()),
            "total_time_s":    float(latencies.sum()),
            "detection_delay": _detection_delay(y_true, y_pred),
        })

    if not rows:
        print("[otta] no otta_stream.npz found — skipping")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ["kernel", "dataset", "scenario_id", "preprocessing"]
    )

    if rpm_to_domain is not None:
        df["source"] = df.apply(
            lambda r: rpm_to_domain.get(r["dataset"], {}).get(r["source_rpm"], str(r["source_rpm"])),
            axis=1,
        )
        df["target"] = df.apply(
            lambda r: rpm_to_domain.get(r["dataset"], {}).get(r["target_rpm"], str(r["target_rpm"])),
            axis=1,
        )
        df["scenario"] = df["source"] + "->" + df["target"]
    else:
        df["scenario"] = df["scenario_id"]

    front = [
        "kernel", "dataset", "scenario", "preprocessing",
        "AUC", "F1", "Recall", "precision", "Accuracy",
        "TP", "TN", "FP", "FN",
        "R_pretrain", "R_final", "R_growth_pct",
        "detection_delay",
        "latency_mean_ms", "latency_p99_ms",
    ]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    df.to_csv(full_csv, index=False, float_format="%.6f")
    print(f"[otta] {len(rows)} runs evaluated → {full_csv}")
    return df


_PREP_ORDER = ("p4_cepstrum",)
_PREP_LABEL = {"p4_cepstrum": "Cepstrum"}
_KERNEL_LABEL = {"rbf": "RBF", "linear": "Linear", "poly": "Poly"}


def _scenario_label(dataset: str, scenario_id: str,
                    rpm_domain_map: dict | None) -> str:
    """'1797_to_1772' → 'A→B'. 매핑 없으면 원본 반환."""
    if rpm_domain_map is None:
        return scenario_id
    ds_map = rpm_domain_map.get(dataset, {})
    try:
        src_str, tgt_str = scenario_id.split("_to_")
        src = ds_map.get(int(src_str), src_str)
        tgt = ds_map.get(int(tgt_str), tgt_str)
        return f"{src}→{tgt}"
    except ValueError:
        return scenario_id


def plot_R_trace_by_scenario(
        results_root: pathlib.Path,
        rpm_domain_map: dict | None = None,
    ) -> None:
    """시나리오별 R_trace 2x2 panel (4 prep).

    저장: {kernel}/{ds}/{sc}/R_trace.png
    표시 항목:
      - 분홍 음영: anomaly 구간 (y_true == 1)
      - 연두 음영: warmup 구간 (초기 n_warmup 샘플)
      - 녹색 점:  ADAPTED 결정 (adaptation 발생 위치)
      - 빨간 점:  False Negative (고장인데 정상으로 판단)
    """
    results_root = pathlib.Path(results_root)
    groups: dict[tuple[str, str, str], list[tuple[str, pathlib.Path]]] = {}
    for stream_path in sorted(results_root.rglob("otta_stream.npz")):
        rel = stream_path.relative_to(results_root)
        if len(rel.parts) == 5:
            kernel, dataset, scenario_id, prep_id = (
                rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
            )
        elif len(rel.parts) == 4:
            kernel = "linear"
            dataset, scenario_id, prep_id = rel.parts[0], rel.parts[1], rel.parts[2]
        elif len(rel.parts) == 3:
            kernel = "linear"
            dataset, scenario_id = rel.parts[0], rel.parts[1]
            prep_id = "p4_cepstrum"
        else:
            continue
        groups.setdefault((kernel, dataset, scenario_id), []).append((prep_id, stream_path))

    for (kernel, dataset, scenario_id), runs in groups.items():
        runs_by_prep = {p: pth for p, pth in runs}
        preps_present = [p for p in _PREP_ORDER if p in runs_by_prep]
        if not preps_present:
            continue

        sc_label = _scenario_label(dataset, scenario_id, rpm_domain_map)
        kernel_label = _KERNEL_LABEL.get(kernel, kernel.upper())

        n_plots = len(preps_present)
        n_cols = min(n_plots, 3)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), sharex=False)
        axes_flat = np.array(axes).flatten()

        # warmup N은 첫 prep에서 한 번만 읽음 (시나리오 내 공통값)
        n_warmup = 0
        if preps_present:
            metrics_path = runs_by_prep[preps_present[0]].parent / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    n_warmup = json.load(f).get("n_warmup", 0)

        for ax, prep in zip(axes_flat, preps_present):
            stream_path = runs_by_prep[prep]
            with np.load(stream_path) as data:
                R_trace    = data["R_trace"]
                y_true     = data["y_true"]
                decisions  = data["decisions"]
                R_pretrain = float(data["R_pretrain"])
            x = np.arange(len(R_trace))

            anom_idx = np.where(y_true == 1)[0]
            if anom_idx.size > 0:
                ax.axvspan(anom_idx[0], anom_idx[-1], color="red", alpha=0.08)
            if n_warmup > 0:
                ax.axvspan(0, n_warmup - 1, color="green", alpha=0.10)

            ax.plot(x, R_trace, color="C0", lw=1.0)
            ax.axhline(R_pretrain, color="gray", ls="--", lw=0.8,
                       label=f"$R_{{pre}}$={R_pretrain:.3f}")

            # ADAPTED — label 없음 (figure legend로 이동)
            an_idx = np.where((decisions == 1) & (y_true == 0))[0]
            af_idx = np.where((decisions == 1) & (y_true == 1))[0]
            if an_idx.size > 0:
                ax.scatter(an_idx, R_trace[an_idx], color="green",
                           s=4, alpha=0.5, zorder=4)
            if af_idx.size > 0:
                ax.scatter(af_idx, R_trace[af_idx], color="orange",
                           s=8, alpha=0.8, zorder=5)

            ax.set_title(_PREP_LABEL.get(prep, prep), fontsize=13)
            ax.set_xlabel("Sample index", fontsize=11)
            ax.set_ylabel("R (radius)", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=11, loc="best")

        for ax in axes_flat[len(preps_present):]:
            ax.axis("off")

        # 공유 figure legend — 4개 항목, 상단 한 줄
        shared_handles = [
            mpatches.Patch(color="red",    alpha=0.25, label="Anomaly region"),
            mpatches.Patch(color="green",  alpha=0.30, label=f"Warmup (N={n_warmup})"),
            mlines.Line2D([], [], color="green",  marker="o", ms=6,
                          ls="None", alpha=0.7, label="Adapted-Normal"),
            mlines.Line2D([], [], color="orange", marker="o", ms=6,
                          ls="None", alpha=0.9, label="Adapted-Fault"),
        ]
        fig.suptitle(
            f"{dataset.upper()} | {sc_label} ({kernel_label})", fontsize=13
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        fig.legend(handles=shared_handles, loc="upper center", ncol=4,
                   fontsize=11, bbox_to_anchor=(0.5, 0.95),
                   frameon=True, framealpha=0.8)

        kroot = results_root / kernel if (results_root / kernel).is_dir() else results_root
        out_path = kroot / dataset / scenario_id / "R_trace.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"[R_trace] {len(groups)} scenarios processed")


def run_analysis_otta(
    results_root: pathlib.Path,
    out_dir: pathlib.Path | None = None,
    kernel: str | None = None,
) -> None:
    results_root = pathlib.Path(results_root)
    eval_dir = out_dir or (results_root / "evaluation")
    cfg = funs.get_config()
    rpm_to_domain = build_rpm_domain_map(cfg)

    if kernel:
        kernels = [kernel]
    else:
        kernels = [k.name for k in sorted(results_root.iterdir()) if k.is_dir() and k.name in KERNELS]
        # 커널 서브디렉토리 없이 데이터가 바로 있는 경우 (flat 구조)
        if not kernels and any(results_root.rglob("otta_stream.npz")):
            kernels = ["linear"]

    for k in kernels:
        # flat 구조면 results_root 자체를 kernel root로 사용
        if not (results_root / k).exists() and not any(results_root.rglob("otta_stream.npz")):
            print(f"[otta] {k} 폴더 없음 — skip")
            continue

        print(f"\n{'='*50}\n[otta] kernel={k}\n{'='*50}")

        evaluate_otta_performance(
            results_root, eval_dir,
            rpm_to_domain=rpm_to_domain, kernel_filter=k,
        )

    plot_R_trace_by_scenario(results_root, rpm_domain_map=rpm_to_domain)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OTTA streaming 결과 사후 분석 (analysis.py 의 OTTA 버전)"
    )
    parser.add_argument(
        "--results-root", type=pathlib.Path, required=True,
        help="run.py 산출물이 들어 있는 루트 (예: results/0520).",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=None,
        help="출력 디렉토리. 미지정 시 `<results-root>/evaluation`",
    )
    parser.add_argument(
        "--kernel", type=str, default=None, choices=KERNELS,
        help="지정 시 해당 커널만 처리. 미지정 시 results-root 에서 발견된 모든 커널 처리.",
    )
    args = parser.parse_args()
    run_analysis_otta(
        results_root=args.results_root,
        out_dir=args.out_dir,
        kernel=args.kernel,
    )


if __name__ == "__main__":
    main()
