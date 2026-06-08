"""Anomaly Detection 평가 모듈.

Contents:
  Class
    AnomalyDetectionEvaluator   — TP/TN/FP/FN/precision/recall/F1/AUC + CM 시각화
  Function
    evaluate_ad_performance     — distances.npz scan → AD_performance_all.csv
    summarize_distance_ratios   — distances.npz scan → distance_ratio_summary.csv
    build_rpm_domain_map        — config rpm → domain key 역매핑
  Constant
    KERNELS, _PREP_LABEL
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as _sklearn_roc_auc


KERNELS = ("rbf", "linear")

_PREP_LABEL = {
    "p1_raw":       "P1 Raw",
    "p3_envspec":   "P3 EnvSpec",
    "p4_cepstrum":  "P4 Cepstrum",
    "p6_tds":       "P6 TDS",
    "p7_orderspec": "P7 OrderSpec",
}


# ─────────────────────────────────────────────────────────────────────────────
# AD Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetectionEvaluator:
    """Source/Target AD 성능 평가기.

    모든 metric 은 positive class = anomaly (y_true == 1) 기준.
    y_score 는 anomaly score (클수록 이상) — SVDD distance d(x) 권장.
    AUC 는 sklearn.metrics.roc_auc_score 위임, 나머지는 NumPy-only.

    AUC 계산 주의:
      SVDD decision_function(x) = R² - d²(x) → 양수=정상, 음수=이상.
      이 값을 그대로 anomaly score 로 쓰면 ROC 곡선 역전 → AUC < 0.5.
      반드시 distance d(x) (또는 -decision_function) 를 anomaly score 로 사용.
    """

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[pathlib.Path] = None,
        title: str = "Confusion Matrix",
    ) -> dict[str, int]:
        """TP, TN, FP, FN 계산 및 반환. save_path 지정 시 시각화 저장."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        TP = int(np.sum((y_true == 1) & (y_pred == 1)))
        TN = int(np.sum((y_true == 0) & (y_pred == 0)))
        FP = int(np.sum((y_true == 0) & (y_pred == 1)))
        FN = int(np.sum((y_true == 1) & (y_pred == 0)))

        if save_path is not None:
            cm_mat = np.array([[TN, FP], [FN, TP]], dtype=float)
            row_sums = cm_mat.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm_mat / row_sums, 0.0)

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Normal", "Fault"], fontsize=10)
            ax.set_yticklabels(["Normal", "Fault"], fontsize=10)
            ax.set_xlabel("Predicted label", fontsize=11)
            ax.set_ylabel("Actual label", fontsize=11)
            for i in range(2):
                for j in range(2):
                    rate = cm_norm[i, j]
                    ax.text(
                        j, i, f"{rate:.3}",
                        ha="center", va="center", fontsize=11,
                        color="white" if rate > 0.5 else "black",
                    )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=10, pad=10)

            save_path = pathlib.Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        total = len(y_true)
        return float(np.sum(y_true == y_pred)) / total if total > 0 else 0.0

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm = AnomalyDetectionEvaluator.get_confusion_matrix(y_true, y_pred)
        denom = cm["TP"] + cm["FN"]
        return cm["TP"] / denom if denom > 0 else 0.0

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm = AnomalyDetectionEvaluator.get_confusion_matrix(y_true, y_pred)
        denom = 2 * cm["TP"] + cm["FP"] + cm["FN"]
        return 2 * cm["TP"] / denom if denom > 0 else 0.0

    @staticmethod
    def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """ROC-AUC — sklearn 위임. y_score 는 anomaly score (높을수록 이상)."""
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        if int(np.sum(y_true == 1)) == 0 or int(np.sum(y_true == 0)) == 0:
            return float("nan")
        return float(_sklearn_roc_auc(y_true, y_score))

    def evaluate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """모든 AD 지표 계산 후 dict 반환 (accuracy/recall/f1_score/auc)."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        cm = self.get_confusion_matrix(y_true, y_pred)
        TP, TN, FP, FN = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
        total = TP + TN + FP + FN
        acc = (TP + TN) / total if total > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1  = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        auc_val = self.auc(y_true, y_score)
        return {
            "accuracy": float(acc),
            "recall":   float(rec),
            "f1_score": float(f1),
            "auc":      float(auc_val),
        }



# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_rpm_domain_map(cfg) -> dict[str, dict[int, str]]:
    """config 에서 {dataset: {rpm: domain_key}} 역매핑 생성."""
    result: dict[str, dict[int, str]] = {}
    for dataset in ("cwru",):
        domain_def = cfg.get(f"{dataset}_domain", {})
        result[dataset] = {int(rpm): key for key, rpm in domain_def.items()}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# distances.npz scan → AD_performance / distance_ratio
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ad_performance(
    results_root: pathlib.Path,
    out_dir: pathlib.Path,
    rpm_to_domain: dict | None = None,
) -> pd.DataFrame:
    """AD 성능 평가 — distances.npz 로드 → 지표 계산 → CSV 저장.

    distances.npz 경로: {results_root}/{kernel}/{dataset}/{scenario_id}/{prep_id}/distances.npz
    SVDD radius R 기준 이진 예측: y_pred = (distance > R).astype(int)
    AUC score: distance d(x) 사용 (decision_function = R²-d² 는 역전으로 AUC < 0.5)

    per-run 산출물 (distances.npz 와 같은 폴더):
      confusion_matrix.png  — row-normalized CM 이미지
      confusion_matrix.json — CM 값 + 지표 JSON

    aggregate 산출물 ({out_dir}/):
      AD_performance_all.csv  — run 전체 원본 결과
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv_path = out_dir / "AD_performance_all.csv"

    evaluator = AnomalyDetectionEvaluator()
    raw_rows: list[dict] = []

    for dist_path in sorted(results_root.rglob("distances.npz")):
        rel = dist_path.relative_to(results_root)
        if len(rel.parts) < 5:
            continue
        kernel, dataset, scenario_id, prep_id = (
            rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
        )

        try:
            src_rpm_str, tgt_rpm_str = scenario_id.split("_to_")
            source_rpm, target_rpm = int(src_rpm_str), int(tgt_rpm_str)
        except ValueError:
            source_rpm, target_rpm = None, None

        try:
            data = np.load(dist_path)
        except Exception as e:
            print(f"[eval] skip {dist_path}: {e}")
            continue

        target_normal = data["target_normal"]
        target_fault  = data["target_fault"]
        R = float(data["R"])

        if target_normal.size == 0 or target_fault.size == 0:
            continue

        y_true = np.concatenate([
            np.zeros(len(target_normal), dtype=int),
            np.ones(len(target_fault), dtype=int),
        ])
        y_score = np.concatenate([target_normal, target_fault])
        y_pred  = (y_score > R).astype(int)

        result = evaluator.evaluate(y_true, y_score, y_pred)

        run_dir = dist_path.parent
        cm_title = f"{dataset} | {scenario_id} | {prep_id} ({kernel})"
        cm = evaluator.get_confusion_matrix(
            y_true, y_pred,
            save_path=run_dir / "confusion_matrix.png",
            title=cm_title,
        )
        with open(run_dir / "confusion_matrix.json", "w") as fp:
            json.dump(
                {"kernel": kernel, "dataset": dataset,
                 "scenario_id": scenario_id, "preprocessing": prep_id,
                 **cm, **result},
                fp, indent=2,
            )

        raw_rows.append({
            "kernel": kernel, "dataset": dataset,
            "scenario_id": scenario_id,
            "source_rpm": source_rpm, "target_rpm": target_rpm,
            "preprocessing": prep_id,
            **cm, **result,
        })

    if not raw_rows:
        print("[eval] no distances.npz found — skipping evaluation")
        return pd.DataFrame()

    raw_df = pd.DataFrame(raw_rows)

    full_df = raw_df.rename(columns={
        "scenario_id": "scenario", "preprocessing": "prep",
        "accuracy": "Accuracy", "recall": "Recall", "f1_score": "F1", "auc": "AUC",
    })
    full_df["prep_label"] = full_df["prep"].map(_PREP_LABEL).fillna(full_df["prep"])
    if rpm_to_domain is not None:
        full_df["source"] = full_df.apply(
            lambda r: rpm_to_domain.get(r["dataset"], {}).get(r["source_rpm"], str(r["source_rpm"])),
            axis=1,
        )
        full_df["target"] = full_df.apply(
            lambda r: rpm_to_domain.get(r["dataset"], {}).get(r["target_rpm"], str(r["target_rpm"])),
            axis=1,
        )
    else:
        full_df["source"] = full_df["source_rpm"].astype(str)
        full_df["target"] = full_df["target_rpm"].astype(str)
    full_df = full_df[[
        "kernel", "dataset", "scenario", "source", "target",
        "prep", "prep_label", "AUC", "Recall", "F1", "Accuracy", "TP", "TN", "FP", "FN",
    ]]
    full_df.to_csv(full_csv_path, index=False, float_format="%.6f")

    print(f"[eval] {len(raw_rows)} scenarios evaluated → {full_csv_path}")
    return full_df


def summarize_distance_ratios(
    results_root: pathlib.Path,
    out_dir: pathlib.Path,
    rpm_to_domain: dict | None = None,
) -> pd.DataFrame:
    """distances.npz → 그룹별 d/R 통계 CSV 생성.

    산출물 ({out_dir}/):
      distance_ratio_summary.csv — (kernel, dataset, scenario, prep) × 그룹별 d/R 통계
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "distance_ratio_summary.csv"

    _GROUPS = ("source_train", "source_val", "target_normal", "target_fault")
    _PCTS = (5, 25, 50, 75, 95)

    rows: list[dict] = []

    for dist_path in sorted(results_root.rglob("distances.npz")):
        rel = dist_path.relative_to(results_root)
        if len(rel.parts) < 5:
            continue
        kernel, dataset, scenario_id, prep_id = (
            rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
        )

        try:
            src_rpm_str, tgt_rpm_str = scenario_id.split("_to_")
            source_rpm, target_rpm = int(src_rpm_str), int(tgt_rpm_str)
        except ValueError:
            source_rpm, target_rpm = None, None

        try:
            data = np.load(dist_path)
        except Exception as e:
            print(f"[ratio] skip {dist_path}: {e}")
            continue

        R = float(data["R"])
        if R <= 0:
            continue

        row: dict = {
            "kernel": kernel, "dataset": dataset,
            "scenario_id": scenario_id,
            "source_rpm": source_rpm, "target_rpm": target_rpm,
            "preprocessing": prep_id,
            "R": R,
        }

        for group in _GROUPS:
            d = data[group] if group in data else np.empty(0)
            if d.size == 0:
                row[f"{group}_n"] = 0
                for stat in ("mean", "std"):
                    row[f"{group}_{stat}_dR"] = None
                for p in _PCTS:
                    row[f"{group}_p{p}_dR"] = None
                continue
            dR = d / R
            row[f"{group}_n"] = len(dR)
            row[f"{group}_mean_dR"] = float(np.mean(dR))
            row[f"{group}_std_dR"]  = float(np.std(dR))
            for p in _PCTS:
                row[f"{group}_p{p}_dR"] = float(np.percentile(dR, p))

        rows.append(row)

    if not rows:
        print("[ratio] no distances.npz found — skipping")
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

    front = ["kernel", "dataset", "scenario", "preprocessing", "R"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    df.to_csv(save_path, index=False, float_format="%.4f")
    print(f"[ratio] {len(rows)} runs → {save_path}")
    return df
