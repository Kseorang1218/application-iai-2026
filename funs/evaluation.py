"""Anomaly Detection 평가 모듈."""
from __future__ import annotations

import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score as _sklearn_roc_auc


KERNELS = ("rbf", "linear")


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



def build_rpm_domain_map(cfg) -> dict[str, dict[int, str]]:
    """config 에서 {dataset: {rpm: domain_key}} 역매핑 생성."""
    result: dict[str, dict[int, str]] = {}
    for dataset in ("cwru", "pu"):
        domain_def = cfg.get(dataset, {}).get("domains", {})
        result[dataset] = {int(rpm): key for key, rpm in domain_def.items()}
    return result
