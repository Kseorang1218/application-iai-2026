"""DualBoundarySVDD — Inner/Outer boundary 기반 selective adaptation.

본 논문의 핵심 제안:
    pre-trained SVDD hypersphere 위에 두 경계를 두어 스트림 샘플을
    세 영역으로 분기. boundary 는 pre-train 완료 시점의 R 에 대한 ratio
    (rho_inner, rho_outer) 로 한 번 고정되며, streaming 중 partial_fit 으로
    R 이 갱신되더라도 절대 경계값은 변하지 않는다.

       d² < r_inner_fixed²   → NORMAL_SKIP   (이미 잘 학습된 영역)
       d² > r_outer_fixed²   → ANOMALY       (외부, 탐지만)
       otherwise             → ADAPTED       (Adaptation Zone, partial_fit)

고정 경계 산출 (fit() 완료 직후 1회):
    r_inner_fixed = rho_inner · R_pretrain
    r_outer_fixed = rho_outer · R_pretrain

조건: 0 ≤ rho_inner < 1 < rho_outer 권장 (R 안쪽 / 바깥쪽 의미).
실제 코드 상에선 0 ≤ rho_inner < rho_outer 만 enforce.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from typing import Any

from .kernels import RBFKernel
from .online_svdd import OnlineSVDD


class Decision(Enum):
    """Stream sample dispatch 결과."""
    NORMAL_SKIP = "normal_skip"   # d² < R_inner²
    ADAPTED = "adapted"           # R_inner² ≤ d² ≤ R_outer²
    ANOMALY = "anomaly"           # d² > R_outer²


class DualBoundarySVDD(OnlineSVDD):
    """OnlineSVDD + dual-boundary selective adaptation (R-ratio 기반).

    Boundary 는 R 에 대한 ratio 로 정의되어, partial_fit 으로 R 이 갱신되면
    절대 거리 기준의 boundary 도 자동으로 이동한다.

    Parameters
    ----------
    kernel, C, buffer_size, max_iter, tol : `OnlineSVDD` 와 동일.
    rho_inner : float
        내부 경계 비율. d² < (rho_inner · R)² 인 샘플은 업데이트 skip.
        0 ≤ rho_inner < rho_outer.
    rho_outer : float
        외부 경계 비율. d² > (rho_outer · R)² 인 샘플은 anomaly, 업데이트 skip.
    n_target_normal, warmup_ratio : int, float
        초기 n_warmup = int(n_target_normal · warmup_ratio) 샘플은 거리와
        무관하게 ADAPTED 처리. (현재 oracle leakage 보류 상태)
    """

    def __init__(
        self,
        kernel: Any,
        C: float,
        buffer_size: int,
        rho_inner: float,
        rho_outer: float,
        max_iter: int = 1000,
        tol: float = 1e-3,
        n_target_normal: int = 0,
        warmup_ratio: float = 0.0,
        max_inner_iter: int = 10,
    ):
        super().__init__(
            kernel, C, buffer_size, max_iter, tol,
            max_inner_iter=max_inner_iter,
        )
        if not (0.0 <= rho_inner < rho_outer):
            raise ValueError(
                f"need 0 ≤ rho_inner < rho_outer, got {rho_inner}, {rho_outer}"
            )
        self.rho_inner: float = float(rho_inner)
        self.rho_outer: float = float(rho_outer)
        self._rho_inner_sq: float = self.rho_inner ** 2
        self._rho_outer_sq: float = self.rho_outer ** 2
        self.n_target_normal: int = int(n_target_normal)
        self.warmup_ratio: float = float(warmup_ratio)
        self.n_warmup: int = int(self.n_target_normal * self.warmup_ratio)
        self.sample_count: int = 0

        # pre-train 완료 후 fit() 에서 한 번만 계산·고정됨
        self._r_inner_sq_fixed: float | None = None
        self._r_outer_sq_fixed: float | None = None

    # --------------------------------------------------------------- fit ---
    def fit(self, X: np.ndarray) -> "DualBoundarySVDD":
        """Batch SMO 학습 (parent OnlineSVDD) 후 고정 경계 산출.

        pre-train 완료 직후의 R 을 기준으로 r_inner_fixed / r_outer_fixed 를
        절대값으로 한 번만 계산·저장한다. 이후 streaming 중 R 이 갱신되더라도
        경계는 변하지 않는다.
        """
        super().fit(X)
        R_pretrain = float(np.sqrt(max(self.R2, 0.0)))
        self._r_inner_sq_fixed = (self.rho_inner * R_pretrain) ** 2
        self._r_outer_sq_fixed = (self.rho_outer * R_pretrain) ** 2
        return self

    # ------------------------------------------------------------- process ---
    def process(self, x: np.ndarray) -> Decision:
        """단일 샘플 dispatch.

        Boundary 비교는 pre-train 완료 시점에 고정된 절대 임계값을 사용:
            r_inner_fixed = rho_inner · R_pretrain   (fit() 에서 1회 산출)
            r_outer_fixed = rho_outer · R_pretrain

        streaming 중 partial_fit 으로 R 이 갱신되더라도 경계는 이동하지 않는다.

        Returns
        -------
        Decision
            NORMAL_SKIP / ANOMALY → 모델 변경 없음.
            ADAPTED               → `partial_fit(x)` 수행 후 반환.
        """
        if self._X_buf is None:
            raise RuntimeError("DualBoundarySVDD must be fit() before process().")
        if self._r_inner_sq_fixed is None or self._r_outer_sq_fixed is None:
            raise RuntimeError(
                "Fixed boundaries not set. Call fit() before process()."
            )
        x_2d = np.asarray(x, dtype=np.float32).reshape(1, -1)

        # 초기 n_warmup 샘플은 거리와 무관하게 정상으로 가정하고 무조건 업데이트
        if self.sample_count < self.n_warmup:
            self.sample_count += 1
            self.partial_fit(x_2d)
            return Decision.ADAPTED

        self.sample_count += 1
        d2 = float(self._distance_sq(x_2d)[0])

        # pre-train R 기준으로 고정된 경계 사용 (streaming 중 R 갱신과 무관)
        if d2 < self._r_inner_sq_fixed:
            return Decision.NORMAL_SKIP
        if d2 > self._r_outer_sq_fixed:
            return Decision.ANOMALY
        # adaptation zone: partial_fit 으로 R 은 갱신되지만 경계는 고정
        self.partial_fit(x_2d)
        return Decision.ADAPTED


# =================================== sanity check ==================================
if __name__ == "__main__":
    import sys
    from collections import Counter
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from sklearn.metrics import roc_auc_score

    from models.kernels import RBFKernel       # type: ignore
    from models.online_svdd import OnlineSVDD  # type: ignore
    from models.svdd import SVDD               # type: ignore

    rng = np.random.default_rng(0)
    n_normal, n_anom = 200, 50
    X_normal = rng.standard_normal((n_normal, 2)) * 0.7
    theta = rng.uniform(0, 2 * np.pi, n_anom)
    r = rng.uniform(3.0, 5.0, n_anom)
    X_anom = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    X_test = np.vstack([X_normal, X_anom])
    y_test = np.concatenate([np.ones(n_normal), -np.ones(n_anom)])

    # init: 50 normal (베이스라인 학습), stream: 150 normal + 50 anom
    n_init = 50
    X_init = X_normal[:n_init]
    X_stream_normal = X_normal[n_init:]
    X_stream = np.vstack([X_stream_normal, X_anom])
    y_stream_kind = (
        ["normal"] * len(X_stream_normal) + ["anomaly"] * len(X_anom)
    )

    gamma, C = 0.5, max(0.05, 1.1 / n_init)

    print("===== DualBoundarySVDD sanity check =====")
    print(f"n_init={n_init}, n_stream(normal+anom)={len(X_stream)}, gamma={gamma}, C={C:.4f}")

    # ---- vanilla SVDD baseline (init only, no adaptation) ----
    base = SVDD(kernel=RBFKernel(gamma=gamma), C=C, max_iter=2000, tol=1e-4)
    base.fit(X_init)
    auc_base = roc_auc_score(
        (y_test == 1).astype(int), base.decision_function(X_test)
    )

    # init data 거리 분포의 25%/75% 분위수를 R 로 나눠 rho 결정
    d2_init = base._distance_sq(X_init)
    d_init = np.sqrt(np.maximum(d2_init, 0.0))
    R_init = float(np.sqrt(max(base.R2, 0.0)))
    rho_inner = float(np.quantile(d_init, 0.25)) / max(R_init, 1e-12)
    rho_outer = float(np.quantile(d_init, 0.75)) / max(R_init, 1e-12)
    print(f"\n[thresholds] rho_inner={rho_inner:.4f}  rho_outer={rho_outer:.4f}  "
          f"(25/75 percentile of training d / R)")
    print(f"             R²={base.R2:.4f}  →  R={R_init:.4f}")

    # ---- OnlineSVDD reference: 모든 샘플을 partial_fit (selective 없음) ----
    online = OnlineSVDD(
        kernel=RBFKernel(gamma=gamma), C=C, buffer_size=200, max_iter=2000, tol=1e-4
    )
    online.fit(X_init)
    for x in X_stream:
        online.partial_fit(x)
    auc_online = roc_auc_score(
        (y_test == 1).astype(int), online.decision_function(X_test)
    )

    # ---- DualBoundarySVDD ----
    # [검증] n_target_normal=150 (정상 샘플 수), warmup_ratio=0.10을 설정하여
    # 내부적으로 n_warmup = 150 * 0.10 = 15개 강제 업데이트 검증
    n_target_normal_val = 150
    warmup_ratio_val = 0.10
    n_warmup_val = int(n_target_normal_val * warmup_ratio_val)  # 15

    dbs = DualBoundarySVDD(
        kernel=RBFKernel(gamma=gamma), C=C, buffer_size=200,
        rho_inner=rho_inner, rho_outer=rho_outer, max_iter=2000, tol=1e-4,
        n_target_normal=n_target_normal_val,
        warmup_ratio=warmup_ratio_val,
    )
    dbs.fit(X_init)

    decisions: list[tuple[str, Decision]] = []
    for idx, (kind, x) in enumerate(zip(y_stream_kind, X_stream)):
        decision = dbs.process(x)
        decisions.append((kind, decision))
        if idx < n_warmup_val:
            # 초기 n_warmup개 샘플은 거리 분포상 NORMAL_SKIP이나 ANOMALY에 걸리더라도
            # 반드시 ADAPTED가 반환되어 무조건 업데이트가 일어났어야 함.
            assert decision == Decision.ADAPTED, f"warmup index {idx} failed to adapt (got {decision})"

    # sample_count가 전체 스트림 길이만큼 카운트되었는지 확인
    assert dbs.sample_count == len(X_stream), f"sample count mismatch: {dbs.sample_count} vs {len(X_stream)}"

    auc_dbs = roc_auc_score(
        (y_test == 1).astype(int), dbs.decision_function(X_test)
    )

    # ---- 분기 카운트 ----
    by_group: dict[str, Counter] = {"normal": Counter(), "anomaly": Counter()}
    for kind, dec in decisions:
        by_group[kind][dec] += 1

    print(f"\n[dispatch counts (n_warmup={n_warmup_val})]")
    print(f"  {'group':<10}  {'NORMAL_SKIP':>12}  {'ADAPTED':>10}  {'ANOMALY':>10}  {'total':>6}")
    for kind in ("normal", "anomaly"):
        c = by_group[kind]
        total = sum(c.values())
        print(f"  {kind:<10}  {c[Decision.NORMAL_SKIP]:>12}  "
              f"{c[Decision.ADAPTED]:>10}  {c[Decision.ANOMALY]:>10}  {total:>6}")

    print(f"\n[AUROC]")
    print(f"  vanilla SVDD (init only)   = {auc_base:.4f}")
    print(f"  OnlineSVDD (adapt all)     = {auc_online:.4f}   (Δ={auc_online-auc_base:+.4f})")
    print(f"  DualBoundarySVDD           = {auc_dbs:.4f}   (Δ={auc_dbs-auc_base:+.4f})")

    # ----- assertions -----
    # 결함 샘플은 대부분 ANOMALY 로 분류되어야 (외곽이라 d² 큼)
    anom_anomaly_rate = by_group["anomaly"][Decision.ANOMALY] / max(1, sum(by_group["anomaly"].values()))
    print(f"\n[check] anomaly→ANOMALY rate = {anom_anomaly_rate:.2%}  (expect ≥ 0.95)")
    assert anom_anomaly_rate >= 0.95, "Most anomaly samples should be classified as ANOMALY."

    # 정상 stream 중 ANOMALY 비율: r_outer 가 75% 분위수라 정의상 ~25% 가 자연 발생.
    # sampling variance 고려해 ≤ 0.45 까지 허용 (그 이상이면 r_outer/모델 이상)
    normal_anomaly_rate = by_group["normal"][Decision.ANOMALY] / max(1, sum(by_group["normal"].values()))
    print(f"[check] normal→ANOMALY rate  = {normal_anomaly_rate:.2%}  (expect ≤ 0.45 @ 75-pct r_outer)")
    assert normal_anomaly_rate <= 0.45, "Too many normal samples flagged as ANOMALY."

    # AUROC 가 base 대비 크게 떨어지면 안 됨
    print(f"[check] |AUROC(dbs) - AUROC(base)| = {abs(auc_dbs - auc_base):.4f}  (expect ≤ 0.05)")
    assert abs(auc_dbs - auc_base) <= 0.05, "DualBoundary AUROC drifted vs baseline."

    print("\nAll DualBoundarySVDD sanity checks passed.")
