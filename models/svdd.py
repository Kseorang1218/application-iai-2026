"""Vanilla SVDD — batch SMO 학습 + dual-form 추론.

Reference: Tax & Duin, "Support Vector Data Description" (2004).

Dual problem (soft-margin):
    max_α   Σ_i α_i K(x_i, x_i) - Σ_{ij} α_i α_j K(x_i, x_j)
    s.t.    0 ≤ α_i ≤ C,   Σ_i α_i = 1

(RBF 커널은 K(x, x) = 1 이므로 첫 항이 상수가 되어
 max_α  - α^T K α 와 동등.)

추론 (dual form):
    d²(x) = K(x, x) - 2 Σ_i α_i K(x, x_i) + α^T K_sv α
    decision_function(x) = R² - d²(x)        (≥ 0: 정상, < 0: 이상)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as xp
import numpy as np

from typing import Any

from .kernels import RBFKernel

# float32 연산에서 의미 있는 최소 임계값 (float32 eps ≈ 1.19e-7, ×10 ≈ 1.2e-6)
EPS_F32: float = float(np.finfo(np.float32).eps) * 10


def _to_numpy(arr) -> np.ndarray:
    """CuPy 배열이면 CPU로 이동, numpy이면 그대로 반환."""
    return np.asarray(arr)


# 공통 R² 산출 헬퍼 (OnlineSVDD 도 import 해서 사용)
def _compute_R2(
    d2,
    alpha,
    C: float,
    eps: float = EPS_F32,
) -> float:
    """R² fallback 3-tier.

    Tier 1 (표준)  : margin SV (eps < α < C-eps) 의 d² 평균.
    Tier 2 (fallback): active SV (α > eps, outlier 포함) 의 d² max.
                       α=0 빈 버퍼 슬롯은 제외 (R² 오염 방지).
                       전부 outlier(α=C)인 경우를 처리.
    Tier 3 (robust): active SV 가 없는 극단 케이스 (전부 α=0 또는 α≈0).
                     d² 의 95-quantile. SMO 수렴 실패 or C 과소 설정 의심.

    이력:
    - C3 (2026-05-12): Tier 1 → max(d² of ALL SV) 단일 fallback 을 3-tier 로 분리.
    - 후속 fix       : Tier 2 mask 에서 α=0 빈 슬롯 제외 (OnlineSVDD 오염 차단).
    - 2026-05-28    : Tier 2 mask 버그 수정 (Tier 1 과 동일한 조건 복붙 → alpha > eps 로 교정).
                      Tier 3 발동 시 warnings.warn 추가.
    """
    margin_mask = (alpha > eps) & (alpha < C - eps)
    if margin_mask.any():
        return float(d2[margin_mask].mean())
    # Tier 2: active SV (α > eps). outlier(α=C) 포함, α=0 빈 슬롯만 제외.
    active_mask = alpha > eps
    if active_mask.any():
        return float(d2[active_mask].max())
    # Tier 3: active SV 가 없음 — SMO 수렴 실패 또는 C 과소 설정 의심.
    warnings.warn(
        "_compute_R2: no active SV (all α≈0). "
        "R² falls back to 95th-percentile of d² — check C and SMO convergence.",
        RuntimeWarning,
        stacklevel=2,
    )
    return float(xp.quantile(d2, 0.95))


class SVDD:
    """Batch SMO 기반 SVDD.

    Parameters
    ----------
    kernel : Any
        커널 함수 객체 (RBFKernel, LinearKernel, PolyKernel 등).
    C : float
        soft-margin 파라미터. C * N ≥ 1 이어야 sum α = 1 이 feasible.
    max_iter : int
        SMO 외부 반복 상한.
    tol : float
        KKT 위반 허용치 (gap = max g - min g ≤ tol 시 수렴).

    Attributes (after fit)
    ----------------------
    X_sv : (n_sv, d)
    alpha : (n_sv,)
    R2 : float
    """

    def __init__(
        self,
        kernel: Any,
        C: float,
        max_iter: int = 1000,
        tol: float = 1e-3,
    ):
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")
        self.kernel = kernel
        self.C: float = float(C)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)

        self.X_sv = None
        self.alpha = None
        self.R2: float | None = None
        self._center_norm_sq: float | None = None  # α^T K_sv α (재추론 시 상수)
        self.n_iter_: int = 0

    # --------------------------------------------------------------- fit ---
    def fit(self, X) -> "SVDD":
        """Batch SMO 학습.

        SMO inner step (maximum violating pair):
            i = argmax_{α_i < C} g_i,   j = argmin_{α_j > 0} g_j
            여기서 g_l = K_ll - 2 (Kα)_l = ∂W/∂α_l.
            gap = g_i - g_j ≤ tol 이면 KKT 만족 → 수렴.

        Sub-problem (α_i + α_j 보존, α_i ← α_i + Δ, α_j ← α_j - Δ):
            ΔW = (g_i - g_j) Δ - η Δ²,   η = K_ii + K_jj - 2 K_ij
            Δ_unc = (g_i - g_j) / (2 η)
            Δ ∈ [max(-α_i, α_j - C), min(C - α_i, α_j)]

        f = Kα 캐시는 한 행만 업데이트:
            f_l ← f_l + Δ (K_il - K_jl)
        """
        X = xp.asarray(X, dtype=xp.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        N = X.shape[0]
        if self.C * N < 1.0 - 1e-12:
            raise ValueError(
                f"C * N = {self.C * N:.4g} < 1: Σα=1 infeasible. "
                f"Increase C (≥ 1/N = {1.0 / N:.4g})."
            )

        K = self.kernel(X, X)                # (N, N) — GPU
        K_diag = self.kernel.diag(X)         # (N,) — GPU

        alpha = xp.full(N, 1.0 / N, dtype=xp.float32)
        f = K @ alpha                        # (N,) — GPU

        eps = EPS_F32
        # C2 fix (2026-05-12): 막힌 (i,j) 페어 (eta≈0 또는 box-clipped Δ≈0)
        # 를 일시 mask 하고 다른 violating pair 로 진행.
        blocked: set[tuple[int, int]] = set()
        self._c2_n_blocked_events: int = 0
        self._c2_n_progress_resets: int = 0

        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            g = K_diag - 2.0 * f             # ∂W/∂α — GPU

            up_mask = alpha < self.C - eps   # α_i 를 더 늘릴 수 있는 인덱스
            low_mask = alpha > eps           # α_j 를 더 줄일 수 있는 인덱스
            if not up_mask.any() or not low_mask.any():
                break

            # ── pair 선택 ───────────────────────────────────────────────
            g_up  = xp.where(up_mask,  g, float('-inf'))
            g_low = xp.where(low_mask, g, float('+inf'))
            i = int(xp.argmax(g_up))
            j = int(xp.argmin(g_low))

            if i == j or (i, j) in blocked:
                # Slow path: CPU 로 이동 후 탐색 (동기화 1회)
                _g_up  = _to_numpy(g_up)
                _g_low = _to_numpy(g_low)
                i_order = np.argsort(-_g_up)
                j_order = np.argsort(_g_low)
                found = False
                for ic in i_order:
                    if _g_up[ic] == float('-inf'):
                        break
                    for jc in j_order:
                        if _g_low[jc] == float('+inf'):
                            break
                        if ic == jc:
                            continue
                        pair = (int(ic), int(jc))
                        if pair in blocked:
                            continue
                        i, j = pair
                        found = True
                        break
                    if found:
                        break
                if not found:
                    break

            gap = float(g[i] - g[j])
            if gap < self.tol:
                break

            eta = float(K[i, i] + K[j, j] - 2.0 * K[i, j])
            if eta < EPS_F32:
                blocked.add((i, j))
                self._c2_n_blocked_events += 1
                continue

            delta_unc = gap / (2.0 * eta)
            L = max(-float(alpha[i]), float(alpha[j]) - self.C)
            H = min(self.C - float(alpha[i]), float(alpha[j]))
            delta = float(np.clip(delta_unc, L, H))
            if abs(delta) < eps:
                blocked.add((i, j))
                self._c2_n_blocked_events += 1
                continue

            alpha[i] += delta
            alpha[j] -= delta
            f += delta * (K[i] - K[j])       # GPU: 한 행만 갱신

            if blocked:
                blocked.clear()
                self._c2_n_progress_resets += 1

        self.n_iter_ = n_iter

        # SV 추출 (α > eps 인 것만 유지)
        sv_mask = alpha > eps
        self.X_sv = X[sv_mask].copy()
        self.alpha = alpha[sv_mask].copy()

        # 추론 시 재사용할 상수 (Python float 으로 저장)
        K_sv = self.kernel(self.X_sv, self.X_sv)
        self._center_norm_sq = float(self.alpha @ K_sv @ self.alpha)

        d2_all = self._distance_sq(self.X_sv)
        self.R2 = _compute_R2(d2_all, self.alpha, self.C, eps=eps)

        return self

    # ----------------------------------------------------------- distance ---
    def _distance_sq(self, X):
        """d²(x) = K(x,x) - 2 Σ α_i K(x, x_i) + α^T K_sv α — GPU 배열 반환."""
        K_xs = self.kernel(X, self.X_sv)
        K_xx = self.kernel.diag(X)
        return K_xx - 2.0 * (K_xs @ self.alpha) + self._center_norm_sq

    def distance(self, X) -> np.ndarray:
        """d(x) = sqrt(d²(x)) — numpy 배열 반환."""
        if self.X_sv is None:
            raise RuntimeError("SVDD is not fitted")
        X = xp.asarray(X, dtype=xp.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        return _to_numpy(xp.sqrt(xp.maximum(self._distance_sq(X), 0.0)))

    def decision_function(self, X) -> np.ndarray:
        """R² - d²(x).  ≥ 0: 정상, < 0: 이상 (sklearn 부호 규약) — numpy 배열 반환."""
        if self.X_sv is None:
            raise RuntimeError("SVDD is not fitted")
        X = xp.asarray(X, dtype=xp.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        return _to_numpy(self.R2 - self._distance_sq(X))

    def predict(self, X) -> np.ndarray:
        """+1: 정상, -1: 이상."""
        return np.where(self.decision_function(X) >= 0, 1, -1)

    # ---------------------------------------------------------- save/load ---
    def save(self, path: str | Path) -> None:
        """`<path>.npz` (X_sv, alpha) + `<path>.json` (kernel/scalar 메타)."""
        if self.X_sv is None:
            raise RuntimeError("SVDD is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "kernel": self.kernel.to_dict(),
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "R2": self.R2,
            "center_norm_sq": float(self._center_norm_sq),
            "n_iter_": self.n_iter_,
        }
        np.savez(
            path.with_suffix(".npz"),
            X_sv=_to_numpy(self.X_sv),
            alpha=_to_numpy(self.alpha),
        )
        with open(path.with_suffix(".json"), "w") as fp:
            json.dump(meta, fp, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SVDD":
        path = Path(path)
        with open(path.with_suffix(".json")) as fp:
            meta = json.load(fp)
        data = np.load(path.with_suffix(".npz"))
        from .kernels import create_kernel
        kernel = create_kernel(meta["kernel"])
        obj = cls(kernel=kernel, C=meta["C"], max_iter=meta["max_iter"], tol=meta["tol"])
        obj.X_sv = xp.asarray(data["X_sv"].astype(np.float32))
        obj.alpha = xp.asarray(data["alpha"].astype(np.float32))
        obj.R2 = float(meta["R2"])
        obj._center_norm_sq = float(meta["center_norm_sq"])
        obj.n_iter_ = int(meta.get("n_iter_", 0))
        return obj


# =================================== sanity check ==================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.kernels import RBFKernel  # type: ignore

    from sklearn.metrics import roc_auc_score
    from sklearn.svm import OneClassSVM

    rng = np.random.default_rng(0)

    # 정상: 원점 가우시안, 이상: 외곽
    n_normal, n_anom = 200, 50
    X_normal = rng.standard_normal((n_normal, 2)) * 0.7
    theta = rng.uniform(0, 2 * np.pi, n_anom)
    r = rng.uniform(3.0, 5.0, n_anom)
    X_anom = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    X_train = X_normal
    X_test = np.vstack([X_normal, X_anom])
    y_test = np.concatenate([np.ones(n_normal), -np.ones(n_anom)])  # +1: 정상

    gamma = 0.5
    C = 0.05
    nu = 1.0 / (C * n_normal)

    print("===== SVDD sanity check =====")
    print(f"n_train={n_normal}, n_test_anom={n_anom}, gamma={gamma}, C={C}, nu(equiv)={nu:.3f}")

    # ----- our SVDD -----
    svdd = SVDD(kernel=RBFKernel(gamma=gamma), C=C, max_iter=2000, tol=1e-4)
    svdd.fit(X_train)
    score_ours = svdd.decision_function(X_test)   # numpy 반환
    auc_ours = roc_auc_score((y_test == 1).astype(int), score_ours)

    print(f"\n[ours]    n_iter={svdd.n_iter_}, n_sv={len(svdd.alpha)}, "
          f"R²={svdd.R2:.4f}, α range=[{float(svdd.alpha.min()):.4f}, {float(svdd.alpha.max()):.4f}]")
    print(f"          AUROC = {auc_ours:.4f}")

    # ----- sklearn OneClassSVM -----
    ocsvm = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
    ocsvm.fit(X_train)
    score_sk = ocsvm.decision_function(X_test)
    auc_sk = roc_auc_score((y_test == 1).astype(int), score_sk)

    print(f"\n[sklearn] n_sv={len(ocsvm.support_)}, AUROC = {auc_sk:.4f}")

    # ----- compare -----
    def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))

    rho = spearman_rho(score_ours, score_sk)
    print(f"\n[compare] |Δn_sv| = {abs(len(svdd.alpha) - len(ocsvm.support_))}")
    print(f"          spearman(score_ours, score_sk) = {rho:.4f}  (expect ≥ 0.95)")
    print(f"          AUROC diff = {abs(auc_ours - auc_sk):.4f}  (expect ≤ 0.05)")

    # ----- save/load round-trip -----
    tmpdir = Path("/tmp/svdd_test")
    svdd.save(tmpdir / "model")
    svdd2 = SVDD.load(tmpdir / "model")
    score_reload = svdd2.decision_function(X_test)
    diff = float(np.max(np.abs(score_ours - score_reload)))
    print(f"\n[save/load] max |score - score_reload| = {diff:.2e}  (expect ~ 0)")
    assert diff < 1e-10

    assert auc_ours > 0.95, f"SVDD AUROC too low: {auc_ours}"
    assert rho > 0.95, f"spearman vs sklearn too low: {rho}"
    assert abs(auc_ours - auc_sk) < 0.05, f"AUROC mismatch with sklearn: {auc_ours} vs {auc_sk}"

    # =========== C3: R² fallback 3-tier 극단 케이스 검증 =====================
    print("\n===== C3: R² fallback 3-tier checks =====")

    C_tiny = 1.0 / n_normal
    svdd_tiny = SVDD(kernel=RBFKernel(gamma=gamma), C=C_tiny, max_iter=2000, tol=1e-4)
    svdd_tiny.fit(X_train)

    eps = 1e-10
    a_t = svdd_tiny.alpha
    margin_t = (a_t > eps) & (a_t < svdd_tiny.C - eps)
    nonout_t = a_t < svdd_tiny.C - eps
    d2_all_t = svdd_tiny._distance_sq(svdd_tiny.X_sv)
    print(f"[tiny C={C_tiny:.4g}] n_sv={len(a_t)}, margin SV count={int(margin_t.sum())}, "
          f"non-outlier count={int(nonout_t.sum())}, R²={svdd_tiny.R2:.4f}")
    assert not np.isnan(svdd_tiny.R2), "R² is NaN under tiny C"
    d2_min = float(d2_all_t.min())
    d2_max = float(d2_all_t.max())
    assert d2_min - 1e-9 <= svdd_tiny.R2 <= d2_max + 1e-9, (
        f"R²={svdd_tiny.R2} outside [d²_min={d2_min}, d²_max={d2_max}]"
    )

    a_forced = xp.full_like(a_t, svdd_tiny.C)
    a_forced = a_forced / a_forced.sum()
    a_forced = xp.minimum(a_forced, svdd_tiny.C)
    d2_forced = svdd_tiny._distance_sq(svdd_tiny.X_sv)
    r2_all_outlier = _compute_R2(d2_forced, a_forced, svdd_tiny.C, eps=eps)
    max_d2 = float(d2_forced[a_forced > eps].max())
    print(f"[all α=C → Tier 2] R²={r2_all_outlier:.6f}, max(active d²)={max_d2:.6f}")
    assert abs(r2_all_outlier - max_d2) < 1e-6, "all-α=C must use Tier 2 (max of active d²)"

    a_all_zero = xp.zeros_like(a_forced)
    r2_t3 = _compute_R2(d2_forced, a_all_zero, svdd_tiny.C, eps=eps)
    q95 = float(xp.quantile(d2_forced, 0.95))
    print(f"[Tier 3 / all α=0] R²={r2_t3:.6f}, 95-quantile(d²)={q95:.6f}")
    assert abs(r2_t3 - q95) < 1e-6, "Tier 3 must use 95-quantile"

    a_margin = xp.array([0.1, 0.2, 0.3, 0.4])
    d2_margin = xp.array([1.0, 2.0, 3.0, 4.0])
    r2_t1 = _compute_R2(d2_margin, a_margin, C=1.0, eps=eps)
    assert abs(r2_t1 - float(d2_margin.mean())) < 1e-12, "Tier 1 must be mean of margin d²"
    print(f"[Tier 1 unit ] mean(d²)={float(d2_margin.mean()):.4f}, R²={r2_t1:.4f}  OK")

    C_unit = 0.5
    a_t2 = xp.array([0.0, 0.0, 0.5, 0.5])
    d2_t2 = xp.array([9.0, 9.5, 1.0, 2.5])
    r2_t2 = _compute_R2(d2_t2, a_t2, C=C_unit, eps=eps)
    expected_tier2 = float(d2_t2[a_t2 > eps].max())
    assert abs(r2_t2 - expected_tier2) < 1e-12, (
        f"α=0 slot must not enter Tier 2 max (expect {expected_tier2}): got {r2_t2}"
    )
    print(f"[α=0 unpoll ] R²={r2_t2:.4f} (Tier 2 active max={expected_tier2:.4f}), α=0 slot d² excluded  OK")

    print("\nAll SVDD + C3 R² fallback sanity checks passed.")

    # =========== C2: SMO outer-loop pair-masking 검증 =======================
    print("\n===== C2: SMO outer-loop pair-masking checks =====")

    gamma_x, C_x = 1e3, 1.0 / n_normal
    svdd_x = SVDD(kernel=RBFKernel(gamma=gamma_x), C=C_x, max_iter=2000, tol=1e-4)
    svdd_x.fit(X_train)
    eps_x = 1e-10
    a_x = svdd_x.alpha
    up_x  = a_x < svdd_x.C - eps_x
    low_x = a_x > eps_x
    K_sv  = svdd_x.kernel(svdd_x.X_sv, svdd_x.X_sv)
    g_sv  = svdd_x.kernel.diag(svdd_x.X_sv) - 2.0 * (K_sv @ a_x)
    if up_x.any() and low_x.any():
        kkt_gap = float(xp.max(xp.where(up_x, g_sv, float('-inf')))
                        - xp.min(xp.where(low_x, g_sv, float('+inf'))))
    else:
        kkt_gap = 0.0
    print(f"[extreme] gamma={gamma_x}, C={C_x:.4g}, max_iter=2000")
    print(f"           n_iter={svdd_x.n_iter_}, n_sv={len(a_x)}, "
          f"blocked_events={svdd_x._c2_n_blocked_events}, "
          f"progress_resets={svdd_x._c2_n_progress_resets}")
    print(f"           R²={svdd_x.R2:.4f}, KKT gap (post)={kkt_gap:.4e}")
    assert svdd_x.n_iter_ <= 2000, "max_iter 초과 — 무한루프"
    assert not np.isnan(svdd_x.R2), "R² NaN"

    X_dup = np.vstack([X_normal, X_normal[:40]])
    svdd_d = SVDD(kernel=RBFKernel(gamma=gamma), C=C, max_iter=2000, tol=1e-4)
    svdd_d.fit(X_dup)
    print(f"[duplicates] N={len(X_dup)}, n_iter={svdd_d.n_iter_}, n_sv={len(svdd_d.alpha)}, "
          f"blocked_events={svdd_d._c2_n_blocked_events}, "
          f"progress_resets={svdd_d._c2_n_progress_resets}")
    print(f"             R²={svdd_d.R2:.4f}")
    assert svdd_d.n_iter_ <= 2000, "중복점 데이터에서 max_iter 초과"
    assert not np.isnan(svdd_d.R2), "R² NaN under duplicates"

    a_d = svdd_d.alpha
    eps_d = 1e-10
    up_d  = a_d < svdd_d.C - eps_d
    low_d = a_d > eps_d
    K_sv_d = svdd_d.kernel(svdd_d.X_sv, svdd_d.X_sv)
    g_d    = svdd_d.kernel.diag(svdd_d.X_sv) - 2.0 * (K_sv_d @ a_d)
    if up_d.any() and low_d.any():
        kkt_gap_d = float(xp.max(xp.where(up_d, g_d, float('-inf')))
                          - xp.min(xp.where(low_d, g_d, float('+inf'))))
    else:
        kkt_gap_d = 0.0
    print(f"             KKT gap (post)={kkt_gap_d:.4e}  (expect ≤ tol or progress saturated)")

    assert svdd._c2_n_blocked_events == 0 or svdd._c2_n_blocked_events < svdd.n_iter_, (
        "정상 케이스에서 blocked_events 가 n_iter 만큼 누적 — 진전 없는 의심"
    )
    print(f"[baseline] gamma={gamma}, C={C}: blocked_events={svdd._c2_n_blocked_events}, "
          f"progress_resets={svdd._c2_n_progress_resets}, n_iter={svdd.n_iter_}")

    print("\nAll SVDD + C2 SMO outer-loop sanity checks passed.")
