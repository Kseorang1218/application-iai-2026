"""OnlineSVDD — SV 버퍼 + pairwise SMO partial_fit (streaming).

OTTA 전제: 데이터 한 샘플씩 순차 유입, 반복 학습 불가.
메모리 상한: 버퍼 크기 B 고정 (B×B 커널 캐시 + B×d SV 저장 + B 캐시 벡터).

수식은 SVDD와 동일:
    max_α  Σ α_i K(x_i, x_i) - Σ α_i α_j K(x_i, x_j)
    s.t.   0 ≤ α_i ≤ C,   Σ α_i = 1

다만 working set 선택을 (new sample, *) 페어로 제한해 한 step만 수행.
"""

from __future__ import annotations

try:
    import cupy as xp
except ImportError:
    import numpy as xp

import numpy as np

from typing import Any

from .kernels import RBFKernel
from .svdd import SVDD, _compute_R2, EPS_F32, _to_numpy


# ── Helper: simplex 에 box 제약을 합쳐 projection ─────────────────────────────
def _project_onto_simplex_box(
    a, C: float, max_iter: int = 50, tol: float = EPS_F32,
):
    """{α ∈ ℝ^B : 0 ≤ α_i ≤ C, Σ α_i = 1} 위로 projection (water-filling).

    매 iter 마다:
      1) α 를 [0, C] 로 clip
      2) Σα 와 1 의 차이 (deficit) 를 box 에 안 막힌 항목 (0 < α < C) 에 균등 분배
      3) Σα = 1 ± tol 이면 종료

    feasibility: B × C ≥ 1 이어야 simplex+box 내 점 존재.
    """
    a = xp.clip(a, 0.0, C)
    for _ in range(max_iter):
        s = float(a.sum())
        if abs(s - 1.0) < tol:
            return a
        free = (a > 0.0) & (a < C - EPS_F32)
        if not free.any():
            return a  # 더 분배할 자유도 없음 — infeasible 그대로
        a[free] += (1.0 - s) / int(free.sum())
        a = xp.clip(a, 0.0, C)
    return a


class OnlineSVDD(SVDD):
    """SVDD + 고정 크기 SV 버퍼 + 1-step pairwise SMO partial_fit.

    Parameters
    ----------
    kernel, C, max_iter, tol : `SVDD` 와 동일.
    buffer_size : int
        SV 버퍼 슬롯 수 B. 메모리 사용은 O(B²) (커널 캐시) + O(B·d).

    Attributes (after fit)
    ----------------------
    X_sv, alpha : 버퍼 활성 슬롯의 view (`(_n_sv, d)`, `(_n_sv,)`).
    R2, _center_norm_sq : SVDD 와 동일.
    """

    def __init__(
        self,
        kernel: Any,
        C: float,
        buffer_size: int,
        max_iter: int = 1000,
        tol: float = 1e-3,
        evict_policy: str = "redistribute",
        max_inner_iter: int = 10,
    ):
        super().__init__(kernel, C, max_iter, tol)
        if buffer_size < 2:
            raise ValueError(f"buffer_size must be ≥ 2, got {buffer_size}")
        if evict_policy not in ("redistribute", "transfer"):
            raise ValueError(
                f"evict_policy must be 'redistribute' or 'transfer', got {evict_policy!r}"
            )
        if max_inner_iter < 1:
            raise ValueError(f"max_inner_iter must be ≥ 1, got {max_inner_iter}")
        self.buffer_size: int = int(buffer_size)
        self.max_inner_iter: int = int(max_inner_iter)
        # C1 fix (2026-05-12): default 'redistribute'.
        self.evict_policy: str = evict_policy

        self._X_buf = None
        self._alpha_buf = None
        self._K_buf = None
        self._f_buf = None
        self._diag_buf = None
        self._n_sv: int = 0

    # --------------------------------------------------------------- fit ---
    def fit(self, X) -> "OnlineSVDD":
        """Batch SMO 학습 (parent SVDD) 후 SV를 버퍼에 적재."""
        super().fit(X)

        n = len(self.alpha)
        if n > self.buffer_size:
            top = xp.argsort(self.alpha)[-self.buffer_size:]
            X_init = self.X_sv[top].copy()
            a_init = self.alpha[top].copy()
            a_init = a_init / a_init.sum()
            a_init = _project_onto_simplex_box(a_init, self.C)
            if abs(float(a_init.sum()) - 1.0) > EPS_F32:
                import warnings
                warnings.warn(
                    f"OnlineSVDD: buffer_size × C = {self.buffer_size * self.C:.4g} < 1, "
                    f"Σα = {float(a_init.sum()):.4g} after projection (infeasible buffer for SMO).",
                    stacklevel=2,
                )
            n = self.buffer_size
        else:
            X_init = self.X_sv.copy()
            a_init = self.alpha.copy()

        d = X_init.shape[1]
        self._X_buf     = xp.zeros((self.buffer_size, d), dtype=xp.float32)
        self._alpha_buf = xp.zeros(self.buffer_size, dtype=xp.float32)
        self._K_buf     = xp.zeros((self.buffer_size, self.buffer_size), dtype=xp.float32)
        self._f_buf     = xp.zeros(self.buffer_size, dtype=xp.float32)
        self._diag_buf  = xp.zeros(self.buffer_size, dtype=xp.float32)

        self._X_buf[:n]     = X_init
        self._alpha_buf[:n] = a_init
        self._n_sv = n

        K = self.kernel(X_init, X_init)
        self._K_buf[:n, :n] = K
        self._f_buf[:n]     = K @ a_init
        self._diag_buf[:n]  = self.kernel.diag(X_init)

        self._sync_views()
        self._center_norm_sq = float(a_init @ K @ a_init)
        self._update_R2()
        return self

    # ------------------------------------------------------------ helpers ---
    def _sync_views(self) -> None:
        """X_sv, alpha 를 버퍼 활성 영역의 view 로 재바인딩."""
        n = self._n_sv
        self.X_sv  = self._X_buf[:n]
        self.alpha = self._alpha_buf[:n]

    def _update_R2(self) -> None:
        """R² fallback 3-tier — SVDD._compute_R2 위임.

        K_buf 캐시 활용: d²(x_i) = K_ii - 2·f_i + center_norm_sq 이므로
        kernel(X_sv, X_sv) 재계산(O(B²·d)) 대신 캐시된 f_buf를 사용(O(B·d)).
        """
        if len(self.alpha) == 0:
            return
        n = self._n_sv
        d2 = self._diag_buf[:n] - 2.0 * self._f_buf[:n] + self._center_norm_sq
        self.R2 = _compute_R2(d2, self.alpha, self.C, eps=EPS_F32)

    def _update_center_norm_sq(self) -> None:
        # α^T K α = α^T (Kα) = α^T f_buf — O(B) since f_buf is already cached
        n = self._n_sv
        self._center_norm_sq = float(self._alpha_buf[:n] @ self._f_buf[:n])

    # ---------------------------------------------------------- partial_fit ---
    def partial_fit(self, x) -> "OnlineSVDD":
        """단일 샘플 streaming update.

        Steps
        -----
        1) 버퍼 슬롯 결정
           - 여유 있음 (n < B): 끝에 append, α_new = 0
           - 가득 (n = B): argmin α 슬롯 evict, 같은 슬롯에 swap.
        2) K, f 캐시 incremental update (한 행/열만 재계산).
        3) Pairwise SMO 1 step — pair에 new sample 반드시 포함.
        4) R², center_norm_sq 갱신.
        """
        if self._X_buf is None:
            raise RuntimeError("OnlineSVDD must be fit() before partial_fit().")
        x = xp.asarray(x, dtype=xp.float32).reshape(1, -1)
        if x.shape[1] != self._X_buf.shape[1]:
            raise ValueError(
                f"x dim mismatch: {x.shape[1]} vs {self._X_buf.shape[1]}"
            )

        eps = EPS_F32
        n_sv = self._n_sv
        K_xx = float(self.kernel.diag(x)[0])  # 1.0 for RBF

        # 1) 버퍼 슬롯 결정 + K/f 캐시 갱신 -----------------------------------
        if n_sv < self.buffer_size:
            slot = n_sv
            K_x_sv = self.kernel(x, self._X_buf[:n_sv])[0] if n_sv > 0 else xp.empty(0, dtype=xp.float32)
            self._X_buf[slot]        = x[0]
            self._alpha_buf[slot]    = 0.0
            self._K_buf[slot, :n_sv] = K_x_sv
            self._K_buf[:n_sv, slot] = K_x_sv
            self._K_buf[slot, slot]  = K_xx
            self._diag_buf[slot]     = K_xx
            self._f_buf[slot] = (
                float(K_x_sv @ self._alpha_buf[:n_sv]) if n_sv > 0 else 0.0
            )
            self._n_sv += 1
        else:
            slot = int(xp.argmin(self._alpha_buf[:n_sv]))
            alpha_evicted = float(self._alpha_buf[slot])

            self._X_buf[slot] = x[0]
            K_new_col = self.kernel(x, self._X_buf[:n_sv])[0]
            self._K_buf[slot, :n_sv] = K_new_col
            self._K_buf[:n_sv, slot] = K_new_col
            self._diag_buf[slot]     = K_xx

            if self.evict_policy == "transfer":
                pass  # 새 슬롯이 evicted α 상속 (Σα 보존 trivially)
            else:  # "redistribute" (C1 default)
                self._alpha_buf[slot] = 0.0
                if alpha_evicted > 0.0 and n_sv > 1:
                    idx_all   = xp.arange(n_sv)
                    other_idx = idx_all[idx_all != slot]
                    a_others  = self._alpha_buf[other_idx]
                    s = float(a_others.sum())
                    if s > 0.0:
                        self._alpha_buf[other_idx] = a_others * (1.0 + alpha_evicted / s)
                        a_full = _project_onto_simplex_box(
                            self._alpha_buf[:n_sv].copy(), self.C
                        )
                        self._alpha_buf[:n_sv] = a_full

            self._f_buf[:n_sv] = self._K_buf[:n_sv, :n_sv] @ self._alpha_buf[:n_sv]

        self._sync_views()
        n_sv = self._n_sv

        # 2) Pairwise SMO — KKT gap < tol 또는 max_inner_iter 까지 반복 -------
        i_new = slot
        K_diag_active = self._diag_buf[:n_sv]
        idx = xp.arange(n_sv)

        for _ in range(self.max_inner_iter):
            g = K_diag_active - 2.0 * self._f_buf[:n_sv]

            # Pair A: i = new (α↑), j = argmin g (α>0, ≠new)
            mask_dec = (self.alpha > eps) & (idx != i_new)
            if (float(self.alpha[i_new]) < self.C - eps) and mask_dec.any():
                j_a   = int(xp.argmin(xp.where(mask_dec, g, float('+inf'))))
                gap_a = float(g[i_new] - g[j_a])
            else:
                j_a, gap_a = -1, float('-inf')

            # Pair B: i = argmax g (α<C, ≠new), j = new (α↓)
            mask_inc = (self.alpha < self.C - eps) & (idx != i_new)
            if (float(self.alpha[i_new]) > eps) and mask_inc.any():
                i_b   = int(xp.argmax(xp.where(mask_inc, g, float('-inf'))))
                gap_b = float(g[i_b] - g[i_new])
            else:
                i_b, gap_b = -1, float('-inf')

            if gap_a >= gap_b and gap_a > self.tol:
                i, j = i_new, j_a
            elif gap_b > self.tol:
                i, j = i_b, i_new
            else:
                break  # KKT gap ≤ tol — 수렴

            # SMO sub-problem
            eta = float(self._K_buf[i, i] + self._K_buf[j, j] - 2.0 * self._K_buf[i, j])
            if eta < eps:
                break

            delta_unc = float(g[i] - g[j]) / (2.0 * eta)
            L = max(-float(self.alpha[i]), float(self.alpha[j]) - self.C)
            H = min(self.C - float(self.alpha[i]), float(self.alpha[j]))
            delta = float(np.clip(delta_unc, L, H))

            if abs(delta) < eps:
                break

            self._alpha_buf[i] += delta
            self._alpha_buf[j] -= delta
            self._f_buf[:n_sv] += delta * (
                self._K_buf[i, :n_sv] - self._K_buf[j, :n_sv]
            )

        self._sync_views()
        self._update_center_norm_sq()
        self._update_R2()
        return self


# =================================== sanity check ==================================
if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from sklearn.metrics import roc_auc_score

    from models.kernels import RBFKernel  # type: ignore
    from models.svdd import SVDD          # type: ignore

    rng = np.random.default_rng(0)
    n_normal, n_anom = 200, 50
    X_normal = rng.standard_normal((n_normal, 2)) * 0.7
    theta = rng.uniform(0, 2 * np.pi, n_anom)
    r = rng.uniform(3.0, 5.0, n_anom)
    X_anom = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    X_train = X_normal
    X_test = np.vstack([X_normal, X_anom])
    y_test = np.concatenate([np.ones(n_normal), -np.ones(n_anom)])

    gamma, C = 0.5, 0.05
    n_init = 50
    X_init = X_train[:n_init]
    X_stream = X_train[n_init:]

    print("===== OnlineSVDD sanity check =====")
    print(f"n_init={n_init}, n_stream={len(X_stream)}, gamma={gamma}")

    # ---- batch reference ----
    svdd = SVDD(kernel=RBFKernel(gamma=gamma), C=C, max_iter=2000, tol=1e-4)
    svdd.fit(X_train)
    auc_batch = roc_auc_score((y_test == 1).astype(int), svdd.decision_function(X_test))
    print(f"\n[batch ref]  n_sv={len(svdd.alpha)}, R²={svdd.R2:.4f}, AUROC={auc_batch:.4f}")

    # ---- buffer trade-off ----
    print(f"\n[buffer size sweep]")
    print(f"  {'B':>4}  {'mem_KB':>8}  {'n_sv_final':>10}  {'R²':>8}  {'AUROC':>8}  {'ΔAUROC':>8}  {'time_s':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    results = {}
    oc = max(C, 1.1 / n_init)
    for B in [50, 100, 200]:
        online = OnlineSVDD(
            kernel=RBFKernel(gamma=gamma), C=oc, buffer_size=B, max_iter=2000, tol=1e-4
        )
        online.fit(X_init)
        t0 = time.perf_counter()
        for x in X_stream:
            online.partial_fit(x)
        t_stream = time.perf_counter() - t0

        auc_online = roc_auc_score((y_test == 1).astype(int), online.decision_function(X_test))
        mem_bytes = (
            online._K_buf.nbytes
            + online._X_buf.nbytes
            + online._alpha_buf.nbytes
            + online._f_buf.nbytes
        )
        print(f"  {B:>4}  {mem_bytes/1024:>8.1f}  {online._n_sv:>10}  "
              f"{online.R2:>8.4f}  {auc_online:>8.4f}  "
              f"{auc_online-auc_batch:>+8.4f}  {t_stream:>8.4f}")
        results[B] = (auc_online, online)

    diff_b200 = abs(results[200][0] - auc_batch)
    print(f"\n[check]   |AUROC(B=200) - AUROC(batch)| = {diff_b200:.4f}  (expect ≤ 0.05)")
    assert diff_b200 < 0.05, f"OnlineSVDD(B=200) vs batch SVDD: AUROC diff {diff_b200} > 0.05"

    diff_b50 = abs(results[50][0] - auc_batch)
    print(f"[check]   |AUROC(B=50)  - AUROC(batch)| = {diff_b50:.4f}  (expect ≤ 0.10)")
    assert diff_b50 < 0.10, f"OnlineSVDD(B=50) AUROC drop too large: {diff_b50}"

    # save/load round-trip via parent SVDD
    online = results[200][1]
    online.save("/tmp/online_svdd_test/model")
    base = SVDD.load("/tmp/online_svdd_test/model")
    diff = float(np.max(np.abs(
        online.decision_function(X_test) - base.decision_function(X_test)
    )))
    print(f"\n[save/load] inference round-trip via parent: max |Δ| = {diff:.2e}")
    assert diff < 1e-10

    print("\nAll OnlineSVDD sanity checks passed.")

    # ============= C5: SV truncation projection 검증 =========================
    print("\n===== C5: truncation projection checks =====")

    a_test = xp.array([0.5, 0.4, 0.3, 0.2, 0.1])
    a_test = a_test / a_test.sum()
    C_test = 0.3
    proj = _project_onto_simplex_box(a_test.copy(), C_test)
    print(f"[unit] input  = {_to_numpy(a_test).round(4)}, C={C_test}")
    print(f"       output = {_to_numpy(proj).round(4)}, Σ={float(proj.sum()):.6f}")
    assert (proj <= C_test + 1e-10).all(), f"projection: α > C: {proj}"
    assert (proj >= -1e-10).all(), f"projection: α < 0: {proj}"
    assert abs(float(proj.sum()) - 1.0) < 1e-9, f"projection: Σα ≠ 1: {float(proj.sum())}"

    n_lots = 600
    X_lots = rng.standard_normal((n_lots, 2)) * 0.7
    B_t, C_t = 20, 0.06
    online_t = OnlineSVDD(
        kernel=RBFKernel(gamma=gamma), C=C_t, buffer_size=B_t,
        max_iter=2000, tol=1e-4,
    )
    online_t.fit(X_lots)
    a_t = online_t.alpha
    print(f"[trunc] N={n_lots}, B={B_t}, C={C_t}, n_sv_buf={online_t._n_sv}, "
          f"α range=[{float(a_t.min()):.4f}, {float(a_t.max()):.4f}], Σα={float(a_t.sum()):.6f}")
    assert (a_t <= C_t + 1e-10).all(), "C5: truncation 후 α > C"
    assert (a_t >= -1e-10).all(),       "C5: truncation 후 α < 0"
    assert abs(float(a_t.sum()) - 1.0) < 1e-6, "C5: truncation 후 Σα ≠ 1"

    print("All C5 truncation checks passed.")

    # ============= C1: eviction policy 검증 =================================
    print("\n===== C1: eviction policy checks =====")

    B_c, C_c = 10, 0.15
    n_init_c, n_stream_c = 50, 300
    X_full = rng.standard_normal((n_init_c + n_stream_c, 2)) * 0.7
    online_s = OnlineSVDD(
        kernel=RBFKernel(gamma=gamma), C=C_c, buffer_size=B_c,
        max_iter=500, tol=1e-3,
    )
    online_s.fit(X_full[:n_init_c])

    max_rel_dR2 = 0.0
    prev_R2 = online_s.R2
    for step, x in enumerate(X_full[n_init_c:], start=1):
        online_s.partial_fit(x)
        a = online_s.alpha
        assert abs(float(a.sum()) - 1.0) < 1e-6, f"step {step}: Σα={float(a.sum()):.6f} ≠ 1"
        assert (a <= C_c + 1e-9).all(), f"step {step}: α > C max={float(a.max())}"
        assert (a >= -1e-9).all(),       f"step {step}: α < 0 min={float(a.min())}"
        if prev_R2 > 0:
            max_rel_dR2 = max(max_rel_dR2, abs(online_s.R2 - prev_R2) / max(prev_R2, 1e-12))
        prev_R2 = online_s.R2
    print(f"[stream] B={B_c}, n_stream={n_stream_c}, max relative ΔR² per step = {max_rel_dR2*100:.2f}%")
    assert max_rel_dR2 < 0.5, f"step 당 ΔR² 비율 너무 큼: {max_rel_dR2*100:.1f}%"

    n_seeds = 20
    metrics = {"redistribute": {"dR2": [], "a_new": []},
               "transfer":     {"dR2": [], "a_new": []}}
    for seed in range(n_seeds):
        rng_s = np.random.default_rng(seed + 100)
        X_seed_fit   = rng_s.standard_normal((200, 2)) * 0.7
        X_seed_noise = rng_s.standard_normal((1, 2)) * 5.0
        for policy in ("redistribute", "transfer"):
            on_p = OnlineSVDD(
                kernel=RBFKernel(gamma=gamma), C=C_c, buffer_size=B_c,
                max_iter=500, tol=1e-3, evict_policy=policy,
            )
            on_p.fit(X_seed_fit)
            R2_b = on_p.R2
            on_p.partial_fit(X_seed_noise[0])
            R2_a = on_p.R2
            X_sv_np = _to_numpy(on_p.X_sv)
            x_match = np.where(np.all(X_sv_np == X_seed_noise[0], axis=1))[0]
            a_new = float(on_p.alpha[x_match[0]]) if x_match.size else 0.0
            metrics[policy]["dR2"].append(abs(R2_a - R2_b))
            metrics[policy]["a_new"].append(a_new)
            assert abs(float(on_p.alpha.sum()) - 1.0) < 1e-6
            assert (on_p.alpha <= C_c + 1e-9).all()

    print(f"[noise sensitivity] mean over {n_seeds} seeds "
          f"(1 outlier σ=5 injected after fit on 200 normal, B={B_c} C={C_c})")
    print(f"  {'policy':>12}  {'mean |ΔR²|':>11}  {'std':>8}  {'mean α_new':>11}  {'std':>8}")
    for pol in ("redistribute", "transfer"):
        dR2 = np.array(metrics[pol]["dR2"])
        aN  = np.array(metrics[pol]["a_new"])
        print(f"  {pol:>12}  {dR2.mean():>11.4f}  {dR2.std():>8.4f}  "
              f"{aN.mean():>11.4f}  {aN.std():>8.4f}")

    print("\nAll C1 eviction policy checks passed.")
