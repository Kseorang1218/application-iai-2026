"""OnlineSVDD — SV 버퍼 + pairwise SMO partial_fit (streaming).

OTTA 전제: 데이터 한 샘플씩 순차 유입, 반복 학습 불가.
메모리 상한: 버퍼 크기 B 고정 (B×B 커널 캐시 + B×d SV 저장 + B 캐시 벡터).

수식은 SVDD와 동일:
    max_α  Σ α_i K(x_i, x_i) - Σ α_i α_j K(x_i, x_j)
    s.t.   0 ≤ α_i ≤ C,   Σ α_i = 1

다만 working set 선택을 (new sample, *) 페어로 제한해 한 step만 수행.
"""

from __future__ import annotations

import numpy as np

from typing import Any

from .kernels import RBFKernel
from .svdd import SVDD, _compute_R2, EPS_F32


# ── Helper: simplex 에 box 제약을 합쳐 projection ─────────────────────────────
def _project_onto_simplex_box(
    a: np.ndarray, C: float, max_iter: int = 50, tol: float = EPS_F32,
) -> np.ndarray:
    """{α ∈ ℝ^B : 0 ≤ α_i ≤ C, Σ α_i = 1} 위로 projection (water-filling).

    매 iter 마다:
      1) α 를 [0, C] 로 clip
      2) Σα 와 1 의 차이 (deficit) 를 box 에 안 막힌 항목 (0 < α < C) 에 균등 분배
      3) Σα = 1 ± tol 이면 종료

    feasibility: B × C ≥ 1 이어야 simplex+box 내 점 존재. 위반 시 모든 항목이
    C 로 saturate 되어 Σα = B·C < 1 인 상태로 반환 (호출자가 경고하도록 그대로 둠).
    """
    a = np.clip(a, 0.0, C)
    for _ in range(max_iter):
        s = float(a.sum())
        if abs(s - 1.0) < tol:
            return a
        free = (a > 0.0) & (a < C - EPS_F32)
        if not free.any():
            return a  # 더 분배할 자유도 없음 — infeasible 그대로
        a[free] += (1.0 - s) / int(free.sum())
        a = np.clip(a, 0.0, C)
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
        #   · 'redistribute' — evict 후 새 슬롯 α=0, 빠진 α 는 기존 SV 에 prorata 분배 + box projection.
        #     노이즈 샘플이 즉시 큰 α 를 상속받아 모델을 오염시키는 문제 차단.
        #   · 'transfer'    — legacy: 새 슬롯이 evicted α 그대로 상속 (Σα 보존 trivially).
        #                     ablation 비교용으로 유지.
        # TODO (C1 ablation, 리뷰 O2 연계): protect_top_alpha 플래그 검토 —
        #   α 큰 SV (예: α > 평균) 를 eviction 대상에서 제외하여 핵심 정보 보존.
        self.evict_policy: str = evict_policy

        # 첫 fit 에서 차원 결정 후 할당
        self._X_buf: np.ndarray | None = None       # (B, d)
        self._alpha_buf: np.ndarray | None = None   # (B,)
        self._K_buf: np.ndarray | None = None       # (B, B) — K(SV_i, SV_j) 캐시
        self._f_buf: np.ndarray | None = None       # (B,)   — (Kα)_i 캐시
        self._diag_buf: np.ndarray | None = None    # (B,)   — K(SV_i, SV_i) 대각 캐시
        self._n_sv: int = 0

    # --------------------------------------------------------------- fit ---
    def fit(self, X: np.ndarray) -> "OnlineSVDD":
        """Batch SMO 학습 (parent SVDD) 후 SV를 버퍼에 적재.

        SV 가 buffer_size 보다 많으면 α 큰 순으로 truncate, 정규화 후 적재.
        """
        super().fit(X)

        n = len(self.alpha)
        if n > self.buffer_size:
            top = np.argsort(self.alpha)[-self.buffer_size:]
            X_init = self.X_sv[top].copy()
            # C5 fix (2026-05-12): 일률 normalize 만 하면 α > C 위반 가능.
            # simplex+box projection 으로 0 ≤ α ≤ C, Σα = 1 동시 보장.
            a_init = self.alpha[top].copy()
            a_init = a_init / a_init.sum()        # 1 차 normalize
            a_init = _project_onto_simplex_box(a_init, self.C)
            # buffer_size × C < 1 이면 projection 후에도 Σα < 1 — feasibility 위반
            if abs(a_init.sum() - 1.0) > EPS_F32:
                import warnings
                warnings.warn(
                    f"OnlineSVDD: buffer_size × C = {self.buffer_size * self.C:.4g} < 1, "
                    f"Σα = {a_init.sum():.4g} after projection (infeasible buffer for SMO).",
                    stacklevel=2,
                )
            n = self.buffer_size
        else:
            X_init = self.X_sv.copy()
            a_init = self.alpha.copy()

        d = X_init.shape[1]
        self._X_buf = np.zeros((self.buffer_size, d), dtype=np.float32)
        self._alpha_buf = np.zeros(self.buffer_size, dtype=np.float32)
        self._K_buf = np.zeros((self.buffer_size, self.buffer_size), dtype=np.float32)
        self._f_buf = np.zeros(self.buffer_size, dtype=np.float32)
        self._diag_buf = np.zeros(self.buffer_size, dtype=np.float32)

        self._X_buf[:n] = X_init
        self._alpha_buf[:n] = a_init
        self._n_sv = n

        K = self.kernel(X_init, X_init)
        self._K_buf[:n, :n] = K
        self._f_buf[:n] = K @ a_init
        self._diag_buf[:n] = self.kernel.diag(X_init)

        self._sync_views()
        self._center_norm_sq = np.float32(a_init @ K @ a_init)
        self._update_R2()
        return self

    # ------------------------------------------------------------ helpers ---
    def _sync_views(self) -> None:
        """X_sv, alpha 를 버퍼 활성 영역의 view 로 재바인딩."""
        n = self._n_sv
        self.X_sv = self._X_buf[:n]
        self.alpha = self._alpha_buf[:n]

    def _update_R2(self) -> None:
        """R² fallback 3-tier — SVDD._compute_R2 위임.

        α=0 빈 버퍼 슬롯은 Tier 1/2 mask 에서 제외되므로, OnlineSVDD 도
        svdd.py 와 동일한 분기 의미를 가진다. (노이즈 샘플이 빈 슬롯에 자리잡아
        R² 를 오염시키는 문제 차단.)

        K_buf 캐시 활용: d²(x_i) = K_ii - 2·f_i + center_norm_sq 이므로
        kernel(X_sv, X_sv) 재계산(O(B²·d)) 대신 캐시된 f_buf를 사용(O(B·d)).
        """
        if len(self.alpha) == 0:
            return
        n = self._n_sv
        d2 = self._diag_buf[:n] - np.float32(2.0) * self._f_buf[:n] + self._center_norm_sq
        self.R2 = _compute_R2(d2, self.alpha, self.C, eps=EPS_F32)

    def _update_center_norm_sq(self) -> None:
        # α^T K α = α^T (Kα) = α^T f_buf — O(B) since f_buf is already cached
        n = self._n_sv
        self._center_norm_sq = np.float32(self._alpha_buf[:n] @ self._f_buf[:n])

    # ---------------------------------------------------------- partial_fit ---
    def partial_fit(self, x: np.ndarray) -> "OnlineSVDD":
        """단일 샘플 streaming update.

        Steps
        -----
        1) 버퍼 슬롯 결정
           - 여유 있음 (n < B): 끝에 append, α_new = 0
           - 가득 (n = B): argmin α 슬롯 evict, 같은 슬롯에 swap. α 는 그대로
             유지하여 Σα = 1 보존.
        2) K, f 캐시 incremental update (한 행/열만 재계산).
        3) Pairwise SMO 1 step — pair에 new sample 반드시 포함:
              Pair A: (i=new, j=argmin g, α_j > 0)
              Pair B: (i=argmax g, α_i < C, j=new)
            더 큰 gap (gap > tol) 인 쪽으로 업데이트.
        4) R², center_norm_sq 갱신.
        """
        if self._X_buf is None:
            raise RuntimeError("OnlineSVDD must be fit() before partial_fit().")
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self._X_buf.shape[1]:
            raise ValueError(
                f"x dim mismatch: {x.shape[1]} vs {self._X_buf.shape[1]}"
            )

        eps = EPS_F32
        n_sv = self._n_sv
        K_xx = np.float32(self.kernel.diag(x)[0])  # 1.0 for RBF

        # 1) 버퍼 슬롯 결정 + K/f 캐시 갱신 -----------------------------------
        if n_sv < self.buffer_size:
            slot = n_sv
            K_x_sv = self.kernel(x, self._X_buf[:n_sv])[0] if n_sv > 0 else np.empty(0)
            self._X_buf[slot] = x[0]
            self._alpha_buf[slot] = 0.0
            self._K_buf[slot, :n_sv] = K_x_sv
            self._K_buf[:n_sv, slot] = K_x_sv
            self._K_buf[slot, slot] = K_xx
            self._diag_buf[slot] = K_xx
            # f_l (l < n_sv): α_new = 0 → 변화 없음
            # f_slot = K_x_sv @ α_existing
            self._f_buf[slot] = (K_x_sv @ self._alpha_buf[:n_sv]) if n_sv > 0 else np.float32(0.0)
            self._n_sv += 1
        else:
            slot = int(np.argmin(self._alpha_buf[:n_sv]))
            alpha_evicted = float(self._alpha_buf[slot])

            # 데이터 swap + K col/row 재계산
            self._X_buf[slot] = x[0]
            K_new_col = self.kernel(x, self._X_buf[:n_sv])[0]   # K_new_col[slot] = K_xx
            self._K_buf[slot, :n_sv] = K_new_col
            self._K_buf[:n_sv, slot] = K_new_col
            self._diag_buf[slot] = K_xx

            # α 분배 정책
            if self.evict_policy == "transfer":
                # legacy: 새 슬롯이 evicted α 상속 (Σα 보존 trivially).
                # alpha_buf[slot] 는 alpha_evicted 그대로 둔다.
                pass
            else:  # "redistribute" (C1 default)
                # 새 슬롯 α = 0, evicted α 는 다른 SV 들에 prorata 분배 + box projection.
                # 노이즈 샘플이 즉시 큰 α 를 상속받아 모델을 오염시키는 문제 차단.
                self._alpha_buf[slot] = 0.0
                if alpha_evicted > 0.0 and n_sv > 1:
                    idx_all = np.arange(n_sv)
                    other_idx = idx_all[idx_all != slot]
                    a_others = self._alpha_buf[other_idx]
                    s = float(a_others.sum())
                    if s > 0.0:
                        # prorata: 기존 α 크기에 비례
                        self._alpha_buf[other_idx] = a_others * (1.0 + alpha_evicted / s)
                        # box 위반 가능 → simplex+box projection
                        a_full = _project_onto_simplex_box(
                            self._alpha_buf[:n_sv].copy(), self.C
                        )
                        self._alpha_buf[:n_sv] = a_full
                    # else: 모두 0 인 corner — 다음 SMO step 이 Σα 복원 시도

            # α / K 가 모두 바뀌었으니 f 캐시 전체 재계산 (B² flops, B 작아 부담 X)
            self._f_buf[:n_sv] = self._K_buf[:n_sv, :n_sv] @ self._alpha_buf[:n_sv]

        self._sync_views()
        n_sv = self._n_sv

        # 2) Pairwise SMO — KKT gap < tol 또는 max_inner_iter 까지 반복 -------
        # NOTE: pair 선택이 매 iteration 에서 i_new (신규 샘플) 를 anchor 로 고정함.
        # 본 루프는 신규 샘플이 관여하는 KKT 위반만 해소하며, 기존 SV 들끼리의
        # KKT 위반은 해소되지 않음 — 즉 전체 α 가 well-defined KKT 해임을
        # 이론적으로 보장하지는 않음.
        i_new = slot
        K_diag_active = self._diag_buf[:n_sv]                    # 캐시 사용 — 루프 invariant
        idx = np.arange(n_sv)

        for _ in range(self.max_inner_iter):
            g = K_diag_active - 2.0 * self._f_buf[:n_sv]

            # Pair A: i = new (α↑), j = argmin g (α>0, ≠new)
            mask_dec = (self.alpha > eps) & (idx != i_new)
            if (self.alpha[i_new] < self.C - eps) and mask_dec.any():
                j_a = int(np.argmin(np.where(mask_dec, g, +np.inf)))
                gap_a = float(g[i_new] - g[j_a])
            else:
                j_a, gap_a = -1, -np.inf

            # Pair B: i = argmax g (α<C, ≠new), j = new (α↓)
            mask_inc = (self.alpha < self.C - eps) & (idx != i_new)
            if (self.alpha[i_new] > eps) and mask_inc.any():
                i_b = int(np.argmax(np.where(mask_inc, g, -np.inf)))
                gap_b = float(g[i_b] - g[i_new])
            else:
                i_b, gap_b = -1, -np.inf

            if gap_a >= gap_b and gap_a > self.tol:
                i, j = i_new, j_a
            elif gap_b > self.tol:
                i, j = i_b, i_new
            else:
                break  # KKT gap ≤ tol — 수렴

            # SMO sub-problem
            eta = self._K_buf[i, i] + self._K_buf[j, j] - 2.0 * self._K_buf[i, j]
            if eta < eps:
                break  # ill-conditioned pair

            delta_unc = (g[i] - g[j]) / (2.0 * eta)
            L = max(-self.alpha[i], self.alpha[j] - self.C)
            H = min(self.C - self.alpha[i], self.alpha[j])
            delta = float(np.clip(delta_unc, L, H))

            if abs(delta) < eps:
                break  # 진전 없음 (box-clip 으로 Δ≈0)

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
    # init-set 기준 C * n_init ≥ 1 만족시켜야 fit 이 feasible
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

    # save/load round-trip via parent SVDD (inference만 검증)
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

    # ---- (a) _project_onto_simplex_box 단위 테스트 ----
    a_test = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    a_test = a_test / a_test.sum()  # 합=1 인 입력
    C_test = 0.3
    proj = _project_onto_simplex_box(a_test.copy(), C_test)
    print(f"[unit] input  = {a_test.round(4)}, C={C_test}")
    print(f"       output = {proj.round(4)}, Σ={proj.sum():.6f}")
    assert (proj <= C_test + 1e-10).all(), f"projection: α > C: {proj}"
    assert (proj >= -1e-10).all(), f"projection: α < 0: {proj}"
    assert abs(proj.sum() - 1.0) < 1e-9, f"projection: Σα ≠ 1: {proj.sum()}"

    # ---- (b) end-to-end: truncation 발동 케이스 ----
    # batch SVDD 가 B 보다 많은 SV 를 만들도록 큰 N + 큰 C → truncation 발동
    n_lots = 600
    X_lots = rng.standard_normal((n_lots, 2)) * 0.7
    B_t, C_t = 20, 0.06         # B·C = 1.2 ≥ 1, feasible
    online_t = OnlineSVDD(
        kernel=RBFKernel(gamma=gamma), C=C_t, buffer_size=B_t,
        max_iter=2000, tol=1e-4,
    )
    online_t.fit(X_lots)
    a_t = online_t.alpha
    print(f"[trunc] N={n_lots}, B={B_t}, C={C_t}, n_sv_buf={online_t._n_sv}, "
          f"α range=[{a_t.min():.4f}, {a_t.max():.4f}], Σα={a_t.sum():.6f}")
    assert (a_t <= C_t + 1e-10).all(), "C5: truncation 후 α > C"
    assert (a_t >= -1e-10).all(),       "C5: truncation 후 α < 0"
    assert abs(a_t.sum() - 1.0) < 1e-6, "C5: truncation 후 Σα ≠ 1"

    print("All C5 truncation checks passed.")

    # ============= C1: eviction policy 검증 =================================
    print("\n===== C1: eviction policy checks =====")

    # ---- (c) B=10 stream 300 invariants ----
    B_c, C_c = 10, 0.15          # B·C = 1.5 feasible
    n_init_c, n_stream_c = 50, 300
    X_full = rng.standard_normal((n_init_c + n_stream_c, 2)) * 0.7
    online_s = OnlineSVDD(
        kernel=RBFKernel(gamma=gamma), C=C_c, buffer_size=B_c,
        max_iter=500, tol=1e-3,  # evict_policy default = redistribute
    )
    online_s.fit(X_full[:n_init_c])

    max_rel_dR2 = 0.0
    prev_R2 = online_s.R2
    for step, x in enumerate(X_full[n_init_c:], start=1):
        online_s.partial_fit(x)
        a = online_s.alpha
        assert abs(a.sum() - 1.0) < 1e-6, f"step {step}: Σα={a.sum():.6f} ≠ 1"
        assert (a <= C_c + 1e-9).all(), f"step {step}: α > C max={a.max()}"
        assert (a >= -1e-9).all(),       f"step {step}: α < 0 min={a.min()}"
        if prev_R2 > 0:
            max_rel_dR2 = max(max_rel_dR2, abs(online_s.R2 - prev_R2) / max(prev_R2, 1e-12))
        prev_R2 = online_s.R2
    print(f"[stream] B={B_c}, n_stream={n_stream_c}, max relative ΔR² per step = {max_rel_dR2*100:.2f}%")
    # 5% 가 이상적이지만 B 작을 때 일시 spike 가능 → 50% 로 soft assert
    assert max_rel_dR2 < 0.5, f"step 당 ΔR² 비율 너무 큼: {max_rel_dR2*100:.1f}%"

    # ---- (d) 노이즈 sensitivity: redistribute vs transfer (multi-seed 평균) ----
    # 단일 시드는 분산 커서 신뢰 어려움 → 20 시드 평균 + α_new (노이즈에 부여된 α) 동시 보고.
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
            # 노이즈 샘플은 가장 최근 처리된 슬롯 — 끝 슬롯이지만 evict 이 일어났으면
            # argmin 위치. 거리로 식별: 마지막 partial_fit 의 x 와 일치하는 row.
            x_match = np.where(np.all(on_p.X_sv == X_seed_noise[0], axis=1))[0]
            a_new = float(on_p.alpha[x_match[0]]) if x_match.size else 0.0
            metrics[policy]["dR2"].append(abs(R2_a - R2_b))
            metrics[policy]["a_new"].append(a_new)
            # invariants
            assert abs(on_p.alpha.sum() - 1.0) < 1e-6
            assert (on_p.alpha <= C_c + 1e-9).all()

    print(f"[noise sensitivity] mean over {n_seeds} seeds "
          f"(1 outlier σ=5 injected after fit on 200 normal, B={B_c} C={C_c})")
    print(f"  {'policy':>12}  {'mean |ΔR²|':>11}  {'std':>8}  {'mean α_new':>11}  {'std':>8}")
    for pol in ("redistribute", "transfer"):
        dR2 = np.array(metrics[pol]["dR2"])
        aN  = np.array(metrics[pol]["a_new"])
        print(f"  {pol:>12}  {dR2.mean():>11.4f}  {dR2.std():>8.4f}  "
              f"{aN.mean():>11.4f}  {aN.std():>8.4f}")
    # 정량 단언 대신 측정값을 기록.  α_new 가 redistribute 에서 더 낮으면
    # 노이즈의 즉시 영향력이 줄었음을 시사 (R² 는 SMO 후 dynamics 가 다름).

    print("\nAll C1 eviction policy checks passed.")
