"""시각화 모듈.

본 모듈은 다음 두 종류의 시각화 primitive 를 포함한다:
  1. Figure primitive — `plot_distance_hist`, `plot_tsne`, `tsne_embedding`.
  2. Analysis-level CSV/NPZ 스캐너 시각화 — `plot_mmd_heatmap`,
     `plot_dual_boundary_polar`, `plot_tradeoff_csv`, `plot_cm_by_dataset`.
     analysis.py / analysis_otta.py 가 사용.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


# ── analysis 공용 상수 ────────────────────────────────────────────────────────

PREP_ORDER   = ["p1_raw", "p2_fft", "p3_envspec", "p4_cepstrum", "p5_order", "p6_tds", "p7_orderspec"]
DATASET_ORDER = ["cwru", "pu", "uos"]
KERNELS      = ("rbf", "linear")

# 시각화 라벨 (짧은 이름). evaluate_ad_performance 의 풀 이름 _PREP_LABEL 과 별개.
_PREP_LABEL = {
    "p1_raw":      "Raw",
    "p2_fft":      "FFT",
    "p3_envspec":  "Envelope",
    "p4_cepstrum": "Cepstrum",
    "p5_order":    "Order",
    "p6_tds":        "TDS",
    "p6_stat":       "TDS",
    "p7_orderspec":  "Order Spec",
}

_GROUP_LABEL = {
    "source_train":  "Source Normal",
    "target_normal": "Target Normal",
    "target_fault":  "Target Fault",
}


def _scenario_sort_key(scenario_id: str) -> tuple[int, int]:
    src, tgt = scenario_id.split("_to_")
    return int(src), int(tgt)


def _ordered_scenarios(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(dataset, scenario_id) 쌍을 dataset → 시나리오 순으로 정렬."""
    pairs = df[["dataset", "scenario_id"]].drop_duplicates()
    rows: list[tuple[str, str]] = []
    for ds in DATASET_ORDER:
        sub = pairs[pairs["dataset"] == ds]
        if sub.empty:
            continue
        ordered = sorted(sub["scenario_id"].tolist(), key=_scenario_sort_key)
        rows.extend((ds, sid) for sid in ordered)
    return rows


def plot_distance_hist(
    d_groups: dict[str, np.ndarray],
    R: float | None = None,
    r_inner: float | None = None,
    r_outer: float | None = None,
    bins: int = 50,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """그룹별 거리 분포 히스토그램 + R / r_inner / r_outer vline."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in d_groups.items():
        n_uniq = len(np.unique(d))
        safe_bins = min(bins, n_uniq) if n_uniq > 1 else 1
        ax.hist(d, bins=safe_bins, alpha=0.5, label=f"{name} (n={len(d)})", density=True)
    if R is not None:
        ax.axvline(R, color="k", linestyle=":", label=f"R={R:.3f}")
    if r_inner is not None:
        ax.axvline(r_inner, color="g", linestyle="--", label=f"r_inner={r_inner:.3f}")
    if r_outer is not None:
        ax.axvline(r_outer, color="r", linestyle="--", label=f"r_outer={r_outer:.3f}")
    ax.set_xlabel("distance to hypersphere center  d(x)")
    ax.set_ylabel("density")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def tsne_embedding(
    X_s: np.ndarray,
    X_t: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """source + target 결합 후 t-SNE 2D embedding."""
    X_s = np.asarray(X_s, dtype=float)
    X_t = np.asarray(X_t, dtype=float)
    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(f"X_s, X_t must share feature dim. got {X_s.shape}, {X_t.shape}")
    n_s = X_s.shape[0]
    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=random_state,
    ).fit_transform(np.vstack([X_s, X_t]))
    return Z[:n_s], Z[n_s:]


def plot_tsne(
    X_s: np.ndarray,
    X_t: np.ndarray,
    y_s: np.ndarray | None = None,
    y_t: np.ndarray | None = None,
    *,
    n_subsample: int = 1000,
    perplexity: float = 30.0,
    random_state: int = 42,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """source(원) / target(삼각) t-SNE scatter."""
    rng = np.random.default_rng(random_state)

    def _sub(X: np.ndarray, y: np.ndarray | None, n: int):
        if X.shape[0] <= n:
            return X, y
        idx = rng.choice(X.shape[0], n, replace=False)
        return X[idx], (None if y is None else np.asarray(y)[idx])

    X_s, y_s = _sub(X_s, y_s, n_subsample)
    X_t, y_t = _sub(X_t, y_t, n_subsample)
    Z_s, Z_t = tsne_embedding(X_s, X_t, perplexity, random_state)

    fig, ax = plt.subplots(figsize=(7, 6))
    NORMAL_C, FAULT_C = "tab:blue", "tab:red"

    def _scatter(Z: np.ndarray, y: np.ndarray | None, marker: str, domain: str):
        if y is None:
            ax.scatter(
                Z[:, 0], Z[:, 1],
                color="C0" if domain == "source" else "C1",
                marker=marker, s=18, alpha=0.6, label=domain,
            )
            return
        for v in np.unique(y):
            m = (y == v)
            ax.scatter(
                Z[m, 0], Z[m, 1],
                color=NORMAL_C if v == 0 else FAULT_C,
                marker=marker, s=18, alpha=0.6,
                label=f"{domain} {'normal' if v == 0 else 'fault'}",
            )

    _scatter(Z_s, y_s, "o", "source")
    _scatter(Z_t, y_t, "^", "target")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Analysis-level: CSV/NPZ scanner 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_mmd_heatmap(
    eval_out: pathlib.Path,
    save_path: pathlib.Path,
) -> None:
    """44 시나리오 × 6 전처리 mmd² heatmap (RBF 대표).

    입력: {eval_out}/domain_shift_performance.csv  (kernel 컬럼으로 rbf 필터)
    출력: save_path (PNG)
    """
    perf_path = eval_out / "domain_shift_performance.csv"
    if not perf_path.exists():
        print("[analysis] domain_shift_performance.csv 없음 — mmd_heatmap skip")
        return
    full_df = pd.read_csv(perf_path)
    df = full_df[full_df["kernel"] == "rbf"].copy() if "kernel" in full_df.columns else full_df
    scenarios = _ordered_scenarios(df)
    if not scenarios:
        print("[analysis] no scenarios — mmd_heatmap skip")
        return
    scenario_labels = [f"{ds}/{sid}" for ds, sid in scenarios]

    df = df.set_index(["dataset", "scenario_id", "preprocessing"])["mmd2"]
    rows = []
    for ds, sid in scenarios:
        row = []
        for prep in PREP_ORDER:
            try:
                row.append(float(df.loc[(ds, sid, prep)]))
            except KeyError:
                row.append(np.nan)
        rows.append(row)
    mat = np.array(rows, dtype=float)
    if not np.isfinite(mat).any():
        print("[analysis] all mmd² nan — mmd_heatmap skip")
        return

    vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(PREP_ORDER)))
    ax.set_xticklabels(PREP_ORDER, rotation=30, ha="right")

    prev = None
    for i, label in enumerate(scenario_labels):
        ds = label.split("/", 1)[0]
        if prev is not None and ds != prev:
            ax.axhline(i - 0.5, color="white", linewidth=0.8)
        prev = ds

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v < (vmin + vmax) / 2 else "black",
                    fontsize=6,
                )

    ax.set_yticks(range(len(scenario_labels)))
    ax.set_yticklabels(scenario_labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mmd²")
    fig.suptitle(
        "Preprocessing × Scenario mmd² (lower = source/target distribution closer)",
        fontsize=12,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] saved {save_path}")


def _draw_one_polar_ax(
    ax: plt.Axes,
    npz_path: pathlib.Path,
    prep: str,
    r_inner_pct: float,
    r_outer_pct: float,
    rng: np.random.Generator,
    max_per_group: int,
    group_style: dict,
    show_title: bool = True,
    fault_clip_pct: float = 95.0,
    zoom_factor: float = 1.3,
) -> bool:
    """Single polar scatter subplot. Returns True on success."""
    if not npz_path.exists():
        ax.set_axis_off()
        if show_title:
            ax.set_title(f"{_PREP_LABEL.get(prep, prep)}\n(missing)", fontsize=18)
        return False

    data  = np.load(npz_path)
    d_src = np.asarray(data["source_train"], dtype=float)
    d_tn  = np.asarray(data["target_normal"],  dtype=float)
    d_tf  = np.asarray(data["target_fault"],   dtype=float)
    R     = float(data["R"])

    if d_src.size == 0:
        ax.set_axis_off()
        if show_title:
            ax.set_title(f"{_PREP_LABEL.get(prep, prep)}\n(empty)", fontsize=18)
        return False

    if "rho_inner" in data and "rho_outer" in data:
        r_inner = float(data["rho_inner"]) * R
        r_outer = float(data["rho_outer"]) * R
        _inner_lbl = f"$R_{{inner}}$ (ρ={float(data['rho_inner']):.2f})"
        _outer_lbl = f"$R_{{outer}}$ (ρ={float(data['rho_outer']):.2f})"
    else:
        d_ref   = (np.asarray(data["source_val"], dtype=float)
                   if "source_val" in data and data["source_val"].size > 0 else d_src)
        r_inner = float(np.percentile(d_ref, r_inner_pct))
        r_outer = float(np.percentile(d_ref, r_outer_pct))
        _inner_lbl = f"$R_{{inner}}$ (P{int(r_inner_pct)})"
        _outer_lbl = f"$R_{{outer}}$ (P{int(r_outer_pct)})"
    # r_max: source_train / target_normal 최대값 기준 × zoom_factor
    # target_fault는 r_max 밖이면 matplotlib xlim/ylim에 의해 자연히 클리핑
    r_max_normal = max(R, r_outer, float(np.max(d_src)))
    if d_tn.size > 0:
        r_max_normal = max(r_max_normal, float(np.max(d_tn)))
    r_max = r_max_normal * zoom_factor

    for name, d in (("source_train", d_src), ("target_normal", d_tn), ("target_fault", d_tf)):
        if d.size == 0:
            continue
        if d.size > max_per_group:
            idx = rng.choice(d.size, max_per_group, replace=False)
            d = d[idx]
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d.size)
        ax.scatter(
            d * np.cos(theta), d * np.sin(theta), s=12,
            color=group_style[name]["color"],
            alpha=group_style[name]["alpha"],
            label=_GROUP_LABEL.get(name, name), edgecolors="none",
        )

    if d_tf.size > 0:
        n_clipped = int(np.sum(d_tf > r_max))
        if n_clipped > 0:
            ax.text(
                0.98, 0.02, f"+{n_clipped} TF outside",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=group_style["target_fault"]["color"], alpha=0.85,
            )

    for radius, color, lbl, lw in (
        (r_inner, "#1f77b4", _inner_lbl, 2.0),
        (r_outer, "#d62728", _outer_lbl, 2.0),
        (R,       "#888888", "R",        2.0),
    ):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, linestyle="--",
                                color=color, linewidth=lw, label=lbl))

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title(_PREP_LABEL.get(prep, prep), fontsize=22)
    return True


def plot_dual_boundary_polar(
    results_root: pathlib.Path,
    kernel: str,
    scenarios: list[tuple[str, str]],
    r_inner_pct: float = 50,
    r_outer_pct: float = 95,
    max_per_group: int = 500,
    seed: int = 42,
    prep_order: list[str] | None = None,
    show_final: bool = True,
    fault_clip_pct: float = 95.0,
    zoom_factor: float = 1.3,
) -> None:
    """SVDD dual-boundary 도식 — 동심원 R_inner/R_outer + 실거리 d(x) scatter.

    각 점 (r=d(x), θ ~ U(0, 2π)). 각도는 시각화 보조 (의미 없음).
    show_final=True(기본)이면 distances_final.npz 존재 시 pre-train / post-adaptation
    2행 레이아웃으로 자동 전환. 없으면 1행으로 fallback.

    prep_order: 그릴 prep_id 목록. None이면 PREP_ORDER(전체 6종) 사용.
    입력: {results_root}/{kernel}/{dataset}/{scenario}/{prep}/distances.npz
          {results_root}/{kernel}/{dataset}/{scenario}/{prep}/distances_final.npz  (optional)
    출력: {results_root}/{kernel}/{dataset}/{scenario}/dual_boundary_polar.png
    """
    _preps = prep_order if prep_order is not None else PREP_ORDER
    rng = np.random.default_rng(seed)

    group_style = {
        "source_train":  {"color": "#1f77b4", "alpha": 0.22},
        "target_normal": {"color": "#ff7f0e", "alpha": 0.55},
        "target_fault":  {"color": "#d62728", "alpha": 0.55},
    }

    for dataset, scenario_id in scenarios:
        run_dir = results_root / kernel / dataset / scenario_id

        def _prep_dir(prep: str) -> pathlib.Path:
            d = run_dir / prep
            if not d.exists() and prep == "p6_tds":
                d = run_dir / "p6_stat"
            return d

        has_final = show_final and any(
            (_prep_dir(p) / "distances_final.npz").exists() for p in _preps
        )
        n_rows     = 2 if has_final else 1
        row_labels = ["Pre-train", "Post-adaptation"] if has_final else [None]
        npz_names  = ["distances.npz", "distances_final.npz"] if has_final else ["distances.npz"]

        # subplot cell width ≈ figure_width * (right-left) / (n_cols + (n_cols-1)*wspace)
        # figsize height is set so each subplot cell is square → no whitespace from aspect='equal'
        _n_cols    = len(_preps)
        _fig_w     = 2.5 * _n_cols
        _wspace    = 0.05
        _cell_w    = _fig_w * (0.99 - 0.08) / (_n_cols + (_n_cols - 1) * _wspace)
        _margin_h  = 1.95   # inches: suptitle + col-headers (top) + legend (bottom)
        _hspace    = 0.06 if has_final else 0.0
        _fig_h     = n_rows * _cell_w * (1 + (n_rows - 1) * _hspace / n_rows) + _margin_h

        fig, axes = plt.subplots(
            n_rows, _n_cols,
            figsize=(_fig_w, _fig_h),
            squeeze=False,
        )

        for row, (npz_name, row_label) in enumerate(zip(npz_names, row_labels)):
            for col, prep in enumerate(_preps):
                ax = axes[row, col]
                npz_path = _prep_dir(prep) / npz_name
                _draw_one_polar_ax(
                    ax, npz_path, prep,
                    r_inner_pct, r_outer_pct,
                    rng, max_per_group, group_style,
                    show_title=(row == 0),
                    fault_clip_pct=fault_clip_pct,
                    zoom_factor=zoom_factor,
                )
            if row_label is not None:
                axes[row, 0].set_ylabel(row_label, fontsize=18, labelpad=14)

        handles, labels = [], []
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
        seen, unique = set(), []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique.append((h, l))

        _legend_h  = 1.05   # inches reserved for legend
        _title_h   = 0.90   # inches reserved for suptitle + col-header titles
        plot_bottom = _legend_h / _fig_h
        plot_top    = 1.0 - _title_h / _fig_h
        legend_y    = plot_bottom - 0.01

        if unique:
            fig.legend(
                *zip(*unique), loc="upper center", ncol=3,
                fontsize=16, bbox_to_anchor=(0.5, legend_y), frameon=False,
                markerscale=3.5,
            )

        fig.suptitle(
            f"{dataset.upper()} {scenario_id.replace('_to_', ' → ')} — {kernel.upper()} kernel",
            fontsize=20, y=0.99,
        )
        fig.subplots_adjust(
            top=plot_top, bottom=plot_bottom,
            left=0.08, right=0.99,
            hspace=_hspace, wspace=_wspace,
        )

        out_dir = results_root / kernel / dataset / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dual_boundary_polar.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_tradeoff_csv(
    eval_out: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """R_outer sweep별 Adaptation% vs AUC trade-off 요약 CSV 생성.

    입력:
      {eval_out}/{kernel}/zone_ratio_by_percentile.csv  — zone 점유율
      {eval_out}/AD_performance_all.csv                 — AUC/F1 시나리오별
    출력: out_path (CSV)
    """
    zone_frames: list[pd.DataFrame] = []
    for kernel in KERNELS:
        z_path = eval_out / kernel / "zone_ratio_by_percentile.csv"
        if not z_path.exists():
            continue
        df = pd.read_csv(z_path)
        df = df[df["group"] == "target_normal"].copy()
        avg = (
            df.groupby(["preprocessing", "r_inner_percentile", "r_outer_percentile"])
              [["inner_pct", "adaptation_pct", "outer_pct"]]
              .mean()
              .reset_index()
        )
        avg["kernel"] = kernel
        zone_frames.append(avg)

    full_path = eval_out / "AD_performance_all.csv"
    if not zone_frames or not full_path.exists():
        print("[tradeoff] 필요 CSV 없음 — skip")
        return

    zone_df = pd.concat(zone_frames, ignore_index=True)
    zone_df = zone_df[abs(zone_df["r_inner_percentile"] - 50.0) < 1e-6].copy()
    zone_df = zone_df.rename(columns={
        "r_inner_percentile": "r_inner_pct",
        "r_outer_percentile": "r_outer_pct",
    })

    full_df = pd.read_csv(full_path)

    def _pivot_metric(col: str, col_prefix: str) -> pd.DataFrame:
        agg = (
            full_df.groupby(["dataset", "prep"])[col]
            .mean()
            .reset_index()
        )
        pivoted = agg.pivot(index="prep", columns="dataset", values=col)
        pivoted.columns = [f"{col_prefix}_{c}" for c in pivoted.columns]
        pivoted[f"{col_prefix}_avg"] = pivoted.mean(axis=1)
        return pivoted.reset_index().rename(columns={"prep": "preprocessing"})

    auc_pivot = _pivot_metric("AUC", "auc")
    f1_pivot  = _pivot_metric("F1",  "f1")

    merged = (
        zone_df
        .merge(auc_pivot, on="preprocessing", how="left")
        .merge(f1_pivot,  on="preprocessing", how="left")
    )
    merged = merged.sort_values(["preprocessing", "kernel", "r_outer_pct"])

    base_cols = ["preprocessing", "kernel", "r_outer_pct",
                 "adaptation_pct", "inner_pct", "outer_pct"]
    auc_cols = [c for c in merged.columns if c.startswith("auc_")]
    f1_cols  = [c for c in merged.columns if c.startswith("f1_")]
    ordered  = base_cols + sorted(auc_cols) + sorted(f1_cols)
    available = [c for c in ordered if c in merged.columns]
    merged[available].to_csv(out_path, index=False, float_format="%.4f")
    print(f"[tradeoff] saved {out_path}  ({len(merged)} rows)")


def plot_cm_by_dataset(
    cm_csv: pathlib.Path,
    out_dir: pathlib.Path,
    kernel_filter: str | None = None,
    suffix: str = "",
) -> None:
    """데이터셋별 전처리 confusion matrix 시각화.

    입력: cm_csv (AD_performance_all.csv — TP/TN/FP/FN 포함)
    출력: {out_dir}/cm_{dataset}{suffix}.png
    kernel_filter 지정 시 해당 커널 행만 사용.
    """
    raw = pd.read_csv(cm_csv)
    if kernel_filter is not None and "kernel" in raw.columns:
        raw = raw[raw["kernel"] == kernel_filter]
    col_prep_raw = "prep" if "prep" in raw.columns else "preprocessing"
    raw[col_prep_raw] = raw[col_prep_raw].replace({"p6_stat": "p6_tds"})
    df = (
        raw.groupby(["dataset", col_prep_raw])[["TP", "TN", "FP", "FN"]]
        .sum().reset_index()
    )
    col_ds   = "dataset"
    col_prep = col_prep_raw
    prep_order_local = [p for p in PREP_ORDER if p in df[col_prep].unique()]

    for dataset in DATASET_ORDER:
        sub = df[df[col_ds] == dataset]
        if sub.empty:
            continue

        n = len(prep_order_local)
        fig, axes = plt.subplots(2, 2, figsize=(8, 7))
        axes_flat = axes.flatten()

        for ax, prep in zip(axes_flat, prep_order_local):
            row = sub[sub[col_prep] == prep]
            if row.empty:
                ax.axis("off")
                ax.set_title(_PREP_LABEL.get(prep, prep), fontsize=10)
                continue

            cm_mat = np.array([
                [int(row["TN"].iloc[0]), int(row["FP"].iloc[0])],
                [int(row["FN"].iloc[0]), int(row["TP"].iloc[0])],
            ], dtype=float)
            row_sums = cm_mat.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm_mat / row_sums, 0.0)

            im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Normal", "Fault"], fontsize=9)
            ax.set_yticklabels(["Normal", "Fault"], fontsize=9)
            ax.set_xlabel("Predicted label", fontsize=10)
            ax.set_ylabel("Actual label", fontsize=10)
            for i in range(2):
                for j in range(2):
                    rate = cm_norm[i, j]
                    ax.text(j, i, f"{rate:.3f}", ha="center", va="center", fontsize=11,
                            color="white" if rate > 0.5 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(_PREP_LABEL.get(prep, prep), fontsize=10, pad=10)

        for ax in axes_flat[n:]:
            ax.axis("off")

        fig.suptitle(f"{dataset.upper()} — Confusion Matrices by Preprocessing", fontsize=13)
        fig.tight_layout()
        out_path = out_dir / f"cm_{dataset}{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[analysis] saved {out_path}")
