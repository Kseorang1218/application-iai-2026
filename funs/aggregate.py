"""run.py 산출물 집계."""
from __future__ import annotations

import json
import pathlib

import pandas as pd


def aggregate_summary(
    results_root: pathlib.Path,
    out_dir: pathlib.Path | None = None,
    kernel: str | None = None,
) -> pd.DataFrame:
    """모든 (dataset, scenario, prep) 결과를 스캔하여 summary CSV 생성.

    산출물:
      zone_ratio_by_percentile.csv — percentile 조합 × group 전체 zone 점유율
                                     (zone_ratio.json 이 없으면 빈 결과)

    반환:
      metrics_df — run × 핵심 metric DataFrame (kernel 컬럼 포함 시). 호출측에서
                   전 커널 병합 후 domain_shift_performance.csv 로 저장.
    """
    summary_dir = pathlib.Path(out_dir) if out_dir is not None else results_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    zone_full_path = summary_dir / "zone_ratio_by_percentile.csv"

    metric_rows: list[dict] = []
    zone_rows:   list[dict] = []
    for metrics_path in results_root.rglob("metrics.json"):
        try:
            with open(metrics_path) as fp:
                m = json.load(fp)
        except json.JSONDecodeError:
            continue
        rel = metrics_path.relative_to(results_root)
        # rel = {dataset}/{scenario_id}/{prep_id}/metrics.json
        if len(rel.parts) < 4:
            continue
        dataset, scenario_id, prep_id = rel.parts[0], rel.parts[1], rel.parts[2]
        metric_rows.append({
            "dataset":                 dataset,
            "scenario_id":             scenario_id,
            "preprocessing":           prep_id,
            "R":                       m.get("R"),
            "mmd2":                    m.get("mmd2"),
            "wasserstein_1d_distance": m.get("wasserstein_1d_distance"),
            "feature_dim":             m.get("feature_dim"),
            "n_source_train":          m.get("n_source_train"),
            "n_target_normal":         m.get("n_target_normal"),
            "n_target_fault":          m.get("n_target_fault"),
        })

        zone_path = metrics_path.parent / "zone_ratio.json"
        if not zone_path.exists():
            continue
        try:
            with open(zone_path) as fp:
                z = json.load(fp)
        except json.JSONDecodeError:
            continue
        for combo in z.get("zone_combinations", []):
            for group_name, stats in combo.get("by_group", {}).items():
                zone_rows.append({
                    "dataset":            dataset,
                    "scenario_id":        scenario_id,
                    "preprocessing":      prep_id,
                    "group":              group_name,
                    "r_inner_percentile": combo.get("r_inner_percentile"),
                    "r_outer_percentile": combo.get("r_outer_percentile"),
                    "r_inner":            combo.get("r_inner"),
                    "r_outer":            combo.get("r_outer"),
                    **{k: stats.get(k) for k in
                       ("count", "inner_pct", "adaptation_pct", "outer_pct", "mean_d", "std_d")},
                })

    if not metric_rows:
        print("[summary] no metrics.json found — skip")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["dataset", "scenario_id", "preprocessing"]
    )
    if kernel is not None:
        metrics_df.insert(0, "kernel", kernel)

    if zone_rows:
        zone_df = pd.DataFrame(zone_rows)
        zone_df.to_csv(zone_full_path, index=False)

    print(f"[summary] aggregated {kernel or 'unknown'}  ({len(metrics_df)} rows)")
    return metrics_df
