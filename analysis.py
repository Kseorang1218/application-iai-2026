"""Main 결과 사후 분석 — 오케스트레이터.

`run.py` 가 모든 (kernel, dataset, scenario, prep) 디렉토리를 채운 뒤 실행한다.
한 번의 호출로 전 커널을 일괄 처리한다: per-kernel 단계(Steps 1, 6)는 발견된 커널마다
반복하고, cross-kernel 단계(Steps 2-5, 7)는 전 커널을 묶어 수행한다. Step 4 의 CM 은
cross-kernel(`analysis/`) + per-kernel(`analysis/{kernel}/`) 양쪽 모두 생성한다.

I/O 요약 (--results-root 를 ROOT 로 표기):
  Step 1  aggregate_summary
    reads : ROOT/{kernel}/{ds}/{sc}/{prep}/metrics.json
            ROOT/{kernel}/{ds}/{sc}/{prep}/zone_ratio.json
    writes: ROOT/evaluation/{kernel}/run_metrics.csv       (kernel 컬럼 포함)
            ROOT/evaluation/{kernel}/zone_ratio_by_percentile.csv

  Step 2  merge_run_metrics
    reads : ROOT/evaluation/{kernel}/run_metrics.csv
    writes: ROOT/evaluation/domain_shift_performance.csv   (전 커널 병합)

  Step 3  evaluate_ad_performance
    reads : ROOT/{kernel}/{ds}/{sc}/{prep}/distances.npz
    writes: ROOT/{kernel}/{ds}/{sc}/{prep}/confusion_matrix.png  (per-run)
            ROOT/{kernel}/{ds}/{sc}/{prep}/confusion_matrix.json (per-run)
            ROOT/evaluation/AD_performance_all.csv

  Step 4  plot_cm_by_dataset
    reads : ROOT/evaluation/AD_performance_all.csv
    writes: ROOT/analysis/{kernel}/cm_{dataset}.png   (per-kernel)

  Step 5  plot_tradeoff_csv
    reads : ROOT/evaluation/{kernel}/zone_ratio_by_percentile.csv
            ROOT/evaluation/AD_performance_all.csv
    writes: ROOT/evaluation/tradeoff_summary.csv

  Step 6  plot_dual_boundary_polar
    reads : ROOT/{kernel}/{ds}/{sc}/{prep}/distances.npz
    writes: ROOT/{kernel}/{ds}/{sc}/dual_boundary_polar.png  (all scenarios)

  Step 7  summarize_distance_ratios
    reads : ROOT/{kernel}/{ds}/{sc}/{prep}/distances.npz
    writes: ROOT/analysis/distance_ratio_summary.csv
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

import funs
from funs.aggregate import aggregate_summary
from funs.evaluation import (
    KERNELS,
    build_rpm_domain_map,
    evaluate_ad_performance,
    summarize_distance_ratios,
)
from funs.visualize import (
    plot_cm_by_dataset,
    plot_dual_boundary_polar,
    plot_dual_boundary_polar_grid,
    plot_tradeoff_csv,
)

def _discover_scenarios(results_root: pathlib.Path, kernel: str) -> list[tuple[str, str]]:
    """results_root/{kernel}/{dataset}/{scenario_id}/ 구조에서 모든 시나리오 탐색.
    커널 서브디렉토리가 없으면 results_root 자체를 탐색.
    scenario_id 는 '{rpm}_to_{rpm}' 형태인 것만 포함 (analysis/, evaluation/ 등 배제)."""
    kernel_root = _kernel_root(results_root, kernel)
    pairs: list[tuple[str, str]] = []
    if not kernel_root.exists():
        return pairs
    for dataset_dir in sorted(kernel_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for scenario_dir in sorted(dataset_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            if "_to_" not in scenario_dir.name:
                continue
            pairs.append((dataset_dir.name, scenario_dir.name))
    return pairs


def _discover_kernels(results_root: pathlib.Path) -> list[str]:
    """results_root 하위에 존재하는 커널 디렉토리 목록 (KERNELS 순서 유지).
    커널 서브디렉토리 없이 results_root 바로 아래에 데이터가 있으면 ["linear"] 반환."""
    found = [k for k in KERNELS if (results_root / k).is_dir()]
    if not found and any(results_root.rglob("metrics.json")):
        found = ["linear"]
    return found


def _kernel_root(results_root: pathlib.Path, kernel: str) -> pathlib.Path:
    """커널 서브디렉토리가 있으면 results_root/kernel, 없으면 results_root 자체 반환."""
    candidate = results_root / kernel
    return candidate if candidate.is_dir() else results_root


def run_analysis(
    results_root: pathlib.Path,
    out_dir: pathlib.Path | None = None,
) -> None:
    results_root = pathlib.Path(results_root)
    analysis_dir = out_dir or (results_root / "analysis")
    eval_dir     = results_root / "evaluation"

    kernels = _discover_kernels(results_root)
    if not kernels:
        print(f"[analysis] {results_root} 에서 커널 디렉토리({'/'.join(KERNELS)})를 찾지 못함 — 종료")
        return
    print(f"[analysis] 발견된 커널: {kernels}")

    cfg = funs.get_config()
    rpm_to_domain = build_rpm_domain_map(cfg)
    preps = list(cfg.preprocessing_ids.values())

    # ── Step 1 (per-kernel): aggregate → run_metrics.csv + zone_ratio_by_percentile.csv
    for kernel in kernels:
        print(f"\n[summary] aggregating {kernel} …")
        df = aggregate_summary(_kernel_root(results_root, kernel), out_dir=eval_dir / kernel, kernel=kernel)
        if not df.empty:
            (eval_dir / kernel).mkdir(parents=True, exist_ok=True)
            run_metrics_path = eval_dir / kernel / "run_metrics.csv"
            df.to_csv(run_metrics_path, index=False)
            print(f"[summary] run_metrics.csv → {run_metrics_path}  ({len(df)} rows)")

    # ── Step 2 (cross-kernel): per-kernel run_metrics 병합 → domain_shift_performance.csv
    frames = [
        pd.read_csv(eval_dir / k / "run_metrics.csv")
        for k in kernels if (eval_dir / k / "run_metrics.csv").exists()
    ]
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / "domain_shift_performance.csv"
        merged.to_csv(out_path, index=False)
        print(f"[summary] domain_shift_performance.csv → {out_path}  ({len(merged)} rows)")

    # ── Step 3 (cross-kernel): distances.npz → AD_performance_all.csv + per-run CM
    evaluate_ad_performance(results_root, eval_dir, rpm_to_domain=rpm_to_domain)

    # ── Step 4: AD_performance_all.csv → cm_{dataset}.png (per-kernel)
    cm_csv = eval_dir / "AD_performance_all.csv"
    if cm_csv.exists():
        for kernel in kernels:
            (analysis_dir / kernel).mkdir(parents=True, exist_ok=True)
            plot_cm_by_dataset(cm_csv, analysis_dir / kernel, kernel_filter=kernel, suffix="_baseline")

    # ── Step 5 (cross-kernel): zone_ratio + AD_performance → tradeoff_summary.csv
    plot_tradeoff_csv(eval_dir, eval_dir / "tradeoff_summary.csv")

    # ── Step 6 (per-kernel): distances.npz → {ds}/{sc}/dual_boundary_polar.png
    for kernel in kernels:
        scenarios = _discover_scenarios(results_root, kernel)
        if scenarios:
            print(f"\n[dual_boundary] {kernel}: {len(scenarios)} scenarios")
            plot_dual_boundary_polar(_kernel_root(results_root, kernel), kernel, scenarios, prep_order=preps)

    # ── Step 6b (per-kernel): 전 시나리오 통합 polar grid → analysis/{kernel}/dual_boundary_polar.png
    for kernel in kernels:
        scenarios = _discover_scenarios(results_root, kernel)
        if scenarios and preps:
            (analysis_dir / kernel).mkdir(parents=True, exist_ok=True)
            plot_dual_boundary_polar_grid(
                _kernel_root(results_root, kernel), kernel, scenarios,
                out_path=analysis_dir / kernel / "dual_boundary_polar.png",
                prep=preps[0],
                rpm_domain_map=rpm_to_domain,
            )

    # ── Step 7 (cross-kernel): distances.npz → analysis/distance_ratio_summary.csv
    summarize_distance_ratios(results_root, analysis_dir, rpm_to_domain=rpm_to_domain)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Main 결과 사후 분석 (전 커널 일괄: aggregate + AD evaluation + 시각화)"
    )
    parser.add_argument(
        "--results-root", type=pathlib.Path, required=True,
        help="run.py 산출물이 들어 있는 루트 (예: results/0504).",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=None,
        help="출력 디렉토리. 미지정 시 `<results-root>/analysis`",
    )
    args = parser.parse_args()
    run_analysis(results_root=args.results_root, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
