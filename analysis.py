"""Main 결과 사후 분석 — 오케스트레이터.

`run.py` 가 모든 (kernel, dataset, scenario, prep) 디렉토리를 채운 뒤 실행한다.

I/O 요약 (--results-root 를 ROOT 로 표기):
  evaluate_ad_performance
    reads : ROOT/{kernel}/{ds}/{sc}/{prep}/distances.npz
    writes: ROOT/evaluation/AD_performance_all.csv
"""
from __future__ import annotations

import argparse
import pathlib

import funs
from funs.evaluation import (
    KERNELS,
    build_rpm_domain_map,
    evaluate_ad_performance,
)

def _discover_kernels(results_root: pathlib.Path) -> list[str]:
    """results_root 하위에 존재하는 커널 디렉토리 목록 (KERNELS 순서 유지).
    커널 서브디렉토리 없이 results_root 바로 아래에 데이터가 있으면 ["linear"] 반환."""
    found = [k for k in KERNELS if (results_root / k).is_dir()]
    if not found and any(results_root.rglob("metrics.json")):
        found = ["linear"]
    return found


def run_analysis(
    results_root: pathlib.Path,
    out_dir: pathlib.Path | None = None,
) -> None:
    results_root = pathlib.Path(results_root)
    eval_dir = results_root / "evaluation"

    kernels = _discover_kernels(results_root)
    if not kernels:
        print(f"[analysis] {results_root} 에서 커널 디렉토리({'/'.join(KERNELS)})를 찾지 못함 — 종료")
        return
    print(f"[analysis] 발견된 커널: {kernels}")

    cfg = funs.get_config()
    rpm_to_domain = build_rpm_domain_map(cfg)

    evaluate_ad_performance(results_root, eval_dir, rpm_to_domain=rpm_to_domain)


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
