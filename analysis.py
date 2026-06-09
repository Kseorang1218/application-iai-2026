from __future__ import annotations

import argparse
import pathlib

def run_analysis(
    results_root: pathlib.Path,
    out_dir: pathlib.Path | None = None,
) -> None:
    pass


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
