"""전체 실험 파이프라인 오케스트레이터.

사용법:
  python main.py --date 0607
  python main.py --date 0607 --kernel rbf --workers 4

로그: log/run_<timestamp>.log (stdout 동시 출력)
"""
import argparse
import pathlib
import sys
import time
from datetime import datetime

from run import run_experiment
from analysis import run_analysis
from analysis_otta import run_analysis_otta

_ROOT = pathlib.Path(__file__).parent


class _Tee:
    """stdout을 터미널과 파일에 동시에 쓰는 래퍼."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()


def main() -> None:
    parser = argparse.ArgumentParser(description="전체 실험 파이프라인")
    parser.add_argument(
        "--date", required=True, metavar="DIR",
        help="결과 저장 디렉토리명 (예: 0607 → results/0607/)",
    )
    parser.add_argument(
        "--kernel", default="linear", choices=["linear", "rbf", "poly"],
        help="SVDD 커널 (기본값: linear)",
    )
    parser.add_argument(
        "--dataset", default="cwru", choices=["cwru", "pu", "uos"],
        help="데이터셋 (기본값: cwru)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="N",
        help="병렬 시나리오 수 (기본값: 8)",
    )
    parser.add_argument(
        "--otta-mode", default="dual_boundary",
        choices=["dual_boundary", "single_boundary"],
        help="OTTA 모드 (기본값: dual_boundary)",
    )
    args = parser.parse_args()

    log_dir = _ROOT / "log"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    results_root = _ROOT / "results" / args.date
    out_dir = results_root / args.kernel

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = _Tee(sys.stdout, log_file)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = tee

        try:
            tee.write(f"로그 경로: {log_path}\n")
            tee.write(f"결과 경로: {results_root}\n")
            tee.write(f"커널: {args.kernel} | 데이터셋: {args.dataset} | workers: {args.workers}\n\n")

            # ── Step 1: 실험 ───────────────────────────────────────────────
            t0 = time.time()
            tee.write(f"{'='*60}\n[Step 1] 실험 시작\n{'='*60}\n")
            run_experiment(
                dataset=args.dataset,
                kernel=args.kernel,
                out_dir=str(out_dir),
                workers=args.workers,
                otta_mode=args.otta_mode,
            )
            tee.write(f"\n[Step 1 완료] {time.time()-t0:.1f}s\n")

            # ── Step 2: 사후 분석 ──────────────────────────────────────────
            t1 = time.time()
            tee.write(f"\n{'='*60}\n[Step 2] 사후 분석 시작\n{'='*60}\n")
            run_analysis(results_root=results_root)
            tee.write(f"\n[Step 2 완료] {time.time()-t1:.1f}s\n")

            # ── Step 3: OTTA 평가 ──────────────────────────────────────────
            t2 = time.time()
            tee.write(f"\n{'='*60}\n[Step 3] OTTA 평가 시작\n{'='*60}\n")
            run_analysis_otta(results_root=results_root)
            tee.write(f"\n[Step 3 완료] {time.time()-t2:.1f}s\n")

            elapsed = time.time() - t0
            tee.write(f"\n{'='*60}\n전체 완료: {elapsed:.1f}s\n{'='*60}\n")

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    print(f"로그 저장: {log_path}")


if __name__ == "__main__":
    main()
