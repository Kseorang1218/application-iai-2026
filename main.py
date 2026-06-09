"""전체 실험 파이프라인 오케스트레이터.

사용법:
  python main.py --out-dir ./results/0607

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
        "--out-dir", required=True, metavar="DIR",
        help="실험 결과 저장 경로 (예: ./results/0607)",
    )
    parser.add_argument(
        "--dataset", default=None, choices=["cwru", "pu"],
        help="사용할 데이터셋. 미지정 시 cwru + pu 모두 실행",
    )
    args = parser.parse_args()

    log_dir = _ROOT / "log"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    out_dir = pathlib.Path(args.out_dir)
    results_root = out_dir

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = _Tee(sys.stdout, log_file)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = tee

        try:
            tee.write(f"로그 경로: {log_path}\n")
            tee.write(f"결과 경로: {results_root}\n\n")

            # ── Step 1: 실험 ───────────────────────────────────────────────
            t0 = time.time()
            tee.write(f"{'='*60}\n[Step 1] 실험 시작\n{'='*60}\n")
            datasets = [args.dataset] if args.dataset else ["cwru", "pu"]
            for ds in datasets:
                tee.write(f"\n--- 데이터셋: {ds} ---\n")
                run_experiment(
                    dataset=ds,
                    out_dir=str(out_dir),
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
