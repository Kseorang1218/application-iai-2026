#!/usr/bin/env bash
set -euo pipefail

# ── 인자 파싱 및 필수 인자 검사 ─────────────────────────────────────────
DIR_TO_SAVE="" # 초기값을 비워둠

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --date)
            DIR_TO_SAVE="$2"
            shift 2
            ;;
        --help)
            echo "사용법: $0 --date <디렉토리명>"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

# DIR_TO_SAVE가 비어있는지(-z) 확인하여 비어있다면 에러 출력 후 종료
if [[ -z "$DIR_TO_SAVE" ]]; then
    echo "오류: --date 인자는 필수입니다. 값을 지정해주세요."
    echo "사용법: $0 --date <디렉토리명>"
    exit 1
fi
# ───────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

echo "로그 파일 경로: $LOG_FILE"
echo "저장 디렉토리: $DIR_TO_SAVE"

{
    source "$SCRIPT_DIR/iai/bin/activate"
    export PYTHONUNBUFFERED=1

    # ── Step 1: Main experiments (Linear kernel) ────────────────────────────
    python "$SCRIPT_DIR/run.py" --dataset cwru --kernel linear --out-dir "./results/$DIR_TO_SAVE/linear" --workers 8

    # # ── Step 2: Main experiments (RBF kernel) ───────────────────────────────
    # python "$SCRIPT_DIR/run.py" --dataset pu --kernel rbf --out-dir "./results/$DIR_TO_SAVE/rbf" --workers 8

    # ── Step 3: 사후 분석 (전 커널 일괄 — per-kernel + cross-kernel 산출물 모두 생성)
    python "$SCRIPT_DIR/analysis.py" --results-root "./results/$DIR_TO_SAVE"

    # ── Step 4: OTTA streaming 결과 평가 (전 커널 일괄, otta_stream.npz 기반)
    python "$SCRIPT_DIR/analysis_otta.py" --results-root "./results/$DIR_TO_SAVE"

} < /dev/null > "$LOG_FILE" 2>&1 &

BG_PID=$!
disown $BG_PID

sleep 1
tail -f --pid=$BG_PID "$LOG_FILE"