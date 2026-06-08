# application-iai-2026
[2026-1] 산업인공지능응용

## 환경 설정

```bash
source iai/bin/activate
pip install -r requirements.txt
```

## 실행 방법

### 전체 파이프라인

```bash
python main.py --date <결과_디렉토리명>
```

예시:
```bash
python main.py --date 0607
```

옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--date` | (필수) | 결과 저장 디렉토리명 (`results/<date>/`) |
| `--kernel` | `linear` | SVDD 커널 (`linear`, `rbf`, `poly`) |
| `--dataset` | `cwru` | 데이터셋 (`cwru`, `pu`, `uos`) |
| `--workers` | `8` | 병렬 시나리오 수 |
| `--otta-mode` | `dual_boundary` | OTTA 모드 (`dual_boundary`, `single_boundary`) |

로그는 `log/run_<timestamp>.log`에 저장됩니다.

---

### 개별 실행

**실험만:**
```bash
python run.py --dataset cwru --kernel linear --out-dir ./results/0607/linear --workers 8
python run.py --dataset cwru --kernel rbf --source A --target B  # 특정 시나리오
```

**분석만:**
```bash
python analysis.py --results-root ./results/0607
python analysis_otta.py --results-root ./results/0607
```

## 결과 구조

```
results/<date>/<kernel>/<dataset>/<scenario_id>/<prep_id>/
    otta_stream.npz
    distances.npz
    metrics.json
results/<date>/evaluation/
    <kernel>/run_metrics.csv
    AD_performance_all.csv
    otta_performance_all.csv (per kernel)
log/
    run_<timestamp>.log
```
