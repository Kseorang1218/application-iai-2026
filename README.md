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
python main.py --out-dir <결과_디렉토리_경로>
```

예시:
```bash
python main.py --out-dir ./results/0607/linear
```

옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--out-dir` | (필수) | 실험 결과 저장 경로 (예: `./results/0607/linear`) |
| `--workers` | `8` | 병렬 시나리오 수 |
| `--otta-mode` | `dual_boundary` | OTTA 모드 (`dual_boundary`, `single_boundary`) |

로그는 `log/run_<timestamp>.log`에 저장됩니다.

---

### 개별 실행

**실험만:**
```bash
python run.py --out-dir ./results/0607/linear --workers 8
python run.py --source A --target B  # 특정 시나리오
```

**분석만:**
```bash
python analysis.py --results-root ./results/0607
python analysis_otta.py --results-root ./results/0607
```

## 결과 구조

```
results/<date>/linear/cwru/<scenario_id>/<prep_id>/
    otta_stream.npz
    distances.npz
    metrics.json
results/<date>/evaluation/
    linear/run_metrics.csv
    AD_performance_all.csv
    otta_performance_all.csv
log/
    run_<timestamp>.log
```
