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
python main.py --out-dir ./results/0607
```

옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--out-dir` | (필수) | 실험 결과 저장 경로 (예: `./results/0607`) |
| `--dataset` | (없으면 전체) | `cwru` 또는 `pu`. 미지정 시 둘 다 실행 |
| `--workers` | `8` | 병렬 시나리오 수 |
| `--otta-mode` | `dual_boundary` | OTTA 모드 (`dual_boundary`, `single_boundary`) |

로그는 `log/run_<timestamp>.log`에 저장


## 전처리

Cepstrum (`p4_cepstrum`) 단일 사용.

| 데이터셋 | window size | lifter_n |
|---------|------------|---------|
| CWRU | 2048 | 64 |
| PU | 8192 | 4096 |

raw window → per-window z-score → real cepstrum → liftering

## 결과 구조

```
results/<date>/cwru/<scenario_id>/p4_cepstrum/
    otta_stream.npz
    distances.npz
    metrics.json
results/<date>/pu/<scenario_id>/p4_cepstrum/
    otta_stream.npz
    distances.npz
    metrics.json
results/<date>/evaluation/
    AD_performance_all.csv
    otta_performance_all.csv
log/
    run_<timestamp>.log
```
