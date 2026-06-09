# application-iai-2026
[2026-1] 산업인공지능응용 — DualBoundarySVDD 기반 Online Test-Time Adaptation (OTTA) for Anomaly Detection

## 환경 설정

```bash
source iai/bin/activate
pip install -r requirements.txt
```

## 실행 방법

### 단일 실행 (`run.py`)

```bash
python run.py --dataset cwru --out-dir ./results/0609
python run.py --dataset pu --out-dir ./results/0609
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `cwru` | `cwru` 또는 `pu` |
| `--out-dir` | `results` | 결과 저장 경로 |

### 전체 파이프라인 (`main.py`)

실험 → 사후 분석 → OTTA 평가를 순서대로 실행하며 로그를 `log/`에 저장.

```bash
python main.py --out-dir ./results/0609
python main.py --out-dir ./results/0609 --dataset cwru
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--out-dir` | (필수) | 결과 저장 경로 |
| `--dataset` | (없으면 전체) | `cwru` 또는 `pu`. 미지정 시 둘 다 실행 |

## 결과 구조

```
results/<date>/
├── cwru/<scenario_id>/p4_cepstrum/
│   ├── otta_stream.npz
│   ├── distances.npz
│   ├── distances_final.npz
│   ├── svdd_model.npz
│   └── metrics.json
├── pu/<scenario_id>/p4_cepstrum/
│   └── ...
└── evaluation/
    ├── AD_performance_all.csv
    └── otta_performance_all.csv
log/
    run_<timestamp>.log
```
