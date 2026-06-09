# application-iai-2026
[2026-1] 산업인공지능응용 — DualBoundarySVDD 기반 Online Test-Time Adaptation (OTTA) for Anomaly Detection

## 환경 설정

```bash
source iai/bin/activate
pip install -r requirements.txt
```

## 데이터셋

`dataset/` 디렉토리 구조:

```
dataset/
├── cwru/          # CWRU 12k 진동 데이터 (.mat)
└── pu/            # Paderborn University 진동 데이터
    ├── K001/      # 정상 베어링
    └── KI04/      # 내륜 결함 베어링
```

| 데이터셋 | 도메인 | 샘플링 주파수 | window size | lifter_n |
|---------|--------|-------------|-------------|---------|
| CWRU | A: 1730 rpm, D: 1797 rpm | 12,000 Hz | 2048 | 64 |
| PU | A: 900 rpm, B: 1500 rpm | 64,000 Hz | 4096 | 512 |

## 실행 방법

### 단일 실행 (`run.py`)

```bash
python run.py --dataset cwru --out-dir ./results/0609
python run.py --dataset pu --out-dir ./results/0609
python run.py --dataset cwru --source A --target D --out-dir ./results/0609
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `cwru` | `cwru` 또는 `pu` |
| `--source` | (전체) | 소스 도메인 키 (예: `A`) |
| `--target` | (전체) | 타겟 도메인 키 (예: `D`) |
| `--out-dir` | `results` | 결과 저장 경로 |
| `--otta-mode` | `dual_boundary` | `dual_boundary` 또는 `single_boundary` |
| `--workers` | `1` | 병렬 시나리오 수 |

### 전체 파이프라인 (`main.py`)

실험 → 사후 분석 → OTTA 평가를 순서대로 실행하며 로그를 `log/`에 저장.

```bash
python main.py --out-dir ./results/0609
python main.py --out-dir ./results/0609 --dataset cwru
python main.py --out-dir ./results/0609 --workers 4
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--out-dir` | (필수) | 결과 저장 경로 |
| `--dataset` | (없으면 전체) | `cwru` 또는 `pu`. 미지정 시 둘 다 실행 |
| `--workers` | `8` | 병렬 시나리오 수 |
| `--otta-mode` | `dual_boundary` | OTTA 모드 |

### 사후 분석만 실행

```bash
python analysis.py --results-root ./results/0609
python analysis_otta.py --results-root ./results/0609
```

## OTTA 모드

| 모드 | 설명 |
|------|------|
| `dual_boundary` | DualBoundarySVDD — `r_inner`/`r_outer` 고정 3-zone 분기 (inner: skip, middle: adapt, outer: anomaly) |
| `single_boundary` | OnlineSVDD — warmup 후 예측 기반 TTA (정상→adapt, 고장→skip) |

## 전처리

raw window → per-window z-score → real cepstrum → liftering (`p4_cepstrum` 단일 사용)

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
