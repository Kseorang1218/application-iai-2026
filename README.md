# application-iai-2026
[2026-1] 산업인공지능응용 — DualBoundarySVDD 기반 Online Test-Time Adaptation (OTTA) for Anomaly Detection

## 환경 설정

```bash
source iai/bin/activate
pip install -r requirements.txt
```

## 실행 방법

### 전체 파이프라인 (`main.py`)

실험 → OTTA 평가를 순서대로 실행하며 로그를 `log/`에 저장.

```bash
python main.py --out-dir ./results/0608              # cwru + pu 모두 실행
python main.py --out-dir ./results/0608 --dataset cwru
python main.py --out-dir ./results/0608 --dataset pu
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--out-dir` | (필수) | 결과 저장 경로 |
| `--dataset` | (없으면 전체) | `cwru` 또는 `pu`. 미지정 시 둘 다 실행 |

### 단일 실험 (`run.py`)

```bash
python run.py --dataset cwru --out-dir ./results/0608
python run.py --dataset pu   --out-dir ./results/0608
```

## 결과 구조

```
results/<date>/
├── <dataset>/
│   └── <source_rpm>_to_<target_rpm>/
│       └── p4_cepstrum/
│           ├── svdd_model.npz   # 최종 적응 모델 (Support Vectors, alpha)
│           └── svdd_model.json  # 모델 메타 (kernel, C, R2 등)
└── evaluation/
    └── otta_performance_all.csv
log/
└── run_<timestamp>.log
```

### 저장된 모델 로드

```python
from models.svdd import SVDD
model = SVDD.load("results/0608/cwru/1730_to_1750/p4_cepstrum/svdd_model")
scores = model.decision_function(X_new)  # 양수: 정상, 음수: 이상
```
