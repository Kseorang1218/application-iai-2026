# application-iai-2026
[2026-1] 산업인공지능응용 — Dual-BoundarySVDD 기반 OTTA를 통한 베어링 결함 탐지

> 💡 **[참고사항]**
> 본 코드는 제출을 위해 데이터셋 일부 파일만을 테스트 파일로 포함합니다.
> 전체 실험 코드는 [application-iai-2026 저장소](https://github.com/Kseorang1218/application-iai-2026.git)에서 확인하실 수 있습니다.

## 환경 설정

```bash
python -m venv iai      # 가상환경 생성
source iai/bin/activate
pip install -r requirements.txt
```

## 실행 방법

### 전체 파이프라인 (`main.py`)

실험을 실행하고 로그를 `log/`에 저장

```bash
python main.py --out-dir ./results
```

| 옵션 | 설명 |
|------|------|
| `--out-dir` | (필수) 결과 저장 경로 |
| `--dataset` | `cwru` 또는 `pu`. 미지정 시 둘 다 순차 실행 |


## 결과 구조

```
results/<date>/
├── <dataset>/
│   └── <source_rpm>_to_<target_rpm>/
│       ├── otta_stream.npz       # decisions, scores, y_true, latencies, R_trace, R_pretrain, R_final
│       ├── distances.npz         # pre-train 모델 거리 (source_train/val, target_normal/fault, R)
│       ├── distances_final.npz   # 적응 후 거리
│       ├── svdd_model.npz        # 최종 모델 (Support Vectors, alpha)
│       ├── svdd_model.json       # 모델 메타 (kernel, C, R², n_iter)
│       └── metrics.json
└── evaluation/
    ├── AD_performance_all.csv        # 소스 도메인 AD 성능
    └── otta_performance_all.csv      # 전체 시나리오 × OTTA 지표
log/
└── run_<timestamp>.log
```

`otta_performance_all.csv` 주요 컬럼: `AUC`, `F1`, `Recall`, `precision`, `R_pretrain`, `R_final`, `R_growth_pct`, `detection_delay`, `latency_mean_ms`

## 모듈 구조

```
models/
  svdd.py           — Batch SMO 기반 SVDD (base class)
  online_svdd.py    — OnlineSVDD: 고정 SV 버퍼 + partial_fit() 스트리밍
  dual_boundary.py  — DualBoundarySVDD: 3-zone 고정 경계 기반 OTTA (핵심 기여)
  kernels.py        — RBFKernel, LinearKernel, PolyKernel

funs/
  pipeline.py       — run_otta(): pre-train → 스트리밍 루프 → 평가 메트릭 반환
  databuilder.py    — split_dataframe(), build_source_xy(), build_target_stream()
  features.py       — make_feature_fns(), extract_target_stream_features()
  preprocessing.py  — cepstrum(): real cepstrum (IFFT(log|FFT(x)|))
  train.py          — make_kernel()
  evaluation.py     — AnomalyDetectionEvaluator, build_rpm_domain_map()
  download.py       — load_cwru(), load_paderborn()
  config.py         — get_config()
  utils.py          — parse_args(), median_heuristic_gamma()
```
