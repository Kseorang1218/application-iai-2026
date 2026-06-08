"""공통 유틸리티 — CLI 인자 파싱, RBF gamma 휴리스틱."""
import argparse

import numpy as np

ALLOWED_KERNELS = ["rbf", "linear", "poly"]
ALLOWED_DATASETS = ["cwru"]


def parse_args(description: str = "Experiment Runner") -> argparse.Namespace:
    """
    CLI 인자 파싱 함수

    Parameters
    ----------
    description : str
        argparse parser description

    Returns
    -------
    argparse.Namespace
        파싱된 인자 (dataset, kernel, source, target)

    Notes
    -----
    source / target 도메인 키의 유효성 검증은 config 로드 후 호출측에서 수행.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=ALLOWED_DATASETS,
        help="사용할 데이터셋 (cwru)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=ALLOWED_KERNELS,
        help="SVDD kernel type (rbf, linear, poly)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        metavar="DOMAIN_KEY",
        help="소스 도메인 키 (예: A, B). 미지정 시 해당 데이터셋의 모든 도메인 사용",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        metavar="DOMAIN_KEY",
        help="타겟 도메인 키 (예: A, B). 미지정 시 해당 데이터셋의 모든 도메인 사용",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="결과물을 저장할 디렉토리 이름 (기본값: results)",
    )
    parser.add_argument(
        "--otta-mode",
        type=str,
        default="dual_boundary",
        choices=["dual_boundary", "single_boundary"],
        help=(
            "OTTA 모드 선택 (기본값: dual_boundary).\n"
            "  dual_boundary   — DualBoundarySVDD: inner skip / adaptation / outer anomaly 3-zone 분기\n"
            "  single_boundary — OnlineSVDD: 초기 warmup 후 예측 기반 TTA (정상→adapt, 고장→skip)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="병렬로 실행할 시나리오 수 (기본값: 1 = 순차). RPi 배포 시 반드시 1 사용.",
    )
    args = parser.parse_args()

    if args.kernel not in ALLOWED_KERNELS:
        raise ValueError(
            f"허용되지 않은 커널 타입입니다: {args.kernel}. (허용: {', '.join(ALLOWED_KERNELS)})"
        )

    return args


def median_heuristic_gamma(
    X: np.ndarray,
    n_subsample: int = 2000,
    random_state: int = 42,
) -> float:
    """RBF gamma = 1 / median(||x_i - x_j||²)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] < 2:
        return 1.0
    rng = np.random.default_rng(random_state)
    if X.shape[0] > n_subsample:
        idx = rng.choice(X.shape[0], n_subsample, replace=False)
        X = X[idx]
    XX = np.sum(X * X, axis=1)
    sq = XX[:, None] + XX[None, :] - 2.0 * (X @ X.T)
    np.maximum(sq, 0.0, out=sq)
    iu = np.triu_indices(X.shape[0], k=1)
    sigma_sq = float(np.median(sq[iu]))
    return 1.0 / sigma_sq if sigma_sq > 0 else 1.0
