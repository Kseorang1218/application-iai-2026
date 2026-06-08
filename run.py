"""Main Experiment Runner — DualBoundarySVDD 기반 Online Test-Time Adaptation.

목적: 4 전처리 (raw, env_spec, cepstrum, tds) × 44 시나리오에 대해
source domain에서 DualBoundarySVDD를 pre-train한 뒤, target stream을
1-sample씩 순차 처리하며 selective adaptation을 수행한다.

산출물 디렉토리: `results/{date}/{kernel}/{dataset}/{scenario_id}/{prep_id}/`

per-window z-score: env_spec/cepstrum은 raw window에 먼저 적용.
  tds는 feature 추출 후 z-score 적용.
  SVDD 학습 하이퍼파라미터: median heuristic γ + ν=0.1 → C=1/(ν·N).
  Dual boundary: r_inner = rho_inner · R, r_outer = rho_outer · R (R-ratio).
"""
import contextlib
import copy
import io
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from box import Box
import funs

_root = pathlib.Path(__file__).parent

_DOWNLOAD = {
    "cwru": lambda d: funs.download_cwru(str(d / "cwru"), "12k"),
    "pu":   lambda d: funs.download_paderborn(str(d / "pu")),
    "uos":  lambda d: funs.download_uos(str(d / "uos"), "16k"),
}
_POSTPROCESS = {
    "cwru": lambda df: df[df["label"] != 999],
    "pu":   lambda df: df,
    "uos":  lambda df: df,
}

def run_scenario(
    config,
    df,
    dataset: str,
    fs: float,
    source_rpm: int,
    target_rpm: int,
    source_key: str,
    target_key: str,
    results_root: pathlib.Path,
    kernel_name: str,
    otta_mode: str = "dual_boundary",
) -> list[dict]:
    """한 (source, target) 시나리오에 대해 4 전처리 모두 실행."""
    # 데이터셋별 window_size 오버라이드 (없으면 전역 기본값 사용)
    ws = int(config.get("window_size_override", {}).get(dataset.lower(), config["window_size"]))
    seed = config['seed']
    scenario_id = f"{source_rpm}_to_{target_rpm}"
    scenario_label = f"{dataset} {source_key}({source_rpm}) → {target_key}({target_rpm})"
    scenario_dir = results_root / dataset / scenario_id

    # ----- Source/Target raw split -----
    source_train_raw, source_val_raw, _, target_raw = funs.split_dataframe(
        df, source_domain=source_rpm, target_domain=target_rpm,
    )

    # ----- Windowing: source train + target stream -----
    S_X_train, _ = funs.build_source_xy(
        source_train_raw, window_size=ws, stride=ws,
        shuffle=True, random_state=seed,
    )
    T_X_stream, T_y_stream, _ = funs.build_target_stream(
        target_raw, window_size=ws, stride=ws,
    )

    main_cfg = config['main']
    _ds_cfg = config.get("dataset_overrides", {}).get(dataset.lower(), {})
    preprocessing_ids = _ds_cfg.get("preprocessing_ids", main_cfg['preprocessing_ids'])

    # ----- Windowing: source validation (normal only, boundary calibration용) -----
    source_val_normal = source_val_raw[source_val_raw["is_anomaly"] == 0]
    S_X_val, _ = funs.build_source_xy(
        source_val_normal, window_size=ws, stride=ws,
    )

    # ----- Feature 추출 공통 파라미터 -----
    lifter_n = config.main.cepstrum_lifter_n.get(dataset.lower())
    # 데이터셋별 env_spec bandpass 조회 (config에 없으면 "auto" 기본값)
    _bp_cfg = config.get("env_spec_bandpass", {}).get(dataset.lower(), "auto")
    bandpass = "auto" if _bp_cfg == "auto" else (_bp_cfg if _bp_cfg is not None else None)

    # order_spec: samples_per_rev / max_order 는 전역, n_revs 는 dataset별
    _osp_cfg = dict(config.get("order_spec_params", {}))
    _n_revs_map = dict(_osp_cfg.pop("n_revs", {})) if "n_revs" in _osp_cfg else {}
    order_params = {**_osp_cfg, "n_revs": _n_revs_map.get(dataset.lower())}

    # source/target 도메인은 RPM이 다르므로 별도 feature_fns 생성
    feature_fns_src = funs.make_feature_fns(
        fs, lifter_n=lifter_n, log1p=False, bandpass=bandpass,
        rpm=source_rpm, order_spec_params=order_params,
    )
    feature_fns_tgt = funs.make_feature_fns(
        fs, lifter_n=lifter_n, log1p=False, bandpass=bandpass,
        rpm=target_rpm, order_spec_params=order_params,
    )

    # ----- Source feature 추출 (batch OK; SVDD pre-train) -----
    S_train_feats = {k: fn(S_X_train) for k, fn in feature_fns_src.items()}

    # ----- Source validation feature 추출 (동일 파이프라인 적용) -----
    S_val_feats = {k: fn(S_X_val) for k, fn in feature_fns_src.items()}

    # ----- Target stream feature 추출 (window 1개씩 — OTTA 전제) -----
    T_feats = funs.extract_target_stream_features(T_X_stream, feature_fns_tgt, {})

    # ----- 데이터셋별 하이퍼파라미터 오버라이드 적용 -----
    _ds_overrides = dict(config.get("dataset_overrides", {}).get(dataset.lower(), {}))
    if _ds_overrides:
        otta_config = copy.deepcopy(config.to_dict())
        otta_config["main"].update(_ds_overrides)
        otta_config = Box(otta_config)
        print(f"  [override] {dataset} rho: inner={otta_config.main.rho_inner} outer={otta_config.main.rho_outer}")
    else:
        otta_config = config

    # ----- Per-preprocessing OTTA run -----
    print(f"\n[{scenario_label}] running 4 preprocessings → {scenario_dir}")
    scenario_metrics = []
    for prep_name, prep_id in preprocessing_ids.items():
        T_y = T_y_stream
        T_X = T_feats[prep_name]
        if T_X.shape[0] != T_y.shape[0]:
            raise RuntimeError(
                f"target feature/label length mismatch for {prep_name}: "
                f"{T_X.shape[0]} vs {T_y.shape[0]}"
            )

        X_src     = S_train_feats[prep_name]
        X_src_val = S_val_feats[prep_name]

        prep_dir = scenario_dir / prep_id
        try:
            buffer_cap = main_cfg.get('no_constraint_buffer_cap', None)
            m = funs.run_otta(
                prep_id=prep_id,
                X_src_train=X_src,
                X_src_val=X_src_val,
                X_tgt_stream=T_X,
                y_tgt_stream=T_y,
                scaler=None,
                save_dir=prep_dir,
                kernel_name=kernel_name,
                scenario_label=scenario_label,
                config=otta_config,
                config_section='main',
                otta_mode=otta_mode,
                buffer_cap=buffer_cap,
            )
        except Exception as e:
            print(f"  [FAIL {prep_id}] {type(e).__name__}: {e}")
            m = {
                "preprocessing": prep_id,
                "scenario": scenario_label,
                "error": f"{type(e).__name__}: {e}",
            }
        m["dataset"] = dataset
        m["scenario_id"] = scenario_id
        m["source_rpm"] = source_rpm
        m["target_rpm"] = target_rpm
        scenario_metrics.append(m)
        status = "OK" if "error" not in m else "ERR"
        print(f"  [{status}] {prep_id:12s}  "
              f"R={m.get('R', float('nan')):.3f}  "
              f"R_final={m.get('R_final', float('nan')):.3f}  ")

    return scenario_metrics



def _run_scenario_worker(kwargs: dict) -> tuple[list[dict], str]:
    """ProcessPoolExecutor 워커용 래퍼. stdout을 캡처해 메인 프로세스에서 일괄 출력."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_scenario(**kwargs)
    return result, buf.getvalue()


# ============================================================================
# CLI entry
# ============================================================================

def run_experiment(
    dataset: str,
    kernel: str,
    out_dir: str = "results",
    workers: int = 1,
    source: str | None = None,
    target: str | None = None,
    otta_mode: str = "dual_boundary",
) -> None:
    config = funs.get_config()
    config["kernel"] = kernel

    domain_dict = dict(config[f"{dataset}_domain"])
    fs = config["sampling_rate"][dataset]
    valid_keys = list(domain_dict.keys())

    if source and source not in domain_dict:
        raise ValueError(f"유효하지 않은 소스 도메인: {source}. ({dataset} 허용: {valid_keys})")
    if target and target not in domain_dict:
        raise ValueError(f"유효하지 않은 타겟 도메인: {target}. ({dataset} 허용: {valid_keys})")

    sources = [source] if source else valid_keys
    targets = [target] if target else valid_keys
    pairs = [(s, t) for s in sources for t in targets if s != t]

    if not pairs:
        raise ValueError("실행 가능한 (source, target) 쌍이 없습니다. source != target 이어야 합니다.")

    print(f"데이터셋: {dataset} | 시나리오: {len(pairs)}개 | 커널: {kernel} | OTTA 모드: {otta_mode}")
    for s, t in pairs:
        print(f"  {s}({domain_dict[s]}rpm) → {t}({domain_dict[t]}rpm)")

    dataset_dir = _root / "dataset"
    df = _POSTPROCESS[dataset](_DOWNLOAD[dataset](dataset_dir))

    results_root = pathlib.Path(out_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    job_kwargs = [
        dict(
            config=config, df=df, dataset=dataset, fs=fs,
            source_rpm=domain_dict[s_key], target_rpm=domain_dict[t_key],
            source_key=s_key, target_key=t_key,
            results_root=results_root,
            kernel_name=kernel,
            otta_mode=otta_mode,
        )
        for s_key, t_key in pairs
    ]

    if workers == 1:
        for idx, (kw, (s_key, t_key)) in enumerate(zip(job_kwargs, pairs)):
            print(f"\n{'='*60}")
            print(f"시나리오 {idx+1}/{len(pairs)}: {dataset} {s_key}({kw['source_rpm']}rpm) → {t_key}({kw['target_rpm']}rpm)")
            print(f"{'='*60}")
            run_scenario(**kw)
    else:
        print(f"\n[병렬 실행] workers={workers}, 총 {len(pairs)}개 시나리오")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_scenario_worker, kw): (s, t) for kw, (s, t) in zip(job_kwargs, pairs)}
            done = 0
            for f in as_completed(futs):
                s_key, t_key = futs[f]
                done += 1
                try:
                    _, captured = f.result()
                    print(f"\n{'='*60}")
                    print(f"[{done}/{len(pairs)}] {s_key} → {t_key}")
                    print(f"{'='*60}")
                    print(captured, end="")
                except Exception as e:
                    print(f"\n[{done}/{len(pairs)}] 실패: {s_key} → {t_key} — {type(e).__name__}: {e}")


def main():
    args = funs.parse_args(description="Main Experiment Runner (DualBoundarySVDD OTTA)")
    run_experiment(
        dataset=args.dataset,
        kernel=args.kernel,
        out_dir=getattr(args, "out_dir", "results"),
        workers=getattr(args, "workers", 1),
        source=args.source,
        target=args.target,
        otta_mode=getattr(args, "otta_mode", "dual_boundary"),
    )

if __name__ == "__main__":
    main()
