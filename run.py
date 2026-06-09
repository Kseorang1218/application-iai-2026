"""Main Experiment Runner — DualBoundarySVDD 기반 Online Test-Time Adaptation."""
import pathlib

import pandas as pd

import funs
from analysis_otta import plot_R_trace_single

_root = pathlib.Path(__file__).parent

_LOAD = {
    "cwru": lambda d: funs.load_cwru(str(d / "cwru"), "12k"),
    "pu":   lambda d: funs.load_paderborn(str(d / "pu")),
}
_POSTPROCESS = {
    "cwru": lambda df: df[df["label"] != 999],
    "pu":   lambda df: df,
}


def run_scenario(
    config,
    df,
    dataset: str,
    source_rpm: int,
    target_rpm: int,
    source_key: str,
    target_key: str,
    out_dir: str = "results",
) -> dict:
    """한 (source, target) 시나리오에 대해 cepstrum 전처리 + OTTA 실행."""
    ws = config[dataset]["window_size"]
    seed = config['seed']
    scenario_id = f"{source_rpm}_to_{target_rpm}"
    scenario_label = f"{dataset} {source_key}({source_rpm}) → {target_key}({target_rpm})"

    # ----- Source/Target raw split -----
    source_train_raw, _, _, target_raw = funs.split_dataframe(
        df, source_domain=source_rpm, target_domain=target_rpm,
    )

    # ----- Windowing -----
    S_X_train, _ = funs.build_source_xy(
        source_train_raw, window_size=ws, stride=ws,
        shuffle=True, random_state=seed,
    )
    T_X_stream, T_y_stream, _ = funs.build_target_stream(
        target_raw, window_size=ws, stride=ws,
    )

    # ----- Feature 추출 -----
    lifter_n = config[dataset]["lifter_n"]
    feature_fns = funs.make_feature_fns(lifter_n=lifter_n)

    S_train_feats = {k: fn(S_X_train) for k, fn in feature_fns.items()}
    T_feats = funs.extract_target_stream_features(T_X_stream, feature_fns)

    otta_config = {**config, **config[dataset]}

    print(f"\n[{scenario_label}] running ...")
    T_X   = T_feats["cepstrum"]
    X_src = S_train_feats["cepstrum"]

    if T_X.shape[0] != T_y_stream.shape[0]:
        raise RuntimeError(
            f"target feature/label length mismatch: {T_X.shape[0]} vs {T_y_stream.shape[0]}"
        )

    try:
        buffer_cap = otta_config.get('no_constraint_buffer_cap', None)
        model_dir = pathlib.Path(out_dir) / dataset / scenario_id
        m = funs.run_otta(
            X_src_train=X_src,
            X_tgt_stream=T_X,
            y_tgt_stream=T_y_stream,
            kernel_name="linear",
            scenario_label=scenario_label,
            config=otta_config,
            otta_mode="dual_boundary",
            buffer_cap=buffer_cap,
            save_dir=str(model_dir),
        )
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        m = {"scenario": scenario_label, "error": f"{type(e).__name__}: {e}"}

    m["dataset"]    = dataset
    m["scenario_id"] = scenario_id
    m["source"]     = source_key
    m["target"]     = target_key
    m["source_rpm"] = source_rpm
    m["target_rpm"] = target_rpm

    R_trace    = m.pop("_R_trace", None)
    y_true_arr = m.pop("_y_true", None)
    decisions  = m.pop("_decisions", None)
    n_warmup   = m.pop("_n_warmup", 0)

    if R_trace is not None:
        plot_R_trace_single(
            R_trace=R_trace,
            y_true=y_true_arr,
            decisions=decisions,
            R_pretrain=m.get("R_pretrain", 0.0),
            n_warmup=n_warmup,
            dataset=dataset,
            sc_label=f"{source_key}→{target_key}",
            kernel="linear",
            save_path=pathlib.Path(out_dir) / dataset / scenario_id / "R_trace.png",
        )

    status = "OK" if "error" not in m else "ERR"
    print(f"  [{status}]  "
          f"R={m.get('R_pretrain', float('nan')):.3f}  "
          f"R_final={m.get('R_final', float('nan')):.3f}  ")
    return m


# ============================================================================
# CLI entry
# ============================================================================

def run_experiment(
    dataset: str = "cwru",
    out_dir: str = "results",
) -> None:
    config = funs.get_config()
    domain_dict = dict(config[dataset]["domains"])
    valid_keys = list(domain_dict.keys())

    pairs = [(s, t) for s in valid_keys for t in valid_keys if s != t]

    if not pairs:
        raise ValueError("실행 가능한 (source, target) 쌍이 없습니다.")

    print(f"데이터셋: {dataset} | 시나리오: {len(pairs)}개")
    for s, t in pairs:
        print(f"  {s}({domain_dict[s]}rpm) → {t}({domain_dict[t]}rpm)")

    dataset_dir = _root / "dataset"
    df = _POSTPROCESS[dataset](_LOAD[dataset](dataset_dir))

    results_root = pathlib.Path(out_dir)

    all_metrics = []
    for idx, (s_key, t_key) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"시나리오 {idx+1}/{len(pairs)}: {dataset} {s_key}({domain_dict[s_key]}rpm) → {t_key}({domain_dict[t_key]}rpm)")
        print(f"{'='*60}")
        m = run_scenario(
            config=config, df=df, dataset=dataset,
            source_rpm=domain_dict[s_key], target_rpm=domain_dict[t_key],
            source_key=s_key, target_key=t_key,
            out_dir=out_dir,
        )
        all_metrics.append(m)

    # ----- evaluation/otta_performance_all.csv 저장 -----
    eval_dir = results_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "otta_performance_all.csv"

    df_result = pd.DataFrame(all_metrics)
    front = [
        "dataset", "source", "target", "scenario_id",
        "AUC", "F1", "Recall", "precision", "Accuracy",
        "TP", "TN", "FP", "FN",
        "R_pretrain", "R_final", "R_growth_pct",
        "detection_delay", "latency_mean_ms", "latency_p99_ms",
    ]
    existing_front = [c for c in front if c in df_result.columns]
    rest = [c for c in df_result.columns if c not in front]
    df_result = df_result[existing_front + rest]

    # 여러 데이터셋 실행 시 기존 CSV에 append
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_result = pd.concat([df_existing, df_result], ignore_index=True)

    df_result.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n[결과] {len(all_metrics)} 시나리오 → {csv_path}")


def main():
    args = funs.parse_args(description="Main Experiment Runner (DualBoundarySVDD OTTA)")
    run_experiment(
        dataset=args.dataset,
        out_dir=getattr(args, "out_dir", "results"),
    )


if __name__ == "__main__":
    main()
