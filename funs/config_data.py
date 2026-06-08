"""실험 설정."""
from box import Box

_CONFIG: dict = {
    "seed": 42,

    "cwru_domain": {"A": 1730, "B": 1750, "C": 1772, "D": 1797},
    "pu_domain":   {"A": 900,  "B": 1500},
    "uos_domain":  {"A": 600,  "B": 800, "C": 1000, "D": 1200, "E": 1400, "F": 1600},

    "sampling_rate": {"cwru": 12000, "pu": 64000, "uos": 16000},

    "window_size": 2048,
    "window_size_override": {"cwru": 2048, "pu": 8192, "uos": 2048},

    "env_spec_bandpass": {"cwru": [2000, 5900], "pu": None, "uos": None},

    "dataset_overrides": {
        "pu": {
            "rho_inner": 0.5,
            "rho_outer": 1.2,
            "preprocessing_ids": {
                "env_spec": "p3_envspec",
                "cepstrum": "p4_cepstrum",
                "tds":      "p6_tds",
            },
        },
    },

    "order_spec_params": {
        "samples_per_rev": 64,
        "max_order": 20.0,
        "n_revs": {"cwru": 4, "pu": 1, "uos": 1},
    },

    "main": {
        "preprocessing_ids": {
            "raw":        "p1_raw",
            "env_spec":   "p3_envspec",
            "cepstrum":   "p4_cepstrum",
            "tds":        "p6_tds",
            "order_spec": "p7_orderspec",
        },
        "per_window_norm_keys": ["env_spec", "cepstrum", "tds"],
        "cepstrum_lifter_n": {"cwru": 64, "uos": 2048, "pu": 4096},
        "svdd_nu":       0.1,
        "svdd_max_iter": 1000,
        "svdd_tol":      0.001,
        "buffer_size":             100,
        "no_constraint_buffer_cap": 5000,
        "rho_inner":      0.5,
        "rho_outer":      1.5,
        "warmup_ratio":   0.10,
        "max_inner_iter": 10,
        "r_inner_percentiles": [50],
        "r_outer_percentiles": [95, 99, 99.5],
        "tsne_subsample": 1000,
    },
}


def get_config() -> Box:
    return Box(_CONFIG, box_dots=False)
