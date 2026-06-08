"""실험 설정."""
from box import Box

_CONFIG: dict = {
    "seed": 42,

    "cwru_domain": {"A": 1730, "B": 1750, "C": 1772, "D": 1797},
    "pu_domain":   {"A": 900,  "B": 1500},

    "sampling_rate": {"cwru": 12000, "pu": 64000},

    "window_size": 2048,
    "window_size_override": {"cwru": 2048, "pu": 8192},

    "env_spec_bandpass": {"cwru": [2000, 5900], "pu": None},

    "dataset_overrides": {
        "pu": {
            "rho_inner": 0.5,
            "rho_outer": 1.2,
        },
    },

    "order_spec_params": {
        "samples_per_rev": 64,
        "max_order": 20.0,
        "n_revs": {"cwru": 4, "pu": 1},
    },

    "preprocessing_ids": {
        "cepstrum": "p4_cepstrum",
    },
    "cepstrum_lifter_n": {"cwru": 64, "pu": 4096},
    "svdd_nu":       0.1,
    "svdd_max_iter": 1000,
    "svdd_tol":      0.001,
    "no_constraint_buffer_cap": 5000,
    "rho_inner":      0.5,
    "rho_outer":      1.5,
    "warmup_ratio":   0.10,
    "max_inner_iter": 10,
}


def get_config() -> Box:
    return Box(_CONFIG, box_dots=False)
