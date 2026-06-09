"""실험 설정."""

_CONFIG: dict = {
    "seed": 42,

    "cwru": {
        "domains":       {"A": 1730, "D": 1797},
        "sampling_rate": 12000,
        "window_size":   2048,
        "lifter_n":      64,
        "n_revs":        4,
        "rho_inner":     0.5,
        "rho_outer":     1.5,
    },
    "pu": {
        "domains":       {"A": 900, "B": 1500},
        "sampling_rate": 64000,
        "window_size":   4096,
        "lifter_n":      512,
        "n_revs":        1,
        "rho_inner":     0.5,
        "rho_outer":     1.2,
    },

    "preprocessing_ids": {
        "cepstrum": "p4_cepstrum",
    },

    "svdd_nu":                0.1,
    "svdd_max_iter":          1000,
    "svdd_tol":               0.001,
    "no_constraint_buffer_cap": 5000,
    "warmup_ratio":           0.10,
    "max_inner_iter":         10,
}


def get_config() -> dict:
    return _CONFIG
