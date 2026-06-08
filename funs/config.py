"""실험 설정."""

_CONFIG: dict = {
    "seed": 42,

    "cwru": {
        "domains":       {"A": 1730, "B": 1750, "C": 1772, "D": 1797},
        "sampling_rate": 12000,
        "window_size":   2048,
        "lifter_n":      64,
        "bandpass":      [2000, 5900],
        "n_revs":        4,
        "rho_inner":     0.5,
        "rho_outer":     1.5,
    },
    "pu": {
        "domains":       {"A": 900, "B": 1500},
        "sampling_rate": 64000,
        "window_size":   4096,
        "lifter_n":      512,
        "bandpass":      None,
        "n_revs":        1,
        "rho_inner":     0.5,
        "rho_outer":     1.2,
    },

    "preprocessing_ids": {
        "cepstrum": "p4_cepstrum",
    },

    "samples_per_rev":        64,
    "max_order":              20.0,
    "svdd_nu":                0.1,
    "svdd_max_iter":          1000,
    "svdd_tol":               0.001,
    "no_constraint_buffer_cap": 5000,
    "warmup_ratio":           0.10,
    "max_inner_iter":         10,
}


def get_config() -> dict:
    return _CONFIG
