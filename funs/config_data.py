"""실험 설정."""
from box import Box

_CONFIG: dict = {
    "seed": 42,

    "domain": {"A": 1730, "B": 1750, "C": 1772, "D": 1797},

    "sampling_rate": 12000,

    "window_size": 2048,

    "preprocessing_ids": {
        "cepstrum": "p4_cepstrum",
    },
    "cepstrum_lifter_n": 64,
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
