"""A14 PC-DAPS pinned-YAML scalar normalization (infrastructure-only).

PyYAML SafeLoader interprets the upstream spelling '5e-5' as a string,
although PC-DAPS MCMCSampler.get_lr requires a float. This changes only
its Python type, preserving the published learning-rate value exactly.
"""
from __future__ import annotations

def normalize_mcmc_config(mcmc: dict) -> dict:
    if not isinstance(mcmc, dict):
        raise ValueError("STOP_G3_MCMC_CONFIG_NOT_DICT")
    expected = {
        "num_steps": 100,
        "tau": 0.01,
        "lr_min_ratio": 0.01,
        "mc_algo": "langevin",
        "prior_solver": "gaussian",
    }
    if any(mcmc.get(k) != v for k, v in expected.items()):
        raise ValueError("STOP_G3_MCMC_FROZEN_CONFIG_DRIFT")
    original = mcmc.get("lr")
    if not (original == "5e-5" or
            (type(original) is float and original == 5e-5)):
        raise ValueError("STOP_G3_MCMC_LR_FROZEN_VALUE_DRIFT")
    corrected = dict(mcmc)
    corrected["lr"] = float(original)
    if not (type(corrected["lr"]) is float and corrected["lr"] == 5e-5):
        raise ValueError("STOP_G3_MCMC_LR_FLOAT_CAST")
    return corrected
