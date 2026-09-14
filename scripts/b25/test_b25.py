#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "configs/b25/b25_cpu_spec.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("tests require CUDA_VISIBLE_DEVICES=''")
    cfg = json.loads(SPEC.read_text())
    if cfg["resource_envelope"] != {
        "max_cpu_workers": 4,
        "max_ram_gib": 16,
        "max_scientific_wall_hours": 4,
        "cuda_visible_devices_required": "",
        "pretrained_model_inference": False,
    }:
        raise RuntimeError("resource envelope drift")
    if cfg["protected_data"]["confirmation_payload_access"] is not False:
        raise RuntimeError("confirmation policy drift")
    if cfg["protected_data"]["new_ffhq_measurements"] is not False:
        raise RuntimeError("FFHQ measurement policy drift")
    if cfg["protected_data"]["new_ffhq_reconstructions"] is not False:
        raise RuntimeError("FFHQ reconstruction policy drift")

    syn = load("b25_syn_test", REPO / "scripts/b25/run_b25_synthetic.py")
    dev = load("b25_dev_test", REPO / "scripts/b25/run_b25_dev_diagnostics.py")
    ana = load("b25_ana_test", REPO / "scripts/b25/analyze_b25_results.py")

    # Named stream determinism and separation.
    r1 = syn.rng_for(cfg, "TEST_A").normal(size=16)
    r2 = syn.rng_for(cfg, "TEST_A").normal(size=16)
    r3 = syn.rng_for(cfg, "TEST_B").normal(size=16)
    if not np.array_equal(r1, r2) or np.array_equal(r1, r3):
        raise RuntimeError("named RNG stream contract failed")

    # K=1 order statistic is exactly standard normal up to quadrature tolerance.
    m, v = syn.order_stat_reference(1)
    if abs(m) > 2e-6 or abs(v - 1.0) > 2e-5:
        raise RuntimeError(f"K=1 order statistic reference drift: mean={m} var={v}")

    # Exact real-array 180-degree Fourier-magnitude ambiguity.
    amb = syn.templates_for("ambiguity_unequal")
    mags = syn.forward_magnitude(amb)
    ambiguity_rel = np.linalg.norm(mags[0] - mags[1]) / max(np.linalg.norm(mags[0]), 1e-300)
    if ambiguity_rel > cfg["experiment2"]["validation_tolerances"]["ambiguity_measurement_rel"]:
        raise RuntimeError(f"toy ambiguity failed: {ambiguity_rel}")

    # VP bridge variance stays nonnegative on every frozen adjacent pair.
    alphas = cfg["experiment2"]["alpha_schedule_clean_to_noise"]
    bridge_vars = []
    for t in range(len(alphas) - 1, 0, -1):
        s = t - 1
        at, ass = float(alphas[t]), float(alphas[s])
        if ass == 1.0:
            var = 0.0
        else:
            ratio = math.sqrt(at / ass) if at > 0 else 0.0
            var = (1.0 - ass) - ratio * ratio * (1.0 - ass) ** 2 / (1.0 - at)
        bridge_vars.append(var)
        if var < -1e-12:
            raise RuntimeError(f"negative frozen bridge variance {var}")

    # Intermediate template weights normalize and alpha=0 returns the prior.
    templates = syn.templates_for("distinguishable_equal").reshape(3, -1)
    pi = np.array([1/3, 1/3, 1/3], float)
    z = np.zeros((5, templates.shape[1]))
    w0 = syn.template_weights_z(z, templates, pi, 0.0)
    if np.max(np.abs(w0 - pi[None, :])) > 1e-15 or np.max(np.abs(w0.sum(1) - 1.0)) > 1e-15:
        raise RuntimeError("alpha=0 prior weight check failed")

    # Synthetic 3-channel symmetry family under the exact DEV operator.
    inv = dev.synthetic_invariance(cfg["experiment3"]["invariance_tolerances"]["synthetic_relative_l2"])
    if not inv["pass"]:
        raise RuntimeError(f"DEV operator symmetry check failed: {inv}")

    # Analysis-side template construction is locked to the simulator.
    for family in ["distinguishable_equal", "ambiguity_unequal", "near_ambiguity_equal", "near_ambiguity_unequal"]:
        if not np.array_equal(syn.templates_for(family), ana.templates_for(family)):
            raise RuntimeError(f"analysis template drift: {family}")

    # No B25 scientific runner imports project model loaders or names a checkpoint.
    for rel in [
        "scripts/b25/run_b25_synthetic.py",
        "scripts/b25/run_b25_dev_diagnostics.py",
        "scripts/b25/analyze_b25_results.py",
    ]:
        text = (REPO / rel).read_text(encoding="utf-8")
        forbidden = ["load_guided_diffusion_model(", "load_model(", "ffhq_10m.pt", "torch.cuda.set_device"]
        hits = [token for token in forbidden if token in text]
        if hits:
            raise RuntimeError(f"forbidden inference/GPU token in {rel}: {hits}")

    print(json.dumps({
        "status": "PASS",
        "rng_determinism": True,
        "k1_order_stat_mean": m,
        "k1_order_stat_variance": v,
        "toy_ambiguity_relative_l2": ambiguity_rel,
        "bridge_variances": bridge_vars,
        "dev_symmetry_max_relative_l2": inv["max_relative_l2"],
        "gpu_work_performed": False,
        "pretrained_model_inference_performed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
