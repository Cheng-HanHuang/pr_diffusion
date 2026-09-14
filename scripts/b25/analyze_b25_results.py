#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def readj(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def template_base():
    x = np.zeros((6, 6), float)
    x[1, 1], x[2, 4], x[4, 2], x[3, 3] = 1.0, 0.72, 0.43, 0.21
    return x


def templates_for(name):
    a = template_base()
    reverse = np.flip(a, (0, 1)).copy()
    b = np.zeros((6, 6), float)
    b[1, 1], b[1, 4], b[4, 1], b[3, 4] = 1.0, 0.65, 0.35, 0.22
    c = np.zeros((6, 6), float)
    c[1, 2], c[2, 2], c[3, 2], c[4, 4] = 0.9, 0.55, 0.35, 0.8
    if name == "distinguishable_equal":
        return np.stack([a, b, c])
    if name == "ambiguity_unequal":
        return np.stack([a, reverse, b])
    if name.startswith("near_ambiguity"):
        near = reverse.copy()
        near[1, 4] += 0.055
        near[4, 1] += 0.025
        return np.stack([a, near])
    raise ValueError(name)


def reconstruction_mse(probs, templates, truth_index):
    truth = templates[int(truth_index)]
    losses = np.mean((templates - truth[None, :, :]) ** 2, axis=(1, 2))
    return float(np.dot(np.asarray(probs, float), losses)), losses.tolist()


def exp1_analysis(spec, exp1):
    cfg = spec["experiment1"]
    tol = cfg["engineering_reference_tolerances"]
    order_checks = []
    for row in exp1["rows"]:
        check = row.get("order_stat_check_for_hard_selection")
        if not check:
            continue
        mean_err = abs(float(check["hard_selected_projection_mean"]) - float(check["analytic_mean"]))
        var_err = abs(float(check["hard_selected_projection_variance"]) - float(check["analytic_variance"]))
        hist = float(check["histogram_l1"])
        order_checks.append({
            "energy": row["energy"],
            "K": int(row["K"]),
            "h": float(row["h"]),
            "mean_abs_error": mean_err,
            "variance_abs_error": var_err,
            "histogram_l1": hist,
            "pass": (
                mean_err <= float(tol["linear_order_stat_mean_abs"])
                and var_err <= float(tol["linear_order_stat_variance_abs"])
                and hist <= float(tol["linear_order_stat_histogram_l1"])
            ),
        })
    if not order_checks or not all(x["pass"] for x in order_checks):
        raise RuntimeError(f"Experiment 1 engineering order-statistic reference failed: {order_checks}")

    bands = cfg["prospective_interpretation_bands"]
    fits = []
    for row in exp1["small_step_fits"]:
        slope = row["slope"]
        supportive = None
        if slope is not None and int(row["K"]) > 1:
            if row["method"] == "hard_min":
                lo, hi = bands["hard_min_K_gt_1_small_step_exponent_supportive"]
                supportive = float(lo) <= float(slope) <= float(hi)
            elif row["method"] == "finite_likelihood_weighted":
                lo, hi = bands["finite_likelihood_weighted_K_gt_1_small_step_exponent_supportive"]
                supportive = float(lo) <= float(slope) <= float(hi)
        fits.append({**row, "prospective_band_supportive": supportive})
    return {
        "engineering_order_statistic_checks": order_checks,
        "all_engineering_checks_pass": all(x["pass"] for x in order_checks),
        "small_step_fits": fits,
        "hard_min_supportive_count": sum(
            x["prospective_band_supportive"] is True and x["method"] == "hard_min" for x in fits
        ),
        "weighted_supportive_count": sum(
            x["prospective_band_supportive"] is True and x["method"] == "finite_likelihood_weighted" for x in fits
        ),
    }


def exp2_analysis(spec, exp2):
    cfg = spec["experiment2"]
    tol = cfg["validation_tolerances"]
    validation = []
    for check in exp2["reference_checks"]:
        row = {
            "family": check["family"],
            "normalization_pass": float(check["posterior_sum_abs_error"]) <= float(tol["normalization_abs"]),
            "bayes_enumeration_pass": float(check["bayes_enumeration_max_abs"]) <= float(tol["bayes_enumeration_abs"]),
            "limiting_prior_pass": float(check["large_noise_prior_tv"]) <= float(tol["limiting_prior_tv"]),
        }
        if "ambiguity_measurement_relative_l2" in check:
            row["ambiguity_measurement_pass"] = (
                float(check["ambiguity_measurement_relative_l2"]) <= float(tol["ambiguity_measurement_rel"])
            )
            row["ambiguity_posterior_odds_pass"] = (
                float(check["ambiguity_posterior_odds_rel_error"]) <= float(tol["ambiguity_posterior_ratio_rel"])
            )
        row["pass"] = all(v is True for k, v in row.items() if k.endswith("_pass"))
        validation.append(row)
    if not validation or not all(v["pass"] for v in validation):
        raise RuntimeError(f"Experiment 2 exact-reference validation failed: {validation}")

    families = {f["name"]: f for f in cfg["families"]}
    rows = []
    for result in exp2["rows"]:
        family = result["family"]
        fcfg = families[family]
        templates = templates_for(family)
        emp_mse, losses = reconstruction_mse(
            result["terminal_probabilities"], templates, int(fcfg["truth_index"])
        )
        ref_mse, _ = reconstruction_mse(
            result["reference_posterior"], templates, int(fcfg["truth_index"])
        )
        standardized = []
        for p, r, se in zip(
            result["terminal_probabilities"],
            result["reference_posterior"],
            result["terminal_probability_se"],
        ):
            denom = max(float(se), 1.0 / float(result["n"]))
            standardized.append((float(p) - float(r)) / denom)
        rows.append({
            "family": family,
            "method": result["method"],
            "K": int(result["K"]),
            "n": int(result["n"]),
            "tv_error": float(result["tv_error"]),
            "max_abs_mode_weight_error": float(result["max_abs_mode_weight_error"]),
            "max_abs_standardized_mode_weight_error": float(max(abs(x) for x in standardized)),
            "standardized_mode_weight_errors": standardized,
            "mode_omissions": result["mode_omissions"],
            "empirical_reconstruction_mse_to_truth": emp_mse,
            "reference_posterior_reconstruction_mse_to_truth": ref_mse,
            "reconstruction_mse_excess": emp_mse - ref_mse,
            "template_reconstruction_mse_to_truth": losses,
            "likelihood_evaluations": int(result["likelihood_evaluations"]),
            "cpu_wall_seconds": float(result["cpu_wall_seconds"]),
            "intermediate_errors": result["intermediate_errors"],
        })

    pairwise = []
    for family in families:
        for k in cfg["proposal_counts"]:
            exact = next(r for r in rows if r["family"] == family and r["K"] == k and r["method"] == "weighted_exact_intermediate")
            point = next(r for r in rows if r["family"] == family and r["K"] == k and r["method"] == "weighted_denoised_point")
            hard = next(r for r in rows if r["family"] == family and r["K"] == k and r["method"] == "hard_exact_intermediate")
            pairwise.append({
                "family": family,
                "K": int(k),
                "denoised_point_minus_exact_weighted_tv": point["tv_error"] - exact["tv_error"],
                "hard_minus_exact_weighted_tv": hard["tv_error"] - exact["tv_error"],
                "denoised_point_minus_exact_weighted_reconstruction_mse": (
                    point["empirical_reconstruction_mse_to_truth"] - exact["empirical_reconstruction_mse_to_truth"]
                ),
                "hard_minus_exact_weighted_reconstruction_mse": (
                    hard["empirical_reconstruction_mse_to_truth"] - exact["empirical_reconstruction_mse_to_truth"]
                ),
            })
    return {
        "exact_reference_validation": validation,
        "all_exact_reference_checks_pass": all(v["pass"] for v in validation),
        "rows": rows,
        "pairwise_error_decomposition": pairwise,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--synthetic-dir", type=Path, required=True)
    ap.add_argument("--dev-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    spec = readj(args.spec)
    exp1 = readj(args.synthetic_dir / "EXP1_SELECTION_DISTRIBUTION.json")
    exp2 = readj(args.synthetic_dir / "EXP2_EXACT_PRIOR.json")
    synth_summary = readj(args.synthetic_dir / "SYNTHETIC_SUMMARY.json")
    exp3 = readj(args.dev_dir / "EXP3_SYMMETRY_SUMMARY.json")
    exp4 = readj(args.dev_dir / "EXP4_PREPROCESSING_AUDIT.json")
    dev_resource = readj(args.dev_dir / "DEV_RESOURCE.json")
    allowlist = readj(args.dev_dir / "DEV_ALLOWLIST.json")

    if synth_summary.get("status") != "PASS" or exp3.get("status") != "PASS":
        raise RuntimeError("upstream B25 output not PASS")
    if allowlist.get("intersection_count") != 0 or allowlist.get("confirmation_payloads_accessed") is not False:
        raise RuntimeError("protected-data gate failed")
    if dev_resource.get("gpu_work_performed") is not False or dev_resource.get("pretrained_model_inference_performed") is not False:
        raise RuntimeError("forbidden work flag")

    a1 = exp1_analysis(spec, exp1)
    a2 = exp2_analysis(spec, exp2)
    resource_total = float(synth_summary["wall_seconds"]) + float(dev_resource["wall_seconds"])
    max_rss = float(dev_resource["max_rss_gib"])
    resource_pass = (
        resource_total <= float(spec["resource_envelope"]["max_scientific_wall_hours"]) * 3600.0
        and max_rss <= float(spec["resource_envelope"]["max_ram_gib"])
    )
    if not resource_pass:
        raise RuntimeError(f"resource envelope exceeded wall={resource_total}s maxrss={max_rss}GiB")

    payload = {
        "schema_version": "b25.analysis.v1",
        "status": "PASS",
        "experiment1": a1,
        "experiment2": a2,
        "experiment3": exp3,
        "experiment4": exp4,
        "resource_accounting": {
            "synthetic_wall_seconds": float(synth_summary["wall_seconds"]),
            "dev_wall_seconds": float(dev_resource["wall_seconds"]),
            "aggregate_scientific_wall_seconds": resource_total,
            "max_observed_rss_gib": max_rss,
            "resource_envelope_pass": resource_pass,
            "gpu_work_performed": False,
            "pretrained_model_inference_performed": False,
        },
        "protected_data": {
            "confirmation_payloads_accessed": False,
            "new_ffhq_measurements_generated": False,
            "new_ffhq_reconstructions_generated": False,
        },
    }
    writej(args.output / "B25_ANALYSIS.json", payload)
    print(json.dumps({
        "status": "PASS",
        "aggregate_scientific_wall_seconds": resource_total,
        "max_observed_rss_gib": max_rss,
        "exp1_engineering_checks": len(a1["engineering_order_statistic_checks"]),
        "exp2_rows": len(a2["rows"]),
        "shared_failure_count": exp3["shared_failure_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
