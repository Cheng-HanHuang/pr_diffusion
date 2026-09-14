#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from pathlib import Path
import numpy as np


def readj(p):
    return json.loads(Path(p).read_text())


def writej(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, sort_keys=True, allow_nan=False) + "\n")


def seed_for(master, name):
    material = f"B25|{name}|{master}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def rng_for(spec, name):
    return np.random.Generator(np.random.PCG64(seed_for(int(spec["rng"]["master_seed"]), name)))


def normcdf_scalar(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_bin_probs(edges, mean=0.0, sd=1.0):
    return np.array([
        normcdf_scalar((edges[i + 1] - mean) / sd) - normcdf_scalar((edges[i] - mean) / sd)
        for i in range(len(edges) - 1)
    ])


def moments(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    var = float(x.var())
    sd = math.sqrt(max(var, 1e-300))
    skew = float(np.mean(((x - mean) / sd) ** 3)) if sd > 0 else 0.0
    kurt = float(np.mean(((x - mean) / sd) ** 4) - 3.0) if sd > 0 else 0.0
    return mean, var, skew, kurt


def summarize_selected(sel_e, h, x, b, direction, hist_edges, target_mean_proj, target_sd_proj):
    n = sel_e.shape[0]
    disp = b * h + math.sqrt(h) * sel_e
    mean = disp.mean(0)
    cov = np.cov(disp, rowvar=False, bias=True)
    proj = disp @ direction
    pm, pv, ps, pk = moments(proj)
    hist, _ = np.histogram(proj, bins=hist_edges, density=False)
    hist_prob = hist.astype(float) / n
    ref_prob = normal_bin_probs(hist_edges, target_mean_proj, target_sd_proj)
    return {
        "n": int(n),
        "mean_displacement": mean.tolist(),
        "mean_displacement_se": np.sqrt(np.diag(cov) / n).tolist(),
        "covariance": cov.tolist(),
        "cov_trace": float(np.trace(cov)),
        "directional_projection": {
            "mean": pm,
            "variance": pv,
            "skewness": ps,
            "excess_kurtosis": pk,
            "quantiles": {str(q): float(np.quantile(proj, q)) for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]},
            "histogram_probs": hist_prob.tolist(),
            "normal_reference_l1": float(np.abs(hist_prob - ref_prob).sum()),
        },
    }


def order_stat_reference(k):
    z = np.linspace(-8.0, 8.0, 200001)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = np.array([normcdf_scalar(float(v)) for v in z])
    density = k * phi * np.power(np.maximum(1.0 - Phi, 0.0), k - 1)
    density /= np.trapezoid(density, z)
    mean = float(np.trapezoid(z * density, z))
    var = float(np.trapezoid((z - mean) ** 2 * density, z))
    return mean, var


def sample_soft_indices(rng, scores):
    logits = -scores
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    uniforms = rng.random((scores.shape[0], 1))
    return (uniforms > np.cumsum(weights, axis=1)).sum(axis=1)


def run_exp1(spec):
    cfg = spec["experiment1"]
    d = int(cfg["dimension"])
    x = np.array(cfg["base_state"], float)
    drift = np.array(cfg["drift"], float)
    g = np.array(cfg["linear_energy_direction"], float)
    g /= np.linalg.norm(g)
    center = np.array(cfg["quadratic_energy_center"], float)
    hist_edges = np.linspace(
        cfg["projection_histogram"]["min"],
        cfg["projection_histogram"]["max"],
        cfg["projection_histogram"]["bins"] + 1,
    )
    order_refs = {
        str(k): {"mean": order_stat_reference(k)[0], "variance": order_stat_reference(k)[1]}
        for k in cfg["proposal_counts"]
    }
    rows = []
    for energy in ["linear", "quadratic"]:
        direction = -g if energy == "linear" else (center - x) / np.linalg.norm(center - x)
        for k in cfg["proposal_counts"]:
            for h in cfg["step_sizes"]:
                rng = rng_for(spec, f"EXP1_{energy.upper()}|K={k}|h={h}")
                selected = {name: [] for name in cfg["methods"]}
                hard_proj_e = []
                remaining = int(cfg["mc_samples_per_cell"])
                while remaining:
                    n = min(remaining, int(cfg["batch_size"]))
                    remaining -= n
                    eps = rng.normal(size=(n, k, d))
                    states = x[None, None, :] + drift[None, None, :] * h + math.sqrt(h) * eps
                    if energy == "linear":
                        scores = np.einsum("nkd,d->nk", states, g)
                    else:
                        scores = 0.5 * np.sum((states - center[None, None, :]) ** 2, axis=2)
                    idx0 = np.zeros(n, dtype=int)
                    idx_hard = np.argmin(scores, axis=1)
                    idx_soft = sample_soft_indices(rng, scores)
                    for name, idx in [
                        ("unselected", idx0),
                        ("hard_min", idx_hard),
                        ("finite_likelihood_weighted", idx_soft),
                    ]:
                        selected[name].append(eps[np.arange(n), idx])
                    if energy == "linear":
                        hard_proj_e.append(np.einsum("nd,d->n", eps[np.arange(n), idx_hard], g))
                proposal_mean = x + drift * h
                if energy == "linear":
                    target_mean = x + drift * h - h * g
                    target_cov = h
                else:
                    target_mean = (proposal_mean + h * center) / (1.0 + h)
                    target_cov = h / (1.0 + h)
                for name in cfg["methods"]:
                    sel = np.concatenate(selected[name], axis=0)
                    ref_mean_proj = float((target_mean - x) @ direction)
                    row = {
                        "energy": energy,
                        "K": k,
                        "h": h,
                        "method": name,
                        "analytic_target_mean_displacement": (target_mean - x).tolist(),
                        "analytic_target_covariance_scalar": target_cov,
                    }
                    row.update(summarize_selected(
                        sel, h, x, drift, direction, hist_edges, ref_mean_proj, math.sqrt(target_cov)
                    ))
                    rows.append(row)
                if energy == "linear":
                    z = np.concatenate(hard_proj_e)
                    bins = np.linspace(-5.0, 5.0, 101)
                    empirical, _ = np.histogram(z, bins)
                    empirical = empirical / empirical.sum()
                    ref = []
                    for i in range(len(bins) - 1):
                        a, bb = bins[i], bins[i + 1]
                        Fa = 1.0 - (1.0 - normcdf_scalar(a)) ** k
                        Fb = 1.0 - (1.0 - normcdf_scalar(bb)) ** k
                        ref.append(Fb - Fa)
                    ref = np.array(ref)
                    ref /= ref.sum()
                    rows[-1]["order_stat_check_for_hard_selection"] = {
                        "hard_selected_projection_mean": float(z.mean()),
                        "hard_selected_projection_variance": float(z.var()),
                        "analytic_mean": order_refs[str(k)]["mean"],
                        "analytic_variance": order_refs[str(k)]["variance"],
                        "histogram_l1": float(np.abs(empirical - ref).sum()),
                    }
    fits = []
    for energy in ["linear", "quadratic"]:
        for k in cfg["proposal_counts"]:
            for method in cfg["methods"]:
                subset = [
                    r for r in rows
                    if r["energy"] == energy and r["K"] == k and r["method"] == method
                    and r["h"] in cfg["small_step_fit_points"]
                ]
                subset.sort(key=lambda r: r["h"])
                hs = np.array([r["h"] for r in subset])
                ys = np.array([abs(r["directional_projection"]["mean"]) for r in subset])
                mask = ys > 1e-14
                slope = float(np.polyfit(np.log(hs[mask]), np.log(ys[mask]), 1)[0]) if mask.sum() >= 2 else None
                fits.append({
                    "energy": energy,
                    "K": k,
                    "method": method,
                    "slope": slope,
                    "points": [[float(a), float(b)] for a, b in zip(hs, ys)],
                })
    return {"schema_version": "b25.exp1.v1", "order_stat_references": order_refs, "rows": rows, "small_step_fits": fits}


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


def forward_magnitude(x):
    if x.ndim == 3:
        return np.abs(np.fft.fft2(x, norm="ortho")).reshape(x.shape[0], -1)
    return np.abs(np.fft.fft2(x, norm="ortho")).reshape(-1)


def loglik_y_templates(y, measurements, sigma):
    return -0.5 * np.sum((measurements - y[None, :]) ** 2, axis=1) / (sigma * sigma)


def softmax_rows(logw):
    shifted = logw - logw.max(axis=1, keepdims=True)
    w = np.exp(shifted)
    return w / w.sum(axis=1, keepdims=True)


def template_weights_z(z, templates_flat, pi, alpha):
    n = z.shape[0]
    if alpha == 0:
        return np.broadcast_to(pi, (n, len(pi))).copy()
    if alpha == 1.0:
        distances = np.sum((z[:, None, :] - templates_flat[None, :, :]) ** 2, axis=2)
        idx = np.argmin(distances, axis=1)
        w = np.zeros((n, len(pi)), dtype=np.float64)
        w[np.arange(n), idx] = 1.0
        return w
    var = 1.0 - alpha
    means = math.sqrt(alpha) * templates_flat
    logp = np.log(pi)[None, :] - 0.5 * np.sum((z[:, None, :] - means[None, :, :]) ** 2, axis=2) / var
    return softmax_rows(logp)


def sample_categorical_rows(rng, w, k=1):
    uniforms = rng.random((w.shape[0], k))
    cumulative = np.cumsum(w, axis=1)
    return (uniforms[:, :, None] > cumulative[:, None, :]).sum(axis=2)


def exact_intermediate_likelihood(z, templates_flat, pi, alpha, likelihood_relative):
    w = template_weights_z(z, templates_flat, pi, alpha)
    return w @ likelihood_relative, w


def denoised_point_likelihood(z, templates, pi, alpha, y, sigma):
    flat_templates = templates.reshape(len(templates), -1)
    w = template_weights_z(z, flat_templates, pi, alpha)
    xhat = (w @ flat_templates).reshape((-1, 6, 6))
    pred = forward_magnitude(xhat)
    return np.exp(-0.5 * np.sum((pred - y[None, :]) ** 2, axis=1) / (sigma * sigma))


def mixture_moments(templates_flat, posterior, alpha):
    template_mean = posterior @ templates_flat
    mean = math.sqrt(alpha) * template_mean
    centered = templates_flat - template_mean
    cov = (1.0 - alpha) * np.eye(templates_flat.shape[1]) + alpha * (centered.T * posterior) @ centered
    return mean, cov


def simulate_exp2_method(spec, family, templates, pi, y, likelihood_relative, posterior, k, method):
    cfg = spec["experiment2"]
    alphas = cfg["alpha_schedule_clean_to_noise"]
    n_total = int(cfg["trajectories_per_cell"])
    batch_size = int(cfg["batch_size"])
    flat_templates = templates.reshape(len(templates), -1)
    d = flat_templates.shape[1]
    rng = rng_for(spec, f"EXP2|{family}|K={k}|{method}")
    counts = np.zeros(len(pi), dtype=np.int64)
    sums = {i: np.zeros(d) for i in range(len(alphas))}
    seconds = {i: np.zeros((d, d)) for i in range(len(alphas))}
    likelihood_evals = 0
    started = time.perf_counter()
    done = 0
    while done < n_total:
        n = min(batch_size, n_total - done)
        done += n
        z = rng.normal(size=(n, d))
        sums[len(alphas) - 1] += z.sum(0)
        seconds[len(alphas) - 1] += z.T @ z
        for t in range(len(alphas) - 1, 0, -1):
            s = t - 1
            alpha_t, alpha_s = float(alphas[t]), float(alphas[s])
            weights_t = template_weights_z(z, flat_templates, pi, alpha_t)
            template_idx = sample_categorical_rows(rng, weights_t, k)
            candidates = np.empty((n, k, d), float)
            for j in range(k):
                chosen_template = flat_templates[template_idx[:, j]]
                if alpha_s == 1.0:
                    candidates[:, j, :] = chosen_template
                else:
                    ratio = math.sqrt(alpha_t / alpha_s) if alpha_t > 0 else 0.0
                    coef = ratio * (1.0 - alpha_s) / (1.0 - alpha_t)
                    mean = math.sqrt(alpha_s) * chosen_template + coef * (z - math.sqrt(alpha_t) * chosen_template)
                    var = (1.0 - alpha_s) - ratio * ratio * (1.0 - alpha_s) ** 2 / (1.0 - alpha_t)
                    if var < -1e-12:
                        raise RuntimeError(f"negative bridge variance {var}")
                    candidates[:, j, :] = mean + math.sqrt(max(var, 0.0)) * rng.normal(size=(n, d))
            flat_candidates = candidates.reshape(n * k, d)
            exact, _ = exact_intermediate_likelihood(
                flat_candidates, flat_templates, pi, alpha_s, likelihood_relative
            )
            exact = exact.reshape(n, k)
            if method == "hard_exact_intermediate":
                choose = np.argmax(exact, axis=1)
            elif method == "weighted_exact_intermediate":
                weights = exact / np.maximum(exact.sum(1, keepdims=True), 1e-300)
                choose = sample_categorical_rows(rng, weights, 1)[:, 0]
            elif method == "weighted_denoised_point":
                approx = denoised_point_likelihood(
                    flat_candidates, templates, pi, alpha_s, y, float(cfg["measurement_noise_sigma"])
                ).reshape(n, k)
                weights = approx / np.maximum(approx.sum(1, keepdims=True), 1e-300)
                choose = sample_categorical_rows(rng, weights, 1)[:, 0]
            else:
                raise ValueError(method)
            z = candidates[np.arange(n), choose]
            likelihood_evals += n * k
            sums[s] += z.sum(0)
            seconds[s] += z.T @ z
        distances = ((z[:, None, :] - flat_templates[None, :, :]) ** 2).sum(2)
        terminal_idx = np.argmin(distances, axis=1)
        counts += np.bincount(terminal_idx, minlength=len(pi))
    probs = counts / n_total
    intermediate = []
    for i, alpha in enumerate(alphas):
        mean = sums[i] / n_total
        cov = seconds[i] / n_total - np.outer(mean, mean)
        ref_mean, ref_cov = mixture_moments(flat_templates, posterior, float(alpha))
        intermediate.append({
            "alpha": float(alpha),
            "mean_l2_error": float(np.linalg.norm(mean - ref_mean)),
            "cov_fro_error": float(np.linalg.norm(cov - ref_cov)),
            "mean_norm": float(np.linalg.norm(mean)),
            "cov_trace": float(np.trace(cov)),
        })
    omissions = [
        int(i) for i, p in enumerate(posterior)
        if p >= cfg["mode_present_reference_threshold"] and probs[i] < cfg["mode_omission_empirical_threshold"]
    ]
    return {
        "method": method,
        "K": k,
        "n": n_total,
        "terminal_probabilities": probs.tolist(),
        "terminal_probability_se": [math.sqrt(max(p * (1.0 - p), 0.0) / n_total) for p in probs],
        "reference_posterior": posterior.tolist(),
        "tv_error": 0.5 * float(np.abs(probs - posterior).sum()),
        "max_abs_mode_weight_error": float(np.max(np.abs(probs - posterior))),
        "mode_omissions": omissions,
        "intermediate_errors": intermediate,
        "likelihood_evaluations": likelihood_evals,
        "cpu_wall_seconds": time.perf_counter() - started,
    }


def run_exp2(spec):
    cfg = spec["experiment2"]
    sigma = float(cfg["measurement_noise_sigma"])
    results, checks = [], []
    for family_cfg in cfg["families"]:
        family = family_cfg["name"]
        templates = templates_for(family)
        pi = np.array(family_cfg["prior_weights"], float)
        pi /= pi.sum()
        measurements = forward_magnitude(templates)
        rng = rng_for(spec, f"EXP2_MEAS|{family}")
        y = measurements[int(family_cfg["truth_index"])] + sigma * rng.normal(size=measurements.shape[1])
        loglik = loglik_y_templates(y, measurements, sigma)
        likelihood_relative = np.exp(loglik - loglik.max())
        posterior = pi * likelihood_relative
        posterior /= posterior.sum()
        brute = np.array([pi[i] * math.exp(float(loglik[i] - loglik.max())) for i in range(len(pi))])
        brute /= brute.sum()
        check = {
            "family": family,
            "posterior_sum_abs_error": abs(float(posterior.sum()) - 1.0),
            "bayes_enumeration_max_abs": float(np.max(np.abs(posterior - brute))),
        }
        huge_sigma = 1e6
        huge_loglik = loglik_y_templates(y, measurements, huge_sigma)
        huge_post = pi * np.exp(huge_loglik - huge_loglik.max())
        huge_post /= huge_post.sum()
        check["large_noise_prior_tv"] = 0.5 * float(np.abs(huge_post - pi).sum())
        if family == "ambiguity_unequal":
            check["ambiguity_measurement_relative_l2"] = float(
                np.linalg.norm(measurements[0] - measurements[1]) / max(np.linalg.norm(measurements[0]), 1e-300)
            )
            check["ambiguity_posterior_odds_rel_error"] = float(
                abs((posterior[0] / posterior[1]) / (pi[0] / pi[1]) - 1.0)
            )
        checks.append(check)
        for k in cfg["proposal_counts"]:
            for method in cfg["methods"]:
                row = simulate_exp2_method(
                    spec, family, templates, pi, y, likelihood_relative, posterior, k, method
                )
                row.update({"family": family, "prior_weights": pi.tolist(), "measurement": y.tolist()})
                results.append(row)
    return {"schema_version": "b25.exp2.v1", "reference_checks": checks, "rows": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("B25 requires CUDA_VISIBLE_DEVICES='' for every scientific subprocess")
    spec = readj(args.spec)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    exp1 = run_exp1(spec)
    writej(args.output / "EXP1_SELECTION_DISTRIBUTION.json", exp1)
    exp2 = run_exp2(spec)
    writej(args.output / "EXP2_EXACT_PRIOR.json", exp2)
    summary = {
        "schema_version": "b25.synthetic-summary.v1",
        "status": "PASS",
        "gpu_work_performed": False,
        "pretrained_model_inference_performed": False,
        "synthetic_measurements_generated": len(spec["experiment2"]["families"]),
        "wall_seconds": time.perf_counter() - started,
        "exp1_rows": len(exp1["rows"]),
        "exp2_rows": len(exp2["rows"]),
    }
    writej(args.output / "SYNTHETIC_SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
