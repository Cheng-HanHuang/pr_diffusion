#!/usr/bin/env python3
"""GT-free best-of-K protocol for diffusion-prior Fourier phase retrieval (Track R, protocol v1).

Deployable rule (uses only the measurement y, sigma, the solver, a per-solver threshold tau,
and a zero-shot orientation scorer; never the ground truth):

  1. Draw runs x_1, x_2, ... of the solver with fresh seeds; after each run compute its
     measurement residual rho_j = ||y - A(x_j)|| / sqrt(m).
     STOP as soon as M runs have rho_j <= tau ("measurement-consistent"), or after N_MAX runs.
  2. Pick x_hat = argmin_j rho_j over the runs drawn.
  3. FLAG_C ("no consistent reconstruction") if no drawn run has rho_j <= tau.
  4. Orientation: score the pick and its 180-degree rotation with the zero-shot scorer and output
     the higher-scoring member. FLAG_O ("orientation uncertain") if |margin| is below the
     Q_FLAG quantile of |margin| over the outputs of the panel (label-free).
  5. Return (output, FLAG_C, FLAG_O).

Offline evaluation on existing multi-run panels: runs of an image are exchangeable draws
(prereg A9.2(a)), so random orders of an image's n available runs simulate sequential sampling
with N_MAX <= n. Labels (G/O/C/I) and the oracle are used ONLY to score the policy.

Baselines are exact all-subset U-statistics over the same runs: fixed best-of-K with the residual
pick (E0), with the residual pick plus orientation test (E1), and the GT oracles O_raw / O_orbit.

Run: python3 -I gtfree_protocol.py --strings S.csv [--clip CLIP.csv] (--tau T | --tau-rule self_mad3)
                                   [--m 2] [--nmax 16] [--R 4000] [--seed 20261029] [--qflag 0.30]
                                   [--out result.json]
"""
import argparse, csv, json, math, hashlib, sys
import numpy as np

GOOD = {"G"}
ORBIT = {"G", "O", "O_g", "P", "O_p"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_labels(s):
    s = s.strip()
    return s.split(";") if ";" in s else list(s)


def load_strings(path):
    rows = list(csv.DictReader(open(path, newline="")))
    out = []
    for r in rows:
        lab = parse_labels(r.get("labels_by_run") or r.get("run_labels"))
        rho = [float(x) for x in r["rho_signed_by_run"].split(";")]
        assert len(lab) == len(rho), r.get("image_id")
        out.append({"image_id": r["image_id"], "cohort": r.get("cohort") or r.get("source") or "all",
                    "lab": np.array(lab), "rho": np.array(rho)})
    return out


def attach_clip(imgs, path, good_col, margin_col):
    rows = list(csv.DictReader(open(path, newline="")))
    d = {(r["image_id"], int(r["run"])): (int(r[good_col]), float(r[margin_col])) for r in rows}
    for im in imgs:
        n = len(im["rho"])
        im["cgood"] = np.array([d[(im["image_id"], j)][0] for j in range(n)], dtype=float)
        im["cmarg"] = np.array([abs(d[(im["image_id"], j)][1]) for j in range(n)], dtype=float)
    return imgs


def self_mad3(imgs, n_pilot=4):
    # NOTE: computed after any --run-range slicing, i.e. on the first n_pilot runs of the evaluated range
    """Label-free threshold: median + 3 * 1.4826 * MAD of per-image minimum residuals over runs 0..n_pilot-1."""
    mins = np.array([im["rho"][:n_pilot].min() for im in imgs])
    med = np.median(mins)
    return float(med + 3 * 1.4826 * np.median(np.abs(mins - med)))


def subset_weights(n, K):
    # probability that the rank-r element (0-based, ascending rho) is the minimum of a uniform K-subset
    return np.array([math.comb(n - r - 1, K - 1) / math.comb(n, K) if n - r >= K else 0.0 for r in range(n)])


def fixed_policies(im, K):
    n = len(im["rho"])
    order = np.lexsort((np.arange(n), im["rho"]))
    w = subset_weights(n, K)
    e0 = float(w @ np.isin(im["lab"][order], list(GOOD)))
    e1 = float(w @ im["cgood"][order]) if "cgood" in im else float("nan")
    ng = int(np.isin(im["lab"], list(GOOD)).sum()); no = int(np.isin(im["lab"], list(ORBIT)).sum())
    pass_at = lambda c: 1.0 - (math.comb(n - c, K) / math.comb(n, K) if n - c >= K else 0.0)
    return e0, e1, pass_at(ng), pass_at(no)


def run_protocol(im, tau, M, nmax, R, rng):
    n = len(im["rho"]); nmax = min(nmax, n)
    perm = np.argsort(rng.random((R, n)), axis=1)[:, :nmax]
    rho = im["rho"][perm]
    passed = rho <= tau
    cp = np.cumsum(passed, axis=1)
    hit = cp >= M
    stop = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, nmax)          # runs used
    mask = np.arange(nmax)[None, :] < stop[:, None]
    pick_pos = np.argmin(np.where(mask, rho, np.inf), axis=1)
    pick = perm[np.arange(R), pick_pos]
    flag_c = ~(passed & mask).any(axis=1)
    good_e0 = np.isin(im["lab"][pick], list(GOOD)).astype(float)
    res = {"runs": stop.astype(float), "flag_c": flag_c, "good_e0": good_e0, "pick": pick}
    if "cgood" in im:
        res["good_e1"] = im["cgood"][pick]
        res["margin"] = im["cmarg"][pick]
    return res


def interp(curve, k):
    lo = int(math.floor(k)); hi = min(lo + 1, len(curve)); lo = max(lo, 1)
    if hi == lo:
        return curve[lo - 1]
    t = k - lo
    return (1 - t) * curve[lo - 1] + t * curve[hi - 1]


def evaluate(imgs, tau, M, nmax, R, seed, qflag):
    rng = np.random.default_rng(seed)
    has_clip = all("cgood" in im for im in imgs)
    n_panel = min(len(im["rho"]) for im in imgs)
    nmax = min(nmax, n_panel)
    per = []
    for im in imgs:
        r = run_protocol(im, tau, M, nmax, R, rng)
        per.append(r)
    # label-free orientation flag cutoff: Q_FLAG quantile of |margin| over all outputs not flagged C
    c_cut = None
    if has_clip:
        allm = np.concatenate([p["margin"][~p["flag_c"]] for p in per])
        c_cut = float(np.quantile(allm, qflag)) if len(allm) else 0.0
    never_good_panel = np.array([not np.isin(im["lab"], list(GOOD)).any() for im in imgs])
    never_orbit_panel = np.array([not np.isin(im["lab"], list(ORBIT)).any() for im in imgs])
    unsolved = never_orbit_panel if has_clip else never_good_panel
    succ_key = "good_e1" if has_clip else "good_e0"
    S = np.array([p[succ_key].mean() for p in per])
    U = np.array([p["runs"].mean() for p in per])
    FC = np.array([p["flag_c"].mean() for p in per])
    out = {"tau": tau, "M": M, "N_MAX": nmax, "R": R, "seed": seed, "orientation_step": has_clip,
           "n_images": len(imgs), "n_runs_per_image_panel": n_panel,
           "success_all": float(S.mean()), "success_all_e0_pick": float(np.mean([p["good_e0"].mean() for p in per])),
           "runs_mean": float(U.mean()), "runs_fraction_of_nmax": float(U.mean() / nmax),
           "flag_c_rate": float(FC.mean()),
           "flag_c_expected_count": float(FC.sum()),
           "flag_c_on_never_good_panel": float(FC[never_good_panel].sum()),
           "flag_c_on_never_orbit_panel": float(FC[never_orbit_panel].sum()),
           # "unsolved" = no run in the panel that the policy could turn into a GOOD output:
           # no GOOD run (E0 panels) or no orbit-GOOD run (panels with the orientation step)
           "flag_c_precision_unsolved": (float(FC[unsolved].sum() / FC.sum()) if FC.sum() > 0 else None),
           "flag_c_recall_unsolved": (float(FC[unsolved].mean()) if unsolved.any() else None),
           "n_unsolved_panel": int(unsolved.sum()),
           "n_never_good_panel": int(never_good_panel.sum()), "n_never_orbit_panel": int(never_orbit_panel.sum()),
           "success_given_not_flag_c": float(np.sum([p[succ_key][~p["flag_c"]].sum() for p in per]) /
                                             max(1, np.sum([(~p["flag_c"]).sum() for p in per])))}
    if has_clip:
        tot = ok = ans = 0.0; okf = totf = 0.0
        for p in per:
            keep = (~p["flag_c"]) & (p["margin"] >= c_cut)
            fl = (~p["flag_c"]) & (p["margin"] < c_cut)
            ok += p["good_e1"][keep].sum(); tot += keep.sum()
            okf += p["good_e1"][fl].sum(); totf += fl.sum()
            ans += (~p["flag_c"]).sum()
        out.update({"flag_o_cut_abs_margin": c_cut, "qflag": qflag,
                    "coverage_unflagged": float(tot / (R * len(imgs))),
                    "success_unflagged": float(ok / tot) if tot else None,
                    "success_flag_o_only": float(okf / totf) if totf else None})
    # exact fixed best-of-K baselines on the same runs
    Ks = list(range(1, n_panel + 1))
    fx = np.array([[fixed_policies(im, K) for im in imgs] for K in Ks])  # K x img x 4
    curves = {name: fx[:, :, i].mean(axis=1).tolist() for i, name in enumerate(["E0", "E1", "O_raw", "O_orbit"])}
    out["fixed_curves"] = {k: [round(v, 6) for v in c] for k, c in curves.items() if not all(math.isnan(x) for x in c)}
    base = "E1" if has_clip else "E0"
    out["baseline"] = base
    out["fixed_at_equal_mean_runs"] = float(interp(curves[base], U.mean()))
    out["fixed_at_nmax"] = float(curves[base][nmax - 1])
    out["oracle_orbit_at_nmax"] = float(curves["O_orbit"][nmax - 1])
    out["oracle_raw_at_nmax"] = float(curves["O_raw"][nmax - 1])
    out["gain_vs_fixed_equal_runs"] = out["success_all"] - out["fixed_at_equal_mean_runs"]
    out["gap_vs_fixed_nmax"] = out["success_all"] - out["fixed_at_nmax"]
    # per-cohort
    coh = sorted(set(im["cohort"] for im in imgs))
    if len(coh) > 1:
        out["by_cohort"] = {}
        for c in coh:
            m_ = np.array([im["cohort"] == c for im in imgs])
            out["by_cohort"][c] = {"n": int(m_.sum()), "success_all": float(S[m_].mean()), "runs_mean": float(U[m_].mean()),
                                   "flag_c_count": float(FC[m_].sum()),
                                   "fixed_at_nmax": float(fx[nmax - 1, m_, 0 if base == "E0" else 1].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strings", required=True)
    ap.add_argument("--clip")
    ap.add_argument("--clip-good-col", default="g1_chosen_good")
    ap.add_argument("--clip-margin-col", default="g1_margin")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tau", type=float)
    g.add_argument("--tau-rule", choices=["self_mad3"])
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=16)
    ap.add_argument("--R", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--qflag", type=float, default=0.30)
    ap.add_argument("--cohort", default=None, help="restrict to one cohort value")
    ap.add_argument("--run-range", default=None, help="lo:hi, keep runs lo..hi-1 only (e.g. 4:16 = fresh D1 phase-2 runs)")
    ap.add_argument("--frontier", action="store_true", help="also evaluate M in 1..4 x tau multipliers 0.997/1/1.003")
    ap.add_argument("--out")
    a = ap.parse_args()
    imgs = load_strings(a.strings)
    if a.cohort:
        imgs = [im for im in imgs if im["cohort"] == a.cohort]
    if a.clip:
        imgs = attach_clip(imgs, a.clip, a.clip_good_col, a.clip_margin_col)
    if a.run_range:
        lo, hi = (int(x) for x in a.run_range.split(":"))
        for im in imgs:
            for k in ("lab", "rho", "cgood", "cmarg"):
                if k in im:
                    im[k] = im[k][lo:hi]
    tau = a.tau if a.tau is not None else self_mad3(imgs)
    res = evaluate(imgs, tau, a.m, a.nmax, a.R, a.seed, a.qflag)
    res["tau_rule"] = "frozen" if a.tau is not None else a.tau_rule
    res["run_range"] = a.run_range
    res["inputs"] = {"strings": a.strings, "strings_sha256": sha256(a.strings)}
    if a.clip:
        res["inputs"].update({"clip": a.clip, "clip_sha256": sha256(a.clip), "clip_cols": [a.clip_good_col, a.clip_margin_col]})
    res["protocol_version"] = "v1"
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    if a.frontier:
        pts = []
        for M in (1, 2, 3, 4):
            for mult in (0.997, 1.0, 1.003):
                r = evaluate(imgs, tau * mult, M, a.nmax, a.R, a.seed, a.qflag)
                pts.append({"M": M, "tau_mult": mult, "tau": tau * mult, "success_all": r["success_all"], "runs_mean": r["runs_mean"],
                            "fixed_at_equal_mean_runs": r["fixed_at_equal_mean_runs"], "gain": r["gain_vs_fixed_equal_runs"],
                            "flag_c_expected_count": r["flag_c_expected_count"], "flag_c_precision_unsolved": r["flag_c_precision_unsolved"]})
        res["frontier"] = pts
        if a.out:
            open(a.out, "w").write(json.dumps(res, indent=1) + "\n")
        print("frontier: gain>0 at", sum(p["gain"] > 0 for p in pts), "of", len(pts), "points; min gain", round(min(p["gain"] for p in pts), 4),
              "max gain", round(max(p["gain"] for p in pts), 4))
    brief = {k: res[k] for k in ["tau", "M", "N_MAX", "success_all", "runs_mean", "flag_c_expected_count", "flag_c_precision_unsolved",
                                 "n_unsolved_panel", "baseline", "fixed_at_equal_mean_runs", "fixed_at_nmax", "oracle_orbit_at_nmax"]}
    for k in ["coverage_unflagged", "success_unflagged", "success_flag_o_only"]:
        if k in res:
            brief[k] = res[k]
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in brief.items()}))


if __name__ == "__main__":
    main()
