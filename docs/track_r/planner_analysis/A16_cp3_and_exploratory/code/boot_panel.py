"""Planner: image-bootstrap (stratified by cohort) CI for the frozen protocol's equal-mean-runs gain on a panel.
Per-cohort protocol evaluation reproduces the frozen code exactly (fresh rng per cohort, file order), then images are
resampled within cohorts; tau is held at its panel value. Exploratory uncertainty statement, not a new test."""
import sys, importlib.util, json, math, csv, argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--code"); ap.add_argument("--strings"); ap.add_argument("--clip"); ap.add_argument("--gcol"); ap.add_argument("--mcol")
ap.add_argument("--taus", help="cohort=tau,... or ALL=tau"); ap.add_argument("--nmax", type=int); ap.add_argument("--runs", default=None)
ap.add_argument("--B", type=int, default=5000); ap.add_argument("--name")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("gp", a.code); gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)
imgs = gp.load_strings(a.strings)
lo, hi = (int(x) for x in a.runs.split(":")) if a.runs else (0, None)
for im in imgs:
    im["lab"] = im["lab"][lo:hi]; im["rho"] = im["rho"][lo:hi]
if a.clip:
    d = {(r["image_id"], int(r["run"])): (int(r[a.gcol]), abs(float(r[a.mcol]))) for r in csv.DictReader(open(a.clip))}
    for im in imgs:
        n = len(im["rho"])
        im["cgood"] = np.array([d[(im["image_id"], lo + j)][0] for j in range(n)], float)
        im["cmarg"] = np.array([d[(im["image_id"], lo + j)][1] for j in range(n)], float)
taus = dict(kv.split("=") for kv in a.taus.split(","))
key = "good_e1" if a.clip else "good_e0"; idx_pol = 1 if a.clip else 0
coh = {}
for c, t in taus.items():
    sub = imgs if c == "ALL" else [im for im in imgs if im["cohort"] == c]
    rng = np.random.default_rng(20261029); n = len(sub[0]["rho"])
    S = []; U = []; F = []
    for im in sub:
        r = gp.run_protocol(im, float(t), 2, a.nmax, 4000, rng)
        S.append(r[key].mean()); U.append(r["runs"].mean()); F.append([gp.fixed_policies(im, K)[idx_pol] for K in range(1, n + 1)])
    coh[c] = (np.array(S), np.array(U), np.array(F))
def cgain(c, idx):
    S, U, F = coh[c]; return S[idx].mean() - gp.interp(list(F[idx].mean(axis=0)), U[idx].mean())
ns = {c: len(v[0]) for c, v in coh.items()}; N = sum(ns.values())
g0 = sum(ns[c] * cgain(c, np.arange(ns[c])) for c in coh) / N
rng = np.random.default_rng(7); G = []
for _ in range(a.B):
    G.append(sum(ns[c] * cgain(c, rng.integers(0, ns[c], ns[c])) for c in coh) / N)
G = np.array(G)
print(json.dumps({"panel": a.name, "nmax": a.nmax, "n_images": N, "gain": round(float(g0), 6), "se": round(float(G.std(ddof=1)), 6),
                  "ci95": [round(float(np.quantile(G, .025)), 6), round(float(np.quantile(G, .975)), 6)]}))
