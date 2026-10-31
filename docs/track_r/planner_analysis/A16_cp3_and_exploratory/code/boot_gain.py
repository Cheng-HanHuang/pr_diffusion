"""Planner: image-bootstrap predictive interval for the protocol's equal-mean-runs gain on a NEW panel drawn
from the same image population (used to register XP-N predictions before any XP-N data exist).
Uses the frozen protocol functions (imported from the frozen file path given as argv[1])."""
import sys, importlib.util, json, math
import numpy as np
spec = importlib.util.spec_from_file_location("gp", sys.argv[1]); gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)
S, C, tau, nmax, runlo, runhi = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
imgs = gp.load_strings(S)
for im in imgs:
    im["lab"] = im["lab"][runlo:runhi]; im["rho"] = im["rho"][runlo:runhi]
imgs = gp.attach_clip(imgs, C, "g1_chosen_good", "g1_margin")
for im in imgs:  # attach_clip indexes runs 0..n-1 of the sliced arrays; redo with offset
    pass
# re-attach with offset correctly
import csv
d = {(r["image_id"], int(r["run"])): (int(r["g1_chosen_good"]), abs(float(r["g1_margin"]))) for r in csv.DictReader(open(C))}
for im in imgs:
    n = len(im["rho"])
    im["cgood"] = np.array([d[(im["image_id"], runlo + j)][0] for j in range(n)], float)
    im["cmarg"] = np.array([d[(im["image_id"], runlo + j)][1] for j in range(n)], float)
rng = np.random.default_rng(20261029); R = 4000
n = len(imgs[0]["rho"]); Ks = list(range(1, n + 1))
Sx = []; Ux = []; F = []
for im in imgs:
    r = gp.run_protocol(im, tau, 2, nmax, R, rng)
    Sx.append(r["good_e1"].mean()); Ux.append(r["runs"].mean())
    F.append([gp.fixed_policies(im, K)[1] for K in Ks])
Sx = np.array(Sx); Ux = np.array(Ux); F = np.array(F)
def gain(idx):
    curve = F[idx].mean(axis=0); return Sx[idx].mean() - gp.interp(list(curve), Ux[idx].mean())
g0 = gain(np.arange(len(imgs)))
bs = np.random.default_rng(7).integers(0, len(imgs), size=(5000, len(imgs)))
G = np.array([gain(b) for b in bs]); se = G.std(ddof=1)
out = {"panel": S.split("/")[-1], "runs": [runlo, runhi], "nmax": nmax, "tau": tau, "gain": g0, "boot_se": se,
       "boot_95": [float(np.quantile(G, .025)), float(np.quantile(G, .975))],
       "predictive_new_panel_95": [g0 - 1.96 * math.sqrt(2) * se, g0 + 1.96 * math.sqrt(2) * se],
       "P_new_gain_below_minus_0p01_normal": float(0.5 * (1 + math.erf(((-0.01) - g0) / (math.sqrt(2) * se) / math.sqrt(2))))}
print(json.dumps(out, indent=1))
