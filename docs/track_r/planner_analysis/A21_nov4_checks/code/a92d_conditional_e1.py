"""A9.2(d) follow-up (planner, exploratory, post hoc): selection-only check for test 1's E1.

Same idea as A9_2b_FAST_conditional_selection/cond_select.py, extended from E0 to E1.
Given each image's REALIZED 64 classes and its REALIZED face-CLIP outcome per run, each frozen
rank law predicts which class the residual pick falls in; the orientation outcome of the picked
run is the realized one (runs of a class are exchangeable under every law, so a specific run of
class k is picked with probability P(pick in k)/n_k). Hence
    predicted E1(K) = (g1/g) P_G(K) + (o1/o) P_O(K),
with g1 = GOOD runs the test keeps upright, o1 = global orbit errors it fixes.
Only selection is modelled; generation and the orientation test are plugged in as realized.
Frozen parameters only: race (w_O, w_C) from v1 fit_diagnostics; race_tau1 and tail as in A9.2(b).

Usage: python3 -I a92d_conditional_e1.py strings64.csv face.csv v1.json out.json
"""
import csv, json, math, sys
from collections import defaultdict

strp, facep, v1p, outp = sys.argv[1:5]
v1 = json.load(open(v1p))
fd = [d for d in v1["fit_diagnostics"] if d["family"] == "FAST"][0]["rank_weights"]
W_RACE = (fd["w_O"], fd["w_C"])
W_TAU1 = (0.1467, 0.0981)   # A9.2(b) OUTPUT.txt (theory code, tau = 1), rounded as printed there
BASE, EXTREME, TARGET = 1e-6, 1000.0, 5 / 205
EPS = (TARGET - BASE / (1 + BASE)) / (EXTREME / (1 + EXTREME) - BASE / (1 + BASE))
LAWS = ("race", "race_tau1", "S2", "tail")
Ks = (8, 12, 16, 24, 32, 48, 64)

S = list(csv.DictReader(open(strp)))
F = defaultdict(dict)
for r in csv.DictReader(open(facep)):
    F[r["image_id"]][int(r["run"])] = r

def class_pick_probs(g, o, c, i, K, law):
    """P(residual pick is G), P(pick is O) over all K-subsets of the realized panel."""
    n = g + o + c + i
    tot = math.comb(n, K)
    if law == "S2":
        wo, wc = 1.0, 1.0
    elif law == "race_tau1":
        wo, wc = W_TAU1
    else:
        wo, wc = W_RACE
    pg = po = 0.0
    for a in range(0, min(g, K) + 1):
        for b in range(0, min(o, K - a) + 1):
            for cc in range(0, min(c, K - a - b) + 1):
                d = K - a - b - cc
                if d > i or (a == 0 and b == 0):
                    continue
                p = math.comb(g, a) * math.comb(o, b) * math.comb(c, cc) * math.comb(i, d) / tot
                if law == "tail":
                    den = a + W_RACE[0] * b + BASE * cc
                    f = (1 - EPS) ** cc
                    pg += p * f * a / den
                    po += p * f * W_RACE[0] * b / den
                else:
                    den = a + wo * b + wc * cc
                    pg += p * a / den
                    po += p * wo * b / den
    return pg, po

def w(n, K):
    d = math.comb(n, K)
    return [math.comb(n - r - 1, K - 1) / d if n - r >= K else 0.0 for r in range(n)]

res = defaultdict(lambda: defaultdict(list))
for s in S:
    iid = s["image_id"]
    labs = s["labels_by_run"].split(";")
    rho = [float(x) for x in s["rho_signed_by_run"].split(";")]
    f = [F[iid][j]["f_chosen_good"] == "1" for j in range(64)]
    g = sum(l == "G" for l in labs); o = sum(l in ("O_g", "O_p") for l in labs)
    c = sum(l == "C" for l in labs); i = sum(l == "I" for l in labs)
    g1 = sum(f[j] for j in range(64) if labs[j] == "G")
    o1 = sum(f[j] for j in range(64) if labs[j] in ("O_g", "O_p"))
    order = sorted(range(64), key=lambda j: (rho[j], j))
    for K in Ks:
        ww = w(64, K)
        obs = sum(ww[r] * f[j] for r, j in enumerate(order))
        for co in (s["cohort"], "POOLED"):
            res[(co, K)]["obs"].append(obs)
            for law in LAWS:
                pg, po = class_pick_probs(g, o, c, i, K, law)
                pred = (g1 / g * pg if g else 0.0) + (o1 / o * po if o else 0.0)
                res[(co, K)][law].append(pred)

# Registered v1 E1_CLIP race point for the same cell, for the decomposition.
reg = {(p["cohort"], p["K"], p["model_variant"]): p["point"] for p in v1["predictions"]
       if p["family"] == "FAST" and p["policy"] == "E1_CLIP" and p["panel"] == "primary"}
out = dict(status="PLANNER_A92D_CONDITIONAL_E1_EXPLORATORY", weights=dict(race=W_RACE, race_tau1=W_TAU1, tail_eps=EPS),
           rows=[])
for co in ("B22", "DEV80", "POOLED"):
    for K in Ks:
        m = {k: sum(v) / len(v) for k, v in res[(co, K)].items()}
        out["rows"].append(dict(cohort=co, K=K, n_images=len(res[(co, K)]["obs"]), observed_E1=m["obs"],
                                **{f"cond_{l}": m[l] for l in LAWS},
                                registered_race=reg[(co, K, "race")], registered_S2=reg[(co, K, "S2")]))
json.dump(out, open(outp, "w"), indent=1)
print("weights race %.4f %.4f  tau1 %.4f %.4f  eps %.5f" % (*W_RACE, *W_TAU1, EPS))
print("cohort K | obsE1 | cond race  tau1   S2    tail | reg race  reg S2 | race:cond-obs  reg-cond")
for r in out["rows"]:
    print(f'{r["cohort"]:6s} {r["K"]:2d} | {r["observed_E1"]:.4f} | {r["cond_race"]:.4f} {r["cond_race_tau1"]:.4f} '
          f'{r["cond_S2"]:.4f} {r["cond_tail"]:.4f} | {r["registered_race"]:.4f} {r["registered_S2"]:.4f} | '
          f'{100*(r["cond_race"]-r["observed_E1"]):+.2f} {100*(r["registered_race"]-r["cond_race"]):+.2f}')
