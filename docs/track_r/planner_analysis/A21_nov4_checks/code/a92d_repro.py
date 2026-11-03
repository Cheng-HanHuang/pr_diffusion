"""A9.2(d) planner reproduction: FAST FFHQ E1 (frozen N2 face-CLIP test) exact curves and the
comparison with v1's registered E1_CLIP forecasts. Independent of the executor's code.

E1(K) for one image = sum_r w_r(n,K) * f_chosen_good[run ranked r by (rho_signed, run index)],
w_r(n,K) = C(n-r-1,K-1)/C(n,K) (the exact average over all K-subsets of the argmin pick).
Primary panel: n = 64, K in (8,12,16,24,32,48,64). Secondary: runs 0-15, n = 16, K in (8,12,16).
Coverage: observed inside [lo95, hi95] of the frozen v1 E1_CLIP prediction (cohort, K, variant).

Usage: python3 -I a92d_repro.py v1.json strings64.csv face.csv exec_curves.json exec_cov.json out.json
"""
import csv, json, math, sys
from collections import defaultdict

v1p, strp, facep, ecp, ecovp, outp = sys.argv[1:7]
v1 = json.load(open(v1p))
S = {r["image_id"]: r for r in csv.DictReader(open(strp))}
F = defaultdict(dict)
for r in csv.DictReader(open(facep)):
    F[r["image_id"]][int(r["run"])] = r
assert len(S) == 180 and len(F) == 180 and all(len(v) == 64 for v in F.values())

# Consistency: labels and cohorts in the face CSV equal the frozen 64-run strings.
for iid, s in S.items():
    labs = s["labels_by_run"].split(";")
    assert len(labs) == 64
    for j in range(64):
        assert F[iid][j]["label"] == labs[j], (iid, j)
        assert F[iid][j]["cohort"] == s["cohort"]
        good = F[iid][j]["f_chosen_good"] == "1"
        # f_chosen_good must follow from label and choice: G kept upright, O_g flipped.
        lab, ch = labs[j], F[iid][j]["f_choice"]
        exp = (lab == "G" and ch == "000") or (lab == "O_g" and ch == "111")
        assert good == exp, (iid, j, lab, ch, good)

def w(n, K):
    d = math.comb(n, K)
    return [math.comb(n - r - 1, K - 1) / d if n - r >= K else 0.0 for r in range(n)]

def curves(iid, n, K):
    s = S[iid]
    rho = [float(x) for x in s["rho_signed_by_run"].split(";")][:n]
    labs = s["labels_by_run"].split(";")[:n]
    order = sorted(range(n), key=lambda j: (rho[j], j))
    ww = w(n, K)
    e0 = sum(ww[r] * (labs[j] == "G") for r, j in enumerate(order))
    e1 = sum(ww[r] * (F[iid][j]["f_chosen_good"] == "1") for r, j in enumerate(order))
    return e0, e1

cohorts = {"B22": [i for i in S if S[i]["cohort"] == "B22"],
           "DEV80": [i for i in S if S[i]["cohort"] == "DEV80"]}
cohorts["POOLED"] = cohorts["B22"] + cohorts["DEV80"]
Kgrid = {64: (1, 2, 4, 8, 12, 16, 24, 32, 48, 64), 16: (1, 2, 4, 8, 12, 16)}
obs = {}
for co, ids in cohorts.items():
    for n, Ks in Kgrid.items():
        for K in Ks:
            v = [curves(i, n, K) for i in ids]
            obs[(co, n, K)] = (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v))

# 1) Compare with the executor's exact curves.
ec = json.load(open(ecp))["curves"]
maxdiff = 0.0
for c in ec:
    e0, e1 = obs[(c["cohort"], c["n"], c["K"])]
    maxdiff = max(maxdiff, abs(e0 - c["exact_subset_E0"]), abs(e1 - c["exact_subset_E1_face"]))
assert len(ec) == 48

# 2) Comparison with v1 E1_CLIP forecasts.
panels = {"primary": (64, (8, 12, 16, 24, 32, 48, 64)), "secondary": (16, (8, 12, 16))}
comp = []
for p in v1["predictions"]:
    if p["family"] != "FAST" or p["policy"] != "E1_CLIP":
        continue
    n, Ks = panels[p["panel"]]
    assert p["K"] in Ks
    o = obs[(p["cohort"], n, p["K"])][1]
    half = (p["hi95"] - p["lo95"]) / 3.92
    comp.append(dict(panel=p["panel"], cohort=p["cohort"], K=p["K"], variant=p["model_variant"],
                     point=p["point"], lo95=p["lo95"], hi95=p["hi95"], observed=o,
                     inside=p["lo95"] <= o <= p["hi95"],
                     std_err=abs(o - p["point"]) / max(half, 0.005)))
assert len(comp) == 120
cov = defaultdict(lambda: [0, 0])
for c in comp:
    k = (c["panel"], c["cohort"], c["variant"])
    cov[k][0] += c["inside"]; cov[k][1] += 1

# 3) Compare coverage with the executor's file (counts and each comparison).
ex = json.load(open(ecovp))
mism = 0
exmap = {(c["panel"], c["cohort"], c["K"], c["model_variant"]): c for c in ex["comparisons"]}
for c in comp:
    e = exmap[(c["panel"], c["cohort"], c["K"], c["variant"])]
    if e["inside"] != c["inside"] or abs(e["observed"] - c["observed"]) > 1e-12:
        mism += 1
excov = {(c["panel"], c["cohort"], c["variant"]): (c["covered"], c["total"]) for c in ex["coverage"]}
cov_match = all(tuple(cov[k]) == excov[k] for k in cov)

# Signed direction of the misses (forecast minus observed), primary, per variant.
direction = defaultdict(list)
for c in comp:
    if c["panel"] == "primary":
        direction[c["variant"]].append(c["point"] - c["observed"])

out = dict(
    status="PLANNER_A92D_REPRODUCTION",
    inputs=dict(v1_sha256_prefix="c19798f9", strings64_sha256_prefix="d616dd50",
                face_csv_sha256_prefix="1be37fa9", exec_curves_sha256_prefix="6e470cc7",
                exec_coverage_sha256_prefix="1b55c007"),
    consistency="labels and cohorts of all 11,520 face rows equal the frozen 64-run strings; "
                "f_chosen_good follows from label and choice in every row",
    curves_max_abs_diff_vs_executor=maxdiff,
    comparisons=len(comp), comparison_mismatches_vs_executor=mism,
    coverage_counts_match_executor=cov_match,
    coverage={"|".join(k): f"{a}/{b}" for k, (a, b) in sorted(cov.items())},
    primary_totals={v: f"{sum(cov[('primary', co, v)][0] for co in ('B22', 'DEV80', 'POOLED'))}/21"
                    for v in ("race", "S2", "race_tau1", "tail")},
    primary_forecast_minus_observed={v: [round(min(d), 4), round(max(d), 4)] for v, d in direction.items()},
    observed_E0_E1={f"{co}|n{n}|K{K}": [round(a, 6), round(b, 6)] for (co, n, K), (a, b) in sorted(obs.items())},
    comparisons_detail=comp,
)
json.dump(out, open(outp, "w"), indent=1, sort_keys=False)
print(json.dumps({k: out[k] for k in ("curves_max_abs_diff_vs_executor", "comparison_mismatches_vs_executor",
                                       "coverage_counts_match_executor", "primary_totals",
                                       "primary_forecast_minus_observed")}, indent=1))
for k in sorted(cov):
    print(k, cov[k])
