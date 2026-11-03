"""XP-I2 (PC-DAPS on the 100 ImageNet images, one bag of 4 particles) planner reproduction, A15.5 form.

(a) Same particles: first particle; PC-DAPS native reporting (mean of the four particles' metrics);
    E0 (smallest residual); E1 (E0 then the frozen generic CLIP test G1); O_raw; O_orbit.
    Recomputed from the per-particle CSV and compared with the executor's policy CSV; paired image
    bootstrap (20,000 reps, seed 20261103) of E1 minus native for PSNR, SSIM, VGG-LPIPS and GOOD20.
(b) DAPS-4K comparators recomputed from the frozen G5 strings and G1 decisions (run 0; fixed E1 at K = 1, 2, 8
    on runs 0..K-1) and compared with the executor's comparison CSV (GOOD20).

Usage: python3 -I xpi2_repro.py <xpi2 dir> <g5_strings.csv> <g5_clip.csv> <out.json>
"""
import csv, json, os, sys
from collections import Counter, defaultdict
import numpy as np

D, g5s, g5c, outp = sys.argv[1:5]
PP = list(csv.DictReader(open(os.path.join(D, "XP_I2_A15_5_PER_PARTICLE.csv"))))
POL = list(csv.DictReader(open(os.path.join(D, "XP_I2_A15_5_POLICIES.csv"))))
CMP = list(csv.DictReader(open(os.path.join(D, "XP_I2_A15_5_DAPS_COMPUTE_COMPARISON.csv"))))
G1 = {(r["image_id"], int(r["run"])): r for r in csv.DictReader(open(os.path.join(D, "XP_I2_A15_5_G1_DECISIONS.csv")))}
by = defaultdict(dict)
for r in PP:
    by[r["image_id"]][int(r["particle"])] = r
assert len(by) == 100 and all(sorted(v) == [0, 1, 2, 3] for v in by.values())
f = lambda r, k: float(r[k])
GOOD = 20.0
ids = sorted(by)
pol = defaultdict(lambda: defaultdict(list))
checks = Counter()
for iid in ids:
    ps = [by[iid][j] for j in range(4)]
    for j, r in enumerate(ps):
        g = G1[(iid, j)]
        assert g["g1_choice"] == r["g1_choice"] and g["label"] == r["R0_label"]
        # chosen metrics equal raw metrics when the test keeps the particle upright
        if r["g1_choice"] == "000":
            assert f(r, "PSNR_chosen") == f(r, "PSNR_raw"); checks["upright_same"] += 1
        assert int(g["g1_chosen_good"]) == int(f(r, "PSNR_chosen") >= GOOD); checks["g1_good_consistent"] += 1
        assert f(r, "PSNR_orbit_best") >= max(f(r, "PSNR_raw"), f(r, "PSNR_chosen")) - 1e-9
    e0 = min(range(4), key=lambda j: (f(ps[j], "rho"), j))
    rows = {
        "FIRST_PARTICLE": dict(PSNR=f(ps[0], "PSNR_raw"), SSIM=f(ps[0], "SSIM_raw"), VGG_LPIPS=f(ps[0], "VGG_LPIPS_raw"),
                               GOOD20=float(f(ps[0], "PSNR_raw") >= GOOD)),
        "PC_NATIVE_MEAN4": {k: float(np.mean([f(p, k + "_raw") for p in ps])) for k in ("PSNR", "SSIM", "VGG_LPIPS")}
                           | dict(GOOD20=float(np.mean([f(p, "PSNR_raw") >= GOOD for p in ps]))),
        "E0": dict(PSNR=f(ps[e0], "PSNR_raw"), SSIM=f(ps[e0], "SSIM_raw"), VGG_LPIPS=f(ps[e0], "VGG_LPIPS_raw"),
                   GOOD20=float(f(ps[e0], "PSNR_raw") >= GOOD)),
        "E1_G1": dict(PSNR=f(ps[e0], "PSNR_chosen"), SSIM=f(ps[e0], "SSIM_chosen"), VGG_LPIPS=f(ps[e0], "VGG_LPIPS_chosen"),
                      GOOD20=float(f(ps[e0], "PSNR_chosen") >= GOOD)),
        "O_RAW": dict(PSNR=max(f(p, "PSNR_raw") for p in ps), GOOD20=float(max(f(p, "PSNR_raw") for p in ps) >= GOOD)),
        "O_ORBIT": dict(PSNR=max(f(p, "PSNR_orbit_best") for p in ps),
                        GOOD20=float(max(f(p, "PSNR_orbit_best") for p in ps) >= GOOD)),
    }
    for k, v in rows.items():
        for m, x in v.items():
            pol[k][m].append(x)
agg = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in pol.items()}
# Compare with the executor's policy CSV (per image and metric).
ex = defaultdict(lambda: defaultdict(list))
for r in sorted(POL, key=lambda r: r["image_id"]):
    for m in ("PSNR", "SSIM", "VGG_LPIPS", "GOOD20"):
        if r[m] not in ("", "NA", "nan"):
            ex[r["policy"]][m].append(float(r[m]))
maxdiff = 0.0
for k in pol:
    for m in pol[k]:
        if m in ex[k] and len(ex[k][m]) == 100:
            maxdiff = max(maxdiff, float(np.max(np.abs(np.array(ex[k][m]) - np.array(pol[k][m])))))
            checks["policy_metric_compared"] += 1
# Paired image bootstrap of E1 - native.
rng = np.random.default_rng(20261103)
B = rng.integers(0, 100, size=(20000, 100))
boot = {}
for m in ("PSNR", "SSIM", "VGG_LPIPS", "GOOD20"):
    d = np.array(pol["E1_G1"][m]) - np.array(pol["PC_NATIVE_MEAN4"][m])
    bs = d[B].mean(axis=1)
    boot[m] = dict(point=float(d.mean()), lo95=float(np.quantile(bs, .025)), hi95=float(np.quantile(bs, .975)))
# Image-level anatomy: bags mixing good and not-good particles (orbit-aware at the chosen orientation).
mix = dict(all_good=0, none_good=0, mixed=0, mixed_E1_good=0, mixed_expected_random=0.0)  # by RAW GOOD20
for iid in ids:
    ps = [by[iid][j] for j in range(4)]
    raw_good = [f(p, "PSNR_raw") >= GOOD for p in ps]
    if all(raw_good):
        mix["all_good"] += 1
    elif not any(raw_good):
        mix["none_good"] += 1
    else:
        e0 = min(range(4), key=lambda j: (f(ps[j], "rho"), j))
        mix["mixed"] += 1; mix["mixed_expected_random"] += float(np.mean(raw_good))
        mix["mixed_E1_good"] += int(f(ps[e0], "PSNR_chosen") >= GOOD)
labs = Counter(r["R0_label"] for r in PP); pcl = Counter(r["PC_pr_outcomes_label"] for r in PP)
# (b) DAPS-4K comparators from the frozen G5 strings and G1 decisions.
S = {r["image_id"]: r for r in csv.DictReader(open(g5s))}
GC = {(r["image_id"], int(r["run"])): int(r["g1_chosen_good"]) for r in csv.DictReader(open(g5c))}
dd = defaultdict(list)
for iid in ids:
    labs8 = S[iid]["labels_by_run"].split(";"); rho = [float(x) for x in S[iid]["rho_signed_by_run"].split(";")]
    dd["DAPS_RUN0"].append(float(labs8[0] == "G"))
    for K in (1, 2, 8):
        j = min(range(K), key=lambda j: (rho[j], j))
        dd[f"DAPS_E1_K{K}"].append(float(GC[(iid, j)]))
exb = defaultdict(list)
for r in sorted(CMP, key=lambda r: r["image_id"]):
    exb[r["policy"]].append(float(r["GOOD20"]))
bcheck = {k: dict(planner=float(np.mean(v)), executor=float(np.mean(exb[k])),
                  per_image_identical=bool(np.array_equal(np.array(v), np.array(exb[k])))) for k, v in dd.items()}
out = dict(status="PLANNER_XPI2_REPRODUCTION", checks=dict(checks), policy_max_abs_diff_vs_executor=maxdiff,
           aggregates=agg, bootstrap_E1_minus_native=boot, bag_anatomy=mix, R0_labels=dict(labs),
           PC_labels=dict(pcl), daps_comparators_GOOD20=bcheck)
json.dump(out, open(outp, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("checks", "policy_max_abs_diff_vs_executor", "bag_anatomy", "R0_labels", "PC_labels")}, indent=0))
for k, v in agg.items():
    print(k, {m: round(x, 4) for m, x in v.items()})
for m, b in boot.items():
    print("E1 - native", m, round(b["point"], 4), [round(b["lo95"], 4), round(b["hi95"], 4)])
for k, v in bcheck.items():
    print(k, v)
