"""Planner check of the A22 Table 3 E1 (G1) return (XP df6e8cd).
DAPS-4K: the E0 pick among the first K runs and the G1 decision are recomputed from the frozen G5 strings and
G1 decisions (CP3, XP 1981dea); K = 8 PSNR per image is compared with the XP-I2 DAPS comparator (XP 1eae9bf).
SITCOM: no committed per-run inputs exist (A7 detail is local); only internal consistency is checked.
Usage: python3 -I tab3_check.py tab3_image.csv tab3_agg.csv g5_strings.csv g5_clip.csv xpi2_daps_comparison.csv"""
import csv, sys
from collections import defaultdict
img, agg, g5s, g5c, xcmp = sys.argv[1:6]
R = list(csv.DictReader(open(img))); A = list(csv.DictReader(open(agg)))
S = {r["image_id"]: r for r in csv.DictReader(open(g5s))}
C = {(r["image_id"], int(r["run"])): r for r in csv.DictReader(open(g5c))}
X = {r["image_id"]: float(r["PSNR"]) for r in csv.DictReader(open(xcmp)) if r["policy"] == "DAPS_E1_K8"}
by = defaultdict(list)
for r in R:
    by[(r["panel"], r["family"], int(r["K"]))].append(r)
out = {}
for a in A:
    key = (a["panel"], a["family"], int(a["K"])); rows = by[key]
    assert len(rows) == int(a["n_images"]) == 100
    for m in ("PSNR", "SSIM", "VGG_LPIPS", "GOOD20"):
        mean = sum(float(r[m]) for r in rows) / 100
        assert abs(mean - float(a[m])) < 1e-9, (key, m, mean, a[m])
    assert all(int(r["GOOD20"]) == int(float(r["PSNR"]) >= 20.0) for r in rows)
    chk = dict(aggregate_equals_mean=True, good_equals_psnr_ge_20=True)
    if a["family"] == "DAPS-4K":
        K = key[2]; sel = 0; g1 = 0; good = 0; k8 = 0
        for r in rows:
            s = S[r["image_id"]]; rho = [float(x) for x in s["rho_signed_by_run"].split(";")]
            j = min(range(K), key=lambda j: (rho[j], j))
            sel += int(r["selected_run"]) == j
            c = C[(r["image_id"], j)]
            g1 += c["g1_choice"] == r["g1_choice"]; good += int(c["g1_chosen_good"]) == int(r["GOOD20"])
            if K == 8:
                k8 += abs(X[r["image_id"]] - float(r["PSNR"])) < 1e-6
        chk.update(selected_run_matches=sel, g1_choice_matches=g1, good_matches_frozen_clip=good,
                   k8_psnr_matches_xpi2=(k8 if K == 8 else None))
    out["%s K=%d" % (a["family"], key[2])] = chk
for k, v in out.items():
    print(k, v)
