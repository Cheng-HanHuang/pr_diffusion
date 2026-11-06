"""Planner sensitivity checks prompted by two AI reviews (2026-11-05). CPU only.
Reads committed XP returns with `git show`; writes one JSON. Exploratory and post hoc.

(A) GOOD-threshold sensitivity on the panels that carry per-run PSNR:
    SITCOM FFHQ, 16 runs x 180 images (exact subset estimators at K = 4, 16), and
    PC-DAPS FFHQ / ImageNet, one sample of four particles per image.
    E1 for SITCOM needs PSNR(R_111 x) when the face test flips a run whose best orbit member
    is not R_111 (34 of 2,880 runs); E1 is then bracketed by counting those runs as failures (lo)
    or successes (hi).
(B) Label-free threshold (median + 3 * 1.4826 * MAD of per-image minimum residual) against the
    labeled threshold, for pilots of 2, 3 and 4 runs (the first n runs of each image).
(C) Run-order dependence: concordance of the GOOD indicator for adjacent runs minus that for all
    pairs of runs of an image; two-sided permutation p-value, run order shuffled within images.

Usage (repository root): python -I <this file> <out.json>
"""
import collections, csv, io, json, random, statistics, subprocess, sys
from math import comb

def show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], check=True, capture_output=True, text=True).stdout
def rows(rev, path):
    return list(csv.DictReader(io.StringIO(show(rev, path))))

B16R = "origin/codex/prdiff-track-r-xp-a11-b16-return-20261029"
B16 = "docs/track_r/executors/xp/returns/XP_B16_V1_V3_PROSPECTIVE_20261029/"
I2 = ("1eae9bf", "docs/track_r/executors/xp/returns/XP_I2_A15_5_20261103T183150Z/XP_I2_A15_5_PER_PARTICLE.csv")
XPI = ("origin/codex/prdiff-track-r-xpi-chosen-20261101T013250Z",
       "docs/track_r/executors/xp/returns/XP_A19_XPI_CHOSEN_20261101T013250Z/A17_XPI_PER_PARTICLE_CHOSEN_METRICS.csv")
CP3 = ("1981dea", "docs/track_r/executors/xp/theory_inputs/XP_A17_CP3_20261031T173255Z/{}_STRINGS.csv")
res = {}

# ---------- (A) threshold sensitivity ----------
sit = rows(B16R, B16 + "XP_B16_2880_SCORED_ROOTS.csv")
clip = {(r["image_id"], r["run"]): r["f_choice"] for r in rows(B16R, B16 + "XP_B16_16RUN_FACE_CLIP_DECISIONS.csv")}
by = collections.defaultdict(list)
for r in sit: by[r["image_id"]].append(r)
def e1psnr(r, lo):
    c = clip[(r["image_id"], r["run"])]
    if c == "000": return float(r["psnr_raw"])
    if c == r["s_gt"]: return float(r["psnr_orbit_best"])
    return -1.0 if lo else 99.0
A = {"SITCOM_FFHQ": {}, "PCDAPS_FFHQ": {}, "PCDAPS_ImageNet": {}}
A["SITCOM_FFHQ"]["flipped_with_unknown_member"] = sum(clip[(r["image_id"], r["run"])] not in ("000", r["s_gt"]) for r in sit)
for K in (4, 16):
    for th in (22, 23, 24, 25, 26, 27):
        t = collections.Counter()
        for ps in by.values():
            n = len(ps); ps = sorted(ps, key=lambda r: (float(r["rho_signed"]), int(r["run"])))
            w = [comb(n - 1 - i, K - 1) / comb(n, K) for i in range(n)]
            t["E0"] += sum(wi * (float(r["psnr_raw"]) >= th) for wi, r in zip(w, ps))
            t["E1_lo"] += sum(wi * (e1psnr(r, True) >= th) for wi, r in zip(w, ps))
            t["E1_hi"] += sum(wi * (e1psnr(r, False) >= th) for wi, r in zip(w, ps))
            g = sum(float(r["psnr_raw"]) >= th for r in ps); go = sum(float(r["psnr_orbit_best"]) >= th for r in ps)
            t["O_raw"] += 1 - comb(n - g, K) / comb(n, K); t["O_orbit"] += 1 - comb(n - go, K) / comb(n, K)
            t["first"] += g / n
        A["SITCOM_FFHQ"][f"K{K}_theta{th}"] = {k: round(100 * v / len(by), 2) for k, v in t.items()}
def pc(rws, kp, kc, korb, ths):
    b = collections.defaultdict(list)
    for r in rws: b[r["image_id"]].append(r)
    out = {}
    for th in ths:
        t = collections.Counter()
        for ps in b.values():
            ps = sorted(ps, key=lambda r: int(r["particle"]))
            k = min(range(len(ps)), key=lambda i: (float(ps[i]["rho"]), i))
            t["random_particle"] += sum(float(r[kp]) >= th for r in ps) / len(ps)
            t["E0"] += float(ps[k][kp]) >= th; t["E1"] += float(ps[k][kc]) >= th
            t["O_raw"] += any(float(r[kp]) >= th for r in ps)
            if korb: t["O_orbit"] += any(float(r[korb]) >= th for r in ps)
        out[f"theta{th}"] = {k: round(100 * v / len(b), 2) for k, v in t.items()}
    return out
A["PCDAPS_FFHQ"] = pc(rows(*XPI), "PSNR", "PSNR_chosen", None, (22, 23, 24, 25, 26, 27, 28))
A["PCDAPS_ImageNet"] = pc(rows(*I2), "PSNR_raw", "PSNR_chosen", "PSNR_orbit_best", (17, 18, 19, 20, 21, 22, 23))
res["A_threshold_sensitivity"] = A

# ---------- (B) label-free threshold vs pilot size ----------
def mad3(vals):
    m = statistics.median(vals); return m + 3 * 1.4826 * statistics.median(abs(v - m) for v in vals)
panels = {}
for cohort, tau in (("B22", 0.0448503), ("DEV80", 0.0449706)):   # SITCOM labeled tau (C-G3, runs 0-3)
    imgs = [sorted(p, key=lambda r: int(r["run"])) for p in by.values() if p[0]["cohort"] == cohort]
    panels[f"SITCOM_FFHQ_{cohort}"] = ([[float(r["rho_signed"]) for r in p] for p in imgs], tau)
for g, name in (("G4", "DAPS4K_FFHQ"), ("G5", "DAPS4K_ImageNet"), ("G2_0p02", "FAST_FFHQ_sigma0.02"), ("G2_0p10", "FAST_FFHQ_sigma0.10")):
    rw = rows(CP3[0], CP3[1].format(g))
    for cohort in sorted(set(r["cohort"] for r in rw)):
        sub = [r for r in rw if r["cohort"] == cohort]
        panels[f"{name}_{cohort}"] = ([[float(x) for x in r["rho_signed_by_run"].split(";")] for r in sub], float(sub[0]["tau_signed_good_q95"]))
B = {}
for name, (rhos, tau) in panels.items():
    B[name] = {"tau_labeled": tau, "images": len(rhos)}
    for n in (2, 3, 4):
        t = mad3([min(r[:n]) for r in rhos]); B[name][f"pilot{n}_ratio"] = round(t / tau, 4)
res["B_label_free_tau_vs_pilot"] = B

# ---------- (C) run-order dependence ----------
def conc(seqs):
    adj = allp = na = nb = 0
    for s in seqs:
        n = len(s)
        adj += sum(s[i] == s[i + 1] for i in range(n - 1)); na += n - 1
        allp += sum(s[i] == s[j] for i in range(n) for j in range(i + 1, n)); nb += n * (n - 1) // 2
    return adj / na - allp / nb
rng = random.Random(20261106); C = {}
seqsets = {"SITCOM_FFHQ_16": [[r["label"] == "G" for r in sorted(p, key=lambda r: int(r["run"]))] for p in by.values()]}
for g, name in (("G4", "DAPS4K_FFHQ_8"), ("G5", "DAPS4K_ImageNet_8"), ("G2_0p02", "FAST_FFHQ_s0.02_8"), ("G2_0p10", "FAST_FFHQ_s0.10_8")):
    seqsets[name] = [[x == "G" for x in r["labels_by_run"].split(";")] for r in rows(CP3[0], CP3[1].format(g))]
for name, seqs in seqsets.items():
    obs = conc(seqs); null = []
    for _ in range(2000):
        null.append(conc([rng.sample(s, len(s)) for s in seqs]))
    p = (1 + sum(abs(x) >= abs(obs) for x in null)) / (1 + len(null))
    C[name] = {"images": len(seqs), "adjacent_minus_all_pairs": round(obs, 4), "perm_p_two_sided": round(p, 4)}
res["C_run_order_dependence"] = C
json.dump(res, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(json.dumps({"B": B, "C": C}, indent=0, sort_keys=True))
