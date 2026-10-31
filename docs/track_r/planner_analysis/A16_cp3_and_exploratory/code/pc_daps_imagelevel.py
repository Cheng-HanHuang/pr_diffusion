"""Planner reproduction of XP-I (PC-DAPS) GOOD25 shares and image-level decomposition.
Inputs: G3_STRINGS.csv (labels, rho per particle), G3_CLIP.csv (face orientation decision per particle).
GT enters only through the labels (evaluation). Exploratory/descriptive."""
import csv, sys, json, random
from collections import Counter, defaultdict
S, C = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(S)))
clip = defaultdict(dict)
for r in csv.DictReader(open(C)):
    clip[r["image_id"]][int(r["run"])] = r
out = {}
per = []
for r in rows:
    lab = r["labels_by_run"].split(";"); rho = [float(x) for x in r["rho_signed_by_run"].split(";")]
    n = len(lab); i0 = min(range(n), key=lambda j: (rho[j], j))
    cl = clip[r["image_id"]]
    assert len(cl) == n
    e1 = int(cl[i0]["face_chosen_good"]); e0 = int(lab[i0] == "G")
    nat = sum(l == "G" for l in lab) / n
    ng = sum(l == "G" for l in lab); no = sum(l in ("O_g", "O_p", "O", "P") for l in lab)
    # orbit oracle: some particle G or orbit error (GOOD in some orientation)
    oorb = int(ng + no > 0); oraw = int(ng > 0)
    # E1 applied to each particle (zero-shot orientation on every particle): mean
    e1_each = sum(int(cl[j]["face_chosen_good"]) for j in range(n)) / n
    per.append(dict(id=r["image_id"], coh=r["cohort"], lab=lab, ng=ng, no=no, e0=e0, e1=e1, nat=nat, oorb=oorb, oraw=oraw, e1_each=e1_each,
                    pick_lab=lab[i0]))
def mean(xs): return sum(xs) / len(xs)
for coh in ["B22", "DEV80", "ALL"]:
    P = [p for p in per if coh == "ALL" or p["coh"] == coh]
    out[coh] = dict(n=len(P), native=mean([p["nat"] for p in P]), E0=mean([p["e0"] for p in P]), E1=mean([p["e1"] for p in P]),
                    E1_each_particle_mean=mean([p["e1_each"] for p in P]), O_raw4=mean([p["oraw"] for p in P]), O_orbit4=mean([p["oorb"] for p in P]))
# image-level decomposition (ALL)
P = per
dec = Counter()
for p in P:
    if p["ng"] == 4: k = "all4_good"
    elif p["oorb"] == 0: k = "none_good_any_orientation"
    else: k = "mixed"
    dec[k] += 1
    if k == "mixed":
        dec["mixed_E1_good"] += p["e1"]; dec["mixed_E0_good"] += p["e0"]
        dec["mixed_native_expected_good"] += p["nat"]
        dec["mixed_E1_each_expected_good"] += p["e1_each"]
    if k == "all4_good":
        dec["all4_E1_good"] += p["e1"]
# rescue/harm vs a random particle (native): per image delta = e1 - nat
delta = [p["e1"] - p["nat"] for p in P]
out["image_level"] = dict(dec)
out["delta_E1_minus_native"] = dict(pos=sum(d > 0 for d in delta), zero=sum(d == 0 for d in delta), neg=sum(d < 0 for d in delta),
                                     sum_pos=sum(d for d in delta if d > 0), sum_neg=sum(d for d in delta if d < 0))
# E1 minus E0 decomposition: orientation fixes vs breaks
fix = sum(1 for p in P if p["e1"] == 1 and p["e0"] == 0); brk = sum(1 for p in P if p["e1"] == 0 and p["e0"] == 1)
out["E1_vs_E0"] = dict(orientation_fixes=fix, orientation_breaks=brk, picked_label_counts=dict(Counter(p["pick_lab"] for p in P)))
# selection loss vs orientation loss relative to O_orbit (4 particles)
# selection loss: O_orbit - P(picked particle is GOOD in some orientation); orientation loss: P(picked in orbit-good) - E1
pick_orbgood = mean([int(p["pick_lab"] in ("G", "O_g", "O_p", "O", "P")) for p in P])
out["gap_decomposition"] = dict(O_orbit=out["ALL"]["O_orbit4"], picked_orbit_good=pick_orbgood, E1=out["ALL"]["E1"],
                                selection_loss=out["ALL"]["O_orbit4"] - pick_orbgood, orientation_loss=pick_orbgood - out["ALL"]["E1"])
# paired bootstrap of E1 - native GOOD (complete-image resampling), seed 20261031, 20000 reps
rng = random.Random(20261031); n = len(delta); B = []
for _ in range(20000):
    B.append(sum(delta[rng.randrange(n)] for _ in range(n)) / n)
B.sort(); out["boot_E1_minus_native_GOOD"] = dict(mean=sum(delta) / n, lo=B[int(0.025 * 20000)], hi=B[int(0.975 * 20000) - 1])
json.dump(out, sys.stdout, indent=1)
