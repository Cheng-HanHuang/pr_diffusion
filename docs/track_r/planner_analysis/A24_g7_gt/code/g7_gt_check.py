"""Planner check of the G7 ground-truth orientation diagnostics (XP da130b5, 967e73c, 6d2afbf, 9d379d2, 9090a50).
For each test: does it keep the upright ground truth? Same 280 images (180 FFHQ, 100 ImageNet) in every file.
Usage: python3 -I g7_gt_check.py <dir with global2_per.csv all8_per.csv fwd2_per.csv univfd_per.csv clip_gt.csv and *_agg.csv>"""
import csv, os, sys
from collections import Counter
D = sys.argv[1]
rd = lambda f: list(csv.DictReader(open(os.path.join(D, f))))
g2 = rd("global2_per.csv"); a8 = rd("all8_per.csv"); f2 = rd("fwd2_per.csv"); uf = rd("univfd_per.csv"); cl = rd("clip_gt.csv")
key = lambda r: (r["dataset"], r["basename"])
sha = lambda r: r.get("gt_sha256") or r.get("original_source_sha256")
base = {key(r): sha(r) for r in cl}
assert len(base) == 280 and Counter(k[0] for k in base) == Counter(ffhq=180, imagenet=100)
for name, rows in (("global2", g2), ("all8", a8), ("2fwd", f2), ("univfd", uf)):
    m = {key(r): sha(r) for r in rows}
    assert m == base, name                       # same images, same source hashes
    assert len(rows) == 280
for r in g2:                                     # threshold rule and kept flag
    mg = float(r["prior_margin"])
    assert r["prior_choice"] == ("111" if mg > 1e-7 else "000")
    assert int(r["upright_kept"]) == int(r["prior_choice"] == "000")
choice = {
    "CLIP face": {key(r): r["face_orientation"] for r in cl if r["dataset"] == "ffhq"},
    "CLIP G1": {key(r): r["g1_orientation"] for r in cl},
    "CLIP G2": {key(r): r["g2_orientation"] for r in cl},
    "S_mid global2": {key(r): r["prior_choice"] for r in g2},
    "S_mid all8": {key(r): r["choice"] for r in a8},
    "2-forward": {key(r): r["choice"] for r in f2},
    "UnivFD": {key(r): r["choice"] for r in uf},
}
for name, rows in (("S_mid all8", a8), ("2-forward", f2), ("UnivFD", uf)):
    assert all(int(r["upright_kept"]) == int(r["choice"] == "000") for r in rows), name
print("upright kept (FFHQ / ImageNet), and the orbit elements chosen when not upright")
flips = {}
for name, c in choice.items():
    ff = [k for k in c if k[0] == "ffhq"]; im = [k for k in c if k[0] == "imagenet"]
    up_f = sum(c[k] == "000" for k in ff); up_i = sum(c[k] == "000" for k in im)
    flips[name] = {k for k in c if c[k] != "000"}
    print(f"  {name:14s} {up_f}/{len(ff)}  {up_i}/{len(im) if im else 0}  non-upright choices: {dict(Counter(c[k] for k in flips[name]))}")
assert choice["S_mid all8"] == choice["S_mid global2"]
print("S_mid all8 choices identical to S_mid global2 on all 280 images: True")
names = ["CLIP G1", "S_mid global2", "2-forward", "UnivFD"]
print("ImageNet non-upright overlaps:")
for i, a in enumerate(names):
    for b in names[i + 1:]:
        A = {k for k in flips[a] if k[0] == "imagenet"}; B = {k for k in flips[b] if k[0] == "imagenet"}
        print(f"  {a} & {b}: {len(A & B)} (of {len(A)} and {len(B)})")
allfour = set.intersection(*[{k for k in flips[n] if k[0] == "imagenet"} for n in names])
anyone = set.union(*[{k for k in flips[n] if k[0] == "imagenet"} for n in names])
print(f"  flipped by all four: {len(allfour)}; by at least one: {len(anyone)}")
# aggregates agree with per-image counts
for f, name in (("global2_agg.csv", "S_mid global2"), ("all8_agg.csv", "S_mid all8"), ("fwd2_agg.csv", "2-forward"), ("univfd_agg.csv", "UnivFD")):
    for r in rd(f):
        if r["dataset"] == "pooled":
            continue
        kept = int(r.get("upright_kept") or r.get("n_upright_kept"))
        c = choice[name]; assert kept == sum(c[k] == "000" for k in c if k[0] == r["dataset"]), (f, r["dataset"])
print("aggregate files agree with per-image counts: True")
