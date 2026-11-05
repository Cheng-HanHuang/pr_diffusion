"""Planner checks for the 2026-11-05 writing pass (CPU only; reads committed returns).

1. PC-DAPS ImageNet (XP-I2) oracle-gap decomposition: O_orbit - E1 = selection + orientation.
   Selection: images with an orbit-GOOD particle whose E0 pick is not orbit-GOOD.
   Orientation: images whose E0 pick is orbit-GOOD but whose E1 output is not GOOD.
2. The threshold rule of the two DAPS-4K protocol panels (CP3 G4, G5).

Usage (from the repository root):
  python -I docs/track_r/planner_analysis/A25_writing_pass/code/checks_20261105.py <out.json>
"""
import collections, csv, io, json, subprocess, sys

def show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], check=True, capture_output=True, text=True).stdout

I2 = "docs/track_r/executors/xp/returns/XP_I2_A15_5_20261103T183150Z/XP_I2_A15_5_PER_PARTICLE.csv"
rows = list(csv.DictReader(io.StringIO(show("1eae9bf", I2))))
by = collections.defaultdict(list)
for r in rows:
    by[r["image_id"]].append(r)
oo = e1 = sel = ori = 0
for ps in by.values():
    ps.sort(key=lambda r: int(r["particle"]))
    orb_any = any(float(r["PSNR_orbit_best"]) >= 20 for r in ps)
    k = min(range(len(ps)), key=lambda i: (float(ps[i]["rho"]), i))  # E0: argmin rho, ties to the lower index
    pick_orb = float(ps[k]["PSNR_orbit_best"]) >= 20
    out_good = float(ps[k]["PSNR_chosen"]) >= 20                      # E1 output after the G1 decision
    oo += orb_any; e1 += out_good
    sel += orb_any and not pick_orb
    ori += pick_orb and not out_good
res = {"pcdaps_imagenet": {"images": len(by), "O_orbit": oo, "E1": e1, "gap": oo - e1,
                           "selection": sel, "orientation": ori,
                           "labels": dict(collections.Counter(r["R0_label"] for r in rows))}}
CP3 = "docs/track_r/executors/xp/returns/XP_A17_CP3_20261031T173255Z/CP3_{}_PROTOCOL.json"
for g in ("G4", "G5"):
    d = json.loads(show("1981dea", CP3.format(g)))
    res[g] = {"threshold_rule": d.get("threshold_rule"), "tau": d.get("weighted", {}).get("tau")}
json.dump(res, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(json.dumps(res, sort_keys=True))
