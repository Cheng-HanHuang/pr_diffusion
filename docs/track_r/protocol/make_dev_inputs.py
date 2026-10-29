#!/usr/bin/env python3
"""Build the historical-prefix development panels (4 runs per image) from the A8b per-image file.
Usage: python3 -I make_dev_inputs.py R0N2_A8B_PER_IMAGE.csv OUTDIR"""
import csv, sys, os
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, newline="")))
for fam, srcs in (("SITCOM", ("S1_b22", "S2_b24_dev80")), ("DAPS", ("S2_b24_dev80",)), ("NP", ("S2_b24_dev80",))):
    sel = [r for r in rows if r["family"] == fam and r["source"] in srcs and int(r["n_runs"]) == 4]
    with open(os.path.join(out, f"dev_{fam}_prefix4.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "cohort", "run_labels", "rho_signed_by_run"])
        w.writeheader()
        for r in sel:
            w.writerow({"image_id": r["image_id"], "cohort": r["source"], "run_labels": r["run_labels"],
                        "rho_signed_by_run": r["rho_signed_by_run"]})
    print(fam, len(sel))
