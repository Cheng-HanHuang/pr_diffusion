#!/usr/bin/env python3
"""Deterministic reformat of the executor's D1 16-run strings (A9 schema, XP 5f6b213,
SHA 48169156...) into the A8-style schema that theory v4 (fit_th1_imagenet_v4.py) reads.
No value is changed: O_g -> O, O_p -> P (one character per run), e0_rank_order = stable
argsort of the signed residuals, counts recomputed and checked against the source columns.
Usage: python3 -I make_v4_pilot_csv.py XP_D1P2_16RUN_STRINGS.csv OUT.csv"""
import csv, sys, hashlib
SRC_SHA = "48169156486de4438e715d97a5a3251a728f52e055f135caf14fbcb02fd33e27"
src, out = sys.argv[1], sys.argv[2]
assert hashlib.sha256(open(src, "rb").read()).hexdigest() == SRC_SHA, "source SHA mismatch"
MAP = {"G": "G", "O_g": "O", "O_p": "P", "C": "C", "I": "I"}
rows = list(csv.DictReader(open(src, newline="")))
assert len(rows) == 100
fields = ["image_id", "source", "family", "n_runs", "n_GOOD", "n_ORBIT_global", "n_ORBIT_partial", "n_CONSISTENT",
          "n_INCONSISTENT", "run_labels", "e0_rank_labels", "e0_rank_order", "rho_signed_by_run", "tau_signed_good_q95"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        lab = [MAP[x] for x in r["labels_by_run"].split(";")]
        rho = [float(x) for x in r["rho_signed_by_run"].split(";")]
        assert len(lab) == 16 and len(rho) == 16
        order = sorted(range(16), key=lambda i: (rho[i], i))
        cnt = [lab.count(c) for c in "GOPCI"]
        assert cnt == [int(r[k]) for k in ("n_G", "n_O_g", "n_O_p", "n_C", "n_I")], r["image_id"]
        w.writerow({"image_id": r["image_id"], "source": "XP_D1_P1P2", "family": "FAST", "n_runs": 16,
                    "n_GOOD": cnt[0], "n_ORBIT_global": cnt[1], "n_ORBIT_partial": cnt[2], "n_CONSISTENT": cnt[3],
                    "n_INCONSISTENT": cnt[4], "run_labels": "".join(lab), "e0_rank_labels": "".join(lab[i] for i in order),
                    "e0_rank_order": ",".join(map(str, order)), "rho_signed_by_run": r["rho_signed_by_run"],
                    "tau_signed_good_q95": r["tau_signed_good_q95"]})
print("wrote", out, hashlib.sha256(open(out, "rb").read()).hexdigest())
