"""XP-C3 readout (planner; registered as exploratory in A20.6): E1-prior beside E1-CLIP on every complete panel.

E1-x(K) for one image = sum_r w_r(n,K) * x_chosen_good[run ranked r by (rho_signed, run index)],
w_r(n,K) = C(n-r-1,K-1)/C(n,K): the exact average over all K-subsets of the residual pick, then
orientation by x in {CLIP G1 (frozen protocol test), prior S_mid global2 (A20)}.
Also E0, O_raw, O_orbit; the oracle-gap split; candidate-level orientation accuracy; margins;
and a paired image bootstrap of E1-prior minus E1-CLIP (exploratory; not registered).

Usage: python3 -I xpc3_readout.py <in dir> <out.json>
  in dir: {xpn,g5,d4}_{strings,clip,prior}.csv
"""
import csv, json, math, os, sys
from collections import Counter, defaultdict
import numpy as np

D, outp = sys.argv[1:3]
PANELS = {"XP_N": ("xpn", 64, (1, 2, 4, 8, 16, 32, 64)), "G5": ("g5", 8, (1, 2, 4, 8)),
          "XP_D4": ("d4", 64, (1, 2, 4, 8, 16, 32, 64))}
ORB = ("G", "O_g", "O_p")

def w(n, K):
    return np.array([math.comb(n - r - 1, K - 1) / math.comb(n, K) if n - r >= K else 0.0 for r in range(n)])

NORM = {"O": "O_g", "P": "O_p"}  # older label names in the D4 CLIP file

def check_good(lab, ch, g):
    """chosen_good is GT-evaluated at the chosen orientation. Labels fix it except when the other
    orientation of a G run, or the global flip of a partial-orbit run, also passes the threshold."""
    if lab in ("C", "I"):
        return g == 0
    if ch == "000":
        return g == (1 if lab == "G" else 0)
    if lab == "O_g":
        return g == 1
    return g in (0, 1)  # G or O_p flipped: GT-dependent, checked across files below

res = {}
rng = np.random.default_rng(20261103)
for P, (f, n, Ks) in PANELS.items():
    S = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(D, f + "_strings.csv")))}
    C = {(r["image_id"], int(r["run"])): r for r in csv.DictReader(open(os.path.join(D, f + "_clip.csv")))}
    Pr = list(csv.DictReader(open(os.path.join(D, f + "_prior.csv"))))
    assert list(Pr[0].keys()) == ["image_id", "run", "label", "prior_choice", "prior_margin", "prior_chosen_good"]
    PR = {(r["image_id"], int(r["run"])): r for r in Pr}
    assert len(Pr) == len(PR) == 100 * n == len(C) and len(S) == 100
    chk = Counter()
    for iid, s in S.items():
        labs = s["labels_by_run"].split(";"); assert len(labs) == n
        for j in range(n):
            p, c = PR[(iid, j)], C[(iid, j)]
            assert p["label"] == labs[j] == NORM.get(c["label"], c["label"]), (P, iid, j)
            m = float(p["prior_margin"])
            assert p["prior_choice"] == ("111" if m > 1e-7 else "000"), (P, iid, j, m)
            pg, cg = int(p["prior_chosen_good"]), int(c["g1_chosen_good"])
            assert check_good(labs[j], p["prior_choice"], pg) and check_good(labs[j], c["g1_choice"], cg), (P, iid, j)
            if p["prior_choice"] == c["g1_choice"]:
                assert pg == cg, (P, iid, j)   # same orientation, same GT outcome
                chk["same_choice"] += 1
            if (labs[j] == "G" and "111" in (p["prior_choice"], c["g1_choice"]) and
                    (pg if p["prior_choice"] == "111" else cg) == 1):
                chk["G_flipped_still_good"] += 1
            chk["rows"] += 1
    ids = sorted(S)
    per = defaultdict(dict)  # K -> metric -> per-image array
    for K in Ks:
        ww = w(n, K)
        cols = defaultdict(list)
        for iid in ids:
            s = S[iid]; labs = np.array(s["labels_by_run"].split(";"))
            rho = np.array([float(x) for x in s["rho_signed_by_run"].split(";")])
            order = np.lexsort((np.arange(n), rho)); lo = labs[order]
            cg = np.array([int(C[(iid, j)]["g1_chosen_good"]) for j in range(n)])[order]
            pg = np.array([int(PR[(iid, j)]["prior_chosen_good"]) for j in range(n)])[order]
            nG = int((labs == "G").sum()); nO = int(np.isin(labs, ORB).sum())
            pa = lambda c: 1.0 - (math.comb(n - c, K) / math.comb(n, K) if n - c >= K else 0.0)
            cols["E0"].append(ww @ (lo == "G")); cols["E1_CLIP"].append(ww @ cg); cols["E1_prior"].append(ww @ pg)
            cols["O_raw"].append(pa(nG)); cols["O_orbit"].append(pa(nO))
            cols["O_orient_E0"].append(ww @ np.isin(lo, ORB))
        per[K] = {k: np.array(v) for k, v in cols.items()}
    curves = {K: {k: float(v.mean()) for k, v in per[K].items()} for K in Ks}
    for K in Ks:
        c = curves[K]
        c["prior_minus_CLIP"] = c["E1_prior"] - c["E1_CLIP"]
        c["orientation_loss_CLIP"] = c["O_orient_E0"] - c["E1_CLIP"]
        c["orientation_loss_prior"] = c["O_orient_E0"] - c["E1_prior"]
        c["selection_loss"] = c["O_orbit"] - c["O_orient_E0"]
    boot = {}
    for K in sorted({4, 16, n} & set(Ks)):
        d = per[K]["E1_prior"] - per[K]["E1_CLIP"]
        b = np.array([d[rng.integers(0, 100, 100)].mean() for _ in range(10000)])
        boot[K] = dict(point=float(d.mean()), lo95=float(np.quantile(b, .025)), hi95=float(np.quantile(b, .975)),
                       images_prior_better=int((d > 1e-12).sum()), images_CLIP_better=int((d < -1e-12).sum()))
    # candidate-level orientation behaviour
    cand = {}
    for lab in ("G", "O_g", "O_p", "C", "I"):
        rows = [k for k, p in PR.items() if p["label"] == lab]
        if not rows:
            continue
        pf = sum(PR[k]["prior_choice"] == "111" for k in rows); cf = sum(C[k]["g1_choice"] == "111" for k in rows)
        agree = sum(PR[k]["prior_choice"] == C[k]["g1_choice"] for k in rows)
        am = np.array([abs(float(PR[k]["prior_margin"])) for k in rows])
        cand[lab] = dict(n=len(rows), prior_flip=pf, clip_flip=cf, agree=agree,
                         prior_abs_margin_median=float(np.median(am)), prior_near_tie_le_1e5=int((am <= 1e-5).sum()))
    go = [k for k, p in PR.items() if p["label"] in ("G", "O_g")]
    cand["G_and_O_g_correct"] = dict(n=len(go), prior=sum(int(PR[k]["prior_chosen_good"]) for k in go),
                                     clip=sum(int(C[k]["g1_chosen_good"]) for k in go))
    res[P] = dict(n_images=100, runs_per_image=n, rows_checked=chk["rows"], same_choice_rows=chk["same_choice"],
                  G_rows_flipped_but_still_good=chk["G_flipped_still_good"], curves=curves, bootstrap_prior_minus_clip=boot,
                  candidates=cand)
json.dump(res, open(outp, "w"), indent=1)
for P, r in res.items():
    print(f"== {P} (n={r['runs_per_image']}; rows checked {r['rows_checked']})")
    print(" K  | E0     E1_CLIP E1_prior O_raw  O_orbit | prior-CLIP | orientLoss CLIP prior | selLoss")
    for K, c in r["curves"].items():
        print(f"{K:3d} | {c['E0']:.4f} {c['E1_CLIP']:.4f}  {c['E1_prior']:.4f}  {c['O_raw']:.4f} {c['O_orbit']:.4f} | {100*c['prior_minus_CLIP']:+6.2f}     | {100*c['orientation_loss_CLIP']:5.2f} {100*c['orientation_loss_prior']:5.2f} | {100*c['selection_loss']:5.2f}")
    for K, b in r["bootstrap_prior_minus_clip"].items():
        print(f"   boot K={K}: {100*b['point']:+.2f} [{100*b['lo95']:+.2f}, {100*b['hi95']:+.2f}] images prior>CLIP {b['images_prior_better']}, CLIP>prior {b['images_CLIP_better']}")
    for lab, c in r["candidates"].items():
        print("   ", lab, c)
