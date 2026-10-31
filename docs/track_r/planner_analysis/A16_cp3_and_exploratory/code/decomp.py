"""Exploratory (planner): split the gap O_orbit(K) - E1(K) into selection loss and orientation loss,
using exact all-subset weights. GT labels used only for diagnosis.
  O_orient(E0)(K) = P(residual pick is orbit-GOOD: G, O_g or O_p)
  selection loss  = O_orbit - O_orient(E0)   (pick is C or I although an orbit-GOOD run exists)
  orientation loss= O_orient(E0) - E1        (pick orbit-GOOD but the output is not GOOD)
Usage: decomp.py STRINGS CLIP good_col margin_col K1,K2,..."""
import csv, sys, math, json
import numpy as np
S, C, gcol, mcol, Ks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], [int(k) for k in sys.argv[5].split(',')]
ORB = {"G", "O", "O_g", "P", "O_p"}
clip = {}
for r in csv.DictReader(open(C)):
    clip[(r["image_id"], int(r["run"]))] = (int(r[gcol]), float(r[mcol]))
imgs = []
for r in csv.DictReader(open(S)):
    lab = r["labels_by_run"].split(";"); rho = np.array([float(x) for x in r["rho_signed_by_run"].split(";")])
    n = len(lab); cg = np.array([clip[(r["image_id"], j)][0] for j in range(n)])
    imgs.append((r["image_id"], np.array(lab), rho, cg))
def w(n, K):
    return np.array([math.comb(n - r - 1, K - 1) / math.comb(n, K) if n - r >= K else 0.0 for r in range(n)])
out = {}
for K in Ks:
    acc = dict(O_orbit=0, O_raw=0, E0=0, E1=0, O_orient_E0=0, pick_G=0, pick_Og=0, pick_Op=0, pick_C=0, pick_I=0,
               G_broken=0, Og_fixed=0, Og_unfixed=0, Op_lost=0, C_rescued=0, I_rescued=0,
               sel_loss_pickC=0, sel_loss_pickI=0)
    for iid, lab, rho, cg in imgs:
        n = len(lab); order = np.lexsort((np.arange(n), rho)); ww = w(n, K)
        lo = lab[order]; co = cg[order]
        orb = np.isin(lo, list(ORB)).astype(float)
        nO = int(np.isin(lab, list(ORB)).sum()); nG = int((lab == "G").sum())
        pa = lambda c: 1.0 - (math.comb(n - c, K) / math.comb(n, K) if n - c >= K else 0.0)
        acc["O_orbit"] += pa(nO); acc["O_raw"] += pa(nG)
        acc["E0"] += ww @ (lo == "G"); acc["E1"] += ww @ co; acc["O_orient_E0"] += ww @ orb
        for k, cls in [("pick_G", "G"), ("pick_Og", "O_g"), ("pick_Op", "O_p"), ("pick_C", "C"), ("pick_I", "I")]:
            acc[k] += ww @ (lo == cls)
        acc["G_broken"] += ww @ ((lo == "G") & (co == 0)); acc["Og_fixed"] += ww @ ((lo == "O_g") & (co == 1))
        acc["Og_unfixed"] += ww @ ((lo == "O_g") & (co == 0)); acc["Op_lost"] += ww @ ((lo == "O_p") & (co == 0))
        acc["C_rescued"] += ww @ ((lo == "C") & (co == 1)); acc["I_rescued"] += ww @ ((lo == "I") & (co == 1))
        # selection loss by picked class: P(pick is C/I and subset contains an orbit-GOOD run)
        # P(pick = rank r AND subset has an orbit-good run) = P(pick=r) - P(pick=r AND no orbit-good in subset)
        # P(pick=r and no orbit-good): subset = {r} + (K-1) from ranks >r that are non-orbit; needs lo[r] non-orbit
        for r_ in range(n):
            if lo[r_] in ("C", "I"):
                m_non = int((~np.isin(lo[r_ + 1:], list(ORB))).sum())
                p_none = math.comb(m_non, K - 1) / math.comb(n, K) if m_non >= K - 1 else 0.0
                key = "sel_loss_pickC" if lo[r_] == "C" else "sel_loss_pickI"
                acc[key] += ww[r_] - p_none
    N = len(imgs)
    o = {k: v / N for k, v in acc.items()}
    o["selection_loss"] = o["O_orbit"] - o["O_orient_E0"]; o["orientation_loss"] = o["O_orient_E0"] - o["E1"]
    o["gap_O_orbit_minus_E1"] = o["O_orbit"] - o["E1"]
    out[K] = o
json.dump(out, sys.stdout, indent=1)
