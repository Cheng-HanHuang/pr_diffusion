"""Exploratory replay (planner, post hoc on a seen panel): protocol v1 with a confidence condition.
Stop when >= M runs pass rho<=tau AND |CLIP margin| of the current residual pick >= c; else continue to N_MAX.
c is fixed in advance to the FLAG_O cut of the D1-P2 primary panel (0.0076269954442977905), not tuned here."""
import csv, sys, json, math
import numpy as np
S, C, tau, c, M, R, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), 2, 4000, 20261029
nmaxes = [int(x) for x in sys.argv[5].split(',')]
curve = json.load(open(sys.argv[6]))["fixed_curves"]["E1"]
clip = {(r["image_id"], int(r["run"])): (int(r["g1_chosen_good"]), abs(float(r["g1_margin"]))) for r in csv.DictReader(open(C))}
imgs = []
for r in csv.DictReader(open(S)):
    rho = np.array([float(x) for x in r["rho_signed_by_run"].split(";")]); n = len(rho)
    cg = np.array([clip[(r["image_id"], j)][0] for j in range(n)], float); cm = np.array([clip[(r["image_id"], j)][1] for j in range(n)])
    imgs.append((rho, cg, cm))
def interp(k):
    lo = int(math.floor(k)); hi = min(lo + 1, len(curve)); lo = max(lo, 1)
    return curve[lo - 1] if hi == lo else (1 - (k - lo)) * curve[lo - 1] + (k - lo) * curve[hi - 1]
out = {}
for N in nmaxes:
    rng = np.random.default_rng(seed); S_ = []; U_ = []
    for rho, cg, cm in imgs:
        n = len(rho); perm = np.argsort(rng.random((R, n)), axis=1)[:, :N]
        rr = rho[perm]; passed = rr <= tau; cp = np.cumsum(passed, axis=1)
        # running argmin position
        run_min_pos = np.zeros((R, N), int); best = np.full(R, np.inf); bp = np.zeros(R, int)
        for t in range(N):
            better = rr[:, t] < best; best = np.where(better, rr[:, t], best); bp = np.where(better, t, bp); run_min_pos[:, t] = bp
        pick_run = perm[np.arange(R)[:, None], run_min_pos]            # run index of current pick at each t
        conf = cm[pick_run] >= c
        stop_ok = (cp >= M) & conf
        stop = np.where(stop_ok.any(axis=1), stop_ok.argmax(axis=1) + 1, N)
        final_pick = pick_run[np.arange(R), stop - 1]
        S_.append(cg[final_pick].mean()); U_.append(stop.mean())
    s, u = float(np.mean(S_)), float(np.mean(U_))
    out[N] = dict(success=s, runs=u, fixed_eq=interp(u), gain=s - interp(u))
for N, o in out.items(): print(f"N_MAX={N:3d} success={o['success']:.4f} runs={o['runs']:.3f} fixed_E1_eq={o['fixed_eq']:.4f} gain={o['gain']:+.4f}")
