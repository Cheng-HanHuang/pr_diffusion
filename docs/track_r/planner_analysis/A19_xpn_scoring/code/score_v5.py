"""Planner: score registered v5 (XP-N) against the complete 64-run panel with the frozen comparison rule.
Observed values: exact all-subset estimators from the frozen protocol code (fixed_policies)."""
import sys, json, importlib.util, csv
import numpy as np
code, strings, clip, v5path = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("gp", code); gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)
imgs = gp.attach_clip(gp.load_strings(strings), clip, "g1_chosen_good", "g1_margin")
assert len(imgs) == 100 and all(len(im["rho"]) == 64 for im in imgs)
KS = [4, 8, 16, 32, 64]
obs = {}
for K in KS:
    v = np.array([gp.fixed_policies(im, K) for im in imgs]).mean(axis=0)
    obs[K] = dict(E0=v[0], E1_CLIP=v[1], O_raw=v[2], O_orbit=v[3])
v5 = json.load(open(v5path))
out = {"observed": {K: {k: float(x) for k, x in d.items()} for K, d in obs.items()}, "variants": {}}
for var in ["race", "S2", "tail", "race_tau1"]:
    rows = []; cov = 0; errs = []; other = {}
    for p in v5["predictions"]:
        if p["model_variant"] != var: continue
        o = obs[p["K"]][p["policy"]]
        inside = p["lo95"] - 1e-12 <= o <= p["hi95"] + 1e-12
        key = (p["policy"], p["decision_model"])
        if key in (("E0", None), ("E1_CLIP", "beta_class")):
            cov += inside
            errs.append(abs(o - p["point"]) / max((p["hi95"] - p["lo95"]) / 3.92, 0.005))
        other.setdefault(f"{p['policy']}|{p['decision_model']}", []).append(int(inside))
        rows.append(dict(K=p["K"], policy=p["policy"], decision_model=p["decision_model"], point=p["point"], lo95=p["lo95"], hi95=p["hi95"], observed=float(o), inside=bool(inside)))
    out["variants"][var] = dict(primary_coverage=int(cov), primary_n=len(errs), mean_abs_std_error=float(np.mean(errs)),
                                coverage_by_policy={k: f"{sum(v)}/{len(v)}" for k, v in other.items()}, rows=rows)
print(json.dumps({k: (v if k != "variants" else {vv: {kk: x for kk, x in d.items() if kk != "rows"} for vv, d in v.items()}) for k, v in out.items()}, indent=1, default=str))
json.dump(out, open(sys.argv[5], "w"), indent=1, default=str)
