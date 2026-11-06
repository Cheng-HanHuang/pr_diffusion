import json, os, sys, collections
d = sys.argv[1]
J = lambda f: json.load(open(os.path.join(d, f)))
pts = []
for r in J("test1_fast_ffhq_v1.json")["forecast_comparison"]:
    if r["panel"] == "primary": pts.append((1, r["model_variant"], r["policy"], r["point"], r["lo95"], r["hi95"], r["observed"]))
for r in J("test1_fast_ffhq_e1_a92d.json")["comparisons_detail"]:
    if r["panel"] == "primary": pts.append((1, r["variant"], "E1_CLIP", r["point"], r["lo95"], r["hi95"], r["observed"]))
cp2 = J("test2_5_imagenet_cp2.json")
for key, test in (("v2", 2), ("v2 E1", 2), ("v4", 5)):
    for r in cp2[key]["comparisons"]:
        pass
        pts.append((test, r["model_variant"], r["policy"], r["forecast"], r["lo95"], r["hi95"], r["observed"]))
for r in J("test3_sitcom_b16.json")["forecast_comparisons"]:
    if r["forecast_source"] == "v1" and r["panel"] == "primary": pts.append((3, r["model_variant"], r["policy"], r["point"], r["lo95"], r["hi95"], r["observed"]))
for r in J("test4_daps_f8.json")["comparisons"]:
    if r["version"] == "v1": pts.append((4, r["model_variant"], r["policy"], r["point"], r["lo95"], r["hi95"], r["observed"]))
t6 = J("test6_imagenet_new_v5.json")["variants"]
for var in ("race", "S2"):
    for r in t6[var]["rows"]:
        if r["decision_model"] in (None, "beta_class"): pts.append((6, var, r["policy"], r["point"], r["lo95"], r["hi95"], r["observed"]))
A = 0.05
def isc(lo, hi, y): return (hi - lo) + (2 / A) * max(lo - y, 0) + (2 / A) * max(y - hi, 0)
grp = lambda pol: "selected" if pol in ("E0", "E1_CLIP") else ("oracle" if pol in ("O_raw", "O_orbit") else None)
agg = collections.defaultdict(lambda: collections.defaultdict(float))
for t, v, pol, p, lo, hi, y in pts:
    g = grp(pol)
    if g is None or v not in ("race", "S2"): continue
    for key in ((t, v, g), ("all", v, g), ("not1", v, g) if t != 1 else None):
        if key is None: continue
        a = agg[key]; a["n"] += 1; a["cov"] += lo <= y <= hi; a["mae"] += abs(p - y); a["width"] += hi - lo; a["is"] += isc(lo, hi, y); a["bias"] += p - y
out = {}
for k, a in sorted(agg.items(), key=lambda kv: (str(kv[0][0]), kv[0][2], kv[0][1])):
    n = a["n"]; out["|".join(map(str, k))] = {"n": int(n), "covered": int(a["cov"]), "MAE_pts": round(100 * a["mae"] / n, 2),
        "mean_width_pts": round(100 * a["width"] / n, 2), "mean_interval_score_pts": round(100 * a["is"] / n, 2), "mean_bias_pts": round(100 * a["bias"] / n, 2)}
json.dump(out, open(sys.argv[2], "w"), indent=1)
for k, v in out.items(): print(k, v)
