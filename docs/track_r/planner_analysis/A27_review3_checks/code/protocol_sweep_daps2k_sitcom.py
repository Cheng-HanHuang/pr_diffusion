import json, subprocess, os, csv
W = os.path.dirname(os.path.abspath(__file__))
CFG = {"F8": ("f_chosen_good", "f_margin", 8), "B16": ("f_chosen_good", "f_margin", 16)}
def run(g, cohort, tau=None, rule=None, m=2, tag=""):
    gc, mc, nmax = CFG[g]
    args = ["python3", "-I", f"{W}/gtfree_protocol.py", "--strings", f"{W}/in/{g}_STRINGS.csv", "--clip", f"{W}/in/{g}_CLIP.csv",
            "--clip-good-col", gc, "--clip-margin-col", mc, "--m", str(m), "--nmax", str(nmax), "--cohort", cohort]
    args += (["--tau-rule", rule] if rule else ["--tau", repr(tau)])
    out = f"{W}/out/{g}_{cohort}_{tag}.json"; args += ["--out", out]
    subprocess.run(args, check=True, capture_output=True)
    d = json.load(open(out))
    return {"tau": d["tau"], "success": d["success_all"], "runs": d["runs_mean"], "gain": d["gain_vs_fixed_equal_runs"], "flag_c": d["flag_c_expected_count"], "n": d["n_images"]}
res = {}
for g in ("F8", "B16"):
    rows = list(csv.DictReader(open(f"{W}/in/{g}_STRINGS.csv")))
    labt = {}
    for r in rows: labt.setdefault(r["cohort"], float(r["tau_signed_good_q95"]))
    for c in ("B22", "DEV80"):
        key = f"{g}_{c}"; res[key] = {}; t0 = labt[c]
        res[key]["registered_labeled"] = run(g, c, tau=t0, tag="reg")
        res[key]["label_free_self_mad3"] = run(g, c, rule="self_mad3", tag="lf")
        for f in (0.965, 0.98, 0.995, 1.005, 1.02, 1.035):
            res[key][f"x{f}"] = run(g, c, tau=t0 * f, tag=f"x{f}")
        for m in (1, 3, 4):
            res[key][f"M{m}"] = run(g, c, tau=t0, m=m, tag=f"M{m}")
json.dump(res, open(f"{W}/out/SWEEP2.json", "w"), indent=1)
for k, v in res.items():
    print(k)
    for kk, vv in v.items():
        print("  %-22s tau %.6f  succ %.4f  runs %.3f  gain %+.4f  flagC %.1f" % (kk, vv["tau"], vv["success"], vv["runs"], vv["gain"], vv["flag_c"]))
