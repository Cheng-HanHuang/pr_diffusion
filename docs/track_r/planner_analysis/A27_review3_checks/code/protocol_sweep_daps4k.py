import json, subprocess, sys, os
W = os.path.dirname(os.path.abspath(__file__))
def run(g, cohort, tau=None, rule=None, m=2, tag=""):
    args = ["python3", "-I", f"{W}/gtfree_protocol.py", "--strings", f"{W}/in/{g}_STRINGS.csv", "--clip", f"{W}/in/{g}_CLIP.csv", "--m", str(m), "--nmax", "8"]
    if g == "G4": args += ["--clip-good-col", "face_chosen_good", "--clip-margin-col", "face_margin"]
    if cohort: args += ["--cohort", cohort]
    args += (["--tau-rule", rule] if rule else ["--tau", repr(tau)])
    out = f"{W}/out/{g}_{cohort}_{tag}.json"; args += ["--out", out]
    subprocess.run(args, check=True, capture_output=True)
    d = json.load(open(out))
    return {"tau": d["tau"], "success": d["success_all"], "runs": d["runs_mean"], "gain": d["gain_vs_fixed_equal_runs"], "flag_c": d["flag_c_expected_count"], "n": d["n_images"]}
res = {}
panels = [("G4", "B22"), ("G4", "DEV80"), ("G5", None)]
for g, c in panels:
    key = f"{g}_{c}"; res[key] = {}
    reg = run(g, c, rule="self_mad3", tag="reg"); res[key]["registered_self_mad3"] = reg
    t0 = reg["tau"]
    lab = 0.04746491 if g == "G4" else None
    import csv
    rows = list(csv.DictReader(open(f"{W}/in/{g}_STRINGS.csv")))
    lab = float(rows[0]["tau_signed_good_q95"])
    res[key]["labeled_q95"] = run(g, c, tau=lab, tag="lab")
    for f in (0.965, 0.98, 1.02, 1.035):
        res[key][f"x{f}"] = run(g, c, tau=t0 * f, tag=f"x{f}")
    for m in (1, 3, 4):
        res[key][f"M{m}"] = run(g, c, tau=t0, m=m, tag=f"M{m}")
# cross-cohort: tau of the other cohort's pilot
res["G4_B22"]["cross_tau_from_DEV80"] = run("G4", "B22", tau=res["G4_DEV80"]["registered_self_mad3"]["tau"], tag="cross")
res["G4_DEV80"]["cross_tau_from_B22"] = run("G4", "DEV80", tau=res["G4_B22"]["registered_self_mad3"]["tau"], tag="cross")
json.dump(res, open(f"{W}/out/SWEEP.json", "w"), indent=1)
for k, v in res.items():
    print(k)
    for kk, vv in v.items():
        print("  %-22s tau %.6f  succ %.4f  runs %.3f  gain %+.4f  flagC %.1f" % (kk, vv["tau"], vv["success"], vv["runs"], vv["gain"], vv["flag_c"]))
