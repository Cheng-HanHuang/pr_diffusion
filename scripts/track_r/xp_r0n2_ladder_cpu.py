#!/usr/bin/env python3
"""R0-N.2 A4.3 protocol ladder, CPU-only, from existing frozen R0.7/N2 scores.

All outcomes are historical R0 candidates; no new inference or file traversal.
Only stratum-level PSNR/Good25 aggregates are exported. Per-run records remain PAC-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from collections import Counter,defaultdict
import subprocess

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
R07=BASE/"r0_7_cpu_20261023T080434Z/analysis"
N2=BASE/"r0n_n2clip_recovery_20261024T073405Z"
BAD=("confirmation","confirm305")
METHODS=("mean_of_K","first_run","E0","E1","E1_CLIP","E2","O_raw","O_orbit")

def safe(p):
    p=Path(p)
    if any(w in str(p).lower() for w in BAD):
        raise RuntimeError("STOP_PROTECTED_PATH")
    if p.is_symlink() or not p.is_file():
        raise RuntimeError("STOP_MISSING_OR_SYMLINK")
    if any(w in str(p.resolve()).lower() for w in BAD):
        raise RuntimeError("STOP_PROTECTED_REALPATH")
    return p

def read_json(p):
    return json.loads(safe(p).read_text(encoding="utf-8"))

def digest(p):
    h=hashlib.sha256()
    with safe(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def read_rows(p):
    with safe(p).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def per_image(rows,clip,k):
    """First K frozen-index runs only; score selectors never inspect GT."""
    runs=sorted(rows,key=lambda r:int(r["run_index"]))
    if not (1<=k<=len(runs)):
        raise RuntimeError("STOP_K")
    if [int(r["run_index"]) for r in runs]!=list(range(len(runs))):
        raise RuntimeError("STOP_RUN_INDICES_NOT_CONTIGUOUS")
    runs=runs[:k]
    e0=min(runs,key=lambda r:(float(r["rho_signed_raw"]),int(r["run_index"])))
    clipchoice=clip[e0["uid"]]
    if clipchoice not in ("000","111"):
        raise RuntimeError("STOP_MISSING_CLIP_DECISION")
    tau=float(runs[0]["tau_signed_good_q95"])
    if any(float(r["tau_signed_good_q95"])!=tau for r in runs):
        raise RuntimeError("STOP_E2_TAU_DRIFT")
    feasible=[r for r in runs if float(r["rho_signed_repaired"])<=tau]
    if feasible:
        e2=min(feasible,key=lambda r:(float(r["S_repaired"]),int(r["run_index"])))
    else:
        e2=min(runs,key=lambda r:(float(r["rho_signed_repaired"]),int(r["run_index"])))
    x0=float(e0["psnr_raw"])
    xclip=float(e0["psnr_by_orbit"][clipchoice])
    psnr={
        "first_run":float(runs[0]["psnr_raw"]),
        "mean_of_K":sum(float(r["psnr_raw"]) for r in runs)/len(runs),
        "E0":x0,"E1":float(e0["psnr_repaired"]),
        "E1_CLIP":xclip,"E2":float(e2["psnr_repaired"]),
        "O_raw":max(float(r["psnr_raw"]) for r in runs),
        "O_orbit":max(float(r["psnr_orbit_best"]) for r in runs)
    }
    anatomy=None
    if k==len(rows):
        if psnr["O_orbit"]<25:
            anatomy="generation_failure"
        elif psnr["E1_CLIP"]>=25:
            anatomy="success"
        elif float(e0["psnr_orbit_best"])<25:
            anatomy="selection_miss"
        else:
            anatomy="uncorrected_orientation"
    return {"psnr":psnr,"anatomy":anatomy,
            "fix":int(psnr["E0"]<25<=psnr["E1_CLIP"]),
            "harm":int(psnr["E0"]>=25>psnr["E1_CLIP"])}

def samples_summary(values,seed,n_boot=10000):
    import numpy as np
    a=np.asarray(values,dtype=np.float64)
    if not len(a):raise RuntimeError("STOP_NO_IMAGES")
    success=(a>=25).astype(float)
    h=int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8],"big")%(2**63)
    rng=np.random.default_rng(h)
    ind=rng.integers(0,len(a),size=(n_boot,len(a)))
    mean_bs=a[ind].mean(axis=1)
    good_bs=success[ind].mean(axis=1)
    return {
        "n_images":len(a),
        "good25":float(success.mean()),
        "mean_db":float(a.mean()),
        "q05_db":float(np.quantile(a,.05,method="linear")),
        "good25_ci95":[float(v) for v in np.quantile(good_bs,[.025,.975])],
        "mean_db_ci95":[float(v) for v in np.quantile(mean_bs,[.025,.975])]
    }

def bootstrap_image_rate(rates,seed,n_boot=10000):
    import numpy as np
    v=np.asarray(rates,dtype=np.float64)
    h=int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8],"big")%(2**63)
    rng=np.random.default_rng(h)
    samples=v[rng.integers(0,len(v),size=(n_boot,len(v)))].mean(axis=1)
    return float(v.mean()),[float(x) for x in np.quantile(samples,[.025,.975])]

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_CPU_ONLY")
    wt=Path(__file__).resolve().parents[2]
    got=subprocess.check_output(["git","-C",str(wt),"rev-parse","HEAD"],
                                text=True,timeout=30).strip()
    dirty=subprocess.check_output(["git","-C",str(wt),"status","--porcelain"],
                                  text=True,timeout=30).strip()
    if dirty or got!=a.expected_head:
        raise RuntimeError("STOP_GIT_IDENTITY")
    output=Path(a.out)
    if not str(output).startswith(str(BASE/"r0n2_a43_")) or output.exists():
        raise RuntimeError("STOP_OUTPUT_NAMESPACE")
    evaluation=R07/"R0_7_SCORED_GT_ROWS.jsonl"
    receipt=read_json(R07/"R0_7_COMPLETE_RECEIPT.json")
    rows=read_rows(evaluation)
    if (receipt.get("n_scored")!=3240 or receipt.get("no_protected_access") is not True or
        receipt.get("evaluated_rows_sha256")!=digest(evaluation) or
        len(rows)!=3240 or len({r["uid"] for r in rows})!=3240):
        raise RuntimeError("STOP_R07_REPLAY")
    n2path=N2/"N2_GT_FREE_CLIP_SCORES.json"
    n2=read_json(n2path);n2receipt=read_json(N2/"N2_CLIP_GPU_RECEIPT.json")
    if (n2.get("status")!="R0N_N2_ALL_3240_GT_FREE_CLIP_PASS" or
        n2receipt.get("scored_sha256")!=digest(n2path) or
        n2.get("scoring_gt_access") is not False):
        raise RuntimeError("STOP_N2_REPLAY")
    clip={r["uid"]:r["choice"] for r in n2["rows"]}
    if len(clip)!=3240 or set(clip)!={r["uid"] for r in rows}:
        raise RuntimeError("STOP_N2_UID")
    from scripts.track_r.xp_r0n2_b2_a8_cpu import frozen_cohort_map
    source_to_cohort=frozen_cohort_map(rows)
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        c=source_to_cohort.get(str(int(r["image_id"])),"OTHER_EXPOSED")
        key=(r["source"],r["family"],c)
        groups[key][str(r["image_id"])].append(r)
        if r["source"]=="S5_r0_fast":
            groups[(r["source"],r["family"],"POOLED_180")][str(r["image_id"])].append(r)
    output.mkdir(parents=True,exist_ok=False)
    tables=[]
    strata=[]
    for (src,family,cohort),images in sorted(groups.items()):
        ns={len(r) for r in images.values()}
        if len(ns)!=1:raise RuntimeError("STOP_NONUNIFORM_K "+src+"/"+family)
        maxk=ns.pop()
        chosen={}
        for k in sorted({1,maxk}):
            picked=[per_image(rs,clip,k) for _,rs in sorted(images.items())]
            vals={method:[p["psnr"][method] for p in picked] for method in METHODS}
            for meth in METHODS:
                seed="R0N2_A43|"+src+"|"+family+"|"+cohort+"|"+meth+"|"+str(k)
                stats=samples_summary(vals[meth],seed)
                if meth=="mean_of_K":
                    byimg=[sum(float(r["psnr_raw"])>=25 for r in sorted(rs,key=lambda z:int(z["run_index"]))[:k])/k
                           for _,rs in sorted(images.items())]
                    stats["good25"],stats["good25_ci95"]=bootstrap_image_rate(byimg,seed+"|RUN_GOOD")
                tables.append({"source":src,"family":family,"cohort":cohort,
                               "K":k,"policy":meth,**stats})
            chosen[k]=dict(zip(METHODS,[sum(vals[m])/len(vals[m]) for m in METHODS]))
            if k==maxk:
                anatomy=Counter(p["anatomy"] for p in picked)
                if sum(anatomy.values())!=len(images):
                    raise RuntimeError("STOP_ANATOMY_TOTAL")
                strata.append({"source":src,"family":family,"cohort":cohort,
                     "max_K":maxk,"n_images":len(images),
                     "anatomy":dict(anatomy),
                     "e1_clip_fixes":sum(p["fix"] for p in picked),
                     "e1_clip_harms":sum(p["harm"] for p in picked)})
        first=chosen[1]["first_run"]
        maxvals=chosen[maxk]
        denom_orbit=maxvals["O_orbit"]-first
        denom_raw=maxvals["O_raw"]-first
        strata[-1].update({
            "E1_CLIP_minus_E1_db":maxvals["E1_CLIP"]-maxvals["E1"],
            "clip_share_orbit_gain":(maxvals["E1_CLIP"]-first)/denom_orbit
                if abs(denom_orbit)>1e-9 else None,
            "prior_share_orbit_gain":(maxvals["E1"]-first)/denom_orbit
                if abs(denom_orbit)>1e-9 else None,
            "clip_share_raw_gain":(maxvals["E1_CLIP"]-first)/denom_raw
                if abs(denom_raw)>1e-9 else None,
            "prior_share_raw_gain":(maxvals["E1"]-first)/denom_raw
                if abs(denom_raw)>1e-9 else None})
    csv_path=output/"R0N2_A43_LADDER_AGGREGATES.csv"
    fields=("source","family","cohort","K","policy","n_images","good25",
            "mean_db","q05_db","good25_ci95","mean_db_ci95")
    with csv_path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in tables:
            writer.writerow({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k]
                             for k in fields})
    summary=output/"R0N2_A43_POSTSELECTION.json"
    summary.write_text(json.dumps({"status":"R0N2_A43_CPU_PASS","strata":strata},
                                  indent=2,sort_keys=True)+"\n",encoding="utf-8")
    complete={"status":"R0N2_A43_CPU_PASS","head":got,
      "n_candidates":len(rows),"n_strata":len(strata),"n_ladder_rows":len(tables),
      "r07_sha256":digest(evaluation),"n2_sha256":digest(n2path),
      "ladder_sha256":digest(csv_path),"anatomy_sha256":digest(summary),
      "no_new_inference":True,"cpu_only":True}
    (output/"R0N2_A43_COMPLETE.json").write_text(json.dumps(complete,indent=2,sort_keys=True)+"\n")
    print("R0N2_A43_CPU_PASS strata="+str(len(strata))+" rows="+str(len(tables)))
    print("LADDER_SHA256 "+complete["ladder_sha256"])
    print("OK")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    try:main(ap.parse_args())
    except Exception as e:
        print("ACTION NEEDED: R0N2_A43 "+str(e))
        raise SystemExit(8)
