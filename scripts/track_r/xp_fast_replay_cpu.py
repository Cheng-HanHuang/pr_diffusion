#!/usr/bin/env python3
"""Frozen GT-free FAST-DIPS replay task preparation and offline GT evaluation.

R0.5 roots and R0.7 evaluated labels are immutable; no bundled datasets.
The 'prepare' phase never reads GT; 'score' accesses GT only on CPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
ROOTS=BASE/"r0_5_canary_20261023T004927Z/prepare/R0_5_GT_FREE_ROOTS.json"
ROOTS_SHA="2291aa793b612c973ae76ab3d36b1f20263cd747f3b23cf593e51517a26c3bfe"
R07_CATALOG=BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json"
R07_SCORED=BASE/"r0_7_cpu_20261023T080434Z/analysis/R0_7_SCORED_GT_ROWS.jsonl"

def safe(p):
    p=Path(p)
    if any(w in str(p).lower() for w in BAD):
        raise RuntimeError("STOP_PROTECTED_PATH")
    real=p.resolve(strict=False)
    if any(w in str(real).lower() for w in BAD):
        raise RuntimeError("STOP_PROTECTED_REALPATH")
    root=ROOT.resolve(strict=False)
    if not (real==root or root in real.parents):
        raise RuntimeError("STOP_OUTSIDE_PAC_ROOT")
    if not p.is_file():raise RuntimeError("STOP_SOURCE_MISSING "+str(p))
    return p

def sha(p):
    h=hashlib.sha256()
    with safe(p).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def obj(p):
    return json.loads(safe(p).read_text())

def verify_head(expected):
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    if head!=expected or subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
                                                 text=True).strip():
        raise RuntimeError("STOP_WRONG_OR_DIRTY_GIT_HEAD")

def frozen_selection(rows):
    """Two independently seeded runs on first two frozen ids per source."""
    target=[]
    for source in ("S1_b22","S2_b24_dev80"):
        ids=sorted({r["image_id"] for r in rows if r["measurement_source"]==source})
        if len(ids)!=(100 if source=="S1_b22" else 80):
            raise RuntimeError("STOP_FROZEN_180_SOURCE_IDS")
        for iid in ids[:2]:
            for rep in (0,1):
                match=[r for r in rows if r["measurement_source"]==source and
                       r["image_id"]==iid and r["rep"]==rep]
                if len(match)!=1:raise RuntimeError("STOP_MISSING_FROZEN_RUN")
                target.append(match[0])
    if len(target)!=8:raise RuntimeError("STOP_REPLAY_8_ROOTS")
    return target

def prepare(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_CPU_PREP_GPU_VISIBLE")
    verify_head(args.expected_head)
    from scripts.track_r.r0_5_fast_gpu import frozen_roots, CANARY_HEAD
    if sha(ROOTS)!=ROOTS_SHA:raise RuntimeError("STOP_R05_ROOT_MANIFEST_HASH")
    roots=frozen_roots(obj(ROOTS),CANARY_HEAD)
    rows=frozen_selection(roots)
    if rows is None:raise RuntimeError("STOP_REPLAY_8_ROOTS")
    from scripts.track_r.xp_0_inventory_cpu import CATALOG
    reference=obj(CATALOG)
    identities={(r["measurement_source"],str(r["image_id"])):r["measurement_sha256"]
                for r in reference["rows"] if r["source"]=="S5_r0_fast"}
    for r in rows:
        key=(r["measurement_source"],str(r["image_id"]))
        if key not in identities or sha(r["measurement_path"])!=r["measurement_sha256"] or identities[key]!=r["measurement_sha256"]:
            raise RuntimeError("STOP_REPLAY_FROZEN_Y_DRIFT "+r["uid"])
    target=Path(args.out)
    if not str(target).startswith(str(BASE/"xp_fast_replay_")) or target.exists() or target.is_symlink():
        raise RuntimeError("STOP_REPLAY_TARGET")
    target.mkdir(parents=True,exist_ok=False)
    manifest={"schema":"prdiff.track_r.xp.fast-tasks.v1",
              "status":"XP_FAST_TASKS_FROZEN","dataset":"ffhq",
              "oversample":2.0,"head":args.expected_head,
              "fast_head":"a9758b38906b7c7950fa04491f37f6634948db26",
              "n_roots":len(rows),"gt_paths_supplied":False,"rows":rows,
              "source_r05_roots_sha256":ROOTS_SHA,
              "selection_rule":"first two sorted image IDs from each R0 cohort, reps 0 and 1"}
    out=target/"XP_FAST_REPLAY_TASKS.json"
    out.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    report={"status":"XP_FAST_REPLAY_PREP_PASS","head":args.expected_head,
            "manifest_sha256":sha(out),"n_roots":len(rows),
            "measurement_sha256":sorted(set(r["measurement_sha256"] for r in rows))}
    (target/"XP_FAST_REPLAY_PREP_RECEIPT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("XP_FAST_REPLAY_PREP_PASS n=8")
    print("TASK_SHA="+report["manifest_sha256"])
    print("OK")

def score(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_REPLAY_SCORER_NOT_CPU")
    verify_head(args.expected_head)
    import torch
    from prdiffusion.track_r.physics import load_png_rgb8,to_rgb8,psnr_rgb8
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse,label_candidate
    from scripts.track_r.r0_2_g0_cpu import gt_source
    from scripts.track_r.xp_fast_gpu import validate_complete,root_folder,frozen_roots
    manifest_path=Path(args.out)/"XP_FAST_REPLAY_TASKS.json"
    manifest=obj(manifest_path)
    rows=frozen_roots(manifest,args.expected_head)
    receipt=obj(Path(args.out)/"XP_FAST_REPLAY_PREP_RECEIPT.json")
    if sha(manifest_path)!=receipt["manifest_sha256"] or len(rows)!=8:
        raise RuntimeError("STOP_REPLAY_TASK_DRIFT")
    chunk=Path(args.out)/"gpu"
    chunk_receipt=obj(chunk/"R0_5_CHUNK_COMPLETE.json")
    if chunk_receipt.get("status")!="R0_5_CHUNK_PASS" or chunk_receipt.get("count")!=8:
        raise RuntimeError("STOP_REPLAY_CHUNK_STATUS")
    # Ground-truth paths are read exclusively in this CPU-only scorer.
    catalog=obj(R07_CATALOG)["rows"]
    scores=[json.loads(line) for line in safe(R07_SCORED).read_text().splitlines() if line.strip()]
    old={r["uid"]:r for r in scores if r["source"]=="S5_r0_fast"}
    oldcat={r["uid"]:r for r in catalog if r["source"]=="S5_r0_fast"}
    output=[]
    for r in rows:
        uid=r["uid"]
        if uid not in old or uid not in oldcat:raise RuntimeError("STOP_REPLAY_UID")
        prev=old[uid]
        meta=oldcat[uid]
        if (meta["measurement_sha256"]!=r["measurement_sha256"] or
            prev["measurement_sha256"]!=r["measurement_sha256"] or
            prev["run_index"]!=r["rep"]):
            raise RuntimeError("STOP_REPLAY_IDENTITY")
        folder=root_folder(chunk,r)
        native=validate_complete(folder,r,args.expected_head,sha(manifest_path))
        safe(meta["gt_path"])
        gt8=to_rgb8(gt_source(meta))
        x8=load_png_rgb8(folder/"reconstruction.png")
        raw=float(psnr_rgb8(x8,gt8)[0])
        orbit=orbit_psnr_channel_mse(x8,gt8)
        best=max(float(x) for x in orbit.values())
        tau=float(prev["tau_signed_good_q95"])
        if raw>=25:label="GOOD"
        elif best>=25:label="ORBIT_ERROR"
        else:
            from prdiffusion.track_r.physics import from_rgb8,rho_signed
            from scripts.track_a.a15_fastdips_preflight import load_tensor
            observation=load_tensor(safe(r["measurement_path"]),
                                    ("measurement","y","observation"),(1,3,384,384))
            residual=float(rho_signed(from_rgb8(x8),observation.float())[0])
            label="NONORBIT_CONSISTENT" if residual<=tau else "NONORBIT_INCONSISTENT"
        delta=abs(raw-float(prev["psnr_raw"]))
        output.append({"uid":uid,"source":r["measurement_source"],"rep":r["rep"],
                       "raw_psnr":raw,"historical_psnr":prev["psnr_raw"],
                       "abs_delta_db":delta,"label":label,
                       "historical_label":prev["raw_label"],
                       "exact_png_match":sha(folder/"reconstruction.png")==meta["candidate_sha256"],
                       "pass":delta<=0.1 and label==prev["raw_label"]})
    status="XP_FAST_REPLAY_PASS" if len(output)==8 and all(x["pass"] for x in output) else "XP_FAST_REPLAY_FAIL"
    report={"status":status,"head":args.expected_head,
        "n_runs":len(output),"max_abs_delta_db":max(x["abs_delta_db"] for x in output),
        "n_png_exact":sum(x["exact_png_match"] for x in output),
        "measurement_gate":"FROZEN_R0_SHA_PASS",
        "results":output}
    target=Path(args.out)/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json"
    if target.exists():raise RuntimeError("STOP_REPLAY_RECEIPT_ALREADY_EXISTS")
    target.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(status+" n=8 max_psnr_delta_db="+format(report["max_abs_delta_db"],".5f")+
          " png_exact="+str(report["n_png_exact"]))
    print("REPLAY_RECEIPT_SHA="+sha(target))
    print("OK" if status.endswith("PASS") else "ACTION NEEDED: FAST_REPLAY_FAILED")
    if status.endswith("FAIL"):raise RuntimeError("STOP_REPLAY_SCIENCE")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("prepare","score"))
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    try:
        a=ap.parse_args()
        prepare(a) if a.mode=="prepare" else score(a)
    except Exception as exc:
        print("ACTION NEEDED: XP_FAST_REPLAY "+str(exc))
        raise SystemExit(8)
