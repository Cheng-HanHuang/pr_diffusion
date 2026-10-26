#!/usr/bin/env python3
"""CPU-only replay scorer recovery for the existing 2026-10-26 FAST run.

A newer CPU source checkout may score an old, immutable GPU-generation run.
Never rerun CUDA, rewrite original supervisor status, or alter PNGs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
RUN=ROOT/"outputs/pr_diffusion/track_r/xp_fast_replay_20261026T203415Z"
SOURCE_HEAD="314180b314674a24d5ddc40a0186b24b6082a5e0"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def safe(p):
    p=Path(p)
    if any(t in str(p).lower() for t in BAD):
        raise RuntimeError("STOP_PROTECTED_PATH")
    resolved=p.resolve(strict=False)
    if any(t in str(resolved).lower() for t in BAD):
        raise RuntimeError("STOP_PROTECTED_REALPATH")
    root=ROOT.resolve(strict=False)
    if not (resolved==root or root in resolved.parents) or not p.is_file():
        raise RuntimeError("STOP_UNSAFE_OR_MISSING_FILE "+str(p))
    return p

def read(p):
    return json.loads(safe(p).read_text(encoding="utf-8"))

def sha(p):
    h=hashlib.sha256()
    with safe(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def validate_original():
    original=read(RUN/"XP_FAST_REPLAY_STATUS.json")
    gpu=read(RUN/"gpu/R0_5_CHUNK_COMPLETE.json")
    meta=read(RUN/"gpu/R0_5_CHUNK_META.json")
    prep=read(RUN/"XP_FAST_REPLAY_PREP_RECEIPT.json")
    timing=read(RUN/"XP_FAST_REPLAY_TIMING.json")
    manifest=read(RUN/"XP_FAST_REPLAY_TASKS.json")
    if not (
        original.get("state")=="REPLAY_GATE_FAILED" and
        original.get("worker_returncode")==0 and
        original.get("cpu_score_rc")==8 and
        original.get("head")==SOURCE_HEAD and
        original.get("n_roots")==8 and
        gpu.get("status")=="R0_5_CHUNK_PASS" and
        gpu.get("count")==8 and gpu.get("new")==8 and gpu.get("reused")==0 and
        gpu.get("head")==SOURCE_HEAD and
        gpu.get("meta_sha256")==sha(RUN/"gpu/R0_5_CHUNK_META.json") and
        meta.get("head")==SOURCE_HEAD and
        prep.get("head")==SOURCE_HEAD and
        prep.get("n_roots")==8 and
        prep.get("manifest_sha256")==sha(RUN/"XP_FAST_REPLAY_TASKS.json") and
        manifest.get("head")==SOURCE_HEAD and
        manifest.get("n_roots")==8 and
        timing.get("head")==SOURCE_HEAD and
        timing.get("worker_returncode")==0 and
        float(timing.get("gpu_hours",-1))>0):
        raise RuntimeError("STOP_OLD_GPU_EVIDENCE_NOT_FROZEN")
    return original,gpu,prep,timing

def validate_scored(generated_on,now):
    receipt=read(RUN/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json")
    rows=receipt.get("results",[])
    uids={r.get("uid") for r in rows}
    if not (receipt.get("head")==now and
            receipt.get("generation_head")==generated_on and
            receipt.get("status") in ("XP_FAST_REPLAY_PASS","XP_FAST_REPLAY_FAIL") and
            receipt.get("n_runs")==8 and len(rows)==8 and len(uids)==8 and
            all(r.get("pass") is
                (float(r["abs_delta_db"])<=0.1 and
                 r["label"]==r["historical_label"]) for r in rows)):
        raise RuntimeError("STOP_RECOVERED_REPLAY_SCHEMA")
    return receipt

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_CPU_ONLY")
    current=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if dirty or current!=a.expected_head or a.source_head!=SOURCE_HEAD:
        raise RuntimeError("STOP_GIT_IDENTITY")
    original,gpu,prep,timing=validate_original()
    report=RUN/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json"
    if report.exists():
        receipt=validate_scored(SOURCE_HEAD,current)
    else:
        command=[sys.executable,str(WT/"scripts/track_r/xp_fast_replay_cpu.py"),
                 "score","--out",str(RUN),"--expected-head",current,
                 "--source-head",SOURCE_HEAD]
        env=dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"]=""
        env["PYTHONDONTWRITEBYTECODE"]="1"
        env["PYTHONNOUSERSITE"]="1"
        env["PYTHONPATH"]=str(WT)
        result=subprocess.run(command,env=env,cwd=str(WT),capture_output=True,text=True,
                              timeout=3600)
        print(result.stdout.strip(),flush=True)
        if result.returncode:
            print(result.stderr[-5000:],flush=True)
            raise RuntimeError("STOP_CPU_REPLAY_SCORER_RC_"+str(result.returncode))
        receipt=validate_scored(SOURCE_HEAD,current)
    if receipt["status"]!="XP_FAST_REPLAY_PASS":
        print("REPLAY_SCIENCE_FAIL "+json.dumps(
             [(r["uid"],r["abs_delta_db"],r["label"],r["historical_label"])
              for r in receipt["results"]],sort_keys=True))
        raise RuntimeError("STOP_REPLAY_NOT_EXCHANGEABLE")
    out=RUN/"XP_FAST_REPLAY_RECOVERY_STATUS.json"
    if out.exists():
        raise RuntimeError("STOP_RECOVERY_RECEIPT_ALREADY_EXISTS")
    proof={
        "status":"XP_FAST_REPLAY_RECOVERY_PASS",
        "science":"XP_FAST_REPLAY_PASS",
        "gpu_source_head":SOURCE_HEAD,
        "cpu_recovery_head":current,
        "root":str(RUN),
        "n_roots":8,
        "max_abs_delta_db":receipt["max_abs_delta_db"],
        "n_png_exact":receipt["n_png_exact"],
        "r07_label_matches":sum(x["label"]==x["historical_label"] for x in receipt["results"]),
        "original_supervisor_state":original["state"],
        "original_gpu_worker_rc":original["worker_returncode"],
        "gpu_hours_original":timing["gpu_hours"],
        "timing":timing["timing"],
        "original_chunk_receipt_sha256":sha(RUN/"gpu/R0_5_CHUNK_COMPLETE.json"),
        "immutable_gpu_manifest_sha256":prep["manifest_sha256"],
        "scored_replay_sha256":sha(report),
        "gpu_rerun_performed":False,
        "gt_access_in_generation":False,
        "cpu_only_evaluation":True
    }
    with out.open("x",encoding="utf-8") as f:
        json.dump(proof,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("XP_FAST_REPLAY_RECOVERY_PASS n=8 max_delta_db="+str(proof["max_abs_delta_db"]))
    print("REPLAY_LABELS_MATCH=8/8")
    print("PNG_EXACT="+str(proof["n_png_exact"])+"/8")
    print("GPU_RERUN_PERFORMED=False")
    print("RECOVERY_SHA256="+sha(out))
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--source-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        print("ACTION NEEDED: FAST_REPLAY_CPU_RECOVERY "+str(e))
        raise SystemExit(8)
