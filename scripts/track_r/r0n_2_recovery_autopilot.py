#!/usr/bin/env python3
"""N2-only bounded recovery using the immutable 20261024 prepared PNGs.

The prior N2 attempt used 6.83 seconds of worker wall time and failed
before any candidate inference. Authorize at most 870 additional
worker wall seconds, for <900 seconds total. No N0/N1/N3 reruns,
new reconstructions, new measurements, or protected data access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
CACHE=ROOT/"outputs/.cache/track_r"
BRANCH="codex/prdiff-track-r-r0n-20261023"
SOURCE_N0N1N3="5af47dbb4f6dba5ba5aa140b9b30677039b31e39"
PRIOR_STAGE_SECONDS=6.83
RETRY_LIMIT_SECONDS=870.0
PREP=BASE/"r0n_n2prep_20261024T061111Z"
PREP_SHA="7be734de4cd2e6f395add521717506be9260c37708f8902e248c1848b68b705b"
PREP_HEAD="b1637f7c27a3caaa03a8b204171fca0d3fccfcd5"
FORBIDDEN=("confirmation","confirm305")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda:f.read(4*1024*1024),b""):h.update(x)
    return h.hexdigest()


def load(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or any(
       token in str(p.resolve()).lower() for token in FORBIDDEN):
        raise RuntimeError("STOP_N2_RECOVERY_FILE")
    return json.loads(p.read_text(encoding="utf-8"))


def write_x(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")


def git(*args,check=True,timeout=45):
    env=dict(os.environ)
    env["GIT_TERMINAL_PROMPT"]="0"
    home=env.get("R0N_GIT_ORIGINAL_HOME")
    if home:env["HOME"]=home
    p=subprocess.run(["git","-C",str(WT),*args],capture_output=True,
        text=True,env=env,timeout=timeout)
    if p.returncode and check:
        raise RuntimeError("STOP_N2_RECOVERY_GIT "+p.stdout[-400:]+p.stderr[-400:])
    return p


def environment(gpu=None):
    env=dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES":"" if gpu is None else str(gpu),
        "PYTHONPATH":str(WT),"PYTHONDONTWRITEBYTECODE":"1",
        "PYTHONNOUSERSITE":"1",
        "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
        "XDG_CACHE_HOME":str(CACHE),"XDG_CONFIG_HOME":str(CACHE/"config"),
        "XDG_DATA_HOME":str(CACHE/"data"),"XDG_STATE_HOME":str(CACHE/"state"),
        "HF_HOME":str(CACHE/"hf"),"TORCH_HOME":str(CACHE/"torch"),
        "PIP_CACHE_DIR":str(CACHE/"pip"),"OMP_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    for name in ("home","tmp","config","data","state","hf","torch","pip"):
        (CACHE/name).mkdir(parents=True,exist_ok=True)
    return env


def stage(out,status,name,args,gpu=None,timeout=1200):
    log=out/"logs"/(name+".log")
    with log.open("x") as f:
        f.write("COMMAND "+repr(args)+"\n")
        f.flush()
        start=time.monotonic()
        child=subprocess.Popen(args,cwd=WT,env=environment(gpu),
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        status["active"]={"stage":name,"pid":child.pid}
        (out/"N2_RECOVERY_STATUS.json").write_text(
           json.dumps(status,indent=2)+"\n")
        try:
            rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
            rc=124
        elapsed=time.monotonic()-start
    status["active"]=None
    status["stages"].append({"name":name,"rc":rc,"wall_seconds":elapsed,
                            "log":str(log)})
    (out/"N2_RECOVERY_STATUS.json").write_text(
           json.dumps(status,indent=2)+"\n")
    print("R0N_N2_RECOVERY_STAGE",name,"RC",rc,
          "WALL_SECONDS",round(elapsed,3),flush=True)
    return rc,elapsed


def preflight(args):
    if (args.gpu not in (0,1,2,3) or
        args.budget_seconds>RETRY_LIMIT_SECONDS or
        args.budget_seconds<1 or
        PRIOR_STAGE_SECONDS+args.budget_seconds>=900):
        raise RuntimeError("STOP_N2_RECOVERY_GPU_BUDGET")
    if (git("rev-parse","HEAD").stdout.strip()!=args.expected_head or
        git("status","--porcelain").stdout.strip()):
        raise RuntimeError("STOP_N2_RECOVERY_GIT_IDENTITY")
    if not git("merge-base","--is-ancestor",SOURCE_N0N1N3,
               args.expected_head,check=False).returncode==0:
        raise RuntimeError("STOP_N2_RECOVERY_N1N3_ANCESTRY")
    if not PY.is_file():raise RuntimeError("STOP_N2_RECOVERY_PY")
    tasks=PREP/"N2_GT_FREE_CANDIDATES.json"
    receipt=load(PREP/"N2_PREP_RECEIPT.json")
    payload=load(tasks)
    if (sha(tasks)!=PREP_SHA or
        receipt.get("status")!="R0N_N2_PREP_PASS" or
        receipt.get("tasks_sha256")!=PREP_SHA or
        receipt.get("executor_head")!=PREP_HEAD or
        receipt.get("n_candidates")!=3240 or
        payload.get("executor_head")!=PREP_HEAD or
        payload.get("count")!=3240 or
        payload.get("no_gt") is not True or
        len(payload["tasks"])!=3240):
        raise RuntimeError("STOP_N2_RECOVERY_PREP_SHA")
    allowed={"uid","source","family","image_id","run_index",
             "png_path","png_sha256"}
    if any(set(row)!=allowed or
           any(x in str(row["png_path"]).lower() for x in FORBIDDEN)
           for row in payload["tasks"]):
        raise RuntimeError("STOP_N2_RECOVERY_NOT_GT_FREE")
    if (not (WT/"docs/track_r/executors/r0n/returns/"
             "R0N_20261024T061111Z/R0N_LETTER_EVIDENCE.json").is_file()):
        raise RuntimeError("STOP_N2_RECOVERY_NO_N1_COMMIT")
    return tasks


def publish(out,status,args,score_root,eval_root):
    comparison=eval_root/"R0N_N2_CLIP_COMPARISON.json"
    receipt=load(eval_root/"R0N_N2_COMPLETE_RECEIPT.json")
    gpu=load(score_root/"N2_CLIP_GPU_RECEIPT.json")
    c=load(comparison)
    if (receipt["status"]!="R0N_N2_CLIP_OFFLINE_EVAL_PASS" or
        receipt["comparison_sha256"]!=sha(comparison) or
        c["total_n"]!=3240 or gpu["n"]!=3240 or
        gpu["gpu_seconds"]>args.budget_seconds or
        gpu["scored_sha256"]!=sha(score_root/"N2_GT_FREE_CLIP_SCORES.json")):
        raise RuntimeError("STOP_N2_RECOVERY_RECEIPT_INTEGRITY")
    # Write new addendum directory; never edit the published N1/N3 return.
    name="R0N_N2_RECOVERY_"+out.name.replace("r0n_n2_recovery_","")
    target=WT/"docs/track_r/executors/r0n/returns"/name
    target.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(comparison,target/"R0N_N2_CLIP_COMPARISON.json")
    shutil.copyfile(eval_root/"R0N_N2_COMPLETE_RECEIPT.json",
                    target/"R0N_N2_COMPLETE_RECEIPT.json")
    main=[r for r in c["rows"] if r["source"]=="ALL"]
    lines=["# R0-N N2 CLIP zero-shot orientation: final supplemental return",
           "",
           "R0N original N0/N1/N3 published at "+SOURCE_N0N1N3+".",
           "This addendum supplies only optional N2; original reports unchanged.",
           "","## Provenance",
           "- Frozen N2 source candidate manifest SHA256: "+PREP_SHA,
           "- Frozen original N2 preparation HEAD: "+PREP_HEAD,
           "- Corrected evaluator HEAD: "+args.expected_head,
           "- Pinned UnivFD CLIP source changed only in text-encoding wrapper;",
           "  the external repository and checkpoint were never changed.",
           "- First failed stage wall: 6.83 seconds (no candidates scored).",
           "- Recovery GPU stage wall (including imports): "+
             f"{status['stages'][0]['wall_seconds']:.3f} seconds.",
           "- Recovery instrumented GPU score wall: "+
             f"{gpu['gpu_seconds']:.3f} seconds.",
           "- Combined upper bound (stage wall): "+
             f"{PRIOR_STAGE_SECONDS+status['stages'][0]['wall_seconds']:.3f} seconds / 900.",
           "","## Frozen K2 comparison",
           "GLOBAL errors only; raw partial-orbit failures are recorded",
           "separately; all successes and harms use the precomputed R0.7 PSNR.",
           "","| Selector | GLOBAL errors | Repaired | GOOD harm |",
           "|---|---:|---:|---:|"]
    for r in main:
        success=r["global_repair_success"]
        harm=r["good_harm"]
        repaired=(round(success["estimate"]*r["n_global_orbit"])
                  if success["estimate"] is not None else 0)
        harmed=(round(harm["estimate"]*r["n_good"])
                if harm["estimate"] is not None else 0)
        lines.append(
          f"| {r['method']} | {r['n_global_orbit']} | "
          f"{repaired} | {harmed}/{r['n_good']} |")
    lines.extend([
        "","The frozen planner, not this executor, adjudicates K2:",
        "CLIP versus prior-global success, GOOD harm and relative compute cost.",
        "No claim of a new solver, R1 authorization or protected validation.",
        "","## Artifacts",
        "- R0N_N2_CLIP_COMPARISON.json: all source/family cluster CIs",
        "- R0N_N2_COMPLETE_RECEIPT.json: frozen offline-evaluation receipt",
        "- R0N_N2_RECOVERY_README.md: this return",
        ""])
    (target/"R0N_N2_RECOVERY_README.md").write_text(
        "\n".join(lines),encoding="utf-8")
    files=sorted(p for p in target.rglob("*") if p.is_file())
    sums="".join(f"{sha(p)}  {p.name}\n" for p in files)
    (target/"SHA256SUMS.txt").write_text(sums,encoding="utf-8")
    archive=out/"R0N_N2_RECOVERY_RETURN.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(target.iterdir()):
            z.write(p,arcname=p.name)
    git("add","--",str(target.relative_to(WT)))
    git("commit","-m","track-r R0N N2: publish frozen CLIP orientation comparator addendum")
    new_head=git("rev-parse","HEAD").stdout.strip()
    result=git("push","origin",f"HEAD:refs/heads/{BRANCH}",check=False,timeout=35)
    pushed=result.returncode==0
    if not pushed:
        print("R0N_N2_PUSH_PENDING",result.stderr[-450:],flush=True)
    status.update(
        status="R0N_N2_RECOVERY_COMPLETE_PLANNER_READY" if pushed else
               "R0N_N2_RECOVERY_COMPLETE_LOCAL_PUSH_PENDING",
        final_head=new_head,push_success=pushed,
        package_path=str(archive),package_sha256=sha(archive),
        planner_addendum_path=str(target),gpu_score_seconds=gpu["gpu_seconds"])
    return new_head


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--budget-seconds",type=float,default=870.0)
    args=p.parse_args()
    tasks=preflight(args)
    out=Path(args.out)
    if not str(out).startswith(str(BASE)+"/") or out.exists():
        raise RuntimeError("STOP_N2_RECOVERY_OUT_NAMESPACE")
    out.mkdir(parents=True,exist_ok=False)
    (out/"logs").mkdir()
    state={"status":"STARTED","source_head":args.expected_head,
      "original_N1N3_head":SOURCE_N0N1N3,
      "previous_N2_gpu_wall_seconds":PRIOR_STAGE_SECONDS,
      "new_stage_budget_seconds":args.budget_seconds,
      "n_reused_pngs":3240,
      "protected_cohort_access":False,"new_reconstructions":0,
      "active":None,"stages":[]}
    statefile=out/"N2_RECOVERY_STATUS.json"
    write_x(statefile,state)
    tag=out.name.replace("r0n_n2_recovery_","")
    score_root=BASE/("r0n_n2clip_recovery_"+tag)
    eval_root=BASE/("r0n_n2eval_recovery_"+tag)
    cmd=[str(PY),"-u","scripts/track_r/r0n_2_clip_orientation.py",
        "score","--expected-head",args.expected_head,
        "--reuse-prep-head",PREP_HEAD,"--tasks",str(tasks),
        "--tasks-sha",PREP_SHA,
        "--physical-gpu",str(args.gpu),
        "--budget-seconds",str(args.budget_seconds),
        "--out",str(score_root)]
    rc,wall=stage(out,statefile,state,"N2_CLIP_GPU_RETRY",cmd,
                  gpu=args.gpu,timeout=args.budget_seconds)
    if rc:
        state.update(status="STOP_R0N_N2_GPU_RETRY",
                     failure="N2 scorer RC "+str(rc))
        (out/"N2_RECOVERY_STATUS.json").write_text(
            json.dumps(state,indent=2)+"\n")
        return 8
    if PRIOR_STAGE_SECONDS+wall>900:
        raise RuntimeError("STOP_N2_RECOVERY_TOTAL_BUDGET")
    cmd=[str(PY),"-u","scripts/track_r/r0n_2_clip_orientation.py",
         "evaluate","--expected-head",args.expected_head,
         "--scores",str(score_root/"N2_GT_FREE_CLIP_SCORES.json"),
         "--out",str(eval_root)]
    rc,_=stage(out,statefile,state,"N2_OFFLINE_CPU_EVAL",cmd,timeout=2400)
    if rc:
        state.update(status="STOP_R0N_N2_OFFLINE_EVAL",
                     failure="N2 evaluator RC "+str(rc))
        (out/"N2_RECOVERY_STATUS.json").write_text(
            json.dumps(state,indent=2)+"\n")
        return 8
    head=publish(out,state,args,score_root,eval_root)
    (out/"N2_RECOVERY_STATUS.json").write_text(
        json.dumps(state,indent=2)+"\n")
    print("R0N_N2_RECOVERY_COMPLETE",state["status"],head,
          state["package_path"],state["package_sha256"],flush=True)
    return 0


if __name__=="__main__":
    try:sys.exit(main())
    except Exception as error:
        print("STOP_R0N_N2_RECOVERY",repr(error),flush=True)
        traceback.print_exc()
        sys.exit(22)
