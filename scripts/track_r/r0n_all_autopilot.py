#!/usr/bin/env python3
"""Single PAC R0-N executor: N0, N1, optional N2, N3, and planner return."""
from __future__ import annotations
import argparse
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
CACHE=ROOT/"outputs/.cache/track_r"
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
BRANCH="codex/prdiff-track-r-r0n-20261023"
R0="c8eb60877fadeb1b9c7be63b0ff389f7681e7107"
PLANNER="6686322b919e891b5dd37d14ce5dddfa96a97dcd"
MERGE="1936e79acd4040f67e754276415bdac503b1121c"
ZIP=BASE/"r0_8_overnight_20261023T082909Z/R0_8_PLANNER_RETURN.zip"
N0=WT/"docs/track_r/executors/r0/returns/R0_20261023"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda:f.read(4*1024*1024),b""):h.update(x)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def atomic(path,obj):
    tmp=Path(str(path)+".partial")
    with tmp.open("w") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    tmp.replace(path)

def git(*parts):
    result=subprocess.run(["git","-C",str(WT),*parts],
        capture_output=True,text=True,timeout=120)
    if result.returncode:
        raise RuntimeError("STOP_R0N_GIT "+result.stdout[-300:]+result.stderr[-300:])
    return result.stdout.strip()

def clean():
    if git("status","--porcelain"):raise RuntimeError("STOP_R0N_DIRTY")

def environment(gpu=None):
    d=dict(os.environ)
    d.update({
        "CUDA_VISIBLE_DEVICES":"" if gpu is None else str(gpu),
        "PYTHONPATH":str(WT),"PYTHONDONTWRITEBYTECODE":"1",
        "PYTHONNOUSERSITE":"1",
        "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
        "XDG_CACHE_HOME":str(CACHE),"XDG_CONFIG_HOME":str(CACHE/"config"),
        "XDG_DATA_HOME":str(CACHE/"data"),"XDG_STATE_HOME":str(CACHE/"state"),
        "TORCH_HOME":str(CACHE/"torch"),"HF_HOME":str(CACHE/"hf"),
        "PIP_CACHE_DIR":str(CACHE/"pip"),
        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1"})
    for key in ("home","tmp","config","data","state","torch","hf","pip"):
        (CACHE/key).mkdir(parents=True,exist_ok=True)
    return d

def runstage(root,status,state,name,args,gpu=None,timeout=1200):
    log=root/"logs"/(name+".log")
    with log.open("x") as f:
        f.write("COMMAND "+repr(args)+"\n");f.flush()
        start=time.monotonic()
        proc=subprocess.Popen(args,cwd=WT,env=environment(gpu),
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state["active"]={"stage":name,"pid":proc.pid}
        atomic(status,state)
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            rc=124
        elapsed=time.monotonic()-start
    state["active"]=None
    state["stages"].append({
        "stage":name,"rc":rc,"elapsed_s":elapsed,"log":str(log)})
    atomic(status,state)
    print("R0N_STAGE",name,"RC",rc,"S",round(elapsed,2),flush=True)
    return rc

def stop(status,state,label,detail):
    state["status"]=label;state["detail"]=detail
    atomic(status,state)
    print(label,detail,flush=True)
    return 15

def commit(path,message):
    git("add","--",str(path))
    git("commit","-m",message)
    clean()
    new=git("rev-parse","HEAD")
    git("push","origin",f"HEAD:refs/heads/{BRANCH}")
    print("R0N_COMMITTED",new,flush=True)
    return new

def checked_copy(src,dst):
    src=Path(src);dst=Path(dst)
    if not src.is_file() or src.is_symlink() or dst.exists():
        raise RuntimeError("STOP_R0N_RETURN_SOURCE")
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    if sha(src)!=sha(dst):raise RuntimeError("STOP_R0N_RETURN_SHA")

def return_package(root,head,state,status):
    stamp=root.name.replace("r0n_all_","")
    directory=WT/"docs/track_r/executors/r0n/returns"/("R0N_"+stamp)
    directory.mkdir(parents=True,exist_ok=False)
    n1=Path(state["N1_run"])
    checked_copy(n1/"R0N_LETTER_EVIDENCE.json",
                 directory/"R0N_LETTER_EVIDENCE.json")
    checked_copy(n1/"R0N_REPORT.md",directory/"R0N_REPORT.md")
    for p in sorted(n1.glob("N1*.csv")):
        checked_copy(p,directory/p.name)
    for p in sorted((n1/"figures").iterdir()):
        if p.suffix not in (".pdf",".png"):
            raise RuntimeError("STOP_R0N_BAD_FIGURE")
        checked_copy(p,directory/"figures"/p.name)
    if state["N2_status"]=="PASS":
        n2=Path(state["N2_eval"])
        for name in ("R0N_N2_CLIP_COMPARISON.json","R0N_N2_COMPLETE_RECEIPT.json"):
            checked_copy(n2/name,directory/name)
    if state["N3_status"]=="PASS":
        checked_copy(Path(state["N3_run"])/"R0N_N3_INVENTORY.json",
                     directory/"R0N_N3_INVENTORY.json")
    ev=load(n1/"R0N_LETTER_EVIDENCE.json")
    text=[
      "# R0-N executor return to the Track R scientific planner",
      "",
      "Source R0 executor "+R0,
      "Planner decision "+PLANNER,
      "Merge commit "+MERGE,
      "R0N code "+head,
      "N0 original ZIP SHA "+sha(ZIP),
      "N0 commit "+state["N0_head"],
      "",
      "R0 methodology stopped; no R1 and no protected data access.",
      "K3 replay completed: 3240; GOOD 2040; ORBIT 85;",
      "consistent 324; inconsistent 791; repairs 84/85; GOOD harm 0/2040.",
      "",
      "## K1 B22 repair counts (planner decides)",
      "",
      "| Source | Family | K | n images | Post-selection repaired |",
      "|---|---|---:|---:|---:|"]
    for r in ev["K1_B22_postselection_fixes"]:
        text.append(f"| {r['source']} | {r['family']} | {r['K']} | {r['n_images']} | {r['n_fixed']} |")
    text.extend([
      "",
      "## N1d dev selection and B22 evaluation",
      "",
      json.dumps(ev["N1d"]["selected_on_DEV80"],indent=2),
      json.dumps(ev["N1d"]["selected_B22_evaluation"],indent=2),
      "",
      "## N2 CLIP comparator and K2 inputs",
      "Status: "+state["N2_status"],
      "N2 notes: "+state.get("N2_error","PASS"),
      ])
    if state["N2_status"]=="PASS":
        n2=load(directory/"R0N_N2_CLIP_COMPARISON.json")
        text.append(json.dumps([r for r in n2["rows"]
            if r["source"]=="ALL"],indent=2))
    text.extend(["",
      "## N3 bounded ImageNet asset inventory",
      "Status: "+state["N3_status"],
      "No ImageNet runs authorized.",
      "",
      "## Anomalies / deviations",
      "Optional environmental blockers are reported by status,",
      "not interpreted as failures of the scientific baseline.",
      "",
      "## PAC roots",
      "N1: "+state["N1_run"],
      "N2: "+state.get("N2_eval","unavailable"),
      "N3: "+state.get("N3_run","unavailable"),
      "",
      "K1 and K2 rulings reserved for the scientific planner.",
      "No R1 or protected confirmation305 access.",
      ""])
    (directory/"R0N_EXECUTOR_RETURN.md").write_text("\n".join(text))
    files=sorted(x for x in directory.rglob("*") if x.is_file())
    (directory/"SHA256SUMS.txt").write_text("".join(
        f"{sha(x)}  {x.relative_to(directory)}\n" for x in files))
    pack=root/"R0N_PLANNER_RETURN.zip"
    with zipfile.ZipFile(pack,"x",compression=zipfile.ZIP_DEFLATED,
                         compresslevel=7) as z:
        for p in sorted(x for x in directory.rglob("*") if x.is_file()):
            z.write(p,arcname=str(p.relative_to(directory)))
    state["package_path"]=str(pack)
    state["package_sha256"]=sha(pack)
    state["return_dir"]=str(directory)
    atomic(status,state)
    newhead=commit(directory.relative_to(WT),
        "track-r R0N: publish letter tables, optional orientation and asset inventory")
    state["final_head"]=newhead
    state["status"]=(
        "R0N_ALL_STAGES_COMPLETE_PLANNER_READY"
        if state["N2_status"]=="PASS" and state["N3_status"]=="PASS"
        else "R0N_PARTIAL_OPTIONALS_PLANNER_READY")
    atomic(status,state)
    print("R0N_PLANNER_READY",state["status"],newhead,str(pack),
          state["package_sha256"],flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--physical-gpu",type=int,default=1)
    a=p.parse_args()
    run=Path(a.out)
    if (not str(run).startswith(str(BASE)+"/") or run.exists() or
        a.physical_gpu not in range(4) or not PY.is_file()):
        raise RuntimeError("STOP_R0N_RUN_ENV")
    if git("rev-parse","HEAD")!=a.expected_head:raise RuntimeError("STOP_R0N_HEAD")
    clean()
    from scripts.track_r.r0n_0_import_tables_cpu import ZIP_SHA
    if sha(ZIP)!=ZIP_SHA:raise RuntimeError("STOP_R0N_ZIP_SHA")
    if N0.exists():raise RuntimeError("STOP_R0N_N0_ALREADY_EXISTS")
    run.mkdir(parents=True,exist_ok=False)
    (run/"logs").mkdir()
    status=run/"AUTOPILOT_STATUS.json"
    state={"schema":"prdiff.track_r.r0n.all.v1",
       "status":"RUNNING","initial_head":a.expected_head,
       "R0_parent":R0,"planner_parent":PLANNER,"merge_commit":MERGE,
       "N0_status":"PENDING","N1_status":"PENDING",
       "N2_status":"PENDING","N3_status":"PENDING",
       "active":None,"stages":[],"protected_access":False,
       "R1_runs_started":0}
    atomic(status,state)
    rc=runstage(run,status,state,"N0_import",[
        str(PY),"-u","scripts/track_r/r0n_0_import_tables_cpu.py",
        "--zip",str(ZIP),"--expected-head",a.expected_head],timeout=600)
    if rc:return stop(status,state,"STOP_R0N_N0","N0 importer RC="+str(rc))
    verify=subprocess.run(["sha256sum","-c","SHA256SUMS.txt"],
        cwd=N0,capture_output=True,text=True,timeout=60)
    if verify.returncode:
        return stop(status,state,"STOP_R0N_N0_SHA",
                    verify.stdout+verify.stderr)
    n0head=commit(N0.relative_to(WT),
        "track-r R0N N0: commit original SHA-verified R0.8 aggregate tables")
    state.update(N0_status="PASS",N0_head=n0head)
    atomic(status,state)
    rc=runstage(run,status,state,"N1_tests",[
        str(PY),"-m","pytest","-q","-p","no:cacheprovider",
        "tests/track_r"],timeout=300)
    if rc:return stop(status,state,"STOP_R0N_TESTS",str(rc))
    stem=run.name.replace("r0n_all_","")
    n1=BASE/("r0n_n1_"+stem)
    rc=runstage(run,status,state,"N1_CPU",[
       str(PY),"-u","scripts/track_r/r0n_1_letter_tables_cpu.py",
       "--expected-head",n0head,"--out",str(n1)],timeout=3600)
    if rc:return stop(status,state,"STOP_R0N_K3_OR_N1",str(rc))
    receipt=load(n1/"R0N_N1_COMPLETE_RECEIPT.json")
    if (receipt.get("status")!="R0N_N1_CPU_EVIDENCE_PASS" or
        receipt.get("rows")!=3240 or receipt.get("n_configurations")!=60 or
        receipt.get("evidence_sha256")!=sha(n1/"R0N_LETTER_EVIDENCE.json")):
        return stop(status,state,"STOP_R0N_N1_RECEIPT","MISMATCH")
    state.update(N1_status="PASS",N1_run=str(n1));atomic(status,state)
    prep=BASE/("r0n_n2prep_"+stem)
    scorer=BASE/("r0n_n2clip_"+stem)
    evaluation=BASE/("r0n_n2eval_"+stem)
    rc=runstage(run,status,state,"N2_prep",[
       str(PY),"-u","scripts/track_r/r0n_2_clip_orientation.py",
       "prepare","--expected-head",n0head,"--out",str(prep)],
       timeout=1200)
    if rc:
        state.update(N2_status="UNAVAILABLE_PREP",N2_error="Prep RC "+str(rc))
    else:
        tasks=prep/"N2_GT_FREE_CANDIDATES.json"
        prep_receipt=load(prep/"N2_PREP_RECEIPT.json")
        if prep_receipt.get("tasks_sha256")!=sha(tasks):
            state.update(N2_status="UNAVAILABLE_TASK_SHA",
                         N2_error="GT-free manifest SHA mismatch")
        else:
            rc=runstage(run,status,state,"N2_CLIP_GPU",[
                str(PY),"-u","scripts/track_r/r0n_2_clip_orientation.py",
                "score","--expected-head",n0head,
                "--tasks",str(tasks),
                "--tasks-sha",prep_receipt["tasks_sha256"],
                "--physical-gpu",str(a.physical_gpu),
                "--budget-seconds","900","--out",str(scorer)],
                gpu=a.physical_gpu,timeout=900)
            if rc:
                state.update(N2_status="UNAVAILABLE_GPU",
                             N2_error="CLIP stage RC "+str(rc))
            else:
                scores=scorer/"N2_GT_FREE_CLIP_SCORES.json"
                got=load(scorer/"N2_CLIP_GPU_RECEIPT.json")
                if (got.get("status")!="R0N_N2_CLIP_GPU_PASS" or
                    got.get("gpu_seconds",901)>900 or
                    got.get("scored_sha256")!=sha(scores)):
                    state.update(N2_status="UNAVAILABLE_GPU_RECEIPT",
                                 N2_error="GPU budget or SHA mismatch")
                else:
                    rc=runstage(run,status,state,"N2_CPU_EVAL",[
                      str(PY),"-u","scripts/track_r/r0n_2_clip_orientation.py",
                      "evaluate","--expected-head",n0head,
                      "--scores",str(scores),"--out",str(evaluation)],
                      timeout=1800)
                    if rc:
                        state.update(N2_status="UNAVAILABLE_EVALUATION",
                                     N2_error="N2 evaluation RC "+str(rc))
                    else:
                        r=load(evaluation/"R0N_N2_COMPLETE_RECEIPT.json")
                        if r.get("status")!="R0N_N2_CLIP_OFFLINE_EVAL_PASS":
                            state.update(N2_status="UNAVAILABLE_EVAL_RECEIPT")
                        else:
                            state.update(N2_status="PASS",
                                         N2_eval=str(evaluation),
                                         N2_gpu_seconds=got["gpu_seconds"])
    atomic(status,state)
    n3=BASE/("r0n_n3_"+stem)
    rc=runstage(run,status,state,"N3_CPU_INVENTORY",[
        str(PY),"-u","scripts/track_r/r0n_3_imagenet_inventory_cpu.py",
        "--expected-head",n0head,"--out",str(n3)],timeout=180)
    if rc:state.update(N3_status="UNAVAILABLE",N3_error="N3 RC "+str(rc))
    elif load(n3/"R0N_N3_INVENTORY.json").get("status")!="R0N_N3_BOUNDED_INVENTORY_PASS":
        state.update(N3_status="UNAVAILABLE_RECEIPT")
    else:state.update(N3_status="PASS",N3_run=str(n3))
    atomic(status,state)
    clean()
    return_package(run,a.expected_head,state,status)
    return 0

if __name__=="__main__":
    try:rc=main()
    except Exception as e:
        print("STOP_R0N_ALL_EXCEPTION",repr(e),flush=True)
        traceback.print_exc()
        sys.exit(21)
    sys.exit(rc)
