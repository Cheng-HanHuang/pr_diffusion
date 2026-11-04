#!/usr/bin/env python3
"""A36 isolated CPU-only UnivFD repair for the A35 G7 registered GT test.

The *original* A35 UnivFD worker, model, orientation score, 280 frozen GT
inputs and A35 readout/publication routines remain unchanged. A36 only:
(1) isolates official setuptools==80.9.0 pkg_resources from shared Conda;
(2) avoids re-running accepted S_mid all8 / 2-forward GPU experiments;
(3) saves per-image CPU-only progress with a resumable, separate stage;
(4) invokes the frozen A35 readout/publisher after verifying 280 receipts.

No GPU allocation, no new model, no inverse-problem reconstruction, no GT in
training/selection. The source is diagnostic ground truth by A23 prereg.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
PY=ROOT/"conda-envs/daps/bin/python"
GT=BASE/"xp_c_c2_a22_20261104T025355Z/XP_C_GT_IMAGES.json"
SOURCE_LOCK=WT/"docs/track_r/executors/xp/theory_inputs/XP_A23_G7_REMAINING_SOURCE_LOCK.json"
A35_ORIGINAL=BASE/"xp_a35_g7_remaining_20261104T083110Z"
G7_TOTAL=5.971352510419156
PROJECT_TOTAL=247.80191737622505
UNIVFD="UnivFD"
LAST_TWO={
    "S_mid_all8":(
        ROOT/"pr_diffusion_track_r_a35_pub_S_mid_all8_20261104T084401Z",
        "967e73cc7adcc5cc3347d257572bd779ff0a6dea"),
    "2_forward":(
        ROOT/"pr_diffusion_track_r_a35_pub_2_forward_20261104T084622Z",
        "6d2afbfd8c405cbf7db4a6f193ba21c885c7eaa4"),
}
VENDOR=BASE/"a36_univfd_vendor_setuptools_80_9_0"
VERSION="80.9.0"
PROJECT_LOCK=BASE/"XP_A36_UNIVFD_RECOVERY_CPU.lock"
WORKER=WT/"scripts/track_r/xp_a35_g7_worker.py"
EXPECTED_ORIGINAL_HEAD="e5852aedc272aa3c5546b39d3b2f8b7a224ba9af"


def gate(test,label):
    if not test:
        raise RuntimeError("STOP_A36_UNIVFD_"+label)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def git(path,*args):
    return subprocess.check_output(
        ["git","-C",str(path),*args],text=True,timeout=60).strip()


def write(path,data):
    path=Path(path)
    temp=path.with_name("."+path.name+".tmp")
    with temp.open("x") as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def check_accepted_history():
    gate(git(WT,"rev-parse","HEAD")==HEAD and
         not git(WT,"status","--porcelain"),"CLEAN_SOURCE_HEAD")
    spec=load(SOURCE_LOCK)
    gate(spec["status"]=="A23_G7_REMAINING_FROZEN_BEFORE_PRODUCTION"
         and spec["methods"]==["S_mid_all8","2_forward","UnivFD"]
         and sha(GT)==spec["GT_manifest_sha256"],
         "SOURCE_LOCK_AND_280_GT")
    gt=load(GT)
    gate(gt["n_images"]==280 and gt["n_ffhq"]==180
         and gt["n_imagenet"]==100
         and len(gt["images"])==280,
         "EXACT_ORIGINAL_280_GRID")
    for method,(worktree,commit) in LAST_TWO.items():
        gate(worktree.is_dir()
             and git(worktree,"rev-parse","HEAD")==commit
             and not git(worktree,"status","--porcelain"),
             "ACCEPTED_PUBLISH_"+method)
    fail_stage=load(A35_ORIGINAL/UNIVFD/"STAGE_TERMINAL.json")
    gate(fail_stage["status"]=="STOP_A35_NATIVE_OR_CONTROLLER_FAILURE"
         and fail_stage["n_complete"]==0
         and float(fail_stage["method_gpu_h"])==0
         and abs(fail_stage["G7_total_gpu_h"]-G7_TOTAL)<1e-8,
         "HISTORICAL_CPU_FAILURE_PRESERVED")
    gate(not any((A35_ORIGINAL/UNIVFD/"results").glob("*.json")),
         "HISTORICAL_CPU_HAS_RESULTS")
    # The previous two stages already updated the global project ledger.
    gate(G7_TOTAL<10 and PROJECT_TOTAL<400,
         "PUBLISHED_G7_AND_GLOBAL_BUDGET")
    return spec,gt["images"]


def check_vendor_python():
    module=VENDOR/"pkg_resources"/"__init__.py"
    gate(module.is_file() and not module.is_symlink(),
         "PINNED_PKG_RESOURCES_MODULE_MISSING")
    code=(
        "import importlib.metadata as md, pathlib, pkg_resources\n"
        "from pkg_resources import packaging\n"
        "v=md.version('setuptools')\n"
        "assert v=='80.9.0',v\n"
        "assert pathlib.Path(pkg_resources.__file__).resolve().is_relative_to("
        +repr(str(VENDOR)) + ")\n"
        "assert str(packaging.version.parse('1.2.3'))=='1.2.3'\n"
        "print('A36_PKG_RESOURCES_VENDOR_PASS',v,pkg_resources.__file__)\n"
    )
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",
             PYTHONPATH=f"{VENDOR}:{WT}",
             PYTHONNOUSERSITE="1",
             PYTHONDONTWRITEBYTECODE="1",
             PYTHONPYCACHEPREFIX=str(BASE/".pycache_a36_vendor_check"),
             OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
             OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
    test=subprocess.run([str(PY),"-W","ignore::UserWarning","-c",code],
        env=env,cwd=str(WT),capture_output=True,text=True,timeout=90)
    print(test.stdout[-900:],flush=True)
    gate(test.returncode==0,"PKG_RESOURCES_IMPORT_"+test.stderr[-400:])
    return sha(module)


def valid_partial(stage,images):
    from scripts.track_r.xp_a35_g7_worker import STATUS
    done=set()
    for path in (stage/"results").glob("*.json"):
        gate(path.stem.isdecimal() and 0<=int(path.stem)<280,
             "RESUME_RESULT_NAME")
        i=int(path.stem)
        gate(i not in done,"RESUME_DUPLICATE_INDEX")
        r=load(path);img=images[i]
        gate(r["status"]==STATUS and r["method"]==UNIVFD
             and r["index"]==i
             and r["dataset"]==img["dataset"]
             and r["basename"]==img["basename"]
             and r["source_sha256"]==img["source_sha256"]
             and r["source_manifest_sha256"]==sha(GT)
             and r["source_lock_sha256"]==sha(SOURCE_LOCK)
             and r["implementation_head"]==HEAD
             and r["diagnostic_GT_only"] is True
             and r["GT_not_used_for_inverse_problem_generation"] is True
             and r["physical_gpu"] is None
             and r["n_evals"]==8
             and r["score"]["canonical_orbit_batch"]==8
             and r["choice"]==r["score"]["best_orbit"]
             and r["upright_kept"]==(r["choice"]=="000"),
             "RESUME_RESULT_INVALID_"+str(i))
        done.add(i)
    return done


def run_group(run,images,dataset,indices,tag,head,vendor_sha):
    if not indices:
        return
    stage=run/UNIVFD
    task=stage/"tasks"/(tag+".json")
    gate(not task.exists(),"NO_DUPLICATE_TASK")
    task.write_text(json.dumps(indices,indent=2)+"\n")
    log=stage/"logs"/(tag+".log")
    gate(not log.exists(),"NO_DUPLICATE_LOG")
    # Same A35 worker, in original CPU mode; only environment
    # gained the official pinned pkg_resources provider.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=f"{VENDOR}:{WT}",
        PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(stage/"pycache"/tag),
        OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
    args=[str(PY),"-u",str(WORKER),
        "--expected-head",head,
        "--run",str(run),"--method",UNIVFD,
        "--dataset",dataset,"--tasks",str(task)]
    start=time.monotonic()
    print("A36_UNIVFD_GROUP_START",tag,"n",len(indices),
          "CPU_ONLY",flush=True)
    with log.open("x") as f:
        child=subprocess.run(args,cwd=str(WT),env=env,
            stdout=f,stderr=subprocess.STDOUT,
            timeout=21600)
    elapsed=time.monotonic()-start
    done=valid_partial(stage,images)
    complete=set(indices).issubset(done)
    print("A36_UNIVFD_GROUP_TERMINAL",tag,"rc",child.returncode,
          "received",len(done),"elapsed_s",round(elapsed,1),flush=True)
    gate(child.returncode==0 and complete,
         "CPU_WORKER_FAIL_"+tag+"_LOG_"+str(log))
    return dict(group=tag,n_requested=len(indices),
        elapsed_wall_s=elapsed,pkg_resources_vendor_module_sha256=vendor_sha,
        cpu_only=True,group_log=str(log))


def existing_local_publication(readout):
    """Reuse a previously created exact result commit after a push timeout.

    Prevent a second publication commit for the same frozen CPU results.
    """
    roots=sorted(ROOT.glob("pr_diffusion_track_r_a35_pub_UnivFD_*"))
    if not roots:return None
    gate(len(roots)==1, "MULTIPLE_LOCAL_PUBLICATION_TREES")
    tree=roots[0]
    gate(tree.is_dir() and not tree.is_symlink()
         and not git(tree,"status","--porcelain"),
         "LOCAL_PUBLICATION_TREE_DIRTY")
    branch=git(tree,"branch","--show-current")
    gate(branch.startswith("codex/prdiff-track-r-a23-g7-univfd-"),
         "LOCAL_PUBLICATION_BRANCH")
    base=tree/"docs/track_r/executors/xp/returns"
    pubs=list(base.glob("XP_A35_G7_UnivFD_*/XP_A35_G7_PUBLICATION.json"))
    gate(len(pubs)==1,"LOCAL_PUBLICATION_RECEIPT")
    published=load(pubs[0])
    gate(published["status"]=="A35_G7_STAGE_280_PUBLISHED"
         and published["method"]=="UnivFD"
         and published["original_scoring_head"]==HEAD
         and published["per_image_sha256"]==readout["per_image_sha256"]
         and published["table_sha256"]==readout["aggregate_sha256"],
         "EXISTING_LOCAL_PUBLICATION_CONTENT")
    commit=git(tree,"rev-parse","HEAD")
    for attempt in (1,2):
        # Query remote authoritatively; never assume a timed-out push failed.
        listing=subprocess.check_output(
            ["git","-C",str(tree),"ls-remote","--heads","origin",
             "refs/heads/"+branch],text=True,timeout=90).strip()
        remote_sha=listing.split()[0] if listing else None
        if remote_sha==commit:break
        gate(remote_sha is None,"LOCAL_PUBLISH_REMOTE_DIFFERENT")
        cmd=["git","-C",str(tree)]
        if attempt==2:cmd+=["-c","http.version=HTTP/1.1"]
        cmd+=["push","origin","HEAD:refs/heads/"+branch]
        try:
            result=subprocess.run(cmd,timeout=300,
                capture_output=True,text=True,
                env=dict(os.environ,GIT_TERMINAL_PROMPT="0",
                         GCM_INTERACTIVE="never"))
            print("A36_PUSH_EXISTING_COMMIT_ATTEMPT",attempt,result.returncode,
                  result.stderr[-400:],flush=True)
        except subprocess.TimeoutExpired:
            print("A36_PUSH_EXISTING_COMMIT_TIMEOUT",attempt,flush=True)
    listing=subprocess.check_output(
        ["git","-C",str(tree),"ls-remote","--heads","origin",
         "refs/heads/"+branch],text=True,timeout=90).strip()
    gate(listing and listing.split()[0]==commit,
         "LOCAL_PUBLICATION_PUSH_NOT_CONFIRMED")
    print("A36_UNIVFD_LOCAL_PUBLICATION_REUSE",branch,commit,flush=True)
    return dict(branch=branch,commit=commit)


def main(a):
    global HEAD
    HEAD=a.expected_head
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and Path(sys.executable).resolve()==PY.resolve()
         and os.environ.get("PYTHONNOUSERSITE")=="1",
         "STRICT_CPU_NO_CUDA_ENV")
    spec,images=check_accepted_history()
    module_sha=check_vendor_python()
    run=Path(a.run)
    gate(run.parent==BASE
         and run.name.startswith("xp_a35_g7_remaining_a36_univfd_")
         and not run.is_symlink(),"INDEPENDENT_RECOVERY_NAMESPACE")
    with PROJECT_LOCK.open("a+") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage=run/UNIVFD
        if not run.exists():
            gate(not a.resume,"NO_RESUME_OF_NEW_RUN")
            run.mkdir(mode=0o700)
            stage.mkdir()
            for d in ("results","tasks","logs","pycache"):
                (stage/d).mkdir()
            initial=dict(status="A36_UNIVFD_CPU_RECOVERY_IN_PROGRESS",
                code_head=HEAD,
                source_lock_sha256=sha(SOURCE_LOCK),
                source_GT_manifest_sha256=sha(GT),
                original_A35_run=str(A35_ORIGINAL),
                original_A35_failure_preserved=True,
                pkg_resources_provider="official_setuptools_80.9.0_vendor",
                pkg_resources_module_sha256=module_sha,
                two_forward_published_commit=LAST_TWO["2_forward"][1],
                S_mid_all8_published_commit=LAST_TWO["S_mid_all8"][1],
                previous_G7_cumulative_gpu_h=G7_TOTAL,
                previous_project_cumulative_gpu_h=PROJECT_TOTAL,
                univfd_GPU_h=0,CPU_only=True,counts_completed=0,
                group_receipts=[])
            write(run/"A36_UNIVFD_RECOVERY_STATE.json",initial)
        else:
            gate(a.resume and stage.is_dir()
                 and (run/"A36_UNIVFD_RECOVERY_STATE.json").is_file(),
                 "EXPLICIT_RESUME_REQUIRED")
            initial=load(run/"A36_UNIVFD_RECOVERY_STATE.json")
            gate(initial["code_head"]==HEAD
                 and initial["source_GT_manifest_sha256"]==sha(GT)
                 and initial["source_lock_sha256"]==sha(SOURCE_LOCK)
                 and initial["pkg_resources_module_sha256"]==module_sha
                 and initial["original_A35_failure_preserved"] is True,
                 "EXACT_RESUME_PROVENANCE")
        done=valid_partial(stage,images)
        print("A36_UNIVFD_RECOVERY_EXISTING_ACCEPTED",len(done),flush=True)
        # Each dataset has a first-image canary. Using the exact original
        # A35 worker means subsequent image receipts have identical science.
        groups=(("ffhq",[0],"ffhq_first"),
                ("ffhq",list(range(1,180)),"ffhq_remainder"),
                ("imagenet",[180],"imagenet_first"),
                ("imagenet",list(range(181,280)),"imagenet_remainder"))
        for dataset,inds,tag in groups:
            pending=[i for i in inds if i not in done]
            if not pending:continue
            seq=len(list((stage/"logs").glob("*.log")))
            receipt=run_group(run,images,dataset,pending,tag+f"_{seq:02d}",HEAD,module_sha)
            done=valid_partial(stage,images)
            initial["counts_completed"]=len(done)
            initial["group_receipts"].append(receipt)
            write(run/"A36_UNIVFD_RECOVERY_STATE.json",initial)
        gate(len(done)==280,"CPU_280_NOT_COMPLETE")
        from scripts.track_r.xp_a35_g7_readout_publish_cpu import (
            analyze,publish,check_results)
        gate(len(check_results(stage,UNIVFD,HEAD))==280,
             "ORIGINAL_A35_CHECK_RESULTS_GATE")
        term=dict(
            status="A35_G7_STAGE_280_COMPLETE",method=UNIVFD,
            n_complete=280,expected=280,code_head=HEAD,
            source_lock_sha256=sha(SOURCE_LOCK),
            GT_manifest_sha256=sha(GT),
            recovery_of_original_A35_failure=str(A35_ORIGINAL),
            original_error="ModuleNotFoundError: No module named 'pkg_resources'",
            no_change_to_model_or_score=True,
            official_vendor_setuptools_version=VERSION,
            official_vendor_pkg_resources_sha256=module_sha,
            stage_method_cap_gpu_h=0,
            method_gpu_h=0,
            G7_total_gpu_h=G7_TOTAL,
            project_gpu_h=PROJECT_TOTAL,
            GT_diagnostic_only=True,CPU_only=True,
            no_inverse_solver=True,no_matched_compute=True,
            no_A21_10=True,
            original_S_mid_all8_commit=LAST_TWO["S_mid_all8"][1],
            original_2_forward_commit=LAST_TWO["2_forward"][1])
        stage_term=stage/"STAGE_TERMINAL.json"
        if not stage_term.exists():
            write(stage_term,term)
        else:
            gate(load(stage_term)==term,"TERMINAL_ALREADY_DIFFERS")
        costs=dict(G7_total_gpu_h=G7_TOTAL,
                   project_gpu_h=PROJECT_TOTAL)
        readout_path=stage/"readout"/"XP_A35_G7_METHOD_READOUT_RECEIPT.json"
        if not readout_path.exists():
            readout=analyze(run,UNIVFD,HEAD,costs)
        else:
            readout=load(readout_path)
            gate(readout["head"]==HEAD and
                 readout["status"]=="A35_G7_METHOD_READOUT_280_PASS",
                 "READOUT_ALREADY_DIFFERENT")
        print("A36_UNIVFD_280_SCIENCE_AND_READOUT_PASS",flush=True)
        existing=initial.get("publication")
        if existing:
            print("A36_UNIVFD_PREVIOUS_PUBLICATION",existing,flush=True)
        else:
            pub=existing_local_publication(readout)
            if pub is None:
                pub=publish(run,UNIVFD,HEAD,readout)
            initial["publication"]=pub
            initial["status"]="A36_UNIVFD_CPU_280_PUBLISHED"
            initial["counts_completed"]=280
            write(run/"A36_UNIVFD_RECOVERY_STATE.json",initial)
            print("A36_UNIVFD_CPU_RECOVERY_COMPLETE_AND_PUBLISHED",
                pub["branch"],pub["commit"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    try:main(a)
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A36_UNIVFD_CPU_ONLY",repr(e),flush=True)
        sys.exit(8)
