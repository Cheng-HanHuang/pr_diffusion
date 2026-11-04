#!/usr/bin/env python3
"""A23 S_mid global2 GT-only publication, separate branch from immutable code.

Create a single local result commit from verified 280-image frozen receipts.
Push same commit with bounded remote checks/retries; never re-score and never
claim that later G7 policies are finished. If publication fails, STOP with
the exact existing local worktree/branch/commit printed for separate repair.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_a34_a23_g7_smid_controller_cpu import (
    ROOT,BASE,WT,LOCK,FREEZE,SOURCE_SHA,git,sha,jsonread,fail,
    N,valid_result,
)

def run_git(root,*args,timeout=90):
    return subprocess.check_output(
        ["git","-C",str(root),*args],text=True,timeout=timeout).strip()

def rows(path,n):
    with Path(path).open(newline="") as f:
        rr=list(csv.DictReader(f))
    fail(len(rr)==n,"PUBLISH_CSV_GRID_"+str(path))
    return rr

def remote_head(root,branch):
    out=run_git(root,"ls-remote","--heads","origin",
                "refs/heads/"+branch,timeout=90)
    if not out:return None
    a=out.split()
    fail(len(a)==2 and a[1]=="refs/heads/"+branch,
         "MALFORMED_REMOTE_BRANCH")
    return a[0]

def push_same_commit(tree,branch,commit):
    for attempt in (1,2):
        remote=remote_head(tree,branch)
        if remote==commit:return
        fail(remote is None,"DIFFERENT_REMOTE_REF")
        cmd=["git","-C",str(tree)]
        if attempt==2:cmd+=["-c","http.version=HTTP/1.1"]
        cmd+=["push","origin","HEAD:refs/heads/"+branch]
        try:
            completed=subprocess.run(cmd,cwd=str(tree),
                env=dict(os.environ,GIT_TERMINAL_PROMPT="0",GCM_INTERACTIVE="never"),
                capture_output=True,text=True,timeout=300)
            print("A34_G7_PUSH_ATTEMPT",attempt,
                  "rc",completed.returncode,
                  "stderr",completed.stderr[-400:],flush=True)
        except subprocess.TimeoutExpired:
            print("A34_G7_PUSH_ATTEMPT_TIMEOUT",attempt,flush=True)
        remote=remote_head(tree,branch)
        if remote==commit:return
        fail(remote is None,"DIFFERENT_REMOTE_AFTER_PUSH")
    raise RuntimeError(
        "A34_G7_PUBLICATION_PUSH_NOT_CONFIRMED_LOCAL_COMMIT_PRESERVED")

def main(a):
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and git("rev-parse","HEAD")==a.expected_head
         and not git("status","--porcelain"),
         "PUBLISH_CLEAN_CODE")
    run=Path(a.run)
    fail(run.parent==BASE and run.name.startswith(
         "xp_a34_a23_g7_smid_prod_"),"PUBLISH_RUN")
    status=run/"XP_A34_G7_S_MID_FINAL_STATUS.json"
    readout=run/"readout"/"XP_A23_G7_S_MID_GLOBAL2_READOUT_RECEIPT.json"
    st=jsonread(status);r=jsonread(readout)
    fail(st["status"]=="A34_G7_S_MID_GLOBAL2_280_COMPLETE"
         and st["n_complete"]==280
         and st["head"]==a.expected_head
         and r["status"]=="A34_G7_S_MID_GLOBAL2_GT_ONLY_READOUT_PASS"
         and r["head"]==a.expected_head
         and r["n_images"]==280 and r["n_pilot_images_reused"]==2
         and r["source_native_controller_state_sha256"]==sha(status)
         and r["original_G7_C_source_sha256"]==SOURCE_SHA,
         "PUBLISH_FROZEN_STAGES")
    per=run/"readout"/"XP_A23_G7_S_MID_GLOBAL2_GT_PER_IMAGE.csv"
    agg=run/"readout"/"XP_A23_G7_S_MID_GLOBAL2_GT_AGGREGATE.csv"
    fail(sha(per)==r["per_image_sha256"] and
         sha(agg)==r["aggregate_sha256"],
         "PUBLISH_CSV_SHA")
    perrows=rows(per,280)
    aggrows=rows(agg,3)
    fail(len({(x["dataset"],x["basename"]) for x in perrows})==280
         and {x["dataset"] for x in aggrows}=={"ffhq","imagenet","pooled"}
         and all(valid_result(
            run/"results"/f"{i:03d}.json",i) for i in range(N)),
         "PUBLISH_ALL_280_NATIVE_RECEIPTS")
    tag=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    branch=f"codex/prdiff-track-r-a23-g7-smid-global2-{tag}"
    tree=ROOT/f"pr_diffusion_track_r_a23_g7_smid_pub_{tag}"
    fail(not tree.exists(),"PUBLISH_NEW_TREE")
    subprocess.run(["git","-C",str(WT),"worktree","add",
        "-b",branch,str(tree),a.expected_head],check=True,timeout=120)
    out=tree/"docs/track_r/executors/xp/returns"/f"XP_A23_G7_S_MID_GLOBAL2_{tag}"
    ti=tree/"docs/track_r/executors/xp/theory_inputs"
    out.mkdir()
    for source in (per,agg,status,readout,LOCK):
        shutil.copyfile(source,out/source.name)
    for source in (per,agg):
        shutil.copyfile(source,ti/source.name)
    receipt=dict(
        status="A23_G7_S_MID_GLOBAL2_GT_280_ONLY_PUBLISHED",
        not_full_G7_completion=True,not_main_paper_data=True,
        stage_name="S_mid_global2",n_images=280,n_ffhq=180,n_imagenet=100,
        accepted_pilot_image_reuse=2,
        source_GT_manifest_sha256=SOURCE_SHA,
        source_lock_sha256=sha(LOCK),
        source_accepted_C2_CLIP_publication="9090a50a4d4ffe922bdb43c5ee9e877c059a7dd1",
        G7_total_gpu_h=st["G7_total_charged_gpu_h"],
        project_total_gpu_h=st["project_charged_gpu_h"],
        per_image_csv_sha256=sha(per),aggregate_csv_sha256=sha(agg),
        controller_final_status_sha256=sha(status),
        CPU_readout_receipt_sha256=sha(readout),
        remaining_G7_policies_not_claimed=["S_mid_all8","2_forward","UnivFD"],
        XP_K_not_run=True,no_A21_10_replication=True,
        GT_only_diagnostics=True,no_inverse_solver_generation=True,
        not_matched_compute=True,
        original_scoring_head=a.expected_head)
    (out/"XP_A23_G7_S_MID_GLOBAL2_PUBLICATION.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    subprocess.run(["git","-C",str(tree),"add",
        str(out.relative_to(tree)),
        str((ti/per.name).relative_to(tree)),
        str((ti/agg.name).relative_to(tree))],check=True,timeout=90)
    subprocess.run(["git","-C",str(tree),"commit","-m",
        "results(track-r): A23 G7 S_mid global2 GT-only upright retention 280"],check=True,timeout=90)
    commit=run_git(tree,"rev-parse","HEAD")
    print("A34_G7_LOCAL_PUBLICATION_COMMITTED",branch,commit,
          "local_tree",tree,flush=True)
    push_same_commit(tree,branch,commit)
    fail(remote_head(tree,branch)==commit,"REMOTE_EXACT_COMMIT")
    print("A34_G7_S_MID_GLOBAL2_REMOTE_PUBLISHED",branch,commit,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A34_G7_PUBLISH",repr(e),flush=True)
        sys.exit(8)
