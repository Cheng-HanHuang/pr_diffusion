#!/usr/bin/env python3
"""CPU-only A21.10 global terminal receipt publication.

Dedicated branch; no changes to frozen experiment checkout, no GPU use.
Receipts include occupancy union, worker hours, pilot, rates, yields, OOM,
cross-GPU replication, completed and skipped panels.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def git(root,*xs):return subprocess.check_output(
    ["git","-C",str(root),*xs],text=True,timeout=60).strip()
def gate(v,k):
    if not v:raise RuntimeError("STOP_A21_FINAL_RETURN_"+k)
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and git(WT,"rev-parse","HEAD")==a.expected_head
         and not git(WT,"status","--porcelain"),"PINNED_SOURCE")
    run=Path(a.run)
    gate(run.parent==BASE and
         run.name.startswith("xp_a25_xpc3_prod_a21_"),"RUN")
    final=run/"XP_C3_A21_FINAL_STATUS.json"
    gate(final.is_file(),"TERMINAL_RECEIPT")
    d=json.loads(final.read_text())
    gate(d["status"]!="A21_XPC3_RUNNING"
         and d["expected_head"]==a.expected_head
         and d["source_sha256"]==
         "298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27",
         "FROZEN_RECEIPT")
    cols=("pid","gpu","panel","mode","pilot","wall_seconds","rc","was_yielded")
    sanitized=dict(
        status=d["status"],expected_head=a.expected_head,
        source_sha256=d["source_sha256"],
        A21_freeze_sha256=d["a21_freeze_sha256"],
        original_R2_final_sha256=d["r2_final_sha256"],
        pilot=d["pilot"],
        equivalent_batch_passed=d.get("equivalence_pass"),
        production_mode=d.get("chosen_mode"),
        fallback_to_R2=d.get("fallback_to_original_R2"),
        measured_peak_mib=d.get("measured_peak_mib"),
        complete_panels=d.get("completed_panels",[]),
        skipped_panels=d.get("skipped_panels",[]),
        configurations=d.get("configurations",[]),
        gpu_occupancy_h_by_gpu=d.get("occupancy_gpu_h_by_gpu",{}),
        charged_cumulative_gpu_h=d.get("charged_gpu_h"),
        prior_R2_gpu_h=d.get("prior_r2_gpu_occupancy_h"),
        project_gpu_h=d.get("project_gpu_h"),
        worker_hours_info_only=d.get("worker_hours"),
        worker_records=[{k:v.get(k) for k in cols}
                        for v in d.get("worker_records",[])],
        yield_events=[dict(gpu=v.get("gpu"),reason=v.get("reason"))
            for v in d.get("yield_events",[])],
        oom_events=[dict(gpu=v.get("gpu"),mode=v.get("mode"),
                         panel=v.get("panel"))
            for v in d.get("oom_events",[])],
        rescored_chunks=d.get("rescored_chunks"),
        cross_gpu_replication=d.get("cross_gpu_replication",{}),
        replication_pending=d.get("replication_pending",[]),
        postprocess=d.get("postprocess",[]),
        error=d.get("error"),
        no_GT_in_scoring=True,no_matched_compute_claim=True,
        no_2forward_secondary=True,
        raw_internal_ledger_sha256=sha(final))
    tag=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    branch=f"codex/prdiff-track-r-a21-xpc3-final-{tag}"
    worktree=ROOT/f"pr_diffusion_track_r_a21_final_pub_{tag}"
    gate(not worktree.exists(),"NEW_PUBLICATION_TREE")
    subprocess.run(["git","-C",str(WT),"worktree","add","-b",
              branch,str(worktree),a.expected_head],check=True)
    out=worktree/"docs/track_r/executors/xp/returns"/f"XP_C3_A21_FINAL_{tag}"
    out.mkdir()
    file=out/"XP_C3_A21_FINAL_PUBLICATION_RECEIPT.json"
    file.write_text(json.dumps(sanitized,indent=2,sort_keys=True,
                       allow_nan=False)+"\n")
    subprocess.run(["git","-C",str(worktree),"add",
                str(out.relative_to(worktree))],check=True)
    subprocess.run(["git","-C",str(worktree),"commit","-m",
              "results(track-r): A21 terminal GPU occupancy, pilot and audit"],check=True)
    subprocess.run(["git","-C",str(worktree),"push","origin",
               "HEAD:refs/heads/"+branch],check=True,timeout=180)
    print("A21_FINAL_RECEIPT_PUBLISHED",branch,
         git(worktree,"rev-parse","HEAD"),
         "complete",len(sanitized["complete_panels"]),
         "gpu_occupancy_h",sanitized["charged_cumulative_gpu_h"],
         flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    try:main(p.parse_args())
    except Exception as exc:
        traceback.print_exc();print("ACTION NEEDED: A21_FINAL_PUBLISH",
             exc,flush=True);sys.exit(8)
