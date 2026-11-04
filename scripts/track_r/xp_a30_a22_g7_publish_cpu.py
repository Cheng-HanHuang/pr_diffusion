#!/usr/bin/env python3
"""Publish only complete G7 C2 + CLIP diagnostics, never full G7 claim."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
def gate(ok,k):
 if not ok:raise RuntimeError("STOP_A22_G7_PUBLISH_"+k)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for part in iter(lambda:f.read(1048576),b""):h.update(part)
 return h.hexdigest()
def git(root,*args):return subprocess.check_output(
 ["git","-C",str(root),*args],text=True,timeout=60).strip()
def checkcsv(p,n):
 with Path(p).open(newline="") as f:rows=list(csv.DictReader(f))
 gate(len(rows)==n,"CSV_GRID_"+str(p.name))
 return rows
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
      and git(WT,"rev-parse","HEAD")==a.expected_head
      and not git(WT,"status","--porcelain"),"CLEAN_CODE")
 run,report,clip=(Path(x) for x in (a.run,a.report,a.clip))
 gate(run.parent==report.parent==clip.parent==BASE
      and run.name.startswith("xp_a12_g7_prod_a22_")
      and report.name.startswith("xp_a30_a22_g7_report_")
      and clip.name.startswith("xp_a12_g7_clip_a22_"),"ROOT_NAMES")
 r=run/"XP_A22_G7_C2_FINAL_STATUS.json"
 c=clip/"XP_A12_G7_C_CLIP_RECEIPT.json"
 rr=report/"XP_A22_G7_C2_READOUT_RECEIPT.json"
 gate(all(p.is_file() for p in (r,c,rr)),"RECEIPTS")
 state=json.loads(r.read_text())
 cr=json.loads(c.read_text())
 tr=json.loads(rr.read_text())
 gate(state["status"]=="A22_G7_C2_280_COMPLETE"
      and len(state["completed"])==280
      and tr["status"]=="A22_G7_C2_IMAGE_BOOTSTRAP_CPU_COMPLETE"
      and tr["n_summary_rows"]==36
      and tr["n_input_images"]==280
      and tr["n_task_rows"]==10080
      and cr["status"]=="XP_A12_G7_GT_CLIP_G1_G2_FACE_CPU_PASS"
      and cr["n_images"]==280,"COMPLETE_COMPONENT_RECEIPTS")
 eq=report/"XP_A22_G7_C2_EQUIVARIANCE.csv"
 cl=clip/"XP_A12_G7_C_CLIP_GT_ORIENTATION.csv"
 gate(sha(eq)==tr["table_sha256"]
      and sha(cl)==cr["csv_sha256"],"ORIGINAL_SHA")
 eqrows=checkcsv(eq,36);clrows=checkcsv(cl,280)
 gate(len({(x["dataset"],x["t_ddpm1000"],x["rotation"])
        for x in eqrows})==36
      and len({(x["dataset"],x["basename"]) for x in clrows})==280,
      "UNIQUE_GRID")
 tag=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
 branch=f"codex/prdiff-track-r-a22-g7-diagnostics-{tag}"
 tree=ROOT/f"pr_diffusion_track_r_a22_g7_pub_{tag}"
 gate(not tree.exists(),"FRESH_TREE")
 subprocess.run(["git","-C",str(WT),"worktree","add","-b",
     branch,str(tree),a.expected_head],check=True)
 out=tree/"docs/track_r/executors/xp/returns"/f"XP_A22_G7_C2_CLIP_{tag}"
 ti=tree/"docs/track_r/executors/xp/theory_inputs"
 out.mkdir()
 for src in (eq,cl,r,rr,c):
  shutil.copyfile(src,out/src.name)
 for src in (eq,cl):
  shutil.copyfile(src,ti/src.name)
 receipt=dict(status="A22_G7_C2_AND_CPU_CLIP_DIAGNOSTICS_PUBLISHED",
  not_full_G7_completion=True,
  remaining_G7_policies_not_claimed=[
    "PRIOR_2_FORWARD","PRIOR_S_MID_GLOBAL2",
    "PRIOR_S_MID_ALL8","UNIVFD"],
  original_A22_execution_head=a.expected_head,
  original_C2_frozen_manifest_sha256=state["source_hash"],
  c2_native_final_receipt_sha256=sha(r),
  c2_bootstrap_receipt_sha256=sha(rr),
  clip_GT_only_receipt_sha256=sha(c),
  C2_n_images=280,C2_n_evaluations=10080,
  C2_aggregate_rows=36,CLIP_n_images=280,
  C2_table_sha256=sha(eq),CLIP_decisions_sha256=sha(cl),
  GPU_occupancy_h=state["gpu_occupancy_h"],
  no_inverse_solver=True,no_A21_10_replication=True)
 (out/"XP_A22_G7_PUBLICATION_RECEIPT.json").write_text(
   json.dumps(receipt,indent=2,sort_keys=True)+"\n")
 subprocess.run(["git","-C",str(tree),"add",
   str(out.relative_to(tree)),
   str((ti/eq.name).relative_to(tree)),
   str((ti/cl.name).relative_to(tree))],check=True)
 subprocess.run(["git","-C",str(tree),"commit","-m",
   "results(track-r): A22 G7 C2 equivariance and CLIP GT diagnostic only"],check=True)
 subprocess.run(["git","-C",str(tree),"push","origin",
   "HEAD:refs/heads/"+branch],check=True,timeout=900)
 print("A22_G7_C2_CLIP_PUBLISHED",branch,
     git(tree,"rev-parse","HEAD"),
     "C2_H",receipt["GPU_occupancy_h"],flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 for k in ("expected-head","run","report","clip"):
  p.add_argument("--"+k,required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A22_G7_PUBLISH",str(e),flush=True);sys.exit(8)
