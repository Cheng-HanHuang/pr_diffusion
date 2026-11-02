#!/usr/bin/env python3
"""A20 XP-C3: publish COMPLETE, GT-isolated frozen prior rows after GPU stop.

Executes only after controller has zero active GPU workers and a final receipt.
Exact six G1-style columns, one CSV per COMPLETE panel, both returns and
theory_inputs. Checks all original local SHA receipts. No GT/model inference.
Auto-commits/pushes only these requested artifacts, never modifies code.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
THEORY=WT/"docs/track_r/executors/xp/theory_inputs"
RETURN=WT/"docs/track_r/executors/xp/returns/XP_C3_A20_20261102"
PANELS=("XP_N","G5","XP_D4")
SIZES={"XP_N":6400,"G5":800,"XP_D4":6400}
COLUMNS=("image_id","run","label","prior_choice","prior_margin","prior_chosen_good")

def gate(c,msg):
 if not c:raise RuntimeError("STOP_A20_XPC3_PUBLISH_"+msg)
def sha(p):
 p=Path(p);gate(p.is_file() and not p.is_symlink(),"SHA_SOURCE")
 h=hashlib.sha256()
 with p.open("rb") as f:
  for z in iter(lambda:f.read(1<<20),b""):h.update(z)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 with Path(p).open("x") as f:
  json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def git(*args):return subprocess.check_output(
  ["git","-C",str(WT),*args],text=True,timeout=90).strip()
def stage(a):
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a25_xpc3_prod_")
      and run.is_dir(),"RUN")
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"CODE_HEAD_STILL_FROZEN")
 ledger=read(run/"XP_C3_A20_FINAL_STATUS.json")
 gate(ledger["status"] in (
      "A20_XPC3_ALL_THREE_PANELS_COMPLETE",
      "A20_XPC3_COMPLETE_PANELS_BUDGET_STOP")
      and not RETURN.exists(),"NO_PARTIAL_OR_DUPLICATE_PUBLISH")
 completed=ledger["completed_panels"]
 gate(completed and completed==list(PANELS[:len(completed)]),
      "ONLY_COMPLETED_PREFIX_PANELS")
 gate(0<=ledger["charged_gpu_h"]<=8.
      and ledger["project_gpu_h"]<=400.
      and ledger["source_manifest_sha256"]==
      sha(THEORY/"XP_C3_A20_SOURCE_MANIFEST.json"),
      "GLOBAL_PROJECT_BUDGET_AND_SOURCES")
 RETURN.mkdir()
 entries=[]
 for panel in completed:
  n=SIZES[panel]
  f=f"XP_C3_A20_{panel}_{n}_PRIOR_DECISIONS.csv"
  local=run/"final"/f
  receipt=run/"final"/f"XP_C3_A20_{panel}_OFFLINE_RECEIPT.json"
  proof=read(receipt)
  gate(proof["status"]=="A20_XPC3_COMPLETE_PANEL_OFFLINE_GT_EVALUATED"
       and proof["panel"]==panel
       and proof["n_candidates"]==n
       and sha(local)==proof["csv_sha256"]
       and ledger["final_csv_sha256"][panel]==proof["csv_sha256"],
       "LOCAL_PANEL_SHA_"+panel)
  with local.open(newline="") as fobj:
   r=list(csv.DictReader(fobj))
  gate(len(r)==n and tuple(r[0])==COLUMNS
       and len({(v["image_id"],int(v["run"])) for v in r})==n
       and all(v["prior_choice"] in ("000","111") and
           (v["prior_choice"]=="111")==(float(v["prior_margin"])>1e-7)
           and v["prior_chosen_good"] in ("0","1") for v in r),
       "PRIOR_DECISION_REPLAY_"+panel)
  target=RETURN/f
  copied=THEORY/f
  gate(not copied.exists(),"THEORY_FILE_NO_OVERWRITE")
  shutil.copyfile(local,target)
  shutil.copyfile(local,copied)
  gate(sha(target)==sha(copied)==sha(local),"TWO_COPIES_SAME_SHA")
  entries.append(dict(panel=panel,n=n,filename=f,
       sha256=sha(target),source_receipt_sha256=sha(receipt),
       chosen_good=sum(int(v["prior_chosen_good"]) for v in r)))
 receiptfile=RETURN/"XP_C3_A20_PUBLICATION_RECEIPT.json"
 write(receiptfile,dict(
  status="A20_XPC3_COMPLETE_PANELS_PUBLISHED",
  original_code_freeze_sha256=ledger["scorer_freeze_sha256"],
  source_manifest_sha256=ledger["source_manifest_sha256"],
  original_gpu_ledger_sha256=sha(run/"XP_C3_A20_FINAL_STATUS.json"),
  source_execution_head=a.expected_head,
  n_complete_panels=len(entries),
  rows_total=sum(x["n"] for x in entries),
  panels=entries,no_GT_prior_selection=True,
  GT_only_after_frozen_selection=True,
  only_S_mid_global2=True,
  no_2forward_secondary=True,
  no_matched_compute_claim=True,
  experiment_exploratory=True,
  charged_gpu_h=ledger["charged_gpu_h"],
  project_gpu_h=ledger["project_gpu_h"]))
 return entries

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 entries=stage(a)
 addpaths=["docs/track_r/executors/xp/returns/XP_C3_A20_20261102"]
 addpaths += [
  "docs/track_r/executors/xp/theory_inputs/"+e["filename"]
  for e in entries]
 subprocess.run(["git","-C",str(WT),"add","--",*addpaths],check=True)
 names=git("diff","--cached","--name-only").splitlines()
 gate(len(names)==2*len(entries)+1,"PUBLICATION_FILES_ONLY")
 subprocess.run(["git","-C",str(WT),"commit","-m",
  "results(track-r): A20 complete XP-C3 prior panels and G1-schema receipts"],
  check=True)
 current=git("rev-parse","HEAD")
 # Never force; nonfastforward rejects and preserves local commit.
 subprocess.run(["git","-C",str(WT),"push","origin",
  "HEAD:refs/heads/"+a.branch],check=True,timeout=180)
 print("A20_XPC3_PUBLICATION_PUSHED",current,
       "PANELS",",".join(e["panel"] for e in entries),
       "ROWS",sum(e["n"] for e in entries),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--run",required=True)
 p.add_argument("--branch",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A20_XPC3_PUBLISH",e,flush=True)
  sys.exit(8)
