#!/usr/bin/env python3
"""A21 immediate per-complete-panel offline GT evaluation and publication.

Runs in a new Git worktree and publishes independent panel-specific branch.
The pinned A21 scoring worktree is NEVER modified, even as G5/D4 run.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r import xp_a29_a21_settings as S

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
SRC=WT/"docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"

def gate(v,msg):
    if not v:raise RuntimeError("STOP_A21_PANEL_PUBLICATION_"+msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],
                                   text=True,timeout=60).strip()

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and git(WT,"rev-parse","HEAD")==a.expected_head
         and not git(WT,"status","--porcelain")
         and a.panel in S.PANELS,"CLEAN_CODE")
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a25_xpc3_prod_a21_")
         and sha(SRC)==S.SOURCE_SHA,"SOURCE")
    folder=run/a.panel
    expected=S.SIZES[a.panel]//100
    ids=set()
    for i in range(100):
        p=folder/f"{i:03d}.json"
        gate(p.is_file(),"INCOMPLETE_100")
        doc=read(p)
        gate(doc["status"]=="A20_XPC3_IMAGE_PRIOR_PASS"
             and doc["panel"]==a.panel
             and doc["source_manifest_sha256"]==S.SOURCE_SHA
             and len(doc["rows"])==expected,"ROW_RECEIPT")
        for row in doc["rows"]:
            k=(row["image_id"],row["run"])
            gate(k not in ids and row["prior_choice"] in ("000","111"),
                 "DUPLICATE_OR_BAD_DECISION")
            ids.add(k)
    gate(len(ids)==S.SIZES[a.panel],"EXACT_PANEL_SIZE")
    # The previously frozen offline readout enforces GT never influencing
    # any scoring choice, and computes chosen-good only after completion.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(WT),
             PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
             OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
             OPENBLAS_NUM_THREADS="4")
    subprocess.run([sys.executable,
        str(WT/"scripts/track_r/xp_a25_xpc3_offline_final_cpu.py"),
        "--panel",a.panel,"--run",str(run),
        "--expected-head",a.expected_head],check=True,cwd=str(WT),env=env)
    name=f"XP_C3_A20_{a.panel}_{S.SIZES[a.panel]}_PRIOR_DECISIONS.csv"
    csvpath=run/"final"/name
    receiptpath=run/"final"/f"XP_C3_A20_{a.panel}_OFFLINE_RECEIPT.json"
    r=read(receiptpath)
    gate(r["status"]=="A20_XPC3_COMPLETE_PANEL_OFFLINE_GT_EVALUATED"
         and r["csv_sha256"]==sha(csvpath),"OFFLINE_GATE")
    with csvpath.open(newline="") as f:
        rows=list(csv.DictReader(f))
    gate(len(rows)==S.SIZES[a.panel] and
         list(rows[0])==["image_id","run","label",
                        "prior_choice","prior_margin","prior_chosen_good"],
         "SIX_COLUMNS")
    tag=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    branch=f"codex/prdiff-track-r-a21-xpc3-{a.panel.lower()}-{tag}"
    pub=ROOT/f"pr_diffusion_track_r_a21_pub_{a.panel.lower()}_{tag}"
    gate(not pub.exists(),"NEW_PUBLICATION_WORKTREE")
    subprocess.run(["git","-C",str(WT),"worktree","add","-b",
                    branch,str(pub),a.expected_head],check=True)
    theory=pub/"docs/track_r/executors/xp/theory_inputs"
    ret=pub/"docs/track_r/executors/xp/returns"/f"XP_C3_A21_{a.panel}_{tag}"
    ret.mkdir()
    theory.mkdir(exist_ok=True)
    for target in (ret/name,theory/name):shutil.copyfile(csvpath,target)
    meta=dict(status="A21_XPC3_COMPLETE_PANEL_PUBLISHED",
         panel=a.panel,n_candidates=len(rows),n_images=100,
         original_A20_manifest_sha256=S.SOURCE_SHA,
         original_A20_score_freeze_sha256=sha(
              WT/"docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"),
         A21_execution_head=a.expected_head,
         A21_freeze_sha256=sha(
              WT/"docs/track_r/executors/xp/XP_C3_A21_FREEZE.json"),
         local_offline_receipt_sha256=sha(receiptpath),
         source_decision_csv_sha256=sha(csvpath),
         independent_two_copies_sha256=[sha(ret/name),sha(theory/name)],
         complete_panels_only=True,
         no_GT_in_prior_scoring=True,no_matched_compute_claim=True,
         published_while_remaining_panels_may_be_scoring=True)
    (ret/"A21_PANEL_PUBLICATION_RECEIPT.json").write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n")
    subprocess.run(["git","-C",str(pub),"add",
                    "--",str(ret.relative_to(pub)),
                    str((theory/name).relative_to(pub))],check=True)
    gate(len(git(pub,"diff","--cached","--name-only").splitlines())==3,
         "THREE_ARTIFACTS")
    subprocess.run(["git","-C",str(pub),"commit","-m",
                    "results(track-r): A21 complete "+a.panel+
                    " frozen S_mid prior six-column decisions"],check=True)
    subprocess.run(["git","-C",str(pub),"push","origin",
            "HEAD:refs/heads/"+branch],check=True,timeout=180)
    print("A21_COMPLETE_PANEL_PUBLISHED",a.panel,
          "BRANCH",branch,"HEAD",git(pub,"rev-parse","HEAD"),
          "ROWS",len(rows),"CSV_SHA",sha(csvpath),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("expected-head","run","panel"):
        p.add_argument("--"+key,required=True)
    try:main(p.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A21_PUBLISH",exc,flush=True)
        sys.exit(8)
