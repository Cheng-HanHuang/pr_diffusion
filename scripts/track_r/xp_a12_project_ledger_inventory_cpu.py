#!/usr/bin/env python3
"""CPU-only A12 project-wide GPU-cost source inventory, XP-0 onward.

This is intentionally an INVENTORY, NOT a published authoritative ledger:
historical receipts use heterogeneous elapsed fields and overlapping stage
summaries. Reconciliation is required before G1a GPU admission. Never
silently add per-root and supervisor totals (double counting).
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,re,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,WT,ROOT,gate
FIELDS=("gpu_hours","gpu_h","gpu_hours_used","total_gpu_hours",
        "spent_gpu_seconds","gate_spent_gpu_seconds",
        "elapsed_gpu_allocation_seconds","incremental_gpu_h",
        "elapsed_seconds","A10_total_gpu_h","A10_F8_gpu_h",
        "total_gpu_h","project_gpu_hours","gpu_seconds")
PATTERNS=("*STATE*.json","*COMPLETE*.json","*RECEIPT*.json",
          "*SUMMARY*.json","*COST*.json","*LEDGER*.json")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda:f.read(1<<20),b""):h.update(piece)
    return h.hexdigest()
def numeric(x):
    return isinstance(x,(float,int)) and not isinstance(x,bool)
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(Path(a.out).parent==BASE and
         Path(a.out).name.startswith("XP_A12_PROJECT_LEDGER_INVENTORY_") and
         not Path(a.out).exists(),"INVENTORY_OUTPUT")
    entries=[]
    roots=sorted(p for p in BASE.iterdir()
       if p.is_dir() and p.name.startswith(("xp_","XP_")))
    for root in roots:
        files=set()
        for pat in PATTERNS:
            files.update(p for p in root.glob(pat) if p.is_file() and not p.is_symlink())
        # Also consume wave1/2 replay cost receipts under every run.
        for p in sorted(files):
            try:
                doc=json.loads(p.read_text())
                if not isinstance(doc,dict):continue
            except (UnicodeError,ValueError,PermissionError):
                continue
            matched={}
            for key in FIELDS:
                if numeric(doc.get(key)):matched[key]=doc[key]
            if numeric(doc.get("A10_limit_gpu_hours")):
                matched["A10_limit_gpu_hours"]=doc["A10_limit_gpu_hours"]
            if numeric(doc.get("cap_gpu_h")):
                matched["cap_gpu_h"]=doc["cap_gpu_h"]
            if not matched:continue
            entries.append(dict(run=root.name,path=str(p),sha256=sha(p),
                 status=doc.get("status"),values=matched))
    obj=dict(status="XP_A12_PROJECT_LEDGER_UNRECONCILED_INVENTORY",
       utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       n_runs=len(roots),n_cost_candidates=len(entries),sources=entries,
       WARNING="Do not sum all candidate fields: supervisor/stage/root cost overlap; certify per-stage exact wall-hours before G1a admission.",
       project_cap_gpu_h=400,A12_campaign_cap_gpu_h=150,
       A10_verified_final_gpu_h=55.54578541457511,
       project_total_before_A12_gpu_h=None,
       has_every_stage_since_XP0=False)
    with Path(a.out).open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print("XP_A12_LEDGER_INVENTORY_PASS",
          "stage_runs",len(roots),"candidate_receipts",len(entries),flush=True)
    print("A12_INVENTORY="+str(a.out),flush=True)
    print("A12_LEDGER_UNRECONCILED_STOP_GPU_ADMISSION",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A12_PROJECT_INVENTORY "+str(e),flush=True)
        sys.exit(8)
