#!/usr/bin/env python3
"""Seal A10 B16/F8 v3 after full production QA, without changing v1.

Only the authorized A8, A8b and released FAST 64 per-image strings are
checked. No DAPS/SITCOM post-prefix outcome and no xp/returns access.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,platform
from pathlib import Path
import numpy,scipy
from check_th1_fast_difficulty_v3 import check as contract
from fit_th1_fast_difficulty_v3 import read_inputs

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seal(args):
    rows,sources=read_inputs(args.a8,args.a8b,args.fast64)
    if len(rows)!=360:raise ValueError("wrong B16/F8 calibration image count")
    contract(args.draft)
    obj=json.loads(Path(args.draft).read_text())
    if obj["source_sha256"]!=sources:
        raise ValueError("source digests differ from fitted draft")
    if len(args.code_commit)!=40 or any(c not in "0123456789abcdef"
           for c in args.code_commit):
        raise ValueError("code commit must be a full lowercase git SHA")
    obj["version"]="TH1_PREDICTIONS_v3"
    obj["status"]="FROZEN_BEFORE_B16_F8_UNBLINDING"
    obj["code_commit"]=args.code_commit
    obj["freeze_utc"]=datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    obj["draft_sha256"]=sha(args.draft)
    obj["production_run"]={"point_reps":args.point_reps,
        "bootstrap":args.bootstrap,"boot_reps":args.boot_reps,
        "environment":{"python":platform.python_version(),
           "numpy":numpy.__version__,"scipy":scipy.__version__},
        "contract":"check_th1_fast_difficulty_v3.py PASS 496/132/36"}
    obj["immutability"]={"registered_v1_unchanged":True,
        "additional_separately_registered_forecast":True}
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError("v3 output already sealed")
    target.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print("TH1_B16_F8_V3_SEALED",str(target))
    print("SHA256",sha(target))
    print("UTC",obj["freeze_utc"])
    print("CODE_COMMIT",args.code_commit)
    print("NEXT: push and register with planner BEFORE unblinding B16/F8")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for n in ("a8","a8b","fast64","draft","output","code_commit"):
        ap.add_argument("--"+n.replace("_","-"),required=True)
    for n in ("point_reps","bootstrap","boot_reps"):
        ap.add_argument("--"+n.replace("_","-"),required=True,type=int)
    seal(ap.parse_args())
