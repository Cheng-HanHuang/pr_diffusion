#!/usr/bin/env python3
"""Seal optional ImageNet D1 v2 E1 addendum without modifying registered v2.

Inputs ONLY registered TH1_PREDICTIONS_v2.json, the phase-1 D1 K4
theory file and 400 phase-1 G1/G2 decisions with planner-specified SHA.
Calls the addendum contract before producing an immutable new file.
No XP return / phase-2 input.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,platform
from pathlib import Path

import numpy,scipy
from check_th1_imagenet_v2_e1_contract import check as contract

BASE_SHA="8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55"
PHASE_SHA="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seal(args):
    if sha(args.base_v2)!=BASE_SHA or sha(args.input)!=PHASE_SHA:
        raise ValueError("registered v2 or authorized phase-1 input differs")
    if sha(args.clip)!=args.clip_sha256:
        raise ValueError("phase-1 CLIP hash differs from predeclared checksum")
    contract(args.draft,args.base_v2)
    obj=json.loads(Path(args.draft).read_text())
    if obj["phase1_clip_sha256"]!=args.clip_sha256:
        raise ValueError("addendum draft not from supplied CLIP data")
    if len(args.code_commit)!=40 or any(
        c not in "0123456789abcdef" for c in args.code_commit):
        raise ValueError("invalid code commit")
    obj["version"]="TH1_PREDICTIONS_v2_E1"
    obj["status"]="FROZEN_BEFORE_IMAGENET_D1_PHASE2_UNBLINDING"
    obj["freeze_utc"]=datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    obj["code_commit"]=args.code_commit
    obj["draft_sha256"]=sha(args.draft)
    obj["production_run"]={
      "script":"scripts/track_r/theory/fit_th1_imagenet_v2_e1.py",
      "point_reps":args.point_reps,"bootstrap":args.bootstrap,
      "boot_reps":args.boot_reps,
      "contract":"check_th1_imagenet_v2_e1_contract.py PASS 24",
      "environment":{"python":platform.python_version(),
        "numpy":numpy.__version__,"scipy":scipy.__version__}}
    obj["immutability"]={
      "existing_v2_predictions_unchanged":True,
      "registered_v2_sha256":BASE_SHA,
      "new_artifact_only":"TH1_PREDICTIONS_v2_E1.json",
      "registered_phase1_SHA256":PHASE_SHA}
    obj["source_data_disclosure"]="Only ImageNet D1 FAST run indices 0-3 and their G1/G2 CLIP choices/margins/GT decision were inspected; NO phase-2 or xp/returns accessed."
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError("refuse to overwrite an already sealed E1 addendum")
    out.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    print("TH1_V2_E1_ADDENDUM_SEALED",out)
    print("SHA256",sha(out))
    print("UTC",obj["freeze_utc"])
    print("CODE_COMMIT",args.code_commit)
    print("NEXT: commit+push, then planner registers v2 E1 before D1 phase-2 unblinding")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("input","base_v2","clip","clip_sha256","draft",
                "output","code_commit"):
        p.add_argument("--"+key.replace("_","-"),required=True)
    p.add_argument("--point-reps",type=int,required=True)
    p.add_argument("--bootstrap",type=int,required=True)
    p.add_argument("--boot-reps",type=int,required=True)
    seal(p.parse_args())
