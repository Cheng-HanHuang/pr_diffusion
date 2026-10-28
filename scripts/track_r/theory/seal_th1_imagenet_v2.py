#!/usr/bin/env python3
"""Seal a contract-passing ImageNet FAST TH1 v2 draft (prereg A6.6, A7.4).

Packaging only: it does not change the model or any number in the draft.
Mirrors seal_th1_a6_v1.py. Reads ONLY the authorized phase-1 theory input
(XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv) and the locally produced v2 draft.
The draft must pass check_th1_imagenet_v2_contract.validate first.

Usage (from the repo root, on the theory branch; normally through
run_th1_imagenet_v2_freeze.sh):
python scripts/track_r/theory/seal_th1_imagenet_v2.py \
  --input /path/to/XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv \
  --draft /path/to/TH1_IMAGENET_V2_DRAFT.json \
  --output docs/track_r/theory/TH1_PREDICTIONS_v2.json \
  --code-commit $(git rev-parse HEAD) \
  --point-reps 350 --bootstrap 60 --boot-reps 3

NO data is read from docs/track_r/executors/xp/returns/.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,platform
from pathlib import Path

import numpy,scipy

from check_th1_imagenet_v2_contract import validate

IMAGENET_SHA="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seal(args):
    if sha(args.input)!=IMAGENET_SHA:
        raise ValueError("ImageNet phase-1 input checksum does not match")
    validate(args.draft)
    with open(args.draft,encoding="utf8") as f: obj=json.load(f)
    if obj["source_sha256"]!=IMAGENET_SHA:
        raise ValueError("forecast source does not match the phase-1 input")
    if obj.get("interval_type")!="predictive_realized_panel":
        raise ValueError("wrong interval interpretation")
    if obj["E1_CLIP"]["status"]!="ABSTAIN":
        raise ValueError("v2 draft must abstain on E1_CLIP")
    if len(args.code_commit)!=40 or any(c not in "0123456789abcdef"
                                    for c in args.code_commit):
        raise ValueError("code commit must be 40-digit hex")
    obj["version"]="TH1_PREDICTIONS_v2"
    obj["status"]="FROZEN_BEFORE_PHASE2_OUTCOMES"
    obj["freeze_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    obj["code_commit"]=args.code_commit
    obj["draft_sha256"]=sha(args.draft)
    obj["production_run"]={
        "script":"scripts/track_r/theory/fit_th1_imagenet_v2.py",
        "point_reps":args.point_reps,"bootstrap":args.bootstrap,
        "boot_reps":args.boot_reps,
        "contract":"check_th1_imagenet_v2_contract.py PASS 36 12",
        "environment":{"python":platform.python_version(),
                       "numpy":numpy.__version__,"scipy":scipy.__version__}}
    obj["model_comparison_lock"]=(
        "Primary: E0 predictive-interval coverage at K=8,12,16 on the"
        " realized 16-run panel of the same 100 images; tie-break by mean"
        " standardized residual (model_comparison_rule). race_tau1 and the"
        " transferred FFHQ tail are sensitivities.")
    obj["E1_CLIP_addendum_rule"]=(
        "E1_CLIP is abstained here. An E1_CLIP addendum may be frozen later"
        " only from G1/G2 decisions on the phase-1 candidates (run index"
        " 0-3), without any phase-2 information, and must be registered"
        " before 2026-10-29 12:01 ET (prereg A8). It may not change any"
        " number in this file.")
    obj["interpretation"]=(
        "Predictive realized-panel intervals conditional on the parametric"
        " Dirichlet + class-race model; not distribution-free"
        " identification intervals.")
    obj["source_data_disclosure"]=(
        "No ImageNet FAST outcome with run index >= 4 and no FFHQ K>4"
        " outcome was accessed before this freeze. The model code was fixed"
        " by the theory executor at f23549b (2026-10-27T21:30:41Z). The"
        " planner, who executed this production run, had seen the XP-D2/D3"
        " K<=4 aggregates (DAPS-4K and SITCOM-ODE on the same 100 images,"
        " published 2026-10-28T05:46Z) and changed nothing in the model.")
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError("refusing to replace existing frozen v2")
    path.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    print("TH1_IMAGENET_V2_SEALED",str(path))
    print("SHA256",sha(path))
    print("UTC",obj["freeze_utc"])
    print("CODE_COMMIT",args.code_commit)
    print("NEXT: commit, push, then planner registers the artifact SHA")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("input","draft","output","code_commit"):
        p.add_argument("--"+name.replace("_","-"),required=True)
    p.add_argument("--point-reps",type=int,required=True)
    p.add_argument("--bootstrap",type=int,required=True)
    p.add_argument("--boot-reps",type=int,required=True)
    seal(p.parse_args())
