#!/usr/bin/env python3
"""Seal a QA-passed A6 TH1 draft as v1; no GitHub registry mutation.

Reads ONLY the authorized A8/A8b calibration and the locally produced
A6 draft. The draft is tested by check_a6_panel_contract first.
Output SHA-256 and UTC time are printed for Cheng/planner registration.

Usage (from repo root, on the dedicated theory branch, after executing the
updated fit_th1_prefix_v1.py and checking its source code head):
python scripts/track_r/theory/seal_th1_a6_v1.py \
  --a8 /path/to/R0N2_PER_IMAGE_COUNTS.csv \
  --a8b /path/to/R0N2_A8B_PER_IMAGE.csv \
  --tau /path/to/R0N2_A8B_TAU_BY_SOURCE_FAMILY.csv \
  --draft /path/to/TH1_A6_DRAFT.json \
  --output docs/track_r/theory/TH1_PREDICTIONS_v1.json \
  --code-commit $(git rev-parse HEAD)

NO data is read from docs/track_r/executors/xp/returns/.
Run git commit/push and planner SHA registration SEPARATELY;
no runner may view unsealed K>4 XP outcomes before registration.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,subprocess
from pathlib import Path

from check_a6_panel_contract import run as check_contract

A8_SHA="197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6"
A8B_SHA="e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901"
TAU_SHA="a86d625ec574614ad42131ae648b84beefe7ce8cb340516ce6a2ae65f20db9f5"
IMAGENET_SHA="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_a8b(a8,a8b,tau):
    if sha(a8)!=A8_SHA or sha(a8b)!=A8B_SHA or sha(tau)!=TAU_SHA:
        raise ValueError("A8, A8b, or frozen tau checksum does not match")
    with open(a8,newline="",encoding="utf8") as f:
        first=list(csv.DictReader(f))
    with open(a8b,newline="",encoding="utf8") as f:
        second=list(csv.DictReader(f))
    with open(tau,newline="",encoding="utf8") as f:
        taurows=list(csv.DictReader(f))
    if len(first)!=760 or len(second)!=760 or len(taurows)!=12:
        raise ValueError("wrong input row count")
    initial=list(first[0])
    if list(second[0])[:len(initial)]!=initial:
        raise ValueError("A8b's original schema differs from A8")
    tau_map={(r["source"],r["family"]):
             float(r["tau_exact_float64_repr"])for r in taurows}
    for i,(a,b)in enumerate(zip(first,second)):
        if any(a[k]!=b[k]for k in initial):
            raise ValueError("A8b changed original A8 row %d"%i)
        n=int(a["n_runs"])
        ro=[float(x) for x in b["rho_signed_by_run"].split(";")]
        order=[int(x)for x in b["e0_rank_order"].split(",")]
        if len(ro)!=n or sorted(order)!=list(range(n)):
            raise ValueError("bad residual source rank %d"%i)
        if any(ro[order[j]]>ro[order[j+1]]+1e-11
               for j in range(n-1)):
            raise ValueError("residual order mismatch %d"%i)
        tv=tau_map[(b["source"],b["family"])]
        # A8b stores the per-image tau as a 6-significant-digit display string;
        # compare it with the exact float64 tau rounded the same way.
        if f"{tv:.6g}"!=b["tau_signed_good_q95"].strip():
            raise ValueError("tau mismatch %d"%i)
        for c,r in zip(b["run_labels"],ro):
            if c=="C" and r>tv+1e-9:
                raise ValueError("C threshold mismatch %d"%i)
            if c=="I" and r<=tv-1e-9:
                raise ValueError("I threshold mismatch %d"%i)
    return {"A8_sha256":A8_SHA,"A8b_sha256":A8B_SHA,
            "A8b_tau_sha256":TAU_SHA,"A8b_used_for":"source/rank/threshold QA only",
            "A8b_residual_gaps_used_for_rank_fit":False,
            "A8b_rows_checked":len(second)}

def seal(args):
    meta=check_a8b(args.a8,args.a8b,args.tau)
    check_contract(args.draft)
    with open(args.draft,encoding="utf8") as f: obj=json.load(f)
    if obj["source_sha256"]!=A8_SHA:
        raise ValueError("forecast source does not match A8")
    if obj.get("interval_type")!="predictive_realized_panel":
        raise ValueError("wrong interval interpretation")
    if len(args.code_commit)!=40 or any(c not in "0123456789abcdef"
                                    for c in args.code_commit):
        raise ValueError("code commit must be 40-digit hex")
    obj["version"]="TH1_PREDICTIONS_v1"
    obj["status"]="FROZEN_BEFORE_HIGH_K_OUTCOMES"
    obj["freeze_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    obj["code_commit"]=args.code_commit
    obj["calibration_integrity"]=meta
    obj["model_comparison_lock"]="Use primary panel E0 and E1_CLIP predictive interval coverage counts; standardized residual tie-break. Secondary panels and race_tau1 are sensitivities."
    obj["interpretation"]="Predictive realized-panel intervals conditional on parametric Dirichlet + class-race/rare-C model; not distribution-free identification intervals."
    obj["source_data_disclosure"]="No FFHQ K>4 or ImageNet phase-2 outcomes accessed before freeze."
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError("refusing to replace existing frozen v1")
    text=json.dumps(obj,sort_keys=True,indent=2)+"\n"
    path.write_text(text)
    print("TH1_A6_V1_SEALED",str(path))
    print("SHA256",sha(path))
    print("UTC",obj["freeze_utc"])
    print("CODE_COMMIT",args.code_commit)
    print("NEXT: git add / commit / push, then planner registers artifact SHA; DO NOT OPEN XP RETURNS YET")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("a8","a8b","tau","draft","output","code_commit"):
        p.add_argument("--"+name.replace("_","-"),required=True)
    seal(p.parse_args())
