#!/usr/bin/env python3
"""Publish frozen historical A8b per-run signed residuals next to original A8.

No new candidate files, no GT, no models, no GPU. Fail closed on byte
drift, A8 row changes, order, thresholds, or protected cohort paths.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
from collections import Counter
from scripts.track_r import xp_r0n2_a8b_cpu as mod

SRC=mod.BASE/"r0n2_a8b_20261027T041156Z"
DEST=mod.WT/"docs/track_r/executors/r0n/returns/R0N2_A8B_20261027T041156Z"
CSV_SHA="e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901"
TAU_SHA="a86d625ec574614ad42131ae648b84beefe7ce8cb340516ce6a2ae65f20db9f5"
DATA_FILES=("R0N2_A8B_PER_IMAGE.csv","R0N2_A8B_TAU_BY_SOURCE_FAMILY.csv","R0N2_A8B_RECEIPT.json")

def validate(a8,rows,tau,receipt):
    if not(len(a8)==len(rows)==760 and len(tau)==12):
        raise RuntimeError("STOP_A8B_PUBLISH_COUNTS")
    if not (receipt.get("status")=="R0N2_A8B_PASS"
      and receipt.get("n_rows")==760
      and receipt.get("n_candidates")==3240
      and receipt.get("tau_groups")==12
      and receipt.get("A8_original_sha256")==mod.A8_SHA
      and receipt.get("R07_original_sha256")==mod.R07_SHA
      and receipt.get("A8B_csv_sha256")==CSV_SHA
      and receipt.get("tau_csv_sha256")==TAU_SHA
      and receipt.get("historical_only") is True
      and receipt.get("gt_images_read") is False
      and receipt.get("gpu_used") is False
      and receipt.get("new_candidate_outcomes_inspected") is False):
        raise RuntimeError("STOP_A8B_PUBLISH_SOURCE_RECEIPT")
    prefix=list(a8[0])
    if list(rows[0])!=prefix+["rho_signed_by_run","tau_signed_good_q95"]:
        raise RuntimeError("STOP_A8B_PUBLISH_COLUMN_SCHEMA")
    groups={(x["source"],x["family"]):x for x in tau}
    if len(groups)!=12 or any(int(x["n_candidates"])<=0 for x in tau):
        raise RuntimeError("STOP_A8B_PUBLISH_TAU")
    total=0
    for original,modified in zip(a8,rows):
        if {k:modified[k] for k in prefix}!=original:
            raise RuntimeError("STOP_A8B_PUBLISH_A8_MUTATED")
        n=int(original["n_runs"])
        rs=modified["rho_signed_by_run"].split(";")
        if len(rs)!=n or any(mod.fmt6(float(q))!=q for q in rs):
            raise RuntimeError("STOP_A8B_PUBLISH_RHO_FORMAT")
        key=(original["source"],original["family"])
        if key not in groups or modified["tau_signed_good_q95"]!=groups[key]["tau_signed_good_q95"]:
            raise RuntimeError("STOP_A8B_PUBLISH_TAU_MATCH")
        if mod.fmt6(float(modified["tau_signed_good_q95"]))!=modified["tau_signed_good_q95"]:
            raise RuntimeError("STOP_A8B_PUBLISH_TAU_FORMAT")
        total+=n
    if total!=3240 or sum(int(x["n_candidates"]) for x in tau)!=3240:
        raise RuntimeError("STOP_A8B_PUBLISH_CANDIDATES")
    return True

def main(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A8B_PUBLISH_NOT_CPU")
    head=subprocess.check_output(["git","-C",str(mod.WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(mod.WT),"status","--porcelain"],text=True).strip()
    if head!=args.expected_head or dirty:
        raise RuntimeError("STOP_A8B_PUBLISH_GIT")
    if SRC.is_symlink() or DEST.exists() or DEST.is_symlink():
        raise RuntimeError("STOP_A8B_PUBLISH_STAGE_ALREADY_PRESENT")
    if (mod.sha(mod.A8)!=mod.A8_SHA
        or mod.sha(SRC/DATA_FILES[0])!=CSV_SHA
        or mod.sha(SRC/DATA_FILES[1])!=TAU_SHA):
        raise RuntimeError("STOP_A8B_PUBLISH_FROZEN_HASH")
    with mod.safe(mod.A8).open(newline="") as f:a8=list(csv.DictReader(f))
    with mod.safe(SRC/DATA_FILES[0]).open(newline="") as f:rows=list(csv.DictReader(f))
    with mod.safe(SRC/DATA_FILES[1]).open(newline="") as f:tau=list(csv.DictReader(f))
    receipt=json.loads(mod.safe(SRC/DATA_FILES[2]).read_text())
    validate(a8,rows,tau,receipt)
    DEST.mkdir(parents=True,exist_ok=False)
    for name in DATA_FILES:shutil.copyfile(mod.safe(SRC/name),DEST/name)
    with (DEST/"SHA256SUMS.txt").open("x") as f:
        for name in DATA_FILES:f.write(mod.sha(DEST/name)+"  "+name+"\n")
    print("R0N2_A8B_PUBLICATION_STAGE_PASS rows=760 candidates=3240 tau_groups=12")
    print("A8B_SHA256="+CSV_SHA)
    print("A8B_TAU_SHA256="+TAU_SHA)
    print("SOURCE_A8_UNCHANGED=true")
    print("DEST="+str(DEST))
    print("OK")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--expected-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A8B_PUBLISH "+str(e),flush=True)
        sys.exit(8)
