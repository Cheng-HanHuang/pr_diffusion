#!/usr/bin/env python3
"""Verify and stage only the A8 D1-P1 CLIP CSV and receipt to theory_inputs.

No image, candidate tensor, GT score or protected D1-P2 outcome enters
the theory tree. Inputs must pass hash/shape/choice and label checks.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
from scripts.track_r.xp_a8_d1p1_clip_cpu import (
    ROOT,BASE,A7WT,WT,ORIGINAL_A6_HEAD,D1_LABEL_FILE,
    D1_LABEL_SHA,CLIP_SHA,FIELDS,sha)

DEST=A7WT/"docs/track_r/executors/xp/theory_inputs"
PUB_CSV=DEST/"XP_D1_P1_K4_CLIP_DECISIONS.csv"
PUB_RECEIPT=DEST/"XP_D1_P1_K4_CLIP_RECEIPT.json"

def validate(inp,expected_head):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A8_CLIP_PUBLISH_NOT_CPU")
    for worktree,head in ((A7WT,expected_head),(WT,ORIGINAL_A6_HEAD)):
        a=subprocess.check_output(["git","-C",str(worktree),"rev-parse","HEAD"],
                                  text=True).strip()
        dirty=subprocess.check_output(["git","-C",str(worktree),"status","--porcelain"],
                                     text=True).strip()
        if a!=head or dirty:raise RuntimeError("STOP_A8_CLIP_PUBLISH_WORKTREE_DRIFT")
    if (inp.parent!=BASE or not inp.name.startswith("xp_a8_d1p1_clip_")
        or not inp.is_dir() or inp.is_symlink()):
        raise RuntimeError("STOP_A8_CLIP_PUBLISH_SOURCE_ROOT")
    source=inp/"XP_D1_P1_K4_CLIP_DECISIONS.csv"
    receipt_file=inp/"XP_D1_P1_K4_CLIP_RECEIPT.json"
    receipt=json.loads(receipt_file.read_text())
    if (receipt.get("status")!="XP_D1_P1_K4_CLIP_DECISIONS_PASS"
        or receipt.get("analysis_head")!=expected_head
        or receipt.get("n_candidates")!=400
        or receipt.get("n_images")!=100
        or receipt.get("run_indices")!=[0,1,2,3]
        or receipt.get("model_sha256")!=CLIP_SHA
        or receipt.get("D1_P2_outcomes_opened") is not False
        or receipt.get("gt_used_for_binary_chosen_good_only") is not True
        or receipt.get("D1_P1_frozen_labels_sha256")!=D1_LABEL_SHA
        or receipt.get("csv_sha256")!=sha(source)):
        raise RuntimeError("STOP_A8_CLIP_PUBLISH_RECEIPT")
    with source.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=FIELDS:
            raise RuntimeError("STOP_A8_CLIP_PUBLISH_COLUMNS")
        rows=list(reader)
    if len(rows)!=400:
        raise RuntimeError("STOP_A8_CLIP_PUBLISH_N")
    frozen={}
    if sha(D1_LABEL_FILE)!=D1_LABEL_SHA:
        raise RuntimeError("STOP_A8_CLIP_PUBLISH_LABEL_SHA")
    with D1_LABEL_FILE.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for run,ch in enumerate(row["run_labels"]):
                frozen[(row["image_id"],str(run))]=ch
    groups=defaultdict(set)
    for row in rows:
        key=(row["image_id"],row["run"])
        if key not in frozen or row["label"]!=frozen[key]:
            raise RuntimeError("STOP_A8_CLIP_PUBLISH_UNFROZEN_LABEL")
        if row["run"] not in ("0","1","2","3"):
            raise RuntimeError("STOP_A8_CLIP_PUBLISH_RUN")
        if row["run"] in groups[row["image_id"]]:
            raise RuntimeError("STOP_A8_CLIP_PUBLISH_DUPLICATE")
        groups[row["image_id"]].add(row["run"])
        for mode in ("g1","g2"):
            d=float(row[mode+"_margin"])
            choice=row[mode+"_choice"]
            good=row[mode+"_chosen_good"]
            if (not math.isfinite(d) or choice not in ("000","111")
                or choice!=("111" if d>0 else "000")
                or good not in ("0","1")):
                raise RuntimeError("STOP_A8_CLIP_PUBLISH_DECISION")
    if (len(groups)!=100 or len(frozen)!=400
        or any(v!={"0","1","2","3"} for v in groups.values())):
        raise RuntimeError("STOP_A8_CLIP_PUBLISH_IMAGE_RUN_SET")
    if any(p.exists() for p in (PUB_CSV,PUB_RECEIPT)):
        raise RuntimeError("STOP_A8_CLIP_PUBLISH_OVERWRITE")
    return source,receipt_file,receipt,rows

def main(a):
    source,rfile,receipt,rows=validate(Path(a.source),a.expected_head)
    if a.verify_only:
        print("A8_D1P1_CLIP_PUBLISH_VERIFY_PASS 400",flush=True)
        return
    PUB_CSV.write_bytes(source.read_bytes())
    PUB_RECEIPT.write_bytes(rfile.read_bytes())
    if sha(PUB_CSV)!=receipt["csv_sha256"]:
        raise RuntimeError("STOP_A8_CLIP_PUBLISH_COPY_SHA")
    print("A8_D1P1_CLIP_THEORY_INPUT_STAGED_PASS")
    print("CSV_SHA256="+sha(PUB_CSV))
    print("THEORY_CSV="+str(PUB_CSV))
    print("THEORY_RECEIPT="+str(PUB_RECEIPT))
    print("D1_P2_OUTCOME_EMBARGO_PRESERVED=true")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--verify-only",action="store_true")
    try:main(p.parse_args())
    except Exception as ex:
        traceback.print_exc()
        print("ACTION NEEDED: A8_CLIP_PUBLISH "+str(ex),flush=True)
        sys.exit(8)
