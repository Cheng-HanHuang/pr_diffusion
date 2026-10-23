#!/usr/bin/env python3
"""N0: materialize only SHA-pinned, aggregate R0.8 return files into Git.

Executed on PAC against the user's verified original ZIP. Writes only
docs/track_r/executors/r0/returns/R0_20261023/. Never reads GT files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/"docs/track_r/executors/r0/returns/R0_20261023"
ZIP_SHA="13c008f3bca59b7d8d717ec3215fa1f783ca453b31b9001886c84633b513efab"
FILES={
 "R0_7_ANALYSIS.json":"5b4b14fd9d104a7f1d5dd6f228adba9046f7f466ebc7e06f33a7282f575586b6",
 "R0_7_REPORT.md":"10a9ac826f15b970997043ce65ebba58c4827655b08fa9bee0b76517bdbfd738",
 "R0_8_P4_MATRIX.csv":"d73a9693c825bd58b881bad6c871964fad387151013d91cd9b1ed00f16c411d9",
 "R0_8_PROVISIONAL_DECISIONS.json":"aaed4b654aa41ca5e20fdf4ddb73e8655b1d15a272d205ae267298525d916f32",
 "R0_8_READINESS.json":"e8cf5cb141c97272c44dfc78ccf1bd5a5d1abe4f1529841c6e77b3ee5e48b9a1",
 "R0_8_EXPLORATORY_CASEBOOK.json":"ff892a48d6000339b53d2747db5fee41d38090b4289d32a1e66a006c2b344c35",
 "R0_8_RESEARCH_AUDIT.md":"26dc399ab4a5926c5e2be55c04ec2064ff6222bb331e655902beabbe75e42ec3"}
FORBIDDEN_FIELDS={"gt_path","candidate_path","score_receipt_path",
                  "measurement_path","image_path","gt_file",
                  "R0_7_OFFLINE_GT_CATALOG.json","R0_7_SCORED_GT_ROWS.jsonl"}

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def validate_aggregate(name,data):
    if b"\x00" in data:
        raise RuntimeError("STOP_N0_BINARY_CONTENT")
    text=data.decode("utf-8")
    if name.endswith(".json"):
        x=json.loads(text)
        def walk(obj):
            if isinstance(obj,dict):
                for k,v in obj.items():
                    if k in FORBIDDEN_FIELDS or k.lower().endswith("_gt_path"):
                        raise RuntimeError("STOP_N0_GT_OR_CANDIDATE_PATH_FIELD "+str(k))
                    walk(v)
            elif isinstance(obj,list):
                for v in obj:walk(v)
        walk(x)
        if name=="R0_7_ANALYSIS.json":
            if x.get("status")!="R0_7_ANALYSIS_COMPLETE" or x.get("n_candidates")!=3240:
                raise RuntimeError("STOP_N0_ANALYSIS_STATUS")
        if name=="R0_8_PROVISIONAL_DECISIONS.json":
            if x.get("automatic_R1_authorization") is not False:
                raise RuntimeError("STOP_N0_R1_AUTHORIZATION")
    if name=="R0_8_P4_MATRIX.csv" and len(text.splitlines())<37:
        raise RuntimeError("STOP_N0_POLICY_TABLE_ROWS")

def materialize(archive,expected_head):
    if (subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],
          text=True).strip()!=expected_head or
        subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True)):
        raise RuntimeError("STOP_N0_GIT_IDENTITY")
    if DEST.exists():
        raise RuntimeError("STOP_N0_ALREADY_EXISTS")
    archive=Path(archive)
    if not archive.is_file() or archive.is_symlink():
        raise RuntimeError("STOP_N0_SOURCE_ZIP")
    raw=archive.read_bytes()
    if digest(raw)!=ZIP_SHA:
        raise RuntimeError("STOP_N0_ZIP_SHA256")
    extracted={}
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError("STOP_N0_CORRUPT_ZIP")
        if set(FILES)-set(z.namelist()):raise RuntimeError("STOP_N0_MISSING_TABLE")
        for name,want in FILES.items():
            data=z.read(name)
            if digest(data)!=want:
                raise RuntimeError("STOP_N0_FILE_SHA256 "+name)
            validate_aggregate(name,data)
            extracted[name]=data
    DEST.mkdir(parents=True,exist_ok=False)
    for name,data in extracted.items():
        with (DEST/name).open("xb") as f:f.write(data)
    sums="".join(f"{FILES[k]}  {k}\n" for k in sorted(FILES))
    sums+=f"{ZIP_SHA}  R0_8_PLANNER_RETURN.zip (PAC original; not committed)\n"
    (DEST/"SHA256SUMS.txt").write_text(sums,encoding="utf-8")
    print("R0N_N0_TABLES_STAGED",len(extracted),"ZIP_SHA",ZIP_SHA,flush=True)
    print("R0N_N0_DEST",DEST,flush=True)
    print("R0N_N0_NEXT git add/commit/push this directory",flush=True)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--zip",required=True)
    p.add_argument("--expected-head",required=True)
    a=p.parse_args()
    try:sys.exit(materialize(a.zip,a.expected_head))
    except Exception as e:
        print("STOP_R0N_N0",repr(e),flush=True)
        sys.exit(8)
