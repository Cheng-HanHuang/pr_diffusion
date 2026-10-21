#!/usr/bin/env python3
"""Package A16 metadata only: no FFHQ faces, GT, measurements, or model weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import sys

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_a_np_paper/a16_np_fast_autopilot_20261021"

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--allow-incomplete",action="store_true")
    a=p.parse_args()
    final=BASE/"A16_AUTOPILOT_FINAL.json"
    if not final.is_file() and not a.allow_incomplete:
        raise RuntimeError("STOP no final pass receipt; use --allow-incomplete for debugging")
    outfile=BASE/("A16_PARTIAL_METADATA_RETURN.zip" if a.allow_incomplete else "A16_COMPLETE_METADATA_RETURN.zip")
    if outfile.exists():raise RuntimeError("STOP existing archive; no overwrites")
    paths=[]
    for path in BASE.rglob("*"):
        if not path.is_file() or path==outfile or path.suffix==".zip":
            continue
        parts=path.relative_to(BASE).parts
        if path.name in ("RUNNING.json","A16_CANDIDATE_COMPLETE.json","A16_SUBPROCESS_WALL.json",
                         "A16_CPU_PLAN_PASS.json","A16_SAVED_FAST_SELECTOR_FORENSICS.json",
                         "A16_AUTOPILOT_FINAL.json") or (
                         path.parent==BASE and path.name.startswith("A16_STAGE") and path.suffix==".json"
        ):
            paths.append(path)
        elif parts[0]=="logs" and path.suffix==".log":
            paths.append(path)
        elif parts[0]=="arm_specs" and path.suffix==".json":
            paths.append(path)
    files=[]
    # Retain an integrity manifest *inside* the metadata archive.
    with zipfile.ZipFile(outfile,mode="x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(paths):
            relative=str(path.relative_to(BASE))
            if "reconstruction" in relative.lower() or "ground_truth" in relative.lower():
                raise RuntimeError("STOP unexpected restricted filename")
            raw=path.read_bytes()
            if len(raw)>8_000_000:raise RuntimeError(f"STOP oversized metadata entry {relative}")
            z.writestr(relative,raw)
            files.append({"path":relative,"sha256":hashlib.sha256(raw).hexdigest(),
                          "bytes":len(raw)})
        z.writestr("A16_METADATA_ARCHIVE_MANIFEST.json",json.dumps({
             "status":"A16_METADATA_ARCHIVE_PASS",
             "full_run_final_receipt_present":final.is_file(),
             "entry_count":len(files),"entries":files},indent=2,sort_keys=True)+"\n")
    with zipfile.ZipFile(outfile) as z:
        bad=z.testzip()
    if bad:raise RuntimeError("STOP corrupt ZIP member "+bad)
    print(json.dumps({"status":"A16_METADATA_ZIP_PASS",
        "zip_path":str(outfile),"sha256":digest(outfile),
        "n_metadata_entries":len(files),"full_run_complete":final.is_file()},sort_keys=True))
if __name__=="__main__":
    try:main()
    except Exception as e:
        print(f"A16_METADATA_PACKAGE_STOP: {type(e).__name__}: {e}",file=sys.stderr)
        sys.exit(19)
