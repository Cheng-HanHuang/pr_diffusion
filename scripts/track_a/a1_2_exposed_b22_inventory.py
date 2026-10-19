#!/usr/bin/env python3
"""A1.2 source-only exposed B22 DEV pilot inventory: NO image/measurement/GT bytes.

Prepares a transparent EXPLORED/EXPOSED eight-image engineering-only cohort.
This is explicitly NOT the fresh FFHQ development or untouched confirmation
pilot in A1_STUDY_PREREGISTRATION.md; never reads B24 confirmation registry.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path("/egr/research-pac/huang248")
REPO=ROOT/"pr_diffusion_track_a_a1_oracle_budget"
INPUT=ROOT/"outputs/pr_diffusion/b22_baselines/B22_2_overnight_20260727_155135/full/manifest.json"
B22_SHA="94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219"
BASE=ROOT/"outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
OUT=BASE/"a1_2_exposed_b22_dev8_inventory_v1"
MEAS_PARENT=ROOT/"outputs/pr_diffusion/b21_solver"
GT_PARENT=ROOT/"data/ffhq/ffhq-dataset/images1024x1024"
RANK_DOMAIN="A1_2_EXPOSED_B22_DEV8_FIXED_RANK_20261019"
N=8


def sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(65536),b""):h.update(b)
    return h.hexdigest()


def rank(row:dict)->tuple[str,str]:
    identifier=str(row["image_id"])
    return hashlib.sha256(f"{RANK_DOMAIN}|{identifier}".encode()).hexdigest(),identifier


def select_dev8(rows:list[dict])->list[dict]:
    if len(rows)!=100:raise ValueError("B22 exposed manifest must contain 100 known rows")
    ids=[str(r.get("image_id","")) for r in rows]
    if len(set(ids))!=100 or any(len(i)!=5 or not i.isdigit() or int(i)<60000 or int(i)>69999 for i in ids):
        raise ValueError("B22 exposed panel ID count, duplication, or validation-range drift")
    return sorted(rows,key=rank)[:N]


def checked_b22_rows(blob:dict)->list[dict]:
    if not isinstance(blob,dict) or not isinstance(blob.get("rows"),list):
        raise ValueError("B22 manifest rows unavailable")
    rows=blob["rows"]
    result=[]
    for source in select_dev8(rows):
        for k in ("image_id","row_id","measurement_path","measurement_file_sha256",
                  "measurement_content_sha256","ground_truth_path",
                  "ground_truth_file_sha256","ground_truth_content_sha256"):
            if k not in source:raise ValueError(f"missing frozen B22 metadata column {k}")
        pid=str(source["image_id"])
        mp=Path(source["measurement_path"])
        gp=Path(source["ground_truth_path"])
        # Validate recorded path strings BEFORE touching metadata on the filesystem.
        if not mp.is_absolute() or MEAS_PARENT not in mp.parents or mp.suffix!=".pt":
            raise ValueError(f"measurement outside permitted exposed B21 path: {pid}")
        if not gp.is_absolute() or GT_PARENT not in gp.parents or gp.name!=pid+".png":
            raise ValueError(f"GT outside permitted exposed FFHQ image directory: {pid}")
        if not mp.is_file() or not gp.is_file():
            raise ValueError(f"preexisting B22 exposed path missing: {pid}")
        for k in ("measurement_file_sha256","measurement_content_sha256",
                  "ground_truth_file_sha256","ground_truth_content_sha256"):
            h=source[k]
            if not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError(f"invalid frozen SHA metadata {k} for {pid}")
        result.append({
            "image_id":pid,"b22_row_id":int(source["row_id"]),
            "measurement_path":str(mp),"measurement_file_sha256":source["measurement_file_sha256"],
            "measurement_tensor_sha256":source["measurement_content_sha256"],
            "ground_truth_source_path":str(gp),
            "ground_truth_source_sha256":source["ground_truth_file_sha256"],
            "ground_truth_tensor_sha256_prior_b22":source["ground_truth_content_sha256"],
            "rank_sha256":rank(source)[0],
            "measurement_data_opened":False,
            "gt_image_opened":False,
        })
    return result


def git_head(path:Path)->str:
    r=subprocess.run(["git","-C",str(path),"rev-parse","HEAD"],capture_output=True,
                     text=True,check=False,timeout=20,
                     env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if r.returncode:raise RuntimeError(f"cannot verify A1 source at {path}")
    return r.stdout.strip()


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    args=ap.parse_args()
    if git_head(REPO)!=args.expected_head:
        raise RuntimeError("A1 source HEAD mismatch")
    if OUT.exists():raise FileExistsError(f"existing A1.2 inventory preserved: {OUT}")
    if sha_file(INPUT)!=B22_SHA:
        raise RuntimeError("known exposed B22 manifest changed; STOP before candidate choice")
    blob=json.loads(INPUT.read_text(encoding="utf8"))
    selected=checked_b22_rows(blob)
    canonical="\n".join(r["image_id"] for r in selected)
    manifest={
        "schema":"a1.2.exposed-b22-development-inventory.v1",
        "status":"PASS_EXPOSED_DEVELOPMENT_INVENTORY_NOT_PROSPECTIVE",
        "purpose":"ENGINEERING_DEV_ONLY_NOT_NEW_HELDOUT_FFHQ",
        "gt_model_used":False,"gt_images_opened":False,
        "measurements_opened":False,"measurement_regeneration":False,
        "gpu_work":False,"protected_b24_confirmation_access":False,
        "source_a1_head":args.expected_head,
        "source_exposed_b22_manifest":str(INPUT),
        "source_exposed_b22_sha256":B22_SHA,
        "selection_domain":RANK_DOMAIN,
        "selected_image_ids":[r["image_id"] for r in selected],
        "selected_id_list_sha256":hashlib.sha256(canonical.encode()).hexdigest(),
        "rows":selected,
        "intended_future_k4_methods":["DAPS","SITCOM","NP_LF","NP_S2"],
        "future_gpu_candidates":N*16,
        "future_exploratory_metric":"SAVED_PNG8_RAW_RGB_PSNR_NOT_FLOAT",
        "approved_gpu_run":False,
        "locked_pristine_cohort_reserved":False,
        "note":"EXPOSED/B22 historical images are calibration only; cannot be prospective FFHQ reliability evidence",
    }
    OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/"A1_2_EXPOSED_DEV8_INVENTORY.json").open("x",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps({"status":manifest["status"],
                      "selected":manifest["selected_image_ids"],
                      "source_sha256":B22_SHA,
                      "selected_list_sha256":manifest["selected_id_list_sha256"],
                      "gpu_run":False,"confirmation_access":False},sort_keys=True))
    return 0


if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print(f"A1_2_INVENTORY_STOP: {type(e).__name__}: {e}")
        sys.exit(3)
