#!/usr/bin/env python3
"""A1.2 CPU-only freeze of eight already-exposed B22 images and measurements.

No CUDA, no diffusion/model, no new measurement generation, no confirmation305.
Only the eight pinned B22 files enumerated in our verified A1.2 inventory.
This stage checks their bytes and creates uniform B22-preprocessed GT tensors.
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
BASE=ROOT/"outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
INVENTORY=BASE/"a1_2_exposed_b22_dev8_inventory_v1"/"A1_2_EXPOSED_DEV8_INVENTORY.json"
OUT=BASE/"a1_2_exposed_b22_inputs_v1"
B22_MANIFEST_SHA="94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219"
SELECTION_SHA="faa2e292e915690dc8dd7d477591bb47a12a5f5977fd1c9648849e25c2d09f32"
SELECTED=("69073","68539","67948","60140","67293","67174","61101","66631")
MEAS_ROOT=ROOT/"outputs/pr_diffusion/b21_solver"
GT_ROOT=ROOT/"data/ffhq/ffhq-dataset/images1024x1024"


def file_sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def git_head(p:Path)->str:
    r=subprocess.run(["git","-C",str(p),"rev-parse","HEAD"],capture_output=True,
                     text=True,check=False,timeout=15,
                     env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if r.returncode:raise RuntimeError("cannot verify A1 checkout")
    return r.stdout.strip()


def inspect_manifest(doc:dict)->list[dict]:
    if doc.get("schema")!="a1.2.exposed-b22-development-inventory.v1" or \
       doc.get("status")!="PASS_EXPOSED_DEVELOPMENT_INVENTORY_NOT_PROSPECTIVE":
        raise ValueError("wrong selected inventory schema/scope")
    if doc.get("source_exposed_b22_sha256")!=B22_MANIFEST_SHA:
        raise ValueError("B22 archived source SHA mismatch")
    ids=doc.get("selected_image_ids")
    if ids!=list(SELECTED) or doc.get("selected_id_list_sha256")!=SELECTION_SHA or \
       hashlib.sha256("\n".join(ids).encode()).hexdigest()!=SELECTION_SHA:
        raise ValueError("selected ID order or digest changed")
    rows=doc.get("rows")
    if not isinstance(rows,list) or len(rows)!=len(ids) or [r.get("image_id") for r in rows]!=ids:
        raise ValueError("inventory row count/order drift")
    if doc.get("gpu_work") is not False or doc.get("protected_b24_confirmation_access") is not False:
        raise ValueError("cannot use an inventory without protected source firewall")
    seen=set()
    for row in rows:
        ident=row["image_id"]
        if ident in seen:raise ValueError("duplicate image ID")
        seen.add(ident)
        for name,root,suffix in (("ground_truth_source_path",GT_ROOT,".png"),
                                 ("measurement_path",MEAS_ROOT,".pt")):
            p=Path(row[name])
            if not p.is_absolute() or not p.is_relative_to(root) or p.suffix!=suffix:
                raise ValueError(f"{ident}: forbidden input path {name}")
        expected_gt=GT_ROOT/f"{(int(ident)//1000)*1000:05d}"/f"{ident}.png"
        if Path(row["ground_truth_source_path"])!=expected_gt:
            raise ValueError(f"{ident}: GT path does not match frozen image ID")
        if f"ffhq{ident}_" not in Path(row["measurement_path"]).name:
            raise ValueError(f"{ident}: measurement image identity drift")
        for h in ("ground_truth_source_sha256","measurement_file_sha256",
                  "ground_truth_tensor_sha256_prior_b22","measurement_tensor_sha256"):
            v=row.get(h)
            if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):
                raise ValueError(f"{ident}: invalid SHA field {h}")
    return rows


def compute_image_gt(path:Path):
    """Frozen B21/B22 preprocessing, on CPU, no oracle candidate selection."""
    import torchvision.transforms as T
    from PIL import Image
    transform=T.Compose([T.ToTensor(),T.Resize(256),T.CenterCrop(256)])
    image=(transform(Image.open(path).convert("RGB"))*2.0-1.0).unsqueeze(0)
    return image.contiguous()


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    args=ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly empty, stage is CPU only")
    if git_head(REPO)!=args.expected_head:
        raise RuntimeError("A1 source HEAD mismatch")
    if OUT.exists():
        raise FileExistsError(f"refusing existing A1.2 CPU input freeze output: {OUT}")
    if not INVENTORY.is_file():
        raise FileNotFoundError("frozen A1.2 inventory absent")
    doc=json.loads(INVENTORY.read_text(encoding="utf-8"))
    rows=inspect_manifest(doc)
    import torch
    sys.path.insert(0,str(REPO/"scripts/b22"))
    from b22_smoke_common import (tensor_content_sha256,load_measurement)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("torch CUDA unexpectedly visible in CPU input freeze")
    OUT.mkdir(parents=True,exist_ok=False)
    results=[]
    for row in rows:
        ident=row["image_id"]
        src=Path(row["ground_truth_source_path"])
        meas=Path(row["measurement_path"])
        if src.is_symlink() or meas.is_symlink():
            raise RuntimeError(f"refusing symlinked original input {ident}")
        if file_sha256(src)!=row["ground_truth_source_sha256"]:
            raise RuntimeError(f"GT source bytes modified for {ident}")
        if file_sha256(meas)!=row["measurement_file_sha256"]:
            raise RuntimeError(f"archived measurement bytes changed for {ident}")
        gt=compute_image_gt(src)
        y=load_measurement(meas,device="cpu")
        if tuple(gt.shape)!=(1,3,256,256) or gt.dtype!=torch.float32 or \
           not bool(torch.isfinite(gt).all()):
            raise RuntimeError(f"invalid GT tensor for {ident}")
        gt_sha=tensor_content_sha256(gt)
        y_sha=tensor_content_sha256(y)
        if gt_sha!=row["ground_truth_tensor_sha256_prior_b22"]:
            raise RuntimeError(f"B22 GT preprocessing/content mismatch for {ident}")
        if y_sha!=row["measurement_tensor_sha256"]:
            raise RuntimeError(f"B22 measurement tensor-content mismatch for {ident}")
        rowdir=OUT/ident
        rowdir.mkdir(exist_ok=False)
        gtpath=rowdir/"ground_truth_tensor.pt"
        torch.save({"ground_truth":gt},gtpath)
        item={
            "image_id":ident,"row_id":row["b22_row_id"],
            "gt_data_source":"B22_ALREADY_EXPOSED_FINAL100_NOT_NEW_HELDOUT",
            "split":"A1_2_B22_EXPOSED_ENGINEERING_ONLY",
            "measurement_path":str(meas),
            "measurement_file_sha256":row["measurement_file_sha256"],
            "measurement_tensor_sha256":y_sha,
            "measurement_shape":[1,3,384,384],
            "measurement_dtype":"torch.float32",
            "ground_truth_tensor_path":str(gtpath),
            "ground_truth_tensor_sha256":gt_sha,
            "ground_truth_tensor_file_sha256":file_sha256(gtpath),
            "ground_truth_source_path":str(src),
            "ground_truth_source_sha256":row["ground_truth_source_sha256"],
        }
        results.append(item)
        with (rowdir/"item.json").open("x",encoding="utf8") as f:
            json.dump(item,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
        print(json.dumps({"status":"A1_2_INPUT_VERIFIED","image_id":ident,
                          "gt_tensor_hash":gt_sha,"measurement_hash":y_sha},
                         sort_keys=True),flush=True)
    summary={
       "schema":"prdiff.a1.2.exposed-input-freeze.v1",
       "status":"PASS_EXPOSED_B22_INPUTS_FROZEN_NOT_HELDOUT",
       "a1_head":args.expected_head,
       "frozen_inventory_sha256":file_sha256(INVENTORY),
       "selected_id_list_sha256":SELECTION_SHA,
       "image_ids":list(SELECTED),
       "rows":results,
       "no_new_measurements":True,
       "no_gpu_work":True,
       "no_model_inference":True,
       "protected_confirmation_access":False,
       "candidate_method_runs":0,
       "potential_future_runs":8*16,
       "long_gpu_run_reserved":False,
       "score_representation_future":"COMMON_SAVED_PNG8_RGB_RAW_PRIMARY_WITH_ROT180_AUX",
       "scope":"EXPOSED_ENGINEERING_NO_PUBLICATION_GENERALIZATION_INFERENCE",
    }
    with (OUT/"A1_2_INPUT_FREEZE_COMPLETE.json").open("x",encoding="utf8") as f:
        json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps({"status":summary["status"],"images":len(results),
                      "new_measurements":0,"gpu":False},sort_keys=True))
    return 0


if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print(f"A1_2_INPUT_FREEZE_STOP: {type(e).__name__}: {e}",flush=True)
        sys.exit(4)
