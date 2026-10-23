#!/usr/bin/env python3
"""N3: bounded ImageNet asset inventory. Never read data trees or download."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
MODEL_FILES=(
 ROOT/"models/imagenet256.pt",
 ROOT/"models/256x256_diffusion_uncond.pt")
DATA_ROOT=ROOT/"data"
FAST_REPO=ROOT/"external/FAST_DIPS_A15"
FORBIDDEN=("confirmation","confirm305")

def forbid(path):
    if any(t in str(path).lower() for t in FORBIDDEN):
        raise RuntimeError("STOP_R0N_N3_PROTECTED_TOKEN")
    return path

def audit(expected_head,out):
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    if head!=expected_head or subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True):
        raise RuntimeError("STOP_R0N_N3_GIT")
    out=Path(out)
    if not str(out).startswith(str(BASE)+"/") or out.exists():
        raise RuntimeError("STOP_R0N_N3_OUTPUT")
    items=[]
    for p in MODEL_FILES:
        forbid(p)
        items.append({"path":str(p),"kind":"model_file",
         "exists":p.is_file() and not p.is_symlink(),
         "file_size_bytes":p.stat().st_size
             if p.is_file() and not p.is_symlink() else None})
    dirs=[]
    # One bounded depth-1 scan; no recursive data tree enumeration.
    if DATA_ROOT.is_dir() and not DATA_ROOT.is_symlink():
        for item in DATA_ROOT.iterdir():
            if not item.name.lower().startswith("imagenet"):
                continue
            forbid(item)
            dirs.append({"path":str(item),"exists":True,
                "kind":"directory" if item.is_dir() else "file",
                "size_bytes":item.stat().st_size if not item.is_symlink() else None,
                "is_symlink":item.is_symlink()})
    # Bounded grep over tracked FAST-DIPS *configuration* files only.
    grep_names=[]
    grep_status="FAST_REPO_MISSING"
    if FAST_REPO.is_dir() and not FAST_REPO.is_symlink():
        frozen=ROOT/"pr_diffusion_track_r/configs/track_r/pac_paths_track_r_v1.json"
        pinned=json.loads(frozen.read_text(encoding="utf-8"))["external"]["fast_dips"]["commit"]
        observed=subprocess.check_output(
            ["git","-C",str(FAST_REPO),"rev-parse","HEAD"],text=True,timeout=30).strip()
        if observed!=pinned:
            raise RuntimeError("STOP_R0N_N3_FAST_SOURCE_CHANGED")
        # Use git ls-files with config-related patterns; never recurse
        # into data, assets, model checkpoints, or image directories.
        config_paths=subprocess.run(
          ["git","-C",str(FAST_REPO),"ls-files","--",
           "*config*.py","*config*.json","*config*.yaml","*config*.yml",
           "*/config/*","*/configs/*"],
          text=True,capture_output=True,timeout=30,check=False)
        if config_paths.returncode==0:
            candidates=[x for x in config_paths.stdout.splitlines()
                if ".." not in Path(x).parts and
                   not any(t in x.lower() for t in FORBIDDEN)]
            if len(candidates)>1000:
                raise RuntimeError("STOP_R0N_N3_UNBOUNDED_CONFIG_SCAN")
            for relative in candidates:
                path=FAST_REPO/relative
                if path.is_file() and not path.is_symlink() and path.stat().st_size<100000:
                    text=path.read_text(encoding="utf-8",errors="replace")
                    if "imagenet256" in text.lower():
                        grep_names.append(relative)
            grep_status="BOUNDED_CONFIG_GREP_COMPLETE"
        else:
            grep_status="FAST_GIT_TRACKED_CONFIG_UNAVAILABLE"
    out.mkdir(parents=True,exist_ok=False)
    result={
      "status":"R0N_N3_BOUNDED_INVENTORY_PASS",
      "head":head,"cpu_only":True,"new_gpu_hours":0,
      "protected_cohort_access":False,
      "assets":items,"imagenet_depth1_matches":sorted(dirs,key=lambda r:r["path"]),
      "fast_config_grep_status":grep_status,
      "fast_config_files_mentioning_imagenet256":sorted(grep_names),
      "no_downloads":True,"no_image_tree_enumeration":True,
      "ImageNet_expansion_authorized":False}
    with (out/"R0N_N3_INVENTORY.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print("R0N_N3_BOUNDED_INVENTORY_PASS",len(dirs),
        "DEPTH1_MATCHES",len(grep_names),"FAST_CONFIG_MATCHES",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    try:audit(a.expected_head,a.out)
    except Exception as e:
        print("STOP_R0N_N3",repr(e),flush=True)
        traceback.print_exc()
        sys.exit(8)
