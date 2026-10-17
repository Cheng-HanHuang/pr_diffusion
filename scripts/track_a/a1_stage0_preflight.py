#!/usr/bin/env python3
"""Track A A1.0: bounded metadata/source/hardware inventory, no GPU computation.

Pure stdlib. No model import, image decode, measurement load/generation,
reconstruction, FFT, GT access or cohort selection. Never opens locked
confirmation artifacts. Does not modify external or historical worktrees.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/egr/research-pac/huang248")
SHA_MODEL = "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
B22_META = "94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219"
B21_META = "e0ed1106076c493b6494d82e59f2bddf534aa43a05895a2b6569784283a32654"
B24_DEV_META = "3945c2ed26e10766a64ab4e7fb64ea673fa2f478fbf080ea8d1d2c85584f6d5f"
A0_ZIP = "e0641c4c78e05489d8df53dd42587995b2af03460403b4ac3b22928c4b20d242"
SOURCES = {
    "DAPS": (ROOT / "pr_diffusion_b19_solver/external/daps",
             "e7a77d094167084faed19b599b96673b7bb11447"),
    "SITCOM": (ROOT / "external/SITCOM_ODE",
               "275ab67efbd8146bffca20155171ba6be1169c09"),
    "DiffFPR": (ROOT / "external/DiffFPR",
                "a45ffe58f18fed8a63d3446600424e2b08733524")
}
DAPS_TRACKED_DIFF = "fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69"

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def file_sha(p: Path, *, maxbytes: int) -> tuple[str,int]:
    if not p.is_file() or p.is_symlink():
        raise ValueError(f"missing/nonregular/symlink required file: {p}")
    n = p.stat().st_size
    if n <= 0 or n > maxbytes:
        raise ValueError(f"file exceeds declared bounds: {p.name} size={n}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest(), n

def cmd(args: list[str], timeout=25) -> tuple[int,str]:
    env=dict(os.environ, GIT_OPTIONAL_LOCKS="0", PYTHONDONTWRITEBYTECODE="1")
    try:
        p=subprocess.run(args, capture_output=True, timeout=timeout,
                         check=False, env=env, text=True)
        return p.returncode, (p.stdout if p.returncode == 0 else p.stderr)[:30000]
    except (OSError,subprocess.TimeoutExpired) as e:
        return 90, f"{type(e).__name__}: {e}"

def git_snapshot(path: Path, expected: str, label: str) -> tuple[dict,list[str]]:
    issues=[]
    if not (path/".git").exists() and not (path/".git").is_file():
        return {"path":str(path),"status":"STOP_NOT_GIT"}, [f"{label}: repository absent"]
    rc,head=cmd(["git","-C",str(path),"rev-parse","HEAD"])
    rc2,diff=cmd(["git","-C",str(path),"diff","HEAD","--"],timeout=40)
    rc3,names=cmd(["git","-C",str(path),"diff","--name-only","HEAD","--"],timeout=40)
    if rc or rc2 or rc3 or len(diff)>=30000:
        issues.append(f"{label}: bounded Git inspection failed")
    changed=names.splitlines() if rc3==0 else []
    data={"path":str(path),"head":head.strip(),"expected_head":expected,
          "tracked_diff_sha256":sha(diff.encode()) if rc2==0 else None,
          "tracked_source_changed":[s for s in changed if s.endswith(".py")],
          "tracked_bytecode_changed_count":sum(s.endswith(".pyc") for s in changed),
          "tracked_other_changed":[s for s in changed if not s.endswith((".py",".pyc"))]}
    if head.strip()!=expected:
        issues.append(f"{label}: Git HEAD differs from declared historical source")
    if label=="DAPS":
        if data["tracked_diff_sha256"]!=DAPS_TRACKED_DIFF:
            issues.append("DAPS: current tracked diff differs from recorded native DAPS diff")
        if data["tracked_source_changed"]!=["posterior_sample.py","sampler.py"]:
            issues.append("DAPS: unexpected Python source file changes")
    elif label=="SITCOM":
        if data["tracked_source_changed"]!=["posterior_sample.py"]:
            issues.append("SITCOM: unexpected tracked Python source changes")
        r,patch=cmd(["git","-C",str(path),"diff","HEAD","--","posterior_sample.py"])
        delta=[line for line in patch.splitlines()
               if (line.startswith("+") and not line.startswith("+++")) or
                  (line.startswith("-") and not line.startswith("---"))]
        if r or delta!=["-        Path(dir).mkdir()",
                        "+        Path(dir).mkdir(parents=True, exist_ok=True)"]:
            issues.append("SITCOM: frozen B22 one-line source patch not reproduced")
        data["posterior_source_patch_changed_lines"]=delta
        data["full_tracked_diff_b24_certified"]=False
    elif label=="DiffFPR" and (changed or diff):
        issues.append("DiffFPR: unexpected tracked changes")
    data["status"]="STOP" if issues else "SOURCE_SIGNATURE_MATCH"
    return data,issues

def read_meta(path: Path, expected_sha: str, maxbytes=5000000) -> tuple[bytes,int]:
    digest,n=file_sha(path,maxbytes=maxbytes)
    if digest!=expected_sha:
        raise ValueError(f"archived metadata changed: {path.name}, {digest}")
    return path.read_bytes(),n

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--checkout",type=Path,required=True)
    ap.add_argument("--expected-head",required=True)
    a=ap.parse_args()
    out=a.output.resolve()
    owner=ROOT/"outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
    if out.parent!=owner or out.exists() or a.checkout.resolve()!=ROOT/"pr_diffusion_track_a_a1_oracle_budget":
        print("A1_STOP: wrong preflight output/checkout namespace or result already exists")
        return 2
    if not out.parent.is_dir():
        out.parent.mkdir(parents=True,exist_ok=True)
    # One new output only; no existing file/worktree is overwritten.
    out.mkdir(exist_ok=False)
    issues=[]
    rc,head=cmd(["git","-C",str(a.checkout),"rev-parse","HEAD"])
    if rc or head.strip()!=a.expected_head:
        issues.append("A1 pinned execution checkout source HEAD mismatch")
    result={"schema_version":"a1.0-readonly-preflight.v1",
            "stage":"A1.0", "gpu_work_performed":False,
            "inference_performed":False,"measurements_created":False,
            "candidate_images_selected":False,
            "user_checkout_head":head.strip(),"expected_checkout_head":a.expected_head,
            "native_sources":{},"checkpoints":{},"metadata":{},
            "hardware_observation":{},"cohort_status":"NOT_RESERVED",
            "run_status":"NOT_RESERVED","locked_confirmation_access":False}
    for label,(path,h) in SOURCES.items():
        data,bad=git_snapshot(path,h,label)
        result["native_sources"][label]=data
        issues.extend(bad)
    for label,path in {
        "SITCOM_native":ROOT/"external/SITCOM_ODE/checkpoint/ffhq256.pt",
        "NP_DAPS_ffhq_model":ROOT/"models/ffhq_10m.pt"
    }.items():
        try:
            d,n=file_sha(path,maxbytes=2_000_000_000)
            result["checkpoints"][label]={"path":str(path),"size_bytes":n,
                                          "sha256":d,"expected_sha256":SHA_MODEL}
            if d!=SHA_MODEL:
                issues.append(f"{label}: checkpoint SHA differs from historical frozen model")
        except (OSError,ValueError) as e:
            issues.append(f"{label}: {e}")
    b21=ROOT/"outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401/analysis_theta0.7/fresh2_final_rows.csv"
    b22=ROOT/"outputs/pr_diffusion/b22_baselines/B22_2_overnight_20260727_155135/full/manifest.json"
    dev=ROOT/"outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z/DEV80_MANIFEST.json"
    archive=ROOT/"outputs/pr_diffusion/track_a_np_paper/a0_native_selector_audit_20261015/A0_B22_METADATA_REPLAY_V3_RETURN.zip"
    for name,p,h in [("B21_exposed",b21,B21_META),("B22_exposed",b22,B22_META),
                     ("B24_DEV80_exposed",dev,B24_DEV_META),("A0_verified_zip",archive,A0_ZIP)]:
        try:
            _,n=read_meta(p,h,5000000)
            result["metadata"][name]={"path":str(p),"sha256":h,"bytes":n,"verified":True}
        except (OSError,ValueError) as e:
            issues.append(f"{name}: {e}")
    if all(k in result["metadata"] for k in ("B21_exposed","B22_exposed","B24_DEV80_exposed")):
        try:
            with b21.open(newline="",encoding="utf8") as f:
                b21_ids=[str(row["image_id"]).zfill(5) for row in csv.DictReader(f)]
            b22_ids=[str(row["image_id"]).zfill(5) for row in json.loads(b22.read_text())["rows"]]
            j=json.loads(dev.read_text())
            if j.get("image_count")!=80 or j.get("confirmation_exposed") is not False:
                raise ValueError("DEV80 status/exposure mismatch")
            dev_ids=[str(row["image_id"]).zfill(5) for row in j["tasks"]]
            if len(b21_ids)!=100 or len(set(b21_ids))!=100 or set(b21_ids)!=set(b22_ids):
                raise ValueError("B21/B22 exposed 100 image ID sets disagree")
            if len(dev_ids)!=80 or len(set(dev_ids))!=80:
                raise ValueError("DEV80 exposed ID count/uniqueness disagrees")
            result["metadata"]["exposed_panels"]={
                "B21_B22_unique":100, "B24_DEV80_unique":80,
                "B22_DEV80_overlap_count":len(set(b22_ids)&set(dev_ids)),
                "exclusion_union_count":len(set(b22_ids)|set(dev_ids)),
                "only_exposed_membership_recorded":True,
                "new_cohort_allocated":False
            }
        except (ValueError,KeyError,TypeError,UnicodeError) as e:
            issues.append(f"exposed-panel IDs: {e}")
    for name,path in {
        "prdiff_ffhq":ROOT/"conda-envs/prdiff_ffhq/bin/python",
        "daps":ROOT/"conda-envs/daps/bin/python",
        "sitcom_ode_bw":ROOT/"conda-envs/sitcom_ode_bw/bin/python"
    }.items():
        result.setdefault("environment_paths",{})[name]={"present":path.is_file(),"path":str(path)}
    imgroot=ROOT/"data/ffhq/ffhq-dataset/images1024x1024"
    result["public_image_root"]={"path":str(imgroot),"directory_exists":imgroot.is_dir(),
                                 "image_bytes_accessed":False,"image_filenames_listed":False}
    # Observes GPU state only; never allocates memory, submits a kernel or reserves GPU.
    if shutil.which("nvidia-smi"):
        rr,desc=cmd(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],timeout=15)
        result["hardware_observation"]={"gpu_query_rc":rr,"gpu_query_text":desc,
                                        "gpu_reservation":False}
    else:
        result["hardware_observation"]={"gpu_query_rc":127,"gpu_query_text":"nvidia-smi absent"}
    if not result["public_image_root"]["directory_exists"]:
        issues.append("public FFHQ root missing; no development cohort can be proposed")
    result["issues"]=issues
    result["status"]="PASS_SOURCE_AND_EXPOSED_METADATA_PREFLIGHT" if not issues else "STOP_PREFLIGHT"
    with (out/"A1_0_PREFLIGHT.json").open("x",encoding="utf8") as f:
        json.dump(result,f,indent=2,sort_keys=True)
        f.write("\n")
    print(json.dumps({"status":result["status"],"issues":issues,
                      "sources":{k:v["status"] for k,v in result["native_sources"].items()},
                      "exposed":result["metadata"].get("exposed_panels"),
                      "GPU_work":False,"cohort":"NOT_RESERVED"},sort_keys=True))
    return 0 if not issues else 3

if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"A1_STOP_PREFLIGHT_EXCEPTION: {type(e).__name__}: {e}")
        sys.exit(4)
