#!/usr/bin/env python3
"""CPU-only publication of the completed, source-pinned SITCOM replay receipts.

Hashes the exact PAC files reported by the executor, validates eight offline
comparisons and clean timing, and stages only three permitted non-GT receipts
plus SHA256SUMS in a fresh worktree folder. Never loads an image or GPU.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
RUN=ROOT/"outputs/pr_diffusion/track_r/xp_sitcom_replay_20261026T223925Z"
BASE=WT/"docs/track_r/executors/xp/returns"
NAME="XP_SITCOM_NATIVE_REPLAY_20261026T223925Z"
EXPECTED_MANIFEST_SHA="82d38a2cadc47c8ad864e97809e9cea9baafd62c261fa26d50fc1d332d48d525"
EXPECTED_GPU_SHA="f96f534a820c6c1f59c6e8f63b5eebd91a3dda5b2aec299a6d18410563d816e2"
EXPECTED_OFFLINE_SHA="d59c10a34b1b1986d6e78d86d125306f0b88862122123a20740d0941c6035d4f"
SOURCE_RUNNER_HEAD="8eaa106740b134a3edc051d8f10cbb8f3ea37321"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def safe(path):
    p=Path(path)
    top=ROOT.resolve()
    actual=p.resolve(strict=False)
    if (not p.is_absolute() or not p.is_file() or p.is_symlink()
        or not (actual==top or top in actual.parents)
        or any(x in str(p).lower() or x in str(actual).lower() for x in BAD)):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_UNSAFE_SOURCE "+str(p))
    return p

def sha(path):
    h=hashlib.sha256()
    with safe(path).open("rb") as f:
        for data in iter(lambda:f.read(4*1024*1024),b""):h.update(data)
    return h.hexdigest()

def read(path):return json.loads(safe(path).read_text())

def verify(status,gpu,score):
    if (status.get("state")!="PASS" or status.get("reason") is not None or
        status.get("completed_roots")!=8 or
        status.get("head")!=SOURCE_RUNNER_HEAD):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_NOT_PASS")
    if (gpu.get("status")!="XP_SITCOM_REPLAY_GENERATION_PASS"
        or gpu.get("runner_head")!=SOURCE_RUNNER_HEAD
        or gpu.get("prep_head")!="c055423234441fe907fef8dfda825815632ca15f"
        or gpu.get("manifest_sha256")!=EXPECTED_MANIFEST_SHA
        or gpu.get("n_roots")!=8 or gpu.get("GT_free_generation") is not True
        or gpu.get("timing_clean_eligible") is not True):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_GENERATION")
    timing=gpu.get("timing",{})
    if (timing.get("timing_class")!="CLEAN" or timing.get("timing_valid") is not True
        or timing.get("foreign_pids")!=[] or timing.get("reasons")!=[]
        or timing.get("n_samples")!=48):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_TIMING")
    gpu_h=float(gpu.get("gpu_hours",-1))
    if not (0<gpu_h<2):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_BUDGET")
    if (abs(gpu_h-float(status.get("gpu_hours",-1)))>1e-8
        or not (gpu.get("sample_wall_s_median",0)>0)):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_TIMING_IDENTITY")
    if (score.get("status")!="XP_SITCOM_REPLAY_PASS"
        or score.get("head")!=SOURCE_RUNNER_HEAD
        or score.get("manifest_sha256")!=EXPECTED_MANIFEST_SHA
        or score.get("n_roots")!=8 or score.get("n_label_match")!=8
        or score.get("n_exact_historical")!=8
        or score.get("no_gpu_scoring") is not True
        or score.get("generation_gt_free") is not True
        or score.get("original_outputs_immutable") is not True):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_SCIENCE")
    if (not 0<=float(score.get("max_abs_delta_db",-1))<=0.1
        or len(score.get("results",[]))!=8):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_PSNRS")
    if any((x.get("pass") is not True or
            x.get("exact_historical") is not True or
            x.get("replay_label")!=x.get("historical_label") or
            float(x.get("abs_delta_db",1))>0.1)
           for x in score["results"]):
        raise RuntimeError("STOP_SITCOM_PUBLICATION_ROOTS")
    if len({x["uid"] for x in score["results"]})!=8:
        raise RuntimeError("STOP_SITCOM_PUBLICATION_DUPLICATE_UID")
    return True

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_SITCOM_PUBLICATION_GPU_VISIBLE")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_SITCOM_PUBLICATION_CHECKOUT")
    if sha(RUN/"XP_SITCOM_REPLAY_TASKS.json")!=EXPECTED_MANIFEST_SHA:
        raise RuntimeError("STOP_SITCOM_PUBLICATION_MANIFEST_SHA")
    for name,expected in (
        ("XP_SITCOM_REPLAY_GPU_SUMMARY.json",EXPECTED_GPU_SHA),
        ("XP_SITCOM_REPLAY_OFFLINE_SCORE.json",EXPECTED_OFFLINE_SHA)):
        if sha(RUN/name)!=expected:
            raise RuntimeError("STOP_SITCOM_PUBLICATION_SOURCE_SHA "+name)
    status=read(RUN/"XP_SITCOM_REPLAY_STATUS.json")
    gpu=read(RUN/"XP_SITCOM_REPLAY_GPU_SUMMARY.json")
    score=read(RUN/"XP_SITCOM_REPLAY_OFFLINE_SCORE.json")
    verify(status,gpu,score)
    dest=BASE/NAME
    if dest.exists() or dest.is_symlink():
        raise RuntimeError("STOP_SITCOM_PUBLICATION_ALREADY_STAGED")
    dest.mkdir(parents=True,exist_ok=False)
    names=("XP_SITCOM_REPLAY_STATUS.json",
           "XP_SITCOM_REPLAY_GPU_SUMMARY.json",
           "XP_SITCOM_REPLAY_OFFLINE_SCORE.json")
    hashes=[]
    for name in names:
        target=dest/name
        shutil.copyfile(RUN/name,target)
        hashes.append((sha(target),name))
    with (dest/"SHA256SUMS.txt").open("x") as f:
        for digest,name in hashes:f.write(digest+"  "+name+"\n")
    print("SITCOM_PUBLICATION_STAGE_PASS",str(dest))
    print("SITCOM_GENERATION_SHA="+EXPECTED_GPU_SHA)
    print("SITCOM_OFFLINE_SHA="+EXPECTED_OFFLINE_SHA)
    print("SITCOM_NATIVE_REPLAY_EXACT=8/8")
    print("SITCOM_TIMING_CLEAN=48")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: SITCOM_PUBLICATION "+str(e),flush=True)
        sys.exit(8)
