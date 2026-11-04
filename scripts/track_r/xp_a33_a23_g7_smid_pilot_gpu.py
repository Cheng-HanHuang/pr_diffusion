#!/usr/bin/env python3
"""A23 G7 GT-only S_mid global2: two-image B2/B8 numerical pilot, NOT production.

Scientific score is imported unchanged from A20/A21. Uses the 280 frozen
C2/CLIP GT-only image entries, not inverse-solver candidates or measurements.
A PAC user invokes this after GPU admission; never run from a scheduled cloud
automation. No outputs are overwritten and no full G7 claim is made.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/egr/research-pac/huang248")
BASE = ROOT/"outputs/pr_diffusion/track_r"
WT = Path(__file__).resolve().parents[2]
LOCK = WT/"docs/track_r/executors/xp/theory_inputs/XP_A23_G7_GT_SOURCE_LOCK.json"
FREEZE = BASE/"xp_c_c2_a22_20261104T025355Z"
PY_NATIVE = ROOT/"conda-envs/daps/bin/python"

def gate(x, name):
    if not x: raise RuntimeError("STOP_A23_G7_S_MID_PILOT_"+name)

def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def jsonread(p):
    return json.loads(Path(p).read_text())

def git(*a):
    return subprocess.check_output(
        ["git","-C",str(WT),*a],text=True,timeout=60).strip()

def source(index,head):
    gate(git("rev-parse","HEAD")==head and
         not git("status","--porcelain") and
         git("ls-files","--error-unmatch",
             str(LOCK.relative_to(WT)))==str(LOCK.relative_to(WT)),
         "CLEAN_GIT_AND_REGISTERED_SOURCE")
    spec=jsonread(LOCK)
    gate(spec["status"] ==
        "A23_G7_GT_SOURCE_AND_S_MID_GLOBAL2_DEFINITION_FROZEN_BEFORE_NEW_RESULTS"
        and spec["governing_planner"].startswith("6f5ee7f")
        and spec["S_mid_global2"]["timesteps_DDPM1000"] == [300,400,500,600]
        and spec["S_mid_global2"]["draws_per_time"] == [0,1,2,3]
        and spec["S_mid_global2"]["rotations"] == ["000","111"]
        and spec["S_mid_global2"]["seed_uid_rule"] ==
           "XPCGT|<dataset>|<basename>"
        and spec["GPU_launch_authorized_by_this_freeze"] is False,
        "FROZEN_S_MID_DEFINITION")
    proof=jsonread(FREEZE/"XP_C_C2_PREP.json")
    c=FREEZE/"XP_C_GT_IMAGES.json"
    gate(proof["head"]==spec["source_original_head"]
         and sha(c)==proof["C_manifest_sha256"] ==
             spec["source_image_manifest_sha256"]
         and sha(FREEZE/"XP_C2_EQUIVARIANCE_TASKS.json") ==
             spec["source_c2_task_manifest_sha256"],"FROZEN_G7_GT_INPUT")
    obj=jsonread(c)
    images=obj["images"]
    gate(len(images)==280 and obj["n_ffhq"]==180 and
         obj["n_imagenet"]==100 and index in (0,180),
         "PILOT_FIRST_IMAGE_PER_DATASET")
    e=images[index]
    expected="ffhq" if index==0 else "imagenet"
    gate(e["dataset"]==expected and
         sha(e["source_path"])==e["source_sha256"],
         "EXACT_GT_IMAGE_SHA")
    cfg=spec["models"][expected]
    gate(sha(cfg["checkpoint"])==cfg["checkpoint_sha256"],
         "NATIVE_MODEL_CHECKPOINT_SHA")
    return spec,e,cfg,sha(c)

def gpu_memory_admission(gpu):
    """A21-style pre-load worst-case headroom, conservative for B8."""
    gate(gpu in (0,1,2,3) and
         os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu),
         "PHYSICAL_GPU_BINDING")
    txt=subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"],text=True,timeout=30)
    rows={}
    for ln in txt.splitlines():
        v=[a.strip() for a in ln.split(",")]
        gate(len(v)==5,"GPU_QUERY_COLUMNS")
        rows[int(v[0])]=dict(total=int(v[1]),used=int(v[2]),
            free=int(v[3]),util=int(v[4]))
    gate(gpu in rows,"PHYSICAL_GPU_PRESENT")
    card=rows[gpu]
    # B8 pilot bounded request: conservative 16 GiB peak * 1.25 + 8 GiB.
    request=math.ceil(16384*1.25+8192)
    gate(card["free"]>=request and
         card["used"]+request<=math.floor(card["total"]*.80),
         "GPU_MEMORY_CO_TENANCY_HEADROOM")
    return dict(card,admission_request_mib=request,
        prior_peak_reservation_assumption_mib=16384,
        not_clean_gpu_timing=True)

def main(a):
    if a.preflight_only:
        gate(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_PREFLIGHT_NO_VISIBLE_GPU")
        spec,entry,cfg,source_sha=source(a.index,a.expected_head)
        from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
        gate(verify_clean_sparse(NATIVE), "NATIVE_DAPS_SOURCE_PREFLIGHT")
        print("A23_G7_S_MID_CPU_SOURCE_PREFLIGHT_PASS",a.index,
              entry["dataset"],"SOURCE_SHA",source_sha,
              "CODE_HEAD",a.expected_head,flush=True)
        return
    from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
    from scripts.track_r.xp_t_daps4k_worker_gpu import (
        model_config,load_model_native,validate_weight_trace)
    from scripts.track_r.xp_a25_xpc3_mid_gpu import (
        fix_exact_native_timestep,orient_sm)
    from scripts.track_r.xp_a29_a21_gpu_worker import score_group
    from scripts.track_r.xp_a12_g7_c2_gpu import load_gt
    from prdiffusion.track_r.prior_score import frozen_alpha_schedule,prior_seed
    import torch

    gate(Path(sys.executable).resolve()==PY_NATIVE.resolve()
         and os.environ.get("PYTHONDONTWRITEBYTECODE")=="1"
         and os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
             str(Path(a.run)/"pycache")+"/"),
         "NATIVE_PY_AND_SOURCE_CACHE")
    spec,entry,cfg,source_sha=source(a.index,a.expected_head)
    gate(verify_clean_sparse(NATIVE),"NATIVE_DAPS_SOURCE")
    root=Path(a.run)
    gate(root.parent==BASE
         and root.name.startswith("xp_a33_a23_g7_smid_pilot_")
         and root.is_dir() and not root.is_symlink(),
         "PILOT_RUN_DIR")
    out=root/f"G7_S_MID_GLOBAL2_PILOT_{a.index:03d}.json"
    gate(not out.exists() and not out.is_symlink(),
         "NO_PILOT_OVERWRITE")
    mem=gpu_memory_admission(a.gpu)
    gate(torch.cuda.is_available() and torch.cuda.device_count()==1,
         "SINGLE_VISIBLE_GPU")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    origin=Path.cwd()
    sys.path.insert(0,str(NATIVE))
    os.chdir(NATIVE)
    try:
        model,load_seconds,trace=load_model_native(model_config(cfg))
        validate_weight_trace(trace)
        model.eval()
        alpha=frozen_alpha_schedule()
        mapping=fix_exact_native_timestep(model,alpha,torch)
        x=load_gt(entry).float().to("cuda:0")
        gate(tuple(x.shape)==(1,3,256,256)
             and bool(torch.isfinite(x).all()),"GT_TENSOR")
        uid=f'XPCGT|{entry["dataset"]}|{entry["basename"]}'
        # Source-faithful A20 unbatched oracle vs A21 fixed-t B8.
        single=orient_sm(x,uid,model,torch,alpha,prior_seed)
        batched=score_group([(dict(uid=uid),x)],model,torch,alpha,
                            prior_seed,batch=8)[0]
        torch.cuda.synchronize()
        diffs={k:abs(float(single[k])-float(batched[k]))
               for k in ("L000","L111","prior_margin")}
        agree=single["prior_choice"]==batched["prior_choice"]
        gate(agree and max(diffs.values())<=1e-5,
             "A20_VERSUS_A21_B8_NUMERICAL_EQUIVALENCE")
        peak=int(math.ceil(torch.cuda.max_memory_reserved()/1048576))
        gate(peak<=16384,
             "NATIVE_B8_PEAK_EXCEEDED_ADMISSION_ESTIMATE")
        elapsed=time.monotonic()-start
        gate(0<elapsed<=1800.,"BOUNDED_PILOT_ELAPSED")
        result=dict(status="A23_G7_S_MID_GLOBAL2_TWO_IMAGE_PILOT_PASS",
           index=a.index,dataset=entry["dataset"],
           basename=entry["basename"],source_sha256=entry["source_sha256"],
           source_manifest_sha256=source_sha,
           source_lock_sha256=sha(LOCK),code_head=a.expected_head,
           native_source_HEAD="e7a77d094167084faed19b599b96673b7bb11447",
           model_checkpoint_sha256=cfg["checkpoint_sha256"],
           native_weight_trace=trace,model_time_mapping=mapping,
           model_load_seconds=load_seconds,elapsed_wall_seconds=elapsed,
           physical_gpu=a.gpu,gpu_occupancy_h=elapsed/3600.,
           peak_reserved_mib=peak,source_admission_snapshot=mem,
           B2_unbatched=single,B8_fixed_t_batched=batched,
           max_abs_loss_or_margin_diff=max(diffs.values()),
           B2_B8_choice_equal=True,
           GT_only_diagnostic=True,GT_available_for_scoring_only=True,
           no_inverse_problem_generation=True,
           no_main_paper_claim=True,no_full_G7_completion_claim=True)
        with out.open("x") as f:
            json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n");f.flush();os.fsync(f.fileno())
        print("A23_G7_S_MID_PILOT_IMAGE_PASS",a.index,
              entry["dataset"],"maxdiff",max(diffs.values()),
              "gpu_h",elapsed/3600.,"peak_mib",peak,flush=True)
    finally:
        os.chdir(origin)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--index",required=True,type=int)
    p.add_argument("--gpu",required=True,type=int)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A23_G7_S_MID_PILOT",repr(e),flush=True)
        sys.exit(8)
