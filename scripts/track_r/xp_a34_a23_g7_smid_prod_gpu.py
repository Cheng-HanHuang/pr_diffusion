#!/usr/bin/env python3
"""A23 GT-only G7 S_mid global2, native per-image worker (not solver generation).

Identical A20.3 paired-noise estimator and A21 fixed-t B8 score_group math.
Source is the ORIGINAL A22 280 GT images; no reconstructed candidates/measurements.
One worker scores one image, then atomically publishes one immutable JSON receipt.
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

from scripts.track_r.xp_a33_a23_g7_smid_pilot_gpu import (
    ROOT, BASE, WT, LOCK, FREEZE, PY_NATIVE,
    gate, sha, jsonread, git, gpu_memory_admission,
)

PILOT_HEAD = "ae09517c76d057fb076f737cd4e6d3c33c63ad66"
N = 280
SOURCE_SHA = "d517a8ea337c5fed8925f925bf24bae21cea60a566eb29f6795bc31e350abfde"
C2_SOURCE_SHA = "75e8fe296932044eb4ff1cd30300ef58a8f48ece02acd9b22ba3aa6f3bd7c7ac"
RESULT_STATUS = "A23_G7_S_MID_GLOBAL2_GT_IMAGE_PASS"


def check(a):
    rel = str(LOCK.relative_to(WT))
    gate(git("rev-parse", "HEAD") == a.expected_head
         and not git("status", "--porcelain")
         and git("ls-files", "--error-unmatch", rel) == rel,
         "A34_COMMITTED_CLEAN_HEAD")
    spec = jsonread(LOCK)
    sm = spec["S_mid_global2"]
    gate(spec["status"] ==
         "A23_G7_GT_SOURCE_AND_S_MID_GLOBAL2_DEFINITION_FROZEN_BEFORE_NEW_RESULTS"
         and spec["governing_planner"].startswith("6f5ee7f")
         and sm["timesteps_DDPM1000"] == [300, 400, 500, 600]
         and sm["draws_per_time"] == [0, 1, 2, 3]
         and sm["rotations"] == ["000", "111"]
         and sm["seed_uid_rule"] == "XPCGT|<dataset>|<basename>"
         and sm["comparison"] ==
             "margin=L000-L111; choose 111 iff margin>1e-7 else 000"
         and spec["g7_order"] ==
             ["S_mid_global2", "S_mid_all8", "2_forward", "UnivFD"],
         "A20_A23_FROZEN_DEFINITION")
    require_source = (spec["source_image_manifest_sha256"] == SOURCE_SHA
                      and spec["source_c2_task_manifest_sha256"] == C2_SOURCE_SHA)
    gate(require_source, "A22_FROZEN_GT_SOURCE")
    proof = jsonread(FREEZE/"XP_C_C2_PREP.json")
    src = FREEZE/"XP_C_GT_IMAGES.json"
    tasks = FREEZE/"XP_C2_EQUIVARIANCE_TASKS.json"
    gate(proof["status"] == "XP_C_C2_PREP_PASS"
         and proof["head"] == spec["source_original_head"]
         and sha(src) == proof["C_manifest_sha256"] == SOURCE_SHA
         and sha(tasks) == proof["C2_manifest_sha256"] == C2_SOURCE_SHA,
         "A22_SOURCE_BYTE_HASH")
    images = jsonread(src)
    gate(images["n_images"] == 280
         and images["n_ffhq"] == 180
         and images["n_imagenet"] == 100
         and len(images["images"]) == 280
         and 0 <= a.index < 280,
         "EXACT_280_GT_GRID")
    image = images["images"][a.index]
    gate(image["dataset"] == ("ffhq" if a.index < 180 else "imagenet")
         and sha(image["source_path"]) == image["source_sha256"],
         "EXACT_ONE_GT_FILE")
    cfg = spec["models"][image["dataset"]]
    gate(sha(cfg["checkpoint"]) == cfg["checkpoint_sha256"],
         "NATIVE_CHECKPOINT_SHA")
    run = Path(a.run)
    gate(run.parent == BASE
         and run.name.startswith("xp_a34_a23_g7_smid_prod_")
         and run.is_dir()
         and not run.is_symlink()
         and (run/"results").is_dir()
         and (run/"pycache").is_dir(),
         "PREPARED_RUN_ROOT")
    return spec, image, cfg


def main(a):
    spec, image, cfg = check(a)
    if a.preflight_only:
        from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE, verify_clean_sparse
        gate(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and verify_clean_sparse(NATIVE),
             "CPU_PREFLIGHT_SOURCE")
        print("A34_G7_S_MID_NATIVE_CPU_PREFLIGHT_PASS", a.index,
              image["dataset"], sha(LOCK), flush=True)
        return
    gate(Path(sys.executable).resolve() == PY_NATIVE.resolve()
         and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
         and os.environ.get("PYTHONPYCACHEPREFIX", "").startswith(
              str(Path(a.run)/"pycache") + "/"),
         "DAPS_NATIVE_PY_ENV")
    from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE, verify_clean_sparse
    from scripts.track_r.xp_a29_a21_gpu_worker import score_group
    from scripts.track_r.xp_a33_a23_g7_smid_pilot_gpu import gpu_memory_admission
    from scripts.track_r.xp_a12_g7_c2_gpu import load_gt
    from scripts.track_r.xp_a25_xpc3_mid_gpu import fix_exact_native_timestep
    from scripts.track_r.xp_t_daps4k_worker_gpu import (
        model_config, load_model_native, validate_weight_trace)
    from prdiffusion.track_r.prior_score import frozen_alpha_schedule, prior_seed
    import torch

    gate(verify_clean_sparse(NATIVE), "NATIVE_DAPS_CHECKOUT")
    receipt_path = Path(a.run)/"results"/f"{a.index:03d}.json"
    gate(not receipt_path.exists() and not receipt_path.is_symlink(),
         "NO_OVERWRITE_ACCEPTED_GT_SCORE")
    mem = gpu_memory_admission(a.gpu)
    gate(torch.cuda.is_available() and torch.cuda.device_count() == 1,
         "SINGLE_VISIBLE_GPU")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    uid = f'XPCGT|{image["dataset"]}|{image["basename"]}'
    start = time.monotonic()
    origin = Path.cwd()
    sys.path.insert(0, str(NATIVE))
    os.chdir(NATIVE)
    try:
        model,load_s,trace = load_model_native(model_config(cfg))
        validate_weight_trace(trace)
        model.eval()
        alpha = frozen_alpha_schedule()
        mapping = fix_exact_native_timestep(model, alpha, torch)
        x = load_gt(image).float().to("cuda:0")
        gate(x.dtype == torch.float32 and tuple(x.shape) == (1,3,256,256)
             and bool(torch.isfinite(x).all()), "GT_NATIVE_TENSOR")
        with torch.inference_mode():
            result = score_group([(dict(uid=uid),x)], model, torch, alpha,
                                 prior_seed, batch=8)
        gate(len(result)==1, "ONE_GT_SCORE")
        result = result[0]
        gate(result["denoiser_examples"]==32
             and result["prior_choice"] in ("000","111")
             and all(math.isfinite(float(result[k]))
                     for k in ("L000","L111","prior_margin"))
             and (float(result["prior_margin"]) > 1e-7) ==
                 (result["prior_choice"]=="111"),
             "A20_SCORE_SCHEMA_AND_DECISION")
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        peak = math.ceil(torch.cuda.max_memory_reserved()/1048576)
        gate(0<elapsed<1800 and peak <= 16384,
             "BOUNDED_NATIVE_PRODUCTION_EVAL")
        payload=dict(
            status=RESULT_STATUS,index=a.index,
            dataset=image["dataset"],basename=image["basename"],
            source_sha256=image["source_sha256"],
            source_manifest_sha256=SOURCE_SHA,
            C2_manifest_sha256=C2_SOURCE_SHA,
            source_lock_sha256=sha(LOCK),
            source_pilot_head=PILOT_HEAD,implementation_head=a.expected_head,
            uid=uid,checkpoint_sha256=cfg["checkpoint_sha256"],
            B8_fixed_t_native_score=result,
            selected_orientation=result["prior_choice"],
            upright_kept=(result["prior_choice"]=="000"),
            denoiser_examples=32,GT_only_diagnostic=True,
            GT_not_passed_to_inverse_solver=True,
            original_XPC3_A20_estimator_preserved=True,
            A21_B8_equivalence_pilot_previously_passed=True,
            model_load_seconds=load_s,elapsed_wall_seconds=elapsed,
            pilot_or_generation="G7_280_GT_S_MID_GLOBAL2",
            physical_gpu=a.gpu,peak_reserved_mib=peak,
            memory_admission=mem,native_model_time_mapping=mapping,
            strict_native_weight_trace=trace,
            not_clean_timing=True,not_matched_compute=True,
            no_main_paper_claim=True)
        tmp=receipt_path.with_name("."+receipt_path.name+f".{os.getpid()}.tmp")
        with tmp.open("x") as f:
            json.dump(payload,f,sort_keys=True,indent=2,allow_nan=False)
            f.write("\n");f.flush();os.fsync(f.fileno())
        try:
            os.link(tmp,receipt_path)
        finally:
            tmp.unlink()
        gate(receipt_path.is_file(), "ATOMIC_RECEIPT")
        print("A34_G7_S_MID_GLOBAL2_GT_ONE_IMAGE_PASS",a.index,
              image["dataset"],result["prior_choice"],
              "seconds",round(elapsed,2),"peak_mib",peak,flush=True)
    finally:
        os.chdir(origin)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--index",required=True,type=int)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A34_G7_S_MID_GT_WORKER",repr(e),flush=True)
        sys.exit(8)
