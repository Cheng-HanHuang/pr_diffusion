#!/usr/bin/env python3
"""A24 XP-K CPU-only, zero-GPU preparation. This DOES NOT authorize generation.

Produce two exact one-leaf config snapshot screens (SITCOM, FAST-DIPS),
read-only native source pin diagnostics, and cited prior F8 replay proof.
DAPS historical calibration remains pending native Hydra tau binding evidence.
The actual A23.5b gate (i) requires native adapter binding/source diff; gate (ii)
needs original-parameter two-B22-root identity replays; gate (iii) before DAPS
requires accepted F8 replay receipt. Do not turn a config-screen into a gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=Path(__file__).resolve().parents[2]
BASE=ROOT/"outputs/pr_diffusion/track_r"
EXPECTED_PLANNER="aafaf7f35e9d9d0beafa459259ca10b7fda4e8f5"
F8_NOTE=WT/"docs/track_r/executors/xp/returns/XP_F8_LEGACY_DAPS_8ROOT_REPLAY_EXACT_PASS_20261029.md"
SITCOM_NATIVE=ROOT/"external/SITCOM_ODE"
FAST_NATIVE=ROOT/"external/FAST_DIPS_A15"
DAPS_NATIVE=ROOT/"external/DAPS_XP_NATIVE_4K"
EXPECTED_HEADS={
"SITCOM":"275ab67efbd8146bffca20155171ba6be1169c09",
"FAST_DIPS":"a9758b38906b7c7950fa04491f37f6634948db26",
"DAPS_FFHQ":"e7a77d094167084faed19b599b96673b7bb11447"}
SOURCE_FILES={
"SITCOM":"sampler.py",
"FAST_DIPS":"sampler.py",
"DAPS_FFHQ":"configs/task/phase_retrieval.yaml"}
SOURCE_LOCKS={
"SITCOM":SITCOM_NATIVE,
"FAST_DIPS":FAST_NATIVE,
"DAPS_FFHQ":DAPS_NATIVE}
B22=WT/"configs/b22/b22_2_overnight.json"
FAST=WT/"configs/track_a/a15_fastdips_exposed_v1.json"
FROZEN_DAPS=WT/"configs/b23/fresh1_frozen.yaml"

def fail(ok,why):
    if not ok:
        raise RuntimeError("STOP_A39_A24_CPU_"+why)

def git(root,*args):
    return subprocess.check_output(["git","-C",str(root),*args],
                                   timeout=45,text=True).strip()

def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            d.update(b)
    return d.hexdigest()

def read(path):
    p=Path(path)
    fail(p.is_file() and not p.is_symlink(),"CONFIG_SOURCE_FILE")
    return json.loads(p.read_text())

def jsonout(path,obj):
    with Path(path).open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def snapshot(name,before,key,value):
    from scripts.track_r.xp_a32_a23_config_diff_cpu import check_exact_one_change
    proposed=copy.deepcopy(before)
    fail(key in proposed,"SOURCE_PARAMETER_NOT_PRESENT_"+name)
    proposed[key]=value
    screen=check_exact_one_change(before,proposed,name,key)
    fail(screen["status"]=="A23_ONE_CONFIG_LEAF_DIFF_ONLY_NOT_FULL_PARAMETER_GATE"
         and screen["changed_key_count"]==1
         and screen["complete_native_source_diff_gate_passed"] is False
         and screen["future_GPU_generation_authorized"] is False,
         "PREMATURE_SCIENCE_AUTHORIZATION")
    return proposed,screen

def validate_F8_receipt():
    text=F8_NOTE.read_text()
    for line in ("8/8 pixel-exact PASS",
                 "400 annealing ×5 ODE",
                 "B22:",
                 "DEV80:",
                 "8/8 EXACT"):
        fail(line in text,"F8_PRIOR_RECEIPT_UNEXPECTED_"+line)
    return {"receipt_file":str(F8_NOTE.relative_to(WT)),
            "receipt_sha256":sha(F8_NOTE),
            "prior_F8_status":"PRIOR_NATIVE_DAPS_8ROOT_EXACT_REPLAY_DOCUMENTED",
            "XP_K_DAPS_original_parameter_two_B22_root_replay":"NOT_RUN",
            "F8_receipt_does_NOT_replace_XPK_identity_replay":True}

def native_source_inventory():
    out={}
    for name,root in SOURCE_LOCKS.items():
        entry={"root":str(root),
               "expected_git_head":EXPECTED_HEADS[name],
               "git_head":None,"native_source_path":SOURCE_FILES[name],
               "native_source_sha256":None,
               "native_source_binding_proof":"NOT_YET_VERIFIED",
               "source_modifications_allowed":False}
        if root.is_dir() and not root.is_symlink():
            try:
                actual=git(root,"rev-parse","HEAD")
                entry["git_head"]=actual
                fail(actual==EXPECTED_HEADS[name],"NATIVE_SOURCE_PIN_"+name)
                source=root/SOURCE_FILES[name]
                if source.is_file() and not source.is_symlink():
                    entry["native_source_sha256"]=sha(source)
                # Do not silently equate a path/hash with a proof of the
                # parameter's downstream mathematical use.
            except (subprocess.CalledProcessError,OSError) as exc:
                entry["native_error"]=type(exc).__name__
        out[name]=entry
    return out

def main(a):
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")=="","NO_CUDA_EXPOSED")
    fail(git(WT,"rev-parse","HEAD")==a.expected_head
         and not git(WT,"status","--porcelain"),"DIRTY_OR_UNPINNED_WORKTREE")
    target=Path(a.out)
    fail(target.parent==BASE and target.name.startswith("xp_a39_xpk_cpu_")
         and not target.exists() and not target.is_symlink(),
         "UNSAFE_CPU_OUTPUT_PATH")
    # Two native-facing config shapes; do not invent a DAPS tau binding.
    sitcom=read(B22)["sitcom"]
    fast=read(FAST)["fast_native_pixel"]
    fail(sitcom["lgvd_tau"]==.01
         and sitcom["lgvd_lr"]==5e-5
         and sitcom["lgvd_steps"]==100
         and sitcom["timestep"]=="poly-7",
         "SITCOM_NATIVE_HISTORICAL_CONFIG")
    fail(fast["epsilon"]==.05
         and fast["sampler"]=="edm_FAST_DIPS"
         and fast["T"]==150
         and fast["K_ADMM"]==2
         and fast["S_x"]==5
         and fast["rho"]==200,
         "FAST_NATIVE_HISTORICAL_CONFIG")
    s_after,s_proof=snapshot("SITCOM",sitcom,"lgvd_tau",.05)
    f_after,f_proof=snapshot("FAST_DIPS",fast,"epsilon",33.26)
    daps=read(FROZEN_DAPS)
    fail(daps["scheduler"]["annealing_steps"]==400
         and daps["scheduler"]["diffusion_steps_per_anneal"]==5
         and daps["source"]["upstream_commit"]==EXPECTED_HEADS["DAPS_FFHQ"],
         "DAPS_HISTORICAL_CONFIG")
    f8=validate_F8_receipt()
    native=native_source_inventory()
    out={
      "status":"A39_A24_CPU_SOURCE_SCREENS_COMPLETE_NOT_NATIVE_GATE",
      "code_head":a.expected_head,"planner_head":EXPECTED_PLANNER,
      "policy":"A24 clean GPUs not mandatory for CPU preparation; quiet preferred for identity replay; unchanged tolerance",
      "registered_order":["SITCOM","FAST_DIPS","DAPS_FFHQ"],
      "proposed_parameter_values":{"SITCOM":.05,"FAST_DIPS":33.26,"DAPS_FFHQ":math.sqrt(2)*.05},
      "native_snapshot_screens":{"SITCOM":s_proof,"FAST_DIPS":f_proof},
      "DAPS_config_note":"DAPS B23 parent freeze specifies 400x5 but not native Hydra tau; no fabricated one-leaf snapshot",
      "native_inventory":native,"historical_F8_evidence":f8,
      "gate_i_committed_native_parameter_diff":"PENDING_ALL_FAMILIES",
      "gate_ii_original_two_B22_root_identity_replay":"NOT_RUN_ALL_FAMILIES",
      "gate_iii_DAPS_F8":"PRIOR_8ROOT_REPLAY_DOCUMENTED_SEPARATE_AUTHORITY",
      "GPU_requested":False,"GPU_allocated":False,
      "GPU_generation_authorized":False,
      "main_paper_cut_closed":True,
      "no_new_selector":True,"no_matched_compute_claim":True}
    target.mkdir(mode=0o700)
    jsonout(target/"SITCOM_BASELINE.json",sitcom)
    jsonout(target/"SITCOM_PROPOSED.json",s_after)
    jsonout(target/"FAST_DIPS_BASELINE.json",fast)
    jsonout(target/"FAST_DIPS_PROPOSED.json",f_after)
    jsonout(target/"A39_XPK_CPU_GATE_INVENTORY.json",out)
    print("A39_XPK_CPU_PREFLIGHT_COMPLETE_NO_GPU",target,flush=True)
    print("A39_CONFIG_ONLY_ONE_LEAF_SCREENS=2",flush=True)
    print("A39_DAPS_F8_PREVIOUS_8_OF_8_EXACT_DOCUMENTED",flush=True)
    print("A39_NATIVE_BINDING_GATE_NOT_YET_PASSED",flush=True)
    print("A39_XPK_ORIGINAL_TWO_ROOT_IDENTITY_REPLAYS_PENDING",flush=True)
    print("A39_NO_GPU_RUNS_LAUNCHED",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION_NEEDED_A39_A24_XPK_CPU_GATE",repr(e),flush=True)
        sys.exit(8)
