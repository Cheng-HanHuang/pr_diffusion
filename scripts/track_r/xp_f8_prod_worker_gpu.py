#!/usr/bin/env python3
"""Native GT-free XP-F8 400x5 production worker.

One new root from frozen 920-task manifest. No R0 historical candidate or
GT path ever supplied to native DAPS. Source and checkpoint are verified.
Scientific outcomes and metrics are NOT computed: A9.4 embargo.
"""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from PIL import Image
from scripts.track_r.xp_f8_legacy_preflight_cpu import (
 ROOT,BASE,WT,ORIG_DAPS,DAPS_HEAD,MODEL,MODEL_SHA,A10_DEADLINE,
 git,sha,read_json,fail,PATCHED_SAMPLER_BLOB)
from scripts.track_r.xp_f8_legacy_replay_gpu import (
 DAPS_PY,TRACE_SOURCE,native_command,env,check_cfg)
def load(a):
    root=Path(a.run)
    fail(root.parent==BASE and root.name.startswith("xp_f8_prod_")
         and root.is_dir() and not root.is_symlink(),"RUN_DIR")
    fail(git(WT,"rev-parse","HEAD")==a.expected_head
         and not git(WT,"status","--porcelain"),"SOURCE_HEAD")
    prep=read_json(root/"XP_F8_PROD_PREP.json")
    manifest=root/"XP_F8_PROD_TASKS.json"
    fail(prep["head"]==a.expected_head
         and prep["status"]=="XP_F8_920_GT_FREE_PRODUCTION_PREP_PASS"
         and sha(manifest)==prep["manifest_sha256"],"FROZEN_INPUT")
    doc=read_json(manifest)
    fail(doc["status"]=="XP_F8_PROD_MANIFEST_GT_FREE_FROZEN"
         and doc["head"]==a.expected_head
         and doc["n_total"]==920
         and len(doc["rows"])==920 and 0<=a.index<920,"PROD_920_GRID")
    r=doc["rows"][a.index]
    fail(r["index"]==a.index and
         r["cohort"] in ("S1_b22","S2_b24_dev80")
         and r["annealing_steps"]==400
         and r["diffusion_steps"]==5
         and int(r["native_seed"])==int(r["canonical_seed"])%(2**32),
         "TASK_SOURCE")
    fail(sha(r["measurement_path"])==r["measurement_sha256"],
         "MEASUREMENT_SHA")
    fail(git(ORIG_DAPS,"rev-parse","HEAD")==DAPS_HEAD
         and git(ORIG_DAPS,"hash-object","sampler.py")==PATCHED_SAMPLER_BLOB
         and sha(MODEL)==MODEL_SHA,"NATIVE_CODE_OR_CHECKPOINT")
    fail(datetime.datetime.now(ZoneInfo("America/Detroit"))<A10_DEADLINE,
         "A10_DEADLINE")
    return root,doc,r,prep
def do(a,root,doc,row,prep):
    fail(a.gpu in range(4) and
         os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu),
         "GPU_BINDING")
    folder=root/"roots"
    fail(folder.is_dir(),"ROOTS_DIR_MISSING")
    dest=folder/f"r{a.index:04d}_a{a.attempt}"
    fail(not dest.exists() and dest.parent==folder and
         0<=a.attempt<=2,"ATTEMPT_OUTPUT_EXISTS")
    # The same originally patched native DAPS 400x5 invocation that passed
    # 8/8 pixel-exact historical replay; each new root changes ONLY its
    # frozen signed measurement, frozen seed and output location.
    cmd=native_command(row,dest)
    cfg=check_cfg(cmd,row)
    dest.mkdir(parents=False,exist_ok=False)
    synthetic=dest/"synthetic_input"
    synthetic.mkdir()
    placeholder=synthetic/"00000.png"
    Image.new("RGB",(256,256),(0,0,0)).save(placeholder)
    hook=dest/"sitehook"
    hook.mkdir()
    (hook/"sitecustomize.py").write_text(TRACE_SOURCE)
    trace=dest/"model_load_trace.json"
    stdout=dest/"native_stdout.log"
    start=time.monotonic()
    with stdout.open("x") as f:
        proc=subprocess.run(cmd,cwd=str(ORIG_DAPS),
            env=env(row,a.gpu,trace),stdout=f,stderr=subprocess.STDOUT)
    elapsed=time.monotonic()-start
    fail(proc.returncode==0,
         "NATIVE_EXIT_"+str(proc.returncode)+" "+str(stdout))
    fail(trace.is_file(),"CHECKPOINT_TRACE_MISSING")
    checkpoint=json.loads(trace.read_text())
    fail(checkpoint["class_name"]=="UNetModel"
         and checkpoint["strict"] is True
         and checkpoint["missing_keys"]==[]
         and checkpoint["unexpected_keys"]==[]
         and checkpoint["tensor_count"]>=100,
         "NATIVE_CHECKPOINT_LOAD")
    output=dest/"legacy_daps400x5/samples/00000_run0000.png"
    fail(output.is_file() and output.stat().st_size>1000
         and not output.is_symlink(),"NATIVE_PNG_MISSING")
    with Image.open(output) as img:
        fail(img.mode in ("RGB","RGBA") and img.size==(256,256),
             "NATIVE_OUTPUT_DIMENSIONS")
        img.verify()
    fail("Randomly initialize" not in stdout.read_text(errors="replace"),
         "NATIVE_RANDOM_CHECKPOINT")
    receipt=dict(
        status="XP_F8_NEW_ROOT_GENERATION_PASS_EMBARGOED",
        root_index=a.index,attempt=a.attempt,cohort=row["cohort"],
        image_id=row["image_id"],rep=row["rep"],
        uid=row["uid"],native_seed=row["native_seed"],
        measurement_sha256=row["measurement_sha256"],
        output_path=str(output),output_sha256=sha(output),
        output_image_mode="RGB_256",
        manifest_sha256=prep["manifest_sha256"],head=a.expected_head,
        native_source_head=DAPS_HEAD,checkpoint_sha256=MODEL_SHA,
        native_log_sha256=sha(stdout),model_trace_sha256=sha(trace),
        original_replay_gate="8_of_8_exact_pass",
        resolved_config=cfg,native_wall_seconds=elapsed,
        physical_gpu=a.gpu,
        no_GT_loaded=True,no_GT_scoring=True,
        no_labels_calculated=True,no_scientific_outcome_opened=True,
        generation_only_not_timing=True,
        telemetry_sidecar_owned_by_controller=True)
    p=dest/"XP_F8_NEW_ROOT_COMPLETE.json"
    with p.open("x") as f:
        json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("XP_F8_NEW_ROOT_PASS_EMBARGOED",a.index,
          "gpu",a.gpu,"seconds",round(elapsed,2),
          "receipt_sha",sha(p),flush=True)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--run",required=True)
    parser.add_argument("--index",required=True,type=int)
    parser.add_argument("--attempt",required=True,type=int)
    parser.add_argument("--gpu",required=True,type=int)
    a=parser.parse_args()
    root,doc,row,prep=load(a)
    do(a,root,doc,row,prep)
if __name__=="__main__":
    try:main()
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_F8_NEW_ROOT "+str(e),flush=True)
        sys.exit(8)
