#!/usr/bin/env python3
"""A1.2 B22 EXPOSED-DEV K4 engineering study: eight images, 128 independent candidates.

No protected B24 cohort, no new measurements. Requires a discrete dated GPU
reservation before launch. Until specifically enabled in a later pinned commit,
only --mode plan is allowed. GT for offline scores after all candidates.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.track_a.a1_1b_native_one_each import (
    ROOT, REPO, BASE, DAPS, SITCOM, DIFFFPR, MODEL, MODEL_SHA,
    NP_PY, METHODS, GPU_UUIDS, MAX_DEVICE_USED_MIB, MIN_DEVICE_FREE_MIB,
    MAX_INITIAL_DEVICE_USED_MIB, trusted_git, git, check_gpu,
    gpu_state, import_b24_helper, np_config, sha_file, read_json, save_new,
    score_pngs, kill_own_group,
)
from scripts.track_a.a1_protocol import seed32_numpy, candidate_matrix, mixed_np8, mixed_np4

INPUT_FREEZE=BASE/"a1_2_exposed_b22_inputs_v1"/"A1_2_INPUT_FREEZE_COMPLETE.json"
EXPECTED_INPUT_FREEZE_SHA="fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490"
EXPECTED_SELECTION_SHA="faa2e292e915690dc8dd7d477591bb47a12a5f5977fd1c9648849e25c2d09f32"
IDS=("69073","68539","67948","60140","67293","67174","61101","66631")
OUT=BASE/"a1_2_exposed_b22_k4_v1"
K=4
MAX_EACH_WALL_S=600
MAX_TOTAL_WALL_S=10800
# Explicit dated A1.2 engineering-only GPU reservation, not held-out FFHQ data.
# 2026-10-19 22:30 to 2026-10-20 01:00 Eastern (UTC offsets -04:00).
from datetime import datetime, timezone
EXECUTION_ENABLED=True
RESERVATION_ID="TRACK_A_A1_2_EXPOSED_B22_K4_GPU3_20261019_NIGHT"
RUN_WINDOW_UTC=(datetime.fromisoformat("2026-10-20T02:30:00+00:00"),
                datetime.fromisoformat("2026-10-20T05:00:00+00:00"))


def validate_input_manifest(doc:dict)->list[dict]:
    if doc.get("schema")!="prdiff.a1.2.exposed-input-freeze.v1" or \
       doc.get("status")!="PASS_EXPOSED_B22_INPUTS_FROZEN_NOT_HELDOUT" or \
       doc.get("scope")!="EXPOSED_ENGINEERING_NO_PUBLICATION_GENERALIZATION_INFERENCE":
        raise RuntimeError("A1.2 frozen input schema/scope is incompatible")
    if doc.get("selected_id_list_sha256")!=EXPECTED_SELECTION_SHA or \
       doc.get("image_ids")!=list(IDS) or doc.get("candidate_method_runs")!=0 or \
       doc.get("protected_confirmation_access") is not False or \
       doc.get("no_new_measurements") is not True or doc.get("no_gpu_work") is not True:
        raise RuntimeError("A1.2 frozen input identity/governance drift")
    rows=doc.get("rows")
    if not isinstance(rows,list) or len(rows)!=len(IDS):
        raise RuntimeError("A1.2 missing eight known development rows")
    if [r.get("image_id") for r in rows]!=list(IDS):
        raise RuntimeError("A1.2 row order changed")
    for row in rows:
        identifier=row["image_id"]
        if row.get("split")!="A1_2_B22_EXPOSED_ENGINEERING_ONLY" or \
           row.get("gt_data_source")!="B22_ALREADY_EXPOSED_FINAL100_NOT_NEW_HELDOUT":
            raise RuntimeError(f"{identifier} wrongly marked pristine")
        if row.get("measurement_dtype")!="torch.float32" or \
           row.get("measurement_shape")!=[1,3,384,384]:
            raise RuntimeError(f"{identifier} signed measurement schema drift")
        gtpath=Path(row["ground_truth_tensor_path"])
        if gtpath!=BASE/"a1_2_exposed_b22_inputs_v1"/identifier/"ground_truth_tensor.pt":
            raise RuntimeError(f"{identifier} A1 GT path not isolated")
        from_source=Path(row["ground_truth_source_path"])
        expected_source=ROOT/"data/ffhq/ffhq-dataset/images1024x1024"/f"{(int(identifier)//1000)*1000:05d}"/f"{identifier}.png"
        if from_source!=expected_source:
            raise RuntimeError(f"{identifier} unexpected exposed GT path")
        meas=Path(row["measurement_path"])
        if not meas.is_relative_to(ROOT/"outputs/pr_diffusion/b21_solver") or \
           f"ffhq{identifier}_" not in meas.name:
            raise RuntimeError(f"{identifier} expected B21 exposed observation only")
        for p,h in ((gtpath,row.get("ground_truth_tensor_file_sha256")),
                    (meas,row.get("measurement_file_sha256")),
                    (from_source,row.get("ground_truth_source_sha256"))):
            if not p.is_file() or p.is_symlink() or sha_file(p)!=h:
                raise RuntimeError(f"{identifier}: bytes changed/missing for {p.name}")
    return rows


def make_plan(rows:list[dict],head:str,sources:dict)->dict:
    if [r["image_id"] for r in rows]!=list(IDS):
        raise RuntimeError("not the frozen exposed B22 panel")
    candidates=[]
    for row in rows:
        identifier=row["image_id"]
        for c in candidate_matrix(identifier):
            n32=seed32_numpy(c.family,identifier,c.replicate)
            candidates.append({
                "image_id":identifier, "method":c.family,
                "rep":c.replicate,"seed63":c.seed,"native_seed32":n32,
                "candidate_id":c.candidate_id,"config":c.config,
                "measurement_tensor_sha256":row["measurement_tensor_sha256"],
                "ground_truth_tensor_sha256":row["ground_truth_tensor_sha256"],
            })
    if len(candidates)!=128 or len({c["candidate_id"] for c in candidates})!=128 or \
       len({(c["image_id"],c["native_seed32"]) for c in candidates})!=128:
        raise RuntimeError("incomplete independent K4 candidate plan")
    return {
        "schema":"prdiff.a1.2.b22-exposed-independent-k4.plan.v1",
        "status":"CPU_PLAN_VALIDATED_EXECUTION_REQUIRES_LEASE_ACK",
        "a1_head":head,"source_heads":sources,
        "checkpoint_sha256":MODEL_SHA,
        "selected_images":list(IDS),
        "measurement_reused":True,
        "source_input_freeze_sha256":EXPECTED_INPUT_FREEZE_SHA,
        "source_input_sha256s":{r["image_id"]:r["measurement_tensor_sha256"] for r in rows},
        "candidate_count":128,
        "per_method_candidates":{m:32 for m in METHODS},
        "per_image_candidates":16,
        "method_order":list(METHODS),
        "replicate_order":[0,1,2,3],
        "candidate_registry":candidates,
        "np_mixed8_additional_trajectories":0,
        "original_B22_exposed_data":True,
        "not_prospective_heldout":True,
        "protected_confirmation_access":False,
        "score":"COMMON_SAVED_PNG8_RAW_RGB_PSNR_PRIMARY_ROT180_AUX",
        "max_each_wall_seconds":MAX_EACH_WALL_S,
        "max_total_wall_seconds":MAX_TOTAL_WALL_S,
        "initial_device_used_mib_max":MAX_INITIAL_DEVICE_USED_MIB,
        "min_initial_device_free_mib":MIN_DEVICE_FREE_MIB,
        "max_device_used_mib":MAX_DEVICE_USED_MIB,
        "gpu_run_reserved":False,
        "execution_enabled_only_with_gpu3_and_reservation_ack":EXECUTION_ENABLED,
        "conditional_run_window_utc": [v.isoformat() for v in RUN_WINDOW_UTC] if RUN_WINDOW_UTC else None,
    }


def freeze_preflight(expected_head:str)->tuple[dict,list[dict]]:
    if git(REPO,"rev-parse","HEAD")!=expected_head or \
       git(REPO,"branch","--show-current")!="codex/prdiff-track-a-a1-oracle-budget-20261016":
        raise RuntimeError("A1 source branch/commit not pinned")
    if git(REPO,"status","--porcelain","--untracked-files=no"):
        raise RuntimeError("A1 tracked source modified")
    sources=trusted_git()
    if not INPUT_FREEZE.is_file() or sha_file(INPUT_FREEZE)!=EXPECTED_INPUT_FREEZE_SHA:
        raise RuntimeError("verified A1.2 CPU frozen input manifest missing/drifted")
    rows=validate_input_manifest(read_json(INPUT_FREEZE))
    if sha_file(MODEL)!=MODEL_SHA:
        raise RuntimeError("checkpoint hash mismatch")
    for p in (ROOT/"conda-envs/daps/bin/python",
              ROOT/"conda-envs/sitcom_ode_bw/bin/python",
              NP_PY):
        if not p.is_file():raise RuntimeError(f"missing Python solver environment: {p}")
    return sources,rows


def make_np_spec(method:str,item:dict,seed:int,gpu:int,base_config:dict,run_root:Path):
    run_root.mkdir(parents=True,exist_ok=False)
    input_dir=run_root/"input"
    input_dir.mkdir()
    helper=import_b24_helper()
    save_new(input_dir/"input_manifest.json",helper.compatible_input_manifest(item,REPO))
    cfg=np_config(base_config,method,seed)
    config_path=run_root/"frozen_runtime_config.json"
    save_new(config_path,cfg)
    timing=run_root/"cuda_timing.json"
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO)+os.pathsep+env.get("PYTHONPATH",""))
    for key in ("B21_CONT_ENABLE","B21_CONT_STATE_PATH","B21_CONT_NOISE_SEED",
                "B21_SAVE_STATE_STEPS","B23_TRACE_PATH"):
        env.pop(key,None)
    return {
        "method":method,"native_seed":seed,"run_dir":run_root,"env":env,
        "log":run_root/"run.log","timing":timing,"terminal":run_root/"np1/reconstruction.pt",
        "command":[str(NP_PY),str(REPO/"scripts/b23/run_python_cuda_timed.py"),
                   "--script",str(REPO/"scripts/b22/run_b22_1_np_smoke.py"),
                   "--cwd",str(REPO/"scripts/b22"),"--timing-json",str(timing),"--",
                   "--config",str(config_path),"--repo_root",str(REPO),"--run_root",str(run_root)],
    }


def native_spec(method:str,row:dict,rep:int,seed:int,gpu:int,runroot:Path,config:dict):
    """Returns spec and optional *exclusive* DAPS symlink temp for cleanup."""
    helper=import_b24_helper()
    if method=="DAPS":
        data_name=f"a1-2-exposed-{row['image_id']}-{rep}-"+hashlib.sha256(str(OUT).encode()).hexdigest()[:8]
        data_dir=config_file=None
        try:
            data_dir,config_file=helper.prepare_daps_dataset(row,data_name)
            spec=helper.child_spec(method,REPO,row,data_name,runroot,rep,seed,gpu)
            spec["command"].append(f"hydra.run.dir={spec['run_dir'] / 'hydra'}")
        except BaseException:
            cleanup_daps(data_dir,config_file,row["image_id"])
            raise
        spec["env"].pop("B23_TRACE_PATH",None)
        return spec,(data_dir,config_file)
    if method=="SITCOM":
        spec=helper.child_spec(method,REPO,row,None,runroot,rep,seed,gpu)
        spec["env"].pop("B23_TRACE_PATH",None)
        return spec,None
    return make_np_spec(method,row,seed,gpu,config,runroot/f"rep{rep}"),None


def cleanup_daps(data_dir,config_file,identifier):
    if config_file is not None and config_file.is_file():
        config_file.unlink()
    if data_dir is not None and data_dir.is_dir():
        link=data_dir/f"{identifier}.png"
        if link.is_symlink():link.unlink()
        # Never recursively delete unknown files or user data.
        data_dir.rmdir()


def run_bounded(spec:dict,gpu:int,start:float)->dict:
    check_gpu(gpu)
    logfile=Path(spec["log"])
    t0=time.perf_counter()
    max_used=0
    with logfile.open("x",encoding="utf8") as f:
        proc=subprocess.Popen(spec["command"],cwd=spec["run_dir"],env=spec["env"],
                              stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                if time.perf_counter()-start>MAX_TOTAL_WALL_S:
                    raise RuntimeError("A1.2 aggregate 3h GPU wall ceiling")
                if time.perf_counter()-t0>MAX_EACH_WALL_S:
                    raise RuntimeError("A1.2 single full native candidate 600s ceiling")
                uuid,free,used=gpu_state(gpu)
                if uuid!=GPU_UUIDS[gpu] or used>MAX_DEVICE_USED_MIB:
                    raise RuntimeError(f"A1.2 GPU/memory guard: uuid={uuid},device_used={used}")
                max_used=max(used,max_used)
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f"{spec['method']} native rc={proc.returncode}; preserve {logfile}")
        finally:
            kill_own_group(proc)
    timer=read_json(Path(spec["timing"]))
    term=Path(spec["terminal"])
    if timer.get("status")!=0 or not term.is_file() or term.stat().st_size==0:
        raise RuntimeError(f"native output/timer invalid: {spec['method']} {term}")
    return {
      "method":spec["method"],"native_seed32":spec["native_seed"],
      "terminal":str(term),"terminal_sha256":sha_file(term),
      "timing_json":str(spec["timing"]),"timing_content_sha256":sha_file(Path(spec["timing"])),
      "log_file":str(logfile),
      "wall_seconds_subprocess":time.perf_counter()-t0,
      "gpu_active_seconds_event":float(timer["gpu_active_seconds"]),
      "timing_runtime_counters":timer.get("runtime_counters"),
      "peak_torch_reserved_mib":float(timer.get("peak_reserved_bytes",0))/1048576.0,
      "max_device_used_mib_observed":max_used,
    }


def select_best_by_psnr(rows:list[dict],indices:list[int])->dict:
    """Offline GT oracle only AFTER the full 128 candidate generation loop."""
    subset=[rows[j] for j in indices]
    best=max(subset,key=lambda r:(r["offline_raw_png_psnr_db"],-r["rep"]))
    return {"best_candidate_id":best["candidate_id"],
            "best_psnr_raw_db":best["offline_raw_png_psnr_db"],
            "best_psnr_rot180_aware_db_on_raw_winner":best["offline_ambiguity_aware_png_psnr_db"],
            "good25":best["offline_raw_png_psnr_db"]>=25.0,
            "candidate_count":len(subset),
            "charged_gpu_event_seconds":sum(v["gpu_active_seconds_event"] for v in subset),
            "charged_wall_seconds":sum(v["wall_seconds_subprocess"] for v in subset)}


def compute_offline_oracles(images:list[dict],inputs:list[dict])->dict:
    if len(images)!=len(IDS) or any(len(item["candidates"])!=16 for item in images):
        raise RuntimeError("incomplete candidate population; cannot score")
    out={}
    from collections import defaultdict
    for image,inp in zip(images,inputs):
        ident=image["image_id"]
        if ident!=inp["image_id"]:
            raise RuntimeError("image/GT mismatch at offline oracle score")
        rows=image["candidates"]
        # Native outputs are loaded only AFTER all 128 native candidates completed.
        score_pngs(rows,{"ground_truth_tensor_path":inp["ground_truth_tensor_path"]})
        for candidate in rows:
            candidate["metrics_caution"]="B22_EXPOSED_ENGINEERING_ONLY_NOT_HELDOUT"
        families=defaultdict(list)
        for c in rows:families[c["method"]].append(c)
        if set(families)!=set(METHODS) or any(len(v)!=4 for v in families.values()):
            raise RuntimeError(f"{ident}: not four full candidates per method")
        detail={}
        for method,four in families.items():
            ordered=sorted(four,key=lambda v:v["rep"])
            for k in (1,2,4):
                indices=[rows.index(c) for c in ordered[:k]]
                detail[f"{method}_K{k}"]=select_best_by_psnr(rows,indices)
        # Fixed NP mixed ordering LF0,S20,LF1,S21,..., *no new diffusion*.
        mixed=[c for j in range(4) for c in (families["NP_LF"][j],families["NP_S2"][j])]
        for k in (4,8):
            detail[f"NP_MIXED_K{k}"]=select_best_by_psnr(rows,[rows.index(c) for c in mixed[:k]])
        # Store only concise per-candidate metrics and SHA for archived return.
        out[ident]={"oracle_by_policy":detail,
                    "measurement_tensor_sha256":inp["measurement_tensor_sha256"],
                    "candidate_metrics":[{
                        "candidate_id":r["candidate_id"],
                        "raw_psnr_db":r["offline_raw_png_psnr_db"],
                        "rot180_psnr_db":r["offline_rot180_png_psnr_db"],
                        "ambiguity_aware_psnr_db":r["offline_ambiguity_aware_png_psnr_db"],
                        "png_sha256":r["comparable_saved_png_sha256"],
                    } for r in rows]}
    policies=sorted(next(iter(out.values()))["oracle_by_policy"])
    totals={}
    for policy in policies:
        vals=[out[i]["oracle_by_policy"][policy] for i in IDS]
        arr=sorted(v["best_psnr_raw_db"] for v in vals)
        totals[policy]={
            "n_images":len(IDS),
            "mean_psnr_raw_db":sum(arr)/len(arr),
            "median_psnr_raw_db":(arr[3]+arr[4])/2,
            "good25_count":sum(v["good25"] for v in vals),
            "bad20_count":sum(v["best_psnr_raw_db"]<20 for v in vals),
            "min_psnr_db":arr[0],
            "charged_gpu_event_seconds_sum":sum(v["charged_gpu_event_seconds"] for v in vals),
            "charged_wall_seconds_sum":sum(v["charged_wall_seconds"] for v in vals),
        }
    return {"schema":"prdiff.a1.2.exposed-dev-only-oracle.v1",
            "method":"offline Ground Truth PSNR best-of-K, NOT an executable selector",
            "score_representation":"COMMON_SAVED_PNG8_RGB_RAW_PRIMARY",
            "subject_to_180_degree_phase_ambiguity":True,
            "no_prospective_claims":True,"per_image":out,"policies":totals}


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("plan","execute"),required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--physical-gpu",type=int,choices=range(4))
    ap.add_argument("--reservation-ack",default="")
    args=ap.parse_args()
    sources,inputs=freeze_preflight(args.expected_head)
    plan=make_plan(inputs,args.expected_head,sources)
    if args.mode=="plan":
        plan["reservation_id_for_future_execution"]=RESERVATION_ID
        plan.pop("candidate_registry")   # use compact summary, full seed registry verified in memory
        print(json.dumps(plan,sort_keys=True))
        return 0
    if not EXECUTION_ENABLED or RESERVATION_ID.endswith("NOT_RESERVED") or RUN_WINDOW_UTC is None:
        raise RuntimeError("NO_GPU_LAUNCH_AUTHORIZED: A1.2 requires explicit dated reservation/source revision")
    if args.physical_gpu is None or args.reservation_ack!=RESERVATION_ID:
        raise RuntimeError("physical GPU and exact reserved A1.2 execution acknowledgement required")
    if args.physical_gpu!=3:
        raise RuntimeError("this physical-GPU lease is valid only for GPU index 3")
    # A discrete, dated source-committed RUN_WINDOW_UTC is required for actual launch.
    utc=datetime.now(timezone.utc)
    opening,latest_start=RUN_WINDOW_UTC
    if not (opening<=utc<=latest_start):
        raise RuntimeError("A1.2 dated reservation expired/not yet open, do not launch")
    if OUT.exists():
        raise RuntimeError("A1.2 output exists: never replace/implicitly resume existing trajectories")
    initial=check_gpu(args.physical_gpu)
    started=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=False)
    plan["gpu_start"]=initial
    plan["gpu_run_reserved"]=True
    plan["status"]="A1_2_EXECUTION_RESERVATION_ACCEPTED"
    plan["reservation_id"]=RESERVATION_ID
    plan["reservation_start_utc"]=opening.isoformat()
    plan["reservation_latest_launch_utc"]=latest_start.isoformat()
    save_new(OUT/"RUN_IDENTITY.json",plan)
    methods_completed=[]
    whole_images=[]
    config=read_json(REPO/"configs/b22/b22_1_smoke.json")
    helper=import_b24_helper()
    try:
        for item in inputs:
            image_id=item["image_id"]
            if time.perf_counter()-started>MAX_TOTAL_WALL_S:
                raise RuntimeError("A1.2 3-hour total compute budget exhausted")
            if sha_file(Path(item["measurement_path"]))!=item["measurement_file_sha256"]:
                raise RuntimeError(f"{image_id}: frozen measurement bytes changed")
            image_root=OUT/image_id
            image_root.mkdir(exist_ok=False)
            image_result={"image_id":image_id,"candidates":[]}
            for method in METHODS:
                method_parent=image_root/method.lower()
                method_parent.mkdir(exist_ok=False)
                for rep in range(K):
                    candidate_id=f"{image_id}:{method}:{rep}"
                    seed=seed32_numpy(method,image_id,rep)
                    if time.perf_counter()-started>MAX_TOTAL_WALL_S:
                        raise RuntimeError("A1.2 aggregate budget exhausted before new trajectory")
                    # Sanity-check isolation: each candidate identity is unique in frozen registry.
                    registered=[c for c in plan["candidate_registry"] if c["candidate_id"]==candidate_id]
                    if len(registered)!=1 or seed!=registered[0]["native_seed32"]:
                        raise RuntimeError(f"{candidate_id}: plan/native seed mismatch")
                    spec,scratch=native_spec(method,item,rep,seed,args.physical_gpu,method_parent,config)
                    try:
                        result=run_bounded(spec,args.physical_gpu,started)
                    finally:
                        if scratch is not None:
                            cleanup_daps(scratch[0],scratch[1],image_id)
                    result.update(image_id=image_id,method=method,rep=rep,
                                  candidate_id=candidate_id,
                                  measurement_tensor_sha256=item["measurement_tensor_sha256"],
                                  ground_truth_tensor_sha256=item["ground_truth_tensor_sha256"])
                    if sha_file(Path(item["measurement_path"]))!=item["measurement_file_sha256"]:
                        raise RuntimeError(f"{candidate_id}: shared measurement bytes changed during solver")
                    save_new(image_root/f"{method}_rep{rep}_COMPLETE.json",result)
                    image_result["candidates"].append(result)
                    methods_completed.append(result)
                    print(json.dumps({"status":"A1_2_CANDIDATE_PASS",
                           "candidate_id":candidate_id,
                           "completed":len(methods_completed),
                           "target":128,
                           "wall_seconds":result["wall_seconds_subprocess"],
                           "gpu_event_seconds":result["gpu_active_seconds_event"]},
                           sort_keys=True),flush=True)
            if len(image_result["candidates"])!=16:
                raise RuntimeError(f"{image_id}: incomplete method K4 matrix")
            whole_images.append(image_result)
            save_new(image_root/"IMAGE_CANDIDATES_COMPLETE.json",image_result)
            print(json.dumps({"status":"A1_2_IMAGE_K4_COMPLETE","image_id":image_id,
                              "images_completed":len(whole_images)},sort_keys=True),flush=True)
        if len(methods_completed)!=128 or len(whole_images)!=8:
            raise RuntimeError("missing full 8x16 candidate grid")
        scored=compute_offline_oracles(whole_images,inputs)
        save_new(OUT/"A1_2_OFFLINE_ORACLE_SUMMARY.json",scored)
        totals={
            "status":"PASS_A1_2_EXPOSED_B22_128_COMPLETE",
            "head":args.expected_head,
            "reservation":RESERVATION_ID,
            "image_ids":list(IDS),
            "method_counts":dict(Counter(v["method"] for v in methods_completed)),
            "candidate_count":len(methods_completed),
            "all_unique_candidate_ids":len({v["candidate_id"] for v in methods_completed})==128,
            "total_wall_seconds":time.perf_counter()-started,
            "sum_subprocess_wall_seconds":sum(v["wall_seconds_subprocess"] for v in methods_completed),
            "sum_cuda_event_seconds":sum(v["gpu_active_seconds_event"] for v in methods_completed),
            "maximum_reported_torch_reserved_mib":max(v["peak_torch_reserved_mib"] for v in methods_completed),
            "maximum_reported_device_used_mib":max(v["max_device_used_mib_observed"] for v in methods_completed),
            "native_seed_count":len({(v["image_id"],v["native_seed32"]) for v in methods_completed}),
            "frozen_inputs_sha256":EXPECTED_INPUT_FREEZE_SHA,
            "score_representation":"COMMON_SAVED_PNG8_RGB_RAW_PRIMARY",
            "np_mixed8_extra_gpu_work":0,
            "all_data_already_exposed":True,
            "not_pristine_or_confirmatory":True,
            "protected_confirmation_access":False,
            "next_stage_automatically_authorized":False,
        }
        save_new(OUT/"A1_2_COMPLETE.json",totals)
        print(json.dumps({"status":totals["status"],
                          "total_wall_seconds":totals["total_wall_seconds"],
                          "methods":totals["method_counts"],
                          "128_complete":True},sort_keys=True),flush=True)
        return 0
    except BaseException as e:
        save_new(OUT/"A1_2_STOP.json",{
            "status":"STOP_A1_2_NATIVE_K4",
            "error":f"{type(e).__name__}: {e}",
            "a1_head":args.expected_head,
            "total_wall_seconds":time.perf_counter()-started,
            "complete_candidate_count":len(methods_completed),
            "completed_candidate_ids":[v["candidate_id"] for v in methods_completed],
            "latest_native_logs_preserved":True,
            "restart_same_output_forbidden":True,
        })
        raise


if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print(f"A1_2_K4_STOP: {type(e).__name__}: {e}",flush=True)
        sys.exit(5)
