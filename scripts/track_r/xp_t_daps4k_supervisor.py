#!/usr/bin/env python3
"""Four-physical-GPU DAPS-4K native GT-free timing supervisor.

Smoke two W=1 roots first (one FFHQ, one ImageNet), validate strict model
loading, provenance and per-root timing, project compute; then admit the
remaining 16 independently seeded native roots on up to four GPUs.
Do not modify existing PAC experiments. Do not kill active native workers:
on failure/budget/stop, finish active workers and preserve all receipts.
"""
from __future__ import annotations
import argparse,json,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
from prdiffusion.track_r.xp_gpu_telemetry import (
    collect,choose_gpu,classify_timing,event)
from scripts.track_r.xp_t_daps4k_worker_gpu import (
    ROOT,WT,BASE,RUN,PY_DAPS,PREP_HEAD,MANIFEST_SHA,
    immutable_manifest,checked_row,root_folder,sha,read)
CACHE=ROOT/"outputs/.cache/track_r"
GPU_INDICES=(0,1,2,3)
SMOKE=(0,9)
REMAINDER=tuple(i for i in range(18) if i not in SMOKE)

def dump_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def update(path,obj):
    p=Path(path)
    temp=p.with_name("."+p.name+"."+str(os.getpid()))
    dump_x(temp,obj)
    os.replace(temp,p)

def append(path,obj):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+"\n")

def worker_env(gpu,index):
    env=dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=str(WT),
        PYTHONPYCACHEPREFIX=str(RUN/"pycache"/f"worker_{index:02d}"),
        XDG_CACHE_HOME=str(CACHE),
        TMPDIR=str(CACHE/"tmp"),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        B20_LF_ENABLE="0", B21_CONT_ENABLE="0",
    )
    return env

def worker_args(head,index,gpu):
    return [str(PY_DAPS),
       str(WT/"scripts/track_r/xp_t_daps4k_worker_gpu.py"),
       "--expected-head",head,"--index",str(index),"--gpu",str(gpu)]

def verify_one(row,head):
    dest=root_folder(row)
    receipt=read(dest/"XP_T_DAPS4K_ROOT_COMPLETE.json")
    if (receipt.get("status")!="XP_T_DAPS4K_ROOT_PASS" or
        receipt.get("uid")!=row["uid"] or
        receipt.get("dataset")!=row["dataset"] or
        receipt.get("image_id")!=row["image_id"] or
        receipt.get("rep")!=row["rep"] or
        receipt.get("seed")!=row["seed"] or
        receipt.get("runner_head")!=head or
        receipt.get("prep_head")!=PREP_HEAD or
        receipt.get("manifest_sha256")!=MANIFEST_SHA or
        receipt.get("method")!="DAPS-4K" or
        receipt.get("model_load_strict") is not True or
        receipt.get("model_load_trace",{}).get("unet_calls")!=1 or
        receipt.get("model_load_trace",{}).get("strict") is not True or
        receipt.get("model_load_trace",{}).get("missing_keys")!=[] or
        receipt.get("model_load_trace",{}).get("unexpected_keys")!=[] or
        receipt.get("gt_free_generation") is not True or
        receipt.get("historical_candidate_access") is not False or
        receipt.get("evaluation_outcomes_embargo")!="NOT_SCORED" or
        receipt.get("measurement_sha256")!=row["measurement_sha256"] or
        sha(dest/"reconstruction.pt")!=receipt.get("reconstruction_pt_sha256") or
        sha(dest/"reconstruction.png")!=receipt.get("reconstruction_png_sha256")):
        raise RuntimeError("STOP_XPT_DAPS_ROOT_RECEIPT_"+row["uid"])
    for fld in ("native_sampling_wall_s","native_sampling_cuda_event_s",
                "native_model_loading_wall_s"):
        if not 0<float(receipt.get(fld,0))<3600:
            raise RuntimeError("STOP_XPT_DAPS_ROOT_TIME_"+row["uid"])
    if (receipt.get("solver_config",{}).get("annealing_scheduler_config",{})
            .get("num_steps")!=400 or
        receipt.get("solver_config",{}).get("diffusion_scheduler_config",{})
            .get("num_steps")!=10 or
        receipt.get("solver_config",{}).get("mcmc_sampler_config",{})
            .get("num_steps")!=100):
        raise RuntimeError("STOP_XPT_DAPS_NOT_4K")
    return receipt

def qualification(samples,used,errors):
    result={}
    for gpu in sorted(used):
        data=samples.get(gpu,[])
        info=classify_timing(data,gpu)
        active_seen=any(
           any(card.index==gpu and card.process_pids and
               card.utilization_pct>0 for card in observed)
           for observed in data)
        if not active_seen:
            info["timing_valid"]=False
            info["timing_class"]="UNVERIFIED"
            info["reasons"].append("no_active_pid_utilization_observed")
        if errors.get(gpu,0):
            info["timing_valid"]=False
            info["timing_class"]="UNVERIFIED"
            info["reasons"].append("missing_telemetry_samples")
        result[str(gpu)]=info
    return result

def report(head,rows,elapsed,samples,used,errors,smoke):
    receipts=[verify_one(row,head) for row in rows]
    if len(receipts)!=18:
        raise RuntimeError("STOP_XPT_DAPS_FINAL_ROOTS")
    by_dataset={}
    for dataset in ("ffhq","imagenet"):
        selected=[r for r in receipts if r["dataset"]==dataset]
        if len(selected)!=9:
            raise RuntimeError("STOP_XPT_DAPS_DATASET_COUNT")
        by_dataset[dataset]={
           "n_roots":9,"n_images":3,"runs_per_image":3,
           "sampling_wall_s_median":statistics.median(
               x["native_sampling_wall_s"] for x in selected),
           "sampling_cuda_event_s_median":statistics.median(
               x["native_sampling_cuda_event_s"] for x in selected),
           "sampling_wall_s_mean":statistics.mean(
               x["native_sampling_wall_s"] for x in selected),
           "model_loading_wall_s_median":statistics.median(
               x["native_model_loading_wall_s"] for x in selected),
           "gt_scored":False,
        }
    timing=qualification(samples,used,errors)
    total_s=sum(elapsed.values())
    result={
       "status":"XP_T_DAPS4K_BATCH1_PASS",
       "head":head,"prep_head":PREP_HEAD,"manifest_sha256":MANIFEST_SHA,
       "family":"DAPS","method":"DAPS-4K",
       "n_roots":18,"n_images":6,"batch_size":1,
       "workers_per_physical_gpu":1,
       "physical_gpu_used":sorted(used),
       "total_worker_seconds":total_s,"total_gpu_hours":total_s/3600,
       "clean_timing_eligible":all(x["timing_valid"] for x in timing.values()),
       "per_device_timing":timing,
       "smoke_gate":smoke,
       "per_dataset":by_dataset,
       "gt_free_generation":True,
       "all_model_loads_strictly_verified":True,
       "original_measurements_unchanged":True,
       "historical_population_not_extended":True,
       "evaluation_outcomes_embargo":"NOT_SCORED",
       "per_root":[{
            "uid":r["uid"],"dataset":r["dataset"],"rep":r["rep"],
            "native_sampling_wall_s":r["native_sampling_wall_s"],
            "native_sampling_cuda_event_s":r["native_sampling_cuda_event_s"],
            "model_loading_wall_s":r["native_model_loading_wall_s"],
            "peak_gpu_allocated_mib":r["peak_gpu_allocated_mib"],
            "gpu_physical":r["gpu_physical"],
            "model_sha256":r["model_sha256"],
            "measurement_sha256":r["measurement_sha256"],
            "root_receipt_sha256":sha(
                root_folder(next(row for row in rows
                    if row["uid"]==r["uid"]))/"XP_T_DAPS4K_ROOT_COMPLETE.json"),
       } for r in receipts],
    }
    dump_x(RUN/"XP_T_DAPS4K_COMPLETE.json",result)
    return result

def smoke_projection(first,second,cap):
    if first["dataset"]!="ffhq" or second["dataset"]!="imagenet":
        raise RuntimeError("STOP_XPT_DAPS_SMOKE_DATASET")
    # 9 samples per dataset; conservative 1.75x allowance for variance.
    projected=9*(float(first["native_sampling_wall_s"])+
                 float(second["native_sampling_wall_s"]))*1.75/3600
    if not 0<projected<cap:
        raise RuntimeError("STOP_XPT_DAPS_SMOKE_PROJECTED_BUDGET")
    return projected

def main(a):
    if not (5<=a.poll_seconds<=60 and 16000<=a.required_vram_mib<=60000
            and 1<=a.gpu_hour_cap<=12):
        raise RuntimeError("STOP_XPT_DAPS_SETTINGS")
    git=lambda *xs:subprocess.check_output(
        ["git","-C",str(WT),*xs],text=True,timeout=30).strip()
    if git("rev-parse","HEAD")!=a.expected_head or git("status","--porcelain"):
        raise RuntimeError("STOP_XPT_DAPS_WRONG_OR_DIRTY_HEAD")
    if RUN.is_symlink() or not RUN.is_dir():
        raise RuntimeError("STOP_XPT_DAPS_RUN_DIRECTORY")
    for name in ("gpu","pycache","XP_T_DAPS4K_STATUS.json",
                 "XP_T_DAPS4K_COMPLETE.json","GPU_EVENTS.jsonl",
                 "XP_T_DAPS4K_SUPERVISOR_PID.json"):
        if (RUN/name).exists() or (RUN/name).is_symlink():
            raise RuntimeError("STOP_XPT_DAPS_ALREADY_STARTED_"+name)
    if sha(RUN/"XP_T_DAPS4K_TASKS.json")!=MANIFEST_SHA:
        raise RuntimeError("STOP_XPT_DAPS_MANIFEST_SHA")
    frozen=read(RUN/"XP_T_DAPS4K_TASKS.json")
    if frozen.get("head")!=PREP_HEAD or frozen.get("n_roots")!=18:
        raise RuntimeError("STOP_XPT_DAPS_MANIFEST_PREP")
    rows=frozen["rows"]
    (RUN/"gpu").mkdir(exist_ok=False)
    (RUN/"pycache").mkdir(exist_ok=False)
    (CACHE/"tmp").mkdir(parents=True,exist_ok=True)
    dump_x(RUN/"XP_T_DAPS4K_SUPERVISOR_PID.json",{
        "pid":os.getpid(),"head":a.expected_head,
        "manifest_sha256":MANIFEST_SHA,"status":"STARTED"})
    status=RUN/"XP_T_DAPS4K_STATUS.json"
    events=RUN/"GPU_EVENTS.jsonl"
    state={
       "state":"WAIT_GPU","head":a.expected_head,
       "manifest_sha256":MANIFEST_SHA,"total_roots":18,
       "completed_roots":0,"running_roots":0,
       "gpu_hours":0.0,"reason":None,"active_pids":[],
       "phase":"SMOKE","physical_gpus_used":[],
    }
    update(status,state)
    active={};elapsed={};observations={};errors={};used=set()
    done=set();phase="SMOKE";queue=list(SMOKE)
    fail=None;smoke_receipt=None;last_progress=time.monotonic()
    while len(done)<18 or active:
        if (RUN/"STOP_AFTER_ACTIVE").exists():
            fail=fail or "STOP_AFTER_ACTIVE_REQUESTED"
        current_gpu_s=sum(elapsed.values())+sum(
            time.monotonic()-task["started"] for task in active.values())
        if current_gpu_s>a.gpu_hour_cap*3600:
            fail=fail or "GPU_HOUR_CAP"
        try:
            own=[task["proc"].pid for task in active.values()]
            snap=collect(own)
            for gpu,task in active.items():
                observations.setdefault(gpu,[]).append(snap)
                append(events,event(snap,stage="XP-T",
                    job_id="DAPS4K_W1_4GPU",action="HEARTBEAT",
                    selected_gpu=gpu,own_roots=[task["proc"].pid]))
                card=next(x for x in snap if x.index==gpu)
                if card.used_mib>int(0.80*card.total_mib):
                    fail=fail or ("HARD_80PCT_VRAM_"+str(gpu))
            candidates=[c for c in snap if c.index in GPU_INDICES
                        and c.index not in active]
            if queue and not fail:
                for card in candidates:
                    if not queue:break
                    gpu,_=choose_gpu([card],a.required_vram_mib,
                        safety_margin_mib=2048,fraction=0.80,clean_only=True)
                    if gpu is None:continue
                    idx=queue[0]
                    cmd=worker_args(a.expected_head,idx,gpu)
                    env=worker_env(gpu,idx)
                    pre=subprocess.run(cmd+["--preflight-only"],
                        cwd=str(WT),env=env,capture_output=True,
                        text=True,timeout=600)
                    (RUN/f"PREFLIGHT_{idx:02d}.log").write_text(
                        pre.stdout+pre.stderr)
                    if pre.returncode:
                        fail="FAILED_PREFLIGHT_"+str(idx)
                        break
                    before=collect([task["proc"].pid for task in active.values()])
                    observations.setdefault(gpu,[]).append(before)
                    append(events,event(before,stage="XP-T",
                        job_id="DAPS4K_W1_4GPU",action="PRE_GPU",
                        selected_gpu=gpu))
                    with (RUN/f"WORKER_{idx:02d}.log").open("xb") as log:
                        proc=subprocess.Popen(cmd,env=env,cwd=str(WT),
                            stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True)
                    active[gpu]={"index":idx,"proc":proc,
                                 "started":time.monotonic()}
                    used.add(gpu);queue.pop(0)
                    print("XP_T_DAPS4K_WORKER_START",idx,
                          rows[idx]["dataset"],"gpu",gpu,flush=True)
                    state.update(state="RUNNING_GPU",running_roots=len(active),
                        active_pids=[v["proc"].pid for v in active.values()],
                        physical_gpus_used=sorted(used))
                    update(status,state)
        except Exception as e:
            for gpu in active:
                errors[gpu]=errors.get(gpu,0)+1
            append(events,{"action":"GPU_TELEMETRY_ERROR",
                           "reason":str(e)[:200]})
            if time.monotonic()-last_progress>1200:
                fail=fail or "ADMISSION_OR_TELEMETRY_TIMEOUT"
        for gpu,task in list(active.items()):
            proc=task["proc"]
            if proc.poll() is None:continue
            duration=time.monotonic()-task["started"]
            elapsed[gpu]=elapsed.get(gpu,0)+duration
            active.pop(gpu)
            try:
                post=collect([x["proc"].pid for x in active.values()])
                observations.setdefault(gpu,[]).append(post)
                append(events,event(post,stage="XP-T",
                    job_id="DAPS4K_W1_4GPU",action="POST_GPU",selected_gpu=gpu))
            except Exception:
                errors[gpu]=errors.get(gpu,0)+1
            if proc.returncode!=0:
                fail=fail or f"FAILED_WORKER_{task['index']}_RC{proc.returncode}"
            else:
                try:
                    verify_one(rows[task["index"]],a.expected_head)
                    done.add(task["index"])
                    last_progress=time.monotonic()
                    print("XP_T_DAPS4K_WORKER_PASS",
                          task["index"],rows[task["index"]]["dataset"],
                          "gpu",gpu,"worker_s",round(duration,2),flush=True)
                except Exception as e:
                    fail=fail or f"FAILED_ROOT_{task['index']}_{str(e)[:80]}"
            state.update(completed_roots=len(done),
                running_roots=len(active),
                active_pids=[x["proc"].pid for x in active.values()],
                gpu_hours=(sum(elapsed.values())+sum(
                    time.monotonic()-x["started"] for x in active.values()))/3600)
            update(status,state)
        # No more work is submitted between phases until both smoke
        # reconstructions have completed with validated root receipts.
        if phase=="SMOKE" and not active and not queue and not fail:
            if set(done)!={0,9}:
                fail="STOP_XPT_DAPS_INCOMPLETE_SMOKE"
            else:
                first=verify_one(rows[0],a.expected_head)
                second=verify_one(rows[9],a.expected_head)
                try:
                    projected=smoke_projection(first,second,a.gpu_hour_cap)
                    smoke_receipt={
                      "status":"XP_T_DAPS4K_SMOKE_PASS",
                      "indices":[0,9],
                      "native_wall_s_ffhq":first["native_sampling_wall_s"],
                      "native_wall_s_imagenet":second["native_sampling_wall_s"],
                      "conservative_total_projected_gpu_hours":projected,
                      "gpu_hour_cap":a.gpu_hour_cap,
                      "model_loads_strictly_verified":True,
                      "gt_free":True,
                    }
                    dump_x(RUN/"XP_T_DAPS4K_SMOKE_GATE.json",smoke_receipt)
                    phase="MAIN";queue=list(REMAINDER)
                    state.update(state="SMOKE_PASS",phase="MAIN",
                                 projected_gpu_hours=projected)
                    update(status,state)
                    print("XP_T_DAPS4K_SMOKE_PASS",
                          "projected_gpu_hours",round(projected,4),flush=True)
                except Exception as e:
                    fail=str(e)[:160]
        if fail and not active:
            state.update(state="FAILED",reason=fail,phase=phase,
                         gpu_hours=sum(elapsed.values())/3600,
                         running_roots=0,active_pids=[])
            update(status,state)
            print("ACTION NEEDED: XP_T_DAPS4K_STOP "+fail,flush=True)
            return 8
        if not active and queue and time.monotonic()-last_progress>1800:
            state.update(state="WAIT_GPU_TIMEOUT",reason="NO_ADMISSIBLE_GPU")
            update(status,state)
            print("ACTION NEEDED: XP_T_DAPS4K_GPU_TIMEOUT",flush=True)
            return 8
        if active or queue:
            time.sleep(a.poll_seconds)
    if done!=set(range(18)) or smoke_receipt is None:
        raise RuntimeError("STOP_XPT_DAPS_18_ROOTS_NOT_COMPLETE")
    final=report(a.expected_head,rows,elapsed,observations,used,errors,
                 smoke_receipt)
    state.update(state="PASS",reason=None,phase="COMPLETE",
        gpu_hours=final["total_gpu_hours"],
        clean_timing_eligible=final["clean_timing_eligible"],
        per_dataset=final["per_dataset"],running_roots=0,active_pids=[])
    update(status,state)
    print("XP_T_DAPS4K_BATCH1_PASS roots=18",
          "gpu_h",round(final["total_gpu_hours"],5),
          "ffhq_median_s",round(
             final["per_dataset"]["ffhq"]["sampling_wall_s_median"],3),
          "imagenet_median_s",round(
             final["per_dataset"]["imagenet"]["sampling_wall_s_median"],3),
          "clean_timing",final["clean_timing_eligible"],flush=True)
    print("XP_T_DAPS4K_COMPLETE_SHA="+sha(RUN/"XP_T_DAPS4K_COMPLETE.json"),
          flush=True)
    print("OK",flush=True)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--poll-seconds",type=int,default=10)
    p.add_argument("--required-vram-mib",type=int,default=32000)
    p.add_argument("--gpu-hour-cap",type=float,default=6.0)
    try:sys.exit(main(p.parse_args()))
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_T_DAPS4K_SUPERVISOR "+str(e),flush=True)
        sys.exit(8)
