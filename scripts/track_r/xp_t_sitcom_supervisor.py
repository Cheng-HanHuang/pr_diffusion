#!/usr/bin/env python3
"""Four-physical-GPU SITCOM XP-T W=1 calibration, isolated & GT-free.

18 native independent-seed candidate workers, one process at a time per GPU.
No ground truth or historical candidates. No worker co-tenancy on same GPU.
Preserves GPU occupancy samples, verifies roots, and fails closed on drift.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
from prdiffusion.track_r.xp_gpu_telemetry import (
    collect,choose_gpu,classify_timing,event)
from scripts.track_r.xp_t_sitcom_worker_gpu import (
    ROOT,WT,BASE,RUN,NATIVE_PY,PREP_HEAD,MANIFEST_SHA,
    immutable_manifest,checked_row,record_folder,sha,safe)

CPU=ROOT/"conda-envs/prdiff_track_r/bin/python"
CACHE=ROOT/"outputs/.cache/track_r"
GPU_INDICES=(0,1,2,3)
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def dump_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def update(path,data):
    p=Path(path)
    temp=p.with_name("."+p.name+"."+str(os.getpid()))
    dump_x(temp,data)
    os.replace(temp,p)

def append(path,record):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")

def job_env(gpu,index):
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
    )
    return env

def worker_args(head,index,gpu):
    return [str(NATIVE_PY),
      str(WT/"scripts/track_r/xp_t_sitcom_worker_gpu.py"),
      "--expected-head",head,"--index",str(index),"--gpu",str(gpu)]

def verify_one(row,head):
    dest=record_folder(row)
    r=json.loads(safe(dest/"XP_T_SITCOM_ROOT_COMPLETE.json").read_text())
    if (r.get("status")!="XP_T_SITCOM_ROOT_PASS" or
        r.get("runner_head")!=head or r.get("prep_head")!=PREP_HEAD or
        r.get("manifest_sha256")!=MANIFEST_SHA or
        r.get("uid")!=row["uid"] or
        r.get("dataset")!=row["dataset"] or
        r.get("rep")!=row["rep"] or
        r.get("seed")!=row["seed"] or
        r.get("model_loaded_strictly") is not True or
        r.get("gt_free_generation") is not True or
        r.get("historical_candidate_access") is not False or
        r.get("evaluation_outcomes_embargo")!="NOT_SCORED" or
        sha(dest/"reconstruction.pt")!=r.get("reconstruction_pt_sha256") or
        sha(dest/"reconstruction.png")!=r.get("reconstruction_png_sha256")):
        raise RuntimeError("STOP_XPT_SITCOM_ROOT_RECEIPT "+row["uid"])
    for k in ("native_sampling_wall_s","native_sampling_cuda_event_s"):
        if not 0<float(r.get(k,0))<3600:
            raise RuntimeError("STOP_XPT_SITCOM_ROOT_TIMING "+row["uid"])
    return r

def finish_report(head,rows,runtime_s,observations,events,starts,errors):
    complete=[verify_one(r,head) for r in rows]
    if len(complete)!=18:
        raise RuntimeError("STOP_XPT_SITCOM_18_EXPECTED")
    timing={}
    for gpu in sorted(starts):
        measures=observations.get(gpu,[])
        result=classify_timing(measures,gpu)
        worker_seen=False
        for entry in measures:
            card=next((x for x in entry if x.index==gpu),None)
            if card is not None and card.process_pids and card.utilization_pct>0:
                worker_seen=True
        if not worker_seen:
            result["timing_class"]="UNVERIFIED"
            result["timing_valid"]=False
            result["reasons"].append("no_active_pid_utilization_observed")
        if errors.get(gpu,0):
            result["timing_class"]="UNVERIFIED"
            result["timing_valid"]=False
            result["reasons"].append("telemetry_gap")
        timing[str(gpu)]=result
    wall=[r["native_sampling_wall_s"] for r in complete]
    cuda=[r["native_sampling_cuda_event_s"] for r in complete]
    per_dataset={}
    for dataset in ("ffhq","imagenet"):
        selected=[r for r in complete if r["dataset"]==dataset]
        if len(selected)!=9:
            raise RuntimeError("STOP_XPT_SITCOM_DATASET_COUNT")
        per_dataset[dataset]={
          "n_roots":9,"n_images":3,"runs_per_image":3,
          "sampling_wall_s_median":statistics.median(
              r["native_sampling_wall_s"] for r in selected),
          "sampling_cuda_event_s_median":statistics.median(
              r["native_sampling_cuda_event_s"] for r in selected),
          "model_loading_wall_s_median":statistics.median(
              r["native_model_loading_wall_s"] for r in selected),
          "sampling_wall_s_mean":statistics.mean(
              r["native_sampling_wall_s"] for r in selected),
          "gt_access":False,
        }
    clean=all(q["timing_valid"] for q in timing.values())
    total_s=sum(float(v) for v in runtime_s.values())
    report={
      "status":"XP_T_SITCOM_BATCH1_PASS",
      "head":head,"prep_head":PREP_HEAD,"manifest_sha256":MANIFEST_SHA,
      "family":"SITCOM","n_roots":18,"n_images":6,
      "workers_per_physical_gpu":1,
      "physical_gpu_used":sorted(starts),
      "total_worker_seconds":total_s,"total_gpu_hours":total_s/3600,
      "clean_timing_eligible":clean,
      "per_device_timing":timing,
      "per_dataset":per_dataset,
      "sampling_wall_s_median":statistics.median(wall),
      "sampling_cuda_event_s_median":statistics.median(cuda),
      "gt_free_generation":True,
      "all_model_loads_strictly_verified":True,
      "historical_population_not_extended":True,
      "evaluation_outcomes_embargo":"NOT_SCORED",
      "per_root":[{
         "uid":r["uid"],"dataset":r["dataset"],"rep":r["rep"],
         "native_sampling_wall_s":r["native_sampling_wall_s"],
         "native_sampling_cuda_event_s":r["native_sampling_cuda_event_s"],
         "model_loading_wall_s":r["native_model_loading_wall_s"],
         "gpu_physical":r["gpu_physical"],
         "model_sha256":r["model_sha256"],
         "measurement_sha256":r["measurement_sha256"],
         "root_receipt_sha256":sha(
             record_folder(next(x for x in rows if x["uid"]==r["uid"]))/
             "XP_T_SITCOM_ROOT_COMPLETE.json"),
      } for r in complete],
    }
    dump_x(RUN/"XP_T_SITCOM_COMPLETE.json",report)
    return report

def main(a):
    if not (5<=a.poll_seconds<=60 and
            12000<=a.required_vram_mib<=60000 and
            0.5<=a.gpu_hour_cap<=5.0):
        raise RuntimeError("STOP_XPT_SITCOM_PARAMETERS")
    git=lambda *args:subprocess.check_output(
         ["git","-C",str(WT),*args],text=True,timeout=30).strip()
    if (git("rev-parse","HEAD")!=a.expected_head or
        git("status","--porcelain")):
        raise RuntimeError("STOP_XPT_SITCOM_SUPERVISOR_HEAD")
    if RUN.is_symlink() or not RUN.is_dir():
        raise RuntimeError("STOP_XPT_SITCOM_RUN")
    for name in ("XP_T_SITCOM_STATUS.json","GPU_EVENTS.jsonl",
                 "XP_T_SITCOM_COMPLETE.json","XP_T_SITCOM_SUPERVISOR_PID.json"):
        if (RUN/name).exists():
            raise RuntimeError("STOP_XPT_SITCOM_ALREADY_STARTED")
    if sha(RUN/"XP_T_SITCOM_TASKS.json")!=MANIFEST_SHA:
        raise RuntimeError("STOP_XPT_SITCOM_MANIFEST_SHA")
    mf=json.loads(safe(RUN/"XP_T_SITCOM_TASKS.json").read_text())
    if mf.get("head")!=PREP_HEAD or mf.get("n_roots")!=18:
        raise RuntimeError("STOP_XPT_SITCOM_FROZEN_PREP")
    rows=mf["rows"]
    # Do not create any GPU worker directory unless all exact source
    # files are frozen and complete and no live run already exists.
    if (RUN/"gpu").exists() or (RUN/"pycache").exists():
        raise RuntimeError("STOP_XPT_SITCOM_EXISTING_GPU_OUTPUT")
    (RUN/"gpu").mkdir(exist_ok=False)
    (RUN/"pycache").mkdir(exist_ok=False)
    (CACHE/"tmp").mkdir(parents=True,exist_ok=True)
    dump_x(RUN/"XP_T_SITCOM_SUPERVISOR_PID.json",{
       "status":"STARTED","pid":os.getpid(),"head":a.expected_head,
       "manifest_sha256":MANIFEST_SHA})
    state={"state":"WAIT_VRAM","head":a.expected_head,
           "manifest_sha256":MANIFEST_SHA,"total_roots":18,
           "completed_roots":0,"running_roots":0,
           "gpu_hours":0.0,"reason":None,"active_pids":[],
           "physical_gpus_used":[]}
    status=RUN/"XP_T_SITCOM_STATUS.json"
    events=RUN/"GPU_EVENTS.jsonl"
    update(status,state)
    active={}
    elapsed_gpu_s={}
    observations={}
    errors={}
    started=set()
    next_i=0;fail_reason=None
    start_wait=time.monotonic()
    while next_i<18 or active:
        if (RUN/"STOP_AFTER_ACTIVE").exists():
            fail_reason="STOP_AFTER_ACTIVE_REQUESTED"
        if (sum(elapsed_gpu_s.values())+
            sum(time.monotonic()-task["start"] for task in active.values())
            >a.gpu_hour_cap*3600):
            fail_reason="GPU_HOUR_BUDGET"
        try:
            snap=collect([x["process"].pid for x in active.values()])
            for gpu,task in active.items():
                observations.setdefault(gpu,[]).append(snap)
                append(events,event(snap,stage="XP-T",job_id="SITCOM_W1_4GPU",
                    action="HEARTBEAT",selected_gpu=gpu,
                    own_roots=[task["process"].pid]))
                card=next(x for x in snap if x.index==gpu)
                if card.used_mib>int(card.total_mib*0.80):
                    fail_reason="HARD_80PCT_VRAM_"+str(gpu)
            candidates=[c for c in snap if c.index in GPU_INDICES and
                        c.index not in active]
            if next_i<18 and not fail_reason and candidates:
                # Fill a genuinely clean and uncongested GPU only. At most
                # one native worker per physical GPU, so per-root timing W=1.
                for card in candidates:
                    if next_i>=18:break
                    gpu,_=choose_gpu([card],a.required_vram_mib,
                         safety_margin_mib=2048,fraction=.80,clean_only=True)
                    if gpu is None:continue
                    idx=next_i
                    row=rows[idx]
                    cmd=worker_args(a.expected_head,idx,gpu)
                    env=job_env(gpu,idx)
                    pre=subprocess.run(cmd+["--preflight-only"],
                         cwd=str(WT),env=env,capture_output=True,text=True,
                         timeout=300)
                    (RUN/f"PREFLIGHT_{idx:02d}.log").write_text(
                         pre.stdout+pre.stderr)
                    if pre.returncode:
                        fail_reason="WORKER_PREFLIGHT_"+str(idx)
                        break
                    before=collect([x["process"].pid for x in active.values()])
                    observations.setdefault(gpu,[]).append(before)
                    append(events,event(before,stage="XP-T",
                        job_id="SITCOM_W1_4GPU",
                        action="PRE_GPU",selected_gpu=gpu))
                    path=RUN/f"WORKER_{idx:02d}.log"
                    with path.open("xb") as f:
                        proc=subprocess.Popen(cmd,env=env,cwd=str(WT),
                             stdout=f,stderr=subprocess.STDOUT,
                             start_new_session=True)
                    active[gpu]={"index":idx,"process":proc,
                                 "start":time.monotonic()}
                    started.add(gpu)
                    next_i+=1
                    state.update(state="RUNNING_GPU",reason=None,
                        running_roots=len(active),active_pids=[
                            v["process"].pid for v in active.values()],
                        physical_gpus_used=sorted(started))
                    update(status,state)
        except Exception as exc:
            if not active and not fail_reason:
                # Do not wait indefinitely if NVML or process telemetry fails.
                if time.monotonic()-start_wait>900:
                    fail_reason="GPU_ADMISSION_TELEMETRY_TIMEOUT"
            for gpu in active:
                errors[gpu]=errors.get(gpu,0)+1
            append(events,{"action":"TELEMETRY_ERROR",
                           "error":str(exc)[:180]})
        for gpu,task in list(active.items()):
            proc=task["process"]
            if proc.poll() is None:continue
            wall=time.monotonic()-task["start"]
            elapsed_gpu_s[gpu]=elapsed_gpu_s.get(gpu,0.0)+wall
            active.pop(gpu)
            try:
                after=collect([x["process"].pid for x in active.values()])
                observations.setdefault(gpu,[]).append(after)
                append(events,event(after,stage="XP-T",
                    job_id="SITCOM_W1_4GPU",action="POST_GPU",
                    selected_gpu=gpu))
            except Exception as exc:
                errors[gpu]=errors.get(gpu,0)+1
            if proc.returncode:
                fail_reason=fail_reason or ("WORKER_"+str(task["index"])+
                                           "_RC_"+str(proc.returncode))
            else:
                try:
                    verify_one(rows[task["index"]],a.expected_head)
                    state["completed_roots"]+=1
                    print("XP_T_SITCOM_WORKER_PASS",
                          task["index"],rows[task["index"]]["dataset"],
                          "gpu",gpu,"wall",round(wall,1),flush=True)
                except Exception as exc:
                    fail_reason=fail_reason or ("RECEIPT_"+str(task["index"])+
                                              "_"+str(exc)[:70])
            state.update(running_roots=len(active),
                         active_pids=[v["process"].pid for v in active.values()],
                         gpu_hours=(sum(elapsed_gpu_s.values())+
                            sum(time.monotonic()-v["start"] for v in active.values()))
                            /3600,physical_gpus_used=sorted(started))
            update(status,state)
        if fail_reason and not active:
            state.update(state="FAILED",reason=fail_reason,
                         gpu_hours=sum(elapsed_gpu_s.values())/3600)
            update(status,state)
            print("ACTION NEEDED: XP_T_SITCOM_STOP "+fail_reason,flush=True)
            return 8
        if not active and next_i<18 and time.monotonic()-start_wait>1800:
            state.update(state="STOP_ADMISSION_TIMEOUT",reason="NO_FREE_GPUS")
            update(status,state)
            print("ACTION NEEDED: XP_T_SITCOM_NO_ADMISSION",flush=True)
            return 8
        if active or next_i<18:time.sleep(a.poll_seconds)
    if state["completed_roots"]!=18:
        raise RuntimeError("STOP_XPT_SITCOM_NOT_18_COMPLETE")
    report=finish_report(a.expected_head,rows,elapsed_gpu_s,
                         observations,events,started,errors)
    state.update(state="PASS",running_roots=0,active_pids=[],
                 reason=None,gpu_hours=report["total_gpu_hours"],
                 clean_timing_eligible=report["clean_timing_eligible"],
                 per_dataset=report["per_dataset"])
    update(status,state)
    print("XP_T_SITCOM_BATCH1_PASS n=18 gpu_h="+str(round(report["total_gpu_hours"],5))+
          " ffhq_median_s="+str(round(
             report["per_dataset"]["ffhq"]["sampling_wall_s_median"],3))+
          " imagenet_median_s="+str(round(
             report["per_dataset"]["imagenet"]["sampling_wall_s_median"],3))+
          " timing_clean="+str(report["clean_timing_eligible"]),flush=True)
    print("XP_T_SITCOM_COMPLETE_SHA="+sha(RUN/"XP_T_SITCOM_COMPLETE.json"),flush=True)
    print("OK",flush=True)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--poll-seconds",type=int,default=10)
    p.add_argument("--required-vram-mib",type=int,default=24000)
    p.add_argument("--gpu-hour-cap",type=float,default=2.0)
    try:sys.exit(main(p.parse_args()))
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: XP_T_SITCOM_SUPERVISOR "+str(exc),flush=True)
        sys.exit(8)
