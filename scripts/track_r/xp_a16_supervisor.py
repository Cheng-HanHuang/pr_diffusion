#!/usr/bin/env python3
"""XP-A16 four-GPU FAST candidate generator, strictly GT-free and embargo-safe.

Smoke one 12-root image per frozen cohort. Afterwards submit 36 image-aligned
chunks (up to 60 roots each), one independent native FAST worker per GPU.
No GT import, score, PSNR, selected-image rendering or candidate disclosure.
Outputs are immutable. Stop new admissions on failures; preserve any partial
chunk for exact audited recovery. Never kill foreign workers.
"""
from __future__ import annotations
import argparse,json,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu,classify_timing,event
from scripts.track_r.xp_a16_contract import (
    ROOT,WT,RUN,TASKS,MANIFEST_SHA,PREP_HEAD,TOTAL,validate_a16_source,build_chunks)
from scripts.track_r.xp_fast_gpu import (
    sha,frozen_roots,root_folder,validate_complete)
from scripts.track_r.r0_5_prepare_cpu import native_seed

FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
CACHE=ROOT/"outputs/.cache/track_r"
GPUS=(0,1,2,3)

def write_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def atomic(path,obj):
    path=Path(path)
    temp=path.with_name("."+path.name+"."+str(os.getpid()))
    write_x(temp,obj)
    os.replace(temp,path)

def emit(path,obj):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+"\n")

def chunk_path(start,count):
    return RUN/"gpu"/f"chunk_{start:04d}_{count:02d}"

def chunk_args(head,start,count,gpu):
    return [
      str(FAST),str(WT/"scripts/track_r/xp_fast_gpu.py"),
      "--expected-head",head,
      "--roots",str(TASKS),"--roots-sha",MANIFEST_SHA,
      "--roots-source-head",PREP_HEAD,
      "--out",str(chunk_path(start,count)),
      "--start",str(start),"--count",str(count),
      "--physical-gpu",str(gpu),"--dataset","ffhq",
      "--oversample","2.0"]

def environment(gpu,start):
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WT),
       PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
       PYTHONPYCACHEPREFIX=str(RUN/"pycache"/f"chunk_{start:04d}"),
       TMPDIR=str(CACHE/"tmp"),XDG_CACHE_HOME=str(CACHE),
       CUBLAS_WORKSPACE_CONFIG=":4096:8")
    return env

def validate_chunk(head,rows,start,count,gpu):
    path=chunk_path(start,count)
    metadata=json.loads((path/"R0_5_CHUNK_META.json").read_text())
    result=json.loads((path/"R0_5_CHUNK_COMPLETE.json").read_text())
    if not (result.get("status")=="R0_5_CHUNK_PASS" and
      result.get("start")==start and result.get("count")==count and
      result.get("new")==count and result.get("reused")==0 and
      result.get("head")==head and
      result.get("meta_sha256")==sha(path/"R0_5_CHUNK_META.json") and
      metadata.get("status")=="R0_5_CHUNK_CONFIG" and
      metadata.get("head")==head and
      metadata.get("root_sha256")==MANIFEST_SHA and
      metadata.get("start")==start and metadata.get("count")==count and
      metadata.get("gpu")==gpu and
      metadata.get("uids")==[x["uid"] for x in rows[start:start+count]]):
        raise RuntimeError("STOP_A16_CHUNK_RECEIPT")
    for row in rows[start:start+count]:
        native=validate_complete(root_folder(path,row),row,head,MANIFEST_SHA)
        if (native.get("method")!="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5"
            or native.get("dataset")!="ffhq"
            or native.get("rep")!=row["rep"]
            or native.get("oversample")!=2.0
            or not 0<float(native.get("native_sample_wall_seconds",0))<1200):
            raise RuntimeError("STOP_A16_NATIVE_CONFIG_OR_ROOT_RECEIPT")
    return {"start":start,"count":count,"gpu":gpu,
        "native_chunk_sha256":sha(path/"R0_5_CHUNK_COMPLETE.json"),
        "chunk_metadata_sha256":sha(path/"R0_5_CHUNK_META.json"),
        "worker_allocation_seconds":float(
             result["elapsed_gpu_allocation_seconds"])}

def smoke_gate(receipts,cap):
    if len(receipts)!=2 or receipts[0]["start"]!=0 or receipts[1]["start"]!=1200:
        raise RuntimeError("STOP_A16_SMOKE_COHORTS")
    # 2160 roots / 24 smoke roots, conservative factor for future chunks.
    per_root=sum(x["worker_allocation_seconds"] for x in receipts)/24
    projected=per_root*TOTAL*1.75/3600
    if not 0<projected<cap:
        raise RuntimeError("STOP_A16_SMOKE_GPUH_PROJECTION")
    return projected

def qualify(samples,devices,errors):
    result={}
    for gpu in sorted(devices):
        observations=samples.get(gpu,[])
        info=classify_timing(observations,gpu)
        active_seen=any(
             c.index==gpu and bool(c.process_pids) and c.utilization_pct>0
             for sample in observations for c in sample)
        if not active_seen:
            info["timing_valid"]=False
            info["timing_class"]="UNVERIFIED"
            info["reasons"].append("no_active_pid_utilization_sample")
        if errors.get(gpu,0):
            info["timing_valid"]=False
            info["timing_class"]="UNVERIFIED"
            info["reasons"].append("telemetry_gap")
        result[str(gpu)]=info
    return result

def status_template(head):
    return {"state":"WAIT_GPU","head":head,"manifest_sha256":MANIFEST_SHA,
       "phase":"SMOKE","completed_roots":0,"completed_chunks":0,
       "running_chunks":0,"gpu_hours":0.0,"reason":None,
       "active_pids":[],"physical_gpus_used":[],
       "gt_free_generation":True,"outcomes_embargoed":True}

def main(a):
    if (not 5<=a.poll_seconds<=60 or not 16000<=a.vram_mib<=64000
        or not 3.0<=a.gpu_hour_cap<=12.0):
        raise RuntimeError("STOP_A16_BAD_BUDGET")
    if (subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
            text=True,timeout=30).strip()!=a.expected_head or
        subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
            text=True,timeout=30).strip()):
        raise RuntimeError("STOP_A16_WORKTREE")
    prep=json.loads((RUN/"XP_A16_PREP.json").read_text())
    if (prep.get("status")!="XP_A16_PREP_PASS" or
        prep.get("head")!=PREP_HEAD or prep.get("n_roots")!=TOTAL or
        prep.get("n_images")!=180 or
        prep.get("manifest_sha256")!=MANIFEST_SHA or
        prep.get("gpu_work_performed") is not False or
        prep.get("outcomes_embargoed") is not True or
        sha(TASKS)!=MANIFEST_SHA):
        raise RuntimeError("STOP_A16_FROZEN_PREP")
    manifest=json.loads(TASKS.read_text())
    rows=frozen_roots(manifest,PREP_HEAD)
    smoke,main_chunks=build_chunks()
    validate_a16_source(TASKS,MANIFEST_SHA,manifest,rows,PREP_HEAD,0,12)
    if RUN.is_symlink() or not RUN.is_dir():
        raise RuntimeError("STOP_A16_RUN_DIRECTORY")
    for name in ("gpu","pycache","XP_A16_STATUS.json","GPU_EVENTS.jsonl",
                 "XP_A16_COMPLETE.json","XP_A16_SUPERVISOR_PID.json"):
        if (RUN/name).exists() or (RUN/name).is_symlink():
            raise RuntimeError("STOP_A16_ALREADY_STARTED_"+name)
    (RUN/"gpu").mkdir(exist_ok=False)
    (RUN/"pycache").mkdir(exist_ok=False)
    (CACHE/"tmp").mkdir(parents=True,exist_ok=True)
    write_x(RUN/"XP_A16_SUPERVISOR_PID.json",{
       "status":"STARTED","pid":os.getpid(),
       "head":a.expected_head,"manifest_sha256":MANIFEST_SHA})
    state=status_template(a.expected_head)
    status=RUN/"XP_A16_STATUS.json"
    telemetry=RUN/"GPU_EVENTS.jsonl"
    atomic(status,state)
    queue=list(smoke);phase="SMOKE"
    active={};elapsed={};samples={};errors={};devices=set()
    results={};failed=None;smoke_proof=None
    last_progress=time.monotonic()
    while len(results)<len(smoke)+len(main_chunks) or active:
        if (RUN/"STOP_AFTER_ACTIVE").exists():
            failed=failed or "USER_STOP_AFTER_ACTIVE"
        consumed=sum(elapsed.values())+sum(
           time.monotonic()-v["started"] for v in active.values())
        if consumed>a.gpu_hour_cap*3600:
            failed=failed or "GPU_HOUR_CAP"
        try:
            snap=collect([v["proc"].pid for v in active.values()])
            for gpu,task in active.items():
                samples.setdefault(gpu,[]).append(snap)
                emit(telemetry,event(snap,stage="XP-A16",
                    job_id="FAST_FFHQ_K16",action="HEARTBEAT",
                    selected_gpu=gpu,own_roots=[task["proc"].pid]))
                card=next(x for x in snap if x.index==gpu)
                if card.used_mib>int(card.total_mib*0.8):
                    failed=failed or f"HARD_VRAM_CEILING_GPU{gpu}"
            idle=[c for c in snap if c.index in GPUS and c.index not in active]
            if queue and not failed:
                for card in idle:
                    if not queue:break
                    gpu,_=choose_gpu([card],a.vram_mib,
                         safety_margin_mib=2048,fraction=.8,clean_only=True)
                    if gpu is None:continue
                    start,count=queue[0]
                    cmd=chunk_args(a.expected_head,start,count,gpu)
                    env=environment(gpu,start)
                    # Never create GPU output before passing full source,
                    # model, y, RNG and frozen-run manifest preflight.
                    trial=subprocess.run(cmd+["--preflight-only"],
                        env=env,cwd=str(WT),capture_output=True,
                        text=True,timeout=600)
                    (RUN/f"PREFLIGHT_{start:04d}.log").write_text(
                        trial.stdout+trial.stderr)
                    if trial.returncode:
                        failed=f"FAST_PREFLIGHT_{start:04d}_RC{trial.returncode}"
                        break
                    prior=collect([v["proc"].pid for v in active.values()])
                    samples.setdefault(gpu,[]).append(prior)
                    emit(telemetry,event(prior,stage="XP-A16",
                        job_id="FAST_FFHQ_K16",action="PRE_GPU",
                        selected_gpu=gpu))
                    log=RUN/f"WORKER_{start:04d}.log"
                    with log.open("xb") as f:
                        proc=subprocess.Popen(cmd,env=env,cwd=str(WT),
                          stdout=f,stderr=subprocess.STDOUT,
                          start_new_session=True)
                    active[gpu]={"proc":proc,"start":start,"count":count,
                                 "started":time.monotonic()}
                    devices.add(gpu);queue.pop(0)
                    state.update(state="RUNNING_GPU",phase=phase,
                        running_chunks=len(active),
                        active_pids=[x["proc"].pid for x in active.values()],
                        physical_gpus_used=sorted(devices))
                    atomic(status,state)
                    print("XP_A16_CHUNK_START",start,count,"gpu",gpu,flush=True)
        except Exception as exc:
            for gpu in active:
                errors[gpu]=errors.get(gpu,0)+1
            emit(telemetry,{"action":"TELEMETRY_OR_ADMISSION_ERROR",
                            "message":str(exc)[:160]})
            if time.monotonic()-last_progress>1200:
                failed=failed or "TELEMETRY_OR_ADMISSION_TIMEOUT"
        for gpu,task in list(active.items()):
            proc=task["proc"]
            if proc.poll() is None:continue
            runtime=time.monotonic()-task["started"]
            elapsed[gpu]=elapsed.get(gpu,0.0)+runtime
            del active[gpu]
            try:
                after=collect([x["proc"].pid for x in active.values()])
                samples.setdefault(gpu,[]).append(after)
                emit(telemetry,event(after,stage="XP-A16",
                    job_id="FAST_FFHQ_K16",action="POST_GPU",selected_gpu=gpu))
            except Exception:
                errors[gpu]=errors.get(gpu,0)+1
            if proc.returncode:
                failed=failed or (
                    f"FAST_WORKER_{task['start']:04d}_RC{proc.returncode}")
            else:
                try:
                    receipt=validate_chunk(a.expected_head,rows,
                        task["start"],task["count"],gpu)
                    receipt["worker_process_seconds"]=runtime
                    results[task["start"]]=receipt
                    last_progress=time.monotonic()
                    print("XP_A16_CHUNK_PASS",task["start"],task["count"],
                          "gpu",gpu,flush=True)
                except Exception as exc:
                    failed=failed or (
                        f"FAST_CHUNK_VALIDATION_{task['start']:04d}_{str(exc)[:70]}")
            state.update(
                completed_roots=sum(x["count"] for x in results.values()),
                completed_chunks=len(results),running_chunks=len(active),
                active_pids=[v["proc"].pid for v in active.values()],
                gpu_hours=(sum(elapsed.values())+sum(
                    time.monotonic()-v["started"] for v in active.values()))/3600)
            atomic(status,state)
        if phase=="SMOKE" and not active and not queue and not failed:
            try:
                baseline=[results[i] for i,_ in smoke]
                projected=smoke_gate(baseline,a.gpu_hour_cap)
                smoke_proof={"status":"XP_A16_SMOKE_PASS",
                   "image_aligned_chosen_starts":[0,1200],
                   "new_roots":24,"new_image_counts":{"S1_b22":1,
                                                      "S2_b24_dev80":1},
                   "conservative_projected_gpu_hours":projected,
                   "gpu_hour_cap":a.gpu_hour_cap,
                   "native_fast_replay_preserved":True,
                   "gt_read":False,"outcomes_embargoed":True}
                write_x(RUN/"XP_A16_SMOKE_GATE.json",smoke_proof)
                phase="MAIN";queue=list(main_chunks)
                state.update(state="SMOKE_PASS",phase=phase,
                    projected_gpu_hours=projected)
                atomic(status,state)
                print("XP_A16_SMOKE_PASS",
                    "projected_gpu_h",round(projected,4),flush=True)
            except Exception as exc:
                failed=failed or ("SMOKE_GATE_"+str(exc)[:100])
        if failed and not active:
            state.update(state="FAILED",reason=failed,phase=phase,
                active_pids=[],running_chunks=0,
                gpu_hours=sum(elapsed.values())/3600)
            atomic(status,state)
            print("ACTION NEEDED: XP_A16_STOP "+failed,flush=True)
            return 8
        if not active and queue and time.monotonic()-last_progress>1800:
            state.update(state="STOP_ADMISSION_TIMEOUT",
                         reason="NO_ADMISSIBLE_GPU")
            atomic(status,state)
            print("ACTION NEEDED: XP_A16_ADMISSION_TIMEOUT",flush=True)
            return 8
        if active or queue:time.sleep(a.poll_seconds)
    if len(results)!=38 or sum(x["count"] for x in results.values())!=TOTAL:
        raise RuntimeError("STOP_A16_COMPLETION_ACCOUNTING")
    timing=qualify(samples,devices,errors)
    total_gpu_seconds=sum(elapsed.values())
    complete={
      "status":"XP_A16_GENERATION_PASS","stage":"XP-A16",
      "head":a.expected_head,"prep_head":PREP_HEAD,
      "manifest_sha256":MANIFEST_SHA,"n_images":180,
      "historical_roots_reused_without_regeneration":720,
      "new_roots":TOTAL,"n_chunks":38,"n_smoke_roots":24,
      "run_indices":[4,5,6,7,8,9,10,11,12,13,14,15],
      "per_cohort_new_roots":{"S1_b22":1200,"S2_b24_dev80":960},
      "total_gpu_hours":total_gpu_seconds/3600,
      "total_gpu_seconds":total_gpu_seconds,
      "gpu_hour_cap":a.gpu_hour_cap,
      "workers_per_physical_gpu":1,
      "physical_gpu_used":sorted(devices),
      "per_device_timing":timing,
      "clean_timing_eligible":all(x["timing_valid"] for x in timing.values()),
      "strict_gt_free_generation":True,
      "historical_output_paths_not_supplied":True,
      "quality_metrics_generated":False,
      "quality_scoring_authorized":False,
      "outcomes_embargoed":True,
      "theory_freeze_required_for_scoring":"TH1_PREDICTIONS_v1.json",
      "smoke_gate":smoke_proof,
      "chunks":[results[start] for start in sorted(results)],
    }
    write_x(RUN/"XP_A16_GENERATION_COMPLETE.json",complete)
    state.update(state="PASS",phase="COMPLETE",reason=None,
        gpu_hours=complete["total_gpu_hours"],
        running_chunks=0,active_pids=[],
        clean_timing_eligible=complete["clean_timing_eligible"])
    atomic(status,state)
    print("XP_A16_GENERATION_PASS roots=2160 chunks=38",
          "gpu_h",round(complete["total_gpu_hours"],5),
          "clean_timing",complete["clean_timing_eligible"],flush=True)
    print("XP_A16_COMPLETION_SHA="+sha(
          RUN/"XP_A16_GENERATION_COMPLETE.json"),flush=True)
    print("OUTCOMES_EMBARGOED=true",flush=True)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--poll-seconds",type=int,default=10)
    p.add_argument("--vram-mib",type=int,default=32000)
    p.add_argument("--gpu-hour-cap",type=float,default=12.0)
    try:sys.exit(main(p.parse_args()))
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A16_SUPERVISOR "+str(exc),flush=True)
        sys.exit(8)
