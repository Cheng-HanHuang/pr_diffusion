#!/usr/bin/env python3
"""Guarded 4-GPU overnight XP critical-lane controller.

Smokes first, then main. D1 phase2 intentionally PREPARED BUT NOT LAUNCHED
until phase1 A8-like outcome publication and the v2 prediction registry gate.
During that gate, work-conserving ready critical jobs A64 and D2 may run.
Unimplemented background experiments are NEVER treated as ready.
GT/PSNR are never loaded. No foreign process is killed. No blind retries.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_night_prepare_cpu import ROOT,WT,BASE,COUNTS,STAGES
from scripts.track_r.xp_t_fast_prepare_cpu import sha
from scripts.track_r.xp_fast_gpu import frozen_roots,validate_complete,root_folder
from scripts.track_r.xp_t_daps4k_worker_gpu import root_folder as daps_folder
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu,classify_timing,event

FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
DAPS=ROOT/"conda-envs/daps/bin/python"
CACHE=ROOT/"outputs/.cache/track_r"
RUNNER="scripts/track_r/xp_night_supervisor.py"
# W=1 native sampled per-run medians; remain fixed across overnight run.
REFERENCE_S={"D1_P1":7.190894621,"D1_P2":7.190894621,
             "A64":3.32078449,"D2":166.23148921877146}
# FAST peaks are sampled total device used, DAPS peak is torch allocated.
# Conservative allocation ~1.5x peak + 2048 MiB, upward rounded.
MEASURED_PEAK_MIB={"D1_P1":4023,"D1_P2":4023,"A64":1603,"D2":2755}
RESERVE_MIB={k:int(((v*1.5+2048+511)//512)*512)
             for k,v in MEASURED_PEAK_MIB.items()}
REPS={"D1_P1":4,"D1_P2":12,"A64":48,"D2":4}
CRITICAL_READY=("D1_P1","A64","D2")
BLOCKED_UNTIL_REGISTRY=("D1_P2",)
STOP=False

def atomic(p,v):
    p=Path(p);tmp=p.with_name("."+p.name+"."+str(os.getpid()))
    with tmp.open("x") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    os.replace(tmp,p)

def write_x(p,v):
    with Path(p).open("x") as f:
        json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def append(p,v):
    with Path(p).open("a") as f:
        f.write(json.dumps(v,sort_keys=True,allow_nan=False)+"\n")

def job_id(stage,start,count):
    return f"{stage}_{start:05d}_{count:03d}"

def jobs():
    all_jobs=[]
    for stage in STAGES:
        reps=REPS[stage]
        n=COUNTS[stage]
        if n%reps:raise RuntimeError("STOP_NIGHT_BAD_STAGE_DIMENSIONS")
        if stage=="D2":
            entries=[(i,1,i<8) for i in range(n)]
        else:
            smoke=(0,2*reps,True)
            main=[];start=2*reps
            batch=(120//reps)*reps
            while start<n:
                count=min(batch,n-start)
                if count%reps:raise RuntimeError("STOP_NIGHT_NONIMAGE_ALIGNED")
                main.append((start,count,False))
                start+=count
            entries=[smoke]+main
        for start,count,is_smoke in entries:
            all_jobs.append(dict(id=job_id(stage,start,count),stage=stage,
                start=start,count=count,smoke=is_smoke,lane="CRITICAL",
                vram_mib=RESERVE_MIB[stage]))
    for stage in STAGES:
        chosen=[j for j in all_jobs if j["stage"]==stage]
        expanded=[i for job in chosen for i in range(job["start"],job["start"]+job["count"])]
        if len(expanded)!=COUNTS[stage] or sorted(expanded)!=list(range(COUNTS[stage])):
            raise RuntimeError("STOP_NIGHT_JOB_PARTITION")
    return all_jobs

def paths(root,job):
    stage=job["stage"];start=job["start"]
    if stage=="D2":
        # Native DAPS root receipt uses a hash of the immutable uid.
        meta=json.loads((root/"D2.json").read_text())
        uid=meta["rows"][start]["uid"]
        token=hashlib.sha256(uid.encode()).hexdigest()[:24]
        return root/"gpu"/("root_"+token)
    return root/"gpu"/stage/f"chunk_{start:05d}_{job['count']:03d}"

def argv(root,head,job,gpu,preflight=False):
    stage=job["stage"];start=job["start"];count=job["count"]
    prep=json.loads((root/"XP_NIGHT_PREP.json").read_text())
    info=prep["stages"][stage]
    if stage=="D2":
        cmd=[str(DAPS),str(WT/"scripts/track_r/xp_night_d2_worker.py"),
             "--root",str(root),"--expected-head",head,
             "--index",str(start),"--gpu",str(gpu)]
    else:
        cmd=[str(FAST),str(WT/"scripts/track_r/xp_fast_gpu.py"),
             "--expected-head",head,"--roots",info["path"],
             "--roots-sha",info["sha256"],
             "--out",str(paths(root,job)),"--start",str(start),
             "--count",str(count),"--physical-gpu",str(gpu),
             "--dataset","imagenet" if stage!="A64" else "ffhq",
             "--oversample","2.0"]
    if preflight:cmd.append("--preflight-only")
    return cmd

def env(root,job,gpu):
    v=dict(os.environ)
    v.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WT),
        PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(root/"pycache"/job["id"]),
        TMPDIR=str(CACHE/"tmp"),XDG_CACHE_HOME=str(CACHE),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        B20_LF_ENABLE="0",B21_CONT_ENABLE="0")
    return v

def validate_root(root,job,head):
    stage=job["stage"];start=job["start"];count=job["count"]
    prep=json.loads((root/"XP_NIGHT_PREP.json").read_text())
    info=prep["stages"][stage]
    manifest=json.loads(Path(info["path"]).read_text())
    rows=manifest["rows"]
    if sha(info["path"])!=info["sha256"]:
        raise RuntimeError("STOP_NIGHT_STAGE_MANIFEST_DRIFT")
    folder=paths(root,job)
    native_samples=[]
    if stage=="D2":
        row=rows[start]
        # DAPS native receipt layout is UID-hashed, from calibrated worker.
        candidate=paths(root,job)
        receipt=candidate/"XP_T_DAPS4K_ROOT_COMPLETE.json"
        if not receipt.is_file():
            raise RuntimeError("STOP_NIGHT_D2_ROOT_RECEIPT_MISSING")
        data=json.loads(receipt.read_text())
        if (data.get("status")!="XP_T_DAPS4K_ROOT_PASS"
            or data.get("uid")!=row["uid"]
            or data.get("runner_head")!=head
            or data.get("manifest_sha256")!=info["sha256"]
            or data.get("measurement_sha256")!=row["measurement_sha256"]
            or data.get("method")!="DAPS-4K"
            or data.get("model_load_strict") is not True
            or data.get("model_load_trace",{}).get("unet_calls")!=1
            or data.get("model_load_trace",{}).get("strict") is not True
            or data.get("model_load_trace",{}).get("missing_keys")!=[]
            or data.get("model_load_trace",{}).get("unexpected_keys")!=[]
            or data.get("gt_free_generation") is not True
            or data.get("evaluation_outcomes_embargo")!="NOT_SCORED"
            or sha(candidate/"reconstruction.pt")!=data.get("reconstruction_pt_sha256")
            or sha(candidate/"reconstruction.png")!=data.get("reconstruction_png_sha256")):
            raise RuntimeError("STOP_NIGHT_D2_ROOT_INVALID")
        native_samples.append(float(data["native_sampling_wall_s"]))
        return native_samples
    meta=json.loads((folder/"R0_5_CHUNK_META.json").read_text())
    done=json.loads((folder/"R0_5_CHUNK_COMPLETE.json").read_text())
    if (meta.get("head")!=head
        or meta.get("root_sha256")!=info["sha256"]
        or meta.get("start")!=start or meta.get("count")!=count
        or meta.get("uids")!=[r["uid"] for r in rows[start:start+count]]
        or done.get("status")!="R0_5_CHUNK_PASS"
        or done.get("head")!=head
        or done.get("new")!=count
        or done.get("reused")!=0
        or done.get("start")!=start or done.get("count")!=count
        or done.get("meta_sha256")!=sha(folder/"R0_5_CHUNK_META.json")):
        raise RuntimeError("STOP_NIGHT_FAST_CHUNK_RECEIPT")
    for row in rows[start:start+count]:
        r=validate_complete(root_folder(folder,row),row,head,info["sha256"])
        if (r.get("rep")!=row["rep"]
            or r.get("method")!="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5"
            or r.get("dataset")!=manifest["dataset"]
            or r.get("oversample")!=2.0
            or r.get("gt_access") is not False):
            raise RuntimeError("STOP_NIGHT_FAST_ROOT_CONFIG")
        native_samples.append(float(r["native_sample_wall_seconds"]))
    return native_samples

def check_inputs(root,head):
    if (root.parent!=BASE or not root.name.startswith("xp_overnight_")
        or root.is_symlink() or not root.is_dir()):
        raise RuntimeError("STOP_NIGHT_BAD_ROOT")
    prep=json.loads((root/"XP_NIGHT_PREP.json").read_text())
    if (prep.get("status")!="XP_NIGHT_INPUTS_FROZEN"
        or prep.get("head")!=head
        or prep.get("total_new_roots")!=10640
        or prep.get("gpu_work_performed") is not False
        or set(prep.get("stages",{}))!=set(STAGES)):
        raise RuntimeError("STOP_NIGHT_PREP")
    for stage,item in prep["stages"].items():
        if (Path(item["path"])!=root/(stage+".json")
            or sha(item["path"])!=item["sha256"]
            or item["rows"]!=COUNTS[stage]):
            raise RuntimeError("STOP_NIGHT_STAGE_SHA_"+stage)
    actual=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
        text=True,timeout=30).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
        text=True,timeout=30).strip()
    if actual!=head or dirty:
        raise RuntimeError("STOP_NIGHT_GIT")
    return prep

def smoke_verified(root,all_jobs,completed,head):
    outcome={}
    for stage in CRITICAL_READY:
        chosen=[x for x in all_jobs if x["stage"]==stage and x["smoke"]]
        if not chosen or any(x["id"] not in completed for x in chosen):
            raise RuntimeError("STOP_NIGHT_SMOKE_INCOMPLETE_"+stage)
        samples=[t for job in chosen for t in validate_root(root,job,head)]
        # A5's ±50% timing gate applies to the representative stage
        # median, not every stochastic candidate's wall-clock outlier.
        median=statistics.median(samples) if samples else float("nan")
        if not samples or not (
                0.5*REFERENCE_S[stage]<=median<=1.5*REFERENCE_S[stage]):
            raise RuntimeError("STOP_NIGHT_SMOKE_TIMING_"+stage)
        outcome[stage]={"n_new_roots":sum(x["count"] for x in chosen),
             "median_native_sampling_wall_s":statistics.median(samples),
             "calibration_reference_s":REFERENCE_S[stage],
             "source_outcome_inspected":False}
    return outcome

def running_pid(pid):
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def stop_handler(_sig,_frame):
    global STOP
    STOP=True

def main(a):
    if not (8<=a.hours<=12 and 1<=a.gpu_hours<=60 and 5<=a.poll<=60):
        raise RuntimeError("STOP_NIGHT_INVALID_LIMITS")
    root=Path(a.root)
    check_inputs(root,a.expected_head)
    all_jobs=jobs()
    byid={j["id"]:j for j in all_jobs}
    allowed=[j for j in all_jobs if j["stage"] in CRITICAL_READY and
             (j["smoke"] if a.mode=="smoke" else not j["smoke"])]
    state_path=root/"XP_NIGHT_CONTROLLER_STATE.json"
    lock=root/"XP_NIGHT_ACTIVE_OWNER.json"
    if lock.is_file():
        old=json.loads(lock.read_text())
        if running_pid(old.get("pid",-1)):
            raise RuntimeError("STOP_NIGHT_LIVE_CONTROLLER_DO_NOT_INTERRUPT")
    if state_path.is_file():
        state=json.loads(state_path.read_text())
        if state.get("head")!=a.expected_head:
            raise RuntimeError("STOP_NIGHT_UNMATCHED_RESUME_HEAD")
        for j in state.get("active_jobs",[]):
            # A crashed supervisor cannot blindly orphan/kill live CUDA roots.
            if running_pid(j.get("pid",-1)):
                raise RuntimeError("STOP_NIGHT_LIVE_ORPHANED_WORKER")
            if j["id"] not in state["done"]:
                raise RuntimeError("STOP_NIGHT_INCOMPLETE_ORPHAN_AUDIT_REQUIRED")
        completed=set(state.get("done",[]))
        if not completed<=set(byid):
            raise RuntimeError("STOP_NIGHT_STATE_JOB_DRIFT")
        for jid in completed:
            validate_root(root,byid[jid],a.expected_head)
    else:
        if a.mode!="smoke":
            raise RuntimeError("STOP_NIGHT_SMOKE_FIRST")
        state={"status":"RUNNING","head":a.expected_head,
             "done":[],"active_jobs":[],"gpu_hours":0.0,
             "smoke_pass":False,"critical_completed":[],
             "time_aware_ready":list(CRITICAL_READY),
             "background_ready":[],
             "deferred_prediction_gate":list(BLOCKED_UNTIL_REGISTRY)}
        completed=set()
        (root/"gpu").mkdir(exist_ok=False)
        (root/"pycache").mkdir(exist_ok=False)
        (CACHE/"tmp").mkdir(parents=True,exist_ok=True)
    if a.mode=="smoke" and (root/"XP_NIGHT_SMOKE_COMPLETE.json").exists():
        raise RuntimeError("STOP_NIGHT_SMOKE_ALREADY_COMPLETE")
    if a.mode=="main":
        # Confirm all smoke roots from 3 authorized critical stage families,
        # and verify the immutable completed artifact rather than user flag.
        path=root/"XP_NIGHT_SMOKE_COMPLETE.json"
        if not path.is_file() or not state.get("smoke_pass"):
            raise RuntimeError("STOP_NIGHT_NO_PASSED_SMOKE")
        proof=json.loads(path.read_text())
        if (proof.get("status")!="XP_NIGHT_ALL_SMOKES_PASS"
            or proof.get("head")!=a.expected_head or
            set(proof.get("stages",{}))!=set(CRITICAL_READY)):
            raise RuntimeError("STOP_NIGHT_SMOKE_RECEIPT")
        smoke_verified(root,all_jobs,completed,a.expected_head)
    # No automatic retry after a failed or partially written chunk.
    for job in allowed:
        if job["id"] not in completed and (
            paths(root,job).exists() or paths(root,job).is_symlink()):
            raise RuntimeError(
                "STOP_NIGHT_PARTIAL_OUTPUT_NEEDS_AUDITED_RECOVERY_"+job["id"])
    write_x(lock,{"pid":os.getpid(),"head":a.expected_head,"mode":a.mode,
                 "started_unix":time.time()}) if not lock.exists() else atomic(lock,{
                  "pid":os.getpid(),"head":a.expected_head,"mode":a.mode,
                  "started_unix":time.time()})
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    events=root/"XP_NIGHT_EVENTS.jsonl"
    active={};last=time.monotonic();now_start=time.monotonic()
    cost=float(state.get("gpu_hours",0))*3600
    # The main period is independently capped at [8,12] wall hours.
    phase_done=set()
    try:
      while True:
        check_inputs(root,a.expected_head)
        due=time.monotonic()-now_start>=a.hours*3600
        if (root/"STOP_AFTER_ACTIVE").is_file() or STOP:
            due=True
        if cost+sum(time.monotonic()-v["start"] for v in active.values())>=a.gpu_hours*3600:
            due=True
        for gpu,task in list(active.items()):
            rc=task["proc"].poll()
            if rc is not None:
                cost+=time.monotonic()-task["start"]
                del active[gpu]
                # A successful reconstruction can have contaminated timing.
                # Record timing validity independently; do not discard output.
                timing=classify_timing(task.get("samples",[]),gpu)
                state.setdefault("per_job_timing",{})[
                    task["job"]["id"]]=timing
                append(events,{"action":"POST_GPU_TIMING",
                     "id":task["job"]["id"],"gpu":gpu,"timing":timing})
                if rc!=0:
                    state.update(status="STOP_WORKER_FAILURE",
                        reason=task["job"]["id"]+"_rc"+str(rc))
                    due=True
                else:
                    try:
                        samples=validate_root(root,task["job"],a.expected_head)
                        if not all(0<float(x)<1800 for x in samples):
                            raise RuntimeError("STOP_NIGHT_INVALID_NATIVE_TIMES")
                        completed.add(task["job"]["id"])
                        append(events,{"action":"PASS","id":task["job"]["id"],
                                       "gpu":gpu,"n":task["job"]["count"],
                                       "native_s_median":statistics.median(samples)})
                        print("XP_NIGHT_JOB_PASS",task["job"]["id"],
                               "gpu",gpu,flush=True)
                    except Exception as exc:
                        state.update(status="STOP_INVALID_RECEIPT",reason=str(exc)[:150])
                        due=True
                state.update(done=sorted(completed),gpu_hours=cost/3600,
                    active_jobs=[{"id":v["job"]["id"],"pid":v["proc"].pid}
                                 for v in active.values()])
                atomic(state_path,state)
        if str(state.get("status","")).startswith("STOP_"):
            due=True
        if not due:
            try:
                pid_list=[v["proc"].pid for v in active.values()]
                cards=collect(pid_list)
                for card in cards:
                    if card.index in active:
                        own=active[card.index]
                        own.setdefault("samples",[]).append(cards)
                        append(events,event(cards,stage=own["job"]["stage"],
                           job_id=own["job"]["id"],action="HEARTBEAT",
                           selected_gpu=card.index,own_roots=pid_list))
                        if card.used_mib>0.8*card.total_mib:
                            due=True
                            state.update(status="STOP_VRAM_CEILING",
                                         reason=f"gpu{card.index}")
                pending=[j for j in allowed if j["id"] not in completed
                         and j["id"] not in {v["job"]["id"] for v in active.values()}]
                pending.sort(key=lambda z:(CRITICAL_READY.index(z["stage"]),
                                           z["start"]))
                for card in cards:
                    gpu=card.index
                    if due or not pending:break
                    if gpu not in range(4) or gpu in active:continue
                    job=pending[0]
                    candidate,_=choose_gpu([card],job["vram_mib"],
                              safety_margin_mib=2048,fraction=.8,clean_only=True)
                    if candidate is None:continue
                    call=argv(root,a.expected_head,job,gpu)
                    environment=env(root,job,gpu)
                    # CPU-only code/provenance preflight runs with same device
                    # visibility but --preflight-only, no tensor generation.
                    chk=subprocess.run(call+["--preflight-only"],
                          env=environment,cwd=str(WT),capture_output=True,
                          text=True,timeout=600)
                    (root/f"PREFLIGHT_{job['id']}.log").write_text(
                          chk.stdout+chk.stderr)
                    if chk.returncode:
                        due=True
                        state.update(status="STOP_PREFLIGHT",
                            reason=job["id"]+"_rc"+str(chk.returncode))
                        break
                    snap=collect(pid_list)
                    append(events,event(snap,stage=job["stage"],
                        job_id=job["id"],action="PRE_GPU",selected_gpu=gpu,
                        own_roots=pid_list))
                    log=root/f"WORKER_{job['id']}.log"
                    with log.open("xb") as f:
                        proc=subprocess.Popen(call,env=environment,cwd=str(WT),
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    active[gpu]={"proc":proc,"job":job,
                                  "start":time.monotonic(),
                                  "samples":[snap]}
                    pending.pop(0)
                    append(events,{"action":"START","id":job["id"],
                        "physical_gpu":gpu,"pid":proc.pid,
                        "vram_admission_mib":job["vram_mib"]})
                    print("XP_NIGHT_JOB_START",job["id"],"GPU",gpu,flush=True)
                if pending and not active and time.monotonic()-last>1800:
                    due=True;state.update(status="STOP_GPU_UNAVAILABLE",
                                           reason="30m_no_clean_slot")
                if active or pending:last=time.monotonic() if active else last
            except Exception as exc:
                due=True
                state.update(status="STOP_GPU_TELEMETRY_OR_ADMISSION",
                             reason=str(exc)[:160])
        state.update(done=sorted(completed),gpu_hours=(
            cost+sum(time.monotonic()-v["start"] for v in active.values()))/3600,
            active_jobs=[{"id":v["job"]["id"],"pid":v["proc"].pid,
                          "gpu":gpu} for gpu,v in active.items()],
            completed_roots=sum(byid[j]["count"] for j in completed),
            timing_unverified_jobs=sum(
                not rec.get("timing_valid",False) for rec in
                state.get("per_job_timing",{}).values()),
            mode=a.mode,head=a.expected_head,
            outcomes_scored=False,sealed_outcomes_not_inspected=True,
            critical_stages_completed=[
                stage for stage in CRITICAL_READY if
                  all(j["id"] in completed for j in allowed
                      if j["stage"]==stage)])
        if due and not active:
            if state.get("status","") not in (
                "STOP_INVALID_RECEIPT","STOP_WORKER_FAILURE","STOP_VRAM_CEILING",
                "STOP_PREFLIGHT","STOP_GPU_UNAVAILABLE","STOP_GPU_TELEMETRY_OR_ADMISSION"):
                state["status"]="PAUSED_AT_TIME_OR_BUDGET" if any(
                    j["id"] not in completed for j in allowed) else "PASS"
            atomic(state_path,state)
            break
        # A STOP_* gate is sticky across worker-draining iterations.
        # It must never be silently downgraded to a normal RUNNING state.
        if not str(state.get("status","")).startswith("STOP_"):
            state["status"]="DRAINING_ACTIVE" if due else "RUNNING"
        atomic(state_path,state)
        if not active and all(j["id"] in completed for j in allowed):
            if a.mode=="smoke":
                stages=smoke_verified(root,all_jobs,completed,a.expected_head)
                write_x(root/"XP_NIGHT_SMOKE_COMPLETE.json",{
                    "status":"XP_NIGHT_ALL_SMOKES_PASS",
                    "head":a.expected_head,"stages":stages,
                    "no_gt_access":True,
                    "outcomes_scored":False,
                    "stages_excluded_from_overnight_main":{
                      "D1_P2":"awaits D1_P1 analysis/prediction registry",
                      "XP_C_C2":"distinct GT diagnostic worker not yet qualified",
                      "F8":"historical DAPS config and exact replay not certified"}})
                state.update(status="SMOKE_PASS",smoke_pass=True)
            else:
                state.update(status="MAIN_PASS",reason=None)
            atomic(state_path,state)
            break
        time.sleep(a.poll)
      print("XP_NIGHT_STATUS",state["status"],"mode",a.mode,
            "completed_roots",state["completed_roots"],
            "gpu_h",round(state["gpu_hours"],4),flush=True)
      print("XP_NIGHT_RUN_ROOT="+str(root),flush=True)
      print("XP_NIGHT_CONTROLLER_SHA="+sha(state_path),flush=True)
      print("D1_P2_AND_XP_A16_OUTCOMES_SEALED=true",flush=True)
      return 0 if state["status"] in (
          "SMOKE_PASS","MAIN_PASS","PAUSED_AT_TIME_OR_BUDGET") else 8
    finally:
      # Never remove the lock: dead PID is the explicit audited resume signal.
      if active:
          print("STOP_NIGHT_ACTIVE_PIDS_RECONCILE",
                [v["proc"].pid for v in active.values()],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--mode",choices=("smoke","main"),required=True)
    p.add_argument("--hours",type=float,default=11.0)
    p.add_argument("--gpu-hours",type=float,default=46.0)
    p.add_argument("--poll",type=int,default=10)
    try:sys.exit(main(p.parse_args()))
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: NIGHT_SUPERVISOR "+str(e),flush=True)
        sys.exit(8)
