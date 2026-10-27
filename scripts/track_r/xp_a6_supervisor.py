#!/usr/bin/env python3
"""A6 critical overnight controller: D1 phase2 + D3 + E' (all GT-free).

Time-UNAWARE population generation, separately logs time-AWARE GPU telemetry.
One independent native worker/device, smoke per stage then unattended main.
All partial roots are preserved; stop file drains without killing foreign PID.
Durable JSON journal, no blind restart or unvalidated reused candidate.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_a6_prepare_cpu import ROOT,WT,BASE,COUNTS,sha
from scripts.track_r.xp_a6_d1p2_contract import SOURCE_HEAD as D1_HEAD
from scripts.track_r.xp_fast_gpu import (
    frozen_roots,root_folder as fast_folder,validate_complete)
from prdiffusion.track_r.xp_gpu_telemetry import (
    collect,choose_gpu,classify_timing,event)

FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
SITCOM=ROOT/"conda-envs/sitcom_ode_bw/bin/python"
DAPS=ROOT/"conda-envs/daps/bin/python"
CACHE=ROOT/"outputs/.cache/track_r"
STAGES=("D1_P2","D3","EPRIME")
REPS={"D1_P2":12,"D3":4,"EPRIME":4}
TIMING_REFERENCE={"D1_P2":7.191,"D3":58.343,"EPRIME":66.014}
# Calibrated peaks (MiB) + 1.5x model factor + 2 GiB safety.
CALIBRATED_PEAK={"D1_P2":4023,"D3":5053,"EPRIME":2755}
RESERVATION={k:int(math.ceil((v*1.5+2048)/512)*512) for k,v in CALIBRATED_PEAK.items()}
STAGE_GPU_PY={"D1_P2":FAST,"D3":SITCOM,"EPRIME":DAPS}
STAGE_PRI={"D1_P2":0,"D3":1,"EPRIME":2}
SMOKE_COUNTS={"D1_P2":24,"D3":8,"EPRIME":8}
STOP=False

def write_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
def atomic(path,obj):
    path=Path(path)
    temp=path.with_name("."+path.name+"."+str(os.getpid()))
    write_x(temp,obj)
    os.replace(temp,path)
def emit(path,obj):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+"\n")
def pid_alive(pid):
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False
    except PermissionError:return True
def stop_signal(_sig,_frame):
    global STOP
    STOP=True

def jobs():
    out=[]
    for stage in STAGES:
        n=COUNTS[stage]
        reps=REPS[stage]
        if n%reps:raise RuntimeError("STOP_A6_JOB_REP_COUNT")
        if stage=="D1_P2":
            items=[(0,24,True)]
            start=24
            while start<n:
                size=min(120,n-start)
                items.append((start,size,False))
                start+=size
        else:
            items=[(i,1,i<8) for i in range(n)]
        for start,count,smoke in items:
            if (start%reps or count%reps) and stage=="D1_P2":
                raise RuntimeError("STOP_A6_D1P2_ALIGNMENT")
            out.append(dict(stage=stage,id=f"{stage}_{start:05d}_{count:03d}",
                start=start,count=count,smoke=smoke,
                lane="CRITICAL",time_class="TIME_UNAWARE_GENERATION",
                vram_request_mib=RESERVATION[stage]))
    for stage in STAGES:
        ids=[i for j in out if j["stage"]==stage
             for i in range(j["start"],j["start"]+j["count"])]
        if sorted(ids)!=list(range(COUNTS[stage])):
            raise RuntimeError("STOP_A6_PARTITION_"+stage)
    if len({j["id"] for j in out})!=len(out):
        raise RuntimeError("STOP_A6_DUPLICATE_JOB")
    return out

def output_dir(run,job,manifest=None):
    stage=job["stage"]
    if stage=="D1_P2":
        return run/"gpu"/stage/f"chunk_{job['start']:05d}_{job['count']:03d}"
    if manifest is None:
        manifest=json.loads((run/(stage+".json")).read_text())
    row=manifest["rows"][job["start"]]
    token=hashlib.sha256(row["uid"].encode()).hexdigest()[:24]
    return run/"gpu"/("root_"+token)

def run_argv(run,head,job,gpu,preflight=False):
    stage=job["stage"]
    if stage=="D1_P2":
        prep=json.loads((run/"XP_A6_PREP.json").read_text())
        info=prep["stages"][stage]
        cmd=[str(FAST),str(WT/"scripts/track_r/xp_fast_gpu.py"),
             "--expected-head",head,"--roots",info["path"],
             "--roots-sha",info["sha256"],"--roots-source-head",D1_HEAD,
             "--out",str(output_dir(run,job)),"--start",str(job["start"]),
             "--count",str(job["count"]),"--physical-gpu",str(gpu),
             "--dataset","imagenet","--oversample","2.0"]
    else:
        cmd=[str(STAGE_GPU_PY[stage]),
             str(WT/"scripts/track_r/xp_a6_native_worker.py"),
             "--run",str(run),"--expected-head",head,
             "--stage",stage,"--index",str(job["start"]),
             "--gpu",str(gpu)]
    if preflight:cmd.append("--preflight-only")
    return cmd

def env(run,job,gpu):
    e=dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WT),
             PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
             PYTHONPYCACHEPREFIX=str(run/"pycache"/job["id"]),
             XDG_CACHE_HOME=str(CACHE),
             TMPDIR=str(CACHE/"tmp"),CUBLAS_WORKSPACE_CONFIG=":4096:8",
             B20_LF_ENABLE="0",B21_CONT_ENABLE="0")
    return e

def validate(run,head,job):
    mfname=job["stage"]+".json"
    prep=json.loads((run/"XP_A6_PREP.json").read_text())
    info=prep["stages"][job["stage"]]
    if sha(info["path"])!=info["sha256"]:
        raise RuntimeError("STOP_A6_INPUT_DRIFT")
    mf=json.loads(Path(info["path"]).read_text())
    rows=mf["rows"]
    dest=output_dir(run,job,mf)
    native_times=[]
    if job["stage"]=="D1_P2":
        meta=json.loads((dest/"R0_5_CHUNK_META.json").read_text())
        receipt=json.loads((dest/"R0_5_CHUNK_COMPLETE.json").read_text())
        if (meta.get("head")!=head or
            meta.get("root_sha256")!=info["sha256"] or
            meta.get("start")!=job["start"] or
            meta.get("count")!=job["count"] or
            meta.get("uids")!=[r["uid"] for r in rows[job["start"]:job["start"]+job["count"]]]
            or receipt.get("status")!="R0_5_CHUNK_PASS"
            or receipt.get("head")!=head or
            receipt.get("new")!=job["count"] or
            receipt.get("reused")!=0 or
            receipt.get("meta_sha256")!=sha(dest/"R0_5_CHUNK_META.json")):
            raise RuntimeError("STOP_A6_FAST_CHUNK_RECEIPT")
        for row in rows[job["start"]:job["start"]+job["count"]]:
            r=validate_complete(fast_folder(dest,row),row,head,info["sha256"])
            if (r.get("dataset")!="imagenet"
                or r.get("method")!="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5"
                or r.get("rep")!=row["rep"]
                or r.get("gt_access") is not False):
                raise RuntimeError("STOP_A6_D1P2_NATIVE_ROOT")
            native_times.append(float(r["native_sample_wall_seconds"]))
        return native_times
    row=rows[job["start"]]
    key="XP_T_SITCOM_ROOT_COMPLETE.json" if job["stage"]=="D3" else "XP_T_DAPS4K_ROOT_COMPLETE.json"
    receipt=json.loads((dest/key).read_text())
    expected_method="SITCOM" if job["stage"]=="D3" else "DAPS-4K"
    if (receipt.get("status")!=("XP_T_SITCOM_ROOT_PASS" if job["stage"]=="D3" else "XP_T_DAPS4K_ROOT_PASS")
        or receipt.get("runner_head")!=head
        or receipt.get("manifest_sha256")!=info["sha256"]
        or receipt.get("uid")!=row["uid"]
        or receipt.get("seed")!=row["seed"]
        or receipt.get("measurement_sha256")!=row["measurement_sha256"]
        or receipt.get("gt_free_generation") is not True
        or receipt.get("evaluation_outcomes_embargo")!="NOT_SCORED"
        or sha(dest/"reconstruction.pt")!=receipt.get("reconstruction_pt_sha256")
        or sha(dest/"reconstruction.png")!=receipt.get("reconstruction_png_sha256")):
        raise RuntimeError("STOP_A6_NATIVE_ROOT_RECEIPT_"+job["id"])
    trace=receipt.get("strict_load_trace" if job["stage"]=="D3" else "model_load_trace",{})
    if (trace.get("unet_calls")!=1 or trace.get("strict") is not True
        or trace.get("missing_keys")!=[] or trace.get("unexpected_keys")!=[]
        or not receipt.get("model_loaded_strictly" if job["stage"]=="D3" else "model_load_strict",False)):
        raise RuntimeError("STOP_A6_WEIGHT_LOAD_"+job["id"])
    native_times.append(float(receipt["native_sampling_wall_s"]))
    return native_times

def verified_smokes(run,head,completed,all_jobs):
    result={}
    for stage in STAGES:
        chosen=[j for j in all_jobs if j["stage"]==stage and j["smoke"]]
        if not chosen or any(j["id"] not in completed for j in chosen):
            raise RuntimeError("STOP_A6_SMOKE_COVERAGE_"+stage)
        seconds=[s for j in chosen for s in validate(run,head,j)]
        if len(seconds)!=SMOKE_COUNTS[stage]:
            raise RuntimeError("STOP_A6_SMOKE_SAMPLES_"+stage)
        median=statistics.median(seconds)
        if not (0.5*TIMING_REFERENCE[stage]<=median<=1.5*TIMING_REFERENCE[stage]):
            raise RuntimeError("STOP_A6_SMOKE_TIMING_"+stage)
        result[stage]=dict(n_roots=len(seconds),
            median_sampling_wall_seconds=median,
            native_reference_seconds=TIMING_REFERENCE[stage],
            outcome_inspected=False)
    return result

def verify_preparation(run,head):
    run=Path(run)
    if (run.parent!=BASE or not run.name.startswith("xp_a6_")
        or run.is_symlink() or not run.is_dir()):
        raise RuntimeError("STOP_A6_RUN_NAMESPACE")
    source=json.loads((run/"XP_A6_PREP.json").read_text())
    if (source.get("status")!="XP_A6_NEXT_INPUTS_FROZEN"
        or source.get("head")!=head
        or source.get("n_total_new_roots")!=2320
        or source.get("no_gpu_or_gt_scoring_performed") is not True
        or set(source.get("stages",{}))!=set(STAGES)):
        raise RuntimeError("STOP_A6_PREP_CONTRACT")
    for name,proof in source["stages"].items():
        if (proof["n_roots"]!=COUNTS[name]
            or proof["outcomes_embargoed"] is not (name in ("D1_P2","EPRIME"))
            or sha(proof["path"])!=proof["sha256"]):
            raise RuntimeError("STOP_A6_STAGE_SOURCE_"+name)
    actual=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if actual!=head or dirty:
        raise RuntimeError("STOP_A6_GENERATION_GIT")
    return source

def main(a):
    if not (6<=a.hours<=12 and 2<=a.gpu_hours<=50 and 5<=a.poll<=60):
        raise RuntimeError("STOP_A6_CONTROLLER_CAP")
    run=Path(a.run)
    verify_preparation(run,a.expected_head)
    all_jobs=jobs()
    selected=[j for j in all_jobs if j["smoke"]==(a.mode=="smoke")]
    byid={j["id"]:j for j in all_jobs}
    statefile=run/"XP_A6_CONTROLLER_STATE.json"
    ownerfile=run/"XP_A6_ACTIVE_OWNER.json"
    if ownerfile.exists():
        older=json.loads(ownerfile.read_text())
        if pid_alive(older.get("pid",-1)):
            raise RuntimeError("STOP_A6_CONTROLLER_ALREADY_ALIVE")
    if statefile.is_file():
        state=json.loads(statefile.read_text())
        if state.get("head")!=a.expected_head:
            raise RuntimeError("STOP_A6_WRONG_RESUME_HEAD")
        if any(pid_alive(j["pid"]) for j in state.get("active_jobs",[])):
            raise RuntimeError("STOP_A6_LIVE_ORPHANED_CHILD")
        if state.get("active_jobs"):
            raise RuntimeError("STOP_A6_UNVERIFIED_ORPHAN_NEEDS_AUDIT")
        completed=set(state.get("done",[]))
        for jid in completed:
            if jid not in byid:raise RuntimeError("STOP_A6_UNKNOWN_PRIOR_JOB")
            validate(run,a.expected_head,byid[jid])
    else:
        if a.mode!="smoke":
            raise RuntimeError("STOP_A6_SMOKE_FIRST")
        completed=set()
        state=dict(status="STARTED",head=a.expected_head,
           done=[],active_jobs=[],gpu_hours=0.0,
           timing_by_job={},smoke_pass=False,outcomes_scored=False,
           K_gt4_results_sealed=True)
        (run/"gpu").mkdir(exist_ok=False)
        (run/"pycache").mkdir(exist_ok=False)
        (CACHE/"tmp").mkdir(parents=True,exist_ok=True)
    if a.mode=="smoke" and (run/"XP_A6_SMOKE_COMPLETE.json").exists():
        raise RuntimeError("STOP_A6_SMOKE_ALREADY_DONE")
    if a.mode=="main":
        receipt=run/"XP_A6_SMOKE_COMPLETE.json"
        if not state.get("smoke_pass") or not receipt.is_file():
            raise RuntimeError("STOP_A6_SMOKE_NOT_PASSED")
        proof=json.loads(receipt.read_text())
        if proof.get("status")!="XP_A6_ALL_SMOKES_PASS" or proof.get("head")!=a.expected_head:
            raise RuntimeError("STOP_A6_SMOKE_RECEIPT")
        verified_smokes(run,a.expected_head,completed,all_jobs)
    for job in selected:
        path=output_dir(run,job)
        if job["id"] not in completed and (path.exists() or path.is_symlink()):
            raise RuntimeError("STOP_A6_PARTIAL_NEEDS_AUDITED_RESUME_"+job["id"])
    owner=dict(pid=os.getpid(),head=a.expected_head,mode=a.mode,
               admitted_at_unix=time.time())
    if ownerfile.exists():atomic(ownerfile,owner)
    else:write_x(ownerfile,owner)
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    events=run/"XP_A6_GPU_EVENTS.jsonl"
    active={}
    cost=float(state.get("gpu_hours",0))*3600
    started=time.monotonic()
    last_admission_success=time.monotonic()
    terminal_error=None
    print("XP_A6_CONTROLLER_START",a.mode,"stage_count",len(selected),flush=True)
    while True:
        verified=verify_preparation(run,a.expected_head)
        due=(STOP or (run/"STOP_AFTER_ACTIVE").is_file()
             or (time.monotonic()-started)>=a.hours*3600)
        running_cost=sum(time.monotonic()-v["started"] for v in active.values())
        if cost+running_cost>=a.gpu_hours*3600:due=True
        # Drain completed children and hash-validate before journal credit.
        for gpu,task in list(active.items()):
            rc=task["proc"].poll()
            if rc is None:continue
            cost+=time.monotonic()-task["started"]
            del active[gpu]
            measure=classify_timing(task["samples"],gpu)
            state["timing_by_job"][task["job"]["id"]]=measure
            emit(events,dict(action="TIME_AWARE_TELEMETRY_CLASSIFICATION",
                job=task["job"]["id"],gpu=gpu,classification=measure))
            if rc!=0:
                terminal_error=terminal_error or f"WORKER_{task['job']['id']}_RC{rc}"
            else:
                try:
                    vals=validate(run,a.expected_head,task["job"])
                    if len(vals)!=task["job"]["count"] or not all(
                         0<x<2000 and math.isfinite(x) for x in vals):
                        raise RuntimeError("STOP_A6_ROOT_TIMING_OR_COUNT")
                    completed.add(task["job"]["id"])
                    emit(events,dict(action="TIME_UNAWARE_ROOTS_PASS",
                        job=task["job"]["id"],n=len(vals),gpu=gpu))
                    print("XP_A6_JOB_PASS",task["job"]["id"],gpu,flush=True)
                except Exception as exc:
                    terminal_error=terminal_error or f"RECEIPT_{task['job']['id']}_{str(exc)[:100]}"
            state.update(done=sorted(completed),gpu_hours=cost/3600,
                 active_jobs=[{"id":x["job"]["id"],"pid":x["proc"].pid,
                               "gpu":k} for k,x in active.items()])
            atomic(statefile,state)
        if terminal_error:due=True
        if not due:
            try:
                own=[v["proc"].pid for v in active.values()]
                cards=collect(own)
                for card in cards:
                    if card.index in active:
                        active[card.index]["samples"].append(cards)
                        emit(events,event(cards,stage=active[card.index]["job"]["stage"],
                            job_id=active[card.index]["job"]["id"],action="HEARTBEAT",
                            selected_gpu=card.index,own_roots=own))
                        if card.used_mib>card.total_mib*.8:
                            terminal_error=terminal_error or f"VRAM_HARD_GPU{card.index}"
                            due=True
                pending=[x for x in selected
                         if x["id"] not in completed and
                         x["id"] not in {v["job"]["id"] for v in active.values()}]
                pending.sort(key=lambda j:(STAGE_PRI[j["stage"]],j["start"]))
                for card in cards:
                    if not pending or due:break
                    gpu=card.index
                    if gpu not in (0,1,2,3) or gpu in active:continue
                    job=pending[0]
                    g,_=choose_gpu([card],job["vram_request_mib"],
                      safety_margin_mib=2048,fraction=.8,clean_only=True)
                    if g is None:continue
                    environment=env(run,job,gpu)
                    cmd=run_argv(run,a.expected_head,job,gpu)
                    chk=subprocess.run(cmd+["--preflight-only"],env=environment,
                        cwd=str(WT),capture_output=True,text=True,timeout=600)
                    (run/("PREFLIGHT_"+job["id"]+".log")).write_text(chk.stdout+chk.stderr)
                    if chk.returncode:
                        terminal_error="PREFLIGHT_"+job["id"]+"_RC"+str(chk.returncode)
                        due=True;break
                    snap=collect(own)
                    emit(events,event(snap,stage=job["stage"],job_id=job["id"],
                        action="PRE_GPU",selected_gpu=gpu,own_roots=own))
                    log=run/("WORKER_"+job["id"]+".log")
                    with log.open("xb") as f:
                        proc=subprocess.Popen(cmd,env=environment,cwd=str(WT),
                            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    active[gpu]=dict(proc=proc,job=job,started=time.monotonic(),
                                     samples=[snap])
                    pending.pop(0)
                    last_admission_success=time.monotonic()
                    emit(events,dict(action="TIME_UNAWARE_ROOTS_START",job=job["id"],
                        physical_gpu=gpu,pid=proc.pid))
                    print("XP_A6_JOB_START",job["id"],"gpu",gpu,flush=True)
                if pending and not active and time.monotonic()-last_admission_success>1800:
                    terminal_error=terminal_error or "NO_CLEAN_GPU_FOR_30_MIN"
                    due=True
            except Exception as exc:
                terminal_error=terminal_error or "GPU_ADMISSION_"+str(exc)[:130]
                due=True
        state.update(done=sorted(completed),gpu_hours=(
            cost+sum(time.monotonic()-v["started"] for v in active.values()))/3600,
            active_jobs=[dict(id=v["job"]["id"],gpu=gpu,pid=v["proc"].pid)
                         for gpu,v in active.items()],
            completed_roots=sum(byid[x]["count"] for x in completed),
            mode=a.mode,head=a.expected_head,
            timing_unverified_jobs=sum(not x.get("timing_valid",False)
                                       for x in state["timing_by_job"].values()),
            K_gt4_results_sealed=True,outcomes_scored=False)
        if due and not active:
            state.update(status="STOP_"+terminal_error if terminal_error else
                "PAUSED_AT_TIME_OR_BUDGET",reason=terminal_error)
            if not terminal_error and all(j["id"] in completed for j in selected):
                state["status"]="PHASE_PASS"
            atomic(statefile,state);break
        if not active and all(j["id"] in completed for j in selected):
            if a.mode=="smoke":
                gate=verified_smokes(run,a.expected_head,completed,all_jobs)
                write_x(run/"XP_A6_SMOKE_COMPLETE.json",
                    dict(status="XP_A6_ALL_SMOKES_PASS",head=a.expected_head,
                    stage_smoke=gate,gt_scored=False,outcomes_embargoed=True))
                state.update(status="SMOKE_PASS",smoke_pass=True)
            else:
                state.update(status="MAIN_PASS",reason=None)
            atomic(statefile,state);break
        state["status"]="DRAINING_ACTIVE" if due else "RUNNING"
        atomic(statefile,state)
        time.sleep(a.poll)
    print("XP_A6_CONTROLLER_RESULT",a.mode,state["status"],
         "roots",state["completed_roots"],
         "gpu_h",round(state["gpu_hours"],4),flush=True)
    print("XP_A6_STATE_SHA="+sha(statefile),flush=True)
    print("OUTCOMES_EMBARGOED=true",flush=True)
    return 0 if state["status"] in (
       "SMOKE_PASS","MAIN_PASS","PHASE_PASS","PAUSED_AT_TIME_OR_BUDGET") else 8

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--mode",choices=("smoke","main"),required=True)
    p.add_argument("--hours",type=float,default=11.)
    p.add_argument("--gpu-hours",type=float,default=40.)
    p.add_argument("--poll",type=int,default=10)
    try:sys.exit(main(p.parse_args()))
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A6_CONTROLLER "+str(exc),flush=True)
        sys.exit(8)
