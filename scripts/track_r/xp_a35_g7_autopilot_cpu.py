#!/usr/bin/env python3
"""Autonomous A23 G7 remaining GT-only pipeline: all8 -> 2-forward -> UnivFD.

Safe flow: read-only frozen preflight -> native charged two-dataset canary ->
per-policy cost projection -> one persistent model worker per admissible GPU
-> exact 280 immutable receipts -> CPU readout -> separate GitHub publication
-> next registered policy. Never bypass a failed gate or increase total G7
10 GPU-hour limit. No inverse-solver sampling, no main-text modification.
"""
from __future__ import annotations
import argparse,fcntl,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
from prdiffusion.track_r.xp_gpu_telemetry import collect
from scripts.track_r.xp_a29_a21_controller import total_union
from scripts.track_r.xp_a35_g7_worker import (
 ROOT,BASE,WT,LOCK,GT,STATUS,read,sha,git,gate
)
from scripts.track_r.xp_a35_g7_readout_publish_cpu import analyze,publish,check_results

METHODS=("S_mid_all8","2_forward","UnivFD")
PRIOR_TOTAL_G7_H=5.1570269519018215
PRIOR_PROJECT_H=246.9875918177077
G7_LIMIT=10.
PROJECT_LIMIT=400.
# Conservative method caps sum to less than the remaining 4.843 GPUh;
# real runtime plus canaries and failures are charged under each method.
METHOD_CAPS={"S_mid_all8":3.75,"2_forward":0.85,"UnivFD":0.}
N=280
STOP=False
WORKER=WT/"scripts/track_r/xp_a35_g7_worker.py"
GLOBAL_LOCK=BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock"
A34_RETURN=ROOT/"pr_diffusion_track_r_a23_g7_smid_pub_20261104T080759Z"
A34_PUB="da130b50856fd6c495fe73049651b746ab67b56b"
A34_ROOT=BASE/"xp_a34_a23_g7_smid_prod_20261104T072943Z"
HARD_CUTOFF=datetime(2026,11,15,12,0,tzinfo=ZoneInfo("America/Detroit"))

def stopping(*_):
    global STOP
    STOP=True

def check(ok,code):
    if not ok:raise RuntimeError("STOP_A35_G7_"+code)

def atomic(path,d):
    temp=path.with_name("."+path.name+".tmp")
    temp.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temp.replace(path)

def pinned():
    spec=read(LOCK)
    check(spec["status"]=="A23_G7_REMAINING_FROZEN_BEFORE_PRODUCTION"
          and spec["methods"]==list(METHODS)
          and sha(GT)==spec["GT_manifest_sha256"]
          and spec["preceding_method"]["publication_commit"]==A34_PUB
          and abs(spec["preceding_method"]["published_G7_total_gpu_h"]-
                  PRIOR_TOTAL_G7_H)<1e-9,
          "A23_SCIENCE_LOCK")
    check(git(A34_RETURN,"rev-parse","HEAD")==A34_PUB and
          not git(A34_RETURN,"status","--porcelain"),
          "ORIGINAL_A34_PUBLICATION_CLEAN")
    a=read(A34_ROOT/"XP_A34_G7_S_MID_FINAL_STATUS.json")
    check(a["status"]=="A34_G7_S_MID_GLOBAL2_280_COMPLETE"
          and a["n_complete"]==280 and a["workers_active"]==[]
          and abs(a["G7_total_charged_gpu_h"]-PRIOR_TOTAL_G7_H)<1e-9
          and abs(a["project_charged_gpu_h"]-PRIOR_PROJECT_H)<1e-9,
          "G7_PREVIOUS_EXACT_FINAL_LEDGER")
    p=A34_RETURN/"docs/track_r/executors/xp/returns/XP_A23_G7_S_MID_GLOBAL2_20261104T080759Z/XP_A23_G7_S_MID_GLOBAL2_PUBLICATION.json"
    pub=read(p)
    check(pub["status"]=="A23_G7_S_MID_GLOBAL2_GT_280_ONLY_PUBLISHED"
          and pub["n_images"]==280
          and pub["source_GT_manifest_sha256"]==spec["GT_manifest_sha256"],
          "ORIGINAL_A34_PUBLISHED_BYTES")
    # Do not restart if a later project stage has consumed additional hours.
    for path in BASE.glob("xp_*/XP_*FINAL_STATUS.json"):
        if path==A34_ROOT/"XP_A34_G7_S_MID_FINAL_STATUS.json":continue
        try:old=read(path)
        except (ValueError,OSError):continue
        value=old.get("project_gpu_h",
            old.get("project_charged_gpu_h"))
        if isinstance(value,(float,int)):
            check(value<=PRIOR_PROJECT_H+1e-8,
                  "FRESH_PROJECT_LEDGER_HAS_NEWER_CHARGES")
    return spec

def new_state(run,head):
    return dict(status="A35_G7_START",head=head,
        source_lock_sha256=sha(LOCK), GT_manifest_sha256=sha(GT),
        source_A34_commit=A34_PUB,
        previous_G7_h=PRIOR_TOTAL_G7_H,previous_project_h=PRIOR_PROJECT_H,
        method_order=list(METHODS),method_results={},
        stage_cpu_wall_s={},charged_gpu_h=0.,
        G7_total_gpu_h=PRIOR_TOTAL_G7_H,
        project_gpu_h=PRIOR_PROJECT_H,
        all_methods_completed=False,
        no_A21_10=True,GT_diagnostic_only=True,
        no_inverse_solver_generation=True,supplement_only=True,
        active_workers=[],not_run=list(METHODS))

def gpu_admissible():
    cards=collect(own_roots=[])
    result=[]
    for c in cards:
        if c.index not in (0,1,2,3) or c.visibility!="known":continue
        need=28672
        if c.free_mib>=need and c.used_mib+need<=int(.8*c.total_mib):
            result.append((c.index,c.free_mib,c.used_mib))
    return sorted(result,key=lambda x:-x[1])

def made_tasks(stage,kind,groups):
    dest=stage/"tasks"/(kind+".json")
    check(not dest.exists(),"TASK_DUPLICATE_"+kind)
    dest.write_text(json.dumps(groups,indent=2)+"\n")
    return dest

def current_method_h(intervals,active):
    # CPU UnivFD workers have gpu=None and MUST NOT enter the
    # physical-GPU-hour union (total_union casts GPU ids to int).
    gpu_active={pid:v for pid,v in active.items() if v["gpu"] is not None}
    return sum(total_union(intervals,gpu_active).values())

def stage_controller(run,method,head,global_spent):
    stage=run/method
    check(not stage.exists(),"FRESH_STAGE_"+method)
    stage.mkdir()
    for z in ("tasks","results","logs","pycache","readout"):
        if z!="readout":(stage/z).mkdir()
    total_start=time.monotonic()
    records=read(GT)["images"]
    completed=set()
    events=[];intervals=[];active={}
    is_gpu=method!="UnivFD"
    deadline=time.monotonic()+(5.0*3600 if is_gpu else 9.0*3600)
    method_cap=METHOD_CAPS[method]
    status="STAGE_RUNNING"
    # Force source/preset checks for both domains BEFORE native or CPU scoring.
    def preflight(dataset,idx):
        tasks=made_tasks(stage,"preflight_"+dataset,[idx])
        cmd=[str(ROOT/"conda-envs/daps/bin/python"),str(WORKER),
             "--expected-head",head,"--run",str(run),"--method",method,
             "--dataset",dataset,"--tasks",str(tasks),
             "--preflight-only"]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(WT),
            PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
            PYTHONPYCACHEPREFIX=str(stage/"pycache"/f"preflight_{dataset}"))
        p=subprocess.run(cmd,cwd=str(WT),env=env,text=True,
            capture_output=True,timeout=180)
        check(p.returncode==0,
              "SOURCE_PREFLIGHT_"+method+"_"+dataset+"_"+p.stderr[-300:])
        print("A35_SOURCE_PREFLIGHT_PASS",method,dataset,flush=True)
    preflight("ffhq",0);preflight("imagenet",180)
    # Each live worker includes model load in its charged GPU occupancy.
    def write_state(msg):
        charge=current_method_h(intervals,active)
        payload=dict(status=msg,method=method,code_head=head,
           n_complete=len(completed),expected=280,
           completed_indices=sorted(completed),
           active_workers=[dict(pid=p,gpu=v["gpu"],dataset=v["dataset"],
                                indices=v["indices"]) for p,v in active.items()],
           method_gpu_h=charge,prior_method_gpu_h=global_spent,
           G7_total_gpu_h=PRIOR_TOTAL_G7_H+global_spent+charge,
           project_gpu_h=PRIOR_PROJECT_H+global_spent+charge,
           stage_method_cap_gpu_h=method_cap,worker_events=events,
           GT_diagnostic_only=True,no_inverse_solver=True)
        atomic(stage/"STAGE_STATE.json",payload)
        return payload
    def finish():
        for pid,v in list(active.items()):
            rc=v["proc"].poll()
            if rc is None:continue
            end=time.monotonic();v["proc"].wait()
            del active[pid]
            if v["gpu"] is not None:
                intervals.append((v["gpu"],v["start"],end))
            good=rc==0
            for i in v["indices"]:
                p=stage/"results"/f"{i:03d}.json"
                if not p.is_file():
                    good=False;break
                r=read(p)
                if (r["method"]!=method or r["index"]!=i
                    or r["status"]!=STATUS
                    or r["implementation_head"]!=head):
                    good=False;break
            events.append(dict(pid=pid,dataset=v["dataset"],
              gpu=v["gpu"],n=len(v["indices"]),rc=rc,valid=good,
              wall_seconds=end-v["start"],log=v["log"]))
            if good:completed.update(v["indices"])
            else:
                print("ACTION NEEDED: A35_NATIVE_WORKER_STOP",method,
                      "pid",pid,"rc",rc,"log",v["log"],flush=True)
                raise RuntimeError("A35_INCOMPLETE_NATIVE_WORKER_"+method)
    def stop_own():
        (stage/"STOP_AFTER_IMAGE").touch(exist_ok=True)
        limit=time.monotonic()+300
        while active and time.monotonic()<limit:
            try:finish()
            except RuntimeError:break
            time.sleep(.5)
        for pid,v in list(active.items()):
            if v["proc"].poll() is None:
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
        time.sleep(2)
        for pid,v in list(active.items()):
            if v["proc"].poll() is None:
                try:os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        for pid,v in list(active.items()):
            v["proc"].wait()
            if v["gpu"] is not None:
                intervals.append((v["gpu"],v["start"],time.monotonic()))
            events.append(dict(pid=pid,dataset=v["dataset"],
               gpu=v["gpu"],rc=v["proc"].returncode,stopped=True))
            del active[pid]
    def launch(dataset,ids,gpu,tag):
        check(ids and all(records[i]["dataset"]==dataset for i in ids),
              "TASK_COHORT")
        task=made_tasks(stage,tag,ids)
        log=stage/"logs"/(tag+".log")
        env=dict(os.environ,
           CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
           PYTHONPATH=str(WT),PYTHONDONTWRITEBYTECODE="1",
           PYTHONNOUSERSITE="1",
           PYTHONPYCACHEPREFIX=str(stage/"pycache"/tag),
           OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
           OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
        cmd=[str(ROOT/"conda-envs/daps/bin/python"),str(WORKER),
             "--expected-head",head,"--run",str(run),
             "--method",method,"--dataset",dataset,
             "--tasks",str(task)]
        if gpu is not None:cmd+=["--gpu",str(gpu)]
        with log.open("x") as f:
            p=subprocess.Popen(cmd,cwd=str(WT),env=env,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        active[p.pid]=dict(proc=p,gpu=gpu,dataset=dataset,
                           start=time.monotonic(),indices=ids,log=str(log))
        print("A35_G7_WORKER_STARTED",method,dataset,
              "gpu",gpu,"pid",p.pid,"n",len(ids),flush=True)

    try:
        write_state(status)
        if is_gpu:
            # Two *charged* canaries, one per dataset; results are reused.
            for dataset,i in (("ffhq",0),("imagenet",180)):
                if STOP:break
                cards=gpu_admissible()
                waited=time.monotonic()
                while not cards and not STOP and time.monotonic()-waited<900:
                    print("A35_WAIT_SAFE_GPU",method,dataset,flush=True)
                    time.sleep(30);cards=gpu_admissible()
                if not cards:
                    status="PARKED_NO_ADMISSIBLE_GPU";break
                launch(dataset,[i],cards[0][0],"canary_"+dataset)
                while active:
                    finish()
                    if (time.monotonic()>deadline or STOP or
                        current_method_h(intervals,active)>method_cap-.4):
                        status="PARTIAL_BUDGET_OR_DEADLINE_STOP";break
                    time.sleep(2)
                if status!="STAGE_RUNNING":break
                write_state(status)
            if status=="STAGE_RUNNING":
                # Project from actual native per-image work (model-loads
                # measured separately). Stop rather than invent a cheaper
                # setting when G7 remaining budget may be insufficient.
                r0=read(stage/"results"/"000.json")
                r1=read(stage/"results"/"180.json")
                compute_s=179*r0["image_elapsed_s"]+99*r1["image_elapsed_s"]
                # Model load per GPU dataset shard + 30% contingency.
                estimated=1.30*(compute_s+4*(
                    r0["native_model_load_s"]+r1["native_model_load_s"]))/3600.
                spent=current_method_h(intervals,active)
                print("A35_G7_NATIVE_CANARY_PROJECTED_GPU_H",method,
                      round(estimated+spent,5),"cap",method_cap,flush=True)
                if estimated+spent>method_cap-.15:
                    status="PARKED_PROJECTED_GPU_H_EXCEEDS_REGISTERED_CAP"
            if status=="STAGE_RUNNING":
                for dataset,rng in (("ffhq",range(0,180)),
                                    ("imagenet",range(180,280))):
                    ids=[i for i in rng if i not in completed]
                    cards=gpu_admissible()
                    waited=time.monotonic()
                    while not cards and not STOP and time.monotonic()-waited<900:
                        time.sleep(30);cards=gpu_admissible()
                    if not cards:
                        status="PARKED_NO_ADMISSIBLE_GPU";break
                    # One persistent native model per admitted physical GPU;
                    # no concurrent own worker on the same card.
                    gpuids=[c[0] for c in cards]
                    for j,gpu in enumerate(gpuids):
                        subset=ids[j::len(gpuids)]
                        if subset:launch(dataset,subset,gpu,
                                         f"{dataset}_GPU{gpu}")
                    while active and status=="STAGE_RUNNING":
                        finish()
                        used=current_method_h(intervals,active)
                        if (STOP or time.monotonic()>deadline or
                            used>method_cap-.17 or
                            PRIOR_TOTAL_G7_H+global_spent+used>9.90):
                            status="PARTIAL_BUDGET_OR_DEADLINE_STOP";break
                        if int(time.monotonic()-total_start)%30<3:
                            write_state(status)
                        time.sleep(3)
                    if status!="STAGE_RUNNING":break
                    write_state(status)
        else:
            # Pinned UnivFD has a CPU implementation. No GPU-h charged.
            # Two sequential data domains, each loads its own pinned local
            # classifier; no CUDA context, no foreign process interaction.
            for dataset,rng in (("ffhq",list(range(180))),
                                ("imagenet",list(range(180,280)))):
                if STOP:status="PARTIAL_USER_STOP";break
                launch(dataset,rng,None,"cpu_"+dataset)
                while active:
                    finish()
                    if STOP or time.monotonic()>deadline:
                        status="PARTIAL_CPU_DEADLINE_STOP";break
                    time.sleep(3)
                if status!="STAGE_RUNNING":break
                write_state(status)
        if status=="STAGE_RUNNING" and len(completed)==N:
            status="A35_G7_STAGE_280_COMPLETE"
        elif status=="STAGE_RUNNING":
            status="PARTIAL_INCOMPLETE_GRID_STOP"
    except BaseException:
        status="STOP_A35_NATIVE_OR_CONTROLLER_FAILURE"
        raise
    finally:
        if active:stop_own()
        final=write_state(status)
        atomic(stage/"STAGE_TERMINAL.json",final)
        print("A35_G7_STAGE_TERMINAL",method,status,
              "complete",final["n_complete"],
              "charged_method_gpu_h",round(final["method_gpu_h"],6),
              "G7_total_h",round(final["G7_total_gpu_h"],6),flush=True)
    return final

def main(a):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and
          git(WT,"rev-parse","HEAD")==a.expected_head and
          not git(WT,"status","--porcelain"),
          "CLEAN_A35_EXECUTOR_HEAD")
    spec=pinned()
    if a.preflight_only:
        print("A35_G7_ALL_THREE_CPU_SOURCE_PREFLIGHT_PASS",
              "A34_project_h",PRIOR_PROJECT_H,
              "A34_G7_total_h",PRIOR_TOTAL_G7_H,
              "G7_remaining",G7_LIMIT-PRIOR_TOTAL_G7_H,flush=True)
        return
    run=Path(a.run)
    check(run.parent==BASE and
          run.name.startswith("xp_a35_g7_remaining_")
          and not run.exists() and not run.is_symlink(),
          "FRESH_RUN_ROOT")
    with GLOBAL_LOCK.open("a+") as file:
        fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run.mkdir(mode=0o700)
        state=new_state(run,a.expected_head)
        atomic(run/"A35_G7_AUTOPILOT_STATE.json",state)
        spent=0.
        for method in METHODS:
            if STOP or datetime.now(ZoneInfo("America/Detroit"))>=HARD_CUTOFF:
                state["status"]="PARTIAL_HUMAN_OR_DEADLINE_STOP";break
            print("A35_G7_METHOD_START",method,"G7_total_h",
                  round(PRIOR_TOTAL_G7_H+spent,6),flush=True)
            result=stage_controller(run,method,a.expected_head,spent)
            spent+=result["method_gpu_h"]
            state["charged_gpu_h"]=spent
            state["G7_total_gpu_h"]=PRIOR_TOTAL_G7_H+spent
            state["project_gpu_h"]=PRIOR_PROJECT_H+spent
            state["method_results"][method]=dict(
              stage_status=result["status"],n_complete=result["n_complete"],
              stage_gpu_h=result["method_gpu_h"],published=None)
            atomic(run/"A35_G7_AUTOPILOT_STATE.json",state)
            if result["status"]!="A35_G7_STAGE_280_COMPLETE":
                state["status"]="PARTIAL_A35_SCIENCE_OR_RESOURCE_GATE_STOP"
                break
            check(result["n_complete"]==280 and
                  state["G7_total_gpu_h"]<G7_LIMIT and
                  state["project_gpu_h"]<PROJECT_LIMIT,
                  "STAGE_COMPLETE_BUT_BUDGET_FAIL")
            receipt=analyze(run,method,a.expected_head,result)
            pub=publish(run,method,a.expected_head,receipt)
            state["method_results"][method]["published"]=pub
            state["not_run"].remove(method)
            state["status"]="A35_G7_PARTIAL_METHODS_PUBLISHED"
            atomic(run/"A35_G7_AUTOPILOT_STATE.json",state)
        else:
            state["status"]="A35_G7_THREE_REMAINING_METHODS_PUBLISHED"
            state["all_methods_completed"]=True
        check(state["G7_total_gpu_h"]<G7_LIMIT and
              state["project_gpu_h"]<PROJECT_LIMIT,
              "A35_FINAL_BUDGET")
        atomic(run/"A35_G7_AUTOPILOT_FINAL_STATUS.json",state)
        print("A35_G7_AUTOPILOT_TERMINAL",state["status"],
              "published_methods",
              [x for x in METHODS
               if state["method_results"].get(x,{}).get("published")],
              "not_run",state["not_run"],
              "G7_total_h",round(state["G7_total_gpu_h"],6),
              "project_h",round(state["project_gpu_h"],6),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--preflight-only",action="store_true")
    args=p.parse_args()
    signal.signal(signal.SIGINT,stopping)
    signal.signal(signal.SIGTERM,stopping)
    try:main(args)
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A35_G7_AUTOPILOT",repr(e),flush=True)
        sys.exit(8)
