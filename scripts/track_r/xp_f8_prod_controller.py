#!/usr/bin/env python3
"""A10 hard-budgeted restartable four-GPU XP-F8 DAPS 400x5 controller.

B16 + ALL historical DAPS replay wave receipts + F8 worker attempts share
70 GPU-h; old failures count. Deadline Oct 30 12 EDT. F8 outcomes embargoed.
One own worker per physical GPU, dirty co-tenancy allowed with <=80% VRAM
and foreign-process telemetry. Never signals a foreign PID.
"""
from __future__ import annotations
import argparse,datetime,fcntl,json,os,signal,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu
from scripts.track_r.xp_f8_legacy_preflight_cpu import (
 ROOT,BASE,WT,ORIG_DAPS,DAPS_HEAD,MODEL,MODEL_SHA,
 B16,A10_DEADLINE,git,sha,read_json,fail)
from scripts.track_r.xp_f8_legacy_replay_supervisor import foreign_track_r_pid
from scripts.track_r.xp_f8_prod_prepare_cpu import (
 GATE,GATE_SHA,B16_COMPLETE_SHA,A10_LIMIT)
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
MIN_AVAILABLE_MIB=52096
SAFETY_MIB=2048
RESERVE_PER_JOB_SECONDS=1800.
POLL_SECONDS=10
STOPPING=False
def on_signal(sig,frame):
    global STOPPING
    STOPPING=True
for signame in (signal.SIGTERM,signal.SIGINT):
    signal.signal(signame,on_signal)
def write_atomic(path,value):
    tmp=path.with_name(path.name+".tmp."+str(os.getpid()))
    with tmp.open("x") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    os.replace(tmp,path)
def event(root,name,**details):
    with (root/"XP_F8_PROD_EVENTS.jsonl").open("a") as f:
        f.write(json.dumps(dict(event=name,UTC=
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **details),sort_keys=True,allow_nan=False)+"\n")
def initial_state(prep):
    return dict(status="XP_F8_NEW_GENERATION_RUNNING",
      source_head=prep["head"],manifest_sha256=prep["manifest_sha256"],
      done={},attempts={},spent_gpu_seconds=0.,active=[],
      failure_log=[],last_update_utc="",foreign_telemetry_samples=0,
      n_total=920)
def state_receipts_check(root,state):
    receipts=list((root/"ledger").glob("XP_F8_GPU_COST_*.json"))
    accounted=sum(float(read_json(p)["elapsed_seconds"]) for p in receipts)
    fail(abs(accounted-state["spent_gpu_seconds"])<.5,
         "F8_ATTEMPT_LEDGER_DIFFERENCE")
    for key,x in state["done"].items():
        i=int(key)
        p=root/"roots"/f"r{i:04d}_a{x['attempt']}"/"XP_F8_NEW_ROOT_COMPLETE.json"
        fail(p.is_file() and sha(p)==x["receipt_sha256"],
             "F8_DONE_RECEIPT_DRIFT_"+key)
        rec=read_json(p)
        fail(rec["root_index"]==i and
             rec["status"]=="XP_F8_NEW_ROOT_GENERATION_PASS_EMBARGOED"
             and rec["no_scientific_outcome_opened"] is True,
             "F8_DONE_SCHEMA_DRIFT_"+key)
def verify_campaign(a):
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")=="","GPU_CONTROLLER_MUST_HIDE_GPUS")
    fail(git(WT,"rev-parse","HEAD")==a.expected_head and
         not git(WT,"status","--porcelain"),"CONTROLLER_HEAD")
    root=Path(a.run)
    fail(root.parent==BASE and root.name.startswith("xp_f8_prod_")
         and root.is_dir() and not root.is_symlink(),"RUN")
    prep=read_json(root/"XP_F8_PROD_PREP.json")
    mf=root/"XP_F8_PROD_TASKS.json"
    fail(prep["head"]==a.expected_head
         and prep["status"]=="XP_F8_920_GT_FREE_PRODUCTION_PREP_PASS"
         and sha(mf)==prep["manifest_sha256"],"F8_PREP_SHA")
    tasks=read_json(mf)
    fail(tasks["n_total"]==920 and len(tasks["rows"])==920
         and tasks["head"]==a.expected_head
         and tasks["corrected_DEV80_seed_domain"]==
             "B24_DEV80_DAPS_SOLVER_V1","F8_MANIFEST")
    fail(git(ORIG_DAPS,"rev-parse","HEAD")==DAPS_HEAD and
         sha(MODEL)==MODEL_SHA,"DAPS_SOURCE_OR_MODEL")
    for name,digest in GATE_SHA.items():
        fail(sha(GATE/name)==digest,"HISTORICAL_REPLAY_RECEIPT")
    fail(sha(B16/"XP_A11_B16_PROD_COMPLETE.json")==B16_COMPLETE_SHA,
         "B16_COMPLETE_SHA")
    b=read_json(B16/"B16_PRODUCTION_STATE.json")
    used=(b["spent_gpu_seconds"]+b["gate_spent_gpu_seconds"])/3600
    saved=prep["accounting"]["replay_ledger"]
    currents=[]
    for p in sorted(BASE.glob(
      "xp_f8_legacy_replay_*/XP_F8_LEGACY_REPLAY_WAVE[12].json")):
        currents.append(dict(path=str(p),sha256=sha(p),
           gpu_h=read_json(p)["incremental_gpu_h"]))
    fail(currents==saved,"REPLAY_BUDGET_LEDGER_DRIFT")
    used+=sum(item["gpu_h"] for item in saved)
    fail(abs(used-prep["accounting"]["prior_gpu_h"])<1e-5,
         "BASE_GPU_BUDGET_DRIFT")
    fail(used<A10_LIMIT,"BASE_GPU_BUDGET_EXHAUSTED")
    return root,prep,tasks,used*3600
def verify_root(root,prep,row,i,attempt):
    dest=root/"roots"/f"r{i:04d}_a{attempt}"
    p=dest/"XP_F8_NEW_ROOT_COMPLETE.json"
    fail(p.is_file(),"SUCCESS_RECEIPT_MISSING")
    r=read_json(p)
    fail(r["status"]=="XP_F8_NEW_ROOT_GENERATION_PASS_EMBARGOED"
         and r["root_index"]==i and r["attempt"]==attempt
         and r["uid"]==row["uid"] and
         r["manifest_sha256"]==prep["manifest_sha256"]
         and r["native_seed"]==row["native_seed"]
         and r["measurement_sha256"]==row["measurement_sha256"]
         and r["no_GT_scoring"] is True
         and r["no_scientific_outcome_opened"] is True
         and r["no_labels_calculated"] is True
         and sha(r["output_path"])==r["output_sha256"],
         "RECONSTRUCTION_RECEIPT_INVALID_"+str(i))
    return sha(p)
def pid_telemetry(root,cards,active,state):
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for i,v in active.items():
        card=next((c for c in cards if c.index==v["gpu"]),None)
        if card is not None:
            v["telemetry"].append(dict(utc=now,
               device_used_mib=card.used_mib,
               device_free_mib=card.free_mib,
               foreign_pids=list(card.foreign_pids),
               device_process_pids=list(card.process_pids)))
    state["foreign_telemetry_samples"]+=1
def write_cost(root,i,v,rc,elapsed):
    p=root/"ledger"/f"XP_F8_GPU_COST_R{i:04d}_A{v['attempt']}.json"
    write_atomic(p,dict(status="XP_F8_GPU_ATTEMPT_COST_RECORDED",
         root_index=i,attempt=v["attempt"],gpu=v["gpu"],
         elapsed_seconds=elapsed,
         rc=rc,foreign_pids_observed=sorted(
           {x for sample in v["telemetry"] for x in sample["foreign_pids"]}),
         n_telemetry_samples=len(v["telemetry"]),
         min_free_mib=min(
          [x["device_free_mib"] for x in v["telemetry"]],default=None),
         max_device_used_mib=max(
          [x["device_used_mib"] for x in v["telemetry"]],default=None),
         samples=v["telemetry"],
         log=str(v["logpath"])))
    return sha(p)
def stop_owned(active,reason,root):
    event(root,"STOP_OWN_TRACK_R_WORKERS_ONLY",reason=reason,
          pids=[v["proc"].pid for v in active.values()])
    for v in active.values():
        if v["proc"].poll() is None:
            try:os.killpg(v["proc"].pid,signal.SIGTERM)
            except ProcessLookupError:pass
def run(a,root,prep,tasks,base_seconds):
    states=root/"XP_F8_CONTROLLER_STATE.json"
    (root/"roots").mkdir(exist_ok=True)
    (root/"ledger").mkdir(exist_ok=True)
    if a.resume:
        fail(states.is_file(),"F8_NO_STATE_FOR_RESUME")
        state=read_json(states)
        fail(state["source_head"]==a.expected_head
             and state["manifest_sha256"]==prep["manifest_sha256"]
             and state["n_total"]==920 and
             state["active"]==[],"F8_RESUME_STATE_NOT_DRAINED")
        state_receipts_check(root,state)
    else:
        fail(not states.exists() and
             not list((root/"ledger").glob("XP_F8_GPU_COST_*.json")),
             "F8_FRESH_RUN_ONLY")
        state=initial_state(prep)
    active={}
    done=state["done"]
    attempts=state["attempts"]
    completed_since_start=0
    drain_reason=None
    def store(status=None):
        if status is not None:state["status"]=status
        state["done"]=done
        state["attempts"]=attempts
        state["active"]=[dict(index=i,pid=v["proc"].pid,gpu=v["gpu"],
                              attempt=v["attempt"]) for i,v in active.items()]
        state["last_update_utc"]=datetime.datetime.now(
             datetime.timezone.utc).isoformat()
        write_atomic(states,state)
    store("XP_F8_NEW_GENERATION_RUNNING")
    event(root,"XP_F8_START_OR_RESUME",done=len(done),
          base_gpu_h=base_seconds/3600,
          prior_f8_gpu_h=state["spent_gpu_seconds"]/3600,
          manifest_sha=prep["manifest_sha256"])
    try:
        while len(done)<920:
            cards=collect(own_roots=[v["proc"].pid for v in active.values()])
            pid_telemetry(root,cards,active,state)
            # A10.1: a newly arriving foreign job can push a device
            # above 80% after admission. Terminate only our process
            # group, never a foreign PID, and account the failed attempt.
            for i,v in active.items():
                card=next((c for c in cards if c.index==v["gpu"]),None)
                if (card is not None and card.total_mib>0
                    and card.used_mib > .80*card.total_mib
                    and v["proc"].poll() is None):
                    event(root,"A10_80PCT_VRAM_RUNTIME_STOP",
                          root_index=i,gpu=v["gpu"],
                          used_mib=card.used_mib,total_mib=card.total_mib)
                    try:os.killpg(v["proc"].pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for i,v in list(active.items()):
                ret=v["proc"].poll()
                if ret is None:continue
                elapsed=time.monotonic()-v["start"]
                v["handle"].close()
                cost=write_cost(root,i,v,ret,elapsed)
                state["spent_gpu_seconds"]+=elapsed
                active.pop(i)
                # Keep the other still-running PIDs in state; store()
                # reconstructs the active list from the live controller.
                if ret==0:
                    try:
                        sh=verify_root(root,prep,tasks["rows"][i],i,v["attempt"])
                        done[str(i)]=dict(attempt=v["attempt"],
                            receipt_sha256=sh,gpu_cost_sha256=cost)
                        completed_since_start+=1
                        event(root,"ROOT_RECEIPT_PASS",index=i,
                              attempt=v["attempt"],receipt_sha=sh)
                        if len(done)%25==0 or len(done)==920:
                            print("F8_NEW_ROOTS_COMPLETE",len(done),"/920",
                                  "spent_A10_gpu_h",
                                  round((base_seconds+state["spent_gpu_seconds"])/3600,4),
                                  flush=True)
                    except Exception as exc:
                        event(root,"RECEIPT_CHECK_FAILED",index=i,
                              reason=str(exc))
                        state["failure_log"].append(dict(index=i,
                            attempt=v["attempt"],reason=str(exc)))
                else:
                    state["failure_log"].append(dict(index=i,attempt=v["attempt"],
                        reason="WORKER_NONZERO_"+str(ret)))
                    print("F8_WORKER_NONZERO",i,"attempt",v["attempt"],
                          "log",v["logpath"],flush=True)
                store()
            if STOPPING or (root/"STOP_F8").exists():
                drain_reason="XP_F8_STOPPED_OPERATOR"
            if datetime.datetime.now(ZoneInfo("America/Detroit"))>=A10_DEADLINE:
                drain_reason="XP_F8_STOPPED_A10_DEADLINE"
            elapsed_now=sum(time.monotonic()-v["start"] for v in active.values())
            if (base_seconds+state["spent_gpu_seconds"]+elapsed_now
                >=A10_LIMIT*3600-5):
                drain_reason="XP_F8_STOPPED_A10_BUDGET"
            if drain_reason:
                if drain_reason in ("XP_F8_STOPPED_A10_DEADLINE",
                                    "XP_F8_STOPPED_A10_BUDGET"):
                    stop_owned(active,drain_reason,root)
                if not active:
                    break
                time.sleep(POLL_SECONDS)
                continue
            if len(done)>=a.max_completed_this_session and not active:
                drain_reason="XP_F8_STAGED_SMOKE_COMPLETE"
                break
            occupied={v["gpu"] for v in active.values()}
            while len(active)<4 and len(done)+len(active)<920:
                if (base_seconds+state["spent_gpu_seconds"]+
                      sum(time.monotonic()-v["start"] for v in active.values())+
                      (len(active)+1)*RESERVE_PER_JOB_SECONDS
                       >=A10_LIMIT*3600):
                    drain_reason="XP_F8_DRAINED_A10_BUDGET"
                    break
                if len(done)+len(active)>=a.max_completed_this_session:
                    break
                pending=[i for i in range(920) if str(i) not in done
                         and i not in active and int(attempts.get(str(i),0))<=2]
                if not pending:break
                i=pending[0]
                available=[c for c in cards if c.index not in occupied
                    and c.visibility=="known"
                    and not foreign_track_r_pid(
                        c,[v["proc"].pid for v in active.values()])]
                gpu,reasons=choose_gpu(available,MIN_AVAILABLE_MIB,
                    safety_margin_mib=SAFETY_MIB,fraction=.8,clean_only=False)
                if gpu is None:
                    if not active:print("WAIT_DIRTY_GPU_VRAM",reasons,flush=True)
                    break
                attempt=int(attempts.get(str(i),0))
                dest=root/"roots"/f"r{i:04d}_a{attempt}"
                fail(not dest.exists(),"ATTEMPT_DIR_COLLISION_"+str(i))
                cmd=[str(PY),"-u","-m","scripts.track_r.xp_f8_prod_worker_gpu",
                   "--expected-head",a.expected_head,"--run",str(root),
                   "--index",str(i),"--attempt",str(attempt),
                   "--gpu",str(gpu)]
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),
                           PYTHONPATH=str(WT),PYTHONNOUSERSITE="1",
                           PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2")
                logpath=root/"ledger"/f"XP_F8_WORKER_R{i:04d}_A{attempt}.log"
                handle=logpath.open("x")
                proc=subprocess.Popen(cmd,cwd=str(WT),env=env,
                    stdout=handle,stderr=subprocess.STDOUT,
                    start_new_session=True)
                active[i]=dict(proc=proc,gpu=gpu,
                    start=time.monotonic(),attempt=attempt,
                    handle=handle,logpath=logpath,telemetry=[])
                attempts[str(i)]=attempt+1
                occupied.add(gpu)
                event(root,"ROOT_ADMITTED",index=i,attempt=attempt,
                      gpu=gpu,pid=proc.pid)
                print("F8_START",i,"gpu",gpu,"attempt",attempt,
                      "pid",proc.pid,flush=True)
                store()
            if not active:
                if drain_reason:break
                remaining=[i for i in range(920)
                           if str(i) not in done and
                           int(attempts.get(str(i),0))<=2]
                if not remaining:
                    drain_reason="XP_F8_STOPPED_ATTEMPT_FAILURES"
                    break
            time.sleep(POLL_SECONDS)
    finally:
        if active:
            stop_owned(active,"CONTROLLER_DRAIN_ON_EXIT",root)
            for i,v in list(active.items()):
                v["proc"].wait()
                elapsed=time.monotonic()-v["start"]
                v["handle"].close()
                write_cost(root,i,v,v["proc"].returncode,elapsed)
                state["spent_gpu_seconds"]+=elapsed
                active.pop(i)
                store()
    if len(done)==920:
        status="XP_F8_920_GENERATION_COMPLETE_EMBARGOED"
    else:
        status=drain_reason or "XP_F8_PARTIAL"
    store(status)
    summary=dict(status=status,
      complete=len(done),total=920,
      A10_base_gpu_h=base_seconds/3600,
      A10_F8_gpu_h=state["spent_gpu_seconds"]/3600,
      A10_total_gpu_h=(base_seconds+state["spent_gpu_seconds"])/3600,
      A10_gpu_h_remaining=A10_LIMIT-(base_seconds+state["spent_gpu_seconds"])/3600,
      no_GT_scoring=True,scientific_outcomes_embargoed=True,
      source_head=a.expected_head,manifest_sha256=prep["manifest_sha256"],
      failed_attempts=len(state["failure_log"]),
      stop_file=str(root/"STOP_F8"))
    write_atomic(root/"XP_F8_PROD_LAST_STATUS.json",summary)
    print("XP_F8_PROD_STATE",json.dumps(summary,sort_keys=True),flush=True)
    if len(done)==920:
        print("XP_F8_920_GENERATION_COMPLETE_EMBARGOED",flush=True)
    else:
        print("ACTION NEEDED: XP_F8_PARTIAL_OR_STOPPED",status,
              "completed",len(done),"/920",flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--max-completed-this-session",type=int,
                   default=920)
    a=p.parse_args()
    fail(1<=a.max_completed_this_session<=920,"MAX_COMPLETED")
    root,prep,tasks,base=verify_campaign(a)
    # One F8 production controller at a time, even if run directories differ.
    lock=(BASE/"XP_F8_PRODUCTION_CONTROLLER.lock").open("a+")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        fail(False,"OTHER_F8_CONTROLLER_ACTIVE")
    try:run(a,root,prep,tasks,base)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=="__main__":
    try:main()
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_F8_PROD_CONTROLLER "+str(e),flush=True)
        sys.exit(8)
