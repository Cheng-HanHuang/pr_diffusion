#!/usr/bin/env python3
"""Unattended Track R B16 replay->production handoff, four-GPU campaign.

Wait for signed-off 400-root CPU lineage gate from pinned existing R3 run,
verify EVERY replay receipt, then launch 2160 embargoed B16 trajectories.
One worker per physical GPU, admission only with adequate VRAM. The original
12h deadline / 48 GPU-h campaign allowance are inherited, not reset.

No new production GT/labels are ever read, scored, or revealed.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from scripts.track_r.xp_a11_b16_prod_prepare_cpu import (
    ROOT,BASE,WT,A6,A6_HEAD,GATE_NAME,GATE_MANIFEST_SHA,OLD_HEAD,
    safe,sha,git)
from scripts.track_r.xp_a10_b16_gate_autopilot import verify_group
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

STOP=False
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
SPY=ROOT/"conda-envs/sitcom_ode_bw/bin/python"
CACHE=ROOT/"outputs/.cache/track_r"
MAX_GPU_H=48.
EST_EARLIER_GPU_H=68.
VRAM_REQUIRED_MIB=32000
REFERENCE_SEC=32.104482267051935
MARGIN_MIB=2048
B22_SMOKE={0,1}
B24_SMOKE={100,101,112,113}
B32_B22_SMOKE={1060,1061}
B32_B24_SMOKE={1160,1161,1176,1177}
B16_TASKS=1060
ALL_TASKS=2440

def on_stop(sig,frame):
    global STOP
    STOP=True

def assert_ok(x,tag):
    if not x:raise RuntimeError("STOP_B16_HANDOFF_"+tag)

def write_x(path,v):
    with Path(path).open("x") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())

def save(path,v):
    tmp=path.with_name("."+path.name+"."+str(os.getpid()))
    write_x(tmp,v)
    os.replace(tmp,path)

def event(run,event_name,**kw):
    rec={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "event":event_name,**kw}
    with (run/"B16_PRODUCTION_EVENTS.jsonl").open("a") as f:
        f.write(json.dumps(rec,sort_keys=True,allow_nan=False)+"\n")
        f.flush();os.fsync(f.fileno())

def frozen(run,head):
    assert_ok(run.parent==BASE and run.name.startswith("xp_a11_b16_prod_")
              and run.is_dir() and not run.is_symlink(),"RUN")
    assert_ok(git(WT,"rev-parse","HEAD")==head and
              not git(WT,"status","--porcelain"),"SOURCE_SHA")
    assert_ok(git(A6,"rev-parse","HEAD")==A6_HEAD and
              not git(A6,"status","--porcelain"),"A6_FROZEN")
    pre=json.loads(safe(run/"XP_A11_B16_PROD_PREP.json").read_text())
    path=run/"XP_A11_B16_PROD_TASKS.json"
    assert_ok(pre["head"]==head and
              pre["status"]=="B16_PRODUCTION_PREP_PASS_GATE_PENDING" and
              sha(path)==pre["manifest_sha256"],"MANIFEST")
    mf=json.loads(path.read_text())
    groups=(mf["b22_groups"]+mf["b24_groups"]+
            mf["b32_b22_groups"]+mf["b32_b24_groups"])
    assert_ok(mf["head"]==head and
              mf["n_new_roots"]==2160 and
              mf["n_new_b22"]==1200 and mf["n_new_b24"]==960 and
              mf["GT_supplied"] is False and
              mf["historical_candidate_supplied"] is False and
              mf["new_outcomes_embargoed"] is True and
              mf["n_background_b32_roots"]==2880 and
              mf["background_requires_full_b16_completion"] is True and
              len(groups)==ALL_TASKS and
              len({g["task_id"] for g in groups})==ALL_TASKS,
              "PRODUCTION_SCHEMA")
    reference=WT/"docs/track_r/executors/xp/returns/XP_T_SITCOM_BATCH1_20261026T233451Z/XP_T_SITCOM_COMPLETE.json"
    assert_ok(sha(reference)=="bb2c3c377f07be2982e412eb96ff1a8b35805146dc021cc8baab3abd6ddc075e",
              "FFHQ_TIMING_CALIBRATION_SHA")
    refdata=json.loads(reference.read_text())
    assert_ok(abs(float(refdata["per_dataset"]["ffhq"]["sampling_wall_s_median"])
                  -REFERENCE_SEC)<1e-10,
              "FFHQ_TIMING_CALIBRATION_VALUE")
    assert_ok(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_CONTROLLER")
    return mf,groups,pre

def old_gate_status():
    gate=BASE/GATE_NAME
    prep=json.loads(safe(gate/"XP_A10_B16_GATE_PREP.json").read_text())
    assert_ok(prep["head"]==OLD_HEAD and
              prep["manifest_sha256"]==GATE_MANIFEST_SHA and
              sha(gate/"XP_A10_B16_GATE_TASKS.json")==GATE_MANIFEST_SHA,
              "OLD_GATE_FROZEN")
    state_file=gate/"B16_GATE_STATE.json"
    if not state_file.is_file():return "WAIT",None
    state=json.loads(state_file.read_text())
    assert_ok(state["head"]==OLD_HEAD and
              state["manifest_sha256"]==GATE_MANIFEST_SHA and
              isinstance(state["deadline_epoch"],(int,float)) and
              isinstance(state["spent_gpu_seconds"],(int,float)) and
              0<=state["spent_gpu_seconds"]<=MAX_GPU_H*3600+3600,
              "OLD_GATE_STATE")
    if state["status"]=="B16_GATE_FULL_PASS":
        assert_ok(state["active"]==[] and
                  state["done"]==list(range(100)),
                  "GATE_100_GROUPS")
        return "PASS",state
    if state["status"] in ("STOP_WORKER_FAILURE","STOP_UNEXPECTED_DRAINED_UNVERIFIED",
                            "STOP_UNEXPECTED","STOP_B16_GATE_CPU_SCORE_FAILED"):
        return "FAILED",state
    if time.time()>=state["deadline_epoch"]:
        return "DEADLINE",state
    return "WAIT",state

def certify_gate():
    gate=BASE/GATE_NAME
    status,state=old_gate_status()
    assert_ok(status=="PASS","GATE_NOT_COMPLETE")
    result_file=gate/"XP_A10_B16_GATE_COMPARE.json"
    result=json.loads(safe(result_file).read_text())
    assert_ok(result["status"]=="XP_A10_B16_GATE_PASS" and
              result["head"]==OLD_HEAD and
              result["manifest_sha256"]==GATE_MANIFEST_SHA and
              result["n_roots"]==400 and
              result["n_images"]==100 and result["n_pass"]==400 and
              result["n_fail"]==0 and
              result["B16_new_run_labels_opened"] is False and
              result["gt_used_CPU_only"] is True and
              len(result["rows"])==400 and
              all(r["pass_gate"] is True and
                  r["abs_delta_db"]<=0.1 and
                  r["label"]==r["historical_label"] for r in result["rows"]),
              "GATE_HISTORICAL_SCORE")
    old=json.loads(safe(gate/"XP_A10_B16_GATE_TASKS.json").read_text())
    old["_manifest_sha"]=GATE_MANIFEST_SHA
    for i in range(100):
        verify_group(gate,old,i,OLD_HEAD,check_times=False)
    return dict(
       gate_run=str(gate),gate_manifest_sha256=GATE_MANIFEST_SHA,
       gate_score_sha256=sha(result_file),
       gate_n_pass=400,gate_score_status="XP_A10_B16_GATE_PASS",
       gate_first_start_epoch=state["first_start_epoch"],
       gate_deadline_epoch=state["deadline_epoch"],
       gate_spent_gpu_seconds=float(state["spent_gpu_seconds"]),
       B16_new_run_labels_opened=False)

def verify_task(run,groups,manifest_sha,index,head):
    g=groups[index]
    p=run/"gpu"/g["task_id"]
    summary=json.loads(safe(p/"XP_SITCOM_WORKER_COMPLETE.json").read_text())
    assert_ok(summary["status"]=="XP_SITCOM_WORKER_PASS" and
              summary["head"]==head and
              summary["manifest_sha256"]==manifest_sha and
              summary["new_roots"]==g["new_roots"] and
              int(summary.get("consumed_prefix_roots",0))==
              g["historical_prefix_computations"],
              "GROUP_SUMMARY_"+str(index))
    prefix=4 if index<100 else 16 if 1060<=index<1160 else 0
    new=g["rows"][prefix:]
    assert_ok(len(new)==g["new_roots"] and
              len(summary["roots"])==len(new),"NEW_ROOT_COUNT")
    secs=[]
    for row,item in zip(new,summary["roots"]):
        dest=p/("root_"+hashlib.sha256(row["uid"].encode()).hexdigest()[:24])
        r=json.loads(safe(dest/"XP_SITCOM_ROOT_COMPLETE.json").read_text())
        assert_ok(item["uid"]==row["uid"] and
                  item["file_sha256"]==sha(dest/"XP_SITCOM_ROOT_COMPLETE.json") and
                  r["status"]=="XP_SITCOM_REPLAY_ROOT_PASS" and
                  r["uid"]==row["uid"] and
                  r["rep"]==row["rep"] and
                  r["runner_head"]==head and
                  r["manifest_sha256"]==manifest_sha and
                  r["measurement_sha256"]==row["measurement_sha256"] and
                  r["gt_access"] is False and
                  r["historical_access"] is False and
                  sha(dest/"reconstruction.pt")==r["reconstruction_pt_sha256"] and
                  sha(dest/"reconstruction.png")==r["reconstruction_png_sha256"],
                  "ROOT_SHA_"+str(index))
        t=float(r["sampling_wall_s"])
        assert_ok(math.isfinite(t) and 1<t<7200,"ROOT_WALL")
        secs.append(t)
    # Published FFHQ ±50% timing range. A family-specific isolated
    # failure stops the family before expanding beyond smoke.
    mid=sorted(secs)[len(secs)//2]
    if len(secs)%2==0:
        mid=(sorted(secs)[len(secs)//2-1]+mid)/2
    assert_ok(0.5*REFERENCE_SEC<=mid<=1.5*REFERENCE_SEC,
              "FFHQ_SMOKE_SPEED_"+g["source"])
    return sum(secs)

def launch(run,head,groups,index,gpu,attempt):
    g=groups[index]
    log=run/"logs"/f"B16_{index:04d}_{g['task_id']}_try{attempt:02d}.log"
    assert_ok(not log.exists(),"JOB_LOG_EXISTS")
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
               PYTHONPATH=str(WT),
               PYTHONPYCACHEPREFIX=str(run/"pycache"/f"task{index:04d}"),
               TMPDIR=str(CACHE/"tmp"),XDG_CACHE_HOME=str(CACHE),
               CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
    cmd=[str(SPY),str(WT/"scripts/track_r/xp_a11_b16_prod_worker_gpu.py"),
         "--run",str(run),"--expected-head",head,
         "--index",str(index),"--gpu",str(gpu)]
    with log.open("x") as f:
        p=subprocess.Popen(cmd,env=env,stdout=f,
                           stderr=subprocess.STDOUT,start_new_session=True)
    return dict(proc=p,index=index,gpu=gpu,task_id=g["task_id"],
                start=time.monotonic(),log=str(log),samples=0,
                contaminated=False)

def main(a):
    assert_ok(0<a.poll<=60 and 1<=a.max_parallel<=4 and
              0<a.max_gpu_hours<=48,"ARGS")
    run=Path(a.run)
    mf,groups,prep=frozen(run,a.expected_head)
    msha=prep["manifest_sha256"]
    run.joinpath("gpu").mkdir(exist_ok=True)
    run.joinpath("logs").mkdir(exist_ok=True)
    run.joinpath("pycache").mkdir(exist_ok=True)
    run.joinpath("quarantine").mkdir(exist_ok=True)
    (CACHE/"tmp").mkdir(parents=True,exist_ok=True)
    with (run/"B16_PRODUCTION_CONTROLLER.lock").open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("STOP_B16_HANDOFF_ALREADY_RUNNING")
        statepath=run/"B16_PRODUCTION_STATE.json"
        if statepath.is_file():
            state=json.loads(statepath.read_text())
            assert_ok(state["head"]==a.expected_head and
                      state["manifest_sha256"]==msha and
                      state["active"]==[],"RESUME_STATE")
            done=set(state["done"])
            attempts=state["attempts"]
            for i in sorted(done):
                verify_task(run,groups,msha,i,a.expected_head)
        else:
            state=dict(status="WAITING_REPLAY_400_GATE",
                       head=a.expected_head,manifest_sha256=msha,
                       done=[],attempts={},active=[],spent_gpu_seconds=0.,
                       gate_spent_gpu_seconds=None,first_start_epoch=None,
                       deadline_epoch=None,
                       no_new_outcomes_opened=True,
                       b22_smoke_pass=False,b24_smoke_pass=False,
                       b32_b22_smoke_pass=False,b32_b24_smoke_pass=False)
            save(statepath,state)
            done=set()
            attempts=state["attempts"]
        active={}
        def store(status):
            state["status"]=status
            state["done"]=sorted(done)
            state["active"]=[dict(index=i,pid=v["proc"].pid,gpu=v["gpu"])
                             for i,v in active.items()]
            save(statepath,state)
        try:
            while not (run/"B16_PRODUCTION_ADMISSION.json").exists():
                if STOP or (run/"STOP_AFTER_ACTIVE").exists():
                    store("DRAINED_BEFORE_ADMISSION");return 0
                status,old=old_gate_status()
                if status=="FAILED":
                    store("STOP_REPLAY_GATE_FAILED")
                    print("ACTION NEEDED: B16_REPLAY_GATE_FAILED",flush=True)
                    return 8
                if status=="DEADLINE":
                    store("DRAINED_REPLAY_DEADLINE")
                    print("B16_CAMPAIGN_DEADLINE_NO_PRODUCTION",flush=True)
                    return 0
                if status=="PASS":
                    proof=certify_gate()
                    assert_ok(proof["gate_spent_gpu_seconds"]<=
                              a.max_gpu_hours*3600,"GATE_SPENT_BUDGET")
                    admission=dict(
                        status="B16_PRODUCTION_ADMITTED_400_REPLAY_PASS",
                        head=a.expected_head,manifest_sha256=msha,
                        **proof,no_new_outcomes_opened=True)
                    write_x(run/"B16_PRODUCTION_ADMISSION.json",admission)
                    state["gate_spent_gpu_seconds"]=proof["gate_spent_gpu_seconds"]
                    state["first_start_epoch"]=proof["gate_first_start_epoch"]
                    state["deadline_epoch"]=proof["gate_deadline_epoch"]
                    store("B16_PRODUCTION_ADMITTED")
                    event(run,"400_REPLAY_GATE_PASS",n=400,
                          gate_score_sha256=proof["gate_score_sha256"])
                    break
                store("WAITING_REPLAY_400_GATE")
                time.sleep(a.poll)
            admission=json.loads(safe(run/"B16_PRODUCTION_ADMISSION.json").read_text())
            assert_ok(admission["status"]=="B16_PRODUCTION_ADMITTED_400_REPLAY_PASS" and
                      admission["head"]==a.expected_head and
                      admission["manifest_sha256"]==msha and
                      admission["gate_n_pass"]==400 and
                      admission["no_new_outcomes_opened"] is True,
                      "ADMISSION_PROVENANCE")
            state["gate_spent_gpu_seconds"]=float(admission["gate_spent_gpu_seconds"])
            state["first_start_epoch"]=admission["gate_first_start_epoch"]
            state["deadline_epoch"]=admission["gate_deadline_epoch"]
            smoke=B22_SMOKE|B24_SMOKE
            b32_smoke=B32_B22_SMOKE|B32_B24_SMOKE
            fatal=False
            while len(done)<len(groups):
                deadline_reached=time.time()>=state["deadline_epoch"]
                gpu_budget_reached=(state["gate_spent_gpu_seconds"]+
                                    state["spent_gpu_seconds"])>=a.max_gpu_hours*3600
                # Include running jobs and a conservative next-job
                # reservation before admitting more GPU work.
                live_gpu_s=sum(time.monotonic()-v["start"] for v in active.values())
                remaining_gpu_s=(a.max_gpu_hours*3600
                                 -state["gate_spent_gpu_seconds"]
                                 -state["spent_gpu_seconds"]-live_gpu_s)
                halted=(STOP or (run/"STOP_AFTER_ACTIVE").exists()
                        or deadline_reached or gpu_budget_reached or fatal)
                cards=collect(own_roots=[v["proc"].pid for v in active.values()])
                for idx,v in list(active.items()):
                    card=next((c for c in cards if c.index==v["gpu"]),None)
                    if not card or card.visibility!="known":
                        v["contaminated"]=True
                    if card and card.foreign_pids:v["contaminated"]=True
                    v["samples"]+=1
                    rc=v["proc"].poll()
                    if rc is None:
                        elapsed=time.monotonic()-v["start"]
                        g=groups[idx]
                        limit=(32 if 1060<=idx<1160 else
                               16 if idx<100 else 1)*REFERENCE_SEC*3+60
                        if elapsed>limit:
                            if not fatal:
                                event(run,"WORKER_TIMEOUT_WAITING_TO_DRAIN",
                                      index=idx,elapsed=elapsed,limit=limit)
                            fatal=True
                        continue
                    state["spent_gpu_seconds"]+=time.monotonic()-v["start"]
                    active.pop(idx)
                    if rc!=0:
                        attempt=int(attempts.get(str(idx),0))
                        event(run,"WORKER_FAILED",index=idx,rc=rc,
                              attempt=attempt,log=v["log"])
                        # Exactly one same-seed retry. Quarantine any partial
                        # output without deleting it or treating it as valid.
                        if attempt<2:
                            path=run/"gpu"/v["task_id"]
                            if path.exists():
                                assert_ok(path.is_dir() and not path.is_symlink(),
                                          "FAILED_TASK_OUTPUT_SYMLINK")
                                quarantine=run/"quarantine"/(
                                    v["task_id"]+"_failed_try"+str(attempt))
                                assert_ok(not quarantine.exists(),
                                          "QUARANTINE_ALREADY_EXISTS")
                                path.rename(quarantine)
                                event(run,"FAILED_ATTEMPT_QUARANTINED",
                                      index=idx,path=str(quarantine))
                            event(run,"ONE_RETRY_AUTHORIZED",
                                  index=idx,next_attempt=attempt+1)
                        else:
                            fatal=True
                        store("B16_PRODUCTION_RETRY_PENDING" if not fatal
                              else "STOP_B16_PRODUCTION_WORKER")
                        continue
                    try:
                        secs=verify_task(run,groups,msha,idx,a.expected_head)
                    except Exception as exc:
                        event(run,"TASK_RECEIPT_FAILED",index=idx,
                              reason=str(exc),log=v["log"])
                        fatal=True
                        continue
                    done.add(idx)
                    event(run,"ROOTS_COMPLETE",index=idx,
                          n=groups[idx]["new_roots"],
                          native_sample_s=round(secs,4),
                          contaminated=v["contaminated"])
                    store("B16_PRODUCTION_RUNNING")
                if set(range(B16_TASKS))<=done:
                    complete=run/"XP_A11_B16_PROD_COMPLETE.json"
                    if not complete.exists():
                        proof=dict(status="B16_PRODUCTION_GENERATION_PASS_EMBARGOED",
                            head=a.expected_head,manifest_sha256=msha,
                            n_new_roots=2160,n_b22=1200,n_b24=960,
                            no_new_outcomes_opened=True,
                            inherited_first_start_epoch=state["first_start_epoch"],
                            inherited_deadline_epoch=state["deadline_epoch"])
                        write_x(complete,proof)
                        event(run,"B16_PRODUCTION_FULL_PASS_EMBARGOED",n_new=2160)
                if halted or fatal:
                    if not active:break
                    time.sleep(a.poll);continue
                b22ok=B22_SMOKE<=done
                b24ok=B24_SMOKE<=done
                if b22ok and not state["b22_smoke_pass"]:
                    state["b22_smoke_pass"]=True
                    event(run,"B22_SMOKE_NATIVE_PASS",n=24)
                if b24ok and not state["b24_smoke_pass"]:
                    state["b24_smoke_pass"]=True
                    event(run,"B24_SMOKE_NATIVE_PASS",n=4)
                if len(active)<a.max_parallel:
                    if not state["b22_smoke_pass"] or not state["b24_smoke_pass"]:
                        possible=sorted(smoke)
                    elif not set(range(B16_TASKS))<=done:
                        possible=range(B16_TASKS)
                    elif not state["b32_b22_smoke_pass"] or not state["b32_b24_smoke_pass"]:
                        possible=sorted(b32_smoke)
                    else:
                        possible=range(len(groups))
                    idx=next((i for i in possible if i not in done and i not in active),None)
                    if idx is not None:
                        conservative_s=((32 if 1060<=idx<1160
                                         else 16 if idx<100 else 1)*
                                         REFERENCE_SEC*1.5+60)
                        if remaining_gpu_s<conservative_s:
                            event(run,"GPU_BUDGET_TOO_LOW_FOR_NEXT_GROUP",
                                  index=idx,remaining_gpu_s=remaining_gpu_s,
                                  conservative_s=conservative_s)
                            store("B16_PRODUCTION_DRAINED_BUDGET")
                            if not active:break
                            time.sleep(a.poll)
                            continue
                        occupied={v["gpu"] for v in active.values()}
                        # Co-tenancy for generation is authorized; strictly
                        # refuse unknown visibility and insufficient free VRAM.
                        available=[c for c in cards if c.index not in occupied and
                                   c.visibility=="known"]
                        gpu,_=choose_gpu(available,VRAM_REQUIRED_MIB,
                            clean_only=False,safety_margin_mib=MARGIN_MIB,
                            fraction=.80)
                        if gpu is not None:
                            attempt=int(attempts.get(str(idx),0))+1
                            attempts[str(idx)]=attempt
                            v=launch(run,a.expected_head,groups,idx,gpu,attempt)
                            active[idx]=v
                            event(run,"START",index=idx,gpu=gpu,pid=v["proc"].pid)
                            store("B16_PRODUCTION_RUNNING")
                time.sleep(a.poll)
            if fatal:
                store("STOP_B16_PRODUCTION_WORKER_OR_RECEIPT")
                print("ACTION NEEDED: B16_PRODUCTION_WORKER_OR_RECEIPT",
                      "tasks",len(done),"/2440",flush=True)
                return 8
            if len(done)==len(groups):
                store("B16_AND_B32_GENERATION_PASS_EMBARGOED")
                assert_ok(sum(groups[i]["new_roots"] for i in done)==5040,
                          "COMPLETE_5040")
                proof=dict(status="B32_BACKGROUND_GENERATION_PASS_EMBARGOED",
                    head=a.expected_head,manifest_sha256=msha,
                    n_new_roots=2880,n_b22=1600,n_b24=1280,
                    B16_new_roots_already_verified=2160,
                    no_new_outcomes_opened=True,
                    inherited_first_start_epoch=state["first_start_epoch"],
                    inherited_deadline_epoch=state["deadline_epoch"],
                    campaign_gpu_hours=(state["gate_spent_gpu_seconds"]+
                                        state["spent_gpu_seconds"])/3600)
                write_x(run/"XP_A12_B32_COMPLETE.json",proof)
                print("B16_AND_B32_GENERATION_PASS_EMBARGOED",
                      "new_roots=5040",flush=True)
                return 0
            store("B16_PRODUCTION_DRAINED_RESUMABLE")
            print("B16_B32_PRODUCTION_DRAINED_RESUMABLE",
                  "tasks",len(done),"/2440",
                  "new_roots",sum(groups[i]["new_roots"] for i in done),
                  flush=True)
            return 0
        finally:
            # Unexpected controller exceptions must not orphan its own workers.
            # Wait, don't signal/kill any own or foreign GPU process.
            if active:
                event(run,"UNEXPECTED_DRAIN",indices=sorted(active))
                for idx,v in list(active.items()):
                    v["proc"].wait()
                    state["spent_gpu_seconds"]+=time.monotonic()-v["start"]
                    active.pop(idx)
                store("STOP_UNEXPECTED_DRAINED_UNVERIFIED")

if __name__=="__main__":
    signal.signal(signal.SIGTERM,on_stop)
    signal.signal(signal.SIGINT,on_stop)
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--max-parallel",type=int,default=4)
    p.add_argument("--max-gpu-hours",type=float,default=48.)
    p.add_argument("--poll",type=int,default=20)
    try:sys.exit(main(p.parse_args()))
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: B16_AUTOMATIC_HANDOFF "+str(e),flush=True)
        sys.exit(8)
