#!/usr/bin/env python3
"""A12 G1a autonomous replay -> 1600 generation, fully embargo-safe.

Fail closed on missing cumulative project ledger, wrong inputs, GPU source,
historical replay mismatch, oversubscribed VRAM, deadline, or stage budget.
Never scores XP-D4 and never modifies/terminates foreign GPU processes.

Ledger: authoritative input must enumerate project stages from XP-0 and
have a checked project_total_before_g1a_gpu_h. Controller itself charges
*every* replay and production worker attempt from process start to reap.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from PIL import Image,ImageChops
from scripts.track_r.xp_a12_g1a_prepare_cpu import (
  WT,ROOT,BASE,HIST_SHA,HIST_HEAD,FAST_HEAD,gate)
from scripts.track_r.xp_a6_supervisor import output_dir as hist_output_dir
from scripts.track_r.xp_fast_gpu import (
  root_folder,validate_complete,sha)
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

FAST_PY=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
HIST_RUN=BASE/"xp_a6_20261027T165112Z"
HIST_A6_HEAD="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
DEADLINE=datetime.datetime(2026,11,3,12,0,tzinfo=ZoneInfo("America/Detroit"))
STAGE_CAP_GPU_H=25.
A12_CAMPAIGN_CAP_GPU_H=150.
PROJECT_CAP_GPU_H=400.
# A10 B16 + all DAPS replay attempts + F8, already reconciled.
A10_FINAL_GPU_H=55.54578541457511
REQUEST_MIB=8192
RESERVE_SECONDS=900
POLL_SECONDS=5
MAX_ATTEMPTS=2
STOP=False
def stopped(sig,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGINT,stopped)
signal.signal(signal.SIGTERM,stopped)
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):
    return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p)
    temp=p.with_name(p.name+".tmp-"+str(os.getpid()))
    with temp.open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    os.replace(temp,p)
def git(*xs):
    return subprocess.check_output(["git","-C",str(WT),*xs],
                                   text=True,timeout=60).rstrip("\n")
def emit(root,label,**kw):
    with (root/"XP_A12_G1A_EVENTS.jsonl").open("a") as f:
        f.write(json.dumps(dict(utc=now(),event=label,**kw),
                           sort_keys=True,allow_nan=False)+"\n")
def preflight(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","SUPERVISOR_CPU_ONLY")
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"WORKTREE_HEAD")
    gate(FAST_PY.exists(),"FAST_NATIVE_ENV")
    prep=Path(a.prep)
    gate(prep.parent==BASE and prep.name.startswith("xp_a12_g1a_prepare_")
         and prep.is_dir(),"PREP_NAMESPACE")
    receipt=read(prep/"XP_A12_G1A_PREP_RECEIPT.json")
    gate(receipt["status"]=="XP_A12_G1A_GT_FREE_PREP_PASS" and
         receipt["head"]==a.expected_head and
         receipt["n_new"]==1600 and receipt["n_replay"]==2,
         "FROZEN_PREP_RECEIPT")
    file1=prep/"XP_A12_G1A_REPLAY2_TASKS.json"
    file2=prep/"XP_A12_G1A_1600_TASKS.json"
    gate(sha(file1)==receipt["replay_manifest_sha256"] and
         sha(file2)==receipt["production_manifest_sha256"],
         "MANIFEST_DIGEST")
    re=read(file1);mf=read(file2)
    gate(len(re["rows"])==2 and len(mf["rows"])==1600 and
         re["head"]==a.expected_head and mf["head"]==a.expected_head and
         re["gt_paths_supplied"] is False and
         mf["gt_paths_supplied"] is False and
         mf["outcome_embargo"] is True,"MANIFEST_CONTRACT")
    gate([r["rep"] for r in re["rows"]]==[4,4] and
         [r["rep"] for r in mf["rows"][:16]]==list(range(16,32)) and
         len({r["image_id"] for r in mf["rows"]})==100 and
         all([r["rep"] for r in mf["rows"][i:i+16]]==
             list(range(16,32)) for i in range(0,1600,16)),
         "IMAGE_ALIGNED_GRID")
    # The official D1 12-run extension is already done. Reproducing
    # two roots from distinct images certifies the same original sampler.
    hdir=HIST_RUN/"gpu/D1_P2/chunk_00000_024"
    gate(hdir.is_dir(),"ORIGINAL_D1_CHUNK_MISSING")
    old=read(hdir/"R0_5_CHUNK_COMPLETE.json")
    gate(old["status"]=="R0_5_CHUNK_PASS" and
         old["start"]==0 and old["count"]==24 and
         old["new"]==24 and old["reused"]==0 and
         old["head"]==HIST_A6_HEAD,"HISTORICAL_CHUNK_RECEIPT")
    refs=[]
    for r in re["rows"]:
        path=root_folder(hdir,r)
        prev=validate_complete(path,r,HIST_A6_HEAD,HIST_SHA)
        gate(prev["rep"]==4 and prev["dataset"]=="imagenet" and
             prev["gt_access"] is False,"HISTORICAL_ROOT")
        refs.append(dict(uid=r["uid"],
            original_png_path=str(path/"reconstruction.png"),
            original_png_sha256=prev["reconstruction_png_sha256"],
            original_native_sample_wall_seconds=prev["native_sample_wall_seconds"]))
    # Exact cumulative ledger is mandatory for the FIRST A12 receipt.
    # Never guess from the planner's ~123 GPU-h approximation.
    ledger=Path(a.project_ledger)
    gate(ledger.is_file() and not ledger.is_symlink()
         and ledger.parent==BASE,"PROJECT_LEDGER_MISSING")
    L=read(ledger)
    gate(L.get("status")=="XP_A12_CUMULATIVE_PROJECT_LEDGER_VERIFIED"
         and L.get("project_cap_gpu_h")==400 and
         L.get("A12_campaign_base_gpu_h")==A10_FINAL_GPU_H and
         L.get("has_every_stage_since_XP0") is True and
         isinstance(L.get("stages"),list) and len(L["stages"])>=8 and
         all(isinstance(x.get("gpu_h"),(int,float)) and x["gpu_h"]>=0
             and x.get("stage") and x.get("source_receipt_sha256")
             for x in L["stages"]) and
         len({x["stage"] for x in L["stages"]})==len(L["stages"]),
         "PROJECT_LEDGER_SCHEMA")
    project_h=sum(float(x["gpu_h"]) for x in L["stages"])
    gate(abs(project_h-L["project_total_before_A12_gpu_h"])<1e-4 and
         project_h+A12_CAMPAIGN_CAP_GPU_H<=PROJECT_CAP_GPU_H,
         "PROJECT_HARD_CAP_HEADROOM")
    gate("A10" in " ".join(x["stage"] for x in L["stages"]) and
         abs(sum(float(x["gpu_h"]) for x in L["stages"]
                 if x.get("stage")=="A10_B16_REPLAY_F8")-A10_FINAL_GPU_H)<1e-4,
         "A10_RECONCILIATION")
    for x in L["stages"]:
        p=Path(x["source_receipt_path"])
        gate(p.is_file() and sha(p)==x["source_receipt_sha256"],
             "PROJECT_LEDGER_SOURCE_DRIFT_"+x["stage"])
    gate(datetime.datetime.now(ZoneInfo("America/Detroit"))<DEADLINE,
         "G1A_DEADLINE")
    return prep,receipt,mf,re,ledger,project_h,refs
def workers():
    return [dict(stage="replay",index=i,start=i,count=1) for i in range(2)]
def production():
    return [dict(stage="production",index=i//16,
                 start=i,count=16) for i in range(0,1600,16)]
def command(a,job,gpu,run,prep,receipt):
    stage=job["stage"]
    mfile=prep/("XP_A12_G1A_REPLAY2_TASKS.json"
                if stage=="replay" else "XP_A12_G1A_1600_TASKS.json")
    digest=(receipt["replay_manifest_sha256"] if stage=="replay"
            else receipt["production_manifest_sha256"])
    dest=run/stage/f"chunk_{job['start']:05d}_{job['count']:03d}"
    args=[str(FAST_PY),str(WT/"scripts/track_r/xp_a12_g1a_fast_gpu.py"),
          "--expected-head",a.expected_head,
          "--roots",str(mfile),"--roots-sha",digest,
          "--out",str(dest),"--start",str(job["start"]),
          "--count",str(job["count"]),"--physical-gpu",str(gpu),
          "--dataset","imagenet","--oversample","2.0"]
    if job.get("attempt",0)>0:args+=["--resume"]
    return args,dest,digest
def charge(root,job,v,rc):
    elapsed=time.monotonic()-v["start"]
    name="XP_A12_G1A_COST_"+job["stage"]+"_"+str(job["index"])+\
         "_A"+str(job["attempt"])+".json"
    p=root/"cost"/name
    gate(not p.exists(),"DUPLICATE_COST_RECEIPT")
    write(p,dict(status="XP_A12_G1A_ATTEMPT_GPU_COST",
                 stage=job["stage"],index=job["index"],
                 start=job["start"],count=job["count"],
                 attempt=job["attempt"],gpu=v["gpu"],
                 pid=v["proc"].pid,returncode=rc,
                 elapsed_seconds=elapsed,charged_gpu_h=elapsed/3600,
                 foreign_pids_seen=sorted(v["foreign"]),
                 max_used_mib=v["max_used_mib"],n_samples=v["samples"],
                 log=str(v["log"]),utc=now()))
    return elapsed
def verify(root,prep,receipt,job,dest,digest,refs):
    rows=read(prep/("XP_A12_G1A_REPLAY2_TASKS.json"
                    if job["stage"]=="replay"
                    else "XP_A12_G1A_1600_TASKS.json"))["rows"]
    meta=read(dest/"R0_5_CHUNK_META.json")
    completed=read(dest/"R0_5_CHUNK_COMPLETE.json")
    gate(meta["head"]==receipt["head"] and
         meta["root_sha256"]==digest and
         meta["start"]==job["start"] and
         meta["count"]==job["count"] and
         meta["uids"]==[x["uid"] for x in rows[job["start"]:job["start"]+job["count"]]]
         and completed["status"]=="R0_5_CHUNK_PASS"
         and completed["head"]==receipt["head"]
         and completed["new"]==job["count"] and
         completed["meta_sha256"]==sha(dest/"R0_5_CHUNK_META.json"),
         "NATIVE_FAST_CHUNK_SCHEMA")
    for k,r in enumerate(rows[job["start"]:job["start"]+job["count"]]):
        p=root_folder(dest,r)
        rec=validate_complete(p,r,receipt["head"],digest)
        gate(rec["gt_access"] is False
             and rec["dataset"]=="imagenet"
             and rec["method"]=="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5",
             "NATIVE_FAST_RECONSTRUCTION")
        if job["stage"]=="replay":
            reference=refs[job["start"]+k]
            gate(reference["uid"]==r["uid"] and
                 sha(reference["original_png_path"])==
                     reference["original_png_sha256"],
                 "HISTORICAL_REFERENCE")
            # Pixel equality is stronger than A12's 0.1 dB +
            # identical label tolerance, with no GT/labels opened.
            with Image.open(reference["original_png_path"]) as orig, \
                 Image.open(p/"reconstruction.png") as new:
                gate(orig.mode==new.mode and orig.size==new.size and
                     ImageChops.difference(orig,new).getbbox() is None,
                     "HISTORICAL_REPLAY_PIXEL_MISMATCH_"+r["uid"])
    return sha(dest/"R0_5_CHUNK_COMPLETE.json")
def stage_loop(a,run,prep,receipt,refs,stage,state):
    jobs=workers() if stage=="replay" else production()
    complete=state["done_"+stage]
    active={}
    fatal_replay_failure=False
    def store(status):
        state["status"]=status
        state["active"]=[dict(stage=job["stage"],index=job["index"],
               gpu=v["gpu"],pid=v["proc"].pid,attempt=job["attempt"])
               for job,v in active.values()]
        write(run/"XP_A12_G1A_CONTROLLER_STATE.json",state)
    store("XP_A12_G1A_"+stage.upper()+"_RUNNING")
    while len(complete)<len(jobs):
        cards=collect(own_roots=[v["proc"].pid for job,v in active.values()])
        for index,(job,v) in list(active.items()):
            card=next((c for c in cards if c.index==v["gpu"]),None)
            if card:
                v["foreign"].update(card.foreign_pids)
                v["samples"]+=1
                v["max_used_mib"]=max(v["max_used_mib"],card.used_mib)
                if card.used_mib>0.80*card.total_mib and v["proc"].poll() is None:
                    emit(run,"A12_VRAM_RUNTIME_STOP_OWN_PID",
                         gpu=v["gpu"],pid=v["proc"].pid,
                         used=card.used_mib,total=card.total_mib)
                    try:os.killpg(v["proc"].pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            ret=v["proc"].poll()
            if ret is None:continue
            v["handle"].close()
            seconds=charge(run,job,v,ret)
            state["spent_gpu_seconds"]+=seconds
            del active[index]
            if ret==0:
                try:
                    digest=verify(run,prep,receipt,job,v["dest"],
                                  v["manifest_sha"],refs)
                    complete[str(index)]=digest
                    emit(run,"CHUNK_VERIFIED",stage=stage,
                         index=index,count=job["count"],digest=digest)
                except Exception as exc:
                    emit(run,"CHUNK_FAILED_VERIFICATION",stage=stage,
                         index=index,error=str(exc))
                    state["failures"].append(dict(stage=stage,index=index,
                         attempt=job["attempt"],why=str(exc)))
                    if stage=="replay":
                        # Stop only our remaining replay worker; account
                        # its elapsed GPU-hours before returning a STOP.
                        fatal_replay_failure=True
                        for other_job,other_v in active.values():
                            if other_v["proc"].poll() is None:
                                try:os.killpg(other_v["proc"].pid,signal.SIGTERM)
                                except ProcessLookupError:pass
            else:
                state["failures"].append(dict(stage=stage,index=index,
                  attempt=job["attempt"],why="worker_exit_"+str(ret)))
                emit(run,"CHUNK_NONZERO",stage=stage,index=index,
                     attempt=job["attempt"],rc=ret,log=str(v["log"]))
            store("XP_A12_G1A_"+stage.upper()+"_RUNNING")
        if fatal_replay_failure:
            reason="HISTORICAL_REPLAY_MISMATCH"
        elif STOP or (run/"STOP_G1A").exists():
            reason="OPERATOR_STOP"
        elif datetime.datetime.now(ZoneInfo("America/Detroit"))>=DEADLINE:
            reason="STAGE_DEADLINE"
        elif state["spent_gpu_seconds"]+sum(
                time.monotonic()-v["start"] for j,v in active.values()
                )>=STAGE_CAP_GPU_H*3600-10:
            reason="STAGE_GPUH_CEILING"
        else:reason=None
        if reason is not None:
            for job,v in active.values():
                if reason!="OPERATOR_STOP" and v["proc"].poll() is None:
                    try:os.killpg(v["proc"].pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            if not active:
                store("XP_A12_G1A_PARTIAL_"+reason)
                return False
            time.sleep(POLL_SECONDS)
            continue
        occupied={v["gpu"] for job,v in active.values()}
        while len(active)<4:
            if state["spent_gpu_seconds"]+sum(
                 time.monotonic()-v["start"] for job,v in active.values()
                )+(len(active)+1)*RESERVE_SECONDS > STAGE_CAP_GPU_H*3600:
                break
            waiting=[j for j in jobs if str(j["index"]) not in complete
                     and j["index"] not in active
                     and int(state["attempts"].get(
                            stage+":"+str(j["index"]),0))<MAX_ATTEMPTS]
            if not waiting:break
            job=dict(waiting[0])
            choices=[c for c in cards if c.index not in occupied
                     and c.visibility=="known"]
            gpu,reasons=choose_gpu(choices,REQUEST_MIB,
                     safety_margin_mib=2048,fraction=.80,clean_only=False)
            if gpu is None:
                if not active:emit(run,"WAIT_GPU_VRAM",why=reasons)
                break
            key=stage+":"+str(job["index"])
            job["attempt"]=int(state["attempts"].get(key,0))
            cmd,dest,digest=command(a,job,gpu,run,prep,receipt)
            gate((job["attempt"]==0 and not dest.exists()) or
                 (job["attempt"]>0 and dest.exists()),
                 "NATIVE_RESUME_OUTPUT_STATE")
            log=run/"logs"/f"{stage}_{job['index']:03d}_a{job['attempt']}.log"
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WT),
             PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
             OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
             MKL_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2",
             TMPDIR=str(ROOT/"tmp"))
            handle=log.open("x")
            proc=subprocess.Popen(cmd,stdout=handle,
                  stderr=subprocess.STDOUT,cwd=str(WT),
                  env=env,start_new_session=True)
            active[job["index"]]=(job,dict(proc=proc,
               gpu=gpu,start=time.monotonic(),log=log,handle=handle,
               manifest_sha=digest,dest=dest,foreign=set(),
               samples=0,max_used_mib=0))
            occupied.add(gpu)
            state["attempts"][key]=job["attempt"]+1
            emit(run,"CHUNK_LAUNCHED",stage=stage,index=job["index"],
                 start=job["start"],count=job["count"],gpu=gpu,pid=proc.pid)
            store("XP_A12_G1A_"+stage.upper()+"_RUNNING")
        if not active and len(complete)!=len(jobs) and all(
             int(state["attempts"].get(stage+":"+str(j["index"]),0))>=MAX_ATTEMPTS
             for j in jobs if str(j["index"]) not in complete):
            store("XP_A12_G1A_PARTIAL_FAILED_ATTEMPTS")
            return False
        time.sleep(POLL_SECONDS)
    store("XP_A12_G1A_"+stage.upper()+"_PASS")
    return True
def main(a):
    prep,receipt,mf,re,ledger,proj_h,refs=preflight(a)
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a12_g1a_prod_")
         and not run.exists() and not run.is_symlink(),"FRESH_RUN_ONLY")
    lock=BASE/"XP_A12_G1A_CONTROLLER.lock"
    with lock.open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run.mkdir(mode=0o700)
        for sub in ("cost","logs","replay","production"):
            (run/sub).mkdir()
        state=dict(status="A12_G1A_ADMITTED",
          expected_head=a.expected_head,preparation_sha256=sha(prep/"XP_A12_G1A_PREP_RECEIPT.json"),
          project_ledger_sha256=sha(ledger),project_before_A12_gpu_h=proj_h,
          A10_before_A12_gpu_h=A10_FINAL_GPU_H,
          A12_total_ceiling_gpu_h=150,
          A12_stage_ceiling_gpu_h=25,
          project_cap_gpu_h=400,
          spent_gpu_seconds=0.,done_replay={},done_production={},
          attempts={},failures=[],active=[],D4_outcomes_opened=False)
        write(run/"XP_A12_G1A_CONTROLLER_STATE.json",state)
        # First receipt contains full historical project ledger to XP-0.
        write(run/"XP_A12_FIRST_LEDGER_RECEIPT.json",dict(
          status="XP_A12_CUMULATIVE_PROJECT_LEDGER_PUBLISHED",
          source_sha256=sha(ledger),project_total_before_A12_gpu_h=proj_h,
          all_stages_since_XP0=read(ledger)["stages"],A12_limit_gpu_h=150,
          A10_reconciled_gpu_h=A10_FINAL_GPU_H,
          A12_G1a_stage_ceiling_gpu_h=25,
          project_cap_gpu_h=400,
          no_D4_outcomes_opened=True))
        emit(run,"FIRST_PROJECT_LEDGER_PUBLISHED",project_gpu_h=proj_h)
        try:
            print("A12_G1A_TWO_HISTORICAL_ROOT_REPLAY_BEGIN",flush=True)
            ok=stage_loop(a,run,prep,receipt,refs,"replay",state)
            gate(ok and len(state["done_replay"])==2,"REPLAY_NOT_COMPLETE")
            rp=dict(status="XP_A12_G1A_REPLAY2_PIXEL_EXACT_PASS",
              n=2,original_A6_head=HIST_A6_HEAD,
              historical_source_SHA=HIST_SHA,original_UIDs=[r["uid"] for r in re["rows"]],
              replay_receipt_shas=state["done_replay"],
              gpu_h=state["spent_gpu_seconds"]/3600,
              no_GT_opened=True,no_labels_opened=True)
            write(run/"XP_A12_G1A_REPLAY_GATE.json",rp)
            print("XP_A12_G1A_REPLAY2_PIXEL_EXACT_PASS",flush=True)
            print("A12_G1A_1600_AUTODISTRIBUTED_BEGIN",flush=True)
            stage_loop(a,run,prep,receipt,refs,"production",state)
        finally:
            # The stage loop normally drains processes. If an unhandled
            # exception occurs while any are still active, refuse to
            # advertise completion; leave a durable STOP receipt. Process
            # groups are not silently detached for automatic retries.
            state=read(run/"XP_A12_G1A_CONTROLLER_STATE.json")
            if state["active"]:
                write(run/"XP_A12_G1A_STOP_NEEDS_PID_AUDIT.json",
                      dict(status="STOP_INCOMPLETE_CONTROLLER_NEEDS_PID_AUDIT",
                      active=state["active"],utc=now()))
                print("ACTION NEEDED: G1A_ACTIVE_PIDS_AUDIT",state["active"],flush=True)
        allpass=len(state["done_production"])==100
        summary=dict(status=("XP_A12_G1A_1600_GENERATION_COMPLETE_SEALED"
                              if allpass else "XP_A12_G1A_PARTIAL_STOP"),
          completed=16*len(state["done_production"]),total=1600,
          replay_pass=len(state["done_replay"])==2,
          failed_attempts=len(state["failures"]),
          stage_gpu_h=state["spent_gpu_seconds"]/3600,
          G1a_ceiling_gpu_h=25.,
          A12_used_including_A10_gpu_h=A10_FINAL_GPU_H+state["spent_gpu_seconds"]/3600,
          project_total_gpu_h=proj_h+state["spent_gpu_seconds"]/3600,
          project_cap_gpu_h=400,
          outcome_sealed=True,GT_not_read=True,ledger_sha256=sha(ledger),
          source_head=a.expected_head)
        gate(summary["stage_gpu_h"]<=STAGE_CAP_GPU_H+0.05,
             "CEILING_OVERSPEND")
        write(run/"XP_A12_G1A_FINAL_STATUS.json",summary)
        print("A12_G1A_FINAL",json.dumps(summary,sort_keys=True),flush=True)
        print("A12_G1A_RUN",run,flush=True)
        if allpass:print("XP_A12_G1A_ALL_1600_SEALED_PASS",flush=True)
        else:print("ACTION NEEDED: A12_G1A_PARTIAL",flush=True)
def entry():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--prep",required=True)
    p.add_argument("--project-ledger",required=True)
    p.add_argument("--run",required=True)
    a=p.parse_args()
    try:main(a)
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A12_G1A_AUTOPILOT "+str(exc),flush=True)
        sys.exit(8)
if __name__=="__main__":entry()
