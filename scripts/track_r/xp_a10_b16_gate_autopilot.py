#!/usr/bin/env python3
"""A10 B16 400-root replay autopilot: clean-GPU, restartable, 12h cap.

Single-writer per immutable PAC run; independent four-candidate image groups.
No GT or B16 new-run scoring in GPU workers. Re-admit when foreign processes
leave, quarantine a completed but timing-contaminated replay image (never
overwrite it), and retry only on a newly clean GPU. Never signal foreign PIDs.
When all 400 PASS, perform the independent historical-only CPU gate offline.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path

from scripts.track_r.xp_a10_b16_gate_prepare_cpu import (
    ROOT,BASE,WT,A6,A6_HEAD,git,safe,sha)
from scripts.track_r.xp_a10_b16_gate_gpu import image_folder
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

PY_CPU=ROOT/"conda-envs/prdiff_track_r/bin/python"
PY_SITCOM=ROOT/"conda-envs/sitcom_ode_bw/bin/python"
CACHE=ROOT/"outputs/.cache/track_r"
SAMPLE_REFERENCE_SEC=58.343
GPU_VRAM_MIB=32000
STOP=False

def signal_stop(signum,frame):
    global STOP
    STOP=True

def require(ok,name):
    if not ok:raise RuntimeError("STOP_B16_AUTOPILOT_"+name)

def dump(path,obj):
    p=Path(path)
    temp=p.with_name("."+p.name+"."+str(os.getpid()))
    with temp.open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush();os.fsync(f.fileno())
    os.replace(temp,p)

def record(run,action,**kw):
    with (run/"B16_GATE_EVENTS.jsonl").open("a") as f:
        f.write(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action":action,**kw},sort_keys=True,allow_nan=False)+"\n")
        f.flush();os.fsync(f.fileno())

def root_folder(group,uid):
    return group/("root_"+hashlib.sha256(uid.encode()).hexdigest()[:24])

def verify_group(run,mf,index,head,*,check_times=True):
    rows=mf["rows"][index*4:index*4+4]
    out=image_folder(run,rows[0]["image_id"])
    task=json.loads(safe(out/"XP_SITCOM_WORKER_COMPLETE.json").read_text())
    require(task["status"]=="XP_SITCOM_WORKER_PASS" and
            task["head"]==head and task["new_roots"]==4 and
            task["manifest_sha256"]==mf["_manifest_sha"],"GROUP_COMPLETE")
    secs=[]
    for r in rows:
        path=root_folder(out,r["uid"])
        receipt=json.loads(safe(path/"XP_SITCOM_ROOT_COMPLETE.json").read_text())
        require(receipt["status"]=="XP_SITCOM_REPLAY_ROOT_PASS" and
                receipt["uid"]==r["uid"] and
                receipt["image_id"]==r["image_id"] and
                receipt["rep"]==r["rep"] and
                receipt["runner_head"]==head and
                receipt["manifest_sha256"]==mf["_manifest_sha"] and
                receipt["measurement_sha256"]==r["measurement_sha256"] and
                receipt["gt_access"] is False and
                receipt["historical_access"] is False and
                sha(path/"reconstruction.pt")==receipt["reconstruction_pt_sha256"] and
                sha(path/"reconstruction.png")==receipt["reconstruction_png_sha256"],"ROOT_RECEIPT")
        secs.append(float(receipt["sampling_wall_s"]))
    require(all(math.isfinite(s) and 1<s<7200 for s in secs),"TIMING_FINITE")
    if check_times:
        med=sorted(secs)[1:3]
        val=sum(med)/2
        require(0.5*SAMPLE_REFERENCE_SEC<=val<=1.5*SAMPLE_REFERENCE_SEC,
                "SMOKE_SPEED")
    return secs

def validate_start(run,head):
    require(run.parent==BASE and run.name.startswith("xp_a10_b16_gate_")
            and run.is_dir() and not run.is_symlink(),"RUN_PATH")
    require(git(WT,"rev-parse","HEAD")==head and
            not git(WT,"status","--porcelain"),"GIT")
    require(git(A6,"rev-parse","HEAD")==A6_HEAD and
            not git(A6,"status","--porcelain"),"A6")
    prep=json.loads(safe(run/"XP_A10_B16_GATE_PREP.json").read_text())
    mf_path=run/"XP_A10_B16_GATE_TASKS.json"
    require(prep.get("status")=="XP_A10_B16_GATE_PREP_PASS" and
            prep["head"]==head and prep["n_roots"]==400 and
            prep["n_images"]==100 and
            sha(mf_path)==prep["manifest_sha256"],"FROZEN_PREP")
    mf=json.loads(mf_path.read_text())
    require(mf["head"]==head and len(mf["rows"])==400 and
            mf["GT_supplied"] is False and
            mf["historical_candidate_supplied"] is False,
            "FROZEN_MANIFEST")
    mf["_manifest_sha"]=prep["manifest_sha256"]
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CONTROLLER_GPU_VISIBILITY")
    return mf

def spawn(run,head,index,gpu):
    rows=json.loads(safe(run/"XP_A10_B16_GATE_TASKS.json").read_text())["rows"]
    iid=rows[4*index]["image_id"]
    log=run/"logs"/f"B16_{index:03d}_{iid}.log"
    require(not log.exists(),"LOG_EXISTS")
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
               PYTHONPATH=str(WT),
               PYTHONPYCACHEPREFIX=str(run/"pycache"/f"B16_{index:03d}"),
               TMPDIR=str(CACHE/"tmp"),
               XDG_CACHE_HOME=str(CACHE),
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    cmd=[str(PY_SITCOM),str(WT/"scripts/track_r/xp_a10_b16_gate_gpu.py"),
         "--run",str(run),"--expected-head",head,
         "--index",str(index),"--gpu",str(gpu)]
    # No shell, no foreign process inspection/modification inside worker.
    file=log.open("x")
    try:
        child=subprocess.Popen(cmd,stdout=file,stderr=subprocess.STDOUT,
                                env=env,start_new_session=True)
    finally:file.close()
    return dict(proc=child,gpu=gpu,index=index,image_id=iid,log=str(log),
                started=time.monotonic(),contaminated=False,seen_samples=0)

def cpu_gate(run,head):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1",PYTHONPATH=str(WT))
    cmd=[str(PY_CPU),str(WT/"scripts/track_r/xp_a10_b16_gate_score_cpu.py"),
         "--run",str(run),"--expected-head",head]
    with (run/"B16_GATE_CPU_SCORE.log").open("x") as file:
        p=subprocess.run(cmd,stdout=file,stderr=subprocess.STDOUT,env=env)
    if p.returncode!=0:
        raise RuntimeError("STOP_B16_GATE_CPU_SCORE_FAILED")
    result=json.loads(safe(run/"XP_A10_B16_GATE_COMPARE.json").read_text())
    require(result["status"]=="XP_A10_B16_GATE_PASS" and
            result["n_pass"]==400 and result["n_fail"]==0 and
            result["B16_new_run_labels_opened"] is False,
            "CPU_GATE_NOT_PASS")
    return result

def main(a):
    require(0<a.hours<=12 and 0<a.gpu_hours<=48 and
            1<=a.max_parallel<=4 and 5<=a.poll<=60,"ARGUMENTS")
    run=Path(a.run)
    mf=validate_start(run,a.expected_head)
    lock=run/"B16_GATE_CONTROLLER.lock"
    with lock.open("a+") as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("STOP_B16_CONTROLLER_OWNED")
        st=run/"B16_GATE_STATE.json"
        (run/"gpu").mkdir(exist_ok=True)
        (run/"logs").mkdir(exist_ok=True)
        (run/"pycache").mkdir(exist_ok=True)
        (run/"quarantine").mkdir(exist_ok=True)
        (CACHE/"tmp").mkdir(parents=True,exist_ok=True)
        if st.is_file():
            state=json.loads(st.read_text())
            require(state["head"]==a.expected_head and
                    state["manifest_sha256"]==mf["_manifest_sha"] and
                    state["deadline_epoch"]>=state["first_start_epoch"] and
                    state.get("active")==[],"RESTART_STATE")
            done=set(state["done"])
            require(all(isinstance(i,int) and 0<=i<100 for i in done),
                    "RESTART_DONE")
            for i in sorted(done):
                verify_group(run,mf,i,a.expected_head,check_times=False)
        else:
            now=time.time()
            state=dict(status="STARTED",head=a.expected_head,
                manifest_sha256=mf["_manifest_sha"],
                first_start_epoch=now,deadline_epoch=now+a.hours*3600,
                active=[],done=[],quarantined=[],spent_gpu_seconds=0.,
                B16_new_run_labels_opened=False)
            done=set()
            dump(st,state)
        active={}
        stopped=False
        all_jobs=set(range(100))
        def save(status):
            state.update(status=status,active=[
                dict(index=i,pid=v["proc"].pid,gpu=v["gpu"]) for i,v in active.items()],
                done=sorted(done))
            dump(st,state)
        try:
            while done!=all_jobs:
                if STOP or (run/"STOP_AFTER_ACTIVE").exists() or (
                    time.time()>=state["deadline_epoch"]):
                    stopped=True
                if state["spent_gpu_seconds"]>=a.gpu_hours*3600:
                    stopped=True
                # This is telemetry only; when PID visibility is unknown the
                # controller admits NOTHING and labels active replay unverified.
                own=[v["proc"].pid for v in active.values()]
                cards=collect(own_roots=own)
                for i,v in list(active.items()):
                    card=next((c for c in cards if c.index==v["gpu"]),None)
                    if (card is None or card.visibility!="known" or
                        card.foreign_pids):
                        if not v["contaminated"]:
                            record(run,"FOREIGN_OR_UNKNOWN_DETECTED",
                                   index=i,gpu=v["gpu"],foreign_pids=list(
                                       card.foreign_pids) if card else [])
                        v["contaminated"]=True
                    v["seen_samples"]+=1
                    result=v["proc"].poll()
                    if result is None:continue
                    state["spent_gpu_seconds"]+=time.monotonic()-v["started"]
                    active.pop(i)
                    if result!=0:
                        save("STOP_WORKER_FAILURE")
                        record(run,"WORKER_FAILED",index=i,returncode=result,log=v["log"])
                        stopped=True
                        continue
                    timings=verify_group(run,mf,i,a.expected_head,
                         check_times=i<2)
                    if v["contaminated"] or v["seen_samples"]<2:
                        orig=image_folder(run,v["image_id"])
                        dest=run/"quarantine"/f"{orig.name}_attempt_{len(state['quarantined']):04d}"
                        require(not dest.exists(),"QUARANTINE_EXISTS")
                        orig.rename(dest)
                        state["quarantined"].append(dict(index=i,
                            path=str(dest),reason="CONTAMINATED_OR_UNVERIFIED"))
                        record(run,"REPLAY_QUARANTINED",index=i,gpu=v["gpu"])
                    else:
                        done.add(i)
                        record(run,"IMAGE_PASS",index=i,gpu=v["gpu"],
                               sampling_sec=sum(timings))
                    save("RUNNING")
                if stopped:
                    if not active:break
                    time.sleep(a.poll);continue
                if len(active)<a.max_parallel:
                    # Must finish first two preflight/smoke groups before all
                    # remaining historical replay jobs are admitted.
                    eligible=range(2) if not {0,1}<=done else range(100)
                    idx=next((i for i in eligible if i not in done
                              and i not in active),None)
                    if idx is not None:
                        busy={v["gpu"] for v in active.values()}
                        clean_cards=[c for c in cards if c.index not in busy]
                        gpu,_=choose_gpu(clean_cards,GPU_VRAM_MIB,
                            clean_only=True,safety_margin_mib=2048,
                            fraction=0.80)
                        if gpu is not None:
                            v=spawn(run,a.expected_head,idx,gpu)
                            active[idx]=v
                            record(run,"LAUNCH",index=idx,gpu=gpu,pid=v["proc"].pid)
                            save("RUNNING")
                time.sleep(a.poll)
            if active:
                raise RuntimeError("STOP_B16_ACTIVE_CHILDREN_NOT_DRAINED")
            if done==all_jobs:
                save("GPU_400_PASS_CPU_GATE_PENDING")
                result=cpu_gate(run,a.expected_head)
                save("B16_GATE_FULL_PASS")
                record(run,"B16_GATE_FULL_PASS",n=400,
                       max_abs_delta_db=result["max_abs_delta_db"])
                print("B16_GATE_400_REPLAY_PASS",result["n_pass"],flush=True)
                return 0
            save("DRAINED_RESUMABLE" if stopped else "INCOMPLETE")
            print("B16_GATE_DRAINED_RESUMABLE",len(done),"/100 images",
                  "gpu_h",round(state["spent_gpu_seconds"]/3600,4),flush=True)
            return 0
        finally:
            save(state.get("status","STOP_UNEXPECTED"))

if __name__=="__main__":
    signal.signal(signal.SIGTERM,signal_stop)
    signal.signal(signal.SIGINT,signal_stop)
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--hours",type=float,default=12.)
    p.add_argument("--gpu-hours",type=float,default=48.)
    p.add_argument("--max-parallel",type=int,default=2)
    p.add_argument("--poll",type=int,default=20)
    try:sys.exit(main(p.parse_args()))
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: B16_GATE_AUTOPILOT",str(exc),flush=True)
        sys.exit(8)
