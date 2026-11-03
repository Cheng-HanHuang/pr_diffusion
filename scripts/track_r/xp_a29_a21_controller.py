#!/usr/bin/env python3
"""A21 XP-C3 multi-GPU controller; GPU union-time budget and frozen pilot.

All external compute is observed, never signalled. Runs exact A20 prior.
No scoring until fully pushed A21 code/constants/equivalence freeze.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from collections import defaultdict
from prdiffusion.track_r.xp_gpu_telemetry import collect
from scripts.track_r import xp_a29_a21_settings as S

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
FREEZE="docs/track_r/executors/xp/XP_C3_A21_FREEZE.json"
SOURCE="docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
R2_RUN=BASE/S.R2_RUN
LOCK=BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock"
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGINT,stop)
signal.signal(signal.SIGTERM,stop)

def gate(v,k):
    if not v:raise RuntimeError("STOP_A21_CONTROL_"+k)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_name("."+p.name+f".{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def git(*args):return subprocess.check_output(
    ["git","-C",str(WT),*args],text=True,timeout=60).strip()

def validate(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and git("rev-parse","HEAD")==a.expected_head
         and not git("status","--porcelain"),"PINNED_CLEAN_SOURCE")
    reg=read(WT/FREEZE)
    gate(git("ls-files","--error-unmatch",FREEZE)==FREEZE
         and reg["status"]=="A21_XPC3_FROZEN_BEFORE_PILOT"
         and reg["source_sha256"]==digest(WT/SOURCE)==S.SOURCE_SHA
         and reg["worker_sha256"]==digest(WT/"scripts/track_r/xp_a29_a21_gpu_worker.py")
         and reg["controller_sha256"]==digest(Path(__file__))
         and reg["settings_sha256"]==digest(WT/"scripts/track_r/xp_a29_a21_settings.py")
         and reg["gpu_occupancy_cap_h"]==S.CAP_OCCUPANCY_GPU_H
         and reg["batch_options"]==list(S.BATCH_OPTIONS)
         and reg["max_per_gpu"]==S.MAX_WORKERS_PER_GPU
         and reg["pilot_max_L_diff"]==S.EQUIV_MAX_LOSS_ABS
         and reg["pilot_min_abs_margin_for_choice"]==
              S.EQUIV_CHOICE_EXCLUDE_SMALL_MARGIN
         and reg["allowed_gpus"]==list(S.GPU_IDS),
         "A21_PUSHED_CODE_AND_CONSTANTS")
    r2final=R2_RUN/"XP_C3_A20_FINAL_STATUS.json"
    gate(r2final.is_file(),"R2_FINAL_MISSING")
    gate(reg["R2_final_receipt_sha256"]==digest(r2final)
         and reg["R2_64_smoke_sha256"]==digest(R2_RUN/"XP_N"/"000.json"),
         "FROZEN_R2_BASELINE_HASH")
    r2=read(r2final)
    gate(r2["status"]=="A20_XPC3_COMPLETE_PANELS_BUDGET_STOP"
         and r2["completed_panels"]==[]
         and 0<r2["charged_gpu_h"]<1
         and r2["source_manifest_sha256"]==S.SOURCE_SHA,
         "R2_STOP_RECEIPT")
    ref=read(R2_RUN/"XP_N"/"000.json")
    gate(ref["status"]=="A20_XPC3_IMAGE_PRIOR_PASS"
         and ref["source_manifest_sha256"]==S.SOURCE_SHA
         and len(ref["rows"])==64
         and all("L000" in x and "L111" in x for x in ref["rows"]),
         "R2_SMOKE_64_WITH_BOTH_LOSSES")
    return reg,r2,ref

def total_union(intervals,active,now=None):
    """Sum union of [start,end] per physical GPU; prior R2 is separate."""
    now=time.monotonic() if now is None else now
    by=defaultdict(list)
    for g,start,end in intervals:by[int(g)].append((float(start),float(end)))
    for v in active.values():
        by[int(v["gpu"])].append((float(v["start"]),now))
    summary={}
    for g,ls in by.items():
        total=0.;a=b=None
        for u,v in sorted(ls):
            if a is None:a,b=u,v
            elif u<=b:b=max(b,v)
            else:total+=b-a;a,b=u,v
        if a is not None:total+=b-a
        summary[str(g)]=total/3600.
    return summary

def pilot_compare(baseline,produced):
    a={int(x["run"]):x for x in baseline["rows"]}
    b={int(x["run"]):x for x in produced["rows"]}
    gate(len(a)==len(b)==64 and set(a)==set(b),"PILOT_RUN_KEYS")
    d0=max(abs(float(a[i]["L000"])-float(b[i]["L000"])) for i in a)
    d1=max(abs(float(a[i]["L111"])-float(b[i]["L111"])) for i in a)
    choices=sum(a[i]["prior_choice"]!=b[i]["prior_choice"] for i in a
                if abs(float(a[i]["prior_margin"]))>
                   S.EQUIV_CHOICE_EXCLUDE_SMALL_MARGIN)
    passflag=(d0<=S.EQUIV_MAX_LOSS_ABS and
              d1<=S.EQUIV_MAX_LOSS_ABS and choices==0)
    return dict(pass_equivalence=passflag,max_abs_dL000=d0,
        max_abs_dL111=d1,choice_disagreements_away_from_tie=choices,
        eligible_choice_count=sum(abs(float(x["prior_margin"]))>
              S.EQUIV_CHOICE_EXCLUDE_SMALL_MARGIN for x in a.values()),
        n=64)

def ready(card,peak):
    return (card.visibility=="known"
       and card.index in S.GPU_IDS
       and card.free_mib>=S.YIELD_BELOW_FREE_MIB
       and card.free_mib>=math.ceil(
           S.MEMORY_HEADROOM_FACTOR*peak)+S.MEMORY_ADMISSION_EXTRA_MIB)

def main(a):
    reg,r2,ref=validate(a)
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a25_xpc3_prod_a21_")
         and not run.exists() and not run.is_symlink(),"FRESH_ROOT")
    if a.preflight_only:
        print("A21_FROZEN_CONTROLLER_PREFLIGHT_PASS",
              "source",S.SOURCE_SHA,"prior_gpu_h",r2["charged_gpu_h"],
              "allowed_gpus",S.GPU_IDS,flush=True)
        return
    # Global lock held throughout launch; prevents other Track R GPU jobs
    # from using a stale project baseline. Do not remove this lock file.
    with LOCK.open("a+") as handle:
        print("A21_WAIT_GLOBAL_LEDGER_LOCK",flush=True)
        fcntl.flock(handle,fcntl.LOCK_EX)
        gate(not run.exists(),"RUN_RACE")
        run.mkdir(mode=0o700)
        for d in ("logs","pilot","replicate","final",*S.PANELS):
            (run/d).mkdir()
        before=float(r2["charged_gpu_h"])
        project_base=float(r2["project_gpu_h"])-before
        gate(abs(project_base-232.07834495911806)<.01,
             "R2_PROJECT_BASE_CHANGED")
        intervals=[];active={};history=[];post=[];accepted=[]
        state=dict(status="A21_XPC3_RUNNING",
           expected_head=a.expected_head,
           source_sha256=S.SOURCE_SHA,
           a21_freeze_sha256=digest(WT/FREEZE),
           r2_final_sha256=digest(R2_RUN/"XP_C3_A20_FINAL_STATUS.json"),
           prior_r2_gpu_occupancy_h=before,
           project_base_gpu_h=project_base,
           pilot={},configurations=[],
           completed_panels=[],skipped_panels=[],
           yield_events=[],oom_events=[],rescored_chunks=0,
           execution_intervals=[],worker_records=[],
           postprocess=[],no_GT_scoring=True)
        def snapshot():
            now=time.monotonic()
            by=total_union(intervals,active,now)
            used=before+sum(by.values())
            state["occupancy_gpu_h_by_gpu"]=by
            state["charged_gpu_h"]=used
            state["project_gpu_h"]=project_base+used
            state["worker_hours"]=sum(v["end"]-v["start"]
                                    for v in history)/3600. +sum(
                    now-v["start"] for v in active.values())/3600.
            state["active"]=[dict(pid=k,gpu=v["gpu"],mode=v["mode"],
                    panel=v["panel"],start=v["start"]) for k,v in active.items()]
            state["execution_intervals"]=list(intervals)
            return used
        def save():snapshot();write(run/"XP_C3_A21_CONTROLLER_STATE.json",state)
        def spawn(gpu,panel,mode,pilot=False,replicate_task=None,replicate_output=None):
            args=[str(ROOT/"conda-envs/daps/bin/python"),
                str(WT/"scripts/track_r/xp_a29_a21_gpu_worker.py"),
                "--expected-head",a.expected_head,
                "--run",str(run),"--panel",panel,
                "--gpu",str(gpu),"--mode",mode]
            if pilot:args+=["--pilot"]
            if replicate_task:
                args+=["--replicate-task",str(replicate_task),
                       "--replicate-output",str(replicate_output)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=str(WT),PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2")
            idx=len(history)+len(active)+1
            name=f"{panel}_{mode}_{'PILOT' if pilot else 'PROD'}_{idx}"
            log=run/"logs"/(name+".log")
            out=log.open("x")
            try:
                proc=subprocess.Popen(args,cwd=str(WT),env=env,
                    stdout=out,stderr=subprocess.STDOUT,
                    start_new_session=True)
            finally:out.close()
            v=dict(proc=proc,start=time.monotonic(),gpu=gpu,
                   panel=panel,mode=mode,log=str(log),pilot=pilot,
                   replicate=bool(replicate_task),
                   signal_sent=False)
            active[proc.pid]=v
            print("A21_WORKER_START",proc.pid,"gpu",gpu,
                  "panel",panel,"mode",mode,flush=True)
            return proc.pid
        def finish():
            for pid,v in list(active.items()):
                rc=v["proc"].poll()
                if rc is None:continue
                end=time.monotonic()
                v["proc"].wait()
                del active[pid]
                intervals.append([v["gpu"],v["start"],end])
                h=dict(pid=pid,gpu=v["gpu"],panel=v["panel"],mode=v["mode"],
                    pilot=v["pilot"],start=v["start"],end=end,
                    wall_seconds=end-v["start"],rc=rc,log=v["log"],
                    was_yielded=v["signal_sent"])
                history.append(h);state["worker_records"].append(h)
                reclaimed=Path(v["log"]).read_text(
                    errors="replace").count("A21_CHUNK_RECLAIM")
                state["rescored_chunks"]+=reclaimed
                if rc!=0:
                    log=Path(v["log"]).read_text(errors="replace")[-3000:]
                    if "out of memory" in log.lower():
                        state["oom_events"].append(h)
                        print("A21_GPU_OWN_OOM",pid,"gpu",v["gpu"],flush=True)
                    else:
                        raise RuntimeError("WORKER_NONZERO_"+str(pid)+"_"+log[-1200:])
                print("A21_WORKER_END",pid,"rc",rc,"seconds",
                      round(end-v["start"],2),flush=True)
        def charged():return snapshot()
        def can_run(extra=0.):
            return (not STOP and
                time.time()<S.HARD_STOP and
                charged()+extra < S.CAP_OCCUPANCY_GPU_H and
                project_base+charged()+extra<S.PROJECT_CAP_GPU_H)
        def signal_yield(pid,why):
            if pid not in active or active[pid]["signal_sent"]:return
            os.kill(pid,signal.SIGUSR1)
            active[pid]["signal_sent"]=True
            state["yield_events"].append(dict(pid=pid,gpu=active[pid]["gpu"],
                                              reason=why,utc=time.time()))
        def shutdown():
            for pid in list(active):
                try:signal_yield(pid,"CONTROLLER_STOP")
                except ProcessLookupError:pass
            t0=time.monotonic()
            while active and time.monotonic()-t0<900 and time.time()<S.HARD_STOP:
                finish();time.sleep(2)
            # Hard deadline / unbounded chunks: kill only OUR process groups.
            for pid,v in list(active.items()):
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            time.sleep(2)
            for pid,v in list(active.items()):
                if v["proc"].poll() is None:
                    try:os.killpg(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            finish()
            gate(not active,"WORKER_TERMINATION_ACCOUNTING")
        def cards():
            return {c.index:c for c in collect(own_roots=list(active))}
        def eligible_pilot():
            while True:
                gate(not STOP and time.time()<S.HARD_STOP
                     and time.monotonic()-start_pilot<
                         S.PILOT_MAX_WALL_SECONDS,
                     "PILOT_WAIT_STOP")
                pool=cards()
                candidates=[c for c in pool.values() if c.index in S.GPU_IDS
                            and c.visibility=="known"
                            and c.free_mib>=S.CLEAN_UNKNOWN_PILOT_MIN_FREE_MIB]
                if candidates:return max(candidates,key=lambda c:c.free_mib)
                time.sleep(15)
        save()
        try:
            # PILOT: exactly the 64 R2 XP-N smoke roots, no GT.
            start_pilot=time.monotonic()
            for mode in ("B8","B32"):
                gate(time.monotonic()-start_pilot<
                     S.PILOT_MAX_WALL_SECONDS-60,"PILOT_WALL_CAP")
                gate(charged()-before<S.PILOT_MAX_OCCUPANCY_GPU_H-0.1,
                     "PILOT_GPU_H_CAP")
                gpu=eligible_pilot().index
                pid=spawn(gpu,"XP_N",mode,pilot=True)
                while pid in active:
                    finish()
                    if (time.monotonic()-start_pilot>=
                            S.PILOT_MAX_WALL_SECONDS-3 or
                        charged()-before>=S.PILOT_MAX_OCCUPANCY_GPU_H-.01):
                        # Hard pilot gate, no delayed chunk completion.
                        if pid in active:
                            try:os.killpg(pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                            time.sleep(2)
                            if active[pid]["proc"].poll() is None:
                                try:os.killpg(pid,signal.SIGKILL)
                                except ProcessLookupError:pass
                        raise RuntimeError("PILOT_30MIN_OR_2GPUH_CEILING")
                    gate(time.monotonic()-start_pilot<
                         S.PILOT_MAX_WALL_SECONDS and
                         charged()-before<S.PILOT_MAX_OCCUPANCY_GPU_H,
                         "PILOT_OVER_CAP")
                    time.sleep(2)
                result_path=run/"pilot"/f"{mode}.json"
                if not result_path.is_file():
                    # CUDA OOM is a nonviable batch setting, not evidence
                    # about the estimator. Do not pretend equivalence passed.
                    gate(any(x["mode"]==mode and x["pilot"] and
                             x["rc"]!=0 for x in history),
                         "MISSING_SUCCESSFUL_PILOT_OUTPUT")
                    state["pilot"][mode]=dict(
                        mode=mode,pass_equivalence=False,
                        error="PILOT_GPU_OOM_NO_NUMERICAL_EQUIVALENCE",
                        candidate_count=0)
                    state["configurations"].append(dict(
                        stage="PILOT",**state["pilot"][mode]))
                    print("A21_PILOT_MODE_NOT_VIABLE",mode,flush=True)
                    save();continue
                result=read(result_path)
                gate(result["status"]=="A21_PILOT_BATCH_COMPLETE"
                     and result["source_sha256"]==S.SOURCE_SHA
                     and result["candidate_count"]==64,
                     "PILOT_64_SCHEMA")
                diff=pilot_compare(ref,result)
                diff.update(mode=mode,gpu=gpu,
                    seconds=result["seconds"],
                    candidates_per_worker_hour=64*3600/result["seconds"],
                    measured_peak_mib=result["peak_reserved_mib"],
                    foreign_count=len(cards()[gpu].foreign_pids),
                    output_sha256=digest(run/"pilot"/f"{mode}.json"))
                state["pilot"][mode]=diff
                state["configurations"].append(dict(
                    stage="PILOT",**diff))
                print("A21_EQUIVALENCE",json.dumps(diff,sort_keys=True),
                      flush=True)
                save()
            good=[x for x in state["pilot"].values() if x["pass_equivalence"]]
            chosen=max(good,key=lambda x:x["candidates_per_worker_hour"]) if good else None
            mode=chosen["mode"] if chosen else "R2"
            # Fallback B2 uses an observed real peak; never guess
            # when both pilot workloads have failed before measuring.
            observed_peaks=[int(x["measured_peak_mib"])
                for x in state["pilot"].values()
                if "measured_peak_mib" in x]
            gate(bool(observed_peaks),"NO_MEASURED_PEAK_FALLBACK")
            peak=int(chosen["measured_peak_mib"] if chosen else
                 max(observed_peaks))
            state["chosen_mode"]=mode
            state["measured_peak_mib"]=peak
            state["equivalence_pass"]=bool(chosen)
            state["fallback_to_original_R2"]=not bool(chosen)
            print("A21_CONFIG_CHOSEN",mode,"peak_mib",peak,
                  "gpu_occupancy_cap_h",S.CAP_OCCUPANCY_GPU_H,flush=True)
            gate(charged()-before<=S.PILOT_MAX_OCCUPANCY_GPU_H,
                 "PILOT_BILLED_LIMIT")
            save()

            # Production: image chunks are atomically claimed by the workers.
            # Priority XP_N -> G5 -> XP_D4, but skip infeasible panels.
            for panel in S.PANELS:
                if not can_run(extra=.5):
                    state["status"]="A21_XPC3_BUDGET_OR_DEADLINE_STOP";break
                n=S.SIZES[panel]
                throughput=chosen["candidates_per_worker_hour"] if chosen else (
                    64*3600/max(1.,state["pilot"]["B8"]["seconds"]))
                projection=n/throughput + .1
                left=S.CAP_OCCUPANCY_GPU_H-charged()
                wall_left=(S.HARD_STOP-time.time())/3600
                if projection>left-.5 or projection>max(0,wall_left)*len(S.GPU_IDS):
                    state["skipped_panels"].append(dict(panel=panel,
                         projection_h=projection,remaining_h=left,
                         reason="A21_7_PANEL_SKIP"))
                    print("A21_PANEL_SKIPPED",panel,"projection_gpu_h",
                          round(projection,3),"remaining",round(left,3),
                          flush=True)
                    save();continue
                state["status"]="A21_XPC3_SCORING_"+panel
                folder=run/panel
                nimages=100
                target_workers=4   # user proposed 16 on four GPUs, RAM permitting
                last_report=time.monotonic();last_count=0
                while True:
                    finish()
                    completed=len(list(folder.glob("[0-9][0-9][0-9].json")))
                    gate(completed<=100,"OVERCOMPLETE")
                    if completed==100:
                        # Drain all workers; they see the complete image grid.
                        for pid in list(active):signal_yield(pid,"PANEL_COMPLETE")
                        while active:finish();time.sleep(2)
                        break
                    if not can_run(extra=.75):
                        for pid in list(active):signal_yield(pid,"BUDGET")
                        while active:finish();time.sleep(2)
                        state["status"]="A21_XPC3_BUDGET_OR_DEADLINE_STOP"
                        break
                    views=cards()
                    for g,c in views.items():
                        if g not in S.GPU_IDS:continue
                        on=[pid for pid,v in active.items() if v["gpu"]==g]
                        if c.free_mib<S.YIELD_BELOW_FREE_MIB and on:
                            signal_yield(on[-1],"GPU_FREE_BELOW_4G")
                        if not ready(c,peak):continue
                        # Do not treat GPU workers still loading their model
                        # as zero-memory consumers: nvidia-smi may not show
                        # them for several seconds after Popen. Reserve their
                        # prospective peak+headroom until CUDA PID appears.
                        pending=sum(
                            math.ceil(S.MEMORY_HEADROOM_FACTOR*peak)+
                                S.MEMORY_ADMISSION_EXTRA_MIB
                            for pid in on if pid not in c.process_pids)
                        need=math.ceil(S.MEMORY_HEADROOM_FACTOR*peak)+(
                            S.MEMORY_ADMISSION_EXTRA_MIB)
                        if c.free_mib-pending<need:continue
                        if len(on)>=min(target_workers,S.MAX_WORKERS_PER_GPU):
                            continue
                        if completed+len(active)>=100:continue
                        spawn(g,panel,mode)
                    if time.monotonic()-last_report>=30:
                        now=time.monotonic()
                        done=completed-last_count
                        rate=done*3600/max(1.,now-last_report)
                        state["configurations"].append(dict(stage=panel,
                            mode=mode,requested_workers_per_gpu=target_workers,
                            completed_images_in_window=done,
                            image_throughput_per_wall_hour=rate,
                            foreign_processes_by_gpu={
                                str(g):len(c.foreign_pids)
                                for g,c in views.items() if g in S.GPU_IDS},
                            gpu_used_mib={str(g):c.used_mib for g,c in views.items()
                                         if g in S.GPU_IDS},
                            gpu_utilization={str(g):c.utilization_pct
                                         for g,c in views.items() if g in S.GPU_IDS},
                            timestamp=time.time()))
                        print("A21_PANEL_PROGRESS",panel,completed,"/100",
                              "active_workers",len(active),
                              "charged_gpu_h",round(charged(),4),flush=True)
                        last_report=now;last_count=completed
                        save()
                    time.sleep(5)
                if state["status"]=="A21_XPC3_BUDGET_OR_DEADLINE_STOP":
                    break
                # Verify all chunks (not just filenames), then start CPU
                # offline scoring + publication in a separate worktree.
                proofs=[]
                for i in range(nimages):
                    f=folder/f"{i:03d}.json"
                    v=read(f)
                    gate(v["status"]=="A20_XPC3_IMAGE_PRIOR_PASS"
                         and v["panel"]==panel
                         and v["source_manifest_sha256"]==S.SOURCE_SHA
                         and len(v["rows"])==n//100,
                         "COMPLETE_PANEL_SHA_"+panel)
                    proofs.append(digest(f))
                state["completed_panels"].append(panel)
                state.setdefault("complete_chunk_shas",{})[panel]=proofs
                save()
                publog=run/"logs"/f"{panel}_OFFLINE_PUBLISH.log"
                with publog.open("x") as out:
                    pub=subprocess.Popen([
                        sys.executable,
                        str(WT/"scripts/track_r/xp_a29_a21_publish_cpu.py"),
                        "--expected-head",a.expected_head,"--run",str(run),
                        "--panel",panel],
                        cwd=str(WT),
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),
                        stdout=out,stderr=subprocess.STDOUT,
                        start_new_session=True)
                post.append((panel,pub,publog))
                print("A21_COMPLETE_PANEL_CPU_PUBLICATION_STARTED",panel,
                      "pid",pub.pid,flush=True)
            else:
                state["status"]="A21_XPC3_PRODUCTION_PANELS_DONE"

            # A21.10 independent 32-candidate SHA-smallest UID replication,
            # on DIFFERENT physical GPU(s); informational, not selection.
            # Each child uses the same frozen source and scoring path.
            import hashlib as _hash
            for panel in list(state["completed_panels"]):
                all_candidates=[]
                for chunk in sorted((run/panel).glob("[0-9][0-9][0-9].json")):
                    doc=read(chunk)
                    for row in doc["rows"]:
                        uid=f'XPC3|{panel}|{row["image_id"]}|{row["run"]}'
                        all_candidates.append((_hash.sha256(uid.encode()).hexdigest(),
                            uid,int(doc["physical_gpu"]),row))
                gate(len(all_candidates)==S.SIZES[panel],"REPL_SOURCE_GRID")
                selection=sorted(all_candidates)[:32]
                bygpu=defaultdict(list)
                for _,uid,original,row in selection:
                    bygpu[(original+1)%4].append((uid,original,row))
                replication={}
                for target,selected in sorted(bygpu.items()):
                    if not can_run(extra=.2):
                        state.setdefault("replication_pending",[]).append(panel)
                        break
                    task=run/"replicate"/f"{panel}_gpu{target}_task.json"
                    output=run/"replicate"/f"{panel}_gpu{target}_result.json"
                    write(task,dict(panel=panel,
                         destination_physical_gpu=target,
                         source_physical_gpu_by_uid={u:p for u,p,_ in selected},
                         uids=[u for u,_,_ in selected]))
                    wait=time.monotonic()
                    while not ready(cards()[target],peak):
                        if (time.monotonic()-wait>=300 or
                            not can_run(extra=.2)):
                            state.setdefault("replication_pending",[]).append(
                                dict(panel=panel,target_gpu=target,
                                     reason="NO_MEMORY_OR_BUDGET"))
                            break
                        time.sleep(10)
                    if not ready(cards()[target],peak):
                        break
                    pid=spawn(target,panel,mode,replicate_task=task,
                              replicate_output=output)
                    while pid in active:
                        finish();time.sleep(2)
                    r=read(output)
                    gate(r["status"]=="A21_DIFFERENT_GPU_REPLICATION_PASS"
                         and r["physical_gpu"]==target
                         and len(r["rows"])==len(selected),
                         "REPLICATION_OUTPUT")
                    replication.update({(x["image_id"],int(x["run"])):x
                        for x in r["rows"]})
                if len(replication)==32:
                    dmax=0.;agree=0
                    for _,uid,original,row in selection:
                        key=(row["image_id"],int(row["run"]))
                        v=replication[key]
                        dmax=max(dmax,abs(float(row["prior_margin"])-
                                         float(v["prior_margin"])))
                        agree+=row["prior_choice"]==v["prior_choice"]
                    state.setdefault("cross_gpu_replication",{})[panel]=dict(
                        status="A21_INFORMATIONAL_32_SHA_SMALLEST_PASS",
                        n=32,max_abs_margin_diff=dmax,
                        agreeing_choices=agree,
                        all_choices_agree=agree==32,
                        SHA256_source_uid_list=[
                            _hash.sha256(uid.encode()).hexdigest()
                            for _,uid,_,_ in selection],
                        different_gpu_for_each=True,
                        target_gpus=sorted(bygpu))
                    print("A21_32_UID_DIFFERENT_GPU_REPLICATION",panel,
                        "max_margin_diff",dmax,
                        "choice_agreement",agree,"/32",flush=True)
                    save()
            # Publication of complete panels is independent of later work.
            for panel,p,log in post:
                rc=p.wait()
                state["postprocess"].append(dict(panel=panel,rc=rc,
                    log=str(log)))
                gate(rc==0,"POSTPROCESS_"+panel)
        except BaseException as exc:
            state["status"]="STOP_A21_XPC3_REQUIRES_RECONCILIATION"
            state["error"]=str(exc)[:1200]
            raise
        finally:
            try:shutdown()
            finally:
                save()
                state["status_at_final"]=state["status"]
                state["no_matched_compute_claim"]=True
                state["no_2forward_secondary"]=True
                state["gpu_occupancy_hours"]=state["charged_gpu_h"]
                state["worker_hours_info_only"]=state["worker_hours"]
                write(run/"XP_C3_A21_FINAL_STATUS.json",state)
                # Final audit is returned on an independent branch. A
                # publication error never causes fresh GPU scoring.
                try:
                    log=run/"logs"/"A21_FINAL_PUBLICATION.log"
                    with log.open("x") as output:
                        subprocess.run([sys.executable,
                            str(WT/"scripts/track_r/xp_a29_a21_final_publish.py"),
                            "--expected-head",a.expected_head,
                            "--run",str(run)],
                            cwd=str(WT),
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES="",
                                PYTHONPATH=str(WT),PYTHONDONTWRITEBYTECODE="1"),
                            stdout=output,stderr=subprocess.STDOUT,
                            check=True,timeout=180)
                    print("A21_FINAL_PUBLICATION_PASS",log,flush=True)
                except Exception as pub_exc:
                    print("ACTION NEEDED: A21_FINAL_PUBLICATION",
                          str(pub_exc),flush=True)
                print("A21_XPC3_TERMINAL",state["status"],
                      "complete",state["completed_panels"],
                      "skipped",state["skipped_panels"],
                      "GPU_OCCUPANCY_H",round(state["charged_gpu_h"],4),
                      "WORKER_H",round(state["worker_hours"],4),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for k in ("expected-head","run"):ap.add_argument("--"+k,required=True)
    ap.add_argument("--preflight-only",action="store_true")
    try:main(ap.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A21_CONTROLLER",e,flush=True)
        sys.exit(8)
