#!/usr/bin/env python3
"""One-GPU supervised native SITCOM replay with separate offline CPU scoring.

Frozen four B22 roots run in one model/RNG stream. Four B24 roots run in
fresh per-seed Python processes. No GT data flows to workers or env.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu,classify_timing,event
from scripts.track_r.xp_sitcom_replay_gpu import (
    ROOT,WT,BASE,SITCOM_PY,PREP_HEAD,RUN_NAME,MANIFEST_SHA,
    folder,check_manifest,sha,safe)
OUT=BASE/RUN_NAME
CACHE=ROOT/"outputs/.cache/track_r"
CPU=ROOT/"conda-envs/prdiff_track_r/bin/python"

def dump_x(p,data):
    with Path(p).open("x",encoding="utf-8") as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")

def status_update(p,data):
    temp=p.with_name("."+p.name+"."+str(os.getpid()))
    dump_x(temp,data)
    os.replace(temp,p)

def log_event(p,obj):
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+"\n")

def main(a):
    if not (5<=a.poll_seconds<=60 and
            16000<=a.estimated_vram_mib<=60000 and
            0.5<=a.gpu_hour_cap<=3):
        raise RuntimeError("STOP_SITCOM_SUPERVISOR_ARGUMENTS")
    git=lambda *parts: subprocess.check_output(
        ["git","-C",str(WT),*parts],text=True,timeout=30).strip()
    if git("rev-parse","HEAD")!=a.expected_head or git("status","--porcelain"):
        raise RuntimeError("STOP_SITCOM_SUPERVISOR_GIT")
    if not OUT.is_dir() or OUT.is_symlink():
        raise RuntimeError("STOP_SITCOM_SUPERVISOR_RUN_DIR")
    for forbidden in ("XP_SITCOM_REPLAY_STATUS.json","XP_SITCOM_REPLAY_GPU_SUMMARY.json",
                      "GPU_EVENTS.jsonl","XP_SITCOM_REPLAY_OFFLINE_SCORE.json"):
        if (OUT/forbidden).exists():
            raise RuntimeError("STOP_SITCOM_SUPERVISOR_ALREADY_RUN "+forbidden)
    mf=OUT/"XP_SITCOM_REPLAY_TASKS.json"
    if sha(mf)!=MANIFEST_SHA:
        raise RuntimeError("STOP_SITCOM_SUPERVISOR_MANIFEST_SHA")
    manifest=json.loads(mf.read_text())
    rows=check_manifest(manifest,a.expected_head)
    prep=json.loads(safe(OUT/"XP_SITCOM_REPLAY_PREP.json").read_text())
    if (prep.get("status")!="XP_SITCOM_REPLAY_PREP_PASS" or
        prep.get("manifest_sha256")!=MANIFEST_SHA or
        prep.get("head")!=PREP_HEAD):
        raise RuntimeError("STOP_SITCOM_SUPERVISOR_PREP")
    (OUT/"gpu").mkdir(exist_ok=False)
    (OUT/"pycache").mkdir(exist_ok=False)
    dump_x(OUT/"XP_SITCOM_REPLAY_SUPERVISOR_PID.json",{
        "status":"STARTED","pid":os.getpid(),"head":a.expected_head,
        "manifest_sha256":MANIFEST_SHA})
    status=OUT/"XP_SITCOM_REPLAY_STATUS.json"
    events=OUT/"GPU_EVENTS.jsonl"
    state={"state":"WAIT_VRAM","stage":"XP_SITCOM_REPLAY_8",
           "head":a.expected_head,"prep_head":PREP_HEAD,
           "manifest_sha256":MANIFEST_SHA,"gpu":None,
           "completed_roots":0,"gpu_hours":0.0,"reason":None,
           "worker_pid":None}
    status_update(status,state)
    selected=None
    wait_started=time.monotonic()
    while selected is None:
        if (OUT/"STOP_AFTER_ACTIVE").is_file():
            state.update(state="STOP_REQUESTED",reason="STOP_SENTINEL")
            status_update(status,state)
            print("ACTION NEEDED: SITCOM_STOP_REQUESTED",flush=True)
            return 8
        if time.monotonic()-wait_started>3600:
            state.update(state="STOP_WAIT_TIMEOUT",reason="NO_CLEAN_GPU_WITHIN_1H")
            status_update(status,state)
            print("ACTION NEEDED: SITCOM_NO_CLEAN_GPU",flush=True)
            return 8
        try:
            cards=collect()
            selected,_=choose_gpu(cards,a.estimated_vram_mib,
                    clean_only=True,safety_margin_mib=2048,fraction=0.80)
            log_event(events,event(cards,stage="XP_SITCOM_REPLAY",
                                   job_id="sitcom_replay8",action="ADMISSION",
                                   selected_gpu=selected))
        except Exception as exc:
            state["reason"]="TELEMETRY_"+str(exc)[:100]
            status_update(status,state)
            time.sleep(a.poll_seconds)
            continue
        if selected is None:
            time.sleep(a.poll_seconds)
    state.update(state="RUNNING_GPU",gpu=selected,reason=None)
    status_update(status,state)
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(selected),
               PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
               PYTHONPATH=str(WT),
               PYTHONPYCACHEPREFIX=str(OUT/"pycache"),
               CUBLAS_WORKSPACE_CONFIG=":4096:8",
               TMPDIR=str(CACHE/"tmp"),
               XDG_CACHE_HOME=str(CACHE))
    commands=[
        ("b22",-1,OUT/"gpu/b22",rows[:4]),
        *[("b24",i,OUT/"gpu"/f"b24_{i}",[rows[i]]) for i in range(4,8)]
    ]
    samples=[];worker_summaries=[];active_seconds=0.0
    for mode,idx,work,task_rows in commands:
        if (OUT/"STOP_AFTER_ACTIVE").exists():
            state.update(state="STOP_REQUESTED",reason="STOP_BETWEEN_WORKERS")
            status_update(status,state)
            print("ACTION NEEDED: SITCOM_STOP_BETWEEN_WORKERS",flush=True)
            return 8
        args=[
            str(SITCOM_PY),str(WT/"scripts/track_r/xp_sitcom_replay_gpu.py"),
            "--expected-head",a.expected_head,"--mode",mode,
            "--index",str(idx),"--out",str(work),"--gpu",str(selected)
        ]
        preflight=subprocess.run(args+["--preflight-only"],cwd=str(WT),
                   env=env,capture_output=True,text=True,timeout=300)
        (OUT/("PREFLIGHT_"+mode+"_"+str(idx)+".log")).write_text(
            preflight.stdout+preflight.stderr)
        if preflight.returncode:
            state.update(state="FAILED_PREFLIGHT",
                         reason="NATIVE_PREFLIGHT_"+mode+"_"+str(idx))
            status_update(status,state)
            print("ACTION NEEDED: SITCOM_NATIVE_PREFLIGHT",mode,idx,flush=True)
            return 8
        state["worker_mode"]=mode
        state["worker_index"]=idx
        status_update(status,state)
        try:
            before=collect()
            samples.append(before)
            log_event(events,event(before,stage="XP_SITCOM_REPLAY",
                                   job_id="sitcom_replay8",
                                   action="PRE_GPU",selected_gpu=selected))
        except Exception as exc:
            state["telemetry_missing"]=state.get("telemetry_missing",0)+1
            log_event(events,{"action":"PRE_TELEMETRY_ERROR","error":str(exc)[:150]})
        start=time.monotonic()
        logfile=OUT/("WORKER_"+mode+"_"+str(idx)+".log")
        with logfile.open("xb") as outlog:
            worker=subprocess.Popen(args,cwd=str(WT),env=env,stdout=outlog,
                        stderr=subprocess.STDOUT,start_new_session=True)
        state["worker_pid"]=worker.pid
        status_update(status,state)
        missed=0;stopped=False
        while worker.poll() is None:
            time.sleep(a.poll_seconds)
            if (active_seconds+time.monotonic()-start)>a.gpu_hour_cap*3600:
                stopped=True
                state["reason"]="GPU_HOUR_CAP_EXCEEDED"
            if (OUT/"STOP_AFTER_ACTIVE").is_file():
                stopped=True
                state["reason"]="STOP_AFTER_ACTIVE"
            try:
                current=collect([worker.pid])
                samples.append(current)
                log_event(events,event(current,stage="XP_SITCOM_REPLAY",
                         job_id="sitcom_replay8",action="HEARTBEAT",
                         selected_gpu=selected,own_roots=[worker.pid]))
                card=next(x for x in current if x.index==selected)
                # Any total occupied VRAM beyond the frozen 80% device
                # ceiling is unsafe; stop our own child only.
                if card.used_mib>int(.80*card.total_mib):
                    stopped=True
                    state["reason"]="HARD_VRAM_CEILING"
            except Exception as exc:
                missed+=1
                log_event(events,{"action":"TELEMETRY_ERROR","error":str(exc)[:150]})
            if stopped:
                try:os.killpg(worker.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:worker.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    try:os.killpg(worker.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    worker.wait()
                break
        wall=time.monotonic()-start
        active_seconds+=wall
        try:
            last=collect([worker.pid])
            samples.append(last)
            log_event(events,event(last,stage="XP_SITCOM_REPLAY",
                            job_id="sitcom_replay8",action="POST_GPU",
                            selected_gpu=selected))
        except Exception as exc:
            missed+=1
            log_event(events,{"action":"POST_TELEMETRY_ERROR",
                              "error":str(exc)[:150]})
        state["gpu_hours"]=active_seconds/3600
        state["worker_rc"]=worker.returncode
        state["worker_pid"]=None
        state["telemetry_missing"]=state.get("telemetry_missing",0)+missed
        if stopped or worker.returncode:
            state["state"]="FAILED_GPU"
            state["reason"]=state.get("reason") or "WORKER_RC_"+str(worker.returncode)
            status_update(status,state)
            print("ACTION NEEDED: SITCOM_GPU_WORKER",mode,idx,
                  "rc",worker.returncode,"reason",state["reason"],flush=True)
            return 8
        report=json.loads(safe(work/"XP_SITCOM_WORKER_COMPLETE.json").read_text())
        if (report.get("status")!="XP_SITCOM_WORKER_PASS" or
            report.get("new_roots")!=len(task_rows) or
            report.get("head")!=a.expected_head or
            report.get("manifest_sha256")!=MANIFEST_SHA):
            state.update(state="FAILED_ANALYSIS",reason="WORKER_RECEIPT_MISMATCH")
            status_update(status,state)
            return 8
        for row in task_rows:
            result=json.loads(safe(folder(work,row)/"XP_SITCOM_ROOT_COMPLETE.json").read_text())
            if (result.get("uid")!=row["uid"] or
                result.get("gt_access") is not False or
                result.get("historical_access") is not False or
                sha(folder(work,row)/"reconstruction.png")!=
                   result.get("reconstruction_png_sha256")):
                raise RuntimeError("STOP_SITCOM_SUPERVISOR_ROOT_RECEIPT")
        worker_summaries.append({"mode":mode,"index":idx,
            "worker_wall_s":wall,"worker_receipt_sha256":sha(
                work/"XP_SITCOM_WORKER_COMPLETE.json")})
        state["completed_roots"]+=len(task_rows)
        status_update(status,state)
        print("XP_SITCOM_GROUP_PASS",mode,idx,
              "roots",len(task_rows),"wall",round(wall,2),flush=True)
    timing=classify_timing(samples,selected)
    if state.get("telemetry_missing",0):
        timing["timing_class"]="UNVERIFIED"
        timing["timing_valid"]=False
        timing["reasons"].append("telemetry_missing")
    receipts=[]
    for i,row in enumerate(rows):
        work=OUT/"gpu"/("b22" if i<4 else "b24_"+str(i))
        receipt=json.loads(safe(folder(work,row)/"XP_SITCOM_ROOT_COMPLETE.json").read_text())
        receipts.append(receipt)
    values=[float(x["sampling_wall_s"]) for x in receipts]
    total_report={
       "status":"XP_SITCOM_REPLAY_GENERATION_PASS",
       "runner_head":a.expected_head,"prep_head":PREP_HEAD,
       "manifest_sha256":MANIFEST_SHA,"n_roots":8,
       "gpu_index":selected,
       "gpu_worker_seconds":active_seconds,
       "gpu_hours":active_seconds/3600,
       "sample_wall_s_median":statistics.median(values),
       "sample_wall_s_by_source":{
          source:statistics.median(float(x["sampling_wall_s"]) for x in receipts
                          if x["source"]==source)
          for source in ("S1_b22","S2_b24_dev80")},
       "native_worker_groups":worker_summaries,
       "timing":timing,
       "timing_clean_eligible":bool(timing["timing_valid"]),
       "GT_free_generation":True}
    dump_x(OUT/"XP_SITCOM_REPLAY_GPU_SUMMARY.json",total_report)
    state.update(state="CPU_OFFLINE_SCORING",reason=None,timing=timing,
                 median_sampling_wall_s=total_report["sample_wall_s_median"])
    status_update(status,state)
    offline_env=dict(os.environ)
    offline_env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1",PYTHONPATH=str(WT),
                       PYTHONPYCACHEPREFIX=str(OUT/"pycache"))
    scorer=subprocess.run([str(CPU),str(WT/"scripts/track_r/xp_sitcom_replay_score_cpu.py"),
                      "--expected-head",a.expected_head],
                      env=offline_env,cwd=str(WT),
                      capture_output=True,text=True,timeout=1800)
    (OUT/"XP_SITCOM_REPLAY_SCORE.log").write_text(scorer.stdout+scorer.stderr)
    if scorer.returncode:
        state.update(state="REPLAY_GATE_FAILED",reason="OFFLINE_SCORE_RC_"+str(scorer.returncode))
        status_update(status,state)
        print("ACTION NEEDED: SITCOM_REPLAY_GATE_FAILED",flush=True)
        return 8
    score=json.loads(safe(OUT/"XP_SITCOM_REPLAY_OFFLINE_SCORE.json").read_text())
    if score.get("status")!="XP_SITCOM_REPLAY_PASS" or score.get("n_roots")!=8:
        raise RuntimeError("STOP_SITCOM_SUPERVISOR_OFFLINE_RESULT")
    state.update(state="PASS",replay_max_abs_delta_db=score["max_abs_delta_db"],
                 n_exact=score["n_exact_historical"],reason=None)
    status_update(status,state)
    print("XP_SITCOM_REPLAY_PASS n=8 max_db="+str(score["max_abs_delta_db"])+
          " exact="+str(score["n_exact_historical"])+
          " gpu_hours="+str(round(active_seconds/3600,5))+
          " timing="+timing["timing_class"],flush=True)
    print("XP_SITCOM_GPU_SUMMARY_SHA="+sha(OUT/"XP_SITCOM_REPLAY_GPU_SUMMARY.json"))
    print("XP_SITCOM_OFFLINE_SCORE_SHA="+sha(OUT/"XP_SITCOM_REPLAY_OFFLINE_SCORE.json"))
    print("OK",flush=True)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--poll-seconds",type=int,default=10)
    p.add_argument("--estimated-vram-mib",type=int,default=32000)
    p.add_argument("--gpu-hour-cap",type=float,default=2.0)
    try:sys.exit(main(p.parse_args()))
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: XP_SITCOM_SUPERVISOR "+str(exc),flush=True)
        sys.exit(8)
