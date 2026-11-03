#!/usr/bin/env python3
"""A20 XP-C3 budgeted single-worker autopilot: XP-N -> G5 -> XP-D4.

CPU controller. No scoring without source+code freeze COMMITTED and pushed.
First XP-N image is bounded smoke, project one worker's cost per panel,
complete panels only. All failed/time-out worker wall time billed.
Wait for the pre-existing XP-I2 controller using shared campaign lock;
do not touch its files, PID or checked-out HEAD. Dirty GPU permitted.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
LOCK=BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock"
SRC="docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
FREEZE="docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"
PANELS=("XP_N","G5","XP_D4")
SIZES={"XP_N":6400,"G5":800,"XP_D4":6400}
CAP_H=8.
PROJECT_H=400.
RESERVE_H=.025
ALLOWED_PHYSICAL_GPUS=(1,2,3)
ORIGINAL_HEAD="25152a3d6ffcc4611f45d12bf24d891331976d91"
RECOVERY_REL="docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R1.json"
PREVIOUS_FAILED_RUN=BASE/"xp_a25_xpc3_prod_20261103T002834Z"
REQUEST_MIB=24576
MARGIN_MIB=4096
STOP=False
def on_stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,on_stop)
signal.signal(signal.SIGINT,on_stop)

def gate(v,s):
    if not v:raise RuntimeError("STOP_A20_XPC3_CONTROL_"+s)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def rd(p):return json.loads(Path(p).read_text())

def dump(p,v):
    p=Path(p)
    tmp=p.with_name("." + p.name + ".tmp."+str(os.getpid()))
    with tmp.open("x") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    os.replace(tmp,p)

def git(*args):
    return subprocess.check_output(
        ["git","-C",str(WT),*args],text=True,timeout=40).strip()

def preflight(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and git("rev-parse","HEAD")==a.expected_head
         and not git("status","--porcelain"),
         "CLEAN_PINNED_HEAD")
    gate(git("ls-files","--error-unmatch",SRC)==SRC and
         git("ls-files","--error-unmatch",FREEZE)==FREEZE,
         "MANIFEST_AND_CODE_NOT_COMMITTED")
    source=WT/SRC;fr=rd(WT/FREEZE)
    gate(fr["status"]=="A20_XPC3_OPTION_B_S_MID_GLOBAL2_ONLY"
         and sha(source)==fr["source_manifest_sha256"]
         and fr["n_candidates"]==13600,
         "PROSPECTIVE_FREEZE_HASH")
    total=rd(source)
    gate(total["n_candidates"]==13600
         and total["panel_order"]==list(PANELS)
         and len(total["rows"])==13600,"SCOPE")
    xs=list(BASE.glob("xp_a24_xpi2_prod_*/XP_A24_XPI2_FINAL_STATUS.json"))
    if xs:
        gate(len(xs)==1,"UNIQUE_I2_TERMINAL")
        i2=rd(xs[0])
        gate(i2["status"]=="A24_XPI2_COMPLETE"
             and i2["n_completed_bags"]==100
             and i2["active"]==[],
             "XPI2_100_BAGS_OR_MANUAL_A13_4")
        baseline=float(i2["project_total_gpu_h"])
    else:
        # Under shared LOCK, missing I2 final means I2 may have failed
        # or an unaccounted worker still exists. Do NOT reuse A22 ledger.
        gate(not list(BASE.glob("xp_a24_xpi2_prod_*")),
             "XPI2_EXISTS_WITHOUT_FINAL")
        unique=list(BASE.glob(
            "xp_a22_xpn_continue_*/XP_N_CONTINUATION_FINAL_STATUS.json"))
        gate(len(unique)==1,"XPN_LEDGER_UNIQUE")
        baseline=float(rd(unique[0])["project_total_gpu_h"])
    gate(0<baseline and baseline+CAP_H<=PROJECT_H,
         "PROJECT_400_CAP")
    regpath=WT/RECOVERY_REL
    gate(git("ls-files","--error-unmatch",RECOVERY_REL)==RECOVERY_REL
         and regpath.is_file(),"R1_RECOVERY_REGISTRATION_COMMITTED")
    reg=rd(regpath)
    oldpath=PREVIOUS_FAILED_RUN/"XP_C3_A20_FINAL_STATUS.json"
    gate(oldpath.is_file() and sha(oldpath)==reg["failed_final_sha256"],
         "PRIOR_ATTEMPT_LEDGER_SHA")
    old=rd(oldpath)
    attempts=old["attempts"]
    prior=float(old["charged_gpu_h"])
    gate(reg["status"]=="A20_XPC3_ZERO_OUTPUT_R1_CODE_RECOVERY"
         and reg["source_manifest_sha256"]==fr["source_manifest_sha256"]
         and reg["original_freeze_sha256"]==sha(WT/FREEZE)
         and reg["original_execution_head"]==ORIGINAL_HEAD
         and reg["allowed_physical_gpus"]==list(ALLOWED_PHYSICAL_GPUS)
         and reg["scientific_definition_unchanged"] is True
         and old["status"]=="STOP_A20_XPC3_UNVERIFIED_OR_BUDGET"
         and old["completed_panels"]==[]
         and old["source_manifest_sha256"]==reg["source_manifest_sha256"]
         and old["scorer_freeze_sha256"]==sha(WT/FREEZE)
         and len(attempts)==1
         and attempts[0]["gpu"]==0
         and attempts[0]["panel"]=="XP_N"
         and attempts[0]["limit"]==1
         and attempts[0]["rc"]==8
         and abs(prior-attempts[0]["wall_seconds"]/3600.)<1e-7
         and abs(prior-reg["failed_charged_gpu_h"])<1e-10
         and abs(old["project_gpu_h"]-baseline-prior)<1e-6
         and abs(old["project_gpu_h"]-reg["failed_project_gpu_h"])<1e-8
         and not list((PREVIOUS_FAILED_RUN/"XP_N").glob("*.json"))
         and not list((PREVIOUS_FAILED_RUN/"G5").glob("*.json"))
         and not list((PREVIOUS_FAILED_RUN/"XP_D4").glob("*.json"))
         and 0<prior<.01,"R1_ONLY_ZERO_OUTPUT_FOUR_SECOND_FAILURE")
    gate(prior+CAP_H-prior<=CAP_H and baseline+CAP_H<=PROJECT_H,
         "RECOVERY_G7_AND_PROJECT_BUDGET")
    return baseline,prior,reg

def is_track_r(pid):
    try:
        x=subprocess.run(["ps","-p",str(pid),"-o","args="],
                         capture_output=True,text=True,timeout=2)
        return x.returncode==0 and "scripts/track_r/" in x.stdout
    except (OSError,subprocess.SubprocessError):
        return True

def gpu():
    cards=collect()
    avail=[c for c in cards if c.index in ALLOWED_PHYSICAL_GPUS
           and c.visibility=="known" and
           not any(is_track_r(p) for p in c.foreign_pids)]
    return choose_gpu(avail,REQUEST_MIB,safety_margin_mib=MARGIN_MIB,
                      fraction=.80,clean_only=False)[0]

def spawn(panel,limit,run,head,index,remaining_sec):
    gate(remaining_sec>120,"BUDGET_EXHAUSTED")
    g=None
    while g is None:
        gate(not STOP and not (run/"STOP_XPC3").exists(),
             "OPERATOR_STOP")
        g=gpu()
        if g is None:
            time.sleep(15)
    gate(g in ALLOWED_PHYSICAL_GPUS,"NO_GPU_0")
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(g),PYTHONPATH=str(WT),
       PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
       PYTHONPYCACHEPREFIX=str(run/"pycache"/f"{panel}_{index}"),
       CUBLAS_WORKSPACE_CONFIG=":4096:8",
       TMPDIR=str(ROOT/"tmp"),
       OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
       OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2")
    args=[str(ROOT/"conda-envs/daps/bin/python"),
          str(WT/"scripts/track_r/xp_a25_xpc3_mid_gpu.py"),
          "--expected-head",head,"--run",str(run),
          "--panel",panel,"--gpu",str(g)]
    if limit is not None:args+=["--limit-images",str(limit)]
    log=run/"logs"/f"{panel}_{index}.log"
    with log.open("x") as output:
        t0=time.monotonic()
        proc=subprocess.Popen(args,cwd=str(WT),env=env,
            stdout=output,stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            while proc.poll() is None:
                if (time.monotonic()-t0 >= remaining_sec or
                    STOP or (run/"STOP_XPC3").exists()):
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    proc.wait(timeout=30)
                    break
                time.sleep(4)
        finally:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            proc.wait()
    sec=time.monotonic()-t0
    return dict(gpu=g,rc=proc.returncode,wall_seconds=sec,
                log=str(log),panel=panel,limit=limit)

def counts(run,panel):
    pp=run/panel
    result=[]
    for f in sorted(pp.glob("*.json")):
        d=rd(f)
        gate(d["status"]=="A20_XPC3_IMAGE_PRIOR_PASS"
             and d["panel"]==panel
             and d["source_manifest_sha256"]==sha(WT/SRC)
             and all(r["denoiser_examples"]==32 for r in d["rows"]),
             "IMAGE_SHA_"+panel)
        result.append((d["image_id"],len(d["rows"]),sha(f)))
    gate(len({v[0] for v in result})==len(result),
         "IMAGE_DUPLICATE")
    return result

def main(a):
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a25_xpc3_prod_")
         and not run.exists() and not run.is_symlink(),
         "FRESH_RUN_ROOT")
    # Shared lock prevents optimistic cross-project accounting with XP-I2.
    with LOCK.open("a+") as f:
        print("A20_XPC3_WAIT_SHARED_PROJECT_LEDGER_LOCK",flush=True)
        fcntl.flock(f,fcntl.LOCK_EX)
        base,previous_charge,registration=preflight(a)
        if a.preflight_only:
            print("A20_XPC3_CONTROLLER_PREFLIGHT_PASS",
               "project_gpu_h",base,"capacity_gpu_h",CAP_H,
               "previous_failed_gpu_h",previous_charge,
               "allowed_gpus",ALLOWED_PHYSICAL_GPUS,
               "panels",SIZES,flush=True)
            return
        run.mkdir(mode=0o700)
        for name in ("logs","pycache",*PANELS):
            (run/name).mkdir()
        state=dict(status="A20_XPC3_RUNNING",start_project_gpu_h=base,
             source_manifest_sha256=sha(WT/SRC),
             scorer_freeze_sha256=sha(WT/FREEZE),
             cap_gpu_h=CAP_H,charged_gpu_h=previous_charge,
             failed_prior_gpu_h=previous_charge,
             recovery_registry_sha256=sha(WT/RECOVERY_REL),
             failed_original_ledger_sha256=registration["failed_final_sha256"],
             admitted_physical_gpus=list(ALLOWED_PHYSICAL_GPUS),
             completed_panels=[],attempts=[],
             no_GT_scoring=True,no_secondary_prior=True)
        def save():
            state["project_gpu_h"]=base+state["charged_gpu_h"]
            dump(run/"XP_C3_A20_CONTROLLER_STATE.json",state)
        def attempt(panel,limit,index,allowed_seconds):
            outcome=spawn(panel,limit,run,a.expected_head,index,
                          allowed_seconds)
            state["attempts"].append(outcome)
            state["charged_gpu_h"]+=outcome["wall_seconds"]/3600.
            save()
            gate(outcome["rc"]==0,"GPU_WORKER_RC_"+panel)
            return outcome
        try:
            save()
            # Bound the smoke. Exactly 1 image = 64 runs = 2048 denoiser
            # examples. It is billed, and its finished image is reusable.
            first=attempt("XP_N",1,1,
                min(1800.,(CAP_H-state["charged_gpu_h"])*3600-120))
            records=counts(run,"XP_N")
            gate(len(records)==1 and records[0][1]==64,
                 "SMOKE_EXACT_64")
            per_candidate=first["wall_seconds"]/64.
            estimate={k:per_candidate*SIZES[k]/3600
                      for k in PANELS}
            state["smoke_wall_seconds"]=first["wall_seconds"]
            state["projection_gpu_h"]=estimate
            save()
            for panel in PANELS:
                # Never charge incomplete panels as published; skip a panel
                # not conservatively projected to fit in remaining budget.
                already=counts(run,panel)
                done_n=sum(n for _,n,_ in already)
                rem=SIZES[panel]-done_n
                gate(rem>=0,"EXCESS_"+panel)
                projection=per_candidate*rem/3600 + .05
                if state["charged_gpu_h"]+projection+RESERVE_H>CAP_H or (
                    base+state["charged_gpu_h"]+projection+RESERVE_H>PROJECT_H):
                    state["status"]="A20_XPC3_COMPLETE_PANELS_BUDGET_STOP"
                    state["skipped_from_panel"]=panel
                    save();break
                if rem:
                    attempt(panel,None,2,
                       (CAP_H-state["charged_gpu_h"])*3600-60)
                accepted=counts(run,panel)
                gate(len(accepted)==100
                     and sum(x[1] for x in accepted)==SIZES[panel],
                     "COMPLETE_PANEL_"+panel)
                # GT-free scorer is already final for this entire panel.
                # Only NOW a separate CPU process may open GT for
                # prior_chosen_good, with the choice and margin immutable.
                cpu_env=os.environ.copy()
                cpu_env.update(CUDA_VISIBLE_DEVICES="",
                    PYTHONPATH=str(WT),PYTHONNOUSERSITE="1",
                    PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
                offline_log=run/"logs"/f"{panel}_offline_eval.log"
                with offline_log.open("x") as handle:
                    subprocess.run([sys.executable,
                       str(WT/"scripts/track_r/xp_a25_xpc3_offline_final_cpu.py"),
                       "--expected-head",a.expected_head,
                       "--panel",panel,"--run",str(run)],
                       cwd=str(WT),env=cpu_env,
                       stdout=handle,stderr=subprocess.STDOUT,
                       check=True)
                outcsv=run/"final"/f"XP_C3_A20_{panel}_{SIZES[panel]}_PRIOR_DECISIONS.csv"
                gate(outcsv.is_file(),"OFFLINE_MISSING_"+panel)
                state.setdefault("final_csv_sha256",{})[panel]=sha(outcsv)
                state["completed_panels"].append(panel)
                save()
            else:state["status"]="A20_XPC3_ALL_THREE_PANELS_COMPLETE"
        except BaseException as e:
            state["status"]="STOP_A20_XPC3_UNVERIFIED_OR_BUDGET"
            state["error"]=str(e)[:700]
            raise
        finally:
            save()
            dump(run/"XP_C3_A20_FINAL_STATUS.json",state)
            print("A20_XPC3_TERMINAL",state["status"],
                  "completed",state["completed_panels"],
                  "charged_gpu_h",round(state["charged_gpu_h"],5),
                  "project_gpu_h",round(state["project_gpu_h"],5),
                  flush=True)
            # Only after ALL scoring processes have terminated, copy
            # complete panels into theory_inputs and returns and push.
            if (state["completed_panels"]
                and state["status"] in (
                "A20_XPC3_ALL_THREE_PANELS_COMPLETE",
                "A20_XPC3_COMPLETE_PANELS_BUDGET_STOP")):
                pubenv=os.environ.copy()
                pubenv.update(CUDA_VISIBLE_DEVICES="",
                    PYTHONPATH=str(WT),PYTHONNOUSERSITE="1",
                    PYTHONDONTWRITEBYTECODE="1")
                publog=run/"logs"/"A20_XPC3_PUBLICATION.log"
                with publog.open("x") as handle:
                    subprocess.run([sys.executable,
                       str(WT/"scripts/track_r/xp_a25_xpc3_publish_cpu.py"),
                       "--expected-head",a.expected_head,
                       "--run",str(run),"--branch",a.branch],
                       cwd=str(WT),env=pubenv,
                       stdout=handle,stderr=subprocess.STDOUT,
                       check=True)
                print("A20_XPC3_PUBLICATION_LOG",publog,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--run",required=True)
    ap.add_argument("--branch",required=True)
    ap.add_argument("--preflight-only",action="store_true")
    try:main(ap.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A20_XPC3_AUTOPILOT",e,flush=True)
        sys.exit(8)
