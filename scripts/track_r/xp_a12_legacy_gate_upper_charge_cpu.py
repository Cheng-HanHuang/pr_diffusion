#!/usr/bin/env python3
"""A12 conservative upper-billed reconstruction of stale failed B16 gate.

The 2026-10-28 10:48:36 gate left one GPU worker active in RUNNING
state. The worker subsequently wrote a PASS log but its precise OS exit
time was not logged. We CANNOT claim an exact interval.

Source-backed safe accounting: bill worker0's recorded 100.835 seconds,
PLUS the ENTIRE possible wall-clock interval for worker1 from the
immutable LAUNCH event until a new contemporaneous PID-dead audit.
This may substantially OVERCHARGE historical GPU use, by design.
No original event/state/log is modified. The upper bound is suitable
for conservative 400-GPUh safety admission, NOT accurate paper timing.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,math,os,sys
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate

RUN=BASE/"xp_a10_b16_gate_20261028T104836Z"
OUT=BASE/"XP_A12_LEGACY_B16_GATE_104836_CONSERVATIVE_CHARGE.json"
PIN_STATE="2d8cf00361688852aae2216115adc9491203970dffbac8a2565ba2facc975527"
PIN_EVENTS="1da54503b4b94f65103e98e943cd02ee796015033368fdd2227ae1d2567a7ad0"
PID=1057433
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for p in iter(lambda:f.read(1<<20),b""):h.update(p)
    return h.hexdigest()
def main():
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(not OUT.exists() and not OUT.is_symlink(),"OUTPUT_ALREADY_EXISTS")
    statep=RUN/"B16_GATE_STATE.json"
    eventp=RUN/"B16_GATE_EVENTS.jsonl"
    workerp=RUN/"logs/B16_001_60140_attempt_001.log"
    lockp=RUN/"B16_GATE_CONTROLLER.lock"
    gate(sha(statep)==PIN_STATE and sha(eventp)==PIN_EVENTS,
         "HISTORICAL_SOURCE_SHA")
    state=json.loads(statep.read_text())
    events=[json.loads(x) for x in eventp.read_text().splitlines() if x.strip()]
    gate(state["status"]=="RUNNING" and
         state["active"]==[{"gpu":1,"index":1,"pid":PID}] and
         state["done"]==[] and state["attempts"]=={"0":1,"1":1} and
         math.isclose(state["spent_gpu_seconds"],100.83540434669703,abs_tol=1e-8) and
         len(events)==2 and
         [(e["action"],e["index"],e["pid"]) for e in events]==[
           ("LAUNCH",0,1057257),("LAUNCH",1,PID)],
         "HISTORICAL_ACTIVE_ONE_WORKER")
    gate(workerp.is_file() and not workerp.is_symlink() and
         "XP_SITCOM_WORKER_PASS b22 4" in workerp.read_text() and
         "B16_GATE_NATIVE_IMAGE_PASS 60140" in workerp.read_text(),
         "WORKER_SUCCESS_LOG")
    launch=datetime.datetime.fromisoformat(events[1]["utc"])
    gate(launch.tzinfo is not None,"LAUNCH_TZ")
    mtime=datetime.datetime.fromtimestamp(workerp.stat().st_mtime,
                                            datetime.timezone.utc)
    gate(mtime>launch and mtime>=datetime.datetime.fromtimestamp(
         statep.stat().st_mtime,datetime.timezone.utc),"EVENT_LOG_CHRONOLOGY")
    gate(lockp.is_file(),"CONTROLLER_LOCK_MISSING")
    # Linux requires an fd open for writing to acquire LOCK_EX.
    # a+ opens the EXISTING lockfile read/write without truncating or
    # rewriting its historical contents. Never lock via read-only fd.
    with lockp.open("a+") as h:
        try:
            fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            gate(False,"CONTROLLER_STILL_OWNS_LOCK")
        else:
            fcntl.flock(h,fcntl.LOCK_UN)
    # PID must be absent at the actual audit; if it exists (including
    # reuse), refuse to make any upper-bound claim.
    try:os.kill(PID,0)
    except ProcessLookupError:pass
    else:gate(False,"WORKER_PID_PRESENT_OR_REUSED")
    audited=datetime.datetime.now(datetime.timezone.utc)
    gate((audited-launch).total_seconds()>0 and
         (audited-launch).total_seconds()<7*24*3600,
         "AUDIT_WINDOW")
    prior=float(state["spent_gpu_seconds"])
    lower=prior+(mtime-launch).total_seconds()
    upper=prior+(audited-launch).total_seconds()
    gate(0<lower<=upper<400*3600,
         "CONSERVATIVE_INTERVAL_INVALID")
    # The unknown OS exit is SOMEWHERE in [log_mtime, audited_utc].
    # Crucially the charged value is UPPER, NOT claimed exact.
    doc=dict(status="XP_A12_LEGACY_GATE_CONSERVATIVE_UPPER_BOUND_PASS",
       source_stage="A10_FAILED_B16_GATE_2",
       charge_basis="CONSERVATIVE_OCCUPIED_GPU_WALL_UPPER_BOUND",
       exact_gpu_time_known=False,
       billing_is_upper_bound=True,
       recorded_prior_finished_worker_gpu_seconds=prior,
       uncertain_worker_pid=PID,
       uncertain_worker_gpu=1,
       uncertain_worker_launch_utc=launch.isoformat(),
       uncertain_worker_last_pass_log_utc=mtime.isoformat(),
       uncertain_worker_confirmed_absent_utc=audited.isoformat(),
       actual_gpu_seconds_lower_bound=lower,
       charged_gpu_seconds_upper_bound=upper,
       actual_gpu_h_lower_bound=lower/3600,
       charged_gpu_h_upper_bound=upper/3600,
       possible_overbilling_gpu_h=(upper-lower)/3600,
       original_state_sha256=PIN_STATE,
       original_events_sha256=PIN_EVENTS,
       worker_log_sha256=sha(workerp),
       original_source_paths=[str(statep),str(eventp),str(workerp)],
       original_state_and_logs_unchanged=True,
       project_cap_safety_eligible=True,
       accuracy_not_suitable_for_paper_timing=True)
    with OUT.open("x") as f:
        json.dump(doc,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    print("A12_LEGACY_GATE_CONSERVATIVE_UPPER_BOUND_AUDIT_PASS",flush=True)
    print("LOWER_GPU_H",round(lower/3600,9),flush=True)
    print("CHARGED_UPPER_GPU_H",round(upper/3600,9),flush=True)
    print("OVERCHARGE_BAND_GPU_H",round((upper-lower)/3600,9),flush=True)
    print("SOURCE_CORRECTION_SHA",sha(OUT),flush=True)
if __name__=="__main__":
    try:main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("ACTION NEEDED: A12_LEGACY_GATE_COST_BOUND",exc,flush=True)
        sys.exit(8)
