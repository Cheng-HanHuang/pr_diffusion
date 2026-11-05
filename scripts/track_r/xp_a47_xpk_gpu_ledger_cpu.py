#!/usr/bin/env python3
"""A47 fail-closed XP-K GPU occupancy accounting, pure CPU and resumable.

Charge wall-clock occupancy per physical GPU as a union of worker intervals.
Overlapping workers on the same GPU count once; simultaneous GPUs count
separately. Persist START *before* spawn and END after wait. An interrupted
worker without an END is conservatively charged through audit time; it
cannot be silently discarded on restart. This is NOT a PAC controller.
"""
from __future__ import annotations
import argparse,datetime,json,math,os,sys,time
from pathlib import Path

XPK_CAP_H=30.0
PROJECT_CAP_H=400.0
DEADLINE="2026-11-15T12:00:00-05:00"
FAMILIES=("SITCOM","FAST","DAPS")
PHASES=("historical_replay","pilot","calibrated")
def gate(ok,reason):
    if not ok:raise ValueError("STOP_A47_"+reason)
def _number(x):
    gate(type(x) in (int,float) and math.isfinite(x) and x>=0,"NONFINITE_OR_NEGATIVE_TIME")
    return float(x)
def interval_union_seconds(intervals):
    """Intervals in monotonic *same-host* time domain."""
    spans=[]
    for start,end in intervals:
        s,e=_number(start),_number(end)
        gate(e>=s,"NEGATIVE_INTERVAL")
        spans.append((s,e))
    spans.sort()
    if not spans:return 0.0
    lo,hi=spans[0]
    total=0.
    for s,e in spans[1:]:
        if s>hi:
            total+=hi-lo
            lo,hi=s,e
        else:hi=max(hi,e)
    return total+hi-lo
def charge(events,now,*,allowed_gpus=(0,1,2,3)):
    """Events are START/END, unique worker id and GPU; no PID inference.

    START contains wall_unix; END must carry same id and physical GPU.
    Unix times are used for crash-resumable accounting; clock anomalies stop.
    """
    now=_number(now)
    starts={};ends={};seen=set()
    for ev in events:
        gate(type(ev)==dict,"EVENT_FORMAT")
        eid=ev.get("id");kind=ev.get("kind");gpu=ev.get("gpu")
        gate(type(eid)==str and eid and kind in ("START","END")
             and type(gpu)==int and gpu in allowed_gpus,"EVENT_FIELDS")
        t=_number(ev.get("wall_unix"))
        gate(t<=now+1,"FUTURE_EVENT")
        key=(eid,kind)
        gate(key not in seen,"DUPLICATE_EVENT")
        seen.add(key)
        if kind=="START":
            gate(ev.get("family") in FAMILIES and ev.get("phase") in PHASES,
                 "START_FAMILY_PHASE")
            starts[eid]=ev
        else:ends[eid]=ev
    gate(set(ends)<=set(starts),"ORPHAN_END")
    per_gpu={i:[] for i in allowed_gpus}
    unfinished=[]
    for eid,ev in starts.items():
        gpu=ev["gpu"];s=_number(ev["wall_unix"])
        end=ends.get(eid)
        if end is None:
            e=now;unfinished.append(eid)
        else:
            gate(end["gpu"]==gpu,"GPU_CHANGED")
            e=_number(end["wall_unix"])
        gate(e>=s,"CLOCK_ROLLBACK")
        per_gpu[gpu].append((s,e))
    seconds={str(g):interval_union_seconds(spans) for g,spans in per_gpu.items()}
    return dict(physical_gpu_seconds=seconds,
                physical_gpu_h=sum(seconds.values())/3600.,
                unfinished_workers=sorted(unfinished),
                workers_started=len(starts),workers_ended=len(ends),
                charging_rule="per-physical-GPU union of START-to-END; unfinished to audit now")
def remaining(events,now,prior_project_h,other_xpk_h=0.,*,reserve_h=0.):
    ledger=charge(events,now)
    prior=_number(prior_project_h);other=_number(other_xpk_h)
    reserve=_number(reserve_h)
    xpk=other+ledger["physical_gpu_h"]
    project=prior+ledger["physical_gpu_h"]
    gate(xpk<=XPK_CAP_H+1e-9,"XPK_BUDGET_EXCEEDED")
    gate(project<=PROJECT_CAP_H+1e-9,"PROJECT_BUDGET_EXCEEDED")
    ledger.update(xpk_total_h=xpk,project_total_h=project,
        xpk_remaining_h=max(0.,XPK_CAP_H-xpk),
        project_remaining_h=max(0.,PROJECT_CAP_H-project),
        reserve_h=reserve,
        can_admit_full_reserved_stage=(not ledger["unfinished_workers"]
           and xpk+reserve<=XPK_CAP_H
           and project+reserve<=PROJECT_CAP_H))
    return ledger
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    p=Path(a.events)
    gate(p.is_file() and not p.is_symlink(),"EVENT_FILE")
    events=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    d=remaining(events,time.time(),a.prior_project_h,a.other_xpk_h,reserve_h=a.reserve_h)
    d.update(status="A47_CPU_LEDGER_READ_ONLY",
             source_event_file=str(p),
             deadline_et=DEADLINE,
             no_GPU_work_launched=True)
    print(json.dumps(d,indent=2,sort_keys=True))
    return 0
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--events",required=True)
    parser.add_argument("--prior-project-h",required=True,type=float)
    parser.add_argument("--other-xpk-h",default=0.,type=float)
    parser.add_argument("--reserve-h",default=0.,type=float)
    try:sys.exit(main(parser.parse_args()))
    except Exception as exc:
        print("ACTION_NEEDED_A47",repr(exc),file=sys.stderr)
        sys.exit(8)
