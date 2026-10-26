"""Track R GPU occupancy telemetry and contamination accounting.

No GPU allocations, CUDA import, or model inference. Observations are local
nvidia-smi reads. Foreign processes are *observations*, not automatic blockers.
"""
from __future__ import annotations
import dataclasses
import datetime
import re
import subprocess
import time

@dataclasses.dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_pct: int
    process_pids: tuple[int,...]=()
    foreign_pids: tuple[int,...]=()
    visibility: str="known"

def _int(text):
    m=re.search(r"[-+]?\d+",str(text))
    if not m:
        raise ValueError("STOP_GPU_NON_NUMERIC "+str(text))
    return int(m.group(0))

def _call(cmd,timeout=12):
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=False)
    if result.returncode!=0:
        raise RuntimeError("STOP_NVIDIA_SMI_"+str(result.returncode)+": "+result.stderr[:180])
    return result.stdout

def descendants(roots, processes=None):
    """Tracked root PID plus descendants; process list is ps(pid,ppid)."""
    alive=set(int(p) for p in roots if int(p)>0)
    if processes is None:
        output=_call(["ps","-eo","pid=,ppid="])
        processes=[]
        for line in output.splitlines():
            cols=line.strip().split()
            if len(cols)==2:
                try:processes.append((int(cols[0]),int(cols[1])))
                except ValueError:pass
    for _ in range(16):
        more=alive|{pid for pid,ppid in processes if ppid in alive}
        if more==alive:break
        alive=more
    return alive

def collect(own_roots=(), nvidia=_call, processes=None):
    """Queries inventory and compute PIDs; PID telemetry failure -> unknown."""
    raw=nvidia(["nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"])
    inventory={}
    for line in raw.splitlines():
        cols=[s.strip() for s in line.split(",")]
        if len(cols)!=6:raise RuntimeError("STOP_GPU_INVENTORY_SCHEMA")
        index=_int(cols[0])
        if index in inventory:raise RuntimeError("STOP_GPU_DUPLICATE_INDEX")
        inventory[index]=(cols[1],_int(cols[2]),_int(cols[3]),_int(cols[4]),_int(cols[5]))
    if not inventory:raise RuntimeError("STOP_NO_GPU_DEVICES")
    try:
        proc_raw=nvidia(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                         "--format=csv,noheader,nounits"])
        visibility="known"
        by_uuid={v[0]:set() for v in inventory.values()}
        for line in proc_raw.splitlines():
            line=line.strip()
            if not line or "No running processes" in line:continue
            parts=[p.strip() for p in line.split(",")]
            if len(parts)<2:raise ValueError("bad compute process schema")
            if parts[0] in by_uuid:
                by_uuid[parts[0]].add(_int(parts[1]))
    except (RuntimeError,ValueError,subprocess.TimeoutExpired):
        visibility="unknown"
        by_uuid={v[0]:set() for v in inventory.values()}
    own=descendants(own_roots,processes) if own_roots else set()
    results=[]
    for idx,(uuid,total,used,free,util) in sorted(inventory.items()):
        pids=tuple(sorted(by_uuid[uuid]))
        foreign=tuple(p for p in pids if p not in own)
        results.append(GPU(idx,uuid,total,used,free,util,pids,foreign,visibility))
    return results

def classify_timing(samples, gpu_index, *, unknown_used_mib=2048):
    """Any foreign-process overlap contaminates a timing claim, not PSNR."""
    seen=[]
    visibility_missing=False
    nonidle_unattributed=False
    sample_count=0
    for sample in samples:
        card=next((g for g in sample if g.index==gpu_index),None)
        if card is None:
            visibility_missing=True
            continue
        sample_count+=1
        seen.extend(card.foreign_pids)
        if card.visibility!="known":
            visibility_missing=True
        if not card.process_pids and card.used_mib>unknown_used_mib:
            nonidle_unattributed=True
    if not sample_count:
        return {"timing_class":"UNVERIFIED","timing_valid":False,
                "foreign_pids":[],"n_samples":0,
                "reasons":["no_samples"]}
    reasons=[]
    if seen:reasons.append("foreign_gpu_process")
    if visibility_missing:reasons.append("process_visibility_unknown")
    if nonidle_unattributed:reasons.append("unattributed_gpu_memory")
    return {"timing_class":"CONTAMINATED" if seen else
                  ("UNVERIFIED" if reasons else "CLEAN"),
            "timing_valid":not reasons,
            "foreign_pids":sorted(set(seen))[:32],
            "n_samples":sample_count,
            "reasons":reasons}

def choose_gpu(cards, required_mib, *, reservations=None, safety_margin_mib=1024,
               fraction=0.80, preferred=(), clean_only=False):
    """80% hard cap; other processes reduce space but are not a veto."""
    reservations=reservations or {}
    cards=sorted(cards,key=lambda g: (g.index not in preferred,
                           bool(g.foreign_pids) if clean_only else False,
                           -(g.free_mib-reservations.get(g.index,0)),g.index))
    reasons={}
    for c in cards:
        cap=int(fraction*c.total_mib)
        reserved=int(reservations.get(c.index,0))
        # An in-flight reservation can be double-counted after allocation,
        # which is conservative: never oversubscribe the 80% cap.
        occupied=max(c.used_mib,c.total_mib-c.free_mib)+reserved
        if occupied+int(required_mib)+safety_margin_mib>cap:
            reasons[c.index]="WAIT_VRAM"
            continue
        if clean_only and (c.foreign_pids or c.visibility!="known" or
                           (not c.process_pids and c.used_mib>2048)):
            reasons[c.index]="WAIT_CLEAN_TIMING"
            continue
        return c.index,reasons
    return None,reasons

def event(gpus, *, stage, job_id, action, selected_gpu=None, own_roots=()):
    return {
        "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "monotonic_sec":time.monotonic(),
        "stage":stage,"job_id":job_id,"action":action,
        "selected_gpu":selected_gpu,
        "gpu_observations":[dataclasses.asdict(g) for g in gpus],
        "tracked_roots":list(own_roots)}
