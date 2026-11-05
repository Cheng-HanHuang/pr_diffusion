#!/usr/bin/env python3
"""A46 XP-K read-only, co-tenant-safe GPU admission (NOT an execution controller).

Do not treat GPU utilization as exclusivity. Admission checks actual available
VRAM and the 80%-of-total ceiling; optionally prefer quiet cards. Recheck
immediately before worker allocation; admission is not a reservation.
"""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path

MIN_FREE_GIB=4.0
CAP_FRACTION=0.80
# Conservative planning estimates, not asserted measured XP-K peaks.
DEFAULT_RESERVE_GIB={"SITCOM":28.0,"FAST":28.0,"DAPS":28.0}
def parse_gpu_csv(csv):
    """Parse nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu."""
    out=[]
    for line in csv.splitlines():
        if not line.strip():continue
        fields=[s.strip() for s in line.split(",")]
        if len(fields)!=4:raise ValueError("STOP_A46_BAD_GPU_CSV")
        try:i,total,used,util=map(int,fields)
        except ValueError as exc:raise ValueError("STOP_A46_BAD_GPU_NUMBER") from exc
        if i<0 or total<=0 or used<0 or used>total or not 0<=util<=100:
            raise ValueError("STOP_A46_BAD_GPU_TELEMETRY")
        out.append(dict(index=i,total_mib=total,used_mib=used,util_pct=util))
    if not out or len({g["index"] for g in out})!=len(out):
        raise ValueError("STOP_A46_MISSING_OR_DUPLICATE_GPU")
    return out

def eligible(gpus,reserve_gib,occupied_indices=()):
    if not 0<reserve_gib<1000:raise ValueError("STOP_A46_RESERVATION")
    if reserve_gib<4:raise ValueError("STOP_A46_UNSAFE_RESERVATION")
    occupied=set(occupied_indices)
    candidates=[]
    reserve=reserve_gib*1024
    for g in gpus:
        # 4GiB remains free after allocation, and usage stays below 80%.
        projected=g["used_mib"]+reserve
        margin=g["total_mib"]-projected
        cap=g["total_mib"]*CAP_FRACTION
        ok=(g["index"] not in occupied and margin>=MIN_FREE_GIB*1024
            and projected<=cap)
        if ok:
            candidates.append(dict(**g,projected_used_mib=projected,
                                   projected_free_mib=margin))
    # Quiet preferred, NOT required; free memory breaks ties.
    return sorted(candidates,key=lambda x:(x["util_pct"],-x["projected_free_mib"],x["index"]))

def snapshot():
    cmd=["nvidia-smi","--query-gpu=index,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=20,check=False)
    if p.returncode:raise RuntimeError("STOP_A46_NVIDIA_SMI "+p.stderr[-250:])
    return parse_gpu_csv(p.stdout)

def main(a):
    if a.family not in DEFAULT_RESERVE_GIB:raise ValueError("STOP_A46_FAMILY")
    reserve=a.reserve_gib if a.reserve_gib is not None else DEFAULT_RESERVE_GIB[a.family]
    gpus=snapshot()
    admitted=eligible(gpus,reserve,a.busy_index)
    result=dict(status="A46_ADMISSION_CANDIDATE_ONLY" if admitted else "WAIT_SAFE_GPU",
                family=a.family,reserve_gib=reserve,cap_fraction=CAP_FRACTION,
                minimum_remaining_gib=MIN_FREE_GIB,
                observed_at_unix=time.time(),gpus=gpus,
                ranked_candidates=admitted,
                foreign_processes_never_modified=True,
                gpu_utilization_not_a_cleanliness_gate=True,
                requires_recheck_under_controller_lock=True,
                is_reservation=False,no_gpu_work_launched=True)
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--family",required=True,choices=tuple(DEFAULT_RESERVE_GIB))
    p.add_argument("--reserve-gib",type=float)
    p.add_argument("--busy-index",type=int,action="append",default=[])
    try:sys.exit(main(p.parse_args()))
    except Exception as exc:
        print("ACTION_NEEDED_A46",repr(exc),file=sys.stderr)
        sys.exit(8)
