#!/usr/bin/env python3
"""Read-only live GPU admission and timing-contamination snapshot (no CUDA)."""
import argparse
import json
import sys
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--memory-mib",type=int,default=16000)
    ap.add_argument("--memory-margin-mib",type=int,default=2048)
    a=ap.parse_args()
    cards=collect()
    selected,reasons=choose_gpu(cards,a.memory_mib,
                safety_margin_mib=a.memory_margin_mib)
    lines=["GPU_ADMISSION_MONITOR requested_mib=%d margin=%d"%
           (a.memory_mib,a.memory_margin_mib)]
    for c in cards:
        eligible,_=choose_gpu([c],a.memory_mib,
                              safety_margin_mib=a.memory_margin_mib)
        lines.append("GPU=%d used=%d total=%d free=%d util=%d%% foreign_pids=%s app_visibility=%s launch=%s timing=%s"%
            (c.index,c.used_mib,c.total_mib,c.free_mib,c.utilization_pct,
             ",".join(map(str,c.foreign_pids)) if c.foreign_pids else "none",
             c.visibility,
             "YES" if eligible is not None else "WAIT_VRAM",
             "CONTAMINATED" if c.foreign_pids else
                  ("UNVERIFIED" if c.visibility!="known" else "CURRENTLY_CLEAN")))
    lines.append("SELECTED_GPU="+(str(selected) if selected is not None else "WAIT_VRAM"))
    lines.append("OK")
    print("\n".join(lines[:10]))

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("ACTION NEEDED: GPU_PROBE "+str(exc))
        sys.exit(8)
