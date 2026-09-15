#!/usr/bin/env python3
"""Run one B26 CPU scientific command with wall/RSS receipts."""
from __future__ import annotations
import argparse,json,os,resource,subprocess,time
from pathlib import Path

def writej(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(tmp,p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--max-wall-seconds",type=float,default=14400); ap.add_argument("--max-rss-gib",type=float,default=16); ap.add_argument("command",nargs=argparse.REMAINDER); a=ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise RuntimeError("CPU wrapper requires CUDA_VISIBLE_DEVICES='' exactly")
    cmd=a.command[1:] if a.command and a.command[0]=="--" else a.command
    if not cmd: raise RuntimeError("missing command")
    start=time.perf_counter(); timed=False
    try: cp=subprocess.run(cmd,check=False,timeout=a.max_wall_seconds)
    except subprocess.TimeoutExpired: timed=True; rc=124
    else: rc=cp.returncode
    wall=time.perf_counter()-start; ru=resource.getrusage(resource.RUSAGE_CHILDREN); rss=ru.ru_maxrss/1024/1024
    ok=(rc==0 and not timed and rss<=a.max_rss_gib and wall<=a.max_wall_seconds)
    rec={"schema_version":"b26.cpu-resource.v1","status":"PASS" if ok else "FAIL","command":cmd,"returncode":rc,"timed_out":timed,"wall_seconds":wall,"max_wall_seconds":a.max_wall_seconds,"ru_maxrss_raw_kib":ru.ru_maxrss,"max_rss_gib":rss,"max_rss_gib_limit":a.max_rss_gib,"user_cpu_seconds":ru.ru_utime,"system_cpu_seconds":ru.ru_stime,"cuda_visible_devices":""}
    writej(a.output,rec); print(json.dumps(rec,sort_keys=True)); raise SystemExit(0 if ok else 2)
if __name__=="__main__": main()
