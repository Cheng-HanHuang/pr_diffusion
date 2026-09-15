#!/usr/bin/env python3
"""Technical/resource continuation gate for B26.1. Never gates on PSNR direction."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path

def readj(p): return json.loads(Path(p).read_text())
def writej(p,v):
    p=Path(p); tmp=p.with_name(p.name+f".tmp.{os.getpid()}"); tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(tmp,p)

def required(manifest,scope):
    rows=manifest["rows"]
    if scope=="smoke": rows=[r for r in rows if r["smoke_image"]]; roots=[0]
    elif scope=="dev16": rows=[r for r in rows if r["dev16"]]; roots=range(4)
    elif scope=="dev80": roots=range(4)
    else: raise ValueError(scope)
    return [(r,j,a) for r in rows for j in roots for a in ("H","R")]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--gpu-root",type=Path,required=True); ap.add_argument("--scope",choices=("smoke","dev16","dev80"),required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    m=readj(a.manifest); root=a.gpu_root.resolve(); jobs=required(m,a.scope)
    failures=list(root.glob("jobs/*/root*/*/attempt*/FAILURE.json"))
    if failures: raise RuntimeError(f"preserved failures present; scoped decision required: {failures[:4]}")
    results=[]; missing=[]
    by_image={r["image_id"]:r for r in m["rows"]}
    for row,ri,arm in jobs:
        done=root/"jobs"/row["image_id"]/f"root{ri}"/arm/"JOB_COMPLETE.json"
        if not done.is_file(): missing.append(str(done)); continue
        d=readj(done); rp=Path(d["result"]); v=readj(rp)
        if d.get("status")!="PASS" or v.get("status")!="PASS": raise RuntimeError(f"non-PASS {done}")
        if int(v["accounting"]["total_unet_evals"])!=2200: raise RuntimeError(f"UNet drift {rp}")
        if v.get("confirmation_payload_accessed") is not False or v.get("new_measurement_generated") is not False: raise RuntimeError(f"protected-data drift {rp}")
        if v["arm"]=="H":
            rep=v.get("historical_replay") or {}; acc=by_image[row["image_id"]]["np_roots"][ri]
            if not rep.get("terminal_hash_match") or float(rep.get("selector_plus_abs_error",1))>1e-12: raise RuntimeError(f"H replay failure {rp}: {rep}")
            if acc.get("rng_sha256") is not None and not rep.get("rng_hash_match"): raise RuntimeError(f"H RNG replay failure {rp}")
            if acc.get("selected_noise_sha256") is not None and not rep.get("selected_noise_hash_match"): raise RuntimeError(f"H noise replay failure {rp}")
        gm=v.get("gpu_monitor",{}); pm=gm.get("max_b24_process_gpu_mib")
        if pm is not None and int(pm)>52452: raise RuntimeError(f"memory cap {rp}: {pm}")
        results.append(v)
    if missing: raise RuntimeError(f"scope incomplete: have {len(results)}/{len(jobs)}; first missing {missing[0]}")
    receipts=[]
    for p in sorted(root.glob("receipts/WORKER_*.json"))+sorted(root.glob("WORKER_*.json")):
        try: v=readj(p)
        except Exception: continue
        if v.get("status")=="PASS": receipts.append(v)
    reservation=sum(float(v.get("reservation_wall_seconds",0)) for v in receipts)
    attempted=sum(int(v.get("attempted",0)) for v in receipts)
    avg=reservation/max(attempted,1)
    total_target=640
    remaining=max(0,total_target-len({(v["image_id"],v["root_index"],v["arm"]) for v in results}) if a.scope=="dev80" else total_target-len(results))
    projected=reservation+avg*remaining
    if projected>24*3600: raise RuntimeError(f"projected aggregate reservation {projected/3600:.3f} GPU-h > 24")
    max_mem=max([int(v.get("gpu_monitor",{}).get("max_b24_process_gpu_mib") or 0) for v in results]+[0])
    payload={"schema_version":"b26.gpu-gate.v1","status":"PASS","scope":a.scope,"required_jobs":len(jobs),"completed_jobs":len(results),"technical_continuation_only":True,"psnr_used_for_gate":False,"historical_H_replay_pass":True,"confirmation_payload_accessed":False,"new_measurements_generated":False,"max_process_gpu_mib":max_mem,"hard_ceiling_mib":52452,"reservation_seconds_recorded":reservation,"receipt_attempted_jobs":attempted,"mean_reservation_seconds_per_attempted_job":avg,"projected_full_campaign_reservation_seconds":projected,"projected_full_campaign_gpu_hours":projected/3600,"reservation_ceiling_gpu_hours":24.0,"created_unix_time":time.time()}
    writej(a.output,payload); print(json.dumps(payload,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
