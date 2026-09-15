#!/usr/bin/env python3
"""Fail-closed B26 machine completion after CPU + full DEV80 GPU stages."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path

def readj(p): return json.loads(Path(p).read_text())
def writej(p,v):
 p=Path(p); t=p.with_name(p.name+f".tmp.{os.getpid()}"); t.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(t,p)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--run",type=Path,required=True); a=ap.parse_args(); run=a.run.resolve()
 if (run/"B26_COMPLETE.json").exists(): raise FileExistsError(run/"B26_COMPLETE.json")
 pre=readj(run/"PRE_RUN_IDENTITY.json"); gate=readj(run/"gpu/GATE_dev80.json"); ga=readj(run/"gpu/ANALYSIS_dev80.json"); cr=readj(run/"cpu/CPU_RESOURCE.json"); cs=readj(run/"cpu/experiment/CPU_SUMMARY.json")
 if any(v.get("status")!="PASS" for v in (pre,gate,ga,cr,cs)): raise RuntimeError("one or more required B26 components are not PASS")
 if gate.get("completed_jobs")!=640 or ga.get("root_trajectory_count")!=640: raise RuntimeError("GPU trajectory count drift")
 if cs.get("observations")!=128 or cs.get("total_complete_reverse_trajectories")!=393216: raise RuntimeError("CPU trajectory count drift")
 failures=list((run/"gpu/jobs").glob("*/root*/*/attempt*/FAILURE.json"));
 if failures: raise RuntimeError(f"preserved GPU failures require discrepancy decision: {failures[:4]}")
 if gate.get("confirmation_payload_accessed") is not False or cs.get("confirmation_payloads_accessed") is not False: raise RuntimeError("confirmation access flag")
 if gate.get("new_measurements_generated") is not False: raise RuntimeError("new FFHQ measurement flag")
 payload={"schema_version":"b26.machine-complete.v1","status":"PASS","pre_run_commit":pre["pre_run_commit"],"gpu_root_trajectories_attempted":640,"gpu_root_trajectories_completed":640,"cpu_complete_reverse_trajectories":393216,"synthetic_measurements_generated":128,"new_ffhq_measurements_generated":False,"confirmation_payloads_accessed":False,"weighted_ffhq_reconstruction_performed":False,"gpu_gate":str((run/"gpu/GATE_dev80.json").resolve()),"gpu_analysis":str((run/"gpu/ANALYSIS_dev80.json").resolve()),"cpu_summary":str((run/"cpu/experiment/CPU_SUMMARY.json").resolve()),"cpu_resource":str((run/"cpu/CPU_RESOURCE.json").resolve()),"recommendation":"PENDING_EXECUTOR_SCIENTIFIC_INTERPRETATION","completed_unix_time":time.time()}
 writej(run/"B26_COMPLETE.json",payload); print(json.dumps(payload,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
