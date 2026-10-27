#!/usr/bin/env python3
"""CPU-only one-time recovery certification of *pre-GPU* XP-T SITCOM crash.

No cleanup, reset or overwriting: preserve original empty PID file and failed
supervisor log as evidence; reuse only provably empty GPU and cache folders.
No GT paths, held-out data or CUDA operations are used.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
RUN=ROOT/"outputs/pr_diffusion/track_r/xp_t_sitcom_20261026T233451Z"
MANIFEST_SHA="995915c3b7b7a00b1d4d678a4a013c8cc9c2445a513704a92e513d93d6d97131"
OLD_PID=3893497
RECEIPT="XP_T_SITCOM_PRE_GPU_RECOVERY.json"
ORIGINAL_LOG="XP_T_SITCOM_SUPERVISOR.log"
ORIGINAL_PID="XP_T_SITCOM_SUPERVISOR_PID.json"
EXPECTED_ERROR="TypeError: dump() missing 1 required positional argument: 'fp'"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def _read_bytes(path):
    p=Path(path)
    if not p.is_file() or p.is_symlink() or any(t in str(p).lower() for t in BAD):
        raise RuntimeError("STOP_SITCOM_RECOVERY_FILE_OR_SYMLINK "+str(p))
    return p.read_bytes()

def _sha(path):
    h=hashlib.sha256()
    h.update(_read_bytes(path))
    return h.hexdigest()

def _old_supervisor_is_alive():
    r=subprocess.run(["ps","-p",str(OLD_PID),"-o","args="],
                     text=True,capture_output=True,timeout=20,check=False)
    if r.returncode not in (0,1):
        raise RuntimeError("STOP_SITCOM_RECOVERY_PS_QUERY")
    return r.returncode==0 and "xp_t_sitcom_supervisor.py" in r.stdout

def audit_aborted_first_launch(run, *,require_recovery_receipt=False):
    run=Path(run)
    if run!=RUN or run.is_symlink() or not run.is_dir():
        raise RuntimeError("STOP_SITCOM_RECOVERY_WRONG_RUN")
    if _old_supervisor_is_alive():
        raise RuntimeError("STOP_SITCOM_RECOVERY_OLD_SUPERVISOR_ACTIVE")
    if _sha(run/"XP_T_SITCOM_TASKS.json")!=MANIFEST_SHA:
        raise RuntimeError("STOP_SITCOM_RECOVERY_MANIFEST_SHA")
    prep=json.loads(_read_bytes(run/"XP_T_SITCOM_PREP.json"))
    if (prep.get("status")!="XP_T_SITCOM_PREP_PASS" or
        prep.get("manifest_sha256")!=MANIFEST_SHA or
        prep.get("n_roots")!=18 or prep.get("no_gpu_work") is not True):
        raise RuntimeError("STOP_SITCOM_RECOVERY_PREP_RECEIPT")
    for d in ("gpu","pycache"):
        p=run/d
        if not p.is_dir() or p.is_symlink() or next(p.iterdir(),None) is not None:
            raise RuntimeError("STOP_SITCOM_RECOVERY_NONEMPTY_"+d)
    for name in ("XP_T_SITCOM_STATUS.json","GPU_EVENTS.jsonl",
                 "XP_T_SITCOM_COMPLETE.json",
                 "XP_T_SITCOM_RECOVERY_SUPERVISOR_PID.json"):
        if (run/name).exists() or (run/name).is_symlink():
            raise RuntimeError("STOP_SITCOM_RECOVERY_WORK_MAY_HAVE_STARTED "+name)
    retry_log=run/"XP_T_SITCOM_RECOVERY_SUPERVISOR.log"
    if require_recovery_receipt:
        # The shell opens the new log before execing the supervisor.
        if not retry_log.is_file() or retry_log.is_symlink():
            raise RuntimeError("STOP_SITCOM_RECOVERY_RETRY_LOG_MISSING")
    elif retry_log.exists() or retry_log.is_symlink():
        raise RuntimeError("STOP_SITCOM_RECOVERY_RETRY_ALREADY_STARTED")
    if _read_bytes(run/ORIGINAL_PID)!=b"":
        raise RuntimeError("STOP_SITCOM_RECOVERY_PID_NOT_EMPTY")
    failed_log=_read_bytes(run/ORIGINAL_LOG)
    if (EXPECTED_ERROR.encode() not in failed_log or
        b"ACTION NEEDED: XP_T_SITCOM_SUPERVISOR" not in failed_log):
        raise RuntimeError("STOP_SITCOM_RECOVERY_UNEXPECTED_ORIGINAL_CRASH")
    result={
      "old_pid":OLD_PID,
      "failed_log_sha256":hashlib.sha256(failed_log).hexdigest(),
      "failed_pid_sha256":_sha(run/ORIGINAL_PID),
      "frozen_manifest_sha256":MANIFEST_SHA,
      "gpu_dir_empty":True,"pycache_dir_empty":True,
      "no_gpu_workers_started":True,
      "source_error":"json.dump_missing_fp_before_status",
    }
    if require_recovery_receipt:
        r=json.loads(_read_bytes(run/RECEIPT))
        head=subprocess.check_output(
            ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
        if (r.get("status")!="XP_T_SITCOM_PRE_GPU_RECOVERY_AUTHORIZED" or
            r.get("expected_head")!=head or
            r.get("evidence")!=result):
            raise RuntimeError("STOP_SITCOM_RECOVERY_RECEIPT_CHANGED")
    elif (run/RECEIPT).exists() or (run/RECEIPT).is_symlink():
        raise RuntimeError("STOP_SITCOM_RECOVERY_ALREADY_AUTHORIZED")
    return result

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_SITCOM_RECOVERY_CPU_ONLY")
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30).strip()
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_SITCOM_RECOVERY_GIT")
    evidence=audit_aborted_first_launch(RUN)
    with (RUN/RECEIPT).open("x",encoding="utf-8") as f:
        json.dump({"status":"XP_T_SITCOM_PRE_GPU_RECOVERY_AUTHORIZED",
                   "expected_head":head,"evidence":evidence},
                  f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("XP_T_SITCOM_PRE_GPU_RECOVERY_AUTHORIZED")
    print("ORIGINAL_LOG_SHA="+evidence["failed_log_sha256"])
    print("ORIGINAL_PID_SHA="+evidence["failed_pid_sha256"])
    print("GPU_WORKERS_STARTED=false")
    print("ORIGINAL_FAILED_LOG_PRESERVED=true")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_T_SITCOM_RECOVERY "+str(e),flush=True)
        sys.exit(8)
