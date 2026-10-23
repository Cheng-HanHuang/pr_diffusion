#!/usr/bin/env python3
"""R0.5: freeze 720 GT-free native FAST-DIPS roots, no new measurements.

Inputs are the exact R0.4 canary's SHA-locked scoring tasks and three
successful R0.4 result lineages. This step is CPU only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prdiffusion.track_r.manifest import assert_no_forbidden

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
CANARY=BASE/"r0_4_score_20261022T193238Z"
PARALLEL=BASE/"r0_4_parallel_20261022T200149Z"
CANARY_HEAD="8aee60d98acfeba836fff4ec824137769b0a4698"
PARALLEL_HEAD="717100a17a2326f3fbbccb7cbe782ccf4e90126d"
RECOVERY_HEAD="62eaaafe3e22cd6222fc66858cad6dcef532a2cd"
TASK_SHA="38543cc4e4c394ab7c8936303a8d0fee6b46fb81e53f877a5ea6768e151078f7"
FAST_SOURCE_HEAD="a9758b38906b7c7950fa04491f37f6634948db26"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def safe(p):
    p=Path(assert_no_forbidden(str(p)))
    if (not p.is_absolute() or not p.is_file() or p.is_symlink() or
        assert_no_forbidden(str(p.resolve())) is None):
        raise RuntimeError("STOP_R05_UNSAFE_SOURCE")
    return p

def load(p):
    return json.loads(safe(p).read_text())

def native_seed(source,iid,rep):
    return int.from_bytes(hashlib.sha256(
        f"R0FAST|{source}|{iid}|{rep}".encode()).digest()[:8],
        "little") & ((1<<63)-1)

def write_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def verify_r04(recovery):
    """Audit exact old+new lineage before producing R0.5 inputs."""
    from scripts.track_r.r0_4_score_gpu import (
        validate_input,validate_complete,output_name)
    r=load(recovery/"AUTOPILOT_STATUS.json")
    if (r.get("status")!="R0_4_ALL_2520_COMPLETE_RECOVERED_4GPU" or
        r.get("head")!=RECOVERY_HEAD or
        r.get("completed_count")!=2520 or
        r.get("active_attempts") or
        r.get("audited_previous_scorings")!=1789 or
        r.get("source_canary")!=str(CANARY) or
        r.get("historical_failed_run")!=str(PARALLEL) or
        r.get("task_sha256")!=TASK_SHA or
        len(r.get("completed",[]))!=13 or
        not 0<float(r.get("total_gpu_seconds_including_canary",0))<21600):
        raise RuntimeError("STOP_R05_R04_NOT_COMPLETE")
    c=load(CANARY/"AUTOPILOT_STATUS.json")
    p=load(PARALLEL/"AUTOPILOT_STATUS.json")
    if (c.get("status")!="CANARY_COMPLETE" or
        c.get("head")!=CANARY_HEAD or
        c.get("completed_until")!=16 or
        p.get("status")!="STOP_WORKER" or
        p.get("head")!=PARALLEL_HEAD or
        p.get("completed_count")!=1744 or
        len(p.get("completed",[]))!=27):
        raise RuntimeError("STOP_R05_R04_OLD_LINEAGE_CHANGED")
    manifest=safe(CANARY/"prepare/R0_4_GT_FREE_TASKS.json")
    if sha(manifest)!=TASK_SHA:
        raise RuntimeError("STOP_R05_R04_TASK_SHA")
    tasks=validate_input(load(manifest),CANARY_HEAD)
    if len(tasks)!=2520:
        raise RuntimeError("STOP_R05_R04_TASK_COUNT")
    def validated(folder,items,head):
        for row in items:
            validate_complete(folder/"results"/output_name(row),row,TASK_SHA,head)
    validated(CANARY/"chunks/batch_0000_16",tasks[:16],CANARY_HEAD)
    seen=set(range(16))
    for item in p["completed"]:
        i,n=item["start"],item["count"]
        if len(set(range(i,i+n))&seen):
            raise RuntimeError("STOP_R05_DUPLICATE_OLD_COVERAGE")
        seen.update(range(i,i+n))
        validated(PARALLEL/"chunks"/f"batch_{i:04d}_{n:02d}",
                  tasks[i:i+n],PARALLEL_HEAD)
    partial=PARALLEL/"chunks/batch_1680_64"
    validated(partial,tasks[1680:1725],PARALLEL_HEAD)
    if len(set(range(1680,1725))&seen):
        raise RuntimeError("STOP_R05_PARTIAL_OVERLAP")
    seen.update(range(1680,1725))
    for item in r["completed"]:
        i,n=item["start"],item["count"]
        if len(set(range(i,i+n))&seen):
            raise RuntimeError("STOP_R05_DUPLICATE_RECOVERY_COVERAGE")
        seen.update(range(i,i+n))
        validated(recovery/"chunks"/f"batch_{i:04d}_{n:02d}",
                  tasks[i:i+n],RECOVERY_HEAD)
    if seen!=set(range(2520)):
        raise RuntimeError("STOP_R05_R04_GAPS")
    return tasks,float(r["total_gpu_seconds_including_canary"])

def fast_roots(tasks):
    sources={}
    for r in tasks:
        if r["source"] not in ("S1_b22","S2_b24_dev80"):
            continue
        key=(r["source"],str(r["image_id"]))
        identity=(r["measurement_path"],r["measurement_sha256"])
        if key in sources and sources[key]!=identity:
            raise RuntimeError("STOP_R05_IMAGE_MEASUREMENT_CONFLICT "+str(key))
        sources[key]=identity
    s1={k for k in sources if k[0]=="S1_b22"}
    s2={k for k in sources if k[0]=="S2_b24_dev80"}
    if len(s1)!=100 or len(s2)!=80 or len(sources)!=180:
        raise RuntimeError("STOP_R05_MEASUREMENT_UNIVERSE")
    records=[]
    for source,iid in sorted(sources):
        path,file_sha=sources[source,iid]
        if sha(safe(path))!=file_sha:
            raise RuntimeError("STOP_R05_MEASUREMENT_SHA "+iid)
        for rep in range(4):
            records.append({
                "uid":f"S5_r0_fast|FAST|{source}_{iid}|rep{rep}",
                "source":"S5_r0_fast","measurement_source":source,
                "image_id":iid,"rep":rep,
                "seed":native_seed(source,iid,rep),
                "measurement_path":path,
                "measurement_sha256":file_sha
            })
    if len(records)!=720 or len({r["uid"] for r in records})!=720:
        raise RuntimeError("STOP_R05_DUPLICATE_ROOTS")
    return records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--recovery-run",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    root=Path(a.out)
    if not str(root).startswith(str(BASE)+"/") or root.exists() or root.is_symlink():
        raise RuntimeError("STOP_R05_OUTPUT_FRESHNESS")
    if (subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
          text=True).strip()!=a.expected_head or
        subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
          text=True)):
        raise RuntimeError("STOP_R05_GIT_IDENTITY")
    if Path(a.recovery_run)!=BASE/"r0_4_recovery_20261022T235412Z":
        raise RuntimeError("STOP_R05_UNEXPECTED_R04_RECOVERY")
    tasks,spent=verify_r04(Path(a.recovery_run))
    roots=fast_roots(tasks)
    root.mkdir(parents=True,exist_ok=False)
    work={"schema":"prdiff.track_r.r0.5.gt_free_tasks.v1",
        "status":"R0_5_TASKS_FROZEN","head":a.expected_head,
        "fast_head":FAST_SOURCE_HEAD,
        "n_roots":720,"n_images":180,
        "git_and_input_ids_pinned":True,"gt_paths_supplied":False,
        "rows":roots}
    target=root/"R0_5_GT_FREE_ROOTS.json"
    write_x(target,work)
    receipt={"status":"R0_5_CPU_PREP_PASS","head":a.expected_head,
        "source_r04_recovery":a.recovery_run,
        "r04_gpu_seconds":spent,
        "root_sha256":sha(target),"n_roots":720,
        "n_images":180,"n_s1":100,"n_s2":80,
        "no_gt":True,"no_gpu":True}
    write_x(root/"R0_5_CPU_PREP_RECEIPT.json",receipt)
    print("R05_CPU_PREP_PASS",len(roots),"ROOTS",receipt["root_sha256"],flush=True)

if __name__=="__main__":
    sys.exit(main())
