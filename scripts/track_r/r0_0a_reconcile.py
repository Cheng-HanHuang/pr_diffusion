#!/usr/bin/env python3
"""R0.0a CPU-only FAST wheel/record-layout reconciliation, bounded explicit paths.

No GPU initialization. Inspect only the exposed B22, DEV80 and A1.2 roots
recorded in the planner's path configuration. No recursive directory scans.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path("/egr/research-pac/huang248")
FORBIDDEN=("confirmation","confirm305")


def safe(value):
    p=Path(value)
    if any(word in str(p).lower() for word in FORBIDDEN):
        raise RuntimeError("STOP_PROTECTED_PATH_TOKEN")
    if any(word in str(p.resolve(strict=False)).lower() for word in FORBIDDEN):
        raise RuntimeError("STOP_PROTECTED_SYMLINK")
    return p


def short_json(path, fields=None):
    p=safe(path)
    if not p.is_file():
        return {"status":"MISSING","path":str(p)}
    try:
        value=json.loads(p.read_text())
        if not isinstance(value,dict):
            return {"status":"WARN","path":str(p),"type":type(value).__name__}
        keys=list(value)[:60]
        result={"status":"VERIFIED","path":str(p),"keys":keys}
        if fields:
            result["selected"]={k:value.get(k) for k in fields if k in value}
        return result
    except Exception as e:
        return {"status":"WARN","path":str(p),"reason":repr(e)[:300]}


def bounded_names(path,max_entries=35):
    path=safe(path)
    if not path.is_dir():
        return {"status":"MISSING","path":str(path)}
    return {"status":"VERIFIED","path":str(path),
            "names":sorted(p.name for p in path.iterdir())[:max_entries]}


def run(argv, timeout=300, env=None):
    start=time.monotonic()
    try:
        p=subprocess.run(argv, capture_output=True, text=True, env=env,
                         timeout=timeout, check=False)
        return {"status":"VERIFIED" if p.returncode==0 else "WARN",
                "rc":p.returncode,"elapsed_s":round(time.monotonic()-start,3),
                "stdout":p.stdout[-1500:], "stderr":p.stderr[-1500:]}
    except subprocess.TimeoutExpired:
        return {"status":"WARN","reason":"CPU import timed out","timeout_s":timeout,
                "elapsed_s":round(time.monotonic()-start,3)}
    except OSError as e:
        return {"status":"MISSING","reason":str(e)[:300]}


ARCH_SCRIPT = """
import sys,torch,json
f=getattr(torch._C,"_cuda_getArchFlags",None)
flags=f() if callable(f) else None
print(json.dumps({"python":sys.version.split()[0],"torch":torch.__version__,
    "cuda_wheel":torch.version.cuda,"compiled_arch_flags":flags,
    "sm_120":bool(flags and "sm_120" in flags.split()),
    "gpu_visibility":"hidden"},sort_keys=True))
"""

IMPORT_SCRIPT = """
import importlib,json,sys
names=["forward_operators","main","misc","model"]
import_status={}
for n in names:
    try:
        m=importlib.import_module(n)
        import_status[n]=getattr(m,"__file__",None)
    except Exception as e:
        import_status[n]=repr(e)[:350]
print(json.dumps(import_status,sort_keys=True))
"""


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--paths",required=True)
    a=ap.parse_args()
    out=safe(a.out)
    base=ROOT/"outputs/pr_diffusion/track_r"
    if not str(out).startswith(str(base)+os.sep) or out.exists():
        raise RuntimeError("STOP_OUTPUT_NOT_FRESH_TRACK_R")
    config=json.loads(safe(a.paths).read_text())
    wt=safe(config["track_r"]["worktree"])
    actual=run(["git","-C",str(wt),"rev-parse","HEAD"],timeout=30)
    dirty=run(["git","-C",str(wt),"status","--porcelain"],timeout=30)
    if actual["stdout"].strip()!=a.expected_head or dirty["rc"] or dirty["stdout"].strip():
        raise RuntimeError("STOP_HEAD_OR_CLEAN_TREE_GATE")
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONNOUSERSITE":"1",
         "PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    paths=config["sources"]
    fastpy=safe(config["python"]["fast_dips_a15_blackwell"])
    results={"schema":"prdiff.track_r.r0_0a.cpu_probes.v1",
             "utc":dt.datetime.now(dt.timezone.utc).isoformat(),
             "git_head":a.expected_head,"gpu_forward_calls":0,"checks":{}}
    checks=results["checks"]
    checks["fast_arch"] = run([str(fastpy),"-c",ARCH_SCRIPT],env=env,timeout=240)
    st=checks["fast_arch"]
    if st.get("rc")==0:
        try:
            obj=json.loads(st["stdout"].strip().splitlines()[-1])
            st["reported"]=obj
            st["status"]="VERIFIED" if obj.get("sm_120") else "MISMATCH"
        except (IndexError,ValueError):
            st["status"]="WARN"
    fastenv={**env,"PYTHONPATH":config["external"]["fast_dips"]["path"]}
    checks["fast_imports"] = run([str(fastpy),"-c",IMPORT_SCRIPT],env=fastenv,timeout=240)

    b22=safe(paths["S1_b22"]["b22_run_root"])/"full"
    for fam in ("sitcom","np"):
        rows=sorted((b22/fam).glob("row*"))[:1]
        checks["S1."+fam+".row"] = bounded_names(rows[0]) if rows else {"status":"MISSING"}
        if rows:
            candidates = sorted((rows[0]/"candidates").glob("*"))[:2]
            checks["S1."+fam+".candidate_dirs"] = [bounded_names(p) for p in candidates]
            if candidates:
                checks["S1."+fam+".candidate_receipt"] = short_json(
                    candidates[0]/"result.json",
                    ["image_id","run_index","config_tag","seed","measurement_path",
                     "reconstruction_png_path","reconstruction_tensor_path",
                     "metrics","selector","selector_stats","timing"])
    fresh_root=safe(paths["S1_b22"]["b21_fresh_root"])
    checks["S1.fresh_root"] = bounded_names(fresh_root)
    fresh_csv=safe(paths["S1_b22"]["b21_fresh_rows_csv"])
    if fresh_csv.is_file():
        with fresh_csv.open(newline="",encoding="utf-8") as f:
            reader=csv.DictReader(f)
            checks["S1.fresh_csv"]={"status":"VERIFIED","columns":reader.fieldnames,
                "example":[{k:(str(v)[:220] if v is not None else None)
                            for k,v in list(row.items())[:40]}
                           for row in [next(reader,{})]]}
    else:
        checks["S1.fresh_csv"]={"status":"MISSING"}

    d80=paths["S2_b24_dev80"]
    rows=json.loads(safe(wt/d80["manifest_in_repo"]).read_text())["rows"]
    first=rows[0]
    checks["S2.dev80_run_root"] = bounded_names(d80["run_root"])
    checks["S2.corrected_closeout"] = bounded_names(d80["corrected_closeout"])
    checks["S2.example_image_complete"] = short_json(
        first["b24_image_complete_path"],["status","fresh_baselines","methods","image_id"])
    checks["S2.example_task"] = short_json(
        first["b24_task_path"],["status","method","image_id","output_dir"])
    checks["S2.example_np4_result"] = short_json(
        first["accepted_np4_result_path"],
        ["status","image_id","terminal_paths","candidate_paths","roots",
         "reconstructions","selector","per_root","root_results","metrics"])
    checks["S2.example_np4_dir"] = bounded_names(
        safe(first["accepted_np4_result_path"]).parent)
    task_dir=safe(first["ground_truth_tensor_path"]).parents[1]
    checks["S2.example_task_root"] = bounded_names(task_dir)

    s3=paths["S3_a12"]
    freeze=safe(s3["inputs_root"])/"A1_2_INPUT_FREEZE_COMPLETE.json"
    frozen=json.loads(freeze.read_text())
    checks["S3.freeze_summary"]={"status":"VERIFIED","keys":list(frozen)[:40],
        "row_count":len(frozen.get("rows",[])),
        "first_row_keys":list(frozen["rows"][0])[:45] if frozen.get("rows") else []}
    out.mkdir(parents=True,exist_ok=False)
    result=out/"R0_0A_PROBES.json"
    result.write_text(json.dumps(results,indent=2,sort_keys=True,default=str)+"\n")
    print("R0_0A_PROBES="+str(result))
    print("R0_0A_FAST_ARCH="+checks["fast_arch"]["status"])
    print("R0_0A_FAST_IMPORTS="+checks["fast_imports"]["status"])
    for name,rec in checks.items():
        if isinstance(rec,dict) and rec.get("status") in ("WARN","MISSING","MISMATCH"):
            print(rec["status"],name)
    return 0


if __name__=="__main__":
    sys.exit(main())
