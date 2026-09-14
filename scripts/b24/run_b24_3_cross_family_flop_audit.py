#!/usr/bin/env python3
"""Development-only cross-family dynamic FLOP audit for B24.3.

Selects one DEV80 calibration image prospectively by a fixed hash independent
of outcomes, then reruns DAPS-1, SITCOM-1, and exact PE3_SCORE solely under
PyTorch dispatch FLOP instrumentation.  The existing DEV80 scientific outputs
are not replaced by these audit reruns.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
SMOKE=REPO/"scripts/b24/run_b24_1_method_smoke.py"
FLOP_WRAPPER=REPO/"scripts/b24/run_python_cuda_flop_counted.py"
PE3=REPO/"scripts/b24/run_b24_3_pe3_dev80.py"
CTRL_PY=Path("/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python")
SCREEN_SHA="b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba"
GPU_UUIDS={0:"GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3",1:"GPU-883c037a-34d2-48c4-467f-9a352fd8fdff",2:"GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe",3:"GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a"}


def load_smoke():
    s=importlib.util.spec_from_file_location("b24_flop_smoke",SMOKE)
    if s is None or s.loader is None: raise ImportError(SMOKE)
    m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m

def readj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def writej(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def hkey(image): return hashlib.sha256(f"B24_DEV80_FLOP_AUDIT_V1|{SCREEN_SHA}|{image}".encode()).hexdigest()
def run(cmd,cwd,env,log):
    with Path(log).open("w",encoding="utf-8") as f:
        r=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,text=True,check=False)
    if r.returncode: raise RuntimeError(f"audit command rc={r.returncode}: {log}")
def replace_timing_wrapper(cmd,flop_json):
    out=list(cmd); out[1]=str(FLOP_WRAPPER)
    i=out.index("--timing-json"); out[i]="--flop-json"; out[i+1]=str(flop_json)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dev80-run",type=Path,required=True); ap.add_argument("--gpu",type=int,choices=range(4),required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    dev=args.dev80_run.resolve(); out=args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); (out/"logs").mkdir()
    manifest=readj(dev/"DEV80_MANIFEST.json")
    if manifest.get("image_count")!=80 or manifest.get("confirmation_exposed") is not False: raise RuntimeError("bad DEV80 source")
    tasks=list(manifest["tasks"])
    task=min(tasks,key=lambda t:(hkey(t["image_id"]),t["image_id"]))
    taskdir=dev/"workers"/f"gpu{task['assigned_gpu']}"/task["output_subdir"]
    comp=readj(taskdir/"IMAGE_COMPLETE.json")
    if comp.get("status")!="PASS": raise RuntimeError("audit source completion not PASS")
    input_manifest=Path(task["source_input_manifest"]).resolve() if task["pilot16"] else (taskdir/"input/input_manifest.json").resolve()
    role_row=(taskdir/"role_row.csv").resolve()
    if not input_manifest.is_file() or not role_row.is_file(): raise RuntimeError("audit source files missing")

    raw=subprocess.check_output(["nvidia-smi",f"--id={args.gpu}","--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],text=True).strip()
    uuid,free=[x.strip() for x in raw.split(",")]
    if uuid!=GPU_UUIDS[args.gpu] or int(free)<10240: raise RuntimeError(f"audit GPU gate failed uuid={uuid} free={free}")

    smoke=load_smoke(); smoke.MIN_FREE_MIB=10240
    item=readj(input_manifest)
    token=hkey(task["image_id"])[:10]
    data_name=f"b24-pe3-flop-g{args.gpu}-{task['image_id']}-{token}"
    data_dir=config_path=None
    records={}
    try:
        data_dir,config_path=smoke.prepare_daps_dataset(item,data_name)
        ds=smoke.child_spec("DAPS",REPO,item,data_name,out/"daps1",0,int(task["daps_solver_seeds"][0]),args.gpu)
        djson=out/"DAPS1_FLOPS.json"; dcmd=replace_timing_wrapper(ds["command"],djson)
        run(dcmd,ds["run_dir"],ds["env"],out/"logs/daps1.log")
        records["DAPS1"]=readj(djson)
    finally:
        if config_path is not None and config_path.exists(): config_path.unlink()
        if data_dir is not None and data_dir.exists(): shutil.rmtree(data_dir)

    ss=smoke.child_spec("SITCOM",REPO,item,None,out/"sitcom1",0,int(task["sitcom_solver_seeds"][0]),args.gpu)
    sjson=out/"SITCOM1_FLOPS.json"; scmd=replace_timing_wrapper(ss["command"],sjson)
    run(scmd,ss["run_dir"],ss["env"],out/"logs/sitcom1.log")
    records["SITCOM1"]=readj(sjson)

    penv=os.environ.copy(); penv.update({"CUDA_VISIBLE_DEVICES":str(args.gpu),"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":str(REPO)+(os.pathsep+penv["PYTHONPATH"] if penv.get("PYTHONPATH") else "")})
    pjson=out/"PE3_SCORE_FLOPS.json"
    pcmd=[str(CTRL_PY),str(FLOP_WRAPPER),"--script",str(PE3),"--cwd",str(REPO),"--flop-json",str(pjson),"--","--input-manifest",str(input_manifest),"--role-row",str(role_row),"--output-root",str(out/"pe3_score/methods"),"--physical-gpu",str(args.gpu),"--arms","NP_PE3_SCORE"]
    run(pcmd,REPO,penv,out/"logs/pe3_score.log")
    records["NP_PE3_SCORE"]=readj(pjson)

    for name,r in records.items():
        if int(r.get("status",1))!=0 or not isinstance(r.get("dispatch_supported_dynamic_flops"),int) or r["dispatch_supported_dynamic_flops"]<=0:
            raise RuntimeError(f"invalid FLOP record {name}: {r}")
    pe=float(records["NP_PE3_SCORE"]["dispatch_supported_dynamic_flops"])
    d=float(records["DAPS1"]["dispatch_supported_dynamic_flops"]); s=float(records["SITCOM1"]["dispatch_supported_dynamic_flops"])
    summary={
      "schema_version":"b24.cross-family-flop-audit.v1","status":"PASS","scope":"DEV80_ONLY","confirmation_exposed":False,
      "calibration_selection_domain":"B24_DEV80_FLOP_AUDIT_V1","calibration_image_id":task["image_id"],"calibration_screening_stratum":task["class_label"],
      "measurement_tensor_sha256":item["measurement_tensor_sha256"],"physical_gpu":args.gpu,"gpu_uuid":uuid,
      "metric":"PyTorch dispatch-supported dynamic FLOPs; unsupported operations reported separately and not silently treated as free",
      "dispatch_supported_dynamic_flops":{"NP_PE3_SCORE":int(pe),"DAPS1":int(d),"DAPS4_INDEPENDENT_EQUIVALENT":int(4*d),"SITCOM1":int(s),"SITCOM4_INDEPENDENT_EQUIVALENT":int(4*s)},
      "ratios_to_pe3_score":{"DAPS1":d/pe,"DAPS4":4*d/pe,"SITCOM1":s/pe,"SITCOM4":4*s/pe},
      "runtime_diagnostics":{name:{"gpu_active_seconds":r["gpu_active_seconds"],"wall_seconds":r["wall_seconds"],"runtime_counters":r["runtime_counters"],"fft2_shape_counts":r["fft2_shape_counts"],"ifft2_shape_counts":r["ifft2_shape_counts"]} for name,r in records.items()},
      "interpretation_guard":"Ratios apply to dispatch-supported FLOPs only. FFT/custom unsupported kernels remain separately visible; do not claim exact total-FLOP equivalence unless unsupported work is shown negligible or analytically incorporated.",
      "source_dev80_run":str(dev),"records":{"DAPS1":str(djson),"SITCOM1":str(sjson),"NP_PE3_SCORE":str(pjson)},
    }
    writej(out/"CROSS_FAMILY_FLOP_AUDIT.json",summary); print(json.dumps(summary,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
