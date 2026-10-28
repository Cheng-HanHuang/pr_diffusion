#!/usr/bin/env python3
"""A9 B16 full B22 historical SITCOM replay: one four-run image group.

Reuses the audited native historical B22 sampler and RNG-reset sequence.
GPU process receives ONLY the GT-free 400-row manifest. No score or labels.
The producer never kills, signals or interacts with external processes.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from types import SimpleNamespace

from scripts.track_r.xp_a10_b16_gate_prepare_cpu import (
    ROOT,BASE,WT,A6,A6_HEAD,MODEL_SHA,SITCOM,SITCOM_HEAD,
    settings,check_native_checkout,external_checkpoint_provenance,
    safe,sha,git)
from scripts.track_r import xp_sitcom_replay_gpu as native

PY=ROOT/"conda-envs/sitcom_ode_bw/bin/python"

def check(v,why):
    if not v:raise RuntimeError("STOP_B16_GATE_WORKER_"+why)

def image_folder(run,image):
    return run/"gpu"/f"b22_{image}"

def checked_source(run,expected_head,index,gpu,preflight):
    check(Path(sys.executable).resolve()==PY.resolve(),"PYTHON")
    check(os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu),"CUDA_VISIBLE")
    check(os.environ.get("PYTHONDONTWRITEBYTECODE")=="1","NO_PYC")
    check(os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
          str(run/"pycache")+"/"),"ISOLATED_CACHE")
    check(git(WT,"rev-parse","HEAD")==expected_head and
          not git(WT,"status","--porcelain"),"WORKTREE")
    check(git(A6,"rev-parse","HEAD")==A6_HEAD and
          not git(A6,"status","--porcelain"),"A6")
    check(git(SITCOM,"rev-parse","HEAD")==SITCOM_HEAD,"UPSTREAM_HEAD")
    source=check_native_checkout(
        git(SITCOM,"rev-parse","HEAD"),
        git(SITCOM,"status","--porcelain=v1","--untracked-files=no"),
        git(SITCOM,"diff","--numstat","HEAD","--","posterior_sample.py"))
    source["unused_cli_sha256"]=sha(SITCOM/"posterior_sample.py")
    checkpoint=external_checkpoint_provenance(SITCOM/"checkpoint/ffhq256.pt")
    pre=json.loads(safe(run/"XP_A10_B16_GATE_PREP.json").read_text())
    raw=safe(run/"XP_A10_B16_GATE_TASKS.json")
    check(sha(raw)==pre["manifest_sha256"],"TASKS_SHA")
    manifest=json.loads(raw.read_text())
    check(pre.get("status")=="XP_A10_B16_GATE_PREP_PASS" and
          pre.get("head")==expected_head and pre.get("n_roots")==400 and
          manifest.get("head")==expected_head and
          manifest.get("schema")=="prdiff.track_r.xp.a10.b16-gate.v1" and
          manifest.get("n_roots")==400 and manifest.get("n_images")==100 and
          manifest.get("GT_supplied") is False and
          manifest.get("historical_candidate_supplied") is False and
          manifest.get("source_head")==SITCOM_HEAD and
          manifest.get("model_sha256")==MODEL_SHA and
          manifest.get("config")==settings(
             json.loads(safe(A6/"configs/b22/b22_2_overnight.json").read_text()))
          and manifest.get("native_checkout_provenance")==source and
          manifest.get("native_checkpoint_provenance")==checkpoint,
          "MANIFEST_OR_PROVENANCE")
    rows=manifest["rows"]
    check(len(rows)==400 and 0<=index<100,"IMAGE_INDEX")
    selected=rows[index*4:index*4+4]
    image=selected[0]["image_id"]
    check([int(r["rep"]) for r in selected]==[0,1,2,3] and
          len({r["image_id"] for r in selected})==1 and
          all(r["source"]=="S1_b22" and
              r["measurement_source"]=="S1_b22" and
              r["rng_mode"]=="sequential_post_model" and
              r["seed"]==43 and
              r["family"]=="SITCOM" for r in selected) and
          len({r["uid"] for r in rows})==400,
          "IMAGE_4_SEQUENCE")
    for r in selected:
        check(sha(r["measurement_path"])==r["measurement_sha256"],
              "MEASUREMENT_DRIFT")
    output=image_folder(run,image)
    check(output.parent==run/"gpu" and
          (preflight or not output.exists()) and not output.is_symlink(),
          "OUTPUT_EXISTS")
    return manifest,selected,output,pre

def native_adapter(a,mf):
    """Adapt frozen A10 field names to the audited native B22 sampler API."""
    check(mf["source_head"]==SITCOM_HEAD,"NATIVE_SOURCE_ALIAS")
    check("native_source_head" not in mf,"NO_MANIFEST_FIELD_REWRITE")
    worker_manifest=dict(mf,native_source_head=mf["source_head"])
    args=SimpleNamespace(mode="b22",out=str(a.out),gpu=a.gpu,
                         expected_head=a.expected_head)
    return args,worker_manifest

def main(a):
    run=Path(a.run)
    check(run.parent==BASE and run.name.startswith("xp_a10_b16_gate_")
          and run.is_dir() and not run.is_symlink(),"RUN")
    check(0<=a.gpu<4,"GPU_INDEX")
    mf,rows,out,pre=checked_source(run,a.expected_head,a.index,a.gpu,a.preflight_only)
    if a.preflight_only:
        print("B16_GATE_GPU_PREFLIGHT_PASS",rows[0]["image_id"],flush=True)
        return 0
    # The legacy native implementation is reused ONLY for its native B22
    # run/seed algorithm. New per-image manifests are validated above.
    native.PREP_HEAD=a.expected_head
    native.MANIFEST_SHA=pre["manifest_sha256"]
    # The audited native generator expects the runner's expected_head
    # and its historical native_source_head key. The frozen A10 manifest
    # records this same source hash as source_head. Adapt in memory only:
    # never rewrite the immutable GPU/CPU manifests.
    worker_args,worker_manifest=native_adapter(
        SimpleNamespace(out=out,gpu=a.gpu,expected_head=a.expected_head),mf)
    result=native.generate(worker_args,worker_manifest,rows)
    print("B16_GATE_NATIVE_IMAGE_PASS",rows[0]["image_id"],flush=True)
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--index",required=True,type=int)
    ap.add_argument("--gpu",required=True,type=int)
    ap.add_argument("--preflight-only",action="store_true")
    try:sys.exit(main(ap.parse_args()))
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: B16_GATE_NATIVE",str(exc),flush=True)
        sys.exit(8)
