#!/usr/bin/env python3
"""GPU-only B16 production worker, gated by independently certified 400-root replay.

Never opens the historical GT/offline score or any new outcome label.
B22 generates full RNG prefix 0..3, saves only 4..15. B24 has one
independent model construction/seed per new trajectory.
"""
from __future__ import annotations
import argparse,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
from scripts.track_r.xp_a11_b16_prod_prepare_cpu import (
    ROOT,BASE,WT,A6,A6_HEAD,SITCOM,SITCOM_HEAD,MODEL_SHA,
    sha,git,safe,settings,check_native_checkout,
    external_checkpoint_provenance)
from scripts.track_r import xp_sitcom_replay_gpu as native

PY=ROOT/"conda-envs/sitcom_ode_bw/bin/python"

def check(condition,why):
    if not condition:raise RuntimeError("STOP_B16_PRODUCTION_GPU_"+why)

def preflight(args):
    run=Path(args.run)
    check(run.parent==BASE and run.name.startswith("xp_a11_b16_prod_")
          and run.is_dir() and not run.is_symlink(),"RUN")
    check(Path(sys.executable).resolve()==PY.resolve(),"PYTHON")
    check(0<=args.gpu<4 and os.environ.get("CUDA_VISIBLE_DEVICES")==str(args.gpu),
          "SINGLE_PHYSICAL_GPU")
    check(os.environ.get("PYTHONDONTWRITEBYTECODE")=="1" and
          os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
          str(run/"pycache")+"/"),"ISOLATED_CACHE")
    check(git(WT,"rev-parse","HEAD")==args.expected_head and
          not git(WT,"status","--porcelain"),"CODE_SHA")
    check(git(A6,"rev-parse","HEAD")==A6_HEAD and
          not git(A6,"status","--porcelain"),"A6_PIN")
    check(git(SITCOM,"rev-parse","HEAD")==SITCOM_HEAD,"SOLVER_HEAD")
    pre=json.loads(safe(run/"XP_A11_B16_PROD_PREP.json").read_text())
    manifest_file=run/"XP_A11_B16_PROD_TASKS.json"
    check(pre["head"]==args.expected_head and
          pre["status"]=="B16_PRODUCTION_PREP_PASS_GATE_PENDING" and
          sha(manifest_file)==pre["manifest_sha256"],"FROZEN_MANIFEST_SHA")
    mf=json.loads(manifest_file.read_text())
    check(mf["schema"]=="prdiff.track_r.xp.a11.b16-production.v1" and
          mf["n_new_roots"]==2160 and mf["n_new_b22"]==1200 and
          mf["n_new_b24"]==960 and mf["head"]==args.expected_head and
          mf["GT_supplied"] is False and
          mf["historical_candidate_supplied"] is False and
          mf["new_outcomes_embargoed"] is True and
          mf["source_head"]==SITCOM_HEAD and
          mf["model_sha256"]==MODEL_SHA,"MANIFEST_SCHEMA")
    state=check_native_checkout(
        git(SITCOM,"rev-parse","HEAD"),
        git(SITCOM,"status","--porcelain=v1","--untracked-files=no"),
        git(SITCOM,"diff","--numstat","HEAD","--","posterior_sample.py"))
    state["unused_cli_sha256"]=sha(SITCOM/"posterior_sample.py")
    check(state==mf["native_checkout_provenance"] and
          external_checkpoint_provenance(SITCOM/"checkpoint/ffhq256.pt")
          ==mf["native_checkpoint_provenance"] and
          settings(json.loads(safe(A6/"configs/b22/b22_2_overnight.json").read_text()))
          ==mf["config"],"SOURCE_DRIFT")
    all_groups=mf["b22_groups"]+mf["b24_groups"]
    check(len(all_groups)==1060 and 0<=args.index<1060,"TASK_INDEX")
    task=all_groups[args.index]
    rows=task["rows"]
    check(task["new_roots"]== (12 if args.index<100 else 1) and
          len(rows)== (16 if args.index<100 else 1),"TASK_SHAPE")
    if args.index<100:
        check([r["rep"] for r in rows]==list(range(16)) and
              len({r["image_id"] for r in rows})==1 and
              all(r["rng_mode"]=="sequential_post_model" and r["seed"]==43
                  for r in rows),"B22_NATIVE_RNG")
    else:
        check(4<=rows[0]["rep"]<=15 and
              rows[0]["rng_mode"]=="independent_model_seed" and
              0<=rows[0]["seed"]<2**32,"B24_NATIVE_SEED")
    for r in rows:
        check(sha(r["measurement_path"])==r["measurement_sha256"],"Y_HASH")
    out=run/"gpu"/task["task_id"]
    check(out.parent==run/"gpu" and not out.is_symlink() and
          (args.preflight_only or not out.exists()),"OUT_FRESH")
    if not args.preflight_only:
        admit=json.loads(safe(run/"B16_PRODUCTION_ADMISSION.json").read_text())
        check(admit["status"]=="B16_PRODUCTION_ADMITTED_400_REPLAY_PASS" and
              admit["head"]==args.expected_head and
              admit["manifest_sha256"]==pre["manifest_sha256"] and
              admit["gate_manifest_sha256"]==mf["gate_manifest_sha256"] and
              admit["gate_score_status"]=="XP_A10_B16_GATE_PASS" and
              admit["gate_n_pass"]==400 and
              admit["no_new_outcomes_opened"] is True,"ADMISSION")
    return run,mf,task,out,pre

def main(args):
    run,mf,task,out,pre=preflight(args)
    if args.preflight_only:
        print("B16_PROD_GPU_PREFLIGHT_PASS",args.index,flush=True)
        return 0
    native.PREP_HEAD=args.expected_head
    native.MANIFEST_SHA=pre["manifest_sha256"]
    mode="b22" if args.index<100 else "b24"
    worker_args=SimpleNamespace(
        mode=mode,index=(4 if mode=="b24" else None),
        expected_head=args.expected_head,gpu=args.gpu,out=str(out),
        skip_prefix_n=(4 if mode=="b22" else 0))
    check(mf["native_source_head"]==SITCOM_HEAD,"NATIVE_SOURCE_HEAD")
    rc=native.generate(worker_args,mf,task["rows"])
    check(rc==0,"NATIVE_RETURN")
    print("B16_PRODUCTION_TASK_PASS",
          task["task_id"],"new_roots",task["new_roots"],
          "no_new_labels_read",flush=True)
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--index",type=int,required=True)
    ap.add_argument("--gpu",type=int,required=True)
    ap.add_argument("--preflight-only",action="store_true")
    try:sys.exit(main(ap.parse_args()))
    except Exception as err:
        traceback.print_exc()
        print("ACTION NEEDED: B16_PRODUCTION_GPU "+str(err),flush=True)
        sys.exit(8)
