#!/usr/bin/env python3
"""A9.6 B16 400-root CPU-only historical replay gate scoring.

GT and original historical candidates are exclusively read here, AFTER the
native 400-root GPU replay is complete. New B16 indices >=4 are never opened.
A gate requires ALL 400 raw PSNR deltas <=0.1 dB and identical R0.7 labels.
"""
from __future__ import annotations
import argparse,json,math,os,sys,traceback
from pathlib import Path
import torch
from scripts.track_r.xp_a10_b16_gate_prepare_cpu import (
    BASE,WT,source_sha,sha,git,safe)
from scripts.track_r.xp_a10_b16_gate_gpu import image_folder
from scripts.track_r.r0_2_g0_cpu import gt_source,read_tensor
from prdiffusion.track_r.physics import to_rgb8,from_rgb8,rho_signed
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
from scripts.b22.b22_smoke_common import tensor_content_sha256

def check(ok,reason):
    if not ok:raise RuntimeError("STOP_B16_GATE_SCORE_"+reason)

def score(run,head):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    check(git(WT,"rev-parse","HEAD")==head and
          not git(WT,"status","--porcelain"),"HEAD")
    prep=json.loads(safe(run/"XP_A10_B16_GATE_PREP.json").read_text())
    check(prep["status"]=="XP_A10_B16_GATE_PREP_PASS" and
          prep["head"]==head and prep["n_roots"]==400,"PREP")
    mf=run/"XP_A10_B16_GATE_TASKS.json"
    off=run/"XP_A10_B16_GATE_OFFLINE.json"
    check(sha(mf)==prep["manifest_sha256"] and
          sha(off)==prep["offline_sha256"],"FROZEN_SHA")
    tasks=json.loads(mf.read_text())
    refs=json.loads(off.read_text())
    rows=tasks["rows"]
    historical=refs["rows"]
    check(len(rows)==len(historical)==400 and
          tasks["GT_supplied"] is False and
          tasks["historical_candidate_supplied"] is False and
          refs["status"]=="CPU_ONLY_NEVER_FEED_GPU","SPLIT")
    expected=set(tasks.keys())
    torch.set_num_threads(min(4,torch.get_num_threads()))
    scored=[]
    gtcache={}
    ycache={}
    for i,(row,ref) in enumerate(zip(rows,historical)):
        uid=row["uid"]
        check(uid==ref["uid"] and row["image_id"]==ref["image_id"] and
              row["rep"]==ref["rep"] and
              row["measurement_sha256"]==ref["measurement_sha256"] and
              0<=int(row["rep"])<4,"PAIRED_UID")
        dest=image_folder(run,row["image_id"])/("root_"+__import__("hashlib").sha256(
            uid.encode()).hexdigest()[:24])
        receipt=json.loads(safe(dest/"XP_SITCOM_ROOT_COMPLETE.json").read_text())
        check(receipt.get("status")=="XP_SITCOM_REPLAY_ROOT_PASS" and
              receipt.get("uid")==uid and
              receipt.get("runner_head")==head and
              receipt.get("manifest_sha256")==prep["manifest_sha256"] and
              receipt.get("measurement_sha256")==row["measurement_sha256"] and
              receipt.get("gt_access") is False and
              receipt.get("historical_access") is False and
              sha(dest/"reconstruction.pt")==receipt["reconstruction_pt_sha256"] and
              sha(dest/"reconstruction.png")==receipt["reconstruction_png_sha256"],
              "NATIVE_ROOT_RECEIPT")
        x=torch.load(safe(dest/"reconstruction.pt"),map_location="cpu",
                     weights_only=True)["reconstruction"].float()
        check(tuple(x.shape)==(1,3,256,256) and
              torch.isfinite(x).all().item(),"RECON_SHAPE")
        x8=to_rgb8(x)
        key=(row["source"],row["image_id"])
        if key not in gtcache:
            gtcache[key]=to_rgb8(gt_source(ref).float())
            ycache[key]=read_tensor(safe(row["measurement_path"]),
                                 ("measurement","y","observation"),(1,3,384,384))
        orbit=orbit_psnr_channel_mse(x8,gtcache[key])
        raw=float(orbit["000"])
        best=float(max(orbit.values()))
        residual=float(rho_signed(from_rgb8(x8),ycache[key])[0])
        if raw>=25:label="GOOD"
        elif best>=25:label="ORBIT_ERROR"
        elif residual<=float(ref["historical_tau"]):
            label="NONORBIT_CONSISTENT"
        else:label="NONORBIT_INCONSISTENT"
        delta=abs(raw-ref["historical_raw_psnr"])
        exact=(tensor_content_sha256(x)==ref["historical_tensor_sha256"])
        ok=delta<=0.1 and label==ref["historical_raw_label"]
        scored.append(dict(uid=uid,image_id=row["image_id"],rep=row["rep"],
            raw_psnr=raw,historical_raw_psnr=ref["historical_raw_psnr"],
            abs_delta_db=delta,label=label,historical_label=ref["historical_raw_label"],
            exact_tensor=exact,pass_gate=ok))
        if (i+1)%40==0:
            print("B16_GATE_SCORE_PROGRESS",i+1,"/400",flush=True)
    check(len(scored)==400,"400_SCORED")
    n_bad=sum(not r["pass_gate"] for r in scored)
    summary=dict(
        status="XP_A10_B16_GATE_PASS" if n_bad==0 else "XP_A10_B16_GATE_FAIL",
        head=head,manifest_sha256=prep["manifest_sha256"],
        n_roots=400,n_images=100,n_pass=400-n_bad,n_fail=n_bad,
        n_exact_tensors=sum(r["exact_tensor"] for r in scored),
        max_abs_delta_db=max(r["abs_delta_db"] for r in scored),
        threshold_db=0.1,
        requires_identical_R07_label=True,
        historical_runs_only=True,
        B16_new_run_labels_opened=False,
        gt_used_CPU_only=True,rows=scored)
    out=run/"XP_A10_B16_GATE_COMPARE.json"
    with out.open("x") as f:
        json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(summary["status"],"pass="+str(400-n_bad)+"/400",
          "max_delta="+str(summary["max_abs_delta_db"]),
          "exact="+str(summary["n_exact_tensors"]),flush=True)
    print("B16_GATE_SCORE_SHA="+sha(out),flush=True)
    return 0 if n_bad==0 else 8

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--expected-head",required=True)
    a=p.parse_args()
    try:
        run=Path(a.run)
        check(run.parent==BASE and run.name.startswith("xp_a10_b16_gate_"),
              "RUN_NAMESPACE")
        sys.exit(score(run,a.expected_head))
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: B16_GATE_SCORE",str(exc),flush=True)
        sys.exit(8)
