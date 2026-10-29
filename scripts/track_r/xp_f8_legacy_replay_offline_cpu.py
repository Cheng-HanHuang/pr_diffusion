#!/usr/bin/env python3
"""Offline GT-only historical DAPS replay gate. Never callable from GPU worker.

Compares native 400x5 eight historical roots with R0.7 source outcomes using
canonical RGB8 raw PSNR, same pre-frozen rho_signed threshold and raw label.
Wave1 validates first four before wave2 is admitted. Wave2 validates all eight.
"""
from __future__ import annotations
import argparse,json,math,os,sys,traceback
from pathlib import Path
import torch
from scripts.track_r.xp_f8_legacy_replay_gpu import (
    BASE,PREP,sha,read_json,fail,git,WT,FROZEN_MANIFEST_SHA)
from scripts.track_r.r0_2_g0_cpu import (
    gt_source,candidate_source,read_tensor)
from prdiffusion.track_r.physics import (
    load_png_rgb8,to_rgb8,from_rgb8,rho_signed)
from prdiffusion.track_r.r0_7_stats import (
    orbit_psnr_channel_mse,label_candidate,GOOD)

def save_x(path,obj):
    with path.open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--wave",type=int,choices=(1,2),required=True)
    a=p.parse_args()
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")=="","OFFLINE_CPU_ONLY")
    fail(git(WT,"rev-parse","HEAD")==a.expected_head
         and not git(WT,"status","--porcelain"),"OFFLINE_WORKTREE")
    root=Path(a.run)
    fail(root.parent==BASE and root.name.startswith("xp_f8_legacy_replay_")
         and root.is_dir(),"OFFLINE_ROOT")
    fail(sha(PREP/"XP_F8_REPLAY_INPUTS_GT_FREE.json")==FROZEN_MANIFEST_SHA,
         "INPUT_SHA")
    model=read_json(PREP/"XP_F8_REPLAY_INPUTS_GT_FREE.json")
    refs=read_json(PREP/"XP_F8_REPLAY_OFFLINE_ONLY.json")
    offline={x["uid"]:x for x in refs["rows"]}
    rows=model["rows"]
    need_indices=[0,1,2,3] if a.wave==1 else list(range(8))
    checks=[]
    for i in need_indices:
        row=rows[i]
        prev=offline[row["uid"]]
        native_dir=root/f"replay_{i:02d}"
        receipt=native_dir/"XP_F8_LEGACY_REPLAY_NATIVE_ROOT.json"
        fail(receipt.is_file(),"NATIVE_RECEIPT_MISSING_"+str(i))
        r=read_json(receipt)
        fail(r["status"]=="XP_F8_LEGACY_REPLAY_NATIVE_ROOT_PASS"
             and r["uid"]==row["uid"] and
             r["measurement_sha256"]==row["measurement_sha256"] and
             r["replay_input_manifest_sha256"]==FROZEN_MANIFEST_SHA
             and sha(r["sample_path"])==r["sample_sha256"],
             "NATIVE_RECEIPT_"+str(i))
        # Ground truth is accessed NOW, on CPU, never in the native child.
        source=dict(gt_path=prev["gt_path"],gt_format=prev["gt_format"],
                    candidate_path=prev["historical_candidate_path"],
                    candidate_format=prev["historical_candidate_format"])
        gt=to_rgb8(gt_source(source))
        historical=to_rgb8(candidate_source(source))
        native=load_png_rgb8(r["sample_path"])
        equal=bool(torch.equal(native,historical))
        orbital=orbit_psnr_channel_mse(native,gt)
        raw=float(orbital["000"])
        orbit=max(orbital.values())
        meas=read_tensor(row["measurement_path"],
             ("measurement","y","observation"),(1,3,384,384))
        rho=float(rho_signed(from_rgb8(native),meas)[0])
        cls=label_candidate(dict(psnr_raw=raw,
             psnr_orbit_best=orbit,rho_signed_raw=rho),prev["tau"])
        original=str(prev["raw_label"])
        drift=raw-float(prev["raw_psnr"])
        # Both must hold for the tolerance path, even if the PNG is
        # byte-for-byte pixel identical (record pixel identity separately).
        matched=abs(drift)<=0.1+1e-10 and cls==original
        checks.append(dict(index=i,source=row["source"],image_id=row["image_id"],
           rep=row["rep"],pixel_identical=equal,
           original_raw_psnr=float(prev["raw_psnr"]),replayed_raw_psnr=raw,
           psnr_difference_db=drift,original_label=original,
           replayed_label=cls,label_identical=cls==original,
           raw_psnr_tolerance_pass=abs(drift)<=0.1+1e-10,
           replay_pass=matched,replayed_rho_signed=rho))
        print("REPLAY_MATCH" if matched else "REPLAY_MISMATCH",
              i,"pixel_equal",equal,"dB",round(drift,5),
              "label_equal",cls==original,flush=True)
    passed=all(x["replay_pass"] for x in checks)
    name=f"XP_F8_LEGACY_REPLAY_WAVE{a.wave}_OFFLINE.json"
    status=(f"XP_F8_LEGACY_WAVE{a.wave}_OFFLINE_PASS"
            if passed else f"STOP_F8_LEGACY_WAVE{a.wave}_OFFLINE_FAILURE")
    save_x(root/name,dict(
        status=status,verification_method="canonical_RGB8_PSNR0p1_and_R07_label",
        n_checked=len(checks),n_pass=sum(x["replay_pass"] for x in checks),
        n_pixel_identical=sum(x["pixel_identical"] for x in checks),
        comparisons=checks,raw_psnr_tol_db=0.1,
        historical_data_opened_by_gpu=False,
        gt_opened_by_gpu=False,no_future_B16_F8_outcomes_opened=True))
    fail(passed,"REPLAY_LABEL_OR_PSNR_MISMATCH")
    print("XP_F8_LEGACY_REPLAY_OFFLINE_PASS",a.wave,
          "checked",len(checks),flush=True)
    if a.wave==2:
        print("F8_NATIVE_8_ROOT_GATE_PASS_F8_920_CAN_BE_PREPARED",
              flush=True)
if __name__=="__main__":
    try:main()
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_F8_LEGACY_OFFLINE "+str(e),flush=True)
        sys.exit(8)
