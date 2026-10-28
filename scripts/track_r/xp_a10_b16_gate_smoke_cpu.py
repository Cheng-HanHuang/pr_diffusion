#!/usr/bin/env python3
"""Read-only CPU historical PSNR/label audit for one complete B16 replay image.

Allows inspecting an interrupted replay run without mutating its manifests,
controller state or historical reference files. Never opens B16 runs >= 4.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path

from scripts.track_r.xp_a10_b16_gate_prepare_cpu import BASE,safe,sha
from scripts.track_r.xp_a10_b16_gate_gpu import image_folder

def demand(cond,name):
    if not cond:raise RuntimeError("STOP_B16_SMOKE_AUDIT_"+name)

def evaluate(run,head,index):
    demand(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    demand(run.parent==BASE and run.name.startswith("xp_a10_b16_gate_")
           and run.is_dir() and not run.is_symlink(),"RUN_PATH")
    prep=json.loads(safe(run/"XP_A10_B16_GATE_PREP.json").read_text())
    demand(prep["head"]==head and prep["n_roots"]==400,"GENERATION_HEAD")
    mf=run/"XP_A10_B16_GATE_TASKS.json"
    off=run/"XP_A10_B16_GATE_OFFLINE.json"
    demand(sha(mf)==prep["manifest_sha256"] and
           sha(off)==prep["offline_sha256"],"INPUT_HASH")
    tasks=json.loads(mf.read_text())
    references=json.loads(off.read_text())
    demand(tasks["GT_supplied"] is False and
           tasks["historical_candidate_supplied"] is False and
           len(tasks["rows"])==len(references["rows"])==400 and
           0<=index<100,"MANIFEST_SCHEMA")
    import torch
    from prdiffusion.track_r.physics import to_rgb8,from_rgb8,rho_signed
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    from scripts.track_r.r0_2_g0_cpu import gt_source,read_tensor
    from scripts.b22.b22_smoke_common import tensor_content_sha256
    torch.set_num_threads(min(4,torch.get_num_threads()))
    rows=tasks["rows"][4*index:4*index+4]
    refs=references["rows"][4*index:4*index+4]
    demand(len(rows)==4 and len({r["image_id"] for r in rows})==1,
           "FOUR_RUNS")
    gt=to_rgb8(gt_source(refs[0]).float())
    y=read_tensor(safe(rows[0]["measurement_path"]),
                  ("measurement","y","observation"),(1,3,384,384))
    records=[]
    for row,ref in zip(rows,refs):
        uid=row["uid"]
        demand(uid==ref["uid"] and row["image_id"]==ref["image_id"] and
               row["rep"]==ref["rep"] and
               row["measurement_sha256"]==ref["measurement_sha256"],
               "PAIRED_UID")
        dest=image_folder(run,row["image_id"])/("root_"+hashlib.sha256(
                 uid.encode()).hexdigest()[:24])
        proof=json.loads(safe(dest/"XP_SITCOM_ROOT_COMPLETE.json").read_text())
        demand(proof["status"]=="XP_SITCOM_REPLAY_ROOT_PASS" and
               proof["uid"]==uid and
               proof["runner_head"]==head and
               proof["manifest_sha256"]==prep["manifest_sha256"] and
               proof["measurement_sha256"]==row["measurement_sha256"] and
               proof["gt_access"] is False and
               proof["historical_access"] is False and
               sha(dest/"reconstruction.pt")==proof["reconstruction_pt_sha256"] and
               sha(dest/"reconstruction.png")==proof["reconstruction_png_sha256"],
               "ROOT_RECEIPT")
        tensor=torch.load(safe(dest/"reconstruction.pt"),
                          map_location="cpu",weights_only=True)["reconstruction"].float()
        demand(tuple(tensor.shape)==(1,3,256,256) and
               torch.isfinite(tensor).all().item(),"RECON")
        psnr=orbit_psnr_channel_mse(to_rgb8(tensor),gt)
        raw=float(psnr["000"])
        best=float(max(psnr.values()))
        if raw>=25:label="GOOD"
        elif best>=25:label="ORBIT_ERROR"
        elif float(rho_signed(from_rgb8(to_rgb8(tensor)),y)[0])<=ref["historical_tau"]:
            label="NONORBIT_CONSISTENT"
        else:label="NONORBIT_INCONSISTENT"
        delta=abs(raw-float(ref["historical_raw_psnr"]))
        exact=tensor_content_sha256(tensor)==ref["historical_tensor_sha256"]
        passed=delta<=0.1 and label==ref["historical_raw_label"]
        records.append(dict(uid=uid,run=int(row["rep"]),delta_db=delta,
                    label=label,reference_label=ref["historical_raw_label"],
                    exact_tensor=exact,passed=passed,
                    sampling_wall_s=float(proof["sampling_wall_s"])))
    demand(all(v["passed"] for v in records),"LINEAGE_MISMATCH")
    print("B16_FOUR_RUN_HISTORICAL_SMOKE_PASS",rows[0]["image_id"],
          "matched=4/4",
          "max_delta_db="+format(max(v["delta_db"] for v in records),".8g"),
          "exact="+str(sum(v["exact_tensor"] for v in records))+"/4",
          "median_sec="+format(sum(sorted(v["sampling_wall_s"] for v in records)[1:3])/2,".4f"),
          flush=True)
    return records

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--index",type=int,default=0)
    a=p.parse_args()
    try:evaluate(a.run,a.expected_head,a.index)
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: B16_SMOKE_AUDIT",str(exc),flush=True)
        sys.exit(8)
