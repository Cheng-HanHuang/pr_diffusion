#!/usr/bin/env python3
"""A22 C2 native 280-image diagnostic readout, image-cluster bootstrap.

No GPU. Reports physical-denoiser equivariance delta and delta/denom
for 9 timesteps x 2 rotations on FFHQ180 and ImageNet100.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,statistics,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
def gate(c,k):
 if not c:raise RuntimeError("STOP_A22_G7_READOUT_"+k)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 run=Path(a.run)
 gate(run.parent==BASE and run.name.startswith("xp_a12_g7_prod_a22_")
      and run.is_dir(),"RUN")
 d=json.loads((run/"XP_A22_G7_C2_FINAL_STATUS.json").read_text())
 gate(d["status"]=="A22_G7_C2_280_COMPLETE" and
      len(d["completed"])==280 and d["head" if "head" in d else "expected_head"]==
      a.expected_head,"COMPLETE_STAGE")
 grouped=defaultdict(lambda:defaultdict(list))
 for i in range(280):
  p=run/"results"/f"{i:03d}_equivariance.json"
  gate(sha(p)==d["completed"][str(i)],"FROZEN_RESULT_SHA")
  z=json.loads(p.read_text())
  gate(z["status"]=="XP_A12_G7_C2_IMAGE_EQUIVARIANCE_PASS"
       and z["n"]==36 and len(z["records"])==36,"ROW_PROOF")
  k=(z["dataset"],z["basename"])
  for r in z["records"]:
   gate(r["dataset"]==z["dataset"] and
        r["basename"]==z["basename"] and
        math.isfinite(r["delta"]) and
        math.isfinite(r["denominator"]) and
        r["denominator"]>0,"FINITE_DENOM")
   grouped[(z["dataset"],r["t_ddpm1000"],r["rotation"])][k].append(r)
 import numpy as np
 rng=np.random.default_rng(20261103)
 table=[]
 for (dataset,t,rot),im in sorted(grouped.items()):
  expected=180 if dataset=="ffhq" else 100
  gate(len(im)==expected and all(len(v)==2 for v in im.values()),
       "EXACT_TWO_NOISE_DRAWS")
  means=np.asarray([[statistics.mean(z["delta"] for z in v),
        statistics.mean(z["denominator"] for z in v)]
        for _,v in sorted(im.items())])
  delta=float(np.mean(means[:,0]))
  denominator=float(np.mean(means[:,1]))
  ratio=delta/denominator
  bootstrap=[]
  for _ in range(10000):
   picked=means[rng.integers(0,len(means),size=len(means))]
   bootstrap.append((float(picked[:,0].mean()),
       float(picked[:,0].mean()/picked[:,1].mean())))
  lo,hi=np.quantile(bootstrap,[.025,.975],axis=0)
  table.append(dict(dataset=dataset,t_ddpm1000=t,rotation=rot,
     n_images=expected,n_noise_draws=2,
     delta=delta,delta_lo95=float(lo[0]),delta_hi95=float(hi[0]),
     delta_tilde_ratio_of_means=ratio,
     delta_tilde_lo95=float(lo[1]),delta_tilde_hi95=float(hi[1])))
 gate(len(table)==36,"36_DATASET_TIME_ROTATION_ROWS")
 target=Path(a.out)
 gate(target.parent==BASE and
      target.name.startswith("xp_a30_a22_g7_report_")
      and not target.exists(),"NEW_OUTPUT")
 target.mkdir()
 csvfile=target/"XP_A22_G7_C2_EQUIVARIANCE.csv"
 keys=list(table[0])
 with csvfile.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n")
  w.writeheader();w.writerows(table)
 rec=dict(status="A22_G7_C2_IMAGE_BOOTSTRAP_CPU_COMPLETE",
  n_input_images=280,n_task_rows=10080,n_summary_rows=36,
  n_bootstrap=10000,bootstrap_seed=20261103,
  no_inverse_solver=True,GT_only_diagnostics=True,
  no_A21_10_replication=True,
  frozen_native_input_receipt_sha256=sha(run/"XP_A22_G7_C2_FINAL_STATUS.json"),
  table_sha256=sha(csvfile),source_code_head=a.expected_head)
 (target/"XP_A22_G7_C2_READOUT_RECEIPT.json").write_text(
  json.dumps(rec,sort_keys=True,indent=2)+"\n")
 print("A22_G7_C2_READOUT_PASS",target,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 for k in ("expected-head","run","out"):p.add_argument("--"+k,required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A22_G7_READOUT",e,flush=True);sys.exit(8)
