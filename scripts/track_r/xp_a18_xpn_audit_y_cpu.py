#!/usr/bin/env python3
"""A16 XP-N independent CPU-only GT/y identity audit on all 100 new ids.

No untrusted JPEG decoding, candidate-generation or GT-based selection.
Independently recomputes D1's operator/noise and canonical GT tensor SHA
from isolated 256² .pt output, never edits frozen source or D1.
"""
from __future__ import annotations
import argparse,hashlib,math,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write
from scripts.track_r.xp_a18_xpn_image_inventory_cpu import check_d1,EXPECTED
from prdiffusion.track_r.physics import magnitude,rho_signed

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A18_XPN_AUDIT_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *z:subprocess.check_output(
   ["git","-C",str(wt),*z],text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),
      "A18_XPN_AUDIT_EXECUTOR_HEAD")
 check_d1()
 out=Path(a.run)
 gate(out.parent==BASE
      and out.name.startswith("xp_a18_xpn_measurements_")
      and out.is_dir() and not out.is_symlink(),
      "A18_XPN_AUDIT_RUN_ROOT")
 rec=jread(out/"XP_N_GT_Y_PREP_RECEIPT.json")
 ym=out/"XP_N_IMAGENET100_Y_MANIFEST.json"
 gt=out/"XP_N_GT_ONLY_CPU.json"
 gate(rec["status"]=="A18_XPN_GT_Y_ISOLATED_CPU_PREP_PASS_IDENTITY_REAUDIT_PENDING"
      and rec["expected_head"]==a.source_head
      and rec["y_manifest_sha256"]==sha(ym)
      and rec["gt_only_manifest_sha256"]==sha(gt)
      and rec["isolated_decode_python_I"] is True
      and rec["D1_replay_required_before_XPN_generation"] is True,
      "A18_XPN_AUDIT_SOURCE_FROZEN")
 ys=jread(ym);gg=jread(gt)
 gate(ys["status"]=="XP_N_IMAGENET100_SIGNED_Y_FROZEN"
      and gg["status"]=="XP_N_GT_ONLY_OFFLINE_NOT_FOR_GENERATION"
      and len(ys["rows"])==len(gg["rows"])==100
      and ys["gt_paths_supplied"] is False
      and set(r["basename"] for r in ys["rows"])==set(EXPECTED),
      "A18_XPN_AUDIT_100_ROWS")
 import torch
 torch.set_num_threads(4)
 gby={r["basename"]:r for r in gg["rows"]}
 worst=0.;stats=[]
 for r in ys["rows"]:
  name=r["basename"]
  g=gby[name]
  gate(g["source_sha256"]==r["source_image_sha256"]
       and g["measurement_sha256"]==r["measurement_sha256"]
       and g["seed"]==r["seed"]
       and g["source_path"].startswith("/egr/research-pac/huang248/data/imagenet/val_raw/")
       and g["gt_tensor_shape"]==[1,3,256,256]
       and r["sigma"]==.05
       and g["sigma"]==.05,
       "A18_XPN_AUDIT_SRC_PAIR_"+name)
  gp=Path(g["gt_path"]);yp=Path(r["measurement_path"])
  gate(gp.parent==out/"gt_only" and
       yp.parent==out/"measurements"
       and not gp.is_symlink() and not yp.is_symlink()
       and sha(gp)==g["gt_file_sha256"]
       and sha(yp)==r["measurement_sha256"],
       "A18_XPN_AUDIT_TENSOR_PATH_SHA_"+name)
  x=torch.load(gp,map_location="cpu",weights_only=True)["gt_model_range"]
  y=torch.load(yp,map_location="cpu",weights_only=True)["measurement"]
  gate(x.shape==(1,3,256,256) and y.shape==(1,3,384,384)
       and x.dtype==y.dtype==torch.float32
       and bool(torch.isfinite(x).all())
       and bool(torch.isfinite(y).all()),
       "A18_XPN_AUDIT_TENSOR_SHAPE_"+name)
  calc=hashlib.sha256(x.contiguous().numpy().tobytes(order="C")).hexdigest()
  gate(calc==g["gt_tensor_sha256"],
       "A18_XPN_AUDIT_GT_CANONICAL_SHA_"+name)
  prefix="XPMEAS|imagenet_n|"+name
  seed=int.from_bytes(hashlib.sha256(prefix.encode()).digest()[:8],"big")%(2**63)
  gate(seed==r["seed"],"A18_XPN_AUDIT_SEED_"+name)
  pure=magnitude(x)
  gen=torch.Generator(device="cpu").manual_seed(seed)
  noise=torch.randn(pure.shape,generator=gen,dtype=torch.float32)
  expected=(pure+.05*noise).contiguous()
  err=float((expected-y).double().square().mean().sqrt())
  normalized=float(((y-pure)/.05).double().square().mean().sqrt())
  rho=float(rho_signed(x,y)[0])
  gate(math.isfinite(err) and err<=1e-4
       and abs(normalized-1)<.0043 and .0490<rho<.0510,
       "A18_XPN_AUDIT_SIGNED_Y_IDENTITY_"+name)
  worst=max(worst,err)
  stats.append(dict(image_id=r["image_id"],
                    gt_tensor_sha256=calc,
                    measurement_sha256=r["measurement_sha256"],
                    identity_rms=err,normalized_noise_rms=normalized))
 gate(len(stats)==100 and
      len({r["gt_tensor_sha256"] for r in stats})==100,
      "A18_XPN_AUDIT_100_UNIQUE_GT")
 f=out/"XP_N_GT_Y_INDEPENDENT_AUDIT.json"
 gate(not f.exists() and not f.is_symlink(),
      "A18_XPN_AUDIT_ALREADY_EXISTS")
 write(f,dict(status="A18_XPN_GT_Y_INDEPENDENT_ALL100_PASS",
  expected_head=a.expected_head,source_preparer_head=a.source_head,
  n_images=100,source_y_manifest_sha256=sha(ym),
  gt_only_manifest_sha256=sha(gt),source_prep_receipt_sha256=sha(
   out/"XP_N_GT_Y_PREP_RECEIPT.json"),
  GT_tensor_sha256_verified_all100=True,
  signed_y_reconstituted_all100=True,
  worst_identity_rms=worst,identity_tolerance_rms=1e-4,
  source_cohort_disjoint_from_D1=True,
  no_reconstruction_generated=True,GT_not_supplied_to_GPU=True,
  no_candidate_outcomes_read=True,
  D1_two_root_replay_still_required=True,
  rows=stats))
 print("A18_XPN_GT_Y_INDEPENDENT_ALL100_PASS",
       "WORST_RMS",worst,"AUDIT_SHA",sha(f),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--source-head",required=True)
 p.add_argument("--run",required=True)
 try:main(p.parse_args())
 except Exception as exc:
  traceback.print_exc()
  print("ACTION NEEDED: A18_XPN_GT_Y_INDEPENDENT_AUDIT",exc,flush=True)
  sys.exit(8)
