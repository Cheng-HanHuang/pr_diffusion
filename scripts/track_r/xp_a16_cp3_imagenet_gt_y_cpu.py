#!/usr/bin/env python3
"""A16 CP3 ImageNet GT/y exact reconstitution on all 100 frozen images.

Recomputes the exact seeded measurement from the scorer's native
ToTensor->Resize->CenterCrop GT, with R0 frozen operator and SHA checks.
CPU-only, no generation of candidate reconstructions, no GT for policy.
"""
from __future__ import annotations
import argparse,math,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write,safe
from scripts.track_r.xp_t_sitcom_prepare_cpu import IMAGENET,COMMON_Y_SHA
from scripts.track_r.xp_t_imagenet_prepare_cpu import seed
from prdiffusion.track_r.physics import magnitude

def main(a):
 import torch,torchvision
 from PIL import Image
 from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP3_GT_Y_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *args:subprocess.check_output(["git","-C",str(wt),*args],
                                         text=True,timeout=60).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"CP3_GT_Y_EXECUTOR_HEAD")
 torch.set_num_threads(4)
 sources=jread(IMAGENET/"XP_IMAGENET_GT_ONLY_CPU.json")
 measurements=IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
 gate(sha(measurements)==COMMON_Y_SHA,"CP3_GT_Y_FROZEN_MANIFEST_SHA")
 yman=jread(measurements)
 gate(sources["status"]=="GT_ONLY_OFFLINE_NOT_FOR_GENERATION"
      and yman["status"]=="XP_IMAGENET100_Y_FROZEN"
      and len(sources["rows"])==len(yman["rows"])==100,
      "CP3_GT_Y_100_IMAGES")
 sc={r["basename"]:r for r in sources["rows"]}
 ys={r["basename"]:r for r in yman["rows"]}
 gate(len(sc)==len(ys)==100 and set(sc)==set(ys),"CP3_GT_Y_IDS")
 tf=Compose([ToTensor(),Resize(256),CenterCrop(256)])
 worst=(0.,"")
 rows=[]
 with torch.inference_mode():
  for iid in sorted(sc):
   g=sc[iid];r=ys[iid]
   gate(sha(safe(g["source_path"]))==g["source_sha256"]
        and g["source_sha256"]==r["source_image_sha256"]
        and r["seed"]==seed("XPMEAS|imagenet",iid)
        and float(r["sigma"])==.05
        and sha(safe(r["measurement_path"]))==r["measurement_sha256"],
        "CP3_GT_Y_IMAGE_IDENTITY_"+iid)
   with Image.open(safe(g["source_path"])) as im:
    x=tf(im).float()
   if tuple(x.shape)==(1,256,256):
    x=x.repeat(3,1,1)
   gate(tuple(x.shape)==(3,256,256),"CP3_GT_Y_GT_SHAPE")
   x=(x*2-1).unsqueeze(0)
   pure=magnitude(x)
   generator=torch.Generator(device="cpu").manual_seed(r["seed"])
   noise=torch.randn(pure.shape,generator=generator,dtype=torch.float32)
   expected=pure+float(r["sigma"])*noise
   obs=torch.load(safe(r["measurement_path"]),map_location="cpu",
                  weights_only=True)["measurement"].float()
   gate(obs.shape==expected.shape,"CP3_GT_Y_MEAS_SHAPE")
   err=float((obs-expected).double().square().mean().sqrt())
   noise_rms=float(((obs-pure)/float(r["sigma"])).double().square().mean().sqrt())
   gate(math.isfinite(err) and err<=1e-4 and
        math.isfinite(noise_rms) and abs(noise_rms-1)<=.0043,
        "CP3_GT_Y_EXACT_REPRODUCTION_"+iid)
   rows.append(dict(image_id=iid,reconstruction_rms=err,
                    noise_rms_normalized=noise_rms))
   if err>worst[0]:worst=(err,iid)
 gate(len(rows)==100,"CP3_GT_Y_COMPLETE_100")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a16_cp3_gt_y_")
      and not out.exists() and not out.is_symlink(),
      "CP3_GT_Y_NEW_OUTPUT")
 if a.preflight_only:
  print("CP3_IMAGENET_GT_Y_EXACT_ALL100_PREFLIGHT_PASS",
        worst[1],worst[0],flush=True)
  return
 out.mkdir(mode=0o700)
 receipt=dict(status="XP_A16_CP3_IMAGENET_GT_Y_EXACT_100_PASS",
     expected_head=a.expected_head,
     n_images=100,source_y_manifest_sha256=COMMON_Y_SHA,
     GT_score_preprocessing="ToTensor Resize(256) CenterCrop(256)",
     operator="pad64 magnitude fft, signed y",
     noise_rule="XPMEAS|imagenet|basename first8 SHA256 BE mod2**63",
     sigma=.05,exact_rms_tolerance=1e-4,
     worst_rms=worst[0],worst_basename=worst[1],
     torch_version=torch.__version__,
     torchvision_version=torchvision.__version__,
     image_level=rows,
     no_GPU=True,GT_only_for_evaluation=True,
     no_candidate_rerun=True,CP3_RETURN_AUTHORIZED=False)
 write(out/"XP_A16_CP3_IMAGENET_GT_Y_IDENTITY.json",receipt)
 print("CP3_IMAGENET_GT_Y_EXACT_ALL100_PASS",
       worst[1],worst[0],flush=True)
 print("CP3_GT_Y_RECEIPT_DIR",out,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A16_CP3_IMAGENET_GT_Y",e,flush=True)
  sys.exit(8)
