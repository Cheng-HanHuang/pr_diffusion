#!/usr/bin/env python3
"""A16 CP3 published-protocol piq VGG-LPIPS + SSIM on saved native tensors.

GT used only in CPU evaluation; original model-range float tensors and
their native root SHA are checked. Uses frozen piq.LPIPS(replace_pooling=True).
No model download, no solver rerun, no CLIP rerun, and no CP3 release.
"""
from __future__ import annotations
import argparse,csv,json,math,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,csvload,csvwrite,write,safe
from scripts.track_r.xp_a15_cp3_protocol_cpu import verify
from scripts.track_r.xp_a15_cp3_score_cpu import common_sources,G3_MANIFEST_SHA
from scripts.track_r.r0_2_g0_cpu import gt_source
from prdiffusion.track_r.physics import to_rgb8,load_png_rgb8

COLS=("image_id","cohort","run","uid","candidate_pt_sha256",
      "SSIM_raw","SSIM_111","LPIPS_VGG_raw","LPIPS_VGG_111")
PANELS=("G2_0.02","G2_0.10","G4","G5","G3")
def original_tensor(row,stage,cache):
 import torch
 png=safe(row["reconstruction_path"])
 gate(sha(png)==row["reconstruction_sha256"],"CP3_PIQ_PNG_HASH")
 root=png.parent
 if stage=="G3":
  pt=safe(root/"particles.pt")
  bag=cache.get(str(pt))
  if bag is None:
   r=jread(root/"XP_A12_G3_BAG_COMPLETE.json")
   gate(r["status"]=="XP_A12_G3_NATIVE_FOUR_PARTICLE_BAG_PASS"
        and r["manifest_sha256"]==G3_MANIFEST_SHA
        and r["reported_metric_rule"]=="MEAN_OF_FOUR_PARTICLE_METRICS_NOT_PIXEL_MEAN"
        and sha(pt)==r["particles_pt_sha256"],
        "CP3_PIQ_PC_NATIVE_SOURCE")
   bag=torch.load(pt,map_location="cpu",weights_only=True)["particles"].float()
   gate(tuple(bag.shape)==(1,4,3,256,256) and bool(torch.isfinite(bag).all()),
        "CP3_PIQ_PC_NATIVE_SHAPE")
   cache[str(pt)]=bag
  raw=bag[:,int(row["run"])]
 else:
  pt=safe(root/"reconstruction.pt")
  recname=("R0_FAST_ROOT_COMPLETE.json" if stage.startswith("G2")
           else "XP_T_DAPS4K_ROOT_COMPLETE.json")
  rec=jread(root/recname)
  gate(sha(pt)==rec["reconstruction_pt_sha256"],
       "CP3_PIQ_NATIVE_TENSOR_HASH")
  raw=torch.load(pt,map_location="cpu",weights_only=True)["reconstruction"].float()
 gate(tuple(raw.shape)==(1,3,256,256)
      and bool(torch.isfinite(raw).all()),"CP3_PIQ_NATIVE_SHAPE")
 # Native PNG is derived from exactly the same clamped model-range
 # tensor. Verify round-to-even quantization identity, never replace
 # float tensors by saved PNGs for the published-protocol metric.
 gate(torch.equal(to_rgb8(raw),load_png_rgb8(png)),
      "CP3_PIQ_FLOAT_PNG_IDENTITY")
 return raw
def true_image(iid,stage,ff,gin):
 import torch
 from PIL import Image
 from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
 if stage=="G5":
  g=gin[iid]
  gate(sha(safe(g["source_path"]))==g["source_sha256"],
       "CP3_PIQ_IMAGENET_GT_HASH")
  with Image.open(safe(g["source_path"])) as im:
   z=Compose([ToTensor(),Resize(256),CenterCrop(256)])(
        im.convert("RGB")).unsqueeze(0)*2-1
 else:
  g=ff[iid]
  gate(safe(g["gt_path"]).is_file(),"CP3_PIQ_FFHQ_GT_SOURCE")
  z=gt_source(dict(gt_path=g["gt_path"],gt_format=g["gt_format"])).float()
 gate(tuple(z.shape)==(1,3,256,256) and bool(torch.isfinite(z).all()),
      "CP3_PIQ_GT_FINITE")
 return z
def weights_preflight():
 import torch,torchvision,piq
 p=Path(torch.hub.get_dir())/"checkpoints"
 need=(p/"lpips_weights.pt",p/"vgg16-397923af.pth")
 missing=[str(f) for f in need if not f.is_file()]
 gate(not missing,"CP3_PIQ_PRETRAINED_WEIGHTS_MISSING__"+
      "____".join(x.replace("/","_") for x in missing))
 return dict(torch_version=torch.__version__,
   torchvision_version=torchvision.__version__,
   piq_version=getattr(piq,"__version__","unknown"),
   pretrained_lpips_weights_sha256=sha(need[0]),
   pretrained_vgg16_weights_sha256=sha(need[1]),
   torch_hub_checkpoints_dir=str(p),device="cpu")
def summaries(rows,stage):
 # DAPS's published metrics select the best candidate per metric:
 # max PSNR/SSIM, min LPIPS. Never equate this with a GT-free selector.
 from statistics import mean
 by=defaultdict(list)
 for r in rows:by[(r["cohort"],r["image_id"])].append(r)
 ag=defaultdict(list)
 for (cohort,iid),group in by.items():
  g=sorted(group,key=lambda z:z["run"])
  gate([r["run"] for r in g]==list(range(4 if stage=="G3" else 8)),
       "CP3_PIQ_GROUP_INDEX")
  if stage=="G3":
   for met,best in (("SSIM_raw","mean"),
                    ("LPIPS_VGG_raw","mean")):
    ag[(cohort,"PC_NATIVE_METRIC_MEAN_4",met)].append(
     sum(x[met] for x in g)/4)
  else:
   for k in (4,8):
    for met,opt in (("SSIM_raw","max"),("LPIPS_VGG_raw","min")):
     choose=max if opt=="max" else min
     ag[(cohort,f"DAPS_BEST_OF_{k}_ORACLE_PER_METRIC",met)].append(
      choose(x[met] for x in g[:k]))
 # Numerical summaries are descriptive; do not mix cohorts.
 return [dict(cohort=k[0],rule=k[1],metric=k[2],
              n_images=len(v),mean=mean(v))
         for k,v in sorted(ag.items())]
def main(a):
 import torch,piq
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP3_PIQ_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *xs:subprocess.check_output(["git","-C",str(wt),*xs],
            text=True,timeout=60).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"CP3_PIQ_EXECUTOR_HEAD")
 torch.set_num_threads(4)
 root=BASE/("xp_a15_cp3_"+a.stage.replace(".","p")+"_"+a.stamp)
 rec,sp,cp,rp,strings,clips,runs=verify(root,a.stage)
 info=weights_preflight()
 ff,gin,yy=common_sources()
 gate(set(r["image_id"] for r in runs) <=
      (set(gin) if a.stage=="G5" else set(ff)),
      "CP3_PIQ_GT_IMAGE_KEYS")
 if a.preflight_only:
  print("CP3_PIQ_PREFLIGHT_PASS",a.stage,
        "n_candidates",len(runs),"weights_cached",True,flush=True)
  print("CP3_PIQ_CACHED_WEIGHTS_SHA",info["pretrained_lpips_weights_sha256"],
        info["pretrained_vgg16_weights_sha256"],flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a16_cp3_piq_"+a.stage.replace(".","p")+"_")
      and not out.exists() and not out.is_symlink(),
      "CP3_PIQ_NEW_OUTPUT")
 # Force cached, pinned weights only; this must not download.
 import torch.hub
 original_loader=torch.hub.load_state_dict_from_url
 def cached_only(url,*args,**kwargs):
  name=kwargs.get("file_name") or url.rsplit("/",1)[-1]
  expected=Path(torch.hub.get_dir())/"checkpoints"/name
  gate(expected.is_file(),"CP3_PIQ_CACHED_URL_MISSING")
  return torch.load(expected,map_location=kwargs.get("map_location","cpu"),
                    weights_only=True)
 torch.hub.load_state_dict_from_url=cached_only
 try:
  metric=piq.LPIPS(replace_pooling=True,reduction="none").eval().cpu()
 finally:
  torch.hub.load_state_dict_from_url=original_loader
 out.mkdir(mode=0o700)
 by=defaultdict(list)
 for r in runs:by[r["image_id"]].append(r)
 values=[]; tensor_cache={}
 with torch.inference_mode():
  for j,(iid,group) in enumerate(sorted(by.items())):
   g=true_image(iid,a.stage,ff,gin)
   ordered=sorted(group,key=lambda r:int(r["run"]))
   batch=torch.cat([original_tensor(r,a.stage,tensor_cache)
                    for r in ordered],dim=0)
   gate(batch.shape[0]==rec["n_runs_per_image"],
        "CP3_PIQ_NATIVE_BATCH")
   gt=(g.clamp(-1,1)+1)/2
   raw=(batch.clamp(-1,1)+1)/2
   ref=gt.expand_as(raw)
   flip=torch.flip(raw,(-2,-1))
   s0=piq.ssim(raw,ref,data_range=1.,reduction="none").cpu().tolist()
   s1=piq.ssim(flip,ref,data_range=1.,reduction="none").cpu().tolist()
   p0=metric(raw,ref).cpu().tolist()
   p1=metric(flip,ref).cpu().tolist()
   for k,row in enumerate(ordered):
    vals=(s0[k],s1[k],p0[k],p1[k])
    gate(all(math.isfinite(v) for v in vals),
         "CP3_PIQ_NUMERIC_FINITE")
    values.append(dict(image_id=iid,cohort=row["cohort"],
      run=int(row["run"]),uid=row["uid"],
      candidate_pt_sha256=("G3_BAG_PT" if a.stage=="G3" else
        sha(Path(row["reconstruction_path"]).with_suffix(".pt"))),
      SSIM_raw=vals[0],SSIM_111=vals[1],
      LPIPS_VGG_raw=vals[2],LPIPS_VGG_111=vals[3]))
   tensor_cache.clear()
   if (j+1)%10==0:
    print("CP3_PIQ_IMAGES_SCORED",a.stage,j+1,"/",len(by),flush=True)
 gate(len(values)==rec["n_candidates"],"CP3_PIQ_FULL_NATIVE_GRID")
 path=out/"XP_A16_CP3_NATIVE_PIQ_METRICS.csv"
 csvwrite(path,values,COLS)
 result=dict(status="XP_A16_CP3_PUBLISHED_PROTOCOL_PIQ_CPU_METRICS_COMPLETE_NOT_CP3",
    expected_head=a.expected_head,stage=a.stage,
    images=len(by),candidates=len(values),
    input_stage_scoring_receipt_sha256=sha(root/"XP_A15_CP3_STAGE_SCORING_RECEIPT.json"),
    metric_definition="piq.LPIPS(replace_pooling=True) VGG16 pretrained; piq.ssim data_range 1; raw model float, clamped to [0,1], no PNG quantization",
    exact_float_native_pt=True,zero_shot_clip_not_rerun=True,
    GT_used_for_evaluation_only=True,device="cpu",no_GPU=True,
    pretrained=info,mean_per_metric=summaries(values,a.stage),
    metrics_sha256=sha(path),CP3_RETURN_AUTHORIZED=False)
 write(out/"XP_A16_CP3_NATIVE_PIQ_METRICS_RECEIPT.json",result)
 print("CP3_NATIVE_PIQ_METRICS_COMPLETE",a.stage,
       len(values),"GT_ONLY_NOT_PUBLISHED",flush=True)
 print("CP3_PIQ_METRICS_DIR",out,flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--stage",choices=PANELS,required=True)
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--stamp",default="20261030T210259Z")
 ap.add_argument("--out")
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A16_CP3_PIQ_METRICS",e,flush=True)
  sys.exit(8)
