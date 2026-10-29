#!/usr/bin/env python3
"""A12 G7 / XP-C frozen GT orientation CLIP diagnostics, CPU only.

280 exposed GT images; uses frozen G1/G2 and N2 face prompts.
Does NOT claim to replace the separate prior 2-forward, S_mid or
UnivFD orientation policies; those remain explicit stage obligations.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback,importlib
from pathlib import Path
from PIL import Image
import numpy as np
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a8_d1p1_clip_cpu import (
 G1_UP,G1_INV,G2_UP,G2_INV,model_source)
from scripts.track_r.r0n_2_clip_orientation import PROMPTS_UP,PROMPTS_INV
from scripts.track_r.r0_2_g0_cpu import gt_source
from scripts.track_r.xp_c_c2_prepare_cpu import safe as safe_gt
from prdiffusion.track_r.physics import to_rgb8
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def unit(x):
 return x/x.norm(dim=-1,keepdim=True)
def prompt_pair(model,clip,torch,up,down):
 from prdiffusion.track_r.r0n_clip_compat import encode_text_univfd_tuple_compat
 t=unit(encode_text_univfd_tuple_compat(
    model,clip.tokenize(list(up+down)).to("cpu")).float())
 u=unit(t[:len(up)].mean(0,keepdim=True))[0]
 d=unit(t[len(up):].mean(0,keepdim=True))[0]
 return u,d
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *args:subprocess.check_output(
     ["git","-C",str(wt),*args],text=True).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"SOURCE")
 root=Path(a.freeze)
 gate(root.parent==BASE and root.name.startswith("xp_c_c2_")
      and root.is_dir(),"C_FROZEN_ROOT")
 rec=json.loads((root/"XP_C_C2_PREP.json").read_text())
 c=root/"XP_C_GT_IMAGES.json"
 gate(rec["status"]=="XP_C_C2_PREP_PASS"
      and sha(c)==rec["C_manifest_sha256"],"C_INPUT_SHA")
 images=json.loads(c.read_text())["images"]
 gate(len(images)==280,"C_280")
 outf=Path(a.out)
 gate(outf.parent==BASE and outf.name.startswith("xp_a12_g7_clip_")
      and not outf.exists() and not outf.is_symlink(),"OUTPUT")
 clipfile,native,nativehead=model_source()
 import torch
 torch.set_num_threads(4)
 sys.path.insert(0,str(native))
 clip=importlib.import_module("models.clip")
 model,preprocess=clip.load(str(clipfile),device="cpu",jit=False)
 model.eval()
 with torch.inference_mode():
  prompts={
   "g1":prompt_pair(model,clip,torch,G1_UP,G1_INV),
   "g2":prompt_pair(model,clip,torch,G2_UP,G2_INV),
   "face":prompt_pair(model,clip,torch,PROMPTS_UP,PROMPTS_INV)}
  rows=[]
  for i,entry in enumerate(images):
   gt=safe_gt(entry["source_path"])
   gate(sha(gt)==entry["source_sha256"],"IMAGE_SHA")
   if entry["dataset"]=="ffhq":
    x=gt_source(dict(gt_path=str(gt),gt_format=entry["gt_format"]))
    im=Image.fromarray(to_rgb8(x)[0].permute(1,2,0).numpy(),"RGB")
   else:
    with Image.open(gt) as img:
     from torchvision.transforms import v2 as T
     x=T.Compose([T.ToImage(),T.ToDtype(torch.float32,scale=True),
           T.Resize(256),T.CenterCrop(256)])(img.convert("RGB"))
     im=Image.fromarray(torch.round(x.clamp(0,1)*255).
           byte().permute(1,2,0).numpy(),"RGB")
   inv=Image.fromarray(np.rot90(np.asarray(im),2).copy(),"RGB")
   e=unit(model.encode_image(
      torch.stack([preprocess(im),preprocess(inv)])).float())
   answer=dict(dataset=entry["dataset"],basename=entry["basename"],
               original_source_sha256=entry["source_sha256"],
               diagnostic_gt_only=True)
   for policy,(u,d) in prompts.items():
    if policy=="face" and entry["dataset"]!="ffhq":continue
    score=e@u-e@d
    margin=float(score[1]-score[0])
    gate(math.isfinite(margin),"CLIP_MARGIN_FINITE")
    answer[policy+"_margin"]=format(margin,".10g")
    answer[policy+"_orientation"]=("111" if margin>0 else "000")
   rows.append(answer)
   if (i+1)%40==0:print("A12_G7_CLIP_GT",i+1,"/280",flush=True)
 gate(len(rows)==280 and sum(x["dataset"]=="ffhq" for x in rows)==180,
      "CLIP_280_COMPLETE")
 outf.mkdir(mode=0o700)
 csvpath=outf/"XP_A12_G7_C_CLIP_GT_ORIENTATION.csv"
 names=list(rows[0])
 # ImageNet entries don't carry a face prompt; dict writer tolerates missing keys.
 with csvpath.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=names)
  w.writeheader();w.writerows(rows)
 receipt=dict(status="XP_A12_G7_GT_CLIP_G1_G2_FACE_CPU_PASS",
   n_images=280,n_ffhq=180,n_imagenet=100,
   clip_native_univfd_head=nativehead,clip_checkpoint_sha256=sha(clipfile),
   input_manifest_sha256=sha(c),csv_sha256=sha(csvpath),
   g1_prompts=[list(G1_UP),list(G1_INV)],
   g2_prompts=[list(G2_UP),list(G2_INV)],
   face_prompts=[list(PROMPTS_UP),list(PROMPTS_INV)],
   no_GPU=True,gt_only_diagnostics=True,
   OTHER_G7_POLICIES_PENDING=["PRIOR_2_FORWARD","PRIOR_S_MID_GLOBAL2",
                             "PRIOR_S_MID_ALL8","UNIVFD","C2_DENOISER"])
 with (outf/"XP_A12_G7_C_CLIP_RECEIPT.json").open("x") as f:
  json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
 print("A12_G7_CLIP_GT_280_CPU_PASS",sha(csvpath),flush=True)
 print("G7_CLIP_OUT",outf,flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--freeze",required=True)
 ap.add_argument("--out",required=True)
 try:main(ap.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A12_G7_C_CLIP",e,flush=True);sys.exit(8)
