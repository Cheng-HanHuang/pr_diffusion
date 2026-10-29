#!/usr/bin/env python3
"""A13 F8 N2 face orientation CLIP, 1440 candidates, CPU only.

Frozen N2 prompts, frozen weights and torch compatibility. Uses
only reconstructed images, never GT to choose orientation. The
post-choice GOOD indicator is retained solely for held-out evaluation.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
from PIL import Image
import numpy as np
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a8_d1p1_clip_cpu import model_source
from scripts.track_r.r0n_2_clip_orientation import PROMPTS_UP,PROMPTS_INV,CLIP_SHA
from scripts.track_r.r0_4_score_gpu import candidate_tensor
FIELDS=("image_id","run","label","f_choice","f_margin","f_chosen_good","cohort")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def write(path,obj):
 with Path(path).open("x") as f:
  json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
             text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"SOURCE")
 score=Path(a.score_dir)
 gate(score.parent==BASE and score.name.startswith("xp_a13_f8_score_"),
      "SCORE_PATH")
 rec=json.loads((score/"XP_A13_F8_SCORING_RECEIPT.json").read_text())
 file=score/"XP_A13_F8_1440_SCORED_ROOTS.csv"
 gate(rec["status"]=="XP_A13_F8_1440_CPU_SCORED_PASS"
      and sha(file)==rec["scoring_csv_sha256"],"SCORE_SHA")
 with file.open(newline="") as f:rows=list(csv.DictReader(f))
 gate(len(rows)==1440 and
      len({(r["cohort"],r["image_id"],int(r["run"]))
           for r in rows})==1440,"CANDIDATE_GRID")
 clipfile,native,nativehead=model_source()
 import torch
 from prdiffusion.track_r.r0n_clip_compat import encode_text_univfd_tuple_compat
 torch.set_num_threads(4)
 sys.path.insert(0,str(native))
 clip=importlib.import_module("models.clip")
 model,preprocess=clip.load(str(clipfile),device="cpu",jit=False)
 model.eval()
 with torch.inference_mode():
  tok=clip.tokenize(list(PROMPTS_UP+PROMPTS_INV)).to("cpu")
  text=encode_text_univfd_tuple_compat(model,tok).float()
  text=text/text.norm(dim=-1,keepdim=True)
  u=text[:len(PROMPTS_UP)].mean(0);d=text[len(PROMPTS_UP):].mean(0)
  u=u/u.norm();d=d/d.norm()
  answers=[]
  for start in range(0,len(rows),a.batch):
   batch=rows[start:start+a.batch]
   x=[]
   for r in batch:
    path=Path(r["reconstruction_path"])
    gate(path.is_file() and not path.is_symlink() and
         ROOT.resolve() in path.resolve().parents and
         not any(k in str(path.resolve()).lower() for k in
                 ("confirmation","confirm305","test-ffhq","demo-ffhq"))
         and sha(path)==r["reconstruction_sha256"],
         "CANDIDATE_SHA")
    rgb=candidate_tensor(dict(
      uid=r["uid"],candidate_path=str(path),
      candidate_format=r["reconstruction_format"],
      candidate_sha256=r["reconstruction_sha256"]))
    arr=rgb[0].permute(1,2,0).numpy()
    x.extend([preprocess(Image.fromarray(arr,"RGB")),
       preprocess(Image.fromarray(np.rot90(arr,2).copy(),"RGB"))])
   embedding=model.encode_image(torch.stack(x)).float()
   embedding/=embedding.norm(dim=-1,keepdim=True)
   score=embedding@u-embedding@d
   margins=(score[1::2]-score[::2]).tolist()
   for r,margin in zip(batch,margins):
    gate(math.isfinite(margin),"MARGIN")
    choice="111" if margin>0 else "000"
    good=int(r["flip_good"]) if choice=="111" else (
           int(float(r["psnr_raw"])>=25))
    answers.append(dict(image_id=r["image_id"],
       run=int(r["run"]),label=r["label"],f_choice=choice,
       f_margin=format(margin,".10g"),f_chosen_good=good,
       cohort=r["cohort"]))
   if len(answers)%80==0:
    print("A13_F8_FACECLIP",len(answers),"/1440",flush=True)
 gate(len(answers)==1440,"FACECLIP_FULL")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a13_f8_clip_")
      and not out.exists() and not out.is_symlink(),"OUT")
 out.mkdir(mode=0o700)
 output=out/"XP_A13_F8_8RUN_FACE_CLIP_DECISIONS.csv"
 with output.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(answers)
 write(out/"XP_A13_F8_FACECLIP_RECEIPT.json",dict(
   status="XP_A13_F8_1440_N2_FACE_CLIP_PASS",source_head=a.expected_head,
   n_candidates=1440,score_sha256=rec["scoring_csv_sha256"],
   frozen_clip_weights_sha256=CLIP_SHA,univfd_head=nativehead,
   prompt_u=list(PROMPTS_UP),prompt_inv=list(PROMPTS_INV),
   clip_csv_sha256=sha(output),model_device="cpu",
   no_GT_read_by_selector=True))
 print("XP_A13_F8_1440_FACECLIP_CPU_PASS",sha(output),flush=True)
 print("F8_CLIP_OUT",out,flush=True)
def entry():
 p=argparse.ArgumentParser()
 for q in ("expected-head","score-dir","out"):
  p.add_argument("--"+q,required=True)
 p.add_argument("--batch",type=int,default=8)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A13_F8_FACECLIP",e,flush=True)
  sys.exit(8)
if __name__=="__main__":entry()
