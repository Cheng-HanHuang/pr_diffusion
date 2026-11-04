#!/usr/bin/env python3
"""A22 diagnostic GT-only C/C2 frozen input, from original prereg source.

No candidates or outcomes accessed. Does not regenerate experimental y,
does not supply GT to inverse-problem solvers.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r import xp_c_c2_prepare_cpu as P

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
def gate(v,k):
 if not v:raise RuntimeError("STOP_A22_G7_PREP_"+k)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def git(*xs):return subprocess.check_output(
 ["git","-C",str(WT),*xs],text=True).strip()
def dump(path,data):
 with Path(path).open("x") as f:
  json.dump(data,f,sort_keys=True,indent=2,allow_nan=False)
  f.write("\n")
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
      and git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"CLEAN_CPU_HEAD")
 assert sha(P.IMAGE_INDEX)==P.im.COMMON_Y_SHA
 gate(sha(ROOT/"models/imagenet256.pt")==P.im.IMAGENET_MODEL_SHA,
      "IMAGENET_MODEL")
 gate(sha(ROOT/"models/ffhq_10m.pt")==
      "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa",
      "FFHQ_MODEL")
 catalog=P.jsonread(P.R07);src=P.jsonread(P.SOURCE)
 images=P.freeze(catalog,src)
 tasks=P.c2_tasks(images)
 gate(len(images)==280 and len(tasks)==10080 and
      all(z["source_path"].startswith(str(ROOT)+"/")
          for z in images),"NATIVE_SCOPE")
 root=Path(a.out)
 gate(root.parent==BASE and root.name.startswith("xp_c_c2_a22_")
      and not root.exists() and not root.is_symlink(),
      "NEW_FROZEN_ROOT")
 root.mkdir(mode=0o700)
 c=root/"XP_C_GT_IMAGES.json"
 t=root/"XP_C2_EQUIVARIANCE_TASKS.json"
 dump(c,dict(status="XP_C_GT_IMAGES_FROZEN",head=a.expected_head,
  n_images=280,n_ffhq=180,n_imagenet=100,images=images,
  diagnostic_gt_only=True,scoring_or_GPU_work_performed=False,
  policies=list(P.C_POLICIES),
  ffhq_uses_CLIP_face=True,imagenet_uses_G1_G2=True))
 dump(t,dict(status="XP_C2_GT_EQ_TASKS_FROZEN",head=a.expected_head,
  n_images=280,n_tasks=10080,
  t_ddpm1000=list(P.T),rotations=list(P.ROT),
  draws_per_image_t_rotation=2,
  noise_rule="SHA256(XPC2|dataset|basename|t|j)[:8] unsigned BE mod 2**63",
  denoiser_time_mapping="REQUIRED_BEFORE_GPU: map DDPM-1000 index to each checkpoint native schedule",
  delta_definition="mean||D_t(Rx_t)-R D_t(x_t)||^2 / (3*256^2)",
  delta_tilde_denominator="mean||D_t(x_t)-x_gt||^2 / (3*256^2)",
  diagnostic_gt_only=True,gpu_work_performed=False,tasks=tasks))
 receipt=root/"XP_C_C2_PREP.json"
 dump(receipt,dict(status="XP_C_C2_PREP_PASS",
   head=a.expected_head,n_images=280,
   n_C2_evaluations=10080,
   C_manifest_sha256=sha(c),C2_manifest_sha256=sha(t),
   gt_paths_present_FOR_DIAGNOSTICS_ONLY=True,
   no_candidate_outcomes_inspected=True,gpu_work_performed=False,
   source_R0_GT_catalog_sha256=sha(P.R07),
   source_ImageNet_GT_index_sha256=sha(P.SOURCE)))
 print("A22_G7_DIAGNOSTIC_ONLY_PREP_PASS",root,
  "C_SHA",sha(c),"C2_SHA",sha(t),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A22_G7_PREP",str(e),flush=True)
  sys.exit(8)
