#!/usr/bin/env python3
"""A19 XP-I2 frozen GT-free ImageNet D1 100-bag manifest (CPU only).

Uses EXISTING D1 signed y (not XP-N); native PC-DAPS 59b2f046,
100 ImageNet image IDs, one bag of four particles each. Does NOT run
the FFHQ-only G3 worker. This manifest must be consumed by a separately
pinned, replay-verified ImageNet PC-DAPS worker.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g367_readiness_cpu import PC,PC_HEAD
from scripts.track_r.xp_t_imagenet_prepare_cpu import MODEL_SHA as IMAGE_MODEL_SHA

NATIVE_FILES=("sampler.py","model/__init__.py",
              "configs/sampler/edm_pcdaps.yaml",
              "configs/task/phase_retrieval.yaml")
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
def sha(path):
 p=Path(path)
 gate(p.is_file() and not p.is_symlink() and ROOT.resolve() in p.resolve().parents,
      "I2_SAFE_FILE_"+p.name)
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def seed(basename):
 s=f"XP|PCDAPS|imagenet|{basename}|0"
 return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],"big")%(2**31)
def sources(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","I2_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
                        text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"I2_CLEAN_HEAD")
 gate(PC.is_dir() and git_native(PC,"rev-parse","HEAD").startswith(PC_HEAD)
      and not git_native(PC,"status","--porcelain","--untracked-files=no"),
      "I2_NATIVE_PC_CHECKOUT")
 model=ROOT/"models/imagenet256.pt"
 gate(sha(model)==IMAGE_MODEL_SHA,"I2_IMAGENET_MODEL_SHA")
 for n in NATIVE_FILES:gate(sha(PC/n),"I2_NATIVE_FILE_SHA_"+n)
 candidates=[]
 for d in BASE.glob("xp_t_fast_imagenet_*/XP_IMAGENET100_Y_MANIFEST.json"):
  if d.is_file() and not d.is_symlink():
   j=json.loads(d.read_text())
   if j.get("status")=="XP_IMAGENET100_Y_FROZEN" and j.get("n_images")==100:
    candidates.append((d,j))
 gate(len(candidates)==1,"I2_EXACT_ONE_D1_COMMON_Y")
 f,ym=candidates[0]
 ids=[r["basename"] for r in ym["rows"]]
 gate(len(ids)==len(set(ids))==100 and ids==sorted(ids),
      "I2_D1_IMAGE_ORDER")
 gate(ym.get("gt_paths_supplied") is False,
      "I2_D1_NO_GT_IN_Y")
 rows=[]
 for r in ym["rows"]:
  basename=r["basename"];p=Path(r["measurement_path"])
  gate(p.name.startswith("image_") and p.suffix==".pt"
       and p.is_absolute() and not any(w in str(p).lower() for w in BAD)
       and sha(p)==r["measurement_sha256"],
       "I2_D1_FROZEN_Y_"+basename)
  rows.append(dict(uid=f"XP|PCDAPS|imagenet|{basename}|bag0",
       dataset="imagenet",cohort="XP_IMAGENET100",
       image_id=basename,bag_size=4,seed=seed(basename),
       measurement_path=str(p),measurement_sha256=r["measurement_sha256"]))
 gate(len({x["uid"] for x in rows})==100
      and len({x["seed"] for x in rows})==100,
      "I2_BAG_SEED_COLLISION")
 return wt,f,ym,rows
def git_native(root,*q):
 return subprocess.check_output(["git","-C",str(root),*q],
                               text=True,timeout=30).strip()
def write(p,j):
 with p.open("x") as f:
  json.dump(j,f,sort_keys=True,indent=2);f.write("\n")
def main(a):
 wt,f,ym,rows=sources(a)
 if a.preflight_only:
  print("A24_XPI2_D1_Y_AND_PC_NATIVE_100_PREFLIGHT_PASS",
        "Y_SHA",sha(f),"MODEL_SHA",IMAGE_MODEL_SHA,flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a24_xpi2_prepare_")
      and not out.exists() and not out.is_symlink(),
      "I2_FRESH_OUT")
 out.mkdir(mode=0o700)
 native=git_native(PC,"rev-parse","HEAD")
 doc=dict(schema="prdiff.track_r.a24.xpi2-pcdaps.v1",
  status="A19_XPI2_D1_100_NATIVE_FOUR_PARTICLE_BAGS_FROZEN",
  source_head=native,expected_head=a.expected_head,
  model_sha256=IMAGE_MODEL_SHA,dataset="imagenet",
  checkpoint=str(ROOT/"models/imagenet256.pt"),
  source_y_manifest_sha256=sha(f),
  n_images=100,n_bags=100,bag_size=4,n_particles=400,
  no_GT_paths=True,
  seed_rule="SHA256(XP|PCDAPS|imagenet|<basename>|0)[:8] BE mod 2**31",
  native_source_files={n:sha(PC/n) for n in NATIVE_FILES},
  native_sampler_config="edm_pcdaps native 200x5, 4 particles, prune on",
  strict_native_model_load_required=True,
  two_bag_exact_replay_required=True,
  generation_authorized=False,
  rows=rows)
 mf=out/"XP_A24_XPI2_100_BAGS.json"
 write(mf,doc)
 write(out/"XP_A24_XPI2_PREP_RECEIPT.json",dict(
   status="A19_XPI2_D1_100_BAGS_CPU_FROZEN_NOT_GENERATED",
   manifest_sha256=sha(mf),n_bags=100,n_particles=400,
   original_D1_y_sha256=sha(f),checkpoint_sha256=IMAGE_MODEL_SHA,
   source_head=native,expected_head=a.expected_head,
   no_GT_read=True,one_bag_gpu_worker_required=True,
   imgnet_native_source_sanity_pending=True,
   GPU_clean_required_for_native_quality=False))
 print("A24_XPI2_D1_100_BAGS_FROZEN",mf,sha(mf),flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--out",required=True)
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A24_XPI2_PREP",e,flush=True);sys.exit(8)
