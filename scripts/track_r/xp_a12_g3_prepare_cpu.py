#!/usr/bin/env python3
"""A12 G3 PC-DAPS FFHQ: freeze 180 native four-particle bag tasks, no GT.

Native source exact 59b2f046. Each bag uses only an original frozen R0
signed measurement; no dataset bundle, source GT, labels or predictions.
Absent code emits required exact clone command, never downloads itself.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from collections import defaultdict
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_g367_readiness_cpu import PC,PC_HEAD,CLONE_COMMANDS
from scripts.track_r.xp_t_sitcom_prepare_cpu import MODEL_SHA
CAT=BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def write(p,v):
 with Path(p).open("x") as f:json.dump(v,f,indent=2,sort_keys=True);f.write("\n")
def seed(uid):
 k=f"XP|PCDAPS|ffhq|{uid}|0"
 return int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],"big")%(2**31)
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 cmd=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
                  text=True,timeout=30).strip()
 gate(cmd("rev-parse","HEAD")==a.expected_head and
      not cmd("status","--porcelain"),"WORKTREE")
 if not PC.is_dir():
  print("XP_A12_G3_CONDITIONAL_CODE_ABSENT",flush=True)
  for x in CLONE_COMMANDS:print("CHENG_CLONE",x,flush=True)
  return 0
 pch=subprocess.check_output(["git","-C",str(PC),"rev-parse","HEAD"],text=True).strip()
 gate(pch.startswith(PC_HEAD) and not subprocess.check_output(
      ["git","-C",str(PC),"status","--porcelain","--untracked-files=no"],
      text=True).strip(),"NATIVE_PC_SOURCE")
 for p in ("posterior_sample.py","sampler.py","configs/sampler/edm_pcdaps.yaml",
           "configs/task/phase_retrieval.yaml"):
  gate((PC/p).is_file() and sha(PC/p),"PC_NATIVE_FILES")
 gate(sha(ROOT/"models/ffhq_10m.pt")==MODEL_SHA,"FFHQ_CHECKPOINT")
 cat=json.loads(CAT.read_text())
 sources=defaultdict(list)
 for r in cat["rows"]:
  if r["source"]=="S5_r0_fast":
   k=(r["measurement_source"],str(r["image_id"]))
   gate(k[0] in ("S1_b22","S2_b24_dev80"),"COHORT")
   sources[k].append(r)
 gate(len(sources)==180,"COHORT180")
 rows=[]
 for (cohort,iid),group in sorted(sources.items()):
  # R0.7 offline catalog records run_index; rep belongs to GPU manifests.
  group.sort(key=lambda x:int(x["run_index"]))
  gate([int(r["run_index"]) for r in group]==list(range(4)),"R0_PANEL")
  y={(r["measurement_path"],r["measurement_sha256"]) for r in group}
  gate(len(y)==1,"R0_Y")
  yp,yd=next(iter(y))
  path=Path(yp).resolve(strict=True)
  gate(ROOT.resolve() in path.parents and
       not any(k in str(path).lower() for k in BAD)
       and sha(path)==yd,"R0_Y_SHA")
  uid=cohort+"_"+iid
  rows.append(dict(uid=f"XP|PCDAPS|ffhq|{uid}|bag0",
      cohort=cohort,image_id=iid,seed=seed(uid),bag_size=4,
      measurement_path=yp,measurement_sha256=yd))
 gate(len(rows)==180 and len({x["uid"] for x in rows})==180,"180_BAGS")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a12_g3_prepare_")
      and not out.exists() and not out.is_symlink(),"OUTPUT")
 out.mkdir(mode=0o700)
 manifest=dict(schema="prdiff.track_r.a12.g3-pcdaps.v1",
    status="XP_A12_G3_180_NATIVE_BAGS_FROZEN",
    source_head=pch,head=a.expected_head,
    native_solver="PC-DAPS",native_config="edm_pcdaps",
    task="phase_retrieval",dataset="ffhq",
    bag_size=4,n_images=180,n_bags=180,n_particles=720,
    model_sha256=MODEL_SHA,
    gt_paths_supplied=False,
    output_all_particles=True,output_method_metric_estimate=True,
    native_source_files={p:sha(PC/p) for p in (
       "sampler.py","model/__init__.py","configs/sampler/edm_pcdaps.yaml",
       "configs/task/phase_retrieval.yaml")},
    rows=rows)
 path=out/"XP_A12_G3_180_BAGS.json";write(path,manifest)
 write(out/"XP_A12_G3_PREP_RECEIPT.json",dict(
    status="XP_A12_G3_CPU_PREP_PASS",head=a.expected_head,
    source_head=pch,n_bags=180,n_particles=720,
    manifest_sha256=sha(path),no_GT_read=True,no_GPU=True,
    two_id_smoke_required=True,cohortA_only_if_projected_over_45=True,
    cap_gpu_h=45))
 print("XP_A12_G3_180_BAGS_CPU_PREP_PASS",flush=True)
 print("G3_SOURCE_HEAD",pch,flush=True)
 print("G3_MANIFEST_SHA",sha(path),flush=True)
 print("G3_PREP_DIR",out,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 try:main(p.parse_args())
 except Exception as exc:
  traceback.print_exc();print("ACTION NEEDED: A12_G3_PREP",exc,flush=True)
  sys.exit(8)
