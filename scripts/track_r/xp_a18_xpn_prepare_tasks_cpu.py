#!/usr/bin/env python3
"""A16.5 XP-N post-replay GT-free CPU freeze: pilot 0-3 / main 4-15 / late 16-63.

No model inference or candidate inspection. Three manifests are deliberately
separate to enforce runs 0-3 preregistration and the 4..63 outcome seal.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write
from scripts.track_r.xp_t_imagenet_prepare_cpu import seed as native_seed
from scripts.track_r.xp_a18_xpn_contract import (
  PHASES,MODEL_SHA,FAST_HEAD,FROZEN_Y,Y_SHA,SOURCE_SHA,AUDIT_SHA,
  validate_xpn_source)

FILES={
 "A18_XPN_PILOT_0_3":"XP_N_PILOT_0_3_TASKS.json",
 "A18_XPN_MAIN_4_15":"XP_N_MAIN_4_15_TASKS.json",
 "A18_XPN_LATE_16_63":"XP_N_LATE_16_63_TASKS.json",
}
REPLAY_STATUS="A18_XPN_D1_REPLAY2_PIXEL_EXACT_PASS"
ORIGINAL_Y_PREPARER="03645a5b981d8d2abca7b4cda101e9272844cf93"
STAGE_GPU_CAP_H=20.
PROJECT_GPU_CAP_H=400.

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A18_XPN_PREP_CPU")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *z:subprocess.check_output(
   ["git","-C",str(wt),*z],text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),
      "A18_XPN_PREP_PINNED_CLEAN_HEAD")
 gate(a.cheng_go=="GO","A18_XPN_PREP_CHENG_GO")
 replaydir=Path(a.replay_dir)
 gate(replaydir.parent==BASE and
      replaydir.name.startswith("xp_a18_xpn_replay_")
      and replaydir.is_dir() and not replaydir.is_symlink(),
      "A18_XPN_PREP_REPLAY_DIRECTORY")
 rp=replaydir/"XP_N_D1_REPLAY_RECEIPT.json"
 replay=jread(rp)
 gate(replay["status"]==REPLAY_STATUS and
      replay["n_roots"]==2 and replay["native_worker_rc"]==0
      and replay["visibility_unknown"] is False
      and all(r["pixels_exact"] is True for r in replay["pixel_comparisons"])
      and replay["no_XPN_candidates_generated"] is True,
      "A18_XPN_PREP_TWO_ROOT_REPLAY_EXACT")
 ym=jread(FROZEN_Y)
 gate(sha(FROZEN_Y)==Y_SHA
      and ym["expected_head"]==ORIGINAL_Y_PREPARER
      and ym["status"]=="XP_N_IMAGENET100_SIGNED_Y_FROZEN"
      and ym["source_manifest_sha256"]==SOURCE_SHA
      and ym["n_images"]==100
      and ym["gt_paths_supplied"] is False
      and ym["dataset"]=="imagenet"
      and ym["sigma"]==.05
      and len(ym["rows"])==100,
      "A18_XPN_PREP_FROZEN_SIGNED_Y")
 aud=FROZEN_Y.parent/"XP_N_GT_Y_INDEPENDENT_AUDIT.json"
 d=jread(aud)
 gate(sha(aud)==AUDIT_SHA
      and d["status"]=="A18_XPN_GT_Y_INDEPENDENT_ALL100_PASS"
      and d["n_images"]==100 and d["signed_y_reconstituted_all100"]
      and d["GT_tensor_sha256_verified_all100"]
      and d["worst_identity_rms"]==0.,
      "A18_XPN_PREP_INDEPENDENT_Y_GT_AUDIT")
 gate(replay["XP_N_independent_GT_Y_audit_sha256"]==AUDIT_SHA,
      "A18_XPN_PREP_REPLAY_USES_SAME_Y_AUDIT")
 model=ROOT/"models"/"imagenet256.pt"
 gate(model.is_file() and sha(model)==MODEL_SHA,
      "A18_XPN_PREP_MODEL_SHA")
 rows=ym["rows"]
 expected=[f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(48900,49000)]
 gate([r["basename"] for r in rows]==expected
      and len({r["measurement_sha256"] for r in rows})==100
      and all(sha(r["measurement_path"])==r["measurement_sha256"]
              and r["measurement_source"]=="XP_IMAGENET_N100"
              if "measurement_source" in r else
              sha(r["measurement_path"])==r["measurement_sha256"]
              for r in rows),
      "A18_XPN_PREP_EXPECTED_100_SIGNED_Y")
 phases={}
 for phase,(lo,hi,total) in PHASES.items():
  tasks=[]
  for r in rows:
   iid=r["basename"]
   for rep in range(lo,hi):
    tasks.append(dict(
       uid=f"XP|FAST|imagenet_n|{iid}|rep{rep}",
       source="XP_N",measurement_source="XP_IMAGENET_N100",
       image_id=iid,rep=rep,
       seed=native_seed("XP|FAST|imagenet_n",iid,rep,2**31),
       measurement_path=r["measurement_path"],
       measurement_sha256=r["measurement_sha256"]))
  gate(len(tasks)==total and len({z["uid"] for z in tasks})==total
       and len({z["seed"] for z in tasks})==total,
       "A18_XPN_PREP_PHASE_GRID_"+phase)
  phases[phase]=dict(schema="prdiff.track_r.xp.fast-tasks.v1",
    status="XP_FAST_TASKS_FROZEN",
    head=a.expected_head,fast_head=FAST_HEAD,
    stage=phase,dataset="imagenet",oversample=2.0,sigma=.05,
    model_sha256=MODEL_SHA,n_images=100,n_roots=total,
    gt_paths_supplied=False,outcome_embargo=True,
    new_image_cohort="ILSVRC2012_val_00048900-00048999",
    source_y_manifest_sha256=Y_SHA,
    source_image_inventory_sha256=SOURCE_SHA,
    source_gt_y_audit_sha256=AUDIT_SHA,
    seed_rule="XP|FAST|imagenet_n|<basename>|<rep> SHA256 first8 BE mod2**31",
    rows=tasks)
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a18_xpn_tasks_")
      and not out.exists() and not out.is_symlink(),
      "A18_XPN_PREP_NEW_DIR")
 if a.preflight_only:
  print("A18_XPN_PRODUCTION_TASK_PREFLIGHT_PASS",
        "IMAGE_UNIVERSE",100,"MANIFESTS",len(phases),
        "RUNS",sum(x["n_roots"] for x in phases.values()),
        "REPLAY",replay["n_roots"],flush=True)
  return
 out.mkdir(mode=0o700)
 files={}
 for phase,mf in phases.items():
  f=out/FILES[phase]
  write(f,mf)
  files[phase]=dict(path=str(f),sha256=sha(f),
                  n_roots=mf["n_roots"],range=list(PHASES[phase][:2]))
 write(out/"XP_N_TASKS_PREP_RECEIPT.json",dict(
    status="A18_XPN_GT_FREE_6400_MANIFESTS_FROZEN_PASS",
    expected_head=a.expected_head,cheng_go=True,
    n_images=100,n_roots=6400,
    n_pilot=400,n_main=1200,n_late=4800,
    source_y_manifest_sha256=Y_SHA,
    source_inventory_sha256=SOURCE_SHA,
    source_gt_y_audit_sha256=AUDIT_SHA,
    native_d1_replay_pixel_exact=True,
    replay_receipt_sha256=sha(rp),
    pilot_0_to_3_must_be_scored_and_published_separately=True,
    runs_4_to_63_sealed_until="v5_registered_or_declined_OR_2026-11-04T12:00:00-05:00",
    protocol_v1_unmodified=True,
    no_matched_compute_claim=True,
    CPU_only=True,no_GT_or_candidate_outcomes_read=True,
    native_config_source="D1 pixel FAST T150 ADMM2 S5",
    stage_GPU_cap_h=STAGE_GPU_CAP_H,project_cap_GPU_h=PROJECT_GPU_CAP_H,
    gpu_production_not_started=True,
    phase_files=files))
 # Now the native worker can verify the immutable CPU receipt and bytes.
 for phase,mf in phases.items():
  f=out/FILES[phase]
  validate_xpn_source(f,sha(f),mf,mf["rows"],
                      a.expected_head,0,min(mf["n_roots"],PHASES[phase][1]-PHASES[phase][0]))
 print("A18_XPN_GT_FREE_6400_MANIFESTS_FROZEN_PASS",
       "PREP",out,flush=True)
 for phase,item in files.items():
  print("A18_XPN_PHASE",phase,item["n_roots"],
        item["sha256"],flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--cheng-go",choices=("GO",),required=True)
 p.add_argument("--replay-dir",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A18_XPN_PREP_GT_FREE",e,flush=True)
  sys.exit(8)
