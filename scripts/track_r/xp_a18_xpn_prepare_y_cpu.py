#!/usr/bin/env python3
"""Orchestrate A16 XP-N signed measurement preparation, fail-closed CPU only.

Untrusted JPEG decoding runs ONLY in a separate python -I interpreter
with cwd = the pinned raw ImageNet directory; no image decoder runs here.
This does not authorize GPU generation until independent GT/Y and two-root
replay gates pass. It never reads D1 candidate outcomes or GT scores.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write
from scripts.track_r.xp_a18_xpn_image_inventory_cpu import check_d1,EXPECTED
from scripts.track_r.xp_t_imagenet_prepare_cpu import MODEL_SHA

PINNED_SOURCE="5fa348229a35899f39aa06326840e9a8592eea6e882450bbbd0068895cee371b"
PINNED_SOURCE_DIR=BASE/"xp_a18_xpn_source_20261031T221719Z"
PINNED_IMG_DIR=ROOT/"data/imagenet/val_raw"
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
MODEL=ROOT/"models/imagenet256.pt"

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A18_XPN_CPU_ONLY")
 gate(a.cheng_go=="GO","A18_XPN_EXPLICIT_GO")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *z:subprocess.check_output(
    ["git","-C",str(wt),*z],text=True,timeout=60).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),
      "A18_XPN_PINNED_CLEAN_REPO")
 check_d1()
 source=PINNED_SOURCE_DIR/"XP_N_SOURCE_IMAGES_LOCAL_CPU.json"
 gate(source.is_file() and not source.is_symlink()
      and sha(source)==PINNED_SOURCE,"A18_XPN_PINNED_100_SOURCE_SHA")
 sf=jread(source)
 gate(sf["cheng_go"] is True
      and sf["status"]=="XP_N_GO_RAW_IMAGE_SOURCE_AUDITED_NO_GT_Y_YET"
      and sf["n_images"]==100
      and len(sf["source_rows"])==100
      and all(r["basename"] in EXPECTED and
              r["source_path"]==str(PINNED_IMG_DIR/r["basename"])
              for r in sf["source_rows"]),
      "A18_XPN_GO_AND_RAW_SOURCE_LOCK")
 gate(PY.is_file() and MODEL.is_file() and sha(MODEL)==MODEL_SHA,
      "A18_XPN_IMAGENET_CHECKPOINT")
 gate(PINNED_IMG_DIR.is_dir() and not PINNED_IMG_DIR.is_symlink()
      and PINNED_IMG_DIR.resolve()==PINNED_IMG_DIR,
      "A18_XPN_RAW_IMAGE_DIRECTORY")
 gate(Path(a.out).parent==BASE and
      Path(a.out).name.startswith("xp_a18_xpn_measurements_")
      and not Path(a.out).exists()
      and not Path(a.out).is_symlink(),"A18_XPN_NEW_OUTPUT")
 if a.preflight_only:
  print("A18_XPN_GT_Y_DECODE_PREFLIGHT_PASS",
        "IMAGES",100,"MODEL_SHA",MODEL_SHA,
        "SOURCE_SHA",PINNED_SOURCE,
        "ISOLATED_PYTHON_REQUIRED",flush=True)
  return
 helper=wt/"scripts/track_r/xp_a18_xpn_decode_isolated.py"
 gate(helper.is_file() and not helper.is_symlink(),
      "A18_XPN_PINNED_ISOLATED_HELPER")
 env=dict(os.environ)
 for k in ("PYTHONPATH","PYTHONHOME","PYTHONUSERBASE",
           "PYTHONSTARTUP","PYTHONINSPECT"):
  env.pop(k,None)
 env.update(CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
 cmd=[str(PY),"-I","-B",str(helper),
      "--repo",str(wt),
      "--source-dir",str(PINNED_IMG_DIR),
      "--source-manifest",str(source),
      "--source-manifest-sha",PINNED_SOURCE,
      "--expected-head",a.expected_head,"--out",a.out]
 print("A18_XPN_ISOLATED_JPEG_DECODE_START",
       "PYTHON_FLAG","-I","RAW_DIR",PINNED_IMG_DIR,flush=True)
 result=subprocess.run(cmd,cwd=str(PINNED_IMG_DIR),
                       env=env,timeout=1800,check=False)
 gate(result.returncode==0,"A18_XPN_ISOLATED_WORKER_RC_"+str(result.returncode))
 out=Path(a.out)
 ym=out/"XP_N_IMAGENET100_Y_MANIFEST.json"
 gt=out/"XP_N_GT_ONLY_CPU.json"
 gate(ym.is_file() and gt.is_file(),"A18_XPN_DECODE_COMPLETED_OUTPUTS")
 yman=jread(ym);gman=jread(gt)
 gate(yman["status"]=="XP_N_IMAGENET100_SIGNED_Y_FROZEN"
      and yman["expected_head"]==a.expected_head
      and yman["n_images"]==100
      and yman["source_manifest_sha256"]==PINNED_SOURCE
      and yman["gt_paths_supplied"] is False
      and gman["status"]=="XP_N_GT_ONLY_OFFLINE_NOT_FOR_GENERATION"
      and gman["n_images"]==100
      and gman["isolated_python_flag"] is True
      and gman["all_identity_rms_le_1e_minus_4"] is True
      and len(yman["rows"])==len(gman["rows"])==100
      and all(sha(Path(r["measurement_path"]))==r["measurement_sha256"]
              for r in yman["rows"])
      and all(sha(Path(r["gt_path"]))==r["gt_file_sha256"]
              for r in gman["rows"]),
      "A18_XPN_GT_Y_RECEIPTS_AND_SHA")
 rec=dict(status="A18_XPN_GT_Y_ISOLATED_CPU_PREP_PASS_IDENTITY_REAUDIT_PENDING",
      expected_head=a.expected_head,cheng_go=True,
      n_images=100,source_manifest_sha256=PINNED_SOURCE,
      model_sha256=MODEL_SHA,
      y_manifest_sha256=sha(ym),gt_only_manifest_sha256=sha(gt),
      worst_worker_identity_rms=gman["worst_identity_rms"],
      exact_D1_preprocessing=True,signed_y=True,sigma=.05,
      noise_family_tag="XPMEAS|imagenet_n",
      gt_paths_in_generation=False,isolated_decode_python_I=True,
      no_new_GPU_calls=True,independent_GT_Y_reaudit_required=True,
      D1_replay_required_before_XPN_generation=True,
      XP_N_runs_4_to_63_sealed=True,
      expected_project_GPU_cap_h=400,XP_N_stage_GPU_cap_h=20)
 write(out/"XP_N_GT_Y_PREP_RECEIPT.json",rec)
 print("A18_XPN_GT_Y_CPU_PREP_PASS",out,
       "Y_SHA",rec["y_manifest_sha256"],
       "GT_SHA",rec["gt_only_manifest_sha256"],flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--cheng-go",choices=("GO",),required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as exc:
  traceback.print_exc()
  print("ACTION NEEDED: A18_XPN_GT_Y_PREP",exc,flush=True)
  sys.exit(8)
