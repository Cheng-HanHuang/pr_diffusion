#!/usr/bin/env python3
"""A17.3 immutable XP-I raw and face-chosen 720-particle metrics.

Adds raw/111 face chosen PSNR, piq SSIM, piq VGG-LPIPS from accepted
CP3 receipts. No new GPU, model inference, GT tensor read or selection.
The original A16 table is unchanged and SHA-verified.
"""
from __future__ import annotations
import argparse,csv,hashlib,math,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,write
from scripts.track_r.xp_a16_cp3_joint_readout_cpu import read_stage
from scripts.track_r.xp_a18_xpi_particle_metrics_cpu import rows_from_stage,COLUMNS

SOURCE=BASE/"xp_a18_xpi_particles_20261031T224601Z"
SOURCE_SHA="1e16f70ebd36abf81c9662edef2eef813d061ce98296d1c3bfbaba2a4cc17f02"
COLS=(*COLUMNS,"PSNR_chosen","SSIM_chosen","VGG_LPIPS_chosen")
EXPECTED_E0_ROTATED={
 "S1_b22_62739","S1_b22_67293","S1_b22_69970",
 "S2_b24_dev80_07985","S2_b24_dev80_21603",
 "S2_b24_dev80_38516","S2_b24_dev80_56397"}

def checked_table():
 rpath=SOURCE/"A16_XPI_PER_PARTICLE_METRICS.csv"
 gate(sha(rpath)==SOURCE_SHA,"A19_XPI_ACCEPTED_RAW_SHA")
 with rpath.open(newline="",encoding="utf-8") as f:
  src=list(csv.DictReader(f))
 gate(len(src)==720 and set(src[0])==set(COLUMNS),
      "A19_XPI_SOURCE_RAW_720")
 stage=read_stage("G3")
 original=rows_from_stage(stage)
 for row,prev in zip(original,src):
  gate(str(row["image_id"])==prev["image_id"]
       and int(row["particle"])==int(prev["particle"])
       and row["label"]==prev["label"]
       and row["face_choice"]==prev["face_choice"]
       and all(abs(float(row[k])-float(prev[k]))<=5e-11
               for k in ("PSNR","SSIM","VGG_LPIPS","rho","face_margin")),
       "A19_XPI_IDENTICAL_ACCEPTED_NATIVE_ROWS")
 new=[]
 for row in original:
  iid=row["image_id"];i=int(row["particle"])
  sc=stage["candidates"][iid,i]
  mm=stage["metrics"][iid,i]
  flip=row["face_choice"]=="111"
  psnr=float(sc["psnr_flip"] if flip else sc["psnr_raw"])
  ssim=float(mm["SSIM_111"] if flip else mm["SSIM_raw"])
  lp=float(mm["LPIPS_VGG_111"] if flip else mm["LPIPS_VGG_raw"])
  gate(all(math.isfinite(v) for v in (psnr,ssim,lp)),
       "A19_XPI_CHOSEN_FINITE")
  if not flip:
   gate(abs(psnr-row["PSNR"])<1e-8
        and abs(ssim-row["SSIM"])<1e-10
        and abs(lp-row["VGG_LPIPS"])<1e-10,
        "A19_XPI_ZERO_FLIP_RAW_IDENTICAL")
  new.append(dict(row,PSNR_chosen=psnr,
    SSIM_chosen=ssim,VGG_LPIPS_chosen=lp))
 # Registered face changes on precisely 7 E0 selected particles.
 rotated=set()
 for key,rs in stage["image_groups"].items():
  iid=key[1]; j=min(range(4),key=lambda i:
    (float(stage["candidates"][iid,i]["rho_signed"]),i))
  if stage["clips"][iid,j]["face_choice"]=="111":
   rotated.add(iid)
 gate(rotated==EXPECTED_E0_ROTATED,"A19_XPI_EXACT_7_E0_FACE_FLIPS")
 gate(len(new)==720 and len({r["image_id"] for r in new})==180,
      "A19_XPI_FULL_720")
 return new

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A19_XPI_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *xs:subprocess.check_output(["git","-C",str(wt),*xs],
  text=True,timeout=40).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"A19_XPI_CLEAN_HEAD")
 rows=checked_table()
 if a.preflight_only:
  print("A19_XPI_CHOSEN_720_PREFLIGHT_PASS",flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a19_xpi_chosen_")
      and not out.exists() and not out.is_symlink(),
      "A19_XPI_NEW_OUTPUT")
 out.mkdir(mode=0o700)
 path=out/"A17_XPI_PER_PARTICLE_CHOSEN_METRICS.csv"
 with path.open("x",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=COLS,lineterminator="\n")
  w.writeheader();w.writerows(rows)
 write(out/"A17_XPI_CHOSEN_RECEIPT.json",dict(
  status="A17_XPI_FULL_720_RAW_AND_FACE_CHOSEN_PASS",
  expected_head=a.expected_head,n_particles=720,n_images=180,
  source_accepted_A16_csv_sha256=SOURCE_SHA,
  augmented_csv_sha256=sha(path),raw_columns_unchanged=True,
  exactly_7_E0_flips_verified=True,
  face_decisions_frozen_GT_free=True,
  chosen_metrics_use_GT_evaluation_only=True,no_new_gpu_or_clips=True,
  no_matched_compute_claim=True,
  columns=list(COLS)))
 print("A19_XPI_CHOSEN_720_PASS",path,sha(path),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--preflight-only",action="store_true")
 try: main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A19_XPI_CHOSEN",str(e),flush=True)
  sys.exit(8)
