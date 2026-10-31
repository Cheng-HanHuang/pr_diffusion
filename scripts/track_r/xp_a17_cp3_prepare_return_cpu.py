#!/usr/bin/env python3
"""A15 CP3 sanitized, reproducible checkpoint staging and theory inputs.

Consumes completed A17 descriptive and clean timing receipts. Copies only
frozen per-image string/CLIP decisions plus stage-level aggregated JSON.
No per-candidate PSNR, GT image, local measurement path, or native tensor.
Refuses release if clean G6 extension is not validated (A15 requirement).
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,subprocess,sys,traceback
from pathlib import Path
from collections import defaultdict
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha,jread,write,csvload
from scripts.track_r.xp_a15_cp3_protocol_cpu import verify
from scripts.track_r.xp_a16_cp3_joint_readout_cpu import (
 PROTOCOL_STAMP,PIQ_STAMP,NFE_STAMP)

STAGES=("G2_0.02","G2_0.10","G4","G5","G3")
ORIG=BASE/("xp_a15_cp3_protocol_"+PROTOCOL_STAMP)
A17_PREFIX="xp_a17_cp3_a15_"
TIMING_PREFIX="xp_a17_cp3_timing_"
EXTERNAL={
 "FFHQ":{
  "DAPS_TABLE1":dict(PSNR=30.63,SSIM=.851,LPIPS=.139),
  "PC_DAPS_REPRODUCED_DAPS":dict(PSNR=29.66,SSIM=.770,LPIPS=.177)},
 "IMAGENET":{
  "DAPS_TABLE1":dict(PSNR=25.78,SSIM=.743,LPIPS=.254),
  "PC_DAPS_REPRODUCED_DAPS":dict(PSNR=20.98,SSIM=.461,LPIPS=.387)},
}
def files_for(stage):
 # Producer A15 scored G2 noise directories as G2_0p02/G2_0p10,
 # whereas the actual frozen stage argument is G2_0.02/G2_0.10.
 # Match xp_a15_cp3_protocol_cpu.verify's exact path contract.
 gate(stage in STAGES,"A17_RELEASE_RECOGNIZED_STAGE")
 return BASE/("xp_a15_cp3_"+stage.replace(".","p")+"_"+PROTOCOL_STAMP)
def latest_one(prefix,filename):
 matches=[]
 for d in BASE.glob(prefix+"*"):
  if d.is_dir() and not d.is_symlink() and (d/filename).is_file():
   matches.append(d)
 gate(len(matches)==1,"A17_RELEASE_UNIQUE_"+prefix)
 return matches[0]
def accepted_timing():
 # An earlier A17 attempt failed its native import; it is not
 # an alternative valid result. Preserve and account for its receipt.
 ok=[];failed=[]
 for d in sorted(BASE.glob(TIMING_PREFIX+"*")):
  p=d/"A17_CP3_CLEAN_TIMING_RECEIPT.json"
  if not d.is_dir() or d.is_symlink() or not p.is_file():continue
  x=jread(p)
  gate(x.get("status") in ("A17_CP3_CLEAN_PAIRED_TIMING_PASS",
                          "A17_CP3_TIMING_PARTIAL_OR_DIRTY"),
       "A17_RELEASE_TIMING_STATUS_UNKNOWN")
  if x["status"]=="A17_CP3_CLEAN_PAIRED_TIMING_PASS":ok.append(d)
  else:failed.append(dict(path=str(p),sha256=sha(p),
       charged_gpu_h=float(x["additional_charged_gpu_hours"]),
       status=x["status"]))
 gate(len(ok)==1,"A17_RELEASE_ONE_VALID_CLEAN_TIMING")
 gate(all(0<=f["charged_gpu_h"]<=3 for f in failed),
      "A17_RELEASE_FAILED_ATTEMPT_COSTS")
 return ok[0],failed

def build_digest(desc,summary,timing):
 boot=desc["E1_minus_native_bootstrap"]["metrics"]
 ps=boot["PSNR_db"];ss=boot["SSIM"];good=boot["GOOD"]
 g4=summary["hypotheses"]["DAPS_4K"]["G4"]
 g5=summary["hypotheses"]["DAPS_4K"]["G5"]
 t=timing["mean_native_sampling_seconds"]
 lines=[
  "# CP3 — frozen protocol and A15 XP-I descriptive checkpoint",
  "Planner prereg A15 @ 817f78a; independently scored and audited; no protocol v2.",
  "CP2 ACCEPTED: N32 H1 FAIL -0.0179, H3 FAIL (1/3), H4 PASS +0.131.",
  "CP2 ACCEPTED: N64 H1 FAIL -0.0344, H3 N/A (one flag), H4 PASS +0.136.",
  "CP2 v4: race selection 18/18 vs S2 1/18; orbit oracle 0/6.",
  "G2 sigma .02 H-G1 FAIL (9.89% tau mismatch); H-G2 PASS, 8.0/8 runs (no savings).",
  "G2 sigma .10 H-G1 FAIL (2.12% tau mismatch); H-G2 PASS.",
  f"G4 FFHQ DAPS4K H1 PASS +3.93pp; success {g4['success']:.4f}, mean runs {g4['mean_runs']:.3f}.",
  f"G5 ImageNet DAPS4K H1 PASS only +0.10pp; success {g5['success']:.4f}, mean runs {g5['mean_runs']:.3f}.",
  "XP-I A12 matched-NFE rule WITHDRAWN, UNRESOLVED (planner acknowledged design error).",
  f"XP-I E1 minus native PSNR: {ps['delta_mean']:+.3f} dB, paired bootstrap 95% [{ps['CI95_lower']:+.3f},{ps['CI95_upper']:+.3f}].",
  f"XP-I E1 minus native SSIM: {ss['delta_mean']:+.4f}, paired 95% [{ss['CI95_lower']:+.4f},{ss['CI95_upper']:+.4f}].",
  f"XP-I E1 minus native GOOD25: {100*good['delta_mean']:+.2f}pp, paired 95% [{100*good['CI95_lower']:+.2f},{100*good['CI95_upper']:+.2f}].",
  "XP-I: PC-DAPS E1 vs DAPS4K run0, fixed E1 K1/K2 and frozen Eprime8: see descriptive JSON.",
  "DAPS4K: 400 annealing x 10 ODE = 4000 nominal NFE/run, 100 Langevin/anneal (native receipts).",
  "PC-DAPS: 1920 nominal NFE/bag, logged mean lower bound 1928.4; exact pruning NFE unknown.",
  f"Clean batch1 native sampling seconds, same GPU/3 images: PC bag {t['PC']:.2f}; DAPS4K run {t['DAPS4K']:.2f}.",
  "A8.6: ImageNet GT/y all 100 exact; per-metric GT best-of4/8 reference table in A8_6.json.",
  "Strings/CLIP decisions for all five panels mirrored to theory_inputs; path/GT tensors excluded.",
  "No matched-compute claim, no frozen protocol changes. CP4 by Nov 15; tail I2 before K."
 ]
 gate(len(lines)<=20,"A17_DIGEST_LINE_LIMIT")
 return "\n".join(lines)+"\n"

def a86(summary):
 raw=summary["daps_GT_per_metric_oracles_not_GT_free"]
 out=dict(status="A15_A8_6_EXACT_IDENTITY_AND_DAPS4K_ORACLES",
   notes=("Our per-metric GT-oracle results are not GT-free outputs; "
          "the external paper values below use different data/evaluations "
          "and are contextual references, not same-id reproduction."),
   imagenet_gt_y_status="100/100 exact reproduction, observed RMS zero",
   our_DAPS4K={},
   external_references=EXTERNAL,
   protocol="raw 000 PSNR from scored RGB8; piq SSIM/LPIPS from original float tensors; independently optimize each metric")
 for stage,ds in (("G4","FFHQ"),("G5","IMAGENET")):
  rows=raw[stage]
  result={}
  for k in (4,8):
   vals={}
   for metric in ("PSNR_rgb8_raw_db","SSIM_piq_float_raw",
                  "LPIPS_VGG_piq_float_raw"):
    sub=[r for r in rows
         if r["rule"]=="GT_PER_METRIC_ORACLE_BEST_"+str(k)
         and r["metric"]==metric]
    gate(len(sub)==(2 if stage=="G4" else 1),"A17_A86_ORACLE_COHORTS")
    vals[metric]=sum(z["mean"]*z["n_images"] for z in sub)/sum(z["n_images"] for z in sub)
   result["best_"+str(k)]=vals
  out["our_DAPS4K"][ds]=result
 return out
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A17_RELEASE_CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *z:subprocess.check_output(["git","-C",str(wt),*z],
            text=True,timeout=60).strip()
 gate(git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"A17_RELEASE_EXECUTOR_HEAD")
 descdir=latest_one(A17_PREFIX,"XP_A17_CP3_A15_DESCRIPTIVE.json")
 timingdir,failed_timing=accepted_timing()
 desc=jread(descdir/"XP_A17_CP3_A15_DESCRIPTIVE.json")
 timing=jread(timingdir/"A17_CP3_CLEAN_TIMING_RECEIPT.json")
 gate(desc["status"]=="A15_CP3_DESCRIPTIVE_EXTENSIONS_AUDITED_NOT_CHECKPOINT"
      and desc["DAPS4K_nominal_NFE_per_run"]==4000
      and desc["A15_XPI_matched_NFE_rule"]=="WITHDRAWN_UNRESOLVED"
      and desc["no_matched_compute_claim"] is True
      and desc["E1_minus_native_bootstrap"]["bootstrap_replicates"]==20000
      and desc["E1_minus_native_bootstrap"]["n_image_pairs"]==180
      and timing["status"]=="A17_CP3_CLEAN_PAIRED_TIMING_PASS"
      and timing["clean_timing_valid"] is True
      and len(timing["measurements"])==6
      and all(z["clean"] and z["rc"]==0 for z in timing["measurements"])
      and timing["DAPS4K_anneal_ODE"]=="400x10"
      and timing["same_gpu_serial"] is True
      and timing["three_same_signed_y_ids"] is True,
      "A17_RELEASE_REQUIRED_EVIDENCE")
 jointfile=BASE/"xp_a16_cp3_joint_20261031T034143Z"/"XP_A16_CP3_JOINT_READOUT.json"
 summary=jread(jointfile)
 gate(summary["status"]=="XP_A16_CP3_INDEPENDENT_JOINT_READOUT_PASS_UNMATCHED_NFE"
      and sha(jointfile)==desc["raw_A16_joint_sha256"]
      and summary["CP3_RETURN_AUTHORIZED"] is False,
      "A17_RELEASE_A16_JOINT")
 audit=jread(ORIG/"XP_A15_CP3_INDEPENDENT_AUDIT.json")
 gate(audit["status"]=="XP_A15_CP3_INDEPENDENT_SCORES_PROTOCOL_AUDIT_PASS_NOT_FINAL"
      and len(audit["frozen_protocol_exact_reproductions"])==7,
      "A17_RELEASE_ORIGINAL_PROTOCOL_AUDIT")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a17_cp3_return_")
      and not out.exists() and not out.is_symlink(),
      "A17_RELEASE_NEW_OUTPUT")
 # Exercise every stage path, frozen CSV receipt, digest and A8.6
 # table before creating a release directory. This catches deterministic
 # publication errors at the CPU preflight gate instead of mid-commit.
 checked={}
 for stage in STAGES:
  rec,S,C,R,strings,clips,runs=verify(files_for(stage),stage)
  gate(S.is_file() and C.is_file() and
       len(strings)==rec["n_images"] and
       len(clips)==rec["n_candidates"],
       "A17_RELEASE_FROZEN_PANEL_"+stage)
  checked[stage]=(len(strings),len(clips))
 for stage in ("G2_0p02","G2_0p10","G4","G5"):
  gate((ORIG/("CP3_"+stage+"_PROTOCOL.json")).is_file(),
       "A17_RELEASE_FROZEN_PROTOCOL_"+stage)
 gate((ORIG/"CP3_G3_PC_DAPS_DESCRIPTIVE.json").is_file(),
      "A17_RELEASE_G3_DESCRIPTIVE")
 digest=build_digest(desc,summary,timing)
 oracle=a86(summary)
 gate(len(digest.splitlines())<=20 and
      set(oracle["our_DAPS4K"])=={"FFHQ","IMAGENET"} and
      all(set(oracle["our_DAPS4K"][ds])=={"best_4","best_8"}
          for ds in ("FFHQ","IMAGENET")),
      "A17_RELEASE_DIGEST_ORACLE_REPRODUCED")
 if a.preflight_only:
  print("A17_CP3_RETURN_PREFLIGHT_PASS",
        "FROZEN_PANELS",checked,
        "DIGEST_LINES",len(digest.splitlines()),
        "A8_6_DAPS4K_4AND8",flush=True)
  return
 out.mkdir(mode=0o700)
 ret=out/"returns";ret.mkdir()
 ti=out/"theory_inputs";ti.mkdir()
 for stage in STAGES:
  sd=files_for(stage)
  rec,S,C,R,strings,clips,runs=verify(sd,stage)
  tag=stage.replace(".","p")
  for p,suffix in ((S,"STRINGS.csv"),(C,"CLIP.csv")):
   target=tag+"_"+suffix
   shutil.copyfile(p,ret/target)
   shutil.copyfile(p,ti/target)
   gate(sha(p)==sha(ret/target)==sha(ti/target),"A17_RELEASE_STAGE_COPY_"+tag)
  print("A17_CP3_THEORY_INPUTS_VERIFIED",stage,len(strings),len(clips),flush=True)
 # Frozen protocol outputs use original filenames. These are unchanged.
 for f in sorted(ORIG.glob("PROTOCOL_*.json")):
  shutil.copyfile(f,ret/f.name)
 for stage in ("G2_0p02","G2_0p10","G4","G5"):
  f=ORIG/("CP3_"+stage+"_PROTOCOL.json")
  gate(f.is_file(),"A17_RELEASE_PROTOCOL_"+stage)
  shutil.copyfile(f,ret/f.name)
 shutil.copyfile(ORIG/"CP3_G3_PC_DAPS_DESCRIPTIVE.json",
                 ret/"CP3_G3_PC_DAPS_DESCRIPTIVE.json")
 shutil.copyfile(descdir/"XP_A17_CP3_A15_DESCRIPTIVE.json",
                 ret/"XP_A17_CP3_A15_DESCRIPTIVE.json")
 # Preserve the original immutable receipt by SHA in staging metadata,
 # but do not publish PAC's local absolute paths to failed-run receipts.
 safe_timing=json.loads(json.dumps(timing))
 for item in safe_timing.get("previous_A17_attempts",[]):
  oldpath=Path(item.pop("path"))
  gate(oldpath.name=="A17_CP3_CLEAN_TIMING_RECEIPT.json"
       and oldpath.parent.parent==BASE
       and oldpath.parent.name.startswith(TIMING_PREFIX)
       and sha(oldpath)==item["sha256"],
       "A17_RELEASE_TIMING_FAILED_PRIOR_SHA")
  item["attempt_run"]=oldpath.parent.name
 write(ret/"A17_CP3_CLEAN_TIMING_RECEIPT.json",safe_timing)
 write(ret/"A8_6_PUBLISHED_PROTOCOL_REFERENCES.json",oracle)
 (ret/"CP3_DIGEST.md").write_text(digest)
 (ret/"README.md").write_text(
   "# A15 CP3 — released under planner disposition\n\n"
   "Planner prereg A15 withdraws XP-I exact matched-NFE hypothesis as "
   "**unresolved**. G2/G4/G5 retain the original frozen v1 protocol. "
   "DAPS-4K native receipts authenticate 400 annealing x 10 ODE x 100 "
   "Langevin; do not confuse with legacy F8 400x5/G6. "
   "PC-DAPS is one four-particle bag, not pixel mean. "
   "This return supplies paired bootstrap and descriptive compute-labeled "
   "comparators; **no matched-compute claim**. "
   "Protocol reexecution must match within 0.002. "
   "Candidate-level GT scores and native tensor paths remain local.\n")
 for folder in (ret,ti):
  names=sorted(p.name for p in folder.iterdir() if p.is_file())
  with (folder/"SHA256SUMS").open("x") as f:
   for name in names:
    f.write(sha(folder/name)+"  "+name+"\n")
 hashes=dict(ret_sha256sum=sha(ret/"SHA256SUMS"),
   theory_sha256sum=sha(ti/"SHA256SUMS"),
   a17_description_sha256=sha(descdir/"XP_A17_CP3_A15_DESCRIPTIVE.json"),
   clean_timing_sha256=sha(timingdir/"A17_CP3_CLEAN_TIMING_RECEIPT.json"),
   protocol_independent_audit_sha256=sha(ORIG/"XP_A15_CP3_INDEPENDENT_AUDIT.json"))
 write(out/"CP3_STAGING_RECEIPT.json",dict(
   status="A15_CP3_SANITIZED_RETURN_STAGED_FOR_COMMIT",
   expected_head=a.expected_head,
   counts=dict(scorings=5840,piq=2960,images=280),
   A15_authorized=True,XP_I_matched_NFE="WITHDRAWN_UNRESOLVED",
   no_matched_NFE_claim=True,
   failed_timing_attempts=failed_timing,
   hashes=hashes,
   paths_are_local_only=dict(returns=str(ret),theory_inputs=str(ti))))
 print("A17_CP3_SANITIZED_RETURN_STAGED",out,flush=True)
 print("A17_CP3_DIGEST_BEGIN\n"+(ret/"CP3_DIGEST.md").read_text()+"A17_CP3_DIGEST_END",flush=True)
 print("A17_CP3_THEORY_INPUTS_FILES",len(list(ti.glob("*.csv"))),flush=True)
 print("A17_CP3_STAGING_RETURN",out,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out")
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A17_CP3_STAGING",e,flush=True)
  sys.exit(8)
