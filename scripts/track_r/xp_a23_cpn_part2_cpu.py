#!/usr/bin/env python3
"""A18 CP-N Part 2: full 100x64 native ImageNet-disjoint GT evaluation.

Preconditions:
- A22 native generation full 90/90, SHA-checked.
- CP-N Part1 full 100x16 SHA-accepted (never rescore or change prefix).
- v5 already frozen and registered BEFORE any held candidate was scored.

Scores ONLY previously unscored runs 16..63; the 0..15 strings and G1
choices are reused byte-for-byte from the Part1 accepted input receipts.
Native GT is accessed only for offline quality evaluation; G1 decisions
are the originally frozen zero-shot CLIP rule. No new solver GPU run.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,validate_complete
from scripts.track_r.xp_a15_cp2_d4_cpu import (
 sha,jread,write,csvwrite,clip_new,CLIP_NAMES,NAMES,PROTO,git)
from scripts.track_r.xp_a22_cpn_part1_cpu import (
 TASKS,PROD,GT_FILE,GT_SHA,FROZEN_Y,Y_SHA,AUDIT_SHA,
 TAU_MAIN,TAU_SECONDARY,GOOD,SCORE_COLS,table)
from scripts.track_r.xp_a18_xpn_contract import (
 validate_xpn_source,PHASES)
from scripts.track_r.xp_a20_xpn_recover_gpu import check_chunk
from scripts.track_r.xp_a22_xpn_continue_gpu import job_id
from prdiffusion.track_r.physics import rho_signed,to_rgb8,load_png_rgb8
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse

PART1=BASE/"xp_a22_cpn_part1_20261102T013211Z"
A22=BASE/"xp_a22_xpn_continue_20261102T013122Z"
V5=BASE/"xp_a19_xpn_v5_20261101T194525Z"/"TH1_PREDICTIONS_v5_XP_N.json"
V5_SHA="a09f553292947f4cf9c0f95294a9afd205938f0dc18dd75f4d6ff655916a3ee3"
V5_COMMIT="55ba86b71cafbd8eb066c777ac141823d9cd9a1d"
V5_REPO="docs/track_r/executors/xp/returns/XP_N_V5_20261101T194525Z/TH1_PREDICTIONS_v5_XP_N.json"
V5_BRANCH="codex/prdiff-track-r-xp-n-v5-20261101T194525Z"
S16="XP_N_CPN16_STRINGS.csv"
C16="XP_N_CPN16_G1_DECISIONS.csv"
LATE="A18_XPN_LATE_16_63"
LATEFILE="XP_N_LATE_16_63_TASKS.json"
GRIDS=(4,8,16,32,64)
VARIANTS=("race","S2","tail","race_tau1")
P_N2=(-3.4,-7.8,.9)
P_N4=(22.,11.,33.)
THEORY_N_H1B=-2.7
SOURCE16_SHA="b0ef31bccce5bafe293d97b1ec15e395ab0ba9abb3b7a968b7035fb7a8a227d1"
CLIP16_SHA="3420633a0961574d627d640c4d8c5125e48b60462a0a238d66766301188b70d8"

def approved_inputs(a):
 wt=Path(__file__).resolve().parents[2]
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A23_CPU_ONLY")
 gate(git(wt,"rev-parse","HEAD").decode().strip()==a.expected_head
      and not git(wt,"status","--porcelain"),"A23_CLEAN_HEAD")
 # Exact full native lineage. No complete-case restriction applies.
 final=jread(A22/"XP_N_CONTINUATION_FINAL_STATUS.json")
 gate(final["status"]=="A22_XPN_ALL6400_NATIVE_GENERATION_COMPLETE"
      and final["complete_chunks"]==90
      and final["accepted_full_chunk_roots"]==6400
      and final["stage_total_gpu_h"]<28.
      and final["project_total_gpu_h"]<400.
      and final["native_science_seeds_manifests_unchanged"]
      and final["no_GT_or_scoring_or_predictions_read"]
      and final["prospective_v5_already_registered"],
      "A23_A22_6400_NATIVE_AND_BUDGET")
 s=jread(A22/"XP_N_CONTINUATION_STATE.json")
 gate(s["status"]==final["status"] and len(s["done"])==90
      and not s["active"] and not s["failures"] and s["smoke_pass"],
      "A23_A22_FINAL_90_CHUNK")
 p=jread(PART1/"XP_N_CPN_PART1_RECEIPT.json")
 gate(p["status"]=="A18_CPN_PART1_100_IMAGE_16RUN_PASS"
      and p["n_images"]==100 and p["n_runs"]==1600
      and p["v5_not_scored_at_K16"] is True
      and p["protocol_source_sha256"]==PROTO[2]
      and p["strings_sha256"]==SOURCE16_SHA
      and p["clip_sha256"]==CLIP16_SHA
      and sha(PART1/S16)==SOURCE16_SHA
      and sha(PART1/C16)==CLIP16_SHA,
      "A23_CPN_PART1_SOURCE_SHA")
 s16=table(PART1/S16);c16=table(PART1/C16)
 gate(len(s16)==100 and len(c16)==1600 and
      len({r["image_id"] for r in s16})==100 and
      len({(r["image_id"],int(r["run"])) for r in c16})==1600,
      "A23_N16_UNIVERSE")
 # Verify actual frozen prospective forecast content, not a rewritten file.
 gate(sha(V5)==V5_SHA,"A23_V5_SEALED_SHA")
 # Ensure the remote registered bytes match those present locally.
 try:
  blob=git(wt,"show",V5_COMMIT+":"+V5_REPO)
 except subprocess.CalledProcessError:
  subprocess.run(["git","-C",str(wt),"fetch","--no-tags",
    "origin",V5_BRANCH],check=True,timeout=90)
  blob=git(wt,"show",V5_COMMIT+":"+V5_REPO)
 gate(hashlib.sha256(blob).hexdigest()==V5_SHA and
      blob==V5.read_bytes(),"A23_V5_REGISTERED_REMOTE_BYTES")
 v5=jread(V5)
 gate(v5["status"]=="FROZEN_BEFORE_XP_N_HELD_RUNS_UNBLINDING"
      and v5["version"]=="TH1_PREDICTIONS_v5_XP_N"
      and v5["n_max"]==64 and v5["n_images"]==100
      and v5["K_grid"]==list(GRIDS) and len(v5["predictions"])==100
      and v5["source_restrictions"]["held_runs_read"] is False
      and v5["code_commit"]==
          "81003cc5a9167def74a04281501b4eae2837f44c",
      "A23_V5_PROSPECTIVE_REGISTERED")
 manifest=TASKS/LATEFILE
 d=jread(manifest);digest=sha(manifest)
 prep=jread(TASKS/"XP_N_TASKS_PREP_RECEIPT.json")
 gate(prep["phase_files"][LATE]["sha256"]==digest
      and len(d["rows"])==4800 and d["n_roots"]==4800,
      "A23_LATE_FROZEN_MANIFEST")
 for start in range(0,4800,96):
  validate_xpn_source(manifest,digest,d,d["rows"],
      prep["expected_head"],start,96)
  j=dict(phase=LATE,start=start,count=96)
  chk=check_chunk(PROD,j,(manifest,digest,d["rows"]),
                  final["original_execution_head"],require_complete=True)
  gate(chk["status"]=="COMPLETE"
       and chk["chunk_sha256"]==s["done"][job_id(j)],
       "A23_LATE_VERIFIED_CHUNK_"+str(start))
 gate(sha(GT_FILE)==GT_SHA and sha(FROZEN_Y)==Y_SHA,
      "A23_GT_Y_SHA")
 gt=jread(GT_FILE);ym=jread(FROZEN_Y)
 gate(len(gt["rows"])==len(ym["rows"])==100,
      "A23_GT_Y_100")
 return wt,p,s16,c16,v5,d,digest,gt,ym,final

def score_late(d,digest,gt,ym,execution_head,label_tau):
 import torch
 torch.set_num_threads(4)
 gmap={r["basename"]:r for r in gt["rows"]}
 ymap={r["basename"]:r for r in ym["rows"]}
 results=[]
 with torch.inference_mode():
  for i,iid in enumerate(sorted(gmap)):
   g=gmap[iid];yr=ymap[iid]
   gate(g["measurement_sha256"]==yr["measurement_sha256"]
        and sha(g["gt_path"])==g["gt_file_sha256"]
        and sha(yr["measurement_path"])==yr["measurement_sha256"],
        "A23_GT_Y_PER_IMAGE_SHA")
   rgb_gt=to_rgb8(torch.load(g["gt_path"],map_location="cpu",
                   weights_only=True)["gt_model_range"].float())
   y=torch.load(yr["measurement_path"],map_location="cpu",
                   weights_only=True)["measurement"].float()
   rows=d["rows"][i*48:(i+1)*48]
   gate(len(rows)==48
        and [x["image_id"] for x in rows]==[iid]*48
        and [x["rep"] for x in rows]==list(range(16,64)),
        "A23_LATE_IMAGE_GRID")
   offset=(i//2)*96
   folder=PROD/"production"/LATE/f"chunk_{offset:05d}_096"
   for row in rows:
    p=root_folder(folder,row)
    rec=validate_complete(p,row,execution_head,digest)
    x=torch.load(p/"reconstruction.pt",map_location="cpu",
                 weights_only=True)["reconstruction"].float()
    im=to_rgb8(x)
    gate(torch.equal(im,load_png_rgb8(p/"reconstruction.png")),
         "A23_NATIVE_RGB8_IDENTITY")
    orbit=orbit_psnr_channel_mse(im,rgb_gt)
    best=max(sorted(orbit),key=lambda key:orbit[key])
    raw=float(orbit["000"]);orb=float(orbit[best])
    rho=float(rho_signed(x,y)[0])
    lab=("G" if raw>=GOOD else "O_g" if orb>=GOOD and best=="111"
         else "O_p" if orb>=GOOD else "C" if rho<=label_tau else "I")
    gate(math.isfinite(rho) and rho>=0 and math.isfinite(raw)
         and math.isfinite(orb),"A23_METRIC_FINITE")
    results.append(dict(image_id=iid,run=int(row["rep"]),label=lab,
      rho_signed=format(rho,".15g"),psnr_raw=format(raw,".12g"),
      psnr_orbit_best=format(orb,".12g"),
      flip_good=int(float(orbit["111"])>=GOOD),
      reconstruction_path=str(p/"reconstruction.png"),
      reconstruction_sha256=rec["reconstruction_png_sha256"]))
   if (i+1)%10==0:
    print("A23_CPN_GT_EVALUATION",i+1,"/100",flush=True)
 gate(len(results)==4800,"A23_4800_LATE_SCORING")
 return results

def merge_64(s16,c16,late,clate,label_tau):
 by16={r["image_id"]:r for r in s16}
 bylate={(r["image_id"],int(r["run"])):r for r in late}
 cbase={(r["image_id"],int(r["run"])):r for r in c16}
 cnew={(r["image_id"],int(r["run"])):r for r in clate}
 gate(len(by16)==100 and len(bylate)==4800 and
      len(cbase)==1600 and len(cnew)==4800,"A23_GRIDS_UNIQUE")
 strings=[];allclip=[]
 for iid in sorted(by16):
  old=by16[iid]
  labels=old["labels_by_run"].split(";")
  rhos=old["rho_signed_by_run"].split(";")
  gate(len(labels)==len(rhos)==16,"A23_K16_PREFIX_INTACT")
  for k in range(16):
   row=cbase[iid,k]
   gate(row["label"]==labels[k],"A23_PREFIX_LABEL_CLIP")
   allclip.append(row)
  for k in range(16,64):
   row=bylate[iid,k];cl=cnew[iid,k]
   gate(row["label"]==cl["label"]
        and cl["g1_choice"] in ("000","111"),
        "A23_LATE_LABEL_CLIP")
   labels.append(row["label"]);rhos.append(row["rho_signed"])
   allclip.append(cl)
  count=Counter(labels)
  strings.append(dict(image_id=iid,cohort="IN_NEW",
      labels_by_run=";".join(labels),
      rho_signed_by_run=";".join(rhos),
      tau_signed_good_q95=format(label_tau,".17g"),
      n_G=count["G"],n_O_g=count["O_g"],n_O_p=count["O_p"],
      n_C=count["C"],n_I=count["I"]))
 gate(len(strings)==100 and
      all(len(x["labels_by_run"].split(";"))==64 for x in strings)
      and len(allclip)==6400
      and len({(x["image_id"],int(x["run"])) for x in allclip})==6400,
      "A23_EXACT100_X64")
 return strings,sorted(allclip,key=lambda x:(x["image_id"],int(x["run"])))

def run_protocol(wt,out,sp,cp):
 import sys
 proto=out/"gtfree_protocol_v1_pinned.py"
 blob=git(wt,"show",PROTO[0]+":"+PROTO[1])
 gate(hashlib.sha256(blob).hexdigest()==PROTO[2],
      "A23_FROZEN_PROTOCOL_SOURCE_SHA")
 proto.write_bytes(blob)
 results={}
 for kind,tau in (("PRIMARY",TAU_MAIN),("SECONDARY",TAU_SECONDARY)):
  target=out/("XP_N_CPN64_PROTOCOL_"+kind+".json")
  cmd=[sys.executable,"-I",str(proto),
       "--strings",str(sp),"--clip",str(cp),
       "--tau",str(tau),"--m","2","--run-range","0:64",
       "--nmax","64","--R","4000","--seed","20261029",
       "--out",str(target)]
  if kind=="PRIMARY":cmd.append("--frontier")
  p=subprocess.run(cmd,cwd=str(wt),
      env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),
      capture_output=True,text=True,timeout=1800)
  gate(p.returncode==0,
       "A23_PROTOCOL_"+kind+"_"+p.stderr[-350:])
  d=jread(target)
  gate(d["protocol_version"]=="v1" and d["run_range"]=="0:64"
       and d["N_MAX"]==d["n_runs_per_image_panel"]==64
       and d["n_images"]==100 and d["M"]==2
       and d["orientation_step"] is True
       and d["inputs"]["strings_sha256"]==sha(sp)
       and d["inputs"]["clip_sha256"]==sha(cp)
       and (len(d.get("frontier",[]))==12 if kind=="PRIMARY" else True),
       "A23_PROTOCOL_FULL_PANEL_"+kind)
  results[kind]=d
 return results

def v5_comparison(v5,p):
 curves=p["fixed_curves"]
 observed={(k,pol):float(curves[pol][k-1])
           for k in GRIDS for pol in ("E0","E1","O_raw","O_orbit")}
 rows=[]
 for r in v5["predictions"]:
  k=int(r["K"])
  pol=r["policy"]
  native="E1" if pol=="E1_CLIP" else pol
  obs=observed[k,native]
  lo=float(r["lo95"]);hi=float(r["hi95"]);pt=float(r["point"])
  scale=max((hi-lo)/3.92,.005)
  rows.append(dict(model_variant=r["model_variant"],K=k,
    policy=pol,decision_model=r["decision_model"],
    observed=obs,forecast=pt,lo95=lo,hi95=hi,
    inside=lo<=obs<=hi,
    abs_standardized_residual=abs(obs-pt)/scale))
 gate(len(rows)==100,"A23_V5_ALL_100_PREDICTIONS")
 primary=[r for r in rows if r["model_variant"]=="race"
          and (r["policy"]=="E0" or
               r["policy"]=="E1_CLIP" and r["decision_model"]=="beta_class")]
 gate(len(primary)==10,"A23_V5_PRIMARY_10")
 summaries={}
 for model in VARIANTS:
  relevant=[r for r in rows if r["model_variant"]==model and
            (r["policy"]=="E0" or
            r["policy"]=="E1_CLIP" and r["decision_model"]=="beta_class")]
  gate(len(relevant)==10,"A23_V5_VARIANT_TEN_"+model)
  summaries[model]=dict(primary_e0_e1_coverage=sum(r["inside"] for r in relevant),
    n=10,mean_abs_standardized_residual=sum(
      r["abs_standardized_residual"] for r in relevant)/10)
 best=min(summaries,key=lambda x:(-summaries[x]["primary_e0_e1_coverage"],
      summaries[x]["mean_abs_standardized_residual"],VARIANTS.index(x)))
 return rows,summaries,best

def summarize(out,p,secondary,v5rows,variants,winner,strings):
 gain=100*p["gain_vs_fixed_equal_runs"]
 gain2=100*secondary["gain_vs_fixed_equal_runs"]
 unflag=p["success_unflagged"];ans=p["success_given_not_flag_c"]
 h1b=gain < -1.0
 h4=unflag is not None and ans is not None and unflag>=ans+.05
 never=int(p["n_never_orbit_panel"])
 # Partial-orbit-only images are unrecoverable by 000/111 G1, but may be
 # recoverable with a larger orbit. Keep this diagnostic separate from
 # any stronger causal attribution of all E1-oracle losses.
 partial_only=sum(
    ("O_p" in row["labels_by_run"].split(";")
     and not any(q in ("G","O_g") for q in row["labels_by_run"].split(";")))
    for row in strings)
 p5_abs=100*partial_only/100.
 obs_N2=gain
 obs_N4=float(never)
 cmp2=dict(observed_pp=obs_N2,point_pp=P_N2[0],
   lo95_pp=P_N2[1],hi95_pp=P_N2[2],
   inside=P_N2[1]<=obs_N2<=P_N2[2])
 cmp4=dict(observed_pct=obs_N4,point_pct=P_N4[0],
   lo95_pct=P_N4[1],hi95_pct=P_N4[2],
   inside=P_N4[1]<=obs_N4<=P_N4[2])
 curve64={key:float(p["fixed_curves"][key][63])
          for key in ("E0","E1","O_raw","O_orbit")}
 full=dict(status="A18_XPN_CPN_PART2_ALL100_N64_SCORED",
   panel="100 disjoint ImageNet images x 64 native runs",
   N_H1b=bool(h1b),N_H4_at64=bool(h4),
   primary_gain_pp=gain,secondary_gain_pp=gain2,
   N_H4_unflagged_success=unflag,N_H4_answered_success=ans,
   P_N2=cmp2,P_N4_K64=cmp4,
   P_N5_partial_orbit_only_images=partial_only,
   P_N5_partial_orbit_only_absolute_pp=p5_abs,
   P_N5_partial_only_diagnostic_not_full_gap_attribution=True,
   theory_pre_score_H1b_point_pp=THEORY_N_H1B,
   theory_pre_score_H1b_direction_pass=THEORY_N_H1B < -1.,
   theory_pre_score_H1b_abs_error_pp=abs(gain-THEORY_N_H1B),
   K64_fixed=curve64,
   K64_generation_failure_pct=100*(1-curve64["O_orbit"]),
   K64_selection_loss_E0_pp=100*(curve64["O_raw"]-curve64["E0"]),
   K64_selection_loss_E1_pp=100*(curve64["O_orbit"]-curve64["E1"]),
   v5_registered_sha256=V5_SHA,
   v5_primary_variant="race",v5_primary_winner=winner,
   v5_variant_scores=variants,
   v5_all_K_prediction_rows=len(v5rows),
   full100_K64=True,no_missingness_deviation=True,
   protocol_v1_unmodified=True,no_matched_compute_claim=True,
   n_images=100,n_native_roots=6400,local_GT_only_for_offline_labels=True,
   GT_free_G1_orientations=True,stage_cap_gpu_h=28.)
 write(out/"XP_N_CPN_PART2_RECEIPT.json",full)
 lines=[
  "CP-N PART2 / A18 / 100 new ImageNet images, 64 native FAST runs each; full panel.",
  f"Native generation 90/90; v5 pre-registered SHA {V5_SHA}.",
  f"N-H1b: {'HELD' if h1b else 'FAILED'} gain {gain:+.3f} pp (< -1.0 pp).",
  (f"N-H4 K64: {'HELD' if h4 else 'FAILED'}, unflagged {unflag:.5f}, answered {ans:.5f}; margin {(unflag-ans)*100:+.3f} pp."
   if unflag is not None and ans is not None else
   "N-H4 K64: FAILED / undefined unflagged or answered rate."),
  f"Secondary label-free tau {TAU_SECONDARY} gain {gain2:+.3f} pp.",
  f"P-N2 gain64: {obs_N2:+.3f} pp vs -3.40 [-7.80,+0.90]: {'IN' if cmp2['inside'] else 'OUT'}.",
  f"P-N4 never-orbit K64: {never}/100 vs 22 [11,33]: {'IN' if cmp4['inside'] else 'OUT'}.",
  f"P-N5 partial-orbit-only unrecoverable 000/111 images: {partial_only}/100 ({p5_abs:.2f} pp); separate from total E1-oracle gap.",
  f"Theory N-H1b pre-score: predicted {THEORY_N_H1B:+.2f} pp, observed {gain:+.3f} pp.",
  "K64 fixed/oracles: "+", ".join(f"{k}={v:.6f}" for k,v in curve64.items())+".",
  f"K64 decomposition: generation failure {full['K64_generation_failure_pct']:.3f} pp, E0 selection loss {full['K64_selection_loss_E0_pp']:.3f} pp, E1 selection loss {full['K64_selection_loss_E1_pp']:.3f} pp.",
  "V5 primary race E0+E1 beta_class coverage "+
       f"{variants['race']['primary_e0_e1_coverage']}/10, standardized error "+
       f"{variants['race']['mean_abs_standardized_residual']:.4f}.",
  f"V5 winner (primary coverage then standardized error): {winner}.",
  "V5 variants: "+", ".join(f"{model}: {rec['primary_e0_e1_coverage']}/10, err {rec['mean_abs_standardized_residual']:.3f}" for model,rec in variants.items())+".",
  "v5 at all K: K=4,8,16,32,64 measured on the COMPLETE 64-run panel (not the n16 panel).",
  "Frozen protocol v1 primary tau 0.0500829, M2, nmax64, R4000; 12-point frontier recorded.",
  "No matched-compute claim; GT used only for offline evaluation, G1 decisions GT-free."
 ]
 gate(len(lines)<=20,"A23_PART2_REPORT_20_LINE")
 (out/"CPN_PART2_PLANNER_RETURN.txt").write_text("\n".join(lines)+"\n")
 print("A23_CPN_PART2_PASS",json.dumps(dict(
    N_H1b=h1b,N_H4=h4,gain_pp=gain,
    P_N2_inside=cmp2["inside"],P_N4_inside=cmp4["inside"],
    v5_primary_coverage=variants["race"]["primary_e0_e1_coverage"],
    v5_winner=winner),sort_keys=True),flush=True)

def main(a):
 wt,p1,s16,c16,v5,manifest,digest,gt,ym,native=approved_inputs(a)
 if a.preflight_only:
  print("A23_CPN_FULL100_K64_PREFLIGHT_PASS",
        "ROOTS",6400,"PART1",SOURCE16_SHA,
        "V5",V5_SHA,"STAGE_H",native["stage_total_gpu_h"],flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a23_cpn_part2_")
      and not out.exists() and not out.is_symlink(),
      "A23_NEW_OUTPUT_DIR")
 out.mkdir(mode=0o700)
 label_tau=float(p1["label_CI_tau_from_run0_3"])
 late=score_late(manifest,digest,gt,ym,native["original_execution_head"],
                 label_tau)
 clate,clipmeta=clip_new(late,mode="g1",threshold=20)
 gate(len(clate)==4800,"A23_CLIP_G1_4800")
 strings,clips=merge_64(s16,c16,late,clate,label_tau)
 sp=out/"XP_N_CPN64_STRINGS.csv"
 cp=out/"XP_N_CPN64_G1_DECISIONS.csv"
 csvwrite(sp,strings,NAMES);csvwrite(cp,clips,CLIP_NAMES)
 csvwrite(out/"LOCAL_ONLY_XPN_LATE_4800_GT_SCORES.csv",late,SCORE_COLS)
 ps=run_protocol(wt,out,sp,cp)
 rows,variants,winner=v5_comparison(v5,ps["PRIMARY"])
 write(out/"XP_N_V5_PREDICTION_COMPARISON.json",dict(
  status="A18_V5_100_REGISTERED_PROSPECTIVE_ROWS_EVALUATED",
  original_registration_commit=V5_COMMIT,
  frozen_v5_sha256=V5_SHA,
  measured_from_complete64_run_panel=True,
  n_images=100,n_predictions=len(rows),
  comparison_rule=v5["comparison_rule"],
  variants=variants,primary_winner=winner,
  prediction_rows=rows))
 summarize(out,ps["PRIMARY"],ps["SECONDARY"],rows,
           variants,winner,strings)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--out",required=True)
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as exc:
  traceback.print_exc()
  print("ACTION NEEDED: A23_CPN_PART2",str(exc),flush=True)
  sys.exit(8)
