#!/usr/bin/env python3
"""A18 CP-N Part 1: 100 fresh ImageNet images, 16-run panel, v1 protocol.

No GPU. GT ONLY to score existing native candidates; CLIP G1 decision is
computed from output images, not GT. Never opens run >=16 or any v5 predictions.
Source manifests, every chunk and every root reconstruction are SHA-audited.
Frozen v1 c598176, tau 0.0500829 and secondary 0.050434426327793.
"""
from __future__ import annotations
import argparse,csv,json,math,os,subprocess,sys,traceback
from collections import Counter
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE,gate
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,validate_complete
from scripts.track_r.xp_a15_cp2_d4_cpu import (
    sha,jread,write,csvwrite,clip_new,CLIP_NAMES,NAMES,PROTO,git)
from scripts.track_r.xp_a19_xpn_prefix4_cpu import (
    GT_FILE,GT_SHA,PHASE as PILOT,MF as PILOTFILE,pilot_chunk_receipt_valid)
from scripts.track_r.xp_a18_xpn_contract import (
    Y_SHA,AUDIT_SHA,FROZEN_Y,validate_xpn_source)
from prdiffusion.track_r.physics import rho_signed,to_rgb8,load_png_rgb8
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse

TASKS=BASE/"xp_a18_xpn_tasks_20261031T224645Z"
PROD=BASE/"xp_a18_xpn_prod_20261031T225748Z"
P4=BASE/"xp_a19_xpn_prefix4_20261101T194525Z"
P4FILE=P4/"XP_N_RUNS_0_3_A8_STYLE_PER_IMAGE.csv"
C4FILE=P4/"XP_N_RUNS_0_3_G1_DECISIONS.csv"
GOOD=20.
TAU_MAIN=.0500829
TAU_SECONDARY=.050434426327793
SCORE_COLS=("image_id","run","label","rho_signed","psnr_raw",
            "psnr_orbit_best","flip_good","reconstruction_path","reconstruction_sha256")
FORECASTS={
 "P-N1_gain_at_16_pp":(.3,-3.6,4.2),
 "P-N3_label_free_gain_at_16_pp":(.3,-2.0,2.6),
 "P-N4_never_orbit_at_16_pct":(25.,13.,37.),
 "theory_N_H1a_gain_pp":(.2,-1.8,2.2)
}

def table(p):
 with p.open(newline="",encoding="utf-8") as f:
  rs=list(csv.DictReader(f))
 gate(rs and bool(rs[0]),"A22_CPN_NONEMPTY_CSV")
 return rs

def source(a):
 wt=Path(__file__).resolve().parents[2]
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A22_CPN_CPU_ONLY")
 gate(git(wt,"rev-parse","HEAD").decode().strip()==a.expected_head
      and not git(wt,"status","--porcelain"),
      "A22_CPN_CLEAN_HEAD")
 p4receipt=jread(P4/"XP_N_PREFIX4_RECEIPT.json")
 gate(p4receipt["status"]=="A17_XPN_PREFIX4_CLIP4_CPU_SCORED_AND_SHA_FROZEN"
      and p4receipt["n_images"]==100
      and p4receipt["n_pilot_roots"]==400
      and p4receipt["prefix4_sha256"]==sha(P4FILE)
      and p4receipt["clip4_sha256"]==sha(C4FILE)
      and p4receipt["source_gt_y_audit_sha256"]==AUDIT_SHA
      and math.isfinite(float(p4receipt["frozen_good_q95_tau"]))
      and .0<float(p4receipt["frozen_good_q95_tau"])<.2,
      "A22_CPN_FROZEN_REGISTERED_PREFIX")
 # Published input commits were accepted by planner (A18); never refit
 # the tau or use a K=16 panel to test the v5 64-run forecast.
 p4=table(P4FILE);c4=table(C4FILE)
 gate(len(p4)==100 and len(c4)==400
      and len({r["image_id"] for r in p4})==100,
      "A22_CPN_ACCEPTED_PREFIX_UNIVERSE")
 tasks=jread(TASKS/"XP_N_TASKS_PREP_RECEIPT.json")
 gate(tasks["status"]=="A18_XPN_GT_FREE_6400_MANIFESTS_FROZEN_PASS"
      and tasks["n_roots"]==6400 and tasks["n_images"]==100,
      "A22_CPN_FROZEN_6400_GRID")
 manifests={}
 for phase,filename,step,total in (
    (PILOT,PILOTFILE,20,400),
    ("A18_XPN_MAIN_4_15","XP_N_MAIN_4_15_TASKS.json",60,1200)):
  path=TASKS/filename
  d=jread(path);digest=sha(path)
  gate(tasks["phase_files"][phase]["sha256"]==digest
       and d["n_roots"]==total
       and len(d["rows"])==total,
       "A22_CPN_MANIFEST_"+phase)
  for start in range(0,total,step):
   validate_xpn_source(path,digest,d,d["rows"],
                       tasks["expected_head"],start,step)
   chunk=PROD/"production"/phase/f"chunk_{start:05d}_{step:03d}"
   m=jread(chunk/"R0_5_CHUNK_META.json")
   c=jread(chunk/"R0_5_CHUNK_COMPLETE.json")
   ok=(m["head"]==c["head"] and m["root_sha256"]==digest
       and m["start"]==c["start"]==start
       and m["count"]==c["count"]==step
       and c["status"]=="R0_5_CHUNK_PASS"
       and c["new"]>=0 and c["reused"]>=0
       and c["new"]+c["reused"]==step
       and m["uids"]==[r["uid"] for r in d["rows"][start:start+step]]
       and c["meta_sha256"]==sha(chunk/"R0_5_CHUNK_META.json"))
   gate(ok,"A22_CPN_ALL_NATIVE_CHUNKS_"+phase+"_"+str(start))
   for row in d["rows"][start:start+step]:
    native=root_folder(chunk,row)
    rc=validate_complete(native,row,m["head"],digest)
    gate(rc["gt_access"] is False and
         rc["method"]=="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5",
         "A22_CPN_NATIVE_GT_FREE_ROOT")
  manifests[phase]=(d,digest,step)
 gate(sha(GT_FILE)==GT_SHA and sha(FROZEN_Y)==Y_SHA,
      "A22_CPN_GT_Y_SOURCE_SHA")
 gt=jread(GT_FILE);ym=jread(FROZEN_Y)
 gate(len(gt["rows"])==len(ym["rows"])==100,
      "A22_CPN_EXACT_100_GT")
 return wt,p4,c4,manifests,gt,ym,p4receipt

def new_scores(manifests,gt,ym,label_tau):
 import torch
 torch.set_num_threads(4)
 rows,manifest_sha,step=manifests["A18_XPN_MAIN_4_15"]
 gmap={r["basename"]:r for r in gt["rows"]}
 ymap={r["basename"]:r for r in ym["rows"]}
 result=[]
 with torch.inference_mode():
  for i,iid in enumerate(sorted(gmap)):
   g=gmap[iid];y=ymap[iid]
   gate(g["measurement_sha256"]==y["measurement_sha256"]
        and sha(g["gt_path"])==g["gt_file_sha256"]
        and sha(y["measurement_path"])==y["measurement_sha256"],
        "A22_CPN_SOURCE_GT_Y_"+iid)
   gt8=to_rgb8(torch.load(g["gt_path"],map_location="cpu",
                            weights_only=True)["gt_model_range"].float())
   meas=torch.load(y["measurement_path"],map_location="cpu",
                   weights_only=True)["measurement"].float()
   batch=rows["rows"][i*12:(i+1)*12]
   gate([r["image_id"] for r in batch]==[iid]*12
        and [r["rep"] for r in batch]==list(range(4,16)),
        "A22_CPN_PER_IMAGE_4_TO_15")
   for r in batch:
    off=(i//5)*60
    chunk=PROD/"production/A18_XPN_MAIN_4_15"/f"chunk_{off:05d}_060"
    xdir=root_folder(chunk,r)
    receipt=validate_complete(xdir,r,jread(chunk/"R0_5_CHUNK_META.json")["head"],
                              manifest_sha)
    x=torch.load(xdir/"reconstruction.pt",map_location="cpu",
                 weights_only=True)["reconstruction"].float()
    rgb=to_rgb8(x)
    gate(torch.equal(rgb,load_png_rgb8(xdir/"reconstruction.png")),
         "A22_CPN_NATIVE_RGB8_ROUNDTRIP")
    orbit=orbit_psnr_channel_mse(rgb,gt8)
    best=max(sorted(orbit),key=lambda k:orbit[k])
    raw=float(orbit["000"]);orbest=float(orbit[best])
    rho=float(rho_signed(x,meas)[0])
    lab=("G" if raw>=GOOD else
         "O_g" if orbest>=GOOD and best=="111" else
         "O_p" if orbest>=GOOD else
         "C" if rho<=label_tau else "I")
    gate(math.isfinite(rho) and rho>=0
         and math.isfinite(raw) and math.isfinite(orbest),
         "A22_CPN_NATIVE_METRIC_FINITE")
    result.append(dict(image_id=iid,run=int(r["rep"]),label=lab,
       rho_signed=format(rho,".15g"),psnr_raw=format(raw,".12g"),
       psnr_orbit_best=format(orbest,".12g"),
       flip_good=int(float(orbit["111"])>=GOOD),
       reconstruction_path=str(xdir/"reconstruction.png"),
       reconstruction_sha256=receipt["reconstruction_png_sha256"]))
   if (i+1)%10==0:
    print("A22_CPN_GT_EVALUATION",i+1,"/100",flush=True)
 gate(len(result)==1200,"A22_CPN_EXACT_1200_NEW_SCORES")
 return result

def merge(p4,c4,new,clip,label_tau):
 byp={r["image_id"]:r for r in p4}
 byc={(r["image_id"],int(r["run"])):r for r in c4}
 cn={(r["image_id"],int(r["run"])):r for r in clip}
 bynew={(r["image_id"],int(r["run"])):r for r in new}
 gate(len(byp)==100 and len(byc)==400 and len(cn)==1200
      and len(bynew)==1200,"A22_CPN_UNIQUE_GRIDS")
 decoder={"G":"G","O":"O_g","P":"O_p","C":"C","I":"I"}
 strings=[];allclip=[]
 for iid in sorted(byp):
  prefix=byp[iid]
  a=prefix["run_labels"]
  gate(len(a)==4 and set(a)<=set(decoder),
       "A22_CPN_PREFIX_SINGLE_LETTER")
  labs=[decoder[z] for z in a];rho=[
   float(x) for x in prefix["rho_signed_by_run"].split(";")]
  gate(len(rho)==4,"A22_CPN_PREFIX_RESIDUALS")
  for rep in range(4):
   v=byc[iid,rep]
   gate(v["label"]==a[rep] and v["g1_choice"] in ("000","111"),
        "A22_CPN_PREFIX_G1_VALID")
   allclip.append(dict(v,label=decoder[v["label"]]))
  for rep in range(4,16):
   v=bynew[iid,rep];cl=cn[iid,rep]
   gate(cl["label"]==v["label"] and cl["g1_choice"] in ("000","111"),
        "A22_CPN_NEW_G1_VALID")
   labs.append(v["label"]);rho.append(float(v["rho_signed"]))
   allclip.append(cl)
  count=Counter(labs)
  strings.append(dict(image_id=iid,cohort="IN_NEW",
    labels_by_run=";".join(labs),
    rho_signed_by_run=";".join(format(v,".15g") for v in rho),
    tau_signed_good_q95=format(label_tau,".17g"),
    n_G=count["G"],n_O_g=count["O_g"],n_O_p=count["O_p"],
    n_C=count["C"],n_I=count["I"]))
 gate(len(strings)==100 and len(allclip)==1600
      and len({(r["image_id"],int(r["run"])) for r in allclip})==1600,
      "A22_CPN_FULL_100X16")
 return strings,sorted(allclip,key=lambda r:(r["image_id"],int(r["run"])))

def run_protocol(a,wt,out,strings,clip):
 proto=out/"gtfree_protocol_v1_pinned.py"
 blob=git(wt,"show",PROTO[0]+":"+PROTO[1])
 gate(sha_bytes(blob)==PROTO[2],"A22_CPN_PROTOCOL_SHA")
 proto.write_bytes(blob)
 results={}
 for name,tau in (("PRIMARY",TAU_MAIN),("SECONDARY",TAU_SECONDARY)):
  target=out/("XP_N_CPN16_PROTOCOL_"+name+".json")
  cmd=[sys.executable,"-I",str(proto),
       "--strings",str(strings),"--clip",str(clip),
       "--tau",repr(tau),"--m","2","--run-range","0:16",
       "--nmax","16","--R","4000","--seed","20261029",
       "--out",str(target)]
  proc=subprocess.run(cmd,cwd=str(wt),
       env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),
       capture_output=True,text=True,timeout=180)
  gate(proc.returncode==0,
       "A22_CPN_PROTOCOL_RUN_"+name+"_"+proc.stderr[-250:])
  d=jread(target)
  gate(d["protocol_version"]=="v1" and d["run_range"]=="0:16"
       and d["N_MAX"]==d["n_runs_per_image_panel"]==16
       and d["M"]==2 and d["n_images"]==100
       and d["orientation_step"] is True
       and d["inputs"]["strings_sha256"]==sha(strings)
       and d["inputs"]["clip_sha256"]==sha(clip),
       "A22_CPN_PROTOCOL_INVARIANTS_"+name)
  results[name]=d
 return results

def sha_bytes(s):
 import hashlib
 return hashlib.sha256(s).hexdigest()

def summary(results,strings,out,head):
 p=results["PRIMARY"];s=results["SECONDARY"]
 gain=100*float(p["gain_vs_fixed_equal_runs"])
 gain_secondary=100*float(s["gain_vs_fixed_equal_runs"])
 coverage=100*float(p["n_never_orbit_panel"])/100.
 unflagged=p["success_unflagged"]
 answered=p["success_given_not_flag_c"]
 h1=gain>=-1.
 h4=(unflagged is not None and answered is not None
     and unflagged>=answered+.05)
 def forecast(name,measured):
  point,lo,hi=FORECASTS[name]
  return dict(observed=measured,forecast_point=point,lo95=lo,hi95=hi,
              inside=lo<=measured<=hi)
 comparisons=dict(
   P_N1=forecast("P-N1_gain_at_16_pp",gain),
   P_N3=forecast("P-N3_label_free_gain_at_16_pp",gain_secondary),
   P_N4=forecast("P-N4_never_orbit_at_16_pct",coverage),
   theory_N_H1a=forecast("theory_N_H1a_gain_pp",gain))
 # K=16 decomposition: all-16 selected fixed E0/E1 and exact oracles.
 curve={k:float(p["fixed_curves"][k][15])
        for k in ("E0","E1","O_raw","O_orbit")}
 d=dict(status="A18_CPN_PART1_100_IMAGE_16RUN_PASS",
   expected_head=head,n_images=100,n_runs=1600,
   protocol_source_sha256=PROTO[2],protocol_v1_unmodified=True,
   primary_tau=TAU_MAIN,secondary_tau=TAU_SECONDARY,
   label_CI_tau_from_run0_3=float(jread(P4/"XP_N_PREFIX4_RECEIPT.json")["frozen_good_q95_tau"]),
   N_H1a=bool(h1),N_H4_at_16=bool(h4),
   N_H1a_gain_pp=gain,N_H4_unflagged_success=unflagged,
   N_H4_answered_success=answered,
   label_free_secondary_gain_pp=gain_secondary,
   P_N1_N3_N4_and_theory_pre_score=comparisons,
   never_orbit_good_count=int(p["n_never_orbit_panel"]),
   fixed_K16=curve,full_64_run_panel_not_scored=True,
   v5_not_scored_at_K16=True,no_matched_compute_claim=True,
   theoretical_pre_score_N_H1a_point_pp=.2,
   no_held_16_to_63_outcomes_read=True,
   strings_sha256=sha(strings),clip_sha256=sha(out/"XP_N_CPN16_G1_DECISIONS.csv"),
   primary_protocol_sha256=sha(out/"XP_N_CPN16_PROTOCOL_PRIMARY.json"),
   secondary_protocol_sha256=sha(out/"XP_N_CPN16_PROTOCOL_SECONDARY.json"))
 write(out/"XP_N_CPN_PART1_RECEIPT.json",d)
 lines=[
  "XP-N CP-N PART1 / A18 / 100 disjoint ImageNet images, runs 0-15, v1 c598176.",
  "Status: ALL100_N16; v5 not scored on n16; GT evaluation only; no matched NFE.",
  f"N-H1a: {'HELD' if h1 else 'FAILED'} gain {gain:+.3f} pp (criterion >= -1.0 pp).",
  (f"N-H4(K16): {'HELD' if h4 else 'FAILED'} unflagged {unflagged:.5f}; answered {answered:.5f}; margin {(unflagged-answered)*100:+.3f} pp."
   if unflagged is not None and answered is not None else
   "N-H4(K16): FAILED / undefined answered or unflagged rate."),
  f"Secondary tau {TAU_SECONDARY}: gain {gain_secondary:+.3f} pp.",
  f"Orbit never-GOOD at K16: {int(p['n_never_orbit_panel'])}/100 ({coverage:.1f}%).",
 ]
 for k,name in (("P_N1","Planner P-N1 gain16"),
                ("P_N3","Planner P-N3 secondary gain16"),
                ("P_N4","Planner P-N4 never-orbit K16"),
                ("theory_N_H1a","Theory pre-score N-H1a")):
  x=comparisons[k]
  lines.append(f"{name}: {x['observed']:+.3f} vs {x['forecast_point']:+.2f} [{x['lo95']:+.2f}, {x['hi95']:+.2f}] pp; {'INSIDE' if x['inside'] else 'OUTSIDE'}.")
 lines.append("K16 exact fixed/oracle: "+", ".join(
   f"{k}={v:.6f}" for k,v in curve.items()))
 lines.append(f"Protocol SHAs: primary {d['primary_protocol_sha256']}; secondary {d['secondary_protocol_sha256']}.")
 lines.append(f"Inputs: strings {d['strings_sha256']}; G1 {d['clip_sha256']}.")
 gate(len(lines)<=20,"A22_CPN_REPORT_AT_MOST_20_LINES")
 (out/"CPN_PART1_PLANNER_RETURN.txt").write_text("\n".join(lines)+"\n")
 print("A22_CPN_PART1_PASS",json.dumps(dict(N_H1a=h1,N_H4=h4,
      gain_pp=gain,secondary_pp=gain_secondary,
      never_orbit_pct=coverage,forecasts=comparisons),sort_keys=True),flush=True)

def main(a):
 wt,p4,c4,manifests,gt,ym,prec=source(a)
 if a.preflight_only:
  print("A22_CPN_100X16_RECEIPTS_AND_PREFIX_PASS",
        "ROOTS",1600,"GT_Y",AUDIT_SHA,flush=True)
  return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a22_cpn_part1_")
      and not out.exists() and not out.is_symlink(),
      "A22_CPN_NEW_OUT")
 out.mkdir(mode=0o700)
 label_tau=float(prec["frozen_good_q95_tau"])
 new=new_scores(manifests,gt,ym,label_tau)
 clips,clipmeta=clip_new(new,mode="g1",threshold=20)
 gate(len(clips)==1200,"A22_CPN_1200_FROZEN_G1")
 strings,allclip=merge(p4,c4,new,clips,label_tau)
 sp=out/"XP_N_CPN16_STRINGS.csv";cp=out/"XP_N_CPN16_G1_DECISIONS.csv"
 csvwrite(sp,strings,NAMES);csvwrite(cp,allclip,CLIP_NAMES)
 # Keep GT-local raw scores separate, never publish those paths.
 csvwrite(out/"LOCAL_ONLY_CP_N_1200_RAW_SCORES.csv",new,SCORE_COLS)
 results=run_protocol(a,wt,out,sp,cp)
 summary(results,sp,out,a.expected_head)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--out",required=True)
 ap.add_argument("--preflight-only",action="store_true")
 try:main(ap.parse_args())
 except Exception as exc:
  traceback.print_exc()
  print("ACTION NEEDED: A22_CPN_PART1",str(exc),flush=True)
  sys.exit(8)
