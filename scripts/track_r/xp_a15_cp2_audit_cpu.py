#!/usr/bin/env python3
"""Independent CP2 audit: reconstruct U-statistics and replay locked protocol.

No GT reread, model inference or GPU. Refuse release on any mismatch,
missing 64-run image, unregistered v4 or inconsistent G1 decisions.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import Counter
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a15_cp2_d4_cpu import (
  PROTO,NAMES,CLIP_NAMES,sha,csvload,jread,write)

def evaluate(strings,clips,n,K):
 by={(r["image_id"],int(r["run"])):r for r in clips}
 summ={k:0. for k in ("E0","E1_CLIP","O_raw","O_orbit")}
 for r in strings:
  lab=r["labels_by_run"].split(";")
  rho=[float(x) for x in r["rho_signed_by_run"].split(";")]
  gate(len(lab)==len(rho)==64,"CP2_AUDIT_N64")
  m=math.comb(n,K)
  rank=sorted(range(n),key=lambda i:(rho[i],i))
  w=[math.comb(n-j-1,K-1)/m if n-j>=K else 0. for j in range(n)]
  successes=[x=="G" for x in lab[:n]]
  orbit=[x in ("G","O_g","O_p") for x in lab[:n]]
  summ["E0"]+=sum(w[j]*successes[rank[j]] for j in range(n))
  summ["E1_CLIP"]+=sum(w[j]*int(by[r["image_id"],rank[j]]["g1_chosen_good"])
                        for j in range(n))
  for key,arr in (("O_raw",successes),("O_orbit",orbit)):
   good=sum(arr)
   summ[key]+=1-(math.comb(n-good,K)/m if n-good>=K else 0.)
 return {k:v/len(strings) for k,v in summ.items()}
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP2_AUDIT_CPU")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *x:subprocess.check_output(["git","-C",str(wt),*x],timeout=90)
 gate(git("rev-parse","HEAD").decode().strip()==a.expected_head
      and not git("status","--porcelain"),"CP2_AUDIT_PIN")
 p=Path(a.score_dir)
 gate(p.parent==BASE and p.name.startswith("xp_a15_cp2_d4_")
      and p.is_dir() and not p.is_symlink(),"CP2_AUDIT_SOURCE")
 rec=jread(p/"XP_A15_CP2_D4_SCORING_RECEIPT.json")
 gate(rec["status"]=="XP_A15_CP2_D4_SCORED_READY_FOR_INDEPENDENT_AUDIT"
      and rec["n_candidates"]==6400 and rec["n_native_new"]==4800
      and rec["v4_registered_before_outcomes"] is True
      and rec["protocol_sha256"]==PROTO[2],"CP2_AUDIT_SOURCE_RECEIPT")
 for n,h in rec["csv_sha256"].items():
  gate(sha(p/n)==h,"CP2_AUDIT_CSV_SHA_"+n)
 S=csvload(p/"D4_64RUN_STRINGS.csv")
 C=csvload(p/"D4_64RUN_G1_CLIP.csv")
 R=csvload(p/"D4_NEW_4800_LOCAL_SCORES.csv")
 gate(len(S)==100 and len(C)==6400 and len(R)==4800
      and len({r["image_id"] for r in S})==100
      and len({(r["image_id"],int(r["run"])) for r in C})==6400
      and len({(r["image_id"],int(r["run"])) for r in R})==4800,
      "CP2_AUDIT_64_GRID")
 index={(r["image_id"],int(r["run"])):r for r in C}
 for r in S:
  iid=r["image_id"];labels=r["labels_by_run"].split(";")
  rho=[float(x) for x in r["rho_signed_by_run"].split(";")]
  gate(len(labels)==len(rho)==64 and all(math.isfinite(z) and z>=0 for z in rho),
       "CP2_AUDIT_64_LABELS")
  ct=Counter(labels)
  for k in ("G","O_g","O_p","C","I"):
   gate(int(r["n_"+k])==ct[k],"CP2_AUDIT_LABEL_COUNTS")
  for i in range(64):
   c=index[iid,i]
   gate(c["label"] in ("G","O","P","C","I","O_g","O_p"),
        "CP2_AUDIT_CLIP_LABEL")
   margin=float(c["g1_margin"])
   gate(math.isfinite(margin)
        and c["g1_choice"]==("111" if margin>0 else "000")
        and int(c["g1_chosen_good"]) in (0,1),
        "CP2_AUDIT_CLIP_CHOICE")
 local={(r["image_id"],int(r["run"])):r for r in R}
 for key,r in local.items():
  g=index[key]
  gate(16<=key[1]<64 and r["label"]==g["label"]
       and int(g["g1_chosen_good"])==
       (int(r["flip_good"]) if g["g1_choice"]=="111" else
        int(float(r["psnr_raw"])>=20)),"CP2_AUDIT_G1_GOOD_IDENTITY")
 forecast=jread(p/"CP2_V2_V2E1_V4_FORECAST_COMPARISON.json")
 gate(sha(p/"CP2_V2_V2E1_V4_FORECAST_COMPARISON.json")==
      rec["forecast_comparison_sha256"],"CP2_AUDIT_FORECAST_SHA")
 report={}
 for version in ("v2","v2 E1","v4"):
  block=forecast[version];inside=0
  gate(block["n"]==len(block["comparisons"]),"CP2_AUDIT_FORECAST_ROW_COUNT")
  for item in block["comparisons"]:
   n=int(item["panel_n_max"]) if "panel_n_max" in item else (
        16 if version!="v4" else
        32 if item["panel"]=="primary" else 64)
   val=evaluate(S,C,n,int(item["K"]))[item["policy"]]
   gate(abs(val-float(item["observed"]))<1e-10,"CP2_AUDIT_FORECAST_VALUE")
   band=float(item["lo95"])<=val<=float(item["hi95"])
   gate(band==item["inside"],"CP2_AUDIT_FORECAST_COVERAGE")
   inside+=band
  gate(inside==block["covered"],"CP2_AUDIT_FORECAST_COUNTS")
  report[version]=dict(coverage=inside,n=block["n"])
 # Replay exact frozen executable from the previously staged bytes.
 script=p/"gtfree_protocol_v1_original.py"
 gate(sha(script)==PROTO[2] and
      hashlib.sha256(git("show",PROTO[0]+":"+PROTO[1])).hexdigest()==PROTO[2],
      "CP2_AUDIT_PROTOCOL_SHA")
 proto={}
 for n in (32,64):
  expected=jread(p/f"D4_PROTOCOL_N{n}.json")
  cmd=[sys.executable,"-I",str(script),"--strings",str(p/"D4_64RUN_STRINGS.csv"),
    "--clip",str(p/"D4_64RUN_G1_CLIP.csv"),"--tau","0.0500829","--m","2",
    "--nmax",str(n),"--R","4000","--seed","20261029"]
  # Replay frozen protocol into a fresh local output file, not stdout.
  # The independent audit needs one exact execution per panel.
  temp=p/f"CP2_AUDIT_REPLAY_N{n}.json"
  if n==32:cmd+=["--run-range","0:32"]
  child=subprocess.run(cmd+["--out",str(temp)],capture_output=True,text=True)
  gate(child.returncode==0,"CP2_AUDIT_REPLAY_"+str(n))
  doc=jread(temp)
  for key in ("success_all","runs_mean","gain_vs_fixed_equal_runs",
              "flag_c_expected_count","flag_c_precision_unsolved",
              "success_unflagged","success_given_not_flag_c"):
   v,w=expected.get(key),doc.get(key)
   gate((v is None and w is None) or
        (v is not None and w is not None and abs(float(v)-float(w))<1e-12),
        "CP2_AUDIT_REPEAT_"+str(n)+"_"+key)
  proto[str(n)]=dict(status="EXACT_REPEAT_PASS",sha256=sha(temp))
  temp.unlink()
 out=dict(status="XP_A15_CP2_INDEPENDENT_AUDIT_PASS",
    expected_head=a.expected_head,n_images=100,n_candidates=6400,
    v4_registered_before_outcomes=True,all_forecast_rows_recomputed=True,
    forecast_coverage=report,
    frozen_protocol_replayed=proto,
    source_receipt_sha256=sha(p/"XP_A15_CP2_D4_SCORING_RECEIPT.json"),
    GT_not_used_by_policy=True,
    no_GPU=True,CP2_ready_for_sanitized_commit=True)
 write(p/"XP_A15_CP2_INDEPENDENT_AUDIT.json",out)
 print("CP2_INDEPENDENT_AUDIT_PASS_READY_FOR_CHECKPOINT_COMMIT",flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--score-dir",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A15_CP2_INDEPENDENT_AUDIT",e,flush=True);sys.exit(8)
