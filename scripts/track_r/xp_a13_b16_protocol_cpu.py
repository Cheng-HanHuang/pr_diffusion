#!/usr/bin/env python3
"""A13 CP1: B16 two-cohort protocol-v1, frozen code/inputs, pooled by images.

No protocol edits or retuning. B22 and DEV80 run separately, because
their pre-registered SITCOM rho thresholds differ. Pooled result is
100/180*B22 + 80/180*DEV80. The reference was registered by planner
before B16 outcomes; reproduction tolerance is 0.002.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
STRINGS=BASE/"xp_b16_v1_v3_score_20261029T160718Z/XP_B16_16RUN_STRINGS.csv"
CLIP=BASE/"xp_b16_faceclip_20261029T160718Z/XP_B16_16RUN_FACE_CLIP_DECISIONS.csv"
SOURCE_SHA={
 "strings":"b910b69ec20ef6bb7e0099d4506c1109d0c40d4528a2a3f89f0a3433eb968051",
 "clip":"a5869ba84f56485bbbf98fc2c089688c0eb03cbe7f0493a1324887b84eca77f0",
 "protocol":"d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658"
}
PROTO=Path(__file__).resolve().parents[2]/"docs/track_r/protocol/gtfree_protocol.py"
COHORTS={"B22":(100,0.0448503),"DEV80":(80,0.0449706)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 with Path(p).open("x") as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
              text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"PINNED_WORKTREE")
 gate(sha(STRINGS)==SOURCE_SHA["strings"] and
      sha(CLIP)==SOURCE_SHA["clip"] and
      sha(PROTO)==SOURCE_SHA["protocol"],
      "FROZEN_A11_INPUT_OR_PROTOCOL_SHA")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a13_b16_protocol_")
      and not out.exists() and not out.is_symlink(),"FRESH_OUTPUT")
 out.mkdir(mode=0o700)
 settings=[
   ("primary",True,False,False),
   ("E0",False,False,False),
   ("self_mad3",True,True,False),
   ("frontier",True,False,True)]
 allres={}
 for name,clip,adaptive,frontier in settings:
  results={}
  for cohort,(n,tau) in COHORTS.items():
   path=out/f"B16_{name}_{cohort}.json"
   cmd=[sys.executable,"-I",str(PROTO),
        "--strings",str(STRINGS),"--cohort",cohort,
        "--m","2","--nmax","16","--R","4000",
        "--seed","20261029","--out",str(path)]
   if clip:cmd+=["--clip",str(CLIP),
         "--clip-good-col","f_chosen_good",
         "--clip-margin-col","f_margin"]
   if adaptive:cmd+=["--tau-rule","self_mad3"]
   else:cmd+=["--tau",str(tau)]
   if frontier:cmd.append("--frontier")
   proc=subprocess.run(cmd,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True)
   gate(proc.returncode==0,"PROTOCOL_CHILD_"+name+"_"+cohort+
        "_"+proc.stdout[-1000:])
   z=read(path)
   gate(z["protocol_version"]=="v1"
        and z["n_images"]==n
        and z["N_MAX"]==16 and z["R"]==4000
        and z["orientation_step"] is clip
        and z["baseline"]==("E1" if clip else "E0")
        and z["inputs"]["strings_sha256"]==SOURCE_SHA["strings"],
        "B16_PER_COHORT_PROTOCOL_"+name+"_"+cohort)
   if not adaptive:gate(abs(z["tau"]-tau)<1e-10,"FIXED_TAU")
   if clip:gate(z["inputs"]["clip_sha256"]==SOURCE_SHA["clip"],
                "CLIP_SHA")
   results[cohort]=z
  weighted={}
  common=set(results["B22"])&set(results["DEV80"])
  for k in common:
   v=[results[x][k] for x in COHORTS]
   if all(isinstance(w,(float,int)) and not isinstance(w,bool)
          and math.isfinite(float(w)) for w in v):
    weighted[k]=sum(COHORTS[c][0]/180*float(results[c][k])
                    for c in COHORTS)
  pooled=dict(status="XP_A13_B16_PROTOCOL_V1_POOLED_BY_IMAGES",
    variant=name,source_expected_head=a.expected_head,
    n_images=180,cohorts={c:dict(n=COHORTS[c][0],
      tau=results[c]["tau"],out_sha256=sha(
             out/f"B16_{name}_{c}.json")) for c in COHORTS},
    weighted=weighted,
    frozen_code_sha256=SOURCE_SHA["protocol"],
    string_sha256=SOURCE_SHA["strings"],
    clip_sha256=SOURCE_SHA["clip"] if clip else None,
    frozen_params={"M":2,"N_MAX":16,"R":4000,"seed":20261029,
                   "per_cohort_tau":{c:COHORTS[c][1] for c in COHORTS}})
  if name=="primary":
   gain=weighted["gain_vs_fixed_equal_runs"]
   pooled["H1_PASS"]=gain>=-0.01
   pooled["planner_reference"]={"success_all":0.8186,
      "runs_mean":3.290,"gain_vs_fixed_equal_runs":-0.0043}
   for k,ref in pooled["planner_reference"].items():
    gate(abs(weighted[k]-ref)<0.002,
         "B16_FROZEN_PLANNER_REPRODUCTION_"+k)
  write(out/f"B16_{name}_POOLED.json",pooled)
  allres[name]=pooled
  print("B16_PROTOCOL_VARIANT",name,
        "success",round(weighted.get("success_all",float("nan")),6),
        "runs",round(weighted.get("runs_mean",float("nan")),6),
        "gain",round(weighted.get("gain_vs_fixed_equal_runs",
                                    float("nan")),6),flush=True)
 write(out/"XP_A13_B16_PROTOCOL_COMPLETE.json",dict(
     status="XP_A13_B16_PROTOCOL_V1_FROZEN_COMPLETE",
     source_head=a.expected_head,
     input_source_files={str(STRINGS):sha(STRINGS),str(CLIP):sha(CLIP),
                         str(PROTO):sha(PROTO)},
     frozen_primary=allres["primary"],
     variants={k:sha(out/f"B16_{k}_POOLED.json")
               for k in allres},
     H1_PASS=allres["primary"]["H1_PASS"],
     reproducible_by_planner_within_0p002=True,
     no_GPU=True))
 print("XP_A13_B16_PROTOCOL_V1_REPRODUCED_PASS",flush=True)
 print("B16_PROTOCOL_RESULT_DIR",out,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A13_B16_FROZEN_PROTOCOL",e,flush=True)
  sys.exit(8)
