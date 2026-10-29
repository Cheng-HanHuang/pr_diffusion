#!/usr/bin/env python3
"""A13 CP1 F8 frozen protocol, 8-run two-cohort 100/80 image pool.

Runs exactly the SHA-pinned code of planner c598176 without editing
it, with the registered DAPS 400x5 tau for B22 and DEV80 separately.
H1/H3 tested on F8; H2 is compared to committed B16 and D1 values.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
PROTO_SHA="d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658"
TAU={"B22":(100,0.0474649),"DEV80":(80,0.0475986)}
D1_GAIN=0.0033265974759615347
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,doc):
 with p.open("x") as f:json.dump(doc,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def run(a):
 sc=Path(a.score_dir);cl=Path(a.clip_dir);bc=Path(a.b16_protocol)
 gate(sc.parent==BASE and sc.name.startswith("xp_a13_f8_score_")
      and cl.parent==BASE and cl.name.startswith("xp_a13_f8_clip_")
      and bc.parent==BASE and bc.name.startswith("xp_a13_b16_protocol_"),
      "INPUT_PATH")
 score=read(sc/"XP_A13_F8_SCORING_RECEIPT.json")
 clip=read(cl/"XP_A13_F8_FACECLIP_RECEIPT.json")
 strings=sc/"XP_A13_F8_8RUN_STRINGS.csv"
 decisions=cl/"XP_A13_F8_8RUN_FACE_CLIP_DECISIONS.csv"
 gate(score["status"]=="XP_A13_F8_1440_CPU_SCORED_PASS"
      and clip["status"]=="XP_A13_F8_1440_N2_FACE_CLIP_PASS"
      and sha(strings)==score["strings_csv_sha256"]
      and sha(decisions)==clip["clip_csv_sha256"]
      and clip["score_sha256"]==score["scoring_csv_sha256"],
      "FROZEN_F8_SOURCE")
 b16=read(bc/"B16_primary_POOLED.json")
 gate(b16["status"]=="XP_A13_B16_PROTOCOL_V1_POOLED_BY_IMAGES"
      and b16["variant"]=="primary" and
      abs(b16["weighted"]["success_all"]-0.8186)<0.002,
      "B16_REGISTERED_COMPARATOR")
 wt=Path(__file__).resolve().parents[2]
 blob=subprocess.check_output(
    ["git","-C",str(wt),"show",
     "c598176:docs/track_r/protocol/gtfree_protocol.py"])
 gate(hashlib.sha256(blob).hexdigest()==PROTO_SHA,"FROZEN_PROTOCOL")
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a13_f8_protocol_")
      and not out.exists() and not out.is_symlink(),"OUTPUT")
 out.mkdir(mode=0o700)
 script=out/"gtfree_protocol_v1_original.py"
 with script.open("xb") as f:f.write(blob)
 gate(sha(script)==PROTO_SHA,"CODE_ROUNDTRIP")
 allres={}
 for variant,clip_used,adaptive,frontier in (
     ("primary",True,False,False),
     ("E0",False,False,False),
     ("self_mad3",True,True,False),
     ("frontier",True,False,True)):
  cohorts={}
  for cohort,(n,tau) in TAU.items():
   dest=out/f"F8_{variant}_{cohort}.json"
   cmd=[sys.executable,"-I",str(script),"--strings",str(strings),
      "--cohort",cohort,"--m","2","--nmax","8","--R","4000",
      "--seed","20261029","--out",str(dest)]
   if clip_used:cmd+=["--clip",str(decisions),
      "--clip-good-col","f_chosen_good",
      "--clip-margin-col","f_margin"]
   if adaptive:cmd+=["--tau-rule","self_mad3"]
   else:cmd+=["--tau",str(tau)]
   if frontier:cmd.append("--frontier")
   proc=subprocess.run(cmd,stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True)
   gate(proc.returncode==0,"F8_PROTOCOL_CHILD_"+variant+"_"+cohort+
        "_"+proc.stdout[-1000:])
   d=read(dest)
   gate(d["protocol_version"]=="v1" and
        d["n_images"]==n and d["N_MAX"]==8 and d["M"]==2 and
        d["orientation_step"] is clip_used and
        d["inputs"]["strings_sha256"]==sha(strings),
        "F8_PER_COHORT_PROTOCOL")
   if not adaptive:gate(abs(d["tau"]-tau)<1e-10,"TAU_IDENTITY")
   cohorts[cohort]=d
  common=set.intersection(*(set(q) for q in cohorts.values()))
  weighted={}
  for key in common:
   x=[cohorts[c][key] for c in TAU]
   if all(isinstance(v,(int,float)) and not isinstance(v,bool)
          and math.isfinite(v) for v in x):
    weighted[key]=sum(TAU[c][0]/180*cohorts[c][key] for c in TAU)
  pooled=dict(status="XP_A13_F8_PROTOCOL_V1_POOLED_BY_IMAGES",
      variant=variant,n_images=180,
      weighted=weighted,
      cohorts={c:dict(n=TAU[c][0],tau=cohorts[c]["tau"],
               sha256=sha(out/f"F8_{variant}_{c}.json")) for c in TAU},
      protocol_sha256=PROTO_SHA,
      strings_sha256=sha(strings),clip_sha256=sha(decisions) if clip_used else None)
  if variant=="primary":
   gain=weighted["gain_vs_fixed_equal_runs"]
   h1=gain>=-0.01
   h2=(gain>=0.02 and
       gain>b16["weighted"]["gain_vs_fixed_equal_runs"] and
       gain>D1_GAIN)
   fc=sum(cohorts[c]["flag_c_expected_count"] for c in TAU)
   uns=sum(cohorts[c]["flag_c_on_never_orbit_panel"] for c in TAU)
   precision=(uns/fc if fc else None)
   h3=None if fc<2 else precision>=0.5
   pooled.update(H1_F8=bool(h1),H2=bool(h2),
      H3_F8=h3,
      FLAG_C_expected_count=fc,
      FLAG_C_unsolved_precision=precision,
      frozen_H2_comparators=dict(B16_gain=
          b16["weighted"]["gain_vs_fixed_equal_runs"],D1_gain=D1_GAIN))
  write(out/f"F8_{variant}_POOLED.json",pooled)
  allres[variant]=pooled
  print("A13_F8_PROTOCOL",variant,
        "success",round(weighted.get("success_all",float("nan")),5),
        "runs",round(weighted.get("runs_mean",float("nan")),5),
        "gain",round(weighted.get("gain_vs_fixed_equal_runs",float("nan")),5),
        flush=True)
 write(out/"XP_A13_F8_PROTOCOL_COMPLETE.json",dict(
    status="XP_A13_F8_FROZEN_PROTOCOL_V1_COMPLETE",
    source_head=a.expected_head,n_images=180,
    primary=allres["primary"],
    H1_F8=allres["primary"]["H1_F8"],
    H2=allres["primary"]["H2"],
    H3_F8=allres["primary"]["H3_F8"],
    other_variants={k:sha(out/f"F8_{k}_POOLED.json")
                    for k in allres if k!="primary"},
    protocol_code_sha256=PROTO_SHA,
    no_frozen_rule_changed=True,no_GPU=True))
 print("XP_A13_F8_CP1_FROZEN_PROTOCOL_PASS",flush=True)
 print("F8_PROTOCOL_OUT",out,flush=True)
def main():
 p=argparse.ArgumentParser()
 for k in ("expected-head","score-dir","clip-dir","b16-protocol","out"):
  p.add_argument("--"+k,required=True)
 a=p.parse_args()
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *q:subprocess.check_output(["git","-C",str(wt),*q],
               text=True).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),"SOURCE_HEAD")
 run(a)
if __name__=="__main__":
 try:main()
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A13_F8_PROTOCOL",e,flush=True)
  sys.exit(8)
