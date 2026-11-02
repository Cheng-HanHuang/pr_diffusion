#!/usr/bin/env python3
"""A19 Round15 work-conserving readiness audit, NO experimental scoring.

This is deliberately a CPU preflight, not a surrogate for unavailable
native XP-C3 and XP-I2 implementations. It inventories exactly frozen
A19 inputs, missing adapter/code, and clean/dirty GPU requirements so
autopilot can safely launch stages independently when production-ready.
Never reads unregistered outcome labels or opens candidate pixels.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
TARGETS=("XP_D4","G5","XP_N")
SIZES={"XP_D4":6400,"G5":800,"XP_N":6400}
EXPECTED_FORECAST_SHA="a09f553292947f4cf9c0f95294a9afd205938f0dc18dd75f4d6ff655916a3ee3"
EXPECTED_A23_COMMIT="2d791c5d97fd7a09c216f27ab9b087ad26777882"
DECISION_COLS=["image_id","run","label","prior_choice","prior_margin","prior_chosen_good"]
GPU_RULES={
 "NOV4_CPU":"CPU_ONLY",
 "XPC3_PRIOR":"DIRTY_OK_80PCT_ONE_WORKER_PER_GPU_NO_TIMING_CLAIM",
 "XPI2_PCDAPS":"DIRTY_OK_AFTER_NATIVE_REPLAY_80PCT_ONE_WORKER_PER_GPU",
 "TAB3_E1":"CPU_PREFERRED_DIRTY_OK_FOR_METRIC_VALUES_NOT_TIMING",
 "XPK_G7_CP4":"DIRTY_OK_FOR_NATIVE_QUALITY_CLEAN_REQUIRED_IF_TIMING"
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def safe(p):
 p=Path(p)
 if not p.is_file() or p.is_symlink() or ROOT.resolve() not in p.resolve().parents:
  raise RuntimeError("STOP_A24_UNSAFE_PATH_"+p.name)
 return p
def read(p):return json.loads(safe(p).read_text())
def one(pattern,filename=None,sha_expected=None):
 result=[]
 for d in BASE.glob(pattern):
  if not d.is_dir() or d.is_symlink():continue
  f=d/filename if filename else d
  if not f.is_file() or f.is_symlink():continue
  if sha_expected and sha(f)!=sha_expected:continue
  result.append((d,f))
 return result
def stage(stage,ready,details=None,reason=None):
 return dict(stage=stage,status="READY" if ready else "WAIT_IMPLEMENTATION_OR_INPUT",
             details=details or {},blocker=reason if not ready else None)
def inventory(a):
 wt=Path(__file__).resolve().parents[2]
 git=lambda *x:subprocess.check_output(["git","-C",str(wt),*x],
         text=True,timeout=30).strip()
 if git("rev-parse","HEAD")!=a.expected_head or git("status","--porcelain"):
  raise RuntimeError("STOP_A24_DIRTY_OR_WRONG_HEAD")
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
  raise RuntimeError("STOP_A24_CPU_ONLY")
 data={}
 native=one("xp_a22_xpn_continue_*","XP_N_CONTINUATION_FINAL_STATUS.json")
 passed=[(d,r) for d,p in native for r in [read(p)]
         if r.get("status")=="A22_XPN_ALL6400_NATIVE_GENERATION_COMPLETE"
         and r.get("complete_chunks")==90
         and r.get("accepted_full_chunk_roots")==6400
         and r.get("stage_total_gpu_h",100)<28]
 data["XP_N"]=dict(accepted_full64=len(passed)==1,
                  n_candidate_roots=6400 if len(passed)==1 else 0,
                  native_status_receipts=len(passed),
                  stage_gpu_h=passed[0][1]["stage_total_gpu_h"] if len(passed)==1 else None)
 cpn=one("xp_a23_cpn_part2_*","XP_N_CPN_PART2_RECEIPT.json")
 goodcpn=[(d,r) for d,p in cpn for r in [read(p)]
    if r.get("status")=="A18_XPN_CPN_PART2_ALL100_N64_SCORED"
    and r.get("v5_registered_sha256")==EXPECTED_FORECAST_SHA
    and r.get("n_native_roots")==6400]
 data["XP_N"]["cpn_accepted_local"]=len(goodcpn)==1
 # D4 native/CPU score inventory; never read GT labels at readiness.
 d4=one("xp_a15_cp2_d4_*","D4_NEW_4800_LOCAL_SCORES.csv")
 data["XP_D4"]=dict(n_local_scoring_roots_4_63_sources=len(d4),
                    present_bytes=[sha(p) for d,p in d4[:2]],
                    first_16_native_manifest_required=True)
 # G5 100x8 cohort input, native scorer local outcomes kept private.
 g5=one("xp_a15_cp3_G5_*","CP3_LOCAL_CANDIDATES.csv")
 data["G5"]=dict(n_local_scoring_sources=len(g5),
                 candidate_rows_expected=800,
                 source_shas=[sha(p) for d,p in g5[:4]],
                 stage_receipt_required=True)
 source=wt/"prdiffusion/track_r/prior_score.py"
 c2=wt/"scripts/track_r/xp_a12_g7_c2_gpu.py"
 g3=wt/"scripts/track_r/xp_a12_g3_pcdaps_gpu.py"
 code=wt/"scripts/track_r/xp_a24_xpc3_scoring_gpu.py"
 register=wt/"docs/track_r/executors/xp/XP_C3_A19_CODE_FREEZE.json"
 # Both must be committed/pushed before any XP-C3 candidate is scored.
 # A generic prior_score.S_mid does not fulfill the A19 freeze by itself.
 data["XP_C3"]=stage("XP_C3",
   ready=bool(len(passed)==1 and len(goodcpn)==1 and
              len(d4)>0 and len(g5)>0 and code.is_file() and register.is_file()),
   details=dict(precursor_prior_exists=source.is_file(),native_ddpm_adapter_exists=c2.is_file(),
       n_expected=sum(SIZES.values()),cohorts=SIZES,
       exact_worker_frozen=code.is_file() and register.is_file(),
       stage_cap_gpu_h=3.0,ledger="G7_SHARED_400_GPU_H",
       schema=DECISION_COLS),
   reason="Exact 2-forward/S_mid global2 XP-C3 implementation and SHA registry must be pushed before opening candidate outcomes; D4/G5 native SHA lineage also needs verification")
 # XP-I2 on original ImageNet D1, NOT the disjoint XP-N panel.
 i2worker=wt/"scripts/track_r/xp_a24_xpi2_pcdaps_gpu.py"
 i2prep=wt/"scripts/track_r/xp_a24_xpi2_prepare_cpu.py"
 d1=one("xp_t_imagenet_*","XP_T_IMAGENET_IMAGES.json")
 data["XP_I2"]=stage("XP_I2",
   ready=i2worker.is_file() and i2prep.is_file() and\n         (wt/"scripts/track_r/xp_a24_xpi2_autopilot.py").is_file(),
   details=dict(native_ffhq_g3_source_available=g3.is_file(),
       native_imagenet_adapter=i2worker.is_file(),\n       native_supervisor_registered=(wt/"scripts/track_r/xp_a24_xpi2_autopilot.py").is_file(),
       d1_original_images_need_sha=True,
       n_bags=100,particles_per_bag=4,
       source_image_catalog_candidates=len(d1)),
   reason="Existing G3 worker is FFHQ-only; prepare original D1 ImageNet native GT-free adapter, seed/y SHA and replay before GPU run")
 # Other CPU / supplementary tasks.
 a9=one("xp_a9_exchange_*","A9_FAST_EXCHANGEABILITY.json")
 data["NOV4"]=stage("NOV4_CPU",
    ready=(wt/"scripts/track_r/xp_a24_nov4_cpu.py").is_file(),
    details=dict(confirmed_fast_a9_source=len(a9),
      FFHQ_NEVER_GOOD_EXPECTED=7,D4_NEVER_ORBIT_PANEL_UP_TO=8,
      A9_2D_FACECLIP_11520=True),
    reason="Nov4 exact frozen PNG panel/FFHQ A9.2d production code not yet present; must use historical SHA-selected panels and native face CLIP")
 # A7 ZIP remains a user/PAC private artifact, must not infer location.
 zips=sorted((p for p in BASE.glob("**/*A7*.zip")
             if p.is_file() and not p.is_symlink() and
                ROOT.resolve() in p.resolve().parents),
             key=lambda p:str(p))
 data["TAB3"]=stage("TAB3_E1",
   ready=bool(zips) and (wt/"scripts/track_r/xp_a24_tab3_e1_cpu.py").is_file(),
   details=dict(A7_zip_count=len(zips),
      A7_zip_bytes=[p.stat().st_size for p in zips[:6]],
      A7_zip_sha256=[sha(p) for p in zips[:6]]),
   reason="A7 ZIP and frozen E1 metric reconstruction adapter must be verified before publishing ImageNet DAPS/SITCOM 4-run Table 3")
 data["TAIL"]=stage("XP_K_G7_CP4",ready=False,
    details=dict(deadline="2026-11-15",project_cap_gpu_h=400),
    reason="Follow preregistered separate stage supervisors/caps after Nov4/XP-C3/XP-I2/Tab3; no new selector")
 data["GPU_RULES"]=GPU_RULES
 return data
def main(a):
 d=inventory(a)
 output=dict(status="A19_ROUND15_CPU_READINESS_AUDIT_NO_EXPERIMENTS",
   source_head=a.expected_head,
   utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   required_queue=["NOV4","XP_C3","XP_I2","TAB3","TAIL"],
   source_read_only=True,native_jobs_launched=0,
   GT_or_candidate_images_opened=False,
   gpu_time_charged=0.0,
   stages=d)
 if a.out:
  p=Path(a.out)
  if (p.parent!=BASE or not p.name.startswith("XP_A24_A19_READINESS_")
      or p.exists() or p.is_symlink()):
   raise RuntimeError("STOP_A24_NEW_OUTPUT_PATH")
  with p.open("x") as f:json.dump(output,f,indent=2,sort_keys=True);f.write("\n")
 print("A24_A19_READINESS",json.dumps({
  k:v.get("status", "READ_ONLY") if isinstance(v,dict) else "N/A"
  for k,v in d.items()},sort_keys=True),flush=True)
 print("A24_A19_XPC3_REQUIRE_FREEZE_BEFORE_ANY_CANDIDATE_SCORE",
       d["XP_C3"]["details"]["exact_worker_frozen"],flush=True)
 print("A24_A19_XPI2_IMAGENET_ADAPTER_READY",
       d["XP_I2"]["details"]["native_imagenet_adapter"],flush=True)
 print("A24_A19_GPU_POLICY",json.dumps(GPU_RULES,sort_keys=True),flush=True)
 if a.out:print("A24_READINESS_RECEIPT",a.out,flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--expected-head",required=True)
 ap.add_argument("--out")
 try:main(ap.parse_args())
 except Exception as ex:
  traceback.print_exc()
  print("ACTION NEEDED: A24_A19_READINESS",str(ex),flush=True)
  sys.exit(8)
