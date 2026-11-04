#!/usr/bin/env python3
"""A22 checkpoint autopilot: Table 3 first, frozen prompts, G7 diagnostics.

Single PAC supervisor. CPU stages publish independently. On GPU side,
only preregistered G7 C2 native diagnostics run; stages without validated
A22 scientific adapters are explicitly BLOCKED, not silently substituted.
Never runs A21.10 optional cross-GPU replication.

The supervisor deliberately does not trigger XP-K or CP4 without a pinned
parameter-only native adapter and its separate launch/receipt audit.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
VENDOR=BASE/".vendor/piq_0p8"
FROZEN_PROMPTS=WT/"docs/track_r/executors/xp/theory_inputs/XP_A22_FROZEN_N2_FACE_PROMPTS.json"
PANELS=("TABLE3","G7_C2","G7_CLIP","XP_K","G7_REMAINDER","CP4")
def gate(c,k):
 if not c:raise RuntimeError("STOP_A22_AUTOPILOT_"+k)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def git(*a):return subprocess.check_output(
 ["git","-C",str(WT),*a],text=True,timeout=60).strip()
def write(p,data):
 tmp=Path(str(p)+f".{os.getpid()}.tmp")
 with tmp.open("x") as f:
  json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
 os.replace(tmp,p)
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
      and git("rev-parse","HEAD")==a.expected_head
      and not git("status","--porcelain"),"FROZEN_HEAD")
 root=Path(a.run)
 gate(root.parent==BASE and root.name.startswith("xp_a30_a22_autopilot_")
      and not root.exists() and not root.is_symlink(),"NEW_RUN")
 prompts=read(FROZEN_PROMPTS)
 gate(prompts["source_commit"]==
      "b1637f7c27a3caaa03a8b204171fca0d3fccfcd5"
      and len(prompts["prompts_upright"])==
          len(prompts["prompts_inverted"])==3,
      "N2_FROZEN_PROMPTS")
 gate((VENDOR/"piq"/"__init__.py").is_file(),
      "PINNED_PIQ_0P8_VENDOR")
 root.mkdir()
 (root/"logs").mkdir()
 state=dict(status="A22_AUTOPILOT_RUNNING",
   execution_head=a.expected_head,
   original_A21_science_untouched=True,
   A21_10_optional_replication_excluded=True,
   scientific_source_N2_prompts_sha256=sha(FROZEN_PROMPTS),
   project_ceiling_gpu_h=400,
   stage_results={},blockers={},events=[],
   no_new_selector=True,no_matched_compute_claim=True)
 def record(stage,status,log,extra=None):
  obj=dict(status=status,log=str(log),utc=time.time())
  if extra:obj.update(extra)
  state["stage_results"][stage]=obj
  state["events"].append(dict(stage=stage,status=status,utc=time.time()))
  write(root/"A22_AUTOPILOT_STATE.json",state)
  print("A22_STAGE",stage,status,flush=True)
 def launch(stage,script,extra,log,envextra=None):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES="",
       PYTHONPATH=str(WT),PYTHONNOUSERSITE="1",
       PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",
       OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",
       NUMEXPR_NUM_THREADS="4")
  if envextra:env.update(envextra)
  cmd=[str(PY),"-u",str(WT/"scripts/track_r"/script),
       "--expected-head",a.expected_head,*extra]
  with Path(log).open("x") as f:
   p=subprocess.run(cmd,cwd=str(WT),env=env,
      stdout=f,stderr=subprocess.STDOUT,
      timeout=21600,check=False)
  if p.returncode:
   record(stage,"FAILED",log,dict(rc=p.returncode))
   print("ACTION NEEDED: A22_STAGE",stage,"rc",p.returncode,
     "log",log,flush=True)
   return False
  record(stage,"PASS",log)
  return True
 write(root/"A22_AUTOPILOT_STATE.json",state)
 try:
  # Deadline-critical CPU first, no GPU needed.
  out=BASE/("xp_a30_a22_tab3_"+root.name.rsplit("_",1)[-1])
  tablog=root/"logs"/"TAB3_CPU.log"
  good=launch("TABLE3","xp_a30_a22_tab3_e1_cpu.py",
    ["--out",str(out)],tablog,
    dict(PYTHONPATH=str(VENDOR)+":"+str(WT)))
  if good:
   launch("TABLE3_PUBLICATION","xp_a30_a22_tab3_publish_cpu.py",
      ["--run",str(out)],root/"logs"/"TAB3_PUBLICATION.log",
      dict(PYTHONPATH=str(VENDOR)+":"+str(WT)))

  # Same N2 prompts already committed in theory_inputs, no extra GPU.
  record("N2_PROMPTS","GITHUB_COMMITTED",FROZEN_PROMPTS,
    dict(source_commit=prompts["source_commit"],
         n_upright=3,n_inverted=3))

  # XP-K calibration needs THREE prospective one-parameter workers.
  # The original FAST/SITCOM/DAPS native adapters all enforce their old
  # parameter values, so launching them is NOT an XP-K measurement.
  state["blockers"]["XP_K"]={
      "status":"BLOCKED_MISSING_FROZEN_PARAMETER_ONLY_NATIVE_ADAPTERS",
      "B22_images":100,"runs_per_family":400,
      "SITCOM_tau":0.05,"FAST_epsilon":33.26,
      "DAPS_tau":0.07071067811865475,
      "DAPS_requires_F8_replay":True,
      "XP_K_stage_ceiling_gpu_h":30.,
      "no_ordinary_native_solver_substitution":True}
  record("XP_K","SCIENTIFIC_ADAPTER_GATE_BLOCKED",
      root/"logs"/"XP_K_GATE.txt")

  # The existing frozen G7 C2 worker can run now, independently from
  # the future XP-K parameter-only adaptors. It uses a 4-GPU-hour cap.
  gtf=BASE/("xp_c_c2_a22_"+root.name.rsplit("_",1)[-1])
  prep=launch("G7_C2_PREP","xp_a30_a22_g7_prepare_cpu.py",
    ["--out",str(gtf)],root/"logs"/"G7_PREP_CPU.log")
  if prep:
   # Safe CPU CLIP diagnostic on frozen GT images.
   clipout=BASE/("xp_a12_g7_clip_a22_"+root.name.rsplit("_",1)[-1])
   launch("G7_CLIP_GT_CPU","xp_a12_g7_c_clip_cpu.py",
      ["--freeze",str(gtf),"--out",str(clipout)],
      root/"logs"/"G7_CLIP_CPU.log")
   # Native C2 GPU; uses own shared global flock. Never run concurrently
   # with another controller holding the campaign lock.
   gpurun=BASE/("xp_a12_g7_prod_a22_"+root.name.rsplit("_",1)[-1])
   gpuok=launch("G7_C2_GPU","xp_a30_a22_g7_c2_autopilot.py",
      ["--freeze",str(gtf),"--run",str(gpurun)],
      root/"logs"/"G7_C2_GPU.log")
   if gpuok:
    st=gpurun/"XP_A22_G7_C2_FINAL_STATUS.json"
    if st.is_file() and read(st)["status"]=="A22_G7_C2_280_COMPLETE":
     stageout=BASE/("xp_a30_a22_g7_report_"+root.name.rsplit("_",1)[-1])
     launch("G7_C2_READOUT","xp_a30_a22_g7_c2_report_cpu.py",
       ["--run",str(gpurun),"--out",str(stageout)],
       root/"logs"/"G7_C2_READOUT.log")
    else:
     record("G7_C2_READOUT","PENDING_COMPLETE_PANEL",
       root/"logs"/"G7_C2_GPU.log")

  state["blockers"]["G7_REMAINDER"]={
   "status":"BLOCKED_NEEDS_APPROVED_XP_C_PRIOR_UNIVFD_DIAGNOSTIC_ADAPTERS",
   "uncompleted":["PRIOR_2_FORWARD","PRIOR_S_MID_GLOBAL2",
      "PRIOR_S_MID_ALL8","UNIVFD"],
   "C2_data_may_be_complete":True,
   "must_not_report_as_full_G7":True}
  state["blockers"]["CP4"]={
   "status":"PENDING_XP_K_AND_G7_PROGRAM_GATES",
   "deadline_ET":"2026-11-15",
   "no_new_protocol":True}
  record("G7_REMAINDER","SCIENTIFIC_ADAPTER_GATE_BLOCKED",
    root/"logs"/"G7_REMAINDER_GATE.txt")
  record("CP4","AWAITING_REGISTERED_TIER_C_COMPLETION",
    root/"logs"/"CP4_GATE.txt")
  state["status"]="A22_AUTOPILOT_REACHED_REGISTERED_GATES"
 except BaseException as exc:
  state["status"]="A22_AUTOPILOT_EXCEPTION"
  state["error"]=repr(exc)
  raise
 finally:
  write(root/"A22_AUTOPILOT_FINAL_STATUS.json",state)
  print("A22_AUTOPILOT_TERMINAL",state["status"],
     "stages",[(k,v["status"]) for k,v in state["stage_results"].items()],
     flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--run",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A22_AUTOPILOT",str(e),flush=True);sys.exit(8)
