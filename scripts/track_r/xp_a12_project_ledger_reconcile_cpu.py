#!/usr/bin/env python3
"""Strict A12 XP-0-to-A10 project GPU-hour ledger reconciliation.

Source-backed, no GT, no GPU. Each source is a disjoint physical GPU
generation stage, except A10 whose own final ledger ALREADY consolidates
the successful B16 gate, B16 production, DAPS replay attempts and F8.
The two PRE-success B16 gate failures are charged separately.

If unknown GPU output directories, missing source receipts, or source
disagreement exist, STOP and write NO verified budget ledger.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,math,os,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate

MAIN=[
 ("XP_T_FAST_FFHQ","xp_t_fast_20261026T205902Z/XP_T_FAST_COMPLETE.json","gpu_hours",1),
 ("XP_T_FAST_IMAGENET","xp_t_fast_imagenet_20261026T212512Z/XP_T_FAST_COMPLETE.json","gpu_hours",1),
 ("XP_SITCOM_REPLAY","xp_sitcom_replay_20261026T223925Z/XP_SITCOM_REPLAY_GPU_SUMMARY.json","gpu_hours",1),
 ("XP_T_SITCOM","xp_t_sitcom_20261026T233451Z/XP_T_SITCOM_COMPLETE.json","total_gpu_hours",1),
 ("XP_T_DAPS4K","xp_t_daps4k_20261027T011832Z/XP_T_DAPS4K_COMPLETE.json","total_gpu_hours",1),
 ("XP_A16","xp_a16_20261027T022827Z/XP_A16_GENERATION_COMPLETE.json","total_gpu_hours",1),
 ("XP_NIGHT","xp_overnight_20261027T045215Z/XP_NIGHT_CONTROLLER_STATE.json","gpu_hours",1),
 ("XP_A6","xp_a6_20261027T165112Z/XP_A6_CONTROLLER_STATE.json","gpu_hours",1),
 ("A10_B16_REPLAY_F8","xp_f8_prod_20261029T071352Z/XP_F8_PROD_LAST_STATUS.json","A10_total_gpu_h",1),
 ("A10_FAILED_B16_GATE_1","xp_a10_b16_gate_20261028T104128Z/B16_GATE_STATE.json","spent_gpu_seconds",3600),
 ("A10_FAILED_B16_GATE_2","XP_A12_LEGACY_B16_GATE_104836_CONSERVATIVE_CHARGE.json","charged_gpu_h_upper_bound",1),
]
OPTIONAL=[
 ("XP_FAST_REPLAY","xp_fast_replay_20261026T203415Z/XP_FAST_REPLAY_TIMING.json","gpu_hours",1)
]
A10_EXPECTED=55.54578541457511
PROJECT_CAP=400.
A12_CAP=150.
ALLOWED_CPU_PREFIXES=(
 "xp_0_","xp_0_imagenet_",
 "xp_a7_d2d3_","xp_a8_","xp_a9_",
 "xp_a10_d1p2_fixed_","xp_a12_g1a_prepare_",
 "xp_b16_","xp_c_c2_","xp_d1p1_eval_",
 "xp_d1p2_protocol_","xp_d1p2_protocol_results_",
 "xp_f8_legacy_preflight_","xp_f8_legacy_preflight_dev80seedfix_",
 "xp_imagenet_download_","xp_t_daps4k_native_source_",
 "xp_t_timing_audit_")
ALLOWED_CONSOLIDATED_PREFIXES=(
 "xp_a11_b16_prod_","xp_a10_b16_gate_",
 "xp_f8_prod_","xp_f8_legacy_replay_",)
# Stage-specific directories are covered via MAIN + OPTIONAL receipts.
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):h.update(data)
    return h.hexdigest()
def read(p):
    gate(p.is_file() and not p.is_symlink(),"RECEIPT_MISSING_"+str(p))
    x=json.loads(p.read_text())
    gate(isinstance(x,dict),"NOT_OBJECT_"+str(p))
    return x
def one(stage,relative,key,scale):
    path=BASE/relative
    doc=read(path)
    value=doc.get(key)
    gate(isinstance(value,(int,float)) and not isinstance(value,bool)
         and math.isfinite(float(value)) and 0<float(value)<400*3600,
         "NONFINITE_GPU_TIME_"+stage)
    gpu_h=float(value)/scale
    status=doc.get("status",doc.get("state"))
    # An incomplete RUNNING snapshot is never a terminal cost.
    # The correction for this particular stale worker deliberately
    # upper-bills ALL time from its launch to a contemporaneous
    # confirmed-dead PID audit, rather than claiming exact death time.
    gate(status not in ("RUNNING","STARTED","IN_PROGRESS"),
         "NONTERMINAL_SOURCE_"+stage+"_STATUS_"+str(status))
    if stage=="A10_FAILED_B16_GATE_2":
        gate(doc.get("status")==
             "XP_A12_LEGACY_GATE_CONSERVATIVE_UPPER_BOUND_PASS" and
             doc.get("billing_is_upper_bound") is True and
             doc.get("exact_gpu_time_known") is False and
             doc.get("project_cap_safety_eligible") is True and
             doc.get("original_state_sha256")==
               "2d8cf00361688852aae2216115adc9491203970dffbac8a2565ba2facc975527" and
             doc.get("original_events_sha256")==
               "1da54503b4b94f65103e98e943cd02ee796015033368fdd2227ae1d2567a7ad0" and
             0<doc.get("actual_gpu_h_lower_bound",-1)<=gpu_h and
             doc.get("charged_gpu_h_upper_bound")==gpu_h and
             doc.get("uncertain_worker_pid")==1057433,
             "CONSERVATIVE_HISTORICAL_GATE_ACCOUNTING")
    return dict(stage=stage,gpu_h=gpu_h,
       source_receipt_path=str(path),source_receipt_sha256=sha(path),
       source_field=key,source_value=value,
       original_status=status,
       exact_gpu_time_known=(stage!="A10_FAILED_B16_GATE_2"),
       billed_upper_bound=(stage=="A10_FAILED_B16_GATE_2"),
       source_mode=("conservative_occupied_time_upper_bound_PID_dead_audit"
                    if stage=="A10_FAILED_B16_GATE_2"
                    else "one_final_per_disjoint_stage"))
def time_sources():
    stages=[one(*item) for item in MAIN]
    for task in OPTIONAL:
        # The FAST replay was definitely a GPU run; its source MUST exist.
        stages.append(one(*task))
    return stages
def check_cross(stages):
    by={x["stage"]:x for x in stages}
    gate(abs(by["A10_B16_REPLAY_F8"]["gpu_h"]-A10_EXPECTED)<1e-6,
         "A10_FINAL_TOTAL_CHANGED")
    run=BASE/"xp_f8_prod_20261029T071352Z"
    fstate=read(run/"XP_F8_CONTROLLER_STATE.json")
    blast=read(BASE/"xp_a11_b16_prod_20261028T113034Z/B16_PRODUCTION_STATE.json")
    e1=(float(blast["spent_gpu_seconds"])+float(blast["gate_spent_gpu_seconds"]))/3600
    f8=float(fstate["spent_gpu_seconds"])/3600
    failgates=sum(by[x]["gpu_h"] for x in ("A10_FAILED_B16_GATE_1",
                                          "A10_FAILED_B16_GATE_2"))
    gate(abs(f8-16.936766159897413)<1e-6 and
         abs(e1-38.0214546836381)<0.002,"A10_B16_F8_SOURCE_DRIFT")
    # Historical DAPS replay costs are the A10 remainder; their WAVEs
    # are individually verifiable, and must not be charged twice.
    wave=list(BASE.glob("xp_f8_legacy_replay_*/XP_F8_LEGACY_REPLAY_WAVE[12].json"))
    native=0.
    for p in wave:
        x=read(p)
        gate(x.get("status") and isinstance(x.get("incremental_gpu_h"),(float,int)),
             "DAPS_WAVE_SOURCE_"+str(p))
        native+=float(x["incremental_gpu_h"])
    gate(abs(e1+f8+native-A10_EXPECTED)<0.002,
         "A10_HISTORICAL_DAPS_WAVE_RECONCILIATION")
    return dict(A10_B16_and_gate_gpu_h=e1,A10_F8_gpu_h=f8,
        A10_DAPS_legacy_replay_gpu_h=native,
        A10_failed_B16_gate_gpu_h=failgates,
        A10_final_gpu_h=by["A10_B16_REPLAY_F8"]["gpu_h"])
def audit_folders(stages):
    known=set()
    for stage in stages:known.add(Path(stage["source_receipt_path"]).parent.name)
    # These are NOT additional physical GPU jobs: each belongs to the
    # combined A10 final ledger or was a CPU-only analysis/prep.
    allruns=sorted(p for p in BASE.iterdir()
                   if p.is_dir() and p.name.startswith(("xp_","XP_")))
    unclassified=[]
    for p in allruns:
        name=p.name
        if name in known or name.startswith(ALLOWED_CPU_PREFIXES) or \
           name.startswith(ALLOWED_CONSOLIDATED_PREFIXES):
            continue
        unclassified.append(name)
    gate(not unclassified,
         "UNCLASSIFIED_STAGE_DIRECTORIES_"+",".join(unclassified))
    # XP0's own program exclusively performs CPU measurement/file
    # inventories and nvidia-smi observation; no CUDA kernel launch.
    xp0=ROOT/"pr_diffusion_track_r_xp_a12_g1a/scripts/track_r/xp_0_inventory_cpu.py"
    gate(xp0.is_file(),"XP0_CPU_INVENTORY_CODE_MISSING")
    text=xp0.read_text()
    gate("CPU-only inventory" in text and
         "torch.cuda" not in text,"XP0_CPU_PROVENANCE")
    return allruns,xp0
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    output=Path(a.out)
    gate(output==BASE/"XP_A12_PROJECT_LEDGER_VERIFIED.json"
         and not output.exists() and not output.is_symlink(),
         "CANONICAL_VERIFIED_OUTPUT_MUST_BE_NEW")
    stages=time_sources()
    audit=check_cross(stages)
    dirs,xp0=audit_folders(stages)
    # XP0 is a CPU-only source inventory (not a GPU reconstruction).
    stages.insert(0,dict(stage="XP0_CPU_INVENTORY",
        gpu_h=0.,source_receipt_path=str(xp0),
        source_receipt_sha256=sha(xp0),
        source_field="CPU_only_source",
        source_value=0,original_status="CPU_ONLY",
        source_mode="no_gpu_model_or_cuda_used"))
    project_total=sum(x["gpu_h"] for x in stages)
    gate(project_total>100 and project_total<250 and
         project_total+A12_CAP<=PROJECT_CAP,"PROJECT_CAP_HEADROOM")
    document=dict(status="XP_A12_CUMULATIVE_PROJECT_LEDGER_VERIFIED",
      has_every_stage_since_XP0=True,
      source_completeness="XP0-to-A10 including explicitly upper-billed nonterminal failed gate",
      verification="SHA256 per source; one uncertain worker charged conservatively until PID-dead audit; no overlapping B16 stage snapshots or F8 child receipts",
      gpu_hour_total_type="CONSERVATIVE_PROJECT_CAP_BILLING_UPPER_BOUND_NOT_EXACT_MEASURED_TOTAL",
      nonexact_stages=[x["stage"] for x in stages
                       if x.get("billed_upper_bound")],
      project_reported_total_is_conservative_upper_bound=True,
      project_cap_gpu_h=PROJECT_CAP,
      A12_campaign_base_gpu_h=A10_EXPECTED,
      A12_campaign_ceiling_gpu_h=A12_CAP,
      project_total_before_A12_gpu_h=project_total,
      project_remaining_gpu_h=PROJECT_CAP-project_total,
      project_after_full_A12_upper_bound_gpu_h=project_total+A12_CAP,
      stages=stages,
      cross_checks=audit,
      n_stage_directories_checked=len(dirs),
      n_billable_stages=sum(x["gpu_h"]>0 for x in stages),
      no_GPU_launched=True,GT_not_loaded=True,
      created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with output.open("x") as f:
        json.dump(document,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("A12_PROJECT_LEDGER_STRICT_VERIFIED_PASS",flush=True)
    for x in stages:
        print("LEDGER_STAGE",x["stage"],"GPU_H",round(x["gpu_h"],9),flush=True)
    print("A12_PROJECT_GPU_H_BEFORE",round(project_total,9),flush=True)
    print("A12_PROJECT_HEADROOM_AFTER_FULL_150H",
          round(PROJECT_CAP-project_total-A12_CAP,9),flush=True)
    print("A12_VERIFIED_LEDGER_PATH",output,flush=True)
    print("A12_VERIFIED_LEDGER_SHA",sha(output),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A12_PROJECT_LEDGER_RECONCILIATION "+str(e),flush=True)
        sys.exit(8)
