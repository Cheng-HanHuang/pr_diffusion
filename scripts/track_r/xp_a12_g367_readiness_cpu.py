#!/usr/bin/env python3
"""A12 G3/G6/G7 conditional CPU readiness audits; zero GPU work.

G3: require exact PC-DAPS upstream head 59b2f046 and a separate
native command plus y-injection audit. If missing, write the required
safe clone command, not a fabricated adapter or unauthorized download.
G6: check existing batch-1 timing before any duplicative calibration.
G7: reuse frozen XP-C/C2 diagnostic-only GT manifests; do not send
GT to any inverse solver, and do not guess the DDPM time mapping.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
PC=ROOT/"external/PC_DAPS_xp"
PC_HEAD="59b2f046"
CLONE_COMMANDS=[
 "git clone https://github.com/voilalab/PC-DAPS.git /egr/research-pac/huang248/external/PC_DAPS_xp",
 "git -C /egr/research-pac/huang248/external/PC_DAPS_xp checkout --detach 59b2f046",
]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for k in iter(lambda:f.read(1<<20),b""):h.update(k)
    return h.hexdigest()
def jsonread(p):
    return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open("x") as f:json.dump(v,f,sort_keys=True,indent=2);f.write("\n")
def probe_pc():
    if not PC.is_dir() or PC.is_symlink():
        return dict(status="XP_A12_G3_CODE_ABSENT_REQUIRE_CHENG_CLONE",
          code_present=False,
          clone_commands=CLONE_COMMANDS,
          site_source_url="https://github.com/voilalab/PC-DAPS",
          must_not_download_bundled_FFHQ=True,
          do_not_launch_GPU=True)
    head=subprocess.run(["git","-C",str(PC),"rev-parse","HEAD"],
        capture_output=True,text=True)
    actual=head.stdout.strip()
    clean=subprocess.run(["git","-C",str(PC),"status","--porcelain",
                            "--untracked-files=no"],capture_output=True,text=True)
    gate(head.returncode==0 and actual.startswith(PC_HEAD) and
         clean.returncode==0 and not clean.stdout.strip(),
         "G3_PC_NATIVE_HEAD_OR_DIFF")
    paths=("posterior_sample.py","configs/sampler/edm_pcdaps.yaml",
           "configs/task/phase_retrieval.yaml","model/__init__.py")
    present={x:{"exists":(PC/x).is_file(),
                "sha256":sha(PC/x) if (PC/x).is_file() else None} for x in paths}
    gate(all(z["exists"] for z in present.values()),"G3_NATIVE_FILES")
    # Native posterior_sample.py defaults to a bundled FFHQ dataset
    # and synthesizes y; neither is legal here. Until the injected-y
    # adapter demonstrates exact equality, this is READY_FOR_ENGINEERING
    # but never GPU production ready.
    return dict(status="XP_A12_G3_CODE_PINNED_INPUT_ADAPTER_PENDING",
       code_present=True,native_head=actual,source_path=str(PC),
       verified_source_files=present,native_task="phase_retrieval",
       native_sampler="edm_pcdaps",bag_size=4,prereg_n_images=180,
       needs_native_signed_y_injection=True,
       needs_four_particle_outputs_and_own_estimate=True,
       needs_two_id_smoke=True,
       no_bundled_FFHQ=True,stage_ceiling_gpu_h=45,do_not_launch_GPU=True)
def probe_g6():
    dirs=[BASE/"xp_t_fast_20261026T205902Z",
          BASE/"xp_t_fast_imagenet_20261026T212512Z",
          BASE/"xp_t_daps4k_20261027T011832Z"]
    found=[]
    for root in dirs:
        receipts=sorted(root.glob("*COMPLETE.json"))
        results=[]
        for f in receipts:
            x=jsonread(f)
            results.append(dict(path=str(f),sha256=sha(f),
                         status=x.get("status"),timing_class=x.get("timing_class"),
                         timing_valid=x.get("timing_valid"),
                         n_roots=x.get("n_roots")))
        found.append(dict(root=str(root),receipts=results))
    # G6 conditional "if missing" requires exact clean batch-1 status
    # including an absence of foreign PID contention.
    # Existing receipt presence alone is INSUFFICIENT to certify clean.
    return dict(status="XP_A12_G6_EXISTING_TIMING_REQUIRES_CLEAN_CLASS_AUDIT",
       n_images=3,runs_per_image=3,
       groups=("FAST_IMAGENET","DAPS4K_FFHQ"),found=found,
       clean_gpu_required=True,condition_if_missing=True,
       stage_ceiling_gpu_h=1,do_not_launch_GPU=True)
def probe_g7():
    candidates=sorted(BASE.glob("xp_c_c2_*/XP_C_C2_PREP.json"))
    out=[]
    for p in candidates:
        x=jsonread(p)
        if x.get("status")=="XP_C_C2_PREP_PASS":
            root=p.parent
            out.append(dict(path=str(p),sha256=sha(p),
                c_manifest_sha256=x.get("C_manifest_sha256"),
                c2_manifest_sha256=x.get("C2_manifest_sha256"),
                n_images=x.get("n_images"),
                n_evaluations=x.get("n_C2_evaluations")))
    return dict(status=("XP_A12_G7_DIAGNOSTIC_INPUT_FROZEN_MAPPING_PENDING"
                        if out else "XP_A12_G7_DIAGNOSTIC_INPUT_MISSING"),
       prior_frozen_inputs=out,GT_scope="evaluation_only",
       required_DDPM1000_to_native_schedule_mapping=True,
       must_not_retrieve_bundled_FFHQ=True,
       stage_ceiling_gpu_h=4,do_not_launch_GPU=True,
       deadline_et="2026-11-15T23:59:00-05:00")
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    wt=Path(__file__).resolve().parents[2]
    git=lambda *q:subprocess.check_output(
       ["git","-C",str(wt),*q],text=True,timeout=30).strip()
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"SOURCE_HEAD")
    output=Path(a.out)
    gate(output.parent==BASE and output.name.startswith(
         "XP_A12_G367_CONDITIONAL_AUDIT_") and not output.exists()
         and not output.is_symlink(),"NEW_AUDIT")
    report=dict(status="XP_A12_G367_CPU_READINESS_AUDIT",
       expected_head=a.expected_head,
       G3=probe_pc(),G6=probe_g6(),G7=probe_g7(),
       no_GPU_launched=True,prereg_A12_scope_only=True,
       utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(output,report)
    for stage in ("G3","G6","G7"):
        print("A12_"+stage+"_READINESS",report[stage]["status"],flush=True)
    if not report["G3"]["code_present"]:
        for cmd in CLONE_COMMANDS:print("G3_CHENG_CLONE_COMMAND",cmd,flush=True)
    print("A12_G367_AUDIT_RECEIPT",output,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    try:main(ap.parse_args())
    except Exception as e:
        traceback.print_exc();print("ACTION NEEDED: A12_G367_AUDIT",e,flush=True);sys.exit(8)
