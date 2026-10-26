#!/usr/bin/env python3
"""CPU-only immutable post-hoc audit of the two completed XP-T FAST pilots.

Do not reclassify or modify original records. Audits only two exact,
published generation runs; no dataset searches, no GPU operations.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from prdiffusion.track_r.xp_t_timing_audit import audit_timing_events

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
SOURCES=[
    ("ffhq","xp_t_fast_20261026T205902Z",
      "3e202508dbe2b9dc16ac4ef2f2bd38ee3a1e29a3",
      "378366c0958238ca832ae29b5f66a8e5d406dcaae8cec401be2dfb9bc17aba4a"),
    ("imagenet","xp_t_fast_imagenet_20261026T212512Z",
      "b3d67d55a569e04f9467dc5dc4c7e796fab1a5f3",
      "ca66eb9cfcd84d8b567ff407f35e9027520d977d4f1963ea85a9ae94ce7e820d")
]
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
def safe(p):
    p=Path(p)
    real=p.resolve(strict=False);root=ROOT.resolve(strict=False)
    if not p.is_absolute() or not p.is_file() or p.is_symlink() or \
       (real!=root and root not in real.parents) or \
       any(t in str(p).lower() or t in str(real).lower() for t in BAD):
        raise RuntimeError("STOP_TIMING_AUDIT_PATH "+str(p))
    return p
def sha(p):
    h=hashlib.sha256()
    with safe(p).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):h.update(part)
    return h.hexdigest()
def load(p):return json.loads(safe(p).read_text())
def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_TIMING_AUDIT_GPU_VISIBLE")
    h=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if h!=a.expected_head or dirty:raise RuntimeError("STOP_TIMING_AUDIT_CHECKOUT")
    out=Path(a.out)
    if not str(out).startswith(str(BASE/"xp_t_timing_audit_")) or out.exists() or out.is_symlink():
        raise RuntimeError("STOP_TIMING_AUDIT_OUTPUT")
    audits=[]
    for dataset,name,source_head,complete_sha in SOURCES:
        run=BASE/name
        cp=run/"XP_T_FAST_COMPLETE.json"
        sp=run/"XP_T_FAST_STATUS.json"
        ep=run/"GPU_EVENTS.jsonl"
        if sha(cp)!=complete_sha:raise RuntimeError("STOP_TIMING_COMPLETION_SHA "+dataset)
        comp=load(cp);state=load(sp)
        if comp.get("head")!=source_head or state.get("head")!=source_head or comp.get("dataset")!=dataset:
            raise RuntimeError("STOP_TIMING_ORIGINAL_HEAD "+dataset)
        events=[json.loads(line) for line in safe(ep).read_text().splitlines() if line.strip()]
        result=audit_timing_events(events,state,comp)
        result.update({
            "source_run":name,
            "source_head":source_head,
            "source_complete_sha256":sha(cp),
            "source_status_sha256":sha(sp),
            "source_events_sha256":sha(ep),
            "candidate_reconstruction_outputs_modified":False,
        })
        audits.append(result)
        print("XP_T_TIMING_REAUDIT",dataset,result["qualified_timing_class"],
              "active_pid_samples",result["n_active_worker_visible"],
              "post_exit_orphans",result["n_unattributed_post_exit_samples"],
              flush=True)
    summary={
        "status":"XP_T_FAST_TIMING_REAUDIT_PASS",
        "schema":"prdiff.track_r.xp.timing-reaudit.v1",
        "cpu_audit_head":h,
        "n_datasets":2,
        "original_receipts_immutable":True,
        "gpu_generation_performed":False,
        "gt_access":False,
        "scope":"FAST XP-T FFHQ/ImageNet batch-1 ONLY",
        "not_full_XP_T":True,
        "caveat":"Post-hoc eligibility does not establish absence of sub-poll hidden processes; "
                  "independent clean remeasurement is preferable for publication.",
        "results":audits
    }
    out.mkdir(parents=True,exist_ok=False)
    target=out/"XP_T_FAST_TIMING_REAUDIT.json"
    with target.open("x") as f:
        json.dump(summary,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    print("XP_T_FAST_TIMING_REAUDIT_PASS SHA="+sha(target))
    print("OK")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print("ACTION NEEDED: XP_T_TIMING_REAUDIT "+str(ex),flush=True)
        raise SystemExit(8)
