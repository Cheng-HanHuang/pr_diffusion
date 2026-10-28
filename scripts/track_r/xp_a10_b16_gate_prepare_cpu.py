#!/usr/bin/env python3
"""A9.6 B16 full B22 400-root historical SITCOM lineage gate: CPU preparation.

Native GPU manifest contains only measurement identity, native configuration,
historical RNG rule, and checkpoint/source provenance. Historical outputs and
GT are in a separate CPU-only offline manifest, never passed to the GPU.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path

from scripts.track_r.xp_sitcom_replay_prepare_cpu import (
    ROOT,BASE,MODEL,MODEL_SHA,SITCOM,SITCOM_HEAD,
    check_native_checkout,external_checkpoint_provenance,
    settings,native_seed_metadata,safe,sha,
)
from scripts.track_r.xp_a8_fast_v1_cpu import R07_SCORES,R07_RECEIPT,R07_CATALOG,sha as source_sha

WT=Path(__file__).resolve().parents[2]
A6=ROOT/"pr_diffusion_track_r"
A6_HEAD="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
COHORT="S1_b22"
N_IMAGES=100
N_ROOTS=400
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def checked(cond,reason):
    if not cond:raise RuntimeError("STOP_B16_PREP_"+reason)

def git(path,*args):
    # Git --porcelain uses a significant leading status column: " M" means
    # an unstaged modification. str.strip() destroys it and falsely fails the
    # strict source-integrity gate. Remove only terminal line endings.
    return subprocess.check_output(
        ["git","-C",str(path),*args],text=True,timeout=30).rstrip("\r\n")

def commit_json(p,v):
    with p.open("x",encoding="utf-8") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def construct(catalog,scored):
    roots=[r for r in catalog if r.get("source")==COHORT and r.get("family")=="SITCOM"]
    checked(len(roots)==400,"HIST_COUNT")
    chosen=defaultdict(list)
    for row in roots:
        chosen[str(row["image_id"])].append(row)
    checked(len(chosen)==100,"IMAGE_COUNT")
    output=[];offline=[]
    for image in sorted(chosen):
        rows=sorted(chosen[image],key=lambda r:int(r["run_index"]))
        checked([int(r["run_index"]) for r in rows]==[0,1,2,3],"RUNS_0_3")
        first=rows[0]
        for i,r in enumerate(rows):
            checked(r["uid"] in scored and scored[r["uid"]]["source"]==COHORT
                    and scored[r["uid"]]["family"]=="SITCOM"
                    and str(scored[r["uid"]]["image_id"])==image
                    and int(scored[r["uid"]]["run_index"])==i,
                    "SCORED_UID")
            checked(r["measurement_path"]==first["measurement_path"]
                    and r["measurement_sha256"]==first["measurement_sha256"]
                    and source_sha(r["measurement_path"])==r["measurement_sha256"]
                    and source_sha(r["candidate_path"])==r["candidate_sha256"],
                    "Y_OR_HIST_SOURCE_SHA")
            seed,meta=native_seed_metadata(r,"sequential_post_model")
            checked(seed==43,"HISTORICAL_SEED")
            output.append({
                "uid":r["uid"],"source":COHORT,"family":"SITCOM",
                "measurement_source":COHORT,
                "image_id":image,"rep":i,"seed":43,
                "rng_mode":"sequential_post_model",
                "measurement_path":r["measurement_path"],
                "measurement_sha256":r["measurement_sha256"]})
            prev=scored[r["uid"]]
            offline.append({
                "uid":r["uid"],"image_id":image,"rep":i,
                "measurement_sha256":r["measurement_sha256"],
                "historical_candidate_path":r["candidate_path"],
                "historical_candidate_sha256":r["candidate_sha256"],
                "historical_tensor_sha256":meta.get("native_tensor_hash"),
                "historical_raw_psnr":float(prev["psnr_raw"]),
                "historical_raw_label":prev["raw_label"],
                "historical_tau":float(prev["tau_signed_good_q95"]),
                "gt_path":r["gt_path"],"gt_format":r["gt_format"]})
    checked(len(output)==N_ROOTS and
            len({r["uid"] for r in output})==N_ROOTS,"UNIQUE_ROOTS")
    return output,offline

def verify_checkout(expected):
    checked(git(WT,"rev-parse","HEAD")==expected and
            not git(WT,"status","--porcelain"),"WORKTREE")
    checked(git(A6,"rev-parse","HEAD")==A6_HEAD and
            not git(A6,"status","--porcelain"),"PINNED_A6")
    checked(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")

def main(args):
    verify_checkout(args.expected_head)
    original=json.loads(safe(R07_RECEIPT).read_text())
    checked(original.get("n_scored")==3240 and
            original.get("evaluated_rows_sha256")==source_sha(R07_SCORES),
            "R07_SCORES_RECEIPT")
    catalogue=json.loads(safe(R07_CATALOG).read_text())["rows"]
    checked(len(catalogue)==3240,"R07_CATALOG_COUNT")
    scored={r["uid"]:r for raw in safe(R07_SCORES).read_text().splitlines()
            if raw for r in [json.loads(raw)]}
    checked(len(scored)==3240,"R07_SCORE_UID_UNIQUE")
    native_state=check_native_checkout(
        git(SITCOM,"rev-parse","HEAD"),
        git(SITCOM,"status","--porcelain=v1","--untracked-files=no"),
        git(SITCOM,"diff","--numstat","HEAD","--","posterior_sample.py"))
    native_state["unused_cli_sha256"]=sha(SITCOM/"posterior_sample.py")
    checked(git(SITCOM,"rev-parse","HEAD")==SITCOM_HEAD,"NATIVE_HEAD")
    checked(sha(MODEL)==MODEL_SHA,"MODEL_SHA")
    checkpoint=external_checkpoint_provenance(SITCOM/"checkpoint/ffhq256.pt")
    config=settings(json.loads(safe(A6/"configs/b22/b22_2_overnight.json").read_text()))
    roots,offline=construct(catalogue,scored)
    out=Path(args.out)
    checked(out.parent==BASE and out.name.startswith("xp_a10_b16_gate_") and
            not out.exists() and not out.is_symlink(),"OUT_NAMESPACE")
    out.mkdir(parents=False,exist_ok=False)
    mf={
        "status":"XP_A10_B16_GATE_400_TASKS_FROZEN",
        "schema":"prdiff.track_r.xp.a10.b16-gate.v1",
        "head":args.expected_head,"historical_source":"R0.7",
        "source_head":SITCOM_HEAD,"family":"SITCOM","dataset":"ffhq",
        "source":"S1_b22","n_roots":400,"n_images":100,
        "runs_per_image":4,"rng_mode":"sequential_post_model",
        "seed":43,"model_sha256":MODEL_SHA,"model_relpath":"checkpoint/ffhq256.pt",
        "native_checkpoint_provenance":checkpoint,
        "native_checkout_provenance":native_state,
        "config":config,
        "GT_supplied":False,"historical_candidate_supplied":False,
        "embargo":"GATE_ONLY_NO_B16_NEW_RUN_LABELS",
        "rows":roots}
    manifest=out/"XP_A10_B16_GATE_TASKS.json"
    commit_json(manifest,mf)
    off=out/"XP_A10_B16_GATE_OFFLINE.json"
    commit_json(off,{
        "status":"CPU_ONLY_NEVER_FEED_GPU","head":args.expected_head,
        "n_roots":400,"rows":offline})
    proof={
        "status":"XP_A10_B16_GATE_PREP_PASS","head":args.expected_head,
        "n_images":100,"n_roots":400,"native_head":SITCOM_HEAD,
        "manifest_sha256":sha(manifest),"offline_sha256":sha(off),
        "original_r07_scored_sha256":sha(R07_SCORES),
        "native_checkout_provenance":native_state,
        "native_checkpoint_provenance":checkpoint,
        "gpu_access":False,"gt_access":False,
        "B16_new_run_labels_opened":False}
    commit_json(out/"XP_A10_B16_GATE_PREP.json",proof)
    print("XP_A10_B16_GATE_PREP_PASS images=100 roots=400",flush=True)
    print("B16_GATE_MANIFEST_SHA="+proof["manifest_sha256"],flush=True)
    print("B16_GATE_PREP_PATH="+str(out),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    try:main(ap.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: B16_GATE_PREP",str(exc),flush=True)
        sys.exit(8)
