#!/usr/bin/env python3
"""CPU-only embargo-safe publication of the completed XP-A16 generation.

Verify original immutable 2160 image reconstructions, per-root and chunk
receipts and frozen source inputs; NEVER open GT or reconstruct/image-view.
Commit only status/hashed operational metadata. No per-root PSNR, quality
or best-of-K values are computed or exposed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, subprocess, sys, traceback
from pathlib import Path

from scripts.track_r.xp_a16_contract import (
    ROOT,WT,RUN,TASKS,MANIFEST_SHA,PREP_HEAD,TOTAL,build_chunks,
    validate_a16_source)
from scripts.track_r.xp_fast_gpu import (
    sha,frozen_roots,validate_complete,root_folder)

GEN_HEAD="56c84c457d21b04fdb26e6b866a4071c09c655b7"
COMPLETE_SHA="68cdb23bb6c8092a82464e032f3c7e448d9c2be579a4357ba4597ec6efd8e544"
GPU_H=2.5861581238130262
DEST=WT/"docs/track_r/executors/xp/returns/XP_A16_GENERATION_20261027T022827Z"

def load(path):
    path=Path(path)
    forbidden=("confirmation","confirm305","test-ffhq","demo-ffhq")
    if (path.is_symlink() or not path.is_file()
        or any(x in str(path.resolve(strict=False)).lower() for x in forbidden)):
        raise RuntimeError("STOP_A16_PUBLISH_FILE_OR_PROTECTED_PATH")
    return json.loads(path.read_text())

def near(a,b,tol=1e-7):
    try:
        v=float(a)
        return math.isfinite(v) and abs(v-float(b))<=tol
    except (TypeError,ValueError):
        return False

def verify_metadata(prep,manifest,status,smoke,complete):
    if not (prep.get("status")=="XP_A16_PREP_PASS"
        and prep.get("head")==PREP_HEAD
        and prep.get("manifest_sha256")==MANIFEST_SHA
        and prep.get("n_images")==180 and prep.get("n_roots")==TOTAL
        and prep.get("gpu_work_performed") is False
        and prep.get("outcomes_embargoed") is True):
        raise RuntimeError("STOP_A16_PUBLISH_PREP")
    if not (manifest.get("stage")=="XP-A16"
        and manifest.get("head")==PREP_HEAD
        and manifest.get("n_images")==180
        and manifest.get("n_roots")==TOTAL
        and manifest.get("gt_paths_supplied") is False):
        raise RuntimeError("STOP_A16_PUBLISH_TASKS")
    if not (status.get("state")=="PASS"
        and status.get("phase")=="COMPLETE"
        and status.get("reason") is None
        and status.get("head")==GEN_HEAD
        and status.get("manifest_sha256")==MANIFEST_SHA
        and status.get("completed_roots")==TOTAL
        and status.get("completed_chunks")==38
        and status.get("running_chunks")==0
        and status.get("active_pids")==[]
        and status.get("outcomes_embargoed") is True
        and status.get("clean_timing_eligible") is True
        and near(status.get("gpu_hours"),GPU_H,1e-5)):
        raise RuntimeError("STOP_A16_PUBLISH_STATUS")
    if not (smoke.get("status")=="XP_A16_SMOKE_PASS"
        and smoke.get("image_aligned_chosen_starts")==[0,1200]
        and smoke.get("new_roots")==24
        and smoke.get("new_image_counts")=={"S1_b22":1,"S2_b24_dev80":1}
        and smoke.get("outcomes_embargoed") is True
        and smoke.get("gt_read") is False
        and 0<float(smoke.get("conservative_projected_gpu_hours",-1))<12):
        raise RuntimeError("STOP_A16_PUBLISH_SMOKE")
    if not (complete.get("status")=="XP_A16_GENERATION_PASS"
        and complete.get("stage")=="XP-A16"
        and complete.get("head")==GEN_HEAD
        and complete.get("prep_head")==PREP_HEAD
        and complete.get("manifest_sha256")==MANIFEST_SHA
        and complete.get("n_images")==180
        and complete.get("historical_roots_reused_without_regeneration")==720
        and complete.get("new_roots")==TOTAL
        and complete.get("n_chunks")==38
        and complete.get("n_smoke_roots")==24
        and complete.get("run_indices")==list(range(4,16))
        and complete.get("per_cohort_new_roots")=={"S1_b22":1200,"S2_b24_dev80":960}
        and complete.get("workers_per_physical_gpu")==1
        and complete.get("physical_gpu_used")==[0,1,2,3]
        and complete.get("strict_gt_free_generation") is True
        and complete.get("historical_output_paths_not_supplied") is True
        and complete.get("quality_metrics_generated") is False
        and complete.get("quality_scoring_authorized") is False
        and complete.get("outcomes_embargoed") is True
        and complete.get("theory_freeze_required_for_scoring")=="TH1_PREDICTIONS_v1.json"
        and complete.get("clean_timing_eligible") is True
        and near(complete.get("total_gpu_hours"),GPU_H,1e-5)
        and near(complete.get("total_gpu_seconds"),GPU_H*3600,0.04)
        and complete.get("smoke_gate")==smoke):
        raise RuntimeError("STOP_A16_PUBLISH_COMPLETE")
    telemetry=complete.get("per_device_timing")
    if not isinstance(telemetry,dict) or set(telemetry)!={"0","1","2","3"}:
        raise RuntimeError("STOP_A16_PUBLISH_DEVICE_SET")
    for gpu,data in telemetry.items():
        if (data.get("timing_class")!="CLEAN"
            or data.get("timing_valid") is not True
            or data.get("foreign_pids")!=[]
            or data.get("reasons")!=[]
            or not isinstance(data.get("n_samples"),int)
            or data["n_samples"]<2):
            raise RuntimeError("STOP_A16_PUBLISH_TIMING_"+gpu)
    return True

def verify_chunk_proofs(complete,rows,*,read_file=load,hash_file=sha):
    smoke,main=build_chunks()
    expected=smoke+main
    chunks=complete.get("chunks")
    if (not isinstance(chunks,list) or len(chunks)!=38
        or [(r.get("start"),r.get("count")) for r in chunks]!=
            sorted(expected)):
        raise RuntimeError("STOP_A16_PUBLISH_CHUNK_PARTITION")
    seen=set();count=0
    for summary in chunks:
        start,n,gpu=summary["start"],summary["count"],summary["gpu"]
        if gpu not in range(4):
            raise RuntimeError("STOP_A16_PUBLISH_PHYSICAL_GPU")
        chunk=RUN/"gpu"/f"chunk_{start:04d}_{n:02d}"
        meta_path=chunk/"R0_5_CHUNK_META.json"
        receipt_path=chunk/"R0_5_CHUNK_COMPLETE.json"
        meta=read_file(meta_path);receipt=read_file(receipt_path)
        if not (hash_file(meta_path)==summary.get("chunk_metadata_sha256")
           and hash_file(receipt_path)==summary.get("native_chunk_sha256")
           and meta.get("head")==GEN_HEAD
           and meta.get("root_sha256")==MANIFEST_SHA
           and meta.get("gpu")==gpu
           and meta.get("start")==start and meta.get("count")==n
           and meta.get("uids")==[r["uid"] for r in rows[start:start+n]]
           and receipt.get("status")=="R0_5_CHUNK_PASS"
           and receipt.get("head")==GEN_HEAD
           and receipt.get("start")==start
           and receipt.get("count")==n
           and receipt.get("new")==n
           and receipt.get("reused")==0
           and receipt.get("meta_sha256")==hash_file(meta_path)):
            raise RuntimeError("STOP_A16_PUBLISH_CHUNK_"+str(start))
        for row in rows[start:start+n]:
            uid=row["uid"]
            if uid in seen:
                raise RuntimeError("STOP_A16_PUBLISH_DUPLICATE_ROOT")
            seen.add(uid)
            folder=root_folder(chunk,row)
            root=read_file(folder/"R0_FAST_ROOT_COMPLETE.json")
            if not (root.get("uid")==uid
                and root.get("source_head")==GEN_HEAD
                and root.get("frozen_root_sha256")==MANIFEST_SHA
                and root.get("seed")==row["seed"]
                and root.get("rep")==row["rep"]
                and root.get("measurement_sha256")==row["measurement_sha256"]
                and root.get("gt_access") is False
                and root.get("dataset")=="ffhq"
                and root.get("method")=="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5"
                and root.get("oversample")==2.0
                and 0<float(root.get("native_sample_wall_seconds",0))<1200
                and hash_file(folder/"reconstruction.pt")==root.get("reconstruction_pt_sha256")
                and hash_file(folder/"reconstruction.png")==root.get("reconstruction_png_sha256")):
                raise RuntimeError("STOP_A16_PUBLISH_ROOT_"+str(start))
            count+=1
    if count!=TOTAL or len(seen)!=TOTAL or seen!={x["uid"] for x in rows}:
        raise RuntimeError("STOP_A16_PUBLISH_ROOT_ACCOUNTING")
    return {"chunk_count":38,"new_candidate_count":count,
            "all_reconstruction_file_hashes_verified":True,
            "all_original_native_receipts_verified":True}

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A16_PUBLISH_CPU_ONLY")
    git=lambda *args:subprocess.check_output(
        ["git","-C",str(WT),*args],text=True,timeout=30).strip()
    if git("rev-parse","HEAD")!=a.expected_head or git("status","--porcelain"):
        raise RuntimeError("STOP_A16_PUBLISH_WRONG_OR_DIRTY_HEAD")
    if sha(TASKS)!=MANIFEST_SHA or sha(RUN/"XP_A16_GENERATION_COMPLETE.json")!=COMPLETE_SHA:
        raise RuntimeError("STOP_A16_PUBLISH_FROZEN_SHA")
    prep=load(RUN/"XP_A16_PREP.json")
    manifest=load(TASKS)
    status=load(RUN/"XP_A16_STATUS.json")
    smoke=load(RUN/"XP_A16_SMOKE_GATE.json")
    complete=load(RUN/"XP_A16_GENERATION_COMPLETE.json")
    rows=frozen_roots(manifest,PREP_HEAD)
    validate_a16_source(TASKS,MANIFEST_SHA,manifest,rows,PREP_HEAD,0,12)
    verify_metadata(prep,manifest,status,smoke,complete)
    audit=verify_chunk_proofs(complete,rows)
    if DEST.exists() or DEST.is_symlink():
        raise RuntimeError("STOP_A16_PUBLISH_ALREADY_STAGED")
    DEST.mkdir(parents=True,exist_ok=False)
    names=("XP_A16_PREP.json","XP_A16_STATUS.json",
           "XP_A16_SMOKE_GATE.json","XP_A16_GENERATION_COMPLETE.json")
    digest_rows=[]
    for name in names:
        source=RUN/name
        shutil.copyfile(source,DEST/name)
        digest_rows.append((sha(DEST/name),name))
    record={
        "status":"XP_A16_EMBARGO_SAFE_EVIDENCE_PASS",
        "generation_head":GEN_HEAD,
        "publication_source_head":a.expected_head,
        "frozen_manifest_sha256":MANIFEST_SHA,
        "generation_complete_sha256":COMPLETE_SHA,
        "audit":audit,
        "gt_accessed":False,
        "per_image_outcomes_inspected":False,
        "quality_scoring_performed":False,
        "theory_outcome_embargo_active":True,
        "reconstruction_files_copied_to_git":False,
    }
    with (DEST/"XP_A16_EVIDENCE_AUDIT.json").open("x",encoding="utf-8") as f:
        json.dump(record,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    digest_rows.append((sha(DEST/"XP_A16_EVIDENCE_AUDIT.json"),"XP_A16_EVIDENCE_AUDIT.json"))
    with (DEST/"SHA256SUMS.txt").open("x",encoding="utf-8") as f:
        for digest,name in digest_rows:f.write(digest+"  "+name+"\n")
    print("XP_A16_EMBARGO_SAFE_EVIDENCE_STAGE_PASS")
    print("CHUNKS=38 NEW_ROOTS=2160 GPU_H="+str(complete["total_gpu_hours"]))
    print("COMPLETE_SHA256="+COMPLETE_SHA)
    print("ALL_2160_RECONSTRUCTION_HASHES_VERIFIED=true")
    print("K_GT4_OUTCOME_EMBARGO=ACTIVE")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A16_PUBLISH "+str(e),flush=True)
        sys.exit(8)
