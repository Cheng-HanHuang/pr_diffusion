#!/usr/bin/env python3
"""A45 XP-K independent, read-only, fail-closed A23.5b stage gate audit.

This program never launches native GPU work. It reports whether a future
controller may progress, and records missing/invalid evidence separately.
It does not trust a summary flag without checking all signed root receipts.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
FAMILIES=("SITCOM","FAST","DAPS")
A4_TOLERANCE_DB=0.1
def gate(ok,why):
    if not ok:raise RuntimeError("STOP_XPK_A45_"+why)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def read(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink(),"MISSING_FILE_"+str(p))
    return json.loads(p.read_text())
def verified_group(run,manifest,head,family,phase,image_id):
    rows=manifest["historical_replay_rows" if phase=="historical_replay" else "calibrated_rows"][family]
    selected=[r for r in rows if r["image_id"]==image_id]
    gate(len(selected)==(2 if phase=="historical_replay" else 4),
         "GROUP_EXPECTED_COUNT")
    dest=run/"gpu"/family/phase/image_id
    complete=read(dest/"XPK_GROUP_COMPLETE.json")
    gate(complete.get("status")=="XPK_NATIVE_IMAGE_GROUP_PASS"
         and complete.get("head")==head
         and complete.get("family")==family
         and complete.get("phase")==phase
         and complete.get("image_id")==image_id
         and complete.get("n_roots")==len(selected)
         and complete.get("manifest_sha")==sha(run/"XPK_GPU_GT_FREE_MANIFEST.json")
         and complete.get("no_GT_on_GPU") is True,
         "GROUP_COMPLETION_DRIFT")
    digests=[]
    for r in sorted(selected,key=lambda x:x["run"]):
        folder=dest/f"run{r['run']:02d}"
        receipt=folder/"XPK_ROOT_COMPLETE.json"
        d=read(receipt)
        gate(d.get("status")=="XPK_NATIVE_ROOT_PASS"
             and d.get("uid")==r["uid"]
             and d.get("family")==family
             and d.get("phase")==phase
             and d.get("run")==r["run"]
             and d.get("seed")==r["seed"]
             and d.get("code_head")==head
             and d.get("measurement_sha256")==r["measurement_sha256"]
             and d.get("gt_access") is False
             and d.get("historical_candidate_access") is False
             and d.get("png_sha256")==sha(folder/"reconstruction.png"),
             "ROOT_COMPLETION_DRIFT")
        digests.append(sha(receipt))
    gate(complete.get("root_receipt_shas")==digests,
         "GROUP_ROOT_DIGESTS_DRIFT")
    return digests

def replay_gate(run,manifest,head,family):
    rows=manifest["historical_replay_rows"][family]
    gate(len(rows)==2 and len({r["image_id"] for r in rows})==1,
         "FROZEN_TWO_ROOTS")
    digests=verified_group(run,manifest,head,family,"historical_replay",rows[0]["image_id"])
    score=read(run/"score"/f"{family}_REPLAY_A4_RECEIPT.json")
    gate(score.get("status")=="XP_K_A23_5B_ORIGINAL_TWO_ROOT_IDENTITY_REPLAY_PASS"
         and score.get("family")==family
         and score.get("source_head")==head
         and score.get("passed") is True
         and score.get("n_root")==2
         and score.get("original_A4_PSNR_tolerance_db")==A4_TOLERANCE_DB
         and score.get("original_A4_label_must_match") is True
         and score.get("strict_thresholds_unchanged") is True,
         "REPLAY_SCORE_GATE")
    srows=score.get("rows")
    gate(isinstance(srows,list) and len(srows)==2
         and {r.get("uid") for r in srows}=={r["uid"] for r in rows},
         "REPLAY_SCORE_UIDS")
    for r in srows:
        delta=r.get("abs_raw_psnr_delta_db")
        gate(r.get("pass_A4") is True
             and r.get("computed_native_class")==r.get("frozen_historical_class")
             and type(delta) in (float,int)
             and math.isfinite(delta) and 0<=delta<=A4_TOLERANCE_DB,
             "REPLAY_A4_NUMERICAL_OR_LABEL")
    return {"status":"PASS","root_receipt_shas":digests,
            "replay_score_sha256":sha(run/"score"/f"{family}_REPLAY_A4_RECEIPT.json")}

def f8_gate(run,manifest):
    d=read(run/"XPK_DAPS_F8_GATE_RECEIPT.json")
    gate(d.get("status")=="XPK_DAPS_F8_8_OF_8_EXACT_VERIFIED"
         and d.get("n_exact")==8 and d.get("verified") is True
         and d.get("historical_native_source_head")==
             "e7a77d094167084faed19b599b96673b7bb11447"
         and d.get("prior_f8_manifest_sha256")==manifest["prior_f8_manifest_sha256"],
         "F8_PREVIOUS_8_OF_8")
    return {"status":"PASS","f8_receipt_sha256":sha(run/"XPK_DAPS_F8_GATE_RECEIPT.json")}

def family_status(run,manifest,head,family):
    out={"family":family,"replay":"PENDING","f8":"NOT_APPLICABLE",
         "calibrated_images_complete":0,"calibrated_roots_complete":0,
         "calibration_authorized":False,"calibration_full_panel_complete":False}
    try:
        out["replay_evidence"]=replay_gate(run,manifest,head,family)
        out["replay"]="PASS"
    except Exception as exc:
        out["replay_error"]=str(exc)
    if family=="DAPS":
        try:out["f8_evidence"]=f8_gate(run,manifest);out["f8"]="PASS"
        except Exception as exc:out["f8"]="PENDING";out["f8_error"]=str(exc)
    out["calibration_authorized"]=(out["replay"]=="PASS"
                                   and (family!="DAPS" or out["f8"]=="PASS"))
    rows=manifest["calibrated_rows"][family]
    ids=sorted({r["image_id"] for r in rows})
    gate(len(ids)==100 and len(rows)==400,"CALIBRATION_FROZEN_100x4")
    if not out["calibration_authorized"]:
        return out
    for image_id in ids:
        try:
            verified_group(run,manifest,head,family,"calibrated",image_id)
        except Exception:
            continue
        out["calibrated_images_complete"]+=1
        out["calibrated_roots_complete"]+=4
    out["calibration_full_panel_complete"]=(out["calibrated_images_complete"]==100)
    if out["calibration_full_panel_complete"]:
        score=run/"score"/f"{family}_CAL_AGGREGATE.json"
        if score.is_file() and not score.is_symlink():
            d=read(score)
            out["calibration_score_sha256"]=sha(score)
            out["calibration_score_status"]=d.get("status","UNVERIFIED")
    return out

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
                                 text=True,timeout=30).strip()
    gate(head==a.expected_head,"CODE_HEAD")
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a41_xpk_")
         and run.is_dir() and not run.is_symlink(),"RUN_ROOT")
    mf=run/"XPK_GPU_GT_FREE_MANIFEST.json"
    m=read(mf);freeze=read(run/"XPK_SOURCE_FREEZE_RECEIPT.json")
    gate(m.get("status")=="XP_K_A24_FROZEN_GT_FREE_100x4"
         and m.get("head")==head
         and m.get("original_GT_access") is False
         and m.get("reference_paths_in_GPU_manifest") is False
         and freeze.get("code_head")==head
         and freeze.get("sha_manifest")==sha(mf),
         "SOURCE_FREEZE")
    results={family:family_status(run,m,head,family) for family in FAMILIES}
    report=dict(status="XP_K_A45_READ_ONLY_STAGE_AUDIT",
                source_head=head,manifest_sha256=sha(mf),
                families=results,
                all_replay_gates_pass=all(r["replay"]=="PASS" for r in results.values()),
                all_calibrated_panels_complete=all(r["calibration_full_panel_complete"] for r in results.values()),
                does_not_authorize_GPU_generation=True,
                no_GPU_jobs_launched=True)
    print(json.dumps(report,sort_keys=True,indent=2))
    if a.out:
        dest=Path(a.out)
        gate(dest.parent==run and dest.name.startswith("XPK_A45_AUDIT_")
             and not dest.exists() and not dest.is_symlink(),"NEW_AUDIT")
        with dest.open("x") as f:json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out")
    try:sys.exit(main(p.parse_args()))
    except Exception:
        traceback.print_exc()
        sys.exit(8)
