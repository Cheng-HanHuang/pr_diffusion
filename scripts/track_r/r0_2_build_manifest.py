#!/usr/bin/env python3
"""R0.2: bounded, exposed-only candidate manifest (CPU; immutable sources).

Uses only named manifests/CSV/receipts and narrowly scoped candidate dirs.
All GT fields are provenance only: GT never enters scorer/generator workers.
Fail closed on missing or nonconforming per-run records. No writes to sources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from prdiffusion.track_r.manifest import (
    REQUIRED, assert_no_forbidden, uid_for, validate_row, write_manifest,
)

ROOT = Path("/egr/research-pac/huang248")
B22_FAMILIES = ("sitcom", "np")
BASE = ROOT / "outputs/pr_diffusion/track_r"
EXPECTED = {
    ("S1_b22", "DAPS"): 200,
    ("S1_b22", "SITCOM"): 400,
    ("S1_b22", "NP"): 800,
    ("S2_b24_dev80", "DAPS"): 320,
    ("S2_b24_dev80", "SITCOM"): 320,
    ("S2_b24_dev80", "NP"): 320,
    ("S3_a12", "DAPS"): 32,
    ("S3_a12", "SITCOM"): 32,
    ("S3_a12", "NP_LF"): 32,
    ("S3_a12", "NP_S2"): 32,
    ("S4_a15", "FAST"): 32,
}

def clean(path, must_exist=True):
    p = Path(assert_no_forbidden(str(path)))
    if not p.is_absolute():
        raise RuntimeError("STOP_NONABSOLUTE_PATH")
    resolved = p.resolve(strict=False)
    assert_no_forbidden(str(resolved))
    if must_exist and not p.is_file():
        raise RuntimeError("STOP_MISSING_INPUT " + str(p))
    return p

def read_json(path):
    p=clean(path)
    with p.open(encoding="utf-8") as f: return json.load(f)

def digest(path):
    h=hashlib.sha256()
    with clean(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def value(source,family,image_id,run_index,run_label,
          candidate,gt,gt_format,measurement,recorded_psnr,convention,
          selector_name=None,selector_value=None,cost=None,
          forwards=None,notes="",measurement_sha=None):
    row=dict(uid=uid_for(source,family,image_id,run_label),
        source=source,family=family,image_id=str(image_id),
        run_index=int(run_index),run_label=run_label,
        candidate_path=str(clean(candidate)),candidate_format=clean(candidate).suffix[1:],
        gt_path=str(clean(gt)),gt_format=gt_format,
        measurement_path=str(clean(measurement)),
        measurement_sha256=measurement_sha,
        recorded_psnr_db=recorded_psnr,recorded_psnr_convention=convention,
        native_selector_name=selector_name,
        native_selector_value=selector_value,
        recorded_cost_seconds=cost,recorded_unet_forwards=forwards,
        notes=notes)
    return validate_row(row)

def img_source(image_id,root):
    image_id=str(int(image_id)).zfill(5)
    return clean(Path(root)/f"{(int(image_id)//1000)*1000:05d}"/(image_id+".png"))

def canonical_b22_rows(cfg):
    p=clean(cfg["sources"]["S1_b22"]["b21_fresh_rows_csv"])
    with p.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if len(rows)!=100 or len(set(r["image_id"] for r in rows))!=100:
        raise RuntimeError("STOP_S1_B21_ROW_COUNT")
    return rows

def build_s1(cfg):
    rows=[]
    base=Path(cfg["sources"]["S1_b22"]["b22_run_root"])/"full"
    ffhq=cfg["data"]["ffhq_images1024"]
    for fr in canonical_b22_rows(cfg):
        iid=str(int(fr["image_id"])).zfill(5)
        gt=img_source(iid,ffhq)
        meas=clean(fr["measurement_path"])
        for i in (1,2):
            key=f"arm{i}_sample_path"
            selected=fr[key]
            rows.append(value("S1_b22","DAPS",iid,i-1,f"fresh{i}",
                              selected,gt,"ffhq1024",meas,
                              float(fr[f"arm{i}_psnr"]),"B21_FLOAT_CLAMP",
                              "exact_operator_loss",float(fr[f"arm{i}_exact_operator_loss"]),
                              float(fr[f"arm{i}_wall_seconds"]),notes="B21.11 Fresh1/2; recorded order"))
    b21_by_id={str(int(fr["image_id"])).zfill(5):fr for fr in canonical_b22_rows(cfg)}
    for fam,public_name in (("sitcom","SITCOM"),("np","NP")):
        family_root=base/fam
        # B22 rowNNN ordering is not B21's CSV ordering. Join by image ID.
        rowdirs=sorted(p for p in family_root.glob("row*") if p.is_dir())
        if len(rowdirs)!=100:
            raise RuntimeError("STOP_S1_B22_FAMILY_ROWS "+fam)
        found=set()
        for rowdir in rowdirs:
            assert_no_forbidden(str(rowdir))
            m=re.fullmatch(r"row[0-9]{3}_(\d{5})", rowdir.name)
            if m is None:raise RuntimeError("STOP_S1_B22_ROW_NAME")
            iid=m.group(1)
            if iid not in b21_by_id or iid in found:
                raise RuntimeError("STOP_S1_B22_ID_JOIN "+fam+" "+iid)
            found.add(iid)
            fr=b21_by_id[iid]
            candidatedir=clean(rowdir/"policy.json").parent/"candidates"
            candidates=sorted(p for p in candidatedir.iterdir() if p.is_dir())
            for p in candidates:
                assert_no_forbidden(str(p));assert_no_forbidden(str(p.resolve()))
            want=4 if fam=="sitcom" else 8
            if len(candidates)!=want:
                raise RuntimeError(f"STOP_S1_{fam.upper()}_COUNT {iid} {len(candidates)}")
            cs=[]
            for p in candidates:
                rec=read_json(p/"result.json")
                if rec.get("status") not in ("PASS", "pass", "ok","OK","complete", "COMPLETE", None):
                    raise RuntimeError("STOP_S1_CANDIDATE_STATUS "+str(p))
                if str(int(rec["image_id"])).zfill(5)!=iid:
                    raise RuntimeError("STOP_S1_CANDIDATE_ID "+str(p))
                if fam=="sitcom":
                    pos=int(rec["run_index"])
                    label=f"run{pos}"
                    selector=rec.get("selector",{}).get("correction_norm")
                    selname="correction_norm"
                else:
                    tag=str(rec["config_tag"]);seed=int(rec["seed"])
                    pos=(0 if tag=="lf" else 4)+(seed-100)
                    label=f"{tag}_seed{seed}"
                    selector=rec.get("selector_stats",{}).get(
                        "selector_post_winner_lf_mse_mean")
                    selname="selector_post_winner_lf_mse_mean"
                    if tag not in ("lf","s2_preproj_lam001") or seed not in range(100,104):
                        raise RuntimeError("STOP_S1_NP_ORDER")
                recon=rec["reconstruction_tensor_path"]
                cs.append(value("S1_b22",public_name,iid,pos,label,recon,
                    img_source(iid,ffhq),"ffhq1024",fr["measurement_path"],
                    float(rec["metrics"]["psnr_raw"]),"B22_FLOAT_CLAMP",
                    selname,selector,rec.get("timing",{}).get("reconstruction_s"),
                    notes=f"B22 {fam} native float terminal, B21 GT derivation"))
            if sorted(x["run_index"] for x in cs)!=list(range(want)):
                raise RuntimeError("STOP_S1_ORDER_INVALID "+fam+" "+iid)
            rows.extend(cs)
    return rows

def s2_accepted_np_source(record, completion, iid):
    """Resolve NP4 terminals from accepted pilot16 or fresh DEV80 source."""
    accepted=clean(record["accepted_np4_result_path"])
    adopted=completion.get("np_results",{}).get("NP4_INDEPENDENT")
    if adopted is None or clean(adopted)!=accepted:
        raise RuntimeError("STOP_S2_NP_ADOPTED_SOURCE_MISMATCH "+iid)
    reused=bool(completion.get("np_pilot_result_reused"))
    if reused != bool(record["pilot16"]):
        raise RuntimeError("STOP_S2_NP_PILOT_PROVENANCE "+iid)
    if bool(completion.get("np_execution_performed"))==reused:
        raise RuntimeError("STOP_S2_NP_EXECUTION_REUSE_CONFLICT "+iid)
    # Frozen DEV80 includes 15 NP4 Pilot16 roots and one earlier
    # independent one-image smoke (32909). All are explicitly indexed
    # development sources; do not broaden to arbitrary smoke directories.
    if reused:
        pilot_root=ROOT/"outputs/pr_diffusion/b24/B24_3_pilot16_20260913T045410Z"
        smoke_one=ROOT/"outputs/pr_diffusion/b24/B24_3_one_image_smoke_20260913T044345Z/methods/NP4_INDEPENDENT/result.json"
        is_pilot=(pilot_root in accepted.parents)
        is_smoke=(iid=="32909" and accepted==smoke_one)
        if not (is_pilot or is_smoke):
            raise RuntimeError("STOP_S2_PILOT_SOURCE_ROOT "+iid)
    else:
        dev_root=ROOT/"outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z"
        if dev_root not in accepted.parents:
            raise RuntimeError("STOP_S2_DEV80_SOURCE_ROOT "+iid)
    if digest(accepted)!=record["accepted_np4_result_sha256"]:
        raise RuntimeError("STOP_S2_NP_RECEIPT_SHA "+iid)
    np_rec=read_json(accepted)
    if np_rec.get("status")!="PASS" or np_rec.get("arm")!="NP4_INDEPENDENT":
        raise RuntimeError("STOP_S2_NP4_STATUS_OR_ARM "+iid)
    if len(np_rec.get("terminals",[]))!=4 or len(record.get("np_roots",[]))!=4:
        raise RuntimeError("STOP_S2_NP_ROOTS "+iid)
    terminals=tuple(clean(accepted.parent/f"terminal_{i}.pt") for i in range(4))
    return np_rec,terminals,reused


def build_s2(cfg,repo_root):
    idx=read_json(repo_root/cfg["sources"]["S2_b24_dev80"]["manifest_in_repo"])
    source=idx.get("rows",[])
    if len(source)!=80 or len({x["image_id"] for x in source})!=80:
        raise RuntimeError("STOP_S2_DEVELOPMENT_ROWS")
    rows=[]
    for r in source:
        iid=str(int(r["image_id"])).zfill(5)
        gt=clean(r["ground_truth_tensor_path"])
        y=clean(r["measurement_path"])
        if digest(y)!=r["measurement_file_sha256"]:
            raise RuntimeError("STOP_S2_MEAS_SHA "+iid)
        root=clean(r["b24_image_complete_path"]).parent
        comp=read_json(r["b24_image_complete_path"])
        if str(int(comp["image_id"])).zfill(5)!=iid or comp.get("status")!="PASS":
            raise RuntimeError("STOP_S2_IMAGE_RECEIPT "+iid)
        baseline=read_json(root/"baseline"/"metrics"/"METRICS.json")
        tr=baseline.get("terminal_rows",[])
        for fam in ("DAPS","SITCOM"):
            rr=sorted((x for x in tr if x.get("method")==fam),
                      key=lambda x:int(x["rep"]))
            if [int(x["rep"]) for x in rr]!=[0,1,2,3]:
                raise RuntimeError("STOP_S2_BASELINE_COUNT "+fam+" "+iid)
            for t in rr:
                rows.append(value("S2_b24_dev80",fam,iid,int(t["rep"]),
                    f"rep{int(t['rep'])}",t["terminal_png"],gt,"pt",y,
                    float(t["psnr_raw_rgb_db"]),"B24_RGB8",None,None,
                    t.get("wall_seconds"),
                    notes="Fresh external baseline; B24 development only",
                    measurement_sha=r["measurement_file_sha256"]))
        np_rec, terminals, reused=s2_accepted_np_source(r,comp,iid)
        for i,met in enumerate(r["np_roots"]):
            if int(met["root_index"])!=i:raise RuntimeError("STOP_S2_NP_ORDER")
            terminal=terminals[i]
            native=np_rec["terminals"][i]
            if (int(native.get("terminal_index",-1))!=i or
                clean(native["reconstruction_path"])!=terminal or
                native.get("reconstruction_sha256")!=met["reconstruction_sha256"] or
                abs(float(native["psnr_raw_db"])-float(met["accepted_psnr_raw_db"]))>1e-6):
                raise RuntimeError("STOP_S2_NP_TERMINAL_PROVENANCE "+iid+f" rep{i}")
            rows.append(value("S2_b24_dev80","NP",iid,i,f"rep{i}",
                terminal,gt,"pt",y,float(met["accepted_psnr_raw_db"]),
                "B24_RGB8", "selector_plus_mean",
                met.get("selector_plus_mean"),np_rec.get("wall_seconds"),
                notes=("B24 reused exposed pilot16 NP4 terminal" if reused else
                       "B24 freshly generated DEV80 NP4 terminal") +
                      "; metrics from accepted frozen repo index",
                measurement_sha=r["measurement_file_sha256"]))
    return rows

def build_s3s4(cfg):
    s3=cfg["sources"]["S3_a12"]
    frozen=read_json(Path(s3["inputs_root"])/"A1_2_INPUT_FREEZE_COMPLETE.json")
    source=frozen.get("rows",[])
    if len(source)!=8:raise RuntimeError("STOP_S3_FREEZE_COUNT")
    rows=[]
    root=Path(s3["root"])
    oracle=read_json(root/"A1_2_OFFLINE_ORACLE_SUMMARY.json")
    per_image=oracle.get("per_image",{})
    for fr in source:
        iid=str(int(fr["image_id"])).zfill(5)
        recorded={x["candidate_id"]:x for x in per_image.get(iid,{}).get("candidate_metrics",[])}
        if len(recorded)!=16:
            raise RuntimeError("STOP_S3_ORACLE_CANDIDATE_COUNT "+iid)
        gt=fr["ground_truth_tensor_path"]
        y=fr["measurement_path"]
        if digest(y)!=fr["measurement_file_sha256"]:
            raise RuntimeError("STOP_S3_MEAS_SHA "+iid)
        for fam,native in (("DAPS","daps"),("SITCOM","sitcom"),
                           ("NP_LF","np_lf"),("NP_S2","np_s2")):
            for i in range(4):
                if fam=="DAPS":
                    recon=root/iid/native/f"rep{i}"/"daps_results"/f"rep{i}"/"samples"/"00000_run0000.png"
                elif fam=="SITCOM":
                    recon=root/iid/native/f"rep{i}"/"sitcom1"/"reconstruction.png"
                else:
                    recon=root/iid/native/f"rep{i}"/"np1"/"reconstruction.png"
                recs=sorted((root/iid).glob(f"*rep{i}*COMPLETE.json"))
                # Source receipt structure differs per method. Use the
                # method-specific receipt only, never oracle winner summaries.
                tags={"DAPS":"daps","SITCOM":"sitcom","NP_LF":"np_lf","NP_S2":"np_s2"}
                relevant=[p for p in recs if tags[fam] in p.name.lower()]
                if len(relevant)!=1:
                    raise RuntimeError("STOP_S3_RECEIPT_AMBIGUITY "+fam+" "+iid+f" rep{i} {len(relevant)}")
                record=read_json(relevant[0])
                cid=f"{iid}:{fam}:{i}"
                if cid not in recorded:
                    raise RuntimeError("STOP_S3_ORACLE_METRIC_ID "+cid)
                psnr=float(recorded[cid]["raw_psnr_db"])
                if digest(recon)!=recorded[cid]["png_sha256"]:
                    raise RuntimeError("STOP_S3_CANONICAL_PNG_SHA "+cid)
                rows.append(value("S3_a12",fam,iid,i,f"rep{i}",recon,
                    gt,"pt",y,psnr,"A12_RGB8",
                    None,None,record.get("wall_seconds_subprocess"),
                    notes="A1.2 four independent roots; offline oracle candidate metric",
                    measurement_sha=fr["measurement_file_sha256"]))
        s4=Path(cfg["sources"]["S4_a15"]["root"])
        for i in range(4):
            root4=s4/iid/f"rep{i}"
            rec=read_json(root4/"A15_SMOKE_COMPLETE.json")
            psnr=rec.get("metric",{}).get("raw_rgb8_psnr_db")
            if psnr is None: psnr=rec.get("raw_psnr_db")
            if psnr is None:raise RuntimeError("STOP_S4_METRIC "+iid)
            rows.append(value("S4_a15","FAST",iid,i,f"rep{i}",
                root4/"reconstruction.png",gt,"pt",y,float(psnr),
                "A15_RGB8",None,None,rec.get("sample_only_wall_seconds"),
                notes="Pinned A1.5 FAST saved canonical PNG",
                measurement_sha=fr["measurement_file_sha256"]))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--repo-root",required=True)
    ap.add_argument("--expected-head",required=True)
    args=ap.parse_args()
    out=Path(args.out);repo=Path(args.repo_root).resolve(strict=True)
    assert_no_forbidden(str(out))
    if not str(out).startswith(str(BASE)+os.sep) or out.exists():
        raise RuntimeError("STOP_R0_OUTPUT_NOT_FRESH")
    import subprocess
    head=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    clean_tree=subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True)
    if head!=args.expected_head or clean_tree:
        raise RuntimeError("STOP_GIT_IDENTITY")
    cfg=json.loads((repo/"configs/track_r/pac_paths_track_r_v1.json").read_text())
    # Reuse only explicitly named and inventory-verified exposed sources.
    assert_no_forbidden(json.dumps(cfg["sources"]))
    rows=build_s1(cfg)+build_s2(cfg,repo)+build_s3s4(cfg)
    counts=Counter((r["source"],r["family"]) for r in rows)
    if counts!=Counter(EXPECTED):
        raise RuntimeError("STOP_R0_COUNTS "+str(counts))
    uids=[r["uid"] for r in rows]
    if len(set(uids))!=len(uids): raise RuntimeError("STOP_UID_DUPLICATION")
    out.mkdir(parents=True,exist_ok=False)
    name=out/"R0_CANDIDATE_MANIFEST.csv"
    write_manifest(rows,name)
    receipt={"status":"R0_2_MANIFEST_BUILT_NOT_YET_G0_VALIDATED",
        "git_head":head,"n_candidates":len(rows),"n_unique_images":len(set(
        (r["source"],r["image_id"]) for r in rows)),
        "counts":{"/".join(k):n for k,n in sorted(counts.items())},
        "manifest_sha256":digest(name),
        "has_protected_source_paths":False,
        "historical_sources_read_only":True,
        "candidate_paths_exist":True,
        "gpu_forward_calls":0,
        "gt_used_in_scoring":False}
    with (out/"R0_2_MANIFEST_RECEIPT.json").open("x") as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(receipt,sort_keys=True))
    return 0

if __name__=="__main__":
    sys.exit(main())
