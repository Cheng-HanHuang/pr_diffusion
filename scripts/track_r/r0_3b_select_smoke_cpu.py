#!/usr/bin/env python3
"""Freeze eight exposed R0.3 smoke candidates; CPU-only, GT offline for strata.

The two GPU workers receive a stripped list containing NO GT or PSNR. Frozen
selection is deterministic and source-limited; fails rather than broadens search.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prdiffusion.track_r.manifest import read_manifest,assert_no_forbidden
from prdiffusion.track_r.physics import ORBIT_PATTERNS,to_rgb8,psnr_rgb8,apply_orbit
from scripts.track_r.r0_2_g0_cpu import candidate_source,gt_source

BASE=Path("/egr/research-pac/huang248/outputs/pr_diffusion/track_r")
VALID_SHA="8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890"
G0_HEAD="45d63071b808193f3cae776d9add8782e0e14adf"
G1_HEAD="02751a48b570f40bc0647074608895be55e2726c"
G1_UID="S3_a12|DAPS|68539|rep2"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def identity(repo,expected):
    head=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True)
    if head!=expected or dirty:raise RuntimeError("STOP_SMOKE_GIT_IDENTITY "+head)

def verify_preflight(manifest,g0,g1):
    if sha(manifest)!=VALID_SHA:
        raise RuntimeError("STOP_SMOKE_MANIFEST_SHA")
    gr=json.loads(Path(g0).read_text())
    if gr.get("status")!="G0_CPU_PRECHECK_PASS_PENDING_DETERMINISM" or gr.get("git_head")!=G0_HEAD:
        raise RuntimeError("STOP_SMOKE_G0_CPU_STATE")
    if gr.get("manifest_path")!=str(manifest) or gr.get("gates",{}).get(
        "G0_recorded_psnr_replay",{}).get("n_match")!=2520:
        raise RuntimeError("STOP_SMOKE_G0_MANIFEST_OR_REPLAY")
    jr=json.loads(Path(g1).read_text())
    if jr.get("status")!="G1_PASS" or jr.get("source_head")!=G1_HEAD:
        raise RuntimeError("STOP_SMOKE_G1_RECEIPT")
    if jr.get("prior",{}).get("uid")!=G1_UID or jr.get("prior",{}).get("s_star")!="111":
        raise RuntimeError("STOP_SMOKE_G1_CASE")
    if jr.get("manifest_sha256")!=VALID_SHA or jr.get("repaired_psnr_db",0)<25:
        raise RuntimeError("STOP_SMOKE_G1_BAD_MANIFEST_OR_REPAIR")
    return jr

def orbit_psnr(row):
    gt8=to_rgb8(gt_source(row))
    x8=to_rgb8(candidate_source(row))
    return {p:float(psnr_rgb8(apply_orbit(x8,p),gt8)[0])
            for p in ORBIT_PATTERNS}

def stable_pick(rows,tag):
    if not rows:
        raise RuntimeError("STOP_SMOKE_NO_CANDIDATE_FOR_"+tag)
    return sorted(rows,key=lambda r:(hashlib.sha256(
        ("R0SMOKE|"+tag+"|"+r["uid"]).encode()).hexdigest(),r["uid"]))[0]

def choose(rows):
    by_uid={r["uid"]:r for r in rows}
    if len(by_uid)!=2520 or G1_UID not in by_uid:
        raise RuntimeError("STOP_SMOKE_UNIVERSE")
    chosen=[("G1_CANONICAL",by_uid[G1_UID])]
    daps=sorted((r for r in rows if r["source"]=="S1_b22" and
                 r["family"]=="DAPS"),key=lambda r:r["uid"])
    good=None;nonorbit=None; evidence={}
    for row in daps:
        values=orbit_psnr(row)
        if good is None and values["000"]>=25:
            good=row;evidence["S1_DAPS_GOOD"]={
                "uid":row["uid"],"raw_db":values["000"]}
        if nonorbit is None and all(val<25 for val in values.values()):
            nonorbit=row;evidence["S1_DAPS_NONORBIT"]={
                "uid":row["uid"],"raw_db":values["000"],
                "orbit_best_db":max(values.values())}
        if good is not None and nonorbit is not None:
            break
    if good is None or nonorbit is None:
        raise RuntimeError("STOP_SMOKE_REQUIRED_S1_DAPS_STRATA_MISSING")
    chosen.extend([("S1_DAPS_GOOD",good),("S1_DAPS_NONORBIT",nonorbit)])
    requirements=[
        ("S1_SITCOM",lambda r:r["source"]=="S1_b22" and r["family"]=="SITCOM"),
        ("S2_NP",lambda r:r["source"]=="S2_b24_dev80" and r["family"]=="NP"),
        ("S4_FAST",lambda r:r["source"]=="S4_a15" and r["family"]=="FAST"),
    ]
    for label,test in requirements:
        options=[r for r in rows if test(r) and r["uid"] not in
                 {x["uid"] for _,x in chosen}]
        chosen.append((label,stable_pick(options,label)))
    for label in ("RANDOM_1","RANDOM_2"):
        options=[r for r in rows if r["uid"] not in
                 {x["uid"] for _,x in chosen}]
        chosen.append((label,stable_pick(options,label)))
    if len(chosen)!=8 or len({r["uid"] for _,r in chosen})!=8:
        raise RuntimeError("STOP_SMOKE_NOT_EIGHT_UNIQUE")
    return chosen,evidence

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--g0-receipt",required=True)
    ap.add_argument("--g1-receipt",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    out=Path(a.out)
    if not str(out).startswith(str(BASE)+"/") or out.exists():
        raise RuntimeError("STOP_SMOKE_OUTPUT")
    identity(a.repo,a.expected_head)
    manifest=Path(assert_no_forbidden(a.manifest))
    jr=verify_preflight(manifest,a.g0_receipt,a.g1_receipt)
    rows=read_manifest(manifest)
    chosen,evidence=choose(rows)
    # GPU workers see only candidate bytes/provenance, never source GT.
    stripped=[]
    for label,row in chosen:
        p=Path(assert_no_forbidden(row["candidate_path"]))
        if not p.is_file():
            raise RuntimeError("STOP_SMOKE_CANDIDATE_MISSING")
        stripped.append({
            "role":label,"uid":row["uid"],
            "candidate_path":str(p),"candidate_format":row["candidate_format"],
            "candidate_sha256":sha(p),"source":row["source"],
            "family":row["family"],"image_id":row["image_id"]
        })
    out.mkdir(parents=True,exist_ok=False)
    payload={"schema":"prdiff.track_r.r0.3b.smoke-selection.v1",
             "status":"R0_3B_SELECTION_FROZEN",
             "head":a.expected_head,
             "source_manifest_sha256":VALID_SHA,
             "g0_head":G0_HEAD,"g1_head":G1_HEAD,
             "gt_used_for_offline_stratum_only":True,
             "candidates_have_no_gt_fields":True,
             "candidates":stripped}
    select_file=out/"R0_3B_SMOKE_SELECTION.json"
    with select_file.open("x") as f:
        json.dump(payload,f,indent=2,sort_keys=True);f.write("\n")
    offline={"status":"R0_3B_SELECTION_OFFLINE_ONLY",
             "selected_uids":[r["uid"] for _,r in chosen],
             "roles":[name for name,_ in chosen],
             "stratum_validation":evidence,
             "g1_raw_psnr_db":jr["canonical_gt_orbit_psnr_db"]["000"],
             "g1_repaired_psnr_db":jr["repaired_psnr_db"],
             "selection_sha256":sha(select_file)}
    with (out/"R0_3B_OFFLINE_SELECTION_RECEIPT.json").open("x") as f:
        json.dump(offline,f,indent=2,sort_keys=True);f.write("\n")
    print("R0_3B_SELECTION_PASS",len(stripped),str(select_file),
          sha(select_file),flush=True)

if __name__=="__main__":
    sys.exit(main())
