#!/usr/bin/env python3
"""CPU-only B16 v1-primary/v3-secondary prereg and root-SHA admission.

Runs BEFORE opening any B16 GT or PSNR. Separate worktree from live F8.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a11_b16_handoff_autopilot import verify_task
from scripts.track_r.xp_a10_b16_gate_prepare_cpu import ROOT,BASE,sha
WT=Path(__file__).resolve().parents[2]
RUN=BASE/"xp_a11_b16_prod_20261028T113034Z"
GEN_HEAD="079228b3ebe55dba2ac7bc7ae2c8187113290409"
B16_SHA="778ce4fa7d5ef9299e2530bbb32e8a28b3512e5abe97f0569bb3ec74e00247a9"
B16_RECEIPT_SHA="5a3426464fb1164d98138bfbe52212fe38036f4401cf24c1379b6a283ac9e88d"
REGISTRY_REF="356ec2a87f73eea905cb2930909ff3053b93e18b"
REGISTRY_PATH="docs/track_r/theory/PREDICTIONS_REGISTRY.md"
FREEZES={
 "v1":("8ef5f94d6de8aab4c3cc25b17fa797db0de1560b",
       "docs/track_r/theory/TH1_PREDICTIONS_v1.json",
       "c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753",976),
 "v3":("e71026ef5c632fa5f765589f978abf33b6259b76",
       "docs/track_r/theory/TH1_PREDICTIONS_v3.json",
       "9feeef5f5b28593122d45983ca38e99d95d9bfdee45fdfcbfee8a2fc66d5794d",496)}
def check(cond,why):
    if not cond:raise RuntimeError("STOP_B16_V1_V3_"+why)
def git(*args):
    return subprocess.check_output(["git","-C",str(WT),*args],timeout=90)
def hash_bytes(data):return hashlib.sha256(data).hexdigest()
def freeze():
    # No prospective outcome source is opened until all freeze checks pass.
    registry=git("show",REGISTRY_REF+":"+REGISTRY_PATH).decode()
    results={}
    for key,(ref,path,expected,n) in FREEZES.items():
        rows=[line for line in registry.splitlines()
              if line.startswith("| "+key+" |")]
        check(len(rows)==1,"REGISTRY_ENTRY_"+key)
        cells=[v.strip() for v in rows[0].strip("|").split("|")]
        check(len(cells)>=8 and cells[2].strip(chr(96))==ref and
              cells[3].strip(chr(96))==path and expected in cells[4]
              and "REGISTERED" in cells[7],"REGISTRY_"+key)
        blob=git("show",ref+":"+path)
        check(hash_bytes(blob)==expected,"SHA_"+key)
        model=json.loads(blob)
        check(model["interval_type"]=="predictive_realized_panel"
              and len(model["predictions"])==n,"SCHEMA_"+key)
        pred=[r for r in model["predictions"]
              if r["family"]=="SITCOM" and r["panel"]=="primary"
              and r["panel_n_max"]==16]
        check(len(pred)==144 and
              all(sum(r["cohort"]==co for r in pred)==48
                  for co in ("B22","DEV80","POOLED")),
              "SITCOM_FORECAST_GRID_"+key)
        results[key]=dict(commit=ref,sha256=expected,
                          n_pred=n,n_primary_sitcom=len(pred))
    print("B16_V1_PRIMARY_V3_SECONDARY_REGISTRY_SHA_PASS",flush=True)
    return results
def source():
    receipt=RUN/"XP_A11_B16_PROD_COMPLETE.json"
    check(sha(receipt)==B16_RECEIPT_SHA,"COMPLETE_RECEIPT_SHA")
    complete=json.loads(receipt.read_text())
    check(complete["n_new_roots"]==2160 and
          complete["no_new_outcomes_opened"] is True,"COMPLETE_CONTENT")
    manifest=RUN/"XP_A11_B16_PROD_TASKS.json"
    prep=json.loads((RUN/"XP_A11_B16_PROD_PREP.json").read_text())
    check(sha(manifest)==B16_SHA==prep["manifest_sha256"],
          "FROZEN_MANIFEST")
    tasks=json.loads(manifest.read_text())
    groups=tasks["b22_groups"]+tasks["b24_groups"]
    check(len(groups)==1060 and
          sum(g["new_roots"] for g in groups)==2160 and
          len(tasks["b22_groups"])==100 and
          len(tasks["b24_groups"])==960,"TASK_GRID")
    check(len({g["image_id"] for g in tasks["b22_groups"]})==100
          and len({g["image_id"] for g in tasks["b24_groups"]})==80,
          "180_IMAGE_GRID")
    state=json.loads((RUN/"B16_PRODUCTION_STATE.json").read_text())
    check(state["active"]==[] and len(state["done"])==1060,
          "B16_STATE_NOT_FINAL")
    for i in range(1060):
        verify_task(RUN,groups,B16_SHA,i,GEN_HEAD)
        if (i+1)%100==0:
            print("B16_ROOT_HASH_AUDIT",i+1,"/1060",flush=True)
    print("B16_2160_NATIVE_ROOT_RECEIPTS_VERIFIED",flush=True)
    return dict(n_images=180,n_b22=100,n_dev80=80,
       n_new_roots=2160,n_groups=1060,
       manifest_sha256=B16_SHA,complete_receipt_sha256=B16_RECEIPT_SHA,
       generator_head=GEN_HEAD,GT_opened=False)
def main(a):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    check(git("rev-parse","HEAD").decode().strip()==a.expected_head
          and not git("status","--porcelain"),"WORKTREE_HEAD")
    forecasts=freeze()
    provenance=source()
    out=Path(a.out)
    check(out.parent==BASE and out.name.startswith("xp_b16_v1_v3_")
          and not out.exists() and not out.is_symlink(),"FRESH_OUTPUT")
    out.mkdir(parents=False,exist_ok=False)
    p=out/"B16_V1_V3_CPU_ADMISSION.json"
    payload=dict(status="B16_V1_V3_UNBLIND_ADMISSION_PASS",
       expected_head=a.expected_head,registry_commit=REGISTRY_REF,
       forecasts=forecasts,source=provenance,
       scoring_authorized=True,no_GT_bytes_read=True,
       no_new_B16_outcome_scored=True,
       independent_of_running_F8_checkout=True)
    with p.open("x") as f:
        json.dump(payload,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("B16_V1_V3_UNBLIND_ADMISSION_PASS",flush=True)
    print("B16_CPU_ADMISSION_RECEIPT_SHA="+sha(p),flush=True)
    print("B16_CPU_ADMISSION_DIR="+str(out),flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--out",required=True)
    try:main(parser.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: B16_V1_V3_ADMISSION "+str(e),flush=True)
        sys.exit(8)
