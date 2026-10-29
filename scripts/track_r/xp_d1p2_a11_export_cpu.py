#!/usr/bin/env python3
"""A11 frozen protocol held-out D1-P2 16-run strings and G1/G2 CLIP rows.

Converts already-scored D1 phase-2 return into exact A9 protocol schemas.
No GT, new reconstruction, CLIP inference or GPU needed. The D1 aggregate
source was completed before protocol freeze, but *per-image strings* were not
published to planner until after protocol v1 froze.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,traceback
from collections import Counter
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
D1=BASE/"xp_a10_d1p2_fixed_20261029T044818Z"
D1_SHA="e2c4d4b4993bf8f42a5e87a9978090823ea0aad4c28698ca40a626e9e68ec1e5"
P1=WT/"docs/track_r/executors/xp/theory_inputs/XP_D1_P1_K4_CLIP_DECISIONS.csv"
P1_SHA="0a2989180b67afa6fc9c6fccdffe7680030f4214d7b672ecd4fe50e5b010cf44"
STR_FIELDS=("image_id","cohort","labels_by_run","rho_signed_by_run",
            "tau_signed_good_q95","n_G","n_O_g","n_O_p","n_C","n_I")
CLIP_FIELDS=("image_id","run","label","g1_choice","g1_margin",
             "g1_chosen_good","g2_choice","g2_margin","g2_chosen_good")
LETTERS={"G":"G","O":"O_g","P":"O_p","C":"C","I":"I"}
def need(ok,why):
    if not ok:raise RuntimeError("STOP_D1P2_A11_EXPORT_"+why)
def sha(path):
    with Path(path).open("rb") as f:
        h=hashlib.sha256()
        for buf in iter(lambda:f.read(1<<20),b""):h.update(buf)
    return h.hexdigest()
def load_csv(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))
def git(*args):
    return subprocess.check_output(["git","-C",str(WT),*args],timeout=120)
def freeze():
    path="docs/track_r/protocol/gtfree_protocol.py"
    spec="docs/track_r/protocol/GTFREE_PROTOCOL_v1.md"
    files={path:"d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658",
           spec:"7e9a5fd8b817a97351d07ae371b947688c8430cd5b9268fc99b954a493223919"}
    for name,digest in files.items():
        blob=git("show","c598176:"+name)
        need(hashlib.sha256(blob).hexdigest()==digest,
             "FROZEN_PROTOCOL_SHA_"+name)
    registry=git("show","6585619:"+ "docs/track_r/theory/PREDICTIONS_REGISTRY.md").decode()
    need("Frozen protocols" in registry and "protocol v1" in registry,
         "FROZEN_REGISTRY_RECORD")
    return files
def write_x(path,obj):
    with Path(path).open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def main(a):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    need(git("rev-parse","HEAD").decode().strip()==a.expected_head
         and not git("status","--porcelain"),"HEAD")
    file_sha=freeze()
    result=D1/"XP_A10_D1P2_COMPARISON.json"
    receipt=json.loads((D1/"XP_A10_D1P2_RECEIPT.json").read_text())
    need(sha(result)==D1_SHA,"D1P2_COMPARISON_PIN")
    need(receipt["status"]=="A10_D1P2_PROSPECTIVE_COMPARISON_PASS",
         "D1P2_RECEIPT")
    for name,digest in receipt["sha256"].items():
        need(sha(D1/name)==digest,"D1P2_SOURCE_FILE_"+name)
    study=json.loads(result.read_text())
    need(study["n_images"]==100 and study["n_panel"]==1600 and
         study["n_phase1"]==400 and study["n_phase2"]==1200,
         "D1P2_FROZEN_GRID")
    tau=float(study["tau_good20"])
    need(0.04<tau<0.06,"FROZEN_LABEL_TAU")
    check=load_csv(D1/"XP_A10_D1P2_PER_IMAGE.csv")
    phase1=load_csv(P1)
    phase2=load_csv(D1/"XP_A10_D1P2_CLIP_G1G2.csv")
    need(sha(P1)==P1_SHA and len(check)==100
         and len(phase1)==400 and len(phase2)==1200,"SOURCE_COUNTS")
    rows=[]
    expected={}
    for src in check:
        iid=src["image_id"]
        lab=list(src["run_labels"])
        rho=src["rho_signed_by_run"].split(";")
        rank=[int(x) for x in src["e0_rank_order"].split(",")]
        need(len(lab)==len(rho)==16 and
             set(rank)==set(range(16)) and all(x in LETTERS for x in lab),
             "STRINGS_ROW_"+iid)
        precise=[float(x) for x in rho]
        need(sorted(range(16),key=lambda j:(precise[j],j))==rank,
             "RANK_RESIDUAL_MISMATCH_"+iid)
        labels=[LETTERS[x] for x in lab]
        counts=Counter(labels)
        need(iid not in expected,"DUPLICATE_IMAGE")
        for rep,c in enumerate(lab):
            expected[iid,rep]=(c,int(src["g1_chosen_good_by_run"][rep]),
                               int(src["g2_chosen_good_by_run"][rep]))
        rows.append(dict(image_id=iid,cohort="IN",
             labels_by_run=";".join(labels),
             rho_signed_by_run=";".join(rho),
             tau_signed_good_q95=tau,
             n_G=counts["G"],n_O_g=counts["O_g"],n_O_p=counts["O_p"],
             n_C=counts["C"],n_I=counts["I"]))
    clip={}
    for p in phase1+phase2:
        iid=p["image_id"];rep=int(p["run"]);key=(iid,rep)
        need(key in expected and key not in clip,
             "CLIP_UNIQUE_"+str(key))
        c,g1,g2=expected[key]
        need(p["label"]==c and
             int(p["g1_chosen_good"])==g1 and
             int(p["g2_chosen_good"])==g2,
             "CHOICE_LABEL_DRIFT_"+str(key))
        item=dict(image_id=iid,run=rep,label=c)
        for arm in ("g1","g2"):
            choice=p[arm+"_choice"]
            margin=float(p[arm+"_margin"])
            need(choice in ("000","111") and
                 choice==("111" if margin>0 else "000"),
                 "CLIP_MARGIN_SIGN_"+str(key))
            item[arm+"_choice"]=choice
            item[arm+"_margin"]=p[arm+"_margin"]
            item[arm+"_chosen_good"]=int(p[arm+"_chosen_good"])
        clip[key]=item
    need(set(clip)==set(expected) and len(clip)==1600,"CLIP_1600_COVERAGE")
    out=Path(a.out)
    need(out.parent==BASE and out.name.startswith("xp_d1p2_protocol_")
         and not out.exists(),"FRESH_OUTPUT")
    out.mkdir()
    sfile=out/"XP_D1P2_16RUN_STRINGS.csv"
    cfile=out/"XP_D1P2_16RUN_CLIP_DECISIONS.csv"
    with sfile.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=STR_FIELDS)
        w.writeheader();w.writerows(sorted(rows,key=lambda r:r["image_id"]))
    with cfile.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=CLIP_FIELDS)
        w.writeheader();w.writerows(clip[k] for k in sorted(clip))
    receipt=dict(status="XP_D1P2_A11_1600_FROZEN_PROTOCOL_INPUTS_PASS",
       n_images=100,n_rows=1600,
       D1_comparison_sha256=D1_SHA,D1_receipt_sha256=sha(D1/"XP_A10_D1P2_RECEIPT.json"),
       P1_CLIP_SHA=P1_SHA,protocol_sha256=file_sha,
       strings_sha256=sha(sfile),clip_sha256=sha(cfile),
       frozen_labels_tau_good20=tau,
       no_new_GT_opened=True,no_new_CLIP_inference=True,no_GPU_called=True)
    write_x(out/"XP_D1P2_A11_INPUT_RECEIPT.json",receipt)
    print("XP_D1P2_PROTOCOL_1600_INPUTS_PASS",flush=True)
    print("D1P2_PROTOCOL_INPUT_DIR="+str(out),flush=True)
    print("D1P2_STRINGS_SHA="+sha(sfile),flush=True)
    print("D1P2_CLIP_SHA="+sha(cfile),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: D1P2_A11_EXPORT "+str(e),flush=True)
        sys.exit(8)
