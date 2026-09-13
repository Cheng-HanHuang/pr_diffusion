#!/usr/bin/env python3
"""Aggregate final PE3 DEV80 refinement and apply the frozen NP4 gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

import torch

ARMS = ("NP_PE3_SCORE", "NP_PE3_RANDOM")


def readj(path: Path): return json.loads(path.read_text(encoding="utf-8"))

def writej(path: Path, value):
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)

def unwrap(path: Path, keys):
    v=torch.load(path,map_location="cpu")
    if torch.is_tensor(v): return v.float()
    if isinstance(v,dict):
        for k in keys:
            if torch.is_tensor(v.get(k)): return v[k].float()
    raise RuntimeError(path)

def q01(x): return torch.round(((x.clamp(-1,1)+1)*0.5)*255).clamp(0,255)/255.0

def psnr(x,y):
    mse=torch.mean((x-y).square()).clamp_min(1e-12)
    return float((10*torch.log10(1.0/mse)).item())

def percentile(v,q):
    s=sorted(map(float,v)); p=q*(len(s)-1); lo=int(math.floor(p)); hi=int(math.ceil(p))
    if lo==hi:return s[lo]
    return s[lo]*(hi-p)+s[hi]*(p-lo)

def dist(v):
    return {"n":len(v),"mean_db":statistics.fmean(v),"median_db":statistics.median(v),"min_db":min(v),"q10_db":percentile(v,.1),"q25_db":percentile(v,.25),"good25_count":sum(x>=25 for x in v),"good25_rate":sum(x>=25 for x in v)/len(v)}

def paired(a,b):
    d=[x-y for x,y in zip(a,b)]
    return {"n":len(d),"delta_mean_db":statistics.fmean(d),"delta_median_db":statistics.median(d),"wins_ties_losses":[sum(x>0 for x in d),sum(x==0 for x in d),sum(x<0 for x in d)],"good25_rescues":sum(x>=25 and y<25 for x,y in zip(a,b)),"good25_harms":sum(x<25 and y>=25 for x,y in zip(a,b)),"large_rescues_ge5db":sum(x>=5 for x in d),"large_harms_le_minus5db":sum(x<=-5 for x in d)}

def canonical_result(path: Path, gt01: torch.Tensor):
    r=readj(path)
    if r.get("status")!="PASS" or r.get("arm") not in ARMS or int(r.get("total_unet_evals",-1))!=8800:
        raise RuntimeError(f"bad PE3 result {path}")
    if bool(r.get("runtime_decisions_use_ground_truth",True)) or bool(r.get("terminal_selection_uses_ground_truth",True)):
        raise RuntimeError(f"GT-use drift {path}")
    terms=r.get("terminals",[])
    if len(terms)!=4: raise RuntimeError(f"terminal count drift {path}")
    scores=[]
    for t in terms:
        rec=unwrap(Path(t["reconstruction_path"]),("reconstruction","image","x"))
        scores.append(psnr(q01(rec),gt01))
    sel=int(r["clean_free_selected_terminal_index"])
    oracle=max(range(4),key=lambda i:(scores[i],-i))
    return {"selected":scores[sel],"oracle":scores[oracle],"selector_gap":scores[oracle]-scores[sel],"selected_idx":sel,"oracle_idx":oracle}

def passes_gate(pair_stats, method_good25, np4_good25):
    checks={
      "median_delta_nonnegative": pair_stats["delta_median_db"]>=0.0,
      "good25_count_at_least_np4": method_good25>=np4_good25,
      "good25_rescues_at_least_harms": pair_stats["good25_rescues"]>=pair_stats["good25_harms"],
      "large_rescues_at_least_harms": pair_stats["large_rescues_ge5db"]>=pair_stats["large_harms_le_minus5db"],
    }
    return checks,all(checks.values())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",type=Path,required=True); args=ap.parse_args()
    run=args.run.resolve(); manifest=readj(run/"PE3_MANIFEST.json")
    if manifest.get("image_count")!=80 or manifest.get("confirmation_exposed") is not False: raise RuntimeError("bad PE3 manifest")
    src=Path(manifest["source_dev80_run"]); src_csv=src/"DEV80_PER_IMAGE.csv"
    with src_csv.open(newline="",encoding="utf-8") as f: source={r["image_id"]:dict(r) for r in csv.DictReader(f)}
    if len(source)!=80: raise RuntimeError("source DEV80 CSV drift")

    rows=[]
    for task in manifest["tasks"]:
        td=run/"workers"/f"gpu{task['assigned_gpu']}"/task["output_subdir"]
        comp=readj(td/"PE3_COMPLETE.json")
        if comp.get("status")!="PASS" or comp.get("image_id")!=task["image_id"]: raise RuntimeError(f"bad completion {td}")
        inp=readj(Path(task["input_manifest"])); gt01=q01(unwrap(Path(inp["ground_truth_tensor_path"]),("ground_truth","image","x")))
        sr=source[task["image_id"]]
        row={
          "image_id":task["image_id"],"screening_stratum":task["class_label"],"fresh_baseline_class":task["fresh_baseline_class"],
          "np4_selected_psnr_8bit_db":float(sr["np4_independent_selected_psnr_8bit_db"]),
          "np4_oracle_psnr_8bit_db":float(sr["np4_independent_oracle_psnr_8bit_db"]),
          "daps1_psnr_8bit_db":float(sr["daps1_psnr_8bit_db"]),"daps4_oracle_psnr_8bit_db":float(sr["daps4_oracle_psnr_8bit_db"]),
          "sitcom1_psnr_8bit_db":float(sr["sitcom1_psnr_8bit_db"]),"sitcom4_oracle_psnr_8bit_db":float(sr["sitcom4_oracle_psnr_8bit_db"]),
        }
        for arm in ARMS:
            c=canonical_result(td/"methods"/arm/"result.json",gt01); key=arm.lower()
            row[f"{key}_selected_psnr_8bit_db"]=c["selected"]; row[f"{key}_oracle_psnr_8bit_db"]=c["oracle"]; row[f"{key}_selector_gap_8bit_db"]=c["selector_gap"]
        rows.append(row)
    if len(rows)!=80 or len({r['image_id'] for r in rows})!=80: raise RuntimeError("PE3 count drift")
    rows.sort(key=lambda r:(r["screening_stratum"],r["image_id"]))
    def vec(k,sub=rows): return [float(r[k]) for r in sub]
    np4=vec("np4_selected_psnr_8bit_db"); np4_good=sum(x>=25 for x in np4)
    summaries={}; passing=[]
    for arm in ARMS:
        key=arm.lower(); sel=vec(f"{key}_selected_psnr_8bit_db"); oracle=vec(f"{key}_oracle_psnr_8bit_db")
        p=paired(sel,np4); checks,ok=passes_gate(p,sum(x>=25 for x in sel),np4_good)
        summaries[arm]={
          "selected_distribution":dist(sel),"oracle_distribution":dist(oracle),"vs_np4_selected":p,
          "vs_np4_oracle":paired(oracle,vec("np4_oracle_psnr_8bit_db")),"advancement_gate_checks":checks,"passes_frozen_np4_gate":ok,
          "vs_daps1_selected":paired(sel,vec("daps1_psnr_8bit_db")),"vs_daps4_oracle_ceiling":paired(sel,vec("daps4_oracle_psnr_8bit_db")),
          "vs_sitcom1_selected":paired(sel,vec("sitcom1_psnr_8bit_db")),"vs_sitcom4_oracle_ceiling":paired(sel,vec("sitcom4_oracle_psnr_8bit_db")),
        }
        if ok: passing.append(arm)
    bothfail=[r for r in rows if r["daps4_oracle_psnr_8bit_db"]<25 and r["sitcom4_oracle_psnr_8bit_db"]<25]
    rescue={arm:[r["image_id"] for r in bothfail if r[f"{arm.lower()}_selected_psnr_8bit_db"]>=25] for arm in ARMS}
    decision="STOP_B24_METHOD_REFINEMENT" if not passing else "PLANNER_REVIEW_PASSING_PE3_BEFORE_ANY_CONFIRMATION"
    summary={
      "schema_version":"b24.pe3-dev80-summary.v1","status":"PASS","image_count":80,"confirmation_exposed":False,
      "source_dev80_run":str(src),"primary_representation":"CANONICAL_SAVED_OR_QUANTIZED_RGB_8BIT_RAW_ORIENTATION_V1",
      "np4_good25_count":np4_good,"arms":summaries,"fresh_both_baselines_fail_count":len(bothfail),
      "fresh_both_baselines_fail_image_ids":[r["image_id"] for r in bothfail],"pe3_rescues_where_both_fresh_baselines_fail":rescue,
      "passing_arms":passing,"decision":decision,"gate_frozen_before_execution":True,
      "next":"STOP method refinement if passing_arms empty; otherwise planner review and separate freeze only. Confirmation remains locked.",
    }
    outcsv=run/"PE3_DEV80_PER_IMAGE.csv"
    with outcsv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    writej(run/"PE3_DEV80_SUMMARY.json",summary)
    print(json.dumps(summary,sort_keys=True)); print(f"PER_IMAGE_CSV|{outcsv}"); print(f"SUMMARY_JSON|{run/'PE3_DEV80_SUMMARY.json'}")
    return 0

if __name__=="__main__": raise SystemExit(main())
