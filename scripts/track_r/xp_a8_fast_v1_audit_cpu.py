#!/usr/bin/env python3
"""Independent CPU audit of the first prospective FFHQ FAST A16/A64-v1 return.

Re-opens only FFHQ R0-exposed-180 inputs, AFTER verifying the frozen v1
forecast/companion hashes and chronology. Recomputes all 10,800 new
candidates, each K-subset selection probability, and all interval/coverage
arithmetic. No new training, sampling, theory fitting, GPU, or ImageNet D1-P2.
Writes only one new PAC audit receipt; publication is a separate guarded step.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
from prdiffusion.track_r.physics import to_rgb8,rho_signed
from scripts.track_r.r0_2_g0_cpu import gt_source,read_tensor
from scripts.track_r.xp_a8_fast_v1_cpu import (
    ROOT,BASE,AUTHOR_WT,ORIGINAL,PINNED_A6_HEAD,
    POLICIES,GOOD,head_guard,register_first,historical,future_inputs,
    safe,sha
)

EXPECTED_SCORE_HEAD="f2bcb07dc8ff73c234cb45ae2b6cef3828f44660"
EXPECTED_RESULT_SHA="f1fde0ba2cfa74eca65a7b47ba27edb1af8aaa6ae747449d1e825b7ec650da8e"
EXPECTED_COUNTS={"D1_P2":1200,"D3":400,"EPRIME":720}
A6_RUN=BASE/"xp_a6_20261027T165112Z"

def demand(condition,detail):
    if not condition:
        raise RuntimeError("STOP_A8_FAST_AUDIT_"+detail)

def independently_label(x,g8,y):
    """Independent PSNR implementation: minimize raw/180 MSE by channel."""
    x8=to_rgb8(x)[0].to(torch.float32)
    gt=g8[0].to(torch.float32)
    raw_ch=((x8-gt)**2).flatten(1).mean(1).tolist()
    flip_ch=((torch.flip(x8,(-2,-1))-gt)**2).flatten(1).mean(1).tolist()
    raw_mse=max(sum(raw_ch)/(3*255**2),1e-12)
    orbit_mse=max(sum(min(a,b) for a,b in zip(raw_ch,flip_ch))/(3*255**2),1e-12)
    raw=-10*math.log10(raw_mse)
    best=-10*math.log10(orbit_mse)
    rho=float(rho_signed(x,y)[0])
    demand(all(math.isfinite(v) for v in (raw,best,rho)) and
           rho>=0 and best+1e-10>=raw,"CANDIDATE_METRICS")
    return dict(rho=rho,good_raw=raw>=GOOD,good_orbit=best>=GOOD)

def replay(identities,baseline,a16,a64):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    groups={}
    n_replayed=0
    for cohort,iid in sorted(identities):
        gtpath,gtfmt,ypath,yhash=identities[(cohort,iid)]
        demand(sha(ypath)==yhash,"Y_HASH")
        y=read_tensor(safe(ypath),("measurement","y","observation"),
                      (1,3,384,384))
        g8=to_rgb8(gt_source({"gt_path":gtpath,"gt_format":gtfmt}).float())
        candidates=dict(baseline[(cohort,iid)])
        demand(set(candidates)==set(range(4)),"HIST_FOUR")
        for rep in range(4,64):
            sources=a16 if rep<16 else a64
            entry=sources.get((cohort,iid,rep))
            demand(entry is not None,"MISSING_NATIVE")
            frozen,pt=entry
            demand(frozen["measurement_path"]==ypath and
                   frozen["measurement_sha256"]==yhash,"MEAS_LINEAGE")
            record=torch.load(safe(pt),map_location="cpu",weights_only=True)
            demand(isinstance(record,dict) and
                   isinstance(record.get("reconstruction"),torch.Tensor),
                   "BAD_RECON_CONTAINER")
            x=record["reconstruction"].float()
            demand(tuple(x.shape)==(1,3,256,256) and
                   bool(torch.isfinite(x).all()),"BAD_RECON")
            candidates[rep]=dict(run=rep,**independently_label(x,g8,y))
            n_replayed+=1
        demand(set(candidates)==set(range(64)),"INCOMPLETE_PANEL")
        groups[(cohort,iid)]=candidates
        if n_replayed%1200==0:
            print("A8_FAST_AUDIT_REPLAY",n_replayed,"/10800",flush=True)
    demand(len(groups)==180 and n_replayed==10800,"REPLAY_COUNT")
    return groups,n_replayed

def exact_probs(group,n,k):
    """Separate combinatorial aggregation with explicit rank tie breaking."""
    runs=[group[r] for r in range(n)]
    demand(len(runs)==n and 1<=k<=n,"BAD_K")
    ordered=sorted(runs,key=lambda r:(r["rho"],r["run"]))
    denom=math.comb(n,k)
    e0=math.fsum(
        (int(r["good_raw"])*math.comb(n-i-1,k-1)/denom)
        for i,r in enumerate(ordered) if n-i>=k)
    def oracle(attr):
        hits=sum(int(v[attr]) for v in runs)
        return 1-(math.comb(n-hits,k)/denom if n-hits>=k else 0)
    return {"E0":e0,"O_raw":oracle("good_raw"),
            "O_orbit":oracle("good_orbit")}

def verify_results(record, forecasts, groups):
    demand(record.get("status")=="A8_FAST_V1_FIRST_PROSPECTIVE_COMPARISON_PASS",
           "RESULT_STATUS")
    demand(record.get("analysis_head")==EXPECTED_SCORE_HEAD and
           record.get("forecast_refit_performed") is False and
           record.get("forecasts_frozen_before_outcomes_verified") is True,
           "RESULT_FREEZE")
    demand(record.get("D1_P2_outcomes_opened") is False and
           record.get("B16_F8_outcomes_opened") is False and
           record.get("no_theory_inputs_modified") is True,
           "RESULT_EMBARGO")
    demand(record.get("n_historical")==720 and record.get("n_A16")==2160
           and record.get("n_A64")==8640 and record.get("n_images")==180
           and record.get("per_image_run_indices")==list(range(64))
           and record.get("scope")=="FFHQ_EXPOSED_180_ONLY"
           and record.get("policies")==list(POLICIES),"RESULT_POPULATION")
    cohort_groups={"B22":[],"DEV80":[]}
    for (cohort,iid),data in groups.items():
        key={"S1_b22":"B22","S2_b24_dev80":"DEV80"}.get(cohort)
        demand(key is not None,"UNKNOWN_COHORT")
        cohort_groups[key].append(data)
    demand(len(cohort_groups["B22"])==100 and
           len(cohort_groups["DEV80"])==80,"COHORT_COUNTS")
    cohort_groups["POOLED"]=cohort_groups["B22"]+cohort_groups["DEV80"]
    reported=record.get("forecast_comparison")
    demand(isinstance(reported,list) and len(reported)==360,"COMPARISON_COUNT")
    index={}
    variants={"race","S2","tail","race_tau1"}
    for row in reported:
        key=tuple(row.get(k) for k in ("family","cohort","panel",
                   "panel_n_max","model_variant","K","policy"))
        demand(key not in index and row["family"]=="FAST" and
               row["model_variant"] in variants and
               row["policy"] in POLICIES,"COMPARISON_KEY")
        index[key]=row
    expected={k:f for k,f in forecasts.items() if k[0]=="FAST" and k[6] in POLICIES}
    demand(set(index)==set(expected),"FORECAST_KEY_COVERAGE")
    memo={}
    covered=Counter()
    group_stats=defaultdict(list)
    for key,forecast in expected.items():
        family,cohort,panel,n,variant,k,policy=key
        demand(cohort in cohort_groups and 1<=k<=n<=64,"FORECAST_PANEL")
        mk=(cohort,n,k)
        if mk not in memo:
            images=cohort_groups[cohort]
            observed={p:math.fsum(exact_probs(g,n,k)[p]
                                  for g in images)/len(images) for p in POLICIES}
            demand(0<=observed["E0"]<=observed["O_raw"]+1e-12
                   and observed["O_raw"]<=observed["O_orbit"]+1e-12,
                   "ORACLE_DOMINANCE")
            memo[mk]=observed
        obs=memo[mk][policy]
        entry=index[key]
        point,lo,hi=(float(forecast[x]) for x in ("point","lo95","hi95"))
        demand(lo<=point<=hi and hi>lo and
               forecast["interval_type"]=="predictive_realized_panel",
               "PREDICTIVE_INTERVAL")
        inside=lo<=obs<=hi
        resid=abs(obs-point)/max((hi-lo)/3.92,0.005)
        demand(abs(float(entry["observed"])-obs)<=1e-9,"OBS_MISMATCH")
        demand(all(abs(float(entry[x])-v)<=1e-12 for x,v in
                   (("point",point),("lo95",lo),("hi95",hi))),
               "FROZEN_POINT_MISMATCH")
        demand(entry["inside"] is inside and
               abs(float(entry["standardized_residual"])-resid)<=1e-8,
               "INTERVAL_MISMATCH")
        covered[variant]+=int(inside)
        group_stats[(variant,panel,policy,cohort)].append((inside,resid))
    summary=record.get("per_variant_coverage")
    demand(isinstance(summary,list) and len(summary)==72,"SUMMARY_COUNT")
    actual=set()
    for item in summary:
        key=tuple(item[x] for x in ("variant","panel","policy","cohort"))
        demand(key in group_stats and key not in actual,"SUMMARY_KEY")
        actual.add(key)
        vals=group_stats[key]
        demand(item["n_grid"]==len(vals) and
               item["n_covered"]==sum(x[0] for x in vals) and
               abs(float(item["mean_standardized_residual"])-
                   math.fsum(x[1] for x in vals)/len(vals))<=1e-8,
               "SUMMARY_NUMERICAL")
    demand(actual==set(group_stats),"SUMMARY_COVERAGE")
    return dict(n_forecast_rows=len(index),
                n_summary_rows=len(summary),
                n_distinct_observed_panels=len(memo),
                coverage_by_variant=dict(covered),
                all_observations_independently_recomputed=True)

def a6_final_readonly():
    state=A6_RUN/"XP_A6_CONTROLLER_STATE.json"
    demand(state.is_file() and not state.is_symlink(),"A6_STATE_MISSING")
    data=json.loads(state.read_text())
    done=Counter()
    for item in data.get("done",[]):
        stage,start,count=item.rsplit("_",2)
        done[stage]+=int(count)
    demand(data.get("status")=="MAIN_PASS" and
           data.get("completed_roots")==2320 and
           data.get("active_jobs")==[] and
           data.get("outcomes_scored") is False and
           all(done[stage]==count for stage,count in EXPECTED_COUNTS.items())
           and sum(done.values())==2320,"A6_FINAL_STATE")
    return dict(status="A6_MAIN_PASS_STATE_VERIFIED",
                state_sha256=sha(state),
                completed_roots=2320,stage_counts=dict(done),
                gpu_hours=data.get("gpu_hours"),
                timing_unverified_jobs=data.get("timing_unverified_jobs"),
                outcomes_scored=False)

def main(args):
    demand(os.environ.get("CUDA_VISIBLE_DEVICES")=="","GPU_VISIBLE")
    head_guard(args.expected_head)
    result=safe(args.result)
    demand(sha(result)==EXPECTED_RESULT_SHA,"RESULT_SOURCE_SHA")
    out=Path(args.out)
    demand(out.parent==result.parent and not out.exists() and
           out.name=="A8_FAST_V1_INDEPENDENT_AUDIT.json","OUTPUT_PATH")
    _,_,forecasts=register_first(EXPECTED_SCORE_HEAD)
    print("A8_FAST_AUDIT_FROZEN_FORECAST_PASS",len(forecasts),flush=True)
    identities,historical_runs=historical()
    a16,a64=future_inputs()
    groups,n=replay(identities,historical_runs,a16,a64)
    src=json.loads(result.read_text())
    summary=verify_results(src,forecasts,groups)
    a6=a6_final_readonly()
    receipt=dict(status="A8_FAST_V1_INDEPENDENT_AUDIT_PASS",
                 original_result_path=str(result),
                 original_result_sha256=EXPECTED_RESULT_SHA,
                 scorer_head=EXPECTED_SCORE_HEAD,
                 auditor_head=args.expected_head,
                 n_new_reconstructions_independently_rescored=n,
                 frozen_forecasts_verified_before_outcomes=True,
                 no_theory_refit=True,
                 compared=summary,a6_final=a6)
    with out.open("x") as f:
        json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("A8_FAST_V1_INDEPENDENT_AUDIT_PASS",
          summary["n_forecast_rows"],"comparisons",
          n,"new reconstructions",flush=True)
    print("A8_FAST_V1_COVERAGE_BY_VARIANT",summary["coverage_by_variant"],flush=True)
    print("A6_FINAL_VERIFIED",a6["stage_counts"],flush=True)
    print("A8_FAST_V1_AUDIT_PATH="+str(out),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--result",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A8_FAST_V1_AUDIT",str(e),flush=True)
        sys.exit(8)
