#!/usr/bin/env python3
"""N1: one CPU-only, read-only run to produce R0-N letter evidence N1a–N1g.

Requires N0's committed, SHA-pinned original aggregate R0.8 tables.
Never opens protected cohorts. All GT-related source work happened
offline in the frozen, immutable R0.7 evaluator; here we reuse its
evaluated rows and read only corresponding immutable score receipts.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

import numpy as np

from prdiffusion.track_r.r0_7_stats import (
    GOOD,BOOT_N,clustered_ratio_ci,summarize_psnr)
from prdiffusion.track_r.r0n_letter import (
    TIMES,EXPECTED_LABELS,P4_GOOD25_WITNESSES,
    assert_replay,split_ids,cohort,label_anatomy,postselection,
    residuals,cost_configs,score_from_draws,cost_grid_metrics,choose_dev,
    paired_variance,conditional_detection,table_two)
from scripts.track_r.r0n_0_import_tables_cpu import FILES,DEST as N0

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
R07=BASE/"r0_7_cpu_20261023T080434Z"
CONFIG=WT/"configs/track_r/r0n_letter_analysis_v1.json"
EXPECTED_R07_HEAD="c6dbd6089579c79e9e6fd4a9dd20fc402bc1185b"
CONFIG_STATUS="FROZEN_V1_2026-10-23"
FORBIDDEN=("confirmation","confirm305")
SOURCE_LABELS=("GOOD","ORBIT_ERROR","NONORBIT_CONSISTENT",
               "NONORBIT_INCONSISTENT")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):h.update(part)
    return h.hexdigest()

def file(path):
    p=Path(path)
    if (not p.is_absolute() or not p.is_file() or p.is_symlink() or
        any(tok in str(p.resolve()).lower() for tok in FORBIDDEN)):
        raise RuntimeError("STOP_R0N_N1_BAD_FILE "+str(p))
    return p

def load(path):
    return json.loads(file(path).read_text(encoding="utf-8"))

def save_x(path,payload):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def round_csv(v):
    if v is None:return ""
    if isinstance(v,(float,int)):return str(v)
    if isinstance(v,(dict,list)):return json.dumps(v,sort_keys=True)
    return str(v)

def csv_table(path,rows):
    fields=sorted({key for r in rows for key in r})
    if not fields:fields=["no_rows"]
    with Path(path).open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        for r in rows:w.writerow({k:round_csv(v) for k,v in r.items()})

def assert_inputs():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_R0N_N1_GPU_VISIBILITY")
    if (subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30)
        or not N0.is_dir()):
        raise RuntimeError("STOP_R0N_N1_GIT_DIRTY_OR_N0_NOT_COMMITTED")
    config=load(CONFIG)
    if (config.get("status")!=CONFIG_STATUS or
        config.get("schema")!="prdiff.track_r.r0n.letter_analysis.v1" or
        config["n1d_cost_accuracy"]["selection_rule"].find("dev success >= 0.95")<0 or
        config["n1d_cost_accuracy"]["J_subsets"]!=[1,2,4]):
        raise RuntimeError("STOP_R0N_N1_FROZEN_CONFIG_CHANGED")
    for name,want in FILES.items():
        if sha(file(N0/name))!=want:
            raise RuntimeError("STOP_R0N_N1_UNCOMMITTED_OR_MUTATED_N0 "+name)
    expected=load(R07/"analysis/R0_7_COMPLETE_RECEIPT.json")
    if (expected.get("status")!="R0_7_ANALYSIS_COMPLETE" or
        expected.get("n_scored")!=3240 or
        expected.get("no_protected_access") is not True or
        expected.get("gpu_calls")!=0 or
        expected.get("n_tau_unavailable")!=0 or
        expected.get("n_recorded_psnr_anomalies")!=0 or
        expected.get("head")!=EXPECTED_R07_HEAD or
        sha(file(R07/"analysis/R0_7_ANALYSIS.json"))!=expected["analysis_sha256"] or
        expected["analysis_sha256"]!=FILES["R0_7_ANALYSIS.json"] or
        sha(file(R07/"analysis/R0_7_SCORED_GT_ROWS.jsonl"))!=
          expected["evaluated_rows_sha256"]):
        raise RuntimeError("STOP_R0N_N1_R07_SOURCE_DRIFT")
    catalog=load(R07/"prepare/R0_7_OFFLINE_GT_CATALOG.json")
    if (catalog.get("status")!="R0_7_GT_OFFLINE_CATALOG_PASS" or
        catalog.get("n_candidates")!=3240 or
        catalog.get("head")!=EXPECTED_R07_HEAD or
        len(catalog["rows"])!=3240):
        raise RuntimeError("STOP_R0N_N1_CATALOG_DRIFT")
    # Read the original already-GT-evaluated JSONL without inspecting
    # or reopening any gt_path/candidate_path stored inside its rows.
    records=[]
    with file(R07/"analysis/R0_7_SCORED_GT_ROWS.jsonl").open(encoding="utf-8") as f:
        for line in f:records.append(json.loads(line))
    if len(records)!=3240 or len({r["uid"] for r in records})!=3240:
        raise RuntimeError("STOP_R0N_N1_R07_ROWS")
    report=load(N0/"R0_7_ANALYSIS.json")
    return config,records,report,catalog,expected

def replay_receipts(records,catalog):
    """Require all 3,240 saved per-draw arrays and score hashes to match R0."""
    byuid={r["uid"]:r for r in catalog["rows"]}
    if len(byuid)!=3240:raise RuntimeError("STOP_R0N_N1_RECEIPT_DUPLICATE")
    losses={}
    per_score_wall=[]
    for i,r in enumerate(records):
        src=byuid.get(r["uid"])
        if src is None:raise RuntimeError("STOP_R0N_N1_SCORE_UID")
        path=file(src["score_receipt_path"])
        if sha(path)!=src["score_receipt_sha256"]:
            raise RuntimeError("STOP_R0N_N1_SCORE_SHA "+r["uid"])
        score=load(path)
        prior=score["prior"]
        dl=prior["draw_losses"]
        if (score["uid"]!=r["uid"] or score.get("gt_access") is not False or
            score["unet_forwards"]!=256 or
            set(dl)!={f"{k:03b}" for k in range(8)} or
            any(len(dl[p])!=32 for p in dl) or
            any(not all(math.isfinite(float(v)) for v in dl[p]) for p in dl) or
            score["s_star"]!=r["s_star"]):
            raise RuntimeError("STOP_R0N_N1_DRAW_SCHEMA "+r["uid"])
        for pattern,draws in dl.items():
            a=sum(float(v) for v in draws)/32
            if abs(a-prior["S"][pattern])>1e-5:
                raise RuntimeError("STOP_R0N_N1_DRAW_REPLAY "+r["uid"])
            mid=sum(float(v) for v in draws[12:28])/16
            if abs(mid-prior["S_mid"][pattern])>1e-5:
                raise RuntimeError("STOP_R0N_N1_MID_REPLAY "+r["uid"])
        losses[r["uid"]]=dl
        per_score_wall.append(float(score["gpu_score_wall_seconds"]))
        if (i+1)%500==0:
            print("R0N_N1_SCORE_RECEIPTS",i+1,"/3240",flush=True)
    return losses,per_score_wall

def generate_figures(out,n1a,n1c,grid,selection,records,split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    figdir=out/"figures"
    figdir.mkdir()
    # Fig 1: stacked count fractions by cohort/family (all frozen labels).
    fig,ax=plt.subplots(figsize=(11,5.2),constrained_layout=True)
    order=sorted(n1a,key=lambda r:(r["cohort"],r["family"]))
    xs=np.arange(len(order));bottom=np.zeros(len(order))
    colors=["#4c78a8","#f2a541","#72b7b2","#e45756"]
    labs=["GOOD","ORBIT_ERROR","NONORBIT_CONSISTENT","NONORBIT_INCONSISTENT"]
    for label,color in zip(labs,colors):
        vals=np.asarray([r[label]/r["n_runs"] for r in order])
        ax.bar(xs,vals,bottom=bottom,color=color,label=label)
        bottom+=vals
    ax.set_xticks(xs,[r["cohort"].replace("_random","").replace("_enriched","")+"\n"+r["family"]
                       for r in order],rotation=45,ha="right")
    ax.set_ylabel("Fraction of single-run reconstructions")
    ax.set_ylim(0,1.04);ax.legend(ncol=2,fontsize=8)
    ax.set_title("N1a: frozen single-run failure anatomy, by cohort")
    fig.savefig(figdir/"N1_FIG1_ANATOMY.png",dpi=180)
    fig.savefig(figdir/"N1_FIG1_ANATOMY.pdf")
    plt.close(fig)
    # Fig 2: cost–repair success on DEV80 and B22; marker size reflects harm.
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for cohort_name,color in (("DEV80_enriched","#4c78a8"),
                               ("B22_random","#e45756")):
        selected=[r for r in grid if r["cohort"]==cohort_name and
                  r["success"] is not None]
        ax.scatter([r["forwards"] for r in selected],
                   [r["success"] for r in selected],
                   s=[20+150*(r["harm"] or 0) for r in selected],
                   facecolors="none",edgecolors=color,alpha=.6,
                   label=cohort_name)
    if selection is not None:
        xx=selection["forwards"]
        ax.axvline(xx,ls="--",color="black",alpha=.45,
                   label="DEV80-selected configuration")
    ax.set_xlabel("UNet image forwards / candidate")
    ax.set_ylabel("Repair success on exact-orbit errors")
    ax.set_ylim(-.025,1.05);ax.legend(fontsize=8)
    ax.set_title("N1d: predeclared cost–accuracy grid (marker size ∝ harm)")
    fig.savefig(figdir/"N1_FIG2_COST_SUCCESS.png",dpi=180)
    fig.savefig(figdir/"N1_FIG2_COST_SUCCESS.pdf")
    plt.close(fig)
    # Fig 3: actual per-run signed-residual violins, by family
    # and random/hard-enriched cohort. No GT beyond frozen labels.
    fig,ax=plt.subplots(figsize=(12,5.2),constrained_layout=True)
    violins=[]
    labels=[]
    pos=[]
    for name in ("B22_random","DEV80_enriched"):
        for fam in sorted({r["family"] for r in records}):
            rr=[r["rho_signed_raw"] for r in records
                if cohort(r,split)==name and r["family"]==fam]
            if len(rr)<2:continue
            violins.append(rr);labels.append(name.replace("_random","").replace("_enriched","")+"\\n"+fam)
            pos.append(len(pos)+1)
    artwork=ax.violinplot(violins,positions=pos,
                         showmedians=True,showextrema=False,points=80)
    for i,body in enumerate(artwork["bodies"]):
        body.set_facecolor("#4c78a8" if labels[i].startswith("B22") else "#e45756")
        body.set_alpha(.65)
    ax.set_xticks(pos,labels,rotation=45,ha="right",fontsize=8)
    ax.set_ylabel("Signed measurement residual (raw candidate)")
    ax.set_title("N1c: per-run residual distributions by cohort and family")
    fig.savefig(figdir/"N1_FIG3_RESIDUAL.png",dpi=180)
    fig.savefig(figdir/"N1_FIG3_RESIDUAL.pdf")
    plt.close(fig)
    return {"fig1":"stacked fraction","fig2":"cost-success scatter",
            "fig3":"signed-residual per-run violin"}

def n1_costs(records,per_score_wall,chosen):
    def summarize(values):
        values=[float(v) for v in values if v is not None]
        return {"n":len(values),"mean":statistics.fmean(values) if values else None,
                "median":statistics.median(values) if values else None}
    grouped=defaultdict(list)
    for r in records:grouped[(r["source"],r["family"])].append(r)
    result=[]
    for (src,fam),rr in sorted(grouped.items()):
        result.append({"source":src,"family":fam,"n":len(rr),
             "score_wall_seconds":summarize([r["score_gpu_seconds"] for r in rr]),
             "native_recorded_cost_seconds":summarize(
                 [r.get("recorded_cost_seconds") for r in rr]),
             "FAST_sampling_wall_seconds":summarize(
                 [r.get("native_sample_wall_seconds") for r in rr]),
             "FAST_CUDA_event_seconds":summarize(
                 [r.get("native_cuda_seconds") for r in rr])})
    return {"by_source_family":result,
       "full_score_seconds":summarize(per_score_wall),
       "projected_selected_score_seconds":(
           statistics.fmean(per_score_wall)*chosen["forwards"]/256
           if chosen is not None else None),
       "projected_selected_forwards":chosen["forwards"] if chosen is not None else None,
       "projection_caveat":"throughput-scaled estimate, not a measured N1d GPU benchmark"}

def write_report(out,evidence,figures):
    lines=["# Track R R0-N N1 — Numbers-only letter evidence",
        "","All findings retrospective. B22 random and DEV80 hard-enriched",
        "are disjoint in image ID. No confirmation305 accessed.",
        "","## K3 replay","",
        "```json",json.dumps(evidence["K3_REPLAY"],sort_keys=True),"```",
        "","## N1a — cohort × family anatomy",
        "","| Cohort | Family | n | GOOD | Orbit | Global | Partial | Consistent | Inconsistent |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in evidence["N1a"]:
        lines.append(f"| {r['cohort']} | {r['family']} | {r['n_runs']} | {r['GOOD']} | {r['ORBIT_ERROR']} | {r['ORBIT_GLOBAL']} | {r['ORBIT_PARTIAL']} | {r['NONORBIT_CONSISTENT']} | {r['NONORBIT_INCONSISTENT']} |")
    lines+=["","## N1a image-cluster orbit-share confidence intervals","",
        "| Cohort | Family | Orbit share among raw failures | 95% CI |",
        "|---|---|---:|---|"]
    for r in evidence["N1a"]:
        a=r["orbit_share_failures"]
        val=a["estimate"]
        ci=a["ci95"]
        lines.append(
            f"| {r['cohort']} | {r['family']} | "+
            (f"{val:.4f}" if val is not None else "N/A")+" | "+
            (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A")+" |")
    lines+=["","## N1b — post-selection decomposition",
        "","| Cohort | Source | Family | K | Images | E0 failed | Fixed by E1 | Orbit uncorrected | Selection miss | Generation failure |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in evidence["N1b"]:
        lines.append(f"| {r['cohort']} | {r['source']} | {r['family']} | {r['K']} | {r['n_images']} | {r['E0_failures']} | {r['orbit_fixed']} | {r['orbit_not_fixed']} | {r['selection_miss']} | {r['generation_failure']} |")
    lines+=["","## N1c — signed/clipped residual distributions","",
        "Only observed cohort/family/label quantiles are reported;",
        "the relative overlap in SITCOM versus FAST/DAPS is a descriptive",
        "diagnostic, not an invented causal explanation.",
        "","| Cohort | Family | Label | n | Signed median [q10,q90] | Clipped median [q10,q90] |",
        "|---|---|---|---:|---|---|"]
    for r in evidence["N1c"]:
        signed=r["signed"];clipped=r["clipped"]
        lines.append(f"| {r['cohort']} | {r['family']} | {r['label']} | {r['n']} | "+
            f"{signed['median']:.5f} [{signed['q10']:.5f},{signed['q90']:.5f}] | "+
            f"{clipped['median']:.5f} [{clipped['q10']:.5f},{clipped['q90']:.5f}] |")
    lines+=["","## N1d — DEV80-selected repair configuration",
        "","```json",json.dumps(evidence["N1d"]["selected_on_DEV80"],sort_keys=True),
        "```","",
        "Eval B22 clustered confidence intervals:",
        "```json",json.dumps(evidence["N1d"]["selected_B22_evaluation"],sort_keys=True),
        "```","",
        "## N1e — costs","```json",json.dumps(evidence["N1e"],sort_keys=True),
        "```","","## N1f — exploratory conditional detection",
        "```json",json.dumps(evidence["N1f"],sort_keys=True),"```",
        "","## N1g — max-K deployable versus oracle",
        "","Each cell: Good25 fraction / mean PSNR dB / q05 PSNR dB.", "",
        "| Cohort | Source | Family | K | n | O_raw | O_orbit | E0 | E1 |",
        "|---|---|---|---:|---:|---|---|---|---|"]
    for r in evidence["N1g"]:
        def triple(x):
            return f"{x['good25']:.4f} / {x['mean']:.2f} / {x['q05']:.2f}"
        lines.append(f"| {r['cohort']} | {r['source']} | {r['family']} | {r['K']} | {r['n_images']} | {triple(r['O_raw'])} | {triple(r['O_orbit'])} | {triple(r['E0'])} | {triple(r['E1'])} |")
    lines+=["","## Figures",json.dumps(figures,sort_keys=True),
        "","## Handoff",
        "K1 is the B22 post-selection orbit_fixed count per family (N1b).",
        "K2 awaits optional N2 frozen CLIP zero-shot baseline.",
        "No letter kill-criterion decision is made by the executor.",
        "N1d's unselected 60-config grid is EXPLORATORY except for the",
        "pre-registered DEV80 selection and B22 evaluation.",
        ""] 
    (out/"R0N_REPORT.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    out=Path(a.out)
    if (not str(out).startswith(str(BASE)+"/") or out.exists() or
        out.is_symlink()):
        raise RuntimeError("STOP_R0N_N1_OUTPUT_NAMESPACE")
    if subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()!=a.expected_head:
        raise RuntimeError("STOP_R0N_N1_EXPECTED_HEAD")
    cfg,records,report,catalog,prior_receipt=assert_inputs()
    replay=assert_replay(records,report)
    ids=split_ids(records)
    print("R0N_N1_K3_REPLAY_PASS 3240 2040 85 324 791 REPAIR 84/85 HARM 0/2040",flush=True)
    n1a=label_anatomy(records,ids)
    n1b=postselection(records,ids,report)
    n1c=residuals(records,ids)
    losses,score_walls=replay_receipts(records,catalog)
    configs=cost_configs(cfg)
    decisions={uid:{c["name"]:score_from_draws(draws,c) for c in configs}
               for uid,draws in losses.items()}
    grid=cost_grid_metrics(records,decisions,configs,ids)
    selected=choose_dev(grid,configs)
    chosen=next((c for c in configs if
               selected is not None and c["name"]==selected["configuration"]),None)
    b22=next((r for r in grid if selected is not None and
        r["configuration"]==selected["configuration"] and
        r["cohort"]=="B22_random"),None)
    if b22 is None:
        evaluation={"status":"NO_DEV_QUALIFIED_CONFIGURATION"}
    else:
        evalrows=[r for r in records if cohort(r,ids)=="B22_random"]
        assignment={r["uid"]:decisions[r["uid"]][chosen["name"]] for r in evalrows}
        evaluation={"status":"DEV_SELECTED_EVALUATED_ON_B22",
            "configuration":chosen["name"],"B22":b22,
            "success_image_bootstrap":clustered_ratio_ci(
                evalrows,
                lambda r:r["raw_label"]=="ORBIT_ERROR" and
                 r["psnr_by_orbit"][assignment[r["uid"]]]>=GOOD,
                lambda r:r["raw_label"]=="ORBIT_ERROR",
                "r0n_cost_eval"),
            "harm_image_bootstrap":clustered_ratio_ci(
                evalrows,
                lambda r:r["raw_label"]=="GOOD" and
                 r["psnr_by_orbit"][assignment[r["uid"]]]<GOOD,
                lambda r:r["raw_label"]=="GOOD",
                "r0n_cost_eval")}
    variance=paired_variance(records,losses)
    n1e=n1_costs(records,score_walls,chosen)
    n1f=conditional_detection(records)
    n1g=table_two(records,n1b)
    evidence={
        "schema":"prdiff.track_r.r0n.n1.letter_evidence.v1",
        "status":"R0N_N1_CPU_EVIDENCE_PASS",
        "head":a.expected_head,
        "scope":"EXPOSED_DATA_DIAGNOSTIC_ONLY",
        "GT_access":"R0.7 evaluated labels only, no raw GT files",
        "CUDA_VISIBLE_DEVICES":"","new_gpu_seconds":0,
        "R0N_config_sha256":sha(CONFIG),
        "N0_analysis_sha256":sha(N0/"R0_7_ANALYSIS.json"),
        "R0_7_rows_sha256":prior_receipt["evaluated_rows_sha256"],
        "n_candidates":3240,"n_cost_configs":len(configs),
        "K3_REPLAY":replay,"cohort_counts":{
           label:len(vals) for label,vals in ids.items()},
        "N1a":n1a,"N1b":n1b,"N1c":n1c,
        "N1d":{"grid":grid,"selected_on_DEV80":selected,
               "selected_B22_evaluation":evaluation,
               "paired_variance":variance,
               "grid_label":"EXPLORATORY except DEV80-selected B22 evaluation"},
        "N1e":n1e,"N1f":n1f,"N1g":n1g,
        "K1_B22_postselection_fixes":[{
           "source":r["source"],"family":r["family"],"K":r["K"],
           "n_images":r["n_images"],"n_fixed":r["orbit_fixed"]}
           for r in n1b if r["cohort"]=="B22_random"],
        "K2_CLIP":"NOT_RUN_SEPARATE_OPTIONAL_N2"}
    # Fully evaluate first, then make a fresh immutable output root.
    out.mkdir(parents=True,exist_ok=False)
    save_x(out/"R0N_LETTER_EVIDENCE.json",evidence)
    for key,items in (("N1a_ANATOMY",n1a),("N1b_POSTSELECTION",n1b),
                      ("N1c_RESIDUALS",n1c),("N1d_GRID",grid),
                      ("N1d_PAIRED_VARIANCE",variance),("N1e_COSTS",n1e["by_source_family"]),
                      ("N1f_CONDITIONAL_DETECTION",n1f),("N1g_TABLE2",n1g)):
        csv_table(out/(key+".csv"),items)
    figures=generate_figures(out,n1a,n1c,grid,selected,records,ids)
    write_report(out,evidence,figures)
    receipt={"status":"R0N_N1_CPU_EVIDENCE_PASS","head":a.expected_head,
             "rows":3240,"n_configurations":len(configs),
             "source_r07_analysis_sha256":prior_receipt["analysis_sha256"],
             "evidence_sha256":sha(out/"R0N_LETTER_EVIDENCE.json"),
             "report_sha256":sha(out/"R0N_REPORT.md"),
             "new_gpu_seconds":0,
             "forbidden_cohort_access":False,
             "needs_planner_K1_K2_decisions":True,
             "figure_conventions":figures}
    save_x(out/"R0N_N1_COMPLETE_RECEIPT.json",receipt)
    print("R0N_N1_CPU_EVIDENCE_PASS N=3240 GRID=60",
          "SELECTED",chosen["name"] if chosen else "NONE",flush=True)
    print("R0N_N1_REPORT",out/"R0N_REPORT.md",flush=True)
    return 0

if __name__=="__main__":
    try:rc=main()
    except Exception as e:
        print("STOP_R0N_N1",repr(e),flush=True)
        traceback.print_exc()
        sys.exit(8)
    sys.exit(rc)
