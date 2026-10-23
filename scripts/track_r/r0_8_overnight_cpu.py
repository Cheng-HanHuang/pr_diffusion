#!/usr/bin/env python3
"""Track R overnight R0.8: independent CPU-only audit and planner return.

Uses ONLY completed R0.7 aggregate, its evaluated JSONL, and frozen R0
spec. It NEVER opens any stored gt_path or candidate_path, starts a GPU,
alters R0 outputs, or accesses protected B24 confirmation material.

Independently recomputes labels and key P1–P4 counts/metrics. All
supplementary diagnostics are explicitly EXPLORATORY, with no newly
tuned thresholds. Extracts fixed deterministic casebooks, applies
pre-registered decision cutoffs provisionally, and exports only
path-free research summaries. Final gate is PLANNER_R1_DECISION_REQUIRED.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
import zipfile

ROOT = Path("/egr/research-pac/huang248")
WT = ROOT / "pr_diffusion_track_r"
BASE = ROOT / "outputs/pr_diffusion/track_r"
R07 = BASE / "r0_7_cpu_20261023T080434Z"
SOURCE_HEAD = "c6dbd6089579c79e9e6fd4a9dd20fc402bc1185b"
FROZEN = WT / "configs/track_r/r0_frozen_v1.json"
REQUIRED_R07_STATUS = "R0_7_ANALYSIS_COMPLETE"
REPORT_FILES = (
    "R0_8_RESEARCH_AUDIT.md", "R0_8_PROVISIONAL_DECISIONS.json",
    "R0_8_P4_MATRIX.csv", "R0_8_EXPLORATORY_CASEBOOK.json",
    "R0_8_READINESS.json")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def safe_source(path):
    p=Path(path)
    if not p.is_absolute() or not p.is_file() or p.is_symlink():
        raise RuntimeError("STOP_R08_SOURCE_FILE_OR_SYMLINK "+str(p))
    forbidden=("confirmation","confirm305")
    if any(t in str(p).lower() or t in str(p.resolve()).lower()
           for t in forbidden):
        raise RuntimeError("STOP_R08_PROTECTED_SOURCE")
    return p

def finite(v):
    return isinstance(v,(float,int)) and math.isfinite(v)

def close(a,b,tol=1e-7):
    return finite(a) and finite(b) and abs(float(a)-float(b))<=tol

def assert_equal(a,b,label):
    if a!=b:raise RuntimeError("STOP_R08_"+label+" "+repr((a,b))[:180])

def q05(values):
    a=sorted(float(v) for v in values)
    if not a:return None
    x=.05*(len(a)-1)
    k=int(math.floor(x));w=x-k
    return a[k]*(1-w)+a[min(k+1,len(a)-1)]*w

def validate_upstream():
    if (not FROZEN.is_file() or FROZEN.is_symlink() or
        not R07.is_dir() or R07.is_symlink()):
        raise RuntimeError("STOP_R08_R07_OR_SPEC_MISSING")
    cfg=load(FROZEN)
    if (cfg.get("status")!="FROZEN_V1_2026-10-21" or
        cfg["labels"]["good_db"]!=25 or
        cfg["decisions"]["D_material"].find("0.20")<0 or
        cfg["bootstrap"]["replicates"]!=10000 or
        cfg["scope"]!="EXPOSED_DATA_DIAGNOSTIC_ONLY"):
        raise RuntimeError("STOP_R08_NOT_FROZEN_R0_V1")
    a=R07/"analysis"
    report=safe_source(a/"R0_7_REPORT.md")
    result=safe_source(a/"R0_7_ANALYSIS.json")
    detailed=safe_source(a/"R0_7_SCORED_GT_ROWS.jsonl")
    receipt=load(safe_source(a/"R0_7_COMPLETE_RECEIPT.json"))
    status=load(safe_source(R07/"AUTOPILOT_STATUS.json"))
    if (status.get("status")!=REQUIRED_R07_STATUS or
        status.get("n_candidates")!=3240 or
        status.get("head")!=SOURCE_HEAD or
        status.get("active_process") is not None or
        status.get("gpu_calls")!=0 or
        status.get("protected_cohort_access") is not False or
        len(status.get("stages",[]))!=2 or
        any(x.get("rc")!=0 for x in status["stages"]) or
        receipt.get("status")!=REQUIRED_R07_STATUS or
        receipt.get("n_scored")!=3240 or
        receipt.get("gpu_calls")!=0 or
        receipt.get("no_protected_access") is not True or
        receipt.get("n_tau_unavailable")!=0 or
        receipt.get("n_recorded_psnr_anomalies")!=0 or
        receipt.get("planner_decision_pending") is not True):
        raise RuntimeError("STOP_R08_R07_NOT_FULLY_COMPLETE")
    for name,path in (
        ("analysis_sha256",result),("report_sha256",report),
        ("evaluated_rows_sha256",detailed)):
        if sha(path)!=receipt[name]:
            raise RuntimeError("STOP_R08_R07_RECEIPT_SHA_"+name)
    analysis=load(result)
    if (analysis.get("head")!=SOURCE_HEAD or
        analysis.get("status")!=REQUIRED_R07_STATUS or
        analysis.get("n_candidates")!=3240 or
        analysis.get("frozen_spec_sha256")!=sha(FROZEN) or
        analysis.get("tau_unavailable_groups") or
        analysis.get("psnr_replay_anomalies") or
        analysis.get("D_decisions")!="RESERVED_FOR_PLANNER" or
        not close(analysis.get("aggregate_r0_gpu_hours"),3.4309563972961366,1e-4)):
        raise RuntimeError("STOP_R08_R07_ANALYSIS_IDENTITY")
    return cfg,analysis,report,result,detailed,receipt

def evaluated_rows(filename):
    rows=[]
    allowed_keys=("uid","source","family","image_id","run_index",
        "measurement_source","psnr_raw","psnr_repaired",
        "psnr_orbit_best","psnr_ufd","psnr_mid","s_gt","s_star",
        "s_ufd","psnr_by_orbit","raw_label","rho_signed_raw",
        "rho_signed_repaired","S_repaired","p_fake_repaired",
        "tau_signed_good_q95","after_repair_failure","S_img",
        "rho_clipped_repaired")
    # Never follow paths embedded in these offline evaluation records.
    with safe_source(filename).open(encoding="utf-8") as f:
        for line in f:
            source=json.loads(line)
            rows.append({k:source[k] for k in allowed_keys})
    if len(rows)!=3240 or len({r["uid"] for r in rows})!=3240:
        raise RuntimeError("STOP_R08_3240_UNIQUE_ROWS")
    return rows

def audit_rows(rows,analysis):
    """Independent label and orbit coherence checks, no model calls."""
    by_group=defaultdict(list)
    allbits={f"{i:03b}" for i in range(8)}
    label_counts=Counter()
    for r in rows:
        uid=r["uid"]
        if not isinstance(uid,str) or not uid:
            raise RuntimeError("STOP_R08_INVALID_UID")
        orbit=r["psnr_by_orbit"]
        if set(orbit)!=allbits or not all(finite(x) for x in orbit.values()):
            raise RuntimeError("STOP_R08_ORBIT_SCHEMA "+uid)
        argmax=max(sorted(allbits),key=lambda s:orbit[s])
        if (not close(orbit["000"],r["psnr_raw"],2e-6) or
            not close(max(orbit.values()),r["psnr_orbit_best"],2e-6) or
            not close(orbit[r["s_star"]],r["psnr_repaired"],2e-6) or
            not close(orbit[r["s_ufd"]],r["psnr_ufd"],2e-6) or
            r["s_gt"]!=argmax):
            raise RuntimeError("STOP_R08_ORBIT_PSNR_REPLAY "+uid)
        tau=r["tau_signed_good_q95"]
        if not finite(tau):
            raise RuntimeError("STOP_R08_MISSING_FROZEN_TAU "+uid)
        if r["psnr_raw"]>=25:expected="GOOD"
        elif r["psnr_orbit_best"]>=25:expected="ORBIT_ERROR"
        elif r["rho_signed_raw"]<=tau:expected="NONORBIT_CONSISTENT"
        else:expected="NONORBIT_INCONSISTENT"
        if r["raw_label"]!=expected or bool(r["after_repair_failure"])!=(r["psnr_repaired"]<25):
            raise RuntimeError("STOP_R08_LABEL_REPLAY "+uid)
        if abs(r["rho_signed_raw"]-r["rho_signed_repaired"])>1e-4:
            raise RuntimeError("STOP_R08_ORBIT_RESIDUAL "+uid)
        if not (finite(r["S_repaired"]) and finite(r["S_img"]) and
                finite(r["rho_clipped_repaired"]) and
                finite(r["rho_signed_repaired"]) and
                finite(r["p_fake_repaired"]) and
                0<=r["p_fake_repaired"]<=1):
            raise RuntimeError("STOP_R08_PRIOR_OR_DETECTOR_SCORE "+uid)
        by_group[(r["source"],r["family"])].append(r)
        label_counts[expected]+=1
    assert_equal(len(by_group)>0,True,"NO_GROUPS")
    p1={(x["source"],x["family"]):x for x in analysis["P1"]}
    p2={(x["source"],x["family"]):x for x in analysis["P2"]}
    for (src,fam),rr in by_group.items():
        label=Counter(x["raw_label"] for x in rr)
        primary=p1.get((src,fam))
        repair=p2.get((src,fam))
        if not primary or not repair:
            raise RuntimeError("STOP_R08_MISSING_P1_P2_GROUP")
        for k in ("GOOD","ORBIT_ERROR","NONORBIT_CONSISTENT",
                  "NONORBIT_INCONSISTENT"):
            assert_equal(primary[k],label[k],"P1_"+k)
        assert_equal(primary["n"],len(rr),"P1_GROUP_SIZE")
        assert_equal(primary["single_run_failures"],len(rr)-label["GOOD"],
                     "P1_DENOMINATOR")
        assert_equal(repair["n_orbit_error"],label["ORBIT_ERROR"],"P2_ORBIT_DENOM")
        assert_equal(repair["n_good"],label["GOOD"],"P2_GOOD_DENOM")
        oe=[x for x in rr if x["raw_label"]=="ORBIT_ERROR"]
        good=[x for x in rr if x["raw_label"]=="GOOD"]
        for kind,score,schoice in (
            ("prior","psnr_repaired","s_star"),
            ("univfd","psnr_ufd","s_ufd"),
            ("S_mid_secondary","psnr_mid",None)):
            v=repair["variants"][kind]
            assert_equal(v["n_success"],sum(x[score]>=25 for x in oe),
                         "P2_SUCCESS_"+kind)
            assert_equal(v["n_harmed"],sum(x[score]<25 for x in good),
                         "P2_HARM_"+kind)
            if schoice:
                assert_equal(v["n_exact"],sum(x[schoice]==x["s_gt"] for x in oe),
                             "P2_EXACT_"+kind)
    for row in analysis["P1"]:
        if row["source"]=="POOLED_FAMILY":
            subset=[x for x in rows if x["family"]==row["family"]]
            assert_equal(row["n"],len(subset),"P1_POOLED_SIZE")
    n_fast=sum(x["source"]=="S5_r0_fast" for x in rows)
    assert_equal(n_fast,720,"FAST_COUNT")
    assert_equal(len(rows)-n_fast,2520,"HISTORICAL_COUNT")
    cohorts=Counter(x["measurement_source"] for x in rows if x["source"]=="S5_r0_fast")
    assert_equal(cohorts["S1_b22"],400,"FAST_B22_COUNT")
    assert_equal(cohorts["S2_b24_dev80"],320,"FAST_DEV80_COUNT")
    return by_group,label_counts

def audit_policy_table(rows,analysis):
    by=defaultdict(list)
    for r in rows:by[(r["source"],r["family"],r["measurement_source"],r["image_id"])].append(r)
    summary=analysis["P4"]
    for table in summary:
        src,fam,k=table["source"],table["family"],table["K"]
        instances={}
        for (s,f,origin,iid),records in by.items():
            if (s,f)!=(src,fam):continue
            ordered=sorted(records,key=lambda r:(r["run_index"],r["uid"]))
            if len(ordered)<k:continue
            use=ordered[:k]
            if len({r["run_index"] for r in use})!=k:
                raise RuntimeError("STOP_R08_DUPLICATE_RUN_ORDER")
            e0=min(use,key=lambda r:(r["rho_signed_raw"],r["run_index"]))
            tau=use[0]["tau_signed_good_q95"]
            if any(not close(r["tau_signed_good_q95"],tau,1e-11) for r in use):
                raise RuntimeError("STOP_R08_TAU_DRIFT")
            feasible=[r for r in use if r["rho_signed_repaired"]<=tau]
            fallback=min(use,key=lambda r:(r["rho_signed_repaired"],r["run_index"]))
            e2=min(feasible,key=lambda r:(r["S_repaired"],r["run_index"])) if feasible else fallback
            e3=min(feasible,key=lambda r:(r["p_fake_repaired"],r["run_index"])) if feasible else fallback
            values={
                "E0":e0["psnr_raw"],"E1":e0["psnr_repaired"],
                "E2":e2["psnr_repaired"],"E3":e3["psnr_repaired"],
                "O_raw":max(r["psnr_raw"] for r in use),
                "O_orbit":max(r["psnr_orbit_best"] for r in use)}
            if (values["O_raw"]>values["O_orbit"]+1e-6 or
                any(values[key]>values["O_orbit"]+1e-6 for key in
                    ("E0","E1","E2","E3"))):
                raise RuntimeError("STOP_R08_ORACLE_CEILING")
            instances[(origin,iid)]=values
        assert_equal(len(instances),table["n_images_available"],"P4_IMAGE_COUNT")
        for name,stats in table["policy_stats"].items():
            if stats is None:
                if instances:raise RuntimeError("STOP_R08_P4_MISSING_METRICS")
                continue
            vals=[x[name] for x in instances.values()]
            if (not close(stats["mean"],statistics.fmean(vals),1e-6) or
                not close(stats["median"],statistics.median(vals),1e-6) or
                not close(stats["good25"],sum(x>=25 for x in vals)/len(vals),1e-9) or
                not close(stats["bad20"],sum(x<20 for x in vals)/len(vals),1e-9) or
                not close(stats["bad15"],sum(x<15 for x in vals)/len(vals),1e-9) or
                not close(stats["q05"],q05(vals),1e-6) or
                not close(stats["min"],min(vals),1e-6)):
                raise RuntimeError("STOP_R08_P4_METRIC_"+src+"_"+fam+"_"+str(k)+"_"+name)
    return len(summary)

def independent_auc(labels,scores):
    """Rank-based AUROC independently of R0.7's NumPy implementation."""
    if len(labels)!=len(scores):
        raise RuntimeError("STOP_R08_AUC_LENGTH")
    npos=sum(bool(x) for x in labels)
    nneg=len(labels)-npos
    if npos==0 or nneg==0:return None
    pairs=sorted((float(v),bool(y)) for y,v in zip(labels,scores))
    if any(not math.isfinite(value) for value,_ in pairs):
        raise RuntimeError("STOP_R08_AUC_NONFINITE")
    positives_rank_sum=0.
    i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[i][0]==pairs[j][0]:
            j+=1
        avg_rank=(i+1+j)/2
        positives_rank_sum+=sum(int(y) for _,y in pairs[i:j])*avg_rank
        i=j
    return (positives_rank_sum-npos*(npos+1)/2)/(npos*nneg)


def audit_reported_decision_inputs(rows,summary):
    """Independently recompute pooled denominators and key P3 AUROCs."""
    subset_orbit=[r for r in rows if r["raw_label"]=="ORBIT_ERROR"]
    subset_good=[r for r in rows if r["raw_label"]=="GOOD"]
    success=(sum(r["psnr_repaired"]>=25 for r in subset_orbit)/
             len(subset_orbit)) if subset_orbit else None
    harm=(sum(r["psnr_repaired"]<25 for r in subset_good)/
          len(subset_good)) if subset_good else None
    reference=summary["decision_inputs"]
    for key,value in (("D_repair_pooled_success",success),
                      ("D_repair_pooled_harm",harm)):
        published=reference.get(key)
        if value is None:
            if published is not None:raise RuntimeError("STOP_R08_DECISION_DENOM")
        elif not close(published,value,1e-10):
            raise RuntimeError("STOP_R08_DECISION_NUMERATORS "+key)
    p3={(p["source"],p["family"]):p for p in summary["P3"]
        if p["subtype"]=="ALL"}
    for fam in ("FAST","DAPS"):
        group=[r for r in rows if r["family"]==fam]
        entry=p3.get(("POOLED_FAMILY",fam))
        if not entry or entry["n"]!=len(group):
            raise RuntimeError("STOP_R08_P3_POOLED_SIZE "+fam)
        labels=[r["psnr_repaired"]<25 for r in group]
        assert_equal(sum(labels),entry["n_failure"],"P3_POOLED_FAILURES")
        for name,key in (
            ("S","S_repaired"),("S_img","S_img"),
            ("UnivFD_p_fake","p_fake_repaired"),
            ("rho_signed","rho_signed_repaired"),
            ("rho_clipped","rho_clipped_repaired")):
            calculated=independent_auc(labels,[r[key] for r in group])
            observed=entry["auroc_higher_means_failure"][name]
            if calculated is None:
                if observed is not None:
                    raise RuntimeError("STOP_R08_P3_EXPECT_NONE")
            elif not close(calculated,observed,1e-10):
                raise RuntimeError("STOP_R08_P3_AUC_REPLAY "+fam+"_"+name)
    return {"pooled_orbit_errors":len(subset_orbit),
            "pooled_good":len(subset_good),
            "pooled_repair_success":success,
            "pooled_repair_harm":harm,
            "pooled_family_p3_auc_groups_checked":2}


def derive_decisions(summary):
    """Mechanical frozen thresholds only; scientific verdict planner-owned."""
    p1={(r["source"],r["family"]):r for r in summary["P1"]}
    def orbit_share(fam):
        row=p1.get(("POOLED_FAMILY",fam))
        return row["orbit_share_among_failures"]["estimate"] if row else None
    ds=summary["decision_inputs"]
    fast=orbit_share("FAST");daps=orbit_share("DAPS")
    success=ds.get("D_repair_pooled_success")
    harm=ds.get("D_repair_pooled_harm")
    if any(v is not None and not finite(v) for v in (fast,daps,success,harm)):
        raise RuntimeError("STOP_R08_NONFINITE_DECISION")
    material=(fast is not None and fast>=.20) or (daps is not None and daps>=.20)
    repair_pass=(success is not None and harm is not None and
                 success>=.90 and harm<=.01)
    repair_fail=(success is not None and success<.75) or (
                 harm is not None and harm>.03)
    repair_status=("MEETS_PASS_CUTOFFS" if repair_pass else
                   "MEETS_FAIL_CUTOFFS" if repair_fail else "GREY_OR_UNAVAILABLE")
    p3={(r["source"],r["family"]):r for r in summary["P3"]
        if r["subtype"]=="ALL"}
    gains={}
    for fam in ("FAST","DAPS"):
        row=p3.get(("POOLED_FAMILY",fam))
        if row is None:
            gains[fam]=None
        else:
            a=row["auroc_higher_means_failure"]
            u=a["UnivFD_p_fake"]
            gains[fam]=None if u is None else max(
                (a[x]-u for x in ("S","S_img") if a[x] is not None),
                default=None)
    superior=all(gains[f] is not None and gains[f]>=.03
                 for f in ("FAST","DAPS"))
    noninferior=all(gains[f] is not None and gains[f]>=-.02
                    for f in ("FAST","DAPS"))
    if repair_pass and material:
        next_scenario="R1_REPAIR_AND_DETECT_CANDIDATE"
    elif repair_pass and not material and superior:
        next_scenario="R1_PRIOR_DETECTION_RESTART_CANDIDATE"
    elif repair_fail and superior:
        next_scenario="R1_DETECTION_ONLY_CANDIDATE"
    elif repair_fail:
        next_scenario="STOP_TRACK_R_CANDIDATE"
    else:
        next_scenario="PLANNER_GREY_ZONE_REVIEW"
    return {
        "status":"PROVISIONAL_NUMERIC_CUTOFFS_ONLY",
        "decisions_reserved_for":"TRACK_R_SCIENTIFIC_PLANNER",
        "material":{"FAST_orbit_share":fast,"DAPS_orbit_share":daps,
                    "either_at_least_0p20":material},
        "repair":{"success_pooled":success,"harm_pooled":harm,
                  "numerical_status":repair_status},
        "detect":{"best_prior_minus_univfd":gains,
                  "both_at_least_plus_0p03":superior,
                  "both_at_least_minus_0p02":noninferior},
        "provisional_next_scenario":next_scenario,
        "automatic_R1_authorization":False}

def concise_cases(rows,n=12):
    def minimal(x):
        fields=("uid","source","family","image_id","measurement_source",
                "run_index","raw_label","s_gt","s_star","s_ufd",
                "psnr_raw","psnr_repaired","psnr_orbit_best",
                "psnr_ufd","rho_signed_raw","S_repaired","p_fake_repaired")
        return {key:x[key] for key in fields}
    return {
        "label":"EXPLORATORY_CASE_SELECTION_NOT_A_PRE_REGISTERED_ENDPOINT",
        "largest_prior_repairs":[minimal(r) for r in sorted(
            rows,key=lambda r:(-(r["psnr_repaired"]-r["psnr_raw"]),r["uid"]))[:n]],
        "largest_prior_harm":[minimal(r) for r in sorted(
            rows,key=lambda r:(r["psnr_repaired"]-r["psnr_raw"],r["uid"]))[:n]],
        "largest_attainable_orbit_gaps":[minimal(r) for r in sorted(
            rows,key=lambda r:(-(r["psnr_orbit_best"]-r["psnr_raw"]),r["uid"]))[:n]],
        "largest_missed_oracle_repairs":[minimal(r) for r in sorted(
            rows,key=lambda r:(-(r["psnr_orbit_best"]-r["psnr_repaired"]),r["uid"]))[:n]]}

def markdown(summary,decisions,counts,by_group,n_policy,source_receipt):
    h=summary["aggregate_r0_gpu_hours"]
    lines=[
        "# Track R — R0.8 overnight scientific audit and planner handoff",
        "",
        "**Status: independent numerical replay PASS on completed R0.7 outputs.**",
        "This document contains *retrospective* evidence from exposed sources only.",
        "Every D-stage outcome below is a **provisional numeric threshold check**;",
        "the Track R planner retains the scientific decision and R1 design.",
        "",
        "## Execution and provenance",
        "",
        f"- 3,240/3,240 CPU-offline rows checked; {len(by_group)} source/family groups.",
        f"- {n_policy} P4 source/family/K tables independently replayed.",
        f"- Final R0 aggregate GPU use: **{h:.6f} / 16 GPU-hours**.",
        "- No further GPU calls, no new candidates, no new thresholds.",
        "- Original frozen R0 v1 and completed R0.7 files remain read-only.",
        "- No access to B24 protected confirmation305, including membership.",
        "",
        "## Independent label counts (single-run)",
        "",
        "| Label | Count | Fraction |","|---|---:|---:|"]
    for label in ("GOOD","ORBIT_ERROR","NONORBIT_CONSISTENT",
                  "NONORBIT_INCONSISTENT"):
        c=counts[label]
        lines.append(f"| {label} | {c} | {c/3240:.2%} |")
    lines+=["","## Frozen decision inputs — provisional","",
        "| Gate | Numeric evidence | Provisional reading |",
        "|---|---|---|"]
    material=decisions["material"];repair=decisions["repair"];detect=decisions["detect"]
    def pp(v):
        return "N/A" if v is None else f"{v:.3f}"
    lines.append("| D-material | FAST orbit share "+pp(material["FAST_orbit_share"])+
        "; DAPS "+pp(material["DAPS_orbit_share"])+
        " (threshold 0.20) | "+str(material["either_at_least_0p20"])+" |")
    lines.append("| D-repair | Success "+pp(repair["success_pooled"])+
        " (PASS ≥0.90); harm "+pp(repair["harm_pooled"])+
        " (PASS ≤0.01) | "+repair["numerical_status"]+" |")
    gains=detect["best_prior_minus_univfd"]
    lines.append("| D-detect | ΔAUROC FAST "+pp(gains["FAST"])+
        "; DAPS "+pp(gains["DAPS"])+
        " | both superior "+str(detect["both_at_least_plus_0p03"])+
        "; both noninferior "+str(detect["both_at_least_minus_0p02"])+" |")
    lines+=["","- **Provisional branch:** `"+
            decisions["provisional_next_scenario"]+"`.",
            "- **R1 is NOT authorized automatically.** Planner must adjudicate",
            "  the statistical evidence, overlap/selection issues, method novelty,",
            "  appropriate branch and locked R1 prospective protocol.",
            "",
            "## P1 — anatomy by source and family",
            "",
            "| Source | Family | Runs | Good | Orbit | Nonorbit consistent | Nonorbit inconsistent |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for (src,fam),rs in sorted(by_group.items()):
        cc=Counter(x["raw_label"] for x in rs)
        lines.append(f"| {src} | {fam} | {len(rs)} | {cc['GOOD']} | "+
                     f"{cc['ORBIT_ERROR']} | {cc['NONORBIT_CONSISTENT']} | "+
                     f"{cc['NONORBIT_INCONSISTENT']} |")
    lines+=["","## P4 — all frozen policy tables",
            "",
            "The full P4 comparisons, image-cluster bootstrap intervals and",
            "exact McNemar p-values remain in the accompanying R0.7 analysis",
            "JSON. The companion CSV contains mean, q05, Good25 and failures",
            "for all policies and K values; none of these numbers is re-tuned.",
            "",
            "## Exploratory casebook",
            "",
            "The casebook lists the largest prior repairs, largest harms,",
            "largest GT-orbit gaps and largest remaining oracle gaps",
            "by frozen candidate UID. These are **EXPLORATORY** case selections.",
            "No GT paths or candidate file paths are in the export.",
            "",
            "## Limitations and stop gate",
            "",
            "- Existing source cohorts are exposed and tuning-adjacent, with",
            "  correlated runs; source overlap is not independent replication.",
            "- The primary report uses fixed thresholds, RGB8 PSNR and",
            "  image-cluster bootstrap, not unseen-confirmation evidence.",
            "- FAST S4 and S5 are pooled for the prespecified FAST gate,",
            "  but the exposed eight-image S4 cohort overlaps known images.",
            "- The planner must evaluate margins and confidence intervals,",
            "  not treat cutoff crossings alone as a publication claim.",
            "- No access to confirmation305 before R2; no unattended",
            "  GPU experiments or R1 generation without a new planner spec.",
            "",
            "## Artifacts and continuation",
            "",
            "- `R0_8_PROVISIONAL_DECISIONS.json`: machine-readable gates.",
            "- `R0_8_P4_MATRIX.csv`: all source/family/K policy summaries.",
            "- `R0_8_EXPLORATORY_CASEBOOK.json`: path-free examples.",
            "- `R0_7_REPORT.md`, `R0_7_ANALYSIS.json`: complete frozen report.",
            "- Send `R0_8_PLANNER_RETURN.zip` back for Track R planner review.",
            "",
            "**Stop reason: PLANNER_R1_DECISION_REQUIRED.**",
            ""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    dest=Path(a.out)
    if (not str(dest).startswith(str(BASE)+"/") or dest.is_symlink() or
        dest.exists()):
        raise RuntimeError("STOP_R08_OUTPUT_NAMESPACE")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_R08_REQUIRE_CPU_ONLY")
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30)
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_R08_GIT_IDENTITY")
    # Do not create partial results unless all source validation passes.
    cfg,analysis,report,analysis_file,rows_file,receipt=validate_upstream()
    rows=evaluated_rows(rows_file)
    by_group,counts=audit_rows(rows,analysis)
    policy_count=audit_policy_table(rows,analysis)
    independent=audit_reported_decision_inputs(rows,analysis)
    decisions=derive_decisions(analysis)
    casebook=concise_cases(rows)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"logs").mkdir()
    scorecard=markdown(analysis,decisions,counts,by_group,policy_count,receipt)
    (dest/"R0_8_RESEARCH_AUDIT.md").write_text(scorecard,encoding="utf-8")
    save_x(dest/"R0_8_PROVISIONAL_DECISIONS.json",decisions)
    save_x(dest/"R0_8_EXPLORATORY_CASEBOOK.json",casebook)
    matrix=dest/"R0_8_P4_MATRIX.csv"
    with matrix.open("x",newline="",encoding="utf-8") as f:
        columns=("source","family","K","n_images","policy","n",
                 "mean","median","good25","bad20","bad15","q05","min")
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for group in analysis["P4"]:
            for policy,metrics in group["policy_stats"].items():
                prefix={"source":group["source"],"family":group["family"],
                        "K":group["K"],"n_images":group["n_images_available"],
                        "policy":policy}
                if metrics is not None:prefix.update(metrics)
                writer.writerow(prefix)
    readiness={
        "status":"R0_8_CPU_AUDIT_PASS_PLANNER_GATE",
        "head":head,"source_r07_run":str(R07),"source_r07_head":SOURCE_HEAD,
        "source_r07_analysis_sha256":receipt["analysis_sha256"],
        "source_r07_rows_sha256":receipt["evaluated_rows_sha256"],
        "frozen_config_sha256":sha(FROZEN),
        "n_rows":len(rows),"n_source_family":len(by_group),
        "n_policy_tables":policy_count,
        "independent_decision_replay":independent,
        "n_protected_files_opened":0,"new_gpu_hours":0,
        "prior_r0_gpu_hours":analysis["aggregate_r0_gpu_hours"],
        "scientific_verdict":"NOT_ISSUED_PLANNER_ONLY",
        "planner_R1_approval_required":True,
        "candidate_next_scenario":decisions["provisional_next_scenario"]}
    save_x(dest/"R0_8_READINESS.json",readiness)
    package=dest/"R0_8_PLANNER_RETURN.zip"
    with zipfile.ZipFile(package,"x",compression=zipfile.ZIP_DEFLATED,
                         compresslevel=8) as z:
        for name in REPORT_FILES:
            z.write(dest/name,arcname=name)
        z.write(report,arcname="R0_7_REPORT.md")
        z.write(analysis_file,arcname="R0_7_ANALYSIS.json")
        z.write(R07/"analysis/R0_7_COMPLETE_RECEIPT.json",
                arcname="R0_7_COMPLETE_RECEIPT.json")
    complete={
        "status":"R0_8_PLANNER_HANDOFF_READY",
        "head":head,
        "scored_candidates_audited":3240,
        "scientific_checks":"PASS",
        "planner_review_required":True,
        "R1_gpu_runs_started":0,
        "new_gpu_hours":0,
        "package_path":str(package),
        "package_sha256":sha(package),
        "r08_readiness_sha256":sha(dest/"R0_8_READINESS.json"),
        "report_sha256":sha(dest/"R0_8_RESEARCH_AUDIT.md"),
        "source_r07_report_sha256":sha(report),
        "source_r07_analysis_sha256":sha(analysis_file)}
    save_x(dest/"R0_8_COMPLETE_RECEIPT.json",complete)
    print("R08_OVERNIGHT_AUDIT_PASS",len(rows),"P4_TABLES",policy_count,flush=True)
    print("R08_PLANNER_GATE",decisions["provisional_next_scenario"],flush=True)
    print("R08_PLANNER_RETURN_ZIP",str(package),flush=True)
    print("R08_FINAL_STATUS R0_8_PLANNER_HANDOFF_READY",flush=True)
    return 0

if __name__=="__main__":
    try:
        rc=main()
    except Exception as e:
        print("STOP_R08_OVERNIGHT",repr(e),flush=True)
        traceback.print_exc()
        sys.exit(8)
    # sys.exit raises SystemExit even for success (0). Keep it *outside*
    # the exception handler so a successful audit exits with shell RC=0.
    sys.exit(rc)
