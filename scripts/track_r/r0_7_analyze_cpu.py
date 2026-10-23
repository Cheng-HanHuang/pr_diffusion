#!/usr/bin/env python3
"""R0.7 CPU-only frozen scientific analysis for 3,240 scored candidates.

Read-only GT is permitted ONLY HERE, AFTER GT-free R0.4/R0.6 scoring.
Evaluate P1–P4/secondary as locked on 2026-10-21. Never inspect B24
confirmation305 or any unscoped directory. No GPU imports/usage.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from prdiffusion.track_r.physics import to_rgb8,psnr_rgb8
from prdiffusion.track_r.r0_7_stats import (
    ORBIT,GOOD,BAD,CATASTROPHIC,BOOT_N,
    choose,orbit_psnr_channel_mse,label_candidate,tau_from_good,
    clustered_ratio_ci,clustered_paired_ci,clustered_policy_metric_ci,
    exact_mcnemar,auc_roc,
    summarize_psnr,select_e,spearman,high_frequency_energy)
from scripts.track_r.r0_4_score_gpu import candidate_tensor
from scripts.track_r.r0_2_g0_cpu import gt_source
from scripts.track_r.r0_7_prepare_cpu import (
    BASE,WT,FROZEN_PATH,frozen_spec,load,safe,filehash,
    dump_new)

def textnull(value):
    if value is None:return "N/A"
    if isinstance(value,float):return f"{value:.4f}"
    return str(value)

def rate(n,d):
    return float(n/d) if d else None

def validate_prior_score(rec,scoring):
    scores=scoring["prior"]["S"]
    if (set(scores)!=set(ORBIT) or
        set(scoring["p_fake_orbit"])!=set(ORBIT) or
        scoring.get("uid")!=rec["uid"] or
        scoring.get("candidate_sha256")!=rec["candidate_sha256"] or
        scoring.get("measurement_sha256")!=rec["measurement_sha256"] or
        scoring.get("gt_access") is not False or
        scoring.get("unet_forwards")!=256 or
        choose(scores)!=scoring["s_star"] or
        not all(math.isfinite(float(v)) for v in scores.values())):
        raise RuntimeError("STOP_R07_PRIOR_IDENTITY "+rec["uid"])
    prior=scoring["prior"]
    if (set(prior["S_mid"])!=set(ORBIT) or
        set(prior["per_t_means"])!=set(ORBIT) or
        any(set(prior["per_t_means"][s])!={"50","100","200","300","400","500","600","700"}
            for s in ORBIT) or
        not math.isfinite(float(scoring["p_fake_repaired"])) or
        abs(float(scoring["p_fake_orbit"][scoring["s_star"]])-
            float(scoring["p_fake_repaired"]))>1e-5 or
        abs(float(scoring["S_repaired"])-float(scores[scoring["s_star"]]))>1e-6 or
        abs(float(scoring["S_raw"])-float(scores["000"]))>1e-6):
        raise RuntimeError("STOP_R07_PRIOR_SCORE_FIELDS "+rec["uid"])
    return scores,prior

def add_gt_rows(catalog,progress_every=250):
    """Canonical x8/GT8 per source, 8-channel orbit PSNR via 2^3 choices."""
    gtcache={}
    result=[]
    anomaly=[]
    timings=defaultdict(list)
    start=time.monotonic()
    for i,row in enumerate(catalog):
        uid=row["uid"]
        key=(row["gt_path"],row["gt_format"])
        if key not in gtcache:
            gt=gt_source(row)
            gtcache[key]=to_rgb8(gt.detach().float())
        g8=gtcache[key]
        x8=candidate_tensor(row)   # verifies candidate SHA + canonical RGB8
        score_path=safe(row["score_receipt_path"])
        if filehash(score_path)!=row["score_receipt_sha256"]:
            raise RuntimeError("STOP_R07_SCORE_RECEIPT_SHA "+uid)
        scoring=load(score_path)
        prior_scores,prior=validate_prior_score(row,scoring)
        orbit=orbit_psnr_channel_mse(x8,g8)
        raw=orbit["000"]
        # Verify equivalence of fast per-channel identity PSNR with pinned
        # RGB8 implementation, before interpreting labels.
        official=float(psnr_rgb8(x8,g8)[0])
        if abs(raw-official)>0.0001:
            raise RuntimeError("STOP_R07_CANONICAL_PSNR_WITNESS "+uid)
        best=max(ORBIT,key=lambda p:orbit[p])
        psnr_best=orbit[best]
        s_prior=scoring["s_star"]
        s_ufd=min(ORBIT,key=lambda p:float(scoring["p_fake_orbit"][p]))
        s_mid=choose(prior["S_mid"])
        sep=abs(float(scoring["rho_signed_raw"])-
                float(scoring["rho_signed_repaired"]))
        if sep>1e-4:
            raise RuntimeError("STOP_R07_SIGNED_ORBIT_INVARIANCE "+uid)
        clipped=abs(float(scoring["rho_clipped_raw"])-
                    float(scoring["rho_clipped_repaired"]))
        if clipped>1e-4:
            raise RuntimeError("STOP_R07_CLIPPED_ORBIT_INVARIANCE "+uid)
        raw_db=row.get("recorded_psnr_db")
        if (raw_db is not None and
            not str(row.get("recorded_psnr_convention")).endswith("FLOAT_CLAMP") and
            abs(float(raw_db)-raw)>0.01):
            anomaly.append({"uid":uid,"recorded_db":raw_db,
                            "computed_raw_db":raw,
                            "source_recording":row["recorded_psnr_convention"]})
        r={
            "uid":uid,"source":row["source"],"family":row["family"],
            "image_id":row["image_id"],"run_index":row["run_index"],
            "measurement_source":row["measurement_source"],
            "psnr_raw":raw,"psnr_repaired":orbit[s_prior],
            "psnr_ufd":orbit[s_ufd],"psnr_mid":orbit[s_mid],
            "psnr_orbit_best":psnr_best,"s_gt":best,
            "s_star":s_prior,"s_ufd":s_ufd,"s_mid":s_mid,
            "psnr_by_orbit":orbit,"S_repaired":float(scoring["S_repaired"]),
            "S_raw":float(scoring["S_raw"]),
            "S_mid_repaired":float(prior["S_mid"][s_mid]),
            "p_fake_repaired":float(scoring["p_fake_orbit"][s_prior]),
            "p_fake_orbit":{k:float(v) for k,v in scoring["p_fake_orbit"].items()},
            "rho_signed_raw":float(scoring["rho_signed_raw"]),
            "rho_signed_repaired":float(scoring["rho_signed_repaired"]),
            "rho_clipped_raw":float(scoring["rho_clipped_raw"]),
            "rho_clipped_repaired":float(scoring["rho_clipped_repaired"]),
            "S_per_t_repaired":{
                t:float(prior["per_t_means"][s_prior][t])
                for t in ("50","100","200","300","400","500","600","700")},
            "per_t_s_choices":{
                t:choose({s:prior["per_t_means"][s][t] for s in ORBIT})
                for t in ("50","100","200","300","400","500","600","700")},
            "per_t_scores":prior["per_t_means"],
            "native_selector_name":row.get("native_selector_name"),
            "native_selector_value":row.get("native_selector_value"),
            "recorded_cost_seconds":row.get("recorded_cost_seconds"),
            "recorded_unet_forwards":row.get("recorded_unet_forwards"),
            "score_gpu_seconds":float(scoring["gpu_score_wall_seconds"]),
            "score_unet_forwards":int(scoring["unet_forwards"]),
            "score_univfd_forwards":int(scoring["univfd_forwards"]),
            "prior_margin":float(prior["margin"]),
            "prior_z":float(prior["z"]),
            "candidate_path":row["candidate_path"],
            "gt_path":row["gt_path"]}
        if raw>=GOOD:
            # Complexity bias pre-registered secondary. On GOOD raw images,
            # global/channelwise 180-degree rotations have same Fourier
            # power, so raw and repaired share HF energy.
            r["hf_energy_fraction"]=high_frequency_energy(x8)
        else:
            r["hf_energy_fraction"]=None
        if row["source"]=="S5_r0_fast":
            receipt=load(Path(row["candidate_path"]).parent/"R0_FAST_ROOT_COMPLETE.json")
            if receipt["uid"]!=uid or receipt.get("gt_access") is not False:
                raise RuntimeError("STOP_R07_NATIVE_FAST_TIMING_SOURCE "+uid)
            r["native_sample_wall_seconds"]=float(receipt["native_sample_wall_seconds"])
            r["native_cuda_seconds"]=float(receipt["sample_cuda_event_seconds"])
        else:
            r["native_sample_wall_seconds"]=None
            r["native_cuda_seconds"]=None
        result.append(r)
        if (i+1)%progress_every==0:
            print("R07_GT_PROGRESS",i+1,"/",len(catalog),
                  "CPU_WALL_S",round(time.monotonic()-start,1),flush=True)
    return result,anomaly

def complete_labels(rows):
    taus=tau_from_good(rows)
    for r in rows:
        tau=taus[(r["source"],r["family"])]
        r["tau_signed_good_q95"]=tau
        r["raw_label"]=label_candidate(r,tau)
        r["band"]=("CATASTROPHIC" if r["psnr_raw"]<15 else
                    "BAD" if r["psnr_raw"]<20 else
                    "MARGINAL" if r["psnr_raw"]<25 else "GOOD")
        r["orbit_subtype"]=(
            "GLOBAL" if r["s_gt"]=="111" else "PARTIAL"
        ) if r["raw_label"]=="ORBIT_ERROR" else None
        r["after_repair_failure"]=r["psnr_repaired"]<GOOD
    return taus

def measurement_instance_key(row):
    # S5 pools S1 and S2 measurement cohorts under one method source.
    # For within-image policies the measurement must remain fixed.
    return (row["measurement_source"],str(row["image_id"]))

def groups_source_family(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["source"],r["family"])].append(r)
    return groups

def outcome_p1(rows):
    groups=groups_source_family(rows)
    families=sorted({r["family"] for r in rows})
    for family in families:
        groups[("POOLED_FAMILY",family)]=[r for r in rows if r["family"]==family]
    entries=[]
    for (src,fam),subset in sorted(groups.items()):
        counts=Counter(r["raw_label"] for r in subset)
        errors=[r for r in subset if r["psnr_raw"]<GOOD]
        ci=clustered_ratio_ci(subset,
            lambda r:r["raw_label"]=="ORBIT_ERROR",
            lambda r:r["psnr_raw"]<GOOD,
            "P1|"+src+"|"+fam)
        entries.append({
            "source":src,"family":fam,"n":len(subset),
            "n_images":len({r["image_id"] for r in subset}),
            "GOOD":counts["GOOD"],
            "ORBIT_ERROR":counts["ORBIT_ERROR"],
            "GLOBAL":sum(r["orbit_subtype"]=="GLOBAL" for r in subset),
            "PARTIAL":sum(r["orbit_subtype"]=="PARTIAL" for r in subset),
            "NONORBIT_CONSISTENT":counts["NONORBIT_CONSISTENT"],
            "NONORBIT_INCONSISTENT":counts["NONORBIT_INCONSISTENT"],
            "NONORBIT_UNKNOWN_TAU":counts["NONORBIT_UNKNOWN_TAU"],
            "single_run_failures":len(errors),
            "orbit_share_among_failures":ci})
    return entries

def outcome_p2(rows):
    groups=groups_source_family(rows)
    for family in sorted({r["family"] for r in rows}):
        groups[("POOLED_FAMILY",family)]=[r for r in rows if r["family"]==family]
    entries=[]
    variants={"prior":("psnr_repaired","s_star"),
              "univfd":("psnr_ufd","s_ufd"),
              "S_mid_secondary":("psnr_mid","s_mid")}
    for (src,fam),subset in sorted(groups.items()):
        report={"source":src,"family":fam,"n":len(subset),
                "n_images":len({r["image_id"] for r in subset}),
                "n_orbit_error":sum(r["raw_label"]=="ORBIT_ERROR" for r in subset),
                "n_good":sum(r["raw_label"]=="GOOD" for r in subset),
                "variants":{}}
        for method,(psnrkey,skey) in variants.items():
            ep=[r for r in subset if r["raw_label"]=="ORBIT_ERROR"]
            good=[r for r in subset if r["raw_label"]=="GOOD"]
            success=clustered_ratio_ci(subset,
                lambda r:r["raw_label"]=="ORBIT_ERROR" and r[psnrkey]>=GOOD,
                lambda r:r["raw_label"]=="ORBIT_ERROR",
                "P2|success|"+method+"|"+src+"|"+fam)
            exact=clustered_ratio_ci(subset,
                lambda r:r["raw_label"]=="ORBIT_ERROR" and r[skey]==r["s_gt"],
                lambda r:r["raw_label"]=="ORBIT_ERROR",
                "P2|exact|"+method+"|"+src+"|"+fam)
            harm=clustered_ratio_ci(subset,
                lambda r:r["raw_label"]=="GOOD" and r[psnrkey]<GOOD,
                lambda r:r["raw_label"]=="GOOD",
                "P2|harm|"+method+"|"+src+"|"+fam)
            report["variants"][method]={
                "success_on_orbit_errors":success,
                "exact_s_gt_on_orbit_errors":exact,
                "harm_on_good":harm,
                "n_success":sum(r[psnrkey]>=GOOD for r in ep),
                "n_exact":sum(r[skey]==r["s_gt"] for r in ep),
                "n_harmed":sum(r[psnrkey]<GOOD for r in good)}
        entries.append(report)
    return entries

def outcome_p3(rows):
    medians=defaultdict(list)
    for r in rows:
        medians[(r["source"],r["family"],*measurement_instance_key(r))].append(r["S_repaired"])
    normalized={k:float(np.median(v)) for k,v in medians.items()}
    for r in rows:
        r["S_img"]=r["S_repaired"]-normalized[
            (r["source"],r["family"],*measurement_instance_key(r))]
    metrics={
        "S":"S_repaired","S_img":"S_img",
        "UnivFD_p_fake":"p_fake_repaired",
        "rho_signed":"rho_signed_repaired",
        "rho_clipped":"rho_clipped_repaired"}
    groups=groups_source_family(rows)
    for fam in sorted({r["family"] for r in rows}):
        groups[("POOLED_FAMILY",fam)]=[r for r in rows if r["family"]==fam]
    reports=[]
    for (src,fam),base in sorted(groups.items()):
        for subtype in ("ALL","NONORBIT_CONSISTENT","NONORBIT_INCONSISTENT"):
            if subtype=="ALL":
                subset=base
            else:
                # For subtype AUROC, compare only raw-subtype failures
                # which remain failures after repair against all
                # repaired successes. Repaired successes are the controls.
                subset=[r for r in base if not r["after_repair_failure"] or
                        (r["after_repair_failure"] and r["raw_label"]==subtype)]
            y=[r["after_repair_failure"] for r in subset]
            auc={k:auc_roc(y,[r[v] for r in subset]) for k,v in metrics.items()}
            reports.append({
                "source":src,"family":fam,"subtype":subtype,
                "n":len(subset),"n_failure":sum(y),
                "n_success":len(subset)-sum(y),
                "auroc_higher_means_failure":auc,
                "S_minus_UnivFD":(
                    auc["S"]-auc["UnivFD_p_fake"] if
                    auc["S"] is not None and auc["UnivFD_p_fake"] is not None else None),
                "S_img_minus_UnivFD":(
                    auc["S_img"]-auc["UnivFD_p_fake"] if
                    auc["S_img"] is not None and auc["UnivFD_p_fake"] is not None else None)})
    return reports

def outcome_p4(rows,taus):
    groups=groups_source_family(rows)
    report=[]
    for (src,fam),subset in sorted(groups.items()):
        groups_by_image=defaultdict(list)
        for r in subset:
            groups_by_image[measurement_instance_key(r)].append(r)
        # Named run order is authoritative. If source has fewer than
        # K completed roots, that K is explicitly unavailable.
        supported=[1,2,4,8] if src=="S1_b22" and fam.startswith("NP") else [1,2,4]
        for k in supported:
            byimg={}
            empty=0
            missing_tau=0
            for image,rr in groups_by_image.items():
                if len(rr)<k:continue
                value=select_e(rr,k,taus[(src,fam)])
                if value is None:continue
                byimg[image]=value
                if value["empty_feasible"]:empty+=1
                if value["empty_feasible"] is None:missing_tau+=1
            values={policy:{image:dat["psnr"][policy]
                    for image,dat in byimg.items()
                    if dat["psnr"][policy] is not None}
                    for policy in ("E0","E1","E2","E3","O_raw","O_orbit")}
            stats={key:summarize_psnr(list(vals.values())) for key,vals in values.items()}
            compare={}
            for lhs,rhs in (("E1","E0"),("E2","E1"),("E2","E3")):
                pa=values[lhs];pb=values[rhs]
                if pa and set(pa)==set(pb):
                    idbase=f"P4|{src}|{fam}|K{k}|{lhs}-{rhs}"
                    compare[lhs+"_minus_"+rhs]={
                        "paired_bootstrap_all_reported_metrics":
                            clustered_policy_metric_ci(pa,pb,idbase),
                        "mcnemar_good25_exact":exact_mcnemar(pa,pb)}
                else:
                    compare[lhs+"_minus_"+rhs]=None
            report.append({
                "source":src,"family":fam,"K":k,
                "n_images_available":len(byimg),
                "n_images_no_tau":missing_tau,
                "n_images_total":len(groups_by_image),
                "n_empty_feasible":empty,
                "policy_stats":stats,
                "paired_comparisons":compare})
    return report

def outcomes_secondary(rows):
    families=groups_source_family(rows)
    for family in sorted({r["family"] for r in rows}):
        families[("POOLED_FAMILY",family)]=[r for r in rows if r["family"]==family]
    report=[]
    for (src,fam),subset in sorted(families.items()):
        good=[r for r in subset if r["raw_label"]=="GOOD"]
        hf=[r["hf_energy_fraction"] for r in good]
        s=[r["S_repaired"] for r in good]
        bias=spearman(s,hf)
        image_groups=defaultdict(list)
        for r in good:
            image_groups[measurement_instance_key(r)].append(r)
        lowest=0
        tested=0
        for entries in image_groups.values():
            if len(entries)<2:continue
            by_score=min(entries,key=lambda r:(r["S_repaired"],r["uid"]))
            by_hf=min(entries,key=lambda r:(r["hf_energy_fraction"],r["uid"]))
            lowest+=int(by_score["uid"]==by_hf["uid"])
            tested+=1
        per_t=[]
        for t in ("50","100","200","300","400","500","600","700"):
            repair_orbit=[r for r in subset if r["raw_label"]=="ORBIT_ERROR"]
            goodall=[r for r in subset if r["raw_label"]=="GOOD"]
            positives=[r["after_repair_failure"] for r in subset]
            u=[r["S_per_t_repaired"][t] for r in subset]
            per_t.append({
                "t":int(t),"AUROC_after_primary_repair":auc_roc(positives,u),
                "repair_success_on_orbit_errors":rate(
                    sum(r["psnr_by_orbit"][r["per_t_s_choices"][t]]>=GOOD
                        for r in repair_orbit),len(repair_orbit)),
                "harm_on_good":rate(
                    sum(r["psnr_by_orbit"][r["per_t_s_choices"][t]]<GOOD
                        for r in goodall),len(goodall))})
        sample_times=[r["native_sample_wall_seconds"] for r in subset
                      if r["native_sample_wall_seconds"] is not None]
        cuda_times=[r["native_cuda_seconds"] for r in subset
                    if r["native_cuda_seconds"] is not None]
        report.append({
            "source":src,"family":fam,"n":len(subset),
            "n_good":len(good),
            "spearman_S_vs_high_frequency_energy_good":bias,
            "prior_picks_lowest_hf_within_image":{
                "n_images":tested,"n_matches":lowest,"fraction":rate(lowest,tested)},
            "per_timestep":per_t,
            "mean_score_wall_seconds":float(np.mean([r["score_gpu_seconds"]
                for r in subset])) if subset else None,
            "mean_unet_forwards":float(np.mean([r["score_unet_forwards"]
                for r in subset])) if subset else None,
            "mean_univfd_image_forwards":float(np.mean([r["score_univfd_forwards"]
                for r in subset])) if subset else None,
            "FAST_native_sample_wall_mean_s":float(np.mean(sample_times))
                 if sample_times else None,
            "FAST_native_cuda_event_mean_s":float(np.mean(cuda_times))
                 if cuda_times else None})
    return report

def generate_report(data):
    output=["# Track R R0.7 — Frozen retrospective evaluation",
        "",
        "**Scope:** exposed S1–S5 / B24 DEV80 only. Not an independent validation.",
        "**Change control:** frozen R0 v1; no post-result thresholds introduced.",
        "**Status:** "+data["status"], "",
        "## Provenance and completeness", "",
        f"- Candidates: {data['n_candidates']}; unique images: {data['n_unique_image_ids']}",
        f"- Final aggregate R0 GPU-hours: {data['aggregate_r0_gpu_hours']:.5f} / 16",
        f"- Frozen config SHA-256: `{data['frozen_spec_sha256']}`",
        f"- Offline source-catalog SHA-256: `{data['catalog_sha256']}`",
        "- No protected B24 confirmation305 access.",
        "- GT used only by CPU offline evaluation after all frozen scorer receipts.",
        "", "## P1 — single-run orbit failure anatomy", "",
        "| Source | Family | n | Good | Orbit / fails | Global | Partial | Consistent | Inconsistent | 95% image-cluster CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in data["P1"]:
        v=r["orbit_share_among_failures"]
        if v["estimate"] is not None:
            share=f"{v['estimate']:.1%}"
            ci=f"[{v['ci95'][0]:.1%}, {v['ci95'][1]:.1%}]" if v["ci95"] else "N/A"
        else:share=ci="N/A"
        output.append(f"| {r['source']} | {r['family']} | {r['n']} | {r['GOOD']} | {r['ORBIT_ERROR']} / {r['single_run_failures']} ({share}) | {r['GLOBAL']} | {r['PARTIAL']} | {r['NONORBIT_CONSISTENT']} | {r['NONORBIT_INCONSISTENT']} | {ci} |")
    output.extend(["","## P2 — repair efficacy and harm","",
        "| Source | Family | Method | Orbit error n | Success | Exact s_gt | Good n | Harm |",
        "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in data["P2"]:
        for method,v in r["variants"].items():
            output.append(f"| {r['source']} | {r['family']} | {method} | {r['n_orbit_error']} | {textnull(v['success_on_orbit_errors']['estimate'])} | {textnull(v['exact_s_gt_on_orbit_errors']['estimate'])} | {r['n_good']} | {textnull(v['harm_on_good']['estimate'])} |")
    output.extend(["","## P3 — post-repair failure detection AUROC", "",
        "Higher scores predict failure. Missing one-class comparisons are N/A.",
        "The subtype rows compare remaining failures of that *raw* non-orbit",
        "subtype against all repaired successes (same family/source).","",
        "| Source | Family | Failure subset | n | Failed | S | S_img | UnivFD | Signed | Clipped | S−UnivFD |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in data["P3"]:
        a=r["auroc_higher_means_failure"]
        output.append(f"| {r['source']} | {r['family']} | {r['subtype']} | {r['n']} | {r['n_failure']} | {textnull(a['S'])} | {textnull(a['S_img'])} | {textnull(a['UnivFD_p_fake'])} | {textnull(a['rho_signed'])} | {textnull(a['rho_clipped'])} | {textnull(r['S_minus_UnivFD'])} |")
    output.extend(["","## P4 — frozen K-run policies","",
        "Each tuple is (mean PSNR dB, Good25, Bad20, q05 dB, min dB).",
        "The full JSON includes median, Bad15, all paired 95% bootstrap CIs, and exact McNemar tests.","",
        "| Source | Family | K | Images | E0 | E1 | E2 | E3 | O_raw | O_orbit |",
        "|---|---|---:|---:|---|---|---|---|---|---|"])
    for r in data["P4"]:
        def shape(d):
            if d is None:return "N/A"
            return f"({d['mean']:.2f}, {d['good25']:.1%}, {d['bad20']:.1%}, {d['q05']:.2f}, {d['min']:.2f})"
        v=r["policy_stats"]
        output.append(f"| {r['source']} | {r['family']} | {r['K']} | {r['n_images_available']} | {shape(v['E0'])} | {shape(v['E1'])} | {shape(v['E2'])} | {shape(v['E3'])} | {shape(v['O_raw'])} | {shape(v['O_orbit'])} |")
    output.extend(["","## Registered decision inputs (planner adjudicates)","",
        "These are **measurements**, not an executor-selected D-material/D-repair/D-detect decision.",""])
    for key,value in data["decision_inputs"].items():
        output.append(f"- **{key}:** {textnull(value)}")
    output.extend(["",
        "## Secondary diagnostics and limitations","",
        f"- Number of source/family groups missing a GOOD-based tau: {len(data['tau_unavailable_groups'])}.",
        f"- RGB8/replayed original PSNR discrepancies >0.01 dB: {len(data['psnr_replay_anomalies'])}.",
        "- Secondary per-timestep AUROC/repair, HF-energy complexity bias,",
        "  native FAST timing, and score cost are in R0_7_ANALYSIS.json.",
        "- Quantiles use NumPy's linear interpolation; analysis weights each",
        "  candidate equally unless policy evaluation is explicitly per image.",
        "- Image bootstrap samples physical image IDs, keeping records from",
        "  overlapping source cohorts in the same cluster.",
        "- Native selector values are carried only when originally recorded.",
        "- Subtype AUROC convention is stated above because the frozen document",
        "  did not uniquely specify its positive/negative subset construction.",
        "- This analysis is retrospective and tuning-adjacent. No claim about",
        "  generalization to confirmation305 or unseen data.",""])
    return "\n".join(output)

def summarize_decision_inputs(p1,p2,p3):
    out={}
    for fam,aliases in (("DAPS",("DAPS",)),("FAST",("FAST",))):
        selected=[r for r in p1 if r["source"]=="POOLED_FAMILY" and r["family"] in aliases]
        if selected:
            out[f"D_material_{fam}_orbit_share"]=selected[0][
                "orbit_share_among_failures"]["estimate"]
        d=[r for r in p3 if r["source"]=="POOLED_FAMILY" and
           r["family"]==fam and r["subtype"]=="ALL"]
        if d:
            out[f"D_detect_{fam}_S_minus_UnivFD"]=d[0]["S_minus_UnivFD"]
            out[f"D_detect_{fam}_S_img_minus_UnivFD"]=d[0]["S_img_minus_UnivFD"]
    orbit=sum(r["n_orbit_error"] for r in p2 if r["source"]!="POOLED_FAMILY")
    good=sum(r["n_good"] for r in p2 if r["source"]!="POOLED_FAMILY")
    succ=sum(r["variants"]["prior"]["n_success"] for r in p2 if r["source"]!="POOLED_FAMILY")
    harm=sum(r["variants"]["prior"]["n_harmed"] for r in p2 if r["source"]!="POOLED_FAMILY")
    out["D_repair_pooled_success"]=rate(succ,orbit)
    out["D_repair_pooled_harm"]=rate(harm,good)
    out["D_repair_pooled_n_orbit"]=orbit
    out["D_repair_pooled_n_good"]=good
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--catalog",required=True)
    parser.add_argument("--catalog-sha",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--expected-head",required=True)
    a=parser.parse_args()
    out=Path(a.out)
    if not str(out).startswith(str(BASE)+"/") or out.exists() or out.is_symlink():
        raise RuntimeError("STOP_R07_OUTPUT_MUST_BE_FRESH")
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30)
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_R07_ANALYSIS_HEAD_OR_DIRTY")
    frozen=frozen_spec()
    if filehash(a.catalog)!=a.catalog_sha:
        raise RuntimeError("STOP_R07_CATALOG_SHA")
    catalog=load(a.catalog)
    if (catalog.get("status")!="R0_7_GT_OFFLINE_CATALOG_PASS" or
        catalog.get("n_candidates")!=3240 or
        catalog.get("n_historical")!=2520 or
        catalog.get("n_fast")!=720 or
        catalog.get("frozen_spec_sha256")!=filehash(FROZEN_PATH) or
        catalog.get("scope")!="EXPOSED_DATA_DIAGNOSTIC_ONLY" or
        catalog.get("no_protected_access") is not True or
        len(catalog.get("rows",[]))!=3240):
        raise RuntimeError("STOP_R07_OFFLINE_CATALOG_INVALID")
    rows,anomalies=add_gt_rows(catalog["rows"])
    if len(rows)!=3240 or len({r["uid"] for r in rows})!=3240:
        raise RuntimeError("STOP_R07_DUPLICATED_OR_MISSING_SCORE")
    tau=complete_labels(rows)
    p1=outcome_p1(rows)
    p2=outcome_p2(rows)
    p3=outcome_p3(rows)
    p4=outcome_p4(rows,tau)
    secondary=outcomes_secondary(rows)
    tau_missing=[{"source":k[0],"family":k[1]} for k,v in sorted(tau.items()) if v is None]
    # A missing GOOD threshold cannot be repaired by fitting on failures.
    status="R0_7_ANALYSIS_COMPLETE" if not tau_missing else "R0_7_PARTIAL_TAU_UNAVAILABLE"
    report={
        "schema":"prdiff.track_r.r0.7.analysis.v1",
        "status":status,"head":head,"scope":"EXPOSED_DATA_DIAGNOSTIC_ONLY",
        "frozen_spec_sha256":filehash(FROZEN_PATH),
        "catalog_sha256":a.catalog_sha,
        "n_candidates":len(rows),"n_unique_image_ids":len({r["image_id"] for r in rows}),
        "aggregate_r0_gpu_hours":float(catalog["all_r0_gpu_seconds"])/3600,
        "bootstrap_replicates":BOOT_N,"bootstrap_unit":"image",
        "tau_q95_over_raw_good":[{"source":k[0],"family":k[1],"tau":v}
            for k,v in sorted(tau.items())],
        "tau_unavailable_groups":tau_missing,
        "psnr_replay_anomalies":anomalies,
        "P1":p1,"P2":p2,"P3":p3,"P4":p4,
        "secondary":secondary,
        "decision_inputs":summarize_decision_inputs(p1,p2,p3),
        "D_decisions":"RESERVED_FOR_PLANNER",
        "no_gpu":True,"protected_cohort_access":False,
        "limitations":{
            "subtype_auroc_controls":"all repaired successes; only matching original raw failure subtype as positives",
            "nonorbit_tau_quantile_interpolation":"numpy_linear",
            "image_bootstrap_cluster_key":"image_id_across_sources",
            "univfd_repaired_score_convention":"p_fake_orbit[s_star]",
            "hf_energy_convention":"RGB8 [0,1], FFT radius 0.25 cycles/pixel, energy includes DC",
            "no_prospective_claims":True}}
    out.mkdir(parents=True,exist_ok=False)
    summary=out/"R0_7_ANALYSIS.json"
    dump_new(summary,report)
    with (out/"R0_7_SCORED_GT_ROWS.jsonl").open("x",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    with (out/"R0_7_REPORT.md").open("x",encoding="utf-8") as f:
        f.write(generate_report(report))
    receipt={
        "status":status,"head":head,"n_scored":3240,
        "analysis_sha256":filehash(summary),
        "report_sha256":filehash(out/"R0_7_REPORT.md"),
        "evaluated_rows_sha256":filehash(out/"R0_7_SCORED_GT_ROWS.jsonl"),
        "n_tau_unavailable":len(tau_missing),
        "n_recorded_psnr_anomalies":len(anomalies),
        "gpu_calls":0,"no_protected_access":True,
        "all_r0_gpu_hours":report["aggregate_r0_gpu_hours"],
        "planner_decision_pending":True}
    dump_new(out/"R0_7_COMPLETE_RECEIPT.json",receipt)
    print("R07_ANALYSIS_STATUS",status,"3240/3240",
          "TAU_UNAVAILABLE",len(tau_missing),
          "RECORDED_PSNR_ANOMALIES",len(anomalies),
          "REPORT",out/"R0_7_REPORT.md",flush=True)
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print("STOP_R07_ANALYSIS",repr(e),flush=True)
        traceback.print_exc()
        sys.exit(8)
