#!/usr/bin/env python3
"""Read-only per-image residual rank confidence on 32 saved FAST-DIPS roots.

This is NOT a new GT-free validation cohort; no ground truth tensors, samples,
CUDA inference or original PAC source trees are modified.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path("/egr/research-pac/huang248")
M0=ROOT/"outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020/a15_6_mechanism_cpu_audit_v1"
A16=ROOT/"outputs/pr_diffusion/track_a_np_paper/a16_np_fast_autopilot_20261021"
M0_SHA="b7c2e478982c571bf45cacd3eddc505e6752501047aa1467ead75064115bcb74"
IMAGES=("69073","68539","67948","60140","67293","67174","61101","66631")


def rank(v):
    order=sorted(range(len(v)),key=lambda j:v[j])
    r=[0.]*len(v)
    for i,j in enumerate(order):r[j]=i+1
    return r


def make():
    receipt=M0/"A15_MECHANISM_AUDIT_COMPLETE.json"
    h=hashlib.sha256(receipt.read_bytes()).hexdigest()
    if h!=M0_SHA:raise RuntimeError("STOP M0 CPU receipt SHA256 changed")
    data=json.loads(receipt.read_text())
    if data.get("status")!="A15_CPU_MECHANISM_AUDIT_PASS":
        raise RuntimeError("STOP M0 receipt not PASS")
    with (M0/"A15_PER_CANDIDATE_MECHANISM.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    grouped={i:[] for i in IMAGES}
    for row in rows:
        grouped[row["image_id"]].append(dict(rep=int(row["rep"]),
          psnr=float(row["raw_rgb8_psnr_db_from_pinned_receipt"]),
          residual=float(row["actual_output_measurement_l2"])))
    summaries=[]
    for ident in IMAGES:
        v=sorted(grouped[ident],key=lambda x:x["rep"])
        if len(v)!=4 or [q["rep"] for q in v]!=[0,1,2,3]:
            raise RuntimeError("STOP candidate count per image")
        residual_order=sorted(v,key=lambda r:r["residual"])
        psnr_order=sorted(v,key=lambda r:-r["psnr"])
        p_rank=rank([-q["psnr"] for q in v])
        r_rank=rank([q["residual"] for q in v])
        rho=1-6*sum((p_rank[k]-r_rank[k])**2 for k in range(4))/(4*(4*4-1))
        summaries.append(dict(image_id=ident,
          residual_winner_rep=residual_order[0]["rep"],
          oracle_rep=psnr_order[0]["rep"],
          residual_margin_l2=residual_order[1]["residual"]-residual_order[0]["residual"],
          residual_margin_over_absolute_residual=(residual_order[1]["residual"]-residual_order[0]["residual"])/max(residual_order[0]["residual"],1e-12),
          gt_oracle_regret_db=max(0,psnr_order[0]["psnr"]-residual_order[0]["psnr"]),
          spearman_neg_residual_vs_psnr=rho,
          within_image_psnr_span_db=max(q["psnr"] for q in v)-min(q["psnr"] for q in v)))
    summary=dict(status="A16_SAVED_FAST_SELECTOR_FORENSICS_PASS",
         input_m0_sha256=M0_SHA,cohort="EXPOSED_EIGHT_NOT_PROSPECTIVE",
         all_candidates_preexisting=True,real_gpu_runs=0,
         per_image=summaries,
         mean_gt_oracle_regret_db=statistics.mean(x["gt_oracle_regret_db"] for x in summaries),
         matched_oracle=sum(x["residual_winner_rep"]==x["oracle_rep"] for x in summaries),
         mean_spearman=statistics.mean(x["spearman_neg_residual_vs_psnr"] for x in summaries),
         note="Post-hoc scores on existing FAST candidates; not a learned/deployable cross-method selector")
    out=A16/"A16_SAVED_FAST_SELECTOR_FORENSICS.json"
    with out.open("x",encoding="utf8") as f:
        json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print(json.dumps(dict(status=summary["status"],matches=summary["matched_oracle"],
                         mean_spearman=summary["mean_spearman"],
                         mean_regret_db=summary["mean_gt_oracle_regret_db"]),sort_keys=True))
    return summary


if __name__=="__main__":make()
