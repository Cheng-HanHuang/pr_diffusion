#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

NEW_ARMS = ["NP_EPP_321", "NP_EPP_321_RANDOM_PRUNE", "NP_EPP_321_NO_REALLOCATION"]
OLD_ARMS = ["NP4_INDEPENDENT", "NP_EPP"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def arm_metrics(result):
    return {
        "selected": float(result["clean_free_selected_psnr_raw_db"]),
        "oracle": float(result["oracle_best_psnr_raw_db"]),
        "gap": float(result["selector_gap_psnr_db"]),
        "selected_good25": bool(result["clean_free_selected_good25"]),
        "oracle_good25": bool(result["oracle_best_good25"]),
        "total_unet_evals": int(result["total_unet_evals"]),
        "proposal_unet_evals": int(result["proposal_unet_evals"]),
    }


def summarize_delta(rows, arm, ref="NP4_INDEPENDENT"):
    ds=[r[f"{arm}_selected"]-r[f"{ref}_selected"] for r in rows]
    od=[r[f"{arm}_oracle"]-r[f"{ref}_oracle"] for r in rows]
    wins=sum(d>1e-9 for d in ds); ties=sum(abs(d)<=1e-9 for d in ds); losses=sum(d<-1e-9 for d in ds)
    ow=sum(d>1e-9 for d in od); ot=sum(abs(d)<=1e-9 for d in od); ol=sum(d<-1e-9 for d in od)
    rescues=sum((not r[f"{ref}_selected_good25"]) and r[f"{arm}_selected_good25"] for r in rows)
    harms=sum(r[f"{ref}_selected_good25"] and (not r[f"{arm}_selected_good25"]) for r in rows)
    return {
        "n":len(rows),
        "selected_delta_mean_db":statistics.mean(ds),
        "selected_delta_median_db":statistics.median(ds),
        "selected_wins_ties_losses":[wins,ties,losses],
        "oracle_delta_mean_db":statistics.mean(od),
        "oracle_delta_median_db":statistics.median(od),
        "oracle_wins_ties_losses":[ow,ot,ol],
        "selected_good25_rescues":rescues,
        "selected_good25_harms":harms,
        "selected_large_rescues_ge5db":sum(d>=5.0 for d in ds),
        "selected_large_harms_le_minus5db":sum(d<=-5.0 for d in ds),
        "selector_gap_mean_db":statistics.mean(r[f"{arm}_gap"] for r in rows),
        "selector_gap_median_db":statistics.median(r[f"{arm}_gap"] for r in rows),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--refinement-run",type=Path,required=True)
    args=ap.parse_args()
    run=args.refinement_run.resolve()
    manifest=load(run/"REFINEMENT_MANIFEST.json")
    rows=[]
    for task in manifest["tasks"]:
        new_dir=run/"workers"/f"gpu{task['assigned_gpu']}"/task["output_subdir"]/"methods"
        new_summary=load(new_dir/"SMOKE_COMPLETE.json")
        if new_summary.get("status")!="PASS": raise RuntimeError(f"new summary not PASS: {new_dir}")
        old_dir=Path(task["source_methods_dir"])
        row={"class_label":task["class_label"],"image_id":task["image_id"]}
        for arm in OLD_ARMS:
            p=load(old_dir/arm/"result.json")
            m=arm_metrics(p)
            for k,v in m.items(): row[f"{arm}_{k}"]=v
        for arm in NEW_ARMS:
            p=load(new_dir/arm/"result.json")
            m=arm_metrics(p)
            for k,v in m.items(): row[f"{arm}_{k}"]=v
        rows.append(row)
    rows.sort(key=lambda r:(r["class_label"],r["image_id"]))
    if len(rows)!=16: raise RuntimeError(f"expected 16 rows, got {len(rows)}")

    by_class=defaultdict(list)
    for r in rows: by_class[r["class_label"]].append(r)
    summary={
        "schema_version":"b24.epp321-summary.v1",
        "status":"PASS",
        "image_count":16,
        "source_refinement_run":str(run),
        "compute_policy":{
            "NP_EPP_321_total_unet_evals":8800,
            "NP_EPP_321_RANDOM_PRUNE_total_unet_evals":8800,
            "NP_EPP_321_NO_REALLOCATION_total_unet_evals":6900,
            "NP4_INDEPENDENT_total_unet_evals":8800,
            "note":"Within-NP UNet counts are a fixed-work proxy because model/resolution are identical. Cross-family DAPS/SITCOM comparisons require separate FLOP/FRE accounting before final claims."
        },
        "vs_np4":{arm:summarize_delta(rows,arm) for arm in NEW_ARMS},
        "vs_np4_by_screening_stratum":{
            label:{arm:summarize_delta(vals,arm) for arm in NEW_ARMS}
            for label,vals in sorted(by_class.items())
        },
        "main_vs_old_epp":summarize_delta(rows,"NP_EPP_321",ref="NP_EPP"),
    }
    out_json=run/"EPP321_REFINEMENT_SUMMARY.json"
    out_csv=run/"EPP321_REFINEMENT_PER_IMAGE.csv"
    out_json.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    fields=list(rows[0].keys())
    with out_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,sort_keys=True))
    print(f"PER_IMAGE_CSV|{out_csv}")
    print(f"SUMMARY_JSON|{out_json}")


if __name__=="__main__":
    main()
