#!/usr/bin/env python3
"""Aggregate the completed B24.3 DEV80 development stage.

All cross-family PSNR in this summary uses one common raw-orientation 8-bit RGB
representation. Original screening labels remain strata; fresh baseline classes
are recomputed from the DEV80 measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

NP_ARMS = (
    "NP4_INDEPENDENT",
    "NP_EPP_321",
    "NP_EPP_321_RANDOM_PRUNE",
    "NP_EPP_321_NO_REALLOCATION",
)
EXPECTED_TOTALS = {
    "NP4_INDEPENDENT": 8800,
    "NP_EPP_321": 8800,
    "NP_EPP_321_RANDOM_PRUNE": 8800,
    "NP_EPP_321_NO_REALLOCATION": 6900,
}


def readj(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def writej(path: Path, value) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def unwrap(path: Path, keys: tuple[str, ...]) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, dict):
        for key in keys:
            if key in value and torch.is_tensor(value[key]):
                return value[key].float()
    raise RuntimeError(f"cannot unwrap tensor: {path}")


def quantized01(x: torch.Tensor) -> torch.Tensor:
    return torch.round(((x.clamp(-1, 1) + 1.0) * 0.5) * 255.0).clamp(0, 255) / 255.0


def psnr01(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y).square()).clamp_min(1.0e-12)
    return float((10.0 * torch.log10(1.0 / mse)).item())


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(float(x) for x in values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def dist(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_db": statistics.fmean(values),
        "median_db": statistics.median(values),
        "min_db": min(values),
        "q10_db": percentile(values, 0.10),
        "q25_db": percentile(values, 0.25),
        "good25_count": sum(v >= 25.0 for v in values),
        "good25_rate": sum(v >= 25.0 for v in values) / len(values),
    }


def paired(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b) or not a:
        raise RuntimeError("bad paired vectors")
    d = [x - y for x, y in zip(a, b)]
    return {
        "n": len(d),
        "delta_mean_db": statistics.fmean(d),
        "delta_median_db": statistics.median(d),
        "wins_ties_losses": [sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)],
        "good25_rescues": sum(x >= 25.0 and y < 25.0 for x, y in zip(a, b)),
        "good25_harms": sum(x < 25.0 and y >= 25.0 for x, y in zip(a, b)),
        "large_rescues_ge5db": sum(x >= 5.0 for x in d),
        "large_harms_le_minus5db": sum(x <= -5.0 for x in d),
    }


def canonical_np(result_path: Path, gt01: torch.Tensor) -> dict:
    result = readj(result_path)
    arm = result.get("arm")
    if result.get("status") != "PASS" or arm not in EXPECTED_TOTALS:
        raise RuntimeError(f"bad NP result: {result_path}")
    if int(result.get("total_unet_evals", -1)) != EXPECTED_TOTALS[arm]:
        raise RuntimeError(f"NP work drift: {result_path}")
    if bool(result.get("runtime_decisions_use_ground_truth", True)) or bool(result.get("terminal_selection_uses_ground_truth", True)):
        raise RuntimeError(f"NP runtime/selector GT flag drift: {result_path}")
    terms = result.get("terminals", [])
    if len(terms) != 4:
        raise RuntimeError(f"expected four terminals for {arm}: {result_path}")
    scores = []
    for t in terms:
        rec = unwrap(Path(t["reconstruction_path"]), ("reconstruction", "image", "x"))
        if tuple(rec.shape) != (1, 3, 256, 256):
            raise RuntimeError(f"bad NP terminal shape: {t['reconstruction_path']}")
        scores.append(psnr01(quantized01(rec), gt01))
    selected_idx = int(result["clean_free_selected_terminal_index"])
    oracle_idx = max(range(len(scores)), key=lambda i: (scores[i], -i))
    return {
        "selected_psnr_8bit_db": scores[selected_idx],
        "oracle_psnr_8bit_db": scores[oracle_idx],
        "selector_gap_8bit_db": scores[oracle_idx] - scores[selected_idx],
        "selected_terminal_index": selected_idx,
        "oracle_terminal_index_8bit": oracle_idx,
        "native_float_selected_psnr_db": float(result["clean_free_selected_psnr_raw_db"]),
        "native_float_oracle_psnr_db": float(result["oracle_best_psnr_raw_db"]),
        "total_unet_evals": int(result["total_unet_evals"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    manifest = readj(run / "DEV80_MANIFEST.json")
    if manifest.get("image_count") != 80 or manifest.get("confirmation_exposed") is not False:
        raise RuntimeError("bad DEV80 manifest identity")

    rows = []
    for task in manifest["tasks"]:
        taskdir = run / "workers" / f"gpu{task['assigned_gpu']}" / task["output_subdir"]
        complete_path = taskdir / "IMAGE_COMPLETE.json"
        if not complete_path.is_file():
            raise RuntimeError(f"missing completion: {complete_path}")
        comp = readj(complete_path)
        if comp.get("status") != "PASS" or comp.get("image_id") != task["image_id"]:
            raise RuntimeError(f"bad completion: {complete_path}")
        metrics = readj(Path(comp["baseline_metrics"]))
        if metrics.get("image_id") != task["image_id"]:
            raise RuntimeError("baseline metric identity drift")

        input_manifest = (
            Path(task["source_input_manifest"])
            if task["pilot16"]
            else taskdir / "input" / "input_manifest.json"
        )
        inp = readj(input_manifest)
        gt = unwrap(Path(inp["ground_truth_tensor_path"]), ("ground_truth", "image", "x"))
        gt01 = quantized01(gt)

        terminal_rows = metrics["terminal_rows"]
        daps1 = next(float(r["psnr_raw_rgb_db"]) for r in terminal_rows if r["method"] == "DAPS" and int(r["rep"]) == 0)
        sitcom1 = next(float(r["psnr_raw_rgb_db"]) for r in terminal_rows if r["method"] == "SITCOM" and int(r["rep"]) == 0)
        daps4 = float(metrics["methods"]["DAPS"]["best_psnr_raw_rgb_db"])
        sitcom4 = float(metrics["methods"]["SITCOM"]["best_psnr_raw_rgb_db"])

        np = {arm: canonical_np(Path(comp["np_results"][arm]), gt01) for arm in NP_ARMS}
        row = {
            "image_id": task["image_id"],
            "screening_stratum": task["class_label"],
            "fresh_baseline_class": metrics["class_label"],
            "pilot16": bool(task["pilot16"]),
            "daps1_psnr_8bit_db": daps1,
            "daps4_oracle_psnr_8bit_db": daps4,
            "sitcom1_psnr_8bit_db": sitcom1,
            "sitcom4_oracle_psnr_8bit_db": sitcom4,
        }
        for arm in NP_ARMS:
            key = arm.lower()
            row[f"{key}_selected_psnr_8bit_db"] = np[arm]["selected_psnr_8bit_db"]
            row[f"{key}_oracle_psnr_8bit_db"] = np[arm]["oracle_psnr_8bit_db"]
            row[f"{key}_selector_gap_8bit_db"] = np[arm]["selector_gap_8bit_db"]
            row[f"{key}_total_unet_evals"] = np[arm]["total_unet_evals"]
        rows.append(row)

    if len(rows) != 80 or len({r["image_id"] for r in rows}) != 80:
        raise RuntimeError("DEV80 aggregation count/identity drift")
    rows.sort(key=lambda r: (r["screening_stratum"], r["image_id"]))

    def vec(key, subset=rows): return [float(r[key]) for r in subset]
    np4s = vec("np4_independent_selected_psnr_8bit_db")
    np4o = vec("np4_independent_oracle_psnr_8bit_db")
    epps = vec("np_epp_321_selected_psnr_8bit_db")
    eppo = vec("np_epp_321_oracle_psnr_8bit_db")
    daps1 = vec("daps1_psnr_8bit_db"); daps4 = vec("daps4_oracle_psnr_8bit_db")
    sitcom1 = vec("sitcom1_psnr_8bit_db"); sitcom4 = vec("sitcom4_oracle_psnr_8bit_db")

    method_keys = {
        "DAPS1": "daps1_psnr_8bit_db",
        "DAPS4_ORACLE": "daps4_oracle_psnr_8bit_db",
        "SITCOM1": "sitcom1_psnr_8bit_db",
        "SITCOM4_ORACLE": "sitcom4_oracle_psnr_8bit_db",
        "NP4_SELECTED": "np4_independent_selected_psnr_8bit_db",
        "NP4_ORACLE": "np4_independent_oracle_psnr_8bit_db",
        "EPP321_SELECTED": "np_epp_321_selected_psnr_8bit_db",
        "EPP321_ORACLE": "np_epp_321_oracle_psnr_8bit_db",
        "EPP321_RANDOM_SELECTED": "np_epp_321_random_prune_selected_psnr_8bit_db",
        "EPP321_NOREALLOC_SELECTED": "np_epp_321_no_reallocation_selected_psnr_8bit_db",
    }
    distributions = {name: dist(vec(key)) for name, key in method_keys.items()}

    transitions = Counter(f"{r['screening_stratum']}->{r['fresh_baseline_class']}" for r in rows)
    fresh_counts = Counter(r["fresh_baseline_class"] for r in rows)
    fresh_both_fail = [r for r in rows if r["daps4_oracle_psnr_8bit_db"] < 25.0 and r["sitcom4_oracle_psnr_8bit_db"] < 25.0]
    epp_fresh_d_rescues = [r["image_id"] for r in fresh_both_fail if r["np_epp_321_selected_psnr_8bit_db"] >= 25.0]
    np4_fresh_d_rescues = [r["image_id"] for r in fresh_both_fail if r["np4_independent_selected_psnr_8bit_db"] >= 25.0]

    by_screen = {}
    for label in "ABCD":
        sub = [r for r in rows if r["screening_stratum"] == label]
        by_screen[label] = {
            "n": len(sub),
            "epp321_vs_np4_selected": paired(
                vec("np_epp_321_selected_psnr_8bit_db", sub),
                vec("np4_independent_selected_psnr_8bit_db", sub),
            ),
            "method_distributions": {name: dist(vec(key, sub)) for name, key in method_keys.items()},
            "fresh_class_counts": dict(Counter(r["fresh_baseline_class"] for r in sub)),
        }

    ablations = {}
    for arm, key in (
        ("NP_EPP_321_RANDOM_PRUNE", "np_epp_321_random_prune_selected_psnr_8bit_db"),
        ("NP_EPP_321_NO_REALLOCATION", "np_epp_321_no_reallocation_selected_psnr_8bit_db"),
    ):
        ablations[arm] = {
            "vs_np4_selected": paired(vec(key), np4s),
            "vs_epp321_selected": paired(vec(key), epps),
        }

    summary = {
        "schema_version": "b24.dev80-summary.v1",
        "status": "PASS",
        "image_count": 80,
        "pilot16_reused_np_count": 16,
        "new_np_execution_count": 64,
        "fresh_daps4_count": 80,
        "fresh_sitcom4_count": 80,
        "confirmation_exposed": False,
        "primary_representation": "CANONICAL_SAVED_OR_QUANTIZED_RGB_8BIT_RAW_ORIENTATION_V1",
        "method_distributions": distributions,
        "epp321_vs_np4_selected": paired(epps, np4s),
        "epp321_vs_np4_oracle": paired(eppo, np4o),
        "epp321_selected_vs_daps1": paired(epps, daps1),
        "epp321_selected_vs_sitcom1": paired(epps, sitcom1),
        "epp321_selected_vs_daps4_oracle_ceiling": paired(epps, daps4),
        "epp321_selected_vs_sitcom4_oracle_ceiling": paired(epps, sitcom4),
        "ablations": ablations,
        "screening_to_fresh_class_transitions": dict(sorted(transitions.items())),
        "fresh_class_counts": {k: fresh_counts.get(k, 0) for k in "ABCD"},
        "fresh_both_baselines_fail_count": len(fresh_both_fail),
        "fresh_both_baselines_fail_image_ids": [r["image_id"] for r in fresh_both_fail],
        "epp321_selected_rescues_where_both_fresh_baseline_oracles_fail": len(epp_fresh_d_rescues),
        "epp321_selected_rescue_image_ids_where_both_fresh_fail": epp_fresh_d_rescues,
        "np4_selected_rescues_where_both_fresh_baseline_oracles_fail": len(np4_fresh_d_rescues),
        "np4_selected_rescue_image_ids_where_both_fresh_fail": np4_fresh_d_rescues,
        "by_screening_stratum": by_screen,
        "compute_policy": {
            "NP4_INDEPENDENT_total_unet_evals_per_image": 8800,
            "NP_EPP_321_total_unet_evals_per_image": 8800,
            "NP_EPP_321_RANDOM_PRUNE_total_unet_evals_per_image": 8800,
            "NP_EPP_321_NO_REALLOCATION_total_unet_evals_per_image": 6900,
            "within_np_fixed_work_proxy": "same-model UNet forward evaluations",
            "cross_family_flop_equivalence_claimed": False,
            "cross_family_note": "DAPS/SITCOM native timing and protocol are preserved, but final fixed/similar-compute claims require method-specific FLOPs or a defensible forward/operator-equivalent accounting.",
        },
        "source_run": str(run),
        "next": "PLANNER_REVIEW_DEV80_BEFORE_ANY_CONFIRMATION_EXPOSURE",
    }

    csv_path = run / "DEV80_PER_IMAGE.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    writej(run / "DEV80_SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True))
    print(f"PER_IMAGE_CSV|{csv_path}")
    print(f"SUMMARY_JSON|{run/'DEV80_SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
