#!/usr/bin/env python3
"""A18.7 candidate-set to executable selector audit for Branch A.

This diagnostic script uses existing A8, A11, A14, A16, A17, A17.5, A18,
A18.5, and A18.6 outputs only. It does not run new SITCOM jobs and it does
not change any frozen Branch A policy.

The goal is to ask whether the corrected A18.6 candidate-set signal can be
turned into a real executable selector without using PSNR.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    Image = None  # type: ignore


BASE = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A"
)
OUT_DEFAULT = BASE / "A18_7_candidate_set_executable_selector"
NP_FALLBACK_CSV = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/np_selector_ffhq/"
    "selector_full25_s100_103/lf_s2_selector_20260609_154712/selected_image_level.csv"
)
NP_RUNLEVEL_CSV = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/np_selector_ffhq/"
    "selector_full25_s100_103/lf_s2_selector_20260609_154712/run_level.csv"
)
A18_6_PATH = (
    Path("/egr/research-pac/huang248/pr_diffusion_repo/scripts/npsitcom/")
    / "analyze_branchA_a18_6_corrected_population_score_design.py"
)

DATASETS = {
    "A8": BASE / "A8_sitcom25_trajectory_controller_validation",
    "A11": BASE / "A11_prospective_frozen_policy_validation",
    "A14": BASE / "A14_prospective_dual_policy_validation",
    "A16": BASE / "A16_prospective_dual_policy_replication",
}

TRAIN_SPLITS = [
    ("A8+A11", ["A8", "A11"], ["A14", "A16"]),
    ("A14+A16", ["A14", "A16"], ["A8", "A11"]),
]

SCORE_FIELD = "corrected_health_weighted"
HEALTH_GATE = 0.9581085435181169
IMG_SIZE = 128
LOWFREQ_CROP = 16


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def is_finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if is_finite(x)]
    return float(np.mean(vals)) if vals else math.nan


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def import_a186_module():
    spec = importlib.util.spec_from_file_location("a18_6", str(A18_6_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import A18.6 helper module at {A18_6_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_np_fallback_metadata() -> Dict[str, Dict[str, object]]:
    rows = read_csv(NP_FALLBACK_CSV)
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        image_basename = str(row.get("image_basename", ""))
        if not image_basename:
            continue
        image_id = image_id_from_np_basename(image_basename)
        if str(row.get("alignment_mode", "")) != "resolve":
            continue
        if abs(to_float(row.get("measurement_noise_std")) - 0.05) > 1e-12:
            continue
        # Keep the best fallback according to the recorded LF selector stat.
        rec = {
            "image_id": image_id,
            "np_selected_psnr": to_float(row.get("psnr")),
            "selector_post_winner_lf_mse_mean": to_float(row.get("selector_post_winner_lf_mse_mean")),
            "selected_config": row.get("selected_config", ""),
            "selected_seed": row.get("selected_seed", row.get("seed", "")),
            "selected_config_tag": row.get("config_tag", ""),
        }
        prev = out.get(image_id)
        if prev is None or (
            is_finite(rec["selector_post_winner_lf_mse_mean"])
            and (
                not is_finite(prev["selector_post_winner_lf_mse_mean"])
                or rec["selector_post_winner_lf_mse_mean"] < prev["selector_post_winner_lf_mse_mean"]
                or (
                    rec["selector_post_winner_lf_mse_mean"] == prev["selector_post_winner_lf_mse_mean"]
                    and rec["np_selected_psnr"] > prev["np_selected_psnr"]
                )
            )
        ):
            out[image_id] = rec
    return out


def load_np_target_proxy() -> Dict[str, float]:
    meta = load_np_fallback_metadata()
    return {img: float(rec["selector_post_winner_lf_mse_mean"]) for img, rec in meta.items()}


def load_a18_6_rows():
    mod = import_a186_module()
    rows = mod.merge_rows()
    mod.build_feature_sets(rows)
    train_rows, test_rows = mod.split_rows(rows, ["A8", "A11"], ["A14", "A16"])
    feature_names = [name for name, _, _, _ in mod.RAW_FEATURE_SPECS]
    fit = mod.fit_feature_directions(train_rows, feature_names)
    mod.score_rows(rows, fit, feature_names)
    groups = mod.group_rows(rows)
    np_map = mod.load_np_fallbacks()
    return mod, rows, train_rows, test_rows, groups, np_map, fit


def load_sample_image_index(dataset_dir: Path) -> Dict[Tuple[str, int], Path]:
    """Map (image_id, run_index) -> PNG path for one dataset."""
    mapping: Dict[Tuple[str, int], Path] = {}
    sample_dir = dataset_dir / "samples"
    if not sample_dir.exists():
        return mapping
    pattern = re.compile(r".*_(\d{5})_(\d{5})_run(\d{4})\.png$")
    for path in sample_dir.glob("*.png"):
        m = pattern.match(path.name)
        if not m:
            continue
        image_id = m.group(2)
        run_index = int(m.group(3))
        mapping[(image_id, run_index)] = path
    return mapping


def load_all_sample_images() -> Dict[Tuple[str, str, int], Path]:
    out: Dict[Tuple[str, str, int], Path] = {}
    for dataset, dataset_dir in DATASETS.items():
        mapping = load_sample_image_index(dataset_dir)
        for (image_id, run_index), path in mapping.items():
            out[(dataset, image_id, run_index)] = path
    return out


def img_to_vec(path: Path, size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL/Pillow is not available in the environment")
    with Image.open(path) as im:  # type: ignore[union-attr]
        arr = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    pixel = arr.reshape(-1)
    gray = arr.mean(axis=2)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    c = size // 2
    r = LOWFREQ_CROP // 2
    lowfreq = mag[c - r : c + r, c - r : c + r].reshape(-1)
    lowfreq = lowfreq / (np.linalg.norm(lowfreq) + 1e-12)
    return pixel, lowfreq


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def load_sample_features() -> Dict[Tuple[str, str, int], Dict[str, np.ndarray]]:
    paths = load_all_sample_images()
    feats: Dict[Tuple[str, str, int], Dict[str, np.ndarray]] = {}
    for key, path in paths.items():
        pixel, lowfreq = img_to_vec(path)
        feats[key] = {"pixel": pixel, "lowfreq": lowfreq}
    return feats


def group_by_image(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["image_id"]))].append(row)
    for g in grouped.values():
        g.sort(key=lambda r: int(r["run_index"]))
    return grouped


def sort_by_health(group: List[Dict[str, object]], descending: bool = True, remove_aggressive: bool = False) -> List[Dict[str, object]]:
    rows = group
    if remove_aggressive:
        filtered = [r for r in group if float(r.get("aggressive_flag", 0.0)) < 0.5]
        if filtered:
            rows = filtered
    return sorted(
        rows,
        key=lambda r: (
            -to_float(r.get(SCORE_FIELD)) if descending else to_float(r.get(SCORE_FIELD)),
            int(r["run_index"]),
        ),
    )


def candidate_set_for_policy(
    group: List[Dict[str, object]],
    policy_name: str,
    np_fallback: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, object]], Optional[Dict[str, object]]]:
    if policy_name == "top2_weighted":
        return sort_by_health(group)[:2], None
    if policy_name == "top3_weighted":
        return sort_by_health(group)[:3], None
    if policy_name == "top2_remove_aggressive_weighted":
        return sort_by_health(group, remove_aggressive=True)[:2], None
    if policy_name == "threshold_cap3_weighted":
        healthy = [r for r in sort_by_health(group) if to_float(r.get(SCORE_FIELD)) >= HEALTH_GATE][:3]
        if healthy:
            return healthy, None
        return sort_by_health(group)[:1], None
    if policy_name == "top3_plus_np_fallback":
        return sort_by_health(group)[:3], np_fallback
    raise ValueError(f"Unknown candidate-set policy: {policy_name}")


def candidate_score(row: Dict[str, object]) -> float:
    return to_float(row.get(SCORE_FIELD))


def select_candidate(
    dataset: str,
    image_id: str,
    candidate_rows: List[Dict[str, object]],
    selector_name: str,
    sample_features: Dict[Tuple[str, str, int], Dict[str, np.ndarray]],
    np_proxy_targets: Dict[str, float],
    np_fallback: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    """Return selected row and selector diagnostics."""
    diag: Dict[str, object] = {
        "selector_name": selector_name,
        "selector_used_np_fallback": False,
        "selector_reason": "",
    }

    sit_rows = candidate_rows
    if selector_name == "highest_corrected_health":
        chosen = max(sit_rows, key=lambda r: (candidate_score(r), -int(r["run_index"])))
        return chosen, diag

    if selector_name == "lowest_full_residual_proxy":
        chosen = min(
            sit_rows,
            key=lambda r: (
                to_float(r.get("x0y_full_residual_normed_persist10", math.inf)),
                int(r["run_index"]),
            ),
        )
        return chosen, diag

    if selector_name == "lowest_lowfreq_residual_proxy":
        chosen = min(
            sit_rows,
            key=lambda r: (
                to_float(r.get("x0y_lowfreq_residual_normed_persist10", math.inf)),
                int(r["run_index"]),
            ),
        )
        return chosen, diag

    if selector_name == "closest_to_lowfreq_population_median":
        vecs = [
            sample_features[(dataset, image_id, int(r["run_index"]))]["lowfreq"]
            for r in sit_rows
            if (dataset, image_id, int(r["run_index"])) in sample_features
        ]
        if len(vecs) == len(sit_rows) and vecs:
            median = np.median(np.stack(vecs, axis=0), axis=0)
            chosen = min(
                sit_rows,
                key=lambda r: (
                    l2(sample_features[(dataset, image_id, int(r["run_index"]))]["lowfreq"], median),
                    int(r["run_index"]),
                ),
            )
            return chosen, diag
        diag["selector_reason"] = "missing_lowfreq_samples_fallback_health"
        return max(sit_rows, key=lambda r: (candidate_score(r), -int(r["run_index"]))), diag

    if selector_name == "closest_to_pixel_population_median":
        vecs = [
            sample_features[(dataset, image_id, int(r["run_index"]))]["pixel"]
            for r in sit_rows
            if (dataset, image_id, int(r["run_index"])) in sample_features
        ]
        if len(vecs) == len(sit_rows) and vecs:
            median = np.median(np.stack(vecs, axis=0), axis=0)
            chosen = min(
                sit_rows,
                key=lambda r: (
                    l2(sample_features[(dataset, image_id, int(r["run_index"]))]["pixel"], median),
                    int(r["run_index"]),
                ),
            )
            return chosen, diag
        diag["selector_reason"] = "missing_pixel_samples_fallback_health"
        return max(sit_rows, key=lambda r: (candidate_score(r), -int(r["run_index"]))), diag

    if selector_name == "closest_to_np_fallback_lowfreq_proxy":
        target = np_proxy_targets.get(image_id, math.nan)
        if math.isfinite(target):
            chosen = min(
                sit_rows,
                key=lambda r: (
                    abs(to_float(r.get("x0y_lowfreq_residual_normed_persist10")) - target),
                    int(r["run_index"]),
                ),
            )
            return chosen, diag
        diag["selector_reason"] = "missing_np_proxy_target_fallback_health"
        return max(sit_rows, key=lambda r: (candidate_score(r), -int(r["run_index"]))), diag

    if selector_name == "combined_filter_then_lowfreq":
        filtered = [r for r in sit_rows if float(r.get("aggressive_flag", 0.0)) < 0.5 and candidate_score(r) >= HEALTH_GATE]
        if not filtered:
            filtered = [r for r in sit_rows if candidate_score(r) >= HEALTH_GATE]
        if filtered:
            chosen = min(
                filtered,
                key=lambda r: (
                    to_float(r.get("x0y_lowfreq_residual_normed_persist10", math.inf)),
                    int(r["run_index"]),
                ),
            )
            return chosen, diag
        diag["selector_reason"] = "no_healthy_candidate_fallback_health"
        return max(sit_rows, key=lambda r: (candidate_score(r), -int(r["run_index"]))), diag

    if selector_name == "fallback_if_gate_fail":
        healthy = [r for r in sit_rows if candidate_score(r) >= HEALTH_GATE]
        if healthy:
            chosen = max(healthy, key=lambda r: (candidate_score(r), -int(r["run_index"])))
            return chosen, diag
        if np_fallback is not None:
            diag["selector_used_np_fallback"] = True
            diag["selector_reason"] = "all_sitcom_candidates_failed_gate"
            return np_fallback, diag
        diag["selector_reason"] = "no_np_fallback_available_fallback_health"
        return max(sit_rows, key=lambda r: (candidate_score(r), -int(r["run_index"]))), diag

    raise ValueError(f"Unknown selector: {selector_name}")


def evaluate_policy(
    dataset: str,
    image_id: str,
    group: List[Dict[str, object]],
    policy_name: str,
    selector_name: str,
    sample_features: Dict[Tuple[str, str, int], Dict[str, np.ndarray]],
    np_proxy_targets: Dict[str, float],
    np_fallback_meta: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    fallback = np_fallback_meta.get(image_id)
    candidate_rows, fallback_row = candidate_set_for_policy(group, policy_name, fallback)
    selected, diag = select_candidate(
        dataset,
        image_id,
        candidate_rows,
        selector_name,
        sample_features,
        np_proxy_targets,
        np_fallback=fallback_row,
    )
    if selected is None and fallback_row is not None:
        selected = fallback_row  # diagnostic fallback when selector cannot use samples
        diag["selector_used_np_fallback"] = True
        diag["selector_reason"] = f"{diag.get('selector_reason', '')}|proxy_fallback"
    assert selected is not None, f"No selector result for {dataset}/{image_id}/{policy_name}/{selector_name}"

    candidate_psnrs = [to_float(r.get("final_psnr")) for r in candidate_rows]
    selected_psnr = to_float(selected.get("final_psnr", selected.get("np_selected_psnr", math.nan)))
    candidate_best_psnr = max(candidate_psnrs) if candidate_psnrs else math.nan
    selected_is_np = bool(diag.get("selector_used_np_fallback", False)) or selected.get("image_id") == image_id and "np_selected_psnr" in selected
    return {
        "dataset": dataset,
        "image_id": image_id,
        "policy_name": policy_name,
        "selector_name": selector_name,
        "selected_run_index": selected.get("run_index", ""),
        "selected_psnr": selected_psnr,
        "selected_bad25": selected_psnr < 25.0,
        "selected_bad20": selected_psnr < 20.0,
        "candidate_set_runs": json.dumps([int(r["run_index"]) for r in candidate_rows]),
        "candidate_set_scores": json.dumps([candidate_score(r) for r in candidate_rows]),
        "candidate_best_psnr": candidate_best_psnr,
        "candidate_best_run_index": int(candidate_rows[int(np.argmax(candidate_psnrs))]["run_index"]) if candidate_rows else "",
        "candidate_set_size": len(candidate_rows),
        "selected_is_np_fallback": selected_is_np,
        "np_fallback_psnr": to_float(fallback_row.get("np_selected_psnr")) if fallback_row else math.nan,
        "np_fallback_target_lf_stat": to_float(np_proxy_targets.get(image_id, math.nan)),
        "selector_reason": diag.get("selector_reason", ""),
        "selector_used_np_fallback": bool(diag.get("selector_used_np_fallback", False)),
        "selected_corrected_health": candidate_score(selected) if "run_index" in selected else math.nan,
        "selected_aggressive_flag": bool(selected.get("aggressive_flag", 0.0)) if "aggressive_flag" in selected else False,
        "candidate_best_minus_selected": candidate_best_psnr - selected_psnr if math.isfinite(candidate_best_psnr) and math.isfinite(selected_psnr) else math.nan,
    }


def load_aggressive_controller_image_best(dataset: str) -> Dict[str, float]:
    path = DATASETS[dataset] / "aggressive_policy_applied_runs.csv"
    if not path.exists():
        return {}
    rows = read_csv(path)
    by_img: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_img[str(row["image_id"])].append(to_float(row["policy_final_psnr"]))
    return {img: max(vals) for img, vals in by_img.items() if vals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    mod, rows, train_rows, test_rows, groups, np_map, fit = load_a18_6_rows()
    sample_features = load_sample_features() if PIL_AVAILABLE else {}
    np_proxy_targets = load_np_target_proxy()
    np_fallback_meta = load_np_fallback_metadata()
    all_groups = group_by_image(rows)
    train_groups = group_by_image(train_rows)
    test_groups = group_by_image(test_rows)

    candidate_policies = [
        "top2_weighted",
        "top3_weighted",
        "top2_remove_aggressive_weighted",
        "top3_plus_np_fallback",
        "threshold_cap3_weighted",
    ]
    selector_names = [
        "highest_corrected_health",
        "lowest_full_residual_proxy",
        "lowest_lowfreq_residual_proxy",
        "closest_to_lowfreq_population_median",
        "closest_to_pixel_population_median",
        "closest_to_np_fallback_lowfreq_proxy",
        "combined_filter_then_lowfreq",
        "fallback_if_gate_fail",
    ]

    policy_rows: List[Dict[str, object]] = []
    oracle_gap_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    audit_00017: List[Dict[str, object]] = []
    audit_00007: List[Dict[str, object]] = []

    aggressive_image_best = {
        dataset: load_aggressive_controller_image_best(dataset) for dataset in ["A14", "A16"]
    }

    for split_name, train_datasets, test_datasets in TRAIN_SPLITS:
        split_rows = [r for r in rows if str(r["dataset"]) in test_datasets]
        split_groups = group_by_image(split_rows)
        selected_imgs = sorted(split_groups.keys())

        for policy_name in candidate_policies:
            for selector_name in selector_names:
                per_image_rows: List[Dict[str, object]] = []
                for dataset, image_id in selected_imgs:
                    group = split_groups[(dataset, image_id)]
                    # Ensure we are only auditing the held-out or development split in this block.
                    if policy_name == "threshold_cap3_weighted" and selector_name != "fallback_if_gate_fail":
                        # still evaluate, but mark diagnostic via policy name.
                        pass
                    rec = evaluate_policy(
                        dataset,
                        image_id,
                        group,
                        policy_name,
                        selector_name,
                        sample_features,
                        np_proxy_targets,
                        np_fallback_meta,
                    )
                    sitcom_best = max(to_float(r.get("final_psnr")) for r in group)
                    rec["split"] = split_name
                    rec["is_diagnostic_policy"] = policy_name == "threshold_cap3_weighted"
                    rec["candidate_best_psnr_gap_vs_sitcom"] = sitcom_best - rec["candidate_best_psnr"]
                    rec["selected_gap_vs_candidate_best"] = rec["selected_psnr"] - rec["candidate_best_psnr"]
                    rec["selected_gap_vs_sitcom_best"] = rec["selected_psnr"] - sitcom_best
                    rec["sitcom_best_of_4_psnr"] = sitcom_best
                    if dataset in aggressive_image_best:
                        rec["aggressive_controller_best_psnr"] = aggressive_image_best[dataset].get(image_id, math.nan)
                    else:
                        rec["aggressive_controller_best_psnr"] = math.nan
                    per_image_rows.append(rec)

                    if dataset == "A14" and image_id == "00017":
                        audit_00017.append(rec.copy())
                    if dataset == "A16" and image_id == "00017":
                        audit_00017.append(rec.copy())
                    if dataset == "A8" and image_id == "00007":
                        audit_00007.append(rec.copy())

                    if rec["candidate_best_psnr"] >= 25.0 and rec["selected_psnr"] < 25.0:
                        failure_rows.append(
                            {
                                "split": split_name,
                                "dataset": dataset,
                                "image_id": image_id,
                                "policy_name": policy_name,
                                "selector_name": selector_name,
                                "selected_run_index": rec["selected_run_index"],
                                "selected_psnr": rec["selected_psnr"],
                                "selected_bad25": rec["selected_bad25"],
                                "selected_bad20": rec["selected_bad20"],
                                "candidate_best_psnr": rec["candidate_best_psnr"],
                                "sitcom_best_of_4_psnr": max(to_float(r.get("final_psnr")) for r in group),
                                "selected_gap_vs_candidate_best": rec["selected_gap_vs_candidate_best"],
                                "selected_gap_vs_sitcom_best": rec["selected_gap_vs_sitcom_best"],
                                "candidate_set_runs": rec["candidate_set_runs"],
                                "candidate_set_scores": rec["candidate_set_scores"],
                                "selected_is_np_fallback": rec["selected_is_np_fallback"],
                                "selector_reason": rec["selector_reason"],
                            }
                        )

                selected_psnrs = [r["selected_psnr"] for r in per_image_rows]
                candidate_best_psnrs = [r["candidate_best_psnr"] for r in per_image_rows]
                image_best_psnrs = [max(to_float(x.get("final_psnr")) for x in split_groups[(r["dataset"], r["image_id"])]) for r in per_image_rows]
                np_used = sum(1 for r in per_image_rows if bool(r["selected_is_np_fallback"]))
                np_fallback_costs = [
                    r["candidate_best_psnr"] - r["np_fallback_psnr"]
                    for r in per_image_rows
                    if bool(r["selected_is_np_fallback"]) and math.isfinite(r["candidate_best_psnr"]) and math.isfinite(r["np_fallback_psnr"])
                ]
                policy_rows.append(
                    {
                        "split": split_name,
                        "policy_name": policy_name,
                        "selector_name": selector_name,
                        "is_diagnostic_policy": policy_name == "threshold_cap3_weighted",
                        "selected_mean_psnr": mean(selected_psnrs),
                        "selected_min_psnr": float(np.min([x for x in selected_psnrs if is_finite(x)])) if any(is_finite(x) for x in selected_psnrs) else math.nan,
                        "selected_bad25_count": sum(1 for x in selected_psnrs if x < 25.0),
                        "selected_bad20_count": sum(1 for x in selected_psnrs if x < 20.0),
                        "image_best_of_4_mean_psnr": mean(image_best_psnrs),
                        "image_best_of_4_min_psnr": float(np.min(image_best_psnrs)) if image_best_psnrs else math.nan,
                        "candidate_best_mean_psnr": mean(candidate_best_psnrs),
                        "candidate_best_min_psnr": float(np.min(candidate_best_psnrs)) if candidate_best_psnrs else math.nan,
                        "np_fallback_count": np_used,
                        "np_fallback_cost_mean": mean(np_fallback_costs),
                        "remaining_bad25_count": sum(1 for x in selected_psnrs if x < 25.0),
                        "remaining_bad20_count": sum(1 for x in selected_psnrs if x < 20.0),
                        "images_fixed_at_00017": sum(
                            1
                            for r in per_image_rows
                            if r["dataset"] == "A14" and r["image_id"] == "00017" and r["selected_psnr"] >= 25.0
                        ),
                        "images_fixed_at_00017_A16": sum(
                            1
                            for r in per_image_rows
                            if r["dataset"] == "A16" and r["image_id"] == "00017" and r["selected_psnr"] >= 25.0
                        ),
                        "a8_00007_np_fallback": any(
                            r["dataset"] == "A8" and r["image_id"] == "00007" and bool(r["selected_is_np_fallback"])
                            for r in per_image_rows
                        ),
                    }
                )

                # Oracle diagnostic at the image level.
                for r in per_image_rows:
                    gap_row = {
                        "split": split_name,
                        "dataset": r["dataset"],
                        "image_id": r["image_id"],
                        "policy_name": policy_name,
                        "selector_name": selector_name,
                        "selected_run_index": r["selected_run_index"],
                        "selected_psnr": r["selected_psnr"],
                        "candidate_best_psnr": r["candidate_best_psnr"],
                        "sitcom_best_of_4_psnr": r["sitcom_best_of_4_psnr"],
                        "np_fallback_psnr": r["np_fallback_psnr"],
                        "aggressive_controller_best_psnr": r["aggressive_controller_best_psnr"],
                        "selected_gap_vs_candidate_best": r["selected_gap_vs_candidate_best"],
                        "selected_gap_vs_sitcom_best": r["selected_gap_vs_sitcom_best"],
                        "selected_is_np_fallback": r["selected_is_np_fallback"],
                        "selector_reason": r["selector_reason"],
                        "candidate_set_runs": r["candidate_set_runs"],
                        "candidate_set_scores": r["candidate_set_scores"],
                    }
                    oracle_gap_rows.append(gap_row)

    write_csv(outdir / "executable_selector_policy_summary.csv", policy_rows)
    write_csv(outdir / "executable_selector_failure_cases.csv", failure_rows)
    write_csv(outdir / "candidate_set_oracle_gap_summary.csv", oracle_gap_rows)
    write_csv(outdir / "image00017_executable_selector_audit.csv", audit_00017)
    write_csv(outdir / "A8_00007_fallback_audit.csv", audit_00007)

    summary = [
        "# A18.7 Candidate-Set to Executable Selector Audit",
        "",
        "This pass keeps the A18.6 canonical fit frozen on A8+A11 and asks whether the top-k candidate sets can be turned into a real selector without PSNR.",
        "",
        "## What is executable",
        "",
        f"- sample images available: {'yes' if PIL_AVAILABLE else 'no'}",
        f"- canonical health gate used for fallback logic: {HEALTH_GATE}",
        "- top2_weighted and top3_weighted candidate sets are evaluated with several clean-free selectors",
        "- top3_plus_np_fallback is included to test a real fallback policy",
        "- threshold_cap3_weighted is diagnostic-only",
        "",
        "## What is not oracle",
        "",
        "- candidate-best PSNR is still an oracle upper bound",
        "- SITCOM best-of-4 is still an oracle upper bound",
        "- NP fallback PSNR is a recorded fallback source, not a selector feature",
        "",
        "## Files",
        "",
        "- executable_selector_policy_summary.csv",
        "- executable_selector_failure_cases.csv",
        "- candidate_set_oracle_gap_summary.csv",
        "- image00017_executable_selector_audit.csv",
        "- A8_00007_fallback_audit.csv",
    ]
    if not PIL_AVAILABLE:
        summary.append("- pixel / low-frequency image-based selectors were unavailable because Pillow could not be imported")
    else:
        summary.append("- pixel / low-frequency image-based selectors were computed from saved SITCOM sample PNGs")
    write_text(outdir / "SUMMARY.md", "\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
