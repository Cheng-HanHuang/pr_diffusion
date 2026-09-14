#!/usr/bin/env python3
"""Zero-GPU B24.3 development closeout.

This stage performs analysis/packaging only on already-exposed DEV80 artifacts:
- historical Fresh2 clean-free DAPS selection (exact operator-loss margin 0.7),
- complementarity/failure-overlap tables,
- cross-family compute audit closeout with explicit unsupported Fourier inventory,
- negative-development-result report.

No measurement generation, no GPU work, and no confirmation access are allowed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path("/egr/research-pac/huang248")
DAPS_ROOT = ROOT / "pr_diffusion_b19_solver" / "external" / "daps"
SPEC_PATH = Path(__file__).resolve().parents[2] / "configs" / "b24" / "b24_3_zero_gpu_closeout.json"
FRESH2_THETA = 0.7
GOOD25 = 25.0
LARGE_DB = 5.0

EXECUTABLE_METHODS = (
    "DAPS1",
    "FRESH2_SELECTED",
    "SITCOM1",
    "NP4_SELECTED",
    "EPP321_SELECTED",
    "PE3_SCORE_SELECTED",
    "PE3_RANDOM_SELECTED",
)
ORACLE_METHODS = (
    "DAPS2_ORACLE",
    "DAPS4_ORACLE",
    "SITCOM4_ORACLE",
    "NP4_ORACLE",
    "EPP321_ORACLE",
    "PE3_SCORE_ORACLE",
    "PE3_RANDOM_ORACLE",
)


def readj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def writej(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv_index(path: Path, key: str = "image_id") -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    out = {str(r[key]).zfill(5): r for r in rows}
    if len(out) != len(rows):
        raise RuntimeError(f"duplicate {key} in {path}")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    s = sorted(float(x) for x in values)
    p = q * (len(s) - 1)
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - p) + s[hi] * (p - lo)


def dist(values: list[float]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("empty distribution")
    return {
        "n": len(values),
        "mean_db": statistics.fmean(values),
        "median_db": statistics.median(values),
        "min_db": min(values),
        "q10_db": percentile(values, 0.10),
        "q25_db": percentile(values, 0.25),
        "good25_count": sum(x >= GOOD25 for x in values),
        "good25_rate": sum(x >= GOOD25 for x in values) / len(values),
    }


def paired(a: list[float], b: list[float]) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        raise RuntimeError("bad paired vectors")
    d = [x - y for x, y in zip(a, b)]
    return {
        "n": len(d),
        "delta_mean_db": statistics.fmean(d),
        "delta_median_db": statistics.median(d),
        "wins_ties_losses": [sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)],
        "good25_rescues": sum(x >= GOOD25 and y < GOOD25 for x, y in zip(a, b)),
        "good25_harms": sum(x < GOOD25 and y >= GOOD25 for x, y in zip(a, b)),
        "large_rescues_ge5db": sum(x >= LARGE_DB for x in d),
        "large_harms_le_minus5db": sum(x <= -LARGE_DB for x in d),
    }


def overlap(a: list[float], b: list[float]) -> dict[str, int]:
    ag = [x >= GOOD25 for x in a]; bg = [x >= GOOD25 for x in b]
    return {
        "both_good": sum(x and y for x, y in zip(ag, bg)),
        "a_only_good": sum(x and not y for x, y in zip(ag, bg)),
        "b_only_good": sum((not x) and y for x, y in zip(ag, bg)),
        "both_fail": sum((not x) and (not y) for x, y in zip(ag, bg)),
        "good25_union": sum(x or y for x, y in zip(ag, bg)),
    }


def load_png_model_range(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if arr.shape != (256, 256, 3):
        raise RuntimeError(f"unexpected DAPS PNG shape {arr.shape}: {path}")
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous() * 2.0 - 1.0


def scalar_loss(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().flatten()[0].cpu().item())
    return float(value)


def find_recursive_key(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key:
                found.append(v)
            found.extend(find_recursive_key(v, key))
    elif isinstance(value, list):
        for v in value:
            found.extend(find_recursive_key(v, key))
    return found


def load_resolved_daps_config(audit_run: Path) -> tuple[dict[str, Any] | None, str | None]:
    configs = sorted((audit_run / "daps1").rglob("config.yaml"))
    if not configs:
        return None, None
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(configs[0].read_text(encoding="utf-8"))
        return cfg, str(configs[0].resolve())
    except Exception:
        return None, str(configs[0].resolve())


def extract_num_steps(cfg: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not cfg:
        return result
    anneal_blocks = find_recursive_key(cfg, "annealing_scheduler_config")
    mcmc_blocks = find_recursive_key(cfg, "mcmc_sampler_config")
    diff_blocks = find_recursive_key(cfg, "diffusion_scheduler_config")
    if anneal_blocks:
        result["annealing_num_steps_config"] = int(anneal_blocks[0].get("num_steps"))
    if mcmc_blocks:
        result["mcmc_num_steps"] = int(mcmc_blocks[0].get("num_steps"))
    if diff_blocks:
        result["diffusion_num_steps_config"] = int(diff_blocks[0].get("num_steps"))
    if "annealing_num_steps_config" in result and "mcmc_num_steps" in result:
        # DAPS Scheduler stores config num_steps + one initial point, while DAPS.sample
        # loops self.annealing_scheduler.num_steps - 1. Therefore the number of
        # annealing updates equals the configured num_steps.
        result["core_annealing_updates"] = result["annealing_num_steps_config"]
        result["core_mcmc_operator_gradient_calls"] = result["annealing_num_steps_config"] * result["mcmc_num_steps"]
    return result


def fft_work_units(channels: int = 3, h: int = 384, w: int = 384) -> float:
    # Not FLOPs. This is a transparent N log N scale for one batched/channelwise
    # 2-D complex transform; 384 is non-power-of-two so implementation FLOPs depend
    # on the actual FFT backend/algorithm.
    return float(channels * h * w * (math.log2(h) + math.log2(w)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev80-run", type=Path, required=True)
    ap.add_argument("--pe3-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("zero-GPU closeout requires CUDA_VISIBLE_DEVICES empty/-1")
    if torch.cuda.is_available():
        raise RuntimeError("zero-GPU closeout unexpectedly sees CUDA")
    if not SPEC_PATH.is_file():
        raise FileNotFoundError(SPEC_PATH)
    spec = readj(SPEC_PATH)
    if spec.get("gpu_work_authorized") is not False or spec.get("confirmation_exposed") is not False:
        raise RuntimeError("closeout spec authorization drift")

    dev = args.dev80_run.resolve(); pe3 = args.pe3_run.resolve(); out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    dev_manifest = readj(dev / "DEV80_MANIFEST.json")
    dev_summary = readj(dev / "DEV80_SUMMARY.json")
    pe3_manifest = readj(pe3 / "PE3_MANIFEST.json")
    pe3_summary = readj(pe3 / "PE3_DEV80_SUMMARY.json")
    if dev_manifest.get("image_count") != 80 or dev_summary.get("status") != "PASS":
        raise RuntimeError("DEV80 source identity drift")
    if pe3_manifest.get("image_count") != 80 or pe3_summary.get("status") != "PASS":
        raise RuntimeError("PE3 source identity drift")
    if dev_manifest.get("confirmation_exposed") is not False or pe3_summary.get("confirmation_exposed") is not False:
        raise RuntimeError("confirmation exposure detected")
    if pe3_summary.get("decision") != "STOP_B24_METHOD_REFINEMENT" or pe3_summary.get("passing_arms") != []:
        raise RuntimeError("PE3 frozen stop verdict drift")

    dev_csv = read_csv_index(dev / "DEV80_PER_IMAGE.csv")
    pe3_csv = read_csv_index(pe3 / "PE3_DEV80_PER_IMAGE.csv")
    if set(dev_csv) != set(pe3_csv) or len(dev_csv) != 80:
        raise RuntimeError("DEV80/PE3 image set mismatch")

    # Import the exact pinned DAPS operator on CPU.
    if not DAPS_ROOT.is_dir():
        raise FileNotFoundError(DAPS_ROOT)
    sys.path.insert(0, str(DAPS_ROOT))
    from forward_operator import get_operator  # type: ignore  # noqa: E402
    operator = get_operator(name="phase_retrieval", sigma=0.05, oversample=2.0)

    combined_rows: list[dict[str, Any]] = []
    fresh_rows: list[dict[str, Any]] = []
    for task in dev_manifest["tasks"]:
        image = str(task["image_id"]).zfill(5)
        taskdir = dev / "workers" / f"gpu{task['assigned_gpu']}" / task["output_subdir"]
        comp = readj(taskdir / "IMAGE_COMPLETE.json")
        if comp.get("status") != "PASS" or comp.get("confirmation_exposed") is not False:
            raise RuntimeError(f"bad DEV80 completion {taskdir}")
        metrics = readj(Path(comp["baseline_metrics"]))
        daps_group = readj(Path(comp["daps_group"]))
        candidates = sorted(daps_group.get("candidate_rows", []), key=lambda r: int(r["rep"]))
        if len(candidates) != 4 or [int(r["rep"]) for r in candidates] != [0, 1, 2, 3]:
            raise RuntimeError(f"DAPS candidate identity drift: {image}")
        daps_metric_rows = {
            int(r["rep"]): r for r in metrics["terminal_rows"] if r["method"] == "DAPS"
        }
        if set(daps_metric_rows) != {0, 1, 2, 3}:
            raise RuntimeError(f"DAPS metric rows drift: {image}")

        input_manifest = Path(task["source_input_manifest"]).resolve() if task["pilot16"] else taskdir / "input" / "input_manifest.json"
        inp = readj(input_manifest)
        measurement_path = Path(inp.get("measurement_path") or inp.get("measurement_tensor_path", "")).resolve()
        payload = torch.load(measurement_path, map_location="cpu")
        y = payload.get("measurement") if isinstance(payload, dict) else payload
        if not torch.is_tensor(y) or tuple(y.shape) != (1, 3, 384, 384):
            raise RuntimeError(f"bad measurement payload: {measurement_path}")
        y = y.float().cpu()

        losses: dict[int, float] = {}
        for rep in (0, 1):
            terminal = Path(candidates[rep]["terminal_path"]).resolve()
            if not terminal.is_file():
                raise FileNotFoundError(terminal)
            x = load_png_model_range(terminal)
            with torch.no_grad():
                losses[rep] = scalar_loss(operator.loss(x, y))
            if not math.isfinite(losses[rep]):
                raise RuntimeError(f"nonfinite Fresh2 loss image={image} rep={rep}")

        selected_rep = 1 if losses[1] < losses[0] - FRESH2_THETA else 0
        psnr0 = float(daps_metric_rows[0]["psnr_raw_rgb_db"])
        psnr1 = float(daps_metric_rows[1]["psnr_raw_rgb_db"])
        fresh_psnr = psnr1 if selected_rep == 1 else psnr0
        oracle2_rep = 0 if psnr0 >= psnr1 else 1
        oracle2_psnr = max(psnr0, psnr1)
        fresh = {
            "image_id": image,
            "screening_stratum": task["class_label"],
            "fresh_baseline_class": comp["fresh_baseline_class"],
            "daps_rep0_exact_operator_loss": losses[0],
            "daps_rep1_exact_operator_loss": losses[1],
            "fresh2_loss_improvement_rep0_minus_rep1": losses[0] - losses[1],
            "fresh2_theta": FRESH2_THETA,
            "fresh2_selected_rep": selected_rep,
            "fresh2_accept_extra": selected_rep == 1,
            "fresh2_selected_psnr_8bit_db": fresh_psnr,
            "daps2_oracle_rep": oracle2_rep,
            "daps2_oracle_psnr_8bit_db": oracle2_psnr,
            "fresh2_selector_gap_to_daps2_oracle_db": oracle2_psnr - fresh_psnr,
        }
        fresh_rows.append(fresh)

        d = dev_csv[image]; p = pe3_csv[image]
        row = dict(fresh)
        row.update({
            "DAPS1": float(d["daps1_psnr_8bit_db"]),
            "FRESH2_SELECTED": fresh_psnr,
            "DAPS2_ORACLE": oracle2_psnr,
            "DAPS4_ORACLE": float(d["daps4_oracle_psnr_8bit_db"]),
            "SITCOM1": float(d["sitcom1_psnr_8bit_db"]),
            "SITCOM4_ORACLE": float(d["sitcom4_oracle_psnr_8bit_db"]),
            "NP4_SELECTED": float(d["np4_independent_selected_psnr_8bit_db"]),
            "NP4_ORACLE": float(d["np4_independent_oracle_psnr_8bit_db"]),
            "EPP321_SELECTED": float(d["np_epp_321_selected_psnr_8bit_db"]),
            "EPP321_ORACLE": float(d["np_epp_321_oracle_psnr_8bit_db"]),
            "PE3_SCORE_SELECTED": float(p["np_pe3_score_selected_psnr_8bit_db"]),
            "PE3_SCORE_ORACLE": float(p["np_pe3_score_oracle_psnr_8bit_db"]),
            "PE3_RANDOM_SELECTED": float(p["np_pe3_random_selected_psnr_8bit_db"]),
            "PE3_RANDOM_ORACLE": float(p["np_pe3_random_oracle_psnr_8bit_db"]),
        })
        combined_rows.append(row)

    combined_rows.sort(key=lambda r: (r["screening_stratum"], r["image_id"]))
    fresh_rows.sort(key=lambda r: (r["screening_stratum"], r["image_id"]))
    if len(combined_rows) != 80 or len({r["image_id"] for r in combined_rows}) != 80:
        raise RuntimeError("closeout image count drift")

    def vec(method: str) -> list[float]:
        return [float(r[method]) for r in combined_rows]

    fresh2 = vec("FRESH2_SELECTED"); daps1 = vec("DAPS1"); daps2o = vec("DAPS2_ORACLE")
    fresh_summary = {
        "schema_version": "b24.fresh2-dev80.v1",
        "status": "PASS",
        "n": 80,
        "theta": FRESH2_THETA,
        "selection_rule": "choose rep1 iff exact_loss_rep1 < exact_loss_rep0 - 0.7; otherwise rep0",
        "selection_uses_ground_truth": False,
        "rep1_accept_count": sum(bool(r["fresh2_accept_extra"]) for r in fresh_rows),
        "selected_distribution": dist(fresh2),
        "daps2_oracle_distribution": dist(daps2o),
        "vs_daps1": paired(fresh2, daps1),
        "vs_daps2_oracle": paired(fresh2, daps2o),
        "vs_daps4_oracle": paired(fresh2, vec("DAPS4_ORACLE")),
        "vs_np4_selected": paired(fresh2, vec("NP4_SELECTED")),
        "vs_pe3_score_selected": paired(fresh2, vec("PE3_SCORE_SELECTED")),
        "selector_gap_mean_db": statistics.fmean(float(r["fresh2_selector_gap_to_daps2_oracle_db"]) for r in fresh_rows),
        "selector_gap_median_db": statistics.median(float(r["fresh2_selector_gap_to_daps2_oracle_db"]) for r in fresh_rows),
        "confirmation_exposed": False,
    }

    all_methods = EXECUTABLE_METHODS + ORACLE_METHODS
    method_distributions = {m: dist(vec(m)) for m in all_methods}
    pair_rows: list[dict[str, Any]] = []
    pair_json: dict[str, Any] = {}
    for a, b in combinations(EXECUTABLE_METHODS, 2):
        ov = overlap(vec(a), vec(b)); pp = paired(vec(a), vec(b))
        key = f"{a}__VS__{b}"
        pair_json[key] = {"good25_overlap": ov, "a_minus_b": pp}
        pair_rows.append({"method_a": a, "method_b": b, **ov, **{f"a_minus_b_{k}": v for k, v in pp.items() if k != "wins_ties_losses"}, "a_minus_b_wins": pp["wins_ties_losses"][0], "ties": pp["wins_ties_losses"][1], "a_minus_b_losses": pp["wins_ties_losses"][2]})

    failure_sets = {m: [r["image_id"] for r in combined_rows if float(r[m]) < GOOD25] for m in EXECUTABLE_METHODS}
    unique_good = {
        m: [
            r["image_id"] for r in combined_rows
            if float(r[m]) >= GOOD25 and all(float(r[o]) < GOOD25 for o in EXECUTABLE_METHODS if o != m)
        ]
        for m in EXECUTABLE_METHODS
    }
    both_oracles_fail = [r for r in combined_rows if float(r["DAPS4_ORACLE"]) < GOOD25 and float(r["SITCOM4_ORACLE"]) < GOOD25]
    hard_rows = []
    for r in both_oracles_fail:
        hard_rows.append({"image_id": r["image_id"], "screening_stratum": r["screening_stratum"], **{m: r[m] for m in all_methods}})

    complementarity = {
        "schema_version": "b24.dev80-complementarity.v1",
        "status": "PASS",
        "n": 80,
        "good25_threshold_db": GOOD25,
        "executable_methods": list(EXECUTABLE_METHODS),
        "diagnostic_oracles": list(ORACLE_METHODS),
        "method_distributions": method_distributions,
        "pairwise_executable": pair_json,
        "failure_sets": failure_sets,
        "unique_good25_image_ids": unique_good,
        "fresh_both_baseline_oracles_fail_count": len(both_oracles_fail),
        "fresh_both_baseline_oracles_fail_image_ids": [r["image_id"] for r in both_oracles_fail],
        "confirmation_exposed": False,
        "interpretation_guard": "Pairwise unions/oracles diagnose complementarity only; they are not executable selectors unless explicitly labeled as such.",
    }

    audit_path = pe3 / "compute_audit" / "run" / "CROSS_FAMILY_FLOP_AUDIT.json"
    audit = readj(audit_path)
    if audit.get("status") != "PASS" or audit.get("confirmation_exposed") is not False:
        raise RuntimeError("cross-family audit identity drift")
    supported = dict(audit["dispatch_supported_dynamic_flops"])
    pe_flops = int(supported["NP_PE3_SCORE"]); daps1_flops = int(supported["DAPS1"])
    supported["FRESH2_TWO_DAPS_TRAJECTORIES"] = 2 * daps1_flops
    supported["DAPS2_INDEPENDENT_EQUIVALENT"] = 2 * daps1_flops

    audit_run = audit_path.parent
    daps_cfg, daps_cfg_path = load_resolved_daps_config(audit_run)
    daps_counts = extract_num_steps(daps_cfg)
    daps_source_files = [
        DAPS_ROOT / "forward_operator" / "__init__.py",
        DAPS_ROOT / "forward_operator" / "fastmri_utils.py",
        DAPS_ROOT / "cores" / "mcmc.py",
        DAPS_ROOT / "cores" / "scheduler.py",
    ]
    source_hashes = {str(p.resolve()): sha256_file(p) for p in daps_source_files if p.is_file()}

    units = fft_work_units()
    np_fourier = {
        "derivation": "8796 proposals * 3 measurement-magnitude FFT evaluations/proposal + 2796 projection forward FFTs; each postprojection projection also uses one inverse FFT",
        "forward_complex_2d_fft_calls_384x384x3": 3 * 8796 + 2796,
        "inverse_complex_2d_fft_calls_384x384x3": 2796,
        "total_fft_calls": 3 * 8796 + 2 * 2796,
        "work_units_per_fft_call_C*H*W*(log2H+log2W)": units,
        "forward_work_units": (3 * 8796 + 2796) * units,
        "inverse_work_units": 2796 * units,
        "is_exact_flop_count": False,
    }
    daps_fourier = {
        "phase_retrieval_forward_semantics": "x[-1,1]->x[0,1], pad 64 each side to 384x384, centered orthonormal complex 2-D FFT, magnitude",
        "operator_loss_semantics": "sum((A(x)-y)^2) over flattened measurement coordinates",
        "one_forward_fft_per_operator_forward": True,
        "resolved_config_path": daps_cfg_path,
        **daps_counts,
        "core_forward_fft_calls_if_resolved_counts_apply": daps_counts.get("core_mcmc_operator_gradient_calls"),
        "gradient_note": "Each core operator.gradient call differentiates through the phase-retrieval FFT; backward FFT/custom work is not converted to exact FLOPs here.",
        "additional_noncore_operator_calls": "not claimed exact; evaluator/measurement setup may add calls beyond the core MCMC count",
        "work_units_per_forward_fft_call_C*H*W*(log2H+log2W)": units,
        "source_sha256": source_hashes,
    }
    sitcom_diag = audit.get("runtime_diagnostics", {}).get("SITCOM1", {})
    compute_closeout = {
        "schema_version": "b24.compute-closeout.v1",
        "status": "PASS",
        "metric_primary": audit["metric"],
        "dispatch_supported_dynamic_flops": supported,
        "ratios_to_pe3_score_dispatch_supported": {k: float(v) / pe_flops for k, v in supported.items() if k != "NP_PE3_SCORE"},
        "fresh2_selector_overhead": "two additional terminal DAPS phase-retrieval operator-loss evaluations; these FFTs are not represented in dispatch-supported FLOPs",
        "np_pe3_static_fourier_inventory": np_fourier,
        "daps_static_fourier_inventory": daps_fourier,
        "sitcom_runtime_unsupported_work_evidence": {
            "autograd_backward_calls": sitcom_diag.get("runtime_counters", {}).get("autograd_backward_calls"),
            "note": "SITCOM1 recorded 20k autograd backward calls in the completed audit. The original wrapper only intercepted torch.fft.fft2/ifft2, while the phase-retrieval implementations may use fftn/other operator paths; zero observed fft2 calls therefore do not establish zero Fourier work.",
        },
        "exact_total_flop_equivalence_claimed": False,
        "interpretation_guard": "Dispatch-supported FLOP ratios remain useful but are not exact total-compute ratios. Unsupported FFT/custom forward and backward work is explicitly nonzero and implementation dependent.",
        "confirmation_exposed": False,
        "source_audit": str(audit_path.resolve()),
    }

    # Compact manuscript/planner-oriented closeout.
    np4 = method_distributions["NP4_SELECTED"]
    epp = method_distributions["EPP321_SELECTED"]
    pe_score = method_distributions["PE3_SCORE_SELECTED"]
    lines = [
        "# B24 development closeout",
        "",
        "## Frozen decision",
        "",
        "`STOP_B24_METHOD_REFINEMENT` remains binding. No confirmation image was exposed.",
        "",
        "## Historical Fresh2 on DEV80",
        "",
        f"Fresh2 selected Good25: **{fresh_summary['selected_distribution']['good25_count']}/80**; DAPS1: **{method_distributions['DAPS1']['good25_count']}/80**; DAPS2 oracle: **{fresh_summary['daps2_oracle_distribution']['good25_count']}/80**; DAPS4 oracle: **{method_distributions['DAPS4_ORACLE']['good25_count']}/80**.",
        f"Fresh2 accepted the second trajectory on **{fresh_summary['rep1_accept_count']}/80** cases using only exact operator loss with the historical margin 0.7.",
        "",
        "## NP branching development result",
        "",
        f"NP4 selected Good25: **{np4['good25_count']}/80**. EPP321 selected: **{epp['good25_count']}/80**. PE3_SCORE selected: **{pe_score['good25_count']}/80**. Both PE3 arms failed the prospectively frozen NP4 advancement gate.",
        "",
        "B24 therefore supports a negative method-development conclusion: NP-native retention/reallocation can create qualitatively different basins, including occasional rescues, but the tested clean-free lineage decisions did not robustly improve on independent NP populations and caused too many countervailing harms.",
        "",
        "## Compute interpretation",
        "",
        f"Dispatch-supported dynamic FLOPs: PE3_SCORE={pe_flops:,}; DAPS1={daps1_flops:,}; DAPS2/Fresh2 trajectories={2*daps1_flops:,}; DAPS4={int(supported['DAPS4_INDEPENDENT_EQUIVALENT']):,}; SITCOM1={int(supported['SITCOM1']):,}; SITCOM4={int(supported['SITCOM4_INDEPENDENT_EQUIVALENT']):,}.",
        "",
        "These are **not exact total FLOPs**. The frozen phase-retrieval pipelines contain Fourier/custom work that the dispatch counter did not fully count. The closeout inventory records that work separately; no exact cross-family FLOP-equivalence claim is made.",
        "",
        "## Complementarity",
        "",
        f"Fresh measurements where both DAPS4 and SITCOM4 oracle candidates fail Good25: **{len(both_oracles_fail)}/80**. See `HARD_SUBSET.csv`, `COMPLEMENTARITY.json`, and `PAIRWISE_EXECUTABLE.csv` for method-by-method overlap without promoting any post-hoc portfolio.",
        "",
        "## Scope guard",
        "",
        "This closeout is DEV80-only, zero-GPU analysis. Confirmation remains locked, and no further B24 method tuning is authorized by this artifact.",
    ]

    write_csv(out / "FRESH2_PER_IMAGE.csv", fresh_rows)
    writej(out / "FRESH2_SUMMARY.json", fresh_summary)
    write_csv(out / "DEV80_CLOSEOUT_PER_IMAGE.csv", combined_rows)
    write_csv(out / "PAIRWISE_EXECUTABLE.csv", pair_rows)
    write_csv(out / "HARD_SUBSET.csv", hard_rows if hard_rows else [{"image_id": "NONE"}])
    writej(out / "COMPLEMENTARITY.json", complementarity)
    writej(out / "COMPUTE_CLOSEOUT.json", compute_closeout)
    (out / "B24_3_DEV_CLOSEOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    closeout = {
        "schema_version": "b24.dev-closeout.v1",
        "status": "PASS",
        "decision": "STOP_B24_METHOD_REFINEMENT",
        "confirmation_exposed": False,
        "gpu_work_performed": False,
        "measurement_generation_performed": False,
        "source_dev80_run": str(dev),
        "source_pe3_run": str(pe3),
        "fresh2_summary": fresh_summary,
        "compute_closeout": compute_closeout,
        "complementarity_summary": {
            "fresh_both_baseline_oracles_fail_count": len(both_oracles_fail),
            "unique_good25_counts": {k: len(v) for k, v in unique_good.items()},
        },
        "next": "PLANNER_REVIEW_CLOSEOUT_ONLY; confirmation remains locked",
    }
    writej(out / "B24_3_DEV_CLOSEOUT.json", closeout)

    artifacts = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    with (out / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for p in artifacts:
            f.write(f"{sha256_file(p)}  {p.name}\n")

    print(json.dumps({
        "status": "PASS",
        "decision": "STOP_B24_METHOD_REFINEMENT",
        "fresh2_good25": fresh_summary["selected_distribution"]["good25_count"],
        "fresh2_rep1_accepts": fresh_summary["rep1_accept_count"],
        "np4_good25": method_distributions["NP4_SELECTED"]["good25_count"],
        "pe3_score_good25": method_distributions["PE3_SCORE_SELECTED"]["good25_count"],
        "hard_subset_n": len(both_oracles_fail),
        "output": str(out),
        "confirmation_exposed": False,
        "gpu_work_performed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
