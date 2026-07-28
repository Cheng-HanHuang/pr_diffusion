#!/usr/bin/env python3
"""B22.3 deterministic scientific analysis of the validated 100-image panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

EXECUTABLE = ["Fresh1", "Fresh2", "SITCOM-1", "SITCOM-4S", "NP-1", "NP-8-RS"]
DIAGNOSTIC = ["SITCOM-oracle4", "NP-oracle8"]
MAIN_MULTI = ["Fresh2", "SITCOM-4S", "NP-8-RS"]
PAIRWISE = [
    ("Fresh2", "Fresh1"),
    ("SITCOM-4S", "SITCOM-1"),
    ("NP-8-RS", "NP-1"),
    ("NP-8-RS", "Fresh2"),
    ("SITCOM-4S", "Fresh2"),
    ("NP-8-RS", "SITCOM-4S"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bootstrap_mean(values: Iterable[float], reps: int, seed: int) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(reps, len(array)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize_policy(group: pd.DataFrame, reps: int, seed: int) -> dict[str, Any]:
    raw = group.psnr_raw.to_numpy(dtype=np.float64)
    ambiguity = group.psnr_ambiguity_aware.to_numpy(dtype=np.float64)
    cost = group.policy_gpu_seconds.to_numpy(dtype=np.float64)
    mean_ci = bootstrap_mean(raw, reps, seed)
    rate_ci = bootstrap_mean((raw >= 25).astype(np.float64), reps, seed)
    ordered = np.sort(raw)
    return {
        "policy": str(group.policy.iloc[0]),
        "diagnostic_only": bool(group.diagnostic_only.iloc[0]),
        "n_images": len(group),
        "raw_psnr_mean": float(raw.mean()),
        "raw_psnr_mean_ci95_low": mean_ci[0],
        "raw_psnr_mean_ci95_high": mean_ci[1],
        "raw_psnr_median": float(np.median(raw)),
        "raw_psnr_std": float(raw.std(ddof=1)),
        "raw_psnr_min": float(raw.min()),
        "raw_psnr_q01": float(np.quantile(raw, 0.01)),
        "raw_psnr_q05": float(np.quantile(raw, 0.05)),
        "raw_psnr_q10": float(np.quantile(raw, 0.10)),
        "raw_psnr_trim10_mean": float(ordered[10:-10].mean()),
        "raw_psnr_bottom5_mean": float(ordered[:5].mean()),
        "raw_psnr_bottom10_mean": float(ordered[:10].mean()),
        "raw_good20": int((raw >= 20).sum()),
        "raw_good25": int((raw >= 25).sum()),
        "raw_good25_rate": float((raw >= 25).mean()),
        "raw_good25_rate_ci95_low": rate_ci[0],
        "raw_good25_rate_ci95_high": rate_ci[1],
        "raw_good28": int((raw >= 28).sum()),
        "raw_good30": int((raw >= 30).sum()),
        "raw_bad10": int((raw < 10).sum()),
        "raw_bad15": int((raw < 15).sum()),
        "raw_bad20": int((raw < 20).sum()),
        "ambiguity_psnr_mean": float(ambiguity.mean()),
        "ambiguity_good25": int((ambiguity >= 25).sum()),
        "rot180_good25_rescues": int(((raw < 25) & (ambiguity >= 25)).sum()),
        "mean_policy_gpu_seconds": float(cost.mean()),
        "median_policy_gpu_seconds": float(np.median(cost)),
        "sum_policy_gpu_seconds": float(cost.sum()),
    }


def paired_comparison(wide_psnr: pd.DataFrame, wide_good: pd.DataFrame, a: str, b: str, reps: int, seed: int) -> dict[str, Any]:
    delta = (wide_psnr[a] - wide_psnr[b]).to_numpy(dtype=np.float64)
    good_a = wide_good[a].to_numpy(dtype=np.int64)
    good_b = wide_good[b].to_numpy(dtype=np.int64)
    rescues = int(((good_a == 1) & (good_b == 0)).sum())
    harms = int(((good_a == 0) & (good_b == 1)).sum())
    ci = bootstrap_mean(delta, reps, seed)
    try:
        wilcoxon_p = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    discordant = rescues + harms
    mcnemar_p = (
        float(binomtest(min(rescues, harms), n=discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    return {
        "policy_a": a,
        "policy_b": b,
        "mean_raw_psnr_delta_a_minus_b": float(delta.mean()),
        "mean_delta_ci95_low": ci[0],
        "mean_delta_ci95_high": ci[1],
        "median_raw_psnr_delta": float(np.median(delta)),
        "a_higher_psnr_images": int((delta > 1e-9).sum()),
        "ties": int((np.abs(delta) <= 1e-9).sum()),
        "a_lower_psnr_images": int((delta < -1e-9).sum()),
        "paired_wilcoxon_p": wilcoxon_p,
        "a_good25_b_bad25": rescues,
        "a_bad25_b_good25": harms,
        "raw_good25_rate_delta": float((good_a - good_b).mean()),
        "exact_mcnemar_p": mcnemar_p,
    }


def load_policy_jsons(stage_root: Path, family: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((stage_root / family).glob("row*/policy.json"))]


def load_candidate_jsons(stage_root: Path, family: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((stage_root / family).glob("row*/candidates/*/result.json"))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=100_000)
    parser.add_argument("--bootstrap_seed", type=int, default=5401)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    stage_root = run_root / "full"
    out = Path(args.out).resolve()
    temporary = out.with_name(out.name + ".tmp")
    if out.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {out} or {temporary}")

    validation = json.loads((stage_root / "validation.json").read_text())
    if validation.get("status") != "PASS" or not validation.get("full_panel_complete"):
        raise RuntimeError("B22.3 requires a validated complete B22.2 full panel")

    paired = pd.read_csv(stage_root / "paired_rows.csv", dtype={"image_id": str})
    paired.image_id = paired.image_id.str.zfill(5)
    expected = set(EXECUTABLE + DIAGNOSTIC)
    if set(paired.policy) != expected or len(paired) != 800:
        raise RuntimeError("Expected exactly 800 rows over the eight frozen policies")
    if paired.groupby("policy").size().to_dict() != {policy: 100 for policy in expected}:
        raise RuntimeError("Each policy must contain exactly 100 rows")

    temporary.mkdir(parents=True)
    tables = temporary / "tables"
    tables.mkdir()

    headline_rows = [
        summarize_policy(group, args.bootstrap_reps, args.bootstrap_seed)
        for _, group in paired.groupby("policy", sort=False)
    ]
    headline = pd.DataFrame(headline_rows)
    policy_order = {policy: i for i, policy in enumerate(EXECUTABLE + DIAGNOSTIC)}
    headline["_order"] = headline.policy.map(policy_order)
    headline = headline.sort_values("_order").drop(columns="_order")
    headline.to_csv(tables / "headline_policy_summary.csv", index=False)

    wide_psnr = paired.pivot(index="image_id", columns="policy", values="psnr_raw")
    wide_good = paired.pivot(index="image_id", columns="policy", values="good25_raw").astype(int)

    pairwise = pd.DataFrame(
        [paired_comparison(wide_psnr, wide_good, a, b, args.bootstrap_reps, args.bootstrap_seed) for a, b in PAIRWISE]
    )
    pairwise.to_csv(tables / "paired_comparisons.csv", index=False)

    overlap = []
    for a in MAIN_MULTI:
        for b in MAIN_MULTI:
            bad_a = set(wide_good.index[wide_good[a].eq(0)])
            bad_b = set(wide_good.index[wide_good[b].eq(0)])
            overlap.append({"policy_a": a, "policy_b": b, "shared_raw_bad25": len(bad_a & bad_b)})
    pd.DataFrame(overlap).to_csv(tables / "failure_overlap.csv", index=False)

    failure_union_ids = sorted(set().union(*(set(wide_good.index[wide_good[p].eq(0)]) for p in MAIN_MULTI)))
    failure_union = wide_psnr.loc[failure_union_ids, MAIN_MULTI].reset_index()
    failure_union["best_raw_psnr"] = failure_union[MAIN_MULTI].max(axis=1)
    failure_union["best_policy"] = failure_union[MAIN_MULTI].idxmax(axis=1)
    for policy in MAIN_MULTI:
        failure_union[f"{policy}_good25"] = failure_union[policy].ge(25)
    failure_union.to_csv(tables / "main_policy_failure_union.csv", index=False)

    main_wide = wide_psnr[MAIN_MULTI]
    best_main = main_wide.max(axis=1)
    best_policy = main_wide.idxmax(axis=1)
    pd.DataFrame({
        "image_id": best_main.index,
        "cross_method_oracle_raw_psnr": best_main.values,
        "cross_method_oracle_policy": best_policy.values,
        "cross_method_oracle_good25": best_main.ge(25).values,
    }).to_csv(tables / "cross_method_oracle_rows.csv", index=False)

    selector_rows = []
    for family, records, selected_name, oracle_name in (
        ("SITCOM", load_policy_jsons(stage_root, "sitcom"), "SITCOM-4S", "SITCOM-oracle4"),
        ("NP", load_policy_jsons(stage_root, "np"), "NP-8-RS", "NP-oracle8"),
    ):
        selected = np.array([r["policies"][selected_name]["metrics"]["psnr_raw"] for r in records], dtype=float)
        oracle = np.array([r["policies"][oracle_name]["metrics"]["psnr_raw"] for r in records], dtype=float)
        gap = oracle - selected
        selector_rows.append({
            "family": family,
            "selected_policy": selected_name,
            "oracle_policy": oracle_name,
            "exact_oracle_psnr_matches": int(np.isclose(gap, 0, atol=1e-8).sum()),
            "mean_oracle_gap_db": float(gap.mean()),
            "median_oracle_gap_db": float(np.median(gap)),
            "q90_oracle_gap_db": float(np.quantile(gap, .9)),
            "max_oracle_gap_db": float(gap.max()),
            "gap_gt_0_1_db": int((gap > .1).sum()),
            "gap_gt_1_db": int((gap > 1).sum()),
            "gap_gt_5_db": int((gap > 5).sum()),
            "selected_bad25_oracle_good25": int(((selected < 25) & (oracle >= 25)).sum()),
            "selected_good25": int((selected >= 25).sum()),
            "oracle_good25": int((oracle >= 25).sum()),
        })
    pd.DataFrame(selector_rows).to_csv(tables / "selector_oracle_summary.csv", index=False)

    sit_candidates = pd.DataFrame([
        {
            "image_id": str(r["image_id"]).zfill(5),
            "run_index": int(r["run_index"]),
            "psnr_raw": float(r["metrics"]["psnr_raw"]),
            "selector_value": float(r["selector"]["correction_norm"]),
            "reconstruction_s": float(r["timing"]["reconstruction_s"]),
        }
        for r in load_candidate_jsons(stage_root, "sitcom")
    ])
    np_candidates = pd.DataFrame([
        {
            "image_id": str(r["image_id"]).zfill(5),
            "config_tag": str(r["config_tag"]),
            "seed": int(r["seed"]),
            "psnr_raw": float(r["metrics"]["psnr_raw"]),
            "selector_value": float(r["selector_stats"]["selector_post_winner_lf_mse_mean"]),
            "reconstruction_s": float(r["timing"]["reconstruction_s"]),
        }
        for r in load_candidate_jsons(stage_root, "np")
    ])
    candidate_summary = []
    for run_index, group in sit_candidates.groupby("run_index"):
        candidate_summary.append({
            "family": "SITCOM", "candidate": f"run{run_index}", "n": len(group),
            "raw_psnr_mean": group.psnr_raw.mean(), "raw_psnr_median": group.psnr_raw.median(),
            "raw_good25": int(group.psnr_raw.ge(25).sum()), "raw_psnr_min": group.psnr_raw.min(),
            "raw_psnr_q05": group.psnr_raw.quantile(.05), "mean_reconstruction_s": group.reconstruction_s.mean(),
        })
    for (config_tag, seed), group in np_candidates.groupby(["config_tag", "seed"]):
        candidate_summary.append({
            "family": "NP", "candidate": f"{config_tag}/seed{seed}", "n": len(group),
            "raw_psnr_mean": group.psnr_raw.mean(), "raw_psnr_median": group.psnr_raw.median(),
            "raw_good25": int(group.psnr_raw.ge(25).sum()), "raw_psnr_min": group.psnr_raw.min(),
            "raw_psnr_q05": group.psnr_raw.quantile(.05), "mean_reconstruction_s": group.reconstruction_s.mean(),
        })
    pd.DataFrame(candidate_summary).to_csv(tables / "candidate_arm_summary.csv", index=False)

    pd.DataFrame([
        {
            "family": "SITCOM",
            "pooled_spearman_selector_vs_psnr": float(spearmanr(sit_candidates.selector_value, sit_candidates.psnr_raw).statistic),
            "pooled_spearman_p": float(spearmanr(sit_candidates.selector_value, sit_candidates.psnr_raw).pvalue),
        },
        {
            "family": "NP",
            "pooled_spearman_selector_vs_psnr": float(spearmanr(np_candidates.selector_value, np_candidates.psnr_raw).statistic),
            "pooled_spearman_p": float(spearmanr(np_candidates.selector_value, np_candidates.psnr_raw).pvalue),
        },
    ]).to_csv(tables / "selector_pooled_correlation.csv", index=False)

    findings = {
        "schema_version": 1,
        "input_run_root": str(run_root),
        "input_repo_head": json.loads((stage_root / "manifest.json").read_text())["identities"]["repo_head"],
        "bootstrap": {"seed": args.bootstrap_seed, "repetitions": args.bootstrap_reps, "unit": "image"},
        "headline": {
            "fresh2_raw_good25": int(wide_good["Fresh2"].sum()),
            "sitcom4s_raw_good25": int(wide_good["SITCOM-4S"].sum()),
            "np8_raw_good25": int(wide_good["NP-8-RS"].sum()),
            "fresh2_raw_mean": float(wide_psnr["Fresh2"].mean()),
            "sitcom4s_raw_mean": float(wide_psnr["SITCOM-4S"].mean()),
            "np8_raw_mean": float(wide_psnr["NP-8-RS"].mean()),
            "cross_method_oracle_good25": int(best_main.ge(25).sum()),
            "cross_method_oracle_failures": sorted(best_main.index[best_main.lt(25)]),
            "main_policy_failure_union_count": len(failure_union_ids),
        },
        "input_hashes": {
            name: sha256_file(stage_root / name)
            for name in ("validation.json", "paired_rows.csv", "candidate_selector_audit.csv", "summary.csv")
        },
    }
    write_json(temporary / "analysis_manifest.json", findings)

    summary_lookup = headline.set_index("policy")
    comparison_lookup = pairwise.set_index(["policy_a", "policy_b"])
    lines = [
        "# B22.3 scientific analysis of the 100-image fixed-baseline panel",
        "",
        "## Validation and scope",
        "",
        "- Input panel: validated B22.2 full panel, 100 paired images.",
        "- Primary metric: raw RGB PSNR.",
        "- Reliability threshold: raw PSNR >= 25 dB.",
        "- Rot180-aware results are auxiliary and ground-truth-assisted.",
        "- SITCOM/NP oracle rows and the cross-method oracle are diagnostic only.",
        "",
        "## Headline executable results",
        "",
        "| Policy | Mean | Median | Min | q05 | good25 | bad20 | Mean GPU-s/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in EXECUTABLE:
        r = summary_lookup.loc[policy]
        lines.append(
            f"| {policy} | {r.raw_psnr_mean:.3f} | {r.raw_psnr_median:.3f} | {r.raw_psnr_min:.3f} | "
            f"{r.raw_psnr_q05:.3f} | {int(r.raw_good25)}/100 | {int(r.raw_bad20)} | {r.mean_policy_gpu_seconds:.1f} |"
        )
    lines += [
        "",
        "## Primary scientific interpretation",
        "",
        "1. **Fresh2 remains the central-quality leader.** It has the highest executable median and beats NP-8-RS on 91/100 images, but retains a catastrophic lower tail.",
        "2. **NP-8-RS is the strongest executable reliability policy.** It reaches 95/100 raw good25, removes all sub-10 dB failures, and has the best executable q05/bottom-tail profile, at the highest compute cost.",
        "3. **SITCOM-4S is the lower-cost stabilizer.** It reaches 93/100 raw good25 at about 196 GPU-s/image, but its central PSNR is substantially lower than Fresh2 and NP-8-RS.",
        "4. **The policies are complementary rather than rank-equivalent.** The best-of-{Fresh2,SITCOM-4S,NP-8-RS} diagnostic oracle reaches 99/100; only image `65003` is below 25 dB for all three.",
        "5. **Cross-method selection is the unresolved scientific opportunity.** Fresh2 wins raw PSNR on 91 images and NP-8-RS on 9; NP's few wins include large rescues of Fresh2 catastrophes. No clean-free cross-method selector is authorized by this panel.",
        "",
        "## Paired comparisons",
        "",
        "| A vs B | Mean delta | 95% image-bootstrap CI | A higher / tie / lower | A rescues / harms at 25 |",
        "|---|---:|---:|---:|---:|",
    ]
    for a, b in PAIRWISE:
        r = comparison_lookup.loc[(a, b)]
        lines.append(
            f"| {a} vs {b} | {r.mean_raw_psnr_delta_a_minus_b:+.3f} | "
            f"[{r.mean_delta_ci95_low:+.3f}, {r.mean_delta_ci95_high:+.3f}] | "
            f"{int(r.a_higher_psnr_images)} / {int(r.ties)} / {int(r.a_lower_psnr_images)} | "
            f"{int(r.a_good25_b_bad25)} / {int(r.a_bad25_b_good25)} |"
        )
    lines += [
        "",
        "## Selector versus oracle",
        "",
        "- SITCOM-4S: 93/100 selected good25 versus 95/100 oracle4; two threshold-level selector misses, including one large catastrophic selector miss (`60140`).",
        "- NP-8-RS: 95/100 selected good25 versus 96/100 oracle8; one threshold-level selector miss (`65269`).",
        "- The small average oracle gaps hide isolated important misses; visual/feature audit should focus on those cases rather than aggregate retuning.",
        "",
        "## Next authorized work",
        "",
        "1. Generate the zero-GPU 16-image failure/complementarity atlas from preserved PAC PNGs.",
        "2. Manually classify image `65003`, the Fresh2 catastrophic failures rescued by NP/SITCOM, and the three selector-miss cases.",
        "3. Report the fixed policies as evaluated. Do not tune a cross-method selector on this 100-image estimation panel.",
        "4. Any future selector proposal must be developed on separate historical/development data and evaluated prospectively on new locked images.",
        "",
        "All machine-readable tables are under `tables/`.",
        "",
    ]
    (temporary / "B22_3_SCIENTIFIC_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(out)
    print(json.dumps({"status": "PASS", "out": str(out), "failure_union": len(failure_union_ids), "cross_method_oracle_good25": int(best_main.ge(25).sum())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
