#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


CANDIDATES = ("source_full", "fresh_extra", "branch_a", "branch_b")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_numeric(row: pd.Series, names: Iterable[str]) -> float:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(row[name], errors="coerce")
            if pd.notna(value):
                return float(value)
    return float("nan")


def read_metric(path: Path) -> tuple[float, float, str]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty metric CSV: {path}")
    row = frame.iloc[0]
    exact_loss = pick_numeric(
        row,
        ["exact_operator_loss", "operator_loss", "measurement_loss", "loss"],
    )
    psnr = pick_numeric(
        row,
        [
            "psnr_recomputed_from_png",
            "psnr_metrics_json",
            "final_psnr",
            "psnr",
            "PSNR",
        ],
    )
    sample_path = ""
    for name in ("sample_path", "selected_sample_path", "image_path"):
        if name in row.index and pd.notna(row[name]):
            sample_path = str(row[name])
            break
    if not math.isfinite(exact_loss):
        raise ValueError(f"no finite exact loss in {path}; columns={list(frame.columns)}")
    if not math.isfinite(psnr):
        raise ValueError(f"no finite PSNR in {path}; columns={list(frame.columns)}")
    return exact_loss, psnr, sample_path


def load_timings(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def exact_mcnemar_p(branch_only: int, fresh_only: int) -> float:
    n = int(branch_only + fresh_only)
    if n == 0:
        return 1.0
    k = min(int(branch_only), int(fresh_only))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def selected_by_exact(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(rows, key=lambda row: float(row["exact_operator_loss"]))


def summarize_cases(frame: pd.DataFrame, label: str) -> dict[str, object]:
    n = int(len(frame))
    branch_good = int(frame["branch_any_good"].sum())
    fresh_good = int(frame["fresh_any_good"].sum())
    branch_only = int(frame["branch_only_win"].sum())
    fresh_only = int(frame["fresh_only_win"].sum())
    source_good = int(frame["source_good"].sum())
    both_good = int(((frame["branch_any_good"] == 1) & (frame["fresh_any_good"] == 1)).sum())
    both_bad = int(((frame["branch_any_good"] == 0) & (frame["fresh_any_good"] == 0)).sum())

    fresh_wall = float(frame["fresh_policy_wall_seconds"].sum())
    branch_wall = float(frame["branch_policy_wall_seconds"].sum())
    wall_ratio = branch_wall / fresh_wall if fresh_wall > 0 else float("nan")

    return {
        "group": label,
        "n_cases": n,
        "source_any_good": source_good,
        "fresh_any_good": fresh_good,
        "branch_any_good": branch_good,
        "fresh_good_rate": fresh_good / n if n else float("nan"),
        "branch_good_rate": branch_good / n if n else float("nan"),
        "branch_minus_fresh_cases": branch_good - fresh_good,
        "branch_minus_fresh_rate": (branch_good - fresh_good) / n if n else float("nan"),
        "branch_only_wins": branch_only,
        "fresh_only_wins": fresh_only,
        "both_good": both_good,
        "both_bad": both_bad,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(branch_only, fresh_only),
        "fresh_exact_selected_good": int(frame["fresh_selected_good25"].sum()),
        "branch_exact_selected_good": int(frame["branch_selected_good25"].sum()),
        "fresh_best_psnr_mean": float(frame["fresh_best_psnr"].mean()),
        "branch_best_psnr_mean": float(frame["branch_best_psnr"].mean()),
        "branch_minus_fresh_best_psnr_mean": float(
            (frame["branch_best_psnr"] - frame["fresh_best_psnr"]).mean()
        ),
        "fresh_policy_wall_seconds": fresh_wall,
        "branch_policy_wall_seconds": branch_wall,
        "branch_over_fresh_wall_ratio": wall_ratio,
        "mean_branch_policy_unique_hashes": float(frame["branch_policy_unique_hashes"].mean()),
    }


def markdown_summary(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| group | n | fresh good | branch good | net | branch-only | fresh-only | McNemar p | wall ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {group} | {n} | {fresh} | {branch} | {net:+d} | {bo} | {fo} | {p:.6g} | {ratio:.3f} |".format(
                group=row["group"],
                n=int(row["n_cases"]),
                fresh=int(row["fresh_any_good"]),
                branch=int(row["branch_any_good"]),
                net=int(row["branch_minus_fresh_cases"]),
                bo=int(row["branch_only_wins"]),
                fo=int(row["fresh_only_wins"]),
                p=float(row["mcnemar_exact_two_sided_p"]),
                ratio=float(row["branch_over_fresh_wall_ratio"]),
            )
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ann-steps", type=int, default=400)
    parser.add_argument("--split-step", type=int, default=200)
    args = parser.parse_args()

    out = args.out.resolve()
    repo = args.repo.resolve()
    manifest_path = out / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path, sep="\t")
    expected_columns = {
        "job_id",
        "image_id",
        "parent_seed",
        "gpu",
        "source_seed",
        "fresh_seed",
        "branch_a_seed",
        "branch_b_seed",
    }
    missing_columns = expected_columns.difference(manifest.columns)
    if missing_columns:
        raise KeyError(f"manifest missing columns: {sorted(missing_columns)}")

    candidate_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    missing: list[str] = []

    for manifest_row in manifest.to_dict(orient="records"):
        image = f"{int(manifest_row['image_id']):05d}"
        parent = int(manifest_row["parent_seed"])
        case_dir = out / "cases" / f"{image}_parent{parent}"
        timings = load_timings(case_dir / "timings.tsv")
        seeds = {
            "source_full": int(manifest_row["source_seed"]),
            "fresh_extra": int(manifest_row["fresh_seed"]),
            "branch_a": int(manifest_row["branch_a_seed"]),
            "branch_b": int(manifest_row["branch_b_seed"]),
        }

        local_rows: dict[str, dict[str, object]] = {}
        for candidate in CANDIDATES:
            metric_path = case_dir / "metrics" / f"{candidate}.csv"
            sample_path = (
                case_dir
                / "daps_results"
                / candidate
                / "samples"
                / "00000_run0000.png"
            )
            if not metric_path.exists() or not sample_path.exists():
                missing.append(
                    f"image={image} parent={parent} candidate={candidate} "
                    f"metric={metric_path.exists()} sample={sample_path.exists()}"
                )
                continue
            exact_loss, psnr, metric_sample = read_metric(metric_path)
            row = {
                "job_id": int(manifest_row["job_id"]),
                "image_id": image,
                "parent_seed": parent,
                "candidate": candidate,
                "candidate_seed": seeds[candidate],
                "exact_operator_loss": exact_loss,
                "psnr": psnr,
                "good25": int(psnr >= 25.0),
                "sha256": sha256(sample_path),
                "sample_path": str(sample_path),
                "metric_sample_path": metric_sample,
                "metric_path": str(metric_path),
                "wall_seconds": float(timings.get(candidate, float("nan"))),
            }
            candidate_rows.append(row)
            local_rows[candidate] = row

        if len(local_rows) != len(CANDIDATES):
            continue

        source = local_rows["source_full"]
        fresh = local_rows["fresh_extra"]
        branch_a = local_rows["branch_a"]
        branch_b = local_rows["branch_b"]
        fresh_pool = [source, fresh]
        branch_pool = [source, branch_a, branch_b]
        fresh_selected = selected_by_exact(fresh_pool)
        branch_selected = selected_by_exact(branch_pool)

        source_good = int(source["good25"])
        fresh_any_good = int(any(int(row["good25"]) for row in fresh_pool))
        branch_any_good = int(any(int(row["good25"]) for row in branch_pool))

        fresh_wall = sum(float(row["wall_seconds"]) for row in fresh_pool)
        branch_wall = sum(float(row["wall_seconds"]) for row in branch_pool)

        case_rows.append(
            {
                "job_id": int(manifest_row["job_id"]),
                "image_id": image,
                "parent_seed": parent,
                "source_seed": seeds["source_full"],
                "fresh_seed": seeds["fresh_extra"],
                "branch_a_seed": seeds["branch_a"],
                "branch_b_seed": seeds["branch_b"],
                "source_psnr": float(source["psnr"]),
                "fresh_extra_psnr": float(fresh["psnr"]),
                "branch_a_psnr": float(branch_a["psnr"]),
                "branch_b_psnr": float(branch_b["psnr"]),
                "source_good": source_good,
                "fresh_any_good": fresh_any_good,
                "branch_any_good": branch_any_good,
                "branch_only_win": int(branch_any_good == 1 and fresh_any_good == 0),
                "fresh_only_win": int(fresh_any_good == 1 and branch_any_good == 0),
                "fresh_best_psnr": max(float(row["psnr"]) for row in fresh_pool),
                "branch_best_psnr": max(float(row["psnr"]) for row in branch_pool),
                "fresh_num_good25": sum(int(row["good25"]) for row in fresh_pool),
                "branch_num_good25": sum(int(row["good25"]) for row in branch_pool),
                "fresh_selected_candidate": str(fresh_selected["candidate"]),
                "fresh_selected_exact_loss": float(fresh_selected["exact_operator_loss"]),
                "fresh_selected_psnr": float(fresh_selected["psnr"]),
                "fresh_selected_good25": int(fresh_selected["good25"]),
                "branch_selected_candidate": str(branch_selected["candidate"]),
                "branch_selected_exact_loss": float(branch_selected["exact_operator_loss"]),
                "branch_selected_psnr": float(branch_selected["psnr"]),
                "branch_selected_good25": int(branch_selected["good25"]),
                "fresh_policy_unique_hashes": len({str(row["sha256"]) for row in fresh_pool}),
                "branch_policy_unique_hashes": len({str(row["sha256"]) for row in branch_pool}),
                "continuation_unique_hashes": len(
                    {str(branch_a["sha256"]), str(branch_b["sha256"])}
                ),
                "source_wall_seconds": float(source["wall_seconds"]),
                "fresh_extra_wall_seconds": float(fresh["wall_seconds"]),
                "branch_a_wall_seconds": float(branch_a["wall_seconds"]),
                "branch_b_wall_seconds": float(branch_b["wall_seconds"]),
                "fresh_policy_wall_seconds": fresh_wall,
                "branch_policy_wall_seconds": branch_wall,
                "fresh_policy_step_cost": int(args.ann_steps * 2),
                "branch_policy_step_cost": int(
                    args.ann_steps + 2 * (args.ann_steps - args.split_step)
                ),
            }
        )

    if missing:
        print(f"[missing] {len(missing)} candidate artifacts")
        for item in missing[:40]:
            print("[missing]", item)
        raise RuntimeError("pilot artifacts incomplete")

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["image_id", "parent_seed", "candidate"]
    )
    cases = pd.DataFrame(case_rows).sort_values(["image_id", "parent_seed"])
    if len(cases) != len(manifest):
        raise RuntimeError(f"case count mismatch: cases={len(cases)} manifest={len(manifest)}")

    candidate_path = out / "candidate_rows.csv"
    case_path = out / "case_rows.csv"
    candidates.to_csv(candidate_path, index=False)
    cases.to_csv(case_path, index=False)

    summaries: list[dict[str, object]] = []
    for image, group in cases.groupby("image_id", sort=True):
        summaries.append(summarize_cases(group, str(image)))
    overall = summarize_cases(cases, "ALL")
    summaries.append(overall)
    summary_frame = pd.DataFrame(summaries)
    summary_path = out / "summary_by_image.csv"
    summary_frame.to_csv(summary_path, index=False)

    fresh_losses_by_image = (
        cases.groupby("image_id")["fresh_only_win"].sum().astype(int).to_dict()
    )
    max_fresh_only_on_one_image = max(fresh_losses_by_image.values(), default=0)
    pilot_promotion_pass = bool(
        int(overall["branch_minus_fresh_cases"]) >= 2
        and max_fresh_only_on_one_image <= 1
        and float(overall["branch_over_fresh_wall_ratio"]) <= 1.25
    )
    registered_gate_pass = bool(
        float(overall["branch_minus_fresh_rate"]) >= 0.05
        or float(overall["mcnemar_exact_two_sided_p"]) < 0.05
    )

    verdict = {
        "expected_cases": int(len(manifest)),
        "complete_cases": int(len(cases)),
        "expected_candidates": int(len(manifest) * len(CANDIDATES)),
        "complete_candidates": int(len(candidates)),
        "ann_steps": int(args.ann_steps),
        "split_step": int(args.split_step),
        "fresh_policy_step_cost": int(args.ann_steps * 2),
        "branch_policy_step_cost": int(
            args.ann_steps + 2 * (args.ann_steps - args.split_step)
        ),
        "overall": overall,
        "fresh_only_wins_by_image": {
            str(key): int(value) for key, value in fresh_losses_by_image.items()
        },
        "max_fresh_only_wins_on_one_image": int(max_fresh_only_on_one_image),
        "mini_pilot_gate": {
            "requires_net_case_gain_at_least": 2,
            "requires_max_fresh_only_wins_per_image_at_most": 1,
            "requires_wall_ratio_at_most": 1.25,
            "pass": pilot_promotion_pass,
        },
        "registered_large_validation_gate": {
            "requires_branch_rate_gain_at_least": 0.05,
            "or_mcnemar_p_below": 0.05,
            "pass_on_this_mini_pilot": registered_gate_pass,
        },
    }
    verdict_path = out / "pilot_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    best_arm_counts = Counter()
    for _, row in cases.iterrows():
        values = {
            "source_full": float(row["source_psnr"]),
            "fresh_extra": float(row["fresh_extra_psnr"]),
            "branch_a": float(row["branch_a_psnr"]),
            "branch_b": float(row["branch_b_psnr"]),
        }
        best_arm_counts[max(values, key=values.get)] += 1

    report_path = repo / "docs/b21/b21_3_branch_vs_fresh_pilot.md"
    report_lines = [
        "# B21.3 equal-cost branch-vs-fresh mini-pilot",
        "",
        f"- cases: `{len(cases)}`",
        f"- images: `{', '.join(sorted(cases['image_id'].unique()))}`",
        f"- parent seeds per image: `{cases.groupby('image_id').size().iloc[0]}`",
        f"- fresh policy cost: `{verdict['fresh_policy_step_cost']}` annealing transitions",
        f"- branch policy cost: `{verdict['branch_policy_step_cost']}` annealing transitions",
        f"- mini-pilot promotion gate: **{pilot_promotion_pass}**",
        "",
        "Both policies share `source_full`. Fresh2 adds one independent ann400 run; Branch3 adds two continuations from step 200.",
        "",
        "## Oracle any-good results",
        "",
        *markdown_summary(summary_frame),
        "",
        "## Secondary clean-free exact-loss selection",
        "",
        f"- Fresh2 selected-good cases: `{overall['fresh_exact_selected_good']}/{overall['n_cases']}`",
        f"- Branch3 selected-good cases: `{overall['branch_exact_selected_good']}/{overall['n_cases']}`",
        "",
        "## Best-candidate arm counts",
        "",
    ]
    for name in CANDIDATES:
        report_lines.append(f"- `{name}`: `{best_arm_counts.get(name, 0)}`")
    report_lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- net successful-case gain >= 2: `{int(overall['branch_minus_fresh_cases']) >= 2}`",
            f"- no more than one Fresh2-only win on any image: `{max_fresh_only_on_one_image <= 1}`",
            f"- branch/fresh wall ratio <= 1.25: `{float(overall['branch_over_fresh_wall_ratio']) <= 1.25}`",
            f"- registered +5 pp or McNemar p<0.05 criterion on this mini-pilot: `{registered_gate_pass}`",
            "",
            f"Artifacts: `{out}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")

    print("[write]", candidate_path)
    print("[write]", case_path)
    print("[write]", summary_path)
    print("[write]", verdict_path)
    print("[write]", report_path)
    print()
    print(summary_frame.to_string(index=False))
    print()
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
