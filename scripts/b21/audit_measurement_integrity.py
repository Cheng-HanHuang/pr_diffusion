#!/usr/bin/env python3
"""B21.0 measurement-integrity audit helper.

This script is read-only with respect to existing B19/B20 outputs.  It inventories
locked measurement payloads, checks whether measurement seeds are distinct per
image, and optionally audits per-case PSNR variation across measurement seeds.

It writes machine-readable CSV/JSON artifacts plus a markdown report draft.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MEAS_RE = re.compile(
    r"ffhq(?P<image_id>\d{5}).*?phase.*?noise(?P<noise>\d+).*?meas(?P<meas_seed>\d+).*?\.pt$"
)


IMAGE_COLS = ["image_id", "image", "id", "ffhq_id", "image_basename", "basename"]
SEED_COLS = ["meas_seed", "measurement_seed", "measurement_noise_seed", "seed_meas", "meas"]
RUN_COLS = ["run_index", "run", "run_id", "candidate_index", "candidate", "selected_run", "run_seed"]
PSNR_COLS = ["final_psnr", "selected_psnr", "psnr", "sample_psnr", "best_psnr", "oracle_psnr"]
POLICY_COLS = ["policy", "policy_name", "selection_policy", "selection_method", "method"]


def split_items(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for item in split_items(text):
        if "-" in item:
            a, b = item.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def find_col(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_to_actual = {x.lower(): x for x in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_to_actual:
            return lower_to_actual[cand.lower()]
    return None


def parse_image_id(value: object) -> str:
    text = str(value)
    m = re.search(r"(\d{5})", text)
    return m.group(1) if m else text


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def expand_globs(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        expanded = os.path.expandvars(os.path.expanduser(pattern))
        matches = glob.glob(expanded, recursive=True)
        paths.extend(Path(x) for x in matches)
    return sorted(set(p for p in paths if p.exists()))


def inventory_measurements(roots: Sequence[str], patterns: Sequence[str], seeds: Sequence[int]) -> Tuple[List[Path], List[Dict[str, object]]]:
    files: List[Path] = []
    for root_text in roots:
        root = Path(os.path.expandvars(os.path.expanduser(root_text)))
        if not root.exists():
            continue
        for pattern in patterns:
            files.extend(root.glob(pattern))
    files = sorted(set(p for p in files if p.is_file()))

    rows: List[Dict[str, object]] = []
    seed_set = set(int(s) for s in seeds)
    for path in files:
        m = MEAS_RE.search(path.name)
        if not m:
            continue
        meas_seed = int(m.group("meas_seed"))
        if seed_set and meas_seed not in seed_set:
            continue
        image_id = m.group("image_id")
        row: Dict[str, object] = {
            "image_id": image_id,
            "meas_seed": meas_seed,
            "path": str(path),
            "filename": path.name,
            "load_error": "",
        }
        try:
            import torch

            digest = sha256_file(path)
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict):
                tensor = payload.get("measurement", None)
                payload_keys = ",".join(str(k) for k in payload.keys())
            else:
                tensor = payload
                payload_keys = type(payload).__name__
            if tensor is None:
                raise ValueError("payload has no measurement tensor")
            t = tensor.detach().cpu()
            flat = t.reshape(-1)
            stat = flat.abs().float() if getattr(flat, "is_complex", lambda: False)() else flat.float()
            first4 = [float(x) for x in stat[:4].tolist()]
            row.update(
                {
                    "sha256": digest,
                    "payload_keys": payload_keys,
                    "tensor_shape": "x".join(str(x) for x in tuple(t.shape)),
                    "tensor_dtype": str(t.dtype),
                    "tensor_is_complex": bool(getattr(t, "is_complex", lambda: False)()),
                    "tensor_mean": float(stat.mean().item()),
                    "tensor_std": float(stat.std(unbiased=False).item()),
                    "first4": json.dumps(first4),
                }
            )
        except Exception as exc:  # keep auditing the other files
            row.update({"load_error": repr(exc), "sha256": ""})
        rows.append(row)
    return files, rows


def duplicate_rows(inventory: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_image_sha: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in inventory:
        sha = str(row.get("sha256", ""))
        if sha:
            by_image_sha[(str(row["image_id"]), sha)].append(row)
    out: List[Dict[str, object]] = []
    for (image_id, sha), rows in sorted(by_image_sha.items()):
        seeds = sorted(int(r["meas_seed"]) for r in rows)
        if len(seeds) > 1:
            out.append(
                {
                    "image_id": image_id,
                    "sha256": sha,
                    "duplicate_seed_count": len(seeds),
                    "duplicate_seeds": ",".join(map(str, seeds)),
                    "paths": " | ".join(str(r["path"]) for r in rows),
                }
            )
    return out


def audit_case_csvs(case_csvs: Sequence[Path], policy_filter: str = "") -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    raw_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[Tuple[str, str], ...], List[Tuple[str, float, str]]] = defaultdict(list)

    for path in case_csvs:
        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                fields = list(reader.fieldnames)
                image_col = find_col(fields, IMAGE_COLS)
                seed_col = find_col(fields, SEED_COLS)
                run_col = find_col(fields, RUN_COLS)
                psnr_col = find_col(fields, PSNR_COLS)
                policy_col = find_col(fields, POLICY_COLS)
                if not (image_col and seed_col and psnr_col):
                    continue
                for row in reader:
                    if policy_filter and policy_col and str(row.get(policy_col, "")) != policy_filter:
                        continue
                    psnr = safe_float(row.get(psnr_col, "nan"))
                    if not math.isfinite(psnr):
                        continue
                    image_id = parse_image_id(row.get(image_col, ""))
                    meas_seed = str(row.get(seed_col, ""))
                    run_value = str(row.get(run_col, "selected")) if run_col else "selected"
                    policy_value = str(row.get(policy_col, "all")) if policy_col else "all"
                    group_parts = [("csv_policy", policy_value), ("image_id", image_id), ("run", run_value)]
                    key = tuple(group_parts)
                    grouped[key].append((meas_seed, psnr, str(path)))
                    raw_rows.append(
                        {
                            "csv_path": str(path),
                            "policy": policy_value,
                            "image_id": image_id,
                            "meas_seed": meas_seed,
                            "run": run_value,
                            "psnr": psnr,
                            "source_psnr_col": psnr_col,
                        }
                    )
        except Exception as exc:
            raw_rows.append({"csv_path": str(path), "error": repr(exc)})

    summary: List[Dict[str, object]] = []
    for key, vals in sorted(grouped.items()):
        psnrs = [v[1] for v in vals]
        seeds = [v[0] for v in vals]
        paths = sorted(set(v[2] for v in vals))
        sd = statistics.pstdev(psnrs) if len(psnrs) > 1 else 0.0
        row = {k: v for k, v in key}
        row.update(
            {
                "n_rows": len(vals),
                "n_unique_meas_seeds": len(set(seeds)),
                "meas_seeds": ",".join(sorted(set(seeds))),
                "psnr_mean": statistics.fmean(psnrs),
                "psnr_min": min(psnrs),
                "psnr_max": max(psnrs),
                "psnr_sd_population": sd,
                "sd_lt_0p01": bool(sd < 0.01 and len(set(seeds)) > 1),
                "csv_paths": " | ".join(paths),
            }
        )
        summary.append(row)
    return raw_rows, summary


def collect_runner_snippets(roots: Sequence[str], max_lines: int = 500) -> str:
    needles = ["measurement", "MEAS", "load_measurement", "measurement_path", "B19_20", "5001", "5010"]
    suffixes = {".py", ".sh", ".md", ".txt"}
    lines: List[str] = []
    for root_text in roots:
        root = Path(os.path.expandvars(os.path.expanduser(root_text)))
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes]
        for path in sorted(candidates):
            try:
                text_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for i, line in enumerate(text_lines, start=1):
                if any(n in line for n in needles):
                    lines.append(f"{path}:{i}: {line}")
                    if len(lines) >= max_lines:
                        return "\n".join(lines) + "\n"
    return "\n".join(lines) + ("\n" if lines else "")


def render_report(
    outdir: Path,
    inventory: Sequence[Dict[str, object]],
    dups: Sequence[Dict[str, object]],
    case_summary: Sequence[Dict[str, object]],
    seeds: Sequence[int],
    expected_images: Optional[int],
    case_csvs: Sequence[Path],
) -> str:
    found_pairs = {(str(r.get("image_id")), int(r.get("meas_seed"))) for r in inventory if not r.get("load_error")}
    found_images = sorted({x[0] for x in found_pairs})
    expected_pairs = (expected_images or len(found_images)) * len(seeds) if seeds else None
    loaded = sum(1 for r in inventory if not r.get("load_error"))
    load_errors = sum(1 for r in inventory if r.get("load_error"))
    duplicate_image_count = len({str(r["image_id"]) for r in dups})

    meas_verdict = "PASS"
    if expected_pairs is not None and loaded < expected_pairs:
        meas_verdict = "INCOMPLETE"
    if duplicate_image_count > 0:
        meas_verdict = "FAIL"
    if loaded == 0:
        meas_verdict = "INCOMPLETE"

    eligible = [r for r in case_summary if int(r.get("n_unique_meas_seeds", 0)) > 1]
    low_sd = [r for r in eligible if str(r.get("sd_lt_0p01", False)) == "True" or r.get("sd_lt_0p01") is True]
    low_frac = (len(low_sd) / len(eligible)) if eligible else math.nan
    psnr_verdict = "INCOMPLETE"
    if eligible:
        psnr_verdict = "PASS" if low_frac < 0.10 else "FAIL"

    if meas_verdict == "PASS" and psnr_verdict == "PASS":
        g0 = "B19.20 is n=1000 (distinct measurements, PSNR varies)."
    elif meas_verdict == "FAIL" or psnr_verdict == "FAIL":
        g0 = "B19.20 is n=100 (evidence: duplicate measurements and/or degenerate PSNR variation)."
    else:
        g0 = "G0 incomplete: missing measurement files or per-case CSVs; do not trust n=1000 interpretation yet."

    lines = [
        "# B21.0 measurement integrity audit",
        "",
        "Status: generated by `scripts/b21/audit_measurement_integrity.py`.",
        "",
        "## G0 verdict",
        "",
        f"**{g0}**",
        "",
        "## Inputs found",
        "",
        f"- Requested measurement seeds: `{','.join(map(str, seeds))}`",
        f"- Expected image count: `{expected_images if expected_images is not None else 'inferred'}`",
        f"- Measurement payloads loaded: `{loaded}`",
        f"- Measurement load errors: `{load_errors}`",
        f"- Distinct images with payloads: `{len(found_images)}`",
        f"- Expected image-seed pairs: `{expected_pairs if expected_pairs is not None else 'unknown'}`",
        f"- Per-case CSVs found: `{len(case_csvs)}`",
        "",
        "## Measurement distinctness",
        "",
        f"- Measurement verdict: `{meas_verdict}`",
        f"- Images with duplicate SHA across measurement seeds: `{duplicate_image_count}`",
        f"- Duplicate rows: `{len(dups)}`",
        "",
        "Artifacts:",
        "",
        "```text",
        str(outdir / "measurement_inventory.csv"),
        str(outdir / "measurement_duplicate_sha.csv"),
        "```",
        "",
        "## PSNR seed-variation check",
        "",
        f"- PSNR verdict: `{psnr_verdict}`",
        f"- Eligible groups with >1 measurement seed: `{len(eligible)}`",
        f"- Groups with sd < 0.01 dB: `{len(low_sd)}`",
        f"- Fraction sd < 0.01 dB: `{low_frac if math.isfinite(low_frac) else 'nan'}`",
        "",
        "Artifacts:",
        "",
        "```text",
        str(outdir / "case_rows_extracted.csv"),
        str(outdir / "case_psnr_seed_variation.csv"),
        "```",
        "",
        "## Runner / loader root-cause snippets",
        "",
        "See:",
        "",
        "```text",
        str(outdir / "runner_measurement_path_snippets.txt"),
        "```",
        "",
        "## Notes for executor",
        "",
        "- If this report says `INCOMPLETE`, inspect whether the B19.20 per-case CSV path or measurement root was missing and rerun with explicit paths.",
        "- If duplicate SHA rows exist for the same image across seeds, do not cite B19.20 as n=1000.",
        "- If PSNR seed variation is degenerate for >=10% of groups, do not cite B19.20 as n=1000 even if payload files look distinct; audit the loader path.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="B21.0 measurement integrity audit")
    ap.add_argument("--measurement_roots", default="/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver/measurements")
    ap.add_argument("--measurement_patterns", default="ffhq*_phase_noise*_meas*.pt,**/ffhq*_phase_noise*_meas*.pt")
    ap.add_argument("--seeds", default="5001-5010")
    ap.add_argument("--expected_images", type=int, default=100)
    ap.add_argument("--case_csv_globs", default="/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver/B19_20*/**/*.csv")
    ap.add_argument("--policy_filter", default="")
    ap.add_argument("--runner_search_roots", default="scripts,docs/b19")
    ap.add_argument("--outdir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_0_measurement_integrity_audit")
    ap.add_argument("--report_path", default="docs/b21/b21_0_measurement_integrity_audit.md")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    files, inventory = inventory_measurements(split_items(args.measurement_roots), split_items(args.measurement_patterns), seeds)
    dups = duplicate_rows(inventory)
    write_csv(outdir / "measurement_inventory.csv", inventory)
    write_csv(outdir / "measurement_duplicate_sha.csv", dups)

    case_csvs = expand_globs(split_items(args.case_csv_globs))
    raw_case_rows, case_summary = audit_case_csvs(case_csvs, policy_filter=args.policy_filter)
    write_csv(outdir / "case_rows_extracted.csv", raw_case_rows)
    write_csv(outdir / "case_psnr_seed_variation.csv", case_summary)

    snippets = collect_runner_snippets(split_items(args.runner_search_roots))
    (outdir / "runner_measurement_path_snippets.txt").write_text(snippets, encoding="utf-8")

    report = render_report(outdir, inventory, dups, case_summary, seeds, args.expected_images, case_csvs)
    out_report = Path(args.report_path)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report + "\n", encoding="utf-8")
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "n_measurement_files_matching": len(files),
                "n_inventory_rows": len(inventory),
                "n_duplicate_sha_rows": len(dups),
                "n_case_csvs": len(case_csvs),
                "n_case_groups": len(case_summary),
                "report_path": str(out_report),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {outdir / 'measurement_inventory.csv'}")
    print(f"[write] {outdir / 'measurement_duplicate_sha.csv'}")
    print(f"[write] {outdir / 'case_psnr_seed_variation.csv'}")
    print(f"[write] {out_report}")


if __name__ == "__main__":
    main()
