#!/usr/bin/env python3
"""B21.2 selector-v2 first pass: symmetry-aware clean-free scoring.

This script consumes the B21.2 candidate-recovery CSV and computes measurement
residuals for each candidate and its rot180 version.  It does not use PSNR or
ground truth for the selector; PSNR columns, when present, are diagnostics only.

The exact scale of the historical selector loss depends on local DAPS task
normalization.  To avoid silently mismatching it, the script calibrates over
simple image/measurement conventions against the recorded
`selector_sqrt_loss_over_y_norm` column, then uses the best-matching convention
for original-vs-rot180 comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ffloat(x: object, default: float = math.nan) -> float:
    try:
        if x is None or str(x) == "":
            return default
        return float(x)
    except Exception:
        return default


def first_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("measurement", "y", "observed", "data"):
            if key in obj:
                got = first_tensor(obj[key])
                if got is not None:
                    return got
        for v in obj.values():
            got = first_tensor(v)
            if got is not None:
                return got
    if isinstance(obj, (list, tuple)):
        for v in obj:
            got = first_tensor(v)
            if got is not None:
                return got
    return None


def load_measurement(path: str, target_mode: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    y = first_tensor(obj)
    if y is None:
        raise RuntimeError(f"could not find tensor in measurement payload {path}")
    y = y.detach().cpu().float()
    if y.dim() == 2:
        y = y[None, None]
    elif y.dim() == 3:
        y = y[None]
    if target_mode == "abs":
        y = y.abs()
    elif target_mode == "raw":
        pass
    else:
        raise ValueError(target_mode)
    return y


def load_png(path: str, image_mode: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    if image_mode == "zero_one":
        return x
    if image_mode == "minus_one_one":
        return x * 2.0 - 1.0
    raise ValueError(image_mode)


def pad_center(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if h > out_h or w > out_w:
        raise ValueError(f"image shape {h}x{w} larger than measurement grid {out_h}x{out_w}")
    y = torch.zeros((b, c, out_h, out_w), dtype=x.dtype)
    y[:, :, (out_h - h) // 2 : (out_h - h) // 2 + h, (out_w - w) // 2 : (out_w - w) // 2 + w] = x
    return y


def residual_scores(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    if y.dim() != 4:
        raise ValueError(f"expected measurement tensor as BCHW after normalization, got {tuple(y.shape)}")
    _, _, mh, mw = y.shape
    xpad = pad_center(x, mh, mw)
    mag = torch.fft.fft2(xpad, norm="ortho").abs()
    if y.shape[0] == 1 and mag.shape[0] != 1:
        y = y.expand(mag.shape[0], -1, -1, -1)
    if y.shape[1] == 1 and mag.shape[1] != 1:
        y = y.expand(-1, mag.shape[1], -1, -1)
    diff = mag - y
    sse = float(torch.sum(diff * diff).item())
    yn = float(torch.linalg.vector_norm(y).item())
    sqrt_over_y = math.sqrt(max(sse, 0.0)) / max(yn, 1e-12)
    return sse, sqrt_over_y


def rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-2, -1))


def calibration_rows(rows: Sequence[Dict[str, str]], max_rows: int) -> List[Dict[str, str]]:
    usable = [r for r in rows if r.get("sample_path") and Path(r["sample_path"]).exists() and ffloat(r.get("selector_sqrt_loss_over_y_norm")) == ffloat(r.get("selector_sqrt_loss_over_y_norm"))]
    return usable[:max_rows]


def calibrate(rows: Sequence[Dict[str, str]], max_rows: int) -> Dict[str, object]:
    cal = calibration_rows(rows, max_rows=max_rows)
    if not cal:
        return {"image_mode": "zero_one", "target_mode": "abs", "status": "no_recorded_rows"}
    candidates = []
    for image_mode in ("zero_one", "minus_one_one"):
        for target_mode in ("raw", "abs"):
            errs = []
            for r in cal:
                try:
                    y = load_measurement(r["measurement_path"], target_mode=target_mode)
                    x = load_png(r["sample_path"], image_mode=image_mode)
                    _sse, score = residual_scores(x, y)
                    rec = ffloat(r.get("selector_sqrt_loss_over_y_norm"))
                    if math.isfinite(rec):
                        errs.append(abs(score - rec))
                except Exception:
                    continue
            med = float(np.median(errs)) if errs else math.inf
            mean = float(np.mean(errs)) if errs else math.inf
            candidates.append({"image_mode": image_mode, "target_mode": target_mode, "median_abs_error": med, "mean_abs_error": mean, "n": len(errs)})
    candidates.sort(key=lambda d: (float(d["median_abs_error"]), float(d["mean_abs_error"])))
    best = candidates[0]
    out = dict(best)
    out["status"] = "ok"
    out["candidates"] = candidates
    return out


def score_rows(rows: Sequence[Dict[str, str]], image_mode: str, target_mode: str) -> List[Dict[str, object]]:
    y_cache: Dict[Tuple[str, str], torch.Tensor] = {}
    scored: List[Dict[str, object]] = []
    for i, r in enumerate(rows):
        sample_path = r.get("sample_path", "")
        meas_path = r.get("measurement_path", "")
        out: Dict[str, object] = dict(r)
        out["row_id"] = f"b21r{i:05d}"
        out["diagnostic_psnr_original"] = r.get("selector_psnr_recomputed_from_png", r.get("psnr", ""))
        out["recorded_sqrt_loss_over_y_norm"] = r.get("selector_sqrt_loss_over_y_norm", "")
        out["recorded_exact_operator_loss"] = r.get("selector_exact_operator_loss", "")
        if not sample_path or not Path(sample_path).exists():
            out["score_error"] = "missing_sample_path"
            scored.append(out)
            continue
        try:
            key = (meas_path, target_mode)
            if key not in y_cache:
                y_cache[key] = load_measurement(meas_path, target_mode=target_mode)
            y = y_cache[key]
            x = load_png(sample_path, image_mode=image_mode)
            orig_sse, orig_sqrt = residual_scores(x, y)
            rot_sse, rot_sqrt = residual_scores(rot180(x), y)
            out.update(
                {
                    "b21_orig_sse": orig_sse,
                    "b21_orig_sqrt_loss_over_y_norm": orig_sqrt,
                    "b21_rot180_sse": rot_sse,
                    "b21_rot180_sqrt_loss_over_y_norm": rot_sqrt,
                    "b21_best_oriented_sqrt_loss_over_y_norm": min(orig_sqrt, rot_sqrt),
                    "b21_orientation": "rot180" if rot_sqrt < orig_sqrt else "identity",
                    "b21_rot_minus_orig_sqrt": rot_sqrt - orig_sqrt,
                    "score_error": "",
                }
            )
        except Exception as exc:
            out["score_error"] = repr(exc)
        scored.append(out)
    return scored


def select_by(scored: Sequence[Dict[str, object]], group_cols: Sequence[str], score_col: str, tag: str) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for r in scored:
        if r.get("score_error"):
            continue
        key = tuple(str(r.get(c, "")) for c in group_cols)
        groups[key].append(r)
    rows = []
    for key, cand in sorted(groups.items()):
        cand = [r for r in cand if math.isfinite(ffloat(r.get(score_col)))]
        if not cand:
            continue
        best = min(cand, key=lambda r: ffloat(r.get(score_col)))
        out = {c: v for c, v in zip(group_cols, key)}
        out.update(
            {
                "selector": tag,
                "selected_row_id": best.get("row_id", ""),
                "selected_sample_path": best.get("sample_path", ""),
                "selected_orientation": best.get("b21_orientation", "identity") if tag != "recorded_exact" else "identity",
                "selected_score": ffloat(best.get(score_col)),
                "selected_variant": best.get("variant", ""),
                "selected_run_seed": best.get("run_seed", ""),
                "selected_ann_steps": best.get("ann_steps", ""),
                "selected_diff_steps": best.get("diff_steps", ""),
                "diagnostic_psnr_original": best.get("diagnostic_psnr_original", ""),
                "recorded_sqrt_loss_over_y_norm": best.get("recorded_sqrt_loss_over_y_norm", ""),
                "b21_orig_sqrt_loss_over_y_norm": best.get("b21_orig_sqrt_loss_over_y_norm", ""),
                "b21_rot180_sqrt_loss_over_y_norm": best.get("b21_rot180_sqrt_loss_over_y_norm", ""),
            }
        )
        rows.append(out)
    return rows


def render_report(outdir: Path, calibration: Dict[str, object], scored: Sequence[Dict[str, object]], selections: Sequence[Dict[str, object]]) -> str:
    n = len(scored)
    n_err = sum(1 for r in scored if r.get("score_error"))
    n_rot = sum(1 for r in scored if r.get("b21_orientation") == "rot180")
    psnrs = [ffloat(r.get("diagnostic_psnr_original")) for r in scored]
    psnrs = [x for x in psnrs if math.isfinite(x)]
    bad25 = sum(x < 25 for x in psnrs)
    bad20 = sum(x < 20 for x in psnrs)
    by_selector = defaultdict(list)
    for r in selections:
        by_selector[str(r.get("selector", ""))].append(r)
    lines = [
        "# B21.2 symmetry-aware selector-v2 first pass",
        "",
        "Status: generated by `scripts/b21/score_b21_2_symmetry_selector.py`.",
        "",
        "## Calibration",
        "",
        f"- image mode: `{calibration.get('image_mode')}`",
        f"- target mode: `{calibration.get('target_mode')}`",
        f"- median absolute calibration error: `{calibration.get('median_abs_error', '')}`",
        f"- calibration status: `{calibration.get('status')}`",
        "",
        "## Candidate diagnostics",
        "",
        f"- rows scored: `{n}`",
        f"- rows with scoring error: `{n_err}`",
        f"- rows where rot180 has lower measurement residual: `{n_rot}`",
        f"- diagnostic original-image PSNR bad25 count: `{bad25}`",
        f"- diagnostic original-image PSNR bad20 count: `{bad20}`",
        "",
        "## Per-image selections",
        "",
        "| selector | image_id | selected score | orientation | variant | seed | diagnostic original PSNR |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for r in selections:
        lines.append(
            f"| `{r.get('selector')}` | `{r.get('image_id')}` | {ffloat(r.get('selected_score')):.6g} | `{r.get('selected_orientation')}` | `{r.get('selected_variant')}` | {r.get('selected_run_seed')} | {ffloat(r.get('diagnostic_psnr_original')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "The selector is clean-free. The PSNR column is diagnostic only and is not used for selection. If a row is selected with `rot180` orientation, the reported diagnostic PSNR is still the original unrotated PNG PSNR because this pass does not require ground truth and does not recompute rotated PSNR.",
            "",
            "Artifacts:",
            "",
            "```text",
            str(outdir / "b21_2_symmetry_scores.csv"),
            str(outdir / "b21_2_symmetry_selections.csv"),
            str(outdir / "b21_2_symmetry_summary.json"),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score B21.2 candidate recovery rows with rot180-aware residuals")
    ap.add_argument("--input_csv", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_candidate_recovery/candidate_recovery_rows.csv")
    ap.add_argument("--outdir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_symmetry_selector")
    ap.add_argument("--report_path", default="docs/b21/b21_2_symmetry_selector.md")
    ap.add_argument("--calibration_rows", type=int, default=20)
    args = ap.parse_args()

    rows = read_csv(Path(args.input_csv))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(rows, max_rows=args.calibration_rows)
    image_mode = str(calibration.get("image_mode", "zero_one"))
    target_mode = str(calibration.get("target_mode", "abs"))
    scored = score_rows(rows, image_mode=image_mode, target_mode=target_mode)

    selections: List[Dict[str, object]] = []
    selections.extend(select_by(scored, ["image_id"], "recorded_sqrt_loss_over_y_norm", "recorded_exact"))
    selections.extend(select_by(scored, ["image_id"], "b21_orig_sqrt_loss_over_y_norm", "recomputed_identity"))
    selections.extend(select_by(scored, ["image_id"], "b21_best_oriented_sqrt_loss_over_y_norm", "rot180_aware"))

    write_csv(outdir / "b21_2_symmetry_scores.csv", scored)
    write_csv(outdir / "b21_2_symmetry_selections.csv", selections)
    summary = {
        "input_csv": args.input_csv,
        "rows": len(rows),
        "scored_rows": len(scored),
        "score_errors": sum(1 for r in scored if r.get("score_error")),
        "rot180_better_rows": sum(1 for r in scored if r.get("b21_orientation") == "rot180"),
        "calibration": calibration,
        "outdir": str(outdir),
    }
    (outdir / "b21_2_symmetry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(outdir, calibration, scored, selections)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"[write] {outdir / 'b21_2_symmetry_scores.csv'}")
    print(f"[write] {outdir / 'b21_2_symmetry_selections.csv'}")
    print(f"[write] {outdir / 'b21_2_symmetry_summary.json'}")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
