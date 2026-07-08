#!/usr/bin/env python3
"""Fixed B21.2 symmetry-aware residual scorer.

Consumes B21_2_candidate_recovery/candidate_recovery_rows.csv.  The recovery
CSV does not contain measurement_path, so this script reconstructs the locked
measurement path from (image_id, meas_seed) unless a measurement_path column is
present.

Clean-free selector columns:
- recorded_exact: historical recorded selector_sqrt_loss_over_y_norm
- recomputed_identity: recomputed residual for the saved PNG
- rot180_aware: min(identity residual, rot180 residual)

PSNR is diagnostic only and never used for selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def ffloat(x: object, default: float = math.nan) -> float:
    try:
        if x is None or str(x) == "":
            return default
        return float(x)
    except Exception:
        return default


def norm_image_id(x: object) -> str:
    return f"{int(str(x)):05d}"


def infer_measurement_path(row: Dict[str, str], b19_base: Path) -> str:
    mp = row.get("measurement_path", "")
    if mp:
        return mp
    image_id = norm_image_id(row.get("image_id", "0"))
    meas_seed = int(float(row.get("meas_seed", "5001")))
    return str(b19_base / "measurements" / f"ffhq{image_id}_phase_noise005_meas{meas_seed}.pt")


def first_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ("measurement", "y", "observed", "data"):
            if k in obj:
                t = first_tensor(obj[k])
                if t is not None:
                    return t
        for v in obj.values():
            t = first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = first_tensor(v)
            if t is not None:
                return t
    return None


def load_measurement(path: str, target_mode: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    y = first_tensor(obj)
    if y is None:
        raise RuntimeError(f"no tensor found in measurement payload: {path}")
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


def center_pad(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if h > out_h or w > out_w:
        raise ValueError(f"image {h}x{w} larger than target grid {out_h}x{out_w}")
    y = torch.zeros((b, c, out_h, out_w), dtype=x.dtype)
    top = (out_h - h) // 2
    left = (out_w - w) // 2
    y[:, :, top : top + h, left : left + w] = x
    return y


def residual_scores(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    if y.dim() != 4:
        raise ValueError(f"measurement should be BCHW after loading, got {tuple(y.shape)}")
    _, _, mh, mw = y.shape
    xpad = center_pad(x, mh, mw)
    mag = torch.fft.fft2(xpad, norm="ortho").abs()
    yy = y
    if yy.shape[0] == 1 and mag.shape[0] != 1:
        yy = yy.expand(mag.shape[0], -1, -1, -1)
    if yy.shape[1] == 1 and mag.shape[1] != 1:
        yy = yy.expand(-1, mag.shape[1], -1, -1)
    diff = mag - yy
    sse = float(torch.sum(diff * diff).item())
    yn = float(torch.linalg.vector_norm(yy).item())
    return sse, math.sqrt(max(sse, 0.0)) / max(yn, 1e-12)


def rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-2, -1))


def calibrate(rows: Sequence[Dict[str, str]], b19_base: Path, max_rows: int) -> Dict[str, object]:
    cal = []
    for r in rows:
        sp = r.get("sample_path", "")
        rec = ffloat(r.get("selector_sqrt_loss_over_y_norm"))
        if sp and Path(sp).exists() and math.isfinite(rec):
            cal.append(r)
        if len(cal) >= max_rows:
            break
    candidates = []
    for image_mode in ("zero_one", "minus_one_one"):
        for target_mode in ("raw", "abs"):
            errs = []
            for r in cal:
                try:
                    mp = infer_measurement_path(r, b19_base)
                    y = load_measurement(mp, target_mode)
                    x = load_png(r["sample_path"], image_mode)
                    _, score = residual_scores(x, y)
                    errs.append(abs(score - ffloat(r.get("selector_sqrt_loss_over_y_norm"))))
                except Exception:
                    continue
            candidates.append(
                {
                    "image_mode": image_mode,
                    "target_mode": target_mode,
                    "n": len(errs),
                    "median_abs_error": float(np.median(errs)) if errs else math.inf,
                    "mean_abs_error": float(np.mean(errs)) if errs else math.inf,
                }
            )
    candidates.sort(key=lambda d: (float(d["median_abs_error"]), float(d["mean_abs_error"])))
    best = dict(candidates[0]) if candidates else {"image_mode": "zero_one", "target_mode": "raw", "n": 0}
    best["status"] = "ok" if best.get("n", 0) else "no_successful_calibration_rows"
    best["candidates"] = candidates
    return best


def score(rows: Sequence[Dict[str, str]], b19_base: Path, image_mode: str, target_mode: str) -> List[Dict[str, object]]:
    y_cache: Dict[Tuple[str, str], torch.Tensor] = {}
    out_rows: List[Dict[str, object]] = []
    for i, r in enumerate(rows):
        out: Dict[str, object] = dict(r)
        out["row_id"] = f"b21r{i:05d}"
        out["measurement_path"] = infer_measurement_path(r, b19_base)
        out["diagnostic_psnr_original"] = r.get("selector_psnr_recomputed_from_png", r.get("psnr", ""))
        out["recorded_sqrt_loss_over_y_norm"] = r.get("selector_sqrt_loss_over_y_norm", "")
        out["recorded_exact_operator_loss"] = r.get("selector_exact_operator_loss", "")
        sp = r.get("sample_path", "")
        if not sp or not Path(sp).exists():
            out["score_error"] = "missing_sample_path"
            out_rows.append(out)
            continue
        mp = str(out["measurement_path"])
        if not Path(mp).exists():
            out["score_error"] = f"missing_measurement_path:{mp}"
            out_rows.append(out)
            continue
        try:
            key = (mp, target_mode)
            if key not in y_cache:
                y_cache[key] = load_measurement(mp, target_mode)
            x = load_png(sp, image_mode)
            orig_sse, orig_sqrt = residual_scores(x, y_cache[key])
            rot_sse, rot_sqrt = residual_scores(rot180(x), y_cache[key])
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
        out_rows.append(out)
    return out_rows


def select_by(scored: Sequence[Dict[str, object]], score_col: str, selector: str) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in scored:
        if not r.get("score_error"):
            groups[str(r.get("image_id", ""))].append(r)
    rows: List[Dict[str, object]] = []
    for image_id, cand in sorted(groups.items()):
        cand = [r for r in cand if math.isfinite(ffloat(r.get(score_col)))]
        if not cand:
            continue
        best = min(cand, key=lambda r: ffloat(r.get(score_col)))
        rows.append(
            {
                "selector": selector,
                "image_id": image_id,
                "selected_row_id": best.get("row_id", ""),
                "selected_score": ffloat(best.get(score_col)),
                "selected_orientation": best.get("b21_orientation", "identity") if selector != "recorded_exact" else "identity",
                "selected_variant": best.get("variant", ""),
                "selected_run_seed": best.get("run_seed", ""),
                "selected_ann_steps": best.get("ann_steps", ""),
                "selected_diff_steps": best.get("diff_steps", ""),
                "selected_sample_path": best.get("sample_path", ""),
                "diagnostic_psnr_original": best.get("diagnostic_psnr_original", ""),
                "recorded_sqrt_loss_over_y_norm": best.get("recorded_sqrt_loss_over_y_norm", ""),
                "b21_orig_sqrt_loss_over_y_norm": best.get("b21_orig_sqrt_loss_over_y_norm", ""),
                "b21_rot180_sqrt_loss_over_y_norm": best.get("b21_rot180_sqrt_loss_over_y_norm", ""),
            }
        )
    return rows


def render_report(outdir: Path, calibration: Dict[str, object], scored: Sequence[Dict[str, object]], selections: Sequence[Dict[str, object]]) -> str:
    psnrs = [ffloat(r.get("diagnostic_psnr_original")) for r in scored]
    psnrs = [x for x in psnrs if math.isfinite(x)]
    lines = [
        "# B21.2 symmetry-aware selector-v2 first pass",
        "",
        "Status: generated by `scripts/b21/score_b21_2_symmetry_selector_v2.py`.",
        "",
        "## Calibration",
        "",
        f"- image mode: `{calibration.get('image_mode')}`",
        f"- target mode: `{calibration.get('target_mode')}`",
        f"- calibration rows: `{calibration.get('n')}`",
        f"- median absolute calibration error: `{calibration.get('median_abs_error', '')}`",
        f"- calibration status: `{calibration.get('status')}`",
        "",
        "## Candidate diagnostics",
        "",
        f"- rows scored: `{len(scored)}`",
        f"- rows with scoring error: `{sum(1 for r in scored if r.get('score_error'))}`",
        f"- rows where rot180 has lower measurement residual: `{sum(1 for r in scored if r.get('b21_orientation') == 'rot180')}`",
        f"- diagnostic original-image PSNR bad25 count: `{sum(x < 25 for x in psnrs)}`",
        f"- diagnostic original-image PSNR bad20 count: `{sum(x < 20 for x in psnrs)}`",
        "",
        "## Per-image selections",
        "",
        "| selector | image_id | selected score | orientation | variant | seed | diagnostic original PSNR |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for r in selections:
        lines.append(
            f"| `{r['selector']}` | `{r['image_id']}` | {ffloat(r['selected_score']):.6g} | `{r['selected_orientation']}` | `{r['selected_variant']}` | {r['selected_run_seed']} | {ffloat(r['diagnostic_psnr_original']):.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "The selector is clean-free. PSNR is diagnostic only. `rot180_aware` chooses by measurement residual after allowing an orientation flip, but the reported PSNR remains the original unrotated PNG PSNR.",
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
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="B21.2 fixed rot180-aware residual scoring")
    ap.add_argument("--input_csv", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_candidate_recovery/candidate_recovery_rows.csv")
    ap.add_argument("--b19_base", default="/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
    ap.add_argument("--outdir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_symmetry_selector")
    ap.add_argument("--report_path", default="docs/b21/b21_2_symmetry_selector.md")
    ap.add_argument("--calibration_rows", type=int, default=20)
    args = ap.parse_args()

    rows = read_csv(Path(args.input_csv))
    b19_base = Path(args.b19_base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    calibration = calibrate(rows, b19_base=b19_base, max_rows=args.calibration_rows)
    image_mode = str(calibration.get("image_mode", "zero_one"))
    target_mode = str(calibration.get("target_mode", "raw"))
    scored = score(rows, b19_base=b19_base, image_mode=image_mode, target_mode=target_mode)
    selections = []
    selections += select_by(scored, "recorded_sqrt_loss_over_y_norm", "recorded_exact")
    selections += select_by(scored, "b21_orig_sqrt_loss_over_y_norm", "recomputed_identity")
    selections += select_by(scored, "b21_best_oriented_sqrt_loss_over_y_norm", "rot180_aware")

    write_csv(outdir / "b21_2_symmetry_scores.csv", scored)
    write_csv(outdir / "b21_2_symmetry_selections.csv", selections)
    summary = {
        "input_csv": args.input_csv,
        "b19_base": args.b19_base,
        "rows": len(rows),
        "scored_rows": len(scored),
        "score_errors": sum(1 for r in scored if r.get("score_error")),
        "rot180_better_rows": sum(1 for r in scored if r.get("b21_orientation") == "rot180"),
        "calibration": calibration,
        "outdir": str(outdir),
    }
    (outdir / "b21_2_symmetry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(outdir, calibration, scored, selections)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text(report, encoding="utf-8")

    print(f"[write] {outdir / 'b21_2_symmetry_scores.csv'}")
    print(f"[write] {outdir / 'b21_2_symmetry_selections.csv'}")
    print(f"[write] {outdir / 'b21_2_symmetry_summary.json'}")
    print(f"[write] {args.report_path}")


if __name__ == "__main__":
    main()
