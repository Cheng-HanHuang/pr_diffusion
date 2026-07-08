#!/usr/bin/env python3
"""B21.2 denoiser-prior selector-v2 scorer.

This is a clean-free final-candidate scorer.  It consumes the B21.2 candidate
recovery CSV and evaluates each saved PNG and its rot180 version using the FFHQ
DDPM prior only.  The score is a small-noise denoising self-consistency loss:

    score(x) = E_{sigma,z} mean((Tweedie(x + sigma z, sigma) - x)^2)

Lower is interpreted as more locally consistent with the diffusion prior.  The
selector combines this with the recorded exact data-term via a near-tie rule:
first keep candidates with recorded data residual within (1+eta) of the per-image
minimum, then choose the one with the lowest prior score.

PSNR columns are diagnostic only and never used in selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image


DEFAULT_REPO = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver")
DEFAULT_DAPS = DEFAULT_REPO / "external/daps"
DEFAULT_INPUT = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_candidate_recovery/candidate_recovery_rows.csv")
DEFAULT_OUTDIR = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_denoiser_prior")
DEFAULT_CKPT = Path("/egr/research-pac/huang248/models/ffhq_10m.pt")


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


def load_png_model_range(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).float()
    return x * 2.0 - 1.0


def parse_float_list(s: str) -> List[float]:
    out = []
    for p in s.split(","):
        p = p.strip()
        if p:
            out.append(float(p))
    return out


def parse_eta_list(s: str) -> List[float]:
    return parse_float_list(s)


def build_model(daps_root: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(daps_root))
    from model import get_model  # noqa: WPS433,E402

    model_config = dict(
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        channel_mult="",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions=16,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        model_path=str(checkpoint_path),
    )
    model = get_model(name="ddpm", model_config=model_config, device=str(device), requires_grad=False)
    model.eval()
    return model


def make_orientation_records(rows: Sequence[Dict[str, str]]) -> tuple[List[Dict[str, object]], torch.Tensor]:
    records: List[Dict[str, object]] = []
    tensors: List[torch.Tensor] = []
    for i, row in enumerate(rows):
        sp = row.get("sample_path", "")
        base: Dict[str, object] = dict(row)
        base["source_row_index"] = i
        base["row_id"] = f"b21r{i:05d}"
        base["diagnostic_psnr_original"] = row.get("selector_psnr_recomputed_from_png", row.get("psnr", ""))
        base["recorded_sqrt_loss_over_y_norm"] = row.get("selector_sqrt_loss_over_y_norm", "")
        base["recorded_exact_operator_loss"] = row.get("selector_exact_operator_loss", "")
        if not sp or not Path(sp).exists():
            for orient in ("identity", "rot180"):
                rec = dict(base)
                rec["prior_orientation"] = orient
                rec["score_error"] = "missing_sample_path"
                records.append(rec)
            continue
        x = load_png_model_range(sp)
        for orient, xt in (("identity", x), ("rot180", torch.flip(x, dims=(-2, -1)))):
            rec = dict(base)
            rec["prior_orientation"] = orient
            rec["score_error"] = ""
            rec["orientation_row_index"] = len(records)
            records.append(rec)
            tensors.append(xt)
    if tensors:
        stack = torch.stack(tensors, dim=0)
    else:
        stack = torch.empty(0, 3, 256, 256)
    return records, stack


def score_prior(
    model,
    records: List[Dict[str, object]],
    x_cpu: torch.Tensor,
    device: torch.device,
    sigmas: Sequence[float],
    draws: int,
    batch_size: int,
    noise_seed: int,
) -> None:
    valid_indices = [i for i, r in enumerate(records) if not r.get("score_error")]
    if not valid_indices:
        return
    # records and x_cpu are aligned only for valid rows in insertion order.
    # Build a map from record index to tensor index.
    rec_to_tensor: Dict[int, int] = {}
    t = 0
    for i, r in enumerate(records):
        if not r.get("score_error"):
            rec_to_tensor[i] = t
            t += 1

    n = x_cpu.shape[0]
    accum = torch.zeros(n, dtype=torch.float64)
    accum_shift = torch.zeros(n, dtype=torch.float64)
    total_terms = 0

    model_device = device
    for sigma_idx, sigma in enumerate(sigmas):
        sigma_t = torch.tensor(float(sigma), device=model_device, dtype=torch.float32)
        for draw in range(draws):
            gen = torch.Generator(device=model_device)
            gen.manual_seed(int(noise_seed + 1009 * sigma_idx + 104729 * draw))
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = x_cpu[start:end].to(model_device, non_blocking=True)
                z = torch.randn((end - start, 3, 256, 256), device=model_device, generator=gen, dtype=xb.dtype)
                noisy = xb + float(sigma) * z
                with torch.no_grad():
                    den = model.tweedie(noisy, sigma_t)
                mse_self = (den - xb).pow(2).flatten(1).mean(dim=1).detach().cpu().double()
                mse_shift = (den - noisy).pow(2).flatten(1).mean(dim=1).detach().cpu().double()
                accum[start:end] += mse_self
                accum_shift[start:end] += mse_shift
            total_terms += 1

    mean_self = accum / max(total_terms, 1)
    mean_shift = accum_shift / max(total_terms, 1)
    for rec_idx, tensor_idx in rec_to_tensor.items():
        records[rec_idx]["prior_denoise_self_mse"] = float(mean_self[tensor_idx].item())
        records[rec_idx]["prior_denoise_shift_mse"] = float(mean_shift[tensor_idx].item())
        records[rec_idx]["prior_sigmas"] = ",".join(str(x) for x in sigmas)
        records[rec_idx]["prior_draws"] = draws
        records[rec_idx]["prior_noise_seed"] = noise_seed


def collapse_best_orientation(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_row: Dict[str, List[Dict[str, object]]] = {}
    for r in records:
        by_row.setdefault(str(r.get("row_id", "")), []).append(r)
    out: List[Dict[str, object]] = []
    for row_id, cand in sorted(by_row.items()):
        valid = [r for r in cand if not r.get("score_error") and math.isfinite(ffloat(r.get("prior_denoise_self_mse")))]
        if not valid:
            # Preserve one error row for traceability.
            rr = dict(cand[0])
            rr["prior_best_orientation"] = ""
            rr["prior_best_denoise_self_mse"] = math.nan
            out.append(rr)
            continue
        best = min(valid, key=lambda r: ffloat(r.get("prior_denoise_self_mse")))
        identity = next((r for r in valid if r.get("prior_orientation") == "identity"), None)
        rot = next((r for r in valid if r.get("prior_orientation") == "rot180"), None)
        rr = dict(best)
        rr["prior_best_orientation"] = best.get("prior_orientation", "")
        rr["prior_best_denoise_self_mse"] = ffloat(best.get("prior_denoise_self_mse"))
        rr["prior_identity_denoise_self_mse"] = ffloat(identity.get("prior_denoise_self_mse")) if identity else math.nan
        rr["prior_rot180_denoise_self_mse"] = ffloat(rot.get("prior_denoise_self_mse")) if rot else math.nan
        rr["prior_rot_minus_identity_mse"] = rr["prior_rot180_denoise_self_mse"] - rr["prior_identity_denoise_self_mse"]
        out.append(rr)
    return out


def selection_rows(best_rows: Sequence[Dict[str, object]], etas: Sequence[float]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in best_rows:
        if r.get("score_error"):
            continue
        groups.setdefault(str(r.get("image_id", "")), []).append(r)

    out: List[Dict[str, object]] = []
    for image_id, rows in sorted(groups.items()):
        rec_vals = [ffloat(r.get("recorded_sqrt_loss_over_y_norm")) for r in rows]
        rec_vals = [v for v in rec_vals if math.isfinite(v)]
        if not rec_vals:
            continue
        min_rec = min(rec_vals)

        # Baseline: recorded exact selector alone.
        exact_cand = [r for r in rows if math.isfinite(ffloat(r.get("recorded_sqrt_loss_over_y_norm")))]
        exact_best = min(exact_cand, key=lambda r: ffloat(r.get("recorded_sqrt_loss_over_y_norm")))
        out.append(make_selection("recorded_exact", image_id, exact_best, eta=math.nan, min_recorded=min_rec, n_eligible=len(exact_cand)))

        for eta in etas:
            eligible = [
                r for r in rows
                if math.isfinite(ffloat(r.get("recorded_sqrt_loss_over_y_norm")))
                and ffloat(r.get("recorded_sqrt_loss_over_y_norm")) <= (1.0 + eta) * min_rec
                and math.isfinite(ffloat(r.get("prior_best_denoise_self_mse")))
            ]
            if not eligible:
                continue
            best = min(eligible, key=lambda r: ffloat(r.get("prior_best_denoise_self_mse")))
            out.append(make_selection("exact_neartie_prior", image_id, best, eta=eta, min_recorded=min_rec, n_eligible=len(eligible)))
    return out


def make_selection(selector: str, image_id: str, r: Dict[str, object], eta: float, min_recorded: float, n_eligible: int) -> Dict[str, object]:
    return {
        "selector": selector,
        "image_id": image_id,
        "eta": eta,
        "n_eligible": n_eligible,
        "min_recorded_sqrt_loss_over_y_norm": min_recorded,
        "selected_row_id": r.get("row_id", ""),
        "selected_sample_path": r.get("sample_path", ""),
        "selected_variant": r.get("variant", ""),
        "selected_run_seed": r.get("run_seed", ""),
        "selected_ann_steps": r.get("ann_steps", ""),
        "selected_diff_steps": r.get("diff_steps", ""),
        "selected_orientation": r.get("prior_best_orientation", "identity"),
        "selected_recorded_sqrt_loss_over_y_norm": ffloat(r.get("recorded_sqrt_loss_over_y_norm")),
        "selected_prior_denoise_self_mse": ffloat(r.get("prior_best_denoise_self_mse")),
        "selected_prior_identity_mse": ffloat(r.get("prior_identity_denoise_self_mse")),
        "selected_prior_rot180_mse": ffloat(r.get("prior_rot180_denoise_self_mse")),
        "diagnostic_psnr_original": r.get("diagnostic_psnr_original", ""),
    }


def render_report(outdir: Path, summary: Dict[str, object], selections: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# B21.2 denoiser-prior selector-v2",
        "",
        "Status: generated by `scripts/b21/score_b21_2_denoiser_prior.py`.",
        "",
        "## Summary",
        "",
        f"- input rows: `{summary['input_rows']}`",
        f"- orientation rows: `{summary['orientation_rows']}`",
        f"- score errors: `{summary['score_errors']}`",
        f"- rot180 prior-better candidates: `{summary['rot180_prior_better_candidates']}`",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- sigmas: `{summary['sigmas']}`",
        f"- draws: `{summary['draws']}`",
        "",
        "## Selections",
        "",
        "| selector | eta | image_id | eligible | orientation | variant | seed | recorded data score | prior score | diagnostic original PSNR |",
        "|---|---:|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in selections:
        eta = r.get("eta", "")
        eta_s = "" if not math.isfinite(ffloat(eta)) else f"{ffloat(eta):.4g}"
        lines.append(
            f"| `{r.get('selector')}` | {eta_s} | `{r.get('image_id')}` | {r.get('n_eligible')} | `{r.get('selected_orientation')}` | `{r.get('selected_variant')}` | {r.get('selected_run_seed')} | {ffloat(r.get('selected_recorded_sqrt_loss_over_y_norm')):.6g} | {ffloat(r.get('selected_prior_denoise_self_mse')):.6g} | {ffloat(r.get('diagnostic_psnr_original')):.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "This selector is clean-free: recorded data residual and denoiser-prior scores do not use ground truth. PSNR is diagnostic only. When `selected_orientation=rot180`, the PSNR shown is still the original unrotated PNG's diagnostic PSNR.",
        "",
        "Artifacts:",
        "",
        "```text",
        str(outdir / "b21_2_prior_orientation_scores.csv"),
        str(outdir / "b21_2_prior_candidate_scores.csv"),
        str(outdir / "b21_2_prior_selections.csv"),
        str(outdir / "b21_2_prior_summary.json"),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="B21.2 denoiser-prior clean-free scorer")
    ap.add_argument("--input_csv", default=str(DEFAULT_INPUT))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--daps_root", default=str(DEFAULT_DAPS))
    ap.add_argument("--checkpoint_path", default=str(DEFAULT_CKPT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sigmas", default="0.05,0.10")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--noise_seed", type=int, default=91021)
    ap.add_argument("--etas", default="0,0.005,0.01,0.02,0.05,0.10")
    ap.add_argument("--report_path", default=str(DEFAULT_REPO / "docs/b21/b21_2_denoiser_prior.md"))
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FFHQ prior scoring")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    sigmas = parse_float_list(args.sigmas)
    etas = parse_eta_list(args.etas)
    rows = read_csv(input_csv)
    orientation_records, x_cpu = make_orientation_records(rows)

    print(f"[info] input_csv={input_csv}")
    print(f"[info] rows={len(rows)} orientation_rows={len(orientation_records)} valid_tensors={tuple(x_cpu.shape)}")
    print(f"[info] checkpoint={checkpoint_path}")
    print(f"[info] device={device} sigmas={sigmas} draws={args.draws} batch_size={args.batch_size}")

    model = build_model(Path(args.daps_root), checkpoint_path, device=device)
    score_prior(model, orientation_records, x_cpu, device, sigmas, args.draws, args.batch_size, args.noise_seed)
    candidate_rows = collapse_best_orientation(orientation_records)
    selections = selection_rows(candidate_rows, etas)

    write_csv(outdir / "b21_2_prior_orientation_scores.csv", orientation_records)
    write_csv(outdir / "b21_2_prior_candidate_scores.csv", candidate_rows)
    write_csv(outdir / "b21_2_prior_selections.csv", selections)

    summary = {
        "input_csv": str(input_csv),
        "checkpoint_path": str(checkpoint_path),
        "input_rows": len(rows),
        "orientation_rows": len(orientation_records),
        "candidate_rows": len(candidate_rows),
        "score_errors": sum(1 for r in orientation_records if r.get("score_error")),
        "rot180_prior_better_candidates": sum(1 for r in candidate_rows if r.get("prior_best_orientation") == "rot180"),
        "sigmas": sigmas,
        "draws": args.draws,
        "noise_seed": args.noise_seed,
        "batch_size": args.batch_size,
        "gpu": args.gpu,
        "outdir": str(outdir),
    }
    (outdir / "b21_2_prior_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(outdir, summary, selections)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"[write] {outdir / 'b21_2_prior_orientation_scores.csv'}")
    print(f"[write] {outdir / 'b21_2_prior_candidate_scores.csv'}")
    print(f"[write] {outdir / 'b21_2_prior_selections.csv'}")
    print(f"[write] {outdir / 'b21_2_prior_summary.json'}")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
