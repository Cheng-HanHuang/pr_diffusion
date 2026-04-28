#!/usr/bin/env python3
"""Run NP variants under a DiffFPR-style oversampled Fourier PR setting.

This is an external-benchmark wrapper for comparing the current NP solvers against
paper settings such as DiffFPR.  It intentionally does not replace the in-repo
canonical Phase 10/11 runner, because the measurement operator differs:

- in-repo canonical NP: same-size Fourier magnitude on images in [-1, 1];
- DiffFPR-style benchmark: zero-padded / oversampled Fourier amplitude on images
  in [0, 1], with optional additive amplitude noise.

The algorithmic NP schedules are kept the same as the current docs:

- np_canonical: soft=5, hard=1, proj_start=400;
- np_fixedk_lateproj: soft=5, hard=5, proj_start=400 by default.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from prdiffusion.diffusion import load_model
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.seed import seed_everything


def parse_int_list(text: str) -> List[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


@dataclass(frozen=True)
class PaperPreset:
    name: str
    model_id: str
    image_size: int
    oversample: float
    noise_levels: List[float]
    note: str


PAPER_PRESETS: Dict[str, PaperPreset] = {
    # DiffFPR Table 2 uses FFHQ/ImageNet at 256x256 with oversampling ratio r^2=4.0
    # and noise levels sigma_y in {0.00, 0.01, 0.05}.
    "ffhq": PaperPreset(
        name="ffhq",
        model_id="google/ddpm-celebahq-256",
        image_size=256,
        oversample=4.0,
        noise_levels=[0.0, 0.01, 0.05],
        note="Match DiffFPR FFHQ table setting (oversampling ratio r^2=4.0).",
    ),
    # For ImageNet, use an unconditional 256x256 DDPM prior.
    "imagenet": PaperPreset(
        name="imagenet",
        model_id="google/ddpm-ema-256",
        image_size=256,
        oversample=4.0,
        noise_levels=[0.0, 0.01, 0.05],
        note="Match DiffFPR ImageNet table setting (oversampling ratio r^2=4.0).",
    ),
}


def read_image_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def collect_images(data_root: str, image_list_file: Optional[str]) -> List[str]:
    if image_list_file:
        return read_image_list(image_list_file)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: List[str] = []
    for dirpath, _, filenames in os.walk(data_root):
        for name in filenames:
            if os.path.splitext(name.lower())[1] in exts:
                out.append(os.path.relpath(os.path.join(dirpath, name), data_root))
    return sorted(out)


def resolve_image_path(data_root: str, image_name: str) -> str:
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name
    direct = os.path.join(data_root, image_name)
    if os.path.exists(direct):
        return direct
    found = find_image_by_basename(data_root, os.path.basename(image_name))
    if found is not None:
        return found
    raise FileNotFoundError(f"Could not find {image_name!r} under {data_root}")


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def from01(x01: torch.Tensor) -> torch.Tensor:
    return x01.clamp(0.0, 1.0) * 2.0 - 1.0


def complex_abs_safe(z: torch.Tensor) -> torch.Tensor:
    """CUDA-safe complex magnitude; avoids complex_tensor.abs() kernel issues."""
    return torch.sqrt(z.real.square() + z.imag.square()).contiguous()

def parse_radius_schedule(text: str | None, default_radius: float) -> list[tuple[int, float]]:
    """Parse schedules like '500:0.2,800:0.4,900:0.72'.

    The returned list is sorted by start step. The active radius at step i is
    the radius from the latest pair whose start <= i.
    """
    if text is None or str(text).strip() == "":
        return [(0, float(default_radius))]

    out: list[tuple[int, float]] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid radius schedule item {chunk!r}; expected 'step:radius'."
            )
        step_s, radius_s = chunk.split(":", 1)
        out.append((int(step_s), float(radius_s)))

    if not out:
        return [(0, float(default_radius))]
    out.sort(key=lambda x: x[0])

    # Safety: before the first explicitly scheduled change, use the default
    # projection radius. This prevents a schedule like "800:0.4" from applying
    # 0.4 immediately at proj_start.
    if out[0][0] > 0:
        out.insert(0, (0, float(default_radius)))

    return out


def radius_at_step(schedule: list[tuple[int, float]], step_index: int) -> float:
    radius = float(schedule[0][1])
    for start, r in schedule:
        if step_index >= int(start):
            radius = float(r)
        else:
            break
    return radius


def schedule_to_string(schedule: list[tuple[int, float]]) -> str:
    return ",".join(f"{int(s)}:{float(r):g}" for s, r in schedule)


def centered_fft2(x: torch.Tensor) -> torch.Tensor:
    # Match the centered FFT convention used by DiffFPR / fastMRI utilities.
    # Force contiguous buffers because some CUDA/cuFFT builds fail on shifted
    # or image-derived non-standard strides.
    x_shift = torch.fft.ifftshift(x.contiguous(), dim=(-2, -1)).contiguous()
    X = torch.fft.fftn(x_shift, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(X, dim=(-2, -1)).contiguous()


def centered_ifft2(X: torch.Tensor) -> torch.Tensor:
    X_shift = torch.fft.ifftshift(X.contiguous(), dim=(-2, -1)).contiguous()
    x = torch.fft.ifftn(X_shift, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1)).real.contiguous()


def oversample_pad(image_size: int, oversample: float) -> int:
    # DiffFPR code uses int((oversample / 8.0) * 256).  Generalize to size.
    return int((float(oversample) / 8.0) * int(image_size))


def pad01(x01: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x01
    return F.pad(x01, (pad, pad, pad, pad))


def crop01(xpad01: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return xpad01
    return xpad01[..., pad:-pad, pad:-pad]


def oversampled_magnitude(x: torch.Tensor, pad: int) -> torch.Tensor:
    return complex_abs_safe(centered_fft2(pad01(to01(x), pad)))


def centered_lowfreq_mask(h: int, w: int, radius: float, device: torch.device) -> torch.Tensor:
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, d=1.0, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, d=1.0, device=device))
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(ky**2 + kx**2) <= float(radius)


def oversampled_mag_l2(x: torch.Tensor, mag_target: torch.Tensor, pad: int) -> torch.Tensor:
    return torch.norm(oversampled_magnitude(x, pad) - mag_target)


def oversampled_lowfreq_mag_l2(
    x: torch.Tensor,
    mag_target: torch.Tensor,
    pad: int,
    radius: float,
) -> torch.Tensor:
    mag = oversampled_magnitude(x, pad)
    _, _, h, w = mag.shape
    mask_hw = centered_lowfreq_mask(h, w, radius, mag.device)
    mask = mask_hw[None, None, :, :].expand_as(mag)
    return torch.norm(mag[mask] - mag_target[mask])


def enforce_oversampled_lowfreq(
    x: torch.Tensor,
    mag_target: torch.Tensor,
    pad: int,
    radius: float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """One Gerchberg-Saxton-style magnitude replacement on the padded grid."""
    xpad01 = pad01(to01(x), pad)
    X = centered_fft2(xpad01)
    mag = complex_abs_safe(X)
    phase = X / (mag + eps)
    _, _, h, w = X.shape
    mask_hw = centered_lowfreq_mask(h, w, radius, X.device)
    mask = mask_hw[None, None, :, :]
    new_mag = torch.where(mask, mag_target, mag)
    xpad_new01 = centered_ifft2(phase * new_mag)
    return from01(crop01(xpad_new01, pad))


@dataclass(frozen=True)
class NPVariant:
    name: str
    soft: int
    hard: int
    proj_start: int
    use_lowfreq_score: bool = True
    use_lowfreq_projection: bool = True


def make_variants(names: Iterable[str], *, late_start: int, fixed_k: int) -> List[NPVariant]:
    variants: List[NPVariant] = []
    for name in names:
        if name == "np_canonical":
            variants.append(NPVariant(name=name, soft=5, hard=1, proj_start=late_start))
        elif name == "np_fixedk_lateproj":
            variants.append(NPVariant(name=name, soft=fixed_k, hard=fixed_k, proj_start=late_start))
        else:
            raise ValueError(
                f"Unknown variant {name!r}. Supported: np_canonical, np_fixedk_lateproj"
            )
    return variants


@torch.no_grad()
def pick_noise_oversampled(
    *,
    x0_target: torch.Tensor,
    t_int: int,
    eps_prev: Optional[torch.Tensor],
    mag_target: torch.Tensor,
    pad: int,
    num_candidates: int,
    score_radius: Optional[float],
    unet,
    scheduler,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar_t = scheduler.alphas_cumprod[t_int].to(device=x0_target.device, dtype=x0_target.dtype)
    sqrt_at = torch.sqrt(alpha_bar_t)
    sqrt_1mat = torch.sqrt(1.0 - alpha_bar_t)

    best_x0 = None
    best_eps = None
    best_score = None

    for j in range(num_candidates):
        if j == 0 and eps_prev is not None and num_candidates > 1:
            eps = eps_prev
        else:
            eps = torch.randn_like(x0_target)

        x_t = sqrt_at * x0_target + sqrt_1mat * eps
        t_tensor = torch.tensor([t_int], device=x0_target.device, dtype=torch.long)
        eps_pred = unet(x_t, t_tensor).sample
        x0_hat = (x_t - sqrt_1mat * eps_pred) / sqrt_at

        if score_radius is None:
            score = oversampled_mag_l2(x0_hat, mag_target, pad)
        else:
            score = oversampled_lowfreq_mag_l2(x0_hat, mag_target, pad, score_radius)

        if best_score is None or float(score) < float(best_score):
            best_score = score
            best_x0 = x0_hat
            best_eps = eps.clone()

    assert best_x0 is not None and best_eps is not None
    return best_x0, best_eps


@torch.no_grad()
def noise_picking_reconstruct_oversampled(
    mag_target: torch.Tensor,
    *,
    pad: int,
    seed: int,
    unet,
    scheduler,
    device: torch.device,
    variant: NPVariant,
    num_steps: int,
    score_radius: float,
    proj_radius: float,
    log_every: int,
    proj_radius_schedule: str | None = None,
) -> torch.Tensor:
    seed_everything(seed)
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    proj_schedule = parse_radius_schedule(proj_radius_schedule, proj_radius)

    x_t = torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)
    x_prev = None
    eps_prev = None

    for i in range(len(timesteps) - 1):
        t_int = int(timesteps[i])
        t_next_int = int(timesteps[i + 1])

        if i == 0:
            t_tensor = torch.tensor([t_int], device=device, dtype=torch.long)
            eps = unet(x_t, t_tensor).sample
            alpha_bar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=x_t.dtype)
            sqrt_ab = torch.sqrt(alpha_bar_t)
            sqrt_1mab = torch.sqrt(1.0 - alpha_bar_t)
            x0_hat = (x_t - sqrt_1mab * eps) / sqrt_ab
        else:
            assert x_prev is not None
            x0_hat = x_prev

        if variant.use_lowfreq_projection and i >= variant.proj_start:
            current_proj_radius = radius_at_step(proj_schedule, i)
            x0_hat = enforce_oversampled_lowfreq(
                x0_hat, mag_target, pad, current_proj_radius
            )

        k = variant.soft if i < variant.proj_start else variant.hard
        x_prev, eps_prev = pick_noise_oversampled(
            x0_target=x0_hat,
            t_int=t_next_int,
            eps_prev=eps_prev,
            mag_target=mag_target,
            pad=pad,
            num_candidates=k,
            score_radius=score_radius if variant.use_lowfreq_score else None,
            unet=unet,
            scheduler=scheduler,
        )

        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"[{variant.name}] step {i+1}/{len(timesteps)-1}")

    assert x_prev is not None
    return x_prev.detach()


def best_rot180_channel_alignment(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Choose the best per-channel 180-degree rotation by MSE.

    DiffFPR code contains a similar ambiguity-handling routine using per-channel
    180-degree rotations.  This implementation reports both raw and aligned PSNR,
    and uses MSE to avoid adding a new SSIM dependency.
    """
    candidates: List[torch.Tensor] = []
    chans = [x[:, c : c + 1] for c in range(x.shape[1])]
    for mask in range(1 << x.shape[1]):
        parts = []
        for c, xc in enumerate(chans):
            if (mask >> c) & 1:
                parts.append(torch.rot90(xc, 2, dims=(-2, -1)))
            else:
                parts.append(xc)
        candidates.append(torch.cat(parts, dim=1))
    mses = [torch.mean((to01(cand) - to01(ref)) ** 2) for cand in candidates]
    idx = int(torch.argmin(torch.stack(mses)).item())
    return candidates[idx]


def psnr01_from_model_range(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((to01(x) - to01(y)) ** 2).clamp_min(eps)
    return float((10.0 * torch.log10(1.0 / mse)).cpu().item())


def summarize_image_level(run_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in run_rows:
        key = (str(row["variant"]), str(row["image_basename"]))
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (variant, image), rows in sorted(grouped.items()):
        psnrs = torch.tensor([float(r["aligned_psnr"]) for r in rows])
        raw_psnrs = torch.tensor([float(r["raw_psnr"]) for r in rows])
        times = torch.tensor([float(r["runtime_s"]) for r in rows])
        full_clean = torch.tensor([float(r["clean_mag_l2"]) for r in rows])
        low_clean = torch.tensor([float(r["clean_lowfreq_mag_l2"]) for r in rows])
        out.append(
            {
                "variant": variant,
                "image_basename": image,
                "n_runs": len(rows),
                "aligned_psnr_mean": float(psnrs.mean().item()),
                "aligned_psnr_median": float(psnrs.median().item()),
                "aligned_psnr_max": float(psnrs.max().item()),
                "raw_psnr_mean": float(raw_psnrs.mean().item()),
                "runtime_s_mean": float(times.mean().item()),
                "clean_mag_l2_mean": float(full_clean.mean().item()),
                "clean_lowfreq_mag_l2_mean": float(low_clean.mean().item()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run np_canonical and np_fixedk_lateproj under DiffFPR-style oversampled Fourier PR."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--image_list_file", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--paper_preset", choices=sorted(PAPER_PRESETS.keys()), default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--variants", default="np_canonical,np_fixedk_lateproj")
    parser.add_argument("--seeds", default="100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119")
    parser.add_argument("--np_steps", type=int, default=1000)
    parser.add_argument("--late_start", type=int, default=400)
    parser.add_argument("--fixed_k", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--oversample", type=float, default=None)
    parser.add_argument("--measurement_noise_std", type=float, default=None)
    parser.add_argument("--measurement_noise_seed", type=int, default=20260423)
    parser.add_argument("--clip_noisy_magnitude", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()
    preset = PAPER_PRESETS.get(args.paper_preset) if args.paper_preset else None

    image_names = collect_images(args.data_root, args.image_list_file)
    if not image_names:
        raise ValueError(f"No images found under {args.data_root}")
    seeds = parse_int_list(args.seeds)
    variant_names = [tok.strip() for tok in args.variants.split(",") if tok.strip()]
    variants = make_variants(variant_names, late_start=args.late_start, fixed_k=args.fixed_k)

    model_id = args.model_id if args.model_id else (
        preset.model_id if preset is not None else "google/ddpm-celebahq-256"
    )
    oversample = float(args.oversample) if args.oversample is not None else (
        preset.oversample if preset is not None else 4.0
    )
    measurement_noise_std = (
        float(args.measurement_noise_std)
        if args.measurement_noise_std is not None
        else (preset.noise_levels[-1] if preset is not None else 0.05)
    )

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"difffpr_np_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(model_id, device=device)
    image_size = int(bundle.unet.config.sample_size)
    if preset is not None and image_size != preset.image_size:
        raise ValueError(
            f"paper preset {preset.name!r} expects {preset.image_size}x{preset.image_size}, "
            f"but model {model_id!r} has sample_size={image_size}"
        )
    pad = oversample_pad(image_size, oversample)

    config_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for variant in variants:
        config_rows.append(
            {
                "variant": variant.name,
                "paper_preset": preset.name if preset is not None else "",
                "model_id": model_id,
                "num_steps": args.np_steps,
                "score_radius": args.radius,
                "proj_radius": args.radius,
                "proj_start": variant.proj_start,
                "num_candidates_soft": variant.soft,
                "num_candidates_hard": variant.hard,
                "use_lowfreq_score": variant.use_lowfreq_score,
                "use_lowfreq_projection": variant.use_lowfreq_projection,
                "measurement_operator": "DiffFPR-style centered FFT magnitude after symmetric zero padding",
                "oversample_arg": oversample,
                "pad_pixels_each_side": pad,
                "measurement_noise_std": measurement_noise_std,
                "clip_noisy_magnitude": bool(args.clip_noisy_magnitude),
                "seeds": ",".join(map(str, seeds)),
                "preset_note": preset.note if preset is not None else "",
            }
        )

    for image_index, image_name in enumerate(image_names):
        img_path = resolve_image_path(args.data_root, image_name)
        x_gt = load_image(img_path, size=image_size, device=device)
        mag_clean = oversampled_magnitude(x_gt, pad)

        mag_target = mag_clean
        if measurement_noise_std > 0:
            gen = torch.Generator(device=device).manual_seed(args.measurement_noise_seed + image_index)
            noise = torch.randn(mag_clean.shape, device=device, dtype=mag_clean.dtype, generator=gen)
            mag_target = mag_clean + measurement_noise_std * noise
            if args.clip_noisy_magnitude:
                mag_target = mag_target.clamp_min(0.0)

        for variant in variants:
            for seed in seeds:
                t0 = time.perf_counter()
                x_rec = noise_picking_reconstruct_oversampled(
                    mag_target,
                    pad=pad,
                    seed=seed,
                    unet=bundle.unet,
                    scheduler=bundle.scheduler,
                    device=device,
                    variant=variant,
                    num_steps=args.np_steps,
                    score_radius=args.radius,
                    proj_radius=args.radius,
                    log_every=args.log_every,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                runtime_s = time.perf_counter() - t0

                x_aligned = best_rot180_channel_alignment(x_rec, x_gt)
                row = {
                    "timestamp": stamp,
                    "variant": variant.name,
                    "image_basename": image_name,
                    "seed": seed,
                    "raw_psnr": psnr01_from_model_range(x_rec, x_gt),
                    "aligned_psnr": psnr01_from_model_range(x_aligned, x_gt),
                    "clean_mag_l2": float(oversampled_mag_l2(x_aligned, mag_clean, pad).cpu().item()),
                    "noisy_mag_l2": float(oversampled_mag_l2(x_aligned, mag_target, pad).cpu().item()),
                    "clean_lowfreq_mag_l2": float(
                        oversampled_lowfreq_mag_l2(x_aligned, mag_clean, pad, args.radius).cpu().item()
                    ),
                    "noisy_lowfreq_mag_l2": float(
                        oversampled_lowfreq_mag_l2(x_aligned, mag_target, pad, args.radius).cpu().item()
                    ),
                    "runtime_s": runtime_s,
                    "num_steps": args.np_steps,
                    "proj_start": variant.proj_start,
                    "num_candidates_soft": variant.soft,
                    "num_candidates_hard": variant.hard,
                    "radius": args.radius,
                    "oversample": oversample,
                    "pad_pixels_each_side": pad,
                    "measurement_noise_std": measurement_noise_std,
                }
                run_rows.append(row)
                print(
                    f"[DiffFPR-NP] {variant.name} | {image_name} seed={seed} "
                    f"aligned_psnr={row['aligned_psnr']:.2f} raw_psnr={row['raw_psnr']:.2f} "
                    f"({runtime_s:.1f}s)"
                )

    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "image_level_summary.csv"), summarize_image_level(run_rows))
    print(f"Saved: {run_root}")


if __name__ == "__main__":
    main()
