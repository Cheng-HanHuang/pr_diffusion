from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import UNet2DModel, DDPMScheduler

from ..fft_ops import fft2c, ifft2c, lowfreq_mask
from ..metrics import lowfreq_mag_l2, mag_l2
from ..seed import seed_everything


@dataclass
class NoisePickingConfig:
    num_steps: int = 1000
    score_radius: float = 0.02          # ablation knob
    proj_radius: float = 0.3
    proj_start: int = 400
    num_candidates_soft: int = 5
    num_candidates_hard: int = 2
    log_every: int = 100
    use_lowfreq_score: bool = True
    use_lowfreq_projection: bool = True


def enforce_magnitude_lowfreq(
    x: torch.Tensor,
    mag_target: torch.Tensor,
    radius: float,
    *,
    eps: float = 1e-8,
    num_iter: int = 1,
) -> torch.Tensor:
    """Replace the Fourier magnitude of x with mag_target on low frequencies, keep phase."""
    B, C, H, W = x.shape
    mask_hw = lowfreq_mask(H, W, radius=radius, device=x.device)  # [H,W]
    mask = mask_hw[None, None, :, :]  # [1,1,H,W]
    X = fft2c(x)
    x_new = x
    for _ in range(num_iter):
        mag_X = torch.abs(X)
        phase = X / (mag_X + eps)
        new_mag = torch.where(mask, mag_target, mag_X)
        X = phase * new_mag
        x_new = ifft2c(X)
        X = fft2c(x_new)
    return x_new


@torch.no_grad()
def pick_noise(
    *,
    x0_target: torch.Tensor,
    t_int: int,
    eps_prev: Optional[torch.Tensor],
    mag_target: torch.Tensor,
    num_candidates: int,
    score_radius: Optional[float],
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick epsilon among candidates for timestep t_int by low-frequency magnitude score."""
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

        if score_radius is not None:
            score = lowfreq_mag_l2(x0_hat, mag_target, radius=score_radius)
        else:
            score = mag_l2(x0_hat, mag_target)

        if best_score is None or float(score) < float(best_score):
            best_score = score
            best_x0 = x0_hat
            best_eps = eps.clone()

    assert best_x0 is not None and best_eps is not None
    return best_x0, best_eps


@torch.no_grad()
def noise_picking_reconstruct(
    mag_target: torch.Tensor,
    *,
    seed: int,
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
    cfg: NoisePickingConfig,
) -> torch.Tensor:
    """Reconstruction via candidate-noise picking (your Method A)."""
    seed_everything(seed)
    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps

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

        # late projection
        if cfg.use_lowfreq_projection and i >= cfg.proj_start:
            x0_hat = enforce_magnitude_lowfreq(x0_hat, mag_target, cfg.proj_radius)

        k = cfg.num_candidates_soft if i < cfg.proj_start else cfg.num_candidates_hard

        x_prev, eps_prev = pick_noise(
            x0_target=x0_hat,
            t_int=t_next_int,
            eps_prev=eps_prev,
            mag_target=mag_target,
            num_candidates=k,
            score_radius=cfg.score_radius if cfg.use_lowfreq_score else None,
            unet=unet,
            scheduler=scheduler,
        )

        if cfg.log_every > 0 and (i + 1) % cfg.log_every == 0:
            print(f"[NoisePicking] step {i+1}/{len(timesteps)-1}")

    assert x_prev is not None
    return x_prev.detach()
