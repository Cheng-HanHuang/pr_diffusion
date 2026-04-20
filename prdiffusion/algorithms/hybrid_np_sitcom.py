from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DDPMScheduler, UNet2DModel

from ..diffusion import resample_x_t, tweedie_x0_from_v
from ..metrics import lowfreq_mag_l2, mag_l2, mag_mse
from ..seed import seed_everything
from .noise_picking import pick_noise, enforce_magnitude_lowfreq


@dataclass
class HybridNPSitcomConfig:
    num_steps: int = 1000
    switch_timestep: int = 400

    # NP (early phase)
    np_score_radius: float = 0.5
    np_proj_radius: float = 0.5
    np_proj_start: int = 400
    np_num_candidates_soft: int = 5
    np_num_candidates_hard: int = 1
    np_use_lowfreq_score: bool = True
    np_use_lowfreq_projection: bool = True

    # SITCOM (late phase)
    sitcom_K: int = 20
    sitcom_lr_inner: float = 0.02
    sitcom_lam: float = 0.1
    sitcom_eta_scale: float = 1.0
    sitcom_backprop_unet: bool = True
    sitcom_inner_optim: str = "adam"
    sitcom_meas_radius: Optional[float] = None
    sitcom_stop_meas_mse: Optional[float] = None
    sitcom_stop_meas_l2: Optional[float] = None

    log_every: int = 100


def _nearest_timestep_at_or_below(timesteps: torch.Tensor, threshold: int) -> int:
    for t in timesteps:
        t_int = int(t)
        if t_int <= threshold:
            return t_int
    return int(timesteps[-1])


@torch.no_grad()
def _noise_picking_prefix(
    mag_target: torch.Tensor,
    *,
    seed: int,
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
    cfg: HybridNPSitcomConfig,
) -> tuple[torch.Tensor, int]:
    seed_everything(seed)
    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps

    x_t = torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)
    x_prev = None
    eps_prev = None

    switch_t = _nearest_timestep_at_or_below(timesteps, cfg.switch_timestep)

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

        if cfg.np_use_lowfreq_projection and i >= cfg.np_proj_start:
            x0_hat = enforce_magnitude_lowfreq(x0_hat, mag_target, cfg.np_proj_radius)

        k = cfg.np_num_candidates_soft if i < cfg.np_proj_start else cfg.np_num_candidates_hard

        x_prev, eps_prev = pick_noise(
            x0_target=x0_hat,
            t_int=t_next_int,
            eps_prev=eps_prev,
            mag_target=mag_target,
            num_candidates=k,
            score_radius=cfg.np_score_radius if cfg.np_use_lowfreq_score else None,
            unet=unet,
            scheduler=scheduler,
        )

        if cfg.log_every > 0 and (i + 1) % cfg.log_every == 0:
            print(f"[Hybrid NP prefix] step {i+1}/{len(timesteps)-1} | t_next={t_next_int}")

        if t_next_int <= switch_t:
            assert x_prev is not None
            return x_prev.detach(), t_next_int

    assert x_prev is not None
    return x_prev.detach(), int(timesteps[-1])


def np_to_sitcom_hybrid_reconstruct(
    mag_target: torch.Tensor,
    *,
    seed: int,
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
    cfg: HybridNPSitcomConfig,
) -> torch.Tensor:
    """Hybrid reconstruction: NP prefix, then SITCOM from switch timestep onward."""
    x0_switch, start_t = _noise_picking_prefix(
        mag_target,
        seed=seed,
        unet=unet,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
    )

    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps

    start_idx = None
    for i, t in enumerate(timesteps):
        if int(t) <= start_t:
            start_idx = i
            break
    if start_idx is None:
        start_idx = len(timesteps) - 1

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 12345)

    # handoff: convert NP x0 estimate into a scheduler-consistent x_t latent at switch timestep.
    x_t = resample_x_t(x0_switch, start_t, scheduler=scheduler, eta_scale=cfg.sitcom_eta_scale, generator=gen)

    for i in range(start_idx, len(timesteps) - 1):
        t_int = int(timesteps[i])
        t_next_int = int(timesteps[i + 1])

        v = torch.nn.Parameter(x_t.detach().clone())
        if cfg.sitcom_inner_optim.lower() == "adam":
            opt = torch.optim.Adam([v], lr=cfg.sitcom_lr_inner)
        elif cfg.sitcom_inner_optim.lower() == "sgd":
            opt = torch.optim.SGD([v], lr=cfg.sitcom_lr_inner)
        else:
            raise ValueError(f"Unknown sitcom_inner_optim={cfg.sitcom_inner_optim}")

        for _ in range(cfg.sitcom_K):
            opt.zero_grad(set_to_none=True)
            x0_v = tweedie_x0_from_v(v, t_int, unet=unet, scheduler=scheduler, backprop_unet=cfg.sitcom_backprop_unet)

            if cfg.sitcom_meas_radius is None:
                loss_meas = mag_mse(x0_v, mag_target)
                meas_l2 = mag_l2(x0_v, mag_target)
            else:
                meas_l2 = lowfreq_mag_l2(x0_v, mag_target, cfg.sitcom_meas_radius)
                loss_meas = meas_l2.pow(2)

            loss_reg = cfg.sitcom_lam * torch.mean((v - x_t) ** 2)
            loss = loss_meas + loss_reg
            loss.backward()
            opt.step()

            if cfg.sitcom_stop_meas_l2 is not None and float(meas_l2.detach().cpu()) <= float(cfg.sitcom_stop_meas_l2):
                break
            if cfg.sitcom_stop_meas_mse is not None and float(loss_meas.detach().cpu()) <= float(cfg.sitcom_stop_meas_mse):
                break

        x0_hat = tweedie_x0_from_v(v.detach(), t_int, unet=unet, scheduler=scheduler, backprop_unet=False).detach()
        x_t = resample_x_t(x0_hat, t_next_int, scheduler=scheduler, eta_scale=cfg.sitcom_eta_scale, generator=gen)

        if cfg.log_every > 0 and (i + 1) % cfg.log_every == 0:
            print(f"[Hybrid SITCOM suffix] step {i+1}/{len(timesteps)-1} | t={t_int}")

    t_last = int(timesteps[-1])
    return tweedie_x0_from_v(x_t, t_last, unet=unet, scheduler=scheduler, backprop_unet=False).detach()
