from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from diffusers import UNet2DModel, DDPMScheduler


@dataclass(frozen=True)
class ModelBundle:
    unet: UNet2DModel
    scheduler: DDPMScheduler


def load_model(model_id: str, device: torch.device) -> ModelBundle:
    """Load pretrained UNet + scheduler from diffusers."""
    unet = UNet2DModel.from_pretrained(model_id).to(device)
    scheduler = DDPMScheduler.from_pretrained(model_id)
    return ModelBundle(unet=unet, scheduler=scheduler)


def tweedie_x0_from_v(
    v: torch.Tensor,
    t_int: int,
    *,
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    backprop_unet: bool,
) -> torch.Tensor:
    """Compute x0 estimate from noisy input v at timestep t via Tweedie formula.

    x0 = (v - sqrt(1 - alpha_bar_t) * eps_theta(v,t)) / sqrt(alpha_bar_t)

    If backprop_unet=True, gradients flow through eps_theta.
    """
    a = scheduler.alphas_cumprod[t_int].to(device=v.device, dtype=v.dtype)
    sqrt_a = torch.sqrt(a)
    sqrt_oma = torch.sqrt(1.0 - a)

    t_tensor = torch.tensor([t_int], device=v.device, dtype=torch.long)

    if backprop_unet:
        eps = unet(v, t_tensor).sample
    else:
        with torch.no_grad():
            eps = unet(v, t_tensor).sample

    return (v - sqrt_oma * eps) / sqrt_a


def resample_x_t(
    x0_hat: torch.Tensor,
    t_next_int: int,
    *,
    scheduler: DDPMScheduler,
    eta_scale: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Resample x_{t_next} = sqrt(alpha_bar_{t_next}) x0_hat + sqrt(1-alpha_bar_{t_next}) * eta."""
    a_next = scheduler.alphas_cumprod[t_next_int].to(device=x0_hat.device, dtype=x0_hat.dtype)
    sqrt_a_next = torch.sqrt(a_next)
    sqrt_oma_next = torch.sqrt(1.0 - a_next)
    eta = torch.randn_like(x0_hat, generator=generator) * float(eta_scale)
    return sqrt_a_next * x0_hat + sqrt_oma_next * eta
