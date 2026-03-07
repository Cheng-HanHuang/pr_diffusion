from __future__ import annotations

import torch
import torch.nn.functional as F

from .fft_ops import magnitude


def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """PSNR in dB for images in [-1, 1]."""
    x01 = (x.clamp(-1, 1) + 1) / 2
    y01 = (y.clamp(-1, 1) + 1) / 2
    mse = torch.mean((x01 - y01) ** 2).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def mag_l2(x: torch.Tensor, mag_target: torch.Tensor) -> torch.Tensor:
    """L2 norm of magnitude residual."""
    return torch.norm(magnitude(x) - mag_target)


def mag_mse(x: torch.Tensor, mag_target: torch.Tensor) -> torch.Tensor:
    """Mean squared magnitude residual."""
    return torch.mean((magnitude(x) - mag_target) ** 2)


def lowfreq_mag_l2(x: torch.Tensor, mag_target: torch.Tensor, radius: float) -> torch.Tensor:
    """Low-frequency-only L2 magnitude residual."""
    B, C, H, W = x.shape
    from .fft_ops import lowfreq_mask
    mask_hw = lowfreq_mask(H, W, radius=radius, device=x.device)  # [H,W]
    mask = mask_hw[None, None, :, :].repeat(B, C, 1, 1)
    return torch.norm(magnitude(x)[mask] - mag_target[mask])
