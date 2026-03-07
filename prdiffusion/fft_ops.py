from __future__ import annotations

import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D FFT on last two dims, treating x as real-valued image.

    Returns a complex tensor shaped like x.
    """
    x_c = torch.complex(x, torch.zeros_like(x))
    return torch.fft.fft2(x_c, dim=(-2, -1), norm="ortho")


def ifft2c(X: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D inverse FFT on last two dims; returns real part."""
    x_c = torch.fft.ifft2(X, dim=(-2, -1), norm="ortho")
    return x_c.real


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """Fourier magnitude |FFT(x)|."""
    return torch.abs(fft2c(x))


def freq_radius(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Frequency radius grid using torch.fft.fftfreq in cycles/pixel."""
    fy = torch.fft.fftfreq(h, d=1.0, device=device)
    fx = torch.fft.fftfreq(w, d=1.0, device=device)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(ky**2 + kx**2)  # [H, W]


def lowfreq_mask(h: int, w: int, radius: float, device: torch.device) -> torch.Tensor:
    """Mask (H, W) for low-frequency disk defined by radius in fftfreq units."""
    fr = freq_radius(h, w, device=device)
    return fr <= radius
