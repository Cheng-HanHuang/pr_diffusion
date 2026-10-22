"""Device-agnostic frozen R0 Fourier magnitude, RGB8 and exact channel orbit."""
from __future__ import annotations

import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ORBIT_PATTERNS = tuple(format(i, "03b") for i in range(8))


def _check_image(x):
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1:] != (3, 256, 256):
        raise ValueError("expected image [B,3,256,256]")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("expected finite floating-point image")


def magnitude(x):
    """DAPS PhaseRetrieval A(x)=|fftshift(fftn(ifftshift(pad((x+1)/2))))|."""
    _check_image(x)
    z = F.pad((x + 1.0) * 0.5, (64, 64, 64, 64)).to(torch.complex64)
    return torch.fft.fftshift(torch.fft.fftn(
        torch.fft.ifftshift(z, dim=(-2, -1)),
        dim=(-2, -1), norm="ortho"), dim=(-2, -1)).abs()


def _residual(x, y, clip):
    _check_image(x)
    if not isinstance(y, torch.Tensor) or y.shape != (x.shape[0], 3, 384, 384):
        raise ValueError("expected y [B,3,384,384] paired with x")
    if x.device != y.device or not bool(torch.isfinite(y).all()):
        raise ValueError("y device or finiteness mismatch")
    target = y.clamp_min(0) if clip else y
    delta = magnitude(x) - target
    return torch.sqrt(delta.double().square().flatten(1).mean(dim=1)).float()


def rho_signed(x, y):
    return _residual(x, y, False)


def rho_clipped(x, y):
    return _residual(x, y, True)


def to_rgb8(x):
    """Quantize float model range to canonical RGB8 [B,3,256,256]."""
    _check_image(x)
    return torch.round((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)


def from_rgb8(x8):
    if not isinstance(x8, torch.Tensor) or x8.dtype != torch.uint8 or x8.ndim != 4 or x8.shape[1:] != (3, 256, 256):
        raise ValueError("RGB8 must be uint8 [B,3,256,256]")
    return x8.float() / 127.5 - 1.0


def load_png_rgb8(path):
    """Return uint8 [1,3,256,256], same PIL conversion as A1.2."""
    with Image.open(path) as im:
        if im.size != (256, 256):
            raise ValueError("expected 256x256 PNG")
        a = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()


def psnr_rgb8(x8, g8):
    if not isinstance(x8, torch.Tensor) or not isinstance(g8, torch.Tensor):
        raise ValueError("expected RGB8 tensors")
    if x8.dtype != torch.uint8 or g8.dtype != torch.uint8 or x8.shape != g8.shape:
        raise ValueError("RGB8 shape/dtype mismatch")
    mse = ((x8.float() - g8.float()) / 255.0).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def apply_orbit(x, pattern):
    """Per-channel 180-degree rotations in R,G,B bit order."""
    if pattern not in ORBIT_PATTERNS:
        raise ValueError("invalid orbit pattern")
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("orbit expects [B,3,H,W]")
    if pattern == "000":
        return x.clone()
    channels = [torch.rot90(x[:, i:i+1], 2, (-2, -1))
                if bit == "1" else x[:, i:i+1]
                for i, bit in enumerate(pattern)]
    return torch.cat(channels, dim=1)


def centered_fft_witness(device="cpu"):
    x = torch.full((1,3,256,256), -1, dtype=torch.float32, device=device)
    x[:, :, 128, 128] = 1.0
    got = magnitude(x)
    target = 1.0 / 384.0
    err = float((got - target).abs().max())
    zero = magnitude(torch.full_like(x, -1.0))
    if err > 2e-7 or float(zero.abs().max()) != 0:
        raise AssertionError("centered FFT witness failed")
    return {"impulse_max_absolute_error": err, "zero_max": float(zero.max()),
            "expected_impulse": target}


def orbit_invariance_relative(x):
    """Maximum absolute orbit error normalized by maximum baseline A(x)."""
    baseline = magnitude(x)
    scale = max(float(baseline.abs().max()), 1e-12)
    return {p: float((magnitude(apply_orbit(x,p)) - baseline).abs().max()) / scale
            for p in ORBIT_PATTERNS}
