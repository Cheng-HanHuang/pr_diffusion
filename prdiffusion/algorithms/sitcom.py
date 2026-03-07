from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from diffusers import UNet2DModel, DDPMScheduler

from ..diffusion import tweedie_x0_from_v, resample_x_t
from ..metrics import mag_mse, mag_l2, lowfreq_mag_l2
from ..seed import seed_everything


@dataclass
class SitcomConfig:
    num_steps: int = 1000          # number of diffusion steps used by scheduler.set_timesteps
    K: int = 20                    # inner optimization steps per outer timestep
    lr_inner: float = 0.05         # "gamma" / inner step size
    lam: float = 0.1               # closeness regularizer weight
    eta_scale: float = 1.0         # per-step re-noise scaling
    init_scale: float = 1.0        # initial latent scaling
    log_every: int = 200

    # If set, compares magnitudes only on low frequencies within radius.
    meas_radius: Optional[float] = None

    # If set, early-stop inner loop when measurement MSE falls below this threshold.
    # Useful when measurement noise is nonzero (prevents overfitting).
    stop_meas_mse: Optional[float] = None

    # Paper-faithful early-stop threshold on L2 measurement residual.
    # When set, this takes precedence over stop_meas_mse.
    stop_meas_l2: Optional[float] = None

    # If True, flow gradients through the denoiser (paper-faithful SITCOM).
    backprop_unet: bool = True

    # Optimizer for inner loop: "adam" or "sgd"
    inner_optim: str = "adam"


def sitcom_reconstruct(
    mag_target: torch.Tensor,
    *,
    seed: int,
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
    cfg: SitcomConfig,
) -> torch.Tensor:
    """SITCOM reconstruction from magnitude-only measurements.

    Outer loop: iterate timesteps.
    (S1) Optimize v_t (starting from x_t) to enforce measurement consistency + closeness.
    (S2) x0_hat = Tweedie(v_hat).
    (S3) Resample to next timestep.

    Returns:
        Final x0 estimate at the last timestep.
    """
    seed_everything(seed)
    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps  # descending (e.g. 999 ... 0)

    # init x_t ~ N(0, I)
    x_t = cfg.init_scale * torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)

    # create RNG for reproducible resampling noise
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 12345)

    for i in range(len(timesteps) - 1):
        t_int = int(timesteps[i])
        t_next_int = int(timesteps[i + 1])

        # (S1) optimize v
        v = torch.nn.Parameter(x_t.detach().clone())

        if cfg.inner_optim.lower() == "adam":
            opt = torch.optim.Adam([v], lr=cfg.lr_inner)
        elif cfg.inner_optim.lower() == "sgd":
            opt = torch.optim.SGD([v], lr=cfg.lr_inner)
        else:
            raise ValueError(f"Unknown inner_optim={cfg.inner_optim}")

        for _k in range(cfg.K):
            opt.zero_grad(set_to_none=True)

            x0_v = tweedie_x0_from_v(
                v, t_int, unet=unet, scheduler=scheduler, backprop_unet=cfg.backprop_unet
            )

            # measurement loss
            if cfg.meas_radius is None:
                loss_meas = mag_mse(x0_v, mag_target)
                meas_l2 = mag_l2(x0_v, mag_target)
            else:
                # squared lowfreq l2 (matches your notebook pattern)
                meas_l2 = lowfreq_mag_l2(x0_v, mag_target, cfg.meas_radius)
                loss_meas = meas_l2.pow(2)

            loss_reg = cfg.lam * torch.mean((v - x_t) ** 2)
            loss = loss_meas + loss_reg
            loss.backward()
            opt.step()

            # Paper-faithful stop criterion uses L2 residual threshold.
            if cfg.stop_meas_l2 is not None:
                if float(meas_l2.detach().cpu()) <= float(cfg.stop_meas_l2):
                    break

            # Backward-compatible variant using MSE threshold.
            if cfg.stop_meas_mse is not None:
                if float(loss_meas.detach().cpu()) <= float(cfg.stop_meas_mse):
                    break

        v_hat = v.detach()

        # (S2)
        x0_hat = tweedie_x0_from_v(
            v_hat, t_int, unet=unet, scheduler=scheduler, backprop_unet=False
        ).detach()

        # (S3)
        x_t = resample_x_t(
            x0_hat, t_next_int, scheduler=scheduler, eta_scale=cfg.eta_scale, generator=gen
        )

        if cfg.log_every > 0 and (i + 1) % cfg.log_every == 0:
            with torch.no_grad():
                me = float(torch.norm(torch.abs(torch.fft.fft2(torch.complex(x0_hat, torch.zeros_like(x0_hat)), norm="ortho")) - mag_target).item())
            print(f"[SITCOM] step {i+1}/{len(timesteps)-1} | magerr(l2)={me:.4g}")

    # Final output: x0 at last timestep
    t_last = int(timesteps[-1])
    x0_final = tweedie_x0_from_v(
        x_t, t_last, unet=unet, scheduler=scheduler, backprop_unet=False
    ).detach()
    return x0_final
