from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class GuidedModelBundle:
    """Small compatibility bundle matching the diffusers backend used by NP.

    The current NP implementations only need:

    - ``bundle.unet(x_t, t).sample`` returning an epsilon prediction;
    - ``bundle.unet.config.sample_size``;
    - ``bundle.scheduler.alphas_cumprod``;
    - ``bundle.scheduler.set_timesteps(num_steps, device=...)``.

    This adapter exposes those fields for OpenAI guided-diffusion checkpoints such
    as the FFHQ ``ffhq_10m.pt`` model used by DPS/DiffPIR/DiffFPR-style code.
    """

    unet: torch.nn.Module
    scheduler: "GuidedDiffusionSchedulerAdapter"
    model_config: Dict[str, Any]


class GuidedDiffusionUNetAdapter(torch.nn.Module):
    """Wrap a guided-diffusion UNet so it looks like a diffusers UNet2DModel."""

    def __init__(self, model: torch.nn.Module, *, sample_size: int, learn_sigma: bool):
        super().__init__()
        self.model = model
        self.learn_sigma = bool(learn_sigma)
        self.config = SimpleNamespace(sample_size=int(sample_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        out = self.model(x, t)
        # Guided-diffusion models with learn_sigma=True output [eps, variance].
        # For NP's Tweedie update we only need epsilon, matching GaussianDiffusion's
        # p_mean_variance split convention.
        if out.shape[1] > x.shape[1]:
            out = out[:, : x.shape[1], ...]
        return SimpleNamespace(sample=out)


class GuidedDiffusionSchedulerAdapter:
    """Minimal scheduler adapter for the NP code path."""

    def __init__(self, *, alphas_cumprod, train_num_steps: int = 1000):
        self.train_num_steps = int(train_num_steps)
        self._alphas_np = alphas_cumprod
        self.alphas_cumprod = torch.as_tensor(alphas_cumprod, dtype=torch.float32)
        self.timesteps = torch.arange(self.train_num_steps - 1, -1, -1, dtype=torch.long)

    def set_timesteps(self, num_steps: int, device: Optional[torch.device] = None):
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if num_steps == self.train_num_steps:
            ts = torch.arange(self.train_num_steps - 1, -1, -1, dtype=torch.long)
        else:
            # Diffusers uses an evenly-spaced subset for smaller step counts.  The
            # main FFHQ benchmark uses 1000 steps, but this keeps reduced pilots valid.
            ts = torch.linspace(self.train_num_steps - 1, 0, num_steps).round().long()
            ts = torch.unique_consecutive(ts)
            if ts[-1].item() != 0:
                ts = torch.cat([ts, torch.zeros(1, dtype=torch.long)])
        if device is not None:
            ts = ts.to(device)
            self.alphas_cumprod = torch.as_tensor(self._alphas_np, dtype=torch.float32, device=device)
        self.timesteps = ts
        return self.timesteps


def _add_guided_diffusion_to_path(guided_diffusion_dir: Optional[str]) -> None:
    if not guided_diffusion_dir:
        return
    p = Path(guided_diffusion_dir).expanduser().resolve()
    # Accept either the repository root containing guided_diffusion/, or the
    # package directory itself.
    if (p / "guided_diffusion").is_dir():
        sys.path.insert(0, str(p))
    elif p.name == "guided_diffusion" and p.is_dir():
        sys.path.insert(0, str(p.parent))
    else:
        raise FileNotFoundError(
            f"Could not find guided_diffusion package under {guided_diffusion_dir!r}. "
            "Pass the DiffFPR/DPS/OpenAI guided-diffusion repo root, or set PYTHONPATH."
        )


def guided_model_config(preset: str) -> Dict[str, Any]:
    """Return known guided-diffusion checkpoint architecture settings."""
    if preset == "difffpr_ffhq_10m":
        # Matches DiffFPR/DiffPIR utility defaults for ffhq_10m.pt:
        # image_size=256, num_channels=128, num_res_blocks=1,
        # attention_resolutions=16, learn_sigma=True, dropout=0.1,
        # resblock_updown=True, use_scale_shift_norm=True, num_head_channels=64.
        return dict(
            image_size=256,
            class_cond=False,
            learn_sigma=True,
            num_channels=128,
            num_res_blocks=1,
            channel_mult="",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            attention_resolutions="16",
            dropout=0.1,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
        )
    raise ValueError(
        f"Unknown guided model preset {preset!r}. Currently supported: difffpr_ffhq_10m"
    )


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "ema", "ema_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        if obj and all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise TypeError(
        "Could not identify a PyTorch state_dict in checkpoint. Expected either a "
        "plain state_dict or a dict containing one of: state_dict, model, "
        "model_state_dict, ema, ema_state_dict."
    )


def load_guided_diffusion_model(
    *,
    model_path: str,
    device: torch.device,
    preset: str = "difffpr_ffhq_10m",
    guided_diffusion_dir: Optional[str] = None,
    strict: bool = True,
) -> GuidedModelBundle:
    """Load a guided-diffusion checkpoint behind the NP-compatible interface.

    Parameters
    ----------
    model_path:
        Path to a checkpoint such as ``ffhq_10m.pt``.
    device:
        Target torch device.
    preset:
        Architecture preset.  Use ``difffpr_ffhq_10m`` for the FFHQ checkpoint
        used by DiffFPR/DiffPIR/DPS code paths.
    guided_diffusion_dir:
        Optional local repo root containing the ``guided_diffusion`` package.
        This can point to a cloned DiffFPR repo, DPS repo, or OpenAI
        guided-diffusion repo.
    strict:
        Whether to load the checkpoint with strict state-dict matching.
    """
    _add_guided_diffusion_to_path(guided_diffusion_dir)

    try:
        from guided_diffusion.script_util import create_model_and_diffusion
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "Could not import guided_diffusion.script_util. Clone DiffFPR or "
            "DPS/openai guided-diffusion and pass --guided_diffusion_dir, or set PYTHONPATH."
        ) from exc

    cfg = guided_model_config(preset)
    model, diffusion = create_model_and_diffusion(**cfg)

    ckpt_path = Path(model_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Guided-diffusion checkpoint not found: {ckpt_path}")
    state = _extract_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.load_state_dict(state, strict=strict)
    model.eval().to(device)

    unet = GuidedDiffusionUNetAdapter(
        model,
        sample_size=int(cfg["image_size"]),
        learn_sigma=bool(cfg["learn_sigma"]),
    )
    scheduler = GuidedDiffusionSchedulerAdapter(
        alphas_cumprod=diffusion.alphas_cumprod,
        train_num_steps=int(cfg["diffusion_steps"]),
    )
    scheduler.set_timesteps(int(cfg["diffusion_steps"]), device=device)
    return GuidedModelBundle(unet=unet, scheduler=scheduler, model_config=cfg)
