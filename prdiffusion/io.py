from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image


def load_image(path: str, size: int = 256, device: Optional[torch.device] = None) -> torch.Tensor:
    """Load RGB image, resize, convert to tensor in [-1,1] shaped [1,3,H,W]."""
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    arr = arr.transpose(2, 0, 1)                      # [3,H,W]
    x = torch.from_numpy(arr)[None]                   # [1,3,H,W]
    x = x * 2.0 - 1.0
    if device is not None:
        x = x.to(device)
    return x


def find_image_by_basename(root: str, basename: str) -> Optional[str]:
    """Search recursively under root for a file with exact basename."""
    for dirpath, _, filenames in os.walk(root):
        if basename in filenames:
            return os.path.join(dirpath, basename)
    return None


def maybe_download_celeba_hq_256() -> str:
    """Optionally download CelebA-HQ resized 256×256 via kagglehub.

    Returns:
        Path to the folder containing jpg files.
    Raises:
        RuntimeError if kagglehub is not installed or download fails.

    NOTE: Many HPC clusters block outbound network access. Prefer using --data_root.
    """
    try:
        import kagglehub  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "kagglehub is required to download CelebA-HQ in-script. "
            "Install it or point --data_root to a local dataset."
        ) from e

    path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")
    root = os.path.join(path, "celeba_hq_256")
    return root
