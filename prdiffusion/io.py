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
    x = (x * 2.0 - 1.0).contiguous()
    if device is not None:
        x = x.to(device, non_blocking=True).contiguous()
    return x


def find_image_by_basename(root: str, basename: str) -> Optional[str]:
    """Search recursively under root for a file with exact basename."""
    for dirpath, _, filenames in os.walk(root):
        if basename in filenames:
            return os.path.join(dirpath, basename)
    return None
