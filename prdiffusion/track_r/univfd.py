"""Pinned UnivFD CLIP ViT-L/14 classifier; entirely local weights and cache."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

FC_SHA = "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847"
CLIP_SHA = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
CLIP_MEAN = (0.48145466,0.4578275,0.40821073)
CLIP_STD = (0.26862954,0.26130258,0.27577711)


def verify_sha(path, expected):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    if h.hexdigest()!=expected:
        raise RuntimeError("STOP_UNIVFD_SHA_MISMATCH " + str(path))


class UnivFD:
    def __init__(self, repo_root, fc_path, clip_path, device="cpu"):
        root = Path(repo_root).resolve()
        fc = Path(fc_path).resolve()
        clip_path = Path(clip_path).resolve()
        if not fc.is_relative_to(root) or not (root / "models" / "clip").is_dir():
            raise ValueError("UnivFD path mismatch")
        verify_sha(fc,FC_SHA)
        verify_sha(clip_path,CLIP_SHA)
        # Import the official bundled CLIP module, not a pip-installed substitute.
        if str(root) not in sys.path:
            sys.path.insert(0,str(root))
        existing = sys.modules.get("models")
        if existing and str(root) not in str(getattr(existing,"__file__","")):
            raise RuntimeError("STOP_UNIVFD_IMPORT_COLLISION")
        clip = importlib.import_module("models.clip")
        backbone, _ = clip.load(str(clip_path), device="cpu", jit=False)
        classifier = nn.Linear(768,1)
        try:
            state = torch.load(fc,map_location="cpu",weights_only=True)
        except TypeError:
            state = torch.load(fc,map_location="cpu")
        classifier.load_state_dict(state,strict=True)
        self.device = torch.device(device)
        self.backbone = backbone.eval().to(self.device)
        self.classifier = classifier.eval().to(self.device)
        self.forward_images = 0
        self.forward_calls = 0

    @torch.inference_mode()
    def p_fake(self,x8):
        if not isinstance(x8,torch.Tensor) or x8.dtype!=torch.uint8 or x8.ndim!=4 or x8.shape[1:]!=(3,256,256):
            raise ValueError("expected [B,3,256,256] uint8 RGB")
        # validate.py: ToTensor -> CenterCrop(224), no resize.
        z = TF.center_crop(x8.to(self.device).float()/255.0,[224,224])
        z = TF.normalize(z,CLIP_MEAN,CLIP_STD)
        features = self.backbone.encode_image(z)
        logit = self.classifier(features.float()).squeeze(-1)
        self.forward_images += len(x8)
        self.forward_calls += 1
        return torch.sigmoid(logit).float()
