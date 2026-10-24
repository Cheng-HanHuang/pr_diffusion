"""Compatibility with the pinned UnivFD CLIP text tower.

Pinned external source WisconsinAIVision/UniversalFakeDetect at
030495aea3300a8b54c0ec37ec7fe1dd7e63c619:

Transformer.forward() returns (per_layer_dict, final_LND_tensor)
for feature extraction. VisionTransformer.forward() unpacks that
tuple; CLIP.encode_text() still treats the tuple as a tensor.

Reproduce the unmodified CLIP.encode_text arithmetic, explicitly
unpacking the tensor, without monkeypatching or touching upstream.
Fail closed if the pinned return contract changes.
"""
from __future__ import annotations

import torch


@torch.inference_mode()
def encode_text_univfd_tuple_compat(model, tokens):
    if not isinstance(tokens, torch.Tensor) or tokens.ndim!=2:
        raise RuntimeError("STOP_N2_CLIP_TEXT_TOKENS")
    if tokens.shape[1]!=model.context_length:
        raise RuntimeError("STOP_N2_CLIP_TEXT_CONTEXT_LENGTH")
    # Exact pinned CLIP.encode_text() operations and precision.
    x=model.token_embedding(tokens).type(model.dtype)
    x=x+model.positional_embedding.type(model.dtype)
    x=x.permute(1,0,2)  # [context,batch,width]
    result=model.transformer(x)
    if (not isinstance(result,tuple) or len(result)!=2 or
        not isinstance(result[0],dict) or
        not isinstance(result[1],torch.Tensor) or
        result[1].shape!=x.shape):
        raise RuntimeError("STOP_N2_CLIP_TEXT_TUPLE_SCHEMA")
    x=result[1].permute(1,0,2)
    x=model.ln_final(x).type(model.dtype)
    positions=tokens.argmax(dim=-1)
    features=x[torch.arange(x.shape[0],device=x.device),positions] @ model.text_projection
    if features.ndim!=2 or not bool(torch.isfinite(features).all()):
        raise RuntimeError("STOP_N2_CLIP_TEXT_FEATURES")
    return features
