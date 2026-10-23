"""N2 CLIP GT-free pipeline + N3 bounded inventory synthetic preflight."""
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path

import pytest


def source(name):
    return Path(__file__).resolve().parents[2]/"scripts/track_r"/name


def mod(name):
    p=source(name)
    spec=importlib.util.spec_from_file_location(name[:-3],p)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_n2_prompt_lock_and_budget():
    n=mod("r0n_2_clip_orientation.py")
    assert n.CEILING==900.
    assert n.PROMPTS_UP==(
      "a photo of a face.",
      "a photo of a person's face.",
      "a portrait photo of a person.")
    assert n.PROMPTS_INV==(
      "an upside-down photo of a face.",
      "an upside-down photo of a person's face.",
      "an upside-down portrait photo of a person.")
    assert n.CLIP_SHA=="b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
    assert n.SMOKE==64 and n.BATCH==32
    assert n.frozen_config.__name__=="frozen_config"


def test_n2_strict_identity_and_no_gt_gpu_tasks():
    text=source("r0n_2_clip_orientation.py").read_text()
    assert 'names=("uid","source","family","image_id","run_index",' in text
    assert "GT_FREE_CANDIDATES" in text
    assert "STOP_N2_GT_LEAK_IN_TASKS" in text
    assert 's1>s0' in text
    assert "gt_path" not in text.split("def clip_score(args):")[1].split("def eval_scores")[0]
    assert "torch.inference_mode()" in text
    assert "score['rows']" not in text
    compile(text,"r0n_2_clip_orientation.py","exec")


def test_n3_no_recursive_image_tree_inventory():
    n=mod("r0n_3_imagenet_inventory_cpu.py")
    assert len(n.MODEL_FILES)==2
    assert n.MODEL_FILES[0].name=="imagenet256.pt"
    assert n.MODEL_FILES[1].name=="256x256_diffusion_uncond.pt"
    with pytest.raises(RuntimeError,match="PROTECTED_TOKEN"):
        n.forbid("/some/confirmation305/registry.txt")
    text=source("r0n_3_imagenet_inventory_cpu.py").read_text()
    assert "iterdir()" in text
    assert "ls-files" in text
    assert "no_image_tree_enumeration" in text
    compile(text,"r0n_3_imagenet_inventory_cpu.py","exec")
