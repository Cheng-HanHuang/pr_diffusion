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
    assert "gpu_reads_canonical_png_only" in text
    score=text.split("def clip_score(args):",1)[1].split("def eval_scores",1)[0]
    assert "load_png_rgb8(file)" in score
    assert "candidate_tensor" not in score
    assert "STOP_N2_CANONICAL_PNG_SHA" in score
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


def test_comprehensive_orchestrator_has_hard_stop_and_no_r1():
    source=Path(__file__).resolve().parents[2]/"scripts/track_r/r0n_all_autopilot.py"
    code=source.read_text(encoding="utf-8")
    compile(code,str(source),"exec")
    for value in (
        "STOP_R0N_K3_OR_N1","N2_CLIP_GPU",
        "N3_CPU_INVENTORY","R0N_ALL_STAGES_COMPLETE_PLANNER_READY",
        "R0N_PARTIAL_OPTIONALS_PLANNER_READY",
        "N0_import","N1_tests"):
        assert value in code
    assert "R1_runs_started" in code
    assert "r0n_all_" in code


def test_n0_local_commit_survives_git_push_auth_failure(monkeypatch):
    controller=mod("r0n_all_autopilot.py")
    controller.PUSH_ERRORS.clear()
    observed=[]
    def fake_git(*parts):
        observed.append(parts)
        if parts[0]=="push":
            raise RuntimeError("authentication unavailable")
        if parts[:2]==("rev-parse","HEAD"):
            return "a"*40
        return ""
    monkeypatch.setattr(controller,"git",fake_git)
    monkeypatch.setattr(controller,"clean",lambda:None)
    assert controller.commit("test/path","archival test")=="a"*40
    assert len(controller.PUSH_ERRORS)==1
    assert controller.PUSH_ERRORS[0]["head"]=="a"*40
    assert observed[0][0]=="add"
    assert observed[1][0]=="commit"
    assert observed[-1][0]=="push"
    controller.PUSH_ERRORS.clear()


def test_launcher_warns_on_dry_run_without_failing():
    file=Path(__file__).resolve().parents[2]/"scripts/track_r/r0n_all_launch.sh"
    code=file.read_text(encoding="utf-8")
    assert "R0N_GIT_PUSH_PREFLIGHT_WARNING_LOCAL_FALLBACK" in code
    assert "STOP_R0N_GIT_PUSH_PERMISSION_OR_REMOTE_DIVERGENCE" not in code
    assert "R0N_GIT_ORIGINAL_HOME" in code


def test_n2_shared_gpu_admission_parses_available_memory():
    n=mod("r0n_2_clip_orientation.py")
    usage="0, 76543, 98304, 21761, 89\n1, 76621, 98304, 21683, 97\n"
    observed=n.parse_gpu_inventory(usage)
    assert observed[0]["free_mib"]>n.MIN_FREE_MIB
    assert observed[1]["util_percent"]==97
    assert n.MIN_FREE_MIB==40960
    with pytest.raises(RuntimeError,match="INVENTORY_FIELDS"):
        n.parse_gpu_inventory("1, 70000\n")
    with pytest.raises(RuntimeError,match="INVENTORY_INVALID"):
        n.parse_gpu_inventory("0, 90000, 50000, 5000, 50\n")

def test_r0n_resume_n0_checks_all_original_shas_and_tracking():
    src=(Path(__file__).resolve().parents[2]/
         "scripts/track_r/r0n_all_autopilot.py").read_text()
    assert 'p.add_argument("--resume-n0"' in src
    assert 'STOP_R0N_N0_NOT_TRACKED' in src
    assert 'STOP_R0N_N0_SOURCE_SHA' in src
    assert 'R0N_N0_REUSED_SHA_VERIFIED' in src
    assert "if not a.resume_n0:" in src
    launcher=(Path(__file__).resolve().parents[2]/
              "scripts/track_r/r0n_all_launch.sh").read_text()
    assert 'RESUME_N0="${2:-}"' in launcher
    assert '"--resume-n0"' in launcher


def test_pinned_univfd_two_value_transformer_text_compat_cpu():
    """Minimal exact-path test of the original tuple-vs-tensor crash."""
    import torch
    import torch.nn as nn
    from prdiffusion.track_r.r0n_clip_compat import encode_text_univfd_tuple_compat

    class TupleTransformer(nn.Module):
        def forward(self,x):
            return {"layer0":x[0]},x+.125

    class ToyCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_length=5
            self.token_embedding=nn.Embedding(24,4)
            self.positional_embedding=nn.Parameter(torch.randn(5,4))
            self.transformer=TupleTransformer()
            self.ln_final=nn.LayerNorm(4)
            self.text_projection=nn.Parameter(torch.randn(4,3))
            self.dtype=torch.float32

    torch.manual_seed(4)
    model=ToyCLIP().eval()
    tokens=torch.tensor([[1,2,3,15,0],[2,10,4,0,0]],dtype=torch.long)
    result=encode_text_univfd_tuple_compat(model,tokens)
    x=model.token_embedding(tokens).type(model.dtype)
    x=x+model.positional_embedding.type(model.dtype)
    _,x=model.transformer(x.permute(1,0,2))
    x=model.ln_final(x.permute(1,0,2)).type(model.dtype)
    expected=x[torch.arange(2),tokens.argmax(dim=-1)]@model.text_projection
    assert result.shape==(2,3)
    assert torch.allclose(result,expected,rtol=0,atol=0)

def test_pinned_univfd_text_compat_rejects_changed_tuple_contract():
    import torch
    import torch.nn as nn
    from prdiffusion.track_r.r0n_clip_compat import encode_text_univfd_tuple_compat

    class BadTransformer(nn.Module):
        def forward(self,x):
            return x

    class TinyCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_length=3
            self.token_embedding=nn.Embedding(12,4)
            self.positional_embedding=nn.Parameter(torch.zeros(3,4))
            self.transformer=BadTransformer()
            self.dtype=torch.float32

    with pytest.raises(RuntimeError,match="TUPLE_SCHEMA"):
        encode_text_univfd_tuple_compat(TinyCLIP(),torch.tensor([[1,2,3]]))

def test_n2_retry_uses_only_approved_old_preparation():
    n=mod("r0n_2_clip_orientation.py")
    assert n.RECOVERY_PREP_HEAD=="b1637f7c27a3caaa03a8b204171fca0d3fccfcd5"
    assert n.RECOVERY_TASKS_SHA=="7be734de4cd2e6f395add521717506be9260c37708f8902e248c1848b68b705b"
    source=Path(n.__file__).read_text(encoding="utf-8")
    assert "encode_text_univfd_tuple_compat(model,tokens)" in source
    assert "N2_PREP_RECEIPT.json" in source
    assert "RECOVERY_TASKS_PATH" in source
    assert 'p.add_argument("--reuse-prep-head"' in source
