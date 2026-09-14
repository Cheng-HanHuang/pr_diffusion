#!/usr/bin/env python3
"""B24.3 NP-native branching runner for one locked development measurement.

Implements only the prospectively frozen B24.3 pilot arms:
  NP1
  NP4_INDEPENDENT
  NP_EPP
  NP_EPP_RANDOM_PRUNE
  NP_EPP_NO_REALLOCATION
  NP_DPS

Runtime branch decisions are measurement-only. Ground truth is loaded only for
offline terminal metrics after an arm has completed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
SELECTOR_PATH = SCRIPTS / "pr_external_difffpr_np_guided_lf_s2_selector.py"
MODEL_PATH = Path("/egr/research-pac/huang248/models/ffhq_10m.pt")
DIFFFPR_ROOT = Path("/egr/research-pac/huang248/external/DiffFPR")
SPEC_PATH = REPO / "configs/b24/b24_3_method_dev_spec.json"
SCREEN_MANIFEST_SHA = "b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba"
MODEL_SHA = "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
NP_STEPS = 1000
PROJ_START = 300
SCORE_RADIUS = 0.6
PROJ_RADIUS = 0.2
PROJ_SCHEDULE = "300:0.2"
TRAILING_WINDOW = 32
HARD_CEILING_MIB = 52452
ARM_ORDER = (
    "NP1", "NP4_INDEPENDENT", "NP_EPP", "NP_EPP_RANDOM_PRUNE",
    "NP_EPP_NO_REALLOCATION", "NP_DPS",
)
EXPECTED_COUNTS = {
    "NP1": {"proposal_evals": 2199, "total_unet_evals": 2200, "terminals": 1},
    "NP4_INDEPENDENT": {"proposal_evals": 8796, "total_unet_evals": 8800, "terminals": 4},
    "NP_EPP": {"proposal_evals": 8796, "total_unet_evals": 8800, "terminals": 4},
    "NP_EPP_RANDOM_PRUNE": {"proposal_evals": 8796, "total_unet_evals": 8800, "terminals": 4},
    "NP_EPP_NO_REALLOCATION": {"proposal_evals": 5756, "total_unet_evals": 5760, "terminals": 4},
    "NP_DPS": {"proposal_evals": 8796, "total_unet_evals": 8797, "terminals": 4},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    x = value.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(x.dtype), "shape": list(x.shape)}, sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256(header + b"\0")
    h.update(x.numpy().tobytes(order="C"))
    return h.hexdigest()


def domain_hash(domain: str, *parts: object) -> str:
    material = "|".join([domain, SCREEN_MANIFEST_SHA, *[str(x) for x in parts]])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def seed63(domain: str, *parts: object) -> int:
    return int(domain_hash(domain, *parts)[:16], 16) & ((1 << 63) - 1)


def psnr_raw(x: torch.Tensor, gt: torch.Tensor) -> float:
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    g01 = (gt.clamp(-1, 1) + 1.0) * 0.5
    mse = torch.mean((x01 - g01).square()).clamp_min(1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).detach().cpu().item())


def ssim_raw(x: torch.Tensor, gt: torch.Tensor) -> float:
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    g01 = (gt.clamp(-1, 1) + 1.0) * 0.5
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x01, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(g01, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x01 * x01, 11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(g01 * g01, 11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x01 * g01, 11, stride=1, padding=5) - mu_x * mu_y
    out = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)).clamp_min(1e-12)
    return float(out.mean().detach().cpu().item())


def require_model_range(x: torch.Tensor, label: str) -> None:
    if tuple(x.shape) != (1, 3, 256, 256):
        raise RuntimeError(f"{label}: unexpected shape {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError(f"{label}: non-finite tensor")


def unwrap_tensor(path: Path, keys: tuple[str, ...], device: torch.device) -> torch.Tensor:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            if isinstance(payload.get(key), torch.Tensor):
                return payload[key]
    raise TypeError(f"no expected tensor in {path}")


def rng_snapshot(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()


def rng_restore(cpu_state: torch.Tensor, cuda_state: torch.Tensor, device: torch.device) -> None:
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)


@dataclass
class Branch:
    lineage: str
    x0: torch.Tensor
    eps_prev: torch.Tensor | None
    cpu_rng: torch.Tensor
    cuda_rng: torch.Tensor
    pre_lf_mse: list[float] = field(default_factory=list)
    post_lf_mse: list[float] = field(default_factory=list)
    post_full_mse: list[float] = field(default_factory=list)
    last_transition: int = -1

    def trailing_score(self, window: int = TRAILING_WINDOW) -> float:
        if len(self.pre_lf_mse) < window:
            raise RuntimeError(f"{self.lineage}: need {window} preprojection values, got {len(self.pre_lf_mse)}")
        return float(statistics.mean(self.pre_lf_mse[-window:]))

    def terminal_selector(self) -> float:
        if not self.post_lf_mse:
            raise RuntimeError(f"{self.lineage}: no postprojection selector history")
        return float(statistics.mean(self.post_lf_mse))

    def rng_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.cpu_rng.detach().cpu().numpy().tobytes(order="C"))
        h.update(self.cuda_rng.detach().cpu().numpy().tobytes(order="C"))
        return h.hexdigest()


@dataclass
class RunContext:
    selector: Any
    unet: Any
    scheduler: Any
    device: torch.device
    mag_target: torch.Tensor
    pad: int
    timesteps: torch.Tensor
    proj_schedule: Any
    image_id: str
    gt: torch.Tensor


def initialize_branch(ctx: RunContext, seed: int, lineage: str) -> tuple[Branch, int]:
    ctx.selector.seed_everything(int(seed))
    x_t = torch.randn((1, 3, ctx.unet.config.sample_size, ctx.unet.config.sample_size), device=ctx.device)
    t0 = int(ctx.timesteps[0])
    t_tensor = torch.tensor([t0], device=ctx.device, dtype=torch.long)
    eps = ctx.unet(x_t, t_tensor).sample
    alpha_bar = ctx.scheduler.alphas_cumprod[t0].to(device=ctx.device, dtype=x_t.dtype)
    x0 = (x_t - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
    require_model_range(x0, f"{lineage}/initial")
    cpu_rng, cuda_rng = rng_snapshot(ctx.device)
    return Branch(lineage=lineage, x0=x0.detach(), eps_prev=None, cpu_rng=cpu_rng, cuda_rng=cuda_rng), 1


def _candidate_set(ctx: RunContext, branch: Branch, transition: int, k: int):
    if not (0 <= transition <= 998) or k <= 0:
        raise ValueError(f"invalid transition/k: {transition}/{k}")
    rng_restore(branch.cpu_rng, branch.cuda_rng, ctx.device)
    x0_hat = branch.x0
    if transition >= PROJ_START:
        radius = ctx.selector.base.radius_at_step(ctx.proj_schedule, transition)
        x0_hat = ctx.selector.base.enforce_oversampled_lowfreq(x0_hat, ctx.mag_target, ctx.pad, radius)
    t_next = int(ctx.timesteps[transition + 1])
    alpha_bar = ctx.scheduler.alphas_cumprod[t_next].to(device=ctx.device, dtype=x0_hat.dtype)
    sqrt_at, sqrt_1mat = torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)
    xs, noises, scores, lf_mses, full_mses = [], [], [], [], []
    for j in range(k):
        if j == 0 and branch.eps_prev is not None and k > 1:
            eps_cand = branch.eps_prev
        else:
            eps_cand = torch.randn_like(x0_hat)
        x_t_cand = sqrt_at * x0_hat + sqrt_1mat * eps_cand
        t_tensor = torch.tensor([t_next], device=ctx.device, dtype=torch.long)
        eps_pred = ctx.unet(x_t_cand, t_tensor).sample
        x0_cand = (x_t_cand - sqrt_1mat * eps_pred) / sqrt_at
        require_model_range(x0_cand, f"{branch.lineage}/t{transition}/c{j}")
        score = ctx.selector.base.oversampled_lowfreq_mag_l2(x0_cand, ctx.mag_target, ctx.pad, SCORE_RADIUS)
        xs.append(x0_cand.detach())
        noises.append(eps_cand.detach().clone())
        scores.append(float(score.detach().cpu().item()))
        lf_mses.append(float(ctx.selector.lowfreq_mse_vs_observation(x0_cand, ctx.mag_target, ctx.pad, SCORE_RADIUS)))
        full_mses.append(float(ctx.selector.full_mse_vs_observation(x0_cand, ctx.mag_target, ctx.pad)))
    branch.cpu_rng, branch.cuda_rng = rng_snapshot(ctx.device)
    return xs, noises, scores, lf_mses, full_mses, k


def step_greedy(ctx: RunContext, branch: Branch, transition: int, k: int) -> int:
    xs, noises, scores, lf_mses, full_mses, evals = _candidate_set(ctx, branch, transition, k)
    winner = min(range(k), key=lambda j: (scores[j], j))
    branch.x0, branch.eps_prev = xs[winner], noises[winner]
    branch.last_transition = transition
    if transition < PROJ_START:
        branch.pre_lf_mse.append(lf_mses[winner])
    else:
        branch.post_lf_mse.append(lf_mses[winner])
        branch.post_full_mse.append(full_mses[winner])
    return evals


def _seed_branch_rng(ctx: RunContext, branch: Branch, seed: int) -> None:
    ctx.selector.seed_everything(int(seed))
    branch.cpu_rng, branch.cuda_rng = rng_snapshot(ctx.device)


def expand_retain_all(ctx: RunContext, branch: Branch, transition: int, k: int, domain: str):
    xs, noises, _, lf_mses, full_mses, evals = _candidate_set(ctx, branch, transition, k)
    children = []
    for j in range(k):
        child = Branch(
            lineage=f"{branch.lineage}/e{transition}c{j}", x0=xs[j], eps_prev=noises[j],
            cpu_rng=branch.cpu_rng.clone(), cuda_rng=branch.cuda_rng.clone(),
            pre_lf_mse=list(branch.pre_lf_mse), post_lf_mse=list(branch.post_lf_mse),
            post_full_mse=list(branch.post_full_mse), last_transition=transition,
        )
        if transition < PROJ_START:
            child.pre_lf_mse.append(lf_mses[j])
        else:
            child.post_lf_mse.append(lf_mses[j]); child.post_full_mse.append(full_mses[j])
        _seed_branch_rng(ctx, child, seed63(domain, ctx.image_id, branch.lineage, transition, j))
        children.append(child)
    return children, evals


def checkpoint_event(branches, survivors, checkpoint: int, ctx: RunContext, rule: str):
    survivor_ids = {b.lineage for b in survivors}
    return {
        "checkpoint_before_transition": checkpoint,
        "rule": rule,
        "uses_ground_truth_for_decision": False,
        "branches": [
            {
                "lineage": b.lineage,
                "trailing32_lf_mse": b.trailing_score(),
                "pointwise_lf_mse": float(b.pre_lf_mse[-1]),
                "offline_current_psnr_raw": psnr_raw(b.x0, ctx.gt),
                "survived": b.lineage in survivor_ids,
                "rng_sha256": b.rng_sha256(),
                "selected_noise_sha256": tensor_sha256(b.eps_prev) if b.eps_prev is not None else None,
            }
            for b in branches
        ],
        "survivor_lineages": [b.lineage for b in survivors],
    }


def prune_measurement(ctx, branches, keep, checkpoint, events):
    ranked = sorted(branches, key=lambda b: (b.trailing_score(), b.lineage))
    survivors = ranked[:keep]
    events.append(checkpoint_event(branches, survivors, checkpoint, ctx, "trailing32_lf_mse"))
    return survivors


def prune_random(ctx, branches, keep, checkpoint, events):
    ranked = sorted(branches, key=lambda b: (domain_hash("B24_METHOD_RANDOM_PRUNE_V1", ctx.image_id, checkpoint, b.lineage), b.lineage))
    survivors = ranked[:keep]
    events.append(checkpoint_event(branches, survivors, checkpoint, ctx, "domain_hash_random"))
    return survivors


def hard_fork(ctx: RunContext, parent: Branch, count: int, domain: str):
    out = []
    for j in range(count):
        child = Branch(
            lineage=f"{parent.lineage}/hard{j}", x0=parent.x0.detach().clone(),
            eps_prev=parent.eps_prev.detach().clone() if parent.eps_prev is not None else None,
            cpu_rng=parent.cpu_rng.clone(), cuda_rng=parent.cuda_rng.clone(),
            pre_lf_mse=list(parent.pre_lf_mse), post_lf_mse=[], post_full_mse=[],
            last_transition=parent.last_transition,
        )
        _seed_branch_rng(ctx, child, seed63(domain, ctx.image_id, parent.lineage, j))
        out.append(child)
    return out


def run_np1(ctx, roots):
    b, initial = initialize_branch(ctx, roots[0], "root0")
    proposals = 0
    for i in range(999): proposals += step_greedy(ctx, b, i, 5 if i < PROJ_START else 1)
    return [b], [], proposals, initial + proposals


def run_np4(ctx, roots):
    terminals, proposals, total = [], 0, 0
    for r, seed in enumerate(roots):
        b, initial = initialize_branch(ctx, seed, f"root{r}"); total += initial
        for i in range(999): proposals += step_greedy(ctx, b, i, 5 if i < PROJ_START else 1)
        terminals.append(b)
    return terminals, [], proposals, total + proposals


def run_epp_common(ctx, roots, prune_mode: str, reallocate: bool):
    branches, initial_total = [], 0
    for r, seed in enumerate(roots):
        b, initial = initialize_branch(ctx, seed, f"root{r}"); branches.append(b); initial_total += initial
    proposals, events = 0, []
    for i in range(72):
        for b in branches: proposals += step_greedy(ctx, b, i, 5)
    pruner: Callable = prune_measurement if prune_mode == "measurement" else prune_random
    branches = pruner(ctx, branches, 2, 72, events)
    for i in range(72, 148):
        for b in branches: proposals += step_greedy(ctx, b, i, 10 if reallocate else 5)
    branches = pruner(ctx, branches, 1, 148, events)
    for i in range(148, 300): proposals += step_greedy(ctx, branches[0], i, 20 if reallocate else 5)
    descendants = hard_fork(ctx, branches[0], 4, "B24_METHOD_EPP_HARD_FORK_V1")
    for i in range(300, 999):
        for b in descendants: proposals += step_greedy(ctx, b, i, 1)
    return descendants, events, proposals, initial_total + proposals


def run_epp(ctx, roots): return run_epp_common(ctx, roots, "measurement", True)
def run_epp_random(ctx, roots): return run_epp_common(ctx, roots, "random", True)
def run_epp_no_reallocation(ctx, roots): return run_epp_common(ctx, roots, "measurement", False)


def run_dps(ctx, roots):
    parent, initial = initialize_branch(ctx, roots[0], "root0")
    proposals, events = 0, []
    for i in range(72): proposals += step_greedy(ctx, parent, i, 5)
    branches, used = expand_retain_all(ctx, parent, 72, 5, "B24_METHOD_DPS_BRANCH_V1"); proposals += used
    for i in range(73, 148):
        for b in branches: proposals += step_greedy(ctx, b, i, 5)
    branches = prune_measurement(ctx, branches, 1, 148, events)
    branches, used = expand_retain_all(ctx, branches[0], 148, 5, "B24_METHOD_DPS_BRANCH_V1"); proposals += used
    for i in range(149, 224):
        for b in branches: proposals += step_greedy(ctx, b, i, 5)
    branches = prune_measurement(ctx, branches, 1, 224, events)
    branches, used = expand_retain_all(ctx, branches[0], 224, 5, "B24_METHOD_DPS_BRANCH_V1"); proposals += used
    for i in range(225, 300):
        for b in branches: proposals += step_greedy(ctx, b, i, 5)
    branches = prune_measurement(ctx, branches, 1, 300, events)
    descendants = hard_fork(ctx, branches[0], 4, "B24_METHOD_DPS_HARD_FORK_V1")
    for i in range(300, 999):
        for b in descendants: proposals += step_greedy(ctx, b, i, 1)
    return descendants, events, proposals, initial + proposals


RUNNERS = {
    "NP1": run_np1,
    "NP4_INDEPENDENT": run_np4,
    "NP_EPP": run_epp,
    "NP_EPP_RANDOM_PRUNE": run_epp_random,
    "NP_EPP_NO_REALLOCATION": run_epp_no_reallocation,
    "NP_DPS": run_dps,
}


def gpu_inventory():
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 5:
            rows.append({"index": int(parts[0]), "uuid": parts[1], "total_mib": int(parts[2]), "used_mib": int(parts[3]), "free_mib": int(parts[4])})
    return rows


def process_gpu_memory(uuid: str, pid: int):
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    values = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid and parts[1].isdigit() and int(parts[1]) == pid:
            try: values.append(int(parts[2]))
            except ValueError: pass
    return sum(values) if values else None


class GPUMonitor:
    def __init__(self, physical_gpu: int, interval_s: float = 1.0):
        self.physical_gpu, self.interval_s, self.pid = int(physical_gpu), float(interval_s), os.getpid()
        row = next((r for r in gpu_inventory() if r.get("index") == self.physical_gpu), None)
        if row is None: raise RuntimeError(f"cannot resolve physical GPU {self.physical_gpu}")
        self.uuid = str(row["uuid"]); self.max_process_mib = None; self.max_device_used_mib = None; self.min_device_free_mib = None; self.samples = 0
        self._stop = threading.Event(); self._thread = None
    def _sample(self):
        row = next((r for r in gpu_inventory() if r.get("index") == self.physical_gpu), None)
        if row is not None:
            used, free = int(row["used_mib"]), int(row["free_mib"])
            self.max_device_used_mib = used if self.max_device_used_mib is None else max(self.max_device_used_mib, used)
            self.min_device_free_mib = free if self.min_device_free_mib is None else min(self.min_device_free_mib, free)
        proc = process_gpu_memory(self.uuid, self.pid)
        if proc is not None: self.max_process_mib = proc if self.max_process_mib is None else max(self.max_process_mib, proc)
        self.samples += 1
    def _loop(self):
        while not self._stop.is_set(): self._sample(); self._stop.wait(self.interval_s)
    def start(self):
        self._sample(); self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()
    def stop(self):
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=max(2.0, self.interval_s * 3))
        self._sample()
        return {"physical_gpu": self.physical_gpu, "gpu_uuid": self.uuid, "samples": self.samples, "max_b24_process_gpu_mib": self.max_process_mib, "max_whole_device_used_mib": self.max_device_used_mib, "min_whole_device_free_mib": self.min_device_free_mib}


def torch_memory_snapshot(device):
    return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "final_allocated_bytes": int(torch.cuda.memory_allocated(device)), "final_reserved_bytes": int(torch.cuda.memory_reserved(device))}


def finalize_arm(arm, ctx, terminals, events, proposal_evals, total_evals, arm_dir, wall_s, torch_memory, gpu_monitor):
    expected = EXPECTED_COUNTS[arm]
    if proposal_evals != expected["proposal_evals"] or total_evals != expected["total_unet_evals"] or len(terminals) != expected["terminals"]:
        raise RuntimeError(f"{arm}: accounting mismatch proposals={proposal_evals} total={total_evals} terminals={len(terminals)} expected={expected}")
    arm_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for idx, branch in enumerate(terminals):
        require_model_range(branch.x0, f"{arm}/terminal{idx}")
        p, s, selector_stat = psnr_raw(branch.x0, ctx.gt), ssim_raw(branch.x0, ctx.gt), branch.terminal_selector()
        tpath = arm_dir / f"terminal_{idx}.pt"; torch.save({"reconstruction": branch.x0.detach().cpu()}, tpath)
        rows.append({"terminal_index": idx, "lineage": branch.lineage, "selector_post_winner_lf_mse_mean": selector_stat, "psnr_raw_db": p, "ssim_raw": s, "good25": bool(p >= 25.0), "reconstruction_path": str(tpath.resolve()), "reconstruction_sha256": tensor_sha256(branch.x0), "rng_sha256": branch.rng_sha256(), "selected_noise_sha256": tensor_sha256(branch.eps_prev) if branch.eps_prev is not None else None})
    selected_idx = min(range(len(rows)), key=lambda i: (float(rows[i]["selector_post_winner_lf_mse_mean"]), int(rows[i]["terminal_index"])))
    oracle_idx = max(range(len(rows)), key=lambda i: (float(rows[i]["psnr_raw_db"]), -int(rows[i]["terminal_index"])))
    selected, oracle = rows[selected_idx], rows[oracle_idx]
    peak_reserved_mib = torch_memory["peak_reserved_bytes"] / 1048576.0; peak_allocated_mib = torch_memory["peak_allocated_bytes"] / 1048576.0
    process_peak = gpu_monitor.get("max_b24_process_gpu_mib")
    if process_peak is not None and int(process_peak) > HARD_CEILING_MIB: raise RuntimeError(f"{arm}: process peak {process_peak} > {HARD_CEILING_MIB}")
    result = {
        "schema_version": "b24.np-branching-arm.v1", "status": "PASS", "arm": arm,
        "runtime_decisions_use_ground_truth": False, "terminal_selection_uses_ground_truth": False, "terminal_oracle_is_offline_only": True,
        "proposal_unet_evals": proposal_evals, "initial_unet_evals": total_evals - proposal_evals, "total_unet_evals": total_evals,
        "np1_equivalent_work_fre": total_evals / 2200.0, "terminal_count": len(terminals), "wall_seconds": wall_s,
        "torch_memory": {**torch_memory, "peak_allocated_mib": peak_allocated_mib, "peak_reserved_mib": peak_reserved_mib}, "gpu_monitor": gpu_monitor,
        "checkpoint_events": events, "terminals": rows, "clean_free_selected_terminal_index": selected_idx,
        "clean_free_selected_psnr_raw_db": float(selected["psnr_raw_db"]), "clean_free_selected_ssim_raw": float(selected["ssim_raw"]), "clean_free_selected_good25": bool(selected["good25"]),
        "oracle_best_terminal_index": oracle_idx, "oracle_best_psnr_raw_db": float(oracle["psnr_raw_db"]), "oracle_best_ssim_raw": float(oracle["ssim_raw"]), "oracle_best_good25": bool(oracle["good25"]),
        "selector_gap_psnr_db": float(oracle["psnr_raw_db"]) - float(selected["psnr_raw_db"]),
    }
    write_json_atomic(arm_dir / "result.json", result); return result


def validate_spec():
    spec = read_json(SPEC_PATH); parent = spec["np_parent"]
    exact = {"num_steps": NP_STEPS, "projection_start": PROJ_START, "soft_candidates": 5, "hard_candidates": 1, "score_radius": SCORE_RADIUS, "projection_radius": PROJ_RADIUS, "projection_radius_schedule": PROJ_SCHEDULE}
    for key, expected in exact.items():
        if parent.get(key) != expected: raise RuntimeError(f"spec drift {key}: {parent.get(key)!r} != {expected!r}")
    return spec


def parse_role_row(path: Path):
    if path.suffix.lower() == ".json": return read_json(path)
    with path.open(newline="", encoding="utf-8") as f: rows = [dict(r) for r in csv.DictReader(f)]
    if len(rows) != 1: raise RuntimeError(f"role row CSV must have one row: {path}")
    return rows[0]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--input-manifest", type=Path, required=True); ap.add_argument("--role-row", type=Path, required=True); ap.add_argument("--output-root", type=Path, required=True); ap.add_argument("--physical-gpu", type=int, required=True); ap.add_argument("--arms", default=",".join(ARM_ORDER)); args = ap.parse_args()
    validate_spec(); arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    if not arms or any(a not in ARM_ORDER for a in arms): raise RuntimeError(f"invalid arms {arms}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(args.physical_gpu): raise RuntimeError(f"physical binding mismatch CUDA_VISIBLE_DEVICES={visible!r} expected {args.physical_gpu}")
    if not torch.cuda.is_available(): raise RuntimeError("B24.3 requires CUDA")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    manifest, role = read_json(args.input_manifest.resolve()), parse_role_row(args.role_row.resolve())
    image_id = str(role["image_id"]).zfill(5)
    if str(manifest["image_id"]).zfill(5) != image_id or str(role.get("method_role", "")).upper() != "DEVELOPMENT" or str(role.get("pilot16", "")).upper() != "TRUE": raise RuntimeError("role/input identity is not a frozen Pilot16 DEVELOPMENT row")
    if int(manifest["measurement_seed"]) != int(role["dev_measurement_seed"]): raise RuntimeError("development measurement seed mismatch")
    if int(role["dev_measurement_seed"]) == int(role["source_measurement_seed"]): raise RuntimeError("development measurement reused screening seed")
    if sha256_file(MODEL_PATH) != MODEL_SHA: raise RuntimeError("model SHA mismatch")
    meas_path, gt_path = Path(manifest["measurement_path"]).resolve(), Path(manifest["ground_truth_tensor_path"]).resolve()
    if sha256_file(meas_path) != manifest["measurement_file_sha256"]: raise RuntimeError("measurement file SHA mismatch")
    measurement_raw = unwrap_tensor(meas_path, ("measurement", "y", "observation"), device).to(dtype=torch.float32)
    if tensor_sha256(measurement_raw) != manifest["measurement_tensor_sha256"] or tuple(measurement_raw.shape) != (1, 3, 384, 384) or not bool(torch.isfinite(measurement_raw).all()): raise RuntimeError("measurement tensor identity/schema mismatch")
    measurement_np = measurement_raw.clamp_min(0.0)
    gt = unwrap_tensor(gt_path, ("ground_truth", "image", "x"), device).to(dtype=torch.float32); require_model_range(gt, "ground_truth")
    if tensor_sha256(gt) != manifest["ground_truth_tensor_sha256"]: raise RuntimeError("ground-truth SHA mismatch")
    selector = load_module("b24_3_selector_parent", SELECTOR_PATH)
    load_start = time.perf_counter(); bundle = selector.load_guided_diffusion_model(model_path=str(MODEL_PATH), device=device, preset="difffpr_ffhq_10m", guided_diffusion_dir=str(DIFFFPR_ROOT), strict=True); torch.cuda.synchronize(device); model_load_s = time.perf_counter() - load_start
    bundle.scheduler.set_timesteps(NP_STEPS, device=device); timesteps = bundle.scheduler.timesteps
    if len(timesteps) != NP_STEPS: raise RuntimeError(f"scheduler timestep drift: {len(timesteps)}")
    ctx = RunContext(selector, bundle.unet, bundle.scheduler, device, measurement_np, selector.base.oversample_pad(256, 2.0), timesteps, selector.base.parse_radius_schedule(PROJ_SCHEDULE, PROJ_RADIUS), image_id, gt)
    roots = [int(role[f"np_root_seed_{i}"]) for i in range(4)]
    if len(set(roots)) != 4: raise RuntimeError("NP root seed collision")
    out = args.output_root.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    write_json_atomic(out / "RUN_IDENTITY.json", {"schema_version": "b24.np-branching-run-identity.v1", "image_id": image_id, "class_label": role["class_label"], "method_role": role["method_role"], "pilot16": role["pilot16"], "development_measurement_seed": int(role["dev_measurement_seed"]), "screening_measurement_seed": int(role["source_measurement_seed"]), "measurement_file_sha256": manifest["measurement_file_sha256"], "measurement_tensor_sha256": manifest["measurement_tensor_sha256"], "np_root_seeds": roots, "screen_manifest_sha256": SCREEN_MANIFEST_SHA, "method_spec_sha256": sha256_file(SPEC_PATH), "model_sha256": MODEL_SHA, "physical_gpu": args.physical_gpu, "cuda_visible_devices": visible, "model_load_seconds": model_load_s, "arms": arms})
    all_results, max_process_peak, max_torch_reserved = {}, 0, 0.0; total_wall_start = time.perf_counter()
    for arm in arms:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); bundle.scheduler.set_timesteps(NP_STEPS, device=device); ctx.timesteps = bundle.scheduler.timesteps
        monitor = GPUMonitor(args.physical_gpu, 1.0); monitor.start(); start = time.perf_counter()
        try:
            with torch.no_grad(): terminals, events, proposal_evals, total_evals = RUNNERS[arm](ctx, roots)
            torch.cuda.synchronize(device); wall_s = time.perf_counter() - start; torch_mem = torch_memory_snapshot(device)
        finally: gpu_mem = monitor.stop()
        result = finalize_arm(arm, ctx, terminals, events, proposal_evals, total_evals, out / arm, wall_s, torch_mem, gpu_mem); all_results[arm] = result
        if gpu_mem.get("max_b24_process_gpu_mib") is not None: max_process_peak = max(max_process_peak, int(gpu_mem["max_b24_process_gpu_mib"]))
        max_torch_reserved = max(max_torch_reserved, float(result["torch_memory"]["peak_reserved_mib"]))
        del terminals; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"status": "PASS", "arm": arm, "proposal_unet_evals": proposal_evals, "total_unet_evals": total_evals, "selected_psnr_raw_db": result["clean_free_selected_psnr_raw_db"], "oracle_psnr_raw_db": result["oracle_best_psnr_raw_db"], "wall_seconds": wall_s, "max_process_gpu_mib": gpu_mem.get("max_b24_process_gpu_mib"), "peak_torch_reserved_mib": result["torch_memory"]["peak_reserved_mib"]}, sort_keys=True), flush=True)
    total_wall = time.perf_counter() - total_wall_start
    if max_process_peak > HARD_CEILING_MIB: raise RuntimeError(f"process peak {max_process_peak} > {HARD_CEILING_MIB}")
    recommended_gate = max(10240, int(math.ceil((max_process_peak + 4096) / 1024.0) * 1024) if max_process_peak else 16384)
    if recommended_gate > HARD_CEILING_MIB: raise RuntimeError(f"derived gate {recommended_gate} > hard ceiling")
    complete = {"schema_version": "b24.np-branching-one-image-smoke.v1", "status": "PASS", "image_id": image_id, "class_label": role["class_label"], "method_role": role["method_role"], "pilot16": True, "gpu_work_performed": True, "measurement_generation_performed_by_runner": False, "runtime_ground_truth_decisions": False, "all_requested_arms_passed": list(all_results) == arms, "arms": arms, "arm_results": {arm: {"proposal_unet_evals": all_results[arm]["proposal_unet_evals"], "total_unet_evals": all_results[arm]["total_unet_evals"], "terminal_count": all_results[arm]["terminal_count"], "clean_free_selected_psnr_raw_db": all_results[arm]["clean_free_selected_psnr_raw_db"], "oracle_best_psnr_raw_db": all_results[arm]["oracle_best_psnr_raw_db"], "selector_gap_psnr_db": all_results[arm]["selector_gap_psnr_db"], "wall_seconds": all_results[arm]["wall_seconds"], "peak_torch_reserved_mib": all_results[arm]["torch_memory"]["peak_reserved_mib"], "max_b24_process_gpu_mib": all_results[arm]["gpu_monitor"].get("max_b24_process_gpu_mib")} for arm in arms}, "max_b24_process_gpu_mib": max_process_peak, "max_torch_reserved_mib": max_torch_reserved, "hard_ceiling_mib": HARD_CEILING_MIB, "recommended_pilot_min_free_mib": recommended_gate, "recommended_gate_rule": "ceil_1024(max_observed_B24_process_mib + 4096_MiB_reserve), floor 10240", "total_wall_seconds_excluding_model_load": total_wall, "model_load_seconds": model_load_s, "next": "FIXED_PILOT16_EXECUTION_AUTHORIZED_IF_THIS_PASS_IS_ACCEPTED"}
    write_json_atomic(out / "SMOKE_COMPLETE.json", complete); print(json.dumps(complete, sort_keys=True), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
