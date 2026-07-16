#!/usr/bin/env python3
"""Apply the env-gated B21.3 continuation patch to the local DAPS submodule.

The PAC checkout already carries uncommitted B20 fixed-measurement and LF-guidance
changes inside external/daps.  This script performs narrow, marker/anchor-based
edits on top of those changes and writes the exact incremental unified diff to
``docs/b21/patches/daps_b21_continuation.patch``.

Default DAPS behavior is unchanged unless B21 continuation environment variables
are set.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys


SAMPLER_MARKER = "# --- B21.3 continuation patch -----------------------------------------------"
POSTERIOR_MARKER = "# --- B21.3 continuation runner hook -----------------------------------------"


HELPERS = r'''
# --- B21.3 continuation patch -----------------------------------------------
# Default-off controls:
#   B21_CONT_ENABLE=1
#   B21_CONT_STATE_PATH=/path/to/state.pt
#   B21_CONT_NOISE_SEED=<int>
#   B21_SAVE_STATE_STEPS=0,200
#
# Saved state semantics:
#   x0y  : clean-state tensor immediately before re-noising
#   step : next annealing transition index
#   sigma: scheduler sigma at that next index
#
# Continuation re-enters with xt = x0y + sigma[step] * eps(seed).


def _b21_env_enabled(name):
    return os.environ.get(name, "0").strip() == "1"


def _b21_seed_all(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _b21_parse_save_steps(total_transitions):
    raw = os.environ.get("B21_SAVE_STATE_STEPS", "").strip()
    if not raw:
        return set()
    out = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        step = int(token)
        if step < 0 or step >= total_transitions:
            raise ValueError(
                f"B21 save step {step} must satisfy 0 <= step < {total_transitions}"
            )
        out.add(step)
    return out


def _b21_load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _b21_state_dir(kwargs):
    value = kwargs.get("continuation_state_dir", None)
    if value is None:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _b21_save_payload(path, x0y, step, sigma):
    payload = {
        "version": 1,
        "x0y": x0y.detach().cpu(),
        "step": int(step),
        "sigma": float(torch.as_tensor(sigma).detach().cpu().item()),
        "source_seed": int(os.environ.get("B21_START_NOISE_SEED", os.environ.get("B21_SOURCE_SEED", "-1"))),
        "measurement_path": os.environ.get("B21_MEASUREMENT_PATH", ""),
    }
    torch.save(payload, path)
    print(f"[B21 CONT save] path={path} step={step} sigma={payload['sigma']:.8g}")


def _b21_maybe_save_initial_state(x_start, scheduler, total_transitions, kwargs):
    save_steps = _b21_parse_save_steps(total_transitions)
    if 0 not in save_steps:
        return
    state_dir = _b21_state_dir(kwargs)
    if state_dir is None:
        raise ValueError("B21_SAVE_STATE_STEPS includes 0 but continuation_state_dir was not supplied")
    # Synthetic step-0 clean state.  With B21_START_NOISE_SEED on the source run
    # and the same B21_CONT_NOISE_SEED on continuation, this reconstructs the
    # source x_start exactly as 0 + sigma[0] * eps(seed).
    _b21_save_payload(
        state_dir / "step0000.pt",
        torch.zeros_like(x_start),
        0,
        scheduler.sigma_steps[0],
    )


def _b21_maybe_save_mid_state(x0y, completed_step, scheduler, total_transitions, kwargs):
    next_step = int(completed_step) + 1
    if next_step >= total_transitions:
        return
    save_steps = _b21_parse_save_steps(total_transitions)
    if next_step not in save_steps:
        return
    state_dir = _b21_state_dir(kwargs)
    if state_dir is None:
        raise ValueError("B21_SAVE_STATE_STEPS was set but continuation_state_dir was not supplied")
    _b21_save_payload(
        state_dir / f"step{next_step:04d}.pt",
        x0y,
        next_step,
        scheduler.sigma_steps[next_step],
    )


def _b21_prepare_continuation(x_start, scheduler, total_transitions):
    if not _b21_env_enabled("B21_CONT_ENABLE"):
        return 0, x_start

    state_path = os.environ.get("B21_CONT_STATE_PATH", "").strip()
    if not state_path:
        raise ValueError("B21_CONT_ENABLE=1 requires B21_CONT_STATE_PATH")
    noise_seed_raw = os.environ.get("B21_CONT_NOISE_SEED", "").strip()
    if not noise_seed_raw:
        raise ValueError("B21_CONT_ENABLE=1 requires B21_CONT_NOISE_SEED")

    payload = _b21_load_payload(state_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Continuation payload must be a dict, got {type(payload)}")
    if "step" not in payload or "x0y" not in payload:
        raise KeyError("Continuation payload requires keys: x0y, step")

    start_step = int(payload["step"])
    if start_step < 0 or start_step >= total_transitions:
        raise ValueError(
            f"Continuation step {start_step} must satisfy 0 <= step < {total_transitions}"
        )

    x0y = payload["x0y"]
    if not torch.is_tensor(x0y):
        raise TypeError(f"payload['x0y'] must be a tensor, got {type(x0y)}")
    x0y = x0y.to(device=x_start.device, dtype=x_start.dtype)
    if tuple(x0y.shape) != tuple(x_start.shape):
        raise ValueError(
            f"Continuation shape mismatch: payload={tuple(x0y.shape)} expected={tuple(x_start.shape)}"
        )

    noise_seed = int(noise_seed_raw)
    _b21_seed_all(noise_seed)
    sigma = scheduler.sigma_steps[start_step].to(device=x0y.device, dtype=x0y.dtype)
    xt = x0y + torch.randn_like(x0y) * sigma

    saved_sigma = payload.get("sigma", None)
    print(
        "[B21 CONT load] "
        f"path={state_path} start_step={start_step} "
        f"grid_sigma={float(sigma.detach().cpu().item()):.8g} "
        f"saved_sigma={saved_sigma} noise_seed={noise_seed}"
    )
    return start_step, xt

# --- end B21.3 continuation patch -------------------------------------------
'''.strip("\n")


POSTERIOR_NOTE = r'''
# --- B21.3 continuation runner hook -----------------------------------------
# The sampler reads B21_CONT_* and B21_SAVE_STATE_STEPS directly.  The main
# runner only avoids consuming a random x_start for continuation runs and gives
# the sampler an experiment-local state directory.
# --- end B21.3 continuation runner hook -------------------------------------
'''.strip("\n")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def transform_sampler(text: str) -> str:
    if SAMPLER_MARKER in text:
        return text
    if "# --- B20.11 experimental low-frequency measurement guidance" not in text:
        raise RuntimeError("B20 LF patch marker missing from external/daps/sampler.py")

    text = replace_once(
        text,
        "class DAPS(nn.Module):",
        HELPERS + "\n\nclass DAPS(nn.Module):",
        "DAPS class",
    )

    old_loop = (
        "        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)\n"
        "        xt = x_start\n"
        "        for step in pbar:\n"
    )
    new_loop = (
        "        total_transitions = self.annealing_scheduler.num_steps - 1\n"
        "        start_step, xt = _b21_prepare_continuation(\n"
        "            x_start, self.annealing_scheduler, total_transitions\n"
        "        )\n"
        "        _b21_maybe_save_initial_state(\n"
        "            x_start, self.annealing_scheduler, total_transitions, kwargs\n"
        "        )\n"
        "        step_range = range(start_step, total_transitions)\n"
        "        pbar = tqdm.tqdm(step_range, total=total_transitions - start_step) if verbose else step_range\n"
        "        for step in pbar:\n"
    )
    text = replace_once(text, old_loop, new_loop, "pixel-DAPS loop")

    lf_line = (
        "            x0y = _b20_apply_lf_guidance(x0y, measurement, step, "
        "self.annealing_scheduler.num_steps)\n"
    )
    save_block = lf_line + (
        "            _b21_maybe_save_mid_state(\n"
        "                x0y, step, self.annealing_scheduler, total_transitions, kwargs\n"
        "            )\n"
    )
    text = replace_once(text, lf_line, save_block, "post-LF state-save insertion")

    old_start = (
        "        x_start = torch.randn(batch_size, *in_shape, device=device) * "
        "self.annealing_scheduler.get_prior_sigma()\n"
    )
    new_start = (
        "        start_seed = os.environ.get(\"B21_START_NOISE_SEED\", \"\").strip()\n"
        "        if start_seed:\n"
        "            _b21_seed_all(int(start_seed))\n"
        "            print(f\"[B21 start seed] seed={start_seed}\")\n"
        "        x_start = torch.randn(batch_size, *in_shape, device=device) * "
        "self.annealing_scheduler.get_prior_sigma()\n"
    )
    text = replace_once(text, old_start, new_start, "get_start noise line")
    return text


def transform_posterior(text: str) -> str:
    if POSTERIOR_MARKER in text:
        return text
    if "# --- B20 fixed-measurement patch" not in text:
        raise RuntimeError("B20 fixed-measurement marker missing from external/daps/posterior_sample.py")

    text = replace_once(
        text,
        "@hydra.main(version_base='1.3', config_path='configs', config_name='default.yaml')",
        POSTERIOR_NOTE + "\n\n@hydra.main(version_base='1.3', config_path='configs', config_name='default.yaml')",
        "Hydra decorator",
    )

    old_sig = (
        "def sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose, "
        "record, batch_size, gt, args, root, run_id):"
    )
    new_sig = (
        "def sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose, "
        "record, batch_size, gt, args, root, run_id, continuation_state_dir=None):"
    )
    text = replace_once(text, old_sig, new_sig, "sample_in_batch signature")

    old_call = (
        "        cur_samples = sampler.sample(model, cur_x_start, operator, cur_y, evaluator, "
        "verbose=verbose, record=record, gt=cur_gt)"
    )
    new_call = (
        "        cur_samples = sampler.sample(\n"
        "            model, cur_x_start, operator, cur_y, evaluator,\n"
        "            verbose=verbose, record=record, gt=cur_gt,\n"
        "            continuation_state_dir=continuation_state_dir,\n"
        "        )"
    )
    text = replace_once(text, old_call, new_call, "sampler.sample call")

    old_main = (
        "        x_start = sampler.get_start(images.shape[0], model)\n"
        "        samples, trajs = sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose=True, record=args.save_traj, \n"
        "                                         batch_size=args.batch_size, gt=images, args=args, root=root, run_id=r)\n"
    )
    new_main = (
        "        continuation_enabled = os.environ.get(\"B21_CONT_ENABLE\", \"0\").strip() == \"1\"\n"
        "        if continuation_enabled:\n"
        "            device = next(model.parameters()).device\n"
        "            x_start = torch.zeros(\n"
        "                images.shape[0], *model.get_in_shape(), device=device\n"
        "            )\n"
        "        else:\n"
        "            x_start = sampler.get_start(images.shape[0], model)\n"
        "\n"
        "        continuation_state_dir = None\n"
        "        if os.environ.get(\"B21_SAVE_STATE_STEPS\", \"\").strip():\n"
        "            continuation_state_dir = root / \"continuation_states\" / f\"run{r:04d}\"\n"
        "            continuation_state_dir.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "        samples, trajs = sample_in_batch(\n"
        "            sampler, model, x_start, operator, y, evaluator,\n"
        "            verbose=True, record=args.save_traj, batch_size=args.batch_size,\n"
        "            gt=images, args=args, root=root, run_id=r,\n"
        "            continuation_state_dir=continuation_state_dir,\n"
        "        )\n"
    )
    text = replace_once(text, old_main, new_main, "main sampling invocation")
    return text


def make_diff(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate anchors and print diff summary")
    mode.add_argument("--apply", action="store_true", help="apply edits and write exact unified diff")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/egr/research-pac/huang248/pr_diffusion_b19_solver"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    sampler_path = repo / "external/daps/sampler.py"
    posterior_path = repo / "external/daps/posterior_sample.py"
    patch_path = repo / "docs/b21/patches/daps_b21_continuation.patch"

    for path in (sampler_path, posterior_path):
        if not path.exists():
            raise FileNotFoundError(path)

    sampler_before = sampler_path.read_text()
    posterior_before = posterior_path.read_text()
    sampler_after = transform_sampler(sampler_before)
    posterior_after = transform_posterior(posterior_before)

    diff = make_diff("sampler.py", sampler_before, sampler_after)
    diff += make_diff("posterior_sample.py", posterior_before, posterior_after)

    changed = (sampler_before != sampler_after) or (posterior_before != posterior_after)
    print(f"[B21.3 patch] repo={repo}")
    print(f"[B21.3 patch] changed={changed} diff_lines={len(diff.splitlines())}")

    if args.check:
        if not changed:
            print("[B21.3 patch] markers already present; nothing to apply")
        else:
            print("[B21.3 patch] anchor validation passed")
        return 0

    if changed:
        sampler_path.write_text(sampler_after)
        posterior_path.write_text(posterior_after)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    if diff:
        patch_path.write_text(diff)
    elif not patch_path.exists():
        patch_path.write_text("# B21.3 continuation patch already present; no incremental diff generated.\n")

    print(f"[write] {patch_path}")
    print("[B21.3 patch] applied successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
