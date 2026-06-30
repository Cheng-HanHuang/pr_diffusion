#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import sys
import math
import pandas as pd
import numpy as np
import torch
from PIL import Image


REPO = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver")
BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
DAPS_ROOT = REPO / "external/daps"

sys.path.insert(0, str(DAPS_ROOT))

from forward_operator import get_operator  # noqa: E402


IDS = [x.strip().zfill(5) for x in os.environ.get(
    "IDS", "00046,00154,00171"
).replace(" ", ",").split(",") if x.strip()]

RUN_SEEDS = [int(x) for x in os.environ.get(
    "RUN_SEEDS", "4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700"
).replace(" ", ",").split(",") if x.strip()]

MEAS_SEEDS = [int(x) for x in os.environ.get(
    "MEAS_SEEDS", "5001,5002,5003"
).replace(" ", ",").split(",") if x.strip()]

K = int(os.environ.get("K", "16"))
ONLY_BAD_CASES = int(os.environ.get("ONLY_BAD_CASES", "1"))
GOOD_THRESH = float(os.environ.get("GOOD_THRESH", "25.0"))

NOISE = float(os.environ.get("NOISE", "0.05"))
OVERSAMPLE = float(os.environ.get("OVERSAMPLE", "2.0"))

SHARD_IDX = int(os.environ.get("SHARD_IDX", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))

# Format:
#   name:prox_beta:lr:steps
CONFIGS_RAW = os.environ.get(
    "CONFIGS",
    "free_b0_lr005_s60:0:0.005:60,"
    "prox_b1_lr005_s60:1:0.005:60,"
    "prox_b10_lr005_s60:10:0.005:60",
)


def parse_configs(raw: str):
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, beta, lr, steps = item.split(":")
        out.append({
            "name": name,
            "beta": float(beta),
            "lr": float(lr),
            "steps": int(steps),
        })
    return out


CONFIGS = parse_configs(CONFIGS_RAW)


def exact_path(image_id: str, run_seed: int, meas_seed: int, num_runs: int) -> Path:
    return BASE / f"B20_1_daps{num_runs}S_{image_id}_meas{meas_seed}_runseed{run_seed}_exact_final_loss_selector.csv"


def find_exact_file(image_id: str, run_seed: int, meas_seed: int) -> tuple[Path | None, int]:
    # Prefer 32S if available, but all comparisons restrict to K.
    for n in [32, 16, 6]:
        p = exact_path(image_id, run_seed, meas_seed, n)
        if p.exists():
            return p, n
    return None, 0


def measurement_path(image_id: str, meas_seed: int) -> Path:
    return BASE / "measurements" / f"ffhq{image_id}_phase_noise005_meas{meas_seed}.pt"


def psnr_col(df: pd.DataFrame) -> str:
    for c in ["psnr_metrics_json", "psnr_recomputed_from_png"]:
        if c in df.columns:
            return c
    raise KeyError(f"No PSNR column found. Columns={list(df.columns)}")


def load_png_model_range(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    x01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x01 * 2.0 - 1.0


def psnr_model_range(x: torch.Tensor, gt: torch.Tensor) -> float:
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    g01 = (gt.clamp(-1, 1) + 1.0) / 2.0
    mse = (x01 - g01).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse))[0].detach().cpu().item())


def scalar_loss(operator, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = operator.loss(x, y)
    if torch.is_tensor(loss):
        return loss.flatten()[0]
    return torch.tensor(float(loss), device=x.device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)
    print("[IDS]", IDS)
    print("[RUN_SEEDS]", RUN_SEEDS)
    print("[MEAS_SEEDS]", MEAS_SEEDS)
    print("[K]", K)
    print("[ONLY_BAD_CASES]", ONLY_BAD_CASES)
    print("[CONFIGS]", CONFIGS)
    print("[SHARD]", SHARD_IDX, "/", NUM_SHARDS)

    operator = get_operator(name="phase_retrieval", sigma=NOISE, oversample=OVERSAMPLE)

    # Build work list at candidate level.
    work = []
    missing = []

    for image_id in IDS:
        for run_seed in RUN_SEEDS:
            for meas_seed in MEAS_SEEDS:
                ep, n_avail = find_exact_file(image_id, run_seed, meas_seed)
                mp = measurement_path(image_id, meas_seed)

                if ep is None or not mp.exists():
                    missing.append((image_id, run_seed, meas_seed, ep, mp.exists()))
                    continue

                df = pd.read_csv(ep).copy()
                df["run_index"] = df["run_index"].astype(int)
                col = psnr_col(df)
                df[col] = pd.to_numeric(df[col])

                cand = df[df["run_index"] < K].copy()
                if len(cand) < K:
                    continue

                oracle = cand.loc[cand[col].idxmax()]
                oracle_psnr = float(oracle[col])
                oracle_run = int(oracle["run_index"])

                if ONLY_BAD_CASES and oracle_psnr >= GOOD_THRESH:
                    continue

                for _, r in cand.iterrows():
                    work.append({
                        "image_id": image_id,
                        "run_seed": run_seed,
                        "meas_seed": meas_seed,
                        "num_runs_file": n_avail,
                        "K": K,
                        "candidate_run": int(r["run_index"]),
                        "sample_path": str(r["sample_path"]),
                        "orig_candidate_psnr": float(r[col]),
                        "orig_oracleK_psnr": oracle_psnr,
                        "orig_oracleK_run": oracle_run,
                        "exact_file": str(ep),
                        "measurement_path": str(mp),
                    })

    # Shard candidate-level work.
    work = [w for i, w in enumerate(work) if i % NUM_SHARDS == SHARD_IDX]

    print("[missing combos]", len(missing))
    if missing:
        print("[first missing]", missing[:10])
    print("[candidate work items in this shard]", len(work))

    rows = []

    # Cache measurement payloads by image/meas.
    meas_cache = {}

    for wi, item in enumerate(work):
        image_id = item["image_id"]
        meas_seed = item["meas_seed"]
        mp = Path(item["measurement_path"])
        sample_path = Path(item["sample_path"])

        key = (image_id, meas_seed)
        if key not in meas_cache:
            payload = torch.load(mp, map_location=device)
            y = payload["measurement"] if isinstance(payload, dict) and "measurement" in payload else payload
            y = y.to(device)
            gt = payload["images"].to(device) if isinstance(payload, dict) and "images" in payload else None
            if gt is not None and gt.shape[0] > 1:
                gt = gt[:1]
            y_norm2 = y.flatten(1).pow(2).sum(dim=1)[0].detach().clamp_min(1e-12)
            meas_cache[key] = (y, gt, y_norm2)

        y, gt, y_norm2 = meas_cache[key]

        x0 = load_png_model_range(sample_path).to(device)

        with torch.no_grad():
            orig_exact_loss = float(scalar_loss(operator, x0, y).detach().cpu().item())
            orig_psnr_recomputed = psnr_model_range(x0, gt) if gt is not None else math.nan

        for cfg in CONFIGS:
            beta = cfg["beta"]
            lr = cfg["lr"]
            steps = cfg["steps"]
            name = cfg["name"]

            x = torch.nn.Parameter(x0.detach().clone())
            opt = torch.optim.Adam([x], lr=lr)

            last_total = math.nan
            last_meas = math.nan
            last_prox = math.nan

            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                meas_loss = scalar_loss(operator, x, y) / y_norm2
                prox_loss = (x - x0).pow(2).mean()
                total = meas_loss + beta * prox_loss
                total.backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(-1.0, 1.0)

                last_total = float(total.detach().cpu().item())
                last_meas = float(meas_loss.detach().cpu().item())
                last_prox = float(prox_loss.detach().cpu().item())

            with torch.no_grad():
                refined_exact_loss = float(scalar_loss(operator, x, y).detach().cpu().item())
                refined_psnr = psnr_model_range(x, gt) if gt is not None else math.nan

            rows.append({
                **item,
                "config": name,
                "beta": beta,
                "lr": lr,
                "steps": steps,
                "orig_exact_loss": orig_exact_loss,
                "orig_psnr_recomputed": orig_psnr_recomputed,
                "refined_exact_loss": refined_exact_loss,
                "refined_psnr": refined_psnr,
                "delta_psnr": refined_psnr - orig_psnr_recomputed if gt is not None else math.nan,
                "last_total_obj": last_total,
                "last_normed_meas_obj": last_meas,
                "last_prox_obj": last_prox,
            })

        if (wi + 1) % 20 == 0:
            print(f"[progress] {wi + 1}/{len(work)} candidates")

    out = pd.DataFrame(rows)
    out_path = BASE / f"B20_5A_posthoc_phase_refinement_K{K}_shard{SHARD_IDX}of{NUM_SHARDS}.csv"
    out.to_csv(out_path, index=False)
    print("[write]", out_path)
    print("[rows]", len(out))


if __name__ == "__main__":
    main()
