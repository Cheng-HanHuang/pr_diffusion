#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import torch


def run_index_from_name(path: Path) -> int:
    m = re.search(r"run(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse run index from {path}")
    return int(m.group(1))


def scalarize_loss(loss) -> float:
    if torch.is_tensor(loss):
        return float(loss.detach().flatten()[0].cpu().item())
    return float(loss)


def batch_rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().cpu().item())


def load_metric_psnr(metrics_json: Path) -> dict[int, float]:
    m = json.loads(metrics_json.read_text())
    ps = m["psnr"]["sample"][0]
    return {i: float(v) for i, v in enumerate(ps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daps_root", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--measurement_path", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--out_step_csv", required=True)
    ap.add_argument("--out_window_csv", required=True)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--oversample", type=float, default=2.0)
    ap.add_argument("--good_threshold", type=float, default=25.0)
    args = ap.parse_args()

    daps_root = Path(args.daps_root).resolve()
    sys.path.insert(0, str(daps_root))

    from forward_operator import get_operator  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    operator = get_operator(name="phase_retrieval", sigma=args.noise, oversample=args.oversample)

    payload = torch.load(args.measurement_path, map_location=device)
    y = payload["measurement"] if isinstance(payload, dict) and "measurement" in payload else payload
    y = y.to(device)
    y_norm = float(y.flatten(1).norm(dim=1)[0].detach().cpu().item())

    final_psnr = load_metric_psnr(Path(args.metrics_json))

    rows = []
    raw_dir = Path(args.raw_dir)
    for p in sorted(raw_dir.glob("trajectory_run*.pth")):
        run_i = run_index_from_name(p)
        print(f"[load] {p}")
        traj = torch.load(p, map_location="cpu", weights_only=False)

        xt = traj.tensor_data["xt"]
        x0y = traj.tensor_data["x0y"]
        x0hat = traj.tensor_data["x0hat"]
        sigmas = traj.value_data["sigma"].detach().cpu().float()

        n_steps = int(x0y.shape[0])
        prev_xt = None
        prev_x0y = None
        prev_x0hat = None

        fp = final_psnr[run_i]
        is_good25 = int(fp >= args.good_threshold)

        for step in range(n_steps):
            x0y_s = x0y[step].to(device)
            x0hat_s = x0hat[step].to(device)
            xt_s = xt[step].to(device)

            with torch.no_grad():
                loss_x0y = scalarize_loss(operator.loss(x0y_s, y))
                loss_x0hat = scalarize_loss(operator.loss(x0hat_s, y))

            correction = batch_rms(x0y[step] - x0hat[step])
            xt_jump = 0.0 if prev_xt is None else batch_rms(xt[step] - prev_xt)
            x0y_jump = 0.0 if prev_x0y is None else batch_rms(x0y[step] - prev_x0y)
            x0hat_jump = 0.0 if prev_x0hat is None else batch_rms(x0hat[step] - prev_x0hat)

            rows.append({
                "run_index": run_i,
                "step": step,
                "sigma": float(sigmas[step]),
                "final_psnr": fp,
                "is_good25": is_good25,
                "exact_loss_x0y": loss_x0y,
                "exact_loss_x0hat": loss_x0hat,
                "sqrt_loss_x0y_over_y_norm": math.sqrt(max(loss_x0y, 0.0)) / (y_norm + 1e-12),
                "sqrt_loss_x0hat_over_y_norm": math.sqrt(max(loss_x0hat, 0.0)) / (y_norm + 1e-12),
                "correction_rms": correction,
                "xt_jump_rms": xt_jump,
                "x0y_jump_rms": x0y_jump,
                "x0hat_jump_rms": x0hat_jump,
            })

            prev_xt = xt[step].clone()
            prev_x0y = x0y[step].clone()
            prev_x0hat = x0hat[step].clone()

        del traj, xt, x0y, x0hat
        torch.cuda.empty_cache()

    step_df = pd.DataFrame(rows)

    # Within-step ranks across particles. Low rank means smaller/better for these features.
    rank_features = [
        "exact_loss_x0y",
        "exact_loss_x0hat",
        "sqrt_loss_x0y_over_y_norm",
        "sqrt_loss_x0hat_over_y_norm",
        "correction_rms",
        "xt_jump_rms",
        "x0y_jump_rms",
        "x0hat_jump_rms",
    ]
    for feat in rank_features:
        step_df[f"{feat}_rank"] = step_df.groupby("step")[feat].rank(method="min", ascending=True)

    Path(args.out_step_csv).parent.mkdir(parents=True, exist_ok=True)
    step_df.to_csv(args.out_step_csv, index=False)
    print("[write]", args.out_step_csv)

    checkpoints = [10, 20, 30, 40, 50, 75, 100, 125, 150, 199]
    checkpoints = [c for c in checkpoints if c <= int(step_df["step"].max())]

    window_rows = []
    for run_i, g in step_df.groupby("run_index"):
        g = g.sort_values("step")
        fp = float(g["final_psnr"].iloc[0])
        good = int(g["is_good25"].iloc[0])

        for ckpt in checkpoints:
            w = g[g["step"] <= ckpt]
            last = g[g["step"] == ckpt].iloc[0]

            row = {
                "run_index": int(run_i),
                "checkpoint_step": int(ckpt),
                "checkpoint_sigma": float(last["sigma"]),
                "final_psnr": fp,
                "is_good25": good,
            }

            for feat in rank_features:
                row[f"{feat}_last"] = float(last[feat])
                row[f"{feat}_mean"] = float(w[feat].mean())
                row[f"{feat}_min"] = float(w[feat].min())
                row[f"{feat}_max"] = float(w[feat].max())
                row[f"{feat}_rank_last"] = float(last[f"{feat}_rank"])
                row[f"{feat}_rank_mean"] = float(w[f"{feat}_rank"].mean())

            window_rows.append(row)

    win_df = pd.DataFrame(window_rows).sort_values(["checkpoint_step", "run_index"])
    win_df.to_csv(args.out_window_csv, index=False)
    print("[write]", args.out_window_csv)

    print("\nFinal labels:")
    print(
        step_df.groupby("run_index")[["final_psnr", "is_good25"]]
        .first()
        .reset_index()
        .to_string(index=False)
    )

    print("\nCompact checkpoint view:")
    compact_cols = [
        "run_index",
        "checkpoint_step",
        "checkpoint_sigma",
        "final_psnr",
        "exact_loss_x0y_rank_last",
        "exact_loss_x0hat_rank_last",
        "correction_rms_rank_last",
        "x0y_jump_rms_rank_last",
        "sqrt_loss_x0y_over_y_norm_last",
        "correction_rms_last",
    ]
    print(win_df[compact_cols].to_string(index=False))


if __name__ == "__main__":
    main()
