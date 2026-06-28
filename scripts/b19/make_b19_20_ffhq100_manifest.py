#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


REPO = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver")
BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
DATA_DIR = Path("/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024/00000")

PANEL_NAME = "B19_20_ffhq100_seed20260627_from00000to00999_exclude_ffhq25"
SAMPLE_SEED = 20260627
N = 100

OLD_FFHQ25 = {
    "00000", "00004", "00005", "00007", "00008",
    "00009", "00010", "00011", "00012", "00013",
    "00014", "00015", "00016", "00017", "00018",
    "00019", "00020", "00025", "00027", "00028",
    "00029", "00032", "00034", "00037", "00039",
}


def main() -> None:
    candidates = []
    for i in range(1000):
        image_id = f"{i:05d}"
        p = DATA_DIR / f"{image_id}.png"
        if image_id in OLD_FFHQ25:
            continue
        if p.exists():
            candidates.append(image_id)

    if len(candidates) < N:
        raise RuntimeError(f"Only {len(candidates)} candidates found, need {N}")

    rng = np.random.default_rng(SAMPLE_SEED)
    selected = sorted(rng.choice(candidates, size=N, replace=False).tolist())

    rows = []
    for rank, image_id in enumerate(selected):
        rows.append({
            "panel_name": PANEL_NAME,
            "sample_seed": SAMPLE_SEED,
            "rank": rank,
            "image_id": image_id,
            "source_path": str(DATA_DIR / f"{image_id}.png"),
            "excluded_old_ffhq25": image_id in OLD_FFHQ25,
        })

    outdir = BASE / "manifests"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{PANEL_NAME}.csv"
    txt_path = outdir / f"{PANEL_NAME}_ids.txt"
    json_path = outdir / f"{PANEL_NAME}.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    txt_path.write_text("\n".join(selected) + "\n")

    meta = {
        "panel_name": PANEL_NAME,
        "sample_seed": SAMPLE_SEED,
        "n": N,
        "source_dir": str(DATA_DIR),
        "candidate_range": "00000--00999",
        "excluded_old_ffhq25": sorted(OLD_FFHQ25),
        "image_ids": selected,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")

    # Also place a copy under docs so the panel is version-trackable.
    docs_dir = REPO / "docs/b19"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_txt = docs_dir / f"{PANEL_NAME}_ids.txt"
    docs_json = docs_dir / f"{PANEL_NAME}.json"
    docs_txt.write_text(txt_path.read_text())
    docs_json.write_text(json_path.read_text())

    print("[write]", csv_path)
    print("[write]", txt_path)
    print("[write]", json_path)
    print("[write]", docs_txt)
    print("[write]", docs_json)

    print("\n== selected image IDs ==")
    print(" ".join(selected))


if __name__ == "__main__":
    main()
