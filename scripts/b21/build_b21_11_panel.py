#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

EXCLUDED = {
    "62802", "63282", "63803", "65808", "65960", "66452", "66892", "68263", "68924", "69293",
    "60067", "62957", "63135", "63199", "63319", "63368", "63678", "64050", "64116", "64471",
    "64542", "65317", "65656", "66511", "66731", "67092", "67673", "68111", "68922", "69441",
}


def source_index(root: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = {}
    for path in root.rglob("*.png"):
        stem = path.stem
        if len(stem) != 5 or not stem.isdigit():
            continue
        value = int(stem)
        if 60000 <= value <= 69999:
            found.setdefault(stem, []).append(path.resolve())
    ambiguous = {key: paths for key, paths in found.items() if len(paths) != 1}
    if ambiguous:
        preview = {key: [str(p) for p in paths[:3]] for key, paths in list(ambiguous.items())[:10]}
        raise ValueError(f"ambiguous official-validation image ids: {preview}")
    return {key: paths[0] for key, paths in found.items()}


def digest_for(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"b21.11|{seed}|{image_id}".encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--panel-seed", type=int, default=5401)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed1-base", type=int, default=22000)
    parser.add_argument("--seed2-base", type=int, default=23000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.count != 100:
        raise ValueError("B21.11 is frozen to exactly 100 images")
    if args.panel_seed != 5401:
        raise ValueError("B21.11 panel seed is frozen to 5401")

    root = args.image_root.resolve()
    index = source_index(root)
    available = sorted(set(index) - EXCLUDED)
    if len(available) < args.count:
        raise ValueError(f"only {len(available)} eligible images available")

    ranked = sorted(
        ((digest_for(args.panel_seed, image_id), image_id) for image_id in available),
        key=lambda item: (item[0], item[1]),
    )
    selected = ranked[: args.count]
    rows = []
    for row_id, (digest, image_id) in enumerate(selected):
        rows.append({
            "row_id": row_id,
            "image_id": image_id,
            "selection_digest": digest,
            "source_path": str(index[image_id]),
            "seed1": args.seed1_base + row_id,
            "seed2": args.seed2_base + row_id,
        })

    if len({row["image_id"] for row in rows}) != args.count:
        raise ValueError("duplicate selected image ids")
    overlap = sorted({row["image_id"] for row in rows} & EXCLUDED)
    if overlap:
        raise ValueError(f"excluded images selected: {overlap}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.outdir / "panel_manifest.tsv"
    json_path = args.outdir / "panel_manifest.json"
    checksum_path = args.outdir / "panel_manifest.sha256"

    if tsv_path.exists() and not args.force:
        prior = tsv_path.read_text()
        keys = list(rows[0].keys())
        current_lines = ["\t".join(keys)]
        current_lines.extend("\t".join(str(row[key]) for key in keys) for row in rows)
        current = "\n".join(current_lines) + "\n"
        if prior != current:
            raise RuntimeError(
                f"existing frozen panel differs from deterministic reconstruction: {tsv_path}; "
                "do not overwrite without investigating"
            )
    else:
        with tsv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "protocol": "B21.11 deterministic official-FFHQ-validation panel",
        "image_root": str(root),
        "panel_seed": args.panel_seed,
        "count": args.count,
        "seed1_base": args.seed1_base,
        "seed2_base": args.seed2_base,
        "excluded_images": sorted(EXCLUDED),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    checksum = file_sha256(tsv_path)
    checksum_path.write_text(f"{checksum}  {tsv_path.name}\n")

    print(f"[panel] eligible={len(available)} selected={len(rows)}")
    print(f"[write] {tsv_path}")
    print(f"[write] {json_path}")
    print(f"[write] {checksum_path}")
    print(f"[sha256] {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
