#!/usr/bin/env python3
"""Inspect external DiffFPR/SITCOM_ODE folders and record likely entrypoints.

This does not try to force-run unknown public code.  It creates a manifest that
helps decide whether to wrap official baselines directly or keep them as
separate commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CANDIDATE_NAMES = [
    "README.md", "readme.md", "requirements.txt", "environment.yml",
    "main.py", "test.py", "eval.py", "sample.py", "run.py", "scripts",
]

KEYWORDS = ["phase", "retrieval", "ffhq", "imagenet", "celeba", "eval", "sample", "main", "sitcom", "difffpr"]


def inspect(root: Path):
    info = {"root": str(root), "exists": root.exists(), "candidate_paths": [], "python_files": []}
    if not root.exists():
        return info
    for name in CANDIDATE_NAMES:
        p = root / name
        if p.exists():
            info["candidate_paths"].append(str(p))
    for p in sorted(root.rglob("*.py")):
        rel = str(p.relative_to(root))
        low = rel.lower()
        if any(k in low for k in KEYWORDS):
            info["python_files"].append(rel)
        if len(info["python_files"]) >= 80:
            break
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difffpr", default="/egr/research-pac/huang248/external/DiffFPR")
    ap.add_argument("--sitcom", default="/egr/research-pac/huang248/external/SITCOM_ODE")
    ap.add_argument("--out", default="/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/manifests/external_solver_entrypoints.json")
    args = ap.parse_args()
    report = {"DiffFPR": inspect(Path(args.difffpr)), "SITCOM_ODE": inspect(Path(args.sitcom))}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
