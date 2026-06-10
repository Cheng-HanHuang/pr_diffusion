#!/usr/bin/env python3
"""Inspect local SITCOM_ODE tree for likely runnable entrypoints.

Run on PAC. This script only lists files and grep hits; it does not execute the
external solver.
"""
from __future__ import annotations

import argparse, os
from pathlib import Path

KEYWORDS = ('argparse', 'phase', 'retrieval', 'ffhq', 'inverse', 'ode', 'main')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sitcom_root', default='/egr/research-pac/huang248/external/SITCOM_ODE')
    ap.add_argument('--max_files', type=int, default=200)
    args=ap.parse_args()
    root=Path(args.sitcom_root)
    if not root.exists():
        raise SystemExit(f'missing SITCOM root: {root}')
    py=list(root.rglob('*.py'))[:args.max_files]
    sh=list(root.rglob('*.sh'))[:args.max_files]
    print(f'SITCOM root: {root}')
    print(f'Python files: {len(py)} shown up to {args.max_files}')
    for p in py:
        rel=p.relative_to(root)
        score=0
        try:
            txt=p.read_text(errors='ignore').lower()
            score=sum(k in txt for k in KEYWORDS)
        except Exception:
            pass
        if score >= 2 or p.name in {'main.py','run.py','sample.py','test.py','eval.py'}:
            print(f'[py score={score}] {rel}')
    print(f'Shell files: {len(sh)} shown up to {args.max_files}')
    for p in sh:
        print('[sh]', p.relative_to(root))
    print('\nSuggested next step: inspect the highest-score Python or shell entrypoint and build a cmd_template for run_sitcom_template.py')

if __name__ == '__main__':
    main()
