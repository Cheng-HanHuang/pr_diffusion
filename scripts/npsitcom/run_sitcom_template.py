#!/usr/bin/env python3
"""Run a SITCOM command template over NP handoff states.

The template can use fields from handoff_manifest.csv, especially:
  {state_path} {outdir} {image_basename} {measurement_noise_std} {seed}
  {config_tag} {handoff_timestep}

Use --dry_run first. This wrapper exists because SITCOM_ODE is external and may
have a local command-line interface that is not part of this repo.
"""
from __future__ import annotations

import argparse, csv, os, shlex, subprocess
from pathlib import Path


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def safe_name(s):
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(s))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--cmd_template', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--max_rows', type=int, default=None)
    ap.add_argument('--dry_run', action='store_true')
    args=ap.parse_args()
    rows=read_csv(args.manifest)
    if args.max_rows is not None: rows=rows[:args.max_rows]
    os.makedirs(args.outdir, exist_ok=True)
    for i,row in enumerate(rows):
        case=f"{i:05d}_{safe_name(row.get('image_basename','img'))}_n{safe_name(row.get('measurement_noise_std',''))}_{safe_name(row.get('config_tag','cfg'))}_s{safe_name(row.get('seed',''))}_t{safe_name(row.get('handoff_timestep',''))}"
        case_out=os.path.join(args.outdir,case); os.makedirs(case_out, exist_ok=True)
        fmt=dict(row); fmt['outdir']=case_out
        cmd=args.cmd_template.format(**fmt)
        print('[sitcom-template]', cmd, flush=True)
        if not args.dry_run:
            subprocess.run(cmd, shell=True, check=True)

if __name__ == '__main__':
    main()
