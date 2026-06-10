#!/usr/bin/env python3
"""Convert SITCOM_ODE metrics.json into Branch-A run-level candidate CSV.

SITCOM_ODE writes metrics.json with entries like:
  metrics['psnr']['sample'][image_index][run_index]
  metrics['psnr']['max'][image_index]

This script expands the per-image/per-run sample table into one row per
candidate run so it can be mixed with NP run_level.csv.
"""
from __future__ import annotations

import argparse, csv, json, math, os
from pathlib import Path
from typing import Dict, List


def read_manifest(path: str) -> List[Dict[str,str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str,object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows: raise ValueError('no rows')
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def sample_value(metrics: Dict, key: str, image_i: int, run_i: int):
    try:
        return metrics[key]['sample'][image_i][run_i]
    except Exception:
        return math.nan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metrics_json', required=True)
    ap.add_argument('--image_manifest', required=True, help='manifest.csv from make_sitcom_image_folder.py')
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--noise', required=True)
    ap.add_argument('--source', default='sitcom')
    ap.add_argument('--config_tag', default='sitcom_official')
    ap.add_argument('--alignment', default='resolve')
    args=ap.parse_args()

    metrics=json.loads(Path(args.metrics_json).read_text())
    manifest=read_manifest(args.image_manifest)
    if 'psnr' not in metrics or 'sample' not in metrics['psnr']:
        raise ValueError('metrics_json does not contain metrics["psnr"]["sample"]')
    n_images=len(metrics['psnr']['sample'])
    rows=[]
    for image_i in range(n_images):
        runs=metrics['psnr']['sample'][image_i]
        meta=manifest[image_i] if image_i < len(manifest) else {}
        image_name=meta.get('split_entry') or meta.get('source_path') or f'{image_i:05d}'
        image_base=Path(image_name).stem if '/' in image_name else image_name
        for run_i in range(len(runs)):
            row={
                'candidate_source': args.source,
                'candidate_id': f'{args.source}:{image_i}:{run_i}',
                'config_tag': args.config_tag,
                'image_basename': image_base,
                'image_index_in_split': image_i,
                'seed': run_i,
                'sitcom_run_index': run_i,
                'measurement_noise_std': args.noise,
                'alignment_mode': args.alignment,
                'psnr': sample_value(metrics,'psnr',image_i,run_i),
                'ssim': sample_value(metrics,'ssim',image_i,run_i),
                'lpips': sample_value(metrics,'lpips',image_i,run_i),
                'selector_post_winner_lf_mse_mean': 'nan',
                'noisy_lowfreq_mag_l2': 'nan',
                'noisy_mag_l2': 'nan',
            }
            rows.append(row)
    write_csv(args.output_csv, rows)
    print('wrote', args.output_csv, 'rows', len(rows))

if __name__ == '__main__':
    main()
