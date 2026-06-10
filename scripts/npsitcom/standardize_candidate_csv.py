#!/usr/bin/env python3
"""Standardize an external solver CSV into the run-level schema used by Branch A.

This is intentionally simple and configurable because SITCOM_ODE output column
names may differ across local scripts.  It does not compute metrics from images;
it only renames/copies columns that already exist.
"""
from __future__ import annotations

import argparse, csv, os
from typing import Dict, List


def read_csv(path: str) -> List[Dict[str,str]]:
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
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def pick(row: Dict[str,str], names: str, default: str='') -> str:
    for n in [x.strip() for x in names.split(',') if x.strip()]:
        if n in row and row[n] != '': return row[n]
    return default


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--source', default='sitcom')
    ap.add_argument('--image_cols', default='image_basename,image,image_name,filename,name')
    ap.add_argument('--noise_cols', default='measurement_noise_std,noise,noise_std,sigma')
    ap.add_argument('--seed_cols', default='seed,run_seed,random_seed')
    ap.add_argument('--psnr_cols', default='psnr,PSNR,psnr_resolve')
    ap.add_argument('--ssim_cols', default='ssim,SSIM,ssim_resolve')
    ap.add_argument('--lpips_cols', default='lpips,LPIPS,lpips_resolve')
    ap.add_argument('--alignment', default='resolve')
    args=ap.parse_args()

    rows=[]
    for i,r in enumerate(read_csv(args.input_csv)):
        out=dict(r)
        out.update(
            candidate_source=args.source,
            candidate_id=f'{args.source}:{i}',
            image_basename=pick(r,args.image_cols),
            measurement_noise_std=pick(r,args.noise_cols,'nan'),
            seed=pick(r,args.seed_cols,''),
            alignment_mode=pick(r,'alignment_mode,alignment',args.alignment),
            config_tag=pick(r,'config_tag,method,solver',args.source),
            psnr=pick(r,args.psnr_cols,'nan'),
            ssim=pick(r,args.ssim_cols,'nan'),
            lpips=pick(r,args.lpips_cols,'nan'),
            selector_post_winner_lf_mse_mean='nan',
            noisy_lowfreq_mag_l2=pick(r,'noisy_lowfreq_mag_l2,lowfreq_residual,lf_residual','nan'),
            noisy_mag_l2=pick(r,'noisy_mag_l2,mag_residual,measurement_residual','nan'),
        )
        rows.append(out)
    write_csv(args.output_csv, rows)
    print('wrote', args.output_csv, 'rows', len(rows))

if __name__ == '__main__':
    main()
