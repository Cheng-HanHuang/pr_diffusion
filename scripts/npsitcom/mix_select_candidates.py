#!/usr/bin/env python3
"""Branch A: mix NP and SITCOM candidate CSVs and select without GT.

Input CSVs should be run-level tables with at least image/noise/alignment/psnr.
NP run_level.csv already has measurement-side selector columns.  SITCOM rows can
be standardized with any column names as long as the final names match.
"""
from __future__ import annotations

import argparse, csv, math, os
from collections import defaultdict
from statistics import mean, median, stdev
from typing import Dict, List, Tuple


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f'No rows for {path}')
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def as_float(x, default=math.nan):
    try:
        if x is None or x == '': return default
        return float(x)
    except Exception:
        return default


def fmean(xs):
    xs=[x for x in xs if math.isfinite(x)]
    return mean(xs) if xs else math.nan

def fmedian(xs):
    xs=[x for x in xs if math.isfinite(x)]
    return median(xs) if xs else math.nan

def fstd(xs):
    xs=[x for x in xs if math.isfinite(x)]
    return stdev(xs) if len(xs)>1 else 0.0


def canonicalize_row(row: Dict[str,str], source: str, idx: int) -> Dict[str, object]:
    out=dict(row)
    out['candidate_source']=source
    out['candidate_id']=f'{source}:{idx}'
    if 'measurement_noise_std' not in out and 'noise' in out:
        out['measurement_noise_std']=out['noise']
    if 'alignment_mode' not in out:
        out['alignment_mode']='resolve'
    if 'image_basename' not in out and 'image' in out:
        out['image_basename']=out['image']
    if 'config_tag' not in out:
        out['config_tag']=source
    if 'seed' not in out:
        out['seed']=''
    if 'selector_post_winner_lf_mse_mean' not in out:
        out['selector_post_winner_lf_mse_mean']='nan'
    if 'noisy_lowfreq_mag_l2' not in out:
        out['noisy_lowfreq_mag_l2']='nan'
    if 'noisy_mag_l2' not in out:
        out['noisy_mag_l2']='nan'
    return out


def load_candidates(specs: List[str]) -> List[Dict[str, object]]:
    rows=[]
    for spec in specs:
        if ':' not in spec:
            raise ValueError('--candidate_csv must be source:/path/file.csv')
        source,path=spec.split(':',1)
        for i,r in enumerate(read_csv(path)):
            rows.append(canonicalize_row(r, source, i))
    return rows


def group_key(r):
    return (str(r.get('image_basename','')), str(r.get('measurement_noise_std','')), str(r.get('alignment_mode','')))


def choose(rows: List[Dict[str,object]], method: str) -> Dict[str,object]:
    if method == 'oracle_best_psnr_diagnostic':
        return max(rows, key=lambda r: as_float(r.get('psnr')))
    if method == 'min_selector_post_lf_mse':
        valid=[r for r in rows if math.isfinite(as_float(r.get('selector_post_winner_lf_mse_mean')))]
        return min(valid or rows, key=lambda r: as_float(r.get('selector_post_winner_lf_mse_mean'), 1e99))
    if method == 'min_noisy_lowfreq_mag_l2':
        valid=[r for r in rows if math.isfinite(as_float(r.get('noisy_lowfreq_mag_l2')))]
        return min(valid or rows, key=lambda r: as_float(r.get('noisy_lowfreq_mag_l2'), 1e99))
    if method == 'min_noisy_mag_l2':
        valid=[r for r in rows if math.isfinite(as_float(r.get('noisy_mag_l2')))]
        return min(valid or rows, key=lambda r: as_float(r.get('noisy_mag_l2'), 1e99))
    raise ValueError(method)


def summarize(rows: List[Dict[str,object]], group_cols: List[str]) -> List[Dict[str,object]]:
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(c,'')) for c in group_cols)].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        ps=[as_float(r.get('psnr')) for r in rs]
        worst=min(rs, key=lambda r: as_float(r.get('psnr')))
        row={c:v for c,v in zip(group_cols,key)}
        row.update(n=len(rs), psnr_mean=fmean(ps), psnr_median=fmedian(ps), psnr_min=min(ps), psnr_max=max(ps), psnr_std=fstd(ps), n_below25=sum(p<25 for p in ps), worst_image=worst.get('image_basename',''), worst_psnr=as_float(worst.get('psnr')))
        out.append(row)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate_csv', action='append', required=True, help='source:/path/to/run_level.csv')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--alignment', default='resolve', help='alignment to select/summarize, or all')
    args=ap.parse_args()
    candidates=load_candidates(args.candidate_csv)
    if args.alignment != 'all':
        candidates=[r for r in candidates if str(r.get('alignment_mode','')) == args.alignment]
    write_csv(os.path.join(args.outdir,'candidate_level.csv'), candidates)
    write_csv(os.path.join(args.outdir,'source_summary.csv'), summarize(candidates, ['candidate_source','measurement_noise_std','alignment_mode']))

    methods=['oracle_best_psnr_diagnostic','min_selector_post_lf_mse','min_noisy_lowfreq_mag_l2','min_noisy_mag_l2']
    selected=[]
    by=defaultdict(list)
    for r in candidates: by[group_key(r)].append(r)
    for key,rs in by.items():
        for m in methods:
            ch=dict(choose(rs,m)); ch['selection_method']=m; selected.append(ch)
    write_csv(os.path.join(args.outdir,'selected_image_level.csv'), selected)
    write_csv(os.path.join(args.outdir,'selected_summary.csv'), summarize(selected, ['selection_method','measurement_noise_std','alignment_mode']))
    print('wrote', args.outdir)

if __name__ == '__main__':
    main()
