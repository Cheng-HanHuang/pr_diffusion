#!/usr/bin/env python3
"""Fit a single A14 frozen policy candidate on A8+A11 development data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

WINDOW_FRAC = 0.80
LOWFREQ_FRAC = 0.125
SAMPLE_RE = re.compile(r"_(\d{5})_run(\d{4})\.png$")
NP_FALLBACK_CSV = "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/np_selector_ffhq/selector_full25_s100_103/lf_s2_selector_20260609_154712/run_level.csv"
NOISE = 0.05


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def mean_or_nan(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else math.nan


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def parse_sample_paths(sample_dir: Path) -> Dict[Tuple[str, int], Path]:
    out: Dict[Tuple[str, int], Path] = {}
    for path in sorted(sample_dir.glob('*.png')):
        m = SAMPLE_RE.search(path.name)
        if not m:
            continue
        out[(m.group(1), int(m.group(2)))] = path
    return out


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def lowfreq_repr(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(fft).astype(np.float32)
    h, w = mag.shape
    hh = max(1, int(round(h * LOWFREQ_FRAC / 2.0)))
    ww = max(1, int(round(w * LOWFREQ_FRAC / 2.0)))
    cy, cx = h // 2, w // 2
    return mag[cy - hh: cy + hh, cx - ww: cx + ww]


def l2_normed(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(diff.ravel()) / math.sqrt(diff.size))


def rank_within_group(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    ranks[order] = np.arange(1, arr.size + 1, dtype=float)
    return [float(x) for x in ranks]


def add_interrun_rank_features(step_rows: List[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row['image_id']), int(row['step']))].append(row)
    base = 'x0y_full_residual_normed'
    for rows in grouped.values():
        vals = np.asarray([to_float(r.get(base)) for r in rows], dtype=float)
        if vals.size == 0 or np.any(~np.isfinite(vals)):
            continue
        ranks = rank_within_group(vals)
        for row, rank in zip(rows, ranks):
            row[f'{base}__interrun_rank'] = rank


def slope_or_nan(vals: Sequence[float]) -> float:
    xs = np.asarray(vals, dtype=float)
    if xs.size < 2 or np.any(~np.isfinite(xs)):
        return math.nan
    t = np.linspace(0.0, 1.0, xs.size)
    t = t - t.mean()
    y = xs - xs.mean()
    denom = float(np.dot(t, t))
    if denom <= 0.0:
        return math.nan
    return float(np.dot(t, y) / denom)


def load_np_fallbacks(noise: float, image_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
    rows = read_csv(Path(NP_FALLBACK_CSV))
    candidates = []
    targets = set(image_ids)
    for row in rows:
        if row.get('alignment_mode') != 'resolve':
            continue
        if abs(to_float(row.get('measurement_noise_std')) - noise) > 1e-12:
            continue
        image_id = image_id_from_np_basename(str(row['image_basename']))
        if image_id not in targets:
            continue
        candidates.append({
            'image_id': image_id,
            'config_tag': row['config_tag'],
            'seed': int(row['seed']),
            'psnr': to_float(row['psnr']),
            'selector_post_winner_lf_mse_mean': to_float(row['selector_post_winner_lf_mse_mean']),
        })
    out: Dict[str, Dict[str, object]] = {}
    for image_id in image_ids:
        cur = [r for r in candidates if r['image_id'] == image_id]
        if not cur:
            raise ValueError(f'No NP fallback rows for {image_id}')
        best = min(cur, key=lambda r: (r['selector_post_winner_lf_mse_mean'], -r['psnr'], str(r['config_tag']), int(r['seed'])))
        out[image_id] = {
            'np_selected_psnr': best['psnr'],
            'np_selected_config_tag': best['config_tag'],
            'np_selected_seed': best['seed'],
            'np_selected_selector_post_lf_mse': best['selector_post_winner_lf_mse_mean'],
        }
    return out


def build_dev_rows(dataset_name: str, run_rows: List[Dict[str, str]], step_rows: List[Dict[str, str]], sample_dir: Path) -> List[Dict[str, object]]:
    add_interrun_rank_features(step_rows)
    sample_paths = parse_sample_paths(sample_dir)
    grouped_steps: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped_steps[(str(row['image_id']), int(row['run_index']))].append(row)
    for rows in grouped_steps.values():
        rows.sort(key=lambda r: int(r['step']))

    by_image: Dict[str, List[Tuple[int, Dict[str, str], Path]]] = defaultdict(list)
    for row in run_rows:
        key = (str(row['image_id']), int(row['run_index']))
        if key in sample_paths:
            by_image[key[0]].append((key[1], row, sample_paths[key]))

    out_rows: List[Dict[str, object]] = []
    for image_id, items in sorted(by_image.items()):
        items.sort(key=lambda t: t[0])
        if len(items) != 4:
            continue
        rgbs = [load_rgb(p) for _, _, p in items]
        lowfs = [lowfreq_repr(x) for x in rgbs]
        pixel_pair = np.zeros((4, 4), dtype=np.float32)
        lowf_pair = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(i + 1, 4):
                dp = l2_normed(rgbs[i], rgbs[j])
                dl = l2_normed(lowfs[i], lowfs[j])
                pixel_pair[i, j] = pixel_pair[j, i] = dp
                lowf_pair[i, j] = lowf_pair[j, i] = dl
        for i, (run_index, row, path) in enumerate(items):
            step_key = (image_id, run_index)
            step_list = grouped_steps[step_key]
            k = max(1, int(math.ceil(len(step_list) * WINDOW_FRAC)))
            wr = step_list[:k]
            vals = [to_float(r.get('x0y_full_residual_normed__interrun_rank')) for r in wr]
            out_rows.append({
                'dataset_name': dataset_name,
                'image_id': image_id,
                'run_index': run_index,
                'dataset_image_key': f'{dataset_name}:{image_id}',
                'final_psnr': to_float(row['final_psnr']),
                'bad25': int(str(row['final_bad_below25']).lower() in {'true', '1'}),
                'bad20': int(str(row['final_bad_below20']).lower() in {'true', '1'}),
                'residual_first80_slope': slope_or_nan(vals),
                'residual_first80_last': vals[-1] if vals else math.nan,
                'pixel_dist_to_nearest_neighbor': float(np.min(np.delete(pixel_pair[i], i))),
                'lowfreq_dist_to_nearest_neighbor': float(np.min(np.delete(lowf_pair[i], i))),
                'sample_path': str(path),
            })
    return out_rows


def confusion(y_true: Sequence[int], flags: Sequence[bool]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for y, f in zip(y_true, flags):
        if f and y:
            tp += 1
        elif f and not y:
            fp += 1
        elif (not f) and y:
            fn += 1
        else:
            tn += 1
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def precision(counts: Dict[str, int]) -> float:
    d = counts['tp'] + counts['fp']
    return counts['tp'] / d if d else math.nan


def recall(counts: Dict[str, int]) -> float:
    d = counts['tp'] + counts['fn']
    return counts['tp'] / d if d else math.nan


def balanced_accuracy(counts: Dict[str, int]) -> float:
    pos = counts['tp'] + counts['fn']
    neg = counts['tn'] + counts['fp']
    if pos == 0 or neg == 0:
        return math.nan
    return 0.5 * (counts['tp'] / pos + counts['tn'] / neg)


def simulate_policy(rows: List[Dict[str, object]], np_fallbacks: Dict[str, Dict[str, object]], flags: Sequence[bool], policy_name: str, policy_spec: Dict[str, object]) -> Dict[str, object]:
    run_records = []
    by_dataset_image: Dict[str, List[float]] = defaultdict(list)
    for row, flagged in zip(rows, flags):
        image_id = str(row['image_id'])
        sitcom_psnr = to_float(row['final_psnr'])
        policy_psnr = float(np_fallbacks[image_id]['np_selected_psnr']) if flagged else sitcom_psnr
        run_records.append({
            'dataset_name': row['dataset_name'],
            'dataset_image_key': row['dataset_image_key'],
            'image_id': image_id,
            'run_index': int(row['run_index']),
            'sitcom_psnr': sitcom_psnr,
            'policy_psnr': policy_psnr,
            'flagged': bool(flagged),
            'is_bad25': int(row['bad25']),
            'is_bad20': int(row['bad20']),
            'delta_vs_sitcom': policy_psnr - sitcom_psnr,
        })
        by_dataset_image[str(row['dataset_image_key'])].append(policy_psnr)
    y25 = [int(r['is_bad25']) for r in run_records]
    counts25 = confusion(y25, flags)
    policy_psnrs = [float(r['policy_psnr']) for r in run_records]
    image_best = [max(v) for v in by_dataset_image.values()]
    metrics = {
        'policy_name': policy_name,
        'policy_family': policy_spec['policy_family'],
        'run_level_mean_psnr': mean_or_nan(policy_psnrs),
        'run_level_min_psnr': min(policy_psnrs) if policy_psnrs else math.nan,
        'run_level_num_below25': sum(1 for x in policy_psnrs if x < 25.0),
        'run_level_num_below20': sum(1 for x in policy_psnrs if x < 20.0),
        'num_replaced': int(sum(flags)),
        'num_false_positive_replacements': counts25['fp'],
        'num_true_positive_replacements': counts25['tp'],
        'num_false_negative_remaining_bad25': counts25['fn'],
        'bad25_recall': recall(counts25),
        'bad25_precision': precision(counts25),
        'image_level_best_of_4_mean_psnr': mean_or_nan(image_best),
        'image_level_best_of_4_min_psnr': min(image_best) if image_best else math.nan,
        'balanced_accuracy_bad25': balanced_accuracy(counts25),
        'tp': counts25['tp'], 'fp': counts25['fp'], 'tn': counts25['tn'], 'fn': counts25['fn'],
    }
    return {'metrics': metrics, 'run_records': run_records}


def policy_sort_key(metrics: Dict[str, object]) -> Tuple[object, ...]:
    fp = int(metrics['num_false_positive_replacements'])
    repl = int(metrics['num_replaced'])
    tier = 1
    if fp <= 10 and repl <= 40:
        tier = 0
    prefer_strict35 = 0 if repl <= 35 else 1
    return (
        tier,
        int(metrics['run_level_num_below25']),
        int(metrics['run_level_num_below20']),
        max(0, fp - 10),
        max(0, repl - 40),
        prefer_strict35,
        -to_float(metrics['image_level_best_of_4_min_psnr']),
        -to_float(metrics['run_level_mean_psnr']),
        fp,
        repl,
    )


def top_unique_rows(rows: List[Dict[str, object]], n: int) -> List[Dict[str, object]]:
    rows = sorted(rows, key=lambda r: policy_sort_key(r['metrics']))
    return rows[:n]


def fit_single_feature(rows: List[Dict[str, object]], feature: str, policy_name: str) -> List[Dict[str, object]]:
    vals = sorted({to_float(r[feature]) for r in rows if math.isfinite(to_float(r[feature]))})
    out = []
    image_ids = sorted({str(r['image_id']) for r in rows})
    np_fallbacks = load_np_fallbacks(NOISE, image_ids)
    for thr in vals:
        flags = [to_float(r[feature]) >= thr for r in rows]
        spec = {'policy_family': 'consensus_single', 'feature_name': feature, 'direction': 'high_is_risky', 'threshold': float(thr)}
        sim = simulate_policy(rows, np_fallbacks, flags, policy_name, spec)
        out.append({'policy_name': policy_name, 'policy_spec': spec, 'metrics': sim['metrics']})
    return out


def fit_residual_policy(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    vals1 = sorted({to_float(r['residual_first80_slope']) for r in rows if math.isfinite(to_float(r['residual_first80_slope']))})
    vals2 = sorted({to_float(r['residual_first80_last']) for r in rows if math.isfinite(to_float(r['residual_first80_last']))})
    out = []
    image_ids = sorted({str(r['image_id']) for r in rows})
    np_fallbacks = load_np_fallbacks(NOISE, image_ids)
    for t1 in vals1:
        flags1 = [to_float(r['residual_first80_slope']) >= t1 for r in rows]
        for t2 in vals2:
            flags = [f1 and (to_float(r['residual_first80_last']) >= t2) for r, f1 in zip(rows, flags1)]
            spec = {
                'policy_family': 'residual_and',
                'feature_names': ['x0y_full_residual_normed__interrun_rank__first80pct__slope', 'x0y_full_residual_normed__interrun_rank__first80pct__last_in_window'],
                'feature_directions': {
                    'x0y_full_residual_normed__interrun_rank__first80pct__slope': 'high_is_risky',
                    'x0y_full_residual_normed__interrun_rank__first80pct__last_in_window': 'high_is_risky',
                },
                'thresholds': {
                    'x0y_full_residual_normed__interrun_rank__first80pct__slope': float(t1),
                    'x0y_full_residual_normed__interrun_rank__first80pct__last_in_window': float(t2),
                },
            }
            sim = simulate_policy(rows, np_fallbacks, flags, 'residual_first80_only', spec)
            out.append({'policy_name': 'residual_first80_only', 'policy_spec': spec, 'metrics': sim['metrics']})
    return out


def apply_spec_flags(rows: List[Dict[str, object]], spec: Dict[str, object]) -> List[bool]:
    fam = spec['policy_family']
    if fam == 'residual_and':
        t1 = float(spec['thresholds']['x0y_full_residual_normed__interrun_rank__first80pct__slope'])
        t2 = float(spec['thresholds']['x0y_full_residual_normed__interrun_rank__first80pct__last_in_window'])
        return [to_float(r['residual_first80_slope']) >= t1 and to_float(r['residual_first80_last']) >= t2 for r in rows]
    if fam == 'consensus_single':
        feat = spec['feature_name']
        thr = float(spec['threshold'])
        return [to_float(r[feat]) >= thr for r in rows]
    if fam == 'consensus_or':
        p = float(spec['thresholds']['pixel_dist_to_nearest_neighbor'])
        l = float(spec['thresholds']['lowfreq_dist_to_nearest_neighbor'])
        return [to_float(r['pixel_dist_to_nearest_neighbor']) >= p or to_float(r['lowfreq_dist_to_nearest_neighbor']) >= l for r in rows]
    if fam == 'residual_or_consensus':
        left = apply_spec_flags(rows, spec['residual_spec'])
        right = apply_spec_flags(rows, spec['consensus_spec'])
        return [a or b for a, b in zip(left, right)]
    raise ValueError(fam)


def fit_consensus_or(rows: List[Dict[str, object]], pixel_candidates: List[Dict[str, object]], low_candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    image_ids = sorted({str(r['image_id']) for r in rows})
    np_fallbacks = load_np_fallbacks(NOISE, image_ids)
    out = []
    for p in pixel_candidates:
        p_thr = float(p['policy_spec']['threshold'])
        p_flags = [to_float(r['pixel_dist_to_nearest_neighbor']) >= p_thr for r in rows]
        for l in low_candidates:
            l_thr = float(l['policy_spec']['threshold'])
            flags = [pf or (to_float(r['lowfreq_dist_to_nearest_neighbor']) >= l_thr) for r, pf in zip(rows, p_flags)]
            spec = {
                'policy_family': 'consensus_or',
                'feature_names': ['pixel__dist_to_nearest_neighbor', 'lowfreq__dist_to_nearest_neighbor'],
                'feature_directions': {'pixel__dist_to_nearest_neighbor': 'high_is_risky', 'lowfreq__dist_to_nearest_neighbor': 'high_is_risky'},
                'thresholds': {'pixel_dist_to_nearest_neighbor': p_thr, 'lowfreq_dist_to_nearest_neighbor': l_thr},
            }
            sim = simulate_policy(rows, np_fallbacks, flags, 'consensus_pixel_or_lowfreq', spec)
            out.append({'policy_name': 'consensus_pixel_or_lowfreq', 'policy_spec': spec, 'metrics': sim['metrics']})
    return out


def fit_residual_or_consensus(rows: List[Dict[str, object]], residual_candidates: List[Dict[str, object]], consensus_candidates: List[Dict[str, object]], name: str) -> List[Dict[str, object]]:
    image_ids = sorted({str(r['image_id']) for r in rows})
    np_fallbacks = load_np_fallbacks(NOISE, image_ids)
    out = []
    for r in residual_candidates:
        r_flags = apply_spec_flags(rows, r['policy_spec'])
        for c in consensus_candidates:
            c_flags = apply_spec_flags(rows, c['policy_spec'])
            flags = [a or b for a, b in zip(r_flags, c_flags)]
            spec = {
                'policy_family': 'residual_or_consensus',
                'residual_spec': r['policy_spec'],
                'consensus_spec': c['policy_spec'],
            }
            sim = simulate_policy(rows, np_fallbacks, flags, name, spec)
            out.append({'policy_name': name, 'policy_spec': spec, 'metrics': sim['metrics']})
    return out


def candidate_to_row(candidate: Dict[str, object], selected: bool) -> Dict[str, object]:
    m = candidate['metrics']
    return {
        'policy_name': candidate['policy_name'],
        'selected_final_candidate': selected,
        'policy_family': candidate['policy_spec']['policy_family'],
        'policy_spec_json': json.dumps(candidate['policy_spec'], sort_keys=True),
        **m,
    }


def build_summary(best: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    m = best['metrics']
    lines = [
        '# A14 Frozen Policy Candidate',
        '',
        'This is a development-only frozen candidate fit on A8+A11. No A14 trajectory results have been used.',
        '',
        '## Selected candidate',
        '',
        f"- policy: `{best['policy_name']}`",
        f"- family: `{best['policy_spec']['policy_family']}`",
        f"- remaining bad25: `{int(m['run_level_num_below25'])}`",
        f"- remaining bad20: `{int(m['run_level_num_below20'])}`",
        f"- false-positive replacements: `{int(m['num_false_positive_replacements'])}`",
        f"- total replacements: `{int(m['num_replaced'])}`",
        f"- image best-of-4 min PSNR: `{to_float(m['image_level_best_of_4_min_psnr']):.3f}`",
        f"- run mean PSNR: `{to_float(m['run_level_mean_psnr']):.3f}`",
        '',
        '## Selection rule',
        '',
        '- tier 0: FP <= 10 and total replacements <= 40',
        '- tier 1: anything larger',
        '- within tier: minimize remaining bad25, then remaining bad20, then prefer <=35 replacements when possible, then preserve image best-of-4 min, then maximize run mean PSNR',
        '',
        '## Candidate families considered',
        '',
    ]
    for row in rows:
        lines.append(f"- `{row['policy_name']}`: bad25 `{int(row['run_level_num_below25'])}`, bad20 `{int(row['run_level_num_below20'])}`, FP `{int(row['num_false_positive_replacements'])}`, repl `{int(row['num_replaced'])}`")
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--a8_dir', required=True)
    ap.add_argument('--a11_dir', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    a8_dir = Path(args.a8_dir)
    a11_dir = Path(args.a11_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a8_rows = build_dev_rows('A8', read_csv(a8_dir / 'run_level_summary.csv'), read_csv(a8_dir / 'trajectory_step_metrics.csv'), a8_dir / 'samples')
    a11_rows = build_dev_rows('A11', read_csv(a11_dir / 'run_level_summary.csv'), read_csv(a11_dir / 'trajectory_step_metrics.csv'), a11_dir / 'samples')
    rows = a8_rows + a11_rows

    residual_all = fit_residual_policy(rows)
    residual_top = top_unique_rows(residual_all, 15)
    pixel_all = fit_single_feature(rows, 'pixel_dist_to_nearest_neighbor', 'consensus_pixel_nn')
    pixel_top = top_unique_rows(pixel_all, 15)
    low_all = fit_single_feature(rows, 'lowfreq_dist_to_nearest_neighbor', 'consensus_lowfreq_nn')
    low_top = top_unique_rows(low_all, 15)
    pixlow_all = fit_consensus_or(rows, pixel_top[:10], low_top[:10])
    pixlow_top = top_unique_rows(pixlow_all, 10)
    res_or_pixel_all = fit_residual_or_consensus(rows, residual_top[:10], pixel_top[:10], 'residual_or_pixel_nn')
    res_or_low_all = fit_residual_or_consensus(rows, residual_top[:10], low_top[:10], 'residual_or_lowfreq_nn')
    res_or_pixlow_all = fit_residual_or_consensus(rows, residual_top[:10], pixlow_top[:10], 'residual_or_pixel_or_lowfreq_nn')

    finalists = [residual_top[0], pixel_top[0], low_top[0], pixlow_top[0], top_unique_rows(res_or_pixel_all,1)[0], top_unique_rows(res_or_low_all,1)[0], top_unique_rows(res_or_pixlow_all,1)[0]]
    finalists = sorted(finalists, key=lambda c: policy_sort_key(c['metrics']))
    best = finalists[0]

    summary_rows = [candidate_to_row(c, c is best) for c in finalists]
    write_csv(outdir / 'policy_fit_summary.csv', summary_rows)

    frozen = {
        'policy_name': best['policy_name'],
        'policy_family': best['policy_spec']['policy_family'],
        'policy_spec': best['policy_spec'],
        'fit_dataset': 'A8+A11 development only',
        'fit_source_dirs': {
            'A8': str(a8_dir),
            'A11': str(a11_dir),
        },
        'fallback_source_csv': NP_FALLBACK_CSV,
        'measurement_noise_std': NOISE,
        'selection_criterion': {
            'primary_goal': 'reduce bad25 and bad20 on A8+A11 development pool',
            'preference_budgets': {
                'false_positive_replacements_max_preferred': 10,
                'total_replacements_max_strict_preferred': 35,
                'total_replacements_max_relaxed_preferred': 40,
            },
            'sort_order': [
                'budget tier: fp<=10 and repl<=40 preferred',
                'min remaining bad25',
                'min remaining bad20',
                'prefer <=35 replacements when possible',
                'min budget excess',
                'max image-level best-of-4 min',
                'max run-level mean PSNR',
            ],
        },
        'combined_metrics': best['metrics'],
        'warning': 'Freeze this policy before any new A14 trajectories are evaluated. Do not use A14 results to change thresholds or features.',
    }
    write_text(outdir / 'frozen_policy.json', json.dumps(frozen, indent=2) + '\n')
    write_text(outdir / 'SUMMARY.md', build_summary(best, summary_rows))


if __name__ == '__main__':
    main()
