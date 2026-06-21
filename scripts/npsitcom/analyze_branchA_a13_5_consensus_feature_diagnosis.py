#!/usr/bin/env python3
"""A13.5 clean-free consensus feature diagnosis."""

from __future__ import annotations

import argparse
import csv
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


def auc_roc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, y_true) if math.isfinite(float(s))]
    if not pairs:
        return math.nan
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return math.nan
    pairs.sort(key=lambda t: t[0])
    rank = 1
    pos_rank_sum = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i) - 1))
        block_pos = sum(pairs[k][1] for k in range(i, j))
        pos_rank_sum += avg_rank * block_pos
        rank += (j - i)
        i = j
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def auc_pr(y_true: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, y_true) if math.isfinite(float(s))]
    if not pairs:
        return math.nan
    pos = sum(y for _, y in pairs)
    if pos == 0:
        return math.nan
    pairs.sort(key=lambda t: t[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


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
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def bal_acc(counts: Dict[str, int]) -> float:
    pos = counts["tp"] + counts["fn"]
    neg = counts["tn"] + counts["fp"]
    if pos == 0 or neg == 0:
        return math.nan
    return 0.5 * (counts["tp"] / pos + counts["tn"] / neg)


def precision(counts: Dict[str, int]) -> float:
    denom = counts["tp"] + counts["fp"]
    return counts["tp"] / denom if denom else math.nan


def recall(counts: Dict[str, int]) -> float:
    denom = counts["tp"] + counts["fn"]
    return counts["tp"] / denom if denom else math.nan


def rank_within_group(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    ranks[order] = np.arange(1, arr.size + 1, dtype=float)
    return [float(x) for x in ranks]


def parse_sample_paths(sample_dir: Path) -> Dict[Tuple[str, int], Path]:
    out: Dict[Tuple[str, int], Path] = {}
    for path in sorted(sample_dir.glob('*.png')):
        m = SAMPLE_RE.search(path.name)
        if not m:
            continue
        image_id = m.group(1)
        run_index = int(m.group(2))
        out[(image_id, run_index)] = path
    return out


def load_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return arr


def lowfreq_repr(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(fft).astype(np.float32)
    h, w = mag.shape
    hh = max(1, int(round(h * LOWFREQ_FRAC / 2.0)))
    ww = max(1, int(round(w * LOWFREQ_FRAC / 2.0)))
    cy, cx = h // 2, w // 2
    return mag[cy - hh: cy + hh, cx - ww: cx + ww]


def fftmag_repr(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    return np.abs(np.fft.fft2(gray)).astype(np.float32)


def l2_normed(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(diff.ravel()) / math.sqrt(diff.size))


def add_interrun_rank_features(step_rows: List[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row['image_id']), int(row['step']))].append(row)
    bases = ('x0y_full_residual_normed',)
    for rows in grouped.values():
        for base in bases:
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


def build_residual_first80_features(step_rows: List[Dict[str, str]], run_rows: List[Dict[str, str]]) -> Dict[Tuple[str, int], Dict[str, float]]:
    add_interrun_rank_features(step_rows)
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row['image_id']), int(row['run_index']))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r['step']))
    run_meta = {(str(r['image_id']), int(r['run_index'])): r for r in run_rows}
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for key, rows in grouped.items():
        k = max(1, int(math.ceil(len(rows) * WINDOW_FRAC)))
        wr = rows[:k]
        vals = [to_float(r.get('x0y_full_residual_normed__interrun_rank')) for r in wr]
        last = vals[-1] if vals and math.isfinite(vals[-1]) else math.nan
        out[key] = {
            'image_id': key[0],
            'run_index': key[1],
            'final_psnr': to_float(run_meta[key]['final_psnr']),
            'bad25': int(to_float(run_meta[key]['final_psnr']) < 25.0),
            'bad20': int(to_float(run_meta[key]['final_psnr']) < 20.0),
            'residual_first80_slope': slope_or_nan(vals),
            'residual_first80_last': last,
        }
    return out


def build_consensus_rows(dataset_name: str, run_rows: List[Dict[str, str]], sample_dir: Path) -> List[Dict[str, object]]:
    sample_paths = parse_sample_paths(sample_dir)
    by_image: Dict[str, List[Tuple[int, Dict[str, str], Path]]] = defaultdict(list)
    for row in run_rows:
        key = (str(row['image_id']), int(row['run_index']))
        if key not in sample_paths:
            continue
        by_image[key[0]].append((key[1], row, sample_paths[key]))

    out_rows: List[Dict[str, object]] = []
    for image_id, items in sorted(by_image.items()):
        items.sort(key=lambda t: t[0])
        if len(items) != 4:
            continue
        rgbs = [load_rgb(p) for _, _, p in items]
        lowfs = [lowfreq_repr(x) for x in rgbs]
        ffts = [fftmag_repr(x) for x in rgbs]
        means = {
            'pixel': np.mean(np.stack(rgbs, axis=0), axis=0),
            'lowfreq': np.mean(np.stack(lowfs, axis=0), axis=0),
            'fftmag': np.mean(np.stack(ffts, axis=0), axis=0),
        }
        medians = {
            'pixel': np.median(np.stack(rgbs, axis=0), axis=0),
            'lowfreq': np.median(np.stack(lowfs, axis=0), axis=0),
            'fftmag': np.median(np.stack(ffts, axis=0), axis=0),
        }
        pairwise = {
            'pixel': np.zeros((4, 4), dtype=np.float32),
            'lowfreq': np.zeros((4, 4), dtype=np.float32),
            'fftmag': np.zeros((4, 4), dtype=np.float32),
        }
        reps = {'pixel': rgbs, 'lowfreq': lowfs, 'fftmag': ffts}
        for metric, arrays in reps.items():
            for i in range(4):
                for j in range(i + 1, 4):
                    d = l2_normed(arrays[i], arrays[j])
                    pairwise[metric][i, j] = d
                    pairwise[metric][j, i] = d
        feature_rows: List[Dict[str, object]] = []
        for i, (run_index, row, path) in enumerate(items):
            feat: Dict[str, object] = {
                'dataset_name': dataset_name,
                'image_id': image_id,
                'run_index': run_index,
                'final_psnr': to_float(row['final_psnr']),
                'bad25': int(str(row['final_bad_below25']).lower() in {'true', '1'}),
                'bad20': int(str(row['final_bad_below20']).lower() in {'true', '1'}),
                'sample_path': str(path),
            }
            for metric, arrays in reps.items():
                feat[f'{metric}__dist_to_mean'] = l2_normed(arrays[i], means[metric])
                feat[f'{metric}__dist_to_median_image'] = l2_normed(arrays[i], medians[metric])
                row_d = pairwise[metric][i]
                feat[f'{metric}__dist_to_nearest_neighbor'] = float(np.min(np.delete(row_d, i)))
                feat[f'{metric}__dist_to_mean_pairwise'] = float(np.sum(row_d) / 3.0)
                medoid_idx = int(np.argmin(np.sum(pairwise[metric], axis=1)))
                feat[f'{metric}__dist_to_medoid_run'] = float(pairwise[metric][i, medoid_idx])
            feature_rows.append(feat)
        for metric in ('pixel', 'lowfreq', 'fftmag'):
            for suffix in ('dist_to_mean', 'dist_to_median_image', 'dist_to_nearest_neighbor', 'dist_to_mean_pairwise', 'dist_to_medoid_run'):
                values = [to_float(r[f'{metric}__{suffix}']) for r in feature_rows]
                ranks = rank_within_group(values)
                for r, rank in zip(feature_rows, ranks):
                    r[f'{metric}__{suffix}__rank'] = rank
        out_rows.extend(feature_rows)
    return out_rows


def select_threshold(scores: Sequence[float], y_true: Sequence[int]) -> Dict[str, object]:
    finite = sorted({float(s) for s in scores if math.isfinite(float(s))})
    best = None
    for direction in ('high_is_risky', 'low_is_risky'):
        for thr in finite:
            if direction == 'high_is_risky':
                flags = [float(s) >= thr if math.isfinite(float(s)) else False for s in scores]
            else:
                flags = [float(s) <= thr if math.isfinite(float(s)) else False for s in scores]
            counts = confusion(y_true, flags)
            row = {
                'direction': direction,
                'threshold': float(thr),
                'tp': counts['tp'],
                'fp': counts['fp'],
                'tn': counts['tn'],
                'fn': counts['fn'],
                'balanced_accuracy': bal_acc(counts),
                'precision': precision(counts),
                'recall': recall(counts),
                'num_flagged': int(sum(flags)),
            }
            key = (
                -(row['balanced_accuracy'] if math.isfinite(row['balanced_accuracy']) else -1e9),
                row['fp'],
                row['num_flagged'],
            )
            if best is None or key < best[0]:
                best = (key, row)
    if best is None:
        return {'direction': 'high_is_risky', 'threshold': math.nan, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'balanced_accuracy': math.nan, 'precision': math.nan, 'recall': math.nan, 'num_flagged': 0}
    return best[1]


def apply_threshold(scores: Sequence[float], direction: str, threshold: float) -> List[bool]:
    out = []
    for s in scores:
        v = float(s)
        if not math.isfinite(v):
            out.append(False)
        elif direction == 'high_is_risky':
            out.append(v >= threshold)
        else:
            out.append(v <= threshold)
    return out


def fit_residual_and(rows: List[Dict[str, object]], label_key: str) -> Dict[str, object]:
    vals1 = sorted({to_float(r['residual_first80_slope']) for r in rows if math.isfinite(to_float(r['residual_first80_slope']))})
    vals2 = sorted({to_float(r['residual_first80_last']) for r in rows if math.isfinite(to_float(r['residual_first80_last']))})
    y = [int(r[label_key]) for r in rows]
    best = None
    for t1 in vals1:
        for t2 in vals2:
            flags = [to_float(r['residual_first80_slope']) >= t1 and to_float(r['residual_first80_last']) >= t2 for r in rows]
            c = confusion(y, flags)
            row = {
                'threshold1': float(t1),
                'threshold2': float(t2),
                'tp': c['tp'], 'fp': c['fp'], 'tn': c['tn'], 'fn': c['fn'],
                'balanced_accuracy': bal_acc(c), 'precision': precision(c), 'recall': recall(c), 'num_flagged': int(sum(flags)),
            }
            key = (
                -(row['balanced_accuracy'] if math.isfinite(row['balanced_accuracy']) else -1e9),
                row['fp'],
                row['num_flagged'],
            )
            if best is None or key < best[0]:
                best = (key, row)
    assert best is not None
    return best[1]


def summarize_feature(
    train_name: str,
    test_name: str,
    feature: str,
    train_rows: List[Dict[str, object]],
    test_rows: List[Dict[str, object]],
    label_key: str,
    a12_fn_keys: Sequence[Tuple[str, int]],
) -> Dict[str, object]:
    train_scores = [to_float(r.get(feature)) for r in train_rows]
    test_scores = [to_float(r.get(feature)) for r in test_rows]
    y_train = [int(r[label_key]) for r in train_rows]
    y_test = [int(r[label_key]) for r in test_rows]
    thr = select_threshold(train_scores, y_train)
    train_flags = apply_threshold(train_scores, str(thr['direction']), float(thr['threshold']))
    test_flags = apply_threshold(test_scores, str(thr['direction']), float(thr['threshold']))
    c_train = confusion(y_train, train_flags)
    c_test = confusion(y_test, test_flags)
    test_keyed = {(str(r['image_id']), int(r['run_index'])): f for r, f in zip(test_rows, test_flags)}
    a12_hits = sum(1 for key in a12_fn_keys if test_keyed.get(key, False)) if test_name == 'A11' else 0
    return {
        'train_dataset': train_name,
        'test_dataset': test_name,
        'label_key': label_key,
        'feature_name': feature,
        'train_auroc': auc_roc(y_train, train_scores),
        'train_auprc': auc_pr(y_train, train_scores),
        'test_auroc': auc_roc(y_test, test_scores),
        'test_auprc': auc_pr(y_test, test_scores),
        'direction': thr['direction'],
        'threshold': thr['threshold'],
        'train_balanced_accuracy': bal_acc(c_train),
        'test_balanced_accuracy': bal_acc(c_test),
        'train_recall': recall(c_train),
        'test_recall': recall(c_test),
        'train_precision': precision(c_train),
        'test_precision': precision(c_test),
        'train_tp': c_train['tp'], 'train_fp': c_train['fp'], 'train_tn': c_train['tn'], 'train_fn': c_train['fn'],
        'test_tp': c_test['tp'], 'test_fp': c_test['fp'], 'test_tn': c_test['tn'], 'test_fn': c_test['fn'],
        'train_num_flagged': int(sum(train_flags)), 'test_num_flagged': int(sum(test_flags)),
        'test_a12_fn_hits': a12_hits,
        'test_a12_fn_total': len(a12_fn_keys) if test_name == 'A11' else 0,
    }


def pick_best_consensus_feature(summary_rows: List[Dict[str, object]], label_key: str, train_name: str, test_name: str) -> Dict[str, object]:
    rows = [r for r in summary_rows if r['label_key'] == label_key and r['train_dataset'] == train_name and r['test_dataset'] == test_name]
    rows.sort(key=lambda r: (
        -(to_float(r['train_auroc']) if math.isfinite(to_float(r['train_auroc'])) else -1e9),
        -(to_float(r['train_auprc']) if math.isfinite(to_float(r['train_auprc'])) else -1e9),
        -to_float(r['train_balanced_accuracy']) if math.isfinite(to_float(r['train_balanced_accuracy'])) else 1e9,
        int(r['train_fp']),
        int(r['train_num_flagged']),
        str(r['feature_name']),
    ))
    return rows[0]


def build_failure_cases(
    split_name: str,
    rows: List[Dict[str, object]],
    label_key: str,
    consensus_feature: str,
    consensus_direction: str,
    consensus_threshold: float,
    residual_thresholds: Dict[str, float],
    a12_fn_keys: Sequence[Tuple[str, int]],
) -> List[Dict[str, object]]:
    out = []
    a12_set = set(a12_fn_keys)
    for r in rows:
        key = (str(r['image_id']), int(r['run_index']))
        cons_val = to_float(r.get(consensus_feature))
        cons_flag = apply_threshold([cons_val], consensus_direction, consensus_threshold)[0]
        res_flag = to_float(r['residual_first80_slope']) >= residual_thresholds['threshold1'] and to_float(r['residual_first80_last']) >= residual_thresholds['threshold2']
        if int(r[label_key]) == 1 or cons_flag or res_flag or key in a12_set:
            out.append({
                'split_name': split_name,
                'image_id': key[0],
                'run_index': key[1],
                'label_key': label_key,
                'is_bad': int(r[label_key]),
                'final_psnr': to_float(r['final_psnr']),
                'is_a12_false_negative': key in a12_set,
                'consensus_feature_name': consensus_feature,
                'consensus_feature_value': cons_val,
                'consensus_flag': cons_flag,
                'residual_first80_slope': to_float(r['residual_first80_slope']),
                'residual_first80_last': to_float(r['residual_first80_last']),
                'residual_flag': res_flag,
                'or_flag': bool(cons_flag or res_flag),
                'and_flag': bool(cons_flag and res_flag),
            })
    out.sort(key=lambda x: (x['split_name'], -int(x['is_a12_false_negative']), -int(x['is_bad']), x['image_id'], x['run_index']))
    return out


def build_summary(best_cross: List[Dict[str, object]], combo_rows: List[Dict[str, object]], note_np_image_unavailable: bool, note_raw_tensors_unavailable: bool) -> str:
    lines = [
        '# A13.5 Consensus Feature Diagnosis',
        '',
        'This is offline analysis only. No A13.5 result is prospective evidence.',
        '',
        '## Main takeaways',
        '',
    ]
    for row in best_cross:
        lines.append(
            f"- {row['train_dataset']} -> {row['test_dataset']} best consensus feature for `{row['label_key']}`: `{row['feature_name']}` with test AUROC `{to_float(row['test_auroc']):.3f}`, test AUPRC `{to_float(row['test_auprc']):.3f}`, test FP `{int(row['test_fp'])}`, and A12-FN hits `{int(row['test_a12_fn_hits'])}` / `{int(row['test_a12_fn_total'])}`."
        )
    lines.extend(['', '## Combined residual + consensus', ''])
    for row in combo_rows:
        if row['label_key'] != 'bad25':
            continue
        lines.append(
            f"- {row['train_dataset']} -> {row['test_dataset']} `{row['policy_name']}`: test recall `{to_float(row['test_recall']):.3f}`, test precision `{to_float(row['test_precision']):.3f}`, test FP `{int(row['test_fp'])}`, test FN `{int(row['test_fn'])}`."
        )
    lines.extend(['', '## Availability notes', ''])
    lines.append(f"- Raw trajectory tensors for per-step consensus were {'not available' if note_raw_tensors_unavailable else 'available'} in this pass.")
    lines.append(f"- NP-selected fallback image distance features were {'not available' if note_np_image_unavailable else 'available'} from the existing CSV-only artifacts.")
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--a8_dir', required=True)
    ap.add_argument('--a11_dir', required=True)
    ap.add_argument('--a12_dir', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    a8_dir = Path(args.a8_dir)
    a11_dir = Path(args.a11_dir)
    a12_dir = Path(args.a12_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a8_run = read_csv(a8_dir / 'run_level_summary.csv')
    a11_run = read_csv(a11_dir / 'run_level_summary.csv')
    a8_step = read_csv(a8_dir / 'trajectory_step_metrics.csv')
    a11_step = read_csv(a11_dir / 'trajectory_step_metrics.csv')
    a12_missed = read_csv(a12_dir / 'missed_bad_runs_diagnosis.csv')
    a12_fn_keys = [(str(r['image_id']), int(r['run_index'])) for r in a12_missed]

    a8_res = build_residual_first80_features(a8_step, a8_run)
    a11_res = build_residual_first80_features(a11_step, a11_run)
    a8_cons = build_consensus_rows('A8', a8_run, a8_dir / 'samples')
    a11_cons = build_consensus_rows('A11', a11_run, a11_dir / 'samples')

    def merge(cons_rows, res_map):
        out = []
        for r in cons_rows:
            key = (str(r['image_id']), int(r['run_index']))
            rr = res_map[key]
            merged = dict(r)
            merged.update({
                'residual_first80_slope': rr['residual_first80_slope'],
                'residual_first80_last': rr['residual_first80_last'],
            })
            out.append(merged)
        return out

    a8_rows = merge(a8_cons, a8_res)
    a11_rows = merge(a11_cons, a11_res)

    feature_names = [k for k in a8_rows[0].keys() if '__dist_' in k or k.endswith('__rank')]
    summary_rows: List[Dict[str, object]] = []
    for label_key in ('bad25', 'bad20'):
        for train_name, test_name, train_rows, test_rows in (
            ('A8', 'A11', a8_rows, a11_rows),
            ('A11', 'A8', a11_rows, a8_rows),
        ):
            for feat in feature_names:
                summary_rows.append(summarize_feature(train_name, test_name, feat, train_rows, test_rows, label_key, a12_fn_keys))
    write_csv(outdir / 'consensus_feature_summary.csv', summary_rows)

    combo_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    best_cross: List[Dict[str, object]] = []
    for label_key in ('bad25', 'bad20'):
        for train_name, test_name, train_rows, test_rows in (
            ('A8', 'A11', a8_rows, a11_rows),
            ('A11', 'A8', a11_rows, a8_rows),
        ):
            best = pick_best_consensus_feature(summary_rows, label_key, train_name, test_name)
            best_cross.append(best)
            residual_thr = fit_residual_and(train_rows, label_key)
            cons_train_scores = [to_float(r.get(best['feature_name'])) for r in train_rows]
            cons_test_scores = [to_float(r.get(best['feature_name'])) for r in test_rows]
            cons_train_flags = apply_threshold(cons_train_scores, str(best['direction']), float(best['threshold']))
            cons_test_flags = apply_threshold(cons_test_scores, str(best['direction']), float(best['threshold']))
            res_train_flags = [to_float(r['residual_first80_slope']) >= residual_thr['threshold1'] and to_float(r['residual_first80_last']) >= residual_thr['threshold2'] for r in train_rows]
            res_test_flags = [to_float(r['residual_first80_slope']) >= residual_thr['threshold1'] and to_float(r['residual_first80_last']) >= residual_thr['threshold2'] for r in test_rows]
            policies = {
                'residual_first80_only': (res_train_flags, res_test_flags),
                'consensus_only': (cons_train_flags, cons_test_flags),
                'residual_or_consensus': ([a or b for a, b in zip(res_train_flags, cons_train_flags)], [a or b for a, b in zip(res_test_flags, cons_test_flags)]),
                'residual_and_consensus': ([a and b for a, b in zip(res_train_flags, cons_train_flags)], [a and b for a, b in zip(res_test_flags, cons_test_flags)]),
            }
            y_train = [int(r[label_key]) for r in train_rows]
            y_test = [int(r[label_key]) for r in test_rows]
            test_keyed = {(str(r['image_id']), int(r['run_index'])): idx for idx, r in enumerate(test_rows)}
            for policy_name, (train_flags, test_flags) in policies.items():
                c_train = confusion(y_train, train_flags)
                c_test = confusion(y_test, test_flags)
                a12_hits = 0
                if test_name == 'A11':
                    for key in a12_fn_keys:
                        idx = test_keyed.get(key)
                        if idx is not None and test_flags[idx]:
                            a12_hits += 1
                combo_rows.append({
                    'train_dataset': train_name,
                    'test_dataset': test_name,
                    'label_key': label_key,
                    'policy_name': policy_name,
                    'consensus_feature_name': best['feature_name'],
                    'consensus_direction': best['direction'],
                    'consensus_threshold': best['threshold'],
                    'residual_threshold1': residual_thr['threshold1'],
                    'residual_threshold2': residual_thr['threshold2'],
                    'train_balanced_accuracy': bal_acc(c_train),
                    'test_balanced_accuracy': bal_acc(c_test),
                    'train_recall': recall(c_train),
                    'test_recall': recall(c_test),
                    'train_precision': precision(c_train),
                    'test_precision': precision(c_test),
                    'train_tp': c_train['tp'], 'train_fp': c_train['fp'], 'train_tn': c_train['tn'], 'train_fn': c_train['fn'],
                    'test_tp': c_test['tp'], 'test_fp': c_test['fp'], 'test_tn': c_test['tn'], 'test_fn': c_test['fn'],
                    'test_a12_fn_hits': a12_hits,
                    'test_a12_fn_total': len(a12_fn_keys) if test_name == 'A11' else 0,
                })
            failure_rows.extend(build_failure_cases(f'{train_name}_to_{test_name}', test_rows, label_key, best['feature_name'], str(best['direction']), float(best['threshold']), residual_thr, a12_fn_keys if test_name == 'A11' else []))

    write_csv(outdir / 'combined_residual_consensus_policy_summary.csv', combo_rows)
    write_csv(outdir / 'consensus_failure_cases.csv', failure_rows)
    write_text(outdir / 'SUMMARY.md', build_summary(best_cross, combo_rows, True, True))


if __name__ == '__main__':
    main()
