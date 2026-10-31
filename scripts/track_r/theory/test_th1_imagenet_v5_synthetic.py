#!/usr/bin/env python3
"""Run entirely synthetic v5 source/contract test; NEVER reads XP-N data.

Usage from scripts/track_r/theory: python test_th1_imagenet_v5_synthetic.py
Requires normal NumPy/SciPy and repository's frozen A8 fit modules.
"""
from __future__ import annotations
import csv, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from fit_th1_imagenet_v5 import load_prefix,load_clip,run
from check_th1_imagenet_v5_contract import check

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def gen(root):
    rng=np.random.default_rng(20261101)
    prefixes=[];flags=[]
    for i in range(100):
        uid=f'SYNTH_XPN_ALTIMG_{i:03d}' # intentionally not D1 or actual XP-N IDs
        labels=''.join(rng.choice(list('GOPCI'),size=4,p=[.46,.11,.03,.21,.19]))
        rho=np.asarray([{'G':.09,'O':.13,'P':.14,'C':.25,'I':.60}[v] for v in labels])
        rho=rho+rng.random(4)*.009
        order=np.argsort(rho,kind='stable')
        rec={'image_id':uid,'family':'FAST','source':'SYNTHETIC_ONLY','n_runs':4,
             'run_labels':labels,'e0_rank_labels':''.join(labels[j]for j in order),
             'e0_rank_order':','.join(str(j)for j in order),
             'rho_signed_by_run':';'.join(f'{x:.12f}'for x in rho)}
        for label,col in zip('GOPCI',('n_GOOD','n_ORBIT_global','n_ORBIT_partial','n_CONSISTENT','n_INCONSISTENT')):
            rec[col]=labels.count(label)
        prefixes.append(rec)
        for j,cl in enumerate(labels):
            good=int(rng.random()<{'G':.91,'O':.77,'P':.02,'C':0,'I':0}[cl])
            flags.append({'image_id':uid,'run_index':j,'run_label':cl,'g1_chosen_good':good})
    p=root/'synthetic_prefix.csv';c=root/'synthetic_g1.csv'
    for file,rows in ((p,prefixes),(c,flags)):
        with open(file,'w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return p,c

def main():
    with tempfile.TemporaryDirectory(prefix='xpn_v5_contract_') as t:
        d=Path(t);p,c=gen(d)
        rows,raw=load_prefix(p,sha(p))
        from argparse import Namespace
        a=Namespace(image_col=None,run_col=None,label_col=None,good_col=None)
        flags,succ,tot,cols=load_clip(c,sha(c),rows,raw,a)
        assert len(rows)==100 and sum(tot)==400 and len(flags)==100
        # Confirm failure on a held-index G1 row before any simulation.
        broken=list(csv.DictReader(c.open()))
        broken[0]['run_index']='4'
        bad=d/'illegal_future_index.csv'
        with bad.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=broken[0]);w.writeheader();w.writerows(broken)
        try:load_clip(bad,sha(bad),rows,raw,a)
        except ValueError:pass
        else:raise AssertionError('held-run index was not rejected')
        out,anatomy,anchors,alpha,w,w1=run(rows,flags,succ,tot,64,4,1)
        obj={'version':'DRAFT_XP_N_V5_UNREGISTERED','status':'UNREGISTERED',
          'n_images':100,'n_observed_runs':4,'n_max':64,'K_grid':[4,8,16,32,64],
          'family':'FAST','dataset':'ImageNet','metric':'good20',
          'interval_type':'predictive_realized_panel','structural_zero_component':False,
          'source_restrictions':{'used_prefix_indices':[0,1,2,3],
             'held_runs_indices':list(range(4,64)),'held_runs_read':False,'xp_returns_accessed':False},
          'input_sha256':{'prefix4':sha(p),'clip4':sha(c)},
          'observed_image_ids_sha256':hashlib.sha256(('\n'.join(sorted(r['image_id']for r in rows))+'\n').encode()).hexdigest(),
          'g1_class_counts':{cl:{'total':int(tot[j]),'successes':int(succ[j])}for j,cl in enumerate('GOPCI')},
          'predictions':out,'anatomy_curves':anatomy,'oracle_anchor_checks':anchors}
        draft=d/'draft.json';draft.write_text(json.dumps(obj))
        check(draft)
        print('XP_N_V5_SYNTHETIC_PASS',len(out),len(anatomy),len(anchors),
              'using ONLY fabricated 100-image four-run input')
if __name__=='__main__':main()
