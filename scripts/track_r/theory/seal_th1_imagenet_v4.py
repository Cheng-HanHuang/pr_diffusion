#!/usr/bin/env python3
"""Seal a contract-passing ImageNet v4 A12 D4 forecast before unblinding.

No XP return or held D4 result is touched. Original 0--15 inputs only.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,platform
from pathlib import Path
import numpy,scipy
from check_th1_imagenet_v4_contract import check
from fit_th1_imagenet_v4 import load_pilot,load_clip,sha

def seal(args):
    check(args.draft)
    rows,raw=load_pilot(args)
    flags,succ,tot,cols=load_clip(args,rows,raw)
    if len(rows)!=100 or sum(tot)!=1600:raise ValueError('wrong pilot rows/decisions')
    d=json.loads(Path(args.draft).read_text())
    if (d['source_pilot16_sha256']!=sha(args.pilot) or
        d['source_clip16_sha256']!=sha(args.clip)):
        raise ValueError('pilot/CLIP file differs from validated v4 draft')
    if len(args.code_commit)!=40 or any(c not in '0123456789abcdef'
        for c in args.code_commit):raise ValueError('require full Git SHA')
    d['version']='TH1_PREDICTIONS_v4'
    d['status']='FROZEN_BEFORE_XP_D4_UNBLINDING'
    d['freeze_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    d['code_commit']=args.code_commit
    d['draft_sha256']=sha(args.draft)
    d['production_run']={
      'point_reps':d['point_reps'],'bootstrap':d['bootstrap'],
      'boot_reps':d['boot_reps'],
      'python':platform.python_version(),
      'numpy':numpy.__version__,'scipy':scipy.__version__,
      'contract':'check_th1_imagenet_v4_contract.py PASS 120/24/8'}
    d['registration_instructions']='Planner must PUSH and register this artifact SHA+UTC before D4 results are accessed; v2/v2 E1 remain immutable.'
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('refuse to overwrite existing v4')
    p.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
    print('TH1_V4_FROZEN',p)
    print('SHA256',sha(p))
    print('UTC',d['freeze_utc'])
    print('CODE_COMMIT',d['code_commit'])

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for v in ('prefix4','pilot','pilot_sha256','clip','clip_sha256',
              'draft','output','code_commit'):
        ap.add_argument('--'+v.replace('_','-'),required=True)
    for v in ('image_col','run_col','label_col','good_col'):
        ap.add_argument('--'+v.replace('_','-'))
    seal(ap.parse_args())
