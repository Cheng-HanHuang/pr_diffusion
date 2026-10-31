#!/usr/bin/env python3
"""Strict held-out XP-N v5 draft contract. JSON only; no XP-N outcome reads."""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
VARS={'race','S2','tail','race_tau1'}
GRID={4,8,16,32,64}
POLICIES={'E0','E1_CLIP','O_raw','O_orbit'}

def check(path):
    with open(path,encoding='utf8') as f:o=json.load(f)
    assert o['version']=='DRAFT_XP_N_V5_UNREGISTERED' and o['status']=='UNREGISTERED'
    assert (o['n_images'],o['n_observed_runs'],o['n_max'])==(100,4,64)
    assert o['K_grid']==[4,8,16,32,64]
    assert o['family']=='FAST' and o['dataset']=='ImageNet' and o['metric']=='good20'
    assert o['interval_type']=='predictive_realized_panel'
    assert o['structural_zero_component'] is False
    assert o['source_restrictions']['used_prefix_indices']==[0,1,2,3]
    assert not o['source_restrictions']['held_runs_read']
    assert not o['source_restrictions']['xp_returns_accessed']
    assert len(o['source_restrictions']['held_runs_indices'])==60
    for s in (*o['input_sha256'].values(),o['observed_image_ids_sha256']):
        assert len(s)==64 and all(c in '0123456789abcdef' for c in s)
    assert len(o['predictions'])==100, len(o['predictions'])
    assert len(o['anatomy_curves'])==20, len(o['anatomy_curves'])
    assert len(o['oracle_anchor_checks'])==4, len(o['oracle_anchor_checks'])
    cs=o['g1_class_counts']; assert set(cs)==set('GOPCI')
    assert sum(x['total'] for x in cs.values())==400
    assert all(0<=x['successes']<=x['total'] for x in cs.values())
    assert cs['C']['successes']==cs['I']['successes']==0
    points=defaultdict(dict)
    for p in o['predictions']:
        assert p['model_variant'] in VARS and p['K'] in GRID
        assert p['panel']=='primary' and p['panel_n_max']==64
        assert p['cohort']=='XP_N_fixed100' and p['n_images']==100
        assert p['policy'] in POLICIES
        dm=p['decision_model'];assert dm in ('beta_class','beta_image_cluster') if p['policy']=='E1_CLIP' else dm is None
        key=(p['model_variant'],p['K']);pk=(p['policy'],dm)
        assert pk not in points[key]
        assert all(math.isfinite(float(p[k])) for k in ('point','lo95','hi95'))
        assert -1e-10<=p['lo95']<=p['point']<=p['hi95']<=1+1e-10
        points[key][pk]=p['point']
    assert len(points)==20
    for (v,k),p in points.items():
        assert len(p)==5
        assert p[('E0',None)]<=p[('O_raw',None)]+1e-10
        assert p[('E1_CLIP','beta_class')]<=p[('O_orbit',None)]+1e-10
        assert p[('E1_CLIP','beta_image_cluster')]<=p[('O_orbit',None)]+1e-10
        assert p[('O_raw',None)]<=p[('O_orbit',None)]+1e-10
        for pol in ('O_raw','O_orbit'):
            assert abs(p[(pol,None)]-points[('race',k)][(pol,None)])<1e-12
    seen=set()
    for a in o['anatomy_curves']:
        key=a['model_variant'],a['K'];assert key in points and key not in seen
        seen.add(key);p=points[key]
        assert abs(a['generation_failure']-(1-p[('O_orbit',None)]))<1e-10
        assert abs(a['selection_loss_E0']-(p[('O_raw',None)]-p[('E0',None)]))<1e-10
        assert abs(a['selection_loss_E1']-(p[('O_orbit',None)]-p[('E1_CLIP','beta_class')]))<1e-10
    assert seen==set(points)
    for a in o['oracle_anchor_checks']:
        assert a['variant'] in VARS and a['K']==[4,8,16,32,64]
        assert a['panel']=='primary'
        assert 0<=a['max_abs_shift']<.01
        for key in ('shift_raw','shift_orbit','exact_raw','exact_orbit','simulated_raw','simulated_orbit'):
            assert len(a[key])==5 and all(math.isfinite(float(q)) for q in a[key])
        assert max(abs(x) for x in a['shift_raw']+a['shift_orbit'])<=a['max_abs_shift']+1e-12
    print('XP_N_V5_CONTRACT_PASS 100 20 4')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('draft');check(p.parse_args().draft)
