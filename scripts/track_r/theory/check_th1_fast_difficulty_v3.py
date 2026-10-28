#!/usr/bin/env python3
"""Strict A10 B16/F8 v3 forecast contract; reads draft JSON only."""
import argparse,json,math
from collections import defaultdict
FAMS={'DAPS':{'primary':(8,(4,6,8))},
      'SITCOM':{'primary':(16,(8,12,16)),
                'secondary':(32,(8,12,16,24,32))}}
CO=('B22','DEV80','POOLED')
VAR=('race','S2','tail','race_tau1')
POL=('E0','E1_CLIP','O_raw','O_orbit')
HASHES={'A8':'197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6',
 'A8b':'e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901',
 'FAST_64':'d616dd50f10352e98b891a99b9a153d5d72a93c61e44381827147dcca23f5466'}
def check(path):
 with open(path)as f:o=json.load(f)
 assert o['version']=='DRAFT_TH1_V3_NOT_REGISTERED'
 assert o['interval_type']=='predictive_realized_panel'
 assert o['source_sha256']==HASHES
 assert o['panels']=={f:{p:[n,list(k)]for p,(n,k)in x.items()}for f,x in FAMS.items()}
 assert o['seed']==20261028 and o['alpha_fast_tilt_beta']==.5
 assert o['structural_zero_prior']=={'g0':.6,'g1_4':.25,'g5_15':.08,'g16plus':.015}
 assert o['provenance']['FAST64_results_released_before_v3']is True
 assert o['provenance']['DAPS_post_prefix_outcomes_seen']is False
 assert o['provenance']['SITCOM_post_prefix_outcomes_seen']is False
 assert o['provenance']['xp_returns_directory_opened']is False
 assert len(o['hyperparameter_fits'])==4
 assert len(o['per_image_prior_audit'])==360
 assert len(o['predictions'])==496,len(o['predictions'])
 assert len(o['anatomy_curves'])==132,len(o['anatomy_curves'])
 assert len(o['crossover_K'])==36,len(o['crossover_K'])
 ds=defaultdict(dict);seen=set()
 for r in o['predictions']:
  f=r['family'];p=r['panel'];co=r['cohort'];v=r['model_variant'];k=r['K'];policy=r['policy']
  assert f in FAMS and p in FAMS[f] and co in CO and v in VAR and policy in POL
  assert (r['panel_n_max'],k) in [(FAMS[f][p][0],x)for x in FAMS[f][p][1]]
  assert r['n_images']=={'B22':100,'DEV80':80,'POOLED':180}[co]
  assert not(f=='DAPS' and co!='B22' and k==4)
  key=(f,p,co,v,k,policy);assert key not in seen;seen.add(key)
  assert all(math.isfinite(r[x])for x in ('point','lo95','hi95'))
  assert 0<=r['lo95']<=r['point']<=r['hi95']<=1
  ds[(f,p,co,v,k)][policy]=r['point']
 for key,values in ds.items():
  assert set(values)==set(POL)
  assert values['E0']<=values['O_raw']+1e-9
  assert values['E1_CLIP']<=values['O_orbit']+1e-9
  assert values['E1_CLIP']+1e-9>=values['E0']
  assert values['O_orbit']+1e-9>=values['O_raw']
  can=ds[(*key[:3],'race',key[4])]
  assert abs(values['O_raw']-can['O_raw'])<1e-12
  assert abs(values['O_orbit']-can['O_orbit'])<1e-12
 for r in o['anatomy_curves']:
  f,p,co,v,k=(r[x]for x in ('family','panel','cohort','model_variant','K'))
  if f=='DAPS' and k==4 and co!='B22':continue
  val=ds[(f,p,co,v,k)]
  assert abs(r['generation_failure']-(1-val['O_orbit']))<1e-9
  assert abs(r['selection_loss']-(val['O_orbit']-val['E1_CLIP']))<1e-9
  assert abs(r['generation_failure']+r['selection_loss']+val['E1_CLIP']-1)<1e-9
 for c in o['crossover_K']:
  f,p=c['family'],c['panel']
  assert c['K_star_on_grid'] is None or c['K_star_on_grid'] in FAMS[f][p][1]
 print('TH1_FAST_DIFFICULTY_V3_CONTRACT_PASS',len(o['predictions']),len(o['anatomy_curves']),len(o['crossover_K']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('draft');check(p.parse_args().draft)
