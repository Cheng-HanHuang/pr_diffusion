#!/usr/bin/env python3
"""C-T8: conservative population Good64 identification from four-run FFHQ FAST.

Only authorized A8 four-run GOOD counts. Uses Bonferroni Clopper-Pearson
simultaneous 95% confidence bands for the five binomial-mixture outcome
probabilities q_j=Pr[exactly j successes in four iid runs]. No K>4 data
are read. A grid LP with provable Lipschitz discretization corrections
encloses ALL continuum mixing distributions compatible with the band.

Interpretation: population mixture under iid image sampling AND iid runs
conditional on per-image p, NOT the same-fixed-image conditional-prefix
predictive target of registered v1. DEV80's enrichment mechanism may
violate the iid population-sampling assumption; it is a descriptive
cohort-conditional sensitivity rather than a universal 95% guarantee.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import beta
from scipy.optimize import linprog
A8_SHA='197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6'
K0=4;K=64;ALPHA=.05

def run(a8,grid_size=10001):
 if hashlib.sha256(Path(a8).read_bytes()).hexdigest()!=A8_SHA:
  raise ValueError('A8 source SHA mismatch')
 if grid_size<201:raise ValueError('grid too small for useful outer bound')
 with open(a8,newline='')as f:rows=list(csv.DictReader(f))
 rows=[r for r in rows if r['family']=='FAST' and r['source']=='S5_r0_fast']
 assert len(rows)==180
 x=np.linspace(0,1,grid_size);h=1/(grid_size-1)
 A=np.array([math.comb(4,j)*x**j*(1-x)**(4-j)for j in range(5)])
 obj=1-(1-x)**64
 output={}
 for cohort,name in (('B22_random','B22'),('DEV80_enriched','DEV80')):
  r=[z for z in rows if z['cohort']==cohort]
  n=len(r);counts=np.bincount([int(z['n_GOOD'])for z in r],minlength=5)
  assert n==(100 if name=='B22' else 80)
  low=np.array([0 if k==0 else beta.ppf(ALPHA/(2*5),k,n-k+1)for k in counts])
  high=np.array([1 if k==n else beta.ppf(1-ALPHA/(2*5),k+1,n-k)for k in counts])
  # Any continuum p moved to nearest grid point changes each q_j
  # by <= 4*(h/2) = 2h (Bernstein polynomial derivative <=4).
  qerror=2*h
  ub=np.r_[A,-A];bounds=np.r_[high+qerror,-low+qerror]
  opt={'A_ub':ub,'b_ub':bounds,'A_eq':np.ones((1,len(x))),
       'b_eq':[1.], 'bounds':(0,None),'method':'highs'}
  lo=linprog(obj,**opt);hi=linprog(-obj,**opt)
  if not(lo.success and hi.success):
   raise RuntimeError('LP failed: '+lo.message+' / '+hi.message)
  # |(1-(1-p)^64)'| <=64, so grid objective differs by <=32h.
  ferr=32*h
  output[name]={'n_images':n,'count_hist_0_to4':counts.tolist(),
     'simultaneous_95_CP_lower':low.tolist(),
     'simultaneous_95_CP_upper':high.tolist(),
     'LP_relaxed_grid_min':float(lo.fun),
     'LP_relaxed_grid_max':float(-hi.fun),
     'continuum_coverage_outer_lower':max(0.,float(lo.fun-ferr)),
     'continuum_coverage_outer_upper':min(1.,float(-hi.fun+ferr)),
     'LP_status':'OPTIMAL'}
 return {'version':'C_T8_DATA_SPECIFIC_MOMENT_LP_20261028',
  'source_A8_sha256':A8_SHA,'basis':'FFHQ FAST K4 per-image GOOD-count frequencies',
  'K0':K0,'target_K':K,'alpha_joint':ALPHA,
  'interval_basis':'5 Clopper-Pearson categories at two-sided alpha=.01 each; Bonferroni simultaneous coverage >=.95 under iid sampled images',
  'model_assumptions':'iid image sampling from each cohort population, conditionally iid four candidate outcomes at image-specific p',
  'grid_points':grid_size,'p_mesh_spacing':h,
  'max_qj_grid_error':2*h,'max_objective_grid_error':32*h,
  'guarantee':'Continuum population-mixing outer bound, conditional on CP joint event; not a distribution-free conditional-fixed-image forecast',
  'warning':'DEV80 selection/enrichment can violate iid image sampling; numbers are conditional sensitivity, not unconditional coverage guarantee',
  'results':output}

def main():
 p=argparse.ArgumentParser();p.add_argument('--a8',required=True)
 p.add_argument('--output',required=True);p.add_argument('--grid-size',type=int,default=10001)
 a=p.parse_args();result=run(a.a8,a.grid_size)
 target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 for co,z in result['results'].items():
  print('C_T8_LP_PASS',co,'outer95',z['continuum_coverage_outer_lower'],z['continuum_coverage_outer_upper'])
if __name__=='__main__':main()
