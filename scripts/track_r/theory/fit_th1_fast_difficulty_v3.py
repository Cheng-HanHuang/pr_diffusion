#!/usr/bin/env python3
"""A10 optional FFHQ B16/F8 v3: fixed FAST64 difficulty-covariate forecast.

NEVER reads target-solver K>4 (SITCOM) or K>2/4 (DAPS) outcomes.
Inputs: registered A8/A8b (SHA guarded), allowed FAST 64 per-image
labels (SHA guarded). The external FAST post-prefix data were UNBLINDED
before v3 design and are declared as such. Output is a separate v3 and
MUST NOT overwrite any registered v1/v2 artifact.

Selection: identical family-specific race/S2/tail/race_tau1 rank laws
from v1, including the exact original 0..n0-1 rank permutation.
Generation: Dirichlet class prior tilted by cross-solver FAST goodness,
with a fixed, soft atom of images with P(G/O/P)=0. Hyperparameter choices
are openly model choices, not learned from held-out B16/F8.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from audit_a8_rank_race import read_a8,crosscheck,fit_weights
from fit_th1_dirichlet import fit as fit_alpha
from fit_th1_prefix_v1 import (conditional_old_scores,score_weights,
    exact_subset_values,rank_weights)
A8_SHA='197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6'
A8B_SHA='e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901'
FAST_SHA='d616dd50f10352e98b891a99b9a153d5d72a93c61e44381827147dcca23f5466'
SEED=20261028
FAMILIES=('DAPS','SITCOM')
PANELS={'DAPS':{'primary':(8,(4,6,8))},'SITCOM':{
 'primary':(16,(8,12,16)),'secondary':(32,(8,12,16,24,32))}}
VARIANTS=('race','S2','tail','race_tau1')
POLICIES=('E0','E1_CLIP','O_raw','O_orbit')
COHORTS=('B22','DEV80','POOLED')
TILT_BETA=.5
# Soft structural-zero priors for the three orbit-success classes.
# These prespecified values are deliberately not 1 even if FAST never
# succeeds: one DAPS prefix already contains a success on a FAST-zero image.
ZERO_PRIOR={'g0':.60,'g1_4':.25,'g5_15':.08,'g16plus':.015}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_inputs(a8,a8b,fast):
 if digest(a8)!=A8_SHA or digest(a8b)!=A8B_SHA or digest(fast)!=FAST_SHA:
  raise ValueError('Unauthorized calibration or FAST difficulty checksum')
 with open(a8,newline='') as f:old=list(csv.DictReader(f))
 with open(a8b,newline='') as f:new=list(csv.DictReader(f))
 if len(old)!=760 or len(new)!=760 or any(any(
     r[k]!=s[k] for k in r) for r,s in zip(old,new)):
  raise ValueError('A8b did not exactly preserve original A8 columns')
 rows,_=read_a8(a8);crosscheck(rows)
 with open(fast,newline='') as f:fastrows=list(csv.DictReader(f))
 if len(fastrows)!=180:raise ValueError('FAST 64 file must have 180 images')
 fmap={};baseline={}
 for r in fastrows:
  key=(r['cohort'],r['image_id'])
  if key in fmap:raise ValueError('duplicate FAST image')
  labels=r['labels_by_run'].split(';')
  if len(labels)!=64 or any(x not in ('G','O_g','O_p','C','I') for x in labels):
   raise ValueError('FAST source must have 64 full run labels')
  if (sum(int(r[k]) for k in ('n_G','n_O_g','n_O_p','n_C','n_I'))!=64 or
      int(r['n_G'])!=labels.count('G')):
   raise ValueError('FAST 64 class-count mismatch')
  fmap[key]=int(r['n_G']);baseline[key]=labels[:4]
 for r in rows:
  key=(r['cohort'],r['image_id'])
  if key not in fmap:raise ValueError('FAST image missing '+str(key))
  if r['family']=='FAST':
   normalized=['O' if x=='O_g' else 'P' if x=='O_p' else x
               for x in baseline[key]]
   if list(r['original'])!=normalized:
    raise ValueError('FAST64 prefix changed A8 historical source')
  r['fast_g64']=fmap[key]
 # Strictly keep only DAPS and SITCOM; never consume another solver's
 # new K outcome or any data from xp/returns.
 target=[r for r in rows if r['family'] in FAMILIES]
 for fam in FAMILIES:
  assert len([r for r in target if r['family']==fam])==180
 return target,{'A8':A8_SHA,'A8b':A8B_SHA,'FAST_64':FAST_SHA}

def structural_prior(g):
 if g==0:return ZERO_PRIOR['g0']
 if g<=4:return ZERO_PRIOR['g1_4']
 if g<=15:return ZERO_PRIOR['g5_15']
 return ZERO_PRIOR['g16plus']

def alpha_tilt(alpha,g,cohort_fast_g):
 # Log-odds ratio of the Jeffreys-smoothed FAST G frequency to the
 # original cohort mean FAST G frequency. Fixed exponent beta=0.5.
 p=(g+.5)/65.
 center=(cohort_fast_g+.5)/65.
 logodds=lambda q:math.log(q/(1-q))
 factor=math.exp(np.clip(TILT_BETA*(logodds(p)-logodds(center)),-4.,4.))
 a=np.array(alpha,dtype=float).copy()
 a[:3]*=factor
 if not np.all(a>0):raise ValueError('bad tilted prior')
 return a

def hard_posterior(row,a):
 counts=np.asarray(row['counts'],dtype=int)
 if counts[:3].sum()>0:return 0.
 prior=structural_prior(row['fast_g64'])
 # Exact DM evidence ratio: likelihood(active)/likelihood(hard) for
 # an observed prefix containing only C/I. C/I count terms cancel.
 n=row['n'];A=float(a.sum());B=float(a[3:].sum());ratio=1.
 for j in range(n):ratio*=(B+j)/(A+j)
 return prior/(prior+(1-prior)*ratio)

def sample_theta(row,a,rng):
 C=np.asarray(row['counts'],dtype=float)
 hard=(rng.random()<hard_posterior(row,a))
 if hard:
  probs=rng.dirichlet(C[3:]+a[3:]);return np.r_[np.zeros(3),probs]
 return rng.dirichlet(C+a)

def exact_oracle(row,a,Kgrid,nmax):
 C=np.asarray(row['counts'],dtype=float)
 n=row['n'];m=nmax-n
 if m<0:raise ValueError('bad completed panel')
 hard=hard_posterior(row,a)
 A=float((C+a).sum());answer=np.empty((len(Kgrid),2))
 for typ,success in enumerate(((0,),(0,1,2))):
  old=int(sum(C[j]for j in success))
  non=float(sum((C+a)[j] for j in range(5) if j not in success))
  moment=[1.]
  for t in range(1,max(Kgrid)+1):
   moment.append(moment[-1]*(non+t-1)/(A+t-1))
  for ki,K in enumerate(Kgrid):
   zero=0.
   for j in range(max(0,K-m),min(K,n-old)+1):
    future=K-j
    no_success=hard+(1-hard)*moment[future]
    zero+=(math.comb(n-old,j)*math.comb(m,future)/
           math.comb(nmax,K))*no_success
   answer[ki,typ]=1-zero
 return answer

def sample_panel(row,a,weights,variant,nmax,rng):
 theta=sample_theta(row,a,rng)
 orig,oldscores=conditional_old_scores(row['ranked'],variant,weights,rng)
 rates,eps=score_weights(weights,variant)
 future_base=rng.choice(5,size=nmax-row['n'],p=theta)
 if variant=='tail':
  future=np.array([5 if j==4 else
      (4 if rng.random()<eps else 3) if j==3 else int(j)
      for j in future_base],dtype=int)
 else:future=future_base
 future_scores=np.full(len(future),np.inf)
 for j,c in enumerate(future):
  if rates[c]>0:future_scores[j]=rng.exponential(1/rates[c])
 return np.concatenate((orig,future)),np.concatenate((oldscores,future_scores))

def simulate(rows,alpha,weights,variant,Kgrid,nmax,cohort_fast_g,rng,reps):
 cache={(nmax,K):rank_weights(nmax,K)for K in Kgrid}
 # Alpha tilt is fixed per original image and cohort, not per rollout.
 a=[alpha_tilt(alpha,r['fast_g64'],cohort_fast_g)for r in rows]
 panels=np.empty((reps,len(Kgrid),4))
 for b in range(reps):
  acc=np.zeros((len(Kgrid),4))
  for row,ai in zip(rows,a):
   labels,scores=sample_panel(row,ai,weights,variant,nmax,rng)
   acc+=exact_subset_values(labels,scores,Kgrid,cache)
  panels[b]=acc/len(rows)
 return panels

def run(rows,seed,point_reps,bootstrap,boot_reps):
 rng=np.random.default_rng(seed)
 predictions=[];anatomy=[];crossovers=[];fits=[];difficulty=[]
 for family in FAMILIES:
  fr=[r for r in rows if r['family']==family]
  wf=fit_weights(fr,tau=2.5);w1=fit_weights(fr,tau=1.)
  cohorts={}
  for co in ('B22','DEV80'):
   data=[r for r in fr if r['cohort']==co]
   alpha=fit_alpha(np.array([r['counts']for r in data],float))
   gbar=np.mean([r['fast_g64']for r in data])
   cohorts[co]={'rows':data,'alpha':alpha,'gbar':gbar}
   fits.append({'family':family,'cohort':co,'alpha':alpha.tolist(),
       'mean_fast_g64':float(gbar),'race_weights':wf,'race_tau1_weights':w1})
   for r in data:
    ai=alpha_tilt(alpha,r['fast_g64'],gbar)
    difficulty.append({'family':family,'cohort':co,'image_id':r['image_id'],
       'FAST_G64':r['fast_g64'],'target_prefix_orbit':int(sum(r['counts'][:3])),
       'pi_hard_prior':structural_prior(r['fast_g64']),
       'pi_hard_after_prefix':float(hard_posterior(r,ai))})
  for panel,(nmax,Kgrid) in PANELS[family].items():
   exact={}
   main={}
   for co in ('B22','DEV80'):
    d=cohorts[co]
    exact[co]=np.mean([exact_oracle(r,
        alpha_tilt(d['alpha'],r['fast_g64'],d['gbar']),Kgrid,nmax)
        for r in d['rows']],axis=0)
    for v in VARIANTS:
     weights=w1 if v=='race_tau1' else wf
     main[co,v]=simulate(d['rows'],d['alpha'],weights,v,
         Kgrid,nmax,d['gbar'],rng,point_reps)
   boot={}
   for co in ('B22','DEV80'):
    for v in VARIANTS:boot[co,v]=[]
   for b in range(bootstrap):
    sampled=[];alphas={}
    for co in ('B22','DEV80'):
     d=cohorts[co];indices=rng.integers(0,len(d['rows']),len(d['rows']))
     rs=[d['rows'][int(j)]for j in indices];sampled.extend(rs)
     alphas[co]=fit_alpha(np.array([r['counts']for r in rs],float))
    wb=fit_weights(sampled,tau=2.5)
    wb1=fit_weights(sampled,tau=1.)
    for co in ('B22','DEV80'):
     d=cohorts[co]
     for v in VARIANTS:
      weights=wb1 if v=='race_tau1' else wb
      vals=simulate(d['rows'],alphas[co],weights,v,
          Kgrid,nmax,d['gbar'],rng,boot_reps)
      boot[co,v].append(vals.mean(axis=0))
   for co in COHORTS:
    for v in VARIANTS:
     if co=='POOLED':
      sim=100/180*main['B22',v]+80/180*main['DEV80',v]
      bf=100/180*np.asarray(boot['B22',v])+80/180*np.asarray(boot['DEV80',v])
      oracle=100/180*exact['B22']+80/180*exact['DEV80'];nimg=180
     else:
      sim=main[co,v];bf=np.asarray(boot[co,v]);oracle=exact[co]
      nimg=100 if co=='B22' else 80
     base=sim.mean(axis=0)
     # Exact oracle means and common-loss control variate: no selector
     # mean can exceed its own exact oracle even with noisy Monte Carlo.
     point=base.copy()
     point[:,2:]=oracle
     point[:,0]=oracle[:,0]-(base[:,2]-base[:,0])
     point[:,1]=oracle[:,1]-(base[:,3]-base[:,1])
     # Perturb realized oracle + selection LOSS jointly. This keeps
     # predictive draws physically feasible and intervals comparable.
     ids=rng.integers(0,len(bf),len(sim))
     val=sim+(bf[ids]-bf.mean(axis=0)[None,:,:])
     val[:,:,2:]+=(oracle-base[:,2:])[None,:,:]
     val[:,:,2:]=np.clip(val[:,:,2:],0,1)
     lossraw=np.clip(val[:,:,2]-val[:,:,0],0,val[:,:,2])
     lossorb=np.clip(val[:,:,3]-val[:,:,1],0,val[:,:,3])
     val[:,:,0]=val[:,:,2]-lossraw
     val[:,:,1]=np.maximum(val[:,:,0],val[:,:,3]-lossorb)
     val[:,:,1]=np.minimum(val[:,:,1],val[:,:,3])
     lo,hi=np.quantile(val,(.025,.975),axis=0)
     gf=1-point[:,3];sl=point[:,3]-point[:,1]
     cross=next((int(k)for ki,k in enumerate(Kgrid)
                 if sl[ki]>gf[ki]),None)
     crossovers.append({'family':family,'panel':panel,'panel_n_max':nmax,
       'cohort':co,'model_variant':v,'K_grid':list(Kgrid),
       'K_star_on_grid':cross})
     gd=1-val[:,:,3];ld=val[:,:,3]-val[:,:,1]
     gl,gh=np.quantile(gd,(.025,.975),axis=0)
     ll,lh=np.quantile(ld,(.025,.975),axis=0)
     for ki,K in enumerate(Kgrid):
      anatomy.append({'family':family,'cohort':co,'panel':panel,
       'panel_n_max':nmax,'model_variant':v,'K':K,
       'generation_failure':float(gf[ki]),
       'selection_loss':float(sl[ki]),
       'generation_lo95':float(np.clip(gl[ki],0,1)),
       'generation_hi95':float(np.clip(gh[ki],0,1)),
       'selection_lo95':float(np.clip(ll[ki],0,1)),
       'selection_hi95':float(np.clip(lh[ki],0,1))})
      if family=='DAPS' and co!='B22' and K==4:continue
      for j,policy in enumerate(POLICIES):
       x=float(point[ki,j])
       predictions.append({'family':family,'dataset':'FFHQ',
        'metric':'good25','cohort':co,'panel':panel,'panel_n_max':nmax,
        'model_variant':v,'policy':policy,'K':K,'n_images':nimg,
        'point':x,'lo95':float(np.clip(min(x,lo[ki,j]),0,1)),
        'hi95':float(np.clip(max(x,hi[ki,j]),0,1))})
 return predictions,anatomy,crossovers,fits,difficulty

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--a8',required=True);p.add_argument('--a8b',required=True)
 p.add_argument('--fast64',required=True);p.add_argument('--output',required=True)
 p.add_argument('--point-reps',type=int,default=250)
 p.add_argument('--bootstrap',type=int,default=60)
 p.add_argument('--boot-reps',type=int,default=3)
 p.add_argument('--freeze',action='store_true')
 args=p.parse_args()
 if args.freeze:raise RuntimeError('Only planner can seal/register v3 after QA')
 rows,sha=read_inputs(args.a8,args.a8b,args.fast64)
 pred,ana,cross,fits,diff=run(rows,SEED,args.point_reps,
                                 args.bootstrap,args.boot_reps)
 out={'version':'DRAFT_TH1_V3_NOT_REGISTERED',
  'scope':'FFHQ DAPS F8 and SITCOM B16 only, with B32 secondary',
  'metric':'good25','seed':SEED,
  'interval_type':'predictive_realized_panel',
  'source_sha256':sha,'selection_variants':list(VARIANTS),
  'policies':list(POLICIES),'panels':PANELS,
  'structural_zero_prior':ZERO_PRIOR,'alpha_fast_tilt_beta':TILT_BETA,
  'interpretation':'EXPLORATORY second prospective version; FAST64 cross-solver post-outcome covariate allowed; all DAPS/SITCOM post-prefix runs withheld',
  'comparison_rule':'Primary panels E0 and E1_CLIP predictive coverage [lo95,hi95] at registered K, compare race versus S2, standardized residual tie-break; secondary and prior sensitivity separate',
  'model':'A8 cohort Dirichlet class model, FAST64 covariate log-odds tilt, soft structural-zero (G/O/P absent) hurdle updated on only target historical prefix; same A8 residual race laws and original ranks',
  'hyperparameter_fits':fits,'per_image_prior_audit':diff,
  'predictions':pred,'anatomy_curves':ana,'crossover_K':cross,
  'provenance':{'FAST64_results_released_before_v3':True,
       'DAPS_post_prefix_outcomes_seen':False,
       'SITCOM_post_prefix_outcomes_seen':False,
       'xp_returns_directory_opened':False,
       'v1_frozen_and_unchanged':True}}
 target=Path(args.output)
 if target.exists():raise FileExistsError('draft exists; no overwrite')
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print('V3_DIFFICULTY_DRAFT_PASS',len(pred),len(ana),len(cross),len(diff),target)
if __name__=='__main__':main()
