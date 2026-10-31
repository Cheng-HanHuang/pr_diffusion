#!/usr/bin/env python3
"""A16.5 v5: preregisterable ImageNet FAST Good20 n=64, K=4/8/16/32/64.

NO XP-N results used or allowed. Fit five-class Dirichlet and composite
Bradley--Terry exponential race ONLY on 100 new images' runs 0--3.
CLIP Beta(1,1) class decisions likewise use ONLY those 400 runs.
All later runs are conditionally simulated. No structural-zero spike.
Code intended to be frozen before ANY XP-N run, even calibration runs, exists.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from audit_a8_rank_race import fit_weights
from fit_th1_dirichlet import fit as fit_alpha
from fit_th1_prefix_v1 import sample_panel, rank_weights, pass_at_k, exact_prefix_oracles

NIMAGES=100
NOBS=4
NMAX=64
KGRID=(4,8,16,32,64)
VARIANTS=("race","S2","tail","race_tau1")
LABELS="GOPCI"
SEED=20261101


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sha(path, expected):
    if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('expected exact lowercase SHA-256')
    actual=digest(path)
    if actual!=expected:raise ValueError('SHA-256 mismatch for authorized input '+str(path))
    return actual


def read_records(path):
    with open(path,newline='',encoding='utf8') as f:
        data=list(csv.DictReader(f))
    if not data:raise ValueError('empty CSV')
    return data


def find_column(names, preferred, allowed, description):
    if preferred:
        if preferred not in names:raise ValueError(description+' override absent')
        return preferred
    keys=[c for c in allowed if c in names]
    if len(keys)!=1:raise ValueError(description+' requires one explicit recognizable column: '+repr(keys))
    return keys[0]


def bool_flag(value):
    x=str(value).strip().lower()
    if x in ('1','true','yes'):return 1
    if x in ('0','false','no'):return 0
    raise ValueError('invalid binary G1 flag '+repr(value))


def load_prefix(path,expected_sha):
    check_sha(path,expected_sha)
    source=read_records(path)
    if len(source)!=NIMAGES:raise ValueError('v5 needs exactly 100 new images')
    records=[];by_id={}
    for rec in source:
        uid=rec['image_id'].strip()
        if not uid or uid in by_id:raise ValueError('blank/duplicate input image ID')
        if rec['family'].strip()!='FAST' or int(rec['n_runs'])!=NOBS:
            raise ValueError('wrong solver or illegal run count '+uid)
        orig=rec['run_labels'].strip()
        ranked=rec['e0_rank_labels'].strip()
        order=[int(x.strip()) for x in rec['e0_rank_order'].split(',')]
        if len(orig)!=4 or any(x not in LABELS for x in orig) or len(ranked)!=4:
            raise ValueError('invalid four-run labels '+uid)
        if sorted(order)!=list(range(NOBS)) or ''.join(orig[i]for i in order)!=ranked:
            raise ValueError('bad original four rank order '+uid)
        if 'rho_signed_by_run' in rec and rec['rho_signed_by_run'].strip():
            rho=[float(v) for v in rec['rho_signed_by_run'].split(';')]
            if len(rho)!=NOBS or not all(np.isfinite(rho)) or any(
                rho[order[i]]>rho[order[i+1]]+1e-10 for i in range(NOBS-1)):
                raise ValueError('bad residual rank '+uid)
        count=np.asarray([int(rec[k]) for k in ('n_GOOD','n_ORBIT_global',
                 'n_ORBIT_partial','n_CONSISTENT','n_INCONSISTENT')],dtype=int)
        if np.any(count<0) or count.tolist()!=[orig.count(c) for c in LABELS]:
            raise ValueError('bad class counts '+uid)
        records.append({'image_id':uid,'family':'FAST','cohort':'IMAGENET',
                 'n':NOBS,'counts':count,'ranked':ranked,'original':orig})
        by_id[uid]=rec
    return records,by_id


def load_clip(path,expected_sha,rows,recmap,args):
    check_sha(path,expected_sha)
    recs=read_records(path)
    if len(recs)!=NIMAGES*NOBS:raise ValueError('v5 G1 needs exactly 400 rows, runs 0..3 only')
    cols=list(recs[0])
    col={
      'image':find_column(cols,args.image_col,('image_id','image','image_name'),'image'),
      'run':find_column(cols,args.run_col,('run_index','run_idx','run_id','run'),'run'),
      'label':find_column(cols,args.label_col,('run_label','label','class_label','class'),'label'),
      'good':find_column(cols,args.good_col,('g1_chosen_good','g_tilde_good','chosen_orientation_good','g1_good','g1_selected_good'),'Good20')}
    rowmap={r['image_id']:r for r in rows}
    old={};tot=np.zeros(5,dtype=int);succ=np.zeros(5,dtype=int)
    for v in recs:
        uid=v[col['image']].strip()
        j=int(v[col['run']])
        if uid not in rowmap or not 0<=j<NOBS:raise ValueError('wrong image or held run index')
        key=(uid,j)
        if key in old:raise ValueError('duplicate G1 run')
        label=v[col['label']].strip()
        if label!=rowmap[uid]['original'][j]:raise ValueError('class label conflicts with prefix '+uid)
        flag=bool_flag(v[col['good']])
        if label in 'CI' and flag:raise ValueError('C/I falsely marked GOOD by G1 label')
        old[key]=flag
        cl=LABELS.index(label);tot[cl]+=1;succ[cl]+=flag
    if len(old)!=NIMAGES*NOBS or int(tot.sum())!=NIMAGES*NOBS:raise ValueError('incomplete G1 inputs')
    flags={}
    for uid,v in recmap.items():
        order=[int(x) for x in v['e0_rank_order'].split(',')]
        flags[uid]=np.asarray([old[(uid,j)] for j in order],dtype=int)
    return flags,succ,tot,col


def beta_rates(succ,tot,rng):
    # E1 Good20 can only result from classes G/O/P in this classification.
    p=np.zeros(5,dtype=float)
    for cl in range(3):p[cl]=rng.beta(succ[cl]+1,tot[cl]-succ[cl]+1)
    return p


def generate_panel(rows,alpha,w,variant,old_flags,rates,rng,rank_cache):
    result=np.zeros((len(KGRID),4),dtype=float)
    for row in rows:
        labels,scores=sample_panel(row,alpha,w,variant,NMAX,rng)
        if len(labels)!=NMAX:raise AssertionError('bad panel size')
        chosen=np.zeros(NMAX,dtype=float)
        chosen[:NOBS]=old_flags[row['image_id']]
        for i in range(NOBS,NMAX):
            cl=int(labels[i])
            if variant=='tail':
                # tail-expanded C_base=3, C_extreme=4, I=5
                cl=4 if cl==5 else 3 if cl==4 else cl
            if not 0<=cl<=4:raise AssertionError('illegal class')
            chosen[i]=float(rng.random()<rates[cl])
        order=np.argsort(scores,kind='stable')
        selected_good=(labels[order]==0).astype(float)
        selected_g1=chosen[order]
        ng=int(np.sum(labels==0));no=int(np.sum(labels<=2))
        for ki,K in enumerate(KGRID):
            rank=rank_cache[K]
            result[ki,0]+=float(np.dot(rank,selected_good))
            result[ki,1]+=float(np.dot(rank,selected_g1))
            result[ki,2]+=pass_at_k(NMAX,ng,K)
            result[ki,3]+=pass_at_k(NMAX,no,K)
    return result/len(rows)


def fit_models(rows):
    alpha=fit_alpha(np.stack([row['counts'] for row in rows]))
    pri=fit_weights(rows,tau=2.5)
    tau=fit_weights(rows,tau=1.)
    return alpha,pri,tau


def run(rows,old_flags,succ,tot,point_reps,bootstrap,boot_reps):
    if point_reps<2 or bootstrap<2 or boot_reps<1:raise ValueError('simulation budget too small')
    rng=np.random.default_rng(SEED)
    alpha,primary,tau=fit_models(rows)
    exact=np.mean([exact_prefix_oracles(row,alpha,KGRID,NMAX) for row in rows],axis=0)
    rank={K:rank_weights(NMAX,K) for K in KGRID}
    records=[];anatomy=[];anchors=[]
    for variant in VARIANTS:
        w=tau if variant=='race_tau1' else primary
        sims=np.asarray([generate_panel(rows,alpha,w,variant,old_flags,beta_rates(succ,tot,rng),rng,rank)
                           for _ in range(point_reps)],dtype=float)
        avg=sims.mean(axis=0)
        shifts=exact-avg[:,2:]
        if not np.all(np.isfinite(shifts)) or np.max(np.abs(shifts))>=.01:
            raise ValueError('exact oracle anchoring contract > .01: '+repr(shifts))
        point=avg.copy()
        point[:,:2]+=shifts
        point[:,2:]=exact
        centered=sims.copy()
        centered[:,:,:2]+=shifts[None,:,:]
        centered[:,:,2:]+=shifts[None,:,:]
        boot=[]
        for b in range(bootstrap):
            ix=rng.integers(0,NIMAGES,NIMAGES)
            sample=[rows[int(i)] for i in ix]
            ab,wb,wt=fit_models(sample)
            bw=wt if variant=='race_tau1' else wb
            bs=np.zeros(5,dtype=int);bt=np.zeros(5,dtype=int)
            for i in ix:
                row=rows[int(i)]
                for cl,flag in zip(row['ranked'],old_flags[row['image_id']]):
                    j=LABELS.index(cl);bt[j]+=1;bs[j]+=int(flag)
            for _ in range(boot_reps):
                boot.append(generate_panel(rows,ab,bw,variant,old_flags,
                                           beta_rates(bs,bt,rng),rng,rank))
        boot=np.asarray(boot,dtype=float)
        boot[:,:,:2]+=shifts[None,:,:]
        boot[:,:,2:]+=shifts[None,:,:]
        for draws in (centered,boot):
            if (np.any(draws[:,:,0]>draws[:,:,2]+1e-10) or
                np.any(draws[:,:,1]>draws[:,:,3]+1e-10)):
                raise ValueError('paired oracle contradiction')
        if np.any(point[:,0]>point[:,2]+1e-10) or np.any(point[:,1]>point[:,3]+1e-10):
            raise ValueError('oracle point contradiction')
        primary_interval=np.quantile(centered,[.025,.975],axis=0)
        cluster_interval=np.quantile(np.concatenate((centered,boot),axis=0),[.025,.975],axis=0)
        for ki,K in enumerate(KGRID):
            for policy,col in (('E0',0),('E1_CLIP',1),('O_raw',2),('O_orbit',3)):
                center=float(point[ki,col])
                lo,hi=primary_interval[:,ki,col]
                rec={'family':'FAST','dataset':'ImageNet','metric':'good20',
                     'cohort':'XP_N_fixed100','n_images':NIMAGES,'panel':'primary',
                     'panel_n_max':NMAX,'K':K,'model_variant':variant,
                     'policy':policy,'decision_model':'beta_class' if policy=='E1_CLIP' else None,
                     'point':center,'lo95':float(np.clip(min(center,lo),0,1)),
                     'hi95':float(np.clip(max(center,hi),0,1))}
                records.append(rec)
                if policy=='E1_CLIP':
                    blo,bhi=cluster_interval[:,ki,col]
                    records.append({**rec,'decision_model':'beta_image_cluster',
                       'lo95':float(np.clip(min(center,blo),0,1)),
                       'hi95':float(np.clip(max(center,bhi),0,1))})
            anatomy.append({'panel':'primary','panel_n_max':NMAX,'K':K,
                'model_variant':variant,'generation_failure':float(1-point[ki,3]),
                'selection_loss_E0':float(point[ki,2]-point[ki,0]),
                'selection_loss_E1':float(point[ki,3]-point[ki,1])})
        anchors.append({'variant':variant,'panel':'primary','K':list(KGRID),
            'exact_raw':exact[:,0].tolist(),'exact_orbit':exact[:,1].tolist(),
            'simulated_raw':avg[:,2].tolist(),'simulated_orbit':avg[:,3].tolist(),
            'shift_raw':shifts[:,0].tolist(),'shift_orbit':shifts[:,1].tolist(),
            'max_abs_shift':float(np.max(np.abs(shifts)))})
    return records,anatomy,anchors,alpha,primary,tau


def main():
    p=argparse.ArgumentParser()
    for a in ('prefix4','prefix4_sha256','clip4','clip4_sha256','output'):
        p.add_argument('--'+a.replace('_','-'),required=True)
    for a in ('image_col','run_col','label_col','good_col'):
        p.add_argument('--'+a.replace('_','-'))
    p.add_argument('--point-reps',type=int,default=250)
    p.add_argument('--bootstrap',type=int,default=60)
    p.add_argument('--boot-reps',type=int,default=3)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('do not overwrite v5 draft')
    rows,recmap=load_prefix(a.prefix4,a.prefix4_sha256)
    flags,succ,tot,columns=load_clip(a.clip4,a.clip4_sha256,rows,recmap,a)
    output,anatomy,checks,alpha,w,w1=run(rows,flags,succ,tot,a.point_reps,a.bootstrap,a.boot_reps)
    obj={'version':'DRAFT_XP_N_V5_UNREGISTERED','status':'UNREGISTERED',
         'family':'FAST','dataset':'ImageNet','metric':'good20',
         'n_images':NIMAGES,'n_observed_runs':NOBS,'n_max':NMAX,'K_grid':list(KGRID),
         'model_variants':list(VARIANTS),'structural_zero_component':False,
         'input_sha256':{'prefix4':digest(a.prefix4),'clip4':digest(a.clip4)},
         'observed_image_ids_sha256':hashlib.sha256(('\n'.join(sorted(r['image_id'] for r in rows))+'\n').encode()).hexdigest(),
         'source_schema_columns':columns,'seed':SEED,
         'point_reps':a.point_reps,'bootstrap':a.bootstrap,'boot_reps':a.boot_reps,
         'interval_type':'predictive_realized_panel','predictions':output,
         'anatomy_curves':anatomy,'oracle_anchor_checks':checks,
         'alpha':np.asarray(alpha,dtype=float).tolist(),
         'weights_primary':{k:float(w[k]) for k in ('w_O','w_C')},
         'weights_tau1':{k:float(w1[k]) for k in ('w_O','w_C')},
         'g1_class_counts':{LABELS[i]:{'total':int(tot[i]),'successes':int(succ[i])}
               for i in range(5)},
         'comparison_rule':('Primary n64 K4/8/16/32/64: E0+E1 beta_class coverage; '
           'break ties by mean absolute standardized residual abs(obs-point)/max((hi-lo)/3.92,0.005); '
           'disclose O_raw/O_orbit, S2, tail and race_tau1 and G1 cluster sensitivity'),
         'source_restrictions':{'used_prefix_indices':list(range(NOBS)),
             'held_runs_indices':list(range(NOBS,NMAX)),
             'held_runs_read':False,'xp_returns_accessed':False}}
    target=Path(a.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print('XP_N_V5_DRAFT_COUNTS',len(output),len(anatomy),len(checks),str(target))

if __name__=='__main__':main()
