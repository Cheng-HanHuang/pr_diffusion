"""TB-0: exact homometric four-atom CPU falsification experiment.
No GT information enters arm proposals, refinement, energy or acceptance.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2

N, M, SIGMA = 18, 64, .05
SUPPORTS = ((0,1,4,10,12,17),(0,1,8,11,13,17))
A=np.zeros(N); B=np.zeros(N)
A[list(SUPPORTS[0])]=1.; B[list(SUPPORTS[1])]=1.
MU=np.array([A,A[::-1],B,B[::-1]])
W=np.array([.46,.46,.04,.04]); LOG_W=np.log(W)
ARMS=('S0','S1','S2','S3','S4')
# All frozen params are loaded from JSON, never selected from CHECK outcomes.

def fft(x, n=M):return np.fft.fft(x,n=n,norm='ortho')
def amp(x, n=M):return np.abs(fft(x,n))
def ifft(X):return np.fft.ifft(X,norm='ortho').real

def preflight():
    corrs=[np.correlate(np.asarray(m,dtype=np.int64),np.asarray(m,dtype=np.int64),'full').tolist() for m in MU]
    assert all(c==corrs[0] for c in corrs)
    assert len({tuple(m) for m in MU})==4
    assert all(np.count_nonzero(m)==6 for m in MU)
    gaps={n:float(np.max(np.abs(amp(MU[0],n)-amp(MU[2],n)))) for n in (36,64,128)}
    assert max(gaps.values())<1e-12
    return {'autocorrelation':corrs[0], 'fft_mismatch':gaps,
            'four_distinct_atoms':True, 'orbits':{'A':[0,1], 'B':[2,3]},
            'chi2_64_99':float(chi2.ppf(.99,M))}

def posterior(y):
    # Unnormalised Gaussian likelihood; shared factors cancel.
    logits=LOG_W-np.sum((np.array([amp(m) for m in MU])-y)**2,axis=1)/(2*SIGMA**2)
    return np.exp(logits-logsumexp(logits))

def stats(u,alpha,weights=W):
    tau2=(1.-alpha)/alpha
    l=np.log(weights)-.5*np.sum((MU-u)**2,axis=1)/tau2
    r=np.exp(l-logsumexp(l))
    center=r @ MU
    energy=.5*N*np.log(2*np.pi*tau2)-logsumexp(l)
    score=(center-u)/tau2
    return float(energy),score,center,r

def z_stats(z,alpha,weights=W):
    u=z/np.sqrt(alpha)
    E,g,m,r=stats(u,alpha,weights)
    # E_z = E_clean - (N/2) log(alpha) ; grad_z = grad_u / sqrt(alpha)
    return E-.5*N*np.log(alpha), g/np.sqrt(alpha),m,r

def measurement_error(u,y):return float(np.sum((amp(u)-y)**2)/SIGMA**2)

def energy(u,alpha,ledger):
    ledger['energy']+=1
    return stats(u,alpha)[0]

def score(u,alpha,ledger):
    ledger['score']+=1
    return stats(u,alpha)[1]

def native_step(u,alpha,eta,ledger):
    ledger['refinement_steps']+=1
    # Analytic conditional-mean relaxed step, not a native DAPS implementation.
    g=score(u,alpha,ledger)
    return u+eta*((1-alpha)/alpha)*g

def refine(u,alpha,nsteps,eta,ledger):
    for _ in range(nsteps):u=native_step(u,alpha,eta,ledger)
    return u

def hermitian_direction(rng):
    d=np.zeros(M,dtype=float)
    d[1:M//2]=rng.standard_normal(M//2-1)
    d[M//2+1:]=-d[1:M//2][::-1]
    return d/max(np.linalg.norm(d),1e-14)

def aligned_direction(u,g,eps,floor):
    Z=fft(u);G=fft(g)
    den=np.abs(Z)**2+eps
    d=np.imag(np.conj(Z)*G)/den
    d[np.abs(Z)<floor]=0.
    d[0]=d[M//2]=0.
    # This d is antisymmetric by construction for real input, enforce numerical parity.
    for i in range(1,M//2):
        v=.5*(d[i]-d[M-i]);d[i]=v;d[M-i]=-v
    norm=np.linalg.norm(d)
    return d/max(norm,1e-14),norm

def phase_crop(u,angles,ledger):
    z=fft(u);ledger['fft']+=1
    Zp=np.abs(z)*np.exp(1j*(np.angle(z)+angles))
    # The pre-crop magnitude is frozen, but support projection generally breaks it.
    before=float(np.linalg.norm(np.abs(Zp)-np.abs(z)))
    full=ifft(Zp);ledger['ifft']+=1
    crop=full[:N].copy()
    after=float(np.linalg.norm(amp(crop)-amp(u)));ledger['fft']+=1
    imaginary=float(np.max(np.abs(np.fft.ifft(Zp,norm='ortho').imag)))
    return crop,before,after,imaginary

def new_ledger():return {'score':0,'energy':0,'fft':0,'ifft':0,'refinement_steps':0,'likelihood':0,'proposals':0,'dummy_score':0,'dummy_fft':0}

def budget_pad(u,alpha,ledger,target):
    # Common billed access: count unused score calls explicitly, NOT as actual needed calls.
    for key in ('score','fft','ifft'):
        if ledger[key]<target[key]:
            count=target[key]-ledger[key]
            if key=='score':
                for _ in range(count): score(u,alpha,ledger)
                ledger['dummy_score']+=count
            elif key=='fft':
                for _ in range(count):fft(u)
                ledger['fft']+=count;ledger['dummy_fft']+=count
            else:
                for _ in range(count):ifft(fft(u))
                ledger['ifft']+=count

def proposal(arm,u,alpha,scale,rng,floor,eps,ledger):
    ledger['proposals']+=1
    # All arms compute prior gradient and common reference phase move (charge actual work).
    g=score(u,alpha,ledger)
    direction,norm=aligned_direction(u,g,eps,floor)
    ledger['fft']+=2
    rand=hermitian_direction(rng)
    ref,_,_,_=phase_crop(u,scale*rand,ledger)
    image_norm=float(np.linalg.norm(ref-u))
    aux={'phase_direction_norm':float(norm),'image_reference_norm':image_norm,
         'phase_before':0.,'phase_after':0.,'imaginary_max':0.}
    if arm=='S0':
        # S0's extra conditional updates are made in the caller, not a new start.
        return u.copy(),aux
    if arm=='S1':
        d=rng.standard_normal(N);d/=max(np.linalg.norm(d),1e-14)
        return u+image_norm*d,aux
    if arm=='S4':
        noise=rng.standard_normal(N);noise/=max(np.linalg.norm(noise),1e-14)
        gnorm=np.linalg.norm(g)
        d=(.75*g/max(gnorm,1e-14)+.25*noise)
        d/=max(np.linalg.norm(d),1e-14)
        return u+image_norm*d,aux
    if arm=='S2':
        cand,b,a,im=phase_crop(u,scale*rand,ledger)
    else:
        if norm<1e-10: return u.copy(),aux
        # Fixed total angular norm, 75% direction + 25% stochastic component.
        vec=.75*direction+.25*rand
        vec/=max(np.linalg.norm(vec),1e-14)
        cand,b,a,im=phase_crop(u,scale*vec,ledger)
    aux.update(phase_before=b,phase_after=a,imaginary_max=im)
    return cand,aux

def label(u,y,rad,gate):
    d=np.linalg.norm(MU-u,axis=1)/np.sqrt(N)
    k=int(np.argmin(d));valid=float(d[k])<=rad and measurement_error(u,y)<=gate
    return (('A' if k<2 else 'B') if valid else 'U'),k if valid else -1,float(d[k])

def run_one(arm,initial,y,root_seed,config):
    u=initial.copy();ledger=new_ledger();trace=[];t0=time.perf_counter()
    # For Paired arm seeds, same noise stream per checkpoint but distinct namespace by arm.
    for j,alpha in enumerate(config['alphas']):
        # Small non-optimizing parent, identical in all arms.
        u=refine(u,alpha,config['parent_steps'],config['eta'],ledger)
        incumbent=u.copy()
        rng=np.random.default_rng(seed('tb0/proposal/'+str(root_seed)+'/'+str(j)))
        cand,aux=proposal(arm,u,alpha,config['scale'],rng,config['floor'],config['eps'],ledger)
        if arm=='S0':
            # Equal spending on useful extra native updates vs construction.
            cand=refine(cand,alpha,config['s0_extra_steps'],config['eta'],ledger)
        final=refine(cand,alpha,config['refine_steps'],config['eta'],ledger)
        E0=energy(incumbent,alpha,ledger)
        E1=energy(final,alpha,ledger)
        ledger['fft']+=1;ledger['likelihood']+=1
        gate=measurement_error(final,y)
        accepted=bool(E1 <= E0-config['delta_energy'] and gate<=config['measurement_gate'])
        u=final if accepted else incumbent
        trace.append(dict(accepted=accepted,E_gain=E0-E1,gate=gate,
                          displacement=float(np.linalg.norm(final-incumbent)),**aux))
        # Execute explicitly billed dummy evaluations so score/FFT/ifft are equal
        # across S0--S4 at each checkpoint. Useful native steps remain reported.
        budget_pad(u,alpha,ledger,{'score':9*(j+1),'fft':9*(j+1),'ifft':2*(j+1)})
    final_lab,final_k,dist=label(u,y,config['label_radius'],config['measurement_gate'])
    return dict(arm=arm,label=final_lab,final_k=final_k,distance=dist,
                meas_chi2=measurement_error(u,y),num_accepted=sum(t['accepted'] for t in trace),
                max_phase_crop=max(t['phase_after'] for t in trace),
                max_phase_imag=max(t['imaginary_max'] for t in trace),
                mean_prop_displacement=float(np.mean([t['displacement'] for t in trace])),
                mean_reference_displacement=float(np.mean([t['image_reference_norm'] for t in trace])),
                elapsed_s=time.perf_counter()-t0,ledger=ledger,trace=trace)

def seed(namespace):
    # stable SHA256 namespace (not Python salted hash), independent of process order
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8],'little')

def sample_start(split,cohort,i):
    rng=np.random.default_rng(seed('tb0/'+split+'/'+cohort+'/'+str(i)))
    if cohort=='b-start':k=2+int(rng.integers(2))
    elif cohort=='a-start':k=int(rng.integers(2))
    else:k=int(rng.choice(4,p=W))
    noise=np.random.default_rng(seed('tb0/noise/'+split+'/'+cohort+'/'+str(i))).normal(0,SIGMA,M)
    y=amp(MU[k])+noise
    initial=MU[k]+rng.normal(0,.01,N)
    return k,y,initial

def run(split,counts,cfg):
    results=[]
    for cohort,num in counts.items():
        for i in range(num):
            k,y,initial=sample_start(split,cohort,i)
            for arm in ARMS:
                r=run_one(arm,initial,y,split+'/'+cohort+'/'+str(i),cfg)
                r.update(cohort=cohort,index=i,ground_truth_k=k,initial_orbit='A' if k<2 else 'B')
                results.append(r)
    return results

def paired_ci(rows,reference,cohort,boot_seed,reps=2000):
    by={}
    for r in rows:
        if r['cohort']==cohort:by[(r['index'],r['arm'])]=r['label']
    inds=sorted({i for i,a in by})
    a=np.array([by[(i,'S3')]=='A' for i in inds],dtype=int)
    b=np.array([by[(i,reference)]=='A' for i in inds],dtype=int)
    d=a-b
    gen=np.random.default_rng(seed('tb0/stat/'+boot_seed+'/'+cohort+'/'+reference))
    samples=gen.choice(d,size=(reps,len(d)),replace=True).mean(axis=1)
    return {'delta':float(np.mean(d)),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'discordant_plus':int(np.sum(d==1)),'discordant_minus':int(np.sum(d==-1))}

def summary(rows,split):
    arms=[]
    for cohort in ('b-start','a-start','prior-start'):
        for arm in ARMS:
            x=[r for r in rows if r['cohort']==cohort and r['arm']==arm]
            if not x:continue
            counts={k:sum(r['label']==k for r in x) for k in ('A','B','U')}
            avg={key:float(np.mean([r[key] for r in x])) for key in ('meas_chi2','num_accepted','mean_prop_displacement','mean_reference_displacement','max_phase_crop','elapsed_s')}
            work={k:float(np.mean([r['ledger'][k] for r in x])) for k in new_ledger()}
            arms.append(dict(cohort=cohort,arm=arm,n=len(x),counts=counts,metrics=avg,work=work))
    paired={cohort:{arm:paired_ci(rows,arm,cohort,split) for arm in ('S1','S2','S4')} for cohort in ('b-start','a-start','prior-start') if (cohort,'S3') in {(r['cohort'],r['arm']) for r in rows}}
    return {'arms':arms,'paired':paired}

def save(rows,path):
    # Each row stores detailed counters and accepted/rejected trajectory, without tensor dumps.
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['cohort','idx','gt_k_diagnostic_only','arm','orbit','atom_index','distance','chi2','accepted','crop','imag','displacement','ref_disp','time_s','score','energy','fft','ifft','refinement','proposals','checkpoint_accepts'])
        for r in rows:
            wr.writerow([r['cohort'],r['index'],r['ground_truth_k'],r['arm'],r['label'],r['final_k'],round(r['distance'],6),round(r['meas_chi2'],6),r['num_accepted'],round(r['max_phase_crop'],6),round(r['max_phase_imag'],12),round(r['mean_prop_displacement'],6),round(r['mean_reference_displacement'],6),round(r['elapsed_s'],7),r['ledger']['score'],r['ledger']['energy'],r['ledger']['fft'],r['ledger']['ifft'],r['ledger']['refinement_steps'],r['ledger']['proposals'],''.join('1' if t['accepted'] else '0' for t in r['trace'])])

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=('preflight','dev','check'),required=True);p.add_argument('--out',required=True);args=p.parse_args()
    cfg=json.load(open(args.config));out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    proof=preflight();(out/'preflight.json').write_text(json.dumps(proof,indent=2))
    if args.stage=='preflight':return
    if args.stage=='check' and not cfg['frozen']:
        raise RuntimeError('CHECK requires explicit frozen configuration')
    counts=cfg['dev_counts'] if args.stage=='dev' else cfg['check_counts']
    rows=run(args.stage,counts,cfg)
    save(rows,out/(args.stage+'_rows.csv'))
    report=summary(rows,args.stage)
    (out/(args.stage+'_summary.json')).write_text(json.dumps(report,indent=2))
    print(args.stage,json.dumps([{'c':r['cohort'],'a':r['arm'],'counts':r['counts'],'work':r['work']} for r in report['arms']]))

if __name__=='__main__':main()
