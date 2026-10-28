"""A9.2(b) composition-conditional selection test (planner, exploratory, post hoc).
For each image's realized 64-run panel, each frozen rank law predicts E0(K) given the
realized classes; compared with the observed E0(K) from actual residual ranks.
Frozen parameters only (v1 fit_diagnostics; race_tau1 recomputed with the theory code)."""
import sys,csv,json,math
import numpy as np
theory_dir,a8,strings,v1path,cmp_path=sys.argv[1:6]
sys.path.insert(0,theory_dir)
from audit_a8_rank_race import read_a8,fit_weights
rows,_=read_a8(a8)
fr=[r for r in rows if r["family"]=="FAST"]
w_main=fit_weights(fr); w_tau1=fit_weights(fr,tau=1.0)
v1=json.load(open(v1path))
fd=[d for d in v1["fit_diagnostics"] if d["family"]=="FAST"][0]["rank_weights"]
assert abs(fd["w_O"]-w_main["w_O"])<1e-9 and abs(fd["w_C"]-w_main["w_C"])<1e-9
baseline,extreme,target=1e-6,1000.,5/205
eps=(target-baseline/(1+baseline))/(extreme/(1+extreme)-baseline/(1+baseline))
LAWS={"race":(1,w_main["w_O"],w_main["w_C"],0.),"race_tau1":(1,w_tau1["w_O"],w_tau1["w_C"],0.),
      "S2":(1,1,1,0.),"tail":"tail"}
Ks=(8,12,16,24,32,48,64)
def rank_weights(n,k):
    den=math.comb(n,k)
    return np.array([math.comb(n-r-1,k-1)/den if n-r>=k else 0. for r in range(n)])
RW={k:rank_weights(64,k) for k in Ks}
imgs=list(csv.DictReader(open(strings)))
def cls(l): return {"G":0,"O_g":1,"O_p":1,"C":2,"I":3}[l]
def pred_E0(counts,K,law):
    g,o,c,i=counts; n=g+o+c+i
    tot=math.comb(n,K); s=0.
    for a in range(0,min(g,K)+1):
        if a==0: continue
        for b in range(0,min(o,K-a)+1):
            for cc in range(0,min(c,K-a-b)+1):
                d=K-a-b-cc
                if d>i: continue
                p=math.comb(g,a)*math.comb(o,b)*math.comb(c,cc)*math.comb(i,d)/tot
                if law=="tail":
                    s+=p*((1-eps)**cc)*a/(a+LAWS["race"][1]*b+baseline*cc)
                else:
                    _,wo,wc,_=LAWS[law]; s+=p*a/(a+wo*b+wc*cc)
    return s
def obs_E0(labels,rho,K):
    order=sorted(range(len(rho)),key=lambda j:(rho[j],j))
    good=np.array([1. if labels[j]=="G" else 0. for j in order])
    return float(RW[K]@good)
res={}
for co in ("B22","DEV80","POOLED"):
    sel=[r for r in imgs if co=="POOLED" or r["cohort"]==co]
    for K in Ks:
        obs=[];pr={l:[] for l in LAWS}
        oraw=[]
        for r in sel:
            labels=r["labels_by_run"].split(";"); rho=[float(x) for x in r["rho_signed_by_run"].split(";")]
            cnt=[sum(1 for l in labels if cls(l)==k) for k in range(4)]
            obs.append(obs_E0(labels,rho,K))
            g=cnt[0]; oraw.append(1-(math.comb(64-g,K)/math.comb(64,K) if 64-g>=K else 0.))
            for l in LAWS: pr[l].append(pred_E0(cnt,K,l))
        res[(co,K)]=(np.mean(oraw),np.mean(obs),{l:np.mean(v) for l,v in pr.items()},len(sel))
# Monte Carlo predictive intervals for the race given realized labels (pooled and cohort)
rng=np.random.default_rng(20261028)
def sim(sel,law,reps=400):
    out=np.zeros((reps,len(Ks)))
    W=LAWS[law]
    for b in range(reps):
        acc=np.zeros(len(Ks))
        for r in sel:
            labels=r["labels_by_run"].split(";")
            c=np.array([cls(l) for l in labels])
            if law=="tail":
                w=np.where(c==0,1.,np.where(c==1,LAWS["race"][1],np.where(c==2,np.where(rng.random(len(c))<eps,extreme,baseline),0.)))
            else:
                w=np.array([W[k] for k in c],dtype=float)
            sc=np.where(w>0,rng.exponential(1/np.where(w>0,w,1.)),np.inf)
            order=np.lexsort((np.arange(len(sc)),sc))
            good=(c[order]==0).astype(float)
            acc+=np.array([RW[K]@good for K in Ks])
        out[b]=acc/len(sel)
    return np.quantile(out,[.025,.975],axis=0)
print("weights race",round(w_main["w_O"],4),round(w_main["w_C"],4)," tau1",round(w_tau1["w_O"],4),round(w_tau1["w_C"],4)," eps",round(eps,5))
for co in ("B22","DEV80","POOLED"):
    sel=[r for r in imgs if co=="POOLED" or r["cohort"]==co]
    qi={l:sim(sel,l) for l in ("race","S2","tail")}
    print(f"\n== {co}  (n={len(sel)})  realized-label conditional E0; [95% race-sim interval]")
    print(" K  O_raw   obsE0  | race  [lo,hi]       | tau1  | S2   [lo,hi]        | tail [lo,hi]")
    for ki,K in enumerate(Ks):
        o,ob,p,_=res[(co,K)]
        print(f"{K:3d} {o:.4f} {ob:.4f} | {p['race']:.4f} [{qi['race'][0,ki]:.4f},{qi['race'][1,ki]:.4f}] | {p['race_tau1']:.4f} | {p['S2']:.4f} [{qi['S2'][0,ki]:.4f},{qi['S2'][1,ki]:.4f}] | {p['tail']:.4f} [{qi['tail'][0,ki]:.4f},{qi['tail'][1,ki]:.4f}]")
