#!/usr/bin/env python3
"""A12 v4, UNREGISTERED: ImageNet FAST Good20, 16 observed runs -> 32/64.

This file MUST NOT be used to unseal or compare XP-D4 outcomes. Only
previously permitted run-index 0--15 inputs and locked K4 provenance.
No structural-zero spike (pre-input lock TH_A12_V4_PREINPUT_MODEL_LOCK).
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from audit_a8_rank_race import fit_weights
from fit_th1_dirichlet import fit as fit_alpha
from fit_th1_prefix_v1 import sample_panel,rank_weights,pass_at_k,exact_prefix_oracles

PREFIX4_SHA="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"
PANELS={"primary":(32,(16,24,32)),"secondary":(64,(32,48,64))}
VARS=("race","S2","tail","race_tau1")
LABELS="GOPCI"
SEED=20261029

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def strict_sha(path,expected):
    if len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("SHA-256 must be an exact lowercase 64-hex digest")
    got=sha(path)
    if got!=expected:raise ValueError("SHA-256 differs for "+str(path)+": "+got)
    return got

def read_csv(path):
    with open(path,newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))

def csv_column(cols,preferred,aliases,kind):
    if preferred:
        if preferred not in cols:raise ValueError(kind+" override absent")
        return preferred
    matches=[c for c in aliases if c in cols]
    if len(matches)!=1:raise ValueError("ambiguous "+kind+" column: "+str(cols))
    return matches[0]

def yes_no(raw):
    v=str(raw).strip().lower()
    if v in ("1","true","yes"):return 1
    if v in ("0","false","no"):return 0
    raise ValueError("non-Boolean Good20 CLIP flag: "+repr(raw))

def load_pilot(args):
    strict_sha(args.prefix4,PREFIX4_SHA)
    strict_sha(args.pilot,args.pilot_sha256)
    old=read_csv(args.prefix4)
    new=read_csv(args.pilot)
    if len(old)!=100 or len(new)!=100:raise ValueError("expected exactly 100 original images")
    old_by_id={r["image_id"]:r for r in old}
    if len(old_by_id)!=100:raise ValueError("old duplicate images")
    rows=[];raw={}
    for rec in new:
        uid=rec["image_id"]
        if uid in raw or uid not in old_by_id:raise ValueError("new/unknown image "+uid)
        if rec["family"]!="FAST":raise ValueError("wrong family")
        if int(rec["n_runs"])!=16:raise ValueError("out of permitted 16-run prefix")
        orig=rec["run_labels"].strip()
        ranked=rec["e0_rank_labels"].strip()
        order=[int(i)for i in rec["e0_rank_order"].split(",")]
        if len(orig)!=16 or len(ranked)!=16 or sorted(order)!=list(range(16)):
            raise ValueError("invalid 16-label/rank source "+uid)
        if any(c not in LABELS for c in orig) or ''.join(orig[i]for i in order)!=ranked:
            raise ValueError("rank label permutation mismatch "+uid)
        orig4=old_by_id[uid]
        if orig[:4]!=orig4["run_labels"]:
            raise ValueError("historical 0..3 labels changed "+uid)
        perm4=[i for i in order if i<4]
        if perm4!=[int(i)for i in orig4["e0_rank_order"].split(",")]:
            raise ValueError("historical original-four residual ordering changed "+uid)
        if "rho_signed_by_run" in rec and rec["rho_signed_by_run"].strip():
            rho=[float(x)for x in rec["rho_signed_by_run"].split(";")]
            if len(rho)!=16 or not all(np.isfinite(rho)) or any(
                rho[order[i]]>rho[order[i+1]]+1e-10 for i in range(15)):
                raise ValueError("pilot rho does not generate rank order "+uid)
        counts=np.asarray([int(rec[k])for k in (
            "n_GOOD","n_ORBIT_global","n_ORBIT_partial",
            "n_CONSISTENT","n_INCONSISTENT")],dtype=int)
        if list(counts)!=[orig.count(c)for c in LABELS]:
            raise ValueError("16-run class-count mismatch "+uid)
        rows.append({"image_id":uid,"source":rec.get("source","D1_16"),
             "family":"FAST","cohort":"IMAGENET", "n":16,
             "counts":counts,"ranked":ranked,"original":orig})
        raw[uid]=rec
    if len(raw)!=100 or set(raw)!=set(old_by_id):raise ValueError("incomplete original image set")
    return rows,raw

def load_clip(args,rows,raw):
    strict_sha(args.clip,args.clip_sha256)
    recs=read_csv(args.clip)
    if len(recs)!=1600:raise ValueError("need exactly 1600 RUNS 0--15, not future outcomes")
    names=list(recs[0])
    col={
      "image":csv_column(names,args.image_col,("image_id","image","image_name"),"image"),
      "run":csv_column(names,args.run_col,("run_index","run_idx","run_id","run"),"run"),
      "label":csv_column(names,args.label_col,("run_label","label","class_label","class"),"label"),
      "good":csv_column(names,args.good_col,("g1_chosen_good","g_tilde_good","chosen_orientation_good","g1_good","g1_selected_good"),"good")
    }
    ids={r["image_id"]:r for r in rows}; flags={}
    for r in recs:
        uid=r[col["image"]].strip();j=int(r[col["run"]]);label=r[col["label"]].strip()
        if uid not in ids or not 0<=j<16:raise ValueError("unknown image/post-prefix index")
        key=(uid,j)
        if key in flags:raise ValueError("duplicate run G1 decision")
        if label!=ids[uid]["original"][j]:raise ValueError("G1 class differs from image source")
        outcome=yes_no(r[col["good"]])
        if label in "CI" and outcome:raise ValueError("C/I marked Good20 after G1; audit GT-label logic")
        flags[key]=outcome
    if len(flags)!=1600:raise ValueError("incomplete G1 decisions")
    ranked={}
    for uid,rec in raw.items():
        order=[int(i)for i in rec["e0_rank_order"].split(",")]
        ranked[uid]=np.asarray([flags[(uid,i)]for i in order],dtype=int)
    successes=np.zeros(5,dtype=int);totals=np.zeros(5,dtype=int)
    for row in rows:
        uid=row["image_id"]
        for j,label in enumerate(row["original"]):
            idx=LABELS.index(label);totals[idx]+=1;successes[idx]+=flags[(uid,j)]
    if sum(totals)!=1600 or any(successes[3:]):raise ValueError("bad class-conditional orientation totals")
    return ranked,successes,totals,col

def beta_rates(successes,totals,rng):
    p=np.zeros(5,dtype=float)
    for j in range(3):p[j]=rng.beta(1+successes[j],1+totals[j]-successes[j])
    return p

def panel_draw(rows,alpha,fit,variant,nmax,ks,ranked_old,rate,rng):
    rw={k:rank_weights(nmax,k)for k in ks}
    sums=np.zeros((len(ks),4),dtype=float)
    for row in rows:
        labels,scores=sample_panel(row,alpha,fit,variant,nmax,rng)
        old=ranked_old[row["image_id"]]
        if len(old)!=16:raise ValueError("CLIP old prefix should be 16 long")
        chosen=np.zeros(nmax,dtype=int);chosen[:16]=old
        for j in range(16,nmax):
            cl=int(labels[j]);base=3 if cl in (3,4) and variant=="tail" else (
                4 if cl==5 else cl)
            if not 0<=base<5:raise ValueError("bad tail-expanded class")
            chosen[j]=int(rng.random()<rate[base])
        order=np.argsort(scores,kind="stable")
        good=(labels[order]==0).astype(float)
        orient=chosen[order].astype(float)
        ng=int(np.count_nonzero(labels==0));no=int(np.count_nonzero(labels<=2))
        for ki,k in enumerate(ks):
            w=rw[k]
            sums[ki,0]+=np.dot(w,good)
            sums[ki,1]+=np.dot(w,orient)
            sums[ki,2]+=pass_at_k(nmax,ng,k)
            sums[ki,3]+=pass_at_k(nmax,no,k)
    return sums/len(rows)

def fit_models(rows):
    alpha=fit_alpha(np.stack([r["counts"]for r in rows]))
    w=fit_weights(rows,tau=2.5)
    w1=fit_weights(rows,tau=1.)
    return alpha,w,w1

def run(rows,old_flags,successes,totals,point_reps,bootstrap,boot_reps):
    rng=np.random.default_rng(SEED)
    alpha,fit,fit1=fit_models(rows)
    vals=[];anatomy=[];checks=[]
    ids=[r["image_id"]for r in rows]
    for panel,(nmax,ks)in PANELS.items():
        exact=np.mean([exact_prefix_oracles(row,alpha,ks,nmax)
                       for row in rows],axis=0)
        for variant in VARS:
            w=fit1 if variant=="race_tau1" else fit
            main=np.empty((point_reps,len(ks),4),dtype=float)
            for b in range(point_reps):
                p=beta_rates(successes,totals,rng)
                main[b]=panel_draw(rows,alpha,w,variant,nmax,ks,old_flags,p,rng)
            mc=main.mean(axis=0)
            correction=exact-mc[:,2:]
            if np.max(np.abs(correction))>0.01:
                raise ValueError("A9.5 wrong oracle anchor or insufficient MC: "+str(correction))
            point=mc.copy()
            point[:,0]+=correction[:,0]
            point[:,1]+=correction[:,1]
            point[:,2:]=exact
            if np.any(point[:,:2]<-1e-10) or np.any(point[:,0]>point[:,2]+1e-10) or np.any(point[:,1]>point[:,3]+1e-10):
                raise ValueError("negative or oracle-exceeding selector point")
            if np.any(point[:,2]>point[:,3]+1e-10):raise ValueError("raw oracle exceeds orbit")
            primary=main.copy()
            primary[:,:,0]+=correction[None,:,0]
            primary[:,:,1]+=correction[None,:,1]
            primary[:,:,2:]+=correction[None,:,:]
            # Image-cluster bootstrap refits original-prefix parameters ONLY.
            # Additional realized-panel draws produce a predictive sensitivity
            # distribution; they are not used to retune the point forecast.
            boot=[]
            for b in range(bootstrap):
                idx=rng.integers(0,len(rows),len(rows))
                sample=[rows[int(j)]for j in idx]
                a_b,w_b,w1_b=fit_models(sample)
                wb=w1_b if variant=="race_tau1" else w_b
                succ_b=np.zeros(5,dtype=int);tot_b=np.zeros(5,dtype=int)
                for j in idx:
                    r=rows[int(j)];uid=r["image_id"]
                    # Existing 16-run G1 flags in ascending residual order.
                    for label,flag in zip(r["ranked"],old_flags[uid]):
                        cl=LABELS.index(label);tot_b[cl]+=1;succ_b[cl]+=flag
                for t in range(boot_reps):
                    p_b=beta_rates(succ_b,tot_b,rng)
                    boot.append(panel_draw(rows,a_b,wb,variant,nmax,ks,
                                           old_flags,p_b,rng))
            boot=np.asarray(boot)
            # Parametric prediction: draw from model fitted to original cohort,
            # plus image-cluster fit bootstrap (v2-style sensitivity). Each
            # source draw is a coherent completed panel before recentering.
            boot[:,:,0]+=correction[None,:,0]
            boot[:,:,1]+=correction[None,:,1]
            boot[:,:,2:]+=correction[None,:,:]
            # Unlike separately adjusted policy columns, shifting E and its
            # own oracle by the same delta preserves pathwise E<=O.
            if (np.any(primary[:,:,0]>primary[:,:,2]+1e-10) or
                np.any(primary[:,:,1]>primary[:,:,3]+1e-10) or
                np.any(boot[:,:,0]>boot[:,:,2]+1e-10) or
                np.any(boot[:,:,1]>boot[:,:,3]+1e-10)):
                raise ValueError("A9.5 E exceeds paired oracle in draw")
            intervals={}
            for name,draws in (("primary",primary),
                               ("image_cluster",np.concatenate((primary,boot),axis=0))):
                intervals[name]=np.quantile(draws,[0.025,0.975],axis=0)
            for ki,k in enumerate(ks):
                for policy,col in (("E0",0),("E1_CLIP",1),
                                   ("O_raw",2),("O_orbit",3)):
                    lo,hi=intervals["primary"][:,ki,col]
                    center=float(point[ki,col])
                    vals.append({"dataset":"ImageNet","family":"FAST",
                      "cohort":"D1_fixed100","metric":"good20",
                      "panel":panel,"panel_n_max":nmax,"K":k,
                      "model_variant":variant,"policy":policy,
                      "decision_model":"beta_class" if policy=="E1_CLIP" else None,
                      "n_images":100,"point":center,
                      "lo95":float(max(0,min(center,lo))),
                      "hi95":float(min(1,max(center,hi)))})
                    if policy=="E1_CLIP":
                        blo,bhi=intervals["image_cluster"][:,ki,col]
                        vals.append({**vals[-1],
                          "decision_model":"beta_image_cluster",
                          "lo95":float(max(0,min(center,blo))),
                          "hi95":float(min(1,max(center,bhi)))})
                anatomy.append({"panel":panel,"panel_n_max":nmax,
                   "K":k,"model_variant":variant,
                   "generation_failure":float(1-point[ki,3]),
                   "selection_loss_E0":float(point[ki,2]-point[ki,0]),
                   "selection_loss_E1":float(point[ki,3]-point[ki,1])})
            checks.append({"panel":panel,"variant":variant,
               "K":list(ks),"mc_oracle_raw":mc[:,2].tolist(),
               "mc_oracle_orbit":mc[:,3].tolist(),
               "exact_oracle_raw":exact[:,0].tolist(),
               "exact_oracle_orbit":exact[:,1].tolist(),
               "shift_raw":correction[:,0].tolist(),
               "shift_orbit":correction[:,1].tolist(),
               "max_abs_shift":float(np.max(np.abs(correction)))})
    return vals,anatomy,checks,alpha,fit,fit1

def main():
    ap=argparse.ArgumentParser()
    for flag in ("prefix4","pilot","pilot_sha256","clip","clip_sha256","output"):
        ap.add_argument("--"+flag.replace("_","-"),required=True)
    for flag in ("image_col","run_col","label_col","good_col"):
        ap.add_argument("--"+flag.replace("_","-"))
    ap.add_argument("--point-reps",type=int,default=250)
    ap.add_argument("--bootstrap",type=int,default=60)
    ap.add_argument("--boot-reps",type=int,default=3)
    args=ap.parse_args()
    if Path(args.output).exists():raise FileExistsError("cannot overwrite forecast draft")
    rows,raw=load_pilot(args)
    flags,succ,tot,col=load_clip(args,rows,raw)
    forecast,anatomy,anchor,alpha,w,w1=run(rows,flags,succ,tot,
            args.point_reps,args.bootstrap,args.boot_reps)
    posterior={c:{"successes":int(succ[i]),"total":int(tot[i]),
                  "alpha":int(1+succ[i]),"beta":int(1+tot[i]-succ[i])}
               for i,c in enumerate(LABELS)}
    obj={"version":"DRAFT_V4_UNREGISTERED","status":"UNREGISTERED",
       "dataset":"ImageNet","metric":"good20","family":"FAST",
       "source_pilot16_sha256":sha(args.pilot),"source_clip16_sha256":sha(args.clip),
       "source_prefix4_sha256":PREFIX4_SHA,"code_model_lock":"TH_A12_V4_PREINPUT_MODEL_LOCK_20261029.md",
       "n_images":100,"n_observed_runs":16,"panels":{k:{"nmax":v[0],"K":list(v[1])}for k,v in PANELS.items()},
       "point_reps":args.point_reps,"bootstrap":args.bootstrap,
       "boot_reps":args.boot_reps,"seed":SEED,"structural_zero_component":False,
       "interval_type":"predictive_realized_panel",
       "alpha":np.asarray(alpha,dtype=float).tolist(),
       "weights_primary":{k:float(w[k])for k in ("w_O","w_C")},
       "weights_tau1":{k:float(w1[k])for k in ("w_O","w_C")},
       "g1_class_posterior":posterior,"input_columns":col,
       "predictions":forecast,"anatomy_curves":anatomy,
       "oracle_anchor_checks":anchor,
       "comparison_rule":"primary n32 E0+E1 Beta class 6 coverage points; standardized residual tie-break; disclose sensitivities and n64 secondary",
       "provenance":{"allowed_run_indices":list(range(16)),
          "held_XP_D4_outcomes_seen":False,"xp_returns_accessed":False}}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print("TH1_IMAGENET_V4_DRAFT",len(forecast),len(anatomy),len(anchor),str(target))
if __name__=="__main__":main()
