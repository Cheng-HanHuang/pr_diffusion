#!/usr/bin/env python3
"""A8.3 OPTIONAL ImageNet FAST Good20 v2 E1_CLIP addendum, phase-1 ONLY.

Uses original K4 ImageNet D1 theory input, phase-1 CLIP G1/G2 decisions,
and immutable registered TH1_PREDICTIONS_v2.json (SHA pinned).
The baseline v2 predictions are NEVER rewritten or re-fit. The addendum
forecasts ONLY E1_CLIP at K 8,12,16 for the same 16-run completed panel.

Primary: Beta(1,1) class-specific orientation-success posteriors, with
the four observed old-run decisions conditioned on EXACTLY.
Sensitivity: resample phase-1 images as clusters to estimate the
per-class Beta rates. Conditional independence of future CLIP decisions
given class is a modeling assumption, not a demonstrated property.

The CLIP file schema can have different column headers; pass explicit
--*-col arguments if aliases do not match, and validate all 400 labels.
No code path opens XP returns or any future-run outcomes.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

from fit_th1_imagenet_v2 import load_imagenet
from fit_th1_prefix_v1 import sample_panel, rank_weights
from audit_a8_rank_race import fit_weights

BASE_SHA="8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55"
GRID=(8,12,16)
NMAX=16
VARIANTS=("race","S2","tail","race_tau1")
SOURCE_LABELS="GOPCI"
SEED=20261028

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def column(fieldnames, explicit, aliases, kind):
    if explicit is not None:
        if explicit not in fieldnames:
            raise ValueError("required %s column %r absent (actual: %s)"%
                             (kind,explicit,fieldnames))
        return explicit
    matches=[x for x in aliases if x in fieldnames]
    if len(matches)!=1:
        raise ValueError("cannot unambiguously identify %s column; "
                         "use an explicit --%s-col; actual: %s"%
                         (kind,kind.replace("_","-"),fieldnames))
    return matches[0]

def parse_flag(value,description):
    val=value.strip().lower()
    if val in ("1","true","yes"):return 1
    if val in ("0","false","no"):return 0
    raise ValueError("expected 0/1 for "+description+": "+repr(value))

def load_decisions(args, image_rows):
    digest=sha(args.clip)
    if digest!=args.clip_sha256:
        raise ValueError("phase-1 CLIP file SHA256 mismatch")
    with open(args.clip,newline="",encoding="utf8") as f:
        reader=csv.DictReader(f);hdr=reader.fieldnames
        if hdr is None:raise ValueError("empty CLIP CSV")
        cols={
          "image_id":column(hdr,args.image_col,
             ("image_id","image","image_name"),"image"),
          "run_index":column(hdr,args.run_col,
             ("run_index","run_idx","run_id","run"),"run"),
          "label":column(hdr,args.label_col,
             ("run_label","label","class_label","class"),"label"),
          "chosen_good":column(hdr,args.good_col,
             ("g1_chosen_good","g_tilde_good","chosen_orientation_good",
              "g1_good","g1_selected_good"),"good")
        }
        # Require recorded G1/G2 choice and margins for provenance QA.
        for key,aliases in (
            ("g1_choice",("g1_choice","G1_choice","clip_g1_choice")),
            ("g1_margin",("g1_margin","G1_margin","clip_g1_margin")),
            ("g2_choice",("g2_choice","G2_choice","clip_g2_choice")),
            ("g2_margin",("g2_margin","G2_margin","clip_g2_margin"))):
            override=getattr(args,key+"_col")
            cols[key]=column(hdr,override,aliases,key)
        records=list(reader)
    if len(records)!=400:
        raise ValueError("expected 400 phase-1 candidate decisions")
    images={r["image_id"]:r for r in image_rows}
    out={};raw_by_id={}
    for rec in records:
        image=rec[cols["image_id"]].strip()
        if image not in images:raise ValueError("unknown image "+image)
        run=int(rec[cols["run_index"]])
        if not 0<=run<4:raise ValueError("forbidden/post-prefix run index")
        key=(image,run)
        if key in out:raise ValueError("duplicate phase-1 decision "+str(key))
        label=rec[cols["label"]].strip()
        if label!=images[image]["original"][run]:
            raise ValueError("run class differs from frozen D1 input: "+str(key))
        good=parse_flag(rec[cols["chosen_good"]],"G1 chosen-good")
        for field in ("g1_margin","g2_margin"):
            v=float(rec[cols[field]])
            if not np.isfinite(v):
                raise ValueError("nonfinite CLIP margin "+str(key))
        for field in ("g1_choice","g2_choice"):
            if not rec[cols[field]].strip():
                raise ValueError("empty CLIP orientation choice "+str(key))
        out[key]=good
        raw_by_id.setdefault(image,[]).append((label,good))
    if len(out)!=400 or len(raw_by_id)!=100:
        raise ValueError("incomplete phase-1 decisions")
    for r in image_rows:
        rid=r["image_id"]
        if any((rid,j)not in out for j in range(4)):
            raise ValueError("missing phase-1 run "+rid)
        # Old simulated scores are in recorded ascending residual order;
        # recover exactly the corresponding CLIP decisions, NO resampling.
        # e0_rank_order was not stored by load_imagenet_v2's reduced row,
        # so compute the unique ranked sequence from source raw input below.
    return out,raw_by_id,digest,cols

def get_old_ranked_flags(phase1_csv,flags):
    with open(phase1_csv,newline="",encoding="utf8") as f:
        raw=list(csv.DictReader(f))
    ranked={}
    for r in raw:
        image=r["image_id"]
        order=[int(x)for x in r["e0_rank_order"].split(",")]
        if sorted(order)!=[0,1,2,3]:raise ValueError("invalid old permutation")
        ranked[image]=np.array([flags[(image,j)] for j in order],dtype=int)
    if len(ranked)!=100:raise ValueError("wrong number of ranked flag sets")
    return ranked

def counts_from_images(raw_by_id, image_ids):
    # Observed candidate-wise successes and failures, grouped by
    # the LABEL OF THE ORIGINAL CANDIDATE. The image is the cluster.
    successes=np.zeros(5,dtype=int);totals=np.zeros(5,dtype=int)
    for rid in image_ids:
        for label,flag in raw_by_id[rid]:
            idx=SOURCE_LABELS.index(label)
            totals[idx]+=1
            successes[idx]+=int(flag)
    return successes,totals

def sample_rates(successes,totals,rng):
    rates=np.zeros(5,dtype=float)
    for idx in range(5):
        # C/I remain structurally zero unless phase-1 G1 labels
        # explicitly show that rotation of a C/I reconstruction is GOOD.
        if idx>=3 and successes[idx]==0:
            rates[idx]=0.
        else:
            rates[idx]=rng.beta(1+successes[idx],
                                 1+totals[idx]-successes[idx])
    return rates

def effective_class(c,variant):
    if variant!="tail":return int(c)
    if c==5:return 4            # I
    if c in (3,4):return 3       # C_base or C_extreme
    return int(c)

def run_panel(rows,alpha,params,variant,old_flags,rates,rng,rankw):
    outcomes=np.zeros((len(GRID),2),dtype=float)
    for row in rows:
        labels,scores=sample_panel(row,alpha,params,variant,NMAX,rng)
        correct=np.zeros(NMAX,dtype=float)
        correct[:4]=old_flags[row["image_id"]]
        for j in range(4,NMAX):
            cl=effective_class(labels[j],variant)
            correct[j]=rng.random()<rates[cl]
        order=np.argsort(scores,kind="stable")
        selected=correct[order]
        # Literal policy: a single candidate's chosen orientation,
        # not a second pass ranking on another score.
        orbit=(labels<=2).astype(float)
        for ki,k in enumerate(GRID):
            w=rankw[k]
            outcomes[ki,0]+=np.dot(w,selected)
            outcomes[ki,1]+=np.dot(w,orbit[order])
    return outcomes/len(rows)

def run(rows,old_flags,raw_by_id,base,point_reps,bootstrap,boot_reps):
    alpha=np.asarray(base["alpha"],dtype=float)
    frozen_primary=base["rank_weights_primary"]
    frozen_sensitivity=base["rank_weights_sensitivity"]
    rank_params={
      "race":frozen_primary,"S2":frozen_primary,
      "tail":frozen_primary,"race_tau1":frozen_sensitivity}
    registered_oracle={}
    for p in base["predictions"]:
        if p["policy"]=="O_orbit":
            registered_oracle[p["model_variant"],p["K"]]=p["point"]
    if len(registered_oracle)!=len(VARIANTS)*len(GRID):
        raise ValueError("incomplete registered v2 orbit-oracle predictions")
    stats,totals=counts_from_images(raw_by_id,[r["image_id"]for r in rows])
    if not np.array_equal(totals,[100,67,3,188,42]):
        raise ValueError("phase-1 class totals differ from registered v2")
    rng=np.random.default_rng(SEED)
    rankw={k:rank_weights(NMAX,k)for k in GRID}
    result=[];draw_audit=[]
    for variant in VARIANTS:
        params=rank_params[variant]
        main=np.zeros((point_reps,len(GRID),2))
        for b in range(point_reps):
            rate=sample_rates(stats,totals,rng)
            main[b]=run_panel(rows,alpha,params,variant,
                               old_flags,rate,rng,rankw)
        empirical_mean=main.mean(axis=0)
        targets=np.array([registered_oracle[variant,k]for k in GRID])
        # Paired control variate: exactly preserve the registered
        # O_orbit mean, and transfer the SAME delta to E1. This
        # preserves E1<=O_orbit draw by draw, and does not alter v2.
        shift=targets-empirical_mean[:,1]
        point=empirical_mean[:,0]+shift
        if np.any(point>targets+1e-10):
            raise ValueError("E1 mean exceeds registered orbit oracle")
        boot_mean=np.empty((bootstrap,len(GRID)))
        ids=[r["image_id"]for r in rows]
        for b in range(bootstrap):
            # CLUSTER decisions by IMAGE, but retain the original
            # panel image cohort for forecasting (A8.3 sensitivity).
            sampled_ids=[ids[int(i)]for i in
                rng.integers(0,len(ids),len(ids))]
            a,n=counts_from_images(raw_by_id,sampled_ids)
            vals=np.zeros((boot_reps,len(GRID)))
            for t in range(boot_reps):
                rates=sample_rates(a,n,rng)
                vals[t]=run_panel(rows,alpha,params,variant,
                                old_flags,rates,rng,rankw)[:,0]
            boot_mean[b]=vals.mean(axis=0)
        # Primary predictive interval: original beta uncertainty and
        # future-panel noise; sensitivity adds image-cluster variation
        # of the CLIP class rates only. No refitting frozen v2.
        primary=main[:, :, 0]+shift[None,:]
        indices=rng.integers(0,bootstrap,size=point_reps)
        sensitivity=primary+(boot_mean[indices]-boot_mean.mean(axis=0))
        for decision_model,values in (("beta_class",primary),
                                      ("beta_image_cluster",sensitivity)):
            lo,hi=np.quantile(values,(.025,.975),axis=0)
            for ki,k in enumerate(GRID):
                val=float(point[ki])
                rec={
                   "family":"FAST","dataset":"ImageNet",
                   "metric":"good20","cohort":"D1_fixed100",
                   "model_variant":variant,
                   "decision_model":decision_model,
                   "policy":"E1_CLIP","K":k,"panel":"primary",
                   "panel_n_max":16,"n_images":100,
                   "point":val,
                   "lo95":float(np.clip(min(lo[ki],val),0,1)),
                   "hi95":float(np.clip(max(hi[ki],val),0,1)),
                   "O_orbit_registered_point":float(targets[ki])
                }
                if not 0<=rec["point"]<=targets[ki]+1e-10:
                    raise ValueError("invalid E1 point")
                result.append(rec)
        # Store only aggregate checks; no unregistered future labels
        # are emitted. The underlying common-shift simulation checked
        # the pathwise ordering BEFORE and AFTER the adjustment.
        assert np.all(main[:,:,0]<=main[:,:,1]+1e-12)
        assert np.all(main[:,:,0]+shift<=main[:,:,1]+shift+1e-12)
        draw_audit.append({"variant":variant,"draws":point_reps,
                           "all_E1_le_orbit":True})
    stats_out={SOURCE_LABELS[i]:{"successes":int(stats[i]),
                   "total":int(totals[i]),
                   "beta_alpha":int(stats[i]+1),
                   "beta_beta":int(totals[i]-stats[i]+1),
                   "rate_override_zero":bool(i>=3 and stats[i]==0)}
               for i in range(5)}
    return result,stats_out,draw_audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,
                   help="exact authorized D1 phase-1 100-image CSV")
    p.add_argument("--base-v2",required=True,
                   help="registered immutable TH1_PREDICTIONS_v2.json")
    p.add_argument("--clip",required=True,
                   help="executor's phase-1-only 400-row CLIP CSV")
    p.add_argument("--clip-sha256",required=True)
    p.add_argument("--output",required=True)
    for stem in ("image","run","label","good","g1_choice",
                 "g1_margin","g2_choice","g2_margin"):
        p.add_argument("--"+stem.replace("_","-")+"-col")
    p.add_argument("--point-reps",type=int,default=350)
    p.add_argument("--bootstrap",type=int,default=60)
    p.add_argument("--boot-reps",type=int,default=3)
    args=p.parse_args()
    if len(args.clip_sha256)!=64 or any(
       c not in "0123456789abcdef" for c in args.clip_sha256):
        raise ValueError("require exact lowercase SHA256 of CLIP input")
    if sha(args.base_v2)!=BASE_SHA:
        raise ValueError("registered base v2 hash mismatch")
    with open(args.base_v2,encoding="utf8")as f:base=json.load(f)
    if base["version"]!="TH1_PREDICTIONS_v2":
        raise ValueError("v2 source not registered version")
    rows,phase_sha=load_imagenet(args.input)
    flags,raw_by_id,clip_sha,cols=load_decisions(args,rows)
    ranked=get_old_ranked_flags(args.input,flags)
    forecasts,fit,audits=run(
       rows,ranked,raw_by_id,base,
       args.point_reps,args.bootstrap,args.boot_reps)
    obj={
      "version":"DRAFT_V2_E1_UNREGISTERED",
      "dataset":"ImageNet","family":"FAST",
      "metric":"good20","n_images":100,
      "K_grid":list(GRID),"panel_n_max":NMAX,
      "seed":SEED,"status":"DRAFT_NOT_FROZEN",
      "interval_type":"predictive_realized_panel",
      "base_v2_sha256":BASE_SHA,
      "phase1_sha256":phase_sha,
      "phase1_clip_sha256":clip_sha,
      "decision_columns":cols,
      "decision_estimation":"Beta(1,1) class-conditional posterior,"
          " future runs iid conditional class and shared class rate;"
          " known four-run G1 outcomes conditioned upon exactly",
      "cluster_sensitivity":"beta_image_cluster: resample images as "
          "clusters for the class-conditional decision posterior,"
          " while forecasting the original 100 image IDs",
      "frozen_model":"registered v2 alpha/rank parameters copied"
          " unchanged; no v2 point or band overwritten",
      "policy_definition":"single E0-selected candidate, apply G1"
          " orientation decision to that candidate, score g_tilde",
      "forecast_comparison_rule":"primary beta_class E1 coverage at"
          " 8/12/16; report beta_image_cluster separately;"
          " no posthoc swapping or refitting v2",
      "clip_class_posterior_counts":fit,
      "simulation_draw_checks":audits,
      "predictions":forecasts,
      "provenance":{"phase2_outcomes_seen":False,
                    "xp_returns_accessed":False,
                    "source_limited_to_runs_0_3":True}
    }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError("do not overwrite draft")
    out.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    print("TH1_V2_E1_ADDENDUM_DRAFT",len(forecasts),
          "CLIP",clip_sha,"OUTPUT",str(out))
if __name__=="__main__":
    main()
