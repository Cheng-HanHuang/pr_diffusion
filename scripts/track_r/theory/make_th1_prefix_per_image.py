#!/usr/bin/env python3
"""A6 companion fixed-FFHQ per-image forecasts for lineage-filtered reanalysis.

Produces one CSV row per original image/family/panel/variant/K with
E0, E1_CLIP, O_raw, O_orbit and exact posterior oracle means.

The companion is NOT the headline v1 preregistered primary forecast. It
must use exactly the same original A8 input, family rank weights,
cohort-specific Dirichlet fit, prefix conditioning and panel definitions
as fit_th1_prefix_v1.py. It does not consume any future-run results.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from audit_a8_rank_race import read_a8,fit_weights,FAMILIES
from fit_th1_dirichlet import fit as fit_alpha
from fit_th1_prefix_v1 import (
    SEED,VARIANTS,PANELS,simulate,exact_prefix_oracles)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a8",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--replicates",type=int,default=192)
    a=p.parse_args()
    rows,sha=read_a8(a.a8)
    rng=np.random.default_rng(SEED+1)
    out=[]
    for fam in FAMILIES:
        fr=[r for r in rows if r["family"]==fam]
        w=fit_weights(fr)
        w1=fit_weights(fr,tau=1.0)
        for co in ("B22","DEV80"):
            cr=[r for r in fr if r["cohort"]==co]
            C=np.array([r["counts"]for r in cr],dtype=float)
            alpha=fit_alpha(C)
            for panelname,(nmax,Kgrid)in PANELS[fam].items():
                for r in cr:
                    oracle=exact_prefix_oracles(r,alpha,Kgrid,nmax)
                    for variant in VARIANTS:
                        weights=w1 if variant=="race_tau1" else w
                        vals=simulate([r],alpha,weights,variant,
                                      Kgrid,nmax,rng,a.replicates).mean(axis=0)
                        # Control variate to identical exact per-image oracles.
                        vals[:,0]+=oracle[:,0]-vals[:,2]
                        vals[:,1]+=oracle[:,1]-vals[:,3]
                        vals[:,2:]=oracle
                        for ki,k in enumerate(Kgrid):
                            if fam=="DAPS" and co!="B22" and k==4:
                                continue
                            out.append({"image_id":r["image_id"],
                              "family":fam,"cohort":co,
                              "panel":panelname,"panel_n_max":nmax,
                              "model_variant":variant,"K":k,
                              "E0":float(vals[ki,0]),
                              "E1_CLIP":float(vals[ki,1]),
                              "O_raw":float(vals[ki,2]),
                              "O_orbit":float(vals[ki,3]),
                              "posterior_source_sha256":sha,
                              "replicates":a.replicates})
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as f:
        keys=list(out[0])
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(out)
    print("TH1_A6_PER_IMAGE_COMPANION_DRAFT",len(out),path)

if __name__=="__main__":main()
