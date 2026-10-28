#!/usr/bin/env python3
"""Synthetic-only check of ImageNet v2 E1 class handling and oracle bound.

No external datasets, phase-2 results, or CLIP ground truth are used.
Run in the same Python environment as the model.
"""
import numpy as np
from fit_th1_imagenet_v2_e1 import (
  sample_rates,counts_from_images,run_panel,GRID,NMAX)
from fit_th1_prefix_v1 import rank_weights

def main():
    rng=np.random.default_rng(20261028)
    rows=[
      {"image_id":"A","n":4,"counts":np.array([1,1,0,1,1]),
       "ranked":"GOCI","original":"GOCI"},
      {"image_id":"B","n":4,"counts":np.array([1,0,1,1,1]),
       "ranked":"CIGP","original":"CIGP"},
      {"image_id":"C","n":4,"counts":np.array([2,0,0,2,0]),
       "ranked":"CGCG","original":"CGCG"}
    ]
    old={"A":np.array([1,0,0,0]),
         "B":np.array([0,0,1,1]),
         "C":np.array([0,1,0,1])}
    grouped={
       "A":[("G",1),("O",0),("C",0),("I",0)],
       "B":[("C",0),("I",0),("G",1),("P",1)],
       "C":[("C",0),("G",1),("C",0),("G",1)]
    }
    good,totals=counts_from_images(grouped,["A","B","C"])
    assert np.array_equal(good,[4,0,1,0,0])
    assert np.array_equal(totals,[4,1,1,4,2])
    fit={"w_O":0.77,"w_C":0.18}
    alpha=np.array([.38,.30,.006,.66,.14])
    rankw={k:rank_weights(NMAX,k)for k in GRID}
    for variant in ("race","S2","tail","race_tau1"):
        for _ in range(100):
            rates=sample_rates(good,totals,rng)
            assert rates[3]==rates[4]==0
            vals=run_panel(rows,alpha,fit,variant,old,rates,rng,rankw)
            assert vals.shape==(3,2)
            assert np.all(vals>=0) and np.all(vals<=1)
            assert np.all(vals[:,0]<=vals[:,1]+1e-12),(variant,vals)
    print("TH1_V2_E1_SYNTHETIC_ORBIT_BOUND_PASS 4 variants x 100 panels")

if __name__=="__main__":main()
