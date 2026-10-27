#!/usr/bin/env python3
"""A6 prediction-output structural acceptance gate. Uses only a draft JSON.

Run: python scripts/track_r/theory/check_a6_panel_contract.py DRAFT.json
Never opens XP returns or any observed post-prefix reconstruction output.
"""
import argparse,json,math
from collections import defaultdict

PANELS={
 "DAPS":{"primary":(8,(4,6,8))},
 "SITCOM":{"primary":(16,(8,12,16)),"secondary":(32,(8,12,16,24,32))},
 "FAST":{"primary":(64,(8,12,16,24,32,48,64)),"secondary":(16,(8,12,16))}}
COHORTS=("B22","DEV80","POOLED")
VARIANTS=("race","S2","tail","race_tau1")
POLICIES=("E0","E1_CLIP","O_raw","O_orbit")
def run(path):
    obj=json.load(open(path,encoding="utf-8"))
    assert obj["version"]=="DRAFT_PREFIX_CONDITIONAL_UNREGISTERED"
    assert obj["interval_type"]=="predictive_realized_panel"
    assert obj["panels"]=={f:{x:[n,list(ks)]for x,(n,ks)in q.items()}for f,q in PANELS.items()}
    assert len(obj["fit_diagnostics"])==6
    preds=obj["predictions"]
    assert len(preds)==976,len(preds)
    allkeys=set();d=defaultdict(dict)
    for x in preds:
        f=x["family"];panel=x["panel"];co=x["cohort"];v=x["model_variant"]
        k=x["K"];policy=x["policy"]
        nmax,grid=PANELS[f][panel]
        assert x["panel_n_max"]==nmax and k in grid
        assert co in COHORTS and v in VARIANTS and policy in POLICIES
        assert x["n_images"]==({"B22":100,"DEV80":80,"POOLED":180}[co])
        assert not(f=="DAPS" and k==4 and co!="B22")
        key=(f,panel,co,v,k,policy)
        assert key not in allkeys,key
        allkeys.add(key)
        assert math.isfinite(x["point"])
        assert 0<=x["lo95"]<=x["point"]<=x["hi95"]<=1
        d[(f,panel,co,v,k)][policy]=x["point"]
    assert len(allkeys)==len(preds)
    anatomy=obj["anatomy_curves"];cross=obj["crossover_K"]
    assert len(anatomy)==252,len(anatomy)
    assert len(cross)==60,len(cross)
    for a in anatomy:
        f=a["family"];pn=a["panel"];co=a["cohort"];v=a["model_variant"];k=a["K"]
        assert k in PANELS[f][pn][1]
        if f=="DAPS" and k==4 and co!="B22":continue
        values=d[(f,pn,co,v,k)]
        assert set(values)==set(POLICIES)
        assert -1e-10<=values["O_orbit"]-values["E1_CLIP"]<1+1e-10
        assert values["O_orbit"]+1e-10>=values["O_raw"]
        assert values["E1_CLIP"]+1e-10>=values["E0"]
        assert abs(a["generation_failure"]-(1-values["O_orbit"]))<1e-8
        assert abs(a["selection_loss"]-(values["O_orbit"]-values["E1_CLIP"]))<1e-8
        assert abs(a["generation_failure"]+a["selection_loss"]+
                   values["E1_CLIP"]-1)<1e-8
    for c in cross:
        ks=PANELS[c["family"]][c["panel"]][1]
        assert c["panel_n_max"]==PANELS[c["family"]][c["panel"]][0]
        assert c["K_star_on_grid"] is None or c["K_star_on_grid"] in ks
    for family,pq in PANELS.items():
        for pn,(nmax,Kgrid) in pq.items():
            for co in COHORTS:
                for v in VARIANTS:
                    for k in Kgrid:
                        if family=="DAPS" and k==4 and co!="B22":continue
                        assert set(d[(family,pn,co,v,k)])==set(POLICIES)
    print("TH1_A6_PANEL_CONTRACT_PASS predictions",len(preds),
          "anatomy",len(anatomy),"crossovers",len(cross))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("draft_json")
    args=ap.parse_args();run(args.draft_json)
