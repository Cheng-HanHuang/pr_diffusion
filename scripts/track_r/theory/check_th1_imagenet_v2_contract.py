#!/usr/bin/env python3
"""ImageNet A6.6 v2 draft structure validator; reads NO future outcomes.

Expected: 100 fixed ImageNet FAST Good20 images, nmax16,
K8/12/16, 4 variants, E0/O_raw/O_orbit only.
"""
import argparse,json,math
from collections import defaultdict
VARS=("race","S2","tail","race_tau1")
KS=(8,12,16)
POL=("E0","O_raw","O_orbit")

def validate(path):
    with open(path,encoding="utf8")as f:o=json.load(f)
    assert o["version"]=="DRAFT_V2_UNREGISTERED"
    assert o["source_sha256"]=="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"
    assert o["n_images"]==100 and o["panel_n_max"]==16
    assert o["dataset"]=="ImageNet" and o["metric"]=="good20"
    assert o["K_grid"]==list(KS)
    assert o["interval_type"]=="predictive_realized_panel"
    assert o["E1_CLIP"]["status"]=="ABSTAIN"
    p=o["predictions"];assert len(p)==36,len(p)
    dic=defaultdict(dict)
    for r in p:
        k=r["K"];v=r["model_variant"];policy=r["policy"]
        assert k in KS and v in VARS and policy in POL
        assert r["panel_n_max"]==16 and r["n_images"]==100
        key=(v,k);assert policy not in dic[key]
        assert 0<=r["lo95"]<=r["point"]<=r["hi95"]<=1
        dic[key][policy]=r["point"]
    assert len(dic)==12
    for key,vals in dic.items():
        assert set(vals)==set(POL)
        assert vals["O_orbit"]>=vals["O_raw"]-1e-10
        assert vals["O_raw"]>=vals["E0"]-1e-9
        for k in ("O_raw","O_orbit"):
            assert abs(vals[k]-dic[("race",key[1])][k])<1e-12
    anatomy=o["anatomy_curves"]
    assert len(anatomy)==12
    for a in anatomy:
        assert a["model_variant"]in VARS and a["K"]in KS
        assert a["E1_CLIP_selection_loss"] is None
        assert abs(a["generation_failure"]-(1-dic[(a["model_variant"],a["K"])]["O_orbit"]))<1e-10
    print("TH1_IMAGENET_V2_CONTRACT_PASS",len(p),len(anatomy))

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("draft_json")
    validate(arg.parse_args().draft_json)
