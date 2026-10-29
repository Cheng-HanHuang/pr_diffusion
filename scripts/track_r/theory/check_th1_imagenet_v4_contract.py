#!/usr/bin/env python3
"""A12 ImageNet v4 predictive contract; JSON-only, no D4 result access."""
import argparse,json,math
from collections import defaultdict
VARIANTS={"race","S2","tail","race_tau1"}
PANELS={"primary":(32,{16,24,32}),"secondary":(64,{32,48,64})}
POLICIES={"E0","E1_CLIP","O_raw","O_orbit"}
PREFIX="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"

def check(path):
    o=json.load(open(path,encoding="utf8"))
    assert o["version"]=="DRAFT_V4_UNREGISTERED"
    assert o["status"]=="UNREGISTERED"
    assert o["source_prefix4_sha256"]==PREFIX
    assert o["structural_zero_component"] is False
    assert o["n_images"]==100 and o["n_observed_runs"]==16
    assert o["dataset"]=="ImageNet" and o["metric"]=="good20" and o["family"]=="FAST"
    assert o["interval_type"]=="predictive_realized_panel"
    assert not o["provenance"]["held_XP_D4_outcomes_seen"]
    assert not o["provenance"]["xp_returns_accessed"]
    assert o["provenance"]["allowed_run_indices"]==list(range(16))
    for key in ("source_pilot16_sha256","source_clip16_sha256"):
        assert len(o[key])==64 and all(c in "0123456789abcdef" for c in o[key])
    counts=o["g1_class_posterior"]
    assert set(counts)==set("GOPCI")
    assert sum(v["total"] for v in counts.values())==1600
    assert all(0<=v["successes"]<=v["total"] for v in counts.values())
    assert all(v["alpha"]==v["successes"]+1 and v["beta"]==1+v["total"]-v["successes"] for v in counts.values())
    assert counts["C"]["successes"]==counts["I"]["successes"]==0
    assert len(o["predictions"])==120,len(o["predictions"])
    d=defaultdict(dict)
    for p in o["predictions"]:
        panel=p["panel"];variant=p["model_variant"];K=p["K"]
        assert panel in PANELS and variant in VARIANTS
        n,ks=PANELS[panel];assert K in ks and p["panel_n_max"]==n
        assert p["n_images"]==100 and p["metric"]=="good20"
        assert p["policy"] in POLICIES
        dm=p["decision_model"]
        assert dm in ("beta_class","beta_image_cluster") if p["policy"]=="E1_CLIP" else dm is None
        key=(panel,variant,K)
        policy=(p["policy"],dm)
        assert policy not in d[key],(key,policy)
        assert all(math.isfinite(float(p[k])) for k in ("lo95","point","hi95"))
        assert -1e-12<=p["lo95"]<=p["point"]<=p["hi95"]<=1+1e-12
        d[key][policy]=p["point"]
    assert len(d)==24
    for (panel,variant,K),val in d.items():
        assert len(val)==5
        assert val[("E0",None)]<=val[("O_raw",None)]+1e-10
        assert val[("E1_CLIP","beta_class")]<=val[("O_orbit",None)]+1e-10
        assert val[("E1_CLIP","beta_image_cluster")]<=val[("O_orbit",None)]+1e-10
        assert val[("O_raw",None)]<=val[("O_orbit",None)]+1e-10
        race=d[(panel,"race",K)]
        for policy in ("O_raw","O_orbit"):
            assert abs(val[(policy,None)]-race[(policy,None)])<1e-12
    assert len(o["anatomy_curves"])==24
    seen=set()
    for a in o["anatomy_curves"]:
        key=(a["panel"],a["model_variant"],a["K"])
        assert key not in seen and key in d
        seen.add(key)
        x=d[key]
        assert abs(a["generation_failure"]-(1-x[("O_orbit",None)]))<1e-10
        assert abs(a["selection_loss_E0"]-(x[("O_raw",None)]-x[("E0",None)]))<1e-10
        assert abs(a["selection_loss_E1"]-(x[("O_orbit",None)]-x[("E1_CLIP","beta_class")]))<1e-10
    assert len(o["oracle_anchor_checks"])==8
    for a in o["oracle_anchor_checks"]:
        assert a["panel"] in PANELS and a["variant"] in VARIANTS
        assert a["max_abs_shift"]<0.01
        assert len(a["shift_raw"])==len(a["shift_orbit"])==3
        assert max(abs(x) for x in a["shift_raw"]+a["shift_orbit"])<=a["max_abs_shift"]+1e-14
    print("TH1_IMAGENET_V4_CONTRACT_PASS 120 24 8")

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("draft");args=ap.parse_args();check(args.draft)
