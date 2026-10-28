#!/usr/bin/env python3
"""A8.3 ImageNet FAST Good20 optional E1 addendum contract.

Validation is exclusively against the addendum DRAFT JSON and the
REGISTERED immutable v2 JSON. NO phase-2 data or xp/returns are read.

24 E1 rows = four structural variants x three K values x two
decision-uncertainty variants (Beta pooled, image-cluster sensitivity).
The primary Beta E1 is allowed to be BELOW E0: G1 can flip GOOD.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

BASE_SHA="8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55"
PHASE_SHA="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"
VARIANTS=("race","S2","tail","race_tau1")
DECISION_MODELS=("beta_class","beta_image_cluster")
KGRID=(8,12,16)

def check(draft_path,base_path):
    raw=Path(base_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BASE_SHA:
        raise ValueError("base v2 does not equal frozen registered bytes")
    base=json.loads(raw)
    with open(draft_path,encoding="utf8")as f:x=json.load(f)
    assert base["version"]=="TH1_PREDICTIONS_v2"
    assert base["E1_CLIP"]["status"]=="ABSTAIN"
    assert x["version"]=="DRAFT_V2_E1_UNREGISTERED"
    assert x["status"]=="DRAFT_NOT_FROZEN"
    assert x["base_v2_sha256"]==BASE_SHA
    assert x["phase1_sha256"]==PHASE_SHA
    assert x["interval_type"]=="predictive_realized_panel"
    assert x["dataset"]=="ImageNet" and x["family"]=="FAST"
    assert x["metric"]=="good20" and x["n_images"]==100
    assert x["K_grid"]==list(KGRID) and x["panel_n_max"]==16
    assert x["provenance"]["phase2_outcomes_seen"] is False
    assert x["provenance"]["xp_returns_accessed"] is False
    assert x["provenance"]["source_limited_to_runs_0_3"] is True
    assert len(x["phase1_clip_sha256"])==64
    assert all(c in "0123456789abcdef" for c in x["phase1_clip_sha256"])
    count=x["clip_class_posterior_counts"]
    assert set(count)==set("GOPCI")
    totals={"G":100,"O":67,"P":3,"C":188,"I":42}
    for label,expected in totals.items():
        t=count[label]
        assert t["total"]==expected
        assert 0<=t["successes"]<=expected
        assert t["beta_alpha"]==t["successes"]+1
        assert t["beta_beta"]==expected-t["successes"]+1
        if label in ("C","I"):
            assert t["successes"]==0 and t["rate_override_zero"]
    assert len(x["simulation_draw_checks"])==4
    assert {i["variant"]for i in x["simulation_draw_checks"]}==set(VARIANTS)
    assert all(i["all_E1_le_orbit"] for i in x["simulation_draw_checks"])
    # The control-variate anchor must be the simulated O_orbit ORACLE, so it
    # must reproduce the registered v2 O_orbit up to Monte Carlo error.
    anchors=x["oracle_anchor_checks"]
    assert {a["variant"] for a in anchors}==set(VARIANTS) and len(anchors)==4
    for a in anchors:
        assert len(a["shift"])==len(KGRID)
        assert all(abs(s)<0.01 for s in a["shift"]),(a["variant"],a["shift"])
    targets={(r["model_variant"],r["K"]):r["point"]
        for r in base["predictions"] if r["policy"]=="O_orbit"}
    assert len(targets)==12
    preds=x["predictions"]
    assert len(preds)==24,len(preds)
    seen=set()
    for r in preds:
        k=r["K"];v=r["model_variant"];d=r["decision_model"]
        assert k in KGRID and v in VARIANTS and d in DECISION_MODELS
        key=(v,d,k);assert key not in seen;seen.add(key)
        assert r["policy"]=="E1_CLIP"
        assert (r["family"],r["dataset"],r["metric"])==(
            "FAST","ImageNet","good20")
        assert r["n_images"]==100 and r["panel_n_max"]==16
        assert r["panel"]=="primary" and r["cohort"]=="D1_fixed100"
        assert all(math.isfinite(r[q]) for q in (
            "point","lo95","hi95","O_orbit_registered_point"))
        assert 0<=r["lo95"]<=r["point"]<=r["hi95"]<=1
        orbit=targets[v,k]
        assert abs(r["O_orbit_registered_point"]-orbit)<1e-12
        assert r["point"]<=orbit+1e-10
    assert len(seen)==24
    # No original v2 policy is rewritten in the addendum.
    assert "original_v2_predictions" not in x
    print("TH1_IMAGENET_V2_E1_CONTRACT_PASS",len(preds),
          "phase1 decisions 400, no phase-2 outcomes")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--draft",required=True)
    p.add_argument("--base-v2",required=True)
    a=p.parse_args()
    check(a.draft,a.base_v2)
