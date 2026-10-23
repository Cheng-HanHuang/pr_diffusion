"""R0.7 primary frozen definitions: synthetic CPU-only unit tests.

These tests never open PAC outputs, GPU checkpoints, or GT material.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from prdiffusion.track_r.physics import apply_orbit,psnr_rgb8
from prdiffusion.track_r.r0_7_stats import (
    ORBIT,GOOD,BOOT_N,
    choose,orbit_psnr_channel_mse,label_candidate,tau_from_good,
    auc_roc,clustered_ratio_ci,clustered_paired_ci,exact_mcnemar,
    select_e,summarize_psnr,spearman,high_frequency_energy,
    clustered_policy_metric_ci)


def module(filename):
    p=Path(__file__).resolve().parents[2]/"scripts/track_r"/filename
    spec=importlib.util.spec_from_file_location(filename[:-3],p)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_psnr_all_eight_orbits_equals_direct_rgb8():
    gen=torch.Generator().manual_seed(12)
    x=torch.randint(0,256,(1,3,256,256),generator=gen,dtype=torch.uint8)
    gt=torch.randint(0,256,(1,3,256,256),generator=gen,dtype=torch.uint8)
    m=orbit_psnr_channel_mse(x,gt)
    assert set(m)==set(ORBIT)
    for s in ORBIT:
        exact=float(psnr_rgb8(apply_orbit(x,s),gt)[0])
        assert abs(m[s]-exact)<5e-5


def test_raw_orbit_classification_and_global_partial():
    good={"psnr_raw":27.,"psnr_orbit_best":31.,"rho_signed_raw":.09}
    orbit={"psnr_raw":7.,"psnr_orbit_best":29.,"rho_signed_raw":.09}
    nonorbit={"psnr_raw":10.,"psnr_orbit_best":16.,"rho_signed_raw":.09}
    assert label_candidate(good,.05)=="GOOD"
    assert label_candidate(orbit,.05)=="ORBIT_ERROR"
    assert label_candidate(nonorbit,.10)=="NONORBIT_CONSISTENT"
    assert label_candidate(nonorbit,.05)=="NONORBIT_INCONSISTENT"
    assert label_candidate(nonorbit,None)=="NONORBIT_UNKNOWN_TAU"


def test_good_only_tau_and_no_leak_from_failures():
    group=[dict(source="S1_b22",family="FAST",psnr_raw=26.,
                rho_signed_raw=x) for x in (0.01,0.02,0.03,0.04)]
    group.append(dict(source="S1_b22",family="FAST",
                      psnr_raw=4.,rho_signed_raw=99.))
    tau=tau_from_good(group)
    assert tau[("S1_b22","FAST")]==pytest.approx(
        np.quantile([.01,.02,.03,.04],.95,method="linear"))
    assert tau[("S1_b22","FAST")]<.04
    no_good=[dict(source="S2_b24_dev80",family="NP",psnr_raw=1.,
                  rho_signed_raw=3.)]
    assert tau_from_good(no_good)[("S2_b24_dev80","NP")] is None


def test_prior_choose_tie_identity_and_orbit():
    values={p:0.25+i*.001 for i,p in enumerate(ORBIT)}
    values["111"]=.001
    assert choose(values)=="111"
    values["000"]=.001+0.5e-7
    assert choose(values)=="000"


def test_mann_whitney_auc_handles_ties_and_one_class():
    assert auc_roc([0,0,1,1],[.1,.2,.3,.4])==pytest.approx(1.)
    assert auc_roc([0,1],[.1,.1])==pytest.approx(.5)
    assert auc_roc([0,0],[.1,.2]) is None
    assert auc_roc([1,1],[.1,.2]) is None


def test_10k_bootstrap_is_deterministic_clusters_shared_image():
    rows=[
        dict(image_id="same",n=1,d=1),
        dict(image_id="same",n=0,d=1),
        dict(image_id="other",n=1,d=1),
        dict(image_id="other",n=1,d=1),
        dict(image_id="different",n=0,d=1),
    ]
    kwargs=dict(numerator=lambda r:r["n"],denominator=lambda r:r["d"],
                analysis_id="P1|S5|FAST",reps=10000)
    a=clustered_ratio_ci(rows,**kwargs)
    b=clustered_ratio_ci(rows,**kwargs)
    assert a==b
    assert a["estimate"]==pytest.approx(3/5)
    assert a["n_images"]==3
    assert a["n_valid_boot"]==10000
    assert a["ci95"][0]<=a["estimate"]<=a["ci95"][1]


def test_paired_bootstrap_and_exact_mcnemar():
    e1={"a":26.,"b":19.,"c":28.}
    e0={"a":24.,"b":23.,"c":28.}
    boot=clustered_paired_ci(e1,e0,"P4|a|E1-E0",reps=10000)
    assert boot["estimate"]==pytest.approx((2-4+0)/3)
    assert boot["n_images"]==3
    assert exact_mcnemar(e1,e0)=={"n01":0,"n10":1,"p_exact":1.0}
    with pytest.raises(ValueError,match="image"):
        clustered_paired_ci(e1,{"x":3.},"invalid",reps=10000)


def _candidate(i,raw,rep,sign,after,prior,ufd,ceiling):
    return dict(uid=f"u{i}",run_index=rep,
        psnr_raw=raw,psnr_repaired=after,psnr_orbit_best=ceiling,
        rho_signed_raw=sign,rho_signed_repaired=sign,
        S_repaired=prior,p_fake_repaired=ufd)


def test_e0_e1_e2_e3_and_gt_ceilings():
    candidates=[
        _candidate(0,11.,0,.06,30.,.50,.25,32.),
        _candidate(1,29.,1,.02,28.,.30,.95,31.),
        _candidate(2,22.,2,.04,26.,.05,.01,33.),
        _candidate(3,23.,3,.09,27.,.04,.02,35.)]
    result=select_e(candidates,4,.05)
    assert result["psnr"]=={
        "E0":29.,"E1":28.,"E2":26.,"E3":26.,
        "O_raw":29.,"O_orbit":35.}
    assert result["uids"]["E0"]=="u1"
    assert result["uids"]["E2"]=="u2"
    assert result["n_feasible"]==2
    assert select_e(candidates,4,.001)["psnr"]["E2"]==28.
    unavailable=select_e(candidates,4,None)
    assert unavailable["psnr"]["E0"]==29.
    assert unavailable["psnr"]["E1"]==28.
    assert unavailable["psnr"]["O_orbit"]==35.
    assert unavailable["psnr"]["E2"] is None
    assert unavailable["psnr"]["E3"] is None


def test_policy_summary_good_bad_and_quantile():
    r=summarize_psnr([10.,15.,19.,24.,30.])
    assert r["n"]==5
    assert r["good25"]==pytest.approx(.2)
    assert r["bad20"]==pytest.approx(.6)
    assert r["bad15"]==pytest.approx(.2)
    assert r["min"]==10.


def test_high_frequency_energy_and_spearman():
    constant=torch.full((1,3,256,256),125,dtype=torch.uint8)
    alternating=torch.tensor([0,255],dtype=torch.uint8).repeat(128*256)
    checker=alternating.reshape(256,256)[None,None].expand(1,3,-1,-1).contiguous()
    assert high_frequency_energy(constant)==pytest.approx(0.,abs=1e-12)
    assert high_frequency_energy(checker)>0.3
    assert spearman([1,2,3],[3,2,1])==-1.
    assert spearman([1,1,1],[1,2,3]) is None


def test_offline_gt_catalog_fields_guarded():
    prep=module("r0_7_prepare_cpu.py")
    analyze=module("r0_7_analyze_cpu.py")
    assert prep.ORIGINAL_SHA=="8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890"
    assert prep.FAST_SCORE_TASK_SHA=="f268e29b482da10aee31f79de43a6bf19d460f4203f7813c46dbee83b0591596"
    assert prep.FAST_HEADS["parallel"]=="99a9e5b6a2544c56490a95f2498dd4c2cb54262e"
    assert analyze.BOOT_N==10000
    assert analyze.GOOD==25 and analyze.BAD==20
    assert prep.FROZEN_PATH.name=="r0_frozen_v1.json"


def test_s5_same_image_different_measurements_not_cross_selected():
    analyzer=module("r0_7_analyze_cpu.py")
    left={"source":"S5_r0_fast","measurement_source":"S1_b22","image_id":"69073"}
    right={"source":"S5_r0_fast","measurement_source":"S2_b24_dev80","image_id":"69073"}
    assert analyzer.measurement_instance_key(left)!=analyzer.measurement_instance_key(right)
    assert analyzer.measurement_instance_key(left)==("S1_b22","69073")


def test_report_has_all_four_registered_primary_sections():
    analyzer=module("r0_7_analyze_cpu.py")
    sample={
        "status":"R0_7_ANALYSIS_COMPLETE","n_candidates":3240,
        "n_unique_image_ids":180,"aggregate_r0_gpu_hours":3.431,
        "frozen_spec_sha256":"a"*64,"catalog_sha256":"b"*64,
        "P1":[],"P2":[],"P3":[],"P4":[],
        "decision_inputs":{},"tau_unavailable_groups":[],
        "psnr_replay_anomalies":[]}
    md=analyzer.generate_report(sample)
    for section in ("P1 —","P2 —","P3 —","P4 —"):
        assert section in md
    assert "retrospective" in md.lower()
    assert "confirmation305" in md


def test_paired_policy_bootstrap_reports_all_seven_metrics():
    lhs={"a":26.,"b":21.,"c":30.}
    rhs={"a":24.,"b":20.,"c":27.}
    a=clustered_policy_metric_ci(lhs,rhs,"P4|SRC|FAST|K4|E2-E1")
    b=clustered_policy_metric_ci(lhs,rhs,"P4|SRC|FAST|K4|E2-E1")
    assert a==b
    assert a["n_images"]==3
    assert a["replicates"]==10000
    assert set(a["paired_difference"])=={
        "mean","median","good25","bad20","bad15","q05","min"}
    assert a["paired_difference"]["mean"]["estimate"]==pytest.approx(2.)
    assert a["paired_difference"]["good25"]["estimate"]==pytest.approx(0.)
    for m in a["paired_difference"].values():
        assert m["ci95"][0]<=m["estimate"]<=m["ci95"][1]
