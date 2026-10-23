"""Synthetic contracts for R0.8 offline-only overnight scientific return."""
from __future__ import annotations
import importlib.util
import statistics
from pathlib import Path

import pytest


def load_module():
    path=Path(__file__).resolve().parents[2]/"scripts/track_r/r0_8_overnight_cpu.py"
    spec=importlib.util.spec_from_file_location("r0_8_overnight_cpu",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_planner_decision_branches_do_not_authorize_r1():
    m=load_module()
    analysis={
        "P1":[
            {"source":"POOLED_FAMILY","family":"FAST",
             "orbit_share_among_failures":{"estimate":.22}},
            {"source":"POOLED_FAMILY","family":"DAPS",
             "orbit_share_among_failures":{"estimate":.08}}],
        "P3":[
            {"source":"POOLED_FAMILY","family":"FAST","subtype":"ALL",
             "auroc_higher_means_failure":{
                 "S":.82,"S_img":.84,"UnivFD_p_fake":.79}},
            {"source":"POOLED_FAMILY","family":"DAPS","subtype":"ALL",
             "auroc_higher_means_failure":{
                 "S":.75,"S_img":.76,"UnivFD_p_fake":.70}}],
        "decision_inputs":{"D_repair_pooled_success":.92,
                           "D_repair_pooled_harm":.005}}
    decisions=m.derive_decisions(analysis)
    assert decisions["material"]["either_at_least_0p20"] is True
    assert decisions["repair"]["numerical_status"]=="MEETS_PASS_CUTOFFS"
    assert decisions["detect"]["both_at_least_plus_0p03"] is True
    assert decisions["provisional_next_scenario"]=="R1_REPAIR_AND_DETECT_CANDIDATE"
    assert decisions["automatic_R1_authorization"] is False
    analysis["decision_inputs"]["D_repair_pooled_success"]=.70
    bad=m.derive_decisions(analysis)
    assert bad["repair"]["numerical_status"]=="MEETS_FAIL_CUTOFFS"
    assert bad["provisional_next_scenario"]=="R1_DETECTION_ONLY_CANDIDATE"
    analysis["P3"][1]["auroc_higher_means_failure"]["S"]=.50
    analysis["P3"][1]["auroc_higher_means_failure"]["S_img"]=.50
    stopped=m.derive_decisions(analysis)
    assert stopped["provisional_next_scenario"]=="STOP_TRACK_R_CANDIDATE"


def test_linear_five_percent_quantile_and_numeric_gate():
    m=load_module()
    assert m.q05([10.,20.])==pytest.approx(10.5)
    assert m.q05([2.])==2.
    assert m.q05([]) is None
    assert m.close(1.,1.+1e-8)
    assert not m.close(float("nan"),1.)


def test_policy_replay_independently_validates_all_six_policies():
    m=load_module()
    def row(rep,raw,repaired,orbit_best,res,loss,fake):
        return {
            "source":"S5_r0_fast","family":"FAST",
            "measurement_source":"S1_b22","image_id":"60044",
            "uid":"u"+str(rep),"run_index":rep,
            "psnr_raw":raw,"psnr_repaired":repaired,
            "psnr_orbit_best":orbit_best,
            "rho_signed_raw":res,"rho_signed_repaired":res,
            "tau_signed_good_q95":.10,
            "S_repaired":loss,"p_fake_repaired":fake}
    rows=[
        row(0,19.,30.,31.,.03,.60,.80),
        row(1,26.,26.,27.,.04,.40,.30),
        row(2,20.,29.,29.,.05,.10,.60),
        row(3,27.,26.,28.,.06,.20,.01)]
    by_k={
        1:{"E0":19.,"E1":30.,"E2":30.,"E3":30.,"O_raw":19.,"O_orbit":31.},
        2:{"E0":19.,"E1":30.,"E2":26.,"E3":26.,"O_raw":26.,"O_orbit":31.},
        4:{"E0":19.,"E1":30.,"E2":29.,"E3":26.,"O_raw":27.,"O_orbit":31.}}
    tables=[]
    for k,v in by_k.items():
        stats={}
        for policy,value in v.items():
            stats[policy]=dict(n=1,mean=value,median=value,
                good25=float(value>=25),bad20=float(value<20),
                bad15=float(value<15),q05=value,min=value)
        tables.append(dict(source="S5_r0_fast",family="FAST",
            K=k,n_images_available=1,policy_stats=stats))
    assert m.audit_policy_table(rows,{"P4":tables})==3
    tables[2]["policy_stats"]["E2"]["mean"]+=1.5
    with pytest.raises(RuntimeError,match="P4_METRIC"):
        m.audit_policy_table(rows,{"P4":tables})


def test_archive_has_no_detailed_gt_rows_or_original_paths():
    m=load_module()
    assert "R0_7_SCORED_GT_ROWS.jsonl" not in m.REPORT_FILES
    assert "R0_7_OFFLINE_GT_CATALOG.json" not in m.REPORT_FILES
    assert len(m.REPORT_FILES)==5
    assert m.R07.name=="r0_7_cpu_20261023T080434Z"
    assert m.SOURCE_HEAD=="c6dbd6089579c79e9e6fd4a9dd20fc402bc1185b"


def test_report_emphasizes_planner_gate_and_exploratory_cases():
    m=load_module()
    decisions={
        "material":{"FAST_orbit_share":.21,"DAPS_orbit_share":.07,
                    "either_at_least_0p20":True},
        "repair":{"success_pooled":.91,"harm_pooled":.009,
                  "numerical_status":"MEETS_PASS_CUTOFFS"},
        "detect":{"best_prior_minus_univfd":{"FAST":.04,"DAPS":.04},
                  "both_at_least_plus_0p03":True,
                  "both_at_least_minus_0p02":True},
        "provisional_next_scenario":"R1_REPAIR_AND_DETECT_CANDIDATE"}
    a={"aggregate_r0_gpu_hours":3.4309564}
    labels={"GOOD":2000,"ORBIT_ERROR":200,
            "NONORBIT_CONSISTENT":500,"NONORBIT_INCONSISTENT":540}
    by={("S5_r0_fast","FAST"):[
        {"raw_label":"GOOD"},{"raw_label":"ORBIT_ERROR"}]}
    doc=m.markdown(a,decisions,labels,by,12,{})
    assert "PLANNER_R1_DECISION_REQUIRED" in doc
    assert "EXPLORATORY" in doc
    assert "3.430956" in doc
    assert "confirmation305" in doc
