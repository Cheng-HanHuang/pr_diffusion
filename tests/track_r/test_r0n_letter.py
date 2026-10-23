"""Synthetic CPU-only tests of N0 integrity and frozen N1 letter evidence.

These tests require no PAC data, diffusion model, or GPU access.
"""
from __future__ import annotations
from collections import Counter
import importlib.util
import json
from pathlib import Path

import pytest

from prdiffusion.track_r import r0n_letter as L

def load_script(name):
    p=Path(__file__).resolve().parents[2]/"scripts/track_r"/name
    spec=importlib.util.spec_from_file_location(name[:-3],p)
    x=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x)
    return x

def candidate(source,family,uid,image_id,rep,raw,repaired,best,res,
              gt_orient="111",prior_orient="111"):
    s={f"{j:03b}":5. for j in range(8)}
    s["000"]=raw;s[gt_orient]=best;s[prior_orient]=repaired
    if prior_orient=="000":s["000"]=repaired
    return {
        "uid":uid,"source":source,"family":family,
        "measurement_source":"S1_b22" if source=="S5_r0_fast" else source,
        "image_id":image_id,"run_index":rep,
        "psnr_raw":raw,"psnr_repaired":repaired,
        "psnr_orbit_best":best,"psnr_by_orbit":s,"s_gt":gt_orient,
        "raw_label":"GOOD" if raw>=25 else (
           "ORBIT_ERROR" if best>=25 else "NONORBIT_INCONSISTENT"),
        "rho_signed_raw":res,"rho_signed_repaired":res,
        "rho_clipped_raw":res,"rho_clipped_repaired":res,
        "tau_signed_good_q95":.2,
        "S_repaired":.3+rep*.01,"S_img":.1+rep*.01,
        "p_fake_repaired":.4+rep*.01,
        "after_repair_failure":repaired<25}

def test_n0_rejects_candidate_or_gt_paths_and_requires_exact_zip():
    n0=load_script("r0n_0_import_tables_cpu.py")
    assert n0.ZIP_SHA=="13c008f3bca59b7d8d717ec3215fa1f783ca453b31b9001886c84633b513efab"
    assert len(n0.FILES)==7
    assert "R0_7_SCORED_GT_ROWS.jsonl" not in n0.FILES
    assert "R0_7_OFFLINE_GT_CATALOG.json" not in n0.FILES
    with pytest.raises(RuntimeError,match="PATH_FIELD"):
        n0.validate_aggregate("R0_8_READINESS.json",b'{"gt_path":"/tmp/private"}')
    with pytest.raises(RuntimeError,match="PATH_FIELD"):
        n0.validate_aggregate("R0_8_READINESS.json",b'{"x":{"candidate_path":"/tmp/a"}}')
    n0.validate_aggregate("R0_8_READINESS.json",b'{"source_r07_run":"/safe/exposed"}')

def test_frozen_grid_60_and_cost_correct():
    c={
        "n1d_cost_accuracy":{
            "timestep_index_order":[50,100,200,300,400,500,600,700],
            "J_subsets":[1,2,4],
            "T_subsets":{
                "single":[[50],[100],[200],[300],[400],[500],[600],[700]],
                "mid4":[300,400,500,600],
                "all8":[50,100,200,300,400,500,600,700]},
            "member_sets":{"all8":[f"{i:03b}" for i in range(8)],
                           "global2":["000","111"]}}}
    grid=L.cost_configs(c)
    assert len(grid)==60 and len({r["name"] for r in grid})==60
    costs={r["name"]:r["forwards"] for r in grid}
    assert costs["all8|J4|all8"]==256
    assert costs["mid4|J4|all8"]==128
    assert costs["single_50|J1|global2"]==2

def test_draw_score_uses_ordered_t_j_and_identity_ties():
    cfg=dict(name="single_300|J1|global2",T=[300],J=1,
             members=["000","111"])
    losses={"000":[1.]*32,"111":[1.]*32}
    losses["111"][12]=.3
    assert L.score_from_draws(losses,cfg)=="111"
    losses["000"][12]=.29999997
    assert L.score_from_draws(losses,cfg)=="000"
    losses["111"]=[.1]*31
    with pytest.raises(RuntimeError,match="LOSS_VECTOR"):
        L.score_from_draws(losses,cfg)

def test_dev_only_selection_prefers_cheapest_then_registered_ties():
    configs=[
        {"name":"all8|J1|global2","forwards":16,
         "member_set":"global2","J":1},
        {"name":"all8|J2|all8","forwards":128,
         "member_set":"all8","J":2},
        {"name":"mid4|J2|all8","forwards":64,
         "member_set":"all8","J":2},
        {"name":"single_50|J4|all8","forwards":32,
         "member_set":"all8","J":4}]
    rows=[{"configuration":c["name"],"cohort":"DEV80_enriched",
           "success":.96,"n_harm":0,"forwards":c["forwards"]}
          for c in configs]
    assert L.choose_dev(rows,configs)["configuration"]=="all8|J1|global2"
    rows[0]["n_harm"]=1
    assert L.choose_dev(rows,configs)["configuration"]=="single_50|J4|all8"
    for r in rows:r["n_harm"]=1
    assert L.choose_dev(rows,configs) is None

def test_postselection_partition_distinguishes_generation_and_misses():
    c=[
      candidate("S1_b22","DAPS","a","1",0,18,30,31,.01),
      candidate("S1_b22","DAPS","b","2",0,12,12,14,.01),
      candidate("S1_b22","DAPS","c","2",1,13,13,29,.03),
      candidate("S1_b22","DAPS","d","3",0,18,18,29,.01,prior_orient="000"),
      candidate("S1_b22","DAPS","e","4",0,10,10,15,.01,prior_orient="000")]
    a=L.policy_values([c[0]],1,.2)
    assert a["failure_kind"]=="ORBIT_FIXED"
    a=L.policy_values(c[1:3],2,.2)
    assert a["failure_kind"]=="SELECTION_MISS"
    a=L.policy_values([c[3]],1,.2)
    assert a["failure_kind"]=="ORBIT_NOT_FIXED"
    a=L.policy_values([c[4]],1,.2)
    assert a["failure_kind"]=="GENERATION_FAILURE"

def test_immutable_known_witnesses_and_labels():
    assert L.EXPECTED_LABELS=={"GOOD":2040,"ORBIT_ERROR":85,
             "NONORBIT_CONSISTENT":324,"NONORBIT_INCONSISTENT":791}
    assert L.P4_GOOD25_WITNESSES[("S5_r0_fast","FAST",4)][0]==180
    assert L.P4_GOOD25_WITNESSES[("S1_b22","DAPS",2)][2]==.95

def test_cpu_script_only_has_explicit_no_gpu_gate_and_n0_receipt():
    path=Path(__file__).resolve().parents[2]/"scripts/track_r/r0n_1_letter_tables_cpu.py"
    s=path.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in s
    assert "STOP_R0N_N1_GPU_VISIBILITY" in s
    assert "R0N_N1_CPU_EVIDENCE_PASS" in s
    assert "R0_7_SCORED_GT_ROWS.jsonl" in s
    assert "replay_receipts" in s
    assert "n_configurations" in s


def test_all_new_n0_n1_scripts_are_syntactically_valid():
    root=Path(__file__).resolve().parents[2]
    for relative in (
        "scripts/track_r/r0n_0_import_tables_cpu.py",
        "scripts/track_r/r0n_1_letter_tables_cpu.py",
        "prdiffusion/track_r/r0n_letter.py"):
        path=root/relative
        compile(path.read_text(encoding="utf-8"),str(path),"exec")


def test_pillow_fallback_generates_three_real_png_pdf_figures(tmp_path):
    """Actual plotted data, not placeholder boxes; no matplotlib dependency."""
    from PIL import Image
    from prdiffusion.track_r.r0n_figures_pillow import render
    from prdiffusion.track_r.r0n_letter import cohort
    a={
        "cohort":"B22_random","family":"DAPS","n_runs":5,
        "GOOD":3,"ORBIT_ERROR":1,
        "NONORBIT_CONSISTENT":1,"NONORBIT_INCONSISTENT":0}
    b={
        "cohort":"DEV80_enriched","family":"FAST","n_runs":6,
        "GOOD":2,"ORBIT_ERROR":2,
        "NONORBIT_CONSISTENT":1,"NONORBIT_INCONSISTENT":1}
    sample=[]
    for i in range(5):
        sample.append({"image_id":"B","family":"DAPS",
            "rho_signed_raw":0.01+i*.005})
    for i in range(6):
        sample.append({"image_id":"D","family":"FAST",
            "rho_signed_raw":0.02+i*.007})
    splits={"B22_random":{"B"},"DEV80_enriched":{"D"}}
    grid=[
        {"cohort":"B22_random","success":0.8,
         "harm":0.0,"forwards":16},
        {"cohort":"DEV80_enriched","success":1.0,
         "harm":0.2,"forwards":64}]
    renderer=render(tmp_path,[a,b],[],grid,grid[1],
                    sample,splits,cohort)
    assert "Pillow" in renderer["renderer"]
    assert renderer["formats"]==["PNG","PDF"]
    for item in ("N1_FIG1_ANATOMY","N1_FIG2_COST_SUCCESS",
                 "N1_FIG3_RESIDUAL"):
        png=tmp_path/"figures"/(item+".png")
        pdf=tmp_path/"figures"/(item+".pdf")
        with Image.open(png) as im:
            assert im.size==(1720,1030)
            assert im.convert("RGB").getbbox()
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert pdf.stat().st_size>1000
