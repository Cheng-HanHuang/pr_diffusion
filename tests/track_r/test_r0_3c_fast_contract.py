"""R0.3c CPU-only smoke contracts and aggregate-budget projection."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


def module(name):
    root=Path(__file__).resolve().parents[2]
    path=root/"scripts/track_r"/name
    spec=importlib.util.spec_from_file_location(name[:-3],path)
    out=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def test_frozen_native_config_and_seed():
    gen=module("r0_3c_fast_smoke_gpu.py")
    f={"fast_generation":{
        "sampler":"edm_FAST_DIPS","T":150,"K_ADMM":2,"S_x":5,
        "rho":200,"epsilon":0.05,"backtrack":True,"alpha_max":1}}
    cfg=dict(f["fast_generation"])
    gen.validate_native_config(cfg,f)
    bad=dict(cfg,rho=201)
    with pytest.raises(RuntimeError,match="NATIVE_CONFIG_DRIFT"):
        gen.validate_native_config(bad,f)
    expected=gen.seed_r0("S2_b24_dev80","63070",0)
    assert 0<=expected<(1<<63)
    assert expected==gen.seed_r0("S2_b24_dev80","63070",0)
    assert expected!=gen.seed_r0("S2_b24_dev80","63070",1)
    assert expected!=15150000+63070


def test_fast_gpu_hours_projection_fails_closed():
    budget=module("r0_3c_fast_adjudicate_cpu.py")
    r=budget.budget_projection(2.5031679952924604,0.7151908557978459,
                               [30.,38.],0.01686,0.02)
    assert r["status"]=="PASS"
    assert r["new_fast_roots"]==720
    assert r["native_generation_projected_gpu_h"]==pytest.approx(7.6)
    assert r["conservative_total_gpu_h"]<=16
    over=budget.budget_projection(2.5,0.7,[100.,100.],0.01)
    assert over["status"].startswith("STOP_BUDGET")
    for invalid in ([-1.],[float("nan")],[]):
        with pytest.raises(ValueError):
            budget.budget_projection(2.,1.,invalid,0.1)


def test_replay_tolerance_is_absolute_db():
    test=module("r0_3c_fast_adjudicate_cpu.py")
    assert test.REPLAY_MAX_DB_DIFF==0.05
    assert test.MAX_TOTAL_GPU_HOURS==16.
    assert test.NEW_FAST_ROOTS==720
