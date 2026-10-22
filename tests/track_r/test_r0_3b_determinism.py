"""Synthetic R0.3b gate-contract tests, with no PAC files or GPUs."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[2]/"scripts/track_r/r0_3b_compare_cpu.py"
    spec=importlib.util.spec_from_file_location("r0_3b_comparator",path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(uid="a",shift=0.0):
    orbit=["000","001","010","011","100","101","110","111"]
    score={p:(0.01+0.001*i+shift) for i,p in enumerate(orbit)}
    mid={p:(0.002+0.0001*i+shift) for i,p in enumerate(orbit)}
    by_t={p:{"50":0.01+i*0.001+shift,
              "100":0.005+i*0.0001+shift} for i,p in enumerate(orbit)}
    return {"uid":uid,"input_sha256":"a"*64,
            "prior":{"S":score,"S_mid":mid,"per_t_means":by_t,
                     "s_star":"000","unet_forwards":256}}


def test_same_scores_with_independent_batching_pass():
    m=module()
    a=[row(uid=str(i)) for i in range(8)]
    b=[row(uid=str(i),shift=1e-10) for i in range(8)]
    comparison=m.compare(a,b)
    assert comparison["status"]=="PASS"
    assert comparison["n_compared_values"]==8*(8+8+16)


def test_threshold_is_relative_and_fail_closed():
    m=module()
    a=[row(uid=str(i)) for i in range(8)]
    b=[row(uid=str(i)) for i in range(8)]
    b[2]["prior"]["S"]["111"]*=1.0001
    comparison=m.compare(a,b)
    assert comparison["status"]=="FAIL"
    assert comparison["worst_case"]["uid"]=="2"
    b[2]["prior"]["S"]["111"]=a[2]["prior"]["S"]["111"]
    b[1]["prior"]["s_star"]="111"
    with pytest.raises(RuntimeError,match="FORWARD_COUNT_OR_ORBIT_CHOICE"):
        m.compare(a,b)
    b[1]["prior"]["s_star"]="000"
    b[1]["input_sha256"]="0"*64
    with pytest.raises(RuntimeError,match="DIFFERENT_CANDIDATES"):
        m.compare(a,b)
