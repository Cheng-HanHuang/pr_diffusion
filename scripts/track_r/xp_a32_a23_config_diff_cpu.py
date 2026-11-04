#!/usr/bin/env python3
"""A23 XP-K: CPU-only exact-one-parameter configuration-diff *screen*.

This verifies JSON snapshots supplied by prospective native adapters.
It does NOT prove native binding equivalence or pass prereg A23 gate (i)
alone. Gate (ii) two-root historical replay requires an authorized GPU,
and gate (iii) requires an authentic prior F8 receipt. No GPU operations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

FAMILIES = {
    "SITCOM": {"target": 0.05, "original": 0.01, "semantic": "Langevin_tau", "must_unchanged": ("Euler_measurement_step",)},
    "FAST_DIPS": {"target": 33.26, "original": 0.05, "semantic": "projection_epsilon", "must_unchanged": ()},
    "DAPS_FFHQ": {"target": math.sqrt(2.0) * 0.05, "original": 0.01, "semantic": "Langevin_tau_historical_FFHQ", "must_unchanged": ()},
}
STATUS = "A23_ONE_CONFIG_LEAF_DIFF_ONLY_NOT_FULL_PARAMETER_GATE"

def gate(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError("STOP_A23_XPK_DIFF_" + code)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def normalize(obj, path=()):
    """Flatten all leaves, including list elements and keys; types matter."""
    if isinstance(obj, dict):
        gate(all(isinstance(k, str) and k != "" for k in obj), "EMPTY_KEY")
        out={}
        for k,v in obj.items():
            out.update(normalize(v,path+(k,)))
        return out
    if isinstance(obj, list):
        out={}
        # Track list length so list append/removal cannot go unnoticed.
        out[path+("__len__",)]=len(obj)
        for i,v in enumerate(obj):
            out.update(normalize(v,path+(str(i),)))
        return out
    gate(obj is None or type(obj) in (str,bool,int,float), "UNSUPPORTED_TYPE")
    if type(obj) is float: gate(math.isfinite(obj), "NONFINITE_CONFIG")
    return {path:obj}

def canonical_value(x):
    return (type(x).__name__, x)

def check_exact_one_change(baseline, proposed, family, key_path):
    gate(family in FAMILIES, "FAMILY")
    gate(isinstance(baseline,dict) and isinstance(proposed,dict),"OBJECTS")
    original=normalize(baseline)
    new=normalize(proposed)
    gate(set(original)==set(new), "KEYS_OR_STRUCTURES_CHANGED")
    delta=[p for p in original if canonical_value(original[p])!=canonical_value(new[p])]
    gate(len(delta)==1, "NOT_EXACTLY_ONE_PARAMETER")
    gate(isinstance(key_path,str) and key_path and tuple(key_path.split("."))==delta[0],
         "CHANGED_KEY_NOT_FROZEN_NATIVE_BINDING")
    old=original[delta[0]]
    nxt=new[delta[0]]
    gate(type(old) in (int,float) and type(nxt) in (int,float),
         "NUMERIC_PARAMETER")
    spec=FAMILIES[family]
    # Equal registered scalars; reject float representation drift.
    gate(math.isclose(float(old), spec["original"],rel_tol=0.,abs_tol=1e-12),
         "HISTORICAL_BASELINE_VALUE")
    gate(math.isclose(float(nxt),spec["target"],rel_tol=0.,abs_tol=1e-10),
         "REGISTERED_CALIBRATION_VALUE")
    # A source-code/native binding proof is NOT implied by matching a JSON key.
    return {
        "status":STATUS,
        "family":family,
        "native_binding_path_declared_unverified":key_path,
        "semantic_goal":spec["semantic"],
        "historical_parameter":float(old),
        "calibrated_parameter":float(nxt),
        "changed_key_count":1,
        "exact_diff":key_path,
        "complete_native_source_diff_gate_passed":False,
        "historical_two_root_identity_replay_passed":False,
        "DAPS_F8_replay_gate_passed":False,
        "future_GPU_generation_authorized":False,
        "GPU_resource_hold":True,
        "CPU_only":True,
        "no_experiments_run":True,
    }

def safe_read(path):
    path=Path(path)
    gate(path.is_file() and not path.is_symlink(),"SOURCE_JSON_FILE")
    data=path.read_bytes()
    obj=json.loads(data)
    return obj,sha(data)

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES","")=="","CPU_ONLY")
    gate(a.family in FAMILIES,"FAMILY_ARG")
    before,bs=safe_read(a.baseline)
    after,cs=safe_read(a.proposed)
    proof=check_exact_one_change(before,after,a.family,a.key)
    proof.update(baseline_file_sha256=bs,
                 proposed_file_sha256=cs,
                 historical_snapshot=str(Path(a.baseline).resolve()),
                 proposed_snapshot=str(Path(a.proposed).resolve()))
    payload=json.dumps(proof,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.out:
        out=Path(a.out)
        gate(not out.exists() and not out.is_symlink(),"NO_OVERWRITE")
        with out.open("x") as f:f.write(payload)
    else:
        print(payload,end="")
    print("A23_CONFIG_ONE_LEAF_AUDIT_PASS_BUT_NATIVE_GATE_NOT_PASSED",file=sys.stderr)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",required=True)
    p.add_argument("--proposed",required=True)
    p.add_argument("--family",choices=sorted(FAMILIES),required=True)
    p.add_argument("--key",required=True,
                   help="prospective exact nested JSON key, e.g. sampler.foo, pending native binding source proof")
    p.add_argument("--out",default=None)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A23_XPK_CONFIG_DIFF",e,file=sys.stderr)
        sys.exit(8)
