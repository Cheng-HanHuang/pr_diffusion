"""R0.5 frozen seed/UID/privacy and native FAST canary contracts."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest


def load(script):
    p=Path(__file__).resolve().parents[2]/"scripts/track_r"/script
    spec=importlib.util.spec_from_file_location(script[:-3],p)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_native_root_seed_exact_frozen_sha_rule():
    import hashlib
    prep=load("r0_5_prepare_cpu.py")
    worker=load("r0_5_fast_gpu.py")
    seed=prep.native_seed("S2_b24_dev80","63070",0)
    frozen=int.from_bytes(hashlib.sha256(
        b"R0FAST|S2_b24_dev80|63070|0").digest()[:8],"little") & ((1<<63)-1)
    assert seed==frozen==7452755359293807986
    assert prep.native_seed("S2_b24_dev80","63070",1)!=seed
    assert prep.native_seed("S1_b22","63070",0)!=seed
    assert worker.SOURCE_HEAD==prep.FAST_SOURCE_HEAD

def test_sealed_gpu_roots_disallow_gt_and_forgery():
    m=load("r0_5_fast_gpu.py")
    valid=[]
    for i in range(720):
        src="S1_b22" if i<400 else "S2_b24_dev80"
        iid=str(60000+i//4)
        valid.append({"uid":f"S5_r0_fast|FAST|{src}_{iid}|rep{i%4}",
            "source":"S5_r0_fast",
            "measurement_source":src,"image_id":iid,
            "rep":i%4,"seed":i,
            "measurement_path":"/tmp/signed.pt",
            "measurement_sha256":"a"*64})
    payload={"schema":"prdiff.track_r.r0.5.gt_free_tasks.v1",
        "status":"R0_5_TASKS_FROZEN",
        "head":"fake","fast_head":m.SOURCE_HEAD,
        "n_roots":720,"n_images":180,"gt_paths_supplied":False,
        "rows":valid}
    assert len(m.frozen_roots(payload,"fake"))==720
    dirty=dict(payload)
    dirty["rows"]=[dict(valid[0],gt_path="/tmp/gt.pt")]+valid[1:]
    with pytest.raises(RuntimeError,match="GT_LEAK"):
        m.frozen_roots(dirty,"fake")
    dirty=dict(payload,gt_paths_supplied=True)
    with pytest.raises(RuntimeError,match="NOT_FROZEN"):
        m.frozen_roots(dirty,"fake")

def test_native_config_cannot_drift():
    m=load("r0_5_fast_gpu.py")
    reference={"sampler":"edm_FAST_DIPS","T":150,
       "K_ADMM":2,"S_x":5,"rho":200,
       "epsilon":0.05,"backtrack":True,
       "alpha_max":1.0,"roots_per_image":4,
       "sources":["S1_b22","S2_b24_dev80"]}
    native={k:reference[k] for k in (
        "sampler","T","K_ADMM","S_x","rho",
        "epsilon","backtrack","alpha_max")}
    assert m.validate_native_conf({"fast_native_pixel":native},
        {"fast_generation":reference})==native
    changed=dict(native,T=149)
    with pytest.raises(RuntimeError,match="CONFIG_DRIFT"):
        m.validate_native_conf({"fast_native_pixel":changed},
            {"fast_generation":reference})

def test_reconstruction_root_filename_unique_and_safe(tmp_path):
    m=load("r0_5_fast_gpu.py")
    r1={"uid":"S5_r0_fast|FAST|S1_b22_69073|rep0"}
    r2={"uid":"S5_r0_fast|FAST|S1_b22_69073|rep1"}
    assert m.root_folder(tmp_path,r1)!=m.root_folder(tmp_path,r2)
    assert m.root_folder(tmp_path,r1).parent==tmp_path
    assert m.root_folder(tmp_path,r1).name.startswith("root_")
