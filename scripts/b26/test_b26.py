#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,json,os,sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[2]
SPEC=REPO/"configs/b26/b26_spec.json"
MANIFEST=REPO/"configs/b26/b26_dev80_manifest.json"
CPU=REPO/"scripts/b26/run_b26_cpu_estimator.py"
GPU=REPO/"scripts/b26/run_b26_np_gpu_worker.py"

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--require-manifest",action="store_true"); a=ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise RuntimeError("tests require CUDA_VISIBLE_DEVICES='' exactly")
    spec=json.loads(SPEC.read_text())
    assert spec["signed_b25_base"]=="c906d36e0e396a6abbf761e3d65c91433428170f"
    assert spec["gpu"]["hard_process_group_ceiling_mib"]==52452
    assert spec["gpu"]["min_free_mib"]==10240
    assert spec["b26_1"]["arms"]["H"]=={"score_target":"y_plus","projection_target":"y_plus"}
    assert spec["b26_1"]["arms"]["R"]=={"score_target":"y_raw","projection_target":"y_plus"}
    assert spec["b26_1"]["max_scientific_trajectories"]==640
    assert spec["hard_boundaries"]["confirmation_payload_access"] is False
    assert spec["hard_boundaries"]["new_ffhq_measurements"] is False
    assert spec["hard_boundaries"]["weighted_ffhq_reconstruction"] is False

    cpu=load("b26_cpu_test",CPU)
    t,pi=cpu.templates_for("ambiguity_unequal"); meas=cpu.forward(t)
    rel=np.linalg.norm(meas[0]-meas[1])/max(np.linalg.norm(meas[0]),1e-300)
    assert rel<1e-12
    x=np.array([[-10000.,-10001.,-10002.],[-1.,-2.,-3.]])
    s=cpu.softmax(x); assert np.all(np.isfinite(s)) and np.allclose(s.sum(1),1.)
    r1=cpu.rng("DETERMINISM"); r2=cpu.rng("DETERMINISM"); assert np.array_equal(r1.integers(2**31,size=32),r2.integers(2**31,size=32))
    lv=np.array([[0.,0.,-10.]]*64); rt=cpu.rng("TIE_TEST"); choices=cpu.uniform_hard_choice(lv,rt)
    assert set(np.unique(choices))<={0,1} and len(set(choices.tolist()))==2
    # Arithmetic likelihood mean differs from exp(mean log-likelihood); guard implementation token.
    src=CPU.read_text(); assert "logsumexp(vals,axis=1)-math.log(M)" in src

    gsrc=GPU.read_text()
    for token in ("score_target","projection_target","winner_plus","winner_raw","EXPECTED_UNET = 2200","historical replay mismatch"):
        assert token in gsrc
    assert "weighted_ffhq" not in gsrc.lower()

    manifest_checked=False
    if MANIFEST.is_file():
        m=json.loads(MANIFEST.read_text()); manifest_checked=True
        assert m["status"]=="PASS" and m["development_count"]==80 and m["confirmation_registry_count"]==305
        assert m["development_confirmation_overlap"]==0 and m["confirmation_payloads_accessed"] is False
        assert len(m["rows"])==80 and len(m["dev16_image_ids"])==16 and len(m["smoke_image_ids"])==4
        assert {r["screening_stratum"] for r in m["rows"]}==set("ABCD")
        assert all(len(r["np_roots"])==4 and len({x["root_seed"] for x in r["np_roots"]})==4 for r in m["rows"])
    if a.require_manifest and not manifest_checked: raise RuntimeError("B26 pre-run requires frozen DEV80 manifest")
    out={"status":"PASS","gpu_work_performed":False,"pretrained_model_inference_performed":False,"manifest_checked":manifest_checked,"exact_symmetry_relative_l2":float(rel),"stable_log_weights":True,"tie_randomization":True,"rng_determinism":True}
    print(json.dumps(out,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
