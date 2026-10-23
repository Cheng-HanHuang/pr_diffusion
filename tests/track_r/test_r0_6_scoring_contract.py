"""R0.6 scorer source lineage and sealed GT-free scoring contracts.

No PAC data or model access; dynamic imports must work without CUDA.
"""
from __future__ import annotations
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def module(name):
    p=Path(__file__).resolve().parents[2]/"scripts/track_r"/name
    spec=importlib.util.spec_from_file_location(name[:-3],p)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(i):
    return dict(uid=f"S5_r0_fast|FAST|S1_b22_{60044+i//4}|rep{i%4}",
        source="S5_r0_fast",family="FAST",
        image_id=str(60044+i//4),run_index=i%4,
        measurement_source="S1_b22",
        candidate_path="/tmp/exposed/reconstruction.png",
        candidate_format="png",candidate_sha256="a"*64,
        measurement_path="/tmp/signed_observation.pt",
        measurement_sha256="b"*64)


def test_r06_gt_free_score_tasks_valid_and_extra_gt_rejected():
    score=module("r0_6_score_gpu.py")
    inputs=[row(i) for i in range(720)]
    payload={"schema":"prdiff.track_r.r0.6.score-tasks.v1",
        "status":"R0_6_TASKS_FROZEN","head":"test",
        "n_candidates":720,"gt_or_psnr_fields_in_gpu_tasks":False,
        "tasks":inputs}
    assert len(score.validate_input(payload,"test"))==720
    dirty=dict(payload)
    dirty["tasks"]=[dict(inputs[0],gt_path="/tmp/private.pt")]+inputs[1:]
    with pytest.raises(RuntimeError,match="NOT_GT_FREE"):
        score.validate_input(dirty,"test")
    bad=dict(payload,gt_or_psnr_fields_in_gpu_tasks=True)
    with pytest.raises(RuntimeError,match="BAD_TASKS_RECEIPT"):
        score.validate_input(bad,"test")


def test_r06_per_candidate_receipt_binds_sealed_task_and_source(tmp_path):
    score=module("r0_6_score_gpu.py")
    sample=row(0)
    fake={"status":"R0_6_CANDIDATE_PASS",
        "uid":sample["uid"],"candidate_sha256":sample["candidate_sha256"],
        "task_file_sha256":"c"*64,"executor_head":"head",
        "unet_forwards":256,
        "p_fake_orbit":{f"{k:03b}":0.05*k for k in range(8)}}
    p=tmp_path/"item.json"
    p.write_text(json.dumps(fake))
    assert score.validate_complete(p,sample,"c"*64,"head")
    fake["task_file_sha256"]="d"*64
    p.write_text(json.dumps(fake))
    with pytest.raises(RuntimeError,match="EXISTING_RESULT_INVALID"):
        score.validate_complete(p,sample,"c"*64,"head")


def test_r06_orbit_batch_score_stays_authoritative():
    score=module("r0_6_score_gpu.py")
    values=[0.10+0.02*k for k in range(8)]
    result=score.reconcile_univfd_batch_score(
        values,4,values[4]+5e-5,
        [a+5e-8 for a in values])
    assert result["canonical"]==values[4]
    assert result["diagnosis"]=="BATCH_SHAPE_FP_DIFFERENCE"
    with pytest.raises(RuntimeError,match="REAL_INSTABILITY"):
        score.reconcile_univfd_batch_score(
            values,4,values[4]+0.002,values)


def test_r06_lock_sources_and_bound_stage():
    prep=module("r0_6_prepare_cpu.py")
    canary=module("r0_6_canary_autopilot.py")
    assert prep.CANARY_TASK_SHA=="2291aa793b612c973ae76ab3d36b1f20263cd747f3b23cf593e51517a26c3bfe"
    assert prep.R05_HEAD=="2a65698a3855ddcc65071af90f69f287c824fa13"
    assert "gt_path" not in prep.EXPECTED_FIELDS
    assert "measurement_source" in prep.EXPECTED_FIELDS
    assert canary.KNOWN_STAGE_LIMIT_S==14400
    assert canary.GLOBAL_R0_LIMIT_S==57600
