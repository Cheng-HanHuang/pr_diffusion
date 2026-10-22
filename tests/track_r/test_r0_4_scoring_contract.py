"""Synthetic R0.4 GT-free execution-contract tests (no PAC/GPU files)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load(filename):
    p=Path(__file__).resolve().parents[2]/"scripts/track_r"/filename
    spec=importlib.util.spec_from_file_location(filename[:-3],p)
    x=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x)
    return x


def task(uid):
    return dict(uid=uid,source="S1_b22",family="DAPS",
        image_id="60044",run_index=0,
        candidate_path="/tmp/exposed.png",candidate_format="png",
        candidate_sha256="a"*64,
        measurement_path="/tmp/signed_y.pt",
        measurement_sha256="b"*64)


def test_r04_gpu_worker_rejects_gt_psnr_fields():
    worker=load("r0_4_score_gpu.py")
    rows=[task(str(i)) for i in range(2520)]
    payload=dict(schema="prdiff.track_r.r0.4.gt_free_tasks.v1",
         status="R0_4_TASKS_FROZEN",head="test",
         n_candidates=2520,gt_or_psnr_fields_in_gpu_tasks=False,
         tasks=rows)
    assert len(worker.validate_input(payload,"test"))==2520
    altered=dict(payload)
    altered["tasks"]=[dict(rows[0],gt_path="/tmp/g.png")]+rows[1:]
    with pytest.raises(RuntimeError,match="NOT_GT_FREE"):
        worker.validate_input(altered,"test")
    bad=dict(payload,gt_or_psnr_fields_in_gpu_tasks=True)
    with pytest.raises(RuntimeError,match="BAD_TASKS_RECEIPT"):
        worker.validate_input(bad,"test")


def test_r04_result_names_deterministic_and_unique():
    worker=load("r0_4_score_gpu.py")
    a=worker.output_name(task("candidate-a"))
    b=worker.output_name(task("candidate-b"))
    assert a!=b and len(a)==69 and a.endswith(".json")
    assert a==worker.output_name(task("candidate-a"))


def test_r04_existing_receipt_must_bind_identity_and_scores(tmp_path):
    import json
    worker=load("r0_4_score_gpu.py")
    row=task("uid-a")
    result=dict(status="R0_4_CANDIDATE_PASS",
        uid="uid-a",candidate_sha256=row["candidate_sha256"],
        task_file_sha256="0"*64,
        executor_head="head",unet_forwards=256,
        p_fake_orbit={p:0.5 for p in ("000","001","010","011",
                                   "100","101","110","111")})
    path=tmp_path/"a.json"
    path.write_text(json.dumps(result))
    assert worker.validate_complete(path,row,"0"*64,"head")
    result["unet_forwards"]=255
    path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError,match="EXISTING_RESULT_INVALID"):
        worker.validate_complete(path,row,"0"*64,"head")


def test_r04_stage_budget_and_shard_cap_are_strict():
    auto=load("r0_4_autopilot.py")
    assert auto.ALLOTMENT_GPU_H==6.
    assert auto.CHUNK==64 and auto.MAX_CHUNK_SECONDS<=900
    assert auto.ALLOTMENT_GPU_H<16.


def test_r04_preparation_disallows_unexpected_source():
    prep=load("r0_4_prepare_cpu.py")
    assert "gt_path" not in prep.FIELDS
    assert "recorded_psnr_db" not in prep.FIELDS
    assert "measurement_path" in prep.FIELDS
    assert "candidate_sha256" in prep.FIELDS
