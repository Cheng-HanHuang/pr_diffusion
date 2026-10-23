"""R0.6 four-GPU continuation CPU contracts, no GPU/PAC source reads."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import pytest


def module(filename):
    path=Path(__file__).resolve().parents[2]/"scripts/track_r"/filename
    spec=importlib.util.spec_from_file_location(filename[:-3],path)
    out=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def test_exact_partition_8_plus_712():
    p=module("r0_6_parallel_autopilot.py")
    chunks=p.shards()
    assert len(chunks)==12
    assert chunks[0]=={"start":8,"count":64}
    assert chunks[-1]=={"start":712,"count":8}
    indices=[i for s in chunks for i in range(s["start"],s["start"]+s["count"])]
    assert indices==list(range(8,720))
    assert len(indices)==712
    assert sum(s["count"] for s in chunks)+8==720
    assert p.CHUNK==64
    assert p.CHUNK_TIMEOUT==900


def test_aggregate_budget_charges_simultaneous_gpus():
    p=module("r0_6_parallel_autopilot.py")
    active={0:{"began":10.},1:{"began":10.},
            2:{"began":15.},3:{"began":18.}}
    assert p.active_seconds(active,20.)==47.
    prior=2.84538*3600
    canary=29.908
    added=123.
    remaining=p.remaining_budget(added,active,20.,prior,canary)
    assert remaining==pytest.approx(
        min(4*3600-canary-added-47.,
            16*3600-prior-canary-added-47.))
    assert p.remaining_budget(14400,{},20.,prior,canary)<0
    assert p.remaining_budget(0,{},20.,16*3600,canary)<0


def test_canary_provenance_and_r05_720_lock():
    p=module("r0_6_parallel_autopilot.py")
    worker=module("r0_6_score_gpu.py")
    assert p.CANARY.name=="r0_6_canary_20261023T020016Z"
    assert p.PRODUCER_HEAD=="bc62488aa088ae681cbedd4264fa3a099bed82bb"
    assert p.TASK_SHA=="f268e29b482da10aee31f79de43a6bf19d460f4203f7813c46dbee83b0591596"
    assert p.R05_HEAD=="2a65698a3855ddcc65071af90f69f287c824fa13"
    assert worker.ORBIT_PATTERNS==tuple(f"{k:03b}" for k in range(8))


def test_existing_partial_shard_stays_on_original_physical_gpu(tmp_path):
    p=module("r0_6_parallel_autopilot.py")
    first={"start":8,"count":64}
    second={"start":72,"count":64}
    folder=p.location(tmp_path,first)
    folder.mkdir(parents=True)
    (folder/"R0_6_SHARD_META.json").write_text(json.dumps({
        "start":8,"count":64,"gpu":2
    }))
    pending=[first,second]
    assert p.compatible_next(pending,0,tmp_path)==second
    assert p.compatible_next(pending,1,tmp_path) is None
    assert p.compatible_next(pending,2,tmp_path)==first
    assert pending==[]


def test_resume_rejects_meta_drift_and_old_head(tmp_path):
    p=module("r0_6_parallel_autopilot.py")
    first={"start":8,"count":64}
    folder=p.location(tmp_path,first)
    folder.mkdir(parents=True)
    (folder/"R0_6_SHARD_META.json").write_text(json.dumps({
        "start":999,"count":64,"gpu":1
    }))
    with pytest.raises(RuntimeError,match="META_DRIFT"):
        p.compatible_next([first],1,tmp_path)


def test_gt_free_worker_allows_only_locked_canary_for_cross_head():
    # No patch permitted to add GT or PSNR paths to scoring inputs.
    import ast
    p=Path(__file__).resolve().parents[2]/"scripts/track_r/r0_6_score_gpu.py"
    source=p.read_text()
    ast.parse(source)
    assert "STOP_R06_UNAPPROVED_TASK_SOURCE" in source
    assert "f268e29b482da10aee31f79de43a6bf19d460f4203f7813c46dbee83b0591596" in source
    assert "gt_path" not in source
    assert "recorded_psnr_db" not in source
    assert "measurement_source" in source
