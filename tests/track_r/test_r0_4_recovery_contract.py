"""CPU-only tests of UnivFD batch-shape drift and exact recovery partition."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


def script(name):
    p=Path(__file__).resolve().parents[2]/"scripts/track_r"/name
    spec=importlib.util.spec_from_file_location(name[:-3],p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_univfd_batch_score_is_authoritative():
    m=script("r0_4_score_gpu.py")
    scores=[0.15+0.01*i for i in range(8)]
    r=m.reconcile_univfd_batch_score(scores,3,scores[3]+1e-6)
    assert not r["needs_repeat"]
    assert r["canonical"]==scores[3]
    assert r["diagnosis"]=="SINGLETON_AGREES"


def test_univfd_batch_shape_drift_requires_same_shape_repeat():
    m=script("r0_4_score_gpu.py")
    scores=[0.15+0.01*i for i in range(8)]
    first=m.reconcile_univfd_batch_score(scores,3,scores[3]+4e-5)
    assert first["needs_repeat"]
    r=m.reconcile_univfd_batch_score(
        scores,3,scores[3]+4e-5,
        [v+2e-7 for v in scores])
    assert not r["needs_repeat"]
    assert r["diagnosis"]=="BATCH_SHAPE_FP_DIFFERENCE"
    assert r["canonical"]==scores[3]
    with pytest.raises(RuntimeError,match="REAL_INSTABILITY"):
        m.reconcile_univfd_batch_score(
            scores,3,scores[3]+4e-5,
            [v+2e-4 for v in scores])
    with pytest.raises(RuntimeError,match="REAL_INSTABILITY"):
        m.reconcile_univfd_batch_score(
            scores,3,scores[3]+2e-3,
            scores)
    with pytest.raises(RuntimeError,match="NONFINITE"):
        m.reconcile_univfd_batch_score(scores,3,float("nan"))


def test_recovery_partition_preserves_45_partial_and_27_shards():
    m=script("r0_4_recovery_autopilot.py")
    rows=m.chunks()
    assert len(rows)==13
    assert rows[0]=={"start":1725,"count":19}
    assert rows[1]=={"start":1808,"count":64}
    assert rows[-1]=={"start":2512,"count":8}
    scored={*range(0,1680),*range(1680,1725),*range(1744,1808)}
    new={i for r in rows for i in range(r["start"],r["start"]+r["count"])}
    assert len(scored)==1789
    assert len(new)==731
    assert not (scored & new)
    assert scored | new==set(range(2520))
    assert m.HISTORICAL_SCORED==1789
    assert m.PRODUCER_HEAD=="8aee60d98acfeba836fff4ec824137769b0a4698"
    assert m.FAILED_HEAD=="717100a17a2326f3fbbccb7cbe782ccf4e90126d"


def test_recovery_preserves_global_budget_and_head():
    m=script("r0_4_recovery_autopilot.py")
    prior=5276.953150671441
    canary=54.005350545048714
    assert m.valid_budget(prior,{},0.,canary)
    assert m.stage_deadline_remaining(prior,{},0.,canary)==pytest.approx(
        6*3600-5330.95850121649,abs=1e-5)
    assert not m.valid_budget(6*3600,{},0.,canary)


def test_recovery_retry_keeps_existing_partial_gpu_affinity(tmp_path):
    import json
    m=script("r0_4_recovery_autopilot.py")
    chunk={"start":1725,"count":19}
    folder=m.result_dir(tmp_path,chunk)
    folder.mkdir(parents=True)
    (folder/"R0_4_SHARD_META.json").write_text(
        json.dumps({"gpu":2,"start":1725,"count":19}))
    other={"start":1808,"count":64}
    items=[chunk,other]
    assert m.take_compatible_chunk(items,0,tmp_path)==other
    assert items==[chunk]
    assert m.take_compatible_chunk(items,1,tmp_path) is None
    assert m.take_compatible_chunk(items,2,tmp_path)==chunk
    assert not items
