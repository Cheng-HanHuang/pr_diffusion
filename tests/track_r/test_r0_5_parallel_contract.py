"""R0.5 native four-GPU continuation contracts (no PAC data/GPU)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import json
import pytest


def script(name):
    path=Path(__file__).resolve().parents[2]/"scripts/track_r"/name
    spec=importlib.util.spec_from_file_location(name[:-3],path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_full_720_no_overlap_and_four_independent_reps():
    m=script("r0_5_parallel_autopilot.py")
    chunks=m.chunks()
    assert len(chunks)==30
    assert chunks[0]=={"start":8,"count":24}
    assert chunks[-1]=={"start":704,"count":16}
    assert all(c["start"]%4==0 and c["count"]%4==0 and
               4<=c["count"]<=24 for c in chunks)
    result=[j for c in chunks for j in range(
        c["start"],c["start"]+c["count"])]
    assert result==list(range(8,720))
    assert len(result)==712


def test_concurrent_gpu_seconds_sum_not_wall_clock():
    m=script("r0_5_parallel_autopilot.py")
    active={0:{"started":100},1:{"started":100},
            2:{"started":110},3:{"started":120}}
    assert m.active_seconds(active,130)==90.
    # r04 budget = 2h, 8-root R05 canary = 35s;
    # the total counts every simultaneous GPU.
    allowed=m.budget_remaining(100,active,130,7200,35)
    assert allowed==pytest.approx(m.R05_STAGE_CAP_SECONDS-35-100-90)
    assert m.budget_remaining(m.R05_STAGE_CAP_SECONDS,{},130,
                               7200,35)<0
    assert m.GLOBAL_CAP_SECONDS==57600
    assert m.R05_STAGE_CAP_SECONDS==14400


def test_source_manifest_sha_and_canary_id_are_immutable():
    m=script("r0_5_parallel_autopilot.py")
    worker=script("r0_5_fast_gpu.py")
    assert worker.CANARY_HEAD==m.CANARY_HEAD
    assert worker.CANARY_TASK_SHA==m.ROOT_SHA
    assert worker.CANARY_TASK_PATH==m.CANARY/"prepare/R0_5_GT_FREE_ROOTS.json"
    assert m.ROOT_SHA=="2291aa793b612c973ae76ab3d36b1f20263cd747f3b23cf593e51517a26c3bfe"
    assert m.R04_HEAD=="62eaaafe3e22cd6222fc66858cad6dcef532a2cd"


def test_metadata_partial_retry_does_not_change_gpu(tmp_path):
    m=script("r0_5_parallel_autopilot.py")
    first={"start":8,"count":24}
    second={"start":32,"count":24}
    root=tmp_path
    folder=m.shard_path(root,first)
    folder.mkdir(parents=True)
    (folder/"R0_5_CHUNK_META.json").write_text(json.dumps({
        "start":8,"count":24,"gpu":2
    }))
    remaining=[first,second]
    assert m.compatible_next(remaining,0,root)==second
    assert remaining==[first]
    assert m.compatible_next(remaining,1,root) is None
    assert m.compatible_next(remaining,2,root)==first
    assert remaining==[]


def test_canary_is_read_only_not_duplicate():
    m=script("r0_5_parallel_autopilot.py")
    assert m.CANARY.name=="r0_5_canary_20261023T004927Z"
    assert m.chunks()[0]["start"]==8
    assert sum(x["count"] for x in m.chunks())+8==720
