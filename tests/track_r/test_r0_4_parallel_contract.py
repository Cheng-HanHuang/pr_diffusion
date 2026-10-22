"""CPU-only four-GPU R0.4 scheduler, partition, and budget tests."""
from __future__ import annotations
import importlib.util
from pathlib import Path


def scheduler():
    file=Path(__file__).resolve().parents[2]/"scripts/track_r/r0_4_parallel_autopilot.py"
    spec=importlib.util.spec_from_file_location("r04_parallel",file)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_four_way_partition_covers_every_unscored_candidate_once():
    m=scheduler()
    chunks=m.chunks()
    assert len(chunks)==40
    assert chunks[0]=={"start":16,"count":64}
    assert chunks[-1]=={"start":2512,"count":8}
    assert all(1<=x["count"]<=64 for x in chunks)
    covered=[i for x in chunks for i in range(x["start"],x["start"]+x["count"])]
    assert covered==list(range(16,2520))
    assert len(covered)==2504
    assert all(chunks[i]["start"]<chunks[i+1]["start"]
               for i in range(len(chunks)-1))


def test_parallel_aggregate_gpu_seconds_counts_simultaneous_slots():
    m=scheduler()
    slots={0:{"start_monotonic":10.},1:{"start_monotonic":10.},
           2:{"start_monotonic":12.},3:{"start_monotonic":15.}}
    assert m.active_seconds(slots,20.)==33.
    assert m.stage_deadline_remaining(100.,slots,20.,54.)==(
        m.STAGE_LIMIT_SECONDS-54.-100.-33.)
    assert m.valid_budget(100.,slots,20.,54.)
    assert not m.valid_budget(m.STAGE_LIMIT_SECONDS-50.,slots,20.,54.)


def test_stage_allotment_and_prior_gpu_use_remain_bounded():
    m=scheduler()
    assert m.STAGE_LIMIT_SECONDS==6*3600
    assert m.GLOBAL_LIMIT_SECONDS==16*3600
    assert m.KNOWN_EARLIER_GPU_SECONDS>0
    assert m.KNOWN_EARLIER_GPU_SECONDS<m.GLOBAL_LIMIT_SECONDS
    assert m.CHUNK_TIMEOUT==900
    assert m.POLL<=5
    assert m.PRODUCER_HEAD=="8aee60d98acfeba836fff4ec824137769b0a4698"
    assert len(m.TASK_SHA)==64


def test_exact_existing_canary_task_source_does_not_change():
    import ast
    m=scheduler()
    parent=Path(__file__).resolve().parents[2]
    worker=parent/"scripts/track_r/r0_4_score_gpu.py"
    ast.parse(worker.read_text())
    ast.parse((parent/"scripts/track_r/r0_4_parallel_autopilot.py").read_text())
    assert m.CANARY.name=="r0_4_score_20261022T193238Z"
