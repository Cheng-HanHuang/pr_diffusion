"""S2 DEV80 NP4 source provenance tests, including pilot16 reuse."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[2]/"scripts/track_r/r0_2_build_manifest.py"
    spec=importlib.util.spec_from_file_location("r0_2_sources",path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_case(tmp_path, *, pilot):
    m=module()
    m.ROOT=tmp_path  # synthetic path namespace; real sources remain untouched
    source_kind="B24_3_pilot16_20260913T045410Z" if pilot else "B24_3_dev80_overnight_20260913T082146Z"
    task=tmp_path/source_kind/"workers"/"gpu2"/"row02_24322"
    np_dir=task/"methods"/"NP4_INDEPENDENT"
    np_dir.mkdir(parents=True)
    accepted=np_dir/"result.json"
    receipt={"status":"PASS","arm":"NP4_INDEPENDENT",
             "terminals":[{"root_index":i} for i in range(4)]}
    accepted.write_text(json.dumps(receipt),encoding="utf-8")
    for i in range(4):
        (np_dir/f"terminal_{i}.pt").write_bytes(b"synthetic"+bytes([i]))
    row={
        "accepted_np4_result_path":str(accepted),
        "accepted_np4_result_sha256":hashlib.sha256(accepted.read_bytes()).hexdigest(),
        "pilot16":pilot,
        "np_roots":[{"root_index":i} for i in range(4)],
    }
    complete={"np_results":{"NP4_INDEPENDENT":str(accepted)},
              "np_pilot_result_reused":pilot,
              "np_execution_performed":not pilot}
    return m,row,complete,accepted


@pytest.mark.parametrize("pilot",[True,False])
def test_np_terminal_paths_follow_accepted_historical_source(tmp_path,pilot):
    m,row,complete,accepted=make_case(tmp_path,pilot=pilot)
    rec,terminals,reused=m.s2_accepted_np_source(row,complete,"24322")
    assert rec["arm"]=="NP4_INDEPENDENT"
    assert reused is pilot
    assert all(p.parent==accepted.parent for p in terminals)
    assert [p.name for p in terminals]==[f"terminal_{i}.pt" for i in range(4)]


def test_mismatched_pilot_provenance_or_receipt_hash_stops(tmp_path):
    m,row,complete,accepted=make_case(tmp_path,pilot=True)
    wrong=dict(complete)
    wrong["np_results"]={"NP4_INDEPENDENT":str(accepted.parent/"elsewhere.json")}
    with pytest.raises(RuntimeError,match="STOP_MISSING_INPUT|STOP_S2_NP_ADOPTED_SOURCE_MISMATCH"):
        m.s2_accepted_np_source(row,wrong,"24322")
    wrong=dict(complete);wrong["np_pilot_result_reused"]=False
    with pytest.raises(RuntimeError,match="STOP_S2_NP_PILOT_PROVENANCE"):
        m.s2_accepted_np_source(row,wrong,"24322")
    wrongrow=dict(row);wrongrow["accepted_np4_result_sha256"]="0"*64
    with pytest.raises(RuntimeError,match="STOP_S2_NP_RECEIPT_SHA"):
        m.s2_accepted_np_source(wrongrow,complete,"24322")


def test_missing_candidate_not_substituted(tmp_path):
    m,row,complete,accepted=make_case(tmp_path,pilot=True)
    (accepted.parent/"terminal_2.pt").unlink()
    with pytest.raises(RuntimeError,match="STOP_MISSING_INPUT"):
        m.s2_accepted_np_source(row,complete,"24322")


def test_one_image_smoke_32909_is_exact_approved_reuse(tmp_path):
    m=module()
    m.ROOT=tmp_path
    accepted=(tmp_path/"outputs/pr_diffusion/b24"/
              "B24_3_one_image_smoke_20260913T044345Z"/
              "methods/NP4_INDEPENDENT/result.json")
    accepted.parent.mkdir(parents=True)
    rec={"status":"PASS","arm":"NP4_INDEPENDENT",
         "terminals":[{"terminal_index":i} for i in range(4)]}
    accepted.write_text(json.dumps(rec))
    for i in range(4):
        (accepted.parent/f"terminal_{i}.pt").write_bytes(b"smoke"+bytes([i]))
    row={"accepted_np4_result_path":str(accepted),
         "accepted_np4_result_sha256":hashlib.sha256(accepted.read_bytes()).hexdigest(),
         "pilot16":True,"np_roots":[{"root_index":i} for i in range(4)]}
    complete={"np_results":{"NP4_INDEPENDENT":str(accepted)},
              "np_pilot_result_reused":True,"np_execution_performed":False}
    _,terminals,reused=m.s2_accepted_np_source(row,complete,"32909")
    assert reused and len(terminals)==4
    # A one-image smoke path must never be accepted for an unrelated ID.
    with pytest.raises(RuntimeError,match="STOP_S2_PILOT_SOURCE_ROOT"):
        m.s2_accepted_np_source(row,complete,"24322")


def test_whitelisted_s2_source_families_are_exactly_64_15_1():
    root=Path(__file__).resolve().parents[2]
    frozen=json.loads((root/"configs/b26/b26_dev80_manifest.json").read_text())
    rows=frozen["rows"]
    assert len(rows)==80
    seen={}
    for row in rows:
        path=Path(row["accepted_np4_result_path"])
        parts=path.parts
        names=[x for x in parts if x.startswith("B24_3_")]
        assert len(names)==1
        group=names[0]
        seen[group]=seen.get(group,0)+1
        assert row["method_role"]=="DEVELOPMENT"
        assert bool(row["pilot16"]) == ("dev80_overnight" not in group)
    assert seen=={
        "B24_3_dev80_overnight_20260913T082146Z":64,
        "B24_3_pilot16_20260913T045410Z":15,
        "B24_3_one_image_smoke_20260913T044345Z":1,
    }
