#!/usr/bin/env python3
"""A1.3.0 source/byte integrity gate. No candidate pixels/tensors loaded.

Never scans outside explicit eight-ID, 128-receipt whitelist. Strictly emits
new artifacts in A1.3 output only, after all prerequisite checks have passed.
A1.2 and all historical trees are read-only. All failures STOP closed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.track_a.a13_physics import require_cpu, source_witness, symmetry_witness

HERE=Path(__file__).resolve().parents[2]
CONFIG=HERE/"configs/track_a/a13_frozen_analysis_v1.json"

def loadj(path):
    with Path(path).open("r",encoding="utf8") as f: data=json.load(f)
    if not isinstance(data,dict):raise ValueError(f"not object: {path}")
    return data

def hashfile(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def chkfile(path,expected_hash=None):
    p=Path(path)
    if not p.is_file() or p.is_symlink():raise RuntimeError(f"STOP missing/symlink file: {p}")
    observed=hashfile(p)
    if expected_hash and observed!=expected_hash:
        raise RuntimeError(f"STOP file SHA drift: {p}")
    return observed

def git(repo,*args):
    r=subprocess.run(["git","-C",str(repo),*args],capture_output=True,text=True,
                     timeout=40,check=False,env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if r.returncode:raise RuntimeError(f"STOP git identity failure {repo}: {args}: {r.stderr[-300:]}")
    return r.stdout.strip()

def native_source_gate(cfg):
    """Frozen numerical source identities, excluding explicitly observed pyc churn.

    The B24 historical full SITCOM git-diff digest was not a pinned A1.2
    execution prerequisite. It included tracked Python3.8 bytecode and changed
    after later interpreter imports, without a change to Fourier physics.
    We pin the reviewed one-line mkdir patch and tracked operator source Git
    blobs, reject other tracked drift, and report (not equate) pyc differences.
    """
    out={}
    for name,head in cfg["native_heads"].items():
        d=Path(cfg["native_paths"][name])
        got=git(d,"rev-parse","HEAD")
        if got!=head:raise RuntimeError(f"STOP native HEAD drift {name}: {got}")
        out[name]={"head":got}
        if name in cfg["native_diff_sha256"]:
            r=subprocess.run(["git","-C",str(d),"diff","HEAD","--"],
                            capture_output=True,timeout=60,check=False,
                            env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
            if r.returncode:raise RuntimeError(f"STOP native diff read {name}")
            digest=hashlib.sha256(r.stdout).hexdigest()
            if digest!=cfg["native_diff_sha256"][name]:
                raise RuntimeError(f"STOP native dirty patch mismatch {name}")
            out[name]["tracked_diff_sha256"]=digest

    # SITCOM: fail closed for executable source drift while ignoring ONLY
    # the precise ten tracked Python3.8 bytecode paths observed on PAC.
    name="SITCOM"
    d=Path(cfg["native_paths"][name])
    frozen=cfg["native_source_scope_sitcom"]
    if frozen["source_head"]!=cfg["native_heads"][name]:
        raise RuntimeError("STOP conflicting SITCOM source head pins")
    cp=subprocess.run(["git","-C",str(d),"diff","--name-only","-z","HEAD","--"],
                      capture_output=True,timeout=60,check=False,
                      env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if cp.returncode:raise RuntimeError("STOP SITCOM tracked change roster unreadable")
    changed={p.decode("utf-8") for p in cp.stdout.split(bytes([0])) if p}
    expected_sources=set(frozen["allowed_tracked_modified_source"])
    allowed_pyc=set(frozen["allowed_modified_bytecode_paths"])
    if changed-expected_sources-allowed_pyc or not expected_sources.issubset(changed):
        raise RuntimeError("STOP SITCOM unreviewed tracked modifications: "
                           +repr(sorted(changed-expected_sources-allowed_pyc))
                           +" missing_required="+repr(sorted(expected_sources-changed)))
    if any(not p.endswith(".cpython-38.pyc") for p in changed&allowed_pyc):
        raise RuntimeError("STOP SITCOM unexpected tracked bytecode suffix")
    if expected_sources!={"posterior_sample.py"}:
        raise RuntimeError("STOP SITCOM unrecognized reviewed source list")

    sampler="posterior_sample.py"
    sample_blob=git(d,"hash-object",str(d/sampler))
    if sample_blob!=frozen["expected_posterior_sample_git_blob"]:
        raise RuntimeError("STOP SITCOM posterior_sample.py content drift")
    patch=subprocess.run(["git","-C",str(d),"diff","--no-ext-diff",
                          "HEAD","--",sampler],
                         capture_output=True,timeout=60,check=False,
                         env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if patch.returncode:raise RuntimeError("STOP SITCOM posterior patch unreadable")
    patch_digest=hashlib.sha256(patch.stdout).hexdigest()
    if patch_digest!=frozen["expected_posterior_sample_patch_sha256"]:
        raise RuntimeError("STOP SITCOM posterior source patch mismatch")

    operators={}
    for rel,expected_blob in frozen["expected_forward_operator_git_blobs"].items():
        operator=d/rel
        if not operator.is_file() or operator.is_symlink():
            raise RuntimeError(f"STOP SITCOM operator source missing/symlink: {rel}")
        actual=git(d,"hash-object",str(operator))
        head_blob=git(d,"rev-parse",f"HEAD:{rel}")
        if actual!=expected_blob or head_blob!=expected_blob:
            raise RuntimeError(f"STOP SITCOM operator source blob drift: {rel}")
        operators[rel]=actual

    # Guard against Python source shadowing inside native forward_operator.
    extras=git(d,"ls-files","--others","--exclude-standard","--","forward_operator")
    unexpected=[p for p in extras.splitlines()
                if p.endswith((".py",".so",".pyd",".pyi"))]
    if unexpected:
        raise RuntimeError("STOP SITCOM untracked importable operator source: "
                           +repr(unexpected[:8]))

    whole=subprocess.run(["git","-C",str(d),"diff","HEAD","--"],
                         capture_output=True,timeout=60,check=False,
                         env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if whole.returncode:raise RuntimeError("STOP SITCOM diagnostic diff unreadable")
    out[name].update(
        source_gate="reviewed_source_and_operator_blobs_PASS",
        reviewed_posterior_sample_git_blob=sample_blob,
        reviewed_posterior_patch_sha256=patch_digest,
        forward_operator_git_blobs=operators,
        tracked_modified_bytecode_paths=sorted(changed&allowed_pyc),
        full_tracked_diff_sha256_diagnostic_only=hashlib.sha256(whole.stdout).hexdigest(),
        historical_full_diff_not_used_as_a1_execution_pin=True)

    a1=Path(cfg["root"])/"pr_diffusion_track_a_a1_oracle_budget"
    a1file=a1/"scripts/track_a/a1_2_k4_exposed_b22.py"
    pin=cfg["a1_execution_source"]
    blob=git(a1,"rev-parse",f"{pin}:scripts/track_a/a1_2_k4_exposed_b22.py")
    actual=git(a1,"hash-object",str(a1file))
    if blob!=actual:raise RuntimeError("STOP A1.2 exact execution-controller blob drift")
    out["a1_execution_commit"]=pin
    out["a1_execution_controller_blob"]=blob
    return out

def checkout_gate(cfg,head):
    target=Path(cfg["a13_worktree"])
    if HERE!=target or git(HERE,"branch","--show-current")!="codex/prdiff-track-a-a13-candidate-audit-20261020":
        raise RuntimeError("STOP unexpected A1.3 source worktree/branch")
    if git(HERE,"rev-parse","HEAD")!=head:raise RuntimeError("STOP A1.3 source HEAD mismatch")
    if git(HERE,"status","--porcelain","--untracked-files=no"):
        raise RuntimeError("STOP dirty A1.3 tracked source")
    return head

def require_fixed_paths(cfg,args):
    run=Path(args.run_root)
    inp=Path(args.input_root)
    out=Path(args.out)
    expected_run=Path(cfg["run_root"])
    expected_inp=Path(cfg["input_root"])
    parent=Path(cfg["output_parent"])
    if run!=expected_run or inp!=expected_inp:
        raise RuntimeError("STOP unexpected A1.2 input roots")
    if out!=parent/cfg["inventory_stage"]:
        raise RuntimeError("STOP inventory output path outside reserved stage")
    if out.exists() or out.is_symlink():
        raise RuntimeError("STOP inventory stage already exists; do not overwrite")
    if not run.is_dir() or not inp.is_dir() or not parent.parent.is_dir():
        raise RuntimeError("STOP frozen inputs/parent missing")
    if parent.exists() and (parent.is_symlink() or any(parent.iterdir())):
        raise RuntimeError("STOP A1.3 parent occupied; investigate, no overwrite")
    return run,inp,out

def operator_gate(cfg):
    """CPU numerical comparison to frozen DAPS and SITCOM native implementations."""
    torch=require_cpu()
    from scripts.track_a.a13_physics import magnitude
    import importlib
    import sys
    from scripts.track_a.a13_physics import synthetic_noise_reference
    noise_cfg=cfg["noise_reference"]
    if noise_cfg["consistent_multiplier"]!=1 or noise_cfg["poor_fit_multiplier"]!=3:
        raise RuntimeError("STOP invalid preregistered noise cutoff definition")
    result={"source_witness":source_witness(),"symmetry_witness":symmetry_witness(),
            "noise_reference":synthetic_noise_reference(
                sigma=noise_cfg["sigma_y"],seed=noise_cfg["seed"],draws=noise_cfg["draws"])}
    x=torch.full((1,3,256,256),-.67,dtype=torch.float32)
    x[:,:,11,17]=.8
    x[:,:,141,212]=-.12
    reference=magnitude(x)
    for name in ("DAPS","SITCOM"):
        root=Path(cfg["native_paths"][name])
        # Only import native forward_operator; no sampler/UNet/checkpoint.
        sys.path.insert(0,str(root))
        try:
            for key in tuple(sys.modules):
                if key=="forward_operator" or key.startswith("forward_operator."):
                    del sys.modules[key]
            module=importlib.import_module("forward_operator")
            op=module.PhaseRetrieval(oversample=2.0,resolution=256,sigma=.05)
            with torch.no_grad():native=op(x)
            if native.device.type!="cpu" or native.dtype!=torch.float32 or native.shape!=reference.shape:
                raise RuntimeError(f"STOP OPERATOR_NOT_RECONCILED {name} shape/type/device")
            rel=float(torch.linalg.vector_norm((native-reference).double())/
                      torch.linalg.vector_norm(reference.double()).clamp_min(1e-12))
            if not bool(torch.isfinite(native).all()) or rel>2e-5:
                raise RuntimeError(f"STOP OPERATOR_NOT_RECONCILED {name}: {rel}")
            result[name]={"relative_error":rel,"native_class":"forward_operator.PhaseRetrieval",
                          "numeric_reconciled":True}
        finally:
            sys.path.remove(str(root))
            for key in tuple(sys.modules):
                if key=="forward_operator" or key.startswith("forward_operator."):
                    del sys.modules[key]
    return result

def inventory(cfg,run,inp):
    ids=cfg["ids"]; methods=cfg["methods"]
    fpath=inp/"A1_2_INPUT_FREEZE_COMPLETE.json"
    chkfile(fpath,cfg["input_freeze_file_sha256"])
    frozen=loadj(fpath)
    if frozen.get("schema")!="prdiff.a1.2.exposed-input-freeze.v1" or \
       frozen.get("image_ids")!=ids or frozen.get("selected_id_list_sha256")!=cfg["list_sha256"] or \
       frozen.get("protected_confirmation_access") is not False or len(frozen.get("rows",[]))!=8:
        raise RuntimeError("STOP A1 input freeze identity")
    completion=loadj(run/"A1_2_COMPLETE.json")
    identity=loadj(run/"RUN_IDENTITY.json")
    historical=loadj(run/"A1_2_OFFLINE_ORACLE_SUMMARY.json")
    if completion.get("status")!="PASS_A1_2_EXPOSED_B22_128_COMPLETE" or \
       completion.get("head")!=cfg["a1_execution_source"] or \
       completion.get("candidate_count")!=128 or completion.get("image_ids")!=ids or \
       completion.get("method_counts")!={m:32 for m in methods} or \
       identity.get("a1_head")!=cfg["a1_execution_source"]:
        raise RuntimeError("STOP A1 completion/identity drift")
    if set(historical.get("per_image",{}))!=set(ids) or \
       historical.get("score_representation")!="COMMON_SAVED_PNG8_RGB_RAW_PRIMARY":
        raise RuntimeError("STOP oracle summary identity")
    registry=identity.get("candidate_registry",[])
    if len(registry)!=128:raise RuntimeError("STOP A1 registry size")
    registered={v["candidate_id"]:v for v in registry}
    if len(registered)!=128:raise RuntimeError("STOP repeated registered candidate")
    table=[]; input_checks=[]
    for image_id, row in zip(ids,frozen["rows"]):
        if row.get("image_id")!=image_id or row.get("measurement_shape")!=[1,3,384,384]:
            raise RuntimeError(f"STOP input schema {image_id}")
        gt=inp/image_id/"ground_truth_tensor.pt"
        if Path(row["ground_truth_tensor_path"])!=gt:
            raise RuntimeError(f"STOP GT path {image_id}")
        y=Path(row["measurement_path"])
        permitted=Path(cfg["root"])/"outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401/measurements"/f"ffhq{image_id}_phase_noise005_meas5401.pt"
        if y!=permitted:raise RuntimeError(f"STOP unexpected observation {image_id}")
        input_checks.append(dict(image_id=image_id,ground_truth_path=str(gt),
            ground_truth_file_sha256=chkfile(gt,row["ground_truth_tensor_file_sha256"]),
            measurement_path=str(y),measurement_file_sha256=chkfile(y,row["measurement_file_sha256"]),
            measurement_tensor_sha256=row["measurement_tensor_sha256"],
            ground_truth_tensor_sha256=row["ground_truth_tensor_sha256"]))
        perimage=loadj(run/image_id/"IMAGE_CANDIDATES_COMPLETE.json")
        if perimage.get("image_id")!=image_id or len(perimage.get("candidates",[]))!=16:
            raise RuntimeError(f"STOP candidate block {image_id}")
        fromlist={x["candidate_id"]:x for x in perimage["candidates"]}
        if len(fromlist)!=16:raise RuntimeError(f"STOP per-image duplicate {image_id}")
        archived={x["candidate_id"]:x for x in historical["per_image"][image_id]["candidate_metrics"]}
        for method in methods:
            for rep in cfg["replicas"]:
                cid=f"{image_id}:{method}:{rep}"
                recfile=run/image_id/f"{method}_rep{rep}_COMPLETE.json"
                rec=loadj(recfile)
                if cid not in fromlist or cid not in registered or cid not in archived:
                    raise RuntimeError(f"STOP missing candidate registry {cid}")
                if any(rec.get(k)!=fromlist[cid].get(k) or rec.get(k)!=registered[cid].get(k)
                       for k in ("candidate_id","image_id","method","rep","native_seed32",
                                 "measurement_tensor_sha256","ground_truth_tensor_sha256")):
                    raise RuntimeError(f"STOP receipt and seed mismatch {cid}")
                if rec.get("measurement_tensor_sha256")!=row["measurement_tensor_sha256"] or \
                   rec.get("ground_truth_tensor_sha256")!=row["ground_truth_tensor_sha256"]:
                    raise RuntimeError(f"STOP input content mismatch {cid}")
                term=Path(rec["terminal"])
                # Only exact contained terminal path; do not search for alternative files.
                subtree=run/image_id/method.lower()/f"rep{rep}"
                if term.parent!=subtree and not term.is_relative_to(subtree):
                    raise RuntimeError(f"STOP terminal escaped candidate subtree {cid}")
                h=chkfile(term,rec["terminal_sha256"])
                png=term if method=="DAPS" else term.with_name("reconstruction.png")
                ph=chkfile(png,archived[cid]["png_sha256"])
                timer=Path(rec["timing_json"])
                if timer!=subtree/"cuda_timing.json" or rec.get("timing_content_sha256")!=chkfile(timer):
                    raise RuntimeError(f"STOP timing digest {cid}")
                if loadj(timer).get("status")!=0:raise RuntimeError(f"STOP timer status {cid}")
                if not isinstance(rec.get("wall_seconds_subprocess"),(int,float)) or \
                   not isinstance(rec.get("gpu_active_seconds_event"),(int,float)):
                    raise RuntimeError(f"STOP time missing {cid}")
                table.append(dict(image_id=image_id,method=method,rep=rep,
                    candidate_id=cid,seed32=rec["native_seed32"],terminal=str(term),
                    terminal_sha256=h,png=str(png),png_sha256=ph,
                    receipt=str(recfile),receipt_sha256=chkfile(recfile),
                    timer=str(timer),timer_sha256=rec["timing_content_sha256"],
                    wall_seconds_subprocess=rec["wall_seconds_subprocess"],
                    gpu_active_seconds_event=rec["gpu_active_seconds_event"],
                    stored_psnr_raw_db=archived[cid]["raw_psnr_db"],
                    stored_psnr_rot180_db=archived[cid]["rot180_psnr_db"]))
    if len(table)!=128 or len({r["candidate_id"] for r in table})!=128 or \
       len({r["terminal_sha256"] for r in table})!=128 or \
       Counter(r["method"] for r in table)!={m:32 for m in methods}:
        raise RuntimeError("STOP 128 content/identity uniqueness failure")
    return dict(status="A13_SOURCE_AND_INPUT_AUDIT_PASS",scope="EIGHT_ALREADY_EXPOSED",
                candidate_count=128,input_checks=input_checks,candidates=table,
                a1_completion_file_sha256=chkfile(run/"A1_2_COMPLETE.json"),
                a1_identity_sha256=chkfile(run/"RUN_IDENTITY.json"),
                a1_oracle_summary_sha256=chkfile(run/"A1_2_OFFLINE_ORACLE_SUMMARY.json"),
                input_freeze_file_sha256=cfg["input_freeze_file_sha256"])

def save_new(path,data):
    with Path(path).open("x",encoding="utf8") as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["inventory"],required=True)
    p.add_argument("--run-root",required=True)
    p.add_argument("--input-root",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--expected-head",required=True)
    args=p.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    cfg=loadj(CONFIG)
    run,inp,out=require_fixed_paths(cfg,args)
    checkout_gate(cfg,args.expected_head)
    t0=time.perf_counter()
    source=native_source_gate(cfg)
    # Strong gate: native CPU operator must numerically match a separate formula.
    op=operator_gate(cfg)
    rows=inventory(cfg,run,inp)
    rows.update(operator_test=op,source=source,a13_commit=args.expected_head,
                elapsed_cpu_wall_seconds=time.perf_counter()-t0,
                no_gpu=True,no_diffusion=True,no_new_measurements=True,
                no_new_data=True)
    # All validation has completed. First creation is restricted to exclusive tree.
    parent=out.parent
    if not parent.exists():parent.mkdir(exist_ok=False)
    out.mkdir(exist_ok=False)
    save_new(out/"A13_SOURCE_AND_INPUT_AUDIT.json",rows)
    save_new(out/"A13_INVENTORY_COMPLETE.json",{
        "status":rows["status"],"candidate_count":128,
        "source_head":args.expected_head,
        "audit_file_sha256":chkfile(out/"A13_SOURCE_AND_INPUT_AUDIT.json"),
        "operator_numeric_gate":"PASS","stage":"a13_0_inventory_v1"})
    print(json.dumps(dict(status=rows["status"],candidate_count=128,
        source_head=args.expected_head,operator="PASS",output=str(out)),sort_keys=True))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print(f"A13_INVENTORY_STOP: {type(e).__name__}: {e}",file=sys.stderr)
        sys.exit(7)
