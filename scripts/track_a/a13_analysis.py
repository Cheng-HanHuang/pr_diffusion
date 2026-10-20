#!/usr/bin/env python3
"""A1.3.1 gated CPU-only static analysis of immutable A1.2 saved PNG8 terminals.

No model/GPU, new measurements, native sampler, adaptive selector or GT-fitting.
All selector decisions computed from non-GT feature-only records; GT is loaded
solely after the candidate selector indices are determined.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from scripts.track_a.a13_inventory import (
    CONFIG, checkout_gate, native_source_gate, operator_gate, loadj, hashfile,
    chkfile, inventory, require_fixed_paths
)
from scripts.track_a.a13_physics import (
    require_cpu, magnitude, png_to_model_and_01, png8_gt, psnr_png8,
    residuals, select_min, select_oracle
)

def content_sha(t):
    x=t.detach().cpu().contiguous()
    h=hashlib.sha256()
    header=json.dumps({"dtype":str(x.dtype),"shape":list(x.shape)},
                      sort_keys=True,separators=(",",":")).encode()
    h.update(header);h.update(b"\0");h.update(x.numpy().tobytes(order="C"))
    return h.hexdigest()

def tensor_load(path,keys,shape,dtype):
    torch=require_cpu()
    payload=torch.load(str(path),map_location="cpu",weights_only=True)
    value=payload if isinstance(payload,torch.Tensor) else next(
        (payload.get(k) for k in keys if isinstance(payload,dict) and isinstance(payload.get(k),torch.Tensor)),None)
    if not isinstance(value,torch.Tensor) or value.device.type!="cpu" or \
       tuple(value.shape)!=shape or value.dtype!=dtype or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"STOP unexpected tensor schema/content: {path}")
    return value

def write_csv_x(path,rows):
    if not rows:raise RuntimeError("STOP missing CSV rows")
    with path.open("x",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            if set(r)!=set(rows[0]):raise RuntimeError("inconsistent CSV schema")
            writer.writerow(r)

def write_json_x(path,item):
    with path.open("x",encoding="utf8") as f:
        json.dump(item,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")

def validate_stage0(cfg, inventory_root, requested_head):
    out=Path(inventory_root)
    if out!=Path(cfg["output_parent"])/cfg["inventory_stage"]:
        raise RuntimeError("STOP stage0 unexpected path")
    receipt=loadj(out/"A13_INVENTORY_COMPLETE.json")
    j=out/"A13_SOURCE_AND_INPUT_AUDIT.json"
    if receipt.get("audit_file_sha256")!=chkfile(j) or receipt.get("status")!="A13_SOURCE_AND_INPUT_AUDIT_PASS" or \
       receipt.get("candidate_count")!=128 or receipt.get("operator_numeric_gate")!="PASS" or \
       receipt.get("source_head")!=requested_head:
        raise RuntimeError("STOP stage0 receipt incomplete/changed")
    audit=loadj(j)
    if audit.get("a13_commit")!=requested_head or len(audit.get("candidates",[]))!=128 or \
       not all(audit.get("operator_test",{}).get(n,{}).get("numeric_reconciled") for n in ("DAPS","SITCOM")):
        raise RuntimeError("STOP stage0 evidence invalid")
    return audit

def policy_sets(metrics, ids,methods):
    bykey={(r["image_id"],r["method"],r["rep"]):r for r in metrics}
    for image_id in ids:
        for method in methods:
            four=[bykey[(image_id,method,i)] for i in range(4)]
            for k in (1,2,4):
                yield image_id,f"{method}_K{k}",four[:k]
        mixed=[bykey[(image_id,method,rep)] for rep in range(4) for method in ("NP_LF","NP_S2")]
        for k in (4,8):
            yield image_id,f"NP_MIXED_K{k}",mixed[:k]

def spearman_small(score,quality):
    """Ranks with stable average ties, descriptive n<=8 only."""
    def ranks(data):
        order=sorted(range(len(data)),key=lambda i:data[i])
        out=[0.]*len(data)
        for j in range(len(order)):
            lo=next((i for i in range(j+1) if data[order[i]]==data[order[j]]),j)
            hi=next((i for i in range(len(order)-1,j-1,-1) if data[order[i]]==data[order[j]]),j)
            out[order[j]]=(lo+hi)/2
        return out
    a=ranks(score);b=ranks(quality)
    ma=sum(a)/len(a);mb=sum(b)/len(b)
    den=math.sqrt(sum((v-ma)**2 for v in a)*sum((v-mb)**2 for v in b))
    return ((sum((x-ma)*(y-mb) for x,y in zip(a,b))/den) if den else None)

def run(cfg,source_audit,runroot,output):
    torch=require_cpu()
    gt_and_y={}
    # Selectors/feature extraction cannot reference ground truth. GT tensor is
    # intentionally loaded AFTER all 128 feature records and selections exist.
    raw_features=[]; bytes_examined=0
    for item in source_audit["input_checks"]:
        identifier=item["image_id"]
        y=tensor_load(item["measurement_path"],("measurement","y","observation"),
                      (1,3,384,384),torch.float32)
        if content_sha(y)!=item["measurement_tensor_sha256"]:
            raise RuntimeError(f"STOP signed measurement content SHA {identifier}")
        gt_and_y[identifier]=dict(measurement=y,gt_path=item["ground_truth_path"],
              gt_expected_sha=item["ground_truth_tensor_sha256"])
    for ent in source_audit["candidates"]:
        cid=ent["candidate_id"]; identifier=ent["image_id"]
        # Rehash EVERY original byte file at use-time; no quiet source mutation.
        if chkfile(ent["terminal"],ent["terminal_sha256"])!=ent["terminal_sha256"] or \
           chkfile(ent["png"],ent["png_sha256"])!=ent["png_sha256"]:
            raise RuntimeError(f"STOP terminal mutation {cid}")
        t0=time.perf_counter()
        image,png01=png_to_model_and_01(ent["png"])
        d=residuals(image,gt_and_y[identifier]["measurement"],eps=cfg["operator"]["eps"])
        if not all(math.isfinite(v) for side in ("signed","clipped") for v in
                   (d[side]["relative_l2"],d[side]["sse"],d[side]["rmse"])):
            raise RuntimeError(f"STOP nonfinite external residual {cid}")
        raw_features.append(dict(image_id=identifier,method=ent["method"],rep=ent["rep"],
            candidate_id=cid,terminal_sha256=ent["terminal_sha256"],
            png_sha256=ent["png_sha256"],r_signed=d["signed"]["relative_l2"],
            r_clipped=d["clipped"]["relative_l2"],
            sse_signed=d["signed"]["sse"],sse_clipped=d["clipped"]["sse"],
            rmse_signed=d["signed"]["rmse"],rmse_clipped=d["clipped"]["rmse"],
            signed_ch_sse_json=json.dumps(d["signed"]["per_channel_sse"]),
            clipped_ch_sse_json=json.dumps(d["clipped"]["per_channel_sse"]),
            negative_count=d["negative_count"],negative_fraction=d["negative_fraction"],
            n_meas=d["total_measurements"],
            seed32=ent["seed32"],terminal=ent["terminal"],png=ent["png"],
            wall_seconds_subprocess=ent["wall_seconds_subprocess"],
            gpu_active_seconds_event=ent["gpu_active_seconds_event"],
            extra_cpu_score_wall_s=time.perf_counter()-t0,fft_forward_calls=1))
    if len(raw_features)!=128:raise RuntimeError("STOP incomplete score feature set")
    # Predeclare TWO physically motivated score policies (min signed, min clipped).
    # Selector sees exactly scores and immutable candidate identity, never PSNR/GT.
    chosen={}
    for image_id,pol,subset in policy_sets(raw_features,cfg["ids"],cfg["methods"]):
        for score in ("r_signed","r_clipped"):
            pick=select_min(subset,score)
            chosen[(image_id,pol,score)]=pick["candidate_id"]
    # ORACLE EVALUATION BRANCH START: only now load GT and compute RGB8 metrics.
    metrics=[]
    for ent,f in zip(source_audit["candidates"],raw_features):
        item=gt_and_y[ent["image_id"]]
        if "gt" not in item:
            gt=tensor_load(item["gt_path"],("ground_truth","image","x"),
                           (1,3,256,256),torch.float32)
            if content_sha(gt)!=item["gt_expected_sha"]:
                raise RuntimeError(f"STOP frozen GT tensor SHA {ent['image_id']}")
            item["gt"]=png8_gt(gt)
        _model,png01=png_to_model_and_01(ent["png"])
        raw=psnr_png8(png01,item["gt"])
        rot=psnr_png8(torch.rot90(png01,2,dims=(-2,-1)),item["gt"])
        if abs(raw-ent["stored_psnr_raw_db"])>cfg["selection"]["psnr_replay_tol_db"] or \
           abs(rot-ent["stored_psnr_rot180_db"])>cfg["selection"]["psnr_replay_tol_db"]:
            raise RuntimeError(f"STOP A1 oracle per-candidate PNG8 replay {ent['candidate_id']}: {raw}/{rot}")
        if raw>=25:
            tag="RAW_GOOD25"
        elif rot>=25:
            tag="ROTATION_EQUIVALENT_CANDIDATE"
        else:
            # A reference range for independently known noise has not been
            # prospectively calibrated: do not fabricate a fit threshold.
            tag="UNRESOLVED_LOW_RAW"
        metrics.append(dict(f,psnr_raw_rgb8_db=raw,psnr_rot180_rgb8_db=rot,
                delta_rot180_minus_raw_db=rot-raw,failure_tag=tag))
    # Replay exact A1.2 *per-image* prefixes, NOT just rounded means.
    old=loadj(runroot/"A1_2_OFFLINE_ORACLE_SUMMARY.json")
    p_rows=[]; replay=[]
    for image_id,policy,subset in policy_sets(metrics,cfg["ids"],cfg["methods"]):
        oracle=select_oracle(subset)
        archived=old["per_image"][image_id]["oracle_by_policy"][policy]
        if abs(oracle["psnr_raw_rgb8_db"]-archived["best_psnr_raw_db"])>1e-4 or \
           oracle["candidate_id"]!=archived["best_candidate_id"]:
            raise RuntimeError(f"STOP A1.2 oracle winner mismatch {image_id}:{policy}")
        replay.append(dict(image_id=image_id,policy=policy,status="PASS",oracle_id=oracle["candidate_id"]))
        for score in ("r_signed","r_clipped"):
            cid=chosen[(image_id,policy,score)]
            selected=next(v for v in subset if v["candidate_id"]==cid)
            regret=oracle["psnr_raw_rgb8_db"]-selected["psnr_raw_rgb8_db"]
            if regret< -1e-5:raise RuntimeError("STOP negative selection regret")
            p_rows.append(dict(image_id=image_id,policy=policy,k=len(subset),score=score,
                selector_id=cid,oracle_id=oracle["candidate_id"],
                selected_psnr_raw_db=selected["psnr_raw_rgb8_db"],
                oracle_psnr_raw_db=oracle["psnr_raw_rgb8_db"],regret_db=max(0.,regret),
                selected_good25=int(selected["psnr_raw_rgb8_db"]>=25),
                oracle_good25=int(oracle["psnr_raw_rgb8_db"]>=25),
                selected_good30=int(selected["psnr_raw_rgb8_db"]>=30),
                oracle_good30=int(oracle["psnr_raw_rgb8_db"]>=30),
                selected_bad20=int(selected["psnr_raw_rgb8_db"]<20),
                charge_reconstruction_wall_s=sum(v["wall_seconds_subprocess"] for v in subset),
                charge_gpu_event_elapsed_s=sum(v["gpu_active_seconds_event"] for v in subset),
                charge_external_cpu_s=sum(v["extra_cpu_score_wall_s"] for v in subset),
                charge_external_fft_calls=len(subset),
                spearman_score_vs_negative_psnr=spearman_small(
                    [v[score] for v in subset],[-v["psnr_raw_rgb8_db"] for v in subset])))
    if len(replay)!=8*14 or len(p_rows)!=8*14*2:
        raise RuntimeError("STOP policy counts")
    summary=[]
    for policy in sorted({r["policy"] for r in p_rows}):
        for score in ("r_signed","r_clipped"):
            sub=[r for r in p_rows if r["policy"]==policy and r["score"]==score]
            summary.append(dict(policy=policy,score=score,n_images=len(sub),
                mean_selected_psnr_db=statistics.mean(r["selected_psnr_raw_db"] for r in sub),
                mean_oracle_psnr_db=statistics.mean(r["oracle_psnr_raw_db"] for r in sub),
                mean_regret_db=statistics.mean(r["regret_db"] for r in sub),
                max_regret_db=max(r["regret_db"] for r in sub),
                good25_selected=sum(r["selected_good25"] for r in sub),
                good25_oracle=sum(r["oracle_good25"] for r in sub),
                good30_selected=sum(r["selected_good30"] for r in sub),
                good30_oracle=sum(r["oracle_good30"] for r in sub),
                bad20_selected=sum(r["selected_bad20"] for r in sub),
                charged_reconstruction_wall_s=sum(r["charge_reconstruction_wall_s"] for r in sub),
                charged_external_score_cpu_s=sum(r["charge_external_cpu_s"] for r in sub),
                charged_external_fft_calls=sum(r["charge_external_fft_calls"] for r in sub)))
    # Only after ALL of the 128 GT and historical replays PASS do we write outputs.
    output.mkdir(exist_ok=False)
    write_csv_x(output/"A13_CANDIDATE_METRICS.csv",metrics)
    write_csv_x(output/"A13_K_POLICY_ROWS.csv",p_rows)
    write_csv_x(output/"A13_SUMMARY.csv",summary)
    write_csv_x(output/"A13_FAILURE_TAXONOMY.csv",[
        dict(image_id=v["image_id"],method=v["method"],rep=v["rep"],candidate_id=v["candidate_id"],
             tag=v["failure_tag"],raw_psnr_db=v["psnr_raw_rgb8_db"],
             rot180_psnr_db=v["psnr_rot180_rgb8_db"],signed_residual=v["r_signed"],
             clipped_residual=v["r_clipped"]) for v in metrics])
    write_csv_x(output/"A13_SCORE_RANKING.csv",[
        dict(image_id=v["image_id"],method=v["method"],rep=v["rep"],
             candidate_id=v["candidate_id"],r_signed=v["r_signed"],r_clipped=v["r_clipped"],
             psnr_raw_rgb8_db=v["psnr_raw_rgb8_db"]) for v in metrics])
    write_json_x(output/"A13_ORACLE_REPLAY_CHECKS.json",dict(
        status="PASS_112_PER_IMAGE_POLICY_REPLAYS",n=len(replay),rows=replay))
    write_json_x(output/"A13_SOURCE_AND_INPUT_AUDIT.json",dict(
        a13_head=source_audit["a13_commit"],stage0_input_sha256=hashfile(
            Path(cfg["output_parent"])/cfg["inventory_stage"]/"A13_SOURCE_AND_INPUT_AUDIT.json"),
        original_candidate_count=128,source=source_audit["source"],
        operator_test=source_audit["operator_test"],
        original_terminal_checks=source_audit["candidates"],
        input_checks=source_audit["input_checks"]))
    (output/"A13_SCORE_LEAKAGE_AUDIT.md").write_text(
        "# Score leakage audit\n\nBoth predeclared policies minimize the terminal external FFT "
        "signed/clipped residual on saved RGB8 candidates. All 112×2 selection indices "
        "were frozen from the non-GT `raw_features` before loading any GT tensor. "
        "PSNR, failure labels and GT oracle were attached only for offline evaluation. "
        "These eight images are already exposed; no tuned or prospective selector is claimed.\n",
        encoding="utf8")
    (output/"A13_UNSUPPORTED_CELLS.md").write_text(
        "# Unsupported cells\n\nSITCOM step160 correction norm is not in the A1.2 "
        "per-candidate completion record; NP post-projection trailing LF stat is not "
        "advertised as a verified cross-root policy here. Both native selectors: N/A. "
        "DAPS historical Fresh2 margin is not an A1.2 K4 native policy. No native "
        "float score on DAPS: residuals use each saved RGB8 PNG for uniform comparability. "
        "No independently calibrated signed-noise fit threshold: non-rotation "
        "low-PSNR cases are UNRESOLVED_LOW_RAW, not automatically poor fits or other basins. "
        "No new held-out/OOD/trajectory evidence.\n",encoding="utf8")
    special=[("68539","DAPS",[0]),("67293","DAPS",[1,2,3]),
             ("67174","NP_LF",[0,1,2]),("60140","SITCOM",[0,1,2,3])]
    notes=["# Frozen illustrative cases (A1.2 ONLY)","",
        "The historical B22 SITCOM 60140 native selector failure is a separate "
        "cohort of trajectories: no B22 candidate identity is imported into this analysis.",""]
    for image_id,method,reps in special:
        for rep in reps:
            v=next(q for q in metrics if (q["image_id"],q["method"],q["rep"])==(image_id,method,rep))
            notes.append(f"- {image_id} {method} rep{rep}: raw={v['psnr_raw_rgb8_db']:.4f}, "
                  f"rot180={v['psnr_rot180_rgb8_db']:.4f}, signed={v['r_signed']:.6g}, "
                  f"clipped={v['r_clipped']:.6g}, tag={v['failure_tag']}")
    (output/"A13_EXAMPLE_CASES.md").write_text("\n".join(notes)+"\n",encoding="utf8")
    (output/"A13_SOURCE_AND_INPUT_AUDIT.md").write_text(
        "# A1.3 source and input audit\n\nAll 128 independently saved terminal "
        "bytes and saved PNGs matched the A1.2 frozen receipts twice (stage0 and at "
        "score time). Input file and tensor-content hashes verified for exactly eight "
        "exposed images. Native DAPS and SITCOM operators numerically reconciled on CPU "
        "before original candidates were scored. See machine audit and replay JSON.\n",encoding="utf8")
    return len(metrics),len(p_rows),len(replay)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",required=True,choices=["analyze"])
    ap.add_argument("--run-root",required=True)
    ap.add_argument("--input-root",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    a=ap.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    t0=time.perf_counter()
    cfg=loadj(CONFIG)
    if Path(a.run_root)!=Path(cfg["run_root"]) or Path(a.input_root)!=Path(cfg["input_root"]) or \
       Path(a.out)!=Path(cfg["output_parent"])/cfg["analysis_stage"] or Path(a.out).exists():
        raise RuntimeError("STOP stage1 root incorrect or occupied")
    checkout_gate(cfg,a.expected_head)
    audited=validate_stage0(cfg,Path(cfg["output_parent"])/cfg["inventory_stage"],a.expected_head)
    native_source_gate(cfg)
    operator_gate(cfg)  # rechecks exact witness/native implementation
    # Re-run 128 checksum/identity verification BEFORE examining any tensor.
    current=inventory(cfg,Path(a.run_root),Path(a.input_root))
    if {r["terminal_sha256"] for r in current["candidates"]} != \
       {r["terminal_sha256"] for r in audited["candidates"]}:
        raise RuntimeError("STOP terminal roster changed after stage0")
    current.update(source=audited["source"],operator_test=audited["operator_test"],
                   a13_commit=a.expected_head)
    count,npol,nreplay=run(cfg,current,Path(a.run_root),Path(a.out))
    write_json_x(Path(a.out)/"A13_CPU_RESOURCE.json",dict(cpu_wall_seconds=time.perf_counter()-t0,
        cpu_count_visible=os.cpu_count(),gpu_inference=0,fft_forward_for_scores=128,
        candidate_count=count,policy_rows=npol,replay_rows=nreplay,source_head=a.expected_head,
        stage="a13_1_cpu_forensics_v1",status="PASS",
        metadata_only=True,gt_free_policies=["r_signed","r_clipped"]))
    print(json.dumps(dict(status="A13_CPU_FORENSICS_COMPLETE",candidates=count,
                          policy_rows=npol,oracle_replays=nreplay,out=a.out),sort_keys=True))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print(f"A13_ANALYSIS_STOP: {type(e).__name__}: {e}",file=sys.stderr)
        sys.exit(8)
