#!/usr/bin/env python3
"""CPU-only SITCOM native replay scoring, after all GPU generation is finished.

GT and historical candidate paths occur only in the offline manifest/catalog.
Never launch this script in a CUDA-visible process or send its inputs to GPU.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,traceback
from pathlib import Path
from scripts.track_r.xp_sitcom_replay_prepare_cpu import ROOT,WT,BASE,BAD,safe,sha
from scripts.track_r.xp_sitcom_replay_gpu import PREP_HEAD,RUN_NAME,MANIFEST_SHA,folder,check_manifest

def load(path):return json.loads(safe(path).read_text())
def dump(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def evaluate(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_SITCOM_SCORE_CUDA_VISIBLE")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
                                 text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
                                  text=True).strip()
    if head!=a.expected_head or dirty:raise RuntimeError("STOP_SITCOM_SCORE_HEAD")
    import torch
    from prdiffusion.track_r.physics import to_rgb8,load_png_rgb8,psnr_rgb8,from_rgb8,rho_signed
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    from scripts.track_r.r0_2_g0_cpu import gt_source
    run=BASE/RUN_NAME
    manifest_file=run/"XP_SITCOM_REPLAY_TASKS.json"
    if sha(manifest_file)!=MANIFEST_SHA:raise RuntimeError("STOP_SITCOM_SCORE_TASK_SHA")
    mf=load(manifest_file)
    rows=check_manifest(mf,a.expected_head)
    prep=load(run/"XP_SITCOM_REPLAY_PREP.json")
    offline_file=run/"XP_SITCOM_REPLAY_OFFLINE_HISTORICAL.json"
    if sha(offline_file)!=prep.get("offline_sha256"):
        raise RuntimeError("STOP_SITCOM_SCORE_OFFLINE_SHA")
    offline=load(offline_file)
    refs=offline.get("rows",[])
    if (len(refs)!=8 or [x["uid"] for x in refs]!=[r["uid"] for r in rows]):
        raise RuntimeError("STOP_SITCOM_SCORE_OFFLINE_IDENTITIES")
    catalog=load(BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json")["rows"]
    historical=[json.loads(x) for x in safe(
        BASE/"r0_7_cpu_20261023T080434Z/analysis/R0_7_SCORED_GT_ROWS.jsonl"
        ).read_text().splitlines() if x.strip()]
    hist={r["uid"]:r for r in historical}
    old={r["uid"]:r for r in catalog}
    output=[]
    for r,ref in zip(rows,refs):
        uid=r["uid"]
        if uid not in old or uid not in hist:
            raise RuntimeError("STOP_SITCOM_SCORE_R07_MISSING")
        meta=old[uid];prev=hist[uid]
        if (ref["measurement_sha256"]!=r["measurement_sha256"] or
            ref["historical_candidate_path"]!=meta["candidate_path"] or
            ref["historical_candidate_sha256"]!=meta["candidate_sha256"] or
            meta["measurement_path"]!=r["measurement_path"] or
            meta["measurement_sha256"]!=r["measurement_sha256"] or
            prev["candidate_path"]!=meta["candidate_path"] or
            prev["gt_path"]!=meta["gt_path"] or
            prev["source"]!=r["source"] or prev["family"]!="SITCOM" or
            int(prev["run_index"])!=r["rep"]):
            raise RuntimeError("STOP_SITCOM_SCORE_PAIRING "+uid)
        if sha(meta["candidate_path"])!=ref["historical_candidate_sha256"]:
            raise RuntimeError("STOP_SITCOM_SCORE_HISTORICAL_HASH "+uid)
        if sha(r["measurement_path"])!=r["measurement_sha256"]:
            raise RuntimeError("STOP_SITCOM_SCORE_Y_DRIFT "+uid)
        dest=folder(run/"gpu"/("b22" if r["source"]=="S1_b22" else "b24_"+str(4+rows[4:].index(r))),r)
        receipt=load(dest/"XP_SITCOM_ROOT_COMPLETE.json")
        if (receipt.get("status")!="XP_SITCOM_REPLAY_ROOT_PASS" or
            receipt.get("uid")!=uid or receipt.get("runner_head")!=a.expected_head or
            receipt.get("measurement_sha256")!=r["measurement_sha256"] or
            receipt.get("gt_access") is not False or
            receipt.get("historical_access") is not False or
            sha(dest/"reconstruction.png")!=receipt["reconstruction_png_sha256"] or
            sha(dest/"reconstruction.pt")!=receipt["reconstruction_pt_sha256"]):
            raise RuntimeError("STOP_SITCOM_SCORE_NEW_RECEIPT "+uid)
        new8=load_png_rgb8(dest/"reconstruction.png")
        gt8=to_rgb8(gt_source(meta))
        raw=float(psnr_rgb8(new8,gt8)[0])
        orbit=orbit_psnr_channel_mse(new8,gt8)
        best=max(orbit.values())
        tau=float(prev["tau_signed_good_q95"])
        if raw>=25:label="GOOD"
        elif best>=25:label="ORBIT_ERROR"
        else:
            from scripts.track_a.a15_fastdips_preflight import load_tensor
            y=load_tensor(safe(r["measurement_path"]),
                          ("measurement","y","observation"),(1,3,384,384))
            residual=float(rho_signed(from_rgb8(new8),y.float())[0])
            label="NONORBIT_CONSISTENT" if residual<=tau else "NONORBIT_INCONSISTENT"
        delta=abs(raw-float(prev["psnr_raw"]))
        # B24 historical candidate is RGB8 PNG; B22 uses a float32 tensor.
        if ref["historical_format"]=="png":
            exact=sha(dest/"reconstruction.png")==ref["historical_candidate_sha256"]
            match_type="png_file_sha256"
        elif ref["historical_format"]=="pt":
            expected=ref["historical_native"].get("native_tensor_hash")
            if not isinstance(expected,str) or len(expected)!=64:
                raise RuntimeError("STOP_SITCOM_SCORE_HISTORICAL_TENSOR_HASH")
            exact=receipt["reconstruction_content_sha256"]==expected
            match_type="float32_tensor_content_sha256"
        else:raise RuntimeError("STOP_SITCOM_SCORE_HISTORICAL_FORMAT")
        passed=bool(delta<=0.1 and label==prev["raw_label"])
        output.append({
            "uid":uid,"source":r["source"],"rep":r["rep"],
            "raw_psnr":raw,"historical_psnr":float(prev["psnr_raw"]),
            "abs_delta_db":delta,
            "replay_label":label,"historical_label":prev["raw_label"],
            "orbit_best_db":best,
            "exact_historical":exact,"hash_comparison":match_type,
            "sampling_wall_s":receipt["sampling_wall_s"],
            "sampling_cuda_event_s":receipt["sampling_cuda_event_s"],
            "pass":passed})
        print("SITCOM_COMPARE",uid,"delta_db",round(delta,7),
              "label",label,"exact",exact,"pass",passed,flush=True)
    verdict="XP_SITCOM_REPLAY_PASS" if len(output)==8 and all(x["pass"] for x in output) else "XP_SITCOM_REPLAY_MISMATCH"
    report={
        "status":verdict,"schema":"prdiff.track_r.xp.sitcom-replay-cpu-score.v1",
        "head":head,"prep_head":PREP_HEAD,"manifest_sha256":MANIFEST_SHA,
        "n_roots":len(output),"n_label_match":sum(x["replay_label"]==x["historical_label"] for x in output),
        "n_exact_historical":sum(x["exact_historical"] for x in output),
        "max_abs_delta_db":max(x["abs_delta_db"] for x in output),
        "no_gpu_scoring":True,
        "generation_gt_free":True,
        "original_outputs_immutable":True,
        "results":output}
    dest=run/"XP_SITCOM_REPLAY_OFFLINE_SCORE.json"
    dump(dest,report)
    print(verdict,"n="+str(len(output)),
          "max_delta_db="+str(report["max_abs_delta_db"]),
          "exact="+str(report["n_exact_historical"])+"/8",flush=True)
    print("OFFLINE_SCORE_SHA="+sha(dest))
    print("OK" if verdict.endswith("_PASS") else "ACTION NEEDED: SITCOM_NATIVE_REPLAY_MISMATCH")
    return 0 if verdict.endswith("_PASS") else 8

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    try:raise SystemExit(evaluate(parser.parse_args()))
    except Exception as ex:
        traceback.print_exc()
        print("ACTION NEEDED: XP_SITCOM_SCORE "+str(ex),flush=True)
        raise SystemExit(8)
