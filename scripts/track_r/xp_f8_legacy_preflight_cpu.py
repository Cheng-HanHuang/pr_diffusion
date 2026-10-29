#!/usr/bin/env python3
"""Read-only A10 XP-F8 historical DAPS 400x5 replay preflight.

Constructs exactly eight historical B22/DEV80 replay inputs from the
R0.7 catalog, with frozen native B21 and B24 seed schedules. Never loads GT
or bundled datasets. A GT-free manifest and separate historical-reference
receipt are prepared; NO GPU jobs are launched until a distinct native
adapter is certified and replay validates every trajectory.

This script must fail closed on ambiguous source, seed, measurement,
candidate, checkpoint, schedule, deadline, or budget provenance.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
from prdiffusion.b24_protocol import seed_row

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
R07=BASE/"r0_7_cpu_20261023T080434Z/analysis"
CAT=R07/"R0_7_OFFLINE_GT_CATALOG.json"
SCORES=R07/"R0_7_SCORED_GT_ROWS.jsonl"
R07_RECEIPT=R07/"R0_7_COMPLETE_RECEIPT.json"
B21=ROOT/"outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401"
B21_MANIFEST=B21/"manifest.tsv"
ORIG_DAPS=ROOT/"pr_diffusion_b19_solver/external/daps"
LEGACY_CONFIG=WT/"configs/b23/fresh1_frozen.yaml"
MODEL=ROOT/"models/ffhq_10m.pt"
B16=BASE/"xp_a11_b16_prod_20261028T113034Z"
DAPS_HEAD="e7a77d094167084faed19b599b96673b7bb11447"
PATCHED_SAMPLER_BLOB="ae1fc6908bb1e7f0c0349d4f82cdf20364389db8"
MODEL_SHA="81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
COHORTS=("S1_b22","S2_b24_dev80")
A10_CAP_H=70.
A10_DEADLINE=datetime.datetime(2026,10,30,12,0,tzinfo=ZoneInfo("America/Detroit"))
def fail(x,msg):
    if not x:raise RuntimeError("STOP_F8_LEGACY_"+msg)
def checkpath(path):
    path=Path(path)
    resolved=path.resolve(strict=False)
    fail(path.is_absolute() and path.is_file() and not path.is_symlink() and
         ROOT.resolve() in resolved.parents and
         not any(v in str(path).lower() or v in str(resolved).lower()
                 for v in BAD),"UNSAFE_PATH_"+str(path))
    return path
def sha(path):
    path=checkpath(path);h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()
def git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],
        timeout=60,text=True).strip()
def read_json(path):
    return json.loads(checkpath(path).read_text())
def write_x(path,data):
    with path.open("x",encoding="utf-8") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
def historical_rows():
    receipt=read_json(R07_RECEIPT)
    fail(receipt["n_scored"]==3240 and
         receipt["evaluated_rows_sha256"]==sha(SCORES),"R07_SCORE_HASH")
    raw=read_json(CAT)["rows"]
    fail(len(raw)==3240,"R07_CATALOG_3240")
    scores={}
    with SCORES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line)
                fail(x["uid"] not in scores,"DUPLICATE_SCORE")
                scores[x["uid"]]=x
    fail(len(scores)==3240,"R07_SCORE_3240")
    groups=defaultdict(list)
    for row in raw:
        if row.get("family")=="DAPS" and row.get("source") in COHORTS:
            groups[(row["source"],str(row["image_id"]))].append(row)
    fail(len([k for k in groups if k[0]=="S1_b22"])==100,
         "B22_DAPS_IMAGE_COUNT")
    fail(len([k for k in groups if k[0]=="S2_b24_dev80"])==80,
         "DEV80_DAPS_IMAGE_COUNT")
    chosen=[]
    for cohort in COHORTS:
        ids=sorted(iid for c,iid in groups if c==cohort)
        for iid in ids[:2]:
            rr=sorted(groups[(cohort,iid)],key=lambda q:int(q["run_index"]))
            expect=range(2 if cohort=="S1_b22" else 4)
            fail([int(r["run_index"]) for r in rr]==list(expect),
                 "HIST_REP_GRID_"+cohort+"_"+iid)
            for r in rr[:2]:
                sc=scores[r["uid"]]
                fail(sc["source"]==cohort and sc["family"]=="DAPS"
                     and str(sc["image_id"])==iid and
                     int(sc["run_index"])==int(r["run_index"]),
                     "R07_SCORE_JOIN")
                fail(sha(r["measurement_path"])==r["measurement_sha256"],
                     "MEASUREMENT_SHA_"+iid)
                fail(sha(r["candidate_path"])==r["candidate_sha256"],
                     "CANDIDATE_SHA_"+iid)
                # No GT bytes accessed; historical raw GT labels may only
                # be opened by the offline comparator after GPU generation.
                chosen.append((r,sc))
    fail(len(chosen)==8 and
         len(set(x[0]["uid"] for x in chosen))==8,"EIGHT_UNIQUE")
    return chosen
def b21_table():
    with checkpath(B21_MANIFEST).open(newline="") as f:
        rows=list(csv.DictReader(f,delimiter="\t"))
    fail(len(rows)==100,"B21_MANIFEST_100")
    d={}
    for r in rows:
        iid=str(r["image_id"]).zfill(5)
        fail(iid not in d,"B21_DUPLICATE_IMAGE")
        idx=int(r["row_id"])
        fail(int(r["seed1"])==22000+idx and
             int(r["seed2"])==23000+idx,
             "B21_FRESH2_SEEDS")
        d[iid]=r
    return d
def legacy_source():
    config=read_json(LEGACY_CONFIG)
    fail(config["scheduler"]["annealing_steps"]==400 and
         config["scheduler"]["diffusion_steps_per_anneal"]==5 and
         config["scheduler"]["lf_enabled"] is False and
         config["scheduler"]["hio_enabled"] is False,
         "HIST_400x5_CONFIG")
    fail(config["model"]["checkpoint_sha256"]==MODEL_SHA and
         sha(MODEL)==MODEL_SHA,"MODEL_CHECKPOINT_SHA")
    fail(git(ORIG_DAPS,"rev-parse","HEAD")==DAPS_HEAD,
         "HIST_DAPS_HEAD")
    fail(git(ORIG_DAPS,"hash-object","sampler.py")==PATCHED_SAMPLER_BLOB,
         "HIST_EDITED_SAMPLER_BLOB")
    script=checkpath(ORIG_DAPS/"posterior_sample.py")
    native=script.read_text()
    fail("B21_MEASUREMENT_PATH" in native and
         "B21_START_NOISE_SEED" in native,
         "HIST_MEASUREMENT_SEED_OVERRIDE")
    # The DAPS-4K upstream sparse worker is NOT a valid historical adapter.
    return dict(native_head=DAPS_HEAD,
         posterior_sample_sha256=sha(script),
         patched_sampler_sha256=sha(ORIG_DAPS/"sampler.py"),
         checkpoint_sha256=MODEL_SHA,
         legacy_config_sha256=sha(LEGACY_CONFIG),
         source_snapshot=config["source"]["source_snapshot"],
         source_patch=config["source"]["source_snapshot_patch"],
         default_off_controls=["B20_LF_ENABLE=0","B20_LF_ALPHA=0.0",
                                "B21_CONT_ENABLE unset"],
         ann_steps=400,diffusion_steps=5,
         solver_source="ORIGINAL_MODIFIED_B21_SOURCE_NOT_DAPS4K")
def admission():
    s=read_json(B16/"B16_PRODUCTION_STATE.json")
    rec=read_json(B16/"XP_A11_B16_PROD_COMPLETE.json")
    fail(s["status"]=="A10_B16_GENERATION_COMPLETE_WAIT_F8"
         and s["active"]==[] and rec["n_new_roots"]==2160
         and rec["no_new_outcomes_opened"] is True,
         "B16_NOT_COMPLETE")
    used=(s["gate_spent_gpu_seconds"]+s["spent_gpu_seconds"])/3600
    fail(s["A10_limit_gpu_hours"]==A10_CAP_H
         and used<A10_CAP_H,
         "CAMPAIGN_GPUH")
    fail(datetime.datetime.now(ZoneInfo("America/Detroit"))<A10_DEADLINE,
         "A10_DEADLINE_ELAPSED")
    return dict(used_gpu_h=used,remaining_gpu_h=A10_CAP_H-used,
                deadline_et=A10_DEADLINE.isoformat(),
                b16_complete_receipt_sha256=sha(
                    B16/"XP_A11_B16_PROD_COMPLETE.json"))
def main(args):
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    fail(git(WT,"rev-parse","HEAD")==args.expected_head
         and not git(WT,"status","--porcelain"),
         "AUTHOR_WORKTREE")
    native=legacy_source()
    gate=admission()
    rows=historical_rows()
    b21=b21_table()
    public=[];offline=[]
    for r,s in rows:
        cohort=r["source"];iid=str(r["image_id"]);rep=int(r["run_index"])
        if cohort=="S1_b22":
            fail(iid in b21,"B21_IMAGE_MISSING")
            m=b21[iid]
            seed=int(m["seed1"] if rep==0 else m["seed2"])
            rule="B21_11_seed1_22000_plus_rowid_or_seed2_23000_plus_rowid"
        else:
            values=seed_row(iid)["solver_seeds"]["DAPS"]
            fail(len({int(t)%(2**32) for t in values})==4,
                 "DEV80_NATIVE_SEED_COLLISION")
            seed=int(values[rep])%(2**32)
            rule="B24_SOLVER_SEED_V1_native_mod2pow32"
        public.append(dict(uid=r["uid"],source=cohort,family="DAPS",
            image_id=iid,rep=rep,native_seed=seed,seed_rule=rule,
            measurement_path=r["measurement_path"],
            measurement_sha256=r["measurement_sha256"]))
        offline.append(dict(uid=r["uid"],source=cohort,image_id=iid,
            rep=rep,historical_candidate_path=r["candidate_path"],
            historical_candidate_sha256=r["candidate_sha256"],
            historical_candidate_format=r["candidate_format"],
            gt_path=r["gt_path"],gt_format=r["gt_format"],
            raw_psnr=s["psnr_raw"],raw_label=s["raw_label"],
            tau=s["tau_signed_good_q95"]))
    fail(len({(r["source"],r["image_id"],r["rep"]) for r in public})==8,
         "UNIQUE_REPLAY_JOBS")
    fail(len({(r["source"],r["image_id"],r["native_seed"])
              for r in public})==8,"SEED_COLLISION")
    out=Path(args.out)
    fail(out.parent==BASE and out.name.startswith("xp_f8_legacy_preflight_")
         and not out.exists() and not out.is_symlink(),
         "FRESH_OUTPUT_NAMESPACE")
    out.mkdir(parents=False,exist_ok=False)
    write_x(out/"XP_F8_REPLAY_INPUTS_GT_FREE.json",dict(
        status="XP_F8_LEGACY_INPUTS_FROZEN_CPU_PASS",
        n_replay=8,n_b22=4,n_dev80=4,
        model_source=native,accounting=gate,
        no_gt_paths_supplied=True,no_historical_candidates_supplied=True,
        sampler_config=dict(annealing_steps=400,diffusion_steps=5),
        rows=public))
    write_x(out/"XP_F8_REPLAY_OFFLINE_ONLY.json",dict(
        status="XP_F8_OFFLINE_REPLAY_REFERENCE_FROZEN",
        no_gpu_data_passed=True,rows=offline))
    write_x(out/"XP_F8_PRECHECK_RECEIPT.json",dict(
        status="XP_F8_REPLAY_INPUT_LINEAGE_PREFLIGHT_PASS",
        source_head=args.expected_head,
        native_provenance=native,accounting=gate,
        manifest_sha256=sha(out/"XP_F8_REPLAY_INPUTS_GT_FREE.json"),
        offline_sha256=sha(out/"XP_F8_REPLAY_OFFLINE_ONLY.json"),
        native_adapter_certified=False,any_gpu_job_launched=False,
        next_gate="PROVE_NATIVE_ADAPTER_REPRODUCES_B21_B24_SEEDS_AND_400x5"))
    print("XP_F8_LEGACY_CPU_INPUT_PREFLIGHT_PASS")
    for r in public:
        print("REPLAY_ROW",r["source"],r["image_id"],
              "rep",r["rep"],"seed",r["native_seed"])
    print("REMAINING_SHARED_GPU_H",round(gate["remaining_gpu_h"],3))
    print("PREFLIGHT_OUTPUT",out)
    print("INPUT_MANIFEST_SHA",sha(out/"XP_F8_REPLAY_INPUTS_GT_FREE.json"))
    print("ADAPTER_CERTIFIED=false; GPU_JOBS_LAUNCHED=0")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_F8_LEGACY_PREFLIGHT "+str(e),flush=True)
        sys.exit(8)
