#!/usr/bin/env python3
"""A8 first prospective FAST FFHQ A16/A64 vs registered TH1 v1, CPU-only.

Verifies the frozen v1+companion registry and forecast SHA *before*
accessing A16/A64 reconstructions. Scores only the 180 existing R0
exposed FFHQ ids and 0..63 native FAST runs; primary 64, secondary 16.
No theory refit, no D1-P2, no new simulation or GPU. Every future-native
candidate receipt is SHA-verified. Historical K4 R0.7 scores are SHA
verified, not re-approximated from RGB8.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
from scripts.track_r.xp_a7_postfreeze_kernel_cpu import (
    exact_panel,compare_interval,read_registered_forecasts,verify_freeze)
from scripts.track_r.xp_a16_contract import (
    RUN as A16_RUN,TASKS as A16_TASKS,MANIFEST_SHA as A16_SHA,
    PREP_HEAD as A16_PREP,build_chunks)
from scripts.track_r import xp_night_supervisor as night
from scripts.track_r.xp_night_publish_cpu import (
    RUN as A64_RUN,GEN_HEAD as A64_GEN,EXPECTED_MANIFEST_HASHES)
from scripts.track_r.xp_fast_gpu import frozen_roots,root_folder,validate_complete
from scripts.track_r.xp_a16_publish_cpu import (
    GEN_HEAD as A16_GEN,verify_chunk_proofs)
from scripts.track_r.r0_2_g0_cpu import gt_source
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
from prdiffusion.track_r.physics import to_rgb8,psnr_rgb8,rho_signed

ROOT=Path("/egr/research-pac/huang248")
AUTHOR_WT=Path(__file__).resolve().parents[2]
ORIGINAL=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
PINNED_A6_HEAD="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
R07=BASE/"r0_7_cpu_20261023T080434Z"
R07_SCORES=R07/"analysis/R0_7_SCORED_GT_ROWS.jsonl"
R07_RECEIPT=R07/"analysis/R0_7_COMPLETE_RECEIPT.json"
R07_CATALOG=R07/"prepare/R0_7_OFFLINE_GT_CATALOG.json"
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")
V1_COMMIT="8ef5f94d6de8aab4c3cc25b17fa797db0de1560b"
V1_PATH="docs/track_r/theory/TH1_PREDICTIONS_v1.json"
V1_SHA="c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753"
COMP_COMMIT="5782149155056ea06869de4262dc2b9f259c3d10"
COMP_PATH="docs/track_r/theory/TH1_PREDICTIONS_v1_PER_IMAGE.csv"
COMP_SHA="74190e6722a11d1e7e16118984f89bbc5281f665629902db244b5740cd783462"
REGISTRY=AUTHOR_WT/"docs/track_r/theory/PREDICTIONS_REGISTRY.md"
K4=4
GOOD=25.
POLICIES=("E0","O_raw","O_orbit")

def safe(path):
    p=Path(path)
    r=p.resolve(strict=False)
    if (not p.is_absolute() or not p.is_file() or p.is_symlink()
        or ROOT.resolve() not in r.parents
        or any(x in str(p).lower() or x in str(r).lower() for x in BANNED)):
        raise RuntimeError("STOP_A8_FAST_UNSAFE_SOURCE "+str(p))
    return p

def sha(path):
    h=hashlib.sha256()
    with safe(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def head_guard(head):
    for repo,expected in ((AUTHOR_WT,head),(ORIGINAL,PINNED_A6_HEAD)):
        actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        dirty=subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip()
        if actual!=expected or dirty:raise RuntimeError("STOP_A8_FAST_CHECKOUT_DRIFT")

def register_first(head):
    # Source and registry access is authorized; no outcomes are accessed.
    receipt,v1=read_registered_forecasts(
        AUTHOR_WT,REGISTRY,"v1",V1_COMMIT,V1_PATH,V1_SHA,head)
    creceipt,companion=read_registered_forecasts(
        AUTHOR_WT,REGISTRY,"v1 companion",COMP_COMMIT,COMP_PATH,COMP_SHA,head)
    if len(v1)!=976 or len(companion)!=14800:
        raise RuntimeError("STOP_A8_FAST_REGISTERED_PANEL_CONTRACT")
    return receipt,creceipt,v1

def historical():
    recorded=json.loads(safe(R07_RECEIPT).read_text())
    if (recorded.get("status") not in ("R0_7_ANALYSIS_COMPLETE",
          "R0_7_ANALYSIS_COMPLETE_WITH_WARNINGS") or
        recorded.get("n_scored")!=3240 or
        sha(R07_SCORES)!=recorded.get("evaluated_rows_sha256")):
        raise RuntimeError("STOP_A8_FAST_R07_HISTORY")
    catalog=json.loads(safe(R07_CATALOG).read_text())
    if len(catalog.get("rows",[]))!=3240:
        raise RuntimeError("STOP_A8_FAST_R07_CATALOG")
    identities={}
    for row in catalog["rows"]:
        if row["source"]!="S5_r0_fast":continue
        key=(row["measurement_source"],str(row["image_id"]))
        value=(row["gt_path"],row["gt_format"],
               row["measurement_path"],row["measurement_sha256"])
        if key in identities and identities[key]!=value:
            raise RuntimeError("STOP_A8_FAST_HIST_IDENTITY")
        identities[key]=value
    if len(identities)!=180:
        raise RuntimeError("STOP_A8_FAST_HIST_COUNT")
    scored=collections.defaultdict(dict)
    with safe(R07_SCORES).open() as f:
        for line in f:
            r=json.loads(line)
            if r["source"]!="S5_r0_fast":continue
            key=(r["measurement_source"],str(r["image_id"]))
            run=int(r["run_index"])
            if run in scored[key] or run not in range(4) or r["family"]!="FAST":
                raise RuntimeError("STOP_A8_FAST_HIST_DUPLICATE")
            raw=float(r["psnr_raw"])
            orbit=float(r["psnr_orbit_best"])
            rho=float(r["rho_signed_raw"])
            if not (math.isfinite(raw) and math.isfinite(orbit) and
                    math.isfinite(rho) and orbit+1e-7>=raw):
                raise RuntimeError("STOP_A8_FAST_HIST_METRIC")
            scored[key][run]=dict(run=run,rho=rho,
                     good_raw=raw>=GOOD,good_orbit=orbit>=GOOD)
    if (len(scored)!=180 or any(set(v)!=set(range(4)) for v in scored.values())):
        raise RuntimeError("STOP_A8_FAST_HIST_RUNS")
    return identities,scored

def future_inputs():
    if sha(A16_TASKS)!=A16_SHA:
        raise RuntimeError("STOP_A8_FAST_A16_MANIFEST_SHA")
    a16=json.loads(safe(A16_TASKS).read_text())
    a16rows=frozen_roots(a16,A16_PREP)
    complete=json.loads(safe(A16_RUN/"XP_A16_GENERATION_COMPLETE.json").read_text())
    verify_chunk_proofs(complete,a16rows)  # 2,160 native and chunk receipts
    a16files={}
    for start,count in (build_chunks()[0]+build_chunks()[1]):
        folder=A16_RUN/"gpu"/f"chunk_{start:04d}_{count:02d}"
        for r in a16rows[start:start+count]:
            key=(r["measurement_source"],str(r["image_id"]),r["rep"])
            a16files[key]=(r,root_folder(folder,r)/"reconstruction.pt")
    if len(a16files)!=2160:raise RuntimeError("STOP_A8_FAST_A16_ROOT_COUNT")
    if sha(A64_RUN/"A64.json")!=EXPECTED_MANIFEST_HASHES["A64"]:
        raise RuntimeError("STOP_A8_FAST_A64_MANIFEST_SHA")
    a64=json.loads(safe(A64_RUN/"A64.json").read_text())
    a64rows=frozen_roots(a64,A64_GEN)
    a64files={}
    for job in night.jobs():
        if job["stage"]!="A64":continue
        # Historical collector is proven; validate each complete chunk.
        night.validate_root(A64_RUN,job,A64_GEN)
        folder=night.paths(A64_RUN,job)
        for r in a64rows[job["start"]:job["start"]+job["count"]]:
            key=(r["measurement_source"],str(r["image_id"]),r["rep"])
            if key in a64files:raise RuntimeError("STOP_A8_FAST_A64_COLLISION")
            a64files[key]=(r,root_folder(folder,r)/"reconstruction.pt")
    if len(a64files)!=8640:raise RuntimeError("STOP_A8_FAST_A64_ROOT_COUNT")
    return a16files,a64files

def score_future(identities,scored,a16files,a64files):
    import torch
    from scripts.track_r.r0_2_g0_cpu import gt_source
    torch.set_num_threads(min(4,torch.get_num_threads()))
    total=0
    for cohort,iid in sorted(identities):
        gtpath,gtfmt,ypath,yhash=identities[(cohort,iid)]
        if sha(ypath)!=yhash:raise RuntimeError("STOP_A8_FAST_MEAS_HASH")
        g8=to_rgb8(gt_source({"gt_path":gtpath,"gt_format":gtfmt}).float())
        record=torch.load(safe(ypath),map_location="cpu",weights_only=True)
        y=record["measurement"].float()
        if y.shape!=(1,3,384,384):
            raise RuntimeError("STOP_A8_FAST_Y_SHAPE")
        group=scored[(cohort,iid)]
        for rep in range(4,64):
            candidate=(a16files if rep<16 else a64files)
            if (cohort,iid,rep) not in candidate:
                raise RuntimeError("STOP_A8_FAST_MISSING_NATIVE_REP")
            row,path=candidate[cohort,iid,rep]
            if (row["measurement_path"]!=ypath
                or row["measurement_sha256"]!=yhash):
                raise RuntimeError("STOP_A8_FAST_MEAS_LINEAGE")
            x=torch.load(safe(path),map_location="cpu",weights_only=True)["reconstruction"].float()
            if x.shape!=(1,3,256,256) or not torch.isfinite(x).all():
                raise RuntimeError("STOP_A8_FAST_X_SHAPE")
            x8=to_rgb8(x)
            scores=orbit_psnr_channel_mse(x8,g8)
            raw=float(scores["000"])
            best=max(scores.values())
            rh=float(rho_signed(x,y)[0])
            if not all(math.isfinite(z) for z in (raw,best,rh)) or rh<0:
                raise RuntimeError("STOP_A8_FAST_NONFINITE")
            group[rep]=dict(run=rep,rho=rh,
                 good_raw=raw>=GOOD,good_orbit=best>=GOOD)
            total+=1
        if total%600==0:print("A8_FAST_FFHQ_SCORING",total,"/10800",flush=True)
    if (total!=10800 or any(set(v)!=set(range(64)) for v in scored.values())):
        raise RuntimeError("STOP_A8_FAST_PANEL_GAPS")
    return scored

def compare(scored,forecasts):
    keys=sorted({(row["panel"],row["panel_n_max"],row["K"])
                 for row in forecasts.values() if row["family"]=="FAST"
                    and row["policy"] in POLICIES})
    observed={}
    groups=collections.defaultdict(list)
    for (cohort,iid),values in scored.items():
        if len(values)!=64:raise RuntimeError("STOP_A8_FAST_NOT_64")
        short="B22" if cohort=="S1_b22" else (
              "DEV80" if cohort=="S2_b24_dev80" else None)
        if short is None:raise RuntimeError("STOP_A8_FAST_BAD_COHORT")
        groups[short].append(values)
    if len(groups["B22"])!=100 or len(groups["DEV80"])!=80:
        raise RuntimeError("STOP_A8_FAST_COHORT_SIZE")
    groups["POOLED"]=groups["B22"]+groups["DEV80"]
    for cohort,group in groups.items():
        for panel,n,k in keys:
            if k>n:continue
            policy_rows=[exact_panel([values[j] for j in range(n)],k)
                         for values in group]
            for policy in POLICIES:
                observed[(cohort,panel,n,k,policy)]=(
                    sum(r[policy] for r in policy_rows)/len(policy_rows))
    result=[]
    for (family,cohort,panel,n,variant,k,policy),f in sorted(forecasts.items()):
        if family!="FAST" or policy not in POLICIES:continue
        key=(cohort,panel,n,k,policy)
        if key not in observed:raise RuntimeError("STOP_A8_FAST_UNOBSERVED_FORECAST")
        result.append(dict(family=family,cohort=cohort,panel=panel,
            panel_n_max=n,model_variant=variant,K=k,policy=policy,
            **compare_interval(observed[key],f)))
    if not result:raise RuntimeError("STOP_A8_FAST_NO_COMPARISON")
    report=[]
    for (variant,panel,policy,cohort),subset in sorted(
          {k:[v for v in result if (v["model_variant"],v["panel"],
             v["policy"],v["cohort"])==k] for k in set(
          (r["model_variant"],r["panel"],r["policy"],r["cohort"]) for r in result)}.items()):
        report.append(dict(variant=variant,panel=panel,policy=policy,
             cohort=cohort,n_grid=len(subset),
             n_covered=sum(r["inside"] for r in subset),
             mean_standardized_residual=sum(r["standardized_residual"]
                                      for r in subset)/len(subset)))
    return result,report

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A8_FAST_GPU_VISIBLE")
    head_guard(a.expected_head)
    out=Path(a.out)
    if (out.parent!=BASE or out.exists() or out.is_symlink()
        or not out.name.startswith("xp_a8_fast_v1_")):
        raise RuntimeError("STOP_A8_FAST_OUTPUT_PATH")
    # STRICT ORDER: registered v1 proof is checked before opening any
    # FFHQ candidate with run index >=4.
    receipt,creceipt,forecasts=register_first(a.expected_head)
    print("A8_FAST_V1_REGISTRY_VERIFIED",len(forecasts),flush=True)
    identities,baseline=historical()
    future1,future2=future_inputs()
    if a.preflight_only:
        print("A8_FAST_FFHQ_NATIVE_PREFLIGHT_PASS 720+2160+8640",flush=True)
        return
    values=score_future(identities,baseline,future1,future2)
    rows,groups=compare(values,forecasts)
    publication={
        "status":"A8_FAST_V1_FIRST_PROSPECTIVE_COMPARISON_PASS",
        "scope":"FFHQ_EXPOSED_180_ONLY","GT_metric":"Good25",
        "n_images":180,"per_image_run_indices":list(range(64)),
        "n_historical":720,"n_A16":2160,"n_A64":8640,
        "panels":{"primary":64,"secondary":16},
        "policies":list(POLICIES),
        "E1_CLIP":"PENDING_ORIENTATION_RESULTS",
        "v1_forecast_SHA":V1_SHA,"v1_companion_SHA":COMP_SHA,
        "forecasts_frozen_before_outcomes_verified":True,
        "forecast_refit_performed":False,
        "analysis_head":a.expected_head,
        "forecast_comparison":rows,
        "per_variant_coverage":groups,
        "D1_P2_outcomes_opened":False,
        "B16_F8_outcomes_opened":False,
        "no_theory_inputs_modified":True}
    out.mkdir(parents=True,exist_ok=False)
    with (out/"A8_FAST_V1_COMPARISON.json").open("x") as f:
        json.dump(publication,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("A8_FAST_V1_COMPARISON_PASS",len(rows),"forecast comparisons",
          "coverage rows",len(groups),flush=True)
    print("A8_FAST_V1_OUTPUT="+str(out),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A8_FAST_V1 "+str(e),flush=True)
        sys.exit(8)
