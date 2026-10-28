#!/usr/bin/env python3
"""Fail-closed A7 K4 D2/D3 publication audit, with no sealed-K access.

Recompute independently from the local 400-per-run scoring diagnostics;
publish *only* aggregate performance/quality by metric to the XP return
tree. The local per-run detail contains GT-derived scores and must stay
under PAC outputs, never go into theory_inputs or a repo commit.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a7_d2d3_score_cpu import COMPARE_DAPS
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
A7WT=ROOT/"pr_diffusion_track_r_a7_cpu"
ACTIVE=ROOT/"pr_diffusion_track_r"
PARENT="xp_a7_d2d3_20261028T053026Z"
SOURCE_HEAD="8e1eeb88edff2280b26bd237122228d23b318492"
PINNED_ACTIVE="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
D2_MANIFEST="9b08c8527c3388cfb44b56f934f62b3e48d254ea433867b1c4d87f5ce8335f44"
D3_MANIFEST="19ab2f4ee78c65e5b8c7ab5964358802ce5bd9757ca47927bac6ab4bc0f071cd"
METRICS=("psnr_db","ssim","lpips")
POLICIES=("first","E0_signed_rho","oracle_K4_per_metric","O_orbit_raw_maximum")
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def read(p):
    p=Path(p)
    if (not p.is_file() or p.is_symlink()
        or ROOT not in p.resolve().parents
        or any(w in str(p).lower() for w in BAD)):
        raise RuntimeError("STOP_A7_PUBLISH_SOURCE_FILE")
    return json.loads(p.read_text())

def audit_stage(stage,folder):
    if stage not in ("D2","D3") or folder!=BASE/(PARENT+"_"+stage):
        raise RuntimeError("STOP_A7_PUBLISH_OUTPUT_NAMESPACE")
    aggfile=folder/"XP_A7_K4_AGGREGATE.json"
    detailfile=folder/"XP_A7_K4_LOCAL_DETAIL.json"
    agg=read(aggfile);detail=read(detailfile)
    if (agg.get("status")!="XP_A7_K4_IMAGE_METRIC_AGGREGATE_PASS"
        or agg.get("stage")!=stage or agg.get("n_images")!=100
        or agg.get("n_candidates")!=400 or agg.get("runs")!=[0,1,2,3]
        or agg.get("evaluation_head")!=SOURCE_HEAD
        or agg.get("native_manifest_sha256")!=(
            D2_MANIFEST if stage=="D2" else D3_MANIFEST)
        or agg.get("candidate_ground_truth_metrics_not_in_github_return") is not True
        or agg.get("all_k_gt4_outcomes_untouched") is not True
        or agg.get("E1_CLIP_G1")!="PENDING_CLIP_WORKER"
        or agg.get("metrics")!=list(METRICS)
        or (agg.get("DAPS_published_Table1_reference")!=(COMPARE_DAPS if stage=="D2" else None))):
        raise RuntimeError("STOP_A7_PUBLISH_AGGREGATE_CONTRACT_"+stage)
    raw=detail.get("raw_scores",[])
    if len(raw)!=400:
        raise RuntimeError("STOP_A7_PUBLISH_ROW_COUNT")
    groups=collections.defaultdict(list)
    uids=set()
    for record in raw:
        if (set(record)!={"image_id","rep","uid","rho_signed","psnr_db","ssim","lpips","orbit"}
            or not isinstance(record["image_id"],str)
            or record["rep"] not in (0,1,2,3)
            or record["uid"] in uids
            or not all(isinstance(record[k],(int,float)) and math.isfinite(record[k])
                       for k in ("rho_signed",*METRICS))):
            raise RuntimeError("STOP_A7_PUBLISH_RAW_ROW")
        if (record["rho_signed"]<0
            or not -1.0001<=record["ssim"]<=1.0001
            or record["lpips"]<0
            or record["psnr_db"]<0
            or set(record["orbit"])!=set(METRICS)
            or any(len(record["orbit"][k])!=8 or
                   any(not math.isfinite(float(v)) for v in record["orbit"][k])
                   for k in METRICS)):
            raise RuntimeError("STOP_A7_PUBLISH_BAD_METRIC")
        for metric in METRICS:
            if not math.isclose(record["orbit"][metric][0],record[metric],
                                rel_tol=1e-5,abs_tol=1e-4):
                raise RuntimeError("STOP_A7_PUBLISH_ORBIT_000_DRIFT")
        uids.add(record["uid"])
        groups[record["image_id"]].append(record)
    if len(groups)!=100 or any(len(v)!=4 or
           sorted(r["rep"] for r in v)!=[0,1,2,3] for v in groups.values()):
        raise RuntimeError("STOP_A7_PUBLISH_IMAGE_STRUCTURE")
    rows=[]
    for _,v in sorted(groups.items()):
        v=sorted(v,key=lambda x:x["rep"])
        if [r["rep"] for r in v]!=[0,1,2,3]:
            raise RuntimeError("STOP_A7_PUBLISH_REP_ORDER")
        e0=sorted(v,key=lambda x:(x["rho_signed"],x["rep"]))[0]
        summary={}
        for metric in METRICS:
            sign=-1 if metric=="lpips" else 1
            summary[metric]={
                "first":v[0][metric],
                "E0_signed_rho":e0[metric],
                "oracle_K4_per_metric":max(sign*x[metric] for x in v)*sign,
                "O_orbit_raw_maximum":max(sign*y for x in v
                                            for y in x["orbit"][metric])*sign}
        rows.append(summary)
    recomputed={metric:{policy:sum(row[metric][policy] for row in rows)/100
                        for policy in POLICIES} for metric in METRICS}
    for metric in METRICS:
        for policy in POLICIES:
            actual=float(agg["mean_over_images"][metric][policy])
            if not math.isclose(actual,recomputed[metric][policy],
                                abs_tol=1e-8,rel_tol=1e-9):
                raise RuntimeError("STOP_A7_PUBLISH_INDEPENDENT_MEAN_"+metric+"_"+policy)
        vals=recomputed[metric]
        sign=-1 if metric=="lpips" else 1
        if not (
            sign*(vals["oracle_K4_per_metric"]-vals["first"])>=-1e-7 and
            sign*(vals["oracle_K4_per_metric"]-vals["E0_signed_rho"])>=-1e-7 and
            sign*(vals["O_orbit_raw_maximum"]-vals["oracle_K4_per_metric"])>=-1e-7
        ):
            raise RuntimeError("STOP_A7_PUBLISH_ORACLE_DOMINANCE_"+metric)
    # D2/D3 only. Report reference, not equivalence to DAPS Table1:
    # different image IDs, number of runs, and uncertain SSIM/LPIPS convention.
    public=dict(agg)
    public["publication_status"]="CPU_K4_AGGREGATE_INDEPENDENTLY_RECOMPUTED"
    public["publication_scope"]="XP_RETURNS_ONLY_NOT_THEORY_INPUTS"
    public["source_aggregate_sha256"]=sha(aggfile)
    public["local_per_run_detail_sha256"]=sha(detailfile)
    public["local_per_run_detail_not_committed"]=True
    public["O_orbit_raw_maximum_meaning"]="ORACLE over 8 orbit transforms of all K4 candidates, not deployable selector"
    public["DAPS_Table1_direct_reproduction_claim"]=False
    public["E1_CLIP_G1"]="PENDING_CLIP_WORKER"
    return public

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A7_PUBLISH_GPU_VISIBLE")
    for directory,head in ((A7WT,a.expected_head),(ACTIVE,PINNED_ACTIVE)):
        if (subprocess.check_output(["git","-C",str(directory),"rev-parse","HEAD"],
                 text=True).strip()!=head or
            subprocess.check_output(["git","-C",str(directory),"status","--porcelain"],
                 text=True).strip()):
            raise RuntimeError("STOP_A7_PUBLISH_GIT_WORKTREE")
    results={stage:audit_stage(stage,BASE/(PARENT+"_"+stage))
             for stage in ("D2","D3")}
    targets={stage:A7WT/"docs/track_r/executors/xp/returns"/(
             "XP_A7_IMAGE_K4_"+stage+"_20261028.json") for stage in results}
    if any(p.exists() for p in targets.values()):
        raise RuntimeError("STOP_A7_PUBLISH_OVERWRITE")
    if a.verify_only:
        print("XP_A7_D2D3_INDEPENDENT_AUDIT_PASS",flush=True)
        for stage,r in results.items():
            print(stage,json.dumps(r["mean_over_images"],sort_keys=True),flush=True)
        return
    for stage in results:
        with targets[stage].open("x",encoding="utf-8") as f:
            json.dump(results[stage],f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    print("XP_A7_D2D3_PUBLICATION_STAGED_PASS")
    for stage in results:
        print(stage+"="+str(targets[stage]),"SHA256="+sha(targets[stage]))
    print("NO_THEORY_INPUTS_MODIFIED=true")
    print("NO_K_GT4_SCORED=true")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--verify-only",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A7_D2D3_PUBLISH "+str(e),flush=True)
        sys.exit(8)
