#!/usr/bin/env python3
"""A7 CPU-only ImageNet DAPS-D2 / SITCOM-D3 published-protocol scorer.

Two already completed *K4* panels only, no D1 phase2, no FFHQ,
no sealed outcomes. Implements first-run, GT-free signed-rho E0,
per-metric K4 oracle and 8-element orbit repair. CLIP G1 policy
remains PENDING until XP-C is frozen. No online weight downloads:
AlexNet LPIPS from locally installed pretrained cache or FAIL CLOSED.

Writes per-run diagnostics in a local PAC output directory and a
separate Git-cleared aggregate file, which excludes image paths and
per-run GT scores. No files are automatically committed.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_t_sitcom_prepare_cpu import IMAGENET,COMMON_Y_SHA,ROOT,WT,BASE
from scripts.track_r.xp_night_publish_cpu import RUN as D2_RUN
from scripts.track_r.xp_a6_d1p2_contract import SOURCE_HEAD as SOURCE_FAST
from scripts.track_r.xp_t_fast_prepare_cpu import sha
from prdiffusion.track_r.physics import to_rgb8,rho_signed,psnr_rgb8,apply_orbit

AUTHOR_WT=Path(__file__).resolve().parents[2]
PINNED_A6_HEAD="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
D3_RUN=BASE/"xp_a6_20261027T165112Z"
D2_MANIFEST_SHA="9b08c8527c3388cfb44b56f934f62b3e48d254ea433867b1c4d87f5ce8335f44"
D3_MANIFEST_SHA="19ab2f4ee78c65e5b8c7ab5964358802ce5bd9757ca47927bac6ab4bc0f071cd"
HEAD_D2="8d9bfed0fa62735791987a24791cfb97123540d8"
HEAD_D3="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
GT=IMAGENET/"XP_IMAGENET_GT_ONLY_CPU.json"
COMMON=IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
COMPARE_DAPS={"psnr_db":25.78,"ssim":0.743,"lpips":0.254}

def safe(path):
    p=Path(path)
    if (p.is_symlink() or not p.is_file()
        or ROOT.resolve() not in p.resolve(strict=False).parents
        or any(x in str(p.resolve(strict=False)).lower() for x in BAD)):
        raise RuntimeError("STOP_A7_SCORER_UNSAFE_FILE")
    return p

def jread(path):return json.loads(safe(path).read_text())

def assert_completed(stage):
    if stage=="D2":
        run,expect,producer,meta=D2_RUN,D2_MANIFEST_SHA,HEAD_D2,"XP_NIGHT_CONTROLLER_STATE.json"
        data=jread(run/meta)
        if (data.get("status")!="MAIN_PASS" or data.get("head")!=producer
            or data.get("completed_roots")!=9440
            or data.get("outcomes_scored") is not False):
            raise RuntimeError("STOP_A7_D2_GENERATION_NOT_CERTIFIED")
        mf=jread(run/"D2.json")
        if (sha(run/"D2.json")!=expect or mf.get("stage")!="D2"
            or mf.get("n_roots")!=400 or mf.get("head")!=producer):
            raise RuntimeError("STOP_A7_D2_MANIFEST")
    else:
        run,expect,producer,meta=D3_RUN,D3_MANIFEST_SHA,HEAD_D3,"XP_A6_CONTROLLER_STATE.json"
        data=jread(run/meta)
        if (data.get("head")!=producer or
            data.get("K_gt4_results_sealed") is not True or
            data.get("outcomes_scored") is not False or
            data.get("status") not in ("RUNNING","STOP_NO_CLEAN_GPU_FOR_30_MIN",
               "MAIN_PASS","PAUSED_AT_TIME_OR_BUDGET")):
            raise RuntimeError("STOP_A7_D3_GENERATION_NOT_CERTIFIED")
        done=[x for x in data.get("done",[]) if x.startswith("D3_")]
        if len(done)!=400 or len(set(done))!=400:
            raise RuntimeError("STOP_A7_D3_NOT_COMPLETE")
        mf=jread(run/"D3.json")
        if (sha(run/"D3.json")!=expect or mf.get("stage")!="D3"
            or mf.get("n_roots")!=400 or mf.get("head")!=producer):
            raise RuntimeError("STOP_A7_D3_MANIFEST")
    rows=mf["rows"]
    if (len(rows)!=400 or len({r["uid"] for r in rows})!=400
        or any(r["rep"]!=i%4 for i,r in enumerate(rows))
        or mf.get("gt_paths_supplied") is not False):
        raise RuntimeError("STOP_A7_IMAGE_RUN_STRUCTURE")
    return run,rows,expect,producer

def verify_one(run,stage,row,manifest_hash,head):
    token=hashlib.sha256(row["uid"].encode()).hexdigest()[:24]
    folder=run/"gpu"/("root_"+token)
    proofname=("XP_T_DAPS4K_ROOT_COMPLETE.json" if stage=="D2"
               else "XP_T_SITCOM_ROOT_COMPLETE.json")
    proof=jread(folder/proofname)
    if (proof.get("status")!=("XP_T_DAPS4K_ROOT_PASS" if stage=="D2" else "XP_T_SITCOM_ROOT_PASS")
        or proof.get("uid")!=row["uid"]
        or proof.get("runner_head")!=head
        or proof.get("manifest_sha256")!=manifest_hash
        or proof.get("seed")!=row["seed"]
        or proof.get("measurement_sha256")!=row["measurement_sha256"]
        or proof.get("gt_free_generation") is not True
        or proof.get("evaluation_outcomes_embargo")!="NOT_SCORED"
        or sha(safe(folder/"reconstruction.pt"))!=proof.get("reconstruction_pt_sha256")
        or sha(safe(folder/"reconstruction.png"))!=proof.get("reconstruction_png_sha256")):
        raise RuntimeError("STOP_A7_NATIVE_RECEIPT_"+str(row["uid"]))
    return folder/"reconstruction.pt"

def loader():
    if sha(COMMON)!=COMMON_Y_SHA:
        raise RuntimeError("STOP_A7_SHARED_Y_HASH")
    common=jread(COMMON)
    gt=jread(GT)
    if (common.get("n_images")!=100
        or common.get("gt_paths_supplied") is not False
        or gt.get("status")!="GT_ONLY_OFFLINE_NOT_FOR_GENERATION"
        or len(gt["rows"])!=100):
        raise RuntimeError("STOP_A7_GT_ONLY_PROVENANCE")
    yy={r["basename"]:r for r in common["rows"]}
    xx={r["basename"]:r for r in gt["rows"]}
    if len(xx)!=len(yy) or set(xx)!=set(yy):
        raise RuntimeError("STOP_A7_IMAGE_ID_CORRESPONDENCE")
    for key in xx:
        if (xx[key]["source_sha256"]!=yy[key]["source_image_sha256"]
            or not str(xx[key]["source_path"]).startswith(
                str(ROOT/"data/imagenet/test-imagenet")+"/")):
            raise RuntimeError("STOP_A7_GT_Y_PAIR")
    return xx,yy

def model_metric_setup():
    # We cannot silently substitute another LPIPS/SSIM convention.
    import torch
    try:
        import piq
        import lpips
    except ImportError as e:
        raise RuntimeError("STOP_A7_METRIC_DEPENDENCY_REQUIRES_RECORDED_SETUP: "+str(e))
    torch.set_num_threads(min(4,torch.get_num_threads()))
    # PyTorch's AlexNet feature weights must already be cached.  LPIPS
    # otherwise attempts a silent download (forbidden for this offline audit).
    from torchvision.models import AlexNet_Weights
    from urllib.parse import urlparse
    cache_name=Path(urlparse(AlexNet_Weights.IMAGENET1K_V1.url).path).name
    pretrained_cache=Path(torch.hub.get_dir())/"checkpoints"/cache_name
    if not pretrained_cache.is_file():
        raise RuntimeError(
            "STOP_A7_LPIPS_ALEXNET_PRETRAINED_NOT_CACHED "+str(pretrained_cache))
    try:
        net=lpips.LPIPS(net="alex",verbose=False).eval().cpu()
    except Exception as e:
        raise RuntimeError("STOP_A7_LPIPS_WEIGHTS_NOT_LOCALLY_AVAILABLE: "+str(e))
    return torch,piq,net

def metric_one(tensor,gt,torch,piq,lpips):
    # RGB8 canonical PSNR; float [0,1] SSIM; normalized [-1,1] Alex LPIPS.
    x8=to_rgb8(tensor)
    g8=to_rgb8(gt)
    psnr=float(psnr_rgb8(x8,g8)[0])
    xf=x8.float()/255
    gf=g8.float()/255
    with torch.inference_mode():
        ss=float(piq.ssim(xf,gf,data_range=1.0,reduction="mean"))
        lp=float(lpips(xf*2-1,gf*2-1).mean())
    return {"psnr_db":psnr,"ssim":ss,"lpips":lp}

def selectors(runs):
    if len(runs)!=4 or [r["rep"] for r in runs]!=[0,1,2,3]:
        raise RuntimeError("STOP_A7_SELECT_RUNS")
    first=runs[0]
    e0=min(runs,key=lambda x:(x["rho_signed"],x["rep"]))
    result={}
    for name,sign in (("psnr_db",1),("ssim",1),("lpips",-1)):
        oracle=max(runs,key=lambda x:(sign*x[name],-x["rep"]))
        repaired=max((x for r in runs for x in r["orbit"][name]),key=lambda x:sign*x)
        result[name]={
          "first":first[name], "E0_signed_rho":e0[name],
          "oracle_K4_per_metric":oracle[name],
          "O_orbit_raw_maximum":repaired,
          "E1_CLIP_G1":"PENDING_XP_C_CLIP_D1_WORKER"}
    return result

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A7_SCORING_NOT_CPU_ONLY")
    # Scorer belongs to an isolated A7 checkout, while the original
    # A6 experiment checkout MUST stay pinned at its generation HEAD.
    actual=subprocess.check_output(["git","-C",str(AUTHOR_WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(AUTHOR_WT),"status","--porcelain"],text=True).strip()
    running_head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    running_dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if (actual!=a.expected_head or dirty or
        running_head!=PINNED_A6_HEAD or running_dirty):
        raise RuntimeError("STOP_A7_SCORE_GIT_OR_ACTIVE_A6_WORKTREE")
    if a.stage not in ("D2","D3"):raise RuntimeError("STOP_A7_STAGE")
    out=Path(a.out)
    if (out.parent!=BASE or out.exists() or out.is_symlink()
        or not out.name.startswith("xp_a7_d2d3_")):
        raise RuntimeError("STOP_A7_OUTPUT")
    run,rows,msha,head=assert_completed(a.stage)
    sources,measurements=loader()
    torch,piq,lpips=model_metric_setup()
    from PIL import Image
    from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
    tf=Compose([ToTensor(),Resize(256),CenterCrop(256)])
    # Validate K4 native candidate records for all 400 before GT opening.
    files=[verify_one(run,a.stage,r,msha,head) for r in rows]
    if a.preflight_only:
        # Exhaustive 400/400 provenance checked above; now exercise one real
        # GT, y, reconstruction and metric evaluation.  No files are written.
        row=rows[0]
        iid=row["image_id"]
        if iid not in sources or iid not in measurements:
            raise RuntimeError("STOP_A7_SCORE_PREFLIGHT_ID")
        gtrow=sources[iid]
        measurement=measurements[iid]
        if (sha(safe(gtrow["source_path"]))!=gtrow["source_sha256"]
            or sha(safe(measurement["measurement_path"]))!=measurement["measurement_sha256"]
            or row["measurement_path"]!=measurement["measurement_path"]):
            raise RuntimeError("STOP_A7_SCORE_PREFLIGHT_GT_OR_Y_HASH")
        with Image.open(safe(gtrow["source_path"])) as im:
            g=tf(im.convert("RGB")).unsqueeze(0)*2-1
        y=torch.load(safe(measurement["measurement_path"]),map_location="cpu",
                     weights_only=True)["measurement"].float()
        x=torch.load(safe(files[0]),map_location="cpu",
                     weights_only=True)["reconstruction"].float()
        if tuple(x.shape)!=(1,3,256,256) or not torch.isfinite(x).all():
            raise RuntimeError("STOP_A7_SCORE_PREFLIGHT_X")
        m0=metric_one(x,g,torch,piq,lpips)
        m1=metric_one(apply_orbit(x,"111"),g,torch,piq,lpips)
        rho=float(rho_signed(x,y)[0])
        if (any(not math.isfinite(v) for v in [*m0.values(),*m1.values(),rho])
            or rho<0):
            raise RuntimeError("STOP_A7_SCORE_PREFLIGHT_NONFINITE")
        print("XP_A7_SCORE_PREFLIGHT_PASS stage="+a.stage+
              " roots_checked=400 metrics=PSNR_SSIM_LPIPS orbit=111" +
              " local_gt_only=true no_outputs=true",flush=True)
        return
    scores=[];cache={}
    with torch.inference_mode():
        for i,(r,f) in enumerate(zip(rows,files)):
            iid=r["image_id"]
            if iid not in sources:raise RuntimeError("STOP_A7_IMAGE_ID")
            if iid not in cache:
                if sha(safe(sources[iid]["source_path"]))!=sources[iid]["source_sha256"]:
                    raise RuntimeError("STOP_A7_GT_HASH")
                with Image.open(safe(sources[iid]["source_path"])) as im:
                    g=tf(im.convert("RGB")).unsqueeze(0)*2-1
                m=measurements[iid]
                if (r["measurement_path"]!=m["measurement_path"] or
                    r["measurement_sha256"]!=m["measurement_sha256"]
                    or sha(safe(m["measurement_path"]))!=m["measurement_sha256"]):
                    raise RuntimeError("STOP_A7_Y_HASH")
                y=torch.load(safe(m["measurement_path"]),map_location="cpu",
                    weights_only=True)["measurement"].float()
                cache[iid]=(g,y)
            g,y=cache[iid]
            x=torch.load(safe(f),map_location="cpu",
                weights_only=True)["reconstruction"].float()
            if (tuple(x.shape)!=(1,3,256,256)
                or not torch.isfinite(x).all()):
                raise RuntimeError("STOP_A7_RECON_TENSOR")
            mt=metric_one(x,g,torch,piq,lpips)
            orbit={}
            for orbitname in ("000","001","010","011","100","101","110","111"):
                z=apply_orbit(x,orbitname)
                mm=metric_one(z,g,torch,piq,lpips)
                for metric,val in mm.items():
                    orbit.setdefault(metric,[]).append(val)
            scores.append({"image_id":iid,"rep":r["rep"],"uid":r["uid"],
                "rho_signed":float(rho_signed(x,y)[0]),**mt,"orbit":orbit})
            if (i+1)%25==0:print("XP_A7_SCORE_PROGRESS",a.stage,i+1,"/400",flush=True)
    groups=collections.defaultdict(list)
    for item in scores:groups[item["image_id"]].append(item)
    if len(groups)!=100:raise RuntimeError("STOP_A7_IMAGE_COUNT")
    summary=[selectors(sorted(v,key=lambda x:x["rep"])) for k,v in sorted(groups.items())]
    means={metric:{policy:sum(row[metric][policy] for row in summary)/100
                   for policy in ("first","E0_signed_rho","oracle_K4_per_metric",
                                  "O_orbit_raw_maximum")}
           for metric in ("psnr_db","ssim","lpips")}
    result={
        "status":"XP_A7_K4_IMAGE_METRIC_AGGREGATE_PASS",
        "stage":a.stage,"family":"DAPS-4K" if a.stage=="D2" else "SITCOM",
        "n_images":100,"n_candidates":400,"runs":[0,1,2,3],
        "native_generation_head":head,
        "evaluation_head":a.expected_head,
        "scorer_source_sha256":sha(Path(__file__)),
        "native_manifest_sha256":msha,
        "metrics":["psnr_db","ssim","lpips"],
        "mean_over_images":means,
        "E1_CLIP_G1":"PENDING_CLIP_WORKER",
        "DAPS_published_Table1_reference":COMPARE_DAPS if a.stage=="D2" else None,
        "candidate_ground_truth_metrics_not_in_github_return":True,
        "all_k_gt4_outcomes_untouched":True}
    out.mkdir(parents=True,exist_ok=False)
    with (out/"XP_A7_K4_AGGREGATE.json").open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    # Per-run local diagnostics are NOT approved for theory_inputs.
    with (out/"XP_A7_K4_LOCAL_DETAIL.json").open("x") as f:
        json.dump({"raw_scores":scores},f,indent=2,allow_nan=False);f.write("\n")
    print("XP_A7_SCORING_PASS",a.stage,"400/400")
    print("AGGREGATE="+str(out/"XP_A7_K4_AGGREGATE.json"))
    print("K_GT4_OUTCOMES_UNTOUCHED=true")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("D2","D3"),required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_A7_SCORE "+str(e),flush=True)
        sys.exit(8)
