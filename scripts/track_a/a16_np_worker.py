#!/usr/bin/env python3
"""One A16 NP experiment on exactly one GPU, one exposed B22 image and one seed.

NOT a PAC cluster launcher. Never writes ground-truth or measurement arrays
outside original PAC. Native B22 source/model/FFHQ observations are frozen.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT=Path("/egr/research-pac/huang248")
A16_ROOT=ROOT/"outputs/pr_diffusion/track_a_np_paper/a16_np_fast_autopilot_20261021"
A15_ROOT=ROOT/"outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020"
CHECKOUT=ROOT/"pr_diffusion_track_a_a16_np_fast_autopilot"
MODEL=ROOT/"models/ffhq_10m.pt"
DIFFFPR=ROOT/"external/DiffFPR"
MODEL_SHA="81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
INPUT_SHA="fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490"
DIFFFPR_HEAD="a45ffe58f18fed8a63d3446600424e2b08733524"
SELECTOR_BLOB="36c00612b7b7d44e992534edb9d694d05a73fefd"
BASE_BLOB="ed4d0049f086c8e0792501d8dc568a1b24263d4c"
IDS=("69073","68539","67948","60140","67293","67174","61101","66631")
GPU_UUIDS={
    "0":"GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3",
    "1":"GPU-883c037a-34d2-48c4-467f-9a352fd8fdff",
    "2":"GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe",
    "3":"GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a",
}


def git(path,*args):
    return subprocess.check_output(["git","-C",str(path),*args],text=True).strip()

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def load(path):
    with open(path,encoding="utf8") as f:return json.load(f)

def save_exclusive(path,obj):
    with open(path,"x",encoding="utf8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--arm-json",required=True)
    ap.add_argument("--image-id",required=True,choices=IDS)
    ap.add_argument("--rep",required=True,type=int,choices=(0,1,2,3))
    ap.add_argument("--out",required=True)
    ap.add_argument("--gpu-uuid",required=True)
    ap.add_argument("--reference-replay",action="store_true")
    args=ap.parse_args()

    if os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP bytecode writes disabled")
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if visible not in GPU_UUIDS or args.gpu_uuid!=GPU_UUIDS[visible]:
        raise RuntimeError("STOP GPU physical mapping or UUID mismatch")
    if git(CHECKOUT,"rev-parse","HEAD")!=args.expected_head:
        raise RuntimeError("STOP source HEAD mismatch")
    if git(CHECKOUT,"status","--porcelain","--untracked-files=no"):
        raise RuntimeError("STOP dirty source checkout")
    if git(DIFFFPR,"rev-parse","HEAD")!=DIFFFPR_HEAD:
        raise RuntimeError("STOP DiffFPR backend drift")
    if git(CHECKOUT,"hash-object","scripts/pr_external_difffpr_np_guided_lf_s2_selector.py")!=SELECTOR_BLOB:
        raise RuntimeError("STOP original native NP selector source drift")
    if git(CHECKOUT,"hash-object","scripts/pr_external_difffpr_np_benchmark.py")!=BASE_BLOB:
        raise RuntimeError("STOP original NP math library drift")
    if digest(MODEL)!=MODEL_SHA:
        raise RuntimeError("STOP original FFHQ checkpoint hash")
    manifest_path=ROOT/"outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_inputs_v1/A1_2_INPUT_FREEZE_COMPLETE.json"
    if digest(manifest_path)!=INPUT_SHA:
        raise RuntimeError("STOP original input manifest hash")
    manifest=load(manifest_path)
    entry=next((x for x in manifest["rows"] if str(x["image_id"])==args.image_id),None)
    if entry is None:raise RuntimeError("STOP image outside frozen eight")
    from scripts.track_a.a16_np_hybrid_core import Arm,reconstruct
    arm=Arm(**load(Path(args.arm_json)))
    arm.validate()

    expected=A16_ROOT/"runs"/arm.name/args.image_id/f"rep{args.rep}"
    out=Path(args.out)
    if out != expected or out.exists() or out.is_symlink():
        raise RuntimeError("STOP output root mismatch or occupied")
    if digest(entry["measurement_path"])!=entry["measurement_file_sha256"]:
        raise RuntimeError("STOP saved signed observation byte drift")
    if digest(entry["ground_truth_tensor_path"])!=entry["ground_truth_tensor_file_sha256"]:
        raise RuntimeError("STOP GT tensor byte drift")
    # Real CUDA import and memory gate only after frozen provenance verified.
    import torch
    import numpy as np
    from PIL import Image
    from scripts.track_a.a15_fastdips_preflight import load_tensor,tensor_hash
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("STOP requires exactly one visible CUDA GPU")
    if torch.cuda.get_device_capability(0)!=(12,0):
        raise RuntimeError("STOP expected Blackwell GPU arch sm120")
    y=load_tensor(Path(entry["measurement_path"]),("measurement","y","observation"),(1,3,384,384))
    if tensor_hash(y)!=entry["measurement_tensor_sha256"]:
        raise RuntimeError("STOP signed tensor content hash")
    y=y.to("cuda:0")
    sfile=CHECKOUT/"scripts/pr_external_difffpr_np_guided_lf_s2_selector.py"
    selector=module_at("a16_frozen_b22_np_selector",sfile)
    import torch.backends.cudnn
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device=torch.device("cuda:0")
    # Stable predeclared seed per image/rep, paired across all A16 arms.
    seed=16160000+int(args.image_id)+args.rep
    boot=time.perf_counter()
    bundle=selector.load_guided_diffusion_model(
        model_path=str(MODEL),device=device,preset="difffpr_ffhq_10m",
        guided_diffusion_dir=str(DIFFFPR),strict=True)
    torch.cuda.synchronize(device)
    model_load=time.perf_counter()-boot

    out.mkdir(parents=True,exist_ok=False)
    save_exclusive(out/"RUNNING.json",{
        "source_head":args.expected_head,"image":args.image_id,"rep":args.rep,
        "seed":seed,"arm":arm.name,"gpu_uuid":args.gpu_uuid,"status":"INCOMPLETE"})

    baseline_diff=None
    baseline_native_time=0.0
    if args.reference_replay:
        if arm.proposal!="native" or arm.correction!="none" or arm.placement!="none" or arm.steps!=1000:
            raise RuntimeError("STOP reference replay requires exact NP native arm")
        v=selector.base.NPVariant(name="np_canonical_soft5_hard1",soft=5,hard=1,
                     proj_start=300,use_lowfreq_score=True,use_lowfreq_projection=True)
        t0=time.perf_counter()
        native,_=selector.reconstruct_with_selector_stat(
            y.clamp_min(0),pad=64,seed=seed,unet=bundle.unet,
            scheduler=bundle.scheduler,device=device,variant=v,
            num_steps=1000,score_radius=.6,proj_radius=.2,
            proj_radius_schedule="300:0.2",score_mode="lf" if arm.family=="LF" else "prev_l2",
            score_reg_lambda=0. if arm.family=="LF" else .01,
            score_reg_lambda_schedule="constant" if arm.family=="LF" else "pre_projection_only",
            score_huber_delta=.05,log_every=0)
        torch.cuda.synchronize(device)
        baseline_native_time=time.perf_counter()-t0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_event=torch.cuda.Event(enable_timing=True)
    end_event=torch.cuda.Event(enable_timing=True)
    start_wall=time.perf_counter()
    start_event.record()
    pred,stats=reconstruct(selector=selector,model=bundle.unet,
                           scheduler=bundle.scheduler,measurement_signed=y,
                           pad=64,seed=seed,arm=arm)
    end_event.record()
    torch.cuda.synchronize(device)
    sampling_wall=time.perf_counter()-start_wall
    event_s=float(start_event.elapsed_time(end_event)/1000.)

    if args.reference_replay:
        baseline_diff=float((pred-native).abs().amax())
        if baseline_diff>5e-4:
            raise RuntimeError(f"STOP native NP replay mismatch maxabs {baseline_diff}")
    with torch.no_grad():
        # Post-sampling scored once for offline GT-free selection; excluded
        # from sampler CUDA-event/sampling-wall timers above.
        yhat=selector.base.oversampled_magnitude(pred.detach(),64)
        residual_signed_l2=float((yhat-y).square().sum().sqrt())
        residual_clipped_l2=float((yhat-y.clamp_min(0)).square().sum().sqrt())
    pred=pred.detach().cpu().float()
    if pred.shape!=(1,3,256,256) or not bool(torch.isfinite(pred).all()):
        raise RuntimeError("STOP invalid NP output")
    torch.save({"reconstruction":pred},out/"reconstruction.pt")
    rendered=torch.round(((pred[0].clamp(-1,1)+1)*.5)*255).clamp(0,255).byte()
    png=out/"reconstruction.png"
    Image.fromarray(rendered.permute(1,2,0).contiguous().numpy(),"RGB").save(png)
    # Strict no-GT in sampler: load GT only after full trajectory/float/png exist.
    gt=load_tensor(Path(entry["ground_truth_tensor_path"]),
                   ("ground_truth","image","x"),(1,3,256,256))
    if tensor_hash(gt)!=entry["ground_truth_tensor_sha256"]:
        raise RuntimeError("STOP frozen GT tensor hash")
    gg=torch.round(((gt[0].clamp(-1,1)+1)*.5)*255).clamp(0,255)/255.
    rr=rendered.float()/255.
    psnr=lambda z:float(-10*torch.log10(((z-gg)**2).mean().clamp_min(1e-12)))
    raw=psnr(rr)
    rot=psnr(torch.rot90(rr,2,(-2,-1)))
    result=dict(status="A16_NP_CANDIDATE_PASS",source_head=args.expected_head,
        arm=arm.__dict__,image_id=args.image_id,rep=args.rep,seed=seed,
        gpu_uuid=args.gpu_uuid,checkpoint_sha256=MODEL_SHA,
        measurement_content_sha256=entry["measurement_tensor_sha256"],
        measurement_preprocessing=arm.target,output_directory=str(out),
        raw_rgb8_psnr_db=raw,rot180_rgb8_psnr_db=rot,
        good25=raw>=25,good30=raw>=30,bad20=raw<20,
        signed_full_measurement_residual_l2=residual_signed_l2,
        clipped_full_measurement_residual_l2=residual_clipped_l2,
        residual_score_is_offline_post_sampling=True,
        sampling_wall_s=sampling_wall,sample_cuda_event_s=event_s,
        program_wall_after_imports_s=time.perf_counter()-boot-baseline_native_time,
        reference_native_replay_seconds=baseline_native_time,
        model_load_s=model_load,
        max_torch_reserved_mib=torch.cuda.max_memory_reserved(device)/(1024*1024),
        np_native_replay_maxabs=baseline_diff,
        counters=stats["counters"],trace=stats["trace"],
        reconstruction_sha256=digest(out/"reconstruction.pt"),
        reconstruction_png_sha256=digest(png),
        gt_used_only_after_sampling=True)
    save_exclusive(out/"A16_CANDIDATE_COMPLETE.json",result)
    print(json.dumps(dict(status=result["status"],image=args.image_id,rep=args.rep,
        arm=arm.name,psnr_db=raw,sampling_s=sampling_wall,unet=stats["counters"]["unet"],
        native_replay_diff=baseline_diff),sort_keys=True),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as e:
        print(f"A16_NP_WORKER_STOP: {type(e).__name__}: {e}",file=sys.stderr,flush=True)
        sys.exit(17)
