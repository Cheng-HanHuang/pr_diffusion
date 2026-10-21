#!/usr/bin/env python3
"""A1.5 isolated native FAST-DIPS single-root runner (GPU, user-operated).

Must only run AFTER A1.5 CPU preflight and recorded issue #41 GPU lease.
Uses official pinned FAST-DIPS sampler unchanged, exact frozen signed A1.2
observation and FFHQ checkpoint bytes. GT is read AFTER sampling for score.
No new observations, no checkpoint download, no external FAST source edits.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from scripts.track_a.a15_fastdips_preflight import (
    CFG, sha, load, git, file_ok, tensor_hash, load_tensor, check_source)

def write_x(path:Path,obj):
    with path.open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def load_checkpoint_verified(net, ckpt_path, expected_sha):
    """Abort on upstream's documented silent random-initialization fallback."""
    import torch
    file_ok(ckpt_path,expected_sha,"native FFHQ DDPM checkpoint")
    src=torch.load(str(ckpt_path),map_location="cpu",weights_only=True)
    if not isinstance(src,dict) or not src:
        raise RuntimeError("STOP checkpoint is not nonempty UNet parameter dictionary")
    # FAST wrapper: DDPM.model (VPPrecond).model (native UNet)
    target=net.model.model.state_dict()
    if set(src)!=set(target):
        raise RuntimeError("STOP FAST checkpoint and model state keys mismatch")
    keys=sorted(src)
    for key in keys:
        a=src[key]; b=target[key]
        if not isinstance(a,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:
            raise RuntimeError("STOP FAST model parameter schema mismatch "+key)
    probes=sorted({0,len(keys)//7,len(keys)//4,len(keys)//2,
                   (3*len(keys))//4,(6*len(keys))//7,len(keys)-1})
    for idx in probes:
        k=keys[idx]
        if not torch.equal(target[k].detach().cpu(),src[k]):
            raise RuntimeError("STOP FAST model checkpoint NOT LOADED "+k)
    return {"checkpoint_sha256":expected_sha,"total_parameter_and_buffer_keys":len(keys),
            "matched_probe_keys":[keys[i] for i in probes]}

def result_psnr(saved_png, gt):
    import torch
    import numpy as np
    from PIL import Image
    with Image.open(saved_png) as im:
        a=torch.from_numpy(np.asarray(im.convert("RGB"),dtype=np.float32).copy())
    if tuple(a.shape)!=(256,256,3):
        raise RuntimeError("STOP output PNG shape")
    img=a.permute(2,0,1).unsqueeze(0)/255.
    gg=torch.round((gt.clamp(-1,1)+1)*127.5).clamp(0,255)/255.
    def psnr(v):
        mse=torch.mean((v-gg).square()).clamp_min(1e-12)
        return float(-10.*torch.log10(mse))
    return {"raw_rgb8_psnr_db":psnr(img),
            "rot180_rgb8_psnr_db":psnr(torch.rot90(img,2,(-2,-1)))}

def run(args,cfg,source):
    import numpy as np
    import torch
    from PIL import Image
    if args.image_id not in cfg["ids"] or args.rep not in (0,1,2,3):
        raise RuntimeError("STOP image/rep outside eight frozen exposed K4")
    if args.mode=="smoke":
        if args.image_id!="69073" or args.rep!=0:
            raise RuntimeError("STOP smoke only image69073 rep0")
        if args.lease_ack!="A15_SMOKE_LEASE_RECORDED_ISSUE41":
            raise RuntimeError("STOP explicit issue41 smoke GPU lease missing")
    elif args.mode=="grid":
        if args.lease_ack!="A15_GRID_LEASE_RECORDED_ISSUE41":
            raise RuntimeError("STOP explicit issue41 grid GPU lease missing")
    else:
        raise RuntimeError("STOP unsupported stage mode")
    g=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not g or "," in g or g=="-1":
        raise RuntimeError("STOP requires one explicitly mapped physical GPU ID")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("STOP exactly one CUDA device must be visible")

    receipt=load(Path(cfg["output_root"])/cfg["preflight_stage"]/"A15_PREFLIGHT_PASS.json")
    if receipt.get("status")!="A15_CPU_PREFLIGHT_PASS" or \
       receipt["source"]["a15_commit"]!=args.expected_head or \
       receipt["source"]["fast_head"]!=cfg["fast_head"]:
        raise RuntimeError("STOP A1.5 preflight not pinned to execution source")
    manifest=load(Path(cfg["frozen_input_manifest"]))
    row=next((x for x in manifest["rows"] if str(x["image_id"])==args.image_id),None)
    if row is None:raise RuntimeError("STOP source image not in frozen A1 manifest")
    measured=Path(row["measurement_path"])
    file_ok(measured,row["measurement_file_sha256"],"frozen observation")
    ckpt=Path(cfg["checkpoint"])
    file_ok(ckpt,cfg["checkpoint_sha256"],"native DDPM checkpoint")

    fast=Path(cfg["fast_repo"]);sys.path.insert(0,str(fast))
    import main as upstream
    import misc as settings
    from forward_operators import get_operator
    from model import get_model
    if Path(upstream.__file__).resolve()!=fast/"main.py":
        raise RuntimeError("STOP FAST-DIPS wrong import origin")

    device=torch.device("cuda:0")
    seed=15150000+int(args.image_id)+args.rep
    random.seed(seed);np.random.seed(seed % (2**32))
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    ddpm=copy.deepcopy(settings.MODEL_PRESETS["ffhq256ddpm"])
    if ddpm["name"]!="ddpm":
        raise RuntimeError("STOP official pretrained pixel DDPM preset drift")
    ddpm["model_config"]["model_path"]=str(ckpt)
    model=get_model(**ddpm,device="cuda:0",requires_grad=False)
    check=load_checkpoint_verified(model,ckpt,cfg["checkpoint_sha256"])
    model.eval()

    conf=cfg["fast_native_pixel"]
    official=copy.deepcopy(settings.SAMPLER_PRESETS[conf["sampler"]])
    if official.pop("latent") is not False:
        raise RuntimeError("STOP accidentally selected latent FAST-DIPS")
    schedule=official["diffusion_scheduler_config"]
    schedule["num_steps"]=conf["T"]
    sampler=upstream.FAST_DIPS(diffusion_scheduler_config=schedule,
        K=conf["K_ADMM"],S=conf["S_x"],rho=conf["rho"],
        epsilon=conf["epsilon"],backtrack=conf["backtrack"],
        alpha_max=conf["alpha_max"])
    operator=get_operator(name="phase_retrieval",oversample=2.0,
                          resolution=256,sigma=0.05)
    obs=load_tensor(measured,("measurement","y","observation"),(1,3,384,384))
    if tensor_hash(obs)!=row["measurement_tensor_sha256"]:
        raise RuntimeError("STOP signed observation tensor hash changed")
    signed_negative_frac=float((obs<0).float().mean())
    y=obs.to(device)
    x0=sampler.get_start(1,model)
    del obs

    out=Path(args.out)
    expected=(Path(cfg["output_root"])/cfg["smoke_stage"] if args.mode=="smoke"
              else Path(cfg["output_root"])/cfg["grid_stage"]/args.image_id/f"rep{args.rep}")
    if out!=expected or out.exists() or out.is_symlink():
        raise RuntimeError("STOP output mismatch or occupied")
    # Output creation begins only after successful source/model/measurement checks.
    out.mkdir(parents=True,exist_ok=False)
    (out/"RUNNING.json").write_text(json.dumps({
        "status":"INCOMPLETE_UNTIL_PASS","mode":args.mode,"a15_head":args.expected_head,
        "fast_head":cfg["fast_head"],"image_id":args.image_id,
        "rep":args.rep,"seed":seed,"visible_gpu":g,
        "checkpoint_sha256":cfg["checkpoint_sha256"],
        "measurement_tensor_sha256":row["measurement_tensor_sha256"]
    },indent=2)+"\n")
    print("A15_FAST_SMOKE_START",json.dumps({"image":args.image_id,"seed":seed,
        "gpu_visible":g,"source":args.expected_head}),flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_e=torch.cuda.Event(enable_timing=True)
    end_e=torch.cuda.Event(enable_timing=True)
    started=time.perf_counter()
    start_e.record()
    pred=sampler.sample(model,x0,operator,y,evaluator=None,
            record=False,verbose=False)
    end_e.record()
    torch.cuda.synchronize(device)
    wall=time.perf_counter()-started
    event=float(start_e.elapsed_time(end_e)/1000)
    native=float(sampler.last_sampling_time)
    if pred.shape!=(1,3,256,256) or not bool(torch.isfinite(pred).all()):
        raise RuntimeError("STOP FAST-DIPS output shape/nonfinite")
    if not (0<native<=wall*1.1 and 0<event<=wall*1.1):
        raise RuntimeError("STOP inconsistent sampling timers")
    mem=float(torch.cuda.max_memory_reserved(device)/(1024*1024))

    # Keep original native float terminal for later audits, on PAC only.
    original_cpu=pred.detach().cpu().float()
    import torch
    torch.save({"reconstruction":original_cpu},out/"reconstruction.pt")
    # Common PNG8 score export uses torchvision-style round-to-nearest,
    # not FAST's native truncating tensor_to_pils. Both conventions noted.
    rgba=torch.round(((original_cpu[0].clamp(-1,1)+1.)*.5)*255).to(torch.uint8)
    png=out/"reconstruction.png"
    Image.fromarray(rgba.permute(1,2,0).contiguous().numpy(),"RGB").save(png)
    gtpath=Path(row["ground_truth_tensor_path"])
    file_ok(gtpath,row["ground_truth_tensor_file_sha256"],"frozen GT")
    ground=load_tensor(gtpath,("ground_truth","image","x"),(1,3,256,256))
    if tensor_hash(ground)!=row["ground_truth_tensor_sha256"]:
        raise RuntimeError("STOP frozen GT tensor drift")
    metrics=result_psnr(png,ground)

    summary={"status":"A15_FAST_SINGLE_SMOKE_PASS","image_id":args.image_id,
        "rep":args.rep,"mode":args.mode,"seed":seed,
        "source_head":args.expected_head,"fast_head":cfg["fast_head"],
        "upstream_algorithm":"UNMODIFIED_MAIN_FAST_DIPS_PIXEL_T150_ADMM2_S5",
        "native_checkpoint_load_validation":check,
        "observation_type":"A1_SIGNED_NOISE_SAME_FROZEN_BYTES",
        "measurement_sha256":row["measurement_file_sha256"],
        "measurement_tensor_sha256":row["measurement_tensor_sha256"],
        "negative_observation_fraction":signed_negative_frac,
        "numpy_torch_seed":seed,"visible_physical_gpu_arg":g,
        "sample_only_wall_seconds":wall,
        "sample_cuda_event_seconds":event,
        "native_recorded_step_sum_seconds":native,
        "peak_torch_reserved_mib":mem,
        "reconstruction_float_sha256":sha(out/"reconstruction.pt"),
        "reconstruction_png_sha256":sha(png),
        "png_export":"round_to_nearest_torchvision_style_8bit",
        "gt_touched_only_post_sampling":True,
        "gt_free_runtime_selection":"single_trajectory_no_choice",
        "metric":metrics,
        "full_k4_oracle":"NOT_EXECUTED",
        "output":str(out),"no_new_measurement":True}
    write_x(out/"A15_SMOKE_COMPLETE.json",summary)
    print(json.dumps({"status":summary["status"],"image_id":args.image_id,
        "raw_psnr_db":metrics["raw_rgb8_psnr_db"],
        "wall_s":wall,"native_s":native,"event_s":event,
        "peak_mem_mib":mem,"complete_sha256":sha(out/"A15_SMOKE_COMPLETE.json")},
        sort_keys=True),flush=True)
    return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--mode",choices=["smoke","grid"],default="smoke")
    ap.add_argument("--image-id",required=True)
    ap.add_argument("--rep",type=int,required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--lease-ack",required=True)
    a=ap.parse_args()
    cfg=load(CFG)
    if os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP PYTHONDONTWRITEBYTECODE=1 mandatory")
    source=check_source(cfg,a.expected_head)
    return run(a,cfg,source)

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print(f"A15_GPU_SMOKE_STOP: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        sys.exit(8)
