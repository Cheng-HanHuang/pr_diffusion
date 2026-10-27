#!/usr/bin/env python3
"""Native, GT-free DAPS-4K W=1 timing worker, pinned to pristine upstream.

Strict checkpoint load confirmation is mandatory: upstream DAPS catches and
ignores checkpoint failures, otherwise returning randomly initialized UNet.
No GT, no measurement generation, no historical outputs, no dataset imports.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

from scripts.track_r.xp_t_daps4k_prepare_cpu import (
    ROOT,WT,BASE,DAPS,DAPS_HEAD,PY_DAPS,DAPS4K,MODEL_SHA,
    IMAGENET_MODEL_SHA,new_seed,sha,read)
from scripts.track_r.xp_t_daps4k_native_cpu import (
    NATIVE,RECEIPT_DIR,verify_clean_sparse)
RUN=BASE/"xp_t_daps4k_20261027T011832Z"
PREP_HEAD="0a6a1b69dd3eef39ebfe95be4f2ce6b99abdac33"
MANIFEST_SHA="1f71f0bff208e1823ceb68fbe58a805c6b28241fe04da577efb35bdf65ed30c0"
NATIVE_SHA="f36a348c26e8dd76a5d72dc84ebe4e100b62d705a821c164e06c28a489433b23"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
EXPECTED_KEYS={"uid","family","dataset","image_id","measurement_source",
               "measurement_path","measurement_sha256","rep","seed"}

def dump_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def root_folder(row):
    token=hashlib.sha256(row["uid"].encode()).hexdigest()[:24]
    return RUN/"gpu"/("root_"+token)

def model_config(preset):
    return {
      "image_size":256,"num_channels":preset["num_channels"],
      "num_res_blocks":preset["num_res_blocks"],"channel_mult":"",
      "learn_sigma":True,"class_cond":False,"use_checkpoint":False,
      "attention_resolutions":preset["attention_resolutions"],
      "num_heads":4,"num_head_channels":64,"num_heads_upsample":-1,
      "use_scale_shift_norm":True,"dropout":0.0,
      "resblock_updown":True,"use_fp16":False,
      "use_new_attention_order":False,"model_path":preset["checkpoint"],
    }

def validate_weight_trace(trace):
    if not (trace.get("unet_calls")==1 and
            trace.get("strict") is True and
            trace.get("missing_keys")==[] and
            trace.get("unexpected_keys")==[] and
            isinstance(trace.get("tensor_count"),int) and
            trace["tensor_count"]>=100):
        raise RuntimeError("STOP_XPT_DAPS_UNVERIFIED_MODEL_CHECKPOINT")
    return True

def immutable_manifest(expected_head):
    if (os.environ.get("PYTHONDONTWRITEBYTECODE")!="1" or
        not os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
            str(RUN/"pycache")+"/") or
        Path(sys.executable).resolve()!=PY_DAPS.resolve()):
        raise RuntimeError("STOP_XPT_DAPS_ENV")
    if (subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
         text=True,timeout=30).strip()!=expected_head or
        subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
         text=True,timeout=30).strip()):
        raise RuntimeError("STOP_XPT_DAPS_WORKTREE")
    if (sha(RUN/"XP_T_DAPS4K_TASKS.json")!=MANIFEST_SHA or
        sha(RECEIPT_DIR/"XP_T_DAPS4K_NATIVE_SOURCE.json")!=NATIVE_SHA):
        raise RuntimeError("STOP_XPT_DAPS_FROZEN_SHA")
    prep=read(RUN/"XP_T_DAPS4K_PREP.json")
    source=read(RECEIPT_DIR/"XP_T_DAPS4K_NATIVE_SOURCE.json")
    if (prep.get("status")!="XP_T_DAPS4K_PREP_PASS" or
        prep.get("head")!=PREP_HEAD or
        prep.get("manifest_sha256")!=MANIFEST_SHA or
        prep.get("n_roots")!=18 or
        prep.get("gpu_work_performed") is not False or
        source.get("status")!="XP_T_DAPS4K_PRISTINE_NATIVE_PASS" or
        source.get("pinned_upstream_head")!=DAPS_HEAD or
        source.get("pristine_checkout")!=str(DAPS) or
        source.get("data_directory_materialized") is not False or
        source.get("old_checkout_mutated") is not False):
        raise RuntimeError("STOP_XPT_DAPS_PREP_SOURCE_RECEIPTS")
    verified=verify_clean_sparse(DAPS)
    if any(verified[k]["sha256"]!=source["tracked_files_verified"][k]["sha256"]
           for k in verified):
        raise RuntimeError("STOP_XPT_DAPS_SOURCE_SHA_DRIFT")
    mf=read(RUN/"XP_T_DAPS4K_TASKS.json")
    if (mf.get("schema")!="prdiff.track_r.xp.daps4k.timing.v1" or
        mf.get("status")!="XP_T_DAPS4K_TASKS_FROZEN" or
        mf.get("head")!=PREP_HEAD or
        mf.get("source_head")!=DAPS_HEAD or
        mf.get("source_path")!=str(DAPS) or
        mf.get("native_source_receipt_sha256")!=NATIVE_SHA or
        mf.get("settings")!=DAPS4K or
        mf.get("n_roots")!=18 or mf.get("n_images")!=6 or
        mf.get("runs_per_image")!=3 or
        mf.get("batch_size")!=1 or
        mf.get("gt_paths_supplied") is not False or
        mf.get("historical_paths_supplied") is not False or
        mf.get("historical_population_not_extended") is not True or
        mf.get("evaluation_outcomes_embargo")!="NOT_SCORED"):
        raise RuntimeError("STOP_XPT_DAPS_MANIFEST_SCHEMA")
    rows=mf.get("rows")
    if (not isinstance(rows,list) or len(rows)!=18 or
        len({x["uid"] for x in rows})!=18 or
        [x["dataset"] for x in rows]!=["ffhq"]*9+["imagenet"]*9 or
        any(set(x)!=EXPECTED_KEYS for x in rows)):
        raise RuntimeError("STOP_XPT_DAPS_ROW_SCHEMA")
    for name,digest in (("ffhq",MODEL_SHA),("imagenet",IMAGENET_MODEL_SHA)):
        preset=mf["model_presets"][name]
        if (preset["checkpoint_sha256"]!=digest or
            sha(preset["checkpoint"])!=digest):
            raise RuntimeError("STOP_XPT_DAPS_CHECKPOINT_SHA_"+name)
        if (preset["num_channels"],preset["num_res_blocks"],
            preset["attention_resolutions"])!=(
            (128,1,16) if name=="ffhq" else (256,2,"32,16,8")):
            raise RuntimeError("STOP_XPT_DAPS_CHECKPOINT_MODEL_ARCH")
    return mf,rows

def checked_row(mf,rows,index):
    if index not in range(18):
        raise RuntimeError("STOP_XPT_DAPS_BAD_INDEX")
    row=rows[index]
    ds=row["dataset"]
    expected="S1_b22" if ds=="ffhq" else "XP_IMAGENET100"
    if (row["family"]!="DAPS" or
        row["rep"]!=index%3 or
        row["uid"]!=f"XP_T|DAPS|{ds}|{row['image_id']}|rep{index%3}" or
        row["measurement_source"]!=expected or
        row["seed"]!=new_seed(ds,row["image_id"],index%3) or
        sha(row["measurement_path"])!=row["measurement_sha256"]):
        raise RuntimeError("STOP_XPT_DAPS_Y_OR_SEED")
    return row

def load_model_native(cfg):
    import torch
    from model import get_model
    original=torch.nn.Module.load_state_dict
    trace={"unet_calls":0}
    def checked(self,weights,*args,**kwargs):
        result=original(self,weights,*args,**kwargs)
        if self.__class__.__name__=="UNetModel":
            trace.update(
                unet_calls=trace["unet_calls"]+1,
                tensor_count=len(weights),
                missing_keys=list(result.missing_keys),
                unexpected_keys=list(result.unexpected_keys),
                strict=kwargs.get("strict",args[0] if args else True))
        return result
    t=time.perf_counter()
    torch.nn.Module.load_state_dict=checked
    try:
        model=get_model(name="ddpm",model_config=cfg)
    finally:
        torch.nn.Module.load_state_dict=original
    validate_weight_trace(trace)
    return model,time.perf_counter()-t,trace

def generate(a,mf,row):
    import numpy as np
    import torch
    from scripts.b22.b22_smoke_common import (
         load_measurement,validate_reconstruction,
         save_model_range_png,tensor_content_sha256)
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!=str(a.gpu) or
        not torch.cuda.is_available() or torch.cuda.device_count()!=1 or
        not 0<=a.gpu<4):
        raise RuntimeError("STOP_XPT_DAPS_GPU_ENV")
    dest=root_folder(row)
    if dest.parent!=RUN/"gpu" or dest.exists() or dest.is_symlink():
        raise RuntimeError("STOP_XPT_DAPS_ALREADY_GENERATED")
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    source_cwd=Path.cwd()
    sys.path.insert(0,str(DAPS))
    os.chdir(str(DAPS))
    try:
        # One independently seeded native DAPS process per candidate.
        np.random.seed(row["seed"])
        torch.manual_seed(row["seed"])
        torch.cuda.manual_seed_all(row["seed"])
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        from forward_operator import get_operator
        from sampler import get_sampler
        preset=mf["model_presets"][row["dataset"]]
        op=get_operator(**copy.deepcopy(mf["settings"]["operator"]))
        sampler=get_sampler(latent=False,
            annealing_scheduler_config=copy.deepcopy(
                mf["settings"]["annealing_scheduler_config"]),
            diffusion_scheduler_config=copy.deepcopy(
                mf["settings"]["diffusion_scheduler_config"]),
            mcmc_sampler_config=copy.deepcopy(
                mf["settings"]["mcmc_sampler_config"]))
        # DAPS Scheduler stores num_steps+1 for initial index. Native
        # sample() then runs exactly 400 annealing transitions.
        if (sampler.annealing_scheduler.num_steps!=401 or
            sampler.mcmc_sampler.num_steps!=100):
            raise RuntimeError("STOP_XPT_DAPS_NOT_4K")
        model,load_s,weight_trace=load_model_native(model_config(preset))
        obs=load_measurement(row["measurement_path"],device=device)
        if not bool(torch.isfinite(obs).all()):
            raise RuntimeError("STOP_XPT_DAPS_NONFINITE_MEASUREMENT")
        torch.cuda.reset_peak_memory_stats(device)
        start=torch.cuda.Event(enable_timing=True)
        finish=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        t=time.perf_counter()
        start.record()
        x_start=sampler.get_start(1,model)
        reconstruction=sampler.sample(
            model,x_start,op,obs,evaluator=None,
            record=False,verbose=False)
        finish.record()
        torch.cuda.synchronize(device)
        wall=time.perf_counter()-t
        cuda=float(start.elapsed_time(finish))/1000
        validate_reconstruction(reconstruction)
        # No outputs until full successful native sampling/validation.
        dest.mkdir(parents=True,exist_ok=False)
        tensor=reconstruction.detach().cpu().float()
        pt=dest/"reconstruction.pt"
        png=dest/"reconstruction.png"
        torch.save({"reconstruction":tensor},pt)
        save_model_range_png(tensor,png)
        receipt={
            "status":"XP_T_DAPS4K_ROOT_PASS",
            "uid":row["uid"],"dataset":row["dataset"],
            "image_id":row["image_id"],"rep":row["rep"],
            "seed":row["seed"],"family":"DAPS",
            "method":"DAPS-4K","model_load_strict":True,
            "model_load_trace":weight_trace,
            "model_config":model_config(preset),
            "solver_config":mf["settings"],
            "native_head":DAPS_HEAD,"runner_head":a.expected_head,
            "prep_head":PREP_HEAD,"manifest_sha256":MANIFEST_SHA,
            "source_receipt_sha256":NATIVE_SHA,
            "model_sha256":preset["checkpoint_sha256"],
            "measurement_sha256":row["measurement_sha256"],
            "native_sampling_wall_s":wall,
            "native_sampling_cuda_event_s":cuda,
            "native_model_loading_wall_s":load_s,
            "peak_gpu_allocated_mib":torch.cuda.max_memory_allocated(0)/2**20,
            "gpu_physical":a.gpu,
            "cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
            "reconstruction_sha256":tensor_content_sha256(tensor),
            "reconstruction_pt_sha256":sha(pt),
            "reconstruction_png_sha256":sha(png),
            "gt_free_generation":True,
            "historical_candidate_access":False,
            "evaluation_outcomes_embargo":"NOT_SCORED",
        }
        dump_x(dest/"XP_T_DAPS4K_ROOT_COMPLETE.json",receipt)
        print("XP_T_DAPS4K_ROOT_PASS",row["dataset"],row["image_id"],
              row["rep"],"gpu",a.gpu,"wall",round(wall,3),
              "cuda",round(cuda,3),flush=True)
        return 0
    finally:
        os.chdir(source_cwd)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--preflight-only",action="store_true")
    a=p.parse_args()
    mf,rows=immutable_manifest(a.expected_head)
    row=checked_row(mf,rows,a.index)
    if a.preflight_only:
        print("XP_T_DAPS4K_PREFLIGHT_PASS",a.index,row["dataset"],
              row["image_id"],flush=True)
        return 0
    return generate(a,mf,row)

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as err:
        traceback.print_exc()
        print("ACTION NEEDED: XP_T_DAPS4K_WORKER "+str(err),flush=True)
        sys.exit(8)
