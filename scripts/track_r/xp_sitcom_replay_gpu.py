#!/usr/bin/env python3
"""GT-free native SITCOM B22 sequential/B24 independent eight-root replay.

The only data path accepted on GPU is the eight-row frozen y manifest.
Historical candidates, GT, labels and scoring catalogs are forbidden.
B22: one seeded model, restore post-model RNG state per image; runs 0->1.
B24: one fresh process and newly seeded model per independent run.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random,subprocess,sys,time,types
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
SOURCE=ROOT/"external/SITCOM_ODE"
SITCOM_PY=ROOT/"conda-envs/sitcom_ode_bw/bin/python"
PREP_HEAD="c055423234441fe907fef8dfda825815632ca15f"
RUN_NAME="xp_sitcom_replay_20261026T223925Z"
MANIFEST_SHA="82d38a2cadc47c8ad864e97809e9cea9baafd62c261fa26d50fc1d332d48d525"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def safe(path,allow_symlink=False):
    p=Path(path)
    real=p.resolve(strict=False)
    top=ROOT.resolve(strict=False)
    if (not p.is_absolute() or not p.is_file() or
        (p.is_symlink() and not allow_symlink) or
        not (real==top or top in real.parents) or
        any(t in str(p).lower() or t in str(real).lower() for t in BAD)):
        raise RuntimeError("STOP_SITCOM_REPLAY_UNSAFE "+str(p))
    return p

def sha(path):
    h=hashlib.sha256()
    with safe(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def dump(path,record):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(record,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def git_head(path):
    return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],
                                   text=True,timeout=30).strip()

def check_manifest(payload,head):
    expected={"uid","source","family","measurement_source","image_id","rep",
              "seed","rng_mode","measurement_path","measurement_sha256"}
    if (payload.get("status")!="XP_SITCOM_REPLAY_TASKS_FROZEN" or
        payload.get("schema")!="prdiff.track_r.xp.sitcom-replay-tasks.v1" or
        payload.get("head")!=PREP_HEAD or payload.get("native_source_head")!=
           "275ab67efbd8146bffca20155171ba6be1169c09" or
        payload.get("n_roots")!=8 or payload.get("n_images")!=4 or
        payload.get("dataset")!="ffhq" or
        payload.get("gt_paths_supplied") is not False or
        payload.get("historical_paths_supplied") is not False or
        payload.get("model_sha256")!=
          "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa" or
        payload.get("b22_master_seed")!=43 or
        payload.get("b22_trajectory_count")!=4 or
        not payload.get("native_checkout_provenance",{}).get("generation_require_isolated_pycache")):
        raise RuntimeError("STOP_SITCOM_REPLAY_MANIFEST")
    rows=payload.get("rows",[])
    if (len(rows)!=8 or any(set(r)!=expected for r in rows) or
        len({r["uid"] for r in rows})!=8):
        raise RuntimeError("STOP_SITCOM_REPLAY_FIELDS")
    for idx,r in enumerate(rows):
        source="S1_b22" if idx<4 else "S2_b24_dev80"
        mode="sequential_post_model" if idx<4 else "independent_model_seed"
        image=(("60044","60140") if idx<4 else ("01113","02763"))[((idx%4)//2)]
        if (r["source"]!=source or r["measurement_source"]!=source or
            r["family"]!="SITCOM" or r["rng_mode"]!=mode or
            r["image_id"]!=image or r["rep"]!=idx%2 or
            r["uid"]!=(f"{source}|SITCOM|{image}|"+("run" if idx<4 else "rep")+str(idx%2)) or
            not isinstance(r["seed"],int) or
            (r["seed"]!=43 if idx<4 else not 0<=r["seed"]<2**32)):
            raise RuntimeError("STOP_SITCOM_REPLAY_ORDER_OR_SEED")
    return rows

def validate_preflight(a):
    if Path(sys.executable).resolve()!=SITCOM_PY.resolve():
        raise RuntimeError("STOP_SITCOM_REPLAY_WRONG_PYTHON")
    if (os.environ.get("PYTHONDONTWRITEBYTECODE")!="1" or
        os.environ.get("CUDA_VISIBLE_DEVICES")!=str(a.gpu) or
        not (0<=a.gpu<=3) or
        not os.environ.get("PYTHONPYCACHEPREFIX","").startswith(
            str(BASE/RUN_NAME)+"/pycache")):
        raise RuntimeError("STOP_SITCOM_REPLAY_UNISOLATED_GPU_OR_CACHE")
    if (git_head(WT)!=a.expected_head or
        subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
                                text=True,timeout=30).strip()):
        raise RuntimeError("STOP_SITCOM_REPLAY_WORKTREE")
    if git_head(SOURCE)!="275ab67efbd8146bffca20155171ba6be1169c09":
        raise RuntimeError("STOP_SITCOM_REPLAY_NATIVE_HEAD")
    from scripts.track_r.xp_sitcom_replay_prepare_cpu import (
        check_native_checkout,external_checkpoint_provenance,settings)
    source_state=check_native_checkout(
        git_head(SOURCE),
        subprocess.check_output(["git","-C",str(SOURCE),"status",
                                  "--porcelain=v1","--untracked-files=no"],text=True),
        subprocess.check_output(["git","-C",str(SOURCE),"diff","--numstat",
                                  "HEAD","--","posterior_sample.py"],text=True))
    model_proof=external_checkpoint_provenance(SOURCE/"checkpoint/ffhq256.pt")
    out=Path(a.out)
    if (out.parent!=BASE/RUN_NAME/"gpu" or out.is_symlink() or
        (out.exists() and not a.preflight_only)):
        raise RuntimeError("STOP_SITCOM_REPLAY_GPU_OUTPUT")
    mf=safe(BASE/RUN_NAME/"XP_SITCOM_REPLAY_TASKS.json")
    if sha(mf)!=MANIFEST_SHA:raise RuntimeError("STOP_SITCOM_REPLAY_MANIFEST_HASH")
    prep=json.loads(safe(BASE/RUN_NAME/"XP_SITCOM_REPLAY_PREP.json").read_text())
    if (prep.get("status")!="XP_SITCOM_REPLAY_PREP_PASS" or
        prep.get("head")!=PREP_HEAD or prep.get("manifest_sha256")!=MANIFEST_SHA or
        prep.get("n_roots")!=8):
        raise RuntimeError("STOP_SITCOM_REPLAY_PREP_HASH")
    manifest=json.loads(mf.read_text())
    rows=check_manifest(manifest,a.expected_head)
    # Source facts are held by the manifest; reject drift during replay.
    saved=manifest["native_checkout_provenance"]
    if (saved.get("tracked_modified")!=source_state["tracked_modified"] or
        saved.get("unused_cli_sha256")!=sha(SOURCE/"posterior_sample.py") or
        manifest.get("native_checkpoint_provenance")!=model_proof or
        settings(json.loads(safe(WT/"configs/b22/b22_2_overnight.json").read_text()))
            !=manifest.get("config")):
        raise RuntimeError("STOP_SITCOM_REPLAY_NATIVE_PROVENANCE")
    for row in rows:
        if sha(row["measurement_path"])!=row["measurement_sha256"]:
            raise RuntimeError("STOP_SITCOM_REPLAY_Y_HASH "+row["uid"])
    if a.mode=="b22":
        chosen=rows[:4]
    elif a.mode=="b24":
        if a.index not in range(4,8):
            raise RuntimeError("STOP_SITCOM_REPLAY_B24_INDEX")
        chosen=[rows[a.index]]
    else:raise RuntimeError("STOP_SITCOM_REPLAY_MODE")
    return manifest,chosen

def folder(out,row):
    return Path(out)/("root_"+hashlib.sha256(row["uid"].encode()).hexdigest()[:24])

def model_config():
    return {
      "attention_resolutions":16,"channel_mult":"","class_cond":False,
      "dropout":0.0,"image_size":256,"learn_sigma":True,
      "model_path":"checkpoint/ffhq256.pt","num_channels":128,
      "num_head_channels":64,"num_heads":4,"num_heads_upsample":-1,
      "num_res_blocks":1,"resblock_updown":True,"use_checkpoint":False,
      "use_fp16":False,"use_new_attention_order":False,
      "use_scale_shift_norm":True,
    }

def instantiate(get_sampler,get_model,get_operator,cfg,device,
                seed_after_sampler=None):
    operator=get_operator(name="phase_retrieval",sigma=0.05,oversample=2.0)
    sampler=get_sampler(
       latent=False,
       annealing_scheduler_config={
         "num_steps":cfg["anneal_steps"],"schedule":cfg["anneal_schedule"],
         "sigma_max":cfg["anneal_sigma_max"],"sigma_min":cfg["anneal_sigma_min"],
         "sigma_final":0.0,"timestep":cfg["timestep"]},
       diffusion_scheduler_config={
         "num_steps":cfg["diff_steps"],"schedule":cfg["diff_schedule"],
         "sigma_min":cfg["diff_sigma_min"],"sigma_final":0.0,
         "timestep":cfg["timestep"]},
       lgvd_config={
         "lr":cfg["lgvd_lr"],"lr_min_ratio":cfg["lgvd_lr_min_ratio"],
         "num_steps":cfg["lgvd_steps"],"tau":cfg["lgvd_tau"]})
    # B22 seeded AFTER operator/sampler construction and BEFORE model
    # construction; B24 seeded before importing the native implementation.
    # Keep these historical orders distinct.
    if seed_after_sampler is not None:
        import numpy as np
        import torch
        np.random.seed(seed_after_sampler)
        torch.manual_seed(seed_after_sampler)
        torch.cuda.manual_seed_all(seed_after_sampler)
    t=time.perf_counter()
    model=get_model(name="ddpm",model_config=model_config())
    import torch
    torch.cuda.synchronize(device)
    return operator,sampler,model,time.perf_counter()-t

def generate(a,manifest,rows):
    import numpy as np
    import torch
    from PIL import Image
    from scripts.b22.b22_smoke_common import (
        load_measurement,validate_reconstruction,
        tensor_content_sha256,save_model_range_png)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("STOP_SITCOM_REPLAY_NEED_ONE_GPU")
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    # Preserve the historical source import order for each cohort.
    if a.mode=="b22":
        sys.path.insert(0,str(SOURCE))
        oldcwd=Path.cwd()
        os.chdir(SOURCE)
        from forward_operator import get_operator
        from model import get_model
        from sampler import get_sampler
        # Seed in instantiate(), *after* native operator/sampler creation.
    else:
        seed=int(rows[0]["seed"])
        np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        sys.path.insert(0,str(SOURCE))
        oldcwd=Path.cwd()
        os.chdir(SOURCE)
        from forward_operator import get_operator
        from model import get_model
        from sampler import get_sampler
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    saved=[];wall=time.perf_counter()
    try:
        op,sampler,model,model_load_s=instantiate(
            get_sampler,get_model,get_operator,manifest["config"],device,
            seed_after_sampler=43 if a.mode=="b22" else None)
        if a.mode=="b22":
            post_np=np.random.get_state()
            post_torch=torch.get_rng_state().clone()
            post_cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
        previous_image=None
        for row in rows:
            if a.mode=="b22" and row["image_id"]!=previous_image:
                np.random.set_state(post_np)
                torch.set_rng_state(post_torch.clone())
                torch.cuda.set_rng_state_all([x.clone() for x in post_cuda])
                previous_image=row["image_id"]
            observation=load_measurement(row["measurement_path"],device=device)
            if not bool(torch.isfinite(observation).all()):
                raise RuntimeError("STOP_SITCOM_REPLAY_NONFINITE_Y")
            # The native get_start() is solely randn_like(ref) * sigma_max.
            # Empty shape-only reference has no GT content and consumes no RNG.
            reference=torch.empty((1,3,256,256),dtype=torch.float32,device=device)
            steps=[0]
            original_record=sampler._record
            def lightweight(self,xt,x0y,x0hat,sigma,x0hat_results,x0y_results):
                # Historical B22 code computes selector ONLY at zero-based
                # annealing step 160 and does not retain a trajectory.
                if steps[0]==160:
                    correction=torch.sqrt(torch.mean(
                        (x0y.detach().float()-x0hat.detach().float()).square()
                    ).clamp_min(1e-24))
                    selector[0]=float(correction.detach().cpu().item())
                steps[0]+=1
            selector=[None]
            torch.cuda.reset_peak_memory_stats(device)
            begin_evt=torch.cuda.Event(enable_timing=True)
            finish_evt=torch.cuda.Event(enable_timing=True)
            # B22 historically ran record=True with lightweight _record.
            if a.mode=="b22":
                sampler._record=types.MethodType(lightweight,sampler)
            try:
                ts=time.perf_counter()
                begin_evt.record()
                x_start=sampler.get_start(reference)
                reconstructed=sampler.sample(
                    model,x_start,op,observation,
                    evaluator=None,record=a.mode=="b22",verbose=False)
                finish_evt.record()
                torch.cuda.synchronize(device)
                sample_wall=time.perf_counter()-ts
                sample_cuda=float(begin_evt.elapsed_time(finish_evt))/1000
            finally:
                sampler._record=original_record
            validate_reconstruction(reconstructed)
            if a.mode=="b22" and (steps[0]!=200 or selector[0] is None):
                raise RuntimeError("STOP_SITCOM_REPLAY_B22_SELECTOR")
            reconstruction=reconstructed.detach().cpu().float()
            dest=folder(out,row)
            dest.mkdir(parents=False,exist_ok=False)
            tensor=dest/"reconstruction.pt"
            png=dest/"reconstruction.png"
            torch.save({"reconstruction":reconstruction},tensor)
            save_model_range_png(reconstruction,png)
            receipt={
              "status":"XP_SITCOM_REPLAY_ROOT_PASS",
              "uid":row["uid"],"image_id":row["image_id"],
              "source":row["source"],"rep":row["rep"],
              "rng_mode":row["rng_mode"],"seed":row["seed"],
              "runner_head":a.expected_head,
              "prep_head":PREP_HEAD,"manifest_sha256":MANIFEST_SHA,
              "native_head":manifest["native_source_head"],
              "model_sha256":manifest["model_sha256"],
              "measurement_sha256":row["measurement_sha256"],
              "cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
              "sampling_wall_s":sample_wall,
              "sampling_cuda_event_s":sample_cuda,
              "model_load_s_worker_shared":model_load_s,
              "reconstruction_content_sha256":tensor_content_sha256(reconstruction),
              "reconstruction_pt_sha256":sha(tensor),
              "reconstruction_png_sha256":sha(png),
              "selector_norm":selector[0],
              "native_steps":200,"diffusion_steps_per_anneal":5,
              "lgvd_steps_per_anneal":100,
              "gt_access":False,"historical_access":False}
            dump(dest/"XP_SITCOM_ROOT_COMPLETE.json",receipt)
            saved.append({"uid":row["uid"],"file_sha256":sha(dest/"XP_SITCOM_ROOT_COMPLETE.json")})
            print("XP_SITCOM_ROOT_PASS",row["uid"],"wall_s",round(sample_wall,3),
                  "cuda_s",round(sample_cuda,3),flush=True)
            del reconstructed,reconstruction,observation,reference,x_start
            torch.cuda.empty_cache()
        report={"status":"XP_SITCOM_WORKER_PASS",
                "mode":a.mode,"index":a.index if a.mode=="b24" else None,
                "head":a.expected_head,"prep_head":PREP_HEAD,
                "manifest_sha256":MANIFEST_SHA,"new_roots":len(saved),
                "model_load_s":model_load_s,
                "worker_wall_s":time.perf_counter()-wall,
                "roots":saved}
        dump(out/"XP_SITCOM_WORKER_COMPLETE.json",report)
        print("XP_SITCOM_WORKER_PASS",a.mode,len(saved),flush=True)
        return 0
    finally:
        os.chdir(oldcwd)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--mode",choices=("b22","b24"),required=True)
    p.add_argument("--index",type=int,default=-1)
    p.add_argument("--out",required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--preflight-only",action="store_true")
    a=p.parse_args()
    manifest,rows=validate_preflight(a)
    if a.preflight_only:
        print("XP_SITCOM_PREFLIGHT_PASS mode="+a.mode+" rows="+str(len(rows)))
        return
    return generate(a,manifest,rows)

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("ACTION NEEDED: XP_SITCOM_GPU "+str(exc),flush=True)
        sys.exit(8)
