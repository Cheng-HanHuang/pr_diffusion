#!/usr/bin/env python3
"""A20 XP-C3 *only* S_mid global2 GPU scoring, exactly 32 denoiser examples/root.

Every candidate is authenticated by a Git-COMMITTED source manifest.
No GT paths, no PSNR, no 2-forward secondary. ImageNet D1 checkpoint only.
Resumable by complete SHA-verified image chunks; no output overwrites.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
SRC_REL="docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
FROZEN_REL="docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"
MODEL_SHA="a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"
PANELS=("XP_N","G5","XP_D4")
SIZES={"XP_N":6400,"G5":800,"XP_D4":6400}
T=(300,400,500,600)
J=range(4)
COLUMNS=("image_id","run","label","prior_choice","prior_margin","prior_chosen_good")
DAPS_PY=ROOT/"conda-envs/daps/bin/python"

def gate(x,s):
    if not x:raise RuntimeError("STOP_A20_XPC3_"+s)

def sha(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink(),"FILE_"+p.name)
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def jread(p):return json.loads(Path(p).read_text())

def git(*argv):return subprocess.check_output(
    ["git","-C",str(WT),*argv],text=True,timeout=60).strip()

def source(a):
    from scripts.track_r.xp_t_daps4k_prepare_cpu import (
        NATIVE,IMAGENET_MODEL_SHA,verify_clean_sparse)
    from prdiffusion.track_r.prior_score import (
        frozen_alpha_schedule,prior_seed)
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"WORKTREE_CLEAN_PIN")
    gate(Path(sys.executable).resolve()==DAPS_PY.resolve() and
         os.environ.get("PYTHONDONTWRITEBYTECODE")=="1" and
         os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu),
         "NATIVE_ENV_GPU")
    gate(a.gpu in range(4),"GPU_RANGE")
    gate(verify_clean_sparse(NATIVE),"NATIVE_SOURCE_DRIFT")
    import torch
    gate(torch.cuda.is_available(),"GPU_UNAVAILABLE")
    gate(sha(ROOT/"models/imagenet256.pt")==MODEL_SHA and
         IMAGENET_MODEL_SHA==MODEL_SHA,"D1_CHECKPOINT_SHA")
    src=WT/SRC_REL;frozen=WT/FROZEN_REL
    gate(git("ls-files","--error-unmatch",SRC_REL)==SRC_REL and
         git("ls-files","--error-unmatch",FROZEN_REL)==FROZEN_REL,
         "SOURCE_MANIFEST_MUST_BE_COMMITTED")
    spec=jread(frozen)
    gate(spec["status"]=="A20_XPC3_OPTION_B_S_MID_GLOBAL2_ONLY"
         and spec["source_manifest_sha256"]==sha(src)
         and spec["checkpoint_sha256"]==MODEL_SHA
         and spec["n_candidates"]==13600
         and spec["timesteps"]==list(T)
         and spec["draws"]==list(J)
         and spec["decision_threshold_exclusive"]==1e-7,
         "FROZEN_SPEC_SCHEMA")
    gate(sha(WT/spec["scorer_script"])==spec["scorer_script_sha256"],
         "FROZEN_CODE_SHA")
    gate(sha(WT/"prdiffusion/track_r/prior_score.py")==
         spec["original_prior_score_sha256"],
         "ORIGINAL_R0_PRIOR_SHA")
    mf=jread(src)
    gate(mf["status"]=="A20_XPC3_ALL13600_SOURCE_FROZEN_UNSCORED"
         and mf["checkpoint_sha256"]==MODEL_SHA
         and mf["n_candidates"]==13600
         and mf["panel_order"]==list(PANELS)
         and mf["sizes"]==SIZES
         and len(mf["rows"])==13600,
         "SOURCE_FROZEN_SCHEMA")
    rows=[r for r in mf["rows"] if r["panel"]==a.panel]
    gate(len(rows)==SIZES[a.panel]
         and len({r["uid"] for r in rows})==len(rows),
         "PANEL_ROWS")
    for r in rows:
        gate(r["uid"]==f'XPC3|{a.panel}|{r["image_id"]}|{r["run"]}'
             and r["label"] in ("G","O_g","O_p","C","I"),
             "SOURCE_UID_LABEL")
    groups=defaultdict(list)
    for r in rows:groups[r["image_id"]].append(r)
    gate(len(groups)==100 and all(
        sorted(x["run"] for x in rr)==list(range(
            8 if a.panel=="G5" else 64))
        for rr in groups.values()),"ALL_COMPLETE_100_IMAGES")
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a25_xpc3_prod_")
         and run.is_dir() and not run.is_symlink(),"RUN")
    chunk=run/a.panel
    gate(chunk.is_dir(),"PANEL_OUTPUT_DIR")
    return spec,mf,groups,chunk,NATIVE,torch,frozen_alpha_schedule,prior_seed


def fix_exact_native_timestep(model,alpha,torch):
    """Force A20 discrete DDPM time while retaining frozen alpha_t.

    Native VPPrecond sigma_inv maps the frozen beta-product sigma through
    a continuous approximation. Exact t conditioning is required by A20.
    Only the sigma -> timestep embedding mapping is specialized; pretrained
    weights and Tweedie input/output scaling stay untouched.
    """
    native=model.model
    gate(native.M==1000 and abs(float(native.beta_min)-.1)<1e-6
         and abs(float(native.beta_d)-19.9)<1e-6,
         "NATIVE_DDPM_1000_CONTRACT")
    witness={}
    for t in T:
        at=float(alpha[t])
        sig=math.sqrt((1-at)/at)
        witness[t]=sig
        mapped=float(native.sigma_inv(
           torch.as_tensor(sig,dtype=torch.float32,device="cuda:0")))*999.
        gate(abs(mapped-t)<3.0,
             "NATIVE_TIME_MAP_TOO_DIFFERENT_"+str(t))
    def exact_index(sigma):
        v=torch.as_tensor(sigma)
        gate(v.numel()>=1,"SIGMA_NONEMPTY")
        z=[float(x) for x in v.detach().cpu().flatten()]
        choices=[t for t,sv in witness.items()
                 if all(abs(x-sv)<=max(5e-6,abs(sv)*2e-6)
                        for x in z)]
        gate(len(choices)==1,
             "SIGMA_NOT_ONE_OF_A20_FROZEN_TIMESTEPS")
        return torch.full_like(v,choices[0]/999.)
    native.sigma_inv=exact_index
    for t,sigma in witness.items():
        tick=native.sigma_inv(torch.tensor(
             sigma,device="cuda:0",dtype=torch.float32))
        gate(abs(float(tick)*999.-t)<1e-4,
             "A20_EXACT_MODEL_TIME_WITNESS")
    return {str(t):dict(frozen_sigma=witness[t],model_t=t) for t in T}

def orient_sm(x,uid,model,torch,alpha,seed_fn):
    from prdiffusion.track_r.physics import apply_orbit
    # Gaussian eps on CPU; rotate the SAME eps to both orbit members.
    gate(tuple(x.shape)==(1,3,256,256)
         and x.dtype==torch.float32 and bool(torch.isfinite(x).all()),
         "CANONICAL_X")
    x111=apply_orbit(x,"111")
    losses=[0.,0.]
    n=0
    with torch.inference_mode():
        for t in T:
            at=float(alpha[t])
            sigma=math.sqrt((1-at)/at)
            for j in J:
                rng=torch.Generator(device="cpu").manual_seed(
                      seed_fn(uid,t,j))
                eps=torch.randn((1,3,256,256),generator=rng,
                                dtype=torch.float32).to(x.device)
                e111=apply_orbit(eps,"111")
                # VP-scaled input: z = (sqrt(a)Rx + sqrt(1-a)Reps)/sqrt(a).
                z=torch.cat((x+sigma*eps,x111+sigma*e111))
                target=torch.cat((eps,e111))
                estimated_clean=model.tweedie(z,sigma)
                gate(tuple(estimated_clean.shape)==tuple(z.shape)
                     and bool(torch.isfinite(estimated_clean).all()),
                     "TWEEDIE_NATIVE_CLEAN")
                eps_hat=(z-estimated_clean)/sigma
                each=(eps_hat-target).square().flatten(1).mean(1)
                losses[0]+=float(each[0]);losses[1]+=float(each[1])
                n+=1
    gate(n==16 and all(math.isfinite(v) for v in losses),
         "A20_32_EXAMPLES_FINITE")
    l000=losses[0]/n;l111=losses[1]/n
    margin=l000-l111
    return dict(prior_choice="111" if margin>1e-7 else "000",
                prior_margin=format(margin,".15g"),
                L000=format(l000,".15g"),L111=format(l111,".15g"),
                denoiser_examples=32)

def run(a):
    spec,mf,groups,folder,native,torch,schedule,seed_fn=source(a)
    if a.preflight_only:
        print("A20_XPC3_SOURCE_SHA_PREFLIGHT_PASS",a.panel,
              len(groups),"SOURCE_SHA",sha(WT/SRC_REL),flush=True)
        return
    gate(a.limit_images is None or 1<=a.limit_images<=100,
         "LIMIT_IMAGE_RANGE")
    t0=time.monotonic()
    sys.path.insert(0,str(native))
    prev=Path.cwd()
    os.chdir(str(native))
    try:
        from scripts.track_r.xp_t_daps4k_worker_gpu import (
          model_config,load_model_native,validate_weight_trace)
        torch.cuda.set_device(0)
        cfg=dict(checkpoint=str(ROOT/"models/imagenet256.pt"),
              num_channels=256,num_res_blocks=2,
              attention_resolutions="32,16,8")
        model,load_seconds,trace=load_model_native(model_config(cfg))
        validate_weight_trace(trace)
        model.eval()
        alpha=schedule()
        mapping=fix_exact_native_timestep(model,alpha,torch)
        print("A20_XPC3_FROZEN_DISCRETE_T_MAPPING",
              json.dumps(mapping,sort_keys=True),flush=True)
        done=0
        selected=sorted(groups)[:a.limit_images] if a.limit_images else sorted(groups)
        for idx,iid in enumerate(selected):
            path=folder/("%03d.json"%sorted(groups).index(iid))
            expected=sorted(groups[iid],key=lambda r:r["run"])
            if path.exists():
                prior=jread(path)
                gate(prior["status"]=="A20_XPC3_IMAGE_PRIOR_PASS"
                     and prior["image_id"]==iid
                     and len(prior["rows"])==len(expected)
                     and prior["source_manifest_sha256"]==sha(WT/SRC_REL),
                     "IMAGE_RESUME_PROOF")
                continue
            values=[]
            for row in expected:
                pt=Path(row["pt_path"])
                gate(sha(pt)==row["pt_sha256"],"NATIVE_RECON_HASH")
                source=torch.load(pt,map_location="cpu",weights_only=True)
                gate(isinstance(source,dict) and
                     isinstance(source.get("reconstruction"),torch.Tensor),
                     "NATIVE_RECON_FORMAT")
                x=source["reconstruction"].float().to("cuda:0")
                vals=orient_sm(x,row["uid"],model,torch,alpha,seed_fn)
                values.append(dict(image_id=iid,run=row["run"],
                                   label=row["label"],**vals))
                done+=1
            payload=dict(status="A20_XPC3_IMAGE_PRIOR_PASS",
                panel=a.panel,image_id=iid,
                source_manifest_sha256=sha(WT/SRC_REL),
                checkpoint_sha256=MODEL_SHA,
                no_GT_access=True,rows=values)
            with path.open("x") as f:
                json.dump(payload,f,sort_keys=True,allow_nan=False)
                f.write("\n")
            if (idx+1)%5==0:
                print("A20_XPC3_IMAGE",a.panel,idx+1,"/100",
                      "time_s",round(time.monotonic()-t0,1),flush=True)
        torch.cuda.synchronize()
        print("A20_XPC3_WORKER_PASS",a.panel,
              "new_candidates",done,
              "seconds",round(time.monotonic()-t0,2),
              "limit_images",a.limit_images,flush=True)
    finally:
        os.chdir(prev)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("expected-head","run","panel"):
        parser.add_argument("--"+arg,required=True)
    parser.add_argument("--gpu",required=True,type=int)
    parser.add_argument("--limit-images",type=int)
    parser.add_argument("--preflight-only",action="store_true")
    try:
        a=parser.parse_args()
        gate(a.panel in PANELS,"PANEL")
        run(a)
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A20_XPC3_GPU",str(exc),flush=True)
        sys.exit(8)
