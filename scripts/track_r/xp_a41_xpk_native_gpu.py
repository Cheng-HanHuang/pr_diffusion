#!/usr/bin/env python3
"""A24 XP-K ONE parameter-only native family worker (replay and calibration).

NO ground truth, historical candidates, or scores are accepted. Same native
worker and method construction in both modes, calibrated mode updates
ONLY the registered scalar plus preregistered new RNG seeds.
No method/optimizer/schedule/operator change. Provenance pinned.
Called image-complete by the A41 supervisor; durable per-root receipts.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,random,shutil,subprocess,sys,time,traceback,types
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=Path(__file__).resolve().parents[2]
BASE=ROOT/"outputs/pr_diffusion/track_r"
SITCOM=ROOT/"external/SITCOM_ODE"
FAST=ROOT/"external/FAST_DIPS_A15"
DAPS=ROOT/"pr_diffusion_b19_solver/external/daps"
HEADS={"SITCOM":"275ab67efbd8146bffca20155171ba6be1169c09",
       "FAST":"a9758b38906b7c7950fa04491f37f6634948db26",
       "DAPS":"e7a77d094167084faed19b599b96673b7bb11447"}
TARGET={"SITCOM":.05,"FAST":33.26,"DAPS":math.sqrt(2)*.05}
ORIGINAL={"SITCOM":.01,"FAST":.05,"DAPS":.01}
MODEL_SHA="81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
MODEL=ROOT/"models/ffhq_10m.pt"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def gate(b,k):
    if not b:raise RuntimeError("STOP_XPK_NATIVE_"+k)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):h.update(buf)
    return h.hexdigest()
def git(wt,*args):
    return subprocess.check_output(["git","-C",str(wt),*args],text=True,timeout=45).strip()
def write(path,data):
    p=Path(path)
    with p.open("x") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
def load(path):
    p=Path(path)
    gate(p.is_file() and not p.is_symlink(),"MISSING_SOURCE")
    return json.loads(p.read_text())
def safey(p,h):
    p=Path(p)
    r=p.resolve()
    gate(p.is_absolute() and p.is_file() and not p.is_symlink() and
         ROOT.resolve() in r.parents and
         not any(x in str(p).lower() or x in str(r).lower() for x in BAD) and
         sha(p)==h,"MEASUREMENT_IDENTITY_OR_PROTECTED")
    return p
def valid_input(a):
    gate(git(WT,"rev-parse","HEAD")==a.expected_head and
         not git(WT,"status","--porcelain"),"CODE_PIN")
    gate(a.family in TARGET and a.phase in ("historical_replay","calibrated"),"MODE")
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a41_xpk_") and
         run.is_dir() and not run.is_symlink(),"OUTPUT_ROOT")
    mpath=run/"XPK_GPU_GT_FREE_MANIFEST.json"
    gate(sha(mpath)==a.manifest_sha,"FROZEN_MANIFEST_SHA")
    rec=load(run/"XPK_SOURCE_FREEZE_RECEIPT.json")
    gate(rec["sha_manifest"]==a.manifest_sha
         and rec["code_head"]==a.expected_head,"MANIFEST_RECEIPT")
    m=load(mpath)
    gate(m["status"]=="XP_K_A24_FROZEN_GT_FREE_100x4"
         and m["GT_access"] is False if False else
         m["original_GT_access"] is False
         and m["reference_paths_in_GPU_manifest"] is False,
         "GT_FREE_MANIFEST")
    rows=m["historical_replay_rows"][a.family] if a.phase=="historical_replay" else m["calibrated_rows"][a.family]
    selected=[r for r in rows if r["image_id"]==a.image_id]
    n=2 if a.phase=="historical_replay" else 4
    gate(len(selected)==n and sorted(r["run"] for r in selected)==list(range(n))
         and all(r["GT_supplied"] is False and
                 r["historical_candidate_supplied"] is False and
                 r["stage"]==a.phase and r["family"]==a.family
                 for r in selected),"ROOTS_EXACT_PANEL")
    for r in selected:
        safey(r["measurement_path"],r["measurement_sha256"])
    src={"SITCOM":SITCOM,"FAST":FAST,"DAPS":DAPS}[a.family]
    gate(git(src,"rev-parse","HEAD")==HEADS[a.family],"PINNED_NATIVE_HEAD")
    if a.family=="DAPS":
        gate(git(src,"hash-object","sampler.py")==
            "ae1fc6908bb1e7f0c0349d4f82cdf20364389db8",
            "HISTORICAL_DAPS_PATCHED_SAMPLER")
    gate(sha(MODEL)==MODEL_SHA,"FFHQ_MODEL_WEIGHTS")
    dest=run/"gpu"/a.family/a.phase/a.image_id
    gate(dest.parent==run/"gpu"/a.family/a.phase and not dest.is_symlink(),
         "STAGE_DEST")
    if a.preflight_only:
        print("XPK_NATIVE_SOURCE_AND_Y_PREFLIGHT_PASS",
              a.family,a.phase,a.image_id,len(selected),flush=True)
        return None,None,None
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu)
         and a.gpu in range(4) and
         os.environ.get("PYTHONDONTWRITEBYTECODE")=="1"
         and os.environ.get("PYTHONNOUSERSITE")=="1",
         "STRICT_GPU_ENV")
    for f,python in {
      "SITCOM":ROOT/"conda-envs/sitcom_ode_bw/bin/python",
      "FAST":ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python",
      "DAPS":ROOT/"conda-envs/daps/bin/python"}.items():
        if a.family==f:gate(Path(sys.executable).resolve()==python.resolve(),
                            "PINNED_NATIVE_ENV")
    dest.mkdir(parents=True,exist_ok=True)
    gate(all(ch.name.startswith(("run","root_")) or ch.name in ("pycache",)
             for ch in dest.iterdir()),"UNEXPECTED_NATIVE_OUTPUT")
    return m,selected,dest

def rootdir(dest,row):
    return dest/f"run{int(row['run']):02d}"

def complete_valid(dest,row,family,phase):
    p=rootdir(dest,row)/"XPK_ROOT_COMPLETE.json"
    if not p.exists():return False
    d=load(p)
    gate(d["status"]=="XPK_NATIVE_ROOT_PASS" and
         d["uid"]==row["uid"] and d["family"]==family and
         d["phase"]==phase and d["run"]==row["run"] and
         d["seed"]==row["seed"] and
         d["measurement_sha256"]==row["measurement_sha256"] and
         d["changed_config_parameter"]==
          ("none" if phase=="historical_replay" else
           {"SITCOM":"lgvd_tau","FAST":"epsilon","DAPS":"task.pixel.mcmc_sampler_config.tau"}[family])
         and sha(rootdir(dest,row)/"reconstruction.png")==d["png_sha256"],
         "ROOT_RECEIPT_DRIFT")
    return True

def record(dest,row,a,elapsed,method,extra=None):
    folder=rootdir(dest,row)
    gate((folder/"reconstruction.png").is_file(),"SAMPLE_MISSING")
    receipt=dict(status="XPK_NATIVE_ROOT_PASS",
       family=a.family,phase=a.phase,
       uid=row["uid"],dataset=row["dataset"],
       image_id=row["image_id"],run=row["run"],seed=row["seed"],
       model_sha256=MODEL_SHA,
       native_git_head=HEADS[a.family],code_head=a.expected_head,
       measurement_sha256=row["measurement_sha256"],
       native_method=method,elapsed_gpu_occupied_wall_s=elapsed,
       physical_gpu=a.gpu,
       original_value=ORIGINAL[a.family],
       bound_parameter_value=(ORIGINAL[a.family] if a.phase=="historical_replay"
                              else TARGET[a.family]),
       changed_config_parameter=("none" if a.phase=="historical_replay" else
            {"SITCOM":"lgvd_tau","FAST":"epsilon","DAPS":"task.pixel.mcmc_sampler_config.tau"}[a.family]),
       png_sha256=sha(folder/"reconstruction.png"),
       gt_access=False,historical_candidate_access=False,
       native_model_setting_except_scalar_identical=True,
       no_new_selector=True)
    if extra:receipt.update(extra)
    write(folder/"XPK_ROOT_COMPLETE.json",receipt)
    print("XPK_NATIVE_ROOT_PASS",a.family,a.phase,row["image_id"],row["run"],
          "gpu_seconds",round(elapsed,2),flush=True)

def sit_run(rows,dest,a):
    import numpy as np, torch
    from scripts.track_r.xp_sitcom_replay_gpu import instantiate,model_config
    from scripts.b22.b22_smoke_common import (
        load_measurement,validate_reconstruction,save_model_range_png)
    gate(torch.cuda.is_available() and torch.cuda.device_count()==1,"SITCOM_GPU")
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    srcpath=Path.cwd()
    sys.path.insert(0,str(SITCOM))
    os.chdir(SITCOM)
    try:
        from forward_operator import get_operator
        from model import get_model
        from sampler import get_sampler
        cfg=load(WT/"configs/b22/b22_2_overnight.json")["sitcom"]
        gate(float(cfg["lgvd_tau"])==ORIGINAL["SITCOM"],"SITCOM_HIST_CONFIG")
        if a.phase=="calibrated":cfg["lgvd_tau"]=TARGET["SITCOM"]
        dev=torch.device("cuda:0")
        op,sampler,model,model_load=instantiate(
            get_sampler,get_model,get_operator,cfg,dev,
            seed_after_sampler=(43 if a.phase=="historical_replay" else int(rows[0]["seed"])))
        # Historical B22 replays use the same post-model stochastic stream
        # for run0 then run1; calibrated seeds are independent by XP rule.
        for row in rows:
            if a.phase=="calibrated":
                seed=int(row["seed"])
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            folder=rootdir(dest,row)
            if complete_valid(dest,row,"SITCOM",a.phase):
                # Replay still must consume previous stream on a resume,
                # otherwise later historical roots would use wrong RNG.
                if a.phase=="calibrated":
                    continue
            else:
                gate(not folder.exists(),"SITCOM_NO_PARTIAL_REPLACE")
                folder.mkdir()
            y=load_measurement(row["measurement_path"],device=dev)
            ref=torch.empty((1,3,256,256),dtype=torch.float32,device=dev)
            # B22 original selector uses native lightweight record at
            # step 160, not a changed diffusion/Euler step.
            ctr=[0];selector=[None];original_record=sampler._record
            def lightweight(self,xt,x0y,x0hat,sigma,x0hat_results,x0y_results):
                if ctr[0]==160:
                    val=torch.sqrt(torch.mean(
                        (x0y.detach().float()-x0hat.detach().float()).square()
                    ).clamp_min(1e-24))
                    selector[0]=float(val.detach().cpu().item())
                ctr[0]+=1
            sampler._record=types.MethodType(lightweight,sampler)
            start=time.monotonic()
            try:
                start_x=sampler.get_start(ref)
                rec=sampler.sample(model,start_x,op,y,evaluator=None,
                                   record=True,verbose=False)
                torch.cuda.synchronize(dev)
            finally:sampler._record=original_record
            elapsed=time.monotonic()-start
            validate_reconstruction(rec)
            gate(ctr[0]==int(cfg["anneal_steps"]) and selector[0] is not None,
                 "SITCOM_NATIVE_STEP_RECORD")
            if not (folder/"XPK_ROOT_COMPLETE.json").exists():
                reccpu=rec.detach().float().cpu()
                torch.save({"reconstruction":reccpu},folder/"reconstruction.pt")
                save_model_range_png(reccpu,folder/"reconstruction.png")
                record(dest,row,a,elapsed,"SITCOM_ODE",
                       dict(selector_step160_norm=selector[0],
                            native_anneal_steps=int(cfg["anneal_steps"]),
                            Euler_step_code_unchanged=True,
                            model_load_wall_s=model_load))
            del rec,y,ref,start_x
    finally:os.chdir(srcpath)

def fast_run(rows,dest,a):
    import numpy as np,torch
    from PIL import Image
    from scripts.track_r.xp_fast_gpu import validate_native_conf
    from scripts.track_a.a15_fastdips_smoke import load_checkpoint_verified
    from scripts.track_a.a15_fastdips_preflight import load_tensor
    gate(torch.cuda.is_available() and torch.cuda.device_count()==1,
         "FAST_GPU")
    prop=torch.cuda.get_device_properties(0)
    gate((prop.major,prop.minor)==(12,0),"FAST_BLACKWELL_HARDWARE")
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    srcpath=Path.cwd()
    sys.path.insert(0,str(FAST))
    os.chdir(FAST)
    try:
        import main as upstream
        import misc as settings
        from forward_operators import get_operator
        from model import get_model
        gate(Path(upstream.__file__).resolve()==FAST/"main.py",
             "FAST_PINNED_MAIN_IMPORT")
        config=load(WT/"configs/track_a/a15_fastdips_exposed_v1.json")
        frozen=load(WT/"configs/track_r/r0_frozen_v1.json")
        conf=validate_native_conf(config,frozen)
        gate(float(conf["epsilon"])==ORIGINAL["FAST"],"FAST_HIST_EPS")
        if a.phase=="calibrated":conf["epsilon"]=TARGET["FAST"]
        device=torch.device("cuda:0")
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
        ddpm=copy.deepcopy(settings.MODEL_PRESETS["ffhq256ddpm"])
        gate(ddpm["name"]=="ddpm","FAST_DDPM")
        ddpm["model_config"]["model_path"]=str(MODEL)
        network=get_model(**ddpm,device="cuda:0",requires_grad=False)
        load_checkpoint_verified(network,MODEL,MODEL_SHA)
        network.eval()
        for row in rows:
            folder=rootdir(dest,row)
            if complete_valid(dest,row,"FAST",a.phase):continue
            gate(not folder.exists(),"FAST_NO_PARTIAL_REPLACE")
            seed=int(row["seed"])
            random.seed(seed);np.random.seed(seed%(2**32))
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            preset=copy.deepcopy(settings.SAMPLER_PRESETS[conf["sampler"]])
            gate(preset.pop("latent") is False,"FAST_LATENT")
            preset["diffusion_scheduler_config"]["num_steps"]=conf["T"]
            sampler=upstream.FAST_DIPS(
               diffusion_scheduler_config=preset["diffusion_scheduler_config"],
               K=conf["K_ADMM"],S=conf["S_x"],rho=conf["rho"],
               epsilon=conf["epsilon"],backtrack=conf["backtrack"],
               alpha_max=conf["alpha_max"])
            op=get_operator(name="phase_retrieval",oversample=2.0,
                            resolution=256,sigma=.05)
            y=load_tensor(row["measurement_path"],("measurement","y","observation"),
                          (1,3,384,384)).to(device)
            x0=sampler.get_start(1,network)
            folder.mkdir()
            start=time.monotonic()
            # FAST x update REQUIRES gradients; do not use no_grad.
            rec=sampler.sample(network,x0,op,y,evaluator=None,
                               record=False,verbose=False)
            torch.cuda.synchronize(device)
            elapsed=time.monotonic()-start
            gate(tuple(rec.shape)==(1,3,256,256) and
                 bool(torch.isfinite(rec).all()),"FAST_OUTPUT")
            cpu=rec.detach().float().cpu()
            torch.save({"reconstruction":cpu},folder/"reconstruction.pt")
            png=torch.round((cpu[0].clamp(-1,1)+1)*127.5).byte()
            Image.fromarray(png.permute(1,2,0).numpy(),"RGB").save(
                folder/"reconstruction.png")
            record(dest,row,a,elapsed,"FAST_DIPS",
                   dict(T=int(conf["T"]),K_ADMM=int(conf["K_ADMM"]),
                        S_x=int(conf["S_x"]),rho=float(conf["rho"])))
            del sampler,op,y,x0,rec,cpu
    finally:os.chdir(srcpath)

def daps_run(rows,dest,a):
    from PIL import Image
    from scripts.track_r.xp_f8_legacy_replay_gpu import (
        native_command,env,check_cfg,TRACE_SOURCE)
    from scripts.track_r.xp_f8_legacy_preflight_cpu import MODEL_SHA as F8_MODEL_SHA
    gate(F8_MODEL_SHA==MODEL_SHA,"DAPS_HIST_CHECKPOINT_SHA")
    for row in rows:
        folder=rootdir(dest,row)
        if complete_valid(dest,row,"DAPS",a.phase):continue
        gate(not folder.exists(),"DAPS_NO_PARTIAL_REPLACE")
        folder.mkdir()
        synthetic_dir=folder/"synthetic_input"
        synthetic_dir.mkdir()
        placeholder=synthetic_dir/"00000.png"
        Image.new("RGB",(256,256),color=(0,0,0)).save(placeholder)
        hook=folder/"sitehook"
        hook.mkdir()
        (hook/"sitecustomize.py").write_text(TRACE_SOURCE)
        trace=folder/"model_load_trace.json"
        f8row={"native_seed":int(row["seed"]),
               "measurement_path":str(row["measurement_path"])}
        cmd=native_command(f8row,folder)
        if a.phase=="calibrated":
            cmd.append("task.pixel.mcmc_sampler_config.tau="+repr(TARGET["DAPS"]))
        resolved=check_cfg(cmd,f8row)
        tau=float(resolved["pixel"]["mcmc_sampler_config"]["tau"])
        gate(math.isclose(tau,ORIGINAL["DAPS"] if a.phase=="historical_replay"
                          else TARGET["DAPS"],rel_tol=0,abs_tol=1e-10),
             "DAPS_RESOLVED_SCALAR")
        native_log=folder/"daps_native_stdout.log"
        start=time.monotonic()
        with native_log.open("x") as f:
            response=subprocess.run(cmd,cwd=str(DAPS),env=env(f8row,a.gpu,trace),
                                    stdout=f,stderr=subprocess.STDOUT,
                                    check=False)
        elapsed=time.monotonic()-start
        gate(response.returncode==0,"DAPS_NATIVE_RC_"+str(response.returncode))
        loader=load(trace)
        gate(loader["class_name"]=="UNetModel"
             and loader["strict"] is True and
             loader["missing_keys"]==[] and loader["unexpected_keys"]==[],
             "DAPS_ORIGINAL_CHECKPOINT_TRACE")
        src=folder/"legacy_daps400x5/samples/00000_run0000.png"
        gate(src.is_file() and src.stat().st_size>1000,"DAPS_NATIVE_PNG")
        with Image.open(src) as im:
            gate(im.size==(256,256) and im.mode in ("RGB","RGBA"),
                 "DAPS_NATIVE_IMAGE")
            im.verify()
        shutil.copy2(src,folder/"reconstruction.png")
        record(dest,row,a,elapsed,"DAPS_HISTORICAL_400x5",
               dict(native_source_sampler_blob="ae1fc6908bb1e7f0c0349d4f82cdf20364389db8",
                    original_Hydra_mcmc_tau=tau,
                    native_stdout_sha256=sha(native_log),
                    original_native_png_sha256=sha(src),
                    model_load_trace_sha256=sha(trace),
                    placeholder_only_for_shape=True))
        
def main(a):
    m,rows,dest=valid_input(a)
    if rows is None:return 0
    if a.family=="SITCOM":sit_run(rows,dest,a)
    elif a.family=="FAST":fast_run(rows,dest,a)
    else:daps_run(rows,dest,a)
    for row in rows:gate(complete_valid(dest,row,a.family,a.phase),
                        "INCOMPLETE_GROUP_"+row["uid"])
    report=dict(status="XPK_NATIVE_IMAGE_GROUP_PASS",family=a.family,
         phase=a.phase,image_id=a.image_id,
         head=a.expected_head,manifest_sha=a.manifest_sha,
         n_roots=len(rows),root_receipt_shas=[sha(rootdir(dest,r)/"XPK_ROOT_COMPLETE.json")
           for r in rows],no_GT_on_GPU=True,
         calibrated=(a.phase=="calibrated"))
    done=dest/"XPK_GROUP_COMPLETE.json"
    if done.exists():gate(load(done)==report,"GROUP_COMPLETE_DRIFT")
    else:write(done,report)
    print("XPK_NATIVE_IMAGE_GROUP_PASS",a.family,a.phase,a.image_id,len(rows),
          flush=True)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--manifest-sha",required=True)
    p.add_argument("--family",choices=("SITCOM","FAST","DAPS"),required=True)
    p.add_argument("--phase",choices=("historical_replay","calibrated"),required=True)
    p.add_argument("--image-id",required=True)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--preflight-only",action="store_true")
    try:sys.exit(main(p.parse_args()))
    except Exception as exc:
        traceback.print_exc()
        print("ACTION_NEEDED_XPK_NATIVE",repr(exc),flush=True)
        sys.exit(8)
