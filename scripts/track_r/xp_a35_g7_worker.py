#!/usr/bin/env python3
"""A35 GT-only per-dataset persistent worker for A23 registered G7 sequence.

One native diffusion model per worker (not per image); one canonical UnivFD
model on CPU. Each image saved atomically, SHA checked against A22 frozen
280 GT manifest. No inverse-problem solver, no historical candidate access.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
GT=BASE/"xp_c_c2_a22_20261104T025355Z/XP_C_GT_IMAGES.json"
LOCK=WT/"docs/track_r/executors/xp/theory_inputs/XP_A23_G7_REMAINING_SOURCE_LOCK.json"
STATUS="A35_G7_GT_ONE_IMAGE_COMPLETE"

def gate(x,code):
    if not x:raise RuntimeError("STOP_A35_GT_WORKER_"+code)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for v in iter(lambda:f.read(1<<20),b""):h.update(v)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def git(root,*args):
    return subprocess.check_output(["git","-C",str(root),*args],
                                   text=True,timeout=50).strip()
def atomic(path,obj):
    tmp=path.with_name("."+path.name+f".{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    try:os.link(tmp,path)
    finally:tmp.unlink()
    gate(path.is_file(),"ATOMIC_FILE_MISSING")
def validate(a):
    gate(git(WT,"rev-parse","HEAD")==a.expected_head and
         not git(WT,"status","--porcelain"),"CLEAN_HEAD")
    spec=read(LOCK); m=spec["methods"]
    gate(spec["status"]=="A23_G7_REMAINING_FROZEN_BEFORE_PRODUCTION"
         and m==["S_mid_all8","2_forward","UnivFD"]
         and spec["GT_manifest_sha256"]==sha(GT)
         and spec["preceding_method"]["publication_commit"]==
            "da130b50856fd6c495fe73049651b746ab67b56b",
         "PREREG_SOURCE_LOCK")
    gate(a.method in m and a.dataset in ("ffhq","imagenet"),
         "METHOD_AND_DATASET")
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a35_g7_remaining_")
         and run.is_dir() and not run.is_symlink(), "RUN_ROOT")
    stage=run/a.method
    gate((stage/"results").is_dir() and (stage/"tasks").is_dir(),
         "STAGE_DIRECTORY")
    tasks=Path(a.tasks)
    gate(tasks.parent==stage/"tasks" and not tasks.is_symlink()
         and tasks.is_file(),"TASK_FILE")
    ids=read(tasks)
    gate(isinstance(ids,list) and 1<=len(ids)<=280 and
         len(set(ids))==len(ids) and
         all(type(i) is int and 0<=i<280 for i in ids),
         "TASK_GRID")
    frozen=read(GT)["images"]
    gate(len(frozen)==280 and all(frozen[i]["dataset"]==a.dataset for i in ids),
         "EXACT_FROZEN_DATASET")
    return spec,stage,ids,frozen

def gpu_admission(gpu):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu) and
         gpu in (0,1,2,3),"GPU_BINDING")
    lines=subprocess.check_output(["nvidia-smi",
      "--query-gpu=index,memory.total,memory.used,memory.free",
      "--format=csv,noheader,nounits"],text=True,timeout=20)
    values={}
    for line in lines.splitlines():
        z=[v.strip() for v in line.split(",")]
        gate(len(z)==4,"GPU_INVENTORY_SCHEMA")
        values[int(z[0])]=tuple(map(int,z[1:]))
    gate(gpu in values,"GPU_MISSING")
    total,used,free=values[gpu]
    # conservative B8 all8 native load + batch reserve ~28 GiB.
    need=28672
    gate(free>=need and used+need<=int(.8*total),"FOREIGN_GPU_VRAM_RESERVATION")
    return dict(physical_gpu=gpu,free_mib=free,used_mib=used,
                budget_mib=need,total_mib=total)

def main(a):
    spec,stage,ids,images=validate(a)
    if a.preflight_only:
        gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","PREFLIGHT_CPU_ONLY")
        from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
        gate(verify_clean_sparse(NATIVE),"NATIVE_SOURCE")
        if a.method=="UnivFD":
            ext=spec["univfd"]
            root=Path(ext["source_repo"])
            gate(git(root,"rev-parse","HEAD")==ext["source_head"]
                 and sha(root/ext["fc_path"])==ext["fc_sha256"]
                 and sha(ext["clip_path"])==ext["clip_sha256"],
                 "UNIVFD_SOURCE")
        else:
            x=spec["paired_prior"]["model_checkpoints"][a.dataset]
            gate(sha(x["path"])==x["sha256"],"NATIVE_CHECKPOINT")
        print("A35_GT_CPU_SOURCE_PREFLIGHT_PASS",a.method,a.dataset,
              "images",len(ids),flush=True)
        return
    import torch
    from prdiffusion.track_r.physics import ORBIT_PATTERNS,apply_orbit,to_rgb8
    from scripts.track_r.xp_a12_g7_c2_gpu import load_gt
    torch.set_num_threads(4)
    cuda=a.method!="UnivFD"
    if cuda:
        from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
        from scripts.track_r.xp_t_daps4k_worker_gpu import (
            load_model_native,model_config,validate_weight_trace)
        from scripts.track_r.xp_a25_xpc3_mid_gpu import fix_exact_native_timestep
        from scripts.track_r.xp_a35_g7_scoring import native_scores
        from prdiffusion.track_r.prior_score import frozen_alpha_schedule
        gate(verify_clean_sparse(NATIVE)
             and Path(sys.executable).resolve()==
                 (ROOT/"conda-envs/daps/bin/python").resolve(),
             "PINNED_DAPS_NATIVE")
        admission=gpu_admission(a.gpu)
        gate(torch.cuda.is_available() and torch.cuda.device_count()==1,
             "ONE_VISIBLE_GPU")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        device=torch.device("cuda:0")
    else:
        gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="",
             "UNIVFD_NO_GPU_CONTEXT")
        from prdiffusion.track_r.univfd import UnivFD
        ext=spec["univfd"]
        root=Path(ext["source_repo"])
        gate(git(root,"rev-parse","HEAD")==ext["source_head"]
             and sha(root/ext["fc_path"])==ext["fc_sha256"]
             and sha(ext["clip_path"])==ext["clip_sha256"],
             "PINNED_UNIVFD")
        device=torch.device("cpu")
        admission=dict(cpu_only=True)
    origin=Path.cwd()
    try:
        start=time.monotonic()
        if cuda:
            cfg=spec["paired_prior"]["model_checkpoints"][a.dataset]
            gate(sha(cfg["path"])==cfg["sha256"],"CHECKPOINT_BYTES")
            from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE
            sys.path.insert(0,str(NATIVE))
            os.chdir(str(NATIVE))
            model,model_load_s,trace=load_model_native(model_config(dict(
                checkpoint=cfg["path"],num_channels=cfg["num_channels"],
                num_res_blocks=cfg["num_res_blocks"],
                attention_resolutions=cfg["attention_resolutions"])))
            validate_weight_trace(trace)
            model.eval()
            alpha=frozen_alpha_schedule()
            tmap=fix_exact_native_timestep(model,alpha,torch)
        else:
            fd=UnivFD(root,root/ext["fc_path"],ext["clip_path"],device="cpu")
            model_load_s=time.monotonic()-start
            trace={"univfd":"strict_weight_sha256_verified"}
            tmap=None
        print("A35_WORKER_MODEL_READY",a.method,a.dataset,
              "pid",os.getpid(),"load_seconds",round(model_load_s,3),
              flush=True)
        for k,i in enumerate(ids):
            if (stage/"STOP_AFTER_IMAGE").exists():
                print("A35_WORKER_STOP_FLAG",a.method,"processed",k,flush=True)
                break
            out=stage/"results"/f"{i:03d}.json"
            gate(not out.exists() and not out.is_symlink(),
                 "NO_OVERWRITE_"+str(i))
            im=images[i]
            gate(sha(im["source_path"])==im["source_sha256"],
                 "GT_ORIGINAL_SOURCE_"+str(i))
            t=time.monotonic()
            x=load_gt(im).float().to(device)
            gate(x.dtype==torch.float32 and x.shape==(1,3,256,256)
                 and bool(torch.isfinite(x).all()),"GT_TENSOR")
            uid=f'XPCGT|{im["dataset"]}|{im["basename"]}'
            if cuda:
                score=native_scores(x,uid,model,a.method,alpha)
                decision=score["best_orbit"]
                if a.method=="2_forward":
                    gate(score["denoiser_examples"]==2,"TWO_FORWARD_EVALS")
                else:gate(score["denoiser_examples"]==128,"ALL8_EVALS")
                peak=math.ceil(torch.cuda.max_memory_reserved()/1048576)
                gate(peak<=16384,"MODEL_PEAK_EXCEEDS_ADMISSION_ESTIMATE")
                torch.cuda.synchronize()
            else:
                x8=to_rgb8(x)
                all8=torch.cat([apply_orbit(x8,p) for p in ORBIT_PATTERNS])
                with torch.inference_mode():
                    ps=fd.p_fake(all8).detach().cpu().tolist()
                gate(len(ps)==8 and all(math.isfinite(float(z))
                     and 0<=float(z)<=1 for z in ps),"UNIVFD_PROB")
                decision=ORBIT_PATTERNS[min(range(8),
                           key=lambda j:(float(ps[j]),j))]
                score=dict(method="UnivFD",uid=uid,
                           p_fake={p:float(v) for p,v in zip(ORBIT_PATTERNS,ps)},
                           best_orbit=decision,upright_kept=decision=="000",
                           classifier_images_forwarded=8,
                           canonical_orbit_batch=8,GT_only=True)
                peak=0
            elapsed=time.monotonic()-t
            gate(0<elapsed<900,"IMAGE_RUNTIME")
            payload=dict(status=STATUS,method=a.method,index=i,
                  dataset=im["dataset"],basename=im["basename"],
                  source_sha256=im["source_sha256"],
                  source_manifest_sha256=spec["GT_manifest_sha256"],
                  source_lock_sha256=sha(LOCK),
                  implementation_head=a.expected_head,
                  original_A34_publication_commit=spec["preceding_method"]["publication_commit"],
                  uid=uid,score=score,choice=decision,
                  upright_kept=decision=="000",n_evals=(
                    8 if not cuda else score["denoiser_examples"]),
                  physical_gpu=None if not cuda else a.gpu,
                  peak_reserved_mib=peak,
                  source_admission=admission,
                  native_model_load_s=model_load_s,
                  native_model_trace=trace,
                  native_time_mapping=tmap,
                  image_elapsed_s=elapsed,
                  diagnostic_GT_only=True,
                  GT_not_used_for_inverse_problem_generation=True,
                  no_new_selector=True,no_main_text_update=True)
            atomic(out,payload)
            print("A35_GT_IMAGE_COMPLETE",a.method,i,im["dataset"],
                  decision,"seconds",round(elapsed,3),flush=True)
        print("A35_WORKER_EXIT",a.method,a.dataset,"rows",len(ids),
              "elapsed_s",round(time.monotonic()-start,2),flush=True)
    finally:
        os.chdir(origin)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--method",choices=("S_mid_all8","2_forward","UnivFD"),required=True)
    p.add_argument("--dataset",choices=("ffhq","imagenet"),required=True)
    p.add_argument("--tasks",required=True)
    p.add_argument("--gpu",type=int,default=-1)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A35_GT_NATIVE_WORKER",repr(e),flush=True)
        sys.exit(8)
