#!/usr/bin/env python3
"""A21 GT-free XP-C3 worker. R2 numerical method; grouping only at fixed t.

B8: four j and two orientations for one candidate, B32: four candidates.
R2 fallback calls the frozen orient_sm unchanged. Atomic per-image claims,
exclusive-create output, no foreign process actions, no GPU flags modified.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
SRC="docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
FRZ="docs/track_r/executors/xp/XP_C3_A21_FREEZE.json"
A20="docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"
R2="docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R2.json"
R2_HEAD="c80308aaa026114dff1ee5accc239acbb75f9777"
PANELS=("XP_N","G5","XP_D4")
SIZES={"XP_N":6400,"G5":800,"XP_D4":6400}
T=(300,400,500,600)
STOP_AFTER_CHUNK=False

def stop_after(*_):
    global STOP_AFTER_CHUNK
    STOP_AFTER_CHUNK=True
signal.signal(signal.SIGUSR1,stop_after)
signal.signal(signal.SIGTERM,stop_after)
signal.signal(signal.SIGINT,stop_after)

def gate(v,msg):
    if not v:raise RuntimeError("STOP_A21_GPU_"+msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def git(*x):return subprocess.check_output(
    ["git","-C",str(WT),*x],text=True,timeout=60).strip()

def preflight(a):
    from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
    from scripts.track_r.xp_t_daps4k_prepare_cpu import IMAGENET_MODEL_SHA
    from prdiffusion.track_r.prior_score import frozen_alpha_schedule,prior_seed
    import torch
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==str(a.gpu)
         and a.gpu in (0,1,2,3) and torch.cuda.is_available()
         and torch.cuda.device_count()==1
         and os.environ.get("PYTHONDONTWRITEBYTECODE")=="1",
         "VISIBLE_PHYSICAL_GPU")
    gate(git("rev-parse","HEAD")==a.expected_head
         and not git("status","--porcelain"),"CLEAN_PINNED_CODE")
    f=read(WT/FRZ);s=WT/SRC
    gate(f["status"]=="A21_XPC3_FROZEN_BEFORE_PILOT"
         and f["source_sha256"]==sha(s)
         and f["worker_sha256"]==sha(Path(__file__))
         and f["controller_sha256"]==sha(WT/"scripts/track_r/xp_a29_a21_controller.py")
         and f["settings_sha256"]==sha(WT/"scripts/track_r/xp_a29_a21_settings.py")
         and f["pilot_max_L_diff"]==1e-5
         and f["pilot_min_abs_margin_for_choice"]==1e-5
         and f["batch_options"]==[8,32] and f["max_per_gpu"]==8
         and f["gpu_occupancy_cap_h"]==56.
         and f["model_sha256"]==IMAGENET_MODEL_SHA
         and f["prior_sha256"]==sha(WT/"prdiffusion/track_r/prior_score.py")
         and f["native_checkpoint_path"]==str(ROOT/"models/imagenet256.pt"),
         "A21_FREEZE_MISMATCH")
    old=read(WT/A20);r2=read(WT/R2)
    gate(old["source_manifest_sha256"]==sha(s)
         and r2["source_manifest_sha256"]==sha(s)
         and sha(ROOT/"models/imagenet256.pt")==old["checkpoint_sha256"]
         and verify_clean_sparse(NATIVE)
         and all(x in f["allowed_gpus"] for x in (0,1,2,3))
         and git("ls-files","--error-unmatch",FRZ)==FRZ,
         "A20_AND_A21_SCIENCE_PROVENANCE")
    rr=read(s)
    gate(len(rr["rows"])==13600,"SOURCE_13600")
    groups=defaultdict(list)
    for x in rr["rows"]:
        if x["panel"]==a.panel:groups[x["image_id"]].append(x)
    gate(len(groups)==100
         and all(sorted(z["run"] for z in rows)==
            list(range(8 if a.panel=="G5" else 64))
            for rows in groups.values()),"PANEL_GRID")
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a25_xpc3_prod_a21_")
         and run.is_dir() and not run.is_symlink(),"RUN")
    return groups,NATIVE,torch,frozen_alpha_schedule,prior_seed,sha(s)

def load_native(native,torch,alpha):
    from scripts.track_r.xp_t_daps4k_worker_gpu import (
        load_model_native,model_config,validate_weight_trace)
    from scripts.track_r.xp_a25_xpc3_mid_gpu import fix_exact_native_timestep
    torch.cuda.set_device(0)
    sys.path.insert(0,str(native))
    prev=Path.cwd()
    os.chdir(str(native))
    try:
        cfg=dict(checkpoint=str(ROOT/"models/imagenet256.pt"),
                 num_channels=256,num_res_blocks=2,
                 attention_resolutions="32,16,8")
        model,load_s,trace=load_model_native(model_config(cfg))
        validate_weight_trace(trace)
        model.eval()
        mapping=fix_exact_native_timestep(model,alpha,torch)
        return model,load_s,mapping
    finally:
        os.chdir(prev)

def score_group(rows,model,torch,alpha,seed_fn,batch):
    """Return one R2-schema score for every candidate, ordered t,j,orientation."""
    from prdiffusion.track_r.physics import apply_orbit
    from scripts.track_r.xp_a25_xpc3_mid_gpu import orient_sm
    gate(batch in (2,8,32),"BATCH")
    if batch==2:
        return [orient_sm(x,rec["uid"],model,torch,alpha,seed_fn)
                for rec,x in rows]
    gate(len(rows)*8<=batch,"GROUP_SIZE")
    acc=[[0.,0.] for _ in rows]
    with torch.inference_mode():
        for t in T:
            at=float(alpha[t]);sigma=math.sqrt((1.-at)/at)
            zs=[];targets=[]
            for rec,x in rows:
                gate(tuple(x.shape)==(1,3,256,256)
                     and x.dtype==torch.float32
                     and bool(torch.isfinite(x).all()),"FINITE_X")
                flip=apply_orbit(x,"111")
                for j in range(4):
                    gen=torch.Generator(device="cpu").manual_seed(
                        seed_fn(rec["uid"],t,j))
                    eps=torch.randn((1,3,256,256),generator=gen,
                                    dtype=torch.float32).to(x.device)
                    ef=apply_orbit(eps,"111")
                    zs.extend((x+sigma*eps,flip+sigma*ef))
                    targets.extend((eps,ef))
            z=torch.cat(zs)
            target=torch.cat(targets)
            gate(len(z)==8*len(rows) and len(z)<=batch,
                 "FIXED_T_BATCH")
            estimated=model.tweedie(z,sigma)
            gate(tuple(estimated.shape)==tuple(z.shape)
                 and bool(torch.isfinite(estimated).all()),"TWEEDIE")
            each=(((z-estimated)/sigma-target).square()
                  .flatten(1).mean(1).float().cpu().tolist())
            gate(len(each)==8*len(rows),"LOSS_GRID")
            for i in range(len(rows)):
                for j in range(4):
                    acc[i][0]+=float(each[i*8+j*2])
                    acc[i][1]+=float(each[i*8+j*2+1])
    result=[]
    for losses in acc:
        l0=losses[0]/16.;l1=losses[1]/16.;margin=l0-l1
        gate(all(math.isfinite(x) for x in (l0,l1,margin)),
             "FINITE_S_MID")
        result.append(dict(prior_choice="111" if margin>1e-7 else "000",
             prior_margin=format(margin,".15g"),
             L000=format(l0,".15g"),L111=format(l1,".15g"),
             denoiser_examples=32))
    return result

def evaluate_image(records,model,torch,alpha,seed_fn,batch):
    result=[]
    width=4 if batch==32 else 1
    for k in range(0,len(records),width):
        block=records[k:k+width]
        tensors=[]
        for r in block:
            p=Path(r["pt_path"])
            gate(sha(p)==r["pt_sha256"],"SOURCE_TENSOR_SHA")
            saved=torch.load(p,map_location="cpu",weights_only=True)
            gate(isinstance(saved,dict)
                 and isinstance(saved.get("reconstruction"),torch.Tensor),
                 "SOURCE_TENSOR_SCHEMA")
            x=saved["reconstruction"].float().to("cuda:0")
            tensors.append((r,x))
        outputs=score_group(tensors,model,torch,alpha,seed_fn,batch)
        for (r,_),v in zip(tensors,outputs):
            result.append(dict(image_id=r["image_id"],run=r["run"],
                               label=r["label"],**v))
    gate(len(result)==len(records),"IMAGE_SIZE")
    return result

def proc_tick(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None

def claim(folder,i):
    import fcntl
    lock=folder/".dispatch.lock"
    claimfile=folder/"claims"/f"{i:03d}.claim"
    target=folder/f"{i:03d}.json"
    with lock.open("a+") as lockfd:
        fcntl.flock(lockfd,fcntl.LOCK_EX)
        if target.exists():return None
        if claimfile.exists():
            try:
                owner=read(claimfile)
                if proc_tick(int(owner["pid"]))==owner["start_tick"]:
                    return None
            except (ValueError,KeyError,json.JSONDecodeError):
                # If malformed claim belongs to a live PID, STOP, don't steal.
                gate(False,"MALFORMED_CLAIM")
            claimfile.unlink()
            print("A21_CHUNK_RECLAIM",i,flush=True)
        with claimfile.open("x") as f:
            json.dump(dict(pid=os.getpid(),
                 start_tick=proc_tick(os.getpid()),index=i),f)
            f.write("\n")
        return claimfile

def release(folder,claimfile):
    lock=folder/".dispatch.lock"
    with lock.open("a+") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX)
        gate(claimfile.exists()
             and read(claimfile)["pid"]==os.getpid(),
             "CLAIM_OWNER")
        claimfile.unlink()

def output_once(path,obj):
    tmp=path.with_name("."+path.name+f".{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(obj,f,sort_keys=True,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    try:
        os.link(tmp,path)  # atomic link; never overwrite an accepted image
    finally:
        tmp.unlink()
    gate(path.is_file(),"EXCLUSIVE_OUTPUT")

def main(a):
    gate(a.panel in PANELS and a.mode in ("B8","B32","R2"),"ARGS")
    groups,native,torch,schedule,seedfn,source_hash=preflight(a)
    alpha=schedule()
    model,load_s,mapping=load_native(native,torch,alpha)
    print("A21_NATIVE_MODEL_READY",a.panel,a.mode,"gpu",a.gpu,
          "load_s",round(load_s,3),"time_map",json.dumps(mapping),flush=True)
    torch.cuda.reset_peak_memory_stats()
    batch={"B8":8,"B32":32,"R2":2}[a.mode]
    t0=time.monotonic();ids=sorted(groups)
    if a.replicate_task:
        gate(not a.pilot and a.replicate_output,"REPLICATE_ARGS")
        task=read(a.replicate_task)
        gate(task["panel"]==a.panel and
             task["destination_physical_gpu"]==a.gpu and
             len(task["uids"])>=1,"REPLICATE_TASK_SCOPE")
        allrows={r["uid"]:r for rs in groups.values() for r in rs}
        gate(len(set(task["uids"]))==len(task["uids"]) and
             all(uid in allrows for uid in task["uids"]),
             "REPLICATE_UID_FROZEN")
        chosen=[allrows[uid] for uid in task["uids"]]
        for row in chosen:
            gate(task["source_physical_gpu_by_uid"][row["uid"]]!=a.gpu,
                 "REPLICATE_MUST_BE_DIFFERENT_GPU")
        vals=evaluate_image(chosen,model,torch,alpha,seedfn,batch)
        torch.cuda.synchronize()
        out=Path(a.replicate_output)
        gate(out.parent==Path(a.run)/"replicate" and
             not out.exists(),"REPLICATE_UNIQUE_OUTPUT")
        output_once(out,dict(status="A21_DIFFERENT_GPU_REPLICATION_PASS",
             physical_gpu=a.gpu,panel=a.panel,rows=vals,
             source_sha256=source_hash,source_task_sha256=sha(a.replicate_task),
             peak_reserved_mib=math.ceil(
                 torch.cuda.max_memory_reserved()/1048576),
             wall_seconds=time.monotonic()-t0,no_GT_scoring=True))
        print("A21_REPLICATION_PASS",a.panel,"rows",len(vals),
              "gpu",a.gpu,flush=True)
        return
    if a.pilot:
        gate(a.panel=="XP_N" and a.mode in ("B8","B32"),"PILOT_SCOPE")
        records=sorted(groups[ids[0]],key=lambda r:r["run"])
        vals=evaluate_image(records,model,torch,alpha,seedfn,batch)
        torch.cuda.synchronize()
        peak=int(math.ceil(torch.cuda.max_memory_reserved()/1048576))
        payload=dict(status="A21_PILOT_BATCH_COMPLETE",mode=a.mode,
            panel=a.panel,physical_gpu=a.gpu,source_sha256=source_hash,
            candidate_count=64,rows=vals,
            peak_reserved_mib=peak,
            seconds=time.monotonic()-t0,load_seconds=load_s,
            no_GT_scoring=True)
        output_once(Path(a.run)/"pilot"/f"{a.mode}.json",payload)
        print("A21_PILOT_WORKER_PASS",a.mode,
              "elapsed",round(payload["seconds"],3),
              "peak_mib",peak,flush=True)
        return
    folder=Path(a.run)/a.panel
    (folder/"claims").mkdir(exist_ok=True)
    completed=0
    for i,iid in enumerate(ids):
        if STOP_AFTER_CHUNK:break
        ticket=claim(folder,i)
        if ticket is None:continue
        try:
            records=sorted(groups[iid],key=lambda r:r["run"])
            vals=evaluate_image(records,model,torch,alpha,seedfn,batch)
            torch.cuda.synchronize()
            payload=dict(status="A20_XPC3_IMAGE_PRIOR_PASS",
                panel=a.panel,image_id=iid,rows=vals,
                source_manifest_sha256=source_hash,
                checkpoint_sha256=read(WT/A20)["checkpoint_sha256"],
                no_GT_access=True,a21_batch_mode=a.mode,
                worker_pid=os.getpid(),physical_gpu=a.gpu)
            output_once(folder/f"{i:03d}.json",payload)
            completed+=1
            if completed%2==0:
                print("A21_IMAGE_ACCEPTED",a.panel,iid,
                      "worker_images",completed,
                      "peak_reserved_mib",math.ceil(
                      torch.cuda.max_memory_reserved()/1048576),flush=True)
        finally:
            release(folder,ticket)
    print("A21_WORKER_EXIT",a.panel,a.mode,"gpu",a.gpu,
          "completed",completed,
          "peak_reserved_mib",math.ceil(
              torch.cuda.max_memory_reserved()/1048576),
          "wall_s",round(time.monotonic()-t0,2),
          "yielded",STOP_AFTER_CHUNK,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for k in ("expected-head","run","panel","mode"):
        ap.add_argument("--"+k,required=True)
    ap.add_argument("--gpu",type=int,required=True)
    ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--replicate-task")
    ap.add_argument("--replicate-output")
    try:main(ap.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A21_GPU_WORKER",e,flush=True)
        sys.exit(8)
