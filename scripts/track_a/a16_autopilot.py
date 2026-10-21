#!/usr/bin/env python3
"""A16 four-physical-GPU, user-operated, write-once staged NP research controller.

Modes plan -> smoke -> execute. No GPU work in plan; execution requires explicit
issue41 lease acknowledgement, source/model/GT-frozen provenance and all four GPUs.
Four orthogonal scientific lanes; GT used ONLY after each saved reconstruction
for retrospective engineering promotion. This is NOT a fresh-cohort validation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ast
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path("/egr/research-pac/huang248")
WORK=ROOT/"pr_diffusion_track_a_a16_np_fast_autopilot"
CONF=WORK/"configs/track_a/a16_np_fast_autopilot_v1.json"
BASE=ROOT/"outputs/pr_diffusion/track_a_np_paper/a16_np_fast_autopilot_20261021"
NP_PY=ROOT/"conda-envs/prdiff_ffhq/bin/python"
FROZEN_INPUT=ROOT/"outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_inputs_v1/A1_2_INPUT_FREEZE_COMPLETE.json"
SELECTOR_RESULT=ROOT/"outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020/a15_8_fast_residual_selector_cpu_v1/A15_RESIDUAL_SELECTOR_COMPLETE.json"
LEASE="A16_NP_FAST_FOUR_GPU_RECORDED_ISSUE41"
MAX_EXEC_HOURS=8
KILL_AFTER=15
CANCEL=threading.Event()
ACTIVE=set()
ACTIVE_LOCK=threading.Lock()

def git(*args):
    return subprocess.check_output(["git","-C",str(WORK),*args],text=True).strip()

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def load(path):
    with open(path,encoding="utf8") as f:return json.load(f)

def dump_exclusive(path,data):
    path=Path(path)
    with path.open("x",encoding="utf8") as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def gate_source(args,cfg):
    if Path.cwd().resolve()!=WORK.resolve():
        raise RuntimeError("STOP worktree must equal reserved A16 checkout")
    if git("rev-parse","HEAD")!=args.expected_head:
        raise RuntimeError("STOP A16 execution source SHA changed")
    if git("branch","--show-current")!="codex/prdiff-track-a-a16-np-fast-autopilot-20261021":
        raise RuntimeError("STOP A16 branch changed")
    if git("status","--porcelain","--untracked-files=no"):
        raise RuntimeError("STOP dirty tracked source files")
    if not NP_PY.is_file():
        raise RuntimeError("STOP original Blackwell-capable NP environment absent")
    if digest(FROZEN_INPUT)!=cfg["frozen_input_manifest_sha256"]:
        raise RuntimeError("STOP original B22 exposed input freeze mismatch")
    if digest(SELECTOR_RESULT)!=cfg["prior_cpu_selector_sha256"]:
        raise RuntimeError("STOP A15 GT-free residual selector return changed")
    if digest(ROOT/"models/ffhq_10m.pt")!=cfg["model_sha256"]:
        raise RuntimeError("STOP shared FFHQ checkpoint changed")
    if sorted(cfg["images"])!=sorted(x["image_id"] for x in load(FROZEN_INPUT)["rows"]):
        raise RuntimeError("STOP A16 images no longer match frozen A1.2 exposed eight")
    if cfg.get("cohort")!="EIGHT_B22_EXPOSED_ENGINEERING_NOT_FRESH":
        raise RuntimeError("STOP cohort drift")
    if cfg.get("max_total_hours",0)>MAX_EXEC_HOURS:
        raise RuntimeError("STOP too long an unattended job")
    for gpu,arms in cfg["lane_arms"].items():
        if gpu not in cfg["gpu_uuids"] or not arms:
            raise RuntimeError("STOP unknown GPU lane")
        for item in arms:
            if not item["name"] or item["name"].count("/") or "\\" in item["name"]:
                raise RuntimeError("STOP unsafe arm path")
    if len({r["name"] for a in cfg["lane_arms"].values() for r in a})!=sum(map(len,cfg["lane_arms"].values())):
        raise RuntimeError("STOP duplicate arm names")
    return True

def gpu_probe(cfg,gpus,check_processes=True):
    """Check physical UUID and observed idleness. Never claim it is a future lease."""
    s=subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],text=True)
    info={}
    for line in s.splitlines():
        cols=[x.strip() for x in line.split(",")]
        if len(cols)!=5:continue
        idx,uuid,used,free,util=cols
        info[idx]=dict(uuid=uuid,used=int(used),free=int(free),util=int(util))
    for gid in gpus:
        q=info.get(gid,{})
        if q.get("uuid")!=cfg["gpu_uuids"][gid]:
            raise RuntimeError(f"STOP physical GPU{gid} UUID mismatch {q}")
        if q.get("free",0)<90000 or q.get("used",100000)>1024 or q.get("util",100)>10:
            raise RuntimeError(f"STOP GPU{gid} busy or insufficient free VRAM {q}")
    if check_processes:
        p=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                          "--format=csv,noheader"],capture_output=True,text=True,check=True)
        busy=[line for line in p.stdout.splitlines() if line.strip() and line.split(",")[0].strip() in
              [cfg["gpu_uuids"][x] for x in gpus]]
        if busy:raise RuntimeError("STOP existing CUDA compute processes: "+"; ".join(busy))
    return {i:info[i] for i in gpus}

def make_plan(args,cfg):
    if BASE.exists() or BASE.is_symlink():
        raise RuntimeError("STOP A16 output area already exists; do not overwrite")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("",None):
        raise RuntimeError("STOP CPU plan must hide CUDA devices")
    for sourcefile in ("scripts/track_a/a16_np_hybrid_core.py","scripts/track_a/a16_np_worker.py",
                       "scripts/track_a/a16_autopilot.py","tests/test_a16_np_fast_autopilot.py"):
        ast.parse((WORK/sourcefile).read_text(encoding="utf8"),filename=sourcefile)
    frozen=load(FROZEN_INPUT)
    for row in frozen["rows"]:
        if digest(Path(row["measurement_path"]))!=row["measurement_file_sha256"]:
            raise RuntimeError("STOP frozen observation file altered "+row["image_id"])
        if digest(Path(row["ground_truth_tensor_path"]))!=row["ground_truth_tensor_file_sha256"]:
            raise RuntimeError("STOP frozen GT file altered "+row["image_id"])
    # Declare experiments, not flexible parameter tuning.
    BASE.mkdir(parents=True,exist_ok=False)
    (BASE/"arm_specs").mkdir(exist_ok=False)
    (BASE/"logs").mkdir(exist_ok=False)
    for lane,arms in cfg["lane_arms"].items():
        for entry in arms:
            dump_exclusive(BASE/"arm_specs"/f"{entry['name']}.json",entry)
    receipt=dict(status="A16_CPU_PLAN_PASS",source_head=args.expected_head,
        input_sha256=cfg["frozen_input_manifest_sha256"],
        baseline_selector_receipt_sha256=cfg["prior_cpu_selector_sha256"],
        model_sha256=cfg["model_sha256"],
        image_count=len(cfg["images"]),arms=sum(len(v) for v in cfg["lane_arms"].values()),
        stage0_images=cfg["stage0_images"],physical_gpus=cfg["gpu_uuids"],
        warnings=["Exposed engineering cohort; no prospective inference",
                  "No GPU operations performed in CPU plan",
                  "Original native NP replay smoke is mandatory before execute",
                  "Track B GPU priority; simultaneous cluster users not excluded by nvidia-smi",
                  "GT used only after each candidate is reconstructed and committed",
                  "No hard budget matching implied by different correction operation counts"])
    dump_exclusive(BASE/"A16_CPU_PLAN_PASS.json",receipt)
    print(json.dumps(dict(status=receipt["status"],source_head=args.expected_head,
        arm_count=receipt["arms"],images=receipt["image_count"],
        receipt_sha256=digest(BASE/"A16_CPU_PLAN_PASS.json")),sort_keys=True),flush=True)

def validate_plan(args):
    rec=load(BASE/"A16_CPU_PLAN_PASS.json")
    if rec.get("status")!="A16_CPU_PLAN_PASS" or rec.get("source_head")!=args.expected_head:
        raise RuntimeError("STOP missing or incompatible CPU plan")

def result_path(arm,img,rep):
    return BASE/"runs"/arm/img/f"rep{rep}"

def check_completed(arm,img,rep,head):
    out=result_path(arm,img,rep)
    receipt=out/"A16_CANDIDATE_COMPLETE.json"
    if not out.exists():
        return False
    if not receipt.is_file():
        raise RuntimeError(f"STOP incomplete occupied output {out}")
    d=load(receipt)
    if d.get("status")!="A16_NP_CANDIDATE_PASS" or d.get("source_head")!=head or \
       d.get("image_id")!=img or d.get("rep")!=rep or d.get("arm",{}).get("name")!=arm:
        raise RuntimeError(f"STOP stale or mismatched candidate receipt {receipt}")
    for fname,field in [("reconstruction.pt","reconstruction_sha256"),
                        ("reconstruction.png","reconstruction_png_sha256")]:
        if digest(out/fname)!=d[field]:
            raise RuntimeError(f"STOP candidate tensor/PNG bytes changed {out}")
    return True

def task_command(head,arm,img,rep,gpu,replay):
    cmd=[str(NP_PY),"-m","scripts.track_a.a16_np_worker",
         "--expected-head",head,
         "--arm-json",str(BASE/"arm_specs"/f"{arm}.json"),
         "--image-id",img,"--rep",str(rep),
         "--out",str(result_path(arm,img,rep)),
         "--gpu-uuid",load(CONF)["gpu_uuids"][gpu]]
    if replay:cmd.append("--reference-replay")
    return cmd

def run_task(args,arm,img,rep,gpu,replay,deadline,limit):
    if CANCEL.is_set():
        raise RuntimeError("STOP another A16 GPU lane failed; no new worker started")
    if check_completed(arm,img,rep,args.expected_head):
        return load(result_path(arm,img,rep)/"A16_CANDIDATE_COMPLETE.json")
    if time.monotonic()>deadline:
        raise RuntimeError("STOP eight-hour total execution guard reached")
    log=BASE/"logs"/f"{arm}__{img}__rep{rep}.log"
    if log.exists():
        raise RuntimeError("STOP job log already exists but candidate incomplete")
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu,
               PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",WANDB_MODE="disabled",
               OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTHONPATH=str(WORK)+os.pathsep+env.get("PYTHONPATH",""))
    cmd=task_command(args.expected_head,arm,img,rep,gpu,replay)
    with log.open("x",encoding="utf8") as f:
        print(f"A16_START gpu{gpu} arm={arm} image={img} rep={rep}",flush=True)
        f.write(json.dumps(dict(command=cmd,gpu=gpu,arm=arm,image=img,rep=rep))+"\n")
        f.flush()
        before=time.monotonic()
        child=subprocess.Popen(cmd,cwd=str(WORK),env=env,
                               stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        with ACTIVE_LOCK: ACTIVE.add(child.pid)
        try:
            child.wait(timeout=min(limit,max(1,int(deadline-time.monotonic()))))
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
            raise RuntimeError(f"STOP timed out {arm}/{img}/rep{rep}; output preserved")
        finally:
            with ACTIVE_LOCK: ACTIVE.discard(child.pid)
        elapsed=time.monotonic()-before
        if child.returncode!=0:
            raise RuntimeError(f"STOP worker rc{child.returncode} {arm}/{img}/rep{rep}; see {log}")
    if not check_completed(arm,img,rep,args.expected_head):
        raise RuntimeError(f"STOP worker exited RC0 but no valid receipt {arm}/{img}/rep{rep}")
    wall_meta=result_path(arm,img,rep)/"A16_SUBPROCESS_WALL.json"
    dump_exclusive(wall_meta,dict(subprocess_wall_s=elapsed,
         arm=arm,image=img,rep=rep,physical_gpu=gpu))
    result=load(result_path(arm,img,rep)/"A16_CANDIDATE_COMPLETE.json")
    print(f"A16_PASS gpu{gpu} arm={arm} image={img} rep={rep} psnr={result['raw_rgb8_psnr_db']:.3f} wall={elapsed:.1f}s",flush=True)
    return result


def lane_execute(args,stage,gpu,arms,images,reps,deadline,cfg):
    try:
        for arm in arms:
            for img in images:
                for rep in reps:
                    if CANCEL.is_set():
                        raise RuntimeError("STOP peer GPU lane failure; do not launch further tasks")
                    run_task(args,arm,img,rep,gpu,False,deadline,int(cfg["max_root_seconds"]))
        return gpu
    except BaseException:
        CANCEL.set()
        raise

def score_records(arm,images,reps,head):
    rows=[load(result_path(arm,img,rep)/"A16_CANDIDATE_COMPLETE.json") for img in images for rep in reps]
    raw=[float(x["raw_rgb8_psnr_db"]) for x in rows]
    bad20=sum(x<20 for x in raw)
    meansum=sum(x["counters"]["unet"] for x in rows)/len(rows)
    runtime=sum(x["sampling_wall_s"] for x in rows)/len(rows)
    # Quality/failure first, measured UNet and runtime penalty second.
    utility=sum(raw)/len(raw) - 8*bad20/len(raw) - .35*math.log(max(1,meansum/150.)) - .08*math.log(max(1.,runtime))
    out=dict(arm=arm,mean_psnr_db=sum(raw)/len(raw),
                minimum_psnr_db=min(raw),bad20=bad20,good25=sum(x>=25 for x in raw),
                n=len(rows),mean_unet_calls=meansum,mean_sampling_s=runtime,
                mean_real_process_wall_s=sum(
                    load(result_path(arm,img,rep)/"A16_SUBPROCESS_WALL.json")["subprocess_wall_s"]
                    for img in images for rep in reps)/len(rows),
                promotion_utility_exposed_only=utility)
    if reps==[0,1,2,3]:
        # Fully charge all 4 generated trajectories before GT-free selection.
        per_image=[]
        for img in images:
            subset=[r for r in rows if r["image_id"]==img]
            oracle=max(subset,key=lambda r:(r["raw_rgb8_psnr_db"],-r["rep"]))
            entry=dict(image_id=img,oracle_raw_db=oracle["raw_rgb8_psnr_db"])
            for mode,key in (("signed","signed_full_measurement_residual_l2"),
                             ("clipped","clipped_full_measurement_residual_l2")):
                chosen=min(subset,key=lambda r:(r[key],r["rep"]))
                entry[mode+"_psnr_db"]=chosen["raw_rgb8_psnr_db"]
                entry[mode+"_regret_db"]=max(0,oracle["raw_rgb8_psnr_db"]-chosen["raw_rgb8_psnr_db"])
                entry[mode+"_oracle_rep_match"]=int(chosen["rep"]==oracle["rep"])
                entry[mode+"_chosen_good25"]=int(chosen["raw_rgb8_psnr_db"]>=25)
            per_image.append(entry)
        out["K4_gt_free_selector_replay"]={
            mode:dict(mean_psnr_db=sum(r[mode+"_psnr_db"] for r in per_image)/len(per_image),
                      mean_oracle_regret_db=sum(r[mode+"_regret_db"] for r in per_image)/len(per_image),
                      matches_oracle=sum(r[mode+"_oracle_rep_match"] for r in per_image),
                      good25=sum(r[mode+"_chosen_good25"] for r in per_image))
            for mode in ("signed","clipped")}
        out["K4_per_image_selector"] = per_image
    return out


def summarize_stage(args,name,lanes,images,reps):
    receipt_path=BASE/f"A16_{name.upper()}_COMPLETE.json"
    if receipt_path.exists():
        r=load(receipt_path)
        if r.get("status")!=f"A16_{name.upper()}_PASS" or r.get("source_head")!=args.expected_head:
            raise RuntimeError("STOP occupied inconsistent stage summary")
        return r
    out={}
    for gpu,arms in lanes.items():
        v=[score_records(arm,images,reps,args.expected_head) for arm in arms]
        out[gpu]=sorted(v,key=lambda q:(-q["promotion_utility_exposed_only"],q["arm"]))
    receipt=dict(status=f"A16_{name.upper()}_PASS",source_head=args.expected_head,
        images=images,reps=reps,lanes=out,
        metrics_note="Retrospective GT-engineering promotion on already-exposed images, NOT prospective statistical support")
    dump_exclusive(receipt_path,receipt)
    print(json.dumps(dict(status=receipt["status"],
        stage=name,summary_path=str(receipt_path)),sort_keys=True),flush=True)
    return receipt


def promotion(analysis,cfg,stage):
    next_arms={}
    for gpu,rows in analysis["lanes"].items():
        candidates=[v["arm"] for v in rows]
        if stage=="stage1":
            selected=candidates[:2]
            if gpu=="0":
                selected=list(dict.fromkeys(["lf_native",candidates[0]]))
                if len(selected)<2:selected.append(candidates[1])
            elif gpu=="1":
                selected=list(dict.fromkeys(["s2_native",candidates[0]]))
                if len(selected)<2:selected.append(candidates[1])
        else:
            selected=candidates[:1]
            if gpu=="0" and "lf_native" not in selected:selected.append("lf_native")
            if gpu=="1" and "s2_native" not in selected:selected.append("s2_native")
        next_arms[gpu]=selected
    return next_arms


def run_stage(args,name,lanes,images,reps,deadline,cfg):
    CANCEL.clear()
    # Four independent physical devices: one serial lane per GPU.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(lane_execute,args,name,gpu,arms,images,reps,deadline,cfg)
                 for gpu,arms in sorted(lanes.items())]
        # Caller thread only reports all PASS if all workers returned successfully.
        failures=[]
        for f in futures:
            try:f.result()
            except BaseException as exc:failures.append(str(exc))
        if failures:
            raise RuntimeError("STOP lane failures; preserve outputs: "+" | ".join(failures))
    return summarize_stage(args,name,lanes,images,reps)


def execute(args,cfg):
    validate_plan(args)
    smoke=check_completed("lf_native","69073",0,args.expected_head)
    if not smoke:
        raise RuntimeError("STOP mandatory native-replay smoke not completed")
    smoke_json=load(result_path("lf_native","69073",0)/"A16_CANDIDATE_COMPLETE.json")
    if smoke_json.get("np_native_replay_maxabs") is None or smoke_json["np_native_replay_maxabs"]>5e-4:
        raise RuntimeError("STOP no valid native-exact NP replay from smoke")
    print("A16_CHECK_ALL_FOUR_GPUS",gpu_probe(cfg,list("0123")),flush=True)
    deadline=time.monotonic()+min(float(args.max_hours),float(cfg["max_total_hours"]))*3600
    lanes0={g:[r["name"] for r in arms] for g,arms in cfg["lane_arms"].items()}
    s0=run_stage(args,"stage0",lanes0,cfg["stage0_images"],[0],deadline,cfg)
    lanes1=promotion(s0,cfg,"stage1")
    s1=run_stage(args,"stage1",lanes1,cfg["images"],[0,1],deadline,cfg)
    lanes2=promotion(s1,cfg,"stage2")
    s2=run_stage(args,"stage2",lanes2,cfg["images"],[0,1,2,3],deadline,cfg)
    final=dict(status="A16_AUTOPILOT_COMPLETE",source_head=args.expected_head,
         mode="A16_EXPOSED_ENGINEERING_ONLY",stages=[s0["status"],s1["status"],s2["status"]],
         selected_by_stage={"stage1":lanes1,"stage2":lanes2},
         samples_without_new_images=True,all_four_gpus_utilized=True,
         note="No protected B24 data. ALL results are exploratory GT-informed arm promotion, not independent confirmatory evidence.")
    dump_exclusive(BASE/"A16_AUTOPILOT_FINAL.json",final)
    print(json.dumps(final,sort_keys=True),flush=True)

def on_interrupt(sig,frame):
    """Kill ONLY our registered child process groups; never systemwide CUDA jobs."""
    CANCEL.set()
    with ACTIVE_LOCK: pids=list(ACTIVE)
    for pid in pids:
        try:os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    raise KeyboardInterrupt("A16 user interrupt; own children terminated")


def main():
    signal.signal(signal.SIGINT,on_interrupt)
    signal.signal(signal.SIGTERM,on_interrupt)
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["plan","smoke","execute"],required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--lease-ack",default="")
    ap.add_argument("--max-hours",type=float,default=8.0)
    args=ap.parse_args()
    cfg=load(CONF)
    gate_source(args,cfg)
    if args.mode=="plan":
        return make_plan(args,cfg)
    if args.lease_ack!=LEASE:
        raise RuntimeError("STOP no explicit 4GPU issue41 lease acknowledgement")
    if args.max_hours<=0 or args.max_hours>MAX_EXEC_HOURS:
        raise RuntimeError("STOP maximum aggregate wall time outside [0,8] hours")
    if args.mode=="smoke":
        validate_plan(args)
        gpu_probe(cfg,["0"])
        if check_completed("lf_native","69073",0,args.expected_head):
            raise RuntimeError("STOP smoke result already exists")
        run_task(args,"lf_native","69073",0,"0",True,time.monotonic()+600,600)
        rec=load(result_path("lf_native","69073",0)/"A16_CANDIDATE_COMPLETE.json")
        if rec.get("np_native_replay_maxabs") is None or rec["np_native_replay_maxabs"]>5e-4:
            raise RuntimeError("STOP native reproduction smoke threshold")
        print("A16_NATIVE_REPLAY_SMOKE_PASS",rec["np_native_replay_maxabs"],flush=True)
        return
    if args.mode=="execute":
        return execute(args,cfg)
    raise RuntimeError("unrecognized mode")

if __name__=="__main__":
    try:main()
    except (Exception,KeyboardInterrupt) as e:
        print("A16_AUTOPILOT_STOP:",str(e),file=sys.stderr,flush=True)
        sys.exit(15)
