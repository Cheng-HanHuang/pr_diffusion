#!/usr/bin/env python3
"""A10 four-GPU historical 400x5 DAPS replay wave controller.

Wave 1: exactly indices 0..3. Wave 2: exactly 4..7, requiring wave-1
offline replay PASS. Worker paths immutable, no launch on foreign processes,
and short polling with per-sample GPU PID inventory.
"""
from __future__ import annotations
import argparse,dataclasses,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu
from scripts.track_r.xp_f8_legacy_replay_gpu import (
    ROOT,BASE,B16,PREP,A10_DEADLINE,sha,read_json,fail)
from scripts.track_r.xp_f8_legacy_replay_gpu import FROZEN_MANIFEST_SHA

WT=Path(__file__).resolve().parents[2]
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
REQUIRED_MIB=24576
MARGIN_MIB=2048
def write_x(path,obj):
    with path.open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def ensure_wave(a):
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_CONTROLLER")
    fail(subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
         text=True).strip()==a.expected_head,"RUNNER_HEAD")
    fail(not subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
         text=True).strip(),"DIRTY")
    mf=read_json(PREP/"XP_F8_REPLAY_INPUTS_GT_FREE.json")
    fail(sha(PREP/"XP_F8_REPLAY_INPUTS_GT_FREE.json")==FROZEN_MANIFEST_SHA
         and len(mf["rows"])==8,"MANIFEST")
    b16=read_json(B16/"B16_PRODUCTION_STATE.json")
    gpu_h=(b16["gate_spent_gpu_seconds"]+
           b16["spent_gpu_seconds"])/3600
    fail(b16["status"]=="A10_B16_GENERATION_COMPLETE_WAIT_F8"
         and b16["active"]==[] and gpu_h<70,"B16_ACCOUNTING")
    fail(datetime.datetime.now(ZoneInfo("America/Detroit"))<A10_DEADLINE,
         "A10_WALL_DEADLINE")
    root=Path(a.run)
    fail(root.parent==BASE and root.name.startswith("xp_f8_legacy_replay_")
         and not root.is_symlink(),"RUN_NAMESPACE")
    if a.wave==1:
        fail(not root.exists(),"WAVE1_ALREADY_EXISTS")
        root.mkdir(parents=False,exist_ok=False)
    else:
        fail(root.is_dir() and
             (root/"XP_F8_LEGACY_REPLAY_WAVE1.json").is_file(),
             "WAVE1_NOT_RECEIPTED")
        w1=read_json(root/"XP_F8_LEGACY_REPLAY_WAVE1.json")
        fail(w1["status"]=="XP_F8_LEGACY_WAVE1_GPU_PASS"
             and w1["accepted_indices"]==[0,1,2,3],
             "WAVE1_GPU_FAIL")
        # Native outputs alone do not certify historical lineage:
        # wave 1 must have passed the OFFLINE comparison as well.
        offline=read_json(root/"XP_F8_LEGACY_REPLAY_WAVE1_OFFLINE.json")
        fail(offline["status"]=="XP_F8_LEGACY_WAVE1_OFFLINE_PASS",
             "WAVE1_OFFLINE_NOT_PASSED")
        fail(not (root/"XP_F8_LEGACY_REPLAY_WAVE2.json").exists(),
             "WAVE2_ALREADY_DONE")
    existing=root.glob("replay_*/XP_F8_LEGACY_REPLAY_NATIVE_ROOT.json")
    replay_gpu_h=sum(read_json(p)["native_wall_seconds"] for p in existing)/3600
    fail(gpu_h+replay_gpu_h<70,"A10_SHARED_REPLAY_BUDGET")
    return root,gpu_h,replay_gpu_h
def frozen_preflight(a,indices):
    # One full native Hydra import per wave, using a representative B22
    # or DEV80 row. All rows use identical pinned source and CLI template,
    # varying only frozen seed, measurement and output. Each child performs
    # its OWN resolved-Hydra check before starting native GPU inference,
    # so no individual row bypasses the scientific configuration gate.
    i=indices[0]
    cmd=[str(PY),"-u","-m","scripts.track_r.xp_f8_legacy_replay_gpu",
         "--expected-head",a.expected_head,"--index",str(i),
         "--preflight-only"]
    e=os.environ.copy();e["PYTHONPATH"]=str(WT);e["CUDA_VISIBLE_DEVICES"]=""
    try:
        p=subprocess.run(cmd,cwd=str(WT),env=e,
                         capture_output=True,text=True,timeout=360)
    except subprocess.TimeoutExpired:
        fail(False,f"HYDRA_PREFLIGHT_INDEX_{i}_CPU_TIMEOUT_360S")
    fail(p.returncode==0 and
         "XP_F8_LEGACY_NATIVE_HYDRA_PREFLIGHT_PASS" in p.stdout,
         f"HYDRA_PREFLIGHT_INDEX_{i}: "+p.stdout[-350:]+p.stderr[-350:])
    print(p.stdout.strip(),flush=True)
    print("WAVE_REPRESENTATIVE_HYDRA_PASS",
          "index",i,"each_native_child_checks_its_own_config=True",
          "wave_indices",indices,flush=True)
def foreign_track_r_pid(card,owned=()):
    """Don't place a second Track R worker on a known occupied GPU."""
    known=set(int(x) for x in owned)
    tagged=[]
    for pid in card.process_pids:
        if pid in known:continue
        try:
            cmd=(Path("/proc")/str(pid)/"cmdline").read_bytes().lower()
        except (OSError,PermissionError):
            continue
        if any(tag in cmd for tag in
               (b"prdiff",b"track_r",b"xp_f8",b"xp_a",b"xp_t",
                b"legacy_daps400x5",b"daps4k")):
            tagged.append(pid)
    return tagged

def sample(cards,log,active):
    doc=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             gpu_inventory=[dataclasses.asdict(c) for c in cards],
             active={str(i):dict(gpu=v["gpu"],pid=v["proc"].pid)
                     for i,v in active.items()})
    log.write(json.dumps(doc,sort_keys=True)+"\n")
    log.flush()
def run(a,root,used,replayed):
    indices=list(range(0,4)) if a.wave==1 else list(range(4,8))
    frozen_preflight(a,indices)
    print("NATIVE_400x5_WAVE_TEMPLATE_PREFLIGHT_PASS",indices,
          "per_root_Hydra_check_before_GPU=True",flush=True)
    starts={}
    active={}
    logs={}
    try:
        telemetry=root/f"XP_F8_LEGACY_WAVE{a.wave}_GPU_TELEMETRY.jsonl"
        with telemetry.open("x") as feed:
            pending=indices.copy()
            failed=[]
            done=[]
            elapsed_all_children=0.
            tick=0
            while pending or active:
                fail(datetime.datetime.now(ZoneInfo("America/Detroit"))<
                     A10_DEADLINE,"DEADLINE_WHILE_WAVE_RUNNING")
                cards=collect(own_roots=[v["proc"].pid for v in active.values()])
                sample(cards,feed,active)
                # Count completed plus conservatively *all* in-flight elapsed.
                active_s=sum(time.monotonic()-v["start"] for v in active.values())
                total=used+replayed+(elapsed_all_children+active_s)/3600
                fail(total<70,"A10_SHARED_70_GPU_HARD_CAP")
                for idx,v in list(active.items()):
                    status=v["proc"].poll()
                    if status is None:continue
                    elapsed=time.monotonic()-v["start"]
                    elapsed_all_children+=elapsed
                    v["log"].close()
                    active.pop(idx)
                    if status!=0:
                        failed.append(dict(index=idx,rc=status,
                                          log=str(v["logpath"])))
                        print("ACTION NEEDED: NATIVE_ROOT_FAILURE",idx,
                              "exit",status,"log",v["logpath"],flush=True)
                    else:
                        p=root/f"replay_{idx:02d}/XP_F8_LEGACY_REPLAY_NATIVE_ROOT.json"
                        if not p.is_file():
                            failed.append(dict(index=idx,rc="RECEIPT_MISSING"))
                        else:
                            rec=read_json(p)
                            if rec.get("status")!="XP_F8_LEGACY_REPLAY_NATIVE_ROOT_PASS" or rec.get("index")!=idx:
                                failed.append(dict(index=idx,rc="RECEIPT_INVALID"))
                            else:
                                done.append(dict(index=idx,elapsed=elapsed,
                                                 receipt_sha256=sha(p)))
                                print("ROOT_GPU_RECEIPT_PASS",idx,
                                      round(elapsed,2),flush=True)
                # Fail-closed: no new work admitted after any worker failure.
                if not failed:
                    occupied={v["gpu"] for v in active.values()}
                    while pending and len(active)<4:
                        owned_pids=[v["proc"].pid for v in active.values()]
                        available=[c for c in cards
                            if c.index not in occupied and c.visibility=="known"
                            and not foreign_track_r_pid(c,owned_pids)]
                        gpu,reasons=choose_gpu(
                            available,REQUIRED_MIB,
                            safety_margin_mib=MARGIN_MIB,fraction=.80,
                            clean_only=False)
                        if gpu is None:
                            if tick%6==0:
                                print("WAIT_FOREIGN_COTENANCY_VRAM",
                                      reasons,flush=True)
                            break
                        idx=pending.pop(0)
                        command=[str(PY),"-u","-m",
                            "scripts.track_r.xp_f8_legacy_replay_gpu",
                            "--expected-head",a.expected_head,
                            "--index",str(idx),"--gpu",str(gpu),
                            "--outroot",str(root)]
                        env=os.environ.copy()
                        env.update(PYTHONPATH=str(WT),
                                   CUDA_VISIBLE_DEVICES=str(gpu),
                                   PYTHONDONTWRITEBYTECODE="1",
                                   PYTHONNOUSERSITE="1")
                        logfile=root/f"XP_F8_REPLAY_NATIVE_W{a.wave}_R{idx:02d}.log"
                        handle=logfile.open("x")
                        p=subprocess.Popen(command,cwd=str(WT),env=env,
                                           stdout=handle,stderr=subprocess.STDOUT)
                        active[idx]=dict(gpu=gpu,proc=p,start=time.monotonic(),
                                         log=handle,logpath=logfile)
                        occupied.add(gpu)
                        print("START_HISTORICAL_DAPS_ROOT",
                              idx,"gpu",gpu,"pid",p.pid,flush=True)
                tick+=1
                if active or pending:time.sleep(10)
        summary=dict(status=("XP_F8_LEGACY_WAVE"+str(a.wave)+"_GPU_PASS"
                             if len(done)==4 and not failed else
                             "STOP_F8_LEGACY_WAVE_GPU"),
               expected_head=a.expected_head,
               wave=a.wave,accepted_indices=sorted(d["index"] for d in done),
               failed=failed,receipt_sha256={str(x["index"]):x["receipt_sha256"] for x in done},
               n_telemetry_samples=sum(1 for _ in telemetry.open()),
               shared_gpu_h_before_wave=used+replayed,
               incremental_gpu_h=elapsed_all_children/3600,
               count_failed_elapsed_towards_budget=True,
               native_400x5=True,clean_timing_claim=False,
               foreign_pids_are_recorded=True,gt_opened=False,
               source_native_checkout_modified=False)
        write_x(root/f"XP_F8_LEGACY_REPLAY_WAVE{a.wave}.json",summary)
        fail(summary["status"].endswith("_GPU_PASS"),"GPU_WAVE_FAILED")
        print("XP_F8_LEGACY_WAVE_GPU_PASS",a.wave,flush=True)
    finally:
        # If a controller preflight fails after a child was already launched,
        # never orphan it: drain owned workers without signalling foreign jobs.
        for v in active.values():
            print("DRAIN_OWN_NATIVE_WORKER",v["proc"].pid,flush=True)
            v["proc"].wait()
            if not v["log"].closed:v["log"].close()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--wave",type=int,choices=(1,2),required=True)
    p.add_argument("--run",required=True)
    a=p.parse_args()
    root,used,replayed=ensure_wave(a)
    run(a,root,used,replayed)
if __name__=="__main__":
    try:main()
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_F8_LEGACY_SUPERVISOR "+str(e),flush=True)
        sys.exit(8)
