#!/usr/bin/env python3
"""B26.1 native-NP H/R root runner and resumable GPU worker.

H reproduces historical NP with y_plus for scoring/projection.
R changes only scoring/selector observation to y_raw; projection remains y_plus.
Historical source files are imported but never modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
HIST = REPO / "scripts" / "b24" / "run_b24_3_np_branching.py"
MODEL = Path("/egr/research-pac/huang248/models/ffhq_10m.pt")
MODEL_SHA = "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
DIFFFPR = Path("/egr/research-pac/huang248/external/DiffFPR")
DIFFFPR_HEAD = "a45ffe58f18fed8a63d3446600424e2b08733524"
NP_STEPS = 1000
PROJ_START = 300
SCORE_RADIUS = 0.6
PROJ_RADIUS = 0.2
PROJ_SCHEDULE = "300:0.2"
EXPECTED_UNET = 2200
HARD_CEILING_MIB = 52452


def load_hist():
    spec = importlib.util.spec_from_file_location("b26_historical_np", HIST)
    if spec is None or spec.loader is None:
        raise ImportError(HIST)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_hist()


def readj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def writej_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def gpu_inventory() -> list[dict[str, Any]]:
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
    rows=[]
    for line in text.splitlines():
        p=[x.strip() for x in line.split(",")]
        if len(p)==5:
            rows.append({"index":int(p[0]),"uuid":p[1],"total_mib":int(p[2]),"used_mib":int(p[3]),"free_mib":int(p[4])})
    return rows


@dataclass
class Ctx:
    selector: Any
    unet: Any
    scheduler: Any
    device: torch.device
    score_target: torch.Tensor
    projection_target: torch.Tensor
    pad: int
    timesteps: torch.Tensor
    proj_schedule: Any
    image_id: str
    arm: str


@torch.no_grad()
def candidate_set(ctx: Ctx, branch, transition: int, k: int):
    base.rng_restore(branch.cpu_rng, branch.cuda_rng, ctx.device)
    x0_hat = branch.x0
    projection_applied = transition >= PROJ_START
    if projection_applied:
        radius = ctx.selector.base.radius_at_step(ctx.proj_schedule, transition)
        x0_hat = ctx.selector.base.enforce_oversampled_lowfreq(
            x0_hat, ctx.projection_target, ctx.pad, radius
        )
    t_next = int(ctx.timesteps[transition + 1])
    alpha = ctx.scheduler.alphas_cumprod[t_next].to(device=ctx.device, dtype=x0_hat.dtype)
    sa, s1 = torch.sqrt(alpha), torch.sqrt(1.0-alpha)
    xs=[]; noises=[]; plus_scores=[]; raw_scores=[]; plus_mses=[]; raw_mses=[]
    mask = None
    score_fft_evals = 0
    diagnostic_fft_evals = 0
    for j in range(k):
        if j == 0 and branch.eps_prev is not None and k > 1:
            eps_cand = branch.eps_prev
        else:
            eps_cand = torch.randn_like(x0_hat)
        z = sa*x0_hat + s1*eps_cand
        tt = torch.tensor([t_next], device=ctx.device, dtype=torch.long)
        eps_pred = ctx.unet(z, tt).sample
        x0 = (z-s1*eps_pred)/sa
        base.require_model_range(x0, f"{ctx.arm}/{ctx.image_id}/{branch.lineage}/t{transition}/c{j}")

        # Actual ranking score uses the historical helper exactly.
        actual_target = ctx.projection_target if ctx.arm == "H" else ctx.score_target
        actual = ctx.selector.base.oversampled_lowfreq_mag_l2(x0, actual_target, ctx.pad, SCORE_RADIUS)
        score_fft_evals += 1

        # One shared diagnostic magnitude gives both plus/raw counterfactual scores and MSEs.
        mag = ctx.selector.base.oversampled_magnitude(x0, ctx.pad)
        diagnostic_fft_evals += 1
        if mask is None:
            _,_,h,w=mag.shape
            mhw=ctx.selector.base.centered_lowfreq_mask(h,w,SCORE_RADIUS,mag.device)
            mask=mhw[None,None,:,:].expand_as(mag)
        rp = mag[mask]-ctx.projection_target[mask]
        rr = mag[mask]-ctx.score_target[mask]  # score_target is raw in R, plus in H; raw overwritten below
        # Always get explicit raw from attached attribute set by caller.
        rr = mag[mask]-ctx.raw_target[mask]
        pscore=torch.norm(rp); rscore=torch.norm(rr)
        if ctx.arm == "H":
            if not torch.allclose(actual, pscore, rtol=1e-6, atol=1e-7):
                raise RuntimeError("historical plus-score replay formula mismatch")
        else:
            if not torch.allclose(actual, rscore, rtol=1e-6, atol=1e-7):
                raise RuntimeError("raw-score formula mismatch")
        xs.append(x0.detach()); noises.append(eps_cand.detach().clone())
        plus_scores.append(float(pscore.detach().cpu()))
        raw_scores.append(float(rscore.detach().cpu()))
        plus_mses.append(float(torch.mean(rp.square()).detach().cpu()))
        raw_mses.append(float(torch.mean(rr.square()).detach().cpu()))
    branch.cpu_rng, branch.cuda_rng = base.rng_snapshot(ctx.device)
    return xs,noises,plus_scores,raw_scores,plus_mses,raw_mses,k,score_fft_evals,diagnostic_fft_evals,projection_applied


def run_root(ctx: Ctx, root_seed: int):
    # base.initialize_branch only needs selector/unet/scheduler/device/timesteps/image_id.
    proxy = type("Proxy", (), {})()
    for name in ("selector","unet","scheduler","device","timesteps","image_id"):
        setattr(proxy,name,getattr(ctx,name))
    b, initial = base.initialize_branch(proxy, int(root_seed), f"root{ctx.root_index}")
    unet = initial
    trace=[]
    post_plus=[]; post_raw=[]
    winner_disagree=0; winner_disagree_pre=0
    score_fft=0; diag_fft=0; projections=0
    for i in range(999):
        k=5 if i<PROJ_START else 1
        xs,noises,ps,rs,pm,rm,used,sf,df,proj = candidate_set(ctx,b,i,k)
        unet += used; score_fft += sf; diag_fft += df; projections += int(proj)
        pwin=min(range(k),key=lambda j:(ps[j],j))
        rwin=min(range(k),key=lambda j:(rs[j],j))
        actual=pwin if ctx.arm=="H" else rwin
        if pwin!=rwin:
            winner_disagree += 1
            if i<PROJ_START: winner_disagree_pre += 1
        b.x0,b.eps_prev=xs[actual],noises[actual]
        b.last_transition=i
        if i>=PROJ_START:
            post_plus.append(pm[actual]); post_raw.append(rm[actual])
        trace.append({
            "transition":i,"k":k,"projection":proj,"winner_plus":pwin,"winner_raw":rwin,
            "winner_actual":actual,"winner_disagree":pwin!=rwin,
            "selected_plus_lf_mse":pm[actual],"selected_raw_lf_mse":rm[actual]
        })
    if unet != EXPECTED_UNET:
        raise RuntimeError(f"UNet count {unet} != historical {EXPECTED_UNET}")
    return b,trace,{
        "initial_unet_evals":initial,"proposal_unet_evals":unet-initial,"total_unet_evals":unet,
        "score_fft_evals":score_fft,"diagnostic_fft_evals":diag_fft,"projection_fft_ifft_ops":projections,
        "winner_disagreement_steps":winner_disagree,"winner_disagreement_preprojection":winner_disagree_pre,
        "post_selector_steps":len(post_plus),
        "selector_plus_mean":float(sum(post_plus)/len(post_plus)),
        "selector_raw_mean":float(sum(post_raw)/len(post_raw)),
    }


def job_list(manifest: dict[str,Any], stage: str):
    rows=manifest["rows"]
    if stage=="smoke": rows=[r for r in rows if r["smoke_image"]]; roots=[0]
    elif stage=="dev16": rows=[r for r in rows if r["dev16"]]; roots=range(4)
    elif stage=="dev80": roots=range(4)
    else: raise ValueError(stage)
    jobs=[]
    for r in rows:
        for root in roots:
            for arm in ("H","R"):
                jobs.append((r,int(root),arm))
    return jobs


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--output-root",type=Path,required=True)
    ap.add_argument("--stage",choices=("smoke","dev16","dev80"),required=True)
    ap.add_argument("--physical-gpu",type=int,choices=range(4),required=True)
    ap.add_argument("--worker-index",type=int,required=True)
    ap.add_argument("--worker-count",type=int,required=True)
    ap.add_argument("--min-free-mib",type=int,default=10240)
    args=ap.parse_args()
    if args.worker_count<1 or not (0<=args.worker_index<args.worker_count): raise ValueError("worker index/count")
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
    if visible!=str(args.physical_gpu): raise RuntimeError(f"CUDA binding {visible!r} != {args.physical_gpu}")
    inv=gpu_inventory(); grow=next((r for r in inv if r["index"]==args.physical_gpu),None)
    if grow is None or grow["free_mib"]<args.min_free_mib: raise RuntimeError(f"GPU admission failed: {grow}")
    if sha256_file(MODEL)!=MODEL_SHA: raise RuntimeError("model SHA drift")
    if git_head(DIFFFPR)!=DIFFFPR_HEAD: raise RuntimeError("DiffFPR head drift")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    device=torch.device("cuda:0"); torch.cuda.set_device(device)
    manifest=readj(args.manifest.resolve())
    if manifest.get("status")!="PASS" or manifest.get("development_count")!=80 or manifest.get("confirmation_payloads_accessed") is not False:
        raise RuntimeError("bad frozen manifest")
    out=args.output_root.resolve(); out.mkdir(parents=True,exist_ok=True)

    selector=base.load_module("b26_np_selector",base.SELECTOR_PATH)
    load0=time.perf_counter()
    bundle=selector.load_guided_diffusion_model(model_path=str(MODEL),device=device,preset="difffpr_ffhq_10m",guided_diffusion_dir=str(DIFFFPR),strict=True)
    torch.cuda.synchronize(device); model_load=time.perf_counter()-load0
    bundle.scheduler.set_timesteps(NP_STEPS,device=device)
    if len(bundle.scheduler.timesteps)!=NP_STEPS: raise RuntimeError("scheduler drift")
    pad=selector.base.oversample_pad(256,2.0)
    schedule=selector.base.parse_radius_schedule(PROJ_SCHEDULE,PROJ_RADIUS)

    jobs=job_list(manifest,args.stage)
    assigned=[j for idx,j in enumerate(jobs) if idx%args.worker_count==args.worker_index]
    worker_started=time.perf_counter(); attempted=completed=skipped=0; max_proc=0; max_reserved=0.0
    for row,root_idx,arm in assigned:
        jid=f"{row['image_id']}_root{root_idx}_{arm}"
        jdir=out/"jobs"/str(row["image_id"])/f"root{root_idx}"/arm
        done=jdir/"JOB_COMPLETE.json"
        if done.is_file():
            v=readj(done)
            if v.get("status")=="PASS": skipped+=1; continue
        if jdir.exists(): raise RuntimeError(f"preserved incomplete/failed job requires scoped decision: {jdir}")
        attempt=jdir/"attempt01"; attempt.mkdir(parents=True,exist_ok=False); attempted+=1
        start=time.perf_counter(); monitor=base.GPUMonitor(args.physical_gpu,1.0); monitor.start()
        try:
            meas=Path(row["measurement_path"]); gtpath=Path(row["ground_truth_tensor_path"])
            if sha256_file(meas)!=row["measurement_file_sha256"]: raise RuntimeError("measurement file SHA drift")
            yraw=base.unwrap_tensor(meas,("measurement","y","observation"),device).to(torch.float32)
            if base.tensor_sha256(yraw)!=row["measurement_tensor_sha256"] or tuple(yraw.shape)!=(1,3,384,384): raise RuntimeError("measurement tensor drift")
            yplus=yraw.clamp_min(0.0)
            ctx=Ctx(selector,bundle.unet,bundle.scheduler,device,yplus if arm=="H" else yraw,yplus,pad,bundle.scheduler.timesteps,schedule,str(row["image_id"]),arm)
            ctx.raw_target=yraw; ctx.root_index=root_idx
            bundle.scheduler.set_timesteps(NP_STEPS,device=device); ctx.timesteps=bundle.scheduler.timesteps
            torch.cuda.reset_peak_memory_stats(device)
            branch,trace,acct=run_root(ctx,int(row["np_roots"][root_idx]["root_seed"]))
            torch.cuda.synchronize(device)
            gpu=monitor.stop(); monitor=None
            peak_reserved=torch.cuda.max_memory_reserved(device)/1048576.0
            max_reserved=max(max_reserved,peak_reserved)
            if gpu.get("max_b24_process_gpu_mib") is not None:
                max_proc=max(max_proc,int(gpu["max_b24_process_gpu_mib"]))
            if max_proc>HARD_CEILING_MIB: raise RuntimeError(f"process memory cap {max_proc}>{HARD_CEILING_MIB}")

            # GT is loaded only after all runtime trajectory decisions are complete.
            gt=base.unwrap_tensor(gtpath,("ground_truth","image","x"),device).to(torch.float32)
            if base.tensor_sha256(gt)!=row["ground_truth_tensor_sha256"]: raise RuntimeError("GT tensor drift")
            psnr=base.psnr_raw(branch.x0,gt)
            term_sha=base.tensor_sha256(branch.x0)
            rng_sha=branch.rng_sha256(); noise_sha=base.tensor_sha256(branch.eps_prev) if branch.eps_prev is not None else None
            accepted=row["np_roots"][root_idx]
            replay=None
            if arm=="H":
                replay={
                    "terminal_hash_match":term_sha==accepted["reconstruction_sha256"],
                    "selector_plus_abs_error":abs(acct["selector_plus_mean"]-float(accepted["selector_plus_mean"])),
                    "rng_hash_match":rng_sha==accepted.get("rng_sha256"),
                    "selected_noise_hash_match":noise_sha==accepted.get("selected_noise_sha256"),
                }
                if not replay["terminal_hash_match"] or replay["selector_plus_abs_error"]>1e-12:
                    raise RuntimeError(f"historical replay mismatch {jid}: {replay}")
            torch.save({"reconstruction":branch.x0.detach().cpu()},attempt/"terminal.pt")
            tmp=attempt/"TRACE.jsonl.tmp"
            with tmp.open("w",encoding="utf-8") as f:
                for tr in trace: f.write(json.dumps(tr,sort_keys=True)+"\n")
            os.replace(tmp,attempt/"TRACE.jsonl")
            result={
                "schema_version":"b26.np-root.v1","status":"PASS","image_id":row["image_id"],"screening_stratum":row["screening_stratum"],
                "shared_failure_10":bool(row["shared_failure_10"]),"arm":arm,"root_index":root_idx,"root_seed":int(accepted["root_seed"]),
                "measurement_file_sha256":row["measurement_file_sha256"],"measurement_tensor_sha256":row["measurement_tensor_sha256"],
                "score_target":"y_plus" if arm=="H" else "y_raw","projection_target":"y_plus",
                "runtime_ground_truth_decisions":False,"confirmation_payload_accessed":False,"new_measurement_generated":False,
                "psnr_raw_db":psnr,"good25":bool(psnr>=25.0),"bad20":bool(psnr<20.0),"reconstruction_sha256":term_sha,
                "rng_sha256":rng_sha,"selected_noise_sha256":noise_sha,"accounting":acct,"historical_replay":replay,
                "wall_seconds":time.perf_counter()-start,"model_load_seconds_worker_shared":model_load,"gpu_monitor":gpu,"peak_torch_reserved_mib":peak_reserved,
                "trace_path":str((attempt/"TRACE.jsonl").resolve()),"terminal_path":str((attempt/"terminal.pt").resolve())
            }
            writej_atomic(attempt/"result.json",result)
            writej_atomic(done,{"schema_version":"b26.job-complete.v1","status":"PASS","result":str((attempt/"result.json").resolve()),"result_sha256":sha256_file(attempt/"result.json")})
            completed+=1
            print(json.dumps({"JOB_PASS":jid,"psnr":psnr,"unet":acct["total_unet_evals"],"wall":result["wall_seconds"]},sort_keys=True),flush=True)
        except Exception as exc:
            try:
                if 'monitor' in locals() and monitor is not None: gpu=monitor.stop()
            except Exception: gpu={}
            writej_atomic(attempt/"FAILURE.json",{"status":"FAIL","job":jid,"error":f"{type(exc).__name__}: {exc}","confirmation_payload_accessed":False})
            print(json.dumps({"JOB_FAIL":jid,"error":f"{type(exc).__name__}: {exc}"},sort_keys=True),flush=True)
            raise
        finally:
            gc.collect(); torch.cuda.empty_cache()
    receipt={
        "schema_version":"b26.gpu-worker.v1","status":"PASS","stage":args.stage,"worker_index":args.worker_index,"worker_count":args.worker_count,
        "physical_gpu":args.physical_gpu,"gpu_uuid":grow["uuid"],"assigned_jobs":len(assigned),"attempted":attempted,"completed":completed,"skipped":skipped,
        "reservation_wall_seconds":time.perf_counter()-worker_started,"model_load_seconds":model_load,"max_process_gpu_mib":max_proc,"max_torch_reserved_mib":max_reserved,
        "confirmation_payload_accessed":False,"new_measurements_generated":False
    }
    writej_atomic(out/f"WORKER_{args.stage}_{args.worker_index:02d}.json",receipt)
    print(json.dumps(receipt,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
