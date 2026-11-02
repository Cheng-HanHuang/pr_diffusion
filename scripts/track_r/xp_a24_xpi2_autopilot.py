#!/usr/bin/env python3
"""A19 XP-I2 D1 ImageNet PC-DAPS: exact 2-bag replay then resumable 100 bags.

The worker is frozen in xp_a24_xpi2_pcdaps_gpu.py. This controller never
scores GT, never changes native settings and never reuses XP-N measurements.
One worker per GPU, dirty GPU permitted at <=80% memory, timing claims forbidden.
All launched worker wall time, including failed attempts, counts against caps.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from prdiffusion.track_r.xp_gpu_telemetry import collect, choose_gpu
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT, BASE, gate
from scripts.track_r.xp_a24_xpi2_prepare_cpu import sha

PROJECT_LOCK = BASE / "XP_A12_CAMPAIGN_CONTROLLER.lock"
STAGE_CAP_H = 40.0
TAIL_CAP_H = 60.0
PROJECT_CAP_H = 400.0
# 80% VRAM admission; reservation sized for native PC-DAPS, not observed free.
REQUEST_MIB = 16384
MARGIN_MIB = 2048
MIN_RESERVATION_H = .20
STOP = False

def on_stop(*_):
    global STOP
    STOP = True

signal.signal(signal.SIGTERM, on_stop)
signal.signal(signal.SIGINT, on_stop)

def dump(path, data):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp." + str(os.getpid()))
    with tmp.open("x") as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)

def load(path):
    p = Path(path)
    gate(p.is_file() and not p.is_symlink() and
         ROOT.resolve() in p.resolve().parents, "I2_SAFE_JSON")
    return json.loads(p.read_text())

def git(wt, *cmd):
    return subprocess.check_output(
        ["git", "-C", str(wt), *cmd], text=True, timeout=45).strip()

def frozen_inputs(a):
    wt = Path(__file__).resolve().parents[2]
    gate(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "I2_CONTROLLER_CPU")
    gate(git(wt, "rev-parse", "HEAD") == a.expected_head and
         not git(wt, "status", "--porcelain"), "I2_EXECUTOR_HEAD")
    prep = Path(a.prep)
    gate(prep.parent == BASE and prep.name.startswith("xp_a24_xpi2_prepare_")
         and prep.is_dir() and not prep.is_symlink(), "I2_PREP_PATH")
    receipt = load(prep / "XP_A24_XPI2_PREP_RECEIPT.json")
    mfpath = prep / "XP_A24_XPI2_100_BAGS.json"
    mf = load(mfpath)
    gate(receipt["status"] == "A19_XPI2_D1_100_BAGS_CPU_FROZEN_NOT_GENERATED"
         and receipt["expected_head"] == a.roots_source_head
         and sha(mfpath) == receipt["manifest_sha256"]
         and mf["n_bags"] == 100 and mf["bag_size"] == 4
         and mf["no_GT_paths"] is True
         and mf["source_y_manifest_sha256"] ==
             receipt["original_D1_y_sha256"], "I2_PROVENANCE")
    gate(len(mf["rows"]) == 100 and
         len({r["uid"] for r in mf["rows"]}) == 100, "I2_GRID")
    source = sorted(BASE.glob("xp_a22_xpn_continue_*/XP_N_CONTINUATION_FINAL_STATUS.json"))
    accepted = [(p, load(p)) for p in source]
    accepted = [(p, j) for p, j in accepted
                if j.get("status") ==
                   "A22_XPN_ALL6400_NATIVE_GENERATION_COMPLETE"
                and j.get("complete_chunks") == 90
                and j.get("accepted_full_chunk_roots") == 6400]
    gate(len(accepted) == 1, "I2_UNIQUE_ACCEPTED_XPN_LEDGER")
    ledger_path, ledger = accepted[0]
    baseline = float(ledger["project_total_gpu_h"])
    gate(0 < baseline < PROJECT_CAP_H, "I2_XPN_PROJECT_LEDGER_RANGE")
    # A22's project total is an immutable historical snapshot; the
    # global campaign lock serializes later ledger-consuming controllers.
    # Fails closed if another registered A24 successor already exists.
    return wt, prep, mf, receipt, ledger_path, baseline

def bagdir(run, row):
    return run / "roots" / ("bag_" +
        hashlib.sha256(row["uid"].encode()).hexdigest()[:24])

def verify_bag(run, row, mf, manifest_sha, runner):
    dest = bagdir(run, row)
    receipt_path = dest / "XP_A24_I2_BAG_COMPLETE.json"
    if not receipt_path.is_file():
        gate(not dest.exists(), "I2_UNRECEIPTED_PARTIAL_BAG")
        return None
    r = load(receipt_path)
    gate(r["status"] == "A24_XPI2_NATIVE_FOUR_PARTICLE_BAG_PASS"
         and r["uid"] == row["uid"]
         and r["image_id"] == row["image_id"]
         and r["measurement_sha256"] == row["measurement_sha256"]
         and r["manifest_sha256"] == manifest_sha
         and r["runner_head"] == runner
         and r["source_head"] == mf["source_head"]
         and r["GT_access"] is False
         and r["no_GT_scoring"] is True
         and len(r["particle_pngs"]) == 4
         and [x["particle"] for x in r["particle_pngs"]] == list(range(4))
         and sha(dest / "particles.pt") == r["particles_pt_sha256"],
         "I2_BAG_RECEIPT_DRIFT")
    for entry in r["particle_pngs"]:
        p = Path(entry["path"])
        gate(p.parent == dest and sha(p) == entry["sha256"],
             "I2_PARTICLE_SHA_DRIFT")
    return r

def replay_equal(a, b):
    # Bitwise outputs: PNG + full saved tensor serialization.
    return (a["particles_pt_sha256"] == b["particles_pt_sha256"] and
            [r["sha256"] for r in a["particle_pngs"]] ==
            [r["sha256"] for r in b["particle_pngs"]])

def spawn(wt, run, prep, head, roots_head, i, gpu, replay, proof, attempt):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(wt),
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        PYTHONPYCACHEPREFIX=str(run / "pycache" / ("bag_%03d_%d" % (i, attempt))),
        CUBLAS_WORKSPACE_CONFIG=":4096:8", TMPDIR=str(ROOT / "tmp"),
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2")
    command = [str(ROOT / "conda-envs/prdiff_track_r/bin/python"),
        str(wt / "scripts/track_r/xp_a24_xpi2_pcdaps_gpu.py"),
        "--run", str(run), "--prep", str(prep),
        "--expected-head", head, "--roots-source-head", roots_head,
        "--index", str(i), "--gpu", str(gpu)]
    if replay:
        command.append("--replay-only")
    else:
        command.extend(["--replay-receipt", str(proof)])
    log = run / "logs" / ("bag_%03d_try%d.log" % (i, attempt))
    handle = log.open("x")
    proc = subprocess.Popen(command, cwd=str(wt), env=env,
        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    return dict(proc=proc, handle=handle, started=time.monotonic(),
                gpu=gpu, i=i, log=str(log), attempt=attempt,
                foreign=set(), peak=0, unknown=False, pressure=False)

def kill_ours(active):
    for v in active.values():
        if v["proc"].poll() is None:
            try:
                os.killpg(v["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

def choose(cards, active, preferred=None):
    occupied = {v["gpu"] for v in active.values()}
    available = [c for c in cards
                 if c.index not in occupied and c.visibility == "known"]
    if preferred is not None:
        available = [c for c in available if c.index == preferred]
    return choose_gpu(available, REQUEST_MIB, safety_margin_mib=MARGIN_MIB,
                      fraction=.80, clean_only=False)[0]

def main(a):
    wt, prep, mf, prep_rec, xp_n_ledger, baseline = frozen_inputs(a)
    out = Path(a.run)
    gate(out.parent == BASE and out.name.startswith("xp_a24_xpi2_prod_")
         and not out.is_symlink(), "I2_RUN_PATH")
    if a.preflight_only:
        cards = collect()
        print("A24_XPI2_SUPERVISOR_PREFLIGHT",
              "BASELINE_H", baseline, "ELIGIBLE_DIRTY_GPUS",
              [c.index for c in cards if choose([c], {}) is not None],
              "SHARED_LEDGER", str(xp_n_ledger), flush=True)
        return
    # A completed or interrupted run is never silently overwritten:
    # resume explicitly with --resume; validate all existing bytes first.
    if out.exists():
        gate(a.resume and (out / "XP_A24_XPI2_CONTROLLER_STATE.json").is_file(),
             "I2_EXISTING_RUN_NEEDS_EXPLICIT_RESUME")
    else:
        gate(not a.resume, "I2_RESUME_MISSING_RUN")
        out.mkdir(mode=0o700)
        for d in ("logs", "pycache", "replayA", "replayB", "roots", "cost"):
            (out / d).mkdir()
    # Global GPU lock: other Track R controllers must not start in
    # parallel; subsequent stages consume this terminal ledger.
    with PROJECT_LOCK.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state_path = out / "XP_A24_XPI2_CONTROLLER_STATE.json"
        if a.resume:
            state = load(state_path)
            gate(state["runner_head"] == a.expected_head
                 and state["manifest_sha256"] == prep_rec["manifest_sha256"]
                 and state["project_prior_ledger_sha256"] == sha(xp_n_ledger)
                 and state["active"] == []
                 and state["status"] != "A24_XPI2_COMPLETE",
                 "I2_RESUME_STATE")
        else:
            state = dict(status="A24_XPI2_INITIAL_REPLAY",
                runner_head=a.expected_head,
                manifest_sha256=prep_rec["manifest_sha256"],
                project_prior_ledger_sha256=sha(xp_n_ledger),
                baseline_project_gpu_h=baseline,
                baseline_tail_gpu_h=0.0,
                charged_gpu_h=0.0, active=[], completed=[],
                attempts=[], replay_receipt=None, no_GT_scoring=True)
        def save():
            dump(state_path, state)
        def spend(active):
            return state["charged_gpu_h"] + sum(
                (time.monotonic()-v["started"])/3600 for v in active.values())
        def admission(active, extra=1):
            projected = spend(active) + (len(active)+extra)*MIN_RESERVATION_H
            return (projected <= STAGE_CAP_H and
                    projected <= TAIL_CAP_H and
                    baseline + projected <= PROJECT_CAP_H)
        def charge(v, rc):
            elapsed = (time.monotonic()-v["started"])/3600
            state["charged_gpu_h"] += elapsed
            record = dict(index=v["i"], attempt=v["attempt"],
                replay=v.get("replay", False),
                stage_gpu_h=elapsed, physical_gpu=v["gpu"], rc=rc,
                foreign_process_pids=sorted(v["foreign"]),
                max_used_mib=v["peak"], telemetry_unknown=v["unknown"],
                memory_pressure=v["pressure"], log=v["log"])
            state["attempts"].append(record)
            dump(out/"cost"/("attempt_%04d.json" % len(state["attempts"])),record)
            save()
        def execute(indices, run, replay=False, selected_gpu=None,
                    proof=None, one_at_time=False):
            active = {}
            finished = {}
            while len(finished) < len(indices):
                cards = collect(own_roots=[x["proc"].pid for x in active.values()])
                for i, v in list(active.items()):
                    card = next((c for c in cards if c.index == v["gpu"]), None)
                    if card is None or card.visibility != "known":
                        v["unknown"] = True
                    else:
                        v["foreign"].update(card.foreign_pids)
                        v["peak"] = max(v["peak"], card.used_mib)
                        if card.used_mib > int(.80*card.total_mib) and not v["pressure"]:
                            v["pressure"] = True
                            os.killpg(v["proc"].pid, signal.SIGTERM)
                    rc = v["proc"].poll()
                    if rc is None:
                        continue
                    rc = v["proc"].wait()
                    v["handle"].close()
                    v["replay"] = replay
                    del active[i]
                    state["active"] = [
                        dict(index=j, pid=x["proc"].pid, gpu=x["gpu"])
                        for j,x in active.items()]
                    charge(v, rc)
                    if rc != 0 or v["unknown"] or v["pressure"]:
                        raise RuntimeError("STOP_A24_I2_NATIVE_RC_OR_TELEMETRY_%03d_%d" % (i,rc))
                    r = verify_bag(run, mf["rows"][i], mf,
                                   prep_rec["manifest_sha256"], a.expected_head)
                    gate(r is not None, "I2_NATIVE_NO_RECEIPT")
                    finished[i] = r
                    if not replay:
                        state["completed"].append(i)
                    save()
                if STOP or (out/"STOP_XPI2").exists():
                    raise RuntimeError("STOP_A24_I2_OPERATOR")
                # Watchdog charges in-flight GPU-hours, not just completed.
                if not admission(active, 0):
                    raise RuntimeError("STOP_A24_I2_STAGE_TAIL_OR_PROJECT_CAP")
                for i in indices:
                    if i in finished or i in active:
                        continue
                    if (one_at_time and active) or len(active) >= 4:
                        break
                    if not admission(active, 1):
                        break
                    gpu = choose(cards, active, preferred=selected_gpu)
                    if gpu is None:
                        break
                    attempt = 1 + sum(int(x["index"]==i and x["replay"]==replay)
                                      for x in state["attempts"])
                    worker = spawn(wt, run, prep, a.expected_head,
                        a.roots_source_head, i, gpu, replay, proof, attempt)
                    active[i] = worker
                    state["active"] = [
                        dict(index=j, pid=x["proc"].pid, gpu=x["gpu"])
                        for j,x in active.items()]
                    save()
                time.sleep(4)
            return finished
        save()
        try:
            # A and B replay are separate output namespaces; the same two
            # bags, fixed seeds and *same GPU* must match bitwise.
            proofpath = BASE / ("XP_A24_XPI2_REPLAY_" + out.name + ".json")
            if not state["replay_receipt"]:
                replayruns = []
                for label in ("replayA","replayB"):
                    d = out / label
                    # Worker permits only xp_a24_xpi2_replay_* as run.parent==BASE.
                    # Use separate top-level runs, never symlink or relax worker.
                    d = BASE / ("xp_a24_xpi2_replay_" + label + "_" + out.name)
                    gate(d.parent == BASE and not d.exists(), "I2_REPLAY_DIR_EXISTS")
                    d.mkdir()
                    for child in ("logs","roots","pycache"): (d/child).mkdir()
                    replayruns.append(d)
                chosen = None
                while chosen is None:
                    if STOP:
                        raise RuntimeError("STOP_A24_I2_REPLAY_WAIT_OPERATOR")
                    chosen = choose(collect(), {})
                    if chosen is None: time.sleep(10)
                answers = []
                for d in replayruns:
                    answers.append(execute([0,1],d,True,chosen,None,True))
                gate(all(replay_equal(answers[0][i],answers[1][i])
                         for i in (0,1)), "I2_REPLAY_NOT_BITWISE_IDENTICAL")
                proof = dict(status="A24_XPI2_TWO_BAG_EXACT_REPLAY_PASS",
                    n_bags=2, manifest_sha256=prep_rec["manifest_sha256"],
                    model_sha256=mf["model_sha256"],
                    native_pc_head=mf["source_head"],
                    replay_roots=[str(p) for p in replayruns],
                    replay_shas=[[answers[k][i]["particles_pt_sha256"]
                                  for i in (0,1)] for k in (0,1)],
                    no_GT_scoring=True)
                gate(not proofpath.exists(), "I2_PROOF_PATH_ALREADY_EXISTS")
                dump(proofpath,proof)
                state["replay_receipt"]=str(proofpath)
                save()
            # This validated replay receipt is required by EVERY production
            # worker. Reverify before resuming.
            proof = load(state["replay_receipt"])
            gate(proof["status"]=="A24_XPI2_TWO_BAG_EXACT_REPLAY_PASS"
                 and proof["manifest_sha256"]==prep_rec["manifest_sha256"],
                 "I2_REPLAY_PROOF_SHA")
            state["status"]="A24_XPI2_PRODUCTION"
            save()
            for i in state["completed"]:
                gate(verify_bag(out,mf["rows"][i],mf,
                                prep_rec["manifest_sha256"],a.expected_head)
                     is not None,"I2_RESUME_COMPLETED_BAG")
            pending=[i for i in range(100) if i not in state["completed"]]
            if pending:
                execute(pending,out,False,None,proofpath,False)
            gate(len(set(state["completed"]))==100 and
                 len(state["completed"])==100,"I2_100_BAGS_GRID")
            state["status"]="A24_XPI2_COMPLETE"
        except BaseException as exc:
            state["status"]="STOP_A24_XPI2_REQUIRES_RECONCILIATION"
            state["error"]=str(exc)[:700]
            raise
        finally:
            # Active workers are always ours. Charge interrupted attempts
            # before exiting, never kill or charge foreign compute PIDs.
            # Every state update is atomic; no partial run is discarded.
            leftover=locals().get("active",{})
            kill_ours(leftover)
            for v in leftover.values():
                v["proc"].wait()
                v["handle"].close()
                v["replay"]=False
                charge(v, v["proc"].returncode)
            state["active"]=[]
            save()
            final=dict(state)
            final["project_total_gpu_h"]=baseline+state["charged_gpu_h"]
            final["tail_total_gpu_h"]=state["charged_gpu_h"]
            final["n_completed_bags"]=len(state["completed"])
            final["n_particles"]=4*len(state["completed"])
            final["gpu_clean_required"]=False
            final["performance_timing_claim"]=False
            dump(out/"XP_A24_XPI2_FINAL_STATUS.json",final)
            print("A24_XPI2_TERMINAL",state["status"],
                  "BAGS",len(state["completed"]),
                  "CHARGED_GPU_H",state["charged_gpu_h"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("expected-head","roots-source-head","prep","run"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    try:main(parser.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A24_XPI2_AUTOPILOT",exc,flush=True)
        sys.exit(8)
