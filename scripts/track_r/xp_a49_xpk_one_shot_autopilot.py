#!/usr/bin/env python3
"""A49 XP-K one-shot gated autopilot.

One PAC launch owns the remaining registered XP-K campaign:
CPU source/binding gates -> frozen manifest -> original-parameter identity
replays -> DAPS F8 lineage check -> one-image pilots -> measured budget gate
-> full 100x4 calibrated panels -> frozen CPU scoring/forecast comparison
-> CP4 ledger -> immutable publication commit.

Scientific gates are fail-closed. Resource contention waits/reroutes; it never
changes tolerances, native settings, seeds, or the registered panel.
"""
from __future__ import annotations
import argparse, atexit, datetime, fcntl, hashlib, json, math, os, shlex, signal
import subprocess, sys, time, traceback, uuid
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/egr/research-pac/huang248")
WT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/pr_diffusion/track_r"
CPU_PY = ROOT / "conda-envs/prdiff_track_r/bin/python"
PYTHON = {
    "SITCOM": ROOT / "conda-envs/sitcom_ode_bw/bin/python",
    "FAST": ROOT / "conda-envs/fast_dips_a15_blackwell/bin/python",
    "DAPS": ROOT / "conda-envs/daps/bin/python",
}
FAMILIES = ("SITCOM", "FAST", "DAPS")
RESERVE_GIB = {"SITCOM": 28.0, "FAST": 28.0, "DAPS": 28.0}
DEADLINE = datetime.datetime(2026, 11, 15, 12, 0, tzinfo=ZoneInfo("America/Detroit"))
F8_RUN = BASE / "xp_f8_legacy_replay_20261029T065122Z"
F8_MANIFEST = BASE / "xp_f8_legacy_preflight_dev80seedfix_20261029T064359Z/XP_F8_REPLAY_INPUTS_GT_FREE.json"
F8_MANIFEST_SHA = "0929a94537c30faa893346a000c6cca5b51d5f45b48fbc2dfb0f6e64fa2537b9"
DAPS_HEAD = "e7a77d094167084faed19b599b96673b7bb11447"
PREFERRED_REPLAY_GPUS = (1, 0, 2, 3)
PROJECT_LOCK = BASE / ".xpk_a49_gpu_admission.lock"
ACTIVE_CHILDREN = {}

def gate(ok, why):
    if not ok:
        raise RuntimeError("STOP_A49_" + why)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def read(path):
    p = Path(path)
    gate(p.is_file() and not p.is_symlink(), "MISSING_FILE_" + str(p))
    return json.loads(p.read_text())

def atomic_json(path, obj):
    p = Path(path)
    tmp = p.with_name("." + p.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)

def git(*args, timeout=120):
    return subprocess.check_output(
        ["git", "-C", str(WT), *args], text=True, timeout=timeout
    ).strip()

def cpu_env():
    e = os.environ.copy()
    e.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(WT),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        NUMEXPR_NUM_THREADS="4",
    )
    return e

def gpu_env(gpu):
    e = cpu_env()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return e

def now_et():
    return datetime.datetime.now(ZoneInfo("America/Detroit"))

def deadline_gate():
    gate(now_et() < DEADLINE, "CP4_DEADLINE")

def cleanup_active_children():
    """Best-effort cleanup of only controller-owned children and journal ENDs."""
    if not ACTIVE_CHILDREN:
        return
    try:
        from scripts.track_r import xp_a48_xpk_event_journal_cpu as j
    except Exception:
        j = None
    for wid, v in list(ACTIVE_CHILDREN.items()):
        p = v.get("proc")
        try:
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=10)
        except Exception:
            pass
        try:
            fh = v.get("fh")
            if fh is not None and not fh.closed:
                fh.close()
        except Exception:
            pass
        try:
            if j is not None:
                j.mutate(Path(v["run"]) / "XPK_GPU_EVENTS.jsonl", "END", wid, int(v["gpu"]))
        except Exception:
            pass
        ACTIVE_CHILDREN.pop(wid, None)

atexit.register(cleanup_active_children)

def _signal_stop(signum, frame):
    cleanup_active_children()
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, _signal_stop)
signal.signal(signal.SIGINT, _signal_stop)

def run_cpu(args, log, timeout=None):
    deadline_gate()
    with Path(log).open("a") as f:
        f.write("\n$ " + " ".join(shlex.quote(str(x)) for x in args) + "\n")
        f.flush()
        p = subprocess.run(
            [str(x) for x in args],
            cwd=str(WT),
            env=cpu_env(),
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return p.returncode

def ensure_code_pin(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CONTROLLER_MUST_BE_CPU")
    gate(git("rev-parse", "HEAD") == a.expected_head, "HEAD_PIN")
    gate(not git("status", "--porcelain"), "DIRTY_WORKTREE")
    remote = subprocess.check_output(
        ["git", "-C", str(WT), "ls-remote", "origin", "refs/heads/" + a.branch],
        text=True,
        timeout=120,
    ).strip().split()
    gate(remote and remote[0] == a.expected_head, "REMOTE_BRANCH_MOVED")
    gate(CPU_PY.is_file(), "CPU_ENV")
    for p in PYTHON.values():
        gate(p.is_file(), "GPU_ENV_" + str(p))

def state_path(run):
    return run / "XPK_A49_CONTROLLER_STATE.json"

def update_state(run, **kw):
    p = state_path(run)
    d = read(p) if p.exists() else {}
    d.update(kw, updated_at_et=now_et().isoformat())
    atomic_json(p, d)

def manifest(run):
    return read(run / "XPK_GPU_GT_FREE_MANIFEST.json")

def manifest_sha(run):
    return sha(run / "XPK_GPU_GT_FREE_MANIFEST.json")

def family_ids(m, family, phase):
    key = "historical_replay_rows" if phase == "historical_replay" else "calibrated_rows"
    return sorted({str(x["image_id"]) for x in m[key][family]})

def group_complete(run, family, phase, image_id):
    p = run / "gpu" / family / phase / str(image_id) / "XPK_GROUP_COMPLETE.json"
    if not p.is_file():
        return False
    d = read(p)
    expected_n = 2 if phase == "historical_replay" else 4
    return (
        d.get("status") == "XPK_NATIVE_IMAGE_GROUP_PASS"
        and d.get("family") == family
        and d.get("phase") == phase
        and str(d.get("image_id")) == str(image_id)
        and d.get("n_roots") == expected_n
    )

def replay_score_pass(run, family, head):
    p = run / "score" / f"{family}_REPLAY_A4_RECEIPT.json"
    if not p.is_file():
        return False
    d = read(p)
    if not (
        d.get("status") == "XP_K_A23_5B_ORIGINAL_TWO_ROOT_IDENTITY_REPLAY_PASS"
        and d.get("family") == family
        and d.get("source_head") == head
        and d.get("passed") is True
        and d.get("n_root") == 2
        and d.get("original_A4_PSNR_tolerance_db") == 0.1
        and d.get("original_A4_label_must_match") is True
        and d.get("strict_thresholds_unchanged") is True
    ):
        return False
    rows = d.get("rows")
    return (
        isinstance(rows, list)
        and len(rows) == 2
        and all(
            r.get("pass_A4") is True
            and r.get("computed_native_class") == r.get("frozen_historical_class")
            and isinstance(r.get("abs_raw_psnr_delta_db"), (int, float))
            and math.isfinite(r["abs_raw_psnr_delta_db"])
            and 0 <= r["abs_raw_psnr_delta_db"] <= 0.1
            for r in rows
        )
    )

def calibration_score_pass(run, family, head):
    p = run / "score" / f"{family}_CAL_AGGREGATE.json"
    if not p.is_file():
        return False
    d = read(p)
    return (
        d.get("status") == "XP_K_A24_CALIBRATED_100x4_CPU_SCORE_COMPLETE"
        and d.get("family") == family
        and d.get("source_head") == head
        and d.get("n_images") == 100
        and d.get("n_candidates") == 400
    )

def adapter_static_gate():
    from scripts.track_r import xp_a41_xpk_native_gpu as n
    gate(n.ORIGINAL == {"SITCOM": .01, "FAST": .05, "DAPS": .01}, "ADAPTER_ORIGINALS")
    gate(
        math.isclose(n.TARGET["SITCOM"], .05, rel_tol=0, abs_tol=1e-12)
        and math.isclose(n.TARGET["FAST"], 33.26, rel_tol=0, abs_tol=1e-12)
        and math.isclose(n.TARGET["DAPS"], math.sqrt(2) * .05, rel_tol=0, abs_tol=1e-12),
        "ADAPTER_TARGETS",
    )
    src = (WT / "scripts/track_r/xp_a41_xpk_native_gpu.py").read_text()
    for fragment in (
        'if a.phase=="calibrated":cfg["lgvd_tau"]=TARGET["SITCOM"]',
        'if a.phase=="calibrated":conf["epsilon"]=TARGET["FAST"]',
        'cmd.append("task.pixel.mcmc_sampler_config.tau="+repr(TARGET["DAPS"]))',
        '"none" if phase=="historical_replay"',
        '"SITCOM":"lgvd_tau"',
        '"FAST":"epsilon"',
        '"DAPS":"task.pixel.mcmc_sampler_config.tau"',
        "native_model_setting_except_scalar_identical=True",
        "no_new_selector=True",
    ):
        gate(fragment in src, "ADAPTER_STATIC_DIFF_" + fragment[:24])
    return {
        "status": "A49_COMMITTED_ONE_PARAMETER_ADAPTER_DIFF_PASS",
        "worker_sha256": sha(WT / "scripts/track_r/xp_a41_xpk_native_gpu.py"),
        "original": n.ORIGINAL,
        "target": n.TARGET,
    }

def ensure_f8_gate(run, head):
    out = run / "XPK_DAPS_F8_GATE_RECEIPT.json"
    if out.exists():
        d = read(out)
        gate(
            d.get("status") == "XPK_DAPS_F8_8_OF_8_EXACT_VERIFIED"
            and d.get("n_exact") == 8
            and d.get("verified") is True
            and d.get("historical_native_source_head") == DAPS_HEAD
            and d.get("prior_f8_manifest_sha256") == F8_MANIFEST_SHA,
            "F8_EXISTING_RECEIPT_DRIFT",
        )
        return d
    gate(sha(F8_MANIFEST) == F8_MANIFEST_SHA, "F8_MANIFEST_SHA")
    w1 = read(F8_RUN / "XP_F8_LEGACY_REPLAY_WAVE1.json")
    w2 = read(F8_RUN / "XP_F8_LEGACY_REPLAY_WAVE2.json")
    off = read(F8_RUN / "XP_F8_LEGACY_REPLAY_WAVE2_OFFLINE.json")
    gate(
        w1.get("status") == "XP_F8_LEGACY_WAVE1_GPU_PASS"
        and w1.get("accepted_indices") == [0, 1, 2, 3]
        and w2.get("status") == "XP_F8_LEGACY_WAVE2_GPU_PASS"
        and w2.get("accepted_indices") == [4, 5, 6, 7],
        "F8_GPU_WAVES",
    )
    gate(
        off.get("status") == "XP_F8_LEGACY_WAVE2_OFFLINE_PASS"
        and off.get("n_checked") == 8
        and off.get("n_pass") == 8
        and off.get("n_pixel_identical") == 8
        and len(off.get("comparisons", [])) == 8
        and all(x.get("replay_pass") and x.get("pixel_identical") for x in off["comparisons"]),
        "F8_OFFLINE_EXACT_8",
    )
    roots = []
    for i in range(8):
        p = F8_RUN / f"replay_{i:02d}/XP_F8_LEGACY_REPLAY_NATIVE_ROOT.json"
        rr = read(p)
        gate(
            rr.get("status") == "XP_F8_LEGACY_REPLAY_NATIVE_ROOT_PASS"
            and rr.get("index") == i
            and rr.get("source_head") == DAPS_HEAD
            and rr.get("replay_input_manifest_sha256") == F8_MANIFEST_SHA,
            "F8_ROOT_" + str(i),
        )
        roots.append(sha(p))
    d = {
        "status": "XPK_DAPS_F8_8_OF_8_EXACT_VERIFIED",
        "verified": True,
        "n_exact": 8,
        "prior_f8_manifest_sha256": F8_MANIFEST_SHA,
        "historical_native_source_head": DAPS_HEAD,
        "prior_run": str(F8_RUN),
        "wave1_receipt_sha256": sha(F8_RUN / "XP_F8_LEGACY_REPLAY_WAVE1.json"),
        "wave2_receipt_sha256": sha(F8_RUN / "XP_F8_LEGACY_REPLAY_WAVE2.json"),
        "offline_8root_receipt_sha256": sha(F8_RUN / "XP_F8_LEGACY_REPLAY_WAVE2_OFFLINE.json"),
        "native_root_receipt_sha256": roots,
        "source_head": head,
        "no_new_gpu_work": True,
    }
    with out.open("x") as f:
        json.dump(d, f, sort_keys=True, indent=2)
        f.write("\n")
    return d

def gpu_snapshot():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=False,
    )
    gate(q.returncode == 0, "NVIDIA_SMI_GPU")
    cards, by_uuid = [], {}
    for line in q.stdout.splitlines():
        if not line.strip():
            continue
        fs = [x.strip() for x in line.split(",")]
        gate(len(fs) == 5, "GPU_CSV")
        idx, uid, total, used, util = fs
        d = {"index": int(idx), "uuid": uid, "total_mib": int(total),
             "used_mib": int(used), "util_pct": int(util), "foreign_pids": []}
        cards.append(d)
        by_uuid[uid] = d
    p = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if p.returncode == 0:
        for line in p.stdout.splitlines():
            fs = [x.strip() for x in line.split(",")]
            if len(fs) != 3 or fs[0] not in by_uuid:
                continue
            try:
                by_uuid[fs[0]]["foreign_pids"].append(
                    {"pid": int(fs[1]), "used_mib": int(fs[2])}
                )
            except ValueError:
                pass
    gate(cards and len({c["index"] for c in cards}) == len(cards), "GPU_INVENTORY")
    return cards

def eligible_cards(cards, family, occupied, preferred=False):
    reserve = RESERVE_GIB[family] * 1024
    out = []
    for c in cards:
        projected = c["used_mib"] + reserve
        ok = (
            c["index"] not in occupied
            and projected <= 0.80 * c["total_mib"]
            and c["total_mib"] - projected >= 4 * 1024
        )
        if ok:
            x = dict(c)
            x["projected_used_mib"] = projected
            out.append(x)
    if preferred:
        rank = {g: i for i, g in enumerate(PREFERRED_REPLAY_GPUS)}
        out.sort(key=lambda c: (rank.get(c["index"], 99), c["util_pct"],
                                -(c["total_mib"] - c["used_mib"])))
    else:
        out.sort(key=lambda c: (c["util_pct"],
                                -(c["total_mib"] - c["used_mib"]), c["index"]))
    return out

def journal_modules():
    from scripts.track_r import xp_a48_xpk_event_journal_cpu as j
    from scripts.track_r import xp_a47_xpk_gpu_ledger_cpu as l
    return j, l

def init_journal(run):
    j, _ = journal_modules()
    p = run / "XPK_GPU_EVENTS.jsonl"
    if not p.exists():
        j.initialize(p)
    else:
        j.read_events(p)
    return p

def ledger(run, a, reserve_h=0.0):
    j, l = journal_modules()
    events = j.read_events(run / "XPK_GPU_EVENTS.jsonl")
    return l.remaining(events, time.time(), a.prior_project_h, a.other_xpk_h,
                       reserve_h=reserve_h)

def recover_unfinished(run, a):
    j, l = journal_modules()
    jp = run / "XPK_GPU_EVENTS.jsonl"
    events = j.read_events(jp)
    d = l.charge(events, time.time())
    if not d["unfinished_workers"]:
        return
    state = read(state_path(run))
    active = state.get("active_workers", {})
    starts = {e["id"]: e for e in events if e["kind"] == "START"}
    for wid in d["unfinished_workers"]:
        meta = active.get(wid, {})
        pid = meta.get("pid")
        if isinstance(pid, int) and (Path("/proc") / str(pid) / "cmdline").is_file():
            gate(False, "RECOVERY_OWN_WORKER_STILL_ALIVE_" + str(pid))
        fam = meta.get("family") or starts[wid].get("family")
        phase = meta.get("phase") or starts[wid].get("phase")
        image_id = meta.get("image_id")
        gate(image_id is not None, "RECOVERY_MISSING_IMAGE_ID")
        actual_phase = "calibrated" if phase == "pilot" else phase
        gate(group_complete(run, fam, actual_phase, image_id),
             "RECOVERY_PARTIAL_NATIVE_GROUP_" + wid)
        j.mutate(jp, "END", wid, starts[wid]["gpu"])
        active.pop(wid, None)
    update_state(run, active_workers=active, recovery_closed_complete_groups=True)

def launch_worker(run, a, family, phase, image_id, gpu, log):
    j, _ = journal_modules()
    jp = run / "XPK_GPU_EVENTS.jsonl"
    pilot_id = family_ids(manifest(run), family, "calibrated")[0]
    journal_phase = "pilot" if phase == "calibrated" and str(image_id) == str(pilot_id) else phase
    wid = f"{family}-{journal_phase}-{image_id}-{uuid.uuid4().hex[:12]}"
    j.mutate(jp, "START", wid, gpu, family=family, phase=journal_phase)
    cmd = [
        str(PYTHON[family]), "-u", "-m", "scripts.track_r.xp_a41_xpk_native_gpu",
        "--expected-head", a.expected_head, "--run", str(run),
        "--manifest-sha", manifest_sha(run), "--family", family,
        "--phase", phase, "--image-id", str(image_id), "--gpu", str(gpu),
    ]
    fh = Path(log).open("a")
    fh.write("\n$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
    fh.flush()
    try:
        p = subprocess.Popen(cmd, cwd=str(WT), env=gpu_env(gpu),
                             stdout=fh, stderr=subprocess.STDOUT)
    except BaseException:
        fh.close()
        j.mutate(jp, "END", wid, gpu)
        raise
    ACTIVE_CHILDREN[wid] = {"proc": p, "fh": fh, "gpu": gpu, "run": str(run)}
    st = read(state_path(run))
    aw = st.get("active_workers", {})
    aw[wid] = {"pid": p.pid, "gpu": gpu, "family": family, "phase": phase,
               "image_id": str(image_id), "log": str(log)}
    update_state(run, active_workers=aw)
    return wid, p, fh

def finish_worker(run, wid, p, fh, gpu):
    j, _ = journal_modules()
    rc = p.wait()
    fh.close()
    j.mutate(run / "XPK_GPU_EVENTS.jsonl", "END", wid, gpu)
    st = read(state_path(run))
    aw = st.get("active_workers", {})
    aw.pop(wid, None)
    update_state(run, active_workers=aw)
    ACTIVE_CHILDREN.pop(wid, None)
    return rc

def group_reserve_h(run, family, phase):
    if phase == "historical_replay":
        return 2.0
    m = manifest(run)
    pilot_id = family_ids(m, family, "calibrated")[0]
    p = run / "gpu" / family / "calibrated" / pilot_id
    if group_complete(run, family, "calibrated", pilot_id):
        sec = sum(float(read(p / f"run{i:02d}/XPK_ROOT_COMPLETE.json")
                        ["elapsed_gpu_occupied_wall_s"]) for i in range(4))
        gate(sec > 0 and math.isfinite(sec), "GROUP_RESERVE_PILOT_" + family)
        return max(0.01, 1.25 * sec / 3600.0)
    return 2.0

def enforce_live_budget(run, a, active):
    if not active:
        return
    j, l = journal_modules()
    events = j.read_events(run / "XPK_GPU_EVENTS.jsonl")
    charged = l.charge(events, time.time())
    xpk = a.other_xpk_h + charged["physical_gpu_h"]
    project = a.prior_project_h + charged["physical_gpu_h"]
    if xpk < 30.0 and project < 400.0:
        return
    for v in active.values():
        if v["proc"].poll() is None:
            v["proc"].terminate()
    time.sleep(2)
    for v in active.values():
        if v["proc"].poll() is None:
            v["proc"].kill()
    raise RuntimeError("STOP_A49_GPU_BUDGET_LIVE_CAP")

def run_task_queue(run, a, tasks, stage, preferred=False):
    pending = [t for t in tasks if not group_complete(run, *t)]
    active, wait_count = {}, 0
    while pending or active:
        deadline_gate()
        enforce_live_budget(run, a, active)
        for wid, v in list(active.items()):
            if v["proc"].poll() is None:
                continue
            rc = finish_worker(run, wid, v["proc"], v["fh"], v["gpu"])
            active.pop(wid)
            gate(rc == 0, "NATIVE_WORKER_FAILED_" + v["family"] + "_" + str(v["image_id"]))
            gate(group_complete(run, v["family"], v["phase"], v["image_id"]),
                 "MISSING_GROUP_RECEIPT")
        if not pending:
            if active:
                time.sleep(5)
            continue
        occupied = {v["gpu"] for v in active.values()}
        admitted_any = False
        with PROJECT_LOCK.open("a+") as lk:
            fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
            cards = gpu_snapshot()
            while pending and len(active) < a.max_gpus:
                family, phase, image_id = pending[0]
                choices = eligible_cards(cards, family, occupied, preferred=preferred)
                if not choices:
                    break
                reserve_h = group_reserve_h(run, family, phase)
                led = ledger(run, a, reserve_h=reserve_h)
                gate(led["can_admit_full_reserved_stage"], "GPU_BUDGET_EXHAUSTED")
                gpu = choices[0]["index"]
                log = run / "logs" / f"{stage}_{family}_{phase}_{image_id}.log"
                wid, proc, fh = launch_worker(run, a, family, phase, image_id, gpu, log)
                active[wid] = {"proc": proc, "fh": fh, "gpu": gpu,
                               "family": family, "phase": phase, "image_id": str(image_id)}
                pending.pop(0)
                occupied.add(gpu)
                admitted_any = True
                update_state(run, status="RUNNING_" + stage,
                             last_gpu_snapshot=cards, pending_tasks=len(pending),
                             active_count=len(active))
                cards = gpu_snapshot()
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)
        if not admitted_any and pending:
            wait_count += 1
            update_state(run, status="WAIT_SAFE_GPU", waiting_stage=stage,
                         pending_tasks=len(pending), active_count=len(active),
                         wait_count=wait_count, last_gpu_snapshot=gpu_snapshot())
            time.sleep(a.wait_seconds)
        elif active:
            time.sleep(5)

def score_replay(run, a, family):
    if replay_score_pass(run, family, a.expected_head):
        return
    rc = run_cpu(
        [CPU_PY, "-u", "-m", "scripts.track_r.xp_a41_xpk_score_cpu",
         "--run", run, "--expected-head", a.expected_head,
         "--family", family, "--phase", "replay"],
        run / "logs" / f"score_replay_{family}.log", timeout=7200)
    gate(rc == 0, "STOP_ASK_A23_5B_GATE_" + family)
    gate(replay_score_pass(run, family, a.expected_head),
         "REPLAY_SCORE_INVALID_" + family)

def score_calibration(run, a, family):
    if calibration_score_pass(run, family, a.expected_head):
        return
    rc = run_cpu(
        [CPU_PY, "-u", "-m", "scripts.track_r.xp_a41_xpk_score_cpu",
         "--run", run, "--expected-head", a.expected_head,
         "--family", family, "--phase", "calibration"],
        run / "logs" / f"score_calibration_{family}.log", timeout=14400)
    gate(rc == 0, "CALIBRATION_SCORE_" + family)
    gate(calibration_score_pass(run, family, a.expected_head),
         "CALIBRATION_SCORE_INVALID_" + family)

def pilot_projection(run, a):
    m, estimates, projected_remaining = manifest(run), {}, 0.0
    for family in FAMILIES:
        iid = family_ids(m, family, "calibrated")[0]
        p = run / "gpu" / family / "calibrated" / iid
        gate(group_complete(run, family, "calibrated", iid), "PILOT_GROUP_" + family)
        roots = [read(p / f"run{i:02d}/XPK_ROOT_COMPLETE.json") for i in range(4)]
        sec = sum(float(x["elapsed_gpu_occupied_wall_s"]) for x in roots)
        gate(sec > 0 and math.isfinite(sec), "PILOT_TIME_" + family)
        estimate_h = sec * 99 / 3600.0
        estimates[family] = {"pilot_image_id": iid,
                             "pilot_root_gpu_seconds_sum": sec,
                             "remaining_99_images_projected_gpu_h": estimate_h}
        projected_remaining += estimate_h
    led = ledger(run, a, reserve_h=projected_remaining)
    gate(led["can_admit_full_reserved_stage"], "PROJECTED_REMAINING_PANEL_BUDGET")
    d = {"status": "A49_MEASURED_PILOTS_AUTHORIZE_FULL_REGISTERED_PANELS",
         "families": estimates, "projected_remaining_gpu_h": projected_remaining,
         "ledger": led,
         "projection_rule": "sum of four native root occupied seconds in each family pilot x remaining 99 images",
         "no_panel_reduction": True}
    p = run / "XPK_A49_PILOT_BUDGET_GATE.json"
    if p.exists():
        gate(read(p) == d, "PILOT_GATE_DRIFT")
    else:
        with p.open("x") as f:
            json.dump(d, f, sort_keys=True, indent=2)
            f.write("\n")
    return d

def make_cp4_report(run, a):
    scores = {f: read(run / "score" / f"{f}_CAL_AGGREGATE.json") for f in FAMILIES}
    led = ledger(run, a)
    gate(not led["unfinished_workers"], "UNFINISHED_WORKERS_AT_CP4")
    rows = []
    for f in FAMILIES:
        s = scores[f]
        rows.append({"family": f,
                     "GOOD_rate": s["registered_class_rates"]["GOOD"],
                     "E1_CLIP_Good25_rate": s["E1_CLIP_Good25_rate"],
                     "O_orbit_Good25_rate": s["O_orbit_Good25_rate"],
                     "H_K1_rho_sigma_median_shift": s["H_K1_descriptive_GOOD_rho_sigma_median_shift"],
                     "H_K2_consistent_share_reduction": s["H_K2_SITCOM_consistent_share_reduction"],
                     "score_sha256": sha(run / "score" / f"{f}_CAL_AGGREGATE.json")})
    report = {"status": "XP_K_CP4_ALL_THREE_FAMILIES_COMPLETE_READY_TO_PUBLISH",
              "source_head": a.expected_head, "run": str(run),
              "manifest_sha256": manifest_sha(run), "families": rows,
              "gpu_ledger": led, "xpk_cap_h": 30.0, "project_cap_h": 400.0,
              "deadline_et": DEADLINE.isoformat(), "not_run_inventory": [],
              "science_gates": {f: sha(run / "score" / f"{f}_REPLAY_A4_RECEIPT.json")
                                for f in FAMILIES},
              "daps_f8_gate_sha256": sha(run / "XPK_DAPS_F8_GATE_RECEIPT.json"),
              "pilot_budget_gate_sha256": sha(run / "XPK_A49_PILOT_BUDGET_GATE.json"),
              "main_paper_cut_closed": True}
    p = run / "XPK_CP4_FINAL_LEDGER.json"
    if p.exists():
        gate(read(p) == report, "CP4_LEDGER_DRIFT")
    else:
        with p.open("x") as f:
            json.dump(report, f, sort_keys=True, indent=2)
            f.write("\n")
    return report

def publication_markdown(report):
    lines = [
        "# XP-K CP4 final return", "",
        f"- Source commit: {report['source_head']}",
        f"- Run: {report['run']}",
        f"- Manifest SHA256: {report['manifest_sha256']}",
        f"- XP-K charged GPU hours: {report['gpu_ledger']['xpk_total_h']:.6f} / 30",
        f"- Project charged GPU hours: {report['gpu_ledger']['project_total_h']:.6f} / 400",
        "- All three A4 two-root identity replay gates: PASS.",
        "- DAPS independent historical F8 8/8 pixel-exact lineage gate: PASS.",
        "- Full registered panel: 100 FFHQ B22 images x 4 calibrated roots for each family.",
        "",
        "| Family | GOOD rate | E1-CLIP Good25 | O-orbit Good25 | H-K1 rho/sigma median shift |",
        "|---|---:|---:|---:|---:|",
    ]
    for rr in report["families"]:
        hk1 = rr["H_K1_rho_sigma_median_shift"]
        h = "NA" if hk1 is None else f"{hk1:.6g}"
        lines.append(f"| {rr['family']} | {rr['GOOD_rate']:.4f} | "
                     f"{rr['E1_CLIP_Good25_rate']:.4f} | "
                     f"{rr['O_orbit_Good25_rate']:.4f} | {h} |")
    lines += ["",
        "Mechanism comparisons use the frozen pre-XP-K v1/v1.1 forecasts embedded in the CPU scoring receipt.",
        "No tolerance, seed domain, selector, solver schedule, measurement, or panel size was changed after observing XP-K outcomes.",
        "", "Terminal status: XP_K_CP4_ALL_THREE_FAMILIES_PUBLISHED.", ""]
    return "\n".join(lines)

def publish(run, a, report):
    gate(git("rev-parse", "HEAD") == a.expected_head, "PREPUBLICATION_HEAD")
    gate(not git("status", "--porcelain"), "PREPUBLICATION_DIRTY")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rel = Path("docs/track_r/executors/xp/returns") / f"XP_K_CP4_A49_{stamp}.md"
    dst = WT / rel
    gate(not dst.exists(), "PUBLICATION_PATH_EXISTS")
    dst.write_text(publication_markdown(report))
    subprocess.check_call(["git", "-C", str(WT), "add", str(rel)], timeout=60)
    subprocess.check_call(
        ["git", "-C", str(WT), "commit", "-m",
         "results(track-r): publish XP-K CP4 A49 final return"], timeout=120)
    pub = git("rev-parse", "HEAD")
    gate(pub != a.expected_head, "PUBLICATION_COMMIT_NOT_CREATED")
    atomic_json(run / "XPK_PUBLICATION_COMMIT.json",
                {"status": "XPK_PUBLICATION_COMMIT_CREATED",
                 "source_head": a.expected_head, "publication_head": pub,
                 "path": str(rel)})
    while True:
        deadline_gate()
        remote = subprocess.check_output(
            ["git", "-C", str(WT), "ls-remote", "origin", "refs/heads/" + a.branch],
            text=True, timeout=120).strip().split()
        if remote and remote[0] == pub:
            return pub, rel
        gate(remote and remote[0] == a.expected_head,
             "REMOTE_BRANCH_MOVED_DURING_CAMPAIGN")
        p = subprocess.run(
            ["git", "-C", str(WT), "push", "origin",
             f"{pub}:refs/heads/{a.branch}"],
            capture_output=True, text=True, timeout=300, check=False)
        if p.returncode == 0:
            continue
        update_state(run, status="PUBLICATION_NETWORK_RETRY",
                     push_stderr=p.stderr[-1000:])
        time.sleep(a.push_retry_seconds)

def ensure_preflight(run, a):
    test_log = run.parent / (run.name + "_A49_PRE_MANIFEST_CPU.log")
    tests = [
        "tests.test_track_r_xp_a41_native_entry_contract",
        "tests.test_track_r_xp_a42_gate_interlock_cpu",
        "tests.test_track_r_xp_a45_xpk_gate_audit",
        "tests.test_track_r_xp_a46_xpk_gpu_admission",
        "tests.test_track_r_xp_a47_xpk_gpu_ledger",
        "tests.test_track_r_xp_a48_xpk_event_journal",
        "tests.test_track_r_xp_a49_one_shot_contract",
    ]
    if not run.exists():
        rc = run_cpu([CPU_PY, "-m", "unittest", "-v", *tests],
                     test_log, timeout=1800)
        gate(rc == 0, "CPU_REGRESSION")
        binding = run.parent / ("xp_a40_xpk_binding_" +
                                run.name.removeprefix("xp_a41_xpk_"))
        if not binding.exists():
            rc = run_cpu(
                [CPU_PY, "-u", "-m",
                 "scripts.track_r.xp_a40_xpk_native_binding_cpu",
                 "--expected-head", a.expected_head, "--out", binding],
                test_log, timeout=1200)
            gate(rc == 0, "A40_NATIVE_BINDING")
        b = read(binding / "A40_NATIVE_BINDING_CPU_RECEIPT.json")
        gate(b.get("status") == "A40_ALL3_NATIVE_SOURCE_BINDINGS_PASS",
             "A40_RECEIPT")
        adapter = adapter_static_gate()
        rc = run_cpu(
            [CPU_PY, "-u", "-m", "scripts.track_r.xp_a41_xpk_manifest_cpu",
             "--expected-head", a.expected_head, "--out", run],
            test_log, timeout=1200)
        gate(rc == 0, "MANIFEST_FREEZE")
        (run / "gpu").mkdir()
        (run / "score").mkdir()
        (run / "logs").mkdir()
        atomic_json(state_path(run),
            {"status": "A49_PREPARED", "source_head": a.expected_head,
             "branch": a.branch,
             "binding_receipt_sha256": sha(binding / "A40_NATIVE_BINDING_CPU_RECEIPT.json"),
             "adapter_gate": adapter, "active_workers": {}})
    else:
        gate(run.is_dir() and not run.is_symlink(), "RUN_ROOT")
        st = read(state_path(run))
        gate(st.get("source_head") == a.expected_head and st.get("branch") == a.branch,
             "RESUME_PIN")
    init_journal(run)
    recover_unfinished(run, a)

def main(a):
    ensure_code_pin(a)
    deadline_gate()
    run = Path(a.run)
    gate(run.parent == BASE and run.name.startswith("xp_a41_xpk_"),
         "RUN_NAMESPACE")
    PROJECT_LOCK.touch(exist_ok=True)
    gate(PROJECT_LOCK.is_file() and not PROJECT_LOCK.is_symlink(),
         "PROJECT_LOCK")
    ensure_preflight(run, a)
    m = manifest(run)
    gate(m.get("head") == a.expected_head and
         m.get("families") == list(FAMILIES), "MANIFEST_HEAD")

    replay_tasks = []
    for family in FAMILIES:
        ids = family_ids(m, family, "historical_replay")
        gate(len(ids) == 1, "REPLAY_ONE_IMAGE_" + family)
        replay_tasks.append((family, "historical_replay", ids[0]))
    run_task_queue(run, a, replay_tasks, "IDENTITY_REPLAY", preferred=True)
    for family in FAMILIES:
        score_replay(run, a, family)
    ensure_f8_gate(run, a.expected_head)

    audit = run / "XPK_A45_AUDIT_POST_REPLAY.json"
    if not audit.exists():
        rc = run_cpu(
            [CPU_PY, "-u", "-m", "scripts.track_r.xp_a45_xpk_gate_audit_cpu",
             "--run", run, "--expected-head", a.expected_head, "--out", audit],
            run / "logs" / "a45_post_replay.log", timeout=1200)
        gate(rc == 0, "A45_POST_REPLAY")

    pilot_tasks = [(family, "calibrated",
                    family_ids(m, family, "calibrated")[0])
                   for family in FAMILIES]
    run_task_queue(run, a, pilot_tasks, "CALIBRATION_PILOT", preferred=True)
    pilot_projection(run, a)

    full = []
    for family in FAMILIES:
        for iid in family_ids(m, family, "calibrated"):
            full.append((family, "calibrated", iid))
    run_task_queue(run, a, full, "CALIBRATION_FULL", preferred=False)

    for family in FAMILIES:
        score_calibration(run, a, family)

    final_audit = run / "XPK_A45_AUDIT_FINAL.json"
    if not final_audit.exists():
        rc = run_cpu(
            [CPU_PY, "-u", "-m", "scripts.track_r.xp_a45_xpk_gate_audit_cpu",
             "--run", run, "--expected-head", a.expected_head,
             "--out", final_audit],
            run / "logs" / "a45_final.log", timeout=1200)
        gate(rc == 0, "A45_FINAL")
    d = read(final_audit)
    gate(d.get("all_replay_gates_pass") is True and
         d.get("all_calibrated_panels_complete") is True,
         "FINAL_AUDIT_NOT_COMPLETE")

    report = make_cp4_report(run, a)
    update_state(run, status="CP4_READY_TO_PUBLISH",
                 cp4_ledger_sha256=sha(run / "XPK_CP4_FINAL_LEDGER.json"))
    pub, rel = publish(run, a, report)
    terminal = {"status": "XP_K_CP4_ALL_THREE_FAMILIES_PUBLISHED",
                "source_head": a.expected_head, "publication_head": pub,
                "publication_path": str(rel), "run": str(run),
                "cp4_ledger_sha256": sha(run / "XPK_CP4_FINAL_LEDGER.json")}
    atomic_json(run / "XPK_A49_TERMINAL.json", terminal)
    update_state(run, **terminal)
    print(json.dumps(terminal, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--expected-head", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--prior-project-h", type=float, required=True)
    p.add_argument("--other-xpk-h", type=float, default=0.0)
    p.add_argument("--max-gpus", type=int, default=2)
    p.add_argument("--wait-seconds", type=int, default=60)
    p.add_argument("--push-retry-seconds", type=int, default=120)
    try:
        a = p.parse_args()
        gate(1 <= a.max_gpus <= 4, "MAX_GPUS")
        gate(a.prior_project_h >= 0 and a.other_xpk_h >= 0, "BUDGET_ARGS")
        sys.exit(main(a))
    except Exception as exc:
        traceback.print_exc()
        try:
            r = Path(a.run)
            if r.exists():
                terminal = {
                    "status": ("STOP_ASK_A23_5B_GATE"
                               if "A23_5B" in str(exc) or "SCIENCE" in str(exc)
                                  or "REPLAY" in str(exc)
                               else "STOP_BUDGET_OR_DEADLINE_WITH_RECEIPTS"
                               if "BUDGET" in str(exc) or "DEADLINE" in str(exc)
                               else "ENGINEERING_STOP_WITH_RECEIPTS"),
                    "error": repr(exc), "at_et": now_et().isoformat()}
                atomic_json(r / "XPK_A49_STOP.json", terminal)
                update_state(r, **terminal)
        except Exception:
            pass
        print("ACTION_NEEDED_XPK_A49", repr(exc), flush=True)
        sys.exit(8)
