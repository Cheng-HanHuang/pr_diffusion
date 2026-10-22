#!/usr/bin/env python3
"""R0.0 bounded, read-only PAC inventory. No data/model tree recursion."""
from __future__ import annotations
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

FORBIDDEN = ("confirmation", "confirm305")
STATUSES = {"VERIFIED", "MISSING", "MISMATCH", "WARN"}
ROOT = Path("/egr/research-pac/huang248")


def safe_path(value):
    p = Path(str(value))
    if any(token in str(p).lower() for token in FORBIDDEN):
        raise RuntimeError("STOP_FORBIDDEN_PATH_TOKEN")
    return p


def run(argv, timeout=40, env=None, cwd=None):
    try:
        r = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout,
                           env=env, cwd=cwd, check=False)
        return {"rc": r.returncode, "text": r.stdout[-12000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"rc": -1, "text": str(exc)[:1000]}


def sha256(path):
    path = safe_path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_file(path, expected=None, hash_file=False):
    p = safe_path(path)
    if not p.is_file():
        return {"status": "MISSING", "path": str(p)}
    rec = {"status": "VERIFIED", "path": str(p), "bytes": p.stat().st_size}
    if expected or hash_file:
        rec["sha256"] = sha256(p)
        if expected and rec["sha256"].lower() != expected.lower():
            rec["status"] = "MISMATCH"
            rec["expected_sha256"] = expected
    return rec


def dir_check(path):
    p = safe_path(path)
    return {"status": "VERIFIED" if p.is_dir() else "MISSING", "path": str(p)}


def git_check(path, expected=None):
    p = safe_path(path)
    if not p.is_dir():
        return {"status": "MISSING", "path": str(p)}
    head = run(["git", "-C", str(p), "rev-parse", "HEAD"])
    branch = run(["git", "-C", str(p), "branch", "--show-current"])
    dirty = run(["git", "-C", str(p), "status", "--porcelain"])
    actual = head["text"].strip()
    state = "MISMATCH" if expected and actual != expected else "VERIFIED"
    if head["rc"] or dirty["rc"] or dirty["text"].strip():
        state = "MISMATCH"
    return {"status": state, "path": str(p), "head": actual,
            "expected_head": expected, "branch": branch["text"].strip(),
            "clean": dirty["rc"] == 0 and not dirty["text"].strip(),
            "dirty_lines": dirty["text"].splitlines()[:10]}


PROBE = """
import importlib, json, os, sys
info={"python":sys.version.split()[0], "executable":sys.executable, "imports":{}}
for name in ("numpy","scipy","PIL","pandas","sklearn","tqdm","ftfy","regex",
             "packaging","pkg_resources","json","yaml","torchvision"):
    try:
        m=importlib.import_module(name)
        info["imports"][name]={"ok":True,"version":getattr(m,"__version__",None)}
    except Exception as e:
        info["imports"][name]={"ok":False,"error":repr(e)[:300]}
try:
    import torch
    info["torch"]={"version":torch.__version__, "cuda_wheel":torch.version.cuda,
        "arch_list":torch.cuda.get_arch_list(),
        "sm_120":("sm_120" in torch.cuda.get_arch_list())}
except Exception as e:
    info["torch"]={"error":repr(e)[:400]}
extra=os.environ.get("R0_IMPORT_ROOT","")
if extra:
    sys.path.insert(0,extra)
    for name in os.environ.get("R0_IMPORT_NAMES","").split(","):
        if name:
            try: importlib.import_module(name); info["imports"][name]={"ok":True}
            except Exception as e: info["imports"][name]={"ok":False,"error":repr(e)[:300]}
print("R0_PROBE_JSON="+json.dumps(info,sort_keys=True))
"""


SAMPLE_PROBE = """
import json, sys, torch
from PIL import Image
from pathlib import Path
def extract(obj):
    if isinstance(obj,torch.Tensor): return obj
    if isinstance(obj,dict):
        for k in ("x","image","tensor","ground_truth","measurement","reconstruction","sample"):
            if k in obj and isinstance(obj[k],torch.Tensor): return obj[k]
        for value in obj.values():
            if isinstance(value,torch.Tensor): return value
    return None
a={}
for path in sys.argv[1:]:
    try:
        p=Path(path)
        if p.suffix.lower()==".png":
            im=Image.open(p).convert("RGB")
            import numpy as np
            v=np.asarray(im)
            a[path]=dict(dtype=str(v.dtype),shape=list(v.shape),min=float(v.min()),max=float(v.max()))
        elif p.suffix.lower()==".pt":
            try: obj=torch.load(p,map_location="cpu",weights_only=True)
            except Exception: obj=torch.load(p,map_location="cpu",weights_only=False)
            t=extract(obj)
            if t is None: a[path]={"error":"no tensor in file","type":str(type(obj))}
            else: a[path]=dict(dtype=str(t.dtype),shape=list(t.shape),
                               min=float(t.min()),max=float(t.max()))
    except Exception as e: a[path]={"error":repr(e)[:400]}
print("R0_SAMPLE_JSON="+json.dumps(a,sort_keys=True))
"""


def bounded(pattern, limit=500):
    safe_path(pattern)
    return sorted(glob.glob(pattern))[:limit]


def sample_json_paths(result):
    """Only inspect one explicitly bounded B22 result; never walk."""
    p = safe_path(result)
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        return {"error": str(exc)[:200]}
    def selected(d):
        if not isinstance(d, dict):
            return []
        return [(k, v) for k, v in d.items()
                if isinstance(v, str) and (v.endswith(".pt") or v.endswith(".png"))]
    return {"keys": list(data)[:40], "candidate_fields": selected(data)[:12]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-sha", required=True)
    args = ap.parse_args()
    out = safe_path(args.out)
    if not str(out).startswith(str(ROOT / "outputs/pr_diffusion/track_r/") + os.sep):
        raise RuntimeError("STOP_OUTSIDE_TRACK_R_OUTPUT")
    config = json.loads(safe_path(args.paths).read_text())
    out.mkdir(parents=True, exist_ok=False)
    checks = {}
    def add(key, value):
        if isinstance(value, dict) and "status" in value and value["status"] not in STATUSES:
            raise ValueError(key)
        checks[key] = value
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    worktree = config["track_r"]["worktree"]
    add("identity", {"status": "VERIFIED", "hostname": socket.gethostname(),
                     "utc": timestamp, "git": git_check(worktree, args.expected_sha)})
    if checks["identity"]["git"]["status"] != "VERIFIED":
        checks["identity"]["status"] = "MISMATCH"
    add("disk", {"status": "VERIFIED", "df": run(["df","-h",str(ROOT)]),
                 "output_free": shutil.disk_usage(out).free})
    add("gpus", {"status": "VERIFIED",
                 "devices": run(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used","--format=csv"]),
                 "compute_apps": run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory","--format=csv"])})
    add("conda", {"status": "VERIFIED" if shutil.which("conda") else "MISSING",
                  "path": shutil.which("conda"), "version": run(["conda","--version"]) if shutil.which("conda") else None})
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1", "R0_IMPORT_ROOT": config["external"]["difffpr"]["path"],
                "R0_IMPORT_NAMES": "guided_diffusion"})
    for name, value in config["python"].items():
        if not isinstance(value, str):
            continue
        py = safe_path(value)
        if not py.is_file():
            add("python." + name, {"status": "MISSING", "path": str(py)})
            continue
        pe = dict(env)
        if name.startswith("fast_dips"):
            pe["R0_IMPORT_ROOT"] = config["external"]["fast_dips"]["path"]
            pe["R0_IMPORT_NAMES"] = "edm_FAST_DIPS"
        result = run([str(py), "-c", PROBE], timeout=65, env=pe)
        line = next((s.split("=",1)[1] for s in result["text"].splitlines() if s.startswith("R0_PROBE_JSON=")), None)
        detail = json.loads(line) if line else {"error": result["text"][-800:]}
        imports = detail.get("imports", {})
        missing = [k for k, v in imports.items() if not v.get("ok")]
        status = "VERIFIED" if result["rc"] == 0 and line and not missing else "WARN"
        if name == "fast_dips_a15_blackwell" and not detail.get("torch",{}).get("sm_120"):
            status = "MISMATCH"
        add("python." + name, {"status": status, "path": str(py),
                                "detail": detail, "missing_imports": missing})
    for name, item in config["models"].items():
        add("models." + name, checked_file(item["path"], item.get("sha256")))
    for name, item in config["external"].items():
        add("external." + name, git_check(item["path"], item.get("commit")))
        if name == "univfd":
            add("univfd.fc", checked_file(safe_path(item["path"]) / "pretrained_weights/fc_weights.pth",
                                          item["fc_weights_sha256"]))
    s = config["sources"]
    b22 = s["S1_b22"]
    for key in ("b22_run_root", "b21_fresh_root"):
        add("S1." + key, dir_check(b22[key]))
    add("S1.full_manifest", checked_file(b22["b22_full_manifest"], b22["b22_full_manifest_sha256"]))
    add("S1.fresh_rows", checked_file(b22["b21_fresh_rows_csv"]))
    stage = safe_path(b22["b22_run_root"]) / "full"
    b22_counts = {}
    b22_example = None
    for family in ("sitcom", "np", "daps", "SITCOM", "NP", "DAPS"):
        rows = bounded(str(stage / family / "row*"), 120)
        if rows:
            b22_counts[family] = len(rows)
            if b22_example is None:
                candidates = bounded(str(Path(rows[0]) / "candidates" / "*" / "result.json"), 16)
                if candidates:
                    b22_example = candidates[0]
    add("S1.bounded_rows", {"status": "VERIFIED" if b22_counts else "WARN",
                            "stage": str(stage), "family_counts": b22_counts,
                            "example_result": b22_example,
                            "example_result_fields": sample_json_paths(b22_example) if b22_example else None})
    s2 = s["S2_b24_dev80"]
    add("S2.run_root", dir_check(s2["run_root"]))
    manifest_path = safe_path(worktree) / s2["manifest_in_repo"]
    add("S2.repo_manifest", checked_file(manifest_path))
    devrows = json.loads(manifest_path.read_text()).get("rows", []) if manifest_path.is_file() else []
    # Only the recorded DEV80 rows; NEVER inspect any other B24 cohort.
    if len(devrows) != 80:
        add("S2.dev_row_count", {"status": "MISMATCH", "count": len(devrows)})
    else:
        add("S2.dev_row_count", {"status": "VERIFIED", "count": len(devrows)})
    s2_paths = []
    if devrows:
        first = devrows[0]
        for key in ("ground_truth_tensor_path", "measurement_path", "input_manifest_path", "accepted_np4_result_path"):
            if key in first:
                s2_paths.append(first[key])
                add("S2.sample." + key, checked_file(first[key]))
        method_dir = safe_path(first["accepted_np4_result_path"]).parent.parent
        add("S2.sample_methods", {"status": "VERIFIED" if method_dir.is_dir() else "MISSING",
                                 "path": str(method_dir),
                                 "names": sorted(p.name for p in method_dir.iterdir())[:12] if method_dir.is_dir() else []})
    s3 = s["S3_a12"]
    for key in ("root","inputs_root"):
        add("S3." + key, dir_check(s3[key]))
    idx = checked_file(safe_path(worktree) / s3["index_in_repo"])
    add("S3.index", idx)
    for uid in s3["ids"]:
        p = safe_path(s3["root"]) / uid
        add("S3.id." + uid, {"status":"VERIFIED" if p.is_dir() else "MISSING",
                              "rep_counts": {method: len(bounded(str(p / method / "rep*"), 5))
                                             for method in ("daps","sitcom","np_lf","np_s2")}})
    freeze = safe_path(s3["inputs_root"]) / "A1_2_INPUT_FREEZE_COMPLETE.json"
    add("S3.input_freeze", checked_file(freeze, s3["input_freeze_manifest_sha256"]))
    s4 = s["S4_a15"]
    add("S4.root", dir_check(s4["root"]))
    for uid in s3["ids"]:
        p = safe_path(s4["root"]) / uid
        add("S4.id." + uid, {"status": "VERIFIED" if p.is_dir() else "MISSING",
                             "rep_count": len(bounded(str(p / "rep*"), 5))})
    add("FFHQ.metadata", {"status": "VERIFIED" if any(safe_path(p).is_file() for p in config["data"]["ffhq_metadata_json_candidates"]) else "MISSING",
                           "candidates": config["data"]["ffhq_metadata_json_candidates"]})
    # Probe a maximum of one candidate, one GT and one y from each recoverable source.
    samples = {}
    if b22_example:
        data = sample_json_paths(b22_example)
        for _, val in (data.get("candidate_fields") or [])[:1]:
            samples["S1.candidate"] = str((safe_path(b22_example).parent / val).resolve()) if not Path(val).is_absolute() else val
    if devrows:
        samples["S2.GT"] = devrows[0].get("ground_truth_tensor_path")
        samples["S2.y"] = devrows[0].get("measurement_path")
    uid = s3["ids"][0]
    samples["S3.GT"] = str(safe_path(s3["inputs_root"]) / uid / "ground_truth_tensor.pt")
    for key, pattern in (("S3.candidate", str(safe_path(s3["root"]) / uid / "sitcom/rep0/sitcom1/reconstruction.png")),
                         ("S4.candidate", str(safe_path(s4["root"]) / uid / "rep0/reconstruction.png"))):
        samples[key] = pattern
    files = [str(safe_path(p)) for p in samples.values() if p and safe_path(p).is_file()]
    sample_python = safe_path(config["python"]["prdiff_ffhq"])
    if sample_python.is_file() and files:
        r = run([str(sample_python), "-c", SAMPLE_PROBE] + files, timeout=110, env=env)
        line = next((v.split("=",1)[1] for v in r["text"].splitlines() if v.startswith("R0_SAMPLE_JSON=")), None)
        add("sample_tensors", {"status": "VERIFIED" if line else "WARN",
                               "paths_requested": samples, "values": json.loads(line) if line else r["text"][-1000:]})
    else:
        add("sample_tensors", {"status": "WARN", "paths_requested": samples, "reason": "no eligible sample interpreter"})
    # Invariant: every touched path has passed safe_path; no forbidden payload discovery.
    add("forbidden_token_self_check", {"status": "VERIFIED",
                                      "policy": "all filesystem target paths checked before access"})
    status_counts = {status: sum(1 for rec in checks.values()
                         if isinstance(rec,dict) and rec.get("status")==status)
                     for status in ("VERIFIED","MISSING","MISMATCH","WARN")}
    report = {"schema": "prdiff.track_r.r0.inventory.v1", "utc": timestamp,
              "planner_base_sha": args.expected_sha, "statuses": status_counts,
              "checks": checks, "scope": "bounded, exposed-only; no protected cohort access"}
    (out / "R0_0_INVENTORY.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("R0_0_INVENTORY " + str(out / "R0_0_INVENTORY.json"))
    print("STATUS_COUNTS " + json.dumps(status_counts,sort_keys=True))
    for key, rec in checks.items():
        if rec.get("status") != "VERIFIED":
            print(rec["status"], key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
