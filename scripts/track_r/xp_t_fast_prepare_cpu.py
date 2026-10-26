#!/usr/bin/env python3
"""XP-T FAST FFHQ batch-1 calibration: freeze 3 runs x 3 B22 images, GT-free.

Conforms to XP prereg v1.1 §2 timing: first 3 B22 ids in frozen order.
Uses independently seeded XP roots, never historical R0 reconstruction seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
R05=BASE/"r0_5_canary_20261023T004927Z/prepare/R0_5_GT_FREE_ROOTS.json"
R05_SHA="2291aa793b612c973ae76ab3d36b1f20263cd747f3b23cf593e51517a26c3bfe"
EXTERNAL_HEAD="a9758b38906b7c7950fa04491f37f6634948db26"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def guard(p):
    p=Path(p)
    if not p.is_absolute() or any(b in str(p).lower() for b in BAD):
        raise RuntimeError("STOP_XPT_PROTECTED_OR_RELATIVE")
    real=p.resolve(strict=False)
    root=ROOT.resolve(strict=False)
    if any(b in str(real).lower() for b in BAD) or not (real==root or root in real.parents):
        raise RuntimeError("STOP_XPT_RESOLVED_PATH")
    if not p.is_file() or p.is_symlink():
        raise RuntimeError("STOP_XPT_MISSING_OR_SYMLINK")
    return p

def sha(p):
    h=hashlib.sha256()
    with guard(p).open("rb") as f:
        for x in iter(lambda:f.read(4*1024*1024),b""):h.update(x)
    return h.hexdigest()

def xp_seed(uid,r):
    """Preregistered first-8-byte big-endian SHA seed modulo 2**31."""
    h=hashlib.sha256(f"XP|FAST|ffhq|{uid}|{r}".encode()).digest()
    return int.from_bytes(h[:8],"big")%(2**31)

def make_rows(root_rows):
    from scripts.track_r.r0_5_fast_gpu import frozen_roots
    from scripts.track_r.r0_5_fast_gpu import CANARY_HEAD
    # Pin source R0.5 task schema and require original 720 roots.
    original=frozen_roots(root_rows,CANARY_HEAD)
    image_ids=sorted({str(z["image_id"]) for z in original
                      if z["measurement_source"]=="S1_b22"})
    if len(image_ids)!=100:raise RuntimeError("STOP_XPT_B22_SOURCE_NOT_100")
    rows=[]
    for iid in image_ids[:3]:
        originals=[z for z in original if
                   z["measurement_source"]=="S1_b22" and
                   str(z["image_id"])==iid]
        if sorted(z["rep"] for z in originals)!=[0,1,2,3]:
            raise RuntimeError("STOP_XPT_MISSING_HISTORICAL_R05_MEASUREMENT")
        paths={z["measurement_path"] for z in originals}
        shas={z["measurement_sha256"] for z in originals}
        if len(paths)!=1 or len(shas)!=1:
            raise RuntimeError("STOP_XPT_HISTORICAL_Y_CONFLICT")
        y=next(iter(paths)); digest=next(iter(shas))
        if sha(y)!=digest:
            raise RuntimeError("STOP_XPT_MEASUREMENT_SHA")
        for rep in range(3):
            uid=f"B22_{iid}"
            rows.append({
                "uid":f"XP_T|FAST|ffhq|{uid}|rep{rep}",
                "source":"XP_T_FAST",
                "measurement_source":"S1_b22",
                "image_id":iid,
                "rep":rep,
                "seed":xp_seed(uid,rep),
                "measurement_path":y,
                "measurement_sha256":digest})
    if (len(rows)!=9 or len({r["uid"] for r in rows})!=9 or
        len({r["image_id"] for r in rows})!=3):
        raise RuntimeError("STOP_XPT_NOT_NINE")
    return rows

def prepare(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_XPT_CPU_ONLY")
    actual=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
                                   text=True,timeout=30).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
                                  text=True,timeout=30).strip()
    if actual!=a.expected_head or dirty:
        raise RuntimeError("STOP_XPT_GIT")
    if sha(R05)!=R05_SHA:
        raise RuntimeError("STOP_XPT_R05_TASK_SHA")
    origin=json.loads(guard(R05).read_text())
    rows=make_rows(origin)
    from scripts.track_r.xp_0_inventory_cpu import CATALOG
    frozen_catalog=json.loads(guard(CATALOG).read_text())
    aliases={(r["measurement_source"],str(r["image_id"])):r["measurement_sha256"]
             for r in frozen_catalog["rows"] if r["source"]=="S5_r0_fast"}
    for r in rows:
        if aliases.get((r["measurement_source"],str(r["image_id"])))!=r["measurement_sha256"]:
            raise RuntimeError("STOP_XPT_XP0_Y_IDENTITY")
    out=Path(a.out)
    if not str(out).startswith(str(BASE/"xp_t_fast_")) or out.exists() or out.is_symlink():
        raise RuntimeError("STOP_XPT_OUTPUT_NAMESPACE")
    out.mkdir(parents=True,exist_ok=False)
    p=out/"XP_T_FAST_ROOTS.json"
    manifest={"schema":"prdiff.track_r.xp.fast-tasks.v1",
              "status":"XP_FAST_TASKS_FROZEN","dataset":"ffhq",
              "oversample":2.0,"head":actual,"fast_head":EXTERNAL_HEAD,
              "n_roots":9,"gt_paths_supplied":False,"rows":rows,
              "stage":"XP-T","purpose":"FAST FFHQ batch1 timing only",
              "seed_rule":"sha256('XP|FAST|ffhq|B22_<iid>|<rep>') first 8 bytes big-endian mod 2^31",
              "selection_rule":"first 3 B22 image IDs sorted, three runs each"}
    p.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    info={"status":"XP_T_FAST_PREP_PASS","head":actual,
          "manifest_sha256":sha(p),"roots":9,"images":3,
          "data_measurements_sha_checked":3,
          "prohibited_gt_paths":False,
          "new_reconstruction_roots":True}
    (out/"XP_T_FAST_PREP.json").write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
    print("XP_T_FAST_PREP_PASS roots=9 images=3 SHA="+info["manifest_sha256"])
    print("OK")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    try:prepare(ap.parse_args())
    except Exception as e:
        print("ACTION NEEDED: XP_T_FAST_PREP "+str(e),flush=True)
        raise SystemExit(8)
