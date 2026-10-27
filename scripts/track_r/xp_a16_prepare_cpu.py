#!/usr/bin/env python3
"""CPU-only frozen XP-A16 FAST-FFHQ inputs; no GT or GPU work."""
import argparse,hashlib,json,os,subprocess,sys
from collections import defaultdict
from pathlib import Path
from scripts.track_r.r0_5_fast_prepare_cpu import native_seed
from scripts.track_r.r0_5_fast_gpu import CANARY_HEAD,frozen_roots,SOURCE_HEAD
from scripts.track_r.xp_t_fast_prepare_cpu import ROOT,WT,BASE,R05,R05_SHA,guard,sha
from scripts.track_r.xp_0_inventory_cpu import CATALOG

PROOF=WT/"docs/track_r/executors/xp/returns/XP_FAST_REPLAY_RECOVER_20261026T204735Z/XP_FAST_REPLAY_RECOVERY_STATUS.json"

def build(history,catalog):
    original=frozen_roots(history,CANARY_HEAD)
    groups=defaultdict(list)
    for row in original:
        if row["source"]!="S5_r0_fast":
            raise RuntimeError("STOP_XPA16_SOURCE")
        groups[(row["measurement_source"],str(row["image_id"]))].append(row)
    if len(groups)!=180 or sum(k[0]=="S1_b22" for k in groups)!=100 or sum(k[0]=="S2_b24_dev80" for k in groups)!=80:
        raise RuntimeError("STOP_XPA16_180_IDS")
    aliases=defaultdict(set)
    for x in catalog:
        if x["source"]=="S5_r0_fast":
            aliases[(x["measurement_source"],str(x["image_id"]))].add(
                (x["measurement_path"],x["measurement_sha256"]))
    if set(aliases)!=set(groups) or any(len(v)!=1 for v in aliases.values()):
        raise RuntimeError("STOP_XPA16_CATALOG")
    rows=[]
    for (cohort,iid),past in sorted(groups.items()):
        past.sort(key=lambda z:z["rep"])
        if [x["rep"] for x in past]!=[0,1,2,3]:
            raise RuntimeError("STOP_XPA16_HISTORICAL_REPS")
        y={(x["measurement_path"],x["measurement_sha256"]) for x in past}
        if len(y)!=1 or y!=aliases[(cohort,iid)]:
            raise RuntimeError("STOP_XPA16_Y_IDENTITY")
        path,digest=next(iter(y))
        if sha(path)!=digest:
            raise RuntimeError("STOP_XPA16_Y_HASH")
        old={x["seed"] for x in past}
        if len(old)!=4 or any(x["seed"]!=native_seed(cohort,iid,x["rep"]) for x in past):
            raise RuntimeError("STOP_XPA16_HISTORICAL_SEEDS")
        new=set()
        for rep in range(4,16):
            seed=native_seed(cohort,iid,rep)
            if seed in old or seed in new:
                raise RuntimeError("STOP_XPA16_SEED_COLLISION")
            new.add(seed)
            rows.append(dict(
                uid=f"S5_r0_fast|FAST|{cohort}_{iid}|rep{rep}",
                source="S5_r0_fast",measurement_source=cohort,image_id=iid,
                rep=rep,seed=seed,measurement_path=path,
                measurement_sha256=digest))
    if len(rows)!=2160 or len({x["uid"] for x in rows})!=2160:
        raise RuntimeError("STOP_XPA16_2160_ROOTS")
    return rows

def write(path,obj):
    with path.open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_XPA16_GPU_VISIBLE")
    git=lambda *args:subprocess.check_output(
        ["git","-C",str(WT),*args],text=True).strip()
    if git("rev-parse","HEAD")!=a.expected_head or git("status","--porcelain"):
        raise RuntimeError("STOP_XPA16_GIT")
    if sha(R05)!=R05_SHA:
        raise RuntimeError("STOP_XPA16_HISTORICAL_MANIFEST")
    proof=json.loads(guard(PROOF).read_text())
    if any(proof.get(k)!=v for k,v in dict(
           status="XP_FAST_REPLAY_RECOVERY_PASS",science="XP_FAST_REPLAY_PASS",
           n_roots=8,n_png_exact=8,r07_label_matches=8,
           gt_access_in_generation=False).items()):
        raise RuntimeError("STOP_XPA16_REPLAY_GATE")
    past=json.loads(guard(R05).read_text())
    catalog=json.loads(guard(CATALOG).read_text())["rows"]
    rows=build(past,catalog)
    out=Path(a.out)
    if out.parent!=BASE or not out.name.startswith("xp_a16_") or out.exists() or out.is_symlink():
        raise RuntimeError("STOP_XPA16_OUTPUT")
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema="prdiff.track_r.xp.fast-tasks.v1",
      status="XP_FAST_TASKS_FROZEN",stage="XP-A16",
      head=a.expected_head,dataset="ffhq",oversample=2.0,
      fast_head=SOURCE_HEAD,n_roots=2160,n_images=180,
      gt_paths_supplied=False,
      historical_indices=[0,1,2,3],new_indices=list(range(4,16)),
      historical_manifest_sha256=R05_SHA,
      replay_receipt_sha256=sha(PROOF),
      outcome_embargo="NO_K_GT4_OUTCOMES_UNTIL_TH1_V1_REGISTRY",
      rows=rows)
    m=out/"XP_A16_TASKS.json"
    write(m,manifest)
    write(out/"XP_A16_PREP.json",dict(
      status="XP_A16_PREP_PASS",head=a.expected_head,
      manifest_sha256=sha(m),n_roots=2160,n_images=180,
      historical_roots=720,measurement_hashes_verified=180,
      seed_collisions=0,gt_access=False,gpu_work_performed=False,
      outcomes_embargoed=True))
    print("XP_A16_PREP_PASS images=180 new_roots=2160 K=16")
    print("XP_A16_TASKS_SHA="+sha(m))
    print("XP_A16_RUN_ROOT="+str(out))
    print("K_GT4_OUTCOME_EMBARGO=ACTIVE")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as exc:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: XP_A16_PREP "+str(exc))
        sys.exit(8)
