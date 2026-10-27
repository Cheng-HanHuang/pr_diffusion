"""A6 D1 phase-2 sealed FAST ImageNet frozen-source contract.

The original manifest was frozen in the preceding overnight campaign under
generator commit 8d9bfed. Only 100 ImageNet images x runs 4..15 allowed.
This contract does not grant permission to SCORE the sealed results.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from scripts.track_r.xp_t_imagenet_prepare_cpu import seed as imagenet_seed
from scripts.track_r.xp_t_sitcom_prepare_cpu import IMAGENET,COMMON_Y_SHA
from scripts.track_r.xp_t_fast_prepare_cpu import sha
from scripts.track_r.xp_night_prepare_cpu import ROOT,BASE

ORIGINAL=BASE/"xp_overnight_20261027T045215Z"
D1P2=ORIGINAL/"D1_P2.json"
SHA="3107d40c495886c52ea2a407f4d8f74079278fffffaa93f4f55eff1edb5272f7"
SOURCE_HEAD="8d9bfed0fa62735791987a24791cfb97123540d8"
REPS=tuple(range(4,16))
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def validate(path,digest,mf,rows,producer,start,count):
    if (Path(path)!=D1P2 or Path(path).is_symlink()
        or digest!=SHA or producer!=SOURCE_HEAD
        or sha(path)!=SHA):
        raise RuntimeError("STOP_A6_D1P2_PINNED_SOURCE")
    if (mf.get("schema")!="prdiff.track_r.xp.fast-tasks.v1"
        or mf.get("status")!="XP_FAST_TASKS_FROZEN"
        or mf.get("stage")!="D1_P2"
        or mf.get("dataset")!="imagenet"
        or mf.get("head")!=SOURCE_HEAD
        or mf.get("outcome_embargo") is not True
        or mf.get("n_roots")!=1200
        or mf.get("n_images")!=100
        or mf.get("gt_paths_supplied") is not False
        or mf.get("model_sha256")!="a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"):
        raise RuntimeError("STOP_A6_D1P2_SCHEMA")
    if (len(rows)!=1200 or start<0 or count<1
        or start%12 or count%12 or start+count>1200):
        raise RuntimeError("STOP_A6_D1P2_ALIGNED_PARTITION")
    common=IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
    if sha(common)!=COMMON_Y_SHA:
        raise RuntimeError("STOP_A6_D1P2_COMMON_Y_DRIFT")
    import json
    measurements=json.loads(common.read_text())["rows"]
    if len(measurements)!=100:
        raise RuntimeError("STOP_A6_D1P2_IMAGENET_IDS")
    alias={x["basename"]:(x["measurement_path"],x["measurement_sha256"]) for x in measurements}
    if len(alias)!=100:
        raise RuntimeError("STOP_A6_D1P2_DUPLICATE_ID")
    seen=defaultdict(list)
    for row in rows:
        iid=row["image_id"];r=row["rep"]
        if (iid not in alias
            or row["uid"]!=f"XP|FAST|imagenet|{iid}|rep{r}"
            or row["source"]!="XP_D1"
            or row["measurement_source"]!="XP_IMAGENET100"
            or (row["measurement_path"],row["measurement_sha256"])!=alias[iid]
            or r not in REPS
            or row["seed"]!=imagenet_seed("XP|FAST|imagenet",iid,r,2**31)
            or any(s in str(row["measurement_path"]).lower() for s in BAD)):
            raise RuntimeError("STOP_A6_D1P2_Y_RNG_IDENTITY")
        seen[iid].append(r)
    if (set(seen)!=set(alias)
        or any(v!=list(REPS) for v in seen.values())):
        raise RuntimeError("STOP_A6_D1P2_COMPLETE_REPS")
    return True
