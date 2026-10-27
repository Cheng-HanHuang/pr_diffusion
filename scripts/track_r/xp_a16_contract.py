"""Immutable XP-A16 expansion contract for the existing native FAST worker.

All checks are GT-free. Generation is allowed before TH1 prediction freeze;
quality evaluations, image previews, per-root PSNR and outcomes are not.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from scripts.track_r.r0_5_prepare_cpu import native_seed

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
RUN=ROOT/"outputs/pr_diffusion/track_r/xp_a16_20261027T022827Z"
TASKS=RUN/"XP_A16_TASKS.json"
MANIFEST_SHA="796429527d01963668a67c5e39949803cb8cc5ecfc35d01dcb20c5ca1e978432"
PREP_HEAD="f67bd915dab3ce39be96264ca6a1ec002ad623ff"
TOTAL=2160
OLD=720
IMAGE_SIZE=12
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def validate_a16_source(path,digest,manifest,rows,producer,start,count):
    path=Path(path)
    if (path!=TASKS or path.is_symlink() or digest!=MANIFEST_SHA
        or producer!=PREP_HEAD):
        raise RuntimeError("STOP_A16_SOURCE_PRODUCER_OR_SHA")
    if (manifest.get("schema")!="prdiff.track_r.xp.fast-tasks.v1"
        or manifest.get("status")!="XP_FAST_TASKS_FROZEN"
        or manifest.get("stage")!="XP-A16"
        or manifest.get("head")!=PREP_HEAD
        or manifest.get("dataset")!="ffhq"
        or manifest.get("fast_head")!="a9758b38906b7c7950fa04491f37f6634948db26"
        or manifest.get("oversample")!=2.0
        or manifest.get("n_roots")!=TOTAL
        or manifest.get("n_images")!=180
        or manifest.get("historical_indices")!=[0,1,2,3]
        or manifest.get("new_indices")!=list(range(4,16))
        or manifest.get("gt_paths_supplied") is not False
        or manifest.get("outcome_embargo")!="NO_K_GT4_OUTCOMES_UNTIL_TH1_V1_REGISTRY"):
        raise RuntimeError("STOP_A16_PREREGISTERED_INPUT")
    if (len(rows)!=TOTAL or
        start<0 or count<1 or count>120 or
        start%IMAGE_SIZE!=0 or count%IMAGE_SIZE!=0 or
        start+count>TOTAL):
        raise RuntimeError("STOP_A16_IMAGE_ALIGNED_SLICE")
    per_image=defaultdict(list)
    for r in rows:
        key=(r["measurement_source"],str(r["image_id"]))
        per_image[key].append(r)
        if (r["source"]!="S5_r0_fast" or
            r["uid"]!=f"S5_r0_fast|FAST|{key[0]}_{key[1]}|rep{r['rep']}" or
            r["seed"]!=native_seed(key[0],key[1],r["rep"]) or
            any(x in str(r["measurement_path"]).lower() for x in BAD) or
            not str(r["measurement_path"]).startswith(str(ROOT)+"/")):
            raise RuntimeError("STOP_A16_UID_RNG_OR_Y_PATH")
    if (len(per_image)!=180 or
        sum(k[0]=="S1_b22" for k in per_image)!=100 or
        sum(k[0]=="S2_b24_dev80" for k in per_image)!=80):
        raise RuntimeError("STOP_A16_COHORT_UNIVERSE")
    for key,group in per_image.items():
        if (len(group)!=IMAGE_SIZE or
            [x["rep"] for x in group]!=list(range(4,16)) or
            len({x["measurement_sha256"] for x in group})!=1 or
            len({x["measurement_path"] for x in group})!=1):
            raise RuntimeError("STOP_A16_RUNS_OR_Y_IDENTITY")
    if ([r["measurement_source"] for r in rows[:1200]]!=["S1_b22"]*1200
        or [r["measurement_source"] for r in rows[1200:]]!=["S2_b24_dev80"]*960):
        raise RuntimeError("STOP_A16_COHORT_ORDER")
    return True

def build_chunks():
    # Immutable 2-cohort smoke; one complete 12-run image from each cohort.
    smoke=[(0,12),(1200,12)]
    # Every other cohort is split into aligned groups of at most 60 roots.
    main=[]
    for begin,end in ((12,1200),(1212,2160)):
        while begin<end:
            count=min(60,end-begin)
            if count%12:
                raise RuntimeError("STOP_A16_PARTIAL_IMAGE_CHUNK")
            main.append((begin,count))
            begin+=count
    all_chunks=smoke+main
    covered=[]
    for start,count in all_chunks:
        covered.extend(range(start,start+count))
    if sorted(covered)!=list(range(TOTAL)) or len(all_chunks)!=38:
        raise RuntimeError("STOP_A16_CHUNK_PARTITION")
    return smoke,main
