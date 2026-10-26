#!/usr/bin/env python3
"""CPU-only XP-T SITCOM batch-1 calibration manifest for both datasets.

One shared signed noisy y per image across all solver families. Freeze first
three FFHQ B22 image IDs and first three ImageNet benchmark basenames,
three *new*, independent SITCOM timing seeds per image. Never read GT.
Historical SITCOM replay and its saved outputs remain completely separate.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from scripts.track_r.xp_sitcom_replay_prepare_cpu import (
    ROOT,WT,BASE,SITCOM,SITCOM_HEAD,MODEL_SHA,
    safe,sha,read,dump,settings,check_native_checkout,head
)

FFHQ=BASE/"xp_t_fast_20261026T205902Z"
IMAGENET=BASE/"xp_t_fast_imagenet_20261026T212512Z"
COMMON_Y_SHA="26619959e3a17aeb2c0ad413f87b1764ecd8bc7aabfc0415c862cbaf27873b78"
IMAGENET_MODEL=ROOT/"models/imagenet256.pt"
IMAGENET_MODEL_SHA="a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"

def seed(dataset,image,rep):
    uid="B22_"+image if dataset=="ffhq" else image
    value=hashlib.sha256(
        f"XP|SITCOM|{dataset}|{uid}|{rep}".encode("utf-8")).digest()
    return int.from_bytes(value[:8],"big")%(2**31)

def from_fast_batch(dataset,root):
    frozen=read(root/"XP_T_FAST_ROOTS.json")
    prep=read(root/"XP_T_FAST_PREP.json")
    if (prep.get("status")!="XP_T_FAST_PREP_PASS" or
        frozen.get("status")!="XP_FAST_TASKS_FROZEN" or
        frozen.get("dataset")!=dataset or
        frozen.get("n_roots")!=9 or
        prep.get("roots")!=9 or
        sha(root/"XP_T_FAST_ROOTS.json")!=prep.get("manifest_sha256")):
        raise RuntimeError("STOP_XPT_SITCOM_FAST_ANCHOR "+dataset)
    rows=frozen["rows"]
    ids=sorted({str(x["image_id"]) for x in rows})
    if (len(ids)!=3 or len(rows)!=9 or
        sorted((x["image_id"],x["rep"]) for x in rows)!=
        [(image,rep) for image in ids for rep in range(3)]):
        raise RuntimeError("STOP_XPT_SITCOM_FIRST3_X3 "+dataset)
    mapping={}
    for iid in ids:
        matches=[x for x in rows if x["image_id"]==iid]
        if len({r["measurement_path"] for r in matches})!=1 or len({
              r["measurement_sha256"] for r in matches})!=1:
            raise RuntimeError("STOP_XPT_SITCOM_SAME_Y_PER_IMAGE")
        path=matches[0]["measurement_path"]
        digest=matches[0]["measurement_sha256"]
        if sha(path)!=digest:
            raise RuntimeError("STOP_XPT_SITCOM_MEASUREMENT_HASH "+iid)
        mapping[iid]=(path,digest)
    return mapping,prep["manifest_sha256"]

def freeze_rows(ffhq,imagenet,common):
    if len(common)!=100 or len({r["basename"] for r in common})!=100:
        raise RuntimeError("STOP_XPT_SITCOM_IMAGENET100")
    names=[str(r["basename"]) for r in common]
    if names!=sorted(names):
        raise RuntimeError("STOP_XPT_SITCOM_IMAGENET_SORT_ORDER")
    aliases={r["basename"]:(r["measurement_path"],r["measurement_sha256"])
             for r in common}
    if list(imagenet)!=names[:3] or any(
        imagenet[x]!=aliases[x] for x in imagenet):
        raise RuntimeError("STOP_XPT_SITCOM_IMAGENET_COMMON_Y")
    out=[];all_seeds=set()
    for dataset,group in (("ffhq",ffhq),("imagenet",imagenet)):
        for image,(ypath,ydigest) in group.items():
            for rep in range(3):
                val=seed(dataset,image,rep)
                key=(dataset,image,val)
                if val==43 or key in all_seeds:
                    raise RuntimeError("STOP_XPT_SITCOM_SEED_COLLISION")
                all_seeds.add(key)
                out.append({
                    "uid":f"XP_T|SITCOM|{dataset}|{image}|rep{rep}",
                    "family":"SITCOM","dataset":dataset,"image_id":image,
                    "measurement_source":"S1_b22" if dataset=="ffhq" else "XP_IMAGENET100",
                    "rep":rep,"seed":val,
                    "measurement_path":ypath,"measurement_sha256":ydigest,
                })
    if (len(out)!=18 or
        len({x["uid"] for x in out})!=18 or
        len({(x["dataset"],x["image_id"]) for x in out})!=6):
        raise RuntimeError("STOP_XPT_SITCOM_NOT_18")
    return out

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_XPT_SITCOM_PREP_GPU_VISIBLE")
    if head(WT)!=a.expected_head or subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True).strip():
        raise RuntimeError("STOP_XPT_SITCOM_PREP_WORKTREE")
    current_native=check_native_checkout(
       head(SITCOM),
       subprocess.check_output(
           ["git","-C",str(SITCOM),"status","--porcelain=v1",
            "--untracked-files=no"],text=True),
       subprocess.check_output(
           ["git","-C",str(SITCOM),"diff","--numstat","HEAD","--",
            "posterior_sample.py"],text=True))
    if sha(ROOT/"models/ffhq_10m.pt")!=MODEL_SHA or (
        sha(IMAGENET_MODEL)!=IMAGENET_MODEL_SHA):
        raise RuntimeError("STOP_XPT_SITCOM_CHECKPOINTS")
    frozen_cfg=settings(read(WT/"configs/b22/b22_2_overnight.json"))
    ffy,ffsha=from_fast_batch("ffhq",FFHQ)
    iny,insha=from_fast_batch("imagenet",IMAGENET)
    ypath=IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
    if sha(ypath)!=COMMON_Y_SHA:
        raise RuntimeError("STOP_XPT_SITCOM_COMMON_IMAGE_Y_HASH")
    common=read(ypath)
    if common.get("status")!="XP_IMAGENET100_Y_FROZEN" or (
        common.get("n_images")!=100) or (
        common.get("gt_paths_supplied") is not False):
        raise RuntimeError("STOP_XPT_SITCOM_COMMON_IMAGE_Y_RECEIPT")
    rows=freeze_rows(ffy,iny,common["rows"])
    manifest={
       "schema":"prdiff.track_r.xp.sitcom.timing-batch1.v1",
       "status":"XP_T_SITCOM_TASKS_FROZEN",
       "head":a.expected_head,"native_source_head":SITCOM_HEAD,
       "n_roots":18,"n_images":6,"runs_per_image":3,
       "new_roots_only":True,"batch_size":1,
       "gt_paths_supplied":False,
       "historical_paths_supplied":False,
       "sampling_model":"SITCOM_NATIVE_PINNED",
       "sampler_config":frozen_cfg,
       "model_presets":{
           "ffhq":{
               "checkpoint_path":str(ROOT/"models/ffhq_10m.pt"),
               "checkpoint_sha256":MODEL_SHA,
               "num_channels":128,"num_res_blocks":1,
               "attention_resolutions":16,
           },
           "imagenet":{
               "checkpoint_path":str(IMAGENET_MODEL),
               "checkpoint_sha256":IMAGENET_MODEL_SHA,
               "num_channels":256,"num_res_blocks":2,
               "attention_resolutions":"32,16,8",
           },
       },
       "seed_rule":"SHA256('XP|SITCOM|dataset|uid|rep') first8 big endian mod 2**31",
       "source_fast_ffhq_task_sha":ffsha,
       "source_fast_imagenet_task_sha":insha,
       "source_imagenet_common_y_sha":COMMON_Y_SHA,
       "external_tracked_diff_allowlist":current_native["tracked_modified"],
       "external_unused_cli_sha256":sha(SITCOM/"posterior_sample.py"),
       "selection_rule":"first three FAST XP-T original B22 images and first three ImageNet y basenames, independent seeds r0-r2",
       "rows":rows,
    }
    out=Path(a.out)
    if (not str(out).startswith(str(BASE/"xp_t_sitcom_")) or
        out.exists() or out.is_symlink()):
        raise RuntimeError("STOP_XPT_SITCOM_OUTPUT_ROOT")
    out.mkdir(parents=True,exist_ok=False)
    mf=out/"XP_T_SITCOM_TASKS.json"
    dump(mf,manifest)
    receipt={
       "status":"XP_T_SITCOM_PREP_PASS",
       "head":a.expected_head,
       "manifest_sha256":sha(mf),
       "dataset_count":2,"n_roots":18,"n_images":6,
       "ffhq_roots":9,"imagenet_roots":9,
       "ffhq_images":list(ffy),
       "imagenet_images":list(iny),
       "no_gpu_work":True,"gt_access":False,
       "no_existing_candidate_rewritten":True,
       "external_source_head":SITCOM_HEAD,
       "model_sha256s":{
           "ffhq":MODEL_SHA,
           "imagenet":IMAGENET_MODEL_SHA,
       }}
    dump(out/"XP_T_SITCOM_PREP.json",receipt)
    print("XP_T_SITCOM_PREP_PASS roots=18 ffhq=9 imagenet=9")
    print("XP_T_SITCOM_MANIFEST_SHA="+receipt["manifest_sha256"])
    print("XP_T_SITCOM_RUN_ROOT="+str(out))
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: XP_T_SITCOM_PREP "+str(e),flush=True)
        sys.exit(8)
