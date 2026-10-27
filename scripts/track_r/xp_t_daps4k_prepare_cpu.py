#!/usr/bin/env python3
"""CPU-only GT-free XP-T DAPS-4K calibration input lock (FFHQ+ImageNet).

Use EXACT shared frozen signed measurements in FAST/SITCOM XP-T:
3 first sorted B22 FFHQ IDs + 3 first sorted ImageNet names x 3 new seeds.
Freeze published nonlinear phase-retrieval DAPS-4K, never shorter CLI 1K.
No ground truth or historical candidate loaded. No GPU inference.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from scripts.track_r.xp_t_sitcom_prepare_cpu import (
    ROOT,WT,BASE,FFHQ,IMAGENET,COMMON_Y_SHA,IMAGENET_MODEL,
    IMAGENET_MODEL_SHA,MODEL_SHA,from_fast_batch,freeze_rows as sitcom_rows,
    safe,sha,read,dump,head
)

DAPS=ROOT/"pr_diffusion_b19_solver/external/daps"
DAPS_HEAD="e7a77d094167084faed19b599b96673b7bb11447"
PY_DAPS=ROOT/"conda-envs/daps/bin/python"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
# The native DAPS pixel model/checkpoint path is reconfigured to the
# already pinned PAC canonical checkpoints, NOT its bundled datasets.
TRACKED_SOURCE=(
    "sampler.py","cores/scheduler.py","cores/mcmc.py",
    "forward_operator/__init__.py","model/__init__.py",
    "model/ddpm/unet.py",
    "configs/sampler/edm_daps.yaml",
    "configs/task/phase_retrieval.yaml",
    "configs/model/ffhq256ddpm.yaml",
    "configs/model/imagenet256ddpm.yaml",
)

# DAPS nonlinear task published DAPS-4K. Values verified against
# DAPS HEAD source's native edm_daps and phase_retrieval/pixel YAML.
DAPS4K={
    "method":"DAPS-4K",
    "sampler":"edm_daps","task":"phase_retrieval","task_group":"pixel",
    "operator":{"name":"phase_retrieval","oversample":2.0,"sigma":0.05},
    "annealing_scheduler_config":{
        "name":"edm","num_steps":400,"sigma_max":100.0,
        "sigma_min":0.1,"timestep":"poly-7",
    },
    "diffusion_scheduler_config":{
        "name":"edm","num_steps":10,"sigma_min":0.01,"timestep":"poly-7",
    },
    "mcmc_sampler_config":{
        "num_steps":100,"lr":5e-5,"tau":0.01,"lr_min_ratio":0.01,
        "mc_algo":"langevin","prior_solver":"gaussian",
    },
    "batch_size":1,
    "record":False,
}
# These values are *not* the 200x5 "DAPS-1K" GitHub shell commands.
assert DAPS4K["annealing_scheduler_config"]["num_steps"]==400
assert DAPS4K["diffusion_scheduler_config"]["num_steps"]==10
assert DAPS4K["mcmc_sampler_config"]["num_steps"]==100

def native_sources():
    if not DAPS.is_dir() or DAPS.is_symlink():
        raise RuntimeError("STOP_XPT_DAPS_SOURCE_DIRECTORY")
    actual=head(DAPS)
    if actual!=DAPS_HEAD:
        raise RuntimeError("STOP_XPT_DAPS_NATIVE_HEAD "+actual)
    result={}
    for relative in TRACKED_SOURCE:
        path=DAPS/relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("STOP_XPT_DAPS_SOURCE_FILE "+relative)
        observed=subprocess.check_output(
            ["git","-C",str(DAPS),"hash-object","--",relative],
            text=True,timeout=30).strip()
        committed=subprocess.check_output(
            ["git","-C",str(DAPS),"rev-parse","HEAD:"+relative],
            text=True,timeout=30).strip()
        if observed!=committed:
            raise RuntimeError("STOP_XPT_DAPS_TRACKED_SOURCE_DRIFT "+relative)
        result[relative]={"sha256":sha(path),"git_blob_sha":observed}
    # Python caches in this checkout may be tracked and modified.
    # Future native runs MUST isolate PYTHONPYCACHEPREFIX, not reset.
    return result

def new_seed(dataset,image,rep):
    """New-family prereg: first8 SHA256 of XP|DAPS|dataset|uid|r."""
    uid="B22_"+image if dataset=="ffhq" else image
    digest=hashlib.sha256(
        f"XP|DAPS|{dataset}|{uid}|{rep}".encode()).digest()
    return int.from_bytes(digest[:8],"big")%(2**31)

def construct(ffhq,imagenet,common):
    reference=sitcom_rows(ffhq,imagenet,common)
    rows=[]
    for old in reference:
        dataset=old["dataset"];image=old["image_id"];rep=old["rep"]
        uid="B22_"+image if dataset=="ffhq" else image
        rows.append({
            "uid":f"XP_T|DAPS|{dataset}|{image}|rep{rep}",
            "family":"DAPS","dataset":dataset,"image_id":image,
            "measurement_source":old["measurement_source"],
            "measurement_path":old["measurement_path"],
            "measurement_sha256":old["measurement_sha256"],
            "rep":rep,
            "seed":new_seed(dataset,image,rep),
        })
    if (len(rows)!=18 or
        len({r["uid"] for r in rows})!=18 or
        len({(r["dataset"],r["image_id"],r["seed"]) for r in rows})!=18 or
        [r["dataset"] for r in rows]!=["ffhq"]*9+["imagenet"]*9):
        raise RuntimeError("STOP_XPT_DAPS_ROWS")
    return rows

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_XPT_DAPS_CPU_ONLY")
    if head(WT)!=a.expected_head or subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True).strip():
        raise RuntimeError("STOP_XPT_DAPS_DIRTY_WORKTREE")
    if not PY_DAPS.is_file():
        raise RuntimeError("STOP_XPT_DAPS_ENV_NOT_FOUND")
    if sha(ROOT/"models/ffhq_10m.pt")!=MODEL_SHA or sha(
            IMAGENET_MODEL)!=IMAGENET_MODEL_SHA:
        raise RuntimeError("STOP_XPT_DAPS_CHECKPOINT_IDENTITY")
    tracked=native_sources()
    ff,ffsha=from_fast_batch("ffhq",FFHQ)
    im,imsha=from_fast_batch("imagenet",IMAGENET)
    frozen_y=IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
    if sha(frozen_y)!=COMMON_Y_SHA:
        raise RuntimeError("STOP_XPT_DAPS_IMAGENET_COMMON_Y_SHA")
    cy=read(frozen_y)
    if (cy.get("status")!="XP_IMAGENET100_Y_FROZEN" or
        cy.get("n_images")!=100 or
        cy.get("gt_paths_supplied") is not False):
        raise RuntimeError("STOP_XPT_DAPS_SHARED_Y_METADATA")
    rows=construct(ff,im,cy["rows"])
    out=Path(a.out)
    if (out.exists() or out.is_symlink() or
        out.parent!=BASE or not out.name.startswith("xp_t_daps4k_")):
        raise RuntimeError("STOP_XPT_DAPS_OUTPUT_NAMESPACE")
    manifest={
        "schema":"prdiff.track_r.xp.daps4k.timing.v1",
        "status":"XP_T_DAPS4K_TASKS_FROZEN",
        "purpose":"native DAPS published nonlinear PR 4K timing; no scoring",
        "family":"DAPS","method":"DAPS-4K",
        "dataset_order":["ffhq","imagenet"],
        "head":a.expected_head,"source_head":DAPS_HEAD,
        "source_path":str(DAPS),"native_sources":tracked,
        "n_roots":18,"n_images":6,"runs_per_image":3,"batch_size":1,
        "gt_paths_supplied":False,"historical_paths_supplied":False,
        "historical_population_not_extended":True,
        "evaluation_outcomes_embargo":"NOT_SCORED",
        "settings":DAPS4K,
        "model_presets":{
            "ffhq":{"checkpoint":str(ROOT/"models/ffhq_10m.pt"),
                    "checkpoint_sha256":MODEL_SHA,
                    "num_channels":128,"num_res_blocks":1,
                    "attention_resolutions":16},
            "imagenet":{"checkpoint":str(IMAGENET_MODEL),
                        "checkpoint_sha256":IMAGENET_MODEL_SHA,
                        "num_channels":256,"num_res_blocks":2,
                        "attention_resolutions":"32,16,8"},
        },
        "source_fast_ffhq_task_sha":ffsha,
        "source_fast_imagenet_task_sha":imsha,
        "source_imagenet_common_y_sha":COMMON_Y_SHA,
        "seed_rule":"sha256('XP|DAPS|dataset|uid|rep') first8 bigend mod 2^31",
        "rows":rows,
    }
    out.mkdir(parents=True,exist_ok=False)
    dump(out/"XP_T_DAPS4K_TASKS.json",manifest)
    receipt={
        "status":"XP_T_DAPS4K_PREP_PASS",
        "head":a.expected_head,"manifest_sha256":sha(
            out/"XP_T_DAPS4K_TASKS.json"),
        "n_roots":18,"ffhq_roots":9,"imagenet_roots":9,
        "n_images":6,"source_head":DAPS_HEAD,
        "gt_free":True,"gpu_work_performed":False,
        "model_weights_not_loaded":True,
        "canonical_4k_configuration":True,
    }
    dump(out/"XP_T_DAPS4K_PREP.json",receipt)
    print("XP_T_DAPS4K_PREP_PASS roots=18 ffhq=9 imagenet=9")
    print("XP_T_DAPS4K_MANIFEST_SHA="+receipt["manifest_sha256"])
    print("XP_T_DAPS4K_RUN_ROOT="+str(out))
    print("GPU_WORK_PERFORMED=false")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: XP_T_DAPS4K_PREP "+str(e),flush=True)
        sys.exit(8)
