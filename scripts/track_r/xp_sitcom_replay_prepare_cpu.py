#!/usr/bin/env python3
"""XP SITCOM CPU-only frozen B22 sequential/B24 independent replay preflight.

Never reads ground truth or the protected cohort. GPU inputs contain only
signed y identities, seeds, native source/settings. Historical output paths
and frozen R0.7 labels live in a SEPARATE offline, CPU-only evaluation manifest.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from collections import defaultdict
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
CATALOG=BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json"
SCORED=BASE/"r0_7_cpu_20261023T080434Z/analysis/R0_7_SCORED_GT_ROWS.jsonl"
SITCOM=ROOT/"external/SITCOM_ODE"
SITCOM_HEAD="275ab67efbd8146bffca20155171ba6be1169c09"
# This checkout predates a native source-cleanliness policy. Git tracks
# ten Python 3.8 cache files, and a historical one-line edit to the unused
# posterior_sample.py CLI. Do not reset it or silently excuse solver changes.
NATIVE_DIRTY_ALLOWLIST=frozenset((
    "forward_operator/__pycache__/__init__.cpython-38.pyc",
    "forward_operator/__pycache__/fastmri_utils.cpython-38.pyc",
    "forward_operator/__pycache__/resizer.cpython-38.pyc",
    "model/__pycache__/__init__.cpython-38.pyc",
    "model/__pycache__/precond.cpython-38.pyc",
    "model/ddpm/__pycache__/fp16_util.cpython-38.pyc",
    "model/ddpm/__pycache__/nn.cpython-38.pyc",
    "model/ddpm/__pycache__/unet.cpython-38.pyc",
    "model/edm/dnnlib/__pycache__/__init__.cpython-38.pyc",
    "model/edm/dnnlib/__pycache__/util.cpython-38.pyc",
    "posterior_sample.py",
))

MODEL=ROOT/"models/ffhq_10m.pt"
MODEL_SHA="81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
SOURCES=(("S1_b22",100,"sequential_post_model"),("S2_b24_dev80",80,"independent_model_seed"))

def safe(path):
    p=Path(path)
    actual=p.resolve(strict=False)
    top=ROOT.resolve(strict=False)
    if (not p.is_absolute() or p.is_symlink() or not p.is_file() or
        not (actual==top or top in actual.parents) or
        any(b in str(p).lower() or b in str(actual).lower() for b in BAD)):
        raise RuntimeError("STOP_XP_SITCOM_BAD_SOURCE "+str(p))
    return p

def sha(path):
    h=hashlib.sha256()
    with safe(path).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):h.update(part)
    return h.hexdigest()

def external_checkpoint_provenance(checkpoint):
    """Pin the one allowed external SITCOM checkpoint, including symlink cases.

    All ordinary experiment sources still go through safe(), which refuses
    symlinks. Only the exact native ffhq checkpoint path can be resolved.
    """
    p=Path(checkpoint)
    if p!=SITCOM/"checkpoint/ffhq256.pt":
        raise RuntimeError("STOP_XP_SITCOM_UNEXPECTED_CHECKPOINT")
    if not p.is_symlink():
        if sha(p)!=MODEL_SHA:
            raise RuntimeError("STOP_XP_SITCOM_EXTERNAL_MODEL")
        return {"checkpoint_path":str(p),"is_symlink":False,
                "target_path":str(p.resolve(strict=True)),
                "sha256":MODEL_SHA}
    try:
        target=p.resolve(strict=True)
    except (OSError,RuntimeError) as exc:
        raise RuntimeError("STOP_XP_SITCOM_CHECKPOINT_LINK") from exc
    top=ROOT.resolve(strict=True)
    if (target==p or not target.is_file() or target.is_symlink() or
        not (target==top or top in target.parents) or
        any(b in str(p).lower() or b in str(target).lower() for b in BAD)):
        raise RuntimeError("STOP_XP_SITCOM_CHECKPOINT_LINK")
    if sha(target)!=MODEL_SHA:
        raise RuntimeError("STOP_XP_SITCOM_EXTERNAL_MODEL")
    return {"checkpoint_path":str(p),"is_symlink":True,
            "target_path":str(target),"sha256":MODEL_SHA}

def read(path):
    return json.loads(safe(path).read_text())

def dump(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")

def head(path):
    return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],
                                   text=True,timeout=30).strip()

def check_native_checkout(native_head, status_text, cli_numstat):
    """Accept ONLY observed tracked cache drift and unused CLI one-line edit.

    Native replay imports sampler, model, forward_operator directly; it does
    not execute or import posterior_sample.py. PYTHONPYCACHEPREFIX must be
    isolated for subsequent replay so these tracked pyc files cannot be read.
    """
    if native_head!=SITCOM_HEAD:
        raise RuntimeError("STOP_XP_SITCOM_NATIVE_HEAD")
    changed=[]
    for line in status_text.splitlines():
        # Disallow staged files, renames and new source changes even when
        # the path happens to be in the exception list.
        if len(line)<4 or line[:2]!=" M":
            raise RuntimeError("STOP_XP_SITCOM_DIRTY_STATUS")
        path=line[3:]
        if path not in NATIVE_DIRTY_ALLOWLIST or path in changed:
            raise RuntimeError("STOP_XP_SITCOM_NATIVE_SOURCE_CHANGED "+path)
        changed.append(path)
    # Both no CLI edits and exactly the previously observed 1+ / 1-
    # tracked CLI edit are allowable, because the CLI is not imported.
    cli_present="posterior_sample.py" in changed
    if cli_present and cli_numstat.strip()!="1\t1\tposterior_sample.py":
        raise RuntimeError("STOP_XP_SITCOM_CLI_EDIT_DIFFERENT")
    if not cli_present and cli_numstat.strip():
        raise RuntimeError("STOP_XP_SITCOM_CLI_STATUS_MISMATCH")
    return {
       "head":native_head,
       "tracked_modified":sorted(changed),
       "changed_python_source_files":[p for p in changed if p.endswith(".py")],
       "native_solver_model_operator_source_clean":True,
       "unused_posterior_sample_cli_modified":cli_present,
       "policy":"only listed tracked Python cache files and unused one-line CLI edit",
       "generation_require_isolated_pycache":True,
    }

def settings(b22):
    m=b22["sitcom"]
    expected={
       "anneal_schedule":"linear","anneal_sigma_max":100.0,
       "anneal_sigma_min":0.1,"anneal_steps":200,
       "diff_schedule":"linear","diff_sigma_min":0.01,"diff_steps":5,
       "lgvd_lr":5e-5,"lgvd_lr_min_ratio":0.01,"lgvd_steps":100,
       "lgvd_tau":0.01,"timestep":"poly-7",
       "model_path_relative_to_sitcom_root":"checkpoint/ffhq256.pt",
       "measurement_preprocessing":"none",
    }
    for k,v in expected.items():
        if m.get(k)!=v:raise RuntimeError("STOP_XP_SITCOM_HISTORICAL_CONFIG "+k)
    if m.get("master_seed")!=43 or m.get("trajectory_count")!=4 or m.get("selector_step")!=160:
        raise RuntimeError("STOP_XP_SITCOM_B22_STREAM_OR_SELECTOR")
    return expected

def selection(catalog):
    if len(catalog)!=3240:raise RuntimeError("STOP_XP_SITCOM_R07_CATALOG_COUNT")
    selected=[]
    for source,n,mode in SOURCES:
        rows=[r for r in catalog if r.get("source")==source and r.get("family")=="SITCOM"]
        ids=sorted({str(r["image_id"]) for r in rows})
        if len(ids)!=n or len(rows)!=4*n:
            raise RuntimeError("STOP_XP_SITCOM_SOURCE_COUNT "+source)
        for image in ids[:2]:
            chosen=sorted((r for r in rows if str(r["image_id"])==image),
                          key=lambda r:int(r["run_index"]))
            if [r["run_index"] for r in chosen]!=[0,1,2,3]:
                raise RuntimeError("STOP_XP_SITCOM_HISTORICAL_FOUR_RUNS")
            selected.extend(chosen[:2])
    if len(selected)!=8 or len({r["uid"] for r in selected})!=8:
        raise RuntimeError("STOP_XP_SITCOM_NOT_EIGHT")
    return selected

def native_seed_metadata(r,method):
    p=safe(r["candidate_path"])
    if method=="sequential_post_model":
        if r["candidate_format"]!="pt":
            raise RuntimeError("STOP_SITCOM_B22_NOT_PT")
        native=read(p.parent/"result.json")
        if (native.get("image_id")!=r["image_id"] or
            native.get("master_seed")!=43 or
            native.get("run_index")!=r["run_index"] or
            native.get("rng_stream_run_index")!=r["run_index"]):
            raise RuntimeError("STOP_SITCOM_B22_NATIVE_ID")
        return 43,{"native_result_sha256":sha(p.parent/"result.json"),
                   "native_tensor_hash":native.get("reconstruction_content_sha256")}
    if r["candidate_format"]!="png":
        raise RuntimeError("STOP_SITCOM_B24_NOT_PNG")
    run=p.parent.parent
    native=read(run/"frozen_runtime_config.json")
    method_config=native.get("sitcom1",{})
    expected=("anneal_schedule","anneal_sigma_max","anneal_sigma_min",
              "anneal_steps","diff_schedule","diff_sigma_min","diff_steps",
              "lgvd_lr","lgvd_lr_min_ratio","lgvd_steps","lgvd_tau","timestep",
              "measurement_preprocessing","model_path_relative_to_sitcom_root")
    b22=settings(read(WT/"configs/b22/b22_2_overnight.json"))
    if any(method_config.get(k)!=b22[k] for k in expected):
        raise RuntimeError("STOP_SITCOM_B24_CONFIG_DIFFERENCE")
    if method_config.get("num_runs")!=1 or not isinstance(method_config.get("seed"),int):
        raise RuntimeError("STOP_SITCOM_B24_SEED_CONFIG")
    # Frozen native output is a replay reference, not a source of GT.
    receipt=read(p.parent/"result.json")
    if (receipt.get("image_id")!=r["image_id"] or
        receipt.get("measurement_file_sha256")!=r["measurement_sha256"]):
        raise RuntimeError("STOP_SITCOM_B24_NATIVE_RECEIPT")
    return method_config["seed"],{"runtime_config_sha256":sha(run/"frozen_runtime_config.json"),
                                   "native_result_sha256":sha(p.parent/"result.json")}

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_XP_SITCOM_CPU_ONLY")
    if head(WT)!=a.expected_head or subprocess.check_output(
            ["git","-C",str(WT),"status","--porcelain"],text=True).strip():
        raise RuntimeError("STOP_XP_SITCOM_WRONG_OR_DIRTY_GIT")
    native_state=check_native_checkout(
        head(SITCOM),
        subprocess.check_output(
            ["git","-C",str(SITCOM),"status","--porcelain=v1",
             "--untracked-files=no"],text=True,timeout=30),
        subprocess.check_output(
            ["git","-C",str(SITCOM),"diff","--numstat","HEAD","--",
             "posterior_sample.py"],text=True,timeout=30),
    )
    native_state["unused_cli_sha256"]=sha(SITCOM/"posterior_sample.py")
    # Later inserted checkpoint proof is recorded in both GPU-safe manifest
    # and CPU prep receipt; its target is never a GT input.
    if sha(MODEL)!=MODEL_SHA:
        raise RuntimeError("STOP_XP_SITCOM_MODEL_SHA")
    checkpoint_state=external_checkpoint_provenance(
        SITCOM/"checkpoint/ffhq256.pt")
    b22=read(WT/"configs/b22/b22_2_overnight.json")
    native=settings(b22)
    catalog=read(CATALOG)
    scored={v["uid"]:v for line in safe(SCORED).read_text().splitlines()
            if line.strip() for v in [json.loads(line)]}
    chosen=selection(catalog["rows"])
    gpu_rows=[];offline=[];seed_sets=defaultdict(set);y_seen={}
    for r in chosen:
        source=str(r["source"]);iid=str(r["image_id"]);rep=int(r["run_index"])
        mode=next(z[2] for z in SOURCES if z[0]==source)
        if r["uid"] not in scored:raise RuntimeError("STOP_XP_SITCOM_NO_R07_SCORE")
        if sha(r["measurement_path"])!=r["measurement_sha256"]:
            raise RuntimeError("STOP_XP_SITCOM_Y_SHA "+r["uid"])
        if sha(r["candidate_path"])!=r["candidate_sha256"]:
            raise RuntimeError("STOP_XP_SITCOM_CANDIDATE_SHA "+r["uid"])
        key=(source,iid)
        old=(r["measurement_path"],r["measurement_sha256"])
        if key in y_seen and y_seen[key]!=old:
            raise RuntimeError("STOP_XP_SITCOM_Y_EXCHANGEABILITY")
        y_seen[key]=old
        seed,native_info=native_seed_metadata(r,mode)
        if not isinstance(seed,int) or not 0<=seed<2**32:
            raise RuntimeError("STOP_XP_SITCOM_SEED_RANGE")
        if mode=="independent_model_seed":
            if seed in seed_sets[key]:raise RuntimeError("STOP_XP_SITCOM_DUPLICATE_SEED")
            seed_sets[key].add(seed)
        gpu_rows.append({
          "uid":r["uid"],"source":source,"family":"SITCOM",
          "measurement_source":r.get("measurement_source",source),
          "image_id":iid,"rep":rep,"seed":seed,
          "rng_mode":mode,"measurement_path":r["measurement_path"],
          "measurement_sha256":r["measurement_sha256"]})
        offline.append({
          "uid":r["uid"],"source":source,"image_id":iid,"rep":rep,
          "historical_candidate_path":r["candidate_path"],
          "historical_candidate_sha256":r["candidate_sha256"],
          "historical_format":r["candidate_format"],
          "measurement_sha256":r["measurement_sha256"],
          "raw_label":scored[r["uid"]]["raw_label"],
          "psnr_raw":scored[r["uid"]]["psnr_raw"],
          "historical_native":native_info})
    out=Path(a.out)
    if (not str(out).startswith(str(BASE/"xp_sitcom_replay_")) or out.exists()
        or out.is_symlink() or any(z in str(out).lower() for z in BAD)):
        raise RuntimeError("STOP_XP_SITCOM_OUTPUT")
    out.mkdir(parents=True,exist_ok=False)
    manifest={
       "schema":"prdiff.track_r.xp.sitcom-replay-tasks.v1",
       "status":"XP_SITCOM_REPLAY_TASKS_FROZEN",
       "head":a.expected_head,"native_source_head":SITCOM_HEAD,
       "dataset":"ffhq","n_roots":8,"n_images":4,"n_y":4,
       "gt_paths_supplied":False,"historical_paths_supplied":False,
       "config":native,"b22_master_seed":43,"b22_trajectory_count":4,
       "model_sha256":MODEL_SHA,"model_relpath":"checkpoint/ffhq256.pt",
       "native_checkpoint_provenance":checkpoint_state,
       "native_checkout_provenance":native_state,
       "provenance":{"b22":"sequential post-model RNG reset per image",
                     "b24":"one independent seeded native model per run"},
       "rows":gpu_rows}
    mf=out/"XP_SITCOM_REPLAY_TASKS.json"
    dump(mf,manifest)
    offline_doc=out/"XP_SITCOM_REPLAY_OFFLINE_HISTORICAL.json"
    dump(offline_doc,{
       "status":"XP_SITCOM_HISTORICAL_REFERENCES_ONLY",
       "warning":"CPU-only, never feed to CUDA worker",
       "gt_paths_supplied":False,"rows":offline})
    receipt={
       "status":"XP_SITCOM_REPLAY_PREP_PASS","head":a.expected_head,
       "manifest_sha256":sha(mf),
       "offline_sha256":sha(offline_doc),"n_roots":8,"n_images":4,
       "historical_native_source_head":SITCOM_HEAD,
       "source_b22":4,"source_b24":4,"n_shared_y":4,
       "model_sha256":MODEL_SHA,
       "native_checkout_provenance":native_state,
       "native_checkpoint_provenance":checkpoint_state,
       "gt_access":False,
       "gpu_access":False}
    dump(out/"XP_SITCOM_REPLAY_PREP.json",receipt)
    print("XP_SITCOM_REPLAY_PREP_PASS roots=8 images=4 b22=4 b24=4")
    print("MANIFEST_SHA="+receipt["manifest_sha256"])
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ACTION NEEDED: XP_SITCOM_PREP "+str(e))
        raise SystemExit(8)
