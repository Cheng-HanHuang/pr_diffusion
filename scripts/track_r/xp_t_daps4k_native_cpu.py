#!/usr/bin/env python3
"""One-time CPU-only materialization of a pristine, sparse native DAPS tree.

Clone only Git objects from the historical local repository (no worktree edits,
no hardlinks). Sparse checkout JUST approved solver/model/config source paths:
NO dataset, examples, demos, user images, or bundled FFHQ files.
Keep original B20/B21 experimental checkout byte-for-byte unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
SOURCE=ROOT/"pr_diffusion_b19_solver/external/daps"
NATIVE=ROOT/"external/DAPS_XP_NATIVE_4K"
RECEIPT_DIR=BASE/"xp_t_daps4k_native_source_20261026"
PIN="e7a77d094167084faed19b599b96673b7bb11447"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
SOURCES=(
    "sampler.py",
    "cores/scheduler.py",
    "cores/mcmc.py",
    "forward_operator/__init__.py",
    "model/__init__.py",
    "model/ddpm/unet.py",
    "configs/sampler/edm_daps.yaml",
    "configs/task/phase_retrieval.yaml",
    "configs/model/ffhq256ddpm.yaml",
    "configs/model/imagenet256ddpm.yaml",
)
# Only these files/directories are materialized from the repository HEAD.
SPARSE=(
    "/sampler.py",
    "/cores/**",
    "/forward_operator/**",
    "/model/**",
    "/configs/sampler/**",
    "/configs/task/**",
    "/configs/model/**",
)

def git(repo,*args):
    return subprocess.check_output(
        ["git","-C",str(repo),*args],text=True,timeout=90).strip()

def checked(p):
    p=Path(p)
    real=p.resolve(strict=False)
    if not p.is_absolute() or any(
        x in str(p).lower() or x in str(real).lower() for x in BAD):
        raise RuntimeError("STOP_DAPS_NATIVE_FORBIDDEN_PATH")
    if ROOT.resolve() not in real.parents:
        raise RuntimeError("STOP_DAPS_NATIVE_OUTSIDE_ROOT")
    return p

def sha(p):
    p=checked(p)
    if p.is_symlink() or not p.is_file():
        raise RuntimeError("STOP_DAPS_NATIVE_NOT_REGULAR_FILE "+str(p))
    digest=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

def verify_clean_sparse(repo):
    if git(repo,"rev-parse","HEAD")!=PIN:
        raise RuntimeError("STOP_DAPS_NATIVE_WRONG_CHECKOUT_HEAD")
    if git(repo,"status","--porcelain","--untracked-files=no"):
        raise RuntimeError("STOP_DAPS_NATIVE_WORKTREE_DIRTY")
    patterns=(repo/".git/info/sparse-checkout")
    if not patterns.is_file() or any(
        x not in patterns.read_text() for x in SPARSE):
        raise RuntimeError("STOP_DAPS_NATIVE_BAD_SPARSE")
    # Exact allowlist—not a recursive dataset/source enumeration.
    proof={}
    for relative in SOURCES:
        path=repo/relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("STOP_DAPS_NATIVE_MISSING_SOURCE "+relative)
        present=git(repo,"hash-object","--",relative)
        pinned=git(repo,"rev-parse","HEAD:"+relative)
        if present!=pinned:
            raise RuntimeError("STOP_DAPS_NATIVE_MODIFIED "+relative)
        proof[relative]={"blob_sha":present,"sha256":sha(path)}
    for relative in ("dataset","data","configs/data","results"):
        if (repo/relative).exists() or (repo/relative).is_symlink():
            raise RuntimeError("STOP_DAPS_NATIVE_DATA_MATERIALIZED")
    return proof

def clone_sparse_pristine(src,dst,revision):
    """Clone Git object database, materializing approved files only.

    Never checkout the source worktree, and never materialize datasets.
    No deletion or cleanup even if the checkout fails.
    """
    src=checked(src);dst=checked(dst)
    if dst.exists() or dst.is_symlink():
        raise RuntimeError("STOP_DAPS_NATIVE_DEST_ALREADY_EXISTS")
    # A Git clone with --no-checkout copies Git objects but does not copy any
    # modified source tree file; sparse selection precedes first checkout.
    subprocess.run(["git","clone","--local","--no-hardlinks","--no-checkout",
                    "--",str(src),str(dst)],check=True,timeout=120)
    # Configure sparse patterns directly *before* any checkout, so even
    # a transient checkout never materializes bundled data or demo files.
    subprocess.run(["git","-C",str(dst),"config","core.sparseCheckout","true"],
                   check=True,timeout=90)
    subprocess.run(["git","-C",str(dst),"config","core.sparseCheckoutCone","false"],
                   check=True,timeout=90)
    pattern_file=dst/".git/info/sparse-checkout"
    if pattern_file.exists() or pattern_file.is_symlink():
        raise RuntimeError("STOP_DAPS_NATIVE_UNEXPECTED_SPARSE_STATE")
    with pattern_file.open("x",encoding="utf-8") as f:
        for line in SPARSE:f.write(line+"\n")
    subprocess.run(["git","-C",str(dst),"checkout","--detach",revision],
                   check=True,timeout=90)


def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_DAPS_NATIVE_GPU_VISIBLE")
    if (git(WT,"rev-parse","HEAD")!=a.expected_head
        or git(WT,"status","--porcelain")):
        raise RuntimeError("STOP_DAPS_NATIVE_WRONG_OR_DIRTY_XP_HEAD")
    src=checked(SOURCE);dst=checked(NATIVE);out=checked(RECEIPT_DIR)
    if src.is_symlink() or not src.is_dir() or git(src,"rev-parse","HEAD")!=PIN:
        raise RuntimeError("STOP_DAPS_NATIVE_ORIGINAL_NOT_PINNED")
    if dst.exists() or dst.is_symlink() or out.exists() or out.is_symlink():
        raise RuntimeError("STOP_DAPS_NATIVE_ALREADY_EXISTS")
    if (git(src,"rev-parse","HEAD:sampler.py")!=
        "0e0d1bf9d7893278c8d5589583ba1d3f10bcd972"):
        raise RuntimeError("STOP_DAPS_NATIVE_PINNED_SAMPLER_UNKNOWN")
    if (git(src,"hash-object","sampler.py")!=
        "ae1fc6908bb1e7f0c0349d4f82cdf20364389db8"):
        raise RuntimeError("STOP_DAPS_NATIVE_SOURCE_AUDIT_DRIFT")
    clone_sparse_pristine(src,dst,PIN)
    proof=verify_clean_sparse(dst)
    # Re-confirm old modified tree remains unchanged and retains the B20/B21
    # experimental sampler blob; no reset or stash was applied.
    if (git(src,"rev-parse","HEAD")!=PIN or
        git(src,"hash-object","sampler.py")!=
        "ae1fc6908bb1e7f0c0349d4f82cdf20364389db8"):
        raise RuntimeError("STOP_DAPS_NATIVE_ORIGINAL_CHANGED")
    out.mkdir(parents=True,exist_ok=False)
    record={
        "status":"XP_T_DAPS4K_PRISTINE_NATIVE_PASS",
        "head":a.expected_head,
        "pinned_upstream_head":PIN,
        "pristine_checkout":str(dst),
        "original_modified_checkout":str(src),
        "original_sampler_blob":"ae1fc6908bb1e7f0c0349d4f82cdf20364389db8",
        "pristine_sampler_blob":"0e0d1bf9d7893278c8d5589583ba1d3f10bcd972",
        "checkout_mode":"local_copy_git_objects_no_hardlinks_sparse_before_checkout",
        "sparse_patterns":SPARSE,"tracked_files_verified":proof,
        "data_directory_materialized":False,
        "no_gpu_work":True,"old_checkout_mutated":False,
    }
    dest=out/"XP_T_DAPS4K_NATIVE_SOURCE.json"
    with dest.open("x",encoding="utf-8") as f:
        json.dump(record,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("XP_T_DAPS4K_PRISTINE_NATIVE_PASS")
    print("NATIVE_CHECKOUT="+str(dst))
    print("NATIVE_SOURCE_RECEIPT="+str(dest))
    print("NATIVE_SOURCE_RECEIPT_SHA256="+sha(dest))
    print("DATA_MATERIALIZED=false")
    print("OLD_EXPERIMENTAL_CHECKOUT_UNCHANGED=true")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: XP_T_DAPS4K_NATIVE_SOURCE "+str(e),flush=True)
        sys.exit(8)
