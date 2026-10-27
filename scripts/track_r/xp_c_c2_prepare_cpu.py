#!/usr/bin/env python3
"""GT-clean XP-C / XP-C2 diagnostic CPU-only input freeze.

280 exact exposed images: FFHQ 100 B22 + 80 DEV80, ImageNet 100.
Uses historical R0.7 catalog's *GT identity only* (no candidate outcomes).
No reconstruction files, GT metrics, CLIP/DDPM inference, GPU, or scored
candidate data accessed. The resulting manifests are diagnostic-only:
never pass their GT paths to FAST/DAPS/SITCOM generation workers.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from collections import Counter
from pathlib import Path
from scripts.track_r import xp_t_sitcom_prepare_cpu as im
from scripts.track_r.xp_r0n2_b2_a8_cpu import ROOT,BASE
from scripts.track_r.xp_t_fast_prepare_cpu import sha
WT=ROOT/"pr_diffusion_track_r"
R07=BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json"
R07_RECEIPT=BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_PREPARE_RECEIPT.json"
SOURCE=im.IMAGENET/"XP_IMAGENET_GT_ONLY_CPU.json"
IMAGE_INDEX=im.IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
FORBID=("confirmation","confirm305","test-ffhq","demo-ffhq")
T=tuple(range(100,901,100))
ROT=("111","100")
C_POLICIES=("CLIP_FACE_FFHQ","CLIP_G1","CLIP_G2","PRIOR_2_FORWARD",
            "PRIOR_S_MID_GLOBAL2","PRIOR_S_MID_ALL8","UNIVFD")
def safe(path):
    p=Path(path)
    actual=p.resolve(strict=False)
    if (p.is_symlink() or not p.is_file() or
        ROOT.resolve() not in actual.parents or
        any(x in str(actual).lower() for x in FORBID)):
        raise RuntimeError("STOP_XPC_PROTECTED_OR_MISSING_FILE")
    return p
def digest(path):
    h=hashlib.sha256()
    with safe(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def jsonread(path):
    return json.loads(safe(path).read_text())
def sealed(row):
    if any(k in row for k in ("candidate_path","candidate_sha256","psnr",
                              "rho_signed","run_label")):
        raise RuntimeError("STOP_XPC_CANDIDATE_LEAK")
    return row
def freeze(ffhq,image):
    # R0 GT catalog is read only to identify historical GT source paths.
    identity={}
    for r in ffhq["rows"]:
        if r["source"]!="S5_r0_fast":continue
        cohort=r["measurement_source"]
        if cohort not in ("S1_b22","S2_b24_dev80"):
            raise RuntimeError("STOP_XPC_FFHQ_COHORT")
        key=(cohort,str(r["image_id"]))
        prev=(r["gt_path"],r["gt_format"])
        if key in identity and identity[key]!=prev:
            raise RuntimeError("STOP_XPC_FFHQ_GT_IDENTITY")
        identity[key]=prev
    if (len(identity)!=180 or
        sum(k[0]=="S1_b22" for k in identity)!=100 or
        sum(k[0]=="S2_b24_dev80" for k in identity)!=80):
        raise RuntimeError("STOP_XPC_FFHQ_N")
    gt={}
    for (cohort,iid),(path,fmt) in sorted(identity.items()):
        verified=safe(path)
        gt[("ffhq",cohort+"_"+iid)]=sealed(dict(
            dataset="ffhq",basename=cohort+"_"+iid,
            image_id=iid,cohort=cohort,source_path=str(verified),
            gt_format=fmt,source_sha256=digest(verified),
            checkpoint=str(ROOT/"models/ffhq_10m.pt")))
    imagedata=image["rows"]
    if (image.get("status")!="GT_ONLY_OFFLINE_NOT_FOR_GENERATION"
        or len(imagedata)!=100):
        raise RuntimeError("STOP_XPC_IMAGENET_GT_MANIFEST")
    for entry in imagedata:
        basename=entry["basename"]
        if (("imagenet",basename) in gt or
            not str(entry["source_path"]).startswith(
                str(ROOT/"data/imagenet/test-imagenet")+"/") or
            digest(entry["source_path"])!=entry["source_sha256"]):
            raise RuntimeError("STOP_XPC_IMAGE_SHA")
        gt[("imagenet",basename)]=sealed(dict(
            dataset="imagenet",basename=basename,image_id=basename,
            cohort="XP_IMAGENET100",source_path=entry["source_path"],
            gt_format="PIL_ToTensor_Resize256_CenterCrop256",
            source_sha256=entry["source_sha256"],
            checkpoint=str(ROOT/"models/imagenet256.pt")))
    if len(gt)!=280:raise RuntimeError("STOP_XPC_UNIVERSE")
    return list(gt.values())
def c2_tasks(images):
    task=[]
    for imrow in images:
        for t in T:
            for rot in ROT:
                for j in (0,1):
                    b=imrow["basename"]
                    dataset=imrow["dataset"]
                    key=f"XPC2|{dataset}|{b}|{t}|{j}"
                    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],
                                        "big")%(2**63)
                    task.append(dict(dataset=dataset,basename=b,source_sha256=imrow["source_sha256"],
                       t_ddpm1000=t,rotation=rot,noise_draw=j,noise_seed=seed,
                       key=key))
    if len(task)!=10080 or len({r["key"]+"|"+r["rotation"] for r in task})!=10080:
        raise RuntimeError("STOP_XPC2_REPLICATION")
    return task
def xdump(path,value):
    with path.open("x") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_XPC_CPU_ONLY")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_XPC_GIT")
    if (digest(IMAGE_INDEX)!=im.COMMON_Y_SHA
        or digest(ROOT/"models/imagenet256.pt")!=im.IMAGENET_MODEL_SHA
        or digest(ROOT/"models/ffhq_10m.pt")!=
           "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"):
        raise RuntimeError("STOP_XPC_MODEL_OR_IMAGE_PROVENANCE")
    catalog=jsonread(R07)
    img=jsonread(SOURCE)
    images=freeze(catalog,img)
    tasks=c2_tasks(images)
    out=Path(a.out)
    if (out.parent!=BASE or not out.name.startswith("xp_c_c2_")
        or out.exists() or out.is_symlink()):
        raise RuntimeError("STOP_XPC_OUTPUT_NAMESPACE")
    out.mkdir(parents=True,exist_ok=False)
    xdump(out/"XP_C_GT_IMAGES.json",{"status":"XP_C_GT_IMAGES_FROZEN",
        "head":head,"n_images":280,"n_ffhq":180,"n_imagenet":100,
        "diagnostic_gt_only":True,
        "policies":list(C_POLICIES),
        "ffhq_uses_CLIP_face":True,"imagenet_uses_G1_G2":True,
        "scoring_or_GPU_work_performed":False,
        "images":images})
    xdump(out/"XP_C2_EQUIVARIANCE_TASKS.json",{
        "status":"XP_C2_GT_EQ_TASKS_FROZEN","head":head,
        "n_images":280,"n_tasks":len(tasks),
        "t_ddpm1000":list(T),"rotations":list(ROT),
        "draws_per_image_t_rotation":2,
        "noise_rule":"SHA256(XPC2|dataset|basename|t|j)[:8] unsigned BE mod 2**63",
        "denoiser_time_mapping":"REQUIRED_BEFORE_GPU: map DDPM-1000 index to each checkpoint native schedule",
        "delta_definition":"mean||D_t(Rx_t)-R D_t(x_t)||^2 / (3*256^2)",
        "delta_tilde_denominator":"mean||D_t(x_t)-x_gt||^2 / (3*256^2)",
        "diagnostic_gt_only":True,"gpu_work_performed":False,
        "tasks":tasks})
    xdump(out/"XP_C_C2_PREP.json",{"status":"XP_C_C2_PREP_PASS",
      "head":head,"n_images":280,"n_C2_evaluations":10080,
      "C_manifest_sha256":digest(out/"XP_C_GT_IMAGES.json"),
      "C2_manifest_sha256":digest(out/"XP_C2_EQUIVARIANCE_TASKS.json"),
      "gt_paths_present_FOR_DIAGNOSTICS_ONLY":True,
      "no_candidate_outcomes_inspected":True,
      "gpu_work_performed":False})
    print("XP_C_C2_PREP_PASS images=280 C2_tuples=10080 GPU=false")
    print("XP_C_C2_RUN_ROOT="+str(out))
    print("XP_C_GT_SHA="+digest(out/"XP_C_GT_IMAGES.json"))
    print("XP_C2_TASKS_SHA="+digest(out/"XP_C2_EQUIVARIANCE_TASKS.json"))
    print("C_C2_GPU_WORK_AUTHORIZED=false")
    print("OK")
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--expected-head",required=True)
    a.add_argument("--out",required=True)
    try:main(a.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_C_C2_PREP "+str(e),flush=True)
        sys.exit(8)
