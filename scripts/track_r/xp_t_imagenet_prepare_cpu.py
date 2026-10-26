#!/usr/bin/env python3
"""Freeze common ImageNet-100 signed measurements and XP-T FAST timing tasks.

CPU-only ground-truth transformation matches FAST-DIPS pinned data.py:
ToTensor -> Resize(256) -> CenterCrop(256). Only y paths/SHAs go to
GPU manifest; no ground-truth data or paths are sent to generation.
"""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
XP0=WT/"docs/track_r/executors/xp/returns/XP0_IMAGENET_20261026T200007Z"
MODEL_SHA="a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"
SOURCE_HEAD="a9758b38906b7c7950fa04491f37f6634948db26"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
def safe(p):
    p=Path(p)
    real=p.resolve(strict=False);base=ROOT.resolve(strict=False)
    if not p.is_absolute() or any(t in str(p).lower() or t in str(real).lower() for t in BAD):
        raise RuntimeError("STOP_PROTECTED_DATA_PATH")
    if not (real==base or base in real.parents) or not p.is_file() or p.is_symlink():
        raise RuntimeError("STOP_MISSING_OR_OUTSIDE_DATA_PATH "+str(p))
    return p
def sha(p):
    h=hashlib.sha256()
    with safe(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def dump(p,v):
    with Path(p).open("x",encoding="utf-8") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def seed(prefix,name,rep=None,mod=2**63):
    text=prefix+"|"+name+(("|"+str(rep)) if rep is not None else "")
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"big")%mod
def validate_images(rows):
    if not isinstance(rows,list) or len(rows)!=100 or [r["basename"] for r in rows]!=sorted(r["basename"] for r in rows):
        raise RuntimeError("STOP_XP0_IMAGENET_NOT_SORTED100")
    if len({r["basename"] for r in rows})!=100:raise RuntimeError("STOP_XP0_DUPLICATE_IMAGE")
    for r in rows:
        p=Path(r["path"])
        if (set(r)!={"basename","path","sha256"} or p.name!=r["basename"]
            or p.suffix.lower() not in (".jpeg",".jpg",".png")
            or not str(p).startswith(str(ROOT/"data/imagenet/test-imagenet")+"/")):
            raise RuntimeError("STOP_IMAGE_PATH_OR_SCHEMA")
    return rows
def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("STOP_IMAGE_CPU_ONLY")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if head!=a.expected_head or dirty:raise RuntimeError("STOP_XP_IMAGE_GIT")
    out=Path(a.out)
    if not str(out).startswith(str(BASE/"xp_t_fast_imagenet_")) or out.exists() or out.is_symlink():
        raise RuntimeError("STOP_IMAGE_OUT_NAMESPACE")
    import torch,torchvision
    import torchvision.transforms as transforms
    from PIL import Image
    from prdiffusion.track_r.physics import magnitude,rho_signed
    root_img=XP0/"XP_0_IMAGENET_IMAGES.json"
    images=validate_images(json.loads(safe(root_img).read_text()))
    inventory=json.loads(safe(XP0/"XP_0_INVENTORY.json").read_text())
    if (inventory.get("status")!="XP_0_PASS" or inventory.get("imagenet_n")!=100 or
        inventory.get("imagenet_model_sha256")!=MODEL_SHA or
        sha(ROOT/"models/imagenet256.pt")!=MODEL_SHA):
        raise RuntimeError("STOP_XP0_IMAGENET_MODEL_OR_RECEIPT")
    xform=transforms.Compose([transforms.ToTensor(),transforms.Resize(256),transforms.CenterCrop(256)])
    torch.set_num_threads(min(torch.get_num_threads(),4))
    out.mkdir(parents=True,exist_ok=False)
    directory=out/"measurements";directory.mkdir()
    full=[];gt_only=[]
    for i,r in enumerate(images):
        if sha(r["path"])!=r["sha256"]:raise RuntimeError("STOP_IMAGE_SOURCE_SHA "+r["basename"])
        with Image.open(safe(r["path"])) as im:x=xform(im)
        if tuple(x.shape)==(1,256,256):x=x.repeat(3,1,1)
        if tuple(x.shape)!=(3,256,256):raise RuntimeError("STOP_IMAGE_CHANNELS "+r["basename"])
        x=(x*2-1).unsqueeze(0).float()
        with torch.no_grad():
            pure=magnitude(x)
            mseed=seed("XPMEAS|imagenet",r["basename"])
            g=torch.Generator(device="cpu").manual_seed(mseed)
            noise=torch.randn(pure.shape,generator=g,dtype=torch.float32)
            y=(pure+0.05*noise).contiguous()
            residual=float(rho_signed(x,y)[0])
            if y.shape!=(1,3,384,384) or not 0.0490<residual<0.0510:
                raise RuntimeError("STOP_IMAGE_Y_PHYSICS "+r["basename"])
        target=directory/("image_%03d.pt"%i)
        torch.save({"measurement":y},target)
        if not torch.equal(torch.load(target,map_location="cpu",weights_only=True)["measurement"],y):
            raise RuntimeError("STOP_IMAGE_Y_SERIALIZATION")
        full.append({"basename":r["basename"],"source_image_sha256":r["sha256"],
                     "measurement_path":str(target),"measurement_sha256":sha(target),
                     "seed":mseed,"sigma":0.05})
        gt_only.append({"basename":r["basename"],"source_path":r["path"],
                        "source_sha256":r["sha256"],"y_residual_rms":residual})
        if (i+1)%25==0:print("IMAGENET_MEASUREMENTS",i+1,"/100",flush=True)
    common={"schema":"prdiff.track_r.xp.imagenet.y.v1","status":"XP_IMAGENET100_Y_FROZEN",
            "head":head,"n_images":100,
            "source_image_list_sha256":sha(root_img),
            "preprocess":"FAST_DIPS_A15/data.py ToTensor Resize(256) CenterCrop(256)",
            "noise_rule":"CPU torch.randn, independent generator; XPMEAS|imagenet|basename, unsigned big-endian SHA[:8] mod 2**63",
            "gt_paths_supplied":False,"rows":full}
    dump(out/"XP_IMAGENET100_Y_MANIFEST.json",common)
    # This separate manifest is for GT-only CPU scoring, NEVER for generation.
    dump(out/"XP_IMAGENET_GT_ONLY_CPU.json",{
        "status":"GT_ONLY_OFFLINE_NOT_FOR_GENERATION","rows":gt_only,
        "torch":torch.__version__,"torchvision":torchvision.__version__})
    taskrows=[]
    for r in full[:3]:
        for rep in range(3):
            name=r["basename"]
            taskrows.append({
                "uid":f"XP_T|FAST|imagenet|{name}|rep{rep}",
                "source":"XP_T_FAST","measurement_source":"XP_IMAGENET100",
                "image_id":name,"rep":rep,
                "seed":seed("XP|FAST|imagenet",name,rep,2**31),
                "measurement_path":r["measurement_path"],
                "measurement_sha256":r["measurement_sha256"]})
    tasks={"schema":"prdiff.track_r.xp.fast-tasks.v1",
           "status":"XP_FAST_TASKS_FROZEN","head":head,
           "fast_head":SOURCE_HEAD,"dataset":"imagenet",
           "oversample":2.0,"model_sha256":MODEL_SHA,
           "n_roots":9,"gt_paths_supplied":False,"rows":taskrows,
           "stage":"XP-T","purpose":"ImageNet FAST batch-1 first three images, three runs"}
    dump(out/"XP_T_FAST_ROOTS.json",tasks)
    prep={"status":"XP_T_FAST_PREP_PASS","head":head,
          "roots":9,"images":3,"dataset":"imagenet",
          "manifest_sha256":sha(out/"XP_T_FAST_ROOTS.json"),
          "common_y_sha256":sha(out/"XP_IMAGENET100_Y_MANIFEST.json"),
          "n_shared_measurements":100,
          "source_image_list_sha256":sha(root_img),
          "gt_paths_in_generation":False,"torch":torch.__version__,
          "torchvision":torchvision.__version__}
    dump(out/"XP_T_FAST_PREP.json",prep)
    print("XP_T_IMAGENET_PREP_PASS y=100 timing_roots=9")
    print("COMMON_Y_SHA="+prep["common_y_sha256"]+" ROOT_SHA="+prep["manifest_sha256"])
    print("OK")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("ACTION NEEDED: XP_IMAGE_PREP "+str(exc))
        raise SystemExit(8)
