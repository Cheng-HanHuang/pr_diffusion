#!/usr/bin/env python3
"""XP-0 bounded, CPU-only inventory. No model, dataset-bundle or GPU loading."""
import argparse,hashlib,json,os,subprocess
from collections import defaultdict
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
CATALOG=BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json"
BAD=("confirmation","confirm305")
def safe(p,require_file=False):
    p=Path(p)
    if any(t in str(p).lower() for t in BAD):raise RuntimeError("STOP_FORBIDDEN_PATH")
    if p.is_symlink():raise RuntimeError("STOP_SYMLINK")
    if require_file and not p.is_file():raise RuntimeError("STOP_MISSING "+str(p))
    if p.exists() and any(t in str(p.resolve()).lower() for t in BAD):raise RuntimeError("STOP_REALPATH")
    return p
def sha(p):
    h=hashlib.sha256()
    with safe(p,True).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("STOP_GPU_VISIBLE")
    wt=Path(__file__).resolve().parents[2]
    head=subprocess.check_output(["git","-C",str(wt),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(wt),"status","--porcelain"],text=True).strip()
    if head!=a.expected_head or dirty:raise RuntimeError("STOP_UNCLEAN_HEAD")
    out=safe(a.out)
    if not str(out).startswith(str(BASE/"xp_0_")) or out.exists():raise RuntimeError("STOP_OUTPUT_ROOT")
    cat=json.loads(safe(CATALOG,True).read_text())
    rows=cat.get("rows",[])
    if len(rows)!=3240:raise RuntimeError("STOP_R07_CATALOG_3240")
    by_id=defaultdict(set);sources=defaultdict(set);checked={}
    for row in rows:
        for k in ("source","family","image_id","measurement_path","measurement_sha256"):
            if k not in row:raise RuntimeError("STOP_CATALOG_SCHEMA "+k)
        p=safe(row["measurement_path"],True)
        if str(p) not in checked:checked[str(p)]=sha(p)
        if checked[str(p)]!=row["measurement_sha256"]:
            raise RuntimeError("STOP_Y_SHA_DRIFT "+str(row["image_id"]))
        # S3/S4 are separate historical exposed comparisons and not the R0 180-id cohort.
        if row["source"] not in ("S1_b22","S2_b24_dev80","S5_r0_fast"):
            continue
        cohort=row.get("measurement_source",row["source"])
        if cohort not in ("S1_b22","S2_b24_dev80"):
            raise RuntimeError("STOP_MEASUREMENT_SOURCE "+str(cohort))
        key=(cohort,str(row["image_id"]))
        by_id[key].add(row["measurement_sha256"])
        sources[key].add(row["family"])
    if any(len(v)!=1 for v in by_id.values()) or len(by_id)!=180:
        raise RuntimeError("STOP_Y_EXCHANGEABILITY")
    ffhq=[dict(source=k[0],id=k[1],y_sha256=next(iter(y)),
               historical_families=sorted(sources[k]))
          for k,y in sorted(by_id.items())]
    model=ROOT/"models/imagenet256.pt"
    paths={
      "imagenet_root":ROOT/"data/imagenet/test-imagenet",
      "imagenet_alternate":ROOT/"external/DAPS/test-imagenet",
      "dps_code":ROOT/"external/dps_xp",
      "pc_daps_code":ROOT/"external/PC_DAPS_xp",
      "difffpr_code":ROOT/"external/DiffFPR",
      "ffhq_model":ROOT/"models/ffhq_10m.pt",
      "imagenet_model":model,
      "python_fast":ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python",
      "python_sitcom":ROOT/"conda-envs/sitcom_ode_bw/bin/python",
      "python_daps":ROOT/"conda-envs/daps/bin/python"}
    # No directory enumeration for any external FFHQ dataset.
    existence={k:{"path":str(p),"exists":safe(p).is_dir() if "code" in k or "root" in k or "alternate" in k else safe(p).is_file()} for k,p in paths.items()}
    envs={}
    for name in ("prdiff_track_r","fast_dips_a15_blackwell","daps","sitcom_ode_bw"):
        exe=ROOT/"conda-envs"/name/"bin/python"
        envs[name]=safe(exe).is_file()
    gpu=subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.total,memory.used",
       "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=30)
    if gpu.returncode:raise RuntimeError("STOP_NVIDIA_SMI")
    lines=gpu.stdout.strip().splitlines()
    if len(lines)!=4:raise RuntimeError("STOP_NOT_FOUR_GPUS")
    # The Fourier operator must be configured with 256 image and exact (512-256)//2=128 pad.
    pad=(256*4//2-256)//2
    if pad!=128 or 256+2*pad!=512:raise RuntimeError("STOP_PAD_FORMULA")
    # ImageNet is listed only at one explicitly named path if present; never an FFHQ bundle.
    imagenet=[]
    imroot=next((p for p in (paths["imagenet_root"],paths["imagenet_alternate"]) if safe(p).is_dir()),None)
    if imroot is not None:
        images=sorted(p for p in imroot.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png"))
        if len(images)!=100:raise RuntimeError("STOP_IMAGENET_NOT_100")
        imagenet=[{"basename":p.name,"path":str(p),"sha256":sha(p)} for p in images]
        if len({p["basename"] for p in imagenet})!=100:
            raise RuntimeError("STOP_IMAGENET_BASENAME_COLLISION")
    out.mkdir(parents=True,exist_ok=False)
    record={"status":"XP_0_PASS" if imroot is not None else "XP_0_NEEDS_IMAGENET",
     "head":head,"planner_ref":"92e4c4d05356fa5bda27273776a2eeaa039c29b4",
     "gpu_lines":lines,"pad_oversample_4":pad,"canvas_oversample_4":512,
     "ffhq_n":180,"historical_candidates_checked":len(rows),
     "unique_y_files_sha_checked":len(checked),"assets":existence,"environments":envs,
     "imagenet_n":len(imagenet),"imagenet_model_sha256":sha(model) if safe(model).is_file() else None}
    (out/"XP_0_INVENTORY.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    (out/"XP_0_FFHQ_MEASUREMENTS.json").write_text(json.dumps(ffhq,indent=2)+"\n")
    (out/"XP_0_IMAGENET_IMAGES.json").write_text(json.dumps(imagenet,indent=2)+"\n")
    print(record["status"],"FFHQ",len(ffhq),"Y_SHA_PASS",len(checked),
      "IMAGENET",len(imagenet),"PAD",pad)
    if not imroot:print("ACTION NEEDED: obtain DAPS test-imagenet only; never download/extract FFHQ archive")
    else:print("OK")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        print("ACTION NEEDED: XP0 "+str(e));raise SystemExit(8)
