#!/usr/bin/env python3
"""A20 XP-C3 offline chosen-GOOD computation for COMPLETED panels only.

The committed GT-free prior choice is never modified. Offline GT is used
only to evaluate PSNR20 for the previously chosen 000 or 111 orientation.
No new selector, no GPU, no model forward. Output exactly six G1-style
columns with panel in file name and SHA-backed local receipt.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
SOURCE_REL="docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
COLUMNS=("image_id","run","label","prior_choice","prior_margin","prior_chosen_good")
SIZES={"XP_N":6400,"G5":800,"XP_D4":6400}
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")
def gate(ok,label):
    if not ok:raise RuntimeError("STOP_A20_XPC3_OFFLINE_"+label)
def sha(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink(),"FILE_"+str(p))
    h=hashlib.sha256()
    with p.open("rb") as f:
        for z in iter(lambda:f.read(1<<20),b""):h.update(z)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open("x") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
def source_gt(panel):
    if panel=="XP_N":
        from scripts.track_r.xp_a19_xpn_prefix4_cpu import (
            GT_FILE,GT_SHA)
        gate(sha(GT_FILE)==GT_SHA,"XPN_FROZEN_GT_SHA")
        data=read(GT_FILE)
        gate(len(data["rows"])==100,"XPN_GT_100")
        return {r["basename"]:dict(path=r["gt_path"],
                 digest=r["gt_file_sha256"],mode="xpn_pt")
                for r in data["rows"]}
    if panel=="XP_D4":
        from scripts.track_r.xp_t_sitcom_prepare_cpu import IMAGENET
        source=IMAGENET/"XP_IMAGENET_GT_ONLY_CPU.json"
        d=read(source)
        gate(len(d["rows"])==100,"D4_GT_100")
        return {r["basename"]:dict(path=r["source_path"],
                digest=r["source_sha256"],mode="imagenet_pil")
                for r in d["rows"]}
    if panel=="G5":
        from scripts.track_r.xp_a15_cp3_score_cpu import common_sources
        _,gt,_=common_sources()
        gate(len(gt)==100,"G5_GT_100")
        return {iid:dict(path=r["source_path"],
              digest=r["source_sha256"],mode="imagenet_pil")
              for iid,r in gt.items()}
    raise ValueError(panel)

def load_ground(ref,torch):
    from prdiffusion.track_r.physics import to_rgb8
    gate(sha(ref["path"])==ref["digest"],"GT_IDENTITY_SHA")
    if ref["mode"]=="xpn_pt":
        obj=torch.load(ref["path"],map_location="cpu",weights_only=True)
        gate(isinstance(obj,dict) and
            isinstance(obj.get("gt_model_range"),torch.Tensor),
            "XPN_GT_PT_SCHEMA")
        gt=obj["gt_model_range"].float()
    else:
        from PIL import Image
        from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
        with Image.open(ref["path"]) as img:
            gt=Compose([ToTensor(),Resize(256),CenterCrop(256)])(
                    img.convert("RGB")).unsqueeze(0)*2-1
    gate(tuple(gt.shape)==(1,3,256,256) and
         bool(torch.isfinite(gt).all()),"GT_256_VALID")
    return to_rgb8(gt)

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
         text=True).strip()==a.expected_head and
         not subprocess.check_output(["git","-C",str(WT),
         "status","--porcelain"],text=True).strip(),
         "PINNED_HEAD")
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a25_xpc3_prod_")
         and run.is_dir() and not run.is_symlink(),"RUN")
    manifest=WT/SOURCE_REL
    data=read(manifest)
    gate(data["status"]=="A20_XPC3_ALL13600_SOURCE_FROZEN_UNSCORED"
         and len(data["rows"])==13600,"SOURCE_SHA")
    source={ (r["image_id"],r["run"]):r
        for r in data["rows"] if r["panel"]==a.panel}
    gate(len(source)==SIZES[a.panel],"SOURCE_PANEL")
    chunks=sorted((run/a.panel).glob("*.json"))
    gate(len(chunks)==100,"PANEL_MUST_BE_COMPLETE_100")
    vals={}
    for path in chunks:
        d=read(path)
        gate(d["status"]=="A20_XPC3_IMAGE_PRIOR_PASS"
             and d["panel"]==a.panel
             and d["source_manifest_sha256"]==sha(manifest),
             "GT_FREE_SCORER_PROOF")
        for r in d["rows"]:
            k=(r["image_id"],r["run"])
            gate(k in source and k not in vals
                 and r["label"]==source[k]["label"]
                 and r["prior_choice"] in ("000","111")
                 and int(r["denoiser_examples"])==32,
                 "PRIOR_ROW")
            margin=float(r["prior_margin"])
            gate(math.isfinite(margin)
                 and (r["prior_choice"]=="111")==(margin>1e-7),
                 "PRIOR_RULE_REPLAY")
            vals[k]=r
    gate(len(vals)==SIZES[a.panel],"ALL_PRIOR_ROWS")
    gtpaths=source_gt(a.panel)
    import torch
    from prdiffusion.track_r.physics import to_rgb8,apply_orbit
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    torch.set_num_threads(4)
    gt={}
    output=[]
    for (iid,k),s in sorted(source.items()):
        gate(iid in gtpaths,"SOURCE_GT_MATCH")
        if iid not in gt:gt[iid]=load_ground(gtpaths[iid],torch)
        pt=Path(s["pt_path"])
        gate(sha(pt)==s["pt_sha256"],"CANDIDATE_SHA")
        record=torch.load(pt,map_location="cpu",weights_only=True)
        x=record["reconstruction"].float()
        gate(tuple(x.shape)==(1,3,256,256) and
             bool(torch.isfinite(x).all()),"CANDIDATE_X")
        x8=to_rgb8(x)
        v=vals[iid,k]
        score=orbit_psnr_channel_mse(x8,gt[iid])
        raw=float(score["000"]);chosen=float(score[v["prior_choice"]])
        gate((raw>=20.)==(s["label"]=="G"),
             "FROZEN_RAW_LABEL_GOOD_DISAGREEMENT")
        output.append(dict(image_id=iid,run=k,label=s["label"],
           prior_choice=v["prior_choice"],
           prior_margin=v["prior_margin"],
           prior_chosen_good=int(chosen>=20.)))
    gate(len(output)==SIZES[a.panel],"COMPLETE_CHOSEN_GOOD")
    out=run/"final"
    out.mkdir(exist_ok=True)
    name=f"XP_C3_A20_{a.panel}_{SIZES[a.panel]}_PRIOR_DECISIONS.csv"
    dst=out/name
    gate(not dst.exists(),"NO_OVERWRITE")
    with dst.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS)
        w.writeheader();w.writerows(output)
    receipt=out/f"XP_C3_A20_{a.panel}_OFFLINE_RECEIPT.json"
    write(receipt,dict(status="A20_XPC3_COMPLETE_PANEL_OFFLINE_GT_EVALUATED",
      panel=a.panel,n_candidates=SIZES[a.panel],n_images=100,
      source_manifest_sha256=sha(manifest),
      code_frozen_before_scoring=True,
      prior_rule="S_mid global2 only",
      no_2forward_secondary=True,
      no_GT_in_model_inputs=True,
      GT_for_offline_chosen_good_only=True,
      csv_columns=list(COLUMNS),csv_sha256=sha(dst),
      good_count=sum(x["prior_chosen_good"] for x in output)))
    print("A20_XPC3_OFFLINE_PANEL_PASS",a.panel,
          "N",len(output),"SHA",sha(dst),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--panel",choices=tuple(SIZES),required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--expected-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A20_XPC3_OFFLINE",str(e),flush=True)
        sys.exit(8)
