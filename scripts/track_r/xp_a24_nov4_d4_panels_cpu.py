#!/usr/bin/env python3
"""A19 Nov4 XP-D4 never-orbit-GOOD eight-case figure (CPU evaluation only).

Read the previously scored and accepted 100x64 ImageNet XP-D4 panel.
Select up to eight of the 22 never-orbit-GOOD image IDs using ascending
SHA256(image_id UTF-8), regardless of any image quality or appearance.
Render anonymized [reference, E0 lowest signed residual, orbit oracle].
Original D1 (runs 0-15) and G1a/G1b (16-63) candidate sources keep
their original frozen SHA/seed/measurement lineage. No new generation.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def gate(ok,label):
    if not ok:raise RuntimeError("STOP_A24_NOV4_D4_"+label)

def sha(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink() and
         ROOT.resolve() in p.resolve().parents and
         not any(z in str(p.resolve()).lower() for z in BAD),
         "SAFE_PATH")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for z in iter(lambda:f.read(1<<20),b""):h.update(z)
    return h.hexdigest()

def read(p):
    with Path(p).open() as f:return json.load(f)

def write(p,d):
    with Path(p).open("x") as f:
        json.dump(d,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def cases_of(strings):
    gate(len(strings)==100,"D4_100")
    by={}
    for s in strings:
        iid=s["image_id"]
        labs=s["labels_by_run"].split(";")
        rhos=[float(x) for x in s["rho_signed_by_run"].split(";")]
        gate(iid not in by and len(labs)==len(rhos)==64,"D4_64")
        by[iid]=dict(labs=labs,rhos=rhos)
    never=[iid for iid,v in by.items()
           if all(z not in ("G","O_g","O_p") for z in v["labs"])]
    gate(len(never)==22,"D4_22_NEVER_ORBIT")
    # Figure-only selection: exactly the image_id string hashed, no GT.
    return by,sorted(never,
        key=lambda iid:(hashlib.sha256(iid.encode("utf-8")).hexdigest(),iid))[:8]

def provenance(a):
    from scripts.track_r.xp_a15_cp2_d4_cpu import (
        provenance as cp2,find_one_glob,csvload)
    from scripts.track_r import xp_a10_d1p2_compare_cpu as d1
    p=cp2(a)
    wt,arts,source,initial,clip0,grouped,gtmap,ymap,g1brun,old,new=p
    a10=getattr(d1,"ARGS",None)
    d1.ARGS=a
    try:
        p1,p2,inputs,sources=d1.source_contract()
    finally:
        d1.ARGS=a10
    gate(len(p1)==400 and len(p2)==1200 and len(grouped)==100,
         "D1_D4_LINEAGE")
    ds=sorted(BASE.glob("xp_a15_cp2_d4_*/XP_A15_CP2_D4_SCORING_RECEIPT.json"))
    good=[]
    for recfile in ds:
        rec=read(recfile)
        sf=recfile.parent/"D4_64RUN_STRINGS.csv"
        if (rec.get("status")==
            "XP_A15_CP2_D4_SCORED_READY_FOR_INDEPENDENT_AUDIT" and
            sha(sf)==rec["csv_sha256"]["D4_64RUN_STRINGS.csv"] and
            rec["n_candidates"]==6400):
            good.append((sf,recfile))
    gate(len(good)==1,"CP2_UNIQUE_COMPLETE_LOCAL_SCORE")
    sf,receipt=good[0]
    records=csvload(sf)
    by,ids=cases_of(records)
    gate(set(by)==set(grouped)==set(gtmap)==set(sources),
         "D4_D1_SAME_IMAGES")
    return p1,p2,grouped,gtmap,by,ids,sf,receipt

def reconstruct(iid,rep,p1,p2,grouped):
    from scripts.track_r.xp_a15_cp2_d4_cpu import safe
    from prdiffusion.track_r.physics import to_rgb8
    import torch
    if rep<16:
        from scripts.track_r.xp_fast_gpu import validate_complete
        from scripts.track_r.xp_a10_d1p2_compare_cpu import (
            P1_HEAD,P1_SHA,A6_HEAD,D1_SHA)
        row,pt=(p1 if rep<4 else p2)[iid,rep]
        validate_complete(pt.parent,row,
                          P1_HEAD if rep<4 else A6_HEAD,
                          P1_SHA["D1_P1"] if rep<4 else D1_SHA)
    else:
        folder,row,rec=grouped[iid][rep]
        pt=folder/"reconstruction.pt"
        gate(sha(pt)==rec["reconstruction_pt_sha256"],
             "G1AB_RECON_SHA")
    data=torch.load(safe(pt),map_location="cpu",weights_only=True)
    gate(isinstance(data,dict) and
         isinstance(data.get("reconstruction"),torch.Tensor),
         "RECON_PT_CONTAINER")
    x=data["reconstruction"].float()
    gate(tuple(x.shape)==(1,3,256,256) and bool(torch.isfinite(x).all()),
         "RECON_SHAPE")
    return to_rgb8(x)

def render(a,orig,by,ids,p1,p2,grouped,gtmap,sf,receipt):
    import torch
    from PIL import Image,ImageDraw
    from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
    from prdiffusion.track_r.physics import to_rgb8,apply_orbit
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    torch.set_num_threads(4)
    out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a24_nov4_d4_")
         and not out.exists() and not out.is_symlink(),
         "NEW_OUTPUT")
    out.mkdir(mode=0o700)
    panels=[]
    for number,iid in enumerate(ids,1):
        src=gtmap[iid]
        gate(sha(src["source_path"])==src["source_sha256"],
             "GT_SOURCE_SHA")
        with Image.open(src["source_path"]) as g:
            raw=Compose([ToTensor(),Resize(256),CenterCrop(256)])(g.convert("RGB"))
        gt=to_rgb8(raw.unsqueeze(0)*2-1)
        rhos=by[iid]["rhos"]
        e0=min(range(64),key=lambda j:(rhos[j],j))
        e0_x=reconstruct(iid,e0,p1,p2,grouped)
        oracle=None
        for j in range(64):
            x=reconstruct(iid,j,p1,p2,grouped)
            for orient,value in orbit_psnr_channel_mse(x,gt).items():
                if oracle is None or (-float(value),j,orient) < (-oracle[0],oracle[1],oracle[2]):
                    oracle=(float(value),j,orient,x)
        gate(oracle is not None and oracle[0]<20.,
             "FROZEN_NEVER_ORBIT_LABEL")
        best=apply_orbit(oracle[3],oracle[2])
        image=Image.new("RGB",(608,224),(255,255,255))
        draw=ImageDraw.Draw(image)
        for k,(title,x) in enumerate(zip(
                ("Reference","Min residual","Orbit oracle"),
                (gt,e0_x,best))):
            pix=Image.fromarray(x[0].permute(1,2,0).numpy(),"RGB")
            pix=pix.resize((192,192),Image.Resampling.BILINEAR)
            ox=8+200*k
            image.paste(pix,(ox,24))
            draw.text((ox,7),title,fill=(0,0,0))
        filename="D4_never_orbit_%02d.png"%number
        path=out/filename
        image.save(path,optimize=True)
        gate(path.stat().st_size<2_000_000,"PANEL_UNDER_2MB")
        panels.append(dict(filename=filename,sha256=sha(path),
            image_id=iid,anonymized=True,
            e0_run=e0,oracle_run=oracle[1],
            oracle_orientation=oracle[2],oracle_psnr=oracle[0]))
        print("A24_D4_PANEL",number,"/8",flush=True)
    write(out/"A24_D4_EIGHT_PANELS_RECEIPT.json",
          dict(status="A19_D4_NEVER_ORBIT_EIGHT_PANELS_PASS",
               n_never_orbit=22,n_selected=8,
               selection_sha_rule="ascending SHA256(image_id UTF-8)",
               original_CP2_strings_sha256=sha(sf),
               original_CP2_receipt_sha256=sha(receipt),
               n_png=8,panels=panels,
               GT_offline_only=True,solver_unchanged=True,
               no_GPU=True))
    print("A24_D4_EIGHT_PANELS_PASS",len(panels),flush=True)

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(subprocess.check_output(["git","-C",str(WT),
           "rev-parse","HEAD"],text=True).strip()==a.expected_head
           and not subprocess.check_output(["git","-C",str(WT),
           "status","--porcelain"],text=True).strip(),
           "PINNED_EXECUTOR_HEAD")
    p1,p2,grouped,gtmap,by,ids,sf,receipt=provenance(a)
    if a.preflight_only:
        print("A24_D4_NEVER_ORBIT_SOURCE_PREFLIGHT_PASS",
              "NEVER_ORBIT",22,"SELECTED",len(ids),flush=True)
        return
    render(a,None,by,ids,p1,p2,grouped,gtmap,sf,receipt)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--preflight-only",action="store_true")
    try:main(parser.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A24_NOV4_D4",str(e),flush=True)
        sys.exit(8)
