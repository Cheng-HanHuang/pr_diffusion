#!/usr/bin/env python3
"""A20 publication of NOV4 frozen FAST + D4 figures and face E1 into XP returns.

No GPU, no selection updates, no new model calls. Regenerates publication-quality
3x256 RGB8 tiles from already-authenticated source candidate reconstructions
rather than interpolating the earlier 192-pixel previews. All selection is
diagnostic only (GT oracle used for figure analysis, not a method).
Source CPU files are SHA audited; no private absolute paths published.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
NAMES=("image_id","run","label","f_choice","f_margin","f_chosen_good","cohort")
FAST=BASE/"xp_a24_nov4_fast_20261102T172620Z"
D4=BASE/"xp_a24_nov4_d4_20261102T172620Z"
SOURCE_CSV_SHA="3595676cd2f7686e747d2659d25a77a6a4a2e34edaa7261a8b7f1fbf53f542e3"
V1_REF="8ef5f94d6de8aab4c3cc25b17fa797db0de1560b"
V1_FILE="docs/track_r/theory/TH1_PREDICTIONS_v1.json"
V1_SHA="c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753"
SRC=FAST/"A24_FAST_11520_FACE_CLIP_DECISIONS.csv"
FREC=FAST/"A24_FAST_FACECLIP_RECEIPT.json"
FPR=FAST/"FAST7_PANELS/FAST7_PANEL_RECEIPT.json"
DREC=D4/"A24_D4_EIGHT_PANELS_RECEIPT.json"

def gate(v,msg):
    if not v: raise RuntimeError("STOP_A25_NOV4_PUBLISH_"+msg)

def sha(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink(),"MISSING_"+p.name)
    h=hashlib.sha256()
    with p.open("rb") as f:
        for v in iter(lambda:f.read(1<<20),b""):h.update(v)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text())

def write(p,x):
    with Path(p).open("x") as f:
        json.dump(x,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def rows(p):
    with Path(p).open(newline="") as f:return list(csv.DictReader(f))

def make_panel(ground,e0,oracle,e0_db,oracle_db):
    gate(all(tuple(x.shape)==(1,3,256,256) for x in
             (ground,e0,oracle)),"RGB8_256_SOURCE")
    # Original full 256x256 bytes, no interpolation or original image id text.
    w=3*256+4*10
    im=Image.new("RGB",(w,304),"white")
    draw=ImageDraw.Draw(im)
    info=(("Reference",ground,None),
          ("E0 lowest residual",e0,e0_db),
          ("Oracle orbit",oracle,oracle_db))
    for z,(title,t,db) in enumerate(info):
        pix=Image.fromarray(t[0].permute(1,2,0).numpy(),"RGB")
        gate(pix.size==(256,256),"NO_RESAMPLE")
        x=10+z*266
        im.paste(pix,(x,26))
        draw.text((x,7),title,fill="black")
        if db is not None:draw.text((x,285),f"PSNR: {db:.2f} dB",fill="black")
    return im

def verify_completed():
    gate(sha(SRC)==SOURCE_CSV_SHA,"FAST_FACE_CSV_PIN")
    fr=load(FREC);fp=load(FPR);dr=load(DREC)
    gate(fr["status"]=="A19_A9_2D_FAST11520_FACECLIP_E1_SCORED"
         and fr["decisions_sha256"]==SOURCE_CSV_SHA
         and fp["status"]=="A19_FAST7_NEVER_GOOD_PANELS_PASS"
         and dr["status"]=="A19_D4_NEVER_ORBIT_EIGHT_PANELS_PASS"
         and len(fp["panels"])==7 and len(dr["panels"])==8,
         "FROZEN_RECEIPT")
    gate(all(sha(FAST/"FAST7_PANELS"/z["filename"])==z["sha256"]
         for z in fp["panels"]),"FAST7_ORIGINAL_SHA")
    gate(all(sha(D4/z["filename"])==z["sha256"]
         for z in dr["panels"]),"D4_ORIGINAL_SHA")
    gate(dr["selection_sha_rule"]=="ascending SHA256(image_id UTF-8)",
         "D4_SHA_ID_PRESELECTION")
    return fr,fp,dr

def publication_figure_fast(out,case,ident,table,hist,n16,n64,number):
    from prdiffusion.track_r.physics import apply_orbit,to_rgb8
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    from scripts.track_r.r0_2_g0_cpu import gt_source
    from scripts.track_r.xp_a24_nov4_fast_cpu import candidate
    group={"B22":"S1_b22","DEV80":"S2_b24_dev80"}[case["cohort"]]
    iid=str(case["image_id"]);key=(group,iid)
    gate(key in ident and all(c!="G" for c in table[key]["labels"]),
         "FAST_NEVER_GOOD")
    gtpath,fmt,_,_=ident[key]
    gt=to_rgb8(gt_source(dict(gt_path=gtpath,gt_format=fmt)).float())
    e0=int(case["rank_winner_run"])
    e0x=candidate(group,iid,e0,hist,n16,n64)
    e0ps=float(orbit_psnr_channel_mse(e0x,gt)["000"])
    oracle=(-1.,0,"000",None)
    for r in range(64):
        x=candidate(group,iid,r,hist,n16,n64)
        for orient,db in orbit_psnr_channel_mse(x,gt).items():
            if (-float(db),r,orient)<(-oracle[0],oracle[1],oracle[2]):
                oracle=(float(db),r,orient,x)
    gate(oracle[0]<25 and oracle[3] is not None,
         "FAST_ORBIT_ORACLE_STAYS_NEVERGOOD")
    dest=out/f"FAST_never_good_{number:02d}.png"
    make_panel(gt,e0x,apply_orbit(oracle[3],oracle[2]),e0ps,
               oracle[0]).save(dest,optimize=True)
    gate(dest.stat().st_size<2_000_000,"FAST_2MB")
    return dict(filename=dest.name,sha256=sha(dest),
       dataset="FFHQ",cohort=case["cohort"],image_id=iid,
       E0_run=e0,E0_psnr=e0ps,
       oracle_run=oracle[1],oracle_orientation=oracle[2],
       oracle_psnr=oracle[0],three_original_256px_tiles=True)

def publication_figure_d4(out,iid,by,p1,p2,grouped,gtmap,number):
    import torch
    from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
    from prdiffusion.track_r.physics import apply_orbit,to_rgb8
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    from scripts.track_r.xp_a24_nov4_d4_panels_cpu import reconstruct
    source=gtmap[iid]
    gate(sha(source["source_path"])==source["source_sha256"],"D4_GT_SHA")
    with Image.open(source["source_path"]) as im:
        base=Compose([ToTensor(),Resize(256),CenterCrop(256)])(im.convert("RGB"))
    gt=to_rgb8(base.unsqueeze(0)*2-1)
    e0=min(range(64),key=lambda j:(by[iid]["rhos"][j],j))
    e0x=reconstruct(iid,e0,p1,p2,grouped)
    e0ps=float(orbit_psnr_channel_mse(e0x,gt)["000"])
    oracle=(-1.,0,"000",None)
    for r in range(64):
        x=reconstruct(iid,r,p1,p2,grouped)
        for orient,db in orbit_psnr_channel_mse(x,gt).items():
            if (-float(db),r,orient)<(-oracle[0],oracle[1],oracle[2]):
                oracle=(float(db),r,orient,x)
    gate(oracle[3] is not None and oracle[0]<20.,"D4_NEVER_ORBIT")
    dest=out/f"D4_never_orbit_{number:02d}.png"
    make_panel(gt,e0x,apply_orbit(oracle[3],oracle[2]),e0ps,
               oracle[0]).save(dest,optimize=True)
    gate(dest.stat().st_size<2_000_000,"D4_2MB")
    return dict(filename=dest.name,sha256=sha(dest),dataset="ImageNet",
       cohort="IN",image_id=iid,E0_run=e0,E0_psnr=e0ps,
       oracle_run=oracle[1],oracle_orientation=oracle[2],
       oracle_psnr=oracle[0],three_original_256px_tiles=True)

def convert_face():
    from scripts.track_r.xp_a24_nov4_fast_cpu import (
        frozen,e1_face_summary)
    raw=rows(SRC)
    gate(len(raw)==11520 and
         len({(r["cohort"],r["image_id"],int(r["run"])) for r in raw})==11520,
         "CSV_11520_UNIQUE")
    output=[]
    for r in raw:
        gate(r["cohort"] in ("B22","DEV80") and
             r["face_choice"] in ("000","111") and
             r["face_chosen_good"] in ("0","1") and
             math.isfinite(float(r["face_margin"])),"FACE_ROW")
        output.append(dict(image_id=r["image_id"],run=int(r["run"]),
          label=r["label"],f_choice=r["face_choice"],
          f_margin=r["face_margin"],
          f_chosen_good=int(r["face_chosen_good"]),
          cohort=r["cohort"]))
    _,table,_,_,_,_=frozen()
    # U-statistic function requires original face-choice naming and
    # source rank/rho; compute by original source, not rounded publication CSV.
    curves=e1_face_summary(raw,table)
    return output,curves

def compare_v1(curves):
    r=subprocess.check_output(["git","-C",str(WT),"show",
                              V1_REF+":"+V1_FILE],timeout=80)
    gate(hashlib.sha256(r).hexdigest()==V1_SHA,"REGISTERED_V1_SHA")
    d=json.loads(r)
    gate(d["status"]=="FROZEN_BEFORE_HIGH_K_OUTCOMES",
         "V1_PREOUTCOME_REGISTRATION")
    measured={(c["cohort"],int(c["n"]),int(c["K"])):c
              for c in curves}
    comparisons=[]
    for x in d["predictions"]:
        if x.get("family")!="FAST" or x.get("policy")!="E1_CLIP":
            continue
        n=int(x["panel_n_max"]);k=int(x["K"])
        key=(x["cohort"],n,k)
        gate(key in measured,"V1_REGISTERED_K_MISSING")
        observed=measured[key]["exact_subset_E1_face"]
        lo=float(x["lo95"]);hi=float(x["hi95"]);point=float(x["point"])
        sd=max((hi-lo)/3.92,.005)
        comparisons.append(dict(cohort=x["cohort"],panel=x["panel"],
            n=n,K=k,model_variant=x["model_variant"],
            forecast_point=point,lo95=lo,hi95=hi,observed=observed,
            inside=lo<=observed<=hi,
            standardized_residual=abs(observed-point)/sd))
    gate(len(comparisons)==120,"V1_FAST_E1_120_PREDICTIONS")
    grouped=defaultdict(list)
    for x in comparisons:grouped[x["cohort"],x["panel"],x["model_variant"]].append(x)
    coverage=[]
    for (cohort,panel,variant),rr in sorted(grouped.items()):
        coverage.append(dict(cohort=cohort,panel=panel,variant=variant,
           covered=sum(z["inside"] for z in rr),total=len(rr),
           mean_standardized_residual=sum(z["standardized_residual"]
                       for z in rr)/len(rr)))
    gate(len(coverage)==len(grouped),"V1_COVERAGE")
    return dict(status="A20_A9_2D_FAST_FACE_E1_V1_FORECAST_COVERAGE",
        original_v1_sha256=V1_SHA,
        n_comparisons=120,comparisons=comparisons,
        coverage=coverage,refit=False,
        posthoc_face_N2_diagnostic=True)

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
         text=True).strip()==a.expected_head
         and not subprocess.check_output(["git","-C",str(WT),
         "status","--porcelain"],text=True).strip(),"PINNED_CLEAN_WORKTREE")
    fr,fp,dr=verify_completed()
    out=WT/"docs/track_r/executors/xp/returns/XP_A25_NOV4_20261102"
    theory=WT/"docs/track_r/executors/xp/theory_inputs"
    gate(not out.exists() and theory.is_dir(),"NEW_PUBLICATION_PATH")
    transformed,curves=convert_face()
    v1=compare_v1(curves)
    if a.preflight_only:
        print("A25_NOV4_PUBLICATION_PREFLIGHT_PASS",
              "FAST",7,"D4",8,"FACE",len(transformed),
              "V1",v1["n_comparisons"],flush=True)
        return
    # Re-evaluate the original 256x256 candidate tiles for publication only.
    from scripts.track_r.xp_a24_nov4_fast_cpu import frozen
    from scripts.track_r.xp_a24_nov4_d4_panels_cpu import provenance
    ident,table,hist,n16,n64,cases=frozen()
    p1,p2,grouped,gtmap,by,d4ids,sf,receipt=provenance(a)
    selected=[z["image_id"] for z in dr["panels"]]
    gate(d4ids==selected,"D4_ORIGINAL_SHA_ORDER")
    out.mkdir()
    figure=out/"png_256"
    figure.mkdir()
    panels=[]
    for number,c in enumerate(sorted(
       (z for z in cases["cases"] if z["category"]=="NEVER_GOOD"),
       key=lambda v:(v["cohort"],v["image_id"])),1):
        panels.append(publication_figure_fast(
            figure,c,ident,table,hist,n16,n64,number))
        print("A25_FAST_PAPER_PANEL",number,"/7",flush=True)
    for number,iid in enumerate(d4ids,1):
        panels.append(publication_figure_d4(
            figure,iid,by,p1,p2,grouped,gtmap,number))
        print("A25_D4_PAPER_PANEL",number,"/8",flush=True)
    gate(len(panels)==15,"FIFTEEN")
    csvname="XP_A25_FAST_FACE_CLIP_11520.csv"
    csvpath=out/csvname
    with csvpath.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=NAMES);w.writeheader();w.writerows(transformed)
    copy=theory/csvname
    gate(not copy.exists(),"NO_THEORY_OVERWRITE")
    shutil.copyfile(csvpath,copy)
    gate(sha(copy)==sha(csvpath),"THEORY_EXACT_COPY")
    cfile=out/"XP_A25_A9_2D_FAST_E1_EXACT_CURVES.json"
    vfile=out/"XP_A25_A9_2D_FAST_E1_V1_COVERAGE.json"
    write(cfile,dict(status="A20_A9_2D_FAST_FACE_E1_EXACT_CURVES",
      n_images=180,n_candidates=11520,source_sha256=SOURCE_CSV_SHA,
      curves=curves,no_new_selector=True,no_timing_claim=True))
    write(vfile,v1)
    manifest=out/"XP_A25_NOV4_PUBLICATION_RECEIPT.json"
    write(manifest,dict(status="A20_NOV4_FIFTEEN_PNG_AND_FAST_E1_PUBLISHED",
      source_head=a.expected_head,n_png=15,panels=panels,
      png_original_resolution=256,
      original_panels_sha256=dict(FAST=sha(FPR),D4=sha(DREC)),
      XP_D4_selection_rule="ascending SHA256(image_id UTF-8)",
      FAST7_selection_rule="seven A9 frozen never-GOOD",
      CSV_columns=list(NAMES),n_csv_rows=11520,
      csv_sha256=sha(csvpath),theory_inputs_csv_sha256=sha(copy),
      registered_v1_sha256=V1_SHA,
      exact_curves_sha256=sha(cfile),comparison_sha256=sha(vfile),
      original_face_decisions_sha256=SOURCE_CSV_SHA,
      no_GPU=True,GT_offline_figure_only=True))
    print("A25_NOV4_PUBLICATION_STAGED_PASS",
          "PNG",len(panels),"CSV_SHA",sha(csvpath),
          "CURVES_SHA",sha(cfile),
          "COVERAGE_SHA",sha(vfile),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A25_NOV4_PUBLICATION",exc,flush=True)
        sys.exit(8)
