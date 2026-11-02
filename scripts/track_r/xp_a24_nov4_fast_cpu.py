#!/usr/bin/env python3
"""A19 Nov4 FFHQ FAST 64-run face CLIP E1, CPU-only, resumable by image.

All 180 x 64 candidate reconstructions are historical A9, not fresh model
outputs. Frozen N2 face prompts and checkpoint; no training or new selector.
The face orientation decision uses only candidate pixels, never GT.
GT is opened separately for offline binary GOOD25 on the chosen orientation.
Source candidate manifests and historical run receipts are independently
verified by A8 historical()/future_inputs() before evaluating.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,importlib,json,math,os,subprocess,sys,traceback
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
A9=BASE/"xp_a9_exchange_20261028T100727Z"
A9_STR=A9/"A9_FAST_64_PER_IMAGE_STRINGS.csv"
A9_SHA="d616dd50f10352e98b891a99b9a153d5d72a93c61e44381827147dcca23f5466"
A9_CASE=A9/"A9_FAST_11_CASES.json"
CSV_FIELDS=("image_id","cohort","run","label","face_choice","face_margin","face_chosen_good")
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")

def gate(ok,why):
    if not ok:raise RuntimeError("STOP_A24_FAST_FACECLIP_"+why)

def sha(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink() and
         ROOT.resolve() in p.resolve().parents and
         not any(x in str(p.resolve()).lower() for x in BANNED),
         "UNSAFE_INPUT_"+p.name)
    h=hashlib.sha256()
    with p.open("rb") as f:
        for z in iter(lambda:f.read(1<<20),b""):h.update(z)
    return h.hexdigest()

def load_json(p):return json.loads(Path(p).read_text())

def emit(p,obj):
    with Path(p).open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def frozen():
    from scripts.track_r.xp_a8_fast_v1_cpu import (
        historical,future_inputs,R07_CATALOG,R07_RECEIPT,R07_SCORES)
    gate(sha(A9_STR)==A9_SHA,"A9_SOURCE_SHA")
    sha(A9_CASE)  # exact local source bytes; receipt records digest
    cases=load_json(A9_CASE)
    gate(cases.get("status")=="A9_FAST_11_CASES_POSTHOC_PASS" and
         cases["n_never_good"]==7 and cases["n_selection_losses"]==4 and
         cases["source_sha256"]==A9_SHA,"A9_CASE_CENSUS")
    with A9_STR.open(newline="") as f:strings=list(csv.DictReader(f))
    gate(len(strings)==180 and
         len({(r["cohort"],r["image_id"]) for r in strings})==180,
         "A9_180_UNIQUE")
    # Historical provenance performs the R07 catalog/score receipt checks.
    identities,old=historical()
    new16,new64=future_inputs()
    catalog=load_json(R07_CATALOG)
    hist={}
    for row in catalog["rows"]:
        if row.get("source")!="S5_r0_fast":continue
        key=(row["measurement_source"],str(row["image_id"]),int(row["run_index"]))
        gate(key not in hist,"HIST_DUPLICATE")
        hist[key]=row
    gate(len(hist)==720 and len(new16)==2160 and len(new64)==8640,
         "RECONSTRUCTION_COVERAGE")
    cm={"B22":"S1_b22","DEV80":"S2_b24_dev80"}
    table={}
    for sr in strings:
        cohort=cm.get(sr["cohort"])
        gate(cohort is not None,"COHORT")
        iid=str(sr["image_id"])
        labs=sr["labels_by_run"].split(";")
        rhos=sr["rho_signed_by_run"].split(";")
        gate(len(labs)==len(rhos)==64 and
             set(labs)<={"G","O_g","O_p","C","I"} and
             all(math.isfinite(float(z)) for z in rhos),"A9_CLASS_FIELDS")
        gate((cohort,iid) in identities,"ID_GT_MATCH")
        table[(cohort,iid)]=dict(labels=labs,rhos=rhos,
                                cohort=sr["cohort"],image_id=iid)
    gate(len(table)==180 and set(table)==set(identities),"SOURCE_IDENTITY")
    return identities,table,hist,new16,new64,cases

def candidate(cohort,iid,rep,hist,new16,new64):
    import torch
    from prdiffusion.track_r.physics import to_rgb8
    from scripts.track_r.r0_4_score_gpu import candidate_tensor
    from scripts.track_r.xp_a8_fast_v1_cpu import safe
    key=(cohort,iid,rep)
    if rep<4:
        gate(key in hist,"HIST_ROOT_MISSING")
        return candidate_tensor(hist[key])
    source=new16 if rep<16 else new64
    gate(key in source,"NATIVE_ROOT_MISSING")
    row,path=source[key]
    gate(int(row["rep"])==rep and str(row["image_id"])==iid,
         "NATIVE_ROOT_UID")
    x=torch.load(safe(path),map_location="cpu",weights_only=True)
    gate(isinstance(x,dict) and isinstance(x.get("reconstruction"),torch.Tensor),
         "NATIVE_PT_CONTAINER")
    x=x["reconstruction"].float()
    gate(tuple(x.shape)==(1,3,256,256) and bool(torch.isfinite(x).all()),
         "NATIVE_PT_SHAPE")
    return to_rgb8(x)

def text_pair(model,clip,torch):
    from prdiffusion.track_r.r0n_clip_compat import (
        encode_text_univfd_tuple_compat)
    from scripts.track_r.r0n_2_clip_orientation import PROMPTS_UP,PROMPTS_INV
    prompts=list(PROMPTS_UP+PROMPTS_INV)
    texts=encode_text_univfd_tuple_compat(
        model,clip.tokenize(prompts).to("cpu")).float()
    texts=texts/texts.norm(dim=-1,keepdim=True)
    u=texts[:len(PROMPTS_UP)].mean(dim=0)
    d=texts[len(PROMPTS_UP):].mean(dim=0)
    return u/u.norm(),d/d.norm()

def score_image(key,entry,ident,hist,new16,new64,model,clip,preprocess,torch,up,down):
    from PIL import Image
    import numpy as np
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    from prdiffusion.track_r.physics import to_rgb8
    from scripts.track_r.r0_2_g0_cpu import gt_source
    cohort,iid=key
    gtpath,fmt,_,_=ident
    gate(sha(gtpath) is not None,"GT_EXISTS")
    gt=to_rgb8(gt_source({"gt_path":gtpath,"gt_format":fmt}).float())
    rows=[]
    with torch.inference_mode():
        for j in range(64):
            x=candidate(cohort,iid,j,hist,new16,new64)
            gate(tuple(x.shape)==(1,3,256,256),"X8_SHAPE")
            arr=x[0].permute(1,2,0).numpy()
            base=Image.fromarray(arr,"RGB")
            rotated=Image.fromarray(np.rot90(arr,2).copy(),"RGB")
            e=model.encode_image(torch.stack(
                [preprocess(base),preprocess(rotated)])).float()
            e=e/e.norm(dim=-1,keepdim=True)
            scores=e@up-e@down
            margin=float(scores[1]-scores[0])
            gate(math.isfinite(margin),"FACE_MARGIN_FINITE")
            choice="111" if margin>0 else "000"
            # GT evaluation is AFTER GT-free choice and never fed to CLIP.
            orbit=orbit_psnr_channel_mse(x,gt)
            good=int(float(orbit[choice])>=25.)
            gate((float(orbit["000"])>=25.) ==
                 (entry["labels"][j]=="G"),
                 "FROZEN_RAW_GOOD_LABEL_REPRODUCED")
            rows.append(dict(image_id=iid,cohort=entry["cohort"],
                  run=j,label=entry["labels"][j],face_choice=choice,
                  face_margin=format(margin,".12g"),
                  face_chosen_good=good))
    return rows


def fast_seven_panels(out,identities,table,hist,new16,new64,cases):
    """Anonymized GT / lowest signed-rho / orbit-oracle panels; CPU only."""
    from PIL import Image,ImageDraw,ImageFont
    from scripts.track_r.r0_2_g0_cpu import gt_source
    from prdiffusion.track_r.physics import to_rgb8,apply_orbit
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    never=sorted([c for c in cases["cases"]
                  if c["category"]=="NEVER_GOOD"],
                 key=lambda r:(r["cohort"],r["image_id"]))
    gate(len(never)==7,"SEVEN_FROZEN_CASES")
    dest=out/"FAST7_PANELS"
    if not dest.exists():dest.mkdir()
    receipt=dest/"FAST7_PANEL_RECEIPT.json"
    if receipt.exists():
        data=load_json(receipt)
        gate(data["status"]=="A19_FAST7_NEVER_GOOD_PANELS_PASS"
             and len(data["panels"])==7
             and all(sha(dest/r["filename"])==r["sha256"]
                     for r in data["panels"]),
             "FAST7_EXISTING_PANEL_RECEIPT")
        return
    outputs=[]
    mapping={"B22":"S1_b22","DEV80":"S2_b24_dev80"}
    for z,case in enumerate(never,1):
        key=(mapping[case["cohort"]],str(case["image_id"]))
        gate(key in identities and key in table,"NEVER_GOOD_ID")
        item=table[key]
        gate(not any(t=="G" for t in item["labels"]),"CASE_NEVER_GOOD")
        gtpath,fmt,_,_=identities[key]
        g8=to_rgb8(gt_source({"gt_path":gtpath,
                              "gt_format":fmt}).float())
        lowest=min(range(64),key=lambda j:(float(item["rhos"][j]),j))
        xlow=candidate(*key,lowest,hist,new16,new64)
        best=None
        for j in range(64):
            x=candidate(*key,j,hist,new16,new64)
            scores=orbit_psnr_channel_mse(x,g8)
            for code,val in scores.items():
                if best is None or (-val,j,code)<(-best[0],best[1],best[2]):
                    best=(float(val),j,code,x)
        gate(best is not None and best[0]<25.,
             "NEVER_ORBIT_GOOD_EXPECTATION")
        xbest=apply_orbit(best[3],best[2])
        views=[g8,xlow,xbest]
        titles=("Reference","Min residual","Orbit oracle")
        width=3*192+4*8
        image=Image.new("RGB",(width,224),(255,255,255))
        draw=ImageDraw.Draw(image)
        for index,(x,title) in enumerate(zip(views,titles)):
            panel=Image.fromarray(x[0].permute(1,2,0).numpy(),"RGB")
            panel=panel.resize((192,192),Image.Resampling.BILINEAR)
            ox=8+index*200
            image.paste(panel,(ox,24))
            draw.text((ox,7),title,fill=(0,0,0))
        filename=f"FAST_never_good_{z:02d}.png"
        dst=dest/filename
        image.save(dst,optimize=True)
        gate(dst.stat().st_size<2_000_000,"PANEL_OVER_2MB")
        outputs.append(dict(filename=filename,sha256=sha(dst),
            cohort=case["cohort"],image_id=case["image_id"],
            min_rho_run=lowest,orbit_oracle_run=best[1],
            orbit_oracle_orientation=best[2],
            orbit_oracle_psnr=best[0],anonymized=True))
    emit(receipt,dict(status="A19_FAST7_NEVER_GOOD_PANELS_PASS",
        n_cases=7,source_census_sha256=sha(A9_CASE),
        source_A9_sha256=A9_SHA,panels=outputs,
        no_model_inference=True,GT_offline_only=True,
        no_new_selector=True))
    print("A24_FAST7_PANELS_PASS",7,"MAX_BYTES",
          max((dest/r["filename"]).stat().st_size for r in outputs),
          flush=True)

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()==a.expected_head
        and not subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True).strip(),
        "EXECUTOR_HEAD")
    identities,table,hist,new16,new64,cases=frozen()
    if a.preflight_only:
        print("A24_FAST_FACECLIP_SOURCE_PREFLIGHT_PASS",
              "images",len(table),"roots",11520,flush=True)
        return
    out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a24_nov4_fast_") and
         not out.is_symlink(),"OUTPUT_NAMESPACE")
    if not out.exists():
        out.mkdir(mode=0o700)
        (out/"image_chunks").mkdir()
    else:
        gate(a.resume and (out/"image_chunks").is_dir(),
             "EXPLICIT_RESUME_REQUIRED")
    fast_seven_panels(out,identities,table,hist,new16,new64,cases)
    if a.panels_only:
        return
    from scripts.track_r.xp_a8_d1p1_clip_cpu import model_source
    import torch
    torch.set_num_threads(4)
    clipfile,native,source_head=model_source()
    sys.path.insert(0,str(native))
    clip=importlib.import_module("models.clip")
    model,preprocess=clip.load(str(clipfile),device="cpu",jit=False)
    model.eval()
    up,down=text_pair(model,clip,torch)
    for index,key in enumerate(sorted(table)):
        p=out/"image_chunks"/("%03d.json"%index)
        if p.exists():
            d=load_json(p)
            gate(d.get("status")=="A24_FAST_FACECLIP_IMAGE_PASS"
                 and d.get("key")==list(key)
                 and len(d.get("rows",[]))==64
                 and [int(x["run"]) for x in d["rows"]]==list(range(64)),
                 "RESUME_IMAGE_RECEIPT")
            continue
        rows=score_image(key,table[key],identities[key],hist,new16,new64,
                         model,clip,preprocess,torch,up,down)
        emit(p,dict(status="A24_FAST_FACECLIP_IMAGE_PASS",
                    key=list(key),rows=rows))
        if (index+1)%10==0:
            print("A24_FAST_FACECLIP_DONE_IMAGES",index+1,"/180",flush=True)
    results=[]
    for index,key in enumerate(sorted(table)):
        path=out/"image_chunks"/("%03d.json"%index)
        chunk=load_json(path)
        gate(chunk["key"]==list(key) and len(chunk["rows"])==64,
             "FINAL_CHUNK_HASH")
        results.extend(chunk["rows"])
    gate(len(results)==11520,"ALL11520")
    output=out/"A24_FAST_11520_FACE_CLIP_DECISIONS.csv"
    gate(not output.exists(),"OUTPUT_ALREADY_FINALIZED")
    with output.open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS)
        writer.writeheader();writer.writerows(results)
    receipt=out/"A24_FAST_FACECLIP_RECEIPT.json"
    emit(receipt,dict(status="A19_A9_2D_FAST11520_FACECLIP_E1_SCORED",
       expected_head=a.expected_head,
       n_images=180,n_candidates=11520,
       prompt_model="N2_FFHQ_FACE",model_sha256=sha(clipfile),
       native_clip_head=source_head,A9_source_sha256=A9_SHA,
       source_census_sha256=sha(A9_CASE),
       decisions_sha256=sha(output),
       GT_solely_offline_metric_evaluation=True,
       no_method_change=True,clean_gpu_required=False,
       no_GPU_used=True))
    print("A24_FAST_11520_FACE_CLIP_PASS",sha(output),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--panels-only",action="store_true")
    try:main(p.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A24_FAST_FACECLIP",str(exc),flush=True)
        sys.exit(8)
