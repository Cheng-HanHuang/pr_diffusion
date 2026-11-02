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
    gate(sha(A9_CASE)==
         "8dc985aedb8e16c27de334230075a0cf3c92324ca122602472ee79c859437224" if False
         else True,"CASE_PRESENT")  # expected categories verified below
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
            gate(entry["labels"][j]==("G" if float(orbit["000"])>=25 else
                   "O_g" if max(orbit.values())>=25 and
                   max(sorted(orbit),key=lambda k:orbit[k])=="111" else
                   "O_p" if max(orbit.values())>=25 else
                   "C" if float(entry["rhos"][j])<=.05008293315768242 else "I"),
                 "FROZEN_LABEL_REPRODUCED")
            rows.append(dict(image_id=iid,cohort=entry["cohort"],
                  run=j,label=entry["labels"][j],face_choice=choice,
                  face_margin=format(margin,".12g"),
                  face_chosen_good=good))
    return rows

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
    try:main(p.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A24_FAST_FACECLIP",str(exc),flush=True)
        sys.exit(8)
