#!/usr/bin/env python3
"""A8.2 D1-P1 K4 ImageNet CLIP G1/G2 orientation decisions, CPU-only.

Uses N2's pinned ViT-L/14 + UnivFD-compatible text tower and strict
choose-111 iff u(R111*x)>u(x), with new fixed G1/G2 text prompts.
Only D1_P1 (100 images x 4 reps). NEVER opens D1_P2, A16 or A64.
GT used offline solely to compute binary chosen-good labels (Good20).
CSV is not auto-published, and neither per-run PSNR nor future outcomes
are emitted. Exact registered D1_P1 A8 labels must match canonical GT.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib,json,math,os,subprocess,sys,traceback
from collections import Counter
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
A7WT=ROOT/"pr_diffusion_track_r_a7_cpu"
ORIGINAL_A6_HEAD="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
CLIP_SHA="b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
D1_LABEL_FILE=A7WT/"docs/track_r/executors/xp/theory_inputs/XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv"
D1_LABEL_SHA="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"
G1_UP=("a photo.",)
G1_INV=("an upside-down photo.",)
G2_UP=("a photo.","a photo of a scene.","a photo of an object.",
       "a good photo.","a natural photo.","a photo of something.")
G2_INV=("an upside-down photo.","an upside-down photo of a scene.",
        "an upside-down photo of an object.","a photo that is upside down.",
        "a photo rotated by 180 degrees.","an inverted photo.")
FIELDS=("image_id","run","label","g1_choice","g1_margin",
        "g1_chosen_good","g2_choice","g2_margin","g2_chosen_good")
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")
MODEL_CONFIG=WT/"configs/track_r/pac_paths_track_r_v1.json"

def safe(path):
    p=Path(path)
    r=p.resolve(strict=False)
    if (not p.is_absolute() or not p.is_file() or p.is_symlink()
        or ROOT.resolve() not in r.parents
        or any(x in str(r).lower() or x in str(p).lower() for x in BANNED)):
        raise RuntimeError("STOP_A8_CLIP_BAD_SOURCE "+str(p))
    return p

def sha(p):
    h=hashlib.sha256()
    with safe(Path(p)).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def check_heads(expected):
    for repo,ref in ((A7WT,expected),(WT,ORIGINAL_A6_HEAD)):
        actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],
                                       text=True).strip()
        dirty=subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],
                                       text=True).strip()
        if actual!=ref or dirty:raise RuntimeError("STOP_A8_CLIP_GIT_DRIFT "+str(repo))

def source_and_rows():
    from scripts.track_r.xp_d1p1_score_cpu import (
        validate_setup,GEN_HEAD,EXPECTED_MANIFEST_HASHES,RUN)
    from scripts.track_r import xp_night_supervisor as night
    from scripts.track_r.xp_fast_gpu import root_folder,validate_complete
    rows,inputs,sources=validate_setup()
    indexed=[]
    for i,row in enumerate(rows):
        if row["rep"]!=i%4 or row["image_id"] not in sources:
            raise RuntimeError("STOP_A8_CLIP_ROW_ORDER")
        job=next((j for j in night.jobs() if j["stage"]=="D1_P1" and
                  j["start"]<=i<j["start"]+j["count"]),None)
        if job is None:raise RuntimeError("STOP_A8_CLIP_UNTRACKED_ROOT")
        folder=root_folder(night.paths(RUN,job),row)
        receipt=validate_complete(folder,row,GEN_HEAD,
                      EXPECTED_MANIFEST_HASHES["D1_P1"])
        x=safe(folder/"reconstruction.pt")
        if sha(x)!=receipt["reconstruction_pt_sha256"]:
            raise RuntimeError("STOP_A8_CLIP_X_SHA")
        indexed.append((row,x))
    if len(indexed)!=400:raise RuntimeError("STOP_A8_CLIP_400")
    return indexed,sources,inputs

def label_map():
    if sha(D1_LABEL_FILE)!=D1_LABEL_SHA:
        raise RuntimeError("STOP_A8_CLIP_FROZEN_D1_LABELS_SHA")
    with D1_LABEL_FILE.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=100 or len({r["image_id"] for r in rows})!=100:
        raise RuntimeError("STOP_A8_CLIP_LABEL_100")
    out={}
    for row in rows:
        if row["family"]!="FAST" or row["n_runs"]!="4" or len(row["run_labels"])!=4:
            raise RuntimeError("STOP_A8_CLIP_LABEL_SCHEMA")
        for rep in range(4):
            out[(row["image_id"],rep)]=row["run_labels"][rep]
    return out

def model_source():
    c=json.loads(safe(MODEL_CONFIG).read_text())
    clip_file=safe(c["models"]["clip_vit_l14"]["path"])
    if sha(clip_file)!=CLIP_SHA:
        raise RuntimeError("STOP_A8_CLIP_MODEL_SHA")
    native=Path(c["external"]["univfd"]["path"])
    if native.is_symlink() or not native.is_dir():
        raise RuntimeError("STOP_A8_CLIP_NATIVE_REPO")
    git=subprocess.check_output(["git","-C",str(native),"rev-parse","HEAD"],
                                text=True).strip()
    if git!=c["external"]["univfd"]["commit"]:
        raise RuntimeError("STOP_A8_CLIP_NATIVE_COMMIT")
    return clip_file,native,git

def make_text(model,clip,torch,prompts):
    from prdiffusion.track_r.r0n_clip_compat import encode_text_univfd_tuple_compat
    tokens=clip.tokenize(list(prompts)).to("cpu")
    t=encode_text_univfd_tuple_compat(model,tokens).float()
    t=t/t.norm(dim=-1,keepdim=True)
    mean=t.mean(dim=0)
    return mean/mean.norm()

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A8_CLIP_CUDA_NOT_DISABLED")
    if not 1<=a.batch<=16:raise RuntimeError("STOP_A8_CLIP_BATCH")
    check_heads(a.expected_head)
    rows,sources,_=source_and_rows()
    labels=label_map()
    clipfile,native,git=model_source()
    if ({(r["image_id"],r["rep"]) for r,_ in rows}!=set(labels)
        or len(rows)!=400):
        raise RuntimeError("STOP_A8_CLIP_LABEL_INPUT_MISMATCH")
    print("A8_CLIP_D1P1_NATIVE_PREFLIGHT_PASS roots=400 images=100",
          "clip_sha256="+CLIP_SHA,"univfd_head="+git,flush=True)
    if a.preflight_only:return
    out=Path(a.out)
    if (out.parent!=BASE or not out.name.startswith("xp_a8_d1p1_clip_")
        or out.exists() or out.is_symlink()):
        raise RuntimeError("STOP_A8_CLIP_OUT_NAMESPACE")
    import torch,numpy as np,torchvision.transforms as T
    from PIL import Image
    from prdiffusion.track_r.physics import to_rgb8,psnr_rgb8
    torch.set_num_threads(min(torch.get_num_threads(),4))
    sys.path.insert(0,str(native))
    clip=importlib.import_module("models.clip")
    model,preprocess=clip.load(str(clipfile),device="cpu",jit=False)
    model=model.eval()
    with torch.inference_mode():
        u1=make_text(model,clip,torch,G1_UP)
        d1=make_text(model,clip,torch,G1_INV)
        u2=make_text(model,clip,torch,G2_UP)
        d2=make_text(model,clip,torch,G2_INV)
    transform=T.Compose([T.ToTensor(),T.Resize(256),T.CenterCrop(256)])
    cached={}
    answers=[]
    with torch.inference_mode():
        for start in range(0,400,a.batch):
            subset=rows[start:start+a.batch]
            batch=[]
            rgba=[]
            for r,path in subset:
                data=torch.load(path,map_location="cpu",weights_only=True)
                x=data["reconstruction"].float()
                if x.shape!=(1,3,256,256) or not bool(torch.isfinite(x).all()):
                    raise RuntimeError("STOP_A8_CLIP_BAD_RECON")
                x8=to_rgb8(x)
                im=x8[0].permute(1,2,0).numpy()
                orig=Image.fromarray(im,"RGB")
                rot=Image.fromarray(np.rot90(im,2).copy(),"RGB")
                batch.extend([preprocess(orig),preprocess(rot)])
                iid=r["image_id"]
                if iid not in cached:
                    src=sources[iid]
                    if sha(src["source_path"])!=src["source_sha256"]:
                        raise RuntimeError("STOP_A8_CLIP_GT_SHA")
                    with Image.open(safe(src["source_path"])) as gtimg:
                        g=transform(gtimg)
                    if g.shape==(1,256,256):
                        g=g.repeat(3,1,1)
                    if g.shape!=(3,256,256):
                        raise RuntimeError("STOP_A8_CLIP_GT_RGB")
                    cached[iid]=to_rgb8((g*2-1).unsqueeze(0))
                g8=cached[iid]
                raw=float(psnr_rgb8(x8,g8)[0])
                flip=float(psnr_rgb8(torch.flip(x8,(-2,-1)),g8)[0])
                original_label=labels[(iid,r["rep"])]
                if (raw>=20)!=(original_label=="G"):
                    raise RuntimeError("STOP_A8_CLIP_LABEL_DISAGREEMENT "+str((iid,r["rep"])))
                rgba.append((r,original_label,raw>=20,flip>=20))
            features=model.encode_image(torch.stack(batch)).float()
            features/=features.norm(dim=-1,keepdim=True)
            margins={}
            for name,up,dn in (("g1",u1,d1),("g2",u2,d2)):
                scores=(features@up)-(features@dn)
                margins[name]=(scores[1::2]-scores[::2]).tolist()
            for j,(r,label,rawgood,flipgood) in enumerate(rgba):
                record={"image_id":r["image_id"],"run":r["rep"],"label":label}
                for key in ("g1","g2"):
                    margin=float(margins[key][j])
                    if not math.isfinite(margin):
                        raise RuntimeError("STOP_A8_CLIP_BAD_MARGIN")
                    chosen="111" if margin>0 else "000"
                    record[key+"_choice"]=chosen
                    record[key+"_margin"]=format(margin,".10g")
                    record[key+"_chosen_good"]=int(flipgood if chosen=="111" else rawgood)
                answers.append(record)
            if (start+len(subset))%25==0 or start==0:
                print("A8_CLIP_D1P1_PROGRESS",len(answers),"/400",flush=True)
    if len(answers)!=400 or len({(x["image_id"],x["run"]) for x in answers})!=400:
        raise RuntimeError("STOP_A8_CLIP_OUTPUT_400")
    # Separate machine artifacts; only the requested CLIP binary decisions
    # and margins are published, no GT metric is included.
    out.mkdir(parents=True,exist_ok=False)
    csvfile=out/"XP_D1_P1_K4_CLIP_DECISIONS.csv"
    with csvfile.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader();w.writerows(answers)
    receipt={
        "status":"XP_D1_P1_K4_CLIP_DECISIONS_PASS",
        "generation_head":"8d9bfed0fa62735791987a24791cfb97123540d8",
        "analysis_head":a.expected_head,"n_images":100,"n_candidates":400,
        "run_indices":[0,1,2,3],"model":"CLIP ViT-L/14",
        "model_sha256":CLIP_SHA,"native_univfd_head":git,
        "scoring_device":"cpu","gt_used_for_binary_chosen_good_only":True,
        "D1_P2_outcomes_opened":False,
        "D1_P1_frozen_labels_sha256":D1_LABEL_SHA,
        "g1_prompts_upright":list(G1_UP),"g1_prompts_inverted":list(G1_INV),
        "g2_prompts_upright":list(G2_UP),"g2_prompts_inverted":list(G2_INV),
        "rule":"choose 111 iff u(R111*x)>u(x), else 000",
        "csv_sha256":sha(csvfile),
        "g1_good":sum(r["g1_chosen_good"] for r in answers),
        "g2_good":sum(r["g2_chosen_good"] for r in answers),
        "g1_111":sum(r["g1_choice"]=="111" for r in answers),
        "g2_111":sum(r["g2_choice"]=="111" for r in answers)}
    with (out/"XP_D1_P1_K4_CLIP_RECEIPT.json").open("x") as f:
        json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("A8_D1P1_CLIP_PASS",len(answers),"csv_sha",receipt["csv_sha256"],flush=True)
    print("A8_D1P1_CLIP_OUT="+str(out),flush=True)
    print("D1_P2_OUTCOMES_UNTOUCHED=true",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out")
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A8_D1P1_CLIP "+str(e),flush=True)
        sys.exit(8)
