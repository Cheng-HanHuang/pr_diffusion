#!/usr/bin/env python3
"""CPU-only N2 face-set CLIP orientation, 2,880 B16 candidates.

GT is NOT read during scoring. Chosen GOOD binary is attached from an
already verified offline-scoring CSV after CLIP decisions are frozen.
Frozen prompts N2, pinned CLIP ViT-L/14 weights and UnivFD weights.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,traceback
from pathlib import Path
from scripts.track_r.xp_b16_v1_v3_admission_cpu import WT,BASE,git,check,sha
from scripts.track_r.xp_a8_d1p1_clip_cpu import model_source
from scripts.track_r.r0_4_score_gpu import candidate_tensor
from prdiffusion.track_r.physics import to_rgb8
from scripts.track_r.xp_a10_b16_gate_prepare_cpu import safe
from scripts.track_r.r0n_2_clip_orientation import PROMPTS_UP,PROMPTS_INV,CLIP_SHA
FIELDS=("image_id","run","label","f_choice","f_margin","f_chosen_good",
        "cohort")
def main(a):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    check(git("rev-parse","HEAD").decode().strip()==a.expected_head
          and not git("status","--porcelain"),"HEAD")
    report=Path(a.score_dir)
    rec=json.loads(safe(report/"XP_B16_2880_SCORING_RECEIPT.json").read_text())
    path=report/"XP_B16_2880_SCORED_ROOTS.csv"
    check(rec["status"]=="XP_B16_16RUN_2880_CPU_SCORED_PASS"
          and sha(path)==rec["per_candidate_sha256"],"B16_SCORE_SOURCE_SHA")
    with safe(path).open(newline="") as f:rows=list(csv.DictReader(f))
    check(len(rows)==2880 and
          len({(x["cohort"],x["image_id"],int(x["run"])) for x in rows})==2880,
          "B16_2880_CLIP_GRID")
    clipfile,native,univfd=model_source()
    import torch,importlib,numpy as np
    from PIL import Image
    from prdiffusion.track_r.r0n_clip_compat import encode_text_univfd_tuple_compat
    torch.set_num_threads(4)
    sys.path.insert(0,str(native))
    clip=importlib.import_module("models.clip")
    model,preprocess=clip.load(str(clipfile),device="cpu",jit=False)
    model=model.eval()
    with torch.inference_mode():
        toks=clip.tokenize(list(PROMPTS_UP+PROMPTS_INV)).to("cpu")
        text=encode_text_univfd_tuple_compat(model,toks).float()
        text=text/text.norm(dim=-1,keepdim=True)
        u=text[:3].mean(dim=0);d=text[3:].mean(dim=0)
        u=u/u.norm();d=d/d.norm()
    out=Path(a.out)
    check(out.parent==BASE and out.name.startswith("xp_b16_faceclip_")
          and not out.exists() and not out.is_symlink(),"NEW_OUT")
    out.mkdir()
    result=[]
    from prdiffusion.track_r.physics import load_png_rgb8
    with torch.inference_mode():
        for start in range(0,len(rows),a.batch):
            subset=rows[start:start+a.batch];tensor=[]
            for r in subset:
                p=safe(Path(r["reconstruction_path"]))
                check(sha(p)==r["reconstruction_sha256"],"RECON_SHA")
                if r["reconstruction_format"]=="pt":
                    x=torch.load(p,map_location="cpu",weights_only=True)["reconstruction"].float()
                    check(tuple(x.shape)==(1,3,256,256) and
                          bool(torch.isfinite(x).all()),"RECON_TENSOR")
                    x8=to_rgb8(x)
                elif r["reconstruction_format"]=="png":
                    x8=load_png_rgb8(p)
                else:
                    raise RuntimeError("STOP_UNKNOWN_IMAGE_FORMAT")
                arr=x8[0].permute(1,2,0).numpy()
                upright=Image.fromarray(arr,"RGB")
                inverted=Image.fromarray(np.rot90(arr,2).copy(),"RGB")
                tensor.extend([preprocess(upright),preprocess(inverted)])
            embedding=model.encode_image(torch.stack(tensor)).float()
            embedding/=embedding.norm(dim=-1,keepdim=True)
            scores=embedding@u-embedding@d
            margins=(scores[1::2]-scores[::2]).tolist()
            for row,margin in zip(subset,margins):
                check(math.isfinite(float(margin)),"CLIP_NONFINITE")
                choose="111" if margin>0 else "000"
                chosen=int(row["flip_good"]) if choose=="111" else int(float(row["psnr_raw"])>=25)
                result.append(dict(image_id=row["image_id"],
                   run=int(row["run"]),label=row["label"],
                   f_choice=choose,f_margin=format(margin,".10g"),
                   f_chosen_good=chosen,cohort=row["cohort"]))
            if len(result)%160==0:
                print("B16_N2_FACECLIP_CPU",len(result),"/2880",flush=True)
    check(len(result)==2880,"COMPLETE_FACECLIP")
    file=out/"XP_B16_16RUN_FACE_CLIP_DECISIONS.csv"
    with file.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader();w.writerows(result)
    receipt=out/"XP_B16_16RUN_FACE_CLIP_RECEIPT.json"
    payload=dict(status="B16_2880_FROZEN_N2_FACECLIP_PASS",
       n_candidates=2880,clip_checkpoint_sha256=CLIP_SHA,
       univfd_head=univfd,model_scoring_device="cpu",
       prompts_up=list(PROMPTS_UP),prompts_inverted=list(PROMPTS_INV),
       orientation_rule="rotate 180 iff u(R111x)>u(x)",
       csv_sha256=sha(file),source_scored_sha256=rec["per_candidate_sha256"],
       GT_used_by_CLIP=False,
       chosen_good_joined_from_frozen_offline_scoring=True)
    with receipt.open("x") as f:
        json.dump(payload,f,indent=2,sort_keys=True);f.write("\n")
    print("B16_N2_FACECLIP_2880_PASS CSV_SHA",sha(file),flush=True)
    print("B16_N2_FACECLIP_OUTPUT",out,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--score-dir",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--batch",type=int,default=8)
    try:
        a=p.parse_args()
        check(1<=a.batch<=16,"BATCH")
        main(a)
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: B16_FACECLIP "+str(e),flush=True)
        sys.exit(8)
