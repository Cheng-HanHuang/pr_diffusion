#!/usr/bin/env python3
"""A10 first prospective ImageNet FAST D1-P2 CPU comparison.

Read GT and D1-P2 outcomes ONLY after authenticating both preregistered
v2/v2-E1 forecast blobs and the planner registry. New 12-run suffix from
A6, original 4-run prefix from D1-P1, 16-run exact U-statistic panel.
Native CLIP G1/G2 on the new 1200 candidates; no CUDA or new GPU jobs.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,importlib,json,math,os,random,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path

from scripts.track_r.xp_t_sitcom_prepare_cpu import ROOT,BASE
from scripts.track_r.xp_a6_d1p2_contract import D1P2,SHA as D1_SHA,SOURCE_HEAD as D1_HEAD,validate as validate_d1_contract
from scripts.track_r.xp_a6_supervisor import jobs,output_dir,validate as validate_chunk
from scripts.track_r.xp_d1p1_score_cpu import validate_setup as validate_p1,GOOD
from scripts.track_r.xp_a8_d1p1_clip_cpu import (
    model_source,make_text,G1_UP,G1_INV,G2_UP,G2_INV,
    CLIP_SHA)
from scripts.track_r.xp_fast_gpu import frozen_roots,root_folder,validate_complete
from scripts.track_r.xp_night_publish_cpu import (
    RUN as P1_RUN,GEN_HEAD as P1_HEAD,
    EXPECTED_MANIFEST_HASHES as P1_SHA)
from scripts.track_r.xp_night_supervisor import jobs as night_jobs, paths as night_paths
from scripts.track_r.xp_a7_postfreeze_kernel_cpu import exact_panel,rank_prob,compare_interval

WT=Path(__file__).resolve().parents[2]
A6=ROOT/"pr_diffusion_track_r"
A6_HEAD="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
A6_RUN=BASE/"xp_a6_20261027T165112Z"
A6_GENERATOR_SHA="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
PLANNER="4144e0bc2cc073b05260b18781bd620f36e37763"
V2_REF="340dc5dcbcde96df4edc938f1b807faedc18936b"
V2_PATH="docs/track_r/theory/TH1_PREDICTIONS_v2.json"
V2_SHA="8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55"
E1_REF="9cb57f51337f2730db1644d570b0e2612855e728"
E1_PATH="docs/track_r/theory/TH1_PREDICTIONS_v2_E1.json"
E1_SHA="13ff550ca42c0941c430eff75e1f9ecc107dbe3801ee19d355371518980ec406"
P1_CLIP_SHA="0a2989180b67afa6fc9c6fccdffe7680030f4214d7b672ecd4fe50e5b010cf44"
P1_CLIP=WT/"docs/track_r/executors/xp/theory_inputs/XP_D1_P1_K4_CLIP_DECISIONS.csv"
P1_STRINGS=WT/"docs/track_r/executors/xp/theory_inputs/XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv"
P1_STRINGS_SHA="c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1"
REG="docs/track_r/theory/PREDICTIONS_REGISTRY.md"
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")
K=(8,12,16)
VARIANTS=("race","S2","tail","race_tau1")
FIELDS=("image_id","run","label","rho_signed","g1_choice","g1_margin",
        "g1_chosen_good","g2_choice","g2_margin","g2_chosen_good",
        "psnr_raw","psnr_orbit_best")
def need(x,label):
    if not x:raise RuntimeError("STOP_A10_D1P2_"+label)
def git(*parts):
    return subprocess.check_output(["git","-C",str(WT),*parts],timeout=90)
def digest(data):
    return hashlib.sha256(data).hexdigest()
def sha(p):
    p=Path(p)
    need(p.is_file() and not p.is_symlink()
         and ROOT.resolve() in p.resolve().parents
         and not any(t in str(p.resolve()).lower() for t in BANNED),"PATH")
    return digest(p.read_bytes())
def json_x(p,data):
    with p.open("x") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def registered():
    # Must run BEFORE inspecting new reconstruction bytes.
    reg=git("show",PLANNER+":"+REG).decode()
    data={}
    for name,ref,path,want in (("v2",V2_REF,V2_PATH,V2_SHA),
                               ("v2 E1",E1_REF,E1_PATH,E1_SHA)):
        matches=[]
        for line in reg.splitlines():
            if not line.startswith("|"):continue
            cells=[x.strip() for x in line.strip("|").split("|")]
            if len(cells)>=5 and cells[0]==name:matches.append(cells)
        need(len(matches)==1,"REGISTRY_"+name)
        c=matches[0]
        need(c[2].strip("`")==ref and c[3].strip("`")==path
             and want in c[4],"REGISTRY_FIELDS_"+name)
        blob=git("show",ref+":"+path)
        need(digest(blob)==want,"FORECAST_SHA_"+name)
        # Registry itself is frozen on the pre-outcome planner branch.
        data[name]=json.loads(blob)
    need(len(data["v2"]["predictions"])==36 and
         len(data["v2 E1"]["predictions"])==24 and
         data["v2"]["interval_type"]=="predictive_realized_panel" and
         data["v2 E1"]["interval_type"]=="predictive_realized_panel" and
         data["v2 E1"]["base_v2_sha256"]==V2_SHA,
         "FORECAST_PANEL")
    return data
def source_contract():
    need(git("rev-parse","HEAD").decode().strip()==ARGS.expected_head and
         not git("status","--porcelain"),"AUTHOR_GIT")
    a6head=subprocess.check_output(["git","-C",str(A6),"rev-parse","HEAD"],text=True).strip()
    need(a6head==A6_HEAD,"A6_GIT")
    need(sha(D1P2)==D1_SHA,"D1_P2_FROZEN_MANIFEST")
    manifest=json.loads(D1P2.read_text())
    rows=frozen_roots(manifest,D1_HEAD)
    validate_d1_contract(D1P2,D1_SHA,manifest,rows,D1_HEAD,0,1200)
    prep=json.loads((A6_RUN/"XP_A6_PREP.json").read_text())
    state=json.loads((A6_RUN/"XP_A6_CONTROLLER_STATE.json").read_text())
    need(prep["head"]==A6_HEAD and prep["stages"]["D1_P2"]["sha256"]==D1_SHA
         and state["head"]==A6_HEAD and
         len(state["done"])>0 and state["active_jobs"]==[]
         and state["outcomes_scored"] is False,"A6_GENERATION")
    completed=set(state["done"])
    tracked=[j for j in jobs() if j["stage"]=="D1_P2"]
    need(all(j["id"] in completed for j in tracked),"D1_P2_1200_COMPLETE")
    for j in tracked:
        validate_chunk(A6_RUN,A6_HEAD,j)
    need(sha(P1_CLIP)==P1_CLIP_SHA and
         sha(P1_STRINGS)==P1_STRINGS_SHA,"P1_FROZEN_INPUTS")
    p1rows,inputs,sources=validate_p1()
    p2map={}
    for j in tracked:
        folder=output_dir(A6_RUN,j)
        for r in rows[j["start"]:j["start"]+j["count"]]:
            key=(r["image_id"],r["rep"])
            need(key not in p2map,"P2_DUPLICATE")
            p2map[key]=(r,root_folder(folder,r)/"reconstruction.pt")
    p1map={}
    for j in night_jobs():
        if j["stage"]!="D1_P1":continue
        folder=night_paths(P1_RUN,j)
        for r in p1rows[j["start"]:j["start"]+j["count"]]:
            key=(r["image_id"],r["rep"])
            p1map[key]=(r,root_folder(folder,r)/"reconstruction.pt")
    need(len(p1map)==400 and len(p2map)==1200 and
         {k[0] for k in p1map}=={k[0] for k in p2map}==set(sources),
         "PANEL_100x16")
    return p1map,p2map,inputs,sources
def read_p1_clip():
    with P1_CLIP.open(newline="") as f:rr=list(csv.DictReader(f))
    need(len(rr)==400 and len({(r["image_id"],int(r["run"])) for r in rr})==400,
         "P1_CLIP_400")
    return {(r["image_id"],int(r["run"])):r for r in rr}
def threshold():
    with P1_STRINGS.open(newline="") as f:rr=list(csv.DictReader(f))
    need(len(rr)==100,"P1_LABELS")
    val={float(x["tau_signed_good_q95"]) for x in rr}
    need(len(val)==1,"P1_TAU_INCONSISTENT")
    tau=val.pop()
    need(0<tau<1,"P1_TAU")
    return tau
def get_x(path):
    import torch
    blob=torch.load(path,map_location="cpu",weights_only=True)
    x=blob.get("reconstruction")
    need(isinstance(x,torch.Tensor) and tuple(x.shape)==(1,3,256,256)
         and bool(torch.isfinite(x).all()),"X_TENSOR")
    return x.float()
def stats_psnr(x,gt,y,tau):
    from prdiffusion.track_r.physics import to_rgb8,rho_signed
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    x8=to_rgb8(x)
    z=orbit_psnr_channel_mse(x8,gt)
    raw=float(z["000"])
    best=max(sorted(z),key=lambda t:z[t])
    orbit=float(z[best])
    rho=float(rho_signed(x,y)[0])
    need(math.isfinite(rho) and rho>=0,"RHO")
    if raw>=GOOD:label="G"
    elif orbit>=GOOD:label="O" if best=="111" else "P"
    elif rho<=tau:label="C"
    else:label="I"
    return x8,dict(label=label,rho=rho,good_raw=raw>=GOOD,
        good_orbit=orbit>=GOOD,flip_good=float(z["111"])>=GOOD,
        raw=raw,orbit=orbit,best=best)
def conditional_pl_given_labels(panel,k,wO,wC):
    """Exact subset-hypergeometric PL GOOD win, conditional on 16 labels.

    Relative rates: G=1, global/partial ORBIT=wO, CONSISTENT=wC,
    INCONSISTENT=0 (last, as frozen in TH1_RANK_RACE_MATH).
    No phase-2 rank is used in this prediction; no refit.
    """
    need(len(panel)==16 and 1<=k<=16 and
         0<wO<=1 and 0<wC<=1,"PL_CONTRACT")
    counts=Counter(r["label"] for r in panel)
    ng=counts["G"];no=counts["O"]+counts["P"];nc=counts["C"];ni=counts["I"]
    total=math.comb(16,k)
    expectation=0.
    for g in range(min(k,ng)+1):
        for o in range(min(k-g,no)+1):
            for c in range(min(k-g-o,nc)+1):
                i=k-g-o-c
                if not 0<=i<=ni:continue
                chance=(math.comb(ng,g)*math.comb(no,o)*
                        math.comb(nc,c)*math.comb(ni,i))/total
                weight=g+o*wO+c*wC
                expectation+=chance*(g/weight if weight else 0.)
    need(-1e-10<=expectation<=1+1e-10,"PL_VALUE")
    return expectation
def main(a):
    global ARGS
    ARGS=a
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    need(1<=a.batch<=16,"CLIP_BATCH")
    forecasts=registered()
    print("A10_D1P2_V2_AND_V2E1_REGISTRY_SHA_PASS",flush=True)
    p1,p2,inputs,sources=source_contract()
    print("A10_D1P2_A6_ALL_1200_RECEIPTS_PASS",flush=True)
    import numpy as np,torch
    import torchvision.transforms as T
    from PIL import Image
    from prdiffusion.track_r.physics import to_rgb8
    torch.set_num_threads(4)
    clipfile,native,univfd_head=model_source()
    sys.path.insert(0,str(native))
    clip=importlib.import_module("models.clip")
    model,preprocess=clip.load(str(clipfile),device="cpu",jit=False)
    model.eval()
    with torch.inference_mode():
        vectors={key:make_text(model,clip,torch,prompts)
           for key,prompts in (("g1_up",G1_UP),("g1_down",G1_INV),
                               ("g2_up",G2_UP),("g2_down",G2_INV))}
    p1choices=read_p1_clip()
    tau=threshold()
    transform=T.Compose([T.ToTensor(),T.Resize(256),T.CenterCrop(256)])
    per_image=[];new_rows=[];panel_data={}
    with torch.inference_mode():
        for number,iid in enumerate(sorted(sources)):
            src=sources[iid]
            need(sha(src["source_path"])==src["source_sha256"],"IMAGE_SHA")
            with Image.open(src["source_path"]) as im:g=transform(im)
            if g.shape==(1,256,256):g=g.repeat(3,1,1)
            need(tuple(g.shape)==(3,256,256),"GT_FORMAT")
            gt=to_rgb8((g*2-1).unsqueeze(0))
            yi=inputs[iid]
            need(sha(yi["measurement_path"])==yi["measurement_sha256"],"Y_SHA")
            measurement=torch.load(yi["measurement_path"],map_location="cpu",weights_only=True)
            y=measurement["measurement"].float()
            need(tuple(y.shape)==(1,3,384,384),"SIGNED_Y")
            items=[];image_clip=[]
            for rep in range(16):
                row,path=(p1 if rep<4 else p2)[(iid,rep)]
                if rep<4:
                    p1job=next(j for j in night_jobs()
                        if j["stage"]=="D1_P1" and
                           j["start"]<=number*4+rep<j["start"]+j["count"])
                    validate_complete(path.parent,row,P1_HEAD,P1_SHA["D1_P1"])
                else:
                    validate_complete(path.parent,row,A6_HEAD,D1_SHA)
                x=get_x(path)
                x8,metrics=stats_psnr(x,gt,y,tau)
                dat=dict(run=rep,rho=metrics["rho"],
                         good_raw=metrics["good_raw"],
                         good_orbit=metrics["good_orbit"],
                         label=metrics["label"],
                         flip_good=metrics["flip_good"])
                if rep<4:
                    prev=p1choices[(iid,rep)]
                    need(prev["label"]==metrics["label"],"P1_CLASS_MATCH")
                    for arm in ("g1","g2"):
                        dat[arm+"_chosen_good"]=int(prev[arm+"_chosen_good"])
                    items.append(dat)
                    continue
                image=x8[0].permute(1,2,0).numpy()
                original=Image.fromarray(image,"RGB")
                inverted=Image.fromarray(np.rot90(image,2).copy(),"RGB")
                image_clip.extend([preprocess(original),preprocess(inverted)])
                items.append(dat)
            need(len(image_clip)==24,"P2_CLIP_INPUTS")
            features=model.encode_image(torch.stack(image_clip)).float()
            features/=features.norm(dim=-1,keepdim=True)
            for arm in ("g1","g2"):
                up=vectors[arm+"_up"];down=vectors[arm+"_down"]
                z=features@up-features@down
                margin=(z[1::2]-z[::2]).tolist()
                for j,delta in enumerate(margin):
                    rep=j+4;r=items[rep]
                    choice="111" if delta>0 else "000"
                    # Frozen A8 rule: 180-degree inversion only.
                    chosen_good=(r["good_raw"] if choice=="000"
                                 else r["flip_good"])
                    r[arm+"_chosen_good"]=int(chosen_good)
                    r[arm+"_choice"]=choice
                    r[arm+"_margin"]=float(delta)
            need(len(items)==16 and
                 [r["run"] for r in items]==list(range(16)),"PER_IMAGE_16")
            panel_data[iid]=items
            for row in items[4:]:
                new_rows.append({k:(row[k] if k in row else None)
                    for k in ("run","label","rho","g1_choice","g1_margin",
                              "g1_chosen_good","g2_choice","g2_margin","g2_chosen_good")}
                    |{"image_id":iid})
            per_image.append(dict(
                 image_id=iid,n_runs=16,
                 run_labels="".join(r["label"] for r in items),
                 e0_rank_order=",".join(str(r["run"])
                       for r in sorted(items,key=lambda x:(x["rho"],x["run"]))),
                 rho_signed_by_run=";".join(format(r["rho"],".10g") for r in items),
                 g1_chosen_good_by_run="".join(str(r["g1_chosen_good"]) for r in items),
                 g2_chosen_good_by_run="".join(str(r["g2_chosen_good"]) for r in items)))
            if (number+1)%10==0:
                print("A10_D1P2_SCORE_AND_CLIP",number+1,"/100",flush=True)
    need(len(per_image)==100 and len(new_rows)==1200,"OUTPUT_COUNTS")
    grid=[]
    for k in K:
        vals=defaultdict(list)
        for iid,items in sorted(panel_data.items()):
            base=exact_panel(items,k)
            rank=sorted(items,key=lambda r:(r["rho"],r["run"]))
            e1={arm:sum(r[arm+"_chosen_good"]*rank_prob(16,k,j+1)
                        for j,r in enumerate(rank)) for arm in ("g1","g2")}
            for pol,v in {**base,"E1_CLIP":e1["g1"],"E1_CLIP_G2":e1["g2"]}.items():
                vals[pol].append(v)
        for pol,v in sorted(vals.items()):
            grid.append(dict(K=k,policy=pol,observed=sum(v)/len(v),
                             n_images=len(v)))
    observed={(row["K"],row["policy"]):row["observed"] for row in grid}
    comparisons=[]
    for source,pred in (("v2",forecasts["v2"]),("v2 E1",forecasts["v2 E1"])):
        for row in pred["predictions"]:
            pol=row["policy"];key=(row["K"],pol)
            need(key in observed,"FORECAST_OBS_KEY")
            comparison=compare_interval(observed[key],
                 dict(row,interval_type=pred["interval_type"]))
            comparisons.append(dict(forecast_source=source,
                  K=row["K"],policy=pol,variant=row["model_variant"],
                  decision_model=row.get("decision_model"),
                  **comparison))
    need(len(comparisons)==60,"COMPARISON_60")
    # Image-paired composition rates, not independent-root p-values.
    class_diff=[]
    for label in "GOPCI":
        phase1_count=sum(r["label"]==label
                         for items in panel_data.values() for r in items[:4])
        phase2_count=sum(r["label"]==label
                         for items in panel_data.values() for r in items[4:])
        per=[sum(r["label"]==label for r in items[4:])/12-
             sum(r["label"]==label for r in items[:4])/4
             for items in panel_data.values()]
        rng=random.Random(20261028+ord(label))
        obs=sum(per)/len(per)
        flips=8192
        hits=sum(abs(sum(v*(1 if rng.getrandbits(1) else -1) for v in per)/len(per))
                 >=abs(obs)-1e-15 for _ in range(flips))
        class_diff.append(dict(label=label,
          p1_count=phase1_count,p2_count=phase2_count,
          p1_rate=phase1_count/400,p2_rate=phase2_count/1200,
          rate_difference=obs,
          paired_image_signflip_p=(hits+1)/(flips+1)))
    conditional=[]
    weights={
      "race":forecasts["v2"]["rank_weights_primary"],
      "race_tau1":forecasts["v2"]["rank_weights_sensitivity"],
      "S2":{"w_O":1.,"w_C":1.}}
    for k in K:
        for variant,w in weights.items():
            modelval=sum(conditional_pl_given_labels(
                items,k,float(w["w_O"]),float(w["w_C"]))
                for items in panel_data.values())/100
            conditional.append(dict(K=k,variant=variant,
                observed_E0=observed[(k,"E0")],
                frozen_PL_given_labels=modelval,
                model_minus_observed=modelval-observed[(k,"E0")],
                n_images=100))
    never=[dict(image_id=iid,run_labels="".join(r["label"] for r in items),
                n_good=0)
           for iid,items in sorted(panel_data.items())
           if not any(r["good_raw"] for r in items)]
    summary=dict(status="A10_D1P2_PROSPECTIVE_COMPARISON_PASS",
         evaluation_head=a.expected_head,original_generation_head=A6_HEAD,
         frozen_d1_manifest_sha=D1_SHA,planner_registry_commit=PLANNER,
         v2_sha256=V2_SHA,v2_e1_sha256=E1_SHA,
         n_images=100,n_phase1=400,n_phase2=1200,n_panel=1600,
         model_clip_sha256=CLIP_SHA,univfd_head=univfd_head,
         model_variants=list(VARIANTS),K_grid=list(K),tau_good20=tau,
         metrics=grid,forecast_comparisons=comparisons,
         phase_exchangeability=class_diff,
         never_good_images=never,
         E1_method="frozen per-candidate G1 choice; G2 secondary",
         composition_conditional_selection=conditional,
         composition_rule="exact hypergeometric subset distribution with frozen exponential-rate rank weights, I weight zero; conditional on realized 16-run class counts, no model refit",
         B16_F8_outcomes_opened=False,
         EPRIME_outcomes_opened=False,
         prospective_predictions_refitted=False)
    out=Path(a.out)
    need(out.parent==BASE and out.name.startswith("xp_a10_d1p2_")
         and not out.exists() and not out.is_symlink(),"OUT")
    out.mkdir(parents=False,exist_ok=False)
    json_x(out/"XP_A10_D1P2_COMPARISON.json",summary)
    with (out/"XP_A10_D1P2_PER_IMAGE.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(per_image[0]))
        w.writeheader();w.writerows(per_image)
    with (out/"XP_A10_D1P2_CLIP_G1G2.csv").open("x",newline="") as f:
        columns=list(new_rows[0])
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader();w.writerows(new_rows)
    hashes={p.name:sha(p) for p in out.iterdir()}
    json_x(out/"XP_A10_D1P2_RECEIPT.json",
        dict(status=summary["status"],head=a.expected_head,
             sha256=hashes,no_B16_F8_outcome_opened=True))
    print("A10_D1P2_PROSPECTIVE_COMPARISON_PASS",
          "100 images 1600 candidates 60 forecast comparisons",flush=True)
    print("A10_D1P2_OUTPUT="+str(out),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--batch",type=int,default=8)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A10_D1P2_CPU "+str(e),flush=True)
        sys.exit(8)
