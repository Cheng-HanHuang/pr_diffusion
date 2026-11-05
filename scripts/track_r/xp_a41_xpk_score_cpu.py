#!/usr/bin/env python3
"""A24 XP-K CPU-only replay gate and frozen 400-run scientific readout.

GT enters ONLY here; native GPU manifests do not include GT/candidate paths.
Original historical A4 identity gate: abs(raw PSNR delta)<=0.1 dB AND
identical frozen native label, on two original-parameter B22 roots/family.
Calibration: G/O(global)/P(partial)/C/I, rho/sigma, rho_star, Δ/D/C,
first-run Good25, E1_CLIP G1, O_orbit, paired historical baseline and
genuine TH2 prereg v1/v1.1 directions/abstentions (no invented predictions).
No inverse solver run, no GPU import, no reranking of native candidates.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib,json,math,os,statistics,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
ORIG_FAMILIES=("SITCOM","FAST","DAPS")
FROZEN_V1=WT/"configs/track_r/xpk_mech_v1_frozen.json"
FROZEN_V11=WT/"configs/track_r/xpk_mech_v1_1_frozen.json"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def gate(ok,why):
    if not ok:raise RuntimeError("STOP_XPK_SCORE_"+why)
def sha(p):
    p=Path(p);h=hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):h.update(buf)
    return h.hexdigest()
def read(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink(),"SOURCE_MISSING_"+str(p))
    return json.loads(p.read_text())
def git(*args):
    return subprocess.check_output(["git","-C",str(WT),*args],text=True,timeout=40).strip()
def write(path,obj):
    with Path(path).open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
def safe(p):
    p=Path(p)
    real=p.resolve()
    gate(p.is_absolute() and p.is_file() and not p.is_symlink()
         and ROOT.resolve() in real.parents and
         not any(x in str(p).lower() or x in str(real).lower() for x in BAD),
         "UNSAFE_INPUT_"+str(p))
    return p
def q(values,percent):
    import numpy as np
    return float(np.quantile(values,percent,method="linear")) if values else None
def summary(values):
    return dict(n=len(values),median=q(values,.5),
                q25=q(values,.25),q75=q(values,.75),
                mean=float(statistics.mean(values)) if values else None)
def canonical_gt(ref):
    from scripts.track_r.r0_2_g0_cpu import gt_source
    from prdiffusion.track_r.physics import to_rgb8
    gt=gt_source(dict(gt_path=str(safe(ref["gt_path"])),
                      gt_format=ref["gt_format"]))
    gate(tuple(gt.shape)==(1,3,256,256),"GT_SHAPE")
    return gt.detach().float(),to_rgb8(gt.detach().float())
def signed_y(path,digest):
    from scripts.track_a.a15_fastdips_preflight import load_tensor
    gate(sha(safe(path))==digest,"SIGNED_Y_SOURCE_HASH")
    y=load_tensor(path,("measurement","y","observation"),(1,3,384,384))
    gate(tuple(y.shape)==(1,3,384,384),"SIGNED_Y_SHAPE")
    return y.float()
def m64(x):
    """A=(unclipped signed FFT magnitude) in complex128 for A5.5 diagnostics."""
    import torch
    import torch.nn.functional as F
    gate(tuple(x.shape)==(1,3,256,256),"M64_SHAPE")
    z=F.pad((x.double()+1.)*.5,(64,64,64,64)).to(torch.complex128)
    return torch.fft.fftshift(torch.fft.fftn(
        torch.fft.ifftshift(z,dim=(-2,-1)),dim=(-2,-1),norm="ortho"),
        dim=(-2,-1)).abs()
def features(x8,g8,y,gtmodel,astar):
    import torch
    from prdiffusion.track_r.physics import from_rgb8,rho_signed
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse,label_candidate
    orbit=orbit_psnr_channel_mse(x8,g8)
    model=from_rgb8(x8)
    rho=float(rho_signed(model,y)[0])
    xm=m64(model)
    y64=y.double()
    star_res=astar-y64
    x_res=xm-y64
    sigma=.05
    target=(star_res.square().sum().item()/star_res.numel())**.5
    meas=(x_res.square().sum().item()/x_res.numel())**.5
    difference=xm-astar
    DD=float(difference.square().sum().item()/sigma**2)
    CC=float(((y64-astar)*difference).sum().item()/sigma**2)
    DEL=float((x_res.square().sum()-star_res.square().sum()).item()/sigma**2)
    gate(abs(DEL-(DD-2*CC))<=max(1e-7,abs(DEL)*1e-9),
         "A55_EXACT_RESIDUAL_IDENTITY")
    return dict(psnr_raw=orbit["000"],psnr_orbit_best=max(orbit.values()),
                psnr_by_orbit=orbit,s_gt=max(orbit,key=orbit.get),
                rho_signed_raw=rho,rho64=meas,rho_star64=target,
                rho_over_sigma64=meas/sigma,rho_below_rho_star=(meas<target),
                Delta=DEL,D=DD,C=CC)
def verify_root(run,m,rec,family,phase):
    d=run/"gpu"/family/phase/rec["image_id"]/f"run{int(rec['run']):02d}"
    path=d/"XPK_ROOT_COMPLETE.json"
    from scripts.track_r.xp_a41_xpk_native_gpu import complete_valid
    gate(complete_valid(d.parent,rec,family,phase),
         "INCOMPLETE_NATIVE_RECEIPT_"+rec["uid"])
    gate(read(path)["code_head"]==m["head"],"NATIVE_CODE_DRIFT")
    from prdiffusion.track_r.physics import load_png_rgb8
    x=load_png_rgb8(safe(d/"reconstruction.png"))
    return x,d
def refs_and_manifests(run,a):
    mpath=run/"XPK_GPU_GT_FREE_MANIFEST.json"
    m=read(mpath)
    rec=read(run/"XPK_SOURCE_FREEZE_RECEIPT.json")
    refs=read(run/"XPK_CPU_ONLY_HIST_REFERENCES.json")
    gate(rec["code_head"]==a.expected_head==m["head"]
         and rec["sha_manifest"]==sha(mpath)
         and rec["sha_cpu_references"]==
             sha(run/"XPK_CPU_ONLY_HIST_REFERENCES.json")
         and refs["status"]=="XP_K_CPU_ONLY_HISTORICAL_SCORES_AND_GT_REFERENCES"
         and m["original_GT_access"] is False
         and m["reference_paths_in_GPU_manifest"] is False,
         "SOURCE_FREEZE")
    return m,refs
def replay(a,run,m,refs,selected):
    family=a.family
    origins=m["historical_replay_rows"][family]
    gate(len(origins)==2 and len({r["image_id"] for r in origins})==1,
         "REPLAY_TWO_ROOTS")
    rows=[]
    for r in origins:
        h=refs["family_references"][family][r["image_id"]][int(r["run"])]
        x8,folder=verify_root(run,m,r,family,"historical_replay")
        gt,g8=canonical_gt(h)
        feat=features(x8,g8,signed_y(r["measurement_path"],r["measurement_sha256"]),
                      gt,m64(gt))
        from prdiffusion.track_r.r0_7_stats import label_candidate
        lbl=label_candidate(feat,float(h["historical_tau_signed_good_q95"]))
        delta=abs(float(feat["psnr_raw"])-float(h["historical_raw_psnr"]))
        png_exact=sha(folder/"reconstruction.png")==h["historical_candidate_sha256"]
        passed=(delta<=.1 and lbl==h["historical_raw_label"])
        rows.append(dict(uid=r["uid"],image_id=r["image_id"],run=r["run"],
          historical_uid=h["uid"],measurement_sha256=r["measurement_sha256"],
          current_raw_psnr=feat["psnr_raw"],
          historical_raw_psnr=h["historical_raw_psnr"],
          abs_raw_psnr_delta_db=delta,computed_native_class=lbl,
          frozen_historical_class=h["historical_raw_label"],
          frozen_native_tau=h["historical_tau_signed_good_q95"],
          exact_png_match=png_exact,pass_A4=passed))
    status="XP_K_A23_5B_ORIGINAL_TWO_ROOT_IDENTITY_REPLAY_PASS" if all(x["pass_A4"] for x in rows) else "STOP_ASK_A23_5B_REPLAY_FAILED"
    receipt=dict(status=status,family=family,source_head=a.expected_head,
       original_parameter={"SITCOM":.01,"FAST":.05,"DAPS":.01}[family],
       n_root=2,rows=rows,original_A4_PSNR_tolerance_db=0.1,
       original_A4_label_must_match=True,
       passed=all(r["pass_A4"] for r in rows),
       max_raw_psnr_delta_db=max(r["abs_raw_psnr_delta_db"] for r in rows),
       n_bitwise_png=sum(r["exact_png_match"] for r in rows),
       GT_used_CPU_scoring_only=True,strict_thresholds_unchanged=True)
    path=run/"score"/f"{family}_REPLAY_A4_RECEIPT.json"
    write(path,receipt)
    print("XPK_REPLAY_A4_GATE",family,status,flush=True)
    gate(receipt["passed"],"SCIENCE_REPLAY_MISMATCH_STOP_ASK_"+family)

def clip_init():
    import torch
    from scripts.track_r.xp_a8_d1p1_clip_cpu import (
        model_source,G1_UP,G1_INV,make_text)
    clipfile,native,head=model_source()
    sys.path.insert(0,str(native))
    clip=importlib.import_module("models.clip")
    model,preprocess=clip.load(str(clipfile),device="cpu",jit=False)
    model.eval()
    with torch.inference_mode():
        up=make_text(model,clip,torch,G1_UP)
        down=make_text(model,clip,torch,G1_INV)
    return dict(model=model,preprocess=preprocess,u=up,d=down,
                checkpoint_sha256=sha(clipfile),native_head=head,
                prompts_up=list(G1_UP),prompts_down=list(G1_INV))
def clip_orient(clip,x8):
    import torch,numpy as np
    from PIL import Image
    img=Image.fromarray(x8[0].permute(1,2,0).numpy(),"RGB")
    inv=Image.fromarray(np.rot90(np.asarray(img),2).copy(),"RGB")
    with torch.inference_mode():
        tensor=torch.stack([clip["preprocess"](img),clip["preprocess"](inv)])
        feat=clip["model"].encode_image(tensor).float()
        feat=feat/feat.norm(dim=-1,keepdim=True)
        u=clip["u"].float(); d=clip["d"].float()
        scores=(feat@u)-(feat@d)
        margin=float(scores[1]-scores[0])
    gate(math.isfinite(margin),"CLIP_FINITE_MARGIN")
    return "111" if margin>0 else "000",margin
def calibration(a,run,m,refs,selected):
    family=a.family
    cal=m["calibrated_rows"][family]
    gate(len(cal)==400 and
         len({r["image_id"] for r in cal})==100,"CAL_PANEL_100x4")
    # Required historical replay accepted using SAME pinned adapter.
    hist=read(run/"score"/f"{family}_REPLAY_A4_RECEIPT.json")
    gate(hist["passed"] is True and hist["source_head"]==a.expected_head
         and hist["n_root"]==2,"REPLAY_NOT_ACCEPTED")
    import torch
    torch.set_num_threads(4)
    clip=clip_init()
    table=[];per_image=[]
    groups=defaultdict(list)
    for row in cal:groups[row["image_id"]].append(row)
    family_refs=refs["family_references"][family]
    historical=[x for iid in sorted(family_refs) for x in family_refs[iid]]
    expected_hist=200 if family=="DAPS" else 400
    gate(len(historical)==expected_hist and len(family_refs)==100,
         "HIST_BASELINE_CARDINALITY_"+family)
    tau_values={float(x["historical_tau_signed_good_q95"]) for x in historical}
    gate(len(tau_values)==1,"HIST_TAU_NOT_FAMILY_CONSTANT_"+family)
    family_tau=next(iter(tau_values))
    for k,(iid,rows) in enumerate(sorted(groups.items())):
        gate(sorted(r["run"] for r in rows)==[0,1,2,3],"CAL_RUNS")
        rr=family_refs[iid]
        gate(len(rr)==(2 if family=="DAPS" else 4),
             "HIST_IMAGE_CARDINALITY_"+family+"_"+iid)
        gt,g8=canonical_gt(rr[0])
        astar=m64(gt)
        y=signed_y(rows[0]["measurement_path"],rows[0]["measurement_sha256"])
        z=[]
        for row in sorted(rows,key=lambda r:r["run"]):
            x8,folder=verify_root(run,m,row,family,"calibrated")
            feat=features(x8,g8,y,gt,astar)
            from prdiffusion.track_r.r0_7_stats import label_candidate
            native_tau=family_tau
            label=label_candidate(feat,native_tau)
            subtype="G" if label=="GOOD" else ("O" if label=="ORBIT_ERROR"
                  and feat["s_gt"]=="111" else ("P" if label=="ORBIT_ERROR"
                  else ("C" if label=="NONORBIT_CONSISTENT" else "I")))
            dat=dict(image_id=iid,run=row["run"],uid=row["uid"],
                family=family,measurement_sha256=row["measurement_sha256"],
                raw_label=label,label_short=subtype,
                frozen_native_tau=native_tau,
                raw_psnr_db=feat["psnr_raw"],orbit_psnr_best=feat["psnr_orbit_best"],
                s_gt=feat["s_gt"],rho_signed=feat["rho_signed_raw"],
                rho64=feat["rho64"],rho_star64=feat["rho_star64"],
                rho_over_sigma64=feat["rho_over_sigma64"],
                rho_below_rho_star=feat["rho_below_rho_star"],
                Delta=feat["Delta"],D=feat["D"],C=feat["C"],
                native_png_sha256=sha(folder/"reconstruction.png"))
            table.append(dat);z.append((dat,feat,x8))
        choice=min(z,key=lambda item:(item[0]["rho_signed"],item[0]["run"]))
        orientation,margin=clip_orient(clip,choice[2])
        ch_psnr=float(choice[1]["psnr_by_orbit"][orientation])
        per_image.append(dict(image_id=iid,family=family,
            runs=4,run_labels="".join(x[0]["label_short"] for x in z),
            first_run_good25=(z[0][0]["raw_psnr_db"]>=25),
            E0_raw_psnr=choice[0]["raw_psnr_db"],
            E0_run=choice[0]["run"],
            E1_CLIP_psnr=ch_psnr,
            E1_CLIP_orientation=orientation,E1_CLIP_margin=margin,
            O_orbit_psnr=max(x[1]["psnr_orbit_best"] for x in z),
            GOOD_count=sum(x[0]["label_short"]=="G" for x in z)))
        if (k+1)%10==0:
            print("XPK_CPU_SCORE",family,k+1,"/100",flush=True)
    gate(len(table)==400 and len(per_image)==100,"SCORING_400_FINAL")
    counts=Counter(x["label_short"] for x in table)
    good=[x["rho_over_sigma64"] for x in table if x["label_short"]=="G"]
    gate(len(historical)==expected_hist,"HIST_BASELINE_COUNT_"+family)
    hist_good=[x["historical_rho_signed"]/.05 for x in historical
               if x["historical_raw_label"]=="GOOD"]
    old_non=[x for x in historical if x["historical_raw_label"]!="GOOD"]
    hist_cons=sum(x["historical_raw_label"]=="NONORBIT_CONSISTENT"
                  for x in old_non)
    new_non=[x for x in table if x["label_short"]!="G"]
    per_label={"G":"GOOD","O":"ORBIT_GLOBAL","P":"ORBIT_PARTIAL",
               "C":"CONSISTENT","I":"INCONSISTENT"}
    baseline_counts=Counter(x["historical_raw_label"] for x in historical)
    frozen1=read(FROZEN_V1)
    frozen11=read(FROZEN_V11)
    gate(frozen1["status"]=="FROZEN_BEFORE_XP_K_AND_XP_G_RESULTS" and
         frozen11["status"]=="PRE_REGISTERED_BEFORE_A5_5_RESULTS",
         "MECHANISM_FORECAST_NOT_PRE_REGISTERED")
    v1=next(x for x in frozen1["xp_k"] if x["family"]==family)
    v11=next(x for x in frozen11["model_conditional_predictions"]
             if x["family"]==family)
    result=dict(status="XP_K_A24_CALIBRATED_100x4_CPU_SCORE_COMPLETE",
       family=family,source_head=a.expected_head,
       dataset="ffhq_cal",GT_scoring_only=True,
       n_images=100,n_candidates=400,
       n_historical_candidates=len(historical),
       historical_roots_per_image=(2 if family=="DAPS" else 4),
       registered_class_rates={per_label[k]:counts[k]/400 for k in per_label},
       registered_class_counts={per_label[k]:counts[k] for k in per_label},
       historical_class_counts=dict(baseline_counts),
       GOOD_rho_over_sigma_summary=summary(good),
       historical_GOOD_rho_over_sigma_summary=summary(hist_good),
       rho_below_rho_star_rate=sum(x["rho_below_rho_star"] for x in table)/400,
       good_first_run_rate=sum(x["first_run_good25"] for x in per_image)/100,
       E1_CLIP_mean_raw_psnr=float(statistics.mean(x["E1_CLIP_psnr"] for x in per_image)),
       E1_CLIP_Good25_rate=sum(x["E1_CLIP_psnr"]>=25 for x in per_image)/100,
       O_orbit_mean_psnr=float(statistics.mean(x["O_orbit_psnr"] for x in per_image)),
       O_orbit_Good25_rate=sum(x["O_orbit_psnr"]>=25 for x in per_image)/100,
       SITCOM_consistent_share_of_nongood=(counts["C"]/len(new_non)
                                          if new_non else None),
       historical_consistent_share_of_nongood=(hist_cons/len(old_non)
                                               if old_non else None),
       conditional_GOOD_D_summary=summary([x["D"] for x in table if x["label_short"]=="G"]),
       conditional_GOOD_C_summary=summary([x["C"] for x in table if x["label_short"]=="G"]),
       conditional_GOOD_Delta_summary=summary([x["Delta"] for x in table if x["label_short"]=="G"]),
       prereg_v1_sha256=sha(FROZEN_V1),prereg_v1_1_sha256=sha(FROZEN_V11),
       prereg_v1_conditional_direction=v1.get("conditional_direction"),
       prereg_v1_NATIVE_effect_size_forecast=v1.get("native_effect_size_prediction"),
       prereg_v1_1_conditional_prediction=v11,
       prediction_scope="conditional surrogate only; not a prediction of exact native effect size or PSNR",
       H_K1_descriptive_GOOD_rho_sigma_median_shift=(
           q(good,.5)-q(hist_good,.5) if good and hist_good else None),
       H_K2_SITCOM_consistent_share_reduction=(
           hist_cons/len(old_non)-counts["C"]/len(new_non)
           if family=="SITCOM" and old_non and new_non else None),
       no_native_change_except_one_scalar=True,
       CLIP_G1_checkpoint_sha256=clip["checkpoint_sha256"],
       CLIP_G1_up_prompts=clip["prompts_up"],
       CLIP_G1_down_prompts=clip["prompts_down"])
    dest=run/"score"
    out=dest/f"{family}_CAL_AGGREGATE.json"
    fields=list(table[0])
    with (dest/f"{family}_CAL_PER_ROOT.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
    with (dest/f"{family}_CAL_PER_IMAGE.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(per_image[0]));w.writeheader();w.writerows(per_image)
    result["per_root_csv_sha256"]=sha(dest/f"{family}_CAL_PER_ROOT.csv")
    result["per_image_csv_sha256"]=sha(dest/f"{family}_CAL_PER_IMAGE.csv")
    write(out,result)
    print("XPK_CPU_COMPLETE_WITH_FROZEN_FORECASTS",family,
          counts,"E1_CLIP_good25",result["E1_CLIP_Good25_rate"],flush=True)

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"PINNED_HEAD")
    gate(a.family in ORIG_FAMILIES and
         a.phase in ("replay","calibration"),"FAMILY_PHASE")
    run=Path(a.run)
    gate(run.parent==BASE and run.name.startswith("xp_a41_xpk_")
         and run.is_dir() and not run.is_symlink(),"EXACT_RUN")
    m,refs=refs_and_manifests(run,a)
    score=run/"score"
    gate(score.is_dir(),"SCORE_DIR")
    path=score/(f"{a.family}_REPLAY_A4_RECEIPT.json" if a.phase=="replay"
                else f"{a.family}_CAL_AGGREGATE.json")
    gate(not path.exists(),"NO_SCORE_RECOMPUTATION_OR_OVERWRITE")
    if a.phase=="replay":
        replay(a,run,m,refs,None)
    else:
        calibration(a,run,m,refs,None)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--expected-head",required=True)
    p.add_argument("--family",choices=ORIG_FAMILIES,required=True)
    p.add_argument("--phase",choices=("replay","calibration"),required=True)
    try:sys.exit(main(p.parse_args()))
    except Exception as e:
        traceback.print_exc()
        print("ACTION_NEEDED_XPK_SCORING",repr(e),flush=True)
        sys.exit(8)
