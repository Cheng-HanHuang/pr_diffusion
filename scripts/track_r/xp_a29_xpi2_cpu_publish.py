#!/usr/bin/env python3
"""A21/A15.5 XP-I2 ImageNet per-particle CPU-only scoring, then publication.

Reuses frozen native PC-DAPS 100x4 bags and D1 GT/y, existing G1 CLIP
CPU implementation and exact native-float piq SSIM/VGG-LPIPS definitions.
No new GPU sampling, no GT in selectors, no matched-compute claims.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,subprocess,sys,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
RUN=BASE/"xp_a24_xpi2_prod_20261102T085625Z"
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")
PER_KEYS=("image_id","particle","R0_label","PC_pr_outcomes_label",
          "rho","g1_choice","g1_margin","PSNR_raw","SSIM_raw",
          "VGG_LPIPS_raw","PSNR_chosen","SSIM_chosen","VGG_LPIPS_chosen",
          "PSNR_orbit_best","best_orbit","measurement_sha256")
NFE_PC_PER_PARTICLE=480

def gate(v,msg):
    if not v:raise RuntimeError("STOP_A21_XPI2_"+msg)
def sha(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink() and
         all(x not in str(p).lower() for x in BANNED),"SAFE_FILE")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def git(root,*x):return subprocess.check_output(
    ["git","-C",str(root),*x],text=True,timeout=60).strip()
def write(p,v):
    with Path(p).open("x") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def csvwrite(p,rows,cols):
    with Path(p).open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,lineterminator="\n")
        w.writeheader();w.writerows(rows)

def inputs(a):
    from scripts.track_r.xp_a24_xpi2_prepare_cpu import sha as oldsha
    from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE as b
    gate(b==BASE and os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and git(WT,"rev-parse","HEAD")==a.expected_head
         and not git(WT,"status","--porcelain"),"CLEAN_PIN_CPU")
    f=RUN/"XP_A24_XPI2_FINAL_STATUS.json"
    gate(f.is_file(),"COMPLETE_NATIVE_RECEIPT")
    fin=read(f)
    gate(fin["status"]=="A24_XPI2_COMPLETE"
         and fin["n_completed_bags"]==100
         and fin["n_particles"]==400 and fin["active"]==[],
         "NATIVE_GRID_COMPLETE")
    prepfiles=list(BASE.glob("xp_a24_xpi2_prepare_*/XP_A24_XPI2_PREP_RECEIPT.json"))
    good=[]
    for p in prepfiles:
        r=read(p)
        if r.get("status")=="A19_XPI2_D1_100_BAGS_CPU_FROZEN_NOT_GENERATED":
            good.append(p)
    gate(len(good)==1,"EXACT_ONE_I2_PREP")
    rec=read(good[0])
    mfpath=good[0].parent/"XP_A24_XPI2_100_BAGS.json"
    mf=read(mfpath)
    gate(sha(mfpath)==rec["manifest_sha256"]
         and mf["n_bags"]==100 and mf["bag_size"]==4
         and len(mf["rows"])==100,"I2_NATIVE_SOURCE_RECEIPT")
    return mf,fin,rec,sha(f)

def pc_pr_outcomes(x8,gt8):
    """A4.1 paper-style per-channel 000/111 and ±1 shift classifier.

    A channel has a raw/rotated assignment only if its maximum channel
    PSNR is >1 dB nearer that orientation. Mixed orientations are upside
    down/color-inverted; missing channel assignment yields FAILED.
    This evaluator is GT-based and is NEVER used for a selector.
    """
    import torch
    from prdiffusion.track_r.physics import apply_orbit
    gate(x8.shape==gt8.shape==(1,3,256,256),"OUTCOME_SHAPE")
    raw=x8.float()
    rot=apply_orbit(x8,"111").float()
    truth=gt8.float()
    outcome=[]
    for c in range(3):
        def best(z):
            return max(
                -10*math.log10(max(1e-12,float(
                  ((torch.roll(z[:,c:c+1],(dy,dx),(-2,-1))-
                    truth[:,c:c+1]).square().mean()))/255.**2))
                for dy in (-1,0,1) for dx in (-1,0,1))
        a=best(raw);b=best(rot)
        if a>b+1.:outcome.append("raw")
        elif b>a+1.:outcome.append("flip")
        else:outcome.append("none")
    if outcome==["raw"]*3:return "FAITHFUL"
    if all(v in ("raw","flip") for v in outcome):
        return "UPSIDE_DOWN_OR_COLOR_INVERTED"
    return "FAILED"

def metrics_pretrained():
    import torch,piq
    from scripts.track_r.xp_a16_cp3_piq_metrics_cpu import weights_preflight
    info=weights_preflight()
    original=torch.hub.load_state_dict_from_url
    def cached(url,*args,**kwargs):
        name=kwargs.get("file_name") or url.rsplit("/",1)[-1]
        p=Path(torch.hub.get_dir())/"checkpoints"/name
        gate(p.is_file(),"NO_DOWNLOAD_VGG")
        return torch.load(p,map_location="cpu",weights_only=True)
    torch.hub.load_state_dict_from_url=cached
    try:lpips=piq.LPIPS(replace_pooling=True,reduction="none").eval().cpu()
    finally:torch.hub.load_state_dict_from_url=original
    return lpips,info

def main(a):
    mf,final,rec,final_sha=inputs(a)
    from scripts.track_r.xp_a15_cp3_score_cpu import common_sources
    from scripts.track_r.xp_a16_cp3_piq_metrics_cpu import true_image
    from scripts.track_r.xp_a15_cp2_d4_cpu import clip_new
    from scripts.track_r.xp_a16_cp3_joint_readout_cpu import read_stage
    from scripts.track_r.xp_a16_cp3_nfe_ledger_cpu import frozen_config
    from scripts.track_r.r0_2_g0_cpu import read_tensor
    from prdiffusion.track_r.physics import (
        to_rgb8,from_rgb8,rho_signed,apply_orbit)
    from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
    import torch,piq,numpy as np
    torch.set_num_threads(4)
    ff,gt_in,y_in=common_sources()
    gate(len(gt_in)==len(y_in)==100
         and set(gt_in)==set(y_in)==set(r["image_id"] for r in mf["rows"]),
         "D1_GT_Y_IMAGE_IDENTITY")
    cfg=frozen_config()
    gate(cfg["nominal_tweedie_calls_per_particle"]==NFE_PC_PER_PARTICLE
         and cfg["nominal_per_bag_particle_equivalent_calls"]==1920,
         "A15_5_PC_NFE")
    if a.preflight_only:
        for row in mf["rows"]:
            dest=RUN/"roots"/("bag_"+hashlib.sha256(
                   row["uid"].encode()).hexdigest()[:24])
            receipt=read(dest/"XP_A24_I2_BAG_COMPLETE.json")
            gate(receipt["uid"]==row["uid"]
                 and receipt["image_id"]==row["image_id"]
                 and receipt["bag_size"]==4
                 and receipt["status"]=="A24_XPI2_NATIVE_FOUR_PARTICLE_BAG_PASS"
                 and sha(dest/"particles.pt")==receipt["particles_pt_sha256"],
                 "I2_SOURCE_BAG_SHA")
        print("A21_XPI2_CPU_PREFLIGHT_PASS",len(mf["rows"]),
              "native_bags, all SHA receipts verified",flush=True)
        return
    lpips,info=metrics_pretrained()
    stage=read_stage("G5")
    gate(len(stage["candidates"])==800
         and len(stage["clips"])==800
         and len(stage["metrics"])==800,
         "DAPS_G5_8RUNS_FROZEN")
    out=Path(a.run)
    gate(out.parent==BASE and out.name.startswith("xp_a29_xpi2_score_")
         and not out.exists() and not out.is_symlink(),"FRESH_CPU_RUN")
    out.mkdir(mode=0o700)
    src=[];record_rows=[];store={}
    for i,row in enumerate(mf["rows"]):
        iid=row["image_id"]
        dest=RUN/"roots"/("bag_"+hashlib.sha256(
                  row["uid"].encode()).hexdigest()[:24])
        receipt=read(dest/"XP_A24_I2_BAG_COMPLETE.json")
        gate(receipt["uid"]==row["uid"]
             and receipt["runner_head"]==final["runner_head"]
             and receipt["manifest_sha256"]==rec["manifest_sha256"]
             and sha(dest/"particles.pt")==receipt["particles_pt_sha256"]
             and len(receipt["particle_pngs"])==4,
             "I2_FULL_BAG_PROVENANCE")
        for j,z in enumerate(receipt["particle_pngs"]):
            gate(z["particle"]==j and sha(dest/f"particle_{j}.png")==z["sha256"],
                 "I2_PNG_PROOF")
        raw=torch.load(dest/"particles.pt",map_location="cpu",
                       weights_only=True)["particles"].float()
        gate(raw.shape==(1,4,3,256,256) and bool(torch.isfinite(raw).all()),
             "I2_FLOAT_TENSOR")
        gt=true_image(iid,"G5",ff,gt_in)
        gt8=to_rgb8(gt)
        yrow=y_in[iid]
        gate(row["measurement_sha256"]==yrow["measurement_sha256"],
             "I2_D1_Y_SHA")
        y=read_tensor(yrow["measurement_path"],
             ("measurement","y","observation"),(1,3,384,384))
        true=((gt.clamp(-1,1)+1)/2).expand(4,-1,-1,-1)
        x=((raw.clamp(-1,1)+1)/2).squeeze(0)
        orient=apply_orbit(raw.squeeze(0),"111")
        xi=(orient.clamp(-1,1)+1)/2
        with torch.inference_mode():
            sr=piq.ssim(x,true,data_range=1.,reduction="none").tolist()
            sf=piq.ssim(xi,true,data_range=1.,reduction="none").tolist()
            lr=lpips(x,true).tolist()
            lf=lpips(xi,true).tolist()
        store[iid]=dict(raw=raw,gt=gt)
        for j in range(4):
            xj=raw[:,j]
            x8=to_rgb8(xj)
            gate(torch.equal(x8,
                __import__("prdiffusion.track_r.physics",fromlist=[
                  "load_png_rgb8"]).load_png_rgb8(dest/f"particle_{j}.png")),
                "I2_NATIVE_FLOAT_PNG_IDENTITY")
            orb=orbit_psnr_channel_mse(x8,gt8)
            best=max(sorted(orb),key=lambda k:orb[k])
            rho=float(rho_signed(from_rgb8(x8),y)[0])
            gate(math.isfinite(rho) and rho>=0,"I2_RHO")
            value=dict(image_id=iid,run=j,particle=j,
                 cohort="XP_IMAGENET100",uid=row["uid"]+"|particle"+str(j),
                 label="",rho_signed=rho,psnr_raw=float(orb["000"]),
                 psnr_flip=float(orb["111"]),
                 psnr_orbit_best=float(orb[best]),best_orbit=best,
                 flip_good=int(orb["111"]>=20),
                 reconstruction_path=str(dest/f"particle_{j}.png"),
                 reconstruction_sha256=sha(dest/f"particle_{j}.png"),
                 measurement_sha256=row["measurement_sha256"])
            src.append(value)
            record_rows.append(dict(value,SSIM_raw=float(sr[j]),
                 SSIM_111=float(sf[j]),LPIPS_VGG_raw=float(lr[j]),
                 LPIPS_VGG_111=float(lf[j]),
                 PC_pr_outcomes_label=pc_pr_outcomes(x8,gt8)))
        if (i+1)%10==0:
            print("A21_XPI2_NATIVE_FLOAT_SCORED",i+1,"/100",flush=True)
        del store[iid]
    # The R0 label threshold for a new ImageNet family is calibrated
    # only on run indices 0..3 at GOOD raw PSNR >= 20.
    calibrated=[r["rho_signed"] for r in record_rows if r["psnr_raw"]>=20]
    gate(len(calibrated)>0,"NO_GOOD20_FOR_FROZEN_TAU")
    tau=float(np.quantile(calibrated,.95,method="linear"))
    gate(math.isfinite(tau) and tau>0,"TAU")
    for r in record_rows:
        r["label"]=("G" if r["psnr_raw"]>=20 else
                "O_g" if r["psnr_orbit_best"]>=20 and r["best_orbit"]=="111" else
                "O_p" if r["psnr_orbit_best"]>=20 else
                "C" if r["rho_signed"]<=tau else "I")
    for r,v in zip(src,record_rows):r["label"]=v["label"]
    clips,clipproof=clip_new(src,mode="g1",threshold=20)
    byclip={(r["image_id"],int(r["run"])):r for r in clips}
    gate(len(byclip)==400,"400_G1_CHOICES")
    table=[]
    by=defaultdict(list)
    for r in record_rows:
        key=(r["image_id"],r["particle"]);c=byclip[key]
        flip=c["g1_choice"]=="111"
        per=dict(image_id=r["image_id"],particle=r["particle"],
                 R0_label=r["label"],
                 PC_pr_outcomes_label=r["PC_pr_outcomes_label"],
                 rho=r["rho_signed"],g1_choice=c["g1_choice"],
                 g1_margin=c["g1_margin"],
                 PSNR_raw=r["psnr_raw"],SSIM_raw=r["SSIM_raw"],
                 VGG_LPIPS_raw=r["LPIPS_VGG_raw"],
                 PSNR_chosen=r["psnr_flip"] if flip else r["psnr_raw"],
                 SSIM_chosen=r["SSIM_111"] if flip else r["SSIM_raw"],
                 VGG_LPIPS_chosen=r["LPIPS_VGG_111"] if flip else
                                        r["LPIPS_VGG_raw"],
                 PSNR_orbit_best=r["psnr_orbit_best"],
                 best_orbit=r["best_orbit"],
                 measurement_sha256=r["measurement_sha256"])
        table.append(per);by[r["image_id"]].append(per)
    gate(len(table)==400 and len(by)==100 and
         all(len(z)==4 for z in by.values()),"I2_400x100_JOIN")
    policies=[]
    for iid,rs in sorted(by.items()):
        rr=sorted(rs,key=lambda r:r["particle"])
        pick=min(rr,key=lambda r:(float(r["rho"]),r["particle"]))
        o_raw=max(rr,key=lambda r:(float(r["PSNR_raw"]),-r["particle"]))
        o_orbit=max(rr,key=lambda r:(float(r["PSNR_orbit_best"]),
                                    -r["particle"]))
        metrics=("PSNR","SSIM","VGG_LPIPS")
        def writepolicy(name,row,suffix):
            policies.append(dict(image_id=iid,policy=name,
                 n_particles=4,selected_particle=row["particle"],
                 PSNR=row["PSNR_"+suffix],SSIM=row["SSIM_"+suffix],
                 VGG_LPIPS=row["VGG_LPIPS_"+suffix],
                 GOOD20=int(row["PSNR_"+suffix]>=20)))
        writepolicy("FIRST_PARTICLE",rr[0],"raw")
        policies.append(dict(image_id=iid,policy="PC_NATIVE_MEAN4",
            n_particles=4,selected_particle="MEAN",
            PSNR=mean(r["PSNR_raw"] for r in rr),
            SSIM=mean(r["SSIM_raw"] for r in rr),
            VGG_LPIPS=mean(r["VGG_LPIPS_raw"] for r in rr),
            GOOD20=mean(int(r["PSNR_raw"]>=20) for r in rr)))
        writepolicy("E0",pick,"raw")
        writepolicy("E1_G1",pick,"chosen")
        writepolicy("O_RAW",o_raw,"raw")
        # Complete oracle at best R0 orbit; evaluate SSIM and LPIPS for
        # the exact chosen per-channel rotation using native float.
        pos=int(o_orbit["particle"])
        raw=next(t for t in src if t["image_id"]==iid
                 and t["particle"]==pos)
        dest=RUN/"roots"/("bag_"+hashlib.sha256(
            next(x["uid"] for x in mf["rows"]
                 if x["image_id"]==iid).encode()).hexdigest()[:24])
        bag=torch.load(dest/"particles.pt",map_location="cpu",
                       weights_only=True)["particles"].float()
        gt=true_image(iid,"G5",ff,gt_in)
        oriented=apply_orbit(bag[:,pos],o_orbit["best_orbit"])
        xx=(oriented.clamp(-1,1)+1)/2
        yy=(gt.clamp(-1,1)+1)/2
        with torch.inference_mode():
            ss=float(piq.ssim(xx,yy,data_range=1.,reduction="none")[0])
            lp=float(lpips(xx,yy)[0])
        policies.append(dict(image_id=iid,policy="O_ORBIT",
             n_particles=4,selected_particle=pos,
             PSNR=o_orbit["PSNR_orbit_best"],SSIM=ss,
             VGG_LPIPS=lp,GOOD20=int(o_orbit["PSNR_orbit_best"]>=20)))
    strings=[];decisions=[]
    for iid,rr in sorted(by.items()):
        sortedrows=sorted(rr,key=lambda r:r["particle"])
        strings.append(dict(image_id=iid,
          R0_labels=";".join(r["R0_label"] for r in sortedrows),
          PC_pr_outcomes_labels=";".join(
                 r["PC_pr_outcomes_label"] for r in sortedrows),
          rho_order=",".join(str(r["particle"]) for r in sorted(
                 rr,key=lambda r:(float(r["rho"]),r["particle"]))),
          n_G=sum(r["R0_label"]=="G" for r in rr),
          n_O_g=sum(r["R0_label"]=="O_g" for r in rr),
          n_O_p=sum(r["R0_label"]=="O_p" for r in rr),
          n_C=sum(r["R0_label"]=="C" for r in rr),
          n_I=sum(r["R0_label"]=="I" for r in rr)))
        for r in sortedrows:
            decisions.append(dict(image_id=iid,run=r["particle"],
                label=r["R0_label"],g1_choice=r["g1_choice"],
                g1_margin=r["g1_margin"],
                g1_chosen_good=int(r["PSNR_chosen"]>=20)))
    # Reuse the frozen DAPS G5 rows (same ImageNet 100 ids, 8 roots).
    comps=[]
    for iid in sorted(by):
        daps=[stage["candidates"][iid,j] for j in range(8)]
        cl=[stage["clips"][iid,j] for j in range(8)]
        mm=[stage["metrics"][iid,j] for j in range(8)]
        def choice(K):
            r=min(range(K),key=lambda j:(float(daps[j]["rho_signed"]),j))
            good=int(float(daps[r]["psnr_raw"])>=20)
            fix=int(cl[r]["g1_chosen_good"])
            return r,good,fix
        for K in (1,2,8):
            r,g,f=choice(K)
            comps.append(dict(image_id=iid,policy=f"DAPS_E1_K{K}",
                K=K,run_selected=r,GOOD20=f,NFE=4000*K,
                PSNR=float(daps[r]["psnr_flip"] if
                          cl[r]["g1_choice"]=="111" else daps[r]["psnr_raw"]),
                E0_GOOD20=g,reported_compute="nominal NFE",
                no_matched_compute_claim=True))
        comps.append(dict(image_id=iid,policy="DAPS_RUN0",
                K=1,run_selected=0,GOOD20=int(float(
                    daps[0]["psnr_raw"])>=20),NFE=4000,
                PSNR=float(daps[0]["psnr_raw"]),
                E0_GOOD20=int(float(daps[0]["psnr_raw"])>=20),
                reported_compute="nominal NFE",
                no_matched_compute_claim=True))
    gate(len(policies)==600 and len(comps)==400 and len(strings)==100,
         "POLICY_COMPARISON_GRID")
    per=out/"XP_I2_A15_5_PER_PARTICLE.csv"
    pol=out/"XP_I2_A15_5_POLICIES.csv"
    cmp=out/"XP_I2_A15_5_DAPS_COMPUTE_COMPARISON.csv"
    sts=out/"XP_I2_A15_5_STRINGS.csv"
    dec=out/"XP_I2_A15_5_G1_DECISIONS.csv"
    csvwrite(per,table,PER_KEYS)
    csvwrite(pol,policies,list(policies[0]))
    csvwrite(cmp,comps,list(comps[0]))
    csvwrite(sts,strings,list(strings[0]))
    csvwrite(dec,decisions,list(decisions[0]))
    summary=dict(status="A21_XPI2_A15_5_CPU_SCORED_COMPLETE",
        source_xpi2_final_sha256=final_sha,
        source_xpi2_manifest_sha256=rec["manifest_sha256"],
        n_images=100,n_particles=400,bag_size=4,
        native_nominal_NFE_per_particle=NFE_PC_PER_PARTICLE,
        native_nominal_NFE_per_bag=1920,
        pruning_NFE_additional_not_in_nominal=True,
        daps_nominal_NFE_per_independent_run=4000,
        daps_G5_reference_8_runs=True,
        R0_tau_good_q95=tau,
        paper_pr_outcomes_rule="per-channel 000 vs 111, PSNR best ±1 circular pixels; orientation gap >1 dB; all raw=FAITHFUL, mixed assigned=UPSIDE_DOWN_OR_COLOR_INVERTED, otherwise FAILED",
        no_GT_in_selection=True,no_new_gpu=True,no_matched_compute_claim=True,
        G1_clip_model=clipproof,pretrained_metric=info,
        results_sha256={p.name:sha(p) for p in (per,pol,cmp,sts,dec)},
        PC_native_policy_aggregate={key:mean(float(r[key]) for r in policies
                    if r["policy"]=="PC_NATIVE_MEAN4")
                    for key in ("PSNR","SSIM","VGG_LPIPS")},
        DAPS_G5_frozen_pooled_E1_K8_good20=mean(
            float(r["GOOD20"]) for r in comps
            if r["policy"]=="DAPS_E1_K8"),
        DAPS_G5_frozen_pooled_E1_K8_PSNR=mean(
            float(r["PSNR"]) for r in comps
            if r["policy"]=="DAPS_E1_K8"),
        DAPS_G5_pooled_K8_n_images=100,
        PC_I2_policy_aggregates={
            name:dict(n_images=100,
                 GOOD20=mean(float(x["GOOD20"]) for x in policies
                             if x["policy"]==name),
                 PSNR=mean(float(x["PSNR"]) for x in policies
                           if x["policy"]==name))
            for name in ("FIRST_PARTICLE","PC_NATIVE_MEAN4","E0",
                         "E1_G1","O_RAW","O_ORBIT")})
    write(out/"XP_I2_A15_5_CPU_RECEIPT.json",summary)
    # Publish on separate branch; do not mutate A21 frozen XP-C3 HEAD.
    tag=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    branch=f"codex/prdiff-track-r-xpi2-a15-5-{tag}"
    worktree=ROOT/f"pr_diffusion_track_r_xpi2_score_pub_{tag}"
    gate(not worktree.exists(),"FRESH_GIT_PUBLICATION")
    subprocess.run(["git","-C",str(WT),"worktree","add","-b",
                 branch,str(worktree),a.expected_head],check=True)
    ret=worktree/"docs/track_r/executors/xp/returns"/f"XP_I2_A15_5_{tag}"
    theory=worktree/"docs/track_r/executors/xp/theory_inputs"
    ret.mkdir();theory.mkdir(exist_ok=True)
    for p in (per,pol,cmp,out/"XP_I2_A15_5_CPU_RECEIPT.json"):
        shutil.copyfile(p,ret/p.name)
    for p in (sts,dec):shutil.copyfile(p,theory/p.name)
    subprocess.run(["git","-C",str(worktree),"add",
               str(ret.relative_to(worktree)),
               str((theory/sts.name).relative_to(worktree)),
               str((theory/dec.name).relative_to(worktree))],check=True)
    subprocess.run(["git","-C",str(worktree),"commit","-m",
                 "results(track-r): XP-I2 A15.5 100 ImageNet bags CPU-scored"],check=True)
    subprocess.run(["git","-C",str(worktree),"push","origin",
             "HEAD:refs/heads/"+branch],check=True,timeout=180)
    print("A21_XPI2_CPU_PUBLISHED",branch,
          git(worktree,"rev-parse","HEAD"),"PER_ROWS",len(table),
          "POLICY_ROWS",len(policies),"COMP_ROWS",len(comps),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--run",required=True)
    ap.add_argument("--preflight-only",action="store_true")
    try:main(ap.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A21_XPI2_CPU",exc,flush=True)
        sys.exit(8)
