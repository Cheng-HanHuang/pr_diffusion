#!/usr/bin/env python3
"""A11 B16 CPU-only 16-run scientific strings and candidate scoring.

First authenticates prospective v1/v3 freeze and 2160 SITCOM receipts,
then opens only the 180 R0-exposed GT/measurement pairs.
Historical four runs per image are reused from R0.7 SHA-verified scores;
2160 new runs independently scored with exact frozen RGB8 conventions.
Does not load any CLIP model; orientation decisions in separate CPU stage.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,traceback
from collections import defaultdict,Counter
from pathlib import Path
import torch
from scripts.track_r.xp_b16_v1_v3_admission_cpu import (
 ROOT,BASE,WT,RUN,GEN_HEAD,B16_SHA,B16_RECEIPT_SHA,freeze,source,check,git,sha)
from scripts.track_r.r0_2_g0_cpu import gt_source,read_tensor
from scripts.track_r.xp_a10_b16_gate_prepare_cpu import safe
from prdiffusion.track_r.physics import to_rgb8,from_rgb8,rho_signed
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
from scripts.track_r.r0_4_score_gpu import candidate_tensor
from scripts.track_r.xp_a11_b16_handoff_autopilot import verify_task
R07=BASE/"r0_7_cpu_20261023T080434Z"
CAT=R07/"prepare/R0_7_OFFLINE_GT_CATALOG.json"
SCORES=R07/"analysis/R0_7_SCORED_GT_ROWS.jsonl"
R07_RECEIPT=R07/"analysis/R0_7_COMPLETE_RECEIPT.json"
GOOD=25.
TAU={"S1_b22":0.0448503,"S2_b24_dev80":0.0449706}
NAMES={"S1_b22":"B22","S2_b24_dev80":"DEV80"}
HEAD_ROWS=("image_id","cohort","labels_by_run","rho_signed_by_run",
           "tau_signed_good_q95","n_G","n_O_g","n_O_p","n_C","n_I")
COLUMNS=("image_id","cohort","run","label","rho_signed","psnr_raw",
         "psnr_orbit_best","s_gt","flip_good",
         "reconstruction_path","reconstruction_format","reconstruction_sha256",
         "measurement_sha256","uid")
def write_json(path,obj):
    with path.open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
def load_history():
    r=R07_RECEIPT
    receipt=json.loads(safe(r).read_text())
    check(receipt["n_scored"]==3240 and
          sha(SCORES)==receipt["evaluated_rows_sha256"],"R07_SCORES_SHA")
    data=json.loads(safe(CAT).read_text())
    check(len(data["rows"])==3240,"R07_CATALOG")
    old=defaultdict(list); identities={}
    for r in data["rows"]:
        if r["family"]!="SITCOM" or r["source"] not in NAMES:continue
        key=(r["source"],str(r["image_id"]).zfill(5))
        old[key].append(r)
        idn=(r["gt_path"],r["gt_format"],r["measurement_path"],
             r["measurement_sha256"])
        if key in identities:check(identities[key]==idn,"R07_PAIR_DRIFT")
        identities[key]=idn
    check(len(identities)==180 and
          len([x for x in identities if x[0]=="S1_b22"])==100 and
          len([x for x in identities if x[0]=="S2_b24_dev80"])==80,
          "R07_180_PAIR_GRID")
    scoremap=defaultdict(dict)
    with safe(SCORES).open() as f:
        for line in f:
            r=json.loads(line)
            if r["family"]!="SITCOM" or r["source"] not in NAMES:continue
            key=(r["source"],str(r["image_id"]).zfill(5))
            rep=int(r["run_index"])
            check(rep not in scoremap[key],"R07_SITCOM_DUPLICATE")
            scoremap[key][rep]=r
    check(set(scoremap)==set(identities)
          and all(set(v)==set(range(4)) for v in scoremap.values()),
          "R07_FOUR_PREFIX")
    return old,identities,scoremap
def label(raw,orbit,best,rho,tau):
    if raw>=GOOD:return "G"
    if orbit>=GOOD:return "O_g" if best=="111" else "O_p"
    return "C" if rho<=tau else "I"
def score(out,header):
    old,identities,history=load_history()
    manifest=json.loads((RUN/"XP_A11_B16_PROD_TASKS.json").read_text())
    groups=manifest["b22_groups"]+manifest["b24_groups"]
    score=[]
    panels=defaultdict(dict)
    torch.set_num_threads(4)
    for cohort,iid in sorted(identities):
        gtpath,gtfmt,ypath,yhash=identities[cohort,iid]
        check(sha(ypath)==yhash,"Y_SHA_"+iid)
        gt8=to_rgb8(gt_source({"gt_path":gtpath,"gt_format":gtfmt}).float())
        y=read_tensor(safe(ypath),("measurement","y","observation"),(1,3,384,384))
        for row in sorted(old[cohort,iid],key=lambda r:int(r["run_index"])):
            rep=int(row["run_index"])
            h=history[cohort,iid][rep]
            check(row["uid"]==h["uid"],"HIST_UID_"+iid)
            raw=float(h["psnr_raw"]); orbit=float(h["psnr_orbit_best"])
            rho=float(h["rho_signed_raw"])
            best=str(h["s_gt"])
            lbl=label(raw,orbit,best,rho,TAU[cohort])
            old_label={"GOOD":"G","ORBIT_ERROR":"O_g" if best=="111" else "O_p",
               "NONORBIT_CONSISTENT":"C","NONORBIT_INCONSISTENT":"I"}.get(h["raw_label"])
            check(lbl==old_label,"PREFIX_LABEL_DRIFT_"+iid)
            item=dict(image_id=iid,cohort=NAMES[cohort],run=rep,
                label=lbl,rho_signed=rho,psnr_raw=raw,
                psnr_orbit_best=orbit,s_gt=best,
                flip_good=int(float(h["psnr_by_orbit"]["111"])>=GOOD),
                reconstruction_path=row["candidate_path"],
                reconstruction_format=row["candidate_format"],
                reconstruction_sha256=row["candidate_sha256"],
                measurement_sha256=yhash,uid=row["uid"])
            check(rep not in panels[cohort,iid],"PREFIX_DUP")
            panels[cohort,iid][rep]=item
            score.append(item)
        for rep in range(4,16):
            index=(next((j for j,g in enumerate(groups)
                   if g["source"]==cohort and g["image_id"]==iid
                   and (cohort=="S1_b22" or g["rows"][0]["rep"]==rep)),None))
            check(index is not None,"B16_ROOT_INDEX_"+iid)
            g=groups[index]
            candidate=next(x for x in g["rows"] if x["rep"]==rep)
            folder=RUN/"gpu"/g["task_id"]
            dest=folder/("root_"+hashlib.sha256(candidate["uid"].encode()).hexdigest()[:24])
            rec=json.loads(safe(dest/"XP_SITCOM_ROOT_COMPLETE.json").read_text())
            path=dest/"reconstruction.pt"
            check(rec["status"]=="XP_SITCOM_REPLAY_ROOT_PASS"
                  and rec["uid"]==candidate["uid"]
                  and rec["manifest_sha256"]==B16_SHA
                  and rec["runner_head"]==GEN_HEAD
                  and rec["measurement_sha256"]==yhash
                  and sha(path)==rec["reconstruction_pt_sha256"],
                  "NATIVE_ROOT_HASH_"+iid+"_"+str(rep))
            x=torch.load(safe(path),map_location="cpu",weights_only=True)["reconstruction"].float()
            check(tuple(x.shape)==(1,3,256,256)
                  and bool(torch.isfinite(x).all()),"RECON_TENSOR")
            x8=to_rgb8(x)
            orbit_map=orbit_psnr_channel_mse(x8,gt8)
            raw=float(orbit_map["000"])
            best=max(sorted(orbit_map),key=lambda k:orbit_map[k])
            orbit=float(orbit_map[best])
            rho=float(rho_signed(from_rgb8(x8),y)[0])
            check(all(math.isfinite(t) for t in (raw,orbit,rho))
                  and rho>=0 and orbit+1e-8>=raw,"NUMERIC")
            lbl=label(raw,orbit,best,rho,TAU[cohort])
            item=dict(image_id=iid,cohort=NAMES[cohort],run=rep,
                label=lbl,rho_signed=rho,psnr_raw=raw,
                psnr_orbit_best=orbit,s_gt=best,
                flip_good=int(orbit_map["111"]>=GOOD),
                reconstruction_path=str(path),
                reconstruction_format="pt",
                reconstruction_sha256=rec["reconstruction_pt_sha256"],
                measurement_sha256=yhash,uid=candidate["uid"])
            check(rep not in panels[cohort,iid],"NEW_DUP")
            panels[cohort,iid][rep]=item
            score.append(item)
        check(set(panels[cohort,iid])==set(range(16)),"PANEL16_"+iid)
        if len(panels)%10==0:
            print("B16_IMAGE_SCORED",len(panels),"/180",flush=True)
    check(len(score)==2880 and len(panels)==180,"B16_2880")
    strings=[]
    for (cohort,iid),group in sorted(panels.items()):
        items=[group[i] for i in range(16)]
        labels=[r["label"] for r in items]
        c=Counter(labels)
        strings.append(dict(image_id=iid,cohort=NAMES[cohort],
           labels_by_run=";".join(labels),
           rho_signed_by_run=";".join(format(r["rho_signed"],".10g") for r in items),
           tau_signed_good_q95=TAU[cohort],
           n_G=c["G"],n_O_g=c["O_g"],n_O_p=c["O_p"],
           n_C=c["C"],n_I=c["I"]))
    output=out/"XP_B16_16RUN_STRINGS.csv"
    with output.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=HEAD_ROWS);w.writeheader();w.writerows(strings)
    scored=out/"XP_B16_2880_SCORED_ROOTS.csv"
    with scored.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader()
        w.writerows(sorted(score,key=lambda r:(r["cohort"],r["image_id"],r["run"])))
    summary=dict(status="XP_B16_16RUN_2880_CPU_SCORED_PASS",
       cohort_images={"B22":100,"DEV80":80},
       n_images=180,n_historical=720,n_new=2160,
       n_scored=2880,n_panel=16,Good25=GOOD,
       historical_R07_receipt_sha256=sha(R07_RECEIPT),
       historical_R07_scores_sha256=sha(SCORES),
       frozen_TAU={"B22":TAU["S1_b22"],"DEV80":TAU["S2_b24_dev80"]},
       v1_primary_v3_secondary_freeze_verified=True,
       no_gpu=True,
       strings_sha256=sha(output),per_candidate_sha256=sha(scored),
       class_totals=dict(Counter(r["label"] for r in score)))
    write_json(out/"XP_B16_2880_SCORING_RECEIPT.json",summary)
    print("B16_2880_CPU_SCORING_PASS 180x16",flush=True)
    print("B16_STRINGS_SHA="+sha(output),flush=True)
    print("B16_ALL_CANDIDATES_SHA="+sha(scored),flush=True)
def main(a):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    check(git("rev-parse","HEAD").decode().strip()==a.expected_head
          and not git("status","--porcelain"),"ANALYSIS_WORKTREE")
    admission=json.loads(safe(Path(a.admission)/"B16_V1_V3_CPU_ADMISSION.json").read_text())
    check(admission["status"]=="B16_V1_V3_UNBLIND_ADMISSION_PASS"
          and admission["expected_head"]==a.expected_head
          and admission["source"]["n_new_roots"]==2160,"B16_ADMISSION")
    forecasts=freeze()
    check(forecasts==admission["forecasts"],"FORECAST_DRIFT")
    out=Path(a.out)
    check(out.parent==BASE and out.name.startswith("xp_b16_v1_v3_")
          and not out.exists() and not out.is_symlink(),"FRESH_OUTPUT")
    out.mkdir()
    score(out,admission)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--admission",required=True)
    parser.add_argument("--out",required=True)
    try:main(parser.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: B16_SCORE "+str(e),flush=True)
        sys.exit(8)
