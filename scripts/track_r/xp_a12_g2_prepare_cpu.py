#!/usr/bin/env python3
"""A12 G2 (XP-G): freeze paired-noise 180x2 y and 180x2x8 FAST roots.

This is CPU measurement generation, explicitly preregistered A12 §1:
n_R0 = (y_R0 - A(x_gt))/0.05; y_sigma=A(x_gt)+sigma*n_R0.
GT is accessed ONLY here, never written to root-task manifests, sent
to FAST workers, read by the sampling stage, or scored during generation.
Any earlier frozen y must match the R0 historical SHA. Produces no
new candidate outcomes or GT-based labels. Separate new root namespace.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,FAST_HEAD,gate
from scripts.track_r.r0_2_g0_cpu import gt_source,read_tensor
from prdiffusion.track_r.physics import magnitude
from scripts.track_r.xp_t_sitcom_prepare_cpu import MODEL_SHA
import torch

CAT=BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json"
SIGMAS=(("0.02",0.02),("0.10",0.10))
FREEZE_SOURCE="S5_r0_fast"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
MODEL=ROOT/"models/ffhq_10m.pt"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()
def write(p,x):
    with Path(p).open("x") as f:
        json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def safe(p):
    p=Path(p)
    path=p.resolve(strict=True)
    gate(p.is_file() and not p.is_symlink() and
         ROOT.resolve() in path.parents and
         not any(s in str(path).lower() for s in BAD),"PROTECTED_PATH")
    return p
def seed(sigma,uid,rep):
    k=f"XP|FAST|ffhq_s{sigma}|{uid}|{rep}"
    return int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],"big")%(2**31)
def original():
    group=defaultdict(list)
    c=json.loads(safe(CAT).read_text())
    gate(len(c["rows"])==3240,"R07_CATALOG")
    for r in c["rows"]:
        if r["source"]!=FREEZE_SOURCE:continue
        key=(r["measurement_source"],str(r["image_id"]))
        gate(key[0] in ("S1_b22","S2_b24_dev80"),"COHORT")
        group[key].append(r)
    gate(len(group)==180 and
         sum(k[0]=="S1_b22" for k in group)==100 and
         sum(k[0]=="S2_b24_dev80" for k in group)==80,"FFHQ_180")
    out=[]
    for (cohort,iid),rr in sorted(group.items()):
        # R0.7 offline catalog uses run_index (not GPU-manifest rep).
        rr.sort(key=lambda x:int(x["run_index"]))
        gate([int(r["run_index"]) for r in rr]==[0,1,2,3],"R0_REPS")
        y={(r["measurement_path"],r["measurement_sha256"]) for r in rr}
        gt={(r["gt_path"],r["gt_format"]) for r in rr}
        gate(len(y)==len(gt)==1,"R0_Y_GT_UNIQUE")
        yp,yd=next(iter(y))
        gp,gfmt=next(iter(gt))
        gate(sha(safe(yp))==yd,"ORIGINAL_Y_SHA")
        safe(gp)
        out.append((cohort,iid,yp,yd,gp,gfmt))
    return out
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    wt=Path(__file__).resolve().parents[2]
    git=lambda *args:subprocess.check_output(
       ["git","-C",str(wt),*args],text=True,timeout=30).strip()
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"WORKTREE")
    gate(sha(safe(MODEL))==MODEL_SHA,"MODEL_SHA")
    out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a12_g2_prepare_")
         and not out.exists() and not out.is_symlink(),"NEW_DIR")
    sources=original()
    out.mkdir(mode=0o700)
    (out/"measurements").mkdir()
    manifests={}; count=0; audit=[]
    for sig,val in SIGMAS:
        (out/"measurements"/("sigma_"+sig)).mkdir()
        rows=[];pairs=0
        for cohort,iid,yp,yd,gp,gfmt in sources:
            # GT never carried into rows, output manifests or GPU commands.
            x=gt_source({"gt_path":gp,"gt_format":gfmt}).to(dtype=torch.float32)
            old=read_tensor(yp,("measurement","y","observation"),(1,3,384,384))
            clean=magnitude(x)
            n=(old-clean)/0.05
            paired=clean+val*n
            gate(bool(torch.isfinite(paired).all()) and
                 tuple(paired.shape)==(1,3,384,384),"PAIRED_Y")
            if val==0.05:gate(bool(torch.allclose(paired,old)),"IDENTITY")
            dest=out/"measurements"/("sigma_"+sig)/(
               hashlib.sha256(f"{cohort}|{iid}".encode()).hexdigest()[:24]+".pt")
            torch.save({"measurement":paired.contiguous()},dest)
            digest=sha(dest)
            baseid=cohort+"_"+iid
            ss=[seed(sig,baseid,r) for r in range(8)]
            gate(len(set(ss))==8,"SEED_COLLISION")
            for r in range(8):
                rows.append(dict(
                   uid=f"XP|FAST|ffhq_s{sig}|{baseid}|rep{r}",
                   source="XP-G",measurement_source=f"XP_G_FFHQ_S{sig}",
                   image_id=baseid,rep=r,seed=ss[r],
                   measurement_path=str(dest),measurement_sha256=digest))
            audit.append(dict(cohort=cohort,image_id=iid,sigma=sig,
                prior_y_sha256=yd,derived_y_sha256=digest,
                noise_rms=float(n.square().mean().sqrt()),
                y_pair_shape=[1,3,384,384]))
            pairs+=1;count+=8
            if pairs%30==0:print("XP_G_PAIRED_MEASUREMENTS",sig,pairs,"/180",flush=True)
        gate(len(rows)==1440 and pairs==180,"PER_SIGMA_GRID")
        manifest=dict(schema="prdiff.track_r.xp.fast-tasks.v1",
           status="XP_FAST_TASKS_FROZEN",stage=f"A12_G2_SIGMA_{sig}",
           head=a.expected_head,fast_head=FAST_HEAD,
           dataset="ffhq",oversample=2.0,
           model_sha256=MODEL_SHA,n_roots=1440,n_images=180,
           gt_paths_supplied=False,original_gt_used_only_for_paired_noise=True,
           no_GT_in_candidate_manifest=True,
           paired_noise_formula="A(xgt)+sigma*(yR0-A(xgt))/0.05",
           sigma=val,seed_rule=f"SHA256(XP|FAST|ffhq_s{sig}|<cohort_id>|<rep>)[:8] mod2**31",
           outcome_embargo=False,rows=rows)
        p=out/("XP_A12_G2_SIGMA_"+sig+"_TASKS.json")
        write(p,manifest)
        manifests[sig]=dict(path=str(p),sha256=sha(p),n_roots=1440)
    gate(count==2880 and len(audit)==360,"TOTAL_GRID")
    write(out/"XP_A12_G2_PAIRED_Y_AUDIT.json",
        dict(status="XP_A12_G2_PAIRED_NOISE_FROZEN",
             no_candidate_outcomes_used=True,gt_used_only_in_measurement_prep=True,
             source_catalog_sha256=sha(CAT),n_measurements=360,pairs=audit))
    write(out/"XP_A12_G2_PREP_RECEIPT.json",
        dict(status="XP_A12_G2_INPUTS_FROZEN_CPU_PASS",head=a.expected_head,
             n_roots=2880,n_images=180,sigmas=["0.02","0.10"],
             per_sigma=manifests,paired_y_audit_sha256=sha(
                  out/"XP_A12_G2_PAIRED_Y_AUDIT.json"),
             no_GPU=True,no_GT_sent_to_worker=True,
             per_sigma_smoke_required=True,stage_ceiling_gpu_h=10,
             source_catalog_sha256=sha(CAT)))
    print("XP_A12_G2_CPU_PREP_2880_PASS",flush=True)
    for sig in manifests:
        print("G2_SIGMA",sig,"MANIFEST_SHA",manifests[sig]["sha256"],flush=True)
    print("G2_PREP_DIR",out,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as exc:
        traceback.print_exc();print("ACTION NEEDED: A12_G2_PREP",exc,flush=True)
        sys.exit(8)
