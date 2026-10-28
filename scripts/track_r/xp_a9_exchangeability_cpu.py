#!/usr/bin/env python3
"""A9.2(a) post-hoc FAST population exchangeability, CPU only.

Compares frozen R0 historical runs 0-3 with native A16 4-15 and A64 16-63
on the same 180 exposed FFHQ images. Verifies preregistered TH1 v1 before
reading any new outcomes, uses the historical S5_r0_fast GOOD-q95 tau, and
writes no theory_inputs. No GPU generation, foreign-process interaction,
new model or refitting. Candidate Z-tests are descriptive because runs
within an image are correlated; paired image-level sign-flip p-values
provide a cluster-respecting sensitivity analysis.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,subprocess,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path

import torch
from prdiffusion.track_r.physics import rho_signed,to_rgb8
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
from scripts.track_r.r0_2_g0_cpu import gt_source,read_tensor
from scripts.track_r.xp_a8_fast_v1_cpu import (
    BASE,AUTHOR_WT,head_guard,register_first,historical,future_inputs,
    R07_SCORES,safe,sha,GOOD
)

CLASSES=("G","O_g","O_p","C","I")
STAGES=(("HIST",0,4),("A16",4,16),("A64",16,64))
COHORTS={"S1_b22":"B22","S2_b24_dev80":"DEV80"}
FROZEN_TAU=0.05008293315768242
A8_SHA="f1fde0ba2cfa74eca65a7b47ba27edb1af8aaa6ae747449d1e825b7ec650da8e"
A8_RESULT=BASE/"xp_a8_fast_v1_20261028T090445Z"/"A8_FAST_V1_COMPARISON.json"

def stop_if(condition,name):
    if condition:raise RuntimeError("STOP_A9_EXCH_"+name)

def encode(psnr,rho):
    raw=float(psnr["000"])
    m=max(float(x) for x in psnr.values())
    stop_if(not math.isfinite(raw) or not math.isfinite(m) or
            not math.isfinite(rho) or rho<0 or m+1e-7<raw,
            "BAD_METRIC")
    if raw>=GOOD:return "G"
    if m>=GOOD:
        best=max(sorted(psnr),key=lambda p:float(psnr[p]))
        return "O_g" if best=="111" else "O_p"
    return "C" if rho<=FROZEN_TAU else "I"

def historical_rows():
    result=defaultdict(dict)
    with safe(R07_SCORES).open() as file:
        for line in file:
            row=json.loads(line)
            if row.get("source")!="S5_r0_fast":continue
            key=(row["measurement_source"],str(row["image_id"]))
            index=int(row["run_index"])
            stop_if(index not in range(4) or index in result[key],
                    "HIST_DUPLICATE")
            tau=float(row["tau_signed_good_q95"])
            stop_if(abs(tau-FROZEN_TAU)>1e-12,"TAU_DRIFT")
            psnr=row["psnr_by_orbit"]
            label=encode(psnr,float(row["rho_signed_raw"]))
            prior={"GOOD":"G","ORBIT_ERROR":"O_g" if row["orbit_subtype"]=="GLOBAL" else "O_p",
                   "NONORBIT_CONSISTENT":"C","NONORBIT_INCONSISTENT":"I"}
            stop_if(label!=prior[row["raw_label"]],"HIST_LABEL_DRIFT")
            result[key][index]=dict(run=index,label=label,rho=float(row["rho_signed_raw"]))
    stop_if(len(result)!=180 or any(set(x)!=set(range(4)) for x in result.values()),
            "HIST_PANEL")
    return result

def collect(identities,base,a16,a64):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    per_image={}
    processed=0
    for key in sorted(identities):
        cohort,iid=key
        gtpath,gtfmt,ypath,yhash=identities[key]
        stop_if(sha(ypath)!=yhash,"Y_SHA")
        gt=to_rgb8(gt_source({"gt_path":gtpath,"gt_format":gtfmt}).float())
        y=read_tensor(safe(ypath),("measurement","y","observation"),(1,3,384,384))
        values=dict(base[key])
        for rep in range(4,64):
            source=a16 if rep<16 else a64
            row,path=source[(cohort,iid,rep)]
            stop_if(row["measurement_path"]!=ypath or
                    row["measurement_sha256"]!=yhash,"MEAS_IDENTITY")
            item=torch.load(safe(path),map_location="cpu",weights_only=True)
            stop_if(not isinstance(item,dict) or
                    not isinstance(item.get("reconstruction"),torch.Tensor),
                    "RECON_CONTAINER")
            x=item["reconstruction"].float()
            stop_if(x.shape!=(1,3,256,256) or not torch.isfinite(x).all(),
                    "RECON_SHAPE")
            psnr=orbit_psnr_channel_mse(to_rgb8(x),gt)
            rho=float(rho_signed(x,y)[0])
            values[rep]=dict(run=rep,label=encode(psnr,rho),rho=rho)
            processed+=1
        stop_if(set(values)!=set(range(64)),"MISSING_RUN")
        per_image[key]=values
        if processed%1200==0:
            print("A9_EXCHANGEABILITY_SCORING",processed,"/10800",flush=True)
    stop_if(processed!=10800 or len(per_image)!=180,"COVERAGE")
    return per_image

def two_proportion(a,n,b,m):
    p=(a+b)/(n+m)
    variance=p*(1-p)*(1/n+1/m)
    if variance==0:return dict(z=0.,p=1.)
    z=(b/m-a/n)/math.sqrt(variance)
    return dict(z=z,p=math.erfc(abs(z)/math.sqrt(2)))

def image_signflip(values,seed,reps=8192):
    """Paired image-level mean-stage-rate differences under sign symmetry."""
    observed=sum(values)/len(values)
    if not any(abs(x)>1e-14 for x in values):
        return dict(p=1.,diff=0.)
    rng=random.Random(seed)
    larger=0
    for _ in range(reps):
        stat=sum((1 if rng.getrandbits(1) else -1)*v for v in values)/len(values)
        larger+=abs(stat)+1e-15>=abs(observed)
    return dict(p=(1+larger)/(reps+1),diff=observed)

def statistics(images):
    result=[]
    for cohort,name in COHORTS.items():
        cohort_images=[v for (c,_),v in sorted(images.items()) if c==cohort]
        stop_if(len(cohort_images)!=(100 if name=="B22" else 80),
                "COHORT_SIZE")
        for a,b in (("HIST","A16"),("A16","A64"),("HIST","A64")):
            stage=dict((s,(lo,hi)) for s,lo,hi in STAGES)
            la,ua=stage[a];lb,ub=stage[b]
            for label in CLASSES:
                c0=sum(v[r]["label"]==label for v in cohort_images for r in range(la,ua))
                c1=sum(v[r]["label"]==label for v in cohort_images for r in range(lb,ub))
                n0=len(cohort_images)*(ua-la);n1=len(cohort_images)*(ub-lb)
                two=two_proportion(c0,n0,c1,n1)
                diffs=[
                    sum(v[r]["label"]==label for r in range(lb,ub))/(ub-lb)-
                    sum(v[r]["label"]==label for r in range(la,ua))/(ua-la)
                    for v in cohort_images]
                perm=image_signflip(diffs,
                     int(hashlib.sha256(f"A9EXCH|{name}|{a}|{b}|{label}".encode())
                     .hexdigest()[:16],16))
                result.append(dict(cohort=name,class_label=label,
                    earlier_stage=a,later_stage=b,
                    n_images=len(cohort_images),earlier_successes=c0,
                    earlier_trials=n0,later_successes=c1,later_trials=n1,
                    earlier_rate=c0/n0,later_rate=c1/n1,
                    rate_difference=c1/n1-c0/n0,
                    two_proportion_z=two["z"],two_proportion_p=two["p"],
                    paired_image_signflip_p=perm["p"]))
    stop_if(len(result)!=30,"TEST_COUNT")
    order=sorted(range(len(result)),key=lambda i:result[i]["two_proportion_p"])
    high=0.
    for pos,index in enumerate(order):
        high=max(high,(len(result)-pos)*result[index]["two_proportion_p"])
        result[index]["two_proportion_holm_p"]=min(1.,high)
    return result

def rows_for_strings(images):
    rows=[]
    for (cohort,iid),group in sorted(images.items()):
        ranks=sorted(range(64),key=lambda j:(group[j]["rho"],j))
        counts=Counter(group[j]["label"] for j in range(64))
        row=dict(image_id=iid,measurement_source=cohort,cohort=COHORTS[cohort],
            labels_by_run=";".join(group[j]["label"] for j in range(64)),
            rho_signed_by_run=";".join(format(group[j]["rho"],".10g")
                                    for j in range(64)),
            e0_rank_order=";".join(map(str,ranks)),
            tau_signed_good_q95=repr(FROZEN_TAU))
        row.update({f"n_{k}":counts[k] for k in CLASSES})
        rows.append(row)
    stop_if(len(rows)!=180,"STRINGS_COUNT")
    return rows

def main(a):
    stop_if(os.environ.get("CUDA_VISIBLE_DEVICES")!="","GPU_VISIBLE")
    head_guard(a.expected_head)
    stop_if(sha(A8_RESULT)!=A8_SHA,"FROZEN_A8_SHA")
    out=Path(a.out)
    stop_if(out.parent!=BASE or not out.name.startswith("xp_a9_exchange_") or
            out.exists() or out.is_symlink(),"OUTPUT_PATH")
    reg,comp,forecast=register_first(a.expected_head)
    print("A9_EXCH_FROZEN_FORECAST_VERIFIED",len(forecast),flush=True)
    identities,scored=historical()
    base=historical_rows()
    stop_if(set(base)!=set(identities),"HIST_IDENTITY")
    a16,a64=future_inputs()
    images=collect(identities,base,a16,a64)
    tests=statistics(images)
    strings=rows_for_strings(images)
    stage_counts=defaultdict(Counter)
    for (cohort,iid),runs in images.items():
        for stage,lo,hi in STAGES:
            stage_counts[(COHORTS[cohort],stage)].update(
                runs[j]["label"] for j in range(lo,hi))
    payload=dict(status="A9_FAST_EXCHANGEABILITY_POSTHOC_PASS",
        frozen_v1_proof_verified=True,
        source_A8_sha256=A8_SHA,author_head=a.expected_head,
        n_images=180,n_candidates=11520,n_new_rescored=10800,
        tau_signed_good_q95=FROZEN_TAU,classes=list(CLASSES),
        strata=[dict(cohort=c,stage=s,n=sum(counter.values()),
                     counts={x:counter[x] for x in CLASSES},
                     rates={x:counter[x]/sum(counter.values()) for x in CLASSES})
                for (c,s),counter in sorted(stage_counts.items())],
        pairwise_two_proportion_tests=tests,
        methods=["nominal pooled two-proportion z-test; assumes independent candidates",
                 "Holm correction across 30 exploratory tests",
                 "paired-image sign-flip permutation sensitivity; 8192 draws"],
        cautions=["posthoc diagnostics; no model refit",
                 "candidate-level p-values are optimistic if repeated runs cluster by image",
                 "class differences are a protocol-drift signal, not automatic proof",
                 "FAST per-image strings must stay in xp/returns until v2 E1 is registered or A8.2 expires"],
        D1_P2_outcomes_opened=False,B16_F8_outcomes_opened=False)
    out.mkdir(parents=True,exist_ok=False)
    report=out/"A9_FAST_EXCHANGEABILITY.json"
    with report.open("x") as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    names=list(strings[0].keys())
    dest=out/"A9_FAST_64_PER_IMAGE_STRINGS.csv"
    with dest.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=names)
        w.writeheader();w.writerows(strings)
    print("A9_FAST_EXCHANGEABILITY_PASS",
          "n_tests=30 n_candidates=11520",flush=True)
    for row in payload["strata"]:
        print("A9_STRATUM",row["cohort"],row["stage"],row["counts"],flush=True)
    print("A9_HOLM_SIGNIFICANT",
          sum(v["two_proportion_holm_p"]<0.05 for v in tests),flush=True)
    print("A9_EXCHANGEABILITY_REPORT="+str(report),flush=True)
    print("A9_STRINGS_PATH="+str(dest),flush=True)
    print("A9_EXCHANGEABILITY_REPORT_SHA256="+sha(report),flush=True)
    print("A9_STRINGS_SHA256="+sha(dest),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A9_EXCHANGEABILITY",str(e),flush=True)
        sys.exit(8)
