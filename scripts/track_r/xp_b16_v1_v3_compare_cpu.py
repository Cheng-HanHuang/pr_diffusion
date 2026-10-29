#!/usr/bin/env python3
"""A11 B16 primary v1 and secondary v3 prospective comparison.

No theory refit. Exact all-subset E0/E1/O_raw/O_orbit probabilities for
each image's 16-run SITCOM panel, Good25. A9.2: phase composition
exchangeability, conditional PL selection and explicit failure case IDs.
Frozen v1/v3 authenticated by original source commits before reading scores.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path
from scripts.track_r.xp_b16_v1_v3_admission_cpu import (
 WT,BASE,FREEZES,freeze,git,sha,check)
from scripts.track_r.xp_a10_b16_gate_prepare_cpu import safe
from scripts.track_r.xp_a7_postfreeze_kernel_cpu import exact_panel,rank_prob,compare_interval
GOOD=25.
KS=(8,12,16)
POLICIES=("E0","E1_CLIP","O_raw","O_orbit")
VARIANTS=("race","S2","tail","race_tau1")
def load_csv(path):
    with safe(path).open(newline="") as f:
        return list(csv.DictReader(f))
def write_json(p,data):
    with p.open("x") as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
def load_inputs(a):
    score=Path(a.score_dir);clip=Path(a.clip_dir)
    r=json.loads(safe(score/"XP_B16_2880_SCORING_RECEIPT.json").read_text())
    file=score/"XP_B16_2880_SCORED_ROOTS.csv"
    strings=score/"XP_B16_16RUN_STRINGS.csv"
    check(r["status"]=="XP_B16_16RUN_2880_CPU_SCORED_PASS"
          and sha(file)==r["per_candidate_sha256"]
          and sha(strings)==r["strings_sha256"],"SCORED_INPUT_SHA")
    candidates=load_csv(file)
    grouped=defaultdict(dict)
    for c in candidates:
        cohort=c["cohort"];iid=c["image_id"];rep=int(c["run"])
        check(cohort in ("B22","DEV80")
              and rep not in grouped[cohort,iid]
              and rep in range(16),"SCORED_GROUP")
        grouped[cohort,iid][rep]=c
    check(len(grouped)==180 and len(candidates)==2880
          and all(set(v)==set(range(16)) for v in grouped.values()),
          "FULL_PANEL")
    clipreceipt=json.loads(safe(clip/"XP_B16_16RUN_FACE_CLIP_RECEIPT.json").read_text())
    clipfile=clip/"XP_B16_16RUN_FACE_CLIP_DECISIONS.csv"
    check(clipreceipt["status"]=="B16_2880_FROZEN_N2_FACECLIP_PASS"
          and sha(clipfile)==clipreceipt["csv_sha256"]
          and clipreceipt["source_scored_sha256"]==r["per_candidate_sha256"],
          "CLIP_SOURCE_SHA")
    face=load_csv(clipfile)
    clipmap={}
    for q in face:
        key=(q["cohort"],q["image_id"],int(q["run"]))
        check(key not in clipmap,"CLIP_DUPLICATE")
        clipmap[key]=q
    check(len(clipmap)==2880,"CLIP_2880")
    panel={}
    for (cohort,iid),runs in sorted(grouped.items()):
        rr=[]
        for j in range(16):
            row=runs[j]
            c=clipmap[cohort,iid,j]
            check(c["label"]==row["label"]
                  and c["f_choice"] in ("000","111")
                  and int(c["f_chosen_good"]) in (0,1),
                  "CLIP_LABEL_CONTRACT")
            rr.append(dict(run=j,rho=float(row["rho_signed"]),
               good_raw=row["label"]=="G",
               good_orbit=row["label"] in ("G","O_g","O_p"),
               chosen_good=int(c["f_chosen_good"]),
               label=row["label"],
               clip_margin=float(c["f_margin"]),
               psnr_raw=float(row["psnr_raw"]),
               psnr_orbit_best=float(row["psnr_orbit_best"])))
        panel[cohort,iid]=rr
    return panel,dict(scoring_sha=sha(file),strings_sha=sha(strings),
       clip_sha=sha(clipfile),scoring_receipt_sha=sha(score/"XP_B16_2880_SCORING_RECEIPT.json"),
       clip_receipt_sha=sha(clip/"XP_B16_16RUN_FACE_CLIP_RECEIPT.json"))
def e1_exact(items,k):
    n=len(items)
    rank=sorted(items,key=lambda r:(r["rho"],r["run"]))
    return sum(float(r["chosen_good"])*rank_prob(n,k,i+1)
               for i,r in enumerate(rank))
def grouped_values(panels):
    out=defaultdict(list)
    for (cohort,iid),rr in panels.items():
        out[cohort].append((iid,rr))
        out["POOLED"].append((cohort+":"+iid,rr))
    check(len(out["B22"])==100 and
          len(out["DEV80"])==80 and len(out["POOLED"])==180,
          "COHORT_IMAGE_COUNT")
    return out
def predictions():
    out={}
    for name,(ref,path,hash_,n) in FREEZES.items():
        blob=git("show",ref+":"+path)
        check(hashlib.sha256(blob).hexdigest()==hash_,"FORECAST_SHA_"+name)
        data=json.loads(blob)
        out[name]=data
    return out
def conditional_pl(runs,k,wO,wC):
    ng=sum(r["label"]=="G" for r in runs)
    no=sum(r["label"] in ("O_g","O_p") for r in runs)
    nc=sum(r["label"]=="C" for r in runs)
    ni=sum(r["label"]=="I" for r in runs)
    n=len(runs)
    denom=math.comb(n,k)
    result=0.
    for g in range(min(k,ng)+1):
      for o in range(min(k-g,no)+1):
        for c in range(min(k-g-o,nc)+1):
          i=k-g-o-c
          if i<0 or i>ni:continue
          chance=(math.comb(ng,g)*math.comb(no,o)*
              math.comb(nc,c)*math.comb(ni,i))/denom
          rate=g+wO*o+wC*c
          result+=chance*(g/rate if rate else 0.)
    return result
def analysis(panel,models):
    groups=grouped_values(panel)
    observed={}
    per_image=[]
    for cohort,rows in groups.items():
        for k in KS:
            values=defaultdict(list)
            for iid,rr in rows:
                x=exact_panel(rr,k)
                x["E1_CLIP"]=e1_exact(rr,k)
                for policy,val in x.items():
                    values[policy].append(val)
                    per_image.append(dict(cohort=cohort,image_id=iid,
                        K=k,policy=policy,exact_success=val))
            for policy,vals in values.items():
                observed[cohort,k,policy]=sum(vals)/len(vals)
            check(observed[cohort,k,"E0"]<=
                  observed[cohort,k,"O_raw"]+1e-10 and
                  observed[cohort,k,"O_raw"]<=
                  observed[cohort,k,"O_orbit"]+1e-10,"ORACLE_DOMINANCE")
    compare=[]
    for name,data in models.items():
        seen=set()
        for row in data["predictions"]:
            if (row["family"]!="SITCOM" or row["panel"]!="primary"
                or row["panel_n_max"]!=16):continue
            key=(row["cohort"],int(row["K"]),row["policy"])
            check(key in observed,"FORECAST_GRID_"+name)
            sig=(row["cohort"],row["model_variant"],row["K"],row["policy"])
            check(sig not in seen,"FORECAST_DUP_"+name)
            seen.add(sig)
            c=compare_interval(observed[key],dict(row,
                 interval_type=data["interval_type"]))
            compare.append(dict(forecast_source=name,
                 family="SITCOM",cohort=row["cohort"],
                 panel="primary",panel_n_max=16,
                 K=int(row["K"]),policy=row["policy"],
                 model_variant=row["model_variant"],**c))
        check(len(seen)==144,"FORECAST_ROWS_"+name)
    coverage={}
    for name in models:
        coverage[name]={}
        for var in VARIANTS:
            items=[r for r in compare if r["forecast_source"]==name
                   and r["model_variant"]==var]
            check(len(items)==36,"COVERAGE_ROWS_"+name+"_"+var)
            coverage[name][var]=dict(n=len(items),
                                    inside=sum(bool(r["inside"]) for r in items),
                                    worst_abs_standardized_residual=max(abs(r["standardized_residual"]) for r in items))
    return observed,compare,coverage,per_image
def phase_audit(panel):
    out=[]
    for cohort,rows in grouped_values(panel).items():
        for label in ("G","O_g","O_p","C","I"):
            early=sum(r["label"]==label for _,items in rows for r in items[:4])
            later=sum(r["label"]==label for _,items in rows for r in items[4:])
            diffs=[sum(r["label"]==label for r in items[4:])/12-
                   sum(r["label"]==label for r in items[:4])/4
                   for _,items in rows]
            rng=random.Random(20261029+sum(ord(x) for x in cohort+label))
            delta=sum(diffs)/len(diffs)
            N=8192
            n_ge=sum(abs(sum(d*(1 if rng.getrandbits(1) else -1) for d in diffs)/len(diffs))
                     >=abs(delta)-1e-15 for _ in range(N))
            out.append(dict(cohort=cohort,label=label,
                early_count=early,later_count=later,
                early_rate=early/(4*len(rows)),
                later_rate=later/(12*len(rows)),
                paired_image_difference=delta,
                signflip_p=(n_ge+1)/(N+1)))
    return out
def cond_audit(panel,models,obs):
    fits={r["cohort"]:r for r in models["v3"]["hyperparameter_fits"]
          if r["family"]=="SITCOM"}
    check(set(fits)=={"B22","DEV80"},"PL_FROZEN_WEIGHTS")
    rows=grouped_values(panel)
    out=[]
    for cohort in ("B22","DEV80","POOLED"):
        for k in KS:
            for variant in ("race","S2","race_tau1"):
                projected=[]
                for name,rr in rows[cohort]:
                    c=cohort if cohort!="POOLED" else name.split(":")[0]
                    f=fits[c]
                    weights=(dict(w_O=1.,w_C=1.) if variant=="S2" else
                         f["race_weights" if variant=="race" else "race_tau1_weights"])
                    projected.append(conditional_pl(rr,k,weights["w_O"],weights["w_C"]))
                mean=sum(projected)/len(projected)
                out.append(dict(cohort=cohort,K=k,variant=variant,
                     frozen_PL_given_realized_class_counts=mean,
                     observed_E0=obs[cohort,k,"E0"],
                     model_minus_observed=mean-obs[cohort,k,"E0"],
                     n_images=len(projected),no_refit=True))
    return out
def cases(panel):
    never=[]
    loss=[]
    for (cohort,iid),runs in sorted(panel.items()):
        ng=sum(r["good_raw"] for r in runs)
        no=sum(r["good_orbit"] for r in runs)
        winner=min(runs,key=lambda r:(r["rho"],r["run"]))
        if ng==0:
            never.append(dict(cohort=cohort,image_id=iid,
               n_G=ng,n_orbit_good=no,
               labels_by_run=";".join(r["label"] for r in runs)))
        if ng>0 and not winner["good_raw"]:
            loss.append(dict(cohort=cohort,image_id=iid,
                n_G=ng,selected_run=winner["run"],
                selected_label=winner["label"],
                selected_rho=winner["rho"],
                selected_good_after_face_clip=bool(winner["chosen_good"]),
                all_labels=";".join(r["label"] for r in runs),
                raw_oracle_16=1,selected_E0_16=0))
    return never,loss
def main(a):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    check(git("rev-parse","HEAD").decode().strip()==a.expected_head
          and not git("status","--porcelain"),"ANALYSIS_HEAD")
    checked=freeze()
    model=predictions()
    check(set(checked)==set(model),"FORECAST_CHECK")
    panel,source=load_inputs(a)
    obs,comparisons,coverage,per_image=analysis(panel,model)
    conditional=cond_audit(panel,model,obs)
    exchange=phase_audit(panel)
    never,loss=cases(panel)
    out=Path(a.out)
    check(out.parent==BASE and out.name.startswith("xp_b16_v1_v3_")
          and not out.exists() and not out.is_symlink(),"OUTPUT")
    out.mkdir()
    summary=dict(status="XP_B16_V1_PRIMARY_V3_SECONDARY_PROSPECTIVE_COMPARISON_PASS",
         family="SITCOM",n_images=180,n_runs=16,
         cohort_images={"B22":100,"DEV80":80},
         frozen_v1_primary_sha256=FREEZES["v1"][2],
         frozen_v3_secondary_sha256=FREEZES["v3"][2],
         registered_before_unblind=True,no_model_refit=True,
         Good25=25,policy_selection="exact_all_subset_U_statistic",
         n_forecast_rows=len(comparisons),
         forecast_comparisons=comparisons,
         per_variant_coverage=coverage,
         observed=[dict(cohort=co,K=k,policy=pol,value=value)
                   for (co,k,pol),value in sorted(obs.items())],
         A9_2_phase_exchangeability=exchange,
         A9_2_composition_conditional=conditional,
         n_never_GOOD=len(never),
         n_K16_raw_selection_loss=len(loss),
         never_GOOD_cases=never,
         K16_selection_loss_cases=loss,
         source=source,
         face_clip="frozen N2 three-face prompts; post-hoc chosen GOOD after orientation",
         no_GPU_called=True)
    target=out/"XP_B16_V1_V3_PROSPECTIVE_COMPARISON.json"
    write_json(target,summary)
    per=out/"XP_B16_V1_V3_PER_IMAGE_EXACT.csv"
    with per.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(per_image[0]))
        w.writeheader();w.writerows(per_image)
    receipt=out/"XP_B16_V1_V3_COMPARISON_RECEIPT.json"
    write_json(receipt,dict(status=summary["status"],
          comparison_sha256=sha(target),per_image_sha256=sha(per),
          expected_head=a.expected_head,n_images=180,
          rows=len(comparisons),n_never_good=len(never),
          n_K16_E0_misses_with_raw_oracle=len(loss)))
    print("B16_V1_V3_PROSPECTIVE_COMPARISON_PASS",
          "forecast_rows",len(comparisons),"never_GOOD",len(never),
          "K16_selection_losses",len(loss),flush=True)
    print("B16_PRIMARY_V1_COVERAGE",coverage["v1"],flush=True)
    print("B16_SECONDARY_V3_COVERAGE",coverage["v3"],flush=True)
    print("B16_ANALYSIS_OUT="+str(out),flush=True)
    print("B16_COMPARISON_SHA="+sha(target),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--score-dir",required=True)
    p.add_argument("--clip-dir",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: B16_COMPARISON "+str(e),flush=True)
        sys.exit(8)
