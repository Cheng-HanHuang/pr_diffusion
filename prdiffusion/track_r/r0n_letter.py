"""R0-N frozen, CPU-only letter-table statistics.

No data-loading or GPU code. Keep S1/S2 physically disjoint and
source/family/measurement-specific policies; pooled analyses use image
cluster uncertainty and never claim prospective validation.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import math
import statistics

import numpy as np
from prdiffusion.track_r.r0_7_stats import (
    GOOD,ORBIT,BOOT_N,auc_roc,clustered_ratio_ci,select_e,summarize_psnr)

TIMES=(50,100,200,300,400,500,600,700)
ORBIT_MEMBERS=tuple(f"{i:03b}" for i in range(8))
EXPECTED_LABELS={"GOOD":2040,"ORBIT_ERROR":85,
 "NONORBIT_CONSISTENT":324,"NONORBIT_INCONSISTENT":791}
P4_GOOD25_WITNESSES={
 ("S1_b22","DAPS",2):(100,.92,.95,.95,.95,.95),
 ("S1_b22","SITCOM",4):(100,.89,.94,.96,.95,.96),
 ("S1_b22","NP",8):(100,.94,.94,.94,.94,.96),
 ("S2_b24_dev80","DAPS",4):(80,.825,.8625,.85,.85,.8625),
 ("S2_b24_dev80","SITCOM",4):(80,.6875,.725,.7125,.675,.725),
 ("S2_b24_dev80","NP",4):(80,.7,.7,.7125,.7125,.7125),
 ("S5_r0_fast","FAST",4):(180,.8555555555555555,
    .8611111111111112,.8611111111111112,.8555555555555555,
    .8666666666666667)}

def assert_replay(rows,report):
    if len(rows)!=3240 or len({r["uid"] for r in rows})!=3240:
        raise RuntimeError("STOP_R0N_K3_COUNT")
    counts=Counter(r["raw_label"] for r in rows)
    if dict(counts)!=EXPECTED_LABELS:
        raise RuntimeError("STOP_R0N_K3_LABEL_COUNTS "+str(counts))
    errors=[r for r in rows if r["raw_label"]=="ORBIT_ERROR"]
    good=[r for r in rows if r["raw_label"]=="GOOD"]
    if (sum(r["psnr_repaired"]>=GOOD for r in errors)!=84 or
        sum(r["psnr_repaired"]<GOOD for r in good)!=0):
        raise RuntimeError("STOP_R0N_K3_REPAIR_REPLAY")
    tbl={(r["source"],r["family"],int(r["K"])):r for r in report["P4"]}
    for k,expected in P4_GOOD25_WITNESSES.items():
        r=tbl.get(k)
        if not r or r["n_images_available"]!=expected[0]:
            raise RuntimeError("STOP_R0N_K3_P4_STRATUM "+str(k))
        for i,name in enumerate(("E0","E1","E2","E3","O_orbit"),1):
            value=r["policy_stats"][name]["good25"]
            if not math.isclose(value,expected[i],abs_tol=1e-10):
                raise RuntimeError("STOP_R0N_K3_P4_GOOD25 "+str((k,name,value)))
    return {"status":"R0N_K3_REPLAY_PASS","n":3240,"labels":dict(counts),
            "repair_orbit_success":84,"repair_orbit_denominator":85,
            "harm_good_count":0,"harm_good_denominator":2040,
            "p4_witnesses_verified":7}

def split_ids(rows):
    b22={str(r["image_id"]) for r in rows if r["source"]=="S1_b22"}
    dev={str(r["image_id"]) for r in rows if r["source"]=="S2_b24_dev80"}
    if len(b22)!=100 or len(dev)!=80 or b22&dev:
        raise RuntimeError("STOP_R0N_COHORT_IDS_OVERLAP_OR_COUNT")
    for r in rows:
        iid=str(r["image_id"])
        if iid not in b22 and iid not in dev:
            raise RuntimeError("STOP_R0N_UNASSIGNED_IMAGE "+r["uid"])
    return {"B22_random":b22,"DEV80_enriched":dev}

def cohort(r,splits):
    img=str(r["image_id"])
    if img in splits["B22_random"]:return "B22_random"
    if img in splits["DEV80_enriched"]:return "DEV80_enriched"
    raise RuntimeError("STOP_R0N_NO_COHORT")

def label_anatomy(rows,splits):
    grouped=defaultdict(list)
    for r in rows:grouped[(cohort(r,splits),r["family"])].append(r)
    tables=[]
    for (name,family),items in sorted(grouped.items()):
        labels=Counter(r["raw_label"] for r in items)
        nbad=len(items)-labels["GOOD"]
        global_n=sum(r["raw_label"]=="ORBIT_ERROR" and r["s_gt"]=="111" for r in items)
        orbit=labels["ORBIT_ERROR"]
        report={"cohort":name,"family":family,
            "n_runs":len(items),"n_images":len({r["image_id"] for r in items}),
            "GOOD":labels["GOOD"],"ORBIT_ERROR":orbit,
            "ORBIT_GLOBAL":global_n,"ORBIT_PARTIAL":orbit-global_n,
            "NONORBIT_CONSISTENT":labels["NONORBIT_CONSISTENT"],
            "NONORBIT_INCONSISTENT":labels["NONORBIT_INCONSISTENT"],
            "n_failures":nbad,
            "failure_fraction":nbad/len(items),
            "orbit_share_failures":clustered_ratio_ci(
                items,lambda r:r["raw_label"]=="ORBIT_ERROR",
                lambda r:r["raw_label"]!="GOOD",
                "R0N_N1a|"+name+"|"+family)}
        tables.append(report)
    return tables

def image_run_groups(rows,splits):
    grouped=defaultdict(list)
    for r in rows:
        key=(r["source"],r["family"],cohort(r,splits),
             r["measurement_source"],str(r["image_id"]))
        grouped[key].append(r)
    return grouped

def policy_values(rows,k,tau):
    val=select_e(rows,k,tau)
    if val is None:return None
    ordered=sorted(rows,key=lambda r:(r["run_index"],r["uid"]))[:k]
    e0=min(ordered,key=lambda r:(r["rho_signed_raw"],r["run_index"]))
    e0_fail=e0["psnr_raw"]<GOOD
    e1_good=e0["psnr_repaired"]>=GOOD
    oracle=max(r["psnr_orbit_best"] for r in ordered)
    if e0_fail:
        if e1_good:failure_kind="ORBIT_FIXED"
        elif e0["psnr_orbit_best"]>=GOOD:failure_kind="ORBIT_NOT_FIXED"
        elif oracle>=GOOD:failure_kind="SELECTION_MISS"
        else:failure_kind="GENERATION_FAILURE"
    else:failure_kind="E0_GOOD"
    return {"psnr":val["psnr"],"failure_kind":failure_kind,
            "n_feasible":val["n_feasible"],"E0_uid":e0["uid"],
            "n_candidates":len(ordered)}

def postselection(rows,splits,report):
    groups=image_run_groups(rows,splits)
    tuples={(r["source"],r["family"],r["K"]) for r in report["P4"]}
    if len(tuples)!=len(report["P4"]):
        raise RuntimeError("STOP_R0N_DUPLICATE_P4_TABLE")
    result=[]
    for src,fam,k in sorted(tuples):
        for name in ("B22_random","DEV80_enriched"):
            relevant=[rr for (s,f,co,origin,iid),rr in groups.items()
                if s==src and f==fam and co==name]
            if not relevant:continue
            evaluations=[]
            for rr in relevant:
                tau=rr[0]["tau_signed_good_q95"]
                v=policy_values(rr,k,tau)
                if v is not None:evaluations.append(v)
            if not evaluations:continue
            kinds=Counter(v["failure_kind"] for v in evaluations)
            n_fail=len(evaluations)-kinds["E0_GOOD"]
            if sum(kinds.values())!=len(evaluations):
                raise RuntimeError("STOP_R0N_N1B_PARTITION")
            scores={label:[v["psnr"][label] for v in evaluations]
                for label in ("O_raw","O_orbit","E0","E1","E2","E3")}
            out={"cohort":name,"source":src,"family":fam,"K":k,
                "n_images":len(evaluations),"E0_failures":n_fail,
                "orbit_fixed":kinds["ORBIT_FIXED"],
                "orbit_not_fixed":kinds["ORBIT_NOT_FIXED"],
                "selection_miss":kinds["SELECTION_MISS"],
                "generation_failure":kinds["GENERATION_FAILURE"],
                "E0_good":kinds["E0_GOOD"],
                "O_raw_minus_E1_mean":statistics.fmean(scores["O_raw"])-
                                      statistics.fmean(scores["E1"]),
                "policy_stats":{p:summarize_psnr(z) for p,z in scores.items()}}
            if n_fail!=sum(out[k] for k in (
                "orbit_fixed","orbit_not_fixed","selection_miss","generation_failure")):
                raise RuntimeError("STOP_R0N_N1B_FAILURE_PARTITION")
            result.append(out)
    # When no cohort split is needed, replay R0.7's existing K table.
    by={(r["source"],r["family"],r["K"]):r for r in report["P4"]}
    for key,t in by.items():
        subset=[r for r in result if (r["source"],r["family"],r["K"])==key]
        if sum(r["n_images"] for r in subset)!=t["n_images_available"]:
            raise RuntimeError("STOP_R0N_N1B_R07_IMAGE_REPLAY "+str(key))
        if len(subset)==1:
            for p in ("E0","E1","E2","E3","O_raw","O_orbit"):
                x=subset[0]["policy_stats"][p]
                y=t["policy_stats"][p]
                if x is None or y is None:
                    if x!=y:raise RuntimeError("STOP_R0N_N1B_NULL_P4")
                    continue
                if abs(x["good25"]-y["good25"])>1e-9 or abs(x["mean"]-y["mean"])>1e-6:
                    raise RuntimeError("STOP_R0N_N1B_METRIC_REPLAY "+str((key,p)))
    return result

def residuals(rows,splits):
    grouped=defaultdict(list)
    for r in rows:grouped[(cohort(r,splits),r["family"],r["raw_label"])].append(r)
    def quantiles(v):
        a=np.asarray(v,dtype=float)
        return {"median":float(np.median(a)),
            "q10":float(np.quantile(a,.10,method="linear")),
            "q90":float(np.quantile(a,.90,method="linear"))}
    out=[]
    for (name,fam,label),rr in sorted(grouped.items()):
        signed=quantiles([r["rho_signed_raw"] for r in rr])
        clipped=quantiles([r["rho_clipped_raw"] for r in rr])
        out.append({"cohort":name,"family":fam,"label":label,
            "n":len(rr),"n_images":len({r["image_id"] for r in rr}),
            "signed":signed,"clipped":clipped,
            "good_signed_below_0p05":(
                sum(r["rho_signed_raw"]<.05 for r in rr)/len(rr)
                if label=="GOOD" else None)})
    return out

def cost_configs(cfg):
    table=cfg["n1d_cost_accuracy"]
    if table["timestep_index_order"]!=list(TIMES) or table["J_subsets"]!=[1,2,4]:
        raise RuntimeError("STOP_R0N_N1D_CONFIG_CHANGED")
    t_sets=[("single_"+str(t),(t,)) for t in TIMES]
    t_sets+=[("mid4",tuple(table["T_subsets"]["mid4"])),
             ("all8",tuple(table["T_subsets"]["all8"]))]
    members=table["member_sets"]
    if list(members)!=(["all8","global2"]) or len(t_sets)!=10:
        raise RuntimeError("STOP_R0N_N1D_SETS_CHANGED")
    out=[]
    for n,ts in t_sets:
        for j in table["J_subsets"]:
            for label,member_set in members.items():
                out.append({"name":f"{n}|J{j}|{label}",
                  "T":list(ts),"J":j,"member_set":label,
                  "members":member_set,
                  "forwards":len(ts)*j*len(member_set)})
    if len(out)!=60 or len({x["name"] for x in out})!=60:
        raise RuntimeError("STOP_R0N_N1D_EXPECT_60")
    return out

def score_from_draws(draws,configuration):
    tmap={t:i for i,t in enumerate(TIMES)}
    ts=configuration["T"];J=configuration["J"]
    scores={}
    for pattern in configuration["members"]:
        losses=draws[pattern]
        if len(losses)!=32 or any(not math.isfinite(float(v)) for v in losses):
            raise RuntimeError("STOP_R0N_N1D_LOSS_VECTOR")
        selected=[float(losses[4*tmap[t]+j]) for t in ts for j in range(J)]
        scores[pattern]=sum(selected)/len(selected)
    best=min(configuration["members"],key=lambda p:(scores[p],p))
    return "000" if scores["000"]-scores[best]<=1e-7 else best

def cost_grid_metrics(rows,decisions,configs,splits):
    # Ground truth appears only in the *analysis* metrics, not in
    # GT-free repair decision selection.
    bysplit={"B22_random":[],"DEV80_enriched":[]}
    for r in rows:bysplit[cohort(r,splits)].append(r)
    out=[]
    for config in configs:
        for name,rs in bysplit.items():
            orbit=[r for r in rs if r["raw_label"]=="ORBIT_ERROR"]
            good=[r for r in rs if r["raw_label"]=="GOOD"]
            selected={r["uid"]:decisions[r["uid"]][config["name"]] for r in rs}
            success=sum(r["psnr_by_orbit"][selected[r["uid"]]]>=GOOD for r in orbit)
            exact=sum(selected[r["uid"]]==r["s_gt"] for r in orbit)
            harm=sum(r["psnr_by_orbit"][selected[r["uid"]]]<GOOD for r in good)
            out.append({"configuration":config["name"],
                "member_set":config["member_set"],"T":config["T"],"J":config["J"],
                "forwards":config["forwards"],"cohort":name,
                "n_candidates":len(rs),"n_orbit_error":len(orbit),
                "n_good":len(good),"n_success":success,"n_exact":exact,
                "n_harm":harm,
                "success":success/len(orbit) if orbit else None,
                "exact":exact/len(orbit) if orbit else None,
                "harm":harm/len(good) if good else None})
    return out

def choose_dev(grid,configs):
    look={c["name"]:c for c in configs}
    eligible=[r for r in grid if r["cohort"]=="DEV80_enriched"
       and r["success"] is not None and r["success"]>=.95
       and r["n_harm"]==0]
    def rank(x):
        c=look[x["configuration"]]
        label=c["name"].split("|")[0]
        priority=0 if label=="all8" else (1 if label=="mid4" else 2)
        t_priority=int(label.split("_")[1]) if label.startswith("single_") else 0
        return (c["forwards"],0 if c["member_set"]=="all8" else 1,
                -c["J"],priority,t_priority)
    return min(eligible,key=rank) if eligible else None

def paired_variance(rows,draws):
    grouped=defaultdict(list)
    for r in rows:
        if r["raw_label"] not in ("GOOD","ORBIT_ERROR"):continue
        a=draws[r["uid"]]["000"]
        b=draws[r["uid"]]["111"]
        for i,t in enumerate(TIMES):
            x=np.asarray(a[i*4:i*4+4],dtype=float)
            y=np.asarray(b[i*4:i*4+4],dtype=float)
            denom=x.var(ddof=1)+y.var(ddof=1)
            if denom<=0:continue
            ratio=float((x-y).var(ddof=1)/denom)
            grouped[(r["raw_label"],t)].append(ratio)
            grouped[(r["raw_label"],"ALL")].append(ratio)
    result=[]
    for (label,t),values in sorted(grouped.items(),key=lambda x:(x[0][0],str(x[0][1]))):
        x=np.asarray(values)
        result.append({"label":label,"t":t,"n":len(values),
            "median":float(np.median(x)),
            "q25":float(np.quantile(x,.25,method="linear")),
            "q75":float(np.quantile(x,.75,method="linear"))})
    return result

def conditional_detection(rows):
    families=sorted({r["family"] for r in rows})
    out=[]
    for fam in families:
        base=[r for r in rows if r["family"]==fam]
        selected=[r for r in base if not r["after_repair_failure"] or
              (r["after_repair_failure"] and
               r["raw_label"]=="NONORBIT_CONSISTENT")]
        y=[bool(r["after_repair_failure"]) for r in selected]
        keys={"S":"S_repaired","S_img":"S_img",
              "UnivFD":"p_fake_repaired","rho_signed":"rho_signed_repaired",
              "rho_clipped":"rho_clipped_repaired"}
        out.append({"family":fam,"label":"EXPLORATORY_CONDITIONAL_DETECTION",
            "n":len(selected),"n_failure":sum(y),
            "n_success":len(y)-sum(y),
            "auroc":{label:auc_roc(y,[r[key] for r in selected])
                for label,key in keys.items()}})
    return out

def table_two(rows,post):
    out=[]
    for r in post:
        p=r["policy_stats"]
        out.append({
            "cohort":r["cohort"],"source":r["source"],
            "family":r["family"],"K":r["K"],
            "n_images":r["n_images"],
            "O_raw":{k:p["O_raw"][k] for k in ("good25","mean","q05")},
            "O_orbit":{k:p["O_orbit"][k] for k in ("good25","mean","q05")},
            "E0":{k:p["E0"][k] for k in ("good25","mean","q05")},
            "E1":{k:p["E1"][k] for k in ("good25","mean","q05")}})
    maxk={}
    for r in out:
        key=(r["source"],r["family"],r["cohort"])
        if key not in maxk or r["K"]>maxk[key]["K"]:maxk[key]=r
    return sorted(maxk.values(),key=lambda r:(r["cohort"],r["source"],r["family"]))
