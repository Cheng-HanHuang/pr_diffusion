#!/usr/bin/env python3
"""R0-N.2 B2 + A8. Offline CPU; immutable R0.7 labels and frozen N2 decisions.
No model inference, image tree walk, or new GT access. Fail closed on drift.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
R07=BASE/"r0_7_cpu_20261023T080434Z"
N2=BASE/"r0n_n2clip_recovery_20261024T073405Z"
META=ROOT/"data/ffhq/ffhq-dataset/ffhq-dataset-v2.json"
FORBIDDEN=("confirmation","confirm305")
EXPECTED=Counter({"GOOD":2040,"ORBIT_ERROR":85,"NONORBIT_CONSISTENT":324,"NONORBIT_INCONSISTENT":791})
FIELDS=("image_id","cohort","source","family","ffhq_category","n_runs","n_GOOD",
    "n_ORBIT_global","n_ORBIT_partial","n_CONSISTENT","n_INCONSISTENT",
    "run_labels","e0_rank_labels","e0_rank_order","orbit_best_member",
    "clip_fix_ok","dec_clip_face","dec_2f","dec_s_all8")

def safe(p):
    s=str(p)
    if any(token in s.lower() for token in FORBIDDEN):
        raise RuntimeError("STOP_PROTECTED_PATH")
    p=Path(s)
    if p.is_symlink() or not p.is_file():
        raise RuntimeError("STOP_MISSING_OR_SYMLINK "+s)
    if any(token in str(p.resolve()).lower() for token in FORBIDDEN):
        raise RuntimeError("STOP_PROTECTED_REALPATH")
    return p

def sha(p):
    h=hashlib.sha256()
    with safe(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def obj(p):
    return json.loads(safe(p).read_text(encoding="utf8"))

def rows_jsonl(p):
    with safe(p).open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]

def category_lookup(meta,iid):
    ident=str(iid)
    candidates=(str(int(Path(ident).stem)),ident,ident.zfill(5))
    for key in candidates:
        x=meta.get(key) if isinstance(meta,dict) else None
        if isinstance(x,dict):
            for y in (x,x.get("metadata"),x.get("image"),x.get("info")):
                if isinstance(y,dict) and y.get("category") in ("training","validation"):
                    return y["category"]
    raise RuntimeError("STOP_B2_CATEGORY_UNRESOLVED "+ident)

def frozen_cohort_map(records):
    """Recover the 100/80 split from R0 sources, not the mixed FAST label."""
    result={}
    seen={"S1_b22":set(),"S2_b24_dev80":set()}
    names={"S1_b22":"B22_random","S2_b24_dev80":"DEV80_enriched"}
    for r in records:
        source=r["source"]
        if source not in names:
            continue
        iid=str(int(r["image_id"]))
        cohort=names[source]
        if iid in result and result[iid]!=cohort:
            raise RuntimeError("STOP_COHORT_ID_OVERLAP")
        result[iid]=cohort
        seen[source].add(iid)
    if len(seen["S1_b22"])!=100 or len(seen["S2_b24_dev80"])!=80 or len(result)!=180:
        raise RuntimeError("STOP_COHORT_100_80_SPLIT")
    for r in records:
        if r["source"]=="S5_r0_fast" and str(int(r["image_id"])) not in result:
            raise RuntimeError("STOP_FAST_ID_NOT_IN_FROZEN_180")
    return result

def check_row(row):
    for k in ("uid","source","family","image_id","run_index","raw_label",
              "psnr_raw","psnr_by_orbit","rho_signed_raw","s_gt"):
        if k not in row:raise RuntimeError("STOP_R07_SCHEMA "+k)
    if row["raw_label"] not in EXPECTED:
        raise RuntimeError("STOP_R07_LABEL "+str(row["uid"]))
    best=max(sorted(row["psnr_by_orbit"]),key=lambda x:row["psnr_by_orbit"][x])
    if row["s_gt"]!=best:raise RuntimeError("STOP_ORBIT_REPLAY")
    if row["raw_label"]=="GOOD" and row["psnr_raw"]<25:
        raise RuntimeError("STOP_GOOD_REPLAY")

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_CPU_ONLY")
    wt=Path(__file__).resolve().parents[2]
    head=subprocess.check_output(["git","-C",str(wt),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(wt),"status","--porcelain"],text=True).strip()
    branch=subprocess.check_output(["git","-C",str(wt),"branch","--show-current"],text=True).strip()
    if dirty or head!=a.expected_head or branch!= "codex/prdiff-track-r-xp-20261026":
        raise RuntimeError("STOP_UNCLEAN_OR_WRONG_HEAD")
    out=Path(a.out)
    if not str(out).startswith(str(BASE/"r0n2_")) or out.exists():
        raise RuntimeError("STOP_OUTPUT_NAMESPACE")
    evaluation=R07/"analysis/R0_7_SCORED_GT_ROWS.jsonl"
    receipt=obj(R07/"analysis/R0_7_COMPLETE_RECEIPT.json")
    if (receipt.get("status")!="R0_7_ANALYSIS_COMPLETE" or
        receipt.get("n_scored")!=3240 or
        receipt.get("no_protected_access") is not True or
        sha(evaluation)!=receipt.get("evaluated_rows_sha256")):
        raise RuntimeError("STOP_R07_DRIFT")
    records=rows_jsonl(evaluation)
    if len(records)!=3240 or len({r["uid"] for r in records})!=3240:
        raise RuntimeError("STOP_R07_COUNT")
    all_counts=Counter(r["raw_label"] for r in records)
    if all_counts!=EXPECTED:raise RuntimeError("STOP_K3_GLOBAL_REPLAY")
    for r in records:check_row(r)
    cohort_ids=frozen_cohort_map(records)
    cats=obj(META)
    category={str(r["image_id"]):category_lookup(cats,r["image_id"]) for r in records}
    scores=obj(N2/"N2_GT_FREE_CLIP_SCORES.json")
    score_receipt=obj(N2/"N2_CLIP_GPU_RECEIPT.json")
    if (scores.get("n")!=3240 or
        scores.get("scoring_gt_access") is not False or
        scores.get("status")!="R0N_N2_ALL_3240_GT_FREE_CLIP_PASS" or
        sha(N2/"N2_GT_FREE_CLIP_SCORES.json")!=score_receipt.get("scored_sha256")):
        raise RuntimeError("STOP_N2_DRIFT")
    clips={r["uid"]:r for r in scores["rows"]}
    if len(clips)!=3240 or set(clips)!={r["uid"] for r in records}:
        raise RuntimeError("STOP_N2_UID_MISMATCH")
    catalog=obj(R07/"prepare/R0_7_OFFLINE_GT_CATALOG.json")
    catalog_rows={r["uid"]:r for r in catalog["rows"]}
    if (catalog.get("n_candidates")!=3240 or len(catalog_rows)!=3240 or
        set(catalog_rows)!=set(clips)):
        raise RuntimeError("STOP_CATALOG_DRIFT")
    by_group=defaultdict(list)
    categories=defaultdict(Counter)
    for r in records:
        src=r["source"]; fam=r["family"];iid=str(r["image_id"])
        cohort=cohort_ids.get(str(int(iid)),"OTHER_EXPOSED")
        cat=category[iid]
        # Read exactly the receipt named by the approved R0.7 catalog, never its GT/candidate paths.
        catrow=catalog_rows[r["uid"]]
        scorepath=safe(catrow["score_receipt_path"])
        if sha(scorepath)!=catrow["score_receipt_sha256"]:
            raise RuntimeError("STOP_SCORE_DRIFT "+r["uid"])
        prior=obj(scorepath)["prior"]
        draws=prior["draw_losses"]
        if set(draws)!={f"{i:03b}" for i in range(8)} or any(len(v)!=32 for v in draws.values()):
            raise RuntimeError("STOP_DRAW_SCHEMA "+r["uid"])
        choice=clips[r["uid"]]["choice"]
        if choice not in ("000","111"):raise RuntimeError("STOP_N2_CHOICE")
        # Recorded N1d t=300,J=1,global2 corresponds to t index 3, draw j=0.
        choice2="111" if float(draws["111"][12])<float(draws["000"][12])-1e-7 else "000"
        s_choice=min(sorted(prior["S"]),key=lambda x:float(prior["S"][x]))
        s_choice="000" if float(prior["S"]["000"])-float(prior["S"][s_choice])<=1e-7 else s_choice
        if s_choice!=r["s_star"]:raise RuntimeError("STOP_PRIOR_REPLAY")
        key=(src,fam,iid)
        by_group[key].append((r,choice,choice2,s_choice))
        categories[(src,cohort,fam,cat)]["n"]+=1
        good=r["psnr_raw"]>=25
        categories[(src,fam,cat)]["good"]+=int(good)
        for name,ch in (("CLIP",choice),("S",("111" if prior["S"]["111"]<prior["S"]["000"]-1e-7 else "000")),
                        ("SMID",("111" if prior["S_mid"]["111"]<prior["S_mid"]["000"]-1e-7 else "000"))):
            if r["raw_label"]=="ORBIT_ERROR" and r["s_gt"]=="111":
                categories[(src,fam,cat)][name+"_orb"]+=1
                categories[(src,fam,cat)][name+"_fix"]+=int(r["psnr_by_orbit"][ch]>=25)
            if good:
                categories[(src,fam,cat)][name+"_harm"]+=int(r["psnr_by_orbit"][ch]<25)
    output=[]
    count=Counter()
    for (src,fam,iid),group in sorted(by_group.items()):
        group=sorted(group,key=lambda x:int(x[0]["run_index"]))
        indices=[int(x[0]["run_index"]) for x in group]
        if indices!=list(range(len(group))):
            raise RuntimeError("STOP_NONCONTIGUOUS_RUNS "+src+"/"+fam+"/"+iid)
        rank=sorted(group,key=lambda x:(float(x[0]["rho_signed_raw"]),int(x[0]["run_index"])))
        def lbl(r):
            if r["raw_label"]=="GOOD":return "G"
            if r["raw_label"]=="ORBIT_ERROR":return "O" if r["s_gt"]=="111" else "P"
            return "C" if r["raw_label"]=="NONORBIT_CONSISTENT" else "I"
        labels="".join(lbl(x[0]) for x in group)
        cc=Counter(labels)
        count.update(x[0]["raw_label"] for x in group)
        output.append(dict(image_id=iid,cohort=cohort_ids.get(str(int(iid)),"OTHER_EXPOSED"),
             source=src,family=fam,ffhq_category=category[iid],n_runs=len(group),
             n_GOOD=cc["G"],n_ORBIT_global=cc["O"],n_ORBIT_partial=cc["P"],
             n_CONSISTENT=cc["C"],n_INCONSISTENT=cc["I"],
             run_labels=labels,e0_rank_labels="".join(lbl(x[0]) for x in rank),
             e0_rank_order=",".join(str(x[0]["run_index"]) for x in rank),
             orbit_best_member="".join(str(int(x[0]["s_gt"],2)) if lbl(x[0]) in "OP" else "." for x in group),
             clip_fix_ok="".join(("1" if x[1]=="111" else "0") if lbl(x[0])=="O" else
                                ("0" if lbl(x[0])=="P" else "-") for x in group),
             dec_clip_face="".join("1" if x[1]=="111" else "0" for x in group),
             dec_2f="".join("1" if x[2]=="111" else "0" for x in group),
             dec_s_all8="".join(str(int(x[3],2)) for x in group)))
    if count!=EXPECTED:raise RuntimeError("STOP_A8_SUMS")
    out.mkdir(parents=True,exist_ok=False)
    with (out/"R0N2_PER_IMAGE_COUNTS.csv").open("x",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(output)
    with (out/"B2_MEMBERSHIP_REPORT.csv").open("x",newline="",encoding="utf8") as f:
        keys=("source","cohort","family","category","n","good","CLIP_orb","CLIP_fix","CLIP_harm",
              "S_orb","S_fix","S_harm","SMID_orb","SMID_fix","SMID_harm")
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for (src,cohort,fam,cat),stats in sorted(categories.items()):
            w.writerow(dict(source=src,cohort=cohort,family=fam,category=cat,**{k:stats[k] for k in keys[4:]}))
    manifest={"status":"R0N2_B2_A8_CPU_PASS","source_r07_sha256":sha(evaluation),
              "source_n2_sha256":sha(N2/"N2_GT_FREE_CLIP_SCORES.json"),
              "categories_sha256":sha(META),"n_candidates":len(records),
              "n_rows":len(output),"label_totals":dict(count),"head":head,
              "A8_csv_sha256":sha(out/"R0N2_PER_IMAGE_COUNTS.csv"),
              "B2_csv_sha256":sha(out/"B2_MEMBERSHIP_REPORT.csv"),
              "training_checkpoint_membership":"UNKNOWN"}
    (out/"B2_A8_RECEIPT.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("R0N2_B2_A8_CPU_PASS",len(output),"ROWS",len(records),"CANDIDATES")
    print("A8_SHA256",manifest["A8_csv_sha256"])
    print("B2_SHA256",manifest["B2_csv_sha256"])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        print("ACTION NEEDED: R0N2 "+str(e),flush=True)
        raise SystemExit(8)
