#!/usr/bin/env python3
"""Track R A8 K<=4 schema/label/rank audit and regularized race calibration.

Only reads a user-specified authorized R0N2_PER_IMAGE_COUNTS.csv.
Does NOT open XP results, actual image files, later run indices or PAC paths.
Example:
  python scripts/track_r/theory/audit_a8_rank_race.py \
    --a8 /path/to/authorized/R0N2_PER_IMAGE_COUNTS.csv \
    --output /tmp/A8_ALLOWED_RANK_AUDIT.json

Input SHA-256 and all 1,960 run labels are checked before output.
The prior on log Bradley-Terry class weights is N(0, 2.5^2).
The composite pairwise likelihood is not an independent-pairs likelihood.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

A8_SHA256 = "197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6"
FAMILIES=("DAPS","SITCOM","FAST")
LABELS="GOPCI"
NEXPECTED={"DAPS":(520,335,23,1,18,143),
           "SITCOM":(720,414,25,0,174,107),
           "FAST":(720,422,10,0,23,265)}
COHORTS={"B22_random":"B22","DEV80_enriched":"DEV80"}
ALLOWED={("DAPS","S1_b22"),("DAPS","S2_b24_dev80"),
         ("SITCOM","S1_b22"),("SITCOM","S2_b24_dev80"),
         ("FAST","S5_r0_fast")}

def read_a8(path,verify_sha=True):
    path=Path(path)
    raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if verify_sha and digest!=A8_SHA256:
        raise ValueError("A8 digest does not match authorized v1 calibration receipt")
    with path.open(newline="",encoding="utf-8") as f:
        rawrows=list(csv.DictReader(f))
    if verify_sha and len(rawrows)!=760:
        raise ValueError("expected 760 A8 rows; got %d"%len(rawrows))
    selected=[]
    for row in rawrows:
        f=row["family"]
        source=row["source"]
        if (f,source) not in ALLOWED:continue
        cohort=COHORTS.get(row["cohort"])
        if cohort is None:raise ValueError("unknown cohort")
        if source=="S1_b22" and cohort!="B22":raise ValueError("wrong B22 cohort")
        if source=="S2_b24_dev80" and cohort!="DEV80":raise ValueError("wrong DEV80 cohort")
        n=int(row["n_runs"])
        if n!=(2 if (f,cohort)==("DAPS","B22") else 4):
            raise ValueError("bad number of allowed calibration runs")
        original=row["run_labels"]
        ranked=row["e0_rank_labels"]
        order=list(map(int,row["e0_rank_order"].split(",")))
        if (len(original)!=n or len(ranked)!=n or
            sorted(order)!=list(range(n)) or
            "".join(original[i] for i in order)!=ranked or
            any(x not in LABELS for x in original)):
            raise ValueError("A8 rank/count inconsistency for "+row["image_id"])
        counts=tuple(int(row[col])for col in
             ["n_GOOD","n_ORBIT_global","n_ORBIT_partial",
              "n_CONSISTENT","n_INCONSISTENT"])
        if counts!=tuple(original.count(x) for x in LABELS):
            raise ValueError("label-count mismatch for "+row["image_id"])
        selected.append({"family":f,"cohort":cohort,
                         "image_id":row["image_id"],"source":source,
                         "ranked":ranked,"original":original,
                         "counts":counts,"n":n})
    expected={(f,c) for f in FAMILIES for c in ("B22","DEV80")}
    assert {(r["family"],r["cohort"])for r in selected}==expected
    for f in FAMILIES:
        fr=[r for r in selected if r["family"]==f]
        assert len(fr)==180
        assert sum(r["cohort"]=="B22" for r in fr)==100
        assert sum(r["cohort"]=="DEV80" for r in fr)==80
        counts=Counter("".join(r["original"] for r in fr))
        assert ((sum(counts.values()),)+tuple(counts.get(c,0) for c in LABELS))==NEXPECTED[f]
    return selected,digest

def crosscheck(rows):
    pairs=defaultdict(Counter)
    mixed=Counter()
    for row in rows:
        family=row["family"]
        labels=row["ranked"]
        for i in range(len(labels)):
            for j in range(i+1,len(labels)):
                w,l=labels[i],labels[j]
                if w==l:continue
                pairs[(family, "".join(sorted((w,l))))][w]+=1
        passers=[x for x in labels if x!="I"]
        if "G" in passers and any(x!="G" for x in passers):
            mixed["N"]+=1
            mixed["wins_G"]+=(labels[0]=="G")
            mixed["expected_S2"]+=passers.count("G")/len(passers)
    assert mixed["N"]==100
    assert mixed["wins_G"]==92
    assert abs(mixed["expected_S2"]-51.916666666666664)<1e-8
    assert sum(pairs[(f,"CG")].get("G",0) for f in FAMILIES)==200
    assert sum(pairs[(f,"CG")].get("C",0) for f in FAMILIES)==5
    assert sum(pairs[(f,"GO")].get("G",0) for f in FAMILIES)==50
    assert sum(pairs[(f,"GO")].get("O",0) for f in FAMILIES)==19
    assert sum(pairs[(f,"GI")].get("G",0) for f in FAMILIES)==570
    assert sum(pairs[(f,"GI")].get("I",0) for f in FAMILIES)==1
    return pairs,mixed

def fit_weights(rows, tau=2.5):
    # One composite likelihood observation per *within-image* unordered pair
    # of distinct passing labels. Partial O shares O's rate by convention.
    ps=[]
    classes={"G":np.array((0.,0.)),
             "O":np.array((1.,0.)),
             "C":np.array((0.,1.))}
    for row in rows:
        labels=[("O" if x=="P" else x)for x in row["ranked"] if x!="I"]
        for i in range(len(labels)):
            for j in range(i+1,len(labels)):
                if labels[i]!=labels[j]:
                    ps.append(classes[labels[i]]-classes[labels[j]])
    X=np.asarray(ps).reshape(-1,2)
    if not len(X):return {"w_O":1.,"w_C":1.,"log_O":0.,"log_C":0.,
                          "number_pairs":0}
    def obj(b):
        z=X@b
        loglike=np.logaddexp(0,-z).sum()
        penalty=(b@b)/(2*tau**2)
        grad=X.T@(expit(z)-1)+b/(tau**2)
        return float(loglike+penalty),grad
    result=minimize(lambda b:obj(b),np.array([-1.,-2.]),jac=True,
                    method="BFGS",options={"gtol":1e-9,"maxiter":400})
    if not np.isfinite(result.fun):raise RuntimeError("pair fit not finite")
    b=result.x
    return {"w_O":float(np.exp(b[0])),"w_C":float(np.exp(b[1])),
            "log_O":float(b[0]),"log_C":float(b[1]),
            "number_pairs":len(X),"objective":float(result.fun),
            "converged":bool(result.success)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a8",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    rows,sha=read_a8(args.a8)
    pairs,mixed=crosscheck(rows)
    fits={f:fit_weights([r for r in rows if r["family"]==f])
          for f in FAMILIES}
    obj={"status":"CALIBRATION_ONLY_NOT_FROZEN",
         "data_sha256":sha,"families":fits,
         "mixed_passing":{"N":mixed["N"],"winner_G":mixed["wins_G"],
                          "S2_expected_G":mixed["expected_S2"]},
         "pair_rank_counts":{f+"|"+pair:dict(c)
              for (f,pair),c in sorted(pairs.items())},
         "n_images":len(rows),
         "counts_total":{"G":1171,"O":58,"P":1,"C":215,"I":515}}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print("TH1_A8_KLE4_RANK_AUDIT_PASS",sha)
    print("family class race weights",fits)

if __name__=="__main__":main()
