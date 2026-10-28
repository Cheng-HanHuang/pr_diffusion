#!/usr/bin/env python3
"""A9.2(c) exploratory eleven-case census from frozen 64-run FAST strings.

No additional model/scoring I/O; only the already verified A9 CPU output.
Read-only source, new case-list file in the same PAC run namespace.
Nothing is written to theory_inputs.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,traceback
from pathlib import Path
from collections import Counter

from scripts.track_r.xp_a8_fast_v1_cpu import BASE,head_guard,sha

A9_RUN=BASE/"xp_a9_exchange_20261028T100727Z"
INPUT=A9_RUN/"A9_FAST_64_PER_IMAGE_STRINGS.csv"
INPUT_SHA="d616dd50f10352e98b891a99b9a153d5d72a93c61e44381827147dcca23f5466"
REPORT=A9_RUN/"A9_FAST_EXCHANGEABILITY.json"
REPORT_SHA="13ca52938820981390969f8cba87d35728aa3f10e12dcf612c54259b261569b8"
SIGMA=0.05
CLASSES=("G","O_g","O_p","C","I")

def check(x,why):
    if not x:raise RuntimeError("STOP_A9_CASES_"+why)

def case_row(r):
    lab=r["labels_by_run"].split(";")
    rho=[float(z) for z in r["rho_signed_by_run"].split(";")]
    ranks=[int(z) for z in r["e0_rank_order"].split(";")]
    check(len(lab)==len(rho)==len(ranks)==64 and
          set(ranks)==set(range(64)) and set(lab)<=set(CLASSES) and
          all(math.isfinite(v) and v>=0 for v in rho),"RUN_SCHEMA")
    check(all((rho[ranks[j]],ranks[j]) <= (rho[ranks[j+1]],ranks[j+1])
              for j in range(63)),"RANK_ORDER")
    counts=Counter(lab)
    check(all(int(r["n_"+c])==counts[c] for c in CLASSES),"CLASS_COUNT")
    win=ranks[0]
    good=[j for j in range(64) if lab[j]=="G"]
    never=not good
    selected_bad=bool(good) and lab[win]!="G"
    best=min(good,key=lambda j:(rho[j],j)) if good else None
    gap=(rho[win]-rho[best])/SIGMA if good else None
    return dict(cohort=r["cohort"],image_id=str(r["image_id"]),
                category="NEVER_GOOD" if never else
                    "K64_SELECTION_LOSS" if selected_bad else "OTHER",
                prefix_4=";".join(lab[:4]),
                class_counts={c:counts[c] for c in CLASSES},
                rank_winner_run=win,rank_winner_class=lab[win],
                winner_rho_signed=rho[win],
                best_good_run=best,
                winner_minus_best_good_rho_over_sigma=gap,
                n_runs=64)

def census(rows):
    check(len(rows)==180,"N_IMAGES")
    ids=set();result=[]
    for r in rows:
        k=(r["cohort"],str(r["image_id"]))
        check(k not in ids,"IMAGE_DUP")
        ids.add(k)
        info=case_row(r)
        if info["category"]!="OTHER":result.append(info)
    never=[r for r in result if r["category"]=="NEVER_GOOD"]
    losses=[r for r in result if r["category"]=="K64_SELECTION_LOSS"]
    check(len(never)==7 and len(losses)==4,"FROZEN_EXPECTED_SEVEN_AND_FOUR")
    check(sum(r["cohort"]=="B22" for r in never)==1 and
          sum(r["cohort"]=="DEV80" for r in never)==6 and
          all(r["cohort"]=="DEV80" for r in losses),
          "FROZEN_COHORT_SPLIT")
    return sorted(result,key=lambda r:(r["category"],r["cohort"],r["image_id"]))

def main(a):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    head_guard(a.expected_head)
    check(sha(INPUT)==INPUT_SHA and sha(REPORT)==REPORT_SHA,"INPUT_SHA")
    report=json.loads(REPORT.read_text())
    check(report.get("status")=="A9_FAST_EXCHANGEABILITY_POSTHOC_PASS"
          and report["D1_P2_outcomes_opened"] is False
          and report["B16_F8_outcomes_opened"] is False,"FROZEN_SOURCE")
    with INPUT.open(newline="") as f:rows=list(csv.DictReader(f))
    cases=census(rows)
    out=Path(a.out)
    check(out==A9_RUN/"A9_FAST_11_CASES.json" and not out.exists(),"DEST")
    payload=dict(status="A9_FAST_11_CASES_POSTHOC_PASS",
                 source_sha256=INPUT_SHA,
                 n_never_good=7,n_selection_losses=4,
                 n_images=180,n_total_candidates=11520,
                 sigma=SIGMA,cases=cases,
                 notes=["Winner is the minimum signed-rho candidate at K=64.",
                        "Gap=(winner signed rho - best raw-GOOD signed rho)/sigma.",
                        "Missing gap for never-GOOD images is represented as null.",
                        "Post-hoc case census only; no model refit.",
                        "Per-image results stay in xp/returns under A9.3."],
                 D1_P2_outcomes_opened=False,B16_F8_outcomes_opened=False)
    with out.open("x") as f:
        json.dump(payload,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("A9_FAST_11_CASES_PASS","never_good=7","selection_losses=4",flush=True)
    print("A9_FAST_CASES_OUT="+str(out),flush=True)
    print("A9_FAST_CASES_SHA="+sha(out),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    try:main(ap.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A9_FAST_CASES",str(exc),flush=True)
        sys.exit(8)
