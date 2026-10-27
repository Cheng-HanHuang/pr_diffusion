#!/usr/bin/env python3
"""A8b historical-only augmentation: 760 A8 rows + run-ordered signed rho.

Uses already-scored R0.7 rows and the *already-published* A8 table.
No candidate tensors, GT images, model or GPU are opened. Each
(source,family) tau is checked constant across image/rep. This script
only creates a new A8b bundle, never alters A8 or preexisting outputs.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
R07=BASE/"r0_7_cpu_20261023T080434Z/analysis"
A8=WT/"docs/track_r/executors/r0n/returns/R0N2_20261026T191819Z/R0N2_PER_IMAGE_COUNTS.csv"
A8_SHA="197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6"
R07_SHA="0c01c80acf54c48771904725bdcf0a7c5f0b966ac168b074c5cb8a5690c6ad59"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
EXPECTED=Counter({"GOOD":2040,"ORBIT_ERROR":85,"NONORBIT_CONSISTENT":324,"NONORBIT_INCONSISTENT":791})

def safe(path):
    p=Path(path)
    if p.is_symlink() or any(t in str(p.resolve(strict=False)).lower() for t in BAD):
        raise RuntimeError("STOP_A8B_FORBIDDEN_PATH")
    if not p.is_file():raise RuntimeError("STOP_A8B_FILE_MISSING")
    return p

def sha(path):
    h=hashlib.sha256()
    with safe(path).open("rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""):h.update(buf)
    return h.hexdigest()

def fmt6(v):
    v=float(v)
    if not math.isfinite(v) or v<0:
        raise RuntimeError("STOP_A8B_NONFINITE_RESIDUAL")
    return format(v,".6g")

def augment(a8,records):
    bykey=defaultdict(list)
    taus=defaultdict(set)
    labels=Counter()
    if len(records)!=3240 or len({r.get("uid") for r in records})!=3240:
        raise RuntimeError("STOP_A8B_HISTORICAL_COUNT")
    for r in records:
        for key in ("source","family","image_id","run_index","rho_signed_raw","tau_signed_good_q95","raw_label"):
            if key not in r:raise RuntimeError("STOP_A8B_R07_SCHEMA_"+key)
        if r["raw_label"] not in EXPECTED:
            raise RuntimeError("STOP_A8B_UNKNOWN_LABEL")
        cohort=(r["source"],r["family"])
        key=cohort+(str(r["image_id"]),)
        bykey[key].append(r)
        tau=float(r["tau_signed_good_q95"])
        if not math.isfinite(tau) or tau<=0:
            raise RuntimeError("STOP_A8B_INVALID_TAU")
        taus[cohort].add(tau)
        labels[r["raw_label"]]+=1
    if labels!=EXPECTED:
        raise RuntimeError("STOP_A8B_LABEL_TOTALS")
    if len(a8)!=760 or len(bykey)!=760:
        raise RuntimeError("STOP_A8B_760_ROWS")
    if any(len(v)!=1 for v in taus.values()):
        raise RuntimeError("STOP_A8B_TAU_SOURCE_FAMILY_DRIFT")
    source_fields=list(a8[0])
    if "rho_signed_by_run" in source_fields or "tau_signed_good_q95" in source_fields:
        raise RuntimeError("STOP_A8B_INPUT_ALREADY_AUGMENTED")
    out=[];seen=set()
    for existing in a8:
        key=(existing["source"],existing["family"],str(existing["image_id"]))
        if key in seen or key not in bykey:
            raise RuntimeError("STOP_A8B_A8_JOIN")
        seen.add(key)
        group=sorted(bykey[key],key=lambda r:int(r["run_index"]))
        n=int(existing["n_runs"])
        if len(group)!=n or [int(z["run_index"]) for z in group]!=list(range(n)):
            raise RuntimeError("STOP_A8B_RUN_INDEX")
        def label(r):
            if r["raw_label"]=="GOOD":return "G"
            if r["raw_label"]=="ORBIT_ERROR":return "O" if r["s_gt"]=="111" else "P"
            return "C" if r["raw_label"]=="NONORBIT_CONSISTENT" else "I"
        if existing["run_labels"]!="".join(label(r) for r in group):
            raise RuntimeError("STOP_A8B_LABEL_ORDER")
        rho=[fmt6(r["rho_signed_raw"]) for r in group]
        if len(rho)!=n:raise RuntimeError("STOP_A8B_RHO_LENGTH")
        out.append(dict(existing,
                        rho_signed_by_run=";".join(rho),
                        tau_signed_good_q95=fmt6(next(iter(taus[key[:2]])))))
    if set(bykey)!=seen or len(out)!=760:
        raise RuntimeError("STOP_A8B_A8_JOIN_COMPLETENESS")
    tbl=[{"source":s,"family":f,
          "tau_signed_good_q95":fmt6(next(iter(val))),
          "tau_exact_float64_repr":repr(next(iter(val))),
          "n_candidates":sum(len(bykey[k]) for k in bykey if k[:2]==(s,f))}
          for (s,f),val in sorted(taus.items())]
    if sum(x["n_candidates"] for x in tbl)!=3240:
        raise RuntimeError("STOP_A8B_TAU_GROUP_SUM")
    return out,tbl,source_fields+["rho_signed_by_run","tau_signed_good_q95"]

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A8B_NOT_CPU_ONLY")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    status=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if head!=a.expected_head or status:raise RuntimeError("STOP_A8B_GIT_HEAD")
    source=R07/"R0_7_SCORED_GT_ROWS.jsonl"
    receipt=json.loads(safe(R07/"R0_7_COMPLETE_RECEIPT.json").read_text())
    if (sha(source)!=R07_SHA or
        receipt.get("evaluated_rows_sha256")!=R07_SHA or
        receipt.get("n_scored")!=3240 or
        receipt.get("no_protected_access") is not True or
        sha(A8)!=A8_SHA):
        raise RuntimeError("STOP_A8B_FROZEN_INPUT")
    with safe(A8).open(newline="") as f:
        original=list(csv.DictReader(f))
    with safe(source).open() as f:
        records=[json.loads(s) for s in f if s.strip()]
    out,taus,cols=augment(original,records)
    dest=Path(a.out)
    if (dest.parent!=BASE or not dest.name.startswith("r0n2_a8b_")
        or dest.exists() or dest.is_symlink()):
        raise RuntimeError("STOP_A8B_OUTPUT_NAMESPACE")
    dest.mkdir(exist_ok=False)
    def csvwrite(path,rows,fields):
        with path.open("x",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    csvwrite(dest/"R0N2_A8B_PER_IMAGE.csv",out,cols)
    csvwrite(dest/"R0N2_A8B_TAU_BY_SOURCE_FAMILY.csv",taus,list(taus[0]))
    doc={
      "status":"R0N2_A8B_PASS",
      "head":head,"n_rows":760,"n_candidates":3240,
      "tau_groups":len(taus),
      "A8_original_sha256":A8_SHA,"R07_original_sha256":R07_SHA,
      "A8B_csv_sha256":sha(dest/"R0N2_A8B_PER_IMAGE.csv"),
      "tau_csv_sha256":sha(dest/"R0N2_A8B_TAU_BY_SOURCE_FAMILY.csv"),
      "rho_encoding":"six significant digits, semicolon separated, frozen run-index order",
      "tau_encoding":"frozen float64 plus six-significant-digit display; source+family",
      "historical_only":True,"gt_images_read":False,"gpu_used":False,
      "new_candidate_outcomes_inspected":False,
    }
    with (dest/"R0N2_A8B_RECEIPT.json").open("x") as f:
        json.dump(doc,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print("R0N2_A8B_PASS 760 rows 3240 historical candidates tau_groups="+str(len(taus)))
    print("A8B_SHA="+doc["A8B_csv_sha256"])
    print("A8B_TAU_SHA="+doc["tau_csv_sha256"])
    print("A8B_DIR="+str(dest))
    print("OK")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as exc:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: A8B "+str(exc),flush=True)
        sys.exit(8)
