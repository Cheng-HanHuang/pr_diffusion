#!/usr/bin/env python3
"""Track R A7 exact all-subset binary-Good U-statistics and forecast audit.

Production panel analysis is embargo-gated. Before the freeze, *only* the
cleared historical A8b K<=4 preview may be loaded and analyzed.
E1_CLIP is intentionally unsupported until exact subset-dependent CLIP
outcomes are available (cannot replace with per-run independent decisions).
"""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,os,random,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r_a7_cpu"
BASE=ROOT/"outputs/pr_diffusion/track_r"
A8B=WT/"docs/track_r/executors/r0n/returns/R0N2_A8B_20261027T041156Z/R0N2_A8B_PER_IMAGE.csv"
A8B_SHA="e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901"
POLS=("E0","O_raw","O_orbit")
PANEL={"ffhq":{"FAST":(64,16),"SITCOM":(16,32),"DAPS":(8,)},
       "imagenet":{"FAST":(16,64)}}
GOOD={"ffhq":25,"imagenet":20}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def rank_prob(n,k,rank):
    if not (1<=k<=n and 1<=rank<=n):
        raise ValueError("bad n/k/rank")
    return (math.comb(n-rank,k-1)/math.comb(n,k)
            if n-rank>=k-1 else 0.)

def exact_panel(runs,k):
    """Exact K-subset probability of Good for each policy, no enumeration."""
    n=len(runs)
    if k not in range(1,n+1) or n<1 or len({r["run"] for r in runs})!=n:
        raise ValueError("invalid K and unique run indices")
    for r in runs:
        if (not math.isfinite(float(r["rho"])) or
            not isinstance(r["good_raw"],bool) or
            not isinstance(r["good_orbit"],bool) or
            (r["good_raw"] and not r["good_orbit"])):
            raise ValueError("invalid candidate")
    rank=sorted(runs,key=lambda r:(r["rho"],r["run"]))
    e0=sum(float(r["good_raw"])*rank_prob(n,k,i+1)
           for i,r in enumerate(rank))
    ng=sum(r["good_raw"] for r in runs)
    no=sum(r["good_orbit"] for r in runs)
    def best(n_success):
        return 1-(math.comb(n-n_success,k)/math.comb(n,k)
                  if n-n_success>=k else 0.)
    return dict(E0=e0,O_raw=best(ng),O_orbit=best(no))

def brute(runs,k):
    combos=list(itertools.combinations(runs,k))
    values={p:0. for p in POLS}
    for subset in combos:
        winner=min(subset,key=lambda r:(r["rho"],r["run"]))
        values["E0"]+=float(winner["good_raw"])
        values["O_raw"]+=float(any(r["good_raw"] for r in subset))
        values["O_orbit"]+=float(any(r["good_orbit"] for r in subset))
    return {p:v/len(combos) for p,v in values.items()}

def load_a8b_k4():
    if not A8B.is_file() or A8B.is_symlink() or sha(A8B)!=A8B_SHA:
        raise RuntimeError("STOP_POSTFREEZE_A8B_SOURCE")
    rows=[]
    with A8B.open(newline="",encoding="utf-8") as f:
        for line in csv.DictReader(f):
            letters=line["run_labels"]
            rho=[float(x) for x in line["rho_signed_by_run"].split(";")]
            if (len(letters)!=int(line["n_runs"]) or len(rho)!=len(letters)
                or not all(c in "GOPCI" for c in letters)
                or not all(math.isfinite(x) for x in rho)):
                raise RuntimeError("STOP_POSTFREEZE_A8B_ROW")
            runs=[dict(run=j,rho=v,good_raw=letters[j]=="G",
                       good_orbit=letters[j] in "GOP")
                  for j,v in enumerate(rho[:4])]
            for k in range(1,len(runs)+1):
                rows.append(dict(dataset="ffhq",cohort=line["source"],
                   family=line["family"],image_id=line["image_id"],
                   panel_n=len(runs),k=k,
                   obs=exact_panel(runs,k)))
    if len(rows)<1000:
        raise RuntimeError("STOP_POSTFREEZE_HISTORICAL_ROWS")
    return rows

def summarize(rows):
    g=defaultdict(list)
    for r in rows:
        k=(r["dataset"],r["cohort"],r["family"],r["panel_n"],r["k"])
        g[k].append(r)
    out=[]
    for (dataset,cohort,family,n,k),group in sorted(g.items()):
        if len({r["image_id"] for r in group})!=len(group):
            raise ValueError("duplicate images")
        mean={p:sum(r["obs"][p] for r in group)/len(group) for p in POLS}
        if not (0<=mean["E0"]<=mean["O_raw"]+1e-12 and
                mean["O_raw"]<=mean["O_orbit"]+1e-12):
            raise ValueError("oracle dominance")
        out.append(dict(dataset=dataset,cohort=cohort,family=family,
            panel_n=n,k=k,n_images=len(group),observed=mean))
    return out

def compare_interval(observed,forecast):
    """Planner A6.5: predictive [lo95,hi95] vs parameter-only SE_comb."""
    point,lo,hi=[float(forecast[x]) for x in ("point","lo95","hi95")]
    typ=forecast["interval_type"]
    if not (all(math.isfinite(z) for z in (observed,point,lo,hi))
            and lo<=point<=hi and hi>lo):
        raise ValueError("bad interval")
    se=(hi-lo)/3.92
    if typ=="predictive_panel":
        ok=lo<=observed<=hi;z=(observed-point)/se
    elif typ=="parameter_only":
        if "se_observed" not in forecast:raise ValueError("missing SE_obs")
        so=float(forecast["se_observed"])
        if not math.isfinite(so) or so<0:raise ValueError("bad SE_obs")
        se=math.hypot(se,so)
        ok=abs(observed-point)<=2*se;z=(observed-point)/se
    else:
        raise ValueError("unknown predictive interval type")
    return {"inside":ok,"standardized_residual":z,
            "point":point,"observed":observed,"lo95":lo,"hi95":hi,
            "interval_type":typ}

def verify_freeze(repo,registry,freeze_key,theory_commit,forecast_path,
                  expected_sha,first_outcome_commit):
    """Call *before opening any prospective outcome file*."""
    if freeze_key not in ("v1","v2 (optional)"):
        raise RuntimeError("STOP_POSTFREEZE_WRONG_SCOPE")
    lines=Path(registry).read_text().splitlines()
    matches=[]
    for line in lines:
        if not line.startswith("|"):continue
        cols=[x.strip() for x in line.strip("|").split("|")]
        if cols and cols[0]==freeze_key:matches.append(cols)
    if len(matches)!=1 or len(matches[0])<7:
        raise RuntimeError("STOP_POSTFREEZE_NO_REGISTRY")
    entry=matches[0]
    if (entry[2]!=theory_commit or
        entry[3].strip(chr(96))!=forecast_path or
        expected_sha not in entry[4] or
        any(not x or x=="—" for x in (theory_commit,expected_sha))):
        raise RuntimeError("STOP_POSTFREEZE_BAD_REGISTRY_RECORD")
    def git(*args):
        return subprocess.check_output(["git","-C",str(repo),*args],
                                       timeout=60)
    data=git("show",theory_commit+":"+forecast_path)
    if hashlib.sha256(data).hexdigest()!=expected_sha:
        raise RuntimeError("STOP_POSTFREEZE_PREDICTION_SHA")
    t=int(git("show","-s","--format=%ct",theory_commit).decode())
    v=int(git("show","-s","--format=%ct",first_outcome_commit).decode())
    if t>=v:
        raise RuntimeError("STOP_POSTFREEZE_FORECAST_AFTER_OUTCOME")
    return dict(status="FREEZE_AUTHORIZED",freeze=freeze_key,
                forecast_sha256=expected_sha,freeze_time=t,outcome_time=v)

def validate_kernel():
    rng=random.Random(20261028)
    for _ in range(600):
        n=rng.randint(1,4)
        records=[dict(run=i,rho=rng.choice((.1,.2,.3)),
                    good_raw=bool(rng.randrange(2)),
                    good_orbit=True)
                 for i in range(n)]
        for k in range(1,n+1):
            a=exact_panel(records,k);b=brute(records,k)
            if any(abs(a[p]-b[p])>1e-12 for p in POLS):
                raise RuntimeError("STOP_POSTFREEZE_RANK_IDENTITY")
    return True

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_POSTFREEZE_CPU_ONLY")
    if not a.k4_preview:
        raise RuntimeError("STOP_POSTFREEZE_PROSPECTIVE_NOT_YET_AUTHORIZED")
    validate_kernel()
    raw=load_a8b_k4()
    grid=summarize(raw)
    summary=dict(status="XP_POSTFREEZE_K4_KERNEL_PASS",
        source_sha256=A8B_SHA,n_historical_image_k=len(raw),
        n_grid_points=len(grid),scope="HISTORICAL_K_LE_4_ONLY",
        exact_all_subset_tested=True,
        prospective_outcomes_accessed=False,
        policies=list(POLS),grid=grid,
        future_scope="Prospective K>4 gated on verified prediction registry",
        caveats=["subset-dependent E1_CLIP requires a separate exact policy implementation",
                 "full image-cluster CI / per-image forecast adapter pending"])
    if a.out:
        p=Path(a.out)
        if p.exists() or p.parent!=BASE or not p.name.startswith("xp_a7_k4_preview_"):
            raise RuntimeError("STOP_POSTFREEZE_OUTPUT_LOCATION")
        with p.open("x") as f:
            json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    print("XP_A7_POSTFREEZE_K4_KERNEL_PASS",
          len(raw),"image-K",len(grid),"grid points",flush=True)
    print("PROSPECTIVE_OUTCOMES_ACCESSED=false",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--k4-preview",action="store_true")
    p.add_argument("--out")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: POSTFREEZE_K4 "+str(e),flush=True)
        sys.exit(8)
