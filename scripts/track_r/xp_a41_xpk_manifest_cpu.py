#!/usr/bin/env python3
"""A24 XP-K full 100 B22-image, 4-new-run per-family manifest; CPU-only.

R0.7's frozen canonical catalog is authoritative for historical y/GT/labels.
Only signed measurement file paths + SHA and seeds go to the GPU manifest.
GT and historical candidate references remain strictly CPU-only in a
separate file never passed to native workers. Failure is fail-closed.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,traceback
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
CAT=BASE/"r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json"
SCORES=BASE/"r0_7_cpu_20261023T080434Z/analysis/R0_7_SCORED_GT_ROWS.jsonl"
R07=BASE/"r0_7_cpu_20261023T080434Z/analysis/R0_7_COMPLETE_RECEIPT.json"
F8=BASE/"xp_f8_legacy_preflight_dev80seedfix_20261029T064359Z/XP_F8_REPLAY_INPUTS_GT_FREE.json"
F8_SHA="0929a94537c30faa893346a000c6cca5b51d5f45b48fbc2dfb0f6e64fa2537b9"
FAMILIES=("SITCOM","FAST","DAPS")
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
METHOD_SOURCE={"SITCOM":"S1_b22","FAST":"S5_r0_fast","DAPS":"S1_b22"}
N=100
K=4
# Historical B22 source cardinality is not the prospective XP-K cardinality.
# SITCOM and FAST have four historical B22 trajectories/image; the accepted
# Fresh2 DAPS lineage has two. XP-K still generates K=4 NEW roots/family/image.
HIST_K={"SITCOM":4,"FAST":4,"DAPS":2}

def gate(ok,reason):
    if not ok:raise RuntimeError("STOP_XPK_MANIFEST_"+reason)
def sha(p):
    p=Path(p)
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def safe_source(p,expected=None):
    p=Path(p)
    real=p.resolve()
    gate(p.is_absolute() and p.is_file() and not p.is_symlink() and
         ROOT.resolve() in real.parents and
         not any(x in str(p).lower() or x in str(real).lower() for x in BAD),
         "UNSAFE_OR_PROTECTED_SOURCE_"+str(p))
    if expected:gate(sha(p)==expected,"SOURCE_HASH_"+str(p))
    return p
def obj(p):
    return json.loads(safe_source(p).read_text())
def git(*args):
    import subprocess
    return subprocess.check_output(["git","-C",str(WT),*args],
                                   text=True,timeout=60).strip()
def seed(family,image_id,run):
    token=f"XP|{family}|ffhq_cal|{image_id}|{run}"
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],"big")%(2**31)
def key(d):return (d["family"],str(d["image_id"]),int(d["run_index"]))
def hist_family(r,method):
    if r["family"]!=method or r["source"]!=METHOD_SOURCE[method]:return False
    if method=="FAST" and r["measurement_source"]!="S1_b22":return False
    if method!="FAST" and r["measurement_source"]!="S1_b22":return False
    return True
def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","NO_CUDA")
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),"PINNED_HEAD")
    out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a41_xpk_") and
         not out.exists() and not out.is_symlink(),"FRESH_RUN_ROOT")
    cat=obj(CAT)
    gate(cat["status"]=="R0_7_GT_OFFLINE_CATALOG_PASS" and
         cat["n_candidates"]==3240,"R07_CANONICAL_CATALOG")
    rows=cat["rows"]
    scored=[json.loads(x) for x in safe_source(SCORES).read_text().splitlines() if x.strip()]
    gate(len(rows)==len(scored)==3240,"FULL_R07_3240")
    scores={r["uid"]:r for r in scored}
    gate(len(scores)==3240 and set(scores)=={r["uid"] for r in rows},
         "R07_SCORE_UIDS")
    gate(sha(F8)==F8_SHA,"F8_ORIGINAL_B22_MANIFEST")
    f8=obj(F8)
    gate(f8["n_replay"]==8 and len(f8["rows"])==8,"F8_8ROOT_FROZEN")
    b22_f8=[r for r in f8["rows"] if r["source"]=="S1_b22"]
    gate(len(b22_f8)==4,"F8_B22_4")
    original={};calibration={};references={}
    all_images=set()
    for family in FAMILIES:
        hist_k=HIST_K[family]
        panel=[r for r in rows if hist_family(r,family)]
        group=defaultdict(dict)
        for r in panel:
            k=(str(r["image_id"]),int(r["run_index"]))
            gate(k[1] in range(4) and k[1] not in group[k[0]],
                 "HIST_RUN_UNIQUE_"+family)
            group[k[0]][k[1]]=r
        gate(len(group)==100 and
             all(set(g)==set(range(hist_k)) for g in group.values()) and
             len(panel)==100*hist_k,
             f"HIST_B22_100x{hist_k}_"+family)
        ids=sorted(group)
        all_images.add(tuple(ids))
        original[family]=[]
        calibration[family]=[]
        references[family]={}
        for image_id in ids:
            refs=[group[image_id][i] for i in range(hist_k)]
            y_paths={r["measurement_path"] for r in refs}
            y_shas={r["measurement_sha256"] for r in refs}
            gate(len(y_paths)==len(y_shas)==1,
                 "HIST_Y_NOT_FIXED_"+family+"_"+image_id)
            mp=safe_source(next(iter(y_paths)),next(iter(y_shas)))
            refscores=[scores[r["uid"]] for r in refs]
            gate(all(x["raw_label"] in ("GOOD","ORBIT_ERROR",
                        "NONORBIT_CONSISTENT","NONORBIT_INCONSISTENT")
                     and x["source"]==refs[i]["source"]
                     and x["family"]==family and
                     x["image_id"]==image_id and
                     int(x["run_index"])==i for i,x in enumerate(refscores)),
                 "HIST_SCORES_JOIN_"+family+"_"+image_id)
            references[family][image_id]=[
                 dict(uid=r["uid"],family=family,image_id=image_id,
                      run=i,source=r["source"],
                      gt_path=r["gt_path"],gt_format=r["gt_format"],
                      historical_candidate_path=r["candidate_path"],
                      historical_candidate_sha256=r["candidate_sha256"],
                      historical_raw_psnr=float(refscores[i]["psnr_raw"]),
                      historical_raw_label=refscores[i]["raw_label"],
                      historical_tau_signed_good_q95=float(refscores[i]["tau_signed_good_q95"]),
                      historical_rho_signed=float(refscores[i]["rho_signed_raw"]),
                      measurement_sha256=r["measurement_sha256"])
                 for i,r in enumerate(refs)]
            for run in range(K):
                rec=dict(uid=f"XPK|{family}|ffhq_cal|{image_id}|{run}",
                   family=family,dataset="ffhq_cal",image_id=image_id,
                   run=run,seed=seed(family,image_id,run),
                   measurement_path=str(mp),measurement_sha256=next(iter(y_shas)),
                   stage="calibrated",GT_supplied=False,
                   historical_candidate_supplied=False)
                calibration[family].append(rec)
            if image_id==ids[0]:
                for run in (0,1):
                    r=refs[run]
                    if family=="SITCOM":
                        ss=43
                    elif family=="DAPS":
                        fr=[f for f in b22_f8 if str(f["image_id"])==image_id and
                             int(f["rep"])==run]
                        gate(len(fr)==1 and
                             fr[0]["measurement_sha256"]==next(iter(y_shas)),
                             "DAPS_F8_B22_SAME_EXACT_IMAGE")
                        ss=int(fr[0]["native_seed"])
                    else:
                        rr=r["candidate_path"]
                        # Native FAST B22 uses frozen per-root source seed.
                        # Only original 2-root replay is ever permitted to
                        # carry a historical run seed; inference from historical
                        # metadata is forbidden if missing.
                        from scripts.track_r.xp_fast_replay_cpu import ROOTS,ROOTS_SHA
                        gate(sha(ROOTS)==ROOTS_SHA,"FAST_ORIGINAL_ROOTS_SHA")
                        canon=obj(ROOTS)["rows"]
                        fr=[x for x in canon if x["measurement_source"]=="S1_b22"
                            and x["image_id"]==image_id and x["rep"]==run]
                        gate(len(fr)==1 and
                             fr[0]["measurement_sha256"]==next(iter(y_shas)),
                             "FAST_CANONICAL_B22_SEED")
                        ss=int(fr[0]["seed"])
                    original[family].append(dict(
                        uid=f"XPK|{family}|REPLAY|{image_id}|{run}",
                        family=family,dataset="ffhq",image_id=image_id,
                        run=run,seed=ss,
                        measurement_path=str(mp),
                        measurement_sha256=next(iter(y_shas)),
                        stage="historical_replay",GT_supplied=False,
                        historical_candidate_supplied=False))
        gate(len(calibration[family])==400 and len(original[family])==2,
             "FROZEN_COUNTS_"+family)
        all_seeds=[r["seed"] for r in calibration[family]]
        gate(len(set(all_seeds))==400,
             "NEW_SEED_COLLISION_"+family)
        old_seeds={r["seed"] for r in original[family]}
        gate(not set(all_seeds)&old_seeds,
             "NEW_VS_HIST_SEED_COLLISION_"+family)
    gate(len(all_images)==1,"B22_SAME_100_FFHQ_IDS")
    # All three families must use the identical *file* and signed
    # measurement SHA for the same FFHQ image.
    for id in next(iter(all_images)):
        specs={(next(r["measurement_path"] for r in calibration[f] if r["image_id"]==id),
                next(r["measurement_sha256"] for r in calibration[f] if r["image_id"]==id))
               for f in FAMILIES}
        gate(len(specs)==1,"CROSS_SOLVER_Y_DRIFT_"+id)
    manifest=dict(status="XP_K_A24_FROZEN_GT_FREE_100x4",
        schema="prdiff.track_r.xpk.a24.v1",head=a.expected_head,
        source_catalog_sha256=sha(CAT),source_score_sha256=sha(SCORES),
        prior_f8_manifest_sha256=F8_SHA,
        source_measurement_signed=True,source_dataset="ffhq_cal",
        n_images=100,n_new_runs_per_image=4,
        n_new_per_family=400,families=list(FAMILIES),
        generator_seeds_rule="sha256 first8 bigendian mod 2**31, XP|{family}|ffhq_cal|{image_id}|{run}",
        historical_replay_rows=original,
        calibrated_rows=calibration,
        original_GT_access=False,
        reference_paths_in_GPU_manifest=False)
    refs=dict(status="XP_K_CPU_ONLY_HISTORICAL_SCORES_AND_GT_REFERENCES",
      source_catalog_sha256=sha(CAT),source_score_sha256=sha(SCORES),
      family_references=references,
      GT_used_for_scoring_only=True)
    out.mkdir(mode=0o700)
    def write(path,value):
        with path.open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
    write(out/"XPK_GPU_GT_FREE_MANIFEST.json",manifest)
    write(out/"XPK_CPU_ONLY_HIST_REFERENCES.json",refs)
    rec=dict(status="XP_K_A24_100_B22_IMAGES_4_NEW_SEEDS_FROZEN",
       code_head=a.expected_head,
       sha_manifest=sha(out/"XPK_GPU_GT_FREE_MANIFEST.json"),
       sha_cpu_references=sha(out/"XPK_CPU_ONLY_HIST_REFERENCES.json"),
       SHA_R07_catalog=sha(CAT),SHA_R07_score=sha(SCORES),
       n_images=100,n_per_family=400,n_replay_per_family=2,
       historical_roots_per_image=HIST_K,
       family_names=list(FAMILIES),GT_never_given_to_GPU=True,
       no_GPU_started=True)
    write(out/"XPK_SOURCE_FREEZE_RECEIPT.json",rec)
    print("XPK_MANIFEST_CPU_SOURCE_LOCK_PASS",out,
          "new",1200,"replay",6,flush=True)
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:sys.exit(main(p.parse_args()))
    except Exception as e:
        traceback.print_exc()
        print("ACTION_NEEDED_XPK_MANIFEST",repr(e),flush=True)
        sys.exit(8)
