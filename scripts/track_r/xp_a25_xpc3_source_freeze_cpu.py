#!/usr/bin/env python3
"""A20 XP-C3 CPU SOURCE-FREEZE ONLY. No candidate pixels/GT/orientation calls.

Generates exact source rows for 3 complete previously scored ImageNet panels.
The *generated manifest must be committed and pushed BEFORE GPU candidate
scoring*. A worker must refuse a not-git-tracked manifest; no auto GPU here.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from collections import Counter
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
OUT=WT/"docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
MODEL=ROOT/"models/imagenet256.pt"
MODEL_SHA="a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"
SIZES={"XP_N":6400,"G5":800,"XP_D4":6400}
ORDER=("XP_N","G5","XP_D4")
CLASSES={"G","O_g","O_p","C","I"}

def gate(x,msg):
    if not x:raise RuntimeError("STOP_XPC3_SOURCE_"+msg)

def sha(p):
    p=Path(p)
    gate(p.is_file() and not p.is_symlink(),"MISSING_"+str(p))
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def jread(p):return json.loads(Path(p).read_text())

def table(p):
    with Path(p).open(newline="") as f:return list(csv.DictReader(f))

def one(pattern,filename):
    xs=[p for p in BASE.glob(pattern) if
        p.is_dir() and not p.is_symlink() and
        (p/filename).is_file()]
    gate(len(xs)==1,"UNIQUE_"+pattern)
    return xs[0]

def item(panel,iid,run,label,pt):
    gate(isinstance(iid,str) and iid and
         isinstance(run,int) and 0<=run<64 and
         label in CLASSES,"ID_RUN_LABEL")
    pt=Path(pt)
    gate(pt.name=="reconstruction.pt"
         and ROOT.resolve() in pt.resolve().parents
         and not pt.is_symlink() and pt.is_file()
         and not any(w in str(pt).lower() for w in
                     ("confirmation","confirm305","test-ffhq","demo-ffhq")),
         "SAFE_CANDIDATE_PATH")
    uid=f"XPC3|{panel}|{iid}|{run}"
    return dict(panel=panel,image_id=iid,run=run,
        label=label,uid=uid,pt_path=str(pt),
        pt_sha256=sha(pt))

def d4(a):
    from scripts.track_r.xp_a15_cp2_d4_cpu import (
        provenance,find_one_glob,jread as canonical_read)
    from scripts.track_r.xp_a10_d1p2_compare_cpu import (
        P1_HEAD,P1_SHA,A6_HEAD,D1_SHA)
    from scripts.track_r.xp_fast_gpu import validate_complete
    s=one("xp_a15_cp2_d4_*","XP_A15_CP2_D4_SCORING_RECEIPT.json")
    rec=jread(s/"XP_A15_CP2_D4_SCORING_RECEIPT.json")
    f=s/"D4_64RUN_STRINGS.csv"
    gate(sha(f)==rec["csv_sha256"]["D4_64RUN_STRINGS.csv"]
         and rec["n_candidates"]==6400,"D4_SCORE_RECEIPT")
    records={r["image_id"]:r for r in table(f)}
    data=provenance(a)
    _,_,_,_,_,grouped,_,_,_,_,_=data
    from scripts.track_r import xp_a10_d1p2_compare_cpu as d1m
    previous=getattr(d1m,"ARGS",None);d1m.ARGS=a
    try:p1,p2,inputs,sources=d1m.source_contract()
    finally:d1m.ARGS=previous
    gate(len(records)==100 and len(p1)==400 and len(p2)==1200
         and len(grouped)==100,"D4_GRID")
    result=[]
    for iid in sorted(records):
        labels=records[iid]["labels_by_run"].split(";")
        gate(len(labels)==64,"D4_LABEL64")
        for k in range(64):
            if k<16:
                row,pt=(p1 if k<4 else p2)[iid,k]
                validate_complete(pt.parent,row,
                       P1_HEAD if k<4 else A6_HEAD,
                       P1_SHA["D1_P1"] if k<4 else D1_SHA)
            else: pt=grouped[iid][k][0]/"reconstruction.pt"
            result.append(item("XP_D4",iid,k,labels[k],pt))
    gate(len(result)==6400,"D4_6400")
    return result,dict(source_CSV_sha256=sha(f),
          source_receipt_sha256=sha(s/"XP_A15_CP2_D4_SCORING_RECEIPT.json"))

def g5():
    from scripts.track_r.xp_a15_cp3_score_cpu import stage_records
    s=one("xp_a15_cp3_G5_*","XP_A15_CP3_STAGE_SCORING_RECEIPT.json")
    rec=jread(s/"XP_A15_CP3_STAGE_SCORING_RECEIPT.json")
    f=s/"CP3_LOCAL_CANDIDATES.csv"
    gate(rec["stage"]=="G5" and
         sha(f)==rec["SHA256SUMS"]["CP3_LOCAL_CANDIDATES.csv"],
         "G5_RECEIPT")
    original,meta=stage_records("G5")
    orig={(r["image_id"],int(r["rep"])):r for r in original}
    scores=table(f)
    gate(len(orig)==len(scores)==800,"G5_GRID")
    result=[]
    for r in scores:
        iid=r["image_id"];j=int(r["run"])
        gate((iid,j) in orig
             and str(orig[iid,j]["png"])==r["reconstruction_path"]
             and orig[iid,j]["hash"]==r["reconstruction_sha256"],
             "G5_ORIGINAL_PNG_RECEIPT")
        pt=Path(r["reconstruction_path"]).with_name("reconstruction.pt")
        result.append(item("G5",iid,j,r["label"],pt))
    return result,dict(source_CSV_sha256=sha(f),
          source_receipt_sha256=sha(s/"XP_A15_CP3_STAGE_SCORING_RECEIPT.json"))

def xpn():
    from scripts.track_r.xp_a18_xpn_contract import (
       PHASES,validate_xpn_source)
    from scripts.track_r.xp_a12_campaign_fast_gpu import (
       root_folder,validate_complete)
    from scripts.track_r.xp_a22_cpn_part1_cpu import (
       TASKS,PROD)
    s=one("xp_a23_cpn_part2_*","XP_N_CPN_PART2_RECEIPT.json")
    rec=jread(s/"XP_N_CPN_PART2_RECEIPT.json")
    f=s/"XP_N_CPN64_STRINGS.csv"
    gate(rec["status"]=="A18_XPN_CPN_PART2_ALL100_N64_SCORED"
         and rec["n_native_roots"]==6400,
         "XPN_SCORED_6400")
    rs=table(f)
    gate(len(rs)==100,"XPN_100_IMAGES")
    labels={r["image_id"]:r["labels_by_run"].split(";") for r in rs}
    gate(all(len(z)==64 for z in labels.values()),"XPN_LABEL64")
    prep=jread(TASKS/"XP_N_TASKS_PREP_RECEIPT.json")
    final=jread(BASE/"xp_a22_xpn_continue_20261102T013122Z"/
                 "XP_N_CONTINUATION_FINAL_STATUS.json")
    gate(final["status"]=="A22_XPN_ALL6400_NATIVE_GENERATION_COMPLETE"
         and final["accepted_full_chunk_roots"]==6400,
         "XPN_NATIVE_FINAL")
    originals={
      "A18_XPN_PILOT_0_3":("XP_N_PILOT_0_3_TASKS.json",20),
      "A18_XPN_MAIN_4_15":("XP_N_MAIN_4_15_TASKS.json",60),
      "A18_XPN_LATE_16_63":("XP_N_LATE_16_63_TASKS.json",96)}
    result=[]
    for phase in originals:
        filename,step=originals[phase]
        p=TASKS/filename
        document=jread(p);digest=sha(p)
        gate(digest==prep["phase_files"][phase]["sha256"]
             and len(document["rows"])==PHASES[phase][2],
             "XPN_MANIFEST_"+phase)
        # Generation validation checks native uid/source/checkpoint/seed.
        validate_xpn_source(p,digest,document,document["rows"],
           prep["expected_head"],0,step)
        for i,row in enumerate(document["rows"]):
            iid=str(row["image_id"]);rep=int(row["rep"])
            gate(iid in labels and labels[iid][rep] in CLASSES,
                 "XPN_LABEL_IDENTITY")
            off=i//step*step
            native=PROD/"production"/phase/f"chunk_{off:05d}_{step:03d}"
            source=root_folder(native,row)
            # This validates the source SHA and exact native root hash.
            validate_complete(source,row,
               final["original_execution_head"],digest)
            result.append(item("XP_N",iid,rep,labels[iid][rep],
                               source/"reconstruction.pt"))
    gate(len(result)==6400,"XPN_6400_ROWS")
    return result,dict(source_CSV_sha256=sha(f),
       source_receipt_sha256=sha(s/"XP_N_CPN_PART2_RECEIPT.json"),
       native_A22_final_receipt_sha256=sha(
        BASE/"xp_a22_xpn_continue_20261102T013122Z"/
        "XP_N_CONTINUATION_FINAL_STATUS.json"))

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    gate(subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
         text=True).strip()==a.expected_head and
         not subprocess.check_output(["git","-C",str(WT),
         "status","--porcelain"],text=True).strip(),"PINNED_CLEAN_WORKTREE")
    gate(sha(MODEL)==MODEL_SHA,"IMAGENET_D1_CHECKPOINT")
    gate(not OUT.exists(),"SOURCE_MANIFEST_NOT_YET_REGISTERED")
    data={};receipts={}
    for panel in ORDER:
        f={"XP_N":lambda:xpn(),"G5":lambda:g5(),
           "XP_D4":lambda:d4(a)}[panel]
        rows,receipts[panel]=f()
        gate(len(rows)==SIZES[panel],"PANEL_"+panel)
        data[panel]=rows
        print("A20_XPC3_SOURCE_PREP",panel,len(rows),flush=True)
    result=[x for panel in ORDER for x in sorted(
            data[panel],key=lambda x:(x["image_id"],x["run"]))]
    gate(len(result)==13600
         and len({x["uid"] for x in result})==13600
         and all((x["uid"]==
             f'XPC3|{x["panel"]}|{x["image_id"]}|{x["run"]}')
             for x in result),"XPC3_UNIQUE_UIDS")
    for panel in ORDER:
        groups=Counter(x["image_id"] for x in data[panel])
        gate(len(groups)==100 and set(groups.values())=={
          8 if panel=="G5" else 64},"COMPLETE_PANEL_"+panel)
    manifest=dict(status="A20_XPC3_ALL13600_SOURCE_FROZEN_UNSCORED",
       expected_source_head=a.expected_head,
       no_candidate_tensor_pixels_read=True,
       no_GT_file_opened=True,
       no_prior_scoring=True,
       checkpoint=str(MODEL),checkpoint_sha256=MODEL_SHA,
       dataset="ImageNet",panel_order=list(ORDER),
       sizes=SIZES,n_candidates=13600,
       uid_rule="XPC3|<panel>|<image_id>|<run>",
       seed_rule="prior_seed(uid,t,j) in prdiffusion/track_r/prior_score.py",
       schedule="frozen_alpha_schedule() 1000-step beta linear",
       orientation=["000","111"],
       timesteps=[300,400,500,600],draws=[0,1,2,3],
       margin="L000-L111",decision_111_threshold_exclusive=1e-7,
       per_panel_source_receipts=receipts,
       rows=result)
    if a.preflight_only:
        print("A20_XPC3_PREP_PREFLIGHT_PASS",len(result),flush=True)
        return
    with OUT.open("x") as f:
        json.dump(manifest,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print("A20_XPC3_SOURCE_MANIFEST_STAGED",str(OUT),
          "SHA256",sha(OUT),"ROWS",len(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A20_XPC3_SOURCE",str(e),flush=True)
        sys.exit(8)
