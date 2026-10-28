#!/usr/bin/env python3
"""B16 automatic production: CPU-only frozen 2,160-root GT-free manifest.

B22 continues exactly the B22 post-model seed-43 stream (runs 0..15,
prefix 0..3 recomputed but not saved); DEV80 continues the frozen B24
independent 63-bit seed hash domain. No new outcomes or GT are read.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,traceback
from collections import defaultdict
from pathlib import Path
from scripts.track_r.xp_a10_b16_gate_prepare_cpu import (
    ROOT,BASE,WT,A6,A6_HEAD,SITCOM,SITCOM_HEAD,MODEL_SHA,
    safe,sha,git,settings,check_native_checkout,
    external_checkpoint_provenance)
from scripts.track_r.xp_sitcom_replay_prepare_cpu import (
    CATALOG,native_seed_metadata)
B24_SCREEN_SHA="b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba"
OLD_HEAD="e9696e9c3fc34da76cbad99c252dcbf9457be74a"
GATE_NAME="xp_a10_b16_gate_20261028T105721Z"
GATE_MANIFEST_SHA="9249b12d5e694e3ba06e19570eaffa35128f360663ed0b114b60851d61dcd423"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def check(c,label):
    if not c:raise RuntimeError("STOP_B16_PROD_PREP_"+label)

def seed63(label,image,rep):
    s="|".join(["B24_DEV80_SITCOM_SOLVER_V1",
                B24_SCREEN_SHA,label,image,str(rep)])
    return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)&((1<<63)-1)

def unique_stored_seed(image,historical):
    # Historical native seed bytes prove the frozen B24 class-dependent hash
    # domain without requiring any GT, oracle value or class outcome.
    labels=[]
    for label in "ABCD":
        candidate=[seed63(label,image,i)%(2**32) for i in range(4)]
        if candidate==historical:labels.append(label)
    check(len(labels)==1,"B24_SEED_DOMAIN_"+image)
    return labels[0]

def build(b22,cat):
    check(len(b22)==400 and len({r["uid"] for r in b22})==400,
          "B22_FROZEN_400")
    hist=defaultdict(list)
    for r in cat:
        if r.get("source")=="S2_b24_dev80" and r.get("family")=="SITCOM":
            hist[str(r["image_id"])].append(r)
    check(len(hist)==80 and sum(map(len,hist.values()))==320,
          "B24_HISTORICAL_320")
    b22_groups=[];b24_groups=[]
    for i in range(100):
        h=b22[i*4:(i+1)*4]
        iid=h[0]["image_id"]
        check(len(h)==4 and [r["rep"] for r in h]==[0,1,2,3]
              and len({r["image_id"] for r in h})==1,"B22_PREFIX_ORDER")
        check([r["uid"] for r in h]==[
              f"S1_b22|SITCOM|{iid}|run{rep}" for rep in range(4)],
              "B22_PREFIX_UID")
        rows=[dict(r) for r in h]
        for rep in range(4,16):
            r=dict(h[0],uid=f"S1_b22|SITCOM|{iid}|run{rep}",rep=rep)
            rows.append(r)
        b22_groups.append(dict(source="S1_b22",image_id=iid,rows=rows,
                  task_id=f"b22_{i:03d}_{iid}",new_roots=12,
                  historical_prefix_computations=4,
                  runtime_mode="sequential_post_model"))
    for iid in sorted(hist):
        h=sorted(hist[iid],key=lambda r:int(r["run_index"]))
        check([int(r["run_index"]) for r in h]==[0,1,2,3],
              "B24_PREFIX_ORDER_"+iid)
        native_seeds=[]
        for row in h:
            seed,_=native_seed_metadata(row,"independent_model_seed")
            native_seeds.append(int(seed))
        label=unique_stored_seed(iid,native_seeds)
        all16=[seed63(label,iid,rep)%(2**32) for rep in range(16)]
        check(len(set(all16))==16 and
              all16[:4]==native_seeds,"B24_SEED_COLLISION_"+iid)
        first=h[0]
        for row in h:
            check(row["measurement_path"]==first["measurement_path"]
                  and row["measurement_sha256"]==first["measurement_sha256"],
                  "B24_Y_EXCHANGE_"+iid)
        for rep in range(4,16):
            uid=f"S2_b24_dev80|SITCOM|{iid}|rep{rep}"
            b24_groups.append(dict(
                source="S2_b24_dev80",image_id=iid,
                task_id=f"b24_{iid}_{rep:02d}",
                rows=[dict(uid=uid,source="S2_b24_dev80",
                     family="SITCOM",measurement_source="S2_b24_dev80",
                     image_id=iid,rep=rep,seed=all16[rep],
                     rng_mode="independent_model_seed",
                     measurement_path=first["measurement_path"],
                     measurement_sha256=first["measurement_sha256"])],
                new_roots=1,historical_prefix_computations=0,
                runtime_mode="independent_model_seed"))
    check(len(b22_groups)==100 and len(b24_groups)==960,
          "GROUP_COUNT")
    pairs=[g["rows"][-1]["uid"] for g in b22_groups+b24_groups]
    check(len(set(pairs))==len(pairs),"LAST_UID_COLLISION")
    return b22_groups,b24_groups

def build_b32(b22_groups,b24_groups):
    """Authorized BACKGROUND after B16, same lineage, new indices 16..31.

    Requires no new outcome / GT reads, only the frozen source-domain seeds.
    """
    b32_b22=[];b32_b24=[]
    for index,g in enumerate(b22_groups):
        prefix=[dict(r) for r in g["rows"]]
        check(len(prefix)==16 and
              [r["rep"] for r in prefix]==list(range(16)),
              "B32_B22_PREFIX")
        iid=g["image_id"]
        rows=prefix
        for rep in range(16,32):
            rows.append(dict(prefix[0],
                uid=f"S1_b22|SITCOM|{iid}|run{rep}",rep=rep))
        b32_b22.append(dict(
            task_id=f"b32_b22_{index:03d}_{iid}",source="S1_b22",
            image_id=iid,rows=rows,new_roots=16,
            historical_prefix_computations=16,
            runtime_mode="sequential_post_model"))
    byimage=defaultdict(list)
    for g in b24_groups:
        byimage[g["image_id"]].append(g["rows"][0])
    check(len(byimage)==80,"B32_B24_IMAGE_COUNT")
    for iid in sorted(byimage):
        old=sorted(byimage[iid],key=lambda r:r["rep"])
        check([r["rep"] for r in old]==list(range(4,16)),
              "B32_B24_PREFIX")
        labels=[c for c in "ABCD" if all(
            seed63(c,iid,r["rep"])%(2**32)==r["seed"] for r in old)]
        check(len(labels)==1,"B32_B24_SEED_LINEAGE")
        label=labels[0]
        full=[seed63(label,iid,rep)%(2**32) for rep in range(32)]
        check(len(set(full))==32,"B32_B24_COLLISION")
        for rep in range(16,32):
            r=dict(old[0],uid=f"S2_b24_dev80|SITCOM|{iid}|rep{rep}",
                   rep=rep,seed=full[rep])
            b32_b24.append(dict(
                task_id=f"b32_b24_{iid}_{rep:02d}",source="S2_b24_dev80",
                image_id=iid,rows=[r],new_roots=1,
                historical_prefix_computations=0,
                runtime_mode="independent_model_seed"))
    check(len(b32_b22)==100 and len(b32_b24)==1280 and
          sum(g["new_roots"] for g in b32_b22+b32_b24)==2880,
          "B32_2880_TASKS")
    return b32_b22,b32_b24

def main(a):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
    check(git(WT,"rev-parse","HEAD")==a.expected_head and
          not git(WT,"status","--porcelain"),"RUNNER_HEAD")
    check(git(A6,"rev-parse","HEAD")==A6_HEAD and
          not git(A6,"status","--porcelain"),"A6_HEAD")
    gate=BASE/GATE_NAME
    prep=json.loads(safe(gate/"XP_A10_B16_GATE_PREP.json").read_text())
    path=gate/"XP_A10_B16_GATE_TASKS.json"
    check(prep["head"]==OLD_HEAD and
          prep["manifest_sha256"]==GATE_MANIFEST_SHA and
          sha(path)==GATE_MANIFEST_SHA,"GATE_MANIFEST")
    gated=json.loads(path.read_text())
    check(gated["head"]==OLD_HEAD and
          gated["n_roots"]==400 and gated["n_images"]==100 and
          gated["GT_supplied"] is False and
          gated["historical_candidate_supplied"] is False,
          "GATE_INPUTS")
    native=check_native_checkout(
        git(SITCOM,"rev-parse","HEAD"),
        git(SITCOM,"status","--porcelain=v1","--untracked-files=no"),
        git(SITCOM,"diff","--numstat","HEAD","--","posterior_sample.py"))
    native["unused_cli_sha256"]=sha(SITCOM/"posterior_sample.py")
    check(native==gated["native_checkout_provenance"],
          "NATIVE_PROVENANCE")
    check(external_checkpoint_provenance(SITCOM/"checkpoint/ffhq256.pt")
          ==gated["native_checkpoint_provenance"],"CHECKPOINT")
    check(settings(json.loads(safe(A6/"configs/b22/b22_2_overnight.json").read_text()))
          ==gated["config"],"SITCOM_CONFIG")
    cat=json.loads(safe(CATALOG).read_text())["rows"]
    check(len(cat)==3240,"R07_CATALOG")
    b22,b24=build(gated["rows"],cat)
    b32_b22,b32_b24=build_b32(b22,b24)
    out=Path(a.out)
    check(out.parent==BASE and out.name.startswith("xp_a11_b16_prod_") and
          not out.exists() and not out.is_symlink(),"OUT")
    out.mkdir(exist_ok=False)
    manifest={
        "schema":"prdiff.track_r.xp.a11.b16-production.v1",
        "status":"B16_PRODUCTION_INPUTS_FROZEN_GATE_PENDING",
        "head":a.expected_head,"gate_run":str(gate),
        "gate_head":OLD_HEAD,"gate_manifest_sha256":GATE_MANIFEST_SHA,
        "dataset":"ffhq","source_head":SITCOM_HEAD,
        "native_source_head":SITCOM_HEAD,"model_sha256":MODEL_SHA,
        "native_checkout_provenance":native,
        "native_checkpoint_provenance":gated["native_checkpoint_provenance"],
        "config":gated["config"],
        "b24_seed_domain":"B24_DEV80_SITCOM_SOLVER_V1",
        "b24_seed_screen_sha":B24_SCREEN_SHA,
        "b22_rng_rule":"seed43_post_model_reset_per_image_then_runs0_to15",
        "n_b22_images":100,"n_b24_images":80,
        "n_new_roots":2160,"n_new_b22":1200,"n_new_b24":960,
        "n_background_b32_roots":2880,"n_b32_b22":1600,
        "n_b32_b24":1280,
        "background_requires_full_b16_completion":True,
        "GT_supplied":False,"historical_candidate_supplied":False,
        "new_outcomes_embargoed":True,
        "b22_groups":b22,"b24_groups":b24,
        "b32_b22_groups":b32_b22,"b32_b24_groups":b32_b24}
    mf=out/"XP_A11_B16_PROD_TASKS.json"
    with mf.open("x") as f:
        json.dump(manifest,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    report=dict(status="B16_PRODUCTION_PREP_PASS_GATE_PENDING",
                head=a.expected_head,manifest_sha256=sha(mf),
                gate_manifest_sha256=GATE_MANIFEST_SHA,
                n_new=2160,n_b22=1200,n_b24=960,
                no_gpu=True,no_gt_read=True,
                no_new_outcomes_read=True)
    with (out/"XP_A11_B16_PROD_PREP.json").open("x") as f:
        json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
    print("B16_PRODUCTION_PREP_PASS_GATE_PENDING new_roots=2160",flush=True)
    print("B16_PRODUCTION_MANIFEST_SHA="+report["manifest_sha256"],flush=True)
    print("B16_PRODUCTION_RUN="+str(out),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: B16_PRODUCTION_PREP "+str(e),flush=True)
        sys.exit(8)
