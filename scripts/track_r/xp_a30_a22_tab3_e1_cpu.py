#!/usr/bin/env python3
"""A22 Tab.3 ImageNet E1(G1) chosen-orientation metrics, strictly CPU.

DAPS D2 K4 and SITCOM D3 K4 derive from the SHA-frozen A7 detail.
DAPS K8 derives from G5 (D2 historical runs 0..3 + G5 runs 4..7).
PSNR uses frozen RGB8 scores; SSIM and VGG-LPIPS are exact piq
0.8.0 native-float metrics. GT is never fed to a selector.

All per-run GT scores remain in a local PAC output; ONLY per-image
chosen-policy metrics + aggregate statistics are returned to GitHub.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
PARENT="xp_a7_d2d3_20261028T053026Z"
A7_AGG={
"D2":("XP_A7_IMAGE_K4_D2_20261028.json",
      "176d7458a8b5017d5b032ae2e3465a7938a783bd660a67c1daa3de139a05"),
"D3":("XP_A7_IMAGE_K4_D3_20261028.json",
      "30f328d63b7d9b54e58c863ea33040d3ae2e8368c9c9cfea6f4712e7ecdec8d3")}
E1_KEYS=("panel","family","K","image_id","selected_run",
         "g1_choice","PSNR","SSIM","VGG_LPIPS","GOOD20")
E1_AGG_KEYS=("panel","family","K","n_images","PSNR","SSIM",
             "VGG_LPIPS","GOOD20")
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")

def gate(c,name):
    if not c:raise RuntimeError("STOP_A22_TAB3_"+name)
def sha(p):
    p=Path(p);r=p.resolve(strict=False)
    gate(p.is_file() and not p.is_symlink()
         and ROOT in r.parents
         and not any(x in str(r).lower() for x in BANNED),"SOURCE_PATH")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    with Path(p).open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
def csvwrite(p,rows,fields):
    with Path(p).open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        w.writeheader();w.writerows(rows)
def git(*args):return subprocess.check_output(
    ["git","-C",str(WT),*args],text=True,timeout=60).strip()

def load_a7(stage):
    from scripts.track_r.xp_a7_d2d3_score_cpu import (
        assert_completed,verify_one)
    file,expected_sha=A7_AGG[stage]
    source=WT/"docs/track_r/executors/xp/returns"/file
    receipt=read(source)
    gate(receipt["status"]=="XP_A7_K4_IMAGE_METRIC_AGGREGATE_PASS"
         and receipt["stage"]==stage
         and receipt["local_per_run_detail_sha256"]==expected_sha
         and receipt["n_candidates"]==400,
         "A7_PUBLISHED_RECEIPT_"+stage)
    folder=BASE/(PARENT+"_"+stage)
    detail=folder/"XP_A7_K4_LOCAL_DETAIL.json"
    gate(sha(detail)==expected_sha,"A7_LOCAL_ZIP_HASH_"+stage)
    rr=read(detail)["raw_scores"]
    gate(len(rr)==400 and len({(r["image_id"],r["rep"]) for r in rr})==400,
         "A7_ALL_K4_"+stage)
    run,manifest,manifest_sha,head=assert_completed(stage)
    gate(len(manifest)==400,"A7_FROZEN_400_"+stage)
    records={}
    for row in manifest:
        key=(row["image_id"],row["rep"])
        gate(key not in records,"A7_SOURCE_DUP")
        p=verify_one(run,stage,row,manifest_sha,head)
        records[key]=(row,p)
    a7={(r["image_id"],r["rep"]):r for r in rr}
    gate(set(records)==set(a7) and
         all(a7[k]["uid"]==row["uid"]
             for k,(row,_) in records.items()),"A7_UID_IDENTITY_"+stage)
    return a7,records,dict(detail_sha256=sha(detail),
        native_manifest_sha256=manifest_sha,
        native_head=head,published_receipt_sha256=sha(source))

def g1_select(runs,K):
    gate(len(runs)>=K and K>=1
         and len({int(r["run"]) for r in runs})==len(runs)
         and all(math.isfinite(float(r["rho"])) for r in runs),
         "SELECTOR_INPUT")
    panel=sorted(runs,key=lambda r:int(r["run"]))[:K]
    gate([int(z["run"]) for z in panel]==list(range(K)),
         "CONTIGUOUS_FROZEN_RUNS")
    return min(panel,key=lambda r:(float(r["rho"]),int(r["run"])))

def daps_g5(a7_d2,refs):
    from scripts.track_r.xp_a16_cp3_joint_readout_cpu import read_stage
    data=read_stage("G5")
    sc,clips,mm=data["candidates"],data["clips"],data["metrics"]
    gate(len(sc)==len(clips)==len(mm)==800,"G5_NATIVE_800")
    keys=set(sc)
    gate(keys==set(clips)==set(mm),"G5_KEY_GRID")
    by=defaultdict(list)
    for key,z in sc.items():
        iid,j=key
        gate(iid.endswith(".JPEG") and 0<=j<8 and
             z["image_id"]==iid and int(z["run"])==j and
             clips[key]["image_id"]==iid and
             int(clips[key]["run"])==j and
             mm[key]["image_id"]==iid and
             int(mm[key]["run"])==j,
             "G5_JOIN")
        if j<4:
            a=a7_d2.get((iid,j))
            gate(a is not None,"A7_D2_G5_IMAGE")
            # Frozen DAPS run indices 0..3 must refer to the same
            # reconstruction, not a second replicate/round.
            old=refs[(iid,j)][1].with_name("reconstruction.png")
            gate(sha(old)==z["reconstruction_sha256"],
                 "A7_D2_G5_NATIVE_RECON_HASH")
        orient=clips[key]["g1_choice"]
        gate(orient in ("000","111"),"G5_G1_CHOICE")
        flip=orient=="111"
        ps=float(z["psnr_flip"] if flip else z["psnr_raw"])
        ss=float(mm["bad"]) if False else float(
           mm[key]["SSIM_111"] if flip else mm[key]["SSIM_raw"])
        lp=float(mm[key]["LPIPS_VGG_111"] if flip else
                 mm[key]["LPIPS_VGG_raw"])
        gate(all(math.isfinite(z) for z in (ps,ss,lp)),
             "G5_FINITE_METRICS")
        by[iid].append(dict(run=j,rho=float(z["rho_signed"]),
             g1_choice=orient,PSNR=ps,SSIM=ss,VGG_LPIPS=lp))
    gate(len(by)==100 and
         all(sorted(v["run"] for v in g)==list(range(8))
             for g in by.values()),"G5_100_8_GRID")
    return by

def sitcom_d3(a7_d3,refs):
    from scripts.track_r.xp_a15_cp2_d4_cpu import clip_new
    from scripts.track_r.xp_a29_xpi2_cpu_publish import metrics_pretrained
    from scripts.track_r.xp_a15_cp3_score_cpu import common_sources
    from scripts.track_r.xp_a16_cp3_piq_metrics_cpu import true_image
    from prdiffusion.track_r.physics import apply_orbit
    import torch,piq
    torch.set_num_threads(4)
    _,gt,yy=common_sources()
    gate(len(gt)==len(yy)==100,"IMAGE_GT_GRID")
    clipsrc=[]
    for key,(record,pt) in sorted(refs.items()):
        z=a7_d3[key]
        gate(record["measurement_sha256"]==yy[key[0]]["measurement_sha256"],
             "SITCOM_SHARED_MEASUREMENT_SHA")
        orbit=z["orbit"]["psnr_db"]
        gate(len(orbit)==8 and
             abs(float(orbit[0])-float(z["psnr_db"]))<1e-4,
             "A7_111_ORBIT_ORDER")
        clipsrc.append(dict(image_id=key[0],run=key[1],label="",
            psnr_raw=float(orbit[0]),flip_good=int(float(orbit[7])>=20),
            reconstruction_path=str(pt.with_name("reconstruction.png"))))
    clip,meta=clip_new(clipsrc,mode="g1",threshold=20)
    decisions={(z["image_id"],int(z["run"])):z for z in clip}
    gate(len(decisions)==400 and set(decisions)==set(refs),"SITCOM_G1_400")
    net,metric_info=metrics_pretrained()
    by=defaultdict(list)
    cache={}
    with torch.inference_mode():
        for k,(record,path) in sorted(refs.items()):
            src=a7_d3[k];c=decisions[k];flip=c["g1_choice"]=="111"
            iid,j=k
            if iid not in cache:cache[iid]=true_image(iid,"G5",{},gt)
            target=((cache[iid].float().clamp(-1,1)+1.)/2.)
            x=torch.load(path,map_location="cpu",
                         weights_only=True)["reconstruction"].float()
            gate(tuple(x.shape)==(1,3,256,256)
                 and bool(torch.isfinite(x).all()),"NATIVE_FLOAT")
            if flip:x=apply_orbit(x,"111")
            z=((x.clamp(-1,1)+1.)/2.)
            ss=float(piq.ssim(z,target,data_range=1.,
                              reduction="none")[0])
            lp=float(net(z,target)[0])
            ps=float(src["orbit"]["psnr_db"][7 if flip else 0])
            gate(all(math.isfinite(v) for v in (ps,ss,lp)),
                 "SITCOM_FINITE_METRICS")
            by[iid].append(dict(run=j,rho=float(src["rho_signed"]),
                 g1_choice=c["g1_choice"],PSNR=ps,SSIM=ss,
                 VGG_LPIPS=lp))
        gate(len(by)==100 and
             all(sorted(x["run"] for x in rs)==list(range(4))
                 for rs in by.values()),"SITCOM_100_4_GRID")
    return by,dict(clip_model=meta,metric_info=metric_info)

def summarize(panel,family,K,groups):
    rows=[]
    for iid,g in sorted(groups.items()):
        chosen=g1_select(g,K)
        rows.append(dict(panel=panel,family=family,K=K,image_id=iid,
            selected_run=chosen["run"],
            g1_choice=chosen["g1_choice"],PSNR=chosen["PSNR"],
            SSIM=chosen["SSIM"],VGG_LPIPS=chosen["VGG_LPIPS"],
            GOOD20=int(chosen["PSNR"]>=20)))
    gate(len(rows)==100,"100_IMAGES_"+panel)
    result=dict(panel=panel,family=family,K=K,n_images=100,
        **{name:mean(float(r[name]) for r in rows)
           for name in ("PSNR","SSIM","VGG_LPIPS","GOOD20")})
    return rows,result

def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and git("rev-parse","HEAD")==a.expected_head
         and not git("status","--porcelain"),
         "PINNED_CPU_GIT")
    base=Path(a.out)
    gate(base.parent==BASE
         and base.name.startswith("xp_a30_a22_tab3_")
         and not base.exists(),"NEW_OUTPUT")
    d2,refs2,proof2=load_a7("D2")
    d3,refs3,proof3=load_a7("D3")
    if a.preflight_only:
        print("A22_TAB3_A7_800_NATIVE_SHA_PREFLIGHT_PASS",flush=True)
        return
    daps=daps_g5(d2,refs2)
    sitcom,meta=sitcom_d3(d3,refs3)
    rows=[];agg=[]
    for family,panel,K,groups in (
        ("DAPS-4K","D2",4,daps),
        ("DAPS-4K","D2",8,daps),
        ("SITCOM","D3",4,sitcom)):
        a,b=summarize(panel,family,K,groups)
        rows.extend(a);agg.append(b)
    gate(len(rows)==300 and len(agg)==3,"TABLE3_GRID")
    base.mkdir(mode=0o700)
    tab=base/"XP_A22_TAB3_IMAGENET_E1_G1.csv"
    detail=base/"XP_A22_TAB3_IMAGE_E1_G1.csv"
    csvwrite(tab,agg,E1_AGG_KEYS)
    csvwrite(detail,rows,E1_KEYS)
    report=dict(status="A22_TABLE3_E1_IMAGE_NET_CPU_PASS",
        n_images=100,n_policy_rows=300,n_tables=3,
        A7={"D2":proof2,"D3":proof3},
        source_DAPS_G5="frozen CP3 read_stage('G5'), first four D2 historical roots verified bytewise",
        source_SITCOM="frozen A7 D3 400 native roots",
        G1="frozen G1 upright 'a photo.' vs inverted 'an upside-down photo.'; choose 111 iff margin>0",
        selector="E0 min signed measurement rho among first K, stable run index; then G1 orientation",
        SSIM="piq 0.8.0, native float [0,1], chosen orientation",
        VGG_LPIPS="piq 0.8.0 pretrained VGG, native float [0,1], chosen orientation",
        PSNR="frozen RGB8 at chosen orientation",
        simulator_or_solver_gpu_calls=0,no_GT_in_selection=True,
        GT_for_offline_metrics_only=True,
        not_matched_compute=True,optional_a21_10_replication_not_run=True,
        implementation_head=a.expected_head,
        SITCOM_G1_and_metrics=meta,
        table_sha256=sha(tab),local_image_detail_sha256=sha(detail))
    receipt=base/"XP_A22_TAB3_RECEIPT.json"
    write(receipt,report)
    print("A22_TABLE3_E1_PASS",json.dumps(agg,sort_keys=True),flush=True)
    print("A22_TAB3_LOCAL",base,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--preflight-only",action="store_true")
    try:main(ap.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A22_TAB3",str(e),flush=True)
        sys.exit(8)
