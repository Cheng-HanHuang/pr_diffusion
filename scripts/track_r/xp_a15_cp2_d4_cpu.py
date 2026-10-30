#!/usr/bin/env python3
"""A15 CP2 FAST ImageNet D4: verifiable n64 scoring and frozen protocol.

CPU-only. Authenticates v4 registry BEFORE opening any new D4 candidate.
Keeps the previously scored 16-run source unchanged, verifies G1a/G1b
native receipts, scores runs16..63, applies native zero-shot G1 CLIP
without GT in selection, compares v2/v4, and runs c598176 protocol.
All output stays local; the checkpoint packager independently audits.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,importlib,json,math,os,subprocess,sys,traceback
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a12_campaign_fast_gpu import root_folder,validate_complete
from scripts.track_r.xp_t_sitcom_prepare_cpu import IMAGENET,COMMON_Y_SHA
from scripts.track_r.xp_d1p2_a11_export_cpu import D1_SHA
from scripts.track_r.xp_a8_d1p1_clip_cpu import model_source,G1_UP,G1_INV
from prdiffusion.track_r.r0_7_stats import orbit_psnr_channel_mse
from prdiffusion.track_r.physics import to_rgb8,rho_signed
from prdiffusion.track_r.r0n_clip_compat import encode_text_univfd_tuple_compat

G1A_RUN=BASE/"xp_a12_g1a_prod_20261029T180624Z"
G1A_PREP=BASE/"xp_a12_g1a_prepare_20261029T170316Z"
G1B_SOURCE=BASE/"XP_A13_G1B_LATEST_RUN.txt"
PREREG_HEAD="f31901c808954f2bee724ab8a3834e2b65ef25c9"
REG_PATH="docs/track_r/theory/PREDICTIONS_REGISTRY.md"
V4=("149331c319bb8697c1a0f40c04bf0e4016ec3420",
    "docs/track_r/theory/TH1_PREDICTIONS_v4.json",
    "b0ff25bc98ef7ba294db3f33c4df4f1f9835951890ebd879afedf807c9d93da3")
V2=("340dc5dcbcde96df4edc938f1b807faedc18936b",
    "docs/track_r/theory/TH1_PREDICTIONS_v2.json",
    "8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55")
V2E1=("9cb57f51337f2730db1644d570b0e2612855e728",
    "docs/track_r/theory/TH1_PREDICTIONS_v2_E1.json",
    "13ff550ca42c0941c430eff75e1f9ecc107dbe3801ee19d355371518980ec406")
PROTO=("c598176","docs/track_r/protocol/gtfree_protocol.py",
       "d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658")
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")
NAMES=("image_id","cohort","labels_by_run","rho_signed_by_run",
       "tau_signed_good_q95","n_G","n_O_g","n_O_p","n_C","n_I")
CLIP_NAMES=("image_id","run","label","g1_choice","g1_margin","g1_chosen_good")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def safe(p):
 p=Path(p)
 gate(p.is_file() and not p.is_symlink() and ROOT.resolve() in p.resolve().parents
      and not any(s in str(p.resolve()).lower() for s in BANNED),"CP2_SAFE_PATH")
 return p
def jread(p):return json.loads(safe(p).read_text())
def write(p,j):
 with p.open("x") as f:json.dump(j,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def csvload(p):
 with safe(p).open(newline="") as f:return list(csv.DictReader(f))
def csvwrite(p,rows,columns):
 with p.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=columns)
  w.writeheader();w.writerows(rows)
def git(wt,*args):
 return subprocess.check_output(["git","-C",str(wt),*args],timeout=120)
def frozen(wt):
 # The registration itself precedes all D4 run16+ analysis.
 reg=git(wt,"show",PREREG_HEAD+":"+REG_PATH).decode()
 artifacts={}
 for key,(ref,path,digest) in (("v4",V4),("v2",V2),
                               ("v2 E1",V2E1),("protocol v1",PROTO)):
  if key!="protocol v1":
   lines=[line for line in reg.splitlines() if line.startswith("| "+key+" |")]
   gate(len(lines)==1 and ref in lines[0] and digest in lines[0]
        and "REGISTERED" in lines[0],"CP2_FORECAST_REGISTRY_"+key)
  blob=git(wt,"show",ref+":"+path)
  gate(hashlib.sha256(blob).hexdigest()==digest,"CP2_FORECAST_SHA_"+key)
  artifacts[key]=json.loads(blob) if key!="protocol v1" else blob
 gate(len(artifacts["v4"]["predictions"])==120
      and artifacts["v4"]["provenance"]["held_XP_D4_outcomes_seen"] is False,
      "CP2_V4_PRE_OUTCOME_REGISTRATION")
 return artifacts
def find_one_glob(pattern,receipt,status,expected_manifest_sha=None):
 candidates=[]
 for path in sorted(BASE.glob(pattern)):
  p=path/receipt
  if not p.is_file():continue
  x=jread(p)
  if x.get("status")!=status:continue
  if expected_manifest_sha and x.get("manifest_sha256")!=expected_manifest_sha:continue
  candidates.append(path)
 gate(len(candidates)==1,"CP2_UNIQUE_SOURCE_"+pattern)
 return candidates[0]
def provenance(a):
 wt=Path(__file__).resolve().parents[2]
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CP2_CPU_ONLY")
 gate(git(wt,"rev-parse","HEAD").decode().strip()==a.expected_head
      and not git(wt,"status","--porcelain"),"CP2_GIT_CLEAN")
 artifacts=frozen(wt)
 g1a=jread(G1A_RUN/"XP_A12_G1A_FINAL_STATUS.json")
 gate(g1a["status"]=="XP_A12_G1A_1600_GENERATION_COMPLETE_SEALED"
      and g1a["completed"]==1600 and g1a["replay_pass"] is True,
      "CP2_G1A_COMPLETE")
 g1b_run=Path(safe(G1B_SOURCE).read_text().strip())
 gate(g1b_run.parent==BASE and g1b_run.name.startswith("xp_a13_g1b_prod_"),
      "CP2_G1B_DIR")
 g1b=jread(g1b_run/"XP_A13_G1B_FINAL_STATUS.json")
 gate(g1b["status"]=="XP_A13_G1B_3200_GENERATION_COMPLETE"
      and g1b["completed"]==3200 and g1b["failed_attempts"]==0
      and g1b["v4_registered_sha256"]==V4[2],"CP2_G1B_COMPLETE")
 old=G1A_PREP/"XP_A12_G1A_1600_TASKS.json"
 mf_a=jread(old);gate(len(mf_a["rows"])==1600,"CP2_G1A_MANIFEST")
 prep_b=find_one_glob("xp_a12_g1b_prepare_*",
   "XP_A12_G1B_PREP_RECEIPT.json",
   "XP_A13_G1B_IMMEDIATE_PREPARED_CPU_PASS",g1b["manifest_sha256"])
 new=prep_b/"XP_A12_G1B_3200_TASKS.json"
 mf_b=jread(new);gate(len(mf_b["rows"])==3200 and
       sha(new)==g1b["manifest_sha256"],"CP2_G1B_MANIFEST")
 gate(sha(old)==g1a["production_manifest_sha256"] if
      "production_manifest_sha256" in g1a else
      sha(old)==jread(G1A_PREP/"XP_A12_G1A_PREP_RECEIPT.json")["production_manifest_sha256"],
      "CP2_G1A_MANIFEST_SHA")
 old_dirs=sorted(p for p in BASE.glob("xp_d1p2_protocol_*")
   if (p/"XP_D1P2_A11_INPUT_RECEIPT.json").is_file())
 gate(len(old_dirs)==1,"CP2_D1_16_PREFIX_UNIQUE")
 source=old_dirs[0]
 record=jread(source/"XP_D1P2_A11_INPUT_RECEIPT.json")
 gate(record["status"]=="XP_D1P2_A11_1600_FROZEN_PROTOCOL_INPUTS_PASS"
      and record["D1_comparison_sha256"]==D1_SHA
      and sha(source/"XP_D1P2_16RUN_STRINGS.csv")==record["strings_sha256"]
      and sha(source/"XP_D1P2_16RUN_CLIP_DECISIONS.csv")==record["clip_sha256"],
      "CP2_PREFIX_RECEIPT")
 initial=csvload(source/"XP_D1P2_16RUN_STRINGS.csv")
 clip0=csvload(source/"XP_D1P2_16RUN_CLIP_DECISIONS.csv")
 gate(len(initial)==100 and len(clip0)==1600,"CP2_16_PREFIX_GRID")
 gt=jread(IMAGENET/"XP_IMAGENET_GT_ONLY_CPU.json")
 commonpath=IMAGENET/"XP_IMAGENET100_Y_MANIFEST.json"
 gate(sha(commonpath)==COMMON_Y_SHA,"CP2_Y_MANIFEST_SHA")
 common=jread(commonpath)
 src={r["basename"]:r for r in gt["rows"]}
 measurements={r["basename"]:r for r in common["rows"]}
 ids={r["image_id"] for r in mf_a["rows"]}
 gate(len(ids)==100 and ids==set(src)==set(measurements)
      and ids=={r["image_id"] for r in initial},
      "CP2_100_SAME_IMAGES")
 grouped={i:{} for i in ids}
 for run,mf,expect,start,count,head in (
  (G1A_RUN,mf_a,g1a,16,16,mf_a["head"]),
  (g1b_run,mf_b,g1b,32,32,mf_b["head"])):
  for row in mf["rows"]:
   rep=row["rep"];iid=row["image_id"]
   gate(start<=rep<start+count and rep not in grouped[iid]
        and row["measurement_sha256"]==measurements[iid]["measurement_sha256"]
        and row["measurement_path"]==measurements[iid]["measurement_path"],
        "CP2_D4_TASK_LINEAGE")
   chunk=(run/"production"/f"chunk_{sorted(ids).index(iid)*count:05d}_{count:03d}")
   # Chunk index corresponds to sorted image IDs, not global run index.
   x=root_folder(chunk,row)
   receipt=validate_complete(x,row,head,sha(old if count==16 else new))
   gate(receipt["gt_access"] is False,"CP2_NEW_D4_NO_GT")
   grouped[iid][rep]=(x,row,receipt)
 gate(all(set(v)==set(range(16,64)) for v in grouped.values()),
      "CP2_NEW_PANEL_4800")
 return wt,artifacts,source,initial,clip0,grouped,src,measurements,g1b_run,old,new
def label(raw,orbit,best,rho,tau=.0500829):
 if raw>=20:return "G"
 if orbit>=20:return "O_g" if best=="111" else "O_p"
 return "C" if rho<=tau else "I"
def new_scores(data,a,out):
 wt,arts,source,initial,clip0,grouped,gtmap,ymap,g1brun,old,new=data
 import torch
 from PIL import Image
 from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
 torch.set_num_threads(4)
 tf=Compose([ToTensor(),Resize(256),CenterCrop(256)])
 prefix={r["image_id"]:r for r in initial}
 gate(len(prefix)==100,"CP2_PREFIX_KEYS")
 newrows=[];strings=[];cache={};pics=[]
 for num,iid in enumerate(sorted(grouped)):
  ref=gtmap[iid];ym=ymap[iid]
  gate(ref["source_sha256"]==ym["source_image_sha256"]
       and sha(ref["source_path"])==ref["source_sha256"]
       and sha(ym["measurement_path"])==ym["measurement_sha256"],
       "CP2_GT_Y_SHA")
  with Image.open(safe(ref["source_path"])) as im:
   g=tf(im.convert("RGB")).unsqueeze(0)*2-1
  g8=to_rgb8(g)
  y=torch.load(safe(ym["measurement_path"]),map_location="cpu",weights_only=True)
  y=(y["measurement"] if "measurement" in y else y["y"]).float()
  gate(tuple(y.shape)==(1,3,384,384),"CP2_SIGNED_Y")
  labels=(prefix[iid]["labels_by_run"].split(";"))
  rhos=prefix[iid]["rho_signed_by_run"].split(";")
  gate(len(labels)==len(rhos)==16,"CP2_PREFIX_16")
  for rep in range(16,64):
   path,row,receipt=grouped[iid][rep]
   pt=safe(path/"reconstruction.pt")
   x=torch.load(pt,map_location="cpu",weights_only=True)["reconstruction"].float()
   gate(tuple(x.shape)==(1,3,256,256) and bool(torch.isfinite(x).all()),
        "CP2_FAST_RECONSTRUCT_SHAPE")
   x8=to_rgb8(x)
   orb=orbit_psnr_channel_mse(x8,g8)
   best=max(sorted(orb),key=lambda k:orb[k])
   raw=float(orb["000"]); orbit=float(orb[best])
   rho=float(rho_signed(x,y)[0])
   gate(math.isfinite(rho) and rho>=0 and all(math.isfinite(q) for q in orb.values()),
        "CP2_FINITE")
   lab=label(raw,orbit,best,rho)
   labels.append(lab);rhos.append(format(rho,".12g"))
   file=safe(path/"reconstruction.png")
   gate(sha(file)==receipt["reconstruction_png_sha256"],"CP2_FAST_PNG_SHA")
   newrows.append(dict(image_id=iid,run=rep,label=lab,
      rho_signed=format(rho,".15g"),psnr_raw=format(raw,".12g"),
      psnr_orbit_best=format(orbit,".12g"),
      flip_good=int(float(orb["111"])>=20),
      reconstruction_path=str(file),reconstruction_sha256=sha(file)))
  ct=Counter(labels)
  strings.append(dict(image_id=iid,cohort="IN",
     labels_by_run=";".join(labels),rho_signed_by_run=";".join(rhos),
     tau_signed_good_q95=".0500829",n_G=ct["G"],n_O_g=ct["O_g"],
     n_O_p=ct["O_p"],n_C=ct["C"],n_I=ct["I"]))
  if (num+1)%10==0:print("CP2_D4_NEW_IMAGES_SCORED",num+1,"/100",flush=True)
 gate(len(newrows)==4800 and len(strings)==100 and
      all(len(r["labels_by_run"].split(";"))==64 for r in strings),
      "CP2_N64_COMPLETE")
 csvwrite(out/"D4_64RUN_STRINGS.csv",strings,NAMES)
 csvwrite(out/"D4_NEW_4800_LOCAL_SCORES.csv",newrows,list(newrows[0]))
 return strings,newrows,prefix
def clip_new(newrows,mode="g1",threshold=20):
 from PIL import Image
 import torch
 from prdiffusion.track_r.r0n_clip_compat import encode_text_univfd_tuple_compat
 clipfile,native,nativehead=model_source()
 if mode=="face":
  from scripts.track_r.r0n_2_clip_orientation import PROMPTS_UP,PROMPTS_INV
  up_prompts,inv_prompts=PROMPTS_UP,PROMPTS_INV
 elif mode=="g1":
  up_prompts,inv_prompts=G1_UP,G1_INV
 else:raise RuntimeError("STOP_A15_UNREGISTERED_CLIP_MODE")
 sys.path.insert(0,str(native))
 clip=importlib.import_module("models.clip")
 model,preprocess=clip.load(str(clipfile),device="cpu",jit=False)
 model.eval();torch.set_num_threads(4)
 torch.manual_seed(0)
 with torch.inference_mode():
  prompts=clip.tokenize(list(up_prompts+inv_prompts)).to("cpu")
  text=encode_text_univfd_tuple_compat(model,prompts).float()
  text=text/text.norm(dim=-1,keepdim=True)
  up=text[:len(up_prompts)].mean(0);down=text[len(up_prompts):].mean(0)
  up=up/up.norm();down=down/down.norm()
  result=[]
  for base in range(0,len(newrows),16):
   batch=newrows[base:base+16];images=[]
   for r in batch:
    with Image.open(safe(r["reconstruction_path"])) as img:
     im=img.convert("RGB");arr=np.asarray(im)
     inv=Image.fromarray(np.rot90(arr,2).copy(),"RGB")
     images.extend((preprocess(im),preprocess(inv)))
   z=model.encode_image(torch.stack(images)).float()
   z=z/z.norm(dim=-1,keepdim=True)
   margins=((z@up)-(z@down))[1::2]-((z@up)-(z@down))[::2]
   for row,margin in zip(batch,margins.tolist()):
    choice="111" if margin>0 else "000"
    good=int(bool(row["flip_good"])) if choice=="111" else int(float(row["psnr_raw"])>=threshold)
    result.append(dict(image_id=row["image_id"],run=row["run"],
       label=row["label"],**{mode+"_choice":choice,
       mode+"_margin":format(margin,".12g"),mode+"_chosen_good":good}))
   if (base+16)%480==0:print("CP2_CLIP_G1",min(base+16,len(newrows)),flush=True)
 gate(len(result)==4800,"CP2_4800_CLIP")
 return result,dict(checkpoint_sha256=sha(clipfile),native_head=nativehead)
def exact_observed(strings,cliprows,panel,K):
 by={(r["image_id"],int(r["run"])):r for r in cliprows}
 results=defaultdict(list)
 for s in strings:
  n=panel; lab=s["labels_by_run"].split(";")[:n]
  rho=[float(x) for x in s["rho_signed_by_run"].split(";")[:n]]
  order=sorted(range(n),key=lambda i:(rho[i],i))
  den=math.comb(n,K)
  w=[math.comb(n-j-1,K-1)/den if n-j>=K else 0 for j in range(n)]
  good=[x=="G" for x in lab]
  orbit=[x in ("G","O_g","O_p") for x in lab]
  E0=sum(w[j]*good[order[j]] for j in range(n))
  E1=sum(w[j]*int(by[s["image_id"],order[j]]["g1_chosen_good"]) for j in range(n))
  raw=1-(math.comb(n-sum(good),K)/den if n-sum(good)>=K else 0)
  orb=1-(math.comb(n-sum(orbit),K)/den if n-sum(orbit)>=K else 0)
  for name,val in (("E0",E0),("E1_CLIP",E1),("O_raw",raw),("O_orbit",orb)):
   results[name].append(float(val))
 return {k:sum(v)/len(v) for k,v in results.items()}
def comparison(artifacts,strings,cliprows):
 result={}
 for version in ("v2","v2 E1","v4"):
  forecast=artifacts[version]
  comparisons=[]
  predictions=forecast["predictions"]
  for row in predictions:
   policy=row["policy"];n=int(row.get("panel_n_max") or 16)
   # Do not silently mix a primary 32-run forecast with secondary 64.
   K=int(row["K"])
   if n not in (16,32,64) or not 1<=K<=n:continue
   obs=exact_observed(strings,cliprows,n,K)[policy]
   sd=max((float(row["hi95"])-float(row["lo95"]))/3.92,.005)
   comparisons.append(dict(panel=row.get("panel"),K=K,policy=policy,
      model_variant=row.get("model_variant"),decision_model=row.get("decision_model"),
      observed=obs,forecast=float(row["point"]),lo95=float(row["lo95"]),
      hi95=float(row["hi95"]),inside=float(row["lo95"])<=obs<=float(row["hi95"]),
      standardized_residual=abs(obs-float(row["point"]))/sd))
  gate(len(comparisons)==len(predictions),"CP2_FORECAST_ALL_ROWS_"+version)
  result[version]=dict(status="PREDICTIONS_COMPARED_NOT_REFIT",
     n=len(comparisons),covered=sum(x["inside"] for x in comparisons),
     comparisons=comparisons)
 return result
def run_protocol(wt,blob,out,strings,clip):
 script=out/"gtfree_protocol_v1_original.py"
 with script.open("xb") as f:f.write(blob)
 gate(sha(script)==PROTO[2],"CP2_PROTOCOL_PIN")
 outputs={}
 for n in (32,64):
  target=out/f"D4_PROTOCOL_N{n}.json"
  cmd=[sys.executable,"-I",str(script),"--strings",str(strings),
       "--clip",str(clip),"--tau","0.0500829","--m","2",
       "--nmax",str(n),"--R","4000","--seed","20261029","--out",str(target)]
  subprocess.run(cmd,check=True,cwd=str(wt),env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
  doc=jread(target)
  gate(doc["N_MAX"]==n and doc["n_images"]==100
       and doc["M"]==2 and doc["orientation_step"] is True,
       "CP2_PROTOCOL_N"+str(n))
  outputs[str(n)]=dict(sha256=sha(target),
     primary={k:doc.get(k) for k in (
       "success_all","runs_mean","gain_vs_fixed_equal_runs",
       "flag_c_expected_count","flag_c_precision_unsolved",
       "success_given_not_flag_c","success_unflagged")})
 return outputs
def main(a):
 data=provenance(a)
 wt,arts,source,initial,clip0,*_=data
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a15_cp2_d4_")
      and not out.exists() and not out.is_symlink(),"CP2_OUTPUT_NEW")
 if a.preflight_only:
  print("CP2_SOURCE_REGISTRY_AND_4800_NATIVE_RECEIPTS_PASS",flush=True)
  return
 out.mkdir(mode=0o700)
 strings,scored,previous=new_scores(data,a,out)
 clip_new_rows,clipmeta=clip_new(scored)
 clips=[{k:r[k] for k in CLIP_NAMES} for r in clip0]+clip_new_rows
 gate(len(clips)==6400 and len({(r["image_id"],int(r["run"])) for r in clips})==6400,
      "CP2_6400_CLIP")
 csvwrite(out/"D4_64RUN_G1_CLIP.csv",sorted(clips,key=lambda x:(x["image_id"],int(x["run"]))),CLIP_NAMES)
 comp=comparison(arts,strings,clips)
 write(out/"CP2_V2_V2E1_V4_FORECAST_COMPARISON.json",comp)
 proto=run_protocol(wt,arts["protocol v1"],out,
       out/"D4_64RUN_STRINGS.csv",out/"D4_64RUN_G1_CLIP.csv")
 n64=proto["64"]["primary"];n32=proto["32"]["primary"]
 def hypothesis(doc):
  h1=doc["gain_vs_fixed_equal_runs"]>=-.01
  h3=(None if doc["flag_c_expected_count"]<2 else doc["flag_c_precision_unsolved"]>=.5)
  h4=(doc["success_unflagged"] is not None and doc["success_given_not_flag_c"] is not None and
      doc["success_unflagged"]>=doc["success_given_not_flag_c"]+.05)
  return dict(H1=bool(h1),H3=h3,H4=bool(h4))
 receipt=dict(status="XP_A15_CP2_D4_SCORED_READY_FOR_INDEPENDENT_AUDIT",
   source_head=a.expected_head,checkpoint="CP2",n_images=100,
   n_candidates=6400,n_historical=1600,n_native_new=4800,
   forecast_sha256={k:v[2] for k,v in (("v2",V2),("v2 E1",V2E1),("v4",V4))},
   forecast_refit=False,v4_registered_before_outcomes=True,
   protocol_sha256=PROTO[2],proto_n32=proto["32"],proto_n64=proto["64"],
   hypothesis_n32=hypothesis(n32),hypothesis_n64=hypothesis(n64),
   clip_native=clipmeta,all_native_receipts_verified=True,
   no_GPU=True,GT_only_for_evaluation=True,
   csv_sha256={n:sha(out/n) for n in
        ("D4_64RUN_STRINGS.csv","D4_64RUN_G1_CLIP.csv","D4_NEW_4800_LOCAL_SCORES.csv")},
   forecast_comparison_sha256=sha(out/"CP2_V2_V2E1_V4_FORECAST_COMPARISON.json"),
   no_checkpoint_publication_yet=True)
 write(out/"XP_A15_CP2_D4_SCORING_RECEIPT.json",receipt)
 print("CP2_D4_SCORING_COMPLETE_INDEPENDENT_AUDIT_PENDING",json.dumps(
   dict(H32=receipt["hypothesis_n32"],H64=receipt["hypothesis_n64"],
        n64_success=n64["success_all"]),sort_keys=True),flush=True)
 print("CP2_LOCAL_OUTPUT",out,flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc();print("ACTION NEEDED: A15_CP2_D4",e,flush=True);sys.exit(8)
