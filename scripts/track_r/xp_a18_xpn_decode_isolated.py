#!/usr/bin/env python3
"""A16 XP-N untrusted raw JPEG decoding and D1-exact signed measurement.

INTENTIONALLY STANDALONE for <python> -I -B <script>, with cwd set to
the exact raw-image directory. No PYTHONPATH or user-site is inherited.
Imports the authenticated repository physics operator through an explicit,
pinned repo path. GT data never enters a GPU worker manifest.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path

def gate(ok,why):
 if not ok:raise RuntimeError("STOP_A18_XPN_ISOLATED_"+why)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(2**20),b""):h.update(block)
 return h.hexdigest()
def seed(prefix,basename,mod=2**63):
 k=prefix+"|"+basename
 return int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],"big")%mod
def write(path,document):
 with Path(path).open("x",encoding="utf-8") as f:
  json.dump(document,f,sort_keys=True,indent=2,allow_nan=False)
  f.write("\n")

def main(a):
 gate(sys.flags.isolated==1 and sys.flags.no_user_site==1,
      "REQUIRE_PYTHON_I")
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_ONLY")
 source=Path(a.source_dir)
 gate(source==Path.cwd() and
      source==Path("/egr/research-pac/huang248/data/imagenet/val_raw")
      and source.is_dir() and not source.is_symlink(),
      "ONLY_VALIDATION_RAW_CWD")
 manifest=Path(a.source_manifest)
 gate(manifest.is_file() and not manifest.is_symlink()
      and sha(manifest)==a.source_manifest_sha,
      "PINNED_SOURCE_MANIFEST_SHA")
 doc=json.loads(manifest.read_text())
 gate(doc["status"]=="XP_N_GO_RAW_IMAGE_SOURCE_AUDITED_NO_GT_Y_YET"
      and doc["n_images"]==100 and doc["cheng_go"] is True
      and doc["D1_nonoverlap_proven"] is True
      and doc["planner_XPN_forecast_sha256"]==
          "9fb8abc26b930cd4f8b5aa0598169732f7ae2e931b3048bd16f1940e357555dd",
      "SOURCE_MANIFEST_RECEIPT")
 rows=doc["source_rows"]
 gate(len(rows)==100 and
      [r["basename"] for r in rows]==[
       f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(48900,49000)],
      "EXPECTED_EXACT_100")
 out=Path(a.out)
 gate(out.parent==Path("/egr/research-pac/huang248/outputs/pr_diffusion/track_r")
      and out.name.startswith("xp_a18_xpn_measurements_")
      and not out.exists() and not out.is_symlink(),
      "OUTPUT_PATH")
 wt=Path(a.repo)
 gate(wt.is_dir() and not wt.is_symlink()
      and (wt/"prdiffusion/track_r/physics.py").is_file(),
      "PINNED_PROJECT_IMPORT")
 # -I ignores PYTHONPATH and removes cwd from sys.path. Add only the
 # validated repository root for its frozen physics functions.
 sys.path.insert(0,str(wt))
 import torch
 import torchvision
 from PIL import Image
 from torchvision.transforms import Compose,ToTensor,Resize,CenterCrop
 from prdiffusion.track_r.physics import magnitude,rho_signed
 torch.set_num_threads(4)
 transform=Compose([ToTensor(),Resize(256),CenterCrop(256)])
 out.mkdir(mode=0o700)
 yd=out/"measurements";yd.mkdir(mode=0o700)
 gd=out/"gt_only";gd.mkdir(mode=0o700)
 ym=[];gm=[];worst=0.
 with torch.inference_mode():
  for j,r in enumerate(rows):
   name=r["basename"];p=source/name
   gate(str(p)==r["source_path"] and not p.is_symlink()
        and p.is_file() and sha(p)==r["source_sha256"],
        "SOURCE_FILE_SHA_"+name)
   with Image.open(p) as im:
    gate(im.format=="JPEG","IMAGE_JPEG_FORMAT_"+name)
    x=transform(im)
   if tuple(x.shape)==(1,256,256):x=x.repeat(3,1,1)
   gate(tuple(x.shape)==(3,256,256)
        and bool(torch.isfinite(x).all()),"GT_SHAPE_"+name)
   x=(x*2-1).unsqueeze(0).float().contiguous()
   canonical_sha=hashlib.sha256(
       x.numpy().tobytes(order="C")).hexdigest()
   pure=magnitude(x)
   mseed=seed("XPMEAS|imagenet_n",name)
   generator=torch.Generator(device="cpu").manual_seed(mseed)
   noise=torch.randn(pure.shape,generator=generator,dtype=torch.float32)
   y=(pure+0.05*noise).contiguous()
   rms=float(rho_signed(x,y)[0])
   z=float(noise.double().square().mean().sqrt())
   gate(tuple(y.shape)==(1,3,384,384)
        and bool(torch.isfinite(y).all())
        and .0490<rms<.0510 and abs(z-1)<.0043,
        "SIGNED_Y_NOISE_PHYSICS_"+name)
   gt=gd/f"image_{j:03d}.pt"
   yf=yd/f"image_{j:03d}.pt"
   torch.save({"gt_model_range":x},gt)
   torch.save({"measurement":y},yf)
   gtr=torch.load(gt,map_location="cpu",weights_only=True)["gt_model_range"]
   yr=torch.load(yf,map_location="cpu",weights_only=True)["measurement"]
   gate(torch.equal(gtr,x) and torch.equal(yr,y)
        and sha(p)==r["source_sha256"],
        "ROUNDTRIP_OR_SOURCE_CHANGED_"+name)
   err=float((yr-(magnitude(gtr)+0.05*torch.randn(
         pure.shape,generator=torch.Generator(device="cpu").manual_seed(mseed),
         dtype=torch.float32))).double().square().mean().sqrt())
   gate(math.isfinite(err) and err<=1e-4,
        "GT_Y_EXACT_RECONSTITUTION_"+name)
   worst=max(worst,err)
   ym.append(dict(basename=name,image_id=r["image_id"],
     source_image_sha256=r["source_sha256"],
     measurement_path=str(yf),measurement_sha256=sha(yf),
     seed=mseed,sigma=.05))
   gm.append(dict(basename=name,image_id=r["image_id"],
     source_path=str(p),source_sha256=r["source_sha256"],
     gt_path=str(gt),gt_file_sha256=sha(gt),
     gt_tensor_sha256=canonical_sha,
     gt_tensor_sha256_encoding="float32_BCHW_C_order_in_model_range_-1_to_1",
     gt_tensor_shape=[1,3,256,256],
     measurement_sha256=sha(yf),seed=mseed,sigma=.05,
     signed_y_noise_rms=rms,normalized_noise_rms=z,
     identity_rms=err))
   if (j+1)%25==0:
    print("A18_XPN_ISOLATED_GT_Y",j+1,"/100",flush=True)
 gate(len(ym)==len(gm)==100,"EXACT_100")
 write(out/"XP_N_IMAGENET100_Y_MANIFEST.json",dict(
    status="XP_N_IMAGENET100_SIGNED_Y_FROZEN",
    schema="prdiff.track_r.xp_n.imagenet.y.v1",
    expected_head=a.expected_head,n_images=100,
    dataset="imagenet",measurement_source="XP_IMAGENET_N100",
    source_manifest_sha256=sha(manifest),gt_paths_supplied=False,
    model_range="[-1,1]",sigma=.05,
    preprocessor="D1 ToTensor Resize(256) CenterCrop(256)",
    operator="prdiffusion.track_r.physics.magnitude centered pad64 fft",
    noise_seed_rule="SHA256('XPMEAS|imagenet_n|<basename>')[:8] big-endian mod 2**63",
    noise_generator="cpu torch.Generator torch.randn float32",
    rows=ym))
 write(out/"XP_N_GT_ONLY_CPU.json",dict(
    status="XP_N_GT_ONLY_OFFLINE_NOT_FOR_GENERATION",
    expected_head=a.expected_head,n_images=100,
    worst_identity_rms=worst,all_identity_rms_le_1e_minus_4=True,
    torch=torch.__version__,torchvision=torchvision.__version__,
    isolated_python_flag=True,source_directory=str(source),
    rows=gm))
 print("A18_XPN_ISOLATED_GT_Y_ALL100_PASS",
       "WORST_IDENTITY_RMS",worst,flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--source-dir",required=True)
 p.add_argument("--source-manifest",required=True)
 p.add_argument("--source-manifest-sha",required=True)
 p.add_argument("--repo",required=True)
 p.add_argument("--out",required=True)
 p.add_argument("--expected-head",required=True)
 main(p.parse_args())
