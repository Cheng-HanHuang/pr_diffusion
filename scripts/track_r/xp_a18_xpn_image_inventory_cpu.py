#!/usr/bin/env python3
"""A16.5 XP-N GO: CPU-only source-file audit before decoding any JPEGs.

No GT tensor is decoded here. This stage checks Cheng's new cohort,
exact validation ID disjointness against the SHA-pinned D1 manifest,
and SHA256 for all raw source files under /data/imagenet. The next
stage MUST decode untrusted JPEGs under python -I from their directory.
Never use FFHQ, a protected split, a bundled dataset, or downloads.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r.xp_a12_g1a_prepare_cpu import ROOT,BASE,gate
from scripts.track_r.xp_a6_d1p2_contract import D1P2,SHA as D1_SHA
from scripts.track_r.xp_a15_cp2_d4_cpu import write

EXPECTED={f"ILSVRC2012_val_{i:08d}.JPEG":f"{i:08d}"
          for i in range(48900,49000)}
D1_EXPECTED={f"ILSVRC2012_val_{i:08d}.JPEG"
             for i in range(49000,49100)}
EXCLUDED=("confirmation","confirm305","test-ffhq","demo-ffhq")

def sha_file(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(2**20),b""):h.update(block)
 return h.hexdigest()

def source_rows(source):
 root=ROOT/"data"/"imagenet"
 source=Path(source)
 gate(source.is_dir() and not source.is_symlink()
      and (source==root or root in source.parents)
      and root.resolve() in source.resolve().parents+[source.resolve()]
      if False else source.is_dir() and not source.is_symlink() and
      (source==root or root in source.parents) and
      (source.resolve()==root.resolve() or
       root.resolve() in source.resolve().parents),
      "A18_XPN_SOURCE_WITHIN_IMAGENET")
 gate(not any(b in str(source).lower() for b in EXCLUDED),
      "A18_XPN_NO_PROTECTED_PATH")
 matches={}
 for p in source.rglob("ILSVRC2012_val_*.JPEG"):
  if p.name not in EXPECTED:continue
  gate(not p.is_symlink() and p.is_file()
       and p.resolve().is_relative_to(root.resolve())
       and not any(b in str(p.resolve()).lower() for b in EXCLUDED),
       "A18_XPN_SOURCE_FILE_PATH")
  gate(p.name not in matches,"A18_XPN_AMBIGUOUS_DUPLICATE_"+p.name)
  matches[p.name]=p
 gate(set(matches)==set(EXPECTED),"A18_XPN_EXPECTED_100_JPEGS_MISSING_OR_EXTRA")
 gate(len({p.parent for p in matches.values()})==1,
      "A18_XPN_ONE_VALIDATION_IMAGE_DIRECTORY")
 rows=[]
 for name,iid in sorted(EXPECTED.items()):
  p=matches[name];size=p.stat().st_size
  gate(1024<=size<=50*1024*1024,"A18_XPN_IMAGE_FILE_SIZE_"+iid)
  rows.append(dict(image_id=iid,basename=name,
                   source_path=str(p),source_sha256=sha_file(p),
                   source_bytes=size))
 return rows

def check_d1():
 gate(D1P2.is_file() and not D1P2.is_symlink()
      and sha_file(D1P2)==D1_SHA,"A18_XPN_D1_FROZEN_MANIFEST_SHA")
 d=json.loads(D1P2.read_text())
 gate(d["status"]=="XP_FAST_TASKS_FROZEN"
      and d["stage"]=="D1_P2" and d["n_roots"]==1200
      and d["dataset"]=="imagenet"
      and d["gt_paths_supplied"] is False,
      "A18_XPN_D1_FROZEN_CONTRACT")
 ids={r["image_id"] for r in d["rows"]}
 gate(ids==D1_EXPECTED and len(d["rows"])==1200,
      "A18_XPN_EXACT_D1_HISTORIC_IDS")
 gate(set(EXPECTED).isdisjoint(ids),
      "A18_XPN_MUST_NOT_OVERLAP_D1")
 return D1_SHA

def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A18_XPN_CPU_ONLY")
 gate(a.cheng_go=="GO","A18_XPN_EXPLICIT_GO")
 wt=Path(__file__).resolve().parents[2]
 git=lambda *z:subprocess.check_output(
    ["git","-C",str(wt),*z],text=True,timeout=30).strip()
 gate(git("rev-parse","HEAD")==a.expected_head and
      not git("status","--porcelain"),
      "A18_XPN_PINNED_CLEAN_WORKTREE")
 d1=check_d1()
 rows=source_rows(a.images_dir)
 gate(len(rows)==100 and
      len({r["source_sha256"] for r in rows})==100,
      "A18_XPN_IMAGE_COUNT_SHA_UNIQUE")
 print("A18_XPN_GO_IMAGE_SOURCE_PREFLIGHT_PASS",
       "IMAGES",len(rows),"D1_NONOVERLAP_100",
       "DIR",Path(rows[0]["source_path"]).parent,flush=True)
 if a.preflight_only:return
 out=Path(a.out)
 gate(out.parent==BASE and out.name.startswith("xp_a18_xpn_source_")
      and not out.exists() and not out.is_symlink(),
      "A18_XPN_NEW_SOURCE_MANIFEST")
 out.mkdir(mode=0o700)
 f=out/"XP_N_SOURCE_IMAGES_LOCAL_CPU.json"
 write(f,dict(status="XP_N_GO_RAW_IMAGE_SOURCE_AUDITED_NO_GT_Y_YET",
     expected_head=a.expected_head,cheng_go=True,
     n_images=100,imagenet_2012_validation_range="00048900-00048999",
     D1_nonoverlap_proven=True,D1_manifest_sha256=d1,
     sources_are_untrusted_jpeg=True,
     next_image_decode_requires_python_isolated_flag="-I",
     GT_tensor_SHA256_pending=True,
     GT_scoring_not_performed=True,GPU_generation_not_authorized_by_this_receipt=True,
     source_rows=rows))
 print("A18_XPN_SOURCE_MANIFEST_PASS",f,sha_file(f),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--images-dir",required=True)
 p.add_argument("--cheng-go",choices=("GO",),required=True)
 p.add_argument("--out")
 p.add_argument("--preflight-only",action="store_true")
 try:main(p.parse_args())
 except Exception as ex:
  traceback.print_exc()
  print("ACTION NEEDED: A18_XPN_IMAGE_INVENTORY",ex,flush=True)
  sys.exit(8)
