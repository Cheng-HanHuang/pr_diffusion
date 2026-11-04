#!/usr/bin/env python3
"""CPU-only publish exact A22 Table-3 E1 metrics and frozen source receipts."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
WT=Path(__file__).resolve().parents[2]
SRC="docs/track_r/executors/xp/theory_inputs/XP_A22_FROZEN_N2_FACE_PROMPTS.json"

def gate(c,k):
 if not c:raise RuntimeError("STOP_A22_TAB3_PUBLISH_"+k)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def git(p,*args):return subprocess.check_output(
 ["git","-C",str(p),*args],text=True,timeout=60).strip()
def main(a):
 gate(os.environ.get("CUDA_VISIBLE_DEVICES")==""
      and git(WT,"rev-parse","HEAD")==a.expected_head
      and not git(WT,"status","--porcelain"),"CLEAN_CPU_CODE")
 root=Path(a.run)
 gate(root.parent==BASE and root.name.startswith("xp_a30_a22_tab3_")
      and root.is_dir(),"INPUT_NAMESPACE")
 rec=root/"XP_A22_TAB3_RECEIPT.json"
 r=json.loads(rec.read_text())
 gate(r["status"]=="A22_TABLE3_E1_IMAGE_NET_CPU_PASS"
      and r["implementation_head"]==a.expected_head
      and r["n_policy_rows"]==300 and r["n_tables"]==3,"RECEIPT")
 agg=root/"XP_A22_TAB3_IMAGENET_E1_G1.csv"
 im=root/"XP_A22_TAB3_IMAGE_E1_G1.csv"
 gate(sha(agg)==r["table_sha256"]
      and sha(im)==r["local_image_detail_sha256"],
      "DATA_SHA")
 with agg.open(newline="") as f:rows=list(csv.DictReader(f))
 gate(len(rows)==3 and
      {(r["panel"],int(r["K"])) for r in rows}==
      {("D2",4),("D2",8),("D3",4)}
      and all(int(r["n_images"])==100 for r in rows),"TABLE3_ROWS")
 with im.open(newline="") as f:detail=list(csv.DictReader(f))
 gate(len(detail)==300 and
      len({(v["image_id"],v["panel"],int(v["K"])) for v in detail})==300,
      "PER_IMAGE_ROWS")
 tag=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
 branch=f"codex/prdiff-track-r-a22-tab3-{tag}"
 pub=ROOT/f"pr_diffusion_track_r_a22_tab3_pub_{tag}"
 gate(not pub.exists(),"FRESH_PUBLICATION_TREE")
 subprocess.run(["git","-C",str(WT),"worktree","add","-b",
                 branch,str(pub),a.expected_head],check=True)
 ret=pub/"docs/track_r/executors/xp/returns"/f"XP_A22_TAB3_{tag}"
 theory=pub/"docs/track_r/executors/xp/theory_inputs"
 ret.mkdir();theory.mkdir(exist_ok=True)
 for p in (agg,im,rec):shutil.copyfile(p,ret/p.name)
 shutil.copyfile(agg,theory/agg.name)
 f=ret/"XP_A22_TAB3_PUBLICATION.json"
 f.write_text(json.dumps(dict(
  status="A22_TAB3_E1_G1_PUBLISHED",
  source_table_sha256=sha(agg),
  source_image_evaluation_sha256=sha(im),
  source_CPU_receipt_sha256=sha(rec),
  source_N2_frozen_prompts_sha256=sha(WT/SRC),
  n_images=100,
  panels=[("D2",4),("D2",8),("D3",4)],
  exact_frozen_G1=True,
  no_new_solver_GPU=True,
  no_matched_compute_claim=True,
  six_orbit_data_unmodified=True,
  originating_code_head=a.expected_head
 ),indent=2,sort_keys=True)+"\n")
 for p in (ret/agg.name,theory/agg.name):
  gate(sha(p)==sha(agg),"POST_COPY_TABLE")
 subprocess.run(["git","-C",str(pub),"add",
                 str(ret.relative_to(pub)),
                 str((theory/agg.name).relative_to(pub))],check=True)
 subprocess.run(["git","-C",str(pub),"commit","-m",
                 "results(track-r): A22 ImageNet Table3 E1 G1 K4/K8 native SSIM VGG-LPIPS"],check=True)
 subprocess.run(["git","-C",str(pub),"push","origin",
                 "HEAD:refs/heads/"+branch],check=True,timeout=900)
 print("A22_TAB3_PUBLISHED",branch,git(pub,"rev-parse","HEAD"),
       "TABLE3_SHA",sha(agg),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--expected-head",required=True)
 p.add_argument("--run",required=True)
 try:main(p.parse_args())
 except Exception as e:
  traceback.print_exc()
  print("ACTION NEEDED: A22_TAB3_PUBLISH",str(e),flush=True)
  sys.exit(8)
