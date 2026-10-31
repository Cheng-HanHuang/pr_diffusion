"""A16.5 XP-N generation-only immutable input contract for native FAST-DIPS.

Does not open GT, candidates, labels, v5 forecasts, or scoring files.
The independent prospective population (validation 48900-48999) uses
D1's unchanged FAST configuration and signed-y operator. Three distinct
run phases enable unconditional 0..3 publication and 4..63 embargo.
"""
from __future__ import annotations
import hashlib,json
from collections import defaultdict
from pathlib import Path
from scripts.track_r.xp_t_imagenet_prepare_cpu import seed as native_seed

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"
FROZEN_Y=BASE/"xp_a18_xpn_measurements_20261031T222303Z"/"XP_N_IMAGENET100_Y_MANIFEST.json"
Y_SHA="742f7397d4377166ee86866f60269407d4b69cbd350766bd1aa15f11c9e6dfe3"
SOURCE_SHA="5fa348229a35899f39aa06326840e9a8592eea6e882450bbbd0068895cee371b"
AUDIT_SHA="ab7cf74b0bacd4f9944e9b5a22078f596641b8c2384ec7c26e306f0931ba4d98"
MODEL_SHA="a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"
FAST_HEAD="a9758b38906b7c7950fa04491f37f6634948db26"
PHASES={
 "A18_XPN_PILOT_0_3":(0,4,400),
 "A18_XPN_MAIN_4_15":(4,16,1200),
 "A18_XPN_LATE_16_63":(16,64,4800)
}
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")

def validate_xpn_source(path,digest,manifest,rows,producer,start,count):
 path=Path(path)
 if (path.is_symlink() or not path.is_file()
     or path.parent.parent!=BASE
     or not path.parent.name.startswith("xp_a18_xpn_tasks_")
     or path.name not in (
        "XP_N_PILOT_0_3_TASKS.json",
        "XP_N_MAIN_4_15_TASKS.json",
        "XP_N_LATE_16_63_TASKS.json")):
  raise RuntimeError("STOP_A18_XPN_MANIFEST_NAMESPACE")
 expected_file={
  "A18_XPN_PILOT_0_3":"XP_N_PILOT_0_3_TASKS.json",
  "A18_XPN_MAIN_4_15":"XP_N_MAIN_4_15_TASKS.json",
  "A18_XPN_LATE_16_63":"XP_N_LATE_16_63_TASKS.json"}
 phase=manifest.get("stage")
 if phase not in PHASES or path.name!=expected_file[phase]:
  raise RuntimeError("STOP_A18_XPN_STAGE_MAPPING")
 receiptfile=path.parent/"XP_N_TASKS_PREP_RECEIPT.json"
 if not receiptfile.is_file() or receiptfile.is_symlink():
  raise RuntimeError("STOP_A18_XPN_PREP_RECEIPT_MISSING")
 rec=json.loads(receiptfile.read_text())
 if (rec.get("status")!="A18_XPN_GT_FREE_6400_MANIFESTS_FROZEN_PASS"
     or rec.get("expected_head")!=producer
     or rec.get("source_y_manifest_sha256")!=Y_SHA
     or rec.get("source_gt_y_audit_sha256")!=AUDIT_SHA
     or rec.get("native_d1_replay_pixel_exact") is not True
     or rec.get("phase_files",{}).get(phase,{}).get("sha256")!=digest
     or hashlib.sha256(path.read_bytes()).hexdigest()!=digest):
  raise RuntimeError("STOP_A18_XPN_STAGE_MANIFEST_FROZEN_SHA")
 lo,hi,total=PHASES[phase]
 if (manifest.get("schema")!="prdiff.track_r.xp.fast-tasks.v1"
     or manifest.get("status")!="XP_FAST_TASKS_FROZEN"
     or manifest.get("head")!=producer
     or manifest.get("fast_head")!=FAST_HEAD
     or manifest.get("dataset")!="imagenet"
     or manifest.get("oversample")!=2.0
     or manifest.get("sigma")!=.05
     or manifest.get("model_sha256")!=MODEL_SHA
     or manifest.get("n_images")!=100
     or manifest.get("n_roots")!=total
     or manifest.get("gt_paths_supplied") is not False
     or manifest.get("outcome_embargo") is not True
     or manifest.get("new_image_cohort")!="ILSVRC2012_val_00048900-00048999"
     or manifest.get("source_y_manifest_sha256")!=Y_SHA
     or manifest.get("source_image_inventory_sha256")!=SOURCE_SHA
     or manifest.get("source_gt_y_audit_sha256")!=AUDIT_SHA
     or manifest.get("seed_rule")!="XP|FAST|imagenet_n|<basename>|<rep> SHA256 first8 BE mod2**31"
     or manifest.get("rows")!=rows):
  raise RuntimeError("STOP_A18_XPN_STAGE_CONTRACT")
 if len(rows)!=total or (not 0<=start<total
                            or not 1<=count<=120
                            or start% (hi-lo)!=0 or count%(hi-lo)!=0
                            or start+count>total):
  raise RuntimeError("STOP_A18_XPN_IMAGE_ALIGNED_SLICE")
 by=defaultdict(list)
 for r in rows:
  if set(r)!={"uid","source","measurement_source","image_id","rep","seed",
              "measurement_path","measurement_sha256"}:
   raise RuntimeError("STOP_A18_XPN_GT_FREE_FIELDS")
  iid=r["image_id"];rep=r["rep"]
  if (r["uid"]!=f"XP|FAST|imagenet_n|{iid}|rep{rep}"
      or r["source"]!="XP_N" or r["measurement_source"]!="XP_IMAGENET_N100"
      or not iid.startswith("ILSVRC2012_val_") or not iid.endswith(".JPEG")
      or not 48900<=int(iid[15:23])<49000
      or rep not in range(lo,hi)
      or r["seed"]!=native_seed("XP|FAST|imagenet_n",iid,rep,2**31)
      or not str(r["measurement_path"]).startswith(str(BASE/"xp_a18_xpn_measurements_20261031T222303Z/measurements")+"/")
      or any(b in str(r["measurement_path"]).lower() for b in BANNED)
      or not isinstance(r["measurement_sha256"],str)
      or len(r["measurement_sha256"])!=64):
   raise RuntimeError("STOP_A18_XPN_ROOT_UID_NOISE_SEED")
  by[iid].append(rep)
 expected_ids={f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(48900,49000)}
 if set(by)!=expected_ids or any(v!=list(range(lo,hi)) for v in by.values()):
  raise RuntimeError("STOP_A18_XPN_100_IMAGE_PHASE_GRID")
 return True
