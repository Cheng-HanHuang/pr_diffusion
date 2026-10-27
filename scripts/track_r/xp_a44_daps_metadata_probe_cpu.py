#!/usr/bin/env python3
"""A4.4 read-only bounded historical DAPS config-provenance probe.

Reads only named run metadata/configs under the three already-known historical
exposed FFHQ sources. No GT, candidate tensors, image tree traversal, or GPU.
Reports evidence fields and status; NEVER assumes missing settings equal DAPS-4K.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
SOURCES={
 "S1_b22": ROOT/"outputs/pr_diffusion/b22_baselines/B22_2_overnight_20260727_155135",
 "S2_b24_dev80": ROOT/"outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z",
 "S3_a12": ROOT/"outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_k4_v1",
}
# Direct paths: no recursive repository/data traversal and no fallback to
# source-directory search outside these fixed exposed experiment roots.
CANDIDATES={
 "S1_b22": [
   "full/manifest.json","full/config.json","full/run_config.json",
   "full/panel_config.json","manifest.json","config.json",
   "full/DAPS4/manifest.json","full/daps/manifest.json"],
 "S2_b24_dev80": [
   "manifest.json","run_config.json","config.json",
   "B24_3_COMPLETE.json","full/manifest.json",
   "analysis/manifest.json"],
 "S3_a12": [
   "manifest.json","config.json","A1_2_COMPLETE.json",
   "A1_2_CONFIG.json","analysis/manifest.json"]
}
FIELDS=("annealing_scheduler_config","diffusion_scheduler_config",
        "mcmc_sampler_config","num_steps","annealing_steps","ode_steps",
        "diffusion_steps","langevin_steps","NFE","nfe",
        "num_diffusion_steps","num_annealing_steps","method",
        "sampler","daps","DAPS","config","sample_config")
def valid(path):
    p=Path(path)
    if (p.is_symlink() or any(x in str(p.resolve(strict=False)).lower() for x in BAD)
        or ROOT.resolve() not in p.resolve(strict=False).parents):
        raise RuntimeError("STOP_A44_PROTECTED_PATH")
    return p

def targeted(obj,depth=0):
    if depth>5:return None
    if isinstance(obj,dict):
        out={}
        for k,v in obj.items():
            if str(k) in FIELDS or ("daps" in str(k).lower() and len(str(k))<80):
                if isinstance(v,(str,int,float,bool)) or v is None:
                    out[k]=v
                elif isinstance(v,dict):
                    # suppress paths and task/measurement metadata
                    out[k]=targeted(v,depth+1)
                else:out[k]={"container_type":type(v).__name__,"length":len(v)}
            elif isinstance(v,dict):
                child=targeted(v,depth+1)
                if child:out[k]=child
        return out
    return None

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A44_CPU_ONLY")
    head=subprocess.check_output(
      ["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    status=subprocess.check_output(
      ["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if head!=a.expected_head or status:
        raise RuntimeError("STOP_A44_GIT")
    evidence={}
    for label,root in SOURCES.items():
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("STOP_A44_SOURCE_ROOT_NOT_FOUND_"+label)
        rows=[]
        for relative in CANDIDATES[label]:
            p=valid(root/relative)
            if not p.is_file():continue
            if p.stat().st_size>10_000_000:
                rows.append({"relative":relative,"status":"TOO_LARGE_TO_INSPECT"})
                continue
            raw=p.read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            try:
                doc=json.loads(raw)
                extract=targeted(doc)
                rows.append({"relative":relative,"sha256":digest,
                             "top_level_keys":sorted(doc)[:50] if isinstance(doc,dict) else [],
                             "targeted_fields":extract,
                             "status":"INSPECTED_METADATA_ONLY"})
            except (ValueError,UnicodeError):
                rows.append({"relative":relative,"sha256":digest,
                             "status":"NOT_JSON"})
        evidence[label]={"root":str(root),"metadata_files":rows,
                         "n_metadata_inspected":len(rows)}
    out=Path(a.out)
    if (out.parent!=BASE or not out.name.startswith("r0n2_a44_")
        or out.exists() or out.is_symlink()):
        raise RuntimeError("STOP_A44_OUTPUT")
    out.mkdir(exist_ok=False)
    result={"status":"A4_4_METADATA_PROBE_REQUIRES_INTERPRETATION",
       "head":head,"historical_sources":evidence,
       "sampler_configuration_confirmed":False,
       "NFE_estimated":False,"gt_read":False,"gpu_used":False}
    with (out/"A4_4_METADATA_PROBE.json").open("x") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    print("A4_4_METADATA_PROBE_DONE")
    for name,d in evidence.items():
        print(name,"metadata_files",d["n_metadata_inspected"])
        for entry in d["metadata_files"]:
            print(" ",entry["relative"],entry["status"],
                  "keys",",".join(entry.get("top_level_keys",[])[:12]))
            print("  config_fields=",json.dumps(entry.get("targeted_fields",{}),
                                  sort_keys=True)[:2000])
    print("A44_PROBE_FILE="+str(out/"A4_4_METADATA_PROBE.json"))
    print("A44_CONFIG_NOT_YET_CERTIFIED")
    print("OK")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    try:main(p.parse_args())
    except Exception as exc:
        import traceback;traceback.print_exc()
        print("ACTION NEEDED: A4_4_PROBE "+str(exc),flush=True)
        sys.exit(8)
