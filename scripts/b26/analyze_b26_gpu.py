#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np

def readj(p): return json.loads(Path(p).read_text())
def writej(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+f".tmp.{os.getpid()}"); t.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(t,p)
def seed63(name): return int(hashlib.sha256(("B26_BOOTSTRAP_V1|"+name).encode()).hexdigest()[:16],16)&((1<<63)-1)
def q(x,p): return float(np.quantile(np.asarray(x,float),p))
def stat(vals):
    a=np.asarray(vals,float); return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"q10":q(a,.1),"min":float(a.min()),"max":float(a.max()),"good25":int((a>=25).sum()),"bad20":int((a<20).sum())}
def bootstrap(rows):
    # rows carry diff and original frozen stratum; resample within strata, 10k replicates.
    rr=np.random.Generator(np.random.PCG64(seed63("H_R_PRIMARY_MEAN_DIFF"))); reps=[]
    groups={c:[r for r in rows if r["screening_stratum"]==c] for c in "ABCD"}
    if any(not g for g in groups.values()): return None
    for _ in range(10000):
        samp=[]
        for c in "ABCD":
            g=groups[c]; idx=rr.integers(len(g),size=len(g)); samp.extend(g[i]["R_minus_H_psnr_db"] for i in idx)
        reps.append(float(np.mean(samp)))
    return {"replicates":10000,"seed_domain":"B26_BOOTSTRAP_V1","estimate_mean_diff_db":float(np.mean([r["R_minus_H_psnr_db"] for r in rows])),"percentile_95_ci_db":[q(reps,.025),q(reps,.975)],"exploratory_dev_only":True}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--gpu-root",type=Path,required=True); ap.add_argument("--scope",choices=("dev16","dev80"),required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    m=readj(a.manifest); rows=[r for r in m["rows"] if a.scope=="dev80" or r["dev16"]]
    image_rows=[]; root_rows=[]
    for r in rows:
        arms={}
        for arm in ("H","R"):
            vals=[]
            for ri in range(4):
                d=a.gpu_root/"jobs"/r["image_id"]/f"root{ri}"/arm/"JOB_COMPLETE.json"
                if not d.is_file(): raise RuntimeError(f"missing {d}")
                c=readj(d); v=readj(Path(c["result"])); vals.append(v); root_rows.append(v)
            arms[arm]=vals
        def sel(vals,key): return min(range(4),key=lambda j:(float(vals[j]["accounting"][key]),j))
        def oracle(vals): return max(range(4),key=lambda j:(float(vals[j]["psnr_raw_db"]),-j))
        hp=sel(arms["H"],"selector_plus_mean"); hr=sel(arms["H"],"selector_raw_mean")
        rp=sel(arms["R"],"selector_plus_mean"); rr=sel(arms["R"],"selector_raw_mean")
        ho=oracle(arms["H"]); ro=oracle(arms["R"])
        row={"image_id":r["image_id"],"screening_stratum":r["screening_stratum"],"shared_failure_10":r["shared_failure_10"],
             "H_plus_index":hp,"H_raw_index":hr,"R_plus_index":rp,"R_raw_index":rr,
             "H_primary_psnr_db":arms["H"][hp]["psnr_raw_db"],"R_primary_psnr_db":arms["R"][rr]["psnr_raw_db"],
             "H_offdiag_raw_psnr_db":arms["H"][hr]["psnr_raw_db"],"R_offdiag_plus_psnr_db":arms["R"][rp]["psnr_raw_db"],
             "H_oracle_psnr_db":arms["H"][ho]["psnr_raw_db"],"R_oracle_psnr_db":arms["R"][ro]["psnr_raw_db"],
             "H_terminal_selector_disagree":hp!=hr,"R_terminal_selector_disagree":rp!=rr,
             "R_minus_H_psnr_db":float(arms["R"][rr]["psnr_raw_db"]-arms["H"][hp]["psnr_raw_db"]),
             "H_primary_good25":arms["H"][hp]["psnr_raw_db"]>=25,"R_primary_good25":arms["R"][rr]["psnr_raw_db"]>=25,
             "fresh_daps4_best_psnr_raw_rgb_db":r.get("fresh_daps4_best_psnr_raw_rgb_db"),"fresh_sitcom4_best_psnr_raw_rgb_db":r.get("fresh_sitcom4_best_psnr_raw_rgb_db"),"fresh2":"UNAVAILABLE_IN_B26_MANIFEST_UNLESS_SEPARATELY_PROVEN_IDENTITY_MATCH"}
        image_rows.append(row)
    def group_summary(g):
        h=[x["H_primary_psnr_db"] for x in g]; r=[x["R_primary_psnr_db"] for x in g]; d=[x["R_minus_H_psnr_db"] for x in g]
        return {"images":len(g),"H_primary":stat(h),"R_primary":stat(r),"R_minus_H":{"mean":float(np.mean(d)),"median":float(np.median(d)),"q10":q(d,.1),"min":min(d),"max":max(d)},"good25_rescues":sum((not x["H_primary_good25"]) and x["R_primary_good25"] for x in g),"good25_harms":sum(x["H_primary_good25"] and (not x["R_primary_good25"]) for x in g),"H_oracle":stat([x["H_oracle_psnr_db"] for x in g]),"R_oracle":stat([x["R_oracle_psnr_db"] for x in g]),"H_terminal_selector_disagreements":sum(x["H_terminal_selector_disagree"] for x in g),"R_terminal_selector_disagreements":sum(x["R_terminal_selector_disagree"] for x in g)}
    groups={"all":group_summary(image_rows),**{f"stratum_{c}":group_summary([x for x in image_rows if x["screening_stratum"]==c]) for c in "ABCD"}}
    shared=[x for x in image_rows if x["shared_failure_10"]]
    if shared: groups["frozen_shared_failure_10"]=group_summary(shared)
    maxmem=max(int(v.get("gpu_monitor",{}).get("max_b24_process_gpu_mib") or 0) for v in root_rows)
    calls=sum(int(v["accounting"]["total_unet_evals"]) for v in root_rows); scorefft=sum(int(v["accounting"]["score_fft_evals"]) for v in root_rows); diagfft=sum(int(v["accounting"]["diagnostic_fft_evals"]) for v in root_rows)
    wall=sum(float(v["wall_seconds"]) for v in root_rows)
    receipts=[]
    for p in sorted((a.gpu_root/"receipts").glob("WORKER_*.json"))+sorted(a.gpu_root.glob("WORKER_*.json")):
        try: receipts.append(readj(p))
        except Exception: pass
    reservation=sum(float(v.get("reservation_wall_seconds",0)) for v in receipts if v.get("status")=="PASS")
    winner={arm:{"preprojection_disagreement_steps":sum(int(v["accounting"]["winner_disagreement_preprojection"]) for v in root_rows if v["arm"]==arm),"trajectory_count":sum(v["arm"]==arm for v in root_rows)} for arm in ("H","R")}
    payload={"schema_version":"b26.gpu-analysis.v1","status":"PASS","scope":a.scope,"image_count":len(image_rows),"root_trajectory_count":len(root_rows),"groups":groups,"paired_stratified_bootstrap":bootstrap(image_rows),"image_rows":image_rows,"candidate_winner_diagnostics":winner,"compute":{"total_unet_evals":calls,"historical_np1_work_fre":calls/2200.,"score_fft_evals":scorefft,"diagnostic_fft_evals":diagfft,"sum_synchronized_job_wall_seconds":wall,"gpu_active_time_note":"Exact kernel-active time was not separately instrumented; synchronized per-root elapsed is reported as a conservative GPU-stage elapsed proxy.","aggregate_worker_reservation_seconds":reservation,"aggregate_worker_reservation_gpu_hours":reservation/3600.,"max_process_gpu_mib":maxmem},"confirmation_payloads_accessed":False,"new_ffhq_measurements_generated":False,"population_inference_authorized":False}
    writej(a.output,payload); print(json.dumps({"status":"PASS","scope":a.scope,"images":len(image_rows),"H_good25":groups["all"]["H_primary"]["good25"],"R_good25":groups["all"]["R_primary"]["good25"],"mean_diff":groups["all"]["R_minus_H"]["mean"]},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
