"""A23 G7 staged GT-only readout and independently published source returns.

One method per result branch. Does NOT claim full G7 before all three verified.
The CPU readout uses fixed image-cluster bootstrap and cannot tune selectors.
"""
from __future__ import annotations
import csv,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from scripts.track_r.xp_a35_g7_worker import (
 ROOT,BASE,WT,GT,LOCK,STATUS,sha,read,gate,git
)

def atomic(path,obj):
    temp=path.with_name("."+path.name+".tmp")
    temp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n")
    temp.replace(path)

def check_results(stage,method,head):
    images=read(GT)["images"];got=[]
    for i,im in enumerate(images):
        p=stage/"results"/f"{i:03d}.json"
        gate(p.is_file() and not p.is_symlink(),"ALL280_FILE_"+str(i))
        r=read(p);score=r["score"]
        gate(r["status"]==STATUS and r["method"]==method
             and r["index"]==i and r["dataset"]==im["dataset"]
             and r["basename"]==im["basename"]
             and r["source_sha256"]==im["source_sha256"]
             and r["implementation_head"]==head
             and r["source_manifest_sha256"]==read(LOCK)["GT_manifest_sha256"]
             and r["source_lock_sha256"]==sha(LOCK)
             and r["diagnostic_GT_only"] is True
             and r["GT_not_used_for_inverse_problem_generation"] is True
             and r["choice"]==score["best_orbit"]
             and r["upright_kept"]==(score["best_orbit"]=="000")
             and (method!="2_forward" or r["n_evals"]==2)
             and (method!="S_mid_all8" or r["n_evals"]==128)
             and (method!="UnivFD" or r["n_evals"]==8),
             "ONE_IMAGE_RECORD_INVALID_"+str(i))
        got.append(r)
    gate(len(got)==280 and
         sum(r["dataset"]=="ffhq" for r in got)==180 and
         sum(r["dataset"]=="imagenet" for r in got)==100,
         "ALL_DATASET_COUNTS")
    return got

def analyze(run,method,head,costs):
    stage=run/method
    got=check_results(stage,method,head)
    readout=stage/"readout"
    gate(not readout.exists(),"ONE_READOUT")
    readout.mkdir()
    per=readout/f"XP_A35_G7_{method}_PER_IMAGE.csv"
    agg=readout/f"XP_A35_G7_{method}_AGGREGATE.csv"
    rows=[]
    for r in got:
        rows.append(dict(method=method,index=r["index"],
              dataset=r["dataset"],basename=r["basename"],
              gt_sha256=r["source_sha256"],uid=r["uid"],
              choice=r["choice"],upright_kept=int(r["upright_kept"]),
              receipt_sha256=sha(stage/"results"/f'{r["index"]:03d}.json')))
    with per.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    aggs=[]
    for dataset in ("ffhq","imagenet","pooled"):
        subset=[r["upright_kept"] for r in rows if
            dataset=="pooled" or r["dataset"]==dataset]
        n=len(subset)
        seed=int.from_bytes(hashlib.sha256(
             f"A23G7A35|{method}|{dataset}".encode()).digest()[:8],"big")
        rng=np.random.default_rng(seed)
        arr=np.asarray(subset,dtype=int)
        vals=rng.choice(arr,(10000,n),replace=True).mean(axis=1)
        low,high=np.quantile(vals,[.025,.975])
        aggs.append(dict(method=method,dataset=dataset,n=n,
            upright_kept=sum(subset),n_flipped=n-sum(subset),
            keep_fraction=sum(subset)/n,
            ci95_low=float(low),ci95_high=float(high),
            n_bootstrap=10000,
            GT_only=True))
    with agg.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(aggs[0]))
        w.writeheader();w.writerows(aggs)
    receipt=dict(status="A35_G7_METHOD_READOUT_280_PASS",method=method,
        head=head,source_manifest_sha256=read(LOCK)["GT_manifest_sha256"],
        source_lock_sha256=sha(LOCK),per_image_sha256=sha(per),
        aggregate_sha256=sha(agg),aggregates=aggs,
        project_gpu_h=costs["project_gpu_h"],
        G7_total_gpu_h=costs["G7_total_gpu_h"],
        GT_diagnostic_only=True,not_full_G7=True,
        future_methods_unrun=[
            m for m in ("S_mid_all8","2_forward","UnivFD")
            if ("S_mid_all8","2_forward","UnivFD").index(m)>
               ("S_mid_all8","2_forward","UnivFD").index(method)],
        no_new_main_text=True,no_new_selector=True)
    atomic(readout/"XP_A35_G7_METHOD_READOUT_RECEIPT.json",receipt)
    print("A35_G7_READOUT_PASS",method,
          [(x["dataset"],x["upright_kept"],x["n"]) for x in aggs],flush=True)
    return receipt

def gitrun(root,*args,timeout=120):
    return subprocess.check_output(["git","-C",str(root),*args],
       text=True,timeout=timeout).strip()

def published_commit(root,branch):
    b=gitrun(root,"ls-remote","--heads","origin","refs/heads/"+branch,timeout=90)
    return None if not b else b.split()[0]

def publish(run,method,head,receipt):
    stage=run/method
    content=stage/"readout"
    per=content/f"XP_A35_G7_{method}_PER_IMAGE.csv"
    agg=content/f"XP_A35_G7_{method}_AGGREGATE.csv"
    gate(sha(per)==receipt["per_image_sha256"]
         and sha(agg)==receipt["aggregate_sha256"],
         "PUBLISH_ORIGINAL_SHA")
    tag=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    branch=f"codex/prdiff-track-r-a23-g7-{method.lower().replace('_','-')}-{tag}"
    tree=ROOT/f"pr_diffusion_track_r_a35_pub_{method}_{tag}"
    gate(not tree.exists(),"PUBLISH_FRESH_TREE")
    subprocess.run(["git","-C",str(WT),"worktree","add","-b",
        branch,str(tree),head],check=True,timeout=90)
    out=tree/"docs/track_r/executors/xp/returns"/f"XP_A35_G7_{method}_{tag}"
    ti=tree/"docs/track_r/executors/xp/theory_inputs"
    out.mkdir()
    for f in (per,agg,content/"XP_A35_G7_METHOD_READOUT_RECEIPT.json",LOCK,
              stage/"STAGE_TERMINAL.json"):
        shutil.copyfile(f,out/f.name)
    for f in (per,agg):
        shutil.copyfile(f,ti/f.name)
    p=dict(status="A35_G7_STAGE_280_PUBLISHED",method=method,
           original_scoring_head=head,source_gt_sha256=read(LOCK)["GT_manifest_sha256"],
           no_full_G7_claim=method!="UnivFD",
           n_images=280,GT_only_diagnostic=True,
           table_sha256=sha(agg),per_image_sha256=sha(per),
           method_readout_receipt_sha256=sha(content/"XP_A35_G7_METHOD_READOUT_RECEIPT.json"),
           stage_terminal_sha256=sha(stage/"STAGE_TERMINAL.json"),
           G7_total_gpu_h=receipt["G7_total_gpu_h"],
           project_gpu_h=receipt["project_gpu_h"],
           no_main_text_new_data=True,not_matched_compute=True)
    atomic(out/"XP_A35_G7_PUBLICATION.json",p)
    subprocess.run(["git","-C",str(tree),"add",
          str(out.relative_to(tree)),
          str((ti/per.name).relative_to(tree)),
          str((ti/agg.name).relative_to(tree))],check=True,timeout=60)
    subprocess.run(["git","-C",str(tree),"commit","-m",
          f"results(track-r): A23 GT-only G7 {method} 280 upright diagnostics"],
          check=True,timeout=60)
    commit=gitrun(tree,"rev-parse","HEAD")
    print("A35_PUBLISH_LOCAL_EXISTING_COMMIT",method,branch,commit,tree,flush=True)
    # If a push times out, retry the SAME exact commit; never re-run science.
    for attempt in (1,2):
        old=published_commit(tree,branch)
        if old==commit:break
        gate(old is None,"PUBLISH_REMOTE_DIFFERENT")
        cmd=["git","-C",str(tree)]
        if attempt==2:cmd+=["-c","http.version=HTTP/1.1"]
        cmd+=["push","origin","HEAD:refs/heads/"+branch]
        try:
            r=subprocess.run(cmd,timeout=300,capture_output=True,text=True,
              env=dict(os.environ,GIT_TERMINAL_PROMPT="0",GCM_INTERACTIVE="never"))
            print("A35_PUBLISH_ATTEMPT",method,attempt,r.returncode,
                  r.stderr[-500:],flush=True)
        except subprocess.TimeoutExpired:
            print("A35_PUBLISH_TIMEOUT_KEEP_EXISTING_COMMIT",attempt,flush=True)
        if published_commit(tree,branch)==commit:break
    gate(published_commit(tree,branch)==commit,
         "PUBLISH_STALLED_EXACT_COMMIT_PRESERVED")
    print("A35_G7_STAGE_REMOTE_PUBLISHED",method,branch,commit,flush=True)
    return dict(branch=branch,commit=commit)
