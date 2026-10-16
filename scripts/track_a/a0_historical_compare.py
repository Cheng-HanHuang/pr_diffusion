#!/usr/bin/env python3
"""A0 historical metadata extractor/selector comparator (no image/model/operator I/O).

Run ONLY after frozen-source audit passes. Reads explicitly given B21/B22/B24
manifest/CSV/JSON records, never measurements, ground-truth payloads, terminal
tensors/PNGs, logits, models or confirmation305. Writes only to a NEW A0 dir.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from a0_selectors import AuditStop, finite, ladders, paired, summarize

MODEL = "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
B22_SOURCE = "191dbd70894b24f0fafa160c08b17d7ab47a17f1"
B22_DIFF = "a45ffe58f18fed8a63d3446600424e2b08733524"
SITCOM_HEAD = "275ab67efbd8146bffca20155171ba6be1169c09"
DAPS_HEAD = "e7a77d094167084faed19b599b96673b7bb11447"
B24_DIFF = "fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69"
SITCOM_DIFF = "a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e"
REPS = {"B22": "B21_B22_FLOAT_RAW_RGB_V1",
        "B24_EXTERNAL": "CANONICAL_SAVED_RGB_8BIT_RAW_ORIENTATION_V1",
        "B24_NP": "B24_NP_NATIVE_FLOAT_RAW_V1"}

def allowed(p: Path, root: Path | None = None) -> Path:
    """Reject locked, unknown, symlink-escaped paths before any read."""
    p = Path(p).resolve()
    if any(part.lower().startswith("confirm") or part.lower() == "305"
           for part in p.parts):
        raise AuditStop("excluded confirmation path")
    if root is not None and not p.is_relative_to(Path(root).resolve()):
        raise AuditStop(f"record points outside permitted exposed root: {p}")
    if not p.is_file() or p.stat().st_size > 8_000_000:
        raise AuditStop(f"missing/oversized metadata file: {p}")
    if p.suffix not in (".json", ".csv"):
        raise AuditStop("only metadata files permitted")
    return p

def readj(p: Path, root: Path | None = None) -> dict:
    v = json.loads(allowed(p, root).read_text(encoding="utf-8"))
    if not isinstance(v, dict):
        raise AuditStop("JSON object expected")
    return v

def readcsv(p: Path, root: Path | None = None, max_rows: int = 110) -> list[dict]:
    with allowed(p, root).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) > max_rows:
        raise AuditStop("oversized CSV row count")
    return rows

def digest(p: Path) -> str:
    return hashlib.sha256(allowed(p).read_bytes()).hexdigest()

def check_digest(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise AuditStop(f"metadata SHA drift: {path.name}")

def native_sitcom_sha(spec: dict) -> str:
    # SITCOM uses checkpoint/ffhq256.pt, NOT necessarily NP/DAPS ffhq_10m.pt.
    h = spec.get("native_models",{}).get("sitcom_checkpoint_sha256")
    if not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
        raise AuditStop("SITCOM native checkpoint SHA-256 not verified; refuse model identity substitution")
    return h

def image(value: Any) -> str:
    s = str(value).zfill(5)
    if len(s) != 5 or not s.isdecimal():
        raise AuditStop("invalid image id")
    return s

def norm(*, cohort: str, method: str, img: str, measurement: str,
         source: str, rep: str, group: str, index: int, psnr: Any,
         candidate_id: str, selector: Any = None, loss: Any = None,
         wall: Any = None, ssim: Any = None, lpips: Any = None,
         stratum: str = "", native_seed: Any = None,
         model_hash: str = MODEL) -> dict:
    if len(str(measurement)) != 64:
        raise AuditStop("measurement SHA absent")
    row = {"cohort": cohort, "image_id": img, "method": method,
           "group": group, "index": index, "candidate_id": candidate_id,
           "representation": rep, "measurement_sha256": measurement,
           "model_sha256": model_hash, "source_id": source, "psnr": finite(psnr, "psnr"),
           "stratum": stratum, "native_seed": native_seed, "wall_s": wall,
           "ssim": ssim, "lpips": lpips,
           "postproj_lf_mse": selector if method == "NP" else None,
           "step160_correction_norm": selector if method == "SITCOM" else None,
           "daps_signed_loss": loss if method == "DAPS" else None}
    if wall is not None:
        row["wall_s"] = finite(wall, "wall time")
    if selector is not None:
        finite(selector, "selector")
    if loss is not None:
        finite(loss, "signed loss")
    return row

def b22_rows(b21: Path, run: Path, spec: dict) -> list[list[dict]]:
    b21_path = b21 / "analysis_theta0.7" / "fresh2_final_rows.csv"
    full = run / "full"
    for p, tag in ((b21_path,"b21_fresh2_rows"), (full/"manifest.json","b22_manifest"),
                   (full/"validation.json","b22_validation")):
        check_digest(p, spec["known_sha256"][tag])
    b21_rows = readcsv(b21_path, b21)
    if len(b21_rows) != 100:
        raise AuditStop("B21 row count drift")
    valid = readj(full/"validation.json",run)
    manifest = readj(full/"manifest.json",run)
    if valid.get("status") != "PASS" or valid.get("full_panel_complete") is not True:
        raise AuditStop("B22 validated full panel not PASS")
    ids = manifest.get("identities", {})
    if (ids.get("repo_head") != B22_SOURCE or ids.get("difffpr_head") != B22_DIFF or
        ids.get("sitcom_head") != SITCOM_HEAD or ids.get("model_sha256") != MODEL):
        raise AuditStop("B22 source/model identity drift")
    # B21 rows are NOT ordered in B22's sorted locked-filename panel order.
    # Join by immutable image ID. Never assume equal row_id or line position.
    b21_by_image = {}
    b21_row_ids = set()
    for row in b21_rows:
        img = image(row["image_id"])
        if img in b21_by_image:
            raise AuditStop("duplicate B21 image ID")
        idx = int(row["row_id"])
        if idx in b21_row_ids:
            raise AuditStop("duplicate B21 row_id")
        b21_by_image[img] = row
        b21_row_ids.add(idx)
    frozen_rows = manifest.get("rows", [])
    if (len(frozen_rows) != 100 or
        int(manifest.get("selected_row_count",-1)) != 100 or
        manifest.get("stage") != "full"):
        raise AuditStop("B22 full manifest cardinality/stage drift")
    b22_ids = [image(x["image_id"]) for x in frozen_rows]
    if len(set(b22_ids)) != 100 or set(b22_ids) != set(b21_by_image):
        raise AuditStop("B21/B22 panel image ID set mismatch")
    all_images = []
    for position, frozen in enumerate(frozen_rows):
        img = image(frozen["image_id"])
        if int(frozen.get("row_id",-1)) != position:
            raise AuditStop("B22 manifest row index/order drift")
        row = b21_by_image[img]
        if str(row["measurement_path"]) != str(frozen["measurement_path"]):
            raise AuditStop(f"B21/B22 frozen measurement path mismatch: {img}")
        if len(str(frozen.get("measurement_file_sha256",""))) != 64:
            raise AuditStop(f"B22 manifest missing measurement file SHA: {img}")
        if len(str(frozen.get("measurement_content_sha256",""))) != 64:
            raise AuditStop(f"B22 manifest missing measurement content SHA: {img}")
        sd = []
        for arm in (1,2):
            sd.append(norm(cohort="B22",method="DAPS",img=img,
                measurement="0"*64,source="B21_fresh2_frozen",rep=REPS["B22"],
                group="rep",index=arm-1,psnr=row[f"arm{arm}_psnr"],
                candidate_id=f"B21:{img}:rep{arm-1}",
                loss=row[f"arm{arm}_exact_operator_loss"],
                wall=row[f"arm{arm}_wall_seconds"], native_seed=row.get(f"seed{arm}")))
        common_sha = None
        common_file_sha = None
        groups = [sd]
        for method, k in (("SITCOM",4),("NP",8)):
            folder = full / method.lower() / f"row{position:03d}_{img}"
            p = readj(folder/"policy.json",run)
            if p.get("status")!="PASS" or image(p.get("image_id"))!=img or int(p.get("candidate_count",-1))!=k:
                raise AuditStop(f"B22 {method} policy mismatch {img}")
            sha = (p.get("measurement_content_sha256") if method=="SITCOM"
                   else p.get("measurement_raw_content_sha256"))
            if (not sha or sha != frozen["measurement_content_sha256"] or
                (common_sha is not None and sha != common_sha)):
                raise AuditStop(f"B22 frozen raw measurement mismatch {img}")
            common_sha = sha
            recs=[]
            tuples=([("run",i,f"run{i:02d}_masterseed43") for i in range(4)]
                    if method=="SITCOM" else
                    [(g,seed,f"cfg{cfg}_{g}_seed{seed}")
                     for cfg,g in enumerate(("lf","s2_preproj_lam001"))
                     for seed in (100,101,102,103)])
            for group,idx,dirname in tuples:
                r=readj(folder/"candidates"/dirname/"result.json",run)
                if r.get("status")!="PASS" or image(r.get("image_id"))!=img:
                    raise AuditStop(f"B22 candidate not PASS {img}:{method}:{idx}")
                if r.get("repo_head") != B22_SOURCE:
                    raise AuditStop(f"B22 candidate repo source mismatch {img}")
                # B21 and B22 must reference the exact same frozen measurement, not just image ID.
                if str(r.get("measurement_path")) != str(row["measurement_path"]):
                    raise AuditStop(f"B21/B22 measurement path mismatch {img}")
                file_sha = r.get("measurement_file_sha256")
                if not isinstance(file_sha, str) or len(file_sha) != 64:
                    raise AuditStop(f"B22 measurement file SHA missing {img}")
                if (file_sha != frozen["measurement_file_sha256"] or
                    (common_file_sha is not None and file_sha != common_file_sha)):
                    raise AuditStop(f"B22 manifest/SITCOM/NP measurement file SHA mismatch {img}")
                if r.get("ground_truth_content_sha256") != frozen.get("ground_truth_content_sha256"):
                    raise AuditStop(f"B22 GT content SHA identity mismatch {img}")
                if int(r.get("row_id",-1)) != position:
                    raise AuditStop(f"B22 candidate row index mismatch {img}")
                common_file_sha = file_sha
                if method=="SITCOM":
                    if (r.get("sitcom_head")!=SITCOM_HEAD or int(r.get("run_index",-1))!=idx
                        or int(r.get("master_seed",-1))!=43 or
                        int(r.get("selector",{}).get("step",-1))!=160 or
                        r.get("measurement_preprocessing")!="none" or
                        r.get("measurement_content_sha256")!=sha):
                        raise AuditStop(f"B22 SITCOM candidate identity {img}")
                    selector=r["selector"]["correction_norm"]
                else:
                    if (r.get("difffpr_head")!=B22_DIFF or r.get("config_tag")!=group or
                        int(r.get("seed",-1))!=idx or
                        r.get("measurement_preprocessing")!="clamp_min_zero_in_memory" or
                        r.get("measurement_raw_content_sha256")!=sha):
                        raise AuditStop(f"B22 NP candidate identity {img}")
                    selector=r["selector_stats"]["selector_post_winner_lf_mse_mean"]
                recs.append(norm(cohort="B22",method=method,img=img,measurement=sha,
                    source=B22_SOURCE,rep=REPS["B22"],group=group,index=idx,
                    candidate_id=f"B22:{method}:{img}:{group}{idx}",
                    psnr=r["metrics"]["psnr_raw"],selector=selector,
                    model_hash=(native_sitcom_sha(spec) if method=="SITCOM" else MODEL),
                    wall=r["timing"]["reconstruction_s"],
                    ssim=r["metrics"].get("ssim_raw"),
                    native_seed=idx))
            # Per-image native and oracle replay against the ACTUAL B22
            # materialized policy identity, not just aggregate Good25 counts.
            archived=p.get("policies", {})
            if method=="SITCOM":
                nidx=min(range(4),key=lambda j:(finite(recs[j]["step160_correction_norm"],"norm"),j))
                oidx=max(range(4),key=lambda j:(finite(recs[j]["psnr"],"psnr"),-j))
                expected_replay={
                    "SITCOM-1":(0,"candidate_run_index"),
                    "SITCOM-4S":(nidx,"candidate_run_index"),
                    "SITCOM-oracle4":(oidx,"candidate_run_index"),
                }
            else:
                nidx=min(range(8),key=lambda j:(finite(recs[j]["postproj_lf_mse"],"lf score"),j))
                oidx=max(range(8),key=lambda j:(finite(recs[j]["psnr"],"psnr"),-j))
                expected_replay={
                    "NP-1":(0,"candidate_seed"),
                    "NP-8-RS":(nidx,"candidate_seed"),
                    "NP-oracle8":(oidx,"candidate_seed"),
                }
            for name,(index,selector_key) in expected_replay.items():
                policy_row=archived.get(name,{})
                want=(int(recs[index]["index"]) if method=="NP"
                      else index)
                if int(policy_row.get(selector_key,-1))!=want:
                    raise AuditStop(f"archived selected candidate mismatch {img} {name}")
                if method=="NP" and policy_row.get("candidate_config_tag")!=recs[index]["group"]:
                    raise AuditStop(f"archived NP config mismatch {img} {name}")
                value=finite(policy_row.get("metrics",{}).get("psnr_raw"),"archived PSNR")
                if abs(value-finite(recs[index]["psnr"],"candidate PSNR"))>1e-7:
                    raise AuditStop(f"archived selected PSNR mismatch {img} {name}")
            groups.append(recs)
        if common_sha != frozen["measurement_content_sha256"]:
            raise AuditStop(f"missing cross-method frozen measurement match {img}")
        for d in sd:
            d["measurement_sha256"]=common_sha
        all_images.extend(groups)
    return all_images

def b24_rows(dev: Path, corrected: Path, safe_root: Path, spec: dict) -> list[list[dict]]:
    for path,tag in ((dev/"DEV80_MANIFEST.json","b24_dev80_manifest"),
                     (dev/"DEV80_SUMMARY.json","b24_dev80_summary"),
                     (corrected/"FRESH2_PER_IMAGE.csv","b24_corrected_fresh2_rows"),
                     (corrected/"FRESH2_SUMMARY.json","b24_corrected_summary")):
        check_digest(path,spec["known_sha256"][tag])
    m=readj(dev/"DEV80_MANIFEST.json",dev)
    summary=readj(dev/"DEV80_SUMMARY.json",dev)
    if (m.get("image_count")!=80 or len(m.get("tasks",[]))!=80 or
        m.get("confirmation_exposed") is not False or summary.get("status")!="PASS"):
        raise AuditStop("B24 DEV80 provenance drift")
    fr=readcsv(corrected/"FRESH2_PER_IMAGE.csv",corrected)
    if len(fr)!=80 or len(set(image(x["image_id"]) for x in fr))!=80:
        raise AuditStop("B24 Fresh2 loss ledger incomplete")
    fresh={image(x["image_id"]):x for x in fr}
    all_images=[]
    strata=Counter()
    for task in m["tasks"]:
        img=image(task["image_id"])
        strata[str(task["class_label"])]+=1
        tdir=dev/"workers"/f"gpu{int(task['assigned_gpu'])}"/str(task["output_subdir"])
        comp=readj(tdir/"IMAGE_COMPLETE.json",dev)
        if (comp.get("status")!="PASS" or comp.get("confirmation_exposed") is not False or
            image(comp.get("image_id"))!=img or
            comp.get("screening_stratum")!=task["class_label"]):
            raise AuditStop(f"B24 completion not PASS: {img}")
        sha=comp["measurement_tensor_sha256"]
        records=[]
        met=readj(Path(comp["baseline_metrics"]),safe_root)
        if (image(met["image_id"])!=img or met.get("evaluation_representation")!=REPS["B24_EXTERNAL"]):
            raise AuditStop(f"B24 baseline metric representation mismatch {img}")
        all_metrics=met.get("terminal_rows",[])
        if len(all_metrics)!=8:
            raise AuditStop(f"B24 missing eight baseline terminal metrics {img}")
        for method in ("DAPS","SITCOM"):
            g=readj(Path(comp[method.lower()+"_group"]),safe_root)
            cand=g.get("candidate_rows",[])
            if g.get("status")!="PASS" or len(cand)!=4:
                raise AuditStop(f"B24 {method} group invalid {img}")
            by_rep={int(x["rep"]):x for x in all_metrics if x["method"]==method}
            source = DAPS_HEAD+"+frozen_dirty" if method=="DAPS" else SITCOM_HEAD+"+frozen_dirty"
            if set(by_rep)!={0,1,2,3} or sorted(int(x["rep"]) for x in cand)!=[0,1,2,3]:
                raise AuditStop(f"B24 {method} candidate identity {img}")
            group=[]
            for x in sorted(cand,key=lambda x:int(x["rep"])):
                j=int(x["rep"]); q=by_rep[j]
                if (int(q["canonical_seed"]) != int(task[method.lower()+"_solver_seeds"][j]) or
                    int(q["native_seed"]) != int(x["native_seed"]) or
                    q["terminal_content_sha256"] != x["terminal_content_sha256"]):
                    raise AuditStop(f"B24 {method} rep/terminal hash mismatch {img} {j}")
                loss= fresh[img][f"daps_rep{j}_exact_operator_loss"] if method=="DAPS" and j<2 else None
                group.append(norm(cohort="B24",method=method,img=img,measurement=sha,
                    source=source,rep=REPS["B24_EXTERNAL"],group="rep",index=j,
                    candidate_id=f"B24:{method}:{img}:rep{j}",
                    psnr=q["psnr_raw_rgb_db"],ssim=q.get("ssim_raw_rgb"),
                    lpips=q.get("lpips_alex_raw_rgb"),
                    loss=loss,wall=q.get("gpu_active_seconds"),
                    model_hash=(native_sitcom_sha(spec) if method=="SITCOM" else MODEL),
                    stratum=task["class_label"],native_seed=q["native_seed"]))
            records.append(group)
        np_path=Path(comp["np_results"]["NP4_INDEPENDENT"])
        npv=readj(np_path,safe_root)
        ident=readj(np_path.parent.parent/"RUN_IDENTITY.json",safe_root)
        seeds=ident.get("np_root_seeds",[])
        if (npv.get("status")!="PASS" or npv.get("arm")!="NP4_INDEPENDENT" or
            int(npv.get("terminal_count",-1))!=4 or
            int(npv.get("total_unet_evals",-1))!=8800 or
            len(seeds)!=4 or len(set(seeds))!=4 or image(ident.get("image_id"))!=img or
            ident.get("measurement_tensor_sha256")!=sha or
            ident.get("model_sha256")!=MODEL):
            raise AuditStop(f"B24 NP4 identity/UNet mismatch {img}")
        nprows=[]
        for j,x in enumerate(npv.get("terminals",[])):
            if int(x["terminal_index"])!=j or str(x.get("lineage"))!=f"root{j}":
                raise AuditStop(f"B24 NP terminal identity/order {img}:{j}")
            nprows.append(norm(cohort="B24",method="NP",img=img,
                measurement=sha,source=str(ident.get("method_spec_sha256","B24_NP_SOURCE")),
                rep=REPS["B24_NP"],group="lf",index=j,
                candidate_id=f"B24:NP:{img}:root{j}",
                psnr=x["psnr_raw_db"],
                selector=x["selector_post_winner_lf_mse_mean"],
                native_seed=seeds[j],stratum=task["class_label"]))
        if (int(npv["clean_free_selected_terminal_index"]) !=
            min(range(4),key=lambda j:(finite(nprows[j]["postproj_lf_mse"],"NP selector"),j))):
            raise AuditStop(f"B24 NP native selector replay mismatch: {img}")
        if (abs(float(npv["clean_free_selected_psnr_raw_db"]) -
                nprows[int(npv["clean_free_selected_terminal_index"])]["psnr"])>1e-6):
            raise AuditStop(f"B24 NP selected PSNR drift: {img}")
        records.append(nprows)
        all_images.extend(records)
    if strata != Counter({"A":20,"B":20,"C":20,"D":20}):
        raise AuditStop("B24 unbalanced/frozen strata drift")
    return all_images

def writecsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AuditStop("refuse empty table")
    fields=sorted({field for row in rows for field in row})
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def reconcile_b22_patch(archive_patch: str, audit: dict) -> None:
    """B22 archived patch omits trailing unchanged whitespace context."""
    current = audit.get("current_patch")
    if (audit.get("status") != "PASS_BOUNDED_SOURCE_INSPECTION" or
        not isinstance(current,str) or
        audit.get("historical_B24_full_diff_reconciled") is not False or
        audit.get("current_patch_sha256") !=
            "d0bb0292e341abe3f6b99984bb2a98acb5919c41efe0f4bd8fd0ffb60b464fc5"):
        raise AuditStop("B22 patch receipt/hash/status not verified")
    if archive_patch.rstrip("\n ") != current.rstrip("\n "):
        raise AuditStop("B22 Python source patch mismatched beyond blank context")
    change_lines = [line for line in current.splitlines()
                    if (line.startswith("+") and not line.startswith("+++")) or
                       (line.startswith("-") and not line.startswith("---"))]
    if change_lines != ["-        Path(dir).mkdir()",
                        "+        Path(dir).mkdir(parents=True, exist_ok=True)"]:
        raise AuditStop("unexpected SITCOM source modification")

def run(args: Any) -> dict:
    spec=readj(args.spec)
    inv=readj(args.inventory)
    recon=readj(args.source_reconcile)
    if inv.get("stop") or not inv.get("no_confirmation_access"):
        raise AuditStop("A0.0 inventory failed/confirmation flag drift")
    if args.cohort not in ("B22", "B24", "all"):
        raise AuditStop("unsupported cohort")
    if spec.get("native_models",{}).get("sitcom_checkpoint_sha256") != MODEL:
        raise AuditStop("native SITCOM checkpoint SHA not independently verified")
    if args.cohort in ("B22", "all"):
        path = args.b22 / "full" / "manifest.json"
        check_digest(path,spec["known_sha256"]["b22_manifest"])
        frozen_manifest = readj(path,args.b22)
        reconcile_b22_patch(frozen_manifest["identities"]["sitcom_posterior_sample_diff"],
                            readj(args.patch_audit))
    daps=inv.get("sources",{}).get("DAPS",{})
    if (daps.get("matches_recorded_B24_diff") is not True or
        daps.get("unstaged_diff_sha256")!=B24_DIFF or daps.get("head")!=DAPS_HEAD):
        raise AuditStop("DAPS dirty-source provenance missing")
    # A frozen B24 FULL tracked diff cannot be certified from a current
    # SOURCE-ONLY diff, even if all remaining changed files are bytecode.
    if args.cohort in ("B24", "all"):
        if (recon.get("status")!="MATCH_B24_FULL_DIFF" or
            recon.get("current_full_diff_sha256")!=SITCOM_DIFF or
            recon.get("head")!=SITCOM_HEAD or
            not recon.get("no_unexpected_source_changes")):
            raise AuditStop("B24 SITCOM source remains uncertain; forbid B24/full analysis")
    if args.out.exists():
        raise AuditStop("A0 output already exists; never overwrite")
    imgs=[]
    if args.cohort in ("B22", "all"):
        imgs.extend(b22_rows(args.b21,args.b22,spec))
    if args.cohort in ("B24", "all"):
        imgs.extend(b24_rows(args.b24,args.corrected,args.b24.parent,spec))
    policies=[];candidates=[];unsupported=[]; by_group=defaultdict(list)
    for group in imgs:
        rows=ladders(group[0]["cohort"],group[0]["method"],group)
        candidates.extend(group)
        for x in rows:
            if x["status"]=="PASS":
                x["stratum"]=group[0]["stratum"]
                if group[0]["cohort"]=="B22":
                    k=x["K"]
                    sub=(group[4:4+k] if x["policy"].endswith("_S2")
                         else group[:k])
                    x["summed_candidate_wall_s"]=sum(finite(y["wall_s"],"wall") for y in sub)
                elif group[0]["method"] in ("DAPS","SITCOM"):
                    x["summed_candidate_gpu_active_s"]=sum(finite(y["wall_s"],"gpu active") for y in group[:x["K"]])
                elif x["K"]==4:
                    x["NP4_full_portfolio_UNet_evals"]=8800
                policies.append(x)
                by_group[(x["cohort"],x["method"],x["policy"],x["K"],x["representation"])].append(x)
            else:
                unsupported.append(x)
    expected={("B22","DAPS","NATIVE",2):92,
              ("B22","SITCOM","NATIVE",4):93,
              ("B22","NP","NATIVE",8):95,
              ("B24","DAPS","NATIVE",2):57,
              ("B24","NP","NATIVE",4):55}
    checks={}
    expected = {key:value for key,value in expected.items()
                if args.cohort in (key[0], "all")}
    for (co,me,po,k),good in expected.items():
        data=[r for r in policies if (r["cohort"],r["method"],r["policy"],r["K"])==(co,me,po,k)]
        actual=sum(r["good25"] for r in data)
        n=100 if co=="B22" else 80
        checks[f"{co}_{me}{k}_{po}"]={"expected_good25":good,"actual_good25":actual,"n":len(data),
                                     "status":"PASS" if len(data)==n and actual==good else "STOP"}
    if any(v["status"]!="PASS" for v in checks.values()):
        raise AuditStop("historical Good25 replay mismatch: "+repr(checks))
    summaries=[]
    for key,rows in sorted(by_group.items()):
        co,me,po,k,rep=key
        expected_count=100 if co=="B22" else 80
        if len(rows)!=expected_count:
            raise AuditStop(f"incomplete policy: {key} count={len(rows)}")
        summaries.append(dict(zip(("cohort","method","policy","K","representation"),key),
                              **summarize(rows)))
    # Predeclared, limited pairs. Never imply K matching implies compute matching.
    pairs=[]
    def pair_policy(left: tuple, right: tuple, reason: str) -> None:
        a,b=by_group.get(left),by_group.get(right)
        if a is None or b is None:
            raise AuditStop(f"missing predeclared paired comparison: {left} vs {right}")
        ak,bk=left,right
        if ak[4]!=bk[4]:
            raise AuditStop("unmatched metric representation in requested pair")
        comparison=paired(a,b,reps=spec["bootstrap"]["reps"],
                          seed=spec["bootstrap"]["seed"],
                          allow_different_k=(ak[3]!=bk[3]))
        pairs.append({"a":"/".join(map(str,ak)), "b":"/".join(map(str,bk)),
                      "purpose":reason, **comparison})
    for (co,me,po,k,rep),rows in sorted(by_group.items()):
        if po!="NATIVE":
            continue
        oracle_key=(co,me,"GT_ORACLE",k,rep)
        if oracle_key in by_group:
            pair_policy((co,me,po,k,rep),oracle_key,"within-family selector regret")
    if args.cohort in ("B22","all"):
        # Historical deployed flagship choices, K=2/4/8. Explicitly NOT matched compute.
        deployed=[("B22","DAPS","NATIVE",2,REPS["B22"]),
                  ("B22","SITCOM","NATIVE",4,REPS["B22"]),
                  ("B22","NP","NATIVE",8,REPS["B22"])]
        for a_idx in range(len(deployed)):
            for b_idx in range(a_idx+1,len(deployed)):
                pair_policy(deployed[a_idx],deployed[b_idx],
                            "historical flagship; unequal K and unmatched compute")
    if args.cohort in ("B24","all"):
        # Only comparable canonical 8-bit external baseline metrics here.
        rep=REPS["B24_EXTERNAL"]
        pair_policy(("B24","DAPS","NATIVE",2,rep),
                    ("B24","SITCOM","GT_ORACLE",4,rep),
                    "native DAPS2 vs SITCOM4 offline oracle; unmatched compute")
    args.out.mkdir(parents=True,exist_ok=False)
    writecsv(args.out/"A0_CANDIDATE_ROWS.csv",candidates)
    writecsv(args.out/"A0_POLICY_ROWS.csv",policies)
    writecsv(args.out/"A0_METRIC_SUMMARY.csv",summaries)
    if pairs:
        writecsv(args.out/"A0_PAIRED.csv",pairs)
    (args.out/"A0_REPLAY_CHECKS.json").write_text(json.dumps(checks,indent=2,sort_keys=True)+"\n")
    missing=[f"{x['cohort']} image={x['image_id']} {x['method']} native K={x['K']}: {x['why']}"
             for x in unsupported]
    (args.out/"A0_UNSUPPORTED_CELLS.md").write_text(
        "# Frozen policy omissions (no imputation)\n\n"+
        "\n".join("* "+x for x in sorted(set(missing)))+
        "\n\nB24 NP tensor-float PSNR and external 8-bit PSNR are not pooled or paired.\n"+
        "No DAPS B22 K4/K8, no executable DAPS B24 K4, and no B24 SITCOM step160 K4 native.\n")
    return {"status":"PASS_REPLAY_METADATA_ONLY","candidates":len(candidates),
            "policies":len(policies),"paired":len(pairs),"replay":checks,
            "external_B24_native_SITCOM_K4":"UNAVAILABLE"}

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",choices=("B22","B24","all"),required=True)
    for arg in ("spec","inventory","source-reconcile","patch-audit",
                "b21","b22","b24","corrected","out"):
        p.add_argument("--"+arg,required=True,type=Path)
    a=p.parse_args()
    # Never create output before passing all read/replay checks.
    try:
        result=run(a)
    except (AuditStop, KeyError, IndexError, TypeError, OSError, ValueError) as e:
        print(json.dumps({"status":"STOP","reason":str(e)},sort_keys=True))
        raise SystemExit(2)
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
