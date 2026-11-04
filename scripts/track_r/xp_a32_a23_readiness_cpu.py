#!/usr/bin/env python3
"""A23 CPU-only proactive PAC readiness audit; NEVER launches CUDA jobs.

Source: accepted A22 G7 C2/CLIP and A23.5. This checks 280 frozen
diagnostic GT source bytes, published native C2 and CLIP CSV/receipt
identity, project budgets and missing XP-K native / replay gates.
Creates a new local receipt, never changes existing outputs, and never
claims that a physically busy GPU is admissible based on memory alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path("/egr/research-pac/huang248")
BASE = ROOT / "outputs/pr_diffusion/track_r"
WT = Path(__file__).resolve().parents[2]
A22 = "20261104T025355Z"
G7_RETURN = "docs/track_r/executors/xp/returns/XP_A22_G7_C2_CLIP_20261104T035034Z"
BASE_HEAD = "999d7eb9ee2a254a41c48354e3721155fa046e2a"
G7_PUB_COMMIT = "9090a50a4d4ffe922bdb43c5ee9e877c059a7dd1"
G7_C_MANIFEST_SHA = "d517a8ea337c5fed8925f925bf24bae21cea60a566eb29f6795bc31e350abfde"
G7_C2_MANIFEST_SHA = "75e8fe296932044eb4ff1cd30300ef58a8f48ece02acd9b22ba3aa6f3bd7c7ac"
TAB3_SHA = "124d67030bdf42c966f6b777c17ef9c1177596d436160fb104caa3cf0d0cc148"
G7_TABLE_SHA = "841f2b77be35e45be94f23716c927065a5df7abab1e08f679c8ba2b460ffa7b9"
G7_CLIP_SHA = "5507d1ce0e6f3822edbcc0430a6800e5bc4cf4791befde61fa375fd7b829b967"
G7_USED = 2.6265298946173345
PROJECT_USED = 244.45709476042322
POLICIES = ("S_mid_global2", "S_mid_all8", "2_forward", "UnivFD")
FORBID = ("confirmation", "confirm305", "test-ffhq", "demo-ffhq")


def require(ok, tag):
    if not ok:
        raise RuntimeError("STOP_A23_CPU_READINESS_" + tag)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "SOURCE_FILE_" + path.name)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path, expected_rows):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None, "CSV_COLUMNS")
        rows = list(reader)
    require(len(rows) == expected_rows, "CSV_N_" + Path(path).name)
    return rows


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(WT), *args], text=True, timeout=60
    ).strip()


def validate_gt_images(manifest, verify_source_bytes=True):
    require(
        manifest.get("status") == "XP_C_GT_IMAGES_FROZEN"
        and manifest.get("n_images") == 280
        and manifest.get("n_ffhq") == 180
        and manifest.get("n_imagenet") == 100
        and manifest.get("diagnostic_gt_only") is True
        and manifest.get("scoring_or_GPU_work_performed") is False,
        "GT_FROZEN_HEADER",
    )
    images = manifest.get("images", [])
    require(len(images) == 280, "GT_IMAGE_ROWS")
    counts = Counter()
    keys = {}
    for item in images:
        d = item.get("dataset")
        k = (d, item.get("basename"))
        counts[d] += 1
        require(k not in keys, "GT_IMAGE_DUPLICATE")
        path = Path(item["source_path"])
        resolved = path.resolve(strict=False)
        require(
            path.is_absolute()
            and not path.is_symlink()
            and path.is_file()
            and ROOT.resolve() in resolved.parents
            and not any(bad in str(resolved).lower() for bad in FORBID),
            "GT_IMAGE_PROTECTED_PATH",
        )
        require(
            isinstance(item.get("source_sha256"), str)
            and len(item["source_sha256"]) == 64,
            "GT_IMAGE_DIGEST_REQUIRED",
        )
        if verify_source_bytes:
            require(sha(path) == item["source_sha256"], "GT_SOURCE_BYTE_SHA")
        keys[k] = item["source_sha256"]
    require(counts == Counter({"ffhq": 180, "imagenet": 100}),
            "EXACT_DATASET_SPLIT")
    return keys


def check_clip_table(rows, keys):
    found = {}
    for row in rows:
        key = (row["dataset"], row["basename"])
        require(key not in found and key in keys, "CLIP_INPUT_IDENTITY")
        require(
            row["original_source_sha256"] == keys[key]
            and row["diagnostic_gt_only"] in ("True", "true", "1")
            and row["g1_orientation"] in ("000", "111")
            and row["g2_orientation"] in ("000", "111"),
            "CLIP_ORIENTATION_OR_SOURCE",
        )
        found[key] = row
    require(set(found) == set(keys), "CLIP_ALL_280_MATCH")
    ff = [v for (dataset,_),v in found.items() if dataset=="ffhq"]
    im = [v for (dataset,_),v in found.items() if dataset=="imagenet"]
    # Descriptive factual check already accepted by the planner.
    n1 = sum(v["g1_orientation"] == "111" for v in im)
    n2 = sum(v["g2_orientation"] == "111" for v in im)
    nf = sum(v.get("face_orientation") == "111" for v in ff)
    require(n1 == n2 == 8 and nf == 0, "A23_PLANNER_CLIP_GT_FIGURES")
    return {"ffhq_face_flipped": nf, "imagenet_g1_flipped": n1,
            "imagenet_g2_flipped": n2}


def read_only_gpu_snapshot():
    """Informational only; never a GPU admission or launch authorization."""
    cmd = ["nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        if r.returncode != 0:
            return {"status": "TELEMETRY_UNAVAILABLE",
                    "error": r.stderr[-300:]}
        rows = []
        for line in r.stdout.splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) < 6:
                continue
            rows.append(dict(
                physical_index=int(v[0]), name=v[1],
                total_mib=int(v[2]), used_mib=int(v[3]),
                free_mib=int(v[4]), utilization_pct=int(v[5])))
        return dict(status="READ_ONLY_SNAPSHOT_NOT_ADMISSION",
                    physical_gpus=rows,
                    no_foreign_process_signaling=True,
                    source="nvidia-smi, no CUDA context")
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        return {"status": "TELEMETRY_UNAVAILABLE", "error": str(e)[:300]}


def collect(a):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "CPU_ONLY_ENV")
    require(
        git("rev-parse", "HEAD") == a.expected_head
        and not git("status", "--porcelain"),
        "PINNED_CLEAN_WORKTREE",
    )
    local = BASE / f"xp_c_c2_a22_{A22}"
    freeze = local / "XP_C_GT_IMAGES.json"
    tasks = local / "XP_C2_EQUIVARIANCE_TASKS.json"
    proof = load_json(local / "XP_C_C2_PREP.json")
    require(
        proof.get("status") == "XP_C_C2_PREP_PASS"
        and proof.get("head") == BASE_HEAD
        and sha(freeze) == proof.get("C_manifest_sha256") == G7_C_MANIFEST_SHA
        and sha(tasks) == proof.get("C2_manifest_sha256") == G7_C2_MANIFEST_SHA,
        "ORIGINAL_G7_NATIVE_SOURCE",
    )
    keys = validate_gt_images(load_json(freeze),
                              verify_source_bytes=(not a.fast_provenance))
    original_g7 = WT / G7_RETURN
    gpu = load_json(original_g7 / "XP_A22_G7_C2_FINAL_STATUS.json")
    readout = load_json(original_g7 / "XP_A22_G7_C2_READOUT_RECEIPT.json")
    clipr = load_json(original_g7 / "XP_A12_G7_C_CLIP_RECEIPT.json")
    published = load_json(original_g7 / "XP_A22_G7_PUBLICATION_RECEIPT.json")
    require(
        gpu["status"] == "A22_G7_C2_280_COMPLETE"
        and len(gpu["completed"]) == 280
        and gpu["source_hash"] == G7_C2_MANIFEST_SHA
        and abs(gpu["gpu_occupancy_h"]-G7_USED) < 1e-10
        and abs(gpu["project_gpu_h"]-PROJECT_USED) < 1e-10,
        "PUBLISHED_GPU_LEDGER",
    )
    require(
        readout["status"] == "A22_G7_C2_IMAGE_BOOTSTRAP_CPU_COMPLETE"
        and readout["n_summary_rows"] == 36
        and readout["n_task_rows"] == 10080
        and readout["n_input_images"] == 280,
        "PUBLISHED_C2_READOUT",
    )
    require(
        clipr["status"] == "XP_A12_G7_GT_CLIP_G1_G2_FACE_CPU_PASS"
        and clipr["n_images"] == 280
        and clipr["input_manifest_sha256"] == G7_C_MANIFEST_SHA
        and published["not_full_G7_completion"] is True
        and published["status"] == "A22_G7_C2_AND_CPU_CLIP_DIAGNOSTICS_PUBLISHED",
        "PUBLISHED_GT_CLIP_SCOPE",
    )
    table = original_g7 / "XP_A22_G7_C2_EQUIVARIANCE.csv"
    clipfile = original_g7 / "XP_A12_G7_C_CLIP_GT_ORIENTATION.csv"
    require(
        sha(table) == readout["table_sha256"] == G7_TABLE_SHA
        and sha(clipfile) == clipr["csv_sha256"] == G7_CLIP_SHA,
        "PUBLISHED_G7_CSV_SHA",
    )
    eqrows = read_csv(table, 36)
    require(
        len({(r["dataset"],r["t_ddpm1000"],r["rotation"]) for r in eqrows}) == 36,
        "C2_36_UNIQUE_GROUPS",
    )
    stats = check_clip_table(read_csv(clipfile, 280), keys)
    table3 = WT / "docs/track_r/executors/xp/theory_inputs/XP_A22_TAB3_IMAGENET_E1_G1.csv"
    # The A23 branch derives from the A22 code branch, not from its independent
    # Tab3 publisher: this file intentionally resides ONLY in the latter branch.
    # Do not silently require or copy it from this engineering checkout.
    main_cut = dict(status="A23_ACCEPTED_CHECKPOINT_ALREADY_PUBLISHED",
        tab3_remote_commit="df6e8cd5c62f506343f2ff8fe6d493c4d032ea68",
        tab3_remote_csv_sha256=TAB3_SHA,
        tab3_present_in_this_code_worktree=table3.is_file(),
        n2_frozen_prompts_in_code_tree=True,
        g7_remote_commit=G7_PUB_COMMIT)
    prompts = WT / "docs/track_r/executors/xp/theory_inputs/XP_A22_FROZEN_N2_FACE_PROMPTS.json"
    require(prompts.is_file(), "N2_SOURCE_EXISTS")
    p = load_json(prompts)
    require(p["source_commit"] ==
            "b1637f7c27a3caaa03a8b204171fca0d3fccfcd5"
            and len(p["prompts_upright"]) == len(p["prompts_inverted"]) == 3,
            "N2_SOURCE_PROVENANCE")
    cap = dict(project_cap_h=400.,
        published_project_used_h=PROJECT_USED,
        published_project_remaining_h=400.-PROJECT_USED,
        G7_total_cap_h=10.,
        G7_published_C2_used_h=G7_USED,
        G7_remaining_before_new_charges_h=10.-G7_USED,
        XP_K_cap_h=30.,
        requires_fresh_PAC_ledger_before_GPU=True)
    return dict(
        status="A23_CPU_READINESS_PASS_GPU_STAGES_PARKED_BY_SCIENCE_GATES",
        code_head=a.expected_head,
        governing_planner="6f5ee7f",
        accepted_main_paper_cut=True,
        no_main_text_new_data=True,
        science_scope="supplement_only",
        CPU_only=True, no_CUDA_context_created=True,
        original_280_GT_image_manifest_sha256=G7_C_MANIFEST_SHA,
        original_10080_task_manifest_sha256=G7_C2_MANIFEST_SHA,
        source_byte_sha256_reverified=not a.fast_provenance,
        n_GT_images=280, FFHQ_images=180, ImageNet_images=100,
        no_GT_for_inverse_problem_generation=True,
        accepted_GT_CLIP_diagnostic_counts=stats,
        published_G7_table_sha256=G7_TABLE_SHA,
        published_G7_clip_sha256=G7_CLIP_SHA,
        accepted_main_cut=main_cut,
        GPU_budget=cap,
        GPU_snapshot=read_only_gpu_snapshot(),
        G7_remaining=[
            dict(stage=p, order=i+1, status="NOT_RUN_MISSING_FROZEN_NATIVE_ADAPTER",
                 clean_GPU_required=False,
                 may_use_cotenant_gpu_after_science_and_admission_gates=True)
            for i,p in enumerate(POLICIES)],
        XP_K={
            "status":"PENDING_A23_SCIENCE_GATES_NO_GENERATION",
            "single_native_parameter_diff_per_family":"NOT_PROVEN",
            "two_historical_B22_roots_identity_replay_per_family":"NOT_RUN",
            "DAPS_F8_replay_gate":"PENDING_PROOF",
            "GPU_clean_required":False,
            "GPU_admission_requires_safe_VRAM_and_foreign_job_protection":True,
            "SITCOM_target_tau":0.05,
            "FAST_DIPS_target_epsilon":33.26,
            "DAPS_FFHQ_target_tau":math.sqrt(2.)*.05,
        },
        cp4={"status":"SUPPLEMENT_PENDING", "due_ET":"2026-11-15T12:00:00-05:00",
             "must_report_missing_stages":True},
        optional_A21_10="EXCLUDED_NOT_RUN",
        no_protocol_v2=True, no_new_selector=True,
        no_matched_compute_claim=True,
        all_GPU_experiments_launched_by_this_script=0,
    )


def main(a):
    data = collect(a)
    out = Path(a.out)
    require(
        out.parent == BASE
        and out.name.startswith("xp_a32_a23_readiness_")
        and not out.exists()
        and not out.is_symlink(),
        "NEW_RECEIPT_DIRECTORY",
    )
    out.mkdir(mode=0o700)
    target = out / "A23_CPU_READINESS_RECEIPT.json"
    with target.open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print("A23_CPU_READINESS_PASS", target, flush=True)
    print("A23_NO_GPU_JOBS_LAUNCHED", flush=True)
    print("A23_G7_REMAINING_GPU_H", data["GPU_budget"]["G7_remaining_before_new_charges_h"], flush=True)
    print("A23_RESOURCE_CAPACITY_ONLY_NOT_NATIVE_READINESS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--expected-head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fast-provenance", action="store_true",
                   help="Skip source bytes for all 280 GT files; default is full SHA validation")
    try:
        main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A23_CPU_READINESS", repr(e), flush=True)
        sys.exit(8)
