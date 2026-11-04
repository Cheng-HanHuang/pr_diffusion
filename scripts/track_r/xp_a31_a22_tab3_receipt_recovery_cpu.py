#!/usr/bin/env python3
"""A22 CPU-only repair for completed Table3 scoring but missing receipt.

Never reruns G1, VGG-LPIPS, native DAPS/SITCOM, or GPU.
Validates original CSV bytes against immutable frozen A7/D2/D3 and G5
chosen-run/metrics, records the old scoring commit and new receipt-only
repair commit, then invokes the existing independent CPU publisher.
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
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts.track_r.xp_a30_a22_tab3_e1_cpu import (
    A7_AGG,
    BASE,
    E1_AGG_KEYS,
    E1_KEYS,
    WT,
    daps_g5,
    g1_select,
    load_a7,
)

ORIGINAL_HEAD = "999d7eb9ee2a254a41c48354e3721155fa046e2a"
ORIGINAL_RUN = "xp_a30_a22_autopilot_20261104T025355Z"
ORIGINAL_TABLE = "xp_a30_a22_tab3_20261104T025355Z"


def gate(test, message):
    if not test:
        raise RuntimeError("STOP_A22_TAB3_RECOVERY_" + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(WT), *args], text=True, timeout=60
    ).strip()


def parse_csv(path, expected_fields, n):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        gate(tuple(reader.fieldnames or ()) == tuple(expected_fields),
             "CSV_SCHEMA_" + Path(path).name)
        records = list(reader)
    gate(len(records) == n, "CSV_ROWCOUNT_" + Path(path).name)
    return records


def almost(actual, expected, label, tol=1e-9):
    gate(
        math.isfinite(float(actual))
        and math.isfinite(float(expected))
        and abs(float(actual) - float(expected)) <= tol,
        "MISMATCH_" + label,
    )


def main(args):
    gate(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and git("rev-parse", "HEAD") == args.expected_head
        and not git("status", "--porcelain"),
        "CPU_PINNED_CODE",
    )
    original = BASE / ORIGINAL_RUN
    data = BASE / ORIGINAL_TABLE
    gate(original.is_dir() and data.is_dir(), "EXISTING_A22_CPU_RUN")
    state = json.loads((original / "A22_AUTOPILOT_STATE.json").read_text())
    gate(
        state["execution_head"] == ORIGINAL_HEAD
        and state["stage_results"]["TABLE3"]["status"] == "FAILED"
        and "TABLE3_PUBLICATION" not in state["stage_results"],
        "ORIGINAL_TABLE3_STAGE_FAILED",
    )

    table_path = data / "XP_A22_TAB3_IMAGENET_E1_G1.csv"
    image_path = data / "XP_A22_TAB3_IMAGE_E1_G1.csv"
    receipt_path = data / "XP_A22_TAB3_RECEIPT.json"
    gate(
        table_path.is_file() and image_path.is_file()
        and not receipt_path.exists(),
        "ORIGINAL_TWO_CSV_ONLY",
    )
    table = parse_csv(table_path, E1_AGG_KEYS, 3)
    images = parse_csv(image_path, E1_KEYS, 300)

    groups = defaultdict(list)
    for row in images:
        label = (row["panel"], int(row["K"]))
        gate(label in {("D2", 4), ("D2", 8), ("D3", 4)},
             "EXPECTED_POLICY_GRID")
        gate(row["family"] == ("SITCOM" if row["panel"] == "D3" else "DAPS-4K"),
             "FAMILY_IDENTITY")
        gate(row["g1_choice"] in ("000", "111"),
             "FROZEN_G1_ORIENTATION")
        gate(
            int(row["GOOD20"]) == int(float(row["PSNR"]) >= 20.0)
            and all(
                math.isfinite(float(row[name]))
                for name in ("PSNR", "SSIM", "VGG_LPIPS")
            )
            and -1.001 <= float(row["SSIM"]) <= 1.001
            and abs(float(row["VGG_LPIPS"])) <= 10,
            "OFFLINE_SCORE_FINITE_GOOD20",
        )
        groups[label].append(row)

    expected_labels = {("D2", 4), ("D2", 8), ("D3", 4)}
    gate(set(groups) == expected_labels, "EXACT_THREE_POLICIES")
    gate(
        all(
            len(g) == 100 and len({r["image_id"] for r in g}) == 100
            for g in groups.values()
        ),
        "ONE_ROW_PER_IMAGE",
    )
    gate(
        len({(r["panel"], int(r["K"]), r["image_id"]) for r in images}) == 300,
        "NO_DUPLICATE",
    )

    for rec in table:
        key = (rec["panel"], int(rec["K"]))
        gate(key in groups and int(rec["n_images"]) == 100, "TABLE_GROUP")
        for metric in ("PSNR", "SSIM", "VGG_LPIPS", "GOOD20"):
            almost(rec[metric], mean(float(r[metric]) for r in groups[key]),
                   f"AGG_{key}_{metric}")

    # Native A7 source re-hash and original receipt status. This reads
    # frozen GT-free root receipts, not new solver inference.
    a7_d2, refs2, proof2 = load_a7("D2")
    a7_d3, refs3, proof3 = load_a7("D3")
    gate(
        proof2["detail_sha256"] == A7_AGG["D2"][1]
        and proof3["detail_sha256"] == A7_AGG["D3"][1],
        "A7_PUBLISHED_SHA_REVALIDATION",
    )

    # DAPS rows can be checked independently against already verified
    # native G5 G1 and piq metrics. No CLIP or piq inference takes place.
    daps = daps_g5(a7_d2, refs2)
    for K in (4, 8):
        for row in groups[("D2", K)]:
            iid = row["image_id"]
            gate(iid in daps, "DAPS_IMAGE_GRID")
            chosen = g1_select(daps[iid], K)
            gate(
                int(row["selected_run"]) == int(chosen["run"])
                and row["g1_choice"] == chosen["g1_choice"],
                "DAPS_G1_FROZEN_SELECTION",
            )
            for metric in ("PSNR", "SSIM", "VGG_LPIPS"):
                almost(row[metric], chosen[metric],
                       f"DAPS_K{K}_{iid}_{metric}")

    # SITCOM first K runs and PSNR at exact saved 000/111 orientation.
    # Its G1 choices are recovered unchanged from the completed CSV;
    # do not run the CLIP model again or suggest that they are revalidated.
    d3by = defaultdict(list)
    for (iid, rep), src in a7_d3.items():
        d3by[iid].append(dict(run=rep, rho=float(src["rho_signed"])))
    for row in groups[("D3", 4)]:
        iid = row["image_id"]
        gate(iid in d3by, "SITCOM_IMAGE_GRID")
        selected = g1_select(d3by[iid], 4)
        j = int(selected["run"])
        gate(int(row["selected_run"]) == j, "SITCOM_MIN_RHO")
        source = a7_d3[(iid, j)]
        idx = 7 if row["g1_choice"] == "111" else 0
        almost(row["PSNR"], source["orbit"]["psnr_db"][idx],
               f"SITCOM_ORIENTED_PSNR_{iid}", tol=1e-6)

    provenance = dict(
        status="A22_TABLE3_E1_IMAGE_NET_CPU_PASS",
        n_images=100,
        n_policy_rows=300,
        n_tables=3,
        A7={"D2": proof2, "D3": proof3},
        source_DAPS_G5="frozen CP3 read_stage('G5'), first four D2 historical roots verified bytewise",
        source_SITCOM="frozen A7 D3 400 native roots",
        G1="frozen G1 original prior scoring; 111 chosen iff positive margin",
        selector="E0 min signed measurement rho among first K, stable run index; then G1 orientation",
        SSIM="piq 0.8.0, native float [0,1], chosen orientation",
        VGG_LPIPS="piq 0.8.0 pretrained VGG, native float [0,1], chosen orientation",
        PSNR="frozen RGB8 at chosen orientation",
        simulator_or_solver_gpu_calls=0,
        no_GT_in_selection=True,
        GT_for_offline_metrics_only=True,
        not_matched_compute=True,
        optional_a21_10_replication_not_run=True,
        implementation_head=args.expected_head,
        original_scoring_head=ORIGINAL_HEAD,
        receipt_only_repair=True,
        repair_reason="original Table3 scoring shadowed argparse.Namespace a with a list at summarize assignment; CSVs completed before receipt creation",
        no_repeat_CLIP_or_LPIPS=True,
        DAPS_G5_metrics_independently_revalidated=True,
        SITCOM_selection_and_PSNR_revalidated=True,
        SITCOM_G1_choice_copied_from_original_scoring=True,
        table_sha256=sha(table_path),
        local_image_detail_sha256=sha(image_path),
    )
    if args.verify_only:
        print(
            "A22_TABLE3_EXISTING_CSVS_VERIFIED",
            "rows", len(table), len(images),
            "original_head", ORIGINAL_HEAD,
            "new_repair_head", args.expected_head,
            "table_sha", provenance["table_sha256"],
            flush=True,
        )
        return

    # Atomic create; do not modify a successful receipt or CSV bytes.
    with receipt_path.open("x") as f:
        json.dump(provenance, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print("A22_TABLE3_RECEIPT_ONLY_REPAIRED", receipt_path, flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(WT))
    cmd = [
        sys.executable,
        "-u",
        str(WT / "scripts/track_r/xp_a30_a22_tab3_publish_cpu.py"),
        "--expected-head", args.expected_head,
        "--run", str(data),
    ]
    subprocess.run(cmd, env=env, cwd=str(WT), check=True, timeout=1200)
    print(
        "A22_TABLE3_RECEIPT_REPAIR_AND_PUBLICATION_PASS",
        provenance["table_sha256"], flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--expected-head", required=True)
    p.add_argument("--verify-only", action="store_true")
    try:
        main(p.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A22_TABLE3_RECEIPT_RECOVERY", repr(exc),
              flush=True)
        sys.exit(8)
