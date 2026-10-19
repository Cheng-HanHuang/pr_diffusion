#!/usr/bin/env python3
"""A1.0d native independent-K4 source check and SYNTHETIC-ONLY dry-run manifest.

No torch, CUDA, model, measurement, image, GT, FFT, external solver or new cohort.
Never executes an argument or touches B24 data. Native source read is bounded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from scripts.track_a.a1_protocol import ARMS, K, candidate_matrix, mixed_np4, mixed_np8, seed32_numpy

ROOT = Path("/egr/research-pac/huang248")
CHECKOUT = ROOT / "pr_diffusion_track_a_a1_oracle_budget"
BASE = ROOT / "outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
DAPS = ROOT / "pr_diffusion_b19_solver/external/daps"
SITCOM = ROOT / "external/SITCOM_ODE"
DAPS_HEAD = "e7a77d094167084faed19b599b96673b7bb11447"
SITCOM_HEAD = "275ab67efbd8146bffca20155171ba6be1169c09"
SYNTHETIC_ID = "11111"  # protocol sentinel; NOT an FFHQ cohort allocation


def git_head(path: Path) -> str:
    call = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=15, check=False)
    if call.returncode:
        raise ValueError(f"native git identity inspection failed for {path}")
    return call.stdout.strip()


def bounded_source(path: Path) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 400_000:
        raise ValueError(f"missing, symlink, or oversized native source: {path}")
    return path.read_text(encoding="utf8")


def source_checks(daps_text: str, sitcom_text: str) -> dict:
    anchor = "def get_start(self, batch_size, model):"
    if daps_text.count(anchor) != 1:
        raise ValueError("DAPS fresh-start method absent/ambiguous")
    method = daps_text.split(anchor, 1)[1].split("\n\nclass ", 1)[0]
    statements = ["os.environ.get(\"B21_START_NOISE_SEED\"",
                  "_b21_seed_all(int(start_seed))",
                  "torch.randn(batch_size, *in_shape, device=device)"]
    positions = [method.find(s) for s in statements]
    if any(i < 0 for i in positions) or positions != sorted(positions):
        raise ValueError("DAPS source start-noise RNG reordering/drift")
    seed_func = daps_text.split("def _b21_seed_all(seed):", 1)
    if len(seed_func) != 2:
        raise ValueError("DAPS seed helper missing")
    helper = seed_func[1].split("\n\ndef ", 1)[0]
    if not all(s in helper for s in ["np.random.seed(seed)", "torch.manual_seed(seed)",
                                     "torch.cuda.manual_seed_all(seed)"]):
        raise ValueError("DAPS start seed helper no longer seeds all required RNGs")
    sitcom_starts = re.findall(r"def get_start\(self, ref\):([\s\S]*?)(?=\n    def |\nclass |\Z)",
                                sitcom_text)
    if len(sitcom_starts) != 2:
        raise ValueError("expected two native SITCOM get_start(ref) definitions")
    for body in sitcom_starts:
        if "torch.randn_like(ref)" not in body:
            raise ValueError("SITCOM native reference is not random shape-only initialization")
        if re.search(r"(ref\.sum|ref\.mean|ref\.clone|ref\s*[+\-*/]|ref\s*\.detach)", body):
            raise ValueError("SITCOM native reference potentially accesses GT pixel values")
    return {
        "status":"PASS_SOURCE_ONLY",
        "daps_seed_before_gaussian_start":True,
        "daps_numpy_torch_cuda_reseed":True,
        "sitcom_random_shape_only_start_methods":len(sitcom_starts),
        "native_complete_replay_performed":False,
    }


def make_synthetic_plan(identifier: str = SYNTHETIC_ID) -> dict:
    if identifier != SYNTHETIC_ID:
        raise ValueError("A1.0d cannot plan actual image IDs; synthetic sentinel only")
    rows = candidate_matrix(identifier)
    plans = []
    for row in rows:
        native32 = seed32_numpy(row.family, identifier, row.replicate)
        plans.append({
            "candidate_id":row.candidate_id,
            "family":row.family,
            "replicate":row.replicate,
            "config":row.config,
            "canonical_seed63":row.seed,
            "native_seed32":native32,
            "independent_process_required":True,
            "execution_spec":"NOT_LAUNCHABLE_UNTIL_A1_OWNED_NATIVE_WRAPPERS_FROZEN",
            "measurement_path":None,
            "ground_truth_path":None,
            "device":None,
        })
    if Counter(p["family"] for p in plans) != Counter(K):
        raise ValueError("candidate balance drift")
    for family in ARMS:
        seeds = [p["native_seed32"] for p in plans if p["family"] == family]
        if len(set(seeds)) != K[family]:
            raise ValueError(f"{family}: duplicate native uint32 seeds")
    if len({p["native_seed32"] for p in plans}) != len(plans):
        raise ValueError("cross-family 32-bit native seed collision")
    return {
        "schema":"a1.synthetic.native-k4-dryrun.v1",
        "status":"CPU_SYNTHETIC_ONLY",
        "image_id":identifier,"image_id_status":"FAKE_SENTINEL_NOT_A_COHORT",
        "new_cohort_reserved":False,"run_reserved":False,"gpu_work":False,
        "measurement_generated":False,"gt_loaded":False,"model_loaded":False,
        "processes_executed":0,"candidates_generated":0,
        "planned_candidate_specs":plans,
        "expected_counts":{"DAPS":4,"SITCOM":4,"NP_LF":4,"NP_S2":4},
        "mixed_np8":[c.candidate_id for c in mixed_np8(rows)],
        "mixed_np4":[c.candidate_id for c in mixed_np4(rows)],
        "mixed_np_reuses_candidates":True,
        "blocked_gates":["A1-owned source-pinned native GPU wrappers",
                         "comparable metric/output pipeline",
                         "source-disjoint approved development cohort",
                         "DATA/RUN reservations and user-executed PAC launcher"],
    }


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-head",required=True)
    args=parser.parse_args()
    output=args.output.absolute()
    if output.parent!=BASE or output.exists():
        print("STOP: A1.0d output namespace mismatch / existing file")
        return 2
    if git_head(CHECKOUT)!=args.expected_head:
        print("STOP: A1.0d pinned A1 commit does not match")
        return 2
    result=make_synthetic_plan()
    src=source_checks(bounded_source(DAPS/"sampler.py"),
                      bounded_source(SITCOM/"sampler.py"))
    if git_head(DAPS)!=DAPS_HEAD or git_head(SITCOM)!=SITCOM_HEAD:
        print("STOP: native parent HEAD differs from verified identity")
        return 3
    result["native_source_checks"]=src
    result["native_heads"]={"DAPS":DAPS_HEAD,"SITCOM":SITCOM_HEAD}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,indent=2)
        f.write("\n")
    print(json.dumps({"status":result["status"],
                      "source_gate":src["status"],
                      "unique_specs":len(result["planned_candidate_specs"]),
                      "native_seed32_distinct":True,
                      "mixed8_reuses_existing":True,
                      "cohort":"NONE","GPU":"NONE"},sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"A1_STOP_DRYRUN: {type(e).__name__}: {e}")
        sys.exit(4)
