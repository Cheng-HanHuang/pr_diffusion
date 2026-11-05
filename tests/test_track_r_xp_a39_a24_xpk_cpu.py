"""A39: CPU-only A24 XP-K preparation must never pass the native/generation gates."""
from __future__ import annotations
import ast
import copy
import json
import unittest
from pathlib import Path
from scripts.track_r.xp_a39_a24_xpk_cpu_preflight import (
    EXPECTED_PLANNER, EXPECTED_HEADS, validate_F8_receipt,
    snapshot, B22, FAST, FROZEN_DAPS
)
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/track_r/xp_a39_a24_xpk_cpu_preflight.py"
LAUNCH=ROOT/"scripts/track_r/xp_a39_a24_xpk_cpu_launch.sh"
F8=ROOT/"docs/track_r/executors/xp/returns/XP_F8_LEGACY_DAPS_8ROOT_REPLAY_EXACT_PASS_20261029.md"

class A39A24Tests(unittest.TestCase):
    def test_planner_A24_native_science_gates_not_weakened(self):
        self.assertEqual(EXPECTED_PLANNER,
            "aafaf7f35e9d9d0beafa459259ca10b7fda4e8f5")
        s=SCRIPT.read_text()
        ast.parse(s,str(SCRIPT))
        for v in ("GPU_generation_authorized","PENDING_ALL_FAMILIES",
                  "NOT_RUN_ALL_FAMILIES","GPU_allocated",
                  "native_source_binding_proof",
                  "two_B22_root_identity_replay"):
            self.assertIn(v,s)
        self.assertNotIn("torch.cuda",s)
        self.assertNotIn("subprocess.Popen",s)
    def test_exact_SITCOM_one_parameter_from_real_B22(self):
        src=json.loads(B22.read_text())["sitcom"]
        new,receipt=snapshot("SITCOM",src,"lgvd_tau",.05)
        self.assertEqual(new["lgvd_tau"],.05)
        self.assertEqual(src["lgvd_tau"],.01)
        for k in src:
            if k!="lgvd_tau":
                self.assertEqual(src[k],new[k])
        self.assertEqual(receipt["changed_key_count"],1)
        self.assertFalse(receipt["future_GPU_generation_authorized"])
        self.assertFalse(receipt["complete_native_source_diff_gate_passed"])
        self.assertEqual(src["lgvd_steps"],100)
        self.assertEqual(src["lgvd_lr"],.00005)
    def test_exact_FAST_one_parameter_from_pinned_native_config(self):
        src=json.loads(FAST.read_text())["fast_native_pixel"]
        new,receipt=snapshot("FAST_DIPS",src,"epsilon",33.26)
        self.assertEqual(new["epsilon"],33.26)
        self.assertEqual(src["epsilon"],.05)
        self.assertEqual({k for k in src if src[k]!=new[k]},{"epsilon"})
        self.assertEqual(receipt["changed_key_count"],1)
        self.assertFalse(receipt["complete_native_source_diff_gate_passed"])
        self.assertEqual(src["T"],150)
        self.assertEqual(src["K_ADMM"],2)
    def test_DAPS_nontechnical_F8_gate_evidence_without_fake_XPK_replay(self):
        self.assertIn("8/8 pixel-exact PASS",F8.read_text())
        evidence=validate_F8_receipt()
        self.assertEqual(evidence["prior_F8_status"],
            "PRIOR_NATIVE_DAPS_8ROOT_EXACT_REPLAY_DOCUMENTED")
        self.assertEqual(evidence["XP_K_DAPS_original_parameter_two_B22_root_replay"],
            "NOT_RUN")
        original=json.loads(FROZEN_DAPS.read_text())
        self.assertEqual(original["scheduler"]["annealing_steps"],400)
        self.assertEqual(original["scheduler"]["diffusion_steps_per_anneal"],5)
        self.assertNotIn("tau",original["scheduler"])
    def test_pinned_native_head_ids(self):
        self.assertEqual(EXPECTED_HEADS["SITCOM"],
            "275ab67efbd8146bffca20155171ba6be1169c09")
        self.assertEqual(EXPECTED_HEADS["FAST_DIPS"],
            "a9758b38906b7c7950fa04491f37f6634948db26")
        self.assertEqual(EXPECTED_HEADS["DAPS_FFHQ"],
            "e7a77d094167084faed19b599b96673b7bb11447")
    def test_launcher_stays_cpu_only_and_new_output(self):
        if not LAUNCH.exists():return
        s=LAUNCH.read_text()
        self.assertIn('export CUDA_VISIBLE_DEVICES=""',s)
        self.assertIn("xp_a39_a24_xpk_cpu_preflight.py",s)
        self.assertIn("test_track_r_xp_a39_a24_xpk_cpu.py",s)
        self.assertNotIn("nvidia-smi",s)
        self.assertNotIn("xp_k_gpu",s)

if __name__=="__main__":
    unittest.main()
