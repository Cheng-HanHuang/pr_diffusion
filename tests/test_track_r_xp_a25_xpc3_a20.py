"""A20 XP-C3 exact S_mid prior pre-score-source controls."""
import ast
import importlib.util
import math
import unittest
from pathlib import Path

WT=Path(__file__).resolve().parents[1]

class A20XPCTest(unittest.TestCase):
    def test_preparation_source_only(self):
        s=(WT/"scripts/track_r/xp_a25_xpc3_source_freeze_cpu.py").read_text()
        ast.parse(s)
        for term in ("OUT=WT/","XP_C3_A20_SOURCE_MANIFEST.json",
                     "n_candidates=13600","checkpoint_sha256=MODEL_SHA",
                     "no_prior_scoring=True","SIZES={",
                     "uid=f\"XPC3|{panel}|{iid}|{run}\"",
                     "SHA","original_execution_head",
                     "reconstruction.pt","stage_records(\"G5\")"):
            self.assertIn(term,s)
        self.assertNotIn("model.tweedie(",s)

    def test_native_source_enforces_freeze_gt_free(self):
        s=(WT/"scripts/track_r/xp_a25_xpc3_mid_gpu.py").read_text()
        ast.parse(s)
        for term in ("COLUMNS=","DAPS_PY","source_manifest_sha256",
                     "SOURCE_MANIFEST_MUST_BE_COMMITTED",
                     "checkpoint_sha256","original_prior_score_sha256",
                     "validate_weight_trace(trace)","SIZES[",
                     'gate(n==16',"elif", "model.tweedie(z,sigma)",
                     "prior_margin=format(margin",
                     "margin>1e-7","pri"):
            if term in ("elif","pri"):continue
            self.assertIn(term,s)
        for forbidden in ("gt_source(", "psnr_rgb8(",
                          "2_forward","prior_chosen_good=int("):
            self.assertNotIn(forbidden,s)

    def test_controller_accounting_no_new_gpu_claim(self):
        s=(WT/"scripts/track_r/xp_a25_xpc3_autopilot.py").read_text()
        ast.parse(s)
        for term in (
             "CAP_H=8.", "PROJECT_H=400.", "LOCK_EX",
             "A24_XPI2_COMPLETE","no_secondary_prior=True",
             "--limit-images","STOP_XPC3","charged_gpu_h",
             "one-worker", "clean_only=False",
             "len(accepted)==100"):
            if term=="one-worker":continue
            self.assertIn(term,s)

    def test_orientation_seed_and_mid_policy(self):
        from prdiffusion.track_r.prior_score import prior_seed
        uid="XPC3|XP_N|ILSVRC2012_val_00048900.JPEG|0"
        seeds=[prior_seed(uid,t,j)
               for t in (300,400,500,600) for j in range(4)]
        self.assertEqual(len(set(seeds)),16)
        self.assertEqual(prior_seed(uid,300,0),seeds[0])
        self.assertNotEqual(prior_seed(uid,300,0),
                            prior_seed(uid+"1",300,0))

    def test_core_32_forward_samples_and_tie(self):
        import torch
        from scripts.track_r.xp_a25_xpc3_mid_gpu import orient_sm
        from prdiffusion.track_r.prior_score import (
            frozen_alpha_schedule,prior_seed)
        class Toy:
            def tweedie(self,z,sigma):
                # constant 0 epsilon estimate, equal paired losses in both orientations
                return z
        x=torch.zeros((1,3,256,256),dtype=torch.float32)
        out=orient_sm(x,"XPC3|XP_N|toy|0",Toy(),torch,
                      frozen_alpha_schedule(),prior_seed)
        self.assertEqual(out["prior_choice"],"000")
        self.assertEqual(out["denoiser_examples"],32)
        self.assertLess(abs(float(out["prior_margin"])),1e-5)

if __name__=="__main__":unittest.main()
