"""Synthetic zero-GPU tests for B16 400-root -> 2160-root autonomous handoff."""
from __future__ import annotations
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_a11_b16_prod_prepare_cpu as prep
from scripts.track_r.xp_a11_b16_handoff_autopilot import (
    B22_SMOKE,B24_SMOKE,B32_B22_SMOKE,B32_B24_SMOKE,
    REFERENCE_SEC,validate_sample_speed_for_smoke,
    RUNNER_WT,launch,A10_PLANNER_SHA,A10_DEADLINE_EPOCH,A10_GPU_H)

class TestB16Production(unittest.TestCase):
    def test_frozen_seed_domain_matches_historical_algorithm(self):
        image="01113"
        label="B"
        expected=[]
        for r in range(4):
            content="|".join(["B24_DEV80_SITCOM_SOLVER_V1",
                    prep.B24_SCREEN_SHA,label,image,str(r)])
            expected.append(int(hashlib.sha256(content.encode()).hexdigest()[:16],16)&(
                       (1<<63)-1))
        self.assertEqual([prep.seed63(label,image,r) for r in range(4)],expected)
        self.assertEqual(prep.unique_stored_seed(image,
               [x%(2**32) for x in expected]),label)
        with self.assertRaisesRegex(RuntimeError,"B24_SEED_DOMAIN"):
            prep.unique_stored_seed(image,[1,2,3,4])

    def test_build_exact_2160_new_roots_and_no_outcomes(self):
        b22=[]
        for n in range(100):
            iid=f"{60000+n:05d}"
            for rep in range(4):
                b22.append(dict(uid=f"S1_b22|SITCOM|{iid}|run{rep}",
                  source="S1_b22",family="SITCOM",image_id=iid,
                  rep=rep,seed=43,rng_mode="sequential_post_model",
                  measurement_source="S1_b22",
                  measurement_path=f"/fake/{iid}.pt",
                  measurement_sha256="y"))
        catalog=[]
        for n in range(80):
            iid=f"{n+1000:05d}"
            for rep in range(4):
                catalog.append(dict(uid=f"S2_b24_dev80|SITCOM|{iid}|rep{rep}",
                    source="S2_b24_dev80",family="SITCOM",
                    image_id=iid,run_index=rep,
                    measurement_path=f"/fake/{iid}.pt",
                    measurement_sha256="y"))
        def historical_seed(row,mode):
            assert mode=="independent_model_seed"
            return prep.seed63("C",row["image_id"],int(row["run_index"]))%(2**32),{}
        with patch.object(prep,"native_seed_metadata",side_effect=historical_seed):
            b22groups,b24groups=prep.build(b22,catalog)
        self.assertEqual(len(b22groups),100)
        self.assertEqual(len(b24groups),960)
        b32_b22,b32_b24=prep.build_b32(b22groups,b24groups)
        self.assertEqual(len(b32_b22),100)
        self.assertEqual(len(b32_b24),1280)
        self.assertEqual(sum(g["new_roots"] for g in b32_b22+b32_b24),
                         2880)
        self.assertEqual([r["rep"] for r in b32_b22[0]["rows"]],
                         list(range(32)))
        self.assertEqual(b32_b24[0]["rows"][0]["rep"],16)
        self.assertEqual(b32_b24[-1]["rows"][0]["rep"],31)
        self.assertEqual(sum(x["new_roots"] for x in b22groups+b24groups),2160)
        self.assertEqual([r["rep"] for r in b22groups[0]["rows"]],
                         list(range(16)))
        self.assertEqual(len(set(r["uid"] for x in b22groups
                                 for r in x["rows"][4:])),1200)
        self.assertEqual(len(set(x["rows"][0]["uid"] for x in b24groups)),960)
        self.assertTrue(all("gt_path" not in r and
                            "historical_candidate_path" not in r
                            for x in b22groups+b24groups for r in x["rows"]))
        self.assertEqual(b22groups[0]["rows"][4]["seed"],43)
        self.assertEqual(b24groups[0]["rows"][0]["seed"],
                         prep.seed63("C","01000",4)%(2**32))

    def test_a10_explicit_budget_and_deadline(self):
        from datetime import datetime,timezone
        self.assertEqual(A10_PLANNER_SHA,
            "4144e0bc2cc073b05260b18781bd620f36e37763")
        self.assertEqual(A10_GPU_H,70.)
        self.assertEqual(datetime.fromtimestamp(
            A10_DEADLINE_EPOCH,timezone.utc).isoformat(),
            "2026-10-30T16:00:00+00:00")

    def test_recovery_launch_uses_original_worker_not_controller(self):
        self.assertEqual(RUNNER_WT,
                         prep.ROOT/"pr_diffusion_track_r_xp_handoff_20261028")
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            (run/"logs").mkdir()
            tasks=[{"task_id":"b22_062_dummy","new_roots":12}]
            with patch("scripts.track_r.xp_a11_b16_handoff_autopilot.subprocess.Popen") as popen:
                popen.return_value.pid=12345
                item=launch(run,"079228b3ebe55dba2ac7bc7ae2c8187113290409",
                            tasks,0,2,1)
            self.assertEqual(item["gpu"],2)
            args,kw=popen.call_args
            self.assertEqual(args[0][1],
                             str(RUNNER_WT/"scripts/track_r/xp_a11_b16_prod_worker_gpu.py"))
            self.assertEqual(kw["env"]["PYTHONPATH"],str(RUNNER_WT))
            self.assertEqual(kw["env"]["CUDA_VISIBLE_DEVICES"],"2")

    def test_slow_native_production_is_not_a_scientific_receipt_failure(self):
        # Observed Oct 28: 77.5 s native CUDA-event times with correct
        # hashes, unchanged method, 200 native steps, GT never read.
        for index in (58,59,60,61):
            self.assertFalse(validate_sample_speed_for_smoke(
                index,77.5,"S1_b22"))
        for index in (0,1,100,101,112,113,1060,1061,1160,1161,1176,1177):
            with self.assertRaisesRegex(RuntimeError,"FFHQ_SMOKE_SPEED"):
                validate_sample_speed_for_smoke(index,77.5,"S1_b22")
        self.assertFalse(validate_sample_speed_for_smoke(
            58,22.8,"S1_b22"))
        self.assertTrue(validate_sample_speed_for_smoke(
            0,22.8,"S1_b22"))

    def test_smoke_sets_and_ffhq_timing_frozen(self):
        self.assertEqual(B22_SMOKE,{0,1})
        self.assertEqual(B24_SMOKE,{100,101,112,113})
        self.assertEqual(B32_B22_SMOKE,{1060,1061})
        self.assertEqual(B32_B24_SMOKE,{1160,1161,1176,1177})
        self.assertAlmostEqual(REFERENCE_SEC,32.104482267051935)
        self.assertTrue(16.05 < 22.8 < 1.5*REFERENCE_SEC)

if __name__=="__main__":
    unittest.main()
