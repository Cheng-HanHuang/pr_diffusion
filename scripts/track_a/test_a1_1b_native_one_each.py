"""A1.1b native integration planning: no GPU, image, B24 data or torch imports."""
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from scripts.track_a.a1_1b_native_one_each import (
    METHODS, SYNTHETIC_ID, MAX_PER_METHOD_S, MAX_AGGREGATE_S,
    np_config, build_input_item, make_native_parent, run_start_in_approved_window,
)
from scripts.track_a.a1_protocol import seed32_numpy


class NativeOneEachPlanTest(unittest.TestCase):
    def test_fixed_four_separated_seeds(self):
        self.assertEqual(METHODS,("DAPS","SITCOM","NP_LF","NP_S2"))
        seeds=[seed32_numpy(method,SYNTHETIC_ID,0) for method in METHODS]
        self.assertEqual(len(set(seeds)),4)
        self.assertTrue(all(0<=s<2**32 for s in seeds))
        self.assertEqual(MAX_PER_METHOD_S,1500)
        self.assertEqual(MAX_AGGREGATE_S,3600)

    def test_np_frozen_config_override_only(self):
        frozen={"np1":{
            "num_steps":1000,"soft_candidates":5,"hard_candidates":1,
            "score_mode":"lf","score_reg_lambda":0,
            "score_reg_lambda_schedule":"constant","seed":100}}
        original_seed=frozen["np1"]["seed"]
        lf=np_config(frozen,"NP_LF",101)["np1"]
        s2=np_config(frozen,"NP_S2",202)["np1"]
        self.assertEqual((lf["score_mode"],lf["score_reg_lambda"],
                          lf["score_reg_lambda_schedule"]),
                         ("lf",0.0,"constant"))
        self.assertEqual((s2["score_mode"],s2["score_reg_lambda"],
                          s2["score_reg_lambda_schedule"]),
                         ("prev_l2",0.01,"pre_projection_only"))
        self.assertEqual(original_seed,frozen["np1"]["seed"])
        for bad in ("NP_S3","SITCOM","DAPS"):
            with self.assertRaises(ValueError):
                np_config(frozen,bad,100)
        with self.assertRaises(ValueError):
            np_config(frozen,"NP_LF",2**32)
        broken={"np1":dict(frozen["np1"],num_steps=999)}
        with self.assertRaises(RuntimeError):
            np_config(broken,"NP_LF",100)

    def test_run_window_is_dated_and_fails_closed(self):
        self.assertFalse(run_start_in_approved_window(
            datetime.fromisoformat("2026-10-19T16:29:59+00:00")))
        self.assertTrue(run_start_in_approved_window(
            datetime.fromisoformat("2026-10-19T16:30:00+00:00")))
        self.assertTrue(run_start_in_approved_window(
            datetime.fromisoformat("2026-10-19T20:00:00+00:00")))
        self.assertFalse(run_start_in_approved_window(
            datetime.fromisoformat("2026-10-19T20:00:01+00:00")))

    def test_native_parent_exists_before_rep0_and_is_nonoverwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for method in ("DAPS","SITCOM"):
                parent=make_native_parent(method,root=root)
                self.assertTrue(parent.is_dir())
                rep0=parent/"rep0"
                rep0.mkdir(exist_ok=False)  # matches historical child_spec semantics
                self.assertTrue(rep0.is_dir())
                with self.assertRaises(FileExistsError):
                    make_native_parent(method,root=root)
            for method in ("NP_LF","NP_S2","DAPS_MORE"):
                with self.assertRaises(RuntimeError):
                    make_native_parent(method,root=root)

    def test_build_procedural_manifest_no_external_data(self):
        probe={
            "image_path":"/synthetic/probe.png",
            "image_file_sha256":"sha_png",
            "ground_truth_tensor_path":"/synthetic/gt.pt",
            "ground_truth_tensor_sha256":"sha_gt",
            "measurement_path":"/synthetic/y.pt",
            "measurement_file_sha256":"sha_file",
            "measurement_tensor_sha256":"sha_y",
            "shape":[1,3,384,384],"dtype":"torch.float32",
        }
        i=build_input_item(probe)
        self.assertEqual(i["image_id"],"11111")
        self.assertEqual(i["split"],"A1_1A_PROCEDURAL_NON_FFHQ")
        self.assertEqual(i["measurement_tensor_sha256"],"sha_y")
        self.assertEqual(i["ground_truth_tensor_sha256"],"sha_gt")
        self.assertNotIn("ffhq_data_root",i)
        self.assertNotIn("b24_confirmation",i)

if __name__=="__main__":
    unittest.main()
