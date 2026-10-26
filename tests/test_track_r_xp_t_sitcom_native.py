"""CPU-only fail-closed SITCOM XP-T native worker and four-GPU tests."""
from __future__ import annotations
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_t_sitcom_worker_gpu as gpu
from scripts.track_r import xp_t_sitcom_supervisor as supervisor

class StrictWeights(unittest.TestCase):
    def test_strict_success(self):
        trace={"unet_calls":1,"missing_keys":[],"unexpected_keys":[],
               "tensor_count":500,"strict":True}
        self.assertEqual(gpu.verify_checkpoint_trace(trace),trace)
    def test_model_load_fallback_never_accepted(self):
        examples=[
           {"unet_calls":0},
           {"unet_calls":2,"tensor_count":600,"strict":True,
            "missing_keys":[],"unexpected_keys":[]},
           {"unet_calls":1,"tensor_count":400,"strict":False,
            "missing_keys":[],"unexpected_keys":[]},
           {"unet_calls":1,"tensor_count":400,"strict":True,
            "missing_keys":["x"],"unexpected_keys":[]},
           {"unet_calls":1,"tensor_count":400,"strict":True,
            "missing_keys":[],"unexpected_keys":["y"]},
        ]
        for item in examples:
            with self.subTest(item=item):
                with self.assertRaisesRegex(RuntimeError,
                                      "STOP_XPT_SITCOM_UNVERIFIED_MODEL_WEIGHTS"):
                    gpu.verify_checkpoint_trace(item)
    def test_separate_native_architectures(self):
        ff={"checkpoint_path":"/research/ffhq.pt","num_channels":128,
            "num_res_blocks":1,"attention_resolutions":16}
        im={"checkpoint_path":"/research/imagenet.pt","num_channels":256,
            "num_res_blocks":2,"attention_resolutions":"32,16,8"}
        ff_cfg=gpu.config("ffhq",ff)
        im_cfg=gpu.config("imagenet",im)
        self.assertEqual(ff_cfg["num_channels"],128)
        self.assertEqual(ff_cfg["num_res_blocks"],1)
        self.assertEqual(im_cfg["attention_resolutions"],"32,16,8")
        self.assertEqual(im_cfg["num_res_blocks"],2)
        self.assertNotEqual(ff_cfg["model_path"],im_cfg["model_path"])
    def test_manifest_root_output_is_uid_hash(self):
        row={"uid":"XP_T|SITCOM|imagenet|ABC|rep0"}
        want=hashlib.sha256(row["uid"].encode()).hexdigest()[:24]
        self.assertEqual(gpu.record_folder(row).name,"root_"+want)
        self.assertEqual(gpu.record_folder(row).parent,supervisor.RUN/"gpu")
    def test_device_env_is_isolated_by_index(self):
        e0=supervisor.job_env(0,0)
        e1=supervisor.job_env(1,1)
        self.assertEqual(e0["CUDA_VISIBLE_DEVICES"],"0")
        self.assertEqual(e1["CUDA_VISIBLE_DEVICES"],"1")
        self.assertNotEqual(e0["PYTHONPYCACHEPREFIX"],e1["PYTHONPYCACHEPREFIX"])
        self.assertEqual(e0["CUBLAS_WORKSPACE_CONFIG"],":4096:8")
        self.assertEqual(e0["PYTHONDONTWRITEBYTECODE"],"1")
        self.assertEqual(e0["PYTHONNOUSERSITE"],"1")

if __name__=="__main__":
    unittest.main()
