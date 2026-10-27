"""CPU-only XP-T DAPS-4K worker/smoke/supervisor regression tests."""
from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

from scripts.track_r import xp_t_daps4k_worker_gpu as wk
from scripts.track_r import xp_t_daps4k_supervisor as sup

def good_trace():
    return {"unet_calls":1,"strict":True,
            "missing_keys":[],"unexpected_keys":[],
            "tensor_count":410}

class DAPS4KNativeTests(unittest.TestCase):
    def test_strict_weight_load_proof(self):
        self.assertTrue(wk.validate_weight_trace(good_trace()))
        for problem in ("unet_calls","strict","missing_keys",
                        "unexpected_keys","tensor_count"):
            trace=good_trace()
            trace[problem]={
                "unet_calls":0,"strict":False,
                "missing_keys":["missing"],"unexpected_keys":["extra"],
                "tensor_count":1}[problem]
            with self.subTest(problem=problem):
                with self.assertRaisesRegex(RuntimeError,
                     "UNVERIFIED_MODEL_CHECKPOINT"):
                    wk.validate_weight_trace(trace)

    def test_four_k_configuration_and_distinct_checkpoint_architecture(self):
        ff={"checkpoint":"/tmp/ffhq.pt","num_channels":128,
            "num_res_blocks":1,"attention_resolutions":16}
        im={"checkpoint":"/tmp/imagenet.pt","num_channels":256,
            "num_res_blocks":2,"attention_resolutions":"32,16,8"}
        a=wk.model_config(ff);b=wk.model_config(im)
        self.assertEqual(a["num_channels"],128)
        self.assertEqual(b["num_channels"],256)
        self.assertNotEqual(a["model_path"],b["model_path"])
        self.assertEqual(wk.DAPS4K["annealing_scheduler_config"]["num_steps"],400)
        self.assertEqual(wk.DAPS4K["diffusion_scheduler_config"]["num_steps"],10)
        self.assertEqual(wk.DAPS4K["mcmc_sampler_config"]["num_steps"],100)

    def test_worker_process_isolated_gpu_no_experimental_guidance(self):
        e=sup.worker_env(3,10)
        self.assertEqual(e["CUDA_VISIBLE_DEVICES"],"3")
        self.assertEqual(e["B20_LF_ENABLE"],"0")
        self.assertEqual(e["B21_CONT_ENABLE"],"0")
        self.assertEqual(e["PYTHONDONTWRITEBYTECODE"],"1")
        self.assertIn("/worker_10",e["PYTHONPYCACHEPREFIX"])
        self.assertEqual(
            sup.worker_args("verified-head",10,3)[-4:],
            ["--index","10","--gpu","3"])

    def test_smoke_budget_projection(self):
        a={"dataset":"ffhq","native_sampling_wall_s":100.0}
        b={"dataset":"imagenet","native_sampling_wall_s":200.0}
        value=sup.smoke_projection(a,b,6.0)
        self.assertAlmostEqual(value,9*300*1.75/3600)
        with self.assertRaisesRegex(RuntimeError,"PROJECTED_BUDGET"):
            sup.smoke_projection(a,b,0.1)
        with self.assertRaisesRegex(RuntimeError,"SMOKE_DATASET"):
            sup.smoke_projection(b,a,6.0)

    def test_exclusive_json_and_atomic_status(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"summary.json"
            sup.dump_x(p,{"ok":1})
            self.assertEqual(json.loads(p.read_text()),{"ok":1})
            with self.assertRaises(FileExistsError):
                sup.dump_x(p,{"ok":2})
            sup.update(p,{"ok":3})
            self.assertEqual(json.loads(p.read_text()),{"ok":3})

    def test_daps_root_frozen_hash_namespace(self):
        row={"uid":"XP_T|DAPS|imagenet|IMG|rep1"}
        token=hashlib.sha256(row["uid"].encode()).hexdigest()[:24]
        self.assertEqual(wk.root_folder(row).name,"root_"+token)
        self.assertEqual(wk.root_folder(row).parent,sup.RUN/"gpu")

if __name__=="__main__":
    unittest.main()
