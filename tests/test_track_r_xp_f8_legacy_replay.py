"""Static tests for native historical DAPS replay adapter and A10 controller."""
from __future__ import annotations
import ast
import inspect
import unittest
from pathlib import Path
from prdiffusion.track_r.xp_gpu_telemetry import GPU
from scripts.track_r.xp_f8_legacy_replay_gpu import (
    native_command, env, TRACE_SOURCE,MODEL,FROZEN_MANIFEST_SHA,PRECHECK_SHA)
from scripts.track_r.xp_f8_legacy_replay_supervisor import (
    foreign_track_r_pid,REQUIRED_MIB,MARGIN_MIB,frozen_preflight)
from scripts.track_r.xp_f8_legacy_replay_gpu import check_cfg,do_gpu

class TestF8LegacyReplayAdapter(unittest.TestCase):
    def test_native_exact_400_by_five(self):
        row=dict(native_seed=22020,measurement_path=
                 "/egr/research-pac/huang248/outputs/mock_measurement.pt")
        cmd=native_command(row,Path("/tmp/replay_00"))
        self.assertIn("sampler.annealing_scheduler_config.num_steps=400",cmd)
        self.assertIn("sampler.diffusion_scheduler_config.num_steps=5",cmd)
        self.assertIn("+data={name:empty,shape:[3,256,256],device:cuda}",cmd)
        self.assertIn("+measurement_path="+row["measurement_path"],cmd)
        self.assertIn("seed=22020",cmd)
        self.assertIn("model.model_config.model_path="+str(MODEL),cmd)
        self.assertTrue(any(x.startswith("hydra.run.dir=/tmp/replay_00/")
                            for x in cmd))
        self.assertFalse(any("gt_path" in x or "test-ffhq" in x for x in cmd))
    def test_historical_native_env_controls(self):
        row=dict(native_seed=23035,measurement_path="/tmp/example_y.pt")
        e=env(row,gpu=3)
        self.assertEqual(e["CUDA_VISIBLE_DEVICES"],"3")
        self.assertEqual(e["B21_START_NOISE_SEED"],"23035")
        self.assertEqual(e["B21_SOURCE_SEED"],"23035")
        self.assertEqual(e["B21_MEASUREMENT_PATH"],row["measurement_path"])
        self.assertEqual(e["B20_LF_ENABLE"],"0")
        self.assertNotIn("B21_CONT_ENABLE",e)
        self.assertNotIn("B21_CONT_NOISE_SEED",e)
    def test_strict_checkpoint_instrumentation(self):
        ast.parse(TRACE_SOURCE)
        self.assertIn("torch.nn.Module.load_state_dict=traced_load",TRACE_SOURCE)
        self.assertIn('self.__class__.__name__=="UNetModel"',TRACE_SOURCE)
    def test_hydra_cotenancy_and_per_root_gate(self):
        # The original 4x serial 90s Hydra subprocesses stalled at
        # historical index1. Wave checks one representative config,
        # then every GPU child checks its own resolved config before
        # starting any native sampling; no scientific configuration
        # verification is removed.
        preflight=inspect.getsource(frozen_preflight)
        self.assertIn("i=indices[0]",preflight)
        self.assertIn("timeout=360",preflight)
        worker=inspect.getsource(do_gpu)
        self.assertLess(worker.index("cfg=check_cfg("),
                        worker.index("subprocess.run(command"))
        self.assertIn("timeout=300",inspect.getsource(check_cfg))

    def test_four_gpu_memory_contract(self):
        self.assertEqual(REQUIRED_MIB,24576)
        self.assertEqual(MARGIN_MIB,2048)
        self.assertEqual(len(FROZEN_MANIFEST_SHA),64)
        self.assertEqual(len(PRECHECK_SHA),64)
        card=GPU(index=2,uuid="g2",total_mib=97887,
                 used_mib=5000,free_mib=92887,
                 utilization_pct=95,process_pids=(),
                 foreign_pids=(),visibility="known")
        self.assertEqual(foreign_track_r_pid(card),[])

if __name__=="__main__":unittest.main()
