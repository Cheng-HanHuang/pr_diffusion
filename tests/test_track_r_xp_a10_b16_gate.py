"""A9 B16 400-root replay manifest partition and fail-closed contract tests.

Pure synthetic metadata: no GT, PAC data, GPU or native solver.
"""
from __future__ import annotations
import unittest
from unittest.mock import patch
from scripts.track_r import xp_a10_b16_gate_prepare_cpu as prep
from scripts.track_r.xp_a10_b16_gate_gpu import native_adapter
from types import SimpleNamespace
from scripts.track_r.xp_a10_b16_gate_autopilot import cpu_smoke
from tempfile import TemporaryDirectory
from pathlib import Path
from scripts.track_r.xp_a10_b16_gate_autopilot import SAMPLE_REFERENCE_SEC,SAMPLE_REFERENCE_SHA

def fake_catalog():
    roots={};scores={}
    for image in range(100):
        iid=f"{image:05d}"
        for rep in range(4):
            uid=f"S1_b22|SITCOM|{iid}|run{rep}"
            r={"uid":uid,"source":"S1_b22","family":"SITCOM",
               "image_id":iid,"run_index":rep,
               "measurement_path":f"/fake/y_{iid}.pt",
               "measurement_sha256":"yhash",
               "candidate_path":f"/fake/c_{iid}_{rep}.pt",
               "candidate_sha256":"chash",
               "gt_path":f"/fake/{iid}.png","gt_format":"png"}
            roots[uid]=r
            scores[uid]={"source":"S1_b22","family":"SITCOM",
                         "image_id":iid,"run_index":rep,
                         "psnr_raw":20.,"raw_label":"NONORBIT_INCONSISTENT",
                         "tau_signed_good_q95":0.045}
    return list(roots.values()),scores

class TestB16FullGate(unittest.TestCase):
    def test_partition_all_100_images_400_roots(self):
        catalog,scores=fake_catalog()
        with patch.object(prep,"source_sha",side_effect=lambda x:
               "yhash" if "/y_" in str(x) else "chash"),patch.object(
               prep,"native_seed_metadata",return_value=(43,{"native_tensor_hash":"x"*64})):
            roots,offline=prep.construct(catalog,scores)
        self.assertEqual(len(roots),len(offline))
        self.assertEqual(len(roots),400)
        self.assertEqual([r["rep"] for r in roots[:8]],[0,1,2,3]*2)
        self.assertEqual(len(set(r["image_id"] for r in roots)),100)
        self.assertNotIn("gt_path",roots[0])
        self.assertNotIn("historical_candidate_path",roots[0])
        self.assertEqual(offline[0]["historical_raw_label"],
                         "NONORBIT_INCONSISTENT")
    def test_ffhq_smoke_timing_reference_not_imagenet(self):
        # Exact published FFHQ median for XP-T SITCOM 9-root calibration,
        # while ImageNet's published median is 58.34277 s.
        self.assertAlmostEqual(SAMPLE_REFERENCE_SEC,32.104482267051935,places=10)
        self.assertEqual(SAMPLE_REFERENCE_SHA,
                         "bb2c3c377f07be2982e412eb96ff1a8b35805146dc021cc8baab3abd6ddc075e")
        samples=(25.193,23.027,22.603,22.542)
        median=sum(sorted(samples)[1:3])/2
        self.assertGreaterEqual(median,0.5*SAMPLE_REFERENCE_SEC)
        self.assertLessEqual(median,1.5*SAMPLE_REFERENCE_SEC)
        self.assertLess(median,0.5*58.342772816773504)

    def test_historical_smoke_is_two_independent_cpu_checks(self):
        with TemporaryDirectory() as tmp:
            run=Path(tmp)
            with patch("scripts.track_r.xp_a10_b16_gate_autopilot.subprocess.run") as invoke:
                invoke.return_value.returncode=0
                cpu_smoke(run,"pinned")
            self.assertEqual(invoke.call_count,2)
            self.assertTrue((run/"logs"/"B16_HISTORICAL_SMOKE_000.log").is_file())
            self.assertTrue((run/"logs"/"B16_HISTORICAL_SMOKE_001.log").is_file())
            self.assertEqual((run/"B16_GATE_EVENTS.jsonl").read_text().count(
                             "B16_HISTORICAL_8ROOT_SMOKE_PASS"),1)

    def test_historical_smoke_failure_blocks_progress(self):
        with TemporaryDirectory() as tmp:
            run=Path(tmp)
            with patch("scripts.track_r.xp_a10_b16_gate_autopilot.subprocess.run") as invoke:
                invoke.return_value.returncode=8
                with self.assertRaisesRegex(RuntimeError,"HISTORICAL_SMOKE_0"):
                    cpu_smoke(run,"pinned")
            self.assertEqual(invoke.call_count,1)
            self.assertFalse((run/"B16_GATE_EVENTS.jsonl").exists())

    def test_native_adapter_contract(self):
        frozen={"source_head":prep.SITCOM_HEAD,"rows":[{"uid":"sample"}]}
        args,source=native_adapter(
            SimpleNamespace(out="/unused",gpu=1,expected_head="abc123"),
            frozen)
        self.assertEqual(args.expected_head,"abc123")
        self.assertEqual(args.mode,"b22")
        self.assertEqual(args.gpu,1)
        self.assertEqual(source["native_source_head"],prep.SITCOM_HEAD)
        self.assertEqual(source["rows"],frozen["rows"])
        self.assertNotIn("native_source_head",frozen)
        with self.assertRaisesRegex(RuntimeError,"NATIVE_SOURCE_ALIAS"):
            native_adapter(SimpleNamespace(out="/unused",gpu=0,
                           expected_head="abc"),{"source_head":"wrong"})

    def test_git_porcelain_preserves_leading_status_column(self):
        with patch.object(prep.subprocess,"check_output",
                          return_value=" M model/__pycache__/precond.cpython-38.pyc\n"):
            status=prep.git("/mock/SITCOM_ODE","status","--porcelain=v1",
                            "--untracked-files=no")
        self.assertEqual(status,
                         " M model/__pycache__/precond.cpython-38.pyc")
        self.assertEqual(status[:2]," M")
        self.assertTrue(
            prep.check_native_checkout(
                prep.SITCOM_HEAD,status,"")["native_solver_model_operator_source_clean"]
        )

    def test_missing_run_fails_closed(self):
        catalog,scores=fake_catalog()
        catalog.pop()
        with self.assertRaisesRegex(RuntimeError,"HIST_COUNT"):
            prep.construct(catalog,scores)
    def test_duplicate_run_fails_closed(self):
        catalog,scores=fake_catalog()
        catalog[1]["run_index"]=0
        with patch.object(prep,"source_sha",return_value="yhash"),patch.object(
                prep,"native_seed_metadata",return_value=(43,{})):
            with self.assertRaisesRegex(RuntimeError,"RUNS_0_3"):
                prep.construct(catalog,scores)

if __name__=="__main__":
    unittest.main()
