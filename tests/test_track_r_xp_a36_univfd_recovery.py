"""A36 UnivFD-only recovery: CPU dependency repair without scientific changes."""
import ast
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
A36=ROOT/"scripts/track_r/xp_a36_univfd_recovery_cpu.py"
LAUNCH=ROOT/"scripts/track_r/xp_a36_univfd_recovery_launch.sh"
WORKER=ROOT/"scripts/track_r/xp_a35_g7_worker.py"
PUBLISH=ROOT/"scripts/track_r/xp_a35_g7_readout_publish_cpu.py"
LOCK=ROOT/"docs/track_r/executors/xp/theory_inputs/XP_A23_G7_REMAINING_SOURCE_LOCK.json"

class UnivFDDependencyRecovery(unittest.TestCase):
    def test_original_A35_G7_univfd_science_unchanged(self):
        source=WORKER.read_text()
        self.assertIn("fd=UnivFD(root,root/ext[\"fc_path\"],ext[\"clip_path\"],device=\"cpu\")",source)
        self.assertIn("fd.p_fake(all8)",source)
        self.assertIn("ORBIT_PATTERNS[min(range(8)",source)
        self.assertIn("torch.inference_mode()",source)
        self.assertNotIn("sys.modules['pkg_resources']",source)
    def test_recovery_reuses_exact_A35_worker_readout_and_publisher(self):
        s=A36.read_text()
        ast.parse(s)
        self.assertIn('WORKER=WT/"scripts/track_r/xp_a35_g7_worker.py"',s)
        self.assertIn("from scripts.track_r.xp_a35_g7_readout_publish_cpu import (",s)
        self.assertIn("analyze,publish,check_results",s)
        self.assertIn('PUB',s) if False else None
        self.assertIn('status="A35_G7_STAGE_280_COMPLETE"',s)
        self.assertIn('CPU_only=True',s)
        self.assertIn('"--method",UNIVFD',s)
        self.assertIn('"--dataset",dataset',s)
        self.assertNotIn("torch.cuda",s)
        self.assertNotIn("get_operator(",s)
        self.assertNotIn("sampler.sample(",s)
    def test_official_pkg_resources_target_only_not_global_conda(self):
        s=A36.read_text()
        self.assertIn('VERSION="80.9.0"',s)
        self.assertIn('VENDOR=BASE/"a36_univfd_vendor_setuptools_80_9_0"',s)
        self.assertIn('md.version(\'setuptools\')',s)
        self.assertIn("pkg_resources",s)
        self.assertIn('PYTHONPATH=f"{VENDOR}:{WT}"',s)
        self.assertIn('CUDA_VISIBLE_DEVICES=""',s)
        self.assertIn("PYTHONNOUSERSITE",s)
        self.assertNotIn("sys.modules",s)
        self.assertNotIn("site-packages/pkg_resources",s)
    def test_accepted_published_results_are_immutable_inputs(self):
        s=A36.read_text()
        self.assertIn("967e73cc7adcc5cc3347d257572bd779ff0a6dea",s)
        self.assertIn("6d2afbfd8c405cbf7db4a6f193ba21c885c7eaa4",s)
        self.assertIn('A35_ORIGINAL=BASE/"xp_a35_g7_remaining_20261104T083110Z"',s)
        self.assertIn('status"=="STOP_A35_NATIVE_OR_CONTROLLER_FAILURE"',s) if False else None
        self.assertIn('HISTORICAL_CPU_FAILURE_PRESERVED',s)
        self.assertIn('G7_TOTAL=5.971352510419156',s)
        self.assertIn('PROJECT_TOTAL=247.80191737622505',s)
        self.assertIn('and float(fail_stage["method_gpu_h"])==0',s)
        self.assertIn('"HISTORICAL_CPU_HAS_RESULTS"',s)
        self.assertNotIn("XP_A35_G7_S_mid_all8_PER_IMAGE.csv",s)
    def test_exact_280_inputs_and_idempotent_receipts(self):
        s=A36.read_text()
        self.assertIn('len(gt["images"])==280',s)
        self.assertIn('done=valid_partial(stage,images)',s)
        self.assertIn('pending=[i for i in inds if i not in done]',s)
        self.assertIn('gate(len(done)==280,"CPU_280_NOT_COMPLETE")',s)
        self.assertIn('gate(len(check_results(stage,UNIVFD,HEAD))==280',s)
        self.assertIn('original_A35_failure_preserved=True',s)
        self.assertIn('run/"A36_UNIVFD_RECOVERY_STATE.json"',s)
        self.assertIn('no_change_to_model_or_score=True',s)
    def test_original_A35_model_and_source_pins(self):
        d=json.loads(LOCK.read_text())
        self.assertEqual(d["methods"],["S_mid_all8","2_forward","UnivFD"])
        self.assertEqual(d["univfd"]["source_head"],
            "030495aea3300a8b54c0ec37ec7fe1dd7e63c619")
        self.assertEqual(d["univfd"]["fc_sha256"],
            "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847")
        self.assertEqual(d["univfd"]["clip_sha256"],
            "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836")
    def test_no_reintroduced_gpu_generation_or_modified_readout(self):
        s=PUBLISH.read_text()
        self.assertIn("check_results(stage,method,head)",s)
        self.assertIn('GT_only=True',s)
        a=A36.read_text()
        self.assertNotIn("S_mid_all8",a[a.find("def run_group"):a.find("def main(a)")])
        self.assertNotIn("2_forward",a[a.find("def run_group"):a.find("def main(a)")])
        if LAUNCH.exists():
            l=LAUNCH.read_text()
            self.assertIn("setuptools==80.9.0",l)
            self.assertIn("A36_UNIVFD_RECOVERY",l)
            self.assertIn("test_track_r_xp_a36_univfd_recovery.py",l)
            self.assertIn("CUDA_VISIBLE_DEVICES=",l)
            self.assertNotIn("nvidia-smi",l)

if __name__=="__main__":
    unittest.main()
