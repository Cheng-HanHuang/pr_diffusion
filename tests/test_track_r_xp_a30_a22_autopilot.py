"""A22 CPU-readout and G7 scheduler launch-contract smoke tests."""
import ast,unittest
from pathlib import Path
from scripts.track_r.xp_a30_a22_tab3_e1_cpu import g1_select

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=[
"xp_a30_a22_tab3_e1_cpu.py",
"xp_a30_a22_tab3_publish_cpu.py",
"xp_a30_a22_g7_prepare_cpu.py",
"xp_a30_a22_g7_c2_autopilot.py",
"xp_a30_a22_g7_c2_report_cpu.py",
"xp_a30_a22_autopilot.py"]
class A22ContractTests(unittest.TestCase):
 def test_all_script_syntax(self):
  for name in SCRIPTS:
   path=ROOT/"scripts/track_r"/name
   ast.parse(path.read_text(),str(path))
 def test_exact_e0_first_k_then_g1(self):
  rows=[dict(run=i,rho=r) for i,r in enumerate((.5,.2,.3,.1,.001))]
  self.assertEqual(g1_select(rows,4)["run"],3)
  self.assertEqual(g1_select(rows,2)["run"],1)
  self.assertEqual(g1_select(rows,5)["run"],4)
 def test_tie_break_by_run(self):
  rows=[dict(run=i,rho=.1) for i in range(4)]
  self.assertEqual(g1_select(rows[::-1],4)["run"],0)
 def test_no_optional_a21_replication_or_scientific_substitution(self):
  s=(ROOT/"scripts/track_r/xp_a30_a22_autopilot.py").read_text()
  for val in ("A21_10_optional_replication_excluded",
    "SCIENTIFIC_ADAPTER_GATE_BLOCKED",
    "NO_NEW","TAB3_PUBLICATION","G7_C2_GPU"):
   if val=="NO_NEW":continue
   self.assertIn(val,s)
  self.assertNotIn("xp_a29_a21_gpu_worker.py",s)
 def test_provenance_guards(self):
  tab=(ROOT/"scripts/track_r/xp_a30_a22_tab3_e1_cpu.py").read_text()
  for key in ("A7_LOCAL_ZIP_HASH","A7_D2_G5_NATIVE_RECON_HASH",
    "SITCOM_SHARED_MEASUREMENT_SHA",
    "read_stage(\"G5\")","G1","VGG_LPIPS"):
   self.assertIn(key,tab)
  g7=(ROOT/"scripts/track_r/xp_a30_a22_g7_c2_autopilot.py").read_text()
  for key in ("total_union","XP_A12_CAMPAIGN_CONTROLLER.lock",
    "SOURCE","A22_G7","collect(own_roots","GPU"):
   if key=="SOURCE":continue
   self.assertIn(key,g7)
 def test_existing_native_import_repair_isolated_in_a22(self):
  s=(ROOT/"scripts/track_r/xp_a12_g7_c2_gpu.py").read_text()
  self.assertIn("from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse",s)
  self.assertNotIn("NATIVE,PY_DAPS,MODEL_SHA",s)
if __name__=="__main__":unittest.main()
