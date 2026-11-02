"""A18 28-GPUh continuation budget and CP-N part1 source regression."""
from __future__ import annotations
import ast,math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class PlannerA18Tests(unittest.TestCase):
 def test_new_scripts_parse_and_avoid_held_scores(self):
  for filename in ("xp_a22_xpn_continue_gpu.py","xp_a22_cpn_part1_cpu.py"):
   path=ROOT/"scripts/track_r"/filename
   ast.parse(path.read_text(),filename=str(path))
  continuation=(ROOT/"scripts/track_r/xp_a22_xpn_continue_gpu.py").read_text()
  self.assertNotIn("torch.load",continuation)
  self.assertNotIn("gt_path",continuation)
  self.assertNotIn("reconstruction.pt",continuation)
  self.assertIn("check_chunk(",continuation)
  self.assertIn("A22_XPN_ONE_CHUNK_SMOKE_PASS",continuation)
  self.assertIn("A22_28H_ATTEMPT_CHARGED",continuation)
  self.assertIn("project_used_at_start+projected-stage_used_at_start",continuation)
  self.assertIn('if not active and not can_admit()',continuation)
  self.assertIn("git(wt,\"rev-parse\",\"HEAD\")",continuation)
  self.assertIn("all_held_GT_outcomes_unread=True",continuation)

 def test_budget_reservation_only_admits_new_workers(self):
  from scripts.track_r.xp_a22_xpn_continue_gpu import (
    STAGE_CAP_H,PROJECT_CAP_H,may_admit_new,hard_watchdog_triggered)
  self.assertEqual(STAGE_CAP_H,28.)
  self.assertEqual(PROJECT_CAP_H,400.)
  old=18.7562
  proj=225.
  self.assertTrue(may_admit_new(old,4,proj,old))
  self.assertTrue(may_admit_new(27.0,1,proj,old))
  self.assertFalse(may_admit_new(27.5,1,proj,old))
  self.assertFalse(hard_watchdog_triggered(27.5,proj+27.5-old))
  self.assertTrue(hard_watchdog_triggered(27.999,proj+27.999-old))
  self.assertTrue(hard_watchdog_triggered(19.,399.999))

 def test_cpu_part1_splits_native_0_3_and_4_15_only(self):
  from scripts.track_r.xp_a22_cpn_part1_cpu import (
      TAU_MAIN,TAU_SECONDARY,FORECASTS,merge)
  self.assertAlmostEqual(TAU_MAIN,.0500829)
  self.assertAlmostEqual(TAU_SECONDARY,.050434426327793)
  self.assertEqual(FORECASTS["P-N1_gain_at_16_pp"],(.3,-3.6,4.2))
  self.assertEqual(FORECASTS["P-N3_label_free_gain_at_16_pp"],(.3,-2.0,2.6))
  self.assertEqual(FORECASTS["P-N4_never_orbit_at_16_pct"],(25.,13.,37.))
  self.assertEqual(FORECASTS["theory_N_H1a_gain_pp"],(.2,-1.8,2.2))
  code=(ROOT/"scripts/track_r/xp_a22_cpn_part1_cpu.py").read_text()
  self.assertIn('range(4,16)',code)
  self.assertIn('range(0,total,step)',code)
  self.assertIn('nmax","16"',code)
  self.assertIn('run-range","0:16"',code)
  self.assertIn("v5_not_scored_at_K16=True",code)
  self.assertIn("GOOD=20.",code)
  self.assertIn('GT_FILE',code)
  self.assertNotIn("A18_XPN_LATE_16_63",code)
  ids=[f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(48900,49000)]
  p4=[];c4=[];new=[];cnew=[]
  for iid in ids:
   p4.append(dict(image_id=iid,run_labels="GOPC",
      rho_signed_by_run=".01;.02;.03;.04"))
   for rep,l in enumerate("GOPC"):
    c4.append(dict(image_id=iid,run=str(rep),label=l,
        g1_choice="000",g1_margin=".1",g1_chosen_good=1 if l=="G" else 0))
   for rep in range(4,16):
    new.append(dict(image_id=iid,run=rep,label="I",rho_signed=".06"))
    cnew.append(dict(image_id=iid,run=rep,label="I",
                     g1_choice="000",g1_margin=".1",g1_chosen_good=0))
  strings,clips=merge(p4,c4,new,cnew)
  self.assertEqual(len(strings),100)
  self.assertEqual(len(clips),1600)
  self.assertEqual(strings[0]["labels_by_run"].split(";")[:4],
                   ["G","O_g","O_p","C"])
  self.assertEqual(set(strings[0]["labels_by_run"].split(";")[4:]),{"I"})

 def test_no_cpu_publication_of_gt_paths(self):
  script=(ROOT/"scripts/track_r/xp_a22_cpn_part1_publish.sh").read_text()
  self.assertIn("XP_N_CPN16_STRINGS.csv",script)
  self.assertIn("XP_N_CPN16_G1_DECISIONS.csv",script)
  self.assertIn("CPN_PART1_PLANNER_RETURN.txt",script)
  self.assertNotIn('cp "$OUT/LOCAL_ONLY_CP_N_1200_RAW_SCORES.csv"',script)
  self.assertIn("ls-remote",script)

if __name__=="__main__":unittest.main()
