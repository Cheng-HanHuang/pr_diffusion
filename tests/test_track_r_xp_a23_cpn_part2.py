"""A18 Part2: full64 100-image frozen inputs and v5 all-K coverage tests."""
from __future__ import annotations
import ast,math
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class A23Part2Tests(unittest.TestCase):
 def test_cpu_only_source_and_frozen_forecast_guards(self):
  p=ROOT/"scripts/track_r/xp_a23_cpn_part2_cpu.py"
  s=p.read_text()
  ast.parse(s,filename=str(p))
  self.assertIn("A22_XPN_ALL6400_NATIVE_GENERATION_COMPLETE",s)
  self.assertIn("len(s[\"done\"])==90",s)
  self.assertIn("A23_V5_REGISTERED_REMOTE_BYTES",s)
  self.assertIn("V5_SHA=",s)
  self.assertIn("V5_COMMIT=",s)
  self.assertIn("len(v5[\"predictions\"])==100",s)
  self.assertIn("range(16,64)",s)
  self.assertIn("len(results)==4800",s)
  self.assertIn('clip_new(late,mode="g1",threshold=20)',s)
  self.assertIn('run-range","0:64"',s)
  self.assertIn('nmax","64"',s)
  self.assertIn("v5_primary_variant=",s)
  self.assertIn("no_matched_compute_claim=True",s)
  self.assertNotIn("A18_XPN_MAIN_4_15",s)
  self.assertNotIn('torch.cuda',s)
  publish=(ROOT/"scripts/track_r/xp_a23_cpn_part2_publish.sh").read_text()
  self.assertIn("XP_N_V5_PREDICTION_COMPARISON.json",publish)
  self.assertNotIn('cp "$OUT/LOCAL_ONLY_XPN_LATE_4800_GT_SCORES.csv"',publish)
  self.assertIn("ls-remote",publish)

 def test_merge_n16_prefix_and_late_to_exact100x64(self):
  from scripts.track_r.xp_a23_cpn_part2_cpu import merge_64
  prefix=[];c16=[];late=[];clate=[]
  for i in range(48900,49000):
   iid=f"ILSVRC2012_val_{i:08d}.JPEG"
   prefix.append(dict(image_id=iid,labels_by_run=";".join(["G"]*16),
      rho_signed_by_run=";".join([".03"]*16)))
   for r in range(16):
    c16.append(dict(image_id=iid,run=str(r),label="G",
       g1_choice="000",g1_margin=".01",g1_chosen_good="1"))
   for r in range(16,64):
    late.append(dict(image_id=iid,run=r,label="I",rho_signed=".06"))
    clate.append(dict(image_id=iid,run=r,label="I",
       g1_choice="000",g1_margin=".01",g1_chosen_good=0))
  ss,cc=merge_64(prefix,c16,late,clate,.05014467891305685)
  self.assertEqual(len(ss),100)
  self.assertEqual(len(cc),6400)
  self.assertEqual(ss[0]["n_G"],16)
  self.assertEqual(ss[0]["n_I"],48)
  self.assertEqual(len(ss[0]["labels_by_run"].split(";")),64)
  self.assertAlmostEqual(float(ss[0]["tau_signed_good_q95"]),
                         .05014467891305685)
  late.pop()
  with self.assertRaises(RuntimeError):
   merge_64(prefix,c16,late,clate,.05014467891305685)

 def test_pre_registered_v5_primary10_and_all100(self):
  from scripts.track_r.xp_a23_cpn_part2_cpu import v5_comparison,VARIANTS,GRIDS
  curve=[.30 + .004*k for k in range(64)]
  observed={"E0":curve,"E1":[x+.07 for x in curve],
     "O_raw":[x+.11 for x in curve],"O_orbit":[x+.20 for x in curve]}
  predicted=[]
  for variant in VARIANTS:
   for k in GRIDS:
    for policy,dm,native in [
     ("E0",None,"E0"),("E1_CLIP","beta_class","E1"),
     ("E1_CLIP","beta_image_cluster","E1"),
     ("O_raw",None,"O_raw"),("O_orbit",None,"O_orbit")]:
     point=observed[native][k-1]
     predicted.append(dict(model_variant=variant,K=k,policy=policy,
       decision_model=dm,point=point,lo95=point-.02,hi95=point+.02))
  rows,variant,winner=v5_comparison(dict(predictions=predicted),
        dict(fixed_curves=observed))
  self.assertEqual(len(rows),100)
  self.assertEqual(variant["race"]["primary_e0_e1_coverage"],10)
  self.assertEqual(variant["race"]["mean_abs_standardized_residual"],0.)
  self.assertEqual(winner,"race")

if __name__=="__main__":unittest.main()
