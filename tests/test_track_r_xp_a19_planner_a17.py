"""A17 prospective v5 input/orientation + leakage-barrier CPU unit tests."""
from __future__ import annotations
import ast
from collections import Counter
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class A19Tests(unittest.TestCase):
    def test_all_new_python_stages_parse(self):
        for name in ("xp_a19_xpi_chosen_cpu.py","xp_a19_xpn_prefix4_cpu.py"):
            path=ROOT/"scripts/track_r"/name
            ast.parse(path.read_text(),filename=str(path))

    def test_prefix4_single_letter_and_stable_rho_rank(self):
        from scripts.track_r.xp_a19_xpn_prefix4_cpu import summaries,COLP
        rows=[]
        labs=("O_g","G","C","O_p")
        for i in range(48900,49000):
            iid=f"ILSVRC2012_val_{i:08d}.JPEG"
            for rep,l in enumerate(labs):
                rows.append(dict(image_id=iid,run=rep,label=l,
                    rho_signed=str([.2,.1,.1,.3][rep])))
        p=summaries(rows,.0500829)
        self.assertEqual(len(p),100)
        self.assertEqual(set(p[0]),set(COLP))
        self.assertEqual(p[0]["family"],"FAST")
        self.assertEqual(p[0]["run_labels"],"OGCP")
        self.assertEqual(p[0]["e0_rank_order"],"1,2,0,3")
        self.assertEqual(p[0]["e0_rank_labels"],"G COP".replace(" ",""))
        self.assertEqual([p[0][x] for x in
            ("n_GOOD","n_ORBIT_global","n_ORBIT_partial",
             "n_CONSISTENT","n_INCONSISTENT")],[1,1,1,1,0])
        rows.pop()
        with self.assertRaises(RuntimeError):summaries(rows,.0500829)

    def test_pilot_only_and_v5_immutable_sha_gate(self):
        prefix=(ROOT/"scripts/track_r/xp_a19_xpn_prefix4_cpu.py").read_text()
        runner=(ROOT/"scripts/track_r/xp_a19_xpn_v5_autopilot.sh").read_text()
        self.assertIn("A18_XPN_PILOT_0_3",prefix)
        self.assertIn('len(rows)==400',prefix)
        self.assertIn('for st in range(0,400,20)',prefix)
        self.assertIn('g1_chosen_good',prefix)
        self.assertIn("e0_rank_order",prefix)
        self.assertNotIn("A18_XPN_MAIN_4_15",prefix)
        self.assertNotIn("A18_XPN_LATE_16_63",prefix)
        self.assertNotIn("XP_N_FINAL_STATUS",prefix)
        self.assertIn("81003cc5a9167def74a04281501b4eae2837f44c",runner)
        self.assertIn("5ac8811f0948ee143f0b2cd90406626a01bc9421121472fccb241292e2745292",runner)
        self.assertIn("7313d46ddb5df33e2b3dfb7bae8c301e66584b0d503b6da117560e41bde4d33c",runner)
        self.assertIn("6f70c8638eda541ce10786b905403a3ac82c2fe2eefef9bfcf25647819ff3f71",runner)
        self.assertIn("--point-reps 250 --bootstrap 60 --boot-reps 3",runner)
        self.assertIn("A19_XPN_V5_REMOTE_REGISTERED_AND_VERIFIED",runner)
        self.assertIn("XP_N_V5_DECLINED_PRODUCTION_FAILED",runner)
        self.assertIn('worktree add --detach "$CODEWT" "$CODE"',runner)

    def test_xpi_raw_and_chosen_are_evaluation_only(self):
        src=(ROOT/"scripts/track_r/xp_a19_xpi_chosen_cpu.py").read_text()
        self.assertIn("PSNR_chosen",src)
        self.assertIn("SSIM_chosen",src)
        self.assertIn("VGG_LPIPS_chosen",src)
        self.assertIn("EXPECTED_E0_ROTATED",src)
        self.assertIn("A19_XPI_PLANNER_NATIVE_E0_REPRODUCTION",src)
        self.assertIn("chosen_metrics_use_GT_evaluation_only=True",src)
        self.assertIn("no_new_gpu_or_clips=True",src)

if __name__=="__main__":unittest.main()
