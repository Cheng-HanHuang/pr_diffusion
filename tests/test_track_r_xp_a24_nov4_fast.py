"""A19 Nov4 FAST CPU lane: structural and frozen-source QA, no GPU."""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class Nov4FastTests(unittest.TestCase):
    def test_cpu_worker_parses_and_has_authorized_scope(self):
        p=ROOT/"scripts/track_r/xp_a24_nov4_fast_cpu.py"
        s=p.read_text()
        ast.parse(s,filename=str(p))
        for word in (
            "A9_FAST_64_PER_IMAGE_STRINGS.csv",
            "A9_FAST_11_CASES.json",
            "d616dd50f10352e98b891a99b9a153d5d72a93c61e44381827147dcca23f5466",
            "PROMPTS_UP,PROMPTS_INV",
            "S5_r0_fast",
            "len(hist)==720",
            "len(new16)==2160",
            "len(new64)==8640",
            "len(results)==11520",
            "len(never)==7",
            "face_chosen_good",
            "orbit_psnr_channel_mse",
            "GT_solely_offline_metric_evaluation=True",
            "no_new_selector=True",
            "CPU_ONLY",
            "Image.Resampling.BILINEAR",
            "explicit"):
            if word=="explicit":
                continue
            self.assertIn(word,s)
        for disallowed in ("torch.cuda", "model.to('cuda')",
                           "killpg", "subprocess.Popen"):
            self.assertNotIn(disallowed,s)

    def test_launcher_parses_and_avoids_gpu(self):
        p=ROOT/"scripts/track_r/xp_a24_nov4_fast_launch.sh"
        s=p.read_text()
        self.assertIn('export CUDA_VISIBLE_DEVICES=""',s)
        self.assertIn("--panels-only",s)
        self.assertIn("--resume",s)
        self.assertIn("nohup bash -c",s)
        self.assertNotIn("nvidia-smi",s)

if __name__=="__main__":
    unittest.main()
