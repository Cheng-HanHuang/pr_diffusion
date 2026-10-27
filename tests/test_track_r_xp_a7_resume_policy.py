"""Synthetic A7 resume GPU policy tests, no CUDA allocation or PAC files."""
import unittest
from prdiffusion.track_r.xp_gpu_telemetry import GPU,choose_gpu
from scripts.track_r.xp_a7_resume_supervisor import a7_gpu_cap,jobs,STAGES,COUNTS,RESERVATION

def card(index,used=3,total=95000,free=None,foreign=(),own=(),visibility="known"):
    return GPU(index,f"GPU-{index}",total,used,
               total-used if free is None else free,0,
               tuple(sorted(set(foreign)|set(own))),
               tuple(foreign),visibility)
class A7PolicyTests(unittest.TestCase):
    def test_clean_four_gpu_cap(self):
        cards=[card(i) for i in range(4)]
        self.assertEqual(a7_gpu_cap(cards),(4,False))
    def test_foreign_on_other_gpu_caps_trackr_at_two(self):
        cards=[card(0,used=30000,foreign=(1234,)),
               card(1),card(2),card(3)]
        self.assertEqual(a7_gpu_cap(cards),(2,True))
        # Independent clean GPU 2 is valid with sufficient free VRAM.
        available,_=choose_gpu([cards[2]],RESERVATION["EPRIME"],
            safety_margin_mib=2048,fraction=.8,clean_only=False)
        self.assertEqual(available,2)
    def test_unknown_telemetry_conservatively_caps(self):
        cards=[card(0,visibility="unknown"),card(1),card(2),card(3)]
        self.assertEqual(a7_gpu_cap(cards),(2,True))
    def test_colocated_generation_requires_80pct_and_vram(self):
        c=card(2,used=68000)
        okay,_=choose_gpu([c],RESERVATION["EPRIME"],
                 safety_margin_mib=2048,fraction=.8,clean_only=False)
        self.assertIsNone(okay)
        c=card(2,used=30000,foreign=(123,))
        okay,_=choose_gpu([c],RESERVATION["EPRIME"],
                 safety_margin_mib=2048,fraction=.8,clean_only=False)
        self.assertEqual(okay,2)
    def test_exact_paused_run_stage_partition(self):
        tasks=jobs()
        self.assertEqual(STAGES,("D1_P2","D3","EPRIME"))
        self.assertEqual(sum(t["count"] for t in tasks),2320)
        for st in STAGES:
            seen=sorted(i for t in tasks if t["stage"]==st
                        for i in range(t["start"],t["start"]+t["count"]))
            self.assertEqual(seen,list(range(COUNTS[st])))
        self.assertTrue(all(t["time_class"]=="TIME_UNAWARE_GENERATION" for t in tasks))
if __name__=="__main__":unittest.main()
