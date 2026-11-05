"""A46 admission policy; pure CPU tests, no GPU invocation."""
import unittest
from scripts.track_r.xp_a46_xpk_gpu_admission_cpu import eligible,parse_gpu_csv
class A46Admission(unittest.TestCase):
    def test_busy_but_free_vram_is_allowed(self):
        g=parse_gpu_csv("0, 81920, 10000, 99\n1, 81920, 70000, 0")
        self.assertEqual([x["index"] for x in eligible(g,28)],[0])
    def test_quiet_preferred_not_required(self):
        g=parse_gpu_csv("0, 81920, 10000, 99\n1, 81920, 12000, 0")
        self.assertEqual([x["index"] for x in eligible(g,28)],[1,0])
    def test_eighty_percent_hard_ceiling(self):
        g=parse_gpu_csv("0, 81920, 50000, 0")
        self.assertEqual(eligible(g,28),[])
    def test_remaining_four_gib(self):
        g=parse_gpu_csv("0, 32768, 16000, 0")
        self.assertEqual(eligible(g,12),[])
    def test_no_double_assignment(self):
        g=parse_gpu_csv("0, 81920, 10000, 0\n1, 81920, 10000, 0")
        self.assertEqual([x["index"] for x in eligible(g,28,[0])],[1])
    def test_bad_telemetry_fails_closed(self):
        for bad in ("", "0, 81920, 100, 101", "0, 81920, 90000, 0",
                    "0, 81920, 100, 0\n0, 81920, 100, 0"):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                parse_gpu_csv(bad)
    def test_reject_tiny_reservation(self):
        with self.assertRaises(ValueError):eligible([],2)
if __name__=="__main__":unittest.main()
