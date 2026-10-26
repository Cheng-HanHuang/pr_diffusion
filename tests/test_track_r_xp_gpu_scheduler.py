"""CPU-only tests for GPU monitoring, opportunity scheduling and receipts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prdiffusion.track_r.xp_gpu_telemetry import GPU,collect,choose_gpu,classify_timing,descendants
from scripts.track_r import xp_gpu_controller as ctl

def gpu(i, used=1000, total=96000, pids=(), foreign=(), visibility="known"):
    return GPU(i,"GPU-%d"%i,total,used,total-used,0,
               tuple(pids),tuple(foreign),visibility)

class TelemetryTests(unittest.TestCase):
    def test_descendants_exclude_owned_workers(self):
        listed=[(100,1),(110,100),(111,110),(220,1)]
        self.assertEqual(descendants([100],listed),{100,110,111})

    def test_foreign_jobs_recorded_not_blocked(self):
        inventory="0, GPU-0, 96000, 33000, 63000, 73\n1, GPU-1, 96000, 5000, 91000, 0\n"
        processes="GPU-0, 220, 28000\nGPU-0, 110, 800\n"
        def fake(cmd):
            return processes if any("compute-apps" in v for v in cmd) else inventory
        cards=collect([100],fake,[(100,1),(110,100),(220,1)])
        self.assertEqual(cards[0].foreign_pids,(220,))
        i,_=choose_gpu(cards,12000,preferred=(0,))
        self.assertEqual(i,0)
        report=classify_timing([cards],0)
        self.assertFalse(report["timing_valid"])
        self.assertEqual(report["timing_class"],"CONTAMINATED")

    def test_timing_remeasure_waits_clean(self):
        dirty=[gpu(0,used=10000,pids=(211,),foreign=(211,))]
        clean=[gpu(0,used=1000)]
        self.assertEqual(choose_gpu(dirty,12000,clean_only=True)[0],None)
        self.assertEqual(choose_gpu(clean,12000,clean_only=True)[0],0)

    def test_pause_when_vram_too_full_resume(self):
        busy=[gpu(0,used=71000,total=96000),gpu(1,used=73000,total=96000)]
        available=[gpu(0,used=40000,total=96000),gpu(1,used=73000,total=96000)]
        target,why=choose_gpu(busy,12000,safety_margin_mib=2048)
        self.assertIsNone(target)
        self.assertEqual(set(why.values()),{"WAIT_VRAM"})
        target,_=choose_gpu(available,12000,safety_margin_mib=2048)
        self.assertEqual(target,0)

    def test_80_percent_rule(self):
        self.assertIsNone(choose_gpu([gpu(0,used=75000,total=96000)],
                                      1000,safety_margin_mib=2048)[0])
        self.assertEqual(choose_gpu([gpu(0,used=10000,total=96000)],12000)[0],0)

    def test_mid_run_foreign_contaminates_timing(self):
        baseline=[gpu(0,used=1400)]
        during=[gpu(0,used=22000,pids=(111,222),foreign=(222,))]
        after=[gpu(0,used=1200)]
        result=classify_timing([baseline,during,after],0)
        self.assertEqual(result["n_samples"],3)
        self.assertFalse(result["timing_valid"])
        self.assertIn(222,result["foreign_pids"])

    def test_unavailable_process_list_fails_timing_openly(self):
        observation=[gpu(0,used=1000,visibility="unknown")]
        self.assertEqual(classify_timing([observation],0)["timing_class"],"UNVERIFIED")
        i,_=choose_gpu(observation,12000)
        self.assertEqual(i,0)

    def test_quality_receipt_requires_gt_free_and_hash(self):
        j={"id":"test","expected_measurement_sha256":"0"*64}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"XP_JOB_PASS.json"
            head="1"*40
            ok={"status":"XP_JOB_PASS","job_id":"test","head":head,
                "measurement_sha256":"0"*64,"gt_used_during_generation":False}
            p.write_text(json.dumps(ok))
            with patch.object(ctl,"ROOT",Path(d)):
                self.assertTrue(ctl.receipt_valid(j,p,head))
                ok["gt_used_during_generation"]=True
                p.write_text(json.dumps(ok))
                self.assertFalse(ctl.receipt_valid(j,p,head))
                ok["gt_used_during_generation"]=False
                ok["measurement_sha256"]="2"*64
                p.write_text(json.dumps(ok))
                self.assertFalse(ctl.receipt_valid(j,p,head))

    def test_lane_priority(self):
        jobs=[{"id":"bg","lane":"BACKGROUND","stage":"zz"},
              {"id":"critical","lane":"CRITICAL","stage":"aa"}]
        self.assertEqual([j["id"] for j in ctl.choose_pending(jobs,{})],
                         ["critical","bg"])

if __name__=="__main__":
    unittest.main()
