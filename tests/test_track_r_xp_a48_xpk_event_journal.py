"""A48 local CPU journal crash/restart tests, no GPU or PAC access."""
import tempfile,unittest
from pathlib import Path
from scripts.track_r.xp_a48_xpk_event_journal_cpu import initialize,mutate,read_events
from scripts.track_r.xp_a47_xpk_gpu_ledger_cpu import charge
class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=initialize(Path(self.temp.name)/"events.jsonl")
    def tearDown(self):self.temp.cleanup()
    def test_durable_start_end_and_restart_charge(self):
        mutate(self.path,"START","job1",0,family="FAST",phase="pilot",now=100)
        self.assertEqual(charge(read_events(self.path),200)["unfinished_workers"],["job1"])
        mutate(self.path,"END","job1",0,now=130)
        self.assertEqual(charge(read_events(self.path),1000)["physical_gpu_seconds"]["0"],30)
        self.assertEqual(charge(read_events(self.path),2000)["physical_gpu_seconds"]["0"],30)
    def test_no_reused_start_or_double_end(self):
        mutate(self.path,"START","job1",0,family="FAST",phase="pilot",now=100)
        with self.assertRaisesRegex(RuntimeError,"WORKER_ID_REUSE"):
            mutate(self.path,"START","job1",0,family="FAST",phase="pilot",now=101)
        mutate(self.path,"END","job1",0,now=130)
        with self.assertRaisesRegex(RuntimeError,"ORPHAN_OR_DUPLICATE_END"):
            mutate(self.path,"END","job1",0,now=131)
    def test_end_gpu_must_match(self):
        mutate(self.path,"START","job1",0,family="SITCOM",phase="calibrated",now=100)
        with self.assertRaisesRegex(RuntimeError,"END_GPU_MISMATCH"):
            mutate(self.path,"END","job1",1,now=120)
    def test_truncated_journal_stops_and_does_not_repair(self):
        self.path.write_bytes(b'{"id":"unfinished"')
        before=self.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError,"TRUNCATED_JOURNAL"):
            mutate(self.path,"START","job2",0,family="FAST",phase="pilot",now=100)
        self.assertEqual(before,self.path.read_bytes())
    def test_end_before_start_stops(self):
        with self.assertRaisesRegex(RuntimeError,"ORPHAN_OR_DUPLICATE_END"):
            mutate(self.path,"END","job1",0,now=100)
    def test_reject_invalid_families_and_gpus(self):
        for family,gpu in [("FAKE",0),("FAST",4)]:
            with self.assertRaises(RuntimeError):
                mutate(self.path,"START","x",gpu,family=family,phase="pilot",now=100)
    def test_init_never_overwrites(self):
        with self.assertRaises(RuntimeError):initialize(self.path)
if __name__=="__main__":unittest.main()
