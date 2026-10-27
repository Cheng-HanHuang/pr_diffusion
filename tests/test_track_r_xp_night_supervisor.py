"""CPU-only overnight science, manifest, scheduler and gate regressions."""
from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_night_prepare_cpu as prep
from scripts.track_r import xp_night_supervisor as sup

class OvernightTests(unittest.TestCase):
    def test_exact_stage_and_cohort_task_sizes(self):
        self.assertEqual(prep.COUNTS,{"D1_P1":400,"D1_P2":1200,
                                      "A64":8640,"D2":400})
        self.assertEqual(set(sup.CRITICAL_READY),{"D1_P1","A64","D2"})
        self.assertEqual(sup.BLOCKED_UNTIL_REGISTRY,("D1_P2",))

    def test_partition_no_overlap_all_stages(self):
        tasks=sup.jobs()
        self.assertEqual(len({j["id"] for j in tasks}),len(tasks))
        for stage,n in prep.COUNTS.items():
            rows=[j for j in tasks if j["stage"]==stage]
            ids=[r for j in rows for r in range(j["start"],j["start"]+j["count"])]
            self.assertEqual(sorted(ids),list(range(n)))
            self.assertEqual(sum(j["smoke"]*j["count"] for j in rows),
                8 if stage in ("D1_P1","D2") else
                24 if stage=="D1_P2" else 96)
            if stage!="D2":
                self.assertTrue(all(j["count"]<=120 for j in rows))
                self.assertTrue(all(j["start"]%sup.REPS[stage]==0
                                    and j["count"]%sup.REPS[stage]==0
                                    for j in rows))

    def test_vram_policy_measured_not_obsolete_32g(self):
        self.assertEqual(sup.MEASURED_PEAK_MIB["D1_P1"],4023)
        self.assertEqual(sup.MEASURED_PEAK_MIB["A64"],1603)
        self.assertEqual(sup.MEASURED_PEAK_MIB["D2"],2755)
        self.assertEqual(sup.RESERVE_MIB["D1_P1"],8192)
        self.assertEqual(sup.RESERVE_MIB["A64"],4608)
        self.assertEqual(sup.RESERVE_MIB["D2"],6656)
        for stage in sup.CRITICAL_READY:
            self.assertLess(sup.RESERVE_MIB[stage],32768)

    def test_d2_native_worker_source_and_model_proof_fields(self):
        from scripts.track_r import xp_night_d2_worker as d2
        self.assertEqual(prep.DAPS_HEAD,d2.DAPS_HEAD)
        self.assertTrue(callable(d2.manifest_checked))
        self.assertTrue(callable(d2.native_worker.generate))

    def test_fast_command_has_same_source_and_no_gt(self):
        all_jobs=sup.jobs()
        fast=next(j for j in all_jobs if j["stage"]=="A64")
        d2=next(j for j in all_jobs if j["stage"]=="D2")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            proof={"stages":{
                 stage:{"path":str(root/(stage+".json")),"sha256":"a"*64}
                 for stage in prep.STAGES}}
            (root/"XP_NIGHT_PREP.json").write_text(json.dumps(proof))
            cmd=sup.argv(root,"hash",fast,3,True)
            self.assertIn("--roots-sha",cmd)
            self.assertIn("--preflight-only",cmd)
            self.assertIn("--oversample",cmd)
            self.assertNotIn("ground_truth",str(cmd))
            cmd2=sup.argv(root,"hash",d2,2,True)
            self.assertIn("--preflight-only",cmd2)
            self.assertIn("--index",cmd2)
            self.assertNotIn("ground_truth",str(cmd2))

    def test_resume_does_not_restart_previous_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row=sup.jobs()[0]
            folder=sup.paths(root,row)
            self.assertIn("chunk_",str(folder))
            d2=next(j for j in sup.jobs() if j["stage"]=="D2")
            (root/"D2.json").write_text(json.dumps(
                {"rows":[{"uid":f"d2_{i}"} for i in range(400)]}))
            self.assertIn("root_",str(sup.paths(root,d2)))

    def test_stage_timings_and_sealed_rules(self):
        for stage,sec in (("D1_P1",7.190894621),
                          ("A64",3.32078449),("D2",166.23148921877146)):
            self.assertAlmostEqual(sup.REFERENCE_S[stage],sec)
        self.assertNotIn("D1_P2",sup.CRITICAL_READY)
        self.assertTrue(all(j["lane"]=="CRITICAL" for j in sup.jobs()))

if __name__=="__main__":
    unittest.main()
