"""CPU-only operational publication tests; no actual images/GT/GPU."""
import unittest
from copy import deepcopy
from scripts.track_r.xp_night_publish_cpu import (
    EXPECTED_STAGE_COUNTS,verify_all_outputs,verify_state,verify_manifests)
from scripts.track_r import xp_night_supervisor as night

def fixture():
    jobs=night.jobs()
    chosen=[j for j in jobs if j["stage"] in EXPECTED_STAGE_COUNTS]
    ids={j["id"] for j in chosen}
    prep={
      "status":"XP_NIGHT_INPUTS_FROZEN","head":
      "8d9bfed0fa62735791987a24791cfb97123540d8",
      "total_new_roots":10640,"gpu_work_performed":False,
      "gt_paths_supplied":False,"new_quality_scores_generated":False,
      "prediction_freeze_required_for_scoring_D1_P2_A64":True}
    smoke={
      "status":"XP_NIGHT_ALL_SMOKES_PASS","head":prep["head"],
      "stages":{stage:{"n_new_roots":k,
               "source_outcome_inspected":False}
                for stage,k in (("D1_P1",8),("A64",96),("D2",8))},
      "no_gt_access":True,"outcomes_scored":False}
    state={
      "status":"MAIN_PASS","mode":"main","head":prep["head"],
      "completed_roots":9440,"active_jobs":[],
      "smoke_pass":True,"outcomes_scored":False,
      "sealed_outcomes_not_inspected":True,
      "critical_stages_completed":list(EXPECTED_STAGE_COUNTS),
      "gpu_hours":33.7,"done":sorted(ids),
      "deferred_prediction_gate":["D1_P2"],
      "background_ready":[],
      "per_job_timing":{key:{"timing_valid":True} for key in ids},
      "timing_unverified_jobs":0}
    return prep,smoke,state,jobs

class OvernightPublicationTests(unittest.TestCase):
    def test_full_9440_operational_coverage(self):
        prep,smoke,state,jobs=fixture()
        result=verify_state(state,smoke,prep,jobs)
        self.assertEqual(result["expected_jobs"],495)
        self.assertEqual(result["timing_unverified_jobs"],0)
        audit=verify_all_outputs(jobs,lambda j:[1.0]*j["count"])
        self.assertEqual(audit["per_stage_candidates"],EXPECTED_STAGE_COUNTS)
        self.assertEqual(audit["total_validated_candidates"],9440)
        self.assertEqual(audit["reconstruction_files_sha256_verified"],18880)

    def test_missing_root_receipt_fails_closed(self):
        _,_,_,jobs=fixture()
        with self.assertRaisesRegex(RuntimeError,"CANDIDATE_VALIDATION"):
            verify_all_outputs(jobs,lambda j:[1.0]*(j["count"]-1))

    def test_attempted_outcome_release_fails_closed(self):
        prep,smoke,state,jobs=fixture()
        state["sealed_outcomes_not_inspected"]=False
        with self.assertRaisesRegex(RuntimeError,"FINAL_CONTROLLER_STATE"):
            verify_state(state,smoke,prep,jobs)
        prep,smoke,state,jobs=fixture()
        smoke["outcomes_scored"]=True
        with self.assertRaisesRegex(RuntimeError,"SMOKE_CONTRACT"):
            verify_state(state,smoke,prep,jobs)

    def test_extra_or_missing_completion_rejected(self):
        prep,smoke,state,jobs=fixture()
        state["done"].pop()
        with self.assertRaisesRegex(RuntimeError,"CLOSED_JOB_COVERAGE"):
            verify_state(state,smoke,prep,jobs)
        prep,smoke,state,jobs=fixture()
        state["done"].append("D1_P2_00000_024")
        with self.assertRaisesRegex(RuntimeError,"CLOSED_JOB_COVERAGE"):
            verify_state(state,smoke,prep,jobs)

    def test_timing_contamination_is_reported_not_erased(self):
        prep,smoke,state,jobs=fixture()
        key=state["done"][0]
        state["per_job_timing"][key]={"timing_valid":False,
          "timing_class":"CONTAMINATED"}
        state["timing_unverified_jobs"]=1
        r=verify_state(state,smoke,prep,jobs)
        self.assertEqual(r["timing_unverified_jobs"],1)
        state["timing_unverified_jobs"]=0
        with self.assertRaisesRegex(RuntimeError,"TIMING_ACCOUNTING"):
            verify_state(state,smoke,prep,jobs)

    def test_no_lost_smoke_or_computation(self):
        prep,smoke,state,jobs=fixture()
        smoke["stages"]["A64"]["n_new_roots"]=48
        with self.assertRaisesRegex(RuntimeError,"SMOKE_STAGE_A64"):
            verify_state(state,smoke,prep,jobs)
        prep,smoke,state,jobs=fixture()
        state["active_jobs"]=[{"pid":1}]
        with self.assertRaisesRegex(RuntimeError,"FINAL_CONTROLLER_STATE"):
            verify_state(state,smoke,prep,jobs)
if __name__=="__main__":
    unittest.main()
