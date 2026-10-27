"""Synthetic CPU-only DAPS-4K timing receipt publication gate regression tests.

No user data or protected cohorts are read. Tests model/load/timing/matched-y
integrity and fail-closed behavior for publication of 18 native roots.
"""
from __future__ import annotations
import copy,unittest
from scripts.track_r.xp_t_daps4k_publish_cpu import (
    validate,RUNNER_HEAD,MANIFEST_SHA,NATIVE_SHA,PREP_HEAD,DAPS_HEAD)

def fixture():
    tasks=[];summaries=[];roots=[]
    for i in range(18):
        ds="ffhq" if i<9 else "imagenet"
        uid=f"XP_T|DAPS|{ds}|image{i//3}|rep{i%3}"
        task={"uid":uid,"dataset":ds,"measurement_sha256":"a"*64,
              "seed":i+200,"rep":i%3}
        tasks.append(task)
        model="b"*64 if ds=="ffhq" else "c"*64
        r={"uid":uid,"dataset":ds,"status":"XP_T_DAPS4K_ROOT_PASS",
           "runner_head":RUNNER_HEAD,"prep_head":PREP_HEAD,
           "manifest_sha256":MANIFEST_SHA,
           "source_receipt_sha256":NATIVE_SHA,"native_head":DAPS_HEAD,
           "method":"DAPS-4K","model_load_strict":True,
           "model_load_trace":{"unet_calls":1,"strict":True,
                               "missing_keys":[],"unexpected_keys":[],
                               "tensor_count":500},
           "seed":i+200,"rep":i%3,"measurement_sha256":"a"*64,
           "model_sha256":model,"gpu_physical":i%4,
           "native_sampling_wall_s":66.01368511701003 if ds=="ffhq" else 166.23148921877146,
           "native_sampling_cuda_event_s":65.9 if ds=="ffhq" else 166.1,
           "gt_free_generation":True,"historical_candidate_access":False,
           "evaluation_outcomes_embargo":"NOT_SCORED"}
        roots.append(r)
        summaries.append({"uid":uid,"dataset":ds,
                          "measurement_sha256":r["measurement_sha256"],
                          "model_sha256":model,"gpu_physical":i%4,
                          "native_sampling_wall_s":r["native_sampling_wall_s"],
                          "native_sampling_cuda_event_s":r["native_sampling_cuda_event_s"]})
    cfg={"method":"DAPS-4K",
         "annealing_scheduler_config":{"num_steps":400},
         "diffusion_scheduler_config":{"num_steps":10},
         "mcmc_sampler_config":{"num_steps":100,"mc_algo":"langevin"}}
    manifest={"status":"XP_T_DAPS4K_TASKS_FROZEN","head":PREP_HEAD,
              "source_head":DAPS_HEAD,"native_source_receipt_sha256":NATIVE_SHA,
              "n_roots":18,"gt_paths_supplied":False,
              "historical_paths_supplied":False,
              "historical_population_not_extended":True,
              "evaluation_outcomes_embargo":"NOT_SCORED",
              "settings":cfg,"rows":tasks}
    ff={"n_roots":9,"n_images":3,"runs_per_image":3,
        "gt_scored":False,"sampling_wall_s_median":66.01368511701003,
        "sampling_cuda_event_s_median":66.0135,
        "model_loading_wall_s_median":0.49}
    im={"n_roots":9,"n_images":3,"runs_per_image":3,
        "gt_scored":False,"sampling_wall_s_median":166.23148921877146,
        "sampling_cuda_event_s_median":166.2304,
        "model_loading_wall_s_median":2.1555}
    smoke={"status":"XP_T_DAPS4K_SMOKE_PASS","indices":[0,9],
           "model_loads_strictly_verified":True,"gt_free":True,
           "conservative_total_projected_gpu_hours":1.0127296631314675}
    comp={"status":"XP_T_DAPS4K_BATCH1_PASS","head":RUNNER_HEAD,
          "prep_head":PREP_HEAD,"manifest_sha256":MANIFEST_SHA,
          "family":"DAPS","method":"DAPS-4K",
          "n_roots":18,"n_images":6,"batch_size":1,
          "workers_per_physical_gpu":1,
          "physical_gpu_used":[0,1,2,3],
          "gt_free_generation":True,
          "all_model_loads_strictly_verified":True,
          "original_measurements_unchanged":True,
          "historical_population_not_extended":True,
          "evaluation_outcomes_embargo":"NOT_SCORED",
          "clean_timing_eligible":True,
          "total_gpu_hours":0.9600079935751596,
          "total_worker_seconds":0.9600079935751596*3600,
          "per_device_timing":{str(i):{"timing_class":"CLEAN",
                  "timing_valid":True,"foreign_pids":[],
                  "reasons":[],"n_samples":15} for i in range(4)},
          "per_dataset":{"ffhq":ff,"imagenet":im},
          "smoke_gate":smoke,"per_root":summaries}
    status={"state":"PASS","phase":"COMPLETE","reason":None,
            "completed_roots":18,"running_roots":0,"active_pids":[],
            "head":RUNNER_HEAD,"manifest_sha256":MANIFEST_SHA,
            "gpu_hours":comp["total_gpu_hours"]}
    return status,comp,manifest,smoke,roots

class TimingPublicationTests(unittest.TestCase):
    def test_all_eighteen_exact_native_receipts_pass(self):
        self.assertTrue(validate(*fixture()))
    def test_foreign_gpu_process_and_incomplete_reconstruction_fail(self):
        s,c,m,sm,r=fixture()
        c["per_device_timing"]["1"]["foreign_pids"]=[432]
        with self.assertRaisesRegex(RuntimeError,"TIMING_1"):
            validate(s,c,m,sm,r)
        s,c,m,sm,r=fixture()
        r[5]["model_load_trace"]["unet_calls"]=0
        with self.assertRaisesRegex(RuntimeError,"ROOT_"):
            validate(s,c,m,sm,r)
    def test_short_daps_1k_and_scored_metadata_fail(self):
        s,c,m,sm,r=fixture()
        m["settings"]["annealing_scheduler_config"]["num_steps"]=200
        with self.assertRaisesRegex(RuntimeError,"NOT_PUBLISHED_4K"):
            validate(s,c,m,sm,r)
        s,c,m,sm,r=fixture()
        r[3]["evaluation_outcomes_embargo"]="SCORED"
        with self.assertRaisesRegex(RuntimeError,"ROOT_"):
            validate(s,c,m,sm,r)
    def test_smoke_population_and_budget_are_locked(self):
        s,c,m,sm,r=fixture()
        sm["conservative_total_projected_gpu_hours"]=8
        with self.assertRaisesRegex(RuntimeError,"SMOKE"):
            validate(s,c,m,sm,r)
        s,c,m,sm,r=fixture()
        c["total_gpu_hours"]=6.5
        with self.assertRaisesRegex(RuntimeError,"BUDGET"):
            validate(s,c,m,sm,r)
        s,c,m,sm,r=fixture()
        m["gt_paths_supplied"]=True
        with self.assertRaisesRegex(RuntimeError,"MANIFEST"):
            validate(s,c,m,sm,r)
    def test_no_repeated_or_nonidentical_roots(self):
        s,c,m,sm,r=fixture()
        c["per_root"][-1]["uid"]=c["per_root"][0]["uid"]
        with self.assertRaisesRegex(RuntimeError,"ROOT_COUNT"):
            validate(s,c,m,sm,r)
        s,c,m,sm,r=fixture()
        r[2]["measurement_sha256"]="b"*64
        with self.assertRaisesRegex(RuntimeError,"ROOT_"):
            validate(s,c,m,sm,r)

if __name__=="__main__":unittest.main()
