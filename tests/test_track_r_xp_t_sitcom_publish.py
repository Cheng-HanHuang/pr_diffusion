"""Synthetic CPU-only SITCOM XP-T timing publication validation."""
from __future__ import annotations
import copy, unittest
from scripts.track_r.xp_t_sitcom_publish_cpu import validate,MANIFEST_SHA,RUNNER_HEAD,GPU_H
from scripts.track_r.xp_t_sitcom_worker_gpu import PREP_HEAD

def fixture():
    rows=[]
    final=[]
    roots=[]
    for i in range(18):
        ds="ffhq" if i<9 else "imagenet"
        uid=f"XP_T|SITCOM|{ds}|ID{i//3}|rep{i%3}"
        y="a"*64
        rows.append({"uid":uid,"dataset":ds,"measurement_sha256":y})
        base={"uid":uid,"dataset":ds,"gpu_physical":i%4,
              "model_sha256":"b"*64,
              "measurement_sha256":y,
              "native_sampling_wall_s":32.104 if ds=="ffhq" else 58.343,
              "native_model_loading_wall_s":1.5,
              "status":"XP_T_SITCOM_ROOT_PASS","runner_head":RUNNER_HEAD,
              "model_loaded_strictly":True,
              "strict_load_trace":{"unet_calls":1},
              "gt_free_generation":True,
              "evaluation_outcomes_embargo":"NOT_SCORED",
              "historical_candidate_access":False}
        roots.append(base)
        final.append({
            "uid":uid,"dataset":ds,"gpu_physical":i%4,
            "model_sha256":"b"*64,
            "measurement_sha256":y,
            "native_sampling_wall_s":base["native_sampling_wall_s"],
            "model_loading_wall_s":1.5})
    manifest={"status":"XP_T_SITCOM_TASKS_FROZEN",
              "head":PREP_HEAD,"n_roots":18,
              "gt_paths_supplied":False,
              "historical_paths_supplied":False,
              "rows":rows}
    ds={
      "ffhq":{"n_roots":9,"n_images":3,"runs_per_image":3,
              "gt_access":False,"sampling_wall_s_median":32.104,
              "sampling_cuda_event_s_median":32.100},
      "imagenet":{"n_roots":9,"n_images":3,"runs_per_image":3,
                  "gt_access":False,"sampling_wall_s_median":58.343,
                  "sampling_cuda_event_s_median":58.331},
    }
    complete={
      "status":"XP_T_SITCOM_BATCH1_PASS",
      "head":RUNNER_HEAD,"prep_head":PREP_HEAD,
      "manifest_sha256":MANIFEST_SHA,
      "family":"SITCOM","n_roots":18,"n_images":6,
      "workers_per_physical_gpu":1,
      "physical_gpu_used":[0,1,2,3],
      "gt_free_generation":True,
      "all_model_loads_strictly_verified":True,
      "historical_population_not_extended":True,
      "evaluation_outcomes_embargo":"NOT_SCORED",
      "clean_timing_eligible":True,"total_gpu_hours":GPU_H,
      "per_dataset":ds,"per_root":final,
      "per_device_timing":{str(i):{
          "timing_class":"CLEAN","timing_valid":True,
          "foreign_pids":[],"reasons":[],"n_samples":9}
          for i in range(4)},
    }
    status={"state":"PASS","reason":None,"completed_roots":18,
            "running_roots":0,"active_pids":[],
            "head":RUNNER_HEAD,"manifest_sha256":MANIFEST_SHA,
            "gpu_hours":GPU_H}
    return status,complete,manifest,roots

class SITCOMTimingPublishTests(unittest.TestCase):
    def test_success(self):
        self.assertTrue(validate(*fixture()))
    def test_timing_contamination_rejected(self):
        s,c,m,r=fixture()
        c["per_device_timing"]["2"]["timing_class"]="UNVERIFIED"
        c["per_device_timing"]["2"]["timing_valid"]=False
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_TELEMETRY"):
            validate(s,c,m,r)
    def test_bad_checkpoint_load_rejected(self):
        s,c,m,r=fixture()
        r[11]["strict_load_trace"]["unet_calls"]=0
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_ROOT_PAIR"):
            validate(s,c,m,r)
    def test_missing_roots_bad_budget_and_wrong_manifest(self):
        s,c,m,r=fixture()
        c["per_root"]=c["per_root"][:-1]
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_ROOT_COUNT"):
            validate(s,c,m,r)
        s,c,m,r=fixture()
        c["total_gpu_hours"]=2.2
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_GPU_BUDGET"):
            validate(s,c,m,r)
        s,c,m,r=fixture()
        m["gt_paths_supplied"]=True
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_MANIFEST"):
            validate(s,c,m,r)
if __name__=="__main__":
    unittest.main()
