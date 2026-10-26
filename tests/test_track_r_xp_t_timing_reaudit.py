"""Synthetic post-hoc XP-T telemetry audit tests; no external process calls."""
import unittest
from prdiffusion.track_r.xp_t_timing_audit import audit_timing_events

def record(kind,i,pid=None,used=0,util=0,foreign=(),visible="known"):
    worker=[pid] if pid else []
    return {
      "action":kind,"stage":"XP-T","selected_gpu":0,
      "monotonic_sec":float(i),
      "tracked_roots":[42] if kind=="HEARTBEAT" else [],
      "gpu_observations":[{
          "index":0,"total_mib":97887,"used_mib":used,
          "utilization_pct":util,"process_pids":worker,
          "foreign_pids":list(foreign),"visibility":visible
      }]
    }
def sample(tail=True):
    data=[
        record("PRE_GPU",0,used=0),
        record("HEARTBEAT",1,used=3),
        record("HEARTBEAT",2,pid=42,used=4023,util=84),
        record("HEARTBEAT",3,pid=42,used=4023,util=81),
        record("HEARTBEAT",4,pid=42,used=4023,util=83)
    ]
    if tail:data.append(record("HEARTBEAT",5,used=4012,util=0))
    data.append(record("POST_GPU",6,used=0))
    timing={"timing_class":"UNVERIFIED" if tail else "CLEAN",
            "timing_valid":not tail,
            "n_samples":len(data),"foreign_pids":[],
            "reasons":["unattributed_gpu_memory"] if tail else []}
    status={"state":"PASS","worker_rc":0,"worker_pid":42,
            "gpu":0,"timing":timing}
    comp={"gpu_index":0,"timing":timing,"new_roots":9,"n_images":3,
          "status":"XP_T_FAST_IMAGENET_BATCH1_PASS" if tail else "XP_T_FAST_FFHQ_BATCH1_PASS",
          "dataset":"imagenet" if tail else "ffhq",
          "sampling_wall_s_median":7.191 if tail else 3.321,
          "sampling_cuda_event_s_median":7.191 if tail else 3.321,
          "gpu_worker_wall_s":112 if tail else 61,
          "gpu_hours":.031 if tail else .017}
    return data,status,comp

class PosthocTiming(unittest.TestCase):
    def test_postexit_nvml_tail_is_qualified_not_rewritten(self):
        a,b,c=sample(True)
        x=audit_timing_events(a,b,c)
        self.assertEqual(x["qualified_timing_class"],"CLEAN_POSTHOC")
        self.assertEqual(x["n_unattributed_post_exit_samples"],1)
        self.assertEqual(b["timing"]["timing_class"],"UNVERIFIED")
    def test_original_clean_retained(self):
        a,b,c=sample(False)
        self.assertEqual(audit_timing_events(a,b,c)["qualified_timing_class"],"CLEAN_ORIGINAL")

    def test_actual_ffhq_small_vram_owned_worker_is_active(self):
        # Real XP-T FFHQ was only 1,603 MiB, below the classifier's
        # separate 2,048 MiB threshold for unattributed orphan memory.
        # Confirm a normal sampling PID is not discarded by that cutoff.
        events=[
            record("PRE_GPU",0,used=0,util=0),
            record("HEARTBEAT",1,used=3,util=0),
            record("HEARTBEAT",2,pid=42,used=3,util=0),
            record("HEARTBEAT",3,pid=42,used=1603,util=65),
            record("HEARTBEAT",4,pid=42,used=1603,util=70),
            record("HEARTBEAT",5,pid=42,used=1603,util=69),
            record("HEARTBEAT",6,used=0,util=0),
            record("POST_GPU",7,used=0,util=0)
        ]
        _,state,receipt=sample(False)
        state["timing"]["n_samples"]=len(events)
        result=audit_timing_events(events,state,receipt)
        self.assertEqual(result["n_active_worker_visible"],3)
        self.assertEqual(result["sampled_peak_used_mib"],1603)
        self.assertEqual(result["qualified_timing_class"],"CLEAN_ORIGINAL")

    def test_small_owned_memory_without_compute_is_not_active(self):
        # PID visibility alone with no GPU utilization cannot establish
        # that a timing interval actually contains computation.
        events=[
            record("PRE_GPU",0),
            record("HEARTBEAT",1,pid=42,used=1603,util=0),
            record("HEARTBEAT",2,pid=42,used=1603,util=0),
            record("HEARTBEAT",3,pid=42,used=1603,util=0),
            record("HEARTBEAT",4,used=0,util=0),
            record("POST_GPU",5)
        ]
        _,state,receipt=sample(False)
        state["timing"]["n_samples"]=len(events)
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_AUDIT_TOO_FEW_ACTIVE"):
            audit_timing_events(events,state,receipt)
    def test_interference_in_middle_rejected(self):
        a,b,c=sample(True)
        a.insert(3,record("HEARTBEAT",2.5,used=4012,util=0))
        b["timing"]["n_samples"]=len(a)
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_AUDIT_NOT_ONLY_POST_EXIT_MEMORY"):
            audit_timing_events(a,b,c)
    def test_unknown_foreign_or_active_ghost_rejected(self):
        a,b,c=sample(True)
        a[2]["gpu_observations"][0]["foreign_pids"]=[999]
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_AUDIT_FOREIGN_OR_UNKNOWN"):
            audit_timing_events(a,b,c)
        a,b,c=sample(True)
        a[-2]["gpu_observations"][0]["utilization_pct"]=55
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_AUDIT_NOT_ONLY_POST_EXIT_MEMORY"):
            audit_timing_events(a,b,c)
        a,b,c=sample(True)
        a[-2]["gpu_observations"][0]["visibility"]="unknown"
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_AUDIT_FOREIGN_OR_UNKNOWN"):
            audit_timing_events(a,b,c)
    def test_no_pid_active_jobs_rejected(self):
        a,b,c=sample(True)
        a[2]["gpu_observations"][0]["process_pids"]=[]
        with self.assertRaisesRegex(RuntimeError,"STOP_XPT_AUDIT_NOT_ONLY_POST_EXIT_MEMORY"):
            audit_timing_events(a,b,c)
if __name__=="__main__":
    unittest.main()
