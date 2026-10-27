"""Synthetic CPU-only XP-A16 embargo-safe publication tests.

No PAC data, GT files, reconstructions or GPU are opened.
"""
import copy, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r.xp_a16_publish_cpu import (
    GEN_HEAD, PREP_HEAD, MANIFEST_SHA, GPU_H,
    verify_metadata, verify_chunk_proofs, near)
from scripts.track_r.xp_a16_contract import build_chunks

def fixture():
    prep={"status":"XP_A16_PREP_PASS","head":PREP_HEAD,
      "manifest_sha256":MANIFEST_SHA,"n_images":180,"n_roots":2160,
      "gpu_work_performed":False,"outcomes_embargoed":True}
    mf={"stage":"XP-A16","head":PREP_HEAD,"n_images":180,
        "n_roots":2160,"gt_paths_supplied":False}
    smoke={"status":"XP_A16_SMOKE_PASS",
      "image_aligned_chosen_starts":[0,1200],"new_roots":24,
      "new_image_counts":{"S1_b22":1,"S2_b24_dev80":1},
      "outcomes_embargoed":True,"gt_read":False,
      "conservative_projected_gpu_hours":4.1}
    complete={"status":"XP_A16_GENERATION_PASS","stage":"XP-A16",
      "head":GEN_HEAD,"prep_head":PREP_HEAD,"manifest_sha256":MANIFEST_SHA,
      "n_images":180,"historical_roots_reused_without_regeneration":720,
      "new_roots":2160,"n_chunks":38,"n_smoke_roots":24,
      "run_indices":list(range(4,16)),
      "per_cohort_new_roots":{"S1_b22":1200,"S2_b24_dev80":960},
      "workers_per_physical_gpu":1,"physical_gpu_used":[0,1,2,3],
      "strict_gt_free_generation":True,
      "historical_output_paths_not_supplied":True,
      "quality_metrics_generated":False,
      "quality_scoring_authorized":False,
      "outcomes_embargoed":True,
      "theory_freeze_required_for_scoring":"TH1_PREDICTIONS_v1.json",
      "clean_timing_eligible":True,
      "total_gpu_hours":GPU_H,"total_gpu_seconds":GPU_H*3600,
      "smoke_gate":smoke,
      "per_device_timing":{str(i):{"timing_class":"CLEAN",
           "timing_valid":True,"foreign_pids":[],"reasons":[],
           "n_samples":17} for i in range(4)}}
    status={"state":"PASS","phase":"COMPLETE","reason":None,
      "head":GEN_HEAD,"manifest_sha256":MANIFEST_SHA,
      "completed_roots":2160,"completed_chunks":38,
      "running_chunks":0,"active_pids":[],
      "outcomes_embargoed":True,"clean_timing_eligible":True,
      "gpu_hours":GPU_H}
    return prep,mf,status,smoke,complete

class PublishA16(unittest.TestCase):
    def test_complete_operational_provenance_only(self):
        self.assertTrue(verify_metadata(*fixture()))
    def test_embargo_or_gt_violation_rejected(self):
        p,m,s,sm,c=fixture()
        c["quality_scoring_authorized"]=True
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_COMPLETE"):
            verify_metadata(p,m,s,sm,c)
        p,m,s,sm,c=fixture()
        sm["gt_read"]=True
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_SMOKE"):
            verify_metadata(p,m,s,sm,c)
    def test_contamination_or_missing_roots_rejected(self):
        p,m,s,sm,c=fixture()
        c["per_device_timing"]["3"]["foreign_pids"]=[12345]
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_TIMING_3"):
            verify_metadata(p,m,s,sm,c)
        p,m,s,sm,c=fixture()
        s["completed_roots"]=2159
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_STATUS"):
            verify_metadata(p,m,s,sm,c)
    def test_chunk_partition_not_inferred_from_total_only(self):
        p,m,s,sm,c=fixture()
        smoke,main=build_chunks()
        c["chunks"]=[{"start":start,"count":n,"gpu":0}
                     for start,n in sorted(smoke+main)]
        c["chunks"][-1]["count"]=36
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_CHUNK_PARTITION"):
            verify_chunk_proofs(c,[])
    def test_receipt_and_file_sha_must_match(self):
        p,m,s,sm,c=fixture()
        smoke,main=build_chunks()
        c["chunks"]=[{"start":start,"count":n,"gpu":0,
          "chunk_metadata_sha256":"a"*64,
          "native_chunk_sha256":"b"*64}
          for start,n in sorted(smoke+main)]
        tasks=[{"uid":f"synthetic{idx}","seed":idx,
                "rep":(idx%12)+4,
                "measurement_sha256":"f"*64}
               for idx in range(2160)]
        def read_file(path):
            path=Path(path)
            if path.name=="R0_5_CHUNK_META.json":
                stem=path.parent.name.split("_")
                start,n=int(stem[1]),int(stem[2])
                return {"head":GEN_HEAD,"root_sha256":MANIFEST_SHA,
                        "gpu":0,"start":start,"count":n,
                        "uids":[x["uid"] for x in tasks[start:start+n]]}
            if path.name=="R0_5_CHUNK_COMPLETE.json":
                stem=path.parent.name.split("_")
                start,n=int(stem[1]),int(stem[2])
                return {"status":"R0_5_CHUNK_PASS",
                        "head":GEN_HEAD,"start":start,"count":n,
                        "new":n,"reused":0,"meta_sha256":"a"*64}
            return {}
        def hash_file(path):
            if path.name=="R0_5_CHUNK_META.json":return "a"*64
            if path.name=="R0_5_CHUNK_COMPLETE.json":return "b"*64
            raise AssertionError("unexpected file")
        with self.assertRaisesRegex(RuntimeError,"PUBLISH_ROOT"):
            verify_chunk_proofs(c,tasks,read_file=read_file,
                                hash_file=hash_file)
    def test_numeric_and_chunk_coverage(self):
        self.assertTrue(near(GPU_H,GPU_H,1e-9))
        self.assertFalse(near("nan",GPU_H))
        smoke,main=build_chunks()
        self.assertEqual(len(smoke+main),38)
        self.assertEqual(sum(n for _,n in smoke+main),2160)

if __name__=="__main__":
    unittest.main()
