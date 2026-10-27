"""XP-A16 image-aligned schedule, native source, and GT-free safety tests."""
from __future__ import annotations
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r.xp_a16_contract import (
    RUN,TASKS,MANIFEST_SHA,PREP_HEAD,build_chunks,validate_a16_source)
from scripts.track_r.r0_5_prepare_cpu import native_seed
from scripts.track_r.xp_a16_supervisor import (
    chunk_path,chunk_args,environment,smoke_gate,write_x,atomic)

def mock_manifest():
    rows=[]
    for cohort,images in (("S1_b22",100),("S2_b24_dev80",80)):
        for number in range(images):
            iid=f"{number:05d}"
            for rep in range(4,16):
                rows.append({
                   "source":"S5_r0_fast",
                   "uid":f"S5_r0_fast|FAST|{cohort}_{iid}|rep{rep}",
                   "measurement_source":cohort,"image_id":iid,"rep":rep,
                   "seed":native_seed(cohort,iid,rep),
                   "measurement_path":f"/egr/research-pac/huang248/testinputs/{cohort}_{iid}.pt",
                   "measurement_sha256":f"{number:064x}",
                })
    manifest={
      "schema":"prdiff.track_r.xp.fast-tasks.v1",
      "status":"XP_FAST_TASKS_FROZEN","stage":"XP-A16",
      "head":PREP_HEAD,"dataset":"ffhq","oversample":2.0,
      "fast_head":"a9758b38906b7c7950fa04491f37f6634948db26",
      "n_roots":2160,"n_images":180,
      "historical_indices":[0,1,2,3],
      "new_indices":list(range(4,16)),"gt_paths_supplied":False,
      "outcome_embargo":"NO_K_GT4_OUTCOMES_UNTIL_TH1_V1_REGISTRY",
      "rows":rows,
    }
    return manifest,rows

class XP_A16ExecutionTests(unittest.TestCase):
    def test_38_chunks_exact_disjoint_2160(self):
        smoke,main=build_chunks()
        self.assertEqual(len(smoke),2)
        self.assertEqual(len(main),36)
        self.assertEqual(smoke,[(0,12),(1200,12)])
        indices=[k for start,count in smoke+main for k in range(start,start+count)]
        self.assertEqual(sorted(indices),list(range(2160)))
        self.assertTrue(all(a%12==0 and b%12==0 and 0<b<=60
                            for a,b in smoke+main))

    def test_full_native_source_identity(self):
        manifest,rows=mock_manifest()
        with patch("scripts.track_r.xp_a16_contract.Path.is_symlink",
                   return_value=False):
            self.assertTrue(validate_a16_source(TASKS,MANIFEST_SHA,
              manifest,rows,PREP_HEAD,1200,60))

    def test_wrong_producer_or_partial_images_fail(self):
        manifest,rows=mock_manifest()
        with self.assertRaisesRegex(RuntimeError,"SOURCE_PRODUCER_OR_SHA"):
            validate_a16_source(TASKS,MANIFEST_SHA,manifest,rows,"other",0,12)
        with self.assertRaisesRegex(RuntimeError,"IMAGE_ALIGNED_SLICE"):
            validate_a16_source(TASKS,MANIFEST_SHA,manifest,rows,PREP_HEAD,0,10)

    def test_drift_in_native_seed_or_embargo_fails(self):
        manifest,rows=mock_manifest()
        rows[0]["seed"]+=1
        with self.assertRaisesRegex(RuntimeError,"UID_RNG_OR_Y_PATH"):
            validate_a16_source(TASKS,MANIFEST_SHA,manifest,rows,PREP_HEAD,0,12)
        manifest,rows=mock_manifest()
        manifest["outcome_embargo"]="NONE"
        with self.assertRaisesRegex(RuntimeError,"PREREGISTERED_INPUT"):
            validate_a16_source(TASKS,MANIFEST_SHA,manifest,rows,PREP_HEAD,0,12)

    def test_smoke_two_cohorts_and_cap(self):
        samples=[dict(start=0,worker_allocation_seconds=54.0),
                 dict(start=1200,worker_allocation_seconds=78.0)]
        proj=smoke_gate(samples,12.0)
        self.assertAlmostEqual(proj,132/24*2160*1.75/3600)
        with self.assertRaisesRegex(RuntimeError,"SMOKE_GPUH_PROJECTION"):
            smoke_gate(samples,2.0)
        with self.assertRaisesRegex(RuntimeError,"SMOKE_COHORTS"):
            smoke_gate(list(reversed(samples)),12.0)

    def test_native_worker_args_and_json_exclusivity(self):
        argv=chunk_args("head",1200,12,3)
        self.assertIn("--roots-source-head",argv)
        self.assertIn(PREP_HEAD,argv)
        self.assertEqual(argv[argv.index("--count")+1],"12")
        self.assertEqual(argv[argv.index("--physical-gpu")+1],"3")
        self.assertEqual(environment(3,1200)["CUDA_VISIBLE_DEVICES"],"3")
        self.assertEqual(chunk_path(1200,12).parent,RUN/"gpu")
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"status.json"
            write_x(p,{"state":"first"})
            with self.assertRaises(FileExistsError):
                write_x(p,{"state":"second"})
            atomic(p,{"state":"updated"})
            self.assertIn("updated",p.read_text())

if __name__=="__main__":unittest.main()
