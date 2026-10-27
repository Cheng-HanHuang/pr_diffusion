"""A6 synthetic CPU-only next critical panel and scheduler gates."""
import unittest,hashlib,json,tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_a6_prepare_cpu as prep
from scripts.track_r import xp_a6_supervisor as sup
from scripts.track_r import xp_a6_d1p2_contract as d1

def fixtures():
    common={"status":"XP_IMAGENET100_Y_FROZEN","n_images":100,
            "gt_paths_supplied":False,
            "rows":[{"basename":f"im_{i:03d}.png",
                     "measurement_path":f"/egr/research-pac/huang248/input/im_{i}.pt",
                     "measurement_sha256":"a"*64}
                   for i in range(100)]}
    a16={"rows":[{"uid":f"ff{i}_{rep}","source":"S5_r0_fast",
                  "measurement_source":"S1_b22" if i<100 else "S2_b24_dev80",
                  "image_id":f"{i:05d}","rep":rep,
                  "seed":rep,"measurement_path":f"/egr/research-pac/huang248/input/ff{i}.pt",
                  "measurement_sha256":"b"*64}
                 for i in range(180) for rep in range(4,16)]}
    return common,a16

class A6Tests(unittest.TestCase):
    def test_exact_partition_and_40_smoke_roots(self):
        all_jobs=sup.jobs()
        self.assertEqual(len({j["id"] for j in all_jobs}),len(all_jobs))
        self.assertEqual(sum(j["count"] for j in all_jobs),2320)
        self.assertEqual(sum(j["count"] for j in all_jobs if j["smoke"]),40)
        for stage in sup.STAGES:
            js=[j for j in all_jobs if j["stage"]==stage]
            indices=sorted(k for j in js for k in range(j["start"],j["start"]+j["count"]))
            self.assertEqual(indices,list(range(prep.COUNTS[stage])))
            self.assertLess(sup.RESERVATION[stage],32768)
            self.assertTrue(all(j["time_class"]=="TIME_UNAWARE_GENERATION" for j in js))
        self.assertEqual(sup.STAGES,("D1_P2","D3","EPRIME"))
    def test_source_unique_and_new_domain_seeds(self):
        common,a16=fixtures()
        # Exercise seed/partition logic on synthetic aliases: check_file
        # intentionally requires real PAC files during an actual freeze.
        # Supply only a fake path verifier inside this single synthetic test.
        def fake_sha(path):
            text=str(path)
            return "a"*64 if "/im_" in text else "b"*64
        with patch.object(prep,"frozen_roots",return_value=a16), \
             patch.object(prep,"check_file",side_effect=Path), \
             patch.object(prep,"sha",side_effect=fake_sha):
            d3,e=prep.build(common,a16)
        self.assertEqual(len(d3),400)
        self.assertEqual(len(e),720)
        self.assertEqual({x["rep"] for x in d3},set(range(4)))
        self.assertEqual({x["rep"] for x in e},set(range(4)))
        self.assertEqual(len({r["uid"] for r in d3}),400)
        self.assertEqual(len({r["uid"] for r in e}),720)
        for r in e:
            k=f"XP|DAPS|ffhq_daps4k|{r['image_id']}|{r['rep']}"
            self.assertEqual(r["seed"],
               int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],"big")%(2**31))
    def test_real_path_guard_still_rejects_synthetic_paths(self):
        with self.assertRaisesRegex(RuntimeError,"MISSING_OR_PROTECTED_SOURCE"):
            prep.check_file("/egr/research-pac/huang248/input/synthetic_missing_for_a6_test.pt")

    def test_d1p2_validation_requires_exact_pinned_digest(self):
        with self.assertRaisesRegex(RuntimeError,"PINNED_SOURCE"):
            d1.validate("not_the_source","bad",{},[],"bad",0,24)
    def test_d1p2_chunk_runs_are_image_aligned(self):
        for j in sup.jobs():
            if j["stage"]=="D1_P2":
                self.assertEqual(j["start"]%12,0)
                self.assertEqual(j["count"]%12,0)
                self.assertLessEqual(j["count"],120)
    def test_smoke_fails_without_three_stages(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t)
            jobs=sup.jobs()
            with self.assertRaisesRegex(RuntimeError,"SMOKE_COVERAGE"):
                sup.verified_smokes(run,"head",set(),jobs)
    def test_output_paths_are_disjoint(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t)
            mf={stage:{"rows":[{"uid":f"{stage}-{i}"}
                    for i in range(prep.COUNTS[stage])]}
                    for stage in ("D3","EPRIME")}
            result=[]
            for job in sup.jobs():
                path=sup.output_dir(run,job,mf.get(job["stage"]))
                result.append(str(path))
            self.assertEqual(len(result),len(set(result)))
    def test_native_stage_identity_restricted(self):
        from scripts.track_r import xp_a6_native_worker as native
        self.assertEqual(native.dd.DAPS4K["diffusion_scheduler_config"]["num_steps"],10)
        self.assertTrue(callable(native.ss.generate))
        self.assertTrue(callable(native.dd.generate))

if __name__=="__main__":unittest.main()
