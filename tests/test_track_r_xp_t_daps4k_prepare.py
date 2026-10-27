"""DAPS-4K CPU manifest: exact config and shared-y input invariants."""
from __future__ import annotations
import hashlib
import unittest
from scripts.track_r.xp_t_daps4k_prepare_cpu import (
    DAPS4K,DAPS_HEAD,TRACKED_SOURCE,new_seed,construct)

def fixture():
    ff={s:(f"/tmp/ff_{s}.pt",hashlib.sha256(s.encode()).hexdigest())
        for s in ("60044","60045","60046")}
    common=[]
    for i in range(100):
        name=f"ILSVRC2012_val_{i+49000:08d}.JPEG"
        common.append({"basename":name,
                       "measurement_path":f"/tmp/im_{i:03d}.pt",
                       "measurement_sha256":hashlib.sha256(name.encode()).hexdigest()})
    im={x["basename"]:(x["measurement_path"],x["measurement_sha256"])
        for x in common[:3]}
    return ff,im,common

class DAPSPublishedTiming(unittest.TestCase):
    def test_published_nonlinear_pr_config_not_short_command(self):
        self.assertEqual(DAPS4K["annealing_scheduler_config"]["num_steps"],400)
        self.assertEqual(DAPS4K["diffusion_scheduler_config"]["num_steps"],10)
        self.assertEqual(DAPS4K["mcmc_sampler_config"]["num_steps"],100)
        self.assertEqual(DAPS4K["mcmc_sampler_config"]["mc_algo"],"langevin")
        self.assertEqual(DAPS4K["mcmc_sampler_config"]["prior_solver"],"gaussian")
        self.assertEqual(DAPS4K["operator"],
                         {"name":"phase_retrieval","oversample":2.0,"sigma":0.05})
        self.assertEqual(DAPS4K["batch_size"],1)
        self.assertEqual(DAPS4K["method"],"DAPS-4K")
    def test_frozen_two_dataset_18_run_manifest(self):
        ff,im,common=fixture()
        rows=construct(ff,im,common)
        self.assertEqual(len(rows),18)
        self.assertEqual([r["dataset"] for r in rows],["ffhq"]*9+["imagenet"]*9)
        self.assertEqual([r["rep"] for r in rows],[0,1,2]*6)
        self.assertEqual(len({r["uid"] for r in rows}),18)
        for i in range(0,18,3):
            triplet=rows[i:i+3]
            self.assertEqual(len({r["seed"] for r in triplet}),3)
            self.assertEqual(len({r["measurement_path"] for r in triplet}),1)
            self.assertEqual(len({r["measurement_sha256"] for r in triplet}),1)
        self.assertNotEqual(rows[0]["seed"],rows[9]["seed"])
    def test_wrong_imagenet_y_fails_closed(self):
        ff,im,common=fixture()
        bad=dict(im)
        k=next(iter(bad))
        bad[k]=(bad[k][0],"0"*64)
        with self.assertRaisesRegex(RuntimeError,"COMMON_Y"):
            construct(ff,bad,common)
    def test_source_provenance_includes_solver_arch_and_operator(self):
        self.assertEqual(DAPS_HEAD,"e7a77d094167084faed19b599b96673b7bb11447")
        self.assertIn("cores/mcmc.py",TRACKED_SOURCE)
        self.assertIn("cores/scheduler.py",TRACKED_SOURCE)
        self.assertIn("model/ddpm/unet.py",TRACKED_SOURCE)
        self.assertIn("configs/task/phase_retrieval.yaml",TRACKED_SOURCE)
    def test_seed_is_prereg_deterministic(self):
        a=new_seed("ffhq","60044",0)
        self.assertEqual(a,new_seed("ffhq","60044",0))
        self.assertNotEqual(a,new_seed("ffhq","60044",1))
        self.assertNotEqual(a,new_seed("imagenet","60044",0))
        self.assertLess(a,2**31)

if __name__=="__main__":unittest.main()
