"""CPU-only SITCOM native replay manifest and output-isolation tests."""
from __future__ import annotations
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.track_r.xp_sitcom_replay_gpu import check_manifest,folder,PREP_HEAD

def manifest():
    rows=[]
    for idx in range(8):
        src="S1_b22" if idx<4 else "S2_b24_dev80"
        image=(("60044","60140") if idx<4 else ("01113","02763"))[
            (idx%4)//2]
        rep=idx%2
        rows.append({
          "uid":f"{src}|SITCOM|{image}|"+("run" if idx<4 else "rep")+str(rep),
          "source":src,"family":"SITCOM",
          "measurement_source":src,"image_id":image,
          "rep":rep,"seed":43 if idx<4 else 1000+idx,
          "rng_mode":"sequential_post_model" if idx<4 else "independent_model_seed",
          "measurement_path":"/egr/research-pac/huang248/outputs/locked.pt",
          "measurement_sha256":"a"*64})
    return {
       "schema":"prdiff.track_r.xp.sitcom-replay-tasks.v1",
       "status":"XP_SITCOM_REPLAY_TASKS_FROZEN",
       "head":PREP_HEAD,
       "native_source_head":"275ab67efbd8146bffca20155171ba6be1169c09",
       "n_roots":8,"n_images":4,"dataset":"ffhq",
       "gt_paths_supplied":False,"historical_paths_supplied":False,
       "model_sha256":"81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa",
       "b22_master_seed":43,"b22_trajectory_count":4,
       "native_checkout_provenance":{"generation_require_isolated_pycache":True},
       "rows":rows}

class ReplayValidation(unittest.TestCase):
    def test_two_histories_in_frozen_order(self):
        rows=check_manifest(manifest(),"anything")
        self.assertEqual(len(rows),8)
        self.assertEqual([r["seed"] for r in rows[:4]],[43]*4)
        self.assertEqual([r["rng_mode"] for r in rows[4:]],
                         ["independent_model_seed"]*4)
    def test_manifest_never_contains_gt_or_history(self):
        x=manifest()
        x["rows"][0]["gt_path"]="/egr/research-pac/huang248/data/private.png"
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_REPLAY_FIELDS"):
            check_manifest(x,"hash")
        x=manifest()
        x["historical_paths_supplied"]=True
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_REPLAY_MANIFEST"):
            check_manifest(x,"hash")
    def test_seed_collision_and_order_rejected(self):
        x=manifest()
        x["rows"][0]["seed"]=44
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_REPLAY_ORDER_OR_SEED"):
            check_manifest(x,"hash")
        x=manifest()
        x["rows"][4]["image_id"]="60044"
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_REPLAY_ORDER_OR_SEED"):
            check_manifest(x,"hash")
        x=manifest()
        x["rows"][3]["rep"]=0
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_REPLAY_ORDER_OR_SEED"):
            check_manifest(x,"hash")
    def test_stable_output_namespace_separated_by_uid(self):
        rows=check_manifest(manifest(),"head")
        out=Path("/egr/research-pac/huang248/outputs/pr_diffusion/track_r"
                 "/xp_sitcom_replay_20261026T223925Z/gpu/b22")
        paths=[folder(out,x) for x in rows[:4]]
        self.assertEqual(len(set(paths)),4)
        self.assertTrue(all(p.parent==out for p in paths))
        self.assertTrue(all(p.name.startswith("root_") for p in paths))

if __name__=="__main__":
    unittest.main()
