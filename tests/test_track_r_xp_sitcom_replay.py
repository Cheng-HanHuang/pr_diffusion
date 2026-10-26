"""Synthetic CPU-only SITCOM B22/B24 replay provenance/selection checks."""
from __future__ import annotations
import unittest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_sitcom_replay_prepare_cpu as prep
from scripts.track_r.xp_sitcom_replay_prepare_cpu import selection,settings,check_native_checkout,SITCOM_HEAD

class SitcomReplayPreflightTests(unittest.TestCase):
    def test_select_two_first_ids_and_two_runs_per_cohort(self):
        rows=[]
        for source,n in (("S1_b22",100),("S2_b24_dev80",80)):
            for j in range(n):
                for rep in range(4):
                    rows.append({
                        "uid":f"{source}|SITCOM|{j:05d}|r{rep}",
                        "source":source,"family":"SITCOM",
                        "image_id":f"{j:05d}","run_index":rep,
                    })
        rows.extend([{
            "uid":f"filler{i}","source":"UNRELATED","family":"UNRELATED",
            "image_id":"other","run_index":0
        } for i in range(2520)])
        got=selection(rows)
        self.assertEqual(len(got),8)
        self.assertEqual([(r["source"],r["image_id"],r["run_index"]) for r in got],
            [(src,f"{j:05d}",rep) for src in ("S1_b22","S2_b24_dev80")
             for j in range(2) for rep in (0,1)])
    def test_missing_historical_run_rejected(self):
        rows=[]
        for src,n in (("S1_b22",100),("S2_b24_dev80",80)):
            for j in range(n):
                for rep in range(4):
                    rows.append({"uid":f"{src}-{j}-{rep}",
                                 "source":src,"family":"SITCOM",
                                 "image_id":f"{j:05d}","run_index":rep})
        rows.extend({"uid":f"f{i}","source":"other","family":"other","image_id":"-"}
                    for i in range(2520))
        rows[0]["run_index"]=2
        with self.assertRaisesRegex(RuntimeError,"STOP_XP_SITCOM_HISTORICAL_FOUR_RUNS"):
            selection(rows)
    def test_known_tracked_pycache_and_unused_cli_dirty_allowed(self):
        changes=(
            " M forward_operator/__pycache__/__init__.cpython-38.pyc\n"
            " M model/ddpm/__pycache__/unet.cpython-38.pyc\n"
            " M posterior_sample.py\n"
        )
        result=check_native_checkout(SITCOM_HEAD,changes,
                                      "1\t1\tposterior_sample.py\n")
        self.assertTrue(result["native_solver_model_operator_source_clean"])
        self.assertTrue(result["generation_require_isolated_pycache"])
        self.assertEqual(result["changed_python_source_files"],
                         ["posterior_sample.py"])
        self.assertEqual(len(result["tracked_modified"]),3)

    def test_any_modified_native_sampler_or_unlisted_source_rejected(self):
        for path in ("sampler.py","model/ddpm/unet.py",
                     "forward_operator/__init__.py","posterior_sample_copy.py",
                     "model/__pycache__/custom.cpython-38.pyc"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(RuntimeError,
                                            "STOP_XP_SITCOM_NATIVE_SOURCE_CHANGED"):
                    check_native_checkout(SITCOM_HEAD," M "+path+"\n","")

    def test_staged_changes_wrong_head_and_cli_shape_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"STOP_XP_SITCOM_NATIVE_HEAD"):
            check_native_checkout("0"*40,"","")
        with self.assertRaisesRegex(RuntimeError,"STOP_XP_SITCOM_DIRTY_STATUS"):
            check_native_checkout(SITCOM_HEAD,"M  posterior_sample.py\n",
                                  "1\t1\tposterior_sample.py\n")
        with self.assertRaisesRegex(RuntimeError,"STOP_XP_SITCOM_CLI_EDIT_DIFFERENT"):
            check_native_checkout(SITCOM_HEAD," M posterior_sample.py\n",
                                  "3\t2\tposterior_sample.py\n")
        clean=check_native_checkout(SITCOM_HEAD,"","")
        self.assertFalse(clean["tracked_modified"])

    def test_exact_native_checkpoint_symlink_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            model=root/"models/ffhq_10m.pt"
            model.parent.mkdir()
            model.write_bytes(b"fixture-checkpoint")
            native=root/"external/SITCOM_ODE/checkpoint/ffhq256.pt"
            native.parent.mkdir(parents=True)
            native.symlink_to(model)
            expected=hashlib.sha256(model.read_bytes()).hexdigest()
            with (patch.object(prep,"ROOT",root),
                  patch.object(prep,"SITCOM",native.parent.parent),
                  patch.object(prep,"MODEL_SHA",expected)):
                got=prep.external_checkpoint_provenance(native)
                self.assertTrue(got["is_symlink"])
                self.assertEqual(got["target_path"],str(model))
                self.assertEqual(got["sha256"],expected)
                # General-purpose input verifier continues to refuse symlinks.
                with self.assertRaisesRegex(RuntimeError,
                                            "STOP_XP_SITCOM_BAD_SOURCE"):
                    prep.safe(native)
                native.unlink()
                native.write_bytes(model.read_bytes())
                got=prep.external_checkpoint_provenance(native)
                self.assertFalse(got["is_symlink"])

    def test_checkpoint_symlink_outside_pac_root_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            top=Path(parent).resolve()
            root=top/"research"
            native=root/"external/SITCOM_ODE/checkpoint/ffhq256.pt"
            native.parent.mkdir(parents=True)
            external=top/"outside.pt"
            external.write_bytes(b"same-checkpoint-content")
            native.symlink_to(external)
            expected=hashlib.sha256(external.read_bytes()).hexdigest()
            with (patch.object(prep,"ROOT",root),
                  patch.object(prep,"SITCOM",native.parent.parent),
                  patch.object(prep,"MODEL_SHA",expected)):
                with self.assertRaisesRegex(RuntimeError,
                                             "STOP_XP_SITCOM_CHECKPOINT_LINK"):
                    prep.external_checkpoint_provenance(native)

    def test_checkpoint_wrong_digest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            target=root/"models/mismatched.pt"
            target.parent.mkdir()
            target.write_bytes(b"wrong")
            native=root/"external/SITCOM_ODE/checkpoint/ffhq256.pt"
            native.parent.mkdir(parents=True)
            native.symlink_to(target)
            with (patch.object(prep,"ROOT",root),
                  patch.object(prep,"SITCOM",native.parent.parent),
                  patch.object(prep,"MODEL_SHA","0"*64)):
                with self.assertRaisesRegex(RuntimeError,
                                            "STOP_XP_SITCOM_EXTERNAL_MODEL"):
                    prep.external_checkpoint_provenance(native)

    def test_b22_native_contract_locked(self):
        m={
          "anneal_schedule":"linear","anneal_sigma_max":100.0,
          "anneal_sigma_min":0.1,"anneal_steps":200,
          "diff_schedule":"linear","diff_sigma_min":0.01,"diff_steps":5,
          "lgvd_lr":5e-5,"lgvd_lr_min_ratio":0.01,"lgvd_steps":100,
          "lgvd_tau":0.01,"timestep":"poly-7",
          "model_path_relative_to_sitcom_root":"checkpoint/ffhq256.pt",
          "measurement_preprocessing":"none",
          "master_seed":43,"trajectory_count":4,"selector_step":160,
        }
        c={"sitcom":m}
        self.assertEqual(settings(c)["anneal_steps"],200)
        c["sitcom"]["master_seed"]=44
        with self.assertRaisesRegex(RuntimeError,"STOP_XP_SITCOM_B22_STREAM_OR_SELECTOR"):
            settings(c)
if __name__=="__main__":
    unittest.main()
