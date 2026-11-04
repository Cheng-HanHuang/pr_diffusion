"""A34 GT-only G7 production contract, no GPU and no PAC source dependency."""
from __future__ import annotations
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_a34_a23_g7_smid_controller_cpu as C
from scripts.track_r import xp_a34_a23_g7_smid_readout_cpu as R

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/"scripts/track_r/xp_a34_a23_g7_smid_prod_gpu.py"
CONTROL=ROOT/"scripts/track_r/xp_a34_a23_g7_smid_controller_cpu.py"
READOUT=ROOT/"scripts/track_r/xp_a34_a23_g7_smid_readout_cpu.py"
PUBLISH=ROOT/"scripts/track_r/xp_a34_a23_g7_smid_publish_cpu.py"
LAUNCH=ROOT/"scripts/track_r/xp_a34_a23_g7_smid_launch.sh"

class A34G7ProdTests(unittest.TestCase):
    def test_exact_frozen_grid_budget_and_pilot_provenance(self):
        self.assertEqual(C.N,280)
        self.assertEqual(C.PILOT_HEAD,"ae09517c76d057fb076f737cd4e6d3c33c63ad66")
        self.assertEqual(C.SOURCE_SHA,"d517a8ea337c5fed8925f925bf24bae21cea60a566eb29f6795bc31e350abfde")
        self.assertEqual(C.C2_SOURCE_SHA,"75e8fe296932044eb4ff1cd30300ef58a8f48ece02acd9b22ba3aa6f3bd7c7ac")
        self.assertLess(C.MAX_NEW_WORK_H+C.C2_USED+.018,C.G7_CAP_H)
        self.assertLess(C.PROJECT_BASE+C.MAX_NEW_WORK_H+.018,C.PROJECT_CAP_H)
        self.assertLessEqual(C.MAX_WORKER_S,1800)
        self.assertEqual(C.RESULT_STATUS,"A23_G7_S_MID_GLOBAL2_GT_IMAGE_PASS")

    def test_worker_reuses_registered_A20_A21_math_no_solver(self):
        src=WORKER.read_text()
        ast.parse(src,str(WORKER))
        for key in (
            "score_group([(dict(uid=uid),x)], model, torch, alpha,",
            "prior_seed, batch=8",
            "fix_exact_native_timestep(model, alpha, torch)",
            'f\'XPCGT|{image["dataset"]}|{image["basename"]}\'',
            "RESULT_STATUS",
            "os.link(tmp,receipt_path)",
            "GT_not_passed_to_inverse_solver=True",
            "torch.cuda.max_memory_reserved",
            "A34_G7_S_MID_NATIVE_CPU_PREFLIGHT_PASS",
        ):self.assertIn(key,src)
        self.assertNotIn("sampler.sample(",src)
        self.assertNotIn("get_operator(",src)

    def test_controller_exact_reuse_and_foreign_safety(self):
        src=CONTROL.read_text()
        ast.parse(src,str(CONTROL))
        for key in (
            'for i in (0,180):',
            'r["B8_fixed_t_batched"]',
            "PILOT_DIR",
            "C2_USED+pilot_cost+MAX_NEW_WORK_H<G7_CAP_H",
            "total_union(intervals,active)",
            'LOCK_FILE.open("a+")',
            "fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)",
            "collect(own_roots=list(active))",
            'c.used_mib+MIN_FREE_MIB>int(.8*c.total_mib)',
            "start_new_session=True",
            "os.killpg(pid,signal.SIGTERM)",
            "A34_G7_S_MID_GLOBAL2_280_COMPLETE",
        ):self.assertIn(key,src)
        self.assertNotIn("A21.10",src)
        self.assertIn("signal.signal(signal.SIGTERM,stop)",src)

    def test_pilot_reuse_valid_result_schema(self):
        src={"status":C.RESULT_STATUS,"index":0,
            "dataset":"ffhq","basename":"A",
            "source_sha256":"a"*64,
            "source_manifest_sha256":C.SOURCE_SHA,
            "C2_manifest_sha256":C.C2_SOURCE_SHA,
            "source_lock_sha256":"b"*64,
            "GT_only_diagnostic":True,
            "GT_not_passed_to_inverse_solver":True,
            "selected_orientation":"000",
            "upright_kept":True,
            "B8_fixed_t_native_score":{
              "L000":"0.25","L111":"0.5","prior_margin":"-0.25",
              "prior_choice":"000","denoiser_examples":32}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/"XP_C_GT_IMAGES.json").write_text(json.dumps(
                {"images":[{"dataset":"ffhq","basename":"A",
                            "source_sha256":"a"*64}]}))
            result=root/"000.json";result.write_text(json.dumps(src))
            with patch.object(C,"FREEZE",root),patch.object(C,"LOCK",root/"lock"):
                (root/"lock").write_text("source")
                src["source_lock_sha256"]=C.sha(root/"lock")
                result.write_text(json.dumps(src))
                self.assertTrue(C.valid_result(result,0))
                src["selected_orientation"]="111"
                result.write_text(json.dumps(src))
                self.assertFalse(C.valid_result(result,0))

    def test_readout_bootstrap_deterministic(self):
        self.assertEqual(R.ci([1]*20,"ffhq"),(1.,1.))
        self.assertEqual(R.ci([0]*20,"imagenet"),(0.,0.))
        self.assertEqual(R.ci([0,1]*10,"pooled"),R.ci([0,1]*10,"pooled"))

    def test_readout_and_publisher_require_280_accepted_receipts(self):
        for p in (READOUT,PUBLISH):
            s=p.read_text()
            ast.parse(s,str(p))
            self.assertIn("280",s)
            self.assertIn("GT_only",s)
        pub=PUBLISH.read_text()
        self.assertIn("all(valid_result(",pub)
        self.assertIn("not_full_G7_completion=True",pub)
        self.assertIn("not_main_paper_data=True",pub)
        self.assertIn("remote_head(tree,branch)",pub)
        self.assertNotIn("torch.cuda",pub)

    def test_launcher_exact_gate_before_gpu(self):
        s=LAUNCH.read_text()
        self.assertIn("test -z",s)
        self.assertIn("test_track_r_xp_a34_a23_g7_prod.py",s)
        self.assertIn("--preflight-only",s)
        self.assertIn("A34_G7_S_MID_GLOBAL2_280_COMPLETE",s)
        self.assertIn("xp_a34_a23_g7_smid_publish_cpu.py",s)

if __name__=="__main__":
    unittest.main()
