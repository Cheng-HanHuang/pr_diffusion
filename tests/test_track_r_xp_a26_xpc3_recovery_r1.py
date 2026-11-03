"""A20 R1: exercise exact failed scope and GPU 0 hard exclusion.

No GT access, model load, GPU allocation, scoring, or network required.
Must PASS before R1 repaired code SHA registration.
"""
from __future__ import annotations
import ast,hashlib,importlib,types,unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA="298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27"
FREEZE_SHA="a6d37d9c4bd319a91cbe4704d8e3f4ac468517f0a56070c83bd6c5c565c980ce"
SOURCE=ROOT/"docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
FROZEN=ROOT/"docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class XPCA20R1(unittest.TestCase):
    def test_original_source_science_bytes_never_touched(self):
        import json
        self.assertEqual(digest(SOURCE),SOURCE_SHA)
        frozen=json.loads(FROZEN.read_text())
        self.assertEqual(frozen["source_manifest_sha256"],SOURCE_SHA)
        self.assertEqual(frozen["scorer_script_sha256"],FREEZE_SHA)
        self.assertEqual(frozen["timesteps"],[300,400,500,600])
        self.assertEqual(frozen["draws"],[0,1,2,3])
        self.assertEqual(frozen["no_2forward_secondary"],True)
        self.assertEqual(frozen["checkpoint_sha256"],
            "a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1")

    def test_run_calls_source_before_loading_model(self):
        # Regression: previous run assigned a local source later in function.
        m=importlib.import_module("scripts.track_r.xp_a25_xpc3_mid_gpu")
        class ReachedSource(Exception):pass
        with mock.patch.object(m,"source",side_effect=ReachedSource):
            with self.assertRaises(ReachedSource):
                m.run(types.SimpleNamespace())
        ast.parse((ROOT/"scripts/track_r/xp_a25_xpc3_mid_gpu.py").read_text())

    def test_source_is_not_assigned_inside_run_scope(self):
        m=ast.parse((ROOT/"scripts/track_r/xp_a25_xpc3_mid_gpu.py").read_text())
        fn=next(x for x in m.body if isinstance(x,ast.FunctionDef) and x.name=="run")
        for x in ast.walk(fn):
            if isinstance(x,ast.Name) and x.id=="source":
                self.assertIsInstance(x.ctx,ast.Load)

    def test_physical_gpu_0_is_never_admitted(self):
        controller=importlib.import_module("scripts.track_r.xp_a25_xpc3_autopilot")
        worker=importlib.import_module("scripts.track_r.xp_a25_xpc3_mid_gpu")
        self.assertEqual(controller.ALLOWED_PHYSICAL_GPUS,(1,2,3))
        self.assertEqual(worker.ALLOWED_PHYSICAL_GPUS,(1,2,3))
        cards=[types.SimpleNamespace(index=i,visibility="known",foreign_pids=[])
               for i in range(4)]
        chosen=[]
        def fake_choose(available,*args,**kwargs):
            chosen.append([z.index for z in available])
            return (available[0].index if available else None),{}
        with mock.patch.object(controller,"collect",return_value=cards),\
             mock.patch.object(controller,"choose_gpu",side_effect=fake_choose):
            self.assertEqual(controller.gpu(),1)
        self.assertEqual(chosen,[[1,2,3]])

    def test_recovery_controller_accounts_failed_4_seconds(self):
        s=(ROOT/"scripts/track_r/xp_a25_xpc3_autopilot.py").read_text()
        ast.parse(s)
        for marker in (
          'PREVIOUS_FAILED_RUN=BASE/"xp_a25_xpc3_prod_20261103T002834Z"',
          'old["completed_panels"]==[]',
          'len(attempts)==1',
          'attempts[0]["gpu"]==0',
          'attempts[0]["rc"]==8',
          'not list((PREVIOUS_FAILED_RUN/"XP_N").glob("*.json"))',
          'cap_gpu_h=CAP_H,charged_gpu_h=previous_charge',
          'recovery_registry_sha256',
          'corrected_controller_sha256',
          'in ALLOWED_PHYSICAL_GPUS',
        ):
            self.assertIn(marker,s)

    def test_exact_smid_and_immutable_repair_registration(self):
        s=(ROOT/"scripts/track_r/xp_a25_xpc3_mid_gpu.py").read_text()
        for marker in (
            'RECOVERY_REL=', 'R1_CODE_SHA_AND_SCIENCE_UNCHANGED',
            'ORIGINAL_A20_REGISTERED_CODE_UNCHANGED',
            'corrected_scorer_sha256', 'margin>1e-7',
            'n==16', 'frozen_alpha_schedule,prior_seed',
            'saved_reconstruction=torch.load(',
            'native.sigma_inv=exact_index',
            'model.tweedie(z,sigma)',
            'a.gpu in ALLOWED_PHYSICAL_GPUS',
        ):
            self.assertIn(marker,s)
        self.assertNotIn('source=torch.load(',s)

if __name__=="__main__":
    unittest.main()
