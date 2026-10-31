"""A16 planner handoff: no GPU / no dataset dependencies in unit tests."""
import ast
import unittest
from pathlib import Path

from scripts.track_r.xp_a18_xpi_particle_metrics_cpu import rows_from_stage
from scripts.track_r.xp_a18_xpn_image_inventory_cpu import EXPECTED, D1_EXPECTED

ROOT=Path(__file__).resolve().parents[1]


class A18PlannerTests(unittest.TestCase):
    def test_xpn_manifest_id_universe(self):
        self.assertEqual(len(EXPECTED),100)
        self.assertEqual(len(D1_EXPECTED),100)
        self.assertTrue(set(EXPECTED).isdisjoint(D1_EXPECTED))
        self.assertEqual(min(EXPECTED.values()),"00048900")
        self.assertEqual(max(EXPECTED.values()),"00048999")
        self.assertEqual(min(D1_EXPECTED),"ILSVRC2012_val_00049000.JPEG")

    def test_xpi_particle_table_exact_720_and_no_paths(self):
        ids=[("B22",f"S1_b22_{60000+i:05d}") for i in range(100)]
        ids += [("DEV80",f"S2_b24_dev80_{70000+i:05d}") for i in range(80)]
        candidates={}; clips={}; metrics={}; groups={}
        for cohort,iid in ids:
            group=[]
            for k in range(4):
                key=(iid,k)
                sc=dict(cohort=cohort,image_id=iid,run=str(k),
                    uid=f"{iid}|particle{k}",label="G",
                    measurement_sha256="a"*64,
                    psnr_raw="28.5",rho_signed="0.012")
                candidates[key]=sc
                clips[key]=dict(cohort=cohort,image_id=iid,
                    run=str(k),label="G",face_choice="111",
                    face_margin="-0.07")
                metrics[key]=dict(cohort=cohort,image_id=iid,
                    run=k,uid=sc["uid"],SSIM_raw=0.81,
                    LPIPS_VGG_raw=0.18)
                group.append(metrics[key])
            groups[cohort,iid]=group
        data=dict(stage="G3",score_rec=dict(n_images=180,n_candidates=720),
                  candidates=candidates,clips=clips,metrics=metrics,
                  image_groups=groups)
        rows=rows_from_stage(data)
        self.assertEqual(len(rows),720)
        self.assertEqual(len({r["image_id"] for r in rows}),180)
        self.assertEqual(set(rows[0]),{"image_id","particle","label",
             "PSNR","SSIM","VGG_LPIPS","rho","face_choice","face_margin"})
        self.assertEqual(rows[0]["face_choice"],"111")

    def test_fail_on_duplicate_signed_measurement_for_same_image(self):
        ids={};cc={};mm={};groups={}
        # Empty collection must fail; there is no permissive empty return.
        d=dict(stage="G3",score_rec=dict(n_images=180,n_candidates=720),
               candidates=ids,clips=cc,metrics=mm,image_groups=groups)
        with self.assertRaises(RuntimeError):
            rows_from_stage(d)

    def test_xpn_isolated_noise_seed_namespace(self):
        from scripts.track_r.xp_a18_xpn_decode_isolated import seed
        from scripts.track_r.xp_t_imagenet_prepare_cpu import seed as d1_seed
        name="ILSVRC2012_val_00048900.JPEG"
        self.assertEqual(seed("XPMEAS|imagenet_n",name),
                         d1_seed("XPMEAS|imagenet_n",name))
        self.assertNotEqual(seed("XPMEAS|imagenet_n",name),
                            d1_seed("XPMEAS|imagenet",name))

    def test_xpn_cpu_stages_keep_raw_gt_out_of_gpu(self):
        root=ROOT/"scripts"/"track_r"
        helper=(root/"xp_a18_xpn_decode_isolated.py").read_text()
        prep=(root/"xp_a18_xpn_prepare_y_cpu.py").read_text()
        audit=(root/"xp_a18_xpn_audit_y_cpu.py").read_text()
        self.assertIn("sys.flags.isolated==1",helper)
        self.assertIn('source==Path.cwd()',helper)
        self.assertIn('with Image.open(p) as im:',helper)
        self.assertIn('torch.Generator(device="cpu").manual_seed(mseed)',helper)
        self.assertIn('"XPMEAS|imagenet_n"',helper)
        self.assertIn('gt_paths_supplied=False',helper)
        self.assertIn('cmd=[str(PY),"-I","-B"',prep)
        self.assertIn('PINNED_SOURCE="5fa348229a35899f39aa06326840e9a8592eea6e882450bbbd0068895cee371b"',prep)
        self.assertIn('independent_GT_Y_reaudit_required=True',prep)
        self.assertIn('torch.load(gp,map_location="cpu",weights_only=True)',audit)
        self.assertIn('D1_two_root_replay_still_required=True',audit)

    def test_xpn_replay_pixel_exact_historical_and_budget(self):
        src=(ROOT/"scripts/track_r/xp_a18_xpn_replay_cpu.py").read_text()
        self.assertIn("A12_G1A_HISTORICAL_REPLAY2",src)
        self.assertIn('old["status"]=="R0_5_CHUNK_PASS"',src)
        self.assertIn("ImageChops.difference(im0,im1).getbbox() is None",src)
        self.assertIn("choose_gpu(cards,8192",src)
        self.assertIn('fraction=.80,clean_only=False',src)
        self.assertIn("REPLAY_CAP=1.",src)
        self.assertIn("XPN_CAP=20.",src)
        self.assertIn("not unknown and nsamples>0",src)
        self.assertIn('no_XPN_candidates_generated=True',src)
        self.assertNotIn("GT_ONLY_CPU.json",src)

    def test_xpn_6400_manifest_phase_sealing_and_native_algorithm(self):
        from scripts.track_r.xp_a18_xpn_contract import PHASES
        self.assertEqual(PHASES,{
            "A18_XPN_PILOT_0_3":(0,4,400),
            "A18_XPN_MAIN_4_15":(4,16,1200),
            "A18_XPN_LATE_16_63":(16,64,4800)})
        self.assertEqual(sum(x[2] for x in PHASES.values()),6400)
        prep=(ROOT/"scripts/track_r/xp_a18_xpn_prepare_tasks_cpu.py").read_text()
        worker=(ROOT/"scripts/track_r/xp_a12_campaign_fast_gpu.py").read_text()
        self.assertIn('"XP_N_TASKS_PREP_RECEIPT.json"',prep)
        self.assertIn('a.cheng_go=="GO"',prep)
        self.assertIn("no_GT_or_candidate_outcomes_read=True",prep)
        self.assertIn("validate_xpn_source(roots,a.roots_sha,manifest,rows",worker)
        self.assertIn("network,x0,operator,y,evaluator=None",worker)
        self.assertIn("seed=int(row[\"seed\"])",worker)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG",(
            ROOT/"scripts/track_r/xp_a18_xpn_replay_cpu.py").read_text())

    def test_xpn_autopilot_90_jobs_v5_forecast_freeze_and_costs(self):
        from scripts.track_r.xp_a18_xpn_autopilot import (
            PHASES,PHASE_BATCH,PHASE_FILES,STAGE_CAP_H,PROJECT_CAP_H,
            v5_gate,V5_CODE_CUTOFF)
        from unittest.mock import patch
        import datetime
        from zoneinfo import ZoneInfo
        self.assertEqual(sum(PHASES[z][2]//PHASE_BATCH[z]
                             for z in PHASES),90)
        self.assertEqual(sum(PHASES[z][2] for z in PHASES),6400)
        self.assertEqual(STAGE_CAP_H,20.)
        self.assertEqual(PROJECT_CAP_H,400.)
        self.assertEqual(len(PHASE_FILES),3)
        with patch("scripts.track_r.xp_a18_xpn_autopilot.v5_code_freeze",
                   return_value=None):
            with patch("scripts.track_r.xp_a18_xpn_autopilot.localnow",
                       return_value=datetime.datetime(
                           2026,11,1,10,0,tzinfo=ZoneInfo("America/Detroit"))):
                self.assertIsNone(v5_gate(ROOT))
            with patch("scripts.track_r.xp_a18_xpn_autopilot.localnow",
                       return_value=V5_CODE_CUTOFF):
                self.assertEqual(v5_gate(ROOT)["status"],
                    "V5_FORECAST_CODE_LAPSED_NO_OPTIONAL_V5_TEST")
        code=(ROOT/"scripts/track_r/xp_a18_xpn_autopilot.py").read_text()
        self.assertIn("no_GT_scoring_or_prediction_access=True",code)
        self.assertIn("outcomes_4_to_63_still_sealed=True",code)
        self.assertIn("A18_XPN_NATIVE_GPU_ATTEMPT_CHARGED",code)
        self.assertIn("XP_A12_CAMPAIGN_CONTROLLER.lock",code)
        self.assertIn("fraction=.80,clean_only=False",code)
        self.assertIn("RUNNING",code)
        self.assertNotIn("gt_path",code)

    def test_python_syntax_and_go_data_isolation(self):
        for filename in ("xp_a18_xpi_particle_metrics_cpu.py",
                         "xp_a18_xpn_image_inventory_cpu.py",
                         "xp_a18_xpn_decode_isolated.py",
                         "xp_a18_xpn_prepare_y_cpu.py",
                         "xp_a18_xpn_audit_y_cpu.py",
                         "xp_a18_xpn_replay_cpu.py",
                         "xp_a18_xpn_contract.py",
                         "xp_a18_xpn_prepare_tasks_cpu.py",
                         "xp_a12_campaign_fast_gpu.py",
                         "xp_a18_xpn_autopilot.py"):
            path=ROOT/"scripts"/"track_r"/filename
            ast.parse(path.read_text(),filename=str(path))
        inventory=(ROOT/"scripts/track_r/xp_a18_xpn_image_inventory_cpu.py").read_text()
        self.assertIn('a.cheng_go=="GO"',inventory)
        self.assertIn("A18_XPN_EXACT_D1_HISTORIC_IDS",inventory)
        self.assertIn("A18_XPN_MUST_NOT_OVERLAP_D1",inventory)
        self.assertIn("GT_tensor_SHA256_pending=True",inventory)
        self.assertNotIn("Image.open(",inventory)


if __name__=="__main__":
    unittest.main()
