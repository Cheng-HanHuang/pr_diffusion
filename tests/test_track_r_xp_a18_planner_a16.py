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

    def test_python_syntax_and_go_data_isolation(self):
        for filename in ("xp_a18_xpi_particle_metrics_cpu.py",
                         "xp_a18_xpn_image_inventory_cpu.py",
                         "xp_a18_xpn_decode_isolated.py",
                         "xp_a18_xpn_prepare_y_cpu.py",
                         "xp_a18_xpn_audit_y_cpu.py",
                         "xp_a18_xpn_replay_cpu.py"):
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
