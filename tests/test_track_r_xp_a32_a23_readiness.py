"""A23 accepted-source provenance checks; pure Python (no GPU or PAC)."""
import ast
import unittest
from pathlib import Path
from scripts.track_r.xp_a32_a23_readiness_cpu import (
    POLICIES,
    G7_C_MANIFEST_SHA,
    G7_C2_MANIFEST_SHA,
    PROJECT_USED,
    G7_USED,
    check_clip_table,
    validate_gt_images,
)

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/track_r/xp_a32_a23_readiness_cpu.py"

class A23ReadinessTests(unittest.TestCase):
    def test_source_script_is_cpu_only_and_valid(self):
        src=SCRIPT.read_text()
        ast.parse(src)
        self.assertNotIn("torch.cuda.",src)
        self.assertNotIn("cuda:0",src)
        self.assertNotIn("subprocess.Popen(",src)
        self.assertIn("nvidia-smi",src) # read-only hardware snapshot
        self.assertIn("read_only_gpu_snapshot",src)
        self.assertIn("original_g7 = G7_PUB_ROOT / G7_RETURN",src)
        self.assertIn("pinned_publication(TAB3_PUB_ROOT, TAB3_PUB_COMMIT)",src)
        self.assertEqual(POLICIES,(
            "S_mid_global2","S_mid_all8","2_forward","UnivFD"))
    def test_frozen_manifest_historical_hash_constants(self):
        self.assertEqual(len(G7_C_MANIFEST_SHA),64)
        self.assertEqual(len(G7_C2_MANIFEST_SHA),64)
        self.assertAlmostEqual(G7_USED+(10-G7_USED),10)
        self.assertAlmostEqual(PROJECT_USED+(400-PROJECT_USED),400)
    def test_gt_manifest_does_not_accept_duplicates(self):
        images=[]
        for i in range(180):
            images.append(dict(dataset="ffhq",basename=f"img_{i}",
                    source_path="/tmp/image-does-not-exist",
                    source_sha256="0"*64))
        for i in range(100):
            images.append(dict(dataset="imagenet",basename=f"img_{i}",
                    source_path="/tmp/image-does-not-exist",
                    source_sha256="0"*64))
        manifest=dict(status="XP_C_GT_IMAGES_FROZEN",n_images=280,
            n_ffhq=180,n_imagenet=100,
            diagnostic_gt_only=True,scoring_or_GPU_work_performed=False,
            images=images)
        with self.assertRaisesRegex(RuntimeError,"PROTECTED_PATH"):
            validate_gt_images(manifest,verify_source_bytes=False)
        images[1]["basename"]=images[0]["basename"]
        with self.assertRaisesRegex(RuntimeError,"PROTECTED_PATH|DUPLICATE"):
            validate_gt_images(manifest,verify_source_bytes=False)
    def test_exact_clips_8_of_100_and_0_of_180(self):
        keys={}
        rows=[]
        for i in range(180):
            keys[("ffhq",str(i))]="0"*64
            rows.append(dict(dataset="ffhq",basename=str(i),
                original_source_sha256="0"*64,diagnostic_gt_only="True",
                g1_orientation="000",g2_orientation="000",
                face_orientation="000"))
        for i in range(100):
            keys[("imagenet",str(i))]="0"*64
            rows.append(dict(dataset="imagenet",basename=str(i),
                original_source_sha256="0"*64,diagnostic_gt_only="True",
                g1_orientation="111" if i<8 else "000",
                g2_orientation="111" if i<8 else "000"))
        stats=check_clip_table(rows,keys)
        self.assertEqual(stats["ffhq_face_flipped"],0)
        self.assertEqual(stats["imagenet_g1_flipped"],8)
        self.assertEqual(stats["imagenet_g2_flipped"],8)
        rows[-1]["g1_orientation"]="111"
        with self.assertRaisesRegex(RuntimeError,"PLANNER_CLIP_GT_FIGURES"):
            check_clip_table(rows,keys)

if __name__=="__main__":
    unittest.main()
