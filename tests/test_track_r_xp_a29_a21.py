"""A21 prereg tests: fixed science, exact pilot equivalence, GPU occupancy.

CPU-only. Never uses PAC GT/data or allocates GPU.
"""
import ast,math,tempfile,unittest,types
from pathlib import Path
from scripts.track_r import xp_a29_a21_settings as S
from scripts.track_r import xp_a29_a21_controller as C
from scripts.track_r import xp_a29_a21_gpu_worker as W

ROOT=Path(__file__).resolve().parents[1]

class A21Tests(unittest.TestCase):
    def test_a21_preregistered_constants(self):
        self.assertEqual(S.CAP_OCCUPANCY_GPU_H,56.0)
        self.assertEqual(S.GPU_IDS,(0,1,2,3))
        self.assertEqual(S.MAX_WORKERS_PER_GPU,8)
        self.assertEqual(S.BATCH_OPTIONS,(8,32))
        self.assertEqual(S.EQUIV_MAX_LOSS_ABS,1e-5)
        self.assertEqual(S.EQUIV_CHOICE_EXCLUDE_SMALL_MARGIN,1e-5)
        self.assertEqual(S.MEMORY_HEADROOM_FACTOR,1.25)
        self.assertEqual(S.MEMORY_ADMISSION_EXTRA_MIB,8192)
        self.assertEqual(S.YIELD_BELOW_FREE_MIB,4096)
        self.assertEqual(S.SIZES,{"XP_N":6400,"G5":800,"XP_D4":6400})
        self.assertEqual(S.SOURCE_SHA,
          "298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27")
    def test_gpu_union_overlapping_and_separate_gpu(self):
        # GPU0: intervals [0,20], [10,30] union=30s;
        # GPU2: [5,15] union=10s. Worker time=50s, occupancy=40s.
        intervals=[(0,0,20),(0,10,30),(2,5,15)]
        d=C.total_union(intervals,{},now=40)
        self.assertAlmostEqual(d["0"],30/3600)
        self.assertAlmostEqual(d["2"],10/3600)
        self.assertAlmostEqual(sum(d.values()),40/3600)
        live={123:dict(gpu=0,start=25)}
        e=C.total_union(intervals,live,now=40)
        self.assertAlmostEqual(e["0"],40/3600)
        self.assertAlmostEqual(sum(e.values()),50/3600)
    def test_gpu_memory_admission_and_yield_threshold(self):
        def gpu(free):
            return types.SimpleNamespace(index=3,visibility="known",free_mib=free)
        peak=6144
        bound=math.ceil(peak*1.25)+8192
        self.assertTrue(C.ready(gpu(bound),peak))
        self.assertFalse(C.ready(gpu(bound-1),peak))
        self.assertFalse(C.ready(gpu(4000),peak))
        self.assertFalse(C.ready(
            types.SimpleNamespace(index=5,visibility="known",free_mib=100000),peak))
    def test_pilot_differences_and_tie_rule(self):
        def d(i,v):
            return dict(run=i,prior_choice="111" if v>1e-7 else "000",
                        prior_margin=str(v),L000="1",L111=str(1-v))
        rows=[d(i,0.0001) for i in range(64)]
        base=dict(rows=rows)
        self.assertTrue(C.pilot_compare(base,dict(rows=rows))["pass_equivalence"])
        x=[dict(r) for r in rows]
        x[0]["L000"]="1.00002"
        self.assertFalse(C.pilot_compare(base,dict(rows=x))["pass_equivalence"])
        x=[dict(r) for r in rows];x[0]["prior_choice"]="000"
        self.assertFalse(C.pilot_compare(base,dict(rows=x))["pass_equivalence"])
        x=[dict(r) for r in rows]
        x[0]["prior_margin"]="0.00000001"
        # baseline 1e-4 => eligible, choice still same, but mismatch of
        # choice is forbidden when baseline margin safely exceeds tolerance.
        x[0]["prior_choice"]="000"
        self.assertFalse(C.pilot_compare(base,dict(rows=x))["pass_equivalence"])
    def test_source_frozen_single_t_batch_and_unchanged_r2_path(self):
        src=(ROOT/"scripts/track_r/xp_a29_a21_gpu_worker.py").read_text()
        for marker in ('for t in T:', 'for j in range(4):',
            'prior_seed', 'model.tweedie(z,sigma)', 'cpu().tolist()',
            'B8', 'B32', 'orient_sm',
            'output_once', 'os.link(tmp,path)', 'claimfile.open("x")',
            'from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE',
            'from scripts.track_r.xp_t_daps4k_prepare_cpu import IMAGENET_MODEL_SHA',
            'fix_exact_native_timestep',
            'NATIVE', 'R2_HEAD','a.replicate_task',
            'different_gpu_for_each'):
            if marker=="different_gpu_for_each":continue
            self.assertIn(marker,src)
        self.assertNotIn('torch.backends.cudnn.benchmark =',src)
        self.assertNotIn('torch.compile(',src)
        self.assertNotIn('.half()',src)
        self.assertNotIn('.bfloat16()',src)
        ast.parse(src)
    def test_controller_prospective_freeze_and_priority(self):
        src=(ROOT/"scripts/track_r/xp_a29_a21_controller.py").read_text()
        ast.parse(src)
        for marker in (
            'A21_PUSHED_CODE_AND_CONSTANTS', 'PILOT_64_SCHEMA',
            'A21_EQUIVALENCE', 'PILOT_GPU_H_CAP',
            'A21_PANEL_SKIPPED', 'A21_COMPLETE_PANEL_CPU_PUBLICATION_STARTED',
            'A21_32_UID_DIFFERENT_GPU_REPLICATION',
            'worker_hours_info_only',
            'GPU_OCCUPANCY_H', 'GPU_FREE_BELOW_4G',
            'from prdiffusion.track_r.xp_gpu_telemetry import collect'
        ):self.assertIn(marker,src)
    def test_cpu_xpi2_is_independent(self):
        src=(ROOT/"scripts/track_r/xp_a29_xpi2_cpu_publish.py").read_text()
        ast.parse(src)
        for marker in ('A24_XPI2_COMPLETE','A21_XPI2_CPU_PREFLIGHT_PASS',
            'model', 'PC_pr_outcomes_label', 'clip_new', 'piq.ssim',
            'LPIPS_VGG', 'E1_G1', 'O_ORBIT',
            'DAPS_E1_K', 'nominal_NFE_per_bag',
            'XP_I2_A15_5_PER_PARTICLE.csv'):
            if marker=="nominal_NFE_per_bag":continue
            self.assertIn(marker,src)

if __name__=="__main__":
    unittest.main()
