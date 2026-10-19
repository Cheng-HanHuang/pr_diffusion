"""Synthetic-only CPU protocol source/dry-run tests; does not inspect PAC assets."""
import unittest

from scripts.track_a.a1_native_k4_dryrun import make_synthetic_plan, source_checks


class NativeDryRunTests(unittest.TestCase):
    def test_candidate_counts_seeds_and_no_execution(self):
        plan = make_synthetic_plan()
        self.assertEqual(len(plan["planned_candidate_specs"]), 16)
        self.assertEqual(plan["expected_counts"],
                         {"DAPS":4, "SITCOM":4, "NP_LF":4, "NP_S2":4})
        self.assertEqual(len(set(x["native_seed32"] for x in
                                 plan["planned_candidate_specs"])), 16)
        self.assertEqual(len(plan["mixed_np8"]), 8)
        self.assertEqual(len(plan["mixed_np4"]), 4)
        ids={x["candidate_id"] for x in plan["planned_candidate_specs"]}
        self.assertTrue(set(plan["mixed_np8"]).issubset(ids))
        self.assertEqual(plan["processes_executed"],0)
        self.assertFalse(plan["new_cohort_reserved"])
        self.assertFalse(plan["run_reserved"])
        self.assertFalse(plan["gt_loaded"])
        self.assertFalse(plan["gpu_work"])
        self.assertIsNone(plan["planned_candidate_specs"][0]["measurement_path"])
        self.assertEqual(make_synthetic_plan(),plan)
        with self.assertRaises(ValueError):
            make_synthetic_plan("22222")

    def test_source_gate_accepts_frozen_semantics(self):
        daps = '''def _b21_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_start(self, batch_size, model):
    start_seed = os.environ.get("B21_START_NOISE_SEED", "").strip()
    if start_seed:
        _b21_seed_all(int(start_seed))
    x_start = torch.randn(batch_size, *in_shape, device=device)
    return x_start


class LatentDAPS:
    pass
'''
        sit = '''class A:
    def get_start(self, ref):
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start

class B:
    def get_start(self, ref):
        x_start = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
        return x_start
'''
        s=source_checks(daps,sit)
        self.assertTrue(s["daps_seed_before_gaussian_start"])
        self.assertEqual(s["sitcom_random_shape_only_start_methods"],2)
        with self.assertRaises(ValueError):
            source_checks(daps.replace("_b21_seed_all(int(start_seed))",
                                       "_b21_seed_all(43)"),sit)
        with self.assertRaises(ValueError):
            source_checks(daps,sit.replace("torch.randn_like(ref) * self.scheduler.sigma_max",
                                         "ref.mean() + torch.randn_like(ref)"))

if __name__ == "__main__":
    unittest.main()
