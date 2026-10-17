"""A1 protocol checks (stdlib only; invented synthetic five-digit IDs)."""
import unittest
from scripts.track_a.a1_protocol import (
    ProtocolStop, candidate_matrix, seed63, seed32_numpy, mixed_np4, mixed_np8,
    select_dev_ids, verify_matrix,
)


class A1ProtocolTests(unittest.TestCase):
    def test_unique_native_16_and_exact_seed_replay(self):
        rows = candidate_matrix("11111")
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r.seed for r in rows}), 16)
        self.assertEqual(len({r.candidate_id for r in rows}), 16)
        self.assertEqual(rows, candidate_matrix("11111"))
        self.assertNotEqual(rows[0].seed, candidate_matrix("11112")[0].seed)
        verify_matrix(rows)
        with self.assertRaises(ProtocolStop):
            verify_matrix(rows[:-1])

    def test_frozen_np_mixed_is_reuse_not_eight_new_runs(self):
        rows = candidate_matrix("11111")
        p8 = mixed_np8(rows)
        self.assertEqual([(r.family, r.replicate) for r in p8],
                         [(f, k) for k in range(4)
                          for f in ("NP_LF", "NP_S2")])
        self.assertEqual(mixed_np4(rows), p8[:4])
        self.assertEqual(len(set(r.candidate_id for r in p8)), 8)
        self.assertTrue(set(p8).issubset(set(rows)))

    def test_seed_domains_and_invalid_inputs_fail_closed(self):
        self.assertNotEqual(seed63("DAPS", "11111", 0),
                            seed63("SITCOM", "11111", 0))
        self.assertNotEqual(seed63("NP_LF", "11111", 0),
                            seed63("NP_S2", "11111", 0))
        self.assertNotEqual(seed63("A1_MEAS_V1", "11111", 0),
                            seed63("NP_LF", "11111", 0))
        for bad in ("1111", "111111", "a1111", "../a1"):
            with self.assertRaises(ProtocolStop):
                candidate_matrix(bad)
        with self.assertRaises(ProtocolStop):
            seed63("UNKNOWN", "11111", 0)
        with self.assertRaises(ProtocolStop):
            seed63("DAPS", "11111", True)

    def test_numpy_32bit_domains_are_stable_and_independent(self):
        ids = [seed32_numpy(f, "11111", j)
               for f in ("DAPS", "SITCOM", "NP_LF", "NP_S2")
               for j in range(4)]
        self.assertEqual(len(set(ids)),16)
        self.assertTrue(all(0 <= seed < (1<<32) for seed in ids))
        self.assertEqual(seed32_numpy("DAPS","11111",0),
                         seed32_numpy("DAPS","11111",0))
        self.assertNotEqual(seed32_numpy("DAPS","11111",0),
                            seed32_numpy("SITCOM","11111",0))
        with self.assertRaises(ProtocolStop):
            seed32_numpy("DAPS","11111",True)

    def test_dev_ids_deterministic_and_exclusions(self):
        # Synthetic only; NOT actual prospective cohort allocation.
        pool = [f"{10000+j:05d}" for j in range(30)]
        excl = set(pool[:6])
        reserved = set(pool[6:10])
        x = select_dev_ids(pool, excl, reserved)
        self.assertEqual(len(x), 8)
        self.assertEqual(x, select_dev_ids(list(reversed(pool)), excl, reserved))
        self.assertFalse(set(x) & (excl | reserved))
        with self.assertRaises(ProtocolStop):
            select_dev_ids(pool, set(pool[:-4]), set())
        with self.assertRaises(ProtocolStop):
            select_dev_ids(pool + [pool[0]], set(), set())
        with self.assertRaises(ProtocolStop):
            select_dev_ids(pool, set(), set(), n=12)


if __name__ == "__main__":
    unittest.main()
