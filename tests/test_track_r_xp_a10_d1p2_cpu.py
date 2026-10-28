"""Synthetic zero-GPU A10 D1-P2 comparison tests."""
import math
import unittest
from scripts.track_r.xp_a10_d1p2_compare_cpu import (
    conditional_pl_given_labels,V2_SHA,E1_SHA,PLANNER,K)
from scripts.track_r.xp_a7_postfreeze_kernel_cpu import exact_panel

class TestD1P2FrozenComparison(unittest.TestCase):
    def test_registered_sha_and_panel(self):
        self.assertEqual(K,(8,12,16))
        self.assertEqual(len(V2_SHA),64)
        self.assertEqual(len(E1_SHA),64)
        self.assertEqual(len(PLANNER),40)
    def test_conditional_good_all(self):
        panel=[dict(label="G") for _ in range(16)]
        for k in K:
            self.assertAlmostEqual(conditional_pl_given_labels(panel,k,.5,.2),1.)
    def test_conditional_good_zero(self):
        panel=[dict(label="I") for _ in range(16)]
        for k in K:
            self.assertAlmostEqual(conditional_pl_given_labels(panel,k,.5,.2),0.)
    def test_s2_uniform_positive_race(self):
        # In G/C-only panels, S2 is exchangeable among the positive-rate
        # runs. Conditional expected G first = 6/16 for every K.
        panel=[dict(label="G") for _ in range(6)]+[
            dict(label="C") for _ in range(10)]
        for k in K:
            self.assertAlmostEqual(conditional_pl_given_labels(panel,k,1.,1.),
                                   6/16,places=12)
    def test_race_favors_good(self):
        panel=[dict(label="G") for _ in range(6)]+[
            dict(label="C") for _ in range(10)]
        for k in K:
            self.assertGreater(conditional_pl_given_labels(panel,k,.77,.18),
                               6/16)
    def test_oversampled_orbit_and_ineffective_inconsistent(self):
        panel=([dict(label="I") for _ in range(12)]+
               [dict(label="O") for _ in range(2)]+
               [dict(label="G") for _ in range(2)])
        for k in K:
            val=conditional_pl_given_labels(panel,k,.77,.18)
            self.assertTrue(0<=val<=1)
        self.assertEqual(conditional_pl_given_labels(panel,16,.77,.18),
                         2/(2+2*.77))

if __name__=="__main__":unittest.main()
