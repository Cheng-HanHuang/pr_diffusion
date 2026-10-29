"""A10 8-root DAPS legacy gate: pure-Python contract tests (no PAC I/O)."""
from __future__ import annotations
import datetime
import unittest
from zoneinfo import ZoneInfo

from scripts.track_r.xp_f8_legacy_preflight_cpu import (
    A10_CAP_H,A10_DEADLINE,DAPS_HEAD,MODEL_SHA,
    PATCHED_SAMPLER_BLOB,COHORTS,CAT,BASE)
from prdiffusion.b24_protocol import seed_row

class TestF8LegacyPreflight(unittest.TestCase):
    def test_legacy_400x5_not_published_daps4k(self):
        self.assertEqual(DAPS_HEAD,
            "e7a77d094167084faed19b599b96673b7bb11447")
        self.assertEqual(len(PATCHED_SAMPLER_BLOB),40)
        self.assertEqual(MODEL_SHA,
            "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa")
        self.assertEqual(COHORTS,("S1_b22","S2_b24_dev80"))
    def test_r07_catalog_from_prepare_and_not_analysis(self):
        self.assertEqual(CAT,BASE/
            "r0_7_cpu_20261023T080434Z/prepare/R0_7_OFFLINE_GT_CATALOG.json")

    def test_legacy_seed_domain(self):
        ids=("00113","00114")
        for iid in ids:
            seeds=[int(x)%(2**32) for x in
                   seed_row(iid)["solver_seeds"]["DAPS"]]
            self.assertEqual(len(set(seeds)),4)
    def test_a10_window_and_shared_cost(self):
        self.assertEqual(A10_CAP_H,70.)
        self.assertEqual(A10_DEADLINE.astimezone(datetime.timezone.utc),
            datetime.datetime(2026,10,30,16,0,
                              tzinfo=datetime.timezone.utc))
if __name__=="__main__":
    unittest.main()
