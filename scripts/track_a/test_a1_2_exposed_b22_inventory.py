"""A1.2 exposed B22 metadata-only inventory selector tests; no PAC files read."""
import unittest

from scripts.track_a.a1_2_exposed_b22_inventory import (
    N, RANK_DOMAIN, rank, select_dev8,
)

class ExposedB22InventoryTest(unittest.TestCase):
    def test_stable_pregenerated_rank_and_length(self):
        rows=[{"image_id":f"{60000+i:05d}","row_id":i} for i in range(100)]
        a=select_dev8(rows)
        b=select_dev8(rows[::-1])
        self.assertEqual(len(a),N)
        self.assertEqual([r["image_id"] for r in a],
                         [r["image_id"] for r in b])
        self.assertEqual([r["image_id"] for r in a],
                         [r["image_id"] for r in sorted(rows,key=rank)[:N]])
        self.assertTrue(RANK_DOMAIN.startswith("A1_2_EXPOSED"))
        self.assertEqual(len({r["image_id"] for r in a}),N)

    def test_invalid_inventory_is_rejected(self):
        rows=[{"image_id":f"{60000+i:05d}","row_id":i} for i in range(100)]
        for invalid in (rows[:99],rows+[rows[0]],
                        rows[:99]+[rows[0]],
                        rows[:99]+[{"image_id":"05000","row_id":99}],
                        rows[:99]+[{"image_id":"70000","row_id":99}]):
            with self.assertRaises(ValueError):
                select_dev8(invalid)

if __name__=="__main__":
    unittest.main()
