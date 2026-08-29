from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# The B24 tests must import the package from this checkout even when unittest
# is launched from the sibling B23 worktree.  ``python -m`` otherwise places
# the caller's cwd first on sys.path, which can shadow B24 with B23's valid
# ``prdiffusion`` package (which naturally has no b24_protocol module).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prdiffusion.b24_protocol import (
    GPU_UUIDS,
    a_audit_keep,
    atomic_write_json,
    canonical_json_sha256,
    classify_good25,
    completion_is_reusable,
    global_allocation,
    render_screen_manifest,
    seed_row,
    shard_rows,
)


class B24ProtocolTests(unittest.TestCase):
    def test_good25_boundary(self):
        self.assertEqual(classify_good25(25.0, 25.0), "A")
        self.assertEqual(classify_good25(24.999, 25.0), "B")
        self.assertEqual(classify_good25(25.0, 24.999), "C")
        self.assertEqual(classify_good25(24.999, 24.999), "D")

    def test_seed_reproducible_and_distinct(self):
        first = seed_row("00042")
        second = seed_row(42)
        self.assertEqual(first, second)
        solver = first["solver_seeds"]["DAPS"] + first["solver_seeds"]["SITCOM"]
        self.assertEqual(len(set(solver)), 8)
        self.assertNotIn(first["measurement_seed"], solver)

    def test_screen_prefix_and_shards(self):
        exposed = {f"{i:05d}" for i in range(333)}
        rows64 = render_screen_manifest(exposed_ids=exposed, count=64)
        rows256 = render_screen_manifest(exposed_ids=exposed, count=256)
        self.assertEqual(rows64, rows256[:64])
        shards = shard_rows(rows64)
        self.assertEqual({k: len(v) for k, v in shards.items()}, {0: 16, 1: 16, 2: 16, 3: 16})
        for shard, rows in shards.items():
            self.assertTrue(all(row["gpu_uuid"] == GPU_UUIDS[shard] for row in rows))
            self.assertTrue(all(global_allocation(row["image_id"]) == "B24_SCREEN_ELIGIBLE" for row in rows))

    def test_a_audit_is_deterministic(self):
        values = [a_audit_keep(i) for i in range(1000, 1100)]
        self.assertEqual(values, [a_audit_keep(i) for i in range(1000, 1100)])
        self.assertGreater(sum(values), 0)
        self.assertLess(sum(values), 30)

    def test_atomic_completion_identity(self):
        identity = {"run_manifest_sha256": "a" * 64, "image_id": "00042", "solver_seeds": [1, 2]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "complete.json"
            value = {"status": "COMPLETE", "identity": identity}
            atomic_write_json(path, value)
            self.assertTrue(completion_is_reusable(path, identity))
            changed = dict(identity, image_id="00043")
            self.assertFalse(completion_is_reusable(path, changed))
            self.assertEqual(canonical_json_sha256(json.loads(path.read_text())["identity"]), canonical_json_sha256(identity))


if __name__ == "__main__":
    unittest.main()
