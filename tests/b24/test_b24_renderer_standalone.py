from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class StandaloneRendererTests(unittest.TestCase):
    def _render(self, count: int) -> dict:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "manifest.json"
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/b24/render_b24_baseline_manifest.py"),
                    "--count", str(count),
                    "--out", str(out),
                ],
                cwd=td,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            return json.loads(out.read_text(encoding="utf-8"))

    def test_renderer_works_outside_repo_without_pythonpath(self):
        value = self._render(64)
        self.assertEqual(value["count"], 64)
        self.assertEqual(len(value["rows"]), 64)

    def test_renderer_supports_cumulative_2048_prefix(self):
        value = self._render(2048)
        self.assertEqual(value["count"], 2048)
        self.assertEqual(len(value["rows"]), 2048)
        self.assertEqual([int(r["row_index"]) for r in value["rows"]], list(range(2048)))
        self.assertEqual([sum(int(r["gpu_id"]) == g for r in value["rows"]) for g in range(4)], [512] * 4)

    def test_renderer_supports_cumulative_6144_and_preserves_2048_prefix(self):
        parent = self._render(2048)
        value = self._render(6144)
        self.assertEqual(value["count"], 6144)
        self.assertEqual(len(value["rows"]), 6144)
        self.assertEqual(value["rows"][:2048], parent["rows"])
        self.assertEqual([int(r["row_index"]) for r in value["rows"]], list(range(6144)))
        self.assertEqual([sum(int(r["gpu_id"]) == g for r in value["rows"]) for g in range(4)], [1536] * 4)
        self.assertEqual(
            [sum(int(r["gpu_id"]) == g and int(r["row_index"]) >= 2048 for r in value["rows"]) for g in range(4)],
            [1024] * 4,
        )


if __name__ == "__main__":
    unittest.main()
