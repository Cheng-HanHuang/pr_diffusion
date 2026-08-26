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
    def test_renderer_works_outside_repo_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "manifest.json"
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/b24/render_b24_baseline_manifest.py"),
                    "--count", "64",
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
            value = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(value["count"], 64)
            self.assertEqual(len(value["rows"]), 64)


if __name__ == "__main__":
    unittest.main()
