from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


class B24LockedInputLookupTests(unittest.TestCase):
    def test_sub1000_image_path_is_not_double_counted(self):
        repo = Path(__file__).resolve().parents[2]
        script = repo / "scripts/b24/generate_b24_locked_input.py"
        spec = importlib.util.spec_from_file_location("b24_locked_input_lookup_test", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "00000/00894.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"not-an-image-needed-for-path-only-test")
            old = module.FFHQ
            module.FFHQ = root
            try:
                self.assertEqual(module.find_image("00894"), source.resolve())
            finally:
                module.FFHQ = old


if __name__ == "__main__":
    unittest.main()
