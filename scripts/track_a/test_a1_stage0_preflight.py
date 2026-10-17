"""Pure synthetic bounded A1.0 audit helper tests; never reads PAC project files."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.track_a.a1_stage0_preflight import file_sha


class PreflightHelpersTest(unittest.TestCase):
    def test_sha_is_complete_and_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder)/"test.json"
            payload=b'{"sample":"synthetic"}'
            src.write_bytes(payload)
            dig,n=file_sha(src,maxbytes=1024)
            self.assertEqual(dig,hashlib.sha256(payload).hexdigest())
            self.assertEqual(n,len(payload))
            with self.assertRaises(ValueError):
                file_sha(src,maxbytes=len(payload)-1)

    def test_refuse_missing_empty_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError):
                file_sha(root/"missing",maxbytes=1024)
            target=root/"empty"
            target.write_bytes(b"")
            with self.assertRaises(ValueError):
                file_sha(target,maxbytes=1024)
            target.write_bytes(b"safe")
            link=root/"link"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                file_sha(link,maxbytes=1024)


if __name__=="__main__":
    unittest.main()
