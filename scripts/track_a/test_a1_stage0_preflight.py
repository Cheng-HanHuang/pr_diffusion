"""Pure synthetic bounded A1.0 audit helper tests; never reads PAC project files."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.track_a.a1_stage0_preflight import file_sha, checkpoint_sha


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


    def test_checkpoint_symlink_within_workspace_is_hashed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            models=root/"models"
            models.mkdir()
            target=models/"ffhq.pt"
            target.write_bytes(b"synthetic-checkpoint")
            ckpt_dir=root/"external"/"sitcom"/"checkpoint"
            ckpt_dir.mkdir(parents=True)
            link=ckpt_dir/"ffhq256.pt"
            link.symlink_to(target)
            digest,n,meta=checkpoint_sha(link,allowed_root=root,maxbytes=1024)
            self.assertEqual(digest,hashlib.sha256(target.read_bytes()).hexdigest())
            self.assertEqual(n,len(target.read_bytes()))
            self.assertTrue(meta["is_symlink"])
            self.assertEqual(meta["resolved_path"],str(target))
            self.assertEqual(meta["link_text"],str(target))
            d2,n2,meta2=checkpoint_sha(target,allowed_root=root,maxbytes=1024)
            self.assertEqual((d2,n2),(digest,n))
            self.assertFalse(meta2["is_symlink"])

    def test_checkpoint_broken_link_and_escaping_target_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            a=root/"dangling"
            a.symlink_to(root/"missing.pt")
            with self.assertRaises(ValueError):
                checkpoint_sha(a,allowed_root=root,maxbytes=1024)
            with tempfile.TemporaryDirectory() as other:
                far=Path(other)/"outside.pt"
                far.write_bytes(b"safe-bounds")
                link=root/"escape"
                link.symlink_to(far)
                with self.assertRaisesRegex(ValueError,"escapes approved"):
                    checkpoint_sha(link,allowed_root=root,maxbytes=1024)


if __name__=="__main__":
    unittest.main()
