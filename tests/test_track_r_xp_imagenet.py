"""ImageNet-only DAPS ZIP import safety tests (synthetic bytes, no real data)."""
import io
import unittest
import zipfile
from scripts.track_r.xp_imagenet_stage_cpu import image_members

def archive(members):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED) as z:
        for name,blob in members:
            z.writestr(name,blob)
    stream.seek(0)
    return stream

class ImportGate(unittest.TestCase):
    def test_valid_hundred(self):
        entries=[("test-imagenet/%03d.png"%j,b"sample") for j in range(100)]
        with zipfile.ZipFile(archive(entries)) as z:
            files=image_members(z)
            self.assertEqual(len(files),100)
            self.assertEqual(files[0][1],"000.png")
            self.assertEqual(files[-1][1],"099.png")

    def test_reject_additional_dataset(self):
        entries=[("test-imagenet/%03d.png"%j,b"sample") for j in range(100)]
        entries.append(("test-ffhq/evil.png",b"excluded"))
        with zipfile.ZipFile(archive(entries)) as z:
            with self.assertRaisesRegex(RuntimeError,"STOP_ARCHIVE_FORBIDDEN_MEMBER"):
                image_members(z)

    def test_reject_traversal(self):
        entries=[("test-imagenet/%03d.png"%j,b"sample") for j in range(100)]
        entries.append(("test-imagenet/../bypass.png",b"sample"))
        with zipfile.ZipFile(archive(entries)) as z:
            with self.assertRaisesRegex(RuntimeError,"STOP_ARCHIVE_TRAVERSAL"):
                image_members(z)

    def test_reject_not_hundred(self):
        with zipfile.ZipFile(archive([("test-imagenet/000.png",b"sample")])) as z:
            with self.assertRaisesRegex(RuntimeError,"STOP_ARCHIVE_EXPECT_100"):
                image_members(z)

    def test_reject_unexpected_root(self):
        entries=[("not-the-dataset/%03d.png"%j,b"sample") for j in range(100)]
        with zipfile.ZipFile(archive(entries)) as z:
            with self.assertRaisesRegex(RuntimeError,"STOP_ARCHIVE_UNEXPECTED_DATASET_ROOT"):
                image_members(z)

if __name__=="__main__":
    unittest.main()
