"""ImageNet-only DAPS ZIP import safety tests (synthetic bytes, no real data)."""
import io
import unittest
import zipfile
from scripts.track_r.xp_imagenet_stage_cpu import image_members,image_relative_path

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

    def test_thousand_selects_preregistered_first_100(self):
        # Scrambled ZIP insertion order must not change DAPS loader order.
        entries=[("test-imagenet/%04d.png"%j,b"sample") for j in reversed(range(1000))]
        with zipfile.ZipFile(archive(entries)) as z:
            selected=image_members(z)
            self.assertEqual(len(selected),100)
            self.assertEqual(selected[0][1],"0000.png")
            self.assertEqual(selected[99][1],"0099.png")
            self.assertEqual(str(image_relative_path(selected[0][0])),"0000.png")
            self.assertEqual(str(image_relative_path(selected[99][0])),"0099.png")

    def test_nested_imagenet_path_is_preserved(self):
        entries=[("test-imagenet/a/%03d.png"%j,b"sample") for j in range(100)]
        with zipfile.ZipFile(archive(entries)) as z:
            selected=image_members(z)
            self.assertEqual(str(image_relative_path(selected[0][0])),"a/000.png")

    def test_reject_200_image_bundle(self):
        entries=[("test-imagenet/%04d.png"%j,b"sample") for j in range(200)]
        with zipfile.ZipFile(archive(entries)) as z:
            with self.assertRaisesRegex(RuntimeError,"STOP_ARCHIVE_EXPECT_100_OR_1000"):
                image_members(z)

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
