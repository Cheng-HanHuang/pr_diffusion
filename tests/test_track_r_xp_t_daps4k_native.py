"""CPU-only DAPS pristine sparse checkout regression tests.

Use a disposable synthetic Git source, never touch PAC's DAPS checkout.
"""
from __future__ import annotations
import tempfile,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_t_daps4k_native_cpu as native

def git(repo,*args):
    return subprocess.check_output(
        ["git","-C",str(repo),*args],text=True).strip()

def origin(root):
    source=root/"origin"
    source.mkdir()
    subprocess.run(["git","init","-q",str(source)],check=True)
    subprocess.run(["git","-C",str(source),"config","user.email",
                    "synthetic@example.invalid"],check=True)
    subprocess.run(["git","-C",str(source),"config","user.name",
                    "Synthetic Tester"],check=True)
    for file in list(native.SOURCES)+[
        "dataset/example_image.png","data/example_image.png",
        "configs/data/example_dataset.yaml","posterior_sample.py"]:
        path=source/file
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("safe synthetic content "+file+"\n")
    subprocess.run(["git","-C",str(source),"add","-A"],check=True)
    subprocess.run(["git","-C",str(source),"commit","-qm","synthetic native"],
                   check=True)
    commit=git(source,"rev-parse","HEAD")
    (source/"sampler.py").write_text("experimental modified sampler\n")
    return source,commit

class PristineDAPSChecks(unittest.TestCase):
    def test_pristine_sparse_and_datasets_not_materialized(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            src,commit=origin(root)
            dst=root/"pristine"
            with patch.object(native,"ROOT",root):
                native.clone_sparse_pristine(src,dst,commit)
                with patch.object(native,"PIN",commit):
                    proof=native.verify_clean_sparse(dst)
                self.assertEqual(len(proof),len(native.SOURCES))
                self.assertEqual(proof["sampler.py"]["blob_sha"],
                                 git(src,"rev-parse","HEAD:sampler.py"))
                self.assertNotEqual(
                    git(src,"hash-object","sampler.py"),
                    proof["sampler.py"]["blob_sha"])
                self.assertEqual((src/"sampler.py").read_text(),
                                 "experimental modified sampler\n")
                for excluded in (
                        "dataset","data","configs/data","posterior_sample.py"):
                    self.assertFalse((dst/excluded).exists())
    def test_existing_destination_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            src,commit=origin(root)
            dst=root/"pristine"
            dst.mkdir()
            sentinel=dst/"keep.txt"
            sentinel.write_text("keep")
            with patch.object(native,"ROOT",root):
                with self.assertRaisesRegex(RuntimeError,
                    "DEST_ALREADY_EXISTS"):
                    native.clone_sparse_pristine(src,dst,commit)
            self.assertEqual(sentinel.read_text(),"keep")
    def test_modified_pristine_approved_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            src,commit=origin(root)
            dst=root/"pristine"
            with patch.object(native,"ROOT",root):
                native.clone_sparse_pristine(src,dst,commit)
                (dst/"sampler.py").write_text("drift\n")
                with patch.object(native,"PIN",commit):
                    with self.assertRaisesRegex(RuntimeError,
                        "WORKTREE_DIRTY"):
                        native.verify_clean_sparse(dst)

if __name__=="__main__":
    unittest.main()
