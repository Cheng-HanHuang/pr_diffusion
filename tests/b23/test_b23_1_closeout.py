from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[2]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B231EvidenceCloseoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collector = load("b23_closeout_collect", "scripts/b23/collect_b23_1_closeout.py")
        cls.packager = load("b23_closeout_package", "scripts/b23/package_b23_1_closeout.py")

    def test_contract_freezes_reviewed_heads_and_failed_h0(self) -> None:
        contract = json.loads((REPO / "manifests/b23/b23_1_evidence_closeout_contract.json").read_text())
        self.assertEqual(contract["reviewed_scientific_head"], self.collector.SCIENTIFIC_HEAD)
        self.assertEqual(contract["reviewed_packaging_head"], self.collector.PACKAGING_HEAD)
        self.assertEqual(contract["gate_outcome"]["cross_family_h0"], "FAIL")
        self.assertEqual(contract["gate_outcome"]["qualified_np_sitcom_adapters"], 0)
        self.assertFalse(contract["gate_outcome"]["b23_2_authorized"])

    def test_closeout_launcher_cannot_run_science(self) -> None:
        source = (REPO / "scripts/b23/run_b23_1_evidence_closeout.sh").read_text()
        for forbidden in ("run_b23_1_parent.py", "prepare_b23_1_inputs.py", "run_b23_1a_b.sh", "measurement.pt"):
            self.assertNotIn(forbidden, source)
        self.assertIn('CUDA_VISIBLE_DEVICES=""', source)
        self.assertIn("GPU_correction=NO", source)

    def test_final_status_rejects_any_nonpass_or_missing_row(self) -> None:
        good = "step\tstatus\trc\n" + "".join(f"{step}\tPASS\t0\n" for step in self.collector.EXPECTED_FINAL_STEPS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FINAL_STATUS.tsv"
            path.write_text(good, encoding="utf-8")
            self.assertEqual(self.collector.validate_final_status(path), 29)
            path.write_text(good.replace("donor_classification\tPASS\t0", "donor_classification\tFAIL\t1"), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.collector.validate_final_status(path)
            path.write_text("\n".join(good.splitlines()[:-1]) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.collector.validate_final_status(path)

    def test_donor_summary_requires_zero_qualified_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "run").mkdir()
            donor = {"status": "PASS", "cross_family_adapter_qualified_donor_count": 0, "b23_2_authorized": False, "adaptive_schedules_authorized": False}
            (source / "run/DONOR_COMPATIBILITY.json").write_text(json.dumps(donor))
            result = self.collector.validate_donor(source)
            self.assertEqual(result["cross_family_h0"], "FAIL")
            donor["cross_family_adapter_qualified_donor_count"] = 1
            (source / "run/DONOR_COMPATIBILITY.json").write_text(json.dumps(donor))
            with self.assertRaises(ValueError):
                self.collector.validate_donor(source)

    def test_packager_is_compact_deterministic_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capsule = Path(directory) / "B23_1_closeout_return_TEST"
            for relative in self.packager.REQUIRED:
                path = capsule / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative + "\n", encoding="utf-8")
            archive, first = self.packager.package(capsule)
            self.assertLess(archive.stat().st_size, 5 * 1024 * 1024)
            archive.unlink()
            Path(str(archive) + ".sha256").unlink()
            second_archive, second = self.packager.package(capsule)
            self.assertEqual(first, second)
            second_archive.unlink()
            Path(str(second_archive) + ".sha256").unlink()
            os.symlink(capsule / "README.md", capsule / "bad-link")
            with self.assertRaises(ValueError):
                self.packager.package(capsule)

    def test_collector_refuses_visible_cuda(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with mock.patch("sys.argv", ["collect", "--repo", ".", "--source-capsule", ".", "--source-archive", "x", "--capsule", "y", "--pre-run-head", "z"]):
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    self.collector.main()


if __name__ == "__main__":
    unittest.main()
