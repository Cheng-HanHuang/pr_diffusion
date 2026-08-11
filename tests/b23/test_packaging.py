from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = REPO / "scripts/b23" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingSafetyTests(unittest.TestCase):
    def test_archive_validator_rejects_oversize(self) -> None:
        publish = load_script("publish_b23_0_evidence.py")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fake.tar.gz"
            path.write_bytes(b"too large")
            with self.assertRaises(ValueError):
                publish.validate_archive(path, 1)

    def test_schemas_are_draft_2020_12(self) -> None:
        for path in (REPO / "schemas/b23").glob("*.schema.json"):
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_untracked_source_classifier_marks_importable_files(self) -> None:
        collector = load_script("collect_b23_0_pac_evidence.py")
        self.assertEqual(collector.classify_untracked("pkg/module.py"), "IMPORTABLE_SOURCE")
        self.assertEqual(collector.classify_untracked("pkg/native.so"), "IMPORTABLE_SOURCE")

    def test_untracked_source_classifier_separates_non_source_artifacts(self) -> None:
        collector = load_script("collect_b23_0_pac_evidence.py")
        self.assertEqual(collector.classify_untracked("__pycache__/module.pyc"), "CACHE")
        self.assertEqual(collector.classify_untracked("data/image.png"), "DATASET")
        self.assertEqual(collector.classify_untracked("outputs/run/log.txt"), "OUTPUT")
        self.assertEqual(collector.classify_untracked("README.local"), "OTHER_ARTIFACT")

    def test_exact_manifest_can_resolve_seeded_unknown_measurement(self) -> None:
        collector = load_script("collect_b23_0_pac_evidence.py")
        builder = collector.ExposureBuilder()
        builder.add(
            "60044", None, stage="B22", role="manual", artifact="taxonomy",
            evidence="taxonomy",
        )
        builder.add(
            "60044", "meas5401:seed123", stage="B21", role="manifest",
            artifact="measurement_manifest", evidence="measurement_manifest",
            exact_replaces_unknown=True,
        )
        self.assertIn(("60044", "meas5401:seed123"), builder.rows)
        self.assertNotIn(("60044", "UNKNOWN_ALL_MEASUREMENTS"), builder.rows)

    def test_publisher_rejects_failed_zero_gpu_step(self) -> None:
        publish = load_script("publish_b23_0_evidence.py")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "steps.tsv"
            path.write_text(
                "step\tstatus\treturn_code\n"
                "unit_tests\tFAIL\t1\n"
                "repository_validation\tPASS\t0\n"
                "b23_1_dry_render\tPASS\t0\n"
                "pac_evidence_collection\tPASS\t0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                publish.validate_step_results(path)

    def test_publisher_accepts_exact_zero_gpu_step_ledger(self) -> None:
        publish = load_script("publish_b23_0_evidence.py")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "steps.tsv"
            path.write_text(
                "step\tstatus\treturn_code\n"
                "unit_tests\tPASS\t0\n"
                "repository_validation\tPASS\t0\n"
                "b23_1_dry_render\tPASS\t0\n"
                "pac_evidence_collection\tPASS\t0\n",
                encoding="utf-8",
            )
            publish.validate_step_results(path)


if __name__ == "__main__":
    unittest.main()
