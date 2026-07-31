from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from prdiffusion.b23_protocol import validate_compute_ledger, validate_replay_report


REPO = Path(__file__).resolve().parents[2]


class RepositoryContractTests(unittest.TestCase):
    def test_json_compatible_yaml_configs(self) -> None:
        parents = set()
        for path in sorted((REPO / "configs/b23").glob("*_frozen.yaml")):
            value = json.loads(path.read_text(encoding="utf-8"))
            parents.add(value["parent_id"])
            self.assertEqual(value["status"], "FROZEN_NATIVE_PARENT_B23_0")
        self.assertEqual(parents, {"Fresh1", "LF-v1", "NP-1", "SITCOM-1"})

    def test_uncalibrated_ledger_has_no_invented_weights(self) -> None:
        example = json.loads(
            (REPO / "manifests/b23/examples/compute_ledger.uncalibrated.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("atomic_weights", example)
        validate_compute_ledger(example)

    def test_replay_not_run_example_is_valid(self) -> None:
        example = json.loads(
            (REPO / "manifests/b23/examples/replay_report.not_run.json").read_text(
                encoding="utf-8"
            )
        )
        validate_replay_report(example)
        self.assertEqual(example["verdict"], "NOT_RUN")

    def test_exposure_manifest_unique(self) -> None:
        with (REPO / "manifests/b23/PRE_B23_EXPOSURE.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        keys = [(row["image_id"], row["measurement_id"]) for row in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreaterEqual(len({row["image_id"] for row in rows}), 25)

    def test_future_registries_are_empty(self) -> None:
        for name in (
            "future_split_registry.csv",
            "b23_1_one_image_smoke.template.csv",
            "b23_1_four_image_smoke.template.csv",
        ):
            with (REPO / "manifests/b23" / name).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_no_schedule_candidates_or_gpu_authorization(self) -> None:
        smoke = json.loads(
            (REPO / "configs/b23/b23_1_smoke_template.yaml").read_text(encoding="utf-8")
        )
        replay = json.loads(
            (REPO / "configs/b23/replay_policy.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(smoke["gpu_commands"], [])
        self.assertFalse(replay["authorization"]["b23_1_gpu_authorized"])
        self.assertFalse(replay["authorization"]["b23_2_schedule_authorized"])


if __name__ == "__main__":
    unittest.main()
