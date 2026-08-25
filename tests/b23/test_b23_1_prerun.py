from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from prdiffusion.b23_protocol import validate_future_split_registry


REPO = Path(__file__).resolve().parents[2]


class B231PrerunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((REPO / "configs/b23/b23_1a_b_execution.yaml").read_text())
        with (REPO / "manifests/b23/b23_1_signed_registry.csv").open(newline="") as handle:
            cls.rows = list(csv.DictReader(handle))
        with (REPO / "manifests/b23/PRE_B23_EXPOSURE.csv").open(newline="") as handle:
            cls.exposed = {row["image_id"] for row in csv.DictReader(handle)}

    def test_scope_boundary_is_exact(self) -> None:
        auth = self.config["authorization"]
        self.assertEqual(auth["authorized_stages"], ["B23.1A", "B23.1B"])
        self.assertTrue(auth["gpu_work_authorized"])
        self.assertEqual(
            set(auth["unauthorized_work"]),
            {"B23.2", "large panels", "B24", "adaptive schedules"},
        )
        self.assertEqual(self.config["stop_after"], "B23.1_RETURN_PENDING_PLANNER_REVIEW")

    def test_signed_registry_is_image_disjoint(self) -> None:
        typed = []
        for row in self.rows:
            value = dict(row)
            for field in ("row_id", "measurement_seed", "solver_base_seed"):
                value[field] = int(value[field])
            for field in ("assigned_before_run", "pre_b23_exposure_checked"):
                value[field] = value[field] == "true"
            typed.append(value)
        validate_future_split_registry(typed, self.exposed)
        self.assertEqual(len(self.rows), 5)
        self.assertFalse({row["image_id"] for row in self.rows} & self.exposed)

    def test_accepted_closeout_exposure_identity_is_frozen(self) -> None:
        exposure_path = REPO / "manifests/b23/PRE_B23_EXPOSURE.csv"
        observed = hashlib.sha256(exposure_path.read_bytes()).hexdigest()
        self.assertEqual(
            observed,
            "a513cb4e3b79b39700ff1d623cb4b2eaf496bc2d6d0fe58bd963709e6a56d288",
        )
        self.assertEqual(self.config["registry"]["pre_b23_exposure_sha256"], observed)
        self.assertEqual({row["source_manifest_sha256"] for row in self.rows}, {observed})

    def test_preflight_false_start_is_preserved_as_zero_gpu(self) -> None:
        ledger = json.loads(
            (REPO / "manifests/b23/b23_1_correction_ledger.json").read_text()
        )
        entry = next(
            value for value in ledger["entries"]
            if value["id"] == "preflight_exposure_digest_stop_20260825T012447Z"
        )
        self.assertEqual(entry["invalidated_pre_run_head"], "6e3e08eba2621f903bd33cd8d818442a34158318")
        self.assertFalse(entry["gpu_work_performed"])
        self.assertFalse(entry["input_generation_performed"])
        self.assertEqual(entry["parent_trajectory_count"], 0)
        self.assertFalse(entry["selection_revalidation"]["selection_changed"])

    def test_seed_range_stop_and_uint32_adapter_are_frozen(self) -> None:
        ledger = json.loads(
            (REPO / "manifests/b23/b23_1_correction_ledger.json").read_text()
        )
        entry = next(
            value for value in ledger["entries"]
            if value["id"] == "fresh1_native_seed_range_stop_20260825T015257Z"
        )
        self.assertTrue(entry["gpu_work_performed"])
        self.assertEqual(entry["completed_measurement_generations"], 5)
        self.assertEqual(entry["completed_parent_trajectories"], 0)
        self.assertEqual(entry["canonical_parent_seed"], 92870567330106893)
        self.assertEqual(entry["corrected_native_entrypoint_seed"], 4157394445)
        self.assertEqual(
            entry["corrected_native_entrypoint_seed"],
            entry["canonical_parent_seed"] % (2 ** 32),
        )

    def test_selection_and_seed_derivation_reproduce(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts/b23/validate_b23_1_prerun.py"), "--repo", str(REPO)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["image_level_disjoint"])
        self.assertEqual(payload["replay_image"], "65082")
        self.assertEqual(payload["smoke_images"], ["61492", "62959", "66821", "68142"])
        replay_fresh = next(
            row for row in payload["native_seed_records"]
            if row["image_id"] == "65082" and row["parent_id"] == "Fresh1"
        )
        self.assertEqual(replay_fresh["canonical_parent_seed"], 92870567330106893)
        self.assertEqual(replay_fresh["native_entrypoint_seed"], 4157394445)

    def test_trajectory_budget_is_bounded(self) -> None:
        execution = self.config["execution"]
        self.assertEqual(execution["expected_replay_parent_trajectories"], 16)
        self.assertEqual(execution["expected_smoke_parent_trajectories"], 16)
        self.assertEqual(execution["max_parent_trajectories"], 32)
        self.assertEqual(execution["terminal_candidates_per_parent_run"], 1)
        self.assertEqual(execution["completed_measurement_generations_before_corrective_run"], 5)
        self.assertEqual(execution["expected_new_measurement_generations"], 0)

    def test_no_adaptive_schedule_in_launcher(self) -> None:
        text = (REPO / "scripts/b23/run_b23_1a_b.sh").read_text()
        self.assertNotIn("B23.2_PREREGISTRATION", text)
        self.assertNotIn("schedule_candidates", text)
        self.assertIn("b23_2_authorized=NO", text)
        self.assertIn("--reuse-inputs", text)
        self.assertIn("--validate-existing", text)
        self.assertNotIn('--output-root "$RUN_ROOT/inputs"', text)


if __name__ == "__main__":
    unittest.main()
