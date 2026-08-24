from __future__ import annotations

import csv
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

    def test_trajectory_budget_is_bounded(self) -> None:
        execution = self.config["execution"]
        self.assertEqual(execution["expected_replay_parent_trajectories"], 16)
        self.assertEqual(execution["expected_smoke_parent_trajectories"], 16)
        self.assertEqual(execution["max_parent_trajectories"], 32)
        self.assertEqual(execution["terminal_candidates_per_parent_run"], 1)

    def test_no_adaptive_schedule_in_launcher(self) -> None:
        text = (REPO / "scripts/b23/run_b23_1a_b.sh").read_text()
        self.assertNotIn("B23.2_PREREGISTRATION", text)
        self.assertNotIn("schedule_candidates", text)
        self.assertIn("b23_2_authorized=NO", text)


if __name__ == "__main__":
    unittest.main()
