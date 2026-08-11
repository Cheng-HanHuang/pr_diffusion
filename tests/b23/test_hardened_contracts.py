from __future__ import annotations

import copy
import importlib.util
import json
import math
import unittest
from pathlib import Path

from prdiffusion.b23_protocol import (
    ProtocolError,
    RAW_COUNT_FIELDS,
    validate_compute_ledger,
    validate_future_split_registry,
    validate_replay_report,
)
from prdiffusion.b23_schema import SchemaValidationError, validate


REPO = Path(__file__).resolve().parents[2]
SHA_A = "a" * 64
SHA_B = "b" * 64


def replay_pass() -> dict:
    comparison = {
        "max_abs_err": 0.1,
        "mean_abs_err": 0.01,
        "relative_l2_err": 0.02,
        "raw_psnr_delta": -0.01,
        "measurement_loss_delta": 0.001,
        "trace_max_abs_err": 0.03,
        "tensor_hash_equal": False,
    }
    return {
        "schema_version": "b23.replay-report.v2",
        "experiment_id": "test",
        "parent_id": "Fresh1",
        "native_run_ids": ["n1", "n2", "n3"],
        "wrapper_run_id": "w1",
        "eligibility": "TOLERANCE_QUALIFIED",
        "eligibility_rationale": "three native runs established a finite envelope",
        "determinism_audit": {
            "torch_deterministic_algorithms": False,
            "cudnn_deterministic": False,
            "cudnn_benchmark": True,
            "cublas_workspace_config": None,
            "flags_change_native_parent": False,
            "notes": [],
        },
        "native_repeatability_envelope": copy.deepcopy(comparison),
        "wrapper_comparison": copy.deepcopy(comparison),
        "native_tensor_sha256s": [SHA_A, SHA_A, SHA_A],
        "wrapper_tensor_sha256": SHA_B,
        "native_trace_sha256s": [SHA_A, SHA_A, SHA_A],
        "wrapper_trace_sha256": SHA_B,
        "native_operation_counts_sha256": SHA_A,
        "wrapper_operation_counts_sha256": SHA_A,
        "operation_count_reconciled": True,
        "native_rng_ledger_sha256": SHA_B,
        "wrapper_rng_ledger_sha256": SHA_B,
        "rng_draws_reconciled": True,
        "serialization_resume_check": "PASS",
        "serialization_not_applicable_reason": None,
        "verdict": "PASS",
        "failure_reasons": [],
    }


def calibrated_ledger() -> dict:
    counts = {name: 0 for name in RAW_COUNT_FIELDS}
    counts["denoiser_forward"] = 2
    reference_counts = dict(counts)
    return {
        "schema_version": "b23.compute-ledger.v1",
        "experiment_id": "test",
        "image_id": "70000",
        "measurement_id": "meas-new",
        "parent_or_policy_id": "Fresh1",
        "hardware_identity": {"identity_id": "pac-gpu-0", "inventory_sha256": SHA_A},
        "raw_counts": counts,
        "coupled_operation_blocks": [],
        "optimizer_iterations_by_type": {},
        "rng_streams": [],
        "branches": {
            "total_created": 1,
            "max_live": 1,
            "retained": 1,
            "terminal_candidates": 1,
            "branch_ids": ["root"],
            "retained_branch_ids": ["root"],
            "terminal_branch_ids": ["root"],
        },
        "timing": {"gpu_active_seconds": 2.0, "wall_seconds": 2.5, "timer_method": "CUDA_EVENTS"},
        "memory_bytes": {"peak_allocated": 10, "peak_reserved": 20},
        "overhead_seconds": {"serialization": 0.1},
        "atomic_weights": {"denoiser_forward": 1.0},
        "fre": {
            "status": "CALIBRATED",
            "work_FRE": 1.0,
            "time_FRE": 1.0,
            "claim_FRE": 1.0,
            "reference": {
                "identity_id": "pac-gpu-0",
                "inventory_sha256": SHA_A,
                "reference_id": "fresh1-paired",
                "reference_ledger_sha256": SHA_A,
                "weight_registry_sha256": SHA_B,
                "raw_counts": reference_counts,
                "coupled_operation_blocks": [],
                "gpu_active_seconds": 2.0,
            },
        },
        "notes": [],
    }


class ReplayNegativeTests(unittest.TestCase):
    def assertRejected(self, mutate) -> None:
        value = replay_pass()
        mutate(value)
        with self.assertRaises(ProtocolError):
            validate_replay_report(value)

    def test_valid_evidence_derived_pass(self) -> None:
        validate_replay_report(replay_pass())

    def test_two_native_runs_rejected(self) -> None:
        self.assertRejected(lambda x: x.update(native_run_ids=["n1", "n2"]))

    def test_null_wrapper_rejected(self) -> None:
        self.assertRejected(lambda x: x.update(wrapper_run_id=None))

    def test_null_comparison_evidence_rejected(self) -> None:
        self.assertRejected(lambda x: x["wrapper_comparison"].update(max_abs_err=None))

    def test_missing_hash_rejected(self) -> None:
        self.assertRejected(lambda x: x.update(native_rng_ledger_sha256=None))

    def test_missing_tensor_evidence_hash_rejected(self) -> None:
        self.assertRejected(lambda x: x.update(native_tensor_sha256s=[]))

    def test_true_booleans_do_not_override_hash_mismatch(self) -> None:
        self.assertRejected(lambda x: x.update(wrapper_operation_counts_sha256=SHA_B))

    def test_unjustified_serialization_not_applicable_rejected(self) -> None:
        self.assertRejected(lambda x: x.update(serialization_resume_check="NOT_APPLICABLE"))

    def test_bitwise_nonzero_delta_rejected(self) -> None:
        def mutate(value):
            value["eligibility"] = "BITWISE"
            value["wrapper_tensor_sha256"] = SHA_A
            value["wrapper_trace_sha256"] = SHA_A
            for section in ("native_repeatability_envelope", "wrapper_comparison"):
                value[section]["tensor_hash_equal"] = True
                for key in value[section]:
                    if key != "tensor_hash_equal":
                        value[section][key] = 0.0
            value["wrapper_comparison"]["trace_max_abs_err"] = 1e-9
        self.assertRejected(mutate)

    def test_not_run_cannot_be_tolerance_qualified(self) -> None:
        value = replay_pass()
        value["verdict"] = "NOT_RUN"
        with self.assertRaises(ProtocolError):
            validate_replay_report(value)


class ComputeNegativeTests(unittest.TestCase):
    def assertRejected(self, mutate) -> None:
        value = calibrated_ledger()
        mutate(value)
        with self.assertRaises(ProtocolError):
            validate_compute_ledger(value)

    def test_valid_recomputed_fre(self) -> None:
        validate_compute_ledger(calibrated_ledger())

    def test_zero_weight_for_nonzero_operation_rejected(self) -> None:
        self.assertRejected(lambda x: x["atomic_weights"].update(denoiser_forward=0.0))

    def test_optimizer_aggregate_mismatch_rejected(self) -> None:
        self.assertRejected(lambda x: x["optimizer_iterations_by_type"].update(adam=1))

    def test_rng_aggregate_mismatch_rejected(self) -> None:
        self.assertRejected(lambda x: x["raw_counts"].update(rng_draws=1))

    def test_duplicate_named_rng_stream_rejected(self) -> None:
        stream = {"name": "proposal", "derived_seed": 1, "draw_calls": 1, "values_drawn": 1, "device": "cpu"}
        def mutate(value):
            value["raw_counts"]["rng_draws"] = 2
            value["rng_streams"] = [stream, dict(stream)]
        self.assertRejected(mutate)

    def test_branch_candidate_subset_mismatch_rejected(self) -> None:
        self.assertRejected(lambda x: x["branches"].update(terminal_branch_ids=["other"]))

    def test_nonfinite_timing_rejected(self) -> None:
        self.assertRejected(lambda x: x["timing"].update(wall_seconds=math.inf))

    def test_memory_consistency_rejected(self) -> None:
        self.assertRejected(lambda x: x["memory_bytes"].update(peak_allocated=21))

    def test_overhead_exceeding_wall_rejected(self) -> None:
        self.assertRejected(lambda x: x["overhead_seconds"].update(serialization=3.0))

    def test_reference_hardware_mismatch_rejected(self) -> None:
        self.assertRejected(lambda x: x["fre"]["reference"].update(identity_id="other"))

    def test_supplied_fre_is_not_trusted(self) -> None:
        self.assertRejected(lambda x: x["fre"].update(claim_FRE=0.1))

    def test_unweighted_coupled_operation_rejected(self) -> None:
        self.assertRejected(lambda x: x["coupled_operation_blocks"].append({
            "operation_type": "parent_native_block", "count": 1,
            "measured_weight_seconds": 0.0, "definition_sha256": SHA_A,
        }))

    def test_unknown_atomic_weight_rejected(self) -> None:
        self.assertRejected(lambda x: x["atomic_weights"].update(free_parent_block=1.0))

    def test_coupled_reference_registry_mismatch_rejected(self) -> None:
        def mutate(value):
            value["coupled_operation_blocks"].append({
                "operation_type": "parent_native_block", "count": 0,
                "measured_weight_seconds": 1.0, "definition_sha256": SHA_A,
            })
        self.assertRejected(mutate)


class ExposureRegistryTests(unittest.TestCase):
    def test_exposed_image_rejected_even_with_new_measurement_and_seed(self) -> None:
        row = {
            "registry_version": "b23.future-split.v1",
            "split": "B23.1-SMOKE-1", "row_id": 0, "image_id": "00046",
            "measurement_id": "brand-new", "measurement_seed": 999,
            "solver_base_seed": 1000,
            "assigned_before_run": True, "pre_b23_exposure_checked": True,
            "source_manifest_sha256": SHA_A,
        }
        with self.assertRaises(ProtocolError):
            validate_future_split_registry([row], {"00046"})

    def test_unresolved_tag_becomes_image_wide_unknown(self) -> None:
        path = REPO / "scripts/b23/collect_b23_0_pac_evidence.py"
        spec = importlib.util.spec_from_file_location("collector_hardened", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        builder = module.ExposureBuilder()
        builder.add("00046", "meas7:DERIVED_SEED_UNRESOLVED", stage="B20", role="test", artifact="a", evidence="e")
        self.assertIn(("00046", "UNKNOWN_ALL_MEASUREMENTS"), builder.rows)
        self.assertNotIn(("00046", "meas7:DERIVED_SEED_UNRESOLVED"), builder.rows)
        self.assertEqual(builder.unresolved_measurement_tag_mentions, 1)


class SchemaEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compute_schema = json.loads((REPO / "schemas/b23/compute_ledger.schema.json").read_text())
        cls.replay_schema = json.loads((REPO / "schemas/b23/replay_report.schema.json").read_text())

    def test_complete_compute_instance_passes_custom_validator(self) -> None:
        validate(calibrated_ledger(), self.compute_schema)

    def test_missing_required_field_rejected_without_jsonschema(self) -> None:
        value = calibrated_ledger()
        del value["timing"]
        with self.assertRaises(SchemaValidationError):
            validate(value, self.compute_schema)

    def test_additional_field_rejected_without_jsonschema(self) -> None:
        value = calibrated_ledger()
        value["free_compute"] = 1
        with self.assertRaises(SchemaValidationError):
            validate(value, self.compute_schema)

    def test_wrong_nested_type_rejected_without_jsonschema(self) -> None:
        value = calibrated_ledger()
        value["raw_counts"]["fft"] = 1.5
        with self.assertRaises(SchemaValidationError):
            validate(value, self.compute_schema)

    def test_hash_pattern_rejected_without_jsonschema(self) -> None:
        value = calibrated_ledger()
        value["hardware_identity"]["inventory_sha256"] = "not-a-hash"
        with self.assertRaises(SchemaValidationError):
            validate(value, self.compute_schema)

    def test_unique_items_rejected_without_jsonschema(self) -> None:
        value = replay_pass()
        value["native_run_ids"] = ["n1", "n1", "n2"]
        with self.assertRaises(SchemaValidationError):
            validate(value, self.replay_schema)

    def test_nonfinite_number_rejected_without_jsonschema(self) -> None:
        value = replay_pass()
        value["wrapper_comparison"]["max_abs_err"] = float("nan")
        with self.assertRaises(SchemaValidationError):
            validate(value, self.replay_schema)

    def test_const_rejected_without_jsonschema(self) -> None:
        value = replay_pass()
        value["schema_version"] = "wrong"
        with self.assertRaises(SchemaValidationError):
            validate(value, self.replay_schema)

    def test_enum_rejected_without_jsonschema(self) -> None:
        value = replay_pass()
        value["eligibility"] = "MADE_UP"
        with self.assertRaises(SchemaValidationError):
            validate(value, self.replay_schema)

    def test_minimum_rejected_without_jsonschema(self) -> None:
        value = calibrated_ledger()
        value["raw_counts"]["fft"] = -1
        with self.assertRaises(SchemaValidationError):
            validate(value, self.compute_schema)

    def test_min_length_rejected_without_jsonschema(self) -> None:
        value = replay_pass()
        value["experiment_id"] = ""
        with self.assertRaises(SchemaValidationError):
            validate(value, self.replay_schema)

    def test_array_item_schema_rejected_without_jsonschema(self) -> None:
        value = replay_pass()
        value["native_run_ids"] = ["n1", "n2", 3]
        with self.assertRaises(SchemaValidationError):
            validate(value, self.replay_schema)

    def test_unsupported_assertion_keyword_fails_closed(self) -> None:
        schema = copy.deepcopy(self.replay_schema)
        schema["properties"]["experiment_id"]["containsMagic"] = True
        with self.assertRaises(SchemaValidationError):
            validate(replay_pass(), schema)


if __name__ == "__main__":
    unittest.main()
