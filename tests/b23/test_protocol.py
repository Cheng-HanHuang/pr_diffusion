from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prdiffusion.b23_protocol import (
    NativeState,
    ProtocolError,
    RNGStreamState,
    SemanticCoordinate,
    calibrated_work,
    derive_seed,
    fre_values,
    load_native_state,
    serialize_native_state,
)


class SeedTests(unittest.TestCase):
    def test_frozen_seed_vector(self) -> None:
        seed = derive_seed(
            1234,
            stream_name="native_start_noise",
            image_id="60044",
            measurement_id="meas5401:seed99",
            parent_id="Fresh1",
        )
        self.assertEqual(seed, 5810464214058005293)

    def test_named_streams_are_isolated(self) -> None:
        common = dict(
            base_seed=7,
            image_id="00046",
            measurement_id="meas5001",
            parent_id="LF-v1",
        )
        first = derive_seed(stream_name="native_start_noise", **common)
        second = derive_seed(stream_name="diffusion_transition", **common)
        self.assertNotEqual(first, second)
        self.assertEqual(first, derive_seed(stream_name="native_start_noise", **common))

    def test_invalid_seed_identity_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            derive_seed(
                -1,
                stream_name="native_start_noise",
                image_id="00046",
                measurement_id="meas5001",
                parent_id="Fresh1",
            )


class LedgerTests(unittest.TestCase):
    def test_calibrated_work_and_fre(self) -> None:
        policy = {"denoiser_forward": 6, "measurement_forward": 2}
        reference = {"denoiser_forward": 4, "measurement_forward": 4}
        weights = {"denoiser_forward": 2.0, "measurement_forward": 1.0}
        self.assertEqual(calibrated_work(policy, weights), 14.0)
        result = fre_values(
            policy_counts=policy,
            reference_counts=reference,
            measured_weights=weights,
            gpu_active_seconds=12.0,
            paired_reference_median_seconds=10.0,
        )
        self.assertAlmostEqual(result["work_FRE"], 14.0 / 12.0)
        self.assertAlmostEqual(result["time_FRE"], 1.2)
        self.assertAlmostEqual(result["claim_FRE"], 1.2)

    def test_nonzero_unmeasured_operation_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            calibrated_work({"denoiser_forward": 1}, {})

    def test_unknown_counter_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            calibrated_work({"free_magic_call": 1}, {"free_magic_call": 0.0})


class StateSerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch is unavailable in this local validation environment")
        cls.torch = torch

    def test_synthetic_tensor_round_trip(self) -> None:
        torch = self.torch
        self.assertFalse(torch.cuda.is_initialized())
        state = NativeState(
            parent_id="Fresh1",
            source_revision="e7a77d094167084faed19b599b96673b7bb11447",
            representation_contract={"x0y": "DAPS post-measurement clean estimate"},
            tensor_payloads={"x0y": torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)},
            coordinate=SemanticCoordinate(
                native_name="annealing_sigma",
                native_value=1.0,
                normalized_noise=0.5,
                cumulative_budget=0.25,
            ),
            model_identity={"sha256": "a" * 64},
            scheduler_identity={"id": "ann400-diff5"},
            measurement_identity={"id": "synthetic"},
            optimizer_state={},
            rng_streams=[
                RNGStreamState(
                    name="native_start_noise",
                    base_seed=1,
                    derived_seed=2,
                    draw_count=12,
                    device="cpu",
                )
            ],
            trace_id="synthetic-trace",
            compute_ledger_id="synthetic-ledger",
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "state"
            before = serialize_native_state(state, directory)
            restored = load_native_state(directory)
        self.assertEqual(before["tensor_fingerprints"], {
            "x0y": {
                "shape": [1, 3, 2, 2],
                "dtype": "torch.float32",
                "sha256": before["tensor_fingerprints"]["x0y"]["sha256"],
            }
        })
        self.assertTrue(torch.equal(restored.tensor_payloads["x0y"], state.tensor_payloads["x0y"]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_payload_contract_is_not_generic_x(self) -> None:
        torch = self.torch
        state = NativeState(
            parent_id="NP-1",
            source_revision="a" * 40,
            representation_contract={"proposal_set": "five NP proposal noises"},
            tensor_payloads={"selected_proposal": torch.zeros(1)},
            coordinate=SemanticCoordinate("timestep", 10.0, 0.5, 0.5),
            model_identity={"id": "m"},
            scheduler_identity={"id": "s"},
            measurement_identity={"id": "y"},
            optimizer_state={},
            rng_streams=[],
            trace_id="t",
            compute_ledger_id="l",
        )
        with self.assertRaises(ProtocolError):
            state.validate()


if __name__ == "__main__":
    unittest.main()
