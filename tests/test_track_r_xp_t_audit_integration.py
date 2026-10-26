"""End-to-end synthetic filesystem exercise of XP-T timing re-audit main().

Exercises the real CPU entrypoint including frozen original SHA validation,
provenance dict merge, output receipt writing, and both timing classifications.
All files and Git status are synthetic; no CUDA or dataset reads.
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.track_r import xp_t_timing_reaudit_cpu as audit


def event(action, idx, pid=None, used=0, util=0, worker=42):
    return {
        "stage": "XP-T", "action": action,
        "selected_gpu": 0, "monotonic_sec": float(idx),
        "tracked_roots": [worker] if action == "HEARTBEAT" else [],
        "gpu_observations": [{
            "index": 0, "total_mib": 97887, "used_mib": used,
            "utilization_pct": util,
            "process_pids": [pid] if pid is not None else [],
            "foreign_pids": [], "visibility": "known",
        }],
    }


def observations(dataset):
    if dataset == "ffhq":
        raw = [
            ("PRE_GPU", None, 0, 0),
            ("HEARTBEAT", None, 3, 0),
            ("HEARTBEAT", 42, 3, 0),
            ("HEARTBEAT", 42, 1603, 65),
            ("HEARTBEAT", 42, 1603, 70),
            ("HEARTBEAT", 42, 1603, 69),
            ("HEARTBEAT", None, 0, 0),
            ("POST_GPU", None, 0, 0),
        ]
    else:
        raw = [
            ("PRE_GPU", None, 0, 0),
            ("HEARTBEAT", None, 0, 0),
            ("HEARTBEAT", None, 3, 0),
            ("HEARTBEAT", None, 3, 0),
            ("HEARTBEAT", None, 3, 0),
            *[("HEARTBEAT", 42, 4023, u) for u in (83, 82, 84, 81, 84, 82)],
            ("HEARTBEAT", None, 4012, 0),
            ("POST_GPU", None, 0, 0),
        ]
    return [event(action, i, pid=pid, used=used, util=util)
            for i, (action, pid, used, util) in enumerate(raw)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FullTimingAudit(unittest.TestCase):
    def test_two_pilots_provenance_and_json_receipt(self):
        expected_head = "a" * 40
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            base = root / "outputs/pr_diffusion/track_r"
            base.mkdir(parents=True)
            wt = root / "pr_diffusion_track_r"
            wt.mkdir()
            source_rows = []
            original_hashes = {}
            for dataset, source_head in (("ffhq", "b" * 40),
                                         ("imagenet", "c" * 40)):
                name = "xp_t_fast_fixture_" + dataset
                folder = base / name
                folder.mkdir()
                events = observations(dataset)
                has_tail = dataset == "imagenet"
                timing = {
                    "timing_class": "UNVERIFIED" if has_tail else "CLEAN",
                    "timing_valid": not has_tail,
                    "n_samples": len(events),
                    "foreign_pids": [],
                    "reasons": ["unattributed_gpu_memory"] if has_tail else [],
                }
                status = {
                    "state": "PASS", "head": source_head,
                    "worker_rc": 0, "worker_pid": 42,
                    "gpu": 0, "timing": timing,
                }
                completion = {
                    "status": "XP_T_FAST_" + dataset.upper() + "_BATCH1_PASS",
                    "head": source_head, "gpu_index": 0,
                    "dataset": dataset, "timing": timing,
                    "new_roots": 9, "n_images": 3,
                    "sampling_wall_s_median": 7.191 if has_tail else 3.321,
                    "sampling_cuda_event_s_median": 7.191 if has_tail else 3.321,
                    "gpu_worker_wall_s": 112.516 if has_tail else 61.347,
                    "gpu_hours": 0.03125 if has_tail else 0.01704,
                }
                mapping = {
                    "GPU_EVENTS.jsonl": "".join(json.dumps(e) + "\n" for e in events),
                    "XP_T_FAST_STATUS.json": json.dumps(status),
                    "XP_T_FAST_COMPLETE.json": json.dumps(completion),
                }
                for filename, content in mapping.items():
                    (folder / filename).write_text(content)
                    original_hashes[str(folder / filename)] = sha(folder / filename)
                source_rows.append((
                    dataset, name, source_head,
                    sha(folder / "XP_T_FAST_COMPLETE.json"),
                ))
            out = base / "xp_t_timing_audit_integration"
            with (patch.object(audit, "ROOT", root),
                  patch.object(audit, "WT", wt),
                  patch.object(audit, "BASE", base),
                  patch.object(audit, "SOURCES", source_rows),
                  patch.object(audit.subprocess, "check_output",
                               side_effect=[expected_head + "\n", ""]),
                  patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""})):
                audit.main(SimpleNamespace(
                    expected_head=expected_head, out=str(out)))
            result = json.loads((out / "XP_T_FAST_TIMING_REAUDIT.json").read_text())
            self.assertEqual(result["status"], "XP_T_FAST_TIMING_REAUDIT_PASS")
            self.assertTrue(result["original_receipts_immutable"])
            self.assertFalse(result["gpu_generation_performed"])
            self.assertEqual(result["n_datasets"], 2)
            by_dataset = {x["dataset"]: x for x in result["results"]}
            self.assertEqual(by_dataset["ffhq"]["qualified_timing_class"],
                             "CLEAN_ORIGINAL")
            self.assertEqual(by_dataset["imagenet"]["qualified_timing_class"],
                             "CLEAN_POSTHOC")
            self.assertEqual(by_dataset["ffhq"]["n_active_worker_visible"], 3)
            self.assertEqual(by_dataset["imagenet"]["n_active_worker_visible"], 6)
            self.assertEqual(by_dataset["imagenet"]["n_unattributed_post_exit_samples"], 1)
            for original, digest in original_hashes.items():
                self.assertEqual(sha(Path(original)), digest)
            self.assertEqual(by_dataset["ffhq"]["source_head"], "b" * 40)
            self.assertEqual(by_dataset["imagenet"]["source_head"], "c" * 40)
            self.assertEqual(by_dataset["imagenet"]["source_events_sha256"],
                             sha(base / "xp_t_fast_fixture_imagenet" / "GPU_EVENTS.jsonl"))


if __name__ == "__main__":
    unittest.main()
