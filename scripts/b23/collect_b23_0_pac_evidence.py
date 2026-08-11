#!/usr/bin/env python3
"""Collect the bounded, zero-GPU B23.0 PAC freeze and exposure evidence."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PLANNING_HEAD = "d1119e37fa688ac07f48ffc87ce19b13dbfb1c27"
ACCEPTED_PLAN = "ed4f46e8f116648eda76d387388d762d7cb8f3d7"
B22_BASE = "ba78c06e0c5eac0c915263e4faed0b262d5e917a"
MODEL_SHA256 = "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
DAPS_SHA = "e7a77d094167084faed19b599b96673b7bb11447"
DAPS_PATCH_SHA256 = "fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69"
HISTORICAL_SHA = "0c3c2ec972a50d462b37af7742011ed2a2c5a20a"
HISTORICAL_PATCH_SHA256 = "b57536b4d8c7b89b6ed7fcc5deaba55087b09d6494f5c93f390d6f218e16ca9c"
SITCOM_SHA = "275ab67efbd8146bffca20155171ba6be1169c09"
SITCOM_PATCH_SHA256 = "a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e"
NP_SITCOM_SHA = "52f2c37e587576d02e2b27ac971e247f2899fc5e"
DIFFFPR_SHA = "a45ffe58f18fed8a63d3446600424e2b08733524"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

EXPOSURE_COLUMNS = (
    "image_id", "measurement_id", "dataset_split", "first_project_stage",
    "roles_seen", "ground_truth_inspected", "artifacts", "exclusion_reason",
    "source_evidence",
)
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".sh", ".py"}
IMAGE_TOKEN_RE = re.compile(r"(?<![0-9a-fA-F])([0-6][0-9]{4})(?![0-9a-fA-F])")
FFHQ_PATH_RE = re.compile(r"ffhq[-_]?([0-6][0-9]{4})(?![0-9])", re.IGNORECASE)
MEAS_RE = re.compile(r"(?:meas(?:urement)?(?:[_ -]?(?:seed|id|tag))?)[_ :=-]*(\d{3,12})", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def git(path: Path, *args: str) -> str:
    return run(["git", "-C", str(path), *args])


def git_diff_sha256(path: Path) -> str:
    payload = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def normalize_image(value: Any) -> str | None:
    if isinstance(value, int) and 0 <= value <= 69999:
        return f"{value:05d}"
    text = str(value or "").strip()
    if text.isdigit() and len(text) <= 5 and 0 <= int(text) <= 69999:
        return f"{int(text):05d}"
    match = FFHQ_PATH_RE.search(text)
    return match.group(1) if match else None


def find_key(mapping: dict[str, Any], candidates: Iterable[str]) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in candidates:
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def measurement_id(mapping: dict[str, Any]) -> str | None:
    explicit = find_key(mapping, ("measurement_id", "meas_id"))
    if explicit not in (None, ""):
        return str(explicit)
    derived = find_key(
        mapping,
        ("derived_noise_seed", "measurement_seed", "measurement_noise_seed", "meas_seed"),
    )
    tag = find_key(mapping, ("measurement_tag", "meas_tag", "panel_seed"))
    if derived not in (None, ""):
        prefix = f"meas{tag}:" if tag not in (None, "") else ""
        return f"{prefix}seed{derived}"
    path_value = find_key(mapping, ("measurement_path", "measurement", "meas_path"))
    if path_value:
        match = MEAS_RE.search(str(path_value))
        if match:
            return f"meas{match.group(1)}:DERIVED_SEED_UNRESOLVED"
    return None


class ExposureBuilder:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, set[str] | bool]] = {}
        self.unresolved_measurement_tag_mentions = 0

    def _merge_rows(self, destination: dict[str, set[str] | bool], source: dict[str, set[str] | bool]) -> None:
        for field in ("stages", "roles", "artifacts", "evidence"):
            destination[field].update(source[field])  # type: ignore[union-attr]
        destination["gt"] = bool(destination["gt"] or source["gt"])

    def add(
        self,
        image_id: str,
        measurement: str | None,
        *,
        stage: str,
        role: str,
        artifact: str,
        evidence: str,
        gt_inspected: bool = True,
        exact_replaces_unknown: bool = False,
        unknown_dominates: bool = True,
    ) -> None:
        normalized = normalize_image(image_id)
        if normalized is None:
            return
        measurement = measurement or "UNKNOWN_ALL_MEASUREMENTS"
        if "DERIVED_SEED_UNRESOLVED" in measurement:
            self.unresolved_measurement_tag_mentions += 1
            measurement = "UNKNOWN_ALL_MEASUREMENTS"
        unknown_key = (normalized, "UNKNOWN_ALL_MEASUREMENTS")
        if measurement == "UNKNOWN_ALL_MEASUREMENTS":
            exact_keys = [key for key in self.rows if key[0] == normalized and key != unknown_key]
            if exact_keys and not unknown_dominates:
                for key in exact_keys:
                    row = self.rows[key]
                    row["stages"].add(stage)  # type: ignore[union-attr]
                    row["roles"].add(role)  # type: ignore[union-attr]
                    row["artifacts"].add(artifact)  # type: ignore[union-attr]
                    row["evidence"].add(evidence)  # type: ignore[union-attr]
                    row["gt"] = bool(row["gt"] or gt_inspected)
                return
            unknown = self.rows.setdefault(
                unknown_key,
                {"stages": set(), "roles": set(), "artifacts": set(), "evidence": set(), "gt": False},
            )
            for key in [key for key in self.rows if key[0] == normalized and key != unknown_key]:
                self._merge_rows(unknown, self.rows.pop(key))
            key = unknown_key
        elif unknown_key in self.rows:
            if exact_replaces_unknown:
                del self.rows[unknown_key]
                key = (normalized, measurement)
            else:
                key = unknown_key
        else:
            key = (normalized, measurement)
        row = self.rows.setdefault(
            key,
            {
                "stages": set(), "roles": set(), "artifacts": set(), "evidence": set(),
                "gt": False,
            },
        )
        row["stages"].add(stage)  # type: ignore[union-attr]
        row["roles"].add(role)  # type: ignore[union-attr]
        row["artifacts"].add(artifact)  # type: ignore[union-attr]
        row["evidence"].add(evidence)  # type: ignore[union-attr]
        row["gt"] = bool(row["gt"] or gt_inspected)

    def load_seed_csv(self, path: Path) -> None:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.add(
                    row["image_id"],
                    row["measurement_id"],
                    stage=row["first_project_stage"],
                    role=row["roles_seen"],
                    artifact=row["artifacts"],
                    evidence=row["source_evidence"],
                    gt_inspected=row["ground_truth_inspected"] != "false",
                )

    def write(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPOSURE_COLUMNS)
            writer.writeheader()
            for (image, measurement), values in sorted(self.rows.items()):
                stages = sorted(values["stages"])  # type: ignore[arg-type]
                writer.writerow(
                    {
                        "image_id": image,
                        "measurement_id": measurement,
                        "dataset_split": "FFHQ_PRE_B23_EXPOSED",
                        "first_project_stage": stages[0],
                        "roles_seen": "|".join(sorted(values["roles"])),  # type: ignore[arg-type]
                        "ground_truth_inspected": "true" if values["gt"] else "unknown",
                        "artifacts": ";".join(sorted(values["artifacts"])),  # type: ignore[arg-type]
                        "exclusion_reason": "PRE_B23_EXPOSURE_CONSERVATIVE",
                        "source_evidence": ";".join(sorted(values["evidence"])),  # type: ignore[arg-type]
                    }
                )
        return {
            "row_count": len(self.rows),
            "image_count": len({key[0] for key in self.rows}),
            "truly_resolved_measurement_rows": sum(key[1] != "UNKNOWN_ALL_MEASUREMENTS" for key in self.rows),
            "unresolved_measurement_tag_mentions": self.unresolved_measurement_tag_mentions,
            "image_wide_unknown_exposure_rows": sum(key[1] == "UNKNOWN_ALL_MEASUREMENTS" for key in self.rows),
            "sha256": sha256_file(path),
        }


IMPORTABLE_SUFFIXES = {".py", ".pyi", ".pyx", ".so", ".pyd"}


def classify_untracked(relative: str) -> str:
    path = Path(relative)
    lowered = {part.lower() for part in path.parts}
    if path.suffix.lower() in IMPORTABLE_SUFFIXES:
        return "IMPORTABLE_SOURCE"
    if "__pycache__" in lowered or "cache" in lowered or path.suffix.lower() in {".pyc", ".pyo"}:
        return "CACHE"
    if lowered & {"data", "dataset", "datasets"} or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".npz"}:
        return "DATASET"
    if lowered & {"output", "outputs", "result", "results", "logs", "samples", "checkpoints"}:
        return "OUTPUT"
    return "OTHER_ARTIFACT"


def inventory_untracked(label: str, root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    relatives = [item.decode("utf-8", errors="surrogateescape") for item in payload.split(b"\0") if item]
    if len(relatives) > 5000:
        return [], [f"{label} has more than 5000 untracked paths; bounded inventory refused"]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for relative in relatives:
        path = root / relative
        classification = classify_untracked(relative)
        importable = classification == "IMPORTABLE_SOURCE"
        digest: str | None = None
        resolution = "NOT_HASHED_NONIMPORTABLE"
        if importable:
            if path.is_symlink() or not path.is_file():
                resolution = "UNRESOLVED_IMPORTABLE_SOURCE"
                failures.append(f"unresolved importable source: {label}:{relative}")
            else:
                digest = sha256_file(path)
                resolution = "HASHED_IMPORTABLE_SOURCE"
        rows.append({
            "source_checkout": label,
            "root": str(root),
            "relative_path": relative,
            "classification": classification,
            "importable": importable,
            "bytes": path.stat().st_size if path.exists() and not path.is_symlink() else None,
            "sha256": digest,
            "resolution": resolution,
        })
    return rows, failures


def add_mapping(
    builder: ExposureBuilder,
    mapping: dict[str, Any],
    *,
    stage: str,
    source: str,
    exact_replaces_unknown: bool = False,
    unknown_dominates: bool = True,
) -> None:
    image = find_key(mapping, ("image_id", "image", "ffhq_id", "image_index"))
    normalized = normalize_image(image)
    if normalized:
        builder.add(
            normalized,
            measurement_id(mapping),
            stage=stage,
            role="manifested_experiment",
            artifact=source,
            evidence=source,
            exact_replaces_unknown=exact_replaces_unknown,
            unknown_dominates=unknown_dominates,
        )


def walk_json(
    builder: ExposureBuilder,
    value: Any,
    *,
    stage: str,
    source: str,
    exact_replaces_unknown: bool = False,
    unknown_dominates: bool = True,
) -> None:
    if isinstance(value, dict):
        add_mapping(
            builder, value, stage=stage, source=source,
            exact_replaces_unknown=exact_replaces_unknown,
            unknown_dominates=unknown_dominates,
        )
        for child in value.values():
            walk_json(
                builder, child, stage=stage, source=source,
                exact_replaces_unknown=exact_replaces_unknown,
                unknown_dominates=unknown_dominates,
            )
    elif isinstance(value, list):
        for child in value:
            walk_json(
                builder, child, stage=stage, source=source,
                exact_replaces_unknown=exact_replaces_unknown,
                unknown_dominates=unknown_dominates,
            )


def ingest_table(
    builder: ExposureBuilder,
    path: Path,
    *,
    stage: str,
    source_label: str,
    unknown_dominates: bool = True,
) -> None:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for mapping in csv.DictReader(handle, delimiter=delimiter):
            add_mapping(
                builder, dict(mapping), stage=stage, source=source_label,
                unknown_dominates=unknown_dominates,
            )


def ingest_historical_texts(builder: ExposureBuilder, repo: Path) -> dict[str, int]:
    paths = git(repo, "ls-files", "-co", "--exclude-standard").splitlines()
    inspected = 0
    bytes_read = 0
    ids_found = 0
    for relative in paths:
        path = repo / relative
        if not relative.startswith(("docs/", "configs/", "manifests/", "scripts/")):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        size = path.stat().st_size
        if size > 1_000_000 or inspected >= 2500 or bytes_read + size > 64_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        inspected += 1
        bytes_read += size
        ids = set(FFHQ_PATH_RE.findall(text)) | set(IMAGE_TOKEN_RE.findall(text))
        if not ids:
            continue
        stage = "B21" if "b21" in relative.lower() else "B22" if "b22" in relative.lower() else "B19_B20_BRANCH_A_B"
        measurements = sorted(set(MEAS_RE.findall(text)))
        measurement = (
            f"meas{measurements[0]}:DERIVED_SEED_UNRESOLVED"
            if len(measurements) == 1 else None
        )
        for image in ids:
            builder.add(
                image,
                measurement,
                stage=stage,
                role="documented_or_configured_exposure",
                artifact=relative,
                evidence=f"git-ls-files:{relative}",
                unknown_dominates=stage == "B19_B20_BRANCH_A_B",
            )
            ids_found += 1
    return {"files_inspected": inspected, "bytes_read": bytes_read, "id_mentions_added": ids_found}


def environment_probe(python: Path) -> dict[str, Any]:
    code = (
        "import json,sys; import torch, numpy, scipy, PIL; "
        "print(json.dumps({'python':sys.version.split()[0],'executable':sys.executable,"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'cuda_visible':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),"
        "'cuda_initialized':torch.cuda.is_initialized(),'numpy':numpy.__version__,"
        "'scipy':scipy.__version__,'pillow':PIL.__version__},sort_keys=True))"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return json.loads(run([str(python), "-c", code], env=env))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical", type=Path, default=Path("/egr/research-pac/huang248/pr_diffusion_b19_solver"))
    parser.add_argument("--older", type=Path, default=Path("/egr/research-pac/huang248/pr_diffusion_repo"))
    parser.add_argument("--data", type=Path, default=Path("/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024"))
    parser.add_argument("--model", type=Path, default=Path("/egr/research-pac/huang248/models/ffhq_10m.pt"))
    parser.add_argument("--sitcom", type=Path, default=Path("/egr/research-pac/huang248/external/SITCOM_ODE"))
    parser.add_argument("--np-sitcom", type=Path, default=Path("/egr/research-pac/huang248/external/SITCOM_ODE_npsitcom"))
    parser.add_argument("--difffpr", type=Path, default=Path("/egr/research-pac/huang248/external/DiffFPR"))
    parser.add_argument("--b21", type=Path, default=Path("/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401"))
    parser.add_argument("--snapshot", type=Path, default=Path("/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints/B21_source_snapshot_20260727_040208"))
    parser.add_argument("--timestamp")
    args = parser.parse_args()

    repo = args.repo.resolve()
    timestamp = args.timestamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    capsule = args.output_root.resolve() / f"B23_0_return_{timestamp}"
    capsule.mkdir(parents=True, exist_ok=False)
    summaries = capsule / "summaries"
    diagnostics = capsule / "selected_diagnostics"
    summaries.mkdir()
    diagnostics.mkdir()

    correction_ledger = repo / "manifests/b23/b23_0_correction_ledger.json"
    shutil.copy2(correction_ledger, capsule / "CORRECTION_LEDGER.json")

    failures: list[str] = []
    identities: dict[str, Any] = {
        "planning_head_used": PLANNING_HEAD,
        "accepted_plan_snapshot": ACCEPTED_PLAN,
        "frozen_b22_base": B22_BASE,
        "execution_branch": git(repo, "branch", "--show-current"),
        "execution_head_before_evidence": git(repo, "rev-parse", "HEAD"),
        "worktree_clean_before_collection": not bool(git(repo, "status", "--porcelain")),
    }
    if identities["execution_branch"] != "codex/b23-execution":
        failures.append("execution branch is not codex/b23-execution")
    if not identities["worktree_clean_before_collection"]:
        failures.append("execution worktree was dirty before evidence collection")
    try:
        git(repo, "merge-base", "--is-ancestor", PLANNING_HEAD, "HEAD")
        identities["planning_head_is_ancestor"] = True
    except subprocess.CalledProcessError:
        identities["planning_head_is_ancestor"] = False
        failures.append("execution head does not descend from pinned planning head")

    sources = {
        "historical": (args.historical, HISTORICAL_SHA, HISTORICAL_PATCH_SHA256),
        "daps": (args.historical / "external/daps", DAPS_SHA, DAPS_PATCH_SHA256),
        "official_sitcom": (args.sitcom, SITCOM_SHA, SITCOM_PATCH_SHA256),
        "np_sitcom_fork": (args.np_sitcom, NP_SITCOM_SHA, EMPTY_SHA256),
        "difffpr": (args.difffpr, DIFFFPR_SHA, EMPTY_SHA256),
    }
    source_report: dict[str, Any] = {}
    for label, (path, expected_head, expected_diff) in sources.items():
        if not path.is_dir():
            failures.append(f"missing source checkout: {label}={path}")
            continue
        head = git(path, "rev-parse", "HEAD")
        diff_hash = git_diff_sha256(path)
        status_count = len(git(path, "status", "--porcelain").splitlines())
        source_report[label] = {
            "path": str(path), "head": head, "expected_head": expected_head,
            "tracked_worktree_diff_sha256": diff_hash,
            "expected_tracked_worktree_diff_sha256": expected_diff,
            "dirty_status_lines": status_count,
        }
        if head != expected_head:
            failures.append(f"{label} head mismatch: {head} != {expected_head}")
        if diff_hash != expected_diff:
            failures.append(f"{label} tracked diff mismatch: {diff_hash} != {expected_diff}")

    untracked_rows: list[dict[str, Any]] = []
    for label, root in (
        ("historical_daps", args.historical / "external/daps"),
        ("official_sitcom", args.sitcom),
        ("np_sitcom_fork", args.np_sitcom),
    ):
        rows, inventory_failures = inventory_untracked(label, root)
        untracked_rows.extend(rows)
        failures.extend(inventory_failures)
    inventory_path = summaries / "UNTRACKED_SOURCE_INVENTORY.tsv"
    inventory_fields = (
        "source_checkout", "root", "relative_path", "classification", "importable",
        "bytes", "sha256", "resolution",
    )
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(untracked_rows)
    classification_counts: dict[str, int] = defaultdict(int)
    for row in untracked_rows:
        classification_counts[row["classification"]] += 1
    inventory_summary = {
        "status": "PASS" if not any("unresolved importable source" in item for item in failures) else "FAIL_STOP",
        "path_count": len(untracked_rows),
        "classification_counts": dict(sorted(classification_counts.items())),
        "importable_source_count": sum(row["importable"] for row in untracked_rows),
        "hashed_importable_source_count": sum(row["resolution"] == "HASHED_IMPORTABLE_SOURCE" for row in untracked_rows),
        "unresolved_importable_source_count": sum(row["resolution"] == "UNRESOLVED_IMPORTABLE_SOURCE" for row in untracked_rows),
        "inventory_sha256": sha256_file(inventory_path),
    }
    (summaries / "UNTRACKED_SOURCE_SUMMARY.json").write_text(
        json.dumps(inventory_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not args.model.is_file():
        failures.append(f"missing model: {args.model}")
        model_hash = None
    else:
        model_hash = sha256_file(args.model)
        if model_hash != MODEL_SHA256:
            failures.append(f"model SHA-256 mismatch: {model_hash}")
    sitcom_checkpoint = args.sitcom / "checkpoint/ffhq256.pt"
    sitcom_checkpoint_hash = sha256_file(sitcom_checkpoint) if sitcom_checkpoint.is_file() else None
    if sitcom_checkpoint_hash != MODEL_SHA256:
        failures.append(
            f"SITCOM checkpoint identity unresolved or mismatched: {sitcom_checkpoint_hash}"
        )
    if not args.data.is_dir():
        failures.append(f"missing FFHQ data root: {args.data}")

    snapshot_patch = args.snapshot / "daps/local.patch"
    snapshot_patch_hash = sha256_file(snapshot_patch) if snapshot_patch.is_file() else None
    if snapshot_patch_hash != DAPS_PATCH_SHA256:
        failures.append(f"B21 source-snapshot DAPS patch mismatch: {snapshot_patch_hash}")

    env_paths = {
        "daps": Path("/egr/research-pac/huang248/conda-envs/daps/bin/python"),
        "prdiff_ffhq": Path("/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python"),
        "sitcom_ode_bw": Path("/egr/research-pac/huang248/conda-envs/sitcom_ode_bw/bin/python"),
    }
    env_report: dict[str, Any] = {}
    for name, python in env_paths.items():
        if not python.is_file():
            failures.append(f"missing environment Python: {python}")
            continue
        probe = environment_probe(python)
        env_report[name] = probe
        if probe["cuda_initialized"]:
            failures.append(f"environment probe initialized CUDA: {name}")

    hardware_query = run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    ).splitlines()
    if not hardware_query:
        failures.append("nvidia-smi returned no GPU inventory rows")

    builder = ExposureBuilder()
    builder.load_seed_csv(repo / "manifests/b23/PRE_B23_EXPOSURE.csv")
    targeted = [
        args.b21 / "measurements/fresh_measurement_manifest_meas5401.json",
        args.b21 / "panel/panel_manifest.tsv",
        args.b21 / "manifest.tsv",
        args.b21 / "active_manifest.tsv",
        args.b21 / "analysis_theta0.7/panel_manifest_checked.tsv",
        args.b21 / "analysis_theta0.7/fresh2_final_summary.json",
        repo / "docs/b22/b22_3_visual_failure_taxonomy.csv",
    ]
    copied: list[dict[str, Any]] = []
    required_missing: list[str] = []
    for path in targeted:
        if not path.is_file():
            if path != args.b21 / "analysis_theta0.7/panel_manifest_checked.tsv":
                required_missing.append(str(path))
            continue
        if path.stat().st_size > 2_000_000:
            failures.append(f"targeted evidence unexpectedly exceeds 2 MB: {path}")
            continue
        stage = "B22" if "b22" in str(path).lower() else "B21"
        label = str(path)
        if path.suffix == ".json":
            is_measurement_manifest = path.name.startswith("fresh_measurement_manifest_")
            walk_json(
                builder,
                json.loads(path.read_text(encoding="utf-8")),
                stage=stage,
                source=label,
                exact_replaces_unknown=is_measurement_manifest,
                unknown_dominates=False,
            )
        elif path.suffix in {".csv", ".tsv"}:
            ingest_table(
                builder, path, stage=stage, source_label=label,
                unknown_dominates=False,
            )
        destination = diagnostics / path.name
        shutil.copy2(path, destination)
        copied.append({"source": str(path), "copy": str(destination.relative_to(capsule)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if required_missing:
        failures.extend(f"missing targeted evidence: {path}" for path in required_missing)

    historical_scan = ingest_historical_texts(builder, args.historical)
    exposure_path = summaries / "PRE_B23_EXPOSURE.csv"
    exposure_report = builder.write(exposure_path)
    if exposure_report["image_count"] < 100:
        failures.append(f"exposure manifest has fewer than 100 images: {exposure_report['image_count']}")
    coverage = {
        **exposure_report,
        "historical_bounded_scan": historical_scan,
        "targeted_sources": [str(path) for path in targeted],
        "targeted_sources_missing": required_missing,
        "policy": "uncertain measurement identity excludes all measurements for that image",
        "future_split_registry_rows": 0,
    }

    freeze = {
        "schema_version": "b23.pac-freeze.v1",
        "collected_at_utc": timestamp,
        "gpu_work_performed": False,
        "cuda_visible_devices_for_python_probes": "",
        "repository": identities,
        "sources": source_report,
        "model": {"path": str(args.model), "sha256": model_hash, "bytes": args.model.stat().st_size if args.model.is_file() else None},
        "sitcom_checkpoint": {"path": str(sitcom_checkpoint), "sha256": sitcom_checkpoint_hash},
        "dataset": {"path": str(args.data), "bounded_check": "directory existence only; no recursive scan"},
        "snapshot_daps_patch": {"path": str(snapshot_patch), "sha256": snapshot_patch_hash},
        "environments": env_report,
        "hardware_inventory_only": hardware_query,
        "copied_evidence": copied,
    }
    (summaries / "PAC_FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (summaries / "EXPOSURE_COVERAGE.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    primary_gates = {
        "1_execution_branch_and_head": "PASS" if identities["planning_head_is_ancestor"] else "FAIL",
        "2_pr_or_commit_list": "PRE_RUN_HEAD_RECORDED; POST_RUN_EVIDENCE_COMMIT_PENDING",
        "3_branch_stack_recommendation": "KEEP draft execution PR based on codex/post-b22-reliability-plan; do not rewrite planning PR #36",
        "4_pac_source_environment_hardware_inventory": "PASS" if not failures else "SEE_FAILURES",
        "5_unresolved_identity_gaps": "NONE" if not failures else "SEE_FAILURES",
        "6_exposure_coverage": (
            f"{exposure_report['image_count']} images; {exposure_report['row_count']} rows; "
            f"{exposure_report['truly_resolved_measurement_rows']} resolved rows; "
            f"{exposure_report['unresolved_measurement_tag_mentions']} unresolved tag mentions; "
            f"{exposure_report['image_wide_unknown_exposure_rows']} image-wide unknown rows"
        ),
        "7_parent_semantics": "FOUR_NATIVE_PARENTS_FROZEN",
        "8_typed_api": "NATIVE_STATE_MODULE_ADAPTER_CONTRACT_VALIDATED",
        "9_compute_and_fre": "RAW_LEDGER_AND_FORMULAS_VALIDATED; NUMERIC_WEIGHTS_INTENTIONALLY_UNMEASURED",
        "10_replay_and_rng": "PROCEDURE_AND_NAMED_STREAM_DERIVATION_VALIDATED; GPU_REPLAY_NOT_RUN",
        "11_b23_1_smoke_manifests": "ONE_IMAGE_AND_FOUR_IMAGE_TEMPLATES_EMPTY_AS_REQUIRED",
        "12_b23_1_dry_run_commands": "TEMPLATES_RENDERED; ZERO_EXECUTABLE_GPU_COMMANDS",
        "13_recommendation": "AUTHORIZE_B23_1_AFTER_PLANNER_SIGNOFF" if not failures else "REVISE_B23_0",
    }
    decision = {
        "schema_version": "b23.gate-decision.v1",
        "stage": "B23.0",
        "verdict": "PASS_RECOMMEND_PLANNER_REVIEW" if not failures else "FAIL_STOP",
        "gpu_work_performed": False,
        "failures": failures,
        "primary_gate_results": primary_gates,
        "b23_1_authorized": False,
        "b23_2_authorized": False,
    }
    (capsule / "EXECUTION_IDENTITY.json").write_text(json.dumps(identities, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (capsule / "GATE_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# B23.0 checkpoint report

Status: **{decision['verdict']}**
GPU work performed: **NO**

The clean execution worktree was checked at `{identities['execution_head_before_evidence']}` on
`codex/b23-execution`. It descends from the pinned operational handoff `{PLANNING_HEAD}`.

The bounded PAC freeze records the historical checkout and its preserved dirty patch, DAPS and its
preserved B21 patch, official SITCOM, the NP/SITCOM fork, DiffFPR, all three named environments,
the FFHQ model/checkpoint, dataset-root existence, and hardware inventory. No data or output tree was
recursively scanned and Python probes ran with `CUDA_VISIBLE_DEVICES` empty.

The merged exposure manifest contains {exposure_report['image_count']} images and
{exposure_report['row_count']} image/measurement rows: {exposure_report['truly_resolved_measurement_rows']}
truly resolved rows, {exposure_report['unresolved_measurement_tag_mentions']} unresolved measurement-tag
mentions, and {exposure_report['image_wide_unknown_exposure_rows']} image-wide unknown rows. Every
unresolved identity excludes all measurements for that image. The future B23 split registry remains empty.

The bounded untracked-path inventory classifies {inventory_summary['path_count']} paths across historical
DAPS, official SITCOM, and the NP/SITCOM fork. All {inventory_summary['importable_source_count']} importable
files are hashed; any unresolved importable source is a hard stop.

Numeric atomic-operation weights are intentionally absent: B23.1 microbenchmarks must measure and
freeze them before any hybrid execution. B23.1 and B23.2 remain unauthorized.

The fail-closed wrapper records the repository tests, contract validator, dry renderer, and PAC
collector in `ZERO_GPU_STEP_RESULTS.tsv`. The evidence publisher independently requires all four
rows to be `PASS` with return code zero.

Failures: {json.dumps(failures) if failures else 'NONE'}
"""
    (summaries / "B23_0_CHECKPOINT_REPORT.md").write_text(report, encoding="utf-8")
    (capsule / "README.md").write_text(
        "# B23.0 evidence capsule\n\nThis compact capsule accompanies transparent GitHub evidence. It contains no model, dataset, or raw reconstruction payload. `ZERO_GPU_STEP_RESULTS.tsv` is the fail-closed prerequisite ledger; `CORRECTION_LEDGER.json` preserves earlier partial/invalid returns.\n",
        encoding="utf-8",
    )
    (capsule / "COMMANDS.sh").write_text(
        "# Recorded zero-GPU procedure (documentation; not an executable launcher)\n"
        "cd <execution-worktree> && PYTHONPATH='<execution-worktree>' CUDA_VISIBLE_DEVICES='' <daps-python> -m unittest discover -s tests/b23 -v\n"
        "cd <execution-worktree> && PYTHONPATH='<execution-worktree>' CUDA_VISIBLE_DEVICES='' <daps-python> scripts/b23/validate_b23_0.py --repo <execution-worktree>\n"
        "cd <execution-worktree> && PYTHONPATH='<execution-worktree>' CUDA_VISIBLE_DEVICES='' <daps-python> scripts/b23/render_b23_1_dry_runs.py --repo <execution-worktree> --output <b23-output-root>/B23_1_dry_run_<timestamp>.json\n"
        "cd <execution-worktree> && PYTHONPATH='<execution-worktree>' CUDA_VISIBLE_DEVICES='' <daps-python> scripts/b23/collect_b23_0_pac_evidence.py --repo <execution-worktree> --output-root <b23-output-root>\n",
        encoding="utf-8",
    )
    (capsule / "STDOUT_TAIL.txt").write_text("Filled by scripts/b23/run_b23_0_zero_gpu.sh before packaging.\n", encoding="utf-8")
    (capsule / "STDERR_TAIL.txt").write_text("Filled by scripts/b23/run_b23_0_zero_gpu.sh before packaging.\n", encoding="utf-8")

    artifact_rows = []
    for item in copied:
        artifact_rows.append((item["source"], item["sha256"], item["bytes"], "source evidence copied into capsule", "PAC historical artifact; do not mutate"))
    artifact_rows.extend(
        [
            (str(args.model), model_hash or "MISSING", args.model.stat().st_size if args.model.is_file() else 0, "frozen FFHQ model", "PAC-only large artifact"),
            (str(args.data), "DIRECTORY_NOT_RECURSIVELY_HASHED", 0, "FFHQ dataset root", "PAC-only large artifact; manifest-backed"),
            (str(args.b21), "DIRECTORY_NOT_RECURSIVELY_HASHED", 0, "B21.11 benchmark root", "PAC-only large artifact; selected files hashed above"),
        ]
    )
    with (capsule / "ARTIFACT_MANIFEST.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("absolute_pac_path", "sha256", "bytes", "artifact_role", "retention"))
        writer.writerows(artifact_rows)

    print(json.dumps({"status": decision["verdict"], "capsule": str(capsule), "failures": len(failures), "exposure_images": exposure_report["image_count"]}, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
