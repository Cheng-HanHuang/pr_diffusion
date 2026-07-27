#!/usr/bin/env bash
set -uo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS_PY=${DAPS_PY:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
B21_ROOT=${B21_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
B21_11=${B21_11:-$B21_ROOT/B21_11_fresh2_final_val100_meas5401}
B21_12=${B21_12:-$B21_11/b21_12_failure_atlas}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$CHECKPOINT_ROOT/B21_to_B22_$STAMP}
CREATE_GIT_BUNDLE=${CREATE_GIT_BUNDLE:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}

mkdir -p \
  "$OUT/repo" \
  "$OUT/environment" \
  "$OUT/artifacts/reports" \
  "$OUT/artifacts/atlas_sheets" \
  "$OUT/artifacts/manifests"

exec > >(tee "$OUT/checkpoint_stdout.log") 2>&1

section() {
  printf '\n===== %s =====\n' "$1"
}

record_command() {
  local outfile=$1
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } >"$outfile" 2>&1 || true
}

copy_if_present() {
  local source=$1
  local destination_dir=$2
  if [[ -f "$source" ]]; then
    cp -p "$source" "$destination_dir/"
    echo "[copy] $source"
  else
    echo "[missing] $source"
  fi
}

section "checkpoint metadata"
{
  echo "checkpoint_created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "checkpoint_created_local=$(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "hostname=$(hostname)"
  echo "user=$(id -un)"
  echo "repo=$REPO"
  echo "daps_python=$DAPS_PY"
  echo "b21_root=$B21_ROOT"
  echo "b21_11=$B21_11"
  echo "b21_12=$B21_12"
  echo "output=$OUT"
  echo "create_git_bundle=$CREATE_GIT_BUNDLE"
  echo "create_archive=$CREATE_ARCHIVE"
} | tee "$OUT/checkpoint_metadata.txt"

record_command "$OUT/environment/uname.txt" uname -a
record_command "$OUT/environment/identity.txt" id
record_command "$OUT/environment/disk_filesystems.txt" df -h
record_command "$OUT/environment/memory.txt" free -h
record_command "$OUT/environment/gpus.txt" nvidia-smi

section "repository state"
if [[ ! -d "$REPO/.git" ]]; then
  echo "[fatal] repository is not a git checkout: $REPO" >&2
  exit 2
fi

cd "$REPO"

git status --short --branch | tee "$OUT/repo/status_short_branch.txt"
git status --porcelain=v2 --branch > "$OUT/repo/status_porcelain_v2.txt"
git rev-parse HEAD | tee "$OUT/repo/head.txt"
git branch --show-current | tee "$OUT/repo/current_branch.txt"
git remote -v > "$OUT/repo/remotes.txt"
git branch -avv > "$OUT/repo/branches.txt"
git log --graph --decorate --oneline --all -n 200 > "$OUT/repo/log_graph_200.txt"
git submodule status --recursive > "$OUT/repo/submodule_status.txt" 2>&1 || true
git diff --stat > "$OUT/repo/worktree_diff_stat.txt"
git diff > "$OUT/repo/worktree.patch"
git diff --cached --stat > "$OUT/repo/index_diff_stat.txt"
git diff --cached > "$OUT/repo/index.patch"
git ls-files --others --exclude-standard > "$OUT/repo/untracked_files.txt"

for ref in \
  b19_solver_integration \
  codex/b21-3-continuation \
  codex/b21-5-hio-warmstart \
  codex/b21-11-fresh2-final-benchmark \
  codex/b21-12-failure-atlas \
  codex/project-checkpoint-b21-to-b22
 do
  if git rev-parse --verify --quiet "$ref" >/dev/null; then
    printf '%s\t%s\n' "$ref" "$(git rev-parse "$ref")"
  elif git rev-parse --verify --quiet "origin/$ref" >/dev/null; then
    printf '%s\t%s\n' "origin/$ref" "$(git rev-parse "origin/$ref")"
  else
    printf '%s\tMISSING\n' "$ref"
  fi
 done > "$OUT/repo/key_refs.tsv"

if [[ -d "$REPO/external/daps/.git" || -f "$REPO/external/daps/.git" ]]; then
  git -C "$REPO/external/daps" status --short --branch > "$OUT/repo/daps_status.txt" 2>&1 || true
  git -C "$REPO/external/daps" rev-parse HEAD > "$OUT/repo/daps_head.txt" 2>&1 || true
  git -C "$REPO/external/daps" diff --stat > "$OUT/repo/daps_diff_stat.txt" 2>&1 || true
  git -C "$REPO/external/daps" diff > "$OUT/repo/daps_local.patch" 2>&1 || true
else
  echo "external/daps git checkout not found" > "$OUT/repo/daps_status.txt"
fi

if [[ "$CREATE_GIT_BUNDLE" == "1" ]]; then
  section "optional git bundle"
  git bundle create "$OUT/repo/pr_diffusion_all_refs.bundle" --all || \
    echo "[warning] git bundle creation failed"
fi

section "Python and environment state"
if [[ -x "$DAPS_PY" ]]; then
  "$DAPS_PY" -V 2>&1 | tee "$OUT/environment/daps_python_version.txt"
  "$DAPS_PY" -m pip freeze > "$OUT/environment/daps_pip_freeze.txt" 2>&1 || true
  "$DAPS_PY" - <<'PY' > "$OUT/environment/daps_runtime_versions.txt" 2>&1 || true
import platform
print("python", platform.python_version())
for name in ["torch", "torchvision", "numpy", "pandas", "PIL", "scipy"]:
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(name, "IMPORT_ERROR", repr(exc))
try:
    import torch
    print("cuda_available", torch.cuda.is_available())
    print("torch_cuda", torch.version.cuda)
    print("device_count", torch.cuda.device_count())
except Exception as exc:
    print("torch_cuda_probe_error", repr(exc))
PY
else
  echo "[warning] DAPS Python not executable: $DAPS_PY" | tee "$OUT/environment/daps_python_version.txt"
fi

if command -v conda >/dev/null 2>&1; then
  conda env list > "$OUT/environment/conda_env_list.txt" 2>&1 || true
else
  echo "conda command unavailable in current shell" > "$OUT/environment/conda_env_list.txt"
fi

section "artifact inventory"
{
  for path in "$B21_ROOT" "$B21_11" "$B21_12"; do
    if [[ -e "$path" ]]; then
      du -sh "$path" 2>/dev/null || true
    else
      echo "MISSING $path"
    fi
  done

  echo
  echo "B21.11 measurement tensors:"
  find "$B21_11/measurements" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l

  echo "B21.11 case directories:"
  find "$B21_11/cases" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l

  echo "B21.11 sample PNG files:"
  find "$B21_11/cases" -type f -path '*/samples/*.png' 2>/dev/null | wc -l

  echo "B21.11 metric CSV files:"
  find "$B21_11/cases" -type f -path '*/metrics/*.csv' 2>/dev/null | wc -l

  echo "B21.12 case panels:"
  find "$B21_12/cases" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l

  echo "B21.12 group sheets:"
  find "$B21_12/sheets" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l
} | tee "$OUT/artifacts/artifact_inventory.txt"

section "copy small authoritative artifacts"

MANIFEST_DIR="$OUT/artifacts/manifests"
REPORT_DIR="$OUT/artifacts/reports"
SHEET_DIR="$OUT/artifacts/atlas_sheets"

for source in \
  "$B21_11/panel/panel_manifest.tsv" \
  "$B21_11/panel/panel_manifest.sha256" \
  "$B21_11/measurements/fresh_measurement_manifest_meas5401.json" \
  "$B21_11/launch_env.txt"
 do
  copy_if_present "$source" "$MANIFEST_DIR"
 done

for source in \
  "$B21_11/analysis_theta0.7/fresh2_final_summary.json" \
  "$B21_11/analysis_theta0.7/fresh2_final_rows.csv" \
  "$B21_11/analysis_theta0.7/fresh2_selected_psnr_summary.csv" \
  "$B21_11/analysis_theta0.7/fresh2_timing_summary.csv" \
  "$B21_11/analysis_theta0.7/fresh2_selector_discordant_rows.csv" \
  "$B21_12/failure_atlas_summary.json" \
  "$B21_12/failure_atlas_rows.csv" \
  "$B21_12/persistent_failure_rows.csv" \
  "$B21_12/fresh2_rescue_rows.csv" \
  "$B21_12/protected_fresh1_success_rows.csv" \
  "$B21_12/manual_failure_labels_template.csv" \
  "$B21_12/manual_failure_labels_reviewed.csv" \
  "$B21_12/b21_12_failure_atlas.md"
 do
  copy_if_present "$source" "$REPORT_DIR"
 done

for source in \
  "$B21_12/sheets/persistent_failure.png" \
  "$B21_12/sheets/fresh2_rescue.png" \
  "$B21_12/sheets/protected_fresh1_success.png"
 do
  copy_if_present "$source" "$SHEET_DIR"
 done

for source in \
  "$REPO/docs/b21/b21_11_fresh2_final_benchmark.md" \
  "$REPO/docs/b21/b21_12_failure_atlas_decision.md" \
  "$REPO/docs/b21/b21_12_visual_failure_interpretation.md" \
  "$REPO/docs/b21/b21_registry.md" \
  "$REPO/docs/checkpoints/2026-07-27_b21_to_b22/00_START_HERE.md" \
  "$REPO/docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md" \
  "$REPO/docs/checkpoints/2026-07-27_b21_to_b22/02_PAC_INVENTORY_AND_FREEZE.md" \
  "$REPO/docs/checkpoints/2026-07-27_b21_to_b22/03_B22_NEW_CHAT_HANDOVER.md"
 do
  copy_if_present "$source" "$REPORT_DIR"
 done

section "checksums"
find "$OUT" -type f \
  ! -name 'checkpoint_manifest.sha256' \
  ! -name 'checkpoint_stdout.log' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$OUT/checkpoint_manifest.sha256"

sha256sum -c "$OUT/checkpoint_manifest.sha256" >/dev/null && \
  echo "checkpoint checksum verification PASS"

cat > "$OUT/README.md" <<EOF
# B21 → B22 PAC checkpoint

Created: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Repository: $REPO
Repository HEAD: $(git rev-parse HEAD)
Repository branch: $(git branch --show-current)

This directory contains repository state, environment metadata, small authoritative B21.11/B21.12 artifacts, and checksums. Large solver samples and locked measurement tensors remain at their original PAC paths and are inventoried rather than duplicated.

Primary scientific outputs:

- $B21_11
- $B21_12

Verify this checkpoint with:

    sha256sum -c checkpoint_manifest.sha256
EOF

if [[ "$CREATE_ARCHIVE" == "1" ]]; then
  section "checkpoint archive"
  ARCHIVE="${OUT}.tar.gz"
  tar -czf "$ARCHIVE" -C "$(dirname "$OUT")" "$(basename "$OUT")"
  sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
  echo "[archive] $ARCHIVE"
fi

section "checkpoint complete"
echo "OUT=$OUT"
echo "MANIFEST=$OUT/checkpoint_manifest.sha256"
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
  echo "ARCHIVE=${OUT}.tar.gz"
fi

echo "No repository files, solver outputs, or measurements were deleted or modified by this collector."
