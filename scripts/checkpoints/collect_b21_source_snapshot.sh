#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$CHECKPOINT_ROOT/B21_source_snapshot_$STAMP}
DAPS=$REPO/external/daps
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}

mkdir -p \
  "$OUT/repo/untracked_source_snapshot" \
  "$OUT/repo/modified_file_snapshot" \
  "$OUT/daps/modified_file_snapshot" \
  "$OUT/daps/final_configs"

exec > >(tee "$OUT/source_snapshot_stdout.log") 2>&1

if [[ ! -d "$REPO/.git" ]]; then
  echo "[fatal] repository is not a git checkout: $REPO" >&2
  exit 2
fi

cd "$REPO"

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REPO"
  echo "repo_head=$(git rev-parse HEAD)"
  echo "repo_branch=$(git branch --show-current)"
  echo "daps=$DAPS"
  echo "create_archive=$CREATE_ARCHIVE"
} | tee "$OUT/source_snapshot_metadata.txt"

# Record remote branch heads without modifying local refs.
: > "$OUT/repo/remote_key_refs.tsv"
for ref in \
  b19_solver_integration \
  codex/b21-3-continuation \
  codex/b21-5-hio-warmstart \
  codex/b21-11-fresh2-final-benchmark \
  codex/b21-12-failure-atlas \
  codex/project-checkpoint-b21-to-b22
 do
  sha=$(git ls-remote --heads origin "refs/heads/$ref" | awk 'NR==1 {print $1}')
  if [[ -n "$sha" ]]; then
    printf '%s\t%s\n' "$ref" "$sha"
  else
    printf '%s\tMISSING\n' "$ref"
  fi
 done | tee "$OUT/repo/remote_key_refs.tsv"

# Preserve untracked research source/docs/patches, but deliberately exclude
# local Aider histories/caches and generated solver outputs.
git ls-files --others --exclude-standard \
  | awk '/^(docs|scripts|patches)\//' \
  | sort > "$OUT/repo/untracked_source_files.txt"

while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  if [[ -f "$REPO/$path" ]]; then
    mkdir -p "$OUT/repo/untracked_source_snapshot/$(dirname "$path")"
    cp -p "$REPO/$path" "$OUT/repo/untracked_source_snapshot/$path"
  fi
done < "$OUT/repo/untracked_source_files.txt"

# Preserve exact contents of modified tracked outer-repo files in addition to
# the patch stored by the main checkpoint collector.
git diff --name-only | sort > "$OUT/repo/modified_tracked_files.txt"
while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  [[ "$path" == external/daps ]] && continue
  if [[ -f "$REPO/$path" ]]; then
    mkdir -p "$OUT/repo/modified_file_snapshot/$(dirname "$path")"
    cp -p "$REPO/$path" "$OUT/repo/modified_file_snapshot/$path"
  fi
done < "$OUT/repo/modified_tracked_files.txt"

if [[ -d "$DAPS/.git" || -f "$DAPS/.git" ]]; then
  git -C "$DAPS" rev-parse HEAD > "$OUT/daps/head.txt"
  git -C "$DAPS" status --short --branch > "$OUT/daps/status.txt"
  git -C "$DAPS" diff > "$OUT/daps/local.patch"
  git -C "$DAPS" diff --stat > "$OUT/daps/diff_stat.txt"
  git -C "$DAPS" diff --name-only | sort > "$OUT/daps/modified_tracked_files.txt"

  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if [[ -f "$DAPS/$path" ]]; then
      mkdir -p "$OUT/daps/modified_file_snapshot/$(dirname "$path")"
      cp -p "$DAPS/$path" "$OUT/daps/modified_file_snapshot/$path"
    fi
  done < "$OUT/daps/modified_tracked_files.txt"

  find "$DAPS/configs/data" -maxdepth 1 -type f \
    -name 'b21-fresh2-final-ffhq-*.yaml' \
    -print | sort > "$OUT/daps/final_config_paths.txt"

  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    cp -p "$path" "$OUT/daps/final_configs/"
  done < "$OUT/daps/final_config_paths.txt"
else
  echo "[fatal] DAPS git checkout not found: $DAPS" >&2
  exit 3
fi

{
  echo "outer untracked research files copied:"
  wc -l < "$OUT/repo/untracked_source_files.txt"
  echo "outer modified tracked files copied:"
  find "$OUT/repo/modified_file_snapshot" -type f | wc -l
  echo "DAPS modified tracked files copied:"
  find "$OUT/daps/modified_file_snapshot" -type f | wc -l
  echo "B21.11 final DAPS configs copied:"
  find "$OUT/daps/final_configs" -maxdepth 1 -type f -name '*.yaml' | wc -l
} | tee "$OUT/source_snapshot_inventory.txt"

find "$OUT" -type f \
  ! -name 'source_snapshot_manifest.sha256' \
  ! -name 'source_snapshot_stdout.log' \
  -print0 | sort -z | xargs -0 sha256sum > "$OUT/source_snapshot_manifest.sha256"

sha256sum -c "$OUT/source_snapshot_manifest.sha256" >/dev/null
echo "source snapshot checksum verification PASS"

if [[ "$CREATE_ARCHIVE" == "1" ]]; then
  ARCHIVE="${OUT}.tar.gz"
  tar -czf "$ARCHIVE" -C "$(dirname "$OUT")" "$(basename "$OUT")"
  sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
  echo "[archive] $ARCHIVE"
fi

echo "OUT=$OUT"
echo "No repository or solver file was modified or deleted."
