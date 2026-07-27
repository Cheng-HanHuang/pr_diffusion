# Dirty-source snapshot amendment

The first PAC checkpoint correctly preserved scientific reports, manifests, checksums, the outer-repo worktree patch, and the DAPS local patch. Its state audit also revealed a large dirty checkout:

- one modified tracked outer-repo analyzer;
- a dirty DAPS submodule with two modified tracked Python files;
- many untracked outer-repo research scripts and reports;
- generated B21.11 DAPS YAML configurations.

The main collector records untracked filenames but does not copy their contents. Therefore a thorough project freeze also requires the companion source snapshot:

```text
scripts/checkpoints/collect_b21_source_snapshot.sh
```

This collector is non-destructive. It preserves:

1. remote branch heads using `git ls-remote`, avoiding stale local branch refs;
2. untracked files under `docs/`, `scripts/`, and `patches/`, excluding Aider histories/caches;
3. exact contents of modified tracked outer-repo files;
4. the DAPS HEAD, status, patch, and exact modified Python files;
5. the 100 generated `b21-fresh2-final-ffhq-*.yaml` files;
6. checksums and an optional compressed archive.

## Run

```bash
cd /egr/research-pac/huang248/pr_diffusion_b19_solver

git fetch origin codex/project-checkpoint-b21-to-b22
git switch codex/project-checkpoint-b21-to-b22
git pull --ff-only origin codex/project-checkpoint-b21-to-b22

bash -n scripts/checkpoints/collect_b21_source_snapshot.sh

REPO=/egr/research-pac/huang248/pr_diffusion_b19_solver
CHECKPOINT_ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$CHECKPOINT_ROOT/B21_source_snapshot_$STAMP"
LOG=/tmp/b21_source_snapshot_$STAMP.log

set -o pipefail

env \
  REPO="$REPO" \
  CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
  STAMP="$STAMP" \
  OUT="$OUT" \
  CREATE_ARCHIVE=1 \
  bash scripts/checkpoints/collect_b21_source_snapshot.sh \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
echo "source snapshot exit code: $STATUS"
echo "source snapshot output: $OUT"
echo "source snapshot log: $LOG"
```

Do not append `exit "$STATUS"` in an interactive terminal.

## Verify

```bash
cd "$OUT"
sha256sum -c source_snapshot_manifest.sha256
cat source_snapshot_metadata.txt
cat source_snapshot_inventory.txt
cat repo/remote_key_refs.tsv
cat repo/modified_tracked_files.txt
cat daps/diff_stat.txt
wc -l repo/untracked_source_files.txt
find repo/untracked_source_snapshot -type f | wc -l
find daps/final_configs -maxdepth 1 -type f -name '*.yaml' | wc -l
ls -lh "${OUT}.tar.gz" "${OUT}.tar.gz.sha256"
```

Expected from the first audit:

- approximately 70 untracked research source/document files copied;
- one modified tracked outer-repo file copied;
- two modified DAPS Python files copied;
- exactly 100 B21.11 final DAPS YAML files copied;
- checksum verification passes.

The original PAC checkpoint and this source snapshot form one logical B21 freeze. Preserve both paths in `HUMAN_SIGNOFF.md` and provide both to the B22 chat.
