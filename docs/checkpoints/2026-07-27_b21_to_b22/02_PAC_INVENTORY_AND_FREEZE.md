# PAC inventory and freeze runbook

This runbook captures the local state that GitHub cannot preserve: the exact checkout, dirty files, DAPS submodule diff, environment versions, output inventory, generated reports, manual labels, and checksums.

The collector is non-destructive. It does not delete, reset, move, or rewrite solver artifacts.

## 1. Pull the checkpoint branch

```bash
export TERM=xterm-256color
stty sane 2>/dev/null || true
hash -r

REPO=/egr/research-pac/huang248/pr_diffusion_b19_solver
cd "$REPO"

git fetch origin codex/project-checkpoint-b21-to-b22
git switch codex/project-checkpoint-b21-to-b22
git pull --ff-only origin codex/project-checkpoint-b21-to-b22

git branch --show-current
git rev-parse HEAD
git status --short
```

Do not use `git clean`, `git reset`, forced checkout, rebase, or forced submodule update to make the checkout look clean.

## 2. Preflight the collector

```bash
cd /egr/research-pac/huang248/pr_diffusion_b19_solver

bash -n scripts/checkpoints/collect_b21_project_checkpoint.sh

test -s docs/checkpoints/2026-07-27_b21_to_b22/00_START_HERE.md
test -s docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md
test -s docs/checkpoints/2026-07-27_b21_to_b22/03_B22_NEW_CHAT_HANDOVER.md

echo "checkpoint collector preflight PASS"
```

## 3. Confirm the manually reviewed atlas CSV

The collector prefers the reviewed CSV but records the template if the reviewed file is absent.

```bash
B21_11=/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401
B21_12="$B21_11/b21_12_failure_atlas"

ls -lh \
  "$B21_12/manual_failure_labels_template.csv" \
  "$B21_12/manual_failure_labels_reviewed.csv"

column -s, -t < "$B21_12/manual_failure_labels_reviewed.csv"
```

If the reviewed file is missing, complete the visual labels before treating the checkpoint as final.

## 4. Run the collector

```bash
cd /egr/research-pac/huang248/pr_diffusion_b19_solver

REPO=/egr/research-pac/huang248/pr_diffusion_b19_solver
DAPS_PY=/egr/research-pac/huang248/conda-envs/daps/bin/python
CHECKPOINT_ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$CHECKPOINT_ROOT/B21_to_B22_$STAMP"
LOG=/tmp/b21_to_b22_checkpoint_$STAMP.log

set -o pipefail

env \
  REPO="$REPO" \
  DAPS_PY="$DAPS_PY" \
  CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
  STAMP="$STAMP" \
  OUT="$OUT" \
  CREATE_GIT_BUNDLE=0 \
  CREATE_ARCHIVE=1 \
  bash scripts/checkpoints/collect_b21_project_checkpoint.sh \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
echo "checkpoint exit code: $STATUS"
echo "checkpoint output: $OUT"
echo "checkpoint log: $LOG"
```

Do not append `exit "$STATUS"` in an interactive VS Code terminal.

## 5. Verify the checkpoint

```bash
cd "$OUT"

sha256sum -c checkpoint_manifest.sha256

cat checkpoint_metadata.txt
cat repo/key_refs.tsv
cat repo/status_short_branch.txt
cat repo/submodule_status.txt
cat repo/daps_status.txt
cat artifacts/artifact_inventory.txt

find artifacts -maxdepth 3 -type f -printf '%P\n' | sort
```

Required minimum:

- repository HEAD and branch recorded;
- all key B21 refs resolved or explicitly marked missing;
- worktree/index patches and untracked-file inventory written;
- DAPS submodule HEAD/status/local patch recorded;
- B21.11 reports and manifests copied;
- B21.12 summary, rows, reviewed labels, and three sheets copied;
- checkpoint checksum verification passes;
- archive and archive checksum exist when `CREATE_ARCHIVE=1`.

## 6. Optional Git bundle

GitHub already preserves the remote branches, so a bundle is optional. For an additional offline copy:

```bash
cd /egr/research-pac/huang248/pr_diffusion_b19_solver

OUT=/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints/B21_to_B22_<STAMP>

git bundle create "$OUT/repo/pr_diffusion_all_refs.bundle" --all

git bundle verify "$OUT/repo/pr_diffusion_all_refs.bundle"
sha256sum "$OUT/repo/pr_diffusion_all_refs.bundle" \
  > "$OUT/repo/pr_diffusion_all_refs.bundle.sha256"
```

Replace `<STAMP>` with the actual completed checkpoint directory.

## 7. Human checkpoint note

After inspection, add a small text file inside the PAC checkpoint directory:

```bash
cat > "$OUT/HUMAN_SIGNOFF.md" <<'EOF'
# Human checkpoint sign-off

- B21.11 final benchmark artifacts inspected: yes/no
- B21.12 failure atlas inspected: yes/no
- Reviewed manual labels present: yes/no
- Repository dirty state understood and preserved: yes/no
- DAPS submodule/local patch understood and preserved: yes/no
- Open PR stack reviewed: yes/no
- Ready to start B22 in a new chat: yes/no

Notes:

EOF
```

Then refresh the local checksum file:

```bash
find "$OUT" -type f \
  ! -name 'checkpoint_manifest.sha256' \
  ! -name 'checkpoint_stdout.log' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$OUT/checkpoint_manifest.sha256"

cd "$OUT"
sha256sum -c checkpoint_manifest.sha256
```

If an archive was already created before sign-off, recreate it after the sign-off file is added.

## 8. What not to archive into Git

Do not commit:

- measurement tensors;
- solver sample PNG populations;
- model checkpoints;
- PAC environment directories;
- large raw output trees;
- private machine/environment dumps containing secrets.

Git should contain the code, small reports, plans, and artifact paths. PAC checkpoint storage should contain the local inventory and small copied artifacts.
