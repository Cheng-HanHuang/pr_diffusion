# B22.1 reproducible one-image smoke runbook

## Status and scope

B22.0 is signed off. This checkpoint runs:

- `SITCOM-1`: one official SITCOM trajectory, seed 43;
- `NP-1`: one frozen LF NP trajectory, seed 100.

Both methods consume the same nondiscretionarily selected B21.11 locked
measurement. The script selects the lexicographically first filename among the
100 locked tensors before loading either method.

This runbook does **not** authorize:

- a 100-image panel;
- SITCOM-4S;
- NP-8-RS;
- configuration changes;
- method tuning;
- measurement regeneration.

## Repository branch

```text
codex/b22-fixed-baseline-comparison
```

The dirty B21 PAC checkout must remain untouched. Use a separate Git worktree.

## What the implementation checks

Before starting a GPU process, the CPU preflight verifies:

- the worktree is on the required branch;
- the locked measurement directory contains exactly 100 expected tensors;
- the selected input follows the frozen filename-only rule;
- measurement shape, dtype, and finiteness;
- unique FFHQ ground-truth image resolution;
- SITCOM and DiffFPR commit identities;
- FFHQ model SHA-256;
- the audited SITCOM mkdir-only local patch;
- ground-truth preprocessing and content hashes.

Method processes independently verify the measurement content hash before
execution.

The validator then independently reloads both reconstruction tensors and checks:

- finite `(1,3,256,256)` output;
- exact method config;
- exact measurement, model, and ground-truth identities;
- recomputed raw and rot180 PSNR;
- positive reconstruction time;
- positive peak GPU-memory measurements;
- raw measurement use by SITCOM;
- in-memory clipping by NP.

## 1. Pull into a clean PAC worktree

Run this from the existing repository checkout:

```bash
REPO=/egr/research-pac/huang248/pr_diffusion_b19_solver
WT=/egr/research-pac/huang248/pr_diffusion_b22
BRANCH=codex/b22-fixed-baseline-comparison

cd "$REPO"
git fetch origin

test ! -e "$WT" || {
  echo "STOP: worktree path already exists: $WT"
  exit 1
}

git worktree add \
  --track \
  -b "$BRANCH" \
  "$WT" \
  "origin/$BRANCH"

cd "$WT"
git status --short --branch
git rev-parse HEAD
```

Expected branch:

```text
codex/b22-fixed-baseline-comparison
```

Do not initialize or modify the dirty DAPS submodule for this checkpoint. B22.1
uses the external frozen SITCOM and DiffFPR checkouts plus the existing conda
environments.

## 2. Launch the smoke without terminal flooding

The two methods run sequentially on one visible GPU. Set `GPU` to a free
physical GPU index.

```bash
WT=/egr/research-pac/huang248/pr_diffusion_b22
TMP=/egr/research-pac/huang248/tmp
GPU=0

mkdir -p "$TMP"
cd "$WT"

nohup bash scripts/b22/run_b22_1_smoke.sh "$GPU" \
  > "$TMP/b22_1_smoke_launcher.log" 2>&1 &

echo "launcher_pid=$!"
echo "launcher_log=$TMP/b22_1_smoke_launcher.log"
```

The launcher redirects verbose method output into the timestamped result
directory. Its own log remains compact.

## 3. Check completion

```bash
tail -n 40 /egr/research-pac/huang248/tmp/b22_1_smoke_launcher.log
```

A successful run ends with:

```text
cpu-preflight  OK
sitcom1        OK
np1            OK
validate       OK
validate_ok=1
full_panel_authorized=0
gate_state=B22.1_SMOKE_COMPLETE_PENDING_EXECUTION_LEAD_REVIEW
```

A `FAIL` or `SKIP` is a checkpoint stop. Do not rerun with modified parameters.
Return the produced archive so the execution lead can inspect the error.

## 4. Locate the return archive

```bash
ARCHIVE=$(
  find /egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines \
    -maxdepth 1 \
    -type f \
    -name 'B22_1_smoke_*.tar.gz' \
    -printf '%T@ %p\n' |
  sort -nr |
  head -n 1 |
  cut -d' ' -f2-
)

printf 'archive=%s\n' "$ARCHIVE"
test -f "$ARCHIVE"
```

Attach that single `.tar.gz` file to the execution-lead chat.

Do not paste the full method logs into the terminal or chat.

## Returned archive contents

The archive contains:

```text
input/input_manifest.json
input/ground_truth.pt
input/ground_truth.png
sitcom1/result.json
sitcom1/reconstruction.pt
sitcom1/reconstruction.png
np1/result.json
np1/reconstruction.pt
np1/reconstruction.png
comparison.csv
validation.json
FINAL_STATUS.txt
RETURN_MANIFEST.txt
status.tsv
logs/
```

The archive contains no model checkpoint and no locked measurement copy.

## Review gate

After the archive is returned, the execution lead will decide one of:

```text
B22.1 SIGNED OFF
B22.1 REPEAT REQUIRED because of an implementation/runtime defect
B22 STOP because of a source/measurement/method incompatibility
```

The full 100-image runner will be implemented and authorized only after explicit
B22.1 sign-off.
