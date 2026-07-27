# B22.2 overnight full-panel implementation and runbook

## Status

B22.1 is signed off. This package is designed for a late-night launch:

1. CPU-only identity, manifest, and sharding preparation;
2. strict four-GPU full-policy smoke on two nondiscretionary panel rows;
3. exact replay of the signed-off B22.1 SITCOM-1 and NP-1 hashes on image 60044;
4. independent smoke validation;
5. automatic 100-image launch only after the smoke passes;
6. independent full-panel validation and compact return archive.

The launcher does not tune any policy and never regenerates a measurement.

## Frozen rows

Executable rows:

- Fresh1: reused from B21.11;
- Fresh2: reused from B21.11;
- SITCOM-1: candidate 0 of the four-candidate SITCOM population;
- SITCOM-4S: minimum correction norm at annealing ratio `tau=0.8`;
- NP-1: LF, seed 100;
- NP-8-RS: minimum post-projection winner LF-MSE across LF/S2 and seeds 100--103.

Diagnostic-only rows:

- SITCOM-oracle4;
- NP-oracle8.

The historical SITCOM population is frozen as one master seed `43` set before
model construction, followed by four sequential trajectories from that RNG
stream. It is not four separately reseeded runs. Candidate 0 is therefore the
exact SITCOM-1 trajectory used by B22.1. The 4S selector reads

```text
sqrt(mean((x0y - x0hat)^2))
```

at annealing step 160 of 200 (`tau=0.8`). Stable ties choose the lower candidate
index.

## Compute plan

Four physical GPUs are used concurrently:

| GPU role | method | full rows |
|---|---|---:|
| first SITCOM GPU | SITCOM sequential candidates 0--3 | 50 |
| second SITCOM GPU | SITCOM sequential candidates 0--3 | 50 |
| first NP GPU | LF/S2 x seeds 100--103 | 50 |
| second NP GPU | LF/S2 x seeds 100--103 | 50 |

SITCOM-1 and NP-1 are obtained from their containing populations, so they do not
cost duplicate trajectories.

Based on B22.1 timings, expected wall time is approximately:

- full-policy smoke: 7--15 minutes;
- full panel after smoke: roughly 5--7 hours;
- total: roughly 5.2--7.3 hours.

These are planning estimates, not result fields.

## 1. Pull the implementation

```bash
WT=/egr/research-pac/huang248/pr_diffusion_b22
cd "$WT"
git fetch origin
git pull --ff-only origin codex/b22-fixed-baseline-comparison

echo "branch=$(git branch --show-current)"
echo "head=$(git rev-parse HEAD)"
git status --short --branch
```

Use only the head recorded in the execution-lead message accompanying this
runbook.

## 2. Check the four GPUs

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv
```

The default assignment is:

```text
SITCOM: GPUs 0,1
NP: GPUs 2,3
```

Change only the four physical GPU indices passed to the launcher. Do not change
method parameters. The launcher refuses duplicate GPU indices.

## 3. Start the overnight gate-and-launch workflow

```bash
WT=/egr/research-pac/huang248/pr_diffusion_b22
TMP=/egr/research-pac/huang248/tmp
LOG=$TMP/b22_2_overnight_launcher.log

mkdir -p "$TMP"
cd "$WT"

nohup bash scripts/b22/launch_b22_2_overnight.sh 0 1 2 3 \
  > "$LOG" 2>&1 &

echo "launcher_pid=$!"
echo "launcher_log=$LOG"
```

The four numbers are physical GPU indices in this order:

```text
SITCOM shard 0, SITCOM shard 1, NP shard 0, NP shard 1
```

The shell may be closed after the launcher has printed the smoke launch status.
The process is protected by `nohup`.

## 4. Verify that the smoke passed and the full run started

The launcher log is compact:

```bash
tail -n 60 /egr/research-pac/huang248/tmp/b22_2_overnight_launcher.log
```

Before sleeping, the required transition is:

```text
[PASS] smoke-gate — all full-policy smoke checks passed; starting 100-image run
[OK  ] full-prepare
[OK  ] full-launch
```

The smoke validator also requires exact reconstruction-content replay of the
signed-off B22.1 SITCOM-1 and NP-1 outputs on image 60044. If either hash differs,
the full workers do not launch.

If the log instead contains `smoke-gate FAIL`, no full worker is launched. Do
not manually bypass the gate.

To see the run root and process IDs:

```bash
RUN_ROOT=$(grep '^run_root=' /egr/research-pac/huang248/tmp/b22_2_overnight_launcher.log \
  | tail -n 1 | cut -d= -f2-)

printf 'run_root=%s\n' "$RUN_ROOT"
cat "$RUN_ROOT/full_worker_pids.txt" 2>/dev/null || true
```

The launch metadata also records GPU state and the exact repository head:

```bash
cat "$RUN_ROOT/launch_metadata.txt"
```

## 5. Morning status

```bash
tail -n 80 /egr/research-pac/huang248/tmp/b22_2_overnight_launcher.log
```

For per-worker progress without terminal flooding:

```bash
RUN_ROOT=$(grep '^run_root=' /egr/research-pac/huang248/tmp/b22_2_overnight_launcher.log \
  | tail -n 1 | cut -d= -f2-)

for f in "$RUN_ROOT"/logs/full_*_shard*.log; do
  echo "===== $(basename "$f") ====="
  tail -n 3 "$f"
done
```

Successful completion ends with:

```text
[PASS] full-gate — 100-image paired baseline artifacts complete; scientific review pending
B22.2 overnight run completed.
return_archive=..._complete.tar.gz
```

## 6. Return artifacts

Locate the compact archive:

```bash
ARCHIVE=$(find /egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines \
  -maxdepth 1 -type f -name 'B22_2_overnight_*_complete.tar.gz' \
  -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)

printf 'archive=%s\n' "$ARCHIVE"
test -f "$ARCHIVE"
```

Attach that archive. It excludes reconstruction tensors and PNGs but includes:

- frozen config snapshot;
- exact 100-row manifest and hashes;
- shard plans and worker summaries;
- per-candidate selector/result JSON;
- paired policy rows;
- summary statistics;
- failure rows;
- logs and validation records.

The large reconstruction artifacts remain in the PAC run root for later visual
inspection.

## Safety gates

- four distinct GPU indices are required;
- full execution starts only after machine validation of the full-policy smoke;
- the B22.1 SITCOM-1 and NP-1 reconstruction hashes must replay exactly;
- SITCOM uses the frozen master-seed-43 sequential four-trajectory population;
- no output directory is silently overwritten;
- valid completed candidates are resumable;
- partial/invalid candidate directories stop rather than being replaced;
- Fresh1/Fresh2 are read from completed B21.11 outputs and are not rerun;
- raw PSNR remains primary;
- rot180-aware PSNR is auxiliary and ground-truth-assisted;
- oracle rows are diagnostic only;
- full scientific interpretation remains blocked until the compact archive is
  returned and reviewed.
