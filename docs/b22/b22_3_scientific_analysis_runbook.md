# B22.3 CPU scientific analysis and failure atlas

## Status

B22.2 execution is complete and independently validated. B22.3 performs no GPU
work and does not alter any frozen policy. It converts the paired 100-image
panel into:

- robust quality/reliability/cost summaries;
- deterministic image-level bootstrap intervals;
- paired policy comparisons;
- selector-versus-oracle diagnostics;
- candidate-arm summaries;
- a 16-image failure/complementarity union;
- individual visual panels and grouped contact sheets.

The analysis is descriptive. It must not be used to tune a cross-method selector
on this 100-image estimation panel.

## Current preliminary result

The validated compact archive already establishes:

| Executable policy | Mean raw PSNR | Median | q05 | raw good25 | raw bad20 | Mean GPU-s/image |
|---|---:|---:|---:|---:|---:|---:|
| Fresh1 | 27.232 | 30.605 | 9.614 | 80/100 | 18 | 156.1 |
| Fresh2 | 29.299 | 30.899 | 15.860 | 92/100 | 8 | 312.6 |
| SITCOM-1 | 22.972 | 27.038 | 7.791 | 71/100 | 25 | 49.3 |
| SITCOM-4S | 26.442 | 27.166 | 23.415 | 93/100 | 4 | 196.1 |
| NP-1 | 25.585 | 29.390 | 10.435 | 75/100 | 24 | 52.7 |
| NP-8-RS | 29.269 | 29.941 | 25.146 | 95/100 | 3 | 411.8 |

The best-of-Fresh2/SITCOM-4S/NP-8-RS diagnostic oracle is good25 on 99/100
images. Image `65003` is the only raw failure shared by all three.

## 1. Pull the analysis implementation

```bash
WT=/egr/research-pac/huang248/pr_diffusion_b22
cd "$WT"
git fetch origin
git pull --ff-only origin codex/b22-fixed-baseline-comparison

echo "branch=$(git branch --show-current)"
echo "head=$(git rev-parse HEAD)"
git status --short --branch
```

Use the exact head stated by the execution lead.

## 2. Run the CPU-only analysis

```bash
WT=/egr/research-pac/huang248/pr_diffusion_b22
RUN=/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines/B22_2_overnight_20260727_155135
TMP=/egr/research-pac/huang248/tmp
LOG=$TMP/b22_3_scientific_analysis.log

mkdir -p "$TMP"
cd "$WT"

nohup bash scripts/b22/run_b22_3_analysis.sh "$RUN" \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "log=$LOG"
```

No GPU is used. The existing selected PNGs and ground-truth images are read only.

## 3. Check completion

```bash
tail -n 60 /egr/research-pac/huang248/tmp/b22_3_scientific_analysis.log
```

Expected ending:

```text
[OK  ] scientific-analysis
[OK  ] failure-atlas
[OK  ] archive
B22.3 analysis completed.
return_archive=...B22_3_scientific_analysis_....tar.gz
```

## 4. Return the archive

```bash
ARCHIVE=$(find /egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines \
  -maxdepth 1 -type f -name 'B22_3_scientific_analysis_*.tar.gz' \
  -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)

printf 'archive=%s\n' "$ARCHIVE"
test -f "$ARCHIVE"
```

Attach that single archive. It contains the Markdown scientific report,
machine-readable tables, 16 individual panels, grouped contact sheets, and an
atlas index.

## Visual review priorities

1. `65003`: shared failure of Fresh2, SITCOM-4S, and NP-8-RS.
2. Fresh2 failures rescued by SITCOM/NP: `62908`, `65365`, `65553`, `66715`,
   `66889`, `67293`, and `68539`.
3. NP-8-RS failures where Fresh2 succeeds: `61252`, `61669`, `65269`, and
   `67520`.
4. SITCOM selector miss `60140`, plus threshold miss `64518`.
5. NP selector miss `65269`.

## Gate after return

```text
B22.2 fixed panel: VALIDATED
B22.3 numerical analysis: READY
B22.3 visual atlas: READY
New GPU experiments: NOT AUTHORIZED
Cross-method selector tuning on this panel: PROHIBITED
```
