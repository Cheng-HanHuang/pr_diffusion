# B21.10 normalized-residual threshold transfer

Status: zero-GPU audit ready; no fallback GPU pilot authorized yet.

## Why the raw threshold failed

The raw exact-loss ranking on B21.9 is perfect for Fresh2 failure, but the threshold learned on B21.8 flags `46/80` rows on both panels. It transfers with full recall but poor precision:

```text
development precision: 0.217
validation precision:  0.130
validation flagged rate: 0.575
```

The failure is consistent with raw operator loss having a measurement-dependent scale. An absolute threshold across different measurements is therefore too permissive even when within-panel ranking is excellent.

## Follow-up audit

Use the already recorded metric field

```text
sqrt_loss_over_y_norm
```

for the Fresh2-selected candidate. This is a clean-free scale-normalized residual. Within one measurement it is monotone in exact loss, so it does not change candidate selection; it is used only as a cross-measurement hard-case detector.

Derive the threshold on B21.8 only:

```text
threshold = minimum selected normalized residual among B21.8 Fresh2 failures
```

Apply it unchanged to B21.9.

## Retrospective support rule

A strong transfer signal requires:

```text
validation recall >= 0.80
validation precision >= 0.50
validation flagged fraction <= 0.25
```

A pass only authorizes a small targeted complementary-candidate development pilot. It does not validate the detector or an adaptive policy. A failure ends the current absolute-threshold route and requires a different clean-free detector before GPU work resumes.

Always create the output directory before piping through `tee`:

```bash
mkdir -p "$OUT"
python ... 2>&1 | tee "$OUT/analyzer_stdout.txt"
```

Otherwise `tee` exits nonzero before the Python analyzer creates the directory, even though the analyzer may finish and write its own artifacts successfully.
