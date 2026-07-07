# B21.2 replay blocker and rerun decision

Status: blocker recorded after strict candidate-path search.

## What was checked

B21.2 requires final candidate images so that selector-v2 can compute the clean-free prior-consistency probe for both `x` and `rot180(x)`. The current B19.20 replay CSVs have no usable `sample_path` rows, so we attempted filesystem recovery.

Searches performed:

1. CSV-level path discovery: `scripts/b21/run_b21_2_candidate_path_discovery.sh`.
2. Broad PNG locator: `scripts/b21/run_b21_2_b19_20_candidate_png_locator.sh` before strict filtering.
3. Final strict PNG locator: same wrapper after removing `runseed4400` as a sufficient match token. Strict rule: sample PNG only and path contains `B19_20`, `b19_20`, or `ffhq100`.

## Strict locator result

```text
Total matches: 0
Sample PNG matches: 0
Root /external/daps/results scanned images: 39495, matches: 0
Root /outputs/pr_diffusion/b19_solver scanned images: 1439, matches: 0
Rejected because missing B19_20/b19_20/ffhq100 token: 13365
Rejected non-target image: 27569
```

Conclusion: **B19.20 candidate sample PNGs are not recoverable from the current result tree by path search.** Earlier apparent matches were false positives caused by `runseed4400` being shared by B19.16/B20 outputs.

## Consequence for B21.2

B21.2 cannot complete the planned B19.20 replay validation from existing artifacts. The B19.16/B20/B20.12 sample paths remain usable for development/replay, but the B19.20 fresh validation panel needs newly generated candidate sample files with explicit sample-path logging.

This does not change G0: B19.20 is already reclassified as an FFHQ100 image-level diagnostic, not n=1000 independent image-measurement cases.

## Rerun decision

Full FFHQ100 trajectory rerun remains **B22 scope** per the B21.0 runbook and master plan. For B21.2, the allowed next step is a **small audited candidate-recovery rerun**, not a full FFHQ100 revalidation. The rerun must:

1. use locked measurement payload paths and log their SHA/statistics at run time;
2. use explicit `NUM_RUNS=1` / one process per run seed if seed matching matters;
3. save final sample paths in a machine-readable CSV;
4. cover only the images needed to unblock selector-v2 development/validation;
5. write a manifest before launch and an analysis report after launch.

Candidate minimal panel:

```text
final-exact/symmetry images: 00136, 00154, 00253, 00480, 00971
measurement: meas5001 first, with optional meas7001+ only after measurement-generation sanity is complete
runs: 6 raw candidates per image initially
schedule: existing B19/B20 raw6S-compatible schedule unless explicitly changed in a new runbook patch
```

Do not launch a full P-FFHQ100 rerun under B21.
