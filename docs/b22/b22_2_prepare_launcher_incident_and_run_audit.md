# B22.2 prepare/launcher incident and completed-run audit

## Decision

The completed run

```text
B22_2_overnight_20260727_155135
```

is **not invalidated** by the two misleading preparation failures.

The scientific outputs and compact records are complete and internally consistent.
The preparation process failed only after it had written the manifest, config snapshot,
and all four shard files. The missing output was the auxiliary `plan.json` and final
success print. The workers consumed the already complete shard files.

The full scientific interpretation remains separate from this execution-integrity
decision.

## Observed status lines

```text
[FAIL] smoke-prepare — smoke_prepare.log rc=0
...
[PASS] smoke-gate — all full-policy smoke checks passed; starting 100-image run
[FAIL] full-prepare — full_prepare.log rc=0
...
[PASS] full-gate — 100-image paired baseline artifacts complete
```

The `rc=0` text was itself a shell-reporting bug; the Python prepare commands actually
raised exceptions.

## Root causes

### Late preparation exception

Both prepare logs ended with:

```text
KeyError: 'seeds'
```

The stale plan-summary expression referenced:

```python
config["sitcom"]["seeds"]
```

After SITCOM population semantics were corrected to one master seed followed by four
sequential trajectories, the frozen config uses:

```python
config["sitcom"]["trajectory_count"]
```

The exception occurred after these files had already been written:

- `manifest.json`;
- `config_snapshot.json`;
- `sitcom_shard0.json` and `sitcom_shard1.json`;
- `np_shard0.json` and `np_shard1.json`.

It occurred while constructing the auxiliary `plan.json`.

### Shell return-code propagation bug

The old `run_logged` and `wait_worker` helpers captured `$?` after an `if ... fi`
compound command. In the failure path this yielded zero, causing a failed command to be
recorded as `FAIL ... rc=0` but returned as success to the caller.

This was a real fail-fast defect. It did not alter candidate computation in this run,
but it could have allowed a more serious earlier failure to continue until later
validation.

## Independent compact-archive audit

The returned archive was unpacked and recounted independently of `status.tsv`.

### Preparation artifacts

- full manifest rows: `100`;
- unique row IDs: `100`;
- unique image IDs: `100`;
- unique measurement file hashes: `100`;
- unique measurement tensor-content hashes: `100`;
- SITCOM shard sizes: `50 + 50`;
- NP shard sizes: `50 + 50`;
- each method's shard union equals exactly rows `0..99`;
- no duplicate or omitted shard row.

### Candidate artifacts

- SITCOM candidate result JSONs: `400 = 100 x 4`;
- NP candidate result JSONs: `800 = 100 x 8`;
- all `1,200` candidate records report `status=PASS` and finite output;
- all candidate row/image, measurement, and ground-truth identities match the manifest;
- all candidate reconstruction hashes are present;
- all candidate raw/rot180/ambiguity-aware metrics are finite;
- all candidate reconstruction times are positive;
- all candidate peak allocated and reserved GPU-memory records are positive.

### Policy selection audit

Selections were recomputed from candidate JSONs:

- `SITCOM-1` equals sequential candidate index `0`;
- `SITCOM-4S` equals minimum `correction_norm`, with candidate-index tie break;
- `SITCOM-oracle4` equals maximum raw PSNR;
- `NP-1` equals LF / seed `100`;
- `NP-8-RS` equals minimum `selector_post_winner_lf_mse_mean`, with frozen tie break;
- `NP-oracle8` equals maximum raw PSNR.

All `600` generated-method selections matched their recorded reconstruction hashes.

### Paired panel

- paired policy rows: `800 = 100 x 8`;
- exactly `100` rows for each of Fresh1, Fresh2, SITCOM-1, SITCOM-4S, NP-1,
  NP-8-RS, SITCOM-oracle4, and NP-oracle8;
- no duplicate `(row_id, policy)` pair;
- Fresh1 and Fresh2 counts reproduce the frozen B21.11 values `80/100` and `92/100`;
- full validator status: `PASS`;
- full validator reports all candidate metric/hash/runtime/memory, selector,
  measurement, ground-truth, shard-completeness, output-hash, and no-omission checks true.

### Smoke replay

The smoke includes image `60044` and exactly reproduces the signed-off B22.1 hashes:

```text
SITCOM-1:
14ed2c4e209d2f00601f73cfd35b87c7db1365a57065c41a3c3ab162cae429d2

NP-1:
b6dee35138e453fc8a1c77aa1dc3331d1b70bce8fa75a10d66fc39d5fd836641
```

## Integrity conclusion

```text
Preparation success reporting: INVALID / PATCHED
Original launcher fail-fast semantics: INVALID / PATCHED
Completed manifests and shards: VALID
Completed 1,200 candidate records: VALID
Completed policy selections: VALID
Completed 800-row paired panel: VALID
Smoke exact replay: VALID
Full execution-integrity gate: PASS
Scientific result interpretation: PENDING SEPARATE ANALYSIS
```

No GPU rerun is required for this incident.

## Corrective implementation

The branch now:

1. uses `trajectory_count` for the expected SITCOM candidate count;
2. builds preparation outputs in a hidden temporary directory and atomically renames
   it only after manifest, config, four shards, and `plan.json` are complete;
3. captures command and worker exit codes directly rather than after an `if` compound;
4. requires nonempty manifest, config, plan, and all four shard files before GPU launch;
5. fixes compact-archive success propagation.

These corrections apply to future runs. The existing completed run remains preserved
under its original repository head and PAC run root.
