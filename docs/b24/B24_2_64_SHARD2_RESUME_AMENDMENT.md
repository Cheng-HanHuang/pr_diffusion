# B24.2 64-image shard-2 resume amendment

## Status before amendment

Accepted run root: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_2_64_20260826T040303Z`.

- shard 0: 16/16 complete, PASS;
- shard 1: 16/16 complete, PASS;
- shard 3: 16/16 complete, PASS;
- shard 2: rows 2, 6, 10, 14, 18 complete; row 22 (`00894`) stopped during locked-input generation before DAPS/SITCOM execution for that row.

Thus 53 image completions are preserved; no completed image is authorized for rerun.

## Root cause

`generate_b24_locked_input.py::find_image` formed three candidate FFHQ source paths. For IDs below 1000, the computed thousand-folder is already `00000`, so the computed path and explicit `00000` fallback were identical. The same existing source therefore appeared twice in the `hits` list and falsely violated the exactly-one-source cardinality check.

For row 22, the actual source remains `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024/00000/00894.png`. The manifest-frozen image ID, measurement seed, and all eight solver seeds remain unchanged.

## Authorized correction

1. Deduplicate equal candidate paths before the source-cardinality check.
2. Add a regression test for sub-1000 ID `00894`.
3. Resume only shard 2 in the existing run root.
4. Reuse an image only when its atomic `IMAGE_COMPLETE.json` matches the frozen manifest file SHA, stage, shard/GPU, row/image identity, measurement seed, and both solver-seed vectors.
5. Preserve any incomplete row directory under `shard2/partial_attempts/` before retrying that row.
6. Preserve the original GPU-2 log/PID as pre-resume evidence.
7. Do not rerun or modify shards 0, 1, or 3.

The correction changes no scientific protocol, method configuration, measurement/solver seed, Good25 threshold, metric representation, best-of-four rule, or class definition.

## Recovery entrypoint

`scripts/b24/resume_b24_2_64_shard2.sh`

The recovery launcher is fail-closed on the existing run root, byte-frozen 64-image manifest, external source identities, complete shards 0/1/3, dead old shard-2 PID, completed shard-2 row identities, and fixed GPU-2 UUID/free-memory gate.
