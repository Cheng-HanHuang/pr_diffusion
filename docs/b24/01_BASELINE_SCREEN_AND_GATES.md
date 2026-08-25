# B24 baseline screen, integrity gates, and future runbook

## Frozen source identities from B24.0 inventory

The B24 baseline wrapper must verify the complete source state, not HEAD alone.

| source | HEAD | tree/index/diff identity |
|---|---|---|
| DAPS | `e7a77d094167084faed19b599b96673b7bb11447` | tree `e63f9715e4704d9cd7a43a166559496d9d94e781`; index `d5487cdba570dbaac0c1909e549da361a0a0fc3fed81e5c13f59fa12925876b6`; tracked diff `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69` |
| official SITCOM | `275ab67efbd8146bffca20155171ba6be1169c09` | tree `80263442e3606824a06dc003504c28da5c59c2c5`; index `3ef63a8a29d0ba65cc642027a57ec102257fd9b387b0e9a5b4aae7f46d6a949f`; tracked diff `a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e` |
| NP/SITCOM | `52f2c37e587576d02e2b27ac971e247f2899fc5e` | tree `cfba2bbce7053d3e8642e5d74a603794bfb56bb2`; index `d0b3e024dee15358dab4ac9350b34f2cd8009eca4df8e4d9fd32a79829cc354e`; tracked diff empty SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

DAPS and official SITCOM are intentionally not assumed clean. Their frozen tracked diffs are part of the scientific identity and must match exactly.

FFHQ checkpoint: `/egr/research-pac/huang248/models/ffhq_10m.pt`, SHA-256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`.

Environment package-set SHA-256 identities:

- `daps`: `8d58352d0858ac02e39a0701abeb2e65623f205fcfe27ef0f210b8571800a7ad`
- `sitcom_ode_bw`: `96a52055561b88f66a25bb1bf09974ab828d394d291830453bd49d2a8d2fe34c`
- `prdiff_ffhq`: `ee361da9717975ac61a766891d91d1ac0a4c8fc9d2b3f7cf96168a9ee10cae5f`

## Baseline protocol

For each screened image: one measurement seed is frozen before outcomes; one locked measurement is reused by both methods; four DAPS and four SITCOM solver seeds are independently domain-derived; DAPS-4 means four independent frozen Fresh1 trajectories; SITCOM-4 means four independent frozen SITCOM-1 trajectories; every terminal gets raw RGB PSNR, SSIM, LPIPS, source/config/input hashes, GPU-active time, wall time, and terminal hash; best-of-four is selected only by raw-orientation RGB PSNR; and A/B/C/D classification uses Good25 only.

The union oracle is diagnostic and counts all eight trajectories.

## Serial/batched equivalence gate (future B24.1)

The same exposed images, measurements, canonical seeds, source state, and terminal candidate semantics must be used in serial and batched modes. Freeze before results:

- terminal tensor max absolute error <= `1e-5`;
- terminal relative L2 error <= `1e-6`;
- raw RGB PSNR difference <= `1e-4 dB`;
- SSIM difference <= `1e-6`;
- LPIPS difference <= `1e-6`;
- identical Good25/Good26/Good28 decisions;
- identical candidate count and seed-to-candidate correspondence.

Bitwise equality supersedes the envelope when obtained. A failed envelope cannot be relaxed after seeing B24.1 outputs merely to make batching pass.

## Memory/throughput gate (future B24.1)

Record PyTorch peak allocated/reserved MiB, B24-process `nvidia-smi` MiB, whole-device free/used MiB samples, GPU-active time, wall time, images/hour, and terminal candidates/hour.

Hard ceiling is 52,452 MiB for the B24 worker. Normal target is 48,000 MiB. Before launch, at least 52,096 MiB (48,000 + 4,096 reserve) must be free on the explicitly assigned GPU. This is a fit gate, not dynamic GPU selection.

## Atomic completion and resume

Each terminal candidate is written into an attempt directory. A candidate becomes reusable only after hashes and metrics validate. Image completion is a JSON record written to a temporary sibling, fsync'ed, and atomically renamed.

Resume verifies run-manifest hash, source hashes, measurement hash, solver seeds, and candidate hashes; skips a completed image only after validation; never silently regenerates a completed candidate; records every retry and reason; and permits at most two retries after the initial attempt. Any hash mismatch is a hard error rather than an overwrite.

## Four-GPU sharding

The scale manifest is hash-frozen before launch. Four shards are deterministic by row position modulo four after the frozen screen order. Each shard records its explicit physical GPU ID and UUID.

- GPU 0: `GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3`
- GPU 1: `GPU-883c037a-34d2-48c4-467f-9a352fd8fdff`
- GPU 2: `GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe`
- GPU 3: `GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a`

Shard sets must be disjoint and their union exactly equal the run manifest. One B24 worker per physical GPU. No worker chooses another GPU because it happens to be freer.

## Storage policy

Never store full DAPS trajectories in the B24 scale sweep. Retain metrics, hashes, timing, and provenance for every terminal candidate; best DAPS and best SITCOM terminals for every image; all eight terminals for every B/C/D image; all eight terminals for a deterministic 10% A audit subset (`B24_A_AUDIT_V1`); and measurement identity/seed plus enough locked input state to reproduce every completed row.

## Immediate-stop conditions

Stop on B24 source ancestry mismatch; source HEAD/tree/index/diff mismatch; model/environment identity mismatch; populated PRE-B23 exposure SHA mismatch or fewer than 328 inherited images; PRE-B24 exposure below 333 unique images or missing any required B23.1 image; future-reserve overlap; seed collision; shard overlap/omission; completed-record hash mismatch; unrecorded retry or silent regeneration; B24-worker memory above 52,452 MiB, OOM, or pre-launch fit failure; serial/batched disagreement; or any runtime path that lets clean ground truth affect early branch dropping, allocation, survival, or routing.

B24.0 itself must stop before any model/GPU command.
