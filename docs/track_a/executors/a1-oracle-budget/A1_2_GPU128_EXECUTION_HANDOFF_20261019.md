# A1.2 independent-K4 GPU3 exposed-development run — executable handoff

**Dated:** Monday 2026-10-19 America/Detroit. **Owner:** Track A A1 executor. **State:** `GPU_SOURCE_COMMITTED / USER_LAUNCH_NOT_YET_EXECUTED`; source-only engineering, not a scientifically held-out FFHQ study.

## Prerequisites (already verified)

- B22 exposed DEV8 inventory: eight fixed IDs `69073, 68539, 67948, 60140, 67293, 67174, 61101, 66631`, selection SHA `faa2e292e915690dc8dd7d477591bb47a12a5f5977fd1c9648849e25c2d09f32`.
- Exposed input-freeze ZIP SHA256 `219412e50d9dc40d2e028d5f41ad4cc8480b4166a7bc9e7bc04593daa7a55d04`, 8/8 historical measurements+GT content identities, summary JSON SHA `fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490`. No new observation generation.
- Source-pinned CPU K4 plan ZIP SHA256 `7777d36f78464a0782d3cde446f88f53fd7b98d0a2b1fde70afff767e0d4dd20`, nine CPU tests PASS, exactly 128 **planned** independent candidate identities, four candidates/method/image, native RNG seed32 domain-separated. No prior A1.2 GPU work.
- Successful A1.1b V2 procedural four-native smoke ZIP SHA256 `875fae9cc3a546c82d72c15cf31964dc8c5728c52528b4e2f5005b5f7b63df3a`.

## Execution contract

User-only PAC code: `scripts/track_a/a1_2_k4_exposed_b22.py`; one-shot wrapper `scripts/track_a/a1_2_run_k4_gpu_v1.sh`. The executable source still requires exact pinned A1 HEAD, physical GPU index 3, reservation token and dated UTC start window. **No other A1.2 GPU launcher/source is authorized**. Do not call Python `--mode execute` directly without wrapper checks.

Run matrix (no optimizer changes): 8 already-exposed B22 images, each DAPS K4, SITCOM K4, NP-LF K4, NP-S2 K4. **128 distinct native full trajectories**, each separately seeded and executed in its own process. Source methods/numerical schedules remain the A1.1b frozen natives; no GT-based start selection, shared raw signed measurement remains fixed; NP clamp-min-zero input convention is recorded. Method order and replicate order frozen. NP mixed4 and mixed8 are **offline reused** LF/S2 candidates, no extra inference.

Launch window **October 19, 2026 22:30 EDT through October 20, 2026 01:00 EDT for START** (UTC `2026-10-20T02:30:00Z` to `2026-10-20T05:00:00Z`). Execution may last **at most 10,800 seconds / 3h**, so all work must finish by no later than 04:00 EDT on October 20. The controller stops immediately if the dated window is missed or if the initial physical GPU3 is contested/unavailable.

Resources: **GPU3 only**, pinned UUID `GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a`, one native worker at a time, all other physical GPUs untouched. No competing Track B use (Track B first priority). Initial physical GPU3 used ≤1,024 MiB, free ≥52,096 MiB; while active, observed physical used ≤52,452 MiB (~51.2GiB); no method trajectory may exceed 600 seconds; aggregate ≤10,800 seconds. Measured one-procedural-input 4-method process sum 171.56s, linear 128-candidate estimate ~1.53 GPUh is **an estimate only**; native FFHQ cost can vary. No hidden retries/resume in the same run directory. STOP immediately preserves current outputs, earlier completed candidate receipts, and failing native log for diagnosis. Shell keeps terminal open on failure.

Full output root **exclusive**, with first attempt `a1_2_exposed_b22_k4_v1/`, parent `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/`. User's return ZIP: `A1_2_K4_GPU_METADATA_RETURN_V1.zip`, summary log `A1_2_K4_GPU_MAIN_V1.log`, 128 candidate JSON receipts and corresponding timers/results, postcompletion `A1_2_OFFLINE_ORACLE_SUMMARY.json` / `A1_2_COMPLETE.json` or on failure `A1_2_STOP.json`, plus last four native log **tails only**. No face PNGs, GT .pt, measurement .pt, or reconstruction tensors transferred. Root output preserves full high-volume native logs and image/tensor bytes **on PAC** for follow-up diagnosis. Archive has no auto-redelivery; user must attach it.

## Scoring and scientific firewalls

**Primary exploratory K4 metric** is identical across all method families: **saved quantized RGB8 PNG raw PSNR against quantized GT**, image orientation fixed; rotation180-aware PSNR is auxiliary. This is a *separate engineering protocol* from originally preregistered float raw PSNR on an untouched final cohort. Do not merge or compare numerically to B22 archived float oracle results as if identical representations. Offline K1/K2/K4 PSNR-max over *completed candidate populations*, NP-MIXED4/8 reuse, all candidate costs charged; the oracle **requires GT and is not deployable**. Dataset is already-exposed B22 final100, not a fresh representative sample. The protected B24 confirmation305 remains entirely inaccessible (including IDs/metadata/membership hash tests). Truly held-out publication comparisons remain separate.

## Exact user execution sequence

1. Fast-forward dedicated A1 checkout from `7971b1975c42f291c4bc6a160380b937977b04c2` to newest pinned source commit displayed in executor instructions. Never merge if branch, HEAD or tracked files don't match.
2. Run updated unit tests `scripts.track_a.test_a1_2_k4_exposed_b22`, `bash -n scripts/track_a/a1_2_run_k4_gpu_v1.sh`, and `--mode plan` (should return candidate_count=128 and GPU enabled **only with actual lease ACK**). No new CPU ZIP or new data reads needed.
3. Check physical GPU3 and `nvidia-smi --query-compute-apps`; if in use, preserve priority for Track B and STOP. Start only during explicitly posted [RUN RESERVATION] time window.
4. Run the **one-shot** V1 GPU shell with pinned HEAD and exact lease token `TRACK_A_A1_2_EXPOSED_B22_K4_GPU3_20261019_NIGHT`, in a durable terminal or tmux session if available; do not run simultaneous jobs. Output progress reports each candidate. User returns terminal ending and ZIP SHA/bytes for independent analysis.
5. After PASS/STOP, do not rerun over the same A1.2 output; subsequent work requires new source, reservation and output namespace. No automatic prospective test or Track B interruption.

The dated lease is **conditional**: it does not preempt any Track B processes; a physical idle snapshot is required immediately prior to start. All PAC work executes only under explicit user command, never performed remotely by the executor.
