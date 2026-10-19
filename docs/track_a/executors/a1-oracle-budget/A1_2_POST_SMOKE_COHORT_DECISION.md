# A1.2 design decision after verified procedural four-method GPU smoke

**2026-10-19. Scientific status:** A1.1b V2 four native methods PASS; longer K4 study **NOT reserved / NOT launched**.

A1.1b verified ZIP SHA `875fae9cc3a546c82d72c15cf31964dc8c5728c52528b4e2f5005b5f7b63df3a`, all four methods produced full reconstructions on the same saved procedural measurement. Sum per-method subprocess wall ≈171.56s, full controller 173.03s, GPU event elapsed ≈153.72s. Peak reported device-used across methods 1,667 MiB, Torch reserved ≤966 MiB. **One-probe extrapolation:** 8 images × (4 DAPS + 4 SITCOM + 4 NP-LF + 4 NP-S2) = 128 trajectories, ≈5,490 seconds = 1.525 GPU-hours serial; allow ~3h total ceiling after image variation/overhead. This is projection, not a verified face-data timing distribution.

## Confirmatory-cohort blocker

The frozen `prdiffusion/b24_protocol.py:render_screen_manifest` considers FFHQ image IDs over `range(70_000)`. Therefore assuming the protected B24 confirmation305 cohort is confined to FFHQ validation IDs 60,000–69,999 is **not supported**; blindly choosing IDs in 0–59,999 also cannot prove disjointness. We must not access protected confirmation membership, registry, hash filter, metadata or GT to resolve this.

The A1 study preregistration requires a **separate eligible disjoint** eight-image engineering pilot and fully prospective publication test. Current exposed sets: B21/B22 final100, B24 DEV80. Their union of 180 is known but does **not** cover the protected set.

## Two scientifically distinct next studies

**A1.2 exposed B22 engineering K4 (possible, but exploratory):** choose exactly 8 from the frozen and fully exposed B22 final100 via a predeclared SHA-256 ranking (no GT-derived allocation); reuse their existing saved signed phase-retrieval measurements. Run K4 independent full native starts in separate processes for each DAPS/SITCOM/NP-LF/NP-S2 family (128 candidate trajectories); compute same-input raw saved RGB8 PSNR and rotation-180 sensitivity, aggregate compute and memory; offline NP8 LF4+S2-4 is reuse, not more inference. This would produce valuable *engineering* data for independent DAPS4 and independent SITCOM4 but **cannot** be called pristine, held-out, prospective confirmatory, or evidence of publication-level reliability. Must record a protocol *amendment* that this is a distinct engineering experiment, not silently satisfy the preregistered prospective A1 pilot.

**Pristine cross-source or independent-custodian pilot (scientific):** secure image data genuinely disjoint by provenance (e.g. independently acquired source with transparent domain shift, or a separate authorized custodian certifying non-overlap without exposing protected B24 records). Verify legal/technical suitability and freeze measurement namespace before running real-image GT. This cannot be inferred from publicly available FFHQ numeric IDs alone.

## Metric and workload amendment needed before A1.2

A1 prereg originally targets float raw RGB PSNR across all families. The working native DAPS wrapper only writes a terminal 8-bit PNG; the A1.1b four-native smoke compared all methods via **saved 8-bit PNG against quantized GT**, properly parity-matched. For *exploratory exposed-dev A1.2*, predeclare `COMMON_SAVED_PNG8_RAW_RGB_PSNR` as primary and rot180-aware as auxiliary (can report NP/SITCOM unquantized float sensitivity separately, but do not compare it to DAPS PNG). For a pristine study, either implement a source-pinned, noninvasive DAPS float terminal or prospectively amend the metric **before seeing fresh GT**, with all method scores recomputed from one representation. Do not quietly mix old B22 float with this PNG8 arm.

A1.1b CUDA event time is elapsed between GPU events and *not precise SM-busy seconds*; `torch.fft.fft2/ifft2` counters zero reflects **uncovered FFT APIs**, not proof of zero Fourier operator work. NP clipped 206,939/442,368 = 46.78% signed negative observation entries on this procedural probe, as its frozen historical method convention dictates.

**Decision:** GO design a small CPU-only metadata inventory for an explicitly exposed B22 engineering panel; STOP automatically advancing the user to new K4 GPU compute before an explicit [DATA RESERVATION] (development-only), source/metric gates, and [RUN RESERVATION] with Track B first priority. Keep A1.1b V1 STOP and V2 PASS outputs immutable.
