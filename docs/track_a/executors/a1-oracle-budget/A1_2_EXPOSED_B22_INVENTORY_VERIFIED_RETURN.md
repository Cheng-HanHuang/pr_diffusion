# A1.2 exposed B22 DEV8 CPU metadata inventory — independently verified PASS

**Date:** 2026-10-19. **Owner:** continuing Track A executor. **Status:** `PASS_EXPOSED_DEVELOPMENT_INVENTORY_NOT_PROSPECTIVE`; **no A1.2 GPU run yet**.

User uploaded the completed `A1_2_EXPOSED_B22_INVENTORY_RETURN_V1.zip`; independently opened the archive, tested integrity (`testzip() is None`), and computed **ZIP SHA-256** `720f30adfa04a7629b766e597afbdc4ae1d853cca7e98ac20757020ebfd2b90b`.

The ZIP contains exactly three entries:
- `A1_2_EXPOSED_B22_CPU_TESTS_V1.log`, SHA-256 `9f556543131b59b1633e6cd01d42ed8412afca3fe0e95a499e13563647ec74ba`; unit tests **2/2 PASS**.
- `A1_2_EXPOSED_B22_INVENTORY_V1.log`, SHA-256 `1bcdb8f37f51be4ab47da3469b1b665c7357c7e8b49d673f486857d605d08945`; `PASS_EXPOSED_DEVELOPMENT_INVENTORY_NOT_PROSPECTIVE`.
- `A1_2_EXPOSED_DEV8_INVENTORY.json`, SHA-256 `468f9d9ee0f7f2d00a788c2d9d3422607179634b92a1c82f0c9d3d6bed45d7b1`; schema validates as eight uniquely selected exposed images. Original A1 source HEAD `3aa058cc796716dc0624b0c6f86f6d7d610d21d0`.

Selected eight **already exposed B22 final100** image IDs, in **deterministic hash ranking**:
`69073, 68539, 67948, 60140, 67293, 67174, 61101, 66631`.
Independently recomputed `SHA256("\\n".join(ids))` from the eight manifest IDs and verified `faa2e292e915690dc8dd7d477591bb47a12a5f5977fd1c9648849e25c2d09f32` matches returned selected-list checksum. Each record contains B22 archived row number, raw signed measurement path/file+content hash, source GT path/file hash, original GT tensor-content hash and rank key. B22 exposed manifest source file hash `94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219` matches previous verified A1.0 evidence.

Per-script assertions `gpu_work=false`, `measurements_opened=false`, `gt_images_opened=false`, `protected_b24_confirmation_access=false`, `approved_gpu_run=false`. These indicate the source-only inventory path; they do not independently establish actual bytes' correctness (which must be verified by the next CPU input-freeze step), nor third-party GPU activity.

**Scope firewall:** These 8 historical B22 exposed validation images are suitable for an **engineering K4 GPU extension only**, not a pristine FFHQ confirmation, prospective scientific reliability claim or statistically untouched final split. Protected B24 confirmation305 registry/IDs/files/metadata remain inaccessible. The paper-level preregistered float PSNR still requires homogeneous cross-solver float output, while the A1.1b pipeline empirically demonstrated uniform saved PNG8 scoring only.

**Next source work:** build a small A1-owned **CPU image+measurement byte-freeze** from these eight known exposed samples, with historical B21 `ToTensor -> Resize(256) -> CenterCrop(256) -> [-1,1]` GT transformation, verify original archived source file/content SHA before constructing any sampler command, save one normalized GT tensor per image into A1.2 exclusive output and a method-input identity manifest. No new noisy observations, model checkpoints, diffusion trajectories or GPU work during this input-freeze stage. Then prepare an independently seeded, separately reserved 128-trajectory GPU controller with its own aggregate GPU/time ceiling and output namespace, with no implicit start from successful CPU input-freeze.
