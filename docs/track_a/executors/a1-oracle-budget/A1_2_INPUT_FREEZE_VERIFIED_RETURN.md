# A1.2 eight exposed B22 inputs — CPU freeze independently verified PASS

**Date:** 2026-10-19. **Owner:** continuing Track A executor. **Status:** `A1_2_EXPOSED_DEV8_INPUT_BYTES_VERIFIED_PASS / GPU_NOT_RESERVED`.

User supplied original PAC `A1_2_INPUT_FREEZE_METADATA_RETURN_V1.zip`, independently read the archive:
- ZIP SHA256 **`219412e50d9dc40d2e028d5f41ad4cc8480b4166a7bc9e7bc04593daa7a55d04`**, ZIP CRC `testzip()==None`.
- Exactly **11 members**, metadata only: unit tests, run log, eight `item.json` and `A1_2_INPUT_FREEZE_COMPLETE.json`. No image/tensor bytes in ZIP.
- `A1_2_INPUT_FREEZE_CPU_TESTS_V1.log` SHA `d152c3e606b258ecbab6cc56d70013d8ea56ab56d9f0518a127fe2e7a7976edf`, **3/3 tests PASS**.
- `A1_2_INPUT_FREEZE_RUN_V1.log` SHA `421f14f84db8bd007a1384b83057d86d06dd37eaf57b0cc12e5e89d8c473dbe7`, exactly **eight `A1_2_INPUT_VERIFIED`** row receipts and terminal `PASS_EXPOSED_B22_INPUTS_FROZEN_NOT_HELDOUT`, no model/measurement generation/GPU.
- `A1_2_INPUT_FREEZE_COMPLETE.json` SHA `fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490`, schema `prdiff.a1.2.exposed-input-freeze.v1`, source A1 checkout HEAD `2d1c2ae350a54879c86cf2547b388c157479b12f`, selected-list SHA `faa2e292e915690dc8dd7d477591bb47a12a5f5977fd1c9648849e25c2d09f32`, inventory source SHA `468f9d9ee0f7f2d00a788c2d9d3422607179634b92a1c82f0c9d3d6bed45d7b1`.

Independently parsed every per-image `item.json` and confirmed its **full JSON object equals** the corresponding row in the eight-image summary, no exceptions. Frozen order:
`69073, 68539, 67948, 60140, 67293, 67174, 61101, 66631`. Frozen signed measurement contents SHA:
- `69073`: `4173cf301a095386de28106df26bbe07252943d257d9bf11bbab4e3dd9af1be5`
- `68539`: `7432e63b4d7d9d96989eb5c5d61ed1bd9145a8d9a99f9c5f1f47cb217cbce116`
- `67948`: `229c3bc572939d3dce36a624704bdd447fd265145b36d7a59ecf76ca6ea6e8a5`
- `60140`: `b5857632cd9488ab285e5d8a9e47a656ee482ec06d533cb3250bfc453ebdf5c7`
- `67293`: `c42d683f8b8928d3bd2cecf3dc80fe63bf141d6e81b645a85c32448fa6ea3bd2`
- `67174`: `1133a359a1fe132776b669da8e22d3ed2c72c0d98035ffce3cef1eb26837fdb0`
- `61101`: `282db1aabb681d9474b2e51c4a3047f009cf6c418165255493d231f09e2f105d`
- `66631`: `3a1c84b0e05d483f046011acdabf9dae3ab1a76bdcacdc77641663799c436740`.

The **PAC source** checked both original source-image byte SHA and frozen historical GT content SHA (B22 exact `ToTensor→Resize(256)→CenterCrop(256)→2x−1` transform), and original .pt file/content hashes of raw noisy phase measurements; PAC then saved eight A1-owned GT tensors. Independently verifying the **ZIP** establishes machine-receipt integrity and row/log consistency; it does not permit personally recomputing these tensor SHA values because the large source/tensor bytes were deliberately not in the ZIP. Do not overstate that limitation.

**Scientific scope:** these eight images are *already exposed B22 final100*, not pristine/held-out. CPU input-freeze uses no GPU, no native solver. No B24 protected confirmation305 metadata, IDs, paths or payload may be touched. One manually scheduled A1.2 exploratory independent K4 GPU engineering run may be planned separately under a new GPU reservation.

**Successor design:** A1.2 independent K4 source-pinned CPU plan and regression tests, candidate grid `8×(4 DAPS+4 SITCOM+4 NP LF+4 NP S2)=128`, domain-separated seeds, total cost/condition accounting, oracle K1/K2/K4 + offline NP-Mixed4/8 from existing candidates. The GPU execution branch of the controller is explicitly **disabled** (`EXECUTION_ENABLED=False`) until CPU PAC plan returns and a dated, physical resource lease is approved. No unconditional GPU script is offered.
