# Track R theory executor — A14 / Round 10 return (2026-10-29 night)

**Integrity and access.** Verified all **2** entries of the SHA256SUMS.txt in uploaded theory_upload_20261029_NIGHT.zip before opening 10_PLANNER_GUIDANCE_20261029_NIGHT.md or the PDF snapshot. The draft paper already states C-T14: H2's gain succeeded while absolute levels were 3–6 percentage points too high. No new access to xp/returns or other held outcome/status files. Only the planner-relayed F8/CP1 and prior released B16/D1 outcomes were used.

## 1. Half-page F8 note (DONE, post-outcome)

Committed docs/track_r/theory/TH_F8_LOCKED_INTERPRETATION_20261029_NIGHT.md. Uses the locked pre-results rule @**7a09153**, with immutable v1 @**8ef5f94** and v3 @**e71026e**. At primary DAPS F8, v1 race covers **21/28** policy/cohort/K points (E0 7/7, E1 5/7, Oraw 5/7, Oorbit 4/7), S2 covers **22/28** (selection E0/E1 14/14), and v3 race **17/28**. At pooled K8: observed (E0,E1,Oraw,Oorbit)=(.928,.939,.939,.944), vs v1 race (.943,.963,.962,.968). v1 Oorbit=.9685 [.94475,.99290] narrowly misses; 11 images never GOOD in the eight runs. v3's Oorbit=.9344 [.9195,.9510] covers this one oracle but overall v3 is worse.

With few CONSISTENT failures in historical DAPS (B22 C0/200; DEV80 C18/320), S2 and race E1 success approach each other: frozen pooled K8 E1 gap is only **1.97 pp**, although the E0 gap remains **4.06 pp** (because residual class competition does not vanish). Thus near-agreement of repaired E1 does not prove exchangeable ranks.

## 2. S1: why H2 gain correct but levels wrong (DONE)

Updated docs/track_r/theory/TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex with a new POST-OUTCOME paragraph. The prospective pre-score note **d42ac3cc0f7c40aabd824eef2a7fa1be654db4ff**, committed **2026-10-29 20:12:55 UTC**, remains untouched. CP1 observed +3.38 pp, inside its +2 to +4 pp interval (central +3), but observed runs **3.195** vs predicted **2.835**, observed stop/fixed E1 **(.931,.897)** vs predicted exchangeable **(.957,.930)** or race **(.993,.953)**. Historical pass threshold rejects **9/165 B22 and 9/170 DEV80 GOOD**; replacing heterogeneous s_i with cohort averages loses nonlinear compute/success structure. Optimistic model errors in protocol and same-budget fixed comparator partly cancel in their difference. Do not claim 3–6 point absolute calibration from correct gain.

## 3. S1: FLAG_C precision formula (DONE)

Let U be latent intrinsically unsolvable, pi=P(U), S the per-run threshold-pass probability, N=Nmax. Under conditional iid runs:
\[
\Pr(U\mid\mathrm{FLAG}_C)=
\frac{\pi\,\mathbb E[(1-S)^N\mid U]}
{\pi\,\mathbb E[(1-S)^N\mid U]+(1-\pi)\,\mathbb E[(1-S)^N\mid U^c]}.
\]
Solvable low-S images can cause enough false flags that precision falls below 1/2; unsolvable CONSISTENT outputs do not generate FLAG_C. The planner's H3 ratios, DAPS 5/11 and ImageNet 1/3, are finite-panel analogs, NOT estimates of latent structural impossibility. The S1 insertion includes the caveat explicitly.

## 4. Paper and registry guards

No changes to frozen v1/v2/v3/v4 or the immutable H2 pre-score note. No main paper mutation or S2/S3 rewrite; S2 symmetry/source and S3 selection/bounds theory-owned drafts remain available for Nov 6 integration. The short F8 interpretation and S1 additions are ready for planner review. All negative coverage and H3 results are reported explicitly.
