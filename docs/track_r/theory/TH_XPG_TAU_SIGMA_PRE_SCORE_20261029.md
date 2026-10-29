# XP-G pre-score theoretical note: should FAST tau scale linearly with sigma?

**Date 2026-10-29, theory status: BEFORE examining XP-G outcomes.** This is a conditional physics analysis of hypothesis H-G1, not an empirical prediction fitted to the held XP-G calibration. Do not open xp/returns or XP-G result/status files until the score is registered.

## Precise H-G1 and thresholds

The frozen target at sigma0=0.05 is tau0=0.0500829. H-G1 proposes both measured GOOD-class signed-residual thresholds tau(sigma) are within 1% (relative) of the linear predictions
\[
 \tau_{\mathrm{lin}}(0.02)=0.02003316,\qquad
 \tau_{\mathrm{lin}}(0.10)=0.1001658.
\]
The 1% bands around these numbers are, respectively, [0.0198328284, 0.0202334916] and [0.099164142, 0.101167458]. Their comparison against the observed tau is a prospective, non-retrofittable test.

## What C-T9 actually implies

Let the signed measurement be y_sigma=a+sigma Z with fixed normalized noise Z and define physical signed residual
\[
 \rho_\sigma(\hat x_\sigma)=
 \frac{1}{\sqrt m}\|A(\hat x_\sigma)-y_\sigma\|_2,\qquad
 d_\sigma=A(\hat x_\sigma)-a.
\]
Exactly,
\[
 \frac{\rho_\sigma(\hat x_\sigma)}{\sigma}=
 \frac{1}{\sqrt m}\left\|Z-\frac{d_\sigma}{\sigma}\right\|_2.
\]
Thus a noise-floor proportionality statement follows **only if** the normalized reconstruction perturbation d_sigma/sigma has an approximately sigma-invariant law, including its correlation with Z and any selection on the event GOOD. In the idealized, locally linear, sigma-homogeneous Gaussian posterior toy with fixed effective precision and dimensionless dynamics, d_sigma is proportional to sigma, so a fixed-quantile threshold scales **exactly**. The same holds for an effectively noise-insensitive exact reconstruction (d_sigma nearly 0) over a range where error is negligible compared with sigma.

However, this is **not a theorem for deployed FAST-DIPS**. The native FAST inner hard-projection tolerance epsilon=0.05 is fixed when the measurement noise sigma changes, so epsilon/sigma is 2.5 at sigma=0.02 but 0.5 at sigma=0.10 (versus 1 at the 0.05 calibration). Its returned object is clamp(x0hat), not necessarily the hard-projected intermediate, and denoiser/prior error may introduce a nonvanishing d_sigma term at small sigma. Neither the local Gaussian toy nor the fixed-epsilon solver audit controls this error to 1% across a fivefold noise-level change.

**Pre-score assessment:** C-T9 supports H-G1 as a plausible *noise-dominated approximate scaling null*, but it **does not predict that the specific 1% test must pass**. Given fixed epsilon/sigma and the output being an unprojected denoiser estimate, we should mark H-G1 **unresolved (at substantive risk of failure)** rather than claim a theorem or choose an outcome retrospectively. At small sigma, any approximately sigma-independent reconstruction bias tends to make rho/sigma *larger*, but data fitting can change the sign; no robust directional sigma trend is implied. This caveat should accompany the source calibration paragraph regardless of the eventual XP-G observation.

**Falsification mapping:** Passing both tests would demonstrate approximate scale invariance of the empirically calibrated GOOD residual quantile only over the tested sigma range; it would not establish general diffusion equivariance or epsilon-independence. A failure would not falsify the noise floor identity above, only the additional sigma-homogeneous solver-output assumption.
