# B25 mathematical model

## 1. Independent proposal selection

Let `e_1,...,e_K` be iid with density `q` on `R^d`, let `S:R^d->R` have a continuous distribution under `q`, and choose

`J = argmin_j S(e_j)`.

For any proposal value `e`, the event that proposal 1 lies near `e` and wins requires the other `K-1` proposal scores to exceed `S(e)`. Hence the selected density is

`q_pick(e) = K q(e) [1-F_S(S(e))]^(K-1)`.

This is a standard order-statistic identity. B25 tests the implementation against it; B25 does not claim this identity as a new theorem.

### Linear check

Take `q=N(0,I_d)` and `S(e)=g^T e` for unit `g`. Write `Z=g^T e`. Then the selected projection is `Z_(1)=min(Z_1,...,Z_K)` with density

`f_K(z)=K phi(z) [1-Phi(z)]^(K-1)`.

The orthogonal components remain standard Gaussian and independent of the winning score conditional on the winning index. Numerical quadrature of `f_K` supplies analytic mean and variance references.

## 2. Small-step transition and likelihood weighting

Consider

`X_h = x + b h + sqrt(h) e`.

### Linear energy

Let `V_lin(u)=g^T u` and likelihood `L(u)=exp(-V_lin(u))`. The continuously likelihood-tilted proposal density is

`p*(e) proportional to phi_d(e) exp(-sqrt(h) g^T e)`,

so

`e ~ N(-sqrt(h) g, I_d)`

and therefore

`X_h ~ N(x + b h - h g, h I_d)`.

Thus a correct continuous exponential tilt produces an `O(h)` mean correction. By contrast, hard minimum selection among fixed `K>1` candidates selects a score projection with nonzero `O(1)` mean, so its induced state displacement contains an `O(sqrt(h))` term.

Finite-candidate likelihood-weighted resampling draws one of the `K` proposals with probability proportional to `L(X_h^j)`. It is not equal to the continuously tilted target for finite `K`; B25 measures its finite-K discrepancy and small-step scaling.

### Quadratic energy

Let

`V_quad(u)=0.5 ||u-c||^2`.

The proposal state is `N(m,hI)` with `m=x+b h`. Multiplication by `exp(-||u-c||^2/2)` gives the analytic Gaussian target

`mean = (m + h c)/(1+h)`,

`covariance = h/(1+h) I`.

This provides a smooth nonlinear score with an exact reference.

## 3. Finite-support exact prior

Let the prior support be `x_1,...,x_M` with weights `pi_i>0`, `sum pi_i=1`. All templates are nonnegative `6x6` real arrays. The synthetic measurement operator is

`A(x)=vec(|FFT2(x; norm='ortho')|)`.

Measurements obey

`Y = A(X_0) + sigma_y eta`, `eta~N(0,I)`.

Therefore

`p(y|x_i) proportional to exp(-||y-A(x_i)||^2/(2 sigma_y^2))`

and the exact terminal posterior is

`w_i(y) = pi_i p(y|x_i) / sum_j pi_j p(y|x_j)`.

## 4. Coherent VP noising chain

The frozen marginal schedule is

`alpha_0=1 > alpha_1 > ... > alpha_T=0`.

For each template `x_i`, the marginal is

`Z_t | X_0=x_i ~ N(sqrt(alpha_t) x_i, (1-alpha_t) I)`.

The Markov transition from `s=t-1` to `t` is

`Z_t = sqrt(alpha_t/alpha_s) Z_s + sqrt(1-alpha_t/alpha_s) xi_t`.

This chain has the stated marginals and makes `Z_T~N(0,I)` independent of the template because `alpha_T=0`.

For `alpha_s>alpha_t`, the Gaussian bridge conditional on template `i` is

`Z_s | Z_t=z, X_0=x_i ~ N(mu_{s|t,i}(z), v_{s|t} I)`

with

`r = sqrt(alpha_t/alpha_s)`,
`c = r (1-alpha_s)/(1-alpha_t)`,
`mu_{s|t,i}(z)=sqrt(alpha_s)x_i + c (z-sqrt(alpha_t)x_i)`,
`v_{s|t}=(1-alpha_s) - r^2 (1-alpha_s)^2/(1-alpha_t)`.

At `s=0`, `v=0` and the bridge returns the exact template.

## 5. Exact and approximate reverse transitions

Unconditional template weights at noisy state `z_t` are

`a_i(z_t) proportional to pi_i N(z_t; sqrt(alpha_t)x_i, (1-alpha_t)I)`.

The unconditional reverse kernel `p(z_s|z_t)` is the mixture of the bridge Gaussians with weights `a_i(z_t)`.

Because `Y <- X_0 -> Z_s -> Z_t` is Markov after conditioning on `Z_s`,

`p(z_s|z_t,y) proportional to p(z_s|z_t) p(y|z_s)`.

The exact intermediate likelihood is

`p(y|z_s) = sum_i p(y|x_i) p(x_i|z_s)`,

where `p(x_i|z_s)` is finite-support Bayes under the prior/noising model.

An exact conditional reverse draw can equivalently sample template index using

`p(i|z_t,y) proportional to pi_i p(y|x_i) N(z_t; sqrt(alpha_t)x_i,(1-alpha_t)I)`

and then draw from the bridge for that index.

B25 compares this exact reference against:

1. **Hard exact-intermediate selection:** draw `K` proposals from `p(z_s|z_t)` and keep the proposal maximizing exact `p(y|z_s)`.
2. **Weighted exact-intermediate selection:** draw the same kind of `K` proposals and resample one with probability proportional to exact `p(y|z_s)`.
3. **Weighted denoised-point approximation:** replace exact `p(y|z_s)` with `p(y|x_hat(z_s))`, where `x_hat(z_s)=E[X_0|z_s]`, then finite-candidate resample.

Items 1–3 are finite-candidate heuristics. Item 2 is a sampling-importance-resampling approximation to the conditional kernel, not an exact draw for finite `K`.

## 6. Exact ambiguity

For a real array `x`, let `R x` be its 180-degree spatial reversal. The discrete Fourier transform obeys a conjugate/phase relation under reversal, so

`|FFT2(Rx)| = |FFT2(x)|`.

Consequently exact reversal pairs have identical likelihood for every measurement `y`. Their posterior odds equal their prior odds:

`P(x_i|y)/P(x_j|y)=pi_i/pi_j`

whenever `A(x_i)=A(x_j)`.

The benchmark verifies this numerically rather than assuming it.

## 7. DEV80 RGB symmetry family

The historical FFHQ operator is channelwise. In the common canonical representation, each RGB channel is a real `256x256` array in `[0,1]`, symmetrically zero padded by 64 pixels per side, transformed by

`ifftshift -> fftn(norm='ortho') -> fftshift`,

and converted to magnitude.

For mask `m in {0,...,7}`, channel `c` is reversed by 180 degrees iff bit `c` of `m` is one. Because channels are measured independently, any combination of per-channel identity/reversal preserves the noiseless channelwise magnitude under the verified implementation.

B25 first checks this on synthetic arrays at the frozen tolerance. Only then is the family applied to existing DEV80 terminal reconstructions. Arbitrary translations, sign changes, and 90-degree rotations are not included.

## 8. Interpretation boundaries

- A selected proposal becoming non-Gaussian is not by itself evidence of incorrect posterior sampling.
- A true conditional transition generally differs from the unconditional transition.
- Agreement of mean/covariance alone does not prove distributional equality; B25 also records directional histogram/quantile discrepancies.
- The independent-proposal small-step model is not automatically a theorem about historical NP, because historical NP reuses the incumbent selected noise as one proposal, denoises proposals with a neural model, scores denoised states, and applies late projection.
- The finite-support benchmark isolates inference mechanisms with an exact prior; it is not a native DAPS/SITCOM benchmark.
- GT-assisted symmetry PSNR is an offline diagnostic upper bound, not a deployment rule.
