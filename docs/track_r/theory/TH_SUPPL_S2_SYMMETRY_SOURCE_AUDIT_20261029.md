# Supplement S2 integration instruction — symmetry and source audit (2026-10-29)

**No change to actual paper Sec. 4 or former symmetry source.** The
planner explicitly chose main-text selection as the theoretical core,
with the orbit lemma as a **classical one-paragraph remark**. The
SHA-256-verified uploaded `paper/X_symmetry_moved_to_suppl.tex`
(SHA `fc34f945d27c31c550a66fa4dee239cc64feefd34555d4ae4c3d6ba05bac32b3`)
is the authoritative source for the relocated Prop. ratio, Cor. blind,
Thm. equivariant, TV defect inequality and the posterior/sampler
triangle bound. Insert that source into Supplement S2 in a later
planner-controlled paper pass; do not silently reprove/renumber the
claims or change their precise hypotheses.

## Additional one-paragraph blur/misalignment note (math verified)

For *shift-invariant circular convolution* with a fixed known kernel k,
`F(x*k)=Fx . Fk`, and hence `|F(x*k)|=|Fx| |Fk|`.
Let g be any Fourier-magnitude twin transformation of x, e.g. a
translation or conjugate reflection (for real images, 180-degree
reflection modulo shifts). Since `|F(gx)|=|Fx|`,
`|F((gx)*k)|=|F(x*k)|` for the **same** fixed kernel k.
Therefore an ordinary known blur, even an asymmetric blur, cannot
resolve the twin using only the Fourier-magnitude data; an unknown
translation or misalignment also cannot add orientation information
when it only multiplies the Fourier transform by a unit-modulus
phase factor. This is a statement about the measurement operator,
not about an algorithm's prior preference or a blur performed in a
nonconvolutional, spatially varying way.

A *known coded illumination mask* inserted before Fourier sensing
`|F(m .* x)|`, or an added *known reference* object measured jointly
with the unknown image, generally does not preserve the same orbit
symmetry because the known structure is not transformed along with x.
Those designs can break the twin ambiguity under their own
identifiability assumptions; they are **different measurement models**,
not an advantage attainable by replacing one phase-retrieval solver or
prior under the present bare-magnitude operator. The planner's
references for this distinction are Candès--Li--Soltanolkotabi (2015),
Barmherzig et al. (2019), and the random-mask sensing used by DOLPH;
exact bibliographic entries and each method's operator convention
should be checked against the bibliography before integration.

## Source-faithful solver qualifications to preserve in S2

1. **DAPS/SITCOM:** prior analysis identified the returned final
   Langevin sample at the terminal outer sigma=0. The Gaussian
   equivariance-TV argument applies only to the part of the chain
   satisfying its isotropic Gaussian-kernel hypotheses, and no claim
   can be made for a non-Gaussian or deterministic post-processing
   step without a separate proof. SITCOM's SSE correction and the
   precise source versions require their per-solver audit citations.
2. **FAST-DIPS:** the terminal output is a denoised, clamped x0-hat,
   not the final Gaussian state. The relocated TV inequality can
   bound the last Gaussian state only, unless a separate deterministic
   pushforward argument is supplied. Do not claim native-output law
   control by a Gaussian-chain theorem without it.
3. **Likelihood-only pruning** is invariant on exact Fourier orbits;
   full twisted-SMC weights can also involve proposal/prior factors,
   so the former property is not a theorem about all SMC variants.
4. **Prior and orbit assumptions:** the classical posterior density
   ratio holds where both densities are defined and positive; the
   symmetry group acts by volume-preserving permutations on the
   even, periodic canvas. The eight-orbit equal-mass conclusion
   requires exactly invariant sampler laws and distinct disjoint
   orbit neighborhoods, neither of which is established for a
   trained denoiser.

This S2 integration note deliberately does **not** restate the long
symmetry proofs; retain the source text as requested by the planner.
