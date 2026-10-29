# Track R theory executor — A11 round-7 response, October 29, 2026

**STATUS:** Protocol theory (b)(1)–(3) and theory-owned Supplement S1/S2/S3 drafts prepared ahead of Nov 3/6. Main paper Sec. 4 was NOT modified. This is **NOT** a v3P freeze. No `docs/track_r/executors/xp/returns` or individual D1-P2 phase-2, SITCOM B16 or DAPS F8 outcome data were opened.

## Input integrity and governance

Before reading the planner's guidance, the upload `theory_upload_20261029.zip` was independently checked against `SHA256SUMS.txt`: **21/21 member SHA-256 digests match**. The frozen protocol source is `protocol/GTFREE_PROTOCOL_v1.md` SHA-256 `7e9a5fd8b817a97351d07ae371b947688c8430cd5b9268fc99b954a493223919`, and `protocol/gtfree_protocol.py` SHA `d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658`; planner freeze c598176. The verified moved-symmetry source is `paper/X_symmetry_moved_to_suppl.tex` SHA `fc34f945d27c31c550a66fa4dee239cc64feefd34555d4ae4c3d6ba05bac32b3`.

The half-page ImageNet D1-P2 interpretation was **already committed** in the previous round at `TH1_IMAGENET_D1_PHASE2_LOCKED_INTERPRETATION_20261029.md` and theory commit 9346620; it uses the locked 7a09153 plan, does not refit the predictions and reports the successful v2 E1 coverage together with the orbit-generation miss.

## New central protocol proposition: PROVED

See `TH_PROTOCOL_STOP_THEORY_20261029.md` and suggested main-text drop-in `TH_PROTOCOL_MAIN_PROPOSITION_20261029.tex` (not merged into the paper). With per-image pass probability s=P(rho<=tau), r=1-s, stop-at-M time truncated at N obeys the negative-binomial tail sum

E[T] = sum_{t=0}^{N-1} P(Bin(t,s)<M).

For the frozen M=2, `E[T]=(2-N*r^(N-1)+(N-2)*r^N)/s` (continuously N at s=0); `P(FLAG_C)=r^N`. Crucially, FLAG_C means **zero** passing runs, not fewer than M.

Let H_j^+ be the chance the minimum-residual candidate among j passing runs succeeds after orientation, and H_j^- the analogous nonpassing selector. The exact protocol success (conditional iid marked-run model) is

S(2,N) = (1-r^N-N*s*r^(N-1))*H_2^+ + N*s*r^(N-1)*H_1^+ + r^N*H_N^-.

The `r^N*H_N^-` term cannot be silently deleted: calibration tau is the 95th GOOD percentile (so v_G about .95, not 1), and ImageNet uses a transferred threshold. Class probabilities and unconditional PL race weights **do not determine** the physical-residual conditional pass/rank law. An explicit competing-risk integral supplies H_j when the needed class-conditional residual densities exist.

### Exact finite-panel random-order QA theorem: PROVED

The frozen evaluator shuffles a **fixed n-run panel**; if L of n runs pass tau, and all n runs are eligible (N=n), then

E_order[T] = M*(n+1)/(L+1) if L>=M, else n;
P_order(FLAG_C) = 1{L=0}.

For L>=M, the first M passing runs are a **uniform M-subset of the L passing runs**, so exact average E1 success is the normal subset-rank weighted average of the physical-residual-ordered passing candidates' frozen orientation-corrected success indicators. For L<M, the protocol exhausts all n runs and outputs their deterministic residual minimizer. A hypergeometric formula extends this to N<n. Thus for protocol success, runs and FLAG_C, the frozen 4000-order Monte Carlo can be independently audited against exact finite-panel means **without changing the frozen evaluator**. A new no-data regression `scripts/track_r/theory/check_protocol_panel_exact.py` passed **199 exhaustive synthetic random-order cases**; the iid stop-time regression passed 308 parameter cases.

## Equal-compute gain: qualified theorem, not an unconditional dominance claim

Stopping allocates more trials to images with low consistency probability s: E[T(s)] is decreasing in s and Cov(s,E[T(s)]) <= 0. This is **not sufficient for better accuracy**: low-s images may be unsolvable, or more than M passing candidates may be necessary for a stronger class-race selection probability. We proved an explicit sufficient inequality in the theory note and gave a synthetic *counterexample*: equally likely images with s=.02 (all passing reconstructions wrong) and s=.9 (half GOOD and half CONSISTENT, PL rates w_G=1,w_C=.1), N=16,M=2. The mean matched budget is 9.01267 runs, but stopping succeeds 0.35227 versus 0.44263 for fixed best-of-K, **a 9.04-point loss**. No universal theorem should be claimed from heterogeneity alone.

## Failure-type mechanism and limit

With `Pr(G)=p`, `Pr(C)=(1-p)*alpha`, `Pr(I)=(1-p)*(1-alpha)`; G/C always threshold-pass, I never does; and passing residual ranks are exchangeable, `alpha=c/(c+u)` and `s=p+c`. Let q=p/s. Then

S_stop = q*(1-u^N), F(k) = q*(1-u^k),
Gain(alpha)=q*[(1-a)*u^floor(Kbar)+a*u^ceil(Kbar)-u^N]
with a=Kbar-floor(Kbar), Kbar=E[T(s)] and the preregistered linear interpolation. The gain is nonnegative in this reduced model and goes to **zero** as alpha→1, since s→1 and both procedures use M runs. Under a class race replace q by its H_j PL winning probability; the vanishing limit still holds but positive gain at intermediate alpha is not guaranteed. Under the **actual** v_G≈.95 model, alpha→1 does not alone imply s→1, so the clean limit needs the stronger s→1 hypothesis.

The observed DAPS DEV80 +4.9 point gain at roughly 12% CONSISTENT failures and SITCOM DEV80 −0.1 at roughly 60% CONSISTENT failures are compatible with the mechanism, but **not a causal proof or universal monotonic trend**.

## Supplement and main-text drafts (THEORY-OWNED, not merged)

- `TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex`: complete truncated waiting-time law, conditional-race selector, FLAG_C precision/recall, exact fixed-panel random-order theorem, sufficient gain criterion, counterexample and C-share limit. This should be S1's mathematical spine.
- `TH_SUPPL_S2_SYMMETRY_SOURCE_AUDIT_20261029.md`: preserve the relocated classical orbit proofs **verbatim** from the supplied X_symmetry file, append why shift-invariant blur or translation cannot resolve twins, and qualify the DAPS/SITCOM/FAST returned-state source audit.
- `TH_SUPPL_S3_SELECTION_BOUNDS_DROPIN_20261029.tex`: class-race formula, rank U-statistics, negative monotone-gap example, explicit Gauss–Radau 4-moment obstruction, the A8-only continuum moment bounds and honest prospective success/failure interpretation.
- `TH_PROTOCOL_MAIN_PROPOSITION_20261029.tex`: a compact candidate for Sec. 4 if the planner approves. No actual paper file was modified; the planner requested to hold that rewrite.

## Optional v3P decision: DECLINE FOR NOW, not secretly refit

The **class-race weights fitted to unconditional residual rankings** plus the first four class labels do *not* specify P(rho<=tau | class), the physical-residual distribution conditional on passing, or the conditional orientation-success distribution needed by the stop-at-M policy. The historical A8b residuals and D1 phase-1 CLIP decisions could support an additional, expressly assumed parametric residual model, but it would be a new unvalidated calibration, not a consequence of registered v2/v3. FLAG_O's global transductive quantile additionally requires a joint model of orientation margins across images. Rather than manufacture v3P success/runs intervals from these incomplete inputs, this executor **declines the optional Nov 1 v3P registry item** and prioritizes provable (b)(1)–(3) and the frozen protocol's own held-out H1–H4. No D1-P2, B16 or F8 later-run data are requested or needed for the delivered theory.

**Planner action:** review the S1 proposition and the explicit counterexample before inserting any claim that heterogeneity alone guarantees a protocol gain. Retain the code/contract frozen; use the new exact random-order calculation as a secondary numerical QA, not a modification of preregistered evaluation. Keep symmetry classical and noncentral as Cheng decided.
