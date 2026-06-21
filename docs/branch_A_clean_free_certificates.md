# Branch A clean-free certificates for phase retrieval

Updated: 2026-06-21

This note records the current conceptual interpretation of Branch A after the A11, A14, and A16 prospective validation runs.

Branch A should now be read less as a final-output NP/SITCOM selector and more as a clean-free reliability-controller experiment:

```text
Given several SITCOM-ODE attempts for the same phase-retrieval measurement,
can we detect likely failed runs without using the ground-truth image,
and replace only those runs by a conservative NP-selected fallback?
```

The current answer is positive but incomplete. Frozen clean-free controllers reduce many catastrophic SITCOM failures on fresh trajectories, especially the aggressive residual-plus-consensus OR policy. However, they still do not eliminate the persistent catastrophic floor case associated with image `00017`.

## 1. Why clean-free certificates are needed

In the phase-retrieval setting we do not observe the clean image during reconstruction. Any executable controller therefore needs evidence that can be computed from:

- the measurement and forward operator;
- the diffusion/SITCOM trajectory;
- the candidate reconstructions produced by multiple runs;
- external fallback candidates such as NP-selected outputs;
- but not the ground-truth image or PSNR.

This is why Branch A uses the language of *certificates*. A certificate is not a proof of correctness. It is a clean-free signal that a run is trustworthy or suspicious.

The core problem is that no single certificate is enough:

```text
measurement fit can be good for a wrong reconstruction;
trajectory behavior can look stable before a late collapse;
consensus can miss failures if several runs share a wrong basin or if the failure is not visible in the chosen scale;
NP fallback is safer on catastrophes but has a lower ceiling than successful SITCOM.
```

Branch A therefore became a study of how multiple certificates complement each other.

## 2. Certificate family 1: measurement-side residual behavior

The first useful signal came from measurement-side trajectory residuals. Early absolute thresholds did not transfer well, but relative inter-run features were much more stable.

The most important residual certificate is the late-window inter-run residual-rank rule:

```text
x0y_full_residual_normed__interrun_rank__first80pct__slope
AND
x0y_full_residual_normed__interrun_rank__first80pct__last_in_window
```

The idea is simple:

```text
For the same image and measurement, if one SITCOM run becomes persistently worse than the other runs in measurement-side residual behavior, then it is more likely to be a failure.
```

A12 and A13 showed why the timing matters. The first50 detector was often too early: many A11 misses became risky only after the first50 window. The first80 residual-rank version therefore acts more like late-trajectory triage than early online control.

This distinction matters:

```text
first50 evidence: closer to early warning / possible intervention;
first80 evidence: stronger as a post-run or late-run triage signal;
full-trajectory evidence: clean-free quality classification, not early control.
```

Branch A currently has stronger evidence for late-trajectory triage than for true online intervention.

## 3. Certificate family 2: cross-run low-frequency consensus

A13.5 introduced a second certificate family: cross-run consensus/outlierness.

The clean-free intuition is:

```text
For the same measurement, successful runs should often agree with each other at least at coarse image scale.
A failed run may be an outlier among the four SITCOM reconstructions.
```

The frozen conservative A14 policy uses exactly this idea:

```text
policy_name: consensus_lowfreq_nn
feature: lowfreq_dist_to_nearest_neighbor
rule: flag if low-frequency nearest-neighbor distance is high
```

This certificate is useful because it is not just another measurement residual. It asks whether a run lands in a different reconstruction basin from the other runs.

However, A16 showed that consensus-only is not stable enough as the main practical controller. In A14 it reduced bad25/bad20 from `20/19` to `2/2`; in A16 it reduced `21/16` to only `7/4`. The low-frequency consensus certificate is therefore real but insufficient by itself.

## 4. Certificate family 3: residual plus consensus OR

The strongest practical Branch A controller is now the aggressive frozen policy:

```text
policy_name: residual_or_lowfreq_nn
rule: residual_first80_certificate OR lowfreq_nearest_neighbor_consensus_certificate
```

Its structure is:

```text
residual arm:
  x0y_full_residual_normed__interrun_rank__first80pct__slope >= 0.8767080745341616
  AND
  x0y_full_residual_normed__interrun_rank__first80pct__last_in_window >= 3.0

consensus arm:
  lowfreq_dist_to_nearest_neighbor >= 27.855274200439453

combined rule:
  residual arm OR consensus arm
```

The OR rule is important. Residual and consensus features catch different failure modes. Requiring both would be high precision but would miss many failures. The OR rule is the safety-net version: it accepts a modest false-positive cost to catch more catastrophes.

Across two fresh prospective runs, this aggressive frozen policy replicated strongly:

| run | SITCOM-only bad25/bad20 | aggressive bad25/bad20 | replacements | false-positive replacements |
|---|---:|---:|---:|---:|
| A14 | `20 / 19` | `1 / 1` | `21` | `2` |
| A16 | `21 / 16` | `1 / 1` | `24` | `4` |

This is the main Branch A empirical result so far:

```text
A frozen clean-free residual+consensus controller generalized to two fresh prospective runs and reduced roughly twenty bad25 SITCOM failures to one remaining bad run.
```

## 5. What the controller is actually doing

The controller is not an oracle selector. It does not choose the better of SITCOM and NP by PSNR.

It is also not yet a true online repair method. It currently works as a late/post-run replacement controller:

```text
1. run multiple SITCOM attempts;
2. compute clean-free trajectory and consensus features;
3. flag suspicious SITCOM runs;
4. replace flagged runs by the pre-existing NP-selected fallback;
5. report the resulting controlled candidate set.
```

This is why `replace_all_np_selected` is always reported as a degenerate diagnostic baseline, not as a valid Branch A solver. Replace-all NP can remove SITCOM catastrophes, but it discards SITCOM's higher ceiling on successful runs.

The useful Branch A behavior is selective replacement:

```text
keep most successful SITCOM runs;
replace the runs that violate clean-free certificates;
pay a small false-positive cost for a large reduction in catastrophic failures.
```

## 6. The persistent image 00017 failure

A15 and A16 sharpened the remaining limitation.

In A14, the aggressive policy missed `image 00017/run0`.
In A16, it missed `image 00017/run1`.

The run index changed, but the image did not. This suggests an image-specific failure mode rather than a purely random one-run accident.

A15 diagnosed the A14 case as follows:

- the SITCOM reconstruction had PSNR `5.084`;
- the NP-selected fallback had PSNR `27.185`;
- both frozen A14 policies failed to flag it;
- the run looked like a strong pixel-space outlier;
- but it did not cross the frozen low-frequency nearest-neighbor threshold;
- and it did not trigger the first80 residual-rank certificate.

This is the current bottleneck:

```text
Some catastrophic failures can evade both the measurement-side residual certificate and the low-frequency consensus certificate.
```

It would be easy to tune a new feature on image `00017`, for example a pixel-space or perceptual consensus feature. But doing that immediately would risk overfitting to a known prospective miss. The cleaner interpretation is that image `00017` exposes the next certificate family that should be studied in a new development cycle, not patched into the frozen A14/A16 policy after the fact.

## 7. Candidate future certificates

The natural next certificates are not new thresholds on the same A14 data. They are new certificate families that should be developed and validated in a fresh cycle.

Potential directions:

### 7.1 Pixel or perceptual consensus

A15 suggests that the remaining miss can be visually/pixel-space isolated even when the low-frequency nearest-neighbor certificate does not fire.

Possible features:

- pixel-space nearest-neighbor distance among the four SITCOM runs;
- distance to the image-wise median reconstruction;
- low-frequency plus high-frequency split features;
- perceptual distances such as LPIPS-like features if available;
- feature-space distance using an encoder, if one can be kept clean-free and reproducible.

### 7.2 Temporal consistency of consensus

A13.5 used final saved samples because raw per-step tensors were not available for consensus analysis. A stronger version would ask whether a run becomes an outlier over time.

Possible features:

- nearest-neighbor distance at several late timesteps;
- growth rate of consensus distance;
- whether a run separates from the majority basin after a particular sigma range;
- consistency of the final output with its own earlier trajectory states.

### 7.3 Agreement with NP fallback

The NP-selected fallback is not ground truth, but it is a conservative independent candidate. Distance between SITCOM and NP fallback may provide another clean-free certificate.

Possible features:

- low-frequency SITCOM-to-NP distance;
- pixel/perceptual SITCOM-to-NP distance;
- whether the flagged SITCOM run is isolated both from other SITCOM runs and from NP.

This must be used carefully: NP is not always better than SITCOM, and a successful high-quality SITCOM run may differ from NP. The useful signal may be not raw distance, but distance combined with residual instability or cross-run isolation.

### 7.4 Multi-candidate decision rather than binary replacement

The current controller makes a binary per-run decision: keep SITCOM or replace by NP.

A more principled solver could return or score a small candidate set:

```text
successful SITCOM candidates;
NP fallback candidate;
possibly an ensemble/median candidate;
clean-free certificates for each candidate.
```

This may be more honest for ill-posed inverse problems, where a single scalar certificate may not fully determine correctness.

## 8. What should not be claimed yet

Branch A should not yet claim:

- elimination of all catastrophic failures;
- a lifted run-level PSNR floor;
- a true online intervention method;
- a universally valid certificate of phase-retrieval correctness;
- robustness beyond `sigma_y = 0.05` unless separately tested.

The current validated claim is narrower and stronger:

```text
At sigma_y = 0.05 on the FFHQ-25 setup, frozen clean-free controllers can substantially reduce catastrophic SITCOM failures on fresh trajectories. The most robust current controller is a residual+consensus OR rule, validated prospectively in A14 and A16.
```

## 9. Recommended next steps

The recommended next steps are:

1. Stop further policy tuning at `sigma_y = 0.05` unless explicitly starting a new development cycle.
2. Treat `residual_or_lowfreq_nn` as the current best practical Branch A controller.
3. Write the method in terms of clean-free certificates rather than posthoc feature mining.
4. Use image `00017` as the motivating example for why the current certificates are incomplete.
5. If continuing empirically, run a separate out-of-distribution stress test at `sigma_y = 0.08` using the same frozen aggressive policy, with no threshold retuning.
6. If starting a new development cycle, predeclare a new candidate certificate family such as pixel/perceptual consensus and validate it on fresh trajectories.

## 10. Short research summary

A concise summary of the current Branch A state is:

```text
Branch A produced a prospectively validated clean-free controller for diffusion-prior phase retrieval.
The best current policy combines a late-window residual-rank certificate with a low-frequency cross-run consensus certificate by OR.
Across A14 and A16, this reduced roughly twenty bad25 SITCOM failures to one remaining bad run, with modest false-positive cost.
The remaining limitation is a persistent image-specific catastrophic case, image 00017, which escapes the current certificates and motivates future work on pixel/perceptual consensus or stronger candidate-set certification.
```
