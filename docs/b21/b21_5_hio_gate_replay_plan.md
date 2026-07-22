# B21.5 HIO auxiliary-arm gate replay plan

Status: analyzer ready; no GPU work required.

## Interpretation after the five-image pilot

The frozen `HIO + DAPS(200:400)` replacement policy is rejected:

- base good25: `16/40`;
- warm good25: `10/40`;
- held-out net: `-5`;
- mean PSNR delta: `-4.04 dB`.

However, the paired rows contain five HIO-only good25 candidates. Therefore this replay asks a narrower question: can the warm candidate be used as a cheap auxiliary arm and accepted only when a clean-free exact-loss gate prefers it?

## Frozen exploratory rule

```text
select HIO iff
  exact_operator_loss_hio < exact_operator_loss_base - 0.7
```

The `0.7` margin is inherited unchanged from the frozen B21.4 LF gate. It is not tuned on the HIO panel. PSNR is used only after selection for diagnostic evaluation.

## Support gate

Promote only an LF-extension development experiment when all hold:

```text
gated net good25 versus base >= +4 / 40
gated harms <= 1 / 40
HELDOUT4 gated net good25 >= 0
marginal HIO arm wall ratio <= 0.70 x base
```

A pass does not validate HIO. It justifies generating matched LF050 candidates on these same 40 development cases to determine whether HIO adds anything beyond the frozen base+LF portfolio.

If that three-arm development experiment is positive, freeze the policy and use entirely fresh images/seeds for validation.
