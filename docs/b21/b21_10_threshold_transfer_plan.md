# B21.10 cross-panel hard-case threshold transfer

Status: zero-GPU audit ready; no fallback GPU run authorized yet.

## Motivation

On B21.9, the six Fresh2 failures are exactly the six highest Fresh2 selected-loss rows. The four persistent `66731` failures occupy ranks 1--4. This is a strong within-panel signal, but a runtime trigger must not be tuned and judged on the same panel.

## Transfer protocol

Use the completed B21.8 FreshK panel as development and B21.9 as validation.

Derive one absolute threshold using development rows only:

```text
threshold = minimum Fresh2 selected loss among development Fresh2 failures
```

This is the largest threshold retaining 100% recall of development failures. Apply it unchanged to B21.9.

Report development and validation:

- Fresh2 failure AUC;
- number and fraction flagged;
- failure recall, precision, and specificity;
- flagged images and image-level loss summaries.

## Interpretation

The audit is retrospective because B21.9 outcomes have already been viewed. It does not validate an adaptive policy.

A strong transfer signal requires validation recall at least `0.80`, precision at least `0.50`, and flagged fraction at most `0.25`. This only authorizes a small targeted complementary-candidate development pilot, with an ordinary third restart retained as a matched control. Both detector and fallback would still require validation on another disjoint panel.

A weak transfer result means no fallback GPU run should begin until a better clean-free detector is developed.
