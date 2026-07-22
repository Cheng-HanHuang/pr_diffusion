# B21.9 Fresh3 validation decision

The frozen three-restart policy completed on 20 untouched official FFHQ validation images, with one locked measurement and four trajectory-seed cases per image.

## Result

```text
Fresh1 selected good25: 67/80
Fresh2 selected good25: 74/80
Fresh3 selected good25: 75/80

Fresh3 incremental rescues: 1
Fresh3 incremental harms:   0
Fresh3 positive images:     1/20
Fresh3 negative images:     0/20
```

Fresh3 met the overall `74/80` reliability floor and had no selector harms, but failed the preregistered incremental-value and image-spread gates. The third full trajectory added only one success beyond Fresh2 and did so on only one image.

The failure is not caused by the clean-free selector. Selected and oracle counts match at every K:

```text
Fresh1: 67 selected / 67 oracle
Fresh2: 74 selected / 74 oracle
Fresh3: 75 selected / 75 oracle
```

Thus the third trajectory itself rarely adds a new good basin on this panel.

## Hard-case structure

Image `66731` remained `0/4` after Fresh1, Fresh2, and Fresh3. This accounts for four of the five remaining Fresh3 failures. Image `67673` improved from `2/4` under Fresh2 to `3/4` under Fresh3 and contains the only third-restart rescue.

This pattern argues against further blind restart scaling. Most images are already solved by Fresh2, while one image-level measurement remains resistant across all twelve trajectories represented by its four cases and three arms.

## Decision

- Adopt **Fresh2** as the default fixed restart budget.
- Reject **Fresh3** as a default fixed budget.
- Do not run Fresh4 or larger blind-restart validation.
- Preserve the Fresh3 outputs for hard-case analysis.
- Next run a zero-GPU clean-free triage audit on the completed Fresh2/Fresh3 rows. Only after that audit should GPU work resume, focused on a complementary candidate for persistent hard measurements rather than another ordinary restart.
