# B21.8 independent-restart budget decision

The frozen Fresh1--Fresh4 development curve completed on 80 cases with no failures.

## Curve

```text
K=1: selected 56/80, oracle 56/80
K=2: selected 70/80, oracle 70/80
K=3: selected 76/80, oracle 77/80
K=4: selected 77/80, oracle 78/80
```

The frozen development target required:

```text
selected good25 >= 76/80
selected-oracle gap <= 1
cumulative selected harms <= 1
every image selected good25 >= 6/8
```

Fresh3 is the smallest qualifying budget:

```text
selected good25:       76/80
oracle good25:         77/80
selected-oracle gap:    1
cumulative harms:       0
minimum image result:  6/8
```

Fresh4 adds only one selected success and one oracle success beyond Fresh3. It is not chosen because the preregistered rule selects the smallest qualifying K.

## Decision

Freeze the following policy for one disjoint validation:

- three independent full DAPS trajectories;
- `ann400`, `diff5` for every trajectory;
- no LF or HIO arm;
- sequential exact-loss selector with margin `theta=0.7`;
- interpret cost as three full-trajectory equivalents.

No threshold, schedule, seed, or reliability target may be changed after this result.
