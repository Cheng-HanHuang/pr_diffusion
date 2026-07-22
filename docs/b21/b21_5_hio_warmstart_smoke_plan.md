# B21.5 HIO warm-start smoke plan

Status: implementation ready; GPU smoke not yet executed.

## Motivation

The equal-cost step-200 branching policy was rejected because late branches remained correlated with a bad shared prefix, especially on image `00171`. B21.5 instead constructs a measurement-informed clean state before entering DAPS.

## Clean-free construction

The HIO generator reads only:

- the locked noisy Fourier-magnitude measurement;
- the known centered `256 x 256` support inside the `384 x 384` oversampled grid;
- fixed HIO hyperparameters and a declared random seed.

It does not read ground truth. Ground truth is used only by the offline checker for PSNR.

The generated image in `[0,1]` is converted to DAPS model range `[-1,1]` and saved as a B21.3-compatible continuation payload at global annealing step 200. DAPS then adds step-200 noise and performs transitions 200--399.

## Frozen smoke configuration

```text
image: 00046
measurement: meas5001
cases: 3
base seeds: 7300, 7301, 7302
HIO seeds: 8300, 8301, 8302
warm DAPS noise seeds: 9300, 9301, 9302
HIO iterations: 240
HIO beta: 0.9
ER projection every: 20 iterations
final ER cleanup: 10 iterations
DAPS: ann400, diff5
injection step: 200
```

Each case compares one full base run against one HIO generation plus a 200-step DAPS continuation.

## Smoke gates

Implementation passes when:

- all three HIO payloads are finite, shape-correct, bounded in `[-1,1]`, and tagged with step 200;
- all base and warm DAPS runs complete;
- at least two distinct warm-output hashes are produced.

Promotion to the five-image pilot additionally requires:

- mean `(HIO generation + warm DAPS) / full base DAPS` wall ratio at most `0.70`;
- either at least one additional `PSNR >= 25` case or mean warm-start PSNR gain at least `+1.0 dB` on this three-case smoke.

The smoke gate is only a calibration rule; it is not final evidence for WARM_hio.
