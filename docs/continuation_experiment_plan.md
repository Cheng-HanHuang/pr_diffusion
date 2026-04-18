# prdiffusion updated continuation plan after Phase 8/9
## Soft-early / hard-late pivot

This note updates the continuation plan after the Phase 8 and Phase 9 pilots.

The main conceptual change is this:

> The current evidence should not be framed primarily as “late projection is good.”
> A better working hypothesis is:
> **early hard measurement consistency is too aggressive, and it is better to use measurement information softly early and harder later.**

This shifts the continuation from a “late projection portability” plan to a **deferred hard consistency / soft early guidance** plan.

---

## 1. What is now frozen from earlier phases

### 1.1 Keep these defaults unless a phase explicitly changes them
- model: `google/ddpm-celebahq-256`
- dataset: current CelebA-HQ 256 face-prior setup
- standard seeds:
  - pilot: `100,101,102,103,104`
  - full: `100,101,102,103,104,105,106,107,108,109`
- primary radius: `r = 0.5`
- secondary radius: `r = 0.2`

### 1.2 Frozen Noise Picking reference
- `num_steps = 1000`
- `score_radius = 0.5`
- `proj_radius = 0.5`
- `num_candidates_soft = 5`
- `num_candidates_hard = 1`
- `proj_start = 400`
- `use_lowfreq_score = True`
- `use_lowfreq_projection = True`

### 1.3 Frozen SITCOM baseline
- `num_steps = 20`
- `K = 20`
- `lr_inner = 0.02`
- `lam = 0.1`
- `eta_scale = 1.0`
- `init_scale = 1.0`
- `backprop_unet = True`
- `inner_optim = adam`
- baseline form: unmasked

---

## 2. What Phase 8 and Phase 9 now mean

### 2.1 Phase 8 conclusion
The Phase 8 pilot was useful as a branch-pruning probe, but it should not be overinterpreted.

What it established:
- `late_r02` is not promising and should be dropped.
- `maskall_r05` is aggressive and can produce larger wins on a few images, but it is not the safest insertion default.
- `late_r05` is the cleanest surviving SITCOM insertion variant from the Phase 8 probe.

What it did **not** establish strongly:
- a strong PSNR improvement claim for SITCOM
- that “late projection” itself is the full reason for Phase 5 gains

### 2.2 Phase 9 conclusion
The start sweep around `200 / 400 / 600 / 800` was essentially tied on the pilot.

Practical consequence:
- keep `400` as the working default
- do not spend more time on Phase 9 schedule tuning unless a later larger run shows a contradiction

### 2.3 Important interpretation correction
The current SITCOM Phase 8/9 runner does **not** implement the same mechanism as Noise Picking.

It implements:
- full-measurement loss early
- masked low-frequency loss later

So it is a **late loss-switching proxy**, not the full “soft early / hard late” principle.

This is why the next experiments should pivot toward **deferred hard consistency** rather than continuing to overfocus on late projection schedules.

---

## 3. Updated main question

The next block should answer this:

> Is the gain really from delaying hard measurement enforcement and using measurement information softly early?

This should be tested first in Noise Picking, where the distinction is natural, before trying to transfer it to SITCOM or another host.

---

## 4. Revised phase overview

### Phase 10
**Decouple the Noise Picking mechanism**

### Phase 11
**Test the deferred-hard-consistency hypothesis inside Noise Picking**

### Phase 12
**Build a stronger SITCOM transfer surrogate**

### Phase 13
**Only then test a second host**

### Phase 14
**Held-out confirmation of the best deferred-hard variant(s)**

### Phase 15
**Compute / restart / robustness analysis**

---

## 5. Phase 10: decouple the current Noise Picking mechanism

### Purpose
Your current `proj_start` changes more than one thing:
- when low-frequency projection begins
- when the candidate count changes from soft to hard

This phase isolates those effects.

### Questions
1. Is the gain mostly from delaying hard projection?
2. Is it mostly from the candidate-count schedule?
3. Is it their interaction?

### Pilot setting
- split: `validation_10`
- seeds: 5
- radius: `0.5`

### Methods
Use the current canonical NP backbone, but compare:

1. **NP-canonical**
   - `soft=5`
   - `hard=1`
   - `proj_start=400`
   - score on, projection on

2. **NP-fixed-candidates-lateproj**
   - keep candidate count fixed across the whole run, preferably `k=5`
   - projection off early, on late
   - this isolates projection timing

3. **NP-fixed-candidates-alwaysproj**
   - same fixed candidate count
   - projection on from the start

4. **NP-fixed-candidates-noproj**
   - same fixed candidate count
   - projection never used

5. **NP-candidate-switch-only**
   - keep the soft/hard candidate switch
   - projection never used

6. **NP-projection-only-switch**
   - keep candidate count fixed
   - vary only projection start

### Count
- `10 × 5 × 6 = 300`

### Decision rule
This phase succeeds if it tells you whether:
- delayed hard enforcement matters by itself,
- candidate scheduling matters by itself,
- or both are needed.

### Full validation follow-up
Take the best 2–3 settings to:
- `validation_25`
- 10 seeds

---

## 6. Phase 11: directly test “hard early is too strong”

### Purpose
Test the hypothesis more explicitly than Phase 10.

### Pilot setting
- split: `validation_10`
- seeds: 5
- radius: `0.5`

### Methods
Use a fixed candidate count to avoid confounding if possible.

1. **hard-from-start**
   - projection from the first valid step

2. **hard-late**
   - projection starts at `400`

3. **hard-never**
   - no projection at all

4. **soft-only**
   - masked scoring only
   - no hard projection

5. **soft-then-hard**
   - current preferred structure:
     - soft measurement cue early
     - hard enforcement later

### Count
- `10 × 5 × 5 = 250`

### Decision rule
This phase is successful if `soft-then-hard` clearly beats `hard-from-start`.
That would support the new core principle:
- do not force measurement consistency too strongly in the early high-noise stage.

### Full validation follow-up
Run the best 2–3 variants on:
- `validation_25`
- 10 seeds

---

## 7. Phase 12: stronger SITCOM surrogate for the new principle

### Purpose
The current SITCOM late-mask experiment is too weak a proxy.
Phase 12 should test a closer analogue of the soft-early / hard-late idea.

### Important design principle
Do **not** reopen broad SITCOM tuning.
Keep the baseline fixed and only change the measurement-use schedule.

### Candidate surrogate designs
Choose whichever is easiest to implement cleanly.

#### Option A: weak-early / strong-late measurement weight
- early outer steps: reduced weight on measurement term
- late outer steps: normal weight

This tests whether early hard data-fidelity is too strong.

#### Option B: full-measurement early / masked-measurement late
- this is close to the current Phase 8/9 proxy
- keep only as a fallback if Option A is hard to implement

#### Option C: softer low-frequency term early, harder term late
- early: low-weight low-frequency loss
- late: full-strength low-frequency loss

### Recommended method set
1. SITCOM-unmasked baseline
2. SITCOM weak-early / strong-late
3. SITCOM hard-from-start masked
4. SITCOM current late-mask proxy (`r=0.5`, `start=400`) as reference only

### Pilot
- `validation_10`
- 5 seeds

### Full run
- `validation_25`
- 10 seeds

### Decision rule
Only carry a SITCOM insertion forward if it produces either:
- meaningful PSNR gain, or
- very clear consistency gain with negligible quality loss

If not, do not spend more time trying to force a SITCOM transfer story.

---

## 8. Phase 13: second-host testing only after Phase 10/11 are clear

### Purpose
Do not jump to a second host before the mechanism is actually isolated.

The second host should now test:
> does deferred hard consistency transfer?

not merely:
> does late projection transfer?

### Recommended host rule
Use a public codebase with:
- a clear iterative reverse loop
- a clear measurement-update or correction location
- low adaptation cost

### What to insert
Insert only the now-better-defined policy:
- soft measurement use early
- hard measurement enforcement later

### Minimal comparison ladder
For the second host:

1. host baseline
2. host hard-from-start
3. host soft-then-hard
4. host never-hard / soft-only, if natural
5. current NP canonical as reference

### Pilot
- `validation_10`
- 5 seeds

### Full validation
- `validation_25`
- 10 seeds

### Decision rule
If the second host shows the same preference for soft-early / hard-late over hard-from-start, then the plug-and-play story becomes much stronger.

---

## 9. Phase 14: held-out confirmation on test_20

### Purpose
Once the best deferred-hard variants are frozen on validation, confirm them on held-out data.

### Method set
Keep this compact:

1. SITCOM-unmasked
2. best surviving SITCOM insertion, if any
3. second-host baseline
4. second-host soft-then-hard insertion
5. NP canonical
6. best decoupled NP variant from Phase 10/11, if different from canonical

### Settings
- split: `test_20`
- seeds: 10

### Output
Report:
- avg image mean / median / max PSNR
- full mag error
- lowfreq mag error
- runtime
- host-level win rates

---

## 10. Phase 15: compute, restarts, and robustness

### Purpose
After the mechanism is frozen, do the practical evaluation.

### 15.1 Best-of-R curves
Use postprocessing where possible.
Methods:
- SITCOM baseline
- best SITCOM insertion
- second-host baseline
- second-host insertion
- NP canonical
- best NP decoupled variant

### 15.2 Compute-matched comparison
Use `test_20`.
Keep runtimes explicit.

### 15.3 Mild noise robustness
Only after the method choice is frozen.
Use:
- `noise = 0.0, 0.01, 0.03`
or a 2-level version if time is tight.

---

## 11. Recommended PAC launch order

Because PAC has 4 GPUs, do this in waves.

### Wave A: mechanism-isolation pilots
- **GPU 0**: Phase 10 pilot
- **GPU 1**: Phase 11 pilot
- **GPU 2**: Phase 12 pilot
- **GPU 3**: second-host repo setup / smoke tests only

### Wave B: full validation
Only after Wave A gives a clear direction.

- **GPU 0**: Phase 10 full validation
- **GPU 1**: Phase 11 full validation
- **GPU 2**: Phase 12 full validation
- **GPU 3**: second-host pilot if the mechanism is already clear

### Wave C: held-out confirmation
- use `test_20`
- run only frozen best variants

### Wave D: final practical evaluation
- compute, restart, noise

---

## 12. Priority order if time is tight

1. Phase 10
2. Phase 11
3. Phase 12
4. Phase 13
5. Phase 14
6. Phase 15

If time becomes very tight:
- finish Phase 10 and 11 first
- because they determine whether the whole plug-and-play story should be framed as:
  - deferred hard consistency,
  - or something narrower and more host-specific

---

## 13. Summary

The continuation should now pivot from:

- “late low-frequency projection portability”

to:

- **deferred hard consistency / soft early guidance**

The next most informative experiments are therefore:
1. isolate the mechanism inside Noise Picking,
2. test a closer SITCOM surrogate for the same principle,
3. and only then push the mechanism to a second host.
