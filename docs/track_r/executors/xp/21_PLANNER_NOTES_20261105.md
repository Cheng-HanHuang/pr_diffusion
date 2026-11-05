# Planner notes to the XP executor, round 21 (2026-11-05)

**Responds to:** no new return. This round records a paper-only change.

**Governing amendment:** prereg **A24** (unchanged). Nothing in this note changes a gate, a budget or a hypothesis.

## 1. What changed in the paper

The planner did a writing pass on the main text and a defect sweep, at Cheng's request:
- every reference verified and completed;
- every number traced to `CLAIMS_LEDGER.md`;
- notation made explicit;
- internal stage names removed from the paper text.

Two planner checks on committed data were added to the ledger:
- **C-G9:** both DAPS-4K protocol panels (G4, G5) used the label-free threshold `self_mad3` (τ 0.04759 and 0.04821), read from your CP3 JSONs at `1981dea`. The paper now says that no step of the protocol on these panels uses a label.
- **C-E1:** PC-DAPS on ImageNet (XP-I2), oracle gap 2.0 = 0 selection + 2.0 orientation, from `XP_I2_A15_5_PER_PARTICLE.csv` at `1eae9bf`. E1 53 and O_orbit 55 reproduce C-X5.

## 2. Names the paper uses

Please use these in human-readable summaries and receipt prose from now on. File names, JSON keys and code stay as they are.

| Internal | Paper |
|---|---|
| DAPS 400 × 10 | DAPS-4K |
| DAPS 400 × 5 (B22/DEV80 historical) | DAPS-2K |
| B22 / DEV80 | cohort A / cohort B |
| XP-N images | ImageNet′ |
| D1/D2/D4 image set | ImageNet |
| PC-DAPS bag | PC-DAPS sample (four particles) |

## 3. What remains (unchanged from round 20)

1. **XP-K,** under the three A23.5b gates:
   - a committed diff showing exactly one changed parameter;
   - an identity replay of two B22 roots per family;
   - DAPS only after the F8 replay gate.

   **Stop and ask if a gate fails.**
2. **CP4** by Nov 15, 12:00 ET.

Standing rules are unchanged:
- Cheng runs every PAC block, from pushed, clean commits.
- Co-tenancy per A10.1; timing only on clean GPUs.
- No B24 confirmation access of any kind.
- No bundled FFHQ sets.
- No merges of `main` or the audit branch.

## Planner's view

- **Current results.** The main text is complete and clean at 8 pages. The supplement is 22 pages; the theory sections get a naming pass by Nov 12. No main-text number depends on pending work.
- **Plan.**
  - XP-K and CP4 feed the supplement only.
  - Go/no-go on Nov 8 (planner recommends GO).
  - Registration on Nov 10; submission on Nov 16; supplement on Nov 23.
- **Image of the publication.** A CVPR audit with theory. Ground-truth-free selection nearly matches the orbit oracle on FFHQ. With four DAPS runs on ImageNet, it beats the published per-metric oracle. Choosing among PC-DAPS particles adds 5.4 points on FFHQ and 22 on ImageNet, at no extra cost. A frozen stopping protocol gains 3.4–3.9 points for DAPS. What no selection fixes are the images on which no run succeeds.
