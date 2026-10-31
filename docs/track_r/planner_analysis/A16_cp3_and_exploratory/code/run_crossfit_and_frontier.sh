#!/usr/bin/env bash
# Planner exploratory runs (A16.2). P = frozen protocol code (c598176). Inputs: XP returns (CP2 ef1cf10, CP3 1981dea).
# Usage: P=/path/gtfree_protocol.py CP2=/dir/with/D4_64RUN_* CP3=/dir/with/CP3_files OUT=/outdir bash run_crossfit_and_frontier.sh
P="python3 -I $P"
for N in 2 3 4 6 8 12 16 24 32 48 64; do
  $P --strings $CP2/D4_64RUN_STRINGS.csv --clip $CP2/D4_64RUN_G1_CLIP.csv --tau 0.0500829 --m 2 --nmax $N --R 4000 --seed 20261029 --out $OUT/D4_nmax$N.json
done
for s in 0p02 0p10; do for c in B22 DEV80; do
  $P --strings $CP3/G2_${s}_STRINGS.csv --tau-rule self_mad3 --m 2 --nmax 8 --R 4000 --seed 20261029 --cohort $c --out $OUT/G2_${s}_selfmad3_$c.json
done; done
# cross-fit: tau from the other cohort (values printed in each own-run JSON field "tau"), e.g.
#   --tau <tau of DEV80 own run> --cohort B22   and   --tau <tau of B22 own run> --cohort DEV80
# G5 halves: cohort := "H" + str(int(sha256(image_id).hexdigest(),16) % 2), then self_mad3 per half and swap.
