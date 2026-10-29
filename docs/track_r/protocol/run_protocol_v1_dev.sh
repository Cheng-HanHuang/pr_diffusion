#!/usr/bin/env bash
# Reproduce the protocol-v1 development results. Inputs (SHA-256 checked):
#   FAST64 = A9_FAST_64_PER_IMAGE_STRINGS.csv   d616dd50f10352e98b891a99b9a153d5d72a93c61e44381827147dcca23f5466
#   A8B    = R0N2_A8B_PER_IMAGE.csv              e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901
#   D1P1   = XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv  c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1
#   D1CLIP = XP_D1_P1_K4_CLIP_DECISIONS.csv      0a2989180b67afa6fc9c6fccdffe7680030f4214d7b672ecd4fe50e5b010cf44
# Usage: FAST64=... A8B=... D1P1=... D1CLIP=... OUT=dir bash run_protocol_v1_dev.sh
PY=${PY:-python3}; HERE=$(cd "$(dirname "$0")" && pwd); P="$PY -I $HERE/gtfree_protocol.py"
ok=1
for pair in "$FAST64 d616dd50f10352e98b891a99b9a153d5d72a93c61e44381827147dcca23f5466" "$A8B e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901" \
            "$D1P1 c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1" "$D1CLIP 0a2989180b67afa6fc9c6fccdffe7680030f4214d7b672ecd4fe50e5b010cf44"; do
  set -- $pair; h=$(sha256sum "$1" | cut -d' ' -f1); if [ "$h" != "$2" ]; then echo "SHA MISMATCH $1"; ok=0; fi
done
if [ $ok = 1 ]; then
mkdir -p "$OUT"; $PY -I $HERE/make_dev_inputs.py "$A8B" "$OUT"
# primary configuration (M=2, N_MAX = panel size, frozen tau) and the label-free threshold, per panel
$P --strings "$FAST64" --tau 0.0500829 --m 2 --nmax 64 --R 4000 --out "$OUT/fast64_m2_n64.json"
$P --strings "$FAST64" --tau 0.0500829 --m 2 --nmax 16 --R 4000 --out "$OUT/fast64_m2_n16.json"
$P --strings "$FAST64" --tau-rule self_mad3 --m 2 --nmax 64 --R 4000 --out "$OUT/fast64_self_m2_n64.json"
$P --strings "$FAST64" --tau 0.0500829 --m 2 --nmax 64 --R 1000 --frontier --out "$OUT/fast64_frontier_n64.json"
$P --strings "$FAST64" --tau 0.0500829 --m 2 --nmax 16 --R 1000 --frontier --out "$OUT/fast64_frontier_n16.json"
$P --strings "$D1P1" --clip "$D1CLIP" --tau 0.0500829 --m 2 --nmax 4 --R 4000 --frontier --out "$OUT/in_d1p1_m2_n4.json"
$P --strings "$D1P1" --clip "$D1CLIP" --tau-rule self_mad3 --m 2 --nmax 4 --R 4000 --out "$OUT/in_d1p1_self_m2_n4.json"
$P --strings "$OUT/dev_DAPS_prefix4.csv" --tau 0.0475986 --m 2 --nmax 4 --R 4000 --frontier --out "$OUT/daps_dev80_m2_n4.json"
$P --strings "$OUT/dev_SITCOM_prefix4.csv" --cohort S1_b22 --tau 0.0448503 --m 2 --nmax 4 --R 4000 --frontier --out "$OUT/sitcom_b22_m2_n4.json"
$P --strings "$OUT/dev_SITCOM_prefix4.csv" --cohort S2_b24_dev80 --tau 0.0449706 --m 2 --nmax 4 --R 4000 --frontier --out "$OUT/sitcom_dev80_m2_n4.json"
$P --strings "$OUT/dev_NP_prefix4.csv" --tau 0.0504971 --m 2 --nmax 4 --R 4000 --frontier --out "$OUT/np_dev80_m2_n4.json"
fi
