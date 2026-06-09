#!/usr/bin/env bash
set -euo pipefail

# Prepare the dated experiment output tree and split files.
# Safe to rerun.

OUT_ROOT="${OUT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608}"
OLD_SPLIT_ROOT="${OLD_SPLIT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/splits" "$OUT_ROOT/logs" "$OUT_ROOT/slurm" "$OUT_ROOT/manifests"

for split in ffhq_available25.txt imagenet_available25.txt; do
  if [[ -f "$OLD_SPLIT_ROOT/$split" ]]; then
    cp -n "$OLD_SPLIT_ROOT/$split" "$OUT_ROOT/splits/$split"
  else
    echo "[warn] missing old split: $OLD_SPLIT_ROOT/$split" >&2
  fi
done

# Create a hard-image split by index from the existing FFHQ-25 split.
# Historical hard images: 00028, 00005, 00013, 00034, 00027, 00007, 00000;
# later attention: 00004, 00025.  We select by substring so it works whether
# the split lines are basenames or relative paths.
OUT_ROOT_ENV="$OUT_ROOT" python - <<'PY'
from pathlib import Path
import os

out_root = Path(os.environ["OUT_ROOT_ENV"])
ffhq = out_root / "splits" / "ffhq_available25.txt"
hard_ids = ["00028", "00005", "00013", "00034", "00027", "00007", "00000", "00004", "00025"]
if ffhq.exists():
    lines = [ln.strip() for ln in ffhq.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    selected = []
    for hid in hard_ids:
        hit = next((ln for ln in lines if hid in Path(ln).stem or hid in ln), None)
        if hit is not None and hit not in selected:
            selected.append(hit)
    (out_root / "splits" / "ffhq_hard9_from_available25.txt").write_text("\n".join(selected) + ("\n" if selected else ""))
    print(f"[prepare] wrote hard split with {len(selected)} images: {out_root / 'splits' / 'ffhq_hard9_from_available25.txt'}")
else:
    print(f"[warn] FFHQ split not found: {ffhq}")
PY

cat > "$OUT_ROOT/manifests/paths.txt" <<EOF
OUT_ROOT=$OUT_ROOT
FFHQ_DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
IMAGENET_DATA_ROOT=/egr/research-pac/huang248/data/imagenet/imagenet256_val
IMAGENET_RAW_ROOT=/egr/research-pac/huang248/data/imagenet/raw_val
DIFFFPR_ROOT=/egr/research-pac/huang248/external/DiffFPR
SITCOM_ODE_ROOT=/egr/research-pac/huang248/external/SITCOM_ODE
LOCAL_REPO=/egr/research-pac/huang248/pr_diffusion_repo
GUIDED_MODEL_PATH=/egr/research-pac/huang248/models/ffhq_10m.pt
EOF

echo "[prepare] ready: $OUT_ROOT"
find "$OUT_ROOT" -maxdepth 2 -type f | sort
