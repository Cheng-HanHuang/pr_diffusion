#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
PATCHER="$REPO/scripts/b21/apply_b21_3_continuation_patch.py"
SAMPLER="$REPO/external/daps/sampler.py"
POSTERIOR="$REPO/external/daps/posterior_sample.py"
PATCH_OUT="$REPO/docs/b21/patches/daps_b21_continuation.patch"

cd "$REPO"

for f in "$PATCHER" "$SAMPLER" "$POSTERIOR"; do
  [[ -f "$f" ]] || { echo "[fatal] missing $f" >&2; exit 2; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }

TMP=$(mktemp -d /tmp/b21_3_patch.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
cp "$SAMPLER" "$TMP/sampler.py.before"
cp "$POSTERIOR" "$TMP/posterior_sample.py.before"

"$PYTHON_BIN" "$PATCHER" --repo "$REPO" --check
"$PYTHON_BIN" "$PATCHER" --repo "$REPO" --apply

# The continuation helper uses pathlib.Path.  Insert the import only when absent.
if ! grep -q '^from pathlib import Path$' "$SAMPLER"; then
  "$PYTHON_BIN" - "$SAMPLER" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text()
anchor = "import os\n"
if text.count(anchor) != 1:
    raise RuntimeError(f"expected one import-os anchor, found {text.count(anchor)}")
p.write_text(text.replace(anchor, anchor + "from pathlib import Path\n", 1))
PY
fi

"$PYTHON_BIN" -m py_compile "$SAMPLER" "$POSTERIOR"

mkdir -p "$(dirname "$PATCH_OUT")"
{
  diff -u --label a/sampler.py --label b/sampler.py \
    "$TMP/sampler.py.before" "$SAMPLER" || [[ $? -eq 1 ]]
  diff -u --label a/posterior_sample.py --label b/posterior_sample.py \
    "$TMP/posterior_sample.py.before" "$POSTERIOR" || [[ $? -eq 1 ]]
} > "$PATCH_OUT"

if [[ ! -s "$PATCH_OUT" ]]; then
  echo "[info] no new diff: B21.3 patch was already present"
else
  echo "[write] $PATCH_OUT"
  echo "[diff lines] $(wc -l < "$PATCH_OUT")"
fi

grep -n 'B21.3 continuation\|B21_CONT_ENABLE\|B21_SAVE_STATE_STEPS' \
  "$SAMPLER" "$POSTERIOR" | head -80

echo "[done] B21.3 continuation patch applied and syntax-checked"
