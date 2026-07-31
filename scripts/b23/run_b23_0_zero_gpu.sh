#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo="" output_root="" publish=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --repo) repo="$2"; shift 2 ;;
      --output-root) output_root="$2"; shift 2 ;;
      --publish) publish=1; shift ;;
      *) echo "[B23.0] unsupported argument: $1" >&2; return 64 ;;
    esac
  done
  [[ -n "$repo" && -n "$output_root" ]] || {
    echo "usage: $0 --repo /egr/research-pac/huang248/pr_diffusion_b23 --output-root /egr/research-pac/huang248/outputs/pr_diffusion/b23 [--publish]" >&2
    return 64
  }
  local python_bin="/egr/research-pac/huang248/conda-envs/daps/bin/python"
  [[ -x "$python_bin" ]] || { echo "[B23.0] missing DAPS Python: $python_bin" >&2; return 2; }
  [[ "$repo" == /egr/research-pac/huang248/* ]] || { echo "[B23.0] repo must be under /egr/research-pac/huang248" >&2; return 2; }
  [[ "$output_root" == /egr/research-pac/huang248/* ]] || { echo "[B23.0] output root must be under /egr/research-pac/huang248" >&2; return 2; }
  mkdir -p "$output_root/logs"
  local timestamp stdout_log stderr_log capsule rc
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  stdout_log="$output_root/logs/B23_0_zero_gpu_${timestamp}.stdout.log"
  stderr_log="$output_root/logs/B23_0_zero_gpu_${timestamp}.stderr.log"
  capsule="$output_root/B23_0_return_${timestamp}"
  rc=0
  {
    echo "timestamp=$timestamp"
    echo "repo=$repo"
    echo "output_root=$output_root"
    echo "python=$python_bin"
    echo "CUDA_VISIBLE_DEVICES=<empty>"
    echo "gpu_work_performed=NO"
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" -m unittest discover -s "$repo/tests/b23" -v
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" "$repo/scripts/b23/validate_b23_0.py" --repo "$repo" --output-json "$output_root/B23_0_validation_${timestamp}.json"
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" "$repo/scripts/b23/render_b23_1_dry_runs.py" --repo "$repo" --output "$output_root/B23_1_dry_run_${timestamp}.json"
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" "$repo/scripts/b23/collect_b23_0_pac_evidence.py" --repo "$repo" --output-root "$output_root" --timestamp "$timestamp"
  } > "$stdout_log" 2> "$stderr_log" || rc=$?
  if [[ -d "$capsule" ]]; then
    tail -n 80 "$stdout_log" > "$capsule/STDOUT_TAIL.txt"
    tail -n 80 "$stderr_log" > "$capsule/STDERR_TAIL.txt"
  fi
  if [[ "$rc" -eq 0 ]]; then
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" "$repo/scripts/b23/package_b23_0_return.py" --capsule "$capsule" >> "$stdout_log" 2>> "$stderr_log" || rc=$?
  fi
  if [[ "$rc" -eq 0 && "$publish" -eq 1 ]]; then
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" "$repo/scripts/b23/publish_b23_0_evidence.py" --repo "$repo" --capsule "$capsule" --commit-archive --push >> "$stdout_log" 2>> "$stderr_log" || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    echo "[B23.0] PASS zero-GPU validation and PAC freeze"
    echo "[B23.0] capsule=$capsule"
    echo "[B23.0] archive=${capsule}.tar.gz"
    echo "[B23.0] checksum=${capsule}.tar.gz.sha256"
    [[ "$publish" -eq 1 ]] && echo "[B23.0] publish_result=${capsule}_PUBLISH_RESULT.json"
    echo "[B23.0] stdout_log=$stdout_log"
    echo "[B23.0] stderr_log=$stderr_log"
  else
    echo "[B23.0] STOP rc=$rc; no GPU work was performed" >&2
    echo "[B23.0] stdout_log=$stdout_log" >&2
    echo "[B23.0] stderr_log=$stderr_log" >&2
    [[ -d "$capsule" ]] && echo "[B23.0] partial_capsule=$capsule" >&2
  fi
  return "$rc"
}

main "$@"
