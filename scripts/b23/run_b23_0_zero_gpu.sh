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
  mkdir -p "$output_root/logs" || {
    echo "[B23.0] cannot create output log directory: $output_root/logs" >&2
    return 2
  }
  local timestamp stdout_log stderr_log step_results capsule rc evidence_rc
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  stdout_log="$output_root/logs/B23_0_zero_gpu_${timestamp}.stdout.log"
  stderr_log="$output_root/logs/B23_0_zero_gpu_${timestamp}.stderr.log"
  step_results="$output_root/logs/B23_0_zero_gpu_${timestamp}.steps.tsv"
  capsule="$output_root/B23_0_return_${timestamp}"
  rc=0

  {
    echo "timestamp=$timestamp"
    echo "repo=$repo"
    echo "output_root=$output_root"
    echo "python=$python_bin"
    echo "PYTHONPATH=$repo"
    echo "CUDA_VISIBLE_DEVICES=<empty>"
    echo "gpu_work_performed=NO"
  } > "$stdout_log"
  : > "$stderr_log"
  printf 'step\tstatus\treturn_code\n' > "$step_results"

  run_python_step() {
    local step_name="$1"
    local step_rc
    shift
    printf 'step_start=%s\n' "$step_name" >> "$stdout_log"
    (
      cd "$repo" &&
      PYTHONDONTWRITEBYTECODE=1 \
      CUDA_VISIBLE_DEVICES="" \
      PYTHONPATH="$repo" \
      "$python_bin" "$@"
    ) >> "$stdout_log" 2>> "$stderr_log"
    step_rc=$?
    if [[ "$step_rc" -eq 0 ]]; then
      printf '%s\tPASS\t0\n' "$step_name" >> "$step_results"
      printf 'step_pass=%s\n' "$step_name" >> "$stdout_log"
    else
      printf '%s\tFAIL\t%s\n' "$step_name" "$step_rc" >> "$step_results"
      printf 'step_fail=%s return_code=%s\n' "$step_name" "$step_rc" >> "$stderr_log"
    fi
    return "$step_rc"
  }

  run_python_step "unit_tests" \
    -m unittest discover -s tests/b23 -v || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    run_python_step "repository_validation" \
      "$repo/scripts/b23/validate_b23_0.py" \
      --repo "$repo" \
      --output-json "$output_root/B23_0_validation_${timestamp}.json" || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    run_python_step "b23_1_dry_render" \
      "$repo/scripts/b23/render_b23_1_dry_runs.py" \
      --repo "$repo" \
      --output "$output_root/B23_1_dry_run_${timestamp}.json" || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    run_python_step "pac_evidence_collection" \
      "$repo/scripts/b23/collect_b23_0_pac_evidence.py" \
      --repo "$repo" \
      --output-root "$output_root" \
      --timestamp "$timestamp" || rc=$?
  fi

  # Keep runtime evidence inside any complete or partial capsule. Packaging and publication remain
  # gated on rc==0, and the publisher independently requires all four PASS rows.
  if [[ -d "$capsule" ]]; then
    evidence_rc=0
    cp "$step_results" "$capsule/ZERO_GPU_STEP_RESULTS.tsv" || evidence_rc=$?
    if [[ "$evidence_rc" -eq 0 ]]; then
      tail -n 80 "$stdout_log" > "$capsule/STDOUT_TAIL.txt" || evidence_rc=$?
    fi
    if [[ "$evidence_rc" -eq 0 ]]; then
      tail -n 80 "$stderr_log" > "$capsule/STDERR_TAIL.txt" || evidence_rc=$?
    fi
    if [[ "$rc" -eq 0 && "$evidence_rc" -ne 0 ]]; then
      rc=$evidence_rc
    fi
  elif [[ "$rc" -eq 0 ]]; then
    echo "[B23.0] evidence collector returned success without creating $capsule" >> "$stderr_log"
    rc=70
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
    echo "[B23.0] step_results=$step_results"
  else
    echo "[B23.0] STOP rc=$rc; no GPU work was performed" >&2
    echo "[B23.0] stdout_log=$stdout_log" >&2
    echo "[B23.0] stderr_log=$stderr_log" >&2
    echo "[B23.0] step_results=$step_results" >&2
    [[ -d "$capsule" ]] && echo "[B23.0] partial_capsule=$capsule" >&2
  fi
  return "$rc"
}

main "$@"
