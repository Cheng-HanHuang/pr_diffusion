#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo="" output_root="" source_capsule="" source_archive="" expected_head="" publish=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --repo) repo="$2"; shift 2 ;;
      --output-root) output_root="$2"; shift 2 ;;
      --source-capsule) source_capsule="$2"; shift 2 ;;
      --source-archive) source_archive="$2"; shift 2 ;;
      --expected-head) expected_head="$2"; shift 2 ;;
      --publish) publish=1; shift ;;
      *) printf '[B23.1] unsupported argument: %s\n' "$1" >&2; return 64 ;;
    esac
  done
  [[ -n "$repo" && -n "$output_root" && -n "$source_capsule" && -n "$source_archive" && -n "$expected_head" ]] || {
    printf 'usage: %s --repo REPO --output-root ROOT --source-capsule DIR --source-archive TAR --expected-head SHA [--publish]\n' "$0" >&2
    return 64
  }
  local python_bin="/egr/research-pac/huang248/conda-envs/daps/bin/python"
  [[ -x "$python_bin" ]] || { printf '[B23.1] missing DAPS Python\n' >&2; return 2; }
  [[ "$(git -C "$repo" branch --show-current)" == "codex/b23-execution" ]] || { printf '[B23.1] wrong branch\n' >&2; return 3; }
  [[ "$(git -C "$repo" rev-parse HEAD)" == "$expected_head" ]] || { printf '[B23.1] wrong pre-run head\n' >&2; return 3; }
  [[ -z "$(git -C "$repo" status --porcelain)" ]] || { printf '[B23.1] dirty worktree\n' >&2; return 3; }
  [[ -d "$source_capsule" && -f "$source_archive" ]] || { printf '[B23.1] accepted evidence source missing\n' >&2; return 4; }

  local timestamp capsule stdout_log stderr_log steps rc evidence_rc
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  capsule="$output_root/B23_1_closeout_return_$timestamp"
  mkdir -p "$output_root/logs" || return 5
  stdout_log="$output_root/logs/B23_1_closeout_${timestamp}.stdout.log"
  stderr_log="$output_root/logs/B23_1_closeout_${timestamp}.stderr.log"
  steps="$output_root/logs/B23_1_closeout_${timestamp}.steps.tsv"
  rc=0
  printf 'step\tstatus\treturn_code\n' > "$steps"
  {
    printf 'timestamp=%s\nrepo=%s\nsource_capsule=%s\nsource_archive=%s\n' "$timestamp" "$repo" "$source_capsule" "$source_archive"
    printf 'pre_run_commit=%s\nCUDA_VISIBLE_DEVICES=<empty>\ngpu_work_performed_during_correction=NO\n' "$expected_head"
  } > "$stdout_log"
  : > "$stderr_log"

  run_step() {
    local name="$1" step_rc
    shift
    (cd "$repo" && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH="$repo" "$python_bin" "$@") >>"$stdout_log" 2>>"$stderr_log"
    step_rc=$?
    if [[ "$step_rc" -eq 0 ]]; then
      printf '%s\tPASS\t0\n' "$name" >> "$steps"
    else
      printf '%s\tFAIL\t%s\n' "$name" "$step_rc" >> "$steps"
    fi
    return "$step_rc"
  }

  run_step unit_tests -m unittest discover -s tests/b23 -v || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    run_step repository_validation "$repo/scripts/b23/validate_b23_0.py" --repo "$repo" --output-json "$output_root/B23_1_closeout_validation_${timestamp}.json" || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    run_step accepted_evidence_validation "$repo/scripts/b23/collect_b23_1_closeout.py" --repo "$repo" --source-capsule "$source_capsule" --source-archive "$source_archive" --capsule "$capsule" --pre-run-head "$expected_head" || rc=$?
  fi
  if [[ -d "$capsule" ]]; then
    evidence_rc=0
    cp "$steps" "$capsule/ZERO_GPU_STEP_RESULTS.tsv" || evidence_rc=$?
    tail -n 100 "$stdout_log" > "$capsule/STDOUT_TAIL.txt" || evidence_rc=$?
    tail -n 100 "$stderr_log" > "$capsule/STDERR_TAIL.txt" || evidence_rc=$?
    [[ "$rc" -ne 0 || "$evidence_rc" -eq 0 ]] || rc=$evidence_rc
  elif [[ "$rc" -eq 0 ]]; then
    rc=70
  fi
  if [[ "$rc" -eq 0 ]]; then
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" "$repo/scripts/b23/package_b23_1_closeout.py" --capsule "$capsule" >>"$stdout_log" 2>>"$stderr_log" || rc=$?
  fi
  if [[ "$rc" -eq 0 && "$publish" -eq 1 ]]; then
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" "$repo/scripts/b23/publish_b23_1_closeout.py" --repo "$repo" --capsule "$capsule" --push >>"$stdout_log" 2>>"$stderr_log" || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    printf '[B23.1] PASS compact zero-GPU evidence closeout\n'
    printf '[B23.1] capsule=%s\narchive=%s.tar.gz\nchecksum=%s.tar.gz.sha256\n' "$capsule" "$capsule" "$capsule"
    [[ "$publish" -eq 1 ]] && printf '[B23.1] publish_result=%s_PUBLISH_RESULT.json\n' "$capsule"
    printf '[B23.1] stdout_log=%s\nstderr_log=%s\nstep_results=%s\n' "$stdout_log" "$stderr_log" "$steps"
    printf '[B23.1] cross_family_H0=FAIL\nqualified_np_sitcom_adapters=0\nB23.2_authorized=NO\nGPU_correction=NO\n'
  else
    printf '[B23.1] STOP rc=%s; scientific execution was not rerun and no correction GPU work was performed\n' "$rc" >&2
    printf '[B23.1] stdout_log=%s\nstderr_log=%s\nstep_results=%s\n' "$stdout_log" "$stderr_log" "$steps" >&2
  fi
  return "$rc"
}

main "$@"
