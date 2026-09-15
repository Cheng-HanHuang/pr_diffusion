#!/usr/bin/env bash
b26_finalize_stage_main() {
    ROOT="/egr/research-pac/huang248"; REPO="$ROOT/pr_diffusion_b26"; PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"; RUN="$1"; STAGE="$2"
    [ -n "$RUN" ] && [ -d "$RUN" ] || { echo "STOP|usage: finalize_b26_gpu_stage.sh RUN smoke|dev16|dev80"; return 2; }
    case "$STAGE" in smoke|dev16|dev80) ;; *) echo "STOP|bad_stage=$STAGE"; return 2;; esac
    echo "GIT_VERSION|$(git --version 2>&1)"; echo "IDENTITY|branch=$(git -C "$REPO" branch --show-current 2>/dev/null)|head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
    MAN="$REPO/configs/b26/b26_dev80_manifest.json"; [ -f "$MAN" ] || { echo "STOP|manifest_missing"; return 3; }
    "$PY" "$REPO/scripts/b26/gate_b26_gpu.py" --manifest "$MAN" --gpu-root "$RUN/gpu" --scope "$STAGE" --output "$RUN/gpu/GATE_${STAGE}.json"; G=$?; echo "GATE_RC=$G"; [ "$G" -eq 0 ] || return 4
    if [ "$STAGE" = "dev16" ] || [ "$STAGE" = "dev80" ]; then
        "$PY" "$REPO/scripts/b26/analyze_b26_gpu.py" --manifest "$MAN" --gpu-root "$RUN/gpu" --scope "$STAGE" --output "$RUN/gpu/ANALYSIS_${STAGE}.json"; A=$?; echo "ANALYSIS_RC=$A"; [ "$A" -eq 0 ] || return 5
    fi
    echo "B26_GPU_STAGE_FINALIZED|run=$RUN|stage=$STAGE"
    if [ "$STAGE" = "smoke" ]; then echo "NEXT|bash $REPO/scripts/b26/launch_b26_gpu_stage.sh $RUN dev16 0,1,2,3"; fi
    if [ "$STAGE" = "dev16" ]; then echo "NEXT|bash $REPO/scripts/b26/launch_b26_gpu_stage.sh $RUN dev80 0,1,2,3"; fi
    return 0
}
b26_finalize_stage_main "$@"
