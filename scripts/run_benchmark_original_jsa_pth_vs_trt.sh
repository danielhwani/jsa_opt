#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RESULT_DIR="$PROJECT_ROOT/benchmark_results"
mkdir -p "$RESULT_DIR/exr" "$RESULT_DIR/trt"

python codes/benchmark_original_jsa_trt.py \
  --limit 1 \
  --tile-size 512 \
  --warmup 20 \
  --iters 100 \
  --opt-pytorch-patch-level pixel \
  --opt-fp16-patch-level pixel \
  --workspace-gb 8 \
  --save-json "$RESULT_DIR/results.json" \
  --save-exr-dir "$RESULT_DIR/exr" \
  --save-trt-dir "$RESULT_DIR/trt" \
  --require-full-trt 
  # --force-recompile
  #--load-trt-dir benchmark_results/trt \

