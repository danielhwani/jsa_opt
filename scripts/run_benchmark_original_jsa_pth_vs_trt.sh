#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RESULT_DIR="$PROJECT_ROOT/benchmark_results/engine"
mkdir -p "$RESULT_DIR"

python codes/benchmark_original_jsa_trt.py \
  --repo-root "$PROJECT_ROOT" \
  --height 1024 \
  --width 1024 \
  --workspace-gb 8 \
  --warmup 20 \
  --iters 100 \
  --out-dir "$RESULT_DIR" \
  --save-ts

