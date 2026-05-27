#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python codes/viewer_jsa_oidn_dynamic_trt.py \
  --data-root "$PROJECT_ROOT/data" \
  --trt-root "$PROJECT_ROOT/benchmark_results/trt" \
  --output-root "$PROJECT_ROOT/benchmark_results/viewer_outputs" \
  --host 0.0.0.0 \
  --port 7860
