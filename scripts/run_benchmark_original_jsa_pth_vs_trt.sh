#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/benchmark_results/engine}"
mkdir -p "$RESULT_DIR"

HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
WORKSPACE_GB="${WORKSPACE_GB:-8}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"

# Optional explicit checkpoints:
#   JSA_CHECKPOINT=/workspace/data/.../__checkpoints__/epoch_..._best.pth
#   CONV_CHECKPOINT=/workspace/data/.../__checkpoints__/epoch_..._best.pth
JSA_CKPT_ARGS=()
if [[ -n "${JSA_CHECKPOINT:-}" ]]; then
  JSA_CKPT_ARGS+=(--checkpoint "$JSA_CHECKPOINT")
fi

CONV_CKPT_ARGS=()
if [[ -n "${CONV_CHECKPOINT:-}" ]]; then
  CONV_CKPT_ARGS+=(--checkpoint "$CONV_CHECKPOINT")
fi

echo "============================================================"
echo "[1/2] Compile original JSA → torch2trt"
echo "============================================================"
python codes/benchmark_original_jsa_trt.py \
  --repo-root "$PROJECT_ROOT" \
  --model-kind jsa \
  --config-module config \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --workspace-gb "$WORKSPACE_GB" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --out-dir "$RESULT_DIR" \
  "${JSA_CKPT_ARGS[@]}"

echo
echo "============================================================"
echo "[2/2] Compile JSA+Conv → torch2trt"
echo "============================================================"
python codes/benchmark_original_jsa_trt.py \
  --repo-root "$PROJECT_ROOT" \
  --model-kind conv \
  --config-module config_cnn \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --workspace-gb "$WORKSPACE_GB" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --out-dir "$RESULT_DIR" \
  "${CONV_CKPT_ARGS[@]}"

echo
echo "[done] Engines written under: $RESULT_DIR"
ls -lh "$RESULT_DIR" | grep -E 'jsa.*(engine|torch2trt\.pth|json)$' || true
