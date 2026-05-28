#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional public sharing/auth:
#   GRADIO_SHARE=1 GRADIO_AUTH_USER=demo GRADIO_AUTH_PASS='strong_password' ./scripts/run_viewer_jsa_family.sh
SHARE_ARGS=()
if [[ "${GRADIO_SHARE:-0}" == "1" ]]; then
  SHARE_ARGS+=(--share)
fi

AUTH_ARGS=()
if [[ -n "${GRADIO_AUTH_USER:-}" || -n "${GRADIO_AUTH_PASS:-}" ]]; then
  if [[ -z "${GRADIO_AUTH_USER:-}" || -z "${GRADIO_AUTH_PASS:-}" ]]; then
    echo "[error] Set both GRADIO_AUTH_USER and GRADIO_AUTH_PASS, or set neither." >&2
    exit 1
  fi
  AUTH_ARGS+=(--auth-user "$GRADIO_AUTH_USER" --auth-pass "$GRADIO_AUTH_PASS")
fi

python codes/viewer_jsa_oidn_dynamic_trt.py \
  --data-root "$PROJECT_ROOT/data" \
  --trt-root "$PROJECT_ROOT/benchmark_results/trt" \
  --engine-root "$PROJECT_ROOT/benchmark_results/engine" \
  --output-root "$PROJECT_ROOT/benchmark_results/viewer_outputs" \
  --host 0.0.0.0 \
  --port 7860 \
  "${SHARE_ARGS[@]}" \
  "${AUTH_ARGS[@]}"
