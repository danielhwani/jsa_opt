#!/usr/bin/env bash
set -e

# Change to the directory where this script is located
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Starting End-to-End Pipeline"
echo "============================================================"

# 1. Generate Dataset
echo ""
echo ">>> [1/4] Running Dataset Generation (generate_dataset.sh)..."
bash "$SCRIPTS_DIR/generate_dataset.sh"

# 2. Train Model
echo ""
echo ">>> [2/4] Running Training (train.sh)..."
bash "$SCRIPTS_DIR/train.sh"
bash "$SCRIPTS_DIR/train_conv.sh"
# 3. Run Benchmark
echo ""
echo ">>> [3/4] Running Benchmark (run_benchmark_original_jsa_pth_vs_trt.sh)..."
bash "$SCRIPTS_DIR/run_benchmark_original_jsa_pth_vs_trt.sh"

# 4. Run Viewer
echo ""
echo ">>> [4/4] Running Viewer (run_viewer.sh)..."
bash "$SCRIPTS_DIR/run_viewer.sh" "$@"

echo ""
echo "============================================================"
echo "Pipeline execution completed successfully!"
echo "============================================================"
