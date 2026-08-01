#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(
  --scene scenes/staircase2/staircase2.xml         # scene file path
  --name staircase2_overfit_jsa                    # dataset name
  --out-data "$PROJECT_ROOT/data"                  # output data root

  --variant cuda_ad_rgb                            # GPU rendering

  --width "${WIDTH:-512}"                          # output width
  --height "${HEIGHT:-512}"                        # output height

  --num-views "${NUM_VIEWS:-2}"                    # number of overfit views

  --input-spp "${INPUT_SPP:-4}"                    # noisy RGB input spp
  --aov-spp "${AOV_SPP:-4}"                         # AOV / feature spp

  --ref-spp "${REF_SPP:-4096}"                      # GT reference spp
  --ref-chunk-spp 512                               # chunk size for ETA/progress

  --camera-mode fixed                               # fixed base camera with jitter
  --base-origin 6.91182 1.65163 2.55414             # base camera origin (from scene sensor matrix)
  --base-target 6.011643 1.651629953 2.118617       # base camera target (origin + forward)
  --base-up -4.21474e-08 1 -2.03917e-08             # Y-up camera up vector
  --base-fov 70

  --origin-jitter 0.00                              # camera origin perturbation
  --target-jitter 0.05                              # target perturbation

  --max-depth 17                                    # path max depth
  --rr-depth 5                                       # RR start depth

  --overwrite                                        # recreate dataset
  --write-npz                                         # also write NPZ immediately
  --seed 1234
)

python codes/generate_dataset.py "${ARGS[@]}"
