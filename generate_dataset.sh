#!/bin/bash

# unbounded cam sampling
python codes/generate_dataset.py \
  --scene house/house.xml \
  --out data/data_scene_exr \
  --variant cuda_ad_rgb \
  --high-w 1920 --high-h 1080 \
  --down 3 \
  --num-train 128 --num-val 16 --num-test 16 \
  --low-spp 1 \
  --gbuf-spp 1 \
  --high-rgb-spp 1 \
  --ref-spp 4096 \
  --ref-chunk-spp 512 \
  --center 0 1 0 \
  --radius-min 0.2 --radius-max 5.0 \
  --height-min 0.7 --height-max 2.0 \
  --split-aovs \
  --max-depth 10 \
  --rr-depth 5 \
  --aovs "albedo:albedo,depth:depth,sh_normal:sh_normal" \
  --overwrite \
  --seed 12

# python codes/generate_dataset-constrained.py \
#   --scene house/house.xml \
#    --out data/data_scene_exr \
#   --variant cuda_ad_rgb \
#   --camera-mode scene \
#   --base-origin -37.4663 -0.614254 32.1223 \
#   --base-target -36.799804 -0.462232 31.392456 \
#   --base-up -0.102514 0.988377 0.112257 \
#   --base-fov 35 \
#   --high-w 1920 --high-h 1080 \
#   --down 3 \
#   --num-train 128 --num-val 16 --num-test 16 \
#   --low-spp 1 \
#   --gbuf-spp 4 \
#   --high-rgb-spp 1 \
#   --ref-spp 2048 \
#   --ref-chunk-spp 512 \
#   --max-depth 10 \
#   --rr-depth 5 \
#   --split-aovs \
#   --aovs "albedo:albedo,depth:depth,sh_normal:sh_normal" \
#   --overwrite \
#   --seed 1234