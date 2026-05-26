#!/usr/bin/env python3
"""
Render an over-fit dataset for the original JSA repository from Mitsuba 3.

Default output is the original EXR directory scheme expected by preprocess.py:

  <out_data>/__train_scenes__/<name>/input/*.exr
  <out_data>/__train_scenes__/<name>/target/*.exr
  <out_data>/__test_scenes__/<name>/input/*.exr
  <out_data>/__test_scenes__/<name>/target/*.exr

The original train/test code will create:
  input_npz/
  target_npz/
automatically if they do not exist.

Use --write-npz if you also want to write the NPZ files immediately for sharing/debugging.

Input EXR layers:
  default : noisy RGB
  albedo  : albedo RGB
  normal  : shading normal RGB, raw [-1,1] convention
  depth   : normalized depth, 1 channel

Target EXR layers:
  default : high-spp reference RGB

The same rendered views are copied to both train and test by default, so this is
an intentional over-fit benchmark dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.2f}h"


@dataclass
class CameraSample:
    origin: Tuple[float, float, float]
    target: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov: float


def sanitize_rgb(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=1e10, neginf=0.0)
    return np.clip(x, 0.0, np.max(x) if x.size else 0.0).astype(np.float32)


def normalize_depth_np(depth: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(d, 0.0, np.max(d) if d.size else 0.0)
    mx = float(np.max(d)) if d.size else 0.0
    if mx > 0.0:
        d = d / mx
    return d.astype(np.float32)


def split_aov(aov: np.ndarray):
    """
    Mitsuba AOV order used by this script:
      albedo:albedo,depth:depth,sh_normal:sh_normal

    aov layout:
      0:3  albedo
      3:4  depth
      4:7  shading normal
    """
    if aov.shape[-1] < 7:
        raise ValueError(f"Expected at least 7 AOV channels, got shape={aov.shape}")
    albedo = np.nan_to_num(aov[..., 0:3].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    albedo = np.clip(albedo, 0.0, 1.0)
    depth = normalize_depth_np(aov[..., 3:4])
    normal = np.nan_to_num(aov[..., 4:7].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    normal = np.clip(normal, -1.0, 1.0)
    return albedo, normal, depth


def write_layered_exr(path: Path, layers: dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    import pyexr
    clean = {}
    for k, v in layers.items():
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 2:
            a = a[..., None]
        clean[k] = np.nan_to_num(a, nan=0.0, posinf=1e10, neginf=-1e10)
    pyexr.write(str(path), clean)


def write_npz_pair(base_root: Path, split: str, stem: str,
                   color: np.ndarray, albedo: np.ndarray, normal: np.ndarray, depth: np.ndarray, ref: np.ndarray):
    input_npz = base_root / split / "input_npz"
    target_npz = base_root / split / "target_npz"
    input_npz.mkdir(parents=True, exist_ok=True)
    target_npz.mkdir(parents=True, exist_ok=True)
    aux = np.concatenate([albedo, normal, depth], axis=-1).astype(np.float32)
    np.savez_compressed(input_npz / f"{stem}.npz", color=color.astype(np.float32), aux=aux)
    np.savez_compressed(target_npz / f"{stem}.npz", color=ref.astype(np.float32))


def sample_fixed_camera(rng: np.random.Generator, args) -> CameraSample:
    origin = np.asarray(args.base_origin, dtype=np.float32)
    target = np.asarray(args.base_target, dtype=np.float32)
    up = np.asarray(args.base_up, dtype=np.float32)
    if args.origin_jitter > 0:
        origin = origin + rng.normal(0.0, args.origin_jitter, size=3).astype(np.float32)
    if args.target_jitter > 0:
        target = target + rng.normal(0.0, args.target_jitter, size=3).astype(np.float32)
    return CameraSample(tuple(map(float, origin)), tuple(map(float, target)), tuple(map(float, up)), float(args.base_fov))


def sample_orbit_camera(rng: np.random.Generator, args) -> CameraSample:
    center = np.asarray(args.center, dtype=np.float32)
    theta = rng.uniform(args.theta_min, args.theta_max) * math.pi / 180.0
    radius = rng.uniform(args.radius_min, args.radius_max)
    height = rng.uniform(args.height_min, args.height_max)
    target = center + rng.normal(0.0, args.target_jitter, size=3).astype(np.float32)
    target[1] = center[1] + (target[1] - center[1]) * 0.5
    origin = np.array([
        center[0] + radius * math.cos(theta),
        height,
        center[2] + radius * math.sin(theta),
    ], dtype=np.float32)
    origin += rng.normal(0.0, args.origin_jitter, size=3).astype(np.float32)
    fov = rng.uniform(args.fov_min, args.fov_max)
    return CameraSample(tuple(map(float, origin)), tuple(map(float, target)), (0.0, 1.0, 0.0), float(fov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--name", default="overfit_jsa")
    ap.add_argument("--out-data", default="../data")
    ap.add_argument("--variant", default="cuda_ad_rgb")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--num-views", type=int, default=4)
    ap.add_argument("--input-spp", type=int, default=1)
    ap.add_argument("--aov-spp", type=int, default=1)
    ap.add_argument("--ref-spp", type=int, default=4096)
    ap.add_argument("--ref-chunk-spp", type=int, default=512)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--rr-depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--write-npz", action="store_true", help="Also write input_npz/target_npz immediately.")
    ap.add_argument("--no-copy-test", action="store_true", help="Only write train split. Default writes same views to test too.")

    ap.add_argument("--camera-mode", choices=["fixed", "orbit"], default="fixed")
    ap.add_argument("--base-origin", nargs=3, type=float, default=(-37.4663, -0.614254, 32.1223))
    ap.add_argument("--base-target", nargs=3, type=float, default=(-36.799804, -0.462232, 31.392456))
    ap.add_argument("--base-up", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    ap.add_argument("--base-fov", type=float, default=35.0)

    ap.add_argument("--center", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    ap.add_argument("--radius-min", type=float, default=0.85)
    ap.add_argument("--radius-max", type=float, default=2.0)
    ap.add_argument("--height-min", type=float, default=-0.8)
    ap.add_argument("--height-max", type=float, default=0.8)
    ap.add_argument("--theta-min", type=float, default=0.0)
    ap.add_argument("--theta-max", type=float, default=360.0)
    ap.add_argument("--target-jitter", type=float, default=0.0)
    ap.add_argument("--origin-jitter", type=float, default=0.0)
    ap.add_argument("--fov-min", type=float, default=35.0)
    ap.add_argument("--fov-max", type=float, default=35.0)

    args = ap.parse_args()

    if args.width % 128 != 0 or args.height % 128 != 0:
        print(f"[warn] width/height should ideally be divisible by 128 for original JSA/TRT. Got {args.width}x{args.height}.")
    if args.width != args.height:
        print("[warn] original full-frame JSA has square-token assumptions. Square resolution is recommended for the first TRT benchmark.")

    import mitsuba as mi
    mi.set_variant(args.variant)
    print(f"[Mitsuba] variant={args.variant}")
    print(f"[Scene] loading {args.scene}")
    scene = mi.load_file(args.scene)

    out_data = Path(args.out_data)
    train_root = out_data / "__train_scenes__" / args.name
    test_root = out_data / "__test_scenes__" / args.name

    if args.overwrite:
        shutil.rmtree(train_root, ignore_errors=True)
        if not args.no_copy_test:
            shutil.rmtree(test_root, ignore_errors=True)

    for root in ([train_root] if args.no_copy_test else [train_root, test_root]):
        (root / "input").mkdir(parents=True, exist_ok=True)
        (root / "target").mkdir(parents=True, exist_ok=True)
        if args.write_npz:
            (root / "input_npz").mkdir(parents=True, exist_ok=True)
            (root / "target_npz").mkdir(parents=True, exist_ok=True)

    aov_integrator = mi.load_dict({
        "type": "aov",
        "aovs": "albedo:albedo,depth:depth,sh_normal:sh_normal",
        "integrator": {
            "type": "path",
            "max_depth": int(args.max_depth),
            "rr_depth": int(args.rr_depth),
        },
    })
    path_integrator = mi.load_dict({
        "type": "path",
        "max_depth": int(args.max_depth),
        "rr_depth": int(args.rr_depth),
    })

    def make_sensor(cam: CameraSample, spp: int):
        return mi.load_dict({
            "type": "perspective",
            "fov": float(cam.fov),
            "to_world": mi.ScalarTransform4f.look_at(
                origin=cam.origin,
                target=cam.target,
                up=cam.up,
            ),
            "sampler": {"type": "independent", "sample_count": int(spp)},
            "film": {
                "type": "hdrfilm",
                "width": int(args.width),
                "height": int(args.height),
                "rfilter": {"type": "box"},
            },
        })

    def to_np(img):
        return np.asarray(mi.Bitmap(img), dtype=np.float32)

    def render_aov(cam: CameraSample, seed: int):
        sensor = make_sensor(cam, args.aov_spp)
        img = mi.render(scene, sensor=sensor, integrator=aov_integrator, seed=int(seed), spp=int(args.aov_spp))
        arr = to_np(img)
        return sanitize_rgb(arr[..., :3]), arr[..., 3:]

    def render_rgb(cam: CameraSample, spp: int, seed: int):
        sensor = make_sensor(cam, spp)
        img = mi.render(scene, sensor=sensor, integrator=path_integrator, seed=int(seed), spp=int(spp))
        return sanitize_rgb(to_np(img)[..., :3])

    def render_ref(cam: CameraSample, seed: int):
        chunk = int(args.ref_chunk_spp or 0)
        if chunk <= 0 or args.ref_spp <= chunk:
            ts = time.time()
            out = render_rgb(cam, args.ref_spp, seed)
            print(f"      <- ref {args.ref_spp}spp: {fmt_seconds(time.time()-ts)}", flush=True)
            return out

        n_chunks = int(math.ceil(args.ref_spp / chunk))
        acc = None
        done = 0
        t0 = time.time()
        for ci in range(n_chunks):
            cur = min(chunk, args.ref_spp - done)
            tc = time.time()
            img = render_rgb(cam, cur, seed + 10007 * ci)
            if acc is None:
                acc = np.zeros_like(img, dtype=np.float64)
            acc += img.astype(np.float64) * float(cur)
            done += cur
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (args.ref_spp - done) / max(rate, 1e-9)
            print(f"         ref chunk [{ci+1:02d}/{n_chunks:02d}] {done}/{args.ref_spp} spp "
                  f"chunk={fmt_seconds(time.time()-tc)} elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(eta)}",
                  flush=True)
        print(f"      <- ref total: {fmt_seconds(time.time()-t0)}", flush=True)
        return (acc / float(args.ref_spp)).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    metadata = {
        "scene": args.scene,
        "variant": args.variant,
        "width": args.width,
        "height": args.height,
        "num_views": args.num_views,
        "input_spp": args.input_spp,
        "aov_spp": args.aov_spp,
        "ref_spp": args.ref_spp,
        "camera_mode": args.camera_mode,
        "write_npz": args.write_npz,
        "note": "Same views are written to train/test unless --no-copy-test is used.",
        "cameras": [],
    }

    roots = [train_root] if args.no_copy_test else [train_root, test_root]

    for i in range(args.num_views):
        cam = sample_fixed_camera(rng, args) if args.camera_mode == "fixed" else sample_orbit_camera(rng, args)
        print(f"[{i+1:04d}/{args.num_views:04d}] camera={cam}", flush=True)

        ts = time.time()
        rgb_from_aov, aov = render_aov(cam, args.seed + i * 17 + 1)
        print(f"      <- input+aov {args.width}x{args.height}@{args.aov_spp}: {fmt_seconds(time.time()-ts)}", flush=True)

        if args.input_spp == args.aov_spp:
            color = rgb_from_aov
        else:
            ts = time.time()
            color = render_rgb(cam, args.input_spp, args.seed + i * 17 + 2)
            print(f"      <- input rgb {args.width}x{args.height}@{args.input_spp}: {fmt_seconds(time.time()-ts)}", flush=True)

        albedo, normal, depth = split_aov(aov)
        ref = render_ref(cam, args.seed + i * 17 + 1000)

        stem = f"{args.name}_{i:04d}"

        for root in roots:
            write_layered_exr(root / "input" / f"{stem}.exr", {
                "default": color,
                "albedo": albedo,
                "normal": normal,
                "depth": depth,
            })
            write_layered_exr(root / "target" / f"{stem}.exr", {
                "default": ref,
            })
            if args.write_npz:
                write_npz_pair(root, "", stem, color, albedo, normal, depth, ref)

        metadata["cameras"].append(asdict(cam))

    for root in roots:
        with open(root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print("\n[done]")
    print(f"train EXR: {train_root / 'input'}")
    print(f"test  EXR: {test_root / 'input'}" if not args.no_copy_test else "test EXR: skipped")
    if args.write_npz:
        print(f"train NPZ: {train_root / 'input_npz'}")
        if not args.no_copy_test:
            print(f"test  NPZ: {test_root / 'input_npz'}")
    else:
        print("NPZ was not written. Original train/test code will create input_npz/target_npz from EXR if needed.")

    print("\nConfig patch:")
    print(f'config["task"] = "jsa_{args.name}"')
    print(f'config["trainDatasetDirectory"] = "../data/__train_scenes__/{args.name}"')
    print(f'config["testDatasetDirectory"] = "../data/__test_scenes__/{args.name}"')
    print('config["train_input"] = "input"')
    print('config["train_target"] = "target"')
    print('config["test_input"] = "input"')
    print('config["test_target"] = "target"')


if __name__ == "__main__":
    main()