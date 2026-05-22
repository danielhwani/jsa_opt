#!/usr/bin/env python3
"""
Generate a scene-specific multi-view dataset with Mitsuba 3 and write separate EXR files.

For each randomized camera view, this script writes a directory:

  <out>/<split>/<index>/
    low_rgb.exr       # low-resolution noisy/path-traced color, e.g. 640x360 @ 1 spp
    low_aov.exr       # low-resolution G-buffer/AOV channels, multi-channel EXR
    high_rgb.exr      # original-resolution noisy/path-traced color, for OIDN or SR baseline
    high_aov.exr      # original-resolution G-buffer/AOV channels, multi-channel EXR
    ref_rgb.exr       # original-resolution high-spp reference color
    camera.json       # camera parameters and render settings

Optional:
    --split-aovs also writes low_<name>.exr / high_<name>.exr for known AOVs.
    --write-npz additionally writes a compressed view.npz for debugging/backward compatibility.

Typical use for FHD target with 3x low-res inference:

  python mi3_multiview_exr_dataset.py \
    --scene scene.xml --out data_scene \
    --high-w 1920 --high-h 1080 --down 3 \
    --num-train 256 --num-val 32 --num-test 32 \
    --low-spp 1 --gbuf-spp 1 --high-rgb-spp 1 --ref-spp 256 \
    --center 0 1 0 --radius-min 2.5 --radius-max 5.0 \
    --height-min 0.7 --height-max 2.0 --split-aovs

Notes:
  - Requires: pip install mitsuba numpy pyexr
  - Use --variant cuda_ad_rgb for GPU rendering.
  - Random orbit cameras are only a starting point. For indoor scenes, pass conservative
    center/radius/height ranges or replace sample_camera() with your valid camera path sampler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import math
import time

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
    origin: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov: float


def sample_camera(rng: np.random.Generator, args) -> CameraSample:
    """Simple orbit-style random camera sampler. Edit this for scene-specific valid paths."""
    center = np.asarray(args.center, dtype=np.float32)

    theta = rng.uniform(args.theta_min, args.theta_max) * math.pi / 180.0
    radius = rng.uniform(args.radius_min, args.radius_max)
    height = rng.uniform(args.height_min, args.height_max)

    target_jitter = rng.normal(0.0, args.target_jitter, size=3).astype(np.float32)
    target_jitter[1] *= 0.5
    target = center + target_jitter

    origin = np.array([
        center[0] + radius * math.cos(theta),
        height,
        center[2] + radius * math.sin(theta),
    ], dtype=np.float32)
    origin += rng.normal(0.0, args.origin_jitter, size=3).astype(np.float32)

    return CameraSample(
        origin=tuple(float(x) for x in origin),
        target=tuple(float(x) for x in target),
        up=(0.0, 1.0, 0.0),
        fov=float(rng.uniform(args.fov_min, args.fov_max)),
    )


# Conservative channel-count map for common Mitsuba AOVs.
# Unknown AOVs are not split, but remain in low_aov.exr/high_aov.exr.
AOV_CHANNEL_COUNTS = {
    "albedo": 3,
    "depth": 1,
    "position": 3,
    "geo_normal": 3,
    "sh_normal": 3,
    "normal": 3,
    "uv": 2,
    "primitive_index": 1,
    "shape_index": 1,
}


def parse_aov_specs(aovs: str):
    """Parse 'name:type,name2:type2' into [(name, type, nchannels_or_None), ...]."""
    specs = []
    for part in [p.strip() for p in aovs.split(',') if p.strip()]:
        if ':' in part:
            name, typ = part.split(':', 1)
        else:
            name, typ = part, part
        name, typ = name.strip(), typ.strip()
        n = AOV_CHANNEL_COUNTS.get(typ, AOV_CHANNEL_COUNTS.get(name))
        specs.append((name, typ, n))
    return specs


def write_exr(path: Path, arr: np.ndarray):
    """Write HxWxC or HxW float32 array to EXR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    # Avoid NaNs/Infs poisoning training or OIDN tests.
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
    try:
        import pyexr
        pyexr.write(str(path), arr)
    except Exception:
        # Fallback to Mitsuba Bitmap writer. This may not preserve channel names,
        # but keeps the data in EXR format.
        import mitsuba as mi
        mi.Bitmap(arr).write(str(path))


def save_split_aovs(view_dir: Path, prefix: str, aov: np.ndarray, specs):
    """Write per-AOV EXR files when channel counts are known."""
    offset = 0
    for name, typ, n in specs:
        if n is None:
            continue
        if offset + n > aov.shape[-1]:
            print(f"[warn] cannot split AOV '{name}', expected {n} channels at offset {offset}, total={aov.shape[-1]}")
            break
        write_exr(view_dir / f"{prefix}_{name}.exr", aov[..., offset:offset + n])
        offset += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Mitsuba XML scene path")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--variant", default="cuda_ad_rgb", help="Mitsuba variant, e.g. cuda_ad_rgb or scalar_rgb")

    ap.add_argument("--high-w", type=int, default=1920)
    ap.add_argument("--high-h", type=int, default=1080)
    ap.add_argument("--down", type=int, default=3)
    ap.add_argument("--low-spp", type=int, default=1)
    ap.add_argument("--gbuf-spp", type=int, default=1, help="SPP for AOV/G-buffer renders")
    ap.add_argument("--high-rgb-spp", type=int, default=None,
                    help="SPP for high_rgb.exr. Defaults to --low-spp. If equal to --gbuf-spp, reuses the high AOV RGB pass.")
    ap.add_argument("--ref-spp", type=int, default=256)
    ap.add_argument("--ref-chunk-spp", type=int, default=0,
                    help="If >0, render ref_rgb in independent chunks of this SPP and print per-chunk ETA. Useful for long high-spp references, e.g. --ref-chunk-spp 64 for ref-spp 1024.")
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--rr-depth", type=int, default=5)

    ap.add_argument("--num-train", type=int, default=256)
    ap.add_argument("--num-val", type=int, default=32)
    ap.add_argument("--num-test", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--center", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    ap.add_argument("--radius-min", type=float, default=2.0)
    ap.add_argument("--radius-max", type=float, default=5.0)
    ap.add_argument("--height-min", type=float, default=0.7)
    ap.add_argument("--height-max", type=float, default=2.0)
    ap.add_argument("--theta-min", type=float, default=0.0)
    ap.add_argument("--theta-max", type=float, default=360.0)
    ap.add_argument("--target-jitter", type=float, default=0.15)
    ap.add_argument("--origin-jitter", type=float, default=0.02)
    ap.add_argument("--fov-min", type=float, default=45.0)
    ap.add_argument("--fov-max", type=float, default=55.0)

    ap.add_argument(
        "--aovs",
        default="albedo:albedo,depth:depth,sh_normal:sh_normal,geo_normal:geo_normal,position:position",
        help="Mitsuba AOV string. RGB image is stored in first 3 channels, AOVs after that.",
    )
    ap.add_argument("--split-aovs", action="store_true", help="Also write per-AOV EXR files such as high_albedo.exr.")
    ap.add_argument("--write-npz", action="store_true", help="Additionally write view.npz for compatibility/debugging.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.high_rgb_spp is None:
        args.high_rgb_spp = args.low_spp

    import mitsuba as mi
    mi.set_variant(args.variant)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(exist_ok=True)

    low_w = args.high_w // args.down
    low_h = args.high_h // args.down
    if args.high_w % args.down != 0 or args.high_h % args.down != 0:
        print(f"[warn] high resolution is not divisible by down={args.down}; using integer floor low res {low_w}x{low_h}.")

    print(f"[Mitsuba] variant={args.variant}")
    print(f"[Scene] loading: {args.scene}")
    scene = mi.load_file(args.scene)

    aov_specs = parse_aov_specs(args.aovs)
    aov_integrator = mi.load_dict({
        "type": "aov",
        "aovs": args.aovs,
        "integrator": {
            "type": "path",
            "max_depth": args.max_depth,
            "rr_depth": args.rr_depth,
        },
    })
    path_integrator = mi.load_dict({
        "type": "path",
        "max_depth": args.max_depth,
        "rr_depth": args.rr_depth,
    })

    def make_sensor(cam: CameraSample, w: int, h: int, spp: int):
        return mi.load_dict({
            "type": "perspective",
            "fov": cam.fov,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=cam.origin,
                target=cam.target,
                up=cam.up,
            ),
            "sampler": {
                "type": "independent",
                "sample_count": int(spp),
            },
            "film": {
                "type": "hdrfilm",
                "width": int(w),
                "height": int(h),
                "rfilter": {"type": "box"},
            },
        })

    def to_np(img):
        return np.asarray(mi.Bitmap(img), dtype=np.float32)

    def render_aov(cam: CameraSample, w: int, h: int, spp: int, seed: int):
        sensor = make_sensor(cam, w, h, spp)
        img = mi.render(scene, sensor=sensor, integrator=aov_integrator, seed=int(seed), spp=int(spp))
        arr = to_np(img)
        return arr[..., :3], arr[..., 3:]

    def render_rgb(cam: CameraSample, w: int, h: int, spp: int, seed: int):
        sensor = make_sensor(cam, w, h, spp)
        img = mi.render(scene, sensor=sensor, integrator=path_integrator, seed=int(seed), spp=int(spp))
        return to_np(img)[..., :3]

    def timed(label: str, fn):
        print(f"      -> {label} ...", flush=True)
        ts = time.time()
        out = fn()
        dt = time.time() - ts
        print(f"      <- {label}: {fmt_seconds(dt)}", flush=True)
        return out, dt

    def render_rgb_ref_with_progress(cam: CameraSample, w: int, h: int, spp: int, seed: int, label: str):
        chunk_spp = int(args.ref_chunk_spp or 0)
        if chunk_spp <= 0 or spp <= chunk_spp:
            return timed(label, lambda: render_rgb(cam, w, h, spp, seed))[0]

        n_chunks = int(math.ceil(spp / chunk_spp))
        print(f"      -> {label}: chunked {spp} spp as {n_chunks} chunks of <= {chunk_spp} spp", flush=True)
        acc = None
        done_spp = 0
        t_phase = time.time()
        for ci in range(n_chunks):
            cur_spp = min(chunk_spp, spp - done_spp)
            t_chunk = time.time()
            img = render_rgb(cam, w, h, cur_spp, seed + ci * 10007)
            if acc is None:
                acc = np.zeros_like(img, dtype=np.float64)
            acc += img.astype(np.float64) * float(cur_spp)
            done_spp += cur_spp
            chunk_dt = time.time() - t_chunk
            elapsed = time.time() - t_phase
            rate = done_spp / max(elapsed, 1e-9)
            rem = (spp - done_spp) / max(rate, 1e-9)
            print(
                f"         [{ci+1:02d}/{n_chunks:02d}] {done_spp}/{spp} spp "
                f"chunk={fmt_seconds(chunk_dt)} elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(rem)}",
                flush=True,
            )
        total_dt = time.time() - t_phase
        print(f"      <- {label}: {fmt_seconds(total_dt)}", flush=True)
        return (acc / float(spp)).astype(np.float32)

    split_counts = {"train": args.num_train, "val": args.num_val, "test": args.num_test}
    rng = np.random.default_rng(args.seed)
    meta = {
        "scene": str(args.scene),
        "variant": args.variant,
        "high_w": args.high_w,
        "high_h": args.high_h,
        "low_w": low_w,
        "low_h": low_h,
        "down": args.down,
        "low_spp": args.low_spp,
        "gbuf_spp": args.gbuf_spp,
        "high_rgb_spp": args.high_rgb_spp,
        "ref_spp": args.ref_spp,
        "aovs": args.aovs,
        "aov_specs": [{"name": n, "type": t, "channels": c} for n, t, c in aov_specs],
        "splits": split_counts,
        "files_per_view": ["low_rgb.exr", "low_aov.exr", "high_rgb.exr", "high_aov.exr", "ref_rgb.exr", "camera.json"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    total = sum(split_counts.values())
    global_index = 0
    completed_count = 0
    t0 = time.time()
    for split, count in split_counts.items():
        for _ in range(count):
            stem = f"{global_index:06d}"
            view_dir = out_dir / split / stem
            done_flag = view_dir / "DONE"
            if done_flag.exists() and not args.overwrite:
                completed_count += 1
                remaining = max(total - completed_count, 0)
                print(f"[skip] {view_dir}  ({completed_count}/{total}, remaining={remaining})", flush=True)
                global_index += 1
                continue

            view_dir.mkdir(parents=True, exist_ok=True)
            cam = sample_camera(rng, args)
            seed_base = args.seed * 1000003 + global_index * 17
            item_start = time.time()

            print(
                f"[{global_index+1:5d}/{total}] {split}/{stem}: "
                f"low {low_w}x{low_h}@{args.low_spp}, "
                f"high_rgb {args.high_w}x{args.high_h}@{args.high_rgb_spp}, "
                f"high_aov@{args.gbuf_spp}, ref@{args.ref_spp}"
            , flush=True)

            (low_rgb, low_aov), _ = timed(
                f"low_aov/rgb {low_w}x{low_h}@{args.low_spp}",
                lambda: render_aov(cam, low_w, low_h, args.low_spp, seed_base + 1),
            )
            (high_rgb_from_aov, high_aov), _ = timed(
                f"high_aov/rgb {args.high_w}x{args.high_h}@{args.gbuf_spp}",
                lambda: render_aov(cam, args.high_w, args.high_h, args.gbuf_spp, seed_base + 2),
            )
            if args.high_rgb_spp == args.gbuf_spp:
                high_rgb = high_rgb_from_aov
                print("      == high_rgb: reused RGB channels from high_aov pass", flush=True)
            else:
                high_rgb, _ = timed(
                    f"high_rgb {args.high_w}x{args.high_h}@{args.high_rgb_spp}",
                    lambda: render_rgb(cam, args.high_w, args.high_h, args.high_rgb_spp, seed_base + 4),
                )
            ref_rgb = render_rgb_ref_with_progress(
                cam, args.high_w, args.high_h, args.ref_spp, seed_base + 3,
                f"ref_rgb {args.high_w}x{args.high_h}@{args.ref_spp}",
            )

            io_start = time.time()
            print("      -> write EXRs ...", flush=True)
            write_exr(view_dir / "low_rgb.exr", low_rgb)
            write_exr(view_dir / "low_aov.exr", low_aov)
            write_exr(view_dir / "high_rgb.exr", high_rgb)
            write_exr(view_dir / "high_aov.exr", high_aov)
            write_exr(view_dir / "ref_rgb.exr", ref_rgb)

            if args.split_aovs:
                save_split_aovs(view_dir, "low", low_aov, aov_specs)
                save_split_aovs(view_dir, "high", high_aov, aov_specs)

            cam_meta = {
                "index": global_index,
                "split": split,
                "camera": asdict(cam),
                "seed_base": int(seed_base),
                "low_resolution": [low_w, low_h],
                "high_resolution": [args.high_w, args.high_h],
                "low_spp": args.low_spp,
                "gbuf_spp": args.gbuf_spp,
                "high_rgb_spp": args.high_rgb_spp,
                "ref_spp": args.ref_spp,
            }
            (view_dir / "camera.json").write_text(json.dumps(cam_meta, indent=2))

            if args.write_npz:
                np.savez_compressed(
                    view_dir / "view.npz",
                    low_rgb=low_rgb,
                    low_aov=low_aov,
                    high_rgb=high_rgb,
                    high_aov=high_aov,
                    ref_rgb=ref_rgb,
                    camera_origin=np.asarray(cam.origin, dtype=np.float32),
                    camera_target=np.asarray(cam.target, dtype=np.float32),
                    camera_up=np.asarray(cam.up, dtype=np.float32),
                    fov=np.asarray([cam.fov], dtype=np.float32),
                )

            done_flag.write_text("ok\n")
            io_dt = time.time() - io_start
            item_dt = time.time() - item_start
            completed_count += 1
            avg_dt = (time.time() - t0) / max(completed_count, 1)
            remaining = max(total - completed_count, 0)
            print(
                f"      <- write EXRs: {fmt_seconds(io_dt)}\n"
                f"[done] {split}/{stem}: item={fmt_seconds(item_dt)} "
                f"avg={fmt_seconds(avg_dt)} eta_total={fmt_seconds(avg_dt * remaining)} "
                f"({completed_count}/{total})",
                flush=True,
            )
            global_index += 1

    print(f"Done. elapsed={(time.time()-t0)/60:.2f} min. Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()