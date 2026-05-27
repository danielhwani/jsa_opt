#!/usr/bin/env python3
"""
Tiled 512x512 benchmark for original JSA / optimized JSA / TensorRT.

What this does:
  - compile TensorRT modules at tile size, default 512x512
  - run full-frame inference by split-and-stitch
  - save TRT TorchScript modules for reuse
  - save eval-style linear EXR + tonemapped PNG
  - compute both log-domain metrics and eval-style postprocessed/tone-mapped metrics

Backends:
  1. original_pytorch_tiled
  2. optimized_pytorch_tiled
  3. naive_trt_fp16_tiled
  4. optimized_trt_fp16_tiled
  5. optimized_trt_int8_tiled, optional

Notes:
  - Saved TRT files are Torch-TensorRT TorchScript modules (.ts), not raw .engine.
  - INT8 calibration uses torch_tensorrt.ts.ptq first, matching the older timeTest.py style.
"""

from __future__ import annotations

import argparse
import glob
import gc
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Optional original repo utilities for eval-style tone mapping
# -----------------------------------------------------------------------------

def import_eval_utils():
    util_image = None
    util_rend = None
    try:
        import utils.utils_image as util_image  # type: ignore
    except Exception as e:
        print(f"[warn] could not import utils.utils_image: {e}")

    for mod_name in ("utils.utils_rend_img", "utils.utils_rend"):
        try:
            util_rend = __import__(mod_name, fromlist=["dummy"])
            break
        except Exception:
            pass

    if util_rend is None:
        print("[warn] could not import utils.utils_rend_img or utils.utils_rend; using fallback tone mapping")

    return util_image, util_rend


# -----------------------------------------------------------------------------
# Preprocessing / metrics
# -----------------------------------------------------------------------------

def preprocess_normal_np(normal: np.ndarray) -> np.ndarray:
    normal = np.nan_to_num(normal.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    normal = (normal + 1.0) * 0.5
    return np.clip(normal, 0.0, 1.0).astype(np.float32)


def preprocess_specular_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def postprocess_specular_np_chw(log_chw: np.ndarray, util_rend=None) -> np.ndarray:
    """Return linear CHW from log-domain CHW."""
    if util_rend is not None and hasattr(util_rend, "postprocess_specular"):
        return util_rend.postprocess_specular(log_chw)
    return np.maximum(np.expm1(log_chw), 0.0).astype(np.float32)


def fallback_tonemap_uint8(linear_chw: np.ndarray) -> np.ndarray:
    """Simple fallback only used if original util_rend.tensor2img is unavailable."""
    img = np.transpose(linear_chw, (1, 2, 0))
    img = np.nan_to_num(img, nan=0.0, posinf=1e10, neginf=0.0)
    img = np.maximum(img, 0.0)
    img = img / (1.0 + img)
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    return (img * 255.0 + 0.5).astype(np.uint8)


def to_eval_png_from_pred_log(pred_log_chw: np.ndarray, util_rend=None):
    if util_rend is not None and hasattr(util_rend, "tensor2img"):
        return util_rend.tensor2img(pred_log_chw, post_spec=True)
    return fallback_tonemap_uint8(postprocess_specular_np_chw(pred_log_chw, util_rend=None))


def to_eval_png_from_gt_linear(gt_linear_chw: np.ndarray, util_rend=None):
    if util_rend is not None and hasattr(util_rend, "tensor2img"):
        return util_rend.tensor2img(gt_linear_chw)
    return fallback_tonemap_uint8(gt_linear_chw)


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    true_mean = torch.mean(target, dim=1, keepdim=True)
    return torch.mean(torch.square(target - pred) / (torch.square(true_mean) + 1e-2))


def mse_psnr(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    mse = torch.mean((pred - target) ** 2).item()
    peak = max(float(target.max().item()), 1.0)
    psnr = 10.0 * math.log10((peak * peak) / max(mse, 1e-12))
    return mse, psnr


def crop_hw(y: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return y[..., :h, :w]


def cleanup_cuda(*objs, note: str = ""):
    """Best-effort cleanup between heavy TRT compile stages."""
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if note:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[mem] after cleanup {note}: allocated={allocated:.3f}GB reserved={reserved:.3f}GB")


def pad_to_tile_multiple_tensor(x: torch.Tensor, tile: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad BCHW to a multiple of tile in H/W; returns padded tensor and original hw."""
    _, _, h, w = x.shape
    pad_h = ((h + tile - 1) // tile) * tile - h
    pad_w = ((w + tile - 1) // tile) * tile - w
    mode = "reflect"
    if pad_h >= h or pad_w >= w:
        mode = "replicate"
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return padded, (h, w)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

def natural_key(path: str):
    import re
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def find_npz_pairs(config, limit: Optional[int]) -> List[Tuple[str, str]]:
    test_dir = Path(config["testDatasetDirectory"])
    input_npz = test_dir / f"{config['test_input']}_npz"
    target_npz = test_dir / f"{config['test_target']}_npz"

    if not input_npz.exists() or not target_npz.exists() or len(list(input_npz.glob("*.npz"))) == 0:
        print(f"[data] NPZ not found under {input_npz}. Trying preprocess.construct_test_dataset_to_npz(config)...")
        import preprocess as pre
        pre.construct_test_dataset_to_npz(config)

    inputs = sorted(glob.glob(str(input_npz / "*.npz")), key=natural_key)
    targets = sorted(glob.glob(str(target_npz / "*.npz")), key=natural_key)

    if len(inputs) == 0:
        raise RuntimeError(f"No input NPZ files found: {input_npz}")
    if len(inputs) != len(targets):
        raise RuntimeError(f"Input/target count mismatch: {len(inputs)} vs {len(targets)}")

    if limit is not None and limit > 0:
        inputs = inputs[:limit]
        targets = targets[:limit]
    return list(zip(inputs, targets))


def load_npz_pair_full(input_path: str, target_path: str, device: torch.device):
    inp = np.load(input_path)
    gt_npz = np.load(target_path)

    color = inp["color"].astype(np.float32)
    aux = inp["aux"].astype(np.float32)
    target_linear = gt_npz["color"].astype(np.float32)

    aux = aux.copy()
    aux[..., 3:6] = preprocess_normal_np(aux[..., 3:6])

    x_linear = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    f = torch.from_numpy(aux).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    gt_linear = torch.from_numpy(target_linear).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

    x_log = preprocess_specular_torch(x_linear)
    gt_log = preprocess_specular_torch(gt_linear)

    return x_log, f, gt_log, gt_linear, tuple(x_log.shape[-2:])


class CalibTileDataset(torch.utils.data.Dataset):
    """Actual-data calibration tiles [10,tile,tile]."""

    def __init__(self, pairs: List[Tuple[str, str]], tile: int, max_tiles: int):
        self.tile = int(tile)
        self.items = []
        for input_path, target_path in pairs:
            color = np.load(input_path)["color"]
            h, w = color.shape[:2]
            ph = ((h + tile - 1) // tile) * tile
            pw = ((w + tile - 1) // tile) * tile
            for y in range(0, ph, tile):
                for x in range(0, pw, tile):
                    self.items.append((input_path, target_path, y, x))
        if max_tiles > 0:
            self.items = self.items[:max_tiles]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        input_path, target_path, y, x = self.items[idx]
        # CPU loading for DataLoaderCalibrator.
        x_log, f, _, _, _ = load_npz_pair_full(input_path, target_path, torch.device("cpu"))
        inp = torch.cat([x_log, f], dim=1)
        inp, _ = pad_to_tile_multiple_tensor(inp, self.tile)
        tile = inp[:, :, y:y+self.tile, x:x+self.tile].contiguous().squeeze(0)
        # Match TensorRT Input(..., dtype=torch.half) used for INT8/PTQ compile.
        return tile.half()


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class OriginalConcatWrapper(nn.Module):
    def __init__(self, model: nn.Module, in_x: int = 3, in_f: int = 7):
        super().__init__()
        self.model = model
        self.in_x = int(in_x)
        self.in_f = int(in_f)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp[:, :self.in_x]
        f = inp[:, self.in_x:self.in_x + self.in_f]
        try:
            return self.model(x=x, f=f)
        except TypeError:
            return self.model(x, f)


class SpaceToDepth(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = int(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r = self.r
        x = x.view(b, c, h // r, r, w // r, r)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * r * r, h // r, w // r)


class DepthToSpace(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = int(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, crr, h, w = x.shape
        r = self.r
        c = crr // (r * r)
        x = x.view(b, c, r, r, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(b, c, h * r, w * r)


def replace_pixel_ops(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.PixelUnshuffle):
            setattr(module, name, SpaceToDepth(child.downscale_factor))
        elif isinstance(child, nn.PixelShuffle):
            setattr(module, name, DepthToSpace(child.upscale_factor))
        else:
            replace_pixel_ops(child)


def build_original_model(config, img_size: int):
    import model.model_joint_sa as model_joint_sa
    return model_joint_sa.JSA_transformer(
        img_size=img_size,
        embedded_dim=config["embed_dim"],
        win_size=8,
        projection_option="linear",
        ffn_option="mlp",
        depths=[1, 2, 4, 8, 2, 8, 4, 2, 4],
        in_x=config["x_dim"],
        in_f=config["f_dim"],
    )


def build_v2_model(config, img_size: int):
    import model.model_joint_sa_v2_int8 as model_joint_sa_v2
    try:
        return model_joint_sa_v2.JSA_transformer(
            img_size=img_size,
            embedded_dim=config["embed_dim"],
            win_size=8,
            projection_option="linear",
            ffn_option="mlp",
            depths=[1, 2, 4, 8, 2, 8, 4, 2, 4],
            in_x=config["x_dim"],
            in_f=config["f_dim"],
        )
    except TypeError:
        return model_joint_sa_v2.JSA_transformer(
            image_size=img_size,
            embedded_dim=config["embed_dim"],
            win_size=8,
            projection_option="linear",
            ffn_option="mlp",
            depths=[1, 2, 4, 8, 2, 8, 4, 2, 4],
            in_x=config["x_dim"],
            in_f=config["f_dim"],
        )


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "params"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                if any(torch.is_tensor(v) for v in sd.values()):
                    return sd
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise RuntimeError("Could not find a state_dict in checkpoint.")


def clean_state_dict(sd):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_checkpoint_strict(net: nn.Module, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = clean_state_dict(extract_state_dict(ckpt))
    msg = net.load_state_dict(sd, strict=True)
    print(f"[ckpt] strict load OK: {checkpoint_path} | {msg}")


def default_checkpoint_path(config, epoch: str):
    task_dir = Path(config["data_dir"]) / config["task"]
    checkpoint_dir = task_dir / "__checkpoints__"
    return str(checkpoint_dir / f"epoch_{config['task']}_{epoch}.pth")


def apply_optimized_patches_after_load(net: nn.Module, patch_level: str):
    if patch_level not in {"none", "pixel", "int8"}:
        raise ValueError(patch_level)

    if patch_level in {"pixel", "int8"}:
        print("[patch] replace PixelShuffle/PixelUnshuffle with DepthToSpace/SpaceToDepth")
        replace_pixel_ops(net)

    if patch_level == "int8":
        import model.model_joint_sa_v2_int8 as model_joint_sa_v2
        if hasattr(model_joint_sa_v2, "patch_model_for_int8"):
            print("[patch] applying patch_model_for_int8 AFTER strict checkpoint load")
            model_joint_sa_v2.patch_model_for_int8(net)
        elif hasattr(model_joint_sa_v2, "patch_model_for_trt"):
            print("[patch] applying patch_model_for_trt AFTER strict checkpoint load")
            model_joint_sa_v2.patch_model_for_trt(net)
        else:
            print("[patch][warn] no patch_model_for_int8/patch_model_for_trt found")


def make_model_backend(config, checkpoint: str, tile: int, kind: str, patch_level: str, device: torch.device):
    if kind == "original":
        m = build_original_model(config, tile)
        load_checkpoint_strict(m, checkpoint)
        return OriginalConcatWrapper(m, config["x_dim"], config["f_dim"]).to(device).eval()

    if kind == "optimized":
        m = build_v2_model(config, tile)
        load_checkpoint_strict(m, checkpoint)
        apply_optimized_patches_after_load(m, patch_level)
        return m.to(device).eval()

    raise ValueError(kind)


# -----------------------------------------------------------------------------
# Tile split/stitch
# -----------------------------------------------------------------------------

class TRTInputCastWrapper(nn.Module):
    """Force Half input immediately before Torch-TensorRT engine call."""
    def __init__(self, trt_module: nn.Module, output_dtype=torch.float32, debug: bool = False):
        super().__init__()
        self.trt_module = trt_module
        self.output_dtype = output_dtype
        self.debug = debug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print("[TRTInputCastWrapper] before cast:", x.dtype, x.device, tuple(x.shape), flush=True)

        # Hard-code this. Do not depend on self.input_dtype for now.
        x = x.contiguous().to(device=x.device, dtype=torch.float16)

        if self.debug:
            print("[TRTInputCastWrapper] after  cast:", x.dtype, x.device, tuple(x.shape), flush=True)

        assert x.dtype == torch.float16, f"TRT input cast failed: got {x.dtype}"

        y = self.trt_module(x)

        if self.output_dtype is not None and y.dtype != self.output_dtype:
            y = y.to(dtype=self.output_dtype)

        return y
    
class TiledInferenceWrapper(nn.Module):
    """Runs a tile module over a full frame by split-and-stitch.

    module_input_dtype is needed for Torch-TensorRT FP16 modules, whose engine
    input type is Half. PyTorch reference modules and INT8 TRT modules usually
    consume Float.
    """

    def __init__(self, tile_module: nn.Module, tile: int, module_input_dtype=None, output_dtype=torch.float32):
        super().__init__()
        self.tile_module = tile_module
        self.tile = int(tile)
        self.module_input_dtype = module_input_dtype
        self.output_dtype = output_dtype

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        tile = self.tile
        inp_pad, orig_hw = pad_to_tile_multiple_tensor(inp, tile)
        _, _, hp, wp = inp_pad.shape
        rows = []
        for y in range(0, hp, tile):
            cols = []
            for x in range(0, wp, tile):
                t = inp_pad[:, :, y:y+tile, x:x+tile].contiguous()
                if self.module_input_dtype is not None and t.dtype != self.module_input_dtype:
                    t = t.to(dtype=self.module_input_dtype)
                y_tile = self.tile_module(t)
                if self.output_dtype is not None and y_tile.dtype != self.output_dtype:
                    y_tile = y_tile.to(dtype=self.output_dtype)
                cols.append(y_tile)
            rows.append(torch.cat(cols, dim=3))
        out = torch.cat(rows, dim=2)
        return crop_hw(out, orig_hw)


# -----------------------------------------------------------------------------
# TensorRT compile/save/load
# -----------------------------------------------------------------------------

def _get_torch_tensorrt_ptq_module():
    try:
        from torch_tensorrt.ts import ptq
        return ptq, "torch_tensorrt.ts.ptq"
    except Exception:
        pass

    try:
        import torch_tensorrt
        if hasattr(torch_tensorrt, "ptq"):
            return torch_tensorrt.ptq, "torch_tensorrt.ptq"
    except Exception:
        pass

    raise RuntimeError("No Torch-TensorRT PTQ calibrator API found.")


def make_int8_calibrator(pairs: List[Tuple[str, str]], tile: int, cache_file: str,
                         max_tiles: int, batch_size: int, device: torch.device):
    ptq, ns = _get_torch_tensorrt_ptq_module()
    print(f"[INT8] using PTQ API: {ns}")

    ds = CalibTileDataset(pairs, tile=tile, max_tiles=max_tiles)
    if len(ds) == 0:
        raise RuntimeError("No calibration tiles available.")

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    algo = None
    if hasattr(ptq, "CalibrationAlgo"):
        for name in ["ENTROPY_CALIBRATION_2", "ENTROPY", "MINMAX_CALIBRATION", "MINMAX"]:
            if hasattr(ptq.CalibrationAlgo, name):
                algo = getattr(ptq.CalibrationAlgo, name)
                break

    kwargs = {"cache_file": cache_file}
    if algo is not None:
        kwargs["algo_type"] = algo

    try:
        return ptq.DataLoaderCalibrator(loader, use_cache=False, device=device, **kwargs)
    except TypeError:
        pass

    try:
        return ptq.DataLoaderCalibrator(loader, **kwargs)
    except TypeError:
        kwargs.pop("algo_type", None)
        return ptq.DataLoaderCalibrator(loader, **kwargs)


def trt_module_path(trt_dir: str | Path, name: str, precision: str, shape: torch.Size) -> Path:
    b, c, h, w = [int(v) for v in shape]
    return Path(trt_dir) / f"{name}_{precision}_{b}x{c}x{h}x{w}.ts"


def compile_trt_ts(module: nn.Module, example_input: torch.Tensor, precision: str, require_full: bool,
                   workspace_gb: int, calibrator=None, debug: bool = False):
    import torch_tensorrt

    module = module.eval().to(example_input.device)

    if precision == "fp16":
        trace_input = example_input.half()
        module = module.half()
        trt_input = torch_tensorrt.Input(trace_input.shape, dtype=torch.half)
        enabled = {torch.half}
    elif precision == "fp32":
        trace_input = example_input.float()
        module = module.float()
        trt_input = torch_tensorrt.Input(trace_input.shape, dtype=torch.float)
        enabled = {torch.float}
    elif precision == "int8":
        # Match the timeTest-style PTQ path:
        # Input binding is Half, with INT8 kernels and FP16 fallback enabled.
        trace_input = example_input.half()
        module = module.half()
        trt_input = torch_tensorrt.Input(trace_input.shape, dtype=torch.half)
        enabled = {torch.int8, torch.half}
    else:
        raise ValueError(precision)

    print(f"[TRT] trace/compile precision={precision}, input={tuple(trace_input.shape)}, dtype={trace_input.dtype}")

    with torch.no_grad():
        ts = torch.jit.trace(module, trace_input)

    kwargs = dict(
        ir="ts",
        inputs=[trt_input],
        enabled_precisions=enabled,
        require_full_compilation=require_full,
        truncate_long_and_double=True,
        workspace_size=int(workspace_gb) << 30,
        min_block_size=1,
    )
    if calibrator is not None:
        kwargs["calibrator"] = calibrator
    if debug:
        kwargs["debug"] = True

    try:
        compiled = torch_tensorrt.compile(ts, **kwargs)
    except TypeError:
        kwargs.pop("debug", None)
        compiled = torch_tensorrt.compile(ts, **kwargs)

    return compiled.eval()


def get_or_compile_tile_trt(name: str, module: nn.Module, example_input: torch.Tensor, precision: str,
                            args, calibrator=None):
    save_path = trt_module_path(args.save_trt_dir, name, precision, example_input.shape) if args.save_trt_dir else None
    load_path = trt_module_path(args.load_trt_dir, name, precision, example_input.shape) if args.load_trt_dir else None

    if args.load_trt_dir and load_path.exists() and not args.force_recompile:
        print(f"[load TRT] {load_path}")
        return torch.jit.load(str(load_path), map_location=example_input.device).eval()

    mod = compile_trt_ts(
        module,
        example_input,
        precision=precision,
        require_full=args.require_full_trt,
        workspace_gb=args.workspace_gb,
        calibrator=calibrator,
        debug=args.trt_debug,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(mod, str(save_path))
        print(f"[saved TRT] {save_path}")

    return mod


# -----------------------------------------------------------------------------
# Evaluation / saving
# -----------------------------------------------------------------------------

def time_module(fn, warmup: int, iters: int):
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()

        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(iters):
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    return {
        "avg_ms": float(sum(times) / len(times)),
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "std_ms": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def eval_style_metrics_and_save(name: str, idx: int, pred_log: torch.Tensor, gt_log: torch.Tensor,
                                gt_linear: torch.Tensor, out_dir: Optional[Path], util_image, util_rend):
    pred_log_chw = pred_log.detach().cpu().numpy()[0].astype(np.float32)
    gt_log_chw = gt_log.detach().cpu().numpy()[0].astype(np.float32)
    gt_linear_chw = gt_linear.detach().cpu().numpy()[0].astype(np.float32)

    pred_linear_chw = postprocess_specular_np_chw(pred_log_chw, util_rend=util_rend)

    # eval-style tonemapped images
    pred_png = to_eval_png_from_pred_log(pred_log_chw, util_rend=util_rend)
    gt_png = to_eval_png_from_gt_linear(gt_linear_chw, util_rend=util_rend)

    rmse = float(np.sqrt(np.mean((np.transpose(pred_linear_chw, (1, 2, 0)) -
                                  np.transpose(gt_linear_chw, (1, 2, 0))) ** 2)))
    psnr_eval = None
    ssim_eval = None
    if util_image is not None and hasattr(util_image, "calculate_psnr"):
        psnr_eval = float(util_image.calculate_psnr(pred_png.copy(), gt_png.copy()))
    if util_image is not None and hasattr(util_image, "calculate_ssim"):
        try:
            ssim_eval = float(util_image.calculate_ssim(pred_png.copy(), gt_png.copy()))
        except Exception:
            ssim_eval = None

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import pyexr
            pyexr.write(str(out_dir / f"{idx:04d}_{name}.linear.exr"), np.transpose(pred_linear_chw, (1, 2, 0)).astype(np.float32))
            pyexr.write(str(out_dir / f"{idx:04d}_{name}.log.exr"), np.transpose(pred_log_chw, (1, 2, 0)).astype(np.float32))
        except Exception as e:
            print(f"[warn] failed to write EXR for {name}: {e}")

        if util_image is not None and hasattr(util_image, "imwrite"):
            util_image.imwrite(pred_png, str(out_dir / f"{idx:04d}_{name}_evalPSNR{psnr_eval if psnr_eval is not None else -1:.4f}.png"))
        else:
            try:
                from imageio.v2 import imwrite
                imwrite(str(out_dir / f"{idx:04d}_{name}.png"), pred_png)
            except Exception as e:
                print(f"[warn] failed to write PNG for {name}: {e}")

    return {
        "eval_rmse_linear": rmse,
        "eval_psnr_tonemapped": psnr_eval,
        "eval_ssim_tonemapped": ssim_eval,
    }


def summarize_output(name: str, pred: torch.Tensor, gt_log: torch.Tensor, ref: Optional[torch.Tensor]):
    mse, psnr = mse_psnr(pred, gt_log)
    out = {
        "rel_l2_log": float(rel_l2(pred, gt_log).item()),
        "mse_log": mse,
        "psnr_log": psnr,
    }
    if ref is not None:
        d = pred - ref
        out.update({
            "diff_mean_abs_to_original_log": float(d.abs().mean().item()),
            "diff_max_abs_to_original_log": float(d.abs().max().item()),
            "diff_rmse_to_original_log": float(torch.sqrt(torch.mean(d * d)).item()),
        })
    print(f"[{name}] relL2_log={out['rel_l2_log']:.8f} mse_log={out['mse_log']:.8e} psnr_log={out['psnr_log']:.4f}"
          + (f" | diff_mean_log={out.get('diff_mean_abs_to_original_log', 0):.3e} diff_max_log={out.get('diff_max_abs_to_original_log', 0):.3e}" if ref is not None else ""))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epoch", default=None)
    parser.add_argument("--limit", type=int, default=1)

    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)

    # Legacy: if the target-specific patch levels below are not set, this value is used.
    parser.add_argument("--optimized-patch-level", choices=["none", "pixel", "int8"], default="pixel")

    # Target-specific optimized model patch levels.
    # Use separate model instances because patch_model_for_int8() and .half()/.float() are in-place-like.
    parser.add_argument("--opt-pytorch-patch-level", choices=["none", "pixel", "int8"], default=None)
    parser.add_argument("--opt-fp16-patch-level", choices=["none", "pixel", "int8"], default=None)
    parser.add_argument("--opt-int8-patch-level", choices=["none", "pixel", "int8"], default="int8")

    parser.add_argument("--compile-int8", action="store_true")
    parser.add_argument("--int8-calib-cache", default=None)
    parser.add_argument("--int8-calib-max-tiles", type=int, default=16)
    parser.add_argument("--int8-calib-batch-size", type=int, default=1)

    parser.add_argument("--require-full-trt", action="store_true")
    parser.add_argument("--workspace-gb", type=int, default=2)
    parser.add_argument("--trt-debug", action="store_true")
    parser.add_argument("--skip-trt", action="store_true")

    parser.add_argument("--save-trt-dir", default=None)
    parser.add_argument("--load-trt-dir", default=None)
    parser.add_argument("--force-recompile", action="store_true")

    parser.add_argument("--save-json", default=None)
    parser.add_argument("--save-exr-dir", default=None)

    parser.add_argument("--seed", type=int, default=3040)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    from config import config
    util_image, util_rend = import_eval_utils()

    device = torch.device("cuda:0")
    epoch = args.epoch if args.epoch is not None else str(config["load_epoch"])
    checkpoint = args.checkpoint or default_checkpoint_path(config, epoch)

    print(f"[config] task={config['task']}")
    print(f"[config] testDatasetDirectory={config['testDatasetDirectory']}")
    opt_pytorch_patch = args.opt_pytorch_patch_level or args.optimized_patch_level
    opt_fp16_patch = args.opt_fp16_patch_level or args.optimized_patch_level
    opt_int8_patch = args.opt_int8_patch_level or args.optimized_patch_level

    print(f"[config] checkpoint={checkpoint}")
    print(f"[config] tile_size={args.tile_size}")
    print(f"[config] opt_pytorch_patch_level={opt_pytorch_patch}")
    print(f"[config] opt_fp16_patch_level={opt_fp16_patch}")
    print(f"[config] opt_int8_patch_level={opt_int8_patch}")

    pairs = find_npz_pairs(config, args.limit)
    all_pairs = find_npz_pairs(config, limit=None)

    # PyTorch tiled modules. Keep these as reference modules and do not mutate them during TRT compile.
    original_tile = make_model_backend(config, checkpoint, args.tile_size, "original", opt_pytorch_patch, device)
    optimized_tile = make_model_backend(config, checkpoint, args.tile_size, "optimized", opt_pytorch_patch, device)

    original_tiled = TiledInferenceWrapper(original_tile, args.tile_size, module_input_dtype=torch.float32, output_dtype=torch.float32).to(device).eval()
    optimized_tiled = TiledInferenceWrapper(optimized_tile, args.tile_size, module_input_dtype=torch.float32, output_dtype=torch.float32).to(device).eval()
    cleanup_cuda(note="after PyTorch reference model build")

    # TRT tile modules. Each compile target gets its own fresh model instance.
    compiled_tile_modules = {}
    if not args.skip_trt:
        example_input = torch.randn(
            1, config["x_dim"] + config["f_dim"], args.tile_size, args.tile_size,
            device=device, dtype=torch.float32
        )

        try:
            trt_orig_model = make_model_backend(config, checkpoint, args.tile_size, "original", opt_fp16_patch, device)
            compiled_tile_modules["naive_trt_fp16"] = get_or_compile_tile_trt(
                "naive_trt_tile", trt_orig_model, example_input, "fp16", args
            )
            cleanup_cuda(trt_orig_model, note="after naive_trt_fp16 compile")
        except Exception as e:
            print(f"[TRT][warn] naive_trt_fp16 unavailable: {e}")
            cleanup_cuda(note="after failed naive_trt_fp16 compile")

        try:
            trt_opt_fp16_model = make_model_backend(config, checkpoint, args.tile_size, "optimized", opt_fp16_patch, device)
            compiled_tile_modules["optimized_trt_fp16"] = get_or_compile_tile_trt(
                "optimized_trt_tile", trt_opt_fp16_model, example_input, "fp16", args
            )
            cleanup_cuda(trt_opt_fp16_model, note="after optimized_trt_fp16 compile")
        except Exception as e:
            print(f"[TRT][warn] optimized_trt_fp16 unavailable: {e}")
            cleanup_cuda(note="after failed optimized_trt_fp16 compile")

        if args.compile_int8:
            try:
                cache = args.int8_calib_cache
                if cache is None:
                    cache_root = Path(args.save_trt_dir or args.load_trt_dir or "./benchmark_results/trt")
                    cache_root.mkdir(parents=True, exist_ok=True)
                    cache = str(cache_root / f"optimized_trt_tile_int8_calib_{args.tile_size}x{args.tile_size}.cache")

                calibrator = make_int8_calibrator(
                    all_pairs,
                    tile=args.tile_size,
                    cache_file=cache,
                    max_tiles=args.int8_calib_max_tiles,
                    batch_size=args.int8_calib_batch_size,
                    device=device,
                )

                trt_opt_int8_model = make_model_backend(config, checkpoint, args.tile_size, "optimized", opt_int8_patch, device)
                compiled_tile_modules["optimized_trt_int8"] = get_or_compile_tile_trt(
                    "optimized_trt_tile", trt_opt_int8_model, example_input, "int8", args, calibrator=calibrator
                )
                cleanup_cuda(trt_opt_int8_model, calibrator, note="after optimized_trt_int8 compile")
            except Exception as e:
                print(f"[INT8][warn] optimized_trt_int8 unavailable: {e}")
                cleanup_cuda(note="after failed optimized_trt_int8 compile")

        cleanup_cuda(example_input, note="after all TRT compiles")

    compiled_tiled = {}
    for name, mod in compiled_tile_modules.items():
        wrapped_trt = TRTInputCastWrapper(
            mod,
            output_dtype=torch.float32,
            debug=True,   
        ).to(device).eval()

        compiled_tiled[name] = TiledInferenceWrapper(
            wrapped_trt,
            args.tile_size,
            module_input_dtype=None,
            output_dtype=torch.float32,
        ).to(device).eval()
    results: Dict[str, List[dict]] = {}
    out_dir = Path(args.save_exr_dir) if args.save_exr_dir else None

    def add_result(name, entry):
        results.setdefault(name, []).append(entry)

    for idx, (input_path, target_path) in enumerate(pairs):
        print(f"\n[view {idx}] {os.path.basename(input_path)}")
        x_log, f, gt_log, gt_linear, orig_hw = load_npz_pair_full(input_path, target_path, device)
        inp = torch.cat([x_log, f], dim=1).contiguous()

        with torch.no_grad():
            pred_orig = original_tiled(inp)
            pred_opt = optimized_tiled(inp)

        backends = [
            ("original_pytorch_tiled", pred_orig, original_tiled),
            ("optimized_pytorch_tiled", pred_opt, optimized_tiled),
        ]

        for name, mod in compiled_tiled.items():
            with torch.no_grad():
                pred = mod(inp)
            backends.append((name + "_tiled", pred, mod))

        for name, pred, mod in backends:
            ref = None if name == "original_pytorch_tiled" else pred_orig
            entry = summarize_output(name, pred, gt_log, ref)
            entry.update(eval_style_metrics_and_save(name, idx, pred, gt_log, gt_linear, out_dir, util_image, util_rend))
            entry.update(time_module(lambda m=mod: m(inp), args.warmup, args.iters))
            add_result(name, entry)

        cleanup_cuda(x_log, f, gt_log, gt_linear, inp, pred_orig, pred_opt, note=f"after view {idx}")

    print("\n=== Aggregate ===")
    agg = {}
    for name, rows in results.items():
        agg[name] = {}
        keys = set().union(*(r.keys() for r in rows))
        for k in keys:
            vals = [r[k] for r in rows if isinstance(r.get(k), (float, int)) and r.get(k) is not None]
            if vals:
                agg[name][k] = float(sum(vals) / len(vals))

        print(f"{name:26s} "
              f"time_avg={agg[name].get('avg_ms', float('nan')):9.4f} ms "
              f"time_med={agg[name].get('median_ms', float('nan')):9.4f} ms "
              f"logPSNR={agg[name].get('psnr_log', float('nan')):.4f} "
              f"evalPSNR={agg[name].get('eval_psnr_tonemapped', float('nan')):.4f} "
              f"diff_log={agg[name].get('diff_mean_abs_to_original_log', 0.0):.3e}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"per_view": results, "aggregate": agg}, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()