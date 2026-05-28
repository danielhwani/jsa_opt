#!/usr/bin/env python3
"""
Minimal JSA-family FP16 torch2trt benchmark/export with export-time graph patching.

This script does NOT edit model_joint_sa.py and does NOT change the trained checkpoint.
It creates an export-only copy of the loaded model, applies mathematically equivalent
inference/export patches, then converts that copy with torch2trt.

Export-only patches:
  1. nn.PixelShuffle / nn.PixelUnshuffle
     -> equivalent view/permute/reshape modules.
     This avoids custom pixel converters.

  2. LinearProjection
     -> equivalent projection module with to_q, to_k, to_v as separate Linear layers.
     This removes q[0], kv[0], kv[1] indexing from the export graph.

  3. WindowJointAttention relative-position bias lookup
     -> precomputed buffer.
     This removes relative_position_bias_table[relative_position_index] tensor indexing
     from the export graph.

The selected PyTorch model is used for reference timing/quality. The export copy is compared
against the original PyTorch output before torch2trt conversion.

Artifacts:
  - .engine          raw TensorRT engine for C++/Unreal/TensorRT Runtime
  - .torch2trt.pth   torch2trt TRTModule Python reload artifact
  - .ts              best-effort torch.jit.trace(TRTModule), may fail on some installs
  - .json            timings and diff metrics
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt as trt  # noqa: F401
from torch2trt import TRTModule, torch2trt


# -----------------------------------------------------------------------------
# Repo imports
# -----------------------------------------------------------------------------

def add_repo_paths(repo_root: Path):
    for p in [repo_root, repo_root / "codes", repo_root / "codes" / "model"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def get_config(config_module: str):
    """Load config dict from config.py or config_cnn.py.

    For original JSA, default is config.
    For JSA+Conv, default is config_cnn because conv_train.py uses it.
    """
    try:
        mod = importlib.import_module(config_module)
        return getattr(mod, "config", {})
    except Exception as e:
        print(f"[warn] failed to import {config_module}.config: {e}")
        return {}


def _common_model_kwargs(config: dict, img_size: int):
    return dict(
        img_size=img_size,
        embedded_dim=config.get("embed_dim", 32),
        win_size=config.get("win_size", 8),
        projection_option=config.get("projection_option", "linear"),
        ffn_option=config.get("ffn_option", "mlp"),
        depths=config.get("depths", [1, 2, 4, 8, 2, 8, 4, 2, 4]),
        num_heads=config.get("num_heads", [1, 2, 4, 8, 16, 8, 4, 2, 1]),
        in_x=config.get("x_dim", 3),
        in_f=config.get("f_dim", 7),
    )


def build_jsa_family_model(config: dict, img_size: int, model_kind: str):
    """Build either original JSA or JSA+Conv with the same constructor size.

    model_kind:
      - jsa: original 4-layer JSA
      - conv: JSA with SwinIR-style convolutional final decoder
    """
    kind = model_kind.lower()
    kwargs = _common_model_kwargs(config, img_size)

    if kind in ("jsa", "original", "original_jsa"):
        # Prefer the stable wrapper if available; fall back to repo's original JSA_transformer.
        try:
            from model.jsa_original import OriginalJSATransformer
            return OriginalJSATransformer(**kwargs)
        except Exception:
            try:
                from jsa_original import OriginalJSATransformer
                return OriginalJSATransformer(**kwargs)
            except Exception:
                try:
                    from model.model_joint_sa import JSA_transformer
                except Exception:
                    from model_joint_sa import JSA_transformer
                try:
                    return JSA_transformer(**kwargs)
                except TypeError:
                    # Fallback for older/smaller JSA constructor.
                    return JSA_transformer(
                        in_x=kwargs["in_x"],
                        in_f=kwargs["in_f"],
                        img_size=img_size,
                        embedded_dim=kwargs["embedded_dim"],
                        depths=[1, 2, 4, 2, 4],
                        num_heads=[1, 2, 4, 2, 1],
                    )

    if kind in ("conv", "jsa_conv", "jsa+conv", "swinir_conv", "jsa_4layer_swinir_conv_decoder"):
        try:
            from model.jsa_4layer_swinir_conv_decoder import JSA4LayerSwinIRConvDecoder
        except Exception:
            from jsa_4layer_swinir_conv_decoder import JSA4LayerSwinIRConvDecoder
        kwargs["decoder_resi_connection"] = config.get("decoder_resi_connection", "3conv")
        return JSA4LayerSwinIRConvDecoder(**kwargs)

    raise ValueError(f"Unknown --model-kind {model_kind!r}. Use 'jsa' or 'conv'.")


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "params"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise RuntimeError("Could not find model state_dict in checkpoint")


def clean_state_dict(sd):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def default_checkpoint(config: dict, epoch: Optional[str]) -> Optional[str]:
    if not config:
        return None
    if epoch is None:
        epoch = str(config.get("load_epoch", "best"))
    task = config.get("task", "")
    data_dir = config.get("data_dir", config.get("DataDirectory", "/workspace/data"))
    if not task:
        return None
    return str(Path(data_dir) / task / "__checkpoints__" / f"epoch_{task}_{epoch}.pth")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = clean_state_dict(extract_state_dict(ckpt))
    msg = model.load_state_dict(sd, strict=True)
    print(f"[ckpt] strict load OK: {checkpoint_path} | {msg}")


# -----------------------------------------------------------------------------
# Export-only equivalent modules
# -----------------------------------------------------------------------------

class ExportPixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor: int):
        super().__init__()
        self.r = int(downscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r = self.r
        # Same mapping as F.pixel_unshuffle.
        x = x.reshape(b, c, h // r, r, w // r, r)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.reshape(b, c * r * r, h // r, w // r)


class ExportPixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = int(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, crr, h, w = x.shape
        r = self.r
        c = crr // (r * r)
        # Same mapping as F.pixel_shuffle.
        x = x.reshape(b, c, r, r, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.reshape(b, c, h * r, w * r)


class ExportLinearProjection(nn.Module):
    """Mathematically equivalent to original LinearProjection, but no q[0]/kv[0] indexing."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.heads = int(src.heads)
        self.dim = int(src.dim)
        self.inner_dim = int(src.inner_dim)

        self.to_q = copy.deepcopy(src.to_q)

        # Split original to_kv Linear into independent to_k and to_v.
        old = src.to_kv
        self.to_k = nn.Linear(old.in_features, self.inner_dim, bias=(old.bias is not None))
        self.to_v = nn.Linear(old.in_features, self.inner_dim, bias=(old.bias is not None))

        with torch.no_grad():
            self.to_k.weight.copy_(old.weight[:self.inner_dim])
            self.to_v.weight.copy_(old.weight[self.inner_dim:])
            if old.bias is not None:
                self.to_k.bias.copy_(old.bias[:self.inner_dim])
                self.to_v.bias.copy_(old.bias[self.inner_dim:])

    def forward(self, x: torch.Tensor):
        b, n, c = x.shape
        h = self.heads
        d = c // h

        q = self.to_q(x).reshape(b, n, h, d).permute(0, 2, 1, 3).contiguous()
        k = self.to_k(x).reshape(b, n, h, d).permute(0, 2, 1, 3).contiguous()
        v = self.to_v(x).reshape(b, n, h, d).permute(0, 2, 1, 3).contiguous()
        return q, k, v


class ExportWindowJointAttention(nn.Module):
    """Equivalent WindowJointAttention with precomputed relative-position bias."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.dim = src.dim
        self.win_size = src.win_size
        self.num_heads = src.num_heads
        self.scale = src.scale
        self.token_projection = getattr(src, "token_projection", "linear")

        if self.token_projection != "linear":
            # Keep conv path as-is if it appears, but current JSA uses linear.
            self.qkv = copy.deepcopy(src.qkv)
            self.qkv_f = copy.deepcopy(src.qkv_f)
        else:
            self.qkv = ExportLinearProjection(src.qkv)
            self.qkv_f = ExportLinearProjection(src.qkv_f)

        self.proj = copy.deepcopy(src.proj)
        self.softmax = copy.deepcopy(src.softmax)
        self.act = copy.deepcopy(getattr(src, "act", nn.ReLU()))

        with torch.no_grad():
            wh = int(self.win_size[0])
            ww = int(self.win_size[1])
            bias = src.relative_position_bias_table[src.relative_position_index.reshape(-1)].reshape(
                wh * ww, wh * ww, -1
            )
            bias = bias.permute(2, 0, 1).contiguous()
        self.register_buffer("relative_position_bias_precomputed", bias)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape

        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        q_f, k_f, v_f = self.qkv_f(f)
        q_f = q_f * self.scale
        attn_f = q_f @ k_f.transpose(-2, -1)

        # In this network, window_partition gives N = win_size*win_size, so no repeat is needed.
        bias = self.relative_position_bias_precomputed
        attn = attn + bias.unsqueeze(0)
        attn_f = attn_f + bias.unsqueeze(0)

        attn = self.softmax(attn * attn_f)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return x


def replace_child_modules(root: nn.Module, stats: dict):
    """Recursively replace export-hostile modules in an export copy only."""
    for name, child in list(root.named_children()):
        cls = child.__class__.__name__

        if isinstance(child, nn.PixelUnshuffle):
            setattr(root, name, ExportPixelUnshuffle(child.downscale_factor))
            stats["pixel_unshuffle"] += 1
            continue

        if isinstance(child, nn.PixelShuffle):
            setattr(root, name, ExportPixelShuffle(child.upscale_factor))
            stats["pixel_shuffle"] += 1
            continue

        if cls == "WindowJointAttention":
            setattr(root, name, ExportWindowJointAttention(child))
            stats["window_attention"] += 1
            continue

        # LinearProjection is normally inside WindowJointAttention and replaced above.
        # Keep this fallback for unusual placements.
        if cls == "LinearProjection":
            setattr(root, name, ExportLinearProjection(child))
            stats["linear_projection"] += 1
            continue

        replace_child_modules(child, stats)


def make_export_model(original: nn.Module, device: torch.device, in_place: bool = False) -> Tuple[nn.Module, dict]:
    """Create export-only copy and apply equivalent graph patches."""
    if in_place:
        export_model = original
    else:
        # Move original to CPU before deepcopy only if user needs to save GPU memory.
        # Here we keep it simple; 4090/24GB generally handles this for 1024.
        export_model = copy.deepcopy(original)

    stats = {
        "pixel_unshuffle": 0,
        "pixel_shuffle": 0,
        "window_attention": 0,
        "linear_projection": 0,
    }
    replace_child_modules(export_model, stats)
    export_model = export_model.to(device).eval()
    return export_model, stats


# -----------------------------------------------------------------------------
# Input / metrics
# -----------------------------------------------------------------------------

def preprocess_normal_np(normal: np.ndarray) -> np.ndarray:
    normal = np.nan_to_num(normal.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    normal = (normal + 1.0) * 0.5
    return np.clip(normal, 0.0, 1.0).astype(np.float32)


def preprocess_specular_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def load_npz_input(input_npz: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    z = np.load(input_npz)
    color = z["color"].astype(np.float32)
    aux = z["aux"].astype(np.float32)
    aux = aux.copy()
    aux[..., 3:6] = preprocess_normal_np(aux[..., 3:6])

    x = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    f = torch.from_numpy(aux).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    x_log = preprocess_specular_torch(x)

    gt = None
    target_npz = input_npz.replace("/input_npz/", "/target_npz/")
    if os.path.exists(target_npz):
        t = np.load(target_npz)["color"].astype(np.float32)
        gt_linear = torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        gt = preprocess_specular_torch(gt_linear)

    return x_log, f, gt


def random_input(batch: int, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(batch, 3, h, w, device=device, dtype=torch.float32)
    f = torch.rand(batch, 7, h, w, device=device, dtype=torch.float32)
    return x, f


def cuda_time_ms(fn, warmup: int, iters: int) -> dict:
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
            times.append(float(start.elapsed_time(end)))

    return {
        "avg_ms": float(sum(times) / len(times)),
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "std_ms": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def diff_metrics(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    d = a.float() - b.float()
    return {
        f"{name}_mean_abs": float(d.abs().mean().item()),
        f"{name}_max_abs": float(d.abs().max().item()),
        f"{name}_rmse": float(torch.sqrt(torch.mean(d * d)).item()),
    }


def gt_metrics(name: str, pred: torch.Tensor, gt: Optional[torch.Tensor]) -> dict:
    if gt is None:
        return {}
    mse = torch.mean((pred.float() - gt.float()) ** 2).item()
    return {
        f"{name}_mse_log_to_gt": float(mse),
        f"{name}_psnr_log_to_gt": float(10.0 * math.log10(1.0 / max(mse, 1e-12))),
    }


# -----------------------------------------------------------------------------
# Save artifacts
# -----------------------------------------------------------------------------

def save_torch2trt_artifacts(model_trt: TRTModule, out_prefix: Path, save_ts: bool, example_inputs):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    engine_path = out_prefix.with_suffix(".engine")
    pth_path = out_prefix.with_suffix(".torch2trt.pth")
    ts_path = out_prefix.with_suffix(".ts")

    with open(engine_path, "wb") as f:
        f.write(model_trt.engine.serialize())
    print(f"[saved] raw TensorRT engine: {engine_path}")

    torch.save(model_trt.state_dict(), pth_path)
    print(f"[saved] torch2trt TRTModule state_dict: {pth_path}")

    ts_ok = False
    if save_ts:
        try:
            model_trt.eval()
            with torch.no_grad():
                traced = torch.jit.trace(model_trt, example_inputs, strict=False)
            torch.jit.save(traced, str(ts_path))
            print(f"[saved] best-effort traced TRTModule TorchScript: {ts_path}")
            ts_ok = True
        except Exception as e:
            print(f"[warn] failed to save .ts from torch2trt TRTModule: {e}")
            print("[warn] Use .engine for C++/Unreal and .torch2trt.pth for Python viewer reload.")

    return {
        "engine": str(engine_path),
        "torch2trt_pth": str(pth_path),
        "ts": str(ts_path) if ts_ok else None,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/workspace")
    ap.add_argument("--model-kind", choices=["jsa", "conv"], default="jsa",
                    help="jsa: original JSA, conv: JSA+SwinIR-style conv final decoder")
    ap.add_argument("--config-module", default=None,
                    help="Python config module. Defaults to config for JSA and config_cnn for JSA+Conv.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--epoch", default=None)

    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--input-npz", default=None)

    ap.add_argument("--img-size", type=int, default=None, help="JSA constructor img_size. Defaults to max(height,width).")
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--save-ts", action="store_true")

    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out-dir", default="benchmark_results/engine")
    ap.add_argument("--name", default=None)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--patch-in-place", action="store_true", help="Lower peak memory; modifies the loaded object after baseline timing.")
    ap.add_argument("--max-export-diff", type=float, default=1e-5, help="Abort if export-patched PyTorch differs too much from original.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    repo_root = Path(args.repo_root).resolve()
    add_repo_paths(repo_root)
    if args.config_module is None:
        args.config_module = "config_cnn" if args.model_kind == "conv" else "config"
    config = get_config(args.config_module)

    checkpoint = args.checkpoint or default_checkpoint(config, args.epoch)
    if checkpoint is None or not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found. Provide --checkpoint. Got: {checkpoint}")

    device = torch.device("cuda:0")
    h, w = int(args.height), int(args.width)
    img_size = int(args.img_size or max(h, w))

    print(f"[config] repo_root={repo_root}")
    print(f"[config] model_kind={args.model_kind}")
    print(f"[config] config_module={args.config_module}")
    print(f"[config] checkpoint={checkpoint}")
    print(f"[config] input size={args.batch_size}x3/7x{h}x{w}")
    print(f"[config] JSA img_size={img_size}")

    print("[1] Load selected model")
    model = build_jsa_family_model(config, img_size=img_size, model_kind=args.model_kind)
    load_checkpoint(model, checkpoint)
    model = model.to(device).eval()

    print("[2] Prepare input")
    if args.input_npz:
        x, f, gt = load_npz_input(args.input_npz, device)
        if x.shape[-2:] != (h, w):
            print(f"[warn] NPZ resolution {tuple(x.shape[-2:])} differs from --height/--width. Using NPZ shape.")
            h, w = x.shape[-2:]
    else:
        x, f = random_input(args.batch_size, h, w, device)
        gt = None
    example_inputs = (x, f)

    print("[3] Selected PyTorch inference / timing")
    with torch.no_grad():
        pred_pt = model(x, f)
    pt_time = cuda_time_ms(lambda: model(x, f), args.warmup, args.iters)
    print(f"[pytorch/original] avg={pt_time['avg_ms']:.4f} ms med={pt_time['median_ms']:.4f} ms")

    print("[4] Create export-only equivalent model")
    export_model, patch_stats = make_export_model(model, device, in_place=args.patch_in_place)
    print(f"[export patch] {patch_stats}")

    with torch.no_grad():
        pred_export = export_model(x, f)
    export_diff = diff_metrics("export_patch_vs_original", pred_export, pred_pt)
    print(
        "[export patch diff] "
        f"mean={export_diff['export_patch_vs_original_mean_abs']:.6e}, "
        f"max={export_diff['export_patch_vs_original_max_abs']:.6e}, "
        f"rmse={export_diff['export_patch_vs_original_rmse']:.6e}"
    )

    if export_diff["export_patch_vs_original_max_abs"] > args.max_export_diff:
        raise RuntimeError(
            "Export-patched PyTorch model differs from original more than allowed. "
            f"max_abs={export_diff['export_patch_vs_original_max_abs']:.6e}, "
            f"threshold={args.max_export_diff:.6e}"
        )

    export_time = cuda_time_ms(lambda: export_model(x, f), max(1, args.warmup // 4), max(5, args.iters // 10))
    print(f"[pytorch/export] avg={export_time['avg_ms']:.4f} ms med={export_time['median_ms']:.4f} ms")

    print("[5] torch2trt conversion")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.inference_mode(), torch.no_grad():
        model_trt = torch2trt(
            export_model,
            list(example_inputs),
            fp16_mode=bool(args.fp16),
            max_workspace_size=int(args.workspace_gb) * (1 << 30),
        )
    model_trt = model_trt.to(device).eval()

    print("[6] TensorRT inference / timing")
    with torch.no_grad():
        pred_trt = model_trt(x, f)
    trt_time = cuda_time_ms(lambda: model_trt(x, f), args.warmup, args.iters)
    print(f"[torch2trt] avg={trt_time['avg_ms']:.4f} ms med={trt_time['median_ms']:.4f} ms")

    trt_diff = diff_metrics("torch2trt_vs_export_patch", pred_trt, pred_export)
    trt_orig_diff = diff_metrics("torch2trt_vs_original", pred_trt, pred_pt)

    print("[7] Save artifacts")
    if args.name is None:
        prefix = "jsa_conv" if args.model_kind == "conv" else "jsa"
        args.name = f"{prefix}_torch2trt_exportpatch_fp16_{args.batch_size}x3x{h}x{w}"
    out_prefix = Path(args.out_dir) / args.name
    paths = save_torch2trt_artifacts(model_trt, out_prefix, args.save_ts, example_inputs)

    results = {
        "model_kind": args.model_kind,
        "config_module": args.config_module,
        "checkpoint": checkpoint,
        "input_npz": args.input_npz,
        "shape_x": list(x.shape),
        "shape_f": list(f.shape),
        "patch_stats": patch_stats,
        "pytorch_original": pt_time,
        "pytorch_export_patch": export_time,
        "torch2trt": trt_time,
        "speedup_avg_vs_original": float(pt_time["avg_ms"] / max(trt_time["avg_ms"], 1e-9)),
        **export_diff,
        **trt_diff,
        **trt_orig_diff,
        **gt_metrics("original", pred_pt, gt),
        **gt_metrics("export_patch", pred_export, gt),
        **gt_metrics("torch2trt", pred_trt, gt),
        "artifacts": paths,
        "notes": {
            "export_patch": "Applied only to a copied model object at export time. Source model and checkpoint are unchanged.",
            "engine": "Raw TensorRT engine for TensorRT Runtime / C++ / Unreal.",
            "torch2trt_pth": "Python torch2trt TRTModule state_dict reload artifact; supported by the patched viewer.",
            "ts": "Best-effort trace only. torch2trt does not natively produce Torch-TensorRT .ts.",
        },
    }

    json_path = out_prefix.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[saved] results: {json_path}")

    print("\n=== Summary ===")
    print(f"Selected PyTorch avg : {pt_time['avg_ms']:.4f} ms")
    print(f"Export PyTorch avg   : {export_time['avg_ms']:.4f} ms")
    print(f"torch2trt avg        : {trt_time['avg_ms']:.4f} ms")
    print(f"speedup vs selected PyTorch  : {results['speedup_avg_vs_original']:.3f}x")
    print(f"export patch max diff: {export_diff['export_patch_vs_original_max_abs']:.6e}")
    print(f"trt/selected max diff    : {trt_orig_diff['torch2trt_vs_original_max_abs']:.6e}")
    print(f"engine               : {paths['engine']}")
    print(f"torch2trt pth         : {paths['torch2trt_pth']}")
    print(f"ts                   : {paths['ts']}")


if __name__ == "__main__":
    main()