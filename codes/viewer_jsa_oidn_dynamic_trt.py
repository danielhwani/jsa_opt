#!/usr/bin/env python3
"""
Interactive viewer for JSA PTH / Torch-TensorRT / OIDN.

Matches the current benchmark script:
  - TRT modules can be either:
      1) Torch-TensorRT TorchScript .ts files under benchmark_results/trt
      2) torch2trt TRTModule state_dict files (*.torch2trt.pth) under benchmark_results/engine
  - Torch-TensorRT .ts file names are parsed dynamically:
      naive_trt_tile_fp16_1x10x512x512.ts
      optimized_trt_tile_fp16_1x10x1024x1024.ts
      optimized_trt_tile_int8_1x10x512x512.ts
  - torch2trt files are parsed from:
      jsa_torch2trt_fp16_1x3x1024x1024.torch2trt.pth
  - TRT runtime input is cast to Half immediately before the engine call.
  - Full images are processed by split/stitch using the tile size parsed from the TRT file name.
  - Denoised outputs are saved under benchmark_results/viewer_outputs/.
"""

from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gradio as gr
except Exception as e:
    raise RuntimeError("Please install gradio: pip install gradio") from e


# -----------------------------------------------------------------------------
# Workspace / imports
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parent.parent

for p in [
    WORKSPACE_ROOT,
    WORKSPACE_ROOT / "codes",
    WORKSPACE_ROOT / "codes" / "model",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from config import config as REPO_CONFIG  # type: ignore
except Exception:
    REPO_CONFIG = {}

try:
    from config_cnn import config as CONV_CONFIG  # type: ignore
except Exception:
    CONV_CONFIG = {}

try:
    import model.model_joint_sa as model_joint_sa
except Exception:
    try:
        import model_joint_sa  # type: ignore
    except Exception as e:
        model_joint_sa = None
        ORIGINAL_IMPORT_ERROR = e
    else:
        ORIGINAL_IMPORT_ERROR = None
else:
    ORIGINAL_IMPORT_ERROR = None

try:
    from model.jsa_original import OriginalJSATransformer
except Exception:
    try:
        from jsa_original import OriginalJSATransformer  # type: ignore
    except Exception:
        OriginalJSATransformer = None  # type: ignore

try:
    from model.jsa_4layer_swinir_conv_decoder import JSA4LayerSwinIRConvDecoder
except Exception:
    try:
        from jsa_4layer_swinir_conv_decoder import JSA4LayerSwinIRConvDecoder  # type: ignore
    except Exception as e:
        JSA4LayerSwinIRConvDecoder = None  # type: ignore
        CONV_IMPORT_ERROR = e
    else:
        CONV_IMPORT_ERROR = None
else:
    CONV_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def cleanup_cuda():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")

def natural_key(path: str):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]

def preprocess_normal_np(normal: np.ndarray) -> np.ndarray:
    normal = np.nan_to_num(normal.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    normal = (normal + 1.0) * 0.5
    return np.clip(normal, 0.0, 1.0).astype(np.float32)

def log1p_nonneg(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x.astype(np.float32), 0.0, None))

def expm1_nonneg(x: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(x.astype(np.float32)), 0.0, None)

def simple_tonemap(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, None)
    x = x / (1.0 + x)
    x = np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(x, 0.0, 1.0)

def rgb_to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def png_bytes(img_uint8: np.ndarray) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(img_uint8).save(bio, format="PNG")
    return bio.getvalue()

def to_data_url_png(img_uint8: np.ndarray) -> str:
    b64 = base64.b64encode(png_bytes(img_uint8)).decode("ascii")
    return f"data:image/png;base64,{b64}"

def make_slider_html(left_u8: np.ndarray, right_u8: np.ndarray, left_label: str, right_label: str, height: int = 520) -> str:
    """Pixel-aligned before/after slider. Uses object-fit: fill to avoid black bars/misalignment."""
    if left_u8.shape[:2] != right_u8.shape[:2]:
        raise ValueError(f"Slider images must have same H/W, got {left_u8.shape} and {right_u8.shape}")

    left_url = to_data_url_png(left_u8)
    right_url = to_data_url_png(right_u8)
    h, w = left_u8.shape[:2]
    width = int(height * (w / max(h, 1)))
    uid = str(abs(hash((left_url[:16], right_url[:16], time.time()))) % 10_000_000)

    return f"""
<div style="max-width:{width}px; margin-bottom: 8px;">
  <div style="position:relative; width:{width}px; height:{height}px; overflow:hidden; border:1px solid #666; border-radius:8px; background:#111;">
    <img src="{left_url}" style="position:absolute; left:0; top:0; width:100%; height:100%; object-fit:fill;" />
    <div id="overlay_{uid}" style="position:absolute; left:0; top:0; width:50%; height:100%; overflow:hidden;">
      <img src="{right_url}" style="position:absolute; left:0; top:0; width:{width}px; height:{height}px; max-width:none; object-fit:fill;" />
    </div>
    <div id="divider_{uid}" style="position:absolute; top:0; left:50%; width:2px; height:100%; background:#fff; box-shadow:0 0 4px rgba(0,0,0,0.85);"></div>
    <input id="slider_{uid}" type="range" min="0" max="100" value="50"
           style="position:absolute; left:12px; right:12px; bottom:12px; width:calc(100% - 24px);"/>
    <div style="position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.62); color:#fff; padding:4px 8px; border-radius:6px; font-size:12px;">{left_label}</div>
    <div style="position:absolute; top:10px; right:10px; background:rgba(0,0,0,0.62); color:#fff; padding:4px 8px; border-radius:6px; font-size:12px;">{right_label}</div>
  </div>
</div>
<script>
(function(){{
  const slider = document.getElementById('slider_{uid}');
  const overlay = document.getElementById('overlay_{uid}');
  const divider = document.getElementById('divider_{uid}');
  if (!slider || !overlay || !divider) return;
  const update = () => {{
    const v = slider.value + '%';
    overlay.style.width = v;
    divider.style.left = v;
  }};
  slider.addEventListener('input', update);
  update();
}})();
</script>
"""

def error_map_u8(pred_linear: np.ndarray, gt_linear: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if gt_linear is None:
        return None
    err = np.abs(pred_linear - gt_linear).mean(axis=2, keepdims=True)
    vmax = float(np.percentile(err, 99.0)) if np.any(err > 0) else 1.0
    vmax = max(vmax, 1e-6)
    e = np.clip(err / vmax, 0.0, 1.0)
    heat = np.concatenate([e, np.sqrt(e), 1.0 - e], axis=2)
    return rgb_to_uint8(heat)

def comparison_u8(left_u8: Optional[np.ndarray], right_u8: Optional[np.ndarray], split_percent: float = 50.0) -> Optional[np.ndarray]:
    """Return a static split comparison image.

    Left side is input, right side is selected output.
    This avoids custom JavaScript, which Gradio may sanitize or not re-run.
    """
    if left_u8 is None or right_u8 is None:
        return None
    if left_u8.shape[:2] != right_u8.shape[:2]:
        # Use PIL resize only for UI comparison. The actual saved outputs are untouched.
        right_u8 = np.asarray(Image.fromarray(right_u8).resize((left_u8.shape[1], left_u8.shape[0]), Image.BILINEAR))
    h, w = left_u8.shape[:2]
    x = int(np.clip(split_percent, 0.0, 100.0) / 100.0 * w)
    out = right_u8.copy()
    out[:, :x] = left_u8[:, :x]
    # white divider
    if 0 <= x < w:
        out[:, max(0, x - 1):min(w, x + 1)] = 255
    return out

def _resize_like(ref: np.ndarray, img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.shape[:2] == ref.shape[:2]:
        return img
    return np.asarray(Image.fromarray(img).resize((ref.shape[1], ref.shape[0]), Image.BILINEAR))


def _draw_label(img: np.ndarray, label: str) -> np.ndarray:
    """Draw a simple black label strip without depending on cv2."""
    out = img.copy()
    h, w = out.shape[:2]
    strip_h = max(24, min(42, h // 9))
    out[:strip_h, :, :] = (out[:strip_h, :, :] * 0.25).astype(np.uint8)
    # PIL text drawing is available through Pillow.
    try:
        from PIL import ImageDraw, ImageFont
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        draw.text((8, max(3, strip_h // 5)), label, fill=(255, 255, 255))
        out = np.asarray(pil)
    except Exception:
        pass
    return out


def zoom_grid_u8(
    input_u8: Optional[np.ndarray],
    pred_u8: Optional[np.ndarray],
    oidn_u8: Optional[np.ndarray],
    ref_u8: Optional[np.ndarray],
    center_x: float,
    center_y: float,
    crop_size: int,
    panel_size: int = 320,
) -> Optional[np.ndarray]:
    """2x2 zoom crop: input / selected engine / OIDN / reference."""
    if input_u8 is None or pred_u8 is None:
        return None

    pred_u8 = _resize_like(input_u8, pred_u8)
    oidn_u8 = _resize_like(input_u8, oidn_u8)
    ref_u8 = _resize_like(input_u8, ref_u8)

    h, w = input_u8.shape[:2]
    crop_size = int(max(16, min(crop_size, max(h, w))))
    cx = int(np.clip(center_x, 0, w - 1))
    cy = int(np.clip(center_y, 0, h - 1))

    x0 = int(np.clip(cx - crop_size // 2, 0, max(0, w - crop_size)))
    y0 = int(np.clip(cy - crop_size // 2, 0, max(0, h - crop_size)))
    x1 = min(w, x0 + crop_size)
    y1 = min(h, y0 + crop_size)

    def crop_and_resize(img: Optional[np.ndarray], label: str) -> np.ndarray:
        if img is None:
            blank = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
            return _draw_label(blank, label + " (missing)")
        crop = img[y0:y1, x0:x1]
        pil = Image.fromarray(crop).resize((panel_size, panel_size), Image.BILINEAR)
        return _draw_label(np.asarray(pil), label)

    p_input = crop_and_resize(input_u8, "Input")
    p_pred = crop_and_resize(pred_u8, "Selected engine")
    p_oidn = crop_and_resize(oidn_u8, "OIDN")
    p_ref = crop_and_resize(ref_u8, "Reference")

    gap = 4
    top = np.concatenate([p_input, np.full((panel_size, gap, 3), 255, dtype=np.uint8), p_pred], axis=1)
    bottom = np.concatenate([p_oidn, np.full((panel_size, gap, 3), 255, dtype=np.uint8), p_ref], axis=1)
    grid = np.concatenate([top, np.full((gap, top.shape[1], 3), 255, dtype=np.uint8), bottom], axis=0)
    return grid

def calc_log_metrics(pred_log: np.ndarray, gt_log: Optional[np.ndarray]) -> Dict[str, float]:
    if gt_log is None:
        return {}
    mse = float(np.mean((pred_log - gt_log) ** 2))
    psnr = float(20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12))))
    rel_l2 = float(np.linalg.norm(pred_log - gt_log) / max(np.linalg.norm(gt_log), 1e-12))
    return {"log_mse": mse, "log_psnr_unitpeak": psnr, "log_relL2": rel_l2}

def calc_linear_metrics(pred_linear: np.ndarray, gt_linear: Optional[np.ndarray]) -> Dict[str, float]:
    if gt_linear is None:
        return {}
    mse = float(np.mean((pred_linear - gt_linear) ** 2))
    psnr = float(20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12))))
    return {"linear_mse": mse, "linear_psnr_unitpeak": psnr}


def calc_exr_relmse(pred_linear: np.ndarray, gt_linear: Optional[np.ndarray]) -> Dict[str, float]:
    """EXR/linear-domain relative MSE, matching the common rendering relMSE style.

    relMSE = mean_c,p ((pred - gt)^2 / (mean_rgb(gt)^2 + 1e-2))
    """
    if gt_linear is None:
        return {}
    denom = np.mean(gt_linear, axis=2, keepdims=True) ** 2 + 1e-2
    relmse = float(np.mean((pred_linear - gt_linear) ** 2 / denom))
    return {"relMSE_linear_exr": relmse}

def save_outputs(out_root: str, sample_key: str, engine_name: str, pred_log: np.ndarray, pred_linear: np.ndarray, pred_u8: np.ndarray, metrics: Dict):
    out_dir = Path(out_root) / slugify(Path(sample_key).stem) / slugify(engine_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(pred_u8).save(out_dir / "denoised.png")
    np.save(out_dir / "denoised_log.npy", pred_log.astype(np.float32))
    np.save(out_dir / "denoised_linear.npy", pred_linear.astype(np.float32))

    try:
        import pyexr
        pyexr.write(str(out_dir / "denoised.linear.exr"), pred_linear.astype(np.float32))
        pyexr.write(str(out_dir / "denoised.log.exr"), pred_log.astype(np.float32))
    except Exception as e:
        metrics["save_exr_warning"] = repr(e)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metrics["saved_output_dir"] = str(out_dir)
    return str(out_dir)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

@dataclass
class SampleInfo:
    key: str
    input_npz: str
    target_npz: Optional[str]

def discover_samples(data_root: str) -> List[SampleInfo]:
    data_root = os.path.abspath(data_root)
    patterns = [
        os.path.join(data_root, "__test_scenes__", "*", "input_npz", "*.npz"),
        os.path.join(data_root, "__train_scenes__", "*", "input_npz", "*.npz"),
        os.path.join(data_root, "**", "input_npz", "*.npz"),
        os.path.join(data_root, "**", "input", "*.npz"),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))

    out: Dict[str, SampleInfo] = {}
    for inp in sorted(set(found), key=natural_key):
        p = Path(inp)
        scene_root = p.parent.parent
        stem = p.stem
        candidates = [
            scene_root / "target_npz" / f"{stem}.npz",
            scene_root / "target" / f"{stem}.npz",
        ]
        target = next((str(c) for c in candidates if c.exists()), None)
        key = os.path.relpath(inp, data_root)
        out[key] = SampleInfo(key=key, input_npz=str(inp), target_npz=target)

    return [out[k] for k in sorted(out.keys())]

def load_input_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x_log, f_net, input_linear, aux_raw."""
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)

    if "color" in keys:
        color = z["color"].astype(np.float32)
    elif "x" in keys:
        # Fallback: assume x is log-domain, reconstruct linear for display.
        x_log = z["x"].astype(np.float32)
        color = expm1_nonneg(x_log)
    else:
        raise KeyError(f"Could not find color/x in {path}. keys={sorted(keys)}")

    if "aux" in keys:
        aux = z["aux"].astype(np.float32)
    elif "f" in keys:
        aux = z["f"].astype(np.float32)
    else:
        if all(k in keys for k in ["albedo", "depth"]) and ("normal" in keys or "sh_normal" in keys):
            albedo = z["albedo"].astype(np.float32)
            normal = z["normal"].astype(np.float32) if "normal" in keys else z["sh_normal"].astype(np.float32)
            depth = z["depth"].astype(np.float32)
            if depth.ndim == 2:
                depth = depth[..., None]
            aux = np.concatenate([albedo[..., :3], normal[..., :3], depth[..., :1]], axis=2)
        else:
            raise KeyError(f"Could not find aux/f or albedo/normal/depth in {path}. keys={sorted(keys)}")

    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"color must be HxWx3, got {color.shape}")
    if aux.ndim != 3 or aux.shape[2] < 7:
        raise ValueError(f"aux must be HxWx7+, got {aux.shape}")

    aux_raw = aux[..., :7].copy()
    f_net = aux_raw.copy()
    f_net[..., 3:6] = preprocess_normal_np(f_net[..., 3:6])

    x_log = log1p_nonneg(color)
    return x_log.astype(np.float32), f_net.astype(np.float32), color.astype(np.float32), aux_raw.astype(np.float32)

def load_target_npz(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    for k in ["color", "gt", "target", "ref_rgb"]:
        if k in keys:
            color = z[k].astype(np.float32)
            if color.ndim == 3 and color.shape[2] == 3:
                return log1p_nonneg(color)
    return None

def checkpoints(data_root: str) -> List[str]:
    files = glob.glob(os.path.join(data_root, "**", "__checkpoints__", "*.pth"), recursive=True)
    return sorted(set(files), key=natural_key)


# -----------------------------------------------------------------------------
# Model / TRT
# -----------------------------------------------------------------------------

def crop_hw(y: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return y[..., :h, :w]

def pad_to_multiple(inp: torch.Tensor, tile: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = inp.shape
    hp = math.ceil(h / tile) * tile
    wp = math.ceil(w / tile) * tile
    pad_h = hp - h
    pad_w = wp - w
    if pad_h == 0 and pad_w == 0:
        return inp, (h, w)
    mode = "reflect"
    if pad_h >= h or pad_w >= w:
        mode = "replicate"
    return F.pad(inp, (0, pad_w, 0, pad_h), mode=mode), (h, w)

class OrigPTHConcat(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net.eval()

    def forward(self, inp10: torch.Tensor) -> torch.Tensor:
        x = inp10[:, :3]
        f = inp10[:, 3:]
        try:
            return self.net(x=x, f=f)
        except TypeError:
            return self.net(x, f)

class TiledWrapper(nn.Module):
    def __init__(self, tile_module: nn.Module, tile: int, input_dtype: Optional[torch.dtype], output_dtype=torch.float32):
        super().__init__()
        self.tile_module = tile_module.eval()
        self.tile = int(tile)
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def forward(self, inp10: torch.Tensor) -> torch.Tensor:
        inp_pad, hw = pad_to_multiple(inp10, self.tile)
        _, _, hp, wp = inp_pad.shape
        rows = []
        for yy in range(0, hp, self.tile):
            cols = []
            for xx in range(0, wp, self.tile):
                t = inp_pad[:, :, yy:yy+self.tile, xx:xx+self.tile].contiguous()
                if self.input_dtype is not None and t.dtype != self.input_dtype:
                    t = t.to(self.input_dtype)
                out = self.tile_module(t)
                if self.output_dtype is not None and out.dtype != self.output_dtype:
                    out = out.to(self.output_dtype)
                cols.append(out)
            rows.append(torch.cat(cols, dim=3))
        return crop_hw(torch.cat(rows, dim=2), hw)

@dataclass
class TRTInfo:
    label: str
    path: str
    tile: int
    precision: str
    kind: str


@dataclass
class Torch2TRTInfo:
    label: str
    path: str
    h: int
    w: int
    precision: str
    kind: str


TRT_RE = re.compile(r"(?P<name>.+)_(?P<precision>fp16|fp32|int8)_1x(?P<c>\d+)x(?P<h>\d+)x(?P<w>\d+)\.ts$")

# Example:
#   jsa_torch2trt_fp16_1x3x1024x1024.torch2trt.pth
# The channel in the filename is the x input channel (=3); the engine itself
# takes two inputs: x=[B,3,H,W], f=[B,7,H,W].
TORCH2TRT_RE = re.compile(
    r"(?P<name>.+?)(?:_(?P<precision>fp16|fp32|int8))?_"
    r"(?P<b>\d+)x(?P<c>\d+)x(?P<h>\d+)x(?P<w>\d+)\.torch2trt\.pth$"
)

def discover_trt(trt_root: str) -> Dict[str, TRTInfo]:
    out: Dict[str, TRTInfo] = {}
    for path in sorted(glob.glob(os.path.join(trt_root, "*.ts")), key=natural_key):
        m = TRT_RE.match(os.path.basename(path))
        if not m:
            continue
        name = m.group("name")
        precision = m.group("precision")
        h = int(m.group("h"))
        w = int(m.group("w"))
        if h != w:
            continue

        low_name = name.lower()
        if "conv" in low_name or "swinir" in low_name:
            kind = "JSA+Conv Torch-TensorRT"
        elif name.startswith("naive_trt"):
            kind = "JSA Torch-TensorRT"
        elif name.startswith("optimized_trt"):
            kind = "JSA Optimized Torch-TensorRT"
        else:
            kind = name

        label = f"{kind} {precision.upper()} {h}x{w}"
        out[label] = TRTInfo(label=label, path=path, tile=h, precision=precision, kind=kind)

    return out


class Torch2TRTConcatWrapper(nn.Module):
    """Wrap torch2trt TRTModule so the viewer can feed [B,10,H,W].

    For timing, use `prepare_inputs()` + `forward_prepared()` to avoid repeating
    the [B,10,H,W] -> (x, f) split/contiguous work inside every measured run.
    """

    def __init__(self, pth_path: str, output_dtype=torch.float32):
        super().__init__()
        try:
            from torch2trt import TRTModule
        except Exception as e:
            raise RuntimeError("torch2trt is required to load *.torch2trt.pth viewer engines.") from e

        self.trt = TRTModule()
        state = torch.load(pth_path)
        self.trt.load_state_dict(state)
        self.trt.eval()
        self.output_dtype = output_dtype

    def prepare_inputs(self, inp10: torch.Tensor):
        x = inp10[:, :3].contiguous()
        f = inp10[:, 3:10].contiguous()
        # The simple torch2trt export is built from original model(x, f).
        # Most torch2trt TRTModule builds accept FP32 bindings even for FP16 engines.
        if x.dtype != torch.float32:
            x = x.float()
            f = f.float()
        return x, f

    def forward_prepared(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        y = self.trt(x, f)
        if self.output_dtype is not None and y.dtype != self.output_dtype:
            y = y.to(dtype=self.output_dtype)
        return y

    def forward(self, inp10: torch.Tensor) -> torch.Tensor:
        x, f = self.prepare_inputs(inp10)
        return self.forward_prepared(x, f)


def discover_torch2trt(engine_root: str) -> Dict[str, Torch2TRTInfo]:
    out: Dict[str, Torch2TRTInfo] = {}
    patterns = [
        os.path.join(engine_root, "*.torch2trt.pth"),
        os.path.join(engine_root, "**", "*.torch2trt.pth"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    for path in sorted(set(paths), key=natural_key):
        base = os.path.basename(path)
        m = TORCH2TRT_RE.match(base)
        if m:
            name = m.group("name")
            precision = m.group("precision") or "fp16"
            h = int(m.group("h"))
            w = int(m.group("w"))
            low_name = name.lower()
            family = "JSA+Conv" if ("conv" in low_name or "swinir" in low_name) else "JSA"
            label = f"{family} Torch2TRT {precision.upper()} {h}x{w} ({name})"
        else:
            precision = "fp16"
            h = 0
            w = 0
            family = "JSA+Conv" if ("conv" in base.lower() or "swinir" in base.lower()) else "JSA"
            label = f"{family} Torch2TRT ({base})"

        out[label] = Torch2TRTInfo(
            label=label,
            path=path,
            h=h,
            w=w,
            precision=precision,
            kind="torch2trt",
        )
    return out

def pth_tile_choices(trt_infos: Dict[str, TRTInfo], t2_infos: Optional[Dict[str, Torch2TRTInfo]] = None) -> List[int]:
    tiles = {512, 1024}
    for info in trt_infos.values():
        tiles.add(info.tile)
    if t2_infos:
        for info in t2_infos.values():
            if info.h:
                tiles.add(info.h)
    return sorted(tiles)


def _common_kwargs(config: dict, tile: int) -> Dict[str, object]:
    return {
        "img_size": tile,
        "embedded_dim": config.get("embed_dim", 32),
        "win_size": config.get("win_size", 8),
        "projection_option": config.get("projection_option", "linear"),
        "ffn_option": config.get("ffn_option", "mlp"),
        "depths": config.get("depths", [1, 2, 4, 8, 2, 8, 4, 2, 4]),
        "num_heads": config.get("num_heads", [1, 2, 4, 8, 16, 8, 4, 2, 1]),
        "in_x": config.get("x_dim", 3),
        "in_f": config.get("f_dim", 7),
    }


def build_original_model(tile: int, device: torch.device) -> nn.Module:
    config = REPO_CONFIG if isinstance(REPO_CONFIG, dict) else {}
    kwargs = _common_kwargs(config, tile)

    if OriginalJSATransformer is not None:
        net = OriginalJSATransformer(**kwargs)
    else:
        if model_joint_sa is None:
            raise RuntimeError(f"Could not import original JSA model: {ORIGINAL_IMPORT_ERROR}")
        # Current repo/original JSA_transformer constructor.
        net = model_joint_sa.JSA_transformer(**kwargs)

    return net.to(device).eval()


def build_conv_model(tile: int, device: torch.device) -> nn.Module:
    if JSA4LayerSwinIRConvDecoder is None:
        raise RuntimeError(f"Could not import JSA+Conv model: {CONV_IMPORT_ERROR}")

    config = CONV_CONFIG if isinstance(CONV_CONFIG, dict) and CONV_CONFIG else REPO_CONFIG
    kwargs = _common_kwargs(config if isinstance(config, dict) else {}, tile)
    kwargs["decoder_resi_connection"] = (config if isinstance(config, dict) else {}).get("decoder_resi_connection", "3conv")
    net = JSA4LayerSwinIRConvDecoder(**kwargs)
    return net.to(device).eval()


PTH_RE = re.compile(r"(?P<family>JSA\+Conv|JSA|Orig) PTH tiled (?P<t>\d+)x(?P=t)$")


def pth_family_from_label(engine_name: str) -> Optional[Tuple[str, int]]:
    m = PTH_RE.match(engine_name)
    if not m:
        return None
    fam = m.group("family")
    if fam == "Orig":
        fam = "JSA"
    return fam, int(m.group("t"))

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "params"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    return ckpt

def load_ckpt(net: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)

def ensure_torch_tensorrt_registered():
    # Required before torch.jit.load() for Torch-TensorRT TS modules.
    try:
        import torch_tensorrt  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Failed to import torch_tensorrt before loading a Torch-TensorRT .ts module. "
            "Without this import, torch.classes.tensorrt.Engine is not registered."
        ) from e

class EngineManager:
    def __init__(self, data_root: str, trt_root: str, engine_root: str, device: Optional[str]):
        self.data_root = os.path.abspath(data_root)
        self.trt_root = os.path.abspath(trt_root)
        self.engine_root = os.path.abspath(engine_root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache: Dict[str, nn.Module] = {}

    def trt_infos(self) -> Dict[str, TRTInfo]:
        return discover_trt(self.trt_root)

    def torch2trt_infos(self) -> Dict[str, Torch2TRTInfo]:
        return discover_torch2trt(self.engine_root)

    def choices(self) -> List[str]:
        infos = self.trt_infos()
        t2_infos = self.torch2trt_infos()
        out = []
        if checkpoints(self.data_root):
            for t in pth_tile_choices(infos, t2_infos):
                out.append(f"JSA PTH tiled {t}x{t}")
                out.append(f"JSA+Conv PTH tiled {t}x{t}")
        out.extend(infos.keys())
        out.extend(t2_infos.keys())
        return out

    def source_info(self, engine_name: str, ckpt_path: Optional[str]) -> Dict[str, object]:
        pth_info = pth_family_from_label(engine_name)
        if pth_info:
            fam, tile = pth_info
            return {
                "engine_kind": f"PTH-{fam}",
                "engine_tile": tile,
                "engine_precision": "fp32",
                "engine_source": ckpt_path or "",
                "engine_source_name": os.path.basename(ckpt_path or ""),
            }
        infos = self.trt_infos()
        info = infos.get(engine_name)
        if info is not None:
            return {
                "engine_kind": "Torch-TensorRT-TS",
                "engine_tile": info.tile,
                "engine_precision": info.precision,
                "engine_source": info.path,
                "engine_source_name": os.path.basename(info.path),
            }

        t2_infos = self.torch2trt_infos()
        t2_info = t2_infos.get(engine_name)
        if t2_info is not None:
            return {
                "engine_kind": "torch2trt-TRTModule",
                "engine_tile": t2_info.h if t2_info.h else None,
                "engine_precision": t2_info.precision,
                "engine_source": t2_info.path,
                "engine_source_name": os.path.basename(t2_info.path),
            }

        return {
            "engine_kind": "unknown",
            "engine_tile": None,
            "engine_precision": None,
            "engine_source": "",
            "engine_source_name": "",
        }

    def get(self, engine_name: str, ckpt_path: Optional[str]) -> nn.Module:
        key = f"{engine_name}|{ckpt_path or ''}"
        if key in self.cache:
            return self.cache[key]

        pth_info = pth_family_from_label(engine_name)
        if pth_info:
            if not ckpt_path:
                raise FileNotFoundError("Select checkpoint for the selected PTH engine.")
            fam, tile = pth_info
            if fam == "JSA+Conv":
                net = build_conv_model(tile, self.device)
            else:
                net = build_original_model(tile, self.device)
            load_ckpt(net, ckpt_path)
            mod = TiledWrapper(OrigPTHConcat(net), tile=tile, input_dtype=torch.float32).to(self.device).eval()
        else:
            infos = self.trt_infos()
            if engine_name in infos:
                info = infos[engine_name]
                ensure_torch_tensorrt_registered()
                ts = torch.jit.load(info.path, map_location=self.device).eval()
                # Current benchmark compiles FP16 and INT8 with TensorRT Input(dtype=torch.half).
                mod = TiledWrapper(ts, tile=info.tile, input_dtype=torch.float16).to(self.device).eval()
            else:
                t2_infos = self.torch2trt_infos()
                if engine_name not in t2_infos:
                    raise FileNotFoundError(
                        f"Cannot find engine '{engine_name}' in {self.trt_root} or {self.engine_root}"
                    )
                t2_info = t2_infos[engine_name]
                t2 = Torch2TRTConcatWrapper(t2_info.path, output_dtype=torch.float32).to(self.device).eval()
                # The torch2trt export is fixed-shape. Use parsed H as tile so full-frame
                # matching-resolution inputs run in one call; larger inputs are split/stiched.
                tile = int(t2_info.h or 8192)
                mod = TiledWrapper(t2, tile=tile, input_dtype=torch.float32).to(self.device).eval()

        self.cache[key] = mod
        return mod


# -----------------------------------------------------------------------------
# OIDN baseline
# -----------------------------------------------------------------------------

def write_exr(path: str, img: np.ndarray):
    import pyexr
    pyexr.write(path, img.astype(np.float32))

def read_exr(path: str) -> np.ndarray:
    import pyexr
    try:
        return pyexr.read(path).astype(np.float32)
    except Exception:
        return pyexr.open(path).get().astype(np.float32)


def write_pfm(path: str, img: np.ndarray):
    """Write RGB float32 PFM. OIDN's oidnDenoise CLI reliably supports PFM."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        header = "Pf"
        data = img[..., 0]
    elif img.shape[2] >= 3:
        header = "PF"
        data = img[..., :3]
    else:
        raise ValueError(f"Unsupported PFM shape: {img.shape}")
    data = np.flipud(data).astype("<f4")
    with open(path, "wb") as f:
        f.write(f"{header}\n{img.shape[1]} {img.shape[0]}\n-1.0\n".encode("ascii"))
        data.tofile(f)


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")
        line = f.readline().decode("ascii").strip()
        while line.startswith("#"):
            line = f.readline().decode("ascii").strip()
        w, h = map(int, line.split())
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(f, endian + "f4")
        data = data.reshape((h, w, channels))
        data = np.flipud(data)
        if channels == 1:
            data = data[..., 0]
        return data.astype(np.float32)


def write_image_for_oidn(path: str, img: np.ndarray):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pfm":
        write_pfm(path, img)
    elif ext == ".exr":
        write_exr(path, img)
    else:
        raise ValueError(f"Unsupported OIDN input extension: {ext}; use .pfm or .exr")


def read_image_for_oidn(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pfm":
        return read_pfm(path)
    if ext == ".exr":
        return read_exr(path)
    raise ValueError(f"Unsupported OIDN output extension: {ext}; use .pfm or .exr")


def run_oidn_cli(input_linear: np.ndarray, aux_raw: np.ndarray, cmd_template: str, warmup: int, runs: int, work_dir: str):
    os.makedirs(work_dir, exist_ok=True)
    # Use PFM by default because this oidnDenoise build rejected pyexr-written EXR.
    color_path = os.path.join(work_dir, "oidn_color.pfm")
    albedo_path = os.path.join(work_dir, "oidn_albedo.pfm")
    normal_path = os.path.join(work_dir, "oidn_normal.pfm")

    # Use PFM for the OIDN CLI output because this OIDN build rejected EXR input.
    # Then convert the result to EXR for convenient downstream inspection.
    out_pfm_path = os.path.join(work_dir, "oidn_out.pfm")
    out_exr_path = os.path.join(work_dir, "oidn_out.exr")

    write_image_for_oidn(color_path, input_linear)
    write_image_for_oidn(albedo_path, np.clip(aux_raw[..., :3], 0.0, None).astype(np.float32))
    write_image_for_oidn(normal_path, aux_raw[..., 3:6].astype(np.float32))

    cmd = cmd_template.format(
        color=color_path,
        albedo=albedo_path,
        normal=normal_path,
        out=out_pfm_path,       # backward-compatible field
        out_pfm=out_pfm_path,
        out_exr=out_exr_path,
    )

    def one():
        if os.path.exists(out_pfm_path):
            os.remove(out_pfm_path)
        if os.path.exists(out_exr_path):
            os.remove(out_exr_path)
        t0 = time.perf_counter()
        proc = subprocess.run(
            shlex.split(cmd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            raise RuntimeError(
                "OIDN command failed\n"
                f"cmd: {cmd}\n"
                f"returncode: {proc.returncode}\n"
                f"stdout:\n{proc.stdout[-4000:]}\n"
                f"stderr:\n{proc.stderr[-4000:]}"
            )
        return elapsed

    for _ in range(max(0, warmup)):
        one()
    vals = [one() for _ in range(max(1, runs))]
    if not os.path.exists(out_pfm_path):
        raise RuntimeError(f"OIDN output was not created: {out_pfm_path}")

    oidn_linear = read_image_for_oidn(out_pfm_path)

    # User-facing OIDN output as EXR. If pyexr/OpenEXR writing is unavailable,
    # keep the PFM output and report the warning in metrics.
    exr_warning = None
    try:
        write_exr(out_exr_path, oidn_linear)
    except Exception as e:
        exr_warning = repr(e)

    stats = {
        "oidn_cmd": cmd,
        "oidn_tmp_out_pfm": out_pfm_path,
        "oidn_tmp_out_exr": out_exr_path if os.path.exists(out_exr_path) else None,
        "oidn_tmp_out_exr_warning": exr_warning,
        "oidn_avg_ms_including_exr_io_subprocess": float(np.mean(vals)),
        "oidn_med_ms_including_exr_io_subprocess": float(np.median(vals)),
        "oidn_min_ms_including_exr_io_subprocess": float(np.min(vals)),
        "oidn_max_ms_including_exr_io_subprocess": float(np.max(vals)),
    }
    return oidn_linear, stats


# -----------------------------------------------------------------------------
# Inference / timing
# -----------------------------------------------------------------------------

def input_tensor(x_log: np.ndarray, f_net: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(x_log).permute(2, 0, 1).unsqueeze(0).float()
    f = torch.from_numpy(f_net).permute(2, 0, 1).unsqueeze(0).float()
    return torch.cat([x, f], dim=1).to(device)

def _can_fast_torch2trt(mod: nn.Module, inp: torch.Tensor) -> bool:
    """True when mod is TiledWrapper(Torch2TRTConcatWrapper) and input exactly matches engine tile."""
    if not isinstance(mod, TiledWrapper):
        return False
    if not isinstance(mod.tile_module, Torch2TRTConcatWrapper):
        return False
    _, _, h, w = inp.shape
    return h == mod.tile and w == mod.tile


def run_engine_once(mod: nn.Module, inp: torch.Tensor) -> torch.Tensor:
    """Run one inference.

    For torch2trt full-frame fixed-shape engines, bypass TiledWrapper and feed
    prepared x/f directly. This removes viewer-side split/stitch overhead for the
    actual displayed result.
    """
    if _can_fast_torch2trt(mod, inp):
        x, f = mod.tile_module.prepare_inputs(inp)
        return mod.tile_module.forward_prepared(x, f)
    return mod(inp)


def measure_engine(mod: nn.Module, inp: torch.Tensor, warmup: int = 3, runs: int = 10) -> Dict[str, float]:
    # Fast path for torch2trt full-frame fixed-shape engine:
    # prepare x/f once, then time only TRTModule(x, f).
    fast_torch2trt = _can_fast_torch2trt(mod, inp)

    with torch.no_grad():
        if fast_torch2trt:
            x, f = mod.tile_module.prepare_inputs(inp)
            def call():
                return mod.tile_module.forward_prepared(x, f)
        else:
            def call():
                return mod(inp)

        for _ in range(warmup):
            _ = call()
        if inp.is_cuda:
            torch.cuda.synchronize()

        vals = []
        for _ in range(runs):
            if inp.is_cuda:
                st = torch.cuda.Event(enable_timing=True)
                ed = torch.cuda.Event(enable_timing=True)
                st.record()
                _ = call()
                ed.record()
                torch.cuda.synchronize()
                vals.append(float(st.elapsed_time(ed)))
            else:
                t0 = time.perf_counter()
                _ = call()
                vals.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(vals, dtype=np.float64)
    return {
        "engine_avg_ms": float(arr.mean()),
        "engine_med_ms": float(np.median(arr)),
        "engine_min_ms": float(arr.min()),
        "engine_max_ms": float(arr.max()),
        "engine_std_ms": float(arr.std()),
        "timing_fast_path": bool(fast_torch2trt),
        "timing_fast_path_note": (
            "torch2trt full-frame: pre-split x/f once and time TRTModule(x,f) directly"
            if fast_torch2trt else "standard viewer wrapper path"
        ),
    }



def compact_engine_kind(engine_name: str, source_info: Dict[str, object]) -> str:
    """Return compact family/runtime label for the Last 5 runs table."""
    name = str(engine_name or "")
    source = str(source_info.get("engine_source_name") or source_info.get("engine_source") or "")
    raw_kind = str(source_info.get("engine_kind") or "")

    text = f"{name} {source} {raw_kind}".lower()

    is_conv = (
        "jsa+conv" in text
        or "jsa_conv" in text
        or "jsacnn" in text
        or "swinir" in text
        or "conv_decoder" in text
    )

    is_pth = (
        name.startswith("JSA PTH")
        or name.startswith("JSA+Conv PTH")
        or raw_kind.startswith("PTH")
        or raw_kind.startswith("PTH-")
    )

    family = "JSA+Conv" if is_conv else "JSA"
    runtime = "pth" if is_pth else "trt"
    return f"{family}-{runtime}"

# -----------------------------------------------------------------------------
# Gradio app
# -----------------------------------------------------------------------------

class ViewerApp:
    def __init__(self, data_root: str, trt_root: str, engine_root: str, output_root: str, device: Optional[str]):
        self.data_root = os.path.abspath(data_root)
        self.trt_root = os.path.abspath(trt_root)
        self.engine_root = os.path.abspath(engine_root)
        self.output_root = os.path.abspath(output_root)
        self.engine_mgr = EngineManager(self.data_root, self.trt_root, self.engine_root, device)
        self.samples = discover_samples(self.data_root)
        self.sample_map = {s.key: s for s in self.samples}
        self.run_history: List[Dict[str, object]] = []

    def history_markdown(self) -> str:
        """Compact HTML history table.

        Columns intentionally avoid both Engine and Source because they duplicate
        information already encoded in Kind and in the saved output folder.
        """
        if not self.run_history:
            return "### Last 5 runs\n_No runs yet._"

        def esc(v):
            return html.escape("" if v is None else str(v), quote=True)

        def fmt(v, nd=4):
            if v is None:
                return "-"
            if isinstance(v, (float, int)):
                return f"{float(v):.{nd}g}"
            return str(v)

        rows = self.run_history[-5:][::-1]
        html_rows = []
        for i, r in enumerate(rows, 1):
            scene = esc(r.get("scene", ""))
            kind = esc(r.get("kind_compact") or r.get("kind", ""))
            relmse = esc(fmt(r.get("relmse_exr")))
            oidn_relmse = esc(fmt(r.get("oidn_relmse_exr")))
            time_ms = esc(fmt(r.get("time_ms"), 5))
            oidn_time_ms = esc(fmt(r.get("oidn_time_ms"), 5))
            saved = esc(r.get("saved", ""))
            source = esc(r.get("source", ""))
            engine = esc(r.get("engine", ""))

            html_rows.append(
                "<tr>"
                f"<td class='idx'>{i}</td>"
                f"<td class='scene' title='{scene}'>{scene}</td>"
                f"<td class='kind' title='engine: {engine}&#10;source: {source}'>{kind}</td>"
                f"<td class='num'>{relmse}</td>"
                f"<td class='num'>{oidn_relmse}</td>"
                f"<td class='num'>{time_ms}</td>"
                f"<td class='num'>{oidn_time_ms}</td>"
                f"<td class='saved'><code title='{saved}'>{saved}</code></td>"
                "</tr>"
            )

        style = """
<style>
.viewer-history-wrap {
  overflow-x: auto;
  max-width: 100%;
  border: 1px solid #ddd;
  border-radius: 6px;
}
.viewer-history {
  border-collapse: collapse;
  width: 100%;
  min-width: 980px;
  table-layout: auto;
  font-size: 13px;
}
.viewer-history th, .viewer-history td {
  border: 1px solid #d0d0d0;
  padding: 6px 8px;
  vertical-align: middle;
}
.viewer-history th {
  background: #f5f7fa;
  font-weight: 700;
  white-space: nowrap;
}
.viewer-history td.idx {
  width: 34px;
  text-align: right;
  white-space: nowrap;
}
.viewer-history td.scene {
  max-width: 210px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.viewer-history td.kind {
  width: 120px;
  white-space: nowrap;
  font-weight: 600;
}
.viewer-history td.num {
  width: 92px;
  white-space: nowrap;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.viewer-history td.saved {
  max-width: 520px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.viewer-history td.saved code {
  font-size: 10.5px;
  white-space: nowrap;
}
</style>
"""
        table = (
            style
            + "<div class='viewer-history-wrap'>"
            + "<table class='viewer-history'>"
            + "<thead><tr>"
            + "<th>#</th><th>Scene</th><th>Kind</th>"
            + "<th>relMSE(EXR)</th><th>OIDN relMSE</th>"
            + "<th>Time(ms)</th><th>OIDN Time(ms)</th><th>Saved</th>"
            + "</tr></thead>"
            + "<tbody>"
            + "".join(html_rows)
            + "</tbody></table></div>"
        )
        return "### Last 5 runs\n" + table

    def refresh(self):
        self.samples = discover_samples(self.data_root)
        self.sample_map = {s.key: s for s in self.samples}
        sample_keys = list(self.sample_map.keys())
        engines = self.engine_mgr.choices()
        ckpts = checkpoints(self.data_root)
        return (
            gr.update(choices=sample_keys, value=(sample_keys[0] if sample_keys else None)),
            gr.update(choices=engines, value=(engines[0] if engines else None)),
            gr.update(choices=ckpts, value=(ckpts[0] if ckpts else None)),
        )

    def run(self, sample_key: str, engine_name: str, ckpt_path: str, run_oidn: bool, oidn_cmd_template: str, center_x: float, center_y: float, crop_size: int, engine_warmup: int, engine_runs: int):
        if sample_key not in self.sample_map:
            raise KeyError(f"Unknown sample: {sample_key}")

        s = self.sample_map[sample_key]
        x_log, f_net, input_linear, aux_raw = load_input_npz(s.input_npz)
        gt_log = load_target_npz(s.target_npz)
        gt_linear = None if gt_log is None else expm1_nonneg(gt_log)

        inp = input_tensor(x_log, f_net, self.engine_mgr.device)
        mod = self.engine_mgr.get(engine_name, ckpt_path)

        with torch.no_grad():
            pred_log_t = run_engine_once(mod, inp)
        pred_log = pred_log_t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        pred_linear = expm1_nonneg(pred_log)
        timing = measure_engine(mod, inp, warmup=int(engine_warmup), runs=int(engine_runs))

        input_u8 = rgb_to_uint8(simple_tonemap(input_linear))
        pred_u8 = rgb_to_uint8(simple_tonemap(pred_linear))
        ref_u8 = None if gt_linear is None else rgb_to_uint8(simple_tonemap(gt_linear))
        main_zoom = zoom_grid_u8(input_u8, pred_u8, None, ref_u8, center_x, center_y, crop_size)
        main_err = error_map_u8(pred_linear, gt_linear)

        source_info = self.engine_mgr.source_info(engine_name, ckpt_path)
        metrics = {
            "sample": sample_key,
            "scene": Path(sample_key).stem,
            "engine": engine_name,
            **source_info,
            "input_npz": s.input_npz,
            "target_npz": s.target_npz,
            "target_available": gt_log is not None,
            "stats_source": "live viewer run, not loaded from benchmark results.json",
            "comparison_left": "Input",
            "comparison_right": engine_name,
            **timing,
            **calc_log_metrics(pred_log, gt_log),
            **calc_linear_metrics(pred_linear, gt_linear),
            **calc_exr_relmse(pred_linear, gt_linear),
        }
        saved_dir = save_outputs(self.output_root, sample_key, engine_name, pred_log, pred_linear, pred_u8, metrics)

        oidn_status = "<em>OIDN disabled. Enable the checkbox above to run OIDN.</em>"
        oidn_comp = None
        engine_oidn_comp = None
        oidn_err_update = gr.update(value=None, visible=False)
        oidn_pred_update = gr.update(value=None, visible=False)
        oidn_state = None

        if run_oidn:
            try:
                tmp_dir = os.path.join(WORKSPACE_ROOT, "benchmark_results", "viewer_oidn_tmp")
                oidn_linear, oidn_stats = run_oidn_cli(
                    input_linear=input_linear,
                    aux_raw=aux_raw,
                    cmd_template=oidn_cmd_template,
                    warmup=1,
                    runs=10,
                    work_dir=tmp_dir,
                )
                oidn_u8 = rgb_to_uint8(simple_tonemap(oidn_linear))
                oidn_comp = comparison_u8(input_u8, oidn_u8, 50.0)
                engine_oidn_comp = comparison_u8(pred_u8, oidn_u8, 50.0)
                main_zoom = zoom_grid_u8(input_u8, pred_u8, oidn_u8, ref_u8, center_x, center_y, crop_size)
                oidn_err = error_map_u8(oidn_linear, gt_linear)
                oidn_metrics = {
                    "oidn_enabled": True,
                    "oidn_comparison_left": "Input",
                    "oidn_comparison_right": "OIDN",
                    **oidn_stats,
                    **{f"oidn_{k}": v for k, v in calc_linear_metrics(oidn_linear, gt_linear).items()},
                    **{f"oidn_{k}": v for k, v in calc_log_metrics(log1p_nonneg(oidn_linear), gt_log).items()},
                    **{f"oidn_{k}": v for k, v in calc_exr_relmse(oidn_linear, gt_linear).items()},
                }
                metrics.update(oidn_metrics)
                save_outputs(self.output_root, sample_key, "OIDN", log1p_nonneg(oidn_linear), oidn_linear, oidn_u8, {**metrics, **oidn_metrics})
                oidn_status = "OIDN completed."
                oidn_err_update = gr.update(value=oidn_err, visible=(oidn_err is not None))
                oidn_pred_update = gr.update(value=oidn_u8, visible=True)
                oidn_state = oidn_u8
            except Exception as e:
                oidn_status = f"<pre>OIDN failed:\n{e}</pre>"
                metrics["oidn_error"] = str(e)

        self.run_history.append({
            "scene": Path(sample_key).stem,
            "engine": engine_name,
            "kind": source_info.get("engine_kind"),
            "kind_compact": compact_engine_kind(engine_name, source_info),
            "source": source_info.get("engine_source_name"),
            "relmse_exr": metrics.get("relMSE_linear_exr"),
            "oidn_relmse_exr": metrics.get("oidn_relMSE_linear_exr"),
            "time_ms": metrics.get("engine_avg_ms"),
            "oidn_time_ms": metrics.get("oidn_avg_ms_including_exr_io_subprocess"),
            "saved": saved_dir,
        })
        self.run_history = self.run_history[-5:]
        hist = self.history_markdown()

        cleanup_cuda()
        return (
            hist,
            main_zoom,
            gr.update(value=main_err, visible=(main_err is not None)),
            pred_u8,
            gr.update(value=oidn_status, visible=True),
            oidn_err_update,
            oidn_pred_update,
            json.dumps(metrics, indent=2),
            input_u8,
            pred_u8,
            oidn_state,
            ref_u8,
        )

    def update_zoom_grid(self, input_u8, pred_u8, oidn_u8, ref_u8, center_x: float, center_y: float, crop_size: int):
        return zoom_grid_u8(input_u8, pred_u8, oidn_u8, ref_u8, center_x, center_y, crop_size)

    def extract_camera_from_xml(self, scene_xml: str, target_distance: float = 1.0):
        """Extract base camera args from a Mitsuba XML sensor matrix.

        Matches the helper logic from the dataset-generation shell comments:
          origin = M[:3, 3]
          up     = M[:3, 1]
          target = origin + target_distance * M[:3, 2]
        """
        xml_path = scene_xml
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(WORKSPACE_ROOT, xml_path)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        sensors = root.findall(".//sensor")
        if not sensors:
            raise RuntimeError(f"No <sensor> found in XML: {xml_path}")
        sensor = sensors[0]

        fov = 60.0
        for child in sensor.findall("float"):
            if child.attrib.get("name") == "fov":
                fov = float(child.attrib["value"])
                break

        transform = None
        for child in sensor.findall("transform"):
            if child.attrib.get("name") == "to_world":
                transform = child
                break
        if transform is None:
            raise RuntimeError("No <transform name='to_world'> found under the first <sensor>.")

        matrix_node = transform.find("matrix")
        if matrix_node is None:
            raise RuntimeError("No <matrix value='...'> found under sensor/to_world.")

        vals = [float(v) for v in matrix_node.attrib["value"].split()]
        if len(vals) != 16:
            raise RuntimeError(f"Expected 16 matrix values, got {len(vals)}")

        M = np.array(vals, dtype=np.float64).reshape(4, 4)
        origin = M[:3, 3]
        up = M[:3, 1]
        forward = M[:3, 2]
        target = origin + float(target_distance) * forward
        return origin, target, up, fov

    def render_new_input(
        self,
        scene_xml: str,
        dataset_name: str,
        width: int,
        height: int,
        input_spp: int,
        aov_spp: int,
        ref_spp: int,
        ref_chunk_spp: int,
        max_depth: int,
        rr_depth: int,
        origin_jitter: float,
        target_jitter: float,
        seed: int,
        overwrite: bool,
    ):
        """Call codes/generate_dataset.py, then refresh the viewer sample pool."""
        if not dataset_name:
            dataset_name = f"viewer_{int(time.time())}"

        scene_arg = scene_xml
        scene_abs = scene_xml if os.path.isabs(scene_xml) else os.path.join(WORKSPACE_ROOT, scene_xml)
        if not os.path.exists(scene_abs):
            raise FileNotFoundError(f"Scene XML not found: {scene_abs}")

        origin, target, up, fov = self.extract_camera_from_xml(scene_xml)

        gen_py = os.path.join(WORKSPACE_ROOT, "codes", "generate_dataset.py")
        if not os.path.exists(gen_py):
            raise FileNotFoundError(f"generate_dataset.py not found: {gen_py}")

        def vec_args(v):
            return [f"{float(x):.10g}" for x in v]

        cmd = [
            sys.executable, gen_py,
            "--scene", scene_arg,
            "--name", dataset_name,
            "--out-data", self.data_root,
            "--variant", "cuda_ad_rgb",
            "--width", str(int(width)),
            "--height", str(int(height)),
            "--num-views", "1",
            "--input-spp", str(int(input_spp)),
            "--aov-spp", str(int(aov_spp)),
            "--ref-spp", str(int(ref_spp)),
            "--ref-chunk-spp", str(int(ref_chunk_spp)),
            "--camera-mode", "fixed",
            "--base-origin", *vec_args(origin),
            "--base-target", *vec_args(target),
            "--base-up", *vec_args(up),
            "--base-fov", f"{float(fov):.10g}",
            "--origin-jitter", str(float(origin_jitter)),
            "--target-jitter", str(float(target_jitter)),
            "--max-depth", str(int(max_depth)),
            "--rr-depth", str(int(rr_depth)),
            "--write-npz",
            "--seed", str(int(seed)),
        ]
        if overwrite:
            cmd.append("--overwrite")

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0

        # refresh pool
        self.samples = discover_samples(self.data_root)
        self.sample_map = {s.key: s for s in self.samples}
        sample_keys = list(self.sample_map.keys())
        preferred = None
        for k in sample_keys:
            if dataset_name in k:
                preferred = k
                break

        status = (
            f"### Render new input\\n"
            f"`cmd`: `{' '.join(shlex.quote(c) for c in cmd)}`\\n\\n"
            f"`elapsed`: {elapsed:.2f}s\\n\\n"
            f"`returncode`: {proc.returncode}\\n\\n"
            f"#### stdout\\n```text\\n{proc.stdout[-4000:]}\\n```\\n"
            f"#### stderr\\n```text\\n{proc.stderr[-4000:]}\\n```"
        )
        if proc.returncode != 0:
            status = "### Render failed\\n" + status

        return (
            status,
            gr.update(choices=sample_keys, value=(preferred or (sample_keys[0] if sample_keys else None))),
        )

    def launch(self, host: str, port: int, share: bool, auth_user: Optional[str] = None, auth_pass: Optional[str] = None):
        sample_keys = list(self.sample_map.keys())
        engines = self.engine_mgr.choices()
        ckpts = checkpoints(self.data_root)
        default_oidn_cmd = "oidnDenoise -f RT -hdr {color} -alb {albedo} -nrm {normal} -o {out}"

        with gr.Blocks(title="JSA / JSA+Conv / TRT / torch2trt / OIDN Viewer") as demo:
            gr.Markdown("## JSA / JSA+Conv / TRT / torch2trt / OIDN Interactive Viewer")
            gr.Markdown(
                f"`data_root={self.data_root}` / `trt_root={self.trt_root}` / "
                f"`engine_root={self.engine_root}` / "
                f"`output_root={self.output_root}`"
            )

            history_md = gr.Markdown(value=self.history_markdown())

            with gr.Row():
                sample_dd = gr.Dropdown(label="Input file", choices=sample_keys, value=(sample_keys[0] if sample_keys else None), scale=4)
                engine_dd = gr.Dropdown(label="Denoising engine", choices=engines, value=(engines[0] if engines else None), scale=3)
            with gr.Row():
                ckpt_dd = gr.Dropdown(label="Checkpoint for selected PTH engine", choices=ckpts, value=(ckpts[0] if ckpts else None), scale=4)
                refresh_btn = gr.Button("Refresh pool", scale=1)
                run_btn = gr.Button("Run", variant="primary", scale=1)

            with gr.Accordion("Timing settings", open=False):
                with gr.Row():
                    engine_warmup = gr.Number(label="Engine warmup runs", value=20, precision=0)
                    engine_runs = gr.Number(label="Engine timed runs", value=100, precision=0)
                gr.Markdown(
                    "For torch2trt full-frame engines, viewer timing uses a fast path: "
                    "x/f are split once, then only `TRTModule(x, f)` is timed."
                )

            with gr.Accordion("Render a new Mitsuba input", open=False):
                with gr.Row():
                    scene_xml = gr.Textbox(label="Scene XML", value="classroom/classroom.xml", scale=3)
                    dataset_name = gr.Textbox(label="Dataset name", value="viewer_classroom_input", scale=2)
                with gr.Row():
                    render_width = gr.Number(label="Width", value=1024, precision=0)
                    render_height = gr.Number(label="Height", value=1024, precision=0)
                    render_input_spp = gr.Number(label="Input spp", value=4, precision=0)
                    render_aov_spp = gr.Number(label="AOV spp", value=4, precision=0)
                    render_ref_spp = gr.Number(label="Ref spp", value=1024, precision=0)
                    render_ref_chunk_spp = gr.Number(label="Ref chunk spp", value=512, precision=0)
                with gr.Row():
                    render_max_depth = gr.Number(label="Max depth", value=17, precision=0)
                    render_rr_depth = gr.Number(label="RR depth", value=5, precision=0)
                    render_origin_jitter = gr.Number(label="Origin jitter", value=0.0)
                    render_target_jitter = gr.Number(label="Target jitter", value=0.05)
                    render_seed = gr.Number(label="Seed", value=1234, precision=0)
                    render_overwrite = gr.Checkbox(label="Overwrite this dataset name only", value=True)
                gr.Markdown(
                    "Overwrite only removes `data/__train_scenes__/<Dataset name>` and "
                    "`data/__test_scenes__/<Dataset name>`. It does not touch other datasets."
                )
                render_btn = gr.Button("Render one new input with Mitsuba3")
                render_status = gr.Markdown()

            with gr.Accordion("OIDN baseline", open=False):
                run_oidn = gr.Checkbox(label="Run OIDN below selected engine", value=True)
                oidn_cmd = gr.Textbox(
                    label="OIDN command template",
                    value=default_oidn_cmd,
                    lines=1,
                    info="Available fields: {color}, {albedo}, {normal}, {out}={out_pfm}, {out_pfm}, {out_exr}. OIDN CLI uses PFM, then viewer writes EXR."
                )

            input_state = gr.State(value=None)
            pred_state = gr.State(value=None)
            oidn_state = gr.State(value=None)
            ref_state = gr.State(value=None)

            gr.Markdown("### Interactive zoom comparison")
            gr.Markdown("2x2 crop: **Input / Selected engine / OIDN / Reference**. Choose x/y center and crop size.")
            with gr.Row():
                center_x_slider = gr.Slider(minimum=0, maximum=1024, value=335, step=1, label="x center", interactive=True)
                center_y_slider = gr.Slider(minimum=0, maximum=1024, value=770, step=1, label="y center", interactive=True)
                crop_size_slider = gr.Slider(minimum=32, maximum=1024, value=128, step=16, label="crop size", interactive=True)

            main_comp = gr.Image(
                label="Zoomed comparison: input / selected / OIDN / reference",
                type="numpy",
                height=520,
            )

            gr.Markdown("### Full-frame outputs")
            with gr.Row():
                main_err = gr.Image(label="Selected engine error map", type="numpy", visible=False, height=300)
                main_pred = gr.Image(label="Selected engine denoised", type="numpy", height=300)

            gr.Markdown("### OIDN baseline")
            oidn_status = gr.HTML(value="<em>OIDN disabled. Enable it above if needed.</em>")
            with gr.Row():
                oidn_err = gr.Image(label="OIDN error map", type="numpy", visible=False, height=300)
                oidn_pred = gr.Image(label="OIDN denoised", type="numpy", visible=False, height=300)

            metrics = gr.Code(label="Timing / metrics", language="json")

            run_btn.click(
                self.run,
                inputs=[sample_dd, engine_dd, ckpt_dd, run_oidn, oidn_cmd, center_x_slider, center_y_slider, crop_size_slider, engine_warmup, engine_runs],
                outputs=[history_md, main_comp, main_err, main_pred, oidn_status, oidn_err, oidn_pred, metrics, input_state, pred_state, oidn_state, ref_state],
            )
            for _ctrl in (center_x_slider, center_y_slider, crop_size_slider):
                _ctrl.change(
                    self.update_zoom_grid,
                    inputs=[input_state, pred_state, oidn_state, ref_state, center_x_slider, center_y_slider, crop_size_slider],
                    outputs=[main_comp],
                )
            render_btn.click(
                self.render_new_input,
                inputs=[
                    scene_xml,
                    dataset_name,
                    render_width,
                    render_height,
                    render_input_spp,
                    render_aov_spp,
                    render_ref_spp,
                    render_ref_chunk_spp,
                    render_max_depth,
                    render_rr_depth,
                    render_origin_jitter,
                    render_target_jitter,
                    render_seed,
                    render_overwrite,
                ],
                outputs=[render_status, sample_dd],
            )
            refresh_btn.click(self.refresh, outputs=[sample_dd, engine_dd, ckpt_dd])

        auth = None
        if auth_user or auth_pass:
            if not (auth_user and auth_pass):
                raise ValueError("Both auth_user and auth_pass must be provided when enabling Gradio auth.")
            auth = (auth_user, auth_pass)

        demo.launch(server_name=host, server_port=port, share=share, auth=auth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(WORKSPACE_ROOT / "data"))
    ap.add_argument("--trt-root", default=str(WORKSPACE_ROOT / "benchmark_results" / "trt"))
    ap.add_argument("--engine-root", default=str(WORKSPACE_ROOT / "benchmark_results" / "engine"))
    ap.add_argument("--output-root", default=str(WORKSPACE_ROOT / "benchmark_results" / "viewer_outputs"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--auth-user", default=os.environ.get("GRADIO_AUTH_USER"))
    ap.add_argument("--auth-pass", default=os.environ.get("GRADIO_AUTH_PASS"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ViewerApp(args.data_root, args.trt_root, args.engine_root, args.output_root, args.device).launch(
        args.host,
        args.port,
        args.share,
        auth_user=args.auth_user,
        auth_pass=args.auth_pass,
    )


if __name__ == "__main__":
    main()