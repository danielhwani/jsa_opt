#  Copyright (c) 2024 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import collections
import copy
import os
import random
import numpy as np
import torch
import datetime
import glob
from pytz import timezone
from torch.utils.data import DataLoader

from config import config
import utils.utils_image as util_image
import utils.utils_options as option
import dataset
import model.model_joint_sa as model_joint_sa
import model.model_joint_sa_v2_int8 as model_joint_sa_v2_int8
import preprocess as pre
import eval as eval
import torch.nn.functional as F
import loss as L


import torch_tensorrt
from torch.export import export
import torch.profiler
import tensorrt as trt
from torch.export import load as load_exported_program
from torch_tensorrt.ts import ptq 
import pyexr
import torch.nn as nn


class SpaceToDepth(nn.Module):
    """PixelUnshuffle(r)와 동일 동작을 reshape/permute로 구현 (TRT 호환)"""
    def __init__(self, r: int):
        super().__init__()
        self.r = int(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        r = self.r
        # H, W가 r로 나누어떨어져야 함
        assert (H % r == 0) and (W % r == 0), f"Input H,W must be divisible by {r}, got {(H,W)}"
        x = x.view(B, C, H // r, r, W // r, r)             # [B,C,H/r,r,W/r,r]
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()       # [B,C,r,r,H/r,W/r]
        x = x.view(B, C * r * r, H // r, W // r)           # [B,C*r*r,H/r,W/r]
        return x

def _replace_pixel_unshuffle(module: nn.Module):
    """모델 안의 nn.PixelUnshuffle을 SpaceToDepth로 교체"""
    for name, child in list(module.named_children()):
        # PyTorch의 PixelUnshuffle 모듈 타입은 아래와 동일
        if isinstance(child, nn.PixelUnshuffle):
            r = int(child.downscale_factor)
            setattr(module, name, SpaceToDepth(r))
        else:
            _replace_pixel_unshuffle(child)

class OriginalConcatWrapper(nn.Module):
    """Expose original JSA_transformer(x, f) as a single concat-input module."""
    def __init__(self, model: nn.Module, in_x: int, in_f: int):
        super().__init__()
        self.model = model
        self.in_x = in_x
        self.in_f = in_f

    def forward(self, input):
        x = input[:, :self.in_x]
        f = input[:, self.in_x:self.in_x + self.in_f]
        return self.model(x, f)


class TiledOriginalFrameWrapper(nn.Module):
    """Run the original 128x128-patch model over a padded full frame."""
    def __init__(self, model: nn.Module, patch_size: int):
        super().__init__()
        self.model = model
        self.patch_size = patch_size

    def forward(self, input_tensor):
        _, _, height, width = input_tensor.shape
        patch = self.patch_size
        outputs = []
        for y0 in range(0, height, patch):
            row = []
            for x0 in range(0, width, patch):
                tile = input_tensor[:, :, y0:y0 + patch, x0:x0 + patch]
                row.append(self.model(tile))
            outputs.append(torch.cat(row, dim=3))
        return torch.cat(outputs, dim=2)


class SquarePadWrapper(nn.Module):
    def __init__(self, model: nn.Module, square_size: int):
        super().__init__()
        self.model = model
        self.square_size = square_size

    def forward(self, input_tensor):
        _, _, height, width = input_tensor.shape
        pad_height = self.square_size - height
        pad_width = self.square_size - width
        if pad_height < 0 or pad_width < 0:
            raise RuntimeError("Input is larger than configured square size.")
        if pad_height >= height or pad_width >= width:
            raise RuntimeError(
                "Reflect padding must be smaller than the input dimension. "
                f"Got input HxW={height}x{width}, pad_h={pad_height}, pad_w={pad_width}."
            )
        padded = F.pad(input_tensor, (0, pad_width, 0, pad_height), "reflect")
        return self.model(padded)


class DownsampleCenterCropWrapper(nn.Module):
    def __init__(self, model: nn.Module, down_size: int, crop_size: int):
        super().__init__()
        self.model = model
        self.down_size = down_size
        self.crop_size = crop_size

    def forward(self, input_tensor):
        x = F.interpolate(
            input_tensor,
            scale_factor=1.0 / self.down_size,
            mode="bilinear",
            align_corners=False,
        )
        _, _, height, width = x.shape
        crop = self.crop_size
        top = max((height - crop) // 2, 0)
        left = max((width - crop) // 2, 0)
        x = x[:, :, top:top + crop, left:left + crop]
        return self.model(x)


def build_original_model(img_size: int, in_x: int, in_f: int, embedded_dim: int, device):
    model = model_joint_sa.JSA_transformer(
        img_size=img_size,
        embedded_dim=embedded_dim,
        win_size=8,
        projection_option='linear',
        ffn_option='mlp',
        depths=[1, 2, 4, 8, 2, 8, 4, 2, 4],
        in_x=in_x,
        in_f=in_f,
    )
    return OriginalConcatWrapper(model, in_x, in_f).to(device).eval()


def build_tiled_original_frame_model(patch_size: int, in_x: int, in_f: int, embedded_dim: int, device):
    patch_model = build_original_model(patch_size, in_x, in_f, embedded_dim, device)
    return TiledOriginalFrameWrapper(patch_model, patch_size).to(device).eval()


def build_optimized_model(img_size: int, in_x: int, in_f: int, embedded_dim: int, device, state_dict=None):
    model = model_joint_sa_v2_int8.JSA_transformer(
        img_size=img_size,
        embedded_dim=embedded_dim,
        win_size=8,
        projection_option='linear',
        ffn_option='mlp',
        depths=[1, 2, 4, 8, 2, 8, 4, 2, 4],
        in_x=in_x,
        in_f=in_f,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    _replace_pixel_unshuffle(model)
    model_joint_sa_v2_int8.patch_model_for_int8(model)
    return model.to(device).eval()


def summarize_timings(timings, trim_percent: float):
    timings_np = np.asarray(timings, dtype=np.float64)
    sorted_np = np.sort(timings_np)
    trim_count = int(len(sorted_np) * trim_percent / 100.0)
    if trim_count * 2 >= len(sorted_np):
        trimmed_np = sorted_np
        trim_count = 0
    else:
        trimmed_np = sorted_np[trim_count:len(sorted_np) - trim_count]

    return {
        "min": float(timings_np.min()),
        "max": float(timings_np.max()),
        "avg": float(timings_np.mean()),
        "std": float(timings_np.std()),
        "median": float(np.median(timings_np)),
        "p10": float(np.percentile(timings_np, 10)),
        "p90": float(np.percentile(timings_np, 90)),
        "trim_avg": float(trimmed_np.mean()),
        "trim_count": trim_count,
    }


def benchmark_cuda(name: str, module: nn.Module, input_tensor: torch.Tensor, warmup: int, iters: int, trim_percent: float):
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(input_tensor)
        torch.cuda.synchronize()

        timings = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = module(input_tensor)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))

    stats = summarize_timings(timings, trim_percent)
    print(
        f"{name:42s}: "
        f"avg={stats['avg']:9.4f} ms  "
        f"med={stats['median']:9.4f}  "
        f"min={stats['min']:9.4f}  "
        f"max={stats['max']:9.4f}  "
        f"std={stats['std']:8.4f}  "
        f"trim{trim_percent:g}%={stats['trim_avg']:9.4f}"
    )
    return stats


def check_close(name: str, reference: torch.Tensor, candidate: torch.Tensor):
    ref = reference.detach().float()
    cand = candidate.detach().float()
    max_abs = (ref - cand).abs().max().item()
    mean_abs = (ref - cand).abs().mean().item()
    rel_l2 = torch.linalg.vector_norm(ref - cand) / torch.clamp(torch.linalg.vector_norm(ref), min=1e-12)
    print(f"{name:42s}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, rel_l2={rel_l2.item():.6e}")


def profile_leaf_modules(name: str, module: nn.Module, input_tensor: torch.Tensor, topk: int):
    records = collections.defaultdict(list)
    pending = {}
    hooks = []

    def is_leaf(m: nn.Module):
        return len(list(m.children())) == 0

    for module_name, child in module.named_modules():
        if not module_name or not is_leaf(child):
            continue

        def pre_hook(_mod, _inputs, key=module_name):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pending.setdefault(key, []).append(event)

        def post_hook(_mod, _inputs, _output, key=module_name):
            start_events = pending.get(key)
            if not start_events:
                return
            start_event = start_events.pop()
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            records[key].append((start_event, end_event, _mod.__class__.__name__))

        hooks.append(child.register_forward_pre_hook(pre_hook))
        hooks.append(child.register_forward_hook(post_hook))

    with torch.no_grad():
        _ = module(input_tensor)
        torch.cuda.synchronize()

    for hook in hooks:
        hook.remove()

    rows = []
    for key, events in records.items():
        elapsed = [start.elapsed_time(end) for start, end, _class_name in events]
        class_name = events[0][2]
        rows.append((sum(elapsed), key, class_name))

    rows.sort(reverse=True)
    print(f"\nLeaf-module CUDA profile: {name}")
    print(f"{'rank':>4s} {'ms':>10s} {'class':24s} module")
    for rank, (elapsed_ms, key, class_name) in enumerate(rows[:topk], start=1):
        print(f"{rank:4d} {elapsed_ms:10.4f} {class_name:24s} {key}")


def export_torch_profile(name: str, module: nn.Module, input_tensor: torch.Tensor, profile_dir: str, iters: int):
    os.makedirs(profile_dir, exist_ok=True)
    trace_path = os.path.join(profile_dir, f"{name}.chrome_trace.json")
    table_path = os.path.join(profile_dir, f"{name}.key_averages.txt")
    tensorboard_dir = os.path.join(profile_dir, "tensorboard")

    with torch.no_grad():
        _ = module(input_tensor)
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(iters):
                with torch.profiler.record_function(name):
                    _ = module(input_tensor)
                torch.cuda.synchronize()
                prof.step()

    prof.export_chrome_trace(trace_path)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    with open(table_path, "w") as f:
        f.write(table)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir, worker_name=name),
    ) as tb_prof:
        with torch.no_grad():
            for _ in range(iters):
                with torch.profiler.record_function(name):
                    _ = module(input_tensor)
                torch.cuda.synchronize()
                tb_prof.step()

    print(f"[profile] {name}: chrome trace -> {trace_path}")
    print(f"[profile] {name}: key averages -> {table_path}")
    print(f"[profile] {name}: tensorboard trace dir -> {tensorboard_dir}")
    print(table)


def padded_square_size(height: int, width: int, factor: int):
    # Match eval.py exactly: ((dim + factor) // factor) * factor.
    padded_height = ((height + factor) // factor) * factor
    padded_width = ((width + factor) // factor) * factor
    return max(padded_height, padded_width)


def downsampled_test_size(height: int, width: int, down_size: int, factor: int):
    down_height = int(height * (1.0 / down_size))
    down_width = int(width * (1.0 / down_size))
    padded_height = ((down_height + factor - 1) // factor) * factor
    padded_width = ((down_width + factor - 1) // factor) * factor
    return min(padded_height, padded_width), down_height, down_width, padded_height, padded_width

def compile_trt_ts(
    name: str,
    module: nn.Module,
    example_input: torch.Tensor,
    precision: str,
    require_full_compilation: bool,
    debug: bool = True,
    dump_graph: bool = True,
):
    print(f"[TRT] compiling {name} ({precision})...")
    print(f"[TRT] require_full_compilation={require_full_compilation}")

    calibrator = None
    if precision == "fp16":
        trace_input = example_input.half()
        module = module.half().eval()
        trt_input = torch_tensorrt.Input(trace_input.shape, dtype=torch.half)
        enabled_precisions = {torch.half}
    elif precision == "int8":
        trace_input = example_input.float()
        module = module.float().eval()
        trt_input = torch_tensorrt.Input(trace_input.shape, dtype=torch.float)
        enabled_precisions = {torch.int8, torch.half}
        
        class _RandomCalibDataset(torch.utils.data.Dataset):
            def __init__(self, ex_shape, num_samples=16):
                self.b, self.c, self.h, self.w = ex_shape
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return torch.randn(self.c, self.h, self.w, dtype=torch.float)

        calib_loader = torch.utils.data.DataLoader(
            _RandomCalibDataset(trace_input.shape, num_samples=16),
            batch_size=trace_input.shape[0],
            shuffle=False,
        )
        
        cache_path = f"int8_calib_{name.replace(' ', '_')}.cache"
        calibrator = ptq.DataLoaderCalibrator(
            calib_loader,
            algo_type=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            cache_file=cache_path,
            use_cache=False,
            device=trace_input.device,
        )
    else:
        trace_input = example_input.float()
        module = module.float().eval()
        trt_input = torch_tensorrt.Input(trace_input.shape, dtype=torch.float)
        enabled_precisions = {torch.float}

    print(f"[TRT] trace input shape={tuple(trace_input.shape)}, dtype={trace_input.dtype}")
    print(f"[TRT] enabled_precisions={enabled_precisions}")

    with torch.no_grad():
        ts_module = torch.jit.trace(module, trace_input)

    if dump_graph:
        print("\n========== TorchScript graph before TRT ==========")
        print(ts_module.graph)
        print("==================================================\n")

    kwargs = dict(
        ir="ts",
        inputs=[trt_input],
        enabled_precisions=enabled_precisions,
        require_full_compilation=require_full_compilation,
        truncate_long_and_double=True,
        workspace_size=2 << 30,
        min_block_size=1,
    )
    if calibrator is not None:
        kwargs["calibrator"] = calibrator

    # torch-tensorrt version에 따라 debug 인자를 지원하지 않을 수 있음
    if debug:
        kwargs["debug"] = True

    try:
        compiled = torch_tensorrt.compile(ts_module, **kwargs)
    except TypeError:
        # older version fallback: debug argument unsupported
        kwargs.pop("debug", None)
        compiled = torch_tensorrt.compile(ts_module, **kwargs)

    if dump_graph:
        print("\n========== Compiled TorchScript graph ==========")
        try:
            print(compiled.graph)
        except Exception as e:
            print(f"[TRT][WARN] Could not print compiled.graph: {e}")

        try:
            print(compiled._c.dump_to_str(True, False, False))
        except Exception as e:
            print(f"[TRT][WARN] Could not dump compiled module: {e}")
        print("================================================\n")

    return compiled

def random_2x2_downsample(x, scale):
    """
    Downsample by selecting 1 random pixel from each 2x2 block.
    x: tensor [B, C, H, W]
    returns: [B, C, H//2, W//2]
    """
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H, W must be divisible by 2"

    x_patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # [B, C, H//2, W//2, 2, 2]
    
    rand_h = torch.randint(0, 2, (B, 1, H//2, W//2, 1), device=x.device).unsqueeze(-1)
    rand_w = torch.randint(0, 2, (B, 1, H//2, W//2, 1), device=x.device).unsqueeze(-1)

    gathered = x_patches.gather(dim=4, index=rand_h.expand(-1, C, -1, -1, -1, 2))
    gathered = gathered.gather(dim=5, index=rand_w.expand(-1, C, -1, -1, -1, -1))

    out = gathered.squeeze(-1).squeeze(-1)
    return out


def random_downsample(x: torch.Tensor, scale: int):
    B, C, H, W = x.shape
    assert H % scale == 0 and W % scale == 0, f"x shape {x.shape} not divisible by {scale}"
    Hs, Ws = H // scale, W // scale

    # [B, C, Hs, scale, Ws, scale]
    x_patches = x.view(B, C, Hs, scale, Ws, scale)

    # generate random indices
    idx_h = torch.randint(0, scale, (B, 1, Hs, 1, Ws, 1), device=x.device)
    idx_w = torch.randint(0, scale, (B, 1, Hs, 1, Ws, 1), device=x.device)

    # gather along dim=3 (height within patch)
    x_picked = x_patches.gather(3, idx_h.expand(B, C, Hs, 1, Ws, scale))  # shape: [B, C, Hs, 1, Ws, scale]
    
    # gather along dim=5 (width within patch)
    x_picked = x_picked.gather(5, idx_w.expand(B, C, Hs, 1, Ws, 1))  # shape: [B, C, Hs, 1, Ws, 1]

    # remove the singleton dims
    return x_picked.squeeze(3).squeeze(-1)  # [B, C, Hs, Ws]

def preprocess_normal(normal):
    normal = np.nan_to_num(normal)
    normal = (normal + 1.0) * 0.5
    normal = np.maximum(np.minimum(normal, 1.0), 0.0)
    return normal

def preprocess_depth(depth):
    depth = np.clip(depth, 0.0, np.max(depth))
    max_feature = np.max(depth)
    if max_feature != 0:
        depth /= max_feature
    return depth

def preprocess_specular(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))

def preprocess_final(x: torch.Tensor):

    with torch.no_grad():
        # Process x (color: 3 * 3 channels)
        x[:, :3, :, :] = preprocess_specular(x[:, :3, :, :])
        x[:, 3:6, :, :] = preprocess_specular(x[:, 3:6, :, :])
        x[:, 6:9, :, :] = preprocess_specular(x[:, 6:9, :, :])

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["patch", "frame"], default="patch")
    parser.add_argument("--img-size", type=int, default=config["patch_size"])
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--down-size", type=int, default=2)
    parser.add_argument("--original-patch-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--trim-percent", type=float, default=10.0)
    parser.add_argument("--embedded-dim", type=int, default=32)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=config["manual_seed"] or 3040)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="./profiles/timeTest")
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument("--component-topk", type=int, default=30)
    opt = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    seed = opt.seed
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0")
    in_x = config["x_dim"]
    in_f = config["f_dim"]
    channels = in_x + in_f
    if opt.mode == "frame":
        eval_factor = 128
        model_size, down_height, down_width, padded_down_height, padded_down_width = downsampled_test_size(
            opt.frame_height,
            opt.frame_width,
            opt.down_size,
            eval_factor,
        )
        original_model_size = opt.original_patch_size
        full_frame_size = padded_square_size(opt.frame_height, opt.frame_width, eval_factor)
        benchmark_shape = (opt.batch_size, channels, opt.frame_height, opt.frame_width)
        if full_frame_size % opt.original_patch_size != 0:
            raise ValueError("Full padded frame size must be divisible by --original-patch-size.")
    else:
        model_size = opt.img_size
        original_model_size = opt.img_size
        benchmark_shape = (opt.batch_size, channels, opt.img_size, opt.img_size)
        if opt.img_size % 128 != 0:
            raise ValueError("--img-size must be divisible by 128 for this U-Net depth.")

    example_input = torch.randn(
        *benchmark_shape,
        device=device,
        dtype=torch.float,
    )

    print("Benchmark mode: random weights, no checkpoint, synthetic input")
    print(f"Mode: {opt.mode}")
    if opt.mode == "frame":
        print(f"Frame HxW: {opt.frame_height}x{opt.frame_width}")
        print(f"Down size: {opt.down_size}")
        print(f"Downsampled HxW: {down_height}x{down_width}")
        print(f"Downsampled padded HxW: {padded_down_height}x{padded_down_width}")
        print(f"Optimized test_size: {model_size}")
        print(f"Original full-frame padded square: {full_frame_size}")
        print(f"Original model patch size: {original_model_size}")
        print(f"Original tiled patches per frame: {(full_frame_size // original_model_size) ** 2}")
    print(f"Input shape: {tuple(example_input.shape)}")
    print(f"TRT precision: {opt.precision}")
    print(f"Warmup/iters: {opt.warmup}/{opt.iters}")
    print(f"Trim percent: bottom/top {opt.trim_percent:g}%")
    print("")

    if opt.mode == "frame":
        original_tiled = build_tiled_original_frame_model(original_model_size, in_x, in_f, opt.embedded_dim, device)
        original = SquarePadWrapper(original_tiled, full_frame_size).to(device).eval()
        original_state = copy.deepcopy(original_tiled.model.model.state_dict())

        original_tiled_for_trt = build_tiled_original_frame_model(original_model_size, in_x, in_f, opt.embedded_dim, device)
        original_tiled_for_trt.model.model.load_state_dict(original_state, strict=True)
        original_for_trt = SquarePadWrapper(original_tiled_for_trt, full_frame_size).to(device).eval()
    else:
        original = build_original_model(original_model_size, in_x, in_f, opt.embedded_dim, device)
        original_state = copy.deepcopy(original.model.state_dict())
        original_for_trt = build_original_model(original_model_size, in_x, in_f, opt.embedded_dim, device)
        original_for_trt.model.load_state_dict(original_state, strict=True)

    optimized = build_optimized_model(
        model_size,
        in_x,
        in_f,
        opt.embedded_dim,
        device,
        state_dict=original_state,
    )
    optimized_for_trt = build_optimized_model(
        model_size,
        in_x,
        in_f,
        opt.embedded_dim,
        device,
        state_dict=original_state,
    )
    optimized_for_trt_int8 = build_optimized_model(
        model_size,
        in_x,
        in_f,
        opt.embedded_dim,
        device,
        state_dict=original_state,
    )

    if opt.mode == "frame":
        optimized = DownsampleCenterCropWrapper(optimized, opt.down_size, model_size).to(device).eval()
        optimized_for_trt = DownsampleCenterCropWrapper(optimized_for_trt, opt.down_size, model_size).to(device).eval()
        optimized_for_trt_int8 = DownsampleCenterCropWrapper(optimized_for_trt_int8, opt.down_size, model_size).to(device).eval()

    optimized_trt_input = example_input.half() if opt.precision == "fp16" else example_input

    if opt.check_correctness:
        with torch.no_grad():
            original_out = original(example_input)
            optimized_out = optimized(example_input)
            torch.cuda.synchronize()
        print("Correctness checks against original PyTorch")
        check_close("Optimized PyTorch", original_out, optimized_out)
        print("")

    print("[1/5] PyTorch original")
    original_ms = benchmark_cuda(
        "Original PyTorch",
        original,
        example_input,
        opt.warmup,
        opt.iters,
        opt.trim_percent,
    )

    print("[2/5] Naive TensorRT original (fallback allowed, fp32)")
    trt_original = compile_trt_ts(
        "original model",
        original_for_trt,
        example_input,
        "fp32",
        require_full_compilation=False,
    )
    trt_original_ms = benchmark_cuda(
        "Original naive TRT (fp32 fallback)",
        trt_original,
        example_input,
        opt.warmup,
        opt.iters,
        opt.trim_percent,
    )

    print("[3/5] PyTorch optimized")
    optimized_ms = benchmark_cuda(
        "Optimized PyTorch",
        optimized,
        example_input,
        opt.warmup,
        opt.iters,
        opt.trim_percent,
    )

    print("[4/5] TensorRT optimized")
    trt_optimized = compile_trt_ts(
        "optimized model",
        optimized_for_trt,
        example_input,
        opt.precision,
        require_full_compilation=True,
    )
    trt_optimized_ms = benchmark_cuda(
        "Optimized TRT",
        trt_optimized,
        optimized_trt_input,
        opt.warmup,
        opt.iters,
        opt.trim_percent,
    )

    print("[5/5] TensorRT optimized (INT8 hybrid)")
    try:
        trt_optimized_int8 = compile_trt_ts(
            "optimized model int8",
            optimized_for_trt_int8,
            example_input,
            "int8",
            require_full_compilation=True,
        )
        trt_optimized_int8_ms = benchmark_cuda(
            "Optimized TRT (INT8)",
            trt_optimized_int8,
            example_input.float(),
            opt.warmup,
            opt.iters,
            opt.trim_percent,
        )
    except Exception as e:
        print(f"[!] INT8 compilation or benchmark failed: {e}")
        trt_optimized_int8 = None
        trt_optimized_int8_ms = None

    print("")
    print("Summary")
    print(
        f"{'Path':42s} "
        f"{'avg':>10s} {'median':>10s} {'min':>10s} {'max':>10s} "
        f"{'trim_avg':>10s} {'speedup(avg)':>14s} {'speedup(trim)':>14s}"
    )
    
    summary_list = [
        ("Original PyTorch", original_ms),
        ("Original naive TRT (fp32 fallback)", trt_original_ms),
        ("Optimized PyTorch", optimized_ms),
        ("Optimized TRT", trt_optimized_ms),
    ]
    if trt_optimized_int8_ms is not None:
        summary_list.append(("Optimized TRT (INT8)", trt_optimized_int8_ms))

    for name, stats in summary_list:
        print(
            f"{name:42s} "
            f"{stats['avg']:10.4f} {stats['median']:10.4f} {stats['min']:10.4f} {stats['max']:10.4f} "
            f"{stats['trim_avg']:10.4f} "
            f"{original_ms['avg'] / stats['avg']:14.3f}x "
            f"{original_ms['trim_avg'] / stats['trim_avg']:14.3f}x"
        )

    if opt.profile:
        profile_dir = os.path.abspath(opt.profile_dir)
        print("")
        print(f"Writing profiler outputs under: {profile_dir}")
        profile_leaf_modules("original_pytorch", original, example_input, opt.component_topk)
        profile_leaf_modules("optimized_pytorch", optimized, example_input, opt.component_topk)
        export_torch_profile("original_pytorch", original, example_input, profile_dir, opt.profile_iters)
        export_torch_profile("original_naive_trt_fp32_fallback", trt_original, example_input, profile_dir, opt.profile_iters)
        export_torch_profile("optimized_pytorch", optimized, example_input, profile_dir, opt.profile_iters)
        export_torch_profile("optimized_trt", trt_optimized, optimized_trt_input, profile_dir, opt.profile_iters)
        if trt_optimized_int8 is not None:
            export_torch_profile("optimized_trt_int8", trt_optimized_int8, example_input.float(), profile_dir, opt.profile_iters)




if __name__ == '__main__':
    main()
