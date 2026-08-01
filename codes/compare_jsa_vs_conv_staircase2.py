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

"""Side-by-side comparison of JSA (attention) vs JSA+Conv on one classroom test view.

Produces: GT / noisy-input / JSA-output / JSA+Conv-output panel image with
PSNR/SSIM/timing captions, plus a markdown metrics table on stdout.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import dataset as dataset_mod
import eval as eval_mod
import utils.utils_image as util_image
import utils.utils_options as option
import utils.utils_rend_img as util_rend
from config_staircase2 import config as config_jsa
from config_cnn_staircase2 import config as config_conv
import model.model_joint_sa as model_jsa_mod
import model.jsa_4layer_swinir_conv_decoder as model_conv_mod


def load_test_view(config, view_index):
    test_dir = config["testDatasetDirectory"]
    input_dir = os.path.join(test_dir, str(config["test_input"]) + "_npz")
    target_dir = os.path.join(test_dir, str(config["test_target"]) + "_npz")

    input_files = dataset_mod.load_image_name(input_dir, ".npz")
    target_files = dataset_mod.load_image_name(target_dir, ".npz")

    input_npz = np.load(input_files[view_index])
    target_npz = np.load(target_files[view_index])
    name = os.path.splitext(os.path.basename(input_files[view_index]))[0]
    return input_npz, target_npz, name


def build_model(kind, config, device, epoch="best"):
    common_kwargs = dict(
        img_size=config["patch_size"],
        embedded_dim=config["embed_dim"],
        win_size=8,
        projection_option="linear",
        ffn_option="mlp",
        depths=[1, 2, 4, 8, 2, 8, 4, 2, 4],
        in_x=config["x_dim"],
        in_f=config["f_dim"],
    )
    if kind == "jsa":
        net = model_jsa_mod.JSA_transformer(**common_kwargs)
    elif kind == "conv":
        net = model_conv_mod.JSA4LayerSwinIRConvDecoder(**common_kwargs)
    else:
        raise ValueError(kind)
    net = net.to(device)
    net.eval()

    checkpoint_dir = os.path.join(config["data_dir"], config["task"], "__checkpoints__")
    option.load_checkpoint(config["task"], checkpoint_dir, net, epoch)
    return net


def prepare_tensors(input_npz, target_npz, device):
    aux = torch.from_numpy(input_npz["aux"]).unsqueeze(0).float()
    aux[:, :, :, 3:6] = torch.FloatTensor(util_rend.preprocess_normal(aux[:, :, :, 3:6]))
    aux = aux.permute(0, 3, 1, 2)

    color_noisy_raw = torch.from_numpy(input_npz["color"]).unsqueeze(0).float()
    color_noisy_log = util_rend.preprocess_specular(color_noisy_raw)
    color_noisy_log = color_noisy_log.permute(0, 3, 1, 2)

    color_gt_raw = torch.from_numpy(target_npz["color"]).unsqueeze(0).float()
    color_gt_raw = color_gt_raw.permute(0, 3, 1, 2)

    h, w = color_noisy_log.shape[2], color_noisy_log.shape[3]
    factor = 128
    hh, ww = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    xx = max(hh, ww)
    padh, padw = xx - h, xx - w

    x = F.pad(color_noisy_log, (0, padw, 0, padh), "reflect").to(device)
    y = F.pad(aux, (0, padw, 0, padh), "reflect").to(device)

    return x, y, color_noisy_log, color_gt_raw, h, w


def run_tiled_inference(net, x, y, h, w, tile_size, tile_overlap, device):
    # one warmup pass so cudnn autotune/first-call overhead doesn't skew the timed run
    with torch.no_grad():
        eval_mod.tiled_forward(net, x, y, tile_size, device, overlap=tile_overlap)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start_event.record()
    with torch.no_grad():
        out = eval_mod.tiled_forward(net, x, y, tile_size, device, overlap=tile_overlap)
    end_event.record()
    end_event.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    out = out[:, :, :h, :w]
    return out, elapsed_ms


def to_255(tensor_bchw, post_spec):
    # tensor2img transposes CHW -> HWC internally, so this already returns an HWC uint8 image
    return util_rend.tensor2img(tensor_bchw.detach().cpu().numpy()[0], post_spec=post_spec)


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_panel_figure(panels, out_path):
    # panels: list of (title, caption_lines, HWC uint8 image)
    img_h, img_w = panels[0][2].shape[:2]
    title_h, caption_h, gap, margin = 36, 54, 12, 16
    cell_w = img_w
    cell_h = title_h + img_h + caption_h
    canvas_w = margin * 2 + cell_w * len(panels) + gap * (len(panels) - 1)
    canvas_h = margin * 2 + cell_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    caption_font = load_font(15)

    x_cursor = margin
    for title, caption_lines, img in panels:
        draw.text((x_cursor, margin), title, fill=(255, 255, 255), font=title_font)
        pil_img = Image.fromarray(img)
        canvas.paste(pil_img, (x_cursor, margin + title_h))
        cap_y = margin + title_h + img_h + 6
        for line in caption_lines:
            draw.text((x_cursor, cap_y), line, fill=(220, 220, 220), font=caption_font)
            cap_y += 18
        x_cursor += cell_w + gap

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "inference_staircase2"),
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    out_dir = os.path.abspath(args.out_dir)
    util_image.mkdir(out_dir)

    input_npz, target_npz, view_name = load_test_view(config_jsa, args.view_index)
    print("Test view: {}".format(view_name))

    net_jsa = build_model("jsa", config_jsa, device)
    net_conv = build_model("conv", config_conv, device)

    x, y, color_noisy_log, color_gt_raw, h, w = prepare_tensors(input_npz, target_npz, device)

    tile_size_jsa = config_jsa.get("eval_tile_size", config_jsa.get("patch_size", 128))
    overlap_jsa = config_jsa.get("eval_tile_overlap", 0)
    tile_size_conv = config_conv.get("eval_tile_size", config_conv.get("patch_size", 128))
    overlap_conv = config_conv.get("eval_tile_overlap", 0)

    out_jsa, time_jsa_ms = run_tiled_inference(net_jsa, x, y, h, w, tile_size_jsa, overlap_jsa, device)
    out_conv, time_conv_ms = run_tiled_inference(net_conv, x, y, h, w, tile_size_conv, overlap_conv, device)

    gt_255 = to_255(color_gt_raw, post_spec=False)
    noisy_255 = to_255(color_noisy_log, post_spec=True)
    jsa_255 = to_255(out_jsa, post_spec=True)
    conv_255 = to_255(out_conv, post_spec=True)

    gt_hwc, noisy_hwc, jsa_hwc, conv_hwc = gt_255, noisy_255, jsa_255, conv_255

    psnr_noisy = util_image.calculate_psnr(noisy_255, gt_255)
    psnr_jsa = util_image.calculate_psnr(jsa_255, gt_255)
    psnr_conv = util_image.calculate_psnr(conv_255, gt_255)

    ssim_noisy = util_image.calculate_ssim(noisy_255, gt_255)
    ssim_jsa = util_image.calculate_ssim(jsa_255, gt_255)
    ssim_conv = util_image.calculate_ssim(conv_255, gt_255)

    panels = [
        ("GT (clean)", ["reference"], gt_hwc),
        ("Noisy input", ["PSNR: {:.2f} dB".format(psnr_noisy), "SSIM: {:.4f}".format(ssim_noisy)], noisy_hwc),
        (
            "JSA (attention)",
            ["PSNR: {:.2f} dB".format(psnr_jsa), "SSIM: {:.4f}".format(ssim_jsa), "Infer: {:.2f} ms".format(time_jsa_ms)],
            jsa_hwc,
        ),
        (
            "JSA+Conv",
            ["PSNR: {:.2f} dB".format(psnr_conv), "SSIM: {:.4f}".format(ssim_conv), "Infer: {:.2f} ms".format(time_conv_ms)],
            conv_hwc,
        ),
    ]

    fig_path = os.path.join(out_dir, "{}_compare.png".format(view_name))
    build_panel_figure(panels, fig_path)

    for name, img in [("gt", gt_hwc), ("noisy", noisy_hwc), ("jsa", jsa_hwc), ("conv", conv_hwc)]:
        util_image.imwrite(img, os.path.join(out_dir, "{}_{}.png".format(view_name, name)))

    print()
    print("| Metric | Noisy input | JSA (attention) | JSA+Conv |")
    print("|---|---|---|---|")
    print("| PSNR (dB) | {:.2f} | {:.2f} | {:.2f} |".format(psnr_noisy, psnr_jsa, psnr_conv))
    print("| SSIM | {:.4f} | {:.4f} | {:.4f} |".format(ssim_noisy, ssim_jsa, ssim_conv))
    print("| Inference time (ms) | - | {:.2f} | {:.2f} |".format(time_jsa_ms, time_conv_ms))
    print()
    print("Saved comparison figure to: {}".format(fig_path))


if __name__ == "__main__":
    main()
