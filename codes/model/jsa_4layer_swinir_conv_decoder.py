#  Copyright (c) 2024 CGLab, GIST. All rights reserved.
#
# Standalone 4-layer JSA variant with a SwinIR-style convolutional final decoder.

import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

try:
    from .model_joint_sa import (
        BasicJSAtransLayer,
        Downsample_shuffle,
        InputProj,
        InputProj_aux,
        OutputProj,
        Upsample_shuffle,
    )
except ImportError:
    from model_joint_sa import (
        BasicJSAtransLayer,
        Downsample_shuffle,
        InputProj,
        InputProj_aux,
        OutputProj,
        Upsample_shuffle,
    )


def _tokens_to_image(x):
    b, l, c = x.shape
    h = int(math.sqrt(l))
    w = h
    if h * w != l:
        raise ValueError(f"Token length must be square, got L={l}.")
    return x.transpose(1, 2).contiguous().view(b, c, h, w), (h, w)


def _image_to_tokens(x):
    return x.flatten(2).transpose(1, 2).contiguous()


def _make_swinir_conv(dim, resi_connection="3conv", reduction=4):
    if resi_connection == "1conv":
        return nn.Conv2d(dim, dim, 3, 1, 1)
    if resi_connection == "3conv":
        hidden_dim = max(dim // reduction, 1)
        return nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_dim, dim, 3, 1, 1),
        )
    raise ValueError(f"Unsupported resi_connection: {resi_connection}")


class SwinIRResidualConv(nn.Module):
    def __init__(self, dim, resi_connection="3conv", reduction=4):
        super().__init__()
        self.conv = _make_swinir_conv(dim, resi_connection, reduction)

    def forward(self, x):
        return x + self.conv(x)


class ResizeConvUpsample(nn.Module):
    def __init__(self, in_dim, out_dim=None, mode="bilinear"):
        super().__init__()
        out_dim = out_dim or in_dim // 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        if x.shape[2] != self.in_dim:
            raise ValueError(f"x channel dim must be {self.in_dim}, got {x.shape[2]}.")

        x_img, _ = _tokens_to_image(x)
        if self.mode == "bilinear":
            x_img = torch.nn.functional.interpolate(
                x_img,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_img = torch.nn.functional.interpolate(
                x_img,
                scale_factor=2,
                mode=self.mode,
            )
        x_img = self.conv(x_img)
        return _image_to_tokens(x_img)

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, mode={self.mode}"


class GBufferGuidedConvDecoderLayer(nn.Module):
    """Final decoder layer that keeps JSA's G-buffer conditioning without W-JSA."""

    def __init__(
        self,
        dim,
        condition_dim,
        depth,
        norm_layer=nn.LayerNorm,
        resi_connection="3conv",
    ):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.depth = depth

        self.norm_x = norm_layer(dim)
        self.norm_f = norm_layer(condition_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim + condition_dim, dim, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.guide_gate = nn.Sequential(
            nn.Conv2d(dim + condition_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.residual_group = nn.Sequential(
            *[
                SwinIRResidualConv(dim=dim, resi_connection=resi_connection)
                for _ in range(depth)
            ]
        )
        self.conv_after_body = _make_swinir_conv(
            dim=dim, resi_connection=resi_connection
        )

    def forward(self, x, f):
        if x.shape[0] != f.shape[0] or x.shape[1] != f.shape[1]:
            raise ValueError(
                f"x and f must share batch/token dimensions, got {x.shape} and {f.shape}."
            )
        if x.shape[2] != self.dim:
            raise ValueError(f"x channel dim must be {self.dim}, got {x.shape[2]}.")
        if f.shape[2] != self.condition_dim:
            raise ValueError(
                f"f channel dim must be {self.condition_dim}, got {f.shape[2]}."
            )

        shortcut, _ = _tokens_to_image(x)
        x_img, x_size = _tokens_to_image(self.norm_x(x))
        f_img, f_size = _tokens_to_image(self.norm_f(f))
        if x_size != f_size:
            raise ValueError(f"x and f spatial sizes must match, got {x_size} and {f_size}.")

        condition = torch.cat([x_img, f_img], dim=1)
        guided = x_img + self.guide_gate(condition) * self.fuse(condition)
        guided = self.residual_group(guided)
        guided = self.conv_after_body(guided)
        return _image_to_tokens(shortcut + guided)

    def extra_repr(self):
        return (
            f"dim={self.dim}, condition_dim={self.condition_dim}, "
            f"depth={self.depth}"
        )


class JSA4LayerSwinIRConvDecoder(nn.Module):
    def __init__(
        self,
        in_x=3,
        in_f=7,
        img_size=128,
        out_channel=3,
        embedded_dim=16,
        depths=None,
        num_heads=None,
        win_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        dowsample=Downsample_shuffle,
        upsample=Upsample_shuffle,
        projection_option="linear",
        ffn_option="mlp",
        decoder_resi_connection="3conv",
    ):
        super().__init__()
        depths = depths or [1, 2, 4, 8, 2, 8, 4, 2, 4]
        num_heads = num_heads or [1, 2, 4, 8, 16, 8, 4, 2, 1]
        if len(depths) != 9 or len(num_heads) != 9:
            raise ValueError("4Layer configuration requires 9 depths and 9 num_heads.")

        self.in_x = in_x
        self.in_f = in_f
        self.win_size = win_size
        self.resolution = img_size
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embedded_dim = embedded_dim
        self.patch_norm = patch_norm
        self.mlp = ffn_option
        self.mlp_ratio = mlp_ratio
        self.projection = projection_option

        self.input_proj = InputProj(
            in_channel=in_x,
            out_channel=embedded_dim,
            kernel_size=3,
            stride=1,
            act_layer=nn.LeakyReLU,
        )
        self.output_proj = OutputProj(
            in_channel=2 * embedded_dim,
            out_channel=out_channel,
            kernel_size=3,
            stride=1,
        )
        self.input_proj_f = InputProj_aux(
            in_channel=in_f,
            out_channel=embedded_dim,
            kernel_size=3,
            stride=1,
            norm_layer=None,
            act_layer=nn.LeakyReLU,
        )

        self.encoderlayer_l0 = BasicJSAtransLayer(
            dim=embedded_dim,
            output_dim=embedded_dim,
            input_resolution=(img_size, img_size),
            depth=depths[0],
            num_heads=num_heads[0],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            token_projection=projection_option,
            token_ffn=ffn_option,
        )
        self.dowsample_l0_1 = dowsample(embedded_dim)

        self.encoderlayer_l1 = BasicJSAtransLayer(
            dim=embedded_dim * 2,
            output_dim=embedded_dim * 2,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[1],
            num_heads=num_heads[1],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            token_projection=projection_option,
            token_ffn=ffn_option,
        )
        self.dowsample_l1_2 = dowsample(embedded_dim * 2)

        self.encoderlayer_l2 = BasicJSAtransLayer(
            dim=embedded_dim * 4,
            output_dim=embedded_dim * 4,
            input_resolution=(img_size // (2**2), img_size // (2**2)),
            depth=depths[2],
            num_heads=num_heads[2],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            token_projection=projection_option,
            token_ffn=ffn_option,
        )
        self.dowsample_l2_3 = dowsample(embedded_dim * 4)

        self.encoderlayer_l3 = BasicJSAtransLayer(
            dim=embedded_dim * 8,
            output_dim=embedded_dim * 8,
            input_resolution=(img_size // (2**3), img_size // (2**3)),
            depth=depths[3],
            num_heads=num_heads[3],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            token_projection=projection_option,
            token_ffn=ffn_option,
        )
        self.dowsample_l3_4 = dowsample(embedded_dim * 8)

        self.conv = BasicJSAtransLayer(
            dim=embedded_dim * 16,
            output_dim=embedded_dim * 16,
            input_resolution=(img_size // (2**4), img_size // (2**4)),
            depth=depths[4],
            num_heads=num_heads[4],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            token_projection=projection_option,
            token_ffn=ffn_option,
        )

        self.upsample_l4_3 = upsample(embedded_dim * 16)
        self.linear_l3 = nn.Linear(embedded_dim * 16, embedded_dim * 8)
        self.decoderlayer_l3 = BasicJSAtransLayer(
            dim=embedded_dim * 8,
            output_dim=embedded_dim * 8,
            input_resolution=(img_size // (2**3), img_size // (2**3)),
            depth=depths[5],
            num_heads=num_heads[5],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            token_projection=projection_option,
            token_ffn=ffn_option,
        )

        self.upsample_l3_2 = upsample(embedded_dim * 8)
        self.linear_l2 = nn.Linear(embedded_dim * 8, embedded_dim * 4)
        self.decoderlayer_l2 = BasicJSAtransLayer(
            dim=embedded_dim * 4,
            output_dim=embedded_dim * 4,
            input_resolution=(img_size // (2**2), img_size // (2**2)),
            depth=depths[6],
            num_heads=num_heads[6],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            token_projection=projection_option,
            token_ffn=ffn_option,
        )

        self.upsample_l2_1 = upsample(embedded_dim * 4)
        self.linear_l1 = nn.Linear(embedded_dim * 4, embedded_dim * 2)
        self.decoderlayer_l1 = BasicJSAtransLayer(
            dim=embedded_dim * 2,
            output_dim=embedded_dim * 2,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[7],
            num_heads=num_heads[7],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            token_projection=projection_option,
            token_ffn=ffn_option,
        )

        self.upsample_l1_0 = ResizeConvUpsample(
            in_dim=embedded_dim * 2,
            out_dim=embedded_dim,
        )
        self.decoderlayer_l0 = GBufferGuidedConvDecoderLayer(
            dim=embedded_dim * 2,
            condition_dim=embedded_dim * 2,
            depth=depths[8],
            norm_layer=norm_layer,
            resi_connection=decoder_resi_connection,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def extra_repr(self):
        return (
            f"embedded_dim={self.embedded_dim}, projection_option={self.projection}, "
            f"token_mlp={self.mlp}, win_size={self.win_size}"
        )

    def forward(self, x, f):
        f_x_l0 = self.input_proj(x)
        f_f_l0 = self.input_proj_f(f)

        out_enc_l0 = self.encoderlayer_l0(f_x_l0, f_f_l0)

        inp_enc_l1 = self.dowsample_l0_1(out_enc_l0)
        f_f_l1 = self.dowsample_l0_1(f_f_l0)
        out_enc_l1 = self.encoderlayer_l1(inp_enc_l1, f_f_l1)

        inp_enc_l2 = self.dowsample_l1_2(out_enc_l1)
        f_f_l2 = self.dowsample_l1_2(f_f_l1)
        out_enc_l2 = self.encoderlayer_l2(inp_enc_l2, f_f_l2)

        inp_enc_l3 = self.dowsample_l2_3(out_enc_l2)
        f_f_l3 = self.dowsample_l2_3(f_f_l2)
        out_enc_l3 = self.encoderlayer_l3(inp_enc_l3, f_f_l3)

        inp_enc_l4 = self.dowsample_l3_4(out_enc_l3)
        f_f_l4 = self.dowsample_l3_4(f_f_l3)
        out_enc_l4 = self.conv(inp_enc_l4, f_f_l4)

        inp_dec_l3_pre = self.upsample_l4_3(out_enc_l4)
        inp_dec_l3_cat = torch.cat([inp_dec_l3_pre, out_enc_l3], dim=-1)
        inp_dec_l3 = self.linear_l3(inp_dec_l3_cat)
        out_dec_l3 = self.decoderlayer_l3(inp_dec_l3, f_f_l3)

        inp_dec_l2_pre = self.upsample_l3_2(out_dec_l3)
        inp_dec_l2_cat = torch.cat([inp_dec_l2_pre, out_enc_l2], dim=-1)
        inp_dec_l2 = self.linear_l2(inp_dec_l2_cat)
        out_dec_l2 = self.decoderlayer_l2(inp_dec_l2, f_f_l2)

        inp_dec_l1_pre = self.upsample_l2_1(out_dec_l2)
        inp_dec_l1_cat = torch.cat([inp_dec_l1_pre, out_enc_l1], dim=-1)
        inp_dec_l1 = self.linear_l1(inp_dec_l1_cat)
        out_dec_l1 = self.decoderlayer_l1(inp_dec_l1, f_f_l1)

        inp_dec_l0_pre = self.upsample_l1_0(out_dec_l1)
        f_f_l0_pre = self.upsample_l1_0(f_f_l1)
        inp_dec_l0 = torch.cat([inp_dec_l0_pre, out_enc_l0], dim=-1)
        inp_dec_l0_f = torch.cat([f_f_l0_pre, f_f_l0], dim=-1)
        out_dec_l0 = self.decoderlayer_l0(inp_dec_l0, inp_dec_l0_f)

        out = self.output_proj(out_dec_l0)
        return x + out


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    model = JSA4LayerSwinIRConvDecoder()
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    f = torch.randn(1, 7, 128, 128)
    y = model(x, f)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    assert torch.isfinite(y).all(), "Model output contains NaN or Inf."
    print(f"output_shape={tuple(y.shape)}")
    print(f"trainable_params={_count_parameters(model)}")
