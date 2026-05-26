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

# Some functions of this script came from the repository of Uformer (https://github.com/ZhendongWang6/Uformer).

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange, repeat

import time

class SpaceToDepth(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = int(r)
    def forward(self, x):
        B, C, H, W = x.shape
        r = self.r
        assert (H % r == 0) and (W % r == 0)
        x = x.view(B, C, H//r, r, W//r, r)          # [B,C,H/r,r,W/r,r]
        x = x.permute(0,1,3,5,2,4).contiguous()     # [B,C,r,r,H/r,W/r]
        x = x.view(B, C*r*r, H//r, W//r)            # [B,C*r*r,H/r,W/r]
        return x

class DepthToSpace(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = int(r)
    def forward(self, x):
        B, C, H, W = x.shape
        r = self.r
        assert C % (r*r) == 0
        x = x.view(B, C//(r*r), r, r, H, W)         # [B,C/(r^2),r,r,H,W]
        x = x.permute(0,1,4,2,5,3).contiguous()     # [B,C/(r^2),H,r,W,r]
        x = x.view(B, C//(r*r), H*r, W*r)           # [B,C/(r^2),H*r,W*r]
        return x

def replace_pixel_ops(module: nn.Module):
    """모델 내부의 Pixel(Un)Shuffle을 TRT 친화 op로 치환 (무가중치 → 재학습 불필요)"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.PixelUnshuffle):
            setattr(module, name, SpaceToDepth(child.downscale_factor))
        elif isinstance(child, nn.PixelShuffle):
            setattr(module, name, DepthToSpace(child.upscale_factor))
        else:
            replace_pixel_ops(child)

# window operation
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().reshape(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.reshape(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.reshape(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)
    return x

# def window_partition(x, win_size, dilation_rate=1):
#     assert dilation_rate == 1, "TensorRT/Export does not support dilation window partition."
#     B, H, W, C = x.shape
#     x = x.reshape(B, H // win_size, win_size, W // win_size, win_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, win_size, win_size, C)
#     return windows

# def window_reverse(windows, win_size, H, W, dilation_rate=1):
#     assert dilation_rate == 1, "TensorRT/Export does not support dilation window reverse."
#     B = windows.shape[0] // ((H * W) // (win_size * win_size))
#     x = windows.reshape(B, H // win_size, W // win_size, win_size, win_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
#     return x

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=32, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

class InputProj_temporal(nn.Module):
    def __init__(self, in_channel=6, out_channel=32, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        # x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        x = self.proj(x)
        B, C_new, H_new, W_new = x.shape
        x = x.reshape(B, C_new, H_new * W_new).permute(0, 2, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x
    
# Input Projection for auxiliary buffer
class InputProj_aux(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, padding=1, norm_layer=nn.LayerNorm,act_layer=nn.LeakyReLU):
        super().__init__()

        pad = (kernel_size - stride)//2
        
        self.proj_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=pad,groups=out_channel),
            act_layer(inplace=True)                  
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0, stride=stride),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=pad,groups=out_channel),
            act_layer(inplace=True)
        )
        self.proj_3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0, stride=stride),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=pad,groups=out_channel),
            act_layer(inplace=True)
        )
        self.proj_fin = nn.Sequential(
            nn.Conv2d(3 * out_channel, out_channel, kernel_size=1, padding=0, stride=stride),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm1 = norm_layer(out_channel)
            self.norm2 = norm_layer(out_channel)
            self.norm3 = norm_layer(out_channel)
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None

        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        # B, C, H, W = x.shape

        x1 = self.proj_1(x).flatten(2).transpose(1, 2).contiguous()  # B C H*W => B H*W C
        if self.norm1 is not None:
            x1 = self.norm1(x1)            
        B_x, L_x, C_x = x1.shape
        H_x = int(math.sqrt(L_x))
        W_x = int(math.sqrt(L_x))
        x1 = x1.transpose(1, 2).reshape(B_x, C_x, H_x, W_x)

        x2 = self.proj_2(x1).flatten(2).transpose(1, 2).contiguous()  # B C H*W => B H*W C
        if self.norm2 is not None:
            x2 = self.norm2(x2)
        x2 = x2.transpose(1, 2).reshape(B_x, C_x, H_x, W_x)

        x3 = self.proj_3(x2).flatten(2).transpose(1, 2).contiguous()  # B C H*W => B H*W C
        if self.norm3 is not None:
            x3 = self.norm3(x3)
        x3 = x3.transpose(1, 2).reshape(B_x, C_x, H_x, W_x)
        
        x_fin = torch.cat([x1,x2,x3], dim = 1)
        x = self.proj_fin(x_fin).flatten(2).transpose(1, 2).contiguous()  # B C H*W => B H*W C

        return x
    
    # def forward(self, x):
    #     # x: B, C, H, W
    #     B, _, H, W = x.shape

    #     # proj_1
    #     x1 = self.proj_1(x)  # B, C1, H, W
    #     if self.norm1 is not None:
    #         x1 = self.norm1(x1)
    #     C1 = x1.shape[1]
    #     x1_reshaped = x1.reshape(B, C1, H * W).transpose(1, 2).contiguous()  # B, H*W, C1

    #     # proj_2
    #     x2 = self.proj_2(x1)  # B, C2, H, W
    #     if self.norm2 is not None:
    #         x2 = self.norm2(x2)
    #     C2 = x2.shape[1]
    #     x2_reshaped = x2.reshape(B, C2, H * W).transpose(1, 2).contiguous()

    #     # proj_3
    #     x3 = self.proj_3(x2)  # B, C3, H, W
    #     if self.norm3 is not None:
    #         x3 = self.norm3(x3)
    #     C3 = x3.shape[1]
    #     x3_reshaped = x3.reshape(B, C3, H * W).transpose(1, 2).contiguous()

    #     # concat along channel axis
    #     x_cat = torch.cat([x1_reshaped.transpose(1, 2), x2_reshaped.transpose(1, 2), x3_reshaped.transpose(1, 2)], dim=1)  # B, C_total, H*W
    #     x_cat = x_cat.reshape(B, -1, H, W)

    #     # final projection
    #     x = self.proj_fin(x_cat)  # B, Cout, H, W
    #     Cout = x.shape[1]
    #     x = x.reshape(B, Cout, H * W).transpose(1, 2).contiguous()  # B, H*W, Cout

    #     return x

      
# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x    
    
# Embedding for q,k,v 
class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v    
    
class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x):
        B_, N, C = x.shape
        attn_kv = x            
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

# feed-forward network
class Simple_mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.ReLU):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,groups=hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):

        x = self.fc1(x)

        B_x, L_x, C_x = x.shape
        H_x = int(math.sqrt(L_x))
        W_x = int(math.sqrt(L_x))
        x = x.transpose(1, 2).reshape(B_x, C_x, H_x, W_x)

        x = self.conv(x)        
        x = self.act(x)

        x = x.flatten(2).transpose(1, 2).contiguous()   

        x = self.fc2(x)
        
        return x

# Resizing modules
class Downsample_shuffle(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_shuffle, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
        return self.body(x).flatten(2).transpose(1,2).contiguous()  # B H*W C

class Upsample_shuffle(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_shuffle, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
        return self.body(x).flatten(2).transpose(1,2).contiguous()  # B H*W C

# window-based joint self-attention
class WindowJointAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, 
                 token_projection='linear', 
                 qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # 1/sqrt(d_k)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww # similar to unsqueeze 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1 # (2*M - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02) 
        
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)    # dconv and conv1x1 
            self.qkv_f = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)    # dconv and conv1x1 
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
            self.qkv_f = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1) # softmax to last dim


    def forward(self, x, f):
        B_, N, C = x.shape
        q, k, v = self.qkv(x) # make qkv # B_, num_heads, N, C // num_heads
        q = q * self.scale  # q * 1/sqrt(d_k)    # B_, num_heads, N, C // num_heads
        attn = (q @ k.transpose(-2, -1)) # @: multiplication matrices # last two dimensions
        # q: B_, num_heads, N, C // num_heads |  k: B_, num_heads, C // num_heads, N
        # attn: B_, num_heads, N, N
        
        # for f
        B_f, N_f, C_f = f.shape
        q_f, k_f, v_f = self.qkv_f(f) # make qkv # B_, num_heads, N, C // num_heads
        q_f = q_f * self.scale  # q * 1/sqrt(d_k)    # B_, num_heads, N, C // num_heads
        attn_f = (q_f @ k_f.transpose(-2, -1)) # @: multiplication matrices # last two dimensions
        # q: B_, num_heads, N, C // num_heads |  k: B_, num_heads, C // num_heads, N
        # attn: B_, num_heads, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww, Wh*Ww, nH(num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH(num_heads), Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1) # N / (Wh*Ww)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
        
        # map + B
        attn = attn + relative_position_bias.unsqueeze(0) # 1, nH(num_heads), l, c*d
        attn_f = attn_f + relative_position_bias.unsqueeze(0) # 1, nH(num_heads), l, c*d
        
        # elementwise product
        attn = attn * attn_f 
        # softmax                            
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) # softmax(map)*v # v: B_, num_heads, N, C // num_heads
        x = self.proj(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

# Joint Self-Attention Transformer Block
class JSATransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 token_projection='linear',token_ffn='mlp'
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_ffn
        self.act_layer = act_layer
         
        self.norm1 = norm_layer(dim)
        self.norm1_f = norm_layer(dim)

        self.attn = WindowJointAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, token_projection=token_projection)  
        
        self.norm2 = norm_layer(dim)
        self.norm2_f = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # FFN check
        if token_ffn in ['ffn','mlp']:
            self.mlp = Simple_mlp(dim=dim, hidden_dim=mlp_hidden_dim,act_layer=self.act_layer)          
        else:
            raise Exception("FFN error!") 
    
    # tensor adding
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, f): # f is feature (G-buffer)
        B, L, C = x.shape
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        H,W = self.input_resolution

        # for g
        B_f, L_f, C_f = f.shape
        # H_f = int(math.sqrt(L_f))
        # W_f = int(math.sqrt(L_f))
        H_f,W_f = self.input_resolution
        
        shortcut = x # X_(l-1)
        x = self.norm1(x) # LN(x)
        x = x.reshape(B, H, W, C)
        # for f
        f = self.norm1_f(f) # LN(f)
        f = f.reshape(B_f, H_f, W_f, C_f)

        # partition windows
        x_windows = window_partition(x, self.win_size)  # nW*B, win_size, win_size, C  # N*C->C
        x_windows = x_windows.reshape(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        # for f
        f_windows = window_partition(f, self.win_size)  # nW*B, win_size, win_size, C  # N*C->C
        f_windows = f_windows.reshape(-1, self.win_size * self.win_size, C_f)  # nW*B, win_size*win_size, C

        # W-JSA
        wm_jsa_in = x_windows
        wm_jsa_in_f = f_windows
        attn_windows = self.attn(wm_jsa_in, wm_jsa_in_f)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.reshape(-1, self.win_size, self.win_size, C)
        x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        x = x.reshape(B, H * W, C)
        # for f 
        f = f.reshape(B_f, H_f * W_f, C_f)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))          

        return x

# Basic layer for Joint Self-Attention
class BasicJSAtransLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 norm_layer=nn.LayerNorm,
                 token_projection='linear',token_ffn='mlp'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            JSATransformerBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, win_size=win_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                norm_layer=norm_layer,token_projection=token_projection,token_ffn=token_ffn)
        for i in range(depth)])        

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"    

    def forward(self, x, f):
        for blk in self.blocks:
            x = blk(x, f)
        return x
    
# Joint Self-Attention Transformer (U-shaped) 
class JSAT_transformer(nn.Module):
    def __init__(self, in_x=6, in_f=7, img_size=128, out_channel=3, embedded_dim=32,
                  depths=[1, 2, 4, 8, 2, 8, 4, 2, 4], num_heads=[1, 2, 4, 8, 16, 8, 4, 2, 1], 
                 win_size=8, mlp_ratio=4., 
                 qkv_bias=True, patch_norm=True,
                 norm_layer=nn.LayerNorm, dowsample=Downsample_shuffle, upsample=Upsample_shuffle,
                 projection_option='linear', ffn_option='mlp'):
        super().__init__()

        # init
        self.in_x = in_x
        self.in_f = in_f
        self.win_size = win_size
        self.resolution = img_size

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embedded_dim = embedded_dim
        self.patch_norm = patch_norm
        
        self.mlp = ffn_option
        self.mlp_ratio = mlp_ratio
        self.projection = projection_option
        
        # build layers

        # input/output projection
        # self.input_proj = InputProj(in_channel=in_x, out_channel=embedded_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.input_proj = InputProj_temporal(in_channel=in_x, out_channel=embedded_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        # self.output_proj = OutputProj(in_channel=2*embedded_dim, out_channel=out_channel, kernel_size=3, stride=1)
        self.output_proj = OutputProj(in_channel=2*embedded_dim, out_channel=(out_channel + 1), kernel_size=3, stride=1)
        # auxiliary buffer projection
        self.input_proj_f = InputProj_aux(in_channel=in_f, out_channel=embedded_dim, kernel_size=3, stride=1, norm_layer=None ,act_layer=nn.LeakyReLU)

        # Encoder
        self.encoderlayer_l0 = BasicJSAtransLayer(dim=embedded_dim,
                            output_dim=embedded_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        self.dowsample_l0_1 = dowsample(embedded_dim)

        self.encoderlayer_l1 = BasicJSAtransLayer(dim=embedded_dim*2,
                            output_dim=embedded_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        self.dowsample_l1_2 = dowsample(embedded_dim*2)

        self.encoderlayer_l2 = BasicJSAtransLayer(dim=embedded_dim*4,
                            output_dim=embedded_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        self.dowsample_l2_3 = dowsample(embedded_dim*4)

        self.encoderlayer_l3 = BasicJSAtransLayer(dim=embedded_dim*8,
                            output_dim=embedded_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        self.dowsample_l3_4 = dowsample(embedded_dim*8)

        # Bottleneck
        self.conv = BasicJSAtransLayer(dim=embedded_dim*16,
                            output_dim=embedded_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)

        # Decoder
        self.upsample_l4_3 = upsample(embedded_dim*16)
        self.linear_l3 = nn.Linear(embedded_dim*16, embedded_dim*8)        
        self.decoderlayer_l3 = BasicJSAtransLayer(dim=embedded_dim*8,
                            output_dim=embedded_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        
        self.upsample_l3_2 = upsample(embedded_dim*8)
        self.linear_l2 = nn.Linear(embedded_dim*8, embedded_dim*4)
        self.decoderlayer_l2 = BasicJSAtransLayer(dim=embedded_dim*4,
                            output_dim=embedded_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        
        self.upsample_l2_1 = upsample(embedded_dim*4)
        self.linear_l1 = nn.Linear(embedded_dim*4, embedded_dim*2)
        self.decoderlayer_l1 = BasicJSAtransLayer(dim=embedded_dim*2,
                            output_dim=embedded_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
         
        self.upsample_l1_0 = upsample(embedded_dim*2)    
        self.decoderlayer_l0 = BasicJSAtransLayer(dim=embedded_dim*2,
                            output_dim=embedded_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embedded_dim={self.embedded_dim}, projection_option={self.projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, input):

        x, f = input[:, :self.in_x], input[:, self.in_x:self.in_x+self.in_f]

        # Input Projection
        F_x_l0 = self.input_proj(x)
        # for f
        F_f_l0 = self.input_proj_f(f)

        #Encoder
        # level 0
        inp_enc_l0 = F_x_l0
        out_enc_l0 = self.encoderlayer_l0(inp_enc_l0, F_f_l0)
        # level 1
        inp_enc_l1 = self.dowsample_l0_1(out_enc_l0)
        F_f_l1 = self.dowsample_l0_1(F_f_l0)
        out_enc_l1 = self.encoderlayer_l1(inp_enc_l1, F_f_l1)
        # level 2
        inp_enc_l2 = self.dowsample_l1_2(out_enc_l1)
        F_f_l2 = self.dowsample_l1_2(F_f_l1)
        out_enc_l2 = self.encoderlayer_l2(inp_enc_l2, F_f_l2)
        # level 3
        inp_enc_l3 = self.dowsample_l2_3(out_enc_l2)
        F_f_l3 = self.dowsample_l2_3(F_f_l2)
        out_enc_l3 = self.encoderlayer_l3(inp_enc_l3, F_f_l3)
        # Bottle neck
        inp_enc_l4 = self.dowsample_l3_4(out_enc_l3)
        F_f_l4 = self.dowsample_l3_4(F_f_l3)
        out_enc_l4 = self.conv(inp_enc_l4, F_f_l4)   

        # Decoder
        # level 3
        inp_dec_l3_pre = self.upsample_l4_3(out_enc_l4)
        inp_dec_l3_cat = torch.cat([inp_dec_l3_pre, out_enc_l3],dim=-1)
        inp_dec_l3 = self.linear_l3(inp_dec_l3_cat)
        out_dec_l3 = self.decoderlayer_l3(inp_dec_l3, F_f_l3)
        # level 2
        inp_dec_l2_pre = self.upsample_l3_2(out_dec_l3)
        inp_dec_l2_cat = torch.cat([inp_dec_l2_pre, out_enc_l2],dim=-1)
        inp_dec_l2 = self.linear_l2(inp_dec_l2_cat)
        out_dec_l2 = self.decoderlayer_l2(inp_dec_l2, F_f_l2)
        # level 1
        inp_dec_l1_pre = self.upsample_l2_1(out_dec_l2)
        inp_dec_l1_cat = torch.cat([inp_dec_l1_pre, out_enc_l1],dim=-1)
        inp_dec_l1 = self.linear_l1(inp_dec_l1_cat)
        out_dec_l1 = self.decoderlayer_l1(inp_dec_l1, F_f_l1)
        # level 0
        inp_dec_l0_pre = self.upsample_l1_0(out_dec_l1)
        F_f_l0_pre = self.upsample_l1_0(F_f_l1)
        inp_dec_l0_cat = torch.cat([inp_dec_l0_pre, out_enc_l0],dim=-1)
        inp_dec_l0_f = torch.cat([F_f_l0_pre, F_f_l0], dim = -1)
        inp_dec_l0 = inp_dec_l0_cat
        out_dec_l0 = self.decoderlayer_l0(inp_dec_l0, inp_dec_l0_f)

        # Output Projection
        out = self.output_proj(out_dec_l0)
        res = x[:, :3, :, :].clone()
        out_color = res + out[:,:3,:,:]
        alpha  = torch.sigmoid(out[:, 3, :, :])

        # return out_color, alpha
        alpha = alpha.unsqueeze(1)
        return torch.cat([out_color, alpha], dim=1)


# ==== ADD: token-linear -> conv1x1 wrapper ====
class TokenLinear1x1(nn.Module):
    """
    nn.Linear(B, L, Cin)->(B, L, Cout)를 Conv1x1로 동치 치환.
    가중치/바이어스를 기존 Linear에서 복사. (재학습 불필요)
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        Cin = linear.in_features
        Cout = linear.out_features
        self.conv = nn.Conv2d(Cin, Cout, kernel_size=1, bias=(linear.bias is not None))
        with torch.no_grad():
            self.conv.weight.copy_(linear.weight.view(Cout, Cin, 1, 1))
            if linear.bias is not None:
                self.conv.bias.copy_(linear.bias)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        # x_tokens: [B, L, C]
        B, L, C = x_tokens.shape
        H = W = int(math.sqrt(L))
        assert H * W == L, f"L={L} is not a square"
        x = x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)   # [B,C,H,W]
        x = self.conv(x)                                             # [B,Cout,H,W]
        x = x.flatten(2).transpose(1, 2).contiguous()                # [B,L,Cout]
        return x

def patch_model_for_int8(net: nn.Module):
    """
    JSAT 모델의 토큰-선형 경로를 Conv1x1로 치환.
    호출 시점: checkpoint 로딩 직후, TRT 컴파일 전에 1회.
    """
    # 1) Simple_mlp: fc1/fc2 -> TokenLinear1x1
    for m in net.modules():
        if isinstance(m, Simple_mlp):
            if isinstance(m.fc1, nn.Linear):
                m.fc1 = TokenLinear1x1(m.fc1)
            if isinstance(m.fc2, nn.Linear):
                m.fc2 = TokenLinear1x1(m.fc2)

    # 2) WindowJointAttention: proj -> TokenLinear1x1
    for m in net.modules():
        if isinstance(m, WindowJointAttention):
            if isinstance(m.proj, nn.Linear):
                m.proj = TokenLinear1x1(m.proj)

    # 3) LinearProjection: to_q/to_kv -> TokenLinear1x1
    #    주: 출력 reshape는 기존과 동일하게 heads로 쪼갭니다.
    for m in net.modules():
        if isinstance(m, LinearProjection):
            # wrap originals
            m._to_q_orig = m.to_q
            m._to_kv_orig = m.to_kv
            m.to_q = TokenLinear1x1(m._to_q_orig)
            m.to_kv = TokenLinear1x1(m._to_kv_orig)

            # 새 forward로 교체 (토큰->conv1x1->heads split)
            def _new_forward(self, x):
                B_, N, C = x.shape
                q_all = self.to_q(x)        # [B_, N, inner_dim]
                kv_all = self.to_kv(x)      # [B_, N, 2*inner_dim]
                H = self.heads
                Dh = q_all.shape[-1] // H
                q = q_all.view(B_, N, H, Dh).permute(0, 2, 1, 3)      # [B_,H,N,Dh]
                kv = kv_all.view(B_, N, 2, H, Dh).permute(2, 0, 3, 1, 4)
                k, v = kv[0], kv[1]                                   # [B_,H,N,Dh] x2
                return q, k, v
            m.forward = _new_forward.__get__(m, LinearProjection)

    # 4) Decoder stage linears (linear_l3/l2/l1) -> TokenLinear1x1
    #    토큰 수 L은 해당 stage의 (H*W)로 이미 완전 제곱. wrapper가 자동 sqrt 사용.
    for name in ["linear_l3", "linear_l2", "linear_l1"]:
        if hasattr(net, name):
            mod = getattr(net, name)
            if isinstance(mod, nn.Linear):
                setattr(net, name, TokenLinear1x1(mod))

    # Pixel(Un)Shuffle 교체(이미 있으나, 혹시 안 하셨다면)
    replace_pixel_ops(net)

    return net



# Joint Self-Attention Transformer (U-shaped) without temporal
class JSA_transformer(nn.Module):
    def __init__(self, in_x=3, in_f=7, img_size=128, out_channel=3, embedded_dim=32,
                  depths=[1, 2, 4, 8, 2, 8, 4, 2, 4], num_heads=[1, 2, 4, 8, 16, 8, 4, 2, 1], 
                 win_size=8, mlp_ratio=4., 
                 qkv_bias=True, patch_norm=True,
                 norm_layer=nn.LayerNorm, dowsample=Downsample_shuffle, upsample=Upsample_shuffle,
                 projection_option='linear', ffn_option='mlp'):
        super().__init__()

        # init
        self.in_x = in_x
        self.in_f = in_f
        self.win_size = win_size
        self.resolution = img_size

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embedded_dim = embedded_dim
        self.patch_norm = patch_norm
        
        self.mlp = ffn_option
        self.mlp_ratio = mlp_ratio
        self.projection = projection_option
        
        # build layers

        # input/output projection
        self.input_proj = InputProj(in_channel=in_x, out_channel=embedded_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        # self.input_proj = InputProj_temporal(in_channel=in_x, out_channel=embedded_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embedded_dim, out_channel=out_channel, kernel_size=3, stride=1)
        # self.output_proj = OutputProj(in_channel=2*embedded_dim, out_channel=(out_channel + 1), kernel_size=3, stride=1)
        # auxiliary buffer projection
        self.input_proj_f = InputProj_aux(in_channel=in_f, out_channel=embedded_dim, kernel_size=3, stride=1, norm_layer=None ,act_layer=nn.LeakyReLU)

        # Encoder
        self.encoderlayer_l0 = BasicJSAtransLayer(dim=embedded_dim,
                            output_dim=embedded_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        self.dowsample_l0_1 = dowsample(embedded_dim)

        self.encoderlayer_l1 = BasicJSAtransLayer(dim=embedded_dim*2,
                            output_dim=embedded_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        self.dowsample_l1_2 = dowsample(embedded_dim*2)

        self.encoderlayer_l2 = BasicJSAtransLayer(dim=embedded_dim*4,
                            output_dim=embedded_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        self.dowsample_l2_3 = dowsample(embedded_dim*4)

        self.encoderlayer_l3 = BasicJSAtransLayer(dim=embedded_dim*8,
                            output_dim=embedded_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        self.dowsample_l3_4 = dowsample(embedded_dim*8)

        # Bottleneck
        self.conv = BasicJSAtransLayer(dim=embedded_dim*16,
                            output_dim=embedded_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)

        # Decoder
        self.upsample_l4_3 = upsample(embedded_dim*16)
        self.linear_l3 = nn.Linear(embedded_dim*16, embedded_dim*8)        
        self.decoderlayer_l3 = BasicJSAtransLayer(dim=embedded_dim*8,
                            output_dim=embedded_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        
        self.upsample_l3_2 = upsample(embedded_dim*8)
        self.linear_l2 = nn.Linear(embedded_dim*8, embedded_dim*4)
        self.decoderlayer_l2 = BasicJSAtransLayer(dim=embedded_dim*4,
                            output_dim=embedded_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
        
        self.upsample_l2_1 = upsample(embedded_dim*4)
        self.linear_l1 = nn.Linear(embedded_dim*4, embedded_dim*2)
        self.decoderlayer_l1 = BasicJSAtransLayer(dim=embedded_dim*2,
                            output_dim=embedded_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)
         
        self.upsample_l1_0 = upsample(embedded_dim*2)    
        self.decoderlayer_l0 = BasicJSAtransLayer(dim=embedded_dim*2,
                            output_dim=embedded_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            norm_layer=norm_layer,
                            token_projection=projection_option,token_ffn=ffn_option)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embedded_dim={self.embedded_dim}, projection_option={self.projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, input):

        x, f = input[:, :self.in_x], input[:, self.in_x:self.in_x+self.in_f]

        # Input Projection
        F_x_l0 = self.input_proj(x)
        # for f
        F_f_l0 = self.input_proj_f(f)

        #Encoder
        # level 0
        inp_enc_l0 = F_x_l0
        out_enc_l0 = self.encoderlayer_l0(inp_enc_l0, F_f_l0)
        # level 1
        inp_enc_l1 = self.dowsample_l0_1(out_enc_l0)
        F_f_l1 = self.dowsample_l0_1(F_f_l0)
        out_enc_l1 = self.encoderlayer_l1(inp_enc_l1, F_f_l1)
        # level 2
        inp_enc_l2 = self.dowsample_l1_2(out_enc_l1)
        F_f_l2 = self.dowsample_l1_2(F_f_l1)
        out_enc_l2 = self.encoderlayer_l2(inp_enc_l2, F_f_l2)
        # level 3
        inp_enc_l3 = self.dowsample_l2_3(out_enc_l2)
        F_f_l3 = self.dowsample_l2_3(F_f_l2)
        out_enc_l3 = self.encoderlayer_l3(inp_enc_l3, F_f_l3)
        # Bottle neck
        inp_enc_l4 = self.dowsample_l3_4(out_enc_l3)
        F_f_l4 = self.dowsample_l3_4(F_f_l3)
        out_enc_l4 = self.conv(inp_enc_l4, F_f_l4)   

        # Decoder
        # level 3
        inp_dec_l3_pre = self.upsample_l4_3(out_enc_l4)
        inp_dec_l3_cat = torch.cat([inp_dec_l3_pre, out_enc_l3],dim=-1)
        inp_dec_l3 = self.linear_l3(inp_dec_l3_cat)
        out_dec_l3 = self.decoderlayer_l3(inp_dec_l3, F_f_l3)
        # level 2
        inp_dec_l2_pre = self.upsample_l3_2(out_dec_l3)
        inp_dec_l2_cat = torch.cat([inp_dec_l2_pre, out_enc_l2],dim=-1)
        inp_dec_l2 = self.linear_l2(inp_dec_l2_cat)
        out_dec_l2 = self.decoderlayer_l2(inp_dec_l2, F_f_l2)
        # level 1
        inp_dec_l1_pre = self.upsample_l2_1(out_dec_l2)
        inp_dec_l1_cat = torch.cat([inp_dec_l1_pre, out_enc_l1],dim=-1)
        inp_dec_l1 = self.linear_l1(inp_dec_l1_cat)
        out_dec_l1 = self.decoderlayer_l1(inp_dec_l1, F_f_l1)
        # level 0
        inp_dec_l0_pre = self.upsample_l1_0(out_dec_l1)
        F_f_l0_pre = self.upsample_l1_0(F_f_l1)
        inp_dec_l0_cat = torch.cat([inp_dec_l0_pre, out_enc_l0],dim=-1)
        inp_dec_l0_f = torch.cat([F_f_l0_pre, F_f_l0], dim = -1)
        inp_dec_l0 = inp_dec_l0_cat
        out_dec_l0 = self.decoderlayer_l0(inp_dec_l0, inp_dec_l0_f)

        # Output Projection
        out = self.output_proj(out_dec_l0)
        # res = x[:, :3, :, :].clone()
        # out_color = res + out[:,:3,:,:]
        # alpha  = torch.sigmoid(out[:, 3, :, :])

        # # return out_color, alpha
        # alpha = alpha.unsqueeze(1)
        return x + out