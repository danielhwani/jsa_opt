# Model architecture notes

## Part 1: `model_joint_sa.py` (`JSA_transformer`)

`JSA_transformer` is a Uformer-style U-shaped window transformer extended to
take **two** inputs — the noisy color image and a G-buffer (albedo/normal/depth) —
and fuse them inside the attention operator itself. This note explains where
the "Joint" in Joint Self-Attention actually happens, since it's easy to miss
reading the forward pass top to bottom.

### 1. Overall shape — U-Net + 2-branch

```
x (RGB, 3ch)      ──InputProj──────┐                                        ┌──OutputProj──> x + out (residual)
                                     │  encoder (4 levels, dim↑) → bottleneck → decoder (4 levels, dim↓, skip-concat)
f (G-buffer, 7ch) ──InputProj_aux──┘  (same down/upsample modules applied to both x and f)
```

- Encoder, 4 levels (`encoderlayer_l0`..`l3`): channels 32→64→128→256,
  `depths=[1,2,4,8]`. `Downsample_shuffle` (conv + `PixelUnshuffle`) halves
  resolution and doubles channels between levels.
- Bottleneck (`self.conv`): 512 channels, `depths[4]=2`.
- Decoder, 4 levels (`decoderlayer_l3`..`l0`): `Upsample_shuffle`
  (conv + `PixelShuffle`) restores resolution, output is concatenated with
  the matching encoder skip, then `nn.Linear` (`linear_l3`/`l2`/`l1`) halves
  the channel count back down.
- `f` (the G-buffer) is downsampled/upsampled through the **same modules**
  used for `x` at every level (e.g. `F_f_l1 = self.dowsample_l0_1(F_f_l0)`),
  so attention at every scale sees both branches at matching resolution.
- `PixelShuffle`/`PixelUnshuffle` are used instead of strided conv/pooling
  because they trade channels for resolution without discarding information
  (same trick used in Restormer-style restoration nets).

### 2. The core idea: `WindowJointAttention` — multiply the two attention maps

A normal Swin/Uformer window attention block computes
`softmax(QKᵀ/√d + bias) @ V` once. Here, **separate Q/K/V are computed for
the color branch (x) and the G-buffer branch (f)**, each producing its own
raw attention score map, and the two maps are combined by **element-wise
multiplication before softmax**:

```python
attn   = q @ k.transpose(-2, -1) + relative_position_bias      # color branch
attn_f = q_f @ k_f.transpose(-2, -1) + relative_position_bias  # G-buffer branch (same bias table)
attn = attn * attn_f     # <-- this is the "Joint" in Joint Self-Attention
attn = softmax(attn)
out = attn @ v           # v comes only from the color branch; f's v is unused
```

Intuition: wherever the G-buffer branch decides two pixels belong to the same
surface (similar albedo/normal/depth), its attention score gates/reinforces
the color branch's attention at that same pair. `f` only decides *where* to
look; the values actually being aggregated (`v`) still come from the noisy
color branch. This acts like a learned, attention-based analogue of joint/
guided bilateral filtering, rather than a hand-designed kernel.

### 3. FFN is not a plain MLP (`Simple_mlp`)

`Linear → depthwise 3x3 conv → activation → Linear`. Window attention alone
only sees inside one window, so the depthwise conv sandwiched between the
two linear layers injects local spatial context across window boundaries —
the same role LeFF plays in Uformer.

### 4. One asymmetry worth knowing: the top decoder level

Every other decoder level concatenates the upsampled features with the
matching encoder skip, then uses a `nn.Linear` (`linear_l3`/`l2`/`l1`) to
halve the channel count back down before running the transformer blocks.
`decoderlayer_l0` is the exception — there is no `linear_l0`, so it operates
directly on the full concatenated width (`dim=embedded_dim*2`, i.e. 64 for
`embedded_dim=32`). This is why `output_proj`'s `in_channel` is
`2*embedded_dim` rather than `embedded_dim`. The `f` branch is handled the
same way only at this last level: `F_f_l0_pre` (upsampled) is concatenated
with the original `F_f_l0` to match the doubled width expected by
`decoderlayer_l0`.

### 5. Output: residual, not a direct prediction

```python
return x + out
```

The network predicts a delta added to the noisy input rather than the clean
image directly, so training starts close to an identity mapping.

### 6. Depths/heads are symmetric around the bottleneck

`depths=[1,2,4,8,2,8,4,2,4]`, `num_heads=[1,2,4,8,16,8,4,2,1]` — both grow
going into the bottleneck and shrink coming back out, mirroring the U-shape
of the network itself.

## Part 2: `jsa_4layer_swinir_conv_decoder.py` (`JSA4LayerSwinIRConvDecoder`, "JSA+Conv")

This is **not** a separate model written from scratch — it directly imports
`BasicJSAtransLayer`, `Downsample_shuffle`, `Upsample_shuffle`, `InputProj`,
`InputProj_aux`, `OutputProj` from `model_joint_sa.py` and reuses them
unchanged. The encoder (`l0`..`l3`), bottleneck, and decoder levels `l3`/`l2`/`l1`
are **identical** to Part 1 — same `WindowJointAttention`, same
attention-map multiplication, same skip-concat + `nn.Linear` down-projection.

The only thing that changes is the **very last decoder stage** (level 0, full
resolution), which is where the actual compute savings behind "JSA+Conv"
being ~1.6-2x faster than the original JSA comes from — the two most
expensive full-resolution window-attention blocks are replaced by
convolutions:

| | Part 1 (`JSA_transformer`) | Part 2 (`JSA4LayerSwinIRConvDecoder`) |
|---|---|---|
| `upsample_l1_0` | `Upsample_shuffle` (conv + `PixelShuffle`, info-preserving) | `ResizeConvUpsample` (bilinear ×2 + conv — cheaper, not info-preserving) |
| `decoderlayer_l0` | `BasicJSAtransLayer` (window joint self-attention, `depths[8]=4` blocks) | `GBufferGuidedConvDecoderLayer` (pure SwinIR-style residual convs, no attention at all) |

Everything upstream of that (encoder + bottleneck + decoder `l3`/`l2`/`l1`)
still pays for full window-attention, so the speedup is specifically "skip
attention at the most expensive (full-resolution) stage," not "skip
attention everywhere."

### `GBufferGuidedConvDecoderLayer` — G-buffer conditioning without attention

Same goal as `WindowJointAttention` (let the G-buffer steer how the color
features get refined) but done with cheap 1x1 convs and a sigmoid gate
instead of an attention softmax:

```python
condition = torch.cat([x_img, f_img], dim=1)             # color + G-buffer, concatenated on channels
guided = x_img + self.guide_gate(condition) * self.fuse(condition)
#                 ^ sigmoid gate, per-pixel/channel        ^ fused conv features
guided = self.residual_group(guided)   # `depth` stacked SwinIR-style residual convs
guided = self.conv_after_body(guided)  # one more residual conv (SwinIR "conv after body" pattern)
return shortcut + guided                # shortcut = pre-norm x, i.e. another residual around the whole block
```

`guide_gate` and `fuse` both look at the same `concat(x, f)` input but end
in different activations (`Sigmoid` vs `LeakyReLU`) — `fuse` proposes new
features mixing color and G-buffer, `guide_gate` decides how much of that
mix to blend into `x_img` per pixel/channel. This is a much cheaper
substitute for the `attn * attn_f` gating in Part 1: no softmax, no
quadratic window attention cost, just three 1x1/3x3 convs.

`residual_group` is a stack of `SwinIRResidualConv` blocks
(`x + conv(x)`), where `conv` itself (`_make_swinir_conv`) has two modes:
- `"1conv"`: a single 3x3 conv.
- `"3conv"` (default): a bottleneck — 3x3 conv down to `dim/reduction`
  channels → LeakyReLU → 1x1 conv → LeakyReLU → 3x3 conv back up to `dim`.
  This mirrors SwinIR's own RSTB residual-conv design (fewer parameters
  than one large conv at full width).

### Everything else stays a diff of Part 1

- Same `InputProj`/`InputProj_aux`/`OutputProj`, same global residual
  `return x + out` at the very end.
- Same `depths=[1,2,4,8,2,8,4,2,4]`/`num_heads=[1,2,4,8,16,8,4,2,1]` default
  (only `depths[8]` still matters here — it sets how many `SwinIRResidualConv`
  blocks are stacked in `decoderlayer_l0`, not transformer blocks).
- Constructor default `embedded_dim=16` (half of Part 1's 32) — but
  `config_cnn.py` explicitly passes `config["embed_dim"] = 32`, so the actual
  trained model matches Part 1's width. The lower constructor default is
  just a fallback, not what training actually uses.
