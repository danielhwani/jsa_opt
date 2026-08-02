# `model_joint_sa.py` architecture notes

`JSA_transformer` is a Uformer-style U-shaped window transformer extended to
take **two** inputs — the noisy color image and a G-buffer (albedo/normal/depth) —
and fuse them inside the attention operator itself. This note explains where
the "Joint" in Joint Self-Attention actually happens, since it's easy to miss
reading the forward pass top to bottom.

## 1. Overall shape — U-Net + 2-branch

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

## 2. The core idea: `WindowJointAttention` — multiply the two attention maps

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

## 3. FFN is not a plain MLP (`Simple_mlp`)

`Linear → depthwise 3x3 conv → activation → Linear`. Window attention alone
only sees inside one window, so the depthwise conv sandwiched between the
two linear layers injects local spatial context across window boundaries —
the same role LeFF plays in Uformer.

## 4. One asymmetry worth knowing: the top decoder level

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

## 5. Output: residual, not a direct prediction

```python
return x + out
```

The network predicts a delta added to the noisy input rather than the clean
image directly, so training starts close to an identity mapping.

## 6. Depths/heads are symmetric around the bottleneck

`depths=[1,2,4,8,2,8,4,2,4]`, `num_heads=[1,2,4,8,16,8,4,2,1]` — both grow
going into the bottleneck and shrink coming back out, mirroring the U-shape
of the network itself.
