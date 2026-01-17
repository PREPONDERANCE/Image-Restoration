import numbers
import jittor as jt
import jittor.nn as nn


##########################################################################
def to_3d(x: jt.Var):
    # ===========================================
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)
    # ===========================================


def to_4d(x: jt.Var, h: int, w: int):
    # ===========================================
    b, hw, c = x.shape
    assert hw == h * w, f"hw={hw} 和 h*w={h * w} 不一致"
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)
    # ===========================================


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # ===========================================
        normalized_shape = tuple(normalized_shape)
        # ===========================================
        assert len(normalized_shape) == 1

        # ===========================================
        self.weight = jt.ones(normalized_shape)
        # ===========================================
        self.normalized_shape = normalized_shape

    def execute(self, x):
        sigma = x.var(-1, keepdims=True, unbiased=False)
        return x / jt.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # ****************************************
        normalized_shape = tuple(normalized_shape)
        # ****************************************
        assert len(normalized_shape) == 1

        # ****************************************
        self.weight = jt.ones(normalized_shape)
        self.bias = jt.zeros(normalized_shape)
        # ****************************************
        self.normalized_shape = normalized_shape

    def execute(self, x):
        mu = x.mean(-1, keepdims=True)
        sigma = x.var(-1, keepdims=True, unbiased=False)
        return (x - mu) / jt.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, ln_type):
        super(LayerNorm, self).__init__()
        if ln_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def execute(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.groups = 12
        self.dim_conv = dim // (2 * self.groups)
        self.dim_untouched = dim // self.groups - self.dim_conv

        self.partial_conv3 = nn.Conv2d(
            self.dim_conv, self.dim_conv, 3, 1, 1, groups=self.dim_conv, bias=False
        )
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def execute(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        # ****************************************
        x1, x2 = jt.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        # ****************************************

        x1 = self.partial_conv3(x1)
        # ****************************************
        x = jt.concat([x1, x2], dim=1)
        # ****************************************
        x = x.reshape(b, -1, h, w)

        # 1x1 卷积升到 2*hidden_features
        x = self.project_in(x)
        c2 = x.shape[1]  # == hidden_features * 2
        half = c2 // 2
        x1, x2 = jt.split(x, [half, c2 - half], dim=1)
        x1 = self.dwconv(x1)
        x = x1 * x2
        x = self.project_out(x)
        return x


##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.gelu = nn.GELU()
        self.temperature = jt.ones(num_heads, 1, 1)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()
        self.att_ca = nn.Conv2d(
            dim // num_heads, 2 * (dim // num_heads), kernel_size=1, bias=bias
        )

    def execute(self, x):
        b, c, h, w = x.shape
        head = self.num_heads
        ch = c // head  # 每个 head 的通道数
        L = h * w  # 空间展平长度

        qkv = self.qkv_dwconv(self.qkv(x))  # (b, 3c, h, w)
        part = qkv.shape[1] // 3
        q, k, v = jt.split(qkv, [part, part, part], dim=1)  # => (b,c,h,w) ×3

        # 等价于：rearrange(q, 'b (head c) h w -> b head c (h w)', head=head)
        q = q.reshape(b, head, ch, h, w).reshape(b, head, ch, L)  # (b, head, ch, L)
        k = k.reshape(b, head, ch, h, w).reshape(b, head, ch, L)
        v = v.reshape(b, head, ch, h, w).reshape(b, head, ch, L)

        # 归一化（torch.nn.functional.normalize -> jt.normalize）
        q = jt.normalize(q, dim=-1)
        k = jt.normalize(k, dim=-1)

        # attn = (q @ k^T) * temperature
        # PyTorch 的 transpose(-2,-1) 等价为 permute 交换最后两个维度
        kt = k.permute(0, 1, 3, 2)  # (b, head, L, ch)
        attn = (q @ kt) * self.temperature  # (b, head, ch, ch)

        attn0 = attn.softmax(dim=-1)  # (b, head, ch, ch)
        attn1 = self.relu(attn) ** 2
        attn1 = self.gelu(attn1) * attn1  # (b, head, ch, ch)

        # 等价于：rearrange(attn1, 'b head L c -> b c head L')
        attn1 = attn1.permute(0, 2, 1, 3)  # (b, ch, head, ch)

        # att_ca 是 2D 卷积，(N,C,H,W) 对应这里的 (b, ch, head, ch)
        x_att = self.att_ca(attn1)  # (b, 2*ch, head, ch)

        # 等价于：rearrange(x_att, 'b c head L -> b head L c')
        x_att = x_att.permute(0, 2, 3, 1)  # (b, head, ch, 2*ch)

        # PyTorch 的 x_att.chunk(2, dim=-1) => Jittor 用“段大小列表”
        scale, shift = jt.split(x_att, [ch, ch], dim=-1)  # (b, head, ch, ch) ×2

        attn = attn0 * (1 + scale) + shift  # (b, head, ch, ch)

        out = attn @ v  # (b, head, ch, L)

        # 等价于：rearrange(out, 'b head c (h w) -> b (head c) h w', ...)
        out = out.reshape(b, head * ch, h, w)  # (b, c, h, w)

        out = self.project_out(out)  # (b, c, h, w)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, ln_type, isAtt):
        super(TransformerBlock, self).__init__()
        self.isAtt = isAtt
        if self.isAtt:
            self.norm1 = LayerNorm(dim, ln_type)
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, ln_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def execute(self, x):
        if self.isAtt:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def execute(self, x):
        x = self.proj(x)

        return x


##########################################################################
class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        assert downscale_factor > 0, (
            f"downscale_factor must be > 0, got {downscale_factor}"
        )
        self.r = int(downscale_factor)

    def execute(self, x):
        n, c, h, w = x.shape
        r = self.r
        assert h % r == 0 and w % r == 0, (
            f"H and W must be divisible by downscale_factor={r} in PixelUnshuffle"
        )
        # 输出: (N, C*r*r, H/r, W/r)
        return x.reindex(
            [n, c * r * r, h // r, w // r],
            [
                "i0",  # n  -> n
                f"i1/{r * r}",  # c' -> c_in
                f"i2*{r} + (i1/{r})%{r}",  # h' -> h_in
                f"i3*{r} + i1%{r}",  # w' -> w_in
            ],
        )


## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            PixelUnshuffle(2),
        )

    def execute(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def execute(self, x):
        return self.body(x)


##########################################################################
##---------- ASTv2 -----------------------
class ASTv2(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        ln_type="WithBias",  ## Other option 'BiasFree'
        dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(ASTv2, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    isAtt=False,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    isAtt=False,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    isAtt=False,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    isAtt=True,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    isAtt=True,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    isAtt=True,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    ln_type=ln_type,
                    isAtt=True,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def execute(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = jt.concat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = jt.concat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
