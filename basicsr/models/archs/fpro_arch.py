import numbers
import jittor as jt
from jittor.einops import rearrange

from jittor import nn
import numpy as np

F = nn

nn.normalize = lambda x, dim=-1, eps=1e-12: x / (
    jt.norm(x, p=2, dim=dim, keepdims=True) + eps
)


def rfft2(x):
    x_np = x.numpy()
    y_np = np.fft.rfft2(x_np, axes=(-2, -1))
    y_np = np.stack([y_np.real, y_np.imag], axis=-1).astype(np.float32)
    return jt.array(y_np)


def irfft2(x, s=None):
    x_np = x.numpy()
    complex_np = x_np[..., 0] + 1j * x_np[..., 1]
    y_np = np.fft.irfft2(complex_np, s=s, axes=(-2, -1))
    return jt.array(y_np.astype(np.float32))


def to_3d(x):
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).transpose(0, 2, 1)


def to_4d(x, H, W):
    B, _, C = x.shape
    return x.transpose(0, 2, 1).reshape(B, C, H, W)


class PreciseGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, x):
        return 0.5 * x * (1.0 + jt.erf(x / 1.41421356237))


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = jt.ones(normalized_shape)
        self.normalized_shape = normalized_shape

    def execute(self, x):
        sigma = ((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True)
        return (x / jt.sqrt(sigma + 1e-05)) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = jt.ones(normalized_shape)
        self.bias = jt.zeros(normalized_shape)
        self.normalized_shape = normalized_shape

    def execute(self, x):
        mu = x.mean(-1, keepdims=True)
        sigma = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (((x - mu) / jt.sqrt(sigma + 1e-05)) * self.weight) + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, ln_type):
        super(LayerNorm, self).__init__()
        if ln_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def execute(self, x):
        (h, w) = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w).clone()


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int((dim * ffn_expansion_factor))
        self.project_in = nn.Conv2d(dim, (hidden_features * 2), 1, bias=bias)
        self.dwconv = nn.Conv2d(
            (hidden_features * 2),
            (hidden_features * 2),
            3,
            stride=1,
            padding=1,
            groups=(hidden_features * 2),
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)
        self.gelu = PreciseGELU()

    def execute(self, x):
        x = self.project_in(x)
        (x1, x2) = jt.chunk(self.dwconv(x), 2, dim=1)
        x = self.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = jt.ones((num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, (dim * 3), 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            (dim * 3), (dim * 3), 3, stride=1, padding=1, groups=(dim * 3), bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def execute(self, x):
        (b, c, h, w) = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        (q, k, v) = jt.chunk(qkv, 3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = jt.matmul(attn, v)
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out


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


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, stride=1, padding=1, bias=bias)

    def execute(self, x):
        x = self.proj(x)
        return x


def window_partition(x, win_size, dilation_rate=1):
    (B, H, W, C) = x.shape
    if dilation_rate != 1:
        x = x.transpose(0, 3, 1, 2)
        assert type(dilation_rate) is int, "dilation_rate should be a int"
        x = F.unfold(
            x,
            kernel_size=win_size,
            dilation=dilation_rate,
            padding=(4 * (dilation_rate - 1)),
            stride=win_size,
        )
        windows = x.transpose(0, 2, 1).reshape((-1, C, win_size, win_size))
        windows = windows.transpose(0, 2, 3, 1)
    else:
        x = x.reshape((B, (H // win_size), win_size, (W // win_size), win_size, C))
        windows = x.transpose(0, 1, 3, 2, 4, 5).reshape((-1, win_size, win_size, C))
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    B = int((windows.shape[0] / (((H * W) / win_size) / win_size)))
    x = windows.reshape((B, (H // win_size), (W // win_size), win_size, win_size, -1))
    if dilation_rate != 1:
        x = windows.transpose(0, 5, 3, 4, 1, 2)
        x = F.fold(
            x,
            (H, W),
            kernel_size=win_size,
            dilation=dilation_rate,
            padding=(4 * (dilation_rate - 1)),
            stride=win_size,
        )
    else:
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape((B, H, W, -1))
    return x


class lowFrequencyPromptFusion(nn.Module):
    def __init__(self, dim, dim_bak, num_heads, win_size=8, bias=False):
        super(lowFrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.temperature = jt.ones((num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, 1, bias=bias)
        self.ap_kv = nn.AdaptiveAvgPool2d(1)
        self.kv = nn.Conv2d(dim_bak, (dim * 2), 1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def execute(self, feature, prompt_feature):
        (b, c1, h, w) = feature.shape
        (_, c2, _, _) = prompt_feature.shape
        query = (
            self.q(feature)
            .reshape((b, (h * w), self.num_heads, (c1 // self.num_heads)))
            .transpose(0, 2, 1, 3)
        )
        prompt_feature = self.ap_kv(prompt_feature)
        key_value = (
            self.kv(prompt_feature)
            .reshape((b, (2 * c1), -1))
            .transpose(0, 2, 1)
            .reshape((b, -1, 2, self.num_heads, (c1 // self.num_heads)))
            .transpose(2, 0, 3, 1, 4)
        )
        (key, value) = (key_value[0], key_value[1])
        attn = jt.matmul(query, key.transpose(0, 1, 3, 2)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = jt.matmul(attn, value)
        out = rearrange(
            out, "b head (h w) c -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=True, isQuery=True):
        super().__init__()
        self.isQuery = isQuery
        inner_dim = dim_head * heads
        self.heads = heads
        if self.isQuery:
            self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        else:
            self.to_kv = nn.Linear(dim, (2 * inner_dim), bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def execute(self, x, attn_kv=None):
        (B_, N, C) = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.shape[1]
        if self.isQuery:
            q = (
                self.to_q(x)
                .reshape((B_, N, 1, self.heads, (C // self.heads)))
                .transpose(2, 0, 3, 1, 4)
            )
            q = q[0]
            return q
        else:
            C_inner = self.inner_dim
            kv = (
                self.to_kv(attn_kv)
                .reshape((B_, N_kv, 2, self.heads, (C_inner // self.heads)))
                .transpose(2, 0, 3, 1, 4)
            )
            (k, v) = (kv[0], kv[1])
            return (k, v)


class highFrequencyPromptFusion(nn.Module):
    def __init__(
        self, dim, dim_bak, win_size, num_heads, qkv_bias=True, qk_scale=None, bias=False
    ):
        super(highFrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.win_size = win_size
        head_dim = dim // num_heads
        self.scale = qk_scale or (head_dim**-0.5)
        self.to_q = LinearProjection(
            dim, num_heads, (dim // num_heads), bias=qkv_bias, isQuery=True
        )
        self.to_kv = LinearProjection(
            dim_bak, num_heads, (dim // num_heads), bias=qkv_bias, isQuery=False
        )
        self.kv_dwconv = nn.Conv2d(
            dim_bak, dim_bak, 3, stride=1, padding=1, groups=dim_bak, bias=bias
        )
        self.softmax = nn.Softmax(dim=-1)
        self.project_out = nn.Linear(dim, dim)

    def execute(self, query_feature, key_value_feature):
        (b, c, h, w) = query_feature.shape
        (_, c_2, _, _) = key_value_feature.shape
        key_value_feature = self.kv_dwconv(key_value_feature)
        query_feature = rearrange(query_feature, " b c1 h w -> b h w c1 ", h=h, w=w)
        query_feature_windows = window_partition(query_feature, self.win_size)
        query_feature_windows = query_feature_windows.reshape(
            (-1, (self.win_size * self.win_size), c)
        )
        key_value_feature = rearrange(
            key_value_feature, " b c2 h w -> b h w c2 ", h=h, w=w
        )
        key_value_feature_windows = window_partition(key_value_feature, self.win_size)
        key_value_feature_windows = key_value_feature_windows.reshape(
            (-1, (self.win_size * self.win_size), c_2)
        )
        (B_, N, C) = query_feature_windows.shape
        query = self.to_q(query_feature_windows)
        query = query * self.scale
        (key, value) = self.to_kv(key_value_feature_windows)
        attn = jt.matmul(query, key.transpose((0, 1, 3, 2)))
        attn = self.softmax(attn)
        out = jt.matmul(attn, value).transpose((0, 1, 3, 2))
        out = out.reshape((B_, N, C))
        out = self.project_out(out)
        attn_windows = out.reshape((-1, self.win_size, self.win_size, C))
        attn_windows = window_reverse(attn_windows, self.win_size, h, w)
        return rearrange(attn_windows, "b h w c -> b c h w", h=h, w=w)


class dynamic_filter_channel(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.conv = nn.Conv2d(inchannels, group * kernel_size**2, 1, bias=False)
        self.conv_gate = nn.Conv2d(
            group * kernel_size**2, group * kernel_size**2, 1, bias=False
        )
        self.act_gate = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(group * kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap_1 = nn.AdaptiveAvgPool2d((1, 1))

    def execute(self, x):
        identity_input = x
        low_filter1 = self.ap_1(x)
        low_filter = self.conv(low_filter1)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        low_filter = self.bn(low_filter)
        n, c, h, w = x.shape
        unfolded_x = F.unfold(self.pad(x), kernel_size=self.kernel_size)
        x_unfolded_reshaped = unfolded_x.reshape(
            (n, self.group, c // self.group, self.kernel_size**2, h * w)
        )
        n_f, c1, p, q = low_filter.shape
        low_filter_reshaped = low_filter.reshape(
            (n_f, c1 // (self.kernel_size**2), self.kernel_size**2, p * q)
        ).unsqueeze(2)
        low_filter_activated = self.act(low_filter_reshaped)
        low_part = jt.sum(x_unfolded_reshaped * low_filter_activated, dim=3).reshape(
            (n, c, h, w)
        )
        out_high = identity_input - low_part
        return low_part, out_high


class frequenctSpecificPromptGenetator(nn.Module):
    def __init__(self, dim=3, h=128, w=65, flag_highF=True):
        super().__init__()
        self.flag_highF = flag_highF
        k_size = 3
        if flag_highF:
            w_init = (w - 1) * 2
            h_init = h
            self.weight = jt.randn((1, dim, h_init, w_init), dtype="float32") * 0.02
            self.body = nn.Sequential(
                nn.Conv2d(dim, dim, (1, k_size), padding=(0, k_size // 2), groups=dim),
                nn.Conv2d(dim, dim, (k_size, 1), padding=(k_size // 2, 0), groups=dim),
                PreciseGELU(),
            )
        else:
            h_init = h
            w_init = w
            self.complex_weight = (
                jt.randn((1, dim, h_init, w_init, 2), dtype="float32") * 0.02
            )
            self.body = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=1), PreciseGELU()
            )

    def execute(self, ffm, H, W):
        F = jt.nn
        if self.flag_highF:
            ffm = F.interpolate(ffm, size=(H, W), mode="bilinear", align_corners=False)
            y_att = self.body(ffm)
            y_f = y_att * ffm
            weight_resized = F.interpolate(
                self.weight, size=(H, W), mode="bilinear", align_corners=False
            )
            y = y_f * weight_resized

        else:
            ffm = F.interpolate(ffm, size=(H, W), mode="bicubic", align_corners=False)
            zeros = jt.zeros_like(ffm)
            y_fft_list_complex = [
                jt.nn.ComplexNumber(ffm[i], zeros[i]).fft2() for i in range(ffm.shape[0])
            ]
            y_fft_reals = [c.real for c in y_fft_list_complex]
            y_fft_imags = [c.imag for c in y_fft_list_complex]
            y_real_stacked = jt.stack(y_fft_reals, dim=0)
            y_imag_stacked = jt.stack(y_fft_imags, dim=0)
            y = jt.nn.ComplexNumber(y_real_stacked, y_imag_stacked)
            y_f = jt.concat([y.real, y.imag], dim=1)
            weight = jt.nn.ComplexNumber(
                self.complex_weight[..., 0], self.complex_weight[..., 1]
            )
            y_att = self.body(y_f)
            y_f = y_f * y_att
            y_real, y_imag = jt.chunk(y_f, 2, dim=1)
            y = jt.nn.ComplexNumber(y_real, y_imag)
            weight_real_resized = F.interpolate(
                weight.real, size=(H, W), mode="bilinear", align_corners=False
            )
            weight_imag_resized = F.interpolate(
                weight.imag, size=(H, W), mode="bilinear", align_corners=False
            )
            weight_resized = jt.nn.ComplexNumber(weight_real_resized, weight_imag_resized)
            y = y * weight_resized
            y_ifft_list = []
            for i in range(y.real.shape[0]):
                sample_real = y.real[i]
                sample_imag = y.imag[i]
                sample_complex = jt.nn.ComplexNumber(sample_real, sample_imag)
                y_ifft_list.append(sample_complex.ifft2())
            y_ifft_reals = [c.real for c in y_ifft_list]
            y = jt.stack(y_ifft_reals, dim=0)

        return y


class PromptModule(nn.Module):
    def __init__(self, basic_dim=32, dim=32, input_resolution=128):
        super().__init__()
        h = input_resolution
        w = (input_resolution // 2) + 1
        self.simple_Fusion = nn.Conv2d(2 * dim, dim, 1, stride=1)
        self.FSPG_high = frequenctSpecificPromptGenetator(
            basic_dim, h, w, flag_highF=True
        )
        self.FSPG_low = frequenctSpecificPromptGenetator(
            basic_dim, h, w, flag_highF=False
        )
        self.modulator_hi = highFrequencyPromptFusion(
            dim, basic_dim, win_size=8, num_heads=2, bias=False
        )
        self.modulator_lo = lowFrequencyPromptFusion(
            dim, basic_dim, win_size=8, num_heads=2, bias=False
        )

    def execute(self, low_part, out_high, x):
        (b, c, h, w) = x.shape
        y_h = self.FSPG_high(out_high, h, w)
        y_l = self.FSPG_low(low_part, h, w)
        y_h = self.modulator_hi(x, y_h)
        y_l = self.modulator_lo(x, y_l)
        x = self.simple_Fusion(jt.concat([y_h, y_l], dim=1))
        return x


class splitFrequencyModule(nn.Module):
    def __init__(self, basic_dim=32, dim=32, input_resolution=128):
        super().__init__()
        self.dyna_channel = dynamic_filter_channel(inchannels=basic_dim)

    def execute(self, F_low):
        low_part, out_high = self.dyna_channel(F_low)
        return low_part, out_high


# PixelUnshuffle
class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.r = downscale_factor

    def execute(self, x):
        b, c, h, w = x.shape
        assert h % self.r == 0 and w % self.r == 0, (
            "Height and Width must be divisible by downscale_factor"
        )

        oc = c * (self.r**2)
        oh = h // self.r
        ow = w // self.r

        output = x.reindex(
            [b, c, oh, ow, self.r, self.r],
            [
                "i0",  # B
                "i1",  # C
                "i2*i4+i5",  # H = oh*r + remainder
                "i3*i4+i6",  # W = ow*r + remainder
            ],
        )

        output = x.reshape((b, c, oh, self.r, ow, self.r))
        output = output.transpose((0, 1, 3, 5, 2, 4))  # (B, C, r, r, oh, ow)
        output = output.reshape((b, oc, oh, ow))

        return output


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, (n_feat // 2), 3, stride=1, padding=1, bias=False),
            PixelUnshuffle(2),
        )

    def execute(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, (n_feat * 2), 3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def execute(self, x):
        return self.body(x)


class FPro(nn.Module):
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
        ln_type="WithBias",
        dual_pixel_task=False,
    ):
        super(FPro, self).__init__()
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
        self.down1_2 = Downsample(dim)
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
        self.down2_3 = Downsample(int(dim * 2**1))
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
        self.splitFre = splitFrequencyModule(
            basic_dim=dim, dim=int(dim * 2**2), input_resolution=32
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
        self.prompt_d3 = PromptModule(
            basic_dim=dim, dim=int(dim * 2**2), input_resolution=64
        )
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), 1, bias=bias
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
        self.prompt_d2 = PromptModule(
            basic_dim=dim, dim=int(dim * 2**1), input_resolution=128
        )
        self.up2_1 = Upsample(int(dim * 2**1))
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
        self.prompt_d1 = PromptModule(
            basic_dim=dim, dim=int(dim * 2**1), input_resolution=256
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
        self.prompt_r = PromptModule(
            basic_dim=dim, dim=int(dim * 2**1), input_resolution=256
        )
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), 1, bias=bias)
        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, 3, stride=1, padding=1, bias=bias
        )

    def execute(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_dec_level3 = self.decoder_level3(out_enc_level3)
        (low_part, out_high) = self.splitFre(inp_enc_level1)
        out_dec_level3 = (
            self.prompt_d3(low_part, out_high, out_dec_level3) + out_dec_level3
        )
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = jt.concat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = (
            self.prompt_d2(low_part, out_high, out_dec_level2) + out_dec_level2
        )
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = jt.concat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = (
            self.prompt_d1(low_part, out_high, out_dec_level1) + out_dec_level1
        )
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = (
            self.prompt_r(low_part, out_high, out_dec_level1) + out_dec_level1
        )
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1
