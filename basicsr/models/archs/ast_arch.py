import math
import collections

import jittor as jt
import jittor.nn as nn

from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def bmm_4d(a, b):
    B, H = a.shape[0], a.shape[1]
    N_a, C_a = a.shape[2], a.shape[3]

    a_ = a.reshape(B * H, N_a, C_a)

    # 直接根据 b 的最后两个维度判断
    if b.shape[-2:] == (N_a, C_a):  # [B,H,N,C]
        b_ = b.reshape(B * H, N_a, C_a).transpose(0, 2, 1)
    elif b.shape[-2:] == (C_a, N_a):  # [B,H,C,N]
        b_ = b.reshape(B * H, C_a, N_a)
    else:
        # 直接打印帮助调试
        raise ValueError(f"Unexpected b shape: {b.shape}, expected last two dims to be {(N_a, C_a)} or {(C_a, N_a)}")

    out = jt.bmm(a_, b_)
    return out.reshape(B, H, N_a, -1)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training(), self.scale_by_keep)


def drop_path(x, drop_prob=0.0, training=True, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = (jt.rand(shape) < keep_prob).float()
    if scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


#########################################
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def execute(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def execute(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.reshape((1,) + attn_kv.shape)  # 相当于 unsqueeze(0)
            attn_kv = attn_kv.broadcast((B_,) + attn_kv.shape[1:])
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


#########################################
########### window-based self-attention #############


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        win_size,
        num_heads,
        token_projection="linear",
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5

        # 相对位置编码参数表
        relative_position_bias_shape = (
            (2 * win_size[0] - 1) * (2 * win_size[1] - 1),
            num_heads,
        )
        self.relative_position_bias_table = jt.Var(jt.zeros(relative_position_bias_shape), requires_grad=True)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 构建相对位置索引
        coords_h = jt.arange(win_size[0])
        coords_w = jt.arange(win_size[1])
        coords = jt.stack(jt.meshgrid([coords_h, coords_w]))  # shape: [2, Wh, Ww]
        coords_flatten = coords.reshape([2, -1])  # shape: [2, Wh*Ww]
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0)  # [N, N, 2]
        relative_coords[:, :, 0] += win_size[0] - 1
        relative_coords[:, :, 1] += win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * win_size[1] - 1
        relative_position_index = (relative_coords[:, :, 0] + relative_coords[:, :, 1]).int()
        self.register_buffer("relative_position_index", relative_position_index)

        if token_projection == "linear":
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise ValueError("Unsupported token projection type!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = bmm_4d(q, k.transpose(0, 1).transpose(2, 3))  # (B, H, N, N)

        # 相对位置偏置
        bias_table = self.relative_position_bias_table[self.relative_position_index.reshape([-1])]
        bias_table = bias_table.reshape(
            [
                self.win_size[0] * self.win_size[1],
                self.win_size[0] * self.win_size[1],
                -1,
            ]
        )
        bias_table = bias_table.permute(2, 0, 1)  # [num_heads, N, N]

        ratio = attn.shape[-1] // bias_table.shape[-1]
        bias_table = bias_table.broadcast((bias_table.shape[0], bias_table.shape[1], bias_table.shape[2] * ratio))

        attn = attn + bias_table.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.broadcast((mask.shape[0], mask.shape[1], mask.shape[2] * ratio))
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N * ratio])
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N * ratio])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        out = bmm_4d(attn, v).transpose(1, 2).reshape([B_, N, C])
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}"


########### window-based self-attention #############
class WindowAttention_sparse(nn.Module):
    def __init__(
        self,
        dim,
        win_size,
        num_heads,
        token_projection="linear",
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5

        # 相对位置偏置参数表
        relative_position_shape = (
            (2 * win_size[0] - 1) * (2 * win_size[1] - 1),
            num_heads,
        )
        self.relative_position_bias_table = jt.zeros(relative_position_shape)
        self.relative_position_bias_table.requires_grad = True
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 相对位置索引计算
        coords_h = jt.arange(win_size[0])
        coords_w = jt.arange(win_size[1])
        coords = jt.stack(jt.meshgrid([coords_h, coords_w]))  # shape: [2, Wh, Ww]
        coords_flatten = coords.reshape([2, -1])  # shape: [2, Wh*Ww]
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0)  # [N, N, 2]
        relative_coords[:, :, 0] += win_size[0] - 1
        relative_coords[:, :, 1] += win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * win_size[1] - 1
        relative_position_index = (relative_coords[:, :, 0] + relative_coords[:, :, 1]).int()
        self.register_buffer("relative_position_index", relative_position_index)

        if token_projection == "linear":
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise ValueError("Unsupported token projection!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = jt.ones(2)
        self.w.requires_grad = True

    def execute(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        # matmul
        attn = bmm_4d(q, k.transpose(0, 1).transpose(2, 3))
        # attn = bmm_4d(q, k.transpose(0, 1).transpose(2, 3))  # shape: [B, H, N, N]

        # 相对位置偏置
        bias = self.relative_position_bias_table[self.relative_position_index.reshape([-1])]
        bias = bias.reshape(
            [
                self.win_size[0] * self.win_size[1],
                self.win_size[0] * self.win_size[1],
                -1,
            ]
        )
        bias = bias.permute(2, 0, 1)  # [H, N, N]

        ratio = attn.shape[-1] // bias.shape[-1]
        bias = bias.broadcast((bias.shape[0], bias.shape[1], bias.shape[2] * ratio))

        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.broadcast((mask.shape[0], mask.shape[1], mask.shape[2] * ratio))
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N * ratio])
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N * ratio])

        attn0 = self.softmax(attn)
        attn1 = (self.relu(attn)) ** 2

        # 权重 softmax 融合
        w_soft = jt.exp(self.w) / jt.sum(jt.exp(self.w))
        w1, w2 = w_soft[0], w_soft[1]
        attn = attn0 * w1 + attn1 * w2

        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}"


########### self-attention #############
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        token_projection="linear",
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"


#########################################
########### feed-execute network #############
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.0, use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                groups=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            act_layer(),
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity()

    def execute(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        b, hw, c = x.shape
        x = x.reshape(b, hh, hh, c).permute(0, 3, 1, 2)
        # x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        # x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x


class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.0, use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                groups=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            act_layer(),
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def execute(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        b, hw, c = x.shape
        x = x.reshape(b, hh, hh, c).permute(0, 3, 1, 2)
        # x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)

        (
            x1,
            x2,
        ) = jt.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = jt.concat((x1, x2), dim=1)

        # flaten
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        # x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)
        b, hw, c = x_1.shape
        x_1 = x_1.reshape(b, hh, hh, c).permute(0, 3, 1, 2)
        # x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        x_1 = self.dwconv(x_1)
        b, c, h, w = x_1.shape
        x_1 = x_1.permute(0, 2, 3, 1).reshape(b, h * w, c)
        # x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h = hh, w = hh)
        x = x_1 * x_2

        x = self.linear2(x)
        # x = self.eca(x)

        return x


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, "dilation_rate should be a int"
        x = nn.unfold(
            x,
            kernel_size=win_size,
            dilation=dilation_rate,
            padding=4 * (dilation_rate - 1),
            stride=win_size,
        )  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = nn.fold(
            x,
            (H, W),
            kernel_size=win_size,
            dilation=dilation_rate,
            padding=4 * (dilation_rate - 1),
            stride=win_size,
        )
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#########################################


# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def execute(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def execute(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=64,
        kernel_size=3,
        stride=1,
        norm_layer=None,
        act_layer=nn.LeakyReLU,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=stride,
                padding=kernel_size // 2,
            ),
            act_layer(),
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def execute(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(
        self,
        in_channel=64,
        out_channel=3,
        kernel_size=3,
        stride=1,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=stride,
                padding=kernel_size // 2,
            ),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def execute(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


#########################################
###########Transformer Block#############
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        win_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        token_projection="linear",
        token_mlp="leff",
        att=True,
        sparseAtt=False,
    ):
        super().__init__()

        self.att = att
        self.sparseAtt = sparseAtt

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if self.att:
            self.norm1 = norm_layer(dim)
            if self.sparseAtt:
                self.attn = WindowAttention_sparse(
                    dim,
                    win_size=to_2tuple(self.win_size),
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    token_projection=token_projection,
                )
            else:
                self.attn = WindowAttention(
                    dim,
                    win_size=to_2tuple(self.win_size),
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    token_projection=token_projection,
                )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ["ffn", "mlp"]:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        elif token_mlp == "leff":
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == "frfn":
            self.mlp = FRFN(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def execute(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask is not None:
            input_mask = nn.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = jt.zeros((1, H, W, 1)).type_as(x)
            h_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        shortcut = x

        if self.att:
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = jt.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
            x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
            shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = jt.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x


#########################################
########### Basic layer of AST ################
class BasicASTLayer(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        input_resolution,
        depth,
        num_heads,
        win_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        token_projection="linear",
        token_mlp="ffn",
        shift_flag=True,
        att=False,
        sparseAtt=False,
    ):
        super().__init__()
        self.att = att
        self.sparseAtt = sparseAtt
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        win_size=win_size,
                        shift_size=0 if (i % 2 == 0) else win_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        token_projection=token_projection,
                        token_mlp=token_mlp,
                        att=self.att,
                        sparseAtt=self.sparseAtt,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        win_size=win_size,
                        shift_size=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        token_projection=token_projection,
                        token_mlp=token_mlp,
                        att=self.att,
                        sparseAtt=self.sparseAtt,
                    )
                    for i in range(depth)
                ]
            )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def execute(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = blk(x)
                # x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x


class AST(nn.Module):
    def __init__(
        self,
        img_size=256,
        in_chans=3,
        dd_in=3,
        embed_dim=32,
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        token_projection="linear",
        token_mlp="leff",
        dowsample=Downsample,
        upsample=Upsample,
        shift_flag=True,
        **kwargs,
    ):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths[: self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(
            in_channel=dd_in,
            out_channel=embed_dim,
            kernel_size=3,
            stride=1,
            act_layer=nn.LeakyReLU,
        )
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicASTLayer(
            dim=embed_dim,
            output_dim=embed_dim,
            input_resolution=(img_size, img_size),
            depth=depths[0],
            num_heads=num_heads[0],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:0]) : sum(depths[:1])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=False,
            sparseAtt=False,
        )
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicASTLayer(
            dim=embed_dim * 2,
            output_dim=embed_dim * 2,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[1],
            num_heads=num_heads[1],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:1]) : sum(depths[:2])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=False,
            sparseAtt=False,
        )
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicASTLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            input_resolution=(img_size // (2**2), img_size // (2**2)),
            depth=depths[2],
            num_heads=num_heads[2],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:2]) : sum(depths[:3])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=False,
            sparseAtt=False,
        )
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicASTLayer(
            dim=embed_dim * 8,
            output_dim=embed_dim * 8,
            input_resolution=(img_size // (2**3), img_size // (2**3)),
            depth=depths[3],
            num_heads=num_heads[3],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:3]) : sum(depths[:4])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=False,
            sparseAtt=False,
        )
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = BasicASTLayer(
            dim=embed_dim * 16,
            output_dim=embed_dim * 16,
            input_resolution=(img_size // (2**4), img_size // (2**4)),
            depth=depths[4],
            num_heads=num_heads[4],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=conv_dpr,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=True,
            sparseAtt=True,
        )

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicASTLayer(
            dim=embed_dim * 16,
            output_dim=embed_dim * 16,
            input_resolution=(img_size // (2**3), img_size // (2**3)),
            depth=depths[5],
            num_heads=num_heads[5],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[: depths[5]],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=True,
            sparseAtt=True,
        )
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicASTLayer(
            dim=embed_dim * 8,
            output_dim=embed_dim * 8,
            input_resolution=(img_size // (2**2), img_size // (2**2)),
            depth=depths[6],
            num_heads=num_heads[6],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:6]) : sum(depths[5:7])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=True,
            sparseAtt=True,
        )
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicASTLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[7],
            num_heads=num_heads[7],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:7]) : sum(depths[5:8])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=True,
            sparseAtt=True,
        )
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicASTLayer(
            dim=embed_dim * 2,
            output_dim=embed_dim * 2,
            input_resolution=(img_size, img_size),
            depth=depths[8],
            num_heads=num_heads[8],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:8]) : sum(depths[5:9])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            att=True,
            sparseAtt=True,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def execute(self, x, mask=None):
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = jt.concat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = jt.concat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = jt.concat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        up3 = self.upsample_3(deconv2)
        deconv3 = jt.concat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)
        return x + y if self.dd_in == 3 else y
