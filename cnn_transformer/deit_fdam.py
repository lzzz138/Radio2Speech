# --------------------------------------------------------
# Linwei Chen
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from functools import partial

from timm.models.vision_transformer import Mlp as MLP

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.modules.mlp import FusedMLP
    has_flash_attn = True
    print('FlashAttention is installed.')
    
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_first'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            input_dtype = x.dtype
            x = x.to(torch.float32)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None].to(torch.float32) * x + self.bias[:, None, None].to(torch.float32)
            x = x.to(input_dtype)
            return x

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
    

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
            
class GroupDynamicScale(nn.Module):
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.125,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True, group=32, init_scale=1e-5,
                 **kwargs):
        super().__init__()
        
        self.size = size
        self.filter_size = size // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        # self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        # self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, group * num_filters, bias=False)
        self.complex_weights = nn.Parameter(
            torch.randn(num_filters, dim//group, self.size, self.filter_size,dtype=torch.float32) * init_scale)
        trunc_normal_(self.complex_weights, std=init_scale)
        self.act2 = act2_layer()
        # self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias) 
        # self.init_reweight_bias(group, num_filters)

    def init_reweight_bias(self, group, num_filters):
            # 创建一个 (group, num_filters) 的矩阵，对角线部分为单位矩阵，其余为 0
            bias_matrix = torch.zeros(group, num_filters)
            min_dim = min(group, num_filters)
            for i in range(min_dim):
                bias_matrix[i][i] = 1.0
            
            # 展开为一维向量
            bias_vector = bias_matrix.view(-1)
            bias_vector = bias_vector.repeat(group * num_filters // len(bias_vector))
            
            # 设置 fc2 的 bias
            self.reweight.fc2.bias.data = bias_vector

    def forward(self, x):
        B, C, H, W, = x.shape
        x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
        B, C, RH, RW, = x_rfft.shape
        x = x.permute(0, 2, 3, 1)

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, -1, self.num_filters).tanh_() # b, num_filters, group
        # routeing_std = self.reweight(x.std(dim=(1, 2))).view(B, -1, self.num_filters) # b, num_filters, group
        # routeing = (routeing + routeing_std).tanh_()
        # routeing = 1.7159 * (0.66 * (routeing + routeing_std)).tanh_()
        # routeing = self.reweight(x.mean(dim=(1, 2))).view(B, -1, self.num_filters) # b, num_filters, group
        # routeing = routeing / (routeing.abs().sum(dim=-1, keepdims=True) + 1e-8)
        # routeing = self.reweight(x.mean(dim=(1, 2))).view(B, -1, self.num_filters).softmax(dim=-1) # b, num_filters, group
        # routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters, -1)
        # routeing = torch.log2(2 ** routeing + 1)
        weight = self.complex_weights
        if not weight.shape[2:4] == x_rfft.shape[2:4]:
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)
        # routeing = routeing.to(torch.complex64)
        # print(routeing.shape, complex_weights.shape)
        weight = torch.einsum('bgf,fchw->bgchw', routeing, weight)
        weight = weight.reshape(B, C, RH, RW)
        # x = x * weight
        x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        x = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')
        return x
    

from torch.nn.modules.activation import constant_



class FlashAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

        self.dy_freq = nn.Linear(dim, self.num_heads, bias=True)
        # constant_(self.dy_freq.bias, -5)
        self.dy_freq_2 = nn.Linear(dim, self.num_heads, bias=True)
        constant_(self.dy_freq.weight, 1e-8)
        # self.dy_freq_gate = nn.Linear(embed_dim, self.num_heads, bias=False)
        self.dy_freq_starrelu = StarReLU()
        self.ignore_cls_token = 0
        
    def _naive_attn(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        else:
            return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        B, N, C = x.shape
        # print(x.shape)
        # hw_cls, b, c = x.shape
        dy_freq = self.dy_freq_starrelu(x[:, self.ignore_cls_token:])
        dy_freq_2 = self.dy_freq_2(dy_freq).tanh_()
        dy_freq = F.softplus(self.dy_freq(dy_freq) - 5)
        # dy_freq = F.softplus(self.dy_freq(dy_freq)) * self.dy_freq_gate(dy_freq).sigmoid() * 2
        # dy_freq = torch.log2(dy_freq.exp() + 1) - 1
        # dy_freq_2 = self.dy_freq_2(self.dy_freq_starrelu_2(value))
        # dy_freq2 = F.relu(dy_freq, inplace=True) * F.relu(dy_freq_2, inplace=True)
        # dy_freq2 = F.relu(dy_freq, inplace=True).clamp(0, 1)
        # dy_freq2 = F.relu(dy_freq, inplace=True) ** 2
        dy_freq2 = dy_freq ** 2
        # dy_freq = dy_freq2 / (dy_freq2 + 1)
        dy_freq = 2 * dy_freq2 / (dy_freq2 + 0.3678) # 1/e
        # dy_freq_clone = dy_freq.transpose(1, 0).clone()
        # dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token,  self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
        # dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token, C) 
        # dy_freq_spatial = dy_freq_spatial * dy_freq_channel
        if self.ignore_cls_token > 0:
            dy_freq = torch.cat([torch.zeros([B, self.ignore_cls_token, dy_freq.size(-1)], device=dy_freq.device), dy_freq], dim=1)

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        # qkv torch.Size([1, 16, 3, 12, 64])

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        # For val w/o deepspeed
        qkv_dtype = qkv.dtype
        if qkv.dtype not in [torch.float16, torch.bfloat16]:
            qkv = qkv.to(torch.float16)        
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        if qkv_dtype not in [torch.float16, torch.bfloat16]:
            context = context.to(qkv_dtype) # context torch.Size([1, 16, 12, 64])
        # print('qkv', qkv.shape)
        # print('context', context.shape)
        v = qkv[:, :, 2]
        v_hf = v - context
        context = context + context * dy_freq_2[..., None] + dy_freq[..., None] * v_hf
        
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, x, H=None, W=None, return_attn=False):
        if return_attn:
            x, attn = self._naive_attn(x, return_attn)
            return x, attn
        else:        
            x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
            return x
        
class AttentionwithAttInv(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., lf_dy_weight=True, hf_dy_weight=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ### AttInv
        self.lf_dy_weight = lf_dy_weight
        self.hf_dy_weight = hf_dy_weight

        if self.lf_dy_weight:
            self.dy_freq_2 = nn.Linear(dim, self.num_heads, bias=True)
            # self.lf_gamma= nn.Parameter(1e-5 * torch.ones((dim, 1, 1)),requires_grad=True) # with decay if dim > 1
            self.lf_gamma= nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True) # no decay

        if self.hf_dy_weight:
            self.dy_freq = nn.Linear(dim, self.num_heads, bias=True)
            # self.dy_freq_gate = nn.Linear(embed_dim, self.num_heads, bias=False)
            # constant_(self.dy_freq.bias, -5)
            # self.hf_gamma= nn.Parameter(1e-5 * torch.ones((dim, 1, 1)),requires_grad=True) # with decay if dim > 1
            self.hf_gamma= nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True) # no decay
        self.dy_freq_starrelu = StarReLU()
        self.ignore_cls_token = 0
        

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape

        # hw_cls, b, c = x.shape
        dy_freq_feat = self.dy_freq_starrelu(x[:, self.ignore_cls_token:])

        if hasattr(self, 'dy_freq_2'):
            dy_freq_lf = self.dy_freq_2(dy_freq_feat).tanh_()
            # dy_freq_lf = F.softplus(dy_freq_lf)
            # dy_freq_lf = dy_freq_lf ** 2
            # dy_freq_lf = 2 * dy_freq_lf / (dy_freq_lf + 0.3678)
            dy_freq_lf = dy_freq_lf.reshape(B, N - self.ignore_cls_token,  self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
            dy_freq_lf = dy_freq_lf.reshape(B, N - self.ignore_cls_token, C) 

        if hasattr(self, 'dy_freq'):
            dy_freq = F.softplus(self.dy_freq(dy_freq_feat))
            # dy_freq = F.softplus(self.dy_freq(dy_freq)) * self.dy_freq_gate(dy_freq).sigmoid() * 2
            # dy_freq = torch.log2(dy_freq.exp() + 1) - 1
            # dy_freq_2 = self.dy_freq_2(self.dy_freq_starrelu_2(value))
            # dy_freq2 = F.relu(dy_freq, inplace=True) * F.relu(dy_freq_2, inplace=True)
            # dy_freq2 = F.relu(dy_freq, inplace=True).clamp(0, 1)
            # dy_freq2 = F.relu(dy_freq, inplace=True) ** 2
            dy_freq2 = dy_freq ** 2
            # dy_freq = dy_freq2 / (dy_freq2 + 1)
            dy_freq = 2 * dy_freq2 / (dy_freq2 + 0.3678)
            # dy_freq_clone = dy_freq.transpose(1, 0).clone()
            dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token,  self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
            dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token, C) 
            # dy_freq_spatial = dy_freq_spatial * dy_freq_channel
            if self.ignore_cls_token > 0:
                dy_freq = torch.cat([torch.zeros([B, self.ignore_cls_token, C], device=dy_freq.device), dy_freq], dim=1)


        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, head, N, C//head
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # B, head, N, C//head
        v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        v_hf = v - x
        # x = x + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)
        # x = x + x * self.lf_gamma.view(1, 1, -1) + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)
        if hasattr(self, 'dy_freq_2'):
            x = x + x * dy_freq_lf * self.lf_gamma.view(1, 1, -1)
        if hasattr(self, 'dy_freq'):
            x = x + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


class WindowedFlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

        self.causal = causal
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

    def forward(self, x, H, W):
        # cls, x = x[:, :1, :], x[:, 1:, :]
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size
        x = x.view(B, H, W, C)
        x = F.pad(x, [0, 0, 0, W_ - W, 0, H_ - H])
        
        def window_partition(x, window_size):
            """
            Args:
                x: (B, H, W, C)
                window_size (int): window size
            Returns:
                windows: (num_windows*B, window_size, window_size, C)
            """
            B, H, W, C = x.shape
            x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            return windows

        x = window_partition(x, window_size=self.window_size)  # nW*B, window_size, window_size, C
        x = x.view(-1, N_, C)

        def window_reverse(windows, window_size, H, W):
            """
            Args:
                windows: (num_windows*B, window_size, window_size, C)
                window_size (int): Window size
                H (int): Height of image
                W (int): Width of image
            Returns:
                x: (B, H, W, C)
            """
            B = int(windows.shape[0] / (H * W / window_size / window_size))
            x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return x
        
        def _naive_attn(x):
            qkv = self.qkv(x).view(-1, N_, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            if self.qk_normalization:
                B_, H_, _, D_ = q.shape
                q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
            x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)
            return x
        
        def _flash_attn(x):
            qkv = self.qkv(x).view(-1, N_, 3, self.num_heads, C // self.num_heads)
            
            if self.qk_normalization:
                q, k, v = qkv.unbind(2)
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
                qkv = torch.stack([q, k, v], dim=2)
                
            # qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
            # For val w/o deepspeed
            qkv_dtype = qkv.dtype
            if qkv.dtype not in [torch.float16, torch.bfloat16]:
                qkv = qkv.to(torch.float16)
            context, _ = self.inner_attn(qkv, causal=self.causal)
            if qkv_dtype not in [torch.float16, torch.bfloat16]:
                context = context.to(qkv_dtype)
            x = context.reshape(-1, self.window_size, self.window_size, C)
            # x = rearrange(context, "b s h d -> b s (h d)").reshape(-1, self.window_size, self.window_size, C)
            return x
        
        x = _naive_attn(x) if not self.use_flash_attn else _flash_attn(x)
        x = x.contiguous()
        
        x = window_reverse(x, self.window_size, H_, W_)
        x = x[:, :H, :W, :].reshape(B, N, C).contiguous()
        # x = torch.cat([cls, x], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 proj_drop=0., qk_scale=None, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

    def forward(self, x, H, W):
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size

        qkv = self.qkv(x)  # [B, N, C]
        qkv = qkv.transpose(1, 2).reshape(B, C * 3, H, W)  # [B, C, H, W]
        qkv = F.pad(qkv, [0, W_ - W, 0, H_ - H], mode='constant')

        dtype = qkv.dtype
        qkv = F.unfold(qkv.float(), kernel_size=(self.window_size, self.window_size),
                       stride=(self.window_size, self.window_size)).to(dtype)
        
        B, C_kw_kw, L = qkv.shape  # L - the num of windows
        qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1)  # [B, L, N_, C]
        qkv = qkv.reshape(B, L, N_, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # q,k,v [B, L, num_head, N_, C/num_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        # attn @ v = [B, L, num_head, N_, C/num_head]
        x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)

        x = F.fold(x.float(), output_size=(H_, W_),
                   kernel_size=(self.window_size, self.window_size),
                   stride=(self.window_size, self.window_size)).to(dtype)  # [B, C, H_, W_]
        x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = AttentionwithAttInv, Mlp_block=MLP
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
   


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = AttentionwithAttInv, Mlp_block=MLP
                 ,init_values=1e-4, window_size=-1, use_flash_attn=False, with_cp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.with_cp = with_cp
                
        if window_size > 0:
            print("! Using window attention")
            if use_flash_attn:
                print("! Using flash attention")
                self.attn = WindowedFlashAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_flash_attn=use_flash_attn, causal=False, window_size=window_size)
            else:
                self.attn = WindowedAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        else:
            if use_flash_attn:
                print("! Using flash attention")
                self.attn = FlashAttn(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_flash_attn=use_flash_attn, causal=False)
            else:
                self.attn = Attention_block(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.freq_scale_1 = GroupDynamicScale(dim=dim, expansion_ratio=1, reweight_expansion_ratio=.0625, group=16, num_filters=4, size=14, act1_layer=StarReLU, act2_layer=nn.Identity, bias=False, weight_resize=True)
        self.freq_scale_2 = GroupDynamicScale(dim=dim, expansion_ratio=1, reweight_expansion_ratio=.0625, group=16, num_filters=4, size=14, act1_layer=StarReLU, act2_layer=nn.Identity, bias=False, weight_resize=True)

    def forward(self, x, H=None, W=None, return_attn=False):
        # x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
        # x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        # return x
        def _inner_forward(x, H, W, return_attn):
            if not return_attn:
                # x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
                x_att = self.attn(self.norm1(x), H, W)

                x_att = nlc_to_nchw(x_att, (H, W))
                x_att = self.freq_scale_1(x_att) + x_att
                x_att = nchw_to_nlc(x_att)

                x = x + self.drop_path(self.gamma_1 * x_att)

                x_mlp = self.mlp(self.norm2(x))

                x_mlp = nlc_to_nchw(x_mlp, (H, W))
                x_mlp = self.freq_scale_2(x_mlp) + x_mlp
                x_mlp = nchw_to_nlc(x_mlp)

                x = x + self.drop_path(self.gamma_2 * x_mlp)
                return x
            else:
                attn, attn_weight = self.attn(self.norm1(x), H, W, return_attn)
                x = x + self.drop_path(self.gamma_1 * attn)
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
                return x, attn_weight

        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, H, W, return_attn)
        else:
            return _inner_forward(x, H, W, return_attn)


class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = AttentionwithAttInv, Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
       
        
class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = AttentionwithAttInv, Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FdamBlock(nn.Module):
    def __init__(self, hidden_size, mlp_dim, num_heads, transformer_dropout_rate, transformer_attention_dropout_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, transformer_dropout_rate)
        #self.attn = Attention(num_heads, hidden_size, transformer_attention_dropout_rate)
        
        # --- FDAM-specific upgrades ---
        # 1. Replace Attention with AttentionwithAttInv
        self.attn = AttentionwithAttInv(hidden_size, num_heads=num_heads)
        
        # 2. Add GroupDynamicScale after Attention and MLP
        self.freq_scale_1 = GroupDynamicScale(dim=hidden_size)
        self.freq_scale_2 = GroupDynamicScale(dim=hidden_size)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
    
    
class FdamBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., norm_layer=nn.LayerNorm, init_values=1e-4):
        super().__init__()
        # --- Standard components ---
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        
        # --- FDAM-specific upgrades ---
        # 1. Replace Attention with AttentionwithAttInv
        self.attn = AttentionwithAttInv(dim, num_heads=num_heads)
        
        # 2. Add GroupDynamicScale after Attention and MLP
        self.freq_scale_1 = GroupDynamicScale(dim=dim)
        self.freq_scale_2 = GroupDynamicScale(dim=dim)

    def forward(self, x, H, W):
        # --- Attention block with FDAM ---
        x_att = self.attn(self.norm1(x))
        
        # Reshape for GroupDynamicScale (N, L, C) -> (N, C, H, W)
        x_att_reshaped = nlc_to_nchw(x_att, (H, W))
        x_att_scaled = self.freq_scale_1(x_att_reshaped) + x_att_reshaped
        x_att = nchw_to_nlc(x_att_scaled)
        
        x = x + self.drop_path(self.gamma_1 * x_att)

        # --- MLP block with FDAM ---
        x_mlp = self.mlp(self.norm2(x))

        # Reshape for GroupDynamicScale
        x_mlp_reshaped = nlc_to_nchw(x_mlp, (H, W))
        x_mlp_scaled = self.freq_scale_2(x_mlp_reshaped) + x_mlp_reshaped
        x_mlp = nchw_to_nlc(x_mlp_scaled)

        x = x + self.drop_path(self.gamma_2 * x_mlp)
        
        return x
