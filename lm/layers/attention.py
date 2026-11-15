"""因果自注意力：分组查询（GQA）、查询和键先归一化（QK-norm）、旋转位置编码，可选只看最近 window 个词。

三条计算路径：
  1. CUDA 上训练和整段前向：默认 FlexAttention。窗口和因果写成一条掩码规则，编成块稀疏的掩码，只算窗口里的那条带子；
     原生支持分组查询，不复制键值。头宽小于 16 时退回第 2 条。
     也可以换成 NVIDIA cuDNN 原生的因果/滑动窗口内核（--attention-backend cudnn，要装 .[cudnn]）。
  2. CPU、MPS：scaled_dot_product_attention。窗口层带一张 T×T 的掩码（只在小模型和短序列上用，简单优先）。
  3. 逐词生成：新来的一个词和缓存里的键值算，窗口层的缓存只留最近 W 个。
CUDA 训练时 Q、K、V 三个投影合成一次矩阵乘法，参数和存档里仍是三组权重。
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .norm import RMSNorm
from .rotary import apply_rotary_qk


_flex_attention = torch.compile(flex_attention, fullgraph=True, dynamic=False,
                                options={"triton.cudagraphs": False})

_cudnn_attention = None
_cudnn_dll_directories = []


def prepare_cudnn_attention():
    """在编译前注册 NVIDIA 的窗口 SDPA 前向和反向算子。"""
    global _cudnn_attention
    if _cudnn_attention is None:
        if os.name == "nt":
            _cudnn_dll_directories.append(os.add_dll_directory(str(Path(torch.__file__).parent / "lib")))
        try:
            import cudnn
            _cudnn_attention = cudnn.sdpa_torch
        except ImportError as exc:
            raise RuntimeError("cuDNN 窗口注意力需要 nvidia-cudnn-frontend>=1.32.3；安装 .[cudnn]。") from exc


def grouped_attention(q, k, v, *, attn_mask=None, is_causal=False):
    """CPU 或逐词解码使用标准 SDPA。"""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=True)


class LayerCache:
    """一层注意力在逐词生成时记住的键和值，形状 (B, 键值头数, 已有的词数, 头宽)。"""

    def __init__(self):
        self.k = self.v = None


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, window: int | None = None):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim, self.window = n_heads, n_kv_heads, dim // n_heads, window
        self.q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self._flex_mask = None
        self._flex_length = 0
        self.fuse_qkv = True
        self.backend = "flex"

    def prepare(self, length, device):
        """在编译前构建稀疏块掩码；同一形状的训练步复用它。"""
        if self.backend == "cudnn":
            return
        if self._flex_length == length and self._flex_mask is not None:
            if self._flex_mask.kv_indices.device == torch.device(device):
                return
        window = self.window
        if window is None:
            def mask(b, h, q, k):
                return q >= k
        else:
            def mask(b, h, q, k):
                return (q >= k) & (q - k < window)
        self._flex_mask = create_block_mask(mask, None, None, length, length, device=str(device))
        self._flex_length = length

    def attend(self, q, k, v):
        length = q.size(-2)
        if q.is_cuda and self.backend == "cudnn":
            # left_bound 排除 q-window 及更早的位置，与 Flex 的 q-k < window 相同。
            # 反向需要前向的 log-sum-exp；由官方自定义算子负责保存和复用。
            return _cudnn_attention(q, k, v, is_causal=True,
                                    window_left=self.window if self.window is not None else -1,
                                    return_lse=True)[0]
        if q.is_cuda and q.size(-1) >= 16:
            if self._flex_length != length or self._flex_mask is None:
                self.prepare(length, q.device)
            return _flex_attention(q, k, v, block_mask=self._flex_mask, enable_gqa=True)
        if self.window is None:
            return grouped_attention(q, k, v, is_causal=True)
        pos = torch.arange(length, device=q.device)
        mask = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] < self.window)
        return grouped_attention(q, k, v, attn_mask=mask)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        if self.fuse_qkv and self.training and x.is_cuda and cache is None:
            q_size, kv_size = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
            q, k, v = F.linear(x, torch.cat((self.q.weight, self.k.weight, self.v.weight))).split((q_size, kv_size, kv_size), dim=-1)
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_qk(self.q_norm(q), self.k_norm(k), cos, sin)
        if cache is not None and cache.k is not None:
            if T != 1:
                raise ValueError("缓存里已经有内容时，一次只能再送进一个词")
            k, v = torch.cat([cache.k, k], dim=2), torch.cat([cache.v, v], dim=2)
            if self.window is not None:
                k, v = k[:, :, -self.window:], v[:, :, -self.window:]
            out = grouped_attention(q, k, v)
        else:
            out = self.attend(q, k, v)
        if cache is not None:
            keep = slice(None) if self.window is None else slice(-self.window, None)
            cache.k, cache.v = k[:, :, keep], v[:, :, keep]
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))
