"""因果自注意力：分组查询（GQA）、查询和键先归一化（QK-norm）、旋转位置编码，可选只看最近 window 个词。

三条计算路径：
  1. 看全文，或者序列不比窗口长：直接调 scaled_dot_product_attention（CUDA 上走 flash 内核）。
  2. 序列比窗口长：把序列切成窗口那么长的块，每块的查询只和「上一块加本块」的键算，
     计算量从 T×T 降到 T×2W。结果和逐位置加掩码完全一样。这一步带掩码，键值先按组复制到和查询一样多的头：
     多数 PyTorch 版本里，带掩码能用的融合内核不支持分组查询，不复制就会退回把整张注意力矩阵算出来的慢路径。
  3. 逐词生成：新来的一个词和缓存里的键值算，窗口层的缓存只留最近 W 个。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rotary import apply_rotary


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
        if window is not None:
            i, j = torch.arange(window)[:, None], torch.arange(2 * window)[None, :]
            # 查询在「上一块+本块」这 2W 个键里的位置是 W+i，它能看 i+1 到 W+i 这 W 个
            self.register_buffer("block_mask", (j > i) & (j <= window + i), persistent=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(self.q_norm(q), cos, sin), apply_rotary(self.k_norm(k), cos, sin)
        if cache is not None and cache.k is not None:
            if T != 1:
                raise ValueError("缓存里已经有内容时，一次只能再送进一个词")
            k, v = torch.cat([cache.k, k], dim=2), torch.cat([cache.v, v], dim=2)
            if self.window is not None:
                k, v = k[:, :, -self.window:], v[:, :, -self.window:]
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        elif self.window is None or T <= self.window:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            out = self._windowed(q, k, v)
        if cache is not None:
            keep = slice(None) if self.window is None else slice(-self.window, None)
            cache.k, cache.v = k[:, :, keep], v[:, :, keep]
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))

    def _windowed(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, _, T, D = q.shape
        W = self.window
        n_blocks = -(-T // W)
        if n_blocks * W != T:  # 末尾补齐成整块：补的位置在所有真实位置之后，因果掩码会挡住
            q, k, v = (F.pad(t, (0, 0, 0, n_blocks * W - T)) for t in (q, k, v))

        def blocks(t, lo, hi):  # 取第 lo 到 hi 块，形状 (B, 块数, 头数, W, 头宽)
            return t.view(B, t.size(1), n_blocks, W, D)[:, :, lo:hi].transpose(1, 2)

        first = F.scaled_dot_product_attention(q[:, :, :W], k[:, :, :W], v[:, :, :W], is_causal=True, enable_gqa=True)
        qb = blocks(q, 1, n_blocks).reshape(-1, self.n_heads, W, D)
        kb = torch.cat([blocks(k, 0, n_blocks - 1), blocks(k, 1, n_blocks)], dim=-2).reshape(-1, self.n_kv_heads, 2 * W, D)
        vb = torch.cat([blocks(v, 0, n_blocks - 1), blocks(v, 1, n_blocks)], dim=-2).reshape(-1, self.n_kv_heads, 2 * W, D)
        group = self.n_heads // self.n_kv_heads
        kb, vb = kb.repeat_interleave(group, dim=1), vb.repeat_interleave(group, dim=1)
        rest = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=self.block_mask)
        rest = rest.view(B, n_blocks - 1, self.n_heads, W, D).transpose(1, 2).reshape(B, self.n_heads, -1, D)
        return torch.cat([first, rest], dim=2)[:, :, :T]
