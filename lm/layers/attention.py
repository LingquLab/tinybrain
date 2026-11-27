"""因果自注意力：分组查询（GQA）、查询和键先归一化（QK-norm）、旋转位置编码（可关），可选只看最近 window 个词。
可选：文档掩码（DocMask，只看同一篇文档）和每头一个可学的 sink 分数。

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

try:  # PyTorch 2.9 起用 return_aux 拿 log-sum-exp，return_lse 在 2.10 之后会去掉
    from torch.nn.attention.flex_attention import AuxRequest
except ImportError:
    AuxRequest = None


def flex_with_lse(q, k, v, block_mask):
    """Flex 注意力，同时返回每个查询的 log-sum-exp（自然对数，形状 (B, H, T)）。"""
    run = flex_attention if torch.compiler.is_compiling() else _flex_attention
    if AuxRequest is not None:
        out, aux = run(q, k, v, block_mask=block_mask, enable_gqa=True, return_aux=AuxRequest(lse=True))
        return out, aux.lse
    return run(q, k, v, block_mask=block_mask, enable_gqa=True, return_lse=True)

_cudnn_attention = None
_cudnn_dll_directories = []


def prepare_cudnn_attention():
    """在编译前注册 NVIDIA 的窗口 SDPA 前向和反向算子。"""
    global _cudnn_attention
    if _cudnn_attention is None:
        if os.name == "nt":
            _cudnn_dll_directories.append(os.add_dll_directory(str(Path(torch.__file__).parent / "lib")))
        try:
            from cudnn import sdpa_torch
            _cudnn_attention = sdpa_torch
        except ImportError as exc:
            raise RuntimeError("cuDNN 窗口注意力需要 nvidia-cudnn-frontend>=1.32.3；安装 .[cudnn]。") from exc


def grouped_attention(q, k, v, *, attn_mask=None, is_causal=False):
    """CPU 或逐词解码使用标准 SDPA。"""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=True)


def reference_attention(q, k, v, attn_mask=None, is_causal=False):
    """一步一步写出来的注意力，顺便返回每个查询的 log-sum-exp（sink 要用）。只在 CPU、MPS 和逐词解码时走这里。
    q: (B, H, T, 头宽)；k、v: (B, 键值头数, S, 头宽)；attn_mask 为 True 的位置能看。返回 (输出, lse)，lse 形状 (B, H, T)。"""
    group = q.size(1) // k.size(1)
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = (q.float() @ k.float().transpose(-1, -2)) / q.size(-1) ** 0.5
    if is_causal:
        T, S = q.size(-2), k.size(-2)
        attn_mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    lse = scores.logsumexp(-1)
    out = (scores - lse[..., None]).exp() @ v.float()
    return out.to(q.dtype), lse


class DocMask:
    """一个批次里每个位置属于第几篇文档，形状 (B, T)。一次前向里所有层（包括激活重计算）共用一份，
    Flex 的块掩码按窗口大小各建一次：看全文的层共用一张，局部层共用另一张。"""

    def __init__(self, doc: torch.Tensor):
        self.doc = doc
        self._flex = {}

    def same(self) -> torch.Tensor:
        """(B, 1, T, T)：查询和键是不是同一篇文档。"""
        return self.doc[:, None, :, None] == self.doc[:, None, None, :]

    def flex(self, window: int | None, length: int, device):
        if window not in self._flex:
            doc = self.doc
            def mask(b, h, q, k):
                # 尾块可含填充坐标，先约束读地址，再排除越界位置。
                same = doc[b, q.clamp_max(length - 1)] == doc[b, k.clamp_max(length - 1)]
                visible = same & (q < length) & (k < length) & (q >= k)
                return visible if window is None else visible & (q - k < window)
            self._flex[window] = create_block_mask(mask, doc.size(0), None, length, length, device=str(device))
        return self._flex[window]


class LayerCache:
    """一层注意力在逐词生成时记住的键和值，形状 (B, 键值头数, 已有的词数, 头宽)。"""

    def __init__(self):
        self.k = self.v = None


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, window: int | None = None, rope: bool = True, sink: bool = False):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim, self.window = n_heads, n_kv_heads, dim // n_heads, window
        self.rope = rope
        self.q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        # sink：softmax 的分母里多一项 exp(sink)。真实键的权重整体乘上 Z/(Z+e^sink) = sigmoid(lse − sink)，
        # 所以只要拿到每个查询的 log-sum-exp，就能在任何注意力内核后面补上，不用改内核。
        self.sink = nn.Parameter(torch.zeros(n_heads)) if sink else None
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

    def with_sink(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        """out: (B, H, T, 头宽)，lse: (B, H, T)。按 sigmoid(lse − sink) 缩小每个查询的输出。"""
        scale = torch.sigmoid(lse.float() - self.sink.float()[None, :, None])
        return out * scale.to(out.dtype)[..., None]

    def dense_mask(self, length: int, device, docs: DocMask | None) -> torch.Tensor | None:
        """CPU、MPS 路径的掩码，True 表示能看；纯因果、没有文档掩码时返回 None（交给 is_causal）。"""
        windowed = self.window is not None and self.window < length
        if not windowed and docs is None:
            return None
        pos = torch.arange(length, device=device)
        mask = pos[:, None] >= pos[None, :]
        if windowed:
            mask = mask & (pos[:, None] - pos[None, :] < self.window)
        return mask & docs.same() if docs is not None else mask

    def training_mask(self, length, device, docs):
        """在层的编译区外准备本批次要传入的 BlockMask 或布尔张量。"""
        if device.type == "cuda" and self.backend == "cudnn":
            return None
        if device.type == "cuda" and self.backend == "flex" and self.head_dim >= 16:
            if docs is not None:
                return docs.flex(self.window, length, device)
            self.prepare(length, device)
            return self._flex_mask
        return self.dense_mask(length, device, docs)

    def attend(self, q, k, v, docs: DocMask | None = None, mask=None):
        length = q.size(-2)
        if q.is_cuda and self.backend == "cudnn":
            if docs is not None or self.sink is not None:
                raise ValueError("cuDNN 注意力后端不支持文档掩码（doc_eos）和 sink（attn_sink）；换成 --attention-backend flex")
            # left_bound 排除 q-window 及更早的位置，与 Flex 的 q-k < window 相同。
            # 反向需要前向的 log-sum-exp；由官方自定义算子负责保存和复用。
            return _cudnn_attention(q, k, v, is_causal=True,
                                    window_left=self.window if self.window is not None else -1,
                                    return_lse=True)[0]
        if q.is_cuda and self.backend == 'flex' and q.size(-1) >= 16:
            block_mask = self.training_mask(length, q.device, docs) if mask is None else mask
            if self.sink is not None:
                return self.with_sink(*flex_with_lse(q, k, v, block_mask))
            run = flex_attention if torch.compiler.is_compiling() else _flex_attention
            return run(q, k, v, block_mask=block_mask, enable_gqa=True)
        if mask is None:
            mask = self.dense_mask(length, q.device, docs)
        if self.sink is not None:
            return self.with_sink(*reference_attention(q, k, v, attn_mask=mask, is_causal=mask is None))
        if mask is None:
            return grouped_attention(q, k, v, is_causal=True)
        return grouped_attention(q, k, v, attn_mask=mask)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: LayerCache | None = None,
                docs: DocMask | None = None, mask=None) -> torch.Tensor:
        B, T, _ = x.shape
        if self.fuse_qkv and self.training and x.is_cuda and cache is None:
            q_size, kv_size = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
            q, k, v = F.linear(x, torch.cat((self.q.weight, self.k.weight, self.v.weight))).split((q_size, kv_size, kv_size), dim=-1)
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            q, k = apply_rotary_qk(q, k, cos, sin)
        if cache is not None and cache.k is not None:
            if T != 1:
                raise ValueError("缓存里已经有内容时，一次只能再送进一个词")
            k, v = torch.cat([cache.k, k], dim=2), torch.cat([cache.v, v], dim=2)
            if self.window is not None:
                k, v = k[:, :, -self.window:], v[:, :, -self.window:]
            out = self.with_sink(*reference_attention(q, k, v)) if self.sink is not None else grouped_attention(q, k, v)
        else:
            out = self.attend(q, k, v, docs, mask)
        if cache is not None:
            keep = slice(None) if self.window is None else slice(-self.window, None)
            cache.k, cache.v = k[:, :, keep], v[:, :, keep]
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))
