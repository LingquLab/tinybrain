"""Engram：用「前面 n-1 个词 + 当前词」直接查表取回一个记忆向量。

每个位置、每个阶数 n（默认 2 和 3）、每个哈希头（默认 8 个）各查一行，拼起来就是这个位置的记忆。
所有表放在一张大表里，各自占一段；每段的行数是互不相同的质数，这样不同哈希头的碰撞不会成对出现。
"""
import math

import torch
import torch.nn as nn

from .norm import RMSNorm


def primes_below(limit: int, count: int) -> list[int]:
    """不超过 limit 的最大的 count 个质数，从大到小。"""
    found, n = [], limit
    while len(found) < count:
        if n < 2:
            raise ValueError(f"{limit} 以内没有 {count} 个质数")
        if all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
        n -= 1
    return found


class EngramMemory(nn.Module):
    """查表部分。forward 返回 (B, T, 阶数, 头数×头宽)，已经归一化。"""

    def __init__(self, vocab_size: int, orders: tuple, heads: int, head_dim: int, rows: int, seed: int = 0, doc_eos: int = -1):
        super().__init__()
        self.orders, self.heads, self.pad_id = tuple(orders), heads, vocab_size
        # doc_eos ≥ 0：结束符算下一篇文档的第一个词，n-gram 里属于上一篇文档的词换成「空」（和序列开头一样），不跨文档查表
        self.doc_eos = doc_eos
        sizes = torch.tensor(primes_below(rows, len(orders) * heads)).view(len(orders), heads)
        gen = torch.Generator().manual_seed(seed)
        # 乘数是奇数且小于 2^31，词表小于 2^31，乘积不会超出 64 位整数
        mult = torch.randint(1 << 20, 1 << 30, (len(orders), heads, max(orders)), generator=gen) * 2 + 1
        self.register_buffer("sizes", sizes)
        self.register_buffer("offsets", (sizes.flatten().cumsum(0) - sizes.flatten()).view_as(sizes))
        self.register_buffer("mult", mult)  # 存进检查点：哈希变了表就废了
        self.table = nn.Embedding(int(sizes.sum()), head_dim)
        self.norm = RMSNorm(heads * head_dim)

    @property
    def history_len(self) -> int:
        return max(self.orders) - 1

    def slots(self, idx: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        """每个位置要查的行号，形状 (B, T, 阶数, 头数)。history 是这段之前的最后几个词（逐词生成时用）。"""
        B, T = idx.shape
        m = self.history_len
        if history is None:
            history = idx.new_full((B, m), self.pad_id)
        ctx = torch.cat([history[:, -m:], idx], dim=1)  # (B, m+T)
        doc = (ctx == self.doc_eos).cumsum(1) if self.doc_eos >= 0 else None  # 每个位置属于第几篇文档
        out = []
        for o, n in enumerate(self.orders):
            h = torch.zeros(B, T, self.heads, dtype=torch.long, device=idx.device)
            for j in range(n):  # j=0 是当前词，j=1 是前一个词……
                word = ctx[:, m - j: m - j + T]
                if doc is not None and j:
                    word = torch.where(doc[:, m - j: m - j + T] == doc[:, m: m + T], word, self.pad_id)
                h = h ^ (word[:, :, None] * self.mult[o, :, j])
            out.append(h % self.sizes[o] + self.offsets[o])
        return torch.stack(out, dim=2)

    def forward(self, idx: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.shape
        e = self.table(self.slots(idx, history))  # (B, T, 阶数, 头数, 头宽)
        return self.norm(e.view(B, T, len(self.orders), -1))


class EngramGate(nn.Module):
    """把记忆注入多条残差流。值投影所有流共用，键投影每条流一份：
    每条流拿自己当前的状态和键比一比，决定这条记忆收多少（0 到 1 之间），再加到自己身上。"""

    def __init__(self, memory_dim: int, dim: int, n_streams: int):
        super().__init__()
        self.n_streams, self.dim = n_streams, dim
        self.value = nn.Linear(memory_dim, dim, bias=False)
        self.keys = nn.Linear(memory_dim, n_streams * dim, bias=False)
        self.state_norm, self.key_norm = RMSNorm(dim), RMSNorm(dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n, d)；memory: (B, T, memory_dim)"""
        B, T = memory.shape[:2]
        keys = self.keys(memory).view(B, T, self.n_streams, self.dim)
        gate = torch.sigmoid((self.state_norm(x) * self.key_norm(keys)).sum(-1, keepdim=True) / math.sqrt(self.dim))
        return x + gate * self.value(memory)[:, :, None, :]
